//! The tree-walking evaluator.
//!
//! Throws are propagated through the `Err` channel of [`Completion`] as the
//! thrown [`Value`]; non-local statement control flow (`return` / `break` /
//! `continue`) is carried by [`Flow`]. This increment covers primitives,
//! control flow, and functions/closures; objects, arrays, and member access
//! arrive next.

use super::env::{Env, Scope};
use super::value::{Callable, Closure, Value, loose_equals, strict_equals};
use crate::ast::{
    ArrowBody, AssignOp, BinaryOp, BindingTarget, Expr, LogicalOp, Param, Program, Stmt, UnaryOp,
    UpdateOp, VarDeclKind,
};
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

/// The result of an evaluation: `Ok(T)` or a thrown JS [`Value`].
pub type Completion<'a, T> = Result<T, Value<'a>>;

/// Statement control flow.
enum Flow<'a> {
    /// Normal completion, carrying the last expression value (for the
    /// script/REPL completion value).
    Normal(Value<'a>),
    /// A `return` with its value.
    Return(Value<'a>),
    /// A `break`, with an optional target label.
    Break(Option<Box<str>>),
    /// A `continue`, with an optional target label.
    Continue(Option<Box<str>>),
}

/// What a loop body iteration decided.
enum LoopCtl<'a> {
    /// Proceed to the next iteration.
    Next,
    /// Stop the loop normally.
    Stop,
    /// Propagate an abrupt completion outward.
    Propagate(Flow<'a>),
}

/// The tree-walking interpreter, holding the global scope.
pub struct Interp<'a> {
    global: Env<'a>,
}

impl<'a> Default for Interp<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Interp<'a> {
    /// Creates an interpreter with a global scope seeded with the standard
    /// value-property globals (`undefined`, `NaN`, `Infinity`).
    #[must_use]
    pub fn new() -> Self {
        let global = Scope::new_global();
        global.declare("undefined", Value::Undefined, false);
        global.declare("NaN", Value::Number(f64::NAN), false);
        global.declare("Infinity", Value::Number(f64::INFINITY), false);
        Self { global }
    }

    /// The global scope, for injecting host bindings (e.g. `console`).
    #[must_use]
    pub fn global(&self) -> &Env<'a> {
        &self.global
    }

    /// Defines a global binding.
    pub fn define_global(&self, name: &str, value: Value<'a>) {
        self.global.declare(name, value, true);
    }

    /// Runs a program, returning its completion value (the value of the last
    /// expression statement) or an uncaught thrown value.
    pub fn run(&mut self, program: &'a Program) -> Completion<'a, Value<'a>> {
        let global = Rc::clone(&self.global);
        self.hoist(&program.body, &global);
        match self.eval_stmts(&program.body, &global)? {
            Flow::Normal(v) | Flow::Return(v) => Ok(v),
            _ => Ok(Value::Undefined),
        }
    }

    // --- statements -----------------------------------------------------

    /// Hoists function declarations in `stmts` into `scope`.
    fn hoist(&mut self, stmts: &'a [Stmt], scope: &Env<'a>) {
        for s in stmts {
            if let Stmt::Function(f) = s
                && let Some(id) = &f.id
            {
                let closure = Value::Function(Rc::new(Closure {
                    def: Callable::Function(f),
                    env: Rc::clone(scope),
                }));
                scope.declare(&id.name, closure, true);
            }
        }
    }

    /// Evaluates a statement list in `scope`, threading control flow and the
    /// completion value.
    fn eval_stmts(&mut self, stmts: &'a [Stmt], scope: &Env<'a>) -> Completion<'a, Flow<'a>> {
        let mut last = Value::Undefined;
        for s in stmts {
            match self.eval_stmt(s, scope)? {
                Flow::Normal(v) => last = v,
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal(last))
    }

    fn eval_stmt(&mut self, stmt: &'a Stmt, env: &Env<'a>) -> Completion<'a, Flow<'a>> {
        self.eval_stmt_labeled(stmt, env, None)
    }

    fn eval_stmt_labeled(
        &mut self,
        stmt: &'a Stmt,
        env: &Env<'a>,
        label: Option<&str>,
    ) -> Completion<'a, Flow<'a>> {
        match stmt {
            Stmt::Empty { .. } | Stmt::Function(_) | Stmt::Debugger { .. } => {
                Ok(Flow::Normal(Value::Undefined))
            }
            Stmt::Expr { expression, .. } => Ok(Flow::Normal(self.eval_expr(expression, env)?)),
            Stmt::Block { body, .. } => {
                let scope = Scope::child(env);
                self.hoist(body, &scope);
                self.eval_stmts(body, &scope)
            }
            Stmt::Var(decl) => {
                for d in &decl.declarations {
                    let value = match &d.init {
                        Some(e) => self.eval_expr(e, env)?,
                        None => Value::Undefined,
                    };
                    self.bind_target(&d.target, value, env, decl.kind)?;
                }
                Ok(Flow::Normal(Value::Undefined))
            }
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval_expr(test, env)?.to_boolean() {
                    self.eval_stmt(consequent, env)
                } else if let Some(alt) = alternate {
                    self.eval_stmt(alt, env)
                } else {
                    Ok(Flow::Normal(Value::Undefined))
                }
            }
            Stmt::While { test, body, .. } => self.eval_while(test, body, env, label),
            Stmt::DoWhile { body, test, .. } => self.eval_do_while(body, test, env, label),
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => self.eval_for(
                init.as_ref(),
                test.as_deref(),
                update.as_deref(),
                body,
                env,
                label,
            ),
            Stmt::Switch {
                discriminant,
                cases,
                ..
            } => self.eval_switch(discriminant, cases, env),
            Stmt::Return { argument, .. } => {
                let v = match argument {
                    Some(e) => self.eval_expr(e, env)?,
                    None => Value::Undefined,
                };
                Ok(Flow::Return(v))
            }
            Stmt::Break { label, .. } => Ok(Flow::Break(label.as_ref().map(|l| l.name.clone()))),
            Stmt::Continue { label, .. } => {
                Ok(Flow::Continue(label.as_ref().map(|l| l.name.clone())))
            }
            Stmt::Throw { argument, .. } => Err(self.eval_expr(argument, env)?),
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => self.eval_try(block, handler.as_ref(), finalizer.as_deref(), env),
            Stmt::Labeled { label: l, body, .. } => {
                let r = self.eval_stmt_labeled(body, env, Some(&l.name))?;
                match r {
                    Flow::Break(Some(bl)) if *bl == *l.name => Ok(Flow::Normal(Value::Undefined)),
                    other => Ok(other),
                }
            }
            Stmt::Class(_) => Err(Value::str("classes are not yet supported at runtime")),
            Stmt::With { .. } => Err(Value::str("`with` is not supported")),
            Stmt::ForIn { .. } | Stmt::ForOf { .. } => {
                Err(Value::str("for-in/of are not yet supported at runtime"))
            }
            Stmt::Import(_) | Stmt::Export(_) => {
                Err(Value::str("modules are not yet supported at runtime"))
            }
        }
    }

    fn eval_while(
        &mut self,
        test: &'a Expr,
        body: &'a Stmt,
        env: &Env<'a>,
        label: Option<&str>,
    ) -> Completion<'a, Flow<'a>> {
        while self.eval_expr(test, env)?.to_boolean() {
            match self.loop_step(body, env, label)? {
                LoopCtl::Next => {}
                LoopCtl::Stop => break,
                LoopCtl::Propagate(f) => return Ok(f),
            }
        }
        Ok(Flow::Normal(Value::Undefined))
    }

    fn eval_do_while(
        &mut self,
        body: &'a Stmt,
        test: &'a Expr,
        env: &Env<'a>,
        label: Option<&str>,
    ) -> Completion<'a, Flow<'a>> {
        loop {
            match self.loop_step(body, env, label)? {
                LoopCtl::Next => {}
                LoopCtl::Stop => break,
                LoopCtl::Propagate(f) => return Ok(f),
            }
            if !self.eval_expr(test, env)?.to_boolean() {
                break;
            }
        }
        Ok(Flow::Normal(Value::Undefined))
    }

    fn eval_for(
        &mut self,
        init: Option<&'a crate::ast::ForInit>,
        test: Option<&'a Expr>,
        update: Option<&'a Expr>,
        body: &'a Stmt,
        env: &Env<'a>,
        label: Option<&str>,
    ) -> Completion<'a, Flow<'a>> {
        // The loop header gets its own scope, so `for (let i …)` binds per loop.
        let scope = Scope::child(env);
        if let Some(init) = init {
            match init {
                crate::ast::ForInit::Var(decl) => {
                    for d in &decl.declarations {
                        let value = match &d.init {
                            Some(e) => self.eval_expr(e, &scope)?,
                            None => Value::Undefined,
                        };
                        self.bind_target(&d.target, value, &scope, decl.kind)?;
                    }
                }
                crate::ast::ForInit::Expr(e) => {
                    self.eval_expr(e, &scope)?;
                }
            }
        }
        loop {
            if let Some(t) = test
                && !self.eval_expr(t, &scope)?.to_boolean()
            {
                break;
            }
            match self.loop_step(body, &scope, label)? {
                LoopCtl::Next => {}
                LoopCtl::Stop => break,
                LoopCtl::Propagate(f) => return Ok(f),
            }
            if let Some(u) = update {
                self.eval_expr(u, &scope)?;
            }
        }
        Ok(Flow::Normal(Value::Undefined))
    }

    /// Evaluates one loop-body iteration and classifies the resulting flow,
    /// consuming `break`/`continue` that target this loop (unlabeled, or this
    /// loop's `label`).
    fn loop_step(
        &mut self,
        body: &'a Stmt,
        env: &Env<'a>,
        label: Option<&str>,
    ) -> Completion<'a, LoopCtl<'a>> {
        Ok(match self.eval_stmt(body, env)? {
            Flow::Normal(_) | Flow::Continue(None) => LoopCtl::Next,
            Flow::Continue(Some(l)) if Some(&*l) == label => LoopCtl::Next,
            Flow::Break(None) => LoopCtl::Stop,
            Flow::Break(Some(l)) if Some(&*l) == label => LoopCtl::Stop,
            other => LoopCtl::Propagate(other),
        })
    }

    fn eval_switch(
        &mut self,
        discriminant: &'a Expr,
        cases: &'a [crate::ast::SwitchCase],
        env: &Env<'a>,
    ) -> Completion<'a, Flow<'a>> {
        let d = self.eval_expr(discriminant, env)?;
        let scope = Scope::child(env);
        // Find the first matching `case` (strict equality), else `default`.
        let mut start = None;
        for (i, c) in cases.iter().enumerate() {
            if let Some(test) = &c.test
                && strict_equals(&self.eval_expr(test, &scope)?, &d)
            {
                start = Some(i);
                break;
            }
        }
        if start.is_none() {
            start = cases.iter().position(|c| c.test.is_none());
        }
        let Some(start) = start else {
            return Ok(Flow::Normal(Value::Undefined));
        };
        // Execute from the matched clause, falling through subsequent clauses.
        for c in &cases[start..] {
            match self.eval_stmts(&c.body, &scope)? {
                Flow::Normal(_) => {}
                Flow::Break(None) => return Ok(Flow::Normal(Value::Undefined)),
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal(Value::Undefined))
    }

    fn eval_try(
        &mut self,
        block: &'a [Stmt],
        handler: Option<&'a crate::ast::CatchClause>,
        finalizer: Option<&'a [Stmt]>,
        env: &Env<'a>,
    ) -> Completion<'a, Flow<'a>> {
        let try_scope = Scope::child(env);
        self.hoist(block, &try_scope);
        let mut outcome = self.eval_stmts(block, &try_scope);

        if let Err(thrown) = outcome {
            if let Some(h) = handler {
                let cscope = Scope::child(env);
                if let Some(param) = &h.param {
                    self.bind_target(param, thrown, &cscope, VarDeclKind::Let)?;
                }
                self.hoist(&h.body, &cscope);
                outcome = self.eval_stmts(&h.body, &cscope);
            } else {
                outcome = Err(thrown);
            }
        }

        if let Some(fin) = finalizer {
            let fscope = Scope::child(env);
            self.hoist(fin, &fscope);
            // An abrupt completion in `finally` overrides the try/catch result.
            match self.eval_stmts(fin, &fscope)? {
                Flow::Normal(_) => {}
                abrupt => return Ok(abrupt),
            }
        }
        outcome
    }

    // --- binding --------------------------------------------------------

    /// Binds a declaration/catch target. Destructuring patterns are deferred to
    /// the object/array increment.
    fn bind_target(
        &mut self,
        target: &'a BindingTarget,
        value: Value<'a>,
        scope: &Env<'a>,
        kind: VarDeclKind,
    ) -> Completion<'a, ()> {
        match target {
            BindingTarget::Ident(id) => {
                scope.declare(&id.name, value, kind != VarDeclKind::Const);
                Ok(())
            }
            BindingTarget::Array(_) | BindingTarget::Object(_) => {
                Err(Value::str("destructuring is not yet supported at runtime"))
            }
        }
    }

    // --- expressions ----------------------------------------------------

    fn eval_expr(&mut self, expr: &'a Expr, env: &Env<'a>) -> Completion<'a, Value<'a>> {
        match expr {
            Expr::Null(_) => Ok(Value::Null),
            Expr::Bool { value, .. } => Ok(Value::Bool(*value)),
            Expr::Number { value, .. } => Ok(Value::Number(*value)),
            Expr::BigInt { digits, .. } => {
                // Placeholder until a real BigInt type exists: coerce the digit
                // string (radix prefix included) to f64.
                Ok(Value::Number(Value::str(digits.clone()).to_number()))
            }
            Expr::Str { value, .. } => Ok(Value::str(value.clone())),
            Expr::Template(t) => {
                let mut out = String::new();
                for (i, q) in t.quasis.iter().enumerate() {
                    out.push_str(q.cooked.as_deref().unwrap_or(""));
                    if let Some(e) = t.expressions.get(i) {
                        out.push_str(&self.eval_expr(e, env)?.to_js_string());
                    }
                }
                Ok(Value::str(out))
            }
            Expr::Ident(id) => match env.get(&id.name) {
                Some(v) => Ok(v),
                None => Err(Value::str(alloc::format!("{} is not defined", id.name))),
            },
            Expr::This(_) => Ok(Value::Undefined),
            Expr::Unary { op, argument, .. } => self.eval_unary(*op, argument, env),
            Expr::Update {
                op,
                prefix,
                argument,
                ..
            } => self.eval_update(*op, *prefix, argument, env),
            Expr::Binary {
                op, left, right, ..
            } => {
                let l = self.eval_expr(left, env)?;
                let r = self.eval_expr(right, env)?;
                self.eval_binary(*op, l, r)
            }
            Expr::Logical {
                op, left, right, ..
            } => {
                let l = self.eval_expr(left, env)?;
                match op {
                    LogicalOp::And if !l.to_boolean() => Ok(l),
                    LogicalOp::Or if l.to_boolean() => Ok(l),
                    LogicalOp::Nullish if !matches!(l, Value::Undefined | Value::Null) => Ok(l),
                    _ => self.eval_expr(right, env),
                }
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval_expr(test, env)?.to_boolean() {
                    self.eval_expr(consequent, env)
                } else {
                    self.eval_expr(alternate, env)
                }
            }
            Expr::Assign {
                op, target, value, ..
            } => self.eval_assign(*op, target, value, env),
            Expr::Sequence { expressions, .. } => {
                let mut last = Value::Undefined;
                for e in expressions {
                    last = self.eval_expr(e, env)?;
                }
                Ok(last)
            }
            Expr::Call {
                callee, arguments, ..
            } => {
                let callee_val = self.eval_expr(callee, env)?;
                let mut args = Vec::with_capacity(arguments.len());
                for a in arguments {
                    match a {
                        crate::ast::Argument::Item(e) => args.push(self.eval_expr(e, env)?),
                        crate::ast::Argument::Spread(_) => {
                            return Err(Value::str("spread arguments are not yet supported"));
                        }
                    }
                }
                self.call(callee_val, args)
            }
            Expr::Function(f) => Ok(Value::Function(Rc::new(Closure {
                def: Callable::Function(f),
                env: Rc::clone(env),
            }))),
            Expr::Arrow(a) => Ok(Value::Function(Rc::new(Closure {
                def: Callable::Arrow(a),
                env: Rc::clone(env),
            }))),
            Expr::Member { .. } | Expr::New { .. } => Err(Value::str(
                "objects/members are not yet supported at runtime",
            )),
            Expr::Array { .. } | Expr::Object { .. } => Err(Value::str(
                "array/object literals are not yet supported at runtime",
            )),
            Expr::Class(_) => Err(Value::str(
                "class expressions are not yet supported at runtime",
            )),
            Expr::Regex { .. } => Err(Value::str("RegExp is not yet supported at runtime")),
            Expr::TaggedTemplate { .. } => Err(Value::str(
                "tagged templates are not yet supported at runtime",
            )),
            Expr::Super(_) => Err(Value::str("`super` is not yet supported at runtime")),
            Expr::Yield { .. } | Expr::Await { .. } => Err(Value::str(
                "generators/async are not yet supported at runtime",
            )),
        }
    }

    fn eval_unary(
        &mut self,
        op: UnaryOp,
        argument: &'a Expr,
        env: &Env<'a>,
    ) -> Completion<'a, Value<'a>> {
        // `typeof` of an unbound identifier yields "undefined" without throwing.
        if op == UnaryOp::Typeof {
            if let Expr::Ident(id) = argument
                && !env.has(&id.name)
            {
                return Ok(Value::str("undefined"));
            }
            return Ok(Value::str(self.eval_expr(argument, env)?.type_of()));
        }
        let v = self.eval_expr(argument, env)?;
        Ok(match op {
            UnaryOp::Plus => Value::Number(v.to_number()),
            UnaryOp::Minus => Value::Number(-v.to_number()),
            UnaryOp::Not => Value::Bool(!v.to_boolean()),
            UnaryOp::BitNot => Value::Number(f64::from(!v.to_int32())),
            UnaryOp::Void => Value::Undefined,
            UnaryOp::Typeof => unreachable!("handled above"),
            UnaryOp::Delete => Value::Bool(true), // no references to delete yet
        })
    }

    fn eval_update(
        &mut self,
        op: UpdateOp,
        prefix: bool,
        argument: &'a Expr,
        env: &Env<'a>,
    ) -> Completion<'a, Value<'a>> {
        let Expr::Ident(id) = argument else {
            return Err(Value::str(
                "invalid update target (members not yet supported)",
            ));
        };
        let old = env
            .get(&id.name)
            .ok_or_else(|| Value::str(alloc::format!("{} is not defined", id.name)))?
            .to_number();
        let new = match op {
            UpdateOp::Inc => old + 1.0,
            UpdateOp::Dec => old - 1.0,
        };
        self.assign_ident(&id.name, Value::Number(new), env)?;
        Ok(Value::Number(if prefix { new } else { old }))
    }

    fn eval_assign(
        &mut self,
        op: AssignOp,
        target: &'a Expr,
        value: &'a Expr,
        env: &Env<'a>,
    ) -> Completion<'a, Value<'a>> {
        let Expr::Ident(id) = target else {
            return Err(Value::str("assignment to members is not yet supported"));
        };
        let new = if op == AssignOp::Assign {
            self.eval_expr(value, env)?
        } else {
            let cur = env
                .get(&id.name)
                .ok_or_else(|| Value::str(alloc::format!("{} is not defined", id.name)))?;
            // Logical assignment short-circuits.
            match op {
                AssignOp::AndAssign => {
                    if !cur.to_boolean() {
                        return Ok(cur);
                    }
                    self.eval_expr(value, env)?
                }
                AssignOp::OrAssign => {
                    if cur.to_boolean() {
                        return Ok(cur);
                    }
                    self.eval_expr(value, env)?
                }
                AssignOp::NullishAssign => {
                    if !matches!(cur, Value::Undefined | Value::Null) {
                        return Ok(cur);
                    }
                    self.eval_expr(value, env)?
                }
                _ => {
                    let rhs = self.eval_expr(value, env)?;
                    self.eval_binary(compound_binop(op), cur, rhs)?
                }
            }
        };
        self.assign_ident(&id.name, new.clone(), env)?;
        Ok(new)
    }

    fn assign_ident(&self, name: &str, value: Value<'a>, env: &Env<'a>) -> Completion<'a, ()> {
        use super::env::AssignOutcome;
        match env.assign(name, value.clone()) {
            AssignOutcome::Assigned => Ok(()),
            // Assignment to an undeclared name creates a global (sloppy mode).
            AssignOutcome::Unbound => {
                self.global.declare(name, value, true);
                Ok(())
            }
            AssignOutcome::Immutable => Err(Value::str(alloc::format!(
                "assignment to constant variable {name}"
            ))),
        }
    }

    fn eval_binary(
        &mut self,
        op: BinaryOp,
        l: Value<'a>,
        r: Value<'a>,
    ) -> Completion<'a, Value<'a>> {
        use BinaryOp::*;
        Ok(match op {
            Add => {
                if matches!(l, Value::Str(_)) || matches!(r, Value::Str(_)) {
                    let mut s = l.to_js_string();
                    s.push_str(&r.to_js_string());
                    Value::str(s)
                } else {
                    Value::Number(l.to_number() + r.to_number())
                }
            }
            Sub => Value::Number(l.to_number() - r.to_number()),
            Mul => Value::Number(l.to_number() * r.to_number()),
            Div => Value::Number(l.to_number() / r.to_number()),
            Mod => Value::Number(l.to_number() % r.to_number()),
            Exp => Value::Number(l.to_number().powf(r.to_number())),
            Lt | Gt | LtEq | GtEq => Value::Bool(self.relational(op, &l, &r)),
            EqEq => Value::Bool(loose_equals(&l, &r)),
            NotEq => Value::Bool(!loose_equals(&l, &r)),
            EqEqEq => Value::Bool(strict_equals(&l, &r)),
            NotEqEq => Value::Bool(!strict_equals(&l, &r)),
            BitAnd => Value::Number(f64::from(l.to_int32() & r.to_int32())),
            BitOr => Value::Number(f64::from(l.to_int32() | r.to_int32())),
            BitXor => Value::Number(f64::from(l.to_int32() ^ r.to_int32())),
            Shl => Value::Number(f64::from(l.to_int32().wrapping_shl(r.to_uint32() & 31))),
            Shr => Value::Number(f64::from(l.to_int32().wrapping_shr(r.to_uint32() & 31))),
            Ushr => Value::Number(f64::from(l.to_uint32() >> (r.to_uint32() & 31))),
            In | Instanceof => {
                return Err(Value::str(
                    "`in`/`instanceof` need objects (not yet supported)",
                ));
            }
        })
    }

    fn relational(&self, op: BinaryOp, l: &Value<'a>, r: &Value<'a>) -> bool {
        // String/string compares lexicographically; otherwise numerically.
        if let (Value::Str(a), Value::Str(b)) = (l, r) {
            return match op {
                BinaryOp::Lt => a < b,
                BinaryOp::Gt => a > b,
                BinaryOp::LtEq => a <= b,
                BinaryOp::GtEq => a >= b,
                _ => unreachable!(),
            };
        }
        let (a, b) = (l.to_number(), r.to_number());
        if a.is_nan() || b.is_nan() {
            return false;
        }
        match op {
            BinaryOp::Lt => a < b,
            BinaryOp::Gt => a > b,
            BinaryOp::LtEq => a <= b,
            BinaryOp::GtEq => a >= b,
            _ => unreachable!(),
        }
    }

    // --- calls ----------------------------------------------------------

    fn call(&mut self, callee: Value<'a>, args: Vec<Value<'a>>) -> Completion<'a, Value<'a>> {
        match callee {
            Value::Native(n) => (n.call)(&args),
            Value::Function(closure) => {
                let scope = Scope::child(&closure.env);
                match closure.def {
                    Callable::Function(f) => {
                        self.bind_params(&f.params, &args, &scope)?;
                        self.hoist(&f.body, &scope);
                        match self.eval_stmts(&f.body, &scope)? {
                            Flow::Return(v) => Ok(v),
                            _ => Ok(Value::Undefined),
                        }
                    }
                    Callable::Arrow(a) => {
                        self.bind_params(&a.params, &args, &scope)?;
                        match &a.body {
                            ArrowBody::Expr(e) => self.eval_expr(e, &scope),
                            ArrowBody::Block(body) => {
                                self.hoist(body, &scope);
                                match self.eval_stmts(body, &scope)? {
                                    Flow::Return(v) => Ok(v),
                                    _ => Ok(Value::Undefined),
                                }
                            }
                        }
                    }
                }
            }
            _ => Err(Value::str(alloc::format!(
                "{} is not a function",
                callee.to_js_string()
            ))),
        }
    }

    fn bind_params(
        &mut self,
        params: &'a [Param],
        args: &[Value<'a>],
        scope: &Env<'a>,
    ) -> Completion<'a, ()> {
        for (i, p) in params.iter().enumerate() {
            if p.rest {
                return Err(Value::str(
                    "rest parameters need arrays (not yet supported)",
                ));
            }
            let BindingTarget::Ident(id) = &p.target else {
                return Err(Value::str("destructuring parameters are not yet supported"));
            };
            let mut v = args.get(i).cloned().unwrap_or(Value::Undefined);
            if matches!(v, Value::Undefined)
                && let Some(d) = &p.default
            {
                v = self.eval_expr(d, scope)?;
            }
            scope.declare(&id.name, v, true);
        }
        Ok(())
    }
}

/// Maps a compound assignment operator to its underlying binary operator.
fn compound_binop(op: AssignOp) -> BinaryOp {
    match op {
        AssignOp::AddAssign => BinaryOp::Add,
        AssignOp::SubAssign => BinaryOp::Sub,
        AssignOp::MulAssign => BinaryOp::Mul,
        AssignOp::DivAssign => BinaryOp::Div,
        AssignOp::ModAssign => BinaryOp::Mod,
        AssignOp::ExpAssign => BinaryOp::Exp,
        AssignOp::ShlAssign => BinaryOp::Shl,
        AssignOp::ShrAssign => BinaryOp::Shr,
        AssignOp::UshrAssign => BinaryOp::Ushr,
        AssignOp::BitAndAssign => BinaryOp::BitAnd,
        AssignOp::BitOrAssign => BinaryOp::BitOr,
        AssignOp::BitXorAssign => BinaryOp::BitXor,
        AssignOp::Assign | AssignOp::AndAssign | AssignOp::OrAssign | AssignOp::NullishAssign => {
            unreachable!("handled before lowering to a binary op")
        }
    }
}
