//! Executing real **statements and functions** over the [`Realm`]/[`NanBox`]
//! model (`ROADMAP.md` §3 → Phase D migration).
//!
//! [`Realm`]: crate::realm::Realm
//! [`NanBox`]: crate::nanbox::NanBox
//!
//! A small tree-walking interpreter whose values are NaN-boxed and whose
//! objects/strings/arrays/functions live in the realm's GC heap — the
//! imperative *and procedural* core of the language on the performance
//! representation. It has:
//! - lexical variable scope ([`Scope`](crate::env::Scope) chains), assignment
//!   (incl. compound and member targets), block scoping, and control flow
//!   (`if`/`while`/`for`, `return`/`break`/`continue`);
//! - **functions and closures**: declarations (hoisted), expressions, and arrows
//!   become heap closures capturing their defining scope, so a returned inner
//!   function still sees its enclosing variables — and calls bind arguments in a
//!   fresh child scope.
//!
//! Exceptions and the stdlib are the remaining structural pieces. Pure, safe
//! `alloc`-only Rust.

use crate::ast::{
    Argument, ArrayElement, Arrow, ArrowBody, AssignOp, BinaryOp, BindingTarget, Expr, ForInit,
    Function, Ident, LogicalOp, ObjectMember, Param, Program, PropertyKey, Stmt, UnaryOp, VarDecl,
};
use crate::env::Scope;
use crate::nanbox::{NanBox, Unpacked};
use crate::realm::Realm;
use alloc::string::String;
use alloc::vec::Vec;

/// Why execution stopped.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExecError {
    /// A construct outside the supported subset (generators, classes, …).
    Unsupported(&'static str),
    /// A reference to an undeclared variable.
    NotDefined(String),
    /// A call of a non-function value.
    NotCallable,
    /// A thrown JS value, propagating until a `catch` handles it.
    Throw(NanBox),
}

/// The control-flow outcome of a statement.
enum Flow {
    /// Fell through normally, carrying the last expression value (for `run`).
    Normal(NanBox),
    /// A `return` (value).
    Return(NanBox),
    /// A `break`.
    Break,
    /// A `continue`.
    Continue,
}

/// The body of a registered function: a block, or a concise arrow expression.
#[derive(Clone, Copy)]
enum Body<'a> {
    Block(&'a [Stmt]),
    Expr(&'a Expr),
}

/// A registered function definition (its AST, held by the interpreter; the heap
/// closure stores only an index into the table plus the captured scope).
#[derive(Clone, Copy)]
struct FnDef<'a> {
    params: &'a [Param],
    body: Body<'a>,
}

/// A tree-walking interpreter over the performance object model.
pub struct Interp<'a> {
    realm: Realm,
    /// The current lexical scope (innermost).
    current: Scope,
    /// Function-AST table; a closure cell holds an index into this.
    functions: Vec<FnDef<'a>>,
}

impl Default for Interp<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Interp<'a> {
    /// A fresh interpreter with a single (global) scope.
    #[must_use]
    pub fn new() -> Self {
        Self {
            realm: Realm::new(),
            current: Scope::root(),
            functions: Vec::new(),
        }
    }

    /// The underlying realm (e.g. to render a result with `to_display_string`).
    #[must_use]
    pub fn realm(&self) -> &Realm {
        &self.realm
    }

    /// Runs a whole program, returning the value of its last expression
    /// statement (or `undefined`).
    pub fn run(&mut self, program: &'a Program) -> Result<NanBox, ExecError> {
        self.hoist(&program.body)?;
        let mut last = NanBox::undefined();
        for stmt in &program.body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = v,
                Flow::Return(v) => return Ok(v),
                Flow::Break | Flow::Continue => {}
            }
        }
        Ok(last)
    }

    // --- functions ---

    /// Pre-declares hoisted `function` declarations in the current scope, so a
    /// declaration is callable before its textual position (and mutual
    /// recursion works).
    fn hoist(&mut self, stmts: &'a [Stmt]) -> Result<(), ExecError> {
        for stmt in stmts {
            if let Stmt::Function(func) = stmt
                && let Some(id) = &func.id
            {
                let value = self.make_function(&func.params, Body::Block(&func.body));
                self.current.declare(&id.name, value);
            }
        }
        Ok(())
    }

    /// Registers a function definition and allocates a closure capturing the
    /// current scope.
    fn make_function(&mut self, params: &'a [Param], body: Body<'a>) -> NanBox {
        let func_id = self.functions.len() as u32;
        self.functions.push(FnDef { params, body });
        let handle = self.realm.new_function(func_id, self.current.clone());
        NanBox::handle(handle.to_raw())
    }

    /// Calls `callee` with `args`.
    fn call(&mut self, callee: NanBox, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let Some(raw) = callee.as_handle() else {
            return Err(ExecError::NotCallable);
        };
        let handle = crate::heap::Handle::from_raw(raw);
        let Some((func_id, captured)) = self.realm.function_at(handle) else {
            return Err(ExecError::NotCallable);
        };
        let def = self.functions[func_id as usize];

        // A fresh scope nested in the closure's captured environment.
        let call_scope = captured.child();
        for (i, param) in def.params.iter().enumerate() {
            let BindingTarget::Ident(Ident { name, .. }) = &param.target else {
                return Err(ExecError::Unsupported("destructuring parameter"));
            };
            let value = args.get(i).copied().unwrap_or(NanBox::undefined());
            call_scope.declare(name, value);
        }

        let saved = core::mem::replace(&mut self.current, call_scope);
        let result = self.run_body(def.body);
        self.current = saved;
        result
    }

    fn run_body(&mut self, body: Body<'a>) -> Result<NanBox, ExecError> {
        match body {
            Body::Expr(e) => self.eval(e),
            Body::Block(stmts) => {
                self.hoist(stmts)?;
                for stmt in stmts {
                    match self.exec(stmt)? {
                        Flow::Return(v) => return Ok(v),
                        Flow::Normal(_) | Flow::Break | Flow::Continue => {}
                    }
                }
                Ok(NanBox::undefined())
            }
        }
    }

    // --- statements ---

    fn exec(&mut self, stmt: &'a Stmt) -> Result<Flow, ExecError> {
        match stmt {
            Stmt::Empty { .. } => Ok(Flow::Normal(NanBox::undefined())),
            Stmt::Expr { expression, .. } => Ok(Flow::Normal(self.eval(expression)?)),
            Stmt::Var(decl) => {
                self.exec_var(decl)?;
                Ok(Flow::Normal(NanBox::undefined()))
            }
            // Function declarations are handled by hoisting; nothing to do here.
            Stmt::Function(_) => Ok(Flow::Normal(NanBox::undefined())),
            Stmt::Block { body, .. } => self.exec_block(body),
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval(test)?.to_boolean() {
                    self.exec(consequent)
                } else if let Some(alt) = alternate {
                    self.exec(alt)
                } else {
                    Ok(Flow::Normal(NanBox::undefined()))
                }
            }
            Stmt::While { test, body, .. } => {
                while self.eval(test)?.to_boolean() {
                    match self.exec(body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Normal(_) | Flow::Continue => {}
                    }
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => self.exec_for(init.as_ref(), test.as_deref(), update.as_deref(), body),
            Stmt::Return { argument, .. } => {
                let v = match argument {
                    Some(e) => self.eval(e)?,
                    None => NanBox::undefined(),
                };
                Ok(Flow::Return(v))
            }
            Stmt::Break { label: None, .. } => Ok(Flow::Break),
            Stmt::Continue { label: None, .. } => Ok(Flow::Continue),
            Stmt::Throw { argument, .. } => {
                let v = self.eval(argument)?;
                Err(ExecError::Throw(v))
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => self.exec_try(block, handler.as_ref(), finalizer.as_deref()),
            _ => Err(ExecError::Unsupported("statement")),
        }
    }

    /// `try { … } catch (e) { … } finally { … }`. A `catch` handles a thrown
    /// value; the `finally` block runs on every exit (normal, thrown, or
    /// `return`/`break`/`continue`), and its own abrupt completion takes over.
    fn exec_try(
        &mut self,
        block: &'a [Stmt],
        handler: Option<&'a crate::ast::CatchClause>,
        finalizer: Option<&'a [Stmt]>,
    ) -> Result<Flow, ExecError> {
        let mut outcome = self.exec_scoped(block);
        // A thrown value is routed to the catch clause, if any.
        if let (Err(ExecError::Throw(value)), Some(catch)) = (&outcome, handler) {
            let thrown = *value;
            let child = self.current.child();
            if let Some(BindingTarget::Ident(Ident { name, .. })) = &catch.param {
                child.declare(name, thrown);
            }
            let saved = core::mem::replace(&mut self.current, child);
            outcome = self.exec_seq(&catch.body);
            self.current = saved;
        }
        // `finally` runs regardless; an abrupt finally overrides the outcome.
        if let Some(fin) = finalizer {
            match self.exec_scoped(fin) {
                Ok(Flow::Normal(_)) => outcome,
                other => other, // finally returned/broke/threw → that wins
            }
        } else {
            outcome
        }
    }

    /// Runs a statement list in a fresh child scope.
    fn exec_scoped(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        let result = self.exec_seq(body);
        self.current = saved;
        result
    }

    fn exec_block(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        self.exec_scoped(body)
    }

    /// Executes a statement sequence in the current scope (with hoisting).
    fn exec_seq(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        self.hoist(body)?;
        let mut last = Flow::Normal(NanBox::undefined());
        for stmt in body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = Flow::Normal(v),
                other => return Ok(other),
            }
        }
        Ok(last)
    }

    fn exec_var(&mut self, decl: &'a VarDecl) -> Result<(), ExecError> {
        for d in &decl.declarations {
            let BindingTarget::Ident(Ident { name, .. }) = &d.target else {
                return Err(ExecError::Unsupported("destructuring binding"));
            };
            let value = match &d.init {
                Some(e) => self.eval(e)?,
                None => NanBox::undefined(),
            };
            self.current.declare(name, value);
        }
        Ok(())
    }

    fn exec_for(
        &mut self,
        init: Option<&'a ForInit>,
        test: Option<&'a Expr>,
        update: Option<&'a Expr>,
        body: &'a Stmt,
    ) -> Result<Flow, ExecError> {
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        let result = (|| {
            match init {
                Some(ForInit::Var(decl)) => self.exec_var(decl)?,
                Some(ForInit::Expr(e)) => {
                    self.eval(e)?;
                }
                None => {}
            }
            loop {
                let go = match test {
                    Some(t) => self.eval(t)?.to_boolean(),
                    None => true,
                };
                if !go {
                    break;
                }
                match self.exec(body)? {
                    Flow::Break => break,
                    Flow::Return(v) => return Ok(Flow::Return(v)),
                    Flow::Normal(_) | Flow::Continue => {}
                }
                if let Some(u) = update {
                    self.eval(u)?;
                }
            }
            Ok(Flow::Normal(NanBox::undefined()))
        })();
        self.current = saved;
        result
    }

    // --- expressions ---

    fn eval(&mut self, expr: &'a Expr) -> Result<NanBox, ExecError> {
        match expr {
            Expr::Null(_) => Ok(NanBox::null()),
            Expr::Bool { value, .. } => Ok(NanBox::boolean(*value)),
            Expr::Number { value, .. } => Ok(NanBox::number(*value)),
            Expr::Str { value, .. } => {
                let h = self.realm.new_string(value);
                Ok(NanBox::handle(h.to_raw()))
            }
            Expr::Ident(id) => match &*id.name {
                "undefined" => Ok(NanBox::undefined()),
                "NaN" => Ok(NanBox::number(f64::NAN)),
                "Infinity" => Ok(NanBox::number(f64::INFINITY)),
                name => self
                    .current
                    .get(name)
                    .ok_or_else(|| ExecError::NotDefined(String::from(name))),
            },
            Expr::Function(func) => Ok(self.eval_fn_expr(func)),
            Expr::Arrow(arrow) => Ok(self.eval_arrow(arrow)),
            Expr::Unary { op, argument, .. } => {
                let v = self.eval(argument)?;
                self.unary(*op, v)
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                let a = self.eval(left)?;
                let b = self.eval(right)?;
                self.binary(*op, a, b)
            }
            Expr::Logical {
                op, left, right, ..
            } => {
                let l = self.eval(left)?;
                let take_right = match op {
                    LogicalOp::And => l.to_boolean(),
                    LogicalOp::Or => !l.to_boolean(),
                    LogicalOp::Nullish => {
                        matches!(l.unpack(), Unpacked::Undefined | Unpacked::Null)
                    }
                };
                if take_right { self.eval(right) } else { Ok(l) }
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval(test)?.to_boolean() {
                    self.eval(consequent)
                } else {
                    self.eval(alternate)
                }
            }
            Expr::Assign {
                op, target, value, ..
            } => self.eval_assign(*op, target, value),
            Expr::Call {
                callee, arguments, ..
            } => {
                let f = self.eval(callee)?;
                let mut args = Vec::with_capacity(arguments.len());
                for a in arguments {
                    match a {
                        Argument::Item(e) => args.push(self.eval(e)?),
                        Argument::Spread(_) => return Err(ExecError::Unsupported("call spread")),
                    }
                }
                self.call(f, &args)
            }
            Expr::Array { elements, .. } => {
                let mut items = Vec::new();
                for el in elements {
                    match el {
                        ArrayElement::Hole => items.push(NanBox::undefined()),
                        ArrayElement::Item(e) => items.push(self.eval(e)?),
                        ArrayElement::Spread(_) => {
                            return Err(ExecError::Unsupported("array spread"));
                        }
                    }
                }
                let h = self.realm.new_array(items);
                Ok(NanBox::handle(h.to_raw()))
            }
            Expr::Object { members, .. } => {
                let handle = self.realm.new_object();
                for m in members {
                    let ObjectMember::Property { key, value, .. } = m else {
                        return Err(ExecError::Unsupported("object member"));
                    };
                    let k = static_key(key)?;
                    let v = self.eval(value)?;
                    self.realm.set_property(handle, &k, v);
                }
                Ok(NanBox::handle(handle.to_raw()))
            }
            Expr::Member {
                object,
                property,
                optional,
                ..
            } => {
                let obj = self.eval(object)?;
                if *optional && matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Ok(NanBox::undefined());
                }
                let Some(raw) = obj.as_handle() else {
                    return Ok(NanBox::undefined());
                };
                let handle = crate::heap::Handle::from_raw(raw);
                self.member(handle, property)
            }
            _ => Err(ExecError::Unsupported("expression")),
        }
    }

    fn eval_fn_expr(&mut self, func: &'a Function) -> NanBox {
        self.make_function(&func.params, Body::Block(&func.body))
    }

    fn eval_arrow(&mut self, arrow: &'a Arrow) -> NanBox {
        let body = match &arrow.body {
            ArrowBody::Expr(e) => Body::Expr(e),
            ArrowBody::Block(b) => Body::Block(b),
        };
        self.make_function(&arrow.params, body)
    }

    fn member(
        &mut self,
        handle: crate::heap::Handle,
        key: &'a PropertyKey,
    ) -> Result<NanBox, ExecError> {
        match key {
            PropertyKey::Number(n) if as_index(*n).is_some() => {
                Ok(self.realm.get_element(handle, as_index(*n).unwrap()))
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                if let Some(i) = k.as_number().and_then(as_index) {
                    return Ok(self.realm.get_element(handle, i));
                }
                let name = self.realm.to_display_string(k);
                Ok(self.member_value(handle, &name))
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(self.member_value(handle, s)),
            PropertyKey::Number(n) => Ok(self.member_value(handle, &alloc::format!("{n}"))),
            PropertyKey::Private(_) => Err(ExecError::Unsupported("private member")),
        }
    }

    fn member_value(&self, handle: crate::heap::Handle, key: &str) -> NanBox {
        if let Some(v) = self.realm.get_property(handle, key) {
            return v;
        }
        if key == "length" {
            if let Some(len) = self.realm.array_length(handle) {
                return NanBox::number(len as f64);
            }
            if let Some(s) = self.realm.string_value(handle) {
                return NanBox::number(s.chars().count() as f64);
            }
        }
        NanBox::undefined()
    }

    fn eval_assign(
        &mut self,
        op: AssignOp,
        target: &'a Expr,
        value: &'a Expr,
    ) -> Result<NanBox, ExecError> {
        let rhs = self.eval(value)?;
        match target {
            Expr::Ident(id) => {
                let name = &*id.name;
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self
                        .current
                        .get(name)
                        .ok_or_else(|| ExecError::NotDefined(String::from(name)))?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                if !self.current.set(name, new) {
                    self.current.declare(name, new); // sloppy global
                }
                Ok(new)
            }
            Expr::Member {
                object, property, ..
            } => {
                let obj = self.eval(object)?;
                let Some(raw) = obj.as_handle() else {
                    return Err(ExecError::Unsupported("member assign to non-object"));
                };
                let handle = crate::heap::Handle::from_raw(raw);
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self.member(handle, property)?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                self.assign_member(handle, property, new)?;
                Ok(new)
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
    }

    fn assign_member(
        &mut self,
        handle: crate::heap::Handle,
        property: &'a PropertyKey,
        new: NanBox,
    ) -> Result<(), ExecError> {
        match property {
            PropertyKey::Number(n) if as_index(*n).is_some() => {
                self.realm.set_element(handle, as_index(*n).unwrap(), new);
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                if let Some(i) = k.as_number().and_then(as_index) {
                    self.realm.set_element(handle, i, new);
                } else {
                    let name = self.realm.to_display_string(k);
                    self.realm.set_property(handle, &name, new);
                }
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                self.realm.set_property(handle, s, new);
            }
            PropertyKey::Number(n) => {
                self.realm.set_property(handle, &alloc::format!("{n}"), new);
            }
            PropertyKey::Private(_) => return Err(ExecError::Unsupported("private assign")),
        }
        Ok(())
    }

    fn unary(&mut self, op: UnaryOp, v: NanBox) -> Result<NanBox, ExecError> {
        Ok(match op {
            UnaryOp::Plus => NanBox::number(self.realm.to_number(v)),
            UnaryOp::Minus => self.realm.neg(v),
            UnaryOp::Not => self.realm.logical_not(v),
            UnaryOp::Typeof => {
                let t = self.realm.type_of_value(v);
                NanBox::handle(self.realm.new_string(t).to_raw())
            }
            UnaryOp::Void => NanBox::undefined(),
            #[cfg(feature = "std")]
            UnaryOp::BitNot => self.realm.bit_not(v),
            #[cfg(not(feature = "std"))]
            UnaryOp::BitNot => return Err(ExecError::Unsupported("~ needs std")),
            UnaryOp::Delete => return Err(ExecError::Unsupported("delete")),
        })
    }

    fn binary(&mut self, op: BinaryOp, a: NanBox, b: NanBox) -> Result<NanBox, ExecError> {
        Ok(match op {
            BinaryOp::Add => self.realm.add(a, b),
            BinaryOp::Sub => self.realm.sub(a, b),
            BinaryOp::Mul => self.realm.mul(a, b),
            BinaryOp::Div => self.realm.div(a, b),
            BinaryOp::Mod => self.realm.rem(a, b),
            BinaryOp::Lt => self.realm.less_than(a, b),
            BinaryOp::Gt => self.realm.greater_than(a, b),
            BinaryOp::LtEq => self.realm.less_equal(a, b),
            BinaryOp::GtEq => self.realm.greater_equal(a, b),
            BinaryOp::EqEq => NanBox::boolean(self.realm.loose_equals(a, b)),
            BinaryOp::NotEq => NanBox::boolean(!self.realm.loose_equals(a, b)),
            BinaryOp::EqEqEq => NanBox::boolean(self.realm.strict_equals(a, b)),
            BinaryOp::NotEqEq => NanBox::boolean(!self.realm.strict_equals(a, b)),
            #[cfg(feature = "std")]
            BinaryOp::Exp => self.realm.pow(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Shl => self.realm.shl(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Shr => self.realm.shr(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Ushr => self.realm.ushr(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitAnd => self.realm.bit_and(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitOr => self.realm.bit_or(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitXor => self.realm.bit_xor(a, b),
            #[cfg(not(feature = "std"))]
            BinaryOp::Exp
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::Ushr
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor => return Err(ExecError::Unsupported("** / bitwise need std")),
            BinaryOp::In | BinaryOp::Instanceof => {
                return Err(ExecError::Unsupported("in / instanceof"));
            }
        })
    }
}

/// The binary operator underlying a compound assignment (`+=` → `+`).
fn compound_op(op: AssignOp) -> Result<BinaryOp, ExecError> {
    Ok(match op {
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
        _ => return Err(ExecError::Unsupported("logical assignment")),
    })
}

/// A non-negative integer array index, if `n` is one.
fn as_index(n: f64) -> Option<usize> {
    if n >= 0.0 && n <= u32::MAX as f64 && (n as u64) as f64 == n {
        Some(n as usize)
    } else {
        None
    }
}

/// A static (non-computed) property key as a string.
fn static_key(key: &PropertyKey) -> Result<String, ExecError> {
    match key {
        PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(String::from(&**s)),
        PropertyKey::Number(n) => Ok(alloc::format!("{n}")),
        PropertyKey::Computed(_) => Err(ExecError::Unsupported("computed key")),
        PropertyKey::Private(_) => Err(ExecError::Unsupported("private key")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    /// Runs `src` and renders the program's final value.
    fn run(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        let value = interp.run(&program).expect("exec");
        interp.realm().to_display_string(value)
    }

    #[test]
    fn variables_assignment_and_control_flow() {
        assert_eq!(run("let x = 1; let y = 2; x + y"), "3");
        assert_eq!(run("let s = 'a'; s += 'b'; s += 'c'; s"), "abc");
        assert_eq!(run("let x = 1; { let x = 99; } x"), "1");
        assert_eq!(
            run("let s = 0; for (let i = 1; i <= 10; i += 1) s += i; s"),
            "55"
        );
        assert_eq!(
            run("let s = 0; for (let i = 0; i < 10; i += 1) { if (i === 5) break; s += i; } s"),
            "10"
        );
    }

    #[test]
    fn functions_and_return() {
        assert_eq!(run("function add(a, b) { return a + b; } add(2, 3)"), "5");
        assert_eq!(run("let sq = function (x) { return x * x; }; sq(7)"), "49");
        assert_eq!(run("let inc = x => x + 1; inc(41)"), "42");
        // Hoisting: callable before its definition.
        assert_eq!(run("f(10); function f(n) { return n; } f(10)"), "10");
    }

    #[test]
    fn closures_capture_their_scope() {
        // A returned inner function still sees the enclosing variable.
        assert_eq!(
            run(
                "function adder(n) { return function (x) { return x + n; }; }
                 let add5 = adder(5);
                 add5(10)"
            ),
            "15"
        );
        // The capture is by reference: a mutable counter.
        assert_eq!(
            run("function counter() {
                   let c = 0;
                   return function () { c += 1; return c; };
                 }
                 let next = counter();
                 next(); next(); next()"),
            "3"
        );
    }

    #[test]
    fn recursion() {
        assert_eq!(
            run("function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); } fact(5)"),
            "120"
        );
        assert_eq!(
            run("function fib(n) { return n < 2 ? n : fib(n-1) + fib(n-2); } fib(10)"),
            "55"
        );
    }

    #[test]
    fn higher_order_and_objects() {
        // A function stored on an object, called, mutating closed-over state.
        assert_eq!(
            run("function makeBox(v) {
                   let value = v;
                   return { get: function () { return value; },
                            set: function (x) { value = x; } };
                 }
                 let b = makeBox(1);
                 b.set(99);
                 b.get()"),
            "99"
        );
    }

    #[test]
    fn calling_a_non_function_errors() {
        let program = Parser::parse_program("let x = 5; x()").unwrap();
        let mut interp = Interp::new();
        assert_eq!(interp.run(&program), Err(ExecError::NotCallable));
    }

    #[test]
    fn try_catch_finally() {
        // A thrown value is caught and bound.
        assert_eq!(
            run("let r; try { throw 'boom'; r = 'no'; } catch (e) { r = 'caught:' + e; } r"),
            "caught:boom"
        );
        // No throw: the catch is skipped.
        assert_eq!(
            run("let r = 'ok'; try { r = 'a'; } catch (e) { r = 'b'; } r"),
            "a"
        );
        // finally always runs.
        assert_eq!(
            run(
                "let log = ''; try { log += '1'; throw 0; } catch (e) { log += '2'; } finally { log += '3'; } log"
            ),
            "123"
        );
        // A throw out of a function is caught at the call site.
        assert_eq!(
            run("function boom() { throw 'x'; }
                 let r; try { boom(); } catch (e) { r = 'got:' + e; } r"),
            "got:x"
        );
        // catch without a binding.
        assert_eq!(
            run("let r = 'a'; try { throw 1; } catch { r = 'b'; } r"),
            "b"
        );
    }

    #[test]
    fn uncaught_throw_propagates() {
        let program = Parser::parse_program("throw 'oops'").unwrap();
        let mut interp = Interp::new();
        match interp.run(&program) {
            Err(ExecError::Throw(v)) => {
                assert_eq!(interp.realm().to_display_string(v), "oops");
            }
            other => panic!("expected a throw, got {other:?}"),
        }
    }

    #[test]
    fn finally_return_overrides() {
        // A `return` in finally overrides the try's outcome.
        assert_eq!(
            run("function f() { try { return 'a'; } finally { return 'b'; } } f()"),
            "b"
        );
    }
}
