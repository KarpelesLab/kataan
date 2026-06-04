//! Executing real **statements** over the [`Realm`]/[`NanBox`] model
//! (`ROADMAP.md` §3 → Phase D migration).
//!
//! [`Realm`]: crate::realm::Realm
//! [`NanBox`]: crate::nanbox::NanBox
//!
//! Where [`nbeval`](crate::nbeval) evaluates a single expression, this adds the
//! machinery a real program needs: **lexical variable scope** (`let`/`const`/
//! `var`), assignment (incl. compound `+=` etc.), block scoping, and control
//! flow (`if`/`while`/`for`, `return`/`break`/`continue`). It is a small
//! tree-walking interpreter whose values are NaN-boxed and whose objects/strings/
//! arrays live in the realm's GC heap — i.e. the imperative core of the language
//! running on the performance representation.
//!
//! Functions/closures, exceptions, and the stdlib are still ahead (they are the
//! remainder of the structural migration); this proves variables, assignment,
//! and control flow execute on the new model. Pure, safe `alloc`-only Rust.

use crate::ast::{
    ArrayElement, AssignOp, BinaryOp, BindingTarget, Expr, ForInit, Ident, LogicalOp, ObjectMember,
    Program, PropertyKey, Stmt, UnaryOp, VarDecl,
};
use crate::nanbox::{NanBox, Unpacked};
use crate::realm::Realm;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Why execution stopped.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExecError {
    /// A construct outside the supported subset (functions, try/catch, …).
    Unsupported(&'static str),
    /// A reference to an undeclared variable.
    NotDefined(String),
}

/// The control-flow outcome of a statement.
enum Flow {
    /// Fell through normally, carrying the last expression value (for REPL/`run`).
    Normal(NanBox),
    /// A `return` (value).
    Return(NanBox),
    /// A `break`.
    Break,
    /// A `continue`.
    Continue,
}

/// A tree-walking interpreter over the performance object model.
pub struct Interp {
    realm: Realm,
    /// Lexical scope stack; the last frame is the innermost.
    scopes: Vec<BTreeMap<String, NanBox>>,
}

impl Default for Interp {
    fn default() -> Self {
        Self::new()
    }
}

impl Interp {
    /// A fresh interpreter with a single (global) scope.
    #[must_use]
    pub fn new() -> Self {
        Self {
            realm: Realm::new(),
            scopes: alloc::vec![BTreeMap::new()],
        }
    }

    /// The underlying realm (e.g. to render a result with `to_display_string`).
    #[must_use]
    pub fn realm(&self) -> &Realm {
        &self.realm
    }

    /// Runs a whole program, returning the value of its last expression
    /// statement (or `undefined`).
    pub fn run(&mut self, program: &Program) -> Result<NanBox, ExecError> {
        let mut last = NanBox::undefined();
        for stmt in &program.body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = v,
                Flow::Return(v) => return Ok(v),
                Flow::Break | Flow::Continue => {} // ignored at top level
            }
        }
        Ok(last)
    }

    // --- scope ---

    fn declare(&mut self, name: &str, value: NanBox) {
        self.scopes
            .last_mut()
            .expect("a scope")
            .insert(String::from(name), value);
    }

    fn lookup(&self, name: &str) -> Option<NanBox> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    /// Assigns to the nearest existing binding; returns false if none exists.
    fn assign(&mut self, name: &str, value: NanBox) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(name) {
                *slot = value;
                return true;
            }
        }
        false
    }

    // --- statements ---

    fn exec(&mut self, stmt: &Stmt) -> Result<Flow, ExecError> {
        match stmt {
            Stmt::Empty { .. } => Ok(Flow::Normal(NanBox::undefined())),
            Stmt::Expr { expression, .. } => Ok(Flow::Normal(self.eval(expression)?)),
            Stmt::Var(decl) => {
                self.exec_var(decl)?;
                Ok(Flow::Normal(NanBox::undefined()))
            }
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
            _ => Err(ExecError::Unsupported("statement")),
        }
    }

    fn exec_block(&mut self, body: &[Stmt]) -> Result<Flow, ExecError> {
        self.scopes.push(BTreeMap::new());
        let mut result = Ok(Flow::Normal(NanBox::undefined()));
        for stmt in body {
            match self.exec(stmt) {
                Ok(Flow::Normal(v)) => result = Ok(Flow::Normal(v)),
                other => {
                    result = other;
                    break;
                }
            }
        }
        self.scopes.pop();
        result
    }

    fn exec_var(&mut self, decl: &VarDecl) -> Result<(), ExecError> {
        for d in &decl.declarations {
            let BindingTarget::Ident(Ident { name, .. }) = &d.target else {
                return Err(ExecError::Unsupported("destructuring binding"));
            };
            let value = match &d.init {
                Some(e) => self.eval(e)?,
                None => NanBox::undefined(),
            };
            self.declare(name, value);
        }
        Ok(())
    }

    fn exec_for(
        &mut self,
        init: Option<&ForInit>,
        test: Option<&Expr>,
        update: Option<&Expr>,
        body: &Stmt,
    ) -> Result<Flow, ExecError> {
        // The loop gets its own scope so a `let` in the header is per-loop.
        self.scopes.push(BTreeMap::new());
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
        self.scopes.pop();
        result
    }

    // --- expressions ---

    fn eval(&mut self, expr: &Expr) -> Result<NanBox, ExecError> {
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
                    .lookup(name)
                    .ok_or_else(|| ExecError::NotDefined(String::from(name))),
            },
            Expr::Unary { op, argument, .. } => {
                let v = self.eval(argument)?;
                Ok(self.unary(*op, v)?)
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

    fn member(
        &mut self,
        handle: crate::heap::Handle,
        key: &PropertyKey,
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
        target: &Expr,
        value: &Expr,
    ) -> Result<NanBox, ExecError> {
        let rhs = self.eval(value)?;
        // Only identifier and `obj.x`/`obj[i]` targets are supported here.
        match target {
            Expr::Ident(id) => {
                let name = &*id.name;
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self
                        .lookup(name)
                        .ok_or_else(|| ExecError::NotDefined(String::from(name)))?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                if !self.assign(name, new) {
                    // An assignment to an undeclared name creates a global (sloppy).
                    self.scopes[0].insert(String::from(name), new);
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
                    PropertyKey::Private(_) => {
                        return Err(ExecError::Unsupported("private assign"));
                    }
                }
                Ok(new)
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
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
        // Logical assignment (`&&=`/`||=`/`??=`) short-circuits — not handled
        // by the plain-binary path.
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
    fn variables_and_assignment() {
        assert_eq!(run("let x = 1; let y = 2; x + y"), "3");
        assert_eq!(run("let x = 5; x = x * 2; x"), "10");
        assert_eq!(run("let x = 1; x += 4; x -= 2; x"), "3");
        assert_eq!(run("let s = 'a'; s += 'b'; s += 'c'; s"), "abc");
    }

    #[test]
    fn block_scope() {
        // An inner block shadows; the outer binding is restored after.
        assert_eq!(run("let x = 1; { let x = 99; } x"), "1");
        // Assignment without a new `let` reaches the outer binding.
        assert_eq!(run("let x = 1; { x = 5; } x"), "5");
    }

    #[test]
    fn if_else() {
        assert_eq!(
            run("let r; if (1 < 2) { r = 'a'; } else { r = 'b'; } r"),
            "a"
        );
        assert_eq!(run("let r; if (1 > 2) r = 'a'; else r = 'b'; r"), "b");
    }

    #[test]
    fn while_and_for_loops() {
        assert_eq!(
            run("let s = 0; let i = 0; while (i < 5) { s += i; i += 1; } s"),
            "10"
        );
        assert_eq!(
            run("let s = 0; for (let i = 1; i <= 10; i += 1) s += i; s"),
            "55"
        );
        // break / continue.
        assert_eq!(
            run("let s = 0; for (let i = 0; i < 10; i += 1) { if (i === 5) break; s += i; } s"),
            "10"
        );
        assert_eq!(
            run(
                "let s = 0; for (let i = 0; i < 6; i += 1) { if (i % 2 === 1) continue; s += i; } s"
            ),
            "6"
        );
    }

    #[test]
    fn builds_an_array_and_sums_it() {
        assert_eq!(
            run(
                "let a = [10, 20, 30]; let s = 0; for (let i = 0; i < a.length; i += 1) s += a[i]; s"
            ),
            "60"
        );
    }

    #[test]
    fn object_mutation() {
        assert_eq!(
            run("let o = { count: 0 }; o.count += 5; o.count = o.count * 2; o.count"),
            "10"
        );
    }

    #[test]
    fn undeclared_variable_is_an_error() {
        let program = Parser::parse_program("y + 1").unwrap();
        let mut interp = Interp::new();
        assert_eq!(
            interp.run(&program),
            Err(ExecError::NotDefined(String::from("y")))
        );
    }
}
