//! True lazy generators for the tree-walking interpreter (`nbexec`).
//!
//! The previous model ran a `function*` body **eagerly** at call time, buffering
//! every `yield`ed value into a `Vec`, then handed back an iterator that merely
//! replayed the buffer. That model could not inject a value at a suspended
//! `yield` (`next(v)`), could not resume a `try`/`finally` for `return()`, could
//! not throw *into* the generator (`throw()`), and — fatally — hung or ran out
//! of memory on an infinite generator (`function*(){ while (true) yield 1 }`).
//!
//! This module implements **true suspension**. Because the interpreter is a
//! recursive tree-walker over `Rc`-based, non-`Send` values and the crate denies
//! `unsafe`, neither a stackful coroutine (would need raw stack switching) nor a
//! worker thread (would need `Send`) is available. Instead the generator body is
//! run by an **explicit-stack state machine** whose execution position is reified
//! on the heap (a `Vec` of [`Step`]s plus an operand stack), so it can be parked
//! at a `yield` and resumed later — without relying on the native Rust call stack
//! to hold the continuation.
//!
//! The machine reifies only the statement/expression forms that can lie on the
//! path from a generator body to a `yield`: blocks, `if`, `while`/`do`/`for`/
//! `for-of`/`for-in`, `try`/`catch`/`finally`, `switch`, labeled statements,
//! `var`/`return`/`throw`/expression statements, and — in expression position —
//! `yield`/`yield*`, sequence, conditional, logical, and assignment to a simple
//! identifier target. **Any subtree that provably contains no reachable `yield`**
//! is delegated to the ordinary, already-correct `exec`/`eval` walker and runs in
//! one shot — so non-generator semantics are reused verbatim and only the
//! yield-bearing spine is interpreted step-by-step.

use super::*;
use crate::ast::{
    Argument, BindingTarget, CatchClause, Expr, ForInit, ForLeft, Ident, LogicalOp, Stmt,
    SwitchCase, VarDeclKind,
};

/// How a generator is being resumed.
#[derive(Clone, Copy)]
pub(crate) enum Resumption {
    /// `gen.next(v)` — `v` becomes the value of the suspended `yield`.
    Next(NanBox),
    /// `gen.return(v)` — resume as if `return v` ran at the suspension point.
    Return(NanBox),
    /// `gen.throw(e)` — resume by throwing `e` at the suspension point.
    Throw(NanBox),
}

/// The result of advancing a generator one step.
pub(crate) enum GenStep {
    /// Hit a `yield`; the operand is the surfaced value (`{value, done:false}`).
    Yielded(NanBox),
    /// Ran to completion / `return`; the operand is the result (`{value, done:true}`).
    Done(NanBox),
}

/// A suspended generator activation.
pub(crate) struct GenFrame<'a> {
    body: &'a [Stmt],
    scope: Scope,
    this_val: NanBox,
    new_target: NanBox,
    home_class: Option<u32>,
    home_static: bool,
    home_object: Option<Handle>,
    strict: bool,
    /// The reified continuation: an explicit stack of pending steps.
    stack: Vec<Step<'a>>,
    /// Operand-value stack used by expression steps.
    values: Vec<NanBox>,
    started: bool,
    done: bool,
    running: bool,
}

/// One pending action on the generator's explicit execution stack.
enum Step<'a> {
    /// Run statements `body[idx..]`; `scope` (if `Some`) is restored on exit.
    Seq {
        body: &'a [Stmt],
        idx: usize,
        scope: Option<Scope>,
        label: Option<String>,
    },
    /// A `while` loop: re-evaluate `test` before each iteration.
    While {
        test: &'a Expr,
        body: &'a Stmt,
        label: Option<String>,
    },
    /// A `do…while` loop. `test_first` true means run the test before the body.
    DoWhile {
        body: &'a Stmt,
        test: &'a Expr,
        label: Option<String>,
        test_first: bool,
    },
    /// A C-style `for (; test; update) body`.
    ForLoop {
        test: Option<&'a Expr>,
        update: Option<&'a Expr>,
        body: &'a Stmt,
        label: Option<String>,
        ran_body: bool,
        scope: Scope,
    },
    /// Drive a `for-of`/`for-in` over a precomputed value list.
    ForEach {
        left: &'a ForLeft,
        body: &'a Stmt,
        values: Vec<NanBox>,
        idx: usize,
        label: Option<String>,
    },
    /// A `try` region marker: while present, a throw is routed to `handler` (if
    /// any) and `finalizer` (if any) runs on the way out.
    TryRegion {
        handler: Option<&'a CatchClause>,
        finalizer: Option<&'a [Stmt]>,
        scope: Scope,
    },
    /// Restore `scope` (block / catch / loop-iteration exit).
    PopScope { scope: Scope },
    /// After a `finally` block completes normally, re-apply the buffered
    /// completion it overrides.
    Finally { pending: Completion },
    /// A `yield`/`yield*` whose operand value is on the value stack.
    YieldExpr { delegate: bool },
    /// A `yield*` delegation pumping `iter`.
    YieldStar { iter: Handle },
    /// Assign the value on the stack to simple target `name`, leaving the value
    /// on the stack as the assignment's result.
    AssignName { name: &'a str },
    /// A logical `&&`/`||`/`??` with its left operand on the stack.
    Logical { op: LogicalOp, right: &'a Expr },
    /// A conditional `test ? consequent : alternate` with `test` on the stack.
    Conditional {
        consequent: &'a Expr,
        alternate: &'a Expr,
    },
    /// The remaining expressions of a comma sequence, evaluated for the result.
    SeqExpr { rest: &'a [Expr] },
    /// Pop and discard the top value (a yield expression run as a statement).
    Discard,
    /// `return <value on stack>`.
    ReturnValue,
    /// `throw <value on stack>`.
    ThrowValue,
    /// `var`/`let`/`const <name> = <value on stack>` (simple identifier target).
    DeclName { name: &'a str, is_const: bool },
    /// The remaining declarators of a `var`/`let`/`const` declaration.
    VarTail {
        kind: VarDeclKind,
        rest: &'a [crate::ast::VarDeclarator],
    },
    /// A `switch` body region: consumes an unlabeled `break`.
    SwitchRegion,
    /// Evaluate `expr`, leaving its value on the value stack (used to evaluate a
    /// binary operator's right operand after the left is computed).
    EvalThen { expr: &'a Expr },
    /// Combine the top two stack values with `op` (left below, right on top).
    BinaryOp { op: crate::ast::BinaryOp },
}

/// A completion threaded through the unwinder.
#[derive(Clone)]
enum Completion {
    Return(NanBox),
    Throw(NanBox),
    Break(Option<String>),
    Continue(Option<String>),
}

/// The result of one [`Interp::gen_step`].
enum StepOut {
    Continue,
    Yield(NanBox),
}

/// An abrupt completion produced while stepping.
enum GenAbrupt {
    Throw(NanBox),
    Return(NanBox),
    Break(Option<String>),
    Continue(Option<String>),
    /// A non-throw interpreter error — abort the whole run.
    Fatal(ExecError),
}

impl From<ExecError> for GenAbrupt {
    fn from(e: ExecError) -> Self {
        match e {
            ExecError::Throw(v) => GenAbrupt::Throw(v),
            other => GenAbrupt::Fatal(other),
        }
    }
}

type StepResult = Result<StepOut, GenAbrupt>;

impl<'a> Interp<'a> {
    /// Builds a suspended lazy-generator object backed by a [`GenFrame`].
    pub(crate) fn make_lazy_generator(&mut self, body: &'a [Stmt], scope: Scope) -> NanBox {
        let frame = GenFrame {
            body,
            scope,
            this_val: self.this_val,
            new_target: self.new_target,
            home_class: self.current_home,
            home_static: self.current_home_static,
            home_object: self.current_home_object,
            strict: self.strict,
            stack: Vec::new(),
            values: Vec::new(),
            started: false,
            done: false,
            running: false,
        };
        let id = if let Some(slot) = self.gen_frames.iter().position(Option::is_none) {
            self.gen_frames[slot] = Some(frame);
            slot
        } else {
            self.gen_frames.push(Some(frame));
            self.gen_frames.len() - 1
        };
        let obj = self.realm.new_object();
        self.realm
            .set_hidden_property(obj, GEN_FRAME, NanBox::number(id as f64));
        for (name, nid) in [
            ("next", N_GEN_NEXT),
            ("return", N_GEN_RETURN),
            ("throw", N_GEN_THROW),
        ] {
            let f = self.realm.new_native(nid);
            self.realm
                .set_property(obj, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(obj, name);
        }
        if let Some(iter_proto) = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_object_proto(obj, Some(iter_proto));
        }
        NanBox::handle(obj.to_raw())
    }

    /// Whether `h` is a lazy-generator object (carries a [`GenFrame`] id).
    pub(crate) fn gen_frame_id(&self, h: Handle) -> Option<usize> {
        self.realm
            .get_property(h, GEN_FRAME)
            .and_then(|v| v.as_number())
            .map(|n| n as usize)
    }

    /// `gen.next/return/throw` on a lazy generator: resumes the machine and
    /// returns an `{value, done}` result (or propagates an escaping throw).
    pub(crate) fn lazy_gen_resume(
        &mut self,
        this: NanBox,
        how: Resumption,
    ) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Generator method called on non-object"));
        };
        let Some(id) = self.gen_frame_id(h) else {
            return Err(self.type_error("Generator method called on a non-generator"));
        };
        match self.run_generator(id, how)? {
            GenStep::Yielded(v) => Ok(self.gen_result(v, false)),
            GenStep::Done(v) => Ok(self.gen_result(v, true)),
        }
    }

    fn gen_result(&mut self, value: NanBox, done: bool) -> NanBox {
        let r = self.realm.new_object();
        self.realm.set_property(r, "value", value);
        self.realm.set_property(r, "done", NanBox::boolean(done));
        NanBox::handle(r.to_raw())
    }

    fn run_generator(&mut self, id: usize, how: Resumption) -> Result<GenStep, ExecError> {
        if self.gen_frames[id].as_ref().is_none_or(|f| f.done) {
            return match how {
                Resumption::Next(_) => Ok(GenStep::Done(NanBox::undefined())),
                Resumption::Return(v) => Ok(GenStep::Done(v)),
                Resumption::Throw(e) => Err(ExecError::Throw(e)),
            };
        }
        if self.gen_frames[id].as_ref().is_some_and(|f| f.running) {
            return Err(self.type_error("Generator is already running"));
        }
        let started = self.gen_frames[id].as_ref().is_some_and(|f| f.started);
        if !started {
            match how {
                Resumption::Return(v) => {
                    self.gen_frames[id].as_mut().unwrap().done = true;
                    return Ok(GenStep::Done(v));
                }
                Resumption::Throw(e) => {
                    self.gen_frames[id].as_mut().unwrap().done = true;
                    return Err(ExecError::Throw(e));
                }
                Resumption::Next(_) => {}
            }
        }

        let (scope, this_val, new_target, home_class, home_static, home_object, strict) = {
            let f = self.gen_frames[id].as_ref().expect("frame present");
            (
                f.scope.clone(),
                f.this_val,
                f.new_target,
                f.home_class,
                f.home_static,
                f.home_object,
                f.strict,
            )
        };
        let saved_scope = core::mem::replace(&mut self.current, scope);
        let saved_this = core::mem::replace(&mut self.this_val, this_val);
        let saved_target = core::mem::replace(&mut self.new_target, new_target);
        let saved_home = core::mem::replace(&mut self.current_home, home_class);
        let saved_home_static = core::mem::replace(&mut self.current_home_static, home_static);
        let saved_home_obj = core::mem::replace(&mut self.current_home_object, home_object);
        let saved_strict = core::mem::replace(&mut self.strict, strict);
        self.gen_frames[id].as_mut().unwrap().running = true;

        let outcome = self.gen_drive(id, how, started);

        if let Some(f) = self.gen_frames[id].as_mut() {
            f.scope = core::mem::replace(&mut self.current, saved_scope);
            f.running = false;
        } else {
            self.current = saved_scope;
        }
        self.this_val = saved_this;
        self.new_target = saved_target;
        self.current_home = saved_home;
        self.current_home_static = saved_home_static;
        self.current_home_object = saved_home_obj;
        self.strict = saved_strict;

        match &outcome {
            Ok(GenStep::Done(_)) | Err(_) => {
                if let Some(f) = self.gen_frames[id].as_mut() {
                    f.done = true;
                    f.stack.clear();
                    f.values.clear();
                }
            }
            Ok(GenStep::Yielded(_)) => {}
        }
        outcome
    }

    fn gen_drive(
        &mut self,
        id: usize,
        how: Resumption,
        started: bool,
    ) -> Result<GenStep, ExecError> {
        let (mut stack, mut values) = {
            let f = self.gen_frames[id].as_mut().expect("frame present");
            (
                core::mem::take(&mut f.stack),
                core::mem::take(&mut f.values),
            )
        };

        if !started {
            let body = self.gen_frames[id].as_ref().expect("frame present").body;
            if let Err(e) = self.hoist(body) {
                self.store_machine(id, stack, values);
                return Err(e);
            }
            stack.push(Step::Seq {
                body,
                idx: 0,
                scope: None,
                label: None,
            });
            self.gen_frames[id].as_mut().unwrap().started = true;
        }

        // Inject the resumption at the suspended position. If the suspension is a
        // `yield*` delegation (a `YieldStar` step on top), the resumption — even a
        // `return`/`throw` — is forwarded to the inner iterator.
        let mut pending: Option<Completion> = None;
        if started && matches!(stack.last(), Some(Step::YieldStar { .. })) {
            let Some(Step::YieldStar { iter }) = stack.pop() else {
                unreachable!()
            };
            match self.gen_yield_star_step(iter, how, &mut stack, &mut values) {
                Ok(StepOut::Yield(v)) => {
                    self.store_machine(id, stack, values);
                    return Ok(GenStep::Yielded(v));
                }
                Ok(StepOut::Continue) => {}
                Err(GenAbrupt::Throw(e)) => pending = Some(Completion::Throw(e)),
                Err(GenAbrupt::Return(v)) => pending = Some(Completion::Return(v)),
                Err(GenAbrupt::Break(l)) => pending = Some(Completion::Break(l)),
                Err(GenAbrupt::Continue(l)) => pending = Some(Completion::Continue(l)),
                Err(GenAbrupt::Fatal(e)) => {
                    self.store_machine(id, stack, values);
                    return Err(e);
                }
            }
        } else {
            match how {
                Resumption::Next(v) => values.push(v),
                Resumption::Return(v) => pending = Some(Completion::Return(v)),
                Resumption::Throw(e) => pending = Some(Completion::Throw(e)),
            }
        }

        let result = loop {
            if let Some(c) = pending.take() {
                match self.gen_unwind(&mut stack, &mut values, c) {
                    Ok(None) => {} // resumed into a finally / catch
                    Ok(Some(done)) => break done,
                    Err(e) => {
                        self.store_machine(id, stack, values);
                        return Err(e);
                    }
                }
                continue;
            }
            // An empty stack means the body ran to normal completion.
            if stack.is_empty() {
                break Ok(GenStep::Done(NanBox::undefined()));
            }
            match self.gen_step(&mut stack, &mut values) {
                Ok(StepOut::Continue) => {}
                Ok(StepOut::Yield(v)) => break Ok(GenStep::Yielded(v)),
                Err(GenAbrupt::Throw(e)) => pending = Some(Completion::Throw(e)),
                Err(GenAbrupt::Return(v)) => pending = Some(Completion::Return(v)),
                Err(GenAbrupt::Break(l)) => pending = Some(Completion::Break(l)),
                Err(GenAbrupt::Continue(l)) => pending = Some(Completion::Continue(l)),
                Err(GenAbrupt::Fatal(e)) => {
                    self.store_machine(id, stack, values);
                    return Err(e);
                }
            }
        };
        self.store_machine(id, stack, values);
        result
    }

    fn store_machine(&mut self, id: usize, stack: Vec<Step<'a>>, values: Vec<NanBox>) {
        if let Some(f) = self.gen_frames[id].as_mut() {
            f.stack = stack;
            f.values = values;
        }
    }
}

// --- yield detection ---------------------------------------------------------

/// Whether a statement may execute a `yield` reachable in the *current*
/// generator (i.e. not nested inside another function/class boundary, which has
/// its own generator context). A yield-free statement is run in one shot by the
/// ordinary `exec` walker.
fn stmt_has_yield(s: &Stmt) -> bool {
    match s {
        Stmt::Expr { expression, .. } => expr_has_yield(expression),
        Stmt::Block { body, .. } => body.iter().any(stmt_has_yield),
        Stmt::Empty { .. }
        | Stmt::Break { .. }
        | Stmt::Continue { .. }
        | Stmt::Debugger { .. }
        | Stmt::Function(_)
        | Stmt::Class(_)
        | Stmt::Import(_)
        | Stmt::Export(_) => false,
        Stmt::Var(decl) => decl
            .declarations
            .iter()
            .any(|d| d.init.as_ref().is_some_and(expr_has_yield)),
        Stmt::If {
            test,
            consequent,
            alternate,
            ..
        } => {
            expr_has_yield(test)
                || stmt_has_yield(consequent)
                || alternate.as_deref().is_some_and(stmt_has_yield)
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            init.as_ref().is_some_and(|i| match i {
                ForInit::Var(d) => d
                    .declarations
                    .iter()
                    .any(|x| x.init.as_ref().is_some_and(expr_has_yield)),
                ForInit::Expr(e) => expr_has_yield(e),
            }) || test.as_deref().is_some_and(expr_has_yield)
                || update.as_deref().is_some_and(expr_has_yield)
                || stmt_has_yield(body)
        }
        Stmt::ForIn { right, body, .. } | Stmt::ForOf { right, body, .. } => {
            expr_has_yield(right) || stmt_has_yield(body)
        }
        Stmt::While { test, body, .. } => expr_has_yield(test) || stmt_has_yield(body),
        Stmt::DoWhile { body, test, .. } => stmt_has_yield(body) || expr_has_yield(test),
        Stmt::Switch {
            discriminant,
            cases,
            ..
        } => {
            expr_has_yield(discriminant)
                || cases.iter().any(|c| {
                    c.test.as_ref().is_some_and(expr_has_yield) || c.body.iter().any(stmt_has_yield)
                })
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            block.iter().any(stmt_has_yield)
                || handler
                    .as_ref()
                    .is_some_and(|h| h.body.iter().any(stmt_has_yield))
                || finalizer
                    .as_ref()
                    .is_some_and(|f| f.iter().any(stmt_has_yield))
        }
        Stmt::Return { argument, .. } => argument.as_deref().is_some_and(expr_has_yield),
        Stmt::Throw { argument, .. } => expr_has_yield(argument),
        Stmt::Labeled { body, .. } => stmt_has_yield(body),
        Stmt::With { object, body, .. } => expr_has_yield(object) || stmt_has_yield(body),
    }
}

/// Whether an expression may execute a `yield` reachable in the current
/// generator. Stops at nested function/arrow/class boundaries (their bodies have
/// their own generator/non-generator context).
fn expr_has_yield(e: &Expr) -> bool {
    match e {
        Expr::Yield { .. } => true,
        // Boundaries: a nested function/arrow/class introduces its own context.
        Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_) => false,
        Expr::Null(_)
        | Expr::Bool { .. }
        | Expr::Number { .. }
        | Expr::BigInt { .. }
        | Expr::Str { .. }
        | Expr::Regex { .. }
        | Expr::Ident(_)
        | Expr::PrivateName(..)
        | Expr::This(_)
        | Expr::Super(_)
        | Expr::NewTarget(_) => false,
        Expr::Template(t) => t.expressions.iter().any(expr_has_yield),
        Expr::TaggedTemplate { tag, quasi, .. } => {
            expr_has_yield(tag) || quasi.expressions.iter().any(expr_has_yield)
        }
        Expr::Array { elements, .. } => elements.iter().any(|el| match el {
            crate::ast::ArrayElement::Hole => false,
            crate::ast::ArrayElement::Item(e) | crate::ast::ArrayElement::Spread(e) => {
                expr_has_yield(e)
            }
        }),
        Expr::Object { members, .. } => members.iter().any(|m| match m {
            crate::ast::ObjectMember::Property { key, value, .. } => {
                key_has_yield(key) || expr_has_yield(value)
            }
            crate::ast::ObjectMember::Spread { value, .. } => expr_has_yield(value),
            crate::ast::ObjectMember::Accessor { .. } => false,
        }),
        Expr::Member {
            object, property, ..
        } => expr_has_yield(object) || key_has_yield(property),
        Expr::Call {
            callee, arguments, ..
        }
        | Expr::New {
            callee, arguments, ..
        } => {
            expr_has_yield(callee)
                || arguments.iter().any(|a| match a {
                    Argument::Item(e) | Argument::Spread(e) => expr_has_yield(e),
                })
        }
        Expr::OptChain { expr, .. } | Expr::Await { argument: expr, .. } => expr_has_yield(expr),
        Expr::Unary { argument, .. } | Expr::Update { argument, .. } => expr_has_yield(argument),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            expr_has_yield(left) || expr_has_yield(right)
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => expr_has_yield(test) || expr_has_yield(consequent) || expr_has_yield(alternate),
        Expr::Assign { target, value, .. } => expr_has_yield(target) || expr_has_yield(value),
        Expr::Sequence { expressions, .. } => expressions.iter().any(expr_has_yield),
    }
}

fn key_has_yield(k: &crate::ast::PropertyKey) -> bool {
    matches!(k, crate::ast::PropertyKey::Computed(e) if expr_has_yield(e))
}

// --- the stepping machine ----------------------------------------------------

impl<'a> Interp<'a> {
    /// Executes one step of the generator machine: pops the top [`Step`] and
    /// processes it, pushing follow-up steps and/or leaving values on the value
    /// stack. Returns `Yield(v)` when a `yield` is reached.
    fn gen_step(&mut self, stack: &mut Vec<Step<'a>>, values: &mut Vec<NanBox>) -> StepResult {
        let Some(step) = stack.pop() else {
            // The stack is empty: the body completed normally.
            return Ok(StepOut::Continue);
        };
        match step {
            Step::Seq {
                body,
                idx,
                scope,
                label,
            } => {
                if idx >= body.len() {
                    if let Some(s) = scope {
                        self.current = s;
                    }
                    return Ok(StepOut::Continue);
                }
                let stmt = &body[idx];
                // Re-push the continuation of this sequence first.
                stack.push(Step::Seq {
                    body,
                    idx: idx + 1,
                    scope,
                    label: label.clone(),
                });
                self.gen_exec_stmt(stmt, stack, values, &label)
            }
            Step::While { test, body, label } => {
                let t = self.eval(test).map_err(GenAbrupt::from)?;
                if self.realm.truthy(t) {
                    // Re-push the loop, then run one body iteration.
                    stack.push(Step::While {
                        test,
                        body,
                        label: label.clone(),
                    });
                    self.gen_exec_loop_body(body, stack, values, &label)
                } else {
                    Ok(StepOut::Continue)
                }
            }
            Step::DoWhile {
                body,
                test,
                label,
                test_first,
            } => {
                if test_first {
                    let t = self.eval(test).map_err(GenAbrupt::from)?;
                    if !self.realm.truthy(t) {
                        return Ok(StepOut::Continue);
                    }
                }
                stack.push(Step::DoWhile {
                    body,
                    test,
                    label: label.clone(),
                    test_first: true,
                });
                self.gen_exec_loop_body(body, stack, values, &label)
            }
            Step::ForLoop {
                test,
                update,
                body,
                label,
                ran_body,
                scope,
            } => {
                // Run the update after a completed body iteration.
                let saved = core::mem::replace(&mut self.current, scope.clone());
                if ran_body
                    && let Some(u) = update
                    && let Err(e) = self.eval(u)
                {
                    self.current = saved;
                    return Err(GenAbrupt::from(e));
                }
                // Evaluate the test (absent test → always true).
                let go = match test {
                    Some(t) => match self.eval(t) {
                        Ok(v) => self.realm.truthy(v),
                        Err(e) => {
                            self.current = saved;
                            return Err(GenAbrupt::from(e));
                        }
                    },
                    None => true,
                };
                self.current = saved;
                if !go {
                    return Ok(StepOut::Continue);
                }
                stack.push(Step::ForLoop {
                    test,
                    update,
                    body,
                    label: label.clone(),
                    ran_body: true,
                    scope: scope.clone(),
                });
                // Run the body in a child of the loop-header scope.
                let child = scope.child();
                let prev = core::mem::replace(&mut self.current, child);
                stack.push(Step::PopScope { scope: prev });
                self.gen_exec_loop_body(body, stack, values, &label)
            }
            Step::ForEach {
                left,
                body,
                values: items,
                idx,
                label,
            } => {
                if idx >= items.len() {
                    return Ok(StepOut::Continue);
                }
                let item = items[idx];
                stack.push(Step::ForEach {
                    left,
                    body,
                    values: items,
                    idx: idx + 1,
                    label: label.clone(),
                });
                // Bind the loop variable in a fresh per-iteration scope.
                let child = self.current.child();
                let prev = core::mem::replace(&mut self.current, child);
                stack.push(Step::PopScope { scope: prev });
                match left {
                    ForLeft::Decl { target, .. } => {
                        self.bind_pattern(target, item).map_err(GenAbrupt::from)?;
                    }
                    ForLeft::Target(expr) => {
                        self.assign_destructure(expr, item)
                            .map_err(GenAbrupt::from)?;
                    }
                }
                self.gen_exec_loop_body(body, stack, values, &label)
            }
            // A `try` region reached with no abrupt completion: run its
            // `finalizer` (if any) on the normal way out, restoring the try scope.
            Step::TryRegion {
                finalizer, scope, ..
            } => {
                self.current = scope.clone();
                if let Some(fin) = finalizer {
                    self.gen_push_seq(stack, fin);
                }
                Ok(StepOut::Continue)
            }
            Step::PopScope { scope } => {
                self.current = scope;
                Ok(StepOut::Continue)
            }
            // A `finally` block finished normally: re-apply the completion it
            // had buffered (a `return`/`throw`/`break`/`continue` in flight).
            Step::Finally { pending } => Err(match pending {
                Completion::Return(v) => GenAbrupt::Return(v),
                Completion::Throw(e) => GenAbrupt::Throw(e),
                Completion::Break(l) => GenAbrupt::Break(l),
                Completion::Continue(l) => GenAbrupt::Continue(l),
            }),
            Step::Discard => {
                values.pop();
                Ok(StepOut::Continue)
            }
            Step::ReturnValue => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                Err(GenAbrupt::Return(v))
            }
            Step::ThrowValue => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                Err(GenAbrupt::Throw(v))
            }
            Step::DeclName { name, is_const } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                if is_const {
                    self.current.declare_const(name, v);
                } else {
                    self.current.declare(name, v);
                }
                Ok(StepOut::Continue)
            }
            Step::VarTail { kind, rest } => self.gen_var_tail(kind, rest, stack, values),
            // Reached normally (no break inside): nothing to do.
            Step::SwitchRegion => Ok(StepOut::Continue),
            Step::EvalThen { expr } => self.gen_eval_expr(expr, stack, values),
            Step::BinaryOp { op } => {
                let right = values.pop().unwrap_or(NanBox::undefined());
                let left = values.pop().unwrap_or(NanBox::undefined());
                let v = self.binary(op, left, right).map_err(GenAbrupt::from)?;
                values.push(v);
                Ok(StepOut::Continue)
            }
            Step::AssignName { name } => {
                let v = *values.last().unwrap_or(&NanBox::undefined());
                self.assign_to_name(name, v).map_err(GenAbrupt::from)?;
                Ok(StepOut::Continue)
            }
            Step::Logical { op, right } => {
                let left = values.pop().unwrap_or(NanBox::undefined());
                let short = match op {
                    LogicalOp::And => !self.realm.truthy(left),
                    LogicalOp::Or => self.realm.truthy(left),
                    LogicalOp::Nullish => {
                        !matches!(left.unpack(), Unpacked::Undefined | Unpacked::Null)
                    }
                };
                if short {
                    values.push(left);
                    Ok(StepOut::Continue)
                } else {
                    self.gen_eval_expr(right, stack, values)
                }
            }
            Step::Conditional {
                consequent,
                alternate,
            } => {
                let test = values.pop().unwrap_or(NanBox::undefined());
                let branch = if self.realm.truthy(test) {
                    consequent
                } else {
                    alternate
                };
                self.gen_eval_expr(branch, stack, values)
            }
            Step::SeqExpr { rest } => self.gen_eval_sequence(rest, stack, values),
            Step::YieldExpr { delegate } => {
                let operand = values.pop().unwrap_or(NanBox::undefined());
                if delegate {
                    // `yield*`: obtain the inner iterator and start pumping with an
                    // initial `next(undefined)`. `gen_yield_star_step` re-pushes a
                    // `YieldStar` step itself on a non-done inner result.
                    let iter = self.gen_delegate_iter(operand).map_err(GenAbrupt::from)?;
                    self.gen_yield_star_step(
                        iter,
                        Resumption::Next(NanBox::undefined()),
                        stack,
                        values,
                    )
                } else {
                    // Plain `yield v`: suspend. On resume, the injected value is
                    // pushed by `gen_drive` and becomes this expression's result.
                    Ok(StepOut::Yield(operand))
                }
            }
            Step::YieldStar { iter } => {
                // A `YieldStar` on top at resume is intercepted by `gen_drive`
                // (which forwards the resumption to the inner iterator); reaching
                // it here means it was not the suspension point, so pump it with
                // `next(undefined)`.
                self.gen_yield_star_step(iter, Resumption::Next(NanBox::undefined()), stack, values)
            }
        }
    }
}

// --- statement lowering ------------------------------------------------------

impl<'a> Interp<'a> {
    /// Lowers one statement into machine steps (if it may yield) or runs it in
    /// one shot via the ordinary walker (if it is yield-free). `label` is the
    /// label of an enclosing labeled statement that targets this loop/block.
    fn gen_exec_stmt(
        &mut self,
        stmt: &'a Stmt,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
        label: &Option<String>,
    ) -> StepResult {
        // A yield-free statement runs in one shot; its abrupt completions become
        // generator-machine completions (so break/continue/return out of it are
        // handled by the unwinder, honoring an enclosing label).
        if !stmt_has_yield(stmt) {
            // A directly-labeled yield-free loop needs its label for inner
            // break/continue; `exec` reads `pending_label`.
            if let Some(l) = label
                && is_loop(stmt)
            {
                self.pending_label = Some(l.clone());
            }
            return match self.exec(stmt) {
                Ok(Flow::Normal(_)) => Ok(StepOut::Continue),
                Ok(Flow::Return(v)) => Err(GenAbrupt::Return(v)),
                Ok(Flow::Break(l)) => {
                    // A labeled break matching this statement's label is consumed.
                    if label.is_some() && l == *label {
                        Ok(StepOut::Continue)
                    } else {
                        Err(GenAbrupt::Break(l))
                    }
                }
                Ok(Flow::Continue(l)) => Err(GenAbrupt::Continue(l)),
                Err(e) => Err(GenAbrupt::from(e)),
            };
        }
        match stmt {
            Stmt::Expr { expression, .. } => {
                stack.push(Step::Discard);
                self.gen_eval_expr(expression, stack, values)
            }
            Stmt::Block { body, .. } => {
                let child = self.current.child();
                let prev = core::mem::replace(&mut self.current, child);
                self.hoist(body).map_err(GenAbrupt::from)?;
                stack.push(Step::Seq {
                    body,
                    idx: 0,
                    scope: Some(prev),
                    label: label.clone(),
                });
                Ok(StepOut::Continue)
            }
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                let t = self.eval(test).map_err(GenAbrupt::from)?;
                if self.realm.truthy(t) {
                    self.gen_exec_stmt(consequent, stack, values, &None)
                } else if let Some(alt) = alternate {
                    self.gen_exec_stmt(alt, stack, values, &None)
                } else {
                    Ok(StepOut::Continue)
                }
            }
            Stmt::While { test, body, .. } => {
                stack.push(Step::While {
                    test,
                    body,
                    label: label.clone(),
                });
                Ok(StepOut::Continue)
            }
            Stmt::DoWhile { body, test, .. } => {
                stack.push(Step::DoWhile {
                    body,
                    test,
                    label: label.clone(),
                    test_first: false,
                });
                Ok(StepOut::Continue)
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => {
                // Run the init in a fresh loop-header scope, then push the loop.
                let child = self.current.child();
                let prev = core::mem::replace(&mut self.current, child);
                if let Some(i) = init {
                    let r = match i {
                        ForInit::Var(d) => self.exec_var(d),
                        ForInit::Expr(e) => self.eval(e).map(|_| ()),
                    };
                    if let Err(e) = r {
                        self.current = prev;
                        return Err(GenAbrupt::from(e));
                    }
                }
                let header = core::mem::replace(&mut self.current, prev);
                stack.push(Step::PopScope {
                    scope: self.current.clone(),
                });
                stack.push(Step::ForLoop {
                    test: test.as_deref(),
                    update: update.as_deref(),
                    body,
                    label: label.clone(),
                    ran_body: false,
                    scope: header,
                });
                Ok(StepOut::Continue)
            }
            Stmt::ForOf {
                left,
                right,
                body,
                is_await,
                ..
            } => {
                let iterable = self.eval(right).map_err(GenAbrupt::from)?;
                // Eager list of values (mirrors the non-await tree-walker for the
                // built-in iterables; user iterators are drained up front here —
                // a documented simplification for yield-bearing for-of bodies).
                let mut items = self.iterate_values(iterable).map_err(GenAbrupt::from)?;
                if *is_await {
                    for v in &mut items {
                        *v = self.await_value(*v).map_err(GenAbrupt::from)?;
                    }
                }
                stack.push(Step::ForEach {
                    left,
                    body,
                    values: items,
                    idx: 0,
                    label: label.clone(),
                });
                Ok(StepOut::Continue)
            }
            Stmt::ForIn {
                left, right, body, ..
            } => {
                let obj = self.eval(right).map_err(GenAbrupt::from)?;
                let keys = if let Some(raw) = obj.as_handle()
                    && let Some(trap_keys) = self
                        .proxy_own_enumerable_keys(Handle::from_raw(raw))
                        .map_err(GenAbrupt::from)?
                {
                    trap_keys.iter().map(|k| self.new_str(k)).collect()
                } else {
                    self.iterate_keys(obj)
                };
                stack.push(Step::ForEach {
                    left,
                    body,
                    values: keys,
                    idx: 0,
                    label: label.clone(),
                });
                Ok(StepOut::Continue)
            }
            Stmt::Switch {
                discriminant,
                cases,
                ..
            } => self.gen_exec_switch(discriminant, cases, stack, values, label),
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                let child = self.current.child();
                let prev = core::mem::replace(&mut self.current, child);
                self.hoist(block).map_err(GenAbrupt::from)?;
                stack.push(Step::TryRegion {
                    handler: handler.as_ref(),
                    finalizer: finalizer.as_deref(),
                    scope: prev.clone(),
                });
                stack.push(Step::Seq {
                    body: block,
                    idx: 0,
                    scope: Some(prev),
                    label: None,
                });
                Ok(StepOut::Continue)
            }
            Stmt::Return { argument, .. } => match argument {
                Some(e) => {
                    stack.push(Step::ReturnValue);
                    self.gen_eval_expr(e, stack, values)
                }
                None => Err(GenAbrupt::Return(NanBox::undefined())),
            },
            Stmt::Throw { argument, .. } => {
                stack.push(Step::ThrowValue);
                self.gen_eval_expr(argument, stack, values)
            }
            Stmt::Var(decl) => self.gen_exec_var(decl, stack, values),
            Stmt::Labeled { label: l, body, .. } => {
                self.gen_exec_stmt(body, stack, values, &Some(String::from(&*l.name)))
            }
            Stmt::With { object, body, .. } => {
                // `with` is sloppy-only and rare in generators; evaluate the
                // object, push the with-object, and run the body (yield-bearing).
                let obj = self.eval(object).map_err(GenAbrupt::from)?;
                let obj = if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Cannot convert undefined or null to object");
                    return Err(GenAbrupt::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                } else if obj.as_handle().is_some() {
                    obj
                } else {
                    self.coerce_to_object(obj)
                };
                self.with_stack.push(obj);
                // A WithPop step restores the with-stack afterwards.
                stack.push(Step::PopScope {
                    scope: self.current.clone(),
                });
                let r = self.gen_exec_stmt(body, stack, values, &None);
                // NB: with-stack popping for the suspended case is a known
                // limitation; pop eagerly on the synchronous path.
                self.with_stack.pop();
                r
            }
            // Yield-bearing forms not otherwise handled fall back to one-shot
            // execution (no reachable yield will actually run).
            _ => match self.exec(stmt) {
                Ok(Flow::Normal(_)) => Ok(StepOut::Continue),
                Ok(Flow::Return(v)) => Err(GenAbrupt::Return(v)),
                Ok(Flow::Break(l)) => Err(GenAbrupt::Break(l)),
                Ok(Flow::Continue(l)) => Err(GenAbrupt::Continue(l)),
                Err(e) => Err(GenAbrupt::from(e)),
            },
        }
    }

    /// Runs a loop body as one machine sub-statement (no label of its own).
    fn gen_exec_loop_body(
        &mut self,
        body: &'a Stmt,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
        _label: &Option<String>,
    ) -> StepResult {
        self.gen_exec_stmt(body, stack, values, &None)
    }

    /// Pushes a fresh-scope statement sequence (used for `finally` blocks).
    fn gen_push_seq(&mut self, stack: &mut Vec<Step<'a>>, body: &'a [Stmt]) {
        let child = self.current.child();
        let prev = core::mem::replace(&mut self.current, child);
        let _ = self.hoist(body);
        stack.push(Step::Seq {
            body,
            idx: 0,
            scope: Some(prev),
            label: None,
        });
    }
}

/// The label of a loop [`Step`] (for matching a labeled `break`/`continue`).
fn loop_label<'a>(step: &'a Step<'_>) -> Option<&'a String> {
    match step {
        Step::While { label, .. }
        | Step::DoWhile { label, .. }
        | Step::ForLoop { label, .. }
        | Step::ForEach { label, .. } => label.as_ref(),
        _ => None,
    }
}

/// Whether a statement is a loop (so a directly-attached label governs its
/// inner `break`/`continue`).
fn is_loop(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::For { .. }
            | Stmt::While { .. }
            | Stmt::DoWhile { .. }
            | Stmt::ForOf { .. }
            | Stmt::ForIn { .. }
    )
}

// --- expression lowering -----------------------------------------------------

impl<'a> Interp<'a> {
    /// Lowers an expression that may contain a `yield` into machine steps that
    /// leave its value on the value stack. A yield-free expression is evaluated
    /// in one shot via the ordinary `eval`.
    fn gen_eval_expr(
        &mut self,
        expr: &'a Expr,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        if !expr_has_yield(expr) {
            let v = self.eval(expr).map_err(GenAbrupt::from)?;
            values.push(v);
            return Ok(StepOut::Continue);
        }
        match expr {
            Expr::Yield {
                argument, delegate, ..
            } => {
                // Evaluate the operand first (it may itself contain a yield).
                stack.push(Step::YieldExpr {
                    delegate: *delegate,
                });
                match argument {
                    Some(e) => self.gen_eval_expr(e, stack, values),
                    None => {
                        values.push(NanBox::undefined());
                        Ok(StepOut::Continue)
                    }
                }
            }
            Expr::Sequence { expressions, .. } => {
                self.gen_eval_sequence(expressions, stack, values)
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                stack.push(Step::Conditional {
                    consequent,
                    alternate,
                });
                self.gen_eval_expr(test, stack, values)
            }
            Expr::Logical {
                op, left, right, ..
            } => {
                stack.push(Step::Logical { op: *op, right });
                self.gen_eval_expr(left, stack, values)
            }
            Expr::Assign {
                op, target, value, ..
            } if matches!(op, crate::ast::AssignOp::Assign)
                && matches!(&**target, Expr::Ident(_)) =>
            {
                // `name = <value-with-yield>`.
                if let Expr::Ident(Ident { name, .. }) = &**target {
                    stack.push(Step::AssignName { name });
                }
                self.gen_eval_expr(value, stack, values)
            }
            // `left <op> right` where a yield hides in an operand. The `#x in obj`
            // brand check (a private left operand) is left to the fallback.
            Expr::Binary {
                op, left, right, ..
            } if !matches!(&**left, Expr::PrivateName(..)) => {
                // Evaluate left, then right, then combine: push the combiner and
                // a step that evaluates `right` after `left` is on the stack.
                stack.push(Step::BinaryOp { op: *op });
                stack.push(Step::EvalThen { expr: right });
                self.gen_eval_expr(left, stack, values)
            }
            // Any other yield-bearing expression shape (e.g. `a + (yield b)`,
            // calls/arrays with a yield argument) is not individually reified;
            // fall back to one-shot eval. The yield-free fast path above means
            // this only runs for genuinely yield-bearing complex operands, which
            // are a documented follow-up.
            _ => {
                let v = self.eval(expr).map_err(GenAbrupt::from)?;
                values.push(v);
                Ok(StepOut::Continue)
            }
        }
    }

    /// Evaluates a comma `Sequence` step-by-step, leaving the last value on the
    /// stack. Every expression but the last is evaluated for its side effects
    /// (its value discarded).
    fn gen_eval_sequence(
        &mut self,
        expressions: &'a [Expr],
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        match expressions {
            [] => {
                values.push(NanBox::undefined());
                Ok(StepOut::Continue)
            }
            [last] => self.gen_eval_expr(last, stack, values),
            [first, rest @ ..] => {
                // Run `first` now (it may yield); its value is discarded, then
                // the tail sequence is evaluated for the final result.
                stack.push(Step::SeqExpr { rest });
                stack.push(Step::Discard);
                self.gen_eval_expr(first, stack, values)
            }
        }
    }
}

// --- unwinding, switch, var, yield* ------------------------------------------

impl<'a> Interp<'a> {
    /// Unwinds the machine stack for an abrupt `completion`, restoring scopes and
    /// routing the completion to a matching `try`/`catch`/`finally`, loop, or
    /// switch. Returns `Ok(None)` when execution resumes (into a catch/finally,
    /// or after a consumed break/continue), `Ok(Some(..))` when the generator
    /// completes, or `Err` on a fatal error.
    fn gen_unwind(
        &mut self,
        stack: &mut Vec<Step<'a>>,
        _values: &mut Vec<NanBox>,
        completion: Completion,
    ) -> Result<Option<Result<GenStep, ExecError>>, ExecError> {
        loop {
            let Some(step) = stack.pop() else {
                // Nothing caught it: the generator completes abnormally.
                return Ok(Some(match completion {
                    Completion::Return(v) => Ok(GenStep::Done(v)),
                    Completion::Throw(e) => Err(ExecError::Throw(e)),
                    // A `break`/`continue` that escaped the body: complete normally.
                    Completion::Break(_) | Completion::Continue(_) => {
                        Ok(GenStep::Done(NanBox::undefined()))
                    }
                }));
            };
            match step {
                Step::PopScope { scope } => self.current = scope,
                Step::Seq { scope, label, .. } => {
                    if let Some(s) = scope {
                        self.current = s;
                    }
                    // A labeled break/continue targeting this block's label is
                    // consumed here (a labeled *block* only matches `break`).
                    if let Completion::Break(Some(l)) = &completion
                        && label.as_ref() == Some(l)
                    {
                        return Ok(None);
                    }
                }
                Step::TryRegion {
                    handler,
                    finalizer,
                    scope,
                } => {
                    self.current = scope.clone();
                    // A throw with a catch clause: run the handler.
                    if let (Completion::Throw(e), Some(catch)) = (&completion, handler) {
                        let thrown = *e;
                        let child = self.current.child();
                        let prev = core::mem::replace(&mut self.current, child);
                        if let Some(target) = &catch.param
                            && let Err(err) = self.bind_pattern(target, thrown)
                        {
                            // Binding the catch param threw: run finally, then
                            // propagate the new throw.
                            self.current = prev.clone();
                            return self.gen_route_finally(
                                stack,
                                finalizer,
                                prev,
                                Completion::Throw(throw_value(err)),
                            );
                        }
                        if let Err(e) = self.hoist(&catch.body) {
                            self.current = prev.clone();
                            return self.gen_route_finally(
                                stack,
                                finalizer,
                                prev,
                                Completion::Throw(throw_value(e)),
                            );
                        }
                        // After the catch body, run finally (if any) on the way out.
                        if let Some(fin) = finalizer {
                            stack.push(Step::TryRegion {
                                handler: None,
                                finalizer: Some(fin),
                                scope: prev.clone(),
                            });
                        } else {
                            stack.push(Step::PopScope {
                                scope: prev.clone(),
                            });
                        }
                        stack.push(Step::Seq {
                            body: &catch.body,
                            idx: 0,
                            scope: Some(prev),
                            label: None,
                        });
                        return Ok(None);
                    }
                    // No catch (or non-throw completion): run finally with the
                    // completion buffered, then re-apply it.
                    return self.gen_route_finally(stack, finalizer, scope, completion);
                }
                Step::While { .. }
                | Step::DoWhile { .. }
                | Step::ForLoop { .. }
                | Step::ForEach { .. } => {
                    let label = loop_label(&step);
                    match &completion {
                        // The innermost loop consumes an unlabeled break, or a
                        // labeled break that targets this loop's label.
                        Completion::Break(None) => return Ok(None),
                        Completion::Break(Some(l)) if label == Some(l) => return Ok(None),
                        // A matching continue resumes the loop (re-push its
                        // advanced state).
                        Completion::Continue(None) => {
                            stack.push(step);
                            return Ok(None);
                        }
                        Completion::Continue(Some(l)) if label == Some(l) => {
                            stack.push(step);
                            return Ok(None);
                        }
                        // Other completions (or labels not ours) keep unwinding.
                        _ => {}
                    }
                }
                Step::SwitchRegion => {
                    if matches!(completion, Completion::Break(None)) {
                        return Ok(None);
                    }
                }
                // Expression/other steps are discarded; drop any partial operand
                // values they would have consumed is unnecessary (the value stack
                // is only meaningful between sub-steps of one expression, which an
                // abrupt completion abandons).
                _ => {}
            }
        }
    }

    /// Pushes a `finally` block (if present) to run with `completion` buffered;
    /// when it finishes normally the completion is re-applied. With no finally,
    /// the completion is applied immediately. Always returns `Ok(None)` to
    /// resume stepping (the finally runs) or applies the completion directly.
    fn gen_route_finally(
        &mut self,
        stack: &mut Vec<Step<'a>>,
        finalizer: Option<&'a [Stmt]>,
        scope: Scope,
        completion: Completion,
    ) -> Result<Option<Result<GenStep, ExecError>>, ExecError> {
        self.current = scope;
        if let Some(fin) = finalizer {
            stack.push(Step::Finally {
                pending: completion,
            });
            self.gen_push_seq(stack, fin);
            Ok(None)
        } else {
            // Re-raise immediately by continuing to unwind.
            self.gen_unwind(stack, &mut Vec::new(), completion)
        }
    }

    /// Lowers a `switch` (which may contain a yield) into machine steps.
    fn gen_exec_switch(
        &mut self,
        discriminant: &'a Expr,
        cases: &'a [SwitchCase],
        stack: &mut Vec<Step<'a>>,
        _values: &mut Vec<NanBox>,
        _label: &Option<String>,
    ) -> StepResult {
        let value = self.eval(discriminant).map_err(GenAbrupt::from)?;
        let mut start = None;
        for (i, case) in cases.iter().enumerate() {
            if let Some(test) = &case.test {
                let t = self.eval(test).map_err(GenAbrupt::from)?;
                if self.realm.strict_equals(value, t) {
                    start = Some(i);
                    break;
                }
            }
        }
        if start.is_none() {
            start = cases.iter().position(|c| c.test.is_none());
        }
        let Some(start) = start else {
            return Ok(StepOut::Continue);
        };
        // Run the matched clause and fall through; a plain `break` ends the
        // switch (consumed by the SwitchRegion marker). Enter a fresh scope.
        let child = self.current.child();
        let prev = core::mem::replace(&mut self.current, child);
        for case in &cases[start..] {
            self.hoist(&case.body).map_err(GenAbrupt::from)?;
        }
        // Push, in reverse, so case[start] body runs first.
        stack.push(Step::PopScope { scope: prev });
        stack.push(Step::SwitchRegion);
        for case in cases[start..].iter().rev() {
            stack.push(Step::Seq {
                body: &case.body,
                idx: 0,
                scope: None,
                label: None,
            });
        }
        Ok(StepOut::Continue)
    }

    /// Lowers a `var`/`let`/`const` declaration that contains a yield in some
    /// initializer, processing one declarator per step via a [`Step::VarTail`].
    fn gen_exec_var(
        &mut self,
        decl: &'a crate::ast::VarDecl,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        self.gen_var_tail(decl.kind, &decl.declarations, stack, values)
    }

    /// Processes the next declarator of a `var`/`let`/`const`: a simple
    /// `ident = <yield-bearing>` is stepped (eval pushed, then `DeclName`); any
    /// other declarator is bound in one shot. The remaining declarators are
    /// re-queued via [`Step::VarTail`].
    fn gen_var_tail(
        &mut self,
        kind: VarDeclKind,
        decls: &'a [crate::ast::VarDeclarator],
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        let Some((d, rest)) = decls.split_first() else {
            return Ok(StepOut::Continue);
        };
        if !rest.is_empty() {
            stack.push(Step::VarTail { kind, rest });
        }
        match (&d.target, &d.init) {
            (BindingTarget::Ident(Ident { name, .. }), Some(init)) if expr_has_yield(init) => {
                let is_const = matches!(kind, VarDeclKind::Const);
                stack.push(Step::DeclName { name, is_const });
                self.gen_eval_expr(init, stack, values)
            }
            _ => {
                self.exec_single_declarator(kind, d)
                    .map_err(GenAbrupt::from)?;
                Ok(StepOut::Continue)
            }
        }
    }
}

/// Extracts the thrown value from an `ExecError::Throw`, or re-wraps a fatal
/// error's display (fatals shouldn't reach here in practice).
fn throw_value(e: ExecError) -> NanBox {
    match e {
        ExecError::Throw(v) => v,
        _ => NanBox::undefined(),
    }
}

// --- yield* delegation -------------------------------------------------------

impl<'a> Interp<'a> {
    /// Obtains the inner iterator object for `yield* operand`
    /// (`GetIterator(operand, sync)`).
    fn gen_delegate_iter(&mut self, operand: NanBox) -> Result<Handle, ExecError> {
        let Some(h) = operand.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("yield* operand is not iterable"));
        };
        // A user object exposing `[Symbol.iterator]` is driven through its real
        // iterator (so an infinite/user generator delegates lazily).
        if let Some(f) = self.find_iterator_fn(h)?
            && f.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            let iterator = self.call_with_this(f, operand, &[])?;
            return iterator
                .as_handle()
                .map(Handle::from_raw)
                .ok_or_else(|| self.type_error("yield* iterator is not an object"));
        }
        // A built-in iterable (array / string / Map / Set) whose `Symbol.iterator`
        // is built-in dispatch (not a readable property): materialize an eager
        // iterator over its values — `yield*` still drives it one `.next()` at a
        // time, preserving the surfaced-value sequence.
        let vals = self.iterate_values(operand)?;
        let iter = self.make_generator(vals);
        iter.as_handle()
            .map(Handle::from_raw)
            .ok_or_else(|| self.type_error("yield* operand is not iterable"))
    }

    /// Advances a `yield*` delegation one step: calls the inner iterator with
    /// `how` (a `next(v)` resume, a forwarded `return(v)`, or a forwarded
    /// `throw(e)`). On a non-done inner result, re-pushes the [`Step::YieldStar`]
    /// and surfaces the inner value (the outer generator yields it). On a done
    /// result, the `yield*` expression evaluates to the inner return value.
    fn gen_yield_star_step(
        &mut self,
        iter: Handle,
        how: Resumption,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        let iter_val = NanBox::handle(iter.to_raw());
        let (method_name, arg) = match how {
            Resumption::Next(v) => ("next", v),
            Resumption::Return(v) => ("return", v),
            Resumption::Throw(e) => ("throw", e),
        };
        // Resolve the method; an absent `return`/`throw` has special handling.
        let method = self
            .read_member(iter, method_name)
            .map_err(GenAbrupt::from)?;
        let absent = method.is_undefined() || method.is_null();
        if absent {
            match how {
                // No inner `return`: the delegation completes, the `yield*` value
                // is the forwarded value, and the outer `return` continues.
                Resumption::Return(v) => return Err(GenAbrupt::Return(v)),
                // No inner `throw`: per spec, close the iterator and throw a
                // TypeError into the outer generator.
                Resumption::Throw(_) => {
                    let _ = self.iterator_close(iter);
                    return Err(GenAbrupt::Throw(
                        self.make_type_error("The iterator does not provide a 'throw' method"),
                    ));
                }
                Resumption::Next(_) => {
                    // `next` must exist; treat as done.
                    values.push(NanBox::undefined());
                    return Ok(StepOut::Continue);
                }
            }
        }
        let res = self
            .call_with_this(method, iter_val, &[arg])
            .map_err(GenAbrupt::from)?;
        let Some(rh) = res.as_handle().map(Handle::from_raw) else {
            return Err(GenAbrupt::Throw(
                self.make_type_error("iterator result is not an object"),
            ));
        };
        let done = self.read_member(rh, "done").map_err(GenAbrupt::from)?;
        let value = self.read_member(rh, "value").map_err(GenAbrupt::from)?;
        if self.realm.truthy(done) {
            // Delegation finished. For a forwarded `return`, the outer generator
            // returns the inner return value; otherwise the `yield*` expression
            // evaluates to it.
            if matches!(how, Resumption::Return(_)) {
                return Err(GenAbrupt::Return(value));
            }
            values.push(value);
            Ok(StepOut::Continue)
        } else {
            // Surface the inner value and stay in delegation.
            stack.push(Step::YieldStar { iter });
            Ok(StepOut::Yield(value))
        }
    }

    fn make_type_error(&mut self, msg: &str) -> NanBox {
        let m = self.new_str(msg);
        self.make_error(N_TYPE_ERROR, Some(m))
    }
}
