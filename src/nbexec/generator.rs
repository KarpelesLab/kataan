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
    Argument, ArrayElement, BindingTarget, CatchClause, Expr, ForInit, ForLeft, Ident, LogicalOp,
    ObjectMember, PropertyKey, Stmt, SwitchCase, VarDeclKind,
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
    /// (Async only) parked at `await`; the operand is the awaited value, on whose
    /// settlement the coroutine is resumed by a microtask.
    Awaited(NanBox),
}

/// A suspended generator activation.
pub(crate) struct GenFrame<'a> {
    body: &'a [Stmt],
    /// For a concise-body async arrow (`async () => expr`), the single expression
    /// to evaluate and return; `body` is then empty. `None` for a block body.
    concise: Option<&'a Expr>,
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
    /// True for an `async function*` generator object. Its `next`/`return`/`throw`
    /// each return a *promise* (resolved with the `{value, done}` result, or
    /// rejected with a thrown value) rather than the bare result object.
    is_async: bool,
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
    /// Drive a `for-of`/`for-in` over a precomputed value list. When `await_each`
    /// is set (a `for await` loop), each value is `await`ed (suspending the async
    /// coroutine) before it is bound and the body runs.
    ForEach {
        left: &'a ForLeft,
        body: &'a Stmt,
        values: Vec<NanBox>,
        idx: usize,
        label: Option<String>,
        await_each: bool,
    },
    /// (`for await` only) Bind the awaited value on the stack to the loop target
    /// in a fresh per-iteration scope, then run the loop body.
    ForEachBind {
        left: &'a ForLeft,
        body: &'a Stmt,
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
    /// An `await` whose operand value is on the value stack: suspend the async
    /// coroutine on the operand's settlement (the resumed value is pushed back
    /// as this expression's result).
    AwaitExpr,
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
    /// Build an array literal: `elements[idx..]` remain to evaluate; `acc` holds
    /// the values gathered so far. On reaching the end a new array is pushed.
    ArrayLit {
        elements: &'a [ArrayElement],
        idx: usize,
        acc: Vec<NanBox>,
    },
    /// Append the top value to an array-literal accumulator (spreading it when the
    /// element was `...spread`), then continue with the next element.
    ArrayLitAppend {
        elements: &'a [ArrayElement],
        idx: usize,
        acc: Vec<NanBox>,
        spread: bool,
    },
    /// Build an object literal step-by-step onto `target`: `members[idx..]` remain.
    /// Only entered for objects whose members are all data properties (static key,
    /// non-function value) or spreads — see `object_lit_steppable`.
    ObjectLit {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
    },
    /// Set the just-evaluated value (top of stack) as own property `key` of
    /// `target`, then continue with the next object-literal member.
    ObjectLitSet {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
        key: String,
    },
    /// Spread the just-evaluated value (top of stack) into `target`
    /// (CopyDataProperties), then continue with the next member.
    ObjectLitSpread {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
    },
    /// Destructure `value` into assignment `target` (an array/object pattern, a
    /// defaulted target, or a leaf) — a `yield` in a default initializer suspends.
    /// The iterator pull / property reads themselves are synchronous.
    Destructure {
        target: &'a Expr,
        value: NanBox,
    },
    /// Destructure `elements[idx..]` of an array pattern from the pre-iterated
    /// `items`; `i` is the source index consumed so far (holes advance it).
    DestructureArrayElem {
        elements: &'a [ArrayElement],
        idx: usize,
        i: usize,
        items: Vec<NanBox>,
    },
    /// Destructure `members[idx..]` of an object pattern from source object `src`;
    /// `used` records keys already consumed (for a `...rest`).
    DestructureObjectMember {
        members: &'a [ObjectMember],
        idx: usize,
        src: Handle,
        used: Vec<String>,
    },
    /// A defaulted destructuring target whose default value (just evaluated, on the
    /// value stack) replaces an `undefined` source; then destructure `inner`.
    DestructureDefault { inner: &'a Expr },
    /// Top of a destructuring assignment: the RHS value is on the value stack (and
    /// stays there as the expression's result); begin destructuring it into
    /// `target` without disturbing that result.
    DestructureStart { target: &'a Expr },
    /// A member destructuring leaf `obj[key] = value` whose computed `key` may
    /// yield: the base object is on the value stack; evaluate `key`, then assign.
    DestructureMemberKey { key: &'a Expr, value: NanBox },
    /// Complete `obj[key] = value`: the base object and (on top) the key are on the
    /// value stack; assign `value` through them.
    DestructureMemberSet { value: NanBox },
    /// Run a `for-of`/`for-in` loop body after its (possibly yield-suspending)
    /// per-iteration target binding has completed.
    RunLoopBody {
        body: &'a Stmt,
        label: Option<String>,
    },
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
    /// An async coroutine reached `await <value>`: the operand is the awaited
    /// value (a promise or plain value). The driver parks the frame and schedules
    /// a microtask resumption on its settlement.
    Await(NanBox),
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
    pub(crate) fn make_lazy_generator(
        &mut self,
        body: &'a [Stmt],
        scope: Scope,
        is_async: bool,
    ) -> NanBox {
        let frame = GenFrame {
            body,
            concise: None,
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
            is_async,
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

    /// Whether `h` is a lazy *async* generator (`async function*`): `Some(true)`,
    /// a sync generator: `Some(false)`, or not a lazy generator at all: `None`.
    /// A `for await` loop uses this to drive an async-generator iterable through
    /// the async-iterator protocol (awaiting each `next()` promise).
    pub(crate) fn lazy_gen_is_async(&self, h: Handle) -> Option<bool> {
        let id = self.gen_frame_id(h)?;
        self.gen_frames[id].as_ref().map(|f| f.is_async)
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
        let is_async = self.gen_frames[id].as_ref().is_some_and(|f| f.is_async);
        // A plain `function*` body never awaits. An *async generator*
        // (`async function*`) runs on this same lazy-generator engine (a
        // documented simplification of the previous eager-async model): its
        // `await` is resolved *eagerly* here — drive the event loop until the
        // awaited value settles and resume in place — preserving the prior async
        // generator behavior. (True per-`await` microtask suspension for async
        // generators is a follow-up; this restores the pre-coroutine semantics so
        // async generators are not regressed by the async-function rework.)
        let mut how = how;
        let result: Result<NanBox, ExecError> = loop {
            match self.run_generator(id, how) {
                Ok(GenStep::Yielded(v)) => break Ok(self.gen_result(v, false)),
                Ok(GenStep::Done(v)) => break Ok(self.gen_result(v, true)),
                Ok(GenStep::Awaited(v)) => {
                    match self.await_value(v) {
                        Ok(resolved) => how = Resumption::Next(resolved),
                        Err(ExecError::Throw(e)) => how = Resumption::Throw(e),
                        Err(other) => break Err(other),
                    }
                    // Loop: resume the generator at the await point with the
                    // settled value (or throw).
                }
                Err(e) => break Err(e),
            }
        };
        // A sync generator returns the bare `{value, done}` result (or propagates
        // an escaping throw). An *async generator* method instead always returns a
        // promise: fulfilled with the result object, or rejected with the thrown
        // value — never throwing synchronously.
        if is_async {
            let p = self.fresh_promise();
            match result {
                Ok(v) => self.settle(p, v, true),
                Err(ExecError::Throw(e)) => self.settle(p, e, false),
                Err(other) => return Err(other),
            }
            return Ok(NanBox::handle(p.to_raw()));
        }
        result
    }

    fn gen_result(&mut self, value: NanBox, done: bool) -> NanBox {
        let r = self.realm.new_object();
        self.realm.set_property(r, "value", value);
        self.realm.set_property(r, "done", NanBox::boolean(done));
        NanBox::handle(r.to_raw())
    }

    // --- async-function coroutines ------------------------------------------

    /// Builds a suspended **async-function** coroutine over `body` (the param-bound
    /// `scope`) and returns the controller object's [`GenFrame`] id together with
    /// the promise the async call resolves. The controller object is internal (not
    /// exposed to JS); it carries the frame id and the result-promise handle so the
    /// microtask resume reactions can find both.
    pub(crate) fn make_async_frame(
        &mut self,
        body: Body<'a>,
        scope: Scope,
    ) -> (usize, Handle, Handle) {
        let (body, concise) = match body {
            Body::Block(stmts) => (stmts, None),
            Body::Expr(e) => (&[][..], Some(e)),
        };
        let frame = GenFrame {
            body,
            concise,
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
            is_async: false,
        };
        let id = if let Some(slot) = self.gen_frames.iter().position(Option::is_none) {
            self.gen_frames[slot] = Some(frame);
            slot
        } else {
            self.gen_frames.push(Some(frame));
            self.gen_frames.len() - 1
        };
        let promise = self.fresh_promise();
        let controller = self.realm.new_object();
        self.realm
            .set_hidden_property(controller, GEN_FRAME, NanBox::number(id as f64));
        self.realm
            .set_hidden_property(controller, ASYNC_PROMISE, NanBox::handle(promise.to_raw()));
        (id, promise, controller)
    }

    /// Advances an async coroutine one step with `how` (initial `Next(undefined)`
    /// at call time, or a microtask resume after an awaited promise settles).
    /// Settles the coroutine's result promise on completion, or parks it on the
    /// next awaited value (scheduling the resume reactions).
    pub(crate) fn async_step(&mut self, id: usize, controller: Handle, how: Resumption) {
        let promise = self
            .realm
            .get_property(controller, ASYNC_PROMISE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        match self.run_generator(id, how) {
            Ok(GenStep::Done(v)) => {
                if let Some(p) = promise {
                    self.resolve_with(p, v);
                }
            }
            Ok(GenStep::Awaited(v)) => {
                // `await v`: PromiseResolve(v), then schedule the coroutine's
                // resume as microtask reactions on its settlement.
                let inner = self.promise_resolve(v);
                let on_f = self
                    .realm
                    .new_bound_native(N_ASYNC_RESUME_FULFILL, controller);
                let on_r = self
                    .realm
                    .new_bound_native(N_ASYNC_RESUME_REJECT, controller);
                self.register_then(
                    inner,
                    NanBox::handle(on_f.to_raw()),
                    NanBox::handle(on_r.to_raw()),
                    false,
                );
            }
            // A generator suspension cannot arise in an async (non-generator) body.
            Ok(GenStep::Yielded(_)) => {
                if let Some(p) = promise {
                    self.resolve_with(p, NanBox::undefined());
                }
            }
            Err(ExecError::Throw(e)) => {
                if let Some(p) = promise {
                    self.settle(p, e, false);
                }
            }
            // A non-throw fatal (stack overflow, resource limit): reject with it as
            // best-effort so the loop does not silently swallow it. There is no
            // surrounding `Result` here (we run inside a microtask), so surface it
            // through the promise.
            Err(other) => {
                if let Some(p) = promise {
                    let msg = self.new_str(&alloc::format!("{other:?}"));
                    let err = self.make_error(N_TYPE_ERROR, Some(msg));
                    self.settle(p, err, false);
                }
            }
        }
    }

    /// Recovers the async coroutine `(frame id)` for a controller object handle
    /// (the `this`/target of a resume reaction).
    pub(crate) fn async_frame_id(&self, controller: Handle) -> Option<usize> {
        self.realm
            .get_property(controller, GEN_FRAME)
            .and_then(|v| v.as_number())
            .map(|n| n as usize)
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
        // A coroutine body's lexical class (private-name scope) is its home class —
        // mirror it so `#x` inside a generator/async method body resolves.
        let saved_lexical_home = core::mem::replace(&mut self.current_lexical_home, home_class);
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
        self.current_lexical_home = saved_lexical_home;
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
            // `Yielded` (generator suspension) and `Awaited` (async suspension)
            // both keep the frame alive for a later resume.
            Ok(GenStep::Yielded(_)) | Ok(GenStep::Awaited(_)) => {}
        }
        outcome
    }

    fn gen_drive(
        &mut self,
        id: usize,
        how: Resumption,
        started: bool,
    ) -> Result<GenStep, ExecError> {
        // Track whether the coroutine being driven is an async generator, so
        // `yield*` delegation uses the async-iterator protocol. Saved/restored
        // because the event loop (a microtask drain inside an `await`) can resume a
        // different coroutine reentrantly.
        let saved_gen_async = self.gen_is_async;
        self.gen_is_async = self.gen_frames[id].as_ref().is_some_and(|f| f.is_async);
        let result = self.gen_drive_inner(id, how, started);
        self.gen_is_async = saved_gen_async;
        result
    }

    fn gen_drive_inner(
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
            let (body, concise) = {
                let f = self.gen_frames[id].as_ref().expect("frame present");
                (f.body, f.concise)
            };
            // The body's TOP LEVEL is a *function* scope, not a block: hoist with
            // function-scope semantics (`var`s + top-level function declarations
            // bind here, not into an enclosing/global scope). Using the block-level
            // `hoist` here let a body-level `function X(){}` whose name also exists
            // in an outer scope (e.g. a built-in like `TypeError`) clobber that
            // outer binding via the Annex-B `set`-walks-parents path — corrupting
            // globals. (Pre-existing for `function*`; now also covers async bodies,
            // which run on this same coroutine engine.)
            if let Err(e) = self.hoist_with(body, true) {
                self.store_machine(id, stack, values);
                return Err(e);
            }
            match concise {
                // A concise-body async arrow: evaluate the single expression and
                // return it (the `EvalThen` lowers any `await` to a suspension).
                Some(expr) => {
                    stack.push(Step::ReturnValue);
                    stack.push(Step::EvalThen { expr });
                }
                None => stack.push(Step::Seq {
                    body,
                    idx: 0,
                    scope: None,
                    label: None,
                }),
            }
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
                Ok(StepOut::Await(v)) => {
                    self.store_machine(id, stack, values);
                    return Ok(GenStep::Awaited(v));
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
                Ok(StepOut::Await(v)) => break Ok(GenStep::Awaited(v)),
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

/// Whether an async function `body` contains a reachable `await` (or `for await`)
/// at this function level (not inside a nested function/class). An async function
/// whose body never suspends is run synchronously and its result wrapped in a
/// settled promise (avoiding the coroutine machine's lowering for forms like
/// `with` that only the suspending path needs to reify); one that may suspend is
/// driven as a coroutine. (`expr_has_yield`/`stmt_has_yield` treat `await` and
/// `for await` as suspension points, so they double as await detectors here.)
pub(crate) fn body_has_await(body: &Body<'_>) -> bool {
    match body {
        Body::Block(stmts) => stmts.iter().any(stmt_has_yield),
        Body::Expr(e) => expr_has_yield(e),
    }
}

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
        // A `for await` loop is itself a suspension point of the async coroutine
        // (each iterated value is `await`ed), so it must be lowered into the
        // machine even when its operand and body are otherwise suspension-free.
        Stmt::ForOf {
            right,
            body,
            is_await: true,
            ..
        } => {
            let _ = (right, body);
            true
        }
        Stmt::ForIn {
            left, right, body, ..
        }
        | Stmt::ForOf {
            left, right, body, ..
        } => {
            // A yield can also hide in an assignment-target pattern's default/key
            // (`for ([ x = yield ] of …)`), which is bound per iteration.
            matches!(left, ForLeft::Target(e) if expr_has_yield(e))
                || expr_has_yield(right)
                || stmt_has_yield(body)
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
        // `await` is itself a suspension point of the *async* coroutine machine
        // (the same explicit-stack engine drives async functions). It can only
        // appear inside an async function, so treating it as a suspension point
        // unconditionally is correct: a plain `function*` body never contains a
        // top-level `await` (a nested async arrow's `await` is past a function
        // boundary, which this walker already stops at).
        Expr::Await { .. } => true,
        Expr::OptChain { expr, .. } => expr_has_yield(expr),
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

/// Whether an object literal can be built by the generator step-machine: every
/// member must be a spread or a plain data property with a *static* key and a
/// *non-function* value (and not the `__proto__:` prototype setter). Methods,
/// accessors, computed keys, function-valued props, and `__proto__` need
/// name/home/proto handling — and never reach a `yield` in their own value — so
/// such objects defer to the one-shot `eval` fast path instead.
fn object_lit_steppable(members: &[ObjectMember]) -> bool {
    members.iter().all(|m| match m {
        ObjectMember::Spread { .. } => true,
        ObjectMember::Property {
            key,
            value,
            shorthand,
            ..
        } => {
            let static_key = !matches!(key, PropertyKey::Computed(_));
            let fn_valued = matches!(
                &**value,
                Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_)
            );
            let proto_setter = !shorthand
                && matches!(key, PropertyKey::Ident(s) if &**s == "__proto__");
            static_key && !fn_valued && !proto_setter
        }
        ObjectMember::Accessor { .. } => false,
    })
}

// --- the stepping machine ----------------------------------------------------

impl<'a> Interp<'a> {
    /// Executes one step of the generator machine: pops the top [`Step`] and
    /// processes it, pushing follow-up steps and/or leaving values on the value
    /// stack. Returns `Yield(v)` when a `yield` is reached.
    /// Disposes the `using` / `await using` resources recorded in `self.current`
    /// (a coroutine block / body scope leaving *normally*), in reverse order. A
    /// scope with no disposers does nothing (the fast path). A throwing disposer
    /// is surfaced as a `GenAbrupt::Throw`. An `await using` disposer's result is
    /// awaited eagerly (via `await_value`) — the documented eager-async-dispose
    /// model for the coroutine path.
    fn gen_dispose_scope_resources(&mut self) -> Result<(), GenAbrupt> {
        if !self.current.has_disposers() {
            return Ok(());
        }
        let disposers = self.current.take_disposers();
        match self.dispose_resources(disposers, Ok(NanBox::undefined())) {
            Ok(_) => Ok(()),
            Err(e) => Err(GenAbrupt::from(e)),
        }
    }

    /// Disposes the `using` resources of `self.current` while *unwinding* an
    /// abrupt `completion`, returning the (possibly updated) completion. A
    /// throwing disposer is aggregated into the completion: it suppresses a
    /// pending throw (SuppressedError chain) or, against a non-throw completion
    /// (`return`/`break`/`continue`), replaces it with the disposer's throw. A
    /// scope with no disposers returns the completion unchanged (fast path).
    fn gen_dispose_unwind(&mut self, completion: Completion) -> Completion {
        if !self.current.has_disposers() {
            return completion;
        }
        let disposers = self.current.take_disposers();
        // Map the in-flight completion to a value-completion the disposal driver
        // can thread: only a `Throw` participates in suppression; a
        // `return`/`break`/`continue` is preserved unless a disposer throws.
        let threaded = match &completion {
            Completion::Throw(e) => Err(ExecError::Throw(*e)),
            _ => Ok(NanBox::undefined()),
        };
        match self.dispose_resources(disposers, threaded) {
            // Disposal did not throw: keep the original completion.
            Ok(_) => completion,
            // A disposer threw: it becomes the (possibly aggregated) completion.
            Err(e) => Completion::Throw(throw_value(e)),
        }
    }

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
                    // Block / body sequence finished normally: dispose any
                    // `using` resources recorded in the just-completed scope
                    // (`self.current`), in reverse order, before restoring the
                    // enclosing scope. A throwing disposer becomes a throw.
                    self.gen_dispose_scope_resources()?;
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
                await_each,
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
                    await_each,
                });
                if await_each {
                    // `for await`: await the item (a real coroutine suspension),
                    // then bind the resolved value and run the body.
                    stack.push(Step::ForEachBind { left, body, label });
                    stack.push(Step::AwaitExpr);
                    values.push(item);
                    return Ok(StepOut::Continue);
                }
                // Bind the loop variable in a fresh per-iteration scope.
                let child = self.current.child();
                let prev = core::mem::replace(&mut self.current, child);
                stack.push(Step::PopScope { scope: prev });
                // `for ([ x = yield ] of …)` — an assignment-target pattern whose
                // default/key yields: destructure through the step-machine (so the
                // yield suspends), then run the body via a deferred step.
                if let ForLeft::Target(expr) = left
                    && expr_has_yield(expr)
                {
                    stack.push(Step::RunLoopBody { body, label });
                    stack.push(Step::Destructure {
                        target: expr,
                        value: item,
                    });
                    return Ok(StepOut::Continue);
                }
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
            Step::ForEachBind { left, body, label } => {
                // The awaited value is on the operand stack; bind it in a fresh
                // per-iteration scope, then run the loop body.
                let item = values.pop().unwrap_or(NanBox::undefined());
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
            Step::ArrayLit {
                elements,
                idx,
                acc,
            } => {
                if idx >= elements.len() {
                    let h = self.realm.new_array(acc);
                    values.push(NanBox::handle(h.to_raw()));
                    return Ok(StepOut::Continue);
                }
                match &elements[idx] {
                    ArrayElement::Hole => {
                        let mut acc = acc;
                        acc.push(NanBox::hole());
                        stack.push(Step::ArrayLit {
                            elements,
                            idx: idx + 1,
                            acc,
                        });
                        Ok(StepOut::Continue)
                    }
                    ArrayElement::Item(e) => {
                        stack.push(Step::ArrayLitAppend {
                            elements,
                            idx,
                            acc,
                            spread: false,
                        });
                        self.gen_eval_expr(e, stack, values)
                    }
                    ArrayElement::Spread(e) => {
                        stack.push(Step::ArrayLitAppend {
                            elements,
                            idx,
                            acc,
                            spread: true,
                        });
                        self.gen_eval_expr(e, stack, values)
                    }
                }
            }
            Step::ArrayLitAppend {
                elements,
                idx,
                mut acc,
                spread,
            } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                if spread {
                    let items = self.iterate_values(v).map_err(GenAbrupt::from)?;
                    acc.extend(items);
                } else {
                    acc.push(v);
                }
                stack.push(Step::ArrayLit {
                    elements,
                    idx: idx + 1,
                    acc,
                });
                Ok(StepOut::Continue)
            }
            Step::ObjectLit {
                members,
                idx,
                target,
            } => {
                if idx >= members.len() {
                    values.push(NanBox::handle(target.to_raw()));
                    return Ok(StepOut::Continue);
                }
                match &members[idx] {
                    ObjectMember::Property { key, value, .. } => {
                        // `object_lit_steppable` guarantees a static key here.
                        let k = self.eval_prop_key(key).map_err(GenAbrupt::from)?;
                        stack.push(Step::ObjectLitSet {
                            members,
                            idx,
                            target,
                            key: k,
                        });
                        self.gen_eval_expr(value, stack, values)
                    }
                    ObjectMember::Spread { value, .. } => {
                        stack.push(Step::ObjectLitSpread {
                            members,
                            idx,
                            target,
                        });
                        self.gen_eval_expr(value, stack, values)
                    }
                    // Excluded by `object_lit_steppable`.
                    ObjectMember::Accessor { .. } => unreachable!(),
                }
            }
            Step::ObjectLitSet {
                members,
                idx,
                target,
                key,
            } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                self.realm.set_property(target, &key, v);
                stack.push(Step::ObjectLit {
                    members,
                    idx: idx + 1,
                    target,
                });
                Ok(StepOut::Continue)
            }
            Step::ObjectLitSpread {
                members,
                idx,
                target,
            } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                self.object_spread_into(target, v).map_err(GenAbrupt::from)?;
                stack.push(Step::ObjectLit {
                    members,
                    idx: idx + 1,
                    target,
                });
                Ok(StepOut::Continue)
            }
            Step::DestructureStart { target } => {
                // The RHS result is on top of the value stack; leave it there and
                // destructure a copy into `target`.
                let rval = *values.last().unwrap_or(&NanBox::undefined());
                stack.push(Step::Destructure {
                    target,
                    value: rval,
                });
                Ok(StepOut::Continue)
            }
            Step::Destructure { target, value } => {
                self.gen_destructure(target, value, stack, values)
            }
            Step::DestructureArrayElem {
                elements,
                idx,
                i,
                items,
            } => {
                if idx >= elements.len() {
                    return Ok(StepOut::Continue);
                }
                match &elements[idx] {
                    ArrayElement::Hole => {
                        stack.push(Step::DestructureArrayElem {
                            elements,
                            idx: idx + 1,
                            i: i + 1,
                            items,
                        });
                        Ok(StepOut::Continue)
                    }
                    ArrayElement::Item(e) => {
                        let v = items.get(i).copied().unwrap_or(NanBox::undefined());
                        stack.push(Step::DestructureArrayElem {
                            elements,
                            idx: idx + 1,
                            i: i + 1,
                            items,
                        });
                        stack.push(Step::Destructure { target: e, value: v });
                        Ok(StepOut::Continue)
                    }
                    ArrayElement::Spread(e) => {
                        let rest = items[i.min(items.len())..].to_vec();
                        let h = NanBox::handle(self.realm.new_array(rest).to_raw());
                        stack.push(Step::DestructureArrayElem {
                            elements,
                            idx: idx + 1,
                            i: items.len(),
                            items,
                        });
                        stack.push(Step::Destructure { target: e, value: h });
                        Ok(StepOut::Continue)
                    }
                }
            }
            Step::DestructureObjectMember {
                members,
                idx,
                src,
                mut used,
            } => {
                if idx >= members.len() {
                    return Ok(StepOut::Continue);
                }
                match &members[idx] {
                    ObjectMember::Property {
                        key, value: tgt, ..
                    } => {
                        let k = self.eval_prop_key(key).map_err(GenAbrupt::from)?;
                        let v = self.read_member(src, &k).map_err(GenAbrupt::from)?;
                        used.push(k);
                        stack.push(Step::DestructureObjectMember {
                            members,
                            idx: idx + 1,
                            src,
                            used,
                        });
                        stack.push(Step::Destructure { target: tgt, value: v });
                        Ok(StepOut::Continue)
                    }
                    ObjectMember::Spread { value: tgt, .. } => {
                        let obj = self.realm.new_object();
                        for k in self.realm.object_keys(src).unwrap_or_default() {
                            if !used.contains(&k) {
                                let pv = self.read_member(src, &k).map_err(GenAbrupt::from)?;
                                self.realm.set_property(obj, &k, pv);
                            }
                        }
                        let h = NanBox::handle(obj.to_raw());
                        stack.push(Step::DestructureObjectMember {
                            members,
                            idx: idx + 1,
                            src,
                            used,
                        });
                        stack.push(Step::Destructure { target: tgt, value: h });
                        Ok(StepOut::Continue)
                    }
                    ObjectMember::Accessor { .. } => {
                        stack.push(Step::DestructureObjectMember {
                            members,
                            idx: idx + 1,
                            src,
                            used,
                        });
                        Ok(StepOut::Continue)
                    }
                }
            }
            Step::DestructureDefault { inner } => {
                // The default value (evaluated because the source was `undefined`)
                // is on top of the value stack.
                let v = values.pop().unwrap_or(NanBox::undefined());
                stack.push(Step::Destructure {
                    target: inner,
                    value: v,
                });
                Ok(StepOut::Continue)
            }
            Step::DestructureMemberKey { key, value } => {
                // The base object is on the value stack (kept there); evaluate the
                // computed key on top of it, then assign.
                stack.push(Step::DestructureMemberSet { value });
                self.gen_eval_expr(key, stack, values)
            }
            Step::DestructureMemberSet { value } => {
                let key = values.pop().unwrap_or(NanBox::undefined());
                let objval = values.pop().unwrap_or(NanBox::undefined());
                if let Some(raw) = objval.as_handle() {
                    self.assign_member_value(Handle::from_raw(raw), key, value)
                        .map_err(GenAbrupt::from)?;
                }
                Ok(StepOut::Continue)
            }
            Step::RunLoopBody { body, label } => {
                self.gen_exec_loop_body(body, stack, values, &label)
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
            Step::AwaitExpr => {
                // `await v`: suspend the async coroutine on `v`'s settlement. On
                // resume, `gen_drive` pushes the fulfilment value (or routes a
                // rejection through the unwinder), which becomes this expression's
                // result.
                let operand = values.pop().unwrap_or(NanBox::undefined());
                Ok(StepOut::Await(operand))
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
                Ok(Flow::Break(l, _)) => {
                    // A labeled break matching this statement's label is consumed.
                    if label.is_some() && l == *label {
                        Ok(StepOut::Continue)
                    } else {
                        Err(GenAbrupt::Break(l))
                    }
                }
                Ok(Flow::Continue(l, _)) => Err(GenAbrupt::Continue(l)),
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
                // `for await` drives the async-iterator protocol via
                // [`Self::for_await_values`] — awaiting each `next()` result for an
                // async iterable (`async function*` / `[Symbol.asyncIterator]`), or
                // awaiting each yielded value of a sync iterable. The values come
                // back already settled, so the `ForEach` step does not re-await.
                let items = if *is_await {
                    self.for_await_values(iterable).map_err(GenAbrupt::from)?
                } else {
                    self.iterate_values(iterable).map_err(GenAbrupt::from)?
                };
                stack.push(Step::ForEach {
                    left,
                    body,
                    values: items,
                    idx: 0,
                    label: label.clone(),
                    await_each: false,
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
                    await_each: false,
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
                // Run the body in a child scope carrying the `with` object (so the
                // object is captured lexically). A PopScope step restores the
                // parent scope for the suspended case.
                let child = self.current.child_with(obj);
                let saved = core::mem::replace(&mut self.current, child);
                stack.push(Step::PopScope {
                    scope: saved.clone(),
                });
                let r = self.gen_exec_stmt(body, stack, values, &None);
                // Restore eagerly on the synchronous path.
                self.current = saved;
                r
            }
            // Yield-bearing forms not otherwise handled fall back to one-shot
            // execution (no reachable yield will actually run).
            _ => match self.exec(stmt) {
                Ok(Flow::Normal(_)) => Ok(StepOut::Continue),
                Ok(Flow::Return(v)) => Err(GenAbrupt::Return(v)),
                Ok(Flow::Break(l, _)) => Err(GenAbrupt::Break(l)),
                Ok(Flow::Continue(l, _)) => Err(GenAbrupt::Continue(l)),
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
    /// One level of destructuring-assignment evaluation for the step-machine,
    /// mirroring [`Interp::assign_destructure`] but reified so a `yield` in a
    /// default initializer suspends. The iterator pull and property reads are
    /// synchronous (they cannot contain a reachable yield in the covered cases).
    fn gen_destructure(
        &mut self,
        target: &'a Expr,
        value: NanBox,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        match target {
            Expr::Array { elements, .. } => {
                let items = self.iterate_values(value).map_err(GenAbrupt::from)?;
                stack.push(Step::DestructureArrayElem {
                    elements,
                    idx: 0,
                    i: 0,
                    items,
                });
                Ok(StepOut::Continue)
            }
            Expr::Object { members, .. } => {
                if matches!(value.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Cannot destructure 'null' or 'undefined' as an object");
                    return Err(GenAbrupt::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let src = self
                    .require_object_coercible_to_object(value, "destructuring")
                    .map_err(GenAbrupt::from)?;
                stack.push(Step::DestructureObjectMember {
                    members,
                    idx: 0,
                    src,
                    used: Vec::new(),
                });
                Ok(StepOut::Continue)
            }
            // A defaulted target (`a = <default>`): use the default (which may
            // yield) only when the source value is `undefined`.
            Expr::Assign {
                op: crate::ast::AssignOp::Assign,
                target: inner,
                value: default_expr,
                ..
            } => {
                if matches!(value.unpack(), Unpacked::Undefined) {
                    stack.push(Step::DestructureDefault { inner });
                    self.gen_eval_expr(default_expr, stack, values)
                } else {
                    stack.push(Step::Destructure {
                        target: inner,
                        value,
                    });
                    Ok(StepOut::Continue)
                }
            }
            // A member leaf whose computed key (or base) contains a yield, e.g.
            // `[ x[yield] ] = vals` — evaluate base then key step-by-step, then
            // assign. A plain / static member leaf has no reachable yield and takes
            // the fast path below.
            Expr::Member {
                object,
                property: PropertyKey::Computed(key),
                ..
            } if expr_has_yield(object) || expr_has_yield(key) => {
                stack.push(Step::DestructureMemberKey { key, value });
                self.gen_eval_expr(object, stack, values)
            }
            // A leaf target (identifier or static member reference).
            _ => {
                self.assign_to(target, value).map_err(GenAbrupt::from)?;
                Ok(StepOut::Continue)
            }
        }
    }

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
            Expr::Await { argument, .. } => {
                // Evaluate the operand (it may itself await/yield), then suspend.
                stack.push(Step::AwaitExpr);
                self.gen_eval_expr(argument, stack, values)
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
            // `[ x = yield ] = rhs` / `{ x = yield } = rhs` — destructuring
            // assignment where a `yield` hides in a default initializer (or the
            // RHS). Evaluate the RHS, then destructure step-by-step so a default's
            // yield suspends. The RHS value is left on the stack as the result.
            Expr::Assign {
                op, target, value, ..
            } if matches!(op, crate::ast::AssignOp::Assign)
                && matches!(&**target, Expr::Array { .. } | Expr::Object { .. }) =>
            {
                stack.push(Step::DestructureStart { target });
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
            // `[a, yield b, ...yield c]` — evaluate each element step-by-step so a
            // yield in any position (including a spread operand) suspends.
            Expr::Array { elements, .. } => {
                stack.push(Step::ArrayLit {
                    elements,
                    idx: 0,
                    acc: Vec::new(),
                });
                Ok(StepOut::Continue)
            }
            // `{ a: yield b, ...yield s }` — step through members so a yield in a
            // data value or spread operand suspends. Restricted to plain
            // data-property (static key, non-function value) and spread members;
            // methods/accessors/computed-keys/`__proto__`/function-valued props
            // (which need name/home/proto handling and never reach a yield in their
            // own value) defer to the one-shot fast path below.
            Expr::Object { members, .. } if object_lit_steppable(members) => {
                let target = self.realm.new_object();
                stack.push(Step::ObjectLit {
                    members,
                    idx: 0,
                    target,
                });
                Ok(StepOut::Continue)
            }
            // Any other yield-bearing expression shape (e.g. `f(yield b)`,
            // a member/computed access, compound/destructuring assignment) is not
            // individually reified; fall back to one-shot eval. The yield-free fast
            // path above means this only runs for genuinely yield-bearing complex
            // operands, which are a documented follow-up.
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
        mut completion: Completion,
    ) -> Result<Option<Result<GenStep, ExecError>>, ExecError> {
        loop {
            let Some(step) = stack.pop() else {
                // Nothing caught it: the body itself is unwinding out. Dispose any
                // body-level `using` resources (recorded in the frame scope,
                // `self.current`) before the generator completes abnormally,
                // aggregating a disposer throw into the completion.
                completion = self.gen_dispose_unwind(completion);
                // The generator completes abnormally.
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
                Step::PopScope { scope } => {
                    // Leaving this scope abruptly: dispose its `using` resources,
                    // aggregating any disposer throw into the in-flight completion
                    // (SuppressedError chain) before continuing to unwind.
                    completion = self.gen_dispose_unwind(completion);
                    self.current = scope;
                }
                Step::Seq { scope, label, .. } => {
                    // The block scope (`self.current`) is leaving abruptly: dispose
                    // its `using` resources, aggregating against the completion.
                    completion = self.gen_dispose_unwind(completion);
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
        // A `using` / `await using` declarator must both bind the name *and*
        // record the disposable resource (and resolve its dispose method) — work
        // the stepped `DeclName` path does not do. Bind it in one shot via
        // `exec_single_declarator` (which records the resource); any `await` in the
        // initializer is resolved eagerly there, consistent with the engine's
        // eager-async model.
        let is_using = matches!(kind, VarDeclKind::Using | VarDeclKind::AwaitUsing);
        match (&d.target, &d.init) {
            (BindingTarget::Ident(Ident { name, .. }), Some(init))
                if expr_has_yield(init) && !is_using =>
            {
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
        // In an async generator, `yield*` uses GetIterator(operand, async): a
        // callable `[Symbol.asyncIterator]` is used directly; a non-callable (but
        // present) one is a TypeError. An absent `[Symbol.asyncIterator]` falls
        // through to the sync `[Symbol.iterator]` (AsyncFromSyncIterator — each
        // result is awaited per step in `gen_yield_star_step`).
        if self.gen_is_async {
            let sym = self.well_known_symbol("asyncIterator");
            let key = self.member_key(sym);
            let m = self.read_member(h, &key)?;
            if !(m.is_undefined() || m.is_null()) {
                if !m
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("[Symbol.asyncIterator] is not a function"));
                }
                let iterator = self.call_with_this(m, operand, &[])?;
                // GetIterator: the result must be an Object (a returned string /
                // symbol / number / boolean is a TypeError).
                if !self.is_object_value(iterator) {
                    return Err(self.type_error("yield* iterator is not an object"));
                }
                return iterator
                    .as_handle()
                    .map(Handle::from_raw)
                    .ok_or_else(|| self.type_error("yield* iterator is not an object"));
            }
        }
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
        // An async generator's inner iterator yields *promises* of results: await
        // each before reading `done`/`value` (eagerly, per the engine's async
        // model). A non-promise — a sync inner iterator (AsyncFromSyncIterator) —
        // passes through unchanged.
        let res = if self.gen_is_async {
            self.await_value(res).map_err(GenAbrupt::from)?
        } else {
            res
        };
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
