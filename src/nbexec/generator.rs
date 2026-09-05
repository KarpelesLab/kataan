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
    Argument, ArrayElement, BindingTarget, CatchClause, Class, ClassMember, Expr, ForInit, ForLeft,
    Ident, LogicalOp, MethodKind, ObjectMember, Param, PropertyKey, Stmt, SwitchCase, VarDeclKind,
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

/// A queued AsyncGenerator request (27.6 `[[AsyncGeneratorQueue]]`). Each
/// `next`/`return`/`throw` on an async generator creates one, appended FIFO;
/// the front element is the request currently being serviced, and its `promise`
/// settles (with the `{value, done}` result, or the thrown reason) when the
/// generator yields/returns/throws for it.
struct AsyncGenRequest {
    how: Resumption,
    promise: Handle,
}

/// The result of advancing a generator one step.
pub(crate) enum GenStep {
    /// Hit a `yield`; the operand is the surfaced value (`{value, done:false}`).
    Yielded(NanBox),
    /// Hit a `yield*` over a sync iterator: the operand is the inner iterator's
    /// result object, passed through *verbatim* (GeneratorYield(innerResult)) — so
    /// its lazy `value` getter is read by the outer consumer, never by `yield*`.
    YieldedResult(NanBox),
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
    /// (async generators only) the `[[AsyncGeneratorQueue]]` of pending
    /// `next`/`return`/`throw` requests. The front element is the request being
    /// serviced; the generator drains them one at a time in FIFO order, per-
    /// `await`/`yield` suspending onto the microtask queue between steps.
    async_queue: Vec<AsyncGenRequest>,
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
    /// The lazy `for await (left of right)` driver (14.7.5 ForIn/OfBodyEvaluation,
    /// async). One `next()` is pulled per iteration and parked on `await` — an
    /// infinite source with a `break` therefore terminates (calling `return`)
    /// rather than draining eagerly. `async_inner` is whether the iterator record
    /// came from `[Symbol.asyncIterator]` (native async, whose `next()` returns a
    /// promise) vs a sync iterator wrapped as AsyncFromSyncIterator (whose result
    /// values are unwrapped/awaited per iteration). This is the loop step the
    /// unwinder recognizes: a body break/return/throw (or non-matching label)
    /// triggers `IteratorClose`; a matching `continue` re-loops without closing.
    ForAwaitLoop {
        ih: Handle,
        next: NanBox,
        left: &'a ForLeft,
        body: &'a Stmt,
        label: Option<String>,
        async_inner: bool,
    },
    /// (`for await`) After awaiting the per-iteration `next()` result (native
    /// async) or the unwrapped `value` (sync-wrapped), bind it and run the body.
    ForAwaitBind {
        ih: Handle,
        next: NanBox,
        left: &'a ForLeft,
        body: &'a Stmt,
        label: Option<String>,
        async_inner: bool,
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
    /// The yield half of an async-generator `yield`: yield the value on the stack
    /// (which has already been `Await`ed). `AsyncGeneratorYield` = Await(value) then
    /// GeneratorYield; this is the second half, pushed after an `AwaitExpr`.
    AsyncYield,
    /// A `yield*` delegation pumping `iter`. `next` is the iterator's `next`
    /// method, cached once at acquisition per spec (GetIterator stores
    /// [[NextMethod]]) so it is not re-read every step — only `return`/`throw` are
    /// fetched per-use. `async_inner` records whether the delegated iterator was
    /// obtained via `[Symbol.asyncIterator]` (a native async iterator, whose
    /// `next()` results are promises) vs a sync iterator (whose result values are
    /// unwrapped per `AsyncFromSyncIteratorContinuation`).
    YieldStar {
        iter: Handle,
        next: NanBox,
        async_inner: bool,
    },
    /// (async `yield*` only) After awaiting a native-async inner iterator's raw
    /// `next`/`return`/`throw` result promise, process it: read `done`/`value` and
    /// either re-yield (non-done), complete the delegation, or forward a return.
    YieldStarResult {
        iter: Handle,
        next: NanBox,
        kind: YsKind,
    },
    /// (async `yield*` only) After awaiting the inner `value` (AsyncGeneratorYield /
    /// AsyncFromSyncIteratorContinuation value-unwrap), finish the step: re-yield
    /// (non-done), forward a return completion, or produce the `yield*` result.
    YieldStarAfterValue {
        iter: Handle,
        next: NanBox,
        async_inner: bool,
        done: bool,
        kind: YsKind,
        /// A [`Step::YieldStarClose`] guard sits directly beneath this step (the
        /// sync-wrapped non-done value-unwrap `Await`): remove it on the fulfil path
        /// so it only fires if that `Await` rejects.
        has_close_guard: bool,
    },
    /// (async `yield*` over a sync-wrapped iterator only) The IteratorClose-on-
    /// rejection guard for AsyncFromSyncIteratorContinuation: while a non-done inner
    /// `value` is being awaited, a rejection must close the sync iterator (call its
    /// `return`). Consumed without effect on the fulfil path (removed by the
    /// matching `YieldStarAfterValue`); on an unwinding throw it runs `IteratorClose`.
    YieldStarClose { iter: Handle },
    /// Complete a member read `<object on stack>.property` after the (possibly
    /// suspending) object operand has been evaluated onto the value stack. Used so
    /// `(await x).prop` / `(yield x).prop` suspend at the `await`/`yield` instead of
    /// falling to the eager one-shot walker (which would eager-await a still-pending
    /// promise). Restricted to non-`super`, non-optional bases with a yield-free key.
    MemberRead { property: &'a PropertyKey },
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
    /// (async generators) `return <awaited value on stack>` — the second half of
    /// `ReturnValue`, reached once the operand's `Await` has settled.
    ReturnAwaited,
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
    /// The ergonomic brand check `#name in <rhs>`, where `<rhs>` (already on the
    /// value stack) may have suspended on a `yield`/`await`. Pops the RHS and
    /// pushes whether the private element is present (TypeError if non-object).
    PrivateIn { name: &'a str },
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
    /// `import(specifier, options)` — the specifier is on the value stack;
    /// evaluate the (optional) second argument next, then perform the import.
    #[cfg(all(feature = "module", feature = "std"))]
    DynamicImportOptions {
        arguments: &'a [crate::ast::Argument],
    },
    /// Perform `import(…)` with its arguments evaluated: pops the options value
    /// (when `has_options`) and the specifier, pushing the resulting promise.
    #[cfg(all(feature = "module", feature = "std"))]
    DynamicImportCall { has_options: bool },
    /// Build a template literal: append quasi `idx`, then evaluate substitution
    /// `idx` (which may suspend). `acc` holds the bytes gathered so far.
    TemplateLit {
        tpl: &'a crate::ast::TemplateLiteral,
        idx: usize,
        acc: Vec<u8>,
    },
    /// `ToString` the top value into a template-literal accumulator, then continue
    /// with the next quasi.
    TemplateAppend {
        tpl: &'a crate::ast::TemplateLiteral,
        idx: usize,
        acc: Vec<u8>,
    },
    /// Build an object literal step-by-step onto `target`: `members[idx..]` remain.
    /// Handles every member form — data / method / accessor / spread / `__proto__`
    /// and computed keys whose expression may `yield`/`await`. Entered for any
    /// object literal that contains a reachable suspension (see the `Expr::Object`
    /// arm of `gen_eval_expr`); each member's key (if computed) and value are driven
    /// through `gen_eval_expr`, then defined with the same semantics as the eager
    /// walker (`obj_define_property_member` / `obj_define_accessor_member`).
    ObjectLit {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
    },
    /// A computed-key property `{ [k]: v }`: the evaluated key is on top of the
    /// stack — coerce it (`ToPropertyKey`), then evaluate the value for the pair.
    ObjectLitPropKey {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
    },
    /// Complete a property: its value is on top of the stack; define it on `target`
    /// under storage key `key` (SetFunctionName / `[[HomeObject]]` for a method or
    /// function-valued property), then continue with the next member.
    ObjectLitPropVal {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
        key: String,
    },
    /// A computed-key accessor `{ get [k]() {} }`: the evaluated key is on top of
    /// the stack — coerce it, create the get/set function, and pair it on `target`.
    ObjectLitAccessorKey {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
    },
    /// A `__proto__:` member whose value (top of stack) may have yielded: apply it
    /// as `target`'s `[[Prototype]]` (only for an Object or `null`), then continue.
    ObjectLitProtoSet {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
    },
    /// Spread the just-evaluated value (top of stack) into `target`
    /// (CopyDataProperties), then continue with the next member.
    ObjectLitSpread {
        members: &'a [ObjectMember],
        idx: usize,
        target: Handle,
    },
    /// Evaluate a class's *computed member keys* in source order so a `yield`/
    /// `await` in one suspends; `keys` accumulates `idx → storage-key`. When all
    /// are done the class is built (`make_class_with_keys`) and its value pushed.
    ClassKeys {
        class: &'a Class,
        idx: usize,
        keys: alloc::collections::BTreeMap<usize, String>,
    },
    /// Store the just-evaluated computed key (top of stack) for the class member at
    /// `idx` (running the `static`-named-`prototype` TypeError check), then continue.
    ClassKeyStore {
        class: &'a Class,
        idx: usize,
        keys: alloc::collections::BTreeMap<usize, String>,
    },
    /// A class *declaration*: the finished class value is on top of the stack;
    /// bind it to the class name (if any), leaving no expression value.
    ClassDeclBind { class: &'a Class },
    /// `export default <expr>` whose expression suspended: the finished value is
    /// on top of the stack; name it (NamedEvaluation) and bind `*default*`.
    #[cfg(all(feature = "module", feature = "std"))]
    ExportDefaultBind { expr: &'a Expr },
    /// Destructure `value` into assignment `target` (an array/object pattern, a
    /// defaulted target, or a leaf) — a `yield` in a default initializer suspends.
    /// The iterator pull / property reads themselves are synchronous.
    Destructure { target: &'a Expr, value: NanBox },
    /// Destructure `elements[idx..]` of an array pattern from the pre-iterated
    /// `items`; `i` is the source index consumed so far (holes advance it).
    DestructureArrayElem {
        elements: &'a [ArrayElement],
        idx: usize,
        i: usize,
        items: Vec<NanBox>,
    },
    /// Lazily destructure `elements[idx..]` over a **user iterator** `ih`, pulling
    /// exactly one `IteratorStep` per element (so `[a] = infiniteIterator`
    /// terminates and a per-element default that `yield`s suspends at the right
    /// moment). `done` is the shared `iteratorRecord.[[done]]`; the matching
    /// `DestructureArrayClose` guard sitting below this step on the stack performs
    /// `IteratorClose` — on the normal way out *or* while `gen_unwind` unwinds an
    /// abrupt completion (`return()`/`throw`/`break` from an element's default).
    DestructureArrayIter {
        elements: &'a [ArrayElement],
        idx: usize,
        ih: Handle,
        done: alloc::rc::Rc<core::cell::Cell<bool>>,
    },
    /// The `IteratorClose` guard for a lazy array destructuring (see
    /// [`Step::DestructureArrayIter`]). Executed normally after the last element,
    /// or matched by `gen_unwind` during an abrupt unwind.
    DestructureArrayClose {
        ih: Handle,
        done: alloc::rc::Rc<core::cell::Cell<bool>>,
    },
    /// A rest target `...obj[key]` (lazy path) whose computed `key` may `yield`:
    /// `obj` is on the value stack; evaluate `key` on top, then drain and assign.
    /// The reference is fully evaluated *before* the iterator is drained (spec
    /// order), so a `yield` in the key that is resumed with `return()`/`throw`
    /// closes the iterator without pulling a value.
    DestructureRestKey {
        key: &'a Expr,
        ih: Handle,
        done: alloc::rc::Rc<core::cell::Cell<bool>>,
    },
    /// Complete `...obj[key]`: `obj` and (on top) the key are on the value stack;
    /// drain the remaining iterator values into a fresh array and assign it.
    DestructureRestSet {
        ih: Handle,
        done: alloc::rc::Rc<core::cell::Cell<bool>>,
    },
    /// An element target `obj[key]` (lazy path) whose computed `key` may `yield`:
    /// `obj` is on the value stack; evaluate `key`, then do the one `IteratorStep`
    /// and assign. The reference is evaluated *before* the step (spec order), so a
    /// `yield` in the key resumed with `return()`/`throw` closes the iterator
    /// without calling `next`.
    DestructureElemKey {
        key: &'a Expr,
        ih: Handle,
        done: alloc::rc::Rc<core::cell::Cell<bool>>,
    },
    /// Complete an element `obj[key]`: `obj` and (on top) the key are on the value
    /// stack; take one `IteratorStep` (unless the iterator is done) and assign.
    DestructureElemSet {
        ih: Handle,
        done: alloc::rc::Rc<core::cell::Cell<bool>>,
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
    /// A reified plain call `f(args)` whose arguments may `yield`. `func` is the
    /// already-evaluated callee value (this path is only taken for a *plain*,
    /// non-method call, so the call's `this` is `undefined`). `arguments[idx..]`
    /// remain to evaluate; `acc` holds the argument values gathered so far. When
    /// `idx` reaches the end the call is performed and its result pushed.
    CallArgs {
        func: NanBox,
        arguments: &'a [Argument],
        idx: usize,
        acc: Vec<NanBox>,
    },
    /// Append the just-evaluated argument (top of stack) to a call's argument
    /// accumulator (spreading it when the argument was `...arg`), then continue.
    CallArgAppend {
        func: NanBox,
        arguments: &'a [Argument],
        idx: usize,
        acc: Vec<NanBox>,
        spread: bool,
    },
    /// A reified method call `recv.property(args)` whose arguments may `await`/
    /// `yield`. The receiver `recv` was evaluated eagerly (its expression is
    /// yield-free) and already checked non-nullish (the pre-argument spec order);
    /// `property` is a yield-free key. `arguments[idx..]` remain to evaluate; `acc`
    /// holds the argument values gathered so far. When `idx` reaches the end the
    /// call is completed via `call_member_dispatch` — identical built-in / own-
    /// property / `this`-binding semantics to the eager path — and its result
    /// pushed. Only entered for non-optional calls on a non-`super`, non-`import`
    /// receiver, so no optional short-circuit can arise inside the step machine.
    MethodCallArgs {
        recv: NanBox,
        property: &'a PropertyKey,
        arguments: &'a [Argument],
        idx: usize,
        acc: Vec<NanBox>,
    },
    /// Append the just-evaluated argument (top of stack) to a method call's
    /// argument accumulator (spreading it when the argument was `...arg`), then
    /// continue with the next argument.
    MethodCallArgAppend {
        recv: NanBox,
        property: &'a PropertyKey,
        arguments: &'a [Argument],
        idx: usize,
        acc: Vec<NanBox>,
        spread: bool,
    },
    /// Complete a static-key member assignment `base.p = <value>` (or
    /// `base['p'] = …`) whose right-hand side may `yield`: the RHS value is on top
    /// of the value stack (and stays there as the assignment's result); `base` and
    /// the static `property` were captured before the RHS was stepped, so a
    /// `return`/`throw` resumption at the RHS `yield` unwinds *without* performing
    /// the assignment (exactly as if the abrupt completion appeared at that point).
    AssignMemberStatic {
        base: Handle,
        property: &'a PropertyKey,
    },
    /// A resumable `DisposeResources` run for a scope that is being left and holds
    /// at least one `await using` resource. Each async disposer's result is
    /// `Await`ed as a *real* coroutine suspension (a `Step::AwaitExpr` pushed above
    /// this step), so the disposers of one scope are separated by microtask turns.
    /// See [`DisposeState`].
    Dispose(alloc::boxed::Box<DisposeState>),
}

/// The resumable state of one `DisposeResources` run (7.5.5), driven by
/// [`Step::Dispose`]. Resources are disposed in **reverse** declaration order (the
/// back of `rest` is the next one), and every async-hint disposer's result is
/// awaited as a real coroutine suspension.
struct DisposeState {
    /// Not-yet-disposed resources, in declaration order: `(value, method,
    /// isAsyncHint)`. The next resource to dispose is the **last** element.
    rest: alloc::vec::Vec<(NanBox, NanBox, bool)>,
    /// The accumulated throw completion (`completion` in the spec): a later
    /// disposer throw suppresses it into a `SuppressedError` chain.
    pending: Option<NanBox>,
    /// The abrupt completion the scope was already unwinding, restored when
    /// disposal ends without a throw. `None` = the scope was leaving normally.
    restore: Option<Completion>,
    /// The enclosing scope to restore once disposal is over (`None` = a body-level
    /// scope, which the frame itself owns).
    scope: Option<Scope>,
    /// The label of the block being left (only set on the unwind path): a
    /// `break <label>` targeting it is consumed once disposal is done.
    label: Option<String>,
    /// `DisposeResources`' *needsAwait*: an async-hint resource had no dispose
    /// method, so an `Await(undefined)` is still owed (step 3.d or step 4).
    needs_await: bool,
    /// `DisposeResources`' *hasAwaited*: a real `Await` already ran for some
    /// resource, which retires the owed step-3.d / step-4 `Await`.
    has_awaited: bool,
}

/// What [`Interp::gen_begin_dispose`] did with a scope's `using` resources.
enum DisposeStart {
    /// A [`Step::Dispose`] was pushed: the step machine drives the rest (and will
    /// re-raise / consume the completion itself).
    Stepped,
    /// Disposal finished inline (no `await using` resource in the scope); this is
    /// the resulting completion, `None` meaning "resume normally".
    Inline(Option<Completion>),
}

/// Which method an async `yield*` delegation step is servicing (mirrors the
/// resume completion forwarded to the inner iterator).
#[derive(Clone, Copy)]
enum YsKind {
    Next,
    Return,
    Throw,
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
    /// A sync `yield*` step surfacing the inner iterator's result object verbatim
    /// (see [`GenStep::YieldedResult`]).
    YieldResult(NanBox),
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
    ///
    /// `ctor_proto` is the invoked generator function's own `.prototype` object,
    /// used as the generator object's `[[Prototype]]` per `GetPrototypeFrom`
    /// `Constructor`. It chains to `%GeneratorPrototype%` /
    /// `%AsyncGeneratorPrototype%`, which carry the shared `next`/`return`/`throw`
    /// and `@@toStringTag`, so the generator object itself has NO own methods —
    /// they are inherited, exactly as the spec requires. When `ctor_proto` is not
    /// an object we fall back to the appropriate intrinsic prototype.
    pub(crate) fn make_lazy_generator(
        &mut self,
        body: &'a [Stmt],
        scope: Scope,
        is_async: bool,
        ctor_proto: Option<Handle>,
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
            async_queue: Vec::new(),
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
        let proto = ctor_proto.or_else(|| {
            if is_async {
                self.async_generator_prototype()
            } else {
                self.generator_prototype()
            }
        });
        if let Some(proto) = proto {
            self.realm.set_object_proto(obj, Some(proto));
        }
        NanBox::handle(obj.to_raw())
    }

    /// The shared `%GeneratorPrototype%`: `next`/`return`/`throw` (length 1, each
    /// dispatching on `this`'s generator frame) and `[Symbol.toStringTag]`
    /// "Generator", inheriting `%IteratorPrototype%`. Created once, cached on the
    /// `Iterator` constructor.
    pub(crate) fn generator_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}genproto";
        if let Some(gp) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(gp);
        }
        let iter_proto = self
            .realm
            .get_property(iter_ctor, "prototype")
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)?;
        let gp = self.realm.new_object_with_proto(Some(iter_proto));
        for (name, nid) in [
            ("next", N_GEN_NEXT),
            ("return", N_GEN_RETURN),
            ("throw", N_GEN_THROW),
        ] {
            let f = self.realm.new_native(nid);
            self.install_fn_name_length(f, name, 1);
            self.realm
                .set_property(gp, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(gp, name);
        }
        self.install_to_string_tag(gp, "Generator");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(gp.to_raw()));
        Some(gp)
    }

    /// `%GeneratorFunction.prototype%` — an ordinary object inheriting
    /// `%Function.prototype%`, whose own `prototype` data property is
    /// `%GeneratorPrototype%` (so `Object.getPrototypeOf(g).prototype` resolves to
    /// it), with `[Symbol.toStringTag]` "GeneratorFunction". A sync generator
    /// function's `[[Prototype]]` is set to this (via `set_native_proto`), which
    /// `object_proto` honors ahead of the `%Function.prototype%` fallback. Cached on
    /// the `Iterator` constructor.
    pub(crate) fn generator_function_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}genfnproto";
        if let Some(gfp) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(gfp);
        }
        let fn_proto = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let gfp = self.realm.new_object_with_proto(fn_proto);
        if let Some(gp) = self.generator_prototype() {
            self.realm
                .set_property(gfp, "prototype", NanBox::handle(gp.to_raw()));
            self.realm.mark_hidden(gfp, "prototype");
            self.realm.set_readonly_property(gfp, "prototype");
            // `%GeneratorPrototype%.constructor` is `%GeneratorFunction.prototype%`
            // (this `gfp`): { writable:false, enumerable:false, configurable:true }.
            self.realm
                .set_property(gp, "constructor", NanBox::handle(gfp.to_raw()));
            self.realm.mark_hidden(gp, "constructor");
            self.realm.set_readonly_property(gp, "constructor");
        }
        self.install_to_string_tag(gfp, "GeneratorFunction");
        // `%GeneratorFunction%` — the constructor, reachable as
        // `Object.getPrototypeOf(function*(){}).constructor`. Its own `[[Prototype]]`
        // is `%Function%`; `prototype` is `%GeneratorFunction.prototype%`
        // { w:false,e:false,c:false }; the prototype's `constructor` points back
        // { w:false,e:false,c:true }.
        let gf = self.realm.new_native(N_GENERATOR_FUNCTION_CTOR);
        // `GetFunctionRealm` tagging: a `%GeneratorFunction%` built lazily while
        // running inside a `$262.createRealm()` realm belongs to *that* realm — so a
        // cross-realm `Reflect.construct(otherRealm.GeneratorFunction, …)` enters the
        // constructor's realm, giving the created function's `.prototype` object and
        // body that realm's `%GeneratorPrototype%` / globals (CreateDynamicFunction
        // step 19 `realmF`). Untagged (main realm) leaves the fast path untouched.
        if let Some(idx) = self.cur_realm {
            self.fn_realm.insert(gf.to_raw(), idx);
        }
        self.install_fn_name_length(gf, "GeneratorFunction", 1);
        self.realm
            .set_property(gf, "prototype", NanBox::handle(gfp.to_raw()));
        self.realm.mark_hidden(gf, "prototype");
        self.realm.set_readonly_property(gf, "prototype");
        self.realm.set_non_configurable_property(gf, "prototype");
        if let Some(fn_ctor) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(gf, fn_ctor);
        }
        self.realm
            .set_property(gfp, "constructor", NanBox::handle(gf.to_raw()));
        self.realm.mark_hidden(gfp, "constructor");
        self.realm.set_readonly_property(gfp, "constructor");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(gfp.to_raw()));
        Some(gfp)
    }

    /// `%AsyncFunction.prototype%` — an ordinary object inheriting
    /// `%Function.prototype%` with `[Symbol.toStringTag]` "AsyncFunction"
    /// ({ w:false, e:false, c:true }). A (non-generator) `async function`'s
    /// `[[Prototype]]` is set to this via `set_native_proto`, so
    /// `Object.prototype.toString.call(asyncFn)` yields "[object AsyncFunction]"
    /// (the tag is read through the prototype chain — including a proxy wrapper).
    /// It has no own `prototype` (async functions are not constructable);
    /// `.constructor` intentionally still resolves up to `%Function%` (the
    /// `AsyncFunction === Function` conflation), so only the tag is added. Cached
    /// on the `Iterator` constructor.
    pub(crate) fn async_function_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asyncfnproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let fn_proto = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let afp = self.realm.new_object_with_proto(fn_proto);
        self.install_to_string_tag(afp, "AsyncFunction");
        // `%AsyncFunction%` — the constructor, reachable as
        // `Object.getPrototypeOf(async function(){}).constructor`. Its own
        // `[[Prototype]]` is `%Function%`; `prototype` is `%AsyncFunction.prototype%`
        // { w:false, e:false, c:false }; the prototype's `constructor` points back
        // { w:false, e:false, c:true }. Distinct from `%Function%` so
        // `asyncFn.constructor.prototype[@@toStringTag]` targets THIS prototype
        // (the `Object.prototype.toString` tag), not `%Function.prototype%`.
        let af = self.realm.new_native(N_ASYNC_FUNCTION_CTOR);
        // `GetFunctionRealm` tagging (see `%GeneratorFunction%`): a lazily-built
        // `%AsyncFunction%` belongs to the realm it was built in.
        if let Some(idx) = self.cur_realm {
            self.fn_realm.insert(af.to_raw(), idx);
        }
        self.install_fn_name_length(af, "AsyncFunction", 1);
        self.realm
            .set_property(af, "prototype", NanBox::handle(afp.to_raw()));
        self.realm.mark_hidden(af, "prototype");
        self.realm.set_readonly_property(af, "prototype");
        self.realm.set_non_configurable_property(af, "prototype");
        if let Some(fn_ctor) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(af, fn_ctor);
        }
        self.realm
            .set_property(afp, "constructor", NanBox::handle(af.to_raw()));
        self.realm.mark_hidden(afp, "constructor");
        self.realm.set_readonly_property(afp, "constructor");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(afp.to_raw()));
        Some(afp)
    }

    /// `%AsyncIteratorPrototype%` — `[Symbol.asyncIterator]` returns `this`,
    /// inheriting `%Object.prototype%`. Cached on the `Iterator` constructor.
    fn async_iterator_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asynciterproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let obj_proto = self
            .current
            .get("Object")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let aip = self.realm.new_object_with_proto(obj_proto);
        let self_iter = self.realm.new_native(N_ITERATOR_PROTO_SELF);
        self.install_fn_name_length(self_iter, "[Symbol.asyncIterator]", 0);
        let sym = self.well_known_symbol("asyncIterator");
        let key = self.member_key(sym);
        self.realm
            .set_hidden_property(aip, &key, NanBox::handle(self_iter.to_raw()));
        // `%AsyncIteratorPrototype%[@@asyncDispose]` (length 0).
        let dispose = self.realm.new_native(N_ASYNC_ITERATOR_DISPOSE);
        self.install_fn_name_length(dispose, "[Symbol.asyncDispose]", 0);
        let dsym = self.well_known_symbol("asyncDispose");
        let dkey = self.member_key(dsym);
        self.realm
            .set_property(aip, &dkey, NanBox::handle(dispose.to_raw()));
        self.realm.mark_hidden(aip, &dkey);
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(aip.to_raw()));
        Some(aip)
    }

    /// `%AsyncGeneratorPrototype%` — `next`/`return`/`throw` (length 1, each
    /// dispatching on `this`'s frame and wrapping the result in a promise) and
    /// `[Symbol.toStringTag]` "AsyncGenerator", inheriting `%AsyncIteratorPrototype%`.
    pub(crate) fn async_generator_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asyncgenproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let aip = self.async_iterator_prototype();
        let agp = self.realm.new_object_with_proto(aip);
        for (name, nid) in [
            ("next", N_ASYNC_GEN_NEXT),
            ("return", N_ASYNC_GEN_RETURN),
            ("throw", N_ASYNC_GEN_THROW),
        ] {
            let f = self.realm.new_native(nid);
            self.install_fn_name_length(f, name, 1);
            self.realm
                .set_property(agp, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(agp, name);
        }
        self.install_to_string_tag(agp, "AsyncGenerator");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(agp.to_raw()));
        Some(agp)
    }

    /// `%AsyncGeneratorFunction.prototype%` — own `prototype` =
    /// `%AsyncGeneratorPrototype%`, `[Symbol.toStringTag]` "AsyncGeneratorFunction",
    /// inheriting `%Function.prototype%`. An `async function*`'s `[[Prototype]]` is
    /// set to this via `set_native_proto`.
    pub(crate) fn async_generator_function_prototype(&mut self) -> Option<Handle> {
        let iter_ctor = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)?;
        const CACHE: &str = "\u{0}asyncgenfnproto";
        if let Some(h) = self
            .realm
            .get_property(iter_ctor, CACHE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return Some(h);
        }
        let fn_proto = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let agfp = self.realm.new_object_with_proto(fn_proto);
        if let Some(agp) = self.async_generator_prototype() {
            self.realm
                .set_property(agfp, "prototype", NanBox::handle(agp.to_raw()));
            self.realm.mark_hidden(agfp, "prototype");
            self.realm.set_readonly_property(agfp, "prototype");
            // `%AsyncGeneratorPrototype%.constructor` is
            // `%AsyncGeneratorFunction.prototype%` (this `agfp`):
            // { writable:false, enumerable:false, configurable:true }.
            self.realm
                .set_property(agp, "constructor", NanBox::handle(agfp.to_raw()));
            self.realm.mark_hidden(agp, "constructor");
            self.realm.set_readonly_property(agp, "constructor");
        }
        self.install_to_string_tag(agfp, "AsyncGeneratorFunction");
        // `%AsyncGeneratorFunction%` — the constructor, reachable as
        // `Object.getPrototypeOf(async function*(){}).constructor`.
        let agf = self.realm.new_native(N_ASYNC_GENERATOR_FUNCTION_CTOR);
        // `GetFunctionRealm` tagging (see `%GeneratorFunction%`): a lazily-built
        // `%AsyncGeneratorFunction%` belongs to the realm it was built in.
        if let Some(idx) = self.cur_realm {
            self.fn_realm.insert(agf.to_raw(), idx);
        }
        self.install_fn_name_length(agf, "AsyncGeneratorFunction", 1);
        self.realm
            .set_property(agf, "prototype", NanBox::handle(agfp.to_raw()));
        self.realm.mark_hidden(agf, "prototype");
        self.realm.set_readonly_property(agf, "prototype");
        self.realm.set_non_configurable_property(agf, "prototype");
        if let Some(fn_ctor) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(agf, fn_ctor);
        }
        self.realm
            .set_property(agfp, "constructor", NanBox::handle(agf.to_raw()));
        self.realm.mark_hidden(agfp, "constructor");
        self.realm.set_readonly_property(agfp, "constructor");
        self.realm
            .set_hidden_property(iter_ctor, CACHE, NanBox::handle(agfp.to_raw()));
        Some(agfp)
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
                // A sync `yield*` passes the inner result object through verbatim.
                Ok(GenStep::YieldedResult(v)) => break Ok(v),
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

    pub(crate) fn gen_result(&mut self, value: NanBox, done: bool) -> NanBox {
        let r = self.realm.new_object();
        self.realm.set_property(r, "value", value);
        self.realm.set_property(r, "done", NanBox::boolean(done));
        NanBox::handle(r.to_raw())
    }

    /// `AsyncFromSyncIteratorContinuation(result, promiseCapability,
    /// syncIteratorRecord, true)` (27.1.4.4) for the `result` object a *sync*
    /// iterator just produced: reads `done`/`value`, wraps the value with
    /// `PromiseResolve(%Promise%, value)`, and returns the promise that settles
    /// with `CreateIterResultObject(unwrapped, done)`.
    ///
    /// The wrapper promise is a real link in the chain — awaiting the returned
    /// promise therefore costs the two microtask turns the spec prescribes, not
    /// one. An abrupt `PromiseResolve` (step 6) closes the sync iterator when the
    /// result was not `done`, then propagates.
    fn async_from_sync_continuation(
        &mut self,
        iter: Handle,
        result: Result<NanBox, ExecError>,
    ) -> Result<Handle, ExecError> {
        let result = result?;
        let Some(rh) = result.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("iterator result is not an object"));
        };
        let done = self.read_member(rh, "done")?;
        let done = self.realm.truthy(done);
        let value = self.read_member(rh, "value")?;
        let w = match self.promise_resolve_checked(value) {
            Ok(w) => w,
            Err(e) => {
                // `closeOnRejection` is true here (this is `next`, not `return`):
                // a non-done result closes the sync iterator before rejecting.
                if !done {
                    let _ = self.iterator_close(iter);
                }
                return Err(e);
            }
        };
        let state = self.realm.new_array(alloc::vec![
            NanBox::boolean(done),
            NanBox::handle(iter.to_raw())
        ]);
        let on_f = self.realm.new_bound_native(N_ASYNC_FROM_SYNC_UNWRAP, state);
        // Step 10: `onRejected` exists only for a non-done result; for a done one
        // the rejection simply passes through to the capability.
        let on_r = if done {
            NanBox::undefined()
        } else {
            let f = self.realm.new_bound_native(N_ASYNC_FROM_SYNC_CLOSE, state);
            NanBox::handle(f.to_raw())
        };
        Ok(self.register_then(w, NanBox::handle(on_f.to_raw()), on_r, false))
    }

    /// `%AsyncFromSyncIteratorPrototype%.next` over a sync iterator: pull one
    /// `next()` and run [`Self::async_from_sync_continuation`] on it, **always**
    /// returning a promise. Every abrupt completion inside is an
    /// `IfAbruptRejectPromise` — it rejects the returned promise rather than
    /// throwing at the call site, so the caller's `Await` still costs its tick and
    /// the error surfaces one turn later (observable, and what `for await` over a
    /// poisoned sync iterator relies on).
    fn async_from_sync_next(&mut self, iter: Handle, next: NanBox) -> Result<Handle, ExecError> {
        let iter_val = NanBox::handle(iter.to_raw());
        let result = self.call_with_this(next, iter_val, &[]);
        match self.async_from_sync_continuation(iter, result) {
            Ok(p) => Ok(p),
            Err(ExecError::Throw(e)) => {
                let p = self.fresh_promise();
                self.settle(p, e, false);
                Ok(p)
            }
            // A non-throw fatal (stack overflow, resource limit) is not a JS
            // completion and must keep unwinding.
            Err(other) => Err(other),
        }
    }

    /// A fresh promise already rejected with a `TypeError` carrying `msg`.
    fn rejected_type_error(&mut self, msg: &str) -> NanBox {
        let p = self.fresh_promise();
        let m = self.new_str(msg);
        let e = self.make_error(N_TYPE_ERROR, Some(m));
        self.settle(p, e, false);
        NanBox::handle(p.to_raw())
    }

    /// `%AsyncGeneratorPrototype%.next/return/throw`: unlike the sync methods
    /// these ALWAYS return a promise (per the AsyncGenerator abstract operations,
    /// which create the promise capability *before* validating `this`). A `this`
    /// that is not an async generator therefore rejects the returned promise with
    /// a `TypeError` rather than throwing synchronously.
    pub(crate) fn async_gen_resume(&mut self, this: NanBox, how: Resumption) -> NanBox {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return self.rejected_type_error("Generator method called on non-object");
        };
        let Some(id) = self.gen_frame_id(h) else {
            return self.rejected_type_error("Generator method called on a non-generator");
        };
        if !self.gen_frames[id].as_ref().is_some_and(|f| f.is_async) {
            return self.rejected_type_error("Generator method called on a non-async-generator");
        }
        // `AsyncGeneratorEnqueue` (27.6.3.5): create the request's promise
        // capability and append the request to the `[[AsyncGeneratorQueue]]`. The
        // returned promise settles when this request reaches the front and the
        // generator yields/returns/throws for it — so multiple `.next()` calls
        // before the first resolves queue and settle in FIFO order.
        let promise = self.fresh_promise();
        let was_idle = self.gen_frames[id]
            .as_ref()
            .is_some_and(|f| f.async_queue.is_empty() && !f.running);
        if let Some(f) = self.gen_frames[id].as_mut() {
            f.async_queue.push(AsyncGenRequest { how, promise });
        }
        // Kick off `AsyncGeneratorResume` only when the generator is idle (no
        // request currently being serviced and not synchronously executing). If it
        // is executing or parked on an `await`, the request just waits its turn:
        // the in-flight drive drains it when the current request settles.
        if was_idle {
            self.async_gen_drive(h, id, how);
        }
        NanBox::handle(promise.to_raw())
    }

    /// Drives an async generator through queued requests, per-`await`/`yield`
    /// suspending onto the microtask queue (27.6.3 AsyncGeneratorResume / Yield /
    /// DrainQueue). `how` is the completion to resume the generator body with: the
    /// front request's own completion when (re)starting a request, or the settled
    /// awaited value when resumed from an `await` microtask. Each iteration either
    /// suspends (parking the generator until a microtask resumes it) or settles the
    /// front request's promise and advances to the next queued request.
    fn async_gen_drive(&mut self, genobj: Handle, id: usize, how: Resumption) {
        self.async_gen_drive_from(genobj, id, how, true);
    }

    /// [`Self::async_gen_drive`], with `unwrap` selecting whether the *initial*
    /// `how` still has to go through `AsyncGeneratorUnwrapYieldResumption`
    /// (27.6.3.7). It is `false` only on the re-entry from that operation's own
    /// `Await` reaction, whose value has already been unwrapped.
    fn async_gen_drive_from(
        &mut self,
        genobj: Handle,
        id: usize,
        mut how: Resumption,
        mut unwrap: bool,
    ) {
        loop {
            let (started, done) = self.gen_frames[id]
                .as_ref()
                .map_or((true, true), |f| (f.started, f.done));
            // `AsyncGeneratorAwaitReturn` (27.6.3.8): a `return(v)` on a generator
            // that has not started or has already completed does not run any
            // generator code — it awaits `v`, then settles the request with
            // `{value: v, done: true}` (or rejects if the await rejects).
            if let Resumption::Return(v) = how
                && (!started || done)
            {
                if let Some(f) = self.gen_frames[id].as_mut() {
                    f.done = true;
                }
                // Step 6/7: `PromiseResolve` is a *completion* — a `constructor`
                // getter that throws completes the step abruptly, rejecting the
                // request's promise instead of parking on an await.
                let inner = match self.promise_resolve_checked(v) {
                    Ok(p) => p,
                    Err(e) => {
                        let e = self.abrupt_value(e, "async generator internal error");
                        self.async_gen_settle(id, e, false);
                        match self.next_queued(id) {
                            Some(next) => {
                                how = next;
                                unwrap = true;
                                continue;
                            }
                            None => return,
                        }
                    }
                };
                let on_f = self
                    .realm
                    .new_bound_native(N_ASYNC_GEN_RETURN_FULFILL, genobj);
                let on_r = self
                    .realm
                    .new_bound_native(N_ASYNC_GEN_RETURN_REJECT, genobj);
                self.register_then(
                    inner,
                    NanBox::handle(on_f.to_raw()),
                    NanBox::handle(on_r.to_raw()),
                    false,
                );
                return;
            }
            // `AsyncGeneratorUnwrapYieldResumption` (27.6.3.7): a `return`
            // completion delivered to a generator suspended *at a yield* is first
            // `Await`ed, and only the settled value becomes the return completion
            // resumed into the body. A rejection (or an abrupt `PromiseResolve`)
            // is instead thrown at the yield point, where the body's `try`/
            // `finally` can observe it.
            if unwrap && let Resumption::Return(v) = how {
                match self.promise_resolve_checked(v) {
                    Ok(inner) => {
                        let on_f = self
                            .realm
                            .new_bound_native(N_ASYNC_GEN_YIELD_RETURN_FULFILL, genobj);
                        let on_r = self
                            .realm
                            .new_bound_native(N_ASYNC_GEN_YIELD_RETURN_REJECT, genobj);
                        self.register_then(
                            inner,
                            NanBox::handle(on_f.to_raw()),
                            NanBox::handle(on_r.to_raw()),
                            false,
                        );
                        return;
                    }
                    Err(e) => {
                        how = Resumption::Throw(
                            self.abrupt_value(e, "async generator internal error"),
                        );
                    }
                }
            }
            unwrap = true;
            match self.run_generator(id, how) {
                // `await value`: `PromiseResolve(value)` then park the generator —
                // a microtask resumes it (at the `await` point) on settlement. The
                // front request stays in place; it is settled on a later yield/done.
                Ok(GenStep::Awaited(v)) => {
                    // An abrupt `PromiseResolve` (a throwing `constructor` getter)
                    // is the `Await`'s own completion: resume the body by throwing
                    // it at the `await` point rather than parking.
                    let inner = match self.promise_resolve_checked(v) {
                        Ok(p) => p,
                        Err(e) => {
                            how = Resumption::Throw(
                                self.abrupt_value(e, "async generator internal error"),
                            );
                            continue;
                        }
                    };
                    let on_f = self
                        .realm
                        .new_bound_native(N_ASYNC_GEN_AWAIT_FULFILL, genobj);
                    let on_r = self
                        .realm
                        .new_bound_native(N_ASYNC_GEN_AWAIT_REJECT, genobj);
                    self.register_then(
                        inner,
                        NanBox::handle(on_f.to_raw()),
                        NanBox::handle(on_r.to_raw()),
                        false,
                    );
                    return;
                }
                // `yield value` (value already `await`ed by the step machine):
                // resolve the front request with `{value, done:false}` and advance.
                Ok(GenStep::Yielded(v)) => {
                    let result = self.gen_result(v, false);
                    self.async_gen_settle(id, result, true);
                }
                // An async `yield*` passes the inner iterator's result object
                // through verbatim (`{value, done}`); resolve the request with it.
                Ok(GenStep::YieldedResult(v)) => {
                    self.async_gen_settle(id, v, true);
                }
                // The body completed / returned: resolve the front request with
                // `{value, done:true}`, then drain any remaining queued requests.
                Ok(GenStep::Done(v)) => {
                    let result = self.gen_result(v, true);
                    self.async_gen_settle(id, result, true);
                }
                // An uncaught throw completes the generator: reject the front
                // request, then drain the rest against the now-completed generator.
                Err(ExecError::Throw(e)) => {
                    self.async_gen_settle(id, e, false);
                }
                Err(_) => {
                    let m = self.new_str("async generator internal error");
                    let e = self.make_error(N_TYPE_ERROR, Some(m));
                    self.async_gen_settle(id, e, false);
                }
            }
            // The current request has been settled and removed. Continue with the
            // next queued request (AsyncGeneratorDrainQueue), or park if none.
            match self.next_queued(id) {
                Some(next) => how = next,
                None => return,
            }
        }
    }

    /// The completion of the front `[[AsyncGeneratorQueue]]` request, if any.
    fn next_queued(&self, id: usize) -> Option<Resumption> {
        self.gen_frames[id]
            .as_ref()
            .and_then(|f| f.async_queue.first().map(|r| r.how))
    }

    /// The thrown value of an abrupt completion reached where no `Result` can be
    /// propagated (inside a microtask reaction): a genuine `throw` surfaces its
    /// value, any other fatal becomes a `TypeError` carrying `msg`.
    fn abrupt_value(&mut self, e: ExecError, msg: &str) -> NanBox {
        match e {
            ExecError::Throw(v) => v,
            _ => {
                let m = self.new_str(msg);
                self.make_error(N_TYPE_ERROR, Some(m))
            }
        }
    }

    /// The `AsyncGeneratorUnwrapYieldResumption` `Await` reaction: the awaited
    /// `return(v)` value settled while the generator was suspended at a `yield`.
    /// On fulfilment the body is resumed with a `return` completion carrying the
    /// *unwrapped* value (skipping a second unwrap); on rejection the reason is
    /// thrown at the yield point.
    pub(crate) fn async_gen_yield_return_settled(
        &mut self,
        genobj: Handle,
        value: NanBox,
        fulfilled: bool,
    ) {
        let Some(id) = self.gen_frame_id(genobj) else {
            return;
        };
        if fulfilled {
            self.async_gen_drive_from(genobj, id, Resumption::Return(value), false);
        } else {
            self.async_gen_drive_from(genobj, id, Resumption::Throw(value), false);
        }
    }

    /// Removes the front `[[AsyncGeneratorQueue]]` request and settles its promise
    /// (`AsyncGeneratorCompleteStep`): resolves with `value` (an iterator-result
    /// object) when `fulfilled`, else rejects with `value` as the thrown reason.
    fn async_gen_settle(&mut self, id: usize, value: NanBox, fulfilled: bool) {
        let req = self.gen_frames[id].as_mut().and_then(|f| {
            if f.async_queue.is_empty() {
                None
            } else {
                Some(f.async_queue.remove(0))
            }
        });
        if let Some(req) = req {
            if fulfilled {
                self.resolve_with(req.promise, value);
            } else {
                self.settle(req.promise, value, false);
            }
        }
    }

    /// Resumes an async generator parked on an `await`, or continues its request
    /// queue after an `AsyncGeneratorAwaitReturn` settles. Called from the bound
    /// microtask reactions (`N_ASYNC_GEN_*_FULFILL`/`_REJECT`); `gen` is the async
    /// generator object, from which the frame id is recovered.
    pub(crate) fn async_gen_resume_await(&mut self, genobj: Handle, how: Resumption) {
        if let Some(id) = self.gen_frame_id(genobj) {
            self.async_gen_drive(genobj, id, how);
        }
    }

    /// The `AsyncGeneratorAwaitReturn` fulfilment reaction: the awaited return
    /// value settled — resolve the front request with `{value, done:true}`, then
    /// drain the remaining queue.
    pub(crate) fn async_gen_return_settled(
        &mut self,
        genobj: Handle,
        value: NanBox,
        fulfilled: bool,
    ) {
        let Some(id) = self.gen_frame_id(genobj) else {
            return;
        };
        if fulfilled {
            let result = self.gen_result(value, true);
            self.async_gen_settle(id, result, true);
        } else {
            self.async_gen_settle(id, value, false);
        }
        // Drain any requests queued behind the completed return.
        if let Some(next) = self.gen_frames[id]
            .as_ref()
            .and_then(|f| f.async_queue.first().map(|r| r.how))
        {
            self.async_gen_drive(genobj, id, next);
        }
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
            async_queue: Vec::new(),
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
        // A coroutine driving a *module* body (top-level await) carries the module
        // key on its controller: re-establish the module's ambient state (import
        // aliases, `import.meta`, active-module key, top-level var environment) for
        // this resume, since `run_generator` only restores the lexical/`this`/strict
        // context captured in the frame, not the module-evaluation context. Restored
        // after the step so the surrounding event-loop tick is unaffected.
        #[cfg(all(feature = "module", feature = "std"))]
        let module_ctx = self.enter_module_context_for_controller(controller);
        let promise = self
            .realm
            .get_property(controller, ASYNC_PROMISE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        self.async_step_inner(id, controller, how, promise);
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some(ctx) = module_ctx {
            self.exit_module_context(ctx);
        }
    }

    /// The body of [`Self::async_step`] (the generator drive and promise
    /// settlement), split out so the module-coroutine ambient state can wrap it.
    fn async_step_inner(
        &mut self,
        id: usize,
        controller: Handle,
        mut how: Resumption,
        promise: Option<Handle>,
    ) {
        // `how` is re-driven (rather than returned) when `Await`'s own
        // `PromiseResolve` completes abruptly: that abrupt completion belongs at
        // the `await` point inside the body, not to the caller.
        loop {
            match self.run_generator(id, how) {
                Ok(GenStep::Done(v)) => {
                    if let Some(p) = promise {
                        self.resolve_with(p, v);
                    }
                }
                Ok(GenStep::Awaited(v)) => {
                    // `await v`: PromiseResolve(v), then schedule the coroutine's
                    // resume as microtask reactions on its settlement.
                    let inner = match self.promise_resolve_checked(v) {
                        Ok(p) => p,
                        Err(e) => {
                            how = Resumption::Throw(self.abrupt_value(e, "await internal error"));
                            continue;
                        }
                    };
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
                Ok(GenStep::Yielded(_)) | Ok(GenStep::YieldedResult(_)) => {
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
            return;
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
        // Enter the generator's realm (derived from its captured scope's root): a
        // generator created in a `$262.createRealm()` realm — e.g. one built by a
        // cross-realm `GeneratorFunction` constructor — must resolve globals (a
        // sloppy undeclared read/write, `globalThis`, error intrinsics) against
        // *that* realm's global object, not the ambient one. `None` (main realm)
        // is the untouched fast path.
        let gen_realm = self.realm_of_scope(&scope);
        let realm_guard = self.enter_realm(gen_realm);
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
        self.leave_realm(realm_guard);
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
            Ok(GenStep::Yielded(_)) | Ok(GenStep::YieldedResult(_)) | Ok(GenStep::Awaited(_)) => {}
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
        // A coroutine body's statement boundaries are not GC-safe: the suspended
        // activation lives in `gen_frames`, which the safepoint does not trace.
        let saved_gc = core::mem::replace(&mut self.gc_ok, false);
        let result = self.gen_drive_inner(id, how, started);
        self.gc_ok = saved_gc;
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
            let Some(Step::YieldStar {
                iter,
                next,
                async_inner,
            }) = stack.pop()
            else {
                unreachable!()
            };
            match self.gen_yield_star_step(iter, how, next, async_inner, &mut stack, &mut values) {
                Ok(StepOut::Yield(v)) => {
                    self.store_machine(id, stack, values);
                    return Ok(GenStep::Yielded(v));
                }
                Ok(StepOut::YieldResult(v)) => {
                    self.store_machine(id, stack, values);
                    return Ok(GenStep::YieldedResult(v));
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
                Ok(StepOut::YieldResult(v)) => break Ok(GenStep::YieldedResult(v)),
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

/// Whether a statement contains a reachable top-level `await` (or `for await`)
/// not nested inside a function/class boundary. `await` and `yield` are the same
/// suspension points to the coroutine walker, so this is [`stmt_has_yield`] under
/// an await-focused name (used by the module top-level-await detector).
pub(crate) fn stmt_has_await(s: &Stmt) -> bool {
    stmt_has_yield(s)
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
        | Stmt::Import(_)
        | Stmt::Export(_) => false,
        // A class declaration with a `yield`-bearing computed member key must be
        // driven through the machine so the key suspends (`class C { get [yield](){} }`).
        Stmt::Class(c) => class_computed_key_has_yield(c),
        // An `await using` declaration is itself a suspension point of the async
        // coroutine: leaving the scope it belongs to performs `DisposeResources`,
        // whose step 4 `Await`s even when every resource was `null`/`undefined`
        // (so there was nothing to call). The declaration must therefore be lowered
        // into the machine — otherwise its whole enclosing block runs in one shot
        // through the eager walker, which cannot suspend, and the statements after
        // the block wrongly observe the same microtask.
        Stmt::Var(decl) => {
            matches!(decl.kind, crate::ast::VarDeclKind::AwaitUsing)
                || decl
                    .declarations
                    .iter()
                    .any(|d| d.init.as_ref().is_some_and(expr_has_yield))
        }
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
        // Boundaries: a nested function/arrow introduces its own context.
        Expr::Function(_) | Expr::Arrow(_) => false,
        // A class body is a boundary (method bodies, field initializers have their
        // own context), EXCEPT a *computed member key* (`class { get [yield]() {} }`)
        // which is evaluated in the enclosing generator context at definition time.
        Expr::Class(c) => class_computed_key_has_yield(c),
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
            // An accessor's function body is a boundary, but its *computed* key
            // (`get [yield]()`) is evaluated in the enclosing generator context.
            crate::ast::ObjectMember::Accessor { key, .. } => key_has_yield(key),
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

/// Whether any of a class's *computed member keys* (`[expr]` on a method or
/// field) contains a `yield`/`await` reachable in the enclosing generator/async
/// context. Method bodies, field initializers, and the `extends` heritage are
/// their own contexts / left to the eager fallback, so only the member keys are
/// examined here.
fn class_computed_key_has_yield(c: &crate::ast::Class) -> bool {
    c.body.iter().any(|m| match m {
        ClassMember::Method(m) => key_has_yield(&m.key),
        ClassMember::Field(f) => key_has_yield(&f.key),
        ClassMember::StaticBlock { .. } => false,
    })
}

/// Whether a call `callee(args)` can be reified by the generator step-machine so a
/// `yield` in an *argument* suspends. Restricted to *plain* calls: the callee must
/// be yield-free (its reference is evaluated eagerly, in the correct pre-argument
/// order) and must not be a method (`obj.m`), `super`, direct `eval`, or dynamic
/// `import` call — those need receiver / this / special-form handling and keep the
/// eager fallback. A plain call's `this` is `undefined`.
fn call_reifiable(callee: &Expr) -> bool {
    match callee {
        Expr::Super(_) | Expr::Member { .. } => false,
        Expr::Ident(id) if matches!(id.name.as_ref(), "eval" | "import") => false,
        _ => !expr_has_yield(callee),
    }
}

/// Whether a *method* call `recv.property(args)` can be reified by the step-machine
/// so an `await`/`yield` in an *argument* suspends. Restricted to a non-optional
/// call whose callee is a non-`super`, non-optional member with a *yield-free*
/// receiver and *yield-free* key (both evaluated eagerly, once, in the correct
/// pre-argument order); the `import.defer`/`import.source`/dynamic-`import` pseudo
/// member forms (receiver identifier `import`) are intercepted specially in the
/// eager path and keep it. Method calls whose receiver or key themselves suspend
/// (e.g. `(await x).m()`, `o[await k]()`) keep the eager fallback (documented).
fn method_call_reifiable(callee: &Expr) -> bool {
    let Expr::Member {
        object,
        property,
        optional,
        ..
    } = callee
    else {
        return false;
    };
    if *optional || matches!(&**object, Expr::Super(_)) {
        return false;
    }
    if matches!(&**object, Expr::Ident(id) if id.name.as_ref() == "import") {
        return false;
    }
    !expr_has_yield(object) && !key_has_yield(property)
}

// --- the stepping machine ----------------------------------------------------

impl<'a> Interp<'a> {
    /// Begins `DisposeResources` (7.5.5) for the coroutine scope being left, whose
    /// recorded `using` / `await using` disposers live in `self.current`.
    ///
    /// `restore` is the abrupt completion already being unwound (`None` when the
    /// scope is leaving normally); `scope` is the enclosing scope to reinstate
    /// afterwards; `label` is the label of the block being left, so a
    /// `break <label>` targeting it is consumed once disposal is done.
    ///
    /// A scope holding at least one **async-hint** (`await using`) resource is
    /// disposed by the resumable [`Step::Dispose`] machine — every async disposer's
    /// result becomes a real coroutine suspension, so the disposers of one scope
    /// are separated by microtask turns — and `Stepped` is returned: the caller
    /// must hand control back to the step loop, which re-raises the completion (or
    /// consumes the label) when disposal finishes.
    ///
    /// A scope with only synchronous `using` resources cannot suspend
    /// (`needsAwait` is only ever set by an async-hint resource), so it is disposed
    /// inline by the shared driver and its completion returned as `Inline`.
    fn gen_begin_dispose(
        &mut self,
        stack: &mut Vec<Step<'a>>,
        restore: Option<Completion>,
        scope: Option<Scope>,
        label: Option<String>,
    ) -> DisposeStart {
        let disposers = if self.current.has_disposers() {
            self.current.take_disposers()
        } else {
            alloc::vec::Vec::new()
        };
        if disposers.iter().any(|(_, _, is_async)| *is_async) {
            // A `Throw` being unwound is the run's initial *pending* completion (it
            // is what a later disposer throw suppresses); any other abrupt
            // completion is merely buffered and restored if nothing throws.
            let (pending, restore) = match restore {
                Some(Completion::Throw(e)) => (Some(e), None),
                other => (None, other),
            };
            stack.push(Step::Dispose(alloc::boxed::Box::new(DisposeState {
                rest: disposers,
                pending,
                restore,
                scope,
                label,
                needs_await: false,
                has_awaited: false,
            })));
            return DisposeStart::Stepped;
        }
        // Sync-only (or empty): run the shared driver in one shot. Only a `Throw`
        // participates in suppression; a `return`/`break`/`continue` is preserved
        // unless a disposer throws, in which case the throw replaces it.
        let mut completion = restore;
        if !disposers.is_empty() {
            let threaded = match &completion {
                Some(Completion::Throw(e)) => Err(ExecError::Throw(*e)),
                _ => Ok(NanBox::undefined()),
            };
            if let Err(e) = self.dispose_resources(disposers, threaded) {
                completion = Some(Completion::Throw(throw_value(e)));
            }
        }
        if let Some(s) = scope {
            self.current = s;
        }
        DisposeStart::Inline(consume_block_label(completion, label.as_deref()))
    }

    /// Runs (or resumes) a [`Step::Dispose`] state machine: disposes resources in
    /// reverse order, suspending the coroutine on each async-hint disposer's result
    /// (`Await`, spec step 3.e.ii) and on the `Await(undefined)` owed by step 3.d /
    /// step 4 when an async-hint resource had no dispose method. A throwing
    /// disposer is aggregated into `pending` as a `SuppressedError` chain.
    fn gen_dispose_step(
        &mut self,
        mut st: alloc::boxed::Box<DisposeState>,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        while let Some((value, method, is_async)) = st.rest.pop() {
            // Step 3.d: a sync-dispose resource reached while an `Await` is owed
            // pays it *before* its dispose method runs.
            if !is_async && st.needs_await && !st.has_awaited {
                st.needs_await = false;
                st.rest.push((value, method, is_async));
                Self::push_dispose_await(st, stack, values, NanBox::undefined());
                return Ok(StepOut::Continue);
            }
            if matches!(method.unpack(), Unpacked::Undefined | Unpacked::Null) {
                // Step 3.f: no dispose method — only reachable for the async-dispose
                // hint, which still owes an `Await(undefined)`.
                st.needs_await |= is_async;
                continue;
            }
            match self.call_with_this(method, value, &[]) {
                // Step 3.e.ii: an `await using` disposer's result is awaited as a
                // real suspension; the resumption re-enters this step.
                Ok(v) if is_async => {
                    st.has_awaited = true;
                    Self::push_dispose_await(st, stack, values, v);
                    return Ok(StepOut::Continue);
                }
                Ok(_) => {}
                Err(ExecError::Throw(e)) => self.dispose_suppress(&mut st, e),
                // A non-throw abrupt completion (engine-internal) aborts the run.
                Err(other) => return Err(GenAbrupt::Fatal(other)),
            }
        }
        // Step 4: an owed `Await(undefined)` that no real `Await` retired.
        if st.needs_await && !st.has_awaited {
            st.needs_await = false;
            st.has_awaited = true;
            Self::push_dispose_await(st, stack, values, NanBox::undefined());
            return Ok(StepOut::Continue);
        }
        self.gen_dispose_finish(*st)
    }

    /// Parks a [`Step::Dispose`] run on `operand`'s settlement: the `AwaitExpr`
    /// suspends the coroutine, `Discard` drops the fulfilment value the resumption
    /// pushes, and the re-pushed `Dispose` continues with the next resource. A
    /// *rejection* instead unwinds into this step's `gen_unwind` arm, which folds
    /// the thrown value into the pending completion and resumes the run.
    fn push_dispose_await(
        st: alloc::boxed::Box<DisposeState>,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
        operand: NanBox,
    ) {
        stack.push(Step::Dispose(st));
        stack.push(Step::Discard);
        stack.push(Step::AwaitExpr);
        values.push(operand);
    }

    /// Folds a disposer's thrown value `e` into a dispose run's pending completion:
    /// the newest error becomes `.error` and the prior completion `.suppressed` of a
    /// `SuppressedError` chain.
    fn dispose_suppress(&mut self, st: &mut DisposeState, e: NanBox) {
        st.pending = Some(match st.pending.take() {
            None => e,
            Some(prev) => self.make_suppressed_error(e, prev),
        });
    }

    /// Concludes a [`Step::Dispose`] run: restores the enclosing scope, then
    /// re-raises the resulting completion — a disposer throw (which replaces any
    /// non-throw completion the scope was unwinding), else the buffered completion,
    /// unless it is a `break` targeting the block that was being left.
    fn gen_dispose_finish(&mut self, st: DisposeState) -> StepResult {
        let DisposeState {
            pending,
            restore,
            scope,
            label,
            ..
        } = st;
        if let Some(s) = scope {
            self.current = s;
        }
        let completion = match pending {
            Some(e) => Some(Completion::Throw(e)),
            None => restore,
        };
        match consume_block_label(completion, label.as_deref()) {
            None => Ok(StepOut::Continue),
            Some(c) => Err(abrupt_of(c)),
        }
    }

    /// Whether an object-literal `Property` member is the `{ __proto__: v }`
    /// prototype setter: the unquoted-identifier `__proto__` key, non-shorthand,
    /// non-`Function` value (a quoted / computed / shorthand / method / `Function`
    /// form makes an ordinary own `__proto__` data property instead). Mirrors the
    /// eager object-literal evaluator's discrimination.
    fn is_proto_member(key: &PropertyKey, value: &Expr, shorthand: bool) -> bool {
        !shorthand
            && !matches!(value, Expr::Function(_))
            && matches!(key, PropertyKey::Ident(s) if &**s == "__proto__")
    }

    /// Sets a `{ __proto__: v }` member's effect on `target`: `[[Prototype]]` is
    /// set only when `v` is an Object or `null`; any other primitive is ignored
    /// (the object keeps its prototype and gains no own `__proto__` property).
    fn obj_apply_proto(&mut self, target: Handle, v: NanBox) {
        if matches!(v.unpack(), Unpacked::Null) {
            self.realm.set_object_proto(target, None);
        } else if self.is_object_value(v)
            && let Some(p) = v.as_handle().map(Handle::from_raw)
        {
            self.realm.set_object_proto(target, Some(p));
        }
    }

    /// Defines a data / method / function-valued object-literal property on
    /// `target` under storage key `k` with the already-evaluated value `v` —
    /// applying NamedEvaluation (SetFunctionName) and, for a concise method,
    /// `[[HomeObject]]` / method-flag / source handling exactly as the inline
    /// object-literal evaluator (`eval`'s `Expr::Object` arm) does. Shared by that
    /// eager walker's stepped counterpart so a computed key that suspended on a
    /// `yield` (its result being `k`) defines the member identically.
    fn obj_define_property_member(
        &mut self,
        target: Handle,
        member: &'a ObjectMember,
        k: String,
        v: NanBox,
    ) -> Result<(), ExecError> {
        let ObjectMember::Property {
            key,
            value,
            method,
            span: member_span,
            ..
        } = member
        else {
            unreachable!()
        };
        let (method, member_span) = (*method, *member_span);
        let value: &'a Expr = value;
        // A method / function-valued property is named after its key when
        // otherwise anonymous (a Symbol computed key → `[description]`).
        if matches!(value, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_)) {
            match key {
                PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                    self.set_fn_name(v, s);
                }
                PropertyKey::Computed(_) => {
                    let params: &[Param] = match value {
                        Expr::Function(f) => &f.params,
                        _ => &[],
                    };
                    if let Some(name) = self.method_display_name(&k, MethodKind::Method)
                        && v.as_handle()
                            .map(Handle::from_raw)
                            .is_some_and(|h| self.fn_name_unset(h))
                    {
                        if matches!(value, Expr::Class(_)) {
                            let nm = self.new_str(&name);
                            if let Some(h) = v.as_handle().map(Handle::from_raw) {
                                self.realm.clear_readonly_property(h, "name");
                                self.realm.set_property(h, "name", nm);
                                self.realm.mark_hidden(h, "name");
                                self.realm.set_readonly_property(h, "name");
                            }
                        } else {
                            self.install_method_meta(v, &name, params);
                        }
                    }
                }
                _ => {}
            }
        }
        // A concise method records this object as its `[[HomeObject]]` (so
        // `super.x` resolves through the object's prototype) and is a method.
        if method
            && matches!(value, Expr::Function(_))
            && let Some(fv) = v.as_handle().map(Handle::from_raw)
        {
            self.set_fn_source(v, member_span);
            self.realm
                .set_hidden_property(fv, HOME_OBJECT, NanBox::handle(target.to_raw()));
            if let Some((fid, _)) = self.realm.function_at(fv) {
                self.functions[fid as usize].is_method = true;
                if !self.functions[fid as usize].is_generator {
                    self.demote_fn_prototype(fv);
                }
            }
        }
        self.realm.set_property(target, &k, v);
        Ok(())
    }

    /// Defines a `get`/`set` accessor object-literal member (the member at
    /// `members[idx]`, or any `ObjectMember::Accessor`) on `target` under storage
    /// key `k` — creating the accessor function with `[[HomeObject]]`, method flag,
    /// source, and SetFunctionName (`"get x"` / `"set x"`) exactly as the eager
    /// evaluator does, then pairing it via `define_accessor`.
    fn gen_define_accessor_member(
        &mut self,
        member: &'a ObjectMember,
        target: Handle,
        k: String,
    ) -> Result<(), GenAbrupt> {
        let ObjectMember::Accessor {
            is_getter,
            value,
            span,
            ..
        } = member
        else {
            unreachable!()
        };
        let f = self.make_function(&value.params, Body::Block(&value.body), false, false);
        self.set_fn_source(f, *span);
        if let Some(fh) = f.as_handle().map(Handle::from_raw) {
            if let Some((fid, _)) = self.realm.function_at(fh) {
                self.functions[fid as usize].is_method = true;
            }
            self.demote_fn_prototype(fh);
            self.realm
                .set_hidden_property(fh, HOME_OBJECT, NanBox::handle(target.to_raw()));
            let kind = if *is_getter {
                MethodKind::Get
            } else {
                MethodKind::Set
            };
            if let Some(nm) = self.method_display_name(&k, kind)
                && self.fn_name_unset(fh)
            {
                let len = value
                    .params
                    .iter()
                    .take_while(|p| p.default.is_none() && !p.rest)
                    .count() as u32;
                self.install_fn_name_length(fh, &nm, len);
            }
        }
        if *is_getter {
            self.realm
                .define_accessor(target, &k, f, NanBox::undefined());
        } else {
            self.realm
                .define_accessor(target, &k, NanBox::undefined(), f);
        }
        Ok(())
    }

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
                    // Block / body sequence finished normally: dispose any
                    // `using` resources recorded in the just-completed scope
                    // (`self.current`), in reverse order, before restoring the
                    // enclosing scope. A throwing disposer becomes a throw; a
                    // scope holding an `await using` resource is disposed by the
                    // resumable `Step::Dispose` machine (one suspension per async
                    // disposer). The block's own label is not consumable here (a
                    // normal exit carries no `break`), so it is not passed on.
                    return match self.gen_begin_dispose(stack, None, scope, None) {
                        DisposeStart::Stepped | DisposeStart::Inline(None) => Ok(StepOut::Continue),
                        DisposeStart::Inline(Some(c)) => Err(abrupt_of(c)),
                    };
                }
                let stmt = &body[idx];
                // Re-push the continuation of this sequence first.
                stack.push(Step::Seq {
                    body,
                    idx: idx + 1,
                    scope,
                    label: label.clone(),
                });
                // The sequence's `label` belongs to the *block* (`outer: { … }`),
                // not to its elements: it is consumed by the unwinder's `Step::Seq`
                // arm, which is what makes `break outer` leave the block. Passing it
                // down would let an element statement swallow that break — the
                // `Flow::Break(l) if l == label` arm of `gen_exec_stmt` — and the
                // sequence would then wrongly carry on with the following statement.
                self.gen_exec_stmt(stmt, stack, values, &None)
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
            Step::ForAwaitLoop {
                ih,
                next,
                left,
                body,
                label,
                async_inner,
            } => {
                // Pull one `next()` and park on the result / value `await`. The loop
                // step is NOT re-pushed until the body is about to run (in
                // `ForAwaitBind`), so a rejection *before* the body (a rejected
                // `next()` result, or the sync value-unwrap) does not re-close via
                // this loop marker — only a body-abrupt completion does.
                if async_inner {
                    // Native async iterator: `next()` returns a promise; await it,
                    // then read `done`/`value` from the settled result in `ForAwaitBind`.
                    let iter_val = NanBox::handle(ih.to_raw());
                    let res = self
                        .call_with_this(next, iter_val, &[])
                        .map_err(GenAbrupt::from)?;
                    stack.push(Step::ForAwaitBind {
                        ih,
                        next,
                        left,
                        body,
                        label,
                        async_inner: true,
                    });
                    return Ok(StepOut::Await(res));
                }
                // Sync iterator wrapped as AsyncFromSyncIterator: the sync result
                // object is turned by `%AsyncFromSyncIteratorPrototype%.next` into a
                // *promise* of an iterator-result object. Awaiting that promise is
                // what the loop does — two microtask turns per iteration (the
                // continuation's value unwrap, then this `Await`), which is
                // observable and must not be collapsed into one.
                let p = self
                    .async_from_sync_next(ih, next)
                    .map_err(GenAbrupt::from)?;
                stack.push(Step::ForAwaitBind {
                    ih,
                    next,
                    left,
                    body,
                    label,
                    async_inner: false,
                });
                Ok(StepOut::Await(NanBox::handle(p.to_raw())))
            }
            Step::ForAwaitBind {
                ih,
                next,
                left,
                body,
                label,
                async_inner,
            } => {
                // The awaited operand is on the value stack: an iterator-result
                // object either way (a native async iterator's own `next()` promise,
                // or the AsyncFromSyncIterator continuation's).
                let awaited = values.pop().unwrap_or(NanBox::undefined());
                let value = {
                    let Some(rh) = awaited.as_handle().map(Handle::from_raw) else {
                        return Err(GenAbrupt::Throw(
                            self.make_type_error("iterator result is not an object"),
                        ));
                    };
                    let done = self.read_member(rh, "done").map_err(GenAbrupt::from)?;
                    if self.realm.truthy(done) {
                        return Ok(StepOut::Continue);
                    }
                    self.read_member(rh, "value").map_err(GenAbrupt::from)?
                };
                // Re-push the loop marker for the next iteration BELOW the body, so a
                // body break/return/throw unwinds into it (IteratorClose), and a
                // matching `continue` re-loops.
                stack.push(Step::ForAwaitLoop {
                    ih,
                    next,
                    left,
                    body,
                    label: label.clone(),
                    async_inner,
                });
                // Bind the loop variable in a fresh per-iteration scope.
                let child = self.current.child();
                let prev = core::mem::replace(&mut self.current, child);
                stack.push(Step::PopScope { scope: prev });
                // `for await ([ x = yield ] of …)` — an assignment-target pattern
                // whose default/computed-key yields: destructure through the step
                // machine (so the yield suspends), then run the body via a deferred
                // step. (Mirrors the plain `for-of` `ForEach` path.)
                if let ForLeft::Target(expr) = left
                    && expr_has_yield(expr)
                {
                    stack.push(Step::RunLoopBody { body, label });
                    stack.push(Step::Destructure {
                        target: expr,
                        value,
                    });
                    return Ok(StepOut::Continue);
                }
                match left {
                    ForLeft::Decl { target, .. } => {
                        self.bind_pattern(target, value).map_err(GenAbrupt::from)?;
                    }
                    ForLeft::Target(expr) => {
                        self.assign_destructure(expr, value)
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
            Step::Finally { pending } => Err(abrupt_of(pending)),
            Step::Discard => {
                values.pop();
                Ok(StepOut::Continue)
            }
            Step::ReturnValue => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                // 14.10.1 step 3: in an async *generator* (GetGeneratorKind() is
                // async) an explicit `return <expr>` awaits the operand before the
                // return completion — so `return undefined` costs a tick that a
                // bare `return;` (and falling off the end) does not.
                if self.gen_is_async {
                    stack.push(Step::ReturnAwaited);
                    return Ok(StepOut::Await(v));
                }
                Err(GenAbrupt::Return(v))
            }
            Step::ReturnAwaited => {
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
            Step::PrivateIn { name } => {
                let obj = values.pop().unwrap_or(NanBox::undefined());
                // §13.10.1: the RHS of `#x in rhs` must be an Object.
                if !self.is_object_value(obj) {
                    let m = self.new_str(
                        "Cannot use 'in' operator to check for a private name in a non-object",
                    );
                    return Err(GenAbrupt::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let key = self.private_access_key(name);
                let present = obj.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.has_own(h, &key) || self.realm.accessor(h, &key).is_some()
                });
                values.push(NanBox::boolean(present));
                Ok(StepOut::Continue)
            }
            Step::ArrayLit { elements, idx, acc } => {
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
            #[cfg(all(feature = "module", feature = "std"))]
            Step::DynamicImportOptions { arguments } => match arguments.get(1) {
                Some(crate::ast::Argument::Item(e)) => {
                    stack.push(Step::DynamicImportCall { has_options: true });
                    self.gen_eval_expr(e, stack, values)
                }
                _ => {
                    stack.push(Step::DynamicImportCall { has_options: false });
                    Ok(StepOut::Continue)
                }
            },
            #[cfg(all(feature = "module", feature = "std"))]
            Step::DynamicImportCall { has_options } => {
                let options = if has_options { values.pop() } else { None };
                let spec = values.pop().unwrap_or(NanBox::undefined());
                let p = self.dynamic_import_values(spec, options);
                values.push(p);
                Ok(StepOut::Continue)
            }
            Step::TemplateLit { tpl, idx, mut acc } => {
                let Some(quasi) = tpl.quasis.get(idx) else {
                    values.push(self.new_str_bytes(acc));
                    return Ok(StepOut::Continue);
                };
                match &quasi.cooked {
                    Some(cooked) => acc.extend_from_slice(cooked),
                    // An invalid escape is allowed only in a *tagged* template.
                    None => {
                        let m = self.new_str("Invalid escape sequence in template literal");
                        return Err(GenAbrupt::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
                    }
                }
                match tpl.expressions.get(idx) {
                    Some(e) => {
                        stack.push(Step::TemplateAppend { tpl, idx, acc });
                        self.gen_eval_expr(e, stack, values)
                    }
                    None => {
                        stack.push(Step::TemplateLit {
                            tpl,
                            idx: idx + 1,
                            acc,
                        });
                        Ok(StepOut::Continue)
                    }
                }
            }
            Step::TemplateAppend { tpl, idx, mut acc } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                let bytes = self.coerce_to_string_bytes(v).map_err(GenAbrupt::from)?;
                acc.extend_from_slice(&bytes);
                stack.push(Step::TemplateLit {
                    tpl,
                    idx: idx + 1,
                    acc,
                });
                Ok(StepOut::Continue)
            }
            Step::CallArgs {
                func,
                arguments,
                idx,
                acc,
            } => {
                if idx >= arguments.len() {
                    let r = self
                        .call_with_this(func, NanBox::undefined(), &acc)
                        .map_err(GenAbrupt::from)?;
                    values.push(r);
                    return Ok(StepOut::Continue);
                }
                let (expr, spread) = match &arguments[idx] {
                    Argument::Item(e) => (e, false),
                    Argument::Spread(e) => (e, true),
                };
                stack.push(Step::CallArgAppend {
                    func,
                    arguments,
                    idx,
                    acc,
                    spread,
                });
                self.gen_eval_expr(expr, stack, values)
            }
            Step::CallArgAppend {
                func,
                arguments,
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
                stack.push(Step::CallArgs {
                    func,
                    arguments,
                    idx: idx + 1,
                    acc,
                });
                Ok(StepOut::Continue)
            }
            Step::MethodCallArgs {
                recv,
                property,
                arguments,
                idx,
                acc,
            } => {
                if idx >= arguments.len() {
                    let r = self
                        .call_member_dispatch(recv, property, false, &acc)
                        .map_err(GenAbrupt::from)?;
                    values.push(r);
                    return Ok(StepOut::Continue);
                }
                let (expr, spread) = match &arguments[idx] {
                    Argument::Item(e) => (e, false),
                    Argument::Spread(e) => (e, true),
                };
                stack.push(Step::MethodCallArgAppend {
                    recv,
                    property,
                    arguments,
                    idx,
                    acc,
                    spread,
                });
                self.gen_eval_expr(expr, stack, values)
            }
            Step::MethodCallArgAppend {
                recv,
                property,
                arguments,
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
                stack.push(Step::MethodCallArgs {
                    recv,
                    property,
                    arguments,
                    idx: idx + 1,
                    acc,
                });
                Ok(StepOut::Continue)
            }
            Step::AssignMemberStatic { base, property } => {
                // The RHS value is on top of the stack; it is the assignment's
                // result, so leave it in place after performing the write.
                let v = values.last().copied().unwrap_or(NanBox::undefined());
                self.assign_member(base, property, v)
                    .map_err(GenAbrupt::from)?;
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
                    ObjectMember::Property {
                        key,
                        value,
                        shorthand,
                        ..
                    } => {
                        // `{ __proto__: v }` — the unquoted-ident, non-shorthand,
                        // non-`Function` form sets `[[Prototype]]`; evaluate the
                        // value (it may yield) then apply it as the prototype.
                        if Self::is_proto_member(key, value, *shorthand) {
                            stack.push(Step::ObjectLitProtoSet {
                                members,
                                idx,
                                target,
                            });
                            return self.gen_eval_expr(value, stack, values);
                        }
                        match key {
                            PropertyKey::Computed(kexpr) => {
                                // Evaluate the computed key first (it may yield);
                                // the value is evaluated once the key completes.
                                stack.push(Step::ObjectLitPropKey {
                                    members,
                                    idx,
                                    target,
                                });
                                self.gen_eval_expr(kexpr, stack, values)
                            }
                            _ => {
                                let k = self.eval_prop_key(key).map_err(GenAbrupt::from)?;
                                stack.push(Step::ObjectLitPropVal {
                                    members,
                                    idx,
                                    target,
                                    key: k,
                                });
                                self.gen_eval_expr(value, stack, values)
                            }
                        }
                    }
                    ObjectMember::Spread { value, .. } => {
                        stack.push(Step::ObjectLitSpread {
                            members,
                            idx,
                            target,
                        });
                        self.gen_eval_expr(value, stack, values)
                    }
                    ObjectMember::Accessor { key, .. } => match key {
                        PropertyKey::Computed(kexpr) => {
                            stack.push(Step::ObjectLitAccessorKey {
                                members,
                                idx,
                                target,
                            });
                            self.gen_eval_expr(kexpr, stack, values)
                        }
                        _ => {
                            let k = self.eval_prop_key(key).map_err(GenAbrupt::from)?;
                            self.gen_define_accessor_member(&members[idx], target, k)?;
                            stack.push(Step::ObjectLit {
                                members,
                                idx: idx + 1,
                                target,
                            });
                            Ok(StepOut::Continue)
                        }
                    },
                }
            }
            Step::ObjectLitPropKey {
                members,
                idx,
                target,
            } => {
                let kv = values.pop().unwrap_or(NanBox::undefined());
                let key = self.coerce_property_key(kv).map_err(GenAbrupt::from)?;
                let ObjectMember::Property { value, .. } = &members[idx] else {
                    unreachable!()
                };
                stack.push(Step::ObjectLitPropVal {
                    members,
                    idx,
                    target,
                    key,
                });
                self.gen_eval_expr(value, stack, values)
            }
            Step::ObjectLitPropVal {
                members,
                idx,
                target,
                key,
            } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                self.obj_define_property_member(target, &members[idx], key, v)
                    .map_err(GenAbrupt::from)?;
                stack.push(Step::ObjectLit {
                    members,
                    idx: idx + 1,
                    target,
                });
                Ok(StepOut::Continue)
            }
            Step::ObjectLitAccessorKey {
                members,
                idx,
                target,
            } => {
                let kv = values.pop().unwrap_or(NanBox::undefined());
                let key = self.coerce_property_key(kv).map_err(GenAbrupt::from)?;
                self.gen_define_accessor_member(&members[idx], target, key)?;
                stack.push(Step::ObjectLit {
                    members,
                    idx: idx + 1,
                    target,
                });
                Ok(StepOut::Continue)
            }
            Step::ObjectLitProtoSet {
                members,
                idx,
                target,
            } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                self.obj_apply_proto(target, v);
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
                self.object_spread_into(target, v)
                    .map_err(GenAbrupt::from)?;
                stack.push(Step::ObjectLit {
                    members,
                    idx: idx + 1,
                    target,
                });
                Ok(StepOut::Continue)
            }
            Step::ClassKeys { class, idx, keys } => {
                // Advance to the next computed member key; evaluate it through the
                // machine so a `yield`/`await` in it suspends. Non-computed and
                // static-block members are skipped (their keys need no evaluation).
                let mut i = idx;
                while i < class.body.len() {
                    let key = match &class.body[i] {
                        ClassMember::Method(m) => &m.key,
                        ClassMember::Field(f) => &f.key,
                        ClassMember::StaticBlock { .. } => {
                            i += 1;
                            continue;
                        }
                    };
                    if let PropertyKey::Computed(kexpr) = key {
                        stack.push(Step::ClassKeyStore {
                            class,
                            idx: i,
                            keys,
                        });
                        return self.gen_eval_expr(kexpr, stack, values);
                    }
                    i += 1;
                }
                // All computed keys evaluated: build the class with them substituted.
                let v = self
                    .make_class_with_keys(class, Some(&keys))
                    .map_err(GenAbrupt::from)?;
                values.push(v);
                Ok(StepOut::Continue)
            }
            Step::ClassKeyStore {
                class,
                idx,
                mut keys,
            } => {
                let kv = values.pop().unwrap_or(NanBox::undefined());
                let k = self.coerce_property_key(kv).map_err(GenAbrupt::from)?;
                // A `static` element whose computed key evaluates to "prototype" is a
                // TypeError, raised here (during key evaluation) as the spec requires.
                let is_static = match &class.body[idx] {
                    ClassMember::Method(m) => m.is_static,
                    ClassMember::Field(f) => f.is_static,
                    ClassMember::StaticBlock { .. } => false,
                };
                if is_static && k == "prototype" {
                    let m =
                        self.new_str("Classes may not have a static property named 'prototype'");
                    return Err(GenAbrupt::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                keys.insert(idx, k);
                stack.push(Step::ClassKeys {
                    class,
                    idx: idx + 1,
                    keys,
                });
                Ok(StepOut::Continue)
            }
            Step::ClassDeclBind { class } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                if let Some(id) = &class.id {
                    self.current.declare(&id.name, v);
                }
                Ok(StepOut::Continue)
            }
            #[cfg(all(feature = "module", feature = "std"))]
            Step::ExportDefaultBind { expr } => {
                let v = values.pop().unwrap_or(NanBox::undefined());
                // NamedEvaluation: an anonymous function/class/arrow gets "default".
                if matches!(
                    expr,
                    Expr::Function(crate::ast::Function { id: None, .. })
                        | Expr::Class(Class { id: None, .. })
                        | Expr::Arrow(_)
                ) {
                    self.set_fn_name(v, "default");
                }
                self.current.declare_const(super::module::DEFAULT_LOCAL, v);
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
                        stack.push(Step::Destructure {
                            target: e,
                            value: v,
                        });
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
                        stack.push(Step::Destructure {
                            target: e,
                            value: h,
                        });
                        Ok(StepOut::Continue)
                    }
                }
            }
            Step::DestructureArrayIter {
                elements,
                idx,
                ih,
                done,
            } => {
                if idx >= elements.len() {
                    // All elements consumed; the `DestructureArrayClose` guard below
                    // performs the normal-completion `IteratorClose`.
                    return Ok(StepOut::Continue);
                }
                let iterator = NanBox::handle(ih.to_raw());
                let step_val = |me: &mut Self, done: &alloc::rc::Rc<core::cell::Cell<bool>>| {
                    if done.get() {
                        return Ok(NanBox::undefined());
                    }
                    match me.dstr_iter_step(ih, iterator) {
                        Ok(Some(v)) => Ok(v),
                        Ok(None) => {
                            done.set(true);
                            Ok(NanBox::undefined())
                        }
                        Err(e) => {
                            // A throwing `next()` finishes the iterator (no close).
                            done.set(true);
                            Err(e)
                        }
                    }
                };
                match &elements[idx] {
                    ArrayElement::Hole => {
                        let _ = step_val(self, &done).map_err(GenAbrupt::from)?;
                        stack.push(Step::DestructureArrayIter {
                            elements,
                            idx: idx + 1,
                            ih,
                            done,
                        });
                        Ok(StepOut::Continue)
                    }
                    ArrayElement::Item(e) => {
                        // `obj[key]` whose base or computed key `yield`s: the target
                        // reference is evaluated **before** the `IteratorStep` (spec
                        // order). Evaluate `obj` then `key` through the step machine;
                        // a `yield` resumed with `return()`/`throw` then closes the
                        // iterator without calling `next`. (A defaulted target is an
                        // `Expr::Assign`, so this narrow `Member` match excludes it.)
                        if let Expr::Member {
                            object,
                            property: PropertyKey::Computed(key),
                            ..
                        } = e
                            && (expr_has_yield(object) || expr_has_yield(key))
                        {
                            stack.push(Step::DestructureArrayIter {
                                elements,
                                idx: idx + 1,
                                ih,
                                done: done.clone(),
                            });
                            stack.push(Step::DestructureElemKey { key, ih, done });
                            return self.gen_eval_expr(object, stack, values);
                        }
                        let v = step_val(self, &done).map_err(GenAbrupt::from)?;
                        stack.push(Step::DestructureArrayIter {
                            elements,
                            idx: idx + 1,
                            ih,
                            done,
                        });
                        stack.push(Step::Destructure {
                            target: e,
                            value: v,
                        });
                        Ok(StepOut::Continue)
                    }
                    ArrayElement::Spread(e) => {
                        // `...obj[key]` whose base or computed key `yield`s: per spec
                        // the rest *reference* is evaluated **before** the iterator is
                        // drained. Evaluate `obj` then `key` through the step machine
                        // (so a `yield` there suspends), and only then drain + assign —
                        // so a `return()`/`throw` resumption closes the iterator
                        // without pulling (and without calling a throwing `next`).
                        if let Expr::Member {
                            object,
                            property: PropertyKey::Computed(key),
                            ..
                        } = e
                            && (expr_has_yield(object) || expr_has_yield(key))
                        {
                            stack.push(Step::DestructureArrayIter {
                                elements,
                                idx: idx + 1,
                                ih,
                                done: done.clone(),
                            });
                            stack.push(Step::DestructureRestKey { key, ih, done });
                            return self.gen_eval_expr(object, stack, values);
                        }
                        // `...rest` (no yielding reference): collect all remaining
                        // values into a fresh array, exhausting the iterator.
                        let mut rest = Vec::new();
                        while !done.get() {
                            match self.dstr_iter_step(ih, iterator) {
                                Ok(Some(v)) => rest.push(v),
                                Ok(None) => done.set(true),
                                Err(e) => {
                                    done.set(true);
                                    return Err(GenAbrupt::from(e));
                                }
                            }
                        }
                        let arr = NanBox::handle(self.realm.new_array(rest).to_raw());
                        stack.push(Step::DestructureArrayIter {
                            elements,
                            idx: idx + 1,
                            ih,
                            done,
                        });
                        stack.push(Step::Destructure {
                            target: e,
                            value: arr,
                        });
                        Ok(StepOut::Continue)
                    }
                }
            }
            Step::DestructureRestKey { key, ih, done } => {
                // `obj` is on the value stack (kept); evaluate the computed key.
                stack.push(Step::DestructureRestSet { ih, done });
                self.gen_eval_expr(key, stack, values)
            }
            Step::DestructureRestSet { ih, done } => {
                let key = values.pop().unwrap_or(NanBox::undefined());
                let objval = values.pop().unwrap_or(NanBox::undefined());
                let iterator = NanBox::handle(ih.to_raw());
                let mut rest = Vec::new();
                while !done.get() {
                    match self.dstr_iter_step(ih, iterator) {
                        Ok(Some(v)) => rest.push(v),
                        Ok(None) => done.set(true),
                        Err(e) => {
                            done.set(true);
                            return Err(GenAbrupt::from(e));
                        }
                    }
                }
                let arr = NanBox::handle(self.realm.new_array(rest).to_raw());
                if let Some(raw) = objval.as_handle() {
                    self.assign_member_value(Handle::from_raw(raw), key, arr)
                        .map_err(GenAbrupt::from)?;
                }
                Ok(StepOut::Continue)
            }
            Step::DestructureElemKey { key, ih, done } => {
                // `obj` is on the value stack (kept); evaluate the computed key.
                stack.push(Step::DestructureElemSet { ih, done });
                self.gen_eval_expr(key, stack, values)
            }
            Step::DestructureElemSet { ih, done } => {
                let key = values.pop().unwrap_or(NanBox::undefined());
                let objval = values.pop().unwrap_or(NanBox::undefined());
                let iterator = NanBox::handle(ih.to_raw());
                // The single `IteratorStep` for this element (spec order: after the
                // reference was evaluated above).
                let v = if done.get() {
                    NanBox::undefined()
                } else {
                    match self.dstr_iter_step(ih, iterator) {
                        Ok(Some(v)) => v,
                        Ok(None) => {
                            done.set(true);
                            NanBox::undefined()
                        }
                        Err(e) => {
                            done.set(true);
                            return Err(GenAbrupt::from(e));
                        }
                    }
                };
                if let Some(raw) = objval.as_handle() {
                    self.assign_member_value(Handle::from_raw(raw), key, v)
                        .map_err(GenAbrupt::from)?;
                }
                Ok(StepOut::Continue)
            }
            Step::DestructureArrayClose { ih, done } => {
                // Normal completion: close a not-yet-exhausted iterator. A `return`
                // that yields a non-Object throws a TypeError (propagated).
                if !done.get() {
                    self.iterator_close(ih).map_err(GenAbrupt::from)?;
                }
                Ok(StepOut::Continue)
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
                        stack.push(Step::Destructure {
                            target: tgt,
                            value: v,
                        });
                        Ok(StepOut::Continue)
                    }
                    ObjectMember::Spread { value: tgt, .. } => {
                        let obj = self.realm.new_object();
                        // CopyDataProperties (proxy-aware, symbol-aware) with the
                        // already-destructured keys excluded.
                        self.copy_data_properties(obj, src, &used)
                            .map_err(GenAbrupt::from)?;
                        let h = NanBox::handle(obj.to_raw());
                        stack.push(Step::DestructureObjectMember {
                            members,
                            idx: idx + 1,
                            src,
                            used,
                        });
                        stack.push(Step::Destructure {
                            target: tgt,
                            value: h,
                        });
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
            Step::MemberRead { property } => {
                let obj = values.pop().unwrap_or(NanBox::undefined());
                let v = self
                    .read_member_of(obj, property, false)
                    .map_err(GenAbrupt::from)?;
                values.push(v);
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
                    let (iter, async_inner) =
                        self.gen_delegate_iter(operand).map_err(GenAbrupt::from)?;
                    // GetIterator caches [[NextMethod]] once (GetV(iter, "next"));
                    // this single read is the only "get next" the delegation makes.
                    let next = self.read_member(iter, "next").map_err(GenAbrupt::from)?;
                    self.gen_yield_star_step(
                        iter,
                        Resumption::Next(NanBox::undefined()),
                        next,
                        async_inner,
                        stack,
                        values,
                    )
                } else if self.gen_is_async {
                    // `AsyncGeneratorYield(operand)` = Await(operand) then
                    // GeneratorYield. Await first (so `yield <rejected promise>`
                    // rejects the `next()` promise, and a fulfilled promise yields
                    // its value); the pushed `AsyncYield` yields the settled value.
                    stack.push(Step::AsyncYield);
                    Ok(StepOut::Await(operand))
                } else {
                    // Plain `yield v`: suspend. On resume, the injected value is
                    // pushed by `gen_drive` and becomes this expression's result.
                    Ok(StepOut::Yield(operand))
                }
            }
            Step::AsyncYield => {
                // The awaited operand is on the stack; surface it as the yielded value.
                let awaited = values.pop().unwrap_or(NanBox::undefined());
                Ok(StepOut::Yield(awaited))
            }
            Step::YieldStar {
                iter,
                next,
                async_inner,
            } => {
                // A `YieldStar` on top at resume is intercepted by `gen_drive`
                // (which forwards the resumption to the inner iterator); reaching
                // it here means it was not the suspension point, so pump it with
                // `next(undefined)`.
                self.gen_yield_star_step(
                    iter,
                    Resumption::Next(NanBox::undefined()),
                    next,
                    async_inner,
                    stack,
                    values,
                )
            }
            Step::YieldStarResult { iter, next, kind } => {
                // The awaited native-async inner result promise settled; its result
                // object is on the value stack. Process it.
                let res = values.pop().unwrap_or(NanBox::undefined());
                self.gen_yield_star_process(iter, next, kind, res, stack, values)
            }
            Step::YieldStarAfterValue {
                iter,
                next,
                async_inner,
                done,
                kind,
                has_close_guard,
            } => {
                // The awaited inner `value` is on the value stack.
                let value = values.pop().unwrap_or(NanBox::undefined());
                // The value `Await` fulfilled: drop the close guard beneath (it only
                // fires if that `Await` rejects).
                if has_close_guard && matches!(stack.last(), Some(Step::YieldStarClose { .. })) {
                    stack.pop();
                }
                if !done {
                    // AsyncGeneratorYield: re-yield the settled value and stay in
                    // delegation (a `YieldStar` beneath is the next suspend point).
                    stack.push(Step::YieldStar {
                        iter,
                        next,
                        async_inner,
                    });
                    Ok(StepOut::Yield(value))
                } else if matches!(kind, YsKind::Return) {
                    // A forwarded `return` whose inner iterator completed (or had no
                    // `return` method): the outer generator returns the value.
                    Err(GenAbrupt::Return(value))
                } else {
                    // `next`/`throw` completed the delegation: the `yield*`
                    // expression evaluates to the (awaited) inner value.
                    values.push(value);
                    Ok(StepOut::Continue)
                }
            }
            // A close guard reached on the normal path (its value `Await` did not
            // reject) is a no-op — `YieldStarAfterValue` already removed it, but a
            // stray one is harmless.
            Step::YieldStarClose { .. } => Ok(StepOut::Continue),
            Step::AwaitExpr => {
                // `await v`: suspend the async coroutine on `v`'s settlement. On
                // resume, `gen_drive` pushes the fulfilment value (or routes a
                // rejection through the unwinder), which becomes this expression's
                // result.
                let operand = values.pop().unwrap_or(NanBox::undefined());
                Ok(StepOut::Await(operand))
            }
            Step::Dispose(st) => self.gen_dispose_step(st, stack, values),
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
        // handled by the unwinder, honoring an enclosing label). A module
        // top-level `import`/`export` is excluded: `exec` rejects it as an
        // unsupported statement, so it must reach the dedicated match arms below
        // (its own declaration payload runs there) even when yield-free.
        if !stmt_has_yield(stmt) && !matches!(stmt, Stmt::Import(_) | Stmt::Export(_)) {
            // A directly-labeled yield-free loop needs its label for inner
            // break/continue; `exec` reads `pending_label`.
            if let Some(l) = label
                && is_loop(stmt)
            {
                self.pending_label = Some(l.clone());
            }
            return match self.exec(stmt) {
                Ok(Flow::Normal(_)) => Ok(StepOut::Continue),
                // 14.10.1 step 3: in an async generator, `return <Expression>;`
                // awaits its operand before the return completion propagates. The
                // statement ran in one shot, so the `Await` is re-entered here.
                Ok(Flow::Return(v)) if self.gen_is_async && self.return_had_expr => {
                    stack.push(Step::ReturnAwaited);
                    Ok(StepOut::Await(v))
                }
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
                // `for await` drives the async-iterator protocol **lazily**: one
                // `next()` per iteration, parked on `await`, closing the iterator
                // (`IteratorClose`) on a `break`/`return`/`throw` out of the body —
                // so an infinite source with a `break` terminates instead of being
                // eagerly drained. A plain `for-of` still materializes an eager list
                // of values (a documented simplification for yield-bearing bodies).
                if *is_await {
                    let (ih, next, async_inner) = self
                        .for_await_iter_record(iterable)
                        .map_err(GenAbrupt::from)?;
                    stack.push(Step::ForAwaitLoop {
                        ih,
                        next,
                        left,
                        body,
                        label: label.clone(),
                        async_inner,
                    });
                    return Ok(StepOut::Continue);
                }
                let items = self.iterate_values(iterable).map_err(GenAbrupt::from)?;
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
                // object is captured lexically). `gen_exec_stmt` only *pushes* the
                // body's steps, so the scope must stay current until they have all
                // run — the `PopScope` step beneath them restores it, on the
                // suspended path as well as the synchronous one.
                let child = self.current.child_with(obj);
                let saved = core::mem::replace(&mut self.current, child);
                stack.push(Step::PopScope { scope: saved });
                self.gen_exec_stmt(body, stack, values, &None)
            }
            // Module top-level statements reached when a module body with
            // top-level `await` is driven on this coroutine engine. An `import`
            // declaration is a no-op at run time (its bindings were wired at link
            // time). An `export` evaluates its inner declaration/default payload
            // (the export *slot* was wired at link time) one-shot — an `await` in
            // an exported initializer runs eagerly, which is a rare corner the
            // suspending path does not reify.
            Stmt::Import(_) => Ok(StepOut::Continue),
            #[cfg(all(feature = "module", feature = "std"))]
            Stmt::Export(decl) => {
                // `export default <expr>` / `export <declaration>` whose payload
                // contains a reachable `await` is driven through the machine, so
                // the suspension is real — an eager `await` here would resume the
                // rest of the module body ahead of the already-queued microtasks.
                // Everything else keeps the one-shot path (the export *slot* was
                // wired at link time).
                let stepped = match decl {
                    crate::ast::ExportDecl::Default { declaration, .. } => match &**declaration {
                        Stmt::Expr { expression, .. } if expr_has_yield(expression) => {
                            stack.push(Step::ExportDefaultBind { expr: expression });
                            stack.push(Step::EvalThen { expr: expression });
                            true
                        }
                        _ => false,
                    },
                    crate::ast::ExportDecl::Decl { declaration, .. }
                        if stmt_has_yield(declaration) =>
                    {
                        return self.gen_exec_stmt(declaration, stack, values, &None);
                    }
                    _ => false,
                };
                if !stepped {
                    self.exec_export(decl).map_err(GenAbrupt::from)?;
                }
                Ok(StepOut::Continue)
            }
            // `class C { get [yield](){} }` (declaration) — step the class's
            // computed member keys (so a `yield`/`await` in one suspends), build the
            // class, then bind it to the class name. Reached only when a computed
            // member key has a reachable suspension.
            Stmt::Class(class) => {
                stack.push(Step::ClassDeclBind { class });
                stack.push(Step::ClassKeys {
                    class,
                    idx: 0,
                    keys: alloc::collections::BTreeMap::new(),
                });
                Ok(StepOut::Continue)
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
        | Step::ForEach { label, .. }
        | Step::ForAwaitLoop { label, .. } => label.as_ref(),
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
                // A *user* iterator is driven one `IteratorStep` per element (so an
                // infinite iterator terminates, a per-element default `yield`s at the
                // right moment, and `IteratorClose`/`return()` runs on abrupt exit).
                // Built-in iterables (arrays/strings/Sets) and generators — which
                // always terminate — take the eager value-list path. Mirrors the
                // non-generator `assign_destructure`.
                if let Some(ih) = self.for_of_get_iterator(value).map_err(GenAbrupt::from)?
                    && self.realm.get_property(ih, GEN_BUF).is_none()
                {
                    let done = alloc::rc::Rc::new(core::cell::Cell::new(false));
                    // The guard sits below the element steps: normal flow reaches it
                    // last (IteratorClose on a not-yet-done iterator); an abrupt
                    // unwind pops the element steps and hits it in `gen_unwind`.
                    stack.push(Step::DestructureArrayClose {
                        ih,
                        done: done.clone(),
                    });
                    stack.push(Step::DestructureArrayIter {
                        elements,
                        idx: 0,
                        ih,
                        done,
                    });
                    return Ok(StepOut::Continue);
                }
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
            // `#name in <rhs>` — the ergonomic brand check where the RHS may hide a
            // `yield`/`await` (`#field in (yield)`). Step the RHS so it suspends,
            // then perform the brand check once it completes.
            Expr::Binary {
                op: crate::ast::BinaryOp::In,
                left,
                right,
                ..
            } if matches!(&**left, Expr::PrivateName(..)) => {
                let Expr::PrivateName(name, _) = &**left else {
                    unreachable!()
                };
                stack.push(Step::PrivateIn { name });
                self.gen_eval_expr(right, stack, values)
            }
            // `left <op> right` where a yield hides in an operand. The `#x in obj`
            // brand check (a private left operand) is handled above.
            Expr::Binary {
                op, left, right, ..
            } if !matches!(&**left, Expr::PrivateName(..)) => {
                // Evaluate left, then right, then combine: push the combiner and
                // a step that evaluates `right` after `left` is on the stack.
                stack.push(Step::BinaryOp { op: *op });
                stack.push(Step::EvalThen { expr: right });
                self.gen_eval_expr(left, stack, values)
            }
            // `import(yield)` / `import('', yield)` — step both arguments so a
            // suspension parks *before* the import is issued: an abrupt completion
            // at that point must skip the import entirely.
            #[cfg(all(feature = "module", feature = "std"))]
            Expr::Call {
                callee,
                arguments,
                optional: false,
                ..
            } if matches!(&**callee, Expr::Ident(id) if id.name.as_ref() == "import") => {
                stack.push(Step::DynamicImportOptions { arguments });
                match arguments.first() {
                    Some(crate::ast::Argument::Item(e)) => self.gen_eval_expr(e, stack, values),
                    _ => {
                        values.push(NanBox::undefined());
                        Ok(StepOut::Continue)
                    }
                }
            }
            // `` `a${yield}b` `` — append each quasi and step each substitution, so
            // a `yield`/`await` in any of them suspends mid-string.
            Expr::Template(tpl) => {
                stack.push(Step::TemplateLit {
                    tpl,
                    idx: 0,
                    acc: Vec::new(),
                });
                Ok(StepOut::Continue)
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
            // `{ [yield k]: v, get [yield](){}, ...yield s }` — step through members
            // so a `yield`/`await` in a computed key, a data value, or a spread
            // operand suspends. Every member form is handled (data / method /
            // accessor / `__proto__` / spread); each computed key and value is driven
            // through the machine, then the member is defined with the same
            // semantics as the eager walker. Reached only when a member has a
            // reachable suspension (the `!expr_has_yield` short-circuit above).
            Expr::Object { members, .. } => {
                let target = self.realm.new_object();
                stack.push(Step::ObjectLit {
                    members,
                    idx: 0,
                    target,
                });
                Ok(StepOut::Continue)
            }
            // `class C { get [yield](){} }` (expression) — evaluate the class's
            // computed member keys through the machine so a `yield`/`await` in one
            // suspends, then build the class. Reached only when a computed member key
            // has a reachable suspension.
            Expr::Class(class) => {
                stack.push(Step::ClassKeys {
                    class,
                    idx: 0,
                    keys: alloc::collections::BTreeMap::new(),
                });
                Ok(StepOut::Continue)
            }
            // `base.p = <value-with-yield>` / `base['p'] = …` — a static-key member
            // assignment whose RHS may yield, and whose base is yield-free. Evaluate
            // the base now (correct pre-RHS order), then step the RHS so a yield in
            // it suspends; the assignment is performed only once the RHS completes
            // NORMALLY, so a `return`/`throw` at the yield unwinds without writing.
            // Computed keys, `super.p`, and private names take the eager fallback.
            Expr::Assign {
                op: crate::ast::AssignOp::Assign,
                target,
                value,
                ..
            } if matches!(&**target,
                Expr::Member { object, property, optional: false, .. }
                    if !matches!(&**object, Expr::Super(_))
                        && !expr_has_yield(object)
                        && matches!(property,
                            PropertyKey::Ident(_) | PropertyKey::Str(_) | PropertyKey::Number(_))) =>
            {
                let Expr::Member {
                    object, property, ..
                } = &**target
                else {
                    unreachable!()
                };
                let base = self.eval(object).map_err(GenAbrupt::from)?;
                if let Some(h) = base.as_handle().map(Handle::from_raw) {
                    stack.push(Step::AssignMemberStatic { base: h, property });
                }
                // A primitive base mirrors `assign_to`: the write is skipped; only
                // the RHS is evaluated (for its value / suspension).
                self.gen_eval_expr(value, stack, values)
            }
            // A *plain* call `f(yield)` whose callee is yield-free and is not a
            // method/`super`/direct-`eval`/`import` call: evaluate the callee now
            // (its reference cannot reach a `yield`), then step through the
            // arguments so a `yield` in any argument suspends. A plain call's `this`
            // is `undefined`. Method calls (`obj.m(yield)`), `super(...)`, direct
            // eval, and `new` still take the eager fallback (documented follow-up).
            Expr::Call {
                callee,
                arguments,
                optional,
                ..
            } if !*optional && call_reifiable(callee) => {
                let func = self.eval(callee).map_err(GenAbrupt::from)?;
                stack.push(Step::CallArgs {
                    func,
                    arguments,
                    idx: 0,
                    acc: Vec::new(),
                });
                Ok(StepOut::Continue)
            }
            // A method call `recv.m(await x)` / `assert.sameValue((await x).v, …)`
            // whose callee member is yield-free (receiver + key) but whose
            // arguments contain an `await`/`yield`. Evaluate the receiver eagerly
            // (correct pre-argument order), run the nullish-base TypeError check
            // (also before arguments), then step the arguments so a suspension
            // parks — completing with `call_member_dispatch` (identical `this`-
            // binding / built-in / own-property semantics to the eager path). The
            // receiver is non-optional and non-`super`/`import`, and the call is
            // non-optional, so no optional short-circuit reaches the step machine.
            Expr::Call {
                callee,
                arguments,
                optional: false,
                ..
            } if method_call_reifiable(callee) => {
                let Expr::Member {
                    object, property, ..
                } = &**callee
                else {
                    unreachable!()
                };
                let recv = self.eval(object).map_err(GenAbrupt::from)?;
                self.method_recv_check(recv, property, false)
                    .map_err(GenAbrupt::from)?;
                stack.push(Step::MethodCallArgs {
                    recv,
                    property,
                    arguments,
                    idx: 0,
                    acc: Vec::new(),
                });
                Ok(StepOut::Continue)
            }
            // `(<expr-with-await/yield>).prop` — a member READ whose *object*
            // suspends. Evaluate the object step-by-step (so the `await`/`yield`
            // parks), then complete the property read with `read_member_of` (same
            // getter/primitive semantics as the eager walker). Restricted to a
            // non-`super`, non-optional base with a yield-free (static or
            // suspension-free computed) key; other shapes take the fallback.
            Expr::Member {
                object,
                property,
                optional: false,
                ..
            } if !matches!(&**object, Expr::Super(_)) && !key_has_yield(property) => {
                stack.push(Step::MemberRead { property });
                self.gen_eval_expr(object, stack, values)
            }
            // Any other yield-bearing expression shape (e.g. `obj.m(yield b)`,
            // a computed access with a yielding key, compound/destructuring
            // assignment) is not individually reified; fall back to one-shot eval.
            // The yield-free fast path above means this only runs for genuinely
            // yield-bearing complex operands, which are a documented follow-up.
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
        _values: &mut [NanBox],
        mut completion: Completion,
    ) -> Result<Option<Result<GenStep, ExecError>>, ExecError> {
        loop {
            let Some(step) = stack.pop() else {
                // Nothing caught it: the body itself is unwinding out. Dispose any
                // body-level `using` resources (recorded in the frame scope,
                // `self.current`) before the generator completes abnormally,
                // aggregating a disposer throw into the completion. With an
                // `await using` among them the disposal is stepped (it suspends per
                // resource) and re-raises the completion when it is done — landing
                // back here with the resources already taken.
                match self.gen_begin_dispose(stack, Some(completion), None, None) {
                    DisposeStart::Stepped => return Ok(None),
                    DisposeStart::Inline(Some(c)) => completion = c,
                    DisposeStart::Inline(None) => {
                        return Ok(Some(Ok(GenStep::Done(NanBox::undefined()))));
                    }
                }
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
                    match self.gen_begin_dispose(stack, Some(completion), Some(scope), None) {
                        DisposeStart::Stepped | DisposeStart::Inline(None) => return Ok(None),
                        DisposeStart::Inline(Some(c)) => completion = c,
                    }
                }
                Step::Seq { scope, label, .. } => {
                    // The block scope (`self.current`) is leaving abruptly: dispose
                    // its `using` resources, aggregating against the completion. A
                    // labeled break/continue targeting this block's label is consumed
                    // once disposal is done (a labeled *block* only matches `break`) —
                    // `gen_begin_dispose` does that on both the inline and the stepped
                    // path, so an `await using` scope keeps the same label semantics
                    // across its suspensions.
                    match self.gen_begin_dispose(stack, Some(completion), scope, label) {
                        DisposeStart::Stepped | DisposeStart::Inline(None) => return Ok(None),
                        DisposeStart::Inline(Some(c)) => completion = c,
                    }
                }
                Step::Dispose(mut st) => {
                    // A dispose run parked on an async disposer's `Await` was
                    // resumed abruptly: a rejected disposal promise is folded into
                    // the run's pending throw (SuppressedError chain), and any other
                    // injected completion is buffered as the one to restore. Either
                    // way `DisposeResources` must still finish, so the run resumes.
                    match completion {
                        Completion::Throw(e) => self.dispose_suppress(&mut st, e),
                        other => {
                            if st.pending.is_none() {
                                st.restore = Some(other);
                            }
                        }
                    }
                    stack.push(Step::Dispose(st));
                    return Ok(None);
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
                Step::ForAwaitLoop { .. } => {
                    // The `for await` loop marker is being unwound by a body-abrupt
                    // completion. A matching `continue` re-loops WITHOUT closing; any
                    // other completion (break/return/throw, or a non-matching label)
                    // runs `IteratorClose` (14.7.5: not-LoopContinues → close), then
                    // consumes a matching break or keeps propagating.
                    let (ih, label) = if let Step::ForAwaitLoop { ih, label, .. } = &step {
                        (*ih, label.clone())
                    } else {
                        unreachable!()
                    };
                    match &completion {
                        Completion::Continue(None) => {
                            stack.push(step);
                            return Ok(None);
                        }
                        Completion::Continue(Some(l)) if label.as_deref() == Some(l.as_str()) => {
                            stack.push(step);
                            return Ok(None);
                        }
                        _ => {}
                    }
                    // IteratorClose: under a throw completion the original throw takes
                    // precedence and any close error is swallowed (spec step 6); under
                    // break/return a close error propagates as a throw.
                    let is_throw = matches!(completion, Completion::Throw(_));
                    if let Err(e) = self.iterator_close(ih)
                        && !is_throw
                    {
                        completion = Completion::Throw(throw_value(e));
                    }
                    match &completion {
                        Completion::Break(None) => return Ok(None),
                        Completion::Break(Some(l)) if label.as_deref() == Some(l.as_str()) => {
                            return Ok(None);
                        }
                        // Return / Throw / a non-matching break keep unwinding.
                        _ => {}
                    }
                }
                Step::SwitchRegion => {
                    if matches!(completion, Completion::Break(None)) {
                        return Ok(None);
                    }
                }
                Step::YieldStarClose { iter } => {
                    // AsyncFromSyncIteratorContinuation's `onRejected`: the non-done
                    // value `Await` rejected — close the sync iterator (call its
                    // `return`) for its side effects. Per IteratorClose under a throw
                    // completion (spec step 6), the original rejection takes
                    // precedence over anything `return` yields, so its result — a
                    // throw *or* a non-object — is discarded.
                    if let Completion::Throw(_) = &completion {
                        let _ = self.iterator_close(iter);
                    }
                }
                Step::DestructureArrayClose { ih, done } => {
                    // A destructuring element completed abruptly (a default's
                    // `yield` got `return()`/`throw`, or a target assignment threw)
                    // while the iterator was not yet done: run `IteratorClose`. A
                    // `throw` completion takes precedence over any error from
                    // `return()`; a non-throw completion is replaced by the close's
                    // throw (e.g. `return()` returning a non-Object → TypeError).
                    if !done.get()
                        && let Err(e) = self.iterator_close(ih)
                        && !matches!(completion, Completion::Throw(_))
                    {
                        completion = Completion::Throw(throw_value(e));
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
        _values: &mut [NanBox],
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

/// Re-raises an unwinder [`Completion`] as a stepping [`GenAbrupt`].
fn abrupt_of(completion: Completion) -> GenAbrupt {
    match completion {
        Completion::Return(v) => GenAbrupt::Return(v),
        Completion::Throw(e) => GenAbrupt::Throw(e),
        Completion::Break(l) => GenAbrupt::Break(l),
        Completion::Continue(l) => GenAbrupt::Continue(l),
    }
}

/// Consumes a `break <label>` that targets the labeled *block* being left
/// (`outer: { … }`): such a break stops here, so `None` (resume normally) is
/// returned. Any other completion — and any completion at all when the block is
/// unlabeled — passes through untouched.
fn consume_block_label(completion: Option<Completion>, label: Option<&str>) -> Option<Completion> {
    match (&completion, label) {
        (Some(Completion::Break(Some(l))), Some(name)) if l == name => None,
        _ => completion,
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
    /// Obtains the inner iterator object for `yield* operand` together with whether
    /// it is a *native async* iterator (obtained via `[Symbol.asyncIterator]`, so
    /// its `next()`/`return()`/`throw()` return promises). A sync iterator (`false`)
    /// is driven as an AsyncFromSyncIterator: each result value is unwrapped
    /// (awaited) per step in `gen_yield_star_step`.
    fn gen_delegate_iter(&mut self, operand: NanBox) -> Result<(Handle, bool), ExecError> {
        // GetIterator reads `[Symbol.iterator]` off the *value*, so a primitive
        // operand resolves the method through its wrapper prototype
        // (`Boolean.prototype[Symbol.iterator] = …; yield* true`). Only `null` /
        // `undefined` have no wrapper. The method is still called with the
        // original primitive as its receiver.
        let h = match operand.as_handle() {
            Some(raw) => Handle::from_raw(raw),
            None if !(operand.is_null() || operand.is_undefined()) => {
                match self.coerce_to_object(operand).as_handle() {
                    Some(raw) => Handle::from_raw(raw),
                    None => return Err(self.type_error("yield* operand is not iterable")),
                }
            }
            None => return Err(self.type_error("yield* operand is not iterable")),
        };
        // In an async generator, `yield*` uses GetIterator(operand, async): a
        // callable `[Symbol.asyncIterator]` is used directly; a non-callable (but
        // present) one is a TypeError. An absent `[Symbol.asyncIterator]` falls
        // through to the sync `[Symbol.iterator]` (AsyncFromSyncIterator).
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
                let ih = iterator
                    .as_handle()
                    .map(Handle::from_raw)
                    .ok_or_else(|| self.type_error("yield* iterator is not an object"))?;
                return Ok((ih, true));
            }
        }
        // A user object exposing `[Symbol.iterator]` is driven through its real
        // iterator (so an infinite/user generator delegates lazily).
        if let Some(f) = self.find_iterator_fn(h)?
            && f.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            let iterator = self.call_with_this(f, operand, &[])?;
            let ih = iterator
                .as_handle()
                .map(Handle::from_raw)
                .ok_or_else(|| self.type_error("yield* iterator is not an object"))?;
            return Ok((ih, false));
        }
        // A built-in iterable (array / string / Map / Set) whose `Symbol.iterator`
        // is built-in dispatch (not a readable property): materialize an eager
        // iterator over its values — `yield*` still drives it one `.next()` at a
        // time, preserving the surfaced-value sequence.
        let vals = self.iterate_values(operand)?;
        let iter = self.make_generator(vals);
        let ih = iter
            .as_handle()
            .map(Handle::from_raw)
            .ok_or_else(|| self.type_error("yield* operand is not iterable"))?;
        Ok((ih, false))
    }

    /// GetIterator(iterable, async) for a `for await` loop: the iterator handle,
    /// its cached `[[NextMethod]]`, and whether it is a native async iterator
    /// (`[Symbol.asyncIterator]`) or a sync iterator wrapped as AsyncFromSyncIterator.
    pub(crate) fn for_await_iter_record(
        &mut self,
        iterable: NanBox,
    ) -> Result<(Handle, NanBox, bool), ExecError> {
        // A native async iterable (async generator object or callable
        // `[Symbol.asyncIterator]`) is driven directly.
        if let Some(ih) = self.async_iterator_of(iterable)? {
            let next = self.read_member(ih, "next")?;
            return Ok((ih, next, true));
        }
        // Otherwise GetIterator(iterable, sync) and wrap as AsyncFromSyncIterator.
        let Some(h) = iterable.as_handle().map(Handle::from_raw) else {
            let m = self.new_str("is not async iterable");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        if let Some(f) = self.find_iterator_fn(h)?
            && f.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            let it = self.call_with_this(f, iterable, &[])?;
            let ih = it
                .as_handle()
                .map(Handle::from_raw)
                .ok_or_else(|| self.type_error("iterator is not an object"))?;
            let next = self.read_member(ih, "next")?;
            return Ok((ih, next, false));
        }
        // A built-in iterable (array / string / Map / Set) with built-in-dispatch
        // `Symbol.iterator`: materialize a real iterator over its values (finite).
        let vals = self.iterate_values(iterable)?;
        let iter = self.make_generator(vals);
        let ih = iter
            .as_handle()
            .map(Handle::from_raw)
            .ok_or_else(|| self.type_error("is not async iterable"))?;
        let next = self.read_member(ih, "next")?;
        Ok((ih, next, false))
    }

    /// Advances a `yield*` delegation one step: calls the inner iterator with
    /// `how` (a `next(v)` resume, a forwarded `return(v)`, or a forwarded
    /// `throw(e)`).
    ///
    /// A **sync** generator surfaces the inner result object verbatim
    /// (GeneratorYield(innerResult)) on a non-done result and completes on a done
    /// one. An **async** generator implements AsyncGeneratorYield's `yield*`:
    /// awaiting the (native-async) inner result promise, unwrapping (awaiting) the
    /// inner `value` before re-yielding, and awaiting a missing-`return`'s value —
    /// each parking the coroutine on the microtask queue (via [`Step::YieldStar`]-
    /// family steps) rather than draining eagerly.
    fn gen_yield_star_step(
        &mut self,
        iter: Handle,
        how: Resumption,
        next_method: NanBox,
        async_inner: bool,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        let iter_val = NanBox::handle(iter.to_raw());
        let (method_name, arg, kind) = match how {
            Resumption::Next(v) => ("next", v, YsKind::Next),
            Resumption::Return(v) => ("return", v, YsKind::Return),
            Resumption::Throw(e) => ("throw", e, YsKind::Throw),
        };
        let is_next = matches!(kind, YsKind::Next);
        // `next` is the cached [[NextMethod]] (read once at acquisition, not every
        // step); `return`/`throw` are fetched per-use (GetMethod each time), and an
        // absent one has special handling below.
        let method = if is_next {
            next_method
        } else {
            self.read_member(iter, method_name)
                .map_err(GenAbrupt::from)?
        };
        let absent = method.is_undefined() || method.is_null();
        if absent {
            match how {
                // No inner `return`: the delegation completes and the outer `return`
                // continues with the forwarded value. In an async generator the
                // value is first `Await`ed (14.4.14: "Return ? Await(received)").
                Resumption::Return(v) => {
                    if self.gen_is_async {
                        stack.push(Step::YieldStarAfterValue {
                            iter,
                            next: next_method,
                            async_inner,
                            done: true,
                            kind: YsKind::Return,
                            has_close_guard: false,
                        });
                        return Ok(StepOut::Await(v));
                    }
                    return Err(GenAbrupt::Return(v));
                }
                // No inner `throw`: per spec (14.4.14 5.b.iii), IteratorClose the
                // inner iterator with a *normal* completion first — giving it a
                // chance to clean up — then throw a TypeError. If `return` itself is
                // abrupt (getting or calling it throws, or it returns a non-object),
                // that abrupt completion propagates *instead of* the TypeError.
                Resumption::Throw(_) => {
                    self.iterator_close(iter).map_err(GenAbrupt::from)?;
                    return Err(GenAbrupt::Throw(
                        self.make_type_error("The iterator does not provide a 'throw' method"),
                    ));
                }
                Resumption::Next(_) => {
                    // `yield*` requires a callable `next` (IteratorNext →
                    // Call(next)); an absent/non-callable one is a TypeError.
                    return Err(GenAbrupt::Throw(
                        self.make_type_error("The iterator does not provide a 'next' method"),
                    ));
                }
            }
        }
        let res = self
            .call_with_this(method, iter_val, &[arg])
            .map_err(GenAbrupt::from)?;
        if !self.gen_is_async {
            // --- Sync generator `yield*` (unchanged) --------------------------
            let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                return Err(GenAbrupt::Throw(
                    self.make_type_error("iterator result is not an object"),
                ));
            };
            let done = self.read_member(rh, "done").map_err(GenAbrupt::from)?;
            if self.realm.truthy(done) {
                let value = self.read_member(rh, "value").map_err(GenAbrupt::from)?;
                if matches!(how, Resumption::Return(_)) {
                    return Err(GenAbrupt::Return(value));
                }
                values.push(value);
                return Ok(StepOut::Continue);
            }
            stack.push(Step::YieldStar {
                iter,
                next: next_method,
                async_inner,
            });
            return Ok(StepOut::YieldResult(res));
        }
        // --- Async generator `yield*` (AsyncGeneratorYield) -------------------
        if async_inner {
            // IteratorNext for a native async iterator returns a promise: await it,
            // then process the settled result object in `YieldStarResult`.
            stack.push(Step::YieldStarResult {
                iter,
                next: next_method,
                kind,
            });
            return Ok(StepOut::Await(res));
        }
        // A sync inner iterator wrapped as async (AsyncFromSyncIterator): its result
        // is a plain object; read `done`/`value`, then `Await` the value (unwrap /
        // AsyncGeneratorYield) before re-yielding or completing.
        let Some(rh) = res.as_handle().map(Handle::from_raw) else {
            return Err(GenAbrupt::Throw(
                self.make_type_error("iterator result is not an object"),
            ));
        };
        let done = self.read_member(rh, "done").map_err(GenAbrupt::from)?;
        let done = self.realm.truthy(done);
        let value = self.read_member(rh, "value").map_err(GenAbrupt::from)?;
        // AsyncFromSyncIteratorContinuation closes the sync iterator if the value
        // `Await` rejects — but only when the result was not done (`onRejected` is
        // undefined for a done result). Push a close guard beneath the after-value
        // step so a rejection during the value `Await` runs IteratorClose.
        if !done {
            stack.push(Step::YieldStarClose { iter });
        }
        stack.push(Step::YieldStarAfterValue {
            iter,
            next: next_method,
            async_inner,
            done,
            kind,
            has_close_guard: !done,
        });
        Ok(StepOut::Await(value))
    }

    /// Processes a native-async inner iterator's already-awaited `next`/`return`/
    /// `throw` result object (`res`) for an async `yield*` (from
    /// [`Step::YieldStarResult`]). Non-done: `Await` the value (AsyncGeneratorYield)
    /// then re-yield. Done: the `yield*` value is the inner value (a forwarded
    /// `return` returns it from the outer generator; `next`/`throw` continue).
    fn gen_yield_star_process(
        &mut self,
        iter: Handle,
        next_method: NanBox,
        kind: YsKind,
        res: NanBox,
        stack: &mut Vec<Step<'a>>,
        values: &mut Vec<NanBox>,
    ) -> StepResult {
        // Only reached for a native-async inner iterator (from `YieldStarResult`).
        let async_inner = true;
        let Some(rh) = res.as_handle().map(Handle::from_raw) else {
            return Err(GenAbrupt::Throw(
                self.make_type_error("iterator result is not an object"),
            ));
        };
        let done = self.read_member(rh, "done").map_err(GenAbrupt::from)?;
        let done = self.realm.truthy(done);
        let value = self.read_member(rh, "value").map_err(GenAbrupt::from)?;
        if !done {
            // AsyncGeneratorYield(value) does NOT itself `Await` the value — a
            // native async iterator's promise value is re-yielded verbatim (only a
            // sync-wrapped iterator unwraps it, inside AsyncFromSyncIteratorContinuation,
            // handled in `gen_yield_star_step`). So re-yield the raw value.
            stack.push(Step::YieldStar {
                iter,
                next: next_method,
                async_inner,
            });
            return Ok(StepOut::Yield(value));
        }
        // Done: no further await of the value (the whole result was already awaited).
        if matches!(kind, YsKind::Return) {
            return Err(GenAbrupt::Return(value));
        }
        values.push(value);
        Ok(StepOut::Continue)
    }

    fn make_type_error(&mut self, msg: &str) -> NanBox {
        let m = self.new_str(msg);
        self.make_error(N_TYPE_ERROR, Some(m))
    }
}
