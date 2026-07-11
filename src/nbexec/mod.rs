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
//!   fresh child scope;
//! - **exceptions** (`try`/`catch`/`finally`/`throw`); and
//! - a **starter stdlib**: native globals (`Math`, `String`/`Number`/`parseInt`)
//!   and built-in String/Array methods, including the higher-order
//!   `map`/`filter`/`reduce`/`forEach` that call back into closures.
//!
//! The *full* stdlib port and folding back into the bytecode VM are the
//! remaining migration work. Pure, safe `alloc`-only Rust.

use crate::ast::{
    Argument, ArrayElement, ArrayPatternElement, Arrow, ArrowBody, AssignOp, BinaryOp,
    BindingTarget, Class, ClassMember, Expr, ForInit, Function, Ident, LogicalOp, MethodKind,
    ObjectMember, Param, Program, PropertyKey, Stmt, UnaryOp, VarDecl,
};
use crate::env::Scope;
use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use crate::realm::{Realm, RealmIntrinsics};
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
    /// An optional-chain short-circuit: a `?.` link found a nullish base. It
    /// propagates (past intervening non-optional links) up to the enclosing
    /// `Expr::OptChain` boundary, which turns it into `undefined`. It is *not* a
    /// throw, so `try`/`catch` never sees it.
    OptShortCircuit,
    /// A **proper tail call** (strict-mode PTC): a `return f(args)` in genuine
    /// tail position hands the resolved callee/`this`/args back to the enclosing
    /// [`Interp::invoke`] instead of calling recursively. `invoke`'s trampoline
    /// re-dispatches it in place, so unbounded tail recursion runs in O(1) native
    /// stack. Produced only where `self.tail_pos` holds (a strict, non-async
    /// function body, outside any `try` Block), so it never escapes the `invoke`
    /// that must consume it, and a `catch` (which only matches `Throw`) ignores it.
    TailCall {
        /// The already-evaluated callee (a function value) to invoke.
        callee: NanBox,
        /// The `this` binding for the call (`undefined` for a plain call).
        this_val: NanBox,
        /// The already-evaluated arguments.
        args: alloc::vec::Vec<NanBox>,
    },
}

/// The control-flow outcome of a statement.
#[derive(Clone)]
pub(crate) enum Flow {
    /// Fell through normally, carrying the last expression value (for `run`).
    Normal(NanBox),
    /// A `return` (value).
    Return(NanBox),
    /// A `break`, optionally targeting a label. The carried `NanBox` is the
    /// completion value propagated by `UpdateEmpty` (the empty-completion
    /// sentinel for a bare `break`); it becomes the value of the breakable
    /// statement the `break` resolves to (`x: { 1; break x; }` evaluates to 1).
    Break(Option<String>, NanBox),
    /// A `continue`, optionally targeting a label. Carries its `UpdateEmpty`
    /// completion value like [`Flow::Break`].
    Continue(Option<String>, NanBox),
}

/// What a loop should do with a `Flow` produced by its body, given the loop's
/// own label (if any).
enum LoopAction {
    /// Proceed to the next iteration (fall through to the update).
    Next,
    /// Stop this loop.
    Stop,
    /// Not for this loop — bubble it up to an enclosing loop / labeled block.
    Propagate(Flow),
}

/// Folds an empty completion value to `undefined` (spec `UpdateEmpty(_,
/// undefined)`), used where a construct must never surface the empty-completion
/// sentinel (an `if`, `try`, `with`, …). Applies to the value carried by an
/// abrupt `break`/`continue` too: e.g. the `break` in `if (c) { break; }`
/// resolves with value `undefined`, not the surrounding list's value.
fn empty_to_undefined(flow: Flow) -> Flow {
    match flow {
        Flow::Normal(v) if v.is_empty_completion() => Flow::Normal(NanBox::undefined()),
        Flow::Break(l, v) if v.is_empty_completion() => Flow::Break(l, NanBox::undefined()),
        Flow::Continue(l, v) if v.is_empty_completion() => Flow::Continue(l, NanBox::undefined()),
        other => other,
    }
}

/// Classifies a loop body's `flow` *and* threads its completion value into the
/// loop's running value `v` (spec ForBodyEvaluation / loop evaluation):
/// - a `Normal` or caught `continue` updates `v` when its value is non-empty;
/// - a caught `break` sets `v` to `UpdateEmpty(break, v)` and stops;
/// - anything else propagates unchanged.
fn loop_step(flow: Flow, label: &Option<String>, v: &mut NanBox) -> LoopAction {
    let matches = |l: &Option<String>| l.is_none() || l.as_deref() == label.as_deref();
    match flow {
        Flow::Normal(bv) => {
            if !bv.is_empty_completion() {
                *v = bv;
            }
            LoopAction::Next
        }
        Flow::Continue(l, bv) if matches(&l) => {
            if !bv.is_empty_completion() {
                *v = bv;
            }
            LoopAction::Next
        }
        Flow::Break(l, bv) if matches(&l) => {
            *v = update_empty(bv, *v);
            LoopAction::Stop
        }
        other => LoopAction::Propagate(other),
    }
}

/// The spec `UpdateEmpty(completionValue, fallback)`: an empty completion value
/// takes the surrounding StatementList's accumulated value, otherwise keeps its
/// own.
fn update_empty(value: NanBox, fallback: NanBox) -> NanBox {
    if value.is_empty_completion() {
        fallback
    } else {
        value
    }
}

/// The body of a registered function: a block, or a concise arrow expression.
#[derive(Clone, Copy)]
pub(crate) enum Body<'a> {
    Block(&'a [Stmt]),
    Expr(&'a Expr),
}

/// A registered function definition (its AST, held by the interpreter; the heap
/// closure stores only an index into the table plus the captured scope).
#[derive(Clone, Copy)]
pub(crate) struct FnDef<'a> {
    params: &'a [Param],
    body: Body<'a>,
    is_async: bool,
    /// Whether this is a generator (`function*`) — run eagerly into an iterator.
    is_generator: bool,
    /// Whether this is an arrow function (no own `arguments` binding).
    is_arrow: bool,
    /// Whether the function is strict (its own `"use strict"`, or defined inside
    /// strict code) — strict functions keep an `undefined`/`null` `this`.
    is_strict: bool,
    /// The function's name (`fn.name`); empty for anonymous functions.
    name: &'a str,
    /// The class this is a method of (for `super.method()`), if any.
    home_class: Option<u32>,
    /// Whether the home is entered as a *static* method, so `super.x` resolves
    /// against the superclass's static members rather than its prototype's.
    home_static: bool,
    /// The lexically-enclosing class at *definition* time — used ONLY to resolve
    /// private names (`#x`), which are lexically scoped and visible inside nested
    /// ordinary functions (where `home_class`/`super` are intentionally `None`).
    /// For a method this is its `home_class`; for an ordinary function it is the
    /// class textually enclosing it (or `None` outside any class).
    lexical_class: Option<u32>,
    /// Whether this is a concise method / accessor (object-literal `{m(){}}` /
    /// `get`/`set`) — such functions have no `[[Construct]]` (`new m()` throws).
    /// Set post-creation like `is_arrow`. Class methods are detected via
    /// `home_class` instead.
    is_method: bool,
    /// Whether this function was defined lexically inside a class field
    /// initializer / static block. Only consulted for arrows (which are
    /// transparent to the ContainsArguments early error on a nested direct
    /// `eval`); a non-arrow resets the runtime flag to `false` on entry.
    field_init: bool,
}

/// One SplitMix64 step: scrambles an input word into a well-distributed output.
/// Used to derive `Math.random`'s initial `xorshift128+` state — SplitMix64 is
/// the standard finalizer for xorshift seeding (it spreads the mixed entropy
/// across the state and avoids the all-zero fixed point).
const fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A high-resolution monotonic cycle counter (the CPU timestamp counter), read
/// for entropy. Roughly tracks machine uptime in cycles and changes on every
/// call. Returns 0 on architectures without a cheap unprivileged counter, where
/// the caller leans on its other entropy sources.
// One of the small, audited VM primitives the crate's `unsafe_code = "deny"`
// policy permits to opt back in: both reads are unprivileged, have no
// preconditions, and no memory effects.
#[allow(unsafe_code)]
fn cycle_counter() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `rdtsc` is unconditionally available on x86_64 and has no
        // preconditions or memory effects.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let v: u64;
        // SAFETY: reading the virtual count register is an unprivileged,
        // side-effect-free instruction on aarch64.
        unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack)) };
        v
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        0
    }
}

/// Produces the initial `xorshift128+` state for a new interpreter's
/// `Math.random`.
///
/// With the `crypto` feature it draws 128 bits from purecrypto's OS CSPRNG
/// ([`purecrypto::rng::OsRng`], i.e. `getrandom(2)` on Linux). Otherwise (the
/// `no_std` / no-`crypto` core) it builds a best-effort, non-deterministic seed
/// by mixing every cheap entropy source available — a high-resolution cycle
/// counter, the wall clock (with `std`), the process id, and an ASLR-derived
/// stack address — through SplitMix64. There is no fixed compile-time seed.
///
/// Either way `Math.random` is *not* a security RNG; WebCrypto
/// (`crypto.getRandomValues`) is the path for that. The fallback may be weak on
/// a target that exposes none of the above, but it is never constant.
fn math_random_seed() -> [u64; 2] {
    #[cfg(feature = "crypto")]
    {
        use purecrypto::rng::{OsRng, RngCore};
        // `OsRng::next_u64` can `panic!` on a getrandom failure (seccomp without
        // the syscall allowed, no `/dev/urandom`, …). A sandboxed embedder must
        // not be aborted by that during `Interp::new`; catch the unwind (when
        // `std` is available) and fall through to the best-effort entropy mix
        // below (RNG-1). Untrusted JS cannot reach this path — it is robustness
        // for the host.
        #[cfg(feature = "std")]
        let drawn: Result<[u64; 2], _> = std::panic::catch_unwind(|| {
            let mut rng = OsRng;
            [rng.next_u64(), rng.next_u64()]
        });
        #[cfg(not(feature = "std"))]
        let drawn: Result<[u64; 2], ()> = {
            let mut rng = OsRng;
            Ok([rng.next_u64(), rng.next_u64()])
        };
        // An all-zero draw (probability 2^-128) would be the generator's fixed
        // point; fall through to the entropy mix rather than accept it.
        if let Ok(s) = drawn
            && s[0] | s[1] != 0
        {
            return s;
        }
    }

    // The golden-ratio value is the SplitMix64 avalanche basis (not a seed): the
    // output varies because the entropy sources below are XOR-mixed into `acc`.
    let mut acc: u64 = 0x9E37_79B9_7F4A_7C15;
    acc ^= cycle_counter();
    #[cfg(feature = "std")]
    {
        if let Ok(d) =
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH)
        {
            acc ^= (d.as_nanos() as u64).rotate_left(32);
        }
        acc ^= u64::from(std::process::id()).wrapping_mul(0x2545_F491_4F6C_DD1D);
    }
    // ASLR / stack-layout entropy: the address of a local variable.
    let here = 0u8;
    acc ^= (&here as *const u8 as u64).rotate_left(17);

    // Avalanche `acc` into two independent state words; forcing a low bit keeps
    // the pair off the all-zero fixed point without reintroducing a constant.
    let s0 = splitmix64(acc) | 1;
    let s1 = splitmix64(acc.wrapping_add(0x9E37_79B9_7F4A_7C15));
    [s0, s1]
}

/// A tree-walking interpreter over the performance object model.
pub struct Interp<'a> {
    realm: Realm,
    /// The current lexical scope (innermost).
    current: Scope,
    /// The current *variable* environment — the function/program/eval scope that
    /// `var` and top-level function declarations hoist into. Tracks where the
    /// Annex B.3.3 runtime binding-update writes a block-level function value.
    var_scope: Scope,
    /// Override for the variable environment of the *next* eval body's hoisting
    /// pass (the spec's `varEnv`, distinct from the fresh lexical env `lexEnv`).
    /// A sloppy `eval` runs its lexical (`let`/`const`/`class`) declarations in a
    /// fresh child scope, but hoists its `var`/function declarations OUT into the
    /// caller's variable environment (direct) or the global environment
    /// (indirect). Set by `eval_string`, consumed (taken) by the eval body's
    /// `hoist_with_kind`; `None` for ordinary function/program hoisting (where
    /// `varEnv == lexEnv`).
    eval_var_scope: Option<Scope>,
    /// True while running a `$262.evalScript` body, which is a *Script* (its
    /// `var`/function global bindings are non-configurable per
    /// GlobalDeclarationInstantiation), not an indirect `eval` (whose global
    /// `var`/function bindings are configurable/deletable). It otherwise shares
    /// the eval-body machinery (`run_eval_body`).
    script_eval_globals: bool,
    /// The names of block-level function declarations that the Annex B.3.3
    /// legacy extension var-hoists into the current variable environment — i.e.
    /// the only names whose outer `var` binding is updated when a block function
    /// declaration is *executed*. Excludes names that conflict with a parameter
    /// or an already-present binding (where the extension does not apply).
    annexb_block_fns: Vec<String>,
    /// Function-AST table; a closure cell holds an index into this.
    functions: Vec<FnDef<'a>>,
    /// Class-AST table; a class cell holds an index into this.
    classes: Vec<&'a Class>,
    /// The source text of the code region currently being *defined* — the program
    /// (or `eval`/`Function` body) whose AST is executing. Set by the run/eval
    /// entry points; `class.span` / `function.span` byte offsets index into it, so
    /// `set_fn_source` / `make_class` can slice a function/class's literal source
    /// for `Function.prototype.toString`. Empty until the first program runs (and
    /// for internal callers that don't set it — those functions then use the
    /// NativeFunction fallback).
    src: &'a str,
    /// Per-class evaluated *computed* member keys, by `class.body` index. Filled
    /// eagerly at class definition (ClassDefinitionEvaluation evaluates every
    /// computed `PropertyName` in source order, so a throwing key is a
    /// definition-time error and side effects run exactly once); the lazy
    /// prototype / private-member / static builders read the stored key instead
    /// of re-evaluating the expression.
    class_member_keys: Vec<alloc::collections::BTreeMap<usize, String>>,
    /// Lazily-created built-in iterator prototypes keyed by `@@toStringTag`
    /// (`"Array Iterator"`, `"String Iterator"`, `"Map Iterator"`, …). Each is an
    /// object chained to `%IteratorPrototype%` with an inherited `next` and the
    /// tag, so `Object.getPrototypeOf(arr.values())` is a real
    /// `%ArrayIteratorPrototype%` (reflection tests).
    builtin_iter_protos: alloc::collections::BTreeMap<&'static str, Handle>,
    /// Per-class static members (`Class.foo`), parallel to `classes`.
    class_statics: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class static *field* names in declaration order — the enumerable own
    /// keys of the constructor (static methods are non-enumerable), for
    /// `Object.keys`/`values`/`entries` of a class.
    class_static_fields: Vec<Vec<String>>,
    /// Per-class static getter functions (`static get x() {}`), called on read.
    class_static_get: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class static setter functions (`static set x(v) {}`), called on write.
    class_static_set: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class captured definition scope, parallel to `classes`.
    class_envs: Vec<Scope>,
    /// Per-class native-constructor superclass id (`class X extends Error`),
    /// parallel to `classes`; `None` when the parent is a class or absent.
    class_native_super: Vec<Option<u16>>,
    /// Per-class ordinary-function superclass handle (`class X extends fn {}`
    /// where `fn` is a plain user function, not a class or native), parallel to
    /// `classes`; `None` otherwise.
    class_fn_super: Vec<Option<NanBox>>,
    /// Per-class **class** superclass id (`class D extends C {}` where `C` is a
    /// class), parallel to `classes`; `None` when the parent is native, an
    /// ordinary function, `null`, or absent. Cached at class-definition time (the
    /// heritage is evaluated exactly once per spec) so the eager `.prototype`
    /// materialization does not re-evaluate the `extends` expression.
    class_super_id: Vec<Option<u32>>,
    /// Per-class resolved `protoParent` — `Get(superclass, "prototype")` captured
    /// once at class-definition time (ECMA-262 ClassDefinitionEvaluation reads it
    /// exactly once, so re-reading in the lazy `.prototype` build would double-fire
    /// a `prototype` getter). `Some(obj)` for an object protoParent, `Some(null)`
    /// for `extends null`, `None` for a class with no heritage (defaults to
    /// `Object.prototype`).
    class_proto_parent: Vec<Option<NanBox>>,
    /// Per-class constructor handle (the class value), parallel to `classes`, so
    /// the lazily-materialized `.prototype` can install a `constructor` back-link
    /// and link a derived prototype to its base's prototype.
    class_handles: Vec<NanBox>,
    /// Cache of a class's private *method/accessor* function values, keyed by
    /// `(class_id, storage_key)`. A private method is defined once per class
    /// evaluation and shared by every instance (so `c1.#m === c2.#m`), so it is
    /// created lazily on first instantiation and reused thereafter.
    private_method_cache: alloc::collections::BTreeMap<(u32, String), NanBox>,
    /// Per-class lexically-enclosing class id, parallel to `classes`. Captured
    /// from `current_home` when the class is set up (the home class of the code
    /// that evaluates the class definition is its lexical parent). Drives
    /// private-name resolution: a private reference `#x` resolves to the nearest
    /// enclosing class that *declares* `#x`, so two classes with `#x` never
    /// collide and a nested class can shadow an outer one's `#x`.
    class_lexical_parent: Vec<Option<u32>>,
    /// Per-class set of bare private names (`x` for `#x`) declared in the class
    /// body — instance and static fields/methods/accessors — parallel to
    /// `classes`. Used with `class_lexical_parent` to resolve a private
    /// reference to its declaring class.
    class_private_names: Vec<alloc::collections::BTreeSet<alloc::boxed::Box<str>>>,
    /// Intrinsic `Temporal.<Type>.prototype` handles, indexed by `TemporalKind`.
    temporal_protos: Vec<Option<Handle>>,
    /// One-shot binding name for NamedEvaluation of an anonymous class
    /// expression (`var C = class {}`, `x = class {}`): the name the class will
    /// receive. `make_class` consumes it so the class's `name` is set *before*
    /// static initializers run (which may read `this.name` / the class name).
    pending_class_name: Option<&'a str>,
    /// Current function-call nesting depth (recursion guard).
    call_depth: usize,
    /// Whether a `return` evaluated *right now* is a proper-tail-call candidate:
    /// set true only while running a strict, non-async function body (in
    /// [`Interp::invoke_inner`]); cleared inside a `try` Block (and a `catch`
    /// that a `finally` follows) by [`Interp::exec_try`]. When true (and the
    /// callee is a plain JS function), `return f(...)` yields
    /// [`ExecError::TailCall`] for `invoke`'s trampoline instead of recursing.
    tail_pos: bool,
    /// Whether `new.target` is lexically in scope at the current execution point —
    /// true inside a non-arrow function/method/constructor body, a class field
    /// initializer, or a static block; false at top-level script/module code. An
    /// arrow inherits the enclosing value (it is transparent to `new.target`). A
    /// *direct* `eval` uses this to decide whether `new.target` is a valid token in
    /// the eval code (the dynamic call depth is wrong here because an arrow defined
    /// at the top level is on the call stack but has no `new.target` in scope).
    new_target_in_scope: bool,
    /// C2: current *tree-walk* recursion depth — `eval`/`exec` descend on the
    /// native stack for nested expressions/statements, and the precedence loop in
    /// the parser flattens `a + a + a + …` into a shallow AST that nonetheless
    /// drives thousands of nested `eval` calls. The function-call `call_depth`
    /// guard does not count these, so a deep expression would overflow the host
    /// stack and abort. This counter is checked against `limits.max_eval_depth`
    /// (a dedicated knob, separate from `max_call_depth`, because each tree-walk
    /// level burns far more native stack than a bytecode call frame) at the
    /// `eval`/`exec` hubs and throws a catchable `RangeError` past the cap.
    eval_depth: usize,
    /// `xorshift128+` PRNG state backing `Math.random` (pure Rust, no foreign
    /// code). Two 64-bit words give a 2^128-1 period; seeded by
    /// [`math_random_seed`].
    rng_state: [u64; 2],
    /// The current `this` binding (method/constructor receiver).
    this_val: NanBox,
    /// `new.target` for the current invocation (the constructor when reached via
    /// `new`, else `undefined`; arrows inherit the enclosing value).
    new_target: NanBox,
    /// One-shot: a pending `new.target` set by `construct`, consumed by the next
    /// non-arrow invocation (so `new.target` is the constructor inside it).
    pending_new_target: Option<NanBox>,
    /// One-shot `newTarget` override for the next `construct` (set by
    /// `Reflect.construct(target, args, newTarget)`); else `new.target` is the callee.
    reflect_new_target: Option<NanBox>,
    /// One-shot: the next `call_method` was reached through a *generic*
    /// `Array.prototype.<m>` invocation (e.g. `Array.prototype.reduce.call(o)`),
    /// so a primitive-wrapper `this` must be treated as an array-like object
    /// (read its own `length`/indexed properties) rather than unwrapped to its
    /// boxed primitive and dispatched as a String/Number/Boolean method.
    array_proto_generic: bool,
    /// When a *generic* array-like receiver is materialized for an iteration
    /// method (`map`/`filter`/`forEach`/`reduce`/…), this records which indices
    /// were actually *present* (`HasProperty`) on the source object, so those
    /// methods skip the holes per spec (`Array.prototype.filter.call({1:11,
    /// length:2})` skips the absent index 0). `None` for a dense real array (no
    /// holes are tracked — the dense fast path is unchanged).
    array_like_present: Option<alloc::vec::Vec<bool>>,
    /// Persistent mutable state (memory/globals) of each live WASM instance, keyed
    /// by an instance id stored on the instance's export wrappers — so a
    /// `WebAssembly.Instance`'s memory and globals survive across export calls.
    wasm_states: alloc::collections::BTreeMap<u32, crate::wasm_rt::InstanceState>,
    /// The decoded `Module` of each live WASM instance, keyed by instance id, so a
    /// hot loop calling an export reuses the parsed+validated module instead of
    /// re-`decode_with_limits`-ing the raw bytes every call (S1: the dominant
    /// per-call cost). A `Module` is immutable once decoded; the mutable per-call
    /// state lives in `wasm_states`. Borrow-checker note: the module is `take`n out
    /// for the duration of an export call (so the import-dispatch closure can borrow
    /// the engine mutably) and put back afterwards.
    wasm_modules: alloc::collections::BTreeMap<u32, crate::wasm_rt::Module>,
    /// The canonical `WebAssembly.Memory` object of each live WASM instance that
    /// exports memory, keyed by instance id. Its `ArrayBuffer`'s `Cell::Bytes` is
    /// the single shared linear-memory store: copied *into* the instance's
    /// `Store.mem` before each export call and copied back *out* after, so a JS
    /// `Uint8Array`/`DataView` over `Memory.buffer` observes wasm writes (and wasm
    /// observes JS writes made before the call). The `ArrayBuffer` object is stable
    /// across `grow` — only its bytes store is resized (A5, #11).
    wasm_mem_objs: alloc::collections::BTreeMap<u32, crate::heap::Handle>,
    /// Next WASM-instance id to hand out.
    wasm_next_id: u32,
    /// When running a generator body eagerly, the buffer `yield` appends to.
    /// (Retained for built-in eager iterables — Map/Set entries, regexp matches —
    /// and as the degraded fallback for complex yield-bearing operands the lazy
    /// machine does not reify.)
    gen_sink: Option<Vec<NanBox>>,
    /// Suspended lazy-generator activations, indexed by the `GEN_FRAME` id stored
    /// on the generator object. A vacated slot (a finished generator) is `None`
    /// and may be reused by the next generator call.
    gen_frames: Vec<Option<generator::GenFrame<'a>>>,
    /// One-shot: an async coroutine `(frame id, controller handle)` whose first
    /// synchronous burst must run once the caller's ambient state is restored
    /// (set while building the frame in `invoke_inner`, consumed immediately
    /// after). See the async path in `call.rs`.
    pending_async_start: Option<(usize, Handle)>,
    /// Whether the coroutine currently being driven (in `gen_drive`) is an *async*
    /// generator. Read by `yield*` delegation to use the async-iterator protocol
    /// (`[Symbol.asyncIterator]`, awaiting each `next()` result). Saved/restored
    /// around `gen_drive` so a reentrant resume (via the event loop) is balanced.
    gen_is_async: bool,
    /// The `Symbol.for` global registry: shared symbols keyed by string.
    symbol_registry: alloc::collections::BTreeMap<String, NanBox>,
    /// Cached well-known symbols (e.g. `Symbol.iterator`), created on first use.
    well_known_symbols: alloc::collections::BTreeMap<&'static str, NanBox>,
    /// The frozen template-strings object for each tagged-template site (keyed by the
    /// AST node's address), so the same array is passed to the tag on every evaluation.
    tagged_template_cache: alloc::collections::BTreeMap<usize, NanBox>,
    /// `RegExp.prototype`, recorded at setup so RegExp instances can link their
    /// `[[Prototype]]` to it (and species lookups resolve cheaply).
    regexp_proto: Option<Handle>,
    /// The `%RegExp%` intrinsic constructor handle, recorded at setup. Used to
    /// brand-check the receiver of the Annex B.2.5 legacy static accessors
    /// (`RegExp.$1`/`input`/`lastMatch`/…), which require `this === RegExp`.
    regexp_ctor: Option<Handle>,
    /// Leak-once cache interning `Intl.NumberFormat` currency/unit codes to `&'static str`
    /// (the `intl` crate's options take `'static`); bounded by the distinct codes a program
    /// uses.
    #[cfg(feature = "intl")]
    intl_intern: alloc::collections::BTreeMap<String, &'static str>,
    /// Leak-once cache interning method names (derived from runtime property keys
    /// or accessor prefixes) to `&'a str` for storage as `FnDef::name`.
    method_name_intern: alloc::collections::BTreeMap<String, &'static str>,
    /// The superclass to invoke for `super(...)` inside the running constructor.
    pending_super: Option<(u32, Scope)>,
    /// The native-constructor superclass for `super(...)` (e.g. extending Error).
    pending_super_native: Option<u16>,
    /// The ordinary-function superclass for `super(...)` (`extends fn`).
    pending_super_fn: Option<NanBox>,
    /// The class of the currently-running method (for `super.method()`).
    current_home: Option<u32>,
    /// The lexically-enclosing class of the currently-running function, for
    /// **private-name** resolution. Unlike `current_home` (which an ordinary
    /// function resets to `None`, so `super` is unavailable), this preserves the
    /// class a nested ordinary `function` was textually defined in, so `#x` inside
    /// it still resolves. Set from each function's `FnDef::lexical_class`.
    current_lexical_home: Option<u32>,
    /// The `[[HomeObject]]` of the currently-running object-literal method — the
    /// object the method was defined on — so its `super.x` resolves through that
    /// object's prototype (when there is no enclosing class home).
    current_home_object: Option<Handle>,
    /// Whether the currently-running method was entered as a static method, so
    /// `super.x` resolves against the superclass's static members.
    current_home_static: bool,
    /// Whether execution is directly inside a class field initializer or static
    /// initialization block (with no intervening non-arrow function boundary). A
    /// *direct* `eval` here inherits the ContainsArguments early error: an
    /// `arguments` reference in the eval body is a SyntaxError. Reset to `false`
    /// across ordinary function/method calls (an arrow keeps it, matching the
    /// lexical `arguments` inheritance).
    in_field_initializer: bool,
    /// While a *derived* class constructor body runs before `super(...)`, holds
    /// `(instanceValue, classId)`: `this` is in its temporal dead zone
    /// (`this_val` is `tdz()`), and the stashed instance + class let `super(...)`
    /// initialize `this` and run this class's field initializers on return.
    /// `None` once `super` has run (or outside a derived constructor). A
    /// derived constructor that completes with this still set never called
    /// `super` — accessing `this` / the implicit return is a ReferenceError.
    pending_this_init: Option<(NanBox, u32)>,
    /// While a *parameter default value* is being evaluated, the BoundNames of
    /// the enclosing function's formal parameters (plus `arguments` for a
    /// non-arrow). A sloppy direct `eval("var X")` running here is an
    /// EvalDeclarationInstantiation early error (SyntaxError) when `X` is one of
    /// these names — the function has a separate parameter environment that
    /// already binds them (`function f(a = eval("var a")) {}`). `None` outside
    /// parameter-default evaluation; cleared across function boundaries so a
    /// nested call / the body never sees the outer parameter set.
    eval_param_names: Option<Vec<String>>,
    /// A label attached to the next loop (for `break`/`continue label`).
    pending_label: Option<String>,
    /// The promise-reaction microtask queue, drained after the script.
    microtasks: Vec<Job>,
    /// Pending `setTimeout` callbacks (macrotasks), run after the microtask queue
    /// drains — ordered by absolute virtual fire-time `at`, then insertion (`seq`).
    macrotasks: Vec<Timer>,
    /// Monotonic id handed out by `setTimeout` (for `clearTimeout`).
    timer_next_id: u64,
    /// Monotonic insertion counter breaking equal-`at` ties.
    timer_seq: u64,
    /// The **virtual clock** (milliseconds). There is no real time source in the
    /// cooperative single-threaded model, so time is modeled: dispatching a
    /// macrotask advances `virtual_now` to that timer's scheduled fire-time
    /// (`at = virtual_now-at-schedule + delay`). `$262.agent.monotonicNow()` reads
    /// this clock, so a worker that parks in `Atomics.waitAsync(…, timeout)` and is
    /// released by the timeout observes `monotonicNow()` advance by ~`timeout`.
    /// (`Date.now()` is deliberately NOT tied to this — it uses the wall clock so
    /// `Date` tests see a real epoch timestamp.)
    virtual_now: f64,
    /// Whether the currently-executing code is in strict mode (`"use strict"`),
    /// which propagates into nested functions.
    strict: bool,
    /// The global object (`globalThis`) — substituted for an `undefined`/`null`
    /// `this` when calling a non-strict function.
    global_this: NanBox,
    /// Captured `console.log` output (a line per call).
    output: String,
    /// The global (root) lexical scope, captured once at construction. Indirect
    /// `eval` and the `Function` constructor run against a fresh child of this,
    /// regardless of the caller's current nesting.
    global_scope: Scope,
    /// Per-`ShadowRealm`-instance persistent global scope (a child of
    /// `global_scope`). A `ShadowRealm` instance stores an index into this vector
    /// under a hidden slot, so successive `evaluate` calls on the same instance
    /// share variable/function declarations. (Intrinsics are shared with the host
    /// realm — a best-effort model, not a fully isolated realm.)
    shadow_realm_scopes: Vec<Scope>,
    /// Per-`$262.createRealm()` global environments — each a *genuinely distinct*
    /// realm: its own global scope populated by a fresh `install_globals` (so its
    /// `Array`/`Object`/`TypeError`/… are separate heap cells from every other
    /// realm's), its own global object, and a snapshot of its intrinsic prototype
    /// pointers. Unlike `shadow_realm_scopes` (which share the host intrinsics),
    /// these back cross-realm identity (`other.Array !== Array`). The realm object
    /// returned to JS stores an index into this vector under a hidden slot.
    created_realms: Vec<CreatedRealm>,
    /// `GetFunctionRealm` side table: maps a callable's raw handle to the index of
    /// the `$262.createRealm()` realm (`created_realms`) it belongs to. A function
    /// *absent* from this map belongs to the main realm. Populated for every
    /// callable installed into a created realm's global (so `other.Object`,
    /// `other.Function`, … carry their realm) and for functions built *by* a
    /// created realm's `Function`/`GeneratorFunction` constructor (so
    /// `new other.Function()` carries the other realm). Consulted only on the
    /// cross-realm `GetPrototypeFromConstructor` fallback (a `newTarget` whose
    /// `.prototype` is not an Object), so it never perturbs same-realm construction.
    fn_realm: alloc::collections::BTreeMap<u64, usize>,
    /// The `created_realms` index of the realm whose function is *currently
    /// executing* — `None` for the main realm. Pushed on entry to every native
    /// method dispatch and every user-closure invocation (derived from the
    /// callee's `GetFunctionRealm`, falling back to the callee's captured global
    /// scope), restored on return. Consulted by [`Interp::make_error`] so an error
    /// thrown by a cross-realm intrinsic (e.g. `otherRealm.String.prototype.valueOf`
    /// called on a bad `this`) carries *that realm's* `%TypeError%`, and by
    /// [`Interp::array_species_create`] to identify the current Realm Record for the
    /// `SameValue(C, realmC.[[%Array%]])` cross-realm nullification step.
    cur_realm: Option<usize>,
    /// Programs parsed at runtime by `eval` / the `Function` constructor, keyed by
    /// source string. The interpreter's function/AST tables hold `&'a` references
    /// into the running program; a dynamically-parsed `Program` must therefore
    /// outlive the borrow. Each distinct source is parsed once, boxed, and leaked
    /// to a `&'static Program` (which coerces to `&'a`); the cache dedupes so a
    /// loop calling `eval` on the same string (or repeated `Function` bodies) does
    /// not re-leak. The leak is bounded by the number of *distinct* eval/Function
    /// sources a program produces.
    eval_programs: alloc::collections::BTreeMap<String, &'static Program>,
    /// The ES-module loader/linker/evaluator state, present only while running a
    /// module graph (or after a dynamic `import()` has loaded one). `None` for a
    /// plain script — module support is purely additive. See [`module`].
    #[cfg(all(feature = "module", feature = "std"))]
    modules: module::ModuleRegistry,
    /// The import-binding alias table of the *currently evaluating* module:
    /// `local name -> (exporting module's scope, that module's local name)`.
    /// An identifier read consults this first so an imported binding resolves —
    /// *live* — through the exporting module's own scope (so a post-evaluation
    /// mutation in the exporter is observed here). Swapped on each module
    /// boundary; empty for a script.
    #[cfg(all(feature = "module", feature = "std"))]
    module_imports: alloc::rc::Rc<alloc::collections::BTreeMap<String, (Scope, String)>>,
    /// `import.meta` for the currently evaluating module (an object with at least
    /// `url`); `None` outside module code.
    #[cfg(all(feature = "module", feature = "std"))]
    import_meta: Option<NanBox>,
    /// A base "referrer" key for dynamic `import()` evaluated from *script* code
    /// (which has no enclosing module). Lets a script's `import("./x.js")`
    /// resolve relative to the script file rather than the process cwd. `None`
    /// for an ordinary script run.
    #[cfg(all(feature = "module", feature = "std"))]
    script_import_base: Option<String>,
    /// Live-binding backing for **module namespace exotic objects**: a namespace
    /// object's heap-handle (raw) maps each exported name to the
    /// `(scope, local name)` slot it reflects. A property read of one of these
    /// names refreshes from the slot so a post-materialisation mutation in the
    /// exporting module is observed (`ns.x` is a *live* binding, per §28.3), while
    /// the property itself stays an ordinary writable/non-configurable data
    /// property so `getOwnPropertyDescriptor` still reports a value.
    #[cfg(all(feature = "module", feature = "std"))]
    module_namespaces:
        alloc::collections::BTreeMap<u64, alloc::collections::BTreeMap<String, (Scope, String)>>,
    /// Deferred Module Namespace exotic objects (import-defer proposal): maps the
    /// object handle to the resolved key of its still-unevaluated target module.
    /// A property access naming one of the module's exports triggers synchronous
    /// evaluation (the spec's deferred semantics) before the live read.
    #[cfg(all(feature = "module", feature = "std"))]
    deferred_namespaces: alloc::collections::BTreeMap<u64, String>,
    /// The key of the module whose body is currently executing, used as the
    /// referrer for a dynamic `import()` when the active scope can't be matched to
    /// a module record (e.g. the `import()` runs inside a nested function/arrow,
    /// so `self.current` is the callee's scope rather than the module's). Saved /
    /// restored around each module body. `None` outside module code.
    #[cfg(all(feature = "module", feature = "std"))]
    active_module_key: Option<String>,
    /// Dynamically-registered **host** native functions (`ROADMAP.md` §4.0),
    /// indexed by the [`Cell::HostFn`](crate::cell::Cell::HostFn) id that names
    /// them. Each slot is `None` while its closure is *taken out* for the
    /// duration of a call (so a re-entrant call to the *same* host function is
    /// detected rather than aliasing a `&mut` closure); it is restored when the
    /// call returns. The boxed closures are runtime state — they capture Rust
    /// data the GC cannot trace and are not serialized (a snapshot drops them,
    /// like IC slots), which is why they live here rather than in the heap.
    host_fns: Vec<Option<HostFn>>,
    /// Test262 `$262.agent` cooperative-scheduler state (see [`agent`]).
    agent: AgentState,
    /// **Mapped `arguments` exotic objects** (sloppy-mode functions with a simple
    /// parameter list): the arguments object's heap-handle (raw) maps each
    /// currently-mapped integer index to the `(scope, parameter name)` binding it
    /// aliases (10.4.4 `[[ParameterMap]]`). A `[[Get]]` of a mapped index reads the
    /// live parameter binding, a `[[Set]]` writes it, and `delete` / a
    /// `defineProperty` that installs an accessor or a non-writable data property
    /// *breaks* the mapping for that index (drops the slot). Empty for strict
    /// functions and any non-simple parameter list (which are unmapped).
    arg_maps: alloc::collections::BTreeMap<u64, ArgMap>,
}

/// The `[[ParameterMap]]` of one mapped `arguments` object: the shared parameter
/// scope plus each still-mapped index's parameter name (see [`Interp::arg_maps`]).
struct ArgMap {
    /// The function-call scope holding the aliased parameter bindings.
    scope: Scope,
    /// Currently-mapped `index → parameter name`. An index is removed when its
    /// mapping breaks (delete / accessor or non-writable redefine).
    slots: alloc::collections::BTreeMap<usize, String>,
}

/// The Test262 `$262.agent` cooperative scheduler + `Atomics.waitAsync` state.
///
/// This engine is single-threaded (the heap is `Rc`, not `Send`), so worker
/// agents are modeled cooperatively: `$262.agent.start(src)` runs the worker to
/// completion *eagerly* in a fresh realm, and a worker that registers a
/// `receiveBroadcast` callback has it invoked later when the main agent
/// `broadcast`s. Reports flow through a shared FIFO queue. True cross-agent
/// interleaving (main *blocks* in `Atomics.wait` while a worker runs and
/// `notify`s) is out of scope — those tests time out and are ledgered.
#[derive(Default)]
struct AgentState {
    /// The shared FIFO report queue: workers `report(msg)` push, the main agent
    /// `getReport()` pops (returns the front, or `null` when empty).
    reports: alloc::collections::VecDeque<String>,
    /// Worker `receiveBroadcast(cb)` callbacks awaiting a `broadcast`: each is
    /// `(created-realm index, callback)`. Drained (in order) by `broadcast(sab)`,
    /// which invokes every callback with the SharedArrayBuffer.
    broadcasts: Vec<(usize, NanBox)>,
    /// Pending `Atomics.waitAsync` waiters, keyed by `(buffer handle, byte
    /// index)`; a matching `Atomics.notify` settles the promise with `"ok"`, and
    /// a finite-timeout waiter is settled `"timed-out"` by a macrotask.
    waiters: Vec<AtomicsWaiter>,
    /// The created-realm index of the worker whose source is *currently* running
    /// eagerly under `$262.agent.start` — so a `receiveBroadcast` callback it
    /// registers is tagged with the realm to restore when it is later invoked.
    current_agent_realm: Option<usize>,
}

/// One pending `Atomics.waitAsync` async waiter.
struct AtomicsWaiter {
    /// The backing buffer's handle (raw), identifying the wait location together
    /// with `byte_index` across all views over the same SharedArrayBuffer.
    buffer: u64,
    /// The absolute byte offset of the waited-on element within the buffer.
    byte_index: usize,
    /// The promise to settle with `"ok"` (a matching `notify`) or `"timed-out"`.
    promise: Handle,
}

/// A dynamically-registered host (native) function: the Rust closure
/// [`Interp::register_fn`] wraps, plus the spec-shaped own `name`/`length` the
/// function object reports. Called from JS exactly like a built-in, receiving a
/// [`Ctx`] to build values, read/write properties, throw, and re-enter JS
/// (`ROADMAP.md` §4.0).
pub struct HostFn {
    name: String,
    length: u32,
    call: HostCallback,
    /// Whether `new f(...)` is allowed: `register_constructor` sets this, so the
    /// closure runs as a `[[Construct]]` (with a fresh `this`); a plain
    /// `register_fn` leaves it `false` and `new f()` is a `TypeError`.
    is_constructor: bool,
}

/// The boxed Rust closure backing a [`HostFn`]: it receives a [`Ctx`] handle,
/// the call's `this` value, and the argument list, and returns either a result
/// value or a value to `throw` (raised as a JS exception). It is `'static`
/// (captured state must outlive the interpreter's runs) and `FnMut` (it may hold
/// mutable host state between calls).
pub type HostCallback =
    alloc::boxed::Box<dyn FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox>>;

/// The context handle a registered host function ([`Interp::register_fn`]) uses
/// to talk back to the engine: construct values, read/write properties, throw
/// JS errors, coerce arguments, and re-enter JS by calling other functions
/// (`ROADMAP.md` §4.0). It borrows the interpreter for the duration of one host
/// call; values it hands out are ordinary [`NanBox`] handles that stay valid
/// while the call runs (the host must not stash a bare handle across calls — a
/// rooted handle scope for that is future §4.0 work).
pub struct Ctx<'c, 'a> {
    interp: &'c mut Interp<'a>,
}

impl<'c, 'a> Ctx<'c, 'a> {
    /// The JS `undefined` value.
    #[must_use]
    pub fn undefined(&self) -> NanBox {
        NanBox::undefined()
    }

    /// The JS `null` value.
    #[must_use]
    pub fn null(&self) -> NanBox {
        NanBox::null()
    }

    /// A JS Number.
    #[must_use]
    pub fn number(&self, n: f64) -> NanBox {
        NanBox::number(n)
    }

    /// A JS Boolean.
    #[must_use]
    pub fn boolean(&self, b: bool) -> NanBox {
        NanBox::boolean(b)
    }

    /// A JS String (heap-allocated, rope-backed).
    pub fn string(&mut self, s: &str) -> NanBox {
        self.interp.new_str(s)
    }

    /// A fresh empty ordinary object (`{}` with `%Object.prototype%`).
    pub fn new_object(&mut self) -> NanBox {
        let h = self.interp.realm.new_object();
        NanBox::handle(h.to_raw())
    }

    /// A fresh dense array with the given `elements`.
    pub fn new_array(&mut self, elements: Vec<NanBox>) -> NanBox {
        let h = self.interp.realm.new_array(elements);
        NanBox::handle(h.to_raw())
    }

    /// The realm's global object (`globalThis`).
    #[must_use]
    pub fn global(&self) -> NanBox {
        self.interp.global_this
    }

    /// Reads property `key` of `obj` (walking the prototype chain), coercing a
    /// primitive receiver to its wrapper first (ordinary `[[Get]]`). A `null` /
    /// `undefined` receiver reads `undefined`.
    ///
    /// # Errors
    /// The thrown value if a getter (or proxy trap) along the chain throws.
    pub fn get(&mut self, obj: NanBox, key: &str) -> Result<NanBox, NanBox> {
        let recv = self.interp.coerce_to_object(obj);
        let Some(h) = recv.as_handle().map(Handle::from_raw) else {
            return Ok(NanBox::undefined());
        };
        self.interp
            .read_member(h, key)
            .map_err(|e| self.interp.exec_error_value(e))
    }

    /// Sets an own data property `key` of the object `obj` to `value`. A
    /// non-object `obj` is a no-op. (This writes an own data property directly;
    /// inherited setters are not invoked — a full `[[Set]]` is future work.)
    pub fn set(&mut self, obj: NanBox, key: &str, value: NanBox) {
        if let Some(h) = obj.as_handle().map(Handle::from_raw) {
            self.interp.realm.set_property(h, key, value);
        }
    }

    /// Builds (but does not raise) a `TypeError` with `message`. Return it as
    /// `Err(..)` from the host closure to throw it.
    pub fn type_error(&mut self, message: &str) -> NanBox {
        let m = self.interp.new_str(message);
        self.interp.make_error(N_TYPE_ERROR, Some(m))
    }

    /// Builds (but does not raise) a `RangeError` with `message`.
    pub fn range_error(&mut self, message: &str) -> NanBox {
        let m = self.interp.new_str(message);
        self.interp.make_error(N_RANGE_ERROR, Some(m))
    }

    /// Builds (but does not raise) a plain `Error` with `message`.
    pub fn error(&mut self, message: &str) -> NanBox {
        let m = self.interp.new_str(message);
        self.interp.make_error(N_ERROR_BASE, Some(m))
    }

    /// `ToNumber(v)` — the JS numeric coercion (calls `valueOf`/`toString` for an
    /// object, parses a string, …).
    ///
    /// # Errors
    /// The thrown value if coercion throws (e.g. a `BigInt`, a `Symbol`, or a
    /// throwing `valueOf`).
    pub fn to_number(&mut self, v: NanBox) -> Result<f64, NanBox> {
        let n = self
            .interp
            .coerce_to_number(v)
            .map_err(|e| self.interp.exec_error_value(e))?;
        Ok(self.interp.realm.to_number(n))
    }

    /// `ToString(v)` — the JS string coercion.
    ///
    /// # Errors
    /// The thrown value if coercion throws (e.g. a `Symbol`, or a throwing
    /// `toString`).
    pub fn to_string(&mut self, v: NanBox) -> Result<String, NanBox> {
        self.interp
            .coerce_to_string(v)
            .map_err(|e| self.interp.exec_error_value(e))
    }

    /// `ToBoolean(v)` — JS truthiness.
    #[must_use]
    pub fn to_boolean(&self, v: NanBox) -> bool {
        self.interp.realm.truthy(v)
    }

    /// Calls the JS function `callee` with the given `this` and `args`,
    /// re-entering the engine. Works for user functions, natives, other host
    /// functions, bound functions, and callable proxies.
    ///
    /// # Errors
    /// The thrown value if `callee` is not callable or the call throws.
    pub fn call(
        &mut self,
        callee: NanBox,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, NanBox> {
        self.interp
            .call_with_this(callee, this, args)
            .map_err(|e| self.interp.exec_error_value(e))
    }

    /// The JS `typeof` of `v` (`"undefined"`, `"boolean"`, `"number"`, `"string"`,
    /// `"symbol"`, `"bigint"`, `"object"`, or `"function"`).
    #[must_use]
    pub fn type_of(&self, v: NanBox) -> &'static str {
        self.interp.realm.type_of_value(v)
    }

    /// Whether `v` is a callable function (an ordinary function, a native, a bound
    /// function, a class constructor, or a callable proxy).
    #[must_use]
    pub fn is_callable(&self, v: NanBox) -> bool {
        self.interp.is_callable_value(v)
    }

    /// Whether `v` is an object (not a primitive, `null`, or `undefined`).
    #[must_use]
    pub fn is_object(&self, v: NanBox) -> bool {
        self.interp.is_object_value(v)
    }

    /// Whether `v` is an `Array` exotic object (`Array.isArray(v)`).
    #[must_use]
    pub fn is_array(&self, v: NanBox) -> bool {
        v.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.interp.realm.is_array(h))
    }

    /// The `length` of the Array `arr`, or `None` if `arr` is not an Array.
    #[must_use]
    pub fn array_len(&self, arr: NanBox) -> Option<usize> {
        arr.as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.interp.realm.array_length(h))
    }

    /// The element at index `i` of the Array `arr` (`undefined` when `arr` is not
    /// an Array or `i` is out of range). Present holes read as `undefined`.
    pub fn array_get(&mut self, arr: NanBox, i: usize) -> NanBox {
        match arr.as_handle().map(Handle::from_raw) {
            Some(h) if self.interp.realm.is_array(h) => self.interp.realm.get_element(h, i),
            _ => NanBox::undefined(),
        }
    }

    /// Sets element `i` of the Array `arr` to `value` (growing the array's length
    /// if `i` is past the end), returning whether the write happened (`false` when
    /// `arr` is not an Array). Pairs with [`array_get`](Self::array_get).
    pub fn array_set(&mut self, arr: NanBox, i: usize, value: NanBox) -> bool {
        match arr.as_handle().map(Handle::from_raw) {
            Some(h) if self.interp.realm.is_array(h) => self.interp.realm.set_element(h, i, value),
            _ => false,
        }
    }

    /// `HasProperty(obj, key)` — whether `obj` or its prototype chain has `key`
    /// (the `in` operator; runs proxy `has` traps). A non-object `obj` is `false`.
    pub fn has(&mut self, obj: NanBox, key: &str) -> bool {
        obj.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.interp.has_property(h, key))
    }

    /// Whether `obj` has an *own* property `key` (`Object.hasOwn`); a non-object
    /// `obj` is `false`.
    #[must_use]
    pub fn has_own(&self, obj: NanBox, key: &str) -> bool {
        obj.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.interp.realm.has_own(h, key))
    }

    /// Deletes own property `key` of `obj` (`delete obj[key]`), returning whether
    /// the property is gone afterward. A non-object `obj` is a vacuous `true`.
    pub fn delete(&mut self, obj: NanBox, key: &str) -> bool {
        match obj.as_handle().map(Handle::from_raw) {
            Some(h) => self.interp.realm.delete_property(h, key),
            None => true,
        }
    }

    /// The own string-keyed property names of `obj`, in ordinary ownKeys order
    /// (integer indices ascending, then insertion order); empty if not an object.
    #[must_use]
    pub fn own_keys(&self, obj: NanBox) -> Vec<String> {
        obj.as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.interp.realm.own_property_names(h))
            .unwrap_or_default()
    }

    /// A promise resolved with `value` (`Promise.resolve(value)`): an existing
    /// promise is returned as-is and a thenable is adopted; any other value
    /// fulfills the new promise. Lets an async host function hand JS a promise.
    pub fn resolved_promise(&mut self, value: NanBox) -> NanBox {
        let p = self.interp.promise_resolve(value);
        NanBox::handle(p.to_raw())
    }

    /// A promise already rejected with `reason` (`Promise.reject(reason)`).
    pub fn rejected_promise(&mut self, reason: NanBox) -> NanBox {
        let p = self.interp.fresh_promise();
        self.interp.settle(p, reason, false);
        NanBox::handle(p.to_raw())
    }

    /// Whether `v` is a promise object (a genuine promise with `[[PromiseState]]`,
    /// not merely a thenable).
    #[must_use]
    pub fn is_promise(&self, v: NanBox) -> bool {
        v.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.interp.realm.promise_state(h).is_some())
    }

    /// Whether `v` is a constructor (usable as the target of `new` / `construct`).
    #[must_use]
    pub fn is_constructor(&self, v: NanBox) -> bool {
        self.interp.is_constructor_value(v)
    }

    /// `Construct(callee, args)` — invoke `callee` as a constructor (`new
    /// callee(...args)`) and return the constructed object, re-entering JS.
    ///
    /// # Errors
    /// A `TypeError` if `callee` is not a constructor, or the thrown value if the
    /// constructor (or a base it chains to) throws.
    pub fn construct(&mut self, callee: NanBox, args: &[NanBox]) -> Result<NanBox, NanBox> {
        self.interp
            .construct(callee, args)
            .map_err(|e| self.interp.exec_error_value(e))
    }

    /// Pins `value` as a **persistent handle**, returning a stable index the host
    /// can hold *across* engine calls — the value survives GC and stays valid when
    /// the moving collector relocates it (a bare `NanBox` handle would not; see the
    /// `Ctx` note). Read it back later with [`Interp::persistent`] and free it with
    /// [`Interp::release_persistent`]. (`ROADMAP.md` §4.0 handle scope.)
    pub fn persist(&mut self, value: NanBox) -> u32 {
        self.interp.realm.persist(value)
    }

    /// Reads a value pinned earlier by [`persist`](Self::persist) (`undefined` if
    /// the index was released or never allocated), reflecting any GC relocation.
    #[must_use]
    pub fn persistent(&self, idx: u32) -> NanBox {
        self.interp
            .realm
            .persistent(idx)
            .unwrap_or(NanBox::undefined())
    }

    /// Creates a **deferred** promise for async host work: returns the promise to
    /// hand back to JS, plus a `token` the host settles later from a timer/IO
    /// completion via [`Interp::resolve_deferred`] / [`Interp::reject_deferred`].
    /// The promise's resolve/reject functions are pinned (a persistent handle)
    /// until it settles, so they survive GC across the wait.
    ///
    /// # Errors
    /// Propagates a failure building the promise capability.
    pub fn deferred(&mut self) -> Result<(NanBox, u32), NanBox> {
        let ctor = self
            .interp
            .current
            .get("Promise")
            .unwrap_or(NanBox::undefined());
        let cap = self
            .interp
            .new_promise_capability(ctor)
            .map_err(|e| self.interp.exec_error_value(e))?;
        let arr = self
            .interp
            .realm
            .new_array(alloc::vec![cap.resolve, cap.reject]);
        let token = self.interp.realm.persist(NanBox::handle(arr.to_raw()));
        Ok((cap.promise, token))
    }

    /// Releases a persistent handle so its value is no longer pinned.
    pub fn release_persistent(&mut self, idx: u32) {
        self.interp.realm.release_persistent(idx);
    }

    /// Attaches opaque native `state` (any `'static` Rust value) to the object
    /// `obj`, à la N-API `napi_wrap`: retrievable with [`native_state`](Self::native_state)
    /// and **dropped (its `Drop` run as a finalizer) when `obj` is
    /// garbage-collected**. The attachment is weak — it does not keep `obj` alive.
    /// A non-object `obj` is ignored.
    pub fn set_native_state<T: core::any::Any>(&mut self, obj: NanBox, state: T) {
        if let Some(h) = obj.as_handle().map(Handle::from_raw) {
            self.interp
                .realm
                .set_native_state(h, alloc::boxed::Box::new(state));
        }
    }

    /// Borrows the native state attached to `obj`, downcast to `T` (`None` if
    /// absent or a different type).
    #[must_use]
    pub fn native_state<T: core::any::Any>(&self, obj: NanBox) -> Option<&T> {
        obj.as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.interp.realm.native_state::<T>(h))
    }

    /// Full `[[Set]]` of `obj[key] = value` (`OrdinarySet`): invokes an own or
    /// inherited accessor **setter**, honors non-writable data properties, and
    /// runs a proxy `set` trap — unlike [`set`](Self::set), which writes an own
    /// data property directly. A non-object `obj` is a no-op.
    ///
    /// # Errors
    /// The thrown value if a setter (or proxy trap) along the chain throws.
    pub fn set_property(&mut self, obj: NanBox, key: &str, value: NanBox) -> Result<(), NanBox> {
        let Some(h) = obj.as_handle().map(Handle::from_raw) else {
            return Ok(());
        };
        let key_box = self.interp.new_str(key);
        self.interp
            .assign_member_value(h, key_box, value)
            .map_err(|e| self.interp.exec_error_value(e))
    }
}

/// A queued promise reaction: run `handler` with `value`, then settle `result`
/// with the outcome (or pass `value` through with the source status when
/// `handler` is `undefined`).
struct Job {
    handler: NanBox,
    value: NanBox,
    result: Handle,
    fulfilled: bool,
    /// A `finally` job: run the handler for side effects, then pass the original
    /// value/rejection through to `result`.
    finally: bool,
    /// A PromiseResolveThenableJob: `handler` is the thenable's `then` method, and
    /// the pair is `(resolve, reject)` to pass to it (with `this` = `value`, the
    /// thenable). When set, the job calls `then.call(thenable, resolve, reject)`
    /// instead of the ordinary reaction handling.
    thenable: Option<(NanBox, NanBox)>,
}

/// A pending `setTimeout` callback.
struct Timer {
    id: u64,
    /// Absolute virtual fire-time (ms on the [`Interp::virtual_now`] clock).
    at: f64,
    seq: u64,
    callback: NanBox,
    args: Vec<NanBox>,
}

impl Default for Interp<'_> {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in (native) function ids.
const N_MATH_MAX: u16 = 0;
const N_MATH_MIN: u16 = 1;
const N_MATH_ABS: u16 = 2;
const N_STRING: u16 = 3;
const N_NUMBER: u16 = 4;
const N_BOOLEAN: u16 = 5;
const N_PARSE_INT: u16 = 6;
const N_CONSOLE_LOG: u16 = 7;
const N_JSON_STRINGIFY: u16 = 8;
const N_JSON_PARSE: u16 = 27;
const N_OBJECT_KEYS: u16 = 9;
const N_OBJECT_VALUES: u16 = 10;
const N_ARRAY_IS_ARRAY: u16 = 11;
const N_MATH_FLOOR: u16 = 12;
const N_MATH_CEIL: u16 = 13;
const N_MATH_ROUND: u16 = 14;
const N_MATH_SQRT: u16 = 15;
const N_MAP: u16 = 16;
const N_SET: u16 = 17;
/// `get Map.prototype.size` — a brand-checking accessor (requires `[[MapData]]`).
const N_MAP_SIZE: u16 = 580;
/// `get Set.prototype.size` — a brand-checking accessor (requires `[[SetData]]`).
const N_SET_SIZE: u16 = 581;
const N_OBJECT_ASSIGN: u16 = 18;
const N_OBJECT_ENTRIES: u16 = 19;
const N_ARRAY_FROM: u16 = 20;
const N_ARRAY_OF: u16 = 21;
const N_PROMISE: u16 = 22;
const N_DATE: u16 = 25;
const N_REGEXP: u16 = 26;
const N_MATH_POW: u16 = 28;
const N_MATH_SIGN: u16 = 29;
const N_MATH_TRUNC: u16 = 30;
const N_OBJECT_FROM_ENTRIES: u16 = 34;
const N_OBJECT_FREEZE: u16 = 35;
const N_OBJECT_IS_FROZEN: u16 = 36;
const N_OBJECT_SEAL: u16 = 125;
const N_OBJECT_IS_SEALED: u16 = 126;
const N_OBJECT_PREVENT_EXT: u16 = 127;
const N_OBJECT_IS_EXTENSIBLE: u16 = 128;
const N_OBJECT_GET_OWN_NAMES: u16 = 37;
const N_OBJECT_GET_OWN_SYMBOLS: u16 = 158;
const N_ENCODE_URI_COMPONENT: u16 = 159;
const N_DECODE_URI_COMPONENT: u16 = 160;
const N_ENCODE_URI: u16 = 161;
const N_DECODE_URI: u16 = 162;
const N_STRUCTURED_CLONE: u16 = 163;
const N_BTOA: u16 = 164;
const N_ATOB: u16 = 165;
const N_INTL_NUMBER_FORMAT: u16 = 166;
const N_INTL_DATETIME_FORMAT: u16 = 167;
const N_SET_TIMEOUT: u16 = 211;
const N_CLEAR_TIMEOUT: u16 = 212;
const N_QUEUE_MICROTASK: u16 = 213;
const N_ARRAY_BUFFER_IS_VIEW: u16 = 214;
const N_INTL_FORMAT: u16 = 215;
const N_EVAL: u16 = 216;
const N_INTL_RESOLVED_OPTIONS: u16 = 217;
const N_INTL_SUPPORTED_LOCALES: u16 = 218;
const N_INTL_FORMAT_TO_PARTS: u16 = 219;
/// `Intl.NumberFormat/DateTimeFormat.prototype.formatRange(x, y)` and
/// `formatRangeToParts(x, y)` (high, collision-free ids).
const N_INTL_FORMAT_RANGE: u16 = 900;
const N_INTL_FORMAT_RANGE_TO_PARTS: u16 = 901;
/// `get %IteratorPrototype%[Symbol.toStringTag]` → the string "Iterator".
const N_ITERATOR_TAG_GET: u16 = 902;
/// `set %IteratorPrototype%[Symbol.toStringTag]` — SetterThatIgnoresPrototypeProperties.
const N_ITERATOR_TAG_SET: u16 = 903;
/// `get %IteratorPrototype%.constructor` → `%Iterator%`.
const N_ITERATOR_CTOR_GET: u16 = 914;
/// `set %IteratorPrototype%.constructor` — SetterThatIgnoresPrototypeProperties.
const N_ITERATOR_CTOR_SET: u16 = 915;
/// A readable static method bound to a `[constructor, name]` pair (so a detached call
/// still routes to the constructor's `call_method` static dispatch).
const N_STATIC_METHOD: u16 = 220;
const N_INTL_COLLATOR: u16 = 207;
const N_INTL_PLURAL_RULES: u16 = 208;
/// `Intl.Collator.prototype.compare` (a bound function value).
const N_INTL_COMPARE: u16 = 209;
/// `Intl.PluralRules.prototype.select`.
const N_INTL_PLURAL_SELECT: u16 = 210;
/// `Intl.PluralRules.prototype.selectRange`.
const N_INTL_PLURAL_SELECT_RANGE: u16 = 671;
/// `Intl.ListFormat` constructor.
const N_INTL_LIST_FORMAT: u16 = 221;
/// `Intl.ListFormat.prototype.format`.
const N_INTL_LIST_FORMAT_FORMAT: u16 = 222;
/// `Intl.RelativeTimeFormat` constructor.
const N_INTL_REL_TIME: u16 = 223;
/// `Intl.RelativeTimeFormat.prototype.format`.
const N_INTL_REL_TIME_FORMAT: u16 = 224;
/// `Intl.DisplayNames` constructor.
const N_INTL_DISPLAY_NAMES: u16 = 225;
/// `Intl.DisplayNames.prototype.of`.
const N_INTL_DISPLAY_NAMES_OF: u16 = 226;
/// `Intl.Segmenter` constructor.
const N_INTL_SEGMENTER: u16 = 227;
/// `Intl.Segmenter.prototype.segment`.
const N_INTL_SEGMENTER_SEGMENT: u16 = 228;
/// `Intl.getCanonicalLocales(locales)` — canonicalizes a locale list.
const N_INTL_GET_CANONICAL_LOCALES: u16 = 460;
/// `Intl.supportedValuesOf(key)` — the supported values for a key.
const N_INTL_SUPPORTED_VALUES_OF: u16 = 461;
/// `Intl.Locale` constructor. (New `N_*` ids for the prototype/branding work start
/// at 500 per the task spec.)
const N_INTL_LOCALE: u16 = 500;
/// A `get Intl.Locale.prototype.<accessor>` getter (`language`/`script`/`region`/
/// `baseName`/`calendar`/`numberingSystem`/`hourCycle`/`caseFirst`/`collation`/
/// `numeric`). A bound native whose target string names the accessor; rejects a
/// `this` lacking the `[[InitializedLocale]]` slot with a TypeError.
const N_INTL_LOCALE_ACCESSOR: u16 = 501;
/// `Intl.Locale.prototype.maximize` / `minimize` / `toString`. A bound native whose
/// target string names the operation; brand-checks the `this` Locale receiver.
const N_INTL_LOCALE_METHOD: u16 = 502;
/// `Intl.DurationFormat` constructor.
const N_INTL_DURATION_FORMAT: u16 = 503;
/// `Intl.DurationFormat.prototype.format` / `formatToParts` / `resolvedOptions`.
/// A bound native whose target string names the method; brand-checks `this`.
const N_INTL_DURATION_METHOD: u16 = 504;
/// A prototype-installed Intl method wrapper that brand-checks its `this` receiver
/// for the service's internal slot before delegating to the underlying native
/// (`format`/`resolvedOptions`/`formatToParts`/`compare`/`select`/`of`/`segment`).
/// The bound-native target is a two-element `[slotMarker, methodName]` array. Lets
/// the existing method natives stay receiver-agnostic while the prototype entry
/// points enforce branding.
const N_INTL_PROTO_METHOD: u16 = 505;
/// A `get Intl.NumberFormat.prototype.format` / `get …DateTimeFormat….format` /
/// `get Intl.Collator.prototype.compare` accessor. A bound native whose target is
/// the `[markerKey, selector]` pair; brand-checks `this`, then returns the
/// per-instance bound function (cached on the instance), an `N_INTL_BOUND_CALL`.
const N_INTL_BOUND_GETTER: u16 = 506;
/// The per-instance bound `format`/`compare` function returned by the
/// `N_INTL_BOUND_GETTER` accessor. A bound native whose target is the
/// `[instance, selector]` pair; it formats/compares against the captured instance,
/// independent of how it is later called (`this` is ignored, per BoundFunction).
const N_INTL_BOUND_CALL: u16 = 507;
/// The shared abstract `%TypedArray%` intrinsic constructor — the value
/// `Object.getPrototypeOf(Int8Array)` returns. Calling or `new`-ing it directly
/// throws a `TypeError`; it carries the generic `from`/`of`/`get [Symbol.species]`
/// statics that every concrete typed-array constructor inherits.
const N_TYPED_ARRAY_ABSTRACT: u16 = 229;
/// `%TypedArray%.from(source, mapFn?, thisArg?)` — builds an instance of the
/// `this` constructor from an array-like / iterable.
const N_TYPED_ARRAY_FROM: u16 = 230;
/// `%TypedArray%.of(...items)` — builds an instance of the `this` constructor.
const N_TYPED_ARRAY_OF: u16 = 231;
/// `get %TypedArray%[Symbol.species]` — returns `this` (the receiver constructor).
const N_TYPED_ARRAY_SPECIES: u16 = 232;
/// `get %TypedArray%.prototype[Symbol.toStringTag]` — the concrete view name.
const N_TYPED_ARRAY_TO_STRING_TAG: u16 = 241;
/// A `get %TypedArray%.prototype.<accessor>` getter (`buffer`/`byteLength`/
/// `byteOffset`/`length`). A bound native whose target string names the accessor;
/// rejects a `this` lacking a `[[TypedArrayName]]` slot with a TypeError.
const N_TYPED_ARRAY_ACCESSOR: u16 = 247;
/// A `get DataView.prototype.<accessor>` getter (`buffer`/`byteLength`/
/// `byteOffset`). A bound native whose target string names the accessor; rejects
/// a `this` lacking a `[[DataView]]` internal slot with a TypeError. (New-id
/// block at 300+ to avoid sibling collisions.)
const N_DATA_VIEW_ACCESSOR: u16 = 300;
/// A `get ArrayBuffer.prototype.<accessor>` getter (`byteLength`/`maxByteLength`/
/// `resizable`/`detached`). A bound native whose target string names the
/// accessor; rejects a `this` lacking an `[[ArrayBufferData]]` slot with a
/// TypeError.
const N_AB_ACCESSOR: u16 = 301;
/// A first-class `DataView.prototype.<method>` (`getInt8`/`setFloat64`/…): a
/// bound native carrying the method name. Calling it requires the call's `this`
/// to have a `[[DataView]]` internal slot (else a TypeError), then re-dispatches
/// through `call_method` so `DataView.prototype.getInt8.call(dv, 0)` and a direct
/// `dv.getInt8(0)` share one implementation.
const N_DATA_VIEW_PROTO_FN: u16 = 302;
/// `$262_detachArrayBuffer(buffer)` — the Test262 host hook (`$262.detachArrayBuffer`).
/// Detaches the given `ArrayBuffer`: zero-lengths its backing store, empties every
/// typed-array view over it, and flags it detached so subsequent operations throw
/// per spec. Returns `null` (the spec-mandated result of `DetachArrayBuffer`).
/// New-id block at 360+ per the batch's allocation rule.
const N_DETACH_ARRAY_BUFFER: u16 = 360;
/// `escape(string)` (Annex B.2.1): legacy percent/`%u`-escape of a string.
const N_ESCAPE: u16 = 420;
/// `unescape(string)` (Annex B.2.2): inverse of [`N_ESCAPE`].
const N_UNESCAPE: u16 = 421;
/// A RegExp legacy static getter (Annex B.2.5) — a bound native carrying the
/// accessor's key (`input`/`lastMatch`/`$1`/…); brand-checks `this === RegExp`.
const N_REGEXP_LEGACY_GET: u16 = 422;
/// The `set RegExp.input` / `set RegExp.$_` legacy setter (Annex B.2.5).
const N_REGEXP_LEGACY_SET: u16 = 423;
/// Sentinel "native base kind" id for `Array` — a namespace-object constructor
/// (no real native id), used by the native-subclassing machinery
/// (`class S extends Array {}`) to mark that a derived instance must be created as
/// a dense `Cell::Array` rather than a plain object. Never a real callable's id.
const N_BASE_ARRAY: u16 = 480;
/// Sentinel "native base kind" id for `Object` — a namespace-object constructor,
/// marking that a derived instance is an ordinary object (its `[[Prototype]]`
/// from the subclass). Never a real callable's id.
const N_BASE_OBJECT: u16 = 481;
/// `%IteratorHelperPrototype%.next` — drives a lazy ES2025 iterator-helper object
/// (`map`/`filter`/`take`/`drop`/`flatMap`) one step at a time.
const N_ITER_HELPER_NEXT: u16 = 340;
/// `%IteratorHelperPrototype%.return` — closes the helper's underlying iterator.
const N_ITER_HELPER_RETURN: u16 = 341;
/// `%WrapForValidIteratorPrototype%.next` — the `Iterator.from` wrapper's `next`.
const N_ITER_WRAP_NEXT: u16 = 342;
/// `%WrapForValidIteratorPrototype%.return` — the `Iterator.from` wrapper's `return`.
const N_ITER_WRAP_RETURN: u16 = 343;
/// `Iterator.concat` — the `iterator-sequencing` static (lazy concatenation).
const N_ITERATOR_CONCAT: u16 = 344;
/// `%ConcatIteratorPrototype%.next` — drives the lazy `Iterator.concat` result.
const N_ITER_CONCAT_NEXT: u16 = 345;
/// The eager-generator iterator's `next` — surfaced as a real method so it can be
/// read once (GetIteratorDirect) and called by the lazy iterator helpers.
const N_GEN_ITER_NEXT: u16 = 346;
/// The eager-generator iterator's `return` method.
const N_GEN_ITER_RETURN: u16 = 347;
/// A lazy generator's `next(v)` — resumes the suspended frame, injecting `v`.
const N_GEN_NEXT: u16 = 520;
/// A lazy generator's `return(v)` — resumes as `return v` (runs `finally`s).
const N_GEN_RETURN: u16 = 521;
/// A lazy generator's `throw(e)` — resumes by throwing `e` at the suspension.
const N_GEN_THROW: u16 = 522;
/// An async generator's `next(v)` — like `N_GEN_NEXT` but *always* returns a
/// promise, so even a brand-check failure (a `this` that is not an async
/// generator) rejects rather than throwing synchronously.
const N_ASYNC_GEN_NEXT: u16 = 523;
/// An async generator's `return(v)` — the promise-returning counterpart.
const N_ASYNC_GEN_RETURN: u16 = 524;
/// An async generator's `throw(e)` — the promise-returning counterpart.
const N_ASYNC_GEN_THROW: u16 = 525;
/// Async-coroutine resume on fulfilment: bound to the controller object, called
/// as a microtask reaction when an awaited promise fulfils — resumes the parked
/// async body with the fulfilment value at the `await` point.
const N_ASYNC_RESUME_FULFILL: u16 = 560;
/// Async-coroutine resume on rejection: bound to the controller object, called as
/// a microtask reaction when an awaited promise rejects — resumes the parked async
/// body by throwing the rejection reason at the `await` point.
const N_ASYNC_RESUME_REJECT: u16 = 561;
/// Async-generator resume on an awaited value fulfilling: bound to the async
/// generator object, called as a microtask reaction to resume the parked body at
/// the `await` point with the settled value.
const N_ASYNC_GEN_AWAIT_FULFILL: u16 = 562;
/// Async-generator resume on an awaited value rejecting: resumes the parked body
/// by throwing the rejection reason at the `await` point.
const N_ASYNC_GEN_AWAIT_REJECT: u16 = 563;
/// `AsyncGeneratorAwaitReturn` fulfilment: the awaited `return(v)` value settled;
/// resolve the front request with `{value, done:true}` and drain the queue.
const N_ASYNC_GEN_RETURN_FULFILL: u16 = 564;
/// `AsyncGeneratorAwaitReturn` rejection: the awaited `return(v)` value rejected;
/// reject the front request with the reason and drain the queue.
const N_ASYNC_GEN_RETURN_REJECT: u16 = 565;
/// `Object.prototype.__defineGetter__(P, getter)` (Annex B).
const N_OBJ_DEFINE_GETTER: u16 = 348;
/// `Object.prototype.__defineSetter__(P, setter)` (Annex B).
const N_OBJ_DEFINE_SETTER: u16 = 349;
/// `Object.prototype.__lookupGetter__(P)` (Annex B).
const N_OBJ_LOOKUP_GETTER: u16 = 350;
/// `Object.prototype.__lookupSetter__(P)` (Annex B).
const N_OBJ_LOOKUP_SETTER: u16 = 351;
/// `get/set Object.prototype.__proto__` (Annex B accessor).
const N_OBJ_PROTO_GET: u16 = 352;
const N_OBJ_PROTO_SET: u16 = 353;
/// `%ConcatIteratorPrototype%.return` — closes the active inner iterator.
const N_ITER_CONCAT_RETURN: u16 = 354;
/// `Iterator.zip` / `Iterator.zipKeyed` (the `joint-iteration` statics).
const N_ITERATOR_ZIP: u16 = 355;
const N_ITERATOR_ZIP_KEYED: u16 = 356;
/// `%ZipIteratorPrototype%.next` / `.return` driving a lazy zip result.
const N_ITER_ZIP_NEXT: u16 = 357;
const N_ITER_ZIP_RETURN: u16 = 358;
/// `%IteratorPrototype%[Symbol.dispose]` — calls the iterator's `return`.
const N_ITERATOR_DISPOSE: u16 = 359;
/// `%ThrowTypeError%` — the shared poisoned accessor used as a strict
/// `arguments` object's `callee` getter/setter; calling it always throws a
/// `TypeError`. (New native-id range starts at 400.)
const N_THROW_TYPE_ERROR: u16 = 400;
/// `Function.prototype[Symbol.hasInstance]` — OrdinaryHasInstance(this, V):
/// reports whether `V` is in `this` function's `.prototype` chain (a bound
/// function defers to its target).
const N_FN_HAS_INSTANCE: u16 = 401;
/// The `%TypedArray%.prototype` methods exposed as first-class own properties,
/// each paired with its spec `length` (own `length` data property). Dispatched
/// through [`N_TYPED_ARRAY_PROTO_FN`].
const TYPED_ARRAY_PROTO_METHODS: &[(&str, u32)] = &[
    ("at", 1),
    ("copyWithin", 2),
    ("entries", 0),
    ("every", 1),
    ("fill", 1),
    ("filter", 1),
    ("find", 1),
    ("findIndex", 1),
    ("findLast", 1),
    ("findLastIndex", 1),
    ("forEach", 1),
    ("includes", 1),
    ("indexOf", 1),
    ("join", 1),
    ("keys", 0),
    ("lastIndexOf", 1),
    ("map", 1),
    ("reduce", 1),
    ("reduceRight", 1),
    ("reverse", 0),
    ("set", 1),
    ("slice", 2),
    ("some", 1),
    ("sort", 1),
    ("subarray", 2),
    ("toLocaleString", 0),
    ("toReversed", 0),
    ("toSorted", 1),
    ("toString", 0),
    ("values", 0),
    ("with", 2),
];
/// The typed-array constructors occupy `[BASE, BASE + KINDS.len())`; the id minus
/// the base indexes [`TYPED_ARRAY_KINDS`].
const N_TYPED_ARRAY_BASE: u16 = 168;
/// `(name, bytes-per-element)` for each typed-array kind, in id order. Kinds 9
/// and 10 hold **BigInt** elements (8 bytes, signed i64 / unsigned u64); the
/// element read/write paths special-case them (see [`encode_typed_element`] /
/// [`decode_typed_element`] and the BigInt coercion in [`Realm::typed_set`]).
const TYPED_ARRAY_KINDS: [(&str, u8); 12] = [
    ("Int8Array", 1),
    ("Uint8Array", 1),
    ("Uint8ClampedArray", 1),
    ("Int16Array", 2),
    ("Uint16Array", 2),
    ("Int32Array", 4),
    ("Uint32Array", 4),
    ("Float32Array", 4),
    ("Float64Array", 8),
    ("BigInt64Array", 8),
    ("BigUint64Array", 8),
    // Kind 11: IEEE-754 half precision (ES2025). Not contiguous with the other
    // float kinds, but the encode/decode paths dispatch by index, so its position
    // after the BigInt kinds is fine (`is_bigint_typed_kind` only matches 9/10).
    ("Float16Array", 2),
];
/// Whether typed-array `kind` index holds BigInt elements (9 = `BigInt64Array`,
/// 10 = `BigUint64Array`).
#[must_use]
pub(crate) fn is_bigint_kind(kind: u8) -> bool {
    kind == 9 || kind == 10
}
// Moved out of [168, 179) so the typed-array kind block can grow to 11 entries
// (the two BigInt kinds occupy the former 177/178 slots).
const N_ARRAY_BUFFER: u16 = 234;
const N_DATA_VIEW: u16 = 235;
// `WebAssembly.validate` — decode a module and report well-formedness.
const N_WASM_VALIDATE: u16 = 184;
// `WebAssembly.instantiate` — build an instance object with callable exports.
const N_WASM_INSTANTIATE: u16 = 185;
// A WASM export wrapper (a bound native whose target carries the module bytes +
// the export name).
const N_WASM_CALL: u16 = 186;
/// The `Function` global — supports `typeof`/`instanceof` (any callable); the
/// dynamic `Function(...)` constructor (runtime code compilation) is unsupported.
const N_FUNCTION: u16 = 187;
/// `new WebAssembly.Module(bytes)` — a decoded/validated module object.
const N_WASM_MODULE: u16 = 188;
/// `new WebAssembly.Instance(module, imports?)` — an instance with `.exports`.
const N_WASM_INSTANCE: u16 = 189;
/// `WebAssembly.compile(bytes)` — async compile → `Promise<Module>`.
const N_WASM_COMPILE: u16 = 190;
/// `new WebAssembly.Global({value, mutable}, init)` — a typed value cell.
const N_WASM_GLOBAL: u16 = 191;
/// The `.value` getter / setter of a `WebAssembly.Global` (bound to the global).
const N_WASM_GLOBAL_GET: u16 = 192;
const N_WASM_GLOBAL_SET: u16 = 193;
/// `new WebAssembly.Memory({initial, maximum?})` and its `.buffer` getter / `grow`.
const N_WASM_MEMORY: u16 = 194;
const N_WASM_MEM_BUFFER_GET: u16 = 195;
const N_WASM_MEM_GROW: u16 = 196;
/// A WASM linear-memory page is 64 KiB.
const WASM_PAGE: usize = 65536;
/// `new WebAssembly.Table({element, initial, maximum?})` and its `.length` getter
/// plus `get`/`set`/`grow` methods.
const N_WASM_TABLE: u16 = 197;
const N_WASM_TABLE_LEN: u16 = 198;
const N_WASM_TABLE_GET: u16 = 199;
const N_WASM_TABLE_SET: u16 = 200;
const N_WASM_TABLE_GROW: u16 = 201;
/// Static `WebAssembly.Module.exports(module)` / `.imports(module)` introspection.
const N_WASM_MODULE_EXPORTS: u16 = 202;
const N_WASM_MODULE_IMPORTS: u16 = 203;
// Hidden slots on a WASM export wrapper's data object.
const WASM_BYTES: &str = "\u{0}wbytes";
const WASM_EXPORT: &str = "\u{0}wexport";
const WASM_IMPORTS: &str = "\u{0}wimports";
/// Marks an object built by `new WebAssembly.Module(...)`.
const WASM_IS_MODULE: &str = "\u{0}wmodule";
/// The instance id on a WASM export wrapper's data object (keys `wasm_states`, so
/// memory/globals persist across calls of the same instance).
const WASM_INSTANCE_ID: &str = "\u{0}winst";
/// Hidden slots on a `WebAssembly.Global`: its current value, value type, and
/// mutability.
const WASM_GLOBAL_VALUE: &str = "\u{0}gval";
const WASM_GLOBAL_TYPE: &str = "\u{0}gtype";
const WASM_GLOBAL_MUTABLE: &str = "\u{0}gmut";
/// Hidden slots on a `WebAssembly.Memory`: its `ArrayBuffer`, page count, and max.
const WASM_MEM_BUFFER: &str = "\u{0}mbuf";
const WASM_MEM_PAGES: &str = "\u{0}mpages";
const WASM_MEM_MAX: &str = "\u{0}mmax";
/// Hidden slots on a `WebAssembly.Table`: its element (function-ref) array and max.
const WASM_TABLE_ELEMS: &str = "\u{0}telems";
const WASM_TABLE_MAX: &str = "\u{0}tmax";
// `Object.prototype.*` methods (the receiver arrives as `this`). (Moved off 179
// to a free id so the 12th typed-array kind, `Float16Array`, can take
// `N_TYPED_ARRAY_BASE + 11 = 179` and keep the kind ids contiguous.)
const N_OBJ_PROTO_TOSTRING: u16 = 675;
/// `Object.prototype.toLocaleString` — per spec it is `return Invoke(this,
/// "toString")` (NOT the `[[Class]]` tag), so a user-overridden `toString` on the
/// receiver (or its prototype) is honored, with the *original* `this` value.
const N_OBJ_PROTO_TOLOCALESTRING: u16 = 964;
/// `Function.prototype.toString` — a dedicated dispatch id (distinct from the
/// shared `N_ARRAY_PROTO_FN` used by `call`/`apply`/`bind`) so an *indirect*
/// invocation (`Function.prototype.toString.call(fn)`, `String(fn)`, `fn + ""`,
/// ToPrimitive) renders the callable's source-representation instead of being
/// misrouted to `Array.prototype.toString` (which would yield `"[object
/// Function]"`). Reads its receiver from `this`; a non-callable `this` throws.
const N_FUNCTION_TO_STRING: u16 = 965;
/// `%Segments.prototype%.containing(index)` — a bound native on the object
/// returned by `Intl.Segmenter.prototype.segment`.
const N_INTL_SEGMENTS_CONTAINING: u16 = 676;
/// `Atomics.*` methods (single-agent semantics over an integer `TypedArray`;
/// atomicity is trivial without concurrent agents).
const N_ATOMICS_ADD: u16 = 677;
const N_ATOMICS_SUB: u16 = 678;
const N_ATOMICS_AND: u16 = 679;
const N_ATOMICS_OR: u16 = 680;
const N_ATOMICS_XOR: u16 = 681;
const N_ATOMICS_EXCHANGE: u16 = 682;
const N_ATOMICS_COMPARE_EXCHANGE: u16 = 683;
const N_ATOMICS_LOAD: u16 = 684;
const N_ATOMICS_STORE: u16 = 685;
const N_ATOMICS_IS_LOCK_FREE: u16 = 686;
/// `Atomics.pause([N])` — a spin-loop hint; a single-agent no-op returning
/// `undefined` (validates that `N`, if present, is an integral Number).
const N_ATOMICS_PAUSE: u16 = 904;
/// `Atomics.notify(typedArray, index, count)` — wakes up to `count` agents
/// waiting on `(buffer, index)`. In this single-agent engine no agent is ever
/// waiting, so after validating the (integer TypedArray, in-range index,
/// non-negative count) it always returns `0`.
const N_ATOMICS_NOTIFY: u16 = 909;
/// `Atomics.wait(typedArray, index, value, timeout)` — blocks the agent until
/// notified. Requires a shared integer TypedArray whose agent CanBlock; the main
/// agent cannot block and a non-shared buffer is invalid, so after validation
/// this throws a TypeError in the single-agent engine.
const N_ATOMICS_WAIT: u16 = 910;
/// `SharedArrayBuffer.prototype` getter (byteLength / maxByteLength / growable).
/// Like [`N_AB_ACCESSOR`] but brand-validates the receiver is a *shared* buffer —
/// so `SharedArrayBuffer.prototype.byteLength` called on a plain `ArrayBuffer`
/// (or vice versa for the AB getter) throws a TypeError.
const N_SAB_ACCESSOR: u16 = 911;
/// `SharedArrayBuffer.prototype` method (`grow` / `slice`). Like
/// [`N_AB_PROTO_FN`] but brand-validates the receiver is a *shared* buffer, so
/// `SharedArrayBuffer.prototype.slice` on a plain `ArrayBuffer` is a TypeError
/// (and the AB methods reject a shared receiver).
const N_SAB_PROTO_FN: u16 = 912;
/// `$262_IsHTMLDDA()` — the Test262 host hook (`$262.IsHTMLDDA`). Returns a fresh
/// [[IsHTMLDDA]] exotic object (the Annex-B `document.all` value): an ordinary
/// object branded with [`crate::realm::HTMLDDA_SLOT`], so `typeof` reports
/// `"undefined"`, it is falsy, it is loosely equal to `null`/`undefined`, and its
/// `[[Call]]` returns `null`.
const N_HTMLDDA: u16 = 913;
/// `$262_createRealm()` — the Test262 host hook (`$262.createRealm`). Builds a
/// second global environment (a fresh set of intrinsics: distinct `Array`,
/// `Object`, `TypeError`, … from the current realm's) on the shared heap and
/// returns a `$262`-shaped realm object whose `.global` is the new global object
/// and whose `.evalScript(src)` runs code in that environment.
const N_262_CREATE_REALM: u16 = 950;
/// `$262.createRealm().evalScript(src)` — evaluates `src` in the receiver
/// realm's global environment (swapping in its scope + intrinsics), returning the
/// completion value.
const N_262_EVAL_SCRIPT: u16 = 951;
/// `$262.evalScript(src)` — evaluates `src` as a **Script** in the *current*
/// realm's global environment (distinct from an indirect `eval`: a script's
/// top-level `let`/`const`/`class` become persistent global lexical bindings,
/// and a sloppy script's `var`/function bindings persist too). Wired by the
/// runner's JS prelude to the top-level `$262.evalScript`.
const N_262_EVAL_SCRIPT_MAIN: u16 = 963;
/// `$262.agent.start(src)` — spawn a worker agent: create a fresh realm, install
/// a worker-side `$262.agent`, and run `src` eagerly to completion in it.
const N_262_AGENT_START: u16 = 952;
/// `$262.agent.broadcast(sab)` / `safeBroadcast(sab)` — deliver the
/// SharedArrayBuffer to every worker's registered `receiveBroadcast` callback.
const N_262_AGENT_BROADCAST: u16 = 953;
/// `$262.agent.getReport()` — pop the front of the shared report queue (or `null`).
const N_262_AGENT_GET_REPORT: u16 = 954;
/// `$262.agent.getReportAsync()` — the same, wrapped in a resolved promise.
const N_262_AGENT_GET_REPORT_ASYNC: u16 = 955;
/// `$262.agent.report(msg)` — push `ToString(msg)` onto the shared report queue.
const N_262_AGENT_REPORT: u16 = 956;
/// `$262.agent.receiveBroadcast(cb)` — register a worker callback for the next
/// `broadcast` (deferred: invoked when the main agent broadcasts).
const N_262_AGENT_RECEIVE_BROADCAST: u16 = 957;
/// `$262.agent.sleep(ms)` — a cooperative no-op (no real clock to block on).
const N_262_AGENT_SLEEP: u16 = 958;
/// `$262.agent.monotonicNow()` — a monotonic millisecond clock reading.
const N_262_AGENT_MONOTONIC_NOW: u16 = 959;
/// `Atomics.waitAsync(view, idx, value, timeout)` — the async, non-blocking
/// counterpart of `Atomics.wait`; returns `{ async, value }` (a promise when it
/// would block) settled `"ok"` by a matching `notify` or `"timed-out"`.
const N_ATOMICS_WAIT_ASYNC: u16 = 960;
/// A bound native (target = the waitAsync promise) that a finite-timeout waiter's
/// macrotask calls to settle it `"timed-out"` if still pending.
const N_ATOMICS_ASYNC_TIMEOUT: u16 = 961;
/// `%GeneratorFunction%` — builds a `function*` from dynamic source (like
/// `%Function%`); reachable via `Object.getPrototypeOf(function*(){}).constructor`.
const N_GENERATOR_FUNCTION_CTOR: u16 = 905;
/// `%AsyncGeneratorFunction%` — builds an `async function*` from dynamic source.
const N_ASYNC_GENERATOR_FUNCTION_CTOR: u16 = 906;
/// `%AsyncIteratorPrototype%[@@asyncDispose]` — calls the iterator's `return` and
/// returns a promise (fulfilled with `undefined`, rejected if `return` throws).
const N_ASYNC_ITERATOR_DISPOSE: u16 = 907;
/// An anonymous fulfillment handler that ignores its argument and returns
/// `undefined` — the `onFulfilled` of the `@@asyncDispose` `.then`-chain.
const N_RETURN_UNDEFINED: u16 = 908;
/// `SharedArrayBuffer` — a growable byte store (single-agent: no cross-agent
/// sharing, so it behaves as an `ArrayBuffer` that only ever grows). Its bytes
/// live in the same `ARRAY_BUFFER_BYTES` slot, so typed arrays and `Atomics`
/// operate over it unchanged.
const N_SHARED_ARRAY_BUFFER: u16 = 687;
const N_OBJ_PROTO_VALUEOF: u16 = 180;
const N_OBJ_PROTO_HASOWN: u16 = 181;
const N_OBJ_PROTO_ISPROTOTYPEOF: u16 = 182;
const N_OBJ_PROTO_PROPISENUM: u16 = 183;
const N_OBJECT_CREATE: u16 = 107;
const N_OBJECT_GET_PROTO: u16 = 108;
const N_OBJECT_SET_PROTO: u16 = 109;
const N_OBJECT_DEFINE_PROP: u16 = 110;
const N_OBJECT_GET_OWN_DESC: u16 = 111;
const N_WEAKMAP: u16 = 112;
const N_OBJECT_IS: u16 = 123;
const N_OBJECT_HAS_OWN: u16 = 129;
const N_OBJECT_GROUP_BY: u16 = 138;
const N_OBJECT_GET_OWN_DESCS: u16 = 130;
const N_WEAKREF: u16 = 131;
const N_FINALIZATION_REGISTRY: u16 = 132;
// The `%Iterator%` abstract constructor and its `Iterator.from` static.
const N_ITERATOR: u16 = 236;
const N_ITERATOR_FROM: u16 = 237;
// `%IteratorPrototype%[Symbol.iterator]` — returns its `this` receiver.
const N_ITERATOR_PROTO_SELF: u16 = 238;
// A first-class `Iterator.prototype.<helper>` (map/filter/take/…) bound native:
// the method name rides in the bound target, the receiver is `this`.
const N_ITERATOR_PROTO_FN: u16 = 239;

/// The ES2025 `Iterator.prototype` helper method names installed on
/// `%IteratorPrototype%` as first-class functions.
const ITERATOR_PROTO_METHODS: &[&str] = &[
    "map", "filter", "take", "drop", "flatMap", "reduce", "toArray", "forEach", "some", "every",
    "find",
];
const N_OBJECT_DEFINE_PROPS: u16 = 124;
const N_WEAKSET: u16 = 113;
const N_REFLECT_GET: u16 = 114;
const N_REFLECT_SET: u16 = 115;
const N_REFLECT_HAS: u16 = 116;
const N_REFLECT_OWN_KEYS: u16 = 117;
const N_REFLECT_DELETE: u16 = 118;
const N_REFLECT_APPLY: u16 = 119;
const N_REFLECT_CONSTRUCT: u16 = 120;
const N_REFLECT_DEFINE_PROP: u16 = 135;
const N_REFLECT_GET_OWN_DESC: u16 = 136;
const N_REFLECT_GET_PROTO: u16 = 137;
/// `Reflect.setPrototypeOf` / `preventExtensions` — like the `Object.*` forms but
/// returning a boolean success flag.
const N_REFLECT_SET_PROTO: u16 = 204;
const N_REFLECT_PREVENT_EXT: u16 = 205;
/// `Reflect.isExtensible(target)` — like `Object.isExtensible` but the target
/// MUST be an Object (a primitive throws a TypeError, where `Object.isExtensible`
/// returns `false`), so it has its own dispatch id.
const N_REFLECT_IS_EXTENSIBLE: u16 = 252;
/// A first-class `Array.prototype.<method>` value: a bound native carrying the
/// method name; calling it (via `.call`/`.apply`) dispatches that array method on
/// the supplied `this` (so `Array.prototype.slice.call(arguments)` works).
const N_ARRAY_PROTO_FN: u16 = 206;
/// A first-class `ArrayBuffer.prototype.<method>` (e.g. `slice`). Like
/// [`N_ARRAY_PROTO_FN`] but validates that the call's `this` has an
/// `[[ArrayBufferData]]` internal slot first (throwing a `TypeError` otherwise),
/// so `ArrayBuffer.prototype.slice.call(nonBuffer)` rejects per spec rather than
/// being silently treated as a generic array-like.
const N_AB_PROTO_FN: u16 = 233;
/// A first-class `BigInt.prototype.<method>` (`toString`/`valueOf`/
/// `toLocaleString`). Like [`N_ARRAY_PROTO_FN`] but applies `thisBigIntValue`:
/// the call's `this` must be a BigInt or a BigInt wrapper object, else a
/// `TypeError` (so `BigInt.prototype.valueOf.call({})` rejects per spec).
const N_BIGINT_PROTO_FN: u16 = 240;
/// A first-class `Date.prototype.<method>`: a bound native carrying the method
/// name. Calling it requires the call's `this` to have a `[[DateValue]]`
/// (i.e. be a Date), else a `TypeError`; otherwise it re-dispatches through
/// `call_method` so `Date.prototype.getTime.call(d)` and direct `d.getTime()`
/// share one implementation.
const N_DATE_PROTO_FN: u16 = 242;
/// The `Date.prototype` methods exposed as first-class values.
const DATE_PROTO_METHODS: &[&str] = &[
    "toTemporalInstant",
    "getTime",
    "valueOf",
    "getFullYear",
    "getUTCFullYear",
    "getMonth",
    "getUTCMonth",
    "getDate",
    "getUTCDate",
    "getDay",
    "getUTCDay",
    "getHours",
    "getUTCHours",
    "getMinutes",
    "getUTCMinutes",
    "getSeconds",
    "getUTCSeconds",
    "getMilliseconds",
    "getUTCMilliseconds",
    "getTimezoneOffset",
    "toISOString",
    "toDateString",
    "toTimeString",
    "toString",
    "toUTCString",
    "toLocaleDateString",
    "toLocaleTimeString",
    "toLocaleString",
    "setTime",
    "setFullYear",
    "setUTCFullYear",
    "setMonth",
    "setUTCMonth",
    "setDate",
    "setUTCDate",
    "setHours",
    "setUTCHours",
    "setMinutes",
    "setUTCMinutes",
    "setSeconds",
    "setUTCSeconds",
    "setMilliseconds",
    "setUTCMilliseconds",
    // Annex B.2.4: legacy two-digit-year accessors.
    "getYear",
    "setYear",
];
/// A first-class `%TypedArray%.prototype.<method>` (e.g. `map`, `slice`, `every`).
/// Like [`N_ARRAY_PROTO_FN`] but validates that the call's `this` has a
/// `[[TypedArrayName]]` internal slot first (throwing a `TypeError` otherwise),
/// and never applies the plain-`Array` result conversion — so e.g.
/// `Int8Array.prototype.map.call(ta, fn)` returns a same-kind typed array.
const N_TYPED_ARRAY_PROTO_FN: u16 = 246;
/// A first-class `RegExp.prototype.<method>` (`exec`/`test`/`compile`/`toString`
/// and the `@@match`/`@@matchAll`/`@@replace`/`@@search`/`@@split` symbol
/// methods). A bound native carrying the method name; calling it brand-validates
/// the call's `this` inside the handler (most methods only require an Object —
/// they read `exec`, `global`, … off it — while `exec`/`compile` require an
/// actual RegExp). The new ids start at 280 to avoid sibling collisions.
const N_REGEXP_PROTO_FN: u16 = 280;
/// A `get RegExp.prototype.<accessor>` getter (`source`/`flags`/`global`/
/// `ignoreCase`/`multiline`/`dotAll`/`sticky`/`unicode`/`unicodeSets`/
/// `hasIndices`). A bound native carrying the accessor name; calling it validates
/// the receiver is a RegExp (or the `RegExp.prototype` sentinel) and returns the
/// flag/source/flags value, else a `TypeError`.
const N_REGEXP_ACCESSOR: u16 = 281;
/// `get RegExp[Symbol.species]` — a bound native (target: the `RegExp`
/// constructor) whose getter returns its `this` receiver.
const N_REGEXP_SPECIES: u16 = 282;
/// The `RegExp.prototype` methods exposed as first-class string-keyed values.
const REGEXP_PROTO_METHODS: &[&str] = &["exec", "test", "compile", "toString"];
/// The `RegExp.prototype` `get` accessors (string keys).
const REGEXP_ACCESSORS: &[&str] = &[
    "source",
    "flags",
    "global",
    "ignoreCase",
    "multiline",
    "dotAll",
    "sticky",
    "unicode",
    "unicodeSets",
    "hasIndices",
];
/// The `RegExp.prototype` well-known-symbol methods (`@@match`, …) and the method
/// name `call_method` dispatches each to.
const REGEXP_SYMBOL_METHODS: &[(&str, &str)] = &[
    ("match", "match"),
    ("matchAll", "matchAll"),
    ("replace", "replace"),
    ("search", "search"),
    ("split", "split"),
];
/// The `Array.prototype` methods exposed as first-class values (each re-dispatched
/// through `call_method`).
const ARRAY_PROTO_METHODS: &[&str] = &[
    "slice",
    "splice",
    "map",
    "filter",
    "forEach",
    "reduce",
    "reduceRight",
    "indexOf",
    "lastIndexOf",
    "includes",
    "find",
    "findIndex",
    "findLast",
    "findLastIndex",
    "some",
    "every",
    "join",
    "concat",
    "reverse",
    "sort",
    "fill",
    "copyWithin",
    "flat",
    "flatMap",
    "at",
    "push",
    "pop",
    "shift",
    "unshift",
    "keys",
    "values",
    "entries",
    "toString",
    "toLocaleString",
    "with",
    "toReversed",
    "toSorted",
    "toSpliced",
];
/// `String.prototype` methods exposed as first-class values.
const STRING_PROTO_METHODS: &[&str] = &[
    "slice",
    "substring",
    "substr",
    "charAt",
    "charCodeAt",
    "codePointAt",
    "indexOf",
    "lastIndexOf",
    "includes",
    "startsWith",
    "endsWith",
    "split",
    "replace",
    "replaceAll",
    "match",
    "matchAll",
    "search",
    "toUpperCase",
    "toLowerCase",
    "toLocaleUpperCase",
    "toLocaleLowerCase",
    "trim",
    "trimStart",
    "trimEnd",
    "padStart",
    "padEnd",
    "repeat",
    "concat",
    "at",
    "normalize",
    "localeCompare",
    "toString",
    "valueOf",
    "isWellFormed",
    "toWellFormed",
    // Annex B.2.3: legacy HTML wrapper methods.
    "anchor",
    "big",
    "blink",
    "bold",
    "fixed",
    "fontcolor",
    "fontsize",
    "italics",
    "link",
    "small",
    "strike",
    "sub",
    "sup",
];
/// `Number.prototype` methods exposed as first-class values.
const NUMBER_PROTO_METHODS: &[&str] = &[
    "toFixed",
    "toPrecision",
    "toExponential",
    "toString",
    "valueOf",
    "toLocaleString",
];
/// `Boolean.prototype` methods exposed as first-class values.
const BOOLEAN_PROTO_METHODS: &[&str] = &["toString", "valueOf"];
/// `BigInt.prototype` methods exposed as first-class values.
const BIGINT_PROTO_METHODS: &[&str] = &["toString", "toLocaleString", "valueOf"];
/// `Set.prototype` methods exposed as first-class values.
const SET_PROTO_METHODS: &[&str] = &[
    "add",
    "has",
    "delete",
    "clear",
    "forEach",
    "keys",
    "values",
    "entries",
    "union",
    "intersection",
    "difference",
    "symmetricDifference",
    "isSubsetOf",
    "isSupersetOf",
    "isDisjointFrom",
];
/// `Map.prototype` methods exposed as first-class values.
const MAP_PROTO_METHODS: &[&str] = &[
    "set",
    "get",
    "has",
    "delete",
    "clear",
    "forEach",
    "keys",
    "values",
    "entries",
    "getOrInsert",
    "getOrInsertComputed",
];
/// `WeakMap.prototype` methods exposed as first-class values.
const WEAKMAP_PROTO_METHODS: &[&str] = &[
    "set",
    "get",
    "has",
    "delete",
    "getOrInsert",
    "getOrInsertComputed",
];
/// `WeakSet.prototype` methods exposed as first-class values.
const WEAKSET_PROTO_METHODS: &[&str] = &["add", "has", "delete"];
/// `Promise.prototype` methods exposed as first-class values.
const PROMISE_PROTO_METHODS: &[&str] = &["then", "catch", "finally"];
/// `Function.prototype` methods exposed as first-class values.
const FUNCTION_PROTO_METHODS: &[&str] = &["call", "apply", "bind", "toString"];
/// `DataView.prototype` accessor methods — dispatched in `call_method`, exposed here as
/// readable bound natives (for `typeof dv.getUint8` and detached `dv.getUint8.call(dv, …)`).
/// `ArrayBuffer.prototype` methods exposed as first-class own functions (dispatched
/// through [`N_AB_PROTO_FN`] → `call_method`). `slice`/`resize`/`transfer`/
/// `transferToFixedLength` each require an `[[ArrayBufferData]]` `this`.
const AB_PROTO_METHODS: &[&str] = &[
    "slice",
    "resize",
    "transfer",
    "transferToFixedLength",
    "transferToImmutable",
    "sliceToImmutable",
];

const DATA_VIEW_METHODS: &[&str] = &[
    "getInt8",
    "getUint8",
    "getInt16",
    "getUint16",
    "getInt32",
    "getUint32",
    "getFloat16",
    "getFloat32",
    "getFloat64",
    "getBigInt64",
    "getBigUint64",
    "setInt8",
    "setUint8",
    "setInt16",
    "setUint16",
    "setInt32",
    "setUint32",
    "setFloat16",
    "setFloat32",
    "setFloat64",
    "setBigInt64",
    "setBigUint64",
];
/// The spec `length` (declared arity) of a first-class built-in *method* — an
/// `Array`/`String`/`Map`/`Set`/… prototype method or a readable static method,
/// exposed as a bound native. Per ECMA-262 each built-in function's `length` is
/// the count of required leading parameters. Names not listed default to 1
/// (the overwhelmingly common arity), which keeps unknown/auxiliary methods from
/// reporting a misleading 0.
#[must_use]
fn builtin_method_arity(name: &str) -> u32 {
    match name {
        // Zero-argument methods (predicates, coercion, iterators, accessors).
        "pop" | "shift" | "reverse" | "keys" | "values" | "entries" | "toString"
        | "toLocaleString" | "valueOf" | "flat" | "clear" | "trim" | "trimStart" | "trimEnd"
        | "toUpperCase" | "toLowerCase" | "toLocaleUpperCase" | "toLocaleLowerCase"
        | "toReversed" | "isWellFormed" | "of" | "toWellFormed" | "getInt8" | "getUint8"
        | "toArray" | "normalize" | "toTemporalInstant"
        // Annex B.2.3 zero-argument HTML wrapper methods.
        | "big" | "blink" | "bold" | "fixed" | "italics" | "small"
        | "strike" | "sub" | "sup"
        // `Date.prototype` getters / serializers (length 0).
        | "getTime" | "getFullYear" | "getUTCFullYear" | "getMonth" | "getUTCMonth"
        | "getDate" | "getUTCDate" | "getDay" | "getUTCDay" | "getHours" | "getUTCHours"
        | "getMinutes" | "getUTCMinutes" | "getSeconds" | "getUTCSeconds"
        | "getMilliseconds" | "getUTCMilliseconds" | "getTimezoneOffset"
        | "toISOString" | "toDateString" | "toTimeString" | "toUTCString"
        | "toLocaleDateString" | "toLocaleTimeString"
        // Annex B.2.4 `Date.prototype.getYear` (length 0).
        | "getYear"
        // `ArrayBuffer.prototype.transfer`/`transferToFixedLength`/
        // `transferToImmutable` — `length` 0 (the optional `newLength` is not counted).
        | "transfer" | "transferToFixedLength" | "transferToImmutable"
        // `Date.now()` takes no arguments.
        | "now" => 0,
        // Two-argument methods.
        "slice" | "sliceToImmutable" | "substring" | "substr" | "splice" | "copyWithin" | "split" | "replace"
        | "replaceAll" | "with" | "setInt8" | "setUint8" | "asIntN"
        | "asUintN" | "setMonth" | "setUTCMonth" | "setSeconds" | "setUTCSeconds" | "subarray"
        // `Map.prototype.getOrInsert(key, value)` / `getOrInsertComputed(key, fn)`.
        | "getOrInsert" | "getOrInsertComputed"
        // `Map/WeakMap.prototype.set(key, value)`; `Map.groupBy(items, cb)` /
        // `Object.groupBy`.
        | "set" | "groupBy" | "assign" | "toSpliced"
        // `Promise.prototype.then(onFulfilled, onRejected)`.
        | "then"
        // `Function.prototype.apply(thisArg, argArray)`.
        | "apply"
        // `Proxy.revocable(target, handler)`.
        | "revocable" => 2,
        // Three-argument `Date` setters.
        "setFullYear" | "setUTCFullYear" | "setMinutes" | "setUTCMinutes" => 3,
        // Four-argument `Date` setters.
        "setHours" | "setUTCHours" => 4,
        // `Date.UTC(year, month, …, ms)` — 7 declared parameters.
        "UTC" => 7,
        // Three-argument typed-array helpers (none currently bound) → fall through.
        // Everything else (map/filter/forEach/reduce/indexOf/slice/at/push/…)
        // declares a single required parameter.
        _ => 1,
    }
}

/// The spec `length` of a built-in *constructor*/global function, keyed by its
/// native dispatch id. Only ids that escape as first-class values need an exact
/// answer for `verifyProperty`; unmapped ids default to 1.
#[must_use]
fn builtin_native_arity(id: u16) -> u32 {
    match id {
        // Length 0.
        N_MATH_RANDOM
        | N_TYPED_ARRAY_ABSTRACT
        | N_TYPED_ARRAY_OF
        | N_TYPED_ARRAY_SPECIES
        | N_MAP_SIZE
        | N_SET_SIZE
        | N_TYPED_ARRAY_TO_STRING_TAG
        // Intl services with optional `locales`/`options` all have `length === 0`.
        | N_INTL_DURATION_FORMAT
        | N_INTL_LIST_FORMAT
        | N_INTL_REL_TIME
        | N_INTL_PLURAL_RULES
        | N_INTL_NUMBER_FORMAT
        | N_INTL_DATETIME_FORMAT
        | N_INTL_COLLATOR
        | N_INTL_SEGMENTER
        // `DisposableStack`/`AsyncDisposableStack`/`ShadowRealm` take no parameters.
        | N_DISPOSABLE_STACK
        | N_ASYNC_DISPOSABLE_STACK
        | N_SHADOW_REALM
        // `Uint8Array.prototype.toBase64([options])` / `.toHex()` — `length` 0
        // (the optional `options` is not counted; `toHex` takes none).
        | N_UINT8_TO_BASE64
        | N_UINT8_TO_HEX
        // `WeakRef.prototype.deref()` / `Symbol.prototype.toString()` /
        // `…valueOf()` / `get Symbol.prototype.description` all take no args.
        | N_WEAKREF_DEREF
        | N_SYMBOL_PROTO_TOSTRING
        | N_SYMBOL_PROTO_VALUEOF
        | N_SYMBOL_PROTO_DESC_GET
        // `get Error.prototype.stack` takes no arguments.
        | N_ERROR_PROTO_STACK_GET
        // Constructors whose sole argument is optional have `length` 0:
        // `Map`/`Set`/`WeakMap`/`WeakSet`([iterable]), `Symbol`([description]),
        // `Iterator`() (abstract).
        | N_MAP
        | N_SET
        | N_WEAKMAP
        | N_WEAKSET
        | N_SYMBOL
        | N_ITERATOR
        // `Array.of(...items)` is variadic → `length` 0.
        | N_ARRAY_OF
        // `Atomics.pause([N])` — the optional `N` is not counted.
        | N_ATOMICS_PAUSE => 0,
        // `Atomics.compareExchange(ta, index, expected, replacement)` = 4;
        // `load(ta, index)` = 2; the read-modify-write ops and `store` = 3;
        // `isLockFree(size)` = 1.
        N_ATOMICS_COMPARE_EXCHANGE | N_ATOMICS_WAIT | N_ATOMICS_WAIT_ASYNC => 4,
        N_ATOMICS_LOAD => 2,
        N_ATOMICS_ADD | N_ATOMICS_SUB | N_ATOMICS_AND | N_ATOMICS_OR | N_ATOMICS_XOR
        | N_ATOMICS_EXCHANGE | N_ATOMICS_STORE | N_ATOMICS_NOTIFY => 3,
        N_GENERATOR_FUNCTION_CTOR | N_ASYNC_GENERATOR_FUNCTION_CTOR => 1,
        N_ASYNC_ITERATOR_DISPOSE => 0,
        N_RETURN_UNDEFINED => 1,
        // Length 2.
        // `FinalizationRegistry.prototype.register(target, heldValue [, token])`.
        N_FINREG_REGISTER
        // `Intl.DisplayNames(locales, options)` — both required (options is not
        // optional), so `length === 2`.
        | N_INTL_DISPLAY_NAMES
        | N_PROXY
        | N_OBJECT_SET_PROTO
        | N_OBJECT_IS
        | N_OBJECT_HAS_OWN
        | N_OBJECT_DEFINE_PROPS
        | N_REFLECT_GET
        | N_REFLECT_HAS
        | N_REFLECT_GET_OWN_DESC
        | N_REFLECT_SET_PROTO
        | N_REFLECT_DELETE
        | N_REFLECT_CONSTRUCT
        | N_PARSE_INT
        | N_OBJECT_GET_OWN_DESC
        | N_MATH_MAX
        | N_MATH_MIN
        | N_MATH_POW
        | N_MATH_ATAN2
        | N_MATH_HYPOT
        | N_MATH_IMUL
        // `JSON.parse(text, reviver)`.
        | N_JSON_PARSE
        // `Object.groupBy(items, callbackfn)`; `Object.assign(target, ...sources)`.
        | N_OBJECT_GROUP_BY
        | N_OBJECT_ASSIGN
        // `Object.create(O, Properties)` — two declared parameters.
        | N_OBJECT_CREATE
        // `RegExp(pattern, flags)` — two declared parameters.
        | N_REGEXP => 2,
        // Length 3.
        N_OBJECT_DEFINE_PROP | N_REFLECT_SET | N_REFLECT_DEFINE_PROP | N_REFLECT_APPLY
        // `SuppressedError(error, suppressed, message)`.
        | N_SUPPRESSED_ERROR
        // `JSON.stringify(value, replacer, space)`.
        | N_JSON_STRINGIFY => 3,
        // A concrete TypedArray constructor (`Int8Array`, …) has `length` 3
        // (`new T(buffer, byteOffset, length)`).
        id if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16)
            .contains(&id) =>
        {
            3
        }
        // `Date` constructor: `Date(year, month, …, ms)` — 7 declared parameters.
        N_DATE => 7,
        // Default (String, Number, Boolean, Array, Object, Error family, RegExp,
        // Promise, Map, Set, Symbol, parseFloat, the single-arg Object/Reflect
        // statics, …) — one declared parameter.
        _ => 1,
    }
}

/// Whether a built-in identified by its native dispatch `id` has a `[[Construct]]`
/// (i.e. `new id(...)` and using it as a `Reflect.construct` newTarget is allowed).
/// This is the set of ids the `construct` method accepts; everything else — global
/// functions (`parseInt`, `eval`, `Symbol`, `BigInt`), `Math`/`JSON` methods, and
/// the `Object`/`Reflect`/`Number`/… statics — is callable but not a constructor.
/// (`Object` and `Array` are matched by identity, not id, so are handled separately.)
#[must_use]
fn is_native_constructor(id: u16) -> bool {
    if crate::nbexec::temporal::is_temporal_ctor_id(id) {
        return true;
    }
    if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id) {
        return true;
    }
    if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
        return true;
    }
    matches!(
        id,
        // The `Object`/`Array` namespace-object constructors carry these native
        // ids; recognizing them here (not just via global-binding identity) makes
        // a *cross-realm* `Object`/`Array` — a distinct heap cell with the same id
        // — pass `IsConstructor` (`Reflect.construct(other.Array, …)`).
        N_BASE_OBJECT
            | N_BASE_ARRAY
            | N_STRING
            | N_NUMBER
            | N_BOOLEAN
            // `Symbol`/`BigInt` have a `[[Construct]]` (so `IsConstructor` is true)
            // even though invoking it always throws a TypeError.
            | N_SYMBOL
            | N_BIGINT
            // The abstract `Iterator` constructor has a `[[Construct]]` (throws when
            // `newTarget` is `Iterator` itself, like `%TypedArray%`), so
            // `IsConstructor(Iterator)` is true and `class X extends Iterator {}` is
            // valid.
            | N_ITERATOR
            | N_MAP
            | N_SET
            | N_WEAKMAP
            | N_WEAKSET
            | N_WEAKREF
            | N_FINALIZATION_REGISTRY
            | N_PROMISE
            | N_PROXY
            | N_DATE
            | N_REGEXP
            | N_FUNCTION
            | N_GENERATOR_FUNCTION_CTOR
            | N_ASYNC_GENERATOR_FUNCTION_CTOR
            | N_ARRAY_BUFFER
            | N_SHARED_ARRAY_BUFFER
            | N_DATA_VIEW
            | N_WASM_MODULE
            | N_WASM_INSTANCE
            | N_WASM_GLOBAL
            | N_WASM_MEMORY
            | N_WASM_TABLE
            | N_INTL_NUMBER_FORMAT
            | N_INTL_DATETIME_FORMAT
            | N_INTL_COLLATOR
            | N_INTL_PLURAL_RULES
            | N_INTL_LIST_FORMAT
            | N_INTL_REL_TIME
            | N_INTL_DISPLAY_NAMES
            | N_INTL_SEGMENTER
            | N_INTL_LOCALE
            | N_INTL_DURATION_FORMAT
            | N_DISPOSABLE_STACK
            | N_ASYNC_DISPOSABLE_STACK
            | N_SHADOW_REALM
            | N_SUPPRESSED_ERROR
    )
}

/// Bound native: the `revoke` function from `Proxy.revocable` (carries the proxy).
const N_PROXY_REVOKE: u16 = 122;
const N_SYMBOL: u16 = 38;
const N_BIGINT: u16 = 39;
const N_PROXY: u16 = 106;
const N_PARSE_FLOAT: u16 = 31;
const N_IS_NAN: u16 = 32;
const N_IS_FINITE: u16 = 33;
// Error constructors (id − N_ERROR_BASE indexes ERROR_NAMES).
const N_ERROR_BASE: u16 = 40;
/// Abbreviated weekday names (index 0 = Sunday), for `Date` string methods.
const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
/// Abbreviated month names (index 0 = January).
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const ERROR_NAMES: [&str; 11] = [
    "Error",
    "TypeError",
    "RangeError",
    "SyntaxError",
    "ReferenceError",
    "AggregateError",
    // The `WebAssembly.*` error subclasses (exposed under the WebAssembly
    // namespace, not as globals — see the registration below).
    "CompileError",
    "LinkError",
    "RuntimeError",
    // Standard global error subclasses (registered as globals below).
    "URIError",
    "EvalError",
];
/// Count of `Error` subclasses exposed as JS globals (`Error`…`AggregateError`);
/// the `WebAssembly.*` ones are namespaced, and `URIError`/`EvalError` are
/// registered separately as globals.
const N_GLOBAL_ERROR_COUNT: usize = 6;
const N_TYPE_ERROR: u16 = N_ERROR_BASE + 1;
const N_RANGE_ERROR: u16 = N_ERROR_BASE + 2;
const N_SYNTAX_ERROR: u16 = N_ERROR_BASE + 3;
const N_REFERENCE_ERROR: u16 = N_ERROR_BASE + 4;
const N_WASM_COMPILE_ERROR: u16 = N_ERROR_BASE + 6;
const N_WASM_LINK_ERROR: u16 = N_ERROR_BASE + 7;
const N_WASM_RUNTIME_ERROR: u16 = N_ERROR_BASE + 8;
const N_URI_ERROR: u16 = N_ERROR_BASE + 9;
const N_EVAL_ERROR: u16 = N_ERROR_BASE + 10;
/// A reserved, non-identifier key under which a `new fn()` instance records its
/// constructor function (a hidden, GC-traced slot) so `instanceof` can match it.
const CTOR_KEY: &str = "\u{0}ctor";
/// Hidden own-property brand stamped onto every genuine `Error` instance (the
/// engine's stand-in for the spec's `[[ErrorData]]` internal slot). Set at every
/// error-construction site (`make_error`, `super()` into an Error base, the
/// `AggregateError`/`SuppressedError` paths) and checked *only* by `Error.isError`.
/// Like `CTOR_KEY` it uses a `\u{0}` prefix so it is non-enumerable / invisible to
/// `Object.keys`, `getOwnPropertyNames`, `for-in`, and `JSON.stringify`.
const ERROR_DATA: &str = "\u{0}errordata";
/// Hidden slot on an object-literal concise method recording its `[[HomeObject]]`
/// (the object it was defined on), for `super` resolution.
const HOME_OBJECT: &str = "\u{0}home";
/// Hidden slots on an arrow function capturing its *lexical* environment at
/// definition: the enclosing `this`, `new.target`, object-literal home object,
/// and class-home (id + static flag). Restored on every call so the arrow's
/// `this`/`super`/`new.target` follow definition site, not the call site.
const ARROW_THIS: &str = "\u{0}athis";
const ARROW_NEW_TARGET: &str = "\u{0}antgt";
const ARROW_HOME_OBJ: &str = "\u{0}ahome";
const ARROW_HOME_CLASS: &str = "\u{0}ahcls";
const ARROW_HOME_STATIC: &str = "\u{0}ahsta";
/// Reserved hidden keys for an eager generator's result object: the buffer of
/// yielded values and the current `next()` cursor.
/// Sentinel description for a `Symbol()` created with no argument (so its
/// `.description` is `undefined`, distinct from `Symbol("")`).
// The call-depth, allocation-length, string-length, native-recursion,
// BigInt-size, and JSON-depth caps now live in [`crate::limits::Limits`] and are
// read live from `self.realm.limits`, so an embedder can tune them per realm.
const SYMBOL_NO_DESC: &str = "\u{0}nodesc";
/// Hidden key holding a `WeakRef`'s target (returned by `deref`).
const WEAKREF_TARGET: &str = "\u{0}wrtarget";
/// Hidden marker tagging a `FinalizationRegistry` instance.
const FINREG_TAG: &str = "\u{0}finreg";
const GEN_BUF: &str = "\u{0}gbuf";
const GEN_IDX: &str = "\u{0}gidx";
/// A generator's `return` value, surfaced once after its yields are exhausted.
const GEN_RET: &str = "\u{0}gret";
/// Hidden slots on a **live** Set/Map iterator (a spec-faithful iterator that
/// re-reads the collection on each `next()`, so mutation mid-iteration is
/// visible). `GEN_COLL` = the collection handle (its presence marks the iterator
/// live); `GEN_KIND` = 0 keys / 1 values / 2 entries; `GEN_LASTKEY` = the last
/// yielded key (absent → not yet started); `GEN_IDX` = its position at yield time
/// (the resume point if that key was since deleted); `GEN_DONE` = detached.
const GEN_COLL: &str = "\u{0}gcoll";
const GEN_KIND: &str = "\u{0}gkind";
const GEN_LASTKEY: &str = "\u{0}glast";
const GEN_DONE: &str = "\u{0}gdone";
/// Hidden slot on a **live** typed-array iterator: the typed-array handle. Its
/// `next()` re-reads the live length (`typed_len`) and each element by index, so
/// growing/shrinking the backing resizable buffer — or writing an element —
/// mid-iteration is observed (`GEN_IDX` holds the cursor, `GEN_KIND` the 0/1/2
/// keys/values/entries selector).
const GEN_TA: &str = "\u{0}gta";
/// Hidden slot on a **live** plain-array iterator (from an explicit
/// `arr.values()`/`.keys()`/`.entries()`): the array handle. Each `next()`
/// re-reads the array's current `length` and `Get`s the element at the cursor,
/// so elements pushed/assigned after the iterator was created are observed (per
/// spec `CreateArrayIterator`). `GEN_IDX` is the cursor, `GEN_KIND` the 0/1/2
/// keys/values/entries selector.
const GEN_ARR: &str = "\u{0}garr";
/// Hidden slot on a *lazy* generator object: the index of its suspended
/// [`generator::GenFrame`] in `Interp::gen_frames`.
const GEN_FRAME: &str = "\u{0}gframe";
/// Hidden slot on an *async* coroutine controller object: the raw handle of the
/// promise the async function call returned (settled when the body completes).
const ASYNC_PROMISE: &str = "\u{0}aprom";
/// Reserved hidden slots for a lazy ES2025 iterator-helper object (the object
/// returned by `Iterator.prototype.{map,filter,take,drop,flatMap}` and
/// `Iterator.from`). The helper pulls from its underlying iterator one step at a
/// time, so it interleaves correctly with direct `.next()` calls and never
/// over-consumes (it works on infinite iterators).
/// The helper kind discriminant (see `HelperKind`).
const HELPER_KIND: &str = "\u{0}hkind";
/// The underlying iterator *object* (what `next`/`return` are invoked on).
const HELPER_SOURCE: &str = "\u{0}hsrc";
/// The cached `next` method of the underlying iterator (looked up once).
const HELPER_NEXT: &str = "\u{0}hnext";
/// The mapper/filter/flatMap callback (absent for take/drop/from).
const HELPER_FN: &str = "\u{0}hfn";
/// A numeric helper parameter: the remaining count for take/drop.
const HELPER_LIMIT: &str = "\u{0}hlimit";
/// The element counter passed to the callback (`fn(value, counter)`).
const HELPER_COUNTER: &str = "\u{0}hcounter";
/// Set once the helper is exhausted/closed; further `next` returns `{done:true}`.
const HELPER_DONE: &str = "\u{0}hdone";
/// Set while an iterator helper's `next` is executing; a reentrant `next` (a
/// callback that resumes the same helper) sees it and throws a TypeError
/// (GeneratorValidate: state is executing).
const HELPER_RUNNING: &str = "\u{0}hrun";
/// For flatMap: the current inner iterator being drained (absent when none).
const HELPER_INNER: &str = "\u{0}hinner";
const HELPER_INNER_NEXT: &str = "\u{0}hinnext";
/// Parallel array of the per-item `@@iterator` methods captured once at
/// `Iterator.concat` call time (`undefined` for a built-in iterable drained
/// directly), so iteration re-invokes the stored method rather than re-reading
/// `@@iterator` (a getter must fire exactly once).
const HELPER_METHODS: &str = "\u{0}hmethods";
/// Hidden slots on the `Iterator` constructor caching the three helper-result
/// prototypes (`%IteratorHelperPrototype%`, `%WrapForValidIteratorPrototype%`,
/// `%ConcatIteratorPrototype%`).
const ITER_HELPER_PROTO_SLOT: &str = "\u{0}ihproto";
const ITER_WRAP_PROTO_SLOT: &str = "\u{0}iwproto";
const ITER_CONCAT_PROTO_SLOT: &str = "\u{0}icproto";
const ITER_ZIP_PROTO_SLOT: &str = "\u{0}izproto";
/// Reserved hidden slots for a lazy `Iterator.zip`/`zipKeyed` result: the array
/// of open underlying iterators, their cached `next` methods, the live/done
/// flags, the mode (0=shortest,1=longest,2=strict), the padding array, the
/// (optional) result keys (zipKeyed), and the done flag.
const ZIP_ITERS: &str = "\u{0}zits";
const ZIP_NEXTS: &str = "\u{0}znexts";
const ZIP_MODE: &str = "\u{0}zmode";
const ZIP_PADDING: &str = "\u{0}zpad";
const ZIP_KEYS: &str = "\u{0}zkeys";
const ZIP_DONE: &str = "\u{0}zdone";
/// Per-iterator "already finished" flags (longest mode), as a parallel array.
const ZIP_FINISHED: &str = "\u{0}zfin";
/// Set once the zip result has yielded a value (generator moved to
/// "suspended-yield"); distinguishes a `return()` that closes as "executing"
/// (reentrant calls throw) from a "suspended-start" one (reentrant calls
/// return done).
const ZIP_STARTED: &str = "\u{0}zstart";
/// Reserved hidden keys for a bound function (`Function.prototype.bind`).
/// Hidden slot holding a primitive-wrapper object's boxed value, and its
/// constructor id (for `instanceof`).
const PRIM_WRAP: &str = "\u{0}prim";
const PRIM_WRAP_TYPE: &str = "\u{0}primtype";
/// Marks an ordinary object as an `arguments` exotic object, so
/// `Object.prototype.toString` reports `[object Arguments]`. (A mapped/sloppy
/// arguments object additionally carries `ARGS_CALLEE` parameter linkage.)
const ARGS_MARKER: &str = "\u{0}args";
/// `ArrayBuffer` byte store (an array of 0–255 numbers) and `DataView` linkage.
const ARRAY_BUFFER_BYTES: &str = "\u{0}abytes";
/// Marks an `ArrayBuffer` as detached (after `transfer()`): its `byteLength` reads 0 and its
/// views have been emptied.
const ARRAY_BUFFER_DETACHED: &str = "\u{0}abdetached";
/// An `ArrayBuffer`'s `maxByteLength` — present iff it was constructed resizable (via
/// `new ArrayBuffer(n, { maxByteLength })`), bounding `resize`.
const ARRAY_BUFFER_MAXLEN: &str = "\u{0}abmaxlen";
/// Marks an `ArrayBuffer` as immutable (produced by `transferToImmutable` /
/// `sliceToImmutable`): its bytes may not be modified, and it cannot be resized
/// or transferred. The `immutable` getter reports `true` (unless detached).
const ARRAY_BUFFER_IMMUTABLE: &str = "\u{0}abimmutable";
/// Marks a buffer as a `SharedArrayBuffer` (vs a plain `ArrayBuffer`): drives its
/// `[Symbol.toStringTag]`, the `growable`/`maxByteLength` accessors, and `grow`.
const SHARED_ARRAY_BUFFER_BRAND: &str = "\u{0}sab";
const DATA_VIEW_BUF: &str = "\u{0}dvbuf";
const DATA_VIEW_OFF: &str = "\u{0}dvoff";
/// An explicit `DataView` byteLength (the 3rd constructor arg); absent → the rest
/// of the buffer from the offset.
const DATA_VIEW_LEN: &str = "\u{0}dvlen";
/// Brands the `ArrayBuffer.prototype` / `DataView.prototype` / `%TypedArray%`-kind
/// prototype objects. The `byteLength`/`buffer`/`byteOffset`/`detached`/… accessors
/// are spec accessor *properties* defined on these prototypes: invoking the getter
/// with a receiver that lacks the matching internal slot throws a `TypeError`. A
/// branded prototype itself has no slot, so reading the accessor on it (or on any
/// non-branded receiver inheriting it) must throw rather than return `undefined`.
const ARRAY_BUFFER_PROTO_BRAND: &str = "\u{0}abproto";
const DATA_VIEW_PROTO_BRAND: &str = "\u{0}dvproto";
const TYPED_ARRAY_PROTO_BRAND: &str = "\u{0}taproto";
/// Marks the `RegExp.prototype` object so its accessor getters (`source`/`flags`/
/// flag getters) recognise it as the sentinel receiver — `source` → `"(?:)"`,
/// `flags` → `""`, and each flag getter → `undefined` — rather than throwing the
/// non-RegExp `TypeError`.
const REGEXP_PROTO_BRAND: &str = "\u{0}reproto";
const BOUND_TARGET: &str = "\u{0}bnd_t";
const BOUND_THIS: &str = "\u{0}bnd_this";
const BOUND_ARGS: &str = "\u{0}bnd_args";
/// Marks a function built by the dynamic `Function`/`GeneratorFunction`/… constructor.
/// Such a function's `.caller`/`.arguments` keep the conservative poisoned-accessor
/// throw (the engine cannot yet distinguish a dynamically-built *generator* from an
/// ordinary one, and a restricted dynamic generator must throw).
const DYN_FN_MARKER: &str = "\u{0}dynfn";
/// A safety cap on eagerly-collected `yield`s (an infinite generator would
/// otherwise hang); exceeding it throws instead.
const GEN_CAP: usize = 1_000_000;
// Bound natives (carry a target promise handle):
const N_RESOLVE: u16 = 100;
const N_REJECT: u16 = 101;
const N_MATH_HYPOT: u16 = 102;
const N_MATH_CBRT: u16 = 103;
const N_MATH_LOG2: u16 = 104;
const N_MATH_LOG10: u16 = 105;
const N_MATH_EXP: u16 = 133;
const N_MATH_LOG: u16 = 134;
const N_MATH_RANDOM: u16 = 139;
// Trig / hyperbolic / extra Math functions (140..=157).
const N_MATH_SIN: u16 = 140;
const N_MATH_COS: u16 = 141;
const N_MATH_TAN: u16 = 142;
const N_MATH_ASIN: u16 = 143;
const N_MATH_ACOS: u16 = 144;
const N_MATH_ATAN: u16 = 145;
const N_MATH_ATAN2: u16 = 146;
const N_MATH_SINH: u16 = 147;
const N_MATH_COSH: u16 = 148;
const N_MATH_TANH: u16 = 149;
const N_MATH_ASINH: u16 = 150;
const N_MATH_ACOSH: u16 = 151;
const N_MATH_ATANH: u16 = 152;
const N_MATH_EXPM1: u16 = 153;
const N_MATH_LOG1P: u16 = 154;
const N_MATH_FROUND: u16 = 155;
const N_MATH_CLZ32: u16 = 156;
const N_MATH_IMUL: u16 = 157;
const N_MATH_F16ROUND: u16 = 248;
/// `Date.prototype[Symbol.toPrimitive]` (a named native, length 1).
const N_DATE_TO_PRIMITIVE: u16 = 243;
/// `Date.prototype.toJSON` — a generic method (length 1) callable on any object.
const N_DATE_TO_JSON: u16 = 244;
/// A first-class `String.prototype.<method>`: applies RequireObjectCoercible and
/// ToString to the call's `this` (so `String.prototype.slice.call(true)` coerces
/// to `"true"` and a `null`/`undefined` `this` throws), then dispatches. The two
/// identity methods `toString`/`valueOf` instead require a String value.
const N_STRING_PROTO_FN: u16 = 245;

/// A first-class `Number.prototype.<method>`: `thisNumberValue(this)` must yield
/// a Number (the `this` is a Number primitive or a Number wrapper object), else a
/// `TypeError` (so `Number.prototype.valueOf.call({})` rejects per spec).
const N_NUMBER_PROTO_FN: u16 = 249;

/// A first-class `Boolean.prototype.<method>`: `thisBooleanValue(this)` must
/// yield a Boolean (the `this` is a Boolean primitive or a Boolean wrapper
/// object), else a `TypeError`.
const N_BOOLEAN_PROTO_FN: u16 = 250;

/// `Error.prototype.toString` — its receiver must be an Object (else a
/// `TypeError`); reads the receiver's `name`/`message` (each ToString'd) and
/// renders `"name: message"` (or just one part when the other is empty).
const N_ERROR_PROTO_TOSTRING: u16 = 251;

/// A first-class `Set.prototype.<method>`: the receiver must have a `[[SetData]]`
/// internal slot (a non-weak Set), else a `TypeError` (so
/// `Set.prototype.add.call(new Map(), …)` rejects).
const N_SET_PROTO_FN: u16 = 253;

/// A first-class `Map.prototype.<method>`: the receiver must have a `[[MapData]]`
/// internal slot (a non-weak Map), else a `TypeError`.
const N_MAP_PROTO_FN: u16 = 254;

/// A first-class `WeakMap.prototype.<method>`: the receiver must be a WeakMap
/// (`[[WeakMapData]]`), else a `TypeError`.
const N_WEAKMAP_PROTO_FN: u16 = 255;

/// A first-class `WeakSet.prototype.<method>`: the receiver must be a WeakSet
/// (`[[WeakSetData]]`), else a `TypeError`.
const N_WEAKSET_PROTO_FN: u16 = 256;

// --- ES2025 explicit resource management + ShadowRealm (see `resource.rs`) ---
/// The `DisposableStack` constructor.
const N_DISPOSABLE_STACK: u16 = 540;
/// A brand-checked `DisposableStack.prototype.<method>` (bound to the method name).
const N_DISPOSABLE_STACK_PROTO: u16 = 541;
/// The `get DisposableStack.prototype.disposed` accessor.
const N_DISPOSABLE_STACK_DISPOSED: u16 = 542;
/// The `AsyncDisposableStack` constructor.
const N_ASYNC_DISPOSABLE_STACK: u16 = 543;
/// A brand-checked `AsyncDisposableStack.prototype.<method>`.
const N_ASYNC_DISPOSABLE_STACK_PROTO: u16 = 544;
/// The `get AsyncDisposableStack.prototype.disposed` accessor.
const N_ASYNC_DISPOSABLE_STACK_DISPOSED: u16 = 545;
/// The `ShadowRealm` constructor.
const N_SHADOW_REALM: u16 = 546;
/// A `ShadowRealm.prototype.<method>` (`evaluate`/`importValue`).
const N_SHADOW_REALM_PROTO: u16 = 547;
/// A wrapped callable returned across a `ShadowRealm` boundary.
const N_SHADOW_REALM_WRAPPED: u16 = 548;
/// The dispose callback recorded by `DisposableStack.prototype.adopt`
/// (`[value, onDispose]`; calls `onDispose(value)`).
const N_DSTACK_ADOPT_CALL: u16 = 549;
/// The `SuppressedError` constructor (ES2025) — thrown when multiple disposers
/// throw during `DisposableStack`/`AsyncDisposableStack` disposal.
const N_SUPPRESSED_ERROR: u16 = 550;

// --- Promise combinator helper natives (resolve/reject element closures). ---
// Each is a *bound* native whose target is a per-call state object carrying the
// shared accounting (remaining counter, values array, capability promise) plus
// this element's index, all as hidden properties. New ids begin at 640.
/// `Promise.all` Resolve Element: records its value at the captured index and,
/// when the last input fulfills, resolves the capability with the values array.
const N_PROMISE_ALL_ELEMENT: u16 = 640;
/// `Promise.allSettled` Resolve Element: records `{status:"fulfilled", value}`.
const N_PROMISE_ALLSETTLED_FULFILL: u16 = 641;
/// `Promise.allSettled` Reject Element: records `{status:"rejected", reason}`.
const N_PROMISE_ALLSETTLED_REJECT: u16 = 642;
/// `Promise.any` Reject Element: records its rejection reason and, when the last
/// input rejects, rejects the capability with an `AggregateError`.
const N_PROMISE_ANY_ELEMENT: u16 = 643;
/// `Promise.allKeyed` Resolve Element (await-dictionary proposal): records its
/// value at the captured *key* of the result object.
const N_PROMISE_ALLKEYED_ELEMENT: u16 = 672;
/// `Promise.allSettledKeyed` Fulfill / Reject Elements: record
/// `{status, value|reason}` at the captured key.
const N_PROMISE_ALLSETTLEDKEYED_FULFILL: u16 = 673;
const N_PROMISE_ALLSETTLEDKEYED_REJECT: u16 = 674;
/// `Promise.prototype.finally` Then Finally / Catch Finally functions: run the
/// captured `onFinally()`, then thread the original value/reason through
/// `C.resolve(result).then(valueThunk)`.
const N_PROMISE_THEN_FINALLY: u16 = 644;
const N_PROMISE_CATCH_FINALLY: u16 = 645;
/// `finally` value/throw thunk (`() => value` / `() => { throw reason }`).
const N_PROMISE_VALUE_THUNK: u16 = 646;
const N_PROMISE_THROW_THUNK: u16 = 647;
/// Hidden keys for the `finally` closures' bound state object.
const PFIN_ONFINALLY: &str = "\u{0}pf_onf";
const PFIN_CTOR: &str = "\u{0}pf_ctor";
const PFIN_VALUE: &str = "\u{0}pf_val";
/// `GetCapabilitiesExecutor` function passed to a user/subclass Promise
/// constructor by `NewPromiseCapability`: captures the `(resolve, reject)` pair
/// into the capability state object (its bound target).
const N_PROMISE_CAPABILITY_EXECUTOR: u16 = 649;
/// `JSON.rawJSON(text)` / `JSON.isRawJSON(value)` (the JSON source-text proposal).
const N_JSON_RAW: u16 = 650;
const N_JSON_IS_RAW: u16 = 651;
/// `Array.fromAsync(asyncItems, mapFn?, thisArg?)` — returns a promise of an
/// array, awaiting each value of an (a)sync iterable / array-like.
const N_ARRAY_FROM_ASYNC: u16 = 652;
/// `RegExp.escape(S)` — escapes `S` so it matches literally in a pattern.
const N_REGEXP_ESCAPE: u16 = 653;
/// The ES2025 `uint8array-base64` proposal methods. Instance methods on
/// `Uint8Array.prototype` (`this` must be a `Uint8Array`) and statics on the
/// `Uint8Array` constructor; all six are pure byte↔string codecs (see
/// [`base64`]).
const N_UINT8_TO_BASE64: u16 = 654;
const N_UINT8_TO_HEX: u16 = 655;
const N_UINT8_SET_FROM_BASE64: u16 = 656;
const N_UINT8_SET_FROM_HEX: u16 = 657;
const N_UINT8_FROM_BASE64: u16 = 658;
const N_UINT8_FROM_HEX: u16 = 659;
/// `Error.isError(arg)` (ES2025) — `true` iff `arg` carries the [`ERROR_DATA`]
/// brand (i.e. is a genuine Error instance). A static on the `Error` constructor.
const N_ERROR_IS_ERROR: u16 = 660;
/// `Math.sumPrecise(items)` (ES2025) — the correctly-rounded exact sum of a
/// sequence of Numbers. Iterates `items` one value at a time (closing the
/// iterator on a non-Number element), runs the spec's Infinity/NaN/-0 state
/// machine, and accumulates finite values with a Shewchuk-style exact
/// (error-free) partials list rounded once at the end. `.length` is 1.
const N_MATH_SUM_PRECISE: u16 = 661;
/// `WeakRef.prototype.deref()` — brand-checks `this` (a `[[WeakRefTarget]]`
/// slot) and returns the held target (never collected here). `.length` is 0.
const N_WEAKREF_DEREF: u16 = 662;
/// `FinalizationRegistry.prototype.register(target, heldValue [, token])` —
/// brand-checks `this` (a `[[Cells]]` slot), validates CanBeHeldWeakly(target)
/// and `target !== heldValue`, appends a cell, returns undefined. `.length` 2.
const N_FINREG_REGISTER: u16 = 663;
/// `FinalizationRegistry.prototype.unregister(token)` — brand-checks `this`,
/// validates CanBeHeldWeakly(token), removes every cell whose unregister token
/// SameValue-matches, returns whether any were removed. `.length` is 1.
const N_FINREG_UNREGISTER: u16 = 664;
/// `Symbol.prototype.toString()` — `thisSymbolValue(this)` then `SymbolDescriptiveString`.
const N_SYMBOL_PROTO_TOSTRING: u16 = 665;
/// `Symbol.prototype.valueOf()` — returns `thisSymbolValue(this)`.
const N_SYMBOL_PROTO_VALUEOF: u16 = 666;
/// `get Symbol.prototype.description` — returns `thisSymbolValue(this).[[Description]]`.
const N_SYMBOL_PROTO_DESC_GET: u16 = 667;
/// `Symbol.prototype[Symbol.toPrimitive](hint)` — returns `thisSymbolValue(this)`.
const N_SYMBOL_PROTO_TOPRIMITIVE: u16 = 668;
/// `get Error.prototype.stack` (the error-stack-accessor proposal). A non-object
/// `this` throws a TypeError; an object lacking the `[[ErrorData]]` brand (see
/// `ERROR_DATA`) returns `undefined`; a genuine Error instance returns an
/// implementation string. Defined on `Error.prototype` as an accessor property.
const N_ERROR_PROTO_STACK_GET: u16 = 669;
/// `set Error.prototype.stack` (the error-stack-accessor proposal). Implements
/// `SetterThatIgnoresPrototypeProperties(this, %Error.prototype%, "stack", v)`:
/// a non-object `this` or a non-String `v` throws a TypeError; `this ===
/// %Error.prototype%` throws; otherwise an own data property "stack" is created
/// (or `[[Set]]` runs if one already exists).
const N_ERROR_PROTO_STACK_SET: u16 = 670;
/// `String.prototype[Symbol.iterator]()` — a real own function property (name
/// `"[Symbol.iterator]"`, length 0). RequireObjectCoercible + ToString the
/// receiver (so a poisoned `toString` propagates), then return a
/// `%StringIteratorPrototype%` iterator over the string's code points.
const N_STRING_PROTO_ITER: u16 = 962;
/// Hidden array property holding a `FinalizationRegistry`'s `[[Cells]]`: each
/// cell is a 3-element array `[target, heldValue, unregisterToken]` where an
/// absent (~empty~) token is stored as `undefined` (safe: `undefined` can never
/// be a real token — `CanBeHeldWeakly(undefined)` is false, so `unregister`
/// never matches against it).
const FINREG_CELLS: &str = "\u{0}finregcells";
/// Hidden brand + payload on a RawJSON object (the validated source text).
const RAW_JSON_BRAND: &str = "\u{0}rawjson";
/// Hidden-property keys for a capability state object built around a foreign `C`.
const PCAP_RESOLVE: &str = "\u{0}pcap_res";
const PCAP_REJECT: &str = "\u{0}pcap_rej";

// Hidden-property keys for combinator element state objects.
const PCOMB_REMAINING: &str = "\u{0}pc_rem";
const PCOMB_VALUES: &str = "\u{0}pc_vals";
const PCOMB_CAP: &str = "\u{0}pc_cap";
const PCOMB_RESOLVE: &str = "\u{0}pc_res";
const PCOMB_REJECT: &str = "\u{0}pc_rej";
const PCOMB_INDEX: &str = "\u{0}pc_idx";
const PCOMB_CALLED: &str = "\u{0}pc_called";

mod agent;
mod base64;
mod call;
mod class;
mod convert;
mod expr;
mod generator;
mod intl_fmt;
mod iterator;
mod json;
mod method_dispatch;
#[cfg(all(feature = "module", feature = "std"))]
pub mod module;
mod native_dispatch;
mod object;
mod promise;
mod regexp;
mod resource;
mod stmt;
mod temporal;
mod temporal_calendar;
mod temporal_duration;
mod temporal_instant;
mod temporal_plaindate;
mod temporal_plaindatetime;
mod temporal_plainmonthday;
mod temporal_plaintime;
mod temporal_plainyearmonth;
mod temporal_zoneddatetime;
mod typed_array;
mod wasm;

/// A second global environment created by `$262.createRealm` — a genuinely
/// distinct realm sharing the host heap. Holds the realm's global scope (its own
/// `Array`/`Object`/… bindings), its global object, and its intrinsic prototype
/// pointers, so `.evalScript` can swap the interpreter into this environment.
struct CreatedRealm {
    /// The realm's populated global scope (a fresh `Scope::root()` filled by a
    /// dedicated `install_globals`).
    global_scope: Scope,
    /// The realm's `globalThis` object (its `.global`).
    global_this: NanBox,
    /// The realm's intrinsic prototype pointers, swapped in during `evalScript`.
    intrinsics: RealmIntrinsics,
    /// The realm's `Intl` service `.prototype` intrinsics (`%Intl.X.prototype%`),
    /// keyed by ctor id. Swapped into `Realm::intl_protos` while executing in this
    /// realm (they are a distinct set of objects from every other realm's), and
    /// consulted by `GetPrototypeFromConstructor` when a cross-realm `newTarget`
    /// belonging to this realm has a non-object `.prototype`.
    intl_protos: alloc::collections::BTreeMap<u16, Handle>,
}

impl<'a> Interp<'a> {
    /// A fresh interpreter with a single (global) scope and a starter stdlib,
    /// using default [`Limits`](crate::limits::Limits).
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_limits(crate::limits::Limits::default())
    }

    /// A fresh interpreter with the given resource [`Limits`](crate::limits::Limits).
    #[must_use]
    pub fn new_with_limits(limits: crate::limits::Limits) -> Self {
        let mut interp = Self {
            realm: Realm::with_limits(limits),
            current: Scope::root(),
            var_scope: Scope::root(),
            eval_var_scope: None,
            script_eval_globals: false,
            annexb_block_fns: Vec::new(),
            functions: Vec::new(),
            classes: Vec::new(),
            src: "",
            pending_this_init: None,
            class_member_keys: Vec::new(),
            builtin_iter_protos: alloc::collections::BTreeMap::new(),
            class_statics: Vec::new(),
            class_static_fields: Vec::new(),
            class_static_get: Vec::new(),
            class_static_set: Vec::new(),
            class_envs: Vec::new(),
            class_native_super: Vec::new(),
            class_fn_super: Vec::new(),
            class_super_id: Vec::new(),
            class_proto_parent: Vec::new(),
            class_handles: Vec::new(),
            private_method_cache: alloc::collections::BTreeMap::new(),
            class_lexical_parent: Vec::new(),
            class_private_names: Vec::new(),
            temporal_protos: Vec::new(),
            pending_class_name: None,
            call_depth: 0,
            tail_pos: false,
            new_target_in_scope: false,
            eval_depth: 0,
            rng_state: math_random_seed(),
            this_val: NanBox::undefined(),
            new_target: NanBox::undefined(),
            pending_new_target: None,
            reflect_new_target: None,
            array_proto_generic: false,
            array_like_present: None,
            wasm_states: alloc::collections::BTreeMap::new(),
            wasm_modules: alloc::collections::BTreeMap::new(),
            wasm_mem_objs: alloc::collections::BTreeMap::new(),
            wasm_next_id: 0,
            gen_sink: None,
            gen_frames: Vec::new(),
            pending_async_start: None,
            gen_is_async: false,
            symbol_registry: alloc::collections::BTreeMap::new(),
            well_known_symbols: alloc::collections::BTreeMap::new(),
            tagged_template_cache: alloc::collections::BTreeMap::new(),
            regexp_proto: None,
            regexp_ctor: None,
            #[cfg(feature = "intl")]
            intl_intern: alloc::collections::BTreeMap::new(),
            method_name_intern: alloc::collections::BTreeMap::new(),
            pending_super: None,
            pending_super_native: None,
            pending_super_fn: None,
            current_home: None,
            current_lexical_home: None,
            current_home_object: None,
            in_field_initializer: false,
            eval_param_names: None,
            current_home_static: false,
            pending_label: None,
            microtasks: Vec::new(),
            macrotasks: Vec::new(),
            timer_next_id: 1,
            timer_seq: 0,
            virtual_now: 0.0,
            strict: false,
            global_this: NanBox::undefined(),
            output: String::new(),
            global_scope: Scope::root(),
            shadow_realm_scopes: Vec::new(),
            created_realms: Vec::new(),
            fn_realm: alloc::collections::BTreeMap::new(),
            cur_realm: None,
            eval_programs: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "module", feature = "std"))]
            modules: module::ModuleRegistry::new(),
            #[cfg(all(feature = "module", feature = "std"))]
            module_imports: alloc::rc::Rc::new(alloc::collections::BTreeMap::new()),
            #[cfg(all(feature = "module", feature = "std"))]
            import_meta: None,
            #[cfg(all(feature = "module", feature = "std"))]
            script_import_base: None,
            #[cfg(all(feature = "module", feature = "std"))]
            module_namespaces: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "module", feature = "std"))]
            deferred_namespaces: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "module", feature = "std"))]
            active_module_key: None,
            host_fns: Vec::new(),
            agent: AgentState::default(),
            arg_maps: alloc::collections::BTreeMap::new(),
        };
        // The constructor's `current` IS the root scope; capture it as the global
        // scope before `install_globals` populates it, so indirect eval can run
        // against it later.
        interp.global_scope = interp.current.clone();
        interp.var_scope = interp.current.clone();
        interp.install_globals();
        interp
    }

    /// The accumulated `console.log` output.
    #[must_use]
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Renders a result value as a display string (for surfacing a completion
    /// value to a caller / REPL).
    #[must_use]
    pub fn display(&self, value: NanBox) -> String {
        self.realm.to_display_string(value)
    }

    /// Installs the built-in function `name` and `length` own data properties on
    /// `f` with the spec attributes `{ writable: false, enumerable: false,
    /// configurable: true }`. Storing them physically (rather than synthesizing on
    /// read) makes `f.hasOwnProperty("name")`, `delete f.length`, and
    /// `defineProperty` redefinitions behave per spec — exactly what Test262's
    /// `verifyProperty` exercises.
    pub(crate) fn install_fn_name_length(&mut self, f: Handle, name: &str, length: u32) {
        // Spec own-key order for a function is `length` before `name`. Clear any
        // prior readonly flag first so a re-install (e.g. a NamedEvaluation name
        // overwriting the "" placeholder set at creation) actually takes effect —
        // `set_property` is a no-op on a readonly property.
        self.realm.clear_readonly_property(f, "length");
        self.realm
            .set_property(f, "length", NanBox::number(f64::from(length)));
        self.realm.mark_hidden(f, "length");
        self.realm.set_readonly_property(f, "length");
        let name_v = self.new_str(name);
        self.realm.clear_readonly_property(f, "name");
        self.realm.set_property(f, "name", name_v);
        self.realm.mark_hidden(f, "name");
        self.realm.set_readonly_property(f, "name");
    }

    /// Whether the user function with id `func_id` carries a `prototype` own
    /// property (i.e. is `[[Construct]]`-able as an ordinary function or a
    /// generator). Per spec, ordinary function declarations/expressions and
    /// generator functions/methods (sync + async) have a `prototype`; arrow
    /// functions, `async` non-generator functions, concise methods, and
    /// getters/setters do **not**.
    ///
    /// NOTE: `is_arrow` / `is_method` are stamped onto the `FnDef` *after*
    /// `make_method` returns, so calling this at creation time over-includes
    /// arrows/plain-methods (they read `is_arrow == false` there). Those callers
    /// rely on [`Self::demote_fn_prototype`] to strip the property once the flag
    /// is set. At read time (the synthesis gate) the flags are settled, so this
    /// is exact.
    pub(crate) fn fn_has_prototype(&self, func_id: u32) -> bool {
        let d = &self.functions[func_id as usize];
        d.is_generator || (!d.is_arrow && !d.is_async && !d.is_method && d.home_class.is_none())
    }

    /// Installs the own `prototype` data property on the constructable function
    /// `f` with value `proto`. Per spec it is `{ enumerable: false, configurable:
    /// false }`; `writable` is `true` for ordinary/generator functions and
    /// `false` for classes. Own-key order is `length`, `name`, `prototype`, so
    /// this must run *after* [`Self::install_fn_name_length`].
    pub(crate) fn install_fn_prototype(&mut self, f: Handle, proto: Handle, writable: bool) {
        self.realm
            .set_property(f, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(f, "prototype");
        self.realm.set_non_configurable_property(f, "prototype");
        if !writable {
            self.realm.set_readonly_property(f, "prototype");
        }
    }

    /// Strips a `prototype` own property that [`Self::make_method`] materialized
    /// on a function later discovered to be non-constructable (an arrow, a
    /// concise method, or an accessor) — such functions must expose no
    /// `prototype` at all. Also clears the descriptor flags so a subsequent user
    /// assignment (`arrow.prototype = x`) creates an ordinary property.
    pub(crate) fn demote_fn_prototype(&mut self, h: Handle) {
        self.realm.delete_data_slot(h, "prototype");
        self.realm.clear_readonly_property(h, "prototype");
        self.realm.clear_non_configurable_property(h, "prototype");
        self.realm.clear_hidden_property(h, "prototype");
    }

    /// Whether a function's `name` is still the empty-string placeholder (or
    /// absent) — the signal that a NamedEvaluation / property-key inference may
    /// still set it. Functions now materialize `name` "" at creation, so a bare
    /// `has_own("name")` no longer distinguishes "named yet".
    pub(crate) fn fn_name_unset(&self, h: Handle) -> bool {
        match self.realm.get_property(h, "name") {
            None => true,
            Some(v) => v
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|nh| self.realm.string_value(nh))
                .is_none_or(|s| s.is_empty()),
        }
    }

    /// Installs the own `name`/`length` data properties on a freshly created
    /// user method/accessor `f` (a class member or object-literal method). Per
    /// spec these are `{ writable: false, enumerable: false, configurable: true }`
    /// own properties — exactly what Test262's `verifyProperty` checks. `length`
    /// is the count of parameters before the first one with a default or rest;
    /// `name` is the property key (prefixed with `get `/`set ` for accessors).
    fn install_method_meta(&mut self, f: NanBox, name: &str, params: &'a [Param]) {
        let Some(raw) = f.as_handle() else { return };
        let handle = Handle::from_raw(raw);
        // Record the name on the FnDef too (so `fn.name` reads / inference align),
        // but only if the function does not already carry one.
        if let Some((func_id, _)) = self.realm.function_at(handle)
            && self.functions[func_id as usize].name.is_empty()
        {
            self.functions[func_id as usize].name = self.intern_method_name(name);
        }
        // Overwrite the `name` "" placeholder installed at creation with the
        // resolved name: a named function-expression value keeps its own
        // `FnDef::name`; a plain method takes the property key `name`.
        let install_name = match self.realm.function_at(handle) {
            Some((func_id, _)) if !self.functions[func_id as usize].name.is_empty() => {
                self.functions[func_id as usize].name
            }
            _ => name,
        };
        let len = params
            .iter()
            .take_while(|p| p.default.is_none() && !p.rest)
            .count() as u32;
        self.install_fn_name_length(handle, install_name, len);
    }

    /// Interns `s` to a `&'a str` for storing as a `FnDef::name`. Method names are
    /// derived from runtime property keys (computed keys, accessor prefixes), so
    /// they are not always borrowable from the source; leak-once dedup keeps the
    /// `'a` lifetime sound without `unsafe`.
    fn intern_method_name(&mut self, s: &str) -> &'a str {
        if let Some(&v) = self.method_name_intern.get(s) {
            return v;
        }
        let leaked: &'static str = alloc::boxed::Box::leak(String::from(s).into_boxed_str());
        self.method_name_intern.insert(String::from(s), leaked);
        leaked
    }

    /// Creates a native function carrying its own `name`/`length` data properties,
    /// per the spec's named built-ins (`Math.max.name === "max"`,
    /// `Math.max.length === 2`), each with attributes `{ writable: false,
    /// enumerable: false, configurable: true }`.
    fn new_named_native(&mut self, name: &str, id: u16) -> Handle {
        let f = self.realm.new_native(id);
        self.install_fn_name_length(f, name, builtin_native_arity(id));
        f
    }

    /// Builds `<ctor>.prototype` as a real object whose `methods` are first-class
    /// values — each a bound native re-dispatching that method on the call's `this`
    /// — so `Ctor.prototype.method.call(thisArg, …)` works. Methods are
    /// non-enumerable; `proto.constructor` links back to the constructor.
    fn setup_first_class_prototype(&mut self, ctor_name: &str, methods: &[&str]) {
        self.setup_first_class_prototype_id(ctor_name, methods, N_ARRAY_PROTO_FN);
    }

    /// Installs `obj[Symbol.toStringTag] = tag` as a data property with the
    /// built-in attributes `{ writable: false, enumerable: false,
    /// configurable: true }` — used for `Set.prototype`, `Map.prototype`,
    /// `Promise.prototype`, the `Reflect`/`JSON`/`Math` namespaces, etc. (so
    /// `Object.prototype.toString.call(new Set())` is `"[object Set]"` and
    /// `Ctor.prototype[Symbol.toStringTag]` is introspectable).
    fn install_to_string_tag(&mut self, obj: Handle, tag: &str) {
        let sym = self.well_known_symbol("toStringTag");
        let key = self.member_key(sym);
        let val = self.new_str(tag);
        self.realm.set_property(obj, &key, val);
        self.realm.mark_hidden(obj, &key);
        self.realm.set_readonly_property(obj, &key);
    }

    /// Installs `Ctor.prototype[Symbol.toStringTag] = tag` for a named global
    /// constructor (no-op if the constructor / its prototype is absent).
    /// Installs `get <Ctor>.prototype.size` (a brand-checking accessor — the
    /// `size` getter native validates `[[MapData]]`/`[[SetData]]`) and the
    /// `get <Ctor>[Symbol.species]` static accessor (returns the receiver
    /// constructor). Matches ECMA-262: `size` is non-enumerable + configurable
    /// with no setter; `[Symbol.species]` likewise on the constructor.
    fn install_collection_accessors(&mut self, ctor_name: &str, size_native: u16) {
        let Some(ctor) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        let Some(proto) = self
            .realm
            .get_property(ctor, "prototype")
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        // `get <Ctor>.prototype.size`.
        let size_get = self.new_named_native("get size", size_native);
        self.install_fn_name_length(size_get, "get size", 0);
        self.realm.define_accessor(
            proto,
            "size",
            NanBox::handle(size_get.to_raw()),
            NanBox::undefined(),
        );
        self.realm.mark_hidden(proto, "size");
        // `get <Ctor>[Symbol.species]` returns `this` (shared species getter).
        let species_sym = self.well_known_symbol("species");
        let species_key = self.member_key(species_sym);
        let species_get = self.new_named_native("get [Symbol.species]", N_TYPED_ARRAY_SPECIES);
        self.install_fn_name_length(species_get, "get [Symbol.species]", 0);
        self.realm.define_accessor(
            ctor,
            &species_key,
            NanBox::handle(species_get.to_raw()),
            NanBox::undefined(),
        );
        self.realm.mark_hidden(ctor, &species_key);
    }

    /// Installs `get <Ctor>[Symbol.species]` (returning `this`, no setter,
    /// non-enumerable + configurable) on a constructor that has no `size`
    /// accessor (e.g. `Array`).
    fn install_ctor_species(&mut self, ctor_name: &str) {
        let Some(ctor) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        let species_sym = self.well_known_symbol("species");
        let species_key = self.member_key(species_sym);
        let species_get = self.new_named_native("get [Symbol.species]", N_TYPED_ARRAY_SPECIES);
        self.install_fn_name_length(species_get, "get [Symbol.species]", 0);
        self.realm.define_accessor(
            ctor,
            &species_key,
            NanBox::handle(species_get.to_raw()),
            NanBox::undefined(),
        );
        self.realm.mark_hidden(ctor, &species_key);
    }

    fn install_proto_to_string_tag(&mut self, ctor_name: &str, tag: &str) {
        if let Some(proto) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.install_to_string_tag(proto, tag);
        }
    }

    /// The realm's `Object.prototype` handle (the root of the ordinary prototype
    /// chain), resolved from the `Object` global's `prototype` property. Returns
    /// `None` only before `Object.prototype` has been installed.
    fn object_prototype(&self) -> Option<Handle> {
        self.current
            .get("Object")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
    }

    /// As [`Self::setup_first_class_prototype`] but binds each method to the given
    /// native id (so a prototype with a `this`-validating dispatch arm — e.g.
    /// `N_BIGINT_PROTO_FN` — can route `.call`/`.apply` through it).
    fn setup_first_class_prototype_id(
        &mut self,
        ctor_name: &str,
        methods: &[&str],
        native_id: u16,
    ) {
        let Some(ns) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        // The prototype object inherits `Object.prototype` (so inherited methods
        // like `hasOwnProperty`/`isPrototypeOf`/`propertyIsEnumerable` and a
        // `toString`/`valueOf` fallback resolve through the chain — e.g.
        // `Number.prototype.hasOwnProperty("constructor")`).
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &name in methods {
            let name_h = self.realm.new_string(name);
            let f = self.realm.new_bound_native(native_id, name_h);
            self.install_fn_name_length(f, name, builtin_method_arity(name));
            self.realm
                .set_property(proto, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, name);
        }
        self.realm
            .set_property(ns, "prototype", NanBox::handle(proto.to_raw()));
        // A built-in constructor's `prototype` is `{ writable: false,
        // enumerable: false, configurable: false }` (ECMA-262 — every built-in
        // constructor object).
        self.realm.mark_hidden(ns, "prototype");
        self.realm.set_readonly_property(ns, "prototype");
        self.realm.set_non_configurable_property(ns, "prototype");
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ns.to_raw()));
    }

    /// Builds `<ctor>.prototype` as a real object whose methods are *direct-id*
    /// natives (each reads its receiver from `this_val` and brand-checks itself) —
    /// for `WeakRef`/`FinalizationRegistry`, whose methods carry distinct logic
    /// rather than name-based re-dispatch. Each method is non-enumerable; the
    /// `prototype` is `{ writable:false, enumerable:false, configurable:false }`
    /// and `proto.constructor` links back to the constructor (non-enumerable).
    fn setup_direct_prototype(&mut self, ctor_name: &str, methods: &[(&str, u16)]) {
        let Some(ns) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &(name, id) in methods {
            let f = self.new_named_native(name, id);
            self.realm
                .set_property(proto, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, name);
        }
        self.realm
            .set_property(ns, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(ns, "prototype");
        self.realm.set_readonly_property(ns, "prototype");
        self.realm.set_non_configurable_property(ns, "prototype");
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ns.to_raw()));
    }

    /// Installs the shared abstract `%TypedArray%` intrinsic constructor and wires
    /// the concrete typed-array constructors (`Int8Array`, …) into its hierarchy:
    ///
    /// - `Object.getPrototypeOf(Int8Array) === %TypedArray%` (every kind shares it),
    /// - `Object.getPrototypeOf(%TypedArray%) === Function.prototype`,
    /// - `%TypedArray%.prototype` is a real object (proto `Object.prototype`) and
    ///   `Object.getPrototypeOf(Int8Array.prototype) === %TypedArray%.prototype`,
    /// - the generic `from`/`of` statics and the `get [Symbol.species]` accessor
    ///   live on `%TypedArray%` and are inherited by every concrete constructor,
    /// - `%TypedArray%` is abstract: calling/`new`-ing it throws a `TypeError`,
    ///   `%TypedArray%.name === "TypedArray"`, `%TypedArray%.length === 0`.
    ///
    /// `obj_proto` is the realm's `Object.prototype`.
    fn setup_typed_array_intrinsic(&mut self, obj_proto: Handle) {
        // The abstract constructor itself (a native; abstract behavior is enforced
        // in `dispatch_native`).
        let ta = self.new_named_native("TypedArray", N_TYPED_ARRAY_ABSTRACT);
        self.realm
            .set_hidden_property(ta, "length", NanBox::number(0.0));
        self.realm.set_readonly_property(ta, "length");
        self.realm.set_typed_array_intrinsic(ta);
        // Its `[[Prototype]]` is `Function.prototype`.
        if let Some(func_proto) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|f| self.realm.get_property(f, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(ta, func_proto);
        }
        // `%TypedArray%.prototype`: a real object inheriting `Object.prototype`,
        // with a back-link to the constructor. Concrete-kind prototypes inherit it.
        let ta_proto = self.realm.new_object_with_proto(Some(obj_proto));
        self.realm
            .set_hidden_property(ta_proto, "constructor", NanBox::handle(ta.to_raw()));
        // Brand `%TypedArray%.prototype` so its `buffer`/`byteLength`/`byteOffset`/
        // `length` accessors throw a TypeError on a receiver without the
        // `[[TypedArrayName]]` internal slot (e.g. the prototype itself).
        self.realm
            .set_hidden_property(ta_proto, TYPED_ARRAY_PROTO_BRAND, NanBox::boolean(true));
        self.realm
            .set_property(ta, "prototype", NanBox::handle(ta_proto.to_raw()));
        self.realm.mark_hidden(ta, "prototype");
        self.realm.set_readonly_property(ta, "prototype");
        self.realm.set_non_configurable_property(ta, "prototype");
        // Generic statics `from` (length 1) and `of` (length 0).
        let from_fn = self.new_named_native("from", N_TYPED_ARRAY_FROM);
        self.realm
            .set_hidden_property(from_fn, "length", NanBox::number(1.0));
        self.realm.set_readonly_property(from_fn, "length");
        self.realm
            .set_property(ta, "from", NanBox::handle(from_fn.to_raw()));
        self.realm.mark_hidden(ta, "from");
        let of_fn = self.new_named_native("of", N_TYPED_ARRAY_OF);
        self.realm
            .set_hidden_property(of_fn, "length", NanBox::number(0.0));
        self.realm.set_readonly_property(of_fn, "length");
        self.realm
            .set_property(ta, "of", NanBox::handle(of_fn.to_raw()));
        self.realm.mark_hidden(ta, "of");
        // `get %TypedArray%[Symbol.species]` (returns `this`).
        let species_sym = self.well_known_symbol("species");
        let species_key = self.member_key(species_sym);
        let species_get = self.new_named_native("get [Symbol.species]", N_TYPED_ARRAY_SPECIES);
        self.realm.define_accessor(
            ta,
            &species_key,
            NanBox::handle(species_get.to_raw()),
            NanBox::undefined(),
        );
        // Install the `%TypedArray%.prototype` methods as first-class own data
        // properties (each a bound native re-dispatched through `call_method` with
        // the call's typed-array `this`), so `typeof ta.map === "function"`, the
        // method's own `name`/`length`, and `%TypedArray%.prototype.map.call(ta, …)`
        // all behave per spec. Arities (the `length` own property) follow the spec.
        // `%TypedArray%.prototype.toString` *is* the same built-in function object
        // as `Array.prototype.toString` (23.2.3.30 — SameValue). Reuse Array's
        // (installed earlier). The shared array-method dispatch already handles a
        // typed-array receiver, so direct `ta.toString()` keeps working.
        let array_to_string = self
            .current
            .get("Array")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            .and_then(|ap| self.realm.get_property(ap, "toString"));
        for &(name, arity) in TYPED_ARRAY_PROTO_METHODS {
            if name == "toString"
                && let Some(shared) = array_to_string
            {
                self.realm.set_property(ta_proto, name, shared);
                self.realm.mark_hidden(ta_proto, name);
                continue;
            }
            let name_h = self.realm.new_string(name);
            let f = self.realm.new_bound_native(N_TYPED_ARRAY_PROTO_FN, name_h);
            self.install_fn_name_length(f, name, arity);
            self.realm
                .set_property(ta_proto, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(ta_proto, name);
        }
        // `%TypedArray%.prototype[Symbol.iterator]` is the same function object as
        // `%TypedArray%.prototype.values` (per spec — SameValue), exposed under the
        // well-known iterator symbol.
        let values_fn = self
            .realm
            .get_property(ta_proto, "values")
            .unwrap_or(NanBox::undefined());
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        self.realm.set_property(ta_proto, &iter_key, values_fn);
        self.realm.mark_hidden(ta_proto, &iter_key);
        // `get %TypedArray%.prototype[Symbol.toStringTag]` — returns the concrete
        // typed-array name (e.g. "Int8Array") for a view, else `undefined`.
        let tag_sym = self.well_known_symbol("toStringTag");
        let tag_key = self.member_key(tag_sym);
        let tag_get =
            self.new_named_native("get [Symbol.toStringTag]", N_TYPED_ARRAY_TO_STRING_TAG);
        self.realm.define_accessor(
            ta_proto,
            &tag_key,
            NanBox::handle(tag_get.to_raw()),
            NanBox::undefined(),
        );
        // A spec accessor property is non-enumerable (`{ enumerable: false,
        // configurable: true }`).
        self.realm.mark_hidden(ta_proto, &tag_key);
        // The `buffer`/`byteLength`/`byteOffset`/`length` accessors as own
        // get-only properties on `%TypedArray%.prototype` (each a bound native
        // carrying its name; rejects a non-TypedArray receiver). `name`/`length`
        // of a getter are `get <accessor>` / 0.
        for accessor in ["buffer", "byteLength", "byteOffset", "length"] {
            let name_h = self.realm.new_string(accessor);
            let getter = self.realm.new_bound_native(N_TYPED_ARRAY_ACCESSOR, name_h);
            self.install_fn_name_length(getter, &alloc::format!("get {accessor}"), 0);
            self.realm.define_accessor(
                ta_proto,
                accessor,
                NanBox::handle(getter.to_raw()),
                NanBox::undefined(),
            );
            // Spec accessor properties are non-enumerable.
            self.realm.mark_hidden(ta_proto, accessor);
        }
        // Wire every concrete typed-array constructor: its `[[Prototype]]` is
        // `%TypedArray%`, and its `.prototype` is a real object inheriting
        // `%TypedArray%.prototype` with a back-link to the concrete constructor.
        for (i, (name, _)) in TYPED_ARRAY_KINDS.iter().enumerate() {
            let Some(ctor) = self
                .current
                .get(name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
            else {
                continue;
            };
            self.realm.set_native_proto(ctor, ta);
            let kind_proto = self.realm.new_object_with_proto(Some(ta_proto));
            self.realm.set_hidden_property(
                kind_proto,
                "constructor",
                NanBox::handle(ctor.to_raw()),
            );
            self.realm
                .set_property(ctor, "prototype", NanBox::handle(kind_proto.to_raw()));
            // A TypedArray constructor's `prototype` is `{ writable: false,
            // enumerable: false, configurable: false }`.
            self.realm.mark_hidden(ctor, "prototype");
            self.realm.set_readonly_property(ctor, "prototype");
            self.realm.set_non_configurable_property(ctor, "prototype");
            // (`length` of 3 comes from `builtin_native_arity` via the
            // `new_named_native` constructor creation above.)
            // `<TypedArray>.BYTES_PER_ELEMENT` and
            // `<TypedArray>.prototype.BYTES_PER_ELEMENT` are real own data
            // properties `{ writable: false, enumerable: false,
            // configurable: false }` whose value is the element size.
            let bpe = f64::from(TYPED_ARRAY_KINDS[i].1);
            for target in [ctor, kind_proto] {
                self.realm
                    .set_property(target, "BYTES_PER_ELEMENT", NanBox::number(bpe));
                self.realm.mark_hidden(target, "BYTES_PER_ELEMENT");
                self.realm
                    .set_readonly_property(target, "BYTES_PER_ELEMENT");
                self.realm
                    .set_non_configurable_property(target, "BYTES_PER_ELEMENT");
            }
        }
        // `ArrayBuffer.prototype` / `DataView.prototype`: real objects (inheriting
        // `Object.prototype`) with a `constructor` back-link, so feature probes like
        // the Test262 harness's `if (ArrayBuffer.prototype.resize)` read `undefined`
        // rather than null-dereferencing on a missing `.prototype`. The actual
        // ArrayBuffer/DataView methods continue to dispatch via `call_method`.
        for name in ["ArrayBuffer", "DataView"] {
            let Some(ctor) = self
                .current
                .get(name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
            else {
                continue;
            };
            let proto = self.realm.new_object_with_proto(Some(obj_proto));
            self.realm
                .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
            // Brand the prototype so its slot-requiring accessors throw a TypeError
            // when read with a receiver (e.g. the prototype itself) that has no
            // internal slot.
            let brand = if name == "ArrayBuffer" {
                ARRAY_BUFFER_PROTO_BRAND
            } else {
                DATA_VIEW_PROTO_BRAND
            };
            self.realm
                .set_hidden_property(proto, brand, NanBox::boolean(true));
            // Install the spec accessor properties (`get`-only) as real
            // getter/setter descriptors on the prototype, so
            // `Object.getOwnPropertyDescriptor(DataView.prototype, "buffer").get`
            // is the getter function and `getter.call(badThis)` throws a
            // TypeError (RequireInternalSlot). Each getter is a bound native
            // carrying its accessor name; a real instance read still takes the
            // fast special-cased path in `read_member` (its slot is present).
            let (accessor_id, accessors): (u16, &[&str]) = if name == "ArrayBuffer" {
                (
                    N_AB_ACCESSOR,
                    &[
                        "byteLength",
                        "maxByteLength",
                        "resizable",
                        "detached",
                        "immutable",
                    ],
                )
            } else {
                (
                    N_DATA_VIEW_ACCESSOR,
                    &["buffer", "byteLength", "byteOffset"],
                )
            };
            for accessor in accessors {
                let name_h = self.realm.new_string(accessor);
                let getter = self.realm.new_bound_native(accessor_id, name_h);
                self.install_fn_name_length(getter, &alloc::format!("get {accessor}"), 0);
                self.realm.define_accessor(
                    proto,
                    accessor,
                    NanBox::handle(getter.to_raw()),
                    NanBox::undefined(),
                );
                // Spec accessor properties are non-enumerable.
                self.realm.mark_hidden(proto, accessor);
            }
            // `DataView.prototype` get*/set* methods as first-class own data
            // properties (each a bound native re-dispatched through `call_method`
            // with a `[[DataView]]`-validated `this`), so `typeof dv.getInt8 ===
            // "function"`, the method's own `name`/`length`
            // (`getXxx`.length === 1, `setXxx`.length === 2), and
            // `DataView.prototype.getInt8.call(dv, 0)` all behave per spec.
            if name == "DataView" {
                for &m in DATA_VIEW_METHODS {
                    let m_h = self.realm.new_string(m);
                    let f = self.realm.new_bound_native(N_DATA_VIEW_PROTO_FN, m_h);
                    let arity = if m.starts_with("set") { 2 } else { 1 };
                    self.install_fn_name_length(f, m, arity);
                    self.realm
                        .set_property(proto, m, NanBox::handle(f.to_raw()));
                    self.realm.mark_hidden(proto, m);
                }
                // `DataView.prototype[Symbol.toStringTag]` is "DataView"
                // `{ writable: false, enumerable: false, configurable: true }`.
                let tag_sym = self.well_known_symbol("toStringTag");
                let tag_key = self.member_key(tag_sym);
                let tag_val = self.realm.new_string("DataView");
                self.realm
                    .set_property(proto, &tag_key, NanBox::handle(tag_val.to_raw()));
                self.realm.mark_hidden(proto, &tag_key);
                self.realm.set_readonly_property(proto, &tag_key);
            }
            // `ArrayBuffer.prototype` methods as first-class own data properties
            // (each a bound native re-dispatched through `call_method` with an
            // `[[ArrayBufferData]]`-validated `this`), so `typeof ab.slice ===
            // "function"`, each method's own `name`/`length`, and
            // `ArrayBuffer.prototype.transfer.call(ab)` all behave per spec.
            if name == "ArrayBuffer" {
                for &m in AB_PROTO_METHODS {
                    let f = self.readable_ab_method(m);
                    self.realm.set_property(proto, m, f);
                    self.realm.mark_hidden(proto, m);
                }
                // `ArrayBuffer.prototype[Symbol.toStringTag]` is "ArrayBuffer"
                // `{ writable: false, enumerable: false, configurable: true }`.
                let tag_sym = self.well_known_symbol("toStringTag");
                let tag_key = self.member_key(tag_sym);
                let tag_val = self.realm.new_string("ArrayBuffer");
                self.realm
                    .set_property(proto, &tag_key, NanBox::handle(tag_val.to_raw()));
                self.realm.mark_hidden(proto, &tag_key);
                self.realm.set_readonly_property(proto, &tag_key);
            }
            self.realm
                .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
            // A constructor's `prototype` is `{ writable: false, enumerable: false,
            // configurable: false }`.
            self.realm.mark_hidden(ctor, "prototype");
            self.realm.set_readonly_property(ctor, "prototype");
            self.realm.set_non_configurable_property(ctor, "prototype");
        }
    }

    /// A readable bound native for a call-only method `name` (dispatched in `call_method`),
    /// so `typeof obj.method === "function"` and a detached `obj.method.call(obj, …)` work.
    fn readable_native_method(&mut self, name: &str) -> NanBox {
        let name_h = self.realm.new_string(name);
        let f = self.realm.new_bound_native(N_ARRAY_PROTO_FN, name_h);
        self.install_fn_name_length(f, name, builtin_method_arity(name));
        NanBox::handle(f.to_raw())
    }

    /// Like [`readable_native_method`], but for an `ArrayBuffer.prototype` method
    /// whose dispatch must first reject a `this` lacking the `[[ArrayBufferData]]`
    /// internal slot (see [`N_AB_PROTO_FN`]).
    fn readable_ab_method(&mut self, name: &str) -> NanBox {
        self.readable_buffer_method(name, N_AB_PROTO_FN)
    }

    /// Like [`readable_ab_method`](Self::readable_ab_method) but with an explicit
    /// dispatch id — `N_AB_PROTO_FN` for `ArrayBuffer.prototype` methods,
    /// `N_SAB_PROTO_FN` for `SharedArrayBuffer.prototype` methods (which
    /// brand-validate a shared receiver).
    fn readable_buffer_method(&mut self, name: &str, id: u16) -> NanBox {
        let name_h = self.realm.new_string(name);
        let f = self.realm.new_bound_native(id, name_h);
        self.install_fn_name_length(f, name, builtin_method_arity(name));
        NanBox::handle(f.to_raw())
    }

    /// Exposes a constructor's *static* methods (dispatched in `call_method`) as readable
    /// own properties — each a bound native that routes a read-then-call back through
    /// `call_method` with the constructor as `this`. So `typeof Promise.allSettled ===
    /// "function"` (feature detection) holds, not just `Promise.allSettled(...)` working.
    fn setup_static_methods(&mut self, ctor_name: &str, methods: &[&str]) {
        let Some(ns) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        for &name in methods {
            let name_h = self.realm.new_string(name);
            let pair = self.realm.new_array(alloc::vec![
                NanBox::handle(ns.to_raw()),
                NanBox::handle(name_h.to_raw()),
            ]);
            let f = self.realm.new_bound_native(N_STATIC_METHOD, pair);
            self.install_fn_name_length(f, name, builtin_method_arity(name));
            self.realm
                .set_property(ns, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(ns, name);
        }
    }

    /// Installs the ES2025 `uint8array-base64` proposal's six methods. The four
    /// instance methods (`toBase64`/`toHex`/`setFromBase64`/`setFromHex`) go on
    /// `Uint8Array.prototype`; the two statics (`fromBase64`/`fromHex`) on the
    /// `Uint8Array` constructor. Each is a named native (carrying its own
    /// `name`/`length`) installed as an own `{ writable: true, enumerable: false,
    /// configurable: true }` data property — exactly the proposal's descriptors.
    /// These are `Uint8Array`-specific, never on `%TypedArray%.prototype`.
    fn install_uint8array_base64(&mut self) {
        let Some(ctor) = self
            .current
            .get("Uint8Array")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        let proto = self
            .realm
            .get_property(ctor, "prototype")
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let install = |this: &mut Self, target: Handle, name: &str, id: u16| {
            let f = this.new_named_native(name, id);
            this.realm
                .set_property(target, name, NanBox::handle(f.to_raw()));
            this.realm.mark_hidden(target, name);
        };
        if let Some(proto) = proto {
            install(self, proto, "toBase64", N_UINT8_TO_BASE64);
            install(self, proto, "toHex", N_UINT8_TO_HEX);
            install(self, proto, "setFromBase64", N_UINT8_SET_FROM_BASE64);
            install(self, proto, "setFromHex", N_UINT8_SET_FROM_HEX);
        }
        install(self, ctor, "fromBase64", N_UINT8_FROM_BASE64);
        install(self, ctor, "fromHex", N_UINT8_FROM_HEX);
    }

    /// Installs a small built-in library: the `Math` object and the global
    /// coercion/parse functions. (A token stdlib to prove the native-call path;
    /// the full port is the remaining migration work.)
    fn install_globals(&mut self) {
        // An object whose properties are native methods, bound to `global_name`.
        let install_namespace = |this: &mut Self, global_name: &str, methods: &[(&str, u16)]| {
            let obj = this.realm.new_object();
            for (name, id) in methods {
                let f = this.new_named_native(name, *id);
                this.realm
                    .set_property(obj, name, NanBox::handle(f.to_raw()));
                // Built-in static/namespace methods are non-enumerable.
                this.realm.mark_hidden(obj, name);
            }
            this.current
                .declare(global_name, NanBox::handle(obj.to_raw()));
        };
        // Like `install_namespace`, but the global itself is a real callable
        // function cell carrying `ctor_id` (so `typeof Array === "function"`,
        // `Array instanceof Function`, `[[Prototype]] === Function.prototype`,
        // and `Object.prototype.toString` report `[object Function]`). Used for
        // the constructor globals `Object`/`Array`, which are otherwise dispatched
        // by global-binding identity in `call`/`construct`.
        let install_ctor_namespace =
            |this: &mut Self, global_name: &str, ctor_id: u16, methods: &[(&str, u16)]| {
                let obj = this.new_named_native(global_name, ctor_id);
                for (name, id) in methods {
                    let f = this.new_named_native(name, *id);
                    this.realm
                        .set_property(obj, name, NanBox::handle(f.to_raw()));
                    this.realm.mark_hidden(obj, name);
                }
                this.current
                    .declare(global_name, NanBox::handle(obj.to_raw()));
            };
        install_namespace(
            self,
            "Math",
            &[
                ("max", N_MATH_MAX),
                ("min", N_MATH_MIN),
                ("abs", N_MATH_ABS),
                ("floor", N_MATH_FLOOR),
                ("ceil", N_MATH_CEIL),
                ("round", N_MATH_ROUND),
                ("sqrt", N_MATH_SQRT),
                ("pow", N_MATH_POW),
                ("sign", N_MATH_SIGN),
                ("hypot", N_MATH_HYPOT),
                ("cbrt", N_MATH_CBRT),
                ("log2", N_MATH_LOG2),
                ("log10", N_MATH_LOG10),
                ("exp", N_MATH_EXP),
                ("log", N_MATH_LOG),
                ("random", N_MATH_RANDOM),
                ("trunc", N_MATH_TRUNC),
                ("sin", N_MATH_SIN),
                ("cos", N_MATH_COS),
                ("tan", N_MATH_TAN),
                ("asin", N_MATH_ASIN),
                ("acos", N_MATH_ACOS),
                ("atan", N_MATH_ATAN),
                ("atan2", N_MATH_ATAN2),
                ("sinh", N_MATH_SINH),
                ("cosh", N_MATH_COSH),
                ("tanh", N_MATH_TANH),
                ("asinh", N_MATH_ASINH),
                ("acosh", N_MATH_ACOSH),
                ("atanh", N_MATH_ATANH),
                ("expm1", N_MATH_EXPM1),
                ("log1p", N_MATH_LOG1P),
                ("fround", N_MATH_FROUND),
                ("f16round", N_MATH_F16ROUND),
                ("clz32", N_MATH_CLZ32),
                ("imul", N_MATH_IMUL),
                ("sumPrecise", N_MATH_SUM_PRECISE),
            ],
        );
        // The `Math` numeric constants.
        if let Some(mh) = self.current.get("Math").and_then(NanBox::as_handle) {
            let math = Handle::from_raw(mh);
            for (name, value) in [
                ("PI", core::f64::consts::PI),
                ("E", core::f64::consts::E),
                ("LN2", core::f64::consts::LN_2),
                ("LN10", core::f64::consts::LN_10),
                ("LOG2E", core::f64::consts::LOG2_E),
                ("LOG10E", core::f64::consts::LOG10_E),
                ("SQRT2", core::f64::consts::SQRT_2),
                ("SQRT1_2", core::f64::consts::FRAC_1_SQRT_2),
            ] {
                self.realm.set_property(math, name, NanBox::number(value));
                // The `Math` constants are `{ writable: false, enumerable: false,
                // configurable: false }`.
                self.realm.mark_hidden(math, name);
                self.realm.set_readonly_property(math, name);
                self.realm.set_non_configurable_property(math, name);
            }
            // `Math[Symbol.toStringTag]` is the string "Math"
            // `{ writable: false, enumerable: false, configurable: true }`.
            let tag_sym = self.well_known_symbol("toStringTag");
            let tag_key = self.member_key(tag_sym);
            let tag_val = self.new_str("Math");
            self.realm.set_property(math, &tag_key, tag_val);
            self.realm.mark_hidden(math, &tag_key);
            self.realm.set_readonly_property(math, &tag_key);
        }
        install_namespace(self, "console", &[("log", N_CONSOLE_LOG)]);
        // `Atomics` — single-agent read-modify-write over an integer `TypedArray`.
        install_namespace(
            self,
            "Atomics",
            &[
                ("add", N_ATOMICS_ADD),
                ("sub", N_ATOMICS_SUB),
                ("and", N_ATOMICS_AND),
                ("or", N_ATOMICS_OR),
                ("xor", N_ATOMICS_XOR),
                ("exchange", N_ATOMICS_EXCHANGE),
                ("compareExchange", N_ATOMICS_COMPARE_EXCHANGE),
                ("load", N_ATOMICS_LOAD),
                ("store", N_ATOMICS_STORE),
                ("isLockFree", N_ATOMICS_IS_LOCK_FREE),
                ("pause", N_ATOMICS_PAUSE),
                ("notify", N_ATOMICS_NOTIFY),
                ("wait", N_ATOMICS_WAIT),
                ("waitAsync", N_ATOMICS_WAIT_ASYNC),
            ],
        );
        if let Some(ah) = self.current.get("Atomics").and_then(NanBox::as_handle) {
            let atomics = Handle::from_raw(ah);
            let tag_sym = self.well_known_symbol("toStringTag");
            let tag_key = self.member_key(tag_sym);
            let tag_val = self.new_str("Atomics");
            self.realm.set_property(atomics, &tag_key, tag_val);
            self.realm.mark_hidden(atomics, &tag_key);
            self.realm.set_readonly_property(atomics, &tag_key);
        }
        // `Promise` is a native constructor (`new Promise(executor)`); its
        // `.resolve`/`.reject` statics are dispatched in `call_method`.
        let promise_ctor = self.new_named_native("Promise", N_PROMISE);
        self.current
            .declare("Promise", NanBox::handle(promise_ctor.to_raw()));
        // `Date` is a native constructor; `Date.now()` is a static.
        let date_ctor = self.new_named_native("Date", N_DATE);
        self.current
            .declare("Date", NanBox::handle(date_ctor.to_raw()));
        // `RegExp` is a native constructor.
        let regexp_ctor = self.new_named_native("RegExp", N_REGEXP);
        self.current
            .declare("RegExp", NanBox::handle(regexp_ctor.to_raw()));
        // `RegExp.escape` (ES2025) — a static method on the constructor.
        let escape_fn = self.new_named_native("escape", N_REGEXP_ESCAPE);
        self.realm
            .set_property(regexp_ctor, "escape", NanBox::handle(escape_fn.to_raw()));
        self.realm.mark_hidden(regexp_ctor, "escape");
        // The `Error` family — native constructors producing `{ name, message }`.
        // Only the standard errors are globals; the `WebAssembly.*` error
        // subclasses are installed under the WebAssembly namespace below.
        for (i, name) in ERROR_NAMES.iter().enumerate().take(N_GLOBAL_ERROR_COUNT) {
            let ctor = self.new_named_native(name, N_ERROR_BASE + i as u16);
            self.current.declare(name, NanBox::handle(ctor.to_raw()));
            // `Error.isError` (ES2025) — a static method on the base `Error`
            // constructor only (a `{writable, !enumerable, configurable}` data
            // property, like `RegExp.escape`).
            if i == 0 {
                let is_error_fn = self.new_named_native("isError", N_ERROR_IS_ERROR);
                self.realm
                    .set_property(ctor, "isError", NanBox::handle(is_error_fn.to_raw()));
                self.realm.mark_hidden(ctor, "isError");
            }
        }
        install_namespace(
            self,
            "JSON",
            &[
                ("stringify", N_JSON_STRINGIFY),
                ("parse", N_JSON_PARSE),
                ("rawJSON", N_JSON_RAW),
                ("isRawJSON", N_JSON_IS_RAW),
            ],
        );
        install_ctor_namespace(
            self,
            "Object",
            N_BASE_OBJECT,
            &[
                ("keys", N_OBJECT_KEYS),
                ("values", N_OBJECT_VALUES),
                ("assign", N_OBJECT_ASSIGN),
                ("entries", N_OBJECT_ENTRIES),
                ("fromEntries", N_OBJECT_FROM_ENTRIES),
                ("freeze", N_OBJECT_FREEZE),
                ("isFrozen", N_OBJECT_IS_FROZEN),
                ("seal", N_OBJECT_SEAL),
                ("isSealed", N_OBJECT_IS_SEALED),
                ("preventExtensions", N_OBJECT_PREVENT_EXT),
                ("isExtensible", N_OBJECT_IS_EXTENSIBLE),
                ("getOwnPropertyNames", N_OBJECT_GET_OWN_NAMES),
                ("getOwnPropertySymbols", N_OBJECT_GET_OWN_SYMBOLS),
                ("create", N_OBJECT_CREATE),
                ("getPrototypeOf", N_OBJECT_GET_PROTO),
                ("setPrototypeOf", N_OBJECT_SET_PROTO),
                ("defineProperty", N_OBJECT_DEFINE_PROP),
                ("defineProperties", N_OBJECT_DEFINE_PROPS),
                ("getOwnPropertyDescriptor", N_OBJECT_GET_OWN_DESC),
                ("getOwnPropertyDescriptors", N_OBJECT_GET_OWN_DESCS),
                ("is", N_OBJECT_IS),
                ("hasOwn", N_OBJECT_HAS_OWN),
                ("groupBy", N_OBJECT_GROUP_BY),
            ],
        );
        install_ctor_namespace(
            self,
            "Array",
            N_BASE_ARRAY,
            &[
                ("isArray", N_ARRAY_IS_ARRAY),
                ("from", N_ARRAY_FROM),
                ("fromAsync", N_ARRAY_FROM_ASYNC),
                ("of", N_ARRAY_OF),
            ],
        );
        install_namespace(
            self,
            "Reflect",
            &[
                ("get", N_REFLECT_GET),
                ("set", N_REFLECT_SET),
                ("has", N_REFLECT_HAS),
                ("ownKeys", N_REFLECT_OWN_KEYS),
                ("defineProperty", N_REFLECT_DEFINE_PROP),
                ("getOwnPropertyDescriptor", N_REFLECT_GET_OWN_DESC),
                ("getPrototypeOf", N_REFLECT_GET_PROTO),
                ("setPrototypeOf", N_REFLECT_SET_PROTO),
                ("deleteProperty", N_REFLECT_DELETE),
                ("apply", N_REFLECT_APPLY),
                ("construct", N_REFLECT_CONSTRUCT),
                ("isExtensible", N_REFLECT_IS_EXTENSIBLE),
                ("preventExtensions", N_REFLECT_PREVENT_EXT),
            ],
        );
        for (name, id) in [
            ("String", N_STRING),
            ("Number", N_NUMBER),
            ("Boolean", N_BOOLEAN),
            ("parseInt", N_PARSE_INT),
            ("parseFloat", N_PARSE_FLOAT),
            ("isNaN", N_IS_NAN),
            ("isFinite", N_IS_FINITE),
            ("Map", N_MAP),
            ("Set", N_SET),
            ("Symbol", N_SYMBOL),
            ("BigInt", N_BIGINT),
            ("Function", N_FUNCTION),
            ("Proxy", N_PROXY),
            ("WeakMap", N_WEAKMAP),
            ("WeakSet", N_WEAKSET),
            ("WeakRef", N_WEAKREF),
            ("FinalizationRegistry", N_FINALIZATION_REGISTRY),
            ("encodeURIComponent", N_ENCODE_URI_COMPONENT),
            ("decodeURIComponent", N_DECODE_URI_COMPONENT),
            ("encodeURI", N_ENCODE_URI),
            ("decodeURI", N_DECODE_URI),
            ("escape", N_ESCAPE),
            ("unescape", N_UNESCAPE),
            ("structuredClone", N_STRUCTURED_CLONE),
            ("setTimeout", N_SET_TIMEOUT),
            ("clearTimeout", N_CLEAR_TIMEOUT),
            ("queueMicrotask", N_QUEUE_MICROTASK),
            ("btoa", N_BTOA),
            ("atob", N_ATOB),
            ("URIError", N_URI_ERROR),
            ("EvalError", N_EVAL_ERROR),
            ("eval", N_EVAL),
            // Test262 host hook: `$262.detachArrayBuffer` is wired (by the runner's
            // JS prelude) to this global so detach-dependent tests can run.
            ("$262_detachArrayBuffer", N_DETACH_ARRAY_BUFFER),
            // Test262 host hook: `$262.IsHTMLDDA` is wired (by the runner's JS
            // prelude) to this global so [[IsHTMLDDA]] tests can obtain the value.
            ("$262_IsHTMLDDA", N_HTMLDDA),
            // Test262 host hook: `$262.createRealm` is wired (by the runner's JS
            // prelude) to this global so cross-realm tests can obtain a second
            // realm with a distinct set of intrinsics.
            ("$262_createRealm", N_262_CREATE_REALM),
            // Test262 host hook: `$262.evalScript` runs code as a Script in the
            // current realm (so top-level lexical declarations persist globally).
            ("$262_evalScript", N_262_EVAL_SCRIPT_MAIN),
            // Test262 `$262.agent` host hooks (wired by the runner's JS prelude,
            // and by the worker-side prelude `$262.agent.start` installs). See the
            // `agent` module for the cooperative-scheduler model.
            ("$262_agent_start", N_262_AGENT_START),
            ("$262_agent_broadcast", N_262_AGENT_BROADCAST),
            ("$262_agent_getReport", N_262_AGENT_GET_REPORT),
            ("$262_agent_getReportAsync", N_262_AGENT_GET_REPORT_ASYNC),
            ("$262_agent_report", N_262_AGENT_REPORT),
            ("$262_agent_receiveBroadcast", N_262_AGENT_RECEIVE_BROADCAST),
            ("$262_agent_sleep", N_262_AGENT_SLEEP),
            ("$262_agent_monotonicNow", N_262_AGENT_MONOTONIC_NOW),
        ] {
            let f = self.new_named_native(name, id);
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
        // The `Intl` namespace with its format constructors.
        let intl = self.realm.new_object();
        for (name, id) in [
            ("NumberFormat", N_INTL_NUMBER_FORMAT),
            ("DateTimeFormat", N_INTL_DATETIME_FORMAT),
            ("Collator", N_INTL_COLLATOR),
            ("PluralRules", N_INTL_PLURAL_RULES),
            ("ListFormat", N_INTL_LIST_FORMAT),
            ("RelativeTimeFormat", N_INTL_REL_TIME),
            ("DisplayNames", N_INTL_DISPLAY_NAMES),
            ("Segmenter", N_INTL_SEGMENTER),
            ("DurationFormat", N_INTL_DURATION_FORMAT),
        ] {
            let f = self.new_named_native(name, id);
            // `Intl.X.supportedLocalesOf(locales)` — static on every constructor.
            let sl = self.new_named_native("supportedLocalesOf", N_INTL_SUPPORTED_LOCALES);
            self.realm
                .set_hidden_property(f, "supportedLocalesOf", NanBox::handle(sl.to_raw()));
            self.realm
                .set_property(intl, name, NanBox::handle(f.to_raw()));
            // ECMA-402: every service constructor is a non-enumerable property of
            // `Intl` (`{ writable:true, enumerable:false, configurable:true }`).
            self.realm.mark_hidden(intl, name);
        }
        // `Intl.Locale` — a constructor with no `supportedLocalesOf` static.
        {
            let f = self.new_named_native("Locale", N_INTL_LOCALE);
            self.realm
                .set_property(intl, "Locale", NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(intl, "Locale");
        }
        // `Intl.getCanonicalLocales` / `Intl.supportedValuesOf` — namespace functions.
        for (name, id) in [
            ("getCanonicalLocales", N_INTL_GET_CANONICAL_LOCALES),
            ("supportedValuesOf", N_INTL_SUPPORTED_VALUES_OF),
        ] {
            let f = self.new_named_native(name, id);
            self.realm
                .set_property(intl, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(intl, name);
        }
        // ECMA-402: the `Intl` namespace object is an ordinary object whose
        // [[Prototype]] is %Object.prototype% and which carries an own
        // `[Symbol.toStringTag]` of `"Intl"`. Both are installed later (with the
        // other namespace objects, once `Object.prototype` exists) — see the
        // `Reflect`/`JSON`/`Math` fix-up loop.
        self.current.declare("Intl", NanBox::handle(intl.to_raw()));
        // The typed-array constructors.
        for (i, (name, _)) in TYPED_ARRAY_KINDS.iter().enumerate() {
            let f = self.new_named_native(name, N_TYPED_ARRAY_BASE + i as u16);
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
        for (name, id) in [("ArrayBuffer", N_ARRAY_BUFFER), ("DataView", N_DATA_VIEW)] {
            // `new_named_native` installs the constructor's `name`/`length`
            // (`DataView.name === "DataView"`, `.length === 1`), each
            // `{ writable: false, enumerable: false, configurable: true }`.
            let f = self.new_named_native(name, id);
            // `ArrayBuffer.isView(x)` — true for a typed array or a DataView.
            if id == N_ARRAY_BUFFER {
                let isview = self.realm.new_native(N_ARRAY_BUFFER_IS_VIEW);
                self.install_fn_name_length(isview, "isView", 1);
                self.realm
                    .set_hidden_property(f, "isView", NanBox::handle(isview.to_raw()));
            }
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
        // `SharedArrayBuffer` — installed additively (its byte store, typed-array
        // and `Atomics` integration reuse the `ArrayBuffer` machinery; only the
        // prototype's accessor names + `[Symbol.toStringTag]` differ).
        {
            let sab_ctor = self.new_named_native("SharedArrayBuffer", N_SHARED_ARRAY_BUFFER);
            self.current
                .declare("SharedArrayBuffer", NanBox::handle(sab_ctor.to_raw()));
            let proto = self.realm.new_object_with_proto(self.object_prototype());
            self.realm
                .set_hidden_property(proto, "constructor", NanBox::handle(sab_ctor.to_raw()));
            // The `byteLength` / `maxByteLength` / `growable` accessors share the
            // ArrayBuffer getter (`N_AB_ACCESSOR`), which validates the receiver via
            // its `ARRAY_BUFFER_BYTES` slot — so a read on `SharedArrayBuffer
            // .prototype` itself throws, but an instance read works.
            for accessor in ["byteLength", "maxByteLength", "growable"] {
                let name_h = self.realm.new_string(accessor);
                let getter = self.realm.new_bound_native(N_SAB_ACCESSOR, name_h);
                self.install_fn_name_length(getter, &alloc::format!("get {accessor}"), 0);
                self.realm.define_accessor(
                    proto,
                    accessor,
                    NanBox::handle(getter.to_raw()),
                    NanBox::undefined(),
                );
                self.realm.mark_hidden(proto, accessor);
            }
            // `grow` (growable SABs) and `slice` (returns a new SAB) — bound natives
            // re-dispatched through `call_method`, like the ArrayBuffer methods.
            for m in ["grow", "slice"] {
                let f = self.readable_buffer_method(m, N_SAB_PROTO_FN);
                self.realm.set_property(proto, m, f);
                self.realm.mark_hidden(proto, m);
            }
            let tag_sym = self.well_known_symbol("toStringTag");
            let tag_key = self.member_key(tag_sym);
            let tag_val = self.new_str("SharedArrayBuffer");
            self.realm.set_property(proto, &tag_key, tag_val);
            self.realm.mark_hidden(proto, &tag_key);
            self.realm.set_readonly_property(proto, &tag_key);
            self.realm
                .set_property(sab_ctor, "prototype", NanBox::handle(proto.to_raw()));
            self.realm.mark_hidden(sab_ctor, "prototype");
            self.realm.set_readonly_property(sab_ctor, "prototype");
            self.realm
                .set_non_configurable_property(sab_ctor, "prototype");
        }
        // The `Iterator` global — the `%Iterator%` abstract constructor. Direct
        // `new Iterator()` / `Iterator()` throw (abstract); `Iterator.from(x)`
        // wraps any iterable for the ES2025 helper methods. Its `prototype`
        // (`%IteratorPrototype%`) carries `[Symbol.iterator]()` returning `this`,
        // so an object inheriting it is itself iterable.
        let iterator_ctor = self.new_named_native("Iterator", N_ITERATOR);
        let from_fn = self.new_named_native("from", N_ITERATOR_FROM);
        self.realm
            .set_hidden_property(iterator_ctor, "from", NanBox::handle(from_fn.to_raw()));
        let iter_proto = self.realm.new_object();
        // `%IteratorPrototype%[Symbol.iterator]` returns `this` (a native bound to
        // the receiver at call time).
        let self_iter = self.realm.new_native(N_ITERATOR_PROTO_SELF);
        self.install_fn_name_length(self_iter, "[Symbol.iterator]", 0);
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        self.realm
            .set_hidden_property(iter_proto, &iter_key, NanBox::handle(self_iter.to_raw()));
        // `%IteratorPrototype%[Symbol.dispose]()` — invokes the iterator's `return`.
        let dispose_native = self.realm.new_native(N_ITERATOR_DISPOSE);
        self.install_fn_name_length(dispose_native, "[Symbol.dispose]", 0);
        let dispose_sym = self.well_known_symbol("dispose");
        let dispose_key = self.member_key(dispose_sym);
        self.realm.set_property(
            iter_proto,
            &dispose_key,
            NanBox::handle(dispose_native.to_raw()),
        );
        self.realm.mark_hidden(iter_proto, &dispose_key);
        // `%IteratorPrototype%[Symbol.toStringTag]` is a get/set accessor
        // (`get` → "Iterator"; `set` = SetterThatIgnoresPrototypeProperties),
        // enumerable: false, configurable: true — so `Object.prototype.toString`
        // on a bare iterator yields `[object Iterator]`.
        let tag_get = self.realm.new_native(N_ITERATOR_TAG_GET);
        self.install_fn_name_length(tag_get, "get [Symbol.toStringTag]", 0);
        let tag_set = self.realm.new_native(N_ITERATOR_TAG_SET);
        self.install_fn_name_length(tag_set, "set [Symbol.toStringTag]", 1);
        let tag_sym = self.well_known_symbol("toStringTag");
        let tag_key = self.member_key(tag_sym);
        self.realm.define_accessor(
            iter_proto,
            &tag_key,
            NanBox::handle(tag_get.to_raw()),
            NanBox::handle(tag_set.to_raw()),
        );
        self.realm.mark_hidden(iter_proto, &tag_key);
        // The ES2025 helper methods as first-class functions on `%IteratorPrototype%`
        // (so `Iterator.prototype.map`, `it.map(...)` resolve through the chain).
        for &name in ITERATOR_PROTO_METHODS {
            let name_h = self.realm.new_string(name);
            let f = self.realm.new_bound_native(N_ITERATOR_PROTO_FN, name_h);
            self.install_fn_name_length(f, name, builtin_method_arity(name));
            self.realm
                .set_property(iter_proto, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(iter_proto, name);
        }
        // `%IteratorPrototype%.constructor` is a get/set accessor (get → %Iterator%;
        // set = SetterThatIgnoresPrototypeProperties), enumerable: false,
        // configurable: true — not a plain data property.
        let ctor_get = self.realm.new_native(N_ITERATOR_CTOR_GET);
        self.install_fn_name_length(ctor_get, "get constructor", 0);
        let ctor_set = self.realm.new_native(N_ITERATOR_CTOR_SET);
        self.install_fn_name_length(ctor_set, "set constructor", 1);
        self.realm.define_accessor(
            iter_proto,
            "constructor",
            NanBox::handle(ctor_get.to_raw()),
            NanBox::handle(ctor_set.to_raw()),
        );
        self.realm.mark_hidden(iter_proto, "constructor");
        self.realm.set_hidden_property(
            iterator_ctor,
            "prototype",
            NanBox::handle(iter_proto.to_raw()),
        );
        self.realm.mark_hidden(iterator_ctor, "prototype");
        self.realm.set_readonly_property(iterator_ctor, "prototype");
        self.realm
            .set_non_configurable_property(iterator_ctor, "prototype");
        // `Iterator.concat` — the `iterator-sequencing` static (lazy concatenation).
        let concat_fn = self.realm.new_native(N_ITERATOR_CONCAT);
        self.install_fn_name_length(concat_fn, "concat", 0);
        self.realm
            .set_property(iterator_ctor, "concat", NanBox::handle(concat_fn.to_raw()));
        self.realm.mark_hidden(iterator_ctor, "concat");
        // `Iterator.zip` / `Iterator.zipKeyed` — the `joint-iteration` statics.
        let zip_fn = self.realm.new_native(N_ITERATOR_ZIP);
        self.install_fn_name_length(zip_fn, "zip", 1);
        self.realm
            .set_property(iterator_ctor, "zip", NanBox::handle(zip_fn.to_raw()));
        self.realm.mark_hidden(iterator_ctor, "zip");
        let zipk_fn = self.realm.new_native(N_ITERATOR_ZIP_KEYED);
        self.install_fn_name_length(zipk_fn, "zipKeyed", 1);
        self.realm
            .set_property(iterator_ctor, "zipKeyed", NanBox::handle(zipk_fn.to_raw()));
        self.realm.mark_hidden(iterator_ctor, "zipKeyed");
        // `%IteratorHelperPrototype%` — the prototype of every lazy helper
        // (`map`/`filter`/`take`/`drop`/`flatMap` results). Inherits
        // `%IteratorPrototype%`; carries `next`/`return` and a `Symbol.toStringTag`.
        let helper_proto = self.realm.new_object_with_proto(Some(iter_proto));
        let hn = self.realm.new_native(N_ITER_HELPER_NEXT);
        self.install_fn_name_length(hn, "next", 0);
        self.realm
            .set_property(helper_proto, "next", NanBox::handle(hn.to_raw()));
        self.realm.mark_hidden(helper_proto, "next");
        let hr = self.realm.new_native(N_ITER_HELPER_RETURN);
        self.install_fn_name_length(hr, "return", 0);
        self.realm
            .set_property(helper_proto, "return", NanBox::handle(hr.to_raw()));
        self.realm.mark_hidden(helper_proto, "return");
        let tag = self.new_str("Iterator Helper");
        let tt_sym = self.well_known_symbol("toStringTag");
        let tt_key = self.member_key(tt_sym);
        self.realm.set_property(helper_proto, &tt_key, tag);
        self.realm.mark_hidden(helper_proto, &tt_key);
        self.realm.set_readonly_property(helper_proto, &tt_key);
        // `%WrapForValidIteratorPrototype%` — the prototype of the `Iterator.from`
        // wrapper. Inherits `%IteratorPrototype%`; carries `next`/`return`.
        let wrap_proto = self.realm.new_object_with_proto(Some(iter_proto));
        let wn = self.realm.new_native(N_ITER_WRAP_NEXT);
        self.install_fn_name_length(wn, "next", 0);
        self.realm
            .set_property(wrap_proto, "next", NanBox::handle(wn.to_raw()));
        self.realm.mark_hidden(wrap_proto, "next");
        let wr = self.realm.new_native(N_ITER_WRAP_RETURN);
        self.install_fn_name_length(wr, "return", 0);
        self.realm
            .set_property(wrap_proto, "return", NanBox::handle(wr.to_raw()));
        self.realm.mark_hidden(wrap_proto, "return");
        // `%ConcatIteratorPrototype%` — the prototype of an `Iterator.concat` result.
        let concat_proto = self.realm.new_object_with_proto(Some(iter_proto));
        let cn = self.realm.new_native(N_ITER_CONCAT_NEXT);
        self.install_fn_name_length(cn, "next", 0);
        self.realm
            .set_property(concat_proto, "next", NanBox::handle(cn.to_raw()));
        self.realm.mark_hidden(concat_proto, "next");
        let cr = self.realm.new_native(N_ITER_CONCAT_RETURN);
        self.install_fn_name_length(cr, "return", 0);
        self.realm
            .set_property(concat_proto, "return", NanBox::handle(cr.to_raw()));
        self.realm.mark_hidden(concat_proto, "return");
        let ctag = self.new_str("Iterator Helper");
        self.realm.set_property(concat_proto, &tt_key, ctag);
        self.realm.mark_hidden(concat_proto, &tt_key);
        self.realm.set_readonly_property(concat_proto, &tt_key);
        // `%ZipIteratorPrototype%` — the prototype of an `Iterator.zip`/`zipKeyed`
        // result.
        let zip_proto = self.realm.new_object_with_proto(Some(iter_proto));
        let zn = self.realm.new_native(N_ITER_ZIP_NEXT);
        self.install_fn_name_length(zn, "next", 0);
        self.realm
            .set_property(zip_proto, "next", NanBox::handle(zn.to_raw()));
        self.realm.mark_hidden(zip_proto, "next");
        let zr = self.realm.new_native(N_ITER_ZIP_RETURN);
        self.install_fn_name_length(zr, "return", 0);
        self.realm
            .set_property(zip_proto, "return", NanBox::handle(zr.to_raw()));
        self.realm.mark_hidden(zip_proto, "return");
        let ztag = self.new_str("Iterator Helper");
        self.realm.set_property(zip_proto, &tt_key, ztag);
        self.realm.mark_hidden(zip_proto, &tt_key);
        self.realm.set_readonly_property(zip_proto, &tt_key);
        self.realm.set_hidden_property(
            iterator_ctor,
            ITER_ZIP_PROTO_SLOT,
            NanBox::handle(zip_proto.to_raw()),
        );
        // Stash the three helper prototypes as hidden slots on the Iterator
        // constructor so the helper-building code can retrieve them.
        self.realm.set_hidden_property(
            iterator_ctor,
            ITER_HELPER_PROTO_SLOT,
            NanBox::handle(helper_proto.to_raw()),
        );
        self.realm.set_hidden_property(
            iterator_ctor,
            ITER_WRAP_PROTO_SLOT,
            NanBox::handle(wrap_proto.to_raw()),
        );
        self.realm.set_hidden_property(
            iterator_ctor,
            ITER_CONCAT_PROTO_SLOT,
            NanBox::handle(concat_proto.to_raw()),
        );
        self.current
            .declare("Iterator", NanBox::handle(iterator_ctor.to_raw()));
        // The `WebAssembly` namespace, backed by the in-house WASM engine
        // (`wasm_rt`). `validate(bytes)` decodes a module and reports whether it
        // is well-formed.
        install_namespace(
            self,
            "WebAssembly",
            &[
                ("validate", N_WASM_VALIDATE),
                ("instantiate", N_WASM_INSTANTIATE),
                ("Module", N_WASM_MODULE),
                ("Instance", N_WASM_INSTANCE),
                ("compile", N_WASM_COMPILE),
                ("Global", N_WASM_GLOBAL),
                ("Memory", N_WASM_MEMORY),
                ("Table", N_WASM_TABLE),
                ("CompileError", N_WASM_COMPILE_ERROR),
                ("LinkError", N_WASM_LINK_ERROR),
                ("RuntimeError", N_WASM_RUNTIME_ERROR),
            ],
        );
        // Static introspection methods on `WebAssembly.Module`.
        if let Some(module_ctor) = self
            .current
            .get("WebAssembly")
            .and_then(|ns| ns.as_handle())
            .map(Handle::from_raw)
            .and_then(|ns| self.realm.get_property(ns, "Module"))
            .and_then(|m| m.as_handle())
            .map(Handle::from_raw)
        {
            for (name, id) in [
                ("exports", N_WASM_MODULE_EXPORTS),
                ("imports", N_WASM_MODULE_IMPORTS),
            ] {
                let f = self.realm.new_native(id);
                self.realm
                    .set_property(module_ctor, name, NanBox::handle(f.to_raw()));
            }
        }
        // A minimal `Object.prototype` carrying the methods commonly invoked via
        // `Object.prototype.<m>.call(x)`. The receiver arrives as `this`.
        let obj_proto = self.realm.new_object();
        for (name, id, arity) in [
            ("toString", N_OBJ_PROTO_TOSTRING, 0u32),
            ("toLocaleString", N_OBJ_PROTO_TOLOCALESTRING, 0),
            ("valueOf", N_OBJ_PROTO_VALUEOF, 0),
            ("hasOwnProperty", N_OBJ_PROTO_HASOWN, 1),
            ("isPrototypeOf", N_OBJ_PROTO_ISPROTOTYPEOF, 1),
            ("propertyIsEnumerable", N_OBJ_PROTO_PROPISENUM, 1),
            // Annex B legacy accessor-manipulation methods.
            ("__defineGetter__", N_OBJ_DEFINE_GETTER, 2),
            ("__defineSetter__", N_OBJ_DEFINE_SETTER, 2),
            ("__lookupGetter__", N_OBJ_LOOKUP_GETTER, 1),
            ("__lookupSetter__", N_OBJ_LOOKUP_SETTER, 1),
        ] {
            let f = self.realm.new_native(id);
            self.install_fn_name_length(f, name, arity);
            self.realm
                .set_property(obj_proto, name, NanBox::handle(f.to_raw()));
            // Non-enumerable, so inheriting objects don't surface them in for-in /
            // Object.keys.
            self.realm.mark_hidden(obj_proto, name);
        }
        // `Object.prototype.__proto__` (Annex B): an accessor pair
        // `{ enumerable: false, configurable: true }` whose getter is the object's
        // `[[GetPrototypeOf]]` and whose setter is `[[SetPrototypeOf]]`.
        {
            let getter = self.realm.new_native(N_OBJ_PROTO_GET);
            self.install_fn_name_length(getter, "get __proto__", 0);
            let setter = self.realm.new_native(N_OBJ_PROTO_SET);
            self.install_fn_name_length(setter, "set __proto__", 1);
            self.realm.define_accessor(
                obj_proto,
                "__proto__",
                NanBox::handle(getter.to_raw()),
                NanBox::handle(setter.to_raw()),
            );
            self.realm.mark_hidden(obj_proto, "__proto__");
        }
        if let Some(obj_ns) = self
            .current
            .get("Object")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm
                .set_property(obj_ns, "prototype", NanBox::handle(obj_proto.to_raw()));
            // `Object.prototype` is `{ writable:false, enumerable:false,
            // configurable:false }` like every built-in constructor's `prototype`.
            self.realm.mark_hidden(obj_ns, "prototype");
            self.realm.set_readonly_property(obj_ns, "prototype");
            self.realm
                .set_non_configurable_property(obj_ns, "prototype");
            // `({}).constructor === Object` (non-enumerable, inherited via the
            // default object prototype), and `Object.name === "Object"`.
            self.realm.set_hidden_property(
                obj_proto,
                "constructor",
                NanBox::handle(obj_ns.to_raw()),
            );
            let name = self.new_str("Object");
            self.realm.set_hidden_property(obj_ns, "name", name);
            self.realm.set_readonly_property(obj_ns, "name");
        }
        // `<Ctor>.prototype` as a real object whose methods are first-class values
        // that dispatch on their `this`, so the classic `Array.prototype.slice.call`
        // / `String.prototype.X.call` / `Function.prototype.bind.call` idioms work.
        self.setup_first_class_prototype("Array", ARRAY_PROTO_METHODS);
        // Record `%Array.prototype%` as the default `[[Prototype]]` of every dense
        // array (so `Object.getPrototypeOf([])`, `[] instanceof Array`, and
        // `"push" in []` resolve through the chain).
        if let Some(arr_proto) = self
            .current
            .get("Array")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_array_proto_intrinsic(arr_proto);
            // `Array.prototype` is an Array exotic object with an own `length`
            // property (`{ value: 0, writable: true, enumerable: false,
            // configurable: false }`). We model the prototype itself as a plain
            // object, but expose the own `length` so `Array.prototype.length === 0`
            // and `"length" in Object.create(Array.prototype)` hold.
            self.realm
                .set_property(arr_proto, "length", NanBox::number(0.0));
            self.realm.mark_hidden(arr_proto, "length");
            self.realm
                .set_non_configurable_property(arr_proto, "length");
            // `Array.prototype[Symbol.iterator]` is the *same* function object as
            // `Array.prototype.values` (per spec), so `[][Symbol.iterator] ===
            // [].values` and the `arguments` object's iterator matches
            // `[][Symbol.iterator]`. Installed as a non-enumerable own property.
            if let Some(values) = self.realm.get_property(arr_proto, "values") {
                let iter_sym = self.well_known_symbol("iterator");
                let iter_key = self.member_key(iter_sym);
                self.realm.set_hidden_property(arr_proto, &iter_key, values);
            }
        }
        // `Array.prototype[Symbol.unscopables]` — a null-prototype object whose
        // own enumerable data properties (all `true`) name the methods excluded
        // from `with` statement scope. The property itself is non-enumerable,
        // non-writable, configurable.
        if let Some(arr_proto) = self
            .current
            .get("Array")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            let unscopables = self.realm.new_object_with_proto(None);
            for name in [
                "at",
                "copyWithin",
                "entries",
                "fill",
                "find",
                "findIndex",
                "findLast",
                "findLastIndex",
                "flat",
                "flatMap",
                "includes",
                "keys",
                "toReversed",
                "toSorted",
                "toSpliced",
                "values",
            ] {
                self.realm
                    .set_property(unscopables, name, NanBox::boolean(true));
            }
            let sym = self.well_known_symbol("unscopables");
            let key = self.member_key(sym);
            self.realm
                .set_property(arr_proto, &key, NanBox::handle(unscopables.to_raw()));
            self.realm.mark_hidden(arr_proto, &key);
            self.realm.set_readonly_property(arr_proto, &key);
        }
        self.setup_first_class_prototype_id("String", STRING_PROTO_METHODS, N_STRING_PROTO_FN);
        // Annex B.2.3: `String.prototype.trimLeft`/`trimRight` are the *same*
        // function objects as `trimStart`/`trimEnd` (`===`-identical, and their
        // `name` is "trimStart"/"trimEnd"). Install the shared handles as
        // additional writable/configurable, non-enumerable data properties.
        if let Some(str_proto) = self
            .current
            .get("String")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|s| self.realm.get_property(s, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            for (alias, original) in [("trimLeft", "trimStart"), ("trimRight", "trimEnd")] {
                if let Some(f) = self.realm.get_property(str_proto, original) {
                    self.realm.set_property(str_proto, alias, f);
                    self.realm.mark_hidden(str_proto, alias);
                }
            }
            // `String.prototype[Symbol.iterator]` — a real own function property
            // (writable, non-enumerable, configurable) so it is observable to
            // `getOwnPropertyDescriptor`/`verifyProperty`, has `length` 0 and
            // `name` "[Symbol.iterator]", and is not a constructor.
            let str_iter = self.realm.new_native(N_STRING_PROTO_ITER);
            self.install_fn_name_length(str_iter, "[Symbol.iterator]", 0);
            let iter_sym = self.well_known_symbol("iterator");
            let iter_key = self.member_key(iter_sym);
            self.realm
                .set_property(str_proto, &iter_key, NanBox::handle(str_iter.to_raw()));
            self.realm.mark_hidden(str_proto, &iter_key);
        }
        self.setup_first_class_prototype_id("Number", NUMBER_PROTO_METHODS, N_NUMBER_PROTO_FN);
        // `Number.prototype.toString ( [ radix ] )` has `length` 1 (the generic
        // `builtin_method_arity` maps "toString" to 0, which is right for
        // Object/Array/Date but not Number). Override just this method's length.
        if let Some(num_proto) = self
            .current
            .get("Number")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            && let Some(ts) = self
                .realm
                .get_property(num_proto, "toString")
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
        {
            self.install_fn_name_length(ts, "toString", 1);
        }
        // The `Number` numeric constants are own data properties of the
        // constructor with the built-in attributes `{ writable: false,
        // enumerable: false, configurable: false }` (so `hasOwnProperty` and
        // `verifyProperty` see them).
        if let Some(num_ctor) = self
            .current
            .get("Number")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            let consts: &[(&str, f64)] = &[
                ("MAX_SAFE_INTEGER", 9_007_199_254_740_991.0),
                ("MIN_SAFE_INTEGER", -9_007_199_254_740_991.0),
                ("MAX_VALUE", f64::MAX),
                ("MIN_VALUE", f64::from_bits(1)),
                ("EPSILON", f64::EPSILON),
                ("POSITIVE_INFINITY", f64::INFINITY),
                ("NEGATIVE_INFINITY", f64::NEG_INFINITY),
                ("NaN", f64::NAN),
            ];
            for &(name, value) in consts {
                self.realm
                    .set_property(num_ctor, name, NanBox::number(value));
                self.realm.mark_hidden(num_ctor, name);
                self.realm.set_readonly_property(num_ctor, name);
                self.realm.set_non_configurable_property(num_ctor, name);
            }
        }
        self.setup_first_class_prototype_id("Boolean", BOOLEAN_PROTO_METHODS, N_BOOLEAN_PROTO_FN);
        // `Number.prototype`/`Boolean.prototype`/`String.prototype` are themselves
        // wrapper objects with a default `[[NumberData]]`/`[[BooleanData]]`/
        // `[[StringData]]` (`+0`, `false`, `""`). They carry the matching
        // `PRIM_WRAP` so `Number.prototype.valueOf()` is `0` and
        // `Object.prototype.toString.call(Number.prototype)` is `"[object Number]"`.
        for (ctor, prim, id) in [
            ("Number", NanBox::number(0.0), N_NUMBER),
            ("Boolean", NanBox::boolean(false), N_BOOLEAN),
            // `String.prototype` is itself a String exotic object with
            // `[[StringData]] = ""` — so `String.prototype.valueOf()` /
            // `.toString()` return `""` (thisStringValue succeeds) and
            // `Object.prototype.toString.call(String.prototype)` is
            // `"[object String]"`. The boxed value is the empty string.
            ("String", NanBox::undefined(), N_STRING),
        ] {
            if let Some(proto) = self
                .current
                .get(ctor)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                .and_then(|c| self.realm.get_property(c, "prototype"))
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw)
            {
                // The empty-string box must be a real String cell (not the
                // `undefined` placeholder used to name the arm) so `string_value`
                // recovers `""`.
                let prim = if id == N_STRING {
                    NanBox::handle(self.realm.new_string("").to_raw())
                } else {
                    prim
                };
                self.realm.set_hidden_property(proto, PRIM_WRAP, prim);
                self.realm.set_hidden_property(
                    proto,
                    PRIM_WRAP_TYPE,
                    NanBox::number(f64::from(id)),
                );
            }
        }
        self.setup_first_class_prototype_id("BigInt", BIGINT_PROTO_METHODS, N_BIGINT_PROTO_FN);
        self.setup_first_class_prototype_id("Date", DATE_PROTO_METHODS, N_DATE_PROTO_FN);
        // Annex B.2.4: `Date.prototype.toGMTString` is the *same* function object
        // as `toUTCString` (`===`-identical), installed as a writable,
        // configurable, non-enumerable data property.
        if let Some(date_proto) = self
            .current
            .get("Date")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|d| self.realm.get_property(d, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            && let Some(f) = self.realm.get_property(date_proto, "toUTCString")
        {
            self.realm.set_property(date_proto, "toGMTString", f);
            self.realm.mark_hidden(date_proto, "toGMTString");
        }
        // `Date.prototype[Symbol.toPrimitive]` — a method (length 1) keyed by the
        // well-known symbol, `{ writable: false, enumerable: false,
        // configurable: true }`.
        if let Some(date_proto) = self
            .current
            .get("Date")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            let f = self.new_named_native("[Symbol.toPrimitive]", N_DATE_TO_PRIMITIVE);
            self.install_fn_name_length(f, "[Symbol.toPrimitive]", 1);
            let sym = self.well_known_symbol("toPrimitive");
            let key = self.member_key(sym);
            self.realm
                .set_property(date_proto, &key, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(date_proto, &key);
            self.realm.set_readonly_property(date_proto, &key);
            // `Date.prototype.toJSON` is a *generic* method (callable on any
            // object), so it is a plain named native rather than a Date-validating
            // first-class prototype method.
            let to_json = self.new_named_native("toJSON", N_DATE_TO_JSON);
            self.install_fn_name_length(to_json, "toJSON", 1);
            self.realm
                .set_property(date_proto, "toJSON", NanBox::handle(to_json.to_raw()));
            self.realm.mark_hidden(date_proto, "toJSON");
        }
        // `BigInt.prototype[Symbol.toStringTag]` is the string "BigInt", an own data
        // property `{writable:false, enumerable:false, configurable:true}`. Being
        // configurable, it can be redefined (e.g. tests overwrite it with a non-string
        // to verify `Object.prototype.toString` ignores non-string tags).
        if let Some(bi_proto) = self
            .current
            .get("BigInt")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|f| self.realm.get_property(f, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            let tag_sym = self.well_known_symbol("toStringTag");
            let tag_key = self.member_key(tag_sym);
            let tag_val = self.new_str("BigInt");
            self.realm.set_property(bi_proto, &tag_key, tag_val);
            self.realm.mark_hidden(bi_proto, &tag_key);
            self.realm.set_readonly_property(bi_proto, &tag_key);
            // Record `%BigInt.prototype%` as the `[[Prototype]]` of every BigInt
            // primitive, so a property read on `1n` (e.g. `@@toStringTag`,
            // `toString`, `valueOf`) resolves through the chain.
            self.realm.set_bigint_proto_intrinsic(bi_proto);
        }
        self.setup_regexp_prototype();
        self.setup_first_class_prototype("Function", FUNCTION_PROTO_METHODS);
        // Replace `Function.prototype.toString` with a *dedicated*-id native so it
        // is not confused with `Array.prototype.toString` when invoked indirectly
        // (both otherwise share `N_ARRAY_PROTO_FN`). Direct `fn.toString()` is
        // intercepted by the method-name shortcut; this value is what a `.call` /
        // `.apply` / ToPrimitive / `String(fn)` reaches.
        if let Some(func_proto) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|f| self.realm.get_property(f, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            let ts = self.new_named_native("toString", N_FUNCTION_TO_STRING);
            self.install_fn_name_length(ts, "toString", 0);
            self.realm
                .set_property(func_proto, "toString", NanBox::handle(ts.to_raw()));
            self.realm.mark_hidden(func_proto, "toString");
        }
        // Record `%Function.prototype%` as the default `[[Prototype]]` of every
        // ordinary/native callable (so `Object.getPrototypeOf(fn)` resolves to it
        // instead of `null`).
        if let Some(func_proto) = self
            .current
            .get("Function")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|f| self.realm.get_property(f, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_function_proto_intrinsic(func_proto);
            // `%Function.prototype%` is itself a function with own `name` "" and
            // `length` 0 (`{ w:false, e:false, c:true }`).
            self.install_fn_name_length(func_proto, "", 0);
            // `Function.prototype.caller` / `.arguments` are poisoned accessors
            // (`%ThrowTypeError%` for both get and set), enumerable: false,
            // configurable: true. Strict and bound functions inherit them, so
            // `boundFn.caller` / `strictFn.arguments` throw a TypeError without a
            // (forbidden) own property.
            let thrower_h = self.realm.new_native(N_THROW_TYPE_ERROR);
            // %ThrowTypeError% is a frozen function: own `name` "" / `length` 0
            // that are uniquely NON-configurable (as well as non-writable/
            // non-enumerable).
            self.install_fn_name_length(thrower_h, "", 0);
            self.realm.set_non_configurable_property(thrower_h, "name");
            self.realm
                .set_non_configurable_property(thrower_h, "length");
            // Record it as the realm's single canonical `%ThrowTypeError%` so a
            // strict `arguments` object's `callee` accessor is the *same* function
            // object (ECMA-262: there is exactly one `%ThrowTypeError%` per realm).
            self.realm.set_throw_type_error_intrinsic(thrower_h);
            let thrower = NanBox::handle(thrower_h.to_raw());
            for key in ["caller", "arguments"] {
                self.realm
                    .define_accessor(func_proto, key, thrower, thrower);
                self.realm.mark_hidden(func_proto, key);
            }
            // `Function.prototype[Symbol.hasInstance]` — OrdinaryHasInstance,
            // non-writable, non-enumerable, non-configurable (a first-class
            // native so `instanceof` and explicit `.call` both work).
            let hi_handle = self.realm.new_native(N_FN_HAS_INSTANCE);
            self.install_fn_name_length(hi_handle, "[Symbol.hasInstance]", 1);
            let has_instance = NanBox::handle(hi_handle.to_raw());
            let sym = self.well_known_symbol("hasInstance");
            let key = self.member_key(sym);
            self.realm
                .set_hidden_property(func_proto, &key, has_instance);
            self.realm.set_readonly_property(func_proto, &key);
            self.realm.set_non_configurable_property(func_proto, &key);
        }
        self.setup_first_class_prototype_id("Set", SET_PROTO_METHODS, N_SET_PROTO_FN);
        self.setup_first_class_prototype_id("Map", MAP_PROTO_METHODS, N_MAP_PROTO_FN);
        self.setup_first_class_prototype_id("WeakMap", WEAKMAP_PROTO_METHODS, N_WEAKMAP_PROTO_FN);
        self.setup_first_class_prototype_id("WeakSet", WEAKSET_PROTO_METHODS, N_WEAKSET_PROTO_FN);
        // Per spec, `Set.prototype.keys` is the *same* function object as
        // `Set.prototype.values`, and `Set.prototype[Symbol.iterator]` is also
        // that same object. Likewise `Map.prototype[Symbol.iterator]` is the
        // same object as `Map.prototype.entries`. Install the shared handles so
        // the `===` identity holds (writable, configurable, non-enumerable).
        if let Some(set_proto) = self
            .current
            .get("Set")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            && let Some(values) = self.realm.get_property(set_proto, "values")
        {
            self.realm.set_property(set_proto, "keys", values);
            self.realm.mark_hidden(set_proto, "keys");
            let iter_sym = self.well_known_symbol("iterator");
            let iter_key = self.member_key(iter_sym);
            self.realm.set_hidden_property(set_proto, &iter_key, values);
        }
        if let Some(map_proto) = self
            .current
            .get("Map")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            && let Some(entries) = self.realm.get_property(map_proto, "entries")
        {
            let iter_sym = self.well_known_symbol("iterator");
            let iter_key = self.member_key(iter_sym);
            self.realm
                .set_hidden_property(map_proto, &iter_key, entries);
        }
        // `WeakRef.prototype.deref` and `FinalizationRegistry.prototype.{register,
        // unregister}` — direct-id, brand-checking natives. Instances link to these
        // prototypes (see the `construct` arms).
        self.setup_direct_prototype("WeakRef", &[("deref", N_WEAKREF_DEREF)]);
        self.setup_direct_prototype(
            "FinalizationRegistry",
            &[
                ("register", N_FINREG_REGISTER),
                ("unregister", N_FINREG_UNREGISTER),
            ],
        );
        // `Promise.prototype` (then/catch/finally) — so `Promise.prototype.then`
        // is readable / detachable and `Promise.prototype[Symbol.toStringTag]`
        // exists. Promise instances link to it below.
        self.setup_first_class_prototype("Promise", PROMISE_PROTO_METHODS);
        // `Ctor.prototype[Symbol.toStringTag]` — a non-enumerable, non-writable,
        // configurable string. (`Object.prototype.toString` reads it.)
        self.install_proto_to_string_tag("Set", "Set");
        self.install_proto_to_string_tag("Map", "Map");
        // `Map.prototype.size`/`Set.prototype.size` accessors + `[Symbol.species]`.
        self.install_collection_accessors("Map", N_MAP_SIZE);
        self.install_collection_accessors("Set", N_SET_SIZE);
        // `get Array[Symbol.species]` — the shared species getter returns `this`
        // (the receiver constructor). `{ get, set: undefined, enumerable: false,
        // configurable: true }` per ECMA-262 23.1.2.5.
        self.install_ctor_species("Array");
        // `get Promise[Symbol.species]` → `this`, so `Promise[Symbol.species]` is
        // `Promise` and a subclass inherits it (`class P extends Promise {}` →
        // `P[Symbol.species] === P`), which is what `then`/combinators use as the
        // dependent-promise constructor (SpeciesConstructor).
        self.install_ctor_species("Promise");
        // `get ArrayBuffer[Symbol.species]` → `this`, used by
        // `ArrayBuffer.prototype.slice` (SpeciesConstructor). TypedArrays inherit
        // theirs from the shared `%TypedArray%` intrinsic.
        self.install_ctor_species("ArrayBuffer");
        self.install_temporal();
        self.install_proto_to_string_tag("WeakMap", "WeakMap");
        self.install_proto_to_string_tag("WeakSet", "WeakSet");
        self.install_proto_to_string_tag("Promise", "Promise");
        self.install_proto_to_string_tag("WeakRef", "WeakRef");
        self.install_proto_to_string_tag("FinalizationRegistry", "FinalizationRegistry");
        // `Symbol.prototype` — a real object carrying brand-checking methods.
        // Symbol PRIMITIVE behavior (`.description`/`.toString()`/`typeof`/keys)
        // keeps flowing through the existing fast paths in `read_member` /
        // `call_method`; this prototype makes `Symbol.prototype`,
        // `Object.getPrototypeOf(Symbol())`, and detached-method `.call(sym)` work.
        if let Some(sym_ctor) = self
            .current
            .get("Symbol")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            // The well-known symbols are own data properties of `%Symbol%` with
            // attributes { writable:false, enumerable:false, configurable:false }.
            for name in [
                "iterator",
                "asyncIterator",
                "match",
                "matchAll",
                "replace",
                "search",
                "split",
                "hasInstance",
                "isConcatSpreadable",
                "species",
                "toPrimitive",
                "toStringTag",
                "unscopables",
                "dispose",
                "asyncDispose",
            ] {
                let sym = self.well_known_symbol(name);
                self.realm.set_property(sym_ctor, name, sym);
                self.realm.mark_hidden(sym_ctor, name);
                self.realm.set_readonly_property(sym_ctor, name);
                self.realm.set_non_configurable_property(sym_ctor, name);
            }
            let proto = self.realm.new_object_with_proto(Some(obj_proto));
            // `toString` / `valueOf` — { writable:true, enumerable:false, configurable:true }.
            for (name, nid) in [
                ("toString", N_SYMBOL_PROTO_TOSTRING),
                ("valueOf", N_SYMBOL_PROTO_VALUEOF),
            ] {
                let f = self.new_named_native(name, nid);
                self.realm
                    .set_property(proto, name, NanBox::handle(f.to_raw()));
                self.realm.mark_hidden(proto, name);
            }
            // `get description` — accessor, { enumerable:false, configurable:true }, no setter.
            let desc_get = self.new_named_native("get description", N_SYMBOL_PROTO_DESC_GET);
            self.realm.define_accessor(
                proto,
                "description",
                NanBox::handle(desc_get.to_raw()),
                NanBox::undefined(),
            );
            self.realm.mark_hidden(proto, "description");
            // `[Symbol.toPrimitive]` — { writable:false, enumerable:false, configurable:true }.
            let to_prim = self.new_named_native("[Symbol.toPrimitive]", N_SYMBOL_PROTO_TOPRIMITIVE);
            let to_prim_sym = self.well_known_symbol("toPrimitive");
            let to_prim_key = self.member_key(to_prim_sym);
            self.realm
                .set_property(proto, &to_prim_key, NanBox::handle(to_prim.to_raw()));
            self.realm.mark_hidden(proto, &to_prim_key);
            self.realm.set_readonly_property(proto, &to_prim_key);
            // `[Symbol.toStringTag]` === "Symbol".
            self.install_to_string_tag(proto, "Symbol");
            // `constructor` — { writable:true, enumerable:false, configurable:true }.
            self.realm
                .set_hidden_property(proto, "constructor", NanBox::handle(sym_ctor.to_raw()));
            // Install on the constructor (read-only, non-enumerable, non-configurable)
            // and register as the intrinsic `[[Prototype]]` of Symbol primitives.
            self.realm
                .set_property(sym_ctor, "prototype", NanBox::handle(proto.to_raw()));
            self.realm.mark_hidden(sym_ctor, "prototype");
            self.realm.set_readonly_property(sym_ctor, "prototype");
            self.realm
                .set_non_configurable_property(sym_ctor, "prototype");
            self.realm.set_symbol_proto_intrinsic(proto);
        }
        // Namespace objects carry their own `[Symbol.toStringTag]` value (not on a
        // prototype): `Reflect`/`JSON`/`Math` → `[object Reflect|JSON|Math]`.
        for (ns_name, tag) in [
            ("Reflect", "Reflect"),
            ("JSON", "JSON"),
            ("Math", "Math"),
            ("Atomics", "Atomics"),
            ("Intl", "Intl"),
        ] {
            if let Some(ns) = self
                .current
                .get(ns_name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
            {
                self.install_to_string_tag(ns, tag);
                // These namespace objects were created (via `install_namespace`)
                // before `Object.prototype` existed, so link them now — their
                // `[[Prototype]]` is `%Object.prototype%` (an ordinary object).
                self.realm.set_object_proto(ns, Some(obj_proto));
            }
        }
        // `<ErrorCtor>.prototype` as a real object so `Error.prototype` /
        // `TypeError.prototype` are introspectable (e.g.
        // `Object.create(Error.prototype)`). `Error.prototype` inherits
        // `Object.prototype`; each subclass prototype inherits `Error.prototype`.
        // Each carries non-enumerable `constructor`/`name`/`message` defaults
        // (`Error.prototype.name === "Error"`, `…message === ""`).
        if let Some(error_proto) = self
            .current
            .get("Error")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .map(|ctor| {
                let proto = self.realm.new_object_with_proto(Some(obj_proto));
                self.realm
                    .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
                let nm = self.new_str("Error");
                self.realm.set_property(proto, "name", nm);
                self.realm.mark_hidden(proto, "name");
                let msg = self.new_str("");
                self.realm.set_property(proto, "message", msg);
                self.realm.mark_hidden(proto, "message");
                // `Error.prototype.toString` — its own (non-enumerable) method,
                // distinct from `Object.prototype.toString`: it requires an Object
                // receiver and renders `"name: message"`. Subclass prototypes
                // inherit it through `Error.prototype`.
                let ts = self.new_named_native("toString", N_ERROR_PROTO_TOSTRING);
                self.install_fn_name_length(ts, "toString", 0);
                self.realm
                    .set_property(proto, "toString", NanBox::handle(ts.to_raw()));
                self.realm.mark_hidden(proto, "toString");
                // `Error.prototype.stack` — the error-stack-accessor proposal: an
                // accessor property (`{ enumerable: false, configurable: true }`)
                // with both a getter and a setter. The getter returns an
                // implementation string for a `[[ErrorData]]`-branded receiver,
                // `undefined` for any other object, and throws for a non-object;
                // the setter shadows the accessor with an own data property. The
                // accessor lives only on `Error.prototype`; subclass prototypes
                // (e.g. `TypeError.prototype`) inherit it.
                // `new_named_native` installs `name`/`length` from
                // `builtin_native_arity` (getter 0, setter 1).
                let stack_get = self.new_named_native("get stack", N_ERROR_PROTO_STACK_GET);
                let stack_set = self.new_named_native("set stack", N_ERROR_PROTO_STACK_SET);
                self.realm.define_accessor(
                    proto,
                    "stack",
                    NanBox::handle(stack_get.to_raw()),
                    NanBox::handle(stack_set.to_raw()),
                );
                self.realm.mark_hidden(proto, "stack");
                self.realm
                    .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
                self.realm.mark_hidden(ctor, "prototype");
                // A constructor's `.prototype` is { writable:false, configurable:false }.
                self.realm.set_readonly_property(ctor, "prototype");
                self.realm.set_non_configurable_property(ctor, "prototype");
                proto
            })
        {
            // Every standard error subclass that is exposed as a JS global:
            // `Error`'s direct globals (`TypeError`…`AggregateError`) and the
            // separately-registered `URIError`/`EvalError`. Each gets a
            // `.prototype` inheriting `Error.prototype` with a `constructor`
            // back-link and a non-enumerable `name`.
            let mut subclass_names: Vec<&str> = ERROR_NAMES[1..N_GLOBAL_ERROR_COUNT].to_vec();
            subclass_names.push("URIError");
            subclass_names.push("EvalError");
            for name in subclass_names {
                if let Some(ctor) = self
                    .current
                    .get(name)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                {
                    let proto = self.realm.new_object_with_proto(Some(error_proto));
                    self.realm.set_hidden_property(
                        proto,
                        "constructor",
                        NanBox::handle(ctor.to_raw()),
                    );
                    let nm = self.new_str(name);
                    self.realm.set_property(proto, "name", nm);
                    self.realm.mark_hidden(proto, "name");
                    let msg = self.new_str("");
                    self.realm.set_property(proto, "message", msg);
                    self.realm.mark_hidden(proto, "message");
                    self.realm
                        .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
                    self.realm.mark_hidden(ctor, "prototype");
                    self.realm.set_readonly_property(ctor, "prototype");
                    self.realm.set_non_configurable_property(ctor, "prototype");
                    // `AggregateError(errors, message)` has `length` 2; the other
                    // error constructors take just `message` (`length` 1).
                    if name == "AggregateError" {
                        self.install_fn_name_length(ctor, "AggregateError", 2);
                    }
                    // `Object.getPrototypeOf(TypeError) === Error` (the subclass
                    // constructor inherits `Error`'s static side).
                    if let Some(error_ctor) = self
                        .current
                        .get("Error")
                        .and_then(|v| v.as_handle())
                        .map(Handle::from_raw)
                    {
                        self.realm.set_native_proto(ctor, error_ctor);
                    }
                }
            }
        }
        // The shared abstract `%TypedArray%` intrinsic constructor and the
        // constructor-side hierarchy that hangs the concrete TA constructors off
        // it (so `Object.getPrototypeOf(Int8Array) === %TypedArray%`).
        self.setup_typed_array_intrinsic(obj_proto);
        // The ES2025 `uint8array-base64` proposal: six `Uint8Array`-specific
        // methods (not on `%TypedArray%.prototype`).
        self.install_uint8array_base64();
        // Static methods that are otherwise call-only (readable for feature detection).
        self.setup_static_methods(
            "Promise",
            &[
                "resolve",
                "reject",
                "all",
                "race",
                "allSettled",
                "any",
                "allKeyed",
                "allSettledKeyed",
                "withResolvers",
                "try",
            ],
        );
        self.setup_static_methods("Map", &["groupBy"]);
        self.setup_static_methods(
            "Number",
            &[
                "isInteger",
                "isFinite",
                "isNaN",
                "isSafeInteger",
                "parseFloat",
                "parseInt",
            ],
        );
        // `Number.parseFloat`/`Number.parseInt` are the *same* function objects
        // as the global `parseFloat`/`parseInt` (ECMA-262 21.1.2.12/13), so
        // `Number.parseFloat === parseFloat`. Overwrite the freshly-installed
        // static-method wrappers with the shared global handles.
        if let Some(num) = self
            .current
            .get("Number")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            for name in ["parseFloat", "parseInt"] {
                if let Some(g) = self.current.get(name) {
                    self.realm.set_property(num, name, g);
                    self.realm.mark_hidden(num, name);
                }
            }
        }
        self.setup_static_methods("String", &["fromCharCode", "fromCodePoint", "raw"]);
        self.setup_static_methods("Symbol", &["for", "keyFor"]);
        self.setup_static_methods("Date", &["now", "parse", "UTC"]);
        self.setup_static_methods("BigInt", &["asIntN", "asUintN"]);
        // `Proxy.revocable` is a readable own function property (name "revocable",
        // length 2, non-constructor) that routes back through `call_method`.
        self.setup_static_methods("Proxy", &["revocable"]);
        // `Object`/`Array`/`Reflect` are modeled as namespace objects (their call
        // behavior is special-cased) rather than native-function cells, so they
        // miss the function `name`/`length` synthesis. Install those own data
        // properties explicitly with the built-in attributes
        // `{ writable: false, enumerable: false, configurable: true }` so
        // `verifyProperty` on e.g. `Object.length`/`Array.name` matches the spec.
        // (`Object.name` was already installed above.) `Object.length === 1`,
        // `Array.length === 1`.
        for (ctor, ctor_name) in [("Array", "Array"), ("Reflect", "Reflect")] {
            if let Some(h) = self.current.get(ctor).and_then(NanBox::as_handle) {
                let h = Handle::from_raw(h);
                let nv = self.new_str(ctor_name);
                self.realm.set_hidden_property(h, "name", nv);
                self.realm.set_readonly_property(h, "name");
            }
        }
        // The callable namespace constructors `Object` and `Array` declare a
        // single parameter; `Reflect` is a non-callable namespace (no `length`).
        for ctor in ["Object", "Array"] {
            if let Some(h) = self.current.get(ctor).and_then(NanBox::as_handle) {
                let h = Handle::from_raw(h);
                self.realm
                    .set_hidden_property(h, "length", NanBox::number(1.0));
                self.realm.set_readonly_property(h, "length");
            }
        }
        // Newly-created plain objects now inherit from `Object.prototype`.
        self.realm.set_default_object_proto(obj_proto);
        // `%IteratorPrototype%` was created before `Object.prototype` existed, so
        // its `[[Prototype]]` was left null. Per spec it is `%Object.prototype%`
        // (25.1.2.1) — link it now so every iterator (array/string/map/set/regexp/
        // generator, and any object inheriting `%IteratorPrototype%`) inherits
        // `toString`/`valueOf`. Without this, ToPrimitive on an iterator (e.g. a
        // generator object used as a computed property key) throws "Cannot convert
        // object to primitive value" instead of yielding "[object <Tag>]".
        if let Some(iter_proto) = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            && self.realm.object_proto(iter_proto).is_none()
        {
            self.realm.set_object_proto(iter_proto, Some(obj_proto));
        }
        // Build the real `.prototype` objects (with branded methods/accessors,
        // `constructor`, and `[Symbol.toStringTag]`) for every `Intl` service
        // constructor — `Object.prototype` now exists, so the prototypes inherit it.
        self.install_intl_prototypes();
        // The ES2025 explicit-resource-management classes (`DisposableStack`,
        // `AsyncDisposableStack`) and the `ShadowRealm` constructor — each a real
        // branded constructor with a `.prototype` (so they appear on `globalThis`
        // and inherit `Object.prototype`).
        self.install_resource_management();
        // `globalThis`: an object mirroring the global bindings, referencing
        // itself. Reads like `globalThis.Math` and `globalThis.globalThis` work.
        // Every standard global (constructors, namespaces, functions) is a
        // *non-enumerable*, writable, configurable own property of the global
        // object (per spec — `Object.getOwnPropertyDescriptor(globalThis, "Array")`
        // is `{ writable, configurable, enumerable: false }`), so the mirror is
        // built from the live root-scope bindings and each is marked hidden.
        let global = self.realm.new_object();
        let bindings = self.global_scope.local_bindings();
        for (name, value, _is_const) in &bindings {
            self.realm.set_property(global, name, *value);
            self.realm.mark_hidden(global, name);
        }
        // `NaN`, `Infinity`, `undefined` are `{ writable: false, enumerable: false,
        // configurable: false }` value properties of the global object.
        for (name, value) in [
            ("NaN", NanBox::number(f64::NAN)),
            ("Infinity", NanBox::number(f64::INFINITY)),
            ("undefined", NanBox::undefined()),
        ] {
            self.realm.set_property(global, name, value);
            self.realm.mark_hidden(global, name);
            self.realm.set_readonly_property(global, name);
            self.realm.set_non_configurable_property(global, name);
        }
        let gbox = NanBox::handle(global.to_raw());
        self.realm.set_property(global, "globalThis", gbox);
        self.realm.mark_hidden(global, "globalThis");
        self.current.declare("globalThis", gbox);
        self.global_this = gbox;
    }

    /// The global object (`globalThis`) handle, if the realm is initialized.
    pub(crate) fn global_object(&self) -> Option<Handle> {
        self.global_this.as_handle().map(Handle::from_raw)
    }

    /// The underlying realm (e.g. to render a result with `to_display_string`).
    #[must_use]
    pub fn realm(&self) -> &Realm {
        &self.realm
    }

    /// The underlying realm, mutably (e.g. for an embedder to read/write a byte
    /// store via `bytes_at`/`bytes_at_mut` or build views with `new_typed_array`).
    pub fn realm_mut(&mut self) -> &mut Realm {
        &mut self.realm
    }

    /// Binds `value` as a global named `name` — declaring it in the global scope
    /// and installing it on `globalThis` — so subsequently-`run` script can read
    /// it (e.g. an embedder-built `ArrayBuffer`). Existing bindings of the same
    /// name are shadowed.
    pub fn declare_global(&mut self, name: &str, value: NanBox) {
        self.current.declare(name, value);
        if let Some(g) = self.global_this.as_handle().map(Handle::from_raw) {
            self.realm.set_property(g, name, value);
        }
    }

    /// Registers a Rust closure as a **host native function** and returns it as a
    /// callable JS value (`ROADMAP.md` §4.0 — the embedding foundation).
    ///
    /// The returned value is a first-class function: it can be bound to a global
    /// (see [`register_global_fn`](Self::register_global_fn)), installed as an
    /// object property, passed to JS, or called via `.call`/`.apply`. Reading its
    /// `name` yields `name`, its `length` yields `length` (the declared arity),
    /// and `typeof` it is `"function"`.
    ///
    /// The closure receives a [`Ctx`] (to build values, read/write properties,
    /// throw, and re-enter JS), the call's `this`, and the arguments. Returning
    /// `Ok(v)` resolves the call to `v`; returning `Err(v)` raises `v` as a JS
    /// exception the script can `catch`. A host function is **not re-entrant onto
    /// itself**: if it (directly or transitively) calls back into the *same*
    /// registered function while a call is in flight, that inner call throws a
    /// `TypeError` rather than aliasing the `FnMut` — distinct host functions may
    /// freely call one another.
    ///
    /// Note (§6 "two engines, one truth"): the bytecode VM (`nbvm`) has no host
    /// registry of its own; a program that calls a host function faults to this
    /// interpreter, exactly as it does for any native.
    pub fn register_fn<F>(&mut self, name: &str, length: u32, f: F) -> NanBox
    where
        F: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
    {
        self.register_host_fn(name, length, false, f)
    }

    /// Registers a host **constructor** — like [`register_fn`](Self::register_fn),
    /// but `new f(...args)` is allowed: the closure runs with a fresh object as
    /// `this` (its `[[Prototype]]` is `f.prototype`) and, per the constructor
    /// return rule, the result is that object unless the closure returns another
    /// object. The function object gets a writable own `prototype` whose
    /// `constructor` points back at it, so `instanceof` and inherited methods work.
    /// A plain call `f(...)` (no `new`) still runs the closure normally.
    pub fn register_constructor<F>(&mut self, name: &str, length: u32, f: F) -> NanBox
    where
        F: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
    {
        let ctor = self.register_host_fn(name, length, true, f);
        // Give it a `prototype` object with a back-reference `constructor`.
        if let Some(ch) = ctor.as_handle().map(Handle::from_raw) {
            let proto = self.realm.new_object();
            self.realm.set_hidden_property(proto, "constructor", ctor);
            self.realm
                .set_property(ch, "prototype", NanBox::handle(proto.to_raw()));
        }
        ctor
    }

    /// [`register_constructor`](Self::register_constructor) plus
    /// [`declare_global`](Self::declare_global): registers the host constructor and
    /// binds it as a global named `name`.
    pub fn register_global_constructor<F>(&mut self, name: &str, length: u32, f: F) -> NanBox
    where
        F: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
    {
        let v = self.register_constructor(name, length, f);
        self.declare_global(name, v);
        v
    }

    fn register_host_fn<F>(&mut self, name: &str, length: u32, is_constructor: bool, f: F) -> NanBox
    where
        F: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
    {
        let id = self.host_fns.len() as u32;
        self.host_fns.push(Some(HostFn {
            name: String::from(name),
            length,
            call: alloc::boxed::Box::new(f),
            is_constructor,
        }));
        let h = self.realm.new_host_fn(id);
        NanBox::handle(h.to_raw())
    }

    /// Reads a persistent handle pinned by [`Ctx::persist`] (or [`persist`](Self::persist)),
    /// reflecting any relocation the collector has applied. `None` if released.
    #[must_use]
    pub fn persistent(&self, idx: u32) -> Option<NanBox> {
        self.realm.persistent(idx)
    }

    /// Pins `value` as a persistent handle from outside a host call (host-side
    /// setup), returning its index. See [`Ctx::persist`].
    pub fn persist(&mut self, value: NanBox) -> u32 {
        self.realm.persist(value)
    }

    /// Releases a persistent handle so its value is no longer a GC root.
    pub fn release_persistent(&mut self, idx: u32) {
        self.realm.release_persistent(idx);
    }

    /// Resolves a deferred promise (from [`Ctx::deferred`]) with `value`, then
    /// drains the microtask queue so its `then` reactions run. Releases the token;
    /// an unknown/already-settled token is a no-op.
    ///
    /// # Errors
    /// A thrown value from a reaction that escapes to the top level.
    pub fn resolve_deferred(&mut self, token: u32, value: NanBox) -> Result<(), NanBox> {
        self.settle_deferred(token, value, true)
    }

    /// Rejects a deferred promise (from [`Ctx::deferred`]) with `reason`, then
    /// drains the microtask queue. Releases the token.
    ///
    /// # Errors
    /// A thrown value from a reaction that escapes to the top level.
    pub fn reject_deferred(&mut self, token: u32, reason: NanBox) -> Result<(), NanBox> {
        self.settle_deferred(token, reason, false)
    }

    fn settle_deferred(&mut self, token: u32, value: NanBox, fulfill: bool) -> Result<(), NanBox> {
        let Some(cap) = self.realm.persistent(token) else {
            return Ok(());
        };
        // The token pins a `[resolve, reject]` array; pick the matching function.
        let f = cap
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
            .and_then(|e| e.get(usize::from(!fulfill)).copied());
        self.realm.release_persistent(token);
        if let Some(f) = f {
            self.call_with_this(f, NanBox::undefined(), &[value])
                .map_err(|e| self.exec_error_value(e))?;
        }
        self.drain_microtasks()
            .map_err(|e| self.exec_error_value(e))
    }

    /// Whether the host function with registry index `id` was registered as a
    /// constructor (`register_constructor`).
    pub(crate) fn host_fn_is_constructor(&self, id: u32) -> bool {
        self.host_fns
            .get(id as usize)
            .and_then(|s| s.as_ref())
            .is_some_and(|hf| hf.is_constructor)
    }

    /// [`register_fn`](Self::register_fn) plus [`declare_global`](Self::declare_global):
    /// registers the closure and binds it as a global named `name`, so a
    /// subsequently-`run` script can call it directly.
    pub fn register_global_fn<F>(&mut self, name: &str, length: u32, f: F) -> NanBox
    where
        F: FnMut(&mut Ctx<'_, '_>, NanBox, &[NanBox]) -> Result<NanBox, NanBox> + 'static,
    {
        let v = self.register_fn(name, length, f);
        self.declare_global(name, v);
        v
    }

    /// Invokes the host function with registry index `id` (a
    /// [`Cell::HostFn`](crate::cell::Cell::HostFn)), passing `this` and `args`.
    ///
    /// The closure is *taken out* of its slot for the duration of the call so a
    /// re-entrant call onto the same function is a clean `TypeError` (see
    /// [`register_fn`](Self::register_fn)) rather than a double `&mut` borrow, and
    /// restored when the call returns (including on a thrown error).
    pub(crate) fn call_host_fn(
        &mut self,
        id: u32,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let taken = self.host_fns.get_mut(id as usize).and_then(Option::take);
        let Some(mut hf) = taken else {
            // Either an unknown id (should not happen for a live cell) or a
            // re-entrant call onto a host function already on the stack.
            return Err(self.type_error("host function is not re-entrant"));
        };
        // Trap a panic from host code at the boundary so it becomes a catchable JS
        // `Error` instead of unwinding across engine frames (and leaving the
        // registry slot empty). `AssertUnwindSafe`: the `&mut` state is ours and we
        // restore invariants (the slot) before re-entering the engine. The `ctx`
        // (which holds the `&mut self` borrow) is scoped to this block so the borrow
        // is released before the slot is restored below.
        let caught = {
            let mut ctx = Ctx { interp: self };
            #[cfg(feature = "std")]
            {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (hf.call)(&mut ctx, this, args)
                }))
            }
            #[cfg(not(feature = "std"))]
            {
                Ok::<Result<NanBox, NanBox>, ()>((hf.call)(&mut ctx, this, args))
            }
        };
        // The `&mut self` borrow is free; restore the closure to its slot.
        self.host_fns[id as usize] = Some(hf);
        match caught {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(v)) => Err(ExecError::Throw(v)),
            Err(_panic) => {
                let m = self.new_str("host function panicked");
                Err(ExecError::Throw(self.make_error(N_ERROR_BASE, Some(m))))
            }
        }
    }

    /// The declared `(name, length)` of the host function at registry index `id`,
    /// used to synthesize its `name`/`length` own properties on read. Available
    /// even while the closure is taken out for an in-flight call.
    pub(crate) fn host_fn_meta(&self, id: u32) -> Option<(&str, u32)> {
        self.host_fns
            .get(id as usize)
            .and_then(Option::as_ref)
            .map(|hf| (hf.name.as_str(), hf.length))
    }

    /// Converts an [`ExecError`] into the JS value it represents, for the host
    /// boundary (which speaks `Result<NanBox, NanBox>`): a `Throw` carries its
    /// value directly; the internal variants are materialized as the error object
    /// a script would observe.
    pub(crate) fn exec_error_value(&mut self, e: ExecError) -> NanBox {
        match e {
            ExecError::Throw(v) => v,
            ExecError::NotCallable => {
                let m = self.new_str("is not a function");
                self.make_error(N_TYPE_ERROR, Some(m))
            }
            ExecError::NotDefined(name) => {
                let m = self.new_str(&alloc::format!("{name} is not defined"));
                self.make_error(N_REFERENCE_ERROR, Some(m))
            }
            ExecError::Unsupported(s) => {
                let m = self.new_str(s);
                self.make_error(N_ERROR_BASE, Some(m))
            }
            ExecError::OptShortCircuit => NanBox::undefined(),
            // A tail call is always consumed by the enclosing `invoke` trampoline,
            // so it never reaches the host boundary; defensively perform it here.
            ExecError::TailCall {
                callee,
                this_val,
                args,
            } => self
                .call_with_this(callee, this_val, &args)
                .unwrap_or(NanBox::undefined()),
        }
    }

    /// Sloppy-mode assignment to an unresolvable reference (`x = 1` with no
    /// binding for `x`): creates a property on the *global* object and a binding
    /// in the global scope — never in the current (function/block) scope, so the
    /// new global outlives the enclosing frame. (Strict mode throws instead.)
    pub(crate) fn declare_sloppy_global(&mut self, name: &str, value: NanBox) {
        // A sloppy assignment to an *unresolvable* reference (`x = 1` with no `x`
        // binding) creates an ordinary, **configurable** data property on the
        // global object — *not* a binding in the global environment record's
        // declarative part. This matters for `delete x`: a property created this
        // way is deletable (returns `true`, then the name is unresolvable again),
        // whereas a declared `var`/`let`/`function` binding is not. Reads resolve
        // via `read_ident_ref`'s global-object fallback, so no declarative binding
        // is needed.
        if let Some(g) = self.global_this.as_handle().map(Handle::from_raw) {
            self.realm.set_property(g, name, value);
        } else {
            // No global object (an unusual embedding): fall back to a declarative
            // binding so the write is not simply lost.
            self.global_scope.declare(name, value);
        }
    }

    /// Mirrors a global `var`/function declaration's binding onto the global
    /// object, so `var x = 1; this.x` (and `globalThis.x`) see it. Only applies
    /// when execution is running directly in the global scope (a `var` inside a
    /// function binds in that function, not on the global object). Per spec a
    /// global `var` property is writable + enumerable but non-configurable.
    fn publish_global_var(&mut self, name: &str, value: NanBox) {
        if !self.var_scope.ptr_eq(&self.global_scope) {
            return;
        }
        if let Some(g) = self.global_this.as_handle().map(Handle::from_raw) {
            self.realm.set_property(g, name, value);
        }
    }

    /// Captures the object graph reachable from `roots` (the heap objects among
    /// them — primitives are skipped) and serializes it to portable bytes: a D′
    /// snapshot of live values that can later be reloaded into a fresh interpreter
    /// holding the same code (see [`restore_snapshot`](Self::restore_snapshot)).
    #[must_use]
    pub fn snapshot(&self, roots: &[NanBox]) -> Vec<u8> {
        let handles: Vec<Handle> = roots
            .iter()
            .filter_map(|v| v.as_handle().map(Handle::from_raw))
            .collect();
        crate::snapshot::serialize(&crate::snapshot::capture(&self.realm, &handles))
    }

    /// Reloads a snapshot produced by [`snapshot`](Self::snapshot) into this
    /// interpreter, returning the restored root values in the order their (heap)
    /// roots were captured. The restored objects are live — a restored closure runs
    /// and carries its snapshotted captured state.
    ///
    /// # Errors
    /// [`SnapError`](crate::snapshot::SnapError) if `bytes` is not a valid snapshot.
    pub fn restore_snapshot(
        &mut self,
        bytes: &[u8],
    ) -> Result<Vec<NanBox>, crate::snapshot::SnapError> {
        let snap = crate::snapshot::deserialize(bytes)?;
        let handles = crate::snapshot::restore(&mut self.realm, &snap);
        Ok(handles
            .into_iter()
            .map(|h| NanBox::handle(h.to_raw()))
            .collect())
    }

    /// Runs a whole program, returning the value of its last expression
    /// statement (or `undefined`).
    pub fn run(&mut self, program: &'a Program) -> Result<NanBox, ExecError> {
        // Retain the program's source so function/class definitions can slice
        // their literal text for `Function.prototype.toString` (AST spans are byte
        // offsets into this source).
        self.src = &program.source;
        self.strict = self.strict || has_use_strict(&program.body);
        // Script-level `this` is the global object (the realm's `globalThis`),
        // regardless of strictness — so a top-level `this.x = …` (sloppy globals)
        // and `this === globalThis` behave per spec.
        if matches!(self.this_val.unpack(), Unpacked::Undefined) {
            self.this_val = self.global_this;
        }
        self.hoist_with(&program.body, true)?;
        let mut last = NanBox::undefined();
        for stmt in &program.body {
            match self.exec(stmt)? {
                // UpdateEmpty: an empty completion (declaration / empty statement)
                // never replaces a preceding non-empty value; the script's value
                // is the last non-empty statement value (undefined if none).
                Flow::Normal(v) => {
                    if !v.is_empty_completion() {
                        last = v;
                    }
                }
                Flow::Return(v) => {
                    self.run_event_loop()?;
                    return Ok(v);
                }
                Flow::Break(..) | Flow::Continue(..) => {}
            }
        }
        // Run the event loop (microtasks + `setTimeout`) before returning.
        self.run_event_loop()?;
        Ok(last)
    }

    // --- dynamic code (`eval` / `Function`) ---

    /// Parses `source` as a Script and returns a `&'a` reference to the resulting
    /// `Program`. A parse failure throws a `SyntaxError` (catchable). The parsed
    /// program is owned by the interpreter for the rest of the run: it is boxed,
    /// leaked once to a `&'static Program` (which coerces to `&'a`), and cached by
    /// source so repeated `eval`/`Function` of the same string parse only once.
    #[allow(clippy::too_many_arguments)]
    fn parse_eval_program(
        &mut self,
        source: &str,
        allow_super_property: bool,
        allow_super_call: bool,
        allow_new_target: bool,
        inherited_strict: bool,
        // Private names visible at the (direct-)eval call site — seed the
        // validator's private scope so the eval body may reference the enclosing
        // class's `#names`. Empty for an indirect eval or outside any class.
        outer_private_names: &[alloc::boxed::Box<str>],
        // Whether the direct eval runs inside a class field initializer / static
        // block (activates the ContainsArguments early error for `arguments`).
        in_field_initializer: bool,
    ) -> Result<&'a Program, ExecError> {
        // The cache is keyed by source *and* the inherited `super`/`new.target`
        // context *and* inherited strictness *and* the visible private names /
        // field-initializer flag: the same text `"super.x"` / `"new.target"` /
        // `"public = 1"` / `"this.#x"` / `"arguments"` is a SyntaxError in one
        // caller and valid in another, so they must not share a cached AST. A
        // flag prefix (outside the JS source grammar) keeps it unambiguous.
        let key = alloc::format!(
            "{}{}{}{}{}{}\0{source}",
            u8::from(allow_super_property),
            u8::from(allow_super_call),
            u8::from(allow_new_target),
            u8::from(inherited_strict),
            u8::from(in_field_initializer),
            outer_private_names.join(","),
        );
        if let Some(p) = self.eval_programs.get(&key) {
            return Ok(p);
        }
        match crate::parser::Parser::parse_eval_program(
            source,
            allow_super_property,
            allow_super_call,
            allow_new_target,
            inherited_strict,
            outer_private_names,
            in_field_initializer,
        ) {
            Ok(program) => {
                // The AST is fully owned (no borrow of `source`); leaking the box
                // yields a `'static` reference that coerces to `'a`.
                let leaked: &'static Program =
                    alloc::boxed::Box::leak(alloc::boxed::Box::new(program));
                self.eval_programs.insert(key, leaked);
                Ok(leaked)
            }
            Err(e) => {
                let m = self.new_str(&alloc::format!("{e}"));
                Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))))
            }
        }
    }

    /// Executes a parsed eval `program`'s statements in the current scope,
    /// returning the completion value (the value of the last value-producing
    /// statement, else `undefined`). Strict-mode and scope setup are the caller's
    /// responsibility; this is the shared statement loop. Unlike `run`, it does
    /// NOT drain the event loop — eval runs synchronously within the surrounding
    /// execution, which drains microtasks at its own top level.
    fn run_eval_body(&mut self, program: &'a Program) -> Result<NanBox, ExecError> {
        // The eval body's function/class spans index into the eval program's own
        // (leaked) source; retain it for `Function.prototype.toString` while its
        // statements run, then restore the enclosing source so a subsequent
        // definition in the surrounding code slices the right text.
        let saved_src = core::mem::replace(&mut self.src, &program.source);
        let result = self.run_eval_body_inner(program);
        self.src = saved_src;
        result
    }

    fn run_eval_body_inner(&mut self, program: &'a Program) -> Result<NanBox, ExecError> {
        self.hoist_with_kind(&program.body, true, true)?;
        let mut last = NanBox::undefined();
        for stmt in &program.body {
            match self.exec(stmt)? {
                // UpdateEmpty + the `eval` rule that a trailing empty completion
                // becomes `undefined`: track the last *non-empty* value (`last`
                // starts at undefined, so an all-empty body yields undefined).
                Flow::Normal(v) => {
                    if !v.is_empty_completion() {
                        last = v;
                    }
                }
                // A `return` is a SyntaxError at parse time at the top level, so
                // it cannot reach here; `break`/`continue` likewise. Treat any
                // such residue as completing normally.
                Flow::Return(v) => return Ok(v),
                Flow::Break(..) | Flow::Continue(..) => {}
            }
        }
        Ok(last)
    }

    /// CanDeclareGlobalVar: a global `var` binding may be created for `name`
    /// unless the global object is non-extensible and does not already have it as
    /// an own property.
    fn can_declare_global_var(&self, g: Handle, name: &str) -> bool {
        self.realm.has_own(g, name) || self.realm.is_extensible(g)
    }

    /// CanDeclareGlobalFunction: a global `function` binding may be created for
    /// `name` when there is no colliding own property (subject to extensibility),
    /// or the existing own property is configurable, or it is a writable +
    /// enumerable data property. A non-configurable accessor / non-writable /
    /// non-enumerable property (e.g. `NaN`, `undefined`, `Infinity`) blocks it.
    fn can_declare_global_function(&self, g: Handle, name: &str) -> bool {
        if !self.realm.has_own(g, name) {
            return self.realm.is_extensible(g);
        }
        if !self.realm.property_is_non_configurable(g, name) {
            return true;
        }
        self.realm.accessor(g, name).is_none()
            && !self.realm.property_is_readonly(g, name)
            && self.realm.property_is_enumerable(g, name)
    }

    /// EvalDeclarationInstantiation static validation for a *sloppy* eval (the
    /// spec steps that throw before any binding is instantiated). `lex_env` is the
    /// caller's lexical environment (the eval's own fresh lexEnv is a child of
    /// it); `var_env` is the variable environment the eval hoists `var`/function
    /// declarations into.
    ///
    /// Throws:
    /// - **SyntaxError** when a `var`/function name collides with a lexical
    ///   (`let`/`const`/`class`) binding in a scope between `lex_env` and
    ///   `var_env`, or with a *global* lexical binding when `var_env` is the
    ///   global environment (a global lexical lives in the global scope frame but,
    ///   unlike a `var`/function, is not an own property of the global object).
    /// - **TypeError** when, at the global variable environment, a `var` fails
    ///   CanDeclareGlobalVar (non-extensible global) or a function fails
    ///   CanDeclareGlobalFunction.
    fn eval_declaration_checks(
        &mut self,
        program: &Program,
        lex_env: &Scope,
        var_env: &Scope,
    ) -> Result<(), ExecError> {
        let mut var_names: Vec<&str> = Vec::new();
        collect_var_names(&program.body, &mut var_names);
        let mut fn_names: Vec<&str> = Vec::new();
        for stmt in &program.body {
            let stmt = unwrap_exported_function(stmt);
            if let Stmt::Function(func) = stmt
                && let Some(id) = &func.id
            {
                fn_names.push(&id.name);
            }
        }
        // Block-level function names that Annex B var-hoists also participate in
        // VarDeclaredNames for the lexical-collision walks.
        let mut block_fns: Vec<&str> = Vec::new();
        collect_block_function_names(&program.body, &mut block_fns);

        let all_var_names: Vec<&str> = var_names
            .iter()
            .chain(fn_names.iter())
            .chain(block_fns.iter())
            .copied()
            .collect();

        // Walk lexical environments from the caller's lexical env up to (but not
        // including) the variable environment: a `var` may not hoist over a
        // like-named lexical binding in a lower scope.
        let mut lex = Some(lex_env.clone());
        while let Some(env) = lex {
            if env.ptr_eq(var_env) {
                break;
            }
            // A `catch (param)` frame is exempt: Annex B.3.4 permits a sloppy
            // `var`/function to redeclare a catch parameter.
            if !env.is_catch() {
                for name in &all_var_names {
                    if env.has_local(name) {
                        let m = self.new_str(&alloc::format!(
                            "Identifier '{name}' has already been declared"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
                    }
                }
            }
            lex = env.parent();
        }

        let at_global = var_env.ptr_eq(&self.global_scope);
        if !at_global {
            return Ok(());
        }
        let Some(g) = self.global_this.as_handle().map(Handle::from_raw) else {
            return Ok(());
        };

        // A `var` may not collide with a global lexical declaration.
        for name in &all_var_names {
            if var_env.has_local(name) && !self.realm.has_own(g, name) {
                let m = self.new_str(&alloc::format!(
                    "Identifier '{name}' has already been declared"
                ));
                return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
            }
        }
        // CanDeclareGlobalFunction for each function name, then CanDeclareGlobalVar
        // for each pure `var` name — all validated before any is created.
        for name in fn_names.iter().chain(block_fns.iter()) {
            if !self.can_declare_global_function(g, name) {
                let m = self.new_str(&alloc::format!("Cannot declare global function '{name}'"));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        for name in &var_names {
            if !self.can_declare_global_var(g, name) {
                let m = self.new_str(&alloc::format!("Cannot declare global variable '{name}'"));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        Ok(())
    }

    /// The `eval(source)` operation. `direct` is true for a direct eval call
    /// (`eval(s)` by that exact name), false for an indirect one (`(0,eval)(s)`,
    /// `var e = eval; e(s)`, `globalThis.eval(s)`).
    ///
    /// Scoping:
    /// - **Indirect** eval runs in a fresh child of the GLOBAL scope, sloppy
    ///   unless the eval code self-declares `"use strict"`.
    /// - **Direct sloppy** eval (caller not strict and code not strict) runs in
    ///   the CURRENT scope, so its `var`/function declarations hoist into the
    ///   surrounding variable environment and it can read/modify locals.
    /// - **Direct strict** eval (caller strict OR code `"use strict"`) gets its
    ///   own child scope for its lexical + var declarations, but still reads the
    ///   surrounding scope.
    fn eval_string(&mut self, source: &str, direct: bool) -> Result<NanBox, ExecError> {
        // A direct eval inherits the caller's `super` context: `super.prop` is
        // legal in the eval code when the calling code has a home object (a
        // method / accessor / constructor / class field initializer / static
        // block). An indirect eval always runs in the global scope, where
        // `super` is never permitted. (`super(…)` needs a derived-constructor
        // context that is not tracked separately, so it stays disallowed.)
        let allow_super_property =
            direct && (self.current_home.is_some() || self.current_home_object.is_some());
        // `new.target` is syntactically valid in a *direct* eval that is contained
        // in function code (the eval inherits the caller's `[[NewTarget]]`).
        // `new_target_in_scope` tracks this lexically — true inside a non-arrow
        // function/constructor/field-initializer/static-block, transparently
        // inherited by arrows — so a direct eval inside a top-level arrow (no
        // `new.target` in scope) and any indirect eval keep `new.target` disallowed.
        let allow_new_target = direct && self.new_target_in_scope;
        // A direct eval inside strict code is strict even without its own
        // directive; an indirect eval starts sloppy (its strictness comes only
        // from a `"use strict"` in the code itself).
        let inherited_strict = direct && self.strict;
        // Private names visible at a *direct* eval call site: the union of all
        // `#names` declared by the lexically-enclosing class chain. Seeding the
        // parser with these lets the eval body reference `this.#x` (which resolves
        // at runtime against the unchanged `current_lexical_home`). An indirect
        // eval runs in the global scope and sees none.
        let outer_private_names = if direct {
            self.visible_private_names()
        } else {
            Vec::new()
        };
        // A direct eval inside a class field initializer / static block inherits
        // the ContainsArguments early error (an `arguments` reference in the eval
        // body is a SyntaxError).
        let in_field_initializer = direct && self.in_field_initializer;
        let program = self.parse_eval_program(
            source,
            allow_super_property,
            false,
            allow_new_target,
            inherited_strict,
            &outer_private_names,
            in_field_initializer,
        )?;
        let code_strict = has_use_strict(&program.body);

        // EvalDeclarationInstantiation early error: a *sloppy direct* eval being
        // run while a parameter default is evaluated may not introduce a `var`
        // (or function) binding that collides with a formal parameter name or
        // `arguments` — the function's separate parameter environment already
        // binds those (`function f(a = eval("var a")) {}` is a SyntaxError, and
        // the body must not run). Strict eval gets its own var scope, so it is
        // exempt; an indirect eval never runs in the parameter scope.
        if direct
            && !self.strict
            && !code_strict
            && let Some(param_names) = self.eval_param_names.take()
        {
            let mut var_names: Vec<&str> = Vec::new();
            collect_var_names(&program.body, &mut var_names);
            let mut block_fns: Vec<&str> = Vec::new();
            collect_block_function_names(&program.body, &mut block_fns);
            if var_names
                .iter()
                .chain(block_fns.iter())
                .any(|n| param_names.iter().any(|p| p == n))
            {
                let m =
                    self.new_str("Identifier declared by `var` in eval conflicts with a parameter");
                return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
            }
            // Not a conflict: keep the parameter set live for any further nested
            // direct eval within the same parameter default.
            self.eval_param_names = Some(param_names);
        }

        // Recursion guard shared with the tree-walk budget.
        if self.eval_depth >= self.realm.limits.max_eval_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(
                self.make_error(N_ERROR_BASE + 2, Some(msg)),
            ));
        }

        let saved_strict = self.strict;
        let saved_scope = self.current.clone();
        let saved_var_scope = self.var_scope.clone();
        let saved_annexb = core::mem::take(&mut self.annexb_block_fns);
        let (saved_this, saved_new_target) = (self.this_val, self.new_target);

        // Final strictness of the eval code: a direct eval inherits the caller's,
        // and either kind may add its own `"use strict"`.
        let eval_strict = if direct {
            saved_strict || code_strict
        } else {
            code_strict
        };
        // The variable environment the eval hoists `var`/function declarations
        // into (spec `varEnv`): the caller's for a direct eval, the realm's global
        // env for an indirect one.
        let var_env = if direct {
            saved_var_scope.clone()
        } else {
            self.global_scope.clone()
        };

        // EvalDeclarationInstantiation validation for a *sloppy* eval — runs
        // BEFORE any binding is created, so a rejected declaration leaves the
        // surrounding environment untouched (no partial hoisting). A strict eval
        // declares into its own fresh var env, so these collisions can't arise.
        if !eval_strict {
            // The lexical-collision walk starts at the eval's lexical env: the
            // caller's scope for a direct eval, the global env for an indirect one
            // (an indirect eval never sees the caller's scope chain).
            let lex_env = if direct {
                saved_scope.clone()
            } else {
                self.global_scope.clone()
            };
            self.eval_declaration_checks(program, &lex_env, &var_env)?;
        }

        // Both kinds get a fresh declarative *lexical* environment (child of the
        // caller's env for direct, of the global env for indirect) so their
        // `let`/`const`/`class` declarations never leak into the surrounding
        // scope. A sloppy eval additionally hoists its `var`/function declarations
        // OUT into `var_env` (recorded in `eval_var_scope`, consumed by the body's
        // hoisting pass); a strict eval keeps them in this fresh env.
        self.strict = eval_strict;
        if direct {
            self.current = saved_scope.child();
            // `this`/`new.target` are inherited from the caller (unchanged).
        } else {
            self.current = self.global_scope.child();
            self.this_val = self.global_this;
            self.new_target = NanBox::undefined();
        }
        if !eval_strict {
            self.eval_var_scope = Some(var_env);
        }

        self.eval_depth += 1;
        let result = self.run_eval_body(program);
        self.eval_depth -= 1;

        self.current = saved_scope;
        self.var_scope = saved_var_scope;
        self.eval_var_scope = None;
        self.annexb_block_fns = saved_annexb;
        self.strict = saved_strict;
        self.this_val = saved_this;
        self.new_target = saved_new_target;
        result
    }

    /// The dynamic `Function(p1, p2, …, body)` / `new Function(…)` constructor.
    /// The trailing argument is the function body; all preceding arguments form
    /// the (comma-joined) formal parameter list. The pieces are assembled into
    /// `(function anonymous(<params>\n) {\n<body>\n})`, parsed, and the resulting
    /// function object is returned. The function is created in the GLOBAL scope
    /// (so it closes over globals only), sloppy unless the body self-declares
    /// `"use strict"`. A parse failure (bad params or body) throws a SyntaxError.
    fn build_function_constructor(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        self.build_function_constructor_kw(args, "function", new_target, callee)
    }

    /// As `build_function_constructor`, but with an explicit function keyword so the
    /// `GeneratorFunction` ("function*") and `AsyncGeneratorFunction`
    /// ("async function*") intrinsics can reuse the same dynamic-source machinery.
    ///
    /// `new_target` / `callee` drive the `GetPrototypeFromConstructor` /
    /// `GetFunctionRealm` half: a cross-realm `new other.Function()` (or
    /// `Reflect.construct(Function, …, crossRealmTarget)`) links the built function
    /// to the appropriate realm's `%Function.prototype%` and tags it with that
    /// realm. Both are `undefined` for a plain-call `Function(…)` (no realm
    /// remapping — the current realm's default applies).
    fn build_function_constructor_kw(
        &mut self,
        args: &[NanBox],
        keyword: &str,
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        // Coerce arguments to strings (ToString). Last is the body; the rest are
        // the parameter-list pieces, joined with commas.
        let (params, body) = match args.split_last() {
            Some((last, rest)) => {
                // `ToString` each argument in order (spec CreateDynamicFunction):
                // a custom `toString`/`valueOf` runs and a thrown value propagates
                // — `new Function({toString(){throw 1}}, "")` throws `1`, it does
                // not stringify to `"[object Object]"` and fail to parse. Parameter
                // pieces are ToString'd before the body.
                let mut parts: Vec<String> = Vec::with_capacity(rest.len());
                for a in rest {
                    parts.push(self.coerce_to_string(*a)?);
                }
                let body = self.coerce_to_string(*last)?;
                (parts.join(","), body)
            }
            // `Function()` with no arguments → an empty-body anonymous function.
            None => (String::new(), String::new()),
        };
        let source = alloc::format!("({keyword} anonymous({params}\n) {{\n{body}\n}})");

        // A `Function(…)` body is global-scoped — no inherited `super`. (Its body
        // is wrapped in a function expression, so `new.target` inside is enabled by
        // the parser's own function-boundary handling, not this top-level flag.)
        let program = self.parse_eval_program(&source, false, false, false, false, &[], false)?;
        // The wrapper parses to a single parenthesized function-expression
        // statement; pull the `Function` node back out.
        let func = program.body.iter().find_map(|s| match s {
            Stmt::Expr { expression, .. } => match &**expression {
                Expr::Function(f) => Some(f),
                _ => None,
            },
            _ => None,
        });
        let Some(func) = func else {
            let m = self.new_str("Function constructor produced invalid source");
            return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
        };

        // Build the closure in the GLOBAL scope (not the caller's), sloppy unless
        // the body opts into strict mode.
        let saved_scope = core::mem::replace(&mut self.current, self.global_scope.clone());
        let saved_strict = self.strict;
        self.strict = has_use_strict(&func.body);
        // A `Function`/`GeneratorFunction`/`AsyncFunction`/`AsyncGeneratorFunction`
        // constructor result deliberately keeps NO retained source, so
        // `Function.prototype.toString` renders the NativeFunction form. The dynamic
        // wrapper's `async`/`*` prefix sits outside the extracted `func` span, so a
        // sliced source would drop it and no longer be valid NativeFunction syntax
        // (which the `AsyncFunction`/`GeneratorFunction` toString tests require).
        let f = self.make_function(
            &func.params,
            Body::Block(&func.body),
            func.is_async,
            func.is_generator,
        );
        self.strict = saved_strict;
        self.current = saved_scope;

        // `Function`-created functions are named "anonymous".
        self.set_fn_name(f, "anonymous");
        if let Some(h) = f.as_handle().map(Handle::from_raw) {
            // Tag it so the `%ThrowTypeError%` poison stays conservative for a
            // dynamically-built function (a dynamic generator must still throw).
            self.realm
                .set_hidden_property(h, DYN_FN_MARKER, NanBox::boolean(true));
            // Surface `name`/`length` as own data properties with the spec
            // attributes, matching other built-in functions.
            let len = func
                .params
                .iter()
                .take_while(|p| !p.rest && p.default.is_none())
                .count();
            self.install_fn_name_length(h, "anonymous", len as u32);
            // `GetFunctionRealm` tagging: a dynamic function belongs to the realm of
            // the `Function`/`GeneratorFunction`/… whose `[[Construct]]` built it
            // (`callee`), so a subsequent `new other.Function()` used as a
            // cross-realm `newTarget` is recognized by `GetPrototypeFromConstructor`.
            // (Its own `[[Prototype]]` is intentionally left as the current realm's
            // `%Function.prototype%`: the `AsyncFunction === Function` conflation —
            // and the fact that `%GeneratorFunction%`/`%AsyncGeneratorFunction%` are
            // not bare globals — makes a realm-derived re-link ambiguous for this
            // path, so the `*-prototype` realm-tagging tests are deferred.)
            let _ = new_target;
            if let Some(idx) = callee
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|ch| self.get_function_realm(ch))
            {
                self.fn_realm.insert(h.to_raw(), idx);
            }
        }
        Ok(f)
    }

    // --- functions ---

    /// Pre-declares hoisted `function` declarations in the current scope, so a
    /// declaration is callable before its textual position (and mutual
    /// recursion works).
    /// Hoists a statement sequence. Function declarations are always hoisted;
    /// `var` names hoist only at a function/program boundary (`hoist_vars`), not
    /// per-block, since `var` is function-scoped.
    fn hoist_with(&mut self, stmts: &'a [Stmt], hoist_vars: bool) -> Result<(), ExecError> {
        self.hoist_with_kind(stmts, hoist_vars, false)
    }

    /// Hoists a statement sequence. `eval_code` is true for an eval body, where
    /// the Annex B.3.3.3 rule differs from function code (B.3.3.2): a block
    /// function whose name matches an enclosing parameter still updates that
    /// binding in eval code, but not in function code.
    fn hoist_with_kind(
        &mut self,
        stmts: &'a [Stmt],
        hoist_vars: bool,
        eval_code: bool,
    ) -> Result<(), ExecError> {
        // `var` names hoist to the function/program scope as `undefined` (so a
        // read before the declaration yields `undefined`, not a ReferenceError).
        // Done first; a same-named function declaration then overwrites it.
        if hoist_vars {
            // This is a function/program/eval variable-environment boundary: the
            // variable scope is where `var`/top-level functions hoist, and where
            // the Annex B.3.3 runtime update for a block function writes. For an
            // eval body this may be an OUTER environment (the spec `varEnv`,
            // supplied via `eval_var_scope`) distinct from the fresh lexical
            // `self.current`; otherwise it is `self.current` itself.
            self.var_scope = self
                .eval_var_scope
                .take()
                .unwrap_or_else(|| self.current.clone());
            self.annexb_block_fns = Vec::new();
            let mut var_names: Vec<&str> = Vec::new();
            collect_var_names(stmts, &mut var_names);
            // Annex B: a function declared inside a block also var-hoists its name
            // to the enclosing function scope (initially `undefined`).
            let mut block_fn_names: Vec<&str> = Vec::new();
            collect_block_function_names(stmts, &mut block_fn_names);
            // A block-function name qualifies for the B.3.3 runtime update unless
            // it collides with a parameter or other binding already present in the
            // variable environment (where the function-code extension B.3.3.2 does
            // not apply). In eval code (B.3.3.3) such a collision is permitted, so
            // the binding is still updated.
            for name in &block_fn_names {
                if eval_code || !self.var_scope.has_local(name) {
                    self.annexb_block_fns.push(String::from(*name));
                }
            }
            var_names.extend_from_slice(&block_fn_names);
            let at_global = self.var_scope.ptr_eq(&self.global_scope);
            let global_obj = self.global_this.as_handle().map(Handle::from_raw);
            for name in var_names {
                // At global scope a `var`/Annex-B name that *already* exists as a
                // global-object own property IS that binding: don't shadow it with
                // a fresh `undefined` scope binding (it must keep its current value
                // and the property's attributes; an identifier read falls back to
                // the global-object property — see `read_ident_ref`). This is the
                // EvalDeclarationInstantiation "binding is not reinitialized" rule.
                let global_has =
                    at_global && global_obj.is_some_and(|g| self.realm.has_own(g, name));
                if !self.var_scope.has_local(name) && !global_has {
                    // A sloppy `eval`'s `var` in a *non-global* variable
                    // environment is a deletable binding (`delete` removes it);
                    // ordinary and global `var` bindings are not.
                    if eval_code && !at_global {
                        self.var_scope.declare_deletable(name, NanBox::undefined());
                    } else {
                        self.var_scope.declare(name, NanBox::undefined());
                    }
                }
                // A global `var` reserves an own property on the global object
                // (initially `undefined` until the declaration's initializer runs),
                // so `typeof x` / `this.x` see the hoisted binding. Don't clobber a
                // pre-existing global property (e.g. a built-in of the same name).
                if at_global
                    && !global_has
                    && let Some(g) = global_obj
                {
                    self.realm.set_property(g, name, NanBox::undefined());
                    // A global `var`/function binding created by *script* code is
                    // non-configurable (CreateGlobalVarBinding with deletable
                    // false). Bindings created by global *eval* code are deletable
                    // (configurable), so only lock script-scope ones — a
                    // `$262.evalScript` body is a Script, so it locks too.
                    if !eval_code || self.script_eval_globals {
                        self.realm.set_non_configurable_property(g, name);
                    }
                }
            }
        }
        for stmt in stmts {
            // A module's `export function f(){}` / `export default function f(){}`
            // hoists `f` exactly like a bare function declaration: unwrap the
            // export wrapper to reach the inner function declaration.
            let stmt = unwrap_exported_function(stmt);
            if let Stmt::Function(func) = stmt
                && let Some(id) = &func.id
            {
                let value = self.make_function(
                    &func.params,
                    Body::Block(&func.body),
                    func.is_async,
                    func.is_generator,
                );
                self.set_fn_name(value, &id.name);
                self.set_fn_source(value, func.span);
                if hoist_vars {
                    // A function/program top-level declaration binds in the
                    // variable environment (for an eval body, the outer `varEnv`).
                    // A sloppy `eval`'s function declaration in a non-global
                    // variable environment is a deletable binding.
                    if eval_code && !self.var_scope.ptr_eq(&self.global_scope) {
                        self.var_scope.declare_deletable(&id.name, value);
                    } else {
                        self.var_scope.declare(&id.name, value);
                    }
                    // A global function declaration also publishes on the global
                    // object (`function f(){}; this.f === f`).
                    if self.var_scope.ptr_eq(&self.global_scope)
                        && let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
                    {
                        // CreateGlobalFunctionBinding. When there is no existing own
                        // property, or it is configurable, (re)define with full data
                        // attributes: writable, enumerable, and configurable set to
                        // the binding's *deletable* flag. A binding created by
                        // *script* code (top-level program or `$262.evalScript`) is
                        // non-configurable (deletable = false); one created by global
                        // `eval` code is configurable (deletable = true). When the
                        // existing property is non-configurable (but declarable —
                        // validated by CanDeclareGlobalFunction), update only the
                        // value and preserve its attributes.
                        let deletable = eval_code && !self.script_eval_globals;
                        let redefine_attrs = !self.realm.has_own(g, &id.name)
                            || !self.realm.property_is_non_configurable(g, &id.name);
                        self.realm.force_set_property(g, &id.name, value);
                        if redefine_attrs {
                            self.realm.clear_readonly_property(g, &id.name);
                            self.realm.clear_hidden_property(g, &id.name);
                            if deletable {
                                self.realm.clear_non_configurable_property(g, &id.name);
                            } else {
                                self.realm.set_non_configurable_property(g, &id.name);
                            }
                        }
                    }
                } else {
                    // A block-level declaration binds *locally* in the block
                    // scope (block scoping). Its name was also `var`-hoisted to
                    // the function scope by `collect_block_function_names` when
                    // the Annex B.3.3 extension applies; the runtime update of
                    // that outer `var` binding happens when the function-decl
                    // statement is evaluated (see `exec_inner`'s `Stmt::Function`).
                    self.current.declare(&id.name, value);
                }
            }
        }
        // Lexical declaration instantiation (script / function body / eval /
        // module / block scope): `let`/`const`/`class` names bound directly in
        // this statement list (not in a nested block — those instantiate at their
        // own scope entry) are pre-declared in their Temporal Dead Zone at scope
        // entry, before any statement runs. A reference before the declaration's
        // initializer executes then throws a ReferenceError — a read via
        // `read_ident_ref` (which checks `is_tdz`) and a write via the assign path
        // (which checks it too). The initializer later clears the sentinel with
        // `declare`/`declare_const`. `has_local` guards against re-declaring a
        // name already bound here (e.g. a top-level function declaration; a true
        // let/const-vs-function collision is an early SyntaxError caught earlier).
        let mut lex_names: Vec<&str> = Vec::new();
        collect_lexical_names(stmts, &mut lex_names);
        for name in lex_names {
            if !self.current.has_local(name) {
                self.current.declare(name, NanBox::tdz());
            }
        }
        Ok(())
    }

    /// Block-level hoisting: function declarations only (`var` is function-scoped
    /// and hoisted at the function/program boundary instead).
    fn hoist(&mut self, stmts: &'a [Stmt]) -> Result<(), ExecError> {
        self.hoist_with(stmts, false)
    }

    /// The literal source slice for AST `span`, from the current source region
    /// (`self.src`), or `None` if no source is retained or the span is out of
    /// range / not on a UTF-8 boundary. The parser's spans are byte offsets into
    /// the source the AST was parsed from, so this reproduces the exact original
    /// text (comments and whitespace included).
    fn src_slice(&self, span: crate::common::Span) -> Option<&'a str> {
        self.src.get(span.start as usize..span.end as usize)
    }

    /// Stamps the literal source text of `span` onto the function/class `value`,
    /// so `Function.prototype.toString` (and `String(fn)` / `"" + fn`) reproduce
    /// it. Stored in the realm keyed by the value's handle, so both the display
    /// path (`Realm::to_display_string`) and the method path
    /// (`function_to_string_repr`) — and both engine tiers — share one source of
    /// truth. A no-op if `value` is not a heap handle or no source is retained.
    fn set_fn_source(&mut self, value: NanBox, span: crate::common::Span) {
        if let Some(slice) = self.src_slice(span)
            && let Some(h) = value.as_handle().map(Handle::from_raw)
        {
            self.realm.set_fn_source(h, alloc::rc::Rc::from(slice));
        }
    }

    /// Registers a function definition and allocates a closure capturing the
    /// current scope.
    fn make_function(
        &mut self,
        params: &'a [Param],
        body: Body<'a>,
        is_async: bool,
        is_generator: bool,
    ) -> NanBox {
        self.make_method(params, body, is_async, is_generator, None, false)
    }

    fn make_method(
        &mut self,
        params: &'a [Param],
        body: Body<'a>,
        is_async: bool,
        is_generator: bool,
        home_class: Option<u32>,
        home_static: bool,
    ) -> NanBox {
        // Strict mode is lexical: inherited from the defining context, or set by
        // the function body's own `"use strict"` directive prologue. A *class*
        // member (it carries a home class) is always strict — all class bodies are
        // strict code per spec, with no directive required.
        let is_strict = self.strict
            || home_class.is_some()
            || matches!(body, Body::Block(stmts) if has_use_strict(stmts));
        let func_id = self.functions.len() as u32;
        // A method's lexical class is its home; any other function captures the
        // *lexical* class enclosing its definition. Use `current_lexical_home`
        // (not `current_home`, which is `None` inside an ordinary function) so a
        // function nested inside a nested ordinary function still sees the class.
        let lexical_class = home_class.or(self.current_lexical_home);
        self.functions.push(FnDef {
            params,
            body,
            is_async,
            is_generator,
            is_arrow: false,
            is_strict,
            name: "",
            home_class,
            home_static,
            lexical_class,
            is_method: false,
            // Capture whether this function is *lexically* inside a class field
            // initializer / static block. An arrow inherits this on invocation (so
            // a direct `eval` reached through nested arrows defined in a field
            // initializer still gets the ContainsArguments early error); a
            // non-arrow shields it (its own `arguments` binding).
            field_init: self.in_field_initializer,
        });
        let handle = self.realm.new_function(func_id, self.current.clone());
        // `GetFunctionRealm` tagging: a closure created while executing in a
        // `$262.createRealm()` realm (e.g. a class method built by that realm's
        // `eval`) belongs to that realm, so a brand-check / type error it later
        // throws carries *that realm's* `%TypeError%` (see `make_error`). The
        // closure's captured scope may root at the main global scope (an indirect
        // eval runs there), so scope-walking alone would miss it — record it
        // explicitly. `None` (main realm) leaves the fast path untouched.
        if let Some(idx) = self.cur_realm {
            self.fn_realm.insert(handle.to_raw(), idx);
        }
        // Materialize `name` ("" until a later NamedEvaluation / method key sets
        // it) and `length` as own data properties so `hasOwnProperty("name")` and
        // `verifyProperty` behave per spec even for anonymous functions. A named
        // context (`set_fn_name`/`install_method_meta`) overwrites the name after.
        let length = params
            .iter()
            .take_while(|p| p.default.is_none() && !p.rest)
            .count() as u32;
        self.install_fn_name_length(handle, "", length);
        // A sync generator function's `[[Prototype]]` is `%GeneratorFunction.prototype%`
        // (distinct from `%Function.prototype%`), so `Object.getPrototypeOf(g).prototype`
        // resolves to `%GeneratorPrototype%`. `object_proto` honors this override.
        if is_generator
            && !is_async
            && let Some(gfp) = self.generator_function_prototype()
        {
            self.realm.set_native_proto(handle, gfp);
            // `g.prototype` inherits `%GeneratorPrototype%` (so a produced generator
            // reaches it, and `getPrototypeOf(getPrototypeOf(g.prototype))` is
            // `%IteratorPrototype%`).
            if let Some(gp) = self.generator_prototype() {
                let proto = self.realm.new_object_with_proto(Some(gp));
                self.realm.set_function_prototype(func_id, proto);
            }
        }
        // An `async function*`'s `[[Prototype]]` is `%AsyncGeneratorFunction.prototype%`.
        if is_generator
            && is_async
            && let Some(agfp) = self.async_generator_function_prototype()
        {
            self.realm.set_native_proto(handle, agfp);
            // `ag.prototype` inherits `%AsyncGeneratorPrototype%`, whose prototype is
            // `%AsyncIteratorPrototype%`.
            if let Some(agp) = self.async_generator_prototype() {
                let proto = self.realm.new_object_with_proto(Some(agp));
                self.realm.set_function_prototype(func_id, proto);
            }
        }
        // Materialize the own `prototype` data property for constructable kinds
        // (ordinary functions + generators). `is_arrow`/`is_method` are not yet
        // set here, so this over-includes arrows and concise methods; those
        // callers call `demote_fn_prototype` once the flag is stamped. For a
        // generator the proto was created above (`set_function_prototype`), so
        // `function_prototype` returns it without adding a `constructor`
        // back-link; for a plain function it lazily builds the default proto
        // (with the back-link) here. `prototype` is `writable: true` for both.
        if self.fn_has_prototype(func_id) {
            let proto = self.realm.function_prototype(func_id);
            self.install_fn_prototype(handle, proto, true);
        }
        NanBox::handle(handle.to_raw())
    }

    /// Calls `callee` with `args`.
    fn call(&mut self, callee: NanBox, args: &[NanBox]) -> Result<NanBox, ExecError> {
        self.call_with_this(callee, NanBox::undefined(), args)
    }

    /// Interns `s` to a `&'static str` (leak-once, deduped), for the `intl` number options
    /// whose `currency`/`unit` fields are `'static`.
    #[cfg(feature = "intl")]
    fn intern_static(&mut self, s: &str) -> &'static str {
        if let Some(&v) = self.intl_intern.get(s) {
            return v;
        }
        let leaked: &'static str = alloc::boxed::Box::leak(String::from(s).into_boxed_str());
        self.intl_intern.insert(String::from(s), leaked);
        leaked
    }

    /// A `TypeError` throw with `message`, ready to bubble out of `read_member`
    /// etc. as `Err(ExecError::Throw(..))`.
    fn type_error(&mut self, message: &str) -> ExecError {
        let m = self.new_str(message);
        ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m)))
    }

    fn make_error(&mut self, id: u16, message: Option<NanBox>) -> NanBox {
        let name = ERROR_NAMES[(id - N_ERROR_BASE) as usize];
        let obj = self.realm.new_object();
        // Link the error object to its constructor's `.prototype` (so
        // `Object.getPrototypeOf(new TypeError) === TypeError.prototype`,
        // `Object.prototype.toString` reports `[object Error]`, and the inherited
        // `Error.prototype.toString`/`constructor` resolve).
        // Resolve the *intrinsic* error constructor from the global scope, not the
        // current lexical scope: a local shadow (`(function(){ function TypeError(){}
        // … })()`) must not divert an engine-created `TypeError` onto the user
        // function's `prototype`. (User functions now carry a real own `prototype`,
        // so a `get_property` on a shadowing binding would otherwise resolve to it
        // instead of falling through.) Fall back to `current` only if the global
        // binding is absent (early bootstrap, before the constructors are bound).
        // When a *cross-realm* intrinsic is executing (`cur_realm` is set — e.g.
        // `otherRealm.String.prototype.valueOf` called on a bad `this`, or a class
        // method from a class evaluated in another realm hitting a brand-check
        // failure), the thrown error must be that realm's `%TypeError%` — resolve
        // the constructor from *that realm's* global object, not the main realm's.
        let realm_ctor = self
            .cur_realm
            .filter(|i| *i < self.created_realms.len())
            .and_then(|i| self.created_realms[i].global_this.as_handle())
            .map(Handle::from_raw)
            .and_then(|gt| self.realm.get_property(gt, name));
        if let Some(proto) = realm_ctor
            .or_else(|| self.global_scope.get(name))
            .or_else(|| self.current.get(name))
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_object_proto(obj, Some(proto));
        }
        let name_v = self.new_str(name);
        self.realm.set_property(obj, "name", name_v);
        // Only when `message` is present (not undefined) is an own, non-enumerable
        // `message` data property created (ECMA-262 20.5.8.1 InstallErrorCause /
        // the Error constructor step: `If message is not undefined …`). Otherwise
        // the instance inherits `Error.prototype.message` (the empty string), and
        // `err.hasOwnProperty("message")` is `false`.
        if let Some(m) = message
            && !matches!(m.unpack(), Unpacked::Undefined)
        {
            let msg_str = self.realm.to_display_string(m);
            let msg = self.new_str(&msg_str);
            self.realm.set_property(obj, "message", msg);
            self.realm.mark_hidden(obj, "message");
        }
        // `name` is non-enumerable (so `Object.keys(err)` is empty).
        self.realm.mark_hidden(obj, "name");
        // No own `stack` property: per the error-stack-accessor proposal, `stack`
        // is an inherited accessor on `Error.prototype` (see
        // `N_ERROR_PROTO_STACK_GET`/`N_ERROR_PROTO_STACK_SET`) driven by the
        // `[[ErrorData]]` brand, not an own data property of each instance.
        // Stamp the `[[ErrorData]]` brand (hidden; see `ERROR_DATA`) so
        // `Error.isError` recognizes this as a genuine Error instance.
        self.realm
            .set_hidden_property(obj, ERROR_DATA, NanBox::boolean(true));
        NanBox::handle(obj.to_raw())
    }

    /// Evaluates a call's arguments.
    fn eval_args(&mut self, arguments: &'a [Argument]) -> Result<Vec<NanBox>, ExecError> {
        let mut args = Vec::with_capacity(arguments.len());
        for a in arguments {
            match a {
                Argument::Item(e) => args.push(self.eval(e)?),
                Argument::Spread(e) => {
                    let v = self.eval(e)?;
                    args.extend(self.iterate_values(v)?);
                }
            }
        }
        Ok(args)
    }

    /// Allocates a heap string and returns its boxed handle.
    fn new_str(&mut self, s: &str) -> NanBox {
        NanBox::handle(self.realm.new_string(s).to_raw())
    }

    /// Allocates a heap string from raw **WTF-8 bytes** (lone surrogates
    /// preserved) and returns its boxed handle.
    fn new_str_bytes(&mut self, bytes: alloc::vec::Vec<u8>) -> NanBox {
        NanBox::handle(self.realm.new_string_wtf8(bytes).to_raw())
    }

    /// The WTF-8 bytes of `v` coerced to a string — lossless when `v` is already
    /// a string (so a surrogate needle matches surrogate haystack bytes), lossy
    /// otherwise (numbers/etc. carry no surrogates). Used by the unit-based
    /// string search ops.
    fn arg_string_bytes(&self, v: NanBox) -> alloc::vec::Vec<u8> {
        if let Some(raw) = v.as_handle()
            && let Some(b) = self.realm.string_bytes(Handle::from_raw(raw))
        {
            return b;
        }
        self.realm.to_display_string(v).into_bytes()
    }

    /// `ToString(v)` as WTF-8 bytes, fallibly: an object runs ToPrimitive(string)
    /// (its `@@toPrimitive`/`toString`/`valueOf`, which may throw); a Symbol is a
    /// TypeError. A surrogate-bearing string value is preserved losslessly.
    fn arg_string_bytes_fallible(&mut self, v: NanBox) -> Result<alloc::vec::Vec<u8>, ExecError> {
        if let Some(raw) = v.as_handle()
            && let Some(b) = self.realm.string_bytes(Handle::from_raw(raw))
        {
            return Ok(b);
        }
        let prim = self.coerce_primitive(v, "string")?;
        if let Some(h) = prim.as_handle().map(Handle::from_raw) {
            if self.realm.symbol_at(h).is_some() {
                return Err(self.type_error("Cannot convert a Symbol value to a string"));
            }
            if let Some(b) = self.realm.string_bytes(h) {
                return Ok(b);
            }
        }
        Ok(self.realm.to_display_string(prim).into_bytes())
    }
}

/// Collects `var`-declared identifier names in `stmts`, recursing through nested
/// statements but NOT into nested function bodies (which have their own scope) —
/// for hoisting `var` bindings to the enclosing function/program scope.
/// If `stmt` is an `export <decl>` / `export default <decl>` wrapper around a
/// function/var/class declaration, returns the inner declaration; otherwise
/// returns `stmt` unchanged. Lets the shared hoisting machinery treat an
/// exported declaration exactly like a bare one.
#[cfg(all(feature = "module", feature = "std"))]
fn unwrap_exported_function(stmt: &Stmt) -> &Stmt {
    use crate::ast::ExportDecl;
    match stmt {
        Stmt::Export(ExportDecl::Decl { declaration, .. })
        | Stmt::Export(ExportDecl::Default { declaration, .. }) => declaration,
        _ => stmt,
    }
}

#[cfg(not(all(feature = "module", feature = "std")))]
fn unwrap_exported_function(stmt: &Stmt) -> &Stmt {
    stmt
}

fn collect_var_names<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>) {
    use crate::ast::VarDeclKind;
    fn from_decl<'a>(decl: &'a crate::ast::VarDecl, out: &mut Vec<&'a str>) {
        if matches!(decl.kind, VarDeclKind::Var) {
            for d in &decl.declarations {
                if let BindingTarget::Ident(id) = &d.target {
                    out.push(&id.name);
                }
            }
        }
    }
    for stmt in stmts {
        match stmt {
            Stmt::Var(decl) => from_decl(decl, out),
            // `export var x = …` var-hoists `x` like a bare `var`.
            Stmt::Export(crate::ast::ExportDecl::Decl { declaration, .. }) => {
                if let Stmt::Var(decl) = &**declaration {
                    from_decl(decl, out);
                }
            }
            Stmt::Block { body, .. } => collect_var_names(body, out),
            Stmt::If {
                consequent,
                alternate,
                ..
            } => {
                collect_var_names(core::slice::from_ref(consequent), out);
                if let Some(alt) = alternate {
                    collect_var_names(core::slice::from_ref(alt), out);
                }
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::Labeled { body, .. }
            | Stmt::With { body, .. } => {
                collect_var_names(core::slice::from_ref(body), out);
            }
            Stmt::For { init, body, .. } => {
                if let Some(crate::ast::ForInit::Var(decl)) = init {
                    from_decl(decl, out);
                }
                collect_var_names(core::slice::from_ref(body), out);
            }
            Stmt::ForIn { left, body, .. } | Stmt::ForOf { left, body, .. } => {
                if let crate::ast::ForLeft::Decl {
                    kind: VarDeclKind::Var,
                    target: BindingTarget::Ident(id),
                    ..
                } = left
                {
                    out.push(&id.name);
                }
                collect_var_names(core::slice::from_ref(body), out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    collect_var_names(&c.body, out);
                }
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                collect_var_names(block, out);
                if let Some(h) = handler {
                    collect_var_names(&h.body, out);
                }
                if let Some(f) = finalizer {
                    collect_var_names(f, out);
                }
            }
            // `Stmt::Function` bodies have their own scope — not traversed.
            _ => {}
        }
    }
}

/// Whether a body's directive prologue contains `"use strict"` — a leading run
/// of string-literal expression statements, one of which is exactly `use strict`.
fn has_use_strict(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Expr { expression, .. } => match &**expression {
                Expr::Str { value, span, .. } => {
                    // A Use Strict Directive's *source text* must be exactly
                    // `use strict` — no escape sequence or line continuation. The
                    // 10-character directive inside two quotes is 12 source bytes;
                    // an escaped form (`'use strict'`, `'use str\<LF>ict'`)
                    // cooks to the same value but is longer, so it is not a
                    // directive and does not trigger strict mode.
                    if &**value == b"use strict" && span.end.saturating_sub(span.start) == 12 {
                        return true;
                    }
                }
                _ => return false,
            },
            _ => return false,
        }
    }
    false
}

/// Collects the lexically-declared names (`let`/`const`/`class`) directly in a
/// statement list — i.e. the names that form the block's lexical environment.
/// Does not recurse into nested blocks or function bodies. Used for the Annex
/// B.3.3 early-error check.
fn collect_lexical_names<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>) {
    use crate::ast::VarDeclKind;
    for stmt in stmts {
        match stmt {
            Stmt::Var(decl) if matches!(decl.kind, VarDeclKind::Let | VarDeclKind::Const) => {
                for d in &decl.declarations {
                    collect_binding_idents(&d.target, out);
                }
            }
            Stmt::Class(c) => {
                if let Some(id) = &c.id {
                    out.push(&id.name);
                }
            }
            // `export let/const/class X …` binds `X` lexically at the module top
            // level (its inner declaration is a `let`/`const`/`class`). Recurse so
            // the name is pre-declared in its TDZ like a bare lexical declaration.
            // (An `export var`/function declaration is *not* lexical, so the
            // recursion — which only collects `let`/`const`/`class` — skips it.)
            #[cfg(feature = "module")]
            Stmt::Export(crate::ast::ExportDecl::Decl { declaration, .. }) => {
                collect_lexical_names(core::slice::from_ref(declaration), out);
            }
            _ => {}
        }
    }
}

/// Pushes every identifier bound by a (possibly destructuring) binding target.
fn collect_binding_idents<'a>(target: &'a BindingTarget, out: &mut Vec<&'a str>) {
    use crate::ast::ArrayPatternElement;
    match target {
        BindingTarget::Ident(id) => out.push(&id.name),
        BindingTarget::Array(arr) => {
            for el in &arr.elements {
                match el {
                    ArrayPatternElement::Item { target, .. }
                    | ArrayPatternElement::Rest { target, .. } => {
                        collect_binding_idents(target, out)
                    }
                    ArrayPatternElement::Hole => {}
                }
            }
        }
        BindingTarget::Object(obj) => {
            for p in &obj.properties {
                collect_binding_idents(&p.value, out);
            }
            if let Some(rest) = &obj.rest {
                collect_binding_idents(rest, out);
            }
        }
    }
}

/// Collects the names of function declarations that appear **inside a block** (at
/// any nesting depth below the immediate statement list). Per Annex B.3.3, such a
/// name is var-hoisted to the enclosing function scope — *unless* doing so would
/// create an early error, i.e. the name is also lexically bound (`let`/`const`/
/// `class`) in one of the block scopes enclosing the function declaration (up to
/// and including the function/eval top-level lexical scope). In that case the
/// legacy var-hoisting extension is skipped. The immediate top-level functions
/// are excluded — they are bound directly by the hoisting loop.
fn collect_block_function_names<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>) {
    use core::slice::from_ref;
    // `blocked` is the set of names lexically declared in any enclosing block on
    // the current path; a block function with such a name is not var-hoisted.
    fn walk<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>, in_block: bool, blocked: &[&'a str]) {
        // Lexical names declared directly in this statement list shadow a
        // same-named block function nested deeper (Annex B early-error guard).
        let mut blocked_here: Vec<&str> = blocked.to_vec();
        collect_lexical_names(stmts, &mut blocked_here);

        for stmt in stmts {
            match stmt {
                Stmt::Function(f) if in_block => {
                    if let Some(id) = &f.id
                        && !blocked.contains(&&*id.name)
                    {
                        out.push(&id.name);
                    }
                }
                Stmt::Block { body, .. } => walk(body, out, true, &blocked_here),
                Stmt::If {
                    consequent,
                    alternate,
                    ..
                } => {
                    walk(from_ref(consequent), out, true, &blocked_here);
                    if let Some(a) = alternate {
                        walk(from_ref(a), out, true, &blocked_here);
                    }
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::Labeled { body, .. } => walk(from_ref(body), out, true, &blocked_here),
                Stmt::For { init, body, .. } => {
                    // A `for (let/const …; …)` head introduces a lexical scope
                    // enclosing the body; its names block the extension.
                    let mut for_lex: Vec<&str> = blocked_here.clone();
                    if let Some(crate::ast::ForInit::Var(decl)) = init
                        && matches!(
                            decl.kind,
                            crate::ast::VarDeclKind::Let | crate::ast::VarDeclKind::Const
                        )
                    {
                        for d in &decl.declarations {
                            collect_binding_idents(&d.target, &mut for_lex);
                        }
                    }
                    walk(from_ref(body), out, true, &for_lex);
                }
                Stmt::ForIn { left, body, .. } | Stmt::ForOf { left, body, .. } => {
                    let mut for_lex: Vec<&str> = blocked_here.clone();
                    if let crate::ast::ForLeft::Decl {
                        kind: crate::ast::VarDeclKind::Let | crate::ast::VarDeclKind::Const,
                        target,
                        ..
                    } = left
                    {
                        collect_binding_idents(target, &mut for_lex);
                    }
                    walk(from_ref(body), out, true, &for_lex);
                }
                Stmt::Try {
                    block,
                    handler,
                    finalizer,
                    ..
                } => {
                    walk(block, out, true, &blocked_here);
                    if let Some(h) = handler {
                        // Annex B.3.5: a *simple* (BindingIdentifier) catch
                        // parameter does NOT block a same-named block function
                        // from var-hoisting (`catch (f) { { function f(){} } }`
                        // still hoists `f`). A *destructuring* catch parameter's
                        // names are lexical and do block (a `var` of the same
                        // name there would be an early error).
                        let mut catch_blocked: Vec<&str> = blocked_here.clone();
                        if let Some(p @ (BindingTarget::Object(_) | BindingTarget::Array(_))) =
                            &h.param
                        {
                            collect_binding_idents(p, &mut catch_blocked);
                        }
                        walk(&h.body, out, true, &catch_blocked);
                    }
                    if let Some(f) = finalizer {
                        walk(f, out, true, &blocked_here);
                    }
                }
                Stmt::Switch { cases, .. } => {
                    // A switch body is a single lexical (block) scope shared by
                    // all cases.
                    let mut switch_lex: Vec<&str> = blocked_here.clone();
                    for case in cases {
                        collect_lexical_names(&case.body, &mut switch_lex);
                    }
                    for case in cases {
                        walk(&case.body, out, true, &switch_lex);
                    }
                }
                _ => {}
            }
        }
    }
    // The function/eval/program top-level lexical scope: its `let`/`const`/`class`
    // names also block the extension for a same-named nested block function.
    let mut top_lex: Vec<&str> = Vec::new();
    collect_lexical_names(stmts, &mut top_lex);
    walk(stmts, out, false, &top_lex);
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

/// The byte offset in WTF-8 `bytes` immediately after the first `unit` UTF-16
/// code units (clamped to `bytes.len()`). A `unit` landing inside an astral
/// surrogate pair rounds *down* to that character's start (a search position
/// never splits a pair).
fn unit_to_byte(bytes: &[u8], unit: usize) -> usize {
    let mut units = 0;
    for (cp, off, len) in wtf8_code_point_iter(bytes) {
        if units >= unit {
            return off;
        }
        units += if cp >= 0x1_0000 { 2 } else { 1 };
        let _ = len;
    }
    bytes.len()
}

/// The number of UTF-16 code units in `bytes[..byte_off]`.
fn byte_to_unit(bytes: &[u8], byte_off: usize) -> usize {
    crate::wtf8::utf16_len(&bytes[..byte_off.min(bytes.len())])
}

/// Iterates `(code_point, byte_offset, byte_len)` over WTF-8 `bytes`. A thin
/// shim over [`crate::wtf8::code_points`] that also tracks the byte offset, for
/// the unit↔byte conversions the search/slice ops need.
fn wtf8_code_point_iter(bytes: &[u8]) -> impl Iterator<Item = (u32, usize, usize)> + '_ {
    let mut off = 0usize;
    core::iter::from_fn(move || {
        if off >= bytes.len() {
            return None;
        }
        let start = off;
        // Re-decode a single code point's byte length from the lead byte.
        let b0 = bytes[off];
        let len = if b0 < 0x80 {
            1
        } else if b0 < 0xE0 {
            2
        } else if b0 < 0xF0 {
            3
        } else {
            4
        }
        .min(bytes.len() - off);
        let cp = crate::wtf8::code_points(&bytes[off..off + len])
            .next()
            .unwrap_or(0xFFFD);
        off += len;
        Some((cp, start, len))
    })
}

/// `String.prototype.indexOf` over UTF-16 units: searches the WTF-8 `hay` for
/// the WTF-8 `needle` starting at UTF-16 unit `from`, returning the unit index
/// of the match (or `-1`). Mirrors JS: an empty needle matches at `from`.
fn index_of_units(hay: &[u8], needle: &[u8], from: usize) -> f64 {
    let start_byte = unit_to_byte(hay, from);
    if needle.is_empty() {
        return byte_to_unit(hay, start_byte) as f64;
    }
    let mut i = start_byte;
    while i + needle.len() <= hay.len() {
        if &hay[i..i + needle.len()] == needle {
            return byte_to_unit(hay, i) as f64;
        }
        i += 1;
    }
    -1.0
}

/// `String.prototype.lastIndexOf` over UTF-16 units: the last match of `needle`
/// in `hay` at or before unit `from` (`usize::MAX` for "anywhere"), as a unit
/// index, or `-1`.
fn last_index_of_units(hay: &[u8], needle: &[u8], from: usize) -> f64 {
    let limit_byte = if from == usize::MAX {
        hay.len()
    } else {
        unit_to_byte(hay, from)
    };
    if needle.is_empty() {
        return byte_to_unit(hay, limit_byte.min(hay.len())) as f64;
    }
    // A needle longer than the haystack can never match.
    if needle.len() > hay.len() {
        return -1.0;
    }
    let max_start = hay.len() - needle.len();
    // The last start byte allowed is min(limit, max_start); scan downward.
    let upper = limit_byte.min(max_start);
    for i in (0..=upper).rev() {
        if &hay[i..i + needle.len()] == needle {
            return byte_to_unit(hay, i) as f64;
        }
    }
    -1.0
}

/// Splits WTF-8 `hay` on the non-empty WTF-8 `sep`, returning the segments as
/// WTF-8 byte buffers (the byte-level split is exact: `sep` is well-formed
/// WTF-8, so matches land on code-point boundaries and never split a surrogate).
fn split_units(hay: &[u8], sep: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i + sep.len() <= hay.len() {
        if &hay[i..i + sep.len()] == sep {
            out.push(hay[start..i].to_vec());
            i += sep.len();
            start = i;
        } else {
            i += 1;
        }
    }
    out.push(hay[start..].to_vec());
    out
}

/// Computes `[start, end)` char indices for `slice`, handling negative indices
/// (from the end) and an `undefined` end (to the length), clamped to `[0, len]`.
fn slice_bounds(start: f64, end: Option<f64>, len: usize) -> (usize, usize) {
    let clamp = |n: f64| -> usize {
        if n < 0.0 {
            (len as f64 + n).max(0.0) as usize
        } else {
            (n as usize).min(len)
        }
    };
    let a = clamp(start);
    let b = end.map_or(len, clamp);
    (a, b.max(a))
}

/// `padStart`/`padEnd` over UTF-16 units, on WTF-8 bytes: pads `s` with `pad`
/// (repeated, then truncated to a unit boundary) so the result is `target` units
/// long. `at_start == true` prepends the filler (padStart); otherwise appends
/// (padEnd). A `target` no greater than `s`'s length, or an empty `pad`, returns
/// `s` unchanged. Unit-length aware so an astral pad character counts as two.
fn pad_units(s: &[u8], target: usize, pad: &[u8], at_start: bool) -> Vec<u8> {
    let len = crate::wtf8::utf16_len(s);
    if len >= target || pad.is_empty() {
        return s.to_vec();
    }
    let need = target - len;
    let pad_unit_count = crate::wtf8::utf16_len(pad);
    // Build the filler from exactly `need` UTF-16 code units (per spec the fill is
    // truncated by *code unit*, which may leave a lone surrogate — not by whole
    // code point). Collect the pad's units and repeat them unit-by-unit.
    let pad_units: Vec<u16> = crate::wtf8::utf16_units(pad).collect();
    let mut units: Vec<u16> = Vec::with_capacity(need);
    let mut idx = 0usize;
    while units.len() < need {
        units.push(pad_units[idx % pad_unit_count]);
        idx += 1;
    }
    let filler = crate::wtf8::from_utf16(&units);
    let mut out = Vec::with_capacity(filler.len() + s.len());
    if at_start {
        out.extend_from_slice(&filler);
        out.extend_from_slice(s);
    } else {
        out.extend_from_slice(s);
        out.extend_from_slice(&filler);
    }
    out
}

/// `Number.prototype.toPrecision(p)`: render `n` with `p` significant digits,
/// choosing fixed or exponential notation by magnitude (as the spec does).
/// `Number.prototype.toExponential` with `frac` fractional digits (`None` =
/// "as many digits as needed to represent the value uniquely"). Uses ties-away
/// rounding (the spec picks the larger `n` on an exact tie), unlike Rust's
/// ties-to-even formatter, and never emits a `-0` sign. The exponent carries an
/// explicit sign (`1.23e+4`, `5e-3`).
fn format_exponential(n: f64, frac: Option<usize>) -> String {
    debug_assert!(n.is_finite());
    let neg = n.is_sign_negative() && n != 0.0;
    let abs = n.abs();
    // Build the mantissa string and its exponent. For a fixed digit count we
    // render with one guard digit and round ties-away (the spec picks the larger
    // value on an exact tie), unlike Rust's ties-to-even formatter. The exponent
    // is read from the *same* rendering so a tie-induced carry (`9.995 → 1.00`,
    // exponent +1) stays consistent.
    let (mantissa, exp) = match frac {
        Some(f) => {
            // Render with many guard digits so the ties-away rounding decision
            // sees the exact decimal expansion (one guard digit alone is itself
            // pre-rounded by Rust's formatter and would mis-round, e.g.
            // `123456 → 1.235` then up to `1.24` instead of `1.23`).
            let mantissa_full = alloc::format!("{:.*e}", f + 25, abs);
            let mut exp: i32 = mantissa_full
                .rfind('e')
                .and_then(|i| mantissa_full[i + 1..].parse().ok())
                .unwrap_or(0);
            let m = round_exp_mantissa(&mantissa_full, f, &mut exp);
            (m, exp)
        }
        None => {
            // Shortest unique mantissa: Rust's default `{:e}` already does this.
            let sci = alloc::format!("{abs:e}");
            let exp: i32 = sci
                .rfind('e')
                .and_then(|i| sci[i + 1..].parse().ok())
                .unwrap_or(0);
            (String::from(sci.split('e').next().unwrap_or("0")), exp)
        }
    };
    let sign = if exp < 0 { '-' } else { '+' };
    if neg {
        alloc::format!("-{mantissa}e{sign}{}", exp.abs())
    } else {
        alloc::format!("{mantissa}e{sign}{}", exp.abs())
    }
}

/// Rounds a scientific-notation mantissa string (e.g. `"2.50"`, from Rust's
/// ties-to-even formatter rendered with one guard digit) to `f` fractional
/// digits using ties-away rounding, adjusting `exp` if a carry bumps the leading
/// digit (`9.95 → 1.00`, exponent +1).
fn round_exp_mantissa(mantissa_full: &str, f: usize, exp: &mut i32) -> String {
    // `mantissa_full` is `d.ddd...` with `f + 1` fractional digits (no exponent
    // part — we strip it).
    let core = mantissa_full.split('e').next().unwrap_or(mantissa_full);
    let digits: Vec<u8> = core.bytes().filter(u8::is_ascii_digit).collect();
    // We keep `f + 1` digits total (1 integer + f fractional) and round on the
    // last guard digit.
    let keep = f + 1;
    let mut kept: Vec<u8> = digits.iter().take(keep).copied().collect();
    while kept.len() < keep {
        kept.push(b'0');
    }
    let round_up = digits.get(keep).is_some_and(|&d| d >= b'5');
    if round_up {
        let mut i = kept.len();
        loop {
            if i == 0 {
                // Carry past the most significant digit: prepend 1 and bump exp.
                kept.insert(0, b'1');
                kept.pop();
                *exp += 1;
                break;
            }
            i -= 1;
            if kept[i] == b'9' {
                kept[i] = b'0';
            } else {
                kept[i] += 1;
                break;
            }
        }
    }
    let int_part = kept[0] as char;
    if f == 0 {
        alloc::format!("{int_part}")
    } else {
        let frac_part: String = kept[1..=f].iter().map(|&b| b as char).collect();
        alloc::format!("{int_part}.{frac_part}")
    }
}

fn format_precision(n: f64, p: usize) -> String {
    if n == 0.0 {
        return alloc::format!("{:.*}", p - 1, 0.0);
    }
    if !n.is_finite() {
        return if n.is_nan() {
            String::from("NaN")
        } else if n > 0.0 {
            String::from("Infinity")
        } else {
            String::from("-Infinity")
        };
    }
    // Derive the decimal exponent from the default scientific rendering (no
    // `log10`, which isn't available in the no_std float set).
    let sci = alloc::format!("{:e}", n.abs());
    let e: i32 = sci
        .rfind('e')
        .and_then(|i| sci[i + 1..].parse().ok())
        .unwrap_or(0);
    if e < -6 || e >= p as i32 {
        // Exponential notation with p-1 fractional digits. Rust omits the `+` on
        // a non-negative exponent; JavaScript includes it (`1e+4`, not `1e4`).
        let s = alloc::format!("{:.*e}", p - 1, n);
        return match s.find('e') {
            Some(epos) if s.as_bytes().get(epos + 1) != Some(&b'-') => {
                alloc::format!("{}e+{}", &s[..epos], &s[epos + 1..])
            }
            _ => s,
        };
    }
    let decimals = (p as i32 - 1 - e).max(0) as usize;
    alloc::format!("{n:.decimals$}")
}

/// Quotes and escapes a string as a JSON string literal (the `&str` form, used
/// for property keys). [`json_quote_wtf8`] is the surrogate-preserving form for
/// string *values*.
fn json_quote(s: &str) -> String {
    json_quote_wtf8(s.as_bytes())
}

/// Quotes and escapes WTF-8 bytes as a JSON string literal, iterating code
/// points so a **lone surrogate** is escaped as `\uXXXX` (well-formed JSON per
/// the spec) and astral scalars emit their characters directly.
fn json_quote_wtf8(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 2);
    out.push('"');
    for cp in crate::wtf8::code_points(bytes) {
        match cp {
            0x22 => out.push_str("\\\""),
            0x5C => out.push_str("\\\\"),
            0x08 => out.push_str("\\b"),
            0x0C => out.push_str("\\f"),
            0x0A => out.push_str("\\n"),
            0x0D => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            cp if cp < 0x20 || crate::wtf8::is_surrogate(cp) => {
                out.push_str(&alloc::format!("\\u{cp:04x}"));
            }
            cp => {
                if let Some(c) = char::from_u32(cp) {
                    out.push(c);
                }
            }
        }
    }
    out.push('"');
    out
}

/// Reads exactly four hex digits from the char slice `c` starting at `at`,
/// returning the `u16` code unit. `None` if fewer than four hex digits are
/// present (a malformed `\u` escape).
fn json_hex4(c: &[char], at: usize) -> Option<u16> {
    let hex: String = c.get(at..at + 4)?.iter().collect();
    u16::from_str_radix(&hex, 16).ok()
}

/// Renders the integer part of `n` in `radix` (2–36), with a leading `-` for
/// negatives (matching `Number.prototype.toString(radix)` for integers).
/// A minimal `Number.prototype.toLocaleString` — groups the integer part with
/// `,` thousands separators (no locale data, so this is the en-US-ish default).
/// Coerces `n` to a typed-array element of the given kind index (see
/// [`TYPED_ARRAY_KINDS`]): integer kinds truncate then wrap (or clamp, for
/// `Uint8Clamped`); float kinds narrow precision.
/// Parses a `DataView` accessor name (`getInt32`, `setFloat64`, …) into
/// `(is_set, byte_size, signed, is_float)`, or `None` if it isn't one.
fn dataview_method(method: &str) -> Option<(bool, usize, bool, bool, bool)> {
    let (is_set, t) = if let Some(t) = method.strip_prefix("get") {
        (false, t)
    } else if let Some(t) = method.strip_prefix("set") {
        (true, t)
    } else {
        return None;
    };
    // (size, signed, is_float, is_bigint)
    let (size, signed, is_float, is_bigint) = match t {
        "Int8" => (1, true, false, false),
        "Uint8" => (1, false, false, false),
        "Int16" => (2, true, false, false),
        "Uint16" => (2, false, false, false),
        "Int32" => (4, true, false, false),
        "Uint32" => (4, false, false, false),
        "Float16" => (2, false, true, false),
        "Float32" => (4, false, true, false),
        "Float64" => (8, false, true, false),
        "BigInt64" => (8, true, false, true),
        "BigUint64" => (8, false, false, true),
        _ => return None,
    };
    Some((is_set, size, signed, is_float, is_bigint))
}

/// Maps a WASM export/import kind byte to its `ExternType` string.
fn wasm_extern_kind(kind: u8) -> &'static str {
    match kind {
        0 => "function",
        1 => "table",
        2 => "memory",
        _ => "global",
    }
}

pub(crate) fn coerce_typed(kind: u16, n: f64) -> f64 {
    match kind {
        7 => f64::from(n as f32),             // Float32
        8 => n,                               // Float64
        11 => f16_to_f64(f64_to_f16_bits(n)), // Float16 (round to nearest f16)
        2 => {
            // Uint8Clamped: clamp to 0..=255 with round-half-to-even (core only).
            if n.is_nan() || n <= 0.0 {
                0.0
            } else if n >= 255.0 {
                255.0
            } else {
                let fl = n as i64;
                let frac = n - fl as f64;
                let r = if frac < 0.5 {
                    fl
                } else if frac > 0.5 || fl % 2 != 0 {
                    fl + 1
                } else {
                    fl
                };
                r as f64
            }
        }
        _ => {
            if !n.is_finite() {
                return 0.0;
            }
            // Truncate toward zero, then reduce into range with integer math (no
            // std float methods, for the `alloc`-only build).
            let i = n as i64;
            let (bits, signed) = match kind {
                0 => (8u32, true), // Int8
                1 => (8, false),   // Uint8
                3 => (16, true),   // Int16
                4 => (16, false),  // Uint16
                5 => (32, true),   // Int32
                _ => (32, false),  // Uint32
            };
            let modulus = 1i64 << bits;
            let mut u = i.rem_euclid(modulus);
            if signed && u >= modulus / 2 {
                u -= modulus;
            }
            u as f64
        }
    }
}

/// The en-US display string for a relative-time `unit` (already singularized) at the
/// given `style`, choosing the singular or plural form. `long` is the full word
/// ("second"/"seconds"); `short`/`narrow` use the CLDR abbreviations.
fn rel_time_unit_display(unit: &str, style: &str, plural: bool) -> &'static str {
    // (long-singular, long-plural, short/narrow-singular, short/narrow-plural)
    let (ls, lp, ss, sp): (&str, &str, &str, &str) = match unit {
        "second" => ("second", "seconds", "sec.", "sec."),
        "minute" => ("minute", "minutes", "min.", "min."),
        "hour" => ("hour", "hours", "hr.", "hr."),
        "day" => ("day", "days", "day", "days"),
        "week" => ("week", "weeks", "wk.", "wk."),
        "month" => ("month", "months", "mo.", "mo."),
        "quarter" => ("quarter", "quarters", "qtr.", "qtrs."),
        "year" => ("year", "years", "yr.", "yr."),
        _ => ("", "", "", ""),
    };
    match (style, plural) {
        ("long", false) => ls,
        ("long", true) => lp,
        (_, false) => ss,
        (_, true) => sp,
    }
}

/// The idiomatic en-US `numeric: "auto"` phrase for `unit` at integer offset `v`
/// ("yesterday"/"this week"/"now"/…), or `None` when CLDR has no special phrase and
/// the explicit numeric form ("in N units") must be used. Only the `long` style has
/// these special phrases in the tested data.
fn rel_time_auto_phrase(unit: &str, v: i64) -> Option<&'static str> {
    Some(match (unit, v) {
        ("year", -1) => "last year",
        ("year", 0) => "this year",
        ("year", 1) => "next year",
        ("quarter", -1) => "last quarter",
        ("quarter", 0) => "this quarter",
        ("quarter", 1) => "next quarter",
        ("month", -1) => "last month",
        ("month", 0) => "this month",
        ("month", 1) => "next month",
        ("week", -1) => "last week",
        ("week", 0) => "this week",
        ("week", 1) => "next week",
        ("day", -1) => "yesterday",
        ("day", 0) => "today",
        ("day", 1) => "tomorrow",
        ("hour", 0) => "this hour",
        ("minute", 0) => "this minute",
        ("second", 0) => "now",
        _ => return None,
    })
}

/// Formats the (non-negative) magnitude of an en-US relative-time value into typed
/// number parts: `(type, value, with_unit=true)` triples of `integer`/`group`/
/// `decimal`/`fraction` (latn digits, `,` grouping, `.` decimal — matching
/// `Intl.NumberFormat("en-US")`). The integer part is grouped in threes from the right.
fn rel_time_number_parts(n: f64) -> alloc::vec::Vec<(&'static str, alloc::string::String, bool)> {
    // Render with the default NumberFormat shape (max 3 fraction digits, no trailing
    // zeros) by formatting then trimming; `n` is finite and non-negative here.
    let s = alloc::format!("{n}");
    let (int_str, frac_str) = match s.split_once('.') {
        Some((i, f)) => (i, f),
        None => (s.as_str(), ""),
    };
    let mut parts: alloc::vec::Vec<(&'static str, alloc::string::String, bool)> =
        alloc::vec::Vec::new();
    // Group the integer digits in threes from the right, emitting `group` parts.
    let digits: alloc::vec::Vec<char> = int_str.chars().collect();
    let len = digits.len();
    let first = len % 3;
    let first = if first == 0 && len > 0 { 3 } else { first };
    let emit =
        |slice: &[char],
         parts: &mut alloc::vec::Vec<(&'static str, alloc::string::String, bool)>| {
            let g: alloc::string::String = slice.iter().collect();
            parts.push(("integer", g, true));
        };
    if len > 0 {
        emit(&digits[..first], &mut parts);
        let mut idx = first;
        while idx < len {
            parts.push(("group", alloc::string::String::from(","), true));
            emit(&digits[idx..idx + 3], &mut parts);
            idx += 3;
        }
    }
    if !frac_str.is_empty() {
        parts.push(("decimal", alloc::string::String::from("."), true));
        parts.push(("fraction", alloc::string::String::from(frac_str), true));
    }
    parts
}

/// Partitions an `Intl.RelativeTimeFormat` `format`/`formatToParts(value, unit)` into
/// `(type, value, with_unit)` parts in en-US. `numeric: "auto"` yields idiomatic
/// single-`literal` phrases for the adjacent `long`-style units ("yesterday", "next
/// week", "now"); otherwise the pattern is the explicit "in N <unit>" / "N <unit> ago"
/// with the numeric magnitude split into typed `integer`/`group`/`decimal`/`fraction`
/// parts (each carrying the unit) surrounded by `literal` text. `unit` is singular.
fn rel_time_parts(
    value: f64,
    unit: &str,
    numeric: &str,
    style: &str,
) -> alloc::vec::Vec<(&'static str, alloc::string::String, bool)> {
    // numeric:"auto" idiomatic phrases (integer offsets only, `long` style).
    if numeric == "auto"
        && style == "long"
        && value == (value as i64) as f64
        && let Some(phrase) = rel_time_auto_phrase(unit, value as i64)
    {
        return alloc::vec![("literal", alloc::string::String::from(phrase), false)];
    }
    let n = value.abs();
    let plural = n != 1.0;
    let unit_disp = rel_time_unit_display(unit, style, plural);
    // Negative magnitudes use the "past" pattern (" <unit> ago"); positive (and `+0`)
    // use the "future" pattern ("in " … " <unit>"). The sign *bit* selects the pattern,
    // so `-0` is past (`format(-0)` → "0 units ago") while `+0` is future.
    let is_past = value.is_sign_negative();
    let mut parts: alloc::vec::Vec<(&'static str, alloc::string::String, bool)> =
        alloc::vec::Vec::new();
    if is_past {
        parts.extend(rel_time_number_parts(n));
        parts.push(("literal", alloc::format!(" {unit_disp} ago"), false));
    } else {
        parts.push(("literal", alloc::string::String::from("in "), false));
        parts.extend(rel_time_number_parts(n));
        parts.push(("literal", alloc::format!(" {unit_disp}"), false));
    }
    parts
}

/// Segments `input` per an `Intl.Segmenter` granularity, returning `(index, segment,
/// isWordLike)` triples (`index` is a code-point offset). `grapheme` is per code point;
/// `word` alternates alphanumeric "word-like" runs with separators; `sentence` splits after
/// terminating punctuation followed by a space.
#[cfg(feature = "intl")]
fn segment_text(
    input: &str,
    granularity: &str,
) -> Vec<(usize, alloc::string::String, Option<bool>)> {
    use intl::unicode::segment;
    // `index` is a code-point offset (kataan strings index by code point, e.g.
    // `"\u{1F600}".length === 1`), so accumulate `chars().count()` per segment.
    let mut out: Vec<(usize, alloc::string::String, Option<bool>)> = Vec::new();
    let mut index = 0usize;
    let push = |seg: &str, is_word_like: Option<bool>, out: &mut Vec<_>, index: &mut usize| {
        out.push((*index, alloc::string::String::from(seg), is_word_like));
        *index += seg.chars().count();
    };
    match granularity {
        "word" => {
            for w in segment::words(input) {
                let wl = Some(w.chars().any(char::is_alphanumeric));
                push(w, wl, &mut out, &mut index);
            }
        }
        "sentence" => {
            for s in segment::sentences(input) {
                push(s, None, &mut out, &mut index);
            }
        }
        // "grapheme" (default): UAX-29 extended grapheme clusters.
        _ => {
            for g in segment::graphemes(input) {
                push(g, None, &mut out, &mut index);
            }
        }
    }
    out
}

/// Hand-rolled en-US fallback used when the `intl` crate is unavailable: per-code-point
/// graphemes, alphanumeric word runs, and `.`/`!`/`?`-plus-space sentence splits.
#[cfg(not(feature = "intl"))]
fn segment_text(
    input: &str,
    granularity: &str,
) -> Vec<(usize, alloc::string::String, Option<bool>)> {
    let chars: Vec<char> = input.chars().collect();
    let mut out: Vec<(usize, alloc::string::String, Option<bool>)> = Vec::new();
    // Approximate UAX-29 word boundaries: group runs of one class — alphanumeric (word-like),
    // whitespace, or other — so punctuation and spaces become distinct segments (matching the
    // `intl` crate, e.g. `","` and `" "` split apart).
    let class = |c: char| -> u8 {
        if c.is_alphanumeric() {
            0
        } else if c.is_whitespace() {
            1
        } else {
            2
        }
    };
    match granularity {
        "word" => {
            let mut i = 0;
            while i < chars.len() {
                let cls = class(chars[i]);
                let start = i;
                while i < chars.len() && class(chars[i]) == cls {
                    i += 1;
                }
                out.push((start, chars[start..i].iter().collect(), Some(cls == 0)));
            }
        }
        "sentence" => {
            let mut start = 0;
            let mut i = 0;
            while i < chars.len() {
                let c = chars[i];
                i += 1;
                // End a sentence after `.`/`!`/`?` followed by whitespace (or end).
                if matches!(c, '.' | '!' | '?') {
                    while i < chars.len() && chars[i].is_whitespace() {
                        i += 1;
                    }
                    out.push((start, chars[start..i].iter().collect(), None));
                    start = i;
                }
            }
            if start < chars.len() {
                out.push((start, chars[start..].iter().collect(), None));
            }
        }
        // "grapheme" (default): one code point per segment.
        _ => {
            for (i, c) in chars.iter().enumerate() {
                out.push((i, alloc::string::String::from(*c), None));
            }
        }
    }
    out
}

/// `Intl.DisplayNames.prototype.of(code)` for the `language`/`region`/`currency`/`script`
/// types (a common en subset). An unrecognized code falls back to itself.
fn display_name(ty: &str, code: &str) -> alloc::string::String {
    let owned;
    let name: &str = match ty {
        "language" => {
            // The primary language subtag, lowercased.
            let primary = code.split(['-', '_']).next().unwrap_or(code);
            owned = primary.to_ascii_lowercase();
            match owned.as_str() {
                "en" => "English",
                "fr" => "French",
                "de" => "German",
                "es" => "Spanish",
                "it" => "Italian",
                "pt" => "Portuguese",
                "nl" => "Dutch",
                "ru" => "Russian",
                "ja" => "Japanese",
                "zh" => "Chinese",
                "ko" => "Korean",
                "ar" => "Arabic",
                "hi" => "Hindi",
                "tr" => "Turkish",
                "pl" => "Polish",
                "sv" => "Swedish",
                "el" => "Greek",
                "he" => "Hebrew",
                "th" => "Thai",
                "vi" => "Vietnamese",
                _ => code,
            }
        }
        "region" => {
            owned = code.to_ascii_uppercase();
            match owned.as_str() {
                "US" => "United States",
                "GB" => "United Kingdom",
                "FR" => "France",
                "DE" => "Germany",
                "ES" => "Spain",
                "IT" => "Italy",
                "PT" => "Portugal",
                "NL" => "Netherlands",
                "RU" => "Russia",
                "JP" => "Japan",
                "CN" => "China",
                "KR" => "South Korea",
                "IN" => "India",
                "BR" => "Brazil",
                "CA" => "Canada",
                "AU" => "Australia",
                "MX" => "Mexico",
                "CH" => "Switzerland",
                "SE" => "Sweden",
                "GR" => "Greece",
                _ => code,
            }
        }
        "currency" => {
            owned = code.to_ascii_uppercase();
            match owned.as_str() {
                "USD" => "US Dollar",
                "EUR" => "Euro",
                "GBP" => "British Pound",
                "JPY" => "Japanese Yen",
                "CNY" => "Chinese Yuan",
                "CHF" => "Swiss Franc",
                "CAD" => "Canadian Dollar",
                "AUD" => "Australian Dollar",
                "INR" => "Indian Rupee",
                "BRL" => "Brazilian Real",
                "RUB" => "Russian Ruble",
                "KRW" => "South Korean Won",
                "MXN" => "Mexican Peso",
                _ => code,
            }
        }
        _ => code,
    };
    alloc::string::String::from(name)
}

/// The CLDR "short" symbol for an `Intl.NumberFormat` `style: "unit"` measurement unit
/// (a common subset). An unrecognized unit renders by its own name. (Used for `style: "unit"`
/// in both builds — the `intl` crate's `number::format` doesn't render units yet.)
fn unit_symbol(unit: &str) -> &str {
    match unit {
        "kilometer" => "km",
        "meter" => "m",
        "centimeter" => "cm",
        "millimeter" => "mm",
        "mile" => "mi",
        "foot" => "ft",
        "inch" => "in",
        "yard" => "yd",
        "kilogram" => "kg",
        "gram" => "g",
        "milligram" => "mg",
        "pound" => "lb",
        "ounce" => "oz",
        "liter" => "L",
        "milliliter" => "mL",
        "gallon" => "gal",
        "second" => "s",
        "millisecond" => "ms",
        "minute" => "min",
        "hour" => "h",
        "day" => "d",
        "week" => "wk",
        "month" => "mth",
        "year" => "yr",
        "celsius" => "°C",
        "fahrenheit" => "°F",
        "byte" => "byte",
        "kilobyte" => "kB",
        "megabyte" => "MB",
        "gigabyte" => "GB",
        "terabyte" => "TB",
        "bit" => "bit",
        "percent" => "%",
        "degree" => "deg",
        "liter-per-100-kilometer" => "L/100km",
        other => other,
    }
}

/// Little-endian encode of one already-coerced typed-array element value of `kind`
/// (index into [`TYPED_ARRAY_KINDS`]) — the inverse of [`decode_typed_element`].
/// Writes into a fixed `[u8; 8]` (no per-element heap allocation) and returns the
/// buffer alongside the number of bytes actually written (`0` for an unknown kind).
/// Callers use `&buf[..n]` as the encoded element.
pub(crate) fn encode_typed_element(kind: u8, v: f64) -> ([u8; 8], usize) {
    let mut out = [0u8; 8];
    let n = match kind {
        0 => {
            out[0] = (v as i64 as i8) as u8; // Int8
            1
        }
        1 => {
            out[0] = v as i64 as u8; // Uint8
            1
        }
        2 => {
            out[0] = v.clamp(0.0, 255.0) as u8; // Uint8Clamped (already integral)
            1
        }
        3 => {
            out[..2].copy_from_slice(&(v as i64 as i16).to_le_bytes()); // Int16
            2
        }
        4 => {
            out[..2].copy_from_slice(&(v as i64 as u16).to_le_bytes()); // Uint16
            2
        }
        5 => {
            out[..4].copy_from_slice(&(v as i64 as i32).to_le_bytes()); // Int32
            4
        }
        6 => {
            out[..4].copy_from_slice(&(v as i64 as u32).to_le_bytes()); // Uint32
            4
        }
        7 => {
            out[..4].copy_from_slice(&(v as f32).to_le_bytes()); // Float32
            4
        }
        8 => {
            out.copy_from_slice(&v.to_le_bytes()); // Float64
            8
        }
        11 => {
            out[..2].copy_from_slice(&f64_to_f16_bits(v).to_le_bytes()); // Float16
            2
        }
        _ => 0,
    };
    (out, n)
}

/// Little-endian decode of one typed-array element of `kind` (index into
/// [`TYPED_ARRAY_KINDS`]) from `bytes` (short/empty slices read as zero).
pub(crate) fn decode_typed_element(kind: u8, bytes: &[u8]) -> f64 {
    let b = |i: usize| bytes.get(i).copied().unwrap_or(0);
    match kind {
        0 => f64::from(b(0) as i8),                                   // Int8
        1 | 2 => f64::from(b(0)),                                     // Uint8 / Clamped
        3 => f64::from(i16::from_le_bytes([b(0), b(1)])),             // Int16
        4 => f64::from(u16::from_le_bytes([b(0), b(1)])),             // Uint16
        5 => f64::from(i32::from_le_bytes([b(0), b(1), b(2), b(3)])), // Int32
        6 => f64::from(u32::from_le_bytes([b(0), b(1), b(2), b(3)])), // Uint32
        7 => f64::from(f32::from_le_bytes([b(0), b(1), b(2), b(3)])), // Float32
        8 => f64::from_le_bytes([b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7)]), // Float64
        11 => f16_to_f64(u16::from_le_bytes([b(0), b(1)])),           // Float16
        // BigInt kinds (9/10) do not decode to an f64 — use `decode_bigint_element`.
        _ => 0.0,
    }
}

/// Little-endian decode of one **BigInt** typed-array element of `kind`
/// (9 = `BigInt64Array`, signed i64; 10 = `BigUint64Array`, unsigned u64) from
/// `bytes` (short/empty slices read as zero), as an arbitrary-precision
/// [`BigInt`](crate::bignum::BigInt).
pub(crate) fn decode_bigint_element(kind: u8, bytes: &[u8]) -> crate::bignum::BigInt {
    use crate::bignum::BigInt;
    let b = |i: usize| bytes.get(i).copied().unwrap_or(0);
    let raw = u64::from_le_bytes([b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7)]);
    if kind == 9 {
        BigInt::from_i128(i128::from(raw as i64)) // signed reinterpretation
    } else {
        BigInt::from_i128(i128::from(raw)) // unsigned
    }
}

/// Little-endian encode of the low 64 bits of a [`BigInt`](crate::bignum::BigInt)
/// into an 8-byte buffer — the element encoding shared by `BigInt64Array` and
/// `BigUint64Array` (`ToBigInt64` / `ToBigUint64` keep only the low 64 bits).
pub(crate) fn encode_bigint_element(value: &crate::bignum::BigInt) -> [u8; 8] {
    value.to_u64_wrapping().to_le_bytes()
}

/// Maps an ISO-4217 currency code to its symbol for `style: "currency"` (a small
/// common set; an unknown code is rendered as `CODE\u{00a0}`, like Intl's fallback).
fn currency_symbol(code: &str) -> String {
    let sym = match code {
        "USD" | "AUD" | "CAD" | "NZD" | "HKD" | "SGD" | "MXN" => "$",
        "EUR" => "€",
        "GBP" => "£",
        "JPY" | "CNY" => "¥",
        "INR" => "₹",
        "KRW" => "₩",
        "RUB" => "₽",
        "BRL" => "R$",
        "CHF" => "CHF\u{00a0}",
        "" => "",
        other => return alloc::format!("{other}\u{00a0}"),
    };
    String::from(sym)
}

fn group_thousands(n: f64) -> String {
    if n.is_nan() {
        return String::from("NaN");
    }
    if n.is_infinite() {
        return String::from(if n > 0.0 { "∞" } else { "-∞" });
    }
    let neg = n.is_sign_negative() && n != 0.0;
    // `n.abs()` maps -0 to +0 so it renders as "0", not "-0".
    let base = alloc::format!("{}", n.abs());
    let grouped = group_thousands_str(&base);
    if neg {
        alloc::format!("-{grouped}")
    } else {
        grouped
    }
}

/// Groups the integer part of a decimal digit string (optional leading `-` and
/// fractional `.NNN`) with `,` thousands separators — shared by `Number`/`BigInt`
/// `toLocaleString`.
fn group_thousands_str(s: &str) -> String {
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let (int_part, frac_part) = match rest.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (rest, None),
    };
    let bytes = int_part.as_bytes();
    let len = bytes.len();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    if let Some(f) = frac_part {
        out.push('.');
        out.push_str(f);
    }
    out
}

/// Case-maps a WTF-8 string, preserving lone surrogates verbatim (a surrogate
/// code point has no case). A surrogate-free string takes the `&str` fast path
/// — byte-identical to `str::to_uppercase`/`to_lowercase`, including full
/// (multi-`char`) mappings like `ß`→`SS`. A surrogate-bearing string maps each
/// scalar code point with `char::to_uppercase`/`to_lowercase` and re-emits any
/// surrogate code point unchanged, building the result as WTF-8.
fn case_map_wtf8(bytes: &[u8], upper: bool) -> alloc::vec::Vec<u8> {
    // Fast path: no surrogates → valid UTF-8 → the standard `&str` mapping.
    if let Some(s) = crate::wtf8::as_str(bytes) {
        let mapped = if upper {
            s.to_uppercase()
        } else {
            s.to_lowercase()
        };
        return mapped.into_bytes();
    }
    let mut out = alloc::vec::Vec::with_capacity(bytes.len());
    for cp in crate::wtf8::code_points(bytes) {
        match char::from_u32(cp) {
            // A scalar value: apply the Unicode case mapping (a one-to-many
            // mapping such as `ß`→`SS` expands here too).
            Some(c) => {
                if upper {
                    for u in c.to_uppercase() {
                        crate::wtf8::encode_code_point(u32::from(u), &mut out);
                    }
                } else {
                    for u in c.to_lowercase() {
                        crate::wtf8::encode_code_point(u32::from(u), &mut out);
                    }
                }
            }
            // A lone surrogate code point: no case — pass it through unchanged.
            None => crate::wtf8::encode_code_point(cp, &mut out),
        }
    }
    out
}

/// Unicode-normalizes a WTF-8 string (`form` is one of `NFC`/`NFD`/`NFKC`/
/// `NFKD`, validated by the caller), preserving lone surrogates in place.
/// Normalization is the identity on a surrogate code point, so a surrogate-free
/// string takes the `&str` fast path (byte-identical to the scalar normalizer),
/// while a surrogate-bearing string normalizes each maximal run of scalars and
/// re-emits each lone surrogate unchanged. The result is WTF-8.
#[cfg(feature = "intl")]
fn normalize_wtf8(bytes: &[u8], form: &str) -> alloc::vec::Vec<u8> {
    use intl::unicode::normalize;
    let norm = |chars: core::str::Chars<'_>| -> String {
        match form {
            "NFC" => normalize::nfc(chars).collect(),
            "NFD" => normalize::nfd(chars).collect(),
            "NFKC" => normalize::nfkc(chars).collect(),
            // The caller validated `form`, so the remaining case is `NFKD`.
            _ => normalize::nfkd(chars).collect(),
        }
    };
    // Fast path: no surrogates → one scalar run.
    if let Some(s) = crate::wtf8::as_str(bytes) {
        return norm(s.chars()).into_bytes();
    }
    let mut out: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(bytes.len());
    // A buffer of consecutive scalar code points, flushed (normalized) whenever a
    // lone surrogate interrupts the run.
    let mut run = String::new();
    for cp in crate::wtf8::code_points(bytes) {
        match char::from_u32(cp) {
            Some(c) => run.push(c),
            None => {
                if !run.is_empty() {
                    out.extend_from_slice(norm(run.chars()).as_bytes());
                    run.clear();
                }
                crate::wtf8::encode_code_point(cp, &mut out);
            }
        }
    }
    if !run.is_empty() {
        out.extend_from_slice(norm(run.chars()).as_bytes());
    }
    out
}

/// Slices a pre-collected `&[u16]` subject over the **code-unit** range
/// `[st, en)` and re-encodes it to WTF-8 bytes (lone surrogates preserved). The
/// native regex subject model is UTF-16 code units, so match/capture spans index
/// this buffer directly (RE-7: the subject is collected once per operation). Also
/// used by the spec-string-method (`@@split`/`@@replace`) builders, which exist
/// without the `regex` feature, so it is not feature-gated.
fn u16_slice(units: &[u16], st: usize, en: usize) -> alloc::vec::Vec<u8> {
    let st = st.min(units.len());
    let en = en.min(units.len()).max(st);
    crate::wtf8::from_utf16(&units[st..en])
}

/// Slices a pre-collected `&[u16]` subject from code-unit index `st` to the end,
/// re-encoded to WTF-8 bytes.
fn u16_slice_from(units: &[u16], st: usize) -> alloc::vec::Vec<u8> {
    crate::wtf8::from_utf16(&units[st.min(units.len())..])
}

/// Advances a code-unit position past a just-consumed empty match. Per spec
/// `AdvanceStringIndex`, a `u`-flag regex steps a whole code point (skipping the
/// low half of a surrogate pair), while a non-`u` regex steps one code unit.
fn advance_index_u16(units: &[u16], i: usize, unicode: bool) -> usize {
    if unicode
        && i + 1 < units.len()
        && (0xD800..=0xDBFF).contains(&units[i])
        && (0xDC00..=0xDFFF).contains(&units[i + 1])
    {
        i + 2
    } else {
        i + 1
    }
}

/// Rounds an `f64` to the nearest IEEE-754 binary16 value, returning its 16-bit
/// pattern. Uses round-to-nearest-ties-to-even, with correct subnormal and
/// overflow-to-infinity handling. (Rust has no stable `f16`.) Pure bit math, so
/// it is `core`-friendly (no `std` float intrinsics).
fn f64_to_f16_bits(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    if value.is_nan() {
        return sign | 0x7E00; // a quiet NaN
    }
    // `abs` via clearing the sign bit (std `f64::abs` is unavailable under no_std).
    let abs = f64::from_bits(bits & 0x7FFF_FFFF_FFFF_FFFF);
    if abs.is_infinite() {
        return sign | 0x7C00;
    }
    if abs == 0.0 {
        return sign;
    }
    // f64 unbiased exponent and 52-bit mantissa, with the implicit leading 1 made
    // explicit to form a 53-bit significand whose binary point sits after bit 52.
    let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
    let signif = 0x0010_0000_0000_0000u64 | (bits & 0x000F_FFFF_FFFF_FFFF);
    // We want a 11-bit half significand (implicit 1 + 10 fraction). The number is
    // signif * 2^(exp - 52). To express it as (half-significand) * 2^(half-exp),
    // we shift the 53-bit significand right so that bit 10 holds the leading 1 for
    // a normal result, or further for a subnormal one. `drop` is how many low bits
    // are discarded (and rounded on).
    // For a normal binary16 the exponent field is `exp + 15` in `[1, 30]`.
    if exp + 15 >= 0x1F {
        return sign | 0x7C00; // overflow → ±Infinity
    }
    // `drop` bits are removed from the 53-bit significand. In the normal range the
    // leading 1 must land at bit 10, i.e. drop = 52 - 10 = 42. Each step the half
    // exponent decreases below 1 (into the subnormal range) drops one more bit.
    let drop: i64 = if exp + 15 >= 1 {
        42
    } else {
        // Subnormal: shift extra by (1 - (exp + 15)) = -14 - exp.
        42 + (1 - (exp + 15))
    };
    if drop >= 64 {
        return sign; // underflow to ±0
    }
    let drop = drop as u32;
    let q = signif >> drop;
    let rem = signif & ((1u64 << drop) - 1);
    let half = 1u64 << (drop - 1);
    let mut out = q;
    // Round to nearest, ties to even.
    if rem > half || (rem == half && (q & 1) == 1) {
        out += 1;
    }
    // For a normal result `out` now holds the implicit-1 significand at bit 10; the
    // exponent field must be added in. A rounding carry that pushes `out` to
    // 0x800 (bit 11) correctly bumps the exponent. For a subnormal result the
    // exponent field is 0 and `out` is the fraction (a carry to 0x400 promotes it
    // to the smallest normal, which is also correct).
    if exp + 15 >= 1 {
        // Re-add the biased exponent, subtracting the implicit-1 bit already in
        // `out` (bit 10) by masking it off and combining with the exponent field.
        let exp_field = (exp + 15) as u64;
        // `out` includes the implicit leading 1 at bit 10; the half format stores
        // exponent in bits 14..10 and fraction in bits 9..0, with the leading 1
        // implicit — so combine (exp_field << 10) with the low 10 fraction bits,
        // accounting for any carry already folded into `out`.
        let combined = (exp_field << 10) + (out - 0x400);
        sign | combined as u16
    } else {
        sign | out as u16
    }
}

/// Expands a binary16 bit pattern to the `f64` it represents. `core`-friendly:
/// powers of two are built directly from the IEEE-754 exponent field rather than
/// via the std-only `f64::powi`.
fn f16_to_f64(h: u16) -> f64 {
    // 2^n for n in the binary16 range, by constructing the f64 exponent field.
    fn pow2(n: i32) -> f64 {
        f64::from_bits(((1023 + n) as u64) << 52)
    }
    let sign = if (h & 0x8000) != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1F;
    let mant = (h & 0x03FF) as f64;
    match exp {
        0 => sign * mant * pow2(-24), // subnormal (and ±0 when mant == 0)
        0x1F => {
            if mant == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + mant / 1024.0) * pow2(exp as i32 - 15),
    }
}

/// Formats a calendar year for `toDateString`/`toUTCString`/`toString`: a
/// non-negative year is zero-padded to at least 4 digits (`0020`, `2020`); a
/// negative year prints a `-` then its magnitude zero-padded to at least 4
/// digits (`-0001`, `-123456`).
fn format_date_year(y: i64) -> String {
    if y < 0 {
        alloc::format!("-{:04}", -y)
    } else {
        alloc::format!("{y:04}")
    }
}

/// Truncates `n` toward zero without the std-only `f64::trunc` intrinsic (kept
/// available in `no_std`). `NaN`/`±Infinity` and magnitudes beyond `i64` range
/// (already integral) pass through unchanged.
fn trunc_toward_zero(n: f64) -> f64 {
    if !n.is_finite() || n.abs() >= 9_223_372_036_854_775_808.0 {
        n
    } else {
        n as i64 as f64
    }
}

/// `TimeClip(t)`: `NaN` for a non-finite value or a magnitude beyond the maximum
/// representable time (8.64e15 ms ≈ ±100,000,000 days), otherwise the integer
/// part (truncated toward zero, normalizing `-0` to `+0`).
fn time_clip(t: f64) -> f64 {
    if !t.is_finite() || t.abs() > 8.64e15 {
        return f64::NAN;
    }
    let truncated = trunc_toward_zero(t);
    if truncated == 0.0 { 0.0 } else { truncated }
}

fn int_to_radix(n: f64, radix: u32) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let neg = n < 0.0;
    let abs = if neg { -n } else { n };
    // Integer part (the `as u64` cast truncates toward zero — no `std` float math).
    let mut v = abs as u64;
    let mut ibuf = Vec::new();
    if v == 0 {
        ibuf.push(b'0');
    }
    while v > 0 {
        ibuf.push(DIGITS[(v % radix as u64) as usize]);
        v /= radix as u64;
    }
    ibuf.reverse();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    out.push_str(&String::from_utf8(ibuf).unwrap_or_default());
    // Fractional part (bounded digit count to terminate on repeating fractions).
    let mut frac = abs - (abs as u64) as f64;
    if frac > 0.0 {
        out.push('.');
        for _ in 0..20 {
            if frac <= 0.0 {
                break;
            }
            frac *= radix as f64;
            let digit = (frac as usize).min(radix as usize - 1);
            out.push(DIGITS[digit] as char);
            frac -= digit as f64;
        }
    }
    out
}

/// Parses the longest leading decimal-float prefix of `s` (à la `parseFloat`),
/// returning `NaN` if none.
const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `bytes` to a standard (`+`/`/`, `=`-padded) base64 string.
fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[(n >> 18 & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[(n >> 12 & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[(n >> 6 & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[(n & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Decodes a base64 string (ASCII whitespace ignored), returning `None` on an
/// invalid character or length.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let cleaned: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(cleaned.len() / 4 * 3);
    for chunk in cleaned.chunks(4) {
        if chunk.len() < 2 {
            return None;
        }
        let mut n = 0u32;
        for &c in chunk {
            n = (n << 6) | val(c)?;
        }
        // Left-align the partial group, then take the available bytes.
        n <<= 6 * (4 - chunk.len());
        out.push((n >> 16 & 0xff) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8 & 0xff) as u8);
        }
        if chunk.len() > 3 {
            out.push((n & 0xff) as u8);
        }
    }
    Some(out)
}

/// Percent-encodes `s`. The unreserved set (`A-Za-z0-9-_.!~*'()`) is always kept;
/// `extra` adds characters preserved by `encodeURI` (the URI reserved set).
fn uri_encode(s: &str, extra: &str) -> String {
    let mut out = String::new();
    let mut buf = [0u8; 4];
    for ch in s.chars() {
        let keep = ch.is_ascii_alphanumeric() || "-_.!~*'()".contains(ch) || extra.contains(ch);
        if keep {
            out.push(ch);
        } else {
            for b in ch.encode_utf8(&mut buf).bytes() {
                out.push('%');
                out.push(
                    char::from_digit((b >> 4) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
                out.push(
                    char::from_digit((b & 0xf) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    out
}

/// Decodes percent-escapes in `s` (`%XX` → byte), returning `None` on a malformed
/// escape or invalid UTF-8.
fn uri_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = (*bytes.get(i + 1)? as char).to_digit(16)?;
            let lo = (*bytes.get(i + 2)? as char).to_digit(16)?;
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `escape(string)` (Annex B.2.1.1). Operates on the UTF-16 code units of the
/// WTF-8 `bytes`: a unit in the unescaped set (`A-Za-z0-9` plus `@*_+-./`) is
/// kept; a unit `< 256` becomes `%XX`; any larger unit becomes `%uXXXX`. The
/// result is pure ASCII, so its WTF-8 form is its UTF-8 form.
fn legacy_escape(bytes: &[u8]) -> Vec<u8> {
    fn hex(n: u32) -> u8 {
        char::from_digit(n, 16).unwrap().to_ascii_uppercase() as u8
    }
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    for u in crate::wtf8::utf16_units(bytes) {
        let keep = matches!(u, 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A)
            || matches!(u as u8 as char, '@' | '*' | '_' | '+' | '-' | '.' | '/') && u < 0x80;
        if keep {
            out.push(u as u8);
        } else if u < 256 {
            out.push(b'%');
            out.push(hex((u as u32) >> 4));
            out.push(hex((u as u32) & 0xF));
        } else {
            out.push(b'%');
            out.push(b'u');
            out.push(hex((u as u32) >> 12));
            out.push(hex(((u as u32) >> 8) & 0xF));
            out.push(hex(((u as u32) >> 4) & 0xF));
            out.push(hex((u as u32) & 0xF));
        }
    }
    out
}

/// `unescape(string)` (Annex B.2.2.1). The inverse of [`legacy_escape`], over the
/// UTF-16 code units of `bytes`: `%uXXXX` decodes to a single unit, `%XX` to a
/// unit `< 256`; an incomplete or non-hex escape is left verbatim. The rebuilt
/// units are re-encoded to WTF-8 (so a decoded surrogate is preserved).
fn legacy_unescape(bytes: &[u8]) -> Vec<u8> {
    let units: Vec<u16> = crate::wtf8::utf16_units(bytes).collect();
    let hex4 = |s: &[u16]| -> Option<u16> {
        let mut v: u32 = 0;
        for &u in s {
            let d = char::from_u32(u32::from(u)).and_then(|c| c.to_digit(16))?;
            v = v * 16 + d;
        }
        Some(v as u16)
    };
    let mut out: Vec<u16> = Vec::with_capacity(units.len());
    let mut i = 0;
    while i < units.len() {
        if units[i] == u16::from(b'%') {
            if i + 5 < units.len()
                && units[i + 1] == u16::from(b'u')
                && let Some(v) = hex4(&units[i + 2..i + 6])
            {
                out.push(v);
                i += 6;
                continue;
            }
            if i + 2 < units.len()
                && let Some(v) = hex4(&units[i + 1..i + 3])
            {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(units[i]);
        i += 1;
    }
    crate::wtf8::from_utf16(&out)
}

fn parse_float_prefix(s: &str) -> f64 {
    // A leading (optionally signed) `Infinity`.
    let (sign, rest) = match s.strip_prefix('-') {
        Some(r) => (-1.0, r),
        None => (1.0, s.strip_prefix('+').unwrap_or(s)),
    };
    if rest.starts_with("Infinity") {
        return sign * f64::INFINITY;
    }
    let bytes = s.as_bytes();
    let mut end = 0;
    let mut seen_dot = false;
    let mut seen_e = false;
    while end < bytes.len() {
        let ch = bytes[end] as char;
        let ok = match ch {
            '0'..='9' => true,
            '+' | '-' if end == 0 || matches!(bytes[end - 1] as char, 'e' | 'E') => true,
            '.' if !seen_dot && !seen_e => {
                seen_dot = true;
                true
            }
            'e' | 'E' if !seen_e && end > 0 => {
                seen_e = true;
                true
            }
            _ => false,
        };
        if !ok {
            break;
        }
        end += 1;
    }
    s[..end].parse::<f64>().unwrap_or(f64::NAN)
}

/// Advances `pos` past JSON whitespace.
fn skip_ws(c: &[char], pos: &mut usize) {
    while c
        .get(*pos)
        .is_some_and(|ch| matches!(ch, ' ' | '\t' | '\n' | '\r'))
    {
        *pos += 1;
    }
}

/// Parses and runs `source` on the new representation, returning the captured
/// `console` output and the program's completion value (as a display string).
///
/// This is the high-level entry point to the new-model engine — the bridge the
/// production pipeline migrates onto.
///
/// # Errors
/// Returns a parse or execution error message on failure.
pub fn eval_source(source: &str) -> Result<(String, String), String> {
    eval_source_with_limits(source, crate::limits::Limits::default())
}

/// Like [`eval_source`], but with caller-supplied resource
/// [`Limits`](crate::limits::Limits).
///
/// # Errors
/// Returns a parse or execution error message on failure.
pub fn eval_source_with_limits(
    source: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), String> {
    let program =
        crate::parser::Parser::parse_program(source).map_err(|e| alloc::format!("{e}"))?;
    let mut interp = Interp::new_with_limits(limits);
    let value = match interp.run(&program) {
        Ok(v) => v,
        // Render an uncaught throw readably: an error object as `name: message`,
        // any other thrown value via its display string.
        Err(ExecError::Throw(thrown)) => return Err(format_thrown(&interp, thrown)),
        Err(other) => return Err(alloc::format!("{other:?}")),
    };
    let completion = interp.display(value);
    Ok((String::from(interp.output()), completion))
}

/// Formats an uncaught thrown value for an error message: `name: message` for an
/// error-shaped object, otherwise the value's display string.
fn format_thrown(interp: &Interp, thrown: NanBox) -> String {
    if let Some((name, message)) = error_name_message(interp, thrown) {
        return if message.is_empty() {
            name
        } else {
            alloc::format!("{name}: {message}")
        };
    }
    // An error-shaped throw lacking a `name` property (e.g. Test262Error, which
    // carries only `message` and a custom `toString`): surface its `message` so
    // uncaught throws remain diagnosable rather than rendering `[object Object]`.
    if let Some(raw) = thrown.as_handle() {
        let h = Handle::from_raw(raw);
        if let Some(m) = interp.realm().get_property(h, "message") {
            let s = interp.realm().to_display_string(m);
            if !s.is_empty() {
                return alloc::format!("Test262Error: {s}");
            }
        }
    }
    interp.display(thrown)
}

/// Extracts `(name, message)` from an error-shaped thrown object — the basis for
/// both the human-readable [`format_thrown`] and the structured [`Thrown`] the
/// conformance runner uses to verify a negative test's declared error *type*.
/// Returns `None` for a non-error thrown value (e.g. `throw 42`).
pub(crate) fn error_name_message(interp: &Interp, thrown: NanBox) -> Option<(String, String)> {
    let raw = thrown.as_handle()?;
    let h = Handle::from_raw(raw);
    let realm = interp.realm();
    // `realm.get_property` reads *own* properties only, so walk the prototype
    // chain by hand to find the first `key` (like an ordinary `[[Get]]`).
    let inherited = |key: &str| -> Option<NanBox> {
        let mut cur = Some(h);
        while let Some(c) = cur {
            if let Some(v) = realm.get_property(c, key) {
                return Some(v);
            }
            cur = realm.object_proto(c);
        }
        None
    };
    let message = inherited("message")
        .map(|m| realm.to_display_string(m))
        .unwrap_or_default();
    // The error's `name` — usually inherited from `Error.prototype.name`.
    if let Some(name) = inherited("name") {
        return Some((realm.to_display_string(name), message));
    }
    // A user-defined error class that omits `name` (notably the Test262 harness's
    // `Test262Error`, which defines only `message` + a custom `toString`): fall
    // back to its constructor's name, so a negative test can still verify the
    // thrown *type* (and an uncaught throw renders as `Name: message`, not the
    // opaque `[object Object]`). A throw whose object has neither is not
    // error-shaped → `None`. The constructor's own `name` is usually stored; a
    // function/class `name` computed on read is synthesized from its definition.
    let ctor = inherited("constructor")
        .and_then(|c| c.as_handle())
        .map(Handle::from_raw)?;
    let ctor_name = if let Some(n) = realm.get_property(ctor, "name") {
        realm.to_display_string(n)
    } else if let Some((cid, _)) = realm.class_at(ctor) {
        interp.classes[cid as usize]
            .id
            .as_ref()
            .map_or_else(String::new, |i| String::from(&*i.name))
    } else if let Some((fid, _)) = realm.function_at(ctor) {
        String::from(interp.functions[fid as usize].name)
    } else {
        String::new()
    };
    if ctor_name.is_empty() {
        return None;
    }
    Some((ctor_name, message))
}

/// The phase at which a program failed: parsing, or runtime execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorPhase {
    /// The program failed to parse (always a `SyntaxError`).
    Parse,
    /// The program threw during execution.
    Runtime,
}

/// A thrown error surfaced to the host with its JS *type*, so a conformance
/// runner can check a Test262 `negative: { phase, type }` expectation. For an
/// error-shaped throw `name` is the constructor name (`"TypeError"`, …); for a
/// non-error throw (`throw 42`) it is the value's display string.
#[derive(Debug, Clone)]
pub struct Thrown {
    /// Whether the failure occurred at parse time or runtime.
    pub phase: ErrorPhase,
    /// The error's `name` (its JS type), e.g. `"TypeError"` or `"SyntaxError"`.
    pub name: String,
    /// The error's `message` (empty when absent).
    pub message: String,
}

/// Like [`eval_source_with_limits`], but on failure returns a structured
/// [`Thrown`] carrying the error's *type* (for Test262 negative-test checking)
/// instead of a flattened message string.
///
/// # Errors
/// Returns [`Thrown`] for a parse failure (`SyntaxError`) or an uncaught throw.
pub fn eval_source_typed(
    source: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), Thrown> {
    let program = match crate::parser::Parser::parse_program(source) {
        Ok(p) => p,
        Err(e) => {
            return Err(Thrown {
                phase: ErrorPhase::Parse,
                name: String::from("SyntaxError"),
                message: alloc::format!("{e}"),
            });
        }
    };
    // `parse_program` infers the goal symbol, promoting a unit with a top-level
    // `import`/`export` to a Module. Run as a *Script* (this entry), such a
    // declaration is an early SyntaxError — `import`/`export` are legal only at a
    // Module's top level. (Module tests go through the module loader instead.)
    if program.source_type == crate::ast::SourceType::Module {
        return Err(Thrown {
            phase: ErrorPhase::Parse,
            name: String::from("SyntaxError"),
            message: String::from("`import`/`export` may only appear at the top level of a module"),
        });
    }
    let mut interp = Interp::new_with_limits(limits);
    match interp.run(&program) {
        Ok(value) => {
            let completion = interp.display(value);
            Ok((String::from(interp.output()), completion))
        }
        Err(ExecError::Throw(thrown)) => {
            let (name, message) = error_name_message(&interp, thrown).unwrap_or_else(|| {
                // A throw lacking a `name` property (e.g. Test262Error, which carries
                // only `message`): surface its `message` so the failure is
                // diagnosable rather than the opaque `[object Object]`.
                if let Some(raw) = thrown.as_handle()
                    && let Some(m) = interp
                        .realm()
                        .get_property(Handle::from_raw(raw), "message")
                {
                    let s = interp.realm().to_display_string(m);
                    if !s.is_empty() {
                        return (String::from("Test262Error"), s);
                    }
                }
                (interp.display(thrown), String::new())
            });
            Err(Thrown {
                phase: ErrorPhase::Runtime,
                name,
                message,
            })
        }
        Err(other) => Err(Thrown {
            phase: ErrorPhase::Runtime,
            name: String::from("Error"),
            message: alloc::format!("{other:?}"),
        }),
    }
}

/// The current time in milliseconds since the Unix epoch (`0.0` without `std`,
/// which has no clock).
fn now_ms() -> f64 {
    #[cfg(feature = "std")]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0)
    }
    #[cfg(not(feature = "std"))]
    {
        0.0
    }
}

/// A minimal `parseInt`: skips leading whitespace, reads an optional sign and
/// the leading decimal digits, and returns `NaN` if there are none.
fn parse_int(s: &str, radix: u32) -> f64 {
    let mut t = s.trim_start();
    let mut neg = false;
    if let Some(rest) = t.strip_prefix('-') {
        neg = true;
        t = rest;
    } else if let Some(rest) = t.strip_prefix('+') {
        t = rest;
    }
    // Radix 0 means infer: `0x` → 16, else 10. A `0x` prefix is also honored
    // when radix is explicitly 16.
    let mut radix = radix;
    if (radix == 0 || radix == 16)
        && let Some(rest) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))
    {
        t = rest;
        radix = 16;
    }
    if radix == 0 {
        radix = 10;
    }
    if !(2..=36).contains(&radix) {
        return f64::NAN;
    }
    // Consume the leading digits valid in this radix.
    let mut value: f64 = 0.0;
    let mut any = false;
    for c in t.chars() {
        match c.to_digit(radix) {
            Some(d) => {
                value = value * f64::from(radix) + f64::from(d);
                any = true;
            }
            None => break,
        }
    }
    if !any {
        return f64::NAN;
    }
    if neg { -value } else { value }
}

/// Renders a `BigInt` in the given radix (2..=36), base 10 by default.
fn bigint_to_radix(n: &crate::bignum::BigInt, radix: u32) -> String {
    let radix = if (2..=36).contains(&radix) { radix } else { 10 };
    n.to_str_radix(radix)
}

/// Parses a normalized `BigInt` digit string (decimal, or `0x`/`0o`/`0b`
/// prefixed) into the arbitrary-precision representation.
fn parse_bigint(digits: &str) -> crate::bignum::BigInt {
    let (radix, body) = match digits.get(0..2) {
        Some("0x" | "0X") => (16, &digits[2..]),
        Some("0o" | "0O") => (8, &digits[2..]),
        Some("0b" | "0B") => (2, &digits[2..]),
        _ => (10, digits),
    };
    crate::bignum::BigInt::from_str_radix(body, radix).unwrap_or_else(crate::bignum::BigInt::zero)
}

/// Normalizes an optional `fromIndex` for `indexOf`/`includes`: undefined → 0,
/// negatives count from the end, clamped to `[0, len]`.
/// `ToInteger` for a string index argument: `NaN` (and no-arg) → `Some(0)`, a
/// non-negative integer → `Some(i)`, and a negative index → `None` (out of range,
/// so `charAt` yields `""` and `charCodeAt`/`codePointAt` yield `NaN`/`undefined`).
fn str_char_index(n: f64) -> Option<usize> {
    let n = if n.is_nan() { 0.0 } else { n };
    (n >= 0.0).then_some(n as usize)
}

/// A non-negative integer array index, if `n` is one.
fn as_index(n: f64) -> Option<usize> {
    if n >= 0.0 && n <= u32::MAX as f64 && (n as u64) as f64 == n {
        Some(n as usize)
    } else {
        None
    }
}

/// `CanonicalNumericIndexString(key)` — the Number a string property key denotes
/// when it is a *canonical* numeric index, else `None`. `"-0"` maps to `-0.0`; any
/// other string is canonical only if `ToString(ToNumber(key)) === key` (so `"1"`,
/// `"-1"`, `"1.5"`, `"Infinity"`, `"NaN"` are canonical, but `"01"`, `"1.0"`,
/// `"0x1"`, `" 1"` are not — those are ordinary named properties). This selects the
/// keys the integer-indexed-exotic `[[Get]]/[[Set]]/[[Has]]/[[DefineOwnProperty]]/
/// [[Delete]]` short-circuit on (never consulting the prototype chain).
fn canonical_numeric_index(key: &str) -> Option<f64> {
    if key == "-0" {
        return Some(-0.0);
    }
    // ToNumber over a string with no radix/whitespace leniency that would survive
    // the round-trip: a leading-zero / hex / padded form will not re-`ToString` to
    // `key`, so a plain `f64` parse (plus the `Infinity` / `NaN` literals) suffices.
    let n = match key {
        "Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        "NaN" => f64::NAN,
        _ => key.parse::<f64>().ok()?,
    };
    // `ToString(n) === key` is the canonicality test; reuse the engine's own
    // Number→String (`js_number_string`) so the round-trip matches JS exactly.
    (crate::realm::js_number_string(n) == key).then_some(n)
}

/// A static (non-computed) property key as a string.
/// Expands `$`-patterns in a string-`replace` template (no capture groups, so
/// `$1`…`$9` stay literal): `$&` → match, `` $` `` → prefix, `$'` → suffix,
/// `$$` → `$`.
fn expand_dollar(template: &str, m: &str, before: &str, after: &str) -> String {
    let chars: Vec<char> = template.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$'
            && i + 1 < chars.len()
            && let Some(rep) = match chars[i + 1] {
                '$' => Some("$"),
                '&' => Some(m),
                '`' => Some(before),
                '\'' => Some(after),
                _ => None,
            }
        {
            out.push_str(rep);
            i += 2;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn static_key(key: &PropertyKey) -> Result<String, ExecError> {
    match key {
        PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(String::from(&**s)),
        // A numeric literal key is the ECMAScript `ToString(Number)` of its value,
        // so a non-canonical literal (`0.0000001`, `0x10`, `1.0`) keys under its
        // canonical form (`"1e-7"`, `"16"`, `"1"`) — matching `obj[n]` access.
        PropertyKey::Number(n) => Ok(crate::realm::js_number_string(*n)),
        // A private name needs its declaring-class scope to form a storage key,
        // which a free function cannot resolve — callers that may see a private
        // key (class member declaration / access) handle it explicitly.
        PropertyKey::Private(_) => Err(ExecError::Unsupported("private key in static_key")),
        PropertyKey::Computed(_) => Err(ExecError::Unsupported("computed key")),
    }
}

/// The internal storage key for a private element `#name` *declared in the class
/// whose id is `scope`*. Prefixed with `\0` so it is a true *internal slot*:
/// filtered from every reflection surface (`Object.keys`,
/// `getOwnPropertyNames`, `for-in`, `JSON`, …) like other engine internals,
/// and — crucially — invisible to `hasOwnProperty("#name")` /
/// `getOwnPropertyDescriptor`, since a user string can never equal it. (Storing
/// under the bare `#name` would collide with a real `obj["#name"]` string key.)
///
/// The trailing `@<scope>` ties the key to the *declaration site* of the private
/// name: per spec each `#x` is a distinct private name bound to its lexically
/// enclosing class, so two classes that both declare `#x` get different keys and
/// never collide (a nested class can shadow an outer one's `#x`).
pub(crate) fn private_storage_key(name: &str, scope: u32) -> String {
    alloc::format!("\u{0}#{name}@{scope}")
}

#[cfg(test)]
mod tests;
