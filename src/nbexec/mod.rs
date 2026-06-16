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
    /// An optional-chain short-circuit: a `?.` link found a nullish base. It
    /// propagates (past intervening non-optional links) up to the enclosing
    /// `Expr::OptChain` boundary, which turns it into `undefined`. It is *not* a
    /// throw, so `try`/`catch` never sees it.
    OptShortCircuit,
}

/// The control-flow outcome of a statement.
pub(crate) enum Flow {
    /// Fell through normally, carrying the last expression value (for `run`).
    Normal(NanBox),
    /// A `return` (value).
    Return(NanBox),
    /// A `break`, optionally targeting a label.
    Break(Option<String>),
    /// A `continue`, optionally targeting a label.
    Continue(Option<String>),
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

/// Classifies a body `Flow` for a loop carrying `label`.
fn loop_action(flow: Flow, label: &Option<String>) -> LoopAction {
    match flow {
        Flow::Normal(_) => LoopAction::Next,
        Flow::Continue(None) => LoopAction::Next,
        Flow::Continue(Some(l)) if Some(&l) == label.as_ref() => LoopAction::Next,
        Flow::Break(None) => LoopAction::Stop,
        Flow::Break(Some(l)) if Some(&l) == label.as_ref() => LoopAction::Stop,
        other => LoopAction::Propagate(other),
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
    /// Function-AST table; a closure cell holds an index into this.
    functions: Vec<FnDef<'a>>,
    /// Class-AST table; a class cell holds an index into this.
    classes: Vec<&'a Class>,
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
    /// Per-class constructor handle (the class value), parallel to `classes`, so
    /// the lazily-materialized `.prototype` can install a `constructor` back-link
    /// and link a derived prototype to its base's prototype.
    class_handles: Vec<NanBox>,
    /// Active `with (obj) …` object environment records, innermost last. A bare
    /// identifier first consults these objects (respecting `@@unscopables`) before
    /// the lexical scope chain.
    with_stack: Vec<NanBox>,
    /// Current function-call nesting depth (recursion guard).
    call_depth: usize,
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
    gen_sink: Option<Vec<NanBox>>,
    /// The `Symbol.for` global registry: shared symbols keyed by string.
    symbol_registry: alloc::collections::BTreeMap<String, NanBox>,
    /// Cached well-known symbols (e.g. `Symbol.iterator`), created on first use.
    well_known_symbols: alloc::collections::BTreeMap<&'static str, NanBox>,
    /// The frozen template-strings object for each tagged-template site (keyed by the
    /// AST node's address), so the same array is passed to the tag on every evaluation.
    tagged_template_cache: alloc::collections::BTreeMap<usize, NanBox>,
    /// Leak-once cache interning `Intl.NumberFormat` currency/unit codes to `&'static str`
    /// (the `intl` crate's options take `'static`); bounded by the distinct codes a program
    /// uses.
    #[cfg(feature = "intl")]
    intl_intern: alloc::collections::BTreeMap<String, &'static str>,
    /// The superclass to invoke for `super(...)` inside the running constructor.
    pending_super: Option<(u32, Scope)>,
    /// The native-constructor superclass for `super(...)` (e.g. extending Error).
    pending_super_native: Option<u16>,
    /// The class of the currently-running method (for `super.method()`).
    current_home: Option<u32>,
    /// The `[[HomeObject]]` of the currently-running object-literal method — the
    /// object the method was defined on — so its `super.x` resolves through that
    /// object's prototype (when there is no enclosing class home).
    current_home_object: Option<Handle>,
    /// Whether the currently-running method was entered as a static method, so
    /// `super.x` resolves against the superclass's static members.
    current_home_static: bool,
    /// A label attached to the next loop (for `break`/`continue label`).
    pending_label: Option<String>,
    /// The promise-reaction microtask queue, drained after the script.
    microtasks: Vec<Job>,
    /// Pending `setTimeout` callbacks (macrotasks), run after the microtask queue
    /// drains — ordered by `delay`, then insertion (`seq`). No real clock: delays
    /// only order callbacks relative to each other.
    macrotasks: Vec<Timer>,
    /// Monotonic id handed out by `setTimeout` (for `clearTimeout`).
    timer_next_id: u64,
    /// Monotonic insertion counter breaking equal-`delay` ties.
    timer_seq: u64,
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
    /// Programs parsed at runtime by `eval` / the `Function` constructor, keyed by
    /// source string. The interpreter's function/AST tables hold `&'a` references
    /// into the running program; a dynamically-parsed `Program` must therefore
    /// outlive the borrow. Each distinct source is parsed once, boxed, and leaked
    /// to a `&'static Program` (which coerces to `&'a`); the cache dedupes so a
    /// loop calling `eval` on the same string (or repeated `Function` bodies) does
    /// not re-leak. The leak is bounded by the number of *distinct* eval/Function
    /// sources a program produces.
    eval_programs: alloc::collections::BTreeMap<String, &'static Program>,
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
}

/// A pending `setTimeout` callback.
struct Timer {
    id: u64,
    delay: f64,
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
/// A readable static method bound to a `[constructor, name]` pair (so a detached call
/// still routes to the constructor's `call_method` static dispatch).
const N_STATIC_METHOD: u16 = 220;
const N_INTL_COLLATOR: u16 = 207;
const N_INTL_PLURAL_RULES: u16 = 208;
/// `Intl.Collator.prototype.compare` (a bound function value).
const N_INTL_COMPARE: u16 = 209;
/// `Intl.PluralRules.prototype.select`.
const N_INTL_PLURAL_SELECT: u16 = 210;
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
const TYPED_ARRAY_KINDS: [(&str, u8); 11] = [
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
// `Object.prototype.*` methods (the receiver arrives as `this`).
const N_OBJ_PROTO_TOSTRING: u16 = 179;
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
];
/// A first-class `%TypedArray%.prototype.<method>` (e.g. `map`, `slice`, `every`).
/// Like [`N_ARRAY_PROTO_FN`] but validates that the call's `this` has a
/// `[[TypedArrayName]]` internal slot first (throwing a `TypeError` otherwise),
/// and never applies the plain-`Array` result conversion — so e.g.
/// `Int8Array.prototype.map.call(ta, fn)` returns a same-kind typed array.
const N_TYPED_ARRAY_PROTO_FN: u16 = 246;
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
    "set", "get", "has", "delete", "clear", "forEach", "keys", "values", "entries",
];
/// `WeakMap.prototype` methods exposed as first-class values.
const WEAKMAP_PROTO_METHODS: &[&str] = &["set", "get", "has", "delete"];
/// `WeakSet.prototype` methods exposed as first-class values.
const WEAKSET_PROTO_METHODS: &[&str] = &["add", "has", "delete"];
/// `Promise.prototype` methods exposed as first-class values.
const PROMISE_PROTO_METHODS: &[&str] = &["then", "catch", "finally"];
/// `Function.prototype` methods exposed as first-class values.
const FUNCTION_PROTO_METHODS: &[&str] = &["call", "apply", "bind", "toString"];
/// `DataView.prototype` accessor methods — dispatched in `call_method`, exposed here as
/// readable bound natives (for `typeof dv.getUint8` and detached `dv.getUint8.call(dv, …)`).
const DATA_VIEW_METHODS: &[&str] = &[
    "getInt8",
    "getUint8",
    "getInt16",
    "getUint16",
    "getInt32",
    "getUint32",
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
        | "toReversed" | "toSorted" | "isWellFormed" | "toWellFormed" | "getInt8" | "getUint8"
        | "toArray" | "normalize"
        // `Date.prototype` getters / serializers (length 0).
        | "getTime" | "getFullYear" | "getUTCFullYear" | "getMonth" | "getUTCMonth"
        | "getDate" | "getUTCDate" | "getDay" | "getUTCDay" | "getHours" | "getUTCHours"
        | "getMinutes" | "getUTCMinutes" | "getSeconds" | "getUTCSeconds"
        | "getMilliseconds" | "getUTCMilliseconds" | "getTimezoneOffset"
        | "toISOString" | "toDateString" | "toTimeString" | "toUTCString"
        | "toLocaleDateString" | "toLocaleTimeString"
        // `Date.now()` takes no arguments.
        | "now" => 0,
        // Two-argument methods.
        "slice" | "substring" | "substr" | "splice" | "copyWithin" | "split" | "replace"
        | "replaceAll" | "padStart" | "padEnd" | "with" | "setInt8" | "setUint8" | "asIntN"
        | "asUintN" | "setMonth" | "setUTCMonth" | "setSeconds" | "setUTCSeconds" | "subarray"
        // `Function.prototype.apply(thisArg, argArray)`.
        | "apply" => 2,
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
        | N_TYPED_ARRAY_TO_STRING_TAG => 0,
        // Length 2.
        N_OBJECT_SET_PROTO
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
        | N_MATH_IMUL => 2,
        // Length 3.
        N_OBJECT_DEFINE_PROP | N_REFLECT_SET | N_REFLECT_DEFINE_PROP | N_REFLECT_APPLY => 3,
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
    if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id) {
        return true;
    }
    if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
        return true;
    }
    matches!(
        id,
        N_STRING
            | N_NUMBER
            | N_BOOLEAN
            // `Symbol`/`BigInt` have a `[[Construct]]` (so `IsConstructor` is true)
            // even though invoking it always throws a TypeError.
            | N_SYMBOL
            | N_BIGINT
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
            | N_ARRAY_BUFFER
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
/// Hidden slot on an object-literal concise method recording its `[[HomeObject]]`
/// (the object it was defined on), for `super` resolution.
const HOME_OBJECT: &str = "\u{0}home";
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
/// Reserved hidden keys for a bound function (`Function.prototype.bind`).
/// Hidden slot holding a primitive-wrapper object's boxed value, and its
/// constructor id (for `instanceof`).
const PRIM_WRAP: &str = "\u{0}prim";
const PRIM_WRAP_TYPE: &str = "\u{0}primtype";
/// `ArrayBuffer` byte store (an array of 0–255 numbers) and `DataView` linkage.
const ARRAY_BUFFER_BYTES: &str = "\u{0}abytes";
/// Marks an `ArrayBuffer` as detached (after `transfer()`): its `byteLength` reads 0 and its
/// views have been emptied.
const ARRAY_BUFFER_DETACHED: &str = "\u{0}abdetached";
/// An `ArrayBuffer`'s `maxByteLength` — present iff it was constructed resizable (via
/// `new ArrayBuffer(n, { maxByteLength })`), bounding `resize`.
const ARRAY_BUFFER_MAXLEN: &str = "\u{0}abmaxlen";
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
const BOUND_TARGET: &str = "\u{0}bnd_t";
const BOUND_THIS: &str = "\u{0}bnd_this";
const BOUND_ARGS: &str = "\u{0}bnd_args";
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

mod call;
mod class;
mod convert;
mod expr;
mod intl_fmt;
mod iterator;
mod json;
mod method_dispatch;
mod native_dispatch;
mod object;
mod promise;
mod regexp;
mod stmt;
mod typed_array;
mod wasm;

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
            functions: Vec::new(),
            classes: Vec::new(),
            class_statics: Vec::new(),
            class_static_fields: Vec::new(),
            class_static_get: Vec::new(),
            class_static_set: Vec::new(),
            class_envs: Vec::new(),
            class_native_super: Vec::new(),
            class_handles: Vec::new(),
            with_stack: Vec::new(),
            call_depth: 0,
            eval_depth: 0,
            rng_state: math_random_seed(),
            this_val: NanBox::undefined(),
            new_target: NanBox::undefined(),
            pending_new_target: None,
            reflect_new_target: None,
            wasm_states: alloc::collections::BTreeMap::new(),
            wasm_modules: alloc::collections::BTreeMap::new(),
            wasm_mem_objs: alloc::collections::BTreeMap::new(),
            wasm_next_id: 0,
            gen_sink: None,
            symbol_registry: alloc::collections::BTreeMap::new(),
            well_known_symbols: alloc::collections::BTreeMap::new(),
            tagged_template_cache: alloc::collections::BTreeMap::new(),
            #[cfg(feature = "intl")]
            intl_intern: alloc::collections::BTreeMap::new(),
            pending_super: None,
            pending_super_native: None,
            current_home: None,
            current_home_object: None,
            current_home_static: false,
            pending_label: None,
            microtasks: Vec::new(),
            macrotasks: Vec::new(),
            timer_next_id: 1,
            timer_seq: 0,
            strict: false,
            global_this: NanBox::undefined(),
            output: String::new(),
            global_scope: Scope::root(),
            eval_programs: alloc::collections::BTreeMap::new(),
        };
        // The constructor's `current` IS the root scope; capture it as the global
        // scope before `install_globals` populates it, so indirect eval can run
        // against it later.
        interp.global_scope = interp.current.clone();
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
    fn install_fn_name_length(&mut self, f: Handle, name: &str, length: u32) {
        // Spec own-key order for a function is `length` before `name`.
        self.realm
            .set_property(f, "length", NanBox::number(f64::from(length)));
        self.realm.mark_hidden(f, "length");
        self.realm.set_readonly_property(f, "length");
        let name_v = self.new_str(name);
        self.realm.set_property(f, "name", name_v);
        self.realm.mark_hidden(f, "name");
        self.realm.set_readonly_property(f, "name");
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
        for &(name, arity) in TYPED_ARRAY_PROTO_METHODS {
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
            self.realm.mark_hidden(ctor, "prototype");
            // `<TypedArray>.BYTES_PER_ELEMENT` is also exposed on each kind's
            // prototype (spec); the constructor static is handled in `read_member`.
            let _ = i;
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
            self.realm
                .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
            self.realm.mark_hidden(ctor, "prototype");
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
        let name_h = self.realm.new_string(name);
        let f = self.realm.new_bound_native(N_AB_PROTO_FN, name_h);
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
        // The `Error` family — native constructors producing `{ name, message }`.
        // Only the standard errors are globals; the `WebAssembly.*` error
        // subclasses are installed under the WebAssembly namespace below.
        for (i, name) in ERROR_NAMES.iter().enumerate().take(N_GLOBAL_ERROR_COUNT) {
            let ctor = self.new_named_native(name, N_ERROR_BASE + i as u16);
            self.current.declare(name, NanBox::handle(ctor.to_raw()));
        }
        install_namespace(
            self,
            "JSON",
            &[("stringify", N_JSON_STRINGIFY), ("parse", N_JSON_PARSE)],
        );
        install_namespace(
            self,
            "Object",
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
        install_namespace(
            self,
            "Array",
            &[
                ("isArray", N_ARRAY_IS_ARRAY),
                ("from", N_ARRAY_FROM),
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
            ("structuredClone", N_STRUCTURED_CLONE),
            ("setTimeout", N_SET_TIMEOUT),
            ("clearTimeout", N_CLEAR_TIMEOUT),
            ("queueMicrotask", N_QUEUE_MICROTASK),
            ("btoa", N_BTOA),
            ("atob", N_ATOB),
            ("URIError", N_URI_ERROR),
            ("EvalError", N_EVAL_ERROR),
            ("eval", N_EVAL),
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
        ] {
            let f = self.new_named_native(name, id);
            // `Intl.X.supportedLocalesOf(locales)` — static on every constructor.
            let sl = self.new_named_native("supportedLocalesOf", N_INTL_SUPPORTED_LOCALES);
            self.realm
                .set_hidden_property(f, "supportedLocalesOf", NanBox::handle(sl.to_raw()));
            self.realm
                .set_property(intl, name, NanBox::handle(f.to_raw()));
        }
        self.current.declare("Intl", NanBox::handle(intl.to_raw()));
        // The typed-array constructors.
        for (i, (name, _)) in TYPED_ARRAY_KINDS.iter().enumerate() {
            let f = self.new_named_native(name, N_TYPED_ARRAY_BASE + i as u16);
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
        for (name, id) in [("ArrayBuffer", N_ARRAY_BUFFER), ("DataView", N_DATA_VIEW)] {
            let f = self.realm.new_native(id);
            // `ArrayBuffer.isView(x)` — true for a typed array or a DataView.
            if id == N_ARRAY_BUFFER {
                let isview = self.realm.new_native(N_ARRAY_BUFFER_IS_VIEW);
                self.realm
                    .set_hidden_property(f, "isView", NanBox::handle(isview.to_raw()));
            }
            self.current.declare(name, NanBox::handle(f.to_raw()));
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
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        self.realm
            .set_hidden_property(iter_proto, &iter_key, NanBox::handle(self_iter.to_raw()));
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
        self.realm.set_hidden_property(
            iter_proto,
            "constructor",
            NanBox::handle(iterator_ctor.to_raw()),
        );
        self.realm.set_hidden_property(
            iterator_ctor,
            "prototype",
            NanBox::handle(iter_proto.to_raw()),
        );
        self.realm.mark_hidden(iterator_ctor, "prototype");
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
        for (name, id) in [
            ("toString", N_OBJ_PROTO_TOSTRING),
            ("toLocaleString", N_OBJ_PROTO_TOSTRING),
            ("valueOf", N_OBJ_PROTO_VALUEOF),
            ("hasOwnProperty", N_OBJ_PROTO_HASOWN),
            ("isPrototypeOf", N_OBJ_PROTO_ISPROTOTYPEOF),
            ("propertyIsEnumerable", N_OBJ_PROTO_PROPISENUM),
        ] {
            let f = self.realm.new_native(id);
            self.realm
                .set_property(obj_proto, name, NanBox::handle(f.to_raw()));
            // Non-enumerable, so inheriting objects don't surface them in for-in /
            // Object.keys.
            self.realm.mark_hidden(obj_proto, name);
        }
        if let Some(obj_ns) = self
            .current
            .get("Object")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm
                .set_property(obj_ns, "prototype", NanBox::handle(obj_proto.to_raw()));
            self.realm.mark_hidden(obj_ns, "prototype"); // non-enumerable
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
        self.setup_first_class_prototype_id("Number", NUMBER_PROTO_METHODS, N_NUMBER_PROTO_FN);
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
        for (ctor, prim) in [
            ("Number", NanBox::number(0.0)),
            ("Boolean", NanBox::boolean(false)),
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
                self.realm.set_hidden_property(proto, PRIM_WRAP, prim);
                let id = if ctor == "Number" {
                    N_NUMBER
                } else {
                    N_BOOLEAN
                };
                self.realm.set_hidden_property(
                    proto,
                    PRIM_WRAP_TYPE,
                    NanBox::number(f64::from(id)),
                );
            }
        }
        self.setup_first_class_prototype_id("BigInt", BIGINT_PROTO_METHODS, N_BIGINT_PROTO_FN);
        self.setup_first_class_prototype_id("Date", DATE_PROTO_METHODS, N_DATE_PROTO_FN);
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
        }
        self.setup_first_class_prototype("Function", FUNCTION_PROTO_METHODS);
        self.setup_first_class_prototype_id("Set", SET_PROTO_METHODS, N_SET_PROTO_FN);
        self.setup_first_class_prototype_id("Map", MAP_PROTO_METHODS, N_MAP_PROTO_FN);
        self.setup_first_class_prototype_id("WeakMap", WEAKMAP_PROTO_METHODS, N_WEAKMAP_PROTO_FN);
        self.setup_first_class_prototype_id("WeakSet", WEAKSET_PROTO_METHODS, N_WEAKSET_PROTO_FN);
        // `Promise.prototype` (then/catch/finally) — so `Promise.prototype.then`
        // is readable / detachable and `Promise.prototype[Symbol.toStringTag]`
        // exists. Promise instances link to it below.
        self.setup_first_class_prototype("Promise", PROMISE_PROTO_METHODS);
        // `Ctor.prototype[Symbol.toStringTag]` — a non-enumerable, non-writable,
        // configurable string. (`Object.prototype.toString` reads it.)
        self.install_proto_to_string_tag("Set", "Set");
        self.install_proto_to_string_tag("Map", "Map");
        self.install_proto_to_string_tag("WeakMap", "WeakMap");
        self.install_proto_to_string_tag("WeakSet", "WeakSet");
        self.install_proto_to_string_tag("Promise", "Promise");
        // Namespace objects carry their own `[Symbol.toStringTag]` value (not on a
        // prototype): `Reflect`/`JSON`/`Math` → `[object Reflect|JSON|Math]`.
        for (ns_name, tag) in [("Reflect", "Reflect"), ("JSON", "JSON"), ("Math", "Math")] {
            if let Some(ns) = self
                .current
                .get(ns_name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
            {
                self.install_to_string_tag(ns, tag);
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
                self.realm
                    .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
                self.realm.mark_hidden(ctor, "prototype");
                proto
            })
        {
            for name in &ERROR_NAMES[1..N_GLOBAL_ERROR_COUNT] {
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
                    self.realm
                        .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
                    self.realm.mark_hidden(ctor, "prototype");
                }
            }
        }
        // The shared abstract `%TypedArray%` intrinsic constructor and the
        // constructor-side hierarchy that hangs the concrete TA constructors off
        // it (so `Object.getPrototypeOf(Int8Array) === %TypedArray%`).
        self.setup_typed_array_intrinsic(obj_proto);
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
                "withResolvers",
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
        self.setup_static_methods("String", &["fromCharCode", "fromCodePoint", "raw"]);
        self.setup_static_methods("Symbol", &["for", "keyFor"]);
        self.setup_static_methods("Date", &["now", "parse", "UTC"]);
        self.setup_static_methods("BigInt", &["asIntN", "asUintN"]);
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
        // `globalThis`: an object mirroring the global bindings, referencing
        // itself. Reads like `globalThis.Math` and `globalThis.globalThis` work.
        let global = self.realm.new_object();
        for n in [
            "Math",
            "JSON",
            "Object",
            "Array",
            "Reflect",
            "String",
            "Number",
            "Boolean",
            "parseInt",
            "parseFloat",
            "isNaN",
            "isFinite",
            "Map",
            "Set",
            "Symbol",
            "BigInt",
            "Proxy",
            "WeakMap",
            "WeakSet",
            "WeakRef",
            "FinalizationRegistry",
            "Promise",
            "Date",
            "console",
            "Error",
            "TypeError",
            "RangeError",
            "SyntaxError",
            "ReferenceError",
            "AggregateError",
            "encodeURIComponent",
            "decodeURIComponent",
            "encodeURI",
            "decodeURI",
            "structuredClone",
            "btoa",
            "atob",
            "Intl",
            "eval",
        ] {
            if let Some(v) = self.current.get(n) {
                self.realm.set_property(global, n, v);
            }
        }
        for (name, _) in TYPED_ARRAY_KINDS {
            if let Some(v) = self.current.get(name) {
                self.realm.set_property(global, name, v);
            }
        }
        self.realm
            .set_property(global, "NaN", NanBox::number(f64::NAN));
        self.realm
            .set_property(global, "Infinity", NanBox::number(f64::INFINITY));
        self.realm
            .set_property(global, "undefined", NanBox::undefined());
        let gbox = NanBox::handle(global.to_raw());
        self.realm.set_property(global, "globalThis", gbox);
        self.current.declare("globalThis", gbox);
        self.global_this = gbox;
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
                Flow::Normal(v) => last = v,
                Flow::Return(v) => {
                    self.run_event_loop()?;
                    return Ok(v);
                }
                Flow::Break(_) | Flow::Continue(_) => {}
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
    fn parse_eval_program(&mut self, source: &str) -> Result<&'a Program, ExecError> {
        if let Some(p) = self.eval_programs.get(source) {
            return Ok(p);
        }
        match crate::parser::Parser::parse_program(source) {
            Ok(program) => {
                // The AST is fully owned (no borrow of `source`); leaking the box
                // yields a `'static` reference that coerces to `'a`.
                let leaked: &'static Program =
                    alloc::boxed::Box::leak(alloc::boxed::Box::new(program));
                self.eval_programs.insert(String::from(source), leaked);
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
        self.hoist_with(&program.body, true)?;
        let mut last = NanBox::undefined();
        for stmt in &program.body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = v,
                // A `return` is a SyntaxError at parse time at the top level, so
                // it cannot reach here; `break`/`continue` likewise. Treat any
                // such residue as completing normally.
                Flow::Return(v) => return Ok(v),
                Flow::Break(_) | Flow::Continue(_) => {}
            }
        }
        Ok(last)
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
        let program = self.parse_eval_program(source)?;
        let code_strict = has_use_strict(&program.body);

        // Recursion guard shared with the tree-walk budget.
        if self.eval_depth >= self.realm.limits.max_eval_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(
                self.make_error(N_ERROR_BASE + 2, Some(msg)),
            ));
        }

        let saved_strict = self.strict;
        let saved_scope = self.current.clone();
        let (saved_this, saved_new_target) = (self.this_val, self.new_target);

        if !direct {
            // Indirect eval: runs against the GLOBAL environment with global
            // `this`, sloppy unless the code self-declares `"use strict"`. Its
            // `var`/function declarations hoist into the global variable
            // environment, so a sloppy indirect eval runs directly in the global
            // scope (mirroring sloppy direct eval, which runs in the caller's
            // scope). Strict eval gets its own child env so its declarations don't
            // leak globally.
            self.strict = code_strict;
            self.current = if code_strict {
                self.global_scope.child()
            } else {
                self.global_scope.clone()
            };
            self.this_val = self.global_this;
            self.new_target = NanBox::undefined();
        } else {
            // Direct eval inherits the caller's strictness; the code may add its own.
            self.strict = saved_strict || code_strict;
            // Strict eval (caller-strict or code-strict) gets its own variable
            // environment so its declarations don't leak into the caller. Sloppy
            // direct eval runs directly in the caller's scope so `var`/function
            // declarations hoist outward (spec sloppy-mode behaviour).
            if self.strict {
                self.current = saved_scope.child();
            }
            // `this`/`new.target` are inherited from the caller (unchanged).
        }

        self.eval_depth += 1;
        let result = self.run_eval_body(program);
        self.eval_depth -= 1;

        self.current = saved_scope;
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
    fn build_function_constructor(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        // Coerce arguments to strings (ToString). Last is the body; the rest are
        // the parameter-list pieces, joined with commas.
        let (params, body) = match args.split_last() {
            Some((last, rest)) => {
                let parts: Vec<String> = rest
                    .iter()
                    .map(|a| self.realm.to_display_string(*a))
                    .collect();
                (parts.join(","), self.realm.to_display_string(*last))
            }
            // `Function()` with no arguments → an empty-body anonymous function.
            None => (String::new(), String::new()),
        };
        let source = alloc::format!("(function anonymous({params}\n) {{\n{body}\n}})");

        let program = self.parse_eval_program(&source)?;
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
            // Surface `name`/`length` as own data properties with the spec
            // attributes, matching other built-in functions.
            let len = func
                .params
                .iter()
                .take_while(|p| !p.rest && p.default.is_none())
                .count();
            self.install_fn_name_length(h, "anonymous", len as u32);
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
        // `var` names hoist to the function/program scope as `undefined` (so a
        // read before the declaration yields `undefined`, not a ReferenceError).
        // Done first; a same-named function declaration then overwrites it.
        if hoist_vars {
            let mut var_names: Vec<&str> = Vec::new();
            collect_var_names(stmts, &mut var_names);
            // Annex B: a function declared inside a block also var-hoists its name
            // to the enclosing function scope (initially `undefined`).
            collect_block_function_names(stmts, &mut var_names);
            for name in var_names {
                if !self.current.has_local(name) {
                    self.current.declare(name, NanBox::undefined());
                }
            }
        }
        for stmt in stmts {
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
                if hoist_vars {
                    // A function/program top-level declaration binds here.
                    self.current.declare(&id.name, value);
                } else {
                    // A block-level declaration (Annex B) assigns the function-scope
                    // `var` binding hoisted above; if none exists, bind locally.
                    if !self.current.set(&id.name, value) {
                        self.current.declare(&id.name, value);
                    }
                }
            }
        }
        Ok(())
    }

    /// Block-level hoisting: function declarations only (`var` is function-scoped
    /// and hoisted at the function/program boundary instead).
    fn hoist(&mut self, stmts: &'a [Stmt]) -> Result<(), ExecError> {
        self.hoist_with(stmts, false)
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
        // the function body's own `"use strict"` directive prologue.
        let is_strict = self.strict || matches!(body, Body::Block(stmts) if has_use_strict(stmts));
        let func_id = self.functions.len() as u32;
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
        });
        let handle = self.realm.new_function(func_id, self.current.clone());
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
        let name_v = self.new_str(name);
        self.realm.set_property(obj, "name", name_v);
        let msg_str = match message {
            Some(m) if !matches!(m.unpack(), Unpacked::Undefined) => {
                self.realm.to_display_string(m)
            }
            _ => String::new(),
        };
        let msg = self.new_str(&msg_str);
        self.realm.set_property(obj, "message", msg);
        // `name`/`message` are non-enumerable (so `Object.keys(err)` is empty).
        self.realm.mark_hidden(obj, "name");
        self.realm.mark_hidden(obj, "message");
        // A minimal `stack` (the `name: message` header; no real frame capture),
        // non-enumerable like the real property.
        let head = if msg_str.is_empty() {
            String::from(name)
        } else {
            alloc::format!("{name}: {msg_str}")
        };
        let stack = self.new_str(&alloc::format!("{head}\n    at <anonymous>"));
        self.realm.set_property(obj, "stack", stack);
        self.realm.mark_hidden(obj, "stack");
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
                Expr::Str { value, .. } => {
                    if &**value == b"use strict" {
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

/// Collects the names of function declarations that appear **inside a block** (at
/// any nesting depth below the immediate statement list). Per Annex B, such a
/// name is var-hoisted to the enclosing function scope. The immediate top-level
/// functions are excluded — they are bound directly by the hoisting loop.
fn collect_block_function_names<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>) {
    use core::slice::from_ref;
    fn walk<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>, in_block: bool) {
        for stmt in stmts {
            match stmt {
                Stmt::Function(f) if in_block => {
                    if let Some(id) = &f.id {
                        out.push(&id.name);
                    }
                }
                Stmt::Block { body, .. } => walk(body, out, true),
                Stmt::If {
                    consequent,
                    alternate,
                    ..
                } => {
                    walk(from_ref(consequent), out, true);
                    if let Some(a) = alternate {
                        walk(from_ref(a), out, true);
                    }
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::Labeled { body, .. }
                | Stmt::For { body, .. }
                | Stmt::ForIn { body, .. }
                | Stmt::ForOf { body, .. } => walk(from_ref(body), out, true),
                Stmt::Try {
                    block, finalizer, ..
                } => {
                    walk(block, out, true);
                    if let Some(f) = finalizer {
                        walk(f, out, true);
                    }
                }
                Stmt::Switch { cases, .. } => {
                    for case in cases {
                        walk(&case.body, out, true);
                    }
                }
                _ => {}
            }
        }
    }
    walk(stmts, out, false);
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
fn slice_bounds(start: f64, end_arg: NanBox, realm: &Realm, len: usize) -> (usize, usize) {
    let clamp = |n: f64| -> usize {
        if n < 0.0 {
            (len as f64 + n).max(0.0) as usize
        } else {
            (n as usize).min(len)
        }
    };
    let a = clamp(start);
    let b = match end_arg.unpack() {
        Unpacked::Undefined => len,
        _ => clamp(realm.to_number(end_arg)),
    };
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
    let pad_units = crate::wtf8::utf16_len(pad);
    // Repeat the pad until it covers `need` units, then trim to exactly `need`.
    let mut filler: Vec<u8> = Vec::new();
    let mut have = 0usize;
    while have < need {
        filler.extend_from_slice(pad);
        have += pad_units;
    }
    if have > need {
        filler = crate::wtf8::slice_utf16(&filler, 0, need);
    }
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
    alloc::format!("{:.*}", decimals, n)
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
        7 => f64::from(n as f32), // Float32
        8 => n,                   // Float64
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

/// Renders `Intl.RelativeTimeFormat.prototype.format(value, unit)` in en-US. `numeric:
/// "auto"` yields idiomatic phrases for the adjacent units ("yesterday", "next week",
/// "now"); otherwise (and as a fallback) it is the explicit "in N units" / "N units ago".
fn relative_time_string(value: f64, unit: &str, numeric: &str) -> alloc::string::String {
    let u = unit.strip_suffix('s').unwrap_or(unit); // singular stem
    // Integer test without std-only float methods (for the `alloc` build).
    if numeric == "auto" && value == (value as i64) as f64 {
        let v = value as i64;
        match (u, v) {
            ("day", -1) => return String::from("yesterday"),
            ("day", 0) => return String::from("today"),
            ("day", 1) => return String::from("tomorrow"),
            ("second", 0) => return String::from("now"),
            ("week" | "month" | "quarter" | "year", -1) => return alloc::format!("last {u}"),
            ("week" | "month" | "quarter" | "year", 1) => return alloc::format!("next {u}"),
            ("minute" | "hour" | "week" | "month" | "quarter" | "year", 0) => {
                return alloc::format!("this {u}");
            }
            _ => {}
        }
    }
    let n = value.abs();
    let unit_disp = if n == 1.0 {
        String::from(u)
    } else {
        alloc::format!("{u}s")
    };
    if value < 0.0 {
        alloc::format!("{n} {unit_disp} ago")
    } else {
        alloc::format!("in {n} {unit_disp}")
    }
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
/// this buffer directly (RE-7: the subject is collected once per operation).
#[cfg(feature = "regex")]
fn u16_slice(units: &[u16], st: usize, en: usize) -> alloc::vec::Vec<u8> {
    let st = st.min(units.len());
    let en = en.min(units.len()).max(st);
    crate::wtf8::from_utf16(&units[st..en])
}

/// Slices a pre-collected `&[u16]` subject from code-unit index `st` to the end,
/// re-encoded to WTF-8 bytes.
#[cfg(feature = "regex")]
fn u16_slice_from(units: &[u16], st: usize) -> alloc::vec::Vec<u8> {
    crate::wtf8::from_utf16(&units[st.min(units.len())..])
}

/// Advances a code-unit position past a just-consumed empty match. Per spec
/// `AdvanceStringIndex`, a `u`-flag regex steps a whole code point (skipping the
/// low half of a surrogate pair), while a non-`u` regex steps one code unit.
#[cfg(feature = "regex")]
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
/// overflow-to-infinity handling. (Rust has no stable `f16`.)
#[cfg(feature = "std")]
fn f64_to_f16_bits(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    if value.is_nan() {
        return sign | 0x7E00; // a quiet NaN
    }
    let abs = value.abs();
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

/// Expands a binary16 bit pattern to the `f64` it represents.
#[cfg(feature = "std")]
fn f16_to_f64(h: u16) -> f64 {
    let sign = if (h & 0x8000) != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1F;
    let mant = (h & 0x03FF) as f64;
    match exp {
        0 => sign * mant * 2.0f64.powi(-24), // subnormal (and ±0 when mant == 0)
        0x1F => {
            if mant == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + mant / 1024.0) * 2.0f64.powi(exp as i32 - 15),
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

/// Expands a `replace` template against `caps`: `$&` (whole match), `` $` ``
/// (prefix), `$'` (suffix), `$1`..`$9` (numbered groups), `$<name>` (named
/// groups), and `$$` (literal `$`). The subject is the pre-collected `&[u16]`
/// (the native UTF-16 regex subject) and capture spans are **code-unit**
/// indices. Substituted slices are re-encoded to WTF-8 so astral characters and
/// lone surrogates survive, and the returned bytes are concatenated WTF-8.
#[cfg(feature = "regex")]
fn expand_replacement_u16(
    templ: &str,
    subj: &[u16],
    caps: &crate::regex::Captures,
    group_names: &[(usize, String)],
) -> alloc::vec::Vec<u8> {
    let group = |i: usize| -> alloc::vec::Vec<u8> {
        caps.groups
            .get(i)
            .and_then(|g| *g)
            .map(|(s, e)| u16_slice(subj, s, e))
            .unwrap_or_default()
    };
    let (m_start, m_end) = caps.groups.first().and_then(|g| *g).unwrap_or((0, 0));
    let mut out: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    let mut tc = templ.chars().peekable();
    while let Some(c) = tc.next() {
        // `$<name>` — a named-group backreference.
        if c == '$' && tc.peek() == Some(&'<') {
            tc.next(); // `<`
            let mut name = String::new();
            while let Some(&ch) = tc.peek() {
                tc.next();
                if ch == '>' {
                    break;
                }
                name.push(ch);
            }
            if let Some((idx, _)) = group_names.iter().find(|(_, n)| *n == name) {
                out.extend_from_slice(&group(*idx));
            }
            continue;
        }
        if c != '$' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        match tc.peek() {
            Some('$') => {
                out.push(b'$');
                tc.next();
            }
            Some('&') => {
                out.extend_from_slice(&group(0));
                tc.next();
            }
            // `` $` `` is the portion before the match; `$'` the portion after.
            Some('`') => {
                out.extend_from_slice(&u16_slice(subj, 0, m_start));
                tc.next();
            }
            Some('\'') => {
                out.extend_from_slice(&u16_slice_from(subj, m_end));
                tc.next();
            }
            // `$n` for an in-range group → the capture (empty if unmatched);
            // out-of-range `$n` is left literal.
            Some(d)
                if d.is_ascii_digit() && {
                    let n = (*d as u8 - b'0') as usize;
                    n >= 1 && n < caps.groups.len()
                } =>
            {
                let n = (*d as u8 - b'0') as usize;
                tc.next();
                out.extend_from_slice(&group(n));
            }
            _ => out.push(b'$'),
        }
    }
    out
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
    interp.display(thrown)
}

/// Extracts `(name, message)` from an error-shaped thrown object — the basis for
/// both the human-readable [`format_thrown`] and the structured [`Thrown`] the
/// conformance runner uses to verify a negative test's declared error *type*.
/// Returns `None` for a non-error thrown value (e.g. `throw 42`).
fn error_name_message(interp: &Interp, thrown: NanBox) -> Option<(String, String)> {
    let raw = thrown.as_handle()?;
    let h = Handle::from_raw(raw);
    let realm = interp.realm();
    let name = realm.get_property(h, "name")?;
    let name = realm.to_display_string(name);
    let message = realm
        .get_property(h, "message")
        .map(|m| realm.to_display_string(m))
        .unwrap_or_default();
    Some((name, message))
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
    let mut interp = Interp::new_with_limits(limits);
    match interp.run(&program) {
        Ok(value) => {
            let completion = interp.display(value);
            Ok((String::from(interp.output()), completion))
        }
        Err(ExecError::Throw(thrown)) => {
            let (name, message) = error_name_message(&interp, thrown)
                .unwrap_or_else(|| (interp.display(thrown), String::new()));
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

fn array_from_index(realm: &Realm, arg: NanBox, len: usize) -> usize {
    if matches!(arg.unpack(), Unpacked::Undefined) {
        return 0;
    }
    let n = realm.to_number(arg);
    if n < 0.0 {
        (len as f64 + n).max(0.0) as usize
    } else {
        (n as usize).min(len)
    }
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
        PropertyKey::Number(n) => Ok(alloc::format!("{n}")),
        // A private field name (`#x`) maps to a `#`-prefixed storage key.
        PropertyKey::Private(s) => Ok(alloc::format!("#{s}")),
        PropertyKey::Computed(_) => Err(ExecError::Unsupported("computed key")),
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
    fn lone_surrogates_round_trip_through_string_ops() {
        // Creation preserves a lone surrogate; length is in UTF-16 units.
        assert_eq!(run(r#""\uD800".length === 1"#), "true");
        assert_eq!(run(r#""\uD800".charCodeAt(0) === 0xD800"#), "true");
        assert_eq!(run(r#""\u{D834}".charCodeAt(0) === 0xD834"#), "true");
        // An astral char is two units, one code point.
        assert_eq!(
            run(
                r#""😀".length === 2 && "😀".codePointAt(0) === 0x1F600 && "😀".charCodeAt(0) === 0xD83D"#
            ),
            "true"
        );
        // slice over UTF-16 units keeps a lone surrogate.
        assert_eq!(
            run(r#""a\uD800b".slice(1,2).charCodeAt(0) === 0xD800"#),
            "true"
        );
        assert_eq!(
            run(r#""a\uD800b".substring(1,2).charCodeAt(0) === 0xD800"#),
            "true"
        );
        assert_eq!(run(r#""a\uD800b".at(1) === "\uD800""#), "true");
        assert_eq!(run(r#""a\uD800b"[1].charCodeAt(0) === 0xD800"#), "true");
        // charAt of a lone surrogate is a one-unit string carrying the surrogate.
        assert_eq!(
            run(r#""\uD800".charAt(0).charCodeAt(0) === 0xD800"#),
            "true"
        );
        // fromCharCode preserves a lone surrogate.
        assert_eq!(
            run("String.fromCharCode(0xD800).charCodeAt(0) === 0xD800"),
            "true"
        );
        assert_eq!(
            run("String.fromCharCode(0xD83D, 0xDE00).codePointAt(0) === 0x1F600"),
            "true"
        );
    }

    /// P1/P5/P6: the string-method fast path (single rope flatten, lazy lossy
    /// `String`, running UTF-16 counts) must not change observable behaviour on
    /// ordinary or surrogate-bearing strings.
    #[test]
    fn string_method_correctness_after_perf_rework() {
        // charCodeAt / slice / indexOf on ordinary strings.
        assert_eq!(run(r#""abc".charCodeAt(1)"#), "98");
        assert_eq!(run(r#""hello".slice(1,3)"#), "el");
        assert_eq!(run(r#""hello".indexOf("ll")"#), "2");
        assert_eq!(run(r#""hello".lastIndexOf("l")"#), "3");
        // The lazily-built lossy `String` arms still work.
        assert_eq!(run(r#""  hi  ".trim()"#), "hi");
        assert_eq!(run(r#""  hi".trimStart()"#), "hi");
        assert_eq!(run(r#""hi  ".trimEnd()"#), "hi");
        assert_eq!(run(r#""abcabc".search("ca")"#), "2");
        assert_eq!(run(r#""a".localeCompare("b") < 0"#), "true");
        // replace / replaceAll, including the function callback whose match-offset
        // is now produced by a running UTF-16 unit count (P6).
        assert_eq!(run(r#""a-b-c".replace("-", "+")"#), "a+b-c");
        assert_eq!(run(r#""a-b-c".replaceAll("-", "+")"#), "a+b+c");
        // The callback receives the correct UTF-16 offsets (1 and 3) at each match.
        assert_eq!(run(r#""a-b-c".replaceAll("-", (m,o)=>o)"#), "a1b3c");
        // P6 offsets stay correct past an astral char (the 😀 counts as 2 UTF-16
        // units, so the first `-` is at unit 2 and the second at unit 4).
        assert_eq!(run(r#""😀-x-y".replaceAll("-", (m,o)=>o)"#), "😀2x4y");
        // Surrogate-bearing strings round-trip through the byte-based ops.
        assert_eq!(run(r#""\uD800".length === 1"#), "true");
        assert_eq!(run(r#""\uD800".charCodeAt(0) === 0xD800"#), "true");
        assert_eq!(
            run(r#""a\uD800b".slice(1,2).charCodeAt(0) === 0xD800"#),
            "true"
        );
        assert_eq!(run(r#""😀".length"#), "2");
    }

    /// RE-P1: a `RegExp` whose compiled program is now cached on its cell must
    /// still behave identically when reused across many calls — the cache returns
    /// a consistent program, `lastIndex` keeps advancing for `g`/`y`, and two
    /// regexes that share a source but differ in flags must not collide.
    #[cfg(feature = "regex")]
    #[test]
    fn regex_compiled_cache_preserves_behaviour() {
        // Reusing one regex across a loop yields the same result every call (the
        // cached program is used, not recompiled into something different).
        assert_eq!(
            run(r#"{
                    let re = /a(\d)/;
                    let out = [];
                    for (let i = 0; i < 5; i++) out.push(re.test("a7") + "" + (re.exec("a7")[1]));
                    out.join(",")
                }"#),
            "true7,true7,true7,true7,true7"
        );

        // A global regex reused via String.match collects every occurrence.
        assert_eq!(
            run(r#"{ let re=/(\d+)/g; "a1b22c333".match(re).join(",") }"#),
            "1,22,333"
        );

        // `lastIndex` advances across repeated stateful `exec`/`test` calls and
        // resets to 0 after the final miss — unaffected by the program cache.
        assert_eq!(
            run(r#"{
                    let re = /\d/g;
                    let s = "a1b2";
                    let idx = [];
                    re.exec(s); idx.push(re.lastIndex);
                    re.exec(s); idx.push(re.lastIndex);
                    re.exec(s); idx.push(re.lastIndex);   // miss -> reset to 0
                    idx.join(",")
                }"#),
            "2,4,0"
        );

        // A sticky regex's lastIndex advances exactly at the match boundary.
        assert_eq!(
            run(r#"{
                    let re = /\d/y;
                    re.lastIndex = 1;
                    let m = re.test("a1b2");
                    m + ":" + re.lastIndex
                }"#),
            "true:2"
        );

        // Same source, different flags are distinct programs and must not collide
        // through the cache: `/x/u` (unicode) vs `/x/` (plain) behave per their
        // own flags. `/😀/u` matches the astral char as one unit-pair; `/./` only
        // ever spans one code unit, while `/./u` spans the whole astral char.
        assert_eq!(run(r#"/x/u.test("x") && !/x/u.global"#), "true");
        assert_eq!(run(r#"/x/.test("x") && /x/g.global"#), "true");
        // `.` with and without `u` over an astral subject: non-`u` `.` matches one
        // code unit (length-1 match), `u` `.` matches the whole code point (2).
        assert_eq!(run(r#""😀".match(/./)[0].length"#), "1");
        assert_eq!(run(r#""😀".match(/./u)[0].length"#), "2");

        // Two regexes built from the same source string but different flags, used
        // in the same scope, keep independent compiled programs.
        assert_eq!(
            run(r#"{
                    let a = /\w+/;
                    let b = /\w+/g;
                    let r1 = "foo bar".match(a).length;     // non-global: 1 match
                    let r2 = "foo bar".match(b).length;     // global: 2 matches
                    r1 + "," + r2
                }"#),
            "1,2"
        );
    }

    /// C1: a dense-array element *write* whose growth would exceed the configured
    /// `max_array_len` cap throws a catchable `RangeError("Invalid array length")`
    /// instead of being a silent no-op. A *length* set to a valid uint32 above the
    /// cap is a spec-conformant sparse length (no allocation), not a RangeError;
    /// only a length above the uint32 ceiling (2^32-1) is invalid.
    #[test]
    fn oversized_array_growth_throws_range_error() {
        // `a[1e9] = 1` (index 1e9 > the 100M default cap) throws RangeError — a
        // dense element write past the cap cannot be served.
        assert_eq!(
            run("var a=[1]; try{a[1e9]=1;'noThrow'}catch(e){e.constructor.name}"),
            "RangeError"
        );
        // `a.length = 1e9` is a valid uint32: a sparse length, reported as-is, no throw.
        assert_eq!(
            run("var a=[1]; a.length=1e9; String(a.length)"),
            "1000000000"
        );
        // Computed `a["length"] = 1e9` behaves the same.
        assert_eq!(
            run("var a=[1]; a['length']=1e9; String(a.length)"),
            "1000000000"
        );
        // A length above the uint32 ceiling (2^32) is invalid → RangeError.
        assert_eq!(
            run("var a=[1]; try{a.length=4294967296;'noThrow'}catch(e){e.constructor.name}"),
            "RangeError"
        );
        // A within-cap grow / length set still works (no regression).
        assert_eq!(
            run("var a=[1]; a[5]=9; JSON.stringify(a)"),
            "[1,null,null,null,null,9]"
        );
        assert_eq!(
            run("var a=[1,2,3,4,5]; a.length=2; JSON.stringify(a)"),
            "[1,2]"
        );
    }

    /// C2: a deeply nested expression (shallow in the AST via the precedence loop,
    /// but thousands of native `eval` recursions) throws a catchable `RangeError`
    /// rather than overflowing the host stack. Run on a generous stack so the
    /// `max_eval_depth` guard fires before the (much larger) real overflow point,
    /// exactly as the production / test262 harness threads do.
    #[test]
    fn deep_expression_throws_instead_of_overflowing() {
        let handle = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let src = core::iter::repeat_n("1", 20_000)
                    .collect::<alloc::vec::Vec<_>>()
                    .join("+");
                // Leak the AST: dropping a 20k-deep boxed expression chain would
                // itself recurse and is unrelated to what we are asserting.
                let program = alloc::boxed::Box::leak(alloc::boxed::Box::new(
                    Parser::parse_program(&src).expect("parse"),
                ));
                let mut interp = Interp::new();
                let threw = matches!(interp.run(program), Err(ExecError::Throw(_)));
                core::mem::forget(interp);
                threw
            })
            .expect("spawn")
            .join()
            .expect("join");
        assert!(handle, "deep expression should throw, not abort");
    }

    /// L1: `ArrayBuffer.prototype.transfer(n)` with an enormous length throws a
    /// catchable `RangeError` (via `validate_alloc_len`) instead of attempting a
    /// `usize::MAX` allocation that aborts the process.
    #[test]
    fn array_buffer_transfer_huge_length_throws() {
        assert_eq!(
            run(
                "var b=new ArrayBuffer(4); try{b.transfer(1e309);'noThrow'}catch(e){e.constructor.name}"
            ),
            "RangeError"
        );
        // A reasonable resize still works.
        assert_eq!(
            run("var b=new ArrayBuffer(4); b.transfer(8).byteLength"),
            "8"
        );
    }

    #[test]
    fn surrogate_search_pad_split_iteration_units() {
        // indexOf/includes over UTF-16 units (astral char shifts the index by 2).
        assert_eq!(run(r#""😀x".indexOf("x") === 2"#), "true");
        assert_eq!(run(r#""😀x".includes("x")"#), "true");
        assert_eq!(run(r#""a😀b".lastIndexOf("b") === 3"#), "true");
        assert_eq!(run(r#""😀b".startsWith("😀")"#), "true");
        assert_eq!(run(r#""a😀".endsWith("😀")"#), "true");
        // padStart/padEnd count UTF-16 units; an astral pad char counts as two.
        assert_eq!(run(r#""x".padStart(3, "😀").length === 3"#), "true");
        assert_eq!(run(r#""x".padEnd(2).length === 2"#), "true");
        // split('') yields one entry per UTF-16 unit (astral → two halves).
        assert_eq!(run(r#""😀".split("").length === 2"#), "true");
        assert_eq!(run(r#""a\uD800b".split("").length === 3"#), "true");
        // for-of yields code points: an astral char is a single iteration.
        assert_eq!(run(r#"[..."😀"].length === 1"#), "true");
        assert_eq!(run(r#"[..."a\uD800b"].length === 3"#), "true");
        // repeat/concat preserve surrogates losslessly.
        assert_eq!(run(r#""\uD800".repeat(2).length === 2"#), "true");
        assert_eq!(
            run(r#""\uD800".repeat(2).charCodeAt(1) === 0xD800"#),
            "true"
        );
        assert_eq!(
            run(r#"("a".concat("\uD800")).charCodeAt(1) === 0xD800"#),
            "true"
        );
    }

    #[cfg(feature = "regex")]
    #[test]
    fn regex_u16_code_unit_indices_and_uflag() {
        // u-flag: `.` matches the whole astral char (one code point) but reports
        // its span in code units (length 2) at code-unit index 0.
        assert_eq!(run(r#"/./u.exec("😀")[0].length === 2"#), "true");
        assert_eq!(run(r#"/./u.exec("😀").index === 0"#), "true");
        // Non-u: `.` matches one code unit, so an astral subject yields two whole
        // matches and the first match is one unit long.
        assert_eq!(run(r#"/./.exec("😀")[0].length === 1"#), "true");
        assert_eq!(run(r#""😀".match(/./g).length === 2"#), "true");
        // A literal astral match under the u-flag, spliced back, is byte-stable.
        assert_eq!(run(r#""a😀b".replace(/😀/u, "X") === "aXb""#), "true");
        // `.index` / `lastIndex` / matchAll index are code-unit indices on an
        // astral subject.
        assert_eq!(run(r#""a😀b".search(/b/) === 3"#), "true");
        assert_eq!(
            run(r#"{ const r=/b/g; r.exec("😀b"); r.lastIndex === 3 }"#),
            "true"
        );
        assert_eq!(
            run(r#"[..."a😀b".matchAll(/(.)/gu)].map(m=>m.index).join(",") === "0,1,3""#),
            "true"
        );
        // split over an astral subject keeps surrounding text whole.
        assert_eq!(run(r#""a😀b".split(/😀/).join("|") === "a|b""#), "true");
        // `$&`/`$1`/`` $` ``/`$'` substitutions operate on code-unit slices and
        // re-encode astral characters losslessly.
        assert_eq!(
            run(r#""x😀y".replace(/(😀)/, "[$1]") === "x[😀]y""#),
            "true"
        );
        assert_eq!(run(r#""a😀b".replace(/😀/, "$`$'") === "aabb""#), "true");
        // A surrogate-bearing subject (forces the tree-walker path) matches via
        // its code units and the captured slice carries the lone surrogate.
        assert_eq!(
            run(r#""a\uD800b".replace(/\uD800/u, "X") === "aXb""#),
            "true"
        );
    }

    #[test]
    fn case_and_normalize_preserve_surrogates() {
        // A lone surrogate has no case and survives toUpperCase/toLowerCase.
        assert_eq!(
            run(r#""\uD800".toUpperCase().charCodeAt(0) === 0xD800"#),
            "true"
        );
        assert_eq!(
            run(r#""\uDC00".toLowerCase().charCodeAt(0) === 0xDC00"#),
            "true"
        );
        // Surrounding scalars still case-map; the surrogate stays put.
        assert_eq!(run(r#""a\uD800b".toUpperCase() === "A\uD800B""#), "true");
        // The surrogate-free fast path is unchanged, including `ß`→`SS`.
        assert_eq!(run(r#""abc".toUpperCase() === "ABC""#), "true");
        assert_eq!(run(r#""ß".toUpperCase() === "SS""#), "true");
        assert_eq!(run(r#""ABC".toLowerCase() === "abc""#), "true");
        // normalize is the identity on a lone surrogate (it round-trips).
        assert_eq!(
            run(r#""\uD800".normalize().charCodeAt(0) === 0xD800"#),
            "true"
        );
        assert_eq!(
            run(r#""a\uD800é".normalize("NFC").charCodeAt(1) === 0xD800"#),
            "true"
        );
        // A surrogate-free string still normalizes (NFC composes here).
        assert_eq!(run(r#""é".normalize("NFC") === "é""#), "true");
    }

    #[test]
    fn json_preserves_lone_surrogates() {
        // stringify escapes a lone surrogate as `\uXXXX` (well-formed JSON).
        assert_eq!(run(r#"JSON.stringify("\uD800") === '"\\ud800"'"#), "true");
        // A valid astral char round-trips as the character.
        assert_eq!(run(r#"JSON.stringify("😀") === '"😀"'"#), "true");
        // parse of a `\uXXXX` lone surrogate preserves it.
        assert_eq!(
            run(r#"JSON.parse('"\\ud800"').charCodeAt(0) === 0xD800"#),
            "true"
        );
        assert_eq!(run(r#"JSON.parse('"\\ud800"').length === 1"#), "true");
        // parse pairs `😀` into one astral code point.
        assert_eq!(
            run(r#"JSON.parse('"\\ud83d\\ude00"').codePointAt(0) === 0x1F600"#),
            "true"
        );
        // Round-trip a string with an embedded lone surrogate.
        assert_eq!(
            run(r#"JSON.parse(JSON.stringify("a\uD800b")).charCodeAt(1) === 0xD800"#),
            "true"
        );
    }

    #[test]
    fn non_surrogate_strings_behave_as_before() {
        // A plain corpus must be unchanged by the WTF-8 storage move.
        assert_eq!(run(r#""hello".length"#), "5");
        assert_eq!(run(r#""héllo 中".length"#), "7");
        assert_eq!(run(r#""abcde".slice(1,3)"#), "bc");
        assert_eq!(run(r#""a,b,c".split(",").length"#), "3");
        assert_eq!(run(r#""banana".indexOf("na")"#), "2");
        assert_eq!(run(r#""banana".lastIndexOf("na")"#), "4");
        assert_eq!(run(r#""x".padStart(3, "ab")"#), "abx");
        assert_eq!(
            run(r#"JSON.stringify({a:1,b:"hi"})"#),
            r#"{"a":1,"b":"hi"}"#
        );
        assert_eq!(run(r#"`a${1}b${2}c`"#), "a1b2c");
    }

    #[test]
    fn limits_override_changes_runtime_caps() {
        use crate::limits::Limits;
        // A lowered `max_string_len` rejects a concatenation the default accepts,
        // proving the cap is read live from `realm.limits` rather than a constant.
        let src = "'abcde'.repeat(3)"; // 15 chars
        assert_eq!(eval_source(src).expect("default ok").1, "abcdeabcdeabcde");
        let low = Limits {
            max_string_len: 10,
            ..Limits::default()
        };
        let err = eval_source_with_limits(src, low).expect_err("should exceed length");
        assert!(err.contains("Invalid string length"), "unexpected: {err}");

        // A low object→dictionary threshold forces the conversion early yet keeps
        // correct property semantics (count, values, insertion order preserved).
        let dict = Limits {
            object_dictionary_threshold: 4,
            ..Limits::default()
        };
        let keys_src = "let o={}; for(let i=0;i<10;i++) o['k'+i]=i; [Object.keys(o).length, o.k0, o.k9, Object.keys(o)[0]].join(',')";
        assert_eq!(
            eval_source_with_limits(keys_src, dict).expect("dict ok").1,
            "10,0,9,k0"
        );
    }

    /// C2 follow-up: a custom low `max_eval_depth` (via `Realm::with_limits`,
    /// threaded through `eval_source_with_limits`) is honored live. The tree-walk
    /// recursion that the interpreter performs on a deeply nested expression
    /// trips the dedicated knob — a depth the *default* realm evaluates fine is
    /// rejected once the cap is lowered, proving `max_eval_depth` bounds the
    /// eval/exec recursion independently of `max_call_depth`.
    #[test]
    fn max_eval_depth_override_honored() {
        // Each tree-walk level burns a lot of native stack, so run on a generous
        // stack (like the production / test262 threads) where the *guard*, not a
        // real overflow, is the limiting factor.
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                use crate::limits::Limits;
                // A left-deep `1+1+…+1`: shallow allocations but `depth` nested
                // native `eval` recursions (one per `+` term), driving
                // `eval_depth` up by one per level within a single frame.
                fn deep_add(depth: usize) -> String {
                    core::iter::repeat_n("1", depth)
                        .collect::<alloc::vec::Vec<_>>()
                        .join("+")
                }

                // 600 terms evaluate cleanly under the default cap…
                let src = deep_add(600);
                assert_eq!(eval_source(&src).expect("default ok").1, "600");

                // …but a realm whose `max_eval_depth` is lowered below that depth
                // rejects the very same source with a catchable stack-overflow
                // `RangeError`, while `max_call_depth` is left at its (much
                // higher) default — proving the dedicated knob is honored live.
                let low = Limits {
                    max_eval_depth: 100,
                    ..Limits::default()
                };
                let err = eval_source_with_limits(&src, low).expect_err("should exceed eval depth");
                assert!(
                    err.contains("Maximum call stack size exceeded"),
                    "unexpected error: {err}"
                );
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    /// C2 follow-up: interpreter recursion past `max_eval_depth` throws a
    /// catchable `RangeError` (caught by a JS `try/catch`, surfacing as
    /// `RangeError`) instead of crashing the host. Run on a generous native
    /// stack so the guard — not a real overflow — is what stops the recursion.
    #[test]
    fn deep_eval_recursion_throws_range_error_catchable() {
        let kind = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                // A left-deep `1+1+…+1` far past the default `max_eval_depth`
                // (1500): each `+` term is a native `eval` recursion, so the
                // guard fires mid-evaluation and the throw is caught by JS.
                let deep = core::iter::repeat_n("1", 20_000)
                    .collect::<alloc::vec::Vec<_>>()
                    .join("+");
                let src = alloc::format!(
                    "try {{ {deep}; 'noThrow' }} catch (e) {{ e.constructor.name }}"
                );
                // Leak the AST: dropping a 20k-deep boxed expression chain would
                // itself recurse and is unrelated to what we are asserting.
                let program = alloc::boxed::Box::leak(alloc::boxed::Box::new(
                    Parser::parse_program(&src).expect("parse"),
                ));
                let mut interp = Interp::new();
                let res = interp.run(program).map(|v| interp.display(v));
                core::mem::forget(interp);
                res
            })
            .expect("spawn")
            .join()
            .expect("join");
        assert_eq!(kind.expect("eval ok"), "RangeError");
    }

    /// Runs `src` and returns its captured `console` output.
    fn out(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        interp.run(&program).expect("exec");
        String::from(interp.output())
    }

    /// With the `intl` crate, `Intl.Segmenter` uses real UAX-29 grapheme clusters — an emoji
    /// stays a single segment (the no-`intl` fallback splits per code point).
    #[cfg(feature = "intl")]
    #[test]
    fn intl_segmenter_real_grapheme_clusters() {
        assert_eq!(
            out(
                r#"console.log([...new Intl.Segmenter("en").segment("a😀b")].map(s=>s.segment).join("|"))"#
            ),
            "a|😀|b\n"
        );
    }

    /// With the `intl` crate, `Intl.PluralRules` applies real CLDR rules — Polish has the
    /// `few`/`many` categories the en-only fallback can't express.
    #[cfg(feature = "intl")]
    #[test]
    fn intl_plural_rules_real_cldr_categories() {
        assert_eq!(
            out(
                r#"var p=new Intl.PluralRules("pl");console.log([1,2,5,22].map(n=>p.select(n)).join(","))"#
            ),
            "one,few,many,few\n"
        );
    }

    /// With the `intl` crate, `Intl.Collator` does real UCA collation: `numeric` order sorts
    /// "a2" before "a10", and accents sort after their base letter (the fallback is code-point
    /// order, where "a10" < "a2" and accents sort far from their base).
    #[cfg(feature = "intl")]
    #[test]
    fn intl_collator_real_uca() {
        assert_eq!(
            out(r#"console.log(new Intl.Collator("en",{numeric:true}).compare("a2","a10"))"#),
            "-1\n"
        );
        assert_eq!(
            out(
                r#"console.log(["é","a","z","b"].sort(new Intl.Collator("en").compare).join(","))"#
            ),
            "a,b,é,z\n"
        );
    }

    /// With the `intl` crate, `Intl.NumberFormat` is locale-aware — German currency uses the
    /// `1.234,50 €` pattern (comma decimal, trailing symbol), unlike the en-only fallback.
    #[cfg(feature = "intl")]
    #[test]
    fn intl_numberformat_is_locale_aware() {
        assert_eq!(
            out(
                r#"console.log(new Intl.NumberFormat("de-DE",{style:"currency",currency:"EUR"}).format(1234.5))"#
            ),
            "1.234,50\u{a0}€\n"
        );
    }

    /// With the `intl` crate, `Intl.DateTimeFormat` is locale-aware (German month name +
    /// day-month-year order), which the en-only fallback can't produce.
    #[cfg(feature = "intl")]
    #[test]
    fn intl_datetime_is_locale_aware() {
        assert_eq!(
            out(
                r#"console.log(new Intl.DateTimeFormat("de",{timeZone:"UTC",dateStyle:"long"}).format(new Date(Date.UTC(2024,0,15))))"#
            ),
            "15. Januar 2024\n"
        );
    }

    /// With the `intl` crate, `Intl.DisplayNames` / `Intl.ListFormat` are locale-aware (the
    /// en-only fallback ignores the locale argument).
    #[cfg(feature = "intl")]
    #[test]
    fn intl_display_and_list_are_locale_aware() {
        assert_eq!(
            out(r#"console.log(new Intl.DisplayNames("de",{type:"region"}).of("US"))"#),
            "Vereinigte Staaten\n"
        );
        assert_eq!(
            out(r#"console.log(new Intl.ListFormat("es").format(["a","b","c"]))"#),
            "a, b y c\n"
        );
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

    /// The public D′ API (`Interp::snapshot` / `restore_snapshot`): snapshot a
    /// live closure's state in one interpreter and reload it into a *fresh* one
    /// holding the same code, through the supported library surface alone — no
    /// reaching into interpreter internals.
    #[test]
    fn public_snapshot_api_round_trips_across_runtimes() {
        let program = Parser::parse_program(
            "function makeCounter(start){ var n = start; return function(){ return ++n; }; } makeCounter(0)",
        )
        .expect("parse");

        // Runtime A: advance a counter to n = 2, snapshot it to bytes.
        let mut a = Interp::new();
        let f = a.run(&program).expect("exec A");
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(1.0));
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(2.0));
        let bytes = a.snapshot(&[f]);
        drop(a);

        // Runtime B: a fresh interpreter compiles the same program, then reloads
        // A's snapshot and runs the restored closure — resuming from n = 2.
        let mut b = Interp::new();
        let own = b.run(&program).expect("exec B");
        let restored = b.restore_snapshot(&bytes).expect("restore");
        assert_eq!(restored.len(), 1, "one heap root restored");
        assert_eq!(
            b.call(restored[0], &[]).unwrap().as_number(),
            Some(3.0),
            "restored closure resumes from snapshotted state"
        );
        assert_eq!(
            b.call(own, &[]).unwrap().as_number(),
            Some(1.0),
            "the fresh runtime's own counter is independent"
        );

        // A malformed snapshot is rejected, not panicked on.
        assert!(b.restore_snapshot(b"not a snapshot").is_err());
    }

    /// Cross-runtime D′ reload: snapshot a closure in one runtime, serialize it,
    /// then restore and **execute** it in a *separate, fresh* runtime that holds
    /// the same code — the load → evict → reload scenario. The restored closure
    /// carries the snapshotted captured state and is independent of the fresh
    /// runtime's own instance of the program.
    #[test]
    fn snapshot_reloads_into_a_fresh_runtime() {
        use crate::snapshot::{capture, deserialize, restore, serialize};

        // `makeCounter` (func 0) returns the inner closure (func 1); both runtimes
        // compile the same program, so the snapshot's `func_id`s line up.
        let program = Parser::parse_program(
            "function makeCounter(start){ var n = start; return function(){ return ++n; }; } makeCounter(0)",
        )
        .expect("parse");

        // Runtime A: build a counter, advance it to n = 2, snapshot it to bytes.
        let mut a = Interp::new();
        let f = a.run(&program).expect("exec A");
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(1.0));
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(2.0));
        let fh = Handle::from_raw(f.as_handle().expect("closure"));
        let bytes = serialize(&capture(&a.realm, &[fh]));
        drop(a); // A is gone — only the bytes survive.

        // Runtime B: a fresh interpreter that compiles the same program (its own
        // counter starts at 0), then reloads A's snapshot and runs the restored
        // closure — which resumes from the snapshotted n = 2.
        let mut b = Interp::new();
        let own = b.run(&program).expect("exec B");
        let snap = deserialize(&bytes).expect("deserialize");
        let restored = restore(&mut b.realm, &snap);
        let f2 = NanBox::handle(restored[0].to_raw());

        assert_eq!(
            b.call(f2, &[]).unwrap().as_number(),
            Some(3.0),
            "restored closure resumes from the snapshotted state in the new runtime"
        );
        assert_eq!(
            b.call(own, &[]).unwrap().as_number(),
            Some(1.0),
            "the fresh runtime's own counter is independent of the reloaded one"
        );
    }

    /// End-to-end D′: a live closure's captured state survives capture →
    /// serialize → deserialize → restore *and remains executable* — the restored
    /// closure runs, carries the snapshotted captured value, and is independent of
    /// the original. (Same interpreter, so its function table already holds the
    /// bodies the snapshot's `func_id`s refer to.)
    #[test]
    fn snapshot_restores_an_executable_closure() {
        use crate::snapshot::{capture, deserialize, restore, serialize};

        let program = Parser::parse_program(
            "function counter(){ var n = 0; return function(){ return ++n; }; } counter()",
        )
        .expect("parse");
        let mut interp = Interp::new();
        let f = interp.run(&program).expect("exec");
        let fh = Handle::from_raw(f.as_handle().expect("closure handle"));

        // Advance the original to n = 2, then snapshot it there.
        assert_eq!(interp.call(f, &[]).unwrap().as_number(), Some(1.0));
        assert_eq!(interp.call(f, &[]).unwrap().as_number(), Some(2.0));
        let snap = capture(&interp.realm, &[fh]);

        // Full on-disk round-trip.
        let snap = deserialize(&serialize(&snap)).expect("deserialize");

        // Restore into the same runtime (whose function table still has the body).
        let restored = restore(&mut interp.realm, &snap);
        let f2 = NanBox::handle(restored[0].to_raw());

        // The original kept counting (n was 2 → 3); the restored closure starts
        // from the *snapshotted* n = 2 → 3, proving it both executes and carries
        // the captured value.
        assert_eq!(
            interp.call(f, &[]).unwrap().as_number(),
            Some(3.0),
            "original continues"
        );
        assert_eq!(
            interp.call(f2, &[]).unwrap().as_number(),
            Some(3.0),
            "restored from snapshot"
        );

        // Independence: advancing the restored copy does not move the original.
        assert_eq!(
            interp.call(f2, &[]).unwrap().as_number(),
            Some(4.0),
            "restored advances"
        );
        assert_eq!(
            interp.call(f, &[]).unwrap().as_number(),
            Some(4.0),
            "original unaffected by restore"
        );
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
        // Calling a non-callable is a *catchable* JS `TypeError` (an
        // `ExecError::Throw`), not an internal `NotCallable` — so user `try/catch`
        // can handle it (per ECMA-262 Call, step 2).
        let program = Parser::parse_program("let x = 5; x()").unwrap();
        let mut interp = Interp::new();
        assert!(matches!(interp.run(&program), Err(ExecError::Throw(_))));
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

    #[test]
    fn builtin_functions() {
        // Math methods (variadic).
        assert_eq!(run("Math.max(3, 7, 2)"), "7");
        assert_eq!(run("Math.min(3, 7, 2)"), "2");
        assert_eq!(run("Math.abs(-5)"), "5");
        // Coercion globals.
        assert_eq!(run("String(42)"), "42");
        assert_eq!(run("Number('3.5')"), "3.5");
        assert_eq!(run("Boolean(0)"), "false");
        assert_eq!(run("Boolean('x')"), "true");
        assert_eq!(run("parseInt('42px')"), "42");
        assert_eq!(run("parseInt('  -7 ')"), "-7");
        // typeof a built-in is "function".
        assert_eq!(run("typeof Math.max"), "function");
        // Built-ins compose with user code.
        assert_eq!(
            run(
                "function clamp(x, lo, hi) { return Math.max(lo, Math.min(x, hi)); }
                 clamp(15, 0, 10)"
            ),
            "10"
        );
    }

    #[test]
    fn string_methods() {
        assert_eq!(run("'hello'.toUpperCase()"), "HELLO");
        assert_eq!(run("'HELLO'.toLowerCase()"), "hello");
        assert_eq!(run("'  hi  '.trim()"), "hi");
        assert_eq!(run("'hello'.charAt(1)"), "e");
        assert_eq!(run("'hello'.includes('ell')"), "true");
        assert_eq!(run("'hello'.indexOf('l')"), "2");
        assert_eq!(run("'ab'.repeat(3)"), "ababab");
    }

    #[test]
    fn array_methods() {
        assert_eq!(run("let a = [1, 2]; a.push(3); a.join('-')"), "1-2-3");
        assert_eq!(run("let a = [1, 2, 3]; a.pop()"), "3");
        assert_eq!(run("[1, 2, 3].includes(2)"), "true");
        assert_eq!(run("[1, 2, 3].indexOf(3)"), "2");
        assert_eq!(run("['a', 'b', 'c'].join(', ')"), "a, b, c");
        // splice (remove + insert), unshift, shift.
        assert_eq!(
            run("let a=[1,2,3,4]; a.splice(1,2,'x'); a.join(',')"),
            "1,x,4"
        );
        assert_eq!(run("[1,2,3,4].splice(1,2).join(',')"), "2,3");
        assert_eq!(run("let a=[2,3]; a.unshift(0,1); a.join(',')"), "0,1,2,3");
        assert_eq!(
            run("let a=[1,2,3]; let f=a.shift(); f + ':' + a.join(',')"),
            "1:2,3"
        );
        // Non-mutating: toSorted/toReversed/with leave the original.
        assert_eq!(
            run(
                "let a=[3,1,2]; let s=a.toSorted(function(x,y){return x-y;}); s.join('')+'|'+a.join('')"
            ),
            "123|312"
        );
        assert_eq!(run("[1,2,3].toReversed().join(',')"), "3,2,1");
        assert_eq!(run("[1,2,3].with(1,9).join(',')"), "1,9,3");
        // includes uses SameValueZero (NaN matches).
        assert_eq!(run("[NaN, 1].includes(NaN)"), "true");
        assert_eq!(run("[1, 2].includes(NaN)"), "false");
    }

    #[test]
    fn define_property_and_locale_compare() {
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'x',{value:42}); o.x"),
            "42"
        );
        assert_eq!(
            run(
                "let o={n:1}; Object.defineProperty(o,'d',{get:function(){return this.n+1;}}); o.d"
            ),
            "2"
        );
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:7}); Object.getOwnPropertyDescriptor(o,'x').value"
            ),
            "7"
        );
        assert_eq!(run("'apple'.localeCompare('banana') < 0"), "true");
        assert_eq!(run("'x'.localeCompare('x')"), "0");
        // fill: whole array, a [start,end) range, and a negative start.
        assert_eq!(run("[0, 0, 0].fill(7).join(',')"), "7,7,7");
        assert_eq!(run("[1, 2, 3, 4].fill(9, 1, 3).join(',')"), "1,9,9,4");
        assert_eq!(run("[1, 2, 3, 4, 5].fill(0, -2).join(',')"), "1,2,3,0,0");
        // reduceRight folds right-to-left, with and without a seed.
        assert_eq!(
            run("['a','b','c'].reduceRight(function(acc,x){ return acc + x; })"),
            "cba"
        );
        assert_eq!(
            run("[1,2,3].reduceRight(function(a,x){ return a + x; }, 10)"),
            "16"
        );
        // findLast / findLastIndex scan right-to-left.
        assert_eq!(
            run("[1,2,3,4].findLast(function(x){ return x % 2 === 1; })"),
            "3"
        );
        assert_eq!(
            run("[1,2,3,4].findLastIndex(function(x){ return x % 2 === 1; })"),
            "2"
        );
        assert_eq!(
            run("[2,4,6].findLast(function(x){ return x > 9; })"),
            "undefined"
        );
        // copyWithin (in place) and flat(depth).
        assert_eq!(run("[1,2,3,4,5].copyWithin(0,3).join(',')"), "4,5,3,4,5");
        assert_eq!(run("[1,[2,[3,[4]]]].flat(2).join(',')"), "1,2,3,4");
    }

    #[test]
    fn math_extras_and_number_coercion() {
        assert_eq!(run("Math.hypot(3, 4)"), "5");
        assert_eq!(run("Math.cbrt(27)"), "3");
        assert_eq!(run("Math.log2(8)"), "3");
        assert_eq!(run("Math.log10(1000)"), "3");
        assert_eq!(run("(1234.5).toExponential(2)"), "1.23e+3");
        // Radix-prefixed string coercion (shared `to_number`, both engines).
        assert_eq!(run("Number('0x1F')"), "31");
        assert_eq!(run("+'0b101'"), "5");
        assert_eq!(run("'0o17' * 1"), "15");
    }

    #[test]
    fn split_limit_and_to_precision() {
        assert_eq!(run("'a,b,c,d'.split(',', 2).length"), "2");
        assert_eq!(run("'aXbXc'.split('X').join('-')"), "a-b-c");
        assert_eq!(run("(123.456).toPrecision(4)"), "123.5");
        assert_eq!(run("(255).toString(2)"), "11111111");
    }

    #[test]
    fn object_freeze_family() {
        // Writes and new properties are no-ops on a frozen object.
        assert_eq!(
            run(
                "let o = Object.freeze({ a: 1 }); o.a = 9; o.b = 2; o.a + ',' + (o.b === undefined)"
            ),
            "1,true"
        );
        assert_eq!(run("Object.isFrozen(Object.freeze({}))"), "true");
        assert_eq!(run("Object.isFrozen({})"), "false");
        assert_eq!(
            run("Object.getOwnPropertyNames({ x: 1, y: 2 }).length"),
            "2"
        );
    }

    #[test]
    fn string_pad_lastindexof_and_number_statics() {
        assert_eq!(run("'5'.padEnd(3, '-')"), "5--");
        assert_eq!(run("'ab'.padEnd(5)"), "ab   ");
        assert_eq!(run("'a-b-c'.lastIndexOf('-')"), "3");
        assert_eq!(run("'abc'.lastIndexOf('x')"), "-1");
        assert_eq!(run("Number.MAX_SAFE_INTEGER"), "9007199254740991");
        assert_eq!(run("Number.POSITIVE_INFINITY"), "Infinity");
        assert_eq!(run("'abc'.concat('def', '!')"), "abcdef!");
    }

    #[test]
    fn console_log_captures_output() {
        let program =
            Parser::parse_program("console.log('hi', 42); let x = [1, 2]; console.log('arr:', x);")
                .unwrap();
        let mut interp = Interp::new();
        interp.run(&program).unwrap();
        assert_eq!(interp.output(), "hi 42\narr: 1,2\n");
    }

    #[test]
    fn json_getters_tojson_and_date_utc() {
        // JSON.stringify invokes getters and honors toJSON.
        assert_eq!(
            run("JSON.stringify({a:1, get b(){ return 2; }})"),
            "{\"a\":1,\"b\":2}"
        );
        assert_eq!(
            run("JSON.stringify({v:42, toJSON(){ return {w:this.v}; }})"),
            "{\"w\":42}"
        );
        assert_eq!(run("JSON.stringify([undefined,1])"), "[null,1]");
        // Date.UTC and getUTC* methods.
        assert_eq!(run("new Date(Date.UTC(2024,0,15)).getUTCDate()"), "15");
        assert_eq!(run("new Date(Date.UTC(2024,0,1)).getUTCDay()"), "1"); // Monday
        assert_eq!(run("new Date(0).toISOString()"), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn date_invalid_toisostring() {
        assert_eq!(
            run("try{new Date(NaN).toISOString();'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        assert_eq!(run("new Date(NaN).toJSON()"), "null");
        assert_eq!(run("JSON.stringify({d:new Date(NaN)})"), "{\"d\":null}");
        assert_eq!(
            run("new Date(Date.UTC(2020,5,15,10,30,45,123)).toISOString()"),
            "2020-06-15T10:30:45.123Z"
        );
        assert_eq!(
            run("try{new Date('garbage').toISOString();'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
    }

    #[test]
    fn objects_inherit_object_prototype() {
        assert_eq!(run("'toString' in {}"), "true");
        assert_eq!(run("'hasOwnProperty' in {}"), "true");
        assert_eq!(
            run("Object.getPrototypeOf({}) === Object.prototype"),
            "true"
        );
        assert_eq!(run("'toString' in Object.create(null)"), "false");
        assert_eq!(
            run("Object.getPrototypeOf(Object.create(null)) === null"),
            "true"
        );
        // Inherited methods are non-enumerable.
        assert_eq!(run("Object.keys({a:1,b:2}).join(',')"), "a,b");
        assert_eq!(
            run("let s=[]; for(let k in {a:1,b:2}) s.push(k); s.join(',')"),
            "a,b"
        );
        assert_eq!(run("JSON.stringify({a:1})"), "{\"a\":1}");
        // hasOwnProperty distinguishes own vs inherited.
        assert_eq!(
            run(
                "let c=Object.create({i:1}); c.o=2; c.hasOwnProperty('o') + ',' + c.hasOwnProperty('i')"
            ),
            "true,false"
        );
    }

    #[test]
    fn object_prototype_tostring_call() {
        assert_eq!(run("typeof Object.prototype"), "object");
        assert_eq!(run("Object.prototype.toString.call({})"), "[object Object]");
        assert_eq!(run("Object.prototype.toString.call([])"), "[object Array]");
        assert_eq!(run("Object.prototype.toString.call(null)"), "[object Null]");
        assert_eq!(
            run("Object.prototype.toString.call(undefined)"),
            "[object Undefined]"
        );
        assert_eq!(
            run("Object.prototype.toString.call(function(){})"),
            "[object Function]"
        );
        assert_eq!(
            run("Object.prototype.toString.call(/x/)"),
            "[object RegExp]"
        );
        assert_eq!(
            run("Object.prototype.toString.call({[Symbol.toStringTag]:'Widget'})"),
            "[object Widget]"
        );
        assert_eq!(
            run("Object.prototype.hasOwnProperty.call({a:1},'a')"),
            "true"
        );
        assert_eq!(
            run("Object.prototype.hasOwnProperty.call({a:1},'b')"),
            "false"
        );
        assert_eq!(
            run("let p={}; Object.prototype.isPrototypeOf.call(p, Object.create(p))"),
            "true"
        );
        assert_eq!(
            run("let o={x:1}; Object.prototype.valueOf.call(o)===o"),
            "true"
        );
    }

    #[test]
    fn webassembly_instantiate_and_call() {
        // `WebAssembly.instantiate` now returns a Promise (per spec); use the
        // synchronous `new Instance(new Module(...))` path to read exports directly.
        let setup = "var b=[0,0x61,0x73,0x6d,1,0,0,0, 1,7,1,0x60,2,0x7f,0x7f,1,0x7f, 3,2,1,0, 7,7,1,3,0x61,0x64,0x64,0,0, 0xa,9,1,7,0,0x20,0,0x20,1,0x6a,0xb]; var e=new WebAssembly.Instance(new WebAssembly.Module(b)).exports; ";
        assert_eq!(run(&alloc::format!("{setup} typeof e.add")), "function");
        assert_eq!(run(&alloc::format!("{setup} e.add(20,22)")), "42");
        assert_eq!(run(&alloc::format!("{setup} e.add(-5,8)")), "3");
    }

    #[test]
    fn webassembly_validate_builtin() {
        assert_eq!(run("typeof WebAssembly"), "object");
        assert_eq!(run("typeof WebAssembly.validate"), "function");
        assert_eq!(
            run("WebAssembly.validate([0,0x61,0x73,0x6d,1,0,0,0])"),
            "true"
        );
        assert_eq!(run("WebAssembly.validate([0,0,0,0,1,0,0,0])"), "false");
        assert_eq!(run("WebAssembly.validate([0,0x61])"), "false");
    }

    #[test]
    fn object_default_tostring_and_tag() {
        assert_eq!(run("({}).toString()"), "[object Object]");
        assert_eq!(run("'abc'.toString()"), "abc");
        assert_eq!(run("({a:1}).valueOf().a"), "1");
        assert_eq!(run("({toString(){return 'custom';}}).toString()"), "custom");
        assert_eq!(
            run("Object.create({toString(){return 'base';}}).toString()"),
            "base"
        );
        assert_eq!(
            run("({[Symbol.toStringTag]:'Widget'}).toString()"),
            "[object Widget]"
        );
        assert_eq!(run("typeof Symbol.toStringTag"), "symbol");
        assert_eq!(run("typeof Symbol.species"), "symbol");
        assert_eq!(
            run("let o={[Symbol.toStringTag]:'X'}; Object.getOwnPropertySymbols(o).length"),
            "1"
        );
        // Existing toStrings unaffected.
        assert_eq!(run("[1,2,3].toString()"), "1,2,3");
        assert_eq!(run("new Error('x').toString()"), "Error: x");
    }

    #[test]
    fn strict_getter_only_write() {
        // Sloppy: silently ignored.
        assert_eq!(run("let o={get x(){return 1;}}; o.x=2; o.x"), "1");
        // Strict: throws TypeError.
        assert_eq!(
            run(
                "(function(){'use strict'; let o={get x(){return 1;}}; try{o.x=2;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
        // A setter still works under strict mode.
        assert_eq!(
            run(
                "(function(){'use strict'; let o={_v:0,get x(){return this._v;},set x(v){this._v=v*2;}}; o.x=5; return o.x;})()"
            ),
            "10"
        );
        // Inherited getter-only accessor.
        assert_eq!(
            run(
                "(function(){'use strict'; let o=Object.create({get y(){return 9;}}); try{o.y=1;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
    }

    #[test]
    fn reduce_empty_throws_typeerror() {
        assert_eq!(
            run("try{[].reduce((a,b)=>a+b);'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
        assert_eq!(
            run("try{[].reduceRight((a,b)=>a+b);'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
        assert_eq!(run("[].reduce((a,b)=>a+b, 99)"), "99");
        assert_eq!(run("[42].reduce((a,b)=>a+b)"), "42");
        assert_eq!(run("[1,2,3,4].reduce((a,b)=>a+b)"), "10");
    }

    #[test]
    fn function_names_tostring_bound() {
        assert_eq!(run("let myFn=function(){}; myFn.name"), "myFn");
        assert_eq!(run("let arrow=()=>1; arrow.name"), "arrow");
        assert_eq!(run("let x=function inner(){}; x.name"), "inner");
        assert_eq!(
            run("let o={method(){},fn:()=>1}; o.method.name + ':' + o.fn.name"),
            "method:fn"
        );
        assert_eq!(run("typeof (function f(){}).toString()"), "string");
        assert_eq!(
            run("(function f(){}).toString().indexOf('function')>=0"),
            "true"
        );
        assert_eq!(run("function t(a,b,c){} t.bind(null,1).length"), "2");
        assert_eq!(run("function t(a,b,c){} t.bind(null,1,2,3,4).length"), "0");
        assert_eq!(run("function t(){} t.bind(null).name"), "bound t");
    }

    #[test]
    fn proto_accessor() {
        // Object.create + __proto__ read.
        assert_eq!(
            run("let p={greet(){return 'hi';}}; let o=Object.create(p); o.__proto__===p"),
            "true"
        );
        assert_eq!(
            run("let p={greet(){return 'hi';}}; Object.create(p).greet()"),
            "hi"
        );
        // __proto__ assignment relinks.
        assert_eq!(
            run("let o={}; o.__proto__={hello(){return 'yo';}}; o.hello()"),
            "yo"
        );
        assert_eq!(
            run("let b={getX(){return this.x;}}; let o={}; o.__proto__=b; o.x=42; o.getX()"),
            "42"
        );
        // Object-literal __proto__ sets the prototype.
        assert_eq!(
            run("let b={getX(){return this.x;}}; let n={__proto__:b, x:5}; n.getX()"),
            "5"
        );
        // The method form is a regular property, not a prototype set.
        assert_eq!(
            run("typeof ({__proto__(){return 1;}}).__proto__"),
            "function"
        );
        // __proto__ = null clears the chain; a primitive is ignored.
        assert_eq!(
            run("let o={__proto__:{}}; o.__proto__=null; Object.getPrototypeOf(o)===null"),
            "true"
        );
        assert_eq!(
            run("let p={g(){return 1;}}; let o=Object.create(p); o.__proto__=5; o.g()"),
            "1"
        );
    }

    #[test]
    fn proxy_passthrough_keys() {
        assert_eq!(run("Object.keys(new Proxy({a:1,b:2},{})).join(',')"), "a,b");
        assert_eq!(
            run("Object.values(new Proxy({a:1,b:2},{})).join(',')"),
            "1,2"
        );
        assert_eq!(
            run("Object.entries(new Proxy({a:1,b:2},{})).map(e=>e.join(':')).join(',')"),
            "a:1,b:2"
        );
        // Nested trap-less proxies forward through.
        assert_eq!(
            run("Object.keys(new Proxy(new Proxy({x:1},{}),{})).join(',')"),
            "x"
        );
    }

    #[test]
    fn bigint_as_uintn_intn() {
        assert_eq!(run("BigInt.asUintN(8, 256n)"), "0");
        assert_eq!(run("BigInt.asUintN(8, -1n)"), "255");
        assert_eq!(run("BigInt.asUintN(4, -1n)"), "15");
        assert_eq!(run("BigInt.asUintN(64, 18446744073709551617n)"), "1");
        assert_eq!(run("BigInt.asIntN(8, 200n)"), "-56");
        assert_eq!(run("BigInt.asIntN(8, 128n)"), "-128");
        assert_eq!(run("BigInt.asIntN(8, 127n)"), "127");
        assert_eq!(run("BigInt.asIntN(16, 40000n)"), "-25536");
        assert_eq!(run("BigInt.asIntN(32, 4294967295n)"), "-1");
        assert_eq!(
            run("BigInt.asIntN(128, 12345678901234567890n)"),
            "12345678901234567890"
        );
    }

    #[test]
    fn json_number_serialization() {
        assert_eq!(run("JSON.stringify(-0)"), "0");
        assert_eq!(run("JSON.stringify([-0])"), "[0]");
        assert_eq!(run("JSON.stringify({x:-0})"), "{\"x\":0}");
        assert_eq!(run("JSON.stringify(1e21)"), "1e+21");
        assert_eq!(run("JSON.stringify(1e-7)"), "1e-7");
        assert_eq!(run("JSON.stringify(1e20)"), "100000000000000000000");
        assert_eq!(run("JSON.stringify(0.001)"), "0.001");
        assert_eq!(run("JSON.stringify([NaN,Infinity])"), "[null,null]");
        assert_eq!(
            run("JSON.stringify([-0,1e21,0.001,-42])"),
            "[0,1e+21,0.001,-42]"
        );
    }

    #[test]
    fn json_stringify_replacer() {
        // Function replacer: omit keys and transform values (recursively).
        assert_eq!(
            run("JSON.stringify({a:1,b:2}, function(k,v){ return k==='b'?undefined:v; })"),
            "{\"a\":1}"
        );
        assert_eq!(
            run("JSON.stringify({x:{n:1}}, function(k,v){ return typeof v==='number'?v+5:v; })"),
            "{\"x\":{\"n\":6}}"
        );
        // Array replacer: an allowlist applied at every level.
        assert_eq!(
            run("JSON.stringify({a:1,b:2,c:3}, ['a','c'])"),
            "{\"a\":1,\"c\":3}"
        );
        assert_eq!(
            run("JSON.stringify({keep:{a:1,b:2},drop:9}, ['keep','a'])"),
            "{\"keep\":{\"a\":1}}"
        );
    }

    #[test]
    fn json_parse_reviver() {
        assert_eq!(
            run(
                "let o = JSON.parse('{\"a\":1,\"b\":2}', function(k,v){ return typeof v==='number'?v*2:v; }); o.a + ',' + o.b"
            ),
            "2,4"
        );
        assert_eq!(
            run(
                "let o = JSON.parse('{\"keep\":1,\"drop\":2}', function(k,v){ return k==='drop'?undefined:v; }); o.keep + ':' + ('drop' in o)"
            ),
            "1:false"
        );
        assert_eq!(
            run(
                "JSON.parse('[1,2,3]', function(k,v){ return typeof v==='number'?v+10:v; }).join(',')"
            ),
            "11,12,13"
        );
    }

    #[test]
    fn json_stringify() {
        assert_eq!(run("JSON.stringify(42)"), "42");
        assert_eq!(run("JSON.stringify('hi')"), "\"hi\"");
        assert_eq!(run("JSON.stringify(true)"), "true");
        assert_eq!(run("JSON.stringify(null)"), "null");
        assert_eq!(run("JSON.stringify([1, 2, 3])"), "[1,2,3]");
        assert_eq!(
            run("JSON.stringify({ a: 1, b: 'x' })"),
            "{\"a\":1,\"b\":\"x\"}"
        );
        assert_eq!(
            run("JSON.stringify({ nested: { list: [1, true, null] } })"),
            "{\"nested\":{\"list\":[1,true,null]}}"
        );
        // Indentation (the `space` argument): numeric and string, empties inline.
        assert_eq!(
            run("JSON.stringify({a:1,b:2}, null, 2)"),
            "{\n  \"a\": 1,\n  \"b\": 2\n}"
        );
        assert_eq!(run("JSON.stringify([1,2], null, '--')"), "[\n--1,\n--2\n]");
        assert_eq!(run("JSON.stringify({}, null, 2)"), "{}");
        assert_eq!(run("JSON.stringify([], null, 4)"), "[]");
        // A quote in a string is escaped.
        assert_eq!(run("JSON.stringify('a\"b')"), "\"a\\\"b\"");
    }

    #[test]
    fn object_and_array_statics() {
        assert_eq!(run("Object.keys({ a: 1, b: 2 }).join(',')"), "a,b");
        assert_eq!(run("Object.values({ a: 1, b: 2 }).join(',')"), "1,2");
        assert_eq!(run("Array.isArray([1, 2])"), "true");
        assert_eq!(run("Array.isArray('nope')"), "false");
        assert_eq!(run("Array.isArray({})"), "false");
    }

    #[cfg(feature = "std")]
    #[test]
    fn math_float_methods() {
        assert_eq!(run("Math.floor(3.7)"), "3");
        assert_eq!(run("Math.ceil(3.2)"), "4");
        assert_eq!(run("Math.round(3.5)"), "4");
        assert_eq!(run("Math.sqrt(144)"), "12");
    }

    #[test]
    fn classes_and_this() {
        // A class with a constructor and a method using `this`.
        assert_eq!(
            run("class Point {
                   constructor(x, y) { this.x = x; this.y = y; }
                   sum() { return this.x + this.y; }
                 }
                 let p = new Point(3, 4);
                 p.sum()"),
            "7"
        );
        // Methods mutate instance state via `this`.
        assert_eq!(
            run("class Counter {
                   constructor() { this.n = 0; }
                   inc() { this.n += 1; return this.n; }
                 }
                 let c = new Counter();
                 c.inc(); c.inc(); c.inc()"),
            "3"
        );
        // A field initializer.
        assert_eq!(
            run("class Box { value = 42; get() { return this.value; } }
                 new Box().get()"),
            "42"
        );
        // A method calling another method on `this`.
        assert_eq!(
            run("class Calc {
                   constructor(v) { this.v = v; }
                   double() { return this.v * 2; }
                   quadruple() { return this.double() * 2; }
                 }
                 new Calc(5).quadruple()"),
            "20"
        );
        // typeof a class is function.
        assert_eq!(run("class A {} typeof A"), "function");
        // A class expression.
        assert_eq!(
            run("let C = class { constructor() { this.k = 9; } }; new C().k"),
            "9"
        );
    }

    #[test]
    fn class_getters_setters_statics() {
        // A getter computes from instance state.
        assert_eq!(
            run("class C { constructor(w, h) { this.w = w; this.h = h; }
                   get area() { return this.w * this.h; } }
                 new C(3, 4).area"),
            "12"
        );
        // A setter mutates instance state.
        assert_eq!(
            run("class Temp {
                   constructor() { this.c = 0; }
                   get fahrenheit() { return this.c * 9 / 5 + 32; }
                   set fahrenheit(f) { this.c = (f - 32) * 5 / 9; }
                 }
                 let t = new Temp(); t.fahrenheit = 212; t.c"),
            "100"
        );
        // Static methods and fields.
        assert_eq!(
            run(
                "class MathX { static square(n) { return n * n; } static pi = 3; }
                 MathX.square(5) + MathX.pi"
            ),
            "28"
        );
        // A static factory returning an instance.
        assert_eq!(
            run("class Point {
                   constructor(x) { this.x = x; }
                   static origin() { return new Point(0); }
                 }
                 Point.origin().x"),
            "0"
        );
    }

    #[test]
    fn class_inheritance() {
        // A subclass inherits a base method.
        assert_eq!(
            run("class Animal {
                   constructor(name) { this.name = name; }
                   describe() { return this.name; }
                 }
                 class Dog extends Animal {}
                 new Dog('Rex').describe()"),
            "Rex"
        );
        // `super(...)` calls the base constructor; the derived adds state.
        assert_eq!(
            run("class Animal {
                   constructor(name) { this.name = name; }
                 }
                 class Dog extends Animal {
                   constructor(name, breed) { super(name); this.breed = breed; }
                   tag() { return this.name + ':' + this.breed; }
                 }
                 new Dog('Rex', 'Lab').tag()"),
            "Rex:Lab"
        );
        // A derived method overrides the base.
        assert_eq!(
            run("class A { kind() { return 'A'; } }
                 class B extends A { kind() { return 'B'; } }
                 new B().kind() + new A().kind()"),
            "BA"
        );
        // Implicit super (no derived constructor) forwards the args.
        assert_eq!(
            run("class Base { constructor(v) { this.v = v; } }
                 class Sub extends Base { get() { return this.v; } }
                 new Sub(7).get()"),
            "7"
        );
        // Three-level chain.
        assert_eq!(
            run("class A { constructor() { this.a = 1; } }
                 class B extends A { constructor() { super(); this.b = 2; } }
                 class C extends B { constructor() { super(); this.c = 3; } }
                 let o = new C(); o.a + o.b + o.c"),
            "6"
        );
    }

    #[test]
    fn object_array_statics_and_number_methods() {
        // Object.assign / entries.
        assert_eq!(
            run("let t = Object.assign({}, { a: 1 }, { b: 2 }); t.a + t.b"),
            "3"
        );
        assert_eq!(
            run("Object.entries({ a: 1, b: 2 }).map(e => e[0] + '=' + e[1]).join(',')"),
            "a=1,b=2"
        );
        // Array.from (string / array) and Array.of.
        assert_eq!(run("Array.from('abc').join('-')"), "a-b-c");
        assert_eq!(
            run("Array.from([1, 2, 3]).map(x => x * 2).join(',')"),
            "2,4,6"
        );
        assert_eq!(run("Array.of(1, 2, 3).join(',')"), "1,2,3");
        // Number methods.
        assert_eq!(run("(255).toString()"), "255");
    }

    #[cfg(feature = "std")]
    #[test]
    fn number_tofixed() {
        assert_eq!(run("(3.14159).toFixed(2)"), "3.14");
        assert_eq!(run("(1).toFixed(3)"), "1.000");
    }

    #[test]
    fn promises_and_microtasks() {
        // The `then` reactions run on the microtask queue, drained after the
        // script — observed via captured `console.log` output.
        let out = |src: &str| {
            let program = Parser::parse_program(src).expect("parse");
            let mut interp = Interp::new();
            interp.run(&program).expect("exec");
            String::from(interp.output())
        };
        // then runs after the synchronous code.
        assert_eq!(
            out("console.log('sync');
                 Promise.resolve(42).then(v => console.log('got:' + v));"),
            "sync\ngot:42\n"
        );
        // Chaining transforms the value through each then.
        assert_eq!(
            out("Promise.resolve(1).then(v => v + 1).then(v => v * 10).then(v => console.log(v));"),
            "20\n"
        );
        // catch handles a rejection.
        assert_eq!(
            out("Promise.reject('boom').catch(e => console.log('caught:' + e));"),
            "caught:boom\n"
        );
        // A throw in a then handler rejects the chain to the next catch.
        assert_eq!(
            out(
                "Promise.resolve(1).then(() => { throw 'x'; }).catch(e => console.log('rej:' + e));"
            ),
            "rej:x\n"
        );
        // `new Promise(executor)` with resolve.
        assert_eq!(
            out("new Promise((resolve) => { resolve(7); }).then(v => console.log(v));"),
            "7\n"
        );
        // Adoption: resolving with a promise chains its value.
        assert_eq!(
            out("Promise.resolve(Promise.resolve(99)).then(v => console.log(v));"),
            "99\n"
        );
        // typeof a promise is object.
        assert_eq!(run("typeof Promise.resolve(1)"), "object");
    }

    #[test]
    fn async_await() {
        // observe results via console.log after the microtask drain.
        let out = |src: &str| {
            let program = Parser::parse_program(src).expect("parse");
            let mut interp = Interp::new();
            interp.run(&program).expect("exec");
            String::from(interp.output())
        };
        // An async function returns a promise; awaiting unwraps values.
        assert_eq!(
            out(
                "async function f() { let x = await Promise.resolve(10); return x + 5; }
                 f().then(v => console.log(v));"
            ),
            "15\n"
        );
        // Awaiting in sequence.
        assert_eq!(
            out("async function g() {
                   let a = await Promise.resolve(2);
                   let b = await Promise.resolve(3);
                   return a * b;
                 }
                 g().then(v => console.log(v));"),
            "6\n"
        );
        // try/catch around a rejected await.
        assert_eq!(
            out("async function h() {
                   try { await Promise.reject('boom'); return 'no'; }
                   catch (e) { return 'caught:' + e; }
                 }
                 h().then(v => console.log(v));"),
            "caught:boom\n"
        );
        // An async arrow, awaiting a plain value.
        assert_eq!(
            out("let f = async (x) => (await x) + 1;
                 f(41).then(v => console.log(v));"),
            "42\n"
        );
        // typeof an async function is function; its call returns a promise.
        assert_eq!(run("async function a() {} typeof a"), "function");
        assert_eq!(run("async function a() { return 1; } typeof a()"), "object");
    }

    #[cfg(feature = "regex")]
    #[test]
    fn regexp() {
        // Regex literal + test.
        assert_eq!(run("/ab+c/.test('xxabbbcyy')"), "true");
        assert_eq!(run("/^\\d+$/.test('12345')"), "true");
        assert_eq!(run("/^\\d+$/.test('12a45')"), "false");
        // case-insensitive flag.
        assert_eq!(run("/hello/i.test('HELLO')"), "true");
        // new RegExp(...).
        assert_eq!(run("new RegExp('a.c').test('axc')"), "true");
        // exec returns the matched substring (or null).
        assert_eq!(run("/b+/.exec('aabbbc')[0]"), "bbb");
        assert_eq!(run("/zzz/.exec('abc') === null"), "true");
        // A regex renders as /source/flags; typeof is object.
        assert_eq!(run("'' + /ab/gi"), "/ab/gi");
        assert_eq!(run("typeof /x/"), "object");
    }

    #[test]
    fn json_parse() {
        // Scalars.
        assert_eq!(run("JSON.parse('42')"), "42");
        assert_eq!(run("JSON.parse('true')"), "true");
        assert_eq!(run("JSON.parse('null') === null"), "true");
        assert_eq!(run("JSON.parse('\"hi\\\\nthere\"')"), "hi\nthere");
        // Arrays and objects.
        assert_eq!(run("JSON.parse('[1, 2, 3]').length"), "3");
        assert_eq!(run("JSON.parse('[1, 2, 3]')[1]"), "2");
        assert_eq!(run("JSON.parse('{\"a\": 1, \"b\": 2}').b"), "2");
        // Nested.
        assert_eq!(
            run("JSON.parse('{\"items\": [{\"id\": 7}, {\"id\": 9}]}').items[1].id"),
            "9"
        );
        // Round-trip with stringify.
        assert_eq!(
            run("let o = JSON.parse('{\"x\": 10, \"y\": 20}'); JSON.stringify(o)"),
            "{\"x\":10,\"y\":20}"
        );
        // Negative / float numbers.
        assert_eq!(run("JSON.parse('-3.5')"), "-3.5");
        // Malformed input throws (caught).
        assert_eq!(
            run("try { JSON.parse('{bad}'); 'no'; } catch (e) { 'threw'; }"),
            "threw"
        );
    }

    #[test]
    fn dates() {
        // A fixed timestamp (2021-06-15T12:30:45.500Z = 1623760245500 ms).
        let ts = "1623760245500";
        assert_eq!(
            run(&alloc::format!("new Date({ts}).getTime()")),
            "1623760245500"
        );
        assert_eq!(run(&alloc::format!("new Date({ts}).getFullYear()")), "2021");
        assert_eq!(run(&alloc::format!("new Date({ts}).getMonth()")), "5"); // June, 0-based
        assert_eq!(run(&alloc::format!("new Date({ts}).getDate()")), "15");
        assert_eq!(run(&alloc::format!("new Date({ts}).getHours()")), "12");
        assert_eq!(run(&alloc::format!("new Date({ts}).getMinutes()")), "30");
        assert_eq!(run(&alloc::format!("new Date({ts}).getSeconds()")), "45");
        assert_eq!(run(&alloc::format!("new Date({ts}).getDay()")), "2"); // Tuesday
        assert_eq!(
            run(&alloc::format!("new Date({ts}).toISOString()")),
            "2021-06-15T12:30:45.500Z"
        );
        // The epoch.
        assert_eq!(run("new Date(0).toISOString()"), "1970-01-01T00:00:00.000Z");
        assert_eq!(run("new Date(0).getDay()"), "4"); // Thursday
        // typeof a date is object.
        assert_eq!(run("typeof new Date(0)"), "object");
    }

    #[test]
    fn eval_source_entry_point() {
        // Captured console output + completion value.
        let (out, completion) = eval_source("console.log('hi'); 1 + 2").expect("ok");
        assert_eq!(out, "hi\n");
        assert_eq!(completion, "3");
        // A program with no trailing expression yields `undefined`.
        let (_, c) = eval_source("let x = 5;").expect("ok");
        assert_eq!(c, "undefined");
        // A thrown error surfaces as an Err.
        assert!(eval_source("throw 'boom'").is_err());
        // A parse error surfaces as an Err.
        assert!(eval_source("let = ;").is_err());
    }

    #[test]
    fn object_is_and_safe_integer_and_computed_methods() {
        assert_eq!(run("Object.is(NaN, NaN)"), "true");
        assert_eq!(run("Object.is(0, -0)"), "false");
        assert_eq!(run("Object.is(-0, -0)"), "true");
        assert_eq!(
            run("Object.is('a', 'a') + ':' + Object.is(1, 2)"),
            "true:false"
        );
        assert_eq!(run("Number.isSafeInteger(9007199254740991)"), "true");
        assert_eq!(run("Number.isSafeInteger(9007199254740992)"), "false");
        assert_eq!(run("Number.isSafeInteger(1.5)"), "false");
        // Computed class method name.
        assert_eq!(
            run("let k='go'; class C { [k](){ return 42; } } new C().go()"),
            "42"
        );
    }

    #[test]
    fn object_spread_of_array_and_string() {
        assert_eq!(
            run("let o={...[10,20,30]}; o[0] + ':' + o[2] + ':' + Object.keys(o).join(',')"),
            "10:30:0,1,2"
        );
        assert_eq!(run("let o={...'ab'}; o[0] + o[1]"), "ab");
        // Mixed with explicit keys.
        assert_eq!(
            run("let o={...[1,2], a:9}; o[0] + ':' + o[1] + ':' + o.a"),
            "1:2:9"
        );
    }

    #[test]
    fn object_spread_invokes_getters() {
        assert_eq!(
            run(
                "let s={a:1, get b(){ return this.a + 1; }}; let c={...s, d:3}; c.a + ',' + c.b + ',' + c.d"
            ),
            "1,2,3"
        );
        // Later keys win; both sources merged.
        assert_eq!(
            run("JSON.stringify({...{x:1},...{y:2},x:9})"),
            "{\"x\":9,\"y\":2}"
        );
    }

    #[test]
    fn custom_symbol_iterator() {
        // for-of and spread drive a user `[Symbol.iterator]`.
        let src = "let o = { [Symbol.iterator]() { let i = 0; return { next() { return i < 3 ? { value: i++, done: false } : { value: undefined, done: true }; } }; } };";
        assert_eq!(
            run(&alloc::format!(
                "{src} let s=[]; for (let x of o) s.push(x); s.join(',')"
            )),
            "0,1,2"
        );
        assert_eq!(run(&alloc::format!("{src} [...o].join('-')")), "0-1-2");
    }

    #[test]
    fn computed_object_literal_keys() {
        // `{ [expr]: v }` evaluates and coerces the key.
        assert_eq!(run("let k = 'a' + 'b'; let o = { [k]: 7 }; o.ab"), "7");
        // A numeric computed key coerces to its string form.
        assert_eq!(run("let o = { [1 + 1]: 'two' }; o['2']"), "two");
    }

    #[test]
    fn private_class_fields() {
        // Private fields store and read through `this.#x`.
        assert_eq!(
            run(
                "class C { #n = 0; bump(){ this.#n++; return this.#n; } } let c = new C(); c.bump(); c.bump()"
            ),
            "2"
        );
        // ...and are non-enumerable.
        assert_eq!(
            run("class C { #s = 1; constructor(){ this.p = 2; } } Object.keys(new C()).join(',')"),
            "p"
        );
    }

    #[test]
    fn bigints() {
        assert_eq!(run("typeof 5n"), "bigint");
        assert_eq!(run("(2n + 3n).toString()"), "5");
        assert_eq!(run("100n * 100n === 10000n"), "true");
        assert_eq!(run("2n ** 16n === 65536n"), "true");
        assert_eq!(run("10n / 3n === 3n"), "true");
        assert_eq!(run("-7n === 0n - 7n"), "true");
        assert_eq!(run("BigInt(99) === 99n"), "true");
        assert_eq!(run("10n === 10"), "false");
        assert_eq!(run("10n == 10"), "true");
        assert_eq!(run("!!0n"), "false");
        // Mixing BigInt and Number in arithmetic throws.
        assert_eq!(
            run("let r='ok'; try { 1n + 1; } catch (e) { r = 'threw'; } r"),
            "threw"
        );
        // Arbitrary precision: results far beyond i128 are exact.
        assert_eq!(
            run("(2n ** 200n).toString()"),
            "1606938044258990275541962092341162602522202993782792835301376"
        );
        assert_eq!(
            run("let f=1n; for(let i=1n;i<=25n;i++) f*=i; f.toString()"),
            "15511210043330985984000000"
        );
        assert_eq!(
            run("((2n ** 128n) - 1n).toString()"),
            "340282366920938463463374607431768211455"
        );
        assert_eq!(run("(~5n).toString()"), "-6");
        // Two's-complement bitwise, including beyond i128.
        assert_eq!(
            run("(12n & 10n).toString() + ',' + (12n | 10n) + ',' + (12n ^ 10n)"),
            "8,14,6"
        );
        assert_eq!(run("(-1n & 255n).toString()"), "255");
        assert_eq!(run("(((2n ** 100n) | 1n) - (2n ** 100n)).toString()"), "1");
    }

    #[test]
    fn new_on_bound_function() {
        assert_eq!(
            run(
                "function P(x,y){this.x=x;this.y=y;} let B=P.bind(null); let p=new B(3,4); p.x + ':' + p.y"
            ),
            "3:4"
        );
        assert_eq!(
            run("function P(x,y){this.x=x;this.y=y;} let B=P.bind(null,10); new B(20).x"),
            "10"
        );
        assert_eq!(
            run("function P(x){this.x=x;} let B=P.bind(null); (new B(1)) instanceof P"),
            "true"
        );
        // Re-bound: bound args accumulate.
        assert_eq!(
            run(
                "function P(x,y){this.x=x;this.y=y;} let B=P.bind(null,5).bind(null,6); let p=new B(); p.x + ':' + p.y"
            ),
            "5:6"
        );
        // A class can be bound and constructed.
        assert_eq!(
            run("class C{constructor(v){this.v=v;}} new (C.bind(null))(42).v"),
            "42"
        );
        assert_eq!(
            run("class C{constructor(v){this.v=v;}} new (C.bind(null,7))().v"),
            "7"
        );
    }

    #[test]
    fn apply_arraylike_and_bound_name() {
        // apply accepts an array-like (length + indexed properties).
        assert_eq!(
            run("function f(){return arguments.length;} f.apply(null,{length:3,0:1,1:2,2:3})"),
            "3"
        );
        assert_eq!(
            run(
                "function s(){let t=0;for(let i=0;i<arguments.length;i++)t+=arguments[i];return t;} s.apply(null,{length:2,0:10,1:20})"
            ),
            "30"
        );
        // A bound function's name.
        assert_eq!(run("function foo(){} foo.bind(null).name"), "bound foo");
        assert_eq!(
            run("function foo(){} foo.bind(null).bind(null).name"),
            "bound bound foo"
        );
    }

    #[test]
    fn function_length_and_name() {
        assert_eq!(run("function f(a, b, c){} f.length"), "3");
        assert_eq!(run("function f(){} f.length"), "0");
        // length counts params before the first default/rest.
        assert_eq!(run("function f(a, b = 1, c){} f.length"), "1");
        assert_eq!(run("function f(a, ...r){} f.length"), "1");
        // name from a declaration and a named function expression.
        assert_eq!(run("function greet(){} greet.name"), "greet");
        assert_eq!(run("let g = function inner(){}; g.name"), "inner");
    }

    #[test]
    fn object_seal_and_extensibility() {
        // preventExtensions: no new props, existing still writable.
        assert_eq!(
            run(
                "let o={a:1}; Object.preventExtensions(o); o.b=2; o.a=9; String(o.b) + ':' + o.a + ':' + Object.isExtensible(o)"
            ),
            "undefined:9:false"
        );
        // seal: no new props, no delete, existing writable.
        assert_eq!(
            run(
                "let o={x:1}; Object.seal(o); o.y=2; o.x=5; delete o.x; o.x + ':' + String(o.y) + ':' + Object.isSealed(o)"
            ),
            "5:undefined:true"
        );
        // freeze implies sealed + non-extensible.
        assert_eq!(
            run("let o={a:1}; Object.freeze(o); Object.isSealed(o) + ':' + Object.isExtensible(o)"),
            "true:false"
        );
    }

    #[test]
    fn non_writable_and_join_nullish() {
        // defineProperty writable:false ignores writes; descriptor reports it.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,writable:false,enumerable:true}); o.x=9; o.x"
            ),
            "1"
        );
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); Object.getOwnPropertyDescriptor(o,'x').writable"
            ),
            "false"
        );
        // Non-enumerable stays out of Object.keys but readable.
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'h',{value:2,enumerable:false}); Object.keys(o).join(',') + ':' + o.h"
            ),
            "a:2"
        );
        // Array.join renders null/undefined as empty.
        assert_eq!(run("[1,null,2,undefined,3].join('-')"), "1--2--3");
    }

    #[test]
    fn map_group_by_and_well_formed() {
        assert_eq!(
            run(
                "let g=Map.groupBy([1,2,3,4,5], x=>x%2?'odd':'even'); (g instanceof Map) + ':' + g.get('odd').join(',') + ':' + g.size"
            ),
            "true:1,3,5:2"
        );
        // Object keys are preserved (not stringified, unlike Object.groupBy).
        assert_eq!(
            run("let k={}; let g=Map.groupBy([1,2], x=>k); g.get(k).join(',')"),
            "1,2"
        );
        assert_eq!(
            run("'abc'.isWellFormed() + ':' + '\u{1f600}'.toWellFormed()"),
            "true:\u{1f600}"
        );
    }

    #[test]
    fn get_own_property_symbols_and_reflect_ownkeys() {
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={a:1}; o[s]=2; let g=Object.getOwnPropertySymbols(o); g.length + ':' + (g[0]===s) + ':' + o[g[0]]"
            ),
            "1:true:2"
        );
        assert_eq!(run("Object.getOwnPropertySymbols({}).length"), "0");
        // Reflect.ownKeys: string keys then symbol keys.
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={a:1,b:2}; o[s]=3; let k=Reflect.ownKeys(o); k.length + ':' + k[0] + ':' + (k[2]===s)"
            ),
            "3:a:true"
        );
    }

    #[test]
    fn assign_and_spread_copy_symbol_keys() {
        assert_eq!(
            run(
                "let s=Symbol('k'); let src={a:1}; src[s]=9; let t=Object.assign({},src); t.a + ':' + t[s]"
            ),
            "1:9"
        );
        assert_eq!(
            run("let s=Symbol('k'); let src={a:1}; src[s]=9; let t={...src}; t[s]"),
            "9"
        );
        // Object.keys still excludes the symbol.
        assert_eq!(
            run("let s=Symbol('k'); let src={a:1}; src[s]=9; Object.keys({...src}).join(',')"),
            "a"
        );
    }

    #[test]
    fn object_group_by() {
        assert_eq!(
            run(
                "let g=Object.groupBy([1,2,3,4,5], x=>x%2?'odd':'even'); g.odd.join(',') + '|' + g.even.join(',')"
            ),
            "1,3,5|2,4"
        );
        assert_eq!(run("Object.groupBy(['a','ab','b'], s=>s[0]).a.length"), "2");
        assert_eq!(run("Object.keys(Object.groupBy([], x=>x)).length"), "0");
        // Works over any iterable + uses the index.
        assert_eq!(run("Object.groupBy('aab', c=>c).a.length"), "2");
    }

    #[test]
    fn integer_key_ordering_and_array_tostring() {
        // Integer keys come first (ascending), then string keys in insertion order.
        assert_eq!(
            run("let o={2:'a',1:'b',3:'c'}; Object.keys(o).join(',')"),
            "1,2,3"
        );
        assert_eq!(
            run("let o={z:1, 2:2, a:3, 1:4}; Object.keys(o).join(',')"),
            "1,2,z,a"
        );
        assert_eq!(
            run("let o={}; o.b=1; o.a=2; Object.keys(o).join(',')"),
            "b,a"
        );
        assert_eq!(
            run("Object.values({10:'x',2:'y',1:'z'}).join(',')"),
            "z,y,x"
        );
        // Array toString joins with comma.
        assert_eq!(run("['a','b','c'].toString()"), "a,b,c");
        assert_eq!(run("[1,[2,3],4].toString()"), "1,2,3,4");
    }

    #[test]
    fn inherited_setter_is_called() {
        // Assigning to a property with an *inherited* setter calls it (rather
        // than shadowing it with an own data property).
        assert_eq!(
            run(
                "let base={_d:0, get c(){return this._d;}, set c(v){this._d=v;}}; let d=Object.create(base); d.c=10; d._d + ':' + base._d"
            ),
            "10:0"
        );
        // A getter-only inherited accessor shadows the data assignment.
        assert_eq!(
            run("let base={get x(){return 1;}}; let d=Object.create(base); d.x=99; d.x"),
            "1"
        );
        // An own data property still assigns normally.
        assert_eq!(run("let o={a:1}; o.a=2; o.a"), "2");
    }

    #[test]
    fn defineproperty_invariants() {
        // Redefining a non-configurable property throws; value is retained.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); try{ Object.defineProperty(o,'x',{value:2}); 'no' }catch(e){ (e instanceof TypeError)+':'+o.x }"
            ),
            "true:1"
        );
        // A configurable property can be redefined (attributes reset).
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:true}); Object.defineProperty(o,'x',{value:2,configurable:true}); o.x"
            ),
            "2"
        );
        // Defining a new property on a non-extensible object throws.
        assert_eq!(
            run(
                "let o={}; Object.preventExtensions(o); try{ Object.defineProperty(o,'z',{value:1}); 'no' }catch(e){ e instanceof TypeError }"
            ),
            "true"
        );
        // Non-configurable but writable: value may still change.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'w',{value:1,writable:true,configurable:false}); Object.defineProperty(o,'w',{value:2,writable:true,configurable:false}); o.w"
            ),
            "2"
        );
    }

    #[test]
    fn descriptor_reports_configurable() {
        // defineProperty defaults to non-configurable.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1}); Object.getOwnPropertyDescriptor(o,'x').configurable"
            ),
            "false"
        );
        // Explicit configurable: true is reported.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:true}); Object.getOwnPropertyDescriptor(o,'x').configurable"
            ),
            "true"
        );
        // A plain literal property is configurable; a frozen one is not.
        assert_eq!(
            run("Object.getOwnPropertyDescriptor({a:1},'a').configurable"),
            "true"
        );
        assert_eq!(
            run("Object.getOwnPropertyDescriptor(Object.freeze({a:1}),'a').configurable"),
            "false"
        );
    }

    #[test]
    fn delete_respects_configurable() {
        assert_eq!(run("let o={a:1}; delete o.a"), "true");
        assert_eq!(run("let o={}; delete o.missing"), "true");
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); delete o.x"),
            "false"
        );
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); delete o.x; o.x"
            ),
            "1"
        );
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'y',{value:2,configurable:true}); delete o.y"),
            "true"
        );
        assert_eq!(run("let o=Object.freeze({a:1}); delete o.a"), "false");
    }

    #[test]
    fn redefine_accessor_as_data() {
        // Accessor → accessor.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{get(){return 1;},configurable:true}); Object.defineProperty(o,'x',{get(){return 2;},configurable:true}); o.x"
            ),
            "2"
        );
        // Accessor → data property (the old getter must no longer apply).
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{get(){return 1;},configurable:true}); Object.defineProperty(o,'x',{value:42,configurable:true}); o.x"
            ),
            "42"
        );
    }

    #[test]
    fn enumerable_accessor_keys() {
        // Object-literal getters are enumerable (appear in Object.keys/JSON).
        assert_eq!(
            run("Object.keys({x:1, get y(){return 2;}}).join(',')"),
            "x,y"
        );
        assert_eq!(
            run("JSON.stringify({a:1, get b(){return 2;}})"),
            "{\"a\":1,\"b\":2}"
        );
        // defineProperty accessor: enumerable per the descriptor.
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'c',{get(){return 9;},enumerable:true}); Object.keys(o).join(',')"
            ),
            "a,c"
        );
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'c',{get(){return 9;}}); Object.keys(o).join(',')"
            ),
            "a"
        );
        // Class accessors are non-enumerable.
        assert_eq!(
            run(
                "class C{ constructor(){this.a=1;} get b(){return 2;} } Object.keys(new C()).join(',')"
            ),
            "a"
        );
    }

    #[test]
    fn get_own_property_descriptors() {
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'b',{value:2,writable:false,enumerable:true}); let d=Object.getOwnPropertyDescriptors(o); d.a.value + ',' + d.a.writable + ',' + d.b.value + ',' + d.b.writable"
            ),
            "1,true,2,false"
        );
        assert_eq!(
            run("Object.keys(Object.getOwnPropertyDescriptors({a:1,b:2})).join(',')"),
            "a,b"
        );
        assert_eq!(
            run(
                "let o={get x(){return 5;}}; let d=Object.getOwnPropertyDescriptors(o); typeof d.x.get"
            ),
            "function"
        );
    }

    #[test]
    fn object_define_properties() {
        assert_eq!(
            run(
                "let o={}; Object.defineProperties(o, { x:{value:1}, y:{get:function(){return 2;}} }); o.x + ',' + o.y"
            ),
            "1,2"
        );
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'a',{value:42}); o.a"),
            "42"
        );
    }

    #[test]
    fn computed_key_destructuring() {
        // Declaration form.
        assert_eq!(
            run("let k='name'; let {[k]: v} = {name:'Alice'}; v"),
            "Alice"
        );
        assert_eq!(
            run("let p='x'; let {[p]: a, ...rest} = {x:1, y:2}; a + ':' + rest.y"),
            "1:2"
        );
        // Assignment form.
        assert_eq!(
            run("let k='m'; let v; ({[k]: v} = {m:42}); String(v)"),
            "42"
        );
    }

    #[test]
    fn destructuring_assignment_with_defaults() {
        assert_eq!(run("let a,b; [a,b]=[1,2]; a+','+b"), "1,2");
        assert_eq!(run("let a,b; [a,b]=[1,2]; [a,b]=[b,a]; a+','+b"), "2,1");
        assert_eq!(run("let a,b; ({x:a,y:b}={x:10,y:20}); a+','+b"), "10,20");
        // Default in an assignment pattern.
        assert_eq!(run("let a,b,c; [a,b,c=99]=[1,2]; String(c)"), "99");
        assert_eq!(run("let x; ({p:x=7}={}); String(x)"), "7");
    }

    #[test]
    fn date_multi_arg_and_subtraction() {
        assert_eq!(
            run("let d=new Date(2026,5,5); d.getFullYear()+'/'+d.getMonth()+'/'+d.getDate()"),
            "2026/5/5"
        );
        assert_eq!(run("let d=new Date(0); d.getTime()"), "0");
        assert_eq!(run("(new Date(2000)) - (new Date(1000))"), "1000");
    }

    #[test]
    fn utf16_string_indexing() {
        assert_eq!(run("'café'.length"), "4");
        assert_eq!(run("'\\u{1F600}'.length"), "2");
        assert_eq!(run("'a\\u{1F600}b'.length"), "4");
        assert_eq!(run("'\\u{1F600}'.charCodeAt(0)"), "55357");
        assert_eq!(run("'\\u{1F600}'.charCodeAt(1)"), "56832");
        assert_eq!(run("'\\u{1F600}'.codePointAt(0)"), "128512");
        assert_eq!(run("'a\\u{1F600}b'.codePointAt(1)"), "128512");
        assert_eq!(run("'hello'.charCodeAt(0)"), "104");
    }

    #[test]
    fn array_call_and_unary_plus_array() {
        // Array(...) without new.
        assert_eq!(run("Array(3).length"), "3");
        assert_eq!(run("Array(1,2,3).join(',')"), "1,2,3");
        assert_eq!(run("Array().length"), "0");
        // Unary + coerces arrays via their string form.
        assert_eq!(run("+[]"), "0");
        assert_eq!(run("+[5]"), "5");
        assert_eq!(run("Number.isNaN(+[1,2])"), "true");
        // Symbol.toPrimitive still gets the number hint for unary +.
        assert_eq!(
            run("+{[Symbol.toPrimitive](h){ return h==='number'?9:0; }}"),
            "9"
        );
    }

    #[test]
    fn reverse_inplace_new_array_string_index() {
        // reverse mutates in place and returns the same array.
        assert_eq!(
            run("let a=[1,2,3]; let b=a.reverse(); (a===b) + ':' + a.join(',')"),
            "true:3,2,1"
        );
        // new Array(n) and new Array(...elements).
        assert_eq!(run("new Array(3).fill(7).join(',')"), "7,7,7");
        assert_eq!(run("new Array(1,2,3).join(',')"), "1,2,3");
        assert_eq!(run("new Array(3).length"), "3");
        // String index access.
        assert_eq!(run("'hello'[0] + 'hello'[4]"), "ho");
        assert_eq!(run("String('abc'[5])"), "undefined");
    }

    #[test]
    fn array_immutable_methods() {
        assert_eq!(
            run("let a=[3,1,2]; a.toSorted().join(',') + '|' + a.join(',')"),
            "1,2,3|3,1,2"
        );
        assert_eq!(run("[1,2,3].toReversed().join(',')"), "3,2,1");
        assert_eq!(run("[1,2,3].with(1,9).join(',')"), "1,9,3");
        assert_eq!(run("[1,2,3].with(-1,9).join(',')"), "1,2,9");
        assert_eq!(
            run("[1,2,3,4,5].toSpliced(1,2,'a','b').join(',')"),
            "1,a,b,4,5"
        );
        assert_eq!(run("[1,2,3,4].toSpliced(2).join(',')"), "1,2");
        // with out-of-range → RangeError.
        assert_eq!(
            run("try { [1,2,3].with(10,0); 'no' } catch(e){ e instanceof RangeError }"),
            "true"
        );
    }

    #[test]
    fn reduce_args_and_sort_in_place() {
        // reduce callback gets (acc, cur, index, array).
        assert_eq!(
            run(
                "let ix=[]; [10,20,30].reduce(function(a,c,i,arr){ ix.push(i + ':' + arr.length); return a+c; }, 0); ix.join(',')"
            ),
            "0:3,1:3,2:3"
        );
        assert_eq!(
            run("['a','b','c'].reduceRight(function(a,c){return a+c;})"),
            "cba"
        );
        // sort is in place and returns the same array.
        assert_eq!(
            run("let a=[3,1,2]; let b=a.sort(); (a===b) + ':' + a.join(',')"),
            "true:1,2,3"
        );
        assert_eq!(run("[3,1,2].sort((x,y)=>y-x).join(',')"), "3,2,1");
    }

    #[cfg(feature = "intl")]
    #[test]
    fn string_normalize_forms() {
        // "é" composed (1 cp) vs decomposed (e + U+0301).
        assert_eq!(run("'\u{e9}'.normalize('NFD').length"), "2");
        assert_eq!(run("'e\u{301}'.normalize('NFC').length"), "1");
        assert_eq!(
            run("'\u{e9}'.normalize() === 'e\u{301}'.normalize()"),
            "true"
        );
        // NFKC expands the ﬁ ligature.
        assert_eq!(run("'\u{fb01}'.normalize('NFKC')"), "fi");
        assert_eq!(run("'abc'.normalize()"), "abc");
        // An unsupported form throws a RangeError *object* (not a bare string).
        assert_eq!(
            run("try{'x'.normalize('BAD');'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        assert_eq!(
            run("try{'x'.normalize('BAD');'no'}catch(e){e.name}"),
            "RangeError"
        );
    }

    #[test]
    fn string_raw_and_member_tag() {
        assert_eq!(run("String.raw`a\\nb`"), "a\\nb");
        assert_eq!(run("String.raw`${1}+${2}=${3}`"), "1+2=3");
        assert_eq!(run("String.raw`line\\tend`.length"), "9"); // backslash + t kept raw
        // A tag read as a member of a plain object also dispatches.
        assert_eq!(
            run("let o={ t(s){ return s.raw[0]; } }; o.t`x\\ny`"),
            "x\\ny"
        );
    }

    #[test]
    fn generator_return_value() {
        // The return value is surfaced once, with done:true, after the yields.
        assert_eq!(
            run(
                "function* g(){ yield 1; yield 2; return 99; } let it=g(); it.next(); it.next(); let r=it.next(); r.value + ':' + r.done"
            ),
            "99:true"
        );
        // Subsequent next() calls yield undefined/done.
        assert_eq!(
            run(
                "function* g(){ yield 1; return 7; } let it=g(); it.next(); it.next(); String(it.next().value) + ':' + it.next().done"
            ),
            "undefined:true"
        );
        // Spread excludes the return value.
        assert_eq!(
            run("function* g(){ yield 1; yield 2; return 9; } [...g()].join(',')"),
            "1,2"
        );
    }

    #[test]
    fn array_iterators() {
        assert_eq!(run("[...['a','b','c'].keys()].join(',')"), "0,1,2");
        assert_eq!(run("[...['a','b','c'].values()].join(',')"), "a,b,c");
        assert_eq!(
            run("let o=[]; for (let [i,v] of ['x','y'].entries()) o.push(i+':'+v); o.join(',')"),
            "0:x,1:y"
        );
        // The iterator supports next().
        assert_eq!(
            run("let it=['p','q'].values(); it.next().value + it.next().value"),
            "pq"
        );
    }

    #[test]
    fn matchall_replaceall_require_global() {
        assert_eq!(
            run("try{ 'aaa'.replaceAll(/a/,'b'); 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(
            run("try{ [...'aaa'.matchAll(/a/)]; 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(run("'aaa'.replaceAll(/a/g,'b')"), "bbb");
        assert_eq!(run("[...'a1b2'.matchAll(/[a-z]\\d/g)].length"), "2");
        assert_eq!(
            run("'2024-06'.replace(/(?<y>\\d+)-(?<m>\\d+)/, '$<m>/$<y>')"),
            "06/2024"
        );
    }

    #[test]
    fn replace_groups_and_split_limit() {
        // The replace function receives a `groups` object for named captures.
        assert_eq!(
            run("'2024-06'.replace(/(?<y>\\d+)-(?<m>\\d+)/, (m,y,mo,o,s,g)=>g.y+'/'+g.m)"),
            "2024/06"
        );
        // Regex split honors the limit.
        assert_eq!(run("'a1b2c3'.split(/(\\d)/,3).join('|')"), "a|1|b");
        // Empty-regex split has no trailing empty; capture split keeps its trailing.
        assert_eq!(run("'abc'.split(/(?:)/).join(',')"), "a,b,c");
        assert_eq!(run("'a1'.split(/(\\d)/).length"), "3");
    }

    #[test]
    fn regex_unicode_property_categories() {
        // Robust across the intl / no-intl matchers. Property escapes require the
        // `u` flag (without it `\p` is the literal `p`, per Annex B).
        assert_eq!(run(r#"'Hello World'.match(/\p{Lu}/gu).join('')"#), "HW");
        assert_eq!(run(r#"'Hello'.match(/\p{Ll}/gu).join('')"#), "ello");
        assert_eq!(run(r#"'abc123'.match(/\p{N}/gu).join('')"#), "123");
        assert_eq!(run(r#"'a.b!c'.match(/\p{P}/gu).join('')"#), ".!");
        assert_eq!(run(r#"'中文字'.match(/\p{Lo}/gu).length"#), "3");
        assert_eq!(run(r#"'a1b2'.match(/\P{N}/gu).join('')"#), "ab");
        // The full subcategory set compiles (matching may need Unicode tables).
        assert_eq!(
            run(r#"'x'.match(/\p{Sm}|\p{Sc}|\p{Mn}|\p{Pd}/gu)===null"#),
            "true"
        );
    }

    #[cfg(feature = "intl")]
    #[test]
    fn regex_unicode_property_precise_with_intl() {
        assert_eq!(run(r#"'3+5'.match(/\p{Sm}/u)[0]"#), "+");
        assert_eq!(run(r#"'$5'.match(/\p{Sc}/u)[0]"#), "$");
        assert_eq!(run(r#"'(a)'.match(/\p{Ps}/u)[0]"#), "(");
        assert_eq!(run(r#"'a-b'.match(/\p{Pd}/u)[0]"#), "-");
    }

    #[test]
    fn regex_on_multibyte_strings() {
        // These previously panicked (char-index spans used as byte ranges).
        assert_eq!(run("'café'.match(/é/)[0]"), "é");
        assert_eq!(run("'café'.match(/(.+)/)[1]"), "café");
        assert_eq!(run("'café'.replace(/é/, 'e')"), "cafe");
        assert_eq!(run("'a→b→c'.split(/→/).join('|')"), "a|b|c");
        assert_eq!(run("'über 123'.match(/\\d+/)[0]"), "123");
        assert_eq!(run("'café'.match(/(?<r>.+)/).groups.r"), "café");
        assert_eq!(run("[...'café déjà'.matchAll(/é/g)].length"), "2");
        // Regex-template `$&`/`` $` ``/`$'` over a multibyte subject previously
        // byte-indexed char offsets and panicked; the template now slices chars
        // (RE-7 refactor).
        assert_eq!(run("'café'.replace(/f/, '[$&]')"), "ca[f]é");
        assert_eq!(run("'café'.replace(/f/, '$`')"), "cacaé");
        assert_eq!(run("'café'.replace(/f/, \"$'\")"), "caéé");
        assert_eq!(run("'aéb'.replace(/(é)/, '<$1>')"), "a<é>b");
    }

    #[test]
    fn regex_empty_and_zerowidth_matches() {
        // Empty-match global replace keeps the characters.
        assert_eq!(run("'abc'.replace(/x*/g, '-')"), "-a-b-c-");
        // Zero-width (lookahead) split keeps the boundary character.
        assert_eq!(
            run("'camelCaseWord'.split(/(?=[A-Z])/).join('|')"),
            "camel|Case|Word"
        );
        // Capture-group split still splices the captures.
        assert_eq!(run("'a1b2'.split(/(\\d)/).join(',')"), "a,1,b,2,");
        // Lookahead-based number grouping replace.
        assert_eq!(
            run("'1234567'.replace(/(?<=\\d)(?=(?:\\d{3})+$)/g, ',')"),
            "1,234,567"
        );
    }

    #[test]
    fn replace_dollar_patterns() {
        // String-pattern replace.
        assert_eq!(run("'hello'.replace('l', '[$&]')"), "he[l]lo");
        assert_eq!(run("'abc'.replace('b', '$`')"), "aac"); // prefix
        assert_eq!(run("'abc'.replace('b', \"$'\")"), "acc"); // suffix
        assert_eq!(run("'test'.replaceAll('t', '$$')"), "$es$"); // literal $
        // Regex-pattern replace.
        assert_eq!(
            run("'2024-06'.replace(/(\\d+)-(\\d+)/, '$2/$1')"),
            "06/2024"
        );
        assert_eq!(run("'x'.replace(/x/, '$1')"), "$1"); // no group 1 → literal
        assert_eq!(run("'abc'.replace(/b/, '$`')"), "aac");
    }

    #[test]
    fn regex_stateful_last_index() {
        // Global exec advances lastIndex and resets on a miss.
        assert_eq!(
            run(
                "let r=/\\d/g; r.exec('a1b2')[0] + ':' + r.lastIndex + ':' + r.exec('a1b2')[0] + ':' + r.lastIndex"
            ),
            "1:2:2:4"
        );
        assert_eq!(
            run("let r=/\\d/g; r.exec('a1'); String(r.exec('a1')) + ':' + r.lastIndex"),
            "null:0"
        );
        // Writing lastIndex resumes from there.
        assert_eq!(run("let r=/\\d/g; r.lastIndex=3; r.exec('12345')[0]"), "4");
        // test() advances; non-global never does.
        assert_eq!(run("let r=/x/g; r.test('axbx'); r.lastIndex"), "2");
        assert_eq!(run("let r=/\\d/; r.exec('a1'); r.lastIndex"), "0");
    }

    #[test]
    fn regex_lookbehind() {
        assert_eq!(run("/(?<=\\$)\\d+/.test('$100')"), "true");
        assert_eq!(run("/(?<=\\$)\\d+/.test('100')"), "false");
        assert_eq!(run("'$100'.match(/(?<=\\$)\\d+/)[0]"), "100");
        assert_eq!(
            run("'price: $50'.replace(/(?<=\\$)\\d+/, 'X')"),
            "price: $X"
        );
        assert_eq!(run("/(?<!a)b/.test('ab')"), "false");
        assert_eq!(run("/(?<!a)b/.test('xb')"), "true");
    }

    #[test]
    fn regex_lookahead_and_backref() {
        // Positive / negative lookahead (zero-width).
        assert_eq!(run("/foo(?=bar)/.test('foobar')"), "true");
        assert_eq!(run("/foo(?=bar)/.test('foobaz')"), "false");
        assert_eq!(run("/foo(?!bar)/.test('foobaz')"), "true");
        assert_eq!(run("'foobar'.replace(/foo(?=bar)/, 'X')"), "Xbar");
        // Backreferences.
        assert_eq!(run("/(\\w)\\1/.test('hello')"), "true");
        assert_eq!(run("/(\\w)\\1/.test('abc')"), "false");
        assert_eq!(run("'hello'.match(/(.)\\1/)[0]"), "ll");
    }

    #[test]
    fn regex_named_groups() {
        assert_eq!(
            run(
                "let m='2024-06'.match(/(?<y>\\d{4})-(?<mo>\\d{2})/); m.groups.y + ':' + m.groups.mo"
            ),
            "2024:06"
        );
        // Still positionally indexable.
        assert_eq!(
            run("'2024-06'.match(/(?<y>\\d{4})-(?<mo>\\d{2})/)[1]"),
            "2024"
        );
        // Named backreference in replacement.
        assert_eq!(
            run("'John Smith'.replace(/(?<first>\\w+) (?<last>\\w+)/, '$<last>, $<first>')"),
            "Smith, John"
        );
        // No named groups → .groups is undefined.
        assert_eq!(run("String('ab'.match(/(a)(b)/).groups)"), "undefined");
    }

    #[test]
    fn match_all_named_groups() {
        assert_eq!(
            run(
                "let m=[...'2024-06 2025-12'.matchAll(/(?<y>\\d{4})-(?<mo>\\d{2})/g)]; m[0].groups.y + ':' + m[1].groups.mo"
            ),
            "2024:12"
        );
        // Positional access + index still work on matchAll results.
        assert_eq!(run("[...'a1b2'.matchAll(/([a-z])(\\d)/g)][1][2]"), "2");
        assert_eq!(
            run("[...'xy'.matchAll(/(?<c>.)/g)].map(m=>m.groups.c).join('')"),
            "xy"
        );
    }

    #[test]
    fn string_match_all() {
        assert_eq!(run("[...'a1b2c3'.matchAll(/([a-z])(\\d)/g)].length"), "3");
        assert_eq!(
            run(
                "let m=[...'a1b2'.matchAll(/([a-z])(\\d)/g)]; m[0][0] + ':' + m[0][1] + ':' + m[0][2]"
            ),
            "a1:a:1"
        );
        assert_eq!(
            run("[...'hello world'.matchAll(/\\w+/g)].map(m=>m[0]).join(',')"),
            "hello,world"
        );
        assert_eq!(run("[...'abc'.matchAll(/\\d/g)].length"), "0");
    }

    #[test]
    fn array_thisarg_and_split_captures() {
        assert_eq!(
            run("[1,2,3].map(function(x){return x*this.m;},{m:3}).join(',')"),
            "3,6,9"
        );
        assert_eq!(
            run("[1,2,3,4].filter(function(x){return x>this.t;},{t:2}).join(',')"),
            "3,4"
        );
        assert_eq!(
            run("[1,2,3].some(function(x){return x===this.g;},{g:2})"),
            "true"
        );
        assert_eq!(
            run("[1,2,3].every(function(x){return x<=this.mx;},{mx:3})"),
            "true"
        );
        // split with a capturing separator splices the captures in.
        assert_eq!(run("'a1b2c3'.split(/(\\d)/).join('|')"), "a|1|b|2|c|3|");
    }

    #[test]
    fn array_last_index_of_from_index() {
        assert_eq!(run("[10,20,30,20,10].lastIndexOf(20)"), "3");
        assert_eq!(run("[10,20,30,20,10].lastIndexOf(10,3)"), "0");
        assert_eq!(run("[10,20,30,20,10].lastIndexOf(20,-3)"), "1");
        assert_eq!(run("[1,2,3].lastIndexOf(9)"), "-1");
        assert_eq!(run("[1,2,3,4].findLastIndex(x => x < 3)"), "1");
    }

    #[test]
    fn frozen_object_blocks_delete() {
        assert_eq!(
            run("let o={a:1,b:2}; Object.freeze(o); delete o.b; o.b"),
            "2"
        );
        assert_eq!(
            run("let o={a:1}; Object.freeze(o); o.a=9; o.c=3; o.a + ':' + String(o.c)"),
            "1:undefined"
        );
        assert_eq!(run("let o={a:1,b:2}; delete o.b; String(o.b)"), "undefined");
    }

    #[test]
    fn array_and_function_named_properties() {
        assert_eq!(
            run("let a=[1,2,3]; a.tag='x'; a.tag + ':' + a.length + ':' + a[0]"),
            "x:3:1"
        );
        assert_eq!(run("let a=[1]; a.tag='y'; a.hasOwnProperty('tag')"), "true");
        assert_eq!(run("function f(){} f.meta=42; f.meta"), "42");
        // Tagged template strings carry `.raw`.
        assert_eq!(run("function t(s){ return s.raw[0]; } t`a\\tb`"), "a\\tb");
    }

    #[test]
    fn error_to_string() {
        assert_eq!(run("new Error('boom').toString()"), "Error: boom");
        assert_eq!(run("new TypeError('bad').toString()"), "TypeError: bad");
        assert_eq!(run("new Error().toString()"), "Error");
        // A user toString still wins.
        assert_eq!(
            run("({ name:'X', message:'y', toString(){ return 'custom'; } }).toString()"),
            "custom"
        );
    }

    #[test]
    fn symbol_to_primitive_hints() {
        let o =
            "let o={[Symbol.toPrimitive](h){ return h==='number'?42:h==='string'?'str':'def'; }};";
        assert_eq!(run(&alloc::format!("{o} +o")), "42");
        assert_eq!(run(&alloc::format!("{o} `${{o}}`")), "str");
        assert_eq!(run(&alloc::format!("{o} o + ''")), "def");
        // Symbol.toPrimitive takes precedence over valueOf/toString.
        assert_eq!(
            run("let o={[Symbol.toPrimitive](){ return 9; }, valueOf(){ return 1; }}; o + 0"),
            "9"
        );
    }

    #[test]
    fn loose_equality_object_coercion() {
        assert_eq!(run("[] == 0"), "true");
        assert_eq!(run("[1] == 1"), "true");
        assert_eq!(run("[1,2] == '1,2'"), "true");
        assert_eq!(run("({}) == ({})"), "false"); // distinct objects
        assert_eq!(run("let o={valueOf(){return 5;}}; o == 5"), "true");
        assert_eq!(run("'' == 0"), "true");
        assert_eq!(run("null == 0"), "false");
    }

    #[test]
    fn to_primitive_in_operators() {
        assert_eq!(run("let o={valueOf(){return 42;}}; o + 8"), "50");
        assert_eq!(run("let o={valueOf(){return 6;}}; o * 7"), "42");
        assert_eq!(run("let o={toString(){return 'x';}}; '' + o"), "x");
        // valueOf is preferred for the default/number hint.
        assert_eq!(
            run("let o={valueOf(){return 5;}, toString(){return 'five';}}; o + 1"),
            "6"
        );
        // Identity comparison does not coerce.
        assert_eq!(run("let o={valueOf(){return 1;}}; o === o"), "true");
    }

    #[test]
    fn template_invokes_custom_tostring() {
        assert_eq!(
            run("let o = { toString() { return 'custom'; } }; `val=${o}`"),
            "val=custom"
        );
        // A plain object with no toString still renders the default form.
        assert_eq!(run("`${ {a:1} }`"), "[object Object]");
        // Arrays/numbers/booleans coerce as usual.
        assert_eq!(run("`${[1,2,3]}-${true}-${null}`"), "1,2,3-true-null");
    }

    #[test]
    fn coercion_string_number_join_freeze_tofixed() {
        // String()/Number() honor toString/valueOf; join too.
        assert_eq!(run("String({toString(){return 'x';}})"), "x");
        assert_eq!(run("Number({valueOf(){return 42;}})"), "42");
        assert_eq!(
            run("[{toString(){return 'a';}},{toString(){return 'b';}}].join(',')"),
            "a,b"
        );
        // Frozen array rejects push.
        assert_eq!(
            run("let a=[1,2,3]; Object.freeze(a); a.push(4); a.length + ':' + Object.isFrozen(a)"),
            "3:true"
        );
        // toFixed rounds half away from zero.
        assert_eq!(run("(0.5).toFixed(0)"), "1");
        assert_eq!(run("(2.5).toFixed(0)"), "3");
        assert_eq!(run("(123.456).toFixed(2)"), "123.46");
    }

    #[test]
    fn math_abs_round_and_create_descriptors() {
        // Math.round rounds half toward +Infinity (not away from zero).
        assert_eq!(run("Math.round(-2.5)"), "-2");
        assert_eq!(run("Math.round(2.5)"), "3");
        assert_eq!(run("Math.round(-0.5) === 0"), "true");
        // Math.abs(-0) is +0.
        assert_eq!(run("Object.is(Math.abs(-0), 0)"), "true");
        // Object.create with a descriptors map.
        assert_eq!(
            run(
                "let p={g(){return 'hi';}}; let o=Object.create(p, {n:{value:5,enumerable:true}}); o.n + ':' + o.g() + ':' + Object.keys(o).join(',')"
            ),
            "5:hi:n"
        );
    }

    #[test]
    fn negative_zero_stringifies_to_zero() {
        assert_eq!(run("String(-0)"), "0");
        assert_eq!(run("(-0).toString()"), "0");
        assert_eq!(run("'' + -0"), "0");
        assert_eq!(run("`${-0}`"), "0");
        assert_eq!(run("[-0, 0].join(',')"), "0,0");
        // But Object.is still distinguishes the bit pattern.
        assert_eq!(run("Object.is(-0, 0)"), "false");
    }

    #[test]
    fn math_minus_zero_indexof_fromindex_number_exponential() {
        // Math.max/min treat +0 > -0.
        assert_eq!(run("Object.is(Math.max(-0, 0), 0)"), "true");
        assert_eq!(run("Object.is(Math.min(0, -0), -0)"), "true");
        // String.indexOf honors fromIndex.
        assert_eq!(run("'hello world'.indexOf('o', 5)"), "7");
        assert_eq!(run("'hello world'.indexOf('o')"), "4");
        // Number.toString exponential thresholds.
        assert_eq!(run("(1e21).toString()"), "1e+21");
        assert_eq!(run("(1e-7).toString()"), "1e-7");
        assert_eq!(run("(1e20).toString()"), "100000000000000000000"); // not exponential
        assert_eq!(run("(0.000001).toString()"), "0.000001"); // 1e-6 stays decimal
    }

    #[test]
    fn math_trig_and_extra() {
        assert_eq!(
            run("Math.sin(0) + ':' + Math.cos(0) + ':' + Math.tan(0)"),
            "0:1:0"
        );
        assert_eq!(run("Math.round(Math.sin(Math.PI/2))"), "1");
        assert_eq!(run("Math.round(Math.atan2(1,1)*4/Math.PI)"), "1");
        assert_eq!(
            run("Math.cosh(0) + ':' + Math.tanh(0) + ':' + Math.expm1(0)"),
            "1:0:0"
        );
        assert_eq!(
            run("Math.fround(1.5) + ':' + (Math.fround(1.1) !== 1.1)"),
            "1.5:true"
        );
        assert_eq!(run("Math.clz32(1) + ':' + Math.clz32(0)"), "31:32");
        assert_eq!(run("Math.imul(3,4) + ':' + Math.imul(-1,8)"), "12:-8");
    }

    #[test]
    fn number_formatting() {
        assert_eq!(run("(3.5).toString(2)"), "11.1");
        assert_eq!(run("(255.5).toString(16)"), "ff.8");
        assert_eq!(run("(-255.5).toString(16)"), "-ff.8");
        assert_eq!(run("(12345).toPrecision(1)"), "1e+4");
        assert_eq!(run("(0.0000001234).toPrecision(2)"), "1.2e-7");
        assert_eq!(run("(1234567).toLocaleString()"), "1,234,567");
        assert_eq!(run("(-1234.5).toLocaleString()"), "-1,234.5");
    }

    #[test]
    fn math_random_in_range() {
        // In [0, 1), and consecutive calls differ (the PRNG advances).
        assert_eq!(run("let a=Math.random(); a >= 0 && a < 1"), "true");
        assert_eq!(run("Math.random() !== Math.random()"), "true");
        assert_eq!(
            run(
                "let xs=[]; for (let i=0;i<100;i++) xs.push(Math.random()); xs.every(x=>x>=0&&x<1)"
            ),
            "true"
        );
    }

    #[test]
    fn math_constants() {
        assert_eq!(run("Math.PI > 3.14 && Math.PI < 3.15"), "true");
        assert_eq!(run("Math.E > 2.71 && Math.E < 2.72"), "true");
        assert_eq!(run("Math.SQRT2 * Math.SQRT2 > 1.999"), "true");
        assert_eq!(run("Math.floor(Math.LN2 * 1000)"), "693");
    }

    #[test]
    fn private_in_brand_check() {
        assert_eq!(
            run(
                "class H{ #s=1; static check(o){ return #s in o; } } H.check(new H()) + ':' + H.check({})"
            ),
            "true:false"
        );
        // Works for an inherited brand too (subclass instances have the field).
        assert_eq!(
            run(
                "class H{ #s=1; static check(o){ return #s in o; } } class D extends H{} H.check(new D())"
            ),
            "true"
        );
    }

    #[test]
    fn class_static_blocks() {
        assert_eq!(
            run("class C{ static x=1; static { C.y = C.x + 1; } } C.y"),
            "2"
        );
        // Multiple blocks run in order.
        assert_eq!(
            run("class C{ static n=0; static { C.n=10; } static { C.n+=5; } } C.n"),
            "15"
        );
        // `this` is the class inside a static block.
        assert_eq!(
            run("class C{ static x=1; static { this.y = this.x + 100; } } C.y"),
            "101"
        );
    }

    #[test]
    fn static_setters_and_symbol_description() {
        // Static setter then getter.
        assert_eq!(
            run(
                "class T{ static _c=0; static get c(){return T._c;} static set c(v){T._c=v;} } T.c=25; T.c"
            ),
            "25"
        );
        // Symbol description: undefined for no-arg, the string otherwise.
        assert_eq!(run("String(Symbol().description)"), "undefined");
        assert_eq!(run("Symbol('d').description"), "d");
        assert_eq!(run("Symbol('').description"), ""); // explicit empty
    }

    #[test]
    fn object_hasown_static_accessors_replaceall_fn() {
        // Object.hasOwn.
        assert_eq!(
            run("Object.hasOwn({a:1},'a') + ':' + Object.hasOwn({a:1},'b')"),
            "true:false"
        );
        assert_eq!(run("Object.hasOwn(Object.create({x:1}),'x')"), "false");
        // Static field write-back and static getter.
        assert_eq!(
            run(
                "class C{ static n=0; static inc(){ return ++C.n; } static get cur(){ return C.n; } } C.inc(); C.inc(); C.cur"
            ),
            "2"
        );
        // replaceAll with a function replacer.
        assert_eq!(
            run("'AAA'.replaceAll('A', function(){ return 'B'; })"),
            "BBB"
        );
        assert_eq!(
            run("'a1b2'.replace('1', function(m){ return '['+m+']'; })"),
            "a[1]b2"
        );
    }

    #[test]
    fn object_reflection_and_static_inheritance() {
        assert_eq!(run("({a:1}).hasOwnProperty('a')"), "true");
        assert_eq!(run("({a:1}).hasOwnProperty('b')"), "false");
        // Static methods are inherited down the `extends` chain.
        assert_eq!(
            run("class A { static make(){ return 'made'; } } class B extends A {} B.make()"),
            "made"
        );
        // `static m(){ return new this(); }` uses the receiver class.
        assert_eq!(
            run(
                "class A { static create(){ return new this(); } get tag(){ return 'a'; } } class B extends A {} B.create().tag"
            ),
            "a"
        );
        // String.raw interleaves a raw-bearing object with substitutions.
        assert_eq!(run("String.raw({ raw: ['a','b','c'] }, 1, 2)"), "a1b2c");
    }

    #[test]
    fn constructor_function_prototype() {
        // Method on the prototype, resolved through the instance.
        assert_eq!(
            run(
                "function A(n){this.n=n;} A.prototype.m=function(){return this.n*2;}; new A(5).m()"
            ),
            "10"
        );
        // Two-level prototype chain via Object.create.
        assert_eq!(
            run(
                "function A(){} A.prototype.greet=function(){return 'hi';}; function B(){} B.prototype=Object.create(A.prototype); new B().greet()"
            ),
            "hi"
        );
        // `.prototype` is a stable object across reads.
        assert_eq!(run("function A(){} A.prototype.x=1; A.prototype.x"), "1");
    }

    #[test]
    fn named_function_expression_recurses() {
        assert_eq!(
            run("let f = function fac(n){ return n <= 1 ? 1 : n * fac(n-1); }; f(5)"),
            "120"
        );
        // The name is scoped to the expression, not visible outside.
        assert_eq!(
            run("let f = function self(n){ return n===0?0:n+self(n-1); }; f(4)"),
            "10"
        );
    }

    #[test]
    fn array_length_assignment_resizes() {
        assert_eq!(run("let a=[1,2,3,4,5]; a.length=3; a.join(',')"), "1,2,3");
        assert_eq!(
            run("let a=[1,2]; a.length=4; String(a[3]) + ':' + a.length"),
            "undefined:4"
        );
        assert_eq!(run("let a=[1,2,3]; a.length=0; a.length"), "0");
        // String.fromCodePoint.
        assert_eq!(run("String.fromCodePoint(97, 98, 99)"), "abc");
    }

    #[test]
    fn var_hoisting() {
        // A `var` read before its declaration line yields `undefined`.
        assert_eq!(
            run("function f(){ var a = b; var b = 5; return String(a); } f()"),
            "undefined"
        );
        assert_eq!(
            run("function f(){ return typeof later; var later = 1; } f()"),
            "undefined"
        );
        // A var inside a block still hoists to the function scope.
        assert_eq!(run("function f(){ { var x = 9; } return x; } f()"), "9");
    }

    #[test]
    fn arrow_inherits_lexical_this() {
        assert_eq!(
            run("let o = { v: 42, m: function(){ let f = () => this.v; return f(); } }; o.m()"),
            "42"
        );
        // Nested arrows keep inheriting.
        assert_eq!(
            run("let o = { v: 7, m: function(){ return (() => (() => this.v)())(); } }; o.m()"),
            "7"
        );
    }

    #[test]
    fn computed_class_members() {
        // Computed instance method, field, and getter names.
        assert_eq!(
            run(
                "let m='go'; class C{ [m](){return 1;} [m+'V']=2; get [m+'G'](){return 3;} } let c=new C(); c.go() + ':' + c.goV + ':' + c.goG"
            ),
            "1:2:3"
        );
        // Computed static method, field, and getter names.
        assert_eq!(
            run(
                "let s='mk'; class C{ static [s](){return 'a';} static [s+'N']=4; static get [s+'G'](){return 'b';} } C.mk() + ':' + C.mkN + ':' + C.mkG"
            ),
            "a:4:b"
        );
    }

    // The recursion guard (infinite recursion → RangeError, deep finite recursion
    // works) is covered by the `recursion-guard` Test262 corpus test, which runs
    // on a large stack; unit tests here run on the default ~2 MB thread stack,
    // too small for the deep recursion the guard permits.

    #[test]
    fn super_member_read() {
        // super.getter (invoked) and super.method (returned then called).
        assert_eq!(
            run(
                "class B{ constructor(){this._v=10;} get d(){return this._v*2;} m(){return this._v;} } class D extends B{ get d(){return super.d+1;} m(){return super.m()+5;} } let x=new D(); x.d + ':' + x.m()"
            ),
            "21:15"
        );
        // super property as a function value.
        assert_eq!(
            run(
                "class A{ greet(){return 'A';} } class C extends A{ greet(){ let f=super.greet; return f.call(this)+'C'; } } new C().greet()"
            ),
            "AC"
        );
    }

    #[test]
    fn date_setters_and_parse() {
        assert_eq!(
            run("let d=new Date(0); d.setUTCFullYear(2000); d.getUTCFullYear()"),
            "2000"
        );
        assert_eq!(
            run("let d=new Date(0); d.setUTCMonth(5); d.getUTCMonth()"),
            "5"
        );
        assert_eq!(
            run("let d=new Date(0); d.setTime(86400000); d.getUTCDate()"),
            "2"
        );
        assert_eq!(run("Date.parse('1970-01-01T00:00:00.000Z')"), "0");
        assert_eq!(
            run("Date.parse('2000-01-01T00:00:00.000Z') === Date.UTC(2000,0,1)"),
            "true"
        );
        assert_eq!(
            run("new Date('2000-01-01T12:00:00.000Z').getUTCHours()"),
            "12"
        );
        assert_eq!(run("Number.isNaN(Date.parse('garbage'))"), "true");
    }

    #[test]
    fn json_date_and_bigint() {
        assert_eq!(
            run("JSON.stringify(new Date(0))"),
            "\"1970-01-01T00:00:00.000Z\""
        );
        assert_eq!(
            run("JSON.stringify({d:new Date(0)})"),
            "{\"d\":\"1970-01-01T00:00:00.000Z\"}"
        );
        assert_eq!(
            run("try{ JSON.stringify(10n); 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(
            run("try{ JSON.stringify({a:1n}); 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(run("JSON.stringify({a:1,b:'x'})"), "{\"a\":1,\"b\":\"x\"}");
    }

    #[test]
    fn iterator_helpers() {
        let g = "function* g(){yield 1;yield 2;yield 3;yield 4;} ";
        assert_eq!(
            run(&alloc::format!("{g}[...g().map(x=>x*10)].join(',')")),
            "10,20,30,40"
        );
        assert_eq!(
            run(&alloc::format!("{g}[...g().filter(x=>x%2===0)].join(',')")),
            "2,4"
        );
        assert_eq!(run(&alloc::format!("{g}[...g().take(2)].join(',')")), "1,2");
        assert_eq!(run(&alloc::format!("{g}[...g().drop(2)].join(',')")), "3,4");
        assert_eq!(
            run(&alloc::format!("{g}g().toArray().join(',')")),
            "1,2,3,4"
        );
        assert_eq!(run(&alloc::format!("{g}g().reduce((a,b)=>a+b,0)")), "10");
        assert_eq!(run(&alloc::format!("{g}g().reduce((a,b)=>a+b)")), "10");
        assert_eq!(run(&alloc::format!("{g}g().some(x=>x>3)")), "true");
        assert_eq!(run(&alloc::format!("{g}g().every(x=>x>2)")), "false");
        assert_eq!(run(&alloc::format!("{g}g().find(x=>x>2)")), "3");
        assert_eq!(
            run(&alloc::format!(
                "{g}[...g().map(x=>x*2).filter(x=>x>4)].join(',')"
            )),
            "6,8"
        );
        // A helper over the remaining values after one `next()`.
        assert_eq!(
            run(&alloc::format!(
                "{g}let it=g(); it.next(); it.map(x=>x).toArray().join(',')"
            )),
            "2,3,4"
        );
    }

    #[test]
    fn labeled_block_and_class_name() {
        // break out of a labeled block.
        assert_eq!(
            run(
                "let r=[]; blk:{ r.push(1); if(true)break blk; r.push(2); } r.push(3); r.join(',')"
            ),
            "1,3"
        );
        assert_eq!(run("let h='no'; a:{ b:{ break a; } h='in'; } h"), "no");
        // continue to a loop label still works.
        assert_eq!(
            run(
                "let r=[]; outer: for(let i=0;i<3;i++){ for(let j=0;j<3;j++){ if(j===1)continue outer; r.push(i+','+j); } } r.join(';')"
            ),
            "0,0;1,0;2,0"
        );
        // Named class self-reference and `.name`.
        assert_eq!(
            run("let C=class Named{ who(){return Named===C;} }; new C().who()"),
            "true"
        );
        assert_eq!(
            run("let C=class Named{ n(){return Named.name;} }; new C().n()"),
            "Named"
        );
        assert_eq!(run("class Declared{} Declared.name"), "Declared");
    }

    #[test]
    fn arraybuffer_and_dataview() {
        assert_eq!(run("new ArrayBuffer(8).byteLength"), "8");
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setInt32(0,42); v.getInt32(0)"),
            "42"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setInt32(0,-1); v.getUint32(0)"),
            "4294967295"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setUint8(0,255); v.getInt8(0)"),
            "-1"
        );
        assert_eq!(
            run(
                "let v=new DataView(new ArrayBuffer(8)); v.setInt16(0,1000,true); v.getInt16(0,true)"
            ),
            "1000"
        );
        assert_eq!(
            run(
                "let v=new DataView(new ArrayBuffer(8)); v.setInt16(0,1000,true); v.getInt16(0,false)"
            ),
            "-6141"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setFloat64(0,3.14159); v.getFloat64(0)"),
            "3.14159"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setFloat32(0,1.5); v.getFloat32(0)"),
            "1.5"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setInt8(0,300); v.getInt8(0)"),
            "44"
        );
        // Offset view shares the buffer.
        assert_eq!(
            run(
                "let b=new ArrayBuffer(8); let v=new DataView(b); new DataView(b,2).setInt32(0,7); v.getInt32(2)"
            ),
            "7"
        );
    }

    #[test]
    fn typed_arrays() {
        assert_eq!(run("new Uint8Array(3).length"), "3");
        assert_eq!(run("let a=new Uint8Array(1); a[0]=256; a[0]"), "0");
        assert_eq!(run("let a=new Uint8Array(1); a[0]=-1; a[0]"), "255");
        assert_eq!(run("let a=new Int8Array(1); a[0]=200; a[0]"), "-56");
        assert_eq!(
            run("new Uint8ClampedArray([300,-5,100]).join(',')"),
            "255,0,100"
        );
        assert_eq!(run("new Int16Array([70000])[0]"), "4464");
        assert_eq!(run("let f=new Float64Array(1); f[0]=3.14; f[0]"), "3.14");
        assert_eq!(run("new Uint8Array([1,2,3])[1]"), "2");
        assert_eq!(run("new Uint16Array(4).byteLength"), "8");
        assert_eq!(run("new Uint8Array(1).BYTES_PER_ELEMENT"), "1");
        assert_eq!(run("new Uint8Array([1,2,3]) instanceof Uint8Array"), "true");
        assert_eq!(
            run("new Uint8Array([1,2,3]).map(x=>x*2).join(',')"),
            "2,4,6"
        );
        assert_eq!(run("[...new Uint8Array([8,9])].join(',')"), "8,9");
    }

    #[test]
    fn bigint_typed_arrays() {
        // Both kinds exist, are 64-bit, and share the %TypedArray% hierarchy.
        assert_eq!(run("typeof BigInt64Array"), "function");
        assert_eq!(run("typeof BigUint64Array"), "function");
        assert_eq!(run("BigInt64Array.BYTES_PER_ELEMENT"), "8");
        assert_eq!(run("BigUint64Array.BYTES_PER_ELEMENT"), "8");
        assert_eq!(run("new BigInt64Array(2).BYTES_PER_ELEMENT"), "8");
        assert_eq!(run("BigInt64Array.name"), "BigInt64Array");
        assert_eq!(
            run("Object.getPrototypeOf(BigInt64Array)===Object.getPrototypeOf(Int8Array)"),
            "true"
        );
        assert_eq!(run("new BigInt64Array(3).length"), "3");
        assert_eq!(run("new BigInt64Array(2)[0] === 0n"), "true");
        // Elements are BigInt; reading yields a BigInt, writing accepts BigInt.
        assert_eq!(run("var a=new BigInt64Array([1n,2n]); a[1]===2n"), "true");
        assert_eq!(run("new BigInt64Array([-1n])[0]"), "-1");
        // Little-endian i64 / u64 codec with low-64-bit two's-complement wrapping.
        assert_eq!(
            run("new BigUint64Array([18446744073709551615n])[0]"),
            "18446744073709551615"
        );
        assert_eq!(
            run("var a=new BigUint64Array(1); a[0]=-1n; a[0]"),
            "18446744073709551615"
        );
        assert_eq!(
            run("var a=new BigInt64Array(1); a[0]=18446744073709551617n; a[0]"),
            "1"
        );
        // ToBigInt on write: a Boolean / String coerces; a Number throws TypeError.
        assert_eq!(
            run("var a=new BigInt64Array(2); a[0]=true; a[1]='5'; a.join(',')"),
            "1,5"
        );
        assert_eq!(
            run("try{var a=new BigInt64Array(1);a[0]=5;'no'}catch(e){e.constructor.name}"),
            "TypeError"
        );
        assert_eq!(
            run("try{new BigInt64Array([1,2]);'no'}catch(e){e.constructor.name}"),
            "TypeError"
        );
        // Methods are BigInt-aware.
        assert_eq!(
            run("new BigInt64Array([5n,6n,7n]).slice(1).join(',')"),
            "6,7"
        );
        assert_eq!(
            run("new BigInt64Array([5n,6n,7n]).subarray(1).join(',')"),
            "6,7"
        );
        assert_eq!(
            run("var a=new BigInt64Array(3); a.set([9n,8n],1); a.join(',')"),
            "0,9,8"
        );
        assert_eq!(run("new BigInt64Array(2).fill(7n).join(',')"), "7,7");
        assert_eq!(
            run("try{new BigInt64Array(2).fill(3);'no'}catch(e){e.constructor.name}"),
            "TypeError"
        );
        assert_eq!(run("BigInt64Array.of(3n,4n).join(',')"), "3,4");
        assert_eq!(
            run("var s=0n; for(var x of new BigInt64Array([1n,2n,3n])) s+=x; s===6n"),
            "true"
        );
        // A typed array constructed from another BigInt typed array copies values.
        assert_eq!(
            run(
                "var a=new BigUint64Array([10n,20n]); var b=new BigUint64Array(a); b[0]===10n && b!==a"
            ),
            "true"
        );
        // DataView round-trips the 64-bit BigInt accessors (little-endian arg).
        assert_eq!(
            run(
                "var dv=new DataView(new ArrayBuffer(8)); dv.setBigInt64(0,-7n); dv.getBigInt64(0)===-7n"
            ),
            "true"
        );
        assert_eq!(
            run(
                "var dv=new DataView(new ArrayBuffer(8)); dv.setBigUint64(0,5n,true); dv.getBigUint64(0,true)===5n"
            ),
            "true"
        );
        assert_eq!(
            run(
                "try{new DataView(new ArrayBuffer(8)).setBigInt64(0,5);'no'}catch(e){e.constructor.name}"
            ),
            "TypeError"
        );
    }

    #[test]
    fn typed_array_view_aliasing() {
        // Sibling views over one ArrayBuffer share bytes intrinsically.
        assert_eq!(
            run(
                "let b=new ArrayBuffer(8); let u=new Uint8Array(b); let f=new Float64Array(b); u[0]=255; f[0]>0"
            ),
            "true"
        );
        // A DataView write is seen by a typed-array view over the same buffer.
        assert_eq!(
            run(
                "let b=new ArrayBuffer(8); let u=new Uint8Array(b); let dv=new DataView(b); dv.setUint8(1,9); u[1]===9"
            ),
            "true"
        );
        // An offset/length view aliases the right window of the buffer.
        assert_eq!(
            run(
                "let b=new ArrayBuffer(8); let u=new Uint8Array(b,2,4); u[0]=42; new Uint8Array(b)[2]"
            ),
            "42"
        );
        // `subarray` shares the parent's buffer (not a copy).
        assert_eq!(
            run("let u=new Uint8Array([1,2,3,4]); let s=u.subarray(1,3); s[0]=99; u[1]"),
            "99"
        );
        // `.set`, `.fill`, `.copyWithin`, `.byteOffset`, and object-form JSON.
        assert_eq!(
            run("let u=new Uint8Array(4); u.set([5,6],1); u.join(',')"),
            "0,5,6,0"
        );
        assert_eq!(run("new Uint8Array([1,2,3]).fill(7).join(',')"), "7,7,7");
        assert_eq!(
            run("new Uint8Array([1,2,3,4]).copyWithin(0,2).join(',')"),
            "3,4,3,4"
        );
        assert_eq!(
            run("new Uint8Array(b=new ArrayBuffer(4),2).byteOffset"),
            "2"
        );
        assert_eq!(run("Array.isArray(new Uint8Array([1]))"), "false");
        assert_eq!(
            run("JSON.stringify(new Uint8Array([1,2,3]))"),
            "{\"0\":1,\"1\":2,\"2\":3}"
        );
        // BigInt64 round-trips through a DataView.
        assert_eq!(
            run(
                "let dv=new DataView(new ArrayBuffer(8)); dv.setBigInt64(0,-1n); dv.getBigInt64(0).toString()"
            ),
            "-1"
        );
    }

    #[test]
    fn primitive_wrapper_objects() {
        // Number wrapper.
        assert_eq!(run("typeof new Number(5)"), "object");
        assert_eq!(run("new Number(5).valueOf()"), "5");
        assert_eq!(run("new Number(5) + 3"), "8");
        assert_eq!(run("new Number(255).toString(16)"), "ff");
        assert_eq!(run("new Number(5) instanceof Number"), "true");
        // String wrapper.
        assert_eq!(run("new String('hello').length"), "5");
        assert_eq!(run("new String('abc')[1]"), "b");
        assert_eq!(run("new String('HELLO').toLowerCase()"), "hello");
        assert_eq!(run("new String('a') + 'b'"), "ab");
        assert_eq!(run("new String('x') instanceof String"), "true");
        // Boolean wrapper.
        assert_eq!(run("new Boolean(false).valueOf()"), "false");
        assert_eq!(run("new Boolean(false) ? 'truthy' : 'falsy'"), "truthy");
        assert_eq!(run("new Boolean(true) instanceof Boolean"), "true");
        // Defaults.
        assert_eq!(run("new Number().valueOf()"), "0");
        assert_eq!(run("new String().valueOf()"), "");
    }

    #[test]
    fn sloppy_this_is_global_object() {
        // Sloppy plain call: `this` is the global object.
        assert_eq!(
            run("(function(){ function f(){return this===globalThis;} return f(); })()"),
            "true"
        );
        assert_eq!(
            run("(function(){ function f(){return typeof this;} return f(); })()"),
            "object"
        );
        assert_eq!(
            run("(function(){ function f(){return this===globalThis;} return f.call(null); })()"),
            "true"
        );
        // Nested plain function.
        assert_eq!(
            run(
                "(function(){ var o={m(){ function inner(){return this===globalThis;} return inner(); }}; return o.m(); })()"
            ),
            "true"
        );
        // Strict (lexical) keeps `this` undefined.
        assert_eq!(
            run(
                "(function(){'use strict'; function f(){return this===undefined;} return f(); })()"
            ),
            "true"
        );
        assert_eq!(
            run("(function(){'use strict'; function f(){return this;} return f.call(null); })()"),
            "null"
        );
        // A method receiver and a lexical arrow `this` are unaffected.
        assert_eq!(
            run("(function(){ var o={x:5,m(){return this.x;}}; return o.m(); })()"),
            "5"
        );
        assert_eq!(
            run("(function(){ var o={x:9,m(){var a=()=>this.x;return a();}}; return o.m(); })()"),
            "9"
        );
    }

    #[test]
    fn strict_mode_undeclared_assignment() {
        // Strict mode: an implicit-global assignment throws ReferenceError.
        assert_eq!(
            run(
                "(function(){'use strict'; try{ undeclaredX=1; return 'no'; }catch(e){ return e instanceof ReferenceError ? 'ref' : 'other'; }})()"
            ),
            "ref"
        );
        // Sloppy mode still creates the global.
        assert_eq!(
            run("(function(){ sloppyG=5; return typeof sloppyG; })()"),
            "number"
        );
        // Strict propagates to a nested function.
        assert_eq!(
            run(
                "(function(){'use strict'; return (function(){ try{nx=1;return 'no';}catch(e){return 'ref';} })(); })()"
            ),
            "ref"
        );
        // A declared binding is assignable under strict mode.
        assert_eq!(
            run("(function(){'use strict'; let x=1; x=2; return x; })()"),
            "2"
        );
        // Program-level `use strict`.
        assert_eq!(run("'use strict'; var ok='y'; ok"), "y");
        // Strict: writing a read-only property throws; sloppy silently ignores it.
        assert_eq!(
            run(
                "(function(){'use strict'; let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); try{o.x=2;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); o.x=2; o.x"),
            "1"
        );
        // Strict: a frozen object rejects writes.
        assert_eq!(
            run(
                "(function(){'use strict'; let o=Object.freeze({a:1}); try{o.a=9;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
    }

    #[test]
    fn block_level_function_hoisting() {
        assert_eq!(
            run("(function(){ {function g(){return 1;}} return typeof g; })()"),
            "function"
        );
        assert_eq!(
            run("(function(){ {function g(){return 42;}} return g(); })()"),
            "42"
        );
        assert_eq!(
            run("(function(){ if(true){function h(){return 5;}} return h(); })()"),
            "5"
        );
        assert_eq!(
            run("(function(){ {{function d(){return 9;}}} return d(); })()"),
            "9"
        );
        // A later block declaration overrides the outer one (function-scoped).
        assert_eq!(
            run(
                "(function(){ function f(){return 'o';} {function f(){return 'i';}} return f(); })()"
            ),
            "i"
        );
        // Top-level hoisting is unaffected.
        assert_eq!(
            run("(function(){ return e(); function e(){return 'h';} })()"),
            "h"
        );
    }

    #[test]
    fn for_await_of_parses_and_runs() {
        // `for await` parses inside an async function and the call yields a promise.
        assert_eq!(
            run(
                "async function f(){ let s=0; for await(const x of [1,2,3]) s+=x; return s; } typeof f()"
            ),
            "object"
        );
        // An async generator is iterable with for-await.
        assert_eq!(
            run(
                "async function* g(){ yield 1; } async function f(){ for await(const x of g()){} } typeof f"
            ),
            "function"
        );
        // A regular for-of (no await) is unaffected by the AST change.
        assert_eq!(run("let s=0; for(const x of [1,2,3]) s+=x; s"), "6");
    }

    #[test]
    fn intl_number_and_datetime_format() {
        assert_eq!(
            run("new Intl.NumberFormat('en-US').format(1234.5)"),
            "1,234.5"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US').format(1000000)"),
            "1,000,000"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(1234.5)"),
            "$1,234.50"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{style:'currency',currency:'JPY'}).format(1234)"),
            "¥1,234"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{style:'percent'}).format(0.25)"),
            "25%"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{minimumFractionDigits:2}).format(5)"),
            "5.00"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{useGrouping:false}).format(1234567)"),
            "1234567"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US').format(-1234.5)"),
            "-1,234.5"
        );
        // Callable without `new`.
        assert_eq!(run("Intl.NumberFormat('en-US').format(42)"), "42");
        assert_eq!(
            run("new Intl.DateTimeFormat('en-US').format(new Date(Date.UTC(2020,5,15)))"),
            "6/15/2020"
        );
    }

    #[test]
    fn date_string_methods() {
        assert_eq!(run("new Date(0).toDateString()"), "Thu Jan 01 1970");
        assert_eq!(
            run("new Date(0).toUTCString()"),
            "Thu, 01 Jan 1970 00:00:00 GMT"
        );
        assert_eq!(
            run("new Date(Date.UTC(2020,5,15,10,30,45)).toLocaleString()"),
            "6/15/2020, 10:30:45"
        );
        assert_eq!(
            run("new Date(Date.UTC(2020,5,15)).toLocaleDateString()"),
            "6/15/2020"
        );
        assert_eq!(
            run("new Date(Date.UTC(2021,11,25)).toDateString()"),
            "Sat Dec 25 2021"
        );
    }

    #[test]
    fn base64_btoa_atob() {
        assert_eq!(run("btoa('hi')"), "aGk=");
        assert_eq!(run("btoa('Man')"), "TWFu");
        assert_eq!(run("btoa('M')"), "TQ==");
        assert_eq!(run("btoa('')"), "");
        assert_eq!(run("atob('aGVsbG8=')"), "hello");
        assert_eq!(run("atob(btoa('round trip!'))"), "round trip!");
        assert_eq!(run("btoa('é')"), "6Q==");
        assert_eq!(run("atob('aG k=')"), "hi"); // whitespace ignored
        assert_eq!(
            run("try{btoa('\\u{1F600}');'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
    }

    #[test]
    fn structured_clone_deep_copy() {
        assert_eq!(
            run("let o={b:{c:2}}; let c=structuredClone(o); c.b.c=9; o.b.c"),
            "2"
        );
        assert_eq!(
            run("let c=structuredClone([1,[2,3]]); c[1][0]=9; c[1][0]"),
            "9"
        );
        assert_eq!(run("structuredClone(new Map([['k',1]])).get('k')"), "1");
        assert_eq!(
            run("[...structuredClone(new Set([1,2,3]))].join(',')"),
            "1,2,3"
        );
        assert_eq!(run("structuredClone(new Date(1000)).getTime()"), "1000");
        // Cycles and shared references.
        assert_eq!(
            run("let o={}; o.self=o; let c=structuredClone(o); c.self===c"),
            "true"
        );
        assert_eq!(
            run("let s={v:1}; let c=structuredClone({x:s,y:s}); c.x===c.y"),
            "true"
        );
        // Primitives pass through; functions throw.
        assert_eq!(run("structuredClone(42)"), "42");
        assert_eq!(
            run("try{structuredClone({f:function(){}});'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
    }

    #[test]
    fn uri_encoding_functions() {
        assert_eq!(run("encodeURIComponent('a b&c=d')"), "a%20b%26c%3Dd");
        assert_eq!(run("decodeURIComponent('a%20b%26c')"), "a b&c");
        assert_eq!(run("encodeURI('http://a.com/x y')"), "http://a.com/x%20y");
        assert_eq!(run("encodeURIComponent('café')"), "caf%C3%A9");
        assert_eq!(run("decodeURIComponent('caf%C3%A9')"), "café");
        assert_eq!(run("encodeURIComponent(\"-_.!~*'()\")"), "-_.!~*'()");
        assert_eq!(run("decodeURIComponent('%2f')"), "/");
        // Malformed percent-encoding throws a URIError (a subclass of Error).
        assert_eq!(
            run("try{decodeURIComponent('%zz');'no'}catch(e){e instanceof URIError}"),
            "true"
        );
        assert_eq!(
            run("try{decodeURIComponent('%zz');'no'}catch(e){e instanceof Error}"),
            "true"
        );
    }

    #[test]
    fn global_this_object() {
        assert_eq!(run("typeof globalThis"), "object");
        assert_eq!(run("globalThis.globalThis === globalThis"), "true");
        assert_eq!(run("globalThis.Math.max(1,2,3)"), "3");
        assert_eq!(run("globalThis.parseInt('42px')"), "42");
        assert_eq!(run("globalThis.Array.isArray([])"), "true");
        assert_eq!(run("globalThis.Infinity"), "Infinity");
        assert_eq!(run("globalThis.x = 7; globalThis.x"), "7");
    }

    #[test]
    fn map_set_samevaluezero_and_set_ops() {
        // SameValueZero key matching.
        assert_eq!(run("let m=new Map(); m.set(NaN,'y'); m.get(NaN)"), "y");
        assert_eq!(run("new Set([NaN,NaN,1]).size"), "2");
        assert_eq!(run("let m=new Map(); m.set(-0,'n'); m.get(0)"), "n");
        // ES2025 Set composition.
        assert_eq!(
            run("[...new Set([1,2,3]).union(new Set([3,4]))].join(',')"),
            "1,2,3,4"
        );
        assert_eq!(
            run("[...new Set([1,2,3]).intersection(new Set([2,3,4]))].join(',')"),
            "2,3"
        );
        assert_eq!(
            run("[...new Set([1,2,3]).difference(new Set([2]))].join(',')"),
            "1,3"
        );
        assert_eq!(
            run("[...new Set([1,2]).symmetricDifference(new Set([2,3]))].join(',')"),
            "1,3"
        );
        assert_eq!(run("new Set([1,2]).isSubsetOf(new Set([1,2,3]))"), "true");
        assert_eq!(run("new Set([1,2,3]).isSupersetOf(new Set([1,2]))"), "true");
        assert_eq!(run("new Set([1,2]).isDisjointFrom(new Set([3,4]))"), "true");
        // The argument may be any iterable.
        assert_eq!(
            run("[...new Set([1,2,3]).intersection([2,3,9])].join(',')"),
            "2,3"
        );
    }

    #[test]
    fn parse_float_infinity() {
        assert_eq!(run("parseFloat('Infinity')"), "Infinity");
        assert_eq!(run("parseFloat('-Infinity')"), "-Infinity");
        assert_eq!(run("parseFloat('  +Infinity x')"), "Infinity");
        assert_eq!(run("parseFloat('InfinityX')"), "Infinity");
        assert_eq!(run("Number.isNaN(parseFloat('Inf'))"), "true");
        assert_eq!(run("parseFloat('3.14abc')"), "3.14");
    }

    #[test]
    fn define_property_with_symbol_key() {
        assert_eq!(
            run("let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); o[s]"),
            "42"
        );
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); Object.getOwnPropertyDescriptor(o,s).value"
            ),
            "42"
        );
        // A non-enumerable symbol (defineProperty's default) still appears here.
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); Object.getOwnPropertySymbols(o).length"
            ),
            "1"
        );
        // A symbol-keyed accessor.
        assert_eq!(
            run(
                "let s=Symbol('a'); let o={}; let v=0; Object.defineProperty(o,s,{get(){return v;},set(n){v=n;}}); o[s]=7; o[s]"
            ),
            "7"
        );
        assert_eq!(
            run(
                "let s=Symbol('r'); let o={}; Reflect.defineProperty(o,s,{value:9}); Reflect.getOwnPropertyDescriptor(o,s).value"
            ),
            "9"
        );
    }

    #[test]
    fn error_stack_and_aggregate() {
        assert_eq!(run("typeof new Error('x').stack"), "string");
        assert_eq!(run("new Error('boom').stack.indexOf('boom') >= 0"), "true");
        assert_eq!(run("Object.keys(new Error('x')).indexOf('stack')"), "-1");
        // AggregateError: message is the 2nd arg, `.errors` collects the 1st.
        assert_eq!(
            run(
                "let a=new AggregateError([new Error('a'),new TypeError('b')],'m'); a.message + ':' + a.errors.length + ':' + a.name"
            ),
            "m:2:AggregateError"
        );
        assert_eq!(run("new AggregateError([],'x') instanceof Error"), "true");
        assert_eq!(
            run("new AggregateError(new Set([new Error('x')]),'s').errors.length"),
            "1"
        );
    }

    #[test]
    fn error_cause_option() {
        assert_eq!(run("new Error('m',{cause:'r'}).cause"), "r");
        assert_eq!(run("new TypeError('t',{cause:42}).cause"), "42");
        assert_eq!(run("String(new Error('m').cause)"), "undefined");
        assert_eq!(run("String(new Error('m',{}).cause)"), "undefined");
        assert_eq!(
            run("new Error('o',{cause:new Error('i')}).cause.message"),
            "i"
        );
    }

    #[test]
    fn class_extends_native_error() {
        assert_eq!(
            run(
                "class E extends Error{ constructor(m,c){ super(m); this.name='E'; this.c=c; } } let e=new E('x',5); e.message + ':' + e.c + ':' + e.name"
            ),
            "x:5:E"
        );
        assert_eq!(
            run("class E extends Error{} (new E('m')) instanceof Error"),
            "true"
        );
        assert_eq!(
            run(
                "class E extends Error{ constructor(m){super(m);} } let e=new E('m'); (e instanceof Error) + ',' + (e instanceof E) + ',' + (e instanceof TypeError)"
            ),
            "true,true,false"
        );
        assert_eq!(
            run(
                "class V extends RangeError{} let v=new V(); (v instanceof RangeError) + ',' + (v instanceof Error)"
            ),
            "true,true"
        );
    }

    #[test]
    fn class_field_init_order_and_computed_fields() {
        // A field declared without an initializer must not clobber a constructor
        // write (fields init before the constructor body).
        assert_eq!(
            run(
                "class A{ #b; constructor(v){ this.#b=v; } get b(){ return this.#b; } } new A(100).b"
            ),
            "100"
        );
        assert_eq!(
            run(
                "class A{ #b; constructor(v){ this.#b=v; } add(n){ this.#b+=n; return this.#b; } } let a=new A(100); a.add(50)"
            ),
            "150"
        );
        // Computed instance field names.
        assert_eq!(run("let k='x'; class C{ [k+'1']=7; } new C().x1"), "7");
    }

    #[test]
    fn class_rest_params_and_string_positions() {
        // Class constructor rest parameter (with spread).
        assert_eq!(
            run("class V{constructor(...c){this.c=c;}} new V(...[1,2,3]).c.length"),
            "3"
        );
        assert_eq!(
            run("class V{constructor(a, ...r){this.r=r;}} new V(1,2,3).r.join(',')"),
            "2,3"
        );
        // Class constructor default parameter.
        assert_eq!(
            run("class P{constructor(x=7){this.x=x;}} new P().x + ':' + new P(2).x"),
            "7:2"
        );
        // String prefix/suffix with positions.
        assert_eq!(run("'hello world'.startsWith('world', 6)"), "true");
        assert_eq!(run("'hello world'.endsWith('hello', 5)"), "true");
        assert_eq!(
            run("'hello'.includes('lo', 3) + ':' + 'hello'.includes('he', 1)"),
            "true:false"
        );
    }

    #[test]
    fn arithmetic_object_coercion() {
        assert_eq!(run("[5] - 2"), "3");
        assert_eq!(run("[10] / 2"), "5");
        assert_eq!(run("[6] & 3"), "2");
        assert_eq!(run("[2] ** 3"), "8");
        assert_eq!(run("String({} - 1)"), "NaN");
        assert_eq!(run("-[5]"), "-5");
        assert_eq!(run("new Date(5000) - new Date(2000)"), "3000");
    }

    #[test]
    fn tostring_in_concat_and_property_key() {
        // String.concat honors a user toString.
        assert_eq!(run("'x'.concat({toString(){return 'TS';}})"), "xTS");
        // An object property key is coerced via ToString (toString).
        assert_eq!(
            run("let k={toString(){return 'key';}}; let m={}; m[k]=42; m.key + ':' + m[k]"),
            "42:42"
        );
    }

    #[test]
    fn relational_object_coercion() {
        assert_eq!(run("String([5] < 10)"), "true");
        assert_eq!(run("String([20] > 10)"), "true");
        assert_eq!(run("String([1] < [2])"), "true"); // "1" < "2"
        assert_eq!(run("String([10] < [9])"), "true"); // lexicographic
        assert_eq!(run("String({} < 1)"), "false"); // NaN
        assert_eq!(run("String(new Date(1) < new Date(2))"), "true"); // by timestamp
    }

    #[test]
    fn loose_eq_object_coercion() {
        assert_eq!(run("String([] == false)"), "true"); // []→""→0, false→0
        assert_eq!(run("String([] == 0)"), "true");
        assert_eq!(run("String([0] == false)"), "true"); // [0]→"0"→0
        assert_eq!(run("String({} == 0)"), "false"); // "[object Object]"→NaN
        assert_eq!(run("String({} == {})"), "false"); // distinct objects
        assert_eq!(run("String([1,2] == '1,2')"), "true");
    }

    #[test]
    fn array_string_index_access() {
        assert_eq!(run("let a=[10,20,30]; a['0'] + ':' + a['2']"), "10:30");
        assert_eq!(run("let a=[10,20,30]; let k='1'; a[k]"), "20");
        assert_eq!(
            run("let a=[10,20,30]; String(a['00']) + ':' + String(a['01'])"),
            "undefined:undefined"
        );
        assert_eq!(run("[[1,2],[3,4]]['0']['1']"), "2");
    }

    #[test]
    fn object_literal_async_methods_parse() {
        // `async`/`get`/`set` remain usable as property names.
        assert_eq!(
            run("let async=5; let o={async, get:6, set:7}; o.async + ':' + o.get + ':' + o.set"),
            "5:6:7"
        );
        // An async method is a function whose call yields a promise (object).
        assert_eq!(
            run("let o={ async f(){return 1;} }; typeof o.f"),
            "function"
        );
        assert_eq!(
            run("let o={ async f(){return 1;} }; typeof o.f()"),
            "object"
        );
        assert_eq!(
            run("let k='m'; let o={ async [k](){return 1;} }; typeof o.m"),
            "function"
        );
    }

    #[test]
    fn object_literal_generator_methods() {
        assert_eq!(
            run("let o={ *g(){yield 1;yield 2;} }; [...o.g()].join(',')"),
            "1,2"
        );
        assert_eq!(
            run("let o={ *[Symbol.iterator](){yield 'a';yield 'b';} }; [...o].join(',')"),
            "a,b"
        );
        assert_eq!(
            run("let k='m'; let o={ *[k](){yield 9;} }; [...o.m()].join(',')"),
            "9"
        );
        // The generator method reads `this`.
        assert_eq!(
            run("let o={ v:5, *items(){yield this.v;yield this.v*2;} }; [...o.items()].join(',')"),
            "5,10"
        );
    }

    #[test]
    fn class_symbol_iterator_method() {
        assert_eq!(
            run("class C{ *[Symbol.iterator](){yield 'x';yield 'y';} } [...new C()].join(',')"),
            "x,y"
        );
        // A non-generator iterator method (manual iterator object).
        assert_eq!(
            run(
                "class C{ [Symbol.iterator](){let i=0;return{next:()=>i<3?{value:i++,done:false}:{done:true}};} } [...new C()].join(',')"
            ),
            "0,1,2"
        );
        // for-of uses it too.
        assert_eq!(
            run(
                "class C{ *[Symbol.iterator](){yield 1;yield 2;} } let s=0; for(let v of new C())s+=v; s"
            ),
            "3"
        );
    }

    #[test]
    fn generator_is_its_own_iterator() {
        assert_eq!(
            run("function* g(){yield 1;} let it=g(); it[Symbol.iterator]() === it"),
            "true"
        );
        assert_eq!(
            run(
                "function* g(){yield 1;yield 2;} let it=g(); it[Symbol.iterator]().next().value + ':' + it.next().value"
            ),
            "1:2"
        );
        assert_eq!(
            run("function* g(){yield* [1,2]; yield* 'ab';} [...g()].join(',')"),
            "1,2,a,b"
        );
    }

    #[test]
    fn explicit_symbol_iterator_call() {
        assert_eq!(
            run("let it=[10,20,30][Symbol.iterator](); it.next().value + ',' + it.next().value"),
            "10,20"
        );
        assert_eq!(run("'abc'[Symbol.iterator]().next().value"), "a");
        assert_eq!(
            run("let m=new Map([['k','v']])[Symbol.iterator]().next().value; m[0] + '=' + m[1]"),
            "k=v"
        );
        assert_eq!(run("new Set([1,2])[Symbol.iterator]().next().value"), "1");
        assert_eq!(run("[...[1,2,3][Symbol.iterator]()].join(',')"), "1,2,3");
    }

    #[test]
    fn in_operator_walks_prototype_chain() {
        assert_eq!(run("'a' in {a:1}"), "true");
        assert_eq!(run("'z' in {a:1}"), "false");
        assert_eq!(run("let o=Object.create({x:1}); 'x' in o"), "true");
        assert_eq!(
            run("let o=Object.create(Object.create({deep:1})); 'deep' in o"),
            "true"
        );
        assert_eq!(run("0 in [10,20]"), "true");
        assert_eq!(run("5 in [10,20]"), "false");
    }

    #[test]
    fn for_in_inherited_enumeration() {
        assert_eq!(
            run(
                "let p={a:1}; let o=Object.create(p); o.b=2; let k=[]; for(let x in o)k.push(x); k.sort().join(',')"
            ),
            "a,b"
        );
        // Non-enumerable prototype methods are not enumerated.
        assert_eq!(run("let k=[]; for(let x in {})k.push(x); k.length"), "0");
        // A shadowed inherited key appears once.
        assert_eq!(
            run("let o=Object.create({v:1}); o.v=2; let k=[]; for(let x in o)k.push(x); k.length"),
            "1"
        );
    }

    #[test]
    fn const_reassignment_throws() {
        assert_eq!(
            run("const x=1; try{ x=2; 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(run("const x=1; try{ x=2; }catch(e){} x"), "1");
        assert_eq!(
            run("const n=10; try{ n+=5; 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        // Mutation through a const reference is allowed; let is reassignable.
        assert_eq!(run("const a=[1]; a.push(2); a.length"), "2");
        assert_eq!(run("let y=1; y=2; y"), "2");
        // An inner const shadows without affecting the outer.
        assert_eq!(run("const a=1; { const a=2; } a"), "1");
    }

    #[test]
    fn destructure_any_iterable() {
        // Array binding patterns destructure any iterable, not just arrays.
        assert_eq!(run("let [a,b,c]='xyz'; a+b+c"), "xyz");
        assert_eq!(
            run("let [f,...r]=new Set([1,2,3,4]); f + ':' + r.join(',')"),
            "1:2,3,4"
        );
        assert_eq!(
            run("function* g(){yield 10;yield 20;} let [x,y]=g(); x+y"),
            "30"
        );
        assert_eq!(run("let [[k,v]]=new Map([['a',1]]); k + ':' + v"), "a:1");
    }

    #[test]
    fn computed_member_assignment_eval_order() {
        // The index is resolved before the RHS (which mutates it).
        assert_eq!(
            run("let a=[0,0]; let i=0; a[i] = i = 1; a[0] + ',' + a[1]"),
            "1,0"
        );
        // Compound assignment on a computed element still works.
        assert_eq!(run("let a=[1,2,3]; a[1] *= 10; a.join(',')"), "1,20,3");
        assert_eq!(run("let o={x:5}; let k='x'; o[k] += 3; o.x"), "8");
        // Computed key honoring a setter.
        assert_eq!(run("let o={set v(n){this._v=n*2;}}; o['v']=10; o._v"), "20");
    }

    #[test]
    fn in_operator_array_bounds_and_delete() {
        assert_eq!(run("0 in [1,2,3]"), "true");
        assert_eq!(run("5 in [1,2,3]"), "false"); // out of bounds
        assert_eq!(run("'length' in [1,2,3]"), "true");
        assert_eq!(run("'a' in {a:1}"), "true");
        assert_eq!(run("'b' in {a:1}"), "false");
        // delete clears an array element.
        assert_eq!(
            run("let a=[1,2,3]; delete a[1]; String(a[1]) + ':' + a.length"),
            "undefined:3"
        );
        assert_eq!(run("let o={a:1}; delete o.a; 'a' in o"), "false");
    }

    #[test]
    fn catch_binding_forms() {
        // Destructured catch binding (object and array patterns).
        assert_eq!(
            run(
                "let r; try { throw {code:42, text:'x'}; } catch({code,text}){ r=code+':'+text; } r"
            ),
            "42:x"
        );
        assert_eq!(
            run("let r; try { throw [1,2,3]; } catch([a,b]){ r=a+b; } r"),
            "3"
        );
        // Optional catch binding (no parameter).
        assert_eq!(
            run("let r=false; try { throw 1; } catch { r=true; } r"),
            "true"
        );
        // Named binding still works.
        assert_eq!(
            run("let r; try { throw new Error('m'); } catch(e){ r=e.message; } r"),
            "m"
        );
    }

    #[test]
    fn arguments_object() {
        assert_eq!(
            run(
                "function s(){ var t=0; for (var i=0;i<arguments.length;i++) t+=arguments[i]; return t; } s(1,2,3,4)"
            ),
            "10"
        );
        assert_eq!(
            run("function f(){ return arguments[1]; } f('a','b','c')"),
            "b"
        );
        assert_eq!(run("function f(){ return arguments.length; } f()"), "0");
        // An arrow inherits the enclosing `arguments`.
        assert_eq!(
            run("function outer(){ var a = () => arguments[0]; return a(); } outer('Z')"),
            "Z"
        );
    }

    #[test]
    fn function_call_apply_bind() {
        assert_eq!(
            run("function f(p){return p + ':' + this.n;} f.call({n:7}, 'a')"),
            "a:7"
        );
        assert_eq!(
            run("function f(a,b){return a+b+this.n;} f.apply({n:1}, [2,3])"),
            "6"
        );
        assert_eq!(
            run(
                "function f(a,b){return a+b+this.n;} let g=f.bind({n:10}, 5); g(20) + ':' + typeof g"
            ),
            "35:function"
        );
        assert_eq!(run("Math.max.apply(null, [3,9,2])"), "9");
    }

    #[test]
    fn prototype_chains() {
        // Inherited data property and method (this-bound), own shadows inherited.
        assert_eq!(
            run(
                "let p={k:'base',m:function(){return this.n;}}; let o=Object.create(p); o.n=7; o.k + ':' + o.m()"
            ),
            "base:7"
        );
        // Object.keys excludes inherited; getPrototypeOf identity.
        assert_eq!(
            run("let p={a:1}; let o=Object.create(p); o.b=2; Object.keys(o).join(',')"),
            "b"
        );
        assert_eq!(
            run("let p={}; Object.getPrototypeOf(Object.create(p)) === p"),
            "true"
        );
        assert_eq!(
            run("Object.getPrototypeOf(Object.create(null)) === null"),
            "true"
        );
        // Two-level chain: nearest prototype wins.
        assert_eq!(
            run("let a={x:1}; let b=Object.create(a); b.x=2; let c=Object.create(b); c.x"),
            "2"
        );
        // setPrototypeOf installs the link.
        assert_eq!(run("let o={}; Object.setPrototypeOf(o,{v:9}); o.v"), "9");
    }

    #[test]
    fn proxy_get_set_traps() {
        // get trap with fallthrough; set trap transforming the value.
        assert_eq!(
            run(
                "let t={a:1}; let p=new Proxy(t,{get:function(o,k){return k in o?o[k]:'def';}}); p.a + ':' + p.zzz"
            ),
            "1:def"
        );
        assert_eq!(
            run(
                "let t={}; let p=new Proxy(t,{set:function(o,k,v){o[k]=v*3;return true;}}); p.n=4; t.n"
            ),
            "12"
        );
        // No-trap handler forwards to the target.
        assert_eq!(
            run("let p=new Proxy({x:5},{}); p.y=6; '' + p.x + p.y"),
            "56"
        );
        assert_eq!(run("typeof new Proxy({}, {})"), "object");
        // has trap (for `in`) and deleteProperty trap (for `delete`).
        assert_eq!(
            run(
                "let p=new Proxy({a:1},{has:function(t,k){return k==='magic'||k in t;}}); '' + ('a' in p) + ('magic' in p) + ('z' in p)"
            ),
            "truetruefalse"
        );
        assert_eq!(
            run(
                "let seen=''; let p=new Proxy({a:1},{deleteProperty:function(t,k){seen=k; delete t[k]; return true;}}); delete p.a; seen + ':' + ('a' in p)"
            ),
            "a:false"
        );
        // Forwarding `in`/`delete` with no traps.
        assert_eq!(
            run("let p=new Proxy({x:1},{}); let r = 'x' in p; delete p.x; '' + r + ('x' in p)"),
            "truefalse"
        );
    }

    #[test]
    fn instanceof_array_object_and_fromentries_map() {
        assert_eq!(run("[] instanceof Array"), "true");
        assert_eq!(run("({}) instanceof Array"), "false");
        assert_eq!(run("[] instanceof Object"), "true");
        assert_eq!(run("({}) instanceof Object"), "true");
        assert_eq!(run("'str' instanceof Object"), "false"); // primitive
        // Object.fromEntries from a Map.
        assert_eq!(
            run("let m=new Map([['x',10],['y',20]]); let o=Object.fromEntries(m); o.x + ':' + o.y"),
            "10:20"
        );
    }

    #[test]
    fn collection_foreach_thisarg() {
        assert_eq!(
            run(
                "let r=[]; new Map([['a',1],['b',2]]).forEach(function(v,k){ r.push(k+':'+v*this.m); }, {m:10}); r.join(',')"
            ),
            "a:10,b:20"
        );
        assert_eq!(
            run("let s=0; new Set([1,2,3]).forEach(function(v){ s+=v*this.m; }, {m:2}); s"),
            "12"
        );
        // The callback also gets the collection as the third argument.
        assert_eq!(
            run("let n; new Set([1]).forEach((v,k,coll)=>{ n=coll.size; }); n"),
            "1"
        );
    }

    #[test]
    fn map_set_clear_and_assign_getters() {
        assert_eq!(
            run("let m=new Map(); m.set('a',1).set('b',2); m.clear(); m.size + ':' + m.has('a')"),
            "0:false"
        );
        assert_eq!(
            run("let s=new Set([1,2,3]); s.clear(); s.add(5).add(5); s.size"),
            "1"
        );
        // Object.assign invokes getters.
        assert_eq!(
            run(
                "let src={a:1, get b(){ return this.a + 1; }}; let t=Object.assign({}, src); t.a + ',' + t.b"
            ),
            "1,2"
        );
        assert_eq!(run("Object.assign({}, {x:1}, {y:2}, {x:9}).x"), "9");
    }

    #[test]
    fn bigint_shifts() {
        assert_eq!(run("(1n << 8n).toString()"), "256");
        assert_eq!(run("(256n >> 2n).toString()"), "64");
        assert_eq!(run("(2n ** 32n).toString()"), "4294967296");
        assert_eq!(run("(-8n >> 1n).toString()"), "-4");
        assert_eq!(run("(-7n >> 1n).toString()"), "-4"); // arithmetic floor
        assert_eq!(run("(5n << -1n).toString()"), "2"); // negative count reverses
    }

    #[test]
    fn number_of_bigint_and_collection_from_iterable() {
        // Number(bigint) → the double value.
        assert_eq!(run("Number(100n)"), "100");
        assert_eq!(run("Number(-7n)"), "-7");
        assert_eq!(run("Number(2n ** 10n)"), "1024");
        // Set/Map seed from any iterable, incl. a string.
        assert_eq!(run("[...new Set('hello')].join('')"), "helo");
        assert_eq!(run("new Set([1,1,2,3]).size"), "3");
        assert_eq!(
            run("let m=new Map([['a',1],['b',2]]); m.get('a') + m.get('b')"),
            "3"
        );
    }

    // `Promise.any` (first-fulfilled / AggregateError-when-all-reject) is covered
    // by the `promise-allsettled-any` Test262 corpus test, which awaits the
    // settled value (microtask timing isn't observable synchronously here).

    #[test]
    fn weakref_and_finalization_registry() {
        assert_eq!(
            run("let o={x:1}; let r=new WeakRef(o); (r.deref()===o) + ':' + r.deref().x"),
            "true:1"
        );
        assert_eq!(
            run("typeof WeakRef + ',' + typeof FinalizationRegistry"),
            "function,function"
        );
        assert_eq!(
            run(
                "let reg=new FinalizationRegistry(()=>{}); reg.register({}, 'h'); reg.unregister('t')"
            ),
            "false"
        );
        assert_eq!(run("new WeakRef([1,2,3]).deref().length"), "3");
    }

    #[test]
    fn math_exp_log_reflect_assign_array() {
        // Math.exp / Math.log.
        assert_eq!(run("Math.round(Math.exp(0))"), "1");
        assert_eq!(run("Math.round(Math.log(Math.E))"), "1");
        // Object.assign spreads an array source's indices.
        assert_eq!(
            run(
                "let o=Object.assign({}, ['a','b','c']); o[0] + o[2] + ':' + Object.keys(o).join(',')"
            ),
            "ac:0,1,2"
        );
        // Reflect.has walks the chain; Reflect.set updates array storage.
        assert_eq!(run("Reflect.has(Object.create({k:1}), 'k')"), "true");
        assert_eq!(
            run("let a=[1,2,3]; Reflect.set(a, 3, 4); a[3] + ':' + a.length"),
            "4:4"
        );
        assert_eq!(
            run("Reflect.defineProperty({}, 'x', {value:5}) === true"),
            "true"
        );
        assert_eq!(
            run(
                "let o={}; Reflect.defineProperty(o,'x',{value:9,enumerable:true}); Reflect.getOwnPropertyDescriptor(o,'x').value"
            ),
            "9"
        );
    }

    #[test]
    fn reflect_and_weak_collections() {
        // Reflect mirrors the fundamental operations.
        assert_eq!(run("let o={a:1}; Reflect.get(o,'a')"), "1");
        assert_eq!(run("let o={}; Reflect.set(o,'x',5); o.x"), "5");
        assert_eq!(
            run("Reflect.has({a:1},'a') + ':' + Reflect.has({a:1},'z')"),
            "true:false"
        );
        assert_eq!(run("Reflect.ownKeys({a:1,b:2,c:3}).length"), "3");
        assert_eq!(
            run("function f(a,b){return a+b+this.n;} Reflect.apply(f,{n:10},[1,2])"),
            "13"
        );
        assert_eq!(
            run("function B(v){this.v=v;} Reflect.construct(B,[9]).v"),
            "9"
        );
        // WeakMap / WeakSet (object-keyed; bounded — no true weakness).
        assert_eq!(
            run("let k={}; let m=new WeakMap(); m.set(k,'v'); m.get(k) + ':' + m.has(k)"),
            "v:true"
        );
        // Weak collections are recognized by instanceof and chain from set/add.
        assert_eq!(run("(new WeakMap()).set({}, 1) instanceof WeakMap"), "true");
        assert_eq!(run("(new WeakSet()).add({}) instanceof WeakSet"), "true");
        assert_eq!(
            run("let s=new WeakSet(); let o={}; s.add(o); s.has(o) + ':' + s.has({})"),
            "true:false"
        );
    }

    #[test]
    fn proxy_revocable() {
        // Works before revoke; every operation throws after.
        assert_eq!(
            run(
                "let r=Proxy.revocable({a:1},{get:function(t,k){return t[k];}}); let b=r.proxy.a; r.revoke(); let after='ok'; try { r.proxy.a; } catch(e){ after='threw'; } b + ':' + after"
            ),
            "1:threw"
        );
        assert_eq!(
            run("let r=Proxy.revocable({},{}); typeof r.proxy + ',' + typeof r.revoke"),
            "object,function"
        );
    }

    #[test]
    fn proxy_apply_construct_traps() {
        // apply trap intercepts a call; typeof a function proxy is "function".
        assert_eq!(
            run(
                "function f(a,b){return a+b;} let p=new Proxy(f,{apply:function(t,th,a){return a[0]*a[1];}}); p(3,4) + ':' + typeof p"
            ),
            "12:function"
        );
        assert_eq!(
            run("function f(a){return a+1;} let p=new Proxy(f,{}); p(9)"),
            "10"
        );
        // construct trap intercepts `new`.
        assert_eq!(
            run(
                "function B(v){this.v=v;} let p=new Proxy(B,{construct:function(t,a){return {v:a[0]*2};}}); (new p(5)).v"
            ),
            "10"
        );
        assert_eq!(
            run("function B(v){this.v=v;} let p=new Proxy(B,{}); (new p(7)).v"),
            "7"
        );
    }

    #[test]
    fn symbols() {
        assert_eq!(run("typeof Symbol('x')"), "symbol");
        assert_eq!(run("Symbol('hi').toString()"), "Symbol(hi)");
        assert_eq!(run("Symbol('hi').description"), "hi");
        assert_eq!(run("Symbol('a') === Symbol('a')"), "false");
        assert_eq!(run("let s = Symbol(); s === s"), "true");
        assert_eq!(run("Symbol.for('k') === Symbol.for('k')"), "true");
        assert_eq!(run("Symbol.keyFor(Symbol.for('k2'))"), "k2");
        assert_eq!(run("typeof Symbol.iterator"), "symbol");
    }

    #[test]
    fn symbol_keyed_properties() {
        // Distinct symbols are distinct keys; symbol keys are non-enumerable.
        assert_eq!(
            run(
                "let a=Symbol('k'),b=Symbol('k'),o={}; o[a]=1; o[b]=2; o.p=3; '' + o[a] + o[b] + Object.keys(o).join('')"
            ),
            "12p"
        );
        assert_eq!(run("let s=Symbol(); let o={}; o[s]='v'; s in o"), "true");
        assert_eq!(
            run("let s=Symbol(); let o={}; o[s]=1; delete o[s]; o[s]"),
            "undefined"
        );
        assert_eq!(
            run("let o={}; o[Symbol.iterator]='it'; o[Symbol.iterator]"),
            "it"
        );
    }

    #[test]
    fn regex_replace_with_function() {
        assert_eq!(
            run("'a1b2'.replace(/[0-9]/g, function(m){ return '<'+m+'>'; })"),
            "a<1>b<2>"
        );
        assert_eq!(
            run("'1-2'.replace(/(\\d)-(\\d)/, function(_, a, b){ return b+'-'+a; })"),
            "2-1"
        );
        // A string replacement still works.
        assert_eq!(run("'foo'.replace(/o/g, '0')"), "f00");
    }

    #[test]
    fn promise_combinators() {
        // Drive the combinators through output (await/then resolve eagerly).
        assert_eq!(
            out(
                "Promise.all([Promise.resolve(1), 2, Promise.resolve(3)]).then(r => console.log(r.join(',')));"
            ),
            "1,2,3\n"
        );
        assert_eq!(
            out(
                "Promise.race([Promise.resolve('a'), Promise.resolve('b')]).then(v => console.log(v));"
            ),
            "a\n"
        );
        assert_eq!(
            out(
                "Promise.all([Promise.resolve(1), Promise.reject('boom')]).catch(e => console.log('caught:' + e));"
            ),
            "caught:boom\n"
        );
    }

    #[test]
    fn empty_string_is_falsy() {
        assert_eq!(run("!!''"), "false");
        assert_eq!(run("!!'x'"), "true");
        assert_eq!(run("'' || 'fallback'"), "fallback");
        assert_eq!(run("if ('') { 'T' } else { 'F' }"), "F");
        assert_eq!(run("Boolean('')"), "false");
        assert_eq!(
            run("[0, '', null, 1, 'a'].filter(function(x){ return x; }).join(',')"),
            "1,a"
        );
    }

    #[test]
    fn array_from_array_like() {
        // `{ length }` array-like with a map callback (tree-walker path).
        assert_eq!(
            run("Array.from({length:3}, function(_,i){ return i*i; }).join(',')"),
            "0,1,4"
        );
        // Array-like with indexed props, no map fn.
        assert_eq!(run("Array.from({length:2, 0:'a', 1:'b'}).join('-')"), "a-b");
        // Still works for real iterables.
        assert_eq!(
            run("Array.from([1,2,3], function(x){ return x*2; }).join(',')"),
            "2,4,6"
        );
    }

    #[test]
    fn array_from_index_and_string_search() {
        assert_eq!(run("[1,2,3,2,1].indexOf(2, 2)"), "3");
        assert_eq!(run("[1,2,3].includes(2, 2)"), "false");
        assert_eq!(run("[5,6,7].indexOf(5, 1)"), "-1");
        assert_eq!(run("'hello world'.search('world')"), "6");
        assert_eq!(run("'abc'.search('z')"), "-1");
    }

    #[test]
    fn collection_iterators() {
        // Map keys/values/entries.
        assert_eq!(
            run("[...new Map([['a',1],['b',2]]).keys()].join(',')"),
            "a,b"
        );
        assert_eq!(
            run("[...new Map([['a',1],['b',2]]).values()].join(',')"),
            "1,2"
        );
        // `entries()` is a real iterator object (with `.next`), not an array.
        assert_eq!(
            run(
                "let e=new Map([['a',1],['b',2]]).entries(); let r=e.next(); r.value.join(':')+'/'+r.done"
            ),
            "a:1/false"
        );
        assert_eq!(
            run("[...new Map([['a',1],['b',2]]).entries()].map(p=>p.join(':')).join(',')"),
            "a:1,b:2"
        );
        // Set values/keys are its elements.
        assert_eq!(run("[...new Set([1,2,3,2]).values()].join(',')"), "1,2,3");
        assert_eq!(run("[...new Set([5,6]).keys()].join(',')"), "5,6");
    }

    #[test]
    fn eager_generators() {
        // for-of and spread over a finite generator.
        assert_eq!(
            run(
                "function* g(n){ for (let i=0;i<n;i++) yield i*i; } let s=[]; for (let v of g(4)) s.push(v); s.join(',')"
            ),
            "0,1,4,9"
        );
        assert_eq!(
            run("function* g(){ yield 'a'; yield 'b'; } [...g()].join('-')"),
            "a-b"
        );
        // The next() iterator protocol.
        assert_eq!(
            run(
                "function* g(){ yield 1; yield 2; } let it=g(); '' + it.next().value + it.next().value + it.next().done"
            ),
            "12true"
        );
        // yield* delegation.
        assert_eq!(
            run(
                "function* inner(){ yield 2; yield 3; } function* outer(){ yield 1; yield* inner(); yield 4; } [...outer()].join(',')"
            ),
            "1,2,3,4"
        );
    }

    #[test]
    fn new_on_constructor_functions() {
        // `this` binding + implicit instance return.
        assert_eq!(run("function P(x){ this.x = x; } new P(7).x"), "7");
        // `instanceof` matches the constructing function, not others.
        assert_eq!(
            run(
                "function P(){} function Q(){} let p = new P(); '' + (p instanceof P) + (p instanceof Q)"
            ),
            "truefalse"
        );
        // An explicit object return overrides the new instance.
        assert_eq!(
            run("function F(){ this.a = 1; return { b: 2 }; } let o = new F(); '' + o.a + o.b"),
            "undefined2"
        );
        // The hidden constructor tag does not enumerate.
        assert_eq!(
            run("function P(){ this.v = 1; } Object.keys(new P()).join(',')"),
            "v"
        );
    }

    #[test]
    fn class_methods_are_non_enumerable() {
        // Methods are callable but absent from enumeration (only public fields
        // show up), and `{...obj}` spread skips them too.
        assert_eq!(
            run(
                "class C { m(){ return 1; } constructor(){ this.a = 1; this.b = 2; } } Object.keys(new C()).join(',')"
            ),
            "a,b"
        );
        assert_eq!(
            run(
                "class C { greet(){ return 'hi'; } } let c = new C(); c.greet() + ':' + Object.keys({ ...c }).length"
            ),
            "hi:0"
        );
    }

    #[test]
    fn optional_calls_destructuring_assign_and_coercion() {
        // Optional calls short-circuit on a nullish callee.
        assert_eq!(run("let o = { f: () => 7 }; o.f?.()"), "7");
        assert_eq!(run("let o = {}; String(o.missing?.())"), "undefined");
        // Destructuring assignment (swap, rest, member targets).
        assert_eq!(run("let a = 1, b = 2; [a, b] = [b, a]; a + ',' + b"), "2,1");
        assert_eq!(
            run("let h, t; [h, ...t] = [1, 2, 3, 4]; h + '|' + t.join(',')"),
            "1|2,3,4"
        );
        assert_eq!(
            run("let p = {}; ({ x: p.px, y: p.py } = { x: 10, y: 20 }); p.px + ',' + p.py"),
            "10,20"
        );
        // `+` ToPrimitive: arrays/objects stringify.
        assert_eq!(run("'' + [1, 2, 3]"), "1,2,3");
        assert_eq!(run("String([1, 2] + [3, 4])"), "1,23,4");
        assert_eq!(run("({}) + '!'"), "[object Object]!");
        // instanceof on error objects.
        assert_eq!(
            run("try { null.x; } catch (e) { '' + (e instanceof TypeError); }"),
            "true"
        );
        assert_eq!(
            run("try { nope; } catch (e) { '' + (e instanceof ReferenceError); }"),
            "true"
        );
    }

    #[test]
    fn labeled_loops_and_do_while() {
        // `continue label` to an outer loop.
        assert_eq!(
            run("let count = 0;
                 outer: for (let i = 0; i < 3; i++) {
                   for (let j = 0; j < 3; j++) {
                     if (j === 1) continue outer;
                     count++;
                   }
                 }
                 count"),
            "3"
        );
        // `break label` out of nested loops.
        assert_eq!(
            run("let hits = 0;
                 search: for (let i = 0; i < 5; i++) {
                   for (let j = 0; j < 5; j++) {
                     hits++;
                     if (i === 1 && j === 1) break search;
                   }
                 }
                 hits"),
            "7"
        );
        // do/while runs the body at least once.
        assert_eq!(
            run("let n = 0, s = 0; do { s += n; n++; } while (n < 4); s"),
            "6"
        );
        assert_eq!(run("let r = 0; do { r++; } while (false); r"), "1");
    }

    #[test]
    fn for_of_for_in_switch() {
        // for-of over an array, a string, a Set, and a Map.
        assert_eq!(run("let s = 0; for (const x of [1, 2, 3]) s += x; s"), "6");
        assert_eq!(
            run("let r = ''; for (const c of 'abc') r += c + '.'; r"),
            "a.b.c."
        );
        assert_eq!(
            run("let s = 0; for (const v of new Set([1, 2, 3, 2])) s += v; s"),
            "6"
        );
        assert_eq!(
            run("let r = ''; for (const [k, v] of new Map([['a', 1], ['b', 2]])) r += k + v; r"),
            "a1b2"
        );
        // for-of with break/continue.
        assert_eq!(
            run("let s = 0; for (const x of [1, 2, 3, 4]) { if (x === 3) break; s += x; } s"),
            "3"
        );
        // for-in over object keys and array indices.
        assert_eq!(
            run("let r = ''; for (const k in { a: 1, b: 2 }) r += k; r"),
            "ab"
        );
        assert_eq!(
            run("let r = ''; for (const i in ['x', 'y', 'z']) r += i; r"),
            "012"
        );
        // for-of binding to an existing variable (no declaration).
        assert_eq!(run("let x; let s = 0; for (x of [10, 20]) s += x; s"), "30");
        // switch with fall-through and default.
        assert_eq!(
            run("function f(n) {
                   let r = '';
                   switch (n) {
                     case 1: r += 'one';
                     case 2: r += 'two'; break;
                     case 3: r += 'three'; break;
                     default: r += 'other';
                   }
                   return r;
                 }
                 f(1) + '|' + f(2) + '|' + f(3) + '|' + f(9)"),
            "onetwo|two|three|other"
        );
    }

    #[test]
    fn destructuring() {
        // Array destructuring with defaults, holes, and rest.
        assert_eq!(run("let [a, b] = [1, 2]; a + b"), "3");
        assert_eq!(run("let [a, , c] = [1, 2, 3]; a + c"), "4");
        assert_eq!(run("let [a, b = 9] = [1]; a + b"), "10");
        assert_eq!(
            run("let [first, ...rest] = [1, 2, 3, 4]; rest.join(',')"),
            "2,3,4"
        );
        // Object destructuring with shorthand, rename, default, and rest.
        assert_eq!(run("let { x, y } = { x: 1, y: 2 }; x + y"), "3");
        assert_eq!(run("let { a: p, b: q } = { a: 10, b: 20 }; p + q"), "30");
        assert_eq!(run("let { m = 7 } = {}; m"), "7");
        assert_eq!(
            run("let { a, ...others } = { a: 1, b: 2, c: 3 }; Object.keys(others).join(',')"),
            "b,c"
        );
        // Nested.
        assert_eq!(run("let { p: { q } } = { p: { q: 42 } }; q"), "42");
        assert_eq!(run("let [[a], [b]] = [[1], [2]]; a + b"), "3");
        // Destructuring function parameters.
        assert_eq!(run("function f([a, b]) { return a * b; } f([3, 4])"), "12");
        assert_eq!(
            run("function g({ x, y }) { return x + y; } g({ x: 5, y: 6 })"),
            "11"
        );
        // Default and rest parameters.
        assert_eq!(run("function h(a, b = 10) { return a + b; } h(5)"), "15");
        assert_eq!(
            run("function r(...xs) { return xs.length; } r(1, 2, 3)"),
            "3"
        );
    }

    #[test]
    fn maps_and_sets() {
        // Map: set/get/has/size/delete.
        assert_eq!(
            run("let m = new Map(); m.set('a', 1); m.set('b', 2); m.get('a') + m.get('b')"),
            "3"
        );
        assert_eq!(run("let m = new Map(); m.set('x', 1); m.has('x')"), "true");
        assert_eq!(run("let m = new Map(); m.set('x', 1); m.size"), "1");
        assert_eq!(
            run("let m = new Map(); m.set('x', 1); m.delete('x'); m.has('x')"),
            "false"
        );
        // set returns the map (chainable); overwriting a key keeps size.
        assert_eq!(
            run("let m = new Map(); m.set('a', 1); m.set('a', 9); m.get('a') + ':' + m.size"),
            "9:1"
        );
        // Map seeded from pairs.
        assert_eq!(
            run("let m = new Map([['a', 1], ['b', 2]]); m.get('b')"),
            "2"
        );
        // Set: add/has/size, dedup, and seeding from an array.
        assert_eq!(
            run("let s = new Set(); s.add(1); s.add(1); s.add(2); s.size"),
            "2"
        );
        assert_eq!(run("let s = new Set([1, 2, 2, 3]); s.size"), "3");
        assert_eq!(run("new Set([1, 2, 3]).has(2)"), "true");
        // forEach over a Map accumulating values.
        assert_eq!(
            run("let m = new Map([['a', 10], ['b', 20]]);
                 let t = 0; m.forEach(v => { t += v; }); t"),
            "30"
        );
        // typeof a Map is object.
        assert_eq!(run("typeof new Map()"), "object");
    }

    #[test]
    fn more_string_methods() {
        assert_eq!(run("'hello world'.slice(0, 5)"), "hello");
        assert_eq!(run("'hello'.slice(-3)"), "llo");
        assert_eq!(run("'a,b,c'.split(',').join('|')"), "a|b|c");
        assert_eq!(run("'hello'.startsWith('he')"), "true");
        assert_eq!(run("'hello'.endsWith('lo')"), "true");
        assert_eq!(run("'a-b-a'.replace('a', 'X')"), "X-b-a"); // first only
        assert_eq!(run("'5'.padStart(3, '0')"), "005");
    }

    #[test]
    fn more_array_methods() {
        assert_eq!(run("[1, 2, 3, 4].slice(1, 3).join(',')"), "2,3");
        assert_eq!(run("[1, 2].concat([3, 4], 5).join(',')"), "1,2,3,4,5");
        assert_eq!(run("[1, 2, 3].reverse().join(',')"), "3,2,1");
        assert_eq!(run("[1, 2, 3, 4].find(x => x > 2)"), "3");
        assert_eq!(run("[1, 2, 3, 4].findIndex(x => x > 2)"), "2");
        assert_eq!(run("[1, 2, 3].some(x => x === 2)"), "true");
        assert_eq!(run("[1, 2, 3].every(x => x > 0)"), "true");
        assert_eq!(run("[1, 2, 3].every(x => x > 1)"), "false");
        // Default sort (string order) and comparator sort.
        assert_eq!(run("[3, 1, 2].sort().join(',')"), "1,2,3");
        assert_eq!(run("[10, 9, 100].sort().join(',')"), "10,100,9"); // string order
        assert_eq!(
            run("[10, 9, 100].sort((a, b) => a - b).join(',')"),
            "9,10,100"
        );
        assert_eq!(run("[3, 1, 2].sort((a, b) => b - a).join(',')"), "3,2,1");
    }

    #[test]
    fn higher_order_array_methods() {
        // map / filter / reduce with closures.
        assert_eq!(run("[1, 2, 3].map(x => x * 2).join(',')"), "2,4,6");
        assert_eq!(
            run("[1, 2, 3, 4].filter(x => x % 2 === 0).join(',')"),
            "2,4"
        );
        assert_eq!(run("[1, 2, 3, 4].reduce((a, b) => a + b, 0)"), "10");
        assert_eq!(run("[1, 2, 3, 4].reduce((a, b) => a + b)"), "10"); // no initial
        // forEach with a closed-over accumulator.
        assert_eq!(
            run("let total = 0; [10, 20, 30].forEach(x => { total += x; }); total"),
            "60"
        );
        // Chained, with a captured multiplier.
        assert_eq!(
            run("let k = 3;
                 [1, 2, 3, 4].filter(x => x > 1).map(x => x * k).reduce((a, b) => a + b, 0)"),
            "27"
        );
    }

    // --- A5: WebAssembly.Memory shares the byte store (#11) ------------------

    /// Hand-assembled wasm module:
    ///   (module
    ///     (memory (export "mem") 1)
    ///     (func (export "store") (param i32 i32) local.get 0 local.get 1 i32.store)
    ///     (func (export "load")  (param i32) (result i32) local.get 0 i32.load))
    fn mem_module_bytes() -> alloc::vec::Vec<u8> {
        let mut m = alloc::vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // Type section: type0 (i32,i32)->(), type1 (i32)->(i32)
        m.extend([
            0x01, 0x0b, 0x02, 0x60, 0x02, 0x7f, 0x7f, 0x00, 0x60, 0x01, 0x7f, 0x01, 0x7f,
        ]);
        // Function section: func0:type0, func1:type1
        m.extend([0x03, 0x03, 0x02, 0x00, 0x01]);
        // Memory section: one memory, min 1
        m.extend([0x05, 0x03, 0x01, 0x00, 0x01]);
        // Export section: "mem" mem0, "store" func0, "load" func1
        m.extend([
            0x07, 0x16, 0x03, 0x03, b'm', b'e', b'm', 0x02, 0x00, 0x05, b's', b't', b'o', b'r',
            b'e', 0x00, 0x00, 0x04, b'l', b'o', b'a', b'd', 0x00, 0x01,
        ]);
        // Code section: func0 stores, func1 loads
        m.extend([
            0x0a, 0x13, 0x02, // section, size, count
            0x09, 0x00, 0x20, 0x00, 0x20, 0x01, 0x36, 0x02, 0x00, 0x0b, // func0
            0x07, 0x00, 0x20, 0x00, 0x28, 0x02, 0x00, 0x0b, // func1
        ]);
        m
    }

    /// A module exporting one memory (`mem`, min 1 page) and one function
    /// (`grow_store`, `(param i32) -> i32`) that grows linear memory by a page,
    /// stores the parameter as a byte at address 70000 (inside the freshly-grown
    /// region), and returns the new page count. Used by the T6 grow-during-call
    /// regression. Hand-assembled because `wat_to_binary` only emits function
    /// exports (not memory exports).
    fn mem_grow_module_bytes() -> alloc::vec::Vec<u8> {
        let mut m = alloc::vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // Type section: type0 (i32)->(i32)
        m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
        // Function section: func0:type0
        m.extend([0x03, 0x02, 0x01, 0x00]);
        // Memory section: one memory, min 1 page (no max).
        m.extend([0x05, 0x03, 0x01, 0x00, 0x01]);
        // Export section: "mem" mem0, "grow_store" func0
        m.extend([
            0x07, 0x14, 0x02, // section id, size, count
            0x03, b'm', b'e', b'm', 0x02, 0x00, // "mem" -> memory 0
            0x0a, b'g', b'r', b'o', b'w', b'_', b's', b't', b'o', b'r', b'e', 0x00,
            0x00, // "grow_store" -> func 0
        ]);
        // Code section: func0 grows by a page, stores param at 70000, returns size.
        m.extend([
            0x0a, 0x14, 0x01, // section id, size, count
            0x12, // body size (18 bytes)
            0x00, // 0 local declarations
            0x41, 0x01, // i32.const 1
            0x40, 0x00, // memory.grow
            0x1a, // drop (the old page count)
            0x41, 0xf0, 0xa2, 0x04, // i32.const 70000
            0x20, 0x00, // local.get 0
            0x3a, 0x00, 0x00, // i32.store8 align=0 offset=0
            0x3f, 0x00, // memory.size
            0x0b, // end
        ]);
        m
    }

    /// Renders `bytes` as a JS array literal (so a test can build a `Uint8Array`).
    fn js_byte_array(bytes: &[u8]) -> alloc::string::String {
        let mut s = alloc::string::String::from("[");
        for (i, b) in bytes.iter().enumerate() {
            if i != 0 {
                s.push(',');
            }
            s.push_str(&alloc::format!("{b}"));
        }
        s.push(']');
        s
    }

    /// Runs `src` with the memory module's bytes pre-installed as the global
    /// `MOD` (a JS array of byte numbers), returning the completion display.
    fn run_wasm(src: &str) -> alloc::string::String {
        let combined = alloc::format!("const MOD = {}; {src}", js_byte_array(&mem_module_bytes()));
        let program = Parser::parse_program(&combined).expect("parse");
        let mut interp = Interp::new();
        let value = interp.run(&program).expect("exec");
        interp.realm().to_display_string(value)
    }

    #[test]
    fn wasm_memory_shares_byte_store_with_js() {
        // A `Uint8Array` over `mem.buffer` sees a write the exported wasm fn made.
        assert_eq!(
            run_wasm(
                "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const u8 = new Uint8Array(mem.buffer);
                 inst.exports.store(16, 0x41);     // wasm writes mem[16] = 65
                 u8[16];"
            ),
            "65"
        );
    }

    #[test]
    fn wasm_reads_js_write_before_call() {
        // A JS write through a view (before the call) is read back by wasm.
        assert_eq!(
            run_wasm(
                "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const u8 = new Uint8Array(mem.buffer);
                 u8[20] = 0;
                 // Store 0x12345678 LE at addr 20 from JS via DataView, then wasm loads it.
                 const dv = new DataView(mem.buffer);
                 dv.setInt32(20, 0x12345678, true);
                 inst.exports.load(20);"
            ),
            "305419896" // 0x12345678
        );
    }

    #[test]
    fn wasm_memory_grow_keeps_same_buffer_object_and_shares() {
        // `mem.grow(1)` keeps the SAME ArrayBuffer object; a view over the grown
        // buffer works and still shares the store with wasm.
        assert_eq!(
            run_wasm(
                "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const before = mem.buffer;
                 const old = mem.grow(1);             // grow by one 64KiB page
                 const same = (mem.buffer === before);
                 const u8 = new Uint8Array(mem.buffer);
                 // Write near the top of the newly-grown region from wasm, read in JS.
                 inst.exports.store(70000, 0x7e);
                 [old, mem.buffer.byteLength, same, u8[70000]].join(',');"
            ),
            "1,131072,true,126"
        );
    }

    /// Runs `src` with the module compiled from `wat` pre-installed as the global
    /// `MOD` (a JS array of byte numbers), returning the completion display.
    fn run_wasm_wat(wat: &str, src: &str) -> alloc::string::String {
        let bin = crate::wasm_spec::wat_to_binary(wat).expect("compile WAT");
        let combined = alloc::format!("const MOD = {}; {src}", js_byte_array(&bin));
        let program = Parser::parse_program(&combined).expect("parse");
        let mut interp = Interp::new();
        let value = interp.run(&program).expect("exec");
        interp.realm().to_display_string(value)
    }

    /// Like [`run_wasm`] but installs the bytes of [`mem_grow_module_bytes`].
    fn run_wasm_grow(src: &str) -> alloc::string::String {
        let combined = alloc::format!(
            "const MOD = {}; {src}",
            js_byte_array(&mem_grow_module_bytes())
        );
        let program = Parser::parse_program(&combined).expect("parse");
        let mut interp = Interp::new();
        let value = interp.run(&program).expect("exec");
        interp.realm().to_display_string(value)
    }

    #[test]
    fn wasm_grow_during_call_persists_new_page(/* T6 */) {
        // An export that GROWS memory by a page *inside the call* and then stores a
        // byte into the freshly-grown region. After the call, a JS `Uint8Array` over
        // `Memory.buffer` must observe both the larger byteLength and the new byte —
        // i.e. the boundary copy-out must not truncate to the pre-call size, and the
        // canonical store must be grown to match the instance's enlarged memory.
        assert_eq!(
            run_wasm_grow(
                "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const pages = inst.exports.grow_store(0x5a);   // store 90 at 70000
                 const u8 = new Uint8Array(mem.buffer);
                 [pages, mem.buffer.byteLength, u8[70000]].join(',');"
            ),
            "2,131072,90"
        );
    }

    #[test]
    fn wasm_repeat_calls_reuse_instance_state() {
        // A mutable global counter incremented by an export must persist across
        // separate JS→wasm calls (proving the same cached module + carried-over
        // instance state are reused rather than re-initialized each call).
        let wat = "(module
            (global $c (mut i32) (i32.const 0))
            (func (export \"inc\") (result i32)
              (global.set $c (i32.add (global.get $c) (i32.const 1)))
              (global.get $c)))";
        assert_eq!(
            run_wasm_wat(
                wat,
                "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const a = inst.exports.inc();   // 1
                 const b = inst.exports.inc();   // 2
                 const c = inst.exports.inc();   // 3
                 [a, b, c].join(',');"
            ),
            "1,2,3"
        );
    }

    // --- A6: embedder buffer-creation API (#11) -----------------------------

    #[test]
    fn embedder_array_buffer_from_bytes_round_trips() {
        // An ArrayBuffer built from owned bytes is visible to JS and round-trips:
        // JS reads the seeded bytes, mutates one, and the owned store reflects it.
        let mut interp = Interp::new();
        let buf = interp.array_buffer_from_bytes(&[10, 20, 30, 40]);
        interp.declare_global("buf", NanBox::handle(buf.to_raw()));
        let program = Parser::parse_program(
            "const v = new Uint8Array(buf); const sum = v[0]+v[1]+v[2]+v[3]; v[1] = 99; sum",
        )
        .expect("parse");
        let value = interp.run(&program).expect("exec");
        assert_eq!(interp.realm().to_display_string(value), "100");
        // The owned store now reads back the JS mutation.
        let bytes_h = interp.array_buffer_bytes_handle(buf).expect("bytes");
        assert_eq!(interp.realm().bytes_at(bytes_h).unwrap(), &[10, 99, 30, 40]);
    }

    #[test]
    #[allow(unsafe_code)] // wraps a leaked 'static region zero-copy (A6)
    fn embedder_array_buffer_from_external_is_zero_copy() {
        // A `'static`/leaked external region wrapped zero-copy: a JS write through
        // a view changes the *external region itself* (proving no copy was made).
        let region: &'static mut [u8] = alloc::vec![0u8; 8].leak();
        region[0] = 1;
        let ptr = region.as_mut_ptr();
        let len = region.len();
        let mut interp = Interp::new();
        // SAFETY: `region` is a leaked `'static` allocation; it stays valid for the
        // realm's lifetime and is never aliased mutably elsewhere during the run.
        let buf = unsafe { interp.array_buffer_from_external(ptr, len, None) };
        interp.declare_global("ext", NanBox::handle(buf.to_raw()));
        let program = Parser::parse_program(
            "const v = new Uint8Array(ext); const seen = v[0]; v[3] = 222; v[7] = 111; seen",
        )
        .expect("parse");
        let value = interp.run(&program).expect("exec");
        // JS saw the externally-seeded byte...
        assert_eq!(interp.realm().to_display_string(value), "1");
        // ...and the external region itself observed the JS writes (zero-copy).
        assert_eq!(region[3], 222);
        assert_eq!(region[7], 111);
    }

    #[test]
    #[allow(unsafe_code)] // wraps a leaked 'static region zero-copy (A6)
    fn embedder_typed_array_over_external_buffer() {
        // `typed_array_over` builds a Float64 view over an external buffer; a JS
        // store is reflected in the raw region (decoded via the same view).
        let region: &'static mut [u8] = alloc::vec![0u8; 16].leak();
        let ptr = region.as_mut_ptr();
        let mut interp = Interp::new();
        // SAFETY: leaked `'static`, uniquely owned for the run.
        let buf = unsafe { interp.array_buffer_from_external(ptr, 16, None) };
        let view = interp.typed_array_over(buf, 8, 0, 2).expect("float64 view");
        // Write 3.5 into element 1 through the realm API, read it back.
        interp.realm_mut().typed_set(view, 1, NanBox::number(3.5));
        let back = interp.realm_mut().typed_get(view, 1).unwrap();
        assert_eq!(back.as_number(), Some(3.5));
        // The external region bytes are the IEEE-754 encoding of 3.5 at offset 8.
        assert_eq!(&region[8..16], &3.5f64.to_le_bytes());
    }

    #[test]
    fn typed_array_view_ctor_validates_bounds() {
        // H2/T1: a length that overruns the buffer is a RangeError.
        assert_eq!(
            run(
                "try{new Uint32Array(new ArrayBuffer(8),0,100);'no'}catch(e){e instanceof RangeError}"
            ),
            "true"
        );
        // A misaligned byteOffset (not a multiple of the element size) is a RangeError.
        assert_eq!(
            run("try{new Uint16Array(new ArrayBuffer(8),1);'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        // A byteOffset past the buffer end is a RangeError.
        assert_eq!(
            run("try{new Uint8Array(new ArrayBuffer(4),8);'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        // A trailing-bytes length not divisible by the element size is a RangeError.
        assert_eq!(
            run("try{new Uint32Array(new ArrayBuffer(6));'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        // A valid aligned view still constructs.
        assert_eq!(
            run("let v=new Uint16Array(new ArrayBuffer(8),2,3); v.length===3 && v.byteOffset===2"),
            "true"
        );
    }

    #[test]
    fn dataview_ctor_and_access_validate_bounds() {
        // M1: an explicit byteLength past the buffer is rejected at construction.
        assert_eq!(
            run(
                "try{new DataView(new ArrayBuffer(8),0,100);'no'}catch(e){e instanceof RangeError}"
            ),
            "true"
        );
        // M1: a stored over-long length cannot be smuggled into an access either;
        // a valid view's out-of-range access still throws.
        assert_eq!(
            run(
                "try{new DataView(new ArrayBuffer(8),0,8).getInt32(6);'no'}catch(e){e instanceof RangeError}"
            ),
            "true"
        );
        // A negative offset is a RangeError.
        assert_eq!(
            run("try{new DataView(new ArrayBuffer(8),-1);'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        // A valid DataView access round-trips.
        assert_eq!(
            run(
                "let dv=new DataView(new ArrayBuffer(8)); dv.setInt32(0,0x01020304); dv.getInt32(0)"
            ),
            "16909060"
        );
    }

    #[test]
    fn typed_set_same_kind_fast_path_and_overlap() {
        // Same-kind copy.
        assert_eq!(
            run(
                "let a=new Uint8Array([1,2,3,4]); let b=new Uint8Array([9,8]); a.set(b,1); a.join(',')"
            ),
            "1,9,8,4"
        );
        // Overlapping copy within the same backing buffer (sibling views).
        assert_eq!(
            run(
                "let buf=new ArrayBuffer(4); let full=new Uint8Array(buf); full.set([1,2,3,4]); \
                 let dst=new Uint8Array(buf,1,3); let src=new Uint8Array(buf,0,3); dst.set(src); full.join(',')"
            ),
            "1,1,2,3"
        );
        // Out-of-range set is a RangeError.
        assert_eq!(
            run("try{new Uint8Array(2).set([1,2,3]);'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        // T2: a saturated offset throws rather than panicking.
        assert_eq!(
            run("try{new Uint8Array(4).set([1,2],1e308);'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        // Different-kind set still coerces correctly (generic path).
        assert_eq!(
            run("let a=new Uint8Array(3); a.set(new Float64Array([1.9,2.9,3.9])); a.join(',')"),
            "1,2,3"
        );
    }

    #[test]
    fn typed_fill_and_copy_within_all_kinds() {
        // fill on each element kind.
        for ctor in [
            "Int8Array",
            "Uint8Array",
            "Uint8ClampedArray",
            "Int16Array",
            "Uint16Array",
            "Int32Array",
            "Uint32Array",
            "Float32Array",
            "Float64Array",
        ] {
            assert_eq!(
                run(&alloc::format!(
                    "let a=new {ctor}(4); a.fill(7,1,3); a.join(',')"
                )),
                "0,7,7,0",
                "fill failed for {ctor}"
            );
            assert_eq!(
                run(&alloc::format!(
                    "let a=new {ctor}([1,2,3,4]); a.copyWithin(0,2); a.join(',')"
                )),
                "3,4,3,4",
                "copyWithin failed for {ctor}"
            );
        }
        // Uint8Clamped fill clamps.
        assert_eq!(
            run("let a=new Uint8ClampedArray(2); a.fill(300); a.join(',')"),
            "255,255"
        );
        // fill with negative bounds (count from the end).
        assert_eq!(
            run("let a=new Int32Array(5); a.fill(9,-2); a.join(',')"),
            "0,0,0,9,9"
        );
    }

    #[test]
    fn typed_array_aliasing_sees_bulk_writes() {
        // A sibling view over the same buffer observes fill/set/copyWithin writes.
        assert_eq!(
            run(
                "let buf=new ArrayBuffer(8); let a=new Uint8Array(buf); let b=new Uint8Array(buf); \
                 a.fill(5); b.join(',')"
            ),
            "5,5,5,5,5,5,5,5"
        );
        assert_eq!(
            run(
                "let buf=new ArrayBuffer(4); let a=new Uint8Array(buf); let b=new Uint8Array(buf); \
                 a.set([10,20,30,40]); a.copyWithin(0,2); b.join(',')"
            ),
            "30,40,30,40"
        );
        // A DataView aliases a typed-array fill.
        assert_eq!(
            run(
                "let buf=new ArrayBuffer(4); let a=new Uint8Array(buf); let dv=new DataView(buf); \
                 a.fill(0xFF); dv.getUint32(0).toString(16)"
            ),
            "ffffffff"
        );
    }
}
