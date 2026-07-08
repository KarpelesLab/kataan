# Kataan Roadmap — the road to a complete JS + WASM engine

Kataan is a JavaScript engine written in pure Rust, usable three ways — a
standalone binary, a Rust library, and a C library. This roadmap describes **the
work that remains** to reach a fully complete, conformant, high-performance
JS + WASM engine, and a runtime that stands beside Node.js / Bun / Deno. Finished
foundations are summarized once (§1) and not re-litigated; everything after is
forward-looking.

> **Headline status (2026-07):** the official tc39/Test262 corpus (~53k tests)
> runs in CI gated by `tests/test262-status.txt`. Current pass-rate **≈ 93.9 %**
> of the ~44k *ran* tests (the ~9k skipped are Temporal / Atomics / agents /
> cross-realm — see §3.9). **ES modules + dynamic `import()` now run** (the
> module-flagged suite is no longer skipped — §3.1). The remaining headline
> language gap is the **Intl services** (now almost entirely the CLDR locale-*data*
> output in the external `intl` crate — unit long/narrow forms, compact-long,
> DurationFormat unit styles, likely-subtags; the *structure* — subclassing,
> `formatRange`, `formatToParts` incl. unit/compact, `resolvedOptions`, Segmenter
> `containing`, `localeCompare`/`toLocaleString`/`Date.toLocale*` option plumbing —
> is done); the long tail is per-builtin and per-construct edges (§3).
>
> Recently converted (ledger-verified): the **live-iteration** cluster (Set/Map +
> typed-array observe mutation mid-iteration), the **class element** cluster
> (private-method timing, static-privates-not-inherited, class-name `const`),
> **Annex B web-compat** (HTML comments + legacy octal in strings/regex),
> **String exotic** own-property surface, **BigInt** on the reused `puremp`
> backend, full **subclassing** across the builtins (incl. Promise / ArrayBuffer /
> WeakRef / AggregateError / Intl `NumberFormat`/`DateTimeFormat`), the **module**
> cluster (cross-module/circular import-closure, re-export-of-import, namespace
> `isSealed`), the **Intl.Segmenter** surface, the **`toLocaleString` family**
> (Number/Date route through the real formatters), **`Float16Array`** (ES2025), and
> a **feature-complete §4.0 Rust embedding API** (register_fn/constructor, handle
> scope, async continuation, native-state + finalizers) with the **C ABI value
> layer** begun. Systematic probing of every builtin/Intl/module area has been
> exhausted at the *structural* level — what remains is CLDR-data, threading
> (Atomics/agents), Temporal, the async-generator microtask rewrite, and the C ABI
> context/callback bridge.

---

## 0. Non-negotiables (unchanged)

- **Pure Rust, no foreign code on the critical path.** Crypto/TLS via
  [`purecrypto`](https://github.com/KarpelesLab/purecrypto), HTTP via
  [`rsurl`](https://github.com/KarpelesLab/rsurl); regex, Intl-lite, GC, and the
  WASM engine are all in-house.
- **`unsafe` is quarantined.** `unsafe_code = "deny"` (not `forbid`); only the
  `ffi` module and a small, audited set of VM/JIT/mmap hot-path primitives opt
  back in with a scoped `#[allow(unsafe_code)]` + safety comment.
- **`no_std` core stays buildable.** `--no-default-features --features alloc`
  must compile in CI; std-only APIs (`f64::fract`/`trunc`, clock, threads) never
  leak into the language core. The full Test262 run uses the `std` build, so a
  no_std break is invisible there — build the matrix.
- **Specification fidelity, measured.** Conformance is the official Test262 (JS)
  and the upstream WebAssembly spec suite (WASM); correctness is never knowingly
  traded for speed without a flag. **Zero-regression rule:** the ledger only ever
  shrinks; a change that fails any not-yet-ledgered test is reverted or fixed
  before commit.
- **Deployable, host-native bytecode + heap.** Compiled bytecode — and an
  initialized heap — are first-class serializable artifacts (§2.3 / §6).

---

## 1. Where we are now (the finished base)

Treat this as done; build on it.

- **Front end:** complete lexer + full ECMAScript parser + AST, with parse-time
  early-error validation (incl. regex literals).
- **Two execution engines that agree:** a tree-walking interpreter (`nbexec`, the
  reference/corpus engine) and a **register bytecode VM** (`nbvm`, the primary
  path for `kataan run` / `eval` / the C ABI). `nbvm` compiles the common
  language and **faults to `nbexec`** for constructs it doesn't lower (a clean
  whole-program fallback via `execute_typed`). A curated dual-path corpus plus
  754 in-crate unit tests gate every change.
- **Object model in production:** NaN-boxed values, hidden classes/shapes +
  transition tree, interned atoms, rope strings, inline-cache slots, a
  generational handle table, and a moving/compacting tracing GC — all behind
  `Realm`.
- **Lazy suspendable coroutines:** generators and `async`/`await` run on a real
  explicit-stack suspension engine — `yield` truly suspends, `next(v)` injection
  and `throw()`-into-`try{yield}` work, and `await` resumes as a **microtask**
  with correct interleaving ordering. Async generators + `for await` + async
  `yield*` (async-iterator protocol) work; remaining async gaps are exact
  per-`await` step ordering in elaborate cases (§3.5).
- **Array model with real holes:** a `NanBox::hole()` sentinel threads through
  iteration, `in`/`hasOwnProperty`, `Object.keys`/`entries`, and the property
  descriptor layer; `Object.defineProperty` on array indices + `length`
  (ArraySetLength: non-writable length, configurable-stop shrink, freeze/seal)
  works via a sparse attribute side-table that leaves the dense fast path
  untouched.
- **ES modules + dynamic `import()`:** module record / link / evaluate with live
  bindings, re-exports, cycles, top-level `await`, namespace objects, and a
  file-resolution host hook (§3.1 for residual edges). **Explicit resource
  management:** `using` / `await using` dispose at scope exit (reverse order, all
  completion paths, SuppressedError aggregation).
- **First-class prototype methods (mostly):** `Array.prototype.map.call(arrayLike)`,
  extracting a method as a value (`const s = [].slice`), and `…prototype.X.call`
  idioms work. **Exception:** `Array`/`Object` are still object-cells, so
  `typeof Array === "object"` and `Array.apply`/`Array instanceof Function` are
  wrong (§3.7).
- **Bytecode codec (`KTBC`)** + **D′ snapshot tier** (`mmap`-reload over the
  moving GC across the reference cell kinds, incl. non-enumerable/internal slots,
  accessors, and `[[Prototype]]` links) + content-addressed, host-tagged artifact
  store. See §2.3 for what remains.
- **Machine-code JIT (x86-64 / Linux, `jit`):** W^X memory; optimizing integer
  path + float path over the SSE-expressible `Math` intrinsics; per-op `eval_*`
  differential oracle. See §2.1.
- **WASM engine (`no_std`):** decode + validate + stack interpreter for the MVP +
  sign-ext + sat-conversion + bulk-memory + multi-value + typed control; a
  `.wast`/WAT harness; the `WebAssembly` builtin with `Module`/`Instance`/
  `Global`/`Memory`/`Table`, host-function imports, and stateful instances. See
  §2.2.
- **Stdlib breadth:** Object/Array/String/Number/Math/JSON (incl. `rawJSON`/
  `isRawJSON`), Map/Set/WeakMap/WeakSet (+ `getOrInsert`/`getOrInsertComputed`),
  **WeakRef** + **FinalizationRegistry**, Symbol (+ real `Symbol.prototype`),
  BigInt, **Promise** (+ combinators, `withResolvers`, `try`), Proxy/Reflect,
  typed arrays (+ `Uint8Array` base64/hex, `from`/`subarray`), DataView, Date,
  in-house RegExp (named groups, lookbehind, `u`/`v` flags, inline modifiers,
  property escapes), `Error.isError`/`Error.prototype.stack`, `RegExp.escape`,
  `Array.fromAsync`, `Math.sumPrecise`, Iterator helpers. C ABI `kt_eval` runs
  scripts end-to-end.

The rest of this document is the gap between that base and "complete."

---

## 2. The three headline engine deliverables

These make Kataan more than a fast interpreter. They are *engine* capabilities
(beyond what Node/Bun expose) and largely independent of the §3 conformance tail.

### 2.1 Complete the machine-code JIT (the *whole* VM, not just numbers) — **generic tier landed**

> **Status (generic value tier, Passes 1-5):** a NanBox-value JIT tier now compiles
> non-numeric hot functions — generic `+`/`-`/`*`/`/`/`%`, comparisons (`<`/`==`/`===`/
> bitwise/shifts), control flow (loops + branches), property access (`GetProp`/`SetProp`),
> calls (`Call`/`CallNative`), and computed element access (`arr[i]`/`.length`) — by
> re-entering the interpreter through runtime helpers over a `*mut Ctx` calling
> convention, with number-fast-paths inline. Exceptions propagate via a `TAG_JIT_THROW`
> sentinel + `jit_pending`; every op shares one `vm_*` implementation with the interpreter
> so the tiers cannot diverge, proven by 36 differential tests (JIT-forced === interpreter,
> incl. exact-once side effects + identical throws). GC-safe by construction today (GC is
> not mid-execution-triggered); a `jit_shadow` root hook is wired for a future allocation-
> triggered GC. **Remaining = the optimization/portability layer** (below): inline
> machine-code shape/element guards **LANDED** (property GET/SET + array element GET/SET
> now emit real inline reads/writes over `repr(C)` heap types, offsets runtime-verified
> by a probe harness that caught a real `Vec`-ptr-offset trap; guard misses + frozen/
> readonly/handle-value/hole/OOB route to the correct helper; heap-churn differential
> proven). Remaining: inline `Call`/`ArrayLen`, generalized deopt/OSR,
> and the shared backend + aarch64. See `JIT_DESIGN.md`.

Today the JIT compiles integer/float numeric functions and bails to the
interpreter for everything else. "Complete" means hot functions JIT regardless of
what they touch.

- **GC-safe native re-entry substrate (the hard core):** a calling convention +
  **stack maps** so the moving GC can find/relocate live references in JIT frames
  and registers across a safepoint; safepoints at allocation/call sites; a sound
  `&mut Realm` re-entry path (the current blocker that keeps object/string ops
  interpreter-only).
- **Non-numeric op lowering:** IC-slot property access (shape guard → slot load,
  deopt on miss), array element load/store with element-kind fast paths,
  string/rope ops, closure/upvalue/`this`/scope access.
- **Calls in JITed code:** direct + polymorphic call sites, native-builtin calls,
  argument/return marshaling, exception propagation across native frames.
- **Tiering & deopt:** a **baseline tier** (copy-and-patch template compile, OSR
  for hot loops) feeding a profiling layer; **deoptimization** back to bytecode
  on a failed speculative guard.
- **Shared native backend:** a Cranelift-style backend (regalloc, exec-memory,
  relocations) that both the JS JIT and the WASM compiler (2.2) lower into,
  targeting native **or WASM output** (the sandbox fallback).
- **Portability:** the backend abstraction makes aarch64 / other OSes additive
  (current path is x86-64/Linux only).

Exit: a hot object/array/string function runs as native code, differentially
verified against the interpreter, conformance corpus green with the JIT forced
on, deopt round-trips proven, no GC-safety holes under fuzz.

### 2.2 A conformant WASM engine + the full upstream suite

Today the WASM engine passes a spec-*derived* corpus over the numeric/control
core. "Complete" means the full standardized feature set and the upstream spec
test suite.

- **Reference types & tables:** `funcref`/`externref`, `table.*`, multiple tables,
  `call_indirect` type checks/traps, element segments (active/passive/declarative),
  `ref.func`/`ref.null`/`ref.is_null`.
- **JS ↔ WASM boundary — remaining:** make `Memory.buffer` a *live shared view*
  with typed arrays/DataView (now unblocked — typed arrays share an ArrayBuffer's
  bytes); imported memories/tables; exposing exported `Memory`/`Table`;
  `compileStreaming`/`instantiateStreaming`; the `externref` bridge. (`validate`,
  `instantiate`/`compile`, host-fn imports, stateful instances, `Global`/`Memory`/
  `Table`, imported/exported globals, traps-as-exceptions are done.)
- **Post-MVP proposals (prioritized):** SIMD (`v128`), threads + `Atomics` on
  shared memory, multi-memory, tail calls, extended-const, then GC types and
  exception handling.
- **Compiling tier:** lower validated WASM through the shared backend (2.1) for a
  real baseline/optimizing WASM tier on the shared GC + heap.
- **Test gate:** wire the official `wabt`/spec `.wast` suite into CI (per-proposal
  pass-rate); fuzz decoder + validator.
- **WASI** (`wasi_snapshot_preview1` core) for running real `.wasm` CLIs.

Exit: the upstream spec suite passes for MVP + reference-types + bulk-memory +
multi-value + sign-ext + sat-conversion (+ a growing post-MVP set); JS↔WASM
round-trips real modules; validator rejects malformed modules under fuzz.

### 2.3 The mmap-able zero-copy D′ layout, fully complete

Snapshot codec round-trips the reference cell kinds and `mmap`-reloads over the
moving GC. "Complete" means *any* live heap snapshots, reloads zero-copy, and
**executes**, with a production-grade shared code-cache.

- **Hidden-state cells — remaining:** generator/async **suspension state**
  (explicit-stack frames are snapshottable now that they're reified — wire them
  in); the `ArrayBuffer` backing-buffer **identity** shared across typed-array
  views; audit each remaining non-object `Cell` variant.
- **Restore-and-execute:** done for closures + cross-runtime, with public Rust
  (`Interp::snapshot`/`restore_snapshot`) and C-ABI (`kt_snapshot`/`kt_restore`)
  bindings. **Remaining:** persist through the content-addressed store below.
- **Shared artifact store (code-cache, §6) — remaining:** lazy per-function bodies
  faulted in on first call; module-local atom remap on load; IC-slot reset; the
  **on-demand byte-swap conversion** for a mismatched host (convert once, re-cache,
  zero-copy thereafter); **read-only pages shared across many concurrent
  contexts/processes** ("hundreds of bases" — immutable bytecode shared, each
  context owns its mutable heap).
- **Untrusted-load verifier:** bounds-checked jumps/indices/stack-depth for
  untrusted artifacts; trusted-cache fast path on version tag + checksum.
- **Heap-snapshot startup:** boot from a snapshot of an initialized realm (skip
  init, not just parsing) — the payoff once the above + GC relocation are solid.

Exit: an arbitrary initialized heap snapshots, reloads zero-copy (or via one-time
conversion) and executes; the code-cache passes the load → run → evict → reload +
cross-tenant dedup churn; the verifier rejects malformed blobs under fuzz.

---

## 3. Language conformance to 100 % Test262

The instrument is the **full upstream Test262** via `tests/test262_official.rs`,
gated by `tests/test262-status.txt`. Items are ordered roughly by ledger weight.
"Done" gaps from earlier roadmaps (sparse holes, lazy generators/async,
first-class prototypes, WeakRef/FinalizationRegistry, regex validation, the
ES2024/25 builtin tail) are intentionally not relisted — see §1.

### 3.1 ES Modules + dynamic `import()` — **largely landed** (residual edges)

The module record / link / evaluate pipeline, live import bindings + TDZ,
re-exports (`export {x} from`, `export *`, `export * as`, default), cycles,
top-level `await`, `import.meta.url`, module namespace exotic objects, and
**dynamic `import()`** (promise of the namespace) are implemented (tree-walker,
`src/nbexec/module.rs`), and the runner now executes `flags:[module]` +
dynamic-import tests with file-relative resolution. Most module early errors are
parse-phase. **Residual:**

- **`import.meta`** exposes only `url` (the full ordinary-object tests remain).
- **Namespace `[[DefineOwnProperty]]`** exotic behavior; a couple of
  ambiguous-binding-propagation edges.
- **Top-level-await ordering** + the strict-mode-loss-after-`await` bug (shared
  with the async/generator model — §3.5).
- **`import.source` / `import.defer`** (source-phase + deferred-import proposals)
  and **import attributes/assertions** — unimplemented (rejected as SyntaxError).
- **Bytecode tier:** modules run on the tree-walker only; nbvm has no module
  support (it faults to nbexec).
- **CLI:** no first-class `kataan run x.mjs` module entry (§4.6).

### 3.2 `class` edge cases (~550)

Classes work broadly, but a long tail fails: private field/method/accessor **brand
checks** in nested/shadowed/before-`super()` contexts, **static blocks** + static
field init order, computed-key evaluation order + abrupt completions,
`#x in obj` ergonomic-brand edges, `super` in field initializers / static
contexts, and `new.target` in edge positions.

### 3.3 Intl services (~525)

`Intl.NumberFormat` (138) / `DateTimeFormat` (95) / `Locale` (77) /
`DurationFormat` (67) / `ListFormat` (52) / `RelativeTimeFormat` (48) /
`Segmenter` / `PluralRules` / `DisplayNames` / `Collator`. The constructors +
option bags + locale negotiation are partly there; the failures are
**formatting-correctness** edges (signDisplay, compact/notation, rounding modes,
currency/unit, range formatting, `formatToParts`) over the embedded trimmed CLDR
data. Regression-prone — fix per-service with the format tests as the gate.

### 3.4 Direct `eval` + Annex B legacy semantics (~360)

`language/eval-code/direct` (173) + `annexB/language` (186): Annex-B
**function-in-block** hoisting (a block `function f(){}` creating a `var`-scoped
binding), direct-eval variable/function declarations into the caller scope,
`with`-scope interactions, and the legacy HTML-comment / octal-escape corners.

### 3.5 Iteration, control, and assignment constructs (~350)

- **`with` statement** (72): scope-object semantics + `@@unscopables`. Touches
  identifier resolution (core) — handle carefully.
- **`for-of` / `for-await-of`** (112): iterator-close on abrupt completion,
  async-from-sync wrapping edges, destructuring-in-head corners.
- **assignment / compound / logical-assignment** (110): destructuring assignment
  targets, getter/setter ordering, `&&=`/`||=`/`??=` short-circuit edges.
- **`super`** (34), **async-generator / generators / yield** (~135): the residual
  exact-ordering and abrupt-completion cases the lazy engine doesn't yet mirror.

### 3.6 Builtin edges (the long per-object tail)

*Landed this cycle (ledger-verified):* the **Array** precise mutators
(sort/reverse/copyWithin run `[[Get]]`/`[[Set]]`/`Delete` on hole/accessor
arrays) + the index-arg **coercion** cluster (`ToIntegerOrInfinity` on
splice/fill/flat/slice/toSpliced) + `C.from`/`C.fromAsync` dense storage;
**String** exotic own-property surface (hasOwnProperty / `in` /
propertyIsEnumerable / descriptors / for-in over index keys); **BigInt** primitive
`[[Prototype]]`; **live** Set/Map/typed-array iteration; and much of **Proxy** §3.7.

Each remaining is "works, but fails spec edges": **Array** (array-like generic
algorithms, species, sort comparator/stability, copyWithin/splice on exotic
receivers), **Object** (185 — descriptor/`defineProperty` corners, property
enumeration order, `__proto__`), **RegExp** (204 — match/replace/split/`Symbol.*`
protocol, sticky/global lastIndex, `d`-indices), **TypedArray** (~165 — full
method set on exotic/resizable buffers, constructor coercion order), **Promise**
(145 — combinator resolve-element identity, thenable adoption order, subclass
species), **Proxy** (66 — per-trap invariant checks; see §3.7), **Iterator** (83 —
helper edges, `Iterator.zip`/`zipKeyed`), **String** (54), **Function** (59 —
`.length`/`.name`/`bind`/`toString`), **ArrayBuffer** (58 — transfer/resize),
**JSON** (46 — `formatToParts`-style edges, source access).

### 3.7 Cross-cutting object-model fixes (high leverage)

- **`Array` / `Object` as functions.** They are object-cells with a `null`
  `[[Prototype]]`, so `typeof Array === "object"`, `Array instanceof Function` is
  false, and `Array.apply`/`.call`/`.bind` are missing. *Naively* setting their
  `[[Prototype]]` to `%Function.prototype%` **broke array method resolution**
  (the ctor object is entangled with `array_proto_intrinsic`) — needs careful
  untangling, not a one-line `set_object_proto`.
- **Trap-less Proxy forwarding through iteration — largely landed.**
  `[...new Proxy([1,2,3],{})]`, `Array.from(proxy)`, generic
  `Array.prototype.*.call(proxy)`, object spread/rest (`{...proxy}` / `{...r} =`),
  `JSON.stringify`, `Object.getOwnPropertyDescriptors`, `Reflect.ownKeys`, and
  `with (proxy)` now route through the proxy protocol (a shared
  `copy_data_properties` for CopyDataProperties; `has_property` delegates to the
  proxy-aware `has_property_proxied`). Proxy as a **write target**
  (`Object.assign`) and `Reflect.set` with a **proxy receiver** run the proper
  `[[Set]]`/`[[DefineOwnProperty]]`. **Remaining:** OrdinarySet must run
  `parent.[[Set]]`, so an array/object index write whose own slot is absent and
  whose **prototype is a proxy/accessor** does not yet fire that trap
  (`Proxy/set/call-parameters-prototype-*`); the per-trap invariant edges in
  `Proxy/set`/`has`/`getOwnPropertyDescriptor`.
- **Static-method `.call` receiver.** `setup_static_methods` binds the
  constructor as `this`; only the `this_aware` list (Promise combinators/`resolve`/
  `reject`/`try`) honors a dynamic `.call` receiver. Generalize for other statics
  whose receiver matters. (`Array.of` now honors a constructor receiver via
  `this_val`, like `Array.from`.)

### 3.8 Resizable / shared memory & atomics (skip-gated today)

- **Resizable/growable `ArrayBuffer`:** length-tracking is partly in; the
  TypedArray methods' out-of-bounds-on-shrink behavior remains.
- **`SharedArrayBuffer` + `Atomics`**: the **single-agent deterministic core is
  done** — `Atomics.add/sub/and/or/xor/exchange/compareExchange/load/store/
  isLockFree` over integer typed arrays, and `SharedArrayBuffer` (constructor,
  `byteLength`/`growable`/`maxByteLength`, `grow`, `slice`, and full typed-array/
  `DataView`/`Atomics` backing). What remains is the genuinely concurrent part:
  `waitAsync`/`notify` + the **`$262.agent`** harness are now IMPLEMENTED via a
  single-threaded COOPERATIVE scheduler (workers run eagerly to completion in a fresh
  realm, shared FIFO report queue, async waiters settle on the microtask queue):
  Atomics 320/388 (82%), un-skipped. True-interleaving (main blocks in `wait` while a
  worker notifies) + real-elapsed-timeout cases are ledgered (need whole-script suspension).

### 3.9 Whole subsystems currently skipped (each a project)

These are removed from the *ran* denominator via `SKIP_FEATURES`; implementing any
one both adds coverage and moves its tests from skipped → ran:

- **Temporal** — IMPLEMENTED (~90%, 4134/4603): all 9 types + Now, un-skipped. Remaining: i128 Duration precision, DST edges, per-type parser polish.
- **Decorators** — IMPLEMENTED (parse + accessor auto-accessors, 24/24 language tests, un-skipped).
- **Tail-call optimization** (proper tail calls).
- **Import attributes / assertions** — IMPLEMENTED (with/assert grammar + JSON/text modules, 94/100, un-skipped).
- **IsHTMLDDA** (`document.all`) — IMPLEMENTED (34/34, un-skipped).
- **cross-realm** (`$262.createRealm`) — IMPLEMENTED (identity bulk 77/204; deep proto-from-ctor-realm ledgered).
- **Tail-call optimization** — IMPLEMENTED (PTC on both tiers, 34/35, un-skipped).

### 3.10 The gate

Keep the full Test262 nightly job; drive the ledger to empty. Fuzz parser, regex,
JSON, and both VMs. Maintain the zero-regression rule and the no_std build matrix.

---

## 4. Host runtime — parity with Node.js / Bun / Deno (`host`)

The language engine is necessary but not sufficient to *replace* `node`/`bun`.
This is the surface a real runtime exposes to scripts.

### 4.0 Embedding & host-function registration API (the foundation)

The bidirectional surface an embedder uses to extend the engine from Rust or C —
and the layer §4.1–4.5's own globals should be (re)built on, so runtime builtins
and third-party host code travel one path. Today this is the biggest embedding
gap: natives are dispatched by a fixed `u16` sentinel id in `call_native` (no
dynamic registration, no per-instance state), and the C ABI is eval-only
(`kt_eval`/`kt_compile`). This makes host code first-class. It is a prerequisite
for §4.1–4.5 and for the `<30 lines` embeddability claim in §7.

- **Dynamic native registry.** A host-function id range above the built-in
  sentinels, each id mapping to a boxed
  `dyn FnMut(&mut Ctx, this, &[Value]) -> Result<Value, Value>` held in a new
  `Cell::HostFn`, routed through `call_native` / `construct` (and faulted to from
  `nbvm` like any native, per §6 "two engines, one truth"). Created from a
  closure, attachable to any object or the global with spec-shaped own
  `name`/`length`.
- **Rust API.** `interp.register_fn(name, |cx, args| …)` plus a `Ctx` handle
  exposing `cx.global()`, object/array/typed-array/error builders, primitive
  marshaling (a `TryFrom<Value>` / `Into<Value>` trait family for the common Rust
  scalars, `String`, `Vec`, `Option`, and byte buffers), `cx.get`/`cx.set`,
  `cx.throw(value)`, and `cx.call(fn, this, args)` / `cx.construct(...)` to
  re-enter JS. Returning `Err(value)` raises it as a JS exception; a Rust
  `panic!` is caught at the boundary and converted to a JS `Error` — it never
  unwinds across an FFI or engine frame.
- **Handle scopes & the moving GC.** Host-held values must survive collection: a
  rooted **handle scope** (an N-API `napi_handle_scope` analog) pins every
  `Value` the host borrows and unpins on drop, with escapable + persistent/global
  handles for values kept across calls. This is the one hard constraint (§6): the
  compacting collector may relocate any un-rooted handle, so the raw `Handle` is
  never exposed to host code directly.
- **C ABI (N-API-shaped, ABI-stable).** Opaque `kt_ctx` / `kt_value`;
  `kt_register_function(ctx, name, cb, userdata)` with a fixed callback signature
  `kt_value (*)(kt_ctx*, kt_value this_, int argc, const kt_value* argv, void* userdata)`;
  value constructors/accessors (`kt_new_number`/`kt_new_string_utf8`/
  `kt_new_object`/`kt_new_array`/`kt_get_property`/`kt_set_property`/
  `kt_to_number`/`kt_get_utf8`/…), `kt_throw`, `kt_call`, and handle-scope
  open/close — a versioned mirror of the Rust API, so a C host reaches parity
  without linking Rust types.
- **Host-backed exotic objects & constructors.** Register host constructors
  (`[[Construct]]`) and objects carrying opaque native state (`Box<dyn Any>` per
  instance, à la N-API `napi_wrap`) with a **finalizer** callback run when the
  instance is collected — the hook native resource cleanup (file handles,
  sockets) needs.
- **Async interop.** A host function may return a promise: `cx.promise()` hands
  back a `(Value, Resolver)` the host settles later from a timer/IO completion,
  integrated with §4.1's loop; plus an adapter so an `async` Rust closure becomes
  a promise-returning JS function.
- **Reentrancy & limits.** Host callbacks run under the same call-depth and
  resource limits (`Limits`); nested `cx.call` shares the microtask/job queue;
  errors from re-entered JS surface back to the host as `Err(Value)`.

Milestone order: (1) dynamic native registry + `Cell::HostFn` + Rust
`register_fn` + value marshaling + handle scope; (2) builders + reentrant
`call`/`construct` + `throw`; (3) the C ABI mirror; (4) promises/async +
host constructors/finalizers. Once (1)–(2) land, migrate a couple of existing
sentinel builtins onto the registry to prove the path end-to-end.

**Status:** the Rust core of (1) + most of (2) has **landed** (tree-walker). A
`Cell::HostFn(u32)` names an entry in an `Interp` host-function registry;
`Interp::register_fn(name, length, closure)` / `register_global_fn` create a
first-class callable (spec-shaped own `name`/`length`, `typeof === "function"`,
`Function.prototype.toString` shape, `new` → `TypeError`). The closure receives
a `Ctx` with value builders (`number`/`string`/`new_object`/`new_array`/…),
property access (`get`/`set`/`has`/`has_own`/`delete`/`own_keys`), value
inspection (`type_of`/`is_callable`/`is_object`/`is_array`), array reads
(`array_len`/`array_get`), argument coercion (`to_number`/`to_string`/
`to_boolean`), error builders (`type_error`/`range_error`/`error`), promise
creation (`resolved_promise`/`rejected_promise`/`is_promise`), reentrant `call`
and `construct` to re-enter JS; `Err(value)` raises a catchable JS exception.
Self-reentrancy onto the same host function is a clean `TypeError` (the `FnMut` is
taken out of its slot for the call). **Host constructors have landed:**
`register_constructor` / `register_global_constructor` make a host function
constructable — `new HostCtor(...)` binds a fresh `this` (its `[[Prototype]]` is
the auto-created `HostCtor.prototype`), runs the closure, and applies the
constructor return rule, so `instanceof` and prototype methods work. A **panic in
host code is trapped** at the boundary (`catch_unwind`, std) and surfaces as a
catchable JS `Error` rather than unwinding across engine frames. The rooted
**handle scope has landed** (§6's one hard constraint): `Ctx::persist(value)`
returns a stable slot index the host holds across calls, `persistent(idx)` reads
it back (reflecting relocation), `release_persistent(idx)` frees it — backed by a
`host_persistent` table that is both a GC root and forwarded on compaction, so a
pinned value survives collection and stays valid when the moving collector relocates
it (never exposing a raw `Handle`). **Async continuation has landed** on top of
it: `Ctx::deferred()` returns a promise plus a token, and
`Interp::resolve_deferred`/`reject_deferred(token, value)` settle it later from a
host timer/IO completion and drain the microtask queue — the capability's
resolve/reject are pinned (persistent) until settled. **Host-backed native state
+ finalizers have landed:** `Ctx::set_native_state(obj, T)` / `native_state::<T>`
wrap opaque Rust state onto a JS object (à la `napi_wrap`) via a weak
`host_native_state` table — pruned in `ephemeron_prune` when the object is
collected (running the state's Rust `Drop` as its finalizer) and forwarded on
compaction. See `examples/embed_host_fn.rs`. **Remaining for §4.0:** the **C ABI**
mirror (a full N-API-shaped FFI surface), `nbvm` host-native fault-through (only
reachable once the VM is the primary path), and migrating a sentinel builtin onto
the registry. The Rust embedding API is otherwise feature-complete.

### 4.1 Event loop & scheduling — **landed** (cooperative)

A complete in-house loop (readiness-based I/O or std threads): `setTimeout`/
`setInterval`/`setImmediate` + `clear*`, `queueMicrotask`, `process.nextTick`,
the microtask/macrotask checkpoint integrated with the Promise job queue,
`AbortController`/`AbortSignal`, unref/ref semantics, and a clean exit when the
loop drains.

### 4.2 Module system & package resolution (with §3.1)

ESM (static + dynamic `import`, `import.meta`, top-level await) **and** CommonJS
interop (`require`, `module.exports`, the wrapper, cycle handling); Node-style
resolution (`node_modules`, `package.json` `exports`/`imports`/`type`, conditional
exports, `.mjs`/`.cjs`), the `node:` builtin prefix, JSON/`.wasm` imports, import
maps. This is the gateway to running **real npm packages** — the practical bar for
Node/Bun parity.

### 4.3 Web platform globals — **largely landed**

`console` (full formatting), `TextEncoder`/`TextDecoder`, `atob`/`btoa`,
`URL`/`URLSearchParams`, `structuredClone`, `performance.now`/marks/measures,
`queueMicrotask`, WHATWG **streams** (Readable/Writable/Transform), `Blob`/`File`/
`FormData`, `EventTarget`/`Event`, `MessageChannel`/`MessagePort`, `WebSocket`,
`btoa` + base64 helpers, timers as globals.

### 4.4 `fetch` + crypto (own crates)

`fetch` (+ `Headers`/`Request`/`Response`/`Blob`/`FormData`, redirects, abort,
streaming bodies) over `rsurl` with `purecrypto` TLS; **`crypto`**
(`getRandomValues`/`randomUUID` done; `subtle` digest/HMAC/AES/RSA/ECDSA/keys)
over `purecrypto`.

### 4.5 Node-compat builtins — **pure subset landed** (Buffer/path/os/util/querystring/process)

A useful, documented subset, `node:`-prefixed: `fs` (sync + promises),
`path`, `os`, `url`, `util` (`inspect`/`promisify`/`TextEncoder`), `events`
(`EventEmitter`), `stream`, `buffer` (`Buffer`), `process` (`argv`/`env`/`cwd`/
`exit`/`platform`/`hrtime`/`stdout`/`stderr`), `net`/`tls`/`http(s)`,
`crypto`, `zlib`, `child_process`, `worker_threads` (shared with §3.8 agents),
`assert`, `timers`. Gaps documented per module.

### 4.6 CLI / runtime ergonomics (the Bun/Deno bar)

`kataan run file.{js,mjs,cjs,ts?}`, a REPL, `--eval`/`-e`, reading from stdin,
source-map-aware stack traces, a permissions/sandbox story, and ideally a
**built-in test runner** + TS/JSX transpile-on-load (Bun/Deno's differentiators).
TypeScript type-stripping (not checking) on import is the high-value, low-cost
piece.

---

## 5. Performance frontier

- **Interpreter:** IC tuning, array element-kind fast paths, rope tuning, finish
  the generational nursery + write barriers on the existing moving GC for
  bump-allocation throughput.
- **Optimizing JIT (after the 2.1 baseline tier):** an SSA IR with inlining,
  escape analysis, range/redundancy elimination, type-feedback speculation with
  guard-based deopt — through the shared backend. The point where we contend with
  V8 on compute-bound code.
- **Benchmarks:** SunSpider/Kraken-style microbenchmarks + realistic scripts;
  cold-start vs `node -e` / `bun`; code-cache load/evict/reload throughput;
  per-object memory vs V8/JSC heaps.

---

## 6. Design invariants that still constrain remaining work

- **Own JS bytecode + VM; WASM is a *peer* engine sharing the backend, not a
  compile target.** Routing JS through WASM would forfeit guard-based deopt. The
  engines share the GC + heap, the native backend, the value/interop boundary,
  and the host runtime; the two bytecodes stay distinct. The one place emitting
  WASM *for JS* is the sandbox fallback (2.1).
- **Serializable, host-native artifacts.** Position-independent (indexed, never
  live pointers), host-native integer encoding with a cheap on-demand byte-swap
  for a mismatched host — not a slow canonical encoding, not a recompile.
  Versioned + integrity-checked: version/flags mismatch or bad checksum →
  recompile; host-encoding mismatch, same version → convert. Atoms module-local +
  remapped on load; IC slots/shapes are runtime state, never serialized; function
  bodies lazy. One artifact store for JS bytecode and compiled WASM.
- **Two engines, one truth.** `nbvm` and `nbexec` must agree; `nbvm` faults
  cleanly to `nbexec` rather than producing a wrong-but-silent result. A new
  builtin landed only in `nbexec` is reached via that fallback — acceptable, but
  note it.
- **No raw handles across the host boundary.** The GC compacts, so any JS value
  held by host (Rust/C) code between engine calls must be rooted through a handle
  scope or a persistent handle (§4.0), never a bare relocatable `Handle`. Host
  callbacks are called at engine safepoints and must not stash un-rooted values;
  a Rust `panic` / C longjmp must be trapped at the boundary and turned into a JS
  throw rather than unwinding through engine frames.

---

## 7. Definition of done

Kataan is "complete" when:

1. **JS:** the full upstream Test262 ledger is **empty** for every implemented
   feature area, and the skip-gated subsystems (Temporal, Atomics/SAB + agents,
   decorators, modules) are themselves implemented and passing — not merely
   skipped.
2. **Runtime:** real npm packages run; the module system, event loop, web-platform
   globals, `fetch`/`crypto`, and the Node-compat subset (§4) cover common
   programs; a CLI + REPL stand beside `node`/`bun` for everyday use.
3. **JIT:** hot functions JIT regardless of object/string/closure content, with a
   baseline + optimizing tier, type-feedback speculation, sound deopt, and a
   GC-safe native re-entry substrate — proven against the interpreter and fuzz.
4. **WASM:** the upstream spec suite passes for MVP + reference-types + the
   implemented proposals, with full JS↔WASM interop, WASI, and a compiling tier
   on the shared backend.
5. **D′ / code-cache:** any initialized heap snapshots, `mmap`-reloads zero-copy
   (or via one-time conversion) and executes; the content-addressed,
   verifier-guarded, cross-context-shareable store passes the hundreds-of-bases
   churn scenario.
6. **Always:** pure Rust, `unsafe` quarantined + audited; CI green across the
   feature matrix incl. `no_std`; embeddable in <30 lines via the Rust and C APIs
   — including the §4.0 host-function registration API (register native
   functions/constructors, marshal values through rooted handle scopes, throw and
   re-enter JS) that the runtime globals themselves are built on.

---

## 8. Reused Karpelès Lab crates

- **`purecrypto`** — `crypto.subtle`/WebCrypto, `getRandomValues`, `randomUUID`,
  TLS. No foreign crypto.
- **`rsurl`** — HTTP/HTTPS behind `fetch` and the Node `http(s)` compat layer.
- **`intl`** — Unicode/Intl primitives (normalization, case mapping/folding, char
  properties, collation) behind the Intl services (§3.3).
- **`puremp`** — multi-precision maths (bignum, …). **Now the `BigInt` backend**:
  `src/bignum.rs` is a thin wrapper (`pub struct BigInt(puremp::Int)`) over
  `puremp = { default-features = false, features = ["int"] }` (no_std + `alloc`),
  preserving the prior API so `Cell::BigInt` and all call sites are untouched.
  Truncated `div_rem`, two's-complement bit ops, `mod_2k` wrapping, and
  `write_radix` map directly.

Patterns shared: tri-modal lib/CLI/C-FFI packaging, `unsafe` quarantine,
feature-gated layered modules, sans-I/O core, cargo-fuzz harnesses.
