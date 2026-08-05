# Kataan Roadmap — the road to a complete JS + WASM engine

Kataan is a JavaScript engine written in pure Rust, usable three ways — a
standalone binary, a Rust library, and a C library. This roadmap describes **the
work that remains** to reach a fully complete, conformant, high-performance
JS + WASM engine, and a runtime that stands beside Node.js / Bun / Deno. Finished
foundations are summarized once (§1) and not re-litigated; everything after is
forward-looking.

> **Headline status (2026-08-05):** the official tc39/Test262 corpus (~53k tests)
> runs in CI gated by `tests/test262-status.txt`. Current pass-rate **≈ 99.98 %**
> — 51,879 of the 51,890 *ran* tests, **11 ledgered failures** (the ~1.5k
> skipped are host-specific or unimplemented proposals; Temporal / Atomics /
> agents / cross-realm are no longer skip-gated — see §3.9). Every subsystem
> that was previously skip-gated now runs, and the Atomics multi-agent
> scheduler is complete (§3.8).
>
> **What the last 11 are.** All but one are Intl locale *output*, and they split
> cleanly by owner. **Upstream in the `intl` crate (5):** it drops the region
> subtag when picking number symbols, so `pt-PT`/`es-MX`/`de-CH`/`en-ZA`/`it-CH`
> format like their base language — this hits plain `format()`, not just
> `formatRange` (1 test); and it ships no `de` *search* collation and no `eor`
> collation, nor any way to enumerate a locale's available collations (2 tests).
> **Ours (5):** non-Gregorian era and month names, the `-u-nu-` numbering system
> reaching DateTimeFormat's numeric fields, and `formatRange` — which does not
> use the crate's range support at all, so it misses the locale separator, the
> approximately-sign form, and shared-affix collapsing. **Deliberate (1):**
> `TypedArray/prototype/slice/speciesctor-return-same-buffer-with-offset.js`
> is an upstream harness bug; passing it would mean writing through an immutable
> buffer and breaking 47 sibling tests.
>
> The Intl *structure* — subclassing, `formatToParts` incl. unit/compact,
> `resolvedOptions`, Segmenter `containing`, `localeCompare` / `toLocaleString` /
> `Date.toLocale*` option plumbing — is done, as are **ES modules + dynamic
> `import()`** (§3.1).
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

> **Status — generic value tier + inline fast paths LANDED (2026-07-09).** The JIT
> was integer/float-numeric-only (scalar `extern "C" fn(i64…)->i64` ABI, no `&mut Realm`).
> A **generic NanBox-value tier** now compiles non-numeric hot functions: generic
> `+`/`-`/`*`/`/`/`%`/`**`/bitwise/shifts, comparisons (`<`/`==`/`===`), `Not`, control
> flow (loops + branches), property access (`GetProp`/`SetProp`), calls (`Call`/
> `CallNative`), and computed element access (`arr[i]`/`.length`).
>
> *Architecture.* A `*mut Ctx` calling convention (`ctx` pinned in r15, NanBox register
> file in rbp spill slots) lets JIT code re-enter the interpreter through runtime helpers
> for anything non-numeric; number cases run inline. Every op shares ONE `vm_*`
> implementation with the interpreter (`vm_add`/`vm_get_prop`/`vm_call`/…) so the tiers
> cannot diverge. Exceptions propagate via a `TAG_JIT_THROW` sentinel + `Ctx::jit_pending`
> (+ `jit_pending_fault`) — no deopt-and-re-run, so side effects fire exactly once.
> Scouting established the key fact: **the compacting GC is never triggered mid-execution**,
> so a re-entering JIT is GC-safe by construction today (the roadmap's "stack maps for a
> moving GC" is forward-insurance, not a current blocker); a `jit_shadow` root hook is
> wired for a future allocation-triggered GC.
>
> *Inline machine-code fast paths.* Property GET/SET and array-element GET/SET and
> `ArrayLen` emit real inline reads/writes over `#[repr(C)]`/`#[repr(u8)]` heap types
> (Slot/Cell/ObjectData/Object/PropertyCache). **Safety bedrock:** a `compute_jit_layout()`
> probe derives every offset/discriminant by pointer arithmetic on real instances (NEVER
> hand-baked) + gate tests prove them against safe reads incl. post-arena-realloc — the
> harness caught a real trap (`Vec`'s data ptr is at +8, `cap` at +0 on this toolchain).
> Guard misses + frozen/readonly/handle-value (generational write barrier)/hole/OOB route
> to the correct helper; heap-churn differentials (loops while the arena reallocs) stay
> JIT === interpreter. Direct generic→generic calls go native via an ABI-tagged registry;
> `CallNative` stays helper-by-design (it re-enters native builtin dispatch).
>
> *Verified:* ~50 differential tests (JIT-forced === interpreter, exact-once side effects,
> identical throws) + layout gate + heap-churn stress; lib 866 / jit 968 green; no_std
> compiles; JIT-on corpus green.
>
> **Remaining (the "large, separate" frontier):** generalized guard-failure **deopt is N/A
> for this non-speculative tier** (compile-time eligibility + helper slow paths mean no
> speculative guard fails mid-execution needing interpreter-reg restoration); **OSR** for
> mid-loop tier-up, and the **Cranelift-style shared backend + aarch64** portability below,
> are genuine standalone efforts.

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

- **Global-object tampering — the write-side design landed.** Reflecting the
  global object into identifier *reads* was tried and dropped: it leaked a
  tampered global into the engine's own intrinsic use and broke dynamic import.
  The direction that works is the **write** side — `globalThis.X = v` mirrors
  into the declarative global binding, so identifier reads stay on the plain
  binding path and never consult the global object. The paired guard is that
  internal machinery must hold captured intrinsics rather than live global
  bindings: `fresh_promise` now links `%Promise.prototype%` from the intrinsics
  snapshot instead of reading global `Promise`, which is exactly the coupling
  that sank the read-side attempt. `Object/{entries,values,
  getOwnPropertyDescriptors}/tamper-with-global-object` pass (1509e251). Any
  further internal use of a live global binding is the same latent bug.

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

### 3.10 Non-ISO calendar arithmetic — **landed**

Was the largest ledger cluster (~61). The calendar *identifiers* had always
resolved; what failed was arithmetic. Fixed: Persian now uses ICU's 33-year rule
rather than the 2820-year Birashk cycle; the Islamic leap term floors instead of
truncating (it lost a day for pre-epoch years); a leap `M<NN>L` constrains onto
its base month for every `NN`; `PlainDate`'s `toPlain*` conversions keep the
calendar; and the Chinese/Dangi calendars are now **computed astronomically**
(`src/nbexec/temporal_astro.rs` — solar longitude, new moons, ΔT, the
observation meridian) instead of read from a 1900-2100 table, which both
unbounds the range and matches the reference implementations across the range
the suite pins exactly.

Residual (2 tests): a Chinese leap month at 1718 comes out a day long — a
sub-day ephemeris difference ~200 years outside the pinned range, in a test
whose own comment notes ICU4X and ICU4C disagree there.

### 3.11 WTF-8 program text (lone surrogates through the parser) — **landed**

`Parser::parse_program` took `&'src str`, so a lone surrogate in program text —
`eval("/" + String.fromCharCode(0xD800) + "/")` — was folded to U+FFFD *before*
any AST node existed. That, not `Cell::RegExp` storage, is why
`RegExp.prototype.source` could not round-trip one.

The input path is now WTF-8 bytes end to end. `Lexer`/`Parser` hold
`&'src [u8]`; the scan was already byte-wise (all ECMAScript punctuation is
ASCII), so only `peek_char` (which decodes a stored surrogate to U+FFFD — same
length, and a surrogate is neither whitespace nor an `IdentifierPart`, so every
predicate answers identically) and the identifier slice needed touching. The
`&str` constructors remain as wrappers over `*_bytes` entry points, so the ~200
existing `parse_program` call sites are unchanged.

The literal payloads that can hold a surrogate carry bytes: `Expr::Regex.pattern`
is `Box<[u8]>`, `cook::string`/`decode_escapes` decode WTF-8 (so a *raw*
surrogate in a string literal or template is preserved, not just a `\uD800`
escape), and `Cell::RegExp.source` is `Box<[u8]>` — with the bytecode and
snapshot encoders carrying it as a length-prefixed blob. `eval`, direct eval and
the `Function` constructor all hand the parser the JS string's bytes.

Two `&str` boundaries remain, neither ledgered:

- The regex engine parses `&str`, so a lone surrogate in a *pattern* compiles
  (and therefore matches) as U+FFFD. `.source` still reports the exact bytes;
  only what such a pattern matches is affected.
- `Program.source` — retained only for `Function.prototype.toString` reslicing —
  is the lossy rendering, so `fn.toString()` of a surrogate-bearing body shows
  U+FFFD. It is byte-length preserving, so every span stays aligned.
`RegExp.prototype[@@replace]` was the third and is now fixed (`f3e001eb`): the
matched text, the captures, the replacer's result and all of `get_substitution`
carry WTF-8. It needed one thing the others did not — canonical *concatenation*.
Reassembling a result from per-match fragments can join a trailing high surrogate
to a leading low one, and the two halves must recombine into the 4-byte astral
form or the result is a different byte sequence for the same UTF-16 units:
unequal on comparison, two code points instead of one, not well-formed.
`Rope::concat` already did this; it is now `wtf8::append`, and **any** code that
assembles a string from fragments should use it rather than `extend_from_slice`.

### 3.12 `Array.fromAsync` as a suspendable frame — **landed**

`array_from_async_core` drove the whole iteration in a Rust loop, so the promise
`Array.fromAsync(items)` returned was already settled by the time the caller got
it, where the spec has read exactly one element. It is now the continuation chain
described below: `array_from_async_start` runs the prefix the spec's
`AsyncFunctionStart` runs synchronously — the `mapfn` check, the
`@@asyncIterator`/`@@iterator` lookups, `Construct(C)`, and the first
`IteratorStep` — and then suspends. Each `Await` resolves a promise whose
reaction is a pair of `BoundNative` continuations over the loop state, which
lives in an ordinary array so the GC traces it. `built-ins/Array` is now clear.

The alternative considered and rejected was self-hosting `fromAsync` in
JavaScript so each `await` suspends for free. It is the smaller change and it
generalises to any other builtin defined in terms of `Await`, but it needs the
intrinsics captured at realm boot to be safe against later tampering, and it
would have moved 95 passing tests onto a new code path all at once.

### 3.13 The gate

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

### 5.0 Algorithmic complexity of the core data structures

Scaling probes (time a construct at *n*, *2n*, *4n*; a ~×4-per-doubling ratio is
quadratic) found several O(n²) paths in constructs that must be linear. **All
five are now fixed** — per-iteration `let` environments, the string `+=` type
test, the global regex match/replace/split path, `Array.prototype.push`/`pop`,
and `Map`/`Set` insert (a lazily-built index from a `SameValueZero`-compatible
key hash to candidate entry indices, dropped on `Relocate`/`delete`/`clear`).

Re-measured 2026-08-05: `Map`/`Set` insert scales linearly to 200 000 (Map
50k/100k/200k = 129/164/368 ms, `Set` of strings 131/222/404 ms, against node
14/27/91 and 18/21/271). The residual is a **~4–9× constant factor**, which is
interpreter dispatch and boxing — §5.1, not a scaling bug.

**The string cluster (2026-08-05) — six more, all fixed.** A second sweep found
that essentially every string operation was O(n) per call:

| n=40 000, n calls | before | after | node |
| --- | --- | --- | --- |
| `s.length` | 3838 ms | 35 ms | ~0 |
| `s.foo` (a *miss*) | 814 ms | 36 ms | ~0 |
| `charCodeAt(i)` | 5534 ms | 46 ms | ~0 |
| `charCodeAt` on a `+=`-built string | 59044 ms | 49 ms | 1 ms |
| `slice(i, i+1)` | 3739 ms | 39 ms | 1 ms |
| `indexOf(x, i)` | 2006 ms | 31 ms | 1 ms |

The causes, in rough order of blast radius: `Realm::string_object_len`
materialized *and* rescanned the string, and is consulted on every property
access (String exotic objects own `length` and their indices); `utf16_len` was
recomputed per call rather than memoized on the rope; UTF-8's variable width
made every indexed read O(i); a `Concat` was re-walked per read; and `concat`
and template literals copied into a flat buffer instead of rope-joining.

Two general tools came out of it. `Rope::is_ascii` — `byte_len == utf16_len`,
exact and O(1) once both are memoized — turns unit indices into byte indices
for the common case. And `Realm::flatten_string` is the **flatten in place**
this section asked for, done at the *cell* level: one flat copy replaces the
tree, so it does not repeat the reverted per-node cache's O(n²) memory (peak
RSS 2.3 MB on the append-and-read-at-every-stage pattern that took 1.5 GB).

What remains here:

- **Residual `push` superlinearity** (~×3.7 after the snapshot fix) and the rest
  of the per-dispatch work in `call_method`.
- **Swept (2026-08-05).** Re-measured at 20k/40k/80k, which separated two real
  quadratics from two false positives — `arr.join` and `JSON.stringify` were
  measurement noise at the smaller sizes. Fixed: **`delete`** was a linear
  `retain` over the dictionary's insertion-order vector (9682 ms → 91 ms for 80k
  deletes), and **built-in iterator `next`** cloned its whole backing buffer
  every step (`Array.from` on an 80 000-char string, 891 ms → 46 ms).
- **Open, and deliberately not guessed at further: the string iterator's drain.**
  Draining `"x".repeat(80000)[Symbol.iterator]()` takes ~830 ms, where draining
  an `Array.from` of the very same characters takes ~144 ms. Ruled out by
  measurement, so do not re-test these: it is **not** GC (identical with
  `KATAAN_GC_THRESHOLD` effectively disabling collection), **not** heap-cell
  locality (arrays of 80k strings and 80k objects drain exactly as fast as
  arrays of numbers), and **not** `gen_iter_next` — instrumentation shows a
  string iterator's `next()` never reaches it, and `call_native` never fires for
  it either, so some other dispatch path serves it. Finding that path is the
  next step; the array iterator, which *does* go through `gen_iter_next`, is
  linear.
- **Array methods copy the whole array on every call — the biggest one left.**
  What looked like slow *closure capture* was not closures at all: the `push`
  in the probe was the cost. `Realm::elements_vec` does `a.to_vec()`, and every
  array method except `push`/`pop` (which have a fast path above it) goes
  through it, so an O(1) call is O(n). On an 80 000-element array, 80 000 calls:

  | | kataan | node |
  | --- | --- | --- |
  | `a.at(0)` | 3880 ms | ~0 |
  | `a.indexOf(0)` | 5573 ms | ~0 |
  | `a.push(i)` (growing) | 1043 ms | 1 ms |
  | plain function call | 52 ms | — |
  | `a[i] = i` | 27 ms | — |

  Indexed assignment and plain calls are linear, so this is specific to the
  array-method dispatch. The fix is per-method: `at` / `indexOf` /
  `lastIndexOf` / `includes` need only `array_length` + `get_element`, never a
  snapshot; methods that genuinely iterate (`map`, `sort`, …) can keep it. Note
  the `precise_array` hole scan (`elems.iter().any(is_hole)`) is a *second*
  O(n) per call and has to be addressed with it.

  **`push` is superlinear for a different, unlocated reason**: it returns from
  its own fast path well before `elements_vec`, and `Realm::array_push` is
  amortized O(1). Something else in the dispatch is O(n) for an array receiver.
- **Inherently quadratic, not bugs**: `arr.unshift` and `arr.indexOf` in a loop
  are O(n) per operation by construction (node's `indexOf` ratio is 17.8 too).
- **Rope reads.** `charCodeAt` on a `+=`-built string re-walks and re-copies the
  whole tree per call. **A plain memo on the `Concat` node is the wrong fix and
  was tried and reverted:** it makes repeated indexing ~7× faster but costs
  O(n²) *memory*, because a rope read at every growth stage caches a full copy
  at each level and the old roots stay alive as left children — 1.5 GB peak RSS
  for a 128 KB string, against 17.6 MB before. The fix has to **flatten in
  place**, replacing the node's children with the flattened bytes so the
  intermediate nodes are released (which also makes that pattern linear in
  time). That means putting the children behind the same interior-mutability
  cell as the cache, so `as_leaf_bytes`/`last_leaf_bytes`/`first_leaf_bytes`
  must stop handing out borrows into the tree first.

### 5.1 Long-standing performance work

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

### 5.2 Regex throughput — **the algorithmic gaps are closed**

Four O(subject) costs on matches that should be O(1) are fixed: the scan retried
every offset for a `^`-anchored pattern (ad422b64), the subject was transcoded to
`Vec<u16>` per call in both tiers (954252dc), offsets that cannot start a match
were not skipped (ea252b82), and every attempt allocated three working vectors
(563db978). Against node on a 10 MB subject:

| | node | before | now |
| --- | --- | --- | --- |
| 4000 anchored `test` | 3 ms | >120 s | 19 ms |
| 20 unanchored scans | 37 ms | 3306 ms | 37 ms |

What remains is **constant-factor**, not algorithmic — `split`, global `replace`
and an `exec` loop are all linear in subject length, just 12–78× slower per unit
of work (600 KB: split 7 ms vs 565 ms, replace 16 ms vs 201 ms, exec loop 5 ms vs
185 ms). That is match-object construction, string allocation and interpreter
dispatch, so it belongs to §5.1 and the JIT rather than here.

**The start filter now covers classes** (`StartSet`, a 256-bit Latin-1 map
unioned over the first-consuming instruction on every path, with membership
taken from the matcher's own `single_consume_matches` so the two cannot
disagree). On a 1 MB subject with no match, 200 scans: `/[0-9]+/` 2393 ms → 74
ms, `/\s+/` 2455 ms → 73 ms, `/[xyz]q/` 2475 ms → 72 ms, against node's 78/70/36
— parity on the first two.

It is worth being precise about what that did *not* fix, because the obvious
benchmark misleads: `"word ".repeat(120_000).split(/\s+/)` is unchanged at ~2.3 s
against node's 27 ms. There a match starts every fifth offset, so the filter can
only skip 4 in 5, and the cost is ~19 µs *per match* — match-object
construction, capture allocation and interpreter dispatch. The start filter
helps sparse matches; dense-match throughput is §5.1.

One contained piece of matcher headroom is left:

- **Widen the filter to a literal prefix.** Matching `abc` at an offset whose
  next two units cannot continue it still starts the matcher; a short memcmp
  first would cut that.

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
