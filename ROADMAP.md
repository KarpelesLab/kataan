# Kataan Roadmap — the road to a complete JS + WASM engine

Kataan is a JavaScript engine written in pure Rust, usable three ways — a
standalone binary, a Rust library, and a C library. This roadmap describes **the
work that remains** to reach a fully complete, conformant, high-performance
JS+WASM engine. Finished foundations are summarized once (below) and not
re-litigated; everything after §1 is forward-looking.

---

## 0. Non-negotiables (unchanged)

- **Pure Rust, no foreign code on the critical path.** Crypto/TLS via
  [`purecrypto`](https://github.com/KarpelesLab/purecrypto), HTTP via
  [`rsurl`](https://github.com/KarpelesLab/rsurl); regex, Intl-lite collation, GC,
  and the WASM engine are all in-house.
- **`unsafe` is quarantined.** `unsafe_code = "deny"` (not `forbid`); only the
  `ffi` module and a small, audited set of VM/JIT/mmap hot-path primitives opt
  back in with a scoped `#[allow(unsafe_code)]` + safety comment.
- **Specification fidelity.** Conformance is measured against Test262 (JS) and
  the upstream WebAssembly spec suite (WASM); correctness is never knowingly
  traded for speed without a flag.
- **Deployable, host-native bytecode + heap.** Compiled bytecode — and an
  initialized heap — are first-class serializable artifacts: exportable,
  content-addressed, and `mmap`-reloadable zero-copy on a matching host, with an
  on-demand byte-swap conversion for a mismatched one (never a re-encode tax on
  the common path). See §6.

---

## 1. Where we are now (the finished base)

Treat this as done; build on it.

- **Front end:** complete lexer + full ECMAScript parser + AST.
- **Two execution engines that agree on every test:** a tree-walking interpreter
  (`nbexec`, the corpus/default engine) and a **register bytecode VM** (`nbvm`,
  the primary path for `kataan run` / `eval` / the C ABI), compiling nearly all of
  the common language. A dual-path conformance corpus (**510/510**, a curated
  Test262-style suite) runs on both.
- **Object model in production:** NaN-boxed values, hidden classes/shapes +
  transition tree, shape+slots objects, interned atoms, rope strings, inline-cache
  slots, a generational handle table, and a tracing GC — all live behind `Realm`,
  not a side experiment.
- **Bytecode codec (`KTBC`)** with a host-native header (byte order + pointer
  width), dedup constant pool, version/magic/truncation checks; survives
  serialize → reload → run.
- **D′ snapshot tier:** a verified codec that captures the live heap and
  `mmap`-reloads it over the **moving/compacting GC**, across eleven reference
  cell kinds (string, array, object, date, bigint, closure, collection, promise,
  proxy, regexp, symbol), cross-kind cycles, insertion-order-preserving.
- **Machine-code JIT (x86-64 / Linux, `jit`):** W^X executable memory via raw
  syscalls; an **optimizing integer path** (four-pass optimizer + register
  allocator) and a **float path** covering `+ - * / %`, comparisons, control flow,
  and the SSE-expressible `Math` intrinsics; each op has an `eval_*` oracle for
  differential testing.
- **WASM engine (`no_std`):** binary decode + validate + stack interpreter for
  the MVP plus sign-extension, saturating conversion, bulk-memory, multi-value,
  and typed structured control; a `.wast`/WAT spec harness drives a spec-derived
  corpus; the `WebAssembly` builtin exists.
- **Stdlib breadth:** Object/Array/String/Number/Math/JSON, Map/Set/WeakMap,
  Symbol, BigInt, Promise + (eager) async, Proxy/Reflect, partial typed arrays,
  Date, RegExp, `Function`, `new.target`. C ABI `kt_eval` runs scripts end-to-end.

The rest of this document is the gap between that base and "complete."

---

## 2. The three headline deliverables

These are the load-bearing items that make Kataan more than a fast interpreter.

### 2.1 Complete the machine-code JIT (for the *whole* VM, not just numbers)

Today the JIT compiles integer/float numeric functions and bails to the
interpreter for everything else. "Complete" means hot functions JIT regardless of
what they touch.

- **The re-entrancy + GC-safety substrate (the hard core).** Native code must be
  able to call back into `Realm` primitives (property get/set, allocation, string
  ops) and survive a moving GC mid-call. Concretely:
  - a calling convention and **stack maps** so the GC can find/relocate live
    object references held in JIT frames and registers across a safepoint;
  - safepoints at allocation/call sites; pinning or re-loading of relocatable
    handles after any call that can collect;
  - a sound `&mut Realm` re-entry path (no aliasing UB when native code re-enters
    the VM) — the current blocker that keeps object/string ops interpreter-only.
- **Non-numeric op lowering:** property access through inline-cache slots (shape
  guard → slot load, deopt on miss), array element load/store with the
  element-kind fast paths, string/rope ops, closure/upvalue access, `this`/scope.
- **Calls in JITed code:** direct + polymorphic call sites, native-builtin calls,
  argument/return marshaling, exception propagation across native frames.
- **Tiering & deopt:** a **baseline tier** (fast copy-and-patch / template
  compile of bytecode, no optimization, with on-stack replacement for hot loops)
  feeding a **profiling** layer; **deoptimization** back to bytecode when a
  speculative guard fails (the mechanism that makes speculative type
  specialization safe).
- **Shared native backend:** factor codegen into a Cranelift-style backend
  (register allocation, executable-memory mgmt, relocations) that both the JS JIT
  and the WASM compiler (Track 2.2) lower into. Designed to target **native or
  WASM output** (the sandbox fallback — JIT-by-emitting-WASM when hosted inside a
  WASM sandbox that forbids native codegen).
- **Portability of the JIT itself:** the current path is x86-64/Linux; the
  backend abstraction is what makes aarch64 and other OSes additive.

Exit criteria: a hot function doing object/array/string work runs as native code,
verified by differential execution against the interpreter and by the conformance
corpus passing with the JIT forced on; deopt round-trips proven; no GC-safety
holes under stress/fuzz.

### 2.2 A conformant (non-numeric) WASM engine + full upstream suite

Today the WASM engine passes a spec-*derived* corpus over the numeric/control
core. "Complete" means the full standardized feature set and the **upstream
WebAssembly spec test suite**.

- **Reference types & tables:** `funcref`/`externref`, `table.*` ops, multiple
  tables, `call_indirect` type checks and traps, element segments (active /
  passive / declarative), `ref.func`/`ref.null`/`ref.is_null`.
- **The JS ↔ WASM boundary (the "non-numeric" surface):** *Done:* `validate`,
  `instantiate` (→ `{module, instance}`), the `Module`/`Instance` constructors,
  `compile` (→ `Promise<Module>`), host-function imports (JS callable from WASM),
  **stateful instances** — mutable globals and linear memory persist across export
  calls, each instance independent — and the **`WebAssembly.Global`** object (typed
  value cell with a `.value` accessor, ToInt32/`BigInt` coercion, immutability
  enforced). *Remaining:* the `Memory`/`Table` objects (and `.buffer` ↔ typed-array
  views), `compileStreaming`/`instantiateStreaming`, the `externref` bridge to JS
  values, and imported memories/tables/globals. Traps already surface as JS
  exceptions.
- **Post-MVP proposals (prioritized):** SIMD (`v128`), threads + `Atomics` on
  shared memory, multi-memory, tail calls, extended-const, then GC types and
  exception handling as they stabilize.
- **Compilation, not just interpretation:** lower validated WASM through the
  shared native backend (Track 2.1) for a real baseline/optimizing WASM tier, on
  the **shared GC and heap** (linear memory as a GC-tracked byte array;
  future WASM-GC objects on the same heap).
- **The test gate:** wire the **official `wabt`/spec `.wast` suite** into CI (not
  only the in-house corpus), tracking a pass-rate per proposal; fuzz the decoder
  and validator.

Exit criteria: the upstream spec suite passes for MVP + reference-types +
bulk-memory + multi-value + sign-ext + sat-conversion (and a tracked, growing set
of post-MVP proposals); JS↔WASM interop round-trips real modules; validator
rejects malformed modules safely under fuzzing.

### 2.3 The mmap-able zero-copy D′ layout, fully complete

Today the snapshot codec round-trips eleven cell kinds and `mmap`-reloads over the
moving GC. "Complete" means *any* live heap snapshots, reloads zero-copy, and
**executes** — and the bytecode code-cache it shares is production-grade.

- **Hidden-state cells.** *Done:* object capture now records **all own data
  properties** — non-enumerable ones and the engine's internal `\0`-prefixed slots
  — with a per-property hidden flag, so bound-function target/this/args, typed-array
  kind, error `stack`, and the `constructor` back-reference round-trip
  (`snapshots_preserve_non_enumerable_and_hidden_slots`), an object's
  `[[Prototype]]` link round-trips so `Object.create(p)` / inheritance chains
  resolve through the restored chain (`snapshots_preserve_prototype_links`), and
  accessor (getter/setter) properties round-trip with their functions and
  enumerability (`snapshots_preserve_accessor_properties`) — so object-cell fidelity
  is now complete (data + non-enumerable + internal slots + accessors + prototype).
  *Remaining:* generator/async suspension state (depends on Track 3 lazy frames);
  the `ArrayBuffer` backing-buffer *identity* shared across typed-array views —
  audit each remaining non-object `Cell` variant.
- **End-to-end restore-and-execute.** *In place:* a restored closure now runs and
  carries its snapshotted captured state, independent of the original
  (`snapshot_restores_an_executable_closure`) — restored state is executable, not
  just structurally equal. *Remaining:* a supported public/C API that binds a
  restored realm + the matching function table into a **fresh runtime** and
  invokes a value (the cross-runtime reload path, vs the same-runtime test today).
- **The shared, versioned, mmap-able artifact store (code-cache, §6):**
  content-addressed by source hash + host tag; lazy per-function bodies faulted in
  on first call; **module-local atom remap** on load; IC slots load reset; the
  **on-demand byte-swap conversion** path for a mismatched host (convert once,
  re-cache, zero-copy thereafter); **read-only pages shareable across many
  concurrent contexts/processes** (the "hundreds of bases" model — immutable
  bytecode shared, each context owns its mutable heap/globals).
- **Untrusted-load verifier.** Loading bytecode *is* executing it: a verifier
  (bounds-checked jumps, constant/atom/register indices, stack-depth invariants)
  for untrusted artifacts; trusted-cache fast path relies on version tag +
  checksum. Safe-Rust means a bad index is a clean rejection, never UB.
- **Heap-snapshot startup.** Skip initialization (not just parsing) by booting
  from a snapshot of an initialized realm — the natural payoff once the above and
  the moving GC's pointer-relocation pass are solid.

Exit criteria: an arbitrary initialized heap snapshots, reloads zero-copy on a
matching host (and via one-time conversion on a mismatched one), and executes;
the code-cache passes the load → run → evict → reload churn + cross-tenant dedup
scenario; the verifier rejects malformed blobs under fuzzing.

---

## 3. Language conformance to "fully complete"

The corpus is curated (510 tests); completeness means the **full upstream
Test262** for the language (non-Intl) at a high, tracked pass-rate, with the known
semantic gaps closed.

- **Sparse arrays / holes.** The array model stores holes as `undefined`; a real
  empty-slot representation (sentinel or presence bitmap) must thread through
  literal construction, every iteration method, `in`/`hasOwnProperty`,
  `Object.keys`/`entries`, and `join`/`toString`. (Currently `Object.keys` on a
  hole array and `forEach`/`in` over holes are non-conformant.)
- **Lazy, suspendable generators.** Generators run *eagerly* into a value buffer:
  `.next(value)` injection and `.throw()` into a `try{ yield }catch` are
  unsupported, and the suspension model can't interleave. Replace with a real
  coroutine/state-machine so `yield` truly suspends.
- **Lazy async / correct microtask interleaving.** `await` runs its continuation
  synchronously instead of yielding to the caller as a microtask (values are
  correct; ordering is not). Same suspendable-frame work as generators.
- **First-class prototype methods.** `Array.prototype.map.call(arrayLike)` and
  extracting a method as a value (`[].slice`) don't work — methods are built-in
  dispatch, not function objects on real `…​.prototype` objects. `Array`/`Object`
  are namespace objects, so `typeof Array === "object"` (should be `"function"`)
  and `Array instanceof Function` is false. Needs real builtin prototype objects
  with installed, callable method functions and `this`-generic algorithms.
- **Remaining builtins / edges:** `Atomics` (+ `wait`/`notify`),
  `SharedArrayBuffer`, resizable/growable `ArrayBuffer`, complete `%TypedArray%`
  method set + `TypedArray.from`/`subarray`, `WeakRef`/`FinalizationRegistry` GC
  cooperation, `Symbol.toPrimitive`/well-known-symbol coverage, tagged-template
  and regex `d`/`v`-flag/named-group/lookaround completeness, JSON source-text
  access.
- **Intl-lite (`intl`):** flesh out `Intl.Collator`/`NumberFormat`/`DateTimeFormat`
  /`PluralRules`/`Segmenter` + locale negotiation over the embedded trimmed CLDR
  data.
- **The gate:** run the **full upstream Test262** in CI with a tracked pass-rate
  (target >95% of the non-Intl suite); fuzz parser, regex, JSON, and the VM.

---

## 4. Host runtime to a real runtime (`host`)

- **Event loop & timers:** a complete in-house loop (mio-style readiness or std
  threads), `setTimeout`/`setInterval`/`setImmediate` + `clear*`,
  `process.nextTick`, microtask checkpoint integration.
- **Modules:** ESM (static + dynamic `import`, `import.meta`, top-level `await`),
  CommonJS interop, JSON modules, import maps / resolution.
- **Web platform:** `TextEncoder`/`Decoder`, `atob`/`btoa`, `Buffer`,
  `URL`/`URLSearchParams`, WHATWG streams, `structuredClone`,
  `performance.now`/marks.
- **`fetch`** (+ `Headers`/`Request`/`Response`/`Blob`/`FormData`) over `rsurl`
  with `purecrypto` TLS; **`crypto`** (`getRandomValues`/`randomUUID`/`subtle`)
  over `purecrypto`.
- **Node-compat subset:** `fs`/`path`/`os`/`net`/`http(s)`/`events`/`stream`/
  `util`/`process`/`Buffer` — a useful subset, gaps documented.

---

## 5. Performance frontier

- **Interpreter pass:** IC tuning, array element-kind fast paths, string-rope
  tuning, the generational/moving GC upgrades that bump-allocation `new` relies
  on (the moving GC exists for D′; finish its generational nursery + write
  barriers for throughput).
- **Optimizing JIT (after the baseline tier in 2.1):** an SSA IR with inlining,
  escape analysis, range/redundancy elimination, type-feedback-driven speculation
  with guard-based deopt — lowering through the shared backend. The point where we
  contend with V8 on compute-bound code.
- **Benchmarks:** SunSpider/Kraken-style microbenchmarks + realistic scripts;
  cold-start vs `node -e`; code-cache load/evict/reload throughput; per-object
  memory vs equivalent V8 heaps.

---

## 6. Design invariants that still constrain remaining work

Kept from the original architecture because Tracks 2.1–2.3 must honor them.

- **Own JS bytecode + VM; WASM is a *peer* engine sharing the backend, not a
  compile target.** Routing JS through WASM would forfeit guard-based deopt — the
  mechanism that makes a JS optimizing JIT fast. The two engines share the **GC +
  heap, the native backend, the value/interop boundary, and the host runtime**;
  the two bytecodes stay distinct (JS = dynamic/deopt-friendly, WASM = statically
  typed). The one place emitting WASM *for JS* is correct: the sandbox fallback in
  2.1.
- **Serializable, host-native artifacts (the code-cache + heap snapshot).** A
  serialized unit is position-independent (cross-referenced by index, never by
  live pointer), with host-native integer encoding and an explicit, cheap,
  on-demand byte-swap conversion for a mismatched host — *not* a slow canonical
  encoding everyone pays for, and *not* a recompile. Versioned + integrity-checked
  with two distinct mismatch paths: version/flags mismatch or bad checksum →
  recompile; host-encoding mismatch, same version → convert. Atoms are
  module-local and remapped on load; IC slots and shapes are runtime state, never
  serialized; function bodies are lazy. The JS bytecode cache and the compiled-WASM
  cache share **one** artifact store.

---

## 7. Definition of done

Kataan is "complete" when:

1. **JS:** the full upstream Test262 (non-Intl) passes at a high tracked rate on
   both engines; the documented semantic gaps (holes, lazy generators/async,
   first-class prototypes) are closed; `Atomics`/`SharedArrayBuffer`/typed-array
   completeness and the host runtime land.
2. **JIT:** hot functions JIT regardless of object/string/closure content, with a
   baseline + optimizing tier, type-feedback speculation, sound deopt, and a
   GC-safe native re-entry substrate — proven against the interpreter and under
   fuzz.
3. **WASM:** the upstream WebAssembly spec suite passes for MVP + reference-types
   + the implemented proposals, with full JS↔WASM interop and a compiling tier on
   the shared backend.
4. **D′ / code-cache:** any initialized heap snapshots, `mmap`-reloads zero-copy
   (or via one-time conversion), and executes; the content-addressed,
   verifier-guarded, cross-context-shareable artifact store passes the
   hundreds-of-bases churn scenario.
5. **Always:** pure Rust, `unsafe` quarantined and audited; CI green across the
   feature matrix; embeddable in <30 lines via the Rust and C APIs.

---

## 8. Reused Karpelès Lab crates

- **`purecrypto`** — `crypto.subtle`/WebCrypto, `getRandomValues`, `randomUUID`,
  TLS. No foreign crypto.
- **`rsurl`** — HTTP/HTTPS behind `fetch` and the Node `http(s)` compat layer.

Patterns shared: tri-modal lib/CLI/C-FFI packaging, `unsafe` quarantine,
feature-gated layered modules, sans-I/O core, cargo-fuzz harnesses.
