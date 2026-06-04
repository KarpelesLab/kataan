# Kataan Roadmap

Kataan is a JavaScript engine written in pure Rust. The goal is a modern,
high-performance ECMAScript engine that is competitive with V8/Node.js for
real workloads, while being usable three ways — as a standalone binary, as a
Rust library, and as a C library (the same tri-modal model proven out in
sibling projects [`purecrypto`](https://github.com/KarpelesLab/purecrypto) and
[`rsurl`](https://github.com/KarpelesLab/rsurl)).

The non-negotiables:

- **Pure Rust, no foreign code on the critical path.** We reuse our own
  pure-Rust crates — `purecrypto` for the `crypto`/WebCrypto surface and TLS,
  `rsurl` for `fetch`/HTTP — rather than binding C libraries. Regex, ICU-lite
  collation, and the GC are all in-house.
- **`unsafe` is quarantined.** The crate is `unsafe_code = "deny"`, not
  `forbid`: only the `ffi` module and a small, audited set of VM hot-path
  primitives (NaN-boxing transmutes, GC interior access) may opt back in with a
  scoped `#[allow(unsafe_code)]` and a safety comment. Everything else is safe
  Rust.
- **Specification fidelity.** Conformance is measured against Test262. We track
  a conformance percentage per milestone and never knowingly trade correctness
  for speed without a flag.

---

## 1. Why another engine, and the performance thesis

The engines that win (V8, JavaScriptCore, SpiderMonkey) all share the same
handful of ideas. None of them are secret; the difficulty is in doing all of
them well, together, and keeping them honest under a conformance suite. Kataan
commits to the full set from the architecture stage so we never have to
retrofit them:

1. **Compact value representation (NaN-boxing).** Every JS value lives in 64
   bits. IEE-754 doubles are stored as-is; every other type (the small-integer
   fast path, `undefined`/`null`/booleans, and pointers to heap objects) is
   encoded in the ~52 bits of NaN payload space. This makes `Value` `Copy`,
   keeps the stack dense, and turns the common "is this a number?" check into a
   single compare.

2. **Hidden classes / shapes + inline caches.** Objects do not carry a hash map
   per instance. Instead each object points to a *shape* (a.k.a. hidden class /
   map) describing its property layout; properties live in a flat slot vector.
   Shapes form a transition tree, so objects built the same way share a shape.
   Property access sites in bytecode carry **inline caches** keyed on shape:
   the second time a site sees a known shape, the lookup is a slot-index load,
   not a hash probe. This is the single largest lever for real-world JS speed.

3. **Register-based bytecode VM.** We compile the AST to a register-based
   bytecode (in the spirit of Ignition/Lua), not a tree-walker. Register VMs
   issue fewer instructions than stack VMs for the same work and are friendlier
   to a future JIT. Dispatch uses a tail-call / computed-goto-style loop where
   the platform allows, with a portable `match` fallback.

4. **Interned strings (atoms) + rope/slice strings.** Identifiers and property
   keys are interned to small integer atoms for O(1) comparison and as inline-
   cache keys. String *values* use a small-string-optimized, ref-counted
   representation with lazy concatenation (ropes) to make `+=` in a loop not
   quadratic.

5. **A precise, generational, moving GC.** Start with a simple precise
   mark-sweep so the object model and rooting discipline are correct, then
   evolve to a generational semi-space young generation (bump allocation +
   copying) with a mark-compact old generation. Bump allocation makes `new`
   nearly free, which is what JS programs actually stress.

6. **Tiered execution.** Ship the interpreter first and make it genuinely fast
   (the above five items). Then add a **baseline JIT** (template/copy-and-patch
   compilation of bytecode to machine code, no optimization) for hot functions,
   and later an **optimizing JIT** driven by type feedback collected by the
   inline caches. The bytecode and IC formats are designed up front to be
   JIT-consumable so tiering is additive, not a rewrite.

The ordering matters: items 1–5 are interpreter-era and deliver most of the
"feels fast" experience; item 6 is what closes the gap to V8 on
compute-bound benchmarks. We do not start on the JIT until the interpreter is
conformant and the IC/feedback plumbing exists for it to consume.

### Sans-I/O core, host on top

Like `purecrypto`'s TLS engine, the language core is **sans-I/O**: the parser,
compiler, VM, and GC know nothing about files, sockets, or clocks. The event
loop, timers, file system, and network live in a separate `host` layer that
drives the core. This keeps the engine embeddable (a Rust program supplies its
own host bindings) and keeps `no_std`-with-`alloc` builds of the pure language
core possible.

---

## 2. Crate architecture

```
kataan/
├── src/
│   ├── lib.rs            crate root, feature-gated module tree
│   ├── common/           spans, interner/atoms, diagnostics, arena
│   ├── lexer/            source → tokens (UTF-8/UTF-16 aware, ASI hints)
│   ├── parser/           tokens → AST (ESTree-ish), error recovery
│   ├── ast/              AST node definitions + visitors
│   ├── compiler/         AST → bytecode, scope/closure analysis, IC slots
│   ├── bytecode/         opcode definitions, encoder/decoder, disassembler
│   ├── value/            NaN-boxed Value, number/string/bigint helpers
│   ├── gc/               heap, allocator, collector, handles/roots
│   ├── object/           shapes (hidden classes), property storage, arrays
│   ├── vm/               interpreter loop, call frames, stack, exceptions
│   ├── builtins/         the ECMAScript standard library (see §4)
│   ├── intl/             in-house Intl-lite (collation, casing, number fmt)
│   ├── regex/            in-house regex engine (no foreign code)
│   ├── module/           ESM + CommonJS loader/linker, import resolution
│   ├── host/             event loop, timers, fs, net, fetch, console, process
│   ├── jit/              (later) baseline + optimizing tiers
│   ├── ffi/              C ABI (only place broad `unsafe` is allowed)
│   └── bin/kataan/       the CLI / REPL / script runner
├── include/kataan.h      generated C header
├── testdata/             Test262 harness, fixtures
└── ROADMAP.md
```

Feature gates (Cargo): `std` (default, implies `alloc`), `alloc`, `regex`,
`intl`, `module`, `host`, `crypto` (→ purecrypto), `fetch` (→ rsurl), `jit`,
`ffi`, `cli`. The language core (lexer→vm + core builtins) builds with just
`alloc`.

---

## 3. Execution pipeline

```
source ──lexer──▶ tokens ──parser──▶ AST ──compiler──▶ bytecode + IC slots
                                                              │
                                                              ▼
                                       interpreter ◀── shapes/ICs ◀── GC heap
                                                              │
                                                  (hot) baseline JIT
                                                              │
                                              (very hot) optimizing JIT
```

- **Lexer**: streaming tokenizer that handles the regex/division ambiguity,
  template literals, Unicode identifiers, numeric separators, BigInt literals,
  and records the newline-before flags the parser needs for Automatic
  Semicolon Insertion.
- **Parser**: a hand-written recursive-descent + Pratt expression parser
  (predictable performance, good error messages, no parser-generator dep). It
  produces an AST and a separate binding/scope table. Handles strict mode,
  modules vs scripts, cover-grammar cases (arrow vs parenthesized expr,
  async), and destructuring.
- **Compiler**: lowers the AST to register bytecode in one or two passes;
  resolves variables to register/upvalue/global slots; classifies closures and
  emits `Environment` capture; reserves inline-cache slots for every property
  access, call, and binary op; performs trivially-safe constant folding.
- **Interpreter**: the register VM. Owns the value stack, call frames, the
  exception/`try`-`finally` unwinder, generators/async suspension (frames are
  heap-relocatable to support `yield`/`await`), and the microtask checkpoint.

---

## 4. Builtins inventory

This is the surface we must implement. Grouped by milestone phase (see §6).
Each line becomes one or more `builtins/` modules with Test262-driven tests.

### 4.1 Core language intrinsics (Phase B–C)

- **Global object / globalThis**: `globalThis`, `undefined`, `NaN`,
  `Infinity`, `eval`, `isNaN`, `isFinite`, `parseInt`, `parseFloat`,
  `encodeURI`, `decodeURI`, `encodeURIComponent`, `decodeURIComponent`.
- **Object**: constructor + `Object.*` statics (`keys`, `values`, `entries`,
  `assign`, `create`, `defineProperty`/`-ies`,
  `getOwnPropertyDescriptor`/`-s`, `getOwnPropertyNames`, `getOwnPropertySymbols`,
  `getPrototypeOf`/`setPrototypeOf`, `freeze`/`isFrozen`,
  `seal`/`isSealed`, `preventExtensions`/`isExtensible`, `is`, `fromEntries`,
  `hasOwn`, `groupBy`), and `Object.prototype.*`.
- **Function**: `call`/`apply`/`bind`, `length`/`name`, `Function.prototype`,
  the (sandbox-gated) `Function` constructor.
- **Boolean**, **Symbol** (incl. well-known + registry `Symbol.for`/`keyFor`),
  **Error** + native subclasses (`TypeError`, `RangeError`, `SyntaxError`,
  `ReferenceError`, `EvalError`, `URIError`, `AggregateError`), with stack
  capture.
- **Number** (`isInteger`, `isSafeInteger`, `parseFloat`/`parseInt`,
  `toFixed`/`toPrecision`/`toExponential`, `MAX_SAFE_INTEGER`, …),
  **BigInt** (incl. `asIntN`/`asUintN`), **Math** (full function set, correctly
  rounded where the spec requires).

### 4.2 Strings, regex, collections (Phase C–D)

- **String** + `String.prototype.*` (full method set incl. `matchAll`,
  `replaceAll`, `normalize`, `localeCompare`, `at`, well-formed Unicode
  methods), `String.raw`, template support.
- **RegExp**: in-house engine (`regex/`) — Unicode mode, named groups,
  lookbehind, the `d`/`v` flags, sticky/global state, `Symbol.replace` et al.
- **Array** + `Array.prototype.*` (full set incl. `flat`/`flatMap`,
  `at`, `findLast`/`findLastIndex`, the `toSorted`/`toReversed`/`toSpliced`/
  `with` copying methods, `group`*), `Array.from`/`of`/`isArray`. Fast packed
  (dense SMI / double / object) element kinds with a dictionary fallback.
- **Typed arrays**: `ArrayBuffer`, `SharedArrayBuffer`, `DataView`, all
  `%TypedArray%` views, resizable/growable buffers, `Atomics`.
- **Keyed collections**: `Map`, `Set`, `WeakMap`, `WeakSet`, `WeakRef`,
  `FinalizationRegistry` (the last three need GC cooperation).

### 4.3 Control, metaprogramming, data (Phase D–E)

- **Iterators/Generators**: the iteration protocol, `%IteratorPrototype%`,
  generator & async-generator objects, the `Iterator` helpers
  (`map`/`filter`/`take`/`drop`/`flatMap`/`reduce`/`toArray`/…).
- **Promise**: `Promise` + `all`/`allSettled`/`any`/`race`/`resolve`/`reject`/
  `withResolvers`, async/await, the job/microtask queue.
- **Proxy** & **Reflect**: all 13 traps and the matching `Reflect.*` operations.
- **JSON**: `parse`/`stringify` (with reviver/replacer, `BigInt` errors,
  the source-text access proposal).
- **Date**: full `Date` with a correct (in-house, no libc) calendar and an IANA
  time-zone database for `Intl`-aware behavior.
- **Atomics**: full operation set incl. `wait`/`notify` for the agent model.

### 4.4 Intl-lite (Phase E, `intl` feature)

A pragmatic subset, in pure Rust, sufficient for common apps: `Intl.Collator`,
`Intl.NumberFormat`, `Intl.DateTimeFormat`, `Intl.PluralRules`,
`Intl.Segmenter`, locale negotiation. Backed by an embedded, trimmed CLDR/ICU
data table rather than a C ICU dependency.

### 4.5 Host runtime (Phase F, `host` feature — Node-compatible-ish)

The "batteries" that make Kataan a usable runtime rather than just a language:

- **Console & diagnostics**: `console.*`, `performance.now`/marks,
  `queueMicrotask`, `structuredClone`.
- **Timers & event loop**: `setTimeout`/`setInterval`/`setImmediate` +
  `clear*`, `process.nextTick`, the libuv-equivalent loop (in-house, built on
  mio-style readiness or std threads).
- **Encoding**: `TextEncoder`/`TextDecoder`, `atob`/`btoa`, `Buffer`.
- **URL**: `URL`/`URLSearchParams` (WHATWG).
- **Streams**: WHATWG `ReadableStream`/`WritableStream`/`TransformStream`.
- **fetch**: `fetch`, `Headers`, `Request`, `Response`, `Blob`, `FormData` —
  implemented over **`rsurl`** for the transport and **`purecrypto`** for TLS.
- **crypto**: `crypto.getRandomValues`, `crypto.randomUUID`, and a WebCrypto
  `crypto.subtle` surface mapped onto **`purecrypto`**.
- **Node-style modules** (optional compat layer): `fs`, `path`, `os`, `net`,
  `http`/`https` (over rsurl), `events` (`EventEmitter`), `stream`, `util`,
  `process`, `Buffer`. Scope to a useful subset; document gaps explicitly.
- **Module system**: ESM (static + dynamic `import`, `import.meta`, top-level
  `await`), CommonJS (`require`/`module.exports`) interop, JSON modules,
  import maps / resolution.

---

## 5. C ABI (`ffi` feature)

Mirrors the `purecrypto` model: opaque handles created/freed by the library,
`KtStatus` return codes (`0` = ok, negative = error), the in/out length
convention for variable-length output, and a panic catch at every boundary so a
Rust panic becomes `KtStatus::Internal` instead of unwinding into C. Surface
(initial): create/destroy a `KtRuntime` and `KtContext`, evaluate a script,
marshal values across the boundary (`KtValue`), define native callbacks, drive
the microtask/event loop, and read exceptions. Header generated to
`include/kataan.h`.

---

## 6. Milestones

Each milestone ends with: it builds clean under `cargo clippy`, has unit tests,
and (from D onward) reports a Test262 pass-rate number.

- **Phase A — Scaffold & lexer** *(current)*
  Crate layout, CI, conventions; a complete, tested tokenizer; CLI that can
  `--tokens` a file. *Deliverable: `kataan lex file.js`.*

- **Phase B — Parser & AST**
  Full ECMAScript expression + statement grammar, modules vs scripts, strict
  mode, ASI, destructuring, classes, async/generators at the syntax level.
  `kataan parse --json file.js`. Round-tripping & error-recovery tests.

- **Phase C — Tree-walk MVP → object model**
  A correct (not yet fast) evaluator to validate semantics, *then* introduce
  NaN-boxed `Value`, shapes, the GC, and core intrinsics (Object/Function/
  Array/Number/String/Boolean/Symbol/Error/Math). First Test262 numbers.

- **Phase D — Bytecode VM**
  AST→register-bytecode compiler, the interpreter loop, inline caches,
  closures/upvalues, exceptions/`try-finally`, generators/iterators, RegExp,
  collections, typed arrays, Promise + microtasks, Proxy/Reflect, JSON, Date.
  Target: broad Test262 coverage of the language proper.

- **Phase E — Conformance & Intl**
  Close Test262 gaps, BigInt edge cases, Atomics, Intl-lite, `WeakRef`/
  `FinalizationRegistry`. Performance pass on the interpreter (IC tuning,
  string ropes, array element-kinds, GC generational upgrade).

- **Phase F — Host runtime**
  Event loop, timers, modules (ESM+CJS), console, encoding, URL, streams,
  `fetch` over rsurl, `crypto` over purecrypto, Node-compat subset. `kataan
  run app.mjs` as a real runtime.

- **Phase G — Baseline JIT**
  Copy-and-patch / template JIT of hot bytecode, consuming IC feedback. On-stack
  replacement for hot loops. Measured against the interpreter and Node on
  microbenchmarks.

- **Phase H — Optimizing JIT**
  Type-feedback-driven optimizing tier (SSA IR, inlining, escape analysis,
  range/redundancy elimination), with deopt back to bytecode. The point where
  we contend with V8 on compute-bound code.

- **Ongoing** — Test262 in CI, fuzzing (parser, regex, JSON, the VM via
  cargo-fuzz like purecrypto), `ffi` surface growth, embedder API docs.

---

## 7. Benchmarks & success criteria

- **Correctness**: Test262 pass-rate tracked in CI; target >95% of the
  language (non-Intl) suite by end of Phase E.
- **Startup**: cold `kataan -e '...'` time competitive with `node -e`.
- **Throughput**: SunSpider/Kraken-style microbenchmarks and a small set of
  realistic scripts; interpreter within a small multiple of V8's interpreter by
  Phase E, JIT closing the gap through G/H.
- **Memory**: per-object overhead bounded by the shape model; heap measured
  against equivalent V8 snapshots.
- **Embeddability**: the Rust and C APIs can run a script, expose host
  functions, and pump the event loop in <30 lines.

---

## 8. Reused Karpelès Lab crates

- **`purecrypto`** — `crypto.subtle`/WebCrypto, `crypto.getRandomValues`,
  `randomUUID`, and TLS for the network stack. No foreign crypto.
- **`rsurl`** — HTTP/HTTPS transport behind `fetch` and the Node `http(s)`
  compat layer.
- Patterns borrowed wholesale: tri-modal lib/CLI/C-FFI packaging, `unsafe`
  quarantine, feature-gated layered modules, sans-I/O cores, cargo-fuzz
  harnesses, and release-plz publishing.
