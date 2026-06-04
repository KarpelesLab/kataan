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
- **Deployable bytecode.** Compiled JS bytecode is a first-class, serializable
  artifact: it can be exported to disk, content-addressed, cached, and reloaded
  (zero-copy via `mmap` on a matching host) without re-parsing the source. The
  on-disk form is **host-native** (endianness matches the machine) for that
  zero-copy path, with an explicit on-demand conversion when a blob is read on a
  differing host — not a single slow canonical encoding. This is a hard design
  constraint on the bytecode format from day one (see §2.2), not a bolt-on — it
  shapes how constants, atoms, and inline-cache slots are encoded.

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

## 2. Architecture decisions

These are the load-bearing, hard-to-reverse choices. They are recorded here so
the rationale survives, and so later layers are designed to fit them from the
start rather than retrofitted.

### 2.1 Own JS bytecode + VM, with WASM as a *peer* engine sharing the backend

Kataan executes JavaScript through its **own** register bytecode and VM — it
does **not** compile JS to WebAssembly. WebAssembly is supported as a separate,
first-class execution engine running *next to* the JS engine (the
`WebAssembly` builtin, and a usable standalone WASM runtime), not as the
substrate JS is lowered onto.

**Why not JS→WASM as the core.** Routing JS through WASM doesn't remove any of
the hard parts of a JS engine — you still need NaN-boxing, hidden classes,
inline caches, and a GC; you'd just be writing them *in* WASM. And it actively
caps performance: the value of a JS optimizing JIT is *speculative type
specialization with cheap guards and deoptimization* ("assume small-int, lower
to a raw load, bail to the interpreter if the shape guard fails"). WASM has no
deopt primitive and no way to express guard-based bailout, so putting it in the
middle of the JS optimization pipeline throws away the very mechanism that makes
JS fast. This is why V8/JSC/SpiderMonkey all JIT JS straight to native, and why
the JS-on-WASM products that exist (Javy, `spidermonkey.wasm`) embed a
conventional engine *compiled to* WASM rather than compiling JS to WASM.

**What the two engines share** (the real economy — share the backend, not the
ISA):

- **GC + heap.** One collector. WASM linear memory is a GC-tracked byte array;
  WASM-GC objects (later) live on the same heap.
- **Native code backend.** One Cranelift-style machine-code generator (register
  allocation, executable-memory management, relocations). The JS optimizing JIT
  and the WASM compiler both lower *into it* (as V8's TurboFan compiles both).
- **Value / interop boundary.** How a JS value crosses into a WASM call and
  back (numbers, the `externref`/reference-types bridge, linear-memory views ↔
  typed arrays).
- **Host runtime.** Event loop, modules, sandbox, executable-memory
  permissions — one set, serving both.

```text
   JS source                          .wasm module
       │                                   │
   parser → AST                       decode / validate
       │                                   │
   JS bytecode ──▶ JS VM             WASM bytecode ──▶ WASM VM
       │          (interp + tiers)        │          (interp / baseline)
       └─────────────┐                    └─────────────┐
                     ▼                                   ▼
        ┌────────  shared native backend (Cranelift-style)  ────────┐
        └─────────  shared GC · heap · sandbox · host runtime  ──────┘
```

The two bytecodes stay genuinely distinct: JS bytecode carries dynamic,
deopt-friendly ops; WASM bytecode is the standard, statically-typed format. We
do not try to unify the two ISAs.

**Conditional WASM-JIT tier.** The one place emitting WASM *for JS* is correct:
when Kataan itself runs **hosted inside a WASM sandbox** and cannot generate
native code. There, the baseline tier can emit WASM and hand it to the host's
`WebAssembly.compile` — the only way to JIT in that environment. So the baseline
tier is designed to target either native *or* WASM, with WASM as a portability
fallback, not the primary path.

**The fork we're explicitly not taking (for now).** If the primary goal were
sandboxed / edge / tiny-startup execution rather than raw throughput, a
WASM-centric design (or embedding a QuickJS-class engine in WASM) would be the
right call — that's the Fastly/Shopify niche. Kataan's stated goal is
Node-competitive performance, so it takes the native-JIT path and treats edge
sandboxing as the conditional tier above.

### 2.2 Serializable, host-native bytecode (the code cache)

Compiled JS is a **persistable artifact**. The motivating use case: a web server
hosting hundreds of compiled JS bases that cannot all stay resident in memory —
compile once, persist the bytecode, then load → run → evict on demand, reloading
from the cache (ideally zero-copy) instead of re-parsing source.

The format is deliberately **not** byte-for-byte portable across architectures.
It is **host-native** — endianness (and any other host-specific encoding
choice) match the producing machine, so a blob can be `mmap`'d and run *as-is*
on a matching host, which is the overwhelmingly common case (a server, or a
homogeneous cluster, reloading what it compiled). Portability is provided by an
explicit, on-demand **conversion** pass rather than by a slow canonical
encoding everyone pays for. This imposes concrete, up-front constraints on the
format:

- **Self-contained, position-independent layout.** A serialized unit is
  `(bytecode + constant pool + atom/string table + function & scope metadata +
  optional source map)`, all cross-referenced by **index**, never by live heap
  pointer. Nothing in the on-disk form may embed a runtime address.
  *Position*-independent (no addresses) is a separate axis from *encoding*-
  independent: the layout is relocatable, but its integer encoding is host-
  native (next bullet). Offsets/sizes use explicit fixed-width fields (`u32`/
  `u64`), so word size never enters — endianness is the only host-encoding axis.
- **Host-native encoding, convertible on demand.** Multi-byte fields are stored
  in the host's byte order; the header records which (plus an alignment/ABI
  tag). On a *matching* host the loader maps and runs with zero transformation.
  On a *mismatched* host (e.g. a big-endian reader of a little-endian blob) the
  loader runs a one-time, deterministic **byte-swap / re-pack** pass that yields
  a matching blob — and re-caches it under the host-tagged key so the cost is
  paid once. This conversion is *cheap and lossless* (a structural swap, no
  semantics), and crucially is **not** a recompile — it is the fast path for a
  host-format mismatch, distinct from the version mismatch below.
- **Module-local atoms, remapped on load.** Interned strings are per-runtime
  integers, so a module carries its own string table and its atoms are
  module-local indices, re-interned into the host runtime's table at load time.
- **Inline caches are runtime state, not serialized.** The IC *slots* (count,
  layout) are part of the format, but they load **reset/uninitialized** — they
  hold live shape pointers and type feedback, which never cross the disk
  boundary. (Type feedback *may* later be persisted as optional profiling
  *hints* for the JIT, never as pointers.)
- **Shapes are rebuilt at runtime.** The format references property keys by
  atom; hidden classes form as objects are constructed. No shape is serialized.
- **Versioned + integrity-checked, with two distinct mismatch paths.** A header
  carries a magic, a format version, an engine-version/flags hash, and the
  host-encoding tag (endianness/alignment). The loader distinguishes:
  *(a)* **version/flags mismatch or failed checksum → recompile from source**
  (the bytecode may mean something different, so it cannot be trusted); and
  *(b)* **host-encoding mismatch, same version → convert** (the cheap byte-swap
  pass above, no recompile). The store is content-addressed by source hash;
  the resident artifact additionally carries the host tag, so a little-endian
  and a big-endian host don't collide on incompatible bytes — they hold (or
  convert into) their own host-tagged variant. Identical sources across tenants
  on the same host class dedup to one artifact.
- **Zero-copy loadable (on a matching host).** The layout is flat and aligned so
  a cached module can be `mmap`'d and the interpreter run directly over the
  mapped bytes — the key to fast load/evict/reload churn. (On a mismatched host
  the one-time conversion runs first, then the result is zero-copy on every
  subsequent load.) The read-only bytecode pages are **shareable across many
  concurrent contexts** (even across processes): the immutable bytecode backs
  the program, while each lightweight context owns its own mutable
  heap/globals — exactly the model needed for "hundreds of bases."
- **Lazy function bodies.** Nested function bytecode is stored per-function and
  can be faulted in on first call, so loading a large base doesn't materialize
  code that never runs.
- **Untrusted-load verification.** Loading bytecode *is* executing it. A loader
  for untrusted artifacts runs a **verifier** (bounds-checked jumps, constant /
  atom / register indices, stack-depth invariants) the way a WASM validator
  does; the trusted-cache fast path relies on the version tag + checksum and
  skips full verification. Safe-Rust means a bad index degrades to a clean
  rejection, never memory unsafety.

Because compiled WASM modules are *also* a serializable, cacheable artifact, the
JS bytecode cache and the compiled-WASM cache share one **versioned, mmap-able
artifact store** — another payoff of the shared-backend design in §2.1.

A heavier **heap-snapshot** tier (serialize an *initialized* module instance —
globals and instantiated objects — to skip init too, à la V8 startup snapshots)
is a natural future extension once the moving GC and a pointer-relocation pass
exist; it is explicitly *later* than the bytecode cache and not required for the
hundreds-of-bases use case, which re-runs cheaply from cached bytecode.

---

## 3. Crate architecture

```
kataan/
├── src/
│   ├── lib.rs            crate root, feature-gated module tree
│   ├── common/           spans, interner/atoms, diagnostics, arena
│   ├── lexer/            source → tokens (UTF-8/UTF-16 aware, ASI hints)
│   ├── parser/           tokens → AST (ESTree-ish), error recovery
│   ├── ast/              AST node definitions + visitors
│   ├── compiler/         AST → bytecode, scope/closure analysis, IC slots
│   ├── bytecode/         opcode definitions, encoder/decoder, disassembler,
│   │                       (de)serialization — the portable code-cache format
│   ├── snapshot/         the versioned, mmap-able artifact store + code cache
│   │                       (shared by JS bytecode and compiled WASM; see §2.2)
│   ├── value/            NaN-boxed Value, number/string/bigint helpers
│   ├── gc/               heap, allocator, collector, handles/roots
│   ├── object/           shapes (hidden classes), property storage, arrays
│   ├── vm/               interpreter loop, call frames, stack, exceptions
│   ├── wasm/             WASM peer engine: decode/validate, VM, JS interop
│   ├── builtins/         the ECMAScript standard library (see §5)
│   ├── intl/             in-house Intl-lite (collation, casing, number fmt)
│   ├── regex/            in-house regex engine (no foreign code)
│   ├── module/           ESM + CommonJS loader/linker, import resolution
│   ├── host/             event loop, timers, fs, net, fetch, console, process
│   ├── backend/          (later) shared Cranelift-style native code generator,
│   │                       consumed by both the JS JIT and the WASM compiler
│   ├── jit/              (later) baseline + optimizing JS tiers
│   ├── ffi/              C ABI (only place broad `unsafe` is allowed)
│   └── bin/kataan/       the CLI / REPL / script runner
├── include/kataan.h      generated C header
├── testdata/             Test262 harness, fixtures
└── ROADMAP.md
```

Feature gates (Cargo): `std` (default, implies `alloc`), `alloc`, `regex`,
`intl`, `module`, `host`, `crypto` (→ purecrypto), `fetch` (→ rsurl),
`serialize` (the bytecode code-cache), `wasm` (the peer WASM engine), `jit`,
`ffi`, `cli`. The language core (lexer→vm + core builtins) builds with just
`alloc`; `serialize` needs only `alloc` (it is `no_std`-friendly so cached
bytecode loads in embedded/edge hosts too).

---

## 4. Execution pipeline

```
source ──lexer──▶ tokens ──parser──▶ AST ──compiler──▶ bytecode + IC slots
                                                              │
                                            serialize ◀───────┤───────▶ code cache
                                            (§2.2)            │         (mmap reload)
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
  access, call, and binary op; performs trivially-safe constant folding. The
  bytecode it emits is position-independent (§2.2) so it can be serialized.
- **Serialize / code cache** (§2.2): the compiled module can be written to, and
  loaded back from, the versioned artifact store — `mmap`'d and run without
  re-parsing. Optional: a compiled module produced this turn can be skipped
  entirely on the next run if the cache holds a current-version artifact for the
  same source hash.
- **Interpreter**: the register VM. Owns the value stack, call frames, the
  exception/`try`-`finally` unwinder, generators/async suspension (frames are
  heap-relocatable to support `yield`/`await`), and the microtask checkpoint.

---

## 5. Builtins inventory

This is the surface we must implement. Grouped by milestone phase (see §7).
Each line becomes one or more `builtins/` modules with Test262-driven tests.

### 5.1 Core language intrinsics (Phase B–C)

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

### 5.2 Strings, regex, collections (Phase C–D)

- **String** + `String.prototype.*` (full method set incl. `matchAll`,
  `replaceAll`, `normalize`, `localeCompare`, `at`, well-formed Unicode
  methods), `String.raw`, template support.
- **RegExp**: in-house engine (`regex/`). **A first version is implemented** —
  a backtracking VM (parser → instruction list → executor) covering literals,
  `.`, character classes/ranges + `\d\w\s`, anchors, `\b\B`, capturing and
  non-capturing groups, alternation, greedy/lazy `* + ? {n,m}`, the `i`/`m`/`s`
  flags, and `replace` with `$&`/`$1`. **Next:** Unicode mode, named groups,
  lookaround, the `d`/`v` flags, sticky/global state, and `Symbol.replace` —
  plus wiring it to the `RegExp` builtin and the String methods.
- **Array** + `Array.prototype.*` (full set incl. `flat`/`flatMap`,
  `at`, `findLast`/`findLastIndex`, the `toSorted`/`toReversed`/`toSpliced`/
  `with` copying methods, `group`*), `Array.from`/`of`/`isArray`. Fast packed
  (dense SMI / double / object) element kinds with a dictionary fallback.
- **Typed arrays**: `ArrayBuffer`, `SharedArrayBuffer`, `DataView`, all
  `%TypedArray%` views, resizable/growable buffers, `Atomics`.
- **Keyed collections**: `Map`, `Set`, `WeakMap`, `WeakSet`, `WeakRef`,
  `FinalizationRegistry` (the last three need GC cooperation).

### 5.3 Control, metaprogramming, data (Phase D–E)

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

### 5.4 Intl-lite (Phase E, `intl` feature)

A pragmatic subset, in pure Rust, sufficient for common apps: `Intl.Collator`,
`Intl.NumberFormat`, `Intl.DateTimeFormat`, `Intl.PluralRules`,
`Intl.Segmenter`, locale negotiation. Backed by an embedded, trimmed CLDR/ICU
data table rather than a C ICU dependency.

### 5.5 Host runtime (Phase F, `host` feature — Node-compatible-ish)

The "batteries" that make Kataan a usable runtime rather than just a language:

- **Console & diagnostics**: `console.*`, `performance.now`/marks,
  `queueMicrotask`, `structuredClone`.
- **Timers & event loop**: `setTimeout`/`setInterval`/`setImmediate` +
  `clear*`, `process.nextTick`, the libuv-equivalent loop (in-house, built on
  mio-style readiness or std threads).
- **Encoding**: `TextEncoder`/`TextDecoder`, `atob`/`btoa`, `Buffer`.
- **URL**: `URL`/`URLSearchParams` (WHATWG).
- **Streams**: WHATWG `ReadableStream`/`WritableStream`/`TransformStream`.
- **WebAssembly**: the `WebAssembly` global (`Module`, `Instance`, `Memory`,
  `Table`, `compile`/`instantiate`, `validate`) backed by the `wasm` peer
  engine (§2.1) and its shared GC/backend. Compiled modules participate in the
  same artifact cache as JS bytecode (§2.2).
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

## 6. C ABI (`ffi` feature)

Mirrors the `purecrypto` model: opaque handles created/freed by the library,
`KtStatus` return codes (`0` = ok, negative = error), the in/out length
convention for variable-length output, and a panic catch at every boundary so a
Rust panic becomes `KtStatus::Internal` instead of unwinding into C. Surface
(initial): create/destroy a `KtRuntime` and `KtContext`, evaluate a script,
marshal values across the boundary (`KtValue`), define native callbacks, drive
the microtask/event loop, and read exceptions. Header generated to
`include/kataan.h`.

The code-cache (§2.2) is part of the C surface, since the motivating
hundreds-of-bases server may be the C embedder: compile a source to a bytecode
blob (`kt_compile` → buffer, via the in/out length convention), and load a blob
back into a context for execution (`kt_load_bytecode`, zero-copy from a
caller-owned/`mmap`'d buffer). The WASM peer engine (§2.1) is likewise reachable
(`kt_wasm_*`) once it lands.

---

## 7. Milestones

Each milestone ends with: it builds clean under `cargo clippy`, has unit tests,
and (from D onward) reports a Test262 pass-rate number. Status as of this
writing: **A ✅ done · B ✅ done · C 🚧 in progress.**

- **Phase A — Scaffold & lexer** ✅ **done**
  Crate layout, CI, conventions; a complete, tested tokenizer; CLI that can
  `lex` a file. *Deliverable: `kataan lex file.js`.*

- **Phase B — Parser & AST** ✅ **done**
  Full ECMAScript grammar, lex → parse → AST: expressions (precedence, the
  `**`/`??` corner cases, optional chaining), statements + ASI, destructuring
  patterns, functions / arrows (cover grammar) / classes, generators & async
  with `yield`/`await`, and module `import`/`export` (source-type inferred).
  *Deliverable: `kataan parse [-e] file.js`.*

- **Phase C — Tree-walk MVP → object model** 🚧 **in progress**
  A correct (not yet fast) evaluator to validate semantics, *then* the
  performance-oriented object model. **Done so far:** a broad tree-walking
  interpreter (`src/interp/`) that runs real programs — primitives + the full
  operator/coercion set, all control flow, functions/closures/arrows with
  `this`, a provisional `Rc`-based object/array model with prototype chains,
  member access + getters/setters, classes with fields/methods/statics/static
  blocks, single inheritance with `super(...)`/`super.method()`, `new`,
  `instanceof`, `in`, destructuring everywhere, `for-of`/`for-in`, Error
  objects (engine-raised TypeError/ReferenceError are catchable), `Map`/`Set`,
  and a substantial standard library (Math, JSON parse/stringify, Object
  statics, the full Array/String/Number method sets, the Number globals).
  `kataan run app.js` with a minimal `console`, plus an interactive `repl`.
  `Date` (in-house calendar), `RegExp` (over the in-house regex engine, with
  `test`/`exec` and the `String` regex methods), and `Promise`
  (`then`/`catch`/`finally`, `resolve`/`reject`/`all`/`race`, thenable
  adoption) over a microtask queue drained after the script are now in; the C
  ABI's `kt_eval` runs a script end-to-end. **Next:** generators and
  `async`/`await` (which need suspendable frames), timers/the event loop, then
  the *real* object model — NaN-boxed `Value`, shapes/inline caches, and the
  GC — which is also the
  gateway to the Phase-D bytecode VM. First Test262 numbers land here.

- **Phase D — Bytecode VM**
  AST→register-bytecode compiler, the interpreter loop, inline caches,
  closures/upvalues, exceptions/`try-finally`, generators/iterators, RegExp,
  collections, typed arrays, Promise + microtasks, Proxy/Reflect, JSON, Date.
  The bytecode format is position-independent (§2.2) from this phase on. Target:
  broad Test262 coverage of the language proper.

- **Phase D′ — Serializable bytecode & code cache** (`serialize` feature)
  Lands alongside / right after D, while the bytecode format is still malleable.
  The versioned, `mmap`-able artifact format (§2.2): export/reload, host-native
  encoding with the on-demand byte-swap conversion path, module-local atom
  remapping, reset IC slots, integrity + version check, content-addressed store,
  lazy per-function bodies, and the untrusted-load verifier. Read-only bytecode
  shareable across many contexts. *Deliverable: `kataan compile
  app.js -o app.ktbc` and `kataan run app.ktbc`; the C `kt_compile` /
  `kt_load_bytecode` pair.* Validated against the hundreds-of-bases server
  scenario (load → run → evict → reload churn, cross-tenant dedup).

- **Phase E — Conformance & Intl**
  Close Test262 gaps, BigInt edge cases, Atomics, Intl-lite, `WeakRef`/
  `FinalizationRegistry`. Performance pass on the interpreter (IC tuning,
  string ropes, array element-kinds, GC generational upgrade).

- **Phase F — Host runtime**
  Event loop, timers, modules (ESM+CJS), console, encoding, URL, streams,
  `fetch` over rsurl, `crypto` over purecrypto, Node-compat subset. `kataan
  run app.mjs` as a real runtime.

- **Phase G — Shared native backend & baseline JIT**
  The Cranelift-style native code backend (`backend/`), then a copy-and-patch /
  template baseline JIT of hot JS bytecode consuming IC feedback, with on-stack
  replacement for hot loops. The baseline tier is designed to target native *or*
  WASM (the §2.1 sandbox fallback). Measured against the interpreter and Node on
  microbenchmarks.

- **Phase H — WASM peer engine** (`wasm` feature)
  A conformant WebAssembly engine next to the JS VM: decode/validate, an
  interpreter/baseline over the *shared* GC, heap, and native backend; the
  `WebAssembly` builtin and JS↔WASM interop boundary; compiled-module caching
  via the §2.2 artifact store. Validated against the WebAssembly spec test
  suite.

- **Phase I — Optimizing JIT**
  Type-feedback-driven optimizing tier (SSA IR, inlining, escape analysis,
  range/redundancy elimination) lowering through the shared backend, with deopt
  back to bytecode. The point where we contend with V8 on compute-bound code.

- **Later — Heap snapshots**
  Serialize an *initialized* module instance (globals + instantiated objects)
  to skip init as well as parsing (§2.2), once the moving GC and a
  pointer-relocation pass exist.

- **Ongoing** — Test262 (and the WebAssembly test suite) in CI, fuzzing (parser,
  regex, JSON, the bytecode loader/verifier, the WASM validator, the VM, via
  cargo-fuzz like purecrypto), `ffi` surface growth, embedder API docs.

---

## 8. Benchmarks & success criteria

- **Correctness**: Test262 pass-rate tracked in CI; target >95% of the
  language (non-Intl) suite by end of Phase E.
- **Startup**: cold `kataan -e '...'` time competitive with `node -e`.
- **Code-cache load**: on a matching host, loading a cached bytecode module is
  dramatically faster than compiling from source (target: parse+compile skipped
  entirely, load dominated by `mmap` + atom remap), and many contexts can share
  one resident read-only module. A cross-arch load pays the one-time byte-swap
  conversion (measured separately, expected ≪ a recompile) and is zero-copy
  thereafter. Measured on the hundreds-of-bases load/evict/reload workload.
- **Throughput**: SunSpider/Kraken-style microbenchmarks and a small set of
  realistic scripts; interpreter within a small multiple of V8's interpreter by
  Phase E, JIT closing the gap through G/H.
- **Memory**: per-object overhead bounded by the shape model; heap measured
  against equivalent V8 snapshots.
- **Embeddability**: the Rust and C APIs can run a script, expose host
  functions, and pump the event loop in <30 lines.

---

## 9. Reused Karpelès Lab crates

- **`purecrypto`** — `crypto.subtle`/WebCrypto, `crypto.getRandomValues`,
  `randomUUID`, and TLS for the network stack. No foreign crypto.
- **`rsurl`** — HTTP/HTTPS transport behind `fetch` and the Node `http(s)`
  compat layer.
- Patterns borrowed wholesale: tri-modal lib/CLI/C-FFI packaging, `unsafe`
  quarantine, feature-gated layered modules, sans-I/O cores, cargo-fuzz
  harnesses, and release-plz publishing.
