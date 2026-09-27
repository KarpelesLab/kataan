# Kataan

[![CI](https://github.com/KarpelesLab/kataan/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/kataan/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/kataan.svg)](https://crates.io/crates/kataan)
[![docs.rs](https://docs.rs/kataan/badge.svg)](https://docs.rs/kataan)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance **JavaScript (ECMAScript) engine written in pure Rust**, with
no foreign code on the critical path. Kataan is usable three ways — as a Rust
library, as a C library, and as a standalone command-line tool — the same
tri-modal model proven out in the sibling projects
[`purecrypto`](https://github.com/KarpelesLab/purecrypto) (cryptography) and
[`rsurl`](https://github.com/KarpelesLab/rsurl) (HTTP/curl).

> **Status: running and broadly conformant; advanced tiers in active build-out.**
> The lexer and the full ECMAScript parser are complete, and **two execution
> engines** run real programs, cross-checked by differential test suites:
>
> - a **tree-walking interpreter** — the reference engine, and the one the
>   Test262 corpus exercises; and
> - a **register bytecode VM** (the primary path for `kataan run` and the C ABI),
>   compiling nearly all of the common language directly — every operator,
>   objects/arrays, method calls with `call`/`apply`/`bind`, `new`/`new.target`,
>   all loops + `for-of`/`for-in`/`switch`/`try`-`catch`-`finally`,
>   closures (incl. mutual recursion), destructuring, rest/spread, **classes**
>   with `extends`/`super` and getters/setters, and **lazy, truly-suspendable
>   generators and `async`/`await`** (`yield`/`next(v)`/`.throw()`, async
>   generators, `for await`, and `await` resuming as a microtask with correct
>   ordering) — faulting to the tree-walker for what it doesn't yet compile,
>   notably `eval`/`Function`, `new` on a plain (non-class) function, and the
>   `arguments` object. (The Test262 harness itself uses the last two, which is
>   why the corpus runs on the tree-walker.)
>
> Conformance is measured against the **full upstream tc39/Test262** (~53k tests,
> `staging/` included), run in CI and gated by a known-failures ledger that only
> ever shrinks. The current pass-rate is **≈ 99.99 %** — 53,374 of the 53,377
> tests run (only the two `CanBlockIsFalse` tests are skipped, as a host choice;
> no feature is skip-gated).
> The 3 ledgered failures are deliberate and documented inline: one upstream
> harness bug, and two SpiderMonkey staging tests that contradict the normative
> Annex B test.
> Working areas include **ES modules** (static `import`/`export`, live bindings,
> re-exports, cycles, top-level `await`, `import.meta`) and **dynamic `import()`**,
> explicit resource management (`using`/`await using`), spec **statement
> completion values** (the value `eval` reports, with `UpdateEmpty` through
> `switch`/loops/`try`/labelled `break`), sloppy-mode **Annex B** block-function
> hoisting (through `catch`/`switch`/`if`) and `super` in direct `eval`, closures,
> classes/inheritance, optional chaining, the iterator protocol,
> `Map`/`Set`/`WeakMap`/`WeakRef`/`FinalizationRegistry`, `Symbol` (with a real
> `Symbol.prototype`), `BigInt`, `Promise` (combinators, `withResolvers`, `try`) +
> async/await, `Proxy`/`Reflect`, typed arrays (incl. `Uint8Array` base64/hex),
> `Date`, an in-house `RegExp` (named groups, lookbehind, `u`/`v` flags, inline
> modifiers, property escapes), real sparse-array holes + array property
> descriptors, **`Intl`** (all 3,341 `intl402` tests, over the pure-Rust
> [`intl`](https://crates.io/crates/intl) crate), **Temporal**, `Atomics` with
> real multi-agent workers, cross-realm `$262.createRealm`, and a large standard
> library (Math, JSON, Object/Array/String/Number, and the ES2024/2025
> additions). Compiled bytecode can be serialized, reloaded, and run without the
> source.
>
> Three advanced tiers are real and tested, though each has named work remaining:
>
> - a **machine-code JIT** (x86-64 / Linux, behind `jit`) with an optimizing
>   integer path (four-pass optimizer + register allocator), a float path
>   (arithmetic, comparisons, control flow, the SSE-expressible `Math`
>   intrinsics), and a **generic value tier** that compiles non-numeric hot
>   functions too — property and element access with inline machine-code fast
>   paths, calls, and generic operators re-entering the interpreter's own
>   implementations through runtime helpers — emitting into W^X memory via raw
>   syscalls;
> - a pure-Rust, `no_std` **WebAssembly engine** — full MVP plus sign-extension,
>   saturating conversion, bulk-memory, multi-value, reference types (multiple
>   tables, all element-segment modes), and typed structured control — with a
>   JS↔WASM boundary (`validate`/`compile`/`instantiate`, the
>   `Module`/`Instance`/`Global`/`Memory` objects, host-function imports, and
>   stateful instances), driven by a `.wast`/WAT spec harness (a spec-derived
>   corpus, not yet the full upstream suite);
> - a **zero-copy "D′" snapshot tier** atop the moving GC: a verified codec that
>   `mmap`-reloads a heap (eleven reference cell kinds, cross-kind cycles,
>   insertion-order-preserving) and runs a restored closure both in place and
>   reloaded into a fresh runtime.
>
> Kataan works as a CLI/REPL, a Rust library, and a C library (`kt_eval`). See
> the [roadmap](ROADMAP.md) for the remaining road to a complete engine.

## Why

Modern JavaScript engines (V8, JavaScriptCore, SpiderMonkey) all rely on the
same handful of techniques. Kataan commits to the full set from the
architecture stage rather than retrofitting them:

- **NaN-boxed values** — every JS value in 64 bits, `Copy`, dense on the stack.
- **Hidden classes (shapes) + inline caches** — property access becomes a slot
  load, not a hash probe; the single biggest lever for real-world JS speed.
- **Register-based bytecode VM** — fewer instructions than a stack VM, and
  JIT-friendly by construction.
- **Interned atoms + rope strings** — O(1) key comparison, non-quadratic
  string building.
- **A precise, generational, moving GC** — bump allocation makes `new` nearly
  free.
- **Tiered execution** — a fast interpreter first, then a baseline JIT, then an
  optimizing JIT driven by inline-cache type feedback.

The language core is **sans-I/O** and `no_std + alloc`; the host runtime (event
loop, timers, `fetch`, `crypto`, modules) is a separate layer on top, so the
engine stays embeddable. See [`ROADMAP.md`](ROADMAP.md) for the road ahead — the
remaining work to a complete JS+WASM engine and the design invariants behind it.

## Pure Rust, no foreign code

Kataan depends on no C libraries. Where it needs cryptography or networking it
reuses sibling **pure-Rust** Karpelès Lab crates:

- [`purecrypto`](https://github.com/KarpelesLab/purecrypto) — `crypto.subtle` /
  WebCrypto, `crypto.getRandomValues`, `randomUUID`, and TLS.
- [`rsurl`](https://github.com/KarpelesLab/rsurl) — HTTP/HTTPS transport behind
  `fetch` and the Node `http(s)` compatibility layer.

`unsafe` is quarantined: the crate is `unsafe_code = "deny"` (not `forbid`),
and only the `ffi` module plus a small, audited set of VM hot-path primitives
opt back in with a scoped `#[allow(unsafe_code)]` and a safety comment.

## Try it

The CLI runs JavaScript today:

```console
$ cargo run -- run -e '
class Animal { constructor(n){ this.n = n } speak(){ return `${this.n} makes a sound` } }
class Dog extends Animal { speak(){ return `${this.n} barks` } }
console.log(new Dog("Rex").speak());
console.log([1,2,3,4].filter(x => x % 2).map(x => x*x).reduce((a,b)=>a+b, 0));
console.log(JSON.stringify({ ok: true, items: [...new Set([1,1,2,3])] }));
'
Rex barks
10
{"ok":true,"items":[1,2,3]}
```

It also exposes each pipeline stage, an interactive REPL, a bytecode compiler,
and a runner with the host event loop:

```console
$ cargo run -- lex     -e 'x => x * 2'        # token stream
$ cargo run -- parse   -e 'x => x * 2'        # AST dump
$ cargo run -- eval    -e '[1,2,3].map(x => x * 2)'
2,4,6
$ cargo run -- repl                           # interactive session
$ cargo run -- hostrun -e 'setTimeout(() => console.log("later"), 10); console.log("now")'
now
later
$ cargo run -- --help
```

Programs compile to a portable bytecode artifact that runs without the source:

```console
$ cat fib.js
function fib(n){ return n < 2 ? n : fib(n-1) + fib(n-2) }
console.log(fib(20));
$ cargo run -- compile fib.js -o fib.ktbc
kataan: wrote fib.ktbc (226 bytes)
$ cargo run -- run fib.ktbc
6765
```

## Use as a Rust library

```rust
use kataan::parser::Parser;
use kataan::{Ctx, Interp};

let mut interp = Interp::new();
// Expose a Rust closure to JavaScript.
interp.register_global_fn("hypot", 2, |cx: &mut Ctx, _this, args| {
    let a = cx.to_number(args.first().copied().unwrap_or(cx.undefined()))?;
    let b = cx.to_number(args.get(1).copied().unwrap_or(cx.undefined()))?;
    Ok(cx.number(a.hypot(b)))
});

let program = Parser::parse_program("const sq = x => x * x; sq(8) + hypot(3, 4)").unwrap();
let value = interp.run(&program).unwrap();
assert_eq!(interp.realm().to_display_string(value), "69");
```

The embedding API also builds objects and arrays, throws catchable JS errors,
calls back into script functions, constructs classes, and attaches opaque Rust
state to JS objects; [`examples/embed_host_fn.rs`](examples/embed_host_fn.rs)
walks through all of it (`cargo run --example embed_host_fn`).

The lower stages are available directly too:

```rust
use kataan::lexer::{Lexer, TokenKind};

let src = "let answer = 42;";
let tokens = Lexer::new(src).tokenize().unwrap();
assert_eq!(tokens.first().unwrap().ascii_text(src.as_bytes()), "let");
assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof);
```

### Feature flags

| Feature   | Default | Description                                                        |
|-----------|:-------:|--------------------------------------------------------------------|
| `std`     |   ✓     | Standard library; implies `alloc`. Needed by the host runtime/CLI. |
| `alloc`   |   ✓     | Heap-backed types; the minimum for the pure language core.         |
| `regex`   |   ✓     | In-house regular-expression engine.                                |
| `intl`    |   ✓     | `Intl` over the pure-Rust `intl` crate (CLDR data).                |
| `intl-tz-names` | ✓ | Localized time-zone names for `Intl.DateTimeFormat`.              |
| `module`  |   ✓     | ESM + CommonJS module loader.                                      |
| `host`    |   ✓     | Host runtime: event loop, timers, console, encoding, URL, streams. |
| `crypto`  |   ✓     | `crypto.getRandomValues` / WebCrypto over `purecrypto`.            |
| `fetch`   |         | `fetch` / Node `http(s)` over `rsurl`.                             |
| `jit`     |         | Machine-code JIT (x86-64/Linux): integer, float, and generic tiers.|
| `ffi`     |         | The C ABI (the only place broad `unsafe` is allowed).              |
| `cli`     |   ✓     | The `kataan` command-line tool.                                    |

Build the bare `no_std` language core with:

```console
cargo build --no-default-features --features alloc
```

## Use as a C library

```console
cargo rustc --lib --release --features ffi --crate-type staticlib   # libkataan.a
cargo rustc --lib --release --features ffi --crate-type cdylib      # libkataan.so
```

The header is [`include/kataan.h`](include/kataan.h); a runnable example lives
in [`tests/ffi_smoke.c`](tests/ffi_smoke.c). The C ABI follows the `purecrypto`
conventions — `KtStatus` return codes, the in/out length convention, opaque
handles, and a panic catch at every boundary.

## License

MIT © 2026 Karpelès Lab Inc. See [LICENSE](LICENSE).
