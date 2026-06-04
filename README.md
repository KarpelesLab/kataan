# Kataan

A high-performance **JavaScript (ECMAScript) engine written in pure Rust**, with
no foreign code on the critical path. Kataan is usable three ways — as a Rust
library, as a C library, and as a standalone command-line tool — the same
tri-modal model proven out in the sibling projects
[`purecrypto`](https://github.com/KarpelesLab/purecrypto) (cryptography) and
[`rsurl`](https://github.com/KarpelesLab/rsurl) (HTTP/curl).

> **Status: early but running (Phase C).** The lexer and the full ECMAScript
> parser are complete, and a tree-walking interpreter already executes real
> programs — functions/closures, classes with inheritance, objects/arrays,
> destructuring, getters/setters, `Map`/`Set`, error handling, and a
> substantial standard library (Math, JSON, Object/Array/String/Number). The
> performance-oriented object model (NaN-boxing, hidden classes, GC), the
> bytecode VM and JIT tiers, the host runtime, and the WASM engine are being
> built out per the [roadmap](ROADMAP.md).

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
engine stays embeddable. See [`ROADMAP.md`](ROADMAP.md) for the full design,
the builtins inventory, and the milestone plan.

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

It also exposes each pipeline stage, and an interactive REPL:

```console
$ cargo run -- lex   -e 'x => x * 2'   # token stream
$ cargo run -- parse -e 'x => x * 2'   # AST dump
$ cargo run -- repl                    # interactive session
$ cargo run -- --help
```

## Use as a Rust library

```rust
use kataan::parser::Parser;
use kataan::interp::Interp;

let program = Parser::parse_program("const sq = x => x * x; sq(8)").unwrap();
let mut interp = Interp::new();
assert_eq!(interp.run(&program).unwrap().to_js_string(), "64");
```

The lower stages are available directly too:

```rust
use kataan::lexer::{Lexer, TokenKind};

let tokens = Lexer::new("let answer = 42;").tokenize().unwrap();
assert_eq!(tokens.first().unwrap().text("let answer = 42;"), "let");
assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof);
```

### Feature flags

| Feature   | Default | Description                                                        |
|-----------|:-------:|--------------------------------------------------------------------|
| `std`     |   ✓     | Standard library; implies `alloc`. Needed by the host runtime/CLI. |
| `alloc`   |   ✓     | Heap-backed types; the minimum for the pure language core.         |
| `regex`   |   ✓     | In-house regular-expression engine.                                |
| `intl`    |   ✓     | In-house `Intl`-lite (collation, number/date formatting).          |
| `module`  |   ✓     | ESM + CommonJS module loader.                                      |
| `host`    |   ✓     | Host runtime: event loop, timers, console, encoding, URL, streams. |
| `fetch`   |         | `fetch` / Node `http(s)` over `rsurl`.                             |
| `crypto`  |         | `crypto.getRandomValues` / WebCrypto over `purecrypto`.            |
| `jit`     |         | Baseline + optimizing JIT tiers (later phases).                    |
| `ffi`     |         | The C ABI (the only place broad `unsafe` is allowed).             |
| `cli`     |   ✓     | The `kataan` command-line tool.                                   |

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
