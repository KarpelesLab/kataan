//! `kataan` — a high-performance JavaScript (ECMAScript) engine written
//! entirely in Rust, depending on no foreign code.
//!
//! Kataan is built bottom-up in layers, and is usable three ways — as a Rust
//! library, as a C library (the `ffi` feature), and as a standalone command-
//! line tool / REPL (the `cli` feature). See `ROADMAP.md` for the full design
//! and milestone plan.
//!
//! The pipeline:
//!
//! ```text
//! source ──[lexer]──▶ tokens ──[parser]──▶ AST ──[compiler]──▶ bytecode
//!                                                                  │
//!                                                            [interpreter]
//! ```
//!
//! # `no_std`
//!
//! The language core is `#![no_std]` and needs only `alloc`. The `std` feature
//! (default, implies `alloc`) adds the host runtime — the event loop, timers,
//! file system, network (`fetch` over `rsurl`), and `crypto` (over
//! `purecrypto`). Build the bare core with `--no-default-features --features
//! alloc`.

#![no_std]
// `missing_docs` / `unreachable_pub` are warned at the crate level (see
// `Cargo.toml [lints]`); module-level `#![allow(...)]` is used sparingly while
// a layer is still a scaffold.
#![cfg_attr(docsrs, feature(doc_cfg))]
// rustdoc 1.99 reports two spanless `redundant_explicit_links` (no module-level
// allow silences them, and it cannot name the link); a cosmetic lint, so it is
// allowed crate-wide rather than failing `-D warnings`.
#![allow(rustdoc::redundant_explicit_links)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// Public API vs engine internals: the modules marked `#[doc(hidden)]` below
// are the engine's implementation (bytecode, VM, JIT, heap/object layout, the
// interpreter's built-ins). They stay `pub` so this crate's own tests, examples
// and tools can reach them, but they are not part of the supported API and are
// exempt from semver: their items change whenever the engine does (an opcode
// added, a cell layout shrunk). The supported surface is the visible modules
// and the re-exports at the end of this file.

pub mod ast;
#[doc(hidden)]
pub mod common;
pub mod error;
pub mod lexer;
#[doc(hidden)]
pub mod nanbox;
pub mod parser;

/// The managed heap (generational handle table) that NaN-boxed handles index
/// into — groundwork for the object model & GC. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod heap;

/// Hidden classes ("shapes"): shared property-layout descriptors with a
/// transition tree — groundwork for the object model. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod shape;

/// The performance-era object: a [`shape::Shape`] paired with
/// [`nanbox::NanBox`] value slots — composes the object-model pillars. Needs
/// `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod object;

/// Heap cells — the reference types (object / string / array / function) a heap
/// slot holds. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod cell;

/// Lexical environments (scope chains) for closures over the new model. Needs
/// `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod env;

/// A mark-and-sweep tracing garbage collector over [`heap::Heap`] — reclaims
/// unreachable objects, including reference cycles. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod gc;

/// Inline caches for property access, keyed on [`shape::Shape`] identity — the
/// fast path that turns a repeated `obj.x` into a slot load. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod ic;

/// Interned strings ("atoms"): distinct identifiers/property keys mapped to
/// small `Copy` integers for O(1) comparison. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod atom;

/// Tunable resource limits (recursion depths, allocation sizes, regex/wasm
/// budgets) with safe defaults, overridable per [`realm::Realm`].
pub mod limits;

/// WTF-8 string codec: UTF-8 extended to encode lone UTF-16 surrogates, so
/// DOMString values round-trip losslessly. The storage foundation for rope/atom.
/// Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod wtf8;

/// A cooperative interrupt flag a host watchdog trips to stop a runaway script.
/// Needs `alloc` (the flag lives in a shared `Arc`).
#[cfg(feature = "alloc")]
pub mod interrupt;

/// Rope strings: lazy O(1) concatenation so building a string piecewise is not
/// quadratic. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod rope;

/// The object-model context (`Realm`) bundling the heap, the shared root shape,
/// and the atom table behind the allocate/get/set/collect API a VM uses. Needs
/// `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod realm;

/// Shared, `alloc`-only `JSON.parse` / `JSON.stringify` over realm values.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod json;

/// Heap snapshots of the live `Cell` object graph — capture an initialized heap
/// and restore it into a fresh realm (Phase D′ heap-snapshot tier). Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod snapshot;

/// The WebAssembly peer engine: lowers the numeric subset of JS functions to
/// WebAssembly text (WAT) — a second compilation target. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod wasm;

/// The WebAssembly execution engine (Phase H): decodes and *runs* `.wasm`
/// binaries — the peer engine proper, distinct from `wasm` (JS→WASM). Needs
/// `alloc`.
#[cfg(feature = "alloc")]
pub mod wasm_rt;

/// A WebAssembly spec-test harness: typed `assert_return`/`assert_trap`/
/// `assert_invalid` commands run against the `wasm_rt` engine. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod wasm_spec;

/// A minimal register VM over the `Realm`/`NanBox` representation — the proof
/// that the performance object model executes code. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod nbvm;

/// Flat, fixed-record bytecode executed in place over a (possibly `mmap`'d) byte
/// buffer — the true zero-copy reload path (Phase D′). Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod flatbc;

/// The baseline machine-code JIT (Phase G): an x86-64 assembler + W^X executable
/// memory (mapped via direct Linux syscalls, no libc) that lowers an arithmetic
/// IR to native code. Real on `linux`/`x86_64`; a no-op fallback elsewhere.
#[cfg(feature = "jit")]
#[doc(hidden)]
pub mod jit;

/// A portable `KTBC` serialization codec for the bytecode VM's compiled programs
/// (the code cache — `ROADMAP.md` Phase D′). Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod bytecode;

/// A pure, `alloc`-only arbitrary-precision integer — the foundation for a
/// conformant `BigInt` (`ROADMAP.md`). Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod bignum;

/// The interpreter: the realm, built-ins, natives and host services (job
/// queues, modules, realms, agents) the bytecode VM runs in. Needs `alloc`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
pub mod nbexec;

/// Host runtime — Node/Bun/Deno-parity globals (ROADMAP §4). Needs `std`.
#[cfg(feature = "host")]
pub mod host;

/// Pure-logic ISO-8601 calendar/time/duration core shared by the `Temporal.*`
/// native types (no engine coupling — plain structs + functions, unit-testable).
#[doc(hidden)]
pub mod temporal_iso;

/// The in-house regular-expression engine (the `regex` feature). Pure Rust,
/// `no_std`-compatible (`alloc` only).
#[cfg(feature = "regex")]
pub mod regex;

#[cfg(feature = "ffi")]
pub mod ffi;

/// A pure-Rust size-class pooling allocator ([`pool_alloc::PoolAlloc`]) that an
/// executable can install as its `#[global_allocator]` (the `kataan` CLI does).
/// The library never installs it itself. Needs `std` (the `pool-alloc` feature).
#[cfg(feature = "pool-alloc")]
pub mod pool_alloc;

pub use error::{Error, Result};
pub use limits::{Limits, WasmLimits};
pub use nanbox::NanBox;
// The embedding surface (`ROADMAP.md` §4.0): register Rust closures as JS
// functions (`Interp::register_fn`) and talk back to the engine through `Ctx`.
pub use nbexec::{Ctx, HostCallback, Interp};

/// The crate version, from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
