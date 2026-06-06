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

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod ast;
pub mod common;
pub mod error;
pub mod lexer;
pub mod nanbox;
pub mod parser;

/// The managed heap (generational handle table) that NaN-boxed handles index
/// into — groundwork for the object model & GC. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod heap;

/// Hidden classes ("shapes"): shared property-layout descriptors with a
/// transition tree — groundwork for the object model. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod shape;

/// The performance-era object: a [`shape::Shape`] paired with
/// [`nanbox::NanBox`] value slots — composes the object-model pillars. Needs
/// `alloc`.
#[cfg(feature = "alloc")]
pub mod object;

/// Heap cells — the reference types (object / string / array / function) a heap
/// slot holds. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod cell;

/// Lexical environments (scope chains) for closures over the new model. Needs
/// `alloc`.
#[cfg(feature = "alloc")]
pub mod env;

/// A mark-and-sweep tracing garbage collector over [`heap::Heap`] — reclaims
/// unreachable objects, including reference cycles. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod gc;

/// Inline caches for property access, keyed on [`shape::Shape`] identity — the
/// fast path that turns a repeated `obj.x` into a slot load. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod ic;

/// Interned strings ("atoms"): distinct identifiers/property keys mapped to
/// small `Copy` integers for O(1) comparison. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod atom;

/// Rope strings: lazy O(1) concatenation so building a string piecewise is not
/// quadratic. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod rope;

/// The object-model context (`Realm`) bundling the heap, the shared root shape,
/// and the atom table behind the allocate/get/set/collect API a VM uses. Needs
/// `alloc`.
#[cfg(feature = "alloc")]
pub mod realm;

/// Shared, `alloc`-only `JSON.parse` / `JSON.stringify` over realm values.
#[cfg(feature = "alloc")]
pub mod json;

/// Heap snapshots of the live `Cell` object graph — capture an initialized heap
/// and restore it into a fresh realm (Phase D′ heap-snapshot tier). Needs `alloc`.
#[cfg(feature = "alloc")]
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

/// A minimal register VM over the `Realm`/`NanBox` representation — the proof
/// that the performance object model executes code. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod nbvm;

/// Flat, fixed-record bytecode executed in place over a (possibly `mmap`'d) byte
/// buffer — the true zero-copy reload path (Phase D′). Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod flatbc;

/// The baseline machine-code JIT (Phase G): an x86-64 assembler + W^X executable
/// memory (mapped via direct Linux syscalls, no libc) that lowers an arithmetic
/// IR to native code. Real on `linux`/`x86_64`; a no-op fallback elsewhere.
#[cfg(feature = "jit")]
pub mod jit;

/// A portable `KTBC` serialization codec for the bytecode VM's compiled programs
/// (the code cache — `ROADMAP.md` Phase D′). Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod bytecode;

/// A pure, `alloc`-only arbitrary-precision integer — the foundation for a
/// conformant `BigInt` (`ROADMAP.md`). Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod bignum;

/// Evaluates the real parser AST (the expression subset) over the
/// `Realm`/`NanBox` model — the front-end → new-representation bridge. Needs
/// `alloc`.
#[cfg(feature = "alloc")]
pub mod nbeval;

/// Executes real statements (variables, scope, control flow, assignment) over
/// the `Realm`/`NanBox` model — the imperative core on the new representation.
/// Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod nbexec;

/// The in-house regular-expression engine (the `regex` feature). Pure Rust,
/// `no_std`-compatible (`alloc` only).
#[cfg(feature = "regex")]
pub mod regex;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use error::{Error, Result};

/// The crate version, from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
