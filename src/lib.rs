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
pub mod bytecode;
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

/// A minimal register VM over the `Realm`/`NanBox` representation — the proof
/// that the performance object model executes code. Needs `alloc`.
#[cfg(feature = "alloc")]
pub mod nbvm;

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

/// The tree-walking interpreter (Phase-C semantics MVP). Gated on `std` for
/// now because it uses floating-point math routines (`powf`, `trunc`, …) that
/// live in the standard library; the float-math dependency will be revisited
/// (`libm` vs `std`) when the bytecode VM lands.
#[cfg(feature = "std")]
pub mod interp;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use error::{Error, Result};

/// The crate version, from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
