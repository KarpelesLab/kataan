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

pub mod common;
pub mod error;
pub mod lexer;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use error::{Error, Result};

/// The crate version, from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
