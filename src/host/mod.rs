//! Host runtime — Node.js / Bun / Deno-parity globals (ROADMAP §4).
//!
//! Built on the §4.0 embedding API (`Interp::register_global_fn` / `register_fn`
//! / `Ctx`), so runtime builtins and third-party host code travel one path. Each
//! submodule installs one area's globals into a live [`Interp`] and is
//! independently testable; `install_all` wires them together for the CLI.
use crate::nbexec::Interp;

pub mod node;
pub mod timers;
pub mod web;

/// Install the full host-runtime surface into `interp` (web globals, timers /
/// event-loop scheduling primitives, and the `node:`-compat builtins).
pub fn install_all(interp: &mut Interp<'_>) {
    web::install(interp);
    node::install(interp);
    timers::install(interp);
}
