//! Build script: defines the engine's target-capability `cfg` aliases, so the
//! many items gated on "this target has the x86-64 JIT backend" name one
//! condition instead of repeating its parts.
//!
//! - `jit_x64`: the x86-64 machine-code JIT (`src/jit.rs`, `src/jit/`, and
//!   the VM's JIT tier) can run on this target: x86-64 on Linux.
//!
//! Every use still adds its own feature condition (`feature = "jit"` or
//! `feature = "alloc"`); this alias only describes the target.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(jit_x64)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if arch == "x86_64" && os == "linux" {
        println!("cargo::rustc-cfg=jit_x64");
    }
    println!("cargo::rerun-if-changed=build.rs");
}
