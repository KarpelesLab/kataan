//! Build script: defines the engine's target-capability `cfg` aliases, so the
//! many items gated on "this target has the x86-64 JIT backend" name one
//! condition instead of repeating its parts.
//!
//! - `jit_x64`: the x86-64 machine-code JIT (`src/jit.rs`, `src/jit/`, and
//!   the VM's JIT tier) can run on this target: x86-64 on Linux, Windows or
//!   macOS (the emitted code uses the System V calling convention on all of
//!   them, see `src/jit.rs`).
//! - `jit_a64`: the AArch64 backend (`src/jit/a64*.rs`) can run on this
//!   target: AArch64 on Linux or macOS.
//! - `jit_native`: either backend can run here, so the JIT maps and calls
//!   native code (`jit_x64` or `jit_a64`).
//!
//! Every use still adds its own feature condition (`feature = "jit"` or
//! `feature = "alloc"`); these aliases only describe the target.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(jit_x64)");
    println!("cargo::rustc-check-cfg=cfg(jit_a64)");
    println!("cargo::rustc-check-cfg=cfg(jit_native)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let x64 = arch == "x86_64" && matches!(os.as_str(), "linux" | "windows" | "macos");
    let a64 = arch == "aarch64" && matches!(os.as_str(), "linux" | "macos");
    if x64 {
        println!("cargo::rustc-cfg=jit_x64");
    }
    if a64 {
        println!("cargo::rustc-cfg=jit_a64");
    }
    if x64 || a64 {
        println!("cargo::rustc-cfg=jit_native");
    }
    println!("cargo::rerun-if-changed=build.rs");
}
