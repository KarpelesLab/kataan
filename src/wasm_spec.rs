//! A WebAssembly spec-test harness (`ROADMAP.md` Phase H — "validated against the
//! WebAssembly spec test suite").
//!
//! The official suite is written in `.wast`: a module followed by `assert_return`
//! / `assert_trap` / `assert_invalid` commands. This module provides the *engine*
//! side of that — typed `Assertion`s and a `run_assertions` runner that decodes a
//! module, instantiates it, and checks each command against the
//! [`crate::wasm_rt`] engine — plus a starter corpus of spec-derived cases. (A
//! `.wast` *text* parser to ingest the upstream files verbatim is the remaining
//! piece; the harness it would feed is here.)
//!
//! Pure, safe `alloc`-only Rust.

use crate::wasm_rt::{Instance, Module, Val};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// A spec-test command over a single instantiated module.
#[derive(Clone, Debug, PartialEq)]
pub enum Assertion {
    /// `assert_return`: calling `func` with `args` yields exactly `expect`.
    Return {
        /// the exported function name
        func: &'static str,
        /// the call arguments
        args: Vec<Val>,
        /// the expected results
        expect: Vec<Val>,
    },
    /// `assert_trap`: calling `func` with `args` traps (errors).
    Trap {
        /// the exported function name
        func: &'static str,
        /// the call arguments
        args: Vec<Val>,
    },
}

/// Runs every [`Assertion`] against a freshly-instantiated `module_bytes`.
///
/// # Errors
/// Returns a human-readable description of the first failing command (a decode or
/// instantiation failure, a wrong result, an unexpected error, or a missing
/// trap); on success returns the number of assertions checked.
pub fn run_assertions(module_bytes: &[u8], asserts: &[Assertion]) -> Result<usize, String> {
    let module = Module::decode(module_bytes).map_err(|e| format!("decode failed: {}", e.0))?;
    let mut inst = Instance::new(&module).map_err(|e| format!("instantiate failed: {}", e.0))?;
    for (i, a) in asserts.iter().enumerate() {
        match a {
            Assertion::Return { func, args, expect } => {
                let got = inst
                    .call_export(func, args)
                    .map_err(|e| format!("assert {i}: {func} errored: {}", e.0))?;
                if &got != expect {
                    return Err(format!(
                        "assert {i}: {func}({args:?}) = {got:?}, expected {expect:?}"
                    ));
                }
            }
            Assertion::Trap { func, args } => {
                if inst.call_export(func, args).is_ok() {
                    return Err(format!("assert {i}: {func}({args:?}) did not trap"));
                }
            }
        }
    }
    Ok(asserts.len())
}

/// Checks `assert_invalid`/`assert_malformed`: the bytes must **fail** to decode.
///
/// # Errors
/// Returns an error if the module unexpectedly decoded successfully.
pub fn assert_invalid(module_bytes: &[u8]) -> Result<(), String> {
    if Module::decode(module_bytes).is_ok() {
        Err(String::from(
            "module decoded but was expected to be invalid",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Builds a single-function `(i32,i32)->i32` module from a body opcode stream
    /// (after `local.get 0; local.get 1`), exported as `op`.
    fn binop_module(tail: &[u8]) -> Vec<u8> {
        let mut body: Vec<u8> = vec![0x00, 0x20, 0x00, 0x20, 0x01];
        body.extend_from_slice(tail);
        body.push(0x0b);
        let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        m.extend([0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f]);
        m.extend([0x03, 0x02, 0x01, 0x00]);
        m.extend([0x07, 0x06, 0x01, 0x02, b'o', b'p', 0x00, 0x00]);
        m.push(0x0a);
        m.push((body.len() + 2) as u8);
        m.push(0x01);
        m.push(body.len() as u8);
        m.extend(body);
        m
    }

    #[test]
    fn spec_return_assertions() {
        // i32.add (0x6a)
        let add = binop_module(&[0x6a]);
        let n = run_assertions(
            &add,
            &[
                Assertion::Return {
                    func: "op",
                    args: vec![Val::I32(20), Val::I32(22)],
                    expect: vec![Val::I32(42)],
                },
                Assertion::Return {
                    func: "op",
                    args: vec![Val::I32(-5), Val::I32(5)],
                    expect: vec![Val::I32(0)],
                },
            ],
        )
        .expect("add assertions pass");
        assert_eq!(n, 2);
    }

    #[test]
    fn spec_trap_assertion() {
        // i32.div_s (0x6d) traps on divide-by-zero.
        let div = binop_module(&[0x6d]);
        run_assertions(
            &div,
            &[
                Assertion::Return {
                    func: "op",
                    args: vec![Val::I32(20), Val::I32(4)],
                    expect: vec![Val::I32(5)],
                },
                Assertion::Trap {
                    func: "op",
                    args: vec![Val::I32(1), Val::I32(0)],
                },
            ],
        )
        .expect("div assertions pass");
    }

    #[test]
    fn spec_runner_reports_a_wrong_result() {
        let add = binop_module(&[0x6a]);
        let err = run_assertions(
            &add,
            &[Assertion::Return {
                func: "op",
                args: vec![Val::I32(1), Val::I32(1)],
                expect: vec![Val::I32(99)], // wrong on purpose
            }],
        )
        .unwrap_err();
        assert!(err.contains("expected"), "diagnostic: {err}");
    }

    #[test]
    fn spec_runner_reports_a_missing_trap() {
        let add = binop_module(&[0x6a]);
        let err = run_assertions(
            &add,
            &[Assertion::Trap {
                func: "op",
                args: vec![Val::I32(1), Val::I32(1)],
            }],
        )
        .unwrap_err();
        assert!(err.contains("did not trap"), "diagnostic: {err}");
    }

    #[test]
    fn assert_invalid_rejects_bad_modules() {
        // Bad magic.
        assert!(assert_invalid(&[0, 0, 0, 0, 1, 0, 0, 0]).is_ok());
        // Truncated.
        assert!(assert_invalid(&[0x00, 0x61]).is_ok());
        // A genuinely valid module must NOT be reported invalid.
        let add = binop_module(&[0x6a]);
        assert!(assert_invalid(&add).is_err());
    }
}
