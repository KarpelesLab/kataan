//! A WebAssembly spec-test harness (`ROADMAP.md` Phase H — "validated against the
//! WebAssembly spec test suite").
//!
//! The official suite is written in `.wast`: a module followed by `assert_return`
//! / `assert_trap` / `assert_invalid` commands. This module provides the *engine*
//! side of that: typed `Assertion`s + a `run_assertions` runner, and a `.wast`
//! *script* parser (`run_wast`) that ingests the suite's S-expression command
//! stream — `(module binary …)` plus `assert_return`/`assert_trap`/
//! `assert_invalid`/`invoke` — and drives it through the [`crate::wasm_rt`]
//! engine. Modules may be given in `(module binary "\NN…")` form *or* as inline
//! WAT text — `(module (func …))` — which `wat_to_binary` compiles (func
//! signatures, locals, exports, and a flat/folded instruction body over the
//! common opcode set).
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
    /// a bare `invoke`: call `func` for its side effects; no result check, but a
    /// trap is still an error.
    Invoke {
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
                if !vals_match(&got, expect) {
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
            Assertion::Invoke { func, args } => {
                inst.call_export(func, args)
                    .map_err(|e| format!("invoke {i}: {func} trapped: {}", e.0))?;
            }
        }
    }
    Ok(asserts.len())
}

/// Spec-faithful result matching: a `NaN` expectation matches *any* `NaN` result
/// (the suite's `nan:canonical`/`nan:arithmetic`), other floats match by exact bit
/// pattern (so `+0.0` ≠ `-0.0`), and integers match by value.
fn vals_match(got: &[Val], expect: &[Val]) -> bool {
    got.len() == expect.len()
        && got.iter().zip(expect).all(|(g, e)| match (g, e) {
            (Val::F32(g), Val::F32(e)) => {
                if e.is_nan() {
                    g.is_nan()
                } else {
                    g.to_bits() == e.to_bits()
                }
            }
            (Val::F64(g), Val::F64(e)) => {
                if e.is_nan() {
                    g.is_nan()
                } else {
                    g.to_bits() == e.to_bits()
                }
            }
            _ => g == e,
        })
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

// ---------------------------------------------------------------------------
// A `.wast` *script* driver: parses the S-expression command stream the official
// suite uses, with modules given in `(module binary "\NN…")` form, and runs the
// `assert_return` / `assert_trap` / `assert_invalid` / `invoke` commands through
// the harness above. (A full WAT *module* text parser — the `(module (func …))`
// form — would let modules be written inline; the binary form covers the suite's
// `.wast` files that embed assembled bytes, which is the common conformance path.)

/// An S-expression: a list, a bare atom, or a (decoded) byte string.
#[derive(Clone, Debug, PartialEq)]
enum Sexpr {
    List(Vec<Sexpr>),
    Atom(String),
    Str(Vec<u8>),
}

/// Tokenizes + parses `.wast` source into top-level S-expressions.
fn parse_sexprs(src: &str) -> Result<Vec<Sexpr>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut stack: Vec<Vec<Sexpr>> = Vec::new();
    let mut top: Vec<Sexpr> = Vec::new();
    while i < b.len() {
        match b[i] {
            c if c.is_ascii_whitespace() => i += 1,
            b';' if i + 1 < b.len() && b[i + 1] == b';' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' if i + 1 < b.len() && b[i + 1] == b';' => {
                // block comment (; … ;)
                i += 2;
                while i + 1 < b.len() && !(b[i] == b';' && b[i + 1] == b')') {
                    i += 1;
                }
                i += 2;
            }
            b'(' => {
                stack.push(Vec::new());
                i += 1;
            }
            b')' => {
                let list = stack.pop().ok_or("unbalanced ')'")?;
                let node = Sexpr::List(list);
                match stack.last_mut() {
                    Some(parent) => parent.push(node),
                    None => top.push(node),
                }
                i += 1;
            }
            b'"' => {
                let (bytes, next) = parse_wast_string(b, i)?;
                i = next;
                push_node(&mut stack, &mut top, Sexpr::Str(bytes));
            }
            _ => {
                let start = i;
                while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'(' && b[i] != b')' {
                    i += 1;
                }
                let atom = core::str::from_utf8(&b[start..i]).map_err(|_| "non-utf8 atom")?;
                push_node(&mut stack, &mut top, Sexpr::Atom(String::from(atom)));
            }
        }
    }
    if !stack.is_empty() {
        return Err(String::from("unbalanced '('"));
    }
    Ok(top)
}

fn push_node(stack: &mut [Vec<Sexpr>], top: &mut Vec<Sexpr>, node: Sexpr) {
    match stack.last_mut() {
        Some(parent) => parent.push(node),
        None => top.push(node),
    }
}

/// Decodes a `.wast` quoted string (with `\NN` hex and the standard escapes) into
/// its bytes, returning the index just past the closing quote.
fn parse_wast_string(b: &[u8], start: usize) -> Result<(Vec<u8>, usize), String> {
    let mut out = Vec::new();
    let mut i = start + 1; // skip opening quote
    while i < b.len() {
        match b[i] {
            b'"' => return Ok((out, i + 1)),
            b'\\' if i + 1 < b.len() => {
                i += 1;
                match b[i] {
                    b't' => out.push(b'\t'),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b'\\' => out.push(b'\\'),
                    b'"' => out.push(b'"'),
                    b'\'' => out.push(b'\''),
                    h => {
                        // `\NN` — two hex digits → one byte.
                        let hi = (h as char).to_digit(16).ok_or("bad hex escape")?;
                        i += 1;
                        let lo = (*b.get(i).ok_or("truncated hex escape")? as char)
                            .to_digit(16)
                            .ok_or("bad hex escape")?;
                        out.push((hi * 16 + lo) as u8);
                    }
                }
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Err(String::from("unterminated string"))
}

// --- WAT (WebAssembly Text) → binary, for inline `(module (func …))` modules ---

fn leb_u(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn leb_i(mut v: i64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0);
        out.push(if done { byte } else { byte | 0x80 });
        if done {
            break;
        }
    }
}

fn valtype_byte(name: &str) -> Option<u8> {
    Some(match name {
        "i32" => 0x7f,
        "i64" => 0x7e,
        "f32" => 0x7d,
        "f64" => 0x7c,
        _ => return None,
    })
}

/// The opcode of an instruction that carries no immediate operand.
fn simple_opcode(name: &str) -> Option<u8> {
    Some(match name {
        "drop" => 0x1a,
        "select" => 0x1b,
        "return" => 0x0f,
        "i32.eqz" => 0x45,
        "i32.eq" => 0x46,
        "i32.ne" => 0x47,
        "i32.lt_s" => 0x48,
        "i32.lt_u" => 0x49,
        "i32.gt_s" => 0x4a,
        "i32.gt_u" => 0x4b,
        "i32.le_s" => 0x4c,
        "i32.le_u" => 0x4d,
        "i32.ge_s" => 0x4e,
        "i32.ge_u" => 0x4f,
        "i64.eqz" => 0x50,
        "i64.eq" => 0x51,
        "i64.ne" => 0x52,
        "i64.lt_s" => 0x53,
        "i64.lt_u" => 0x54,
        "i64.gt_s" => 0x55,
        "i64.gt_u" => 0x56,
        "i64.le_s" => 0x57,
        "i64.le_u" => 0x58,
        "i64.ge_s" => 0x59,
        "i64.ge_u" => 0x5a,
        "f32.eq" => 0x5b,
        "f32.ne" => 0x5c,
        "f32.lt" => 0x5d,
        "f32.gt" => 0x5e,
        "f32.le" => 0x5f,
        "f32.ge" => 0x60,
        "f64.eq" => 0x61,
        "f64.ne" => 0x62,
        "f64.lt" => 0x63,
        "f64.gt" => 0x64,
        "f64.le" => 0x65,
        "f64.ge" => 0x66,
        "i32.add" => 0x6a,
        "i32.sub" => 0x6b,
        "i32.mul" => 0x6c,
        "i32.div_s" => 0x6d,
        "i32.div_u" => 0x6e,
        "i32.rem_s" => 0x6f,
        "i32.rem_u" => 0x70,
        "i32.and" => 0x71,
        "i32.or" => 0x72,
        "i32.xor" => 0x73,
        "i32.shl" => 0x74,
        "i32.shr_s" => 0x75,
        "i32.shr_u" => 0x76,
        "i32.rotl" => 0x77,
        "i32.rotr" => 0x78,
        "i64.add" => 0x7c,
        "i64.sub" => 0x7d,
        "i64.mul" => 0x7e,
        "i64.div_s" => 0x7f,
        "i64.div_u" => 0x80,
        "i64.rem_s" => 0x81,
        "i64.rem_u" => 0x82,
        "i64.and" => 0x83,
        "i64.or" => 0x84,
        "i64.xor" => 0x85,
        "i64.shl" => 0x86,
        "i64.shr_s" => 0x87,
        "i64.shr_u" => 0x88,
        "i64.rotl" => 0x89,
        "i64.rotr" => 0x8a,
        "i32.clz" => 0x67,
        "i32.ctz" => 0x68,
        "i32.popcnt" => 0x69,
        "i64.clz" => 0x79,
        "i64.ctz" => 0x7a,
        "i64.popcnt" => 0x7b,
        "f32.abs" => 0x8b,
        "f32.neg" => 0x8c,
        "f32.ceil" => 0x8d,
        "f32.floor" => 0x8e,
        "f32.trunc" => 0x8f,
        "f32.nearest" => 0x90,
        "f32.sqrt" => 0x91,
        "f32.add" => 0x92,
        "f32.sub" => 0x93,
        "f32.mul" => 0x94,
        "f32.div" => 0x95,
        "f32.min" => 0x96,
        "f32.max" => 0x97,
        "f32.copysign" => 0x98,
        "f64.abs" => 0x99,
        "f64.neg" => 0x9a,
        "f64.ceil" => 0x9b,
        "f64.floor" => 0x9c,
        "f64.trunc" => 0x9d,
        "f64.nearest" => 0x9e,
        "f64.sqrt" => 0x9f,
        "f64.min" => 0xa4,
        "f64.max" => 0xa5,
        "f64.copysign" => 0xa6,
        "f64.add" => 0xa0,
        "f64.sub" => 0xa1,
        "f64.mul" => 0xa2,
        "f64.div" => 0xa3,
        "i32.wrap_i64" => 0xa7,
        "i32.trunc_f32_s" => 0xa8,
        "i32.trunc_f32_u" => 0xa9,
        "i32.trunc_f64_s" => 0xaa,
        "i32.trunc_f64_u" => 0xab,
        "i64.extend_i32_s" => 0xac,
        "i64.extend_i32_u" => 0xad,
        "i64.trunc_f32_s" => 0xae,
        "i64.trunc_f32_u" => 0xaf,
        "i64.trunc_f64_s" => 0xb0,
        "i64.trunc_f64_u" => 0xb1,
        "f32.convert_i32_s" => 0xb2,
        "f32.convert_i32_u" => 0xb3,
        "f32.convert_i64_s" => 0xb4,
        "f32.convert_i64_u" => 0xb5,
        "f32.demote_f64" => 0xb6,
        "f64.convert_i32_s" => 0xb7,
        "f64.convert_i32_u" => 0xb8,
        "f64.convert_i64_s" => 0xb9,
        "f64.convert_i64_u" => 0xba,
        "f64.promote_f32" => 0xbb,
        "i32.reinterpret_f32" => 0xbc,
        "i64.reinterpret_f64" => 0xbd,
        "f32.reinterpret_i32" => 0xbe,
        "f64.reinterpret_i64" => 0xbf,
        "i32.extend8_s" => 0xc0,
        "i32.extend16_s" => 0xc1,
        "i64.extend8_s" => 0xc2,
        "i64.extend16_s" => 0xc3,
        "i64.extend32_s" => 0xc4,
        _ => return None,
    })
}

/// Whether `name` takes one immediate operand (an index or constant).
fn has_immediate(name: &str) -> bool {
    matches!(
        name,
        "local.get"
            | "local.set"
            | "local.tee"
            | "global.get"
            | "global.set"
            | "call"
            | "memory.init"
            | "data.drop"
            | "br"
            | "br_if"
            | "i32.const"
            | "i64.const"
            | "f32.const"
            | "f64.const"
    )
}

/// Emits one instruction by name with its optional immediate.
fn emit_op(
    name: &str,
    imm: Option<&str>,
    locals: &[String],
    funcs: &[String],
    globals: &[String],
    labels: &[String],
    out: &mut Vec<u8>,
) -> Result<(), String> {
    // Resolves an index immediate against a symbol table: `$name` → its position,
    // a bare number → itself.
    let resolve = |table: &[String], kind: &str| -> Result<u64, String> {
        let s = imm.ok_or_else(|| format!("{name} needs an immediate"))?;
        if let Some(stripped) = s.strip_prefix('$') {
            table
                .iter()
                .position(|n| n == stripped)
                .map(|p| p as u64)
                .ok_or_else(|| format!("unknown {kind} ${stripped}"))
        } else {
            s.parse::<u64>()
                .map_err(|_| format!("bad index for {name}"))
        }
    };
    let idx = || resolve(locals, "local");
    match name {
        "i32.const" | "i64.const" => {
            out.push(if name == "i32.const" { 0x41 } else { 0x42 });
            let n = parse_wat_int(imm.ok_or("const needs a value")?)?;
            leb_i(n, out);
        }
        "f32.const" => {
            out.push(0x43);
            let v = parse_wat_float(imm.ok_or("f32.const")?)? as f32;
            out.extend_from_slice(&v.to_le_bytes());
        }
        "f64.const" => {
            out.push(0x44);
            let v = parse_wat_float(imm.ok_or("f64.const")?)?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        "local.get" => {
            out.push(0x20);
            leb_u(idx()?, out);
        }
        "local.set" => {
            out.push(0x21);
            leb_u(idx()?, out);
        }
        "local.tee" => {
            out.push(0x22);
            leb_u(idx()?, out);
        }
        "global.get" => {
            out.push(0x23);
            leb_u(resolve(globals, "global")?, out);
        }
        "global.set" => {
            out.push(0x24);
            leb_u(resolve(globals, "global")?, out);
        }
        "call" => {
            out.push(0x10);
            leb_u(resolve(funcs, "function")?, out);
        }
        // br/br_if: a numeric label depth, or a `$name` resolved against the
        // enclosing label stack (depth 0 = innermost).
        "br" | "br_if" => {
            out.push(if name == "br" { 0x0c } else { 0x0d });
            let s = imm.ok_or("br needs a label")?;
            let n = if let Some(stripped) = s.strip_prefix('$') {
                labels
                    .iter()
                    .rev()
                    .position(|l| l == stripped)
                    .map(|d| d as u64)
                    .ok_or_else(|| format!("unknown label ${stripped}"))?
            } else {
                s.parse::<u64>().map_err(|_| "br label must be numeric")?
            };
            leb_u(n, out);
        }
        // Saturating truncations are 0xfc-prefixed (sub-opcode 0x00..=0x07).
        "i32.trunc_sat_f32_s" => out.extend_from_slice(&[0xfc, 0x00]),
        "i32.trunc_sat_f32_u" => out.extend_from_slice(&[0xfc, 0x01]),
        "i32.trunc_sat_f64_s" => out.extend_from_slice(&[0xfc, 0x02]),
        "i32.trunc_sat_f64_u" => out.extend_from_slice(&[0xfc, 0x03]),
        "i64.trunc_sat_f32_s" => out.extend_from_slice(&[0xfc, 0x04]),
        "i64.trunc_sat_f32_u" => out.extend_from_slice(&[0xfc, 0x05]),
        "i64.trunc_sat_f64_s" => out.extend_from_slice(&[0xfc, 0x06]),
        "i64.trunc_sat_f64_u" => out.extend_from_slice(&[0xfc, 0x07]),
        // Bulk-memory ops carry their reserved memory index(es) after the sub-op.
        "memory.fill" => out.extend_from_slice(&[0xfc, 0x0b, 0x00]),
        "memory.copy" => out.extend_from_slice(&[0xfc, 0x0a, 0x00, 0x00]),
        // `memory.init SEG` and `data.drop SEG` take a data-segment index immediate.
        "memory.init" => {
            out.extend_from_slice(&[0xfc, 0x08]);
            leb_u(
                parse_wat_int(imm.ok_or("memory.init needs a segment")?)? as u64,
                out,
            );
            out.push(0x00); // reserved memory index
        }
        "data.drop" => {
            out.extend_from_slice(&[0xfc, 0x09]);
            leb_u(
                parse_wat_int(imm.ok_or("data.drop needs a segment")?)? as u64,
                out,
            );
        }
        "memory.size" => {
            out.push(0x3f);
            out.push(0x00); // reserved memory index
        }
        "memory.grow" => {
            out.push(0x40);
            out.push(0x00);
        }
        _ => {
            if let Some((opcode, align)) = memory_op(name) {
                // Default memarg: natural alignment, offset 0. (`offset=`/`align=`
                // memarg atoms aren't modeled — the suite usually omits them.)
                out.push(opcode);
                leb_u(u64::from(align), out);
                leb_u(0, out);
            } else {
                out.push(simple_opcode(name).ok_or_else(|| format!("unknown instruction {name}"))?);
            }
        }
    }
    Ok(())
}

/// A memory load/store opcode and its natural alignment (log2 of the byte width).
fn memory_op(name: &str) -> Option<(u8, u8)> {
    Some(match name {
        "i32.load" => (0x28, 2),
        "i64.load" => (0x29, 3),
        "f32.load" => (0x2a, 2),
        "f64.load" => (0x2b, 3),
        "i32.load8_s" => (0x2c, 0),
        "i32.load8_u" => (0x2d, 0),
        "i32.load16_s" => (0x2e, 1),
        "i32.load16_u" => (0x2f, 1),
        "i64.load8_s" => (0x30, 0),
        "i64.load8_u" => (0x31, 0),
        "i64.load16_s" => (0x32, 1),
        "i64.load16_u" => (0x33, 1),
        "i64.load32_s" => (0x34, 2),
        "i64.load32_u" => (0x35, 2),
        "i32.store" => (0x36, 2),
        "i64.store" => (0x37, 3),
        "f32.store" => (0x38, 2),
        "f64.store" => (0x39, 3),
        "i32.store8" => (0x3a, 0),
        "i32.store16" => (0x3b, 1),
        "i64.store8" => (0x3c, 0),
        "i64.store16" => (0x3d, 1),
        "i64.store32" => (0x3e, 2),
        _ => return None,
    })
}

/// Resolves a branch label — `$name` to its depth on the label stack (0 =
/// innermost), or a bare number to itself.
fn resolve_label(s: &str, labels: &[String]) -> Result<u64, String> {
    if let Some(stripped) = s.strip_prefix('$') {
        labels
            .iter()
            .rev()
            .position(|l| l == stripped)
            .map(|d| d as u64)
            .ok_or_else(|| format!("unknown label ${stripped}"))
    } else {
        s.parse::<u64>().map_err(|_| format!("bad label {s}"))
    }
}

/// Resolves a `(type $t)` / `(type N)` annotation to a type index.
fn resolve_type(ann: Option<&Sexpr>, types: &[String]) -> Result<u64, String> {
    if let Some(Sexpr::List(p)) = ann
        && p.first() == Some(&Sexpr::Atom(String::from("type")))
    {
        return match p.get(1) {
            Some(Sexpr::Atom(s)) if s.starts_with('$') => types
                .iter()
                .position(|n| n == &s[1..])
                .map(|x| x as u64)
                .ok_or_else(|| format!("unknown type {s}")),
            Some(Sexpr::Atom(n)) => n.parse::<u64>().map_err(|_| String::from("bad type index")),
            _ => Err(String::from("malformed (type …)")),
        };
    }
    Err(String::from("call_indirect needs a (type …) annotation"))
}

/// Emits a flat sequence of instructions (each an atom — possibly consuming the
/// next atom as an immediate — or a folded `(op …)` list).
fn emit_instrs(
    items: &[Sexpr],
    locals: &[String],
    funcs: &[String],
    globals: &[String],
    types: &[String],
    labels: &mut Vec<String>,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            Sexpr::List(inner) => {
                emit_folded(inner, locals, funcs, globals, types, labels, out)?;
                i += 1;
            }
            // `call_indirect (type $t)` — the type annotation is a following list.
            Sexpr::Atom(name) if name == "call_indirect" => {
                let t = resolve_type(items.get(i + 1), types)?;
                out.push(0x11);
                leb_u(t, out);
                leb_u(0, out); // table index 0
                i += 2;
            }
            Sexpr::Atom(name) => {
                if has_immediate(name) {
                    let imm = match items.get(i + 1) {
                        Some(Sexpr::Atom(s)) => s.as_str(),
                        _ => return Err(format!("{name} needs an immediate")),
                    };
                    emit_op(name, Some(imm), locals, funcs, globals, labels, out)?;
                    i += 2;
                } else {
                    emit_op(name, None, locals, funcs, globals, labels, out)?;
                    i += 1;
                }
            }
            Sexpr::Str(_) => return Err(String::from("unexpected string in function body")),
        }
    }
    Ok(())
}

/// Reads an optional leading `$label` then an optional `(result T)` blocktype from
/// `items`, returning the label (empty if none), the blocktype byte (`0x40` =
/// empty), and how many items were consumed.
fn block_type_of(items: &[Sexpr]) -> (String, u8, usize) {
    let mut skip = 0;
    let mut label = String::new();
    if let Some(Sexpr::Atom(s)) = items.first()
        && let Some(name) = s.strip_prefix('$')
    {
        label = name.into();
        skip += 1;
    }
    if let Some(Sexpr::List(p)) = items.get(skip)
        && p.first() == Some(&Sexpr::Atom(String::from("result")))
        && let Some(Sexpr::Atom(t)) = p.get(1)
        && let Some(b) = valtype_byte(t)
    {
        return (label, b, skip + 1);
    }
    (label, 0x40, skip)
}

/// Emits a folded instruction `(op imm? operands…)`: operands first, then the op.
/// Structured control (`block`/`loop`/`if`) is folded recursively.
fn emit_folded(
    inner: &[Sexpr],
    locals: &[String],
    funcs: &[String],
    globals: &[String],
    types: &[String],
    labels: &mut Vec<String>,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let Some(Sexpr::Atom(name)) = inner.first() else {
        return Err(String::from("folded instruction needs a head opcode"));
    };
    match name.as_str() {
        "call_indirect" => {
            // (call_indirect (type $t) operand… table_index_operand)
            let t = resolve_type(inner.get(1), types)?;
            emit_instrs(&inner[2..], locals, funcs, globals, types, labels, out)?;
            out.push(0x11);
            leb_u(t, out);
            leb_u(0, out); // table index 0
            return Ok(());
        }
        "br_table" => {
            // (br_table L1 … Ln Ldefault (index-operand)): leading label atoms,
            // then a folded index operand. The last label is the default.
            let mut label_depths = Vec::new();
            let mut rest = 1;
            while let Some(Sexpr::Atom(s)) = inner.get(rest) {
                label_depths.push(resolve_label(s, labels)?);
                rest += 1;
            }
            if label_depths.is_empty() {
                return Err(String::from("br_table needs at least a default label"));
            }
            // Emit the index operand(s), then the table.
            emit_instrs(&inner[rest..], locals, funcs, globals, types, labels, out)?;
            out.push(0x0e);
            leb_u((label_depths.len() - 1) as u64, out); // count excludes default
            for d in &label_depths {
                leb_u(*d, out);
            }
            return Ok(());
        }
        "block" | "loop" => {
            let (label, bt, skip) = block_type_of(&inner[1..]);
            out.push(if name == "block" { 0x02 } else { 0x03 });
            out.push(bt);
            labels.push(label);
            emit_instrs(
                &inner[1 + skip..],
                locals,
                funcs,
                globals,
                types,
                labels,
                out,
            )?;
            labels.pop();
            out.push(0x0b); // end
            return Ok(());
        }
        "if" => {
            // (if label? (result T)? cond* (then instr*) (else instr*)?)
            let (label, bt, skip) = block_type_of(&inner[1..]);
            let rest = &inner[1 + skip..];
            let is_clause = |s: &Sexpr, kw: &str| matches!(s, Sexpr::List(p) if p.first() == Some(&Sexpr::Atom(String::from(kw))));
            let then_at = rest
                .iter()
                .position(|s| is_clause(s, "then"))
                .ok_or("if without a (then …) clause")?;
            // The condition precedes `(then …)`, emitted before the `if` (so it
            // doesn't see the if's own label).
            emit_instrs(&rest[..then_at], locals, funcs, globals, types, labels, out)?;
            out.push(0x04); // if
            out.push(bt);
            labels.push(label);
            if let Sexpr::List(then_p) = &rest[then_at] {
                emit_instrs(&then_p[1..], locals, funcs, globals, types, labels, out)?;
            }
            if let Some(else_s) = rest.get(then_at + 1)
                && is_clause(else_s, "else")
                && let Sexpr::List(else_p) = else_s
            {
                out.push(0x05); // else
                emit_instrs(&else_p[1..], locals, funcs, globals, types, labels, out)?;
            }
            labels.pop();
            out.push(0x0b); // end
            return Ok(());
        }
        _ => {}
    }
    if has_immediate(name) {
        let imm = match inner.get(1) {
            Some(Sexpr::Atom(s)) => s.clone(),
            _ => return Err(format!("{name} needs an immediate")),
        };
        emit_instrs(&inner[2..], locals, funcs, globals, types, labels, out)?;
        emit_op(name, Some(&imm), locals, funcs, globals, labels, out)?;
    } else {
        emit_instrs(&inner[1..], locals, funcs, globals, types, labels, out)?;
        emit_op(name, None, locals, funcs, globals, labels, out)?;
    }
    Ok(())
}

/// A parsed `(func …)`: signature, locals, and (after the second pass) the encoded
/// body bytes. `name`/`body_items`/`local_names` support resolving `$name`
/// references once every function's index is known.
struct WatFunc {
    name: String,
    export: Option<String>,
    params: Vec<u8>,
    results: Vec<u8>,
    locals: Vec<u8>,
    local_names: Vec<String>,
    body_items: Vec<Sexpr>,
    body: Vec<u8>,
}

/// Compiles an inline `(module (func …)…)` text module to a binary module.
fn parse_wat_module(items: &[Sexpr]) -> Result<Vec<u8>, String> {
    // Collect `(type $name? (func (param …)(result …)))` declarations: their names
    // (for `call_indirect (type $t)`) and the encoded function-type entries. These
    // occupy the *front* of the type section, before the per-function types.
    let mut type_names: Vec<String> = Vec::new();
    let mut declared_types: Vec<u8> = Vec::new(); // concatenated `0x60 …` entries
    for field in &items[1..] {
        let Sexpr::List(t) = field else { continue };
        if t.first() != Some(&Sexpr::Atom(String::from("type"))) {
            continue;
        }
        let mut k = 1;
        let mut name = String::new();
        if let Some(Sexpr::Atom(s)) = t.get(k)
            && let Some(nm) = s.strip_prefix('$')
        {
            name = nm.into();
            k += 1;
        }
        // The `(func (param …) (result …))` signature.
        let mut params = Vec::new();
        let mut results = Vec::new();
        if let Some(Sexpr::List(f)) = t.get(k) {
            for part in &f[1..] {
                if let Sexpr::List(p) = part {
                    let kw = p.first();
                    if kw == Some(&Sexpr::Atom(String::from("param"))) {
                        for ty in &p[1..] {
                            if let Sexpr::Atom(ty) = ty {
                                params.push(valtype_byte(ty).ok_or("bad param type")?);
                            }
                        }
                    } else if kw == Some(&Sexpr::Atom(String::from("result"))) {
                        for ty in &p[1..] {
                            if let Sexpr::Atom(ty) = ty {
                                results.push(valtype_byte(ty).ok_or("bad result type")?);
                            }
                        }
                    }
                }
            }
        }
        declared_types.push(0x60);
        leb_u(params.len() as u64, &mut declared_types);
        declared_types.extend_from_slice(&params);
        leb_u(results.len() as u64, &mut declared_types);
        declared_types.extend_from_slice(&results);
        type_names.push(name);
    }
    let n_decl_types = type_names.len();

    // First, collect `(global $name? (mut? T) (init))` declarations: their names
    // (for `global.get/set $g`) and the encoded global-section entries.
    let mut global_names: Vec<String> = Vec::new();
    let mut global_section: Vec<u8> = Vec::new();
    for field in &items[1..] {
        let Sexpr::List(g) = field else { continue };
        if g.first() != Some(&Sexpr::Atom(String::from("global"))) {
            continue;
        }
        let mut k = 1;
        let mut name = String::new();
        if let Some(Sexpr::Atom(s)) = g.get(k)
            && let Some(nm) = s.strip_prefix('$')
        {
            name = nm.into();
            k += 1;
        }
        // The type: `T` (immutable) or `(mut T)`.
        let (vt, is_mut) = match g.get(k) {
            Some(Sexpr::Atom(t)) => (valtype_byte(t), false),
            Some(Sexpr::List(m)) if m.first() == Some(&Sexpr::Atom(String::from("mut"))) => (
                m.get(1).and_then(|x| match x {
                    Sexpr::Atom(t) => valtype_byte(t),
                    _ => None,
                }),
                true,
            ),
            _ => (None, false),
        };
        let vt = vt.ok_or("bad global type")?;
        k += 1;
        global_section.push(vt);
        global_section.push(u8::from(is_mut));
        // Init expression: a single folded constant `(T.const N)`.
        if let Some(Sexpr::List(init)) = g.get(k)
            && let (Some(Sexpr::Atom(op)), Some(Sexpr::Atom(n))) = (init.first(), init.get(1))
        {
            emit_op(op, Some(n), &[], &[], &[], &[], &mut global_section)?;
        }
        global_section.push(0x0b); // end of init expr
        global_names.push(name);
    }
    let n_globals = global_names.len();

    // Collect `(table N funcref)` (at most one in the MVP).
    let mut table: Option<(u32, Option<u32>)> = None;
    for field in &items[1..] {
        if let Sexpr::List(t) = field
            && t.first() == Some(&Sexpr::Atom(String::from("table")))
        {
            let nums: Vec<u32> = t[1..]
                .iter()
                .filter_map(|s| match s {
                    Sexpr::Atom(n) => n.parse::<u32>().ok(),
                    _ => None,
                })
                .collect();
            if let Some(&min) = nums.first() {
                table = Some((min, nums.get(1).copied()));
            }
        }
    }

    // Collect `(memory N M?)` (at most one in the MVP) and `(data (offset) "…")`.
    let mut memory: Option<(u32, Option<u32>)> = None;
    let mut data_segments: Vec<(Vec<u8>, Vec<u8>)> = Vec::new(); // (offset-expr, bytes)
    for field in &items[1..] {
        let Sexpr::List(d) = field else { continue };
        match d.first() {
            Some(Sexpr::Atom(a)) if a == "memory" => {
                // skip an optional `$name`
                let nums: Vec<u32> = d[1..]
                    .iter()
                    .filter_map(|s| match s {
                        Sexpr::Atom(n) => n.parse::<u32>().ok(),
                        _ => None,
                    })
                    .collect();
                if let Some(&min) = nums.first() {
                    memory = Some((min, nums.get(1).copied()));
                }
            }
            Some(Sexpr::Atom(a)) if a == "data" => {
                // Active `(data (i32.const OFF) "bytes" …)` — an offset expr then the
                // byte strings; or passive `(data "bytes" …)` — bytes only.
                let mut offset = Vec::new();
                let bytes_from = if let Some(Sexpr::List(off)) = d.get(1) {
                    if let (Some(Sexpr::Atom(op)), Some(Sexpr::Atom(n))) = (off.first(), off.get(1))
                    {
                        emit_op(op, Some(n), &[], &[], &[], &[], &mut offset)?;
                    }
                    offset.push(0x0b);
                    2
                } else {
                    1 // passive: no offset expression
                };
                let mut bytes = Vec::new();
                for it in &d[bytes_from..] {
                    if let Sexpr::Str(s) = it {
                        bytes.extend_from_slice(s);
                    }
                }
                // `offset` is empty for a passive segment (mode 1).
                data_segments.push((offset, bytes));
            }
            _ => {}
        }
    }

    let mut funcs: Vec<WatFunc> = Vec::new();
    for field in &items[1..] {
        let Sexpr::List(f) = field else { continue };
        if f.first() != Some(&Sexpr::Atom(String::from("func"))) {
            continue;
        }
        let mut wf = WatFunc {
            name: String::new(),
            export: None,
            params: Vec::new(),
            results: Vec::new(),
            locals: Vec::new(),
            local_names: Vec::new(),
            body_items: Vec::new(),
            body: Vec::new(),
        };
        let mut body_items: Vec<Sexpr> = Vec::new();
        // Local symbol table (params then declared locals), in index order; an
        // unnamed slot is the empty string. `(param $x i32)` is one named param;
        // `(param i32 i32)` is two unnamed params (same for `(local …)`).
        let mut local_names: Vec<String> = Vec::new();
        let decl =
            |p: &[Sexpr], types: &mut Vec<u8>, names: &mut Vec<String>| -> Result<(), String> {
                if let Some(Sexpr::Atom(first)) = p.first()
                    && let Some(nm) = first.strip_prefix('$')
                {
                    // Named: `($name type)` — exactly one entry.
                    let t = p.get(1).and_then(|s| match s {
                        Sexpr::Atom(t) => valtype_byte(t),
                        _ => None,
                    });
                    types.push(t.ok_or("bad named type")?);
                    names.push(nm.into());
                    return Ok(());
                }
                for t in p {
                    if let Sexpr::Atom(t) = t {
                        types.push(valtype_byte(t).ok_or("bad type")?);
                        names.push(String::new());
                    }
                }
                Ok(())
            };
        for part in &f[1..] {
            match part {
                Sexpr::List(p) => match p.first() {
                    Some(Sexpr::Atom(a)) if a == "export" => {
                        if let Some(Sexpr::Str(s)) = p.get(1) {
                            wf.export = Some(String::from_utf8_lossy(s).into_owned());
                        }
                    }
                    Some(Sexpr::Atom(a)) if a == "param" => {
                        decl(&p[1..], &mut wf.params, &mut local_names)?;
                    }
                    Some(Sexpr::Atom(a)) if a == "result" => {
                        for t in &p[1..] {
                            if let Sexpr::Atom(t) = t {
                                wf.results.push(valtype_byte(t).ok_or("bad result type")?);
                            }
                        }
                    }
                    Some(Sexpr::Atom(a)) if a == "local" => {
                        decl(&p[1..], &mut wf.locals, &mut local_names)?;
                    }
                    // A folded instruction.
                    _ => body_items.push(part.clone()),
                },
                // A leading `$name` is the function's own name; record it (for
                // `call $name`). Any other atom is a flat instruction.
                Sexpr::Atom(s) if s.starts_with('$') => {
                    if wf.name.is_empty() {
                        wf.name = s[1..].into();
                    }
                }
                Sexpr::Atom(_) => body_items.push(part.clone()),
                Sexpr::Str(_) => {}
            }
        }
        wf.local_names = local_names;
        wf.body_items = body_items;
        funcs.push(wf);
    }
    // Second pass: now that every function's index is known, build the function
    // symbol table and emit each body (so `call $name` resolves).
    let func_names: Vec<String> = funcs.iter().map(|f| f.name.clone()).collect();
    let resolve_func = |r: &str| -> Result<u32, String> {
        if let Some(nm) = r.strip_prefix('$') {
            func_names
                .iter()
                .position(|n| n == nm)
                .map(|p| p as u32)
                .ok_or_else(|| format!("unknown function ${nm}"))
        } else {
            r.parse::<u32>()
                .map_err(|_| String::from("bad function index"))
        }
    };
    // Collect `(elem (i32.const off) $f1 $f2 …)` segments (table 0).
    let mut elements: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
    for field in &items[1..] {
        let Sexpr::List(e) = field else { continue };
        if e.first() != Some(&Sexpr::Atom(String::from("elem"))) {
            continue;
        }
        let mut offset = Vec::new();
        let mut rest = 1;
        if let Some(Sexpr::List(off)) = e.get(1)
            && let (Some(Sexpr::Atom(op)), Some(Sexpr::Atom(n))) = (off.first(), off.get(1))
        {
            emit_op(op, Some(n), &[], &[], &[], &[], &mut offset)?;
            rest = 2;
        }
        offset.push(0x0b);
        let mut indices = Vec::new();
        for it in &e[rest..] {
            if let Sexpr::Atom(r) = it {
                indices.push(resolve_func(r)?);
            }
        }
        elements.push((offset, indices));
    }
    // A `(start $f)` declaration: the function run at instantiation.
    let mut start_func: Option<u32> = None;
    for field in &items[1..] {
        if let Sexpr::List(s) = field
            && s.first() == Some(&Sexpr::Atom(String::from("start")))
            && let Some(Sexpr::Atom(r)) = s.get(1)
        {
            let idx = if let Some(nm) = r.strip_prefix('$') {
                func_names
                    .iter()
                    .position(|n| n == nm)
                    .map(|p| p as u32)
                    .ok_or_else(|| format!("unknown start function ${nm}"))?
            } else {
                r.parse::<u32>().map_err(|_| "bad start index")?
            };
            start_func = Some(idx);
        }
    }
    for f in &mut funcs {
        let mut body = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        emit_instrs(
            &f.body_items,
            &f.local_names,
            &func_names,
            &global_names,
            &type_names,
            &mut labels,
            &mut body,
        )?;
        f.body = body;
    }

    // Assemble the binary sections.
    let mut out = alloc::vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    // type section: the declared `(type …)` entries first, then one signature per
    // function (no dedup, which is valid). `call_indirect` references the former.
    let mut types = Vec::new();
    leb_u((n_decl_types + funcs.len()) as u64, &mut types);
    types.extend_from_slice(&declared_types);
    for f in &funcs {
        types.push(0x60);
        leb_u(f.params.len() as u64, &mut types);
        types.extend_from_slice(&f.params);
        leb_u(f.results.len() as u64, &mut types);
        types.extend_from_slice(&f.results);
    }
    section(1, &types, &mut out);
    // function section: function `i`'s type is the `(n_decl_types + i)`-th entry.
    let mut fns = Vec::new();
    leb_u(funcs.len() as u64, &mut fns);
    for i in 0..funcs.len() {
        leb_u((n_decl_types + i) as u64, &mut fns);
    }
    section(3, &fns, &mut out);
    // table section (id 4): a single funcref table's limits.
    if let Some((min, max)) = table {
        let mut tbl = Vec::new();
        leb_u(1, &mut tbl); // one table
        tbl.push(0x70); // funcref
        match max {
            Some(m) => {
                tbl.push(0x01);
                leb_u(u64::from(min), &mut tbl);
                leb_u(u64::from(m), &mut tbl);
            }
            None => {
                tbl.push(0x00);
                leb_u(u64::from(min), &mut tbl);
            }
        }
        section(4, &tbl, &mut out);
    }
    // memory section (id 5): a single memory's limits.
    if let Some((min, max)) = memory {
        let mut mem = Vec::new();
        leb_u(1, &mut mem); // one memory
        match max {
            Some(m) => {
                mem.push(0x01); // has-max flag
                leb_u(u64::from(min), &mut mem);
                leb_u(u64::from(m), &mut mem);
            }
            None => {
                mem.push(0x00);
                leb_u(u64::from(min), &mut mem);
            }
        }
        section(5, &mem, &mut out);
    }
    // global section (id 6): count + the collected per-global entries.
    if n_globals > 0 {
        let mut globals = Vec::new();
        leb_u(n_globals as u64, &mut globals);
        globals.extend_from_slice(&global_section);
        section(6, &globals, &mut out);
    }
    // export section.
    let exported: Vec<(usize, &String)> = funcs
        .iter()
        .enumerate()
        .filter_map(|(i, f)| f.export.as_ref().map(|n| (i, n)))
        .collect();
    let mut exp = Vec::new();
    leb_u(exported.len() as u64, &mut exp);
    for (i, name) in &exported {
        leb_u(name.len() as u64, &mut exp);
        exp.extend_from_slice(name.as_bytes());
        exp.push(0x00); // func export
        leb_u(*i as u64, &mut exp);
    }
    section(7, &exp, &mut out);
    // start section (id 8): the function index run at instantiation.
    if let Some(idx) = start_func {
        let mut s = Vec::new();
        leb_u(u64::from(idx), &mut s);
        section(8, &s, &mut out);
    }
    // element section (id 9): each segment populates table 0 from an offset.
    if !elements.is_empty() {
        let mut elem = Vec::new();
        leb_u(elements.len() as u64, &mut elem);
        for (offset, indices) in &elements {
            leb_u(0, &mut elem); // table index 0
            elem.extend_from_slice(offset);
            leb_u(indices.len() as u64, &mut elem);
            for idx in indices {
                leb_u(u64::from(*idx), &mut elem);
            }
        }
        section(9, &elem, &mut out);
    }
    // code section.
    let mut code = Vec::new();
    leb_u(funcs.len() as u64, &mut code);
    for f in &funcs {
        let mut b = Vec::new();
        leb_u(f.locals.len() as u64, &mut b); // one run per local
        for t in &f.locals {
            leb_u(1, &mut b);
            b.push(*t);
        }
        b.extend_from_slice(&f.body);
        b.push(0x0b); // end
        leb_u(b.len() as u64, &mut code);
        code.extend_from_slice(&b);
    }
    section(10, &code, &mut out);
    // data section (id 11): active `(0x00, offset-expr, bytes)` or, when the offset
    // is absent, passive `(0x01, bytes)`.
    if !data_segments.is_empty() {
        let mut data = Vec::new();
        leb_u(data_segments.len() as u64, &mut data);
        for (offset, bytes) in &data_segments {
            if offset.is_empty() {
                data.push(0x01); // passive
            } else {
                data.push(0x00); // active, memory 0
                data.extend_from_slice(offset);
            }
            leb_u(bytes.len() as u64, &mut data);
            data.extend_from_slice(bytes);
        }
        section(11, &data, &mut out);
    }
    Ok(out)
}

/// Compiles a WAT text module — `(module (func …)…)` — to a binary WebAssembly
/// module. Supports func definitions with `(export …)`, `(param …)`,
/// `(result …)`, `(local …)`, and a flat/folded instruction body over the
/// common opcode set.
///
/// # Errors
/// A parse error, or an unknown instruction / type.
pub fn wat_to_binary(src: &str) -> Result<Vec<u8>, String> {
    let exprs = parse_sexprs(src)?;
    for e in &exprs {
        if let Sexpr::List(items) = e
            && items.first() == Some(&Sexpr::Atom(String::from("module")))
        {
            return parse_wat_module(items);
        }
    }
    Err(String::from("no (module …) found"))
}

/// Appends section `id` with `content` (length-prefixed) to `out`.
fn section(id: u8, content: &[u8], out: &mut Vec<u8>) {
    out.push(id);
    leb_u(content.len() as u64, out);
    out.extend_from_slice(content);
}

/// Parses a `(TYPE.const N)` value expression into a [`Val`].
/// Parses a WAT integer literal: optional sign, optional `0x` hex, and `_`
/// digit separators (e.g. `-0x8000_0000`, `4_294_967_295`). The value is parsed
/// wide (`i128`) then truncated to the i64 the const opcode stores.
fn parse_wat_int(s: &str) -> Result<i64, String> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let clean = body.replace('_', "");
    let mag = if let Some(hex) = clean
        .strip_prefix("0x")
        .or_else(|| clean.strip_prefix("0X"))
    {
        u128::from_str_radix(hex, 16).map_err(|_| format!("bad integer {s}"))?
    } else {
        clean
            .parse::<u128>()
            .map_err(|_| format!("bad integer {s}"))?
    };
    // Wrap into i64 range (wast allows both signed and unsigned spellings).
    Ok(if neg {
        (mag as i128).wrapping_neg() as i64
    } else {
        mag as i64
    })
}

/// Parses a WAT float literal: the special spellings `inf`/`-inf`/`+inf` and the
/// suite's `nan` family (`nan`, `±nan`, `nan:canonical`/`:arithmetic`/`:0x…` — all
/// just a NaN here, matched leniently by `vals_match`), hex floats (`0x1.8p1`),
/// and ordinary decimal (incl. `_` separators) via the standard parser.
fn parse_wat_float(s: &str) -> Result<f64, String> {
    match s {
        _ if s.starts_with("-nan") => return Ok(-f64::NAN),
        _ if s.starts_with("nan") || s.starts_with("+nan") => return Ok(f64::NAN),
        "inf" | "+inf" => return Ok(f64::INFINITY),
        "-inf" => return Ok(f64::NEG_INFINITY),
        _ => {}
    }
    let clean = s.replace('_', "");
    if let Some(v) = parse_hex_float(&clean) {
        return Ok(v);
    }
    clean.parse::<f64>().map_err(|_| format!("bad float {s}"))
}

/// Parses a hex float `±0x<int>.<frac>p±<exp>` (the `.frac`/`pexp` parts optional),
/// returning `None` if `s` is not hex-float syntax. Power-of-two scaling is done by
/// repeated multiply/halve (no `f64::powi`, which is unavailable in `no_std`).
fn parse_hex_float(s: &str) -> Option<f64> {
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let body = rest
        .strip_prefix("0x")
        .or_else(|| rest.strip_prefix("0X"))?;
    let (mant, exp) = match body.find(['p', 'P']) {
        Some(i) => (&body[..i], body[i + 1..].parse::<i32>().ok()?),
        None => (body, 0),
    };
    let (int_part, frac_part) = match mant.find('.') {
        Some(i) => (&mant[..i], &mant[i + 1..]),
        None => (mant, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    let mut value = 0.0f64;
    for c in int_part.chars() {
        value = value * 16.0 + f64::from(c.to_digit(16)?);
    }
    let mut scale = 1.0 / 16.0;
    for c in frac_part.chars() {
        value += f64::from(c.to_digit(16)?) * scale;
        scale /= 16.0;
    }
    let mut e = exp;
    while e > 0 {
        value *= 2.0;
        e -= 1;
    }
    while e < 0 {
        value *= 0.5;
        e += 1;
    }
    Some(if neg { -value } else { value })
}

fn parse_const(s: &Sexpr) -> Result<Val, String> {
    let Sexpr::List(items) = s else {
        return Err(String::from("expected (type.const N)"));
    };
    let (Some(Sexpr::Atom(ty)), Some(Sexpr::Atom(n))) = (items.first(), items.get(1)) else {
        return Err(String::from("malformed const"));
    };
    Ok(match ty.as_str() {
        "i32.const" => Val::I32(parse_wat_int(n)? as i32),
        "i64.const" => Val::I64(parse_wat_int(n)?),
        "f32.const" => Val::F32(parse_wat_float(n)? as f32),
        "f64.const" => Val::F64(parse_wat_float(n)?),
        other => return Err(format!("unknown const type {other}")),
    })
}

/// Extracts `(invoke "name" (const)…)` into `(name, args)`.
fn parse_invoke(s: &Sexpr) -> Result<(String, Vec<Val>), String> {
    let Sexpr::List(items) = s else {
        return Err(String::from("expected (invoke …)"));
    };
    if items.first() != Some(&Sexpr::Atom(String::from("invoke"))) {
        return Err(String::from("expected invoke"));
    }
    let Some(Sexpr::Str(name)) = items.get(1) else {
        return Err(String::from("invoke needs a name"));
    };
    let name = String::from_utf8(name.clone()).map_err(|_| "bad invoke name")?;
    let args = items[2..]
        .iter()
        .map(parse_const)
        .collect::<Result<_, _>>()?;
    Ok((name, args))
}

/// Runs a `.wast` script against the engine: each `(module binary …)` is
/// instantiated and the following `assert_return`/`assert_trap`/`invoke` commands
/// run against it; `assert_invalid` checks a module is rejected. Returns the
/// number of commands executed.
///
/// # Errors
/// A parse error, or the first failing assertion (with a diagnostic).
pub fn run_wast(src: &str) -> Result<usize, String> {
    let cmds = parse_sexprs(src)?;
    let mut executed = 0;
    // The current module's bytes + its deferred assertions.
    let mut cur: Option<Vec<u8>> = None;
    let mut batch: Vec<Assertion> = Vec::new();

    // Flush the current module + its assertions through the harness.
    fn flush(cur: &mut Option<Vec<u8>>, batch: &mut Vec<Assertion>) -> Result<usize, String> {
        if let Some(bytes) = cur.take() {
            let n = run_assertions(&bytes, batch)?;
            batch.clear();
            return Ok(n);
        }
        batch.clear();
        Ok(0)
    }

    for cmd in &cmds {
        let Sexpr::List(items) = cmd else { continue };
        let head = match items.first() {
            Some(Sexpr::Atom(a)) => a.as_str(),
            _ => continue,
        };
        match head {
            "module" => {
                executed += flush(&mut cur, &mut batch)?;
                cur = Some(
                    if items.get(1) == Some(&Sexpr::Atom(String::from("binary"))) {
                        // (module binary "…" "…") — concatenate the byte strings.
                        let mut bytes = Vec::new();
                        for it in &items[2..] {
                            if let Sexpr::Str(s) = it {
                                bytes.extend_from_slice(s);
                            }
                        }
                        bytes
                    } else {
                        // (module (func …)…) — compile the text module to binary.
                        parse_wat_module(items)?
                    },
                );
            }
            "assert_return" => {
                let (func, args) = parse_invoke(items.get(1).ok_or("assert_return needs invoke")?)?;
                let expect = items[2..]
                    .iter()
                    .map(parse_const)
                    .collect::<Result<_, _>>()?;
                // `func` is parsed into an owned String, but `Assertion::Return`
                // wants `&'static str`; leak it (test/conformance harness only).
                let func: &'static str = alloc::boxed::Box::leak(func.into_boxed_str());
                batch.push(Assertion::Return { func, args, expect });
            }
            "assert_trap" => {
                let (func, args) = parse_invoke(items.get(1).ok_or("assert_trap needs invoke")?)?;
                let func: &'static str = alloc::boxed::Box::leak(func.into_boxed_str());
                batch.push(Assertion::Trap { func, args });
            }
            "invoke" => {
                // A bare `invoke` runs in sequence with the batch's assertions, so
                // its side effects (global/memory mutations) are visible to later
                // asserts on the same instance.
                let (func, args) = parse_invoke(cmd)?;
                let func: &'static str = alloc::boxed::Box::leak(func.into_boxed_str());
                batch.push(Assertion::Invoke { func, args });
            }
            "assert_invalid" | "assert_malformed" => {
                executed += flush(&mut cur, &mut batch)?;
                if let Some(Sexpr::List(m)) = items.get(1) {
                    let mut bytes = Vec::new();
                    for it in &m[2..] {
                        if let Sexpr::Str(s) = it {
                            bytes.extend_from_slice(s);
                        }
                    }
                    assert_invalid(&bytes)?;
                    executed += 1;
                }
            }
            _ => {}
        }
    }
    executed += flush(&mut cur, &mut batch)?;
    Ok(executed)
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

    /// Formats module bytes as a `.wast` `(module binary "\NN…")` string.
    fn wast_binary(bytes: &[u8]) -> String {
        let mut s = String::from("(module binary \"");
        for b in bytes {
            s.push_str(&format!("\\{b:02x}"));
        }
        s.push_str("\")");
        s
    }

    #[test]
    fn parses_and_runs_a_wast_script() {
        let add = binop_module(&[0x6a]); // i32.add as "op"
        let div = binop_module(&[0x6d]); // i32.div_s as "op"
        // A real .wast script: two modules, each with assertions + a comment.
        let script = format!(
            ";; an addition module\n{}\n\
             (assert_return (invoke \"op\" (i32.const 20) (i32.const 22)) (i32.const 42))\n\
             (assert_return (invoke \"op\" (i32.const -5) (i32.const 5)) (i32.const 0))\n\
             ;; a division module\n{}\n\
             (assert_return (invoke \"op\" (i32.const 20) (i32.const 4)) (i32.const 5))\n\
             (assert_trap (invoke \"op\" (i32.const 1) (i32.const 0)) \"integer divide by zero\")\n\
             (assert_invalid (module binary \"\\00\\00\\00\\00\") \"bad magic\")",
            wast_binary(&add),
            wast_binary(&div),
        );
        let n = run_wast(&script).expect("wast script passes");
        assert_eq!(
            n, 5,
            "4 assert_returns/traps across two modules + 1 assert_invalid"
        );
    }

    #[test]
    fn wast_runner_surfaces_a_failing_assertion() {
        let add = binop_module(&[0x6a]);
        let script = format!(
            "{}\n(assert_return (invoke \"op\" (i32.const 1) (i32.const 1)) (i32.const 99))",
            wast_binary(&add),
        );
        let err = run_wast(&script).unwrap_err();
        assert!(err.contains("expected"), "diagnostic: {err}");
    }

    #[test]
    fn wast_parser_handles_strings_and_comments() {
        // The S-expression parser tokenizes nested lists, quoted byte strings with
        // \NN escapes, line and block comments.
        let exprs = parse_sexprs("(a (b \"\\01\\02\") ;; line\n (; block ;) c)").unwrap();
        assert_eq!(exprs.len(), 1);
        if let Sexpr::List(items) = &exprs[0] {
            assert_eq!(items[0], Sexpr::Atom(String::from("a")));
            if let Sexpr::List(inner) = &items[1] {
                assert_eq!(inner[1], Sexpr::Str(vec![1, 2]));
            } else {
                panic!("expected inner list");
            }
            assert_eq!(items[2], Sexpr::Atom(String::from("c")));
        } else {
            panic!("expected a list");
        }
    }

    #[test]
    fn wat_memory_and_data_compile_and_run() {
        // 1-page memory with a data segment writing 0x44332211 (LE) at offset 0;
        // `load` reads it back, `rt` stores then reloads an arg.
        let src = "(module \
            (memory 1) \
            (data (i32.const 0) \"\\11\\22\\33\\44\") \
            (func (export \"load\") (result i32) (i32.load (i32.const 0))) \
            (func (export \"rt\") (param $v i32) (result i32) \
              (i32.store (i32.const 8) (local.get $v)) \
              (i32.load (i32.const 8))))";
        let bin = wat_to_binary(src).expect("compile WAT with memory + data");
        let module = crate::wasm_rt::Module::decode(&bin).expect("decode");
        let mut inst = crate::wasm_rt::Instance::new(&module).expect("instantiate");
        // The data segment little-endian bytes form 0x44332211.
        assert_eq!(
            inst.call_export("load", &[]).unwrap(),
            vec![Val::I32(0x4433_2211)]
        );
        // store/reload round-trips an arbitrary value.
        assert_eq!(
            inst.call_export("rt", &[Val::I32(-12345)]).unwrap(),
            vec![Val::I32(-12345)]
        );
    }

    #[test]
    fn wat_call_indirect_through_table() {
        // A type, a table of 2 functions, and `dispatch(i, x)` calling table[i](x).
        let src = "(module \
            (type $unop (func (param i32) (result i32))) \
            (table 2 funcref) \
            (func $double (param $x i32) (result i32) (i32.mul (local.get $x) (i32.const 2))) \
            (func $inc (param $x i32) (result i32) (i32.add (local.get $x) (i32.const 1))) \
            (elem (i32.const 0) $double $inc) \
            (func (export \"dispatch\") (param $i i32) (param $x i32) (result i32) \
              (call_indirect (type $unop) (local.get $x) (local.get $i))))";
        let bin = wat_to_binary(src).expect("compile WAT with call_indirect");
        let module = crate::wasm_rt::Module::decode(&bin).expect("decode");
        let mut inst = crate::wasm_rt::Instance::new(&module).expect("instantiate");
        // table[0] = $double, table[1] = $inc.
        assert_eq!(
            inst.call_export("dispatch", &[Val::I32(0), Val::I32(21)])
                .unwrap(),
            vec![Val::I32(42)]
        );
        assert_eq!(
            inst.call_export("dispatch", &[Val::I32(1), Val::I32(41)])
                .unwrap(),
            vec![Val::I32(42)]
        );
    }

    #[test]
    fn wat_start_function_runs_at_instantiation() {
        // `(start $init)` runs `$init` at instantiation, setting `$g` to 77;
        // `get` returns it without any explicit call.
        let src = "(module \
            (global $g (mut i32) (i32.const 0)) \
            (func $init (global.set $g (i32.const 77))) \
            (func (export \"get\") (result i32) (global.get $g)) \
            (start $init))";
        let bin = wat_to_binary(src).expect("compile WAT with start");
        let module = crate::wasm_rt::Module::decode(&bin).expect("decode");
        let mut inst = crate::wasm_rt::Instance::new(&module).expect("instantiate");
        assert_eq!(inst.call_export("get", &[]).unwrap(), vec![Val::I32(77)]);
    }

    #[test]
    fn wat_globals_compile_and_run() {
        // A mutable global `$counter` (init 100) and an immutable `$base` (init 5):
        // bump(d) adds d to $counter and returns it + $base.
        let src = "(module \
            (global $counter (mut i32) (i32.const 100)) \
            (global $base i32 (i32.const 5)) \
            (func (export \"bump\") (param $d i32) (result i32) \
              (global.set $counter (i32.add (global.get $counter) (local.get $d))) \
              (i32.add (global.get $counter) (global.get $base))))";
        let bin = wat_to_binary(src).expect("compile WAT with globals");
        let module = crate::wasm_rt::Module::decode(&bin).expect("decode");
        let mut inst = crate::wasm_rt::Instance::new(&module).expect("instantiate");
        // Globals persist across calls on a persistent instance:
        // 100+10 + 5 = 115, then 110+20 + 5 = 135.
        assert_eq!(
            inst.call_export("bump", &[Val::I32(10)]).unwrap(),
            vec![Val::I32(115)]
        );
        assert_eq!(
            inst.call_export("bump", &[Val::I32(20)]).unwrap(),
            vec![Val::I32(135)]
        );
        // An unknown global name is rejected.
        assert!(
            wat_to_binary("(module (func (result i32) (global.get $nope)))").is_err(),
            "unknown $global rejected"
        );
    }

    #[test]
    fn wat_named_identifiers_resolve() {
        // Named function, params, and a local — the canonical upstream WAT style.
        let src = "(module (func $add (export \"add\") (param $a i32) (param $b i32) (result i32) \
                   (local $sum i32) \
                   (local.set $sum (i32.add (local.get $a) (local.get $b))) \
                   (local.get $sum)))";
        let bin = wat_to_binary(src).expect("compile named WAT");
        let r = crate::wasm_rt::Module::decode(&bin)
            .unwrap()
            .call(0, &[Val::I32(20), Val::I32(22)])
            .unwrap();
        assert_eq!(r, vec![Val::I32(42)]);
        // It compiles to the same binary as the index-based form.
        let indexed = "(module (func (export \"add\") (param i32 i32) (result i32) (local i32) \
                       (local.set 2 (i32.add (local.get 0) (local.get 1))) (local.get 2)))";
        assert_eq!(bin, wat_to_binary(indexed).unwrap(), "named ≡ indexed");
        // An unknown local name is rejected.
        assert!(
            wat_to_binary("(module (func (result i32) (local.get $nope)))").is_err(),
            "unknown $local rejected"
        );
    }

    #[test]
    fn wat_structured_control_flow() {
        // Folded `if`: abs via (if (result i32) cond (then …) (else …)).
        let abs = "(module (func (export \"abs\") (param $x i32) (result i32) \
                   (if (result i32) (i32.lt_s (local.get $x) (i32.const 0)) \
                     (then (i32.sub (i32.const 0) (local.get $x))) \
                     (else (local.get $x)))))";
        let m = crate::wasm_rt::Module::decode(&wat_to_binary(abs).expect("compile if")).unwrap();
        assert_eq!(m.call(0, &[Val::I32(-7)]).unwrap(), vec![Val::I32(7)]);
        assert_eq!(m.call(0, &[Val::I32(5)]).unwrap(), vec![Val::I32(5)]);

        // A `loop` with a *named* label `br_if $loop`: sum 1..=n.
        let sum = "(module (func (export \"sum\") (param $n i32) (result i32) \
                   (local $i i32) (local $sum i32) \
                   (loop $loop \
                     (local.set $i (i32.add (local.get $i) (i32.const 1))) \
                     (local.set $sum (i32.add (local.get $sum) (local.get $i))) \
                     (br_if $loop (i32.lt_s (local.get $i) (local.get $n)))) \
                   (local.get $sum)))";
        let m2 =
            crate::wasm_rt::Module::decode(&wat_to_binary(sum).expect("compile loop")).unwrap();
        assert_eq!(m2.call(0, &[Val::I32(5)]).unwrap(), vec![Val::I32(15)]); // 1+2+3+4+5
        assert_eq!(m2.call(0, &[Val::I32(10)]).unwrap(), vec![Val::I32(55)]);
        // The named `$loop` compiles to the same binary as the numeric depth `0`.
        let numeric = sum
            .replace("loop $loop", "loop")
            .replace("br_if $loop", "br_if 0");
        assert_eq!(
            wat_to_binary(&numeric).unwrap(),
            wat_to_binary(sum).unwrap(),
            "named label ≡ numeric depth 0"
        );
        // A branch to an *outer* named block from inside a loop (depth 1) compiles.
        let outer = "(module (func (export \"f\") (result i32) \
                     (block $done (result i32) \
                       (loop $l (br $done (i32.const 7))) )))";
        assert!(wat_to_binary(outer).is_ok(), "nested named labels compile");
    }

    #[test]
    fn wat_call_resolves_function_names() {
        // `$caller` calls `$callee` by name; the order is declaration order.
        let src = "(module \
                   (func $callee (param $x i32) (result i32) (i32.mul (local.get $x) (i32.const 3))) \
                   (func $caller (export \"run\") (param $x i32) (result i32) \
                     (i32.add (call $callee (local.get $x)) (local.get $x))))";
        let bin = wat_to_binary(src).expect("compile call-by-name WAT");
        // run(x) = callee(x) + x = 3x + x = 4x.
        let r = crate::wasm_rt::Module::decode(&bin)
            .unwrap()
            .call(1, &[Val::I32(10)])
            .unwrap();
        assert_eq!(r, vec![Val::I32(40)]);
        // An unknown function name is rejected.
        assert!(
            wat_to_binary("(module (func (call $nope)))").is_err(),
            "unknown $function rejected"
        );
    }

    #[test]
    fn wat_text_module_compiles_and_runs() {
        // Folded form.
        let folded = "(module (func (export \"add\") (param i32 i32) (result i32) \
                      (i32.add (local.get 0) (local.get 1))))";
        let bin = wat_to_binary(folded).expect("compile folded WAT");
        let r = crate::wasm_rt::Module::decode(&bin)
            .unwrap()
            .call(0, &[Val::I32(20), Val::I32(22)])
            .unwrap();
        assert_eq!(r, vec![Val::I32(42)]);

        // Flat (stack) form — same function.
        let flat = "(module (func (export \"add\") (param i32 i32) (result i32) \
                    local.get 0 local.get 1 i32.add))";
        let bin2 = wat_to_binary(flat).expect("compile flat WAT");
        assert_eq!(bin, bin2, "folded and flat compile to the same binary");

        // A function with a local and a const.
        let withlocal = "(module (func (export \"f\") (param i32) (result i32) (local i32) \
                         (local.set 1 (i32.mul (local.get 0) (i32.const 3))) \
                         (i32.add (local.get 1) (i32.const 1))))";
        let bin3 = wat_to_binary(withlocal).expect("compile local WAT");
        let r3 = crate::wasm_rt::Module::decode(&bin3)
            .unwrap()
            .call(0, &[Val::I32(5)])
            .unwrap();
        assert_eq!(r3, vec![Val::I32(16)]); // 5*3 + 1
    }

    #[test]
    fn spec_numeric_edge_cases() {
        // The fiddly numeric corners upstream iNN/fNN.wast probe: shift-count
        // masking, rotate, clz/ctz/popcnt, signed remainder, and NaN comparisons.
        let script = "(module \
            (func (export \"shl\")  (param i32 i32) (result i32) (i32.shl (local.get 0) (local.get 1))) \
            (func (export \"rotl\") (param i32 i32) (result i32) (i32.rotl (local.get 0) (local.get 1))) \
            (func (export \"clz\")  (param i32) (result i32) (i32.clz (local.get 0))) \
            (func (export \"ctz\")  (param i32) (result i32) (i32.ctz (local.get 0))) \
            (func (export \"pop\")  (param i32) (result i32) (i32.popcnt (local.get 0))) \
            (func (export \"rems\") (param i32 i32) (result i32) (i32.rem_s (local.get 0) (local.get 1))) \
            (func (export \"nanlt\")(param f64) (result i32) (f64.lt (local.get 0) (local.get 0))) \
            (func (export \"nane\") (param f64) (result i32) (f64.eq (local.get 0) (local.get 0)))) \
            (assert_return (invoke \"shl\"  (i32.const 1) (i32.const 32)) (i32.const 1)) \
            (assert_return (invoke \"shl\"  (i32.const 1) (i32.const 33)) (i32.const 2)) \
            (assert_return (invoke \"rotl\" (i32.const 0x12345678) (i32.const 4)) (i32.const 0x23456781)) \
            (assert_return (invoke \"clz\"  (i32.const 1)) (i32.const 31)) \
            (assert_return (invoke \"clz\"  (i32.const 0)) (i32.const 32)) \
            (assert_return (invoke \"ctz\"  (i32.const 0x80000000)) (i32.const 31)) \
            (assert_return (invoke \"pop\"  (i32.const 0xffffffff)) (i32.const 32)) \
            (assert_return (invoke \"rems\" (i32.const -7) (i32.const 3)) (i32.const -1)) \
            (assert_return (invoke \"nanlt\" (f64.const nan)) (i32.const 0)) \
            (assert_return (invoke \"nane\"  (f64.const nan)) (i32.const 0))";
        let n = run_wast(script).expect("numeric edge-case conformance passes");
        assert_eq!(n, 10);
    }

    #[test]
    fn wat_float_literals_hex_inf_nan() {
        // Hex floats, inf, and nan in *module bodies* (not just assertion operands).
        let script = "(module \
            (func (export \"a\") (result f64) (f64.const 0x1.8p1)) \
            (func (export \"b\") (result f64) (f64.const 0x10)) \
            (func (export \"inf\") (result f64) (f64.const inf)) \
            (func (export \"ninf\") (result f64) (f64.const -inf)) \
            (func (export \"isnan\") (result i32) (f64.ne (f64.const nan) (f64.const nan)))) \
            (assert_return (invoke \"a\") (f64.const 3.0)) \
            (assert_return (invoke \"b\") (f64.const 16.0)) \
            (assert_return (invoke \"inf\") (f64.const inf)) \
            (assert_return (invoke \"ninf\") (f64.const -inf)) \
            (assert_return (invoke \"isnan\") (i32.const 1))";
        let n = run_wast(script).expect("hex/inf/nan float literals parse");
        assert_eq!(n, 5);
        // Direct hex-float checks.
        assert_eq!(parse_hex_float("0x1.8p1"), Some(3.0));
        assert_eq!(parse_hex_float("0x1p-1"), Some(0.5));
        assert_eq!(parse_hex_float("-0x1.0p4"), Some(-16.0));
        assert_eq!(parse_hex_float("12.5"), None); // not hex
    }

    #[test]
    fn wat_hex_and_underscore_int_literals() {
        // Hex (`0x…`) and `_`-separated literals — the upstream `.wast` spelling.
        let script = "(module \
            (func (export \"k\") (result i32) (i32.const 0xff)) \
            (func (export \"big\") (result i32) (i32.const 0xffff_ffff)) \
            (func (export \"sep\") (result i32) (i32.const 1_000_000)) \
            (func (export \"neg\") (result i32) (i32.const -0x80000000))) \
            (assert_return (invoke \"k\")   (i32.const 255)) \
            (assert_return (invoke \"big\") (i32.const -1)) \
            (assert_return (invoke \"sep\") (i32.const 1000000)) \
            (assert_return (invoke \"neg\") (i32.const -2147483648))";
        let n = run_wast(script).expect("hex/underscore literals parse");
        assert_eq!(n, 4);
        // Direct parser check.
        assert_eq!(parse_wat_int("0x10").unwrap(), 16);
        assert_eq!(parse_wat_int("-0x1").unwrap(), -1);
        assert_eq!(parse_wat_int("4_294_967_295").unwrap(), 4_294_967_295);
    }

    #[test]
    fn spec_loop_with_back_edge() {
        // A `loop` + `br_if` back-edge: sum 1..=n by counting down. Exercises a
        // branch target landing at the loop header (a backward jump).
        let script = "(module \
            (func (export \"sum\") (param $n i32) (result i32) \
              (local $acc i32) \
              (block $break \
                (loop $cont \
                  (br_if $break (i32.eqz (local.get $n))) \
                  (local.set $acc (i32.add (local.get $acc) (local.get $n))) \
                  (local.set $n (i32.sub (local.get $n) (i32.const 1))) \
                  (br $cont))) \
              (local.get $acc))) \
            (assert_return (invoke \"sum\" (i32.const 5)) (i32.const 15)) \
            (assert_return (invoke \"sum\" (i32.const 10)) (i32.const 55)) \
            (assert_return (invoke \"sum\" (i32.const 0)) (i32.const 0)) \
            (assert_return (invoke \"sum\" (i32.const 1)) (i32.const 1))";
        let n = run_wast(script).expect("loop back-edge conformance passes");
        assert_eq!(n, 4);
    }

    #[test]
    fn spec_typed_if_and_block_results() {
        // `if (result i32) (then …) (else …)` yields a value from the taken arm; a
        // `block (result i32)` yields the value left on its stack.
        let script = "(module \
            (func (export \"absish\") (param i32) (result i32) \
              (if (result i32) (i32.lt_s (local.get 0) (i32.const 0)) \
                (then (i32.sub (i32.const 0) (local.get 0))) \
                (else (local.get 0)))) \
            (func (export \"clamp\") (param i32) (result i32) \
              (if (result i32) (i32.gt_s (local.get 0) (i32.const 100)) \
                (then (i32.const 100)) \
                (else (local.get 0)))) \
            (func (export \"blk\") (result i32) \
              (block (result i32) (i32.const 7) (i32.const 35) (i32.add)))) \
            (assert_return (invoke \"absish\" (i32.const -7)) (i32.const 7)) \
            (assert_return (invoke \"absish\" (i32.const 5)) (i32.const 5)) \
            (assert_return (invoke \"clamp\" (i32.const 250)) (i32.const 100)) \
            (assert_return (invoke \"clamp\" (i32.const 42)) (i32.const 42)) \
            (assert_return (invoke \"blk\") (i32.const 42))";
        let n = run_wast(script).expect("typed if/block conformance passes");
        assert_eq!(n, 5);
    }

    #[test]
    fn spec_select_and_drop() {
        // `select` picks one of two values by an i32 condition; `drop` discards the
        // top of the stack.
        let script = "(module \
            (func (export \"sel\") (param i32 i32 i32) (result i32) \
              (select (local.get 0) (local.get 1) (local.get 2))) \
            (func (export \"drop2\") (param i32 i32) (result i32) \
              (local.get 0) (local.get 1) (drop))) \
            (assert_return (invoke \"sel\" (i32.const 10) (i32.const 20) (i32.const 1)) (i32.const 10)) \
            (assert_return (invoke \"sel\" (i32.const 10) (i32.const 20) (i32.const 0)) (i32.const 20)) \
            (assert_return (invoke \"sel\" (i32.const 10) (i32.const 20) (i32.const -5)) (i32.const 10)) \
            (assert_return (invoke \"drop2\" (i32.const 7) (i32.const 9)) (i32.const 7))";
        let n = run_wast(script).expect("select/drop conformance passes");
        assert_eq!(n, 4);
    }

    #[test]
    fn spec_multi_value_results() {
        // A function returning two values (the multi-value proposal, now standard).
        let script = "(module \
            (func (export \"pair\") (result i32 i32) (i32.const 7) (i32.const 9)) \
            (func (export \"swap\") (param i32 i32) (result i32 i32) (local.get 1) (local.get 0))) \
            (assert_return (invoke \"pair\") (i32.const 7) (i32.const 9)) \
            (assert_return (invoke \"swap\" (i32.const 1) (i32.const 2)) (i32.const 2) (i32.const 1))";
        let n = run_wast(script).expect("multi-value results conformance passes");
        assert_eq!(n, 2);
    }

    #[test]
    fn spec_f32_operations() {
        // f32 arithmetic at single precision, NaN-propagating min/max with ±0, the
        // ordered comparisons, and abs/neg/sqrt.
        let script = "(module \
            (func (export \"add\") (param f32 f32) (result f32) (f32.add (local.get 0) (local.get 1))) \
            (func (export \"div\") (param f32 f32) (result f32) (f32.div (local.get 0) (local.get 1))) \
            (func (export \"min\") (param f32 f32) (result f32) (f32.min (local.get 0) (local.get 1))) \
            (func (export \"max\") (param f32 f32) (result f32) (f32.max (local.get 0) (local.get 1))) \
            (func (export \"sqrt\") (param f32) (result f32) (f32.sqrt (local.get 0))) \
            (func (export \"lt\") (param f32 f32) (result i32) (f32.lt (local.get 0) (local.get 1))) \
            (func (export \"copysign\") (param f32 f32) (result f32) (f32.copysign (local.get 0) (local.get 1)))) \
            (assert_return (invoke \"add\" (f32.const 0.1) (f32.const 0.2)) (f32.const 0.3)) \
            (assert_return (invoke \"div\" (f32.const 1.0) (f32.const 0.0)) (f32.const inf)) \
            (assert_return (invoke \"min\" (f32.const 0.0) (f32.const -0.0)) (f32.const -0.0)) \
            (assert_return (invoke \"max\" (f32.const 0.0) (f32.const -0.0)) (f32.const 0.0)) \
            (assert_return (invoke \"min\" (f32.const nan) (f32.const 1.0)) (f32.const nan)) \
            (assert_return (invoke \"sqrt\" (f32.const 16.0)) (f32.const 4.0)) \
            (assert_return (invoke \"lt\" (f32.const nan) (f32.const nan)) (i32.const 0)) \
            (assert_return (invoke \"copysign\" (f32.const 5.0) (f32.const -1.0)) (f32.const -5.0))";
        let n = run_wast(script).expect("f32 operations conformance passes");
        assert_eq!(n, 8);
    }

    #[test]
    fn spec_i64_operations() {
        // i64 arithmetic/comparison/bitwise edge cases: wrapping, signed vs unsigned
        // division and comparison, rotate, and clz/ctz on 64-bit values.
        let script = "(module \
            (func (export \"mul\") (param i64 i64) (result i64) (i64.mul (local.get 0) (local.get 1))) \
            (func (export \"divu\") (param i64 i64) (result i64) (i64.div_u (local.get 0) (local.get 1))) \
            (func (export \"ltu\") (param i64 i64) (result i32) (i64.lt_u (local.get 0) (local.get 1))) \
            (func (export \"lts\") (param i64 i64) (result i32) (i64.lt_s (local.get 0) (local.get 1))) \
            (func (export \"rotl\") (param i64 i64) (result i64) (i64.rotl (local.get 0) (local.get 1))) \
            (func (export \"clz\") (param i64) (result i64) (i64.clz (local.get 0))) \
            (func (export \"shru\") (param i64 i64) (result i64) (i64.shr_u (local.get 0) (local.get 1)))) \
            (assert_return (invoke \"mul\" (i64.const 0x100000000) (i64.const 0x100000000)) (i64.const 0)) \
            (assert_return (invoke \"divu\" (i64.const -1) (i64.const 2)) (i64.const 0x7fffffffffffffff)) \
            (assert_return (invoke \"ltu\" (i64.const -1) (i64.const 1)) (i32.const 0)) \
            (assert_return (invoke \"lts\" (i64.const -1) (i64.const 1)) (i32.const 1)) \
            (assert_return (invoke \"rotl\" (i64.const 0x1) (i64.const 4)) (i64.const 0x10)) \
            (assert_return (invoke \"clz\" (i64.const 1)) (i64.const 63)) \
            (assert_return (invoke \"clz\" (i64.const 0)) (i64.const 64)) \
            (assert_return (invoke \"shru\" (i64.const -1) (i64.const 60)) (i64.const 0xf)) \
            (assert_trap (invoke \"divu\" (i64.const 1) (i64.const 0)))";
        let n = run_wast(script).expect("i64 operations conformance passes");
        assert_eq!(n, 9);
    }

    #[test]
    fn spec_memory_grow_and_size() {
        // `memory.size` reports current pages; `memory.grow` returns the old size
        // (or -1 on failure) and makes the new bytes accessible.
        let script = "(module \
            (memory 1 3) \
            (func (export \"size\") (result i32) (memory.size)) \
            (func (export \"grow\") (param i32) (result i32) (memory.grow (local.get 0))) \
            (func (export \"st\") (param i32 i32) (i32.store (local.get 0) (local.get 1))) \
            (func (export \"ld\") (param i32) (result i32) (i32.load (local.get 0)))) \
            (assert_return (invoke \"size\") (i32.const 1)) \
            (assert_return (invoke \"grow\" (i32.const 1)) (i32.const 1)) \
            (assert_return (invoke \"size\") (i32.const 2)) \
            (invoke \"st\" (i32.const 65536) (i32.const 777)) \
            (assert_return (invoke \"ld\" (i32.const 65536)) (i32.const 777)) \
            (assert_return (invoke \"grow\" (i32.const 5)) (i32.const -1)) \
            (assert_return (invoke \"size\") (i32.const 2)) \
            (assert_trap (invoke \"ld\" (i32.const 200000)))";
        let n = run_wast(script).expect("memory.grow/size conformance passes");
        assert_eq!(n, 8);
    }

    #[test]
    fn spec_reinterpret_and_promote_demote() {
        // Bit-preserving reinterpret round-trips and the f32<->f64 width changes.
        let script = "(module \
            (func (export \"f2i\") (param f32) (result i32) (i32.reinterpret_f32 (local.get 0))) \
            (func (export \"i2f\") (param i32) (result f32) (f32.reinterpret_i32 (local.get 0))) \
            (func (export \"d2i\") (param f64) (result i64) (i64.reinterpret_f64 (local.get 0))) \
            (func (export \"promote\") (param f32) (result f64) (f64.promote_f32 (local.get 0))) \
            (func (export \"demote\") (param f64) (result f32) (f32.demote_f64 (local.get 0)))) \
            (assert_return (invoke \"f2i\" (f32.const 1.0)) (i32.const 0x3f800000)) \
            (assert_return (invoke \"i2f\" (i32.const 0x40490fdb)) (f32.const 3.14159274)) \
            (assert_return (invoke \"d2i\" (f64.const 1.0)) (i64.const 0x3ff0000000000000)) \
            (assert_return (invoke \"promote\" (f32.const 1.5)) (f64.const 1.5)) \
            (assert_return (invoke \"demote\" (f64.const 1.5)) (f32.const 1.5)) \
            (assert_return (invoke \"f2i\" (f32.const nan)) (i32.const 0x7fc00000))";
        let n = run_wast(script).expect("reinterpret/promote/demote conformance passes");
        assert_eq!(n, 6);
    }

    #[test]
    fn spec_br_table_and_nested_control() {
        // `br_table` indexes a label vector (clamping out-of-range to the default),
        // and `br`/`br_if` to an outer block/loop exit through nesting.
        let script = "(module \
            (func (export \"sw\") (param i32) (result i32) \
              (block $d (block $c (block $b (block $a \
                (br_table $a $b $c $d (local.get 0))) \
                (return (i32.const 10))) \
                (return (i32.const 20))) \
                (return (i32.const 30))) \
              (i32.const 99)) \
            (func (export \"sum\") (param i32) (result i32) \
              (local $i i32) (local $acc i32) \
              (block $break (loop $cont \
                (br_if $break (i32.ge_s (local.get $i) (local.get 0))) \
                (local.set $acc (i32.add (local.get $acc) (local.get $i))) \
                (local.set $i (i32.add (local.get $i) (i32.const 1))) \
                (br $cont))) \
              (local.get $acc))) \
            (assert_return (invoke \"sw\" (i32.const 0)) (i32.const 10)) \
            (assert_return (invoke \"sw\" (i32.const 1)) (i32.const 20)) \
            (assert_return (invoke \"sw\" (i32.const 2)) (i32.const 30)) \
            (assert_return (invoke \"sw\" (i32.const 3)) (i32.const 99)) \
            (assert_return (invoke \"sw\" (i32.const 7)) (i32.const 99)) \
            (assert_return (invoke \"sum\" (i32.const 5)) (i32.const 10)) \
            (assert_return (invoke \"sum\" (i32.const 10)) (i32.const 45))";
        let n = run_wast(script).expect("br_table + nested control conformance passes");
        assert_eq!(n, 7);
    }

    #[test]
    fn spec_passive_data_init_and_drop() {
        // A passive data segment is copied into memory by `memory.init`, then
        // `data.drop` releases it so a later non-zero `memory.init` traps.
        let script = "(module \
            (memory 1) \
            (data \"\\aa\\bb\\cc\\dd\") \
            (func (export \"init\") (param i32 i32 i32) \
              (memory.init 0 (local.get 0) (local.get 1) (local.get 2))) \
            (func (export \"drop\") (data.drop 0)) \
            (func (export \"ld\") (param i32) (result i32) (i32.load8_u (local.get 0)))) \
            (invoke \"init\" (i32.const 5) (i32.const 0) (i32.const 4)) \
            (assert_return (invoke \"ld\" (i32.const 5)) (i32.const 0xaa)) \
            (assert_return (invoke \"ld\" (i32.const 8)) (i32.const 0xdd)) \
            (assert_return (invoke \"ld\" (i32.const 9)) (i32.const 0)) \
            (invoke \"drop\") \
            (assert_trap (invoke \"init\" (i32.const 0) (i32.const 0) (i32.const 1)))";
        let n = run_wast(script).expect("passive-data init/drop conformance passes");
        assert_eq!(n, 6);
    }

    #[test]
    fn spec_bulk_memory_fill_and_copy() {
        // fill writes a byte run; copy moves a (possibly overlapping) run; both trap
        // out of bounds. `ld` reads a byte back.
        let script = "(module \
            (memory 1) \
            (func (export \"fill\") (param i32 i32 i32) \
              (memory.fill (local.get 0) (local.get 1) (local.get 2))) \
            (func (export \"copy\") (param i32 i32 i32) \
              (memory.copy (local.get 0) (local.get 1) (local.get 2))) \
            (func (export \"ld\") (param i32) (result i32) (i32.load8_u (local.get 0)))) \
            (invoke \"fill\" (i32.const 10) (i32.const 0xab) (i32.const 4)) \
            (assert_return (invoke \"ld\" (i32.const 10)) (i32.const 0xab)) \
            (assert_return (invoke \"ld\" (i32.const 13)) (i32.const 0xab)) \
            (assert_return (invoke \"ld\" (i32.const 14)) (i32.const 0)) \
            (invoke \"copy\" (i32.const 20) (i32.const 10) (i32.const 4)) \
            (assert_return (invoke \"ld\" (i32.const 22)) (i32.const 0xab)) \
            (assert_trap   (invoke \"fill\" (i32.const 65534) (i32.const 1) (i32.const 4)))";
        let n = run_wast(script).expect("bulk-memory conformance passes");
        assert_eq!(n, 7);
    }

    #[test]
    fn spec_trunc_sat_saturates() {
        // Unlike `trunc`, `trunc_sat` never traps: NaN → 0, out-of-range → clamp.
        let script = "(module \
            (func (export \"ss\") (param f64) (result i32) (i32.trunc_sat_f64_s (local.get 0))) \
            (func (export \"su\") (param f64) (result i32) (i32.trunc_sat_f64_u (local.get 0)))) \
            (assert_return (invoke \"ss\" (f64.const 3.9)) (i32.const 3)) \
            (assert_return (invoke \"ss\" (f64.const nan)) (i32.const 0)) \
            (assert_return (invoke \"ss\" (f64.const 1e30)) (i32.const 2147483647)) \
            (assert_return (invoke \"ss\" (f64.const -1e30)) (i32.const -2147483648)) \
            (assert_return (invoke \"su\" (f64.const -1.0)) (i32.const 0)) \
            (assert_return (invoke \"su\" (f64.const 1e30)) (i32.const -1))";
        let n = run_wast(script).expect("trunc_sat conformance passes");
        assert_eq!(n, 6);
    }

    #[test]
    fn spec_sign_extension_ops() {
        let script = "(module \
            (func (export \"e8\")  (param i32) (result i32) (i32.extend8_s (local.get 0))) \
            (func (export \"e16\") (param i32) (result i32) (i32.extend16_s (local.get 0))) \
            (func (export \"e832\")(param i64) (result i64) (i64.extend32_s (local.get 0)))) \
            (assert_return (invoke \"e8\"  (i32.const 255)) (i32.const -1)) \
            (assert_return (invoke \"e8\"  (i32.const 127)) (i32.const 127)) \
            (assert_return (invoke \"e8\"  (i32.const 128)) (i32.const -128)) \
            (assert_return (invoke \"e16\" (i32.const 65535)) (i32.const -1)) \
            (assert_return (invoke \"e16\" (i32.const 32768)) (i32.const -32768)) \
            (assert_return (invoke \"e832\"(i64.const 4294967295)) (i64.const -1))";
        let n = run_wast(script).expect("sign-extension conformance passes");
        assert_eq!(n, 6);
    }

    #[test]
    fn spec_float_rounding_and_copysign() {
        let script = "(module \
            (func (export \"ceil\")  (param f64) (result f64) (f64.ceil (local.get 0))) \
            (func (export \"floor\") (param f64) (result f64) (f64.floor (local.get 0))) \
            (func (export \"trunc\") (param f64) (result f64) (f64.trunc (local.get 0))) \
            (func (export \"near\")  (param f64) (result f64) (f64.nearest (local.get 0))) \
            (func (export \"cs\")    (param f64) (param f64) (result f64) (f64.copysign (local.get 0) (local.get 1)))) \
            (assert_return (invoke \"ceil\"  (f64.const 2.3)) (f64.const 3)) \
            (assert_return (invoke \"floor\" (f64.const 2.7)) (f64.const 2)) \
            (assert_return (invoke \"floor\" (f64.const -2.3)) (f64.const -3)) \
            (assert_return (invoke \"trunc\" (f64.const -2.7)) (f64.const -2)) \
            (assert_return (invoke \"near\"  (f64.const 2.5)) (f64.const 2)) \
            (assert_return (invoke \"near\"  (f64.const 3.5)) (f64.const 4)) \
            (assert_return (invoke \"near\"  (f64.const -1.5)) (f64.const -2)) \
            (assert_return (invoke \"cs\"    (f64.const 3) (f64.const -1)) (f64.const -3)) \
            (assert_return (invoke \"cs\"    (f64.const -3) (f64.const 1)) (f64.const 3))";
        let n = run_wast(script).expect("rounding/copysign conformance passes");
        assert_eq!(n, 9);
    }

    #[test]
    fn spec_convert_i64_signedness() {
        // -1 as i64 converts to -1.0 signed but 1.8446744e19 (2^64-1) unsigned.
        let script = "(module \
            (func (export \"cs\") (param i64) (result f64) (f64.convert_i64_s (local.get 0))) \
            (func (export \"cu\") (param i64) (result f64) (f64.convert_i64_u (local.get 0)))) \
            (assert_return (invoke \"cs\" (i64.const -1)) (f64.const -1)) \
            (assert_return (invoke \"cu\" (i64.const -1)) (f64.const 18446744073709551616)) \
            (assert_return (invoke \"cu\" (i64.const 0)) (f64.const 0)) \
            (assert_return (invoke \"cs\" (i64.const 1000000)) (f64.const 1000000))";
        let n = run_wast(script).expect("convert_i64 signedness passes");
        assert_eq!(n, 4);
    }

    #[test]
    fn spec_trunc_range_traps() {
        // (No `;;` comments — `\`-continued Rust strings have no real newlines, so a
        // WAT line comment would swallow the rest of the script.)
        let script = "(module \
            (func (export \"ts\") (param f64) (result i32) (i32.trunc_f64_s (local.get 0))) \
            (func (export \"tu\") (param f64) (result i32) (i32.trunc_f64_u (local.get 0)))) \
            (assert_return (invoke \"ts\" (f64.const -3.9)) (i32.const -3)) \
            (assert_return (invoke \"ts\" (f64.const 2147483647.0)) (i32.const 2147483647)) \
            (assert_trap   (invoke \"ts\" (f64.const 2147483648.0))) \
            (assert_trap   (invoke \"ts\" (f64.const nan))) \
            (assert_return (invoke \"tu\" (f64.const 4294967295.0)) (i32.const -1)) \
            (assert_trap   (invoke \"tu\" (f64.const -1.0))) \
            (assert_trap   (invoke \"tu\" (f64.const 4294967296.0)))";
        let n = run_wast(script).expect("trunc range-trap conformance passes");
        assert_eq!(n, 7);
    }

    #[test]
    fn spec_div_overflow_and_traps() {
        // i32.div_s(INT_MIN, -1) overflows → traps; rem_s of the same is 0 (no trap).
        let script = "(module \
            (func (export \"divs\") (param i32 i32) (result i32) (i32.div_s (local.get 0) (local.get 1))) \
            (func (export \"rems\") (param i32 i32) (result i32) (i32.rem_s (local.get 0) (local.get 1)))) \
            (assert_trap   (invoke \"divs\" (i32.const -2147483648) (i32.const -1))) \
            (assert_return (invoke \"rems\" (i32.const -2147483648) (i32.const -1)) (i32.const 0)) \
            (assert_return (invoke \"divs\" (i32.const -2147483648) (i32.const 1)) (i32.const -2147483648)) \
            (assert_trap   (invoke \"divs\" (i32.const 1) (i32.const 0)))";
        let n = run_wast(script).expect("div overflow/trap conformance passes");
        assert_eq!(n, 4);
    }

    #[test]
    fn spec_conformance_corpus() {
        // A spec-format `.wast` conformance corpus (WAT text modules + assertions)
        // exercising a broad slice of the engine through the harness, in the shape
        // the upstream suite uses.
        let script = "\
;; --- i32 arithmetic & comparison (cf. i32.wast) ---
(module (func (export \"add\") (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1)))
        (func (export \"sub\") (param i32 i32) (result i32) (i32.sub (local.get 0) (local.get 1)))
        (func (export \"mul\") (param i32 i32) (result i32) (i32.mul (local.get 0) (local.get 1)))
        (func (export \"divs\") (param i32 i32) (result i32) (i32.div_s (local.get 0) (local.get 1)))
        (func (export \"lts\") (param i32 i32) (result i32) (i32.lt_s (local.get 0) (local.get 1)))
        (func (export \"shl\") (param i32 i32) (result i32) (i32.shl (local.get 0) (local.get 1))))
(assert_return (invoke \"add\" (i32.const 1) (i32.const 1)) (i32.const 2))
(assert_return (invoke \"add\" (i32.const -1) (i32.const -1)) (i32.const -2))
(assert_return (invoke \"sub\" (i32.const 5) (i32.const 8)) (i32.const -3))
(assert_return (invoke \"mul\" (i32.const 6) (i32.const 7)) (i32.const 42))
(assert_return (invoke \"divs\" (i32.const 20) (i32.const 4)) (i32.const 5))
(assert_trap   (invoke \"divs\" (i32.const 1) (i32.const 0)))
(assert_return (invoke \"lts\" (i32.const -1) (i32.const 0)) (i32.const 1))
(assert_return (invoke \"lts\" (i32.const 0) (i32.const -1)) (i32.const 0))
(assert_return (invoke \"shl\" (i32.const 1) (i32.const 4)) (i32.const 16))
;; --- f64 arithmetic (cf. f64.wast) ---
(module (func (export \"fadd\") (param f64 f64) (result f64) (f64.add (local.get 0) (local.get 1)))
        (func (export \"fdiv\") (param f64 f64) (result f64) (f64.div (local.get 0) (local.get 1)))
        (func (export \"conv\") (param i32) (result f64) (f64.convert_i32_s (local.get 0))))
(assert_return (invoke \"fadd\" (f64.const 1.5) (f64.const 2.25)) (f64.const 3.75))
(assert_return (invoke \"fdiv\" (f64.const 7) (f64.const 2)) (f64.const 3.5))
(assert_return (invoke \"conv\" (i32.const -8)) (f64.const -8))
;; --- i64 + conversions (cf. i64.wast / conversions.wast) ---
(module (func (export \"i64add\") (param i64 i64) (result i64) (i64.add (local.get 0) (local.get 1)))
        (func (export \"ext\") (param i32) (result i64) (i64.extend_i32_s (local.get 0)))
        (func (export \"wrap\") (param i64) (result i32) (i32.wrap_i64 (local.get 0))))
(assert_return (invoke \"i64add\" (i64.const 100) (i64.const 23)) (i64.const 123))
(assert_return (invoke \"ext\" (i32.const -1)) (i64.const -1))
(assert_return (invoke \"wrap\" (i64.const 4294967297)) (i32.const 1))
;; --- malformed modules are rejected ---
(assert_invalid (module binary \"\\00\\00\\00\\00\") \"bad magic\")
(assert_invalid (module binary \"\\00\\61\\73\\6d\") \"truncated\")";
        let n = run_wast(script).expect("conformance corpus passes");
        // 9 + 3 + 3 i32/f64/i64 returns/traps + 2 invalids = 17.
        assert_eq!(n, 17, "all conformance assertions executed");
    }

    #[test]
    fn wast_full_module_surface_conformance() {
        // A single upstream-style `.wast` exercising the whole module surface
        // together: a type + table + elem + call_indirect, a mutable global, linear
        // memory with a data segment, and control flow — driven by assertions.
        let script = "\
(module
  (type $unop (func (param i32) (result i32)))
  (table 2 funcref)
  (memory 1)
  (global $hits (mut i32) (i32.const 0))
  (data (i32.const 0) \"\\2a\\00\\00\\00\")
  (func $sq   (param $x i32) (result i32) (i32.mul (local.get $x) (local.get $x)))
  (func $negv (param $x i32) (result i32) (i32.sub (i32.const 0) (local.get $x)))
  (elem (i32.const 0) $sq $negv)
  (func (export \"apply\") (param $i i32) (param $x i32) (result i32)
    (global.set $hits (i32.add (global.get $hits) (i32.const 1)))
    (call_indirect (type $unop) (local.get $x) (local.get $i)))
  (func (export \"hits\")   (result i32) (global.get $hits))
  (func (export \"mem42\")  (result i32) (i32.load (i32.const 0)))
  (func (export \"clamp\")  (param $x i32) (result i32)
    (if (result i32) (i32.lt_s (local.get $x) (i32.const 0))
      (then (i32.const 0)) (else (local.get $x)))))
(assert_return (invoke \"apply\" (i32.const 0) (i32.const 9))  (i32.const 81))
(assert_return (invoke \"apply\" (i32.const 1) (i32.const 9))  (i32.const -9))
(assert_return (invoke \"hits\")                               (i32.const 2))
(assert_return (invoke \"mem42\")                              (i32.const 42))
(assert_return (invoke \"clamp\" (i32.const -5))               (i32.const 0))
(assert_return (invoke \"clamp\" (i32.const 7))                (i32.const 7))";
        let n = run_wast(script).expect("full-surface conformance passes");
        assert_eq!(n, 6, "all six assertions executed");
    }

    #[test]
    fn wast_conformance_with_control_flow() {
        // An upstream-style `.wast`: a real WAT text module with named functions,
        // loops, `if`, and inter-function calls, driven by assert_return/assert_trap
        // through the full WAT → binary → decode → validate → execute pipeline.
        let script = "\
(module
  (func $abs (export \"abs\") (param $x i32) (result i32)
    (if (result i32) (i32.lt_s (local.get $x) (i32.const 0))
      (then (i32.sub (i32.const 0) (local.get $x)))
      (else (local.get $x))))
  (func $fact (export \"fact\") (param $n i32) (result i32)
    (local $acc i32) (local $i i32)
    (local.set $acc (i32.const 1))
    (local.set $i (i32.const 1))
    (block $done
      (loop $loop
        (br_if $done (i32.gt_s (local.get $i) (local.get $n)))
        (local.set $acc (i32.mul (local.get $acc) (local.get $i)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))
    (local.get $acc))
  (func $absfact (export \"absfact\") (param $x i32) (result i32)
    ;; fact(abs(x)) — exercises inter-function calls by name
    (call $fact (call $abs (local.get $x))))
  (func $div (export \"div\") (param $a i32) (param $b i32) (result i32)
    (i32.div_s (local.get $a) (local.get $b))))
(assert_return (invoke \"abs\" (i32.const -9)) (i32.const 9))
(assert_return (invoke \"abs\" (i32.const 4)) (i32.const 4))
(assert_return (invoke \"fact\" (i32.const 0)) (i32.const 1))
(assert_return (invoke \"fact\" (i32.const 5)) (i32.const 120))
(assert_return (invoke \"fact\" (i32.const 6)) (i32.const 720))
(assert_return (invoke \"absfact\" (i32.const -4)) (i32.const 24))
(assert_return (invoke \"div\" (i32.const 20) (i32.const 4)) (i32.const 5))
(assert_trap   (invoke \"div\" (i32.const 1) (i32.const 0)))";
        let n = run_wast(script).expect("control-flow .wast conformance passes");
        assert_eq!(n, 8, "all eight assertions executed");
    }

    #[test]
    fn wast_nan_and_signed_zero_matching() {
        // A NaN-producing function: f(x) = x / x  (0/0 = NaN, n/n = 1).
        // `assert_return … (f64.const nan)` must match the NaN result, and a
        // `nan:canonical` token is accepted too. Signed-zero distinguishes by bits.
        let script = "(module \
            (func (export \"dd\") (param $x f64) (result f64) (f64.div (local.get $x) (local.get $x))) \
            (func (export \"negz\") (result f64) (f64.neg (f64.const 0.0)))) \
            (assert_return (invoke \"dd\" (f64.const 0.0)) (f64.const nan)) \
            (assert_return (invoke \"dd\" (f64.const 5.0)) (f64.const 1.0)) \
            (assert_return (invoke \"dd\" (f64.const 0.0)) (f64.const nan:canonical)) \
            (assert_return (invoke \"negz\") (f64.const -0.0))";
        let n = run_wast(script).expect("nan/signed-zero .wast passes");
        assert_eq!(n, 4);
    }

    #[test]
    fn wast_bare_invoke_runs_in_sequence() {
        // A bare `(invoke "bump")` mutates a global between assertions; the later
        // `assert_return (invoke "get")` must observe the side effect.
        let script = "(module \
            (global $n (mut i32) (i32.const 0)) \
            (func (export \"bump\") (global.set $n (i32.add (global.get $n) (i32.const 5)))) \
            (func (export \"get\") (result i32) (global.get $n))) \
            (assert_return (invoke \"get\") (i32.const 0)) \
            (invoke \"bump\") \
            (invoke \"bump\") \
            (assert_return (invoke \"get\") (i32.const 10))";
        // 4 commands run against the one instance, in order.
        let n = run_wast(script).expect("bare-invoke sequence passes");
        assert_eq!(n, 4);
    }

    #[test]
    fn wast_runs_a_text_module() {
        // A .wast script whose module is WAT *text* (not binary).
        let script = "(module (func (export \"sub\") (param i32 i32) (result i32) \
                      (i32.sub (local.get 0) (local.get 1))))\n\
                      (assert_return (invoke \"sub\" (i32.const 10) (i32.const 4)) (i32.const 6))\n\
                      (assert_return (invoke \"sub\" (i32.const 0) (i32.const 7)) (i32.const -7))";
        let n = run_wast(script).expect("text-module wast passes");
        assert_eq!(n, 2);
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
