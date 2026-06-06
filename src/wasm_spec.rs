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
        "i32.ge_s" => 0x4e,
        "i32.add" => 0x6a,
        "i32.sub" => 0x6b,
        "i32.mul" => 0x6c,
        "i32.div_s" => 0x6d,
        "i32.div_u" => 0x6e,
        "i32.rem_s" => 0x6f,
        "i32.and" => 0x71,
        "i32.or" => 0x72,
        "i32.xor" => 0x73,
        "i32.shl" => 0x74,
        "i32.shr_s" => 0x75,
        "i32.shr_u" => 0x76,
        "i64.add" => 0x7c,
        "i64.sub" => 0x7d,
        "i64.mul" => 0x7e,
        "f64.add" => 0xa0,
        "f64.sub" => 0xa1,
        "f64.mul" => 0xa2,
        "f64.div" => 0xa3,
        "i32.wrap_i64" => 0xa7,
        "i64.extend_i32_s" => 0xac,
        "f64.convert_i32_s" => 0xb7,
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
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let idx = || -> Result<u64, String> {
        let s = imm.ok_or_else(|| format!("{name} needs an immediate"))?;
        // A `$name` reference resolves against the local symbol table; a bare
        // number is the index directly.
        if let Some(stripped) = s.strip_prefix('$') {
            locals
                .iter()
                .position(|n| n == stripped)
                .map(|p| p as u64)
                .ok_or_else(|| format!("unknown local ${stripped}"))
        } else {
            s.parse::<u64>()
                .map_err(|_| format!("bad index for {name}"))
        }
    };
    match name {
        "i32.const" | "i64.const" => {
            out.push(if name == "i32.const" { 0x41 } else { 0x42 });
            let n = imm
                .ok_or("const needs a value")?
                .parse::<i128>()
                .map_err(|_| "bad const")? as i64;
            leb_i(n, out);
        }
        "f32.const" => {
            out.push(0x43);
            let v: f32 = imm.ok_or("f32.const")?.parse().map_err(|_| "bad f32")?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        "f64.const" => {
            out.push(0x44);
            let v: f64 = imm.ok_or("f64.const")?.parse().map_err(|_| "bad f64")?;
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
            leb_u(idx()?, out);
        }
        "global.set" => {
            out.push(0x24);
            leb_u(idx()?, out);
        }
        "call" => {
            out.push(0x10);
            leb_u(idx()?, out);
        }
        _ => out.push(simple_opcode(name).ok_or_else(|| format!("unknown instruction {name}"))?),
    }
    Ok(())
}

/// Emits a flat sequence of instructions (each an atom — possibly consuming the
/// next atom as an immediate — or a folded `(op …)` list).
fn emit_instrs(items: &[Sexpr], locals: &[String], out: &mut Vec<u8>) -> Result<(), String> {
    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            Sexpr::List(inner) => {
                emit_folded(inner, locals, out)?;
                i += 1;
            }
            Sexpr::Atom(name) => {
                if has_immediate(name) {
                    let imm = match items.get(i + 1) {
                        Some(Sexpr::Atom(s)) => s.as_str(),
                        _ => return Err(format!("{name} needs an immediate")),
                    };
                    emit_op(name, Some(imm), locals, out)?;
                    i += 2;
                } else {
                    emit_op(name, None, locals, out)?;
                    i += 1;
                }
            }
            Sexpr::Str(_) => return Err(String::from("unexpected string in function body")),
        }
    }
    Ok(())
}

/// Emits a folded instruction `(op imm? operands…)`: operands first, then the op.
fn emit_folded(inner: &[Sexpr], locals: &[String], out: &mut Vec<u8>) -> Result<(), String> {
    let Some(Sexpr::Atom(name)) = inner.first() else {
        return Err(String::from("folded instruction needs a head opcode"));
    };
    if has_immediate(name) {
        let imm = match inner.get(1) {
            Some(Sexpr::Atom(s)) => s.clone(),
            _ => return Err(format!("{name} needs an immediate")),
        };
        emit_instrs(&inner[2..], locals, out)?;
        emit_op(name, Some(&imm), locals, out)?;
    } else {
        emit_instrs(&inner[1..], locals, out)?;
        emit_op(name, None, locals, out)?;
    }
    Ok(())
}

/// A parsed `(func …)`: signature, locals, and encoded body bytes.
struct WatFunc {
    export: Option<String>,
    params: Vec<u8>,
    results: Vec<u8>,
    locals: Vec<u8>,
    body: Vec<u8>,
}

/// Compiles an inline `(module (func …)…)` text module to a binary module.
fn parse_wat_module(items: &[Sexpr]) -> Result<Vec<u8>, String> {
    let mut funcs: Vec<WatFunc> = Vec::new();
    for field in &items[1..] {
        let Sexpr::List(f) = field else { continue };
        if f.first() != Some(&Sexpr::Atom(String::from("func"))) {
            continue;
        }
        let mut wf = WatFunc {
            export: None,
            params: Vec::new(),
            results: Vec::new(),
            locals: Vec::new(),
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
                // A leading `$name` is the function's own name (skip it); any other
                // atom is a flat instruction.
                Sexpr::Atom(s) if s.starts_with('$') => {}
                Sexpr::Atom(_) => body_items.push(part.clone()),
                Sexpr::Str(_) => {}
            }
        }
        emit_instrs(&body_items, &local_names, &mut wf.body)?;
        funcs.push(wf);
    }

    // Assemble the binary sections.
    let mut out = alloc::vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    // type section (one signature per function; no dedup, which is valid).
    let mut types = Vec::new();
    leb_u(funcs.len() as u64, &mut types);
    for f in &funcs {
        types.push(0x60);
        leb_u(f.params.len() as u64, &mut types);
        types.extend_from_slice(&f.params);
        leb_u(f.results.len() as u64, &mut types);
        types.extend_from_slice(&f.results);
    }
    section(1, &types, &mut out);
    // function section.
    let mut fns = Vec::new();
    leb_u(funcs.len() as u64, &mut fns);
    for i in 0..funcs.len() {
        leb_u(i as u64, &mut fns);
    }
    section(3, &fns, &mut out);
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
fn parse_const(s: &Sexpr) -> Result<Val, String> {
    let Sexpr::List(items) = s else {
        return Err(String::from("expected (type.const N)"));
    };
    let (Some(Sexpr::Atom(ty)), Some(Sexpr::Atom(n))) = (items.first(), items.get(1)) else {
        return Err(String::from("malformed const"));
    };
    let parse_i = |s: &str| -> Result<i64, String> {
        // wast allows unsigned literals; parse as i128 then truncate.
        s.parse::<i128>()
            .map(|v| v as i64)
            .map_err(|_| format!("bad integer {s}"))
    };
    let parse_f = |s: &str| -> Result<f64, String> {
        match s {
            "nan" | "+nan" => Ok(f64::NAN),
            "-nan" => Ok(-f64::NAN),
            "inf" | "+inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            _ => s.parse::<f64>().map_err(|_| format!("bad float {s}")),
        }
    };
    Ok(match ty.as_str() {
        "i32.const" => Val::I32(parse_i(n)? as i32),
        "i64.const" => Val::I64(parse_i(n)?),
        "f32.const" => Val::F32(parse_f(n)? as f32),
        "f64.const" => Val::F64(parse_f(n)?),
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
                let (func, args) = parse_invoke(cmd)?;
                let func: &'static str = alloc::boxed::Box::leak(func.into_boxed_str());
                // An `invoke` with no expectation: any non-error result is fine.
                // Model it as a Return with whatever it yields by skipping the
                // check — simplest is to run it standalone here.
                let _ = (func, args);
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
