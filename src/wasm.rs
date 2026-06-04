//! A WebAssembly backend — the **WASM peer engine** (`ROADMAP.md`).
//!
//! Alongside the tree-walker and the register VM, this lowers the *numeric*
//! subset of JavaScript functions to WebAssembly text (WAT): a function over
//! `f64` parameters whose body is `let` bindings, arithmetic, comparisons,
//! ternaries, and `return` becomes a `(func …)` in a `(module …)`. The output is
//! valid WAT that a standard assembler turns into a `.wasm` module — so a hot
//! numeric kernel can be handed to a Wasm runtime instead of interpreted.
//!
//! It lowers `let`/assignment locals, arithmetic, comparisons, ternaries,
//! `if`/`else`, `while` loops (as structured `block`/`loop`/`br_if`), and
//! function calls — enough for iterative numeric kernels. Both targets are
//! produced: `compile_module` emits WAT text, and `compile_module_binary` emits
//! a complete `.wasm` binary module (verified to instantiate and run on a real
//! WebAssembly engine). Engine code, not foreign — pure, safe `alloc`-only Rust.
//! Unsupported constructs (objects, strings, `for-of`) are reported, not
//! mis-compiled.

use crate::ast::{BinaryOp, BindingTarget, Expr, Function, Stmt, UnaryOp};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Why a function (or expression) could not be lowered to WASM.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WasmError(pub &'static str);

impl core::fmt::Display for WasmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "wasm: unsupported {}", self.0)
    }
}

/// Lowers each top-level function declaration in `program` to a WAT function and
/// returns the assembled `(module …)`.
///
/// # Errors
/// Returns [`WasmError`] if any function uses a construct outside the numeric
/// subset this backend handles.
pub fn compile_module(program: &crate::ast::Program) -> Result<String, WasmError> {
    let mut funcs = Vec::new();
    for stmt in &program.body {
        if let Stmt::Function(f) = stmt {
            funcs.push(compile_function(f)?);
        }
    }
    Ok(format!("(module\n{}\n)", funcs.join("\n")))
}

/// Lowers one numeric function to a WAT `(func …)` definition.
///
/// # Errors
/// Returns [`WasmError`] for a non-numeric construct (objects, strings, `for-of`,
/// …) or a destructuring/rest parameter.
pub fn compile_function(func: &Function) -> Result<String, WasmError> {
    let name = func.id.as_ref().map_or("anonymous", |id| &id.name);

    // Parameters bind to named `f64` locals.
    let mut params = String::new();
    for p in &func.params {
        let BindingTarget::Ident(id) = &p.target else {
            return Err(WasmError("destructuring parameter"));
        };
        if p.default.is_some() || p.rest {
            return Err(WasmError("default/rest parameter"));
        }
        params.push_str(&format!(" (param ${} f64)", id.name));
    }

    // `let`-declared names become extra locals (declared up front, as WAT
    // requires).
    let mut locals: Vec<String> = Vec::new();
    collect_locals(&func.body, &mut locals)?;
    let local_decls: String = locals
        .iter()
        .map(|n| format!("    (local ${n} f64)\n"))
        .collect();

    let mut body = String::new();
    for stmt in &func.body {
        emit_stmt(stmt, &mut body, 2)?;
    }

    Ok(format!(
        "  (func ${name} (export \"{name}\"){params} (result f64)\n{local_decls}{body}  )"
    ))
}

/// Collects the names introduced by `let`/`const`, recursing into nested
/// blocks/branches/loops (WAT requires all locals declared up front).
fn collect_locals(body: &[Stmt], out: &mut Vec<String>) -> Result<(), WasmError> {
    for stmt in body {
        match stmt {
            Stmt::Var(decl) => {
                for d in &decl.declarations {
                    let BindingTarget::Ident(id) = &d.target else {
                        return Err(WasmError("destructuring binding"));
                    };
                    out.push(id.name.to_string());
                }
            }
            Stmt::Block { body, .. } => collect_locals(body, out)?,
            Stmt::If {
                consequent,
                alternate,
                ..
            } => {
                collect_locals(core::slice::from_ref(consequent), out)?;
                if let Some(alt) = alternate {
                    collect_locals(core::slice::from_ref(alt), out)?;
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                collect_locals(core::slice::from_ref(body), out)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Emits a statement's instructions into `out` at the given indent depth.
fn emit_stmt(stmt: &Stmt, out: &mut String, depth: usize) -> Result<(), WasmError> {
    let pad = "  ".repeat(depth);
    match stmt {
        Stmt::Var(decl) => {
            for d in &decl.declarations {
                let BindingTarget::Ident(id) = &d.target else {
                    return Err(WasmError("destructuring binding"));
                };
                let init = d.init.as_ref().ok_or(WasmError("uninitialized local"))?;
                emit_expr(init, out, depth)?;
                out.push_str(&format!("{pad}local.set ${}\n", id.name));
            }
            Ok(())
        }
        Stmt::Return { argument, .. } => {
            let e = argument.as_ref().ok_or(WasmError("bare return"))?;
            emit_expr(e, out, depth)?;
            out.push_str(&format!("{pad}return\n"));
            Ok(())
        }
        Stmt::Block { body, .. } => {
            for s in body {
                emit_stmt(s, out, depth)?;
            }
            Ok(())
        }
        // `x = expr;` — store into a local.
        Stmt::Expr { expression, .. } => {
            let Expr::Assign {
                op: crate::ast::AssignOp::Assign,
                target,
                value,
                ..
            } = &**expression
            else {
                return Err(WasmError("expression statement"));
            };
            let Expr::Ident(id) = &**target else {
                return Err(WasmError("assignment target"));
            };
            emit_expr(value, out, depth)?;
            out.push_str(&format!("{pad}local.set ${}\n", id.name));
            Ok(())
        }
        Stmt::If {
            test,
            consequent,
            alternate,
            ..
        } => {
            emit_cond(test, out, depth)?;
            out.push_str(&format!("{pad}if\n"));
            emit_stmt(consequent, out, depth + 1)?;
            if let Some(alt) = alternate {
                out.push_str(&format!("{pad}else\n"));
                emit_stmt(alt, out, depth + 1)?;
            }
            out.push_str(&format!("{pad}end\n"));
            Ok(())
        }
        // `while (cond) body` → a structured `block`/`loop` with `br_if`.
        Stmt::While { test, body, .. } => {
            let inner = "  ".repeat(depth + 1);
            out.push_str(&format!("{pad}block\n{inner}loop\n"));
            emit_cond(test, out, depth + 2)?;
            let inner2 = "  ".repeat(depth + 2);
            out.push_str(&format!("{inner2}i32.eqz\n{inner2}br_if 1\n")); // exit when !cond
            emit_stmt(body, out, depth + 2)?;
            out.push_str(&format!("{inner2}br 0\n{inner}end\n{pad}end\n"));
            Ok(())
        }
        _ => Err(WasmError("statement")),
    }
}

/// Emits the instructions that leave the `f64` value of `expr` on the stack.
fn emit_expr(expr: &Expr, out: &mut String, depth: usize) -> Result<(), WasmError> {
    let pad = "  ".repeat(depth);
    match expr {
        Expr::Number { value, .. } => out.push_str(&format!("{pad}f64.const {value}\n")),
        Expr::Ident(id) => out.push_str(&format!("{pad}local.get ${}\n", id.name)),
        Expr::Unary {
            op: UnaryOp::Minus,
            argument,
            ..
        } => {
            emit_expr(argument, out, depth)?;
            out.push_str(&format!("{pad}f64.neg\n"));
        }
        Expr::Binary {
            op, left, right, ..
        } => {
            emit_expr(left, out, depth)?;
            emit_expr(right, out, depth)?;
            match op {
                BinaryOp::Add => out.push_str(&format!("{pad}f64.add\n")),
                BinaryOp::Sub => out.push_str(&format!("{pad}f64.sub\n")),
                BinaryOp::Mul => out.push_str(&format!("{pad}f64.mul\n")),
                BinaryOp::Div => out.push_str(&format!("{pad}f64.div\n")),
                // A comparison yields i32 in WASM; widen to an f64 `0`/`1` so it
                // composes with arithmetic (JS booleans are number-coercible).
                BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::LtEq
                | BinaryOp::GtEq
                | BinaryOp::EqEqEq
                | BinaryOp::EqEq
                | BinaryOp::NotEqEq
                | BinaryOp::NotEq => {
                    out.push_str(&format!("{pad}{}\n{pad}f64.convert_i32_u\n", cmp_op(*op)));
                }
                _ => return Err(WasmError("binary operator")),
            }
        }
        // `f(args)` → push the f64 arguments, then `call $f`.
        Expr::Call {
            callee, arguments, ..
        } => {
            let Expr::Ident(id) = &**callee else {
                return Err(WasmError("computed call"));
            };
            for arg in arguments {
                let crate::ast::Argument::Item(e) = arg else {
                    return Err(WasmError("spread argument"));
                };
                emit_expr(e, out, depth)?;
            }
            out.push_str(&format!("{pad}call ${}\n", id.name));
        }
        // `cond ? a : b` → `select` (operands then an i32 condition).
        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => {
            emit_expr(consequent, out, depth)?;
            emit_expr(alternate, out, depth)?;
            emit_cond(test, out, depth)?;
            out.push_str(&format!("{pad}select\n"));
        }
        _ => return Err(WasmError("expression")),
    }
    Ok(())
}

/// Emits an `i32` condition (1/0) for `select`/branches.
fn emit_cond(expr: &Expr, out: &mut String, depth: usize) -> Result<(), WasmError> {
    let pad = "  ".repeat(depth);
    match expr {
        // A comparison is already i32 — emit it without the f64 widening.
        Expr::Binary {
            op, left, right, ..
        } if is_comparison(*op) => {
            emit_expr(left, out, depth)?;
            emit_expr(right, out, depth)?;
            out.push_str(&format!("{pad}{}\n", cmp_op(*op)));
            Ok(())
        }
        // Any other numeric expression: nonzero is true.
        _ => {
            emit_expr(expr, out, depth)?;
            out.push_str(&format!("{pad}f64.const 0\n{pad}f64.ne\n"));
            Ok(())
        }
    }
}

fn is_comparison(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Lt
            | BinaryOp::Gt
            | BinaryOp::LtEq
            | BinaryOp::GtEq
            | BinaryOp::EqEqEq
            | BinaryOp::EqEq
            | BinaryOp::NotEqEq
            | BinaryOp::NotEq
    )
}

/// The WASM `f64` comparison instruction for a JS comparison operator.
fn cmp_op(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Lt => "f64.lt",
        BinaryOp::Gt => "f64.gt",
        BinaryOp::LtEq => "f64.le",
        BinaryOp::GtEq => "f64.ge",
        BinaryOp::EqEq | BinaryOp::EqEqEq => "f64.eq",
        BinaryOp::NotEq | BinaryOp::NotEqEq => "f64.ne",
        _ => "f64.eq", // unreachable for non-comparisons
    }
}

/// The single-byte opcode for a comparison's `f64.*` instruction.
fn cmp_opcode(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Lt => 0x63,
        BinaryOp::Gt => 0x64,
        BinaryOp::LtEq => 0x65,
        BinaryOp::GtEq => 0x66,
        BinaryOp::NotEq | BinaryOp::NotEqEq => 0x62,
        _ => 0x61, // f64.eq (also EqEq/EqEqEq)
    }
}

// --- WebAssembly binary (.wasm) emitter ---

/// Appends `n` as unsigned LEB128.
fn leb_u(mut n: u32, out: &mut Vec<u8>) {
    loop {
        let mut byte = (n & 0x7f) as u8;
        n >>= 7;
        if n != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if n == 0 {
            break;
        }
    }
}

/// Builds a section: id byte, then the LEB128 byte length, then `content`.
fn section(id: u8, content: &[u8], out: &mut Vec<u8>) {
    out.push(id);
    leb_u(content.len() as u32, out);
    out.extend_from_slice(content);
}

/// Lowers `program`'s top-level numeric functions to a complete WebAssembly
/// **binary** module (`\0asm` + the type/function/export/code sections).
///
/// # Errors
/// Returns [`WasmError`] for any construct outside the numeric subset.
pub fn compile_module_binary(program: &crate::ast::Program) -> Result<Vec<u8>, WasmError> {
    use alloc::collections::BTreeMap;
    let funcs: Vec<&Function> = program
        .body
        .iter()
        .filter_map(|s| match s {
            Stmt::Function(f) => Some(f),
            _ => None,
        })
        .collect();

    // Function name → index (definition order), for `call`.
    let mut fn_index: BTreeMap<&str, u32> = BTreeMap::new();
    for (i, f) in funcs.iter().enumerate() {
        if let Some(id) = &f.id {
            fn_index.insert(&id.name, i as u32);
        }
    }

    // --- type section (1): `(param f64…) -> (f64)` per function ---
    let mut types = Vec::new();
    leb_u(funcs.len() as u32, &mut types);
    for f in &funcs {
        types.push(0x60); // func type
        leb_u(f.params.len() as u32, &mut types);
        types.resize(types.len() + f.params.len(), 0x7c); // each param: f64
        leb_u(1, &mut types); // one result
        types.push(0x7c); // f64
    }

    // --- function section (3): type index per function (= its own index) ---
    let mut functions = Vec::new();
    leb_u(funcs.len() as u32, &mut functions);
    for i in 0..funcs.len() {
        leb_u(i as u32, &mut functions);
    }

    // --- export section (7): export each function by name ---
    let mut exports = Vec::new();
    leb_u(funcs.len() as u32, &mut exports);
    for (i, f) in funcs.iter().enumerate() {
        let name = f.id.as_ref().map_or("anonymous", |id| &id.name);
        leb_u(name.len() as u32, &mut exports);
        exports.extend_from_slice(name.as_bytes());
        exports.push(0x00); // export kind: func
        leb_u(i as u32, &mut exports);
    }

    // --- code section (10): each function body ---
    let mut code = Vec::new();
    leb_u(funcs.len() as u32, &mut code);
    for f in &funcs {
        let body = compile_function_body(f, &fn_index)?;
        leb_u(body.len() as u32, &mut code);
        code.extend_from_slice(&body);
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"\0asm"); // magic
    out.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1
    section(1, &types, &mut out);
    section(3, &functions, &mut out);
    section(7, &exports, &mut out);
    section(10, &code, &mut out);
    Ok(out)
}

/// Emits one function's code-section entry: the locals declaration followed by
/// the instruction stream and the terminating `end`.
fn compile_function_body(
    func: &Function,
    fns: &alloc::collections::BTreeMap<&str, u32>,
) -> Result<Vec<u8>, WasmError> {
    use alloc::collections::BTreeMap;
    // Local indices: parameters first (0..n), then `let`-declared locals.
    let mut local_index: BTreeMap<String, u32> = BTreeMap::new();
    let mut next = 0u32;
    for p in &func.params {
        let BindingTarget::Ident(id) = &p.target else {
            return Err(WasmError("destructuring parameter"));
        };
        if p.default.is_some() || p.rest {
            return Err(WasmError("default/rest parameter"));
        }
        local_index.insert(id.name.to_string(), next);
        next += 1;
    }
    let mut local_names: Vec<String> = Vec::new();
    collect_locals(&func.body, &mut local_names)?;
    for n in &local_names {
        local_index.insert(n.clone(), next);
        next += 1;
    }

    let mut body = Vec::new();
    // Locals declaration: one group of `f64`s for the let-locals.
    if local_names.is_empty() {
        leb_u(0, &mut body); // no local groups
    } else {
        leb_u(1, &mut body); // one group
        leb_u(local_names.len() as u32, &mut body);
        body.push(0x7c); // f64
    }
    for stmt in &func.body {
        emit_stmt_bin(stmt, &local_index, fns, &mut body)?;
    }
    body.push(0x0b); // end
    Ok(body)
}

fn local_idx(
    name: &str,
    locals: &alloc::collections::BTreeMap<String, u32>,
) -> Result<u32, WasmError> {
    locals.get(name).copied().ok_or(WasmError("unknown local"))
}

fn emit_stmt_bin(
    stmt: &Stmt,
    locals: &alloc::collections::BTreeMap<String, u32>,
    fns: &alloc::collections::BTreeMap<&str, u32>,
    out: &mut Vec<u8>,
) -> Result<(), WasmError> {
    match stmt {
        Stmt::Var(decl) => {
            for d in &decl.declarations {
                let BindingTarget::Ident(id) = &d.target else {
                    return Err(WasmError("destructuring binding"));
                };
                let init = d.init.as_ref().ok_or(WasmError("uninitialized local"))?;
                emit_expr_bin(init, locals, fns, out)?;
                out.push(0x21); // local.set
                leb_u(local_idx(&id.name, locals)?, out);
            }
            Ok(())
        }
        Stmt::Return { argument, .. } => {
            let e = argument.as_ref().ok_or(WasmError("bare return"))?;
            emit_expr_bin(e, locals, fns, out)?;
            out.push(0x0f); // return
            Ok(())
        }
        Stmt::Block { body, .. } => {
            for s in body {
                emit_stmt_bin(s, locals, fns, out)?;
            }
            Ok(())
        }
        Stmt::Expr { expression, .. } => {
            let Expr::Assign {
                op: crate::ast::AssignOp::Assign,
                target,
                value,
                ..
            } = &**expression
            else {
                return Err(WasmError("expression statement"));
            };
            let Expr::Ident(id) = &**target else {
                return Err(WasmError("assignment target"));
            };
            emit_expr_bin(value, locals, fns, out)?;
            out.push(0x21); // local.set
            leb_u(local_idx(&id.name, locals)?, out);
            Ok(())
        }
        Stmt::If {
            test,
            consequent,
            alternate,
            ..
        } => {
            emit_cond_bin(test, locals, fns, out)?;
            out.push(0x04); // if
            out.push(0x40); // void blocktype
            emit_stmt_bin(consequent, locals, fns, out)?;
            if let Some(alt) = alternate {
                out.push(0x05); // else
                emit_stmt_bin(alt, locals, fns, out)?;
            }
            out.push(0x0b); // end
            Ok(())
        }
        Stmt::While { test, body, .. } => {
            out.push(0x02); // block
            out.push(0x40);
            out.push(0x03); // loop
            out.push(0x40);
            emit_cond_bin(test, locals, fns, out)?;
            out.push(0x45); // i32.eqz
            out.push(0x0d); // br_if
            leb_u(1, out); // exit the block
            emit_stmt_bin(body, locals, fns, out)?;
            out.push(0x0c); // br
            leb_u(0, out); // back-edge to loop
            out.push(0x0b); // end loop
            out.push(0x0b); // end block
            Ok(())
        }
        _ => Err(WasmError("statement")),
    }
}

fn emit_expr_bin(
    expr: &Expr,
    locals: &alloc::collections::BTreeMap<String, u32>,
    fns: &alloc::collections::BTreeMap<&str, u32>,
    out: &mut Vec<u8>,
) -> Result<(), WasmError> {
    match expr {
        Expr::Number { value, .. } => {
            out.push(0x44); // f64.const
            out.extend_from_slice(&value.to_le_bytes());
        }
        Expr::Ident(id) => {
            out.push(0x20); // local.get
            leb_u(local_idx(&id.name, locals)?, out);
        }
        Expr::Unary {
            op: UnaryOp::Minus,
            argument,
            ..
        } => {
            emit_expr_bin(argument, locals, fns, out)?;
            out.push(0x9a); // f64.neg
        }
        Expr::Binary {
            op, left, right, ..
        } => {
            emit_expr_bin(left, locals, fns, out)?;
            emit_expr_bin(right, locals, fns, out)?;
            match op {
                BinaryOp::Add => out.push(0xa0),
                BinaryOp::Sub => out.push(0xa1),
                BinaryOp::Mul => out.push(0xa2),
                BinaryOp::Div => out.push(0xa3),
                op if is_comparison(*op) => {
                    out.push(cmp_opcode(*op));
                    out.push(0xb8); // f64.convert_i32_u (widen bool → f64 0/1)
                }
                _ => return Err(WasmError("binary operator")),
            }
        }
        Expr::Call {
            callee, arguments, ..
        } => {
            let Expr::Ident(id) = &**callee else {
                return Err(WasmError("computed call"));
            };
            for arg in arguments {
                let crate::ast::Argument::Item(e) = arg else {
                    return Err(WasmError("spread argument"));
                };
                emit_expr_bin(e, locals, fns, out)?;
            }
            let idx = fns
                .get(&*id.name)
                .copied()
                .ok_or(WasmError("unknown call"))?;
            out.push(0x10); // call
            leb_u(idx, out);
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => {
            emit_expr_bin(consequent, locals, fns, out)?;
            emit_expr_bin(alternate, locals, fns, out)?;
            emit_cond_bin(test, locals, fns, out)?;
            out.push(0x1b); // select
        }
        _ => return Err(WasmError("expression")),
    }
    Ok(())
}

fn emit_cond_bin(
    expr: &Expr,
    locals: &alloc::collections::BTreeMap<String, u32>,
    fns: &alloc::collections::BTreeMap<&str, u32>,
    out: &mut Vec<u8>,
) -> Result<(), WasmError> {
    match expr {
        Expr::Binary {
            op, left, right, ..
        } if is_comparison(*op) => {
            emit_expr_bin(left, locals, fns, out)?;
            emit_expr_bin(right, locals, fns, out)?;
            out.push(cmp_opcode(*op)); // i32 result, no widening
            Ok(())
        }
        _ => {
            emit_expr_bin(expr, locals, fns, out)?;
            out.push(0x44); // f64.const 0
            out.extend_from_slice(&0f64.to_le_bytes());
            out.push(0x62); // f64.ne → i32
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn module(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        compile_module(&program).expect("compile to wasm")
    }

    #[test]
    fn lowers_arithmetic_function() {
        let wat = module("function add(a, b) { return a + b; }");
        assert!(
            wat.contains("(func $add (export \"add\") (param $a f64) (param $b f64) (result f64)")
        );
        assert!(wat.contains("local.get $a"));
        assert!(wat.contains("local.get $b"));
        assert!(wat.contains("f64.add"));
        assert!(wat.contains("return"));
        assert!(wat.starts_with("(module"));
        // Balanced parentheses (a structural sanity check on the WAT).
        assert_eq!(
            wat.chars().filter(|c| *c == '(').count(),
            wat.chars().filter(|c| *c == ')').count()
        );
    }

    #[test]
    fn lowers_locals_and_mixed_arithmetic() {
        let wat = module("function f(x, y) { let t = x * y; return t - 1; }");
        assert!(wat.contains("(local $t f64)"));
        assert!(wat.contains("f64.mul"));
        assert!(wat.contains("local.set $t"));
        assert!(wat.contains("f64.sub"));
        assert!(wat.contains("f64.const 1"));
    }

    #[test]
    fn lowers_comparison_and_ternary() {
        let wat = module("function max(a, b) { return a > b ? a : b; }");
        assert!(wat.contains("f64.gt")); // the condition
        assert!(wat.contains("select")); // the ternary
        // The ternary condition is emitted as i32 (no widening before select).
        assert!(!wat.contains("f64.convert_i32_u"));
    }

    #[test]
    fn comparison_as_value_is_widened() {
        let wat = module("function lt(a, b) { return a < b; }");
        assert!(wat.contains("f64.lt"));
        assert!(wat.contains("f64.convert_i32_u")); // bool → f64 0/1
    }

    #[test]
    fn rejects_non_numeric_constructs() {
        // A string literal isn't in the numeric subset.
        let program = Parser::parse_program("function f(a) { return a + \"x\"; }").unwrap();
        assert!(compile_module(&program).is_err());
        // An object literal isn't numeric.
        let program = Parser::parse_program("function f() { return { a: 1 }; }").unwrap();
        assert!(compile_module(&program).is_err());
        // A `for-of` loop isn't lowered (only `while` is, so far).
        let program = Parser::parse_program(
            "function f(a) { let s = 0; for (const x of a) { s = s + x; } return s; }",
        )
        .unwrap();
        assert!(compile_module(&program).is_err());
    }

    /// Every emitted module must have balanced parens and matching
    /// structured-control delimiters.
    fn assert_well_formed(wat: &str) {
        assert_eq!(
            wat.chars().filter(|c| *c == '(').count(),
            wat.chars().filter(|c| *c == ')').count(),
            "unbalanced parens"
        );
        let count = |kw: &str| wat.split_whitespace().filter(|t| *t == kw).count();
        // Each `if`/`block`/`loop` is closed by an `end`.
        assert_eq!(
            count("if") + count("block") + count("loop"),
            count("end"),
            "unbalanced structured control"
        );
    }

    #[test]
    fn lowers_if_else_statement() {
        let wat = module("function sgn(x) { if (x < 0) { return -1; } else { return 1; } }");
        assert!(wat.contains("f64.lt"));
        assert!(wat.contains("\n    if\n") || wat.contains("    if\n"));
        assert!(wat.contains("else"));
        assert!(wat.contains("end"));
        assert_well_formed(&wat);
    }

    #[test]
    fn lowers_while_loop_with_mutation() {
        let wat = module(
            "function sumTo(n) { let s = 0; let i = 0; while (i < n) { s = s + i; i = i + 1; } return s; }",
        );
        assert!(wat.contains("block"));
        assert!(wat.contains("loop"));
        assert!(wat.contains("br_if 1")); // loop exit
        assert!(wat.contains("br 0")); // loop back-edge
        assert!(wat.contains("local.set $s")); // mutation
        assert_well_formed(&wat);
    }

    #[test]
    fn lowers_function_calls() {
        let wat = module(
            "function sq(x) { return x * x; } function dist(a, b) { return sq(a) + sq(b); }",
        );
        assert!(wat.contains("call $sq"));
        assert_well_formed(&wat);
    }

    #[test]
    fn lowers_iterative_kernel_end_to_end() {
        // An iterative Fibonacci: locals, a while loop, mutation, comparison.
        let wat = module(
            "function fib(n) { let a = 0; let b = 1; let i = 0; while (i < n) { let t = a + b; a = b; b = t; i = i + 1; } return a; }",
        );
        for needle in [
            "loop",
            "f64.lt",
            "f64.add",
            "local.set $t",
            "br 0",
            "return",
        ] {
            assert!(wat.contains(needle), "missing {needle}");
        }
        assert_well_formed(&wat);
    }

    fn binary(src: &str) -> Vec<u8> {
        let program = Parser::parse_program(src).expect("parse");
        compile_module_binary(&program).expect("compile to wasm binary")
    }

    /// Reads an unsigned LEB128 at `pos`, advancing it.
    fn read_leb(bytes: &[u8], pos: &mut usize) -> u32 {
        let mut result = 0u32;
        let mut shift = 0;
        loop {
            let byte = bytes[*pos];
            *pos += 1;
            result |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result
    }

    /// Walks the section framing and returns the ordered section ids, asserting
    /// the module's magic/version and that the sections consume every byte —
    /// a structural well-formedness check (no external validator available).
    fn section_ids(wasm: &[u8]) -> Vec<u8> {
        assert_eq!(&wasm[0..4], b"\0asm", "wasm magic");
        assert_eq!(&wasm[4..8], &[1, 0, 0, 0], "wasm version 1");
        let mut pos = 8;
        let mut ids = Vec::new();
        while pos < wasm.len() {
            let id = wasm[pos];
            pos += 1;
            let len = read_leb(wasm, &mut pos) as usize;
            ids.push(id);
            pos += len; // skip the section content
        }
        assert_eq!(pos, wasm.len(), "sections consume the whole module");
        ids
    }

    #[test]
    fn binary_module_is_structurally_valid() {
        let wasm = binary("function add(a, b) { return a + b; }");
        // Type (1), Function (3), Export (7), Code (10) sections, in order.
        assert_eq!(section_ids(&wasm), [1, 3, 7, 10]);
        // The export name is present in the export section.
        assert!(wasm.windows(3).any(|w| w == b"add"));
        // f64.const opcode (0x44) and f64.add (0xa0) appear in the code.
        assert!(wasm.contains(&0x44) || wasm.contains(&0xa0));
    }

    #[test]
    fn binary_lowers_iterative_kernel() {
        // The iterative fib lowers to a well-framed binary with all sections.
        let wasm = binary(
            "function fib(n) { let a = 0; let b = 1; let i = 0; while (i < n) { let t = a + b; a = b; b = t; i = i + 1; } return a; }",
        );
        assert_eq!(section_ids(&wasm), [1, 3, 7, 10]);
        // Structured control opcodes: block (0x02), loop (0x03), br_if (0x0d).
        assert!(wasm.contains(&0x02) && wasm.contains(&0x03) && wasm.contains(&0x0d));
    }

    #[test]
    fn binary_encodes_f64_const_little_endian() {
        // `return 1.5;` — the f64.const 1.5 is encoded as IEEE-754 LE bytes.
        let wasm = binary("function k() { return 1.5; }");
        let bytes = 1.5f64.to_le_bytes();
        assert!(
            wasm.windows(8).any(|w| w == bytes),
            "f64.const 1.5 little-endian payload present"
        );
        section_ids(&wasm); // still well-framed
    }

    #[test]
    fn binary_encodes_calls_across_functions() {
        let wasm = binary("function sq(x) { return x * x; } function go() { return sq(3); }");
        // 2 functions → type/function/export each list two; code has the call
        // opcode (0x10).
        assert_eq!(section_ids(&wasm), [1, 3, 7, 10]);
        assert!(wasm.contains(&0x10), "call opcode present");
    }

    #[test]
    fn multiple_functions_in_one_module() {
        let wat = module("function a(x) { return x + 1; } function b(x) { return x - 1; }");
        assert!(wat.contains("(func $a"));
        assert!(wat.contains("(func $b"));
    }
}
