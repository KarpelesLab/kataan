//! A WebAssembly backend — the **WASM peer engine** (`ROADMAP.md`).
//!
//! Alongside the tree-walker and the register VM, this lowers the *numeric*
//! subset of JavaScript functions to WebAssembly text (WAT): a function over
//! `f64` parameters whose body is `let` bindings, arithmetic, comparisons,
//! ternaries, and `return` becomes a `(func …)` in a `(module …)`. The output is
//! valid WAT that a standard assembler turns into a `.wasm` module — so a hot
//! numeric kernel can be handed to a Wasm runtime instead of interpreted.
//!
//! This is a first slice: an expression-oriented lowering (no loops/branches
//! yet). It is engine code, not foreign — pure, safe `alloc`-only Rust emitting
//! text. Unsupported constructs are reported rather than mis-compiled.

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
/// Returns [`WasmError`] for a non-numeric construct (objects, strings, loops,
/// calls, …) or a destructuring/rest parameter.
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

/// Collects the names introduced by top-level `let`/`const` in the body.
fn collect_locals(body: &[Stmt], out: &mut Vec<String>) -> Result<(), WasmError> {
    for stmt in body {
        if let Stmt::Var(decl) = stmt {
            for d in &decl.declarations {
                let BindingTarget::Ident(id) = &d.target else {
                    return Err(WasmError("destructuring binding"));
                };
                out.push(id.name.to_string());
            }
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
        // String concatenation isn't in the numeric subset.
        let program = Parser::parse_program("function f(a) { return a + \"x\"; }").unwrap();
        // The string literal becomes an unsupported expression.
        assert!(compile_module(&program).is_err());
        // Loops aren't lowered yet.
        let program =
            Parser::parse_program("function f(a) { while (a) { a = a - 1; } return a; }").unwrap();
        assert!(compile_module(&program).is_err());
    }

    #[test]
    fn multiple_functions_in_one_module() {
        let wat = module("function a(x) { return x + 1; } function b(x) { return x - 1; }");
        assert!(wat.contains("(func $a"));
        assert!(wat.contains("(func $b"));
    }
}
