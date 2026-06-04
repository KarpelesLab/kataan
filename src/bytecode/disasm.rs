//! A human-readable disassembler for [`Module`]s.

use super::{Const, Module, Op};
use alloc::format;
use alloc::string::String;

/// Renders the whole module as text (one chunk per section).
pub(super) fn disassemble(module: &Module) -> String {
    let mut out = String::new();
    for (i, chunk) in module.chunks.iter().enumerate() {
        out.push_str(&format!(
            "chunk #{i} {:?}  (regs={}, params={})\n",
            chunk.name, chunk.register_count, chunk.param_count
        ));
        if !chunk.constants.is_empty() {
            out.push_str("  constants:\n");
            for (k, c) in chunk.constants.iter().enumerate() {
                out.push_str(&format!("    [{k}] {}\n", render_const(c)));
            }
        }
        for (pc, op) in chunk.code.iter().enumerate() {
            out.push_str(&format!("  {pc:>4}  {}\n", render_op(op)));
        }
    }
    out
}

fn render_const(c: &Const) -> String {
    match c {
        Const::Number(n) => format!("Number {n}"),
        Const::Str(s) => format!("Str {s:?}"),
        Const::Func(i) => format!("Func #{i}"),
    }
}

#[allow(clippy::too_many_lines)]
fn render_op(op: &Op) -> String {
    match op {
        Op::LoadConst { dst, k } => format!("LoadConst   r{dst}, k{k}"),
        Op::LoadUndefined { dst } => format!("LoadUndef   r{dst}"),
        Op::LoadNull { dst } => format!("LoadNull    r{dst}"),
        Op::LoadBool { dst, value } => format!("LoadBool    r{dst}, {value}"),
        Op::LoadInt { dst, value } => format!("LoadInt     r{dst}, {value}"),
        Op::Move { dst, src } => format!("Move        r{dst}, r{src}"),
        Op::Add { dst, a, b } => format!("Add         r{dst}, r{a}, r{b}"),
        Op::Sub { dst, a, b } => format!("Sub         r{dst}, r{a}, r{b}"),
        Op::Mul { dst, a, b } => format!("Mul         r{dst}, r{a}, r{b}"),
        Op::Div { dst, a, b } => format!("Div         r{dst}, r{a}, r{b}"),
        Op::Mod { dst, a, b } => format!("Mod         r{dst}, r{a}, r{b}"),
        Op::Pow { dst, a, b } => format!("Pow         r{dst}, r{a}, r{b}"),
        Op::Neg { dst, src } => format!("Neg         r{dst}, r{src}"),
        Op::Not { dst, src } => format!("Not         r{dst}, r{src}"),
        Op::Eq { dst, a, b } => format!("Eq          r{dst}, r{a}, r{b}"),
        Op::StrictEq { dst, a, b } => format!("StrictEq    r{dst}, r{a}, r{b}"),
        Op::Lt { dst, a, b } => format!("Lt          r{dst}, r{a}, r{b}"),
        Op::Le { dst, a, b } => format!("Le          r{dst}, r{a}, r{b}"),
        Op::Gt { dst, a, b } => format!("Gt          r{dst}, r{a}, r{b}"),
        Op::Ge { dst, a, b } => format!("Ge          r{dst}, r{a}, r{b}"),
        Op::GetGlobal { dst, name } => format!("GetGlobal   r{dst}, k{name}"),
        Op::SetGlobal { name, src } => format!("SetGlobal   k{name}, r{src}"),
        Op::NewObject { dst } => format!("NewObject   r{dst}"),
        Op::NewArray { dst, len } => format!("NewArray    r{dst}, {len}"),
        Op::GetProp { dst, obj, key } => format!("GetProp     r{dst}, r{obj}, k{key}"),
        Op::SetProp { obj, key, src } => format!("SetProp     r{obj}, k{key}, r{src}"),
        Op::GetElem { dst, obj, index } => format!("GetElem     r{dst}, r{obj}, r{index}"),
        Op::SetElem { obj, index, src } => format!("SetElem     r{obj}, r{index}, r{src}"),
        Op::Jump { offset } => format!("Jump        {offset:+}"),
        Op::JumpIfFalse { cond, offset } => format!("JumpIfFalse r{cond}, {offset:+}"),
        Op::JumpIfTrue { cond, offset } => format!("JumpIfTrue  r{cond}, {offset:+}"),
        Op::Call {
            dst,
            callee,
            args_base,
            argc,
        } => format!("Call        r{dst}, r{callee}, r{args_base}..+{argc}"),
        Op::CallMethod {
            dst,
            recv,
            key,
            args_base,
            argc,
        } => format!("CallMethod  r{dst}, r{recv}[r{key}], r{args_base}..+{argc}"),
        Op::Return { src } => format!("Return      r{src}"),
        Op::ReturnUndefined => "ReturnUndef".into(),
        Op::Throw { src } => format!("Throw       r{src}"),
        Op::PushHandler { catch, err } => format!("PushHandler {catch:+} -> r{err}"),
        Op::PopHandler => "PopHandler".into(),
    }
}
