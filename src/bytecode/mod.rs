//! The Kataan bytecode: a register-based instruction set, plus the
//! **host-native serializable** container format that lets compiled code be
//! exported and reloaded (the code-cache use case in `ROADMAP.md` §2.2).
//!
//! The serialized form records the host's byte order and pointer width in its
//! header. On a host whose encoding matches, the buffer can eventually be
//! mapped and used zero-copy; on a mismatched host it is read back through the
//! recorded encoding (convert-on-demand) — so the format is portable in
//! *meaning* without paying for portability on the common matched-host path.
//!
//! This module defines the instruction set, the [`Chunk`]/[`Module`]
//! containers, and the [`serialize`]/[`deserialize`] codec with a disassembler.
//! The AST→bytecode compiler and the VM that executes it land on top (Phase D).

mod disasm;
mod io;

#[cfg(test)]
mod tests;

use alloc::string::String;
use alloc::vec::Vec;

pub use io::{BytecodeError, deserialize, serialize};

/// The format magic (`"KTBC"`) at the start of a serialized module.
pub const MAGIC: [u8; 4] = *b"KTBC";

/// The bytecode format version. Mismatched versions are rejected on load (the
/// source is recompiled rather than converted).
pub const VERSION: u16 = 1;

/// A virtual register / local slot index.
pub type Reg = u16;

/// Operator codes for the generic [`Op::Binary`] instruction (stable, part of
/// the serialized format).
#[allow(missing_docs)]
pub mod binop {
    pub const BIT_AND: u8 = 0;
    pub const BIT_OR: u8 = 1;
    pub const BIT_XOR: u8 = 2;
    pub const SHL: u8 = 3;
    pub const SHR: u8 = 4;
    pub const USHR: u8 = 5;
    pub const IN: u8 = 6;
    pub const INSTANCEOF: u8 = 7;
}

/// An index into a chunk's constant pool.
pub type ConstIdx = u32;

/// A constant-pool entry.
#[derive(Clone, Debug, PartialEq)]
pub enum Const {
    /// An IEEE-754 double.
    Number(f64),
    /// A string (UTF-8).
    Str(String),
    /// A nested function chunk (by index into the module).
    Func(u32),
}

/// A single register-machine instruction.
///
/// Three-address where applicable: `dst` is the destination register, and the
/// other register operands are sources. Jumps carry a signed instruction
/// offset relative to the *next* instruction.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum Op {
    // --- loads ---
    LoadConst {
        dst: Reg,
        k: ConstIdx,
    },
    LoadUndefined {
        dst: Reg,
    },
    LoadNull {
        dst: Reg,
    },
    LoadBool {
        dst: Reg,
        value: bool,
    },
    LoadInt {
        dst: Reg,
        value: i32,
    },
    Move {
        dst: Reg,
        src: Reg,
    },

    // --- arithmetic / logic (dst = a OP b) ---
    Add {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Sub {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Mul {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Div {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Mod {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Pow {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Neg {
        dst: Reg,
        src: Reg,
    },
    Not {
        dst: Reg,
        src: Reg,
    },
    /// A generic binary op (`dst = a OP b`) for operators without a dedicated
    /// instruction — bitwise/shift, `in`, `instanceof`. `op` is a [`binop`]
    /// code.
    Binary {
        dst: Reg,
        a: Reg,
        b: Reg,
        op: u8,
    },

    // --- comparison ---
    Eq {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    StrictEq {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Lt {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Le {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Gt {
        dst: Reg,
        a: Reg,
        b: Reg,
    },
    Ge {
        dst: Reg,
        a: Reg,
        b: Reg,
    },

    // --- variables ---
    GetGlobal {
        dst: Reg,
        name: ConstIdx,
    },
    SetGlobal {
        name: ConstIdx,
        src: Reg,
    },

    // --- objects / arrays ---
    NewObject {
        dst: Reg,
    },
    NewArray {
        dst: Reg,
        len: u32,
    },
    GetProp {
        dst: Reg,
        obj: Reg,
        key: ConstIdx,
    },
    SetProp {
        obj: Reg,
        key: ConstIdx,
        src: Reg,
    },
    GetElem {
        dst: Reg,
        obj: Reg,
        index: Reg,
    },
    SetElem {
        obj: Reg,
        index: Reg,
        src: Reg,
    },

    // --- control flow ---
    Jump {
        offset: i32,
    },
    JumpIfFalse {
        cond: Reg,
        offset: i32,
    },
    JumpIfTrue {
        cond: Reg,
        offset: i32,
    },

    // --- calls ---
    /// `dst = callee(args_base .. args_base+argc)`.
    Call {
        dst: Reg,
        callee: Reg,
        args_base: Reg,
        argc: u16,
    },
    /// `dst = new callee(args_base .. args_base+argc)`.
    New {
        dst: Reg,
        callee: Reg,
        args_base: Reg,
        argc: u16,
    },
    /// `dst = recv[key](args…)` with `recv` bound as `this` (and built-in
    /// prototype-method dispatch). `key` is a register holding the key value.
    CallMethod {
        dst: Reg,
        recv: Reg,
        key: Reg,
        args_base: Reg,
        argc: u16,
    },
    Return {
        src: Reg,
    },
    ReturnUndefined,

    // --- exceptions ---
    /// Throw the value in `src`.
    Throw {
        src: Reg,
    },
    /// Install an exception handler: a throw in the guarded region jumps to
    /// `catch` (a signed offset relative to the next instruction) with the
    /// thrown value placed in register `err`.
    PushHandler {
        catch: i32,
        err: Reg,
    },
    /// Remove the most recently installed handler (the guarded region completed
    /// normally).
    PopHandler,
}

/// A unit of compiled code: a flat instruction list, a constant pool, and
/// metadata (name + register count).
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    /// A human-readable name (function name or `"<main>"`).
    pub name: String,
    /// The number of registers this chunk uses.
    pub register_count: u16,
    /// The number of declared parameters.
    pub param_count: u16,
    /// The constant pool.
    pub constants: Vec<Const>,
    /// The instructions.
    pub code: Vec<Op>,
}

impl Chunk {
    /// Creates an empty chunk with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            register_count: 0,
            param_count: 0,
            constants: Vec::new(),
            code: Vec::new(),
        }
    }

    /// Appends an instruction, returning its index.
    pub fn emit(&mut self, op: Op) -> usize {
        self.code.push(op);
        self.code.len() - 1
    }

    /// Interns a constant, returning its pool index (deduplicated).
    pub fn add_constant(&mut self, value: Const) -> ConstIdx {
        if let Some(i) = self.constants.iter().position(|c| *c == value) {
            return i as ConstIdx;
        }
        self.constants.push(value);
        (self.constants.len() - 1) as ConstIdx
    }
}

/// A complete compiled program: a set of chunks (the first is the entry point).
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    /// The chunks; `chunks[0]` is the top-level/entry chunk.
    pub chunks: Vec<Chunk>,
}

impl Module {
    /// Creates a module from its entry chunk.
    #[must_use]
    pub fn new(entry: Chunk) -> Self {
        Self {
            chunks: alloc::vec![entry],
        }
    }

    /// The entry chunk.
    #[must_use]
    pub fn entry(&self) -> &Chunk {
        &self.chunks[0]
    }

    /// Renders the module as human-readable disassembly.
    #[must_use]
    pub fn disassemble(&self) -> String {
        disasm::disassemble(self)
    }
}
