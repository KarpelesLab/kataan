//! Serialization of a [`Module`] to/from the host-native binary container.
//!
//! Integers are written in the host's native byte order; the header records
//! which, so a matched host could later map the buffer zero-copy while a
//! mismatched host reads it back through the recorded order (convert-on-demand).

use super::{Chunk, Const, MAGIC, Module, Op, VERSION};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// An error from [`deserialize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeError {
    /// The magic bytes did not match.
    BadMagic,
    /// The format version is not supported (recompile from source).
    UnsupportedVersion(u16),
    /// The buffer ended unexpectedly.
    Truncated,
    /// An unknown opcode / constant tag.
    BadTag(u8),
    /// A string was not valid UTF-8.
    BadUtf8,
}

impl fmt::Display for BytecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BytecodeError::BadMagic => write!(f, "not a Kataan bytecode module (bad magic)"),
            BytecodeError::UnsupportedVersion(v) => {
                write!(f, "unsupported bytecode version {v} (expected {VERSION})")
            }
            BytecodeError::Truncated => write!(f, "truncated bytecode buffer"),
            BytecodeError::BadTag(t) => write!(f, "invalid bytecode tag {t}"),
            BytecodeError::BadUtf8 => write!(f, "invalid UTF-8 in bytecode string"),
        }
    }
}

/// The host byte-order marker: `0` little-endian, `1` big-endian.
const fn host_endian() -> u8 {
    if cfg!(target_endian = "little") { 0 } else { 1 }
}

/// Serializes `module` to the host-native binary container.
#[must_use]
pub fn serialize(module: &Module) -> Vec<u8> {
    let mut w = Writer::new();
    w.bytes(&MAGIC);
    w.u16(VERSION);
    w.raw(host_endian());
    w.raw(core::mem::size_of::<usize>() as u8); // pointer width (informational)
    w.raw(0); // reserved
    w.raw(0); // reserved (header is 4-byte aligned)
    w.u32(module.chunks.len() as u32);
    for chunk in &module.chunks {
        write_chunk(&mut w, chunk);
    }
    w.into_bytes()
}

/// Deserializes a module previously produced by [`serialize`].
pub fn deserialize(bytes: &[u8]) -> Result<Module, BytecodeError> {
    let mut r = Reader::new(bytes);
    if r.bytes(4)? != MAGIC {
        return Err(BytecodeError::BadMagic);
    }
    let version = r.u16()?;
    if version != VERSION {
        return Err(BytecodeError::UnsupportedVersion(version));
    }
    let endian = r.raw()?; // recorded byte order governs the rest
    r.set_endian(endian);
    let _ptr_width = r.raw()?;
    let _reserved = (r.raw()?, r.raw()?);
    let chunk_count = r.u32()? as usize;
    let mut chunks = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        chunks.push(read_chunk(&mut r)?);
    }
    Ok(Module { chunks })
}

fn write_chunk(w: &mut Writer, chunk: &Chunk) {
    w.string(&chunk.name);
    w.u16(chunk.register_count);
    w.u16(chunk.param_count);
    w.u32(chunk.constants.len() as u32);
    for c in &chunk.constants {
        write_const(w, c);
    }
    w.u32(chunk.code.len() as u32);
    for op in &chunk.code {
        write_op(w, op);
    }
}

fn read_chunk(r: &mut Reader) -> Result<Chunk, BytecodeError> {
    let name = r.string()?;
    let register_count = r.u16()?;
    let param_count = r.u16()?;
    let const_count = r.u32()? as usize;
    let mut constants = Vec::with_capacity(const_count);
    for _ in 0..const_count {
        constants.push(read_const(r)?);
    }
    let code_len = r.u32()? as usize;
    let mut code = Vec::with_capacity(code_len);
    for _ in 0..code_len {
        code.push(read_op(r)?);
    }
    Ok(Chunk {
        name,
        register_count,
        param_count,
        constants,
        code,
    })
}

fn write_const(w: &mut Writer, c: &Const) {
    match c {
        Const::Number(n) => {
            w.raw(0);
            w.f64(*n);
        }
        Const::Str(s) => {
            w.raw(1);
            w.string(s);
        }
        Const::Func(i) => {
            w.raw(2);
            w.u32(*i);
        }
    }
}

fn read_const(r: &mut Reader) -> Result<Const, BytecodeError> {
    match r.raw()? {
        0 => Ok(Const::Number(r.f64()?)),
        1 => Ok(Const::Str(r.string()?)),
        2 => Ok(Const::Func(r.u32()?)),
        t => Err(BytecodeError::BadTag(t)),
    }
}

/// The opcode tag table (the numeric values are part of the format).
mod tag {
    pub(super) const LOAD_CONST: u8 = 0;
    pub(super) const LOAD_UNDEFINED: u8 = 1;
    pub(super) const LOAD_NULL: u8 = 2;
    pub(super) const LOAD_BOOL: u8 = 3;
    pub(super) const LOAD_INT: u8 = 4;
    pub(super) const MOVE: u8 = 5;
    pub(super) const ADD: u8 = 6;
    pub(super) const SUB: u8 = 7;
    pub(super) const MUL: u8 = 8;
    pub(super) const DIV: u8 = 9;
    pub(super) const MOD: u8 = 10;
    pub(super) const POW: u8 = 11;
    pub(super) const NEG: u8 = 12;
    pub(super) const NOT: u8 = 13;
    pub(super) const EQ: u8 = 14;
    pub(super) const STRICT_EQ: u8 = 15;
    pub(super) const LT: u8 = 16;
    pub(super) const LE: u8 = 17;
    pub(super) const GT: u8 = 18;
    pub(super) const GE: u8 = 19;
    pub(super) const GET_GLOBAL: u8 = 20;
    pub(super) const SET_GLOBAL: u8 = 21;
    pub(super) const NEW_OBJECT: u8 = 22;
    pub(super) const NEW_ARRAY: u8 = 23;
    pub(super) const GET_PROP: u8 = 24;
    pub(super) const SET_PROP: u8 = 25;
    pub(super) const GET_ELEM: u8 = 26;
    pub(super) const SET_ELEM: u8 = 27;
    pub(super) const JUMP: u8 = 28;
    pub(super) const JUMP_IF_FALSE: u8 = 29;
    pub(super) const JUMP_IF_TRUE: u8 = 30;
    pub(super) const CALL: u8 = 31;
    pub(super) const RETURN: u8 = 32;
    pub(super) const RETURN_UNDEFINED: u8 = 33;
    pub(super) const CALL_METHOD: u8 = 34;
    pub(super) const THROW: u8 = 35;
    pub(super) const PUSH_HANDLER: u8 = 36;
    pub(super) const POP_HANDLER: u8 = 37;
    pub(super) const NEW: u8 = 38;
    pub(super) const BINARY: u8 = 39;
}

fn write_op(w: &mut Writer, op: &Op) {
    match op {
        Op::LoadConst { dst, k } => {
            w.raw(tag::LOAD_CONST);
            w.u16(*dst);
            w.u32(*k);
        }
        Op::LoadUndefined { dst } => {
            w.raw(tag::LOAD_UNDEFINED);
            w.u16(*dst);
        }
        Op::LoadNull { dst } => {
            w.raw(tag::LOAD_NULL);
            w.u16(*dst);
        }
        Op::LoadBool { dst, value } => {
            w.raw(tag::LOAD_BOOL);
            w.u16(*dst);
            w.raw(u8::from(*value));
        }
        Op::LoadInt { dst, value } => {
            w.raw(tag::LOAD_INT);
            w.u16(*dst);
            w.i32(*value);
        }
        Op::Move { dst, src } => {
            w.raw(tag::MOVE);
            w.u16(*dst);
            w.u16(*src);
        }
        Op::Add { dst, a, b } => write_abc(w, tag::ADD, *dst, *a, *b),
        Op::Sub { dst, a, b } => write_abc(w, tag::SUB, *dst, *a, *b),
        Op::Mul { dst, a, b } => write_abc(w, tag::MUL, *dst, *a, *b),
        Op::Div { dst, a, b } => write_abc(w, tag::DIV, *dst, *a, *b),
        Op::Mod { dst, a, b } => write_abc(w, tag::MOD, *dst, *a, *b),
        Op::Pow { dst, a, b } => write_abc(w, tag::POW, *dst, *a, *b),
        Op::Neg { dst, src } => {
            w.raw(tag::NEG);
            w.u16(*dst);
            w.u16(*src);
        }
        Op::Binary { dst, a, b, op } => {
            w.raw(tag::BINARY);
            w.u16(*dst);
            w.u16(*a);
            w.u16(*b);
            w.raw(*op);
        }
        Op::Not { dst, src } => {
            w.raw(tag::NOT);
            w.u16(*dst);
            w.u16(*src);
        }
        Op::Eq { dst, a, b } => write_abc(w, tag::EQ, *dst, *a, *b),
        Op::StrictEq { dst, a, b } => write_abc(w, tag::STRICT_EQ, *dst, *a, *b),
        Op::Lt { dst, a, b } => write_abc(w, tag::LT, *dst, *a, *b),
        Op::Le { dst, a, b } => write_abc(w, tag::LE, *dst, *a, *b),
        Op::Gt { dst, a, b } => write_abc(w, tag::GT, *dst, *a, *b),
        Op::Ge { dst, a, b } => write_abc(w, tag::GE, *dst, *a, *b),
        Op::GetGlobal { dst, name } => {
            w.raw(tag::GET_GLOBAL);
            w.u16(*dst);
            w.u32(*name);
        }
        Op::SetGlobal { name, src } => {
            w.raw(tag::SET_GLOBAL);
            w.u32(*name);
            w.u16(*src);
        }
        Op::NewObject { dst } => {
            w.raw(tag::NEW_OBJECT);
            w.u16(*dst);
        }
        Op::NewArray { dst, len } => {
            w.raw(tag::NEW_ARRAY);
            w.u16(*dst);
            w.u32(*len);
        }
        Op::GetProp { dst, obj, key } => {
            w.raw(tag::GET_PROP);
            w.u16(*dst);
            w.u16(*obj);
            w.u32(*key);
        }
        Op::SetProp { obj, key, src } => {
            w.raw(tag::SET_PROP);
            w.u16(*obj);
            w.u32(*key);
            w.u16(*src);
        }
        Op::GetElem { dst, obj, index } => write_abc(w, tag::GET_ELEM, *dst, *obj, *index),
        Op::SetElem { obj, index, src } => write_abc(w, tag::SET_ELEM, *obj, *index, *src),
        Op::Jump { offset } => {
            w.raw(tag::JUMP);
            w.i32(*offset);
        }
        Op::JumpIfFalse { cond, offset } => {
            w.raw(tag::JUMP_IF_FALSE);
            w.u16(*cond);
            w.i32(*offset);
        }
        Op::JumpIfTrue { cond, offset } => {
            w.raw(tag::JUMP_IF_TRUE);
            w.u16(*cond);
            w.i32(*offset);
        }
        Op::Call {
            dst,
            callee,
            args_base,
            argc,
        } => {
            w.raw(tag::CALL);
            w.u16(*dst);
            w.u16(*callee);
            w.u16(*args_base);
            w.u16(*argc);
        }
        Op::New {
            dst,
            callee,
            args_base,
            argc,
        } => {
            w.raw(tag::NEW);
            w.u16(*dst);
            w.u16(*callee);
            w.u16(*args_base);
            w.u16(*argc);
        }
        Op::CallMethod {
            dst,
            recv,
            key,
            args_base,
            argc,
        } => {
            w.raw(tag::CALL_METHOD);
            w.u16(*dst);
            w.u16(*recv);
            w.u16(*key);
            w.u16(*args_base);
            w.u16(*argc);
        }
        Op::Return { src } => {
            w.raw(tag::RETURN);
            w.u16(*src);
        }
        Op::ReturnUndefined => w.raw(tag::RETURN_UNDEFINED),
        Op::Throw { src } => {
            w.raw(tag::THROW);
            w.u16(*src);
        }
        Op::PushHandler { catch, err } => {
            w.raw(tag::PUSH_HANDLER);
            w.i32(*catch);
            w.u16(*err);
        }
        Op::PopHandler => w.raw(tag::POP_HANDLER),
    }
}

fn write_abc(w: &mut Writer, tag: u8, dst: u16, a: u16, b: u16) {
    w.raw(tag);
    w.u16(dst);
    w.u16(a);
    w.u16(b);
}

fn read_op(r: &mut Reader) -> Result<Op, BytecodeError> {
    Ok(match r.raw()? {
        tag::LOAD_CONST => Op::LoadConst {
            dst: r.u16()?,
            k: r.u32()?,
        },
        tag::LOAD_UNDEFINED => Op::LoadUndefined { dst: r.u16()? },
        tag::LOAD_NULL => Op::LoadNull { dst: r.u16()? },
        tag::LOAD_BOOL => Op::LoadBool {
            dst: r.u16()?,
            value: r.raw()? != 0,
        },
        tag::LOAD_INT => Op::LoadInt {
            dst: r.u16()?,
            value: r.i32()?,
        },
        tag::MOVE => Op::Move {
            dst: r.u16()?,
            src: r.u16()?,
        },
        tag::ADD => abc(r, |dst, a, b| Op::Add { dst, a, b })?,
        tag::SUB => abc(r, |dst, a, b| Op::Sub { dst, a, b })?,
        tag::MUL => abc(r, |dst, a, b| Op::Mul { dst, a, b })?,
        tag::DIV => abc(r, |dst, a, b| Op::Div { dst, a, b })?,
        tag::MOD => abc(r, |dst, a, b| Op::Mod { dst, a, b })?,
        tag::POW => abc(r, |dst, a, b| Op::Pow { dst, a, b })?,
        tag::NEG => Op::Neg {
            dst: r.u16()?,
            src: r.u16()?,
        },
        tag::NOT => Op::Not {
            dst: r.u16()?,
            src: r.u16()?,
        },
        tag::EQ => abc(r, |dst, a, b| Op::Eq { dst, a, b })?,
        tag::STRICT_EQ => abc(r, |dst, a, b| Op::StrictEq { dst, a, b })?,
        tag::LT => abc(r, |dst, a, b| Op::Lt { dst, a, b })?,
        tag::LE => abc(r, |dst, a, b| Op::Le { dst, a, b })?,
        tag::GT => abc(r, |dst, a, b| Op::Gt { dst, a, b })?,
        tag::GE => abc(r, |dst, a, b| Op::Ge { dst, a, b })?,
        tag::GET_GLOBAL => Op::GetGlobal {
            dst: r.u16()?,
            name: r.u32()?,
        },
        tag::SET_GLOBAL => Op::SetGlobal {
            name: r.u32()?,
            src: r.u16()?,
        },
        tag::NEW_OBJECT => Op::NewObject { dst: r.u16()? },
        tag::NEW_ARRAY => Op::NewArray {
            dst: r.u16()?,
            len: r.u32()?,
        },
        tag::GET_PROP => Op::GetProp {
            dst: r.u16()?,
            obj: r.u16()?,
            key: r.u32()?,
        },
        tag::SET_PROP => Op::SetProp {
            obj: r.u16()?,
            key: r.u32()?,
            src: r.u16()?,
        },
        tag::GET_ELEM => abc(r, |dst, obj, index| Op::GetElem { dst, obj, index })?,
        tag::SET_ELEM => abc(r, |obj, index, src| Op::SetElem { obj, index, src })?,
        tag::JUMP => Op::Jump { offset: r.i32()? },
        tag::JUMP_IF_FALSE => Op::JumpIfFalse {
            cond: r.u16()?,
            offset: r.i32()?,
        },
        tag::JUMP_IF_TRUE => Op::JumpIfTrue {
            cond: r.u16()?,
            offset: r.i32()?,
        },
        tag::CALL => Op::Call {
            dst: r.u16()?,
            callee: r.u16()?,
            args_base: r.u16()?,
            argc: r.u16()?,
        },
        tag::BINARY => Op::Binary {
            dst: r.u16()?,
            a: r.u16()?,
            b: r.u16()?,
            op: r.raw()?,
        },
        tag::NEW => Op::New {
            dst: r.u16()?,
            callee: r.u16()?,
            args_base: r.u16()?,
            argc: r.u16()?,
        },
        tag::CALL_METHOD => Op::CallMethod {
            dst: r.u16()?,
            recv: r.u16()?,
            key: r.u16()?,
            args_base: r.u16()?,
            argc: r.u16()?,
        },
        tag::THROW => Op::Throw { src: r.u16()? },
        tag::PUSH_HANDLER => Op::PushHandler {
            catch: r.i32()?,
            err: r.u16()?,
        },
        tag::POP_HANDLER => Op::PopHandler,
        tag::RETURN => Op::Return { src: r.u16()? },
        tag::RETURN_UNDEFINED => Op::ReturnUndefined,
        t => return Err(BytecodeError::BadTag(t)),
    })
}

fn abc(r: &mut Reader, build: impl Fn(u16, u16, u16) -> Op) -> Result<Op, BytecodeError> {
    Ok(build(r.u16()?, r.u16()?, r.u16()?))
}

// --- byte cursor helpers ---------------------------------------------------

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
    fn raw(&mut self, b: u8) {
        self.buf.push(b);
    }
    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_ne_bytes());
    }
    fn string(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.bytes(s.as_bytes());
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    /// `false` little-endian, `true` big-endian (the buffer's recorded order).
    big_endian: bool,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            big_endian: false,
        }
    }
    fn set_endian(&mut self, marker: u8) {
        self.big_endian = marker == 1;
    }
    fn raw(&mut self) -> Result<u8, BytecodeError> {
        let b = *self.buf.get(self.pos).ok_or(BytecodeError::Truncated)?;
        self.pos += 1;
        Ok(b)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], BytecodeError> {
        let end = self.pos.checked_add(n).ok_or(BytecodeError::Truncated)?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(BytecodeError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }
    fn u16(&mut self) -> Result<u16, BytecodeError> {
        let b: [u8; 2] = self.bytes(2)?.try_into().unwrap();
        Ok(if self.big_endian {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    }
    fn u32(&mut self) -> Result<u32, BytecodeError> {
        let b: [u8; 4] = self.bytes(4)?.try_into().unwrap();
        Ok(if self.big_endian {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    }
    fn i32(&mut self) -> Result<i32, BytecodeError> {
        Ok(self.u32()? as i32)
    }
    fn f64(&mut self) -> Result<f64, BytecodeError> {
        let b: [u8; 8] = self.bytes(8)?.try_into().unwrap();
        Ok(if self.big_endian {
            f64::from_be_bytes(b)
        } else {
            f64::from_le_bytes(b)
        })
    }
    fn string(&mut self) -> Result<String, BytecodeError> {
        let len = self.u32()? as usize;
        let bytes = self.bytes(len)?;
        core::str::from_utf8(bytes)
            .map(String::from)
            .map_err(|_| BytecodeError::BadUtf8)
    }
}
