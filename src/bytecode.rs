//! A portable serialization codec for the bytecode VM's compiled programs
//! (`ROADMAP.md` Phase D′ — serializable bytecode / code cache).
//!
//! A compiled program is a `Vec<FnProto>` (function 0 is the top-level body).
//! `serialize` turns it into a self-describing `KTBC` byte container —
//! `magic + version + function table` — that `deserialize` reloads after
//! validating the magic and version. This is the artifact behind
//! `kataan compile app.js -o app.ktbc` / `kataan run app.ktbc`: parse and compile
//! once, then run from bytecode without re-parsing.
//!
//! Integers are little-endian; strings and lists are length-prefixed; a
//! `LoadConst` value is encoded by its primitive kind (compile-time constants are
//! never heap handles). Pure, safe `alloc`-only Rust.

use crate::nanbox::{NanBox, Unpacked};
use crate::nbvm::{FnProto, Op, Reg};
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

const MAGIC: &[u8; 4] = b"KTBC";
const VERSION: u16 = 2;

/// Why a bytecode artifact failed to load.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// The `KTBC` magic was missing.
    BadMagic,
    /// The version is not understood by this build.
    BadVersion(u16),
    /// The buffer ended mid-record.
    Truncated,
    /// An unknown opcode/value tag (a corrupt or incompatible artifact).
    BadTag(u8),
}

// --- writers ---

fn w_u8(v: u8, out: &mut Vec<u8>) {
    out.push(v);
}
fn w_u16(v: u16, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_u32(v: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_usize(v: usize, out: &mut Vec<u8>) {
    w_u32(v as u32, out);
}
fn w_reg(v: Reg, out: &mut Vec<u8>) {
    w_u16(v, out);
}
fn w_bool(v: bool, out: &mut Vec<u8>) {
    out.push(u8::from(v));
}
fn w_str(s: &str, out: &mut Vec<u8>) {
    w_u32(s.len() as u32, out);
    out.extend_from_slice(s.as_bytes());
}
fn w_regs(rs: &[Reg], out: &mut Vec<u8>) {
    w_u32(rs.len() as u32, out);
    for r in rs {
        w_reg(*r, out);
    }
}
fn w_u32s(xs: &[u32], out: &mut Vec<u8>) {
    w_u32(xs.len() as u32, out);
    for x in xs {
        w_u32(*x, out);
    }
}
fn w_strs(xs: &[String], out: &mut Vec<u8>) {
    w_u32(xs.len() as u32, out);
    for x in xs {
        w_str(x, out);
    }
}

/// Encodes a `LoadConst` value by its primitive kind.
fn w_value(v: NanBox, out: &mut Vec<u8>) {
    match v.unpack() {
        Unpacked::Undefined => w_u8(0, out),
        Unpacked::Null => w_u8(1, out),
        Unpacked::Bool(b) => {
            w_u8(2, out);
            w_bool(b, out);
        }
        Unpacked::Number(n) => {
            w_u8(3, out);
            out.extend_from_slice(&n.to_le_bytes());
        }
        Unpacked::Handle(h) => {
            w_u8(4, out);
            out.extend_from_slice(&h.to_le_bytes());
        }
    }
}

// --- readers ---

/// A cursor over the artifact bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::Truncated)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(DecodeError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, DecodeError> {
        Ok(self.u32()? as usize)
    }
    fn reg(&mut self) -> Result<Reg, DecodeError> {
        self.u16()
    }
    fn boolean(&mut self) -> Result<bool, DecodeError> {
        Ok(self.u8()? != 0)
    }
    fn string(&mut self) -> Result<String, DecodeError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
    fn regs(&mut self) -> Result<Vec<Reg>, DecodeError> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.reg()).collect()
    }
    fn u32s(&mut self) -> Result<Vec<u32>, DecodeError> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.u32()).collect()
    }
    fn strings(&mut self) -> Result<Vec<String>, DecodeError> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.string()).collect()
    }
    fn value(&mut self) -> Result<NanBox, DecodeError> {
        match self.u8()? {
            0 => Ok(NanBox::undefined()),
            1 => Ok(NanBox::null()),
            2 => Ok(NanBox::boolean(self.boolean()?)),
            3 => Ok(NanBox::number(f64::from_le_bytes(
                self.take(8)?.try_into().unwrap(),
            ))),
            4 => Ok(NanBox::handle(u64::from_le_bytes(
                self.take(8)?.try_into().unwrap(),
            ))),
            t => Err(DecodeError::BadTag(t)),
        }
    }
}

/// Why a (well-decoded) artifact failed verification — a reference outside its
/// declared bounds, which an untrusted artifact must not be allowed to run with.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum VerifyError {
    /// A register index `>= n_regs`.
    Register(Reg),
    /// A function index `>= the function-table length`.
    Function(u32),
    /// A jump/handler target outside `0..=ops.len()`.
    Target(usize),
}

/// Verifies a decoded program is **safe to run**: every register reference is in
/// `0..n_regs`, every function index addresses a real function, and every jump /
/// handler target is a valid instruction index. This is the untrusted-load
/// check (`ROADMAP.md` Phase D′) — a corrupt or malicious artifact is rejected
/// here instead of causing an out-of-bounds access during execution.
///
/// # Errors
/// Returns the first [`VerifyError`] found.
pub fn verify(protos: &[FnProto]) -> Result<(), VerifyError> {
    let num_funcs = protos.len();
    for p in protos {
        for op in &p.ops {
            verify_op(op, p.n_regs, num_funcs, p.ops.len())?;
        }
    }
    Ok(())
}

/// Decodes *and* verifies an artifact in one step — the safe entry point for
/// loading untrusted bytecode.
///
/// # Errors
/// `Err(Ok(VerifyError))` if it decodes but fails verification; `Err(Err(..))`
/// if decoding itself fails.
#[allow(clippy::result_large_err)]
pub fn deserialize_verified(
    bytes: &[u8],
) -> Result<Vec<FnProto>, Result<VerifyError, DecodeError>> {
    let protos = deserialize(bytes).map_err(Err)?;
    verify(&protos).map_err(Ok)?;
    Ok(protos)
}

fn verify_op(op: &Op, n_regs: usize, num_funcs: usize, n_ops: usize) -> Result<(), VerifyError> {
    let reg = |r: Reg| {
        if (r as usize) < n_regs {
            Ok(())
        } else {
            Err(VerifyError::Register(r))
        }
    };
    let func = |f: u32| {
        if (f as usize) < num_funcs {
            Ok(())
        } else {
            Err(VerifyError::Function(f))
        }
    };
    let target = |t: usize| {
        if t <= n_ops {
            Ok(())
        } else {
            Err(VerifyError::Target(t))
        }
    };
    let regs = |rs: &[Reg]| -> Result<(), VerifyError> { rs.iter().try_for_each(|r| reg(*r)) };
    match op {
        Op::Add { dst, a, b }
        | Op::Sub { dst, a, b }
        | Op::Mul { dst, a, b }
        | Op::Div { dst, a, b }
        | Op::Mod { dst, a, b }
        | Op::Lt { dst, a, b }
        | Op::AddValue { dst, a, b }
        | Op::StrictEq { dst, a, b }
        | Op::ValueBin { dst, a, b, .. } => {
            reg(*dst)?;
            reg(*a)?;
            reg(*b)
        }
        Op::HasProp { dst, key, obj } | Op::DeleteProp { dst, obj, key } => {
            reg(*dst)?;
            reg(*key)?;
            reg(*obj)
        }
        Op::GetElem {
            dst,
            arr: a,
            index: b,
        }
        | Op::ArraySliceFrom {
            dst,
            src: a,
            from: b,
        }
        | Op::GetKey {
            dst,
            obj: a,
            key: b,
        } => {
            reg(*dst)?;
            reg(*a)?;
            reg(*b)
        }
        Op::SetElem {
            arr: a,
            index: b,
            src: c,
        }
        | Op::SetKey {
            obj: a,
            key: b,
            src: c,
        } => {
            reg(*a)?;
            reg(*b)?;
            reg(*c)
        }
        Op::DefineAccessor {
            obj,
            getter,
            setter,
            ..
        } => {
            reg(*obj)?;
            reg(*getter)?;
            reg(*setter)
        }
        Op::LoadConst { dst, .. }
        | Op::NewString { dst, .. }
        | Op::NewArray { dst, .. }
        | Op::NewRegExp { dst, .. }
        | Op::NewObject { dst } => reg(*dst),
        Op::IsBuiltin { dst, obj, .. }
        | Op::InstanceOf { dst, obj, .. }
        | Op::ObjectSpread { dst, src: obj }
        | Op::EnumKeys { dst, obj }
        | Op::ArrayLen { dst, arr: obj }
        | Op::CollectionSize { dst, recv: obj }
        | Op::ObjectRest { dst, src: obj, .. }
        | Op::TypeOf { dst, a: obj }
        | Op::BitNot { dst, a: obj }
        | Op::Neg { dst, a: obj }
        | Op::Not { dst, a: obj }
        | Op::Move { dst, src: obj }
        | Op::NewArrayCtor { dst, arg: obj }
        | Op::GetProp { dst, obj, .. } => {
            reg(*dst)?;
            reg(*obj)
        }
        Op::ArrayPush { arr: a, src: b } | Op::ArrayExtend { arr: a, src: b } => {
            reg(*a)?;
            reg(*b)
        }
        Op::SetProp { obj, src, .. } => {
            reg(*obj)?;
            reg(*src)
        }
        Op::SetClassTag { obj, .. } => reg(*obj),
        Op::NewCollection { dst, seed, .. } => {
            reg(*dst)?;
            seed.map_or(Ok(()), reg)
        }
        Op::JumpIfFalse { cond, target: t } => {
            reg(*cond)?;
            target(*t)
        }
        Op::Jump { target: t } => target(*t),
        Op::PushHandler { target: t, reg: r } => {
            target(*t)?;
            reg(*r)
        }
        Op::Call { dst, func: f, args } => {
            reg(*dst)?;
            func(*f)?;
            regs(args)
        }
        Op::LoadFunc { dst, func: f } => {
            reg(*dst)?;
            func(*f)
        }
        Op::MakeClosure {
            dst,
            func: f,
            captures,
        } => {
            reg(*dst)?;
            func(*f)?;
            regs(captures)
        }
        Op::CallValue { dst, callee, args } => {
            reg(*dst)?;
            reg(*callee)?;
            regs(args)
        }
        Op::CallValueThis {
            dst,
            callee,
            recv,
            args,
        } => {
            reg(*dst)?;
            reg(*callee)?;
            reg(*recv)?;
            regs(args)
        }
        Op::CallMethod {
            dst, recv, args, ..
        } => {
            reg(*dst)?;
            reg(*recv)?;
            regs(args)
        }
        Op::CallCtor {
            ctor, recv, args, ..
        } => {
            func(*ctor)?;
            reg(*recv)?;
            regs(args)
        }
        Op::CallNative { dst, args, .. } => {
            reg(*dst)?;
            regs(args)
        }
        Op::Throw { src } | Op::Return { src } => reg(*src),
        Op::PopHandler => Ok(()),
    }
}

/// The 64-bit FNV-1a content hash of an artifact — the key a content-addressed
/// store uses so identical bytecode dedups to a single entry.
#[must_use]
pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
    }
    h
}

/// A **content-addressed store** of compiled bytecode artifacts (`ROADMAP.md`
/// §2.2 / Phase D′): an artifact is keyed by its content hash, so byte-identical
/// bytecode — e.g. the same library loaded by many tenants — is stored once and
/// shared. Read-only artifacts are held behind an `Rc`, cheap to hand out across
/// contexts. Supports the load → run → evict → reload churn of a multi-tenant
/// cache.
#[derive(Default)]
pub struct ContentStore {
    entries: alloc::collections::BTreeMap<u64, Rc<[u8]>>,
}

impl ContentStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `artifact` (deduplicating on content) and returns its key. A
    /// second insert of identical bytes is a no-op that returns the same key.
    pub fn put(&mut self, artifact: &[u8]) -> u64 {
        let key = content_hash(artifact);
        self.entries
            .entry(key)
            .or_insert_with(|| Rc::from(artifact));
        key
    }

    /// The shared artifact bytes for `key`, if present.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<Rc<[u8]>> {
        self.entries.get(&key).cloned()
    }

    /// Loads and decodes the artifact at `key` to a runnable function table.
    ///
    /// # Errors
    /// Returns [`DecodeError`] if the stored bytes are corrupt; `None` if absent.
    pub fn load(&self, key: u64) -> Option<Result<Vec<FnProto>, DecodeError>> {
        self.entries.get(&key).map(|bytes| deserialize(bytes))
    }

    /// Whether an artifact with `key` is present.
    #[must_use]
    pub fn contains(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }

    /// Evicts `key`, returning whether it was present.
    pub fn evict(&mut self, key: u64) -> bool {
        self.entries.remove(&key).is_some()
    }

    /// The number of distinct artifacts held (post-dedup).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store holds no artifacts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Serializes a compiled program (the function table) to a `KTBC` artifact.
#[must_use]
pub fn serialize(protos: &[FnProto]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    w_u16(VERSION, &mut out);
    w_u32(protos.len() as u32, &mut out);
    for p in protos {
        w_usize(p.n_regs, &mut out);
        w_usize(p.n_params, &mut out);
        w_usize(p.n_captures, &mut out);
        match p.rest_from {
            Some(r) => {
                w_u8(1, &mut out);
                w_usize(r, &mut out);
            }
            None => w_u8(0, &mut out),
        }
        w_bool(p.is_async, &mut out);
        w_usize(p.length, &mut out);
        w_u32(p.ops.len() as u32, &mut out);
        for op in &p.ops {
            write_op(op, &mut out);
        }
    }
    out
}

/// Reloads a compiled program from a `KTBC` artifact.
///
/// # Errors
/// Returns [`DecodeError`] for a bad magic/version, truncation, or corrupt tag.
pub fn deserialize(bytes: &[u8]) -> Result<Vec<FnProto>, DecodeError> {
    let mut r = Reader { bytes, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let version = r.u16()?;
    if version != VERSION {
        return Err(DecodeError::BadVersion(version));
    }
    let n = r.u32()? as usize;
    let mut protos = Vec::with_capacity(n);
    for _ in 0..n {
        let n_regs = r.usize()?;
        let n_params = r.usize()?;
        let n_captures = r.usize()?;
        let rest_from = if r.u8()? == 1 { Some(r.usize()?) } else { None };
        let is_async = r.boolean()?;
        let length = r.usize()?;
        let n_ops = r.u32()? as usize;
        let mut ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            ops.push(read_op(&mut r)?);
        }
        protos.push(FnProto {
            ops,
            n_regs,
            n_params,
            n_captures,
            rest_from,
            is_async,
            length,
            // The serialized bytecode format does not yet carry function names.
            name: alloc::string::String::new(),
        });
    }
    Ok(protos)
}

/// Writes one op: a tag byte followed by its fields.
fn write_op(op: &Op, out: &mut Vec<u8>) {
    // A 3-register op: `tag dst a b`.
    let bin = |tag: u8, dst: Reg, a: Reg, b: Reg, out: &mut Vec<u8>| {
        w_u8(tag, out);
        w_reg(dst, out);
        w_reg(a, out);
        w_reg(b, out);
    };
    let un = |tag: u8, dst: Reg, a: Reg, out: &mut Vec<u8>| {
        w_u8(tag, out);
        w_reg(dst, out);
        w_reg(a, out);
    };
    match op {
        Op::LoadConst { dst, value } => {
            w_u8(0, out);
            w_reg(*dst, out);
            w_value(*value, out);
        }
        Op::Add { dst, a, b } => bin(1, *dst, *a, *b, out),
        Op::Sub { dst, a, b } => bin(2, *dst, *a, *b, out),
        Op::Mul { dst, a, b } => bin(3, *dst, *a, *b, out),
        Op::Div { dst, a, b } => bin(4, *dst, *a, *b, out),
        Op::Mod { dst, a, b } => bin(5, *dst, *a, *b, out),
        Op::ValueBin { dst, op, a, b } => {
            w_u8(6, out);
            w_reg(*dst, out);
            w_u8(*op, out);
            w_reg(*a, out);
            w_reg(*b, out);
        }
        Op::HasProp { dst, key, obj } => bin(7, *dst, *key, *obj, out),
        Op::IsBuiltin { dst, obj, kind } => {
            w_u8(8, out);
            w_reg(*dst, out);
            w_reg(*obj, out);
            w_u8(*kind, out);
        }
        Op::DeleteProp { dst, obj, key } => bin(9, *dst, *obj, *key, out),
        Op::SetClassTag { obj, class_id } => {
            w_u8(10, out);
            w_reg(*obj, out);
            w_u32(*class_id, out);
        }
        Op::DefineAccessor {
            obj,
            key,
            getter,
            setter,
        } => {
            w_u8(11, out);
            w_reg(*obj, out);
            w_str(key, out);
            w_reg(*getter, out);
            w_reg(*setter, out);
        }
        Op::InstanceOf { dst, obj, ids } => {
            w_u8(12, out);
            w_reg(*dst, out);
            w_reg(*obj, out);
            w_u32s(ids, out);
        }
        Op::TypeOf { dst, a } => un(13, *dst, *a, out),
        Op::BitNot { dst, a } => un(14, *dst, *a, out),
        Op::Neg { dst, a } => un(15, *dst, *a, out),
        Op::Not { dst, a } => un(16, *dst, *a, out),
        Op::Lt { dst, a, b } => bin(17, *dst, *a, *b, out),
        Op::Move { dst, src } => un(18, *dst, *src, out),
        Op::JumpIfFalse { cond, target } => {
            w_u8(19, out);
            w_reg(*cond, out);
            w_usize(*target, out);
        }
        Op::Jump { target } => {
            w_u8(20, out);
            w_usize(*target, out);
        }
        Op::AddValue { dst, a, b } => bin(21, *dst, *a, *b, out),
        Op::StrictEq { dst, a, b } => bin(22, *dst, *a, *b, out),
        Op::NewString { dst, value } => {
            w_u8(23, out);
            w_reg(*dst, out);
            w_str(value, out);
        }
        Op::NewArray { dst, len } => {
            w_u8(24, out);
            w_reg(*dst, out);
            w_usize(*len, out);
        }
        Op::NewArrayCtor { dst, arg } => {
            w_u8(54, out);
            w_reg(*dst, out);
            w_reg(*arg, out);
        }
        Op::GetElem { dst, arr, index } => bin(25, *dst, *arr, *index, out),
        Op::SetElem { arr, index, src } => bin(26, *arr, *index, *src, out),
        Op::GetKey { dst, obj, key } => bin(27, *dst, *obj, *key, out),
        Op::SetKey { obj, key, src } => bin(28, *obj, *key, *src, out),
        Op::ObjectSpread { dst, src } => un(29, *dst, *src, out),
        Op::EnumKeys { dst, obj } => un(30, *dst, *obj, out),
        Op::ArrayLen { dst, arr } => un(31, *dst, *arr, out),
        Op::CollectionSize { dst, recv } => un(32, *dst, *recv, out),
        Op::ArrayPush { arr, src } => un(33, *arr, *src, out),
        Op::ArrayExtend { arr, src } => un(34, *arr, *src, out),
        Op::ArraySliceFrom { dst, src, from } => bin(35, *dst, *src, *from, out),
        Op::ObjectRest { dst, src, exclude } => {
            w_u8(36, out);
            w_reg(*dst, out);
            w_reg(*src, out);
            w_strs(exclude, out);
        }
        Op::NewCollection { dst, is_set, seed } => {
            w_u8(37, out);
            w_reg(*dst, out);
            w_bool(*is_set, out);
            match seed {
                Some(s) => {
                    w_u8(1, out);
                    w_reg(*s, out);
                }
                None => w_u8(0, out),
            }
        }
        Op::NewRegExp { dst, source, flags } => {
            w_u8(38, out);
            w_reg(*dst, out);
            w_str(source, out);
            w_str(flags, out);
        }
        Op::NewObject { dst } => {
            w_u8(39, out);
            w_reg(*dst, out);
        }
        Op::SetProp { obj, key, src } => {
            w_u8(40, out);
            w_reg(*obj, out);
            w_str(key, out);
            w_reg(*src, out);
        }
        Op::GetProp { dst, obj, key } => {
            w_u8(41, out);
            w_reg(*dst, out);
            w_reg(*obj, out);
            w_str(key, out);
        }
        Op::Call { dst, func, args } => {
            w_u8(42, out);
            w_reg(*dst, out);
            w_u32(*func, out);
            w_regs(args, out);
        }
        Op::LoadFunc { dst, func } => {
            w_u8(43, out);
            w_reg(*dst, out);
            w_u32(*func, out);
        }
        Op::MakeClosure {
            dst,
            func,
            captures,
        } => {
            w_u8(44, out);
            w_reg(*dst, out);
            w_u32(*func, out);
            w_regs(captures, out);
        }
        Op::CallValue { dst, callee, args } => {
            w_u8(45, out);
            w_reg(*dst, out);
            w_reg(*callee, out);
            w_regs(args, out);
        }
        Op::CallValueThis {
            dst,
            callee,
            recv,
            args,
        } => {
            w_u8(46, out);
            w_reg(*dst, out);
            w_reg(*callee, out);
            w_reg(*recv, out);
            w_regs(args, out);
        }
        Op::CallMethod {
            dst,
            recv,
            key,
            args,
        } => {
            w_u8(47, out);
            w_reg(*dst, out);
            w_reg(*recv, out);
            w_str(key, out);
            w_regs(args, out);
        }
        Op::CallCtor { ctor, recv, args } => {
            w_u8(48, out);
            w_u32(*ctor, out);
            w_reg(*recv, out);
            w_regs(args, out);
        }
        Op::CallNative { dst, native, args } => {
            w_u8(49, out);
            w_reg(*dst, out);
            w_u16(*native, out);
            w_regs(args, out);
        }
        Op::PushHandler { target, reg } => {
            w_u8(50, out);
            w_usize(*target, out);
            w_reg(*reg, out);
        }
        Op::PopHandler => w_u8(51, out),
        Op::Throw { src } => {
            w_u8(52, out);
            w_reg(*src, out);
        }
        Op::Return { src } => {
            w_u8(53, out);
            w_reg(*src, out);
        }
    }
}

/// Reads one op (the inverse of [`write_op`]).
fn read_op(r: &mut Reader) -> Result<Op, DecodeError> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => Op::LoadConst {
            dst: r.reg()?,
            value: r.value()?,
        },
        1 => Op::Add {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        2 => Op::Sub {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        3 => Op::Mul {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        4 => Op::Div {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        5 => Op::Mod {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        6 => Op::ValueBin {
            dst: r.reg()?,
            op: r.u8()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        7 => Op::HasProp {
            dst: r.reg()?,
            key: r.reg()?,
            obj: r.reg()?,
        },
        8 => Op::IsBuiltin {
            dst: r.reg()?,
            obj: r.reg()?,
            kind: r.u8()?,
        },
        9 => Op::DeleteProp {
            dst: r.reg()?,
            obj: r.reg()?,
            key: r.reg()?,
        },
        10 => Op::SetClassTag {
            obj: r.reg()?,
            class_id: r.u32()?,
        },
        11 => Op::DefineAccessor {
            obj: r.reg()?,
            key: r.string()?,
            getter: r.reg()?,
            setter: r.reg()?,
        },
        12 => Op::InstanceOf {
            dst: r.reg()?,
            obj: r.reg()?,
            ids: Rc::from(r.u32s()?.as_slice()),
        },
        13 => Op::TypeOf {
            dst: r.reg()?,
            a: r.reg()?,
        },
        14 => Op::BitNot {
            dst: r.reg()?,
            a: r.reg()?,
        },
        15 => Op::Neg {
            dst: r.reg()?,
            a: r.reg()?,
        },
        16 => Op::Not {
            dst: r.reg()?,
            a: r.reg()?,
        },
        17 => Op::Lt {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        18 => Op::Move {
            dst: r.reg()?,
            src: r.reg()?,
        },
        19 => Op::JumpIfFalse {
            cond: r.reg()?,
            target: r.usize()?,
        },
        20 => Op::Jump { target: r.usize()? },
        21 => Op::AddValue {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        22 => Op::StrictEq {
            dst: r.reg()?,
            a: r.reg()?,
            b: r.reg()?,
        },
        23 => Op::NewString {
            dst: r.reg()?,
            value: r.string()?,
        },
        24 => Op::NewArray {
            dst: r.reg()?,
            len: r.usize()?,
        },
        25 => Op::GetElem {
            dst: r.reg()?,
            arr: r.reg()?,
            index: r.reg()?,
        },
        26 => Op::SetElem {
            arr: r.reg()?,
            index: r.reg()?,
            src: r.reg()?,
        },
        27 => Op::GetKey {
            dst: r.reg()?,
            obj: r.reg()?,
            key: r.reg()?,
        },
        28 => Op::SetKey {
            obj: r.reg()?,
            key: r.reg()?,
            src: r.reg()?,
        },
        29 => Op::ObjectSpread {
            dst: r.reg()?,
            src: r.reg()?,
        },
        30 => Op::EnumKeys {
            dst: r.reg()?,
            obj: r.reg()?,
        },
        31 => Op::ArrayLen {
            dst: r.reg()?,
            arr: r.reg()?,
        },
        32 => Op::CollectionSize {
            dst: r.reg()?,
            recv: r.reg()?,
        },
        33 => Op::ArrayPush {
            arr: r.reg()?,
            src: r.reg()?,
        },
        34 => Op::ArrayExtend {
            arr: r.reg()?,
            src: r.reg()?,
        },
        35 => Op::ArraySliceFrom {
            dst: r.reg()?,
            src: r.reg()?,
            from: r.reg()?,
        },
        36 => Op::ObjectRest {
            dst: r.reg()?,
            src: r.reg()?,
            exclude: Rc::from(r.strings()?.as_slice()),
        },
        37 => Op::NewCollection {
            dst: r.reg()?,
            is_set: r.boolean()?,
            seed: if r.u8()? == 1 { Some(r.reg()?) } else { None },
        },
        38 => Op::NewRegExp {
            dst: r.reg()?,
            source: r.string()?,
            flags: r.string()?,
        },
        39 => Op::NewObject { dst: r.reg()? },
        40 => Op::SetProp {
            obj: r.reg()?,
            key: r.string()?,
            src: r.reg()?,
        },
        41 => Op::GetProp {
            dst: r.reg()?,
            obj: r.reg()?,
            key: r.string()?,
        },
        42 => Op::Call {
            dst: r.reg()?,
            func: r.u32()?,
            args: r.regs()?,
        },
        43 => Op::LoadFunc {
            dst: r.reg()?,
            func: r.u32()?,
        },
        44 => Op::MakeClosure {
            dst: r.reg()?,
            func: r.u32()?,
            captures: r.regs()?,
        },
        45 => Op::CallValue {
            dst: r.reg()?,
            callee: r.reg()?,
            args: r.regs()?,
        },
        46 => Op::CallValueThis {
            dst: r.reg()?,
            callee: r.reg()?,
            recv: r.reg()?,
            args: r.regs()?,
        },
        47 => Op::CallMethod {
            dst: r.reg()?,
            recv: r.reg()?,
            key: r.string()?,
            args: r.regs()?,
        },
        48 => Op::CallCtor {
            ctor: r.u32()?,
            recv: r.reg()?,
            args: r.regs()?,
        },
        49 => Op::CallNative {
            dst: r.reg()?,
            native: r.u16()?,
            args: r.regs()?,
        },
        50 => Op::PushHandler {
            target: r.usize()?,
            reg: r.reg()?,
        },
        51 => Op::PopHandler,
        52 => Op::Throw { src: r.reg()? },
        53 => Op::Return { src: r.reg()? },
        54 => Op::NewArrayCtor {
            dst: r.reg()?,
            arg: r.reg()?,
        },
        t => return Err(DecodeError::BadTag(t)),
    })
}

/// Zero-copy reload of a `KTBC` artifact from a memory-mapped file (`ROADMAP.md`
/// §2.2, "mmap reload"). Re-exported when available.
#[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
pub use mmap::load_file;

/// `mmap`-based file loading via direct Linux syscalls — no libc, honoring the
/// "no foreign code" charter (the same audited carve-out as the JIT). The
/// mapped, read-only file bytes are decoded in place, then unmapped.
#[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
mod mmap {
    use super::{FnProto, deserialize};
    use alloc::vec::Vec;
    use std::os::unix::io::AsRawFd;

    const PROT_READ: usize = 0x1;
    const MAP_PRIVATE: usize = 0x02;
    const SYS_MMAP: usize = 9;
    const SYS_MUNMAP: usize = 11;

    /// SAFETY: executes the `syscall` instruction with the System V syscall ABI
    /// registers; caller supplies a valid number/arguments.
    #[allow(unsafe_code)]
    unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize) -> isize {
        let ret: isize;
        // SAFETY: a single syscall with documented inputs/clobbers.
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") n as isize => ret,
                in("rdi") a1, in("rsi") a2, in("rdx") a3,
                in("r10") a4, in("r8") a5, in("r9") 0usize,
                out("rcx") _, out("r11") _,
                options(nostack, preserves_flags),
            );
        }
        ret
    }

    /// Loads and decodes a compiled program from `path` by memory-mapping the
    /// file read-only and decoding directly over the mapped bytes — no
    /// intermediate copy of the file contents into a buffer.
    ///
    /// # Errors
    /// Returns an `io::Error` on a failed open/`mmap`, or `InvalidData` if the
    /// mapped bytes are not a valid `KTBC` artifact.
    pub fn load_file(path: &str) -> std::io::Result<Vec<FnProto>> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty artifact",
            ));
        }
        let fd = file.as_raw_fd() as usize;
        // mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)
        // SAFETY: a standard read-only file mapping; the kernel returns the
        // mapping address or a small negative errno.
        #[allow(unsafe_code)]
        let raw = unsafe { syscall6(SYS_MMAP, 0, len, PROT_READ, MAP_PRIVATE, fd) };
        if (-4095..0).contains(&raw) {
            return Err(std::io::Error::last_os_error());
        }
        let ptr = raw as *const u8;
        // SAFETY: `ptr..ptr+len` is a valid, initialized, read-only mapping of
        // the whole file, alive until the `munmap` below. `deserialize` only
        // reads it and copies what it needs into owned `FnProto`s.
        #[allow(unsafe_code)]
        let result = {
            let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
            deserialize(bytes)
        };
        // SAFETY: unmapping exactly the region we mapped.
        #[allow(unsafe_code)]
        unsafe {
            syscall6(SYS_MUNMAP, ptr as usize, len, 0, 0, 0);
        }
        result.map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, alloc::format!("{e:?}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;
    use crate::realm::Realm;

    /// Compiling a program, serializing it, and reloading it via the `mmap` file
    /// path reproduces the same bytecode and runs identically.
    #[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn mmap_reload_runs_identically() {
        let src = "function f(n){ let s=0; for(let i=1;i<=n;i++) s+=i; return s; } f(10) + 1";
        let program = Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let bytes = serialize(&protos);

        // Write to a unique temp file, mmap-load it back, compare.
        let dir = std::env::temp_dir();
        let path = dir.join(alloc::format!(
            "kataan_mmap_{}.ktbc",
            std::process::id() as u64
        ));
        let path_str = path.to_str().unwrap();
        std::fs::write(&path, &bytes).expect("write artifact");

        let reloaded = load_file(path_str).expect("mmap load");
        std::fs::remove_file(&path).ok();

        // The mmap-loaded program is byte-identical to the original.
        assert_eq!(
            serialize(&reloaded),
            bytes,
            "mmap reload not byte-identical"
        );
        // ...and runs to the same result (sum 1..=10 = 55, +1 = 56).
        let mut realm = Realm::new();
        let (value, _out) =
            crate::nbvm::run_program_capturing(&mut realm, &reloaded, 0, &[]).expect("run");
        assert_eq!(realm.to_display_string(value), "56");
    }

    /// `load_file` rejects a non-artifact (bad magic) without UB.
    #[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn mmap_reload_rejects_garbage() {
        let path = std::env::temp_dir().join(alloc::format!(
            "kataan_mmap_bad_{}.ktbc",
            std::process::id() as u64
        ));
        std::fs::write(&path, b"not a ktbc file at all").unwrap();
        let r = load_file(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        assert!(r.is_err(), "garbage must be rejected");
    }

    /// Compiles `src`, round-trips its bytecode through the codec, runs the
    /// reloaded program, and returns its completion display string.
    fn roundtrip_run(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let bytes = serialize(&protos);
        // It's a KTBC container.
        assert_eq!(&bytes[0..4], MAGIC);
        let reloaded = deserialize(&bytes).expect("deserialize");
        // The decode is structurally exact (same byte image when re-serialized).
        assert_eq!(
            serialize(&reloaded),
            bytes,
            "round-trip is not byte-identical"
        );

        let mut realm = Realm::new();
        let (value, _out) =
            crate::nbvm::run_program_capturing(&mut realm, &reloaded, 0, &[]).expect("run");
        realm.to_display_string(value)
    }

    #[test]
    fn roundtrips_and_runs_a_varied_program() {
        // Exercises constants, arithmetic, closures, classes, arrays, control
        // flow, strings, and calls — a broad span of opcodes.
        assert_eq!(roundtrip_run("1 + 2 * 3"), "7");
        assert_eq!(
            roundtrip_run("function makeAdder(x){ return y => x + y; } makeAdder(10)(5)"),
            "15"
        );
        assert_eq!(
            roundtrip_run(
                "class P { constructor(x){ this.x = x; } get(){ return this.x; } } new P(9).get()"
            ),
            "9"
        );
        assert_eq!(
            roundtrip_run("let s = 0; for (let i = 0; i < 5; i++) { s += i; } s"),
            "10"
        );
        assert_eq!(
            roundtrip_run("[1,2,3,4].map(x => x * x).filter(x => x > 4).join(',')"),
            "9,16"
        );
        assert_eq!(roundtrip_run("`hi ${'a' + 'b'}`"), "hi ab");
    }

    fn compile(src: &str) -> Vec<u8> {
        let program = Parser::parse_program(src).expect("parse");
        serialize(&crate::nbvm::compile_program(&program).expect("compile"))
    }

    #[test]
    fn content_store_dedups_and_supports_churn() {
        let mut store = ContentStore::new();

        // Two "tenants" compile the *same* library → one stored entry (dedup).
        let lib = compile("function v(){ return 42; } v()");
        let k1 = store.put(&lib);
        let k2 = store.put(&compile("function v(){ return 42; } v()"));
        assert_eq!(k1, k2, "identical bytecode hashes the same");
        assert_eq!(store.len(), 1, "cross-tenant dedup");

        // A different program is a distinct key/entry.
        let other = compile("function w(){ return 7; } w()");
        let k3 = store.put(&other);
        assert_ne!(k1, k3);
        assert_eq!(store.len(), 2);

        // Load → run the stored artifact.
        let protos = store.load(k1).expect("present").expect("decoded");
        let mut realm = Realm::new();
        let (value, _) =
            crate::nbvm::run_program_capturing(&mut realm, &protos, 0, &[]).expect("run");
        assert_eq!(realm.to_display_string(value), "42");

        // Evict → reload churn: gone after evict, same key on re-put.
        assert!(store.evict(k1));
        assert!(!store.contains(k1) && store.get(k1).is_none());
        assert_eq!(store.len(), 1);
        let k1b = store.put(&lib);
        assert_eq!(k1b, k1, "content addressing is stable across reload");
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn verifier_accepts_valid_and_rejects_out_of_bounds() {
        // A normally-compiled program verifies.
        let program = Parser::parse_program("function f(x){ return x * x; } f(6)").unwrap();
        let mut protos = crate::nbvm::compile_program(&program).unwrap();
        assert_eq!(verify(&protos), Ok(()));

        // Tamper: an out-of-range register reference is rejected.
        let bad_reg = protos[0].n_regs as Reg + 5;
        protos[0].ops.push(Op::Return { src: bad_reg });
        assert_eq!(verify(&protos), Err(VerifyError::Register(bad_reg)));

        // Tamper: a call to a non-existent function index is rejected.
        let mut p2 = crate::nbvm::compile_program(
            &Parser::parse_program("function g(){ return 1; } g()").unwrap(),
        )
        .unwrap();
        let nf = p2.len() as u32;
        p2[0].ops.push(Op::Call {
            dst: 0,
            func: nf + 9,
            args: alloc::vec![],
        });
        assert_eq!(verify(&p2), Err(VerifyError::Function(nf + 9)));

        // Tamper: a jump past the end is rejected.
        let mut p3 =
            crate::nbvm::compile_program(&Parser::parse_program("let x = 1; x").unwrap()).unwrap();
        let beyond = p3[0].ops.len() + 100;
        p3[0].ops.push(Op::Jump { target: beyond });
        assert_eq!(verify(&p3), Err(VerifyError::Target(beyond)));
    }

    #[test]
    fn deserialize_verified_round_trips_and_runs() {
        let program = Parser::parse_program(
            "function fib(n){ return n < 2 ? n : fib(n-1)+fib(n-2); } fib(10)",
        )
        .unwrap();
        let bytes = serialize(&crate::nbvm::compile_program(&program).unwrap());
        let protos = deserialize_verified(&bytes).expect("decode+verify");
        let mut realm = Realm::new();
        let (value, _) =
            crate::nbvm::run_program_capturing(&mut realm, &protos, 0, &[]).expect("run");
        assert_eq!(realm.to_display_string(value), "55");
    }

    #[test]
    fn rejects_corrupt_artifacts() {
        assert_eq!(deserialize(b"XXXX...").unwrap_err(), DecodeError::BadMagic);
        // Right magic, wrong version.
        let mut bad = MAGIC.to_vec();
        bad.extend_from_slice(&999u16.to_le_bytes());
        bad.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(deserialize(&bad).unwrap_err(), DecodeError::BadVersion(999));
        // Truncated mid-record.
        let program = Parser::parse_program("1 + 1").unwrap();
        let protos = crate::nbvm::compile_program(&program).unwrap();
        let bytes = serialize(&protos);
        assert_eq!(
            deserialize(&bytes[..bytes.len() - 1]).unwrap_err(),
            DecodeError::Truncated
        );
    }
}
