//! **Guard elimination** across the ops of one generic body: what a method
//! lookup proved stays proved until something that could change it runs.
//!
//! The proofs here are frame slots that are also receiver-cache keys
//! (`X64Assembler::recv_cache_keys`): the frame prologue and every `call`
//! the body makes empty them, and only Rust code (always behind a call)
//! bumps the prototype epoch, adds an accessor, changes a `[[Prototype]]`,
//! moves, frees or reuses a heap cell, or writes a memo. What the body's
//! own code may change without a call is a property's *value* (an inline
//! store) and an object's layout (an inline add of a new key, never to a
//! watched object: `object::JIT_FLAG_WATCHED`). So a proof of a lookup of
//! key `k` holds until the next call when no op of the body can store a
//! property named `k` in line ([`StoreKeys::may_store`]): the lookup's
//! answer then cannot change (an add of another key leaves it, a store to
//! `k` anywhere — the receiver, a prototype, the holder — is what is ruled
//! out).
//!
//! Two kinds:
//!
//! - a **method read proof** (`Op::GetMethod`, `call_seq::emit_get_method`):
//!   the receiver's bits and the method value its lookup found. The same
//!   receiver value (the same object: no Rust code ran, so no cell was
//!   reused) then has the same method, with no walk, layout, epoch or
//!   accessor check;
//! - a **memo way proof** (an inlined method or accessor call's guard,
//!   `call_seq::emit_guard_fast`): per call site memo and way, that the way's
//!   `GuardMemo` was checked at the current prototype epoch with its
//!   holder's value its callee. While it stands, a receiver of the way's
//!   recorded layout needs neither check again (the memo is written by Rust
//!   code, or by the guard's own fill, which empties the proof).

use alloc::vec::Vec;

use super::GenericOp;
use crate::nbvm::Op;

/// The property keys a body may store without a call.
pub(super) struct StoreKeys<'a> {
    /// Some op may store a property whose key is not known statically (a
    /// computed-key write, an op this analysis does not know).
    any: bool,
    /// Some computed-key write's key is always a Number or a BigInt
    /// ([`numeric_regs`]): its property key is a numeric string (or, from a
    /// register still holding its initial `undefined`, `"undefined"`).
    numeric: bool,
    /// The keys stored by name.
    keys: Vec<&'a str>,
}

impl<'a> StoreKeys<'a> {
    /// Scans the body (`ops` with the spliced `bytecode` beside them).
    pub(super) fn scan(ops: &[GenericOp], bytecode: &'a [Op]) -> Self {
        let mut s = StoreKeys {
            any: false,
            numeric: false,
            keys: Vec::new(),
        };
        if ops.len() != bytecode.len() {
            s.any = true;
            return s;
        }
        let num = if numeric_keys_enabled() {
            numeric_regs(bytecode)
        } else {
            Vec::new()
        };
        for (op, bc) in ops.iter().zip(bytecode) {
            match op {
                GenericOp::Const { .. }
                | GenericOp::Add { .. }
                | GenericOp::Move { .. }
                | GenericOp::GetProp { .. }
                | GenericOp::GetMethod { .. }
                | GenericOp::GetElem { .. }
                | GenericOp::ArrayLen { .. }
                | GenericOp::Arith { .. }
                | GenericOp::Lt { .. }
                | GenericOp::Compare { .. }
                | GenericOp::NumAdd { .. }
                | GenericOp::StrictEq { .. }
                | GenericOp::ValueBin { .. }
                | GenericOp::Not { .. }
                | GenericOp::Jump { .. }
                | GenericOp::JumpIfFalse { .. }
                | GenericOp::CheckTdz { .. }
                | GenericOp::ToNumeric { .. }
                | GenericOp::Step { .. }
                | GenericOp::ArrGet { .. }
                | GenericOp::CellGet { .. }
                | GenericOp::CellSet { .. }
                | GenericOp::Nop
                | GenericOp::PushHandler { .. }
                | GenericOp::PopHandler
                | GenericOp::InlineGuard { .. }
                | GenericOp::InlineLeave
                | GenericOp::PlainPrims { .. }
                | GenericOp::Ret { .. }
                // (Calls: a helper call, a native call sequence, or a
                // built-in computed in line that writes no named property:
                // `Math`, an array push, a read-only direct call.)
                | GenericOp::CallNative { .. }
                | GenericOp::CallMethod { .. }
                | GenericOp::CallValue { .. }
                | GenericOp::CallWithMethod { .. }
                | GenericOp::CallKey { .. }
                | GenericOp::Construct { .. } => {}
                GenericOp::SetProp { .. } | GenericOp::InitProp { .. } => match bc {
                    Op::SetProp { key, .. }
                    | Op::SetPropStrict { key, .. }
                    | Op::InitProp { key, .. } => s.keys.push(key.as_str()),
                    _ => s.any = true,
                },
                // (A String key's store through the key memo, `key_probe`;
                // a key always numeric stores only numeric keys.)
                GenericOp::SetElem { .. } | GenericOp::ArrSet { .. } => match bc {
                    Op::SetKey { key, .. }
                    | Op::SetKeyStrict { key, .. }
                    | Op::SetElem { index: key, .. }
                        if num.get(usize::from(*key)) == Some(&true) =>
                    {
                        s.numeric = true;
                    }
                    _ => s.any = true,
                },
                GenericOp::Interp { .. } => match bc {
                    Op::StoreGlobal { name, .. } => s.keys.push(name.as_str()),
                    Op::NewObjectLit { lit, .. } => {
                        s.keys.extend(lit.keys.iter().map(|k| k.as_str()));
                    }
                    Op::LoadConst { .. }
                    | Op::LoadGlobal { .. }
                    | Op::RequireObjectCoercible { .. }
                    | Op::TypeOf { .. }
                    | Op::StrEqConst { .. }
                    | Op::NewString { .. }
                    | Op::NewStringWtf8 { .. }
                    | Op::NewObject { .. }
                    | Op::NewArray { .. }
                    | Op::NewArrayFrom { .. }
                    | Op::NewBox { .. }
                    | Op::MakeClosure { .. }
                    | Op::BindThis { .. }
                    | Op::Relational { .. }
                    | Op::IterNext { .. }
                    | Op::IterOpen { .. }
                    | Op::FreshArrayIterPlain { .. }
                    | Op::ToStr { .. }
                    | Op::ToStrLazy { .. }
                    | Op::SwitchStr { .. }
                    | Op::Neg { .. }
                    | Op::BitNot { .. }
                    | Op::ToKey { .. }
                    | Op::LoadNewTarget { .. }
                    | Op::Concat { .. }
                    | Op::Throw { .. }
                    | Op::ThrowTypeError { .. }
                    | Op::ThrowReferenceError { .. }
                    | Op::TypeofGlobal { .. }
                    | Op::GlobalExists { .. }
                    | Op::InstanceOf { .. }
                    | Op::HasProp { .. } => {}
                    _ => s.any = true,
                },
            }
        }
        s
    }

    /// Whether the body may store a property named `key` without a call.
    /// (`length` and array-index keys are written by element stores and
    /// in-line array built-ins: always.)
    pub(super) fn may_store(&self, key: &str) -> bool {
        self.any
            || key == "length"
            || key.is_empty()
            || key.bytes().all(|b| b.is_ascii_digit())
            || self.keys.contains(&key)
            || (self.numeric && numeric_like(key))
    }
}

/// Whether `KATAAN_JIT_NUMKEYS=0` turns the numeric-key refinement of
/// [`StoreKeys`] off (read once per process).
fn numeric_keys_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_NUMKEYS").is_none_or(|v| v != "0"))
}

/// Whether `key` may be the property key of a Number or BigInt value
/// (`ToString`: digits, a sign, a point, an exponent, `NaN`, `Infinity`), or
/// `"undefined"` (a register's initial value).
fn numeric_like(key: &str) -> bool {
    matches!(key, "NaN" | "Infinity" | "-Infinity" | "undefined")
        || key
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'+' | b'-'))
}

/// Per register of bytecode `ops`: whether it only ever holds a Number or a
/// BigInt once written — every op that names it is one this analysis knows,
/// and each of those that writes it writes a numeric result (an arithmetic
/// op's, a numeric constant, a `+` of two such registers, a copy of one).
/// Any other op naming a register (it may write it) rules it out, and so does
/// having no write in the body at all (a parameter, a capture). A register
/// read before its first write holds `undefined` there (see `numeric_like`).
pub(super) fn numeric_regs(ops: &[Op]) -> Vec<bool> {
    use crate::nbvm::{VB_BIT_AND, VB_BIT_OR, VB_BIT_XOR, VB_POW, VB_SHL, VB_SHR, VB_USHR};
    #[derive(Clone, Copy)]
    enum Def {
        Num,
        Add(u16, u16),
        Copy(u16),
    }
    let mut n = 0usize;
    for op in ops {
        let top = core::cell::Cell::new(0usize);
        crate::bytecode::op_for_each_reg(op, &|r| top.set(top.get().max(usize::from(r) + 1)));
        n = n.max(top.get());
    }
    let mut out = alloc::vec![false; n];
    let mut bad = alloc::vec![false; n];
    let mut defs: Vec<(u16, Def)> = Vec::new();
    for op in ops {
        let def = match *op {
            Op::LoadConst { dst, value } => Some((dst, value.as_number().map(|_| Def::Num))),
            Op::Add { dst, .. }
            | Op::Sub { dst, .. }
            | Op::Mul { dst, .. }
            | Op::Div { dst, .. }
            | Op::Mod { dst, .. }
            | Op::ToNumeric { dst, .. }
            | Op::Step { dst, .. }
            | Op::Neg { dst, .. }
            | Op::BitNot { dst, .. }
            | Op::ArrayLen { dst, .. } => Some((dst, Some(Def::Num))),
            Op::ValueBin { dst, op, .. } => Some((
                dst,
                matches!(
                    op,
                    VB_BIT_AND | VB_BIT_OR | VB_BIT_XOR | VB_SHL | VB_SHR | VB_USHR | VB_POW
                )
                .then_some(Def::Num),
            )),
            Op::AddValue { dst, a, b } => Some((dst, Some(Def::Add(a, b)))),
            Op::Move { dst, src } => Some((dst, Some(Def::Copy(src)))),
            Op::Compare { dst, .. }
            | Op::Lt { dst, .. }
            | Op::StrictEq { dst, .. }
            | Op::Not { dst, .. }
            | Op::GetKey { dst, .. }
            | Op::GetProp { dst, .. } => Some((dst, None)),
            // (Ops that only read the registers they name.)
            Op::JumpIfFalse { .. }
            | Op::Jump { .. }
            | Op::SetKey { .. }
            | Op::SetKeyStrict { .. }
            | Op::SetElem { .. }
            | Op::SetProp { .. }
            | Op::SetPropStrict { .. }
            | Op::Return { .. }
            | Op::RequireObjectCoercible { .. } => continue,
            _ => {
                let named = core::cell::RefCell::new(Vec::new());
                crate::bytecode::op_for_each_reg(op, &|r| named.borrow_mut().push(r));
                for r in named.into_inner() {
                    if let Some(b) = bad.get_mut(usize::from(r)) {
                        *b = true;
                    }
                }
                continue;
            }
        };
        if let Some((dst, d)) = def {
            match d {
                Some(d) => {
                    defs.push((dst, d));
                    if let Some(o) = out.get_mut(usize::from(dst)) {
                        *o = true;
                    }
                }
                None => {
                    if let Some(b) = bad.get_mut(usize::from(dst)) {
                        *b = true;
                    }
                }
            }
        }
    }
    for (o, b) in out.iter_mut().zip(&bad) {
        *o &= !b;
    }
    // Down to the registers all of whose writes keep them numeric.
    loop {
        let mut changed = false;
        for &(dst, d) in &defs {
            let ok = |r: u16| out.get(usize::from(r)).copied().unwrap_or(false);
            let keeps = match d {
                Def::Num => true,
                Def::Add(a, b) => ok(a) && ok(b),
                Def::Copy(src) => ok(src),
            };
            if !keeps && ok(dst) {
                out[usize::from(dst)] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    out
}

/// The key of the method or accessor an inlined call's guard at `pc`
/// identifies (its call op's), if it is one.
pub(super) fn guard_key<'a>(ops: &[GenericOp], bytecode: &'a [Op], pc: usize) -> Option<&'a str> {
    let GenericOp::InlineGuard { call, .. } = *ops.get(pc)? else {
        return None;
    };
    let call = call as usize;
    match (ops.get(call)?, bytecode.get(call)?) {
        (GenericOp::CallMethod { .. }, Op::CallMethod { key, .. })
        | (GenericOp::GetProp { .. }, Op::GetProp { key, .. })
        | (GenericOp::SetProp { .. }, Op::SetProp { key, .. } | Op::SetPropStrict { key, .. }) => {
            Some(key.as_str())
        }
        _ => None,
    }
}
