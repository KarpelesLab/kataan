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
    /// The keys stored by name.
    keys: Vec<&'a str>,
}

impl<'a> StoreKeys<'a> {
    /// Scans the body (`ops` with the spliced `bytecode` beside them).
    pub(super) fn scan(ops: &[GenericOp], bytecode: &'a [Op]) -> Self {
        let mut s = StoreKeys {
            any: false,
            keys: Vec::new(),
        };
        if ops.len() != bytecode.len() {
            s.any = true;
            return s;
        }
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
                // (A String key's store through the key memo, `key_probe`.)
                GenericOp::SetElem { .. } | GenericOp::ArrSet { .. } => s.any = true,
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
    }
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
