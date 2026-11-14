//! The generic tier's **String-key read memo**: several computed reads with
//! the same key register (`this.lo[a]`, `ray.dir[a]`, `ray.origin[a]` in
//! one loop step) share the slot the first one's probe found.
//!
//! An element read with a String key finds its slot through the realm's
//! String-key memos (`key_probe`): the key's heap slot, its rope, a hash and
//! a table entry to compare, for every read. Reads of one key on objects of
//! one layout (a vector's `x`, `y` or `z` picked by an axis variable) then
//! repeat that probe for the same answer. So per key register read by
//! several element reads, the frame keeps a memo: the key's bits, the
//! receiver's layout word and the slot a probe (or the text memo) found for
//! them. A read whose key register holds those bits on a receiver of that
//! layout (passing the receiver tests every read makes: a shaped ordinary
//! object with no exotic flag and no accessors) takes the slot with no
//! probe.
//!
//! Soundness is the receiver cache's (`recv_cache`): the memo's key slot is
//! one of the keys the frame prologue and every `call` empty, and only Rust
//! code (always behind a call) can allocate, collect or free a heap slot or
//! a layout. So equal bits mean the very String the probe saw, and the
//! receiver holding the layout word means the very layout (alive: the
//! receiver holds it), whose keys and slots never change. A layout's slot
//! for a key text is all the probe answers; the realm-wide conditions it
//! also tests (no String with an auxiliary object) only Rust code changes.

use alloc::vec;
use alloc::vec::Vec;

use super::GenericOp;
use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};

/// Whether element reads share their key's probe (see the module docs);
/// `KATAAN_JIT_KEYREUSE=0` turns it off (read once per process).
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_KEYREUSE").is_none_or(|v| v != "0"))
}

/// A memo's frame slots.
#[derive(Clone, Copy, Debug)]
pub(super) struct Memo {
    /// The key's bits (a receiver-cache key: emptied on entry and by every
    /// call).
    pub key: i32,
    /// The receiver's layout word (as the object holds it).
    pub shape: i32,
    /// The key's slot in that layout.
    pub slot: i32,
}

/// Which element reads share a memo.
pub(super) struct Plan {
    /// The number of memos.
    pub count: usize,
    /// Per op: its memo, for an element read with a shared key register.
    pub at: Vec<Option<usize>>,
}

/// The plan for `ops`: a memo per key register that at least two element
/// reads use (one read alone would only fill it), not counting reads whose
/// key `is_num(pc, reg)` proves a Number, nor any read of a register some
/// op loads with a Number constant (a temporary holding `r[0]`, then
/// `r[1]`): their String-key path is never taken, and each memo costs a
/// store at entry and after every call.
pub(super) fn plan(ops: &[GenericOp], is_num: impl Fn(usize, u16) -> bool) -> Plan {
    let mut at = vec![None; ops.len()];
    if !enabled() {
        return Plan { count: 0, at };
    }
    let num_const: Vec<u16> = ops
        .iter()
        .filter_map(|op| match *op {
            GenericOp::Const { dst, imm } if crate::nanbox::NanBox::from_bits(imm).is_number() => {
                Some(dst)
            }
            _ => None,
        })
        .collect();
    let key_at = |pc: usize| match ops[pc] {
        GenericOp::GetElem { index, .. } if !is_num(pc, index) && !num_const.contains(&index) => {
            Some(index)
        }
        _ => None,
    };
    let mut regs: Vec<(u16, usize)> = Vec::new();
    for pc in 0..ops.len() {
        if let Some(k) = key_at(pc) {
            match regs.iter_mut().find(|(r, _)| *r == k) {
                Some(e) => e.1 += 1,
                None => regs.push((k, 1)),
            }
        }
    }
    regs.retain(|&(_, n)| n >= 2);
    for (pc, a) in at.iter_mut().enumerate() {
        if let Some(k) = key_at(pc) {
            *a = regs.iter().position(|&(r, _)| r == k);
        }
    }
    Plan {
        count: regs.len(),
        at,
    }
}

impl X64Assembler {
    /// To `miss` unless the memo holds the key bits in `rax` (so a String's
    /// handle: only those are recorded).
    pub(super) fn km_key(&mut self, m: Memo, miss: Label) {
        self.g_cmp_mem(r::RAX, r::RBP, m.key);
        self.g_jcc(cc::NE, miss);
    }

    /// After [`km_key`](Self::km_key) and a walk to the receiver's heap slot
    /// (`rcx`) that found a shaped ordinary object without the exotic flag:
    /// when it has no accessors and the memo's layout, `rax` = the memo's
    /// slot and on to `hit`; else to `miss`. Clobbers `r9`.
    pub(super) fn km_hit(&mut self, l: &JitLayout, m: Memo, miss: Label, hit: Label) {
        let no_acc = self.new_label();
        self.g_load(r::R9, r::RCX, l.props.off_extra);
        self.g_test(r::R9);
        self.g_jcc(cc::E, no_acc);
        self.g_load(r::R9, r::R9, l.props.extra_acc_len);
        self.g_test(r::R9);
        self.g_jcc(cc::NE, miss);
        self.bind(no_acc);
        self.g_load(r::R9, r::RCX, l.off_shape);
        self.g_cmp_mem(r::R9, r::RBP, m.shape);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::RBP, m.slot);
        self.jmp(hit);
    }

    /// Records in the memo that the key in register `key_disp` is slot
    /// `rax` of the layout of the receiver whose heap slot is `rcx`.
    /// Preserves `rax` and `rcx`; clobbers `r9`.
    pub(super) fn km_fill(&mut self, l: &JitLayout, m: Memo, key_disp: i32) {
        self.g_store(r::RBP, m.slot, r::RAX);
        self.g_load(r::R9, r::RCX, l.off_shape);
        self.g_store(r::RBP, m.shape, r::R9);
        self.g_load(r::R9, r::RBX, key_disp);
        self.g_store(r::RBP, m.key, r::R9);
    }
}
