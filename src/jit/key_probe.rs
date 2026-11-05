//! `o[k]` with a String key, from the String-key slot memo with no call
//! (`crate::realm::key_memo`): "the text of rope R is slot S of layout L".
//!
//! `jit_helper_get_elem` probes the memo first, so a dictionary read with a
//! memoized key (`for (…) s += d[ks[i]]`) was a helper call around a
//! one-entry lookup. The compiled read makes the same probe inline: the key
//! String's rope address picks the entry (the memo's own index), which must
//! name that rope and the receiver's layout; the receiver must be a shaped
//! ordinary object without the exotic flag or accessors, and no String may
//! have an auxiliary object (`Realm::str_special`), as `key_memo_probe`
//! requires. An entry holds its layout and a weak reference to its rope, so
//! neither address can be reused while it stands: a hit is exact. Anything
//! else is the ordinary path's (whose helper probes again and may record).

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};
use crate::realm::KeyMemoLayout;

/// What the probe needs besides the heap layout.
#[derive(Clone, Copy, Debug)]
pub(super) struct KeyProbe {
    /// The memo's entry layout (see `realm::key_memo`).
    memo: KeyMemoLayout,
    /// `CAP - 1`.
    mask: i32,
    /// A shape's address as the memo records it (`Rc::as_ptr`) less its
    /// word as an object holds it (`super::shape_word`).
    shape_delta: i32,
}

impl KeyProbe {
    /// Derives the probe's layout.
    pub(super) fn new() -> Self {
        let (memo, mask) = crate::realm::jit_key_memo_layout();
        let s = crate::shape::Shape::root().add("k");
        let shape_delta = (alloc::rc::Rc::as_ptr(&s) as u64).wrapping_sub(super::shape_word(&s));
        Self {
            memo,
            mask,
            shape_delta: i32::try_from(shape_delta as i64).expect("an Rc's value offset"),
        }
    }
}

impl X64Assembler {
    /// `rax` = the memoized slot of the String in register `key_disp` in the
    /// layout of the object whose heap slot is `rcx` (a shaped ordinary
    /// object without the exotic flag or accessors: the caller checked),
    /// with the heap arena as the receiver walk left it; else to `miss`.
    /// Clobbers `r9`, `r11`.
    pub(super) fn kp_slot(
        &mut self,
        l: &JitLayout,
        kp: &KeyProbe,
        realm_off: i32,
        key_disp: i32,
        miss: Label,
    ) {
        // The key's heap slot (a register's handle is live), a String's.
        self.g_load(r::RAX, r::RBX, key_disp);
        self.p_cmp_handle_floor(r::RAX);
        self.g_jcc(cc::B, miss);
        self.p_slot_rooted(l, r::R9, miss);
        self.g_cmp_byte_imm(r::R9, l.off_cell_tag, l.cell_str_disc);
        self.g_jcc(cc::NE, miss);
        // r11 = its rope's address (`Rope::addr`).
        self.g_load(r::R11, r::R9, l.off_str_rope);
        self.g_lea(r::R11, r::R11, l.rope.tag);
        // r9 = its entry: `entries + ((p >> 4 ^ p >> 14) & mask) * size`.
        self.g_mov(r::R9, r::R11);
        self.code.extend_from_slice(&[0x49, 0xc1, 0xe9, 0x04]); // shr r9, 4
        self.g_mov(r::RAX, r::R11);
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe8, 0x0e]); // shr rax, 14
        self.code.extend_from_slice(&[0x49, 0x31, 0xc1]); // xor r9, rax
        self.code.extend_from_slice(&[0x49, 0x81, 0xe1]); // and r9, mask
        self.code.extend_from_slice(&kp.mask.to_le_bytes());
        self.code.extend_from_slice(&[0x4d, 0x69, 0xc9]); // imul r9, r9, size
        self.code.extend_from_slice(&kp.memo.size.to_le_bytes());
        self.g_load(r::RAX, r::R15, realm_off);
        self.g_load(r::RAX, r::RAX, kp.memo.entries);
        self.g_test(r::RAX);
        self.g_jcc(cc::E, miss);
        self.g_add(r::R9, r::RAX);
        // The entry is the rope's and the receiver's layout's.
        self.g_cmp_mem(r::R11, r::R9, kp.memo.rope);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::RCX, l.off_shape);
        self.g_lea(r::RAX, r::RAX, kp.shape_delta);
        self.g_cmp_mem(r::RAX, r::R9, kp.memo.shape);
        self.g_jcc(cc::NE, miss);
        // mov eax, [r9 + slot]
        self.code.extend_from_slice(&[0x41, 0x8b, 0x81]);
        self.code.extend_from_slice(&kp.memo.slot.to_le_bytes());
    }
}
