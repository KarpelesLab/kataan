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
//!
//! The text memo (`crate::realm::text_memo`) answers for a key by its text,
//! so equal keys made fresh (`o['k' + i]`) hit too, and it also knows keys
//! **absent** from a receiver's whole chain. [`X64Assembler::tp_probe`]
//! makes its probe inline for a key held in its rope node (`Node::Small`,
//! at most the memo's text limit): the node's buffer is the text's padded
//! words (bytes past the length are zero), hashed as `Text::index` hashes
//! them; the entry must hold that text and the receiver's layout, and, for
//! an absent entry, the receiver's `[[Prototype]]` and the current
//! prototype epoch (every object of the chain is watched, so the epoch moves
//! when any of them changes). Only Rust code records an entry or bumps the
//! epoch.

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};
use crate::realm::{KeyMemoLayout, KeyTextLayout};

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
    /// The text memo's entry layout and index computation.
    text: KeyTextLayout,
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
            text: crate::realm::jit_key_text_layout(),
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
    /// The text memo's answer for the String in register `key_disp` on the
    /// object whose heap slot is `rcx` (a shaped ordinary object without
    /// the exotic flag or accessors, and no String has an auxiliary object:
    /// the caller checked), with the heap arena as the receiver walk left
    /// it: falls through with `eax` = the key's slot in the receiver's
    /// layout (an own data property); goes to `absent` (when given, else
    /// `miss`) when the key is on no object of the receiver's chain; else
    /// to `miss`. Clobbers `rax`, `r8`, `r9`, `r11`.
    pub(super) fn tp_probe(
        &mut self,
        l: &JitLayout,
        kp: &KeyProbe,
        realm_off: i32,
        key_disp: i32,
        absent: Option<Label>,
        miss: Label,
    ) {
        let t = &kp.text;
        let rl = l.rope;
        let w = |k: i32| rl.small_buf + 8 * k;
        // The key's heap slot (a register's handle is live), a String's,
        // held in its node, short enough.
        self.g_load(r::RAX, r::RBX, key_disp);
        self.p_cmp_handle_floor(r::RAX);
        self.g_jcc(cc::B, miss);
        self.p_slot_rooted(l, r::R9, miss);
        self.g_cmp_byte_imm(r::R9, l.off_cell_tag, l.cell_str_disc);
        self.g_jcc(cc::NE, miss);
        self.g_load(r::R11, r::R9, l.off_str_rope);
        self.g_cmp_byte_imm(r::R11, rl.tag, rl.small_tag);
        self.g_jcc(cc::NE, miss);
        // movzx eax, byte [r11 + small_len]; cmp eax, text_max; ja miss
        self.code.extend_from_slice(&[0x41, 0x0f, 0xb6, 0x83]);
        self.code.extend_from_slice(&rl.small_len.to_le_bytes());
        self.code.extend_from_slice(&[0x83, 0xf8, t.text_max]);
        self.g_jcc(cc::A, miss);
        // r8 = the entry's index (`Text::index`): (w0 ^ rol(w1) ^ rol(w2)
        // ^ len) * mul >> 32 & mask.
        self.g_load(r::R8, r::R11, w(1));
        self.code.extend_from_slice(&[0x49, 0xc1, 0xc0, t.rot.0]); // rol r8, rot1
        self.g_load(r::R9, r::R11, w(2));
        self.code.extend_from_slice(&[0x49, 0xc1, 0xc1, t.rot.1]); // rol r9, rot2
        self.code.extend_from_slice(&[0x4d, 0x31, 0xc8]); // xor r8, r9
        self.g_mem(true, &[0x33], r::R8, r::R11, w(0)); // xor r8, [w0]
        self.code.extend_from_slice(&[0x49, 0x31, 0xc0]); // xor r8, rax
        self.g_movabs(r::R9, t.mul);
        self.code.extend_from_slice(&[0x4d, 0x0f, 0xaf, 0xc1]); // imul r8, r9
        self.g_shr_imm(r::R8, 32);
        self.code.extend_from_slice(&[0x41, 0x81, 0xe0]); // and r8d, mask
        self.code.extend_from_slice(&t.mask.to_le_bytes());
        self.code.extend_from_slice(&[0x4d, 0x69, 0xc0]); // imul r8, r8, size
        self.code.extend_from_slice(&t.size.to_le_bytes());
        self.g_load(r::R9, r::R15, realm_off);
        self.g_load(r::R9, r::R9, t.entries);
        self.g_test(r::R9);
        self.g_jcc(cc::E, miss);
        self.g_add(r::R8, r::R9);
        // The entry holds this text ...
        for k in 0..3 {
            self.g_load(r::R9, r::R11, w(k));
            self.g_cmp_mem(r::R9, r::R8, t.words + 8 * k);
            self.g_jcc(cc::NE, miss);
        }
        self.g_mem(false, &[0x3b], r::RAX, r::R8, t.len); // cmp eax, [len]
        self.g_jcc(cc::NE, miss);
        // ... and the receiver's layout.
        self.g_load(r::R9, r::RCX, l.off_shape);
        self.g_lea(r::R9, r::R9, kp.shape_delta);
        self.g_cmp_mem(r::R9, r::R8, t.shape);
        self.g_jcc(cc::NE, miss);
        self.g_mem(false, &[0x8b], r::RAX, r::R8, t.slot); // mov eax, [slot]
        let present = self.new_label();
        self.code.extend_from_slice(&[0x83, 0xf8, 0xff]); // cmp eax, -1
        self.g_jcc(cc::NE, present);
        // An absent entry: the receiver's `[[Prototype]]` and the epoch.
        let absent = absent.unwrap_or(miss);
        self.g_load(r::R9, r::RCX, l.props.off_proto);
        self.g_cmp_mem(r::R9, r::R8, t.proto);
        self.g_jcc(cc::NE, miss);
        self.g_movabs(r::R9, crate::object::proto_epoch_addr() as u64);
        self.g_load(r::R9, r::R9, 0);
        self.g_cmp_mem(r::R9, r::R8, t.stamp);
        self.g_jcc(cc::NE, miss);
        self.jmp(absent);
        self.bind(present);
    }
}
