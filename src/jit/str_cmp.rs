//! `a < b` (and `>`, `<=`, `>=`) of two short Strings, inline.
//!
//! A relational operator on two Strings went through
//! `jit_helper_str_compare`, whose Rust fast path (`Realm::small_string_lt`)
//! compares two texts held in their rope nodes (`Node::Small`) a word at a
//! time. The compiled code makes that same comparison itself: both
//! operands rooted handles of String cells held in their nodes, no String
//! with an auxiliary object (`Realm::str_special`, as the helper requires),
//! and the node buffers' first four words (bytes past a text's length are
//! zero, see `crate::rope`) compared in order. At the first differing word,
//! the lowest differing byte decides: before it both texts hold the same
//! bytes, and WTF-8 byte order is UTF-16 code-unit order there unless a
//! four-byte lead (`0xF0` or above) is involved — that case, as the helper's,
//! is left to the Rust path. Past the shorter text its padding is zero and
//! the longer's byte is not, so a prefix sorts first, as it should. With all
//! four words equal, the lengths decide when both texts fit in them (at most
//! 32 bytes); otherwise the Rust path. Anything else — not two such Strings —
//! goes to `miss`, with nothing done.
//!
//! `KATAAN_JIT_STRCMP_INLINE=0` turns it off.

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};

/// Whether the inline comparison is emitted (see the module docs). Read
/// once per process.
pub(crate) fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_STRCMP_INLINE").is_none_or(|v| v != "0"))
}

impl X64Assembler {
    /// The comparison (`op`: 0 `<`, 1 `>`, 2 `<=`, 3 `>=`, as
    /// `Op::Compare`'s) of the frame slots `a` and `b` when both hold Strings
    /// held in their rope nodes (see the module docs): `rax` = the boolean's
    /// bits (`f` false's, true's being `f + 1`), falling through; else to
    /// `miss` with nothing stored. The realm is at `[r15 + realm_off]`, its
    /// `str_special` flag at `str_special`. Clobbers `rax`, `rcx`, `rdx`,
    /// `rsi`, `r8`–`r11` (what a helper call may).
    pub(crate) fn p_small_str_cmp(
        &mut self,
        l: &JitLayout,
        (realm_off, str_special): (i32, i32),
        (a, b): (i32, i32),
        op: u8,
        f: u64,
        miss: Label,
    ) {
        let rl = l.rope;
        let w = |k: i32| rl.small_buf + 8 * k;
        // No String has an auxiliary object.
        self.g_load(r::R9, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R9, str_special, 0);
        self.g_jcc(cc::NE, miss);
        self.p_arena(realm_off, &l.props);
        // r8 / r11 = the operands' rope nodes, each a String's held in it.
        for (slot, node) in [(a, r::R8), (b, r::R11)] {
            self.load_rax(slot);
            self.p_cmp_handle_floor(r::RAX);
            self.g_jcc(cc::B, miss);
            self.p_slot_rooted(l, r::R9, miss);
            self.g_cmp_byte_imm(r::R9, l.off_cell_tag, l.cell_str_disc);
            self.g_jcc(cc::NE, miss);
            self.g_load(node, r::R9, l.off_str_rope);
            self.g_cmp_byte_imm(node, rl.tag, rl.small_tag);
            self.g_jcc(cc::NE, miss);
        }
        // The first four words, in order: at the first that differ (`rax`
        // a's, `rsi` b's), the lowest differing byte.
        let diff = self.new_label();
        for k in 0..4 {
            self.g_load(r::RAX, r::R8, w(k));
            self.g_load(r::RSI, r::R11, w(k));
            self.g_cmp(r::RAX, r::RSI);
            self.g_jcc(cc::NE, diff);
        }
        // All four equal: the lengths decide, when both texts fit in them.
        // movzx eax, byte [r8 + len]; movzx ecx, byte [r11 + len]
        self.g_mem(false, &[0x0f, 0xb6], r::RAX, r::R8, rl.small_len);
        self.g_mem(false, &[0x0f, 0xb6], r::RCX, r::R11, rl.small_len);
        self.code.extend_from_slice(&[0x83, 0xf8, 0x20]); // cmp eax, 32
        self.g_jcc(cc::A, miss);
        self.code.extend_from_slice(&[0x83, 0xf9, 0x20]); // cmp ecx, 32
        self.g_jcc(cc::A, miss);
        self.code.extend_from_slice(&[0x39, 0xc8]); // cmp eax, ecx
        // a < b: la < lb; a > b: la > lb; a <= b: la <= lb; a >= b: la >= lb.
        let set = match op {
            0 => 0x92, // setb
            1 => 0x97, // seta
            2 => 0x96, // setbe
            _ => 0x93, // setae
        };
        let result = self.new_label();
        self.code.extend_from_slice(&[0x0f, set, 0xc0]); // setcc al
        self.jmp(result);
        self.bind(diff);
        // rcx = the differing byte's bit offset; eax / esi = the bytes.
        self.g_mov(r::RCX, r::RAX);
        self.code.extend_from_slice(&[0x48, 0x31, 0xf1]); // xor rcx, rsi
        self.code.extend_from_slice(&[0x48, 0x0f, 0xbc, 0xc9]); // bsf rcx, rcx
        self.code.extend_from_slice(&[0x83, 0xe1, 0x38]); // and ecx, 0x38
        self.code.extend_from_slice(&[0x48, 0xd3, 0xe8]); // shr rax, cl
        self.code.extend_from_slice(&[0x48, 0xd3, 0xee]); // shr rsi, cl
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
        self.code.extend_from_slice(&[0x40, 0x0f, 0xb6, 0xf6]); // movzx esi, sil
        self.code.extend_from_slice(&[0x3d, 0xf0, 0x00, 0x00, 0x00]); // cmp eax, 0xf0
        self.g_jcc(cc::AE, miss);
        self.code
            .extend_from_slice(&[0x81, 0xfe, 0xf0, 0x00, 0x00, 0x00]); // cmp esi, 0xf0
        self.g_jcc(cc::AE, miss);
        self.code.extend_from_slice(&[0x39, 0xf0]); // cmp eax, esi
        // The bytes differ: `<=` is `<`, `>=` is `>`.
        let set = if op == 0 || op == 2 { 0x92 } else { 0x97 };
        self.code.extend_from_slice(&[0x0f, set, 0xc0]); // setb / seta al
        self.bind(result);
        self.code.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
        self.g_movabs(r::R11, f);
        self.g_add(r::RAX, r::R11);
    }
}
