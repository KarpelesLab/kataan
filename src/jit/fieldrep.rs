//! Per-field **representation** checks of compiled code: what keeps a
//! layout's record of its Number-only slots true (`Shape::may_be_non_number`).
//!
//! A [`Shape`](crate::shape::Shape) records, per slot, whether some object
//! with that layout may hold anything but a Number there. Every write of a
//! value into an object's slot must keep that record: Rust code does it in
//! `Object`'s slot writes, and compiled code's inline stores and adds do it
//! here. An inline write of a value that is not a Number into a slot the
//! layout still has as Number-only takes the write's slow path instead
//! (whose Rust store marks the slot, and every layout descending from that
//! one); a Number, or any value into a slot already marked, is written
//! inline. So no inline code ever changes a record: only Rust code does,
//! and a speculative loop's fast version, which runs no Rust code from its
//! entry checks to its back-edge, may rely on what it checked at entry
//! (`numeric::NumGen`'s field facts).

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};
use alloc::vec::Vec;

impl X64Assembler {
    /// To `number` when register `val` holds a Number (its bits are not all
    /// of the `QNAN` pattern, bits 50..=62). Clobbers `scratch`.
    fn fr_to_if_number(&mut self, val: u8, scratch: u8, number: Label) {
        self.g_mov(scratch, val);
        let w = 0x48 | ((scratch >> 3) & 1);
        let m = scratch & 7;
        // not scratch; shl scratch, 1; shr scratch, 51: the 13 pattern bits
        // inverted, zero exactly when all were set (a boxed value).
        self.code.extend_from_slice(&[w, 0xf7, 0xd0 | m]);
        self.code.extend_from_slice(&[w, 0xc1, 0xe0 | m, 1]);
        self.code.extend_from_slice(&[w, 0xc1, 0xe8 | m, 51]);
        self.g_jcc(cc::NE, number);
    }

    /// `test byte [base + disp], imm`.
    fn fr_test_byte(&mut self, base: u8, disp: i32, imm: u8) {
        if base >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0xf6);
        self.code.push(0x80 | (base & 7));
        if base & 7 == 4 {
            self.code.push(0x24);
        }
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.push(imm);
    }

    /// `cmp reg32, 64`, then to `ok` when it is at least that (an
    /// untracked slot) — and `bt mask, reg` for a tracked one.
    fn fr_tracked_bit(&mut self, slot: u8, mask: u8, ok: Label, miss: Label) {
        if slot >= 8 {
            self.code.push(0x41);
        }
        self.code.extend_from_slice(&[0x83, 0xf8 | (slot & 7), 64]);
        self.g_jcc(cc::AE, ok);
        // bt mask, slot (64-bit): CF = the slot's "may be anything" bit.
        self.code
            .push(0x48 | (((slot >> 3) & 1) << 2) | ((mask >> 3) & 1));
        self.code
            .extend_from_slice(&[0x0f, 0xa3, 0xc0 | ((slot & 7) << 3) | (mask & 7)]);
        self.g_jcc(cc::AE, miss);
    }

    /// The representation check of an inline write of register `val` into
    /// slot `slot` of an object whose layout is compiled in (its shape word
    /// `shape`): to `miss` when the value is not a Number and the layout
    /// still has the slot as Number-only. Emits nothing for an untracked
    /// slot (64 on). Clobbers `scratch`.
    pub(crate) fn p_field_rep_const(
        &mut self,
        l: &JitLayout,
        (val, scratch): (u8, u8),
        shape: u64,
        slot: u32,
        miss: Label,
    ) {
        if slot >= 64 {
            return;
        }
        let ok = self.new_label();
        self.fr_to_if_number(val, scratch, ok);
        self.g_movabs(
            scratch,
            shape.wrapping_add_signed(i64::from(l.props.shape_any)),
        );
        self.fr_test_byte(scratch, (slot / 8) as i32, 1 << (slot % 8));
        self.g_jcc(cc::E, miss);
        self.bind(ok);
    }

    /// [`Self::p_field_rep_const`] for a layout and slot known only at run
    /// time: the shape word in register `shape`, the slot index in the low
    /// 32 bits of register `slot` (zero-extended). Clobbers `scratch` and
    /// `shape`.
    pub(crate) fn p_field_rep_dyn(
        &mut self,
        l: &JitLayout,
        (val, shape, slot, scratch): (u8, u8, u8, u8),
        miss: Label,
    ) {
        let ok = self.new_label();
        self.fr_to_if_number(val, scratch, ok);
        self.g_load(shape, shape, l.props.shape_any);
        self.fr_tracked_bit(slot, shape, ok, miss);
        self.bind(ok);
    }

    /// [`Self::p_field_rep_dyn`] for an inline add to the object whose heap
    /// slot is `rcx`: the new value goes into the slot just past its current
    /// ones (its slot count), of the layout whose shape word is in register
    /// `to`. Clobbers `scratch`, `count` and `to`.
    pub(crate) fn p_field_rep_add(
        &mut self,
        l: &JitLayout,
        (val, to, count, scratch): (u8, u8, u8, u8),
        miss: Label,
    ) {
        let ok = self.new_label();
        self.fr_to_if_number(val, scratch, ok);
        // movzx count32, word [rcx + off_slots_len]
        let rex = 0x40 | (((count >> 3) & 1) << 2);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code
            .extend_from_slice(&[0x0f, 0xb7, 0x80 | ((count & 7) << 3) | (r::RCX & 7)]);
        self.code.extend_from_slice(&l.off_slots_len.to_le_bytes());
        self.g_load(to, to, l.props.shape_any);
        self.fr_tracked_bit(count, to, ok, miss);
        self.bind(ok);
    }

    /// The representation checks of an inline object literal's values (at
    /// the frame displacements `vals`, slots `0..vals.len()` of the layout
    /// whose shape word is at `[cache + shape_off]`): to `miss` when one is
    /// not a Number and the layout still has its slot as Number-only. A
    /// value `proven` a Number needs no check; for the others the slot's
    /// mark is tested first (a literal's slot holds the same kind of value
    /// every time, so a slot that takes a non-Number is marked from its
    /// first one on). The values must be current in the register file.
    /// Clobbers `rax`, `rdx` and `r11`.
    pub(crate) fn p_field_rep_lit(
        &mut self,
        l: &JitLayout,
        (cache, shape_off): (u64, i32),
        vals: &[i32],
        proven: &[bool],
        miss: Label,
    ) {
        let check: Vec<(usize, i32)> = vals
            .iter()
            .enumerate()
            .take(31)
            .filter(|&(i, _)| !proven.get(i).copied().unwrap_or(false))
            .map(|(i, &d)| (i, d))
            .collect();
        if check.is_empty() {
            return;
        }
        // All of them marked (the usual case: a slot that takes a
        // non-Number is marked from its first one on), else each one.
        let all: u32 = check.iter().map(|&(i, _)| 1u32 << i).sum();
        let done = self.new_label();
        self.g_movabs(r::R11, cache);
        self.g_load(r::R11, r::R11, shape_off);
        self.g_load(r::R11, r::R11, l.props.shape_any);
        self.code.extend_from_slice(&[0x49, 0xf7, 0xd3]); // not r11
        // test r11, all
        self.code.extend_from_slice(&[0x49, 0xf7, 0xc3]);
        self.code.extend_from_slice(&all.to_le_bytes());
        self.g_jcc(cc::E, done);
        for (i, d) in check {
            let ok = self.new_label();
            // test r11, 1 << i: a marked slot (clear in `r11`) takes anything.
            self.code.extend_from_slice(&[0x49, 0xf7, 0xc3]);
            self.code.extend_from_slice(&(1u32 << i).to_le_bytes());
            self.g_jcc(cc::E, ok);
            self.g_load(r::RAX, r::RBX, d);
            self.fr_to_if_number(r::RAX, r::RDX, ok);
            self.jmp(miss);
            self.bind(ok);
        }
        self.bind(done);
    }

    /// To `fail` unless every bit of `mask` is clear in the `u64` at
    /// `addr` (a layout's may-be-non-Number mask): a speculative fast
    /// version's check that the Number-only slots it reads without a type
    /// check still are. Clobbers `r11`.
    pub(crate) fn p_fields_still_numbers(&mut self, addr: u64, mask: u64, fail: Label) {
        if mask == 0 {
            return;
        }
        self.g_movabs(r::R11, addr);
        if mask < 1 << 31 {
            // test qword [r11], imm32
            self.code.extend_from_slice(&[0x49, 0xf7, 0x03]);
            self.code
                .extend_from_slice(&u32::try_from(mask).unwrap_or(0).to_le_bytes());
            self.g_jcc(cc::NE, fail);
        } else {
            for (i, b) in mask.to_le_bytes().into_iter().enumerate() {
                if b != 0 {
                    self.fr_test_byte(r::R11, i as i32, b);
                    self.g_jcc(cc::NE, fail);
                }
            }
        }
    }
}
