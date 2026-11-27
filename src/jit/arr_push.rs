//! A compiled `a.push(x)`'s append, and `a.pop()`'s removal, with no call.
//!
//! The site's method proof (`MethodSite::set_str_proof`, made by
//! `native_call::prove_array_push`) says that the lookup of `push` on an
//! array without side-table state or a set `[[Prototype]]` finds the
//! built-in (`Realm::plain_recv_lookup_start` starts it at the realm's first
//! `%Array.prototype%`, which owns it as a data property). With that,
//! `push` of one value is exactly what `Realm::array_push_plain` does — the
//! array's elements gain the value at the end — under the same guards:
//! no array prototype has an index accessor, the array never had side-table
//! state (so it is extensible, its `length` is writable and is its element
//! count), and the new length is within `max_array_len`. The append is done
//! here when it needs no allocation (an inline element left, or the `Vec`'s
//! spare capacity) and nothing from the write barrier
//! (`p_barrier_check_appended`, with an append at or past an `APPENDED`
//! container's scan position covered as `Heap::get_mut_append` covers it).
//! Anything else is the call's.
//!
//! `pop()` likewise (its site's proof, by `native_call::prove_array_pop`, is
//! of `pop`): on such an array it is exactly what `Realm::array_pop_plain`
//! does — the last element removed and returned, `undefined` from an empty
//! array — under the same guards, and declines a hole (whose read the
//! prototype chain would answer). A removal records no new reference, so
//! the write barrier has nothing to do; an inline element's vacated slot
//! is reset to `undefined` (`ArrayElems::inline`'s invariant).
//!
//! `shift()` (its site's proof, by `native_call::prove_array_shift`, is of
//! `shift`): on such an array of at most one element it removes exactly
//! what `pop()` does — `Realm::array_shift_fast`'s outcome there, a hole
//! declined likewise. A longer array, whose shift moves elements (which an
//! incremental marking's scan position would have to cover), is the call's.

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};

/// Whether a compiled `a.shift()` tries the inline removal before its call
/// (`KATAAN_JIT_SHIFT=0` turns it off, read once per process).
#[must_use]
pub(crate) fn shift_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("KATAAN_JIT_SHIFT").map_or(true, |v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// Whether a compiled read of `a.push` / `a.pop` / `a.shift` (the method
/// read before a call's arguments) on an array that owns no named property
/// takes the built-in its site's proof found, with no helper call
/// (`KATAAN_JIT_ARRMETHOD=0` turns it off, read once per process).
#[must_use]
pub(crate) fn method_read_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("KATAAN_JIT_ARRMETHOD").map_or(true, |v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// The most values a compiled `a.push(…)` appends in place
/// ([`X64Assembler::ap_push_n`]); a longer one is the call's.
pub(crate) const PUSH_N_MAX: usize = 4;

impl X64Assembler {
    /// For the receiver whose live heap slot is `rcx`: appends the value at
    /// `[rbx + val_disp]` as the module docs say, leaving `r8` = the new
    /// length; else to `miss` with nothing done. Clobbers `rax`, `rdx`,
    /// `r8`–`r11`; preserves `rcx`.
    pub(super) fn ap_push(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        val_disp: i32,
        miss: Label,
        proved: bool,
    ) {
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_side, 0);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_proto_set, 0);
        self.jne(miss);
        // (`proved`: the site's run proof covers the realm's flag.)
        if !proved {
            self.g_load(r::R11, r::R15, realm_off);
            self.g_cmp_byte_imm(r::R11, l.props.proto_index_dirty, 0);
            self.g_jcc(cc::NE, miss);
        }
        // The write barrier: nothing to do, or an append it covers.
        let appended = self.new_label();
        let checked = self.new_label();
        self.g_load(r::RAX, r::RBX, val_disp);
        self.p_barrier_check_appended(l, realm_off, miss, appended);
        self.bind(checked);
        // r8 = the length, below `max_array_len`.
        self.p_arr_len(l, r::R8);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_cmp_mem(r::R8, r::R11, l.props.max_array_len);
        self.g_jcc(cc::AE, miss);
        let heap = self.new_label();
        let stored = self.new_label();
        self.g_load(r::RDX, r::RBX, val_disp);
        self.p_load8(r::R9, r::RCX, l.off_arr_inline_len);
        self.g_cmp_imm(r::R9, i32::from(crate::cell::ARRAY_ON_HEAP));
        self.g_jcc(cc::E, heap);
        // Inline: an element left in the cell.
        self.g_cmp_imm(
            r::R9,
            i32::try_from(crate::cell::ARRAY_INLINE_CAP).expect("small"),
        );
        self.g_jcc(cc::AE, miss);
        self.p_store_index8(r::RCX, r::R8, l.off_arr_inline, r::RDX);
        // inc byte [rcx + disp32]
        self.raw(&[0xfe, 0x81]);
        self.raw(&l.off_arr_inline_len.to_le_bytes());
        self.jmp(stored);
        // In the `Vec`: its spare capacity.
        self.bind(heap);
        self.g_cmp_mem(r::R8, r::RCX, l.off_arr_cap);
        self.g_jcc(cc::AE, miss);
        self.g_load(r::R11, r::RCX, l.off_arr_ptr);
        self.p_store_index8(r::R11, r::R8, 0, r::RDX);
        self.raw(&[0x49, 0xff, 0xc0]); // inc r8
        self.g_store(r::RCX, l.off_arr_len, r::R8);
        let out = self.new_label();
        self.jmp(out);
        self.bind(stored);
        self.raw(&[0x49, 0xff, 0xc0]); // inc r8
        self.bind(out);
        // (Out of line: an `APPENDED` container, whose scan position must be
        // at or below the new element's index.)
        let past = self.new_label();
        self.jmp(past);
        let cold = self.cold_begin();
        self.bind(appended);
        self.p_arr_len(l, r::RAX);
        self.p_scan_covers(l, miss);
        self.jmp(checked);
        self.cold_end(cold);
        self.bind(past);
    }

    /// [`ap_push`](Self::ap_push) of several values (`a.push(x, y)`, the
    /// registers at `[rbx + vals[k]]`, in order): `Realm::array_push_plain`'s
    /// append of each in turn, under the same guards, done here only when
    /// all of them fit (every value's write barrier has nothing to do, the
    /// new length stays below `max_array_len`, and the inline elements or
    /// the `Vec`'s spare capacity hold them all), so that nothing is done
    /// when it goes to `miss`. Leaves `r8` = the new length. Clobbers `rax`,
    /// `rdx`, `r8`–`r11`; preserves `rcx`.
    pub(super) fn ap_push_n(&mut self, l: &JitLayout, realm_off: i32, vals: &[i32], miss: Label) {
        let n = i32::try_from(vals.len()).expect("few values");
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_side, 0);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_proto_set, 0);
        self.jne(miss);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R11, l.props.proto_index_dirty, 0);
        self.g_jcc(cc::NE, miss);
        // The write barrier, per value: nothing to do, or an append it
        // covers (every new index is at or past the first, the length).
        let mut appended = alloc::vec::Vec::with_capacity(vals.len());
        for &v in vals {
            let hit = self.new_label();
            let checked = self.new_label();
            self.g_load(r::RAX, r::RBX, v);
            self.p_barrier_check_appended(l, realm_off, miss, hit);
            self.bind(checked);
            appended.push((hit, checked));
        }
        // r8 = the length; the last new index below `max_array_len`.
        self.p_arr_len(l, r::R8);
        self.g_lea(r::RAX, r::R8, n - 1);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_cmp_mem(r::RAX, r::R11, l.props.max_array_len);
        self.g_jcc(cc::AE, miss);
        let heap = self.new_label();
        let stored = self.new_label();
        self.p_load8(r::R9, r::RCX, l.off_arr_inline_len);
        self.g_cmp_imm(r::R9, i32::from(crate::cell::ARRAY_ON_HEAP));
        self.g_jcc(cc::E, heap);
        // Inline: room left in the cell for them all.
        let cap = i32::try_from(crate::cell::ARRAY_INLINE_CAP).expect("small");
        if cap >= n {
            self.g_cmp_imm(r::R9, cap - n + 1);
            self.g_jcc(cc::AE, miss);
            for (k, &v) in vals.iter().enumerate() {
                let k = i32::try_from(k).expect("few values");
                self.g_load(r::RDX, r::RBX, v);
                self.p_store_index8(r::RCX, r::R8, l.off_arr_inline + 8 * k, r::RDX);
            }
            // add byte [rcx + disp32], n
            self.raw(&[0x80, 0x81]);
            self.raw(&l.off_arr_inline_len.to_le_bytes());
            self.raw(&[u8::try_from(n).expect("few values")]);
            self.jmp(stored);
        } else {
            self.jmp(miss);
        }
        // In the `Vec`: its spare capacity.
        self.bind(heap);
        self.g_lea(r::RAX, r::R8, n);
        self.g_cmp_mem(r::RAX, r::RCX, l.off_arr_cap);
        self.g_jcc(cc::A, miss);
        self.g_load(r::R11, r::RCX, l.off_arr_ptr);
        for (k, &v) in vals.iter().enumerate() {
            let k = i32::try_from(k).expect("few values");
            self.g_load(r::RDX, r::RBX, v);
            self.p_store_index8(r::R11, r::R8, 8 * k, r::RDX);
        }
        self.g_lea(r::R8, r::R8, n);
        self.g_store(r::RCX, l.off_arr_len, r::R8);
        let out = self.new_label();
        self.jmp(out);
        self.bind(stored);
        self.g_lea(r::R8, r::R8, n);
        self.bind(out);
        // (Out of line: an `APPENDED` container, whose scan position must be
        // at or below the first new element's index.)
        let past = self.new_label();
        self.jmp(past);
        let cold = self.cold_begin();
        for (hit, checked) in appended {
            self.bind(hit);
            self.p_arr_len(l, r::RAX);
            self.p_scan_covers(l, miss);
            self.jmp(checked);
        }
        self.cold_end(cold);
        self.bind(past);
    }

    /// `cmp rax, <the hole>`: against the code's constant word (with
    /// `KATAAN_JIT_ITERCONST`, as the array iteration steps), else through
    /// `r11`.
    fn ap_cmp_hole(&mut self) {
        let hole = crate::nanbox::NanBox::hole().to_bits();
        if super::iter_const_enabled() {
            self.cmp_rax_const(hole);
        } else {
            self.g_movabs(r::R11, hole);
            self.g_cmp(r::RAX, r::R11);
        }
    }

    /// For the receiver whose live heap slot is `rcx`: removes its last
    /// element as the module docs say, leaving it (or `undefined`) in
    /// `rax`; else to `miss` with nothing done. Clobbers `rdx`, `r8`–`r11`;
    /// preserves `rcx`.
    pub(super) fn ap_pop(&mut self, l: &JitLayout, realm_off: i32, miss: Label, proved: bool) {
        self.ap_take(l, realm_off, miss, false, proved);
    }

    /// For the receiver whose live heap slot is `rcx`: `a.shift()` on an
    /// array of at most one element, where it removes exactly what `pop()`
    /// does (as the module docs say), leaving it (or `undefined`) in `rax`;
    /// a longer array (whose shift moves elements) to `miss` with nothing
    /// done. Clobbers `rdx`, `r8`–`r11`; preserves `rcx`.
    pub(super) fn ap_shift(&mut self, l: &JitLayout, realm_off: i32, miss: Label, proved: bool) {
        self.ap_take(l, realm_off, miss, true, proved);
    }

    /// [`ap_pop`](Self::ap_pop), or with `at_most_one` [`ap_shift`](Self::ap_shift).
    fn ap_take(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        miss: Label,
        at_most_one: bool,
        proved: bool,
    ) {
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_side, 0);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_proto_set, 0);
        self.jne(miss);
        // (`proved`: the site's run proof covers the realm's flag.)
        if !proved {
            self.g_load(r::R11, r::R15, realm_off);
            self.g_cmp_byte_imm(r::R11, l.props.proto_index_dirty, 0);
            self.g_jcc(cc::NE, miss);
        }
        let empty = self.new_label();
        let heap = self.new_label();
        let out = self.new_label();
        self.g_movabs(r::RDX, crate::nanbox::NanBox::undefined().to_bits());
        self.p_load8(r::R9, r::RCX, l.off_arr_inline_len);
        self.g_cmp_imm(r::R9, i32::from(crate::cell::ARRAY_ON_HEAP));
        self.g_jcc(cc::E, heap);
        // Inline: `r9` elements in the cell.
        if at_most_one {
            self.g_cmp_imm(r::R9, 1);
            self.g_jcc(cc::A, miss);
        }
        self.g_test(r::R9);
        self.g_jcc(cc::E, empty);
        self.raw(&[0x49, 0xff, 0xc9]); // dec r9
        self.p_load_index8(r::RAX, r::RCX, r::R9, l.off_arr_inline);
        self.ap_cmp_hole();
        self.g_jcc(cc::E, miss);
        self.p_store_index8(r::RCX, r::R9, l.off_arr_inline, r::RDX);
        // dec byte [rcx + disp32]
        self.raw(&[0xfe, 0x89]);
        self.raw(&l.off_arr_inline_len.to_le_bytes());
        self.jmp(out);
        // In the `Vec`: its length, less one.
        self.bind(heap);
        self.g_load(r::R8, r::RCX, l.off_arr_len);
        if at_most_one {
            self.g_cmp_imm(r::R8, 1);
            self.g_jcc(cc::A, miss);
        }
        self.g_test(r::R8);
        self.g_jcc(cc::E, empty);
        self.raw(&[0x49, 0xff, 0xc8]); // dec r8
        self.g_load(r::R11, r::RCX, l.off_arr_ptr);
        self.p_load_index8(r::RAX, r::R11, r::R8, 0);
        self.ap_cmp_hole();
        self.g_jcc(cc::E, miss);
        self.g_store(r::RCX, l.off_arr_len, r::R8);
        self.jmp(out);
        self.bind(empty);
        self.g_mov(r::RAX, r::RDX);
        self.bind(out);
    }
}
