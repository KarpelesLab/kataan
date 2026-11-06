//! A compiled `a.push(x)`'s append with no call.
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

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};

impl X64Assembler {
    /// For the receiver whose live heap slot is `rcx`: appends the value at
    /// `[rbx + val_disp]` as the module docs say, leaving `r8` = the new
    /// length; else to `miss` with nothing done. Clobbers `rax`, `rdx`,
    /// `r8`–`r11`; preserves `rcx`.
    pub(super) fn ap_push(&mut self, l: &JitLayout, realm_off: i32, val_disp: i32, miss: Label) {
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_side, 0);
        self.jne(miss);
        self.cmp_byte_at_rcx_imm(l.off_arr_proto_set, 0);
        self.jne(miss);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R11, l.props.proto_index_dirty, 0);
        self.g_jcc(cc::NE, miss);
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
}
