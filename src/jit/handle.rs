//! The **handle walk**: from a `NanBox` heap handle to the address of its
//! heap slot, as compiled code does it before every inline heap access.
//!
//! A handle is an arena index plus the generation its slot was live in
//! (`heap::Handle`). The fully checked walk (`p_slot_of`) proves four
//! things: the value is a handle, its index is within the arena, the slot
//! is occupied, and the slot's generation is the handle's. The last two
//! cost two loads, two compares and the generation's extraction on every
//! walk. Where the handle comes from decides which of them are needed.
//!
//! **A rooted handle** (read from the register window, a frame slot, or a
//! property of an object reached that way) refers to a live slot of its
//! generation, by the collector's own invariant: a slot is freed only when
//! no root reaches it, and freeing is what changes a generation (the
//! collector does not move objects at run time: `compact_to` serves
//! snapshots and tests, and fixes every root it moves). So for a rooted
//! handle the generation compare can never fail, and neither can the
//! occupancy one. [`X64Assembler::p_slot_rooted`] emits only the handle
//! test, the bounds check and the address.
//!
//! That walk stays **memory-safe even if the invariant were broken** (a
//! collector bug): the index is still bounds-checked against the live arena
//! length, so the address is a real slot of the arena, and every caller
//! compares the slot's cell tag against a live variant's before reading any
//! other byte of the cell. A vacant slot's tag byte holds `Option`'s `None`
//! niche, which is no variant's tag (`heap::SlotLayout::vacant_tag`, asserted
//! in `compute_jit_layout`), so the tag compare also proves the slot
//! occupied; a reused slot holds a real, live cell whose tag is what it is,
//! and the reads that follow are validated against that cell. What a broken
//! invariant would give is a wrong object, never a wild read. (The only
//! header byte read without a tag compare is the barrier's `age`, a plain
//! initialized `u8` of every slot.)
//!
//! **A handle from a side table** (a method cache's holder, the realm's
//! array-iteration memo: raw bits the collector does not trace) can be stale
//! and still takes the checked walk.
//!
//! To keep that invariant tested rather than assumed, **audit mode**
//! (debug builds, or `KATAAN_JIT_HANDLE_AUDIT=1`) emits the occupancy and
//! generation compares after every rooted walk, trapping (`ud2`) on a
//! mismatch: any gap in the root set that reaches compiled code aborts the
//! run there. (`KATAAN_GC_VERIFY` is unaffected: compiled code never
//! consulted its poison set, and in that mode nothing is freed.)

use super::call_seq::r;
use super::{JitLayout, Label, X64Assembler};

/// Whether rooted walks also emit the trapping liveness compares (see the
/// module docs). Read once per process.
pub(crate) fn handle_audit() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(debug_assertions)
            || std::env::var_os("KATAAN_JIT_HANDLE_AUDIT").is_some_and(|v| v != "0")
    })
}

impl X64Assembler {
    /// To `miss` unless `rax` holds a heap handle (all of `HANDLE_TAG`'s
    /// bits, the top 14), via `scratch` (`rcx` or `r9`), which it clobbers.
    pub(crate) fn p_is_handle(&mut self, scratch: u8, miss: Label) {
        const _: () = assert!(crate::nanbox::NanBox::handle(0).to_bits() == !0u64 << 50);
        debug_assert!(scratch == r::RCX || scratch == r::R9);
        let b = (scratch >> 3) & 1;
        let s = scratch & 7;
        // mov scratch, rax
        self.code.extend_from_slice(&[0x48 | b, 0x89, 0xc0 | s]);
        // shr scratch, 50
        self.code.extend_from_slice(&[0x48 | b, 0xc1, 0xe8 | s, 50]);
        // cmp scratch32, 0x3fff
        if b != 0 {
            self.code.push(0x41);
        }
        self.code
            .extend_from_slice(&[0x81, 0xf8 | s, 0xff, 0x3f, 0x00, 0x00]);
        self.jne(miss);
    }

    /// The walk of a **rooted** handle (see the module docs): from the
    /// handle bits in `rax`, `dst` (`rcx` or `r9`) = the address of the
    /// arena slot at its index, else (index out of bounds) to `miss`. The
    /// caller must compare the slot's cell tag before reading any byte of
    /// its cell. Needs the arena (`p_arena`); preserves `rax`; clobbers `r8`
    /// in audit mode only.
    pub(crate) fn p_slot_rooted(&mut self, l: &JitLayout, dst: u8, miss: Label) {
        debug_assert!(dst == r::RCX || dst == r::R9);
        let x = (dst >> 3) & 1;
        let d = dst & 7;
        // mov dst32, eax: the index, zero-extended.
        if x != 0 {
            self.code.push(0x41);
        }
        self.code.extend_from_slice(&[0x89, 0xc0 | d]);
        // cmp dst, <len> (`r13` pinned, else `rdx`)
        let (len_r, len_x) = if self.pinned() { (5, 1) } else { (r::RDX, 0) };
        self.code
            .extend_from_slice(&[0x48 | (len_x << 2) | x, 0x39, 0xc0 | (len_r << 3) | d]);
        self.jae(miss);
        // The arena base: `r12` pinned, else `r10` (both need REX.B).
        let base = if self.pinned() {
            super::pin::R12 & 7
        } else {
            r::R10 & 7
        };
        if l.slot_stride == 72 {
            // lea dst, [dst + dst*8]; lea dst, [base + dst*8]
            let rex = 0x48 | (x << 2) | (x << 1) | x;
            self.code
                .extend_from_slice(&[rex, 0x8d, (d << 3) | 4, 0xc0 | (d << 3) | d]);
            let rex = 0x48 | (x << 2) | (x << 1) | 1;
            self.code
                .extend_from_slice(&[rex, 0x8d, (d << 3) | 4, 0xc0 | (d << 3) | base]);
        } else {
            // imul dst, dst, stride; add dst, base
            self.code
                .extend_from_slice(&[0x48 | (x << 2) | x, 0x69, 0xc0 | (d << 3) | d]);
            self.code.extend_from_slice(&l.slot_stride.to_le_bytes());
            self.code
                .extend_from_slice(&[0x4c | x, 0x01, 0xc0 | (base << 3) | d]);
        }
        if handle_audit() {
            self.p_audit_live(l, dst);
        }
    }

    /// Audit mode's check after a rooted walk: the slot at `dst` occupied
    /// and of `rax`'s generation, or `ud2`. Clobbers `r8`.
    fn p_audit_live(&mut self, l: &JitLayout, dst: u8) {
        let ok = self.new_label();
        let trap = self.new_label();
        self.g_cmp_byte_imm(dst, l.off_slot_occ, l.slot_occupied_disc);
        self.jne(trap);
        self.mov_r8_rax();
        self.shr_r8_imm(32);
        // cmp word [dst + disp32], r8w
        self.code
            .extend_from_slice(&[0x66, 0x44 | ((dst >> 3) & 1), 0x39, 0x80 | (dst & 7)]);
        self.code.extend_from_slice(&l.off_slot_gen.to_le_bytes());
        self.je(ok);
        self.bind(trap);
        self.code.extend_from_slice(&[0x0f, 0x0b]); // ud2
        self.bind(ok);
    }
}

#[cfg(test)]
mod tests {
    use super::super::pin::PinArena;
    use super::*;

    /// The rooted walk's bytes, pinned and not, into `rcx` and `r9`
    /// (outside audit mode, which debug builds turn on).
    #[test]
    fn rooted_walk_encodings() {
        if handle_audit() {
            return;
        }
        let l = super::super::compute_jit_layout();
        assert_eq!(l.slot_stride, 72);
        let mut a = X64Assembler::new();
        let miss = a.new_label();
        a.p_slot_rooted(l, r::RCX, miss);
        let c = a.code();
        assert_eq!(&c[..5], &[0x89, 0xc1, 0x48, 0x39, 0xd1]); // mov ecx, eax; cmp rcx, rdx
        assert_eq!(
            &c[c.len() - 8..],
            &[0x48, 0x8d, 0x0c, 0xc9, 0x49, 0x8d, 0x0c, 0xca] // lea rcx, [rcx+rcx*8]; lea rcx, [r10+rcx*8]
        );
        let mut a = X64Assembler::new();
        a.set_pin(Some(PinArena {
            realm: 0x10,
            heap_ptr: 0x20,
            heap_len: 0x28,
            save: -48,
        }));
        let miss = a.new_label();
        a.p_slot_rooted(l, r::R9, miss);
        let c = a.code();
        assert_eq!(&c[..6], &[0x41, 0x89, 0xc1, 0x4d, 0x39, 0xe9]); // mov r9d, eax; cmp r9, r13
        assert_eq!(
            &c[c.len() - 8..],
            &[0x4f, 0x8d, 0x0c, 0xc9, 0x4f, 0x8d, 0x0c, 0xcc] // lea r9, [r9+r9*8]; lea r9, [r12+r9*8]
        );
        let mut a = X64Assembler::new();
        let miss = a.new_label();
        a.p_is_handle(r::R9, miss);
        assert_eq!(
            &a.code()[..14],
            &[
                0x49, 0x89, 0xc1, // mov r9, rax
                0x49, 0xc1, 0xe9, 50, // shr r9, 50
                0x41, 0x81, 0xf9, 0xff, 0x3f, 0x00, 0x00, // cmp r9d, 0x3fff
            ]
        );
    }
}
