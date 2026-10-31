//! **Element reads in the integer version**: a loop whose only generic ops
//! are element reads (`a[i]`) and `.length` reads, of typed arrays or plain
//! arrays, whose values only feed integer ops, gets an integer version
//! ([`super::plan`] with `elems`), so a hash or bit-set loop (`c =
//! T[(c ^ b[i]) & 0xff] ^ (c >>> 8)`, `rows[R[i]] | cols[C[i]]`) keeps
//! its values as int32s in GPRs instead of converting each one between a
//! double and an integer at every op.
//!
//! The read walks the receiver as the generic emitter does (no hoisting:
//! nothing is cached across accesses), with the index from its int
//! register. A dense array's element is taken when it is exactly an int32
//! Number; a typed-array view's when its kind is an integer one (a Uint32
//! element past `i32::MAX` only where the value is *wrapped*, read only by
//! bitwise ops, see `IntPlan::wrap_bits`: its low 32 bits, which is all
//! `ToInt32` sees); a `.length` when it is at most `i32::MAX`. Anything
//! else (a hole, a Float64 view, out of bounds, not an array) **bails**
//! before the op has done anything: the double version's generic code runs
//! it, exactly. The walk clobbers `rax`, `rcx`, `rdx`, `r8`..`r11`, so the
//! int registers live in `r8`/`r9`/`r10` there are pushed around it (and
//! popped on every way out, a bail's included).

use super::super::{NumGen, RAX, RCX, disp};
use super::{CC_NE, IK};
use crate::jit::X64Assembler;
use crate::jit::typed::{self, Src};

impl NumGen {
    /// The int GPRs the walk clobbers that hold an int register live at the
    /// current pc other than `dst` (whose value the op replaces).
    fn walk_saves(&self, dst: u16) -> alloc::vec::Vec<u8> {
        let pc = self.cur_pc;
        let Some(p) = self.iplan() else {
            return alloc::vec::Vec::new();
        };
        let mut out: alloc::vec::Vec<u8> = p
            .slots
            .iter()
            .filter(|s| s.reg != dst && s.start <= pc && pc <= s.end && (8..=10).contains(&s.gpr))
            .map(|s| s.gpr)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Walks register `obj` (in the register file) to its slot in `rcx`, to
    /// `miss` when it is not a heap handle.
    fn iwalk(&self, a: &mut X64Assembler, obj: u16, miss: crate::jit::Label) {
        let l = &self.env.heap;
        a.p_arena(self.env.realm, &l.props);
        a.load_rax(disp(obj));
        a.p_is_handle(RCX, miss);
        a.p_slot_rooted(l, RCX, miss);
    }

    /// `dst = obj[index]` in the integer version (`dst`, `index` int
    /// registers), see the module docs.
    pub(super) fn iget_elem(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        obj: u16,
        index: u16,
    ) -> Option<()> {
        let IK::Gpr(d) = self.ik(dst)? else {
            return None;
        };
        let wrap = self.iplan()?.wrap_bits.contains(&self.cur_pc);
        let bail = self.ibail(a);
        let saves = self.walk_saves(dst);
        // The index, sign-extended, rides on the stack across the walk.
        match self.ik(index)? {
            IK::Gpr(g) => a.movsxd(RAX, g),
            IK::Imm(v) => a.movabs_rax(i64::from(v)),
        }
        for &g in &saves {
            a.raw(&[0x41, 0x50 + (g & 7)]); // push r8..r10
        }
        a.push_rax();
        let miss = a.new_label();
        let got = a.new_label();
        let not_arr = a.new_label();
        let l = self.env.heap;
        self.iwalk(a, obj, miss);
        // A dense array's element: an exact int32 Number (a hole is not one).
        a.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        a.jne(not_arr);
        a.p_array_plain(&l, self.env.realm, false, miss);
        a.raw(&[0x48, 0x8b, 0x04, 0x24]); // mov rax, [rsp]
        a.p_arr_len_ptr(&l);
        a.cmp_rax_r8();
        a.jae(miss); // past the length, or negative
        a.mov_rax_at_r11_rax8();
        a.exact_i32_check(RCX);
        a.jcc(CC_NE, miss);
        a.mov32_rr(RAX, RCX);
        a.jmp(got);
        // A typed-array view's element of an integer kind.
        a.bind(not_arr);
        let tl = typed::typed_layout();
        a.t_view(&l, tl, miss);
        a.raw(&[0x48, 0x8b, 0x04, 0x24]); // mov rax, [rsp]
        a.t_get_i32(tl, Src::View, wrap, miss, got);
        a.bind(got);
        a.raw(&[0x48, 0x83, 0xc4, 0x08]); // add rsp, 8 (the index)
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]); // pop r8..r10
        }
        a.mov32_rr(d, RAX);
        let done = a.new_label();
        a.jmp(done);
        a.bind(miss);
        a.raw(&[0x48, 0x83, 0xc4, 0x08]); // add rsp, 8
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]);
        }
        a.jmp(bail);
        a.bind(done);
        Some(())
    }

    /// `dst = arr.length` in the integer version (`dst` an int register):
    /// a plain array's or a typed-array view's, at most `i32::MAX`.
    pub(super) fn iarray_len(&mut self, a: &mut X64Assembler, dst: u16, arr: u16) -> Option<()> {
        let IK::Gpr(d) = self.ik(dst)? else {
            return None;
        };
        let bail = self.ibail(a);
        let saves = self.walk_saves(dst);
        for &g in &saves {
            a.raw(&[0x41, 0x50 + (g & 7)]);
        }
        let miss = a.new_label();
        let got = a.new_label();
        let not_arr = a.new_label();
        let l = self.env.heap;
        self.iwalk(a, arr, miss);
        a.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        a.jne(not_arr);
        a.p_array_plain(&l, self.env.realm, false, miss);
        a.p_arr_len(&l, RAX);
        a.jmp(got);
        a.bind(not_arr);
        a.t_view(&l, typed::typed_layout(), miss);
        a.t_len(typed::typed_layout(), self.env.realm, miss, got);
        a.bind(got);
        // cmp rax, 0x7fffffff; ja miss (an unsigned length past i32::MAX)
        a.raw(&[0x48, 0x3d, 0xff, 0xff, 0xff, 0x7f]);
        a.jcc(super::super::CC_A, miss);
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]);
        }
        a.mov32_rr(d, RAX);
        let done = a.new_label();
        a.jmp(done);
        a.bind(miss);
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]);
        }
        a.jmp(bail);
        a.bind(done);
        Some(())
    }
}
