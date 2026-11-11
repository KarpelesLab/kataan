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

use super::super::{NumGen, RAX, RCX, RDX, disp};
use super::{CC_E, CC_NE, IK, R11};
use crate::jit::X64Assembler;
use crate::jit::typed::{self, Src};

/// Whether a hoisted array's walked read makes the row it walked the
/// hoisted one (`NumGen::iwalk_arr_get`; `KATAAN_JIT_ROWMEMO=0` turns it
/// off, read once).
fn row_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_ROWMEMO").is_none_or(|v| v != "0"))
}

impl NumGen {
    /// At inner loop header `h` of the integer version being emitted, on
    /// entry into the loop: its row fills (`Region::fills`) — each
    /// register's array (or view) becomes its hoisted one, as the region's
    /// entry check would make it (else its slots say there is none), so the
    /// loop's accesses of it need no compare (`NumGen::row_known`).
    pub(in super::super) fn irow_fills(&mut self, a: &mut X64Assembler, h: usize) -> Option<()> {
        let ri = self.imode?;
        let Some(mut regs) = self.regions[ri]
            .fills
            .iter()
            .find(|f| f.0 == h)
            .map(|f| f.2.clone())
        else {
            return Some(());
        };
        regs.sort_unstable();
        regs.dedup();
        self.cur_pc = h;
        let l = self.env.heap;
        // Plain arrays' only (no register with a hoisted view too), with the
        // arena pinned: only `rax`, `rcx`, `r11` and `xmm0` (`p_fill_array`).
        let arrays = regs.iter().all(|&r| {
            self.hoisted_array(h, r).is_some() && self.hoisted_view_slots(ri, r).is_none()
        });
        if arrays && a.pinned() {
            for r in regs {
                self.iload_operand(a, r);
                if let Some(s) = self.hoisted_array(h, r) {
                    a.p_fill_array(&l, self.env.realm, s);
                }
            }
            return Some(());
        }
        // (The checks clobber `rax`, `rcx`, `r8`, `r9`, `r11`, unpinned
        // `rdx` / `r10`, and `xmm0`.)
        let saves = self.walk_saves(u16::MAX);
        for &g in &saves {
            a.raw(&[0x41, 0x50 + (g & 7)]); // push r8..r10
        }
        for r in regs {
            // (Its value into the register file, where the check reads it.)
            if let Some(x) = self.xmm_of(self.imode, r) {
                a.movq_gpr_x(RAX, x);
                a.store_rax(disp(r));
            }
            let from = typed::HoistFrom::Reg(disp(r));
            // (Both, for a register with a hoisted array and a hoisted view:
            // every access of it in the loop takes either as known.)
            if let Some(s) = self.hoisted_array(h, r) {
                a.p_hoist_array(&l, self.env.realm, from, s);
            }
            if let Some(v) = self.hoisted_view_slots(ri, r) {
                a.t_hoist(&l, typed::typed_layout(), self.env.realm, from, v);
            }
        }
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]); // pop r8..r10
        }
        Some(())
    }

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
        self.iout(dst)?;
        let wrap = self.iplan()?.wrap_bits.contains(&self.cur_pc);
        let bail = self.ibail(a);
        let saves = self.walk_saves(dst);
        // The index, sign-extended, rides on the stack across the walk.
        self.index_rax(a, index)?;
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
        a.t_get_i32(
            tl,
            Src::View,
            wrap,
            miss,
            got,
            typed::kind_hint_at(self.cur_pc, obj),
        );
        a.bind(got);
        a.raw(&[0x48, 0x83, 0xc4, 0x08]); // add rsp, 8 (the index)
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]); // pop r8..r10
        }
        self.iput(a, dst, RAX)?;
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

    /// For an element read into a register of the double version, with the
    /// array's element in `rax`: to `miss` on a hole, or, for a numeric
    /// element read (`NumGen::num_read`), on anything but a Number (a hole
    /// is none). Clobbers `rcx` and `r11`.
    fn any_elem_check(&self, a: &mut X64Assembler, miss: crate::jit::Label) {
        if self.num_read(self.cur_pc) {
            a.raw(&[0x48, 0x89, 0xc1]); // mov rcx, rax
            a.movabs_r11(super::super::QNAN as i64);
            a.and_rcx_r11();
            a.cmp_rcx_r11();
        } else {
            a.cmp_rax_const(crate::nanbox::NanBox::hole().to_bits());
        }
        a.jcc(CC_E, miss);
    }

    /// Loads register `r` (a generic op's operand: in the register file,
    /// unless promoted) into `rax`.
    fn iload_operand(&self, a: &mut X64Assembler, r: u16) {
        match self.xmm_of(self.imode, r) {
            Some(x) => a.movq_gpr_x(RAX, x),
            None => a.load_rax(disp(r)),
        }
    }

    /// The inline fast path of `dst = obj[index]` in a speculative region's
    /// integer version (`dst`, `index` int registers), `obj` a hoisted
    /// array ([`crate::jit::numeric::ArraySlots`]): the held array's element
    /// when it is in bounds and an exact int32 (as its `ToInt32` bits when
    /// the read is wrapped, `IntPlan::wrap_bits`: any integral value below
    /// 2^63 in magnitude), into `dst`, past the op's generic code
    /// (`int_skip`). Another array in the register is walked first
    /// ([`Self::iwalk_arr_get`]). Anything else falls into that code, which
    /// runs the op from the start and checks its value (`int_post`).
    pub(in super::super) fn ihoisted_arr_get(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        obj: u16,
        index: u16,
        s: crate::jit::numeric::ArraySlots,
    ) -> Option<()> {
        self.iout(dst)?;
        let pc = self.cur_pc;
        let wrap = self.iplan()?.wrap_bits.contains(&pc);
        let rbp = crate::jit::call_seq::r::RBP;
        let slow = a.new_label();
        let walk = a.new_label();
        let done = self.int_skip.take().unwrap_or_else(|| a.new_label());
        // (A receiver known to be the held array whenever there is one: the
        // bounds check alone, the count `0` when the entry found none.)
        let known = self.array_known_fast(pc, obj) || self.row_known(pc, obj);
        if !known {
            self.iload_operand(a, obj);
            a.g_cmp_mem(RAX, rbp, s.held);
            a.jne(walk);
        }
        self.index_rax(a, index)?;
        a.g_cmp_mem(RAX, rbp, s.len);
        a.jae(slow); // past the count, or negative
        a.g_load(R11, rbp, s.base);
        a.mov_rax_at_r11_rax8();
        if wrap {
            a.movq_x_gpr(0, RAX);
            a.cvttsd2si_gpr_x(RCX, 0);
            a.cvtsi2sd_x_gpr(0, RCX);
            a.movq_gpr_x(R11, 0);
            a.cmp_rax_r11();
        } else {
            // (A hole is no Number: not one either.)
            a.exact_i32_check(RCX);
        }
        a.jcc(CC_NE, slow);
        self.iput(a, dst, RCX)?;
        a.jmp(done);
        self.skip_cold(a);
        if !known {
            a.bind(walk);
            // (A wrapped read's bits: the generic code's.)
            if !wrap {
                self.iwalk_arr_get(a, dst, obj, index, true, done, s)?;
            }
        }
        a.bind(slow);
        self.int_skip = Some(done);
        Some(())
    }

    /// For a hoisted-array element read whose receiver is not the held
    /// array (a row of a grid, `grid[y][x]`, another one each time): the
    /// element of the plain array the register holds, walked as the
    /// generic code would but with the index from its int register and no
    /// int GPR saves for calls (only the walk's clobbered ones pushed), into
    /// `dst` (`int`: an int register, the element an exact int32; else its
    /// double-version location, the element any value but a hole), then to
    /// `done`. Anything else falls through, into the op's generic code,
    /// which runs it from the start.
    ///
    /// The array walked becomes the register's hoisted one (`s`): what the
    /// walk found about it holds to the back-edge as the entry check's
    /// would (the fast version changes no array's element count or place),
    /// so the next reads of that row take the inline path. (Its stores of
    /// handles still go the generic way: `wb` says the barrier is needed.)
    #[allow(clippy::too_many_arguments)]
    fn iwalk_arr_get(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        obj: u16,
        index: u16,
        int: bool,
        done: crate::jit::Label,
        s: crate::jit::numeric::ArraySlots,
    ) -> Option<()> {
        let saves = self.walk_saves(dst);
        self.index_rax(a, index)?;
        for &g in &saves {
            a.raw(&[0x41, 0x50 + (g & 7)]); // push r8..r10
        }
        a.push_rax();
        let miss = a.new_label();
        let l = self.env.heap;
        a.p_arena(self.env.realm, &l.props);
        self.iload_operand(a, obj);
        a.p_is_handle(RCX, miss);
        a.p_slot_rooted(&l, RCX, miss);
        a.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        a.jne(miss);
        a.p_array_plain(&l, self.env.realm, false, miss);
        if row_memo_enabled() {
            // (The hoisted array's slots, as `p_hoist_array` fills them.)
            let rbp = crate::jit::call_seq::r::RBP;
            a.p_arr_len_ptr(&l); // r8 = the count, r11 = the first element
            a.g_store(rbp, s.len, 8);
            a.g_store(rbp, s.base, R11);
            a.cvtsi2sd_x_gpr(0, 8);
            a.movq_gpr_x(RCX, 0);
            a.g_store(rbp, s.lend, RCX);
            a.raw(&[0x31, 0xc9]); // xor ecx, ecx
            a.g_store(rbp, s.wb, RCX);
            self.iload_operand(a, obj);
            a.g_store(rbp, s.held, RAX);
            a.raw(&[0x48, 0x8b, 0x04, 0x24]); // mov rax, [rsp]
        } else {
            a.raw(&[0x48, 0x8b, 0x04, 0x24]); // mov rax, [rsp]
            a.p_arr_len_ptr(&l);
        }
        a.cmp_rax_r8();
        a.jae(miss); // past the count, or negative
        a.mov_rax_at_r11_rax8();
        if int {
            // (A hole is no Number: not one either.)
            a.exact_i32_check(RCX);
            a.jcc(CC_NE, miss);
        } else {
            self.any_elem_check(a, miss);
        }
        a.raw(&[0x48, 0x83, 0xc4, 0x08]); // add rsp, 8 (the index)
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]); // pop r8..r10
        }
        if int {
            self.iput(a, dst, RCX)?;
        } else {
            match self.xmm_of(self.imode, dst) {
                Some(x) => a.movq_x_gpr(x, RAX),
                None => a.store_rax(disp(dst)),
            }
        }
        a.jmp(done);
        a.bind(miss);
        a.raw(&[0x48, 0x83, 0xc4, 0x08]); // add rsp, 8
        for &g in saves.iter().rev() {
            a.raw(&[0x41, 0x58 + (g & 7)]);
        }
        Some(())
    }

    /// The inline fast path of `dst = obj[index]` in a speculative region's
    /// integer version, `index` an int register and `dst` none, `obj` a
    /// hoisted array: the held array's element when in bounds and no hole,
    /// into `dst`'s double-version location, past the op's generic code
    /// (`int_skip`); another array in the register is walked first
    /// ([`Self::iwalk_arr_get`]); anything else falls into that code.
    pub(in super::super) fn ihoisted_arr_get_any(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        obj: u16,
        index: u16,
        s: crate::jit::numeric::ArraySlots,
    ) -> Option<()> {
        let pc = self.cur_pc;
        self.ik(index)?;
        let rbp = crate::jit::call_seq::r::RBP;
        let slow = a.new_label();
        let walk = a.new_label();
        let done = self.int_skip.take().unwrap_or_else(|| a.new_label());
        let known = self.array_known_fast(pc, obj) || self.row_known(pc, obj);
        if !known {
            self.iload_operand(a, obj);
            a.g_cmp_mem(RAX, rbp, s.held);
            a.jne(walk);
        }
        self.index_rax(a, index)?;
        a.g_cmp_mem(RAX, rbp, s.len);
        a.jae(slow); // past the count, or negative
        a.g_load(R11, rbp, s.base);
        a.mov_rax_at_r11_rax8();
        self.any_elem_check(a, slow);
        match self.xmm_of(self.imode, dst) {
            Some(x) => a.movq_x_gpr(x, RAX),
            None => a.store_rax(disp(dst)),
        }
        a.jmp(done);
        self.skip_cold(a);
        if !known {
            a.bind(walk);
            self.iwalk_arr_get(a, dst, obj, index, false, done, s)?;
        }
        a.bind(slow);
        self.int_skip = Some(done);
        Some(())
    }

    /// The inline fast path of `obj[index] = val` in a speculative region's
    /// integer version (`index` an int register), `obj` a hoisted array:
    /// written in place, past the op's generic code (`int_skip`), when the
    /// receiver is the held array, the index in bounds over no hole and the
    /// value needs nothing of the write barrier (an int register's Number,
    /// a non-handle, or any value while the entry found the array sparing
    /// every store that work), as the double version's hoisted store does.
    /// Anything else falls into that code. Touches only `rax`, `rdx`,
    /// `r11` and `xmm0`.
    pub(in super::super) fn ihoisted_arr_set(
        &mut self,
        a: &mut X64Assembler,
        obj: u16,
        index: u16,
        val: u16,
        s: crate::jit::numeric::ArraySlots,
    ) -> Option<()> {
        use crate::jit::call_seq::{cc, r::RBP};
        let pc = self.cur_pc;
        self.ik(index)?;
        let val_k = if self.is_int(val) {
            Some(self.ik(val)?)
        } else {
            None
        };
        let slow = a.new_label();
        let done = self.int_skip.take().unwrap_or_else(|| a.new_label());
        if !self.array_known_fast(pc, obj) && !self.row_known(pc, obj) {
            self.iload_operand(a, obj);
            a.g_cmp_mem(RAX, RBP, s.held);
            a.jne(slow);
        }
        if val_k.is_none() && self.xmm_of(self.imode, val).is_none() {
            // (A promoted value is a Number.)
            let plain = a.new_label();
            a.load_rax(disp(val));
            a.p_cmp_handle_floor(RAX);
            a.g_jcc(cc::B, plain);
            a.cmp_frame_imm8(s.wb, 0);
            a.jcc(CC_E, slow);
            a.bind(plain);
        }
        self.index_rax(a, index)?;
        a.g_cmp_mem(RAX, RBP, s.len);
        a.jae(slow); // past the count, or negative
        a.g_load(R11, RBP, s.base);
        a.movabs_argreg(2, crate::nanbox::NanBox::hole().to_bits() as i64);
        a.cmp_at_r11_rax8_rdx();
        a.jcc(CC_E, slow);
        match val_k {
            Some(IK::Gpr(g)) => {
                a.cvtsi2sd32(0, g);
                a.movq_gpr_x(RDX, 0);
            }
            Some(IK::Imm(v)) => a.movabs_argreg(2, f64::from(v).to_bits() as i64),
            Some(IK::Mem(d)) => {
                a.cvtsi2sd32_mem(0, d);
                a.movq_gpr_x(RDX, 0);
            }
            None => match self.xmm_of(self.imode, val) {
                Some(x) => a.movq_gpr_x(RDX, x),
                None => a.load_rdx(disp(val)),
            },
        }
        a.mov_at_r11_rax8_rdx(); // elems[idx] = val
        a.jmp(done);
        self.skip_cold(a);
        a.bind(slow);
        self.int_skip = Some(done);
        Some(())
    }

    /// The inline fast path of `dst = arr.length` in a speculative region's
    /// integer version, `arr` a hoisted array: the held array's count when
    /// at most `i32::MAX`, into `dst`, past the op's generic code; anything
    /// else falls into that code. Touches only `rax`.
    pub(in super::super) fn ihoisted_arr_len(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        arr: u16,
        s: crate::jit::numeric::ArraySlots,
    ) -> Option<()> {
        self.iout(dst)?;
        let pc = self.cur_pc;
        let rbp = crate::jit::call_seq::r::RBP;
        let slow = a.new_label();
        let done = self.int_skip.take().unwrap_or_else(|| a.new_label());
        if self.array_known_fast(pc, arr) || self.row_known(pc, arr) {
            a.cmp_frame_imm8(s.held, crate::jit::RECV_CACHE_EMPTY as i8);
            a.jcc(CC_E, slow);
        } else {
            self.iload_operand(a, arr);
            a.g_cmp_mem(RAX, rbp, s.held);
            a.jne(slow);
        }
        a.g_load(RAX, rbp, s.len);
        // cmp rax, 0x7fffffff; ja slow (a count past i32::MAX)
        a.raw(&[0x48, 0x3d, 0xff, 0xff, 0xff, 0x7f]);
        a.jcc(super::super::CC_A, slow);
        self.iput(a, dst, RAX)?;
        a.jmp(done);
        self.skip_cold(a);
        a.bind(slow);
        self.int_skip = Some(done);
        Some(())
    }

    /// `dst = obj[index]` in a speculative region's integer version, `obj`
    /// a hoisted view ([`typed::ViewSlots`], `NumGen::hoisted_view`): no
    /// walk, and no register touched but `rax` / `r11`. Anything else (not
    /// that view, out of bounds, a float kind) bails before anything is
    /// done.
    pub(super) fn ihoisted_get(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        obj: u16,
        index: u16,
        v: typed::ViewSlots,
    ) -> Option<()> {
        self.iout(dst)?;
        let wrap = self.iplan()?.wrap_bits.contains(&self.cur_pc);
        let bail = self.ibail(a);
        // (A float kind's element, say: the recompilation has `dst` a
        // double.)
        self.demote_last(dst);
        self.index_rax(a, index)?;
        // (Straight into `dst`'s GPR, which a miss leaves as it was or
        // holding the index: it may be the index's.)
        let out = match self.ik(dst)? {
            IK::Gpr(g) => g,
            _ => RAX,
        };
        a.t_hoisted_get_i32(v, disp(obj), wrap, bail, out);
        self.iput(a, dst, out)?;
        Some(())
    }

    /// `dst = obj.length` in a speculative region's integer version, `obj`
    /// a hoisted view: its length (at most `i32::MAX`), else a bail.
    pub(super) fn ihoisted_len(
        &mut self,
        a: &mut X64Assembler,
        dst: u16,
        obj: u16,
        v: typed::ViewSlots,
    ) -> Option<()> {
        self.iout(dst)?;
        let bail = self.ibail(a);
        a.t_hoisted_len(v, disp(obj), bail);
        // cmp rax, 0x7fffffff; ja bail (an unsigned length past i32::MAX)
        a.raw(&[0x48, 0x3d, 0xff, 0xff, 0xff, 0x7f]);
        a.jcc(super::super::CC_A, bail);
        self.iput(a, dst, RAX)?;
        Some(())
    }

    /// `dst = arr.length` in the integer version (`dst` an int register):
    /// a plain array's or a typed-array view's, at most `i32::MAX`.
    pub(super) fn iarray_len(&mut self, a: &mut X64Assembler, dst: u16, arr: u16) -> Option<()> {
        self.iout(dst)?;
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
        self.iput(a, dst, RAX)?;
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
