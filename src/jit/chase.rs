//! Pointer-chasing loops: `while (q.link) q = q.link;`, walking a linked
//! structure to its end. Each hop of the plain code reads the field, tests the
//! value, walks its handle, checks its layout, records it in the receiver
//! cache, and moves it through two temporaries: about 28 instructions.
//!
//! [`plan`] finds the loop; the emitter puts an **accelerator** in front of
//! its first op. While the next object is a plain object of the layout the
//! receiver cache proves (so truthy, and the field read on it is the same
//! read), the accelerator steps `q` to it in about 12 instructions, keeping
//! the handle and the heap slot in registers, and then falls into the ordinary
//! loop, which does the rest: the last hop, the exit, any other shape.
//!
//! Running the accelerator is running iterations of the loop. What those
//! iterations would have changed is `q`, the receiver cache's way (updated on
//! the way out) and two temporaries: the condition's, which the ordinary loop
//! writes first thing, and the moved value's, which [`plan`] requires no later
//! op to read before writing. Nothing else can observe them: the hops run no
//! Rust code, no getter (the layout has a data property there) and allocate
//! nothing. They do not count down the loop's polls; instead at most
//! [`MAX_HOPS`] of them run between two ordinary iterations, which do, so an
//! interrupt still reaches a walk around a cycle, a bounded factor later.

//!
//! Where the field is an inline slot (the usual case), the hops are
//! [`X64Assembler::p_chase_hops`]: unrolled by two, the current object and
//! the next one alternating between two register pairs, so a hop moves no
//! register and only every other one counts down, and a `null`, `undefined`
//! or `false` next value ends the loop right there (the ordinary loop would
//! read that same value into the condition and branch out on it), writing the
//! condition temporary as the ordinary read would.
//!
//! **Stride prediction.** A plain hop is one dependency chain: the field's
//! load, then the handle walk (two `lea`s), then the next load at the
//! address it made, about six cycles where the load alone is four. But a
//! structure built by a loop usually sits at a constant distance between
//! consecutive objects (the allocator hands out neighbouring slots), so the
//! hops run first in a **predicted** form: the next slot's address is the
//! current one plus the distance the first hop measured, a one-cycle add,
//! and the walk of the handle actually read is computed beside it and only
//! *compared* with the prediction. Every check of the plain hop is still
//! made, on the walked address, which the compare proves equal to the
//! predicted one; what changes is that the next load no longer waits for the
//! walk (the processor runs ahead past the compare's branch), so consecutive
//! hops overlap: about half the time per hop on a structure laid out in
//! order. The first two hops of a run are plain (they measure the distance,
//! and a short walk is over before a prediction could pay); the first
//! predicted hop whose address differs from the prediction leaves for the
//! plain hops, which finish the run from that object: a scattered structure
//! pays one mispredicted branch per run.

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};
use crate::nbvm::Op;
use alloc::vec::Vec;

/// The most hops between two ordinary iterations of the loop.
pub(super) const MAX_HOPS: u64 = 256;

/// The frame operands of [`X64Assembler::p_chase_hops`].
pub(super) struct ChaseFrame {
    /// The loop's object register (`rbx`-relative).
    pub q: i32,
    /// The condition temporary (`rbx`-relative).
    pub t: i32,
    /// The receiver-cache way's key and cell slots (`rbp`-relative).
    pub key: i32,
    /// See `key`.
    pub cell: i32,
}

impl X64Assembler {
    /// Whether [`p_chase_hops`](Self::p_chase_hops) can walk field slot
    /// `slot` under layout `l`.
    pub(super) fn chase_hops_ok(&self, l: &JitLayout, slot: u32) -> bool {
        self.pinned()
            && l.slot_stride == 72
            && Self::not_exotic_one_test(l)
            && Self::inline_slot_disp(l, slot).is_some()
    }

    /// The hops, entered with `rax` = `q` (a hit of the way), `rcx` = its heap
    /// slot, of layout `shape` whose data property at inline slot `slot` is
    /// the field. Leaves for `normal` with `q` and the way set to the last
    /// object reached; with `falsy`, when the field's value there is
    /// `undefined`, `null` or `false`, goes to `falsy` instead, having also
    /// stored that value in the condition temporary. Clobbers `rax`, `rcx`,
    /// `rdx`, `rsi`, `rdi`, `r9`, `r10`, `r11` (and `r8` in audit mode).
    pub(super) fn p_chase_hops(
        &mut self,
        l: &JitLayout,
        (shape, slot): (u64, u32),
        f: &ChaseFrame,
        falsy: Option<Label>,
        normal: Label,
    ) {
        let Some(d) = Self::inline_slot_disp(l, slot) else {
            return;
        };
        // Per pair: (current handle, its slot) → (next handle, its slot).
        let pairs = [
            ((r::RAX, r::RCX), (r::RSI, r::RDI)),
            ((r::RSI, r::RDI), (r::RAX, r::RCX)),
        ];
        let leave = [self.new_label(), self.new_label()];
        let value = [self.new_label(), self.new_label()];
        // The plain hops' entries, with the current object in pair `i`.
        let plain = [self.new_label(), self.new_label()];
        self.g_movabs(r::RDX, MAX_HOPS / 2);
        // The first two hops are plain ones (a short walk is over before
        // any prediction could pay); the distance the second went is the
        // prediction (`r10`).
        for (i, &pair) in pairs.iter().enumerate() {
            self.p_chase_plain_hop(l, (shape, d), pair, (value[i], leave[i]));
        }
        self.raw(&[0xff, 0xca]); // dec edx
        self.je(leave[0]);
        self.g_mov(r::R10, r::RCX);
        self.raw(&[0x49, 0x29, 0xfa]); // sub r10, rdi
        // The predicted hops (the current object in pair `i`).
        let top = self.new_label();
        self.bind(top);
        for (i, &((_, cs), (nh, ns))) in pairs.iter().enumerate() {
            self.g_load(nh, cs, d);
            p_lea_sum(self, ns, cs, r::R10);
            self.p_cmp_handle_floor(nh);
            self.g_jcc(cc::B, value[i]);
            p_walk_r9(self, nh, leave[i]);
            // cmp ns, r9
            self.raw(&[0x4c, 0x39, 0xc8 | ns]);
            self.jne(plain[i]);
            if super::handle::handle_audit() {
                self.p_audit_live_of(l, nh, ns);
            }
            self.p_object_not_exotic_at(l, ns, leave[i]);
            self.p_cmp_shape(ns, l.off_shape, shape, r::R11);
            self.jne(leave[i]);
        }
        self.raw(&[0xff, 0xca]); // dec edx
        self.jne(top);
        self.jmp(leave[0]);
        // The plain hops (entered with at least one count left).
        self.bind(plain[0]);
        for (i, &pair) in pairs.iter().enumerate() {
            if i == 1 {
                self.bind(plain[1]);
            }
            self.p_chase_plain_hop(l, (shape, d), pair, (value[i], leave[i]));
        }
        self.raw(&[0xff, 0xca]); // dec edx
        self.jne(plain[0]);
        // (The current object is in the first pair again.)
        self.jmp(leave[0]);
        for (i, &((ch, cs), (nh, _))) in pairs.iter().enumerate() {
            let exit = falsy.map(|_| self.new_label());
            self.bind(value[i]);
            if let Some(exit) = exit {
                // `undefined`, `null`, `false`: the three words from
                // `undefined`'s up.
                const _: () = assert!(
                    crate::nanbox::NanBox::null().to_bits()
                        == crate::nanbox::NanBox::undefined().to_bits() + 1
                        && crate::nanbox::NanBox::boolean(false).to_bits()
                            == crate::nanbox::NanBox::undefined().to_bits() + 2
                );
                self.g_mov(r::R11, nh);
                self.p_sub_const(r::R11, crate::nanbox::NanBox::undefined().to_bits());
                self.g_cmp_imm(r::R11, 2);
                self.g_jcc(cc::BE, exit);
            }
            self.bind(leave[i]);
            self.g_store(r::RBX, f.q, ch);
            self.g_store(r::RBP, f.key, ch);
            self.g_store(r::RBP, f.cell, cs);
            self.jmp(normal);
            if let (Some(falsy), Some(exit)) = (falsy, exit) {
                self.bind(exit);
                self.g_store(r::RBX, f.t, nh);
                self.g_store(r::RBX, f.q, ch);
                self.g_store(r::RBP, f.key, ch);
                self.g_store(r::RBP, f.cell, cs);
                self.jmp(falsy);
            }
        }
    }
}

impl X64Assembler {
    /// One plain hop of [`p_chase_hops`](X64Assembler::p_chase_hops), from
    /// the current pair `(_, cs)` to the next `(nh, ns)`: the field at `d`
    /// into `nh`, to `value` unless a handle, its walk into `ns`, to `leave`
    /// unless an in-bounds plain object of layout `shape`.
    fn p_chase_plain_hop(
        &mut self,
        l: &JitLayout,
        (shape, d): (u64, i32),
        ((_, cs), (nh, ns)): ((u8, u8), (u8, u8)),
        (value, leave): (Label, Label),
    ) {
        self.g_load(nh, cs, d);
        self.p_cmp_handle_floor(nh);
        self.g_jcc(cc::B, value);
        self.p_slot_rooted_rr(l, nh, ns, leave);
        self.p_object_not_exotic_at(l, ns, leave);
        self.p_cmp_shape(ns, l.off_shape, shape, r::R11);
        self.jne(leave);
    }
}

/// `lea dst, [base + index]` (`dst`, `base` among the first eight
/// registers, `base` not `rbp`; `index` any but `rsp`).
fn p_lea_sum(a: &mut X64Assembler, dst: u8, base: u8, index: u8) {
    debug_assert!(dst < 8 && base < 8 && base != 5 && index != 4);
    a.raw(&[
        0x48 | (((index >> 3) & 1) << 1),
        0x8d,
        (dst << 3) | 4,
        ((index & 7) << 3) | base,
    ]);
}

/// The rooted handle walk (`X64Assembler::p_slot_rooted_rr`) of the handle
/// in `h` (one of the first eight registers) into `r9`, pinned code, a
/// 72-byte stride: to `miss` if its index is out of bounds.
fn p_walk_r9(a: &mut X64Assembler, h: u8, miss: Label) {
    debug_assert!(h < 8);
    // mov r9d, h32; cmp r9, r13
    a.raw(&[0x41, 0x89, 0xc1 | (h << 3), 0x4d, 0x39, 0xe9]);
    a.jae(miss);
    // lea r9, [r9 + r9*8]; lea r9, [r12 + r9*8]
    a.raw(&[0x4f, 0x8d, 0x0c, 0xc9, 0x4f, 0x8d, 0x0c, 0xcc]);
}

/// Per op: whether it is the first op of such a loop.
#[must_use]
pub(super) fn plan(ops: &[Op]) -> Vec<bool> {
    let mut at = alloc::vec![false; ops.len()];
    // (A handler could read the temporaries at a throw: nothing here throws,
    // but the liveness below follows no catch edge, so such a body has none.)
    if ops.iter().any(|o| matches!(o, Op::PushHandler { .. })) {
        return at;
    }
    let n = ops.len().saturating_sub(4);
    for (l, x) in at.iter_mut().enumerate().take(n) {
        *x = is_chase(ops, l);
    }
    at
}

/// Whether `ops[l..=l + 4]` is exactly the loop, entered only at `l`, its
/// moved value dead after it.
fn is_chase(ops: &[Op], l: usize) -> bool {
    let (
        Op::GetProp {
            dst: t,
            obj: q,
            key,
        },
        Op::JumpIfFalse { cond, target: exit },
        Op::GetProp {
            dst: u,
            obj: q2,
            key: key2,
        },
        Op::Move { dst: q3, src: u2 },
        Op::Jump { target: back },
    ) = (&ops[l], &ops[l + 1], &ops[l + 2], &ops[l + 3], &ops[l + 4])
    else {
        return false;
    };
    if cond != t || q2 != q || key2 != key || q3 != q || u2 != u || *back != l {
        return false;
    }
    if t == q || u == q || t == u || *exit <= l + 4 {
        return false;
    }
    // Entered only at its first op.
    let inner = l + 1..=l + 4;
    let entered = ops.iter().any(|o| match o {
        Op::Jump { target } | Op::JumpIfFalse { target, .. } => inner.contains(target),
        Op::InlineGuard { call, .. } => inner.contains(call),
        _ => false,
    });
    if entered {
        return false;
    }
    !read_before_written(ops, *exit, *u)
}

/// Whether some path from op `from` reads register `r` before writing it.
fn read_before_written(ops: &[Op], from: usize, r: u16) -> bool {
    let mut seen = alloc::vec![false; ops.len() + 1];
    let mut todo = alloc::vec![from];
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    while let Some(pc) = todo.pop() {
        if pc >= ops.len() || core::mem::replace(&mut seen[pc], true) {
            continue;
        }
        let op = &ops[pc];
        if crate::bytecode::op_mentions_reg(op, r) {
            super::numeric::def_use(op, &mut defs, &mut uses);
            if !defs.contains(&r) || uses.contains(&r) {
                return true;
            }
            continue;
        }
        match op {
            Op::Jump { target } => todo.push(*target),
            Op::JumpIfFalse { target, .. } => {
                todo.push(*target);
                todo.push(pc + 1);
            }
            Op::Return { .. }
            | Op::Throw { .. }
            | Op::ThrowTypeError { .. }
            | Op::ThrowReferenceError { .. } => {}
            Op::InlineGuard { call, .. } => {
                todo.push(*call);
                todo.push(pc + 1);
            }
            _ => todo.push(pc + 1),
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn get(dst: u16, obj: u16) -> Op {
        Op::GetProp {
            dst,
            obj,
            key: "link".to_string(),
        }
    }

    fn the_loop() -> Vec<Op> {
        alloc::vec![
            get(1, 0),
            Op::JumpIfFalse { cond: 1, target: 5 },
            get(2, 0),
            Op::Move { dst: 0, src: 2 },
            Op::Jump { target: 0 },
            Op::Return { src: 0 },
        ]
    }

    #[test]
    fn finds_the_loop() {
        assert!(plan(&the_loop())[0]);
    }

    #[test]
    fn refuses_a_moved_value_read_afterwards() {
        let mut ops = the_loop();
        ops[5] = Op::Return { src: 2 };
        assert!(!plan(&ops)[0]);
    }

    #[test]
    fn refuses_a_branch_into_the_loop() {
        let mut ops = the_loop();
        ops.push(Op::Jump { target: 2 });
        assert!(!plan(&ops)[0]);
    }

    #[test]
    fn refuses_another_key_or_a_handler() {
        let mut ops = the_loop();
        ops[2] = Op::GetProp {
            dst: 2,
            obj: 0,
            key: "next".to_string(),
        };
        assert!(!plan(&ops)[0]);
        let mut ops = the_loop();
        ops.push(Op::PushHandler { target: 0, reg: 3 });
        assert!(!plan(&ops)[0]);
    }
}
