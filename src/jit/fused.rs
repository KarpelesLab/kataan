//! **Fused read-modify-write** of an own data property: `o.k++` / `o.k--`
//! (and their prefix forms), which the bytecode spells as four ops,
//!
//! ```text
//! GetProp   r1 = o.k
//! ToNumeric r2 = r1
//! Step      r3 = r2 ± 1
//! SetProp   o.k = r3
//! ```
//!
//! each with its own receiver walk or way check, shape compare, type guard
//! and register-file round trip. When both property sites compiled in the
//! same layout and slot, the `GetProp`'s inline hit (its receiver walked, its
//! layout compared, the value loaded) goes straight on in one piece: one
//! Number test on the value, the step in SSE, the store guards (the flags
//! byte) and the in-place store at the same slot, then past the `SetProp`.
//!
//! Nothing runs between the read and the store (no call, no other op), so
//! the object is still the one the `GetProp` walked, in the layout it
//! compared: the `SetProp`'s own walk and shape compare would succeed and
//! are skipped. A Number needs no write barrier, and keeps a Number-only
//! slot's record (`fieldrep`) true. The three registers are written as the
//! ops would write them. Wherever the fused path cannot go on (a value that
//! is not a Number, an object whose writes need the helper), it jumps into
//! the ops' own code at the op that would have taken its slow path, with
//! every register and way exactly as the ops before it leave them on their
//! fast paths; so those ops decide what happens, as they always did.

use super::call_seq::r;
use super::{GenericOp, JitLayout, Label, X64Assembler};

/// A fusable `o.k++` at a `GetProp`: the registers its four ops write and
/// whether it steps down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FusedStep {
    /// The `GetProp`'s, the `ToNumeric`'s and the `Step`'s destinations.
    pub(super) dst: [u16; 3],
    /// `--` rather than `++`.
    pub(super) dec: bool,
}

/// The `o.k++` at `ops[i..i + 4]`, if those are the four ops on one
/// receiver (that none of them overwrites) and `same_slot(get_site,
/// set_site)` says both sites compiled in the same layout and slot.
pub(super) fn fused_step_at(
    ops: &[GenericOp],
    i: usize,
    same_slot: impl Fn(u16, u16) -> bool,
) -> Option<FusedStep> {
    let (
        GenericOp::GetProp {
            dst: d1,
            obj,
            site: gs,
            ..
        },
        GenericOp::ToNumeric {
            dst: d2, src: s2, ..
        },
        GenericOp::Step {
            dst: d3,
            src: s3,
            dec,
            ..
        },
        GenericOp::SetProp {
            obj: o2,
            val,
            site: ss,
            ..
        },
    ) = (
        *ops.get(i)?,
        *ops.get(i + 1)?,
        *ops.get(i + 2)?,
        *ops.get(i + 3)?,
    )
    else {
        return None;
    };
    let chained = s2 == d1 && s3 == d2 && val == d3 && o2 == obj;
    let kept = obj != d1 && obj != d2 && obj != d3;
    (chained && kept && same_slot(gs, ss)).then_some(FusedStep {
        dst: [d1, d2, d3],
        dec,
    })
}

impl X64Assembler {
    /// The fused tail of a `GetProp`'s inline hit (see the module docs):
    /// `rcx` = the receiver's heap slot, its layout compared, `rax` = the
    /// value of slot `slot`, stored to nothing yet. `disp` are the three
    /// destination registers' displacements; `step_op` is where the
    /// `ToNumeric` begins, `store_op` the `SetProp`, `past` the op after
    /// it. Clobbers `rax`, `rdx`, `r11`, `xmm0`, `xmm1`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn p_fused_step(
        &mut self,
        l: &JitLayout,
        slot: u32,
        disp: [i32; 3],
        dec: bool,
        step_op: Label,
        store_op: Label,
        past: Label,
    ) {
        const QNAN: u64 = 0x7ffc_0000_0000_0000;
        self.store_rax(disp[0]);
        // Not a Number: the `ToNumeric` takes it from here.
        self.g_mov(r::RDX, r::RAX);
        self.g_movabs(r::R11, QNAN);
        self.code.extend_from_slice(&[0x4c, 0x21, 0xda]); // and rdx, r11
        self.code.extend_from_slice(&[0x4c, 0x39, 0xda]); // cmp rdx, r11
        self.je(step_op);
        self.store_rax(disp[1]);
        self.code.extend_from_slice(&[0x66, 0x48, 0x0f, 0x6e, 0xc0]); // movq xmm0, rax
        self.g_movabs(r::RAX, 1.0f64.to_bits());
        self.cvt_rax_bits_to_xmm1();
        if dec {
            self.subsd_xmm0_xmm1();
        } else {
            self.addsd_xmm0_xmm1();
        }
        // (`NaN ± 1` is that NaN, the canonical one.)
        self.code.extend_from_slice(&[0x66, 0x48, 0x0f, 0x7e, 0xc0]); // movq rax, xmm0
        self.store_rax(disp[2]);
        // The `SetProp`'s store guards; its walk and layout compare are the
        // `GetProp`'s, with nothing run since.
        self.p_test_byte_imm(
            r::RCX,
            l.props.off_jit_flags,
            super::props::FLAGS_NO_INLINE_STORE,
        );
        self.jne(store_op);
        self.g_mov(r::RDX, r::RAX);
        self.p_obj_fixed_slot_store(l, slot);
        self.jmp(past);
    }
}
