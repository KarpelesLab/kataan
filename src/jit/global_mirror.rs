//! Emission of the global-object mirror of a global `var` write (see
//! `crate::nbvm::global_mirror`): after compiled code stores the value into the
//! binding, it stores it into the global object's property too, at the address
//! `jit_helper_global_ref` validated with the binding's (left in the op's
//! `MirrorCache`, `0` when the global object does not own the name).

use super::call_seq::{cc, r};
use super::{Label, X64Assembler};
use crate::nbvm::Op;

/// For a `StoreGlobal` whose compiled code mirrors the write
/// (`MirrorCache::on`, decided when the body was compiled): the address of the
/// cell holding its validated mirror target.
pub(super) fn mirror_cell(op: &Op) -> Option<u64> {
    match op {
        Op::StoreGlobal { cache, .. } if cache.jit.on() => Some(cache.jit.ptr_addr() as u64),
        _ => None,
    }
}

/// Plans a body's own bytecode copy (`crate::nbvm::global_mirror::plan_body`).
pub(super) fn plan(bytecode: &[Op]) {
    crate::nbvm::plan_global_mirrors(bytecode);
}

impl X64Assembler {
    /// The mirror of a global write of the value in GPR `val`, just stored
    /// into its binding: `ptr` = the mirror target (from `cell`); when there is
    /// one, a non-handle value is stored there, and a heap handle (which needs
    /// the write barrier) goes to `handle` with `ptr` and `val` intact. Falls
    /// through having done the mirror otherwise. Clobbers `ptr`.
    pub(super) fn gm_store_gpr(&mut self, cell: u64, ptr: u8, val: u8, handle: Label) {
        let done = self.new_label();
        self.g_movabs(ptr, cell);
        self.g_load(ptr, ptr, 0);
        self.g_test(ptr);
        self.g_jcc(cc::E, done);
        self.p_cmp_handle_floor(val);
        self.g_jcc(cc::AE, handle);
        self.g_store(ptr, 0, val);
        self.bind(done);
    }

    /// The mirror of a global write of the Number in `xmm` (one known not to
    /// be a heap handle: an int32 converted, say),
    /// just stored into its binding. Clobbers `rcx`.
    pub(super) fn gm_store_xmm(&mut self, cell: u64, xmm: u8) {
        let done = self.new_label();
        self.g_movabs(r::RCX, cell);
        self.g_load(r::RCX, r::RCX, 0);
        self.g_test(r::RCX);
        self.g_jcc(cc::E, done);
        // movsd [rcx], xmm
        self.code.push(0xf2);
        if xmm >= 8 {
            self.code.push(0x44);
        }
        self.code
            .extend_from_slice(&[0x0f, 0x11, ((xmm & 7) << 3) | 1]);
        self.bind(done);
    }
}
