//! A compiled `for (k in o)` loop's key reads with no call: its `EnumKeys`,
//! its per-key liveness check `k in o` and its body's `o[k]`.
//!
//! The bytecode of a hosted `for-in` (`crate::nbvm`'s `Stmt::ForIn`) is
//! `obj = <o>; arr = EnumKeys(obj); … cur = arr[i]; live = cur in obj; …`,
//! and a binding `k` of the key is a copy of `cur`. A memoized key list
//! (`crate::realm::forin_cache`) gives the loop the list's shared array,
//! whose key table records each key's slot in the receiver's layout. The
//! loop's `EnumKeys` (`jit_helper_enum_keys`) leaves that table in the
//! loop's [`ForInSite`] (or none), and then, with the slot table's address
//! in the site:
//!
//! - `cur = arr[i]` is the table's key `i` when `arr` is the table's array;
//! - `cur in obj` is `true` when `arr` is the table's array, `obj` (a shaped
//!   ordinary object without the exotic flag) still has the table's layout
//!   and key `i` of the table is `cur` with a slot of its own (and the
//!   `JumpIfFalse` on it is then nothing);
//! - `r[k]` for a key register `k` (`cur` or a copy) is the slot's value when
//!   the receiver `r` is `obj` itself, with no accessors (as
//!   `Realm::for_in_key_slot` reads it), and no String has an auxiliary
//!   object (`Realm::str_special`).
//!
//! Anything else is the helper's. Why this is exact: a table holds its array
//! as a GC root while the site holds the table (`KeyTable`), so while
//! `arr` is the table's array the key Strings in it are the table's, and a
//! key register equal to key `i`'s bits is that very String. `obj` is
//! written by nothing after the loop's `EnumKeys` (checked here), which
//! found it a plain shaped, non-host-exotic object when it gave the loop
//! the table's array; whether an object is host-exotic never changes, and
//! everything else is checked per read.

use super::call_seq::{cc, r};
use super::{GenericOp, JitLayout, Label, X64Assembler};
use crate::nbvm::Op;
use crate::realm::{ForInSite, KeyTable};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

/// One compiled `for-in` loop: its site's index and its registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Loop {
    /// The loop's [`ForInSite`] (an index into [`Plan::sites`]).
    pub(super) site: usize,
    /// The key array's register.
    pub(super) arr: u16,
    /// The key index's register.
    pub(super) index: u16,
    /// The liveness object's register.
    pub(super) obj: u16,
}

/// A compiled `for-in` loop's **memo** (four frame slots): the receiver
/// its liveness check last walked and the key it last found there.
///
/// `obj` (a receiver-cache key: emptied in the prologue and by every call,
/// see `X64Assembler::clear_recv_cache`) holds the bits of an object the
/// check walked to the heap slot `cell` and found a shaped ordinary object
/// without the host-exotic flag and without accessors, with no String
/// having an auxiliary object (`fi_plain_read`); `key` holds the bits of a
/// key the check then found at slot `slot` of the object's layout (the
/// walk empties it). Until Rust code runs (behind a call, which empties
/// `obj`) all of that stays true: only Rust code can move or free a heap
/// slot, make an object a dictionary, host-exotic or give it an accessor,
/// delete or redefine a property, or set `str_special`; compiled code only
/// adds properties after the existing ones (which moves no slot) and
/// writes values. So with `obj` equal to a receiver's bits, `cell` is its
/// heap slot, and with `key` equal to a key's bits too, `r[k]` is the value
/// at `slot`, read now.
///
/// Two more slots hold the loop's **run proofs** in a speculative region's
/// fast version (`numeric::NumGen::forin_run_proof`), both emptied by every
/// entry into that version: `table`, the key table its key read found the
/// key array's (no Rust code runs there, and only Rust code replaces a
/// site's table; the array register is written only before the region),
/// and `held`, the heap slot of the object its liveness check found of
/// that table's layout (no op there changes a shape but a fresh object's,
/// and the object register too is written only before the region).
#[derive(Clone, Copy, Debug)]
pub(super) struct Memo {
    pub(super) obj: i32,
    pub(super) cell: i32,
    pub(super) key: i32,
    pub(super) slot: i32,
    pub(super) table: i32,
    pub(super) held: i32,
}

impl Memo {
    /// Frame slots per memo.
    pub(super) const SLOTS: i32 = 6;

    /// The memo whose slots are the [`Self::SLOTS`] frame slots under `base`.
    pub(super) fn at(base: i32) -> Self {
        Self {
            obj: base - 8,
            cell: base - 16,
            key: base - 24,
            slot: base - 32,
            table: base - 40,
            held: base - 48,
        }
    }
}

/// The `for-in` loops of a body (see the module docs).
#[derive(Default)]
pub(crate) struct Plan {
    /// Per op: the [`ForInSite`] index of a loop's `EnumKeys`.
    pub(super) site_at: Vec<Option<usize>>,
    /// Per op: the loop whose element read of its key array it is.
    pub(super) read_at: Vec<Option<Loop>>,
    /// Per register: the loop whose current key it holds (`None` for any
    /// other register, or one that two loops' keys reach).
    pub(super) key: Vec<Option<Loop>>,
    /// The loops' sites (stable addresses: the body owns them).
    pub(super) sites: Box<[ForInSite]>,
}

impl Plan {
    /// The address of site `i`.
    pub(super) fn site_addr(&self, i: usize) -> Option<u64> {
        self.sites.get(i).map(|s| core::ptr::from_ref(s) as u64)
    }

    /// The loop whose key register is `k`.
    pub(super) fn key_loop(&self, k: u16) -> Option<Loop> {
        self.key.get(k as usize).copied().flatten()
    }
}

/// Finds the body's `for-in` loops (see the module docs).
pub(super) fn plan(ops: &[GenericOp], bytecode: &[Op], n_regs: usize) -> Plan {
    let n = ops.len().min(bytecode.len());
    // The ops that may write `reg` (a loop's own ops write only their
    // destinations; any other op handed to the interpreter may write every
    // register it names).
    let writes = |pc: usize, reg: u16| match bytecode[pc] {
        Op::Relational { dst, .. }
        | Op::StrictEq { dst, .. }
        | Op::Lt { dst, .. }
        | Op::ArrayLen { dst, .. }
        | Op::GetElem { dst, .. }
        | Op::GetKey { dst, .. }
        | Op::LoadConst { dst, .. }
        | Op::Move { dst, .. } => dst == reg,
        Op::Jump { .. } | Op::JumpIfFalse { .. } => false,
        _ => super::recv_cache::writes(&ops[pc], bytecode.get(pc), reg),
    };
    let writers = |reg: u16| (0..n).filter(|&pc| writes(pc, reg)).collect::<Vec<usize>>();
    let mut site_at = vec![None; n];
    let mut read_at = vec![None; n];
    let mut key: Vec<Option<Option<Loop>>> = vec![None; n_regs];
    let mut n_sites = 0;
    for e in 0..n {
        let (Op::EnumKeys { dst: arr, obj }, GenericOp::Interp { pc }) = (&bytecode[e], &ops[e])
        else {
            continue;
        };
        let (arr, obj) = (*arr, *obj);
        if *pc as usize != e
            || arr == obj
            || (arr as usize) >= n_regs
            || (obj as usize) >= n_regs
            || writers(obj).iter().any(|&w| w > e)
            || writers(arr) != [e]
        {
            continue;
        }
        // The loop's one element read of its key array.
        let mut reads = (e + 1..n).filter_map(|pc| match bytecode[pc] {
            Op::GetElem { dst, arr: a, index } if a == arr => Some((pc, dst, index)),
            _ => None,
        });
        let (Some((read, cur, index)), None) = (reads.next(), reads.next()) else {
            continue;
        };
        if (cur as usize) >= n_regs || cur == obj || cur == arr || index == arr || index == obj {
            continue;
        }
        let l = Loop {
            site: n_sites,
            arr,
            index,
            obj,
        };
        n_sites += 1;
        site_at[e] = Some(l.site);
        read_at[read] = Some(l);
        // `cur` and its copies.
        let mut mine = vec![false; n_regs];
        mine[cur as usize] = true;
        loop {
            let mut grew = false;
            for op in &bytecode[..n] {
                if let Op::Move { dst, src } = *op
                    && mine.get(src as usize) == Some(&true)
                    && mine.get(dst as usize) == Some(&false)
                {
                    mine[dst as usize] = true;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        for (r, _) in mine.iter().enumerate().filter(|m| *m.1) {
            key[r] = match key[r] {
                None => Some(Some(l)),
                Some(_) => Some(None),
            };
        }
    }
    Plan {
        site_at,
        read_at,
        key: key.into_iter().map(Option::flatten).collect(),
        sites: (0..n_sites).map(|_| ForInSite::new()).collect(),
    }
}

/// `sa` (`numeric::NumEnv::alloc_inline`) with [`super::numeric::SA_FORIN`]
/// set on the ops of the plan's loops that a promoted loop region may
/// contain: a loop's element read of its key array (`cur = arr[i]`) and its
/// liveness check (`cur in obj`). Each is emitted inline in a region (its
/// key index read from wherever the region keeps it: [`Idx`]), and every
/// path but its inline one leaves a speculative version for the plain copy
/// (`compile_generic`).
pub(super) fn region_flags(
    plan: &Plan,
    ops: &[GenericOp],
    bytecode: &[Op],
    n_regs: usize,
    mut sa: Vec<u8>,
) -> Vec<u8> {
    let n = ops.len().min(bytecode.len());
    let ok = |r: u16| (r as usize) < n_regs;
    for pc in 0..n {
        let flag = match (&ops[pc], &bytecode[pc]) {
            (&GenericOp::ArrGet { arr, index, .. }, _) => plan
                .read_at
                .get(pc)
                .copied()
                .flatten()
                .is_some_and(|l| l.arr == arr && l.index == index),
            (
                &GenericOp::Interp { pc: p },
                &Op::Relational {
                    dst,
                    instanceof: false,
                    a: key,
                    b: obj,
                },
            ) => {
                p as usize == pc
                    && ok(dst)
                    && ok(key)
                    && ok(obj)
                    && plan.key_loop(key).is_some_and(|l| {
                        l.obj == obj && ok(l.arr) && ok(l.index) && dst != l.index && dst != l.arr
                    })
            }
            _ => false,
        };
        if flag {
            if sa.len() < n {
                sa.resize(n, 0);
            }
            sa[pc] |= super::numeric::SA_FORIN;
        }
    }
    sa
}

/// For the liveness check at `q` of loop `l`: the pc of the loop's key read
/// when every way to `q` comes straight from it — only branches between
/// them (no register written), no branch into that stretch from outside it
/// and no entry (`entries`: OSR entries, catch entries) inside it. In a
/// speculative loop's fast version, where the read's every miss leaves,
/// the check then finds the key register still the table's key `i` of the
/// table's array, `i` below its length ([`X64Assembler::fi_slot_proven`]).
pub(super) fn follows_read(
    plan: &Plan,
    ops: &[GenericOp],
    q: usize,
    l: Loop,
    entries: &[usize],
) -> Option<usize> {
    let r = (0..q)
        .rev()
        .find(|&p| plan.read_at.get(p).copied().flatten() == Some(l))?;
    if !matches!(ops.get(r), Some(GenericOp::ArrGet { .. })) {
        return None;
    }
    let between = r + 1..=q;
    let quiet = (r + 1..q).all(|p| {
        matches!(
            ops[p],
            GenericOp::Jump { .. } | GenericOp::JumpIfFalse { .. } | GenericOp::Nop
        )
    });
    let inside = |p: usize| (r..q).contains(&p);
    let from_outside = ops.iter().enumerate().any(|(p, op)| match *op {
        GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } => {
            between.contains(&target) && !inside(p)
        }
        GenericOp::PushHandler { target, .. } => between.contains(&target),
        GenericOp::InlineGuard { call, .. } => between.contains(&(call as usize)),
        _ => false,
    });
    (quiet && !from_outside && !entries.iter().any(|e| between.contains(e))).then_some(r)
}

/// A [`KeyTable`] field's offset.
macro_rules! table_off {
    ($f:ident) => {
        i32::try_from(core::mem::offset_of!(KeyTable, $f)).expect("offset fits in i32")
    };
}

/// Where a compiled loop's key index (register `l.index`) is when one of
/// its ops reads it: the register file, the SSE register a promoted loop
/// keeps it in, or the frame slot (`rbp`-relative) a speculative loop's
/// integer version saved the int GPR holding it to before the op (an
/// int32: see `numeric::int32`, `NumGen::int_save`), or that GPR itself
/// (`rsi` / `rdi`, when the op's code keeps them and saved nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Idx {
    File,
    Xmm(u8),
    Saved(i32),
    Gpr(u8),
}

impl X64Assembler {
    /// `rax` = the bits of key `i` (the key index, at `idx`) of loop `l`'s
    /// key table, when `l`'s array is the table's — what the array holds
    /// there (see the module docs); else to `miss`. Clobbers `r9`, `r11`,
    /// `xmm0`.
    pub(super) fn fi_key(
        &mut self,
        site: u64,
        lp: Loop,
        idx: Idx,
        disp: impl Fn(u16) -> i32,
        miss: Label,
    ) {
        self.fi_index_rax(lp, idx, &disp);
        self.fi_table(site, lp, &disp, miss);
        self.fi_bound(miss);
        self.g_load(r::R9, r::R11, table_off!(bits_ptr));
        self.p_load_index8(r::RAX, r::R9, r::RAX, 0);
    }

    /// [`Self::fi_key`] with the loop's run proof of its table
    /// ([`Memo::table`], frame slot `proof`): once it holds the table, the
    /// table with no check of the site or the array; else the full check,
    /// which records it there. Clobbers `r9`, `r11`, `xmm0`.
    pub(super) fn fi_key_proving(
        &mut self,
        site: u64,
        lp: Loop,
        idx: Idx,
        disp: impl Fn(u16) -> i32,
        proof: i32,
        miss: Label,
    ) {
        let have = self.new_label();
        self.fi_index_rax(lp, idx, &disp);
        self.g_load(r::R11, r::RBP, proof);
        self.g_test(r::R11);
        self.g_jcc(cc::NE, have);
        self.fi_table(site, lp, &disp, miss);
        // (Never `0`: a site holds its empty table when it has none, whose
        // array, `0`, no key array is.)
        self.g_store(r::RBP, proof, r::R11);
        self.bind(have);
        self.fi_bound(miss);
        self.g_load(r::R9, r::R11, table_off!(bits_ptr));
        self.p_load_index8(r::RAX, r::R9, r::RAX, 0);
    }

    /// The slot of loop `l`'s key `i` (its index at `idx`) in the proved
    /// table ([`Memo::table`], frame slot `table`): `rax` = it (maybe
    /// [`crate::realm::NO_SLOT`]). For a liveness check right after the
    /// loop's key read ([`follows_read`]) whose object's run proof
    /// ([`Memo::held`]) holds: the read proved the table, and the object
    /// has the table's layout. Clobbers `r9`, `r11`, `xmm0`.
    pub(super) fn fi_slot_held(
        &mut self,
        lp: Loop,
        idx: Idx,
        disp: impl Fn(u16) -> i32,
        table: i32,
    ) {
        self.fi_index_rax(lp, idx, &disp);
        self.g_load(r::R11, r::RBP, table);
        self.g_load(r::R9, r::R11, table_off!(slots_ptr));
        self.p_load_index8(r::RAX, r::R9, r::RAX, 0);
    }

    /// `r11` = loop `l`'s key table, when `l`'s array is the table's (`0`,
    /// never a handle, when there is none); else to `miss`. Clobbers `r9`.
    fn fi_table(&mut self, site: u64, lp: Loop, disp: &impl Fn(u16) -> i32, miss: Label) {
        let site_table = i32::try_from(core::mem::offset_of!(ForInSite, table)).expect("offset");
        self.g_movabs(r::R11, site);
        self.g_load(r::R11, r::R11, site_table);
        self.g_load(r::R9, r::RBX, disp(lp.arr));
        self.g_cmp_mem(r::R9, r::R11, table_off!(array));
        self.g_jcc(cc::NE, miss);
    }

    /// `rax` = the key index (at `idx`; a Number: its truncation, a NaN's
    /// out of range; an int32 sign-extended, a negative one out of range).
    /// Clobbers `xmm0`.
    fn fi_index_rax(&mut self, lp: Loop, idx: Idx, disp: &impl Fn(u16) -> i32) {
        match idx {
            Idx::Xmm(x) => super::numeric::NumGen::trunc_from_xmm(self, x),
            // movsxd rax, dword [rbp + d]
            Idx::Saved(d) => {
                self.raw(&[0x48, 0x63, 0x85]);
                self.raw(&d.to_le_bytes());
            }
            // movsxd rax, r32(g)
            Idx::Gpr(g) => self.raw(&[0x48 | (g >> 3), 0x63, 0xc0 | (g & 7)]),
            Idx::File => {
                self.g_load(r::RAX, r::RBX, disp(lp.index));
                super::numeric::NumGen::set_xmm_from_rax(self, 0);
                self.cvttsd2si_rax_xmm0();
            }
        }
    }

    /// To `miss` unless the index in `rax` is below the table's (`r11`)
    /// length.
    fn fi_bound(&mut self, miss: Label) {
        self.g_cmp_mem(r::RAX, r::R11, table_off!(len));
        self.g_jcc(cc::AE, miss);
    }

    /// `rax` = the slot of the key in register `key` in loop `l`'s key table
    /// (its index at `idx`, see [`Self::fi_key`]), for the receiver whose
    /// heap slot is `rcx` (a shaped ordinary object: the caller checked) —
    /// when `l`'s array is the table's, the receiver has the table's layout
    /// and the key is the table's `i`-th (see the module docs); else to
    /// `miss`. The slot may be [`crate::realm::NO_SLOT`]. Clobbers `r9`,
    /// `r11`, `xmm0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fi_slot(
        &mut self,
        l: &JitLayout,
        site: u64,
        lp: Loop,
        key: u16,
        idx: Idx,
        disp: impl Fn(u16) -> i32,
        miss: Label,
    ) {
        // (The index first: an int GPR holding it may be `r9`.)
        self.fi_index_rax(lp, idx, &disp);
        self.fi_table(site, lp, &disp, miss);
        // The receiver's layout is the table's.
        self.g_load(r::R9, r::RCX, l.off_shape);
        self.g_cmp_mem(r::R9, r::R11, table_off!(shape_word));
        self.g_jcc(cc::NE, miss);
        self.fi_bound(miss);
        // The key is the table's `i`-th.
        self.g_load(r::R9, r::R11, table_off!(bits_ptr));
        self.p_load_index8(r::R9, r::R9, r::RAX, 0);
        self.g_cmp_mem(r::R9, r::RBX, disp(key));
        self.g_jcc(cc::NE, miss);
        self.g_load(r::R9, r::R11, table_off!(slots_ptr));
        self.p_load_index8(r::RAX, r::R9, r::RAX, 0);
    }

    /// [`Self::fi_slot`] where the loop's key read has just proved `l`'s
    /// array the table's, the index below its length and the key register
    /// the table's key `i` ([`follows_read`]): the layout check and the
    /// slot alone. With `held` (the object's run proof, [`Memo::held`]):
    /// the object's heap slot recorded there once its layout is the
    /// table's. Clobbers `r9`, `r11`, `xmm0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fi_slot_proven(
        &mut self,
        l: &JitLayout,
        site: u64,
        lp: Loop,
        idx: Idx,
        disp: impl Fn(u16) -> i32,
        held: Option<i32>,
        miss: Label,
    ) {
        let site_table = i32::try_from(core::mem::offset_of!(ForInSite, table)).expect("offset");
        self.fi_index_rax(lp, idx, &disp);
        self.g_movabs(r::R11, site);
        self.g_load(r::R11, r::R11, site_table);
        self.g_load(r::R9, r::RCX, l.off_shape);
        self.g_cmp_mem(r::R9, r::R11, table_off!(shape_word));
        self.g_jcc(cc::NE, miss);
        if let Some(held) = held {
            self.g_store(r::RBP, held, r::RCX);
        }
        self.g_load(r::R9, r::R11, table_off!(slots_ptr));
        self.p_load_index8(r::RAX, r::R9, r::RAX, 0);
    }

    /// A loop's `EnumKeys` when the table its site holds is still the key
    /// list of the receiver whose heap slot is `rcx` (a shaped ordinary
    /// object without the host-exotic flag: the caller checked), as
    /// `Realm::for_in_table_hit` decides it: `rax` = the table's array;
    /// else to `miss`. The receiver is plain shaped
    /// (`Object::is_plain_shaped`) with the
    /// table's layout — which decides that it is no primitive wrapper — and
    /// the `[[Prototype]]` the table's chain was found unchanged for at the
    /// prototype epoch that still is current. Clobbers `r8`, `r9`, `r11`.
    pub(super) fn fi_enum_hit(&mut self, l: &JitLayout, site: u64, miss: Label) {
        let site_table = i32::try_from(core::mem::offset_of!(ForInSite, table)).expect("offset");
        // Flags byte 0, an `extra` part (if any) of spilled values only.
        self.cmp_byte_at_rcx_imm(l.props.off_jit_flags, 0);
        self.jne(miss);
        let plain = self.new_label();
        self.g_load(r::R9, r::RCX, l.props.off_extra);
        self.g_test(r::R9);
        self.g_jcc(cc::E, plain);
        self.p_extra_only_spill(l, miss);
        self.bind(plain);
        self.g_movabs(r::R11, site);
        self.g_load(r::R11, r::R11, site_table);
        self.g_load(r::R9, r::RCX, l.off_shape);
        self.g_cmp_mem(r::R9, r::R11, table_off!(shape_word));
        self.g_jcc(cc::NE, miss);
        self.p_load16(r::R9, r::RCX, l.off_slots_len);
        self.g_cmp_mem(r::R9, r::R11, table_off!(shape_len));
        self.g_jcc(cc::NE, miss);
        self.g_load(r::R9, r::RCX, l.props.off_proto);
        self.g_cmp_mem(r::R9, r::R11, table_off!(proto_raw));
        self.g_jcc(cc::NE, miss);
        self.g_movabs(r::R9, crate::object::proto_epoch_addr() as u64);
        self.g_load(r::R9, r::R9, 0);
        self.g_cmp_mem(r::R9, r::R11, table_off!(stamp));
        self.g_jcc(cc::NE, miss);
        self.g_load(r::RAX, r::R11, table_off!(array));
        self.g_test(r::RAX);
        self.g_jcc(cc::E, miss);
    }

    /// To `miss` unless the object whose heap slot is `rcx` has no
    /// accessors and no String has an auxiliary object (`realm_off`,
    /// `str_special`: the context's realm and its flag). Clobbers `r9`.
    pub(super) fn fi_plain_read(
        &mut self,
        l: &JitLayout,
        (realm_off, str_special): (i32, i32),
        miss: Label,
    ) {
        let no_acc = self.new_label();
        self.g_load(r::R9, r::RCX, l.props.off_extra);
        self.g_test(r::R9);
        self.g_jcc(cc::E, no_acc);
        self.g_load(r::R9, r::R9, l.props.extra_acc_len);
        self.g_test(r::R9);
        self.g_jcc(cc::NE, miss);
        self.bind(no_acc);
        self.g_load(r::R9, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R9, str_special, 0);
        self.g_jcc(cc::NE, miss);
    }
}
