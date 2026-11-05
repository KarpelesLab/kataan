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

/// A [`KeyTable`] field's offset.
macro_rules! table_off {
    ($f:ident) => {
        i32::try_from(core::mem::offset_of!(KeyTable, $f)).expect("offset fits in i32")
    };
}

impl X64Assembler {
    /// `rax` = the bits of key `i` (register `l.index`) of loop `l`'s key
    /// table, when `l`'s array is the table's — what the array holds there
    /// (see the module docs); else to `miss`. Clobbers `r9`, `r11`, `xmm0`.
    pub(super) fn fi_key(&mut self, site: u64, lp: Loop, disp: impl Fn(u16) -> i32, miss: Label) {
        self.fi_table(site, lp, &disp, miss);
        self.fi_index(lp, &disp, miss);
        self.g_load(r::R9, r::R11, table_off!(bits_ptr));
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

    /// `rax` = the key index (register `l.index`, a Number: its truncation,
    /// a NaN's out of range) when it is below the table's (`r11`) length;
    /// else to `miss`. Clobbers `xmm0`.
    fn fi_index(&mut self, lp: Loop, disp: &impl Fn(u16) -> i32, miss: Label) {
        self.g_load(r::RAX, r::RBX, disp(lp.index));
        super::numeric::NumGen::set_xmm_from_rax(self, 0);
        self.cvttsd2si_rax_xmm0();
        self.g_cmp_mem(r::RAX, r::R11, table_off!(len));
        self.g_jcc(cc::AE, miss);
    }

    /// `rax` = the slot of the key in register `key` in loop `l`'s key table
    /// (its index in `l.index`), for the receiver whose heap slot is `rcx`
    /// (a shaped ordinary object: the caller checked) — when `l`'s array is
    /// the table's, the receiver has the table's layout and the key is the
    /// table's `i`-th (see the module docs); else to `miss`. The slot may be
    /// [`crate::realm::NO_SLOT`]. Clobbers `r9`, `r11`, `xmm0`.
    pub(super) fn fi_slot(
        &mut self,
        l: &JitLayout,
        site: u64,
        lp: Loop,
        key: u16,
        disp: impl Fn(u16) -> i32,
        miss: Label,
    ) {
        self.fi_table(site, lp, &disp, miss);
        // The receiver's layout is the table's.
        self.g_load(r::R9, r::RCX, l.off_shape);
        self.g_cmp_mem(r::R9, r::R11, table_off!(shape_word));
        self.g_jcc(cc::NE, miss);
        self.fi_index(lp, &disp, miss);
        // The key is the table's `i`-th.
        self.g_load(r::R9, r::R11, table_off!(bits_ptr));
        self.p_load_index8(r::R9, r::R9, r::RAX, 0);
        self.g_cmp_mem(r::R9, r::RBX, disp(key));
        self.g_jcc(cc::NE, miss);
        self.g_load(r::R9, r::R11, table_off!(slots_ptr));
        self.p_load_index8(r::RAX, r::R9, r::RAX, 0);
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
