//! A compiled `x instanceof F` answered in line from a per-site **memo**.
//!
//! The op's helper (`jit_helper_instance_of`, over the interpreter's
//! `instance_of_fast`) answers `x instanceof F` when `F` is a VM closure with
//! the default `[[HasInstance]]` (no own or inherited `@@hasInstance` before
//! `%Function.prototype%`'s, which is non-writable and non-configurable),
//! its `prototype` an object, and `x`'s chain free of proxies. Each answer it
//! gives a site also records, in the site's memo (three words of the body's
//! persistent array, `global_persist`), `F`'s bits, the address of the slot
//! of `F`'s auxiliary object that holds `prototype` (a data property of a
//! shaped object without accessors) and the prototype epoch, after watching
//! `F` and that object (`Realm::watch_cell`): any later change to their
//! layout (an own `@@hasInstance` added, `prototype` deleted or made an
//! accessor, the slots moved) or to `F`'s `[[Prototype]]` bumps the epoch.
//! So while the epoch is the memo's stamp (and the run has one realm, as the
//! helper requires), the same `F` still has the default `[[HasInstance]]`
//! and its `prototype` is the value at that address now, and the answer is
//! OrdinaryHasInstance's:
//!
//! - `x` not a heap value: `false` (no primitive but a String is one, and a
//!   String goes to the helper), whatever the `prototype`;
//! - `x` an ordinary object (`Cell::Object`, whose `[[GetPrototypeOf]]` is
//!   its `proto` word): its chain walked a few links in line, each link an
//!   ordinary object, until the `prototype` (`true`: a link is an object) or
//!   a null link (`false` when the `prototype` is an ordinary object, which
//!   is checked then; otherwise the helper, whose op throws for a primitive
//!   one).
//!
//! Anything else (another `F`, a stale stamp, a `prototype` that is not a
//! heap value, another kind of cell on the chain, a longer chain) is the
//! helper's. The walk reads only rooted handles (`proto` words of live
//! objects, the `prototype` a live object holds) with the rooted walk
//! (`super::handle`), comparing each slot's cell tag before reading it.

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};

/// The memo's words: `F`'s bits, the address of its `prototype`'s slot,
/// the prototype epoch it was validated at (`0`: never).
pub(crate) const MEMO_WORDS: usize = 3;

/// Whether the memo is used (`KATAAN_JIT_INSTMEMO=0` turns it off; read once
/// per process).
pub(crate) fn memo_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_INSTMEMO").is_none_or(|v| v != "0"))
}

/// How many links of `x`'s chain the in-line walk follows.
const HOPS: usize = 3;

/// `NanBox` bits the code stores.
const TRUE_BITS: u64 = crate::nanbox::NanBox::boolean(true).to_bits();
const FALSE_BITS: u64 = crate::nanbox::NanBox::boolean(false).to_bits();

impl X64Assembler {
    /// The memo's answer for `x instanceof F` (see the module docs), the
    /// operands at frame displacements `x` and `f`, the memo at `memo`:
    /// the boolean stored to `dst` and on to `done`; to `miss` (nothing
    /// done) for the helper. Clobbers `rax`, `rcx`, `r9`, `r11` and,
    /// unpinned, `rdx` and `r10`.
    pub(crate) fn instof_memo_hit(
        &mut self,
        l: &JitLayout,
        (realm_off, multi_realm_off): (i32, i32),
        memo: u64,
        (x, f, dst): (i32, i32, i32),
        (miss, done): (Label, Label),
    ) {
        let yes = self.new_label();
        let no = self.new_label();
        let end = self.new_label();
        // The same `F`, at the memo's stamp, one realm.
        self.g_movabs(r::R11, memo);
        let base = self.frame_base(f);
        self.g_load(r::RAX, base, f);
        self.g_cmp_mem(r::RAX, r::R11, 0);
        self.g_jcc(cc::NE, miss);
        self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
        self.g_cmp_mem(r::R9, r::R11, 16);
        self.g_jcc(cc::NE, miss);
        // r9 = its `prototype` now: a heap value.
        self.g_load(r::R9, r::R11, 8);
        self.g_load(r::R9, r::R9, 0);
        self.p_cmp_handle_floor(r::R9);
        self.g_jcc(cc::B, miss);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_cmp_byte_imm(r::R11, multi_realm_off, 0);
        self.g_jcc(cc::NE, miss);
        // (The arena before `x`: unpinned, loading it takes `rax`, `rdx`
        // and `r11`.)
        self.p_arena(realm_off, &l.props);
        let base = self.frame_base(x);
        self.g_load(r::RAX, base, x);
        self.p_cmp_handle_floor(r::RAX);
        self.g_jcc(cc::B, no);
        self.p_slot_rooted(l, r::RCX, miss);
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, l.cell_object_disc);
        self.g_jcc(cc::NE, miss);
        for hop in 0..HOPS {
            self.g_load(r::RAX, r::RCX, l.props.off_proto);
            const _: () = assert!(crate::object::NO_PROTO == u64::MAX);
            self.g_cmp_imm(r::RAX, -1);
            self.g_jcc(cc::E, end);
            // (The link as a `NanBox` handle, to compare with the value.)
            self.p_or_handle_tag_rax();
            self.g_cmp(r::RAX, r::R9);
            self.g_jcc(cc::E, yes);
            if hop + 1 < HOPS {
                self.p_slot_rooted(l, r::RCX, miss);
                self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, l.cell_object_disc);
                self.g_jcc(cc::NE, miss);
            }
        }
        self.jmp(miss);
        // The chain's end: `false` when the `prototype` is an ordinary
        // object (else the helper: the op throws for a primitive).
        self.bind(end);
        self.g_mov(r::RAX, r::R9);
        self.p_slot_rooted(l, r::RCX, miss);
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, l.cell_object_disc);
        self.g_jcc(cc::NE, miss);
        self.bind(no);
        self.g_movabs(r::RAX, FALSE_BITS);
        self.store_rax(dst);
        self.jmp(done);
        self.bind(yes);
        self.g_movabs(r::RAX, TRUE_BITS);
        self.store_rax(dst);
        self.jmp(done);
    }
}
