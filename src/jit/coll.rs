//! A compiled `m.get(k)` / `m.has(k)` / `s.has(k)` on a small `Map`/`Set`
//! with no call.
//!
//! The site's proof ([`CollProof`], made by `native_call::prove_coll` from
//! the fill helper) is the one an inline `a.push(x)` relies on, for the
//! default `Map.prototype` / `Set.prototype` (`Realm::coll_default`) as the
//! start of the lookup: while the prototype epoch is the proof's stamp, the
//! method a collection flagged `default_proto` finds is the data property
//! of that (watched) prototype whose value is at `ptr`, which was the
//! built-in when it was `method`'s bits (read live). A collection so
//! flagged has no auxiliary object, so no own named property, and is not
//! host-exotic (`Realm::coll_default_to_explicit` keeps that), so its
//! `[[Get]]` of the key does continue at that prototype. The proof is for
//! one kind (`is_set`), which the receiver must be. A call whose method is
//! already read (`Op::CallWithMethod`, an argument the compiler evaluates
//! after the method) needs no lookup proof: the method read must be the
//! proof's built-in itself, called on any strong collection of its kind
//! (a subclass instance's too).
//!
//! The lookup is then `Realm::collection_scan`'s, for a strong collection
//! without a hash index and with at most `COLL_SCAN_MAX` entries (a
//! tombstone is a `(hole, hole)` entry, which no key equals): an entry
//! whose key has the key's very bits is a hit (a Number key normalized
//! first: `-0` is stored as `+0`, and every NaN is the canonical one, so
//! for Numbers the bits are `SameValueZero`); otherwise a key that is not
//! a heap value, or is an object, a function or an array (compared by
//! identity), is absent; a String key held in its rope node (`Node::Small`,
//! at most 24 bytes) is compared with each String entry's text, an entry
//! held in its node by its length and padded words (the bytes past a short
//! string's length are zero), a flat entry String of another length being
//! another text, and any other entry String sending the call to the
//! helper. Any other key (a BigInt, a Symbol, a longer or concatenated
//! String) and any other collection go to the helper, which runs the same
//! algorithm or the general one.
//!
//! # Writes
//!
//! `m.set(k, v)`, `s.add(k)` and `.delete(k)` ([`CollWrite`]) take the same
//! proof (of `Map.prototype.set`, `Set.prototype.add` or the kind's
//! `delete`) and the same scan, on a receiver with the default
//! `[[Prototype]]`, and then do what `Realm::collection_set` /
//! `Realm::collection_delete` do for a collection without a hash index:
//!
//! * a key found: `set` replaces the entry's value (when the write
//!   barrier, `p_barrier_check`, has nothing to do for it), `add` changes
//!   nothing, `delete` leaves a `(hole, hole)` tombstone in place and
//!   counts one live entry fewer (a hole is no heap value: nothing for the
//!   barrier). Compaction, which only ever drops tombstones, is left to a
//!   later Rust-side delete: it changes nothing a program can observe;
//! * no key found: `set` / `add` append the entry (a `-0` key as `+0`, a
//!   `Set`'s value its key) in the entries' spare capacity, when the
//!   barrier has nothing to do for the key or the value; one live entry
//!   more. A collection past `COLL_SCAN_MAX` entries this way still has no
//!   index, as the Rust path leaves it; `delete` answers `false`.
//!
//! `set` and `add` answer the receiver. Anything else (no spare capacity,
//! a barrier with work, an index, a larger collection, a key compared out
//! of line) is the call's, with nothing done.

use super::call_seq::{cc, r};
use super::props::MethodSite;
use super::{JitLayout, Label, X64Assembler};

/// A body's **lookup memo**: the frame slots of the last lookup an inline
/// `m.has(k)` / `m.get(k)` made on a `Map` (`recv`: the receiver's bits, or
/// `RECV_CACHE_EMPTY`; `key`: the key's bits as the program passed them;
/// `entry`: the address of the entry found, or `0` for none), so that the
/// `m.get(k)` after an `m.has(k)` (or a repeated lookup) answers without a
/// scan: `if (m.has(k)) return m.get(k)`.
///
/// The memo's receiver slot is one of the body's receiver-cache keys
/// (`X64Assembler::recv_cache_keys`), emptied by the prologue, at every
/// entry and after every `call`: only Rust code (behind a call) can
/// reallocate a collection's entries, add a hash index, free or reuse a heap
/// slot, or change its `[[Prototype]]`. The one inline code that changes
/// entries, `coll_update`, empties it too. So while it holds a receiver's
/// bits, that `Map`'s entry at `entry` is still the one found (its value
/// read live), and `0` still means no entry has the key. A hit needs the
/// site's own proof (the method is the built-in) of `Map` kind; it is
/// filled only from a lookup on a `Map` (the proof's kind is the
/// receiver's).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CollMemo {
    pub(crate) recv: i32,
    pub(crate) key: i32,
    pub(crate) entry: i32,
}

/// Whether the lookup memo is used (`KATAAN_JIT_COLLMEMO=0` turns it off;
/// read once per process).
pub(crate) fn memo_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_COLLMEMO").is_none_or(|v| v != "0"))
}

/// Whether an inline lookup probes a collection's hash index in line
/// (`X64Assembler::coll_hash_probe`) rather than through the helper
/// (`KATAAN_JIT_COLLHASH=0` turns it off; read once per process).
fn hash_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_COLLHASH").is_none_or(|v| v != "0"))
}

/// Whether an inline `Map`/`Set` update serves an indexed or larger
/// collection (and a method already read) through its out-of-line write
/// (`jit_helper_coll_write`) rather than the call
/// (`KATAAN_JIT_COLLWRITE=0` turns it off; read once per process).
pub(crate) fn write_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_COLLWRITE").is_none_or(|v| v != "0"))
}

/// A lookup site's use of the memo ([`memo_plan`]): it answers from the
/// memo first.
pub(crate) const MEMO_CHECK: u8 = 1;
/// See [`MEMO_CHECK`]: it notes its answer in the memo.
pub(crate) const MEMO_FILL: u8 = 2;

/// Per op of bytecode `ops`, how its `m.has(k)` / `m.get(k)` site (should
/// it compile to an inline lookup) uses the lookup memo: a `get` checks it
/// when an earlier `has` or `get` site takes its key from the same
/// register, which then fills it (`if (m.has(k)) return m.get(k)`;
/// `m.get(k) … m.get(k)`). Only a guess of where it pays (the memo is
/// checked at run time). Empty when no site uses it.
pub(crate) fn memo_plan(ops: &[crate::nbvm::Op]) -> alloc::vec::Vec<u8> {
    use crate::nbvm::Op;
    let site = |op: &Op| match op {
        Op::CallMethod { key, args, .. } | Op::CallWithMethod { key, args, .. }
            if args.len() == 1 && matches!(key.as_str(), "has" | "get") =>
        {
            Some((key.as_str() == "get", args[0]))
        }
        _ => None,
    };
    let mut plan = alloc::vec![0u8; ops.len()];
    let mut any = false;
    for (g, op) in ops.iter().enumerate() {
        let Some((true, k)) = site(op) else { continue };
        for (h, earlier) in ops[..g].iter().enumerate() {
            if site(earlier).is_some_and(|(_, k2)| k2 == k) {
                plan[h] |= MEMO_FILL;
                plan[g] |= MEMO_CHECK;
                any = true;
            }
        }
    }
    if !any {
        plan.clear();
    }
    plan
}

/// See the module docs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CollProof {
    /// The prototype epoch it was made under (`0`: never; the epoch starts
    /// at `1`).
    stamp: u64,
    /// The address holding the method, and the method's bits.
    ptr: u64,
    method: u64,
    /// The receivers' `is_set` (`0` / `1`).
    is_set: u64,
    /// How many proofs were tried (capped at [`COLL_TRIES`]).
    tries: u64,
}

impl CollProof {
    /// The empty proof.
    pub(crate) const EMPTY: Self = Self {
        stamp: 0,
        ptr: 0,
        method: 0,
        is_set: 0,
        tries: 0,
    };
}

/// Which update an inline [`X64Assembler::coll_update`] makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollWrite {
    /// `Map.prototype.set`.
    MapSet,
    /// `Set.prototype.add`.
    SetAdd,
    /// `Map.prototype.delete` / `Set.prototype.delete`.
    Delete,
}

/// The proofs a site tries before it leaves its calls to the helper.
const COLL_TRIES: u64 = 64;

/// The longest String key compared inline (three words).
const KEY_MAX: u8 = 24;

/// The layout of a collection cell, as offsets from a heap slot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CollLayout {
    disc: u8,
    is_set: i32,
    is_weak: i32,
    default_proto: i32,
    entries_ptr: i32,
    entries_len: i32,
    entries_cap: i32,
    live: i32,
    index: i32,
    /// The compaction epoch (`u32`).
    epoch: i32,
    /// `CollIndex::head` (`u32`), from the index's box.
    index_head: i32,
    entry_size: i32,
    entry_key: i32,
    entry_val: i32,
}

/// The process's [`CollLayout`] (probed once).
pub(crate) fn coll_layout() -> &'static CollLayout {
    use std::sync::OnceLock;
    static LAYOUT: OnceLock<CollLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        let c = crate::cell::jit_coll_layout();
        let slot = crate::heap::jit_slot_layout(crate::cell::Cell::Box(
            crate::nanbox::NanBox::undefined(),
        ));
        let base = slot.value_off;
        let at = |v: usize| i32::try_from(base + v).expect("coll layout offset fits in i32");
        // A rooted walk proves its slot occupied by the cell-tag compare
        // alone (`super::handle`).
        assert_ne!(c.disc, slot.vacant_tag);
        assert_eq!(
            i32::try_from(base).ok(),
            Some(super::compute_jit_layout().off_cell_tag)
        );
        let i = |v: usize| i32::try_from(v).expect("small");
        CollLayout {
            disc: c.disc,
            is_set: at(c.is_set_off),
            is_weak: at(c.is_weak_off),
            default_proto: at(c.default_proto_off),
            entries_ptr: at(c.entries_ptr_off),
            entries_len: at(c.entries_len_off),
            entries_cap: at(c.entries_cap_off),
            live: at(c.live_off),
            index: at(c.index_off),
            epoch: at(c.epoch_off),
            index_head: i(core::mem::offset_of!(crate::realm::CollIndex, head)),
            entry_size: i(c.entry_size),
            entry_key: i(c.entry_key_off),
            entry_val: i(c.entry_val_off),
        }
    })
}

impl MethodSite {
    /// Records the proof of the module docs for the collection kind
    /// `is_set`, whose default prototype is `start`: `key` is a data
    /// property of that ordinary object (watched now) whose value `proves`
    /// (the built-in the site's key names for that kind). Otherwise the
    /// proof is one no call satisfies. Whether it was made. Runs no code.
    pub(crate) fn fill_coll(
        &mut self,
        realm: &mut crate::realm::Realm,
        start: Option<crate::heap::Handle>,
        is_set: bool,
        proves: impl FnOnce(&crate::realm::Realm, crate::nanbox::NanBox) -> bool,
    ) -> bool {
        let tries = self.coll.tries.saturating_add(1);
        self.coll = CollProof {
            tries,
            ..CollProof::EMPTY
        };
        if tries > COLL_TRIES || realm.multi_realm {
            return false;
        }
        let key = self.key();
        let found = start.and_then(|start| {
            let o = realm.jit_object(start)?;
            let shape = o.shaped_shape()?;
            if o.accessor_index(key).is_some() {
                return None;
            }
            let slot = shape.lookup(key)?;
            let ptr = o.jit_slot_ptr(slot)?;
            let value = o.slot(slot)?;
            Some((start, ptr as u64, value))
        });
        let Some((start, ptr, value)) = found else {
            return false;
        };
        if !proves(realm, value) {
            return false;
        }
        realm.jit_watch(start);
        self.coll = CollProof {
            stamp: crate::object::proto_epoch(),
            ptr,
            method: value.to_bits(),
            is_set: u64::from(is_set),
            tries,
        };
        true
    }

    /// The offsets, from a site, of its [`CollProof`]'s fields: `stamp`,
    /// `ptr`, `method`, `is_set`, `tries`.
    fn coll_offsets() -> [i32; 5] {
        use core::mem::offset_of;
        let m = offset_of!(MethodSite, coll);
        let i = |v: usize| i32::try_from(m + v).expect("site offset fits in i32");
        [
            i(offset_of!(CollProof, stamp)),
            i(offset_of!(CollProof, ptr)),
            i(offset_of!(CollProof, method)),
            i(offset_of!(CollProof, is_set)),
            i(offset_of!(CollProof, tries)),
        ]
    }
}

/// `NanBox` bits the code uses.
const TRUE_BITS: u64 = crate::nanbox::NanBox::boolean(true).to_bits();
const FALSE_BITS: u64 = crate::nanbox::NanBox::boolean(false).to_bits();
const UNDEF_BITS: u64 = crate::nanbox::NanBox::undefined().to_bits();
/// The tag bits every non-Number `NanBox` has all set.
const QNAN_MASK: u64 = 0x7ffc_0000_0000_0000;
const _: () = assert!(crate::nanbox::NanBox::undefined().to_bits() & QNAN_MASK == QNAN_MASK);

impl X64Assembler {
    /// `recv.get(key)` (`get`) or `recv.has(key)` (see the module docs):
    /// the result stored to the register at frame displacement `dst` and on
    /// to `done`; to `fill` when the site's proof is stale (another epoch,
    /// or for the other kind) and may be made again, to `slow` for
    /// anything else, with nothing done. With `callee` (the frame
    /// displacement of a method already read, `Op::CallWithMethod`), that
    /// method must be the proof's built-in itself, which is then called on
    /// any strong collection of its kind: the lookup that found it is done.
    /// Clobbers `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`–`r11`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn coll_lookup(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        site: *const MethodSite,
        get: bool,
        (callee, recv, key, dst): (Option<i32>, i32, i32, i32),
        (fill, slow, done): (Label, Label, Label),
        probe_helper: u64,
        memo_use: u8,
    ) {
        let c = coll_layout();
        let [stamp, ptr, method, is_set, tries] = MethodSite::coll_offsets();
        let stale = self.new_label();
        let probe = self.new_label();
        self.p_arena(realm_off, &l.props);
        self.g_movabs(r::R11, site as u64);
        if let Some(c) = callee {
            // The method read is the built-in the proof names; else a
            // proof of this epoch is the answer (another method), and a
            // stale one is made again.
            let proven = self.new_label();
            let base = self.frame_base(c);
            self.g_load(r::R9, base, c);
            self.g_cmp_mem(r::R9, r::R11, method);
            self.g_jcc(cc::E, proven);
            self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
            self.g_cmp_mem(r::R9, r::R11, stamp);
            self.g_jcc(cc::NE, stale);
            self.jmp(slow);
            self.bind(proven);
        } else {
            // The proof, at the current epoch ...
            self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
            self.g_cmp_mem(r::R9, r::R11, stamp);
            self.g_jcc(cc::NE, stale);
            // ... with the built-in still in the prototype's slot.
            self.g_load(r::R9, r::R11, ptr);
            self.g_load(r::R9, r::R9, 0);
            self.g_cmp_mem(r::R9, r::R11, method);
            self.g_jcc(cc::NE, slow);
        }
        // The memo's answer (see `CollMemo`), under a proof of `Map` kind.
        let memo = self.coll_memo;
        if let Some(m) = memo.filter(|_| memo_use & MEMO_CHECK != 0) {
            let miss = self.new_label();
            let none = self.new_label();
            self.g_mem(true, &[0x83], 7, r::R11, is_set); // cmp qword [r11+is_set], imm8
            self.code.push(0);
            self.g_jcc(cc::NE, miss);
            let base = self.frame_base(recv);
            self.g_load(r::RAX, base, recv);
            self.g_cmp_mem(r::RAX, r::RBP, m.recv);
            self.g_jcc(cc::NE, miss);
            let base = self.frame_base(key);
            self.g_load(r::R9, base, key);
            self.g_cmp_mem(r::R9, r::RBP, m.key);
            self.g_jcc(cc::NE, miss);
            self.g_load(r::R9, r::RBP, m.entry);
            self.g_test(r::R9);
            self.g_jcc(cc::E, none);
            if get {
                self.g_load(r::RAX, r::R9, c.entry_val);
            } else {
                self.g_movabs(r::RAX, TRUE_BITS);
            }
            self.store_rax(dst);
            self.jmp(done);
            self.bind(none);
            self.g_movabs(r::RAX, if get { UNDEF_BITS } else { FALSE_BITS });
            self.store_rax(dst);
            self.jmp(done);
            self.bind(miss);
        }
        // The receiver: a strong collection of the proof's kind with the
        // default `[[Prototype]]` (so no own property), no index, few
        // entries.
        let base = self.frame_base(recv);
        self.g_load(r::RAX, base, recv);
        self.p_is_handle(r::RCX, slow);
        self.p_slot_rooted(l, r::RCX, slow);
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, c.disc);
        self.g_jcc(cc::NE, slow);
        self.g_cmp_byte_imm(r::RCX, c.is_weak, 0);
        self.g_jcc(cc::NE, slow);
        if callee.is_none() {
            self.g_cmp_byte_imm(r::RCX, c.default_proto, 1);
            self.g_jcc(cc::NE, slow);
        }
        // movzx r9d, byte [rcx + is_set]
        self.g_mem(false, &[0x0f, 0xb6], r::R9, r::RCX, c.is_set);
        self.g_cmp_mem(r::R9, r::R11, is_set);
        // (A method read is the other kind's built-in only on a receiver
        // of the wrong kind: the helper's `TypeError`.)
        self.g_jcc(cc::NE, if callee.is_some() { slow } else { stale });
        let found = self.new_label();
        let absent = self.new_label();
        // (An indexed one probed in line, `coll_hash_probe`, when the body
        // pins the arena; else by the helper.)
        let hashed = self.new_label();
        let slots =
            crate::realm::jit_coll_index_layout().filter(|_| self.pinned() && hash_enabled());
        self.g_mem(true, &[0x83], 7, r::RCX, c.index); // cmp qword [rcx+index], imm8
        self.code.push(0);
        self.g_jcc(cc::NE, if slots.is_some() { hashed } else { probe });
        // rdi = the entry count (at most `COLL_SCAN_MAX`), rsi = the first.
        self.g_load(r::RDI, r::RCX, c.entries_len);
        self.g_cmp_imm(
            r::RDI,
            i32::try_from(crate::realm::COLL_SCAN_MAX).expect("small"),
        );
        self.g_jcc(cc::A, probe);
        self.g_load(r::RSI, r::RCX, c.entries_ptr);
        // r11 = the key's bits (`-0` as `+0`).
        let base = self.frame_base(key);
        self.g_load(r::R11, base, key);
        let normal = self.new_label();
        self.g_movabs(r::R9, crate::nanbox::NanBox::number(-0.0).to_bits());
        self.g_cmp(r::R11, r::R9);
        self.g_jcc(cc::NE, normal);
        self.g_movabs(r::R11, crate::nanbox::NanBox::number(0.0).to_bits());
        self.bind(normal);
        self.coll_scan(l, (found, absent, probe));
        if let Some((sp, sl)) = slots {
            // (The scan falls through to `absent`, bound next.)
            self.jmp(absent);
            let cold = self.cold_begin();
            self.bind(hashed);
            let base = self.frame_base(key);
            self.coll_key_bits(base, key);
            self.coll_hash_probe(
                l,
                (
                    i32::try_from(sp).expect("small"),
                    i32::try_from(sl).expect("small"),
                ),
                (found, absent, probe),
            );
            self.cold_end(cold);
        }
        // The answers (each noted in the memo, for a `Map`).
        let memo = memo.filter(|_| memo_use & MEMO_FILL != 0);
        self.bind(absent);
        if let Some(m) = memo {
            self.g_movabs(r::RSI, 0);
            self.coll_memo_fill(m, site, recv, key);
        }
        self.g_movabs(r::RAX, if get { UNDEF_BITS } else { FALSE_BITS });
        self.store_rax(dst);
        self.jmp(done);
        self.bind(found);
        if let Some(m) = memo {
            self.coll_memo_fill(m, site, recv, key);
        }
        if get {
            self.g_load(r::RAX, r::RSI, c.entry_val);
        } else {
            self.g_movabs(r::RAX, TRUE_BITS);
        }
        self.store_rax(dst);
        self.jmp(done);
        // Any other lookup on a receiver proved served here (an indexed or
        // larger collection, a key or entry compared out of line): the
        // helper's probe (`jit_helper_coll_lookup`), which runs no code; a
        // miss (`hole`) is the general path's.
        let cold = self.cold_begin();
        self.bind(probe);
        self.mov_rdi_r15();
        self.load_argreg(1, recv);
        self.load_argreg(2, key);
        // rcx = 0 (`Map.get`), 1 (`Map.has`) or 2 (`Set.has`): the proof's
        // kind, which the receiver's is, plus one for a `has`.
        self.g_movabs(r::R11, site as u64);
        self.g_load(r::RCX, r::R11, is_set);
        self.g_lea(r::RCX, r::RCX, i32::from(!get));
        self.g_movabs(r::RAX, probe_helper);
        self.call_rax();
        self.g_movabs(r::R11, crate::nanbox::NanBox::hole().to_bits());
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::E, slow);
        self.store_rax(dst);
        self.jmp(done);
        self.cold_end(cold);
        // A stale proof: made again (by the caller's `fill`) while the
        // site may still try.
        self.bind(stale);
        self.g_movabs(r::R11, site as u64);
        self.g_mem(true, &[0x83], 7, r::R11, tries); // cmp qword [r11+tries], imm8
        self.code.push(u8::try_from(COLL_TRIES).expect("an imm8"));
        self.g_jcc(cc::AE, slow);
        self.jmp(fill);
    }

    /// Notes a lookup's answer in the memo `m` (the entry's address in
    /// `rsi`, `0` for none) when the site's proof is of `Map` kind (the
    /// receiver's): the receiver's and the key's bits from their frame
    /// slots. Keeps `rsi`; clobbers `r9`, `r11`.
    fn coll_memo_fill(&mut self, m: CollMemo, site: *const MethodSite, recv: i32, key: i32) {
        let [_, _, _, is_set, _] = MethodSite::coll_offsets();
        let skip = self.new_label();
        self.g_movabs(r::R11, site as u64);
        self.g_mem(true, &[0x83], 7, r::R11, is_set); // cmp qword [r11+is_set], imm8
        self.code.push(0);
        self.g_jcc(cc::NE, skip);
        let base = self.frame_base(recv);
        self.g_load(r::R9, base, recv);
        self.g_store(r::RBP, m.recv, r::R9);
        let base = self.frame_base(key);
        self.g_load(r::R9, base, key);
        self.g_store(r::RBP, m.key, r::R9);
        self.g_store(r::RBP, m.entry, r::RSI);
        self.bind(skip);
    }

    /// `recv.set(key, val)` (`Map`), `recv.add(key)` (`Set`) or
    /// `recv.delete(key)` (either, as `w` says; see the module docs on
    /// writes): the result stored to the register at frame displacement
    /// `dst` and on to `done`; to `fill` when the site's proof is stale and
    /// may be made again, to `slow` for anything else, with nothing done.
    /// Clobbers `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`–`r11`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn coll_update(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        site: *const MethodSite,
        w: CollWrite,
        (callee, recv, key, val, dst): (Option<i32>, i32, i32, i32, i32),
        (fill, slow, done): (Label, Label, Label),
        probe_helper: Option<u64>,
    ) {
        let c = coll_layout();
        let [stamp, ptr, method, is_set, tries] = MethodSite::coll_offsets();
        let stale = self.new_label();
        // An indexed or larger collection, or a key compared out of line:
        // the helper's update (`jit_helper_coll_write`), which runs no
        // code, when there is one; else the call's.
        let probe = if probe_helper.is_some() {
            self.new_label()
        } else {
            slow
        };
        // (A change to any collection's entries empties the lookup memo.)
        if let Some(m) = self.coll_memo {
            self.g_movabs(r::R9, u64::MAX);
            self.g_store(r::RBP, m.recv, r::R9);
        }
        self.p_arena(realm_off, &l.props);
        let rbase = self.frame_base(recv);
        if probe_helper.is_none() && callee.is_none() {
            // The receiver first (an indexed or larger collection, whose
            // writes are never inline here, costs no more): a strong
            // collection with the default `[[Prototype]]`, no index, few
            // entries.
            self.g_load(r::RAX, rbase, recv);
            self.p_is_handle(r::RCX, slow);
            self.p_slot_rooted(l, r::RCX, slow);
            self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, c.disc);
            self.g_jcc(cc::NE, slow);
            self.g_mem(true, &[0x83], 7, r::RCX, c.index); // cmp qword [rcx+index], imm8
            self.code.push(0);
            self.g_jcc(cc::NE, slow);
            self.g_load(r::RDI, r::RCX, c.entries_len);
            self.g_cmp_imm(
                r::RDI,
                i32::try_from(crate::realm::COLL_SCAN_MAX).expect("small"),
            );
            self.g_jcc(cc::A, slow);
            self.g_cmp_byte_imm(r::RCX, c.is_weak, 0);
            self.g_jcc(cc::NE, slow);
            self.g_cmp_byte_imm(r::RCX, c.default_proto, 1);
            self.g_jcc(cc::NE, slow);
            // The proof, at the current epoch, with the built-in still in
            // the prototype's slot, for the receiver's kind.
            self.g_movabs(r::R11, site as u64);
            self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
            self.g_cmp_mem(r::R9, r::R11, stamp);
            self.g_jcc(cc::NE, stale);
            self.g_load(r::R9, r::R11, ptr);
            self.g_load(r::R9, r::R9, 0);
            self.g_cmp_mem(r::R9, r::R11, method);
            self.g_jcc(cc::NE, slow);
            self.g_mem(false, &[0x0f, 0xb6], r::R9, r::RCX, c.is_set);
            self.g_cmp_mem(r::R9, r::R11, is_set);
            self.g_jcc(cc::NE, stale);
        } else {
            // The receiver: a strong collection (anything else, the usual
            // other receiver of such a call, is the call's at once) ...
            self.g_load(r::RAX, rbase, recv);
            self.p_is_handle(r::RCX, slow);
            self.p_slot_rooted(l, r::RCX, slow);
            self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, c.disc);
            self.g_jcc(cc::NE, slow);
            self.g_cmp_byte_imm(r::RCX, c.is_weak, 0);
            self.g_jcc(cc::NE, slow);
            // ... under the proof (an indexed or larger collection is
            // served by the helper under it too) ...
            self.g_movabs(r::R11, site as u64);
            if let Some(cd) = callee {
                // The method read is the built-in the proof names; else a
                // proof of this epoch is the answer (another method), and
                // a stale one is made again.
                let proven = self.new_label();
                let base = self.frame_base(cd);
                self.g_load(r::R9, base, cd);
                self.g_cmp_mem(r::R9, r::R11, method);
                self.g_jcc(cc::E, proven);
                self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
                self.g_cmp_mem(r::R9, r::R11, stamp);
                self.g_jcc(cc::NE, stale);
                self.jmp(slow);
                self.bind(proven);
            } else {
                self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
                self.g_cmp_mem(r::R9, r::R11, stamp);
                self.g_jcc(cc::NE, stale);
                self.g_load(r::R9, r::R11, ptr);
                self.g_load(r::R9, r::R9, 0);
                self.g_cmp_mem(r::R9, r::R11, method);
                self.g_jcc(cc::NE, slow);
            }
            // ... of the proof's kind (with the default `[[Prototype]]`
            // unless the method was read).
            if callee.is_none() {
                self.g_cmp_byte_imm(r::RCX, c.default_proto, 1);
                self.g_jcc(cc::NE, slow);
            }
            self.g_mem(false, &[0x0f, 0xb6], r::R9, r::RCX, c.is_set);
            self.g_cmp_mem(r::R9, r::R11, is_set);
            // (A method read is the other kind's built-in only on a
            // receiver of the wrong kind: the call's `TypeError`.)
            self.g_jcc(cc::NE, if callee.is_some() { slow } else { stale });
            self.g_mem(true, &[0x83], 7, r::RCX, c.index); // cmp qword [rcx+index], imm8
            self.code.push(0);
            self.g_jcc(cc::NE, probe);
            self.g_load(r::RDI, r::RCX, c.entries_len);
            self.g_cmp_imm(
                r::RDI,
                i32::try_from(crate::realm::COLL_SCAN_MAX).expect("small"),
            );
            self.g_jcc(cc::A, probe);
        }
        self.g_load(r::RSI, r::RCX, c.entries_ptr);
        // r11 = the key's bits (`-0` as `+0`).
        let kbase = self.frame_base(key);
        self.coll_key_bits(kbase, key);
        let found = self.new_label();
        let absent = self.new_label();
        self.coll_scan(l, (found, absent, probe));
        self.bind(absent);
        let vbase = self.frame_base(val);
        match w {
            CollWrite::Delete => {
                self.g_movabs(r::RAX, FALSE_BITS);
                self.store_rax(dst);
                self.jmp(done);
            }
            CollWrite::MapSet | CollWrite::SetAdd => {
                // Appended in the `Vec`'s spare capacity, when the write
                // barrier has nothing to do for the key or the value.
                self.coll_container(l, realm_off, rbase, recv, probe);
                self.g_load(r::R8, r::RCX, c.entries_len);
                self.g_cmp_mem(r::R8, r::RCX, c.entries_cap);
                self.g_jcc(cc::AE, probe);
                if w == CollWrite::MapSet {
                    self.g_load(r::RAX, vbase, val);
                    self.p_barrier_check(l, realm_off, probe);
                }
                self.g_load(r::RAX, kbase, key);
                self.p_barrier_check(l, realm_off, probe);
                self.coll_key_bits(kbase, key);
                self.g_load(r::R8, r::RCX, c.entries_len);
                self.g_load(r::R9, r::RCX, c.entries_ptr);
                // imul rax, r8, entry_size; add r9, rax
                self.raw(&[0x49, 0x69, 0xc0]);
                self.raw(&c.entry_size.to_le_bytes());
                self.raw(&[0x49, 0x01, 0xc1]);
                self.g_store(r::R9, c.entry_key, r::R11);
                if w == CollWrite::MapSet {
                    self.g_load(r::RAX, vbase, val);
                    self.g_store(r::R9, c.entry_val, r::RAX);
                } else {
                    self.g_store(r::R9, c.entry_val, r::R11);
                }
                self.raw(&[0x49, 0xff, 0xc0]); // inc r8
                self.g_store(r::RCX, c.entries_len, r::R8);
                self.raw(&[0xff, 0x81]); // inc dword [rcx + disp32]
                self.raw(&c.live.to_le_bytes());
                self.g_load(r::RAX, rbase, recv);
                self.store_rax(dst);
                self.jmp(done);
            }
        }
        self.bind(found);
        match w {
            CollWrite::SetAdd => {}
            CollWrite::MapSet => {
                // The entry's value replaced, when the write barrier has
                // nothing to do for it (`rsi` is kept).
                self.coll_container(l, realm_off, rbase, recv, probe);
                self.g_load(r::RAX, vbase, val);
                self.p_barrier_check(l, realm_off, probe);
                self.g_load(r::RAX, vbase, val);
                self.g_store(r::RSI, c.entry_val, r::RAX);
            }
            CollWrite::Delete => {
                // A tombstone in place (a hole is no heap value: nothing for
                // the barrier), one live entry fewer.
                self.coll_container(l, realm_off, rbase, recv, probe);
                self.g_movabs(r::RAX, crate::nanbox::NanBox::hole().to_bits());
                self.g_store(r::RSI, c.entry_key, r::RAX);
                self.g_store(r::RSI, c.entry_val, r::RAX);
                self.raw(&[0xff, 0x89]); // dec dword [rcx + disp32]
                self.raw(&c.live.to_le_bytes());
                self.g_movabs(r::RAX, TRUE_BITS);
                self.store_rax(dst);
                self.jmp(done);
            }
        }
        self.g_load(r::RAX, rbase, recv);
        self.store_rax(dst);
        self.jmp(done);
        if let Some(helper) = probe_helper {
            // The helper's update (`jit_helper_coll_write`: the value, the
            // key for an `add`, a hole for a `delete`), which runs no code;
            // a miss (`hole`) is the call's, with nothing done.
            let cold = self.cold_begin();
            self.bind(probe);
            self.mov_rdi_r15();
            self.load_argreg(1, recv);
            self.load_argreg(2, key);
            match w {
                CollWrite::MapSet => self.load_argreg(3, val),
                CollWrite::SetAdd => self.load_argreg(3, key),
                CollWrite::Delete => {
                    self.movabs_argreg(3, crate::nanbox::NanBox::hole().to_bits() as i64);
                }
            }
            self.g_movabs(r::RAX, helper);
            self.call_rax();
            self.g_movabs(r::R11, crate::nanbox::NanBox::hole().to_bits());
            self.g_cmp(r::RAX, r::R11);
            self.g_jcc(cc::E, slow);
            self.store_rax(dst);
            self.jmp(done);
            self.cold_end(cold);
        }
        // A stale proof: made again (by the caller's `fill`) while the
        // site may still try.
        self.bind(stale);
        self.g_movabs(r::R11, site as u64);
        self.g_mem(true, &[0x83], 7, r::R11, tries); // cmp qword [r11+tries], imm8
        self.code.push(u8::try_from(COLL_TRIES).expect("an imm8"));
        self.g_jcc(cc::AE, slow);
        self.jmp(fill);
    }

    /// The read of a `Map.prototype` / `Set.prototype` method (`m.set`
    /// before its arguments, `Op::GetMethod`) on a strong collection of the
    /// proof's kind with the default `[[Prototype]]`: under the site's
    /// proof (see the module docs: the collection owns no property, and its
    /// prototype's slot holds the built-in), the method is that built-in,
    /// stored to the register at frame displacement `dst`, on to `done`.
    /// A stale proof is made again (`prove`, the fill helper, then
    /// `retry`) while the site may still try; anything else to `miss`,
    /// nothing done. With no `prove` (a fast version, which makes no call),
    /// a stale proof is a miss too. Clobbers `rax`, `rcx`, `rdx`, `r9`–`r11`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn coll_method_read(
        &mut self,
        l: &JitLayout,
        realm_off: i32,
        site: *const MethodSite,
        (recv, dst): (i32, i32),
        (retry, miss, done): (Label, Label, Label),
        prove: Option<u64>,
    ) {
        let c = coll_layout();
        let [stamp, ptr, method, is_set, tries] = MethodSite::coll_offsets();
        let stale = self.new_label();
        self.bind(retry);
        self.p_arena(realm_off, &l.props);
        // (The receiver first: any other kind is the usual miss.)
        let base = self.frame_base(recv);
        self.g_load(r::RAX, base, recv);
        self.p_is_handle(r::RCX, miss);
        self.p_slot_rooted(l, r::RCX, miss);
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, c.disc);
        self.g_jcc(cc::NE, miss);
        self.g_cmp_byte_imm(r::RCX, c.is_weak, 0);
        self.g_jcc(cc::NE, miss);
        self.g_cmp_byte_imm(r::RCX, c.default_proto, 1);
        self.g_jcc(cc::NE, miss);
        self.g_movabs(r::R11, site as u64);
        self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
        self.g_cmp_mem(r::R9, r::R11, stamp);
        self.g_jcc(cc::NE, if prove.is_some() { stale } else { miss });
        self.g_load(r::R9, r::R11, ptr);
        self.g_load(r::R9, r::R9, 0);
        self.g_cmp_mem(r::R9, r::R11, method);
        self.g_jcc(cc::NE, miss);
        self.g_mem(false, &[0x0f, 0xb6], r::R9, r::RCX, c.is_set);
        self.g_cmp_mem(r::R9, r::R11, is_set);
        self.g_jcc(cc::NE, if prove.is_some() { stale } else { miss });
        self.g_load(r::RAX, r::R11, method);
        self.store_rax(dst);
        self.jmp(done);
        if let Some(helper) = prove {
            let cold = self.cold_begin();
            self.bind(stale);
            self.g_movabs(r::R11, site as u64);
            self.g_mem(true, &[0x83], 7, r::R11, tries); // cmp qword [r11+tries], imm8
            self.code.push(u8::try_from(COLL_TRIES).expect("an imm8"));
            self.g_jcc(cc::AE, miss);
            self.mov_rdi_r15();
            self.load_argreg(1, recv);
            self.g_movabs(r::RDX, site as u64);
            self.g_movabs(r::RAX, helper);
            self.call_rax();
            self.g_test(r::RAX);
            self.g_jcc(cc::NE, retry);
            self.jmp(miss);
            self.cold_end(cold);
        }
    }

    /// `r11` = the bits of the key at `[base + disp]`, a `-0` as `+0`.
    fn coll_key_bits(&mut self, base: u8, disp: i32) {
        self.g_load(r::R11, base, disp);
        let normal = self.new_label();
        self.g_movabs(r::R9, crate::nanbox::NanBox::number(-0.0).to_bits());
        self.g_cmp(r::R11, r::R9);
        self.g_jcc(cc::NE, normal);
        self.g_movabs(r::R11, crate::nanbox::NanBox::number(0.0).to_bits());
        self.bind(normal);
    }

    /// `rcx` = the heap slot of the receiver at `[base + disp]`, a rooted
    /// handle already walked once (the arena loaded afresh; `miss` is never
    /// taken, nothing having run since). Clobbers `rax`, `r11` (and `r10`,
    /// `rdx`: the arena).
    fn coll_container(&mut self, l: &JitLayout, realm_off: i32, base: u8, disp: i32, miss: Label) {
        self.p_arena(realm_off, &l.props);
        self.g_load(r::RAX, base, disp);
        self.p_slot_rooted(l, r::RCX, miss);
    }

    /// The scan of an inline `Map`/`Set` call (see the module docs): for
    /// the key's bits in `r11` (`-0` already `+0`), the `rdi` entries from
    /// `rsi` (at most `COLL_SCAN_MAX`), to `found` with `rsi` = the entry
    /// whose key is the key, falling through when no entry's is, to
    /// `probe` when an entry or the key is compared out of line. Needs the
    /// arena (`p_arena`); clobbers `rax`, `rcx`, `rdi`, `r8`, `r9` (and
    /// `rsi`, but at `found`).
    fn coll_scan(&mut self, l: &JitLayout, (found, absent, probe): (Label, Label, Label)) {
        let c = coll_layout();
        let rl = l.rope;
        let by_bits = self.new_label();
        let texts = self.new_label();
        // Not a heap value, or compared by identity: the bits decide.
        self.g_mov(r::RAX, r::R11);
        self.p_cmp_handle_floor(r::RAX);
        self.g_jcc(cc::B, by_bits);
        // (A String first, the usual key here; else one compared by
        // identity.)
        let other = self.new_label();
        self.p_slot_rooted(l, r::RCX, probe);
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, l.cell_str_disc);
        self.g_jcc(cc::NE, other);
        // A String held in its node, short enough: `rcx` = its node.
        self.g_load(r::RCX, r::RCX, l.off_str_rope);
        self.g_cmp_byte_imm(r::RCX, rl.tag, rl.small_tag);
        self.g_jcc(cc::NE, probe);
        self.g_cmp_byte_imm_unsigned_above(r::RCX, rl.small_len, KEY_MAX, probe);
        self.jmp(texts);
        self.bind(other);
        self.p_identity_compared(by_bits, probe);
        self.jmp(probe);
        // The bits scan.
        let next = self.new_label();
        let lp = self.new_label();
        self.bind(by_bits);
        self.g_test(r::RDI);
        self.g_jcc(cc::E, absent);
        self.bind(lp);
        self.g_cmp_mem(r::R11, r::RSI, c.entry_key);
        self.g_jcc(cc::E, found);
        self.g_lea(r::RSI, r::RSI, c.entry_size);
        self.g_dec(r::RDI);
        self.g_jcc(cc::NE, lp);
        self.jmp(absent);
        // The texts scan: each entry's key the same bits, or a String with
        // the same text.
        let tlp = self.new_label();
        self.bind(texts);
        self.g_test(r::RDI);
        self.g_jcc(cc::E, absent);
        self.bind(tlp);
        self.g_load(r::RAX, r::RSI, c.entry_key);
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::E, found);
        self.p_cmp_handle_floor(r::RAX);
        self.g_jcc(cc::B, next); // (a tombstone's hole too)
        self.p_slot_rooted(l, r::R9, probe);
        self.g_cmp_byte_imm(r::R9, l.off_cell_tag, l.cell_str_disc);
        self.g_jcc(cc::NE, next);
        self.g_load(r::R9, r::R9, l.off_str_rope);
        let other = self.new_label();
        self.g_cmp_byte_imm(r::R9, rl.tag, rl.small_tag);
        self.g_jcc(cc::NE, other);
        // movzx eax, byte [r9 + len]; cmp al, [rcx + len]
        self.g_mem(false, &[0x0f, 0xb6], r::RAX, r::R9, rl.small_len);
        self.g_mem(false, &[0x3a], r::RAX, r::RCX, rl.small_len);
        self.g_jcc(cc::NE, next);
        for k in 0..3 {
            self.g_load(r::RAX, r::R9, rl.small_buf + 8 * k);
            self.g_cmp_mem(r::RAX, r::RCX, rl.small_buf + 8 * k);
            self.g_jcc(cc::NE, next);
        }
        self.jmp(found);
        // An entry String not held in its node: a flat one of another
        // length is another text; anything else is the helper's.
        let cold = self.cold_begin();
        self.bind(other);
        let flat = self.new_label();
        for (tag, len) in [(rl.leaf_tag, rl.leaf_len), (rl.build_tag, rl.build_len)] {
            let no = self.new_label();
            self.g_cmp_byte_imm(r::R9, rl.tag, tag);
            self.g_jcc(cc::NE, no);
            self.g_load(r::RAX, r::R9, len);
            self.jmp(flat);
            self.bind(no);
        }
        self.jmp(probe);
        self.bind(flat);
        self.g_mem(false, &[0x0f, 0xb6], r::R8, r::RCX, rl.small_len); // movzx r8d, byte
        self.g_cmp(r::RAX, r::R8);
        self.g_jcc(cc::NE, next);
        self.jmp(probe);
        self.cold_end(cold);
        self.bind(next);
        self.g_lea(r::RSI, r::RSI, c.entry_size);
        self.g_dec(r::RDI);
        self.g_jcc(cc::NE, tlp);
    }

    /// The probe of an inline `Map`/`Set` lookup on a collection with a
    /// hash index, as `Realm::coll_probe_lean` makes it (pinned code only:
    /// `rdx` and `r10` are free): for the key's bits in `r11` (`-0` already
    /// `+0`) and the collection's heap slot in `rcx`, a Number key (not a
    /// NaN) or a String key held in its node (at most [`KEY_MAX`] bytes)
    /// hashed as `Realm::collection_key_hash` hashes it, its slots probed as
    /// `CollIndex::find` probes them, each candidate entry's key compared
    /// with the key by its bits or, for a String, its text: to `found` with
    /// `rsi` = the entry whose key is the key, to `absent` when none is, to
    /// `probe` for anything else (another key, an entry String not held in
    /// its node). Clobbers `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`–`r10`.
    fn coll_hash_probe(
        &mut self,
        l: &JitLayout,
        (slots_ptr, slots_len): (i32, i32),
        (found, absent, probe): (Label, Label, Label),
    ) {
        debug_assert!(self.pinned());
        const K: u64 = 0x9E37_79B9_7F4A_7C15;
        let c = coll_layout();
        let rl = l.rope;
        let string = self.new_label();
        let mix = self.new_label();
        // rdi = the collection, r8 = its index.
        self.g_mov(r::RDI, r::RCX);
        self.g_load(r::R8, r::RCX, c.index);
        // The key: a heap value (a String), else a Number (not a NaN), else
        // the helper's.
        self.g_mov(r::RAX, r::R11);
        self.p_cmp_handle_floor(r::RAX);
        self.g_jcc(cc::AE, string);
        self.g_movabs(r::R9, QNAN_MASK);
        self.g_and(r::RAX, r::R9);
        self.g_cmp(r::RAX, r::R9);
        self.g_jcc(cc::E, probe);
        // shl rax, 1 (of the bits; the sign dropped): above +Inf's is a NaN.
        self.g_mov(r::RAX, r::R11);
        self.raw(&[0x48, 0xd1, 0xe0]);
        self.g_movabs(r::R9, 0xffe0_0000_0000_0000);
        self.g_cmp(r::RAX, r::R9);
        self.g_jcc(cc::A, probe);
        // `coll_num_hash`: the bits themselves (a `-0` already `+0`).
        self.g_mov(r::RDX, r::R11);
        self.jmp(mix);
        // A String held in its node: rcx = the node, rax = its length;
        // `coll_str_hash`'s words, then the mix.
        self.bind(string);
        self.p_slot_rooted(l, r::RCX, probe);
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, l.cell_str_disc);
        self.g_jcc(cc::NE, probe);
        self.g_load(r::RCX, r::RCX, l.off_str_rope);
        self.g_cmp_byte_imm(r::RCX, rl.tag, rl.small_tag);
        self.g_jcc(cc::NE, probe);
        self.g_cmp_byte_imm_unsigned_above(r::RCX, rl.small_len, KEY_MAX, probe);
        self.g_mem(false, &[0x0f, 0xb6], r::RAX, r::RCX, rl.small_len); // movzx eax, byte
        self.g_movabs(r::RDX, K);
        self.g_xor_rr(r::RDX, r::RAX);
        self.g_movabs(r::R9, K);
        for k in 0..3 {
            // Word `k` while the length is past `8k`.
            self.g_cmp_imm(r::RAX, 8 * k);
            self.g_jcc(cc::BE, mix);
            self.g_mem(true, &[0x33], r::RDX, r::RCX, rl.small_buf + 8 * k); // xor rdx, [..]
            self.g_imul_rr(r::RDX, r::R9);
            self.raw(&[0x48, 0xc1, 0xc2, 29]); // rol rdx, 29
        }
        // `coll_mix`, then `fold` (r10 = the 32 hash bits) and `home` (rdx).
        self.bind(mix);
        for m in [0xff51_afd7_ed55_8ccd_u64, 0xc4ce_b9fe_1a85_ec53] {
            self.g_mov(r::RAX, r::RDX);
            self.g_shr_imm(r::RAX, 33);
            self.g_xor_rr(r::RDX, r::RAX);
            self.g_movabs(r::R9, m);
            self.g_imul_rr(r::RDX, r::R9);
        }
        self.g_mov(r::RAX, r::RDX);
        self.g_shr_imm(r::RAX, 33);
        self.g_xor_rr(r::RDX, r::RAX);
        self.g_mov(r::R10, r::RDX);
        self.g_shr_imm(r::R10, 32);
        self.raw(&[0x41, 0x31, 0xd2]); // xor r10d, edx
        self.raw(&[0x41, 0x69, 0xd2]); // imul edx, r10d, imm32
        self.code.extend_from_slice(&0x9E37_79B9_u32.to_le_bytes());
        // An empty table finds nothing; else rdx = home & mask.
        self.g_load(r::RAX, r::R8, slots_len);
        self.g_test(r::RAX);
        self.g_jcc(cc::E, absent);
        self.raw(&[0x48, 0xff, 0xc8]); // dec rax
        self.g_and(r::RDX, r::RAX);
        // The probe: an empty slot ends it, a removed one is passed over,
        // a slot of the same hash bits names a candidate entry.
        let lp = self.new_label();
        let next = self.new_label();
        self.bind(lp);
        self.g_load(r::RAX, r::R8, slots_ptr);
        self.raw(&[0x48, 0x8b, 0x04, 0xd0]); // mov rax, [rax + rdx*8]
        self.g_test(r::RAX);
        self.g_jcc(cc::E, absent);
        self.raw(&[0x48, 0x83, 0xf8, 0xff]); // cmp rax, -1 (`GONE`)
        self.g_jcc(cc::E, next);
        self.g_mov(r::R9, r::RAX);
        self.g_shr_imm(r::R9, 32);
        self.raw(&[0x45, 0x39, 0xd1]); // cmp r9d, r10d
        self.g_jcc(cc::NE, next);
        // rsi = the entry at the slot's position (below the entry count).
        self.raw(&[0x89, 0xc0]); // mov eax, eax
        self.raw(&[0x48, 0xff, 0xc8]); // dec rax
        self.g_cmp_mem(r::RAX, r::RDI, c.entries_len);
        self.g_jcc(cc::AE, next);
        debug_assert_eq!(c.entry_size, 16);
        self.raw(&[0x48, 0xc1, 0xe0, 0x04]); // shl rax, 4
        self.g_add_mem(r::RAX, r::RDI, c.entries_ptr);
        self.g_mov(r::RSI, r::RAX);
        self.g_load(r::RAX, r::RSI, c.entry_key);
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::E, found);
        // A String key: an entry String held in its node of the same text
        // (an entry of any other kind is another key; an entry String not
        // in its node is the helper's).
        let num = self.new_label();
        self.g_mov(r::R9, r::R11);
        self.p_cmp_handle_floor(r::R9);
        self.g_jcc(cc::B, num);
        self.p_cmp_handle_floor(r::RAX);
        self.g_jcc(cc::B, next);
        self.p_slot_rooted(l, r::R9, probe);
        self.g_cmp_byte_imm(r::R9, l.off_cell_tag, l.cell_str_disc);
        self.g_jcc(cc::NE, next);
        self.g_load(r::R9, r::R9, l.off_str_rope);
        self.g_cmp_byte_imm(r::R9, rl.tag, rl.small_tag);
        self.g_jcc(cc::NE, probe);
        self.g_mem(false, &[0x0f, 0xb6], r::RAX, r::R9, rl.small_len);
        self.g_mem(false, &[0x3a], r::RAX, r::RCX, rl.small_len); // cmp al, [rcx + len]
        self.g_jcc(cc::NE, next);
        for k in 0..3 {
            self.g_load(r::RAX, r::R9, rl.small_buf + 8 * k);
            self.g_cmp_mem(r::RAX, r::RCX, rl.small_buf + 8 * k);
            self.g_jcc(cc::NE, next);
        }
        self.jmp(found);
        self.bind(num);
        // (A Number equals only an entry of its very bits.)
        self.bind(next);
        self.raw(&[0x48, 0xff, 0xc2]); // inc rdx
        self.g_load(r::RAX, r::R8, slots_len);
        self.raw(&[0x48, 0xff, 0xc8]); // dec rax
        self.g_and(r::RDX, r::RAX);
        self.jmp(lp);
    }

    /// `xor dst, src` (64-bit).
    fn g_xor_rr(&mut self, dst: u8, src: u8) {
        self.code
            .push(0x48 | (((src >> 3) & 1) << 2) | ((dst >> 3) & 1));
        self.code
            .extend_from_slice(&[0x31, 0xc0 | ((src & 7) << 3) | (dst & 7)]);
    }

    /// `imul dst, src` (64-bit).
    fn g_imul_rr(&mut self, dst: u8, src: u8) {
        self.code
            .push(0x48 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1));
        self.code
            .extend_from_slice(&[0x0f, 0xaf, 0xc0 | ((dst & 7) << 3) | (src & 7)]);
    }

    /// `cmp byte [base + disp], imm; ja miss`.
    fn g_cmp_byte_imm_unsigned_above(&mut self, base: u8, disp: i32, imm: u8, miss: Label) {
        self.g_cmp_byte_imm(base, disp, imm);
        self.g_jcc(cc::A, miss);
    }

    /// `dec reg` (64-bit).
    fn g_dec(&mut self, reg: u8) {
        self.code.push(0x48 | ((reg >> 3) & 1));
        self.code.extend_from_slice(&[0xff, 0xc8 | (reg & 7)]);
    }
}

/// The frame displacements of an `Op::IterNext`'s registers, for
/// [`X64Assembler::p_coll_iter_step`].
#[derive(Clone, Copy)]
pub(crate) struct IterRegs {
    pub(crate) dst: i32,
    pub(crate) done: i32,
    pub(crate) iter: i32,
    pub(crate) next: i32,
}

/// Whether a compiled `for-of` step over a live Map/Set keys/values
/// iterator runs inline (`X64Assembler::p_coll_iter_step`;
/// `KATAAN_JIT_COLLITER=0` turns it off, read once per process).
pub(crate) fn coll_iter_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_COLLITER").is_none_or(|v| v != "0"))
}

impl X64Assembler {
    /// One step of a `for-of` over a live Map/Set `keys` / `values`
    /// iterator, inline: what `Realm::collection_iter_advance_kv` does on
    /// its common path, for an iterator `nbexec::coll_iter_step_lean`
    /// would take — `next` the built-in one the last such step confirmed
    /// (`Realm::coll_iter_next`), `done` not `true`, the iterator an
    /// ordinary shaped object of the learned layout
    /// (`Realm::coll_iter_shape` / `coll_iter_slots`, neither frozen nor
    /// with a read-only property, both slots inline), its state an array
    /// of five elements whose `kind` is 0 or 1, whose `status` is 0 or 1
    /// and whose epoch is its collection's (no compaction since), and a
    /// live entry at or past the cursor (and the index's dead prefix).
    /// Then the step's value (the key, or for a Map's `values` the value)
    /// goes to `dst`, the cursor past the entry, the last key to the
    /// entry's key (when the write barrier has nothing to do for it,
    /// `p_barrier_check`) and the status to 1, extending the index's dead
    /// prefix as the Rust step does; and on to `body`. Anything else
    /// (another iterator, the end, an entries iterator, a renumbered
    /// cursor, a barrier with work) goes to `miss` with nothing done but
    /// `dst` written and the dead prefix extended (which the step there
    /// computes alike). Needs the arena pinned. Clobbers `rax`, `rcx`,
    /// `rdx`, `r8`–`r11`, `xmm0`.
    pub(crate) fn p_coll_iter_step(
        &mut self,
        l: &super::JitLayout,
        realm_off: i32,
        f: IterRegs,
        miss: Label,
        body: Label,
    ) {
        use crate::realm::Realm;
        use core::mem::offset_of;
        debug_assert!(self.pinned());
        let cl = *coll_layout();
        let o = |v: usize| i32::try_from(v).expect("realm offset fits in i32");
        let next_off = o(offset_of!(Realm, coll_iter_next));
        let shape_off = o(offset_of!(Realm, coll_iter_shape));
        let slots_off = o(offset_of!(Realm, coll_iter_slots));
        const ONE: u64 = 1.0f64.to_bits();
        let hole = crate::nanbox::NanBox::hole().to_bits();
        // `done` not `true`.
        self.g_load(r::RAX, r::RBX, f.done);
        self.cmp_gpr_const(r::RAX, TRUE_BITS);
        self.je(miss);
        // `next` the confirmed built-in (a handle: never 0).
        self.g_load(r::R11, r::R15, realm_off);
        self.g_load(r::RDX, r::RBX, f.next);
        self.g_test(r::RDX);
        self.je(miss);
        self.g_cmp_mem(r::RDX, r::R11, next_off);
        self.jne(miss);
        // The iterator: an ordinary shaped object of the layout's shape.
        self.g_load(r::RAX, r::RBX, f.iter);
        self.p_is_handle(r::RCX, miss);
        self.p_slot_rooted(l, r::RCX, miss);
        self.p_object_not_exotic(l, miss);
        self.p_test_byte_imm(
            r::RCX,
            l.props.off_jit_flags,
            crate::object::JIT_FLAG_READONLY | crate::object::OBJ_FLAG_FROZEN,
        );
        self.jne(miss);
        self.g_load(r::RDX, r::R11, shape_off);
        self.g_test(r::RDX);
        self.je(miss);
        self.g_cmp_mem(r::RDX, r::RCX, l.off_shape);
        self.jne(miss);
        // r10 = its collection, rax = its state (inline slots only).
        self.g_mem(false, &[0x8b], r::RAX, r::R11, slots_off); // mov eax, [r11 + slots]
        self.p_obj_slot_load_inline(l, miss, miss);
        self.g_mov(r::R10, r::RAX);
        self.g_load(r::R11, r::R15, realm_off);
        self.g_mem(false, &[0x8b], r::RAX, r::R11, slots_off + 4); // mov eax, [r11 + slots + 4]
        self.p_obj_slot_load_inline(l, miss, miss);
        // The state: an array of five elements (rcx its slot, r11 them).
        self.p_is_handle(r::RCX, miss);
        self.p_slot_rooted(l, r::RCX, miss);
        self.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
        self.jne(miss);
        self.p_arr_len_ptr(l);
        self.g_cmp_imm(r::R8, 5);
        self.jne(miss);
        // kind 0 or 1, status 0 or 1 (their bits).
        let kind_ok = self.new_label();
        self.g_load(r::RDX, r::R11, 0);
        self.g_test(r::RDX);
        self.je(kind_ok);
        self.cmp_gpr_const(r::RDX, ONE);
        self.jne(miss);
        self.bind(kind_ok);
        let status_ok = self.new_label();
        self.g_load(r::RDX, r::R11, 32);
        self.g_test(r::RDX);
        self.je(status_ok);
        self.cmp_gpr_const(r::RDX, ONE);
        self.jne(miss);
        self.bind(status_ok);
        // r10 = the cursor (an integer below 2^32: else the Rust step's);
        // rax = the collection's bits.
        self.g_mov(r::RAX, r::R10);
        self.raw(&[0xf2, 0x41, 0x0f, 0x10, 0x43, 0x08]); // movsd xmm0, [r11 + 8]
        self.raw(&[0xf2, 0x4c, 0x0f, 0x2c, 0xd0]); // cvttsd2si r10, xmm0
        self.g_mov(r::RDX, r::R10);
        self.g_shr_imm(r::RDX, 32);
        self.jne(miss);
        // rdx = the state's epoch bits.
        self.g_load(r::RDX, r::R11, 16);
        // The collection (r9 its slot), of the state's epoch.
        self.p_is_handle(r::RCX, miss);
        self.p_slot_rooted(l, r::R9, miss);
        self.g_cmp_byte_imm(r::R9, l.off_cell_tag, cl.disc);
        self.jne(miss);
        self.g_mem(false, &[0x8b], r::RAX, r::R9, cl.epoch); // mov eax, [r9 + epoch]
        self.raw(&[0xf2, 0x48, 0x0f, 0x2a, 0xc0]); // cvtsi2sd xmm0, rax
        self.raw(&[0x66, 0x48, 0x0f, 0x7e, 0xc0]); // movq rax, xmm0
        self.g_cmp(r::RAX, r::RDX);
        self.jne(miss);
        // xmm0 = the step's mode: 0 to yield the key (keys, or a Set),
        // else the value; read while r11 is the state.
        let mode_set = self.new_label();
        self.g_load(r::RAX, r::R11, 0);
        self.g_cmp_byte_imm(r::R9, cl.is_set, 0);
        self.je(mode_set);
        self.raw(&[0x31, 0xc0]); // xor eax, eax
        self.bind(mode_set);
        self.raw(&[0x66, 0x48, 0x0f, 0x6e, 0xc0]); // movq xmm0, rax
        // from = max(cursor, index head); rdx = the index's box when the
        // head is to be extended to the entry found (cursor <= head).
        let no_index = self.new_label();
        let past_head = self.new_label();
        let from_set = self.new_label();
        self.g_load(r::RDX, r::R9, cl.index);
        self.raw(&[0x45, 0x31, 0xc0]); // xor r8d, r8d
        self.g_test(r::RDX);
        self.je(no_index);
        self.g_mem(false, &[0x8b], r::R8, r::RDX, cl.index_head); // mov r8d, [rdx + head]
        self.bind(no_index);
        self.g_cmp(r::R10, r::R8);
        self.g_jcc(cc::A, past_head);
        self.g_mov(r::R10, r::R8);
        self.jmp(from_set);
        self.bind(past_head);
        self.raw(&[0x31, 0xd2]); // xor edx, edx
        self.bind(from_set);
        // The scan: r11 = &entries[from], r8 = the entry count; the end
        // is the Rust step's.
        self.g_load(r::R8, r::R9, cl.entries_len);
        self.g_mov(r::R11, r::R10);
        debug_assert_eq!(cl.entry_size, 16);
        self.raw(&[0x49, 0xc1, 0xe3, 0x04]); // shl r11, 4
        self.g_add_mem(r::R11, r::R9, cl.entries_ptr);
        let scan = self.new_label();
        let found = self.new_label();
        self.bind(scan);
        self.g_cmp(r::R10, r::R8);
        self.g_jcc(cc::AE, miss);
        self.g_load(r::RAX, r::R11, cl.entry_key);
        self.cmp_gpr_const(r::RAX, hole);
        self.jne(found);
        self.raw(&[0x49, 0xff, 0xc2]); // inc r10
        self.raw(&[0x49, 0x83, 0xc3, 0x10]); // add r11, 16
        self.jmp(scan);
        self.bind(found);
        // The dead prefix extended to the entry (r10 its position).
        let head_kept = self.new_label();
        self.g_test(r::RDX);
        self.je(head_kept);
        self.g_mem(false, &[0x89], r::R10, r::RDX, cl.index_head); // mov [rdx + head], r10d
        self.bind(head_kept);
        // dst = the key (rax) or the entry's value.
        let key_mode = self.new_label();
        let have = self.new_label();
        self.raw(&[0x66, 0x49, 0x0f, 0x7e, 0xc0]); // movq r8, xmm0
        self.g_test(r::R8);
        self.je(key_mode);
        self.g_load(r::R8, r::R11, cl.entry_val);
        self.jmp(have);
        self.bind(key_mode);
        self.g_mov(r::R8, r::RAX);
        self.bind(have);
        self.g_store(r::RBX, f.dst, r::R8);
        // xmm0 = the cursor past the entry.
        self.raw(&[0x4d, 0x8d, 0x42, 0x01]); // lea r8, [r10 + 1]
        self.raw(&[0xf2, 0x49, 0x0f, 0x2a, 0xc0]); // cvtsi2sd xmm0, r8
        // The last key's store into the state (rcx): nothing for the
        // barrier to do, else the Rust step.
        let pop_miss = self.new_label();
        self.raw(&[0x50]); // push rax
        self.p_barrier_check(l, realm_off, pop_miss);
        self.raw(&[0x58]); // pop rax
        self.p_arr_len_ptr(l);
        self.raw(&[0xf2, 0x41, 0x0f, 0x11, 0x43, 0x08]); // movsd [r11 + 8], xmm0
        self.g_store(r::R11, 24, r::RAX);
        self.g_movabs(r::R8, ONE);
        self.g_store(r::R11, 32, r::R8);
        self.jmp(body);
        self.bind(pop_miss);
        self.raw(&[0x58]); // pop rax
        self.jmp(miss);
    }
}
