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
            self.g_movabs(r::R9, crate::object::proto_epoch_addr() as u64);
            self.g_load(r::R9, r::R9, 0);
            self.g_cmp_mem(r::R9, r::R11, stamp);
            self.g_jcc(cc::NE, stale);
            self.jmp(slow);
            self.bind(proven);
        } else {
            // The proof, at the current epoch ...
            self.g_movabs(r::R9, crate::object::proto_epoch_addr() as u64);
            self.g_load(r::R9, r::R9, 0);
            self.g_cmp_mem(r::R9, r::R11, stamp);
            self.g_jcc(cc::NE, stale);
            // ... with the built-in still in the prototype's slot.
            self.g_load(r::R9, r::R11, ptr);
            self.g_load(r::R9, r::R9, 0);
            self.g_cmp_mem(r::R9, r::R11, method);
            self.g_jcc(cc::NE, slow);
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
        self.g_mem(true, &[0x83], 7, r::RCX, c.index); // cmp qword [rcx+index], imm8
        self.code.push(0);
        self.g_jcc(cc::NE, probe);
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
        let found = self.new_label();
        let absent = self.new_label();
        self.coll_scan(l, (found, absent, probe));
        // The answers.
        self.bind(absent);
        self.g_movabs(r::RAX, if get { UNDEF_BITS } else { FALSE_BITS });
        self.store_rax(dst);
        self.jmp(done);
        self.bind(found);
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
        (recv, key, val, dst): (i32, i32, i32, i32),
        (fill, slow, done): (Label, Label, Label),
    ) {
        let c = coll_layout();
        let [stamp, ptr, method, is_set, tries] = MethodSite::coll_offsets();
        let stale = self.new_label();
        self.p_arena(realm_off, &l.props);
        // The receiver first (an indexed or larger collection, whose writes
        // are never inline, costs no more): a strong collection with the
        // default `[[Prototype]]`, no index, few entries.
        let rbase = self.frame_base(recv);
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
        // The proof, at the current epoch, with the built-in still in the
        // prototype's slot, for the receiver's kind.
        self.g_movabs(r::R11, site as u64);
        self.g_movabs(r::R9, crate::object::proto_epoch_addr() as u64);
        self.g_load(r::R9, r::R9, 0);
        self.g_cmp_mem(r::R9, r::R11, stamp);
        self.g_jcc(cc::NE, stale);
        self.g_load(r::R9, r::R11, ptr);
        self.g_load(r::R9, r::R9, 0);
        self.g_cmp_mem(r::R9, r::R11, method);
        self.g_jcc(cc::NE, slow);
        self.g_mem(false, &[0x0f, 0xb6], r::R9, r::RCX, c.is_set);
        self.g_cmp_mem(r::R9, r::R11, is_set);
        self.g_jcc(cc::NE, stale);
        self.g_load(r::RSI, r::RCX, c.entries_ptr);
        // r11 = the key's bits (`-0` as `+0`).
        let kbase = self.frame_base(key);
        self.coll_key_bits(kbase, key);
        let found = self.new_label();
        let absent = self.new_label();
        self.coll_scan(l, (found, absent, slow));
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
                self.coll_container(l, realm_off, rbase, recv, slow);
                self.g_load(r::R8, r::RCX, c.entries_len);
                self.g_cmp_mem(r::R8, r::RCX, c.entries_cap);
                self.g_jcc(cc::AE, slow);
                if w == CollWrite::MapSet {
                    self.g_load(r::RAX, vbase, val);
                    self.p_barrier_check(l, realm_off, slow);
                }
                self.g_load(r::RAX, kbase, key);
                self.p_barrier_check(l, realm_off, slow);
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
                self.coll_container(l, realm_off, rbase, recv, slow);
                self.g_load(r::RAX, vbase, val);
                self.p_barrier_check(l, realm_off, slow);
                self.g_load(r::RAX, vbase, val);
                self.g_store(r::RSI, c.entry_val, r::RAX);
            }
            CollWrite::Delete => {
                // A tombstone in place (a hole is no heap value: nothing for
                // the barrier), one live entry fewer.
                self.coll_container(l, realm_off, rbase, recv, slow);
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
        // A stale proof: made again (by the caller's `fill`) while the
        // site may still try.
        self.bind(stale);
        self.g_movabs(r::R11, site as u64);
        self.g_mem(true, &[0x83], 7, r::R11, tries); // cmp qword [r11+tries], imm8
        self.code.push(u8::try_from(COLL_TRIES).expect("an imm8"));
        self.g_jcc(cc::AE, slow);
        self.jmp(fill);
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
