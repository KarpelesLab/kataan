//! The String-key text memo: "the key text T is slot S of layout L", or
//! "T is on no object of the chain of a receiver of layout L whose
//! `[[Prototype]]` is P, at prototype epoch E".
//!
//! The slot memo (`key_memo`) finds its entry by the key String's rope, so
//! only the very String it recorded hits: a key made fresh for each access
//! (`o['k' + i]`) has a new rope every time and paid the full lookup (the
//! key's UTF-8 check, a hash and probe of the layout's index, the memo's
//! note) on every read and write. This memo is found by the key's **text**:
//! a short flat key (at most [`TEXT_MAX`] bytes) is read as three words,
//! hashed, and compared whole with the entry's, so any String with that text
//! hits. An entry's text was checked when it was recorded (well-formed
//! UTF-8, no leading NUL: its own storage key), so an equal text needs no
//! check again.
//!
//! It also remembers **absent** keys, which the slot memo cannot: `k in o`
//! false and `o[k]` undefined over an ordinary chain otherwise walked every
//! object and, for a read, went the general (host) way. An absent entry
//! records the receiver's layout (which fixes its data keys), its
//! `[[Prototype]]` and the prototype epoch, having flagged every object of
//! the chain from that prototype up watched (`Object::watch`): each one was
//! then an ordinary shaped object (not host-exotic, no primitive wrapper)
//! without the key as a data property or accessor, and the last one's
//! `[[Prototype]]` was `null`. A change to any watched object's keys,
//! accessors, layout or `[[Prototype]]` bumps the epoch, so while it stands
//! the chain is as recorded. The receiver itself is checked on every hit as
//! the slot memo checks it: shaped with the entry's layout, no accessors, no
//! exotic flag, not host-exotic.
//!
//! An entry holds its layout, so the layout's address cannot be reused while
//! it stands; the prototype's handle carries its generation. The memo is
//! direct-mapped by the text's hash and records on every noted lookup (one
//! made for a single read costs a compare and a few stores).

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::object::Object;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::Cell as Slot;

/// The number of entries (a power of two).
const CAP: usize = 512;

/// The longest key text the memo records, in bytes.
pub(super) const TEXT_MAX: usize = 24;

/// [`Entry::slot`] of an absent entry.
const ABSENT: u32 = u32::MAX;

/// [`Entry::slot`] of an inherited-accessor entry: the key is on no object
/// of the chain below [`Entry::holder`], which has it as an accessor (an
/// absent entry's proof up to the holder). Past any slot count, so code
/// that reads an entry's slot without knowing this kind (compiled code's
/// `tp_probe`) bounds-checks it as a miss.
const PROTO_ACC: u32 = u32::MAX - 1;

/// [`Entry::slot`] of an own-accessor entry: the key is no data key of the
/// layout, and a receiver read with it had an accessor of that name (which
/// each read checks again: an entry says nothing of other receivers'
/// accessors). Lets compiled code's probe (`tp_probe_acc`) find such a
/// receiver's getter; anything else takes it for a miss.
const OWN_ACC: u32 = u32::MAX - 2;

/// [`Entry::proto`] for a `null` `[[Prototype]]` (no handle's bits).
const NO_PROTO: u64 = u64::MAX;

/// A key text as the memo compares it: its bytes in three little-endian
/// words (zero past the end) and its length.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Text {
    pub(super) words: [u64; 3],
    pub(super) len: u32,
}

impl Text {
    /// The text of `b`, when it is short enough.
    #[inline]
    pub(super) fn of(b: &[u8]) -> Option<Self> {
        let n = b.len();
        if n > TEXT_MAX {
            return None;
        }
        // (Whole words, then the tail from at most three overlapping
        // reads: no copy loop for these few bytes.)
        let word = |i: usize| {
            let mut x = [0u8; 8];
            x.copy_from_slice(&b[i..i + 8]);
            u64::from_le_bytes(x)
        };
        let mut words = [0u64; 3];
        let full = n / 8;
        for (i, w) in words.iter_mut().enumerate().take(full) {
            *w = word(i * 8);
        }
        let (base, r) = (full * 8, n % 8);
        if r != 0 {
            let t = &b[base..];
            let tail = if r >= 4 {
                let lo = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
                let hi = u32::from_le_bytes([t[r - 4], t[r - 3], t[r - 2], t[r - 1]]);
                u64::from(lo) | u64::from(hi) << (8 * (r - 4))
            } else {
                u64::from(t[0])
                    | u64::from(t[r / 2]) << (8 * (r / 2))
                    | u64::from(t[r - 1]) << (8 * (r - 1))
            };
            words[full] = tail;
        }
        Some(Self {
            words,
            len: n as u32,
        })
    }

    /// The entry index of the text. (Compiled code computes it the same
    /// way, `crate::jit::key_probe`: keep the two in step.)
    #[inline]
    fn index(&self) -> usize {
        let h = (self.words[0]
            ^ self.words[1].rotate_left(ROT1)
            ^ self.words[2].rotate_left(ROT2)
            ^ u64::from(self.len))
        .wrapping_mul(HASH_MUL);
        (h >> 32) as usize & (CAP - 1)
    }
}

/// [`Text::index`]'s rotations of the second and third words.
const ROT1: u32 = 21;
const ROT2: u32 = 42;
/// [`Text::index`]'s multiplier.
const HASH_MUL: u64 = 0x9e37_79b9_7f4a_7c15;

/// Where compiled code finds an entry (`crate::jit::key_probe`): the
/// realm's field holding the entries' address (`0` before the memo is
/// made), the entry stride, the text's three words and length, and the
/// entry's layout, slot, prototype and stamp fields; and how it computes an
/// entry's index ([`Text::index`]): the rotations, the multiplier and the
/// mask (`CAP - 1`). A key held in its rope node (`Node::Small`, whose bytes
/// past its length are zero) has its text's words at its buffer's start.
#[cfg(all(feature = "jit", jit_native))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeyTextLayout {
    pub(crate) entries: i32,
    pub(crate) size: i32,
    pub(crate) words: i32,
    pub(crate) len: i32,
    pub(crate) shape: i32,
    pub(crate) slot: i32,
    pub(crate) proto: i32,
    pub(crate) stamp: i32,
    pub(crate) rot: (u8, u8),
    pub(crate) mul: u64,
    pub(crate) mask: i32,
    /// The longest text recorded, in bytes.
    pub(crate) text_max: u8,
    /// An inherited-accessor entry's slot ([`PROTO_ACC`]) as a sign-extended
    /// byte: compiled code takes it for a miss.
    pub(crate) proto_acc: i8,
    /// An own-accessor entry's slot ([`OWN_ACC`]), likewise.
    pub(crate) own_acc: i8,
}

/// Derives [`KeyTextLayout`].
#[cfg(all(feature = "jit", jit_native))]
#[must_use]
pub(crate) fn jit_key_text_layout() -> KeyTextLayout {
    use core::mem::{offset_of, size_of};
    let i = |v: usize| i32::try_from(v).expect("text memo offset fits in i32");
    const _: () = assert!(CAP.is_power_of_two() && TEXT_MAX <= 24);
    // (A `Cell` is its value: the entry's fields are where their values are.)
    let text = offset_of!(Entry, text);
    KeyTextLayout {
        entries: i(offset_of!(Realm, key_text_entries)),
        size: i(size_of::<Entry>()),
        words: i(text + offset_of!(Text, words)),
        len: i(text + offset_of!(Text, len)),
        shape: i(offset_of!(Entry, shape)),
        slot: i(offset_of!(Entry, slot)),
        proto: i(offset_of!(Entry, proto)),
        stamp: i(offset_of!(Entry, stamp)),
        rot: (ROT1 as u8, ROT2 as u8),
        mul: HASH_MUL,
        mask: i(CAP - 1),
        text_max: TEXT_MAX as u8,
        proto_acc: {
            const _: () = assert!(PROTO_ACC as i32 == -2);
            -2
        },
        own_acc: {
            const _: () = assert!(OWN_ACC as i32 == -3);
            -3
        },
    }
}

/// One memoized answer (see the module docs). `shape == 0`: empty.
#[derive(Default)]
struct Entry {
    text: Slot<Text>,
    /// The layout's address.
    shape: Slot<usize>,
    /// The key's slot in the layout, or [`ABSENT`].
    slot: Slot<u32>,
    /// For an absent entry: the receiver's `[[Prototype]]` (raw handle) and
    /// the prototype epoch.
    proto: Slot<u64>,
    stamp: Slot<u64>,
    /// For an inherited-accessor entry ([`PROTO_ACC`]): the holder's raw
    /// handle (it carries its generation).
    holder: Slot<u64>,
    /// What keeps the layout's address from being reused.
    hold: Slot<Option<Rc<Shape>>>,
}

/// The realm's String-key text memo (see the module docs).
pub(crate) struct KeyText {
    entries: Box<[Entry]>,
}

impl Default for KeyText {
    fn default() -> Self {
        Self {
            entries: (0..CAP).map(|_| Entry::default()).collect(),
        }
    }
}

/// What the memo knows of a String key on a receiver.
pub(crate) enum TextHit<'a> {
    /// An own data property: the receiver and the key's slot.
    Slot(&'a Object, u32),
    /// On no object of the receiver's chain.
    Absent,
}

impl Realm {
    /// The memo, made (and its entries' address published for compiled
    /// code, `Realm::key_text_entries`) on first use.
    #[inline]
    fn key_text_memo(&self) -> &KeyText {
        self.key_text.get_or_init(|| {
            let m: Box<KeyText> = Box::default();
            self.key_text_entries.set(m.entries.as_ptr() as usize);
            m
        })
    }

    /// The text of the String key `key` when the memo may answer for it: a
    /// flat leaf short enough, with no auxiliary object (which may make it
    /// no plain String, see `string_key_in_place`).
    #[inline]
    fn key_text_of(&self, key: NanBox) -> Option<Text> {
        let raw = key.as_handle()?;
        let b = self
            .heap
            .get(Handle::from_raw(raw))?
            .as_str()?
            .as_leaf_bytes()?;
        let t = Text::of(b)?;
        if self.str_special && self.aux_props.contains_key(&raw) {
            return None;
        }
        Some(t)
    }

    /// The receiver at `obj` when the memo may answer for it: a shaped
    /// ordinary object without accessors or the exotic flag, not
    /// host-exotic, and its layout.
    #[inline]
    fn key_text_receiver(&self, obj: NanBox) -> Option<(&Object, &Rc<Shape>)> {
        let h = Handle::from_raw(obj.as_handle()?);
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        if !o.has_no_accessors()
            || o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || self.is_host_exotic(h)
        {
            return None;
        }
        Some((o, shape))
    }

    /// What the memo knows of `obj[key]` / `key in obj` for the String key
    /// `key` (see the module docs), or `None`: the ordinary lookup decides.
    #[inline]
    pub(crate) fn key_text_probe(&self, obj: NanBox, key: NanBox) -> Option<TextHit<'_>> {
        let memo = self.key_text.get()?;
        let t = self.key_text_of(key)?;
        let e = &memo.entries[t.index()];
        if e.text.get() != t {
            return None;
        }
        let (o, shape) = self.key_text_receiver(obj)?;
        if Rc::as_ptr(shape) as usize != e.shape.get() {
            return None;
        }
        let slot = e.slot.get();
        if slot == PROTO_ACC || slot == OWN_ACC {
            return None;
        }
        if slot != ABSENT {
            return Some(TextHit::Slot(o, slot));
        }
        (o.proto().map_or(NO_PROTO, Handle::to_raw) == e.proto.get()
            && crate::object::proto_epoch() == e.stamp.get())
        .then_some(TextHit::Absent)
    }

    /// Notes that the String key `key`, whose text is its own storage key,
    /// is slot `slot` of `shape` (a layout with no internal wrapper or
    /// `arguments` key, of a receiver without accessors).
    #[inline]
    pub(crate) fn key_text_note(&self, key: NanBox, shape: &Rc<Shape>, slot: u32) {
        let Some(t) = self.key_text_of(key) else {
            return;
        };
        let memo = self.key_text_memo();
        let e = &memo.entries[t.index()];
        let p = Rc::as_ptr(shape) as usize;
        if e.text.get() == t && e.shape.get() == p && e.slot.get() == slot {
            return;
        }
        e.text.set(t);
        e.slot.set(slot);
        if e.shape.get() != p {
            e.shape.set(p);
            drop(e.hold.replace(Some(Rc::clone(shape))));
        }
    }

    /// Whether the String key `key` is on no object of `obj`'s prototype
    /// chain, decided over an ordinary chain (see the module docs) and, when
    /// so, recorded (flagging the chain's prototypes watched). `false`: it
    /// is present, or this cannot tell.
    pub(crate) fn key_text_note_absent(&mut self, obj: NanBox, key: NanBox) -> bool {
        matches!(
            self.key_text_note_chain(obj, key, false),
            Some(ChainNote::Absent)
        )
    }

    /// [`key_text_note_absent`](Self::key_text_note_absent), which also
    /// records (when `acc`) a key that the chain's first holder has as an
    /// accessor ([`PROTO_ACC`]): the walk proves the levels below it as an
    /// absent entry proves them all, and flags them and the holder watched
    /// (a watched object's accessor list changing bumps the epoch too). The
    /// receiver may have accessors (`KATAAN_JIT_GETKEY`), but not one of the
    /// key's name.
    fn key_text_note_chain(&mut self, obj: NanBox, key: NanBox, acc: bool) -> Option<ChainNote> {
        const LEVELS: usize = 8;
        let t = self.key_text_of(key)?;
        let k = self
            .heap
            .get(Handle::from_raw(key.as_handle()?))
            .and_then(Cell::as_str)
            .and_then(|r| r.as_leaf_bytes())
            .filter(|b| b.first() != Some(&0))
            .and_then(|b| core::str::from_utf8(b).ok())?;
        let (o, shape) = self.key_text_receiver_any(obj)?;
        if shape.is_wrapper()
            || shape.lookup(k).is_some()
            || (!o.has_no_accessors() && (!getkey_enabled() || o.accessor(k).is_some()))
        {
            return None;
        }
        let (shape, proto) = (Rc::clone(shape), o.proto());
        let mut levels = [None; LEVELS];
        let mut cur = proto;
        let mut n = 0;
        let mut holder = None;
        while let Some(h) = cur {
            if n == LEVELS {
                return None;
            }
            let Some(Cell::Object(p)) = self.heap.get(h) else {
                return None;
            };
            let ps = p.shaped_shape()?;
            if ps.is_wrapper() || self.is_host_exotic(h) || ps.lookup(k).is_some() {
                return None;
            }
            levels[n] = Some(h);
            n += 1;
            if !p.has_no_accessors() && p.accessor(k).is_some() {
                if !acc || p.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0 {
                    return None;
                }
                holder = Some(h);
                break;
            }
            cur = p.proto();
        }
        for h in levels.iter().flatten() {
            if let Some(p) = self.heap.get_mut(*h).and_then(Cell::as_object_mut) {
                p.watch();
            }
        }
        let memo = self.key_text_memo();
        let e = &memo.entries[t.index()];
        e.text.set(t);
        e.slot
            .set(if holder.is_some() { PROTO_ACC } else { ABSENT });
        e.holder.set(holder.map_or(0, Handle::to_raw));
        e.proto.set(proto.map_or(NO_PROTO, Handle::to_raw));
        e.stamp.set(crate::object::proto_epoch());
        let p = Rc::as_ptr(&shape) as usize;
        if e.shape.get() != p {
            e.shape.set(p);
            drop(e.hold.replace(Some(shape)));
        }
        Some(if holder.is_some() {
            ChainNote::Accessor
        } else {
            ChainNote::Absent
        })
    }

    /// The receiver at `obj` when [`key_text_probe_acc`](Self::key_text_probe_acc)
    /// may answer for it: a shaped ordinary object without the exotic flag
    /// (which every host-exotic object carries, `Realm::mark_host_exotic`)
    /// and no wrapper layout, accessors allowed when `KATAAN_JIT_GETKEY` is
    /// on (each answer then checks the receiver has none of the key's
    /// name). A mapped `arguments` object's layout carries its internal key,
    /// so no entry names it, and its own accessor is an accessor like any.
    #[inline]
    fn key_text_receiver_any(&self, obj: NanBox) -> Option<(&Object, &Rc<Shape>)> {
        let Some(Cell::Object(o)) = self.heap.get(Handle::from_raw(obj.as_handle()?)) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        if o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || shape.is_wrapper()
            || (!o.has_no_accessors() && !getkey_enabled())
        {
            return None;
        }
        Some((o, shape))
    }

    /// `obj[key]` for the String key `key` decided without running
    /// anything but a getter, for a receiver that may have accessors
    /// (`KATAAN_JIT_GETKEY`): the receiver's own accessor of the key's name
    /// (its getter), else what the memo knows for the receiver's layout —
    /// an own data slot's value, `undefined` for a key on no object of the
    /// chain, or the getter of the inherited accessor an entry records.
    /// `None`: the ordinary lookup decides.
    pub(crate) fn key_text_probe_acc(&self, obj: NanBox, key: NanBox) -> Option<AccHit<'_>> {
        if !getkey_enabled() {
            return None;
        }
        let raw = key.as_handle()?;
        let b = self
            .heap
            .get(Handle::from_raw(raw))?
            .as_str()?
            .as_leaf_bytes()?;
        if b.first() == Some(&0) || (self.str_special && self.aux_props.contains_key(&raw)) {
            return None;
        }
        let (o, shape) = self.key_text_receiver_any(obj)?;
        if !o.has_no_accessors()
            && let Some((g, _)) = o.accessor_bytes(b)
        {
            self.key_text_note_own_acc(b, shape);
            return Some(AccHit::Getter(g));
        }
        let memo = self.key_text.get()?;
        let t = Text::of(b)?;
        let e = &memo.entries[t.index()];
        if e.text.get() != t || Rc::as_ptr(shape) as usize != e.shape.get() {
            return None;
        }
        let slot = e.slot.get();
        if slot == OWN_ACC {
            return None;
        }
        if slot != ABSENT && slot != PROTO_ACC {
            return Some(AccHit::Slot(o, slot));
        }
        if o.proto().map_or(NO_PROTO, Handle::to_raw) != e.proto.get()
            || crate::object::proto_epoch() != e.stamp.get()
        {
            return None;
        }
        if slot == ABSENT {
            return Some(AccHit::Absent);
        }
        let Some(Cell::Object(p)) = self.heap.get(Handle::from_raw(e.holder.get())) else {
            return None;
        };
        p.accessor_bytes(b).map(|(g, _)| AccHit::Getter(g))
    }

    /// Records that the key text `b` is no data key of `shape` (a layout of
    /// a receiver found to have an accessor of that name: [`OWN_ACC`]), so
    /// that compiled code's probe finds the next such receiver's getter.
    /// (A layout with the key as data keeps or gets its slot's entry.)
    fn key_text_note_own_acc(&self, b: &[u8], shape: &Rc<Shape>) {
        let Some(t) = Text::of(b) else {
            return;
        };
        let memo = self.key_text_memo();
        let e = &memo.entries[t.index()];
        let p = Rc::as_ptr(shape) as usize;
        if e.text.get() == t && e.shape.get() == p && e.slot.get() == OWN_ACC {
            return;
        }
        if core::str::from_utf8(b).is_ok_and(|k| shape.lookup(k).is_some()) || shape.is_wrapper() {
            return;
        }
        e.text.set(t);
        e.slot.set(OWN_ACC);
        if e.shape.get() != p {
            e.shape.set(p);
            drop(e.hold.replace(Some(Rc::clone(shape))));
        }
    }

    /// Records what [`key_text_probe_acc`](Self::key_text_probe_acc) missed
    /// for `obj[key]` (`own_data_lookup` records an own data slot): a key on
    /// no object of the chain, or one a prototype has as an accessor.
    #[cfg_attr(not(all(feature = "jit", jit_native)), allow(dead_code))]
    pub(crate) fn key_text_note_acc(&mut self, obj: NanBox, key: NanBox) {
        if getkey_enabled() {
            let _ = self.key_text_note_chain(obj, key, true);
        }
    }
}

/// What [`Realm::key_text_note_chain`] recorded.
enum ChainNote {
    Absent,
    Accessor,
}

/// What [`Realm::key_text_probe_acc`] found. (The values are read by
/// compiled code's helpers alone.)
#[cfg_attr(not(all(feature = "jit", jit_native)), allow(dead_code))]
pub(crate) enum AccHit<'a> {
    /// An own data property: the receiver and the key's slot in its
    /// layout.
    Slot(&'a Object, u32),
    /// On no object of the receiver's chain (`undefined`).
    Absent,
    /// An accessor's getter (`undefined` for none): the read calls it with
    /// the receiver as `this`.
    Getter(NanBox),
}

/// Whether keyed reads decide accessor-bearing receivers and inherited
/// accessors from the text memo (`KATAAN_JIT_GETKEY=0` turns it off, read
/// once per process).
#[must_use]
pub(crate) fn getkey_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_GETKEY").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Text;

    /// The word reads give exactly the zero-padded little-endian words of
    /// every length up to the limit, and only those texts compare equal.
    #[test]
    fn text_words_match_padded_bytes() {
        let src: alloc::vec::Vec<u8> = (1u8..=30).collect();
        for n in 0..=super::TEXT_MAX {
            let b = &src[..n];
            let mut buf = [0u8; 24];
            buf[..n].copy_from_slice(b);
            let mut want = [0u64; 3];
            for (i, w) in want.iter_mut().enumerate() {
                let mut x = [0u8; 8];
                x.copy_from_slice(&buf[i * 8..i * 8 + 8]);
                *w = u64::from_le_bytes(x);
            }
            let t = Text::of(b).expect("short enough");
            assert_eq!(t.words, want, "length {n}");
            assert_eq!(t.len as usize, n);
            if n > 0 {
                let mut other = b.to_vec();
                other[n - 1] ^= 0x40;
                assert!(Text::of(&other) != Some(t));
            }
        }
        assert!(Text::of(&src[..25]).is_none());
        // A NUL-padded text differs from the shorter one by its length.
        assert!(Text::of(b"ab\0") != Text::of(b"ab"));
    }
}
