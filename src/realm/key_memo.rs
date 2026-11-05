//! The String-key slot memo: "the text of rope R is slot S of layout L".
//!
//! `o[k]` and `k in o` with a String `k` that names an own data property of
//! a shaped object find the key's slot by name: they borrow the key's text
//! (`Realm::string_key_in_place`), hash it and probe the layout's index, or
//! walk a small layout's keys. An object used as a dictionary is read that way
//! with the same few key Strings over and over (`for (k of keys) d[k]`), so
//! the answer is memoized per key text and layout.
//!
//! An entry is found by the key's rope (the String cell's immutable text, by
//! address) and confirmed by its rope and the receiver's layout, both by
//! address. The entry holds the layout and a weak reference to the rope, so
//! neither address can be reused while the entry stands: an entry can only
//! be stale by being absent. A rope's text is fixed and a layout's keys and
//! slots are fixed, so a hit is exact for any object of that layout; what
//! is not in the layout is decided per read: the object must be shaped, with
//! no accessors (an accessor of the same name would shadow the slot), not
//! host-exotic, and the key String must have no auxiliary object (as in
//! `string_key_in_place`). A wrapper's or a mapped `arguments` object's
//! layout carries its internal key, so such a layout is never recorded.
//!
//! The memo is direct-mapped: a lookup is one probe. A miss records its key
//! String's handle at the entry, and only a second miss for the same String
//! there soon after (within [`SEEN_WINDOW`] misses) replaces the entry, so
//! a key String made for one read (a fresh concatenation, whose handle and
//! node a later one may reuse only past a collection) costs the probe and
//! a few stores, and evicts nothing.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::object::Object;
use crate::rope::RopeId;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::Cell as Slot;

/// The number of entries (a power of two).
const CAP: usize = 1024;

/// One memoized slot (see the module docs). `rope == 0`: empty.
#[derive(Default)]
struct Entry {
    /// The key rope's address.
    rope: Slot<usize>,
    /// The layout's address.
    shape: Slot<usize>,
    /// The key's slot in the layout.
    slot: Slot<u32>,
    /// The handle of the key String this entry last missed for (see
    /// [`Realm::key_memo_record`]).
    seen: Slot<u64>,
    /// [`KeyMemo::tick`] at that miss.
    seen_tick: Slot<u32>,
    /// What keeps both addresses from being reused while the entry stands.
    hold: Slot<Option<(RopeId, Rc<Shape>)>>,
}

/// The realm's String-key slot memo (see the module docs).
pub(crate) struct KeyMemo {
    entries: Box<[Entry]>,
    /// The number of noted misses so far (wrapping).
    tick: Slot<u32>,
    /// The key String handle and rope address of the last probe that
    /// missed at its entry.
    last: Slot<(u64, usize)>,
}

impl Default for KeyMemo {
    fn default() -> Self {
        Self {
            entries: (0..CAP).map(|_| Entry::default()).collect(),
            tick: Slot::new(0),
            last: Slot::new((u64::MAX, 0)),
        }
    }
}

/// How many noted misses later a key String's second miss still records it.
const SEEN_WINDOW: u32 = 512;

/// The entry index of a rope at address `p`.
#[inline]
fn index(p: usize) -> usize {
    ((p >> 4) ^ (p >> 14)) & (CAP - 1)
}

impl Realm {
    /// The key String at `key` and its rope's address, when it is a String.
    #[inline]
    fn key_rope(&self, key: NanBox) -> Option<(u64, &crate::rope::Rope)> {
        let raw = key.as_handle()?;
        let r = self.heap.get(Handle::from_raw(raw))?.as_str()?;
        Some((raw, r))
    }

    /// The object at `obj` and the memoized slot of the key String `key` in
    /// its layout, when the memo has one and the read may use it (see the
    /// module docs): `obj[key]` is that slot's value and `key in obj` is
    /// true. `None` otherwise: the ordinary lookup decides, and may
    /// [`key_memo_note`](Self::key_memo_note) its answer.
    #[inline(always)]
    pub(crate) fn key_memo_probe(&self, obj: NanBox, key: NanBox) -> Option<(&Object, u32)> {
        let memo = self.key_memo.get()?;
        let (raw, rope) = self.key_rope(key)?;
        let p = rope.addr();
        let e = &memo.entries[index(p)];
        if e.rope.get() != p {
            // (For the note that may follow.)
            memo.last.set((raw, p));
            return None;
        }
        let h = Handle::from_raw(obj.as_handle()?);
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        if Rc::as_ptr(shape) as usize != e.shape.get()
            || !o.has_no_accessors()
            || o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || self.is_host_exotic(h)
            || (self.str_special && self.aux_props.contains_key(&raw))
        {
            return None;
        }
        Some((o, e.slot.get()))
    }

    /// Notes, after a probe missed, that the key String `key` (whose text is
    /// its own storage key) is slot `slot` of `shape` (a layout with no
    /// internal wrapper or `arguments` key). Only a key String that misses
    /// a second time in a row at its entry is recorded: one made for a
    /// single read (`d['k' + i]`) costs one store.
    #[inline]
    pub(crate) fn key_memo_note(&self, key: NanBox, shape: &Rc<Shape>, slot: u32) {
        let Some(raw) = key.as_handle() else {
            return;
        };
        // The rope's address, as the probe that missed left it.
        let (memo, p) = match self.key_memo.get() {
            Some(m) if m.last.get().0 == raw => (m, m.last.get().1),
            _ => {
                let Some((_, rope)) = self.key_rope(key) else {
                    return;
                };
                (self.key_memo.get_or_init(Box::default), rope.addr())
            }
        };
        let e = &memo.entries[index(p)];
        let tick = memo.tick.get().wrapping_add(1);
        memo.tick.set(tick);
        // (A freed String's handle and node may both be reused by a later
        // one, but only past a collection: not within a short window.)
        if e.seen.get() != raw || tick.wrapping_sub(e.seen_tick.get()) > SEEN_WINDOW {
            e.seen.set(raw);
            e.seen_tick.set(tick);
            return;
        }
        self.key_memo_record(e, raw, p, shape, slot);
    }

    /// [`key_memo_note`](Self::key_memo_note)'s recording.
    #[inline(never)]
    fn key_memo_record(&self, e: &Entry, raw: u64, p: usize, shape: &Rc<Shape>, slot: u32) {
        let Some(rope) = self
            .heap
            .get(Handle::from_raw(raw))
            .and_then(Cell::as_str)
            .filter(|r| r.addr() == p)
        else {
            return;
        };
        e.rope.set(p);
        e.shape.set(Rc::as_ptr(shape) as usize);
        e.slot.set(slot);
        // (The old hold drops after the new one is in place.)
        drop(e.hold.replace(Some((rope.id(), Rc::clone(shape)))));
    }
}
