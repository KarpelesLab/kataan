//! The Symbol-key slot memo: "Symbol S (id I) is slot K of layout L".
//!
//! `o[sym]` names its property by the storage key `"\0sym:<id>"`, which a
//! read had to build and hash (a layout lookup by text) every time. A
//! program that tags objects with a few Symbols (`obj[META]`,
//! `obj[VERSION]`) reads them that way over and over, so the answer is
//! memoized per Symbol and layout, as the String-key memo (`key_memo`)
//! does per key text.
//!
//! An entry is found by the Symbol cell's handle and confirmed by the
//! Symbol's id (a freed cell's slot may hold another Symbol later; the id
//! is what the storage key names) and by the receiver's layout, by
//! address. The entry holds the layout, so its address cannot be reused
//! while the entry stands. A layout's keys and slots are fixed, so a hit is
//! exact for any object of that layout that is a plain shaped object with no
//! accessors (an accessor of that name would shadow the slot), not
//! host-exotic. The memo is direct-mapped: a lookup is one probe, and a miss
//! that finds the key in the receiver's layout replaces the entry.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::Cell as Slot;

/// The number of entries (a power of two).
const CAP: usize = 64;

#[derive(Default)]
struct Entry {
    /// The Symbol's handle bits (0: empty).
    sym: Slot<u64>,
    /// Its id.
    id: Slot<u64>,
    /// The layout's address.
    shape: Slot<usize>,
    /// The key's slot in the layout.
    slot: Slot<u32>,
    /// What keeps the layout's address from being reused.
    hold: Slot<Option<Rc<Shape>>>,
}

/// The realm's Symbol-key slot memo (see the module docs).
pub(crate) struct SymMemo {
    entries: Box<[Entry]>,
}

impl Default for SymMemo {
    fn default() -> Self {
        Self {
            entries: (0..CAP).map(|_| Entry::default()).collect(),
        }
    }
}

#[inline]
fn index(bits: u64) -> usize {
    ((bits >> 3) ^ (bits >> 11)) as usize & (CAP - 1)
}

impl Realm {
    /// `obj[sym]` when `sym` (a Symbol with id `id`) is an own data property
    /// of the plain shaped object `obj` (see the module docs): its value,
    /// from the memo or the layout (recording it). `None` when the object is
    /// not such an object or does not own the key (the caller's general
    /// path then decides).
    pub(crate) fn sym_own_data(&self, obj: Handle, sym: NanBox, id: u64) -> Option<NanBox> {
        let Some(Cell::Object(o)) = self.heap.get(obj) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        if !o.has_no_accessors()
            || o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || self.is_host_exotic(obj)
        {
            return None;
        }
        let bits = sym.to_bits();
        let memo = self.sym_memo.get_or_init(Box::default);
        let e = &memo.entries[index(bits)];
        let addr = Rc::as_ptr(shape) as usize;
        if e.sym.get() == bits && e.id.get() == id && e.shape.get() == addr {
            return o.slot(e.slot.get());
        }
        let mut buf = [0u8; 32];
        let key = crate::shape::symbol_key_in(id, &mut buf);
        let slot = shape.lookup(key)?;
        let v = o.slot(slot)?;
        e.sym.set(bits);
        e.id.set(id);
        e.shape.set(addr);
        e.slot.set(slot);
        e.hold.set(Some(Rc::clone(shape)));
        Some(v)
    }
}
