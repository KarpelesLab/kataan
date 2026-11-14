//! The names of an object literal's method definitions, by key String.
//!
//! Each evaluation of `{ get p() {}, m() {} }` defines its methods anew
//! (`Op::DefineMethod`), and each definition needs two names made from the
//! key: the property key's storage form and the function's name (`"get p"`,
//! SetFunctionName). Both are a function of the key String's text alone, so
//! they are kept per key String handle, for the handle epoch they were made
//! in: a handle can only name another String after a sweep or compaction,
//! which ends the epoch (`Heap::epoch`). Direct-mapped, a lookup is one
//! probe.

use super::Realm;
use crate::nanbox::NanBox;
use alloc::rc::Rc;

/// The number of entries (a power of two).
const ENTRIES: usize = 16;

/// One key's names (see the module docs).
#[derive(Clone)]
struct Entry {
    /// The key's raw bits (a String handle), `0` for none.
    key: u64,
    /// The definition kind (`0` method, `1` getter, `2` setter): the
    /// function name's prefix.
    kind: u8,
    /// The heap epoch the names were made in.
    epoch: u32,
    /// The property key's storage form.
    name: Rc<str>,
    /// The function name's text (WTF-8).
    fname: Rc<[u8]>,
}

/// The memo itself.
pub(crate) struct MethodNames {
    entries: [Option<Entry>; ENTRIES],
}

/// Whether the memo is used (`KATAAN_METHOD_NAMES=0` turns it off, read
/// once per process).
fn on() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_METHOD_NAMES").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

impl Realm {
    /// The names of a method definition (`kind` `0`-`2`) under the String
    /// `key`, as last recorded by [`note_method_names`](Self::note_method_names)
    /// in this handle epoch: `(storage key, function name)`. `None` for a
    /// key that is not a String handle, or not recorded.
    pub(crate) fn method_names(&self, key: NanBox, kind: u8) -> Option<(Rc<str>, Rc<[u8]>)> {
        let raw = key.as_handle()?;
        let m = self.method_names.as_deref()?;
        let e = m.entries[slot(raw, kind)].as_ref()?;
        (e.key == raw && e.kind == kind && e.epoch == self.heap.epoch())
            .then(|| (Rc::clone(&e.name), Rc::clone(&e.fname)))
    }

    /// Records the names of a method definition under `key` (see
    /// [`method_names`](Self::method_names)), when it is a String handle,
    /// and returns them shared.
    pub(crate) fn note_method_names(
        &mut self,
        key: NanBox,
        kind: u8,
        name: &str,
        fname: &[u8],
    ) -> (Rc<str>, Rc<[u8]>) {
        let (name, fname): (Rc<str>, Rc<[u8]>) = (Rc::from(name), Rc::from(fname));
        if let Some(raw) = key.as_handle()
            && on()
            && self.is_string_handle(crate::heap::Handle::from_raw(raw))
        {
            let epoch = self.heap.epoch();
            let m = self.method_names.get_or_insert_with(|| {
                alloc::boxed::Box::new(MethodNames {
                    entries: Default::default(),
                })
            });
            m.entries[slot(raw, kind)] = Some(Entry {
                key: raw,
                kind,
                epoch,
                name: Rc::clone(&name),
                fname: Rc::clone(&fname),
            });
        }
        (name, fname)
    }
}

/// The entry index for a key and kind.
fn slot(raw: u64, kind: u8) -> usize {
    let h = (raw ^ u64::from(kind) << 40).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (h >> 60) as usize & (ENTRIES - 1)
}
