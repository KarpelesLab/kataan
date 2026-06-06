//! Heap snapshots of the live `Cell` object graph (`ROADMAP.md` §2.2, the
//! heap-snapshot tier).
//!
//! Where [`crate::bytecode`] persists *code* and [`crate::gc::relocate`] is the
//! generic pointer-fix-up pass, this captures an *initialized* object graph — the
//! real engine heap reachable from a set of roots — to a portable form and
//! restores it into a fresh [`Realm`](crate::realm::Realm). References are
//! serialized by **index**
//! (never a live handle), and restore is a two-pass rebuild — allocate every cell
//! empty, then fill in references resolved to the freshly-allocated handles —
//! which is exactly the pointer relocation a snapshot reload performs, and
//! handles cycles correctly.
//!
//! Covers the core value cells (objects, arrays, strings) plus the primitive
//! `NanBox` values; richer cells (functions/closures, promises, proxies) are a
//! later extension. Pure, safe `alloc`-only Rust.

use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use crate::realm::Realm;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A serialized value: a primitive, or a reference to a snapshot cell by index.
#[derive(Clone, Debug, PartialEq)]
pub enum SnapVal {
    /// `undefined`
    Undefined,
    /// `null`
    Null,
    /// a boolean
    Bool(bool),
    /// a number
    Number(f64),
    /// a reference to cell `index` in the snapshot
    Ref(usize),
}

/// A serialized heap cell (the core value kinds).
#[derive(Clone, Debug, PartialEq)]
pub enum SnapCell {
    /// a string value
    Str(String),
    /// an array of values
    Array(Vec<SnapVal>),
    /// an object: enumerable own `(key, value)` pairs
    Object(Vec<(String, SnapVal)>),
}

/// A captured object graph: the reachable cells (index-addressed) plus the
/// root indices.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Snapshot {
    /// All reachable cells, in discovery order; references are by index here.
    pub cells: Vec<SnapCell>,
    /// The roots, as indices into `cells` (a primitive root has no cell, so
    /// roots that aren't heap objects are dropped — capture is for object graphs).
    pub roots: Vec<usize>,
}

/// Captures the object graph reachable from `roots` into a [`Snapshot`].
/// Non-object roots (primitives) are skipped; object roots and everything they
/// transitively reference (objects, arrays, strings) are serialized.
#[must_use]
pub fn capture(realm: &Realm, roots: &[Handle]) -> Snapshot {
    let mut index_of: BTreeMap<Handle, usize> = BTreeMap::new();
    let mut order: Vec<Handle> = Vec::new();
    // Assigns an index to `h` (enqueuing it) on first sight.
    let mut intern =
        |index_of: &mut BTreeMap<Handle, usize>, order: &mut Vec<Handle>, h: Handle| -> usize {
            if let Some(i) = index_of.get(&h) {
                *i
            } else {
                let i = order.len();
                index_of.insert(h, i);
                order.push(h);
                i
            }
        };

    let mut root_indices = Vec::new();
    for r in roots {
        // Only object-like cells (string/array/object) are serializable roots.
        let serializable = realm.string_value(*r).is_some()
            || realm.array_elements(*r).is_some()
            || realm.object_keys(*r).is_some();
        if serializable {
            root_indices.push(intern(&mut index_of, &mut order, *r));
        }
    }

    // BFS: serialize each cell, interning child handles as we discover them.
    let mut cells: Vec<SnapCell> = Vec::new();
    let mut pos = 0;
    while pos < order.len() {
        let h = order[pos];
        pos += 1;
        let cell = if let Some(s) = realm.string_value(h) {
            SnapCell::Str(s)
        } else if let Some(elems) = realm.array_elements(h).map(<[_]>::to_vec) {
            let vals = elems
                .iter()
                .map(|v| snap_val(*v, &mut index_of, &mut order, &mut intern))
                .collect();
            SnapCell::Array(vals)
        } else if let Some(keys) = realm.object_keys(h) {
            let pairs = keys
                .into_iter()
                .map(|k| {
                    let v = realm.get_property(h, &k).unwrap_or(NanBox::undefined());
                    let sv = snap_val(v, &mut index_of, &mut order, &mut intern);
                    (k, sv)
                })
                .collect();
            SnapCell::Object(pairs)
        } else {
            // An unsupported cell kind: record it as an empty object so indices
            // stay aligned (its references are dropped).
            SnapCell::Object(Vec::new())
        };
        cells.push(cell);
    }

    Snapshot {
        cells,
        roots: root_indices,
    }
}

/// Serializes a `NanBox`, interning a heap reference into the snapshot.
fn snap_val(
    v: NanBox,
    index_of: &mut BTreeMap<Handle, usize>,
    order: &mut Vec<Handle>,
    intern: &mut impl FnMut(&mut BTreeMap<Handle, usize>, &mut Vec<Handle>, Handle) -> usize,
) -> SnapVal {
    match v.unpack() {
        Unpacked::Undefined => SnapVal::Undefined,
        Unpacked::Null => SnapVal::Null,
        Unpacked::Bool(b) => SnapVal::Bool(b),
        Unpacked::Number(n) => SnapVal::Number(n),
        Unpacked::Handle(raw) => SnapVal::Ref(intern(index_of, order, Handle::from_raw(raw))),
    }
}

/// Restores a [`Snapshot`] into `realm`, returning the new root handles.
///
/// Two passes: allocate every cell empty (so references can be resolved to live
/// handles), then fill in each cell's contents — the relocation step, where the
/// serialized indices become freshly-allocated handles. Cycles are handled
/// because all targets exist before any reference is written.
#[must_use]
pub fn restore(realm: &mut Realm, snap: &Snapshot) -> Vec<Handle> {
    // Pass 1: allocate a handle per cell (strings are immutable, built now).
    let handles: Vec<Handle> = snap
        .cells
        .iter()
        .map(|c| match c {
            SnapCell::Str(s) => realm.new_string(s),
            SnapCell::Array(_) => realm.new_array(Vec::new()),
            SnapCell::Object(_) => realm.new_object(),
        })
        .collect();

    // Pass 2: fill arrays and objects, resolving refs to the new handles.
    let resolve = |sv: &SnapVal, handles: &[Handle]| -> NanBox {
        match sv {
            SnapVal::Undefined => NanBox::undefined(),
            SnapVal::Null => NanBox::null(),
            SnapVal::Bool(b) => NanBox::boolean(*b),
            SnapVal::Number(n) => NanBox::number(*n),
            SnapVal::Ref(i) => handles
                .get(*i)
                .map_or(NanBox::undefined(), |h| NanBox::handle(h.to_raw())),
        }
    };
    for (cell, h) in snap.cells.iter().zip(&handles) {
        match cell {
            SnapCell::Str(_) => {}
            SnapCell::Array(vals) => {
                let elems: Vec<NanBox> = vals.iter().map(|v| resolve(v, &handles)).collect();
                realm.array_set_all(*h, elems);
            }
            SnapCell::Object(pairs) => {
                for (k, v) in pairs {
                    let val = resolve(v, &handles);
                    realm.set_property(*h, k, val);
                }
            }
        }
    }

    snap.roots
        .iter()
        .filter_map(|i| handles.get(*i).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_an_object_graph() {
        let mut realm = Realm::new();
        // Build a graph: obj { name: "root", n: 42, child: { tag: "leaf" }, list: [1, child] }
        let leaf = realm.new_object();
        let leaf_tag = NanBox::handle(realm.new_string("leaf").to_raw());
        realm.set_property(leaf, "tag", leaf_tag);

        let root = realm.new_object();
        let name = NanBox::handle(realm.new_string("root").to_raw());
        realm.set_property(root, "name", name);
        realm.set_property(root, "n", NanBox::number(42.0));
        realm.set_property(root, "child", NanBox::handle(leaf.to_raw()));
        let list = realm.new_array(alloc::vec![
            NanBox::number(1.0),
            NanBox::handle(leaf.to_raw())
        ]);
        realm.set_property(root, "list", NanBox::handle(list.to_raw()));

        // Capture, then restore into a *fresh* realm.
        let snap = capture(&realm, &[root]);
        let mut realm2 = Realm::new();
        let roots2 = restore(&mut realm2, &snap);
        assert_eq!(roots2.len(), 1);
        let r2 = roots2[0];

        // The restored graph is intact, with new handles.
        assert_eq!(
            realm2.get_property(r2, "n").unwrap().unpack(),
            Unpacked::Number(42.0)
        );
        let name2 = realm2
            .get_property(r2, "name")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(name2)).unwrap(),
            "root"
        );
        let child2 = realm2
            .get_property(r2, "child")
            .unwrap()
            .as_handle()
            .unwrap();
        let tag2 = realm2
            .get_property(Handle::from_raw(child2), "tag")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(realm2.string_value(Handle::from_raw(tag2)).unwrap(), "leaf");
        // The array's element 1 references the SAME restored child object
        // (shared structure preserved, not duplicated).
        let list2 = realm2
            .get_property(r2, "list")
            .unwrap()
            .as_handle()
            .unwrap();
        let elems = realm2.array_elements(Handle::from_raw(list2)).unwrap();
        assert_eq!(elems[0].unpack(), Unpacked::Number(1.0));
        assert_eq!(elems[1].as_handle(), Some(child2));
    }

    #[test]
    fn round_trips_a_cycle() {
        let mut realm = Realm::new();
        // a.self_ = a; a.b = b; b.back = a  (a cycle).
        let a = realm.new_object();
        let b = realm.new_object();
        realm.set_property(a, "self_", NanBox::handle(a.to_raw()));
        realm.set_property(a, "b", NanBox::handle(b.to_raw()));
        realm.set_property(b, "back", NanBox::handle(a.to_raw()));

        let snap = capture(&realm, &[a]);
        let mut realm2 = Realm::new();
        let a2 = restore(&mut realm2, &snap)[0];
        // self-reference and the 2-cycle survive under the new handles.
        assert_eq!(
            realm2.get_property(a2, "self_").unwrap().as_handle(),
            Some(a2.to_raw())
        );
        let b2 = realm2.get_property(a2, "b").unwrap().as_handle().unwrap();
        assert_eq!(
            realm2
                .get_property(Handle::from_raw(b2), "back")
                .unwrap()
                .as_handle(),
            Some(a2.to_raw())
        );
    }

    #[test]
    fn primitive_roots_are_skipped() {
        let realm = Realm::new();
        let snap = capture(&realm, &[]);
        assert!(snap.cells.is_empty() && snap.roots.is_empty());
    }
}
