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

/// `KSNP` — the on-disk heap-snapshot magic.
const MAGIC: &[u8; 4] = b"KSNP";
const VERSION: u16 = 1;

/// Why a serialized snapshot failed to load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapError {
    /// Bad magic or unsupported version.
    BadHeader,
    /// The buffer ended mid-record.
    Truncated,
    /// An unknown value/cell tag.
    BadTag(u8),
    /// Non-UTF-8 string bytes.
    BadString,
}

fn w_u32(v: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_str(s: &str, out: &mut Vec<u8>) {
    w_u32(s.len() as u32, out);
    out.extend_from_slice(s.as_bytes());
}
fn w_val(v: &SnapVal, out: &mut Vec<u8>) {
    match v {
        SnapVal::Undefined => out.push(0),
        SnapVal::Null => out.push(1),
        SnapVal::Bool(b) => {
            out.push(2);
            out.push(u8::from(*b));
        }
        SnapVal::Number(n) => {
            out.push(3);
            out.extend_from_slice(&n.to_le_bytes());
        }
        SnapVal::Ref(i) => {
            out.push(4);
            w_u32(*i as u32, out);
        }
    }
}

/// Serializes a [`Snapshot`] to a self-describing `KSNP` byte container — the
/// on-disk heap-snapshot artifact (`ROADMAP.md` §2.2).
#[must_use]
pub fn serialize(snap: &Snapshot) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    w_u32(snap.roots.len() as u32, &mut out);
    for r in &snap.roots {
        w_u32(*r as u32, &mut out);
    }
    w_u32(snap.cells.len() as u32, &mut out);
    for cell in &snap.cells {
        match cell {
            SnapCell::Str(s) => {
                out.push(0);
                w_str(s, &mut out);
            }
            SnapCell::Array(vals) => {
                out.push(1);
                w_u32(vals.len() as u32, &mut out);
                for v in vals {
                    w_val(v, &mut out);
                }
            }
            SnapCell::Object(pairs) => {
                out.push(2);
                w_u32(pairs.len() as u32, &mut out);
                for (k, v) in pairs {
                    w_str(k, &mut out);
                    w_val(v, &mut out);
                }
            }
        }
    }
    out
}

/// A cursor that reads little-endian fields, erroring on truncation.
struct R<'a> {
    b: &'a [u8],
    p: usize,
}
impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], SnapError> {
        let end = self.p.checked_add(n).ok_or(SnapError::Truncated)?;
        let s = self.b.get(self.p..end).ok_or(SnapError::Truncated)?;
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, SnapError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, SnapError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, SnapError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, SnapError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, SnapError> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        core::str::from_utf8(bytes)
            .map(String::from)
            .map_err(|_| SnapError::BadString)
    }
    fn val(&mut self) -> Result<SnapVal, SnapError> {
        Ok(match self.u8()? {
            0 => SnapVal::Undefined,
            1 => SnapVal::Null,
            2 => SnapVal::Bool(self.u8()? != 0),
            3 => SnapVal::Number(self.f64()?),
            4 => SnapVal::Ref(self.u32()? as usize),
            t => return Err(SnapError::BadTag(t)),
        })
    }
}

/// Reloads a [`Snapshot`] from a `KSNP` artifact.
///
/// # Errors
/// [`SnapError`] for a bad header, truncation, or a corrupt tag.
pub fn deserialize(bytes: &[u8]) -> Result<Snapshot, SnapError> {
    let mut r = R { b: bytes, p: 0 };
    if r.take(4)? != MAGIC || r.u16()? != VERSION {
        return Err(SnapError::BadHeader);
    }
    let n_roots = r.u32()? as usize;
    let mut roots = Vec::with_capacity(n_roots);
    for _ in 0..n_roots {
        roots.push(r.u32()? as usize);
    }
    let n_cells = r.u32()? as usize;
    let mut cells = Vec::with_capacity(n_cells);
    for _ in 0..n_cells {
        let cell = match r.u8()? {
            0 => SnapCell::Str(r.string()?),
            1 => {
                let n = r.u32()? as usize;
                let mut vals = Vec::with_capacity(n);
                for _ in 0..n {
                    vals.push(r.val()?);
                }
                SnapCell::Array(vals)
            }
            2 => {
                let n = r.u32()? as usize;
                let mut pairs = Vec::with_capacity(n);
                for _ in 0..n {
                    let k = r.string()?;
                    let v = r.val()?;
                    pairs.push((k, v));
                }
                SnapCell::Object(pairs)
            }
            t => return Err(SnapError::BadTag(t)),
        };
        cells.push(cell);
    }
    Ok(Snapshot { cells, roots })
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
    fn serialize_round_trip_then_restore() {
        let mut realm = Realm::new();
        let leaf = realm.new_object();
        let tag = NanBox::handle(realm.new_string("leaf").to_raw());
        realm.set_property(leaf, "tag", tag);
        let root = realm.new_object();
        realm.set_property(root, "n", NanBox::number(7.0));
        realm.set_property(root, "flag", NanBox::boolean(true));
        realm.set_property(root, "child", NanBox::handle(leaf.to_raw()));
        let arr = realm.new_array(alloc::vec![
            NanBox::number(1.0),
            NanBox::handle(leaf.to_raw()),
            NanBox::null(),
        ]);
        realm.set_property(root, "list", NanBox::handle(arr.to_raw()));

        // capture → serialize → deserialize → the snapshot is byte-stable.
        let snap = capture(&realm, &[root]);
        let bytes = serialize(&snap);
        assert_eq!(&bytes[0..4], MAGIC);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap, "snapshot round-trips losslessly");
        // Re-serializing the reloaded snapshot is byte-identical.
        assert_eq!(serialize(&reloaded), bytes);

        // ...and the reloaded snapshot restores into a fresh realm correctly.
        let mut realm2 = Realm::new();
        let r2 = restore(&mut realm2, &reloaded)[0];
        assert_eq!(
            realm2.get_property(r2, "n").unwrap().unpack(),
            Unpacked::Number(7.0)
        );
        assert_eq!(
            realm2.get_property(r2, "flag").unwrap().unpack(),
            Unpacked::Bool(true)
        );
        let child2 = realm2
            .get_property(r2, "child")
            .unwrap()
            .as_handle()
            .unwrap();
        let list2 = realm2
            .get_property(r2, "list")
            .unwrap()
            .as_handle()
            .unwrap();
        let elems = realm2.array_elements(Handle::from_raw(list2)).unwrap();
        // The shared leaf is the same restored object in both places.
        assert_eq!(elems[1].as_handle(), Some(child2));
        assert_eq!(elems[2].unpack(), Unpacked::Null);
    }

    #[test]
    fn deserialize_rejects_garbage() {
        assert_eq!(deserialize(b"XXXX\x01\x00"), Err(SnapError::BadHeader));
        let good = serialize(&capture(&Realm::new(), &[]));
        assert_eq!(
            deserialize(&good[..good.len() - 1]),
            Err(SnapError::Truncated)
        );
    }

    #[test]
    fn primitive_roots_are_skipped() {
        let realm = Realm::new();
        let snap = capture(&realm, &[]);
        assert!(snap.cells.is_empty() && snap.roots.is_empty());
    }
}
