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
//! Covers the value cells (objects, arrays, strings, `Date`, `BigInt`) and
//! functions/closures (code id + captured scope chain), plus the primitive
//! `NanBox` values; promises/proxies are a later extension. Pure, safe
//! `alloc`-only Rust.

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
    /// a `Date` (millisecond timestamp)
    Date(f64),
    /// a `BigInt` (its base-10 digit string)
    BigInt(String),
    /// a function/closure: its code id plus the captured scope chain (innermost
    /// frame first; each frame's parent is the next). Restorable against the same
    /// compiled program.
    Function {
        /// the function-table index (code identity)
        func_id: u32,
        /// the captured lexical environment, innermost frame first
        frames: Vec<SnapFrame>,
    },
}

/// One captured scope frame: its `(name, value, is_const)` bindings.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapFrame {
    /// the frame's own bindings
    pub vars: Vec<(String, SnapVal, bool)>,
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
        // Only value cells (string/date/bigint/array/object) are serializable roots.
        let serializable = realm.string_value(*r).is_some()
            || realm.date_at(*r).is_some()
            || realm.bigint_at(*r).is_some()
            || realm.function_at(*r).is_some()
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
        } else if let Some(ms) = realm.date_at(h) {
            SnapCell::Date(ms)
        } else if let Some(bi) = realm.bigint_at(h) {
            SnapCell::BigInt(bi.to_str_radix(10))
        } else if let Some((func_id, scope)) = realm.function_at(h) {
            // Walk the closure's scope chain, interning captured handles.
            let mut frames = Vec::new();
            let mut cur = Some(scope);
            while let Some(s) = cur {
                let vars = s
                    .local_bindings()
                    .into_iter()
                    .map(|(k, v, c)| (k, snap_val(v, &mut index_of, &mut order, &mut intern), c))
                    .collect();
                frames.push(SnapFrame { vars });
                cur = s.parent();
            }
            SnapCell::Function { func_id, frames }
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
    // Pass 1: allocate a handle per cell (strings are immutable, built now). For
    // functions, build the (empty) scope chain now and keep it to fill in pass 2,
    // so a closure capturing itself/its siblings resolves correctly.
    let mut handles: Vec<Handle> = Vec::with_capacity(snap.cells.len());
    let mut fn_chains: Vec<Option<Vec<crate::env::Scope>>> = Vec::with_capacity(snap.cells.len());
    for c in &snap.cells {
        let (h, chain) = match c {
            SnapCell::Str(s) => (realm.new_string(s), None),
            SnapCell::Date(ms) => (realm.new_date(*ms), None),
            SnapCell::BigInt(s) => {
                let bi = crate::bignum::BigInt::from_str_radix(s, 10).unwrap_or_default();
                (realm.new_bigint(bi), None)
            }
            SnapCell::Array(_) => (realm.new_array(Vec::new()), None),
            SnapCell::Object(_) => (realm.new_object(), None),
            SnapCell::Function { func_id, frames } => {
                // Build empty scopes outermost→innermost; the function closes over
                // the innermost. `chain[j]` corresponds to `frames[n-1-j]`.
                let n = frames.len();
                let mut chain: Vec<crate::env::Scope> = Vec::with_capacity(n);
                for j in 0..n {
                    let s = if j == 0 {
                        crate::env::Scope::root()
                    } else {
                        chain[j - 1].child()
                    };
                    chain.push(s);
                }
                let innermost = chain
                    .last()
                    .cloned()
                    .unwrap_or_else(crate::env::Scope::root);
                (realm.new_function(*func_id, innermost), Some(chain))
            }
        };
        handles.push(h);
        fn_chains.push(chain);
    }

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
    for (idx, (cell, h)) in snap.cells.iter().zip(&handles).enumerate() {
        match cell {
            // Reference-free cells were built fully in pass 1.
            SnapCell::Str(_) | SnapCell::Date(_) | SnapCell::BigInt(_) => {}
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
            SnapCell::Function { frames, .. } => {
                // Fill each frame's bindings into its (now-allocated) scope.
                // `frames[f]` (f=0 innermost) ↔ `chain[n-1-f]`.
                if let Some(chain) = &fn_chains[idx] {
                    let n = frames.len();
                    for (f, frame) in frames.iter().enumerate() {
                        let scope = &chain[n - 1 - f];
                        for (name, v, is_const) in &frame.vars {
                            let val = resolve(v, &handles);
                            if *is_const {
                                scope.declare_const(name, val);
                            } else {
                                scope.declare(name, val);
                            }
                        }
                    }
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
            SnapCell::Date(ms) => {
                out.push(3);
                out.extend_from_slice(&ms.to_le_bytes());
            }
            SnapCell::BigInt(digits) => {
                out.push(4);
                w_str(digits, &mut out);
            }
            SnapCell::Function { func_id, frames } => {
                out.push(5);
                w_u32(*func_id, &mut out);
                w_u32(frames.len() as u32, &mut out);
                for frame in frames {
                    w_u32(frame.vars.len() as u32, &mut out);
                    for (name, v, is_const) in &frame.vars {
                        w_str(name, &mut out);
                        w_val(v, &mut out);
                        out.push(u8::from(*is_const));
                    }
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
            3 => SnapCell::Date(r.f64()?),
            4 => SnapCell::BigInt(r.string()?),
            5 => {
                let func_id = r.u32()?;
                let nf = r.u32()? as usize;
                let mut frames = Vec::with_capacity(nf);
                for _ in 0..nf {
                    let nv = r.u32()? as usize;
                    let mut vars = Vec::with_capacity(nv);
                    for _ in 0..nv {
                        let name = r.string()?;
                        let v = r.val()?;
                        let is_const = r.u8()? != 0;
                        vars.push((name, v, is_const));
                    }
                    frames.push(SnapFrame { vars });
                }
                SnapCell::Function { func_id, frames }
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
    fn snapshots_survive_a_moving_gc_compaction() {
        // D′ atop a moving GC: a snapshot taken *after* the heap has been compacted
        // (survivors relocated, references rewritten) must still capture the graph.
        let mut realm = Realm::new();
        // Allocate garbage first so compaction has slots to reclaim/relocate.
        for i in 0..8 {
            let g = realm.new_object();
            realm.set_property(g, "junk", NanBox::number(f64::from(i)));
        }
        // The live graph: root { name:"keep", child:{ tag:"leaf" }, items:[1,2,3] }.
        let leaf = realm.new_object();
        let leaf_tag = NanBox::handle(realm.new_string("leaf").to_raw());
        realm.set_property(leaf, "tag", leaf_tag);
        let arr = realm.new_array(alloc::vec![
            NanBox::number(1.0),
            NanBox::number(2.0),
            NanBox::number(3.0),
        ]);
        let root = realm.new_object();
        let name = NanBox::handle(realm.new_string("keep").to_raw());
        realm.set_property(root, "name", name);
        realm.set_property(root, "child", NanBox::handle(leaf.to_raw()));
        realm.set_property(root, "items", NanBox::handle(arr.to_raw()));

        // Compact: only `root` is a root, so the 8 garbage objects are reclaimed and
        // the survivors relocated. `root` is rewritten in place to its new slot.
        let mut roots = [root];
        realm.compact(&mut roots);
        let root = roots[0];

        // Snapshot the post-compaction heap and restore it into a fresh realm.
        let snap = capture(&realm, &[root]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        let mut realm2 = Realm::new();
        let r2 = restore(&mut realm2, &reloaded)[0];

        // The whole graph survived the relocation intact.
        let name2 = realm2
            .get_property(r2, "name")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(name2)),
            Some(String::from("keep"))
        );
        let child = realm2
            .get_property(r2, "child")
            .unwrap()
            .as_handle()
            .unwrap();
        let ctag = realm2
            .get_property(Handle::from_raw(child), "tag")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(ctag)),
            Some(String::from("leaf"))
        );
        let items = realm2
            .get_property(r2, "items")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(
            realm2
                .array_elements(Handle::from_raw(items))
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn snapshots_functions_and_closures() {
        use crate::env::Scope;
        let mut realm = Realm::new();
        // A closure over a 2-level scope: outer { base: 100 (const) }, inner { x: 7 },
        // and a captured heap object { tag: "cap" } bound as `obj` in the inner frame.
        let outer = Scope::root();
        outer.declare_const("base", NanBox::number(100.0));
        let inner = outer.child();
        inner.declare("x", NanBox::number(7.0));
        let cap = realm.new_object();
        let tag = NanBox::handle(realm.new_string("cap").to_raw());
        realm.set_property(cap, "tag", tag);
        inner.declare("obj", NanBox::handle(cap.to_raw()));
        let func = realm.new_function(0xABCD, inner);

        // capture → serialize → deserialize → restore.
        let snap = capture(&realm, &[func]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap);
        let mut realm2 = Realm::new();
        let f2 = restore(&mut realm2, &reloaded)[0];

        // The function's code id and captured environment survive.
        let (func_id, scope) = realm2.function_at(f2).expect("restored function");
        assert_eq!(func_id, 0xABCD);
        assert_eq!(scope.get("x"), Some(NanBox::number(7.0)));
        assert_eq!(scope.get("base"), Some(NanBox::number(100.0)));
        assert!(scope.is_const("base"), "const-ness preserved");
        // The captured object came back too, with its property.
        let obj = scope.get("obj").unwrap().as_handle().unwrap();
        let obj_tag = realm2.get_property(Handle::from_raw(obj), "tag").unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(obj_tag.as_handle().unwrap())),
            Some(String::from("cap"))
        );
    }

    #[test]
    fn snapshots_dates_and_bigints() {
        let mut realm = Realm::new();
        // An object holding a Date and a BigInt, plus a Date root array.
        let when = realm.new_date(1_592_217_045_123.0);
        let big = realm.new_bigint(
            crate::bignum::BigInt::from_str_radix("123456789012345678901234567890", 10).unwrap(),
        );
        let obj = realm.new_object();
        realm.set_property(obj, "when", NanBox::handle(when.to_raw()));
        realm.set_property(obj, "big", NanBox::handle(big.to_raw()));

        // capture → serialize → deserialize → restore.
        let snap = capture(&realm, &[obj]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap);
        let mut realm2 = Realm::new();
        let o2 = restore(&mut realm2, &reloaded)[0];

        // The Date timestamp and BigInt digits survive.
        let when2 = realm2
            .get_property(o2, "when")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(
            realm2.date_at(Handle::from_raw(when2)),
            Some(1_592_217_045_123.0)
        );
        let big2 = realm2.get_property(o2, "big").unwrap().as_handle().unwrap();
        assert_eq!(
            realm2
                .bigint_at(Handle::from_raw(big2))
                .unwrap()
                .to_str_radix(10),
            "123456789012345678901234567890"
        );
    }

    #[test]
    fn primitive_roots_are_skipped() {
        let realm = Realm::new();
        let snap = capture(&realm, &[]);
        assert!(snap.cells.is_empty() && snap.roots.is_empty());
    }
}
