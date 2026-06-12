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
//! Covers the value cells (objects, arrays, strings, `Date`, `BigInt`),
//! functions/closures (code id + captured scope chain), `Map`/`Set` collections,
//! settled `Promise`s, and `Proxy`s, plus the primitive `NanBox` values — the full
//! set of reference cell kinds. Pure, safe `alloc`-only Rust.

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
    /// an object: own `(key, value, hidden)` data properties — including
    /// non-enumerable ones and the engine's internal `\0`-prefixed slots (so
    /// typed-array kind, bound-function target/this/args, error `stack`, etc.
    /// survive). `hidden` is true for a non-enumerable property. `proto` is a
    /// reference to the `[[Prototype]]` object (or `Null` when there is none), so
    /// `Object.create(p)` / inheritance chains round-trip.
    Object {
        /// own data properties, each with its non-enumerable flag
        props: Vec<(String, SnapVal, bool)>,
        /// own accessor properties: `(key, getter, setter, hidden)` — getter/setter
        /// are references to function cells (or `Undefined` when absent)
        accessors: Vec<(String, SnapVal, SnapVal, bool)>,
        /// the prototype link (`Null` if none)
        proto: SnapVal,
    },
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
    /// a `Map` (`is_set == false`) or `Set` (`is_set == true`): its `(key, value)`
    /// entries in insertion order (a `Set` stores `value` as both).
    Collection {
        /// whether this is a `Set` (else a `Map`)
        is_set: bool,
        /// the entries, in insertion order
        entries: Vec<(SnapVal, SnapVal)>,
    },
    /// a `Promise`: its settlement status (0 pending, 1 fulfilled, 2 rejected) and
    /// value/reason. Pending reactions are runtime-coupled and not captured.
    Promise {
        /// 0 = pending, 1 = fulfilled, 2 = rejected
        status: u8,
        /// the fulfillment value or rejection reason
        value: SnapVal,
    },
    /// a `Proxy`: references to its target and handler objects.
    Proxy {
        /// the wrapped target
        target: SnapVal,
        /// the trap handler
        handler: SnapVal,
    },
    /// a `RegExp`: its source, flags, and current `lastIndex`.
    RegExp {
        /// the pattern source
        source: String,
        /// the flag string (e.g. `"gi"`)
        flags: String,
        /// the stateful `lastIndex` search position
        last_index: usize,
    },
    /// a `Symbol`: its description (identity is re-minted on restore but stays
    /// consistent within the snapshot, since the cell is interned once).
    Symbol {
        /// the symbol's description string
        description: String,
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
            || realm.collection_is_set(*r).is_some()
            || realm.promise_state(*r).is_some()
            || realm.proxy_at(*r).is_some()
            || realm.regexp_at(*r).is_some()
            || realm.symbol_at(*r).is_some()
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
        } else if let Some(is_set) = realm.collection_is_set(h) {
            let entries = realm
                .collection_entries(h)
                .unwrap_or_default()
                .iter()
                .map(|(k, v)| {
                    (
                        snap_val(*k, &mut index_of, &mut order, &mut intern),
                        snap_val(*v, &mut index_of, &mut order, &mut intern),
                    )
                })
                .collect();
            SnapCell::Collection { is_set, entries }
        } else if let Some(state) = realm.promise_state(h) {
            let (status, value) = {
                let s = state.borrow();
                let status = match s.status {
                    crate::cell::PromiseStatus::Pending => 0u8,
                    crate::cell::PromiseStatus::Fulfilled => 1,
                    crate::cell::PromiseStatus::Rejected => 2,
                };
                (status, s.value)
            };
            let value = snap_val(value, &mut index_of, &mut order, &mut intern);
            SnapCell::Promise { status, value }
        } else if let Some((target, handler)) = realm.proxy_at(h) {
            let target = snap_val(
                NanBox::handle(target.to_raw()),
                &mut index_of,
                &mut order,
                &mut intern,
            );
            let handler = snap_val(
                NanBox::handle(handler.to_raw()),
                &mut index_of,
                &mut order,
                &mut intern,
            );
            SnapCell::Proxy { target, handler }
        } else if let Some((source, flags)) = realm.regexp_at(h) {
            SnapCell::RegExp {
                source,
                flags,
                last_index: realm.regex_last_index(h),
            }
        } else if let Some((description, _id)) = realm.symbol_at(h) {
            SnapCell::Symbol { description }
        } else if let Some(elems) = realm.array_elements(h).map(<[_]>::to_vec) {
            let vals = elems
                .iter()
                .map(|v| snap_val(*v, &mut index_of, &mut order, &mut intern))
                .collect();
            SnapCell::Array(vals)
        } else if realm.object_keys(h).is_some() {
            // All own *data* properties, including non-enumerable + internal slots
            // (private `#` fields and accessor keys are handled separately). Each
            // pair records whether the property is hidden (non-enumerable).
            let accessor_keys = realm.object_accessor_keys(h);
            let props = realm
                .object_all_keys(h)
                .into_iter()
                .filter(|k| !k.starts_with('#') && !accessor_keys.contains(k))
                .map(|k| {
                    let v = realm.get_property(h, &k).unwrap_or(NanBox::undefined());
                    let sv = snap_val(v, &mut index_of, &mut order, &mut intern);
                    let hidden = !realm.property_is_enumerable(h, &k);
                    (k, sv, hidden)
                })
                .collect();
            // Accessor (getter/setter) properties: capture each function reference.
            let accessors = accessor_keys
                .into_iter()
                .filter(|k| !k.starts_with('#'))
                .filter_map(|k| {
                    let (g, s) = realm.accessor(h, &k)?;
                    let gv = snap_val(g, &mut index_of, &mut order, &mut intern);
                    let sv = snap_val(s, &mut index_of, &mut order, &mut intern);
                    let hidden = !realm.property_is_enumerable(h, &k);
                    Some((k, gv, sv, hidden))
                })
                .collect();
            // The `[[Prototype]]` link (captured as a ref, or Null when absent).
            let proto = match realm.object_proto(h) {
                Some(p) => snap_val(
                    NanBox::handle(p.to_raw()),
                    &mut index_of,
                    &mut order,
                    &mut intern,
                ),
                None => SnapVal::Null,
            };
            SnapCell::Object {
                props,
                accessors,
                proto,
            }
        } else {
            // An unsupported cell kind: record it as an empty object so indices
            // stay aligned (its references are dropped).
            SnapCell::Object {
                props: Vec::new(),
                accessors: Vec::new(),
                proto: SnapVal::Null,
            }
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
            SnapCell::Object { .. } => (realm.new_object(), None),
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
                // SNAP-2: `func_id` is taken from the snapshot verbatim and is
                // NOT validated against any loaded function table here — it is a
                // raw index into whatever program is later run. A restored
                // closure must therefore only be invoked when the matching
                // compiled program is loaded; invoking one against a different
                // (or absent) program is undefined. The FFI restore path never
                // calls restored closures, so this is currently unreachable.
                (realm.new_function(*func_id, innermost), Some(chain))
            }
            SnapCell::Collection { is_set, .. } => (realm.new_collection(*is_set), None),
            SnapCell::Promise { .. } => (realm.new_promise(), None),
            SnapCell::Proxy { .. } => {
                // A placeholder proxy (a dummy object for both slots); pass 2 fills
                // in the real target/handler once every cell's handle exists.
                let d = realm.new_object();
                (realm.new_proxy(d, d), None)
            }
            SnapCell::RegExp {
                source,
                flags,
                last_index,
            } => {
                let h = realm.new_regexp(source, flags);
                realm.set_regex_last_index(h, *last_index);
                (h, None)
            }
            SnapCell::Symbol { description } => (realm.new_symbol(description), None),
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
            SnapCell::Object {
                props,
                accessors,
                proto,
            } => {
                for (k, v, hidden) in props {
                    let val = resolve(v, &handles);
                    realm.set_property(*h, k, val);
                    if *hidden {
                        realm.mark_hidden(*h, k);
                    }
                }
                for (k, g, s, hidden) in accessors {
                    let getter = resolve(g, &handles);
                    let setter = resolve(s, &handles);
                    realm.define_accessor(*h, k, getter, setter);
                    if *hidden {
                        realm.mark_hidden(*h, k);
                    }
                }
                if let Some(p) = resolve(proto, &handles).as_handle() {
                    realm.set_object_proto(*h, Some(Handle::from_raw(p)));
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
            SnapCell::Collection { entries, .. } => {
                for (k, v) in entries {
                    let (key, val) = (resolve(k, &handles), resolve(v, &handles));
                    realm.collection_set(*h, key, val);
                }
            }
            SnapCell::Promise { status, value } => {
                let val = resolve(value, &handles);
                if let Some(state) = realm.promise_state(*h) {
                    let mut s = state.borrow_mut();
                    s.status = match status {
                        1 => crate::cell::PromiseStatus::Fulfilled,
                        2 => crate::cell::PromiseStatus::Rejected,
                        _ => crate::cell::PromiseStatus::Pending,
                    };
                    s.value = val;
                }
            }
            SnapCell::Proxy { target, handler } => {
                if let (Some(t), Some(hd)) = (
                    resolve(target, &handles).as_handle(),
                    resolve(handler, &handles).as_handle(),
                ) {
                    realm.proxy_set_targets(*h, Handle::from_raw(t), Handle::from_raw(hd));
                }
            }
            // A `RegExp`/`Symbol` is fully materialized in pass 1 (no refs to fill).
            SnapCell::RegExp { .. } | SnapCell::Symbol { .. } => {}
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
            SnapCell::Object {
                props,
                accessors,
                proto,
            } => {
                out.push(2);
                w_u32(props.len() as u32, &mut out);
                for (k, v, hidden) in props {
                    w_str(k, &mut out);
                    w_val(v, &mut out);
                    out.push(u8::from(*hidden));
                }
                w_u32(accessors.len() as u32, &mut out);
                for (k, g, s, hidden) in accessors {
                    w_str(k, &mut out);
                    w_val(g, &mut out);
                    w_val(s, &mut out);
                    out.push(u8::from(*hidden));
                }
                w_val(proto, &mut out);
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
            SnapCell::Collection { is_set, entries } => {
                out.push(6);
                out.push(u8::from(*is_set));
                w_u32(entries.len() as u32, &mut out);
                for (k, v) in entries {
                    w_val(k, &mut out);
                    w_val(v, &mut out);
                }
            }
            SnapCell::Promise { status, value } => {
                out.push(7);
                out.push(*status);
                w_val(value, &mut out);
            }
            SnapCell::Proxy { target, handler } => {
                out.push(8);
                w_val(target, &mut out);
                w_val(handler, &mut out);
            }
            SnapCell::RegExp {
                source,
                flags,
                last_index,
            } => {
                out.push(9);
                w_str(source, &mut out);
                w_str(flags, &mut out);
                w_u32(*last_index as u32, &mut out);
            }
            SnapCell::Symbol { description } => {
                out.push(10);
                w_str(description, &mut out);
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
    /// Bytes left unread in the buffer. Used to cap untrusted count fields: each
    /// serialized element costs at least one byte, so a reservation can never
    /// legitimately exceed what remains. Clamping a hostile `0xFFFFFFFF` count
    /// to this bound turns an allocation bomb (capacity-overflow panic or an
    /// uncatchable alloc abort) into a small reservation that the per-element
    /// read loop then rejects with `Truncated`.
    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.p)
    }
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
    let mut roots = Vec::with_capacity(n_roots.min(r.remaining()));
    for _ in 0..n_roots {
        roots.push(r.u32()? as usize);
    }
    let n_cells = r.u32()? as usize;
    let mut cells = Vec::with_capacity(n_cells.min(r.remaining()));
    for _ in 0..n_cells {
        let cell = match r.u8()? {
            0 => SnapCell::Str(r.string()?),
            1 => {
                let n = r.u32()? as usize;
                let mut vals = Vec::with_capacity(n.min(r.remaining()));
                for _ in 0..n {
                    vals.push(r.val()?);
                }
                SnapCell::Array(vals)
            }
            2 => {
                let n = r.u32()? as usize;
                let mut props = Vec::with_capacity(n.min(r.remaining()));
                for _ in 0..n {
                    let k = r.string()?;
                    let v = r.val()?;
                    let hidden = r.u8()? != 0;
                    props.push((k, v, hidden));
                }
                let na = r.u32()? as usize;
                let mut accessors = Vec::with_capacity(na.min(r.remaining()));
                for _ in 0..na {
                    let k = r.string()?;
                    let g = r.val()?;
                    let s = r.val()?;
                    let hidden = r.u8()? != 0;
                    accessors.push((k, g, s, hidden));
                }
                let proto = r.val()?;
                SnapCell::Object {
                    props,
                    accessors,
                    proto,
                }
            }
            3 => SnapCell::Date(r.f64()?),
            4 => SnapCell::BigInt(r.string()?),
            5 => {
                let func_id = r.u32()?;
                let nf = r.u32()? as usize;
                let mut frames = Vec::with_capacity(nf.min(r.remaining()));
                for _ in 0..nf {
                    let nv = r.u32()? as usize;
                    let mut vars = Vec::with_capacity(nv.min(r.remaining()));
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
            6 => {
                let is_set = r.u8()? != 0;
                let n = r.u32()? as usize;
                let mut entries = Vec::with_capacity(n.min(r.remaining()));
                for _ in 0..n {
                    let k = r.val()?;
                    let v = r.val()?;
                    entries.push((k, v));
                }
                SnapCell::Collection { is_set, entries }
            }
            7 => {
                let status = r.u8()?;
                let value = r.val()?;
                SnapCell::Promise { status, value }
            }
            8 => {
                let target = r.val()?;
                let handler = r.val()?;
                SnapCell::Proxy { target, handler }
            }
            9 => {
                let source = r.string()?;
                let flags = r.string()?;
                let last_index = r.u32()? as usize;
                SnapCell::RegExp {
                    source,
                    flags,
                    last_index,
                }
            }
            10 => SnapCell::Symbol {
                description: r.string()?,
            },
            t => return Err(SnapError::BadTag(t)),
        };
        cells.push(cell);
    }
    Ok(Snapshot { cells, roots })
}

pub use store::{ArtifactStore, address_hex, content_address, host_keyed_address, host_tag};

/// Content-addressed store for serialized snapshots (`ROADMAP.md` §2.3): a
/// snapshot's bytes are keyed by a stable hash of their own content, so identical
/// snapshots deduplicate to a single entry and the key doubles as an integrity
/// tag (a fetch can re-verify the bytes still hash to the address they were asked
/// for). Pure, `alloc`-only Rust — the address is deterministic across runs and
/// runtimes (no `Date`/`Random`).
pub mod store {
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec::Vec;

    /// The content address of `bytes`: an FNV-1a 64-bit hash. Deterministic, so
    /// the same content always maps to the same address.
    #[must_use]
    pub fn content_address(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        }
        h
    }

    /// The current host's tag: its pointer width, byte order, and architecture
    /// family. A host-native snapshot is only valid on a matching host (it would
    /// need byte-swap/width conversion otherwise), so artifacts are keyed per host.
    /// Stable for a given build target.
    #[must_use]
    pub fn host_tag() -> u32 {
        let ptr_bits = (core::mem::size_of::<usize>() as u32) & 0xff; // 32 or 64
        let little_endian = u32::from(cfg!(target_endian = "little"));
        let arch: u32 = if cfg!(target_arch = "x86_64") {
            1
        } else if cfg!(target_arch = "aarch64") {
            2
        } else if cfg!(target_arch = "x86") {
            3
        } else {
            0
        };
        (ptr_bits << 8) | (little_endian << 4) | arch
    }

    /// A **host-qualified** address: the content hash of `key_source` (e.g. a
    /// program's source) mixed with `host_tag`, so the same source keyed for two
    /// hosts that differ in byte order or pointer width resolves to *different*
    /// addresses — a host-native artifact never aliases another host's.
    #[must_use]
    pub fn host_keyed_address(key_source: &[u8], host_tag: u32) -> u64 {
        let mut h = content_address(key_source);
        for b in host_tag.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        }
        h
    }

    /// Renders a content address as 16 lowercase hex digits (e.g. for a filename
    /// in an on-disk store).
    #[must_use]
    pub fn address_hex(addr: u64) -> String {
        let mut s = String::with_capacity(16);
        for i in (0..16).rev() {
            let nib = (addr >> (i * 4)) & 0xf;
            // `nib` is 0..=15, always a valid hex digit.
            s.push(char::from_digit(nib as u32, 16).unwrap_or('0'));
        }
        s
    }

    /// An in-memory content-addressed store of serialized snapshot artifacts.
    #[derive(Default)]
    pub struct ArtifactStore {
        objects: BTreeMap<u64, Vec<u8>>,
    }

    impl ArtifactStore {
        /// Creates an empty store.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// Stores `bytes` under their content address and returns it. Storing the
        /// same content again is idempotent — deduplicated to one entry.
        pub fn put(&mut self, bytes: Vec<u8>) -> u64 {
            let addr = content_address(&bytes);
            self.objects.entry(addr).or_insert(bytes);
            addr
        }

        /// Fetches the bytes stored at `addr`, if present.
        #[must_use]
        pub fn get(&self, addr: u64) -> Option<&[u8]> {
            self.objects.get(&addr).map(Vec::as_slice)
        }

        /// Stores `artifact` under the host-qualified address of `key_source` (e.g.
        /// the program source) for `host_tag`, returning that address — so the same
        /// source resolves to a different artifact per host.
        pub fn put_for_host(&mut self, key_source: &[u8], host_tag: u32, artifact: Vec<u8>) -> u64 {
            let addr = host_keyed_address(key_source, host_tag);
            self.objects.entry(addr).or_insert(artifact);
            addr
        }

        /// Fetches the artifact for `key_source` on `host_tag`, if one was stored —
        /// a lookup that misses cleanly for a host the artifact wasn't built for.
        #[must_use]
        pub fn get_for_host(&self, key_source: &[u8], host_tag: u32) -> Option<&[u8]> {
            self.get(host_keyed_address(key_source, host_tag))
        }

        /// Fetches the bytes at `addr`, re-verifying they still hash to it — so a
        /// corrupted or forged entry returns `None` instead of bad data.
        #[must_use]
        pub fn get_verified(&self, addr: u64) -> Option<&[u8]> {
            let b = self.objects.get(&addr)?;
            (content_address(b) == addr).then_some(b.as_slice())
        }

        /// Whether an artifact is stored at `addr`.
        #[must_use]
        pub fn contains(&self, addr: u64) -> bool {
            self.objects.contains_key(&addr)
        }

        /// Number of distinct artifacts held.
        #[must_use]
        pub fn len(&self) -> usize {
            self.objects.len()
        }

        /// Whether the store holds no artifacts.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.objects.is_empty()
        }
    }
}

#[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
pub use mmap::restore_file;

/// The `mmap`-backed snapshot reload — the zero-copy D′ path for heap snapshots:
/// a `KSNP` artifact is mapped read-only and deserialized in place, then restored.
#[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
mod mmap {
    use super::{Snapshot, deserialize, restore};
    use crate::heap::Handle;
    use crate::realm::Realm;
    use alloc::vec::Vec;
    use std::os::unix::io::AsRawFd;

    const PROT_READ: usize = 0x1;
    const MAP_PRIVATE: usize = 0x02;
    const SYS_MMAP: usize = 9;
    const SYS_MUNMAP: usize = 11;

    /// SAFETY: issues the `syscall` instruction with the System V syscall ABI.
    #[allow(unsafe_code)]
    unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize) -> isize {
        let ret: isize;
        // SAFETY: a single syscall with documented inputs/clobbers.
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") n as isize => ret,
                in("rdi") a1, in("rsi") a2, in("rdx") a3,
                in("r10") a4, in("r8") a5, in("r9") 0usize,
                out("rcx") _, out("r11") _,
                options(nostack, preserves_flags),
            );
        }
        ret
    }

    /// Reloads the snapshot at `path` from a memory mapping and restores it into
    /// `realm`, returning the new root handles — the `mmap` zero-copy reload path.
    ///
    /// # Errors
    /// `io::Error` on a failed open/`mmap`, or `InvalidData` wrapping the
    /// [`SnapError`](super::SnapError) if the mapped bytes don't deserialize.
    pub fn restore_file(realm: &mut Realm, path: &str) -> std::io::Result<Vec<Handle>> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty snapshot",
            ));
        }
        let fd = file.as_raw_fd() as usize;
        // SAFETY: a standard read-only file mapping.
        #[allow(unsafe_code)]
        let raw = unsafe { syscall6(SYS_MMAP, 0, len, PROT_READ, MAP_PRIVATE, fd) };
        if (-4095..0).contains(&raw) {
            return Err(std::io::Error::last_os_error());
        }
        let ptr = raw as *const u8;
        // SAFETY: `ptr..ptr+len` is a valid read-only mapping of the whole file,
        // alive until the `munmap` below; deserialize only reads it.
        #[allow(unsafe_code)]
        let parsed: Result<Snapshot, super::SnapError> = {
            let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
            deserialize(bytes)
        };
        // SAFETY: unmapping exactly the region we mapped.
        #[allow(unsafe_code)]
        unsafe {
            syscall6(SYS_MUNMAP, ptr as usize, len, 0, 0, 0);
        }
        let snap = parsed.map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, alloc::format!("{e:?}"))
        })?;
        Ok(restore(realm, &snap))
    }
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
    fn content_address_is_stable_and_distinguishing() {
        use super::store::{address_hex, content_address};
        // Deterministic: equal content → equal address; different → (almost surely)
        // different.
        assert_eq!(content_address(b"hello"), content_address(b"hello"));
        assert_ne!(content_address(b"hello"), content_address(b"hellp"));
        assert_ne!(content_address(b""), content_address(b"\0"));
        // Hex rendering is 16 lowercase digits.
        let h = address_hex(content_address(b"abc"));
        assert_eq!(h.len(), 16);
        assert!(
            h.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn artifact_store_dedups_and_verifies() {
        use super::store::{ArtifactStore, content_address};
        let mut store = ArtifactStore::new();
        assert!(store.is_empty());

        // Two identical artifacts collapse to one entry under the same address.
        let a1 = store.put(alloc::vec![1, 2, 3, 4]);
        let a2 = store.put(alloc::vec![1, 2, 3, 4]);
        assert_eq!(a1, a2, "identical content → same address");
        assert_eq!(store.len(), 1, "deduplicated");

        // A different artifact gets its own address.
        let b = store.put(alloc::vec![9, 9, 9]);
        assert_ne!(a1, b);
        assert_eq!(store.len(), 2);

        // Fetch by address round-trips the exact bytes.
        assert_eq!(store.get(a1), Some(&[1, 2, 3, 4][..]));
        assert!(store.contains(b));
        assert_eq!(store.get(content_address(b"absent")), None);

        // A verified fetch confirms the bytes still hash to the requested address.
        assert_eq!(store.get_verified(a1), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn host_keyed_addresses_separate_hosts() {
        use super::store::{ArtifactStore, host_keyed_address, host_tag};
        // The host tag is stable for this build.
        assert_eq!(host_tag(), host_tag());

        // Same source, different host tag → different address; same tag → same.
        let src = b"function f(){ return 1 }";
        let h_native = host_tag();
        let h_other = h_native ^ 0xff; // a hypothetical different host
        assert_ne!(
            host_keyed_address(src, h_native),
            host_keyed_address(src, h_other),
            "a host-native artifact never aliases another host's"
        );
        assert_eq!(
            host_keyed_address(src, h_native),
            host_keyed_address(src, h_native),
            "deterministic per (source, host)"
        );
        // Different source, same host → different address.
        assert_ne!(
            host_keyed_address(src, h_native),
            host_keyed_address(b"other source", h_native)
        );

        // The store resolves a source to a per-host artifact, and misses cleanly
        // for a host it wasn't built for.
        let mut store = ArtifactStore::new();
        let addr = store.put_for_host(src, h_native, alloc::vec![1, 2, 3]);
        assert_eq!(store.get_for_host(src, h_native), Some(&[1, 2, 3][..]));
        assert_eq!(store.get_for_host(src, h_other), None, "wrong host misses");
        assert_eq!(addr, host_keyed_address(src, h_native));
    }

    #[test]
    fn artifact_store_round_trips_a_snapshot() {
        // End-to-end: capture → serialize → store by content address → fetch →
        // deserialize → restore. The reload reproduces the object graph.
        use super::store::ArtifactStore;
        let mut realm = Realm::new();
        let obj = realm.new_object();
        realm.set_property(obj, "x", NanBox::number(7.0));
        let inner = realm.new_array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        realm.set_property(obj, "list", NanBox::handle(inner.to_raw()));

        let bytes = serialize(&capture(&realm, &[obj]));
        let mut store = ArtifactStore::new();
        let addr = store.put(bytes);

        let reloaded = store.get_verified(addr).expect("stored artifact");
        let snap = deserialize(reloaded).expect("deserialize");
        let mut realm2 = Realm::new();
        let roots = restore(&mut realm2, &snap);
        let o2 = roots[0];
        assert_eq!(realm2.get_property(o2, "x").unwrap().as_number(), Some(7.0));
        let list = realm2
            .get_property(o2, "list")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(realm2.array_length(Handle::from_raw(list)), Some(2));
    }

    #[test]
    fn round_trips_a_cross_kind_cycle() {
        // A cycle threading three different cell kinds: object → array → object,
        // and object → Map → object. The two-pass restore must rejoin them all to
        // the *same* restored object under fresh handles.
        let mut realm = Realm::new();
        let a = realm.new_object();
        let arr = realm.new_array(alloc::vec![NanBox::handle(a.to_raw())]); // arr[0] = a
        let map = realm.new_collection(false);
        realm.collection_set(map, NanBox::number(1.0), NanBox::handle(a.to_raw())); // map[1] = a
        realm.set_property(a, "arr", NanBox::handle(arr.to_raw()));
        realm.set_property(a, "map", NanBox::handle(map.to_raw()));

        // Through the full serialize → bytes → deserialize → restore path.
        let bytes = serialize(&capture(&realm, &[a]));
        let mut realm2 = Realm::new();
        let a2 = restore(&mut realm2, &deserialize(&bytes).expect("deserialize"))[0];

        // arr[0] and map[1] both point back to the *same* restored `a2`.
        let arr2 = Handle::from_raw(realm2.get_property(a2, "arr").unwrap().as_handle().unwrap());
        assert_eq!(
            realm2.array_elements(arr2).unwrap()[0].as_handle(),
            Some(a2.to_raw()),
            "array element rejoins the cycle"
        );
        let map2 = Handle::from_raw(realm2.get_property(a2, "map").unwrap().as_handle().unwrap());
        assert_eq!(
            realm2
                .collection_get(map2, NanBox::number(1.0))
                .unwrap()
                .as_handle(),
            Some(a2.to_raw()),
            "map value rejoins the cycle"
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

    #[cfg(all(feature = "std", target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn mmap_reload_restores_a_snapshot() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        let name = NanBox::handle(realm.new_string("mapped").to_raw());
        realm.set_property(obj, "name", name);
        realm.set_property(obj, "n", NanBox::number(99.0));
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        realm.set_property(obj, "items", NanBox::handle(arr.to_raw()));
        // Richer cells reachable from the root: a Map and a closure.
        let map = realm.new_collection(false);
        realm.collection_set(map, NanBox::number(1.0), NanBox::number(10.0));
        realm.set_property(obj, "map", NanBox::handle(map.to_raw()));
        let scope = crate::env::Scope::root();
        scope.declare("cap", NanBox::number(7.0));
        let func = realm.new_function(3, scope);
        realm.set_property(obj, "fn", NanBox::handle(func.to_raw()));
        // The remaining cell kinds: Date, BigInt, a settled Promise, and a Proxy —
        // so the mmap reload exercises all nine reference cell kinds.
        let date = realm.new_date(1234.0);
        realm.set_property(obj, "date", NanBox::handle(date.to_raw()));
        let big = realm
            .new_bigint(crate::bignum::BigInt::from_str_radix("90071992547409930", 10).unwrap());
        realm.set_property(obj, "big", NanBox::handle(big.to_raw()));
        let prom = realm.new_promise();
        {
            let st = realm.promise_state(prom).unwrap();
            let mut s = st.borrow_mut();
            s.status = crate::cell::PromiseStatus::Fulfilled;
            s.value = NanBox::number(8.0);
        }
        realm.set_property(obj, "prom", NanBox::handle(prom.to_raw()));
        let target = realm.new_object();
        realm.set_property(target, "t", NanBox::number(1.0));
        let handler = realm.new_object();
        let proxy = realm.new_proxy(target, handler);
        realm.set_property(obj, "proxy", NanBox::handle(proxy.to_raw()));

        // Serialize to a temp file, then reload it via the mmap zero-copy path.
        let bytes = serialize(&capture(&realm, &[obj]));
        let path = std::format!("/tmp/kataan-snap-{}.ksnp", std::process::id());
        std::fs::write(&path, &bytes).expect("write snapshot");

        let mut realm2 = Realm::new();
        let roots = restore_file(&mut realm2, &path).expect("mmap reload");
        let _ = std::fs::remove_file(&path);

        let r2 = roots[0];
        let name2 = realm2
            .get_property(r2, "name")
            .unwrap()
            .as_handle()
            .unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(name2)),
            Some(String::from("mapped"))
        );
        assert_eq!(realm2.get_property(r2, "n"), Some(NanBox::number(99.0)));
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
            2
        );
        // The Map and closure came through the mmap reload too.
        let map2 = Handle::from_raw(realm2.get_property(r2, "map").unwrap().as_handle().unwrap());
        assert_eq!(
            realm2.collection_get(map2, NanBox::number(1.0)),
            Some(NanBox::number(10.0))
        );
        let fn2 = Handle::from_raw(realm2.get_property(r2, "fn").unwrap().as_handle().unwrap());
        let (fid, sc) = realm2.function_at(fn2).expect("restored closure");
        assert_eq!(fid, 3);
        assert_eq!(sc.get("cap"), Some(NanBox::number(7.0)));
        // Date / BigInt / Promise / Proxy all came through the mmap reload.
        let date2 = Handle::from_raw(
            realm2
                .get_property(r2, "date")
                .unwrap()
                .as_handle()
                .unwrap(),
        );
        assert_eq!(realm2.date_at(date2), Some(1234.0));
        let big2 = Handle::from_raw(realm2.get_property(r2, "big").unwrap().as_handle().unwrap());
        assert_eq!(
            realm2.bigint_at(big2).map(|b| b.to_str_radix(10)),
            Some(String::from("90071992547409930"))
        );
        let prom2 = Handle::from_raw(
            realm2
                .get_property(r2, "prom")
                .unwrap()
                .as_handle()
                .unwrap(),
        );
        {
            let s = realm2.promise_state(prom2).unwrap();
            let s = s.borrow();
            assert!(matches!(s.status, crate::cell::PromiseStatus::Fulfilled));
            assert_eq!(s.value, NanBox::number(8.0));
        }
        let proxy2 = Handle::from_raw(
            realm2
                .get_property(r2, "proxy")
                .unwrap()
                .as_handle()
                .unwrap(),
        );
        let (t2, _h2) = realm2.proxy_at(proxy2).expect("restored proxy");
        assert_eq!(realm2.get_property(t2, "t"), Some(NanBox::number(1.0)));

        // The mmap-reloaded heap is a valid moving-GC heap: compaction keeps it.
        let mut roots2 = [r2];
        realm2.compact(&mut roots2);
        assert_eq!(
            realm2.get_property(roots2[0], "n"),
            Some(NanBox::number(99.0))
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
    fn snapshots_preserve_accessor_properties() {
        use crate::env::Scope;
        let mut realm = Realm::new();
        // A getter and setter (closures over `func_id`s 11 and 12) installed as an
        // accessor property, marked non-enumerable like a class accessor.
        let getter = realm.new_function(11, Scope::root());
        let setter = realm.new_function(12, Scope::root());
        let obj = realm.new_object();
        realm.define_accessor(
            obj,
            "value",
            NanBox::handle(getter.to_raw()),
            NanBox::handle(setter.to_raw()),
        );
        realm.mark_hidden(obj, "value");

        let snap = deserialize(&serialize(&capture(&realm, &[obj]))).expect("round-trip");
        let mut realm2 = Realm::new();
        let o2 = restore(&mut realm2, &snap)[0];

        // The accessor survives as an accessor (not a data prop), non-enumerable,
        // with both functions restored to the same code identities.
        let (g2, s2) = realm2.accessor(o2, "value").expect("accessor restored");
        assert_eq!(
            realm2
                .function_at(Handle::from_raw(g2.as_handle().unwrap()))
                .map(|(id, _)| id),
            Some(11),
            "getter restored"
        );
        assert_eq!(
            realm2
                .function_at(Handle::from_raw(s2.as_handle().unwrap()))
                .map(|(id, _)| id),
            Some(12),
            "setter restored"
        );
        assert!(
            !realm2.property_is_enumerable(o2, "value"),
            "non-enumerable kept"
        );
        // It is an accessor, so there's no plain data slot of the same name.
        assert_eq!(
            realm2.get_property(o2, "value"),
            None,
            "no shadow data slot"
        );
    }

    #[test]
    fn snapshots_preserve_prototype_links() {
        let mut realm = Realm::new();
        // A prototype with a method-ish property, and a child created over it
        // (`Object.create(proto)` shape) carrying its own property.
        let proto = realm.new_object();
        let pv = NanBox::handle(realm.new_string("inherited").to_raw());
        realm.set_property(proto, "kind", pv);
        let child = realm.new_object_with_proto(Some(proto));
        let cv = NanBox::handle(realm.new_string("own").to_raw());
        realm.set_property(child, "self", cv);

        let snap = deserialize(&serialize(&capture(&realm, &[child]))).expect("round-trip");
        let mut realm2 = Realm::new();
        let c2 = restore(&mut realm2, &snap)[0];

        // The child's own property and its prototype link both survive, and the
        // inherited property resolves through the restored chain.
        let proto2 = realm2.object_proto(c2).expect("prototype link restored");
        let inherited = realm2
            .get_property(proto2, "kind")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| realm2.string_value(h));
        assert_eq!(
            inherited,
            Some(String::from("inherited")),
            "inherited prop via proto"
        );
        let own = realm2
            .get_property(c2, "self")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| realm2.string_value(h));
        assert_eq!(own, Some(String::from("own")), "own prop kept");
    }

    #[test]
    fn snapshots_preserve_non_enumerable_and_hidden_slots() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        // An enumerable property, a non-enumerable one, and an engine-internal
        // hidden slot (the `\0`-prefixed convention used for typed-array kind,
        // bound-function targets, error stacks, …).
        let vis = NanBox::handle(realm.new_string("v").to_raw());
        realm.set_property(obj, "visible", vis);
        let hid = NanBox::handle(realm.new_string("h").to_raw());
        realm.set_property(obj, "secret", hid);
        realm.mark_hidden(obj, "secret");
        realm.set_hidden_property(obj, "\u{0}slot", NanBox::number(42.0));

        let snap = deserialize(&serialize(&capture(&realm, &[obj]))).expect("round-trip");
        let mut realm2 = Realm::new();
        let o2 = restore(&mut realm2, &snap)[0];

        // All three survive, with enumerability preserved.
        assert!(
            realm2.property_is_enumerable(o2, "visible"),
            "enumerable kept"
        );
        assert!(
            !realm2.property_is_enumerable(o2, "secret"),
            "non-enumerable kept"
        );
        assert_eq!(
            realm2
                .get_property(o2, "\u{0}slot")
                .and_then(|v| v.as_number()),
            Some(42.0),
            "internal hidden slot survives"
        );
        // Enumerable view shows only the visible key.
        assert_eq!(
            realm2.object_keys(o2).unwrap(),
            alloc::vec![String::from("visible")]
        );
        assert!(
            realm2.has_own(o2, "secret"),
            "non-enumerable is an own prop"
        );
    }

    #[test]
    fn snapshots_preserve_object_key_order() {
        let mut realm = Realm::new();
        // Insertion order matters for JS object enumeration; it must survive the
        // snapshot round-trip unchanged (no reordering, no dropped keys).
        let obj = realm.new_object();
        for k in ["zebra", "apple", "_mid", "Last", "n2"] {
            let v = NanBox::handle(realm.new_string(k).to_raw());
            realm.set_property(obj, k, v);
        }
        let before = realm.object_keys(obj).unwrap();

        let snap = capture(&realm, &[obj]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        let mut realm2 = Realm::new();
        let o2 = restore(&mut realm2, &reloaded)[0];

        assert_eq!(
            realm2.object_keys(o2).unwrap(),
            before,
            "object key insertion order is preserved across the snapshot"
        );
    }

    #[test]
    fn snapshots_regexps() {
        let mut realm = Realm::new();
        // A RegExp with flags and an advanced lastIndex, reachable from an object.
        let re = realm.new_regexp("\\d+", "gi");
        realm.set_regex_last_index(re, 7);
        let obj = realm.new_object();
        realm.set_property(obj, "re", NanBox::handle(re.to_raw()));

        let snap = capture(&realm, &[obj]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap);
        let mut realm2 = Realm::new();
        let o2 = restore(&mut realm2, &reloaded)[0];

        let re2 = Handle::from_raw(realm2.get_property(o2, "re").unwrap().as_handle().unwrap());
        assert_eq!(
            realm2.regexp_at(re2),
            Some((String::from("\\d+"), String::from("gi")))
        );
        assert_eq!(realm2.regex_last_index(re2), 7, "lastIndex survives");
    }

    #[test]
    fn snapshots_symbols() {
        let mut realm = Realm::new();
        // One symbol referenced from two array slots — its in-snapshot identity
        // must survive (both slots reload to the *same* restored symbol).
        let sym = realm.new_symbol("tag");
        let arr = realm.new_array(alloc::vec![
            NanBox::handle(sym.to_raw()),
            NanBox::handle(sym.to_raw()),
        ]);

        let snap = capture(&realm, &[arr]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap);
        let mut realm2 = Realm::new();
        let a2 = restore(&mut realm2, &reloaded)[0];

        let e0 = realm2.get_element(a2, 0).as_handle().unwrap();
        let e1 = realm2.get_element(a2, 1).as_handle().unwrap();
        assert_eq!(e0, e1, "shared symbol identity preserved");
        assert_eq!(
            realm2.symbol_at(Handle::from_raw(e0)).map(|(d, _)| d),
            Some(String::from("tag")),
            "description round-trips"
        );
    }

    #[test]
    fn snapshots_proxies() {
        let mut realm = Realm::new();
        // A proxy over a target object { v: 7 } with a handler { tag: "h" }.
        let target = realm.new_object();
        realm.set_property(target, "v", NanBox::number(7.0));
        let handler = realm.new_object();
        let htag = NanBox::handle(realm.new_string("h").to_raw());
        realm.set_property(handler, "tag", htag);
        let proxy = realm.new_proxy(target, handler);

        let snap = capture(&realm, &[proxy]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap);
        let mut realm2 = Realm::new();
        let p2 = restore(&mut realm2, &reloaded)[0];

        // The proxy's target and handler survive, with their properties.
        let (t2, h2) = realm2.proxy_at(p2).expect("restored proxy");
        assert_eq!(realm2.get_property(t2, "v"), Some(NanBox::number(7.0)));
        let tag = realm2.get_property(h2, "tag").unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(tag.as_handle().unwrap())),
            Some(String::from("h"))
        );
    }

    #[test]
    fn snapshots_settled_promises() {
        let mut realm = Realm::new();
        // A fulfilled promise (value = an object) and a rejected one (reason = 42).
        let fulfilled = realm.new_promise();
        let payload = realm.new_object();
        let tag = NanBox::handle(realm.new_string("ok").to_raw());
        realm.set_property(payload, "tag", tag);
        {
            let st = realm.promise_state(fulfilled).unwrap();
            let mut s = st.borrow_mut();
            s.status = crate::cell::PromiseStatus::Fulfilled;
            s.value = NanBox::handle(payload.to_raw());
        }
        let rejected = realm.new_promise();
        {
            let st = realm.promise_state(rejected).unwrap();
            let mut s = st.borrow_mut();
            s.status = crate::cell::PromiseStatus::Rejected;
            s.value = NanBox::number(42.0);
        }

        let snap = capture(&realm, &[fulfilled, rejected]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap);
        let mut realm2 = Realm::new();
        let roots = restore(&mut realm2, &reloaded);

        // The fulfilled promise's status + object value survive.
        let st = realm2.promise_state(roots[0]).unwrap();
        let s = st.borrow();
        assert_eq!(s.status, crate::cell::PromiseStatus::Fulfilled);
        let obj = Handle::from_raw(s.value.as_handle().unwrap());
        let tag2 = realm2.get_property(obj, "tag").unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(tag2.as_handle().unwrap())),
            Some(String::from("ok"))
        );
        // The rejected promise's reason survives.
        let st2 = realm2.promise_state(roots[1]).unwrap();
        let s2 = st2.borrow();
        assert_eq!(s2.status, crate::cell::PromiseStatus::Rejected);
        assert_eq!(s2.value, NanBox::number(42.0));
    }

    #[test]
    fn snapshots_maps_and_sets() {
        let mut realm = Realm::new();
        // A Map { 1 → "one", 2 → obj{tag:"two"} } and a Set { 10, 20, 30 }.
        let map = realm.new_collection(false);
        let one = NanBox::handle(realm.new_string("one").to_raw());
        realm.collection_set(map, NanBox::number(1.0), one);
        let two_obj = realm.new_object();
        let two_tag = NanBox::handle(realm.new_string("two").to_raw());
        realm.set_property(two_obj, "tag", two_tag);
        realm.collection_set(map, NanBox::number(2.0), NanBox::handle(two_obj.to_raw()));

        let set = realm.new_collection(true);
        for n in [10.0, 20.0, 30.0] {
            realm.collection_set(set, NanBox::number(n), NanBox::number(n));
        }

        // Both as roots: capture → serialize → deserialize → restore.
        let snap = capture(&realm, &[map, set]);
        let bytes = serialize(&snap);
        let reloaded = deserialize(&bytes).expect("deserialize");
        assert_eq!(reloaded, snap);
        let mut realm2 = Realm::new();
        let roots = restore(&mut realm2, &reloaded);
        let (m2, s2) = (roots[0], roots[1]);

        // The Map survived: 1 → "one", and 2 → an object with tag "two".
        assert_eq!(realm2.collection_is_set(m2), Some(false));
        let v1 = realm2.collection_get(m2, NanBox::number(1.0)).unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(v1.as_handle().unwrap())),
            Some(String::from("one"))
        );
        let v2 = realm2.collection_get(m2, NanBox::number(2.0)).unwrap();
        let tag = realm2
            .get_property(Handle::from_raw(v2.as_handle().unwrap()), "tag")
            .unwrap();
        assert_eq!(
            realm2.string_value(Handle::from_raw(tag.as_handle().unwrap())),
            Some(String::from("two"))
        );

        // The Set survived with its three members.
        assert_eq!(realm2.collection_is_set(s2), Some(true));
        assert_eq!(realm2.collection_size(s2), Some(3));
        assert!(realm2.collection_has(s2, NanBox::number(20.0)));
        assert!(!realm2.collection_has(s2, NanBox::number(99.0)));
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
