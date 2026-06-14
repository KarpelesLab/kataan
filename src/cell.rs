//! Heap cells — the reference types the managed heap holds (`ROADMAP.md` §3).
//!
//! A NaN-boxed handle can point at any *reference* value, not just a plain
//! object: a string, an array, a function, … all live on the heap. `Cell` is
//! that sum — what a [`heap::Heap`](crate::heap::Heap) slot actually stores in
//! the performance model. This turn covers the three core kinds:
//! - `Cell::Object` — a shape + slots object,
//! - `Cell::Str` — a string *value*, backed by a [`rope::Rope`](crate::rope::Rope)
//!   so concatenation stays O(1),
//! - `Cell::Array` — a dense vector of element values.
//!
//! A cell knows how to enumerate its outgoing handles, so the collector traces a
//! mixed object/array/string graph uniformly: objects report their property
//! slots, arrays their elements, strings nothing.
//!
//! Pure, safe `alloc`-only Rust.

use crate::env::Scope;
use crate::gc::Trace;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::object::Object;
use crate::rope::Rope;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// A `Promise`'s settlement status.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PromiseStatus {
    /// Not yet settled.
    Pending,
    /// Settled with a value.
    Fulfilled,
    /// Settled with a reason.
    Rejected,
}

/// A `then` reaction: the handlers to run on settlement and the dependent
/// promise to settle with their result.
pub struct Reaction {
    /// Called on fulfillment (`undefined` to pass through).
    pub on_fulfilled: NanBox,
    /// Called on rejection (`undefined` to pass through).
    pub on_rejected: NanBox,
    /// The promise produced by `then`, settled with the handler's result.
    pub result: Handle,
    /// A `finally` reaction: the callback runs for side effects but the result
    /// mirrors the original promise (its value/rejection passes through).
    pub finally: bool,
}

/// A `Promise`'s internal state, shared (`Rc`) between the promise and its
/// resolve/reject functions.
pub struct PromiseState {
    /// The settlement status.
    pub status: PromiseStatus,
    /// The fulfillment value or rejection reason.
    pub value: NanBox,
    /// Reactions registered while pending, drained on settlement.
    pub reactions: Vec<Reaction>,
}

/// The free/release hook for an externally-owned byte region, run once when the
/// backing [`ByteStore::External`] is dropped (swept). `ptr`/`len` are exactly
/// the values passed to [`ByteStore::external`].
pub type ExternFree = unsafe fn(ptr: *mut u8, len: usize);

/// The contiguous byte backing of an `ArrayBuffer` / typed array.
///
/// `Owned` is a heap `Vec<u8>` the engine manages. `External` is a caller-owned
/// region wrapped zero-copy (e.g. an IPC shared-memory page): the engine reads
/// and writes it in place and never moves it (the moving GC relocates the
/// `Cell`'s 24-byte header, not the region the pointer addresses), and frees it
/// via the optional [`ExternFree`] hook when the cell is collected.
pub enum ByteStore {
    /// Engine-owned bytes.
    Owned(Vec<u8>),
    /// A zero-copy wrapper over a caller-owned region.
    External(ExternalBytes),
}

/// A zero-copy wrapper over an external byte region (see [`ByteStore::External`]).
/// Its [`Drop`] runs the optional free hook exactly once; a move (GC compaction)
/// is a bitwise copy and does not drop, so the region is freed only on sweep.
pub struct ExternalBytes {
    ptr: *mut u8,
    len: usize,
    free: Option<ExternFree>,
}

impl ByteStore {
    /// Wraps an external region. `ptr` must be non-null, valid for reads/writes
    /// of `len` bytes, and remain valid until `free` (if any) is called on drop;
    /// if `free` is `None` the region must outlive the realm.
    ///
    /// # Safety
    /// The caller upholds the validity/lifetime contract above.
    #[must_use]
    #[allow(unsafe_code)]
    pub unsafe fn external(ptr: *mut u8, len: usize, free: Option<ExternFree>) -> Self {
        ByteStore::External(ExternalBytes { ptr, len, free })
    }

    /// The number of bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            ByteStore::Owned(v) => v.len(),
            ByteStore::External(e) => e.len,
        }
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A read view of the bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            ByteStore::Owned(v) => v,
            // SAFETY: per the `external` contract, `ptr` is valid for `len` bytes
            // for the lifetime of this store (an audited primitive under the
            // crate's `unsafe_code = "deny"` policy).
            #[allow(unsafe_code)]
            ByteStore::External(e) => unsafe { core::slice::from_raw_parts(e.ptr, e.len) },
        }
    }

    /// A mutable view of the bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            ByteStore::Owned(v) => v,
            // SAFETY: as `as_slice`, and `&mut self` guarantees exclusive access.
            #[allow(unsafe_code)]
            ByteStore::External(e) => unsafe { core::slice::from_raw_parts_mut(e.ptr, e.len) },
        }
    }
}

impl Drop for ExternalBytes {
    fn drop(&mut self) {
        if let Some(free) = self.free {
            // SAFETY: `free` was supplied for exactly this `(ptr, len)` and Drop
            // runs once (moves are bitwise and do not drop).
            #[allow(unsafe_code)]
            unsafe {
                free(self.ptr, self.len);
            }
        }
    }
}

/// A heap-allocated reference value.
pub enum Cell {
    /// An ordinary object (shape + value slots).
    Object(Object),
    /// A string value (rope-backed for O(1) concatenation).
    Str(Rope),
    /// A dense array of element values.
    Array(Vec<NanBox>),
    /// Contiguous raw bytes — the backing store of an `ArrayBuffer` (and thus of
    /// the typed-array / `DataView` views over it). One byte per byte (no NaN-box
    /// blow-up); may wrap external memory zero-copy. Holds no handles.
    Bytes(ByteStore),
    /// A typed-array *view* over an `ArrayBuffer`'s [`Cell::Bytes`]: it owns no
    /// elements, reading/writing the shared bytes directly so sibling views and
    /// `DataView`s alias the same storage.
    TypedArray {
        /// The backing `ArrayBuffer`'s bytes cell (element reads/writes go here).
        buffer: Handle,
        /// The `[[ViewedArrayBuffer]]` — the `ArrayBuffer` *object* this view was
        /// created over. `.buffer` returns this handle directly (SameValue stable),
        /// and sibling views (`subarray`) propagate it so they share one object.
        array_buffer: Handle,
        /// Byte offset of the view's first element within the buffer.
        byte_offset: usize,
        /// Element count.
        length: usize,
        /// Element kind index into `TYPED_ARRAY_KINDS` (0..=8).
        kind: u8,
    },
    /// A closure: an index into the interpreter's function table plus the
    /// lexical scope it captured at definition.
    Function {
        /// Index into the owning interpreter's function table (the AST body).
        func_id: u32,
        /// The captured lexical environment.
        env: Scope,
    },
    /// A built-in (native) function, identified by an id the interpreter maps to
    /// a Rust implementation.
    Native(u16),
    /// A native function bound to a heap value (e.g. a promise's resolve/reject,
    /// which carry the promise they settle).
    BoundNative {
        /// The built-in id.
        id: u16,
        /// The bound receiver (e.g. the promise to settle).
        target: Handle,
    },
    /// A `Promise` (its shared settlement state).
    Promise(Rc<RefCell<PromiseState>>),
    /// A `Date`: milliseconds since the Unix epoch (UTC).
    Date(f64),
    /// A `RegExp`: its source pattern and flags (compiled on demand by the
    /// `regex` engine, so the variant itself carries no feature-gated type).
    RegExp {
        /// The pattern source.
        source: alloc::boxed::Box<str>,
        /// The flags (e.g. `"gi"`).
        flags: alloc::boxed::Box<str>,
        /// The mutable `lastIndex` (where the next `g`/`y` search resumes).
        last_index: usize,
        /// A transient cache of the compiled program (RE-P1), populated on first
        /// match so a reused RegExp (`for(…) re.test(x)`, `str.replace(/…/g,…)`)
        /// is parsed+compiled once, not per call. It holds no heap handles (the
        /// compiled program is plain instructions), so `Trace`/`Relocate` ignore
        /// it; it is **not** serialized — a snapshot reconstructs the RegExp from
        /// `source`/`flags` with an empty cache, refilled lazily on first use.
        /// `lastIndex` is separate mutable state and is unaffected by the cache.
        #[cfg(feature = "regex")]
        compiled: core::cell::RefCell<Option<alloc::rc::Rc<crate::regex::Regex>>>,
    },
    /// A class constructor: an index into the interpreter's class table plus the
    /// scope it was defined in.
    Class {
        /// Index into the owning interpreter's class table (the AST body).
        class_id: u32,
        /// The captured lexical environment.
        env: Scope,
    },
    /// A `Map` (`is_set = false`) or `Set` (`is_set = true`): insertion-ordered
    /// key/value entries (for a `Set`, the value equals the key).
    Collection {
        /// Whether this is a `Set` (vs a `Map`).
        is_set: bool,
        /// Whether this is a `WeakMap`/`WeakSet` (keys must be objects/symbols).
        is_weak: bool,
        /// The entries, in insertion order, compared by `SameValueZero`-ish
        /// strict equality.
        entries: Vec<(NanBox, NanBox)>,
    },
    /// A `Symbol` primitive: an immutable description and a process-unique id.
    /// Symbols compare by identity (two `Symbol("x")` are distinct).
    Symbol {
        /// The optional description (`Symbol("desc").description`).
        description: alloc::boxed::Box<str>,
        /// A unique id, assigned at creation, giving the symbol its identity.
        id: u64,
    },
    /// A `BigInt` primitive — true arbitrary precision (a pure-Rust bignum).
    BigInt(crate::bignum::BigInt),
    /// A `Proxy`: wraps a `target` object, routing property operations through a
    /// `handler` object's traps (`get`/`set`/`has`/`deleteProperty`).
    Proxy {
        /// The proxied object.
        target: Handle,
        /// The trap handler.
        handler: Handle,
        /// Whether the proxy has been revoked (`Proxy.revocable`); a revoked
        /// proxy throws on every operation.
        revoked: bool,
    },
}

impl Cell {
    /// The object, if this cell is one.
    #[must_use]
    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Cell::Object(o) => Some(o),
            _ => None,
        }
    }

    /// The object mutably, if this cell is one.
    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            Cell::Object(o) => Some(o),
            _ => None,
        }
    }

    /// The string, if this cell is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&Rope> {
        match self {
            Cell::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The array elements, if this cell is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[NanBox]> {
        match self {
            Cell::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The array elements mutably, if this cell is one.
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<NanBox>> {
        match self {
            Cell::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The raw bytes, if this cell is a byte store.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Cell::Bytes(b) => Some(b.as_slice()),
            _ => None,
        }
    }

    /// The byte store mutably, if this cell is one (for in-place writes/resize).
    pub fn as_byte_store_mut(&mut self) -> Option<&mut ByteStore> {
        match self {
            Cell::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// The typed-array view `(buffer, byte_offset, length, kind)`, if this is one.
    /// `buffer` is the backing *bytes* cell (for element access); use
    /// [`Cell::typed_array_object`] for the `[[ViewedArrayBuffer]]` object.
    #[must_use]
    pub fn as_typed_array(&self) -> Option<(Handle, usize, usize, u8)> {
        match self {
            Cell::TypedArray {
                buffer,
                byte_offset,
                length,
                kind,
                ..
            } => Some((*buffer, *byte_offset, *length, *kind)),
            _ => None,
        }
    }

    /// The `[[ViewedArrayBuffer]]` object handle of this view, if it is one.
    #[must_use]
    pub fn typed_array_object(&self) -> Option<Handle> {
        match self {
            Cell::TypedArray { array_buffer, .. } => Some(*array_buffer),
            _ => None,
        }
    }

    /// The function's `(func_id, captured env)`, if this cell is a function.
    #[must_use]
    pub fn as_function(&self) -> Option<(u32, &Scope)> {
        match self {
            Cell::Function { func_id, env } => Some((*func_id, env)),
            _ => None,
        }
    }

    /// The built-in id, if this cell is a native function.
    #[must_use]
    pub fn as_native(&self) -> Option<u16> {
        match self {
            Cell::Native(id) => Some(*id),
            _ => None,
        }
    }

    /// The `(id, target)` if this cell is a bound native function.
    #[must_use]
    pub fn as_bound_native(&self) -> Option<(u16, Handle)> {
        match self {
            Cell::BoundNative { id, target } => Some((*id, *target)),
            _ => None,
        }
    }

    /// The shared promise state, if this cell is a promise.
    #[must_use]
    pub fn as_promise(&self) -> Option<&Rc<RefCell<PromiseState>>> {
        match self {
            Cell::Promise(p) => Some(p),
            _ => None,
        }
    }

    /// The timestamp (ms since epoch), if this cell is a `Date`.
    #[must_use]
    pub fn as_date(&self) -> Option<f64> {
        match self {
            Cell::Date(ms) => Some(*ms),
            _ => None,
        }
    }

    /// The `(source, flags)` if this cell is a `RegExp`.
    #[must_use]
    pub fn as_regexp(&self) -> Option<(&str, &str)> {
        match self {
            Cell::RegExp { source, flags, .. } => Some((source, flags)),
            _ => None,
        }
    }

    /// The `(class_id, captured env)`, if this cell is a class.
    #[must_use]
    pub fn as_class(&self) -> Option<(u32, &Scope)> {
        match self {
            Cell::Class { class_id, env } => Some((*class_id, env)),
            _ => None,
        }
    }

    /// The `(is_set, entries)` of a collection, mutably.
    pub fn as_collection_mut(&mut self) -> Option<(bool, &mut Vec<(NanBox, NanBox)>)> {
        match self {
            Cell::Collection {
                is_set, entries, ..
            } => Some((*is_set, entries)),
            _ => None,
        }
    }

    /// The `(is_set, entries)` of a collection.
    #[must_use]
    pub fn as_collection(&self) -> Option<(bool, &[(NanBox, NanBox)])> {
        match self {
            Cell::Collection {
                is_set, entries, ..
            } => Some((*is_set, entries)),
            _ => None,
        }
    }

    /// The `typeof` string for this reference value (`"string"` for strings,
    /// `"function"` for functions, `"object"` for objects and arrays — JS has no
    /// array primitive type).
    #[must_use]
    pub fn type_of(&self) -> &'static str {
        match self {
            Cell::Str(_) => "string",
            Cell::Function { .. }
            | Cell::Native(_)
            | Cell::BoundNative { .. }
            | Cell::Class { .. } => "function",
            Cell::Symbol { .. } => "symbol",
            Cell::BigInt(_) => "bigint",
            Cell::Object(_)
            | Cell::Array(_)
            | Cell::TypedArray { .. }
            | Cell::Collection { .. }
            | Cell::Promise(_)
            | Cell::Date(_)
            | Cell::RegExp { .. }
            | Cell::Proxy { .. } => "object",
            // An internal byte backing store is never a user-visible value; it is
            // reached only through its owning ArrayBuffer/view.
            Cell::Bytes(_) => "object",
        }
    }

    /// The `(description, id)` of this symbol, if it is one.
    #[must_use]
    pub fn as_symbol(&self) -> Option<(&str, u64)> {
        match self {
            Cell::Symbol { description, id } => Some((description, *id)),
            _ => None,
        }
    }

    /// The value if this cell is a `BigInt`.
    #[must_use]
    pub fn as_bigint(&self) -> Option<&crate::bignum::BigInt> {
        match self {
            Cell::BigInt(n) => Some(n),
            _ => None,
        }
    }

    /// The `(target, handler)` if this cell is a `Proxy`.
    #[must_use]
    pub fn as_proxy(&self) -> Option<(Handle, Handle)> {
        match self {
            Cell::Proxy {
                target, handler, ..
            } => Some((*target, *handler)),
            _ => None,
        }
    }

    /// Whether this proxy has been revoked, if it is a proxy.
    #[must_use]
    pub fn proxy_revoked(&self) -> Option<bool> {
        match self {
            Cell::Proxy { revoked, .. } => Some(*revoked),
            _ => None,
        }
    }

    /// Marks this proxy revoked (no-op if not a proxy).
    pub fn revoke_proxy(&mut self) {
        if let Cell::Proxy { revoked, .. } = self {
            *revoked = true;
        }
    }
}

impl Trace for Cell {
    fn trace(&self, visit: &mut dyn FnMut(Handle)) {
        match self {
            Cell::Object(o) => o.trace_handles(visit),
            Cell::Array(elems) => {
                for e in elems {
                    if let Some(raw) = e.as_handle() {
                        visit(Handle::from_raw(raw));
                    }
                }
            }
            // A closure (or class) keeps its captured environment alive.
            Cell::Function { env, .. } | Cell::Class { env, .. } => env.for_each_handle(visit),
            // A collection's keys and values are reachable.
            Cell::Collection { entries, .. } => {
                for (k, v) in entries {
                    if let Some(raw) = k.as_handle() {
                        visit(Handle::from_raw(raw));
                    }
                    if let Some(raw) = v.as_handle() {
                        visit(Handle::from_raw(raw));
                    }
                }
            }
            // A promise keeps its value and reaction handlers reachable.
            Cell::Promise(p) => {
                let s = p.borrow();
                if let Some(raw) = s.value.as_handle() {
                    visit(Handle::from_raw(raw));
                }
                for r in &s.reactions {
                    for h in [r.on_fulfilled, r.on_rejected] {
                        if let Some(raw) = h.as_handle() {
                            visit(Handle::from_raw(raw));
                        }
                    }
                    visit(r.result);
                }
            }
            // A typed-array view keeps its backing bytes and its viewed
            // ArrayBuffer object reachable.
            Cell::TypedArray {
                buffer,
                array_buffer,
                ..
            } => {
                visit(*buffer);
                visit(*array_buffer);
            }
            // A bound native keeps its target reachable.
            Cell::BoundNative { target, .. } => visit(*target),
            // A proxy keeps its target and handler reachable.
            Cell::Proxy {
                target, handler, ..
            } => {
                visit(*target);
                visit(*handler);
            }
            // Strings, native functions, dates, regexes, and raw byte stores
            // reference no handles.
            Cell::Str(_)
            | Cell::Native(_)
            | Cell::Date(_)
            | Cell::RegExp { .. }
            | Cell::Symbol { .. }
            | Cell::BigInt(_)
            | Cell::Bytes(_) => {}
        }
    }
}

impl crate::gc::Relocate for Cell {
    fn relocate(&mut self, forward: &dyn Fn(Handle) -> Handle) {
        // Rewrites a handle-bearing value through `forward` (no-op otherwise).
        let fwd = |v: &mut NanBox| {
            if let Some(raw) = v.as_handle() {
                *v = NanBox::handle(forward(Handle::from_raw(raw)).to_raw());
            }
        };
        match self {
            Cell::Object(o) => o.relocate_handles(forward),
            Cell::Array(elems) => elems.iter_mut().for_each(fwd),
            Cell::Function { env, .. } | Cell::Class { env, .. } => env.relocate_handles(forward),
            Cell::Collection { entries, .. } => {
                for (k, v) in entries {
                    fwd(k);
                    fwd(v);
                }
            }
            Cell::Promise(p) => {
                let mut s = p.borrow_mut();
                fwd(&mut s.value);
                for r in &mut s.reactions {
                    fwd(&mut r.on_fulfilled);
                    fwd(&mut r.on_rejected);
                    r.result = forward(r.result);
                }
            }
            Cell::TypedArray {
                buffer,
                array_buffer,
                ..
            } => {
                *buffer = forward(*buffer);
                *array_buffer = forward(*array_buffer);
            }
            Cell::BoundNative { target, .. } => *target = forward(*target),
            Cell::Proxy {
                target, handler, ..
            } => {
                *target = forward(*target);
                *handler = forward(*handler);
            }
            Cell::Str(_)
            | Cell::Native(_)
            | Cell::Date(_)
            | Cell::RegExp { .. }
            | Cell::Symbol { .. }
            | Cell::BigInt(_)
            | Cell::Bytes(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap::Heap;
    use crate::shape::Shape;
    use alloc::rc::Rc;

    #[test]
    fn accessors_select_the_right_kind() {
        let obj = Cell::Object(Object::new(Shape::root()));
        assert!(obj.as_object().is_some());
        assert!(obj.as_str().is_none());
        assert!(obj.as_array().is_none());
        assert_eq!(obj.type_of(), "object");

        let s = Cell::Str(Rope::from("hi"));
        assert_eq!(s.as_str().map(Rope::materialize).as_deref(), Some("hi"));
        assert_eq!(s.type_of(), "string");

        let a = Cell::Array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        assert_eq!(a.as_array().map(<[_]>::len), Some(2));
        assert_eq!(a.type_of(), "object");
    }

    #[test]
    fn byte_store_owned_and_external() {
        // Owned: read + mutate in place.
        let mut s = ByteStore::Owned(alloc::vec![1u8, 2, 3]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.as_slice(), &[1, 2, 3]);
        s.as_mut_slice()[1] = 9;
        assert_eq!(s.as_slice(), &[1, 9, 3]);

        // External: zero-copy view over a caller-owned region; writes go through.
        let mut region = alloc::boxed::Box::new([10u8, 20, 30]);
        let ptr = region.as_mut_ptr();
        // SAFETY: `region` outlives `ext`; no free hook (caller owns it).
        #[allow(unsafe_code)]
        let mut ext = unsafe { ByteStore::external(ptr, 3, None) };
        assert_eq!(ext.as_slice(), &[10, 20, 30]);
        ext.as_mut_slice()[0] = 99;
        assert_eq!(region[0], 99); // wrote through to the external region
    }

    #[test]
    fn external_free_hook_runs_once_on_drop() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        static FREED: AtomicUsize = AtomicUsize::new(0);
        #[allow(unsafe_code)]
        unsafe fn record_free(_p: *mut u8, _l: usize) {
            FREED.fetch_add(1, Ordering::SeqCst);
        }
        let mut region = alloc::boxed::Box::new([0u8; 4]);
        let ptr = region.as_mut_ptr();
        // SAFETY: region kept alive past the store; the hook only records.
        #[allow(unsafe_code)]
        let store = unsafe { ByteStore::external(ptr, 4, Some(record_free)) };
        let cell = Cell::Bytes(store);
        assert_eq!(FREED.load(Ordering::SeqCst), 0);
        drop(cell); // sweeping the cell runs the free hook exactly once
        assert_eq!(FREED.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn gc_traces_arrays_and_objects_uniformly() {
        // root array -> object -> (handle to) a leaf; plus an unreachable string.
        let mut heap: Heap<Cell> = Heap::new();
        let root_shape = Shape::root();

        let leaf = heap.alloc(Cell::Object(Object::new(Rc::clone(&root_shape))));
        let mut mid = Object::new(Rc::clone(&root_shape));
        mid.set("leaf", NanBox::handle(leaf.to_raw()));
        let mid_h = heap.alloc(Cell::Object(mid));
        let arr = heap.alloc(Cell::Array(alloc::vec![NanBox::handle(mid_h.to_raw())]));
        let _garbage = heap.alloc(Cell::Str(Rope::from("unreferenced")));
        assert_eq!(heap.len(), 4);

        let stats = crate::gc::collect(&mut heap, &[arr]);
        assert_eq!(stats.marked, 3); // arr -> mid -> leaf
        assert_eq!(stats.swept, 1); // the string
        assert!(heap.is_live(arr) && heap.is_live(mid_h) && heap.is_live(leaf));
    }
}
