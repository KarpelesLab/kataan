//! The object-model context that ties the foundation together (`ROADMAP.md`
//! §3).
//!
//! [`nanbox`](crate::nanbox), [`shape`](crate::shape), [`object`](crate::object),
//! [`heap`](crate::heap), [`atom`](crate::atom), and [`gc`](crate::gc) are
//! independent pieces; a running engine needs them bundled behind one handle. A
//! `Realm` owns:
//! - the managed [`Heap`] of objects,
//! - the shared **root [`Shape`]** every object's layout descends from (so
//!   identically-structured objects share hidden classes across the realm), and
//! - the realm's [`AtomTable`] for interned property keys.
//!
//! It exposes the operations a VM performs on the heap — allocate an object, get
//! and set properties (as [`NanBox`] values addressed by [`Handle`]), and run a
//! collection from a root set — so the eventual VM migration targets this one
//! API rather than wiring the pieces together itself.
//!
//! Pure, safe `alloc`-only Rust.
//!
//! [`Heap`]: crate::heap::Heap
//! [`Shape`]: crate::shape::Shape
//! [`AtomTable`]: crate::atom::AtomTable
//! [`NanBox`]: crate::nanbox::NanBox
//! [`Handle`]: crate::heap::Handle

use crate::atom::{Atom, AtomTable};
use crate::cell::Cell;
use crate::gc::{self, Stats};
use crate::heap::{Handle, Heap};
use crate::nanbox::NanBox;
use crate::nbexec::{
    coerce_typed, decode_bigint_element, decode_typed_element, encode_bigint_element,
    encode_typed_element, is_bigint_kind,
};
use crate::object::Object;
use crate::rope::Rope;
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// Hidden `[[PromiseState]]` slot for a **Promise subclass** instance
/// (`class P extends Promise`): its value is the handle of the backing
/// `Cell::Promise` that carries the real settlement state, which
/// [`Realm::promise_state`] follows so every promise operation works on the
/// subclass instance. `\u{0}`-prefixed like every other internal slot key.
pub(crate) const PROMISE_STATE_SLOT: &str = "\u{0}PromiseState";

/// Bytes-per-element for each typed-array `kind` (index into the engine's
/// `TYPED_ARRAY_KINDS` table: Int8, Uint8, Uint8Clamped, Int16, Uint16, Int32,
/// Uint32, Float32, Float64, BigInt64, BigUint64). A `kind` outside the table
/// reads as size 1.
// Indexed by kind (see `TYPED_ARRAY_KINDS`); kind 11 = `Float16Array` (2 bytes).
const TYPED_ELEM_SIZE: [usize; 12] = [1, 1, 1, 2, 2, 4, 4, 4, 8, 8, 8, 2];

/// The byte size of one element of typed-array `kind`.
#[must_use]
pub fn typed_elem_size(kind: u8) -> usize {
    TYPED_ELEM_SIZE.get(kind as usize).copied().unwrap_or(1)
}

/// An object-model context: the heap, the shared root shape, and the atom table.
pub struct Realm {
    heap: Heap<Cell>,
    root_shape: Rc<Shape>,
    atoms: AtomTable,
    /// The in-flight incremental collection's marker, present only while an
    /// incremental marking cycle is running. The write barrier shades stored
    /// references into it so concurrent mutation stays sound.
    incremental: Option<gc::IncrementalMarker>,
    /// Monotonic counter giving each `Symbol` a unique identity.
    next_symbol_id: u64,
    /// Maps a symbol's id back to its heap handle, so a symbol used as a property
    /// key (stored as `\0sym:{id}`) can be recovered (e.g. by
    /// `Object.getOwnPropertySymbols`).
    symbols_by_id: alloc::collections::BTreeMap<u64, Handle>,
    /// Lazily-created `.prototype` objects for constructor functions, keyed by
    /// the closure's function id. (Keyed by id, not handle, so it survives a
    /// moving collection; distinct closures sharing an id share a prototype — a
    /// bounded approximation, see `[[latent-engine-conformance-bugs]]`.)
    fn_protos: alloc::collections::BTreeMap<u32, Handle>,
    /// The function handle to use as a prototype's `constructor` back-reference,
    /// keyed by function id (the most recent closure created for that id). Set when
    /// a function is allocated; read when its `.prototype` is first materialized so
    /// `Foo.prototype.constructor === Foo` (and thus `instance.constructor`). Same
    /// id-keyed, GC-quiescent caveat as `fn_protos`.
    fn_ctor: alloc::collections::BTreeMap<u32, Handle>,
    /// Lazily-materialized `.prototype` objects for class declarations/expressions,
    /// keyed by the class-table id. Populated on first access with the class's
    /// instance methods/accessors and a non-enumerable `constructor` back-link.
    class_protos: alloc::collections::BTreeMap<u32, Handle>,
    /// Auxiliary named-property objects for non-object cells (arrays, functions),
    /// which have no inline object part. Keyed by the cell's handle. Not a GC root
    /// and not relocated on a moving collection — sound only because collection is
    /// never driven mid-execution (see `[[latent-engine-conformance-bugs]]`).
    aux_props: alloc::collections::BTreeMap<u64, Handle>,
    /// Handles of frozen arrays (`Object.freeze([...])`). Arrays have no inline
    /// object part to carry the flag; same handle-keyed, non-GC-root caveat as
    /// `aux_props`.
    frozen_arrays: alloc::collections::BTreeSet<u64>,
    /// Handles of sealed arrays (`Object.seal`/`freeze`) — same caveat.
    sealed_arrays: alloc::collections::BTreeSet<u64>,
    /// Handles of non-extensible arrays (`Object.preventExtensions`/`seal`/`freeze`)
    /// — element writes past the end are rejected. Same caveat.
    non_extensible_arrays: alloc::collections::BTreeSet<u64>,
    /// Handles of typed-array *views* that auto-track their backing (resizable)
    /// `ArrayBuffer`'s length — i.e. created with no explicit `length` argument
    /// over a resizable buffer. On `resize`, only these views recompute their
    /// element count to span the buffer; a fixed-length view keeps its `length`
    /// and instead becomes *out of bounds* when the buffer can no longer hold it.
    /// Non-GC-root: a stale entry for a dead handle is harmless.
    length_tracking_views: alloc::collections::BTreeSet<u64>,
    /// Handles of arrays whose `length` property was made non-writable
    /// (`Object.defineProperty(arr, "length", {writable:false})`). An array's
    /// `length` is writable by default; this records the explicit demotion so the
    /// descriptor reports it and a later `length` redefine is validated. Same
    /// non-GC-root caveat as `frozen_arrays`.
    nonwritable_array_lengths: alloc::collections::BTreeSet<u64>,
    /// Logical `length` of an array set above its dense backing capacity (a valid
    /// uint32 length up to 2^32-1 that exceeds [`Limits::max_array_len`]). The dense
    /// `Vec` storage is not actually grown that far (a multi-gigabyte allocation);
    /// instead the spec-visible `length` is recorded here so `arr.length` and a
    /// `length` descriptor report it. Cleared once the dense storage catches up or
    /// `length` is set back within the cap. Same non-GC-root caveat as `aux_props`.
    sparse_array_lengths: alloc::collections::BTreeMap<u64, usize>,
    /// The default prototype (`Object.prototype`) installed on objects created by
    /// [`new_object`](Realm::new_object) once the global environment is set up. A
    /// `None`-proto object (`Object.create(null)`) opts out explicitly.
    default_object_proto: Option<Handle>,
    /// Explicit `[[Prototype]]` links for callable cells that have no inline
    /// object part (natives, bound natives) — e.g. each typed-array constructor's
    /// prototype is the shared `%TypedArray%` intrinsic. Keyed by handle; same
    /// non-GC-root, GC-quiescent caveat as `aux_props`.
    native_protos: alloc::collections::BTreeMap<u64, Handle>,
    /// Callable cells whose `[[Prototype]]` was explicitly set to `null`
    /// (`Object.setPrototypeOf(fn, null)`), distinguishing them from the default
    /// (which resolves to `%Function.prototype%`). Same non-GC-root caveat.
    callable_null_protos: alloc::collections::BTreeSet<u64>,
    /// Class tags for *non-object* cells — a derived instance of a native base
    /// (`class S extends Map {}` → a real `Map` cell) records its class here, since
    /// the inline class tag lives only on plain-object cells. Read by
    /// [`class_tag`](Realm::class_tag) so `instanceof Subclass` (the class-chain
    /// walk) recognizes the native-cell instance. Same non-GC-root caveat.
    native_class_tags: alloc::collections::BTreeMap<u64, u32>,
    /// The shared abstract `%TypedArray%` intrinsic constructor (the value
    /// `Object.getPrototypeOf(Int8Array)` returns), installed at global setup.
    typed_array_intrinsic: Option<Handle>,
    /// The realm's `Function.prototype` intrinsic (the default `[[Prototype]]` of
    /// every ordinary/native callable). Installed at global setup; until then a
    /// callable's prototype resolves to `None`. A callable's `[[Prototype]]` can
    /// still be overridden explicitly (`Object.setPrototypeOf(fn, p)`), recorded
    /// in [`native_protos`](Realm::native_protos).
    function_proto_intrinsic: Option<Handle>,
    /// The realm's `Array.prototype` intrinsic (the default `[[Prototype]]` of a
    /// dense `Cell::Array`, which has no inline object part). So
    /// `Object.getPrototypeOf([])`, `[] instanceof Array`, and `"push" in []`
    /// resolve through the chain. An explicit `Object.setPrototypeOf(arr, p)`
    /// override is recorded in [`native_protos`](Realm::native_protos) /
    /// [`callable_null_protos`](Realm::callable_null_protos).
    array_proto_intrinsic: Option<Handle>,
    /// The realm's `Symbol.prototype` intrinsic (the `[[Prototype]]` of a Symbol
    /// primitive `Cell::Symbol`, which has no inline object part). So
    /// `Object.getPrototypeOf(Symbol())` resolves to `Symbol.prototype`. Symbol
    /// primitives are immutable, so this is never overridden per-instance.
    symbol_proto_intrinsic: Option<Handle>,
    /// The realm's `%BigInt.prototype%` intrinsic — the `[[Prototype]]` of
    /// every `Cell::BigInt` primitive (BigInt primitives are immutable).
    bigint_proto_intrinsic: Option<Handle>,
    /// The lazily-materialized `.prototype` objects of the `Intl` service
    /// constructors (`Intl.NumberFormat`, `Intl.Collator`, …, `Intl.Locale`,
    /// `Intl.DurationFormat`), keyed by the constructor's native dispatch id. Each
    /// holds the service's branded methods/accessors and is the `[[Prototype]]` of
    /// every instance that service creates. Same id-keyed, GC-quiescent caveat as
    /// `fn_protos`; also kept alive as GC roots in `gc_side_table_values` (and
    /// reachable through the constructor's `prototype` data property).
    intl_protos: alloc::collections::BTreeMap<u16, Handle>,
    /// Host-held **persistent handles** (`ROADMAP.md` §4.0 handle scope): values an
    /// embedder pins across engine calls. Each slot is a GC root (so the value
    /// survives collection) and is forwarded on compaction (so the handle stays
    /// valid when the moving collector relocates it). The host holds only the slot
    /// index — never a raw `Handle` that compaction would invalidate. `None` slots
    /// are freed indices, reused by the next `persist`.
    host_persistent: Vec<Option<NanBox>>,
    /// The RegExp legacy static match record (Annex B.2.5): updated after every
    /// successful built-in match, read by the `RegExp.$1`..`$9` / `RegExp.input`
    /// / `lastMatch` / … static accessors.
    legacy_regexp: LegacyRegExpState,
    /// Tunable resource limits for work driven in this realm. Defaults to
    /// [`crate::limits::Limits::default`]; override with [`Realm::with_limits`].
    pub limits: crate::limits::Limits,
}

/// The mutable state behind the Annex B.2.5 RegExp legacy static accessors
/// (`RegExp.input`/`$_`, `RegExp.lastMatch`/`$&`, `RegExp.lastParen`/`$+`,
/// `RegExp.leftContext`/`` $` ``, `RegExp.rightContext`/`$'`, `RegExp.$1`..`$9`).
/// Each field stores WTF-8 bytes so a surrogate-bearing subject round-trips.
#[derive(Default, Clone)]
pub struct LegacyRegExpState {
    /// The last subject string matched against (`RegExp.input` / `$_`).
    pub input: alloc::vec::Vec<u8>,
    /// The portion of the subject that matched (`RegExp.lastMatch` / `$&`).
    pub last_match: alloc::vec::Vec<u8>,
    /// The last (highest-index) captured group (`RegExp.lastParen` / `$+`).
    pub last_paren: alloc::vec::Vec<u8>,
    /// The substring preceding the match (`RegExp.leftContext` / `` $` ``).
    pub left_context: alloc::vec::Vec<u8>,
    /// The substring following the match (`RegExp.rightContext` / `$'`).
    pub right_context: alloc::vec::Vec<u8>,
    /// Captured groups 1..=9 (`RegExp.$1`..`$9`); absent groups are empty.
    pub parens: [alloc::vec::Vec<u8>; 9],
}

impl Default for Realm {
    fn default() -> Self {
        Self::new()
    }
}

impl Realm {
    /// Creates an empty realm with default [`Limits`](crate::limits::Limits).
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(crate::limits::Limits::default())
    }

    /// Creates an empty realm with the given resource [`Limits`](crate::limits::Limits).
    #[must_use]
    pub fn with_limits(limits: crate::limits::Limits) -> Self {
        Self {
            heap: Heap::new(),
            root_shape: Shape::root(),
            atoms: AtomTable::new(),
            incremental: None,
            next_symbol_id: 1,
            symbols_by_id: alloc::collections::BTreeMap::new(),
            fn_protos: alloc::collections::BTreeMap::new(),
            fn_ctor: alloc::collections::BTreeMap::new(),
            class_protos: alloc::collections::BTreeMap::new(),
            aux_props: alloc::collections::BTreeMap::new(),
            length_tracking_views: alloc::collections::BTreeSet::new(),
            frozen_arrays: alloc::collections::BTreeSet::new(),
            sealed_arrays: alloc::collections::BTreeSet::new(),
            non_extensible_arrays: alloc::collections::BTreeSet::new(),
            nonwritable_array_lengths: alloc::collections::BTreeSet::new(),
            sparse_array_lengths: alloc::collections::BTreeMap::new(),
            default_object_proto: None,
            native_protos: alloc::collections::BTreeMap::new(),
            callable_null_protos: alloc::collections::BTreeSet::new(),
            native_class_tags: alloc::collections::BTreeMap::new(),
            typed_array_intrinsic: None,
            function_proto_intrinsic: None,
            array_proto_intrinsic: None,
            symbol_proto_intrinsic: None,
            bigint_proto_intrinsic: None,
            intl_protos: alloc::collections::BTreeMap::new(),
            host_persistent: Vec::new(),
            legacy_regexp: LegacyRegExpState::default(),
            limits,
        }
    }

    /// A shared reference to the RegExp legacy static match record (Annex B.2.5).
    #[must_use]
    pub fn legacy_regexp(&self) -> &LegacyRegExpState {
        &self.legacy_regexp
    }

    /// Replaces the RegExp legacy static match record after a successful match.
    pub fn set_legacy_regexp(&mut self, state: LegacyRegExpState) {
        self.legacy_regexp = state;
    }

    /// Records an explicit `[[Prototype]]` for a callable cell (native / bound
    /// native) that has no inline object part. Read back by
    /// [`object_proto`](Realm::object_proto) so `Object.getPrototypeOf` resolves
    /// the constructor-side chain (e.g. `getPrototypeOf(Int8Array)` →
    /// `%TypedArray%`).
    pub fn set_native_proto(&mut self, handle: Handle, proto: Handle) {
        self.native_protos.insert(handle.to_raw(), proto);
    }

    /// Records the shared abstract `%TypedArray%` intrinsic constructor handle.
    pub fn set_typed_array_intrinsic(&mut self, handle: Handle) {
        self.typed_array_intrinsic = Some(handle);
    }

    /// Records the realm's `%Function.prototype%` intrinsic — the default
    /// `[[Prototype]]` for every ordinary/native callable.
    pub fn set_function_proto_intrinsic(&mut self, handle: Handle) {
        self.function_proto_intrinsic = Some(handle);
    }

    /// Records the realm's `%Array.prototype%` intrinsic — the default
    /// `[[Prototype]]` for every dense `Cell::Array`.
    pub fn set_array_proto_intrinsic(&mut self, handle: Handle) {
        self.array_proto_intrinsic = Some(handle);
    }

    /// The realm's `%Array.prototype%` intrinsic, if installed — the default
    /// `[[Prototype]]` of a dense array (used to fast-path element writes: an
    /// array with the default prototype has no inherited index setters/proxy).
    #[must_use]
    pub fn array_proto_intrinsic(&self) -> Option<Handle> {
        self.array_proto_intrinsic
    }

    /// Records the realm's `%Symbol.prototype%` intrinsic — the `[[Prototype]]`
    /// for every `Cell::Symbol` primitive.
    pub fn set_symbol_proto_intrinsic(&mut self, handle: Handle) {
        self.symbol_proto_intrinsic = Some(handle);
    }

    /// Records the realm's `%BigInt.prototype%` intrinsic — the `[[Prototype]]`
    /// for every `Cell::BigInt` primitive.
    pub fn set_bigint_proto_intrinsic(&mut self, handle: Handle) {
        self.bigint_proto_intrinsic = Some(handle);
    }

    /// The shared abstract `%TypedArray%` intrinsic constructor, if installed.
    #[must_use]
    pub fn typed_array_intrinsic(&self) -> Option<Handle> {
        self.typed_array_intrinsic
    }

    /// Every live typed-array view whose backing bytes cell is `bytes_handle`.
    /// With the byte-backed view model, views alias their buffer intrinsically, so
    /// the set is recovered by scanning the heap rather than a registry.
    fn views_over(&self, bytes_handle: Handle) -> alloc::vec::Vec<Handle> {
        self.heap
            .live_handles()
            .into_iter()
            .filter(|h| {
                matches!(
                    self.heap.get(*h),
                    Some(Cell::TypedArray { buffer, .. }) if *buffer == bytes_handle
                )
            })
            .collect()
    }

    /// Sets the intrinsic `length` of the typed-array view at `handle` (used when a
    /// resizable buffer grows/shrinks, or on detach). No-op for a non-view.
    fn set_typed_length(&mut self, handle: Handle, new_len: usize) {
        if let Some(Cell::TypedArray { length, .. }) = self.heap.get_mut(handle) {
            *length = new_len;
        }
    }

    /// Empties every typed-array view over the buffer at `bytes_handle` (length 0)
    /// — used when the buffer is detached by `ArrayBuffer.prototype.transfer`.
    /// Returns the view handles that were emptied.
    pub fn detach_buffer_views(&mut self, bytes_handle: Handle) -> alloc::vec::Vec<Handle> {
        let views = self.views_over(bytes_handle);
        for v in &views {
            self.set_typed_length(*v, 0);
        }
        views
    }

    /// Resizes the owned byte store at `bytes_handle` to `new_byte_len` (zero-filling
    /// on growth) and re-lengths every typed-array view over it to span the resized
    /// buffer. Element reads/writes already go through the shared bytes, so only each
    /// view's intrinsic `length` needs updating. Used by `ArrayBuffer.prototype.resize`.
    pub fn resize_buffer(&mut self, bytes_handle: Handle, new_byte_len: usize) {
        self.bytes_resize(bytes_handle, new_byte_len);
        for v in self.views_over(bytes_handle) {
            let Some((_, off, _, kind)) = self.heap.get(v).and_then(Cell::as_typed_array) else {
                continue;
            };
            // Only an auto-length-tracking view re-spans the resized buffer. A
            // fixed-length view keeps its intrinsic `length`; whether it currently
            // fits is determined dynamically (see `typed_array_out_of_bounds`), so
            // it becomes out of bounds on shrink and valid again on regrow without
            // losing its declared length.
            if self.length_tracking_views.contains(&v.to_raw()) {
                let size = typed_elem_size(kind);
                let view_len = new_byte_len.saturating_sub(off) / size;
                self.set_typed_length(v, view_len);
            }
        }
    }

    /// Marks the typed-array view at `handle` as auto-length-tracking (created
    /// with no explicit length over a resizable buffer).
    pub fn mark_length_tracking(&mut self, handle: Handle) {
        self.length_tracking_views.insert(handle.to_raw());
    }

    /// Whether the typed-array view at `handle` auto-tracks its buffer's length.
    #[must_use]
    pub fn is_length_tracking(&self, handle: Handle) -> bool {
        self.length_tracking_views.contains(&handle.to_raw())
    }

    /// Whether the fixed-length typed-array view at `handle` is currently *out of
    /// bounds*: its `[[ByteOffset]] + length·elem_size` exceeds the backing
    /// buffer's current byte length (e.g. the resizable buffer was shrunk below
    /// the view's declared extent), or its offset itself is past the end. A
    /// length-tracking view is never out of bounds (it re-spans on resize); a
    /// detached buffer is reported separately. Returns `false` for a non-view.
    #[must_use]
    pub fn typed_array_out_of_bounds(&self, handle: Handle) -> bool {
        let Some((bytes, off, len, kind)) = self.heap.get(handle).and_then(Cell::as_typed_array)
        else {
            return false;
        };
        if self.length_tracking_views.contains(&handle.to_raw()) {
            return false;
        }
        let cur = self.bytes_len(bytes).unwrap_or(0);
        let size = typed_elem_size(kind);
        off.checked_add(len.saturating_mul(size))
            .is_none_or(|end| end > cur)
    }

    /// Records the realm's `Object.prototype`, applied to subsequently-created
    /// plain objects (so they inherit `toString`, `hasOwnProperty`, …).
    pub fn set_default_object_proto(&mut self, proto: Handle) {
        self.default_object_proto = Some(proto);
    }

    /// The realm's `%Object.prototype%`, if wired.
    #[must_use]
    pub fn default_object_proto(&self) -> Option<Handle> {
        self.default_object_proto
    }

    /// The auxiliary property object for a non-object cell, created on first use.
    fn aux_object(&mut self, handle: Handle) -> Handle {
        if let Some(h) = self.aux_props.get(&handle.to_raw()) {
            return *h;
        }
        let obj = self.new_object();
        self.aux_props.insert(handle.to_raw(), obj);
        obj
    }

    /// The `.prototype` object for the constructor function with id `func_id`,
    /// creating a fresh empty object on first access.
    pub fn function_prototype(&mut self, func_id: u32) -> Handle {
        if let Some(h) = self.fn_protos.get(&func_id) {
            return *h;
        }
        let proto = self.new_object();
        self.fn_protos.insert(func_id, proto);
        // Back-link `proto.constructor` to the function (non-enumerable), so
        // `instance.constructor === Foo` and `instance.constructor.name`.
        if let Some(ctor) = self.fn_ctor.get(&func_id).copied() {
            self.set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        }
        proto
    }

    /// Reassigns a constructor function's `.prototype` (`Fn.prototype = obj`).
    pub fn set_function_prototype(&mut self, func_id: u32, proto: Handle) {
        self.fn_protos.insert(func_id, proto);
    }

    /// Allocates a fresh empty object in the heap and returns its handle.
    pub fn new_object(&mut self) -> Handle {
        let obj = Object::new(Rc::clone(&self.root_shape));
        let h = self.heap.alloc(Cell::Object(obj));
        if let Some(proto) = self.default_object_proto {
            self.set_object_proto(h, Some(proto));
        }
        h
    }

    /// Allocates a string value in the heap and returns its handle.
    pub fn new_string(&mut self, s: &str) -> Handle {
        self.heap.alloc(Cell::Str(Rope::from(s)))
    }

    /// Allocates a string value from raw **WTF-8 bytes**, preserving any lone
    /// UTF-16 surrogates (DOMString semantics — see [`crate::wtf8`]). The common
    /// case (no surrogates) is byte-identical to [`Realm::new_string`].
    pub fn new_string_wtf8(&mut self, bytes: alloc::vec::Vec<u8>) -> Handle {
        self.heap.alloc(Cell::Str(Rope::from_wtf8(bytes)))
    }

    /// The string at `handle` as raw **WTF-8 bytes** (lossless — lone surrogates
    /// preserved), or `None` if it is not a string. Use this for surrogate-aware
    /// string operations; [`Realm::string_value`] is the lossy `String` form.
    #[must_use]
    pub fn string_bytes(&self, handle: Handle) -> Option<alloc::vec::Vec<u8>> {
        Some(self.heap.get(handle)?.as_str()?.materialize_bytes())
    }

    /// Borrows the WTF-8 bytes of the string at `handle` *without allocating* when
    /// it is an unconcatenated rope leaf (the common case), or `None` when the cell
    /// is not a string or the rope is a `Concat` tree (use [`Realm::string_bytes`]
    /// then). Read-only hot paths (equality, ordering, emptiness) take this fast
    /// path to avoid the owned `Vec` and lossy decode that the `String`-typed
    /// accessors incur.
    #[must_use]
    pub fn string_leaf_bytes(&self, handle: Handle) -> Option<&[u8]> {
        self.heap.get(handle)?.as_str()?.as_leaf_bytes()
    }

    /// Allocates a contiguous byte store (the backing of an `ArrayBuffer`) from
    /// engine-owned bytes and returns its handle.
    pub fn new_bytes(&mut self, data: alloc::vec::Vec<u8>) -> Handle {
        self.heap
            .alloc(Cell::Bytes(crate::cell::ByteStore::Owned(data)))
    }

    /// Allocates a byte store that wraps an external, caller-owned region
    /// zero-copy (e.g. IPC shared memory). The engine reads/writes the region in
    /// place and runs `free` (if any) when the cell is collected.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for reads and writes of `len` bytes until
    /// `free` is invoked (or, if `free` is `None`, for the realm's lifetime). See
    /// [`crate::cell::ByteStore::external`].
    #[allow(unsafe_code)]
    pub unsafe fn wrap_external_bytes(
        &mut self,
        ptr: *mut u8,
        len: usize,
        free: Option<crate::cell::ExternFree>,
    ) -> Handle {
        // SAFETY: forwarded to the caller's contract on `wrap_external_bytes`.
        #[allow(unsafe_code)]
        let store = unsafe { crate::cell::ByteStore::external(ptr, len, free) };
        self.heap.alloc(Cell::Bytes(store))
    }

    /// A read view of the byte store at `handle`, if it is one.
    #[must_use]
    pub fn bytes_at(&self, handle: Handle) -> Option<&[u8]> {
        self.heap.get(handle).and_then(Cell::as_bytes)
    }

    /// A mutable view of the byte store at `handle`, if it is one.
    pub fn bytes_at_mut(&mut self, handle: Handle) -> Option<&mut [u8]> {
        self.heap
            .get_mut(handle)
            .and_then(Cell::as_byte_store_mut)
            .map(crate::cell::ByteStore::as_mut_slice)
    }

    /// Resizes an owned byte store to `new_len` (zero-filling on growth). No-op
    /// for an external store (returns `false`); returns `true` on success.
    pub fn bytes_resize(&mut self, handle: Handle, new_len: usize) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_byte_store_mut) {
            Some(crate::cell::ByteStore::Owned(v)) => {
                v.resize(new_len, 0);
                true
            }
            _ => false,
        }
    }

    /// The length of the byte store at `handle`, if it is one.
    #[must_use]
    pub fn bytes_len(&self, handle: Handle) -> Option<usize> {
        self.bytes_at(handle).map(<[u8]>::len)
    }

    /// Allocates a typed-array *view* — a [`Cell::TypedArray`] over the bytes at
    /// `buffer` starting at `byte_offset`, spanning `length` elements of `kind`.
    /// The view owns no element storage; reads/writes go through the shared bytes.
    /// `array_buffer` is the `[[ViewedArrayBuffer]]` object the view was created
    /// over — `.buffer` returns it directly (so it is stable and shared).
    pub fn new_typed_array(
        &mut self,
        buffer: Handle,
        array_buffer: Handle,
        byte_offset: usize,
        length: usize,
        kind: u8,
    ) -> Handle {
        self.heap.alloc(Cell::TypedArray {
            buffer,
            array_buffer,
            byte_offset,
            length,
            kind,
        })
    }

    /// The `[[ViewedArrayBuffer]]` object handle of the typed-array view at
    /// `handle`, if it is one. This is the `ArrayBuffer` object `.buffer` returns.
    #[must_use]
    pub fn typed_array_object(&self, handle: Handle) -> Option<Handle> {
        self.heap.get(handle)?.typed_array_object()
    }

    /// The element count of the typed-array view at `handle`, if it is one.
    #[must_use]
    pub fn typed_len(&self, handle: Handle) -> Option<usize> {
        let l = self
            .heap
            .get(handle)?
            .as_typed_array()
            .map(|(_, _, l, _)| l)?;
        // An out-of-bounds fixed-length view (resizable buffer shrank below its
        // extent) reports length 0 — its `.length`, iteration count, and index
        // bounds all collapse to empty until the buffer is grown back.
        if self.typed_array_out_of_bounds(handle) {
            return Some(0);
        }
        Some(l)
    }

    /// The element-kind index of the typed-array view at `handle`, if it is one.
    #[must_use]
    pub fn typed_kind(&self, handle: Handle) -> Option<u8> {
        self.heap
            .get(handle)?
            .as_typed_array()
            .map(|(_, _, _, k)| k)
    }

    /// The backing `ArrayBuffer`'s bytes handle of the typed-array view at
    /// `handle`, if it is one.
    #[must_use]
    pub fn typed_buffer(&self, handle: Handle) -> Option<Handle> {
        self.heap.get(handle)?.as_typed_array().map(|(b, ..)| b)
    }

    /// The byte offset of the typed-array view at `handle`, if it is one.
    #[must_use]
    pub fn typed_byte_offset(&self, handle: Handle) -> Option<usize> {
        self.heap
            .get(handle)?
            .as_typed_array()
            .map(|(_, o, _, _)| o)
    }

    /// Rebinds the backing-buffer handle of the typed-array view at `handle` (used
    /// by snapshot restore's second pass, after every cell's handle exists).
    /// No-op for a non-view.
    pub fn set_typed_buffer(&mut self, handle: Handle, new_buffer: Handle) {
        if let Some(Cell::TypedArray { buffer, .. }) = self.heap.get_mut(handle) {
            *buffer = new_buffer;
        }
    }

    /// Rebinds the `[[ViewedArrayBuffer]]` object handle of the view at `handle`
    /// (used by snapshot restore's second pass). No-op for a non-view.
    pub fn set_typed_array_object(&mut self, handle: Handle, new_obj: Handle) {
        if let Some(Cell::TypedArray { array_buffer, .. }) = self.heap.get_mut(handle) {
            *array_buffer = new_obj;
        }
    }

    /// `view[i]` — decodes element `i` from the shared bytes, or `undefined` for
    /// an out-of-range index. `None` if `handle` is not a typed-array view. A
    /// `BigInt64Array`/`BigUint64Array` element decodes to a freshly allocated
    /// `BigInt` (hence `&mut self`).
    pub fn typed_get(&mut self, handle: Handle, i: usize) -> Option<NanBox> {
        let (buffer, byte_offset, length, kind) = self.heap.get(handle)?.as_typed_array()?;
        // A fixed-length view whose resizable buffer shrank below its extent is
        // out of bounds: every integer-indexed read is `undefined` (IsValidInteger
        // Index is false), per spec — never a decoded zero from the truncated bytes.
        if i >= length || self.typed_array_out_of_bounds(handle) {
            return Some(NanBox::undefined());
        }
        let size = typed_elem_size(kind);
        // Checked offset math: `byte_offset + i*size` cannot overflow for an
        // in-range `i`, but compute defensively so a corrupt view never wraps.
        let start = i.checked_mul(size).and_then(|o| byte_offset.checked_add(o));
        let bytes = self.bytes_at(buffer)?;
        let slice = start
            .and_then(|s| s.checked_add(size).and_then(|e| bytes.get(s..e)))
            .unwrap_or(&[]);
        if is_bigint_kind(kind) {
            let big = decode_bigint_element(kind, slice);
            return Some(NanBox::handle(self.new_bigint(big).to_raw()));
        }
        Some(NanBox::number(decode_typed_element(kind, slice)))
    }

    /// `view[i] = value` — coerces `value` to the view's element kind and encodes
    /// it into the shared bytes. A write past the view's length is ignored (typed
    /// arrays are fixed-length). Returns `false` if `handle` is not a view.
    pub fn typed_set(&mut self, handle: Handle, i: usize, value: NanBox) -> bool {
        let Some((buffer, byte_offset, length, kind)) =
            self.heap.get(handle).and_then(Cell::as_typed_array)
        else {
            return false;
        };
        if i >= length {
            return true; // out-of-bounds typed-array write is a silent no-op
        }
        let (enc, enc_len) = if is_bigint_kind(kind) {
            // A BigInt element: the value must be a BigInt (the interpreter has
            // already applied ToBigInt and thrown for a Number). A non-BigInt
            // reaching here is rejected rather than silently written.
            let Some(big) = value
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.bigint_at(h))
            else {
                return false;
            };
            (encode_bigint_element(&big), 8)
        } else {
            let n = self.to_number(value);
            encode_typed_element(kind, coerce_typed(u16::from(kind), n))
        };
        let size = typed_elem_size(kind);
        // Checked offset math: never wrap (debug panic / silent corruption) on a
        // pathological `i`.
        let Some(start) = i.checked_mul(size).and_then(|o| byte_offset.checked_add(o)) else {
            return true;
        };
        if let Some(bytes) = self.bytes_at_mut(buffer) {
            for (j, &b) in enc[..enc_len].iter().enumerate() {
                if let Some(slot) = bytes.get_mut(start + j) {
                    *slot = b;
                }
            }
        }
        true
    }

    /// Bulk `view[start..end] = value` (`TypedArray.prototype.fill`). Resolves the
    /// view geometry **once**, encodes `value` once, then writes the repeating
    /// element pattern in a tight loop over a single borrow of the backing bytes —
    /// no per-element heap lookup or `Vec` allocation. `start`/`end` are clamped to
    /// the view length. No-op if `handle` is not a view.
    pub fn typed_fill_range(&mut self, handle: Handle, value: NanBox, start: usize, end: usize) {
        let Some((buffer, byte_offset, length, kind)) =
            self.heap.get(handle).and_then(Cell::as_typed_array)
        else {
            return;
        };
        let size = typed_elem_size(kind);
        let (enc, enc_len) = if is_bigint_kind(kind) {
            // A BigInt-element fill: the value is a BigInt (already ToBigInt'd by
            // the interpreter). A non-BigInt is a no-op (the interpreter throws
            // before calling).
            let Some(big) = value
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.bigint_at(h))
            else {
                return;
            };
            (encode_bigint_element(&big), 8)
        } else {
            let n = self.to_number(value);
            encode_typed_element(kind, coerce_typed(u16::from(kind), n))
        };
        let pat = &enc[..enc_len];
        let start = start.min(length);
        let end = end.min(length);
        if start >= end {
            return;
        }
        // `byte_offset + end*size` is the highest byte touched; clamp the loop to
        // the live byte slice so a stale view never writes out of bounds.
        let Some(base) = byte_offset.checked_add(start * size) else {
            return;
        };
        if let Some(bytes) = self.bytes_at_mut(buffer) {
            let mut off = base;
            for _ in start..end {
                if let Some(slot) = bytes.get_mut(off..off + size) {
                    slot.copy_from_slice(pat);
                }
                off += size;
            }
        }
    }

    /// Bulk `view.copyWithin(dst, src, count)`: copies `count` elements of the same
    /// width within the view's backing bytes via a raw `copy_within` (handles
    /// overlap correctly), resolving the view geometry once. `dst`/`src`/`count`
    /// are clamped to the view length. No-op if `handle` is not a view.
    pub fn typed_copy_within(&mut self, handle: Handle, dst: usize, src: usize, count: usize) {
        let Some((buffer, byte_offset, length, kind)) =
            self.heap.get(handle).and_then(Cell::as_typed_array)
        else {
            return;
        };
        let size = typed_elem_size(kind);
        // Clamp the element count so neither range exceeds the view length.
        let count = count
            .min(length.saturating_sub(dst))
            .min(length.saturating_sub(src));
        if count == 0 {
            return;
        }
        let (Some(src_byte), Some(dst_byte), Some(byte_count)) = (
            byte_offset.checked_add(src * size),
            byte_offset.checked_add(dst * size),
            count.checked_mul(size),
        ) else {
            return;
        };
        if let Some(bytes) = self.bytes_at_mut(buffer) {
            let hi = src_byte.max(dst_byte) + byte_count;
            if hi <= bytes.len() {
                bytes.copy_within(src_byte..src_byte + byte_count, dst_byte);
            }
        }
    }

    /// Bulk `view[offset + j] = values[j]` for `TypedArray.prototype.set` /
    /// constructor write-backs from a numeric source. ToNumber-coerces each value
    /// up front (which may run user `valueOf`), then writes the encoded elements in
    /// a single borrow of the backing bytes. Out-of-range slots are skipped. No-op
    /// if `handle` is not a view.
    pub fn typed_set_from_numbers(&mut self, handle: Handle, offset: usize, values: &[NanBox]) {
        let Some((.., kind)) = self.heap.get(handle).and_then(Cell::as_typed_array) else {
            return;
        };
        if is_bigint_kind(kind) {
            // BigInt-element write-through: each source value must already be a
            // BigInt (the interpreter applies ToBigInt and throws for a Number
            // before calling). Pre-extract the low-64-bit encodings, then write
            // them in one borrow of the backing bytes. A non-BigInt slot is left
            // unwritten.
            let encs: Vec<Option<[u8; 8]>> = values
                .iter()
                .map(|&v| {
                    v.as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.bigint_at(h))
                        .map(|b| encode_bigint_element(&b))
                })
                .collect();
            let Some((buffer, byte_offset, length, _)) =
                self.heap.get(handle).and_then(Cell::as_typed_array)
            else {
                return;
            };
            if let Some(bytes) = self.bytes_at_mut(buffer) {
                for (j, enc) in encs.iter().enumerate() {
                    let i = offset + j;
                    if i >= length {
                        break;
                    }
                    let Some(enc) = enc else { continue };
                    let Some(start) = i.checked_mul(8).and_then(|o| byte_offset.checked_add(o))
                    else {
                        break;
                    };
                    if let Some(slot) = bytes.get_mut(start..start + 8) {
                        slot.copy_from_slice(enc);
                    }
                }
            }
            return;
        }
        // ToNumber first (may call user code / resize the buffer); collect plain
        // f64s so the subsequent byte write needs only one immutable conversion.
        let nums: Vec<f64> = values.iter().map(|&v| self.to_number(v)).collect();
        // Re-resolve geometry after coercion (a `valueOf` could have detached or
        // resized the buffer).
        let Some((buffer, byte_offset, length, _)) =
            self.heap.get(handle).and_then(Cell::as_typed_array)
        else {
            return;
        };
        let size = typed_elem_size(kind);
        if let Some(bytes) = self.bytes_at_mut(buffer) {
            for (j, &num) in nums.iter().enumerate() {
                let i = offset + j;
                if i >= length {
                    break;
                }
                let (enc, enc_len) = encode_typed_element(kind, coerce_typed(u16::from(kind), num));
                let Some(start) = i.checked_mul(size).and_then(|o| byte_offset.checked_add(o))
                else {
                    break;
                };
                if let Some(slot) = bytes.get_mut(start..start + enc_len) {
                    slot.copy_from_slice(&enc[..enc_len]);
                }
            }
        }
    }

    /// Fast path for `dst.set(src)` when both are the **same element kind**: copies
    /// the raw source bytes into the destination at element `offset` (handling
    /// overlap when they share a backing buffer), skipping the
    /// decode→ToNumber→re-encode round-trip. Returns `false` (caller falls back to
    /// the generic path) if either is not a view, the kinds differ, or the copy
    /// would not fit. `offset + src.len() <= dst.len()` must already hold.
    pub fn typed_set_same_kind(&mut self, dst: Handle, src: Handle, offset: usize) -> bool {
        let (Some((dbuf, doff, dlen, dkind)), Some((sbuf, soff, slen, skind))) = (
            self.heap.get(dst).and_then(Cell::as_typed_array),
            self.heap.get(src).and_then(Cell::as_typed_array),
        ) else {
            return false;
        };
        if dkind != skind || offset.checked_add(slen).is_none_or(|e| e > dlen) {
            return false;
        }
        let size = typed_elem_size(dkind);
        let (Some(dst_byte), Some(src_byte), Some(byte_count)) = (
            doff.checked_add(offset * size),
            Some(soff),
            slen.checked_mul(size),
        ) else {
            return false;
        };
        if dbuf == sbuf {
            // Same backing buffer: overlap-safe move within one byte slice.
            if let Some(bytes) = self.bytes_at_mut(dbuf) {
                let hi = src_byte.max(dst_byte) + byte_count;
                if hi <= bytes.len() {
                    bytes.copy_within(src_byte..src_byte + byte_count, dst_byte);
                }
            }
            return true;
        }
        // Distinct buffers: read the source range, then write it to the dest.
        let Some(chunk) = self
            .bytes_at(sbuf)
            .and_then(|b| b.get(src_byte..src_byte + byte_count))
            .map(<[u8]>::to_vec)
        else {
            return false;
        };
        if let Some(bytes) = self.bytes_at_mut(dbuf)
            && let Some(slot) = bytes.get_mut(dst_byte..dst_byte + byte_count)
        {
            slot.copy_from_slice(&chunk);
        }
        true
    }

    /// The decoded elements of the typed-array view at `handle` as an owned
    /// vector, or `None` if it is not a view. `BigInt64Array`/`BigUint64Array`
    /// elements decode to freshly allocated `BigInt`s (hence `&mut self`).
    pub fn typed_elements(&mut self, handle: Handle) -> Option<Vec<NanBox>> {
        let (buffer, byte_offset, length, kind) = self.heap.get(handle)?.as_typed_array()?;
        let size = typed_elem_size(kind);
        if is_bigint_kind(kind) {
            // Decode the raw BigInts first (immutable borrow of the bytes), then
            // allocate each on the heap (needs `&mut self`).
            let bytes = self.bytes_at(buffer)?;
            let bigs: Vec<crate::bignum::BigInt> = (0..length)
                .map(|i| {
                    let start = byte_offset + i * size;
                    let slice = bytes.get(start..start + size).unwrap_or(&[]);
                    decode_bigint_element(kind, slice)
                })
                .collect();
            return Some(
                bigs.into_iter()
                    .map(|b| NanBox::handle(self.new_bigint(b).to_raw()))
                    .collect(),
            );
        }
        let bytes = self.bytes_at(buffer)?;
        Some(
            (0..length)
                .map(|i| {
                    let start = byte_offset + i * size;
                    let slice = bytes.get(start..start + size).unwrap_or(&[]);
                    NanBox::number(decode_typed_element(kind, slice))
                })
                .collect(),
        )
    }

    /// The elements of an array **or** typed-array view at `handle` as an owned
    /// vector. Unifies the read path so callers (iteration, spread, array
    /// methods, JSON, …) treat both alike. `None` for any other cell.
    pub fn elements_vec(&mut self, handle: Handle) -> Option<Vec<NanBox>> {
        if let Some(a) = self.array_elements(handle) {
            return Some(a.to_vec());
        }
        self.typed_elements(handle)
    }

    /// Whether `handle` is an array **or** a typed-array view — the values that
    /// support indexed element access and a `length`.
    #[must_use]
    pub fn is_array_like(&self, handle: Handle) -> bool {
        self.is_array(handle) || self.typed_len(handle).is_some()
    }

    /// Whether `handle` is a plausible target for a *generic* `Array.prototype`
    /// method applied via `.call`/`.apply` (or inherited through an array
    /// prototype) — i.e. an ordinary object-ish value whose `length`/indexed
    /// properties should be read array-like. This is every heap object **except**
    /// a real `Array` (its own fast path), a typed array, a `Map`/`Set`, a
    /// `Promise`, and the bare `Symbol`/`BigInt`/string/bytes primitives.
    /// `Object`, `Function`, `Date`, `RegExp`, primitive wrappers, `arguments`,
    /// and proxies all qualify.
    #[must_use]
    pub fn is_generic_array_like_target(&self, handle: Handle) -> bool {
        let Some(cell) = self.heap.get(handle) else {
            return false;
        };
        if self.is_array(handle) || self.typed_len(handle).is_some() {
            return false;
        }
        !matches!(
            cell,
            Cell::Collection { .. }
                | Cell::Promise(_)
                | Cell::Symbol { .. }
                | Cell::BigInt(_)
                | Cell::Str(_)
                | Cell::Bytes(_)
        )
    }

    /// Allocates a fresh, unique `Symbol` with the given description.
    pub fn new_symbol(&mut self, description: &str) -> Handle {
        let id = self.next_symbol_id;
        self.next_symbol_id += 1;
        let handle = self.heap.alloc(Cell::Symbol {
            description: alloc::boxed::Box::from(description),
            id,
        });
        self.symbols_by_id.insert(id, handle);
        handle
    }

    /// The heap handle of the symbol with the given `id`, if known.
    #[must_use]
    pub fn symbol_for_id(&self, id: u64) -> Option<Handle> {
        self.symbols_by_id.get(&id).copied()
    }

    /// The `(description, id)` of the symbol at `handle`, if it is one.
    #[must_use]
    pub fn symbol_at(&self, handle: Handle) -> Option<(alloc::string::String, u64)> {
        self.heap
            .get(handle)?
            .as_symbol()
            .map(|(d, id)| (alloc::string::String::from(d), id))
    }

    /// Allocates a `BigInt` with value `n`.
    pub fn new_bigint(&mut self, n: crate::bignum::BigInt) -> Handle {
        self.heap.alloc(Cell::BigInt(n))
    }

    /// Allocates a `Proxy` wrapping `target` with trap `handler`.
    pub fn new_proxy(&mut self, target: Handle, handler: Handle) -> Handle {
        self.heap.alloc(Cell::Proxy {
            target,
            handler,
            revoked: false,
        })
    }

    /// The `(target, handler)` of the proxy at `handle`, if it is one.
    #[must_use]
    pub fn proxy_at(&self, handle: Handle) -> Option<(Handle, Handle)> {
        self.heap.get(handle)?.as_proxy()
    }

    /// Whether the proxy at `handle` has been revoked.
    #[must_use]
    pub fn proxy_revoked(&self, handle: Handle) -> bool {
        self.heap
            .get(handle)
            .and_then(Cell::proxy_revoked)
            .unwrap_or(false)
    }

    /// Revokes the proxy at `handle` (`Proxy.revocable().revoke`).
    pub fn revoke_proxy(&mut self, handle: Handle) {
        if let Some(c) = self.heap.get_mut(handle) {
            c.revoke_proxy();
        }
    }

    /// Rewrites the `target`/`handler` of the proxy at `handle`. Used by snapshot
    /// restore to fill a placeholder proxy once its referents are allocated.
    pub fn proxy_set_targets(&mut self, handle: Handle, target: Handle, handler: Handle) {
        if let Some(Cell::Proxy {
            target: t,
            handler: h,
            ..
        }) = self.heap.get_mut(handle)
        {
            *t = target;
            *h = handler;
        }
    }

    /// The value of the `BigInt` at `handle` (cloned), if it is one.
    #[must_use]
    pub fn bigint_at(&self, handle: Handle) -> Option<crate::bignum::BigInt> {
        self.heap.get(handle)?.as_bigint().cloned()
    }

    /// Allocates an array of `elements` in the heap and returns its handle.
    pub fn new_array(&mut self, elements: Vec<NanBox>) -> Handle {
        self.heap.alloc(Cell::Array(elements))
    }

    /// Allocates a closure: a function-table index plus its captured scope.
    pub fn new_function(&mut self, func_id: u32, env: crate::env::Scope) -> Handle {
        let h = self.heap.alloc(Cell::Function { func_id, env });
        // Remember this function as the `constructor` for its `.prototype` (set when
        // the prototype is first materialized). If the prototype already exists,
        // link it now.
        self.fn_ctor.insert(func_id, h);
        if let Some(proto) = self.fn_protos.get(&func_id).copied() {
            self.set_hidden_property(proto, "constructor", NanBox::handle(h.to_raw()));
        }
        h
    }

    /// The `(func_id, captured env)` of the function at `handle`, or `None` if it
    /// is not callable.
    #[must_use]
    pub fn function_at(&self, handle: Handle) -> Option<(u32, crate::env::Scope)> {
        let (id, env) = self.heap.get(handle)?.as_function()?;
        Some((id, env.clone()))
    }

    /// Allocates a built-in (native) function with the given id.
    pub fn new_native(&mut self, id: u16) -> Handle {
        self.heap.alloc(Cell::Native(id))
    }

    /// Allocates a **host** (dynamically-registered) native function naming the
    /// registry entry `id` (see `Interp::register_fn`, `ROADMAP.md` §4.0).
    pub fn new_host_fn(&mut self, id: u32) -> Handle {
        self.heap.alloc(Cell::HostFn(id))
    }

    /// The host-function registry index at `handle`, or `None` if the cell is
    /// not a host function.
    #[must_use]
    pub fn host_fn_at(&self, handle: Handle) -> Option<u32> {
        self.heap.get(handle)?.as_host_fn()
    }

    /// Allocates a class value (a class-table index plus its captured scope).
    pub fn new_class(&mut self, class_id: u32, env: crate::env::Scope) -> Handle {
        self.heap.alloc(Cell::Class { class_id, env })
    }

    /// The cached `.prototype` object for the class with id `class_id`, if it has
    /// already been materialized.
    #[must_use]
    pub fn class_prototype_cached(&self, class_id: u32) -> Option<Handle> {
        self.class_protos.get(&class_id).copied()
    }

    /// Registers a freshly-created `.prototype` object for the class with id
    /// `class_id` (so repeated `C.prototype` reads return the same object and the
    /// constructor's instances can share it).
    pub fn set_class_prototype(&mut self, class_id: u32, proto: Handle) {
        self.class_protos.insert(class_id, proto);
    }

    /// The `(class_id, captured env)` of the class at `handle`, or `None`.
    #[must_use]
    pub fn class_at(&self, handle: Handle) -> Option<(u32, crate::env::Scope)> {
        let (id, env) = self.heap.get(handle)?.as_class()?;
        Some((id, env.clone()))
    }

    /// Allocates an empty `Map` (`is_set = false`) or `Set` (`is_set = true`).
    pub fn new_collection(&mut self, is_set: bool) -> Handle {
        self.heap.alloc(Cell::Collection {
            is_set,
            is_weak: false,
            entries: Vec::new(),
        })
    }

    /// Marks a collection as weak (`WeakMap`/`WeakSet`), so its keys are validated.
    pub fn set_collection_weak(&mut self, handle: Handle) {
        if let Some(Cell::Collection { is_weak, .. }) = self.heap.get_mut(handle) {
            *is_weak = true;
        }
    }

    /// Whether `handle` is a weak collection (`WeakMap`/`WeakSet`).
    #[must_use]
    pub fn collection_is_weak(&self, handle: Handle) -> bool {
        matches!(
            self.heap.get(handle),
            Some(Cell::Collection { is_weak: true, .. })
        )
    }

    /// `SameValueZero(a, b)` — the key equality `Map`/`Set` use: strict equality,
    /// except `NaN` equals `NaN` (`+0`/`-0` already compare equal under `===`).
    #[must_use]
    pub fn same_value_zero(&self, a: NanBox, b: NanBox) -> bool {
        self.strict_equals(a, b)
            || (a.as_number().is_some_and(f64::is_nan) && b.as_number().is_some_and(f64::is_nan))
    }

    /// `SameValue(a, b)` — like `===` but `NaN` equals `NaN` and `+0`/`-0` differ.
    /// This is the equality `ValidateAndApplyPropertyDescriptor` uses to decide
    /// whether a non-configurable data property's value actually changed.
    #[must_use]
    pub fn same_value(&self, a: NanBox, b: NanBox) -> bool {
        match (a.as_number(), b.as_number()) {
            (Some(x), Some(y)) => {
                (x == y && (x != 0.0 || x.is_sign_positive() == y.is_sign_positive()))
                    || (x.is_nan() && y.is_nan())
            }
            _ => self.strict_equals(a, b),
        }
    }

    /// Sets `key → value` in the collection at `handle` (inserting or updating,
    /// by `SameValueZero` key match). Returns `false` if not a collection.
    pub fn collection_set(&mut self, handle: Handle, key: NanBox, value: NanBox) -> bool {
        // Find an existing key first (immutable borrow), then write.
        let pos = match self.heap.get(handle).and_then(Cell::as_collection) {
            Some((_, entries)) => entries
                .iter()
                .position(|(k, _)| self.same_value_zero(*k, key)),
            None => return false,
        };
        let Some((_, entries)) = self.heap.get_mut(handle).and_then(Cell::as_collection_mut) else {
            return false;
        };
        match pos {
            Some(i) => entries[i].1 = value,
            None => entries.push((key, value)),
        }
        self.write_barrier(handle, key);
        self.write_barrier(handle, value);
        true
    }

    /// The value for `key` in the collection, or `None` if absent / not a
    /// collection.
    #[must_use]
    pub fn collection_get(&self, handle: Handle, key: NanBox) -> Option<NanBox> {
        let (_, entries) = self.heap.get(handle)?.as_collection()?;
        entries
            .iter()
            .find(|(k, _)| self.same_value_zero(*k, key))
            .map(|(_, v)| *v)
    }

    /// Whether the collection contains `key`.
    #[must_use]
    pub fn collection_has(&self, handle: Handle, key: NanBox) -> bool {
        self.heap
            .get(handle)
            .and_then(Cell::as_collection)
            .is_some_and(|(_, e)| e.iter().any(|(k, _)| self.same_value_zero(*k, key)))
    }

    /// Removes `key`; returns whether it was present.
    /// `Map.clear()` / `Set.clear()` — removes all entries.
    pub fn collection_clear(&mut self, handle: Handle) {
        if let Some((_, e)) = self.heap.get_mut(handle).and_then(Cell::as_collection_mut) {
            e.clear();
        }
    }

    /// `Map.delete(key)` / `Set.delete(value)` — removes the entry, returning
    /// whether one was present.
    pub fn collection_delete(&mut self, handle: Handle, key: NanBox) -> bool {
        let pos = match self.heap.get(handle).and_then(Cell::as_collection) {
            Some((_, e)) => e.iter().position(|(k, _)| self.same_value_zero(*k, key)),
            None => return false,
        };
        if let Some(i) = pos
            && let Some((_, e)) = self.heap.get_mut(handle).and_then(Cell::as_collection_mut)
        {
            e.remove(i);
            return true;
        }
        false
    }

    /// The number of entries, or `None` if not a collection.
    #[must_use]
    pub fn collection_size(&self, handle: Handle) -> Option<usize> {
        Some(self.heap.get(handle)?.as_collection()?.1.len())
    }

    /// A snapshot of the collection's entries (for iteration / `forEach`).
    #[must_use]
    pub fn collection_entries(&self, handle: Handle) -> Option<Vec<(NanBox, NanBox)>> {
        Some(self.heap.get(handle)?.as_collection()?.1.to_vec())
    }

    /// Whether the collection at `handle` is a `Set` (vs a `Map`).
    #[must_use]
    pub fn collection_is_set(&self, handle: Handle) -> Option<bool> {
        Some(self.heap.get(handle)?.as_collection()?.0)
    }

    /// The native-function id at `handle`, or `None` if it is not a native.
    #[must_use]
    pub fn native_at(&self, handle: Handle) -> Option<u16> {
        self.heap.get(handle)?.as_native()
    }

    /// The `(id, target)` of a bound native at `handle`.
    #[must_use]
    pub fn bound_native_at(&self, handle: Handle) -> Option<(u16, Handle)> {
        self.heap.get(handle)?.as_bound_native()
    }

    /// Allocates a bound native function (e.g. a promise resolve/reject).
    pub fn new_bound_native(&mut self, id: u16, target: Handle) -> Handle {
        self.heap.alloc(Cell::BoundNative { id, target })
    }

    /// Allocates a `Date` from a millisecond timestamp.
    pub fn new_date(&mut self, ms: f64) -> Handle {
        self.heap.alloc(Cell::Date(ms))
    }

    /// The timestamp (ms) of the `Date` at `handle`, if it is one.
    #[must_use]
    pub fn date_at(&self, handle: Handle) -> Option<f64> {
        self.heap.get(handle)?.as_date()
    }

    /// Sets a `Date`'s timestamp (for the `set*` mutators). Returns whether the
    /// handle was a Date.
    pub fn set_date_ms(&mut self, handle: Handle, ms: f64) -> bool {
        if let Some(cell @ Cell::Date(_)) = self.heap.get_mut(handle) {
            *cell = Cell::Date(ms);
            true
        } else {
            false
        }
    }

    /// Allocates a `RegExp` from its source and flags.
    pub fn new_regexp(&mut self, source: &str, flags: &str) -> Handle {
        self.heap.alloc(Cell::RegExp {
            source: alloc::boxed::Box::from(source),
            flags: alloc::boxed::Box::from(flags),
            last_index: 0,
            #[cfg(feature = "regex")]
            compiled: core::cell::RefCell::new(None),
        })
    }

    /// Recompiles the `RegExp` at `handle` in place (`RegExp.prototype.compile`):
    /// replaces its source/flags, drops the cached compiled program, and resets
    /// `lastIndex` to 0. A no-op if `handle` is not a RegExp.
    pub fn recompile_regexp(&mut self, handle: Handle, source: &str, flags: &str) {
        if let Some(Cell::RegExp {
            source: s,
            flags: f,
            last_index,
            #[cfg(feature = "regex")]
            compiled,
        }) = self.heap.get_mut(handle)
        {
            *s = alloc::boxed::Box::from(source);
            *f = alloc::boxed::Box::from(flags);
            *last_index = 0;
            #[cfg(feature = "regex")]
            {
                *compiled.borrow_mut() = None;
            }
        }
    }

    /// The compiled program for the `RegExp` at `handle`, compiled+cached on first
    /// use (RE-P1) and returned as a cheap `Rc` clone thereafter. Returns `None`
    /// if `handle` is not a `RegExp` or its pattern fails to compile (callers then
    /// surface the same `null`/error behavior as a fresh `Regex::new` failure).
    ///
    /// The cache is a transient `RefCell` on the cell; it holds no heap handles
    /// and is never serialized, so reusing one RegExp across many calls compiles
    /// the pattern exactly once. `lastIndex` is independent mutable state and is
    /// not touched here.
    #[cfg(feature = "regex")]
    #[must_use]
    pub fn regex_compiled(&self, handle: Handle) -> Option<alloc::rc::Rc<crate::regex::Regex>> {
        let Some(Cell::RegExp {
            source,
            flags,
            compiled,
            ..
        }) = self.heap.get(handle)
        else {
            return None;
        };
        if let Some(rc) = compiled.borrow().as_ref() {
            return Some(rc.clone());
        }
        let re = alloc::rc::Rc::new(crate::regex::Regex::new(source, flags).ok()?);
        *compiled.borrow_mut() = Some(re.clone());
        Some(re)
    }

    /// The `RegExp`'s `lastIndex` (0 if not a RegExp).
    #[must_use]
    pub fn regex_last_index(&self, handle: Handle) -> usize {
        match self.heap.get(handle) {
            Some(Cell::RegExp { last_index, .. }) => *last_index,
            _ => 0,
        }
    }

    /// Sets the `RegExp`'s `lastIndex`.
    pub fn set_regex_last_index(&mut self, handle: Handle, n: usize) {
        if let Some(Cell::RegExp { last_index, .. }) = self.heap.get_mut(handle) {
            *last_index = n;
        }
    }

    /// The `(source, flags)` of the `RegExp` at `handle` (owned), if it is one.
    #[must_use]
    pub fn regexp_at(
        &self,
        handle: Handle,
    ) -> Option<(alloc::string::String, alloc::string::String)> {
        let (s, f) = self.heap.get(handle)?.as_regexp()?;
        Some((
            alloc::string::String::from(s),
            alloc::string::String::from(f),
        ))
    }

    /// Allocates a pending `Promise`.
    pub fn new_promise(&mut self) -> Handle {
        self.heap
            .alloc(Cell::Promise(alloc::rc::Rc::new(core::cell::RefCell::new(
                crate::cell::PromiseState {
                    status: crate::cell::PromiseStatus::Pending,
                    value: NanBox::undefined(),
                    reactions: alloc::vec::Vec::new(),
                },
            ))))
    }

    /// The shared promise state at `handle`, if it is a promise — or a **Promise
    /// subclass** instance (`class P extends Promise`), whose backing
    /// `Cell::Promise` is stored in a hidden internal slot so every promise
    /// operation (then/resolve/reject/combinators/microtasks) works on it.
    #[must_use]
    pub fn promise_state(
        &self,
        handle: Handle,
    ) -> Option<alloc::rc::Rc<core::cell::RefCell<crate::cell::PromiseState>>> {
        if let Some(p) = self.heap.get(handle)?.as_promise() {
            return Some(p.clone());
        }
        // A subclass instance: follow the internal slot to its backing promise cell.
        let inner = self.get_property(handle, PROMISE_STATE_SLOT)?.as_handle()?;
        self.heap
            .get(Handle::from_raw(inner))?
            .as_promise()
            .cloned()
    }

    /// The string at `handle` as a `String`, or `None` if it is not a string
    /// (or the handle is stale).
    #[must_use]
    pub fn string_value(&self, handle: Handle) -> Option<alloc::string::String> {
        Some(self.heap.get(handle)?.as_str()?.materialize())
    }

    /// The array elements at `handle`, or `None` if it is not an array.
    #[must_use]
    pub fn array_elements(&self, handle: Handle) -> Option<&[NanBox]> {
        self.heap.get(handle)?.as_array()
    }

    /// Whether `handle` refers to an array.
    #[must_use]
    pub fn is_array(&self, handle: Handle) -> bool {
        self.heap.get(handle).and_then(Cell::as_array).is_some()
    }

    /// Whether `handle` is a bytecode-VM function value — a closure represented as
    /// an array tagged with the reserved `\0vmfn` marker. Such a value backs onto
    /// an array cell but is a function, so `Array.isArray` must reject it.
    #[must_use]
    pub fn is_vm_function(&self, handle: Handle) -> bool {
        self.get_property(handle, "\u{0}vmfn").is_some()
    }

    /// The own property names of the object at `handle`, in insertion order, or
    /// `None` if it is not an object.
    #[must_use]
    pub fn object_keys(&self, handle: Handle) -> Option<Vec<alloc::string::String>> {
        let obj = self.heap.get(handle)?.as_object()?;
        Some(
            obj.enumerable_keys()
                .iter()
                // Private fields (`#`-prefixed) and symbol/internal keys
                // (`\0`-prefixed) are never enumerable, so they stay out of
                // `Object.keys`, spread, `for-in`, and JSON. Methods are marked
                // hidden via `enumerable_keys`.
                .filter(|s| !s.starts_with('#') && !s.starts_with('\u{0}'))
                .map(|s| alloc::string::String::from(*s))
                .collect(),
        )
    }

    /// Enumerable named keys held in a handle's **auxiliary** object — the named
    /// properties an array/function/native carries alongside its elements (e.g.
    /// `arr.custom = …`, or a regex match result's `index`/`input`). Empty if none.
    #[must_use]
    pub fn aux_named_keys(&self, handle: Handle) -> Vec<alloc::string::String> {
        let Some(aux) = self.aux_props.get(&handle.to_raw()) else {
            return alloc::vec::Vec::new();
        };
        let Some(obj) = self.heap.get(*aux).and_then(Cell::as_object) else {
            return alloc::vec::Vec::new();
        };
        obj.enumerable_keys()
            .iter()
            .filter(|s| !s.starts_with('#') && !s.starts_with('\u{0}'))
            .map(|s| alloc::string::String::from(*s))
            .collect()
    }

    /// Own enumerable keys **including** symbol keys (the `\0sym:` internal
    /// names), excluding only private (`#`) fields — for `Object.assign` and
    /// spread, which copy own enumerable string *and* symbol properties.
    #[must_use]
    pub fn object_keys_with_symbols(&self, handle: Handle) -> Vec<alloc::string::String> {
        self.heap
            .get(handle)
            .and_then(Cell::as_object)
            .map(|obj| {
                obj.enumerable_keys()
                    .iter()
                    .filter(|s| !s.starts_with('#'))
                    .map(|s| alloc::string::String::from(*s))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// All own property keys (data and accessor, including non-enumerable) —
    /// for reflection that ignores enumerability (`getOwnPropertySymbols`).
    #[must_use]
    pub fn object_all_keys(&self, handle: Handle) -> Vec<alloc::string::String> {
        self.heap
            .get(handle)
            .and_then(Cell::as_object)
            .map(|obj| {
                obj.all_keys()
                    .iter()
                    .map(|s| alloc::string::String::from(*s))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The names of the object's accessor (getter/setter) properties.
    #[must_use]
    pub fn object_accessor_keys(&self, handle: Handle) -> Vec<alloc::string::String> {
        self.heap
            .get(handle)
            .and_then(Cell::as_object)
            .map(|o| {
                o.accessor_keys()
                    .iter()
                    .map(|s| alloc::string::String::from(*s))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// All own string property names (including non-enumerable ones such as
    /// methods, but not private `#` fields) — for `Object.getOwnPropertyNames`.
    pub fn own_property_names(&self, handle: Handle) -> Option<Vec<alloc::string::String>> {
        // A **String exotic object**'s `[[OwnPropertyKeys]]`: the index keys
        // `"0".."length-1"` (ascending), then `"length"`, then any wrapper named
        // own properties (a `defineProperty` on a `new String(...)`).
        if let Some(slen) = self.string_object_len(handle) {
            let mut names: Vec<alloc::string::String> =
                (0..slen).map(|i| alloc::format!("{i}")).collect();
            names.push(alloc::string::String::from("length"));
            if let Some(obj) = self.heap.get(handle).and_then(Cell::as_object) {
                for k in obj.ordered_keys() {
                    if k.starts_with('#')
                        || k.starts_with('\u{0}')
                        || k == "length"
                        || k.parse::<usize>().is_ok_and(|i| i < slen)
                    {
                        continue;
                    }
                    names.push(alloc::string::String::from(k));
                }
            }
            return Some(names);
        }
        if let Some(obj) = self.heap.get(handle)?.as_object() {
            // `[[OwnPropertyKeys]]` order: integer indices ascending, then the rest in
            // insertion order (so `getOwnPropertyNames`/`Reflect.ownKeys` match `keys`).
            return Some(
                obj.ordered_keys()
                    .iter()
                    .filter(|s| !s.starts_with('#') && !s.starts_with('\u{0}'))
                    .map(|s| alloc::string::String::from(*s))
                    .collect(),
            );
        }
        // An array's own keys: its indices (ascending), then `length`, then any
        // aux-stored named properties — matching `[[OwnPropertyKeys]]` for an Array.
        if let Some(a) = self.heap.get(handle).and_then(Cell::as_array) {
            // Dense present indices (holes are absent — excluded), as a set so an
            // accessor-at-index (stored in the aux map, punched to a dense hole)
            // is not duplicated.
            let dense_len = a.len();
            let mut index_keys: Vec<u32> = (0..dense_len)
                .filter(|&i| !a[i].is_hole())
                .filter_map(|i| u32::try_from(i).ok())
                .collect();
            // Aux-stored properties split into integer-index keys (which join the
            // element keys, ascending) and ordinary named keys (which follow
            // `length`), per the array exotic `[[OwnPropertyKeys]]` ordering.
            let mut named: Vec<alloc::string::String> = Vec::new();
            if let Some(aux) = self
                .aux_props
                .get(&handle.to_raw())
                .and_then(|h| self.heap.get(*h))
                .and_then(Cell::as_object)
            {
                for k in aux
                    .all_keys()
                    .iter()
                    .filter(|s| !s.starts_with('#') && !s.starts_with('\u{0}'))
                {
                    // A canonical array-index key (`"0".."4294967294"`, no leading
                    // zeros) is an element key; anything else is a named key.
                    if let Ok(n) = k.parse::<u32>()
                        && n != u32::MAX
                        && alloc::format!("{n}") == **k
                    {
                        if !index_keys.contains(&n) {
                            index_keys.push(n);
                        }
                    } else {
                        named.push(alloc::string::String::from(*k));
                    }
                }
            }
            index_keys.sort_unstable();
            let mut names: Vec<alloc::string::String> =
                index_keys.iter().map(|i| alloc::format!("{i}")).collect();
            names.push(alloc::string::String::from("length"));
            names.extend(named);
            return Some(names);
        }
        // A typed array's own keys are its integer indices `0..length` (ascending),
        // followed by any aux-stored named properties — matching the integer-indexed
        // exotic `[[OwnPropertyKeys]]` (10.4.5.6).
        if let Some(len) = self.typed_len(handle) {
            let mut names: Vec<alloc::string::String> =
                (0..len).map(|i| alloc::format!("{i}")).collect();
            if let Some(aux) = self
                .aux_props
                .get(&handle.to_raw())
                .and_then(|h| self.heap.get(*h))
                .and_then(Cell::as_object)
            {
                for k in aux
                    .ordered_keys()
                    .iter()
                    .filter(|s| !s.starts_with('#') && !s.starts_with('\u{0}'))
                {
                    names.push(alloc::string::String::from(*k));
                }
            }
            return Some(names);
        }
        // A native / bound-native / VM-function cell keeps its own named properties
        // (e.g. a built-in function's `name`/`length`) in its auxiliary object.
        if matches!(
            self.heap.get(handle),
            Some(
                Cell::Native(_)
                    | Cell::HostFn(_)
                    | Cell::BoundNative { .. }
                    | Cell::Function { .. }
                    | Cell::Class { .. }
            )
        ) {
            let mut names = Vec::new();
            if let Some(aux) = self
                .aux_props
                .get(&handle.to_raw())
                .and_then(|h| self.heap.get(*h))
                .and_then(Cell::as_object)
            {
                // `[[OwnPropertyKeys]]` order: integer indices ascending, then the
                // rest in insertion order. `name`/`length` are stored as ordinary
                // named keys, so `ordered_keys` already yields the spec order.
                for k in aux
                    .ordered_keys()
                    .iter()
                    .filter(|s| !s.starts_with('#') && !s.starts_with('\u{0}'))
                {
                    names.push(alloc::string::String::from(*k));
                }
            }
            return Some(names);
        }
        None
    }

    /// The `[[Prototype]]` handle of the object at `handle`, if any. For a
    /// callable cell with no inline object part (a native / bound native), this
    /// is the explicit link recorded by [`set_native_proto`](Realm::set_native_proto)
    /// — so `Object.getPrototypeOf(Int8Array)` resolves to the shared
    /// `%TypedArray%` intrinsic rather than `null`.
    #[must_use]
    pub fn object_proto(&self, handle: Handle) -> Option<Handle> {
        if let Some(obj) = self.heap.get(handle).and_then(Cell::as_object) {
            return obj.proto();
        }
        if let Some(p) = self.native_protos.get(&handle.to_raw()).copied() {
            return Some(p);
        }
        // A Symbol primitive's `[[Prototype]]` is `%Symbol.prototype%`.
        if matches!(self.heap.get(handle), Some(Cell::Symbol { .. })) {
            return self.symbol_proto_intrinsic;
        }
        // A BigInt primitive's `[[Prototype]]` is `%BigInt.prototype%` (so
        // `Object.prototype.toString.call(1n)` reads its `@@toStringTag` →
        // "BigInt", and `(1n).toString`/`valueOf` resolve).
        if matches!(self.heap.get(handle), Some(Cell::BigInt(_))) {
            return self.bigint_proto_intrinsic;
        }
        // An ordinary/native callable with no explicit override and no inline
        // object part has `[[Prototype]] === %Function.prototype%` (unless it was
        // explicitly set to `null`).
        if self.is_callable_cell(handle) {
            if self.callable_null_protos.contains(&handle.to_raw()) {
                return None;
            }
            return self.function_proto_intrinsic;
        }
        // A dense `Cell::Array` with no explicit override and no inline object part
        // has `[[Prototype]] === %Array.prototype%` (unless set to `null`).
        if matches!(self.heap.get(handle), Some(Cell::Array(_))) {
            if self.callable_null_protos.contains(&handle.to_raw()) {
                return None;
            }
            return self.array_proto_intrinsic;
        }
        None
    }

    /// Whether `handle`'s prototype chain reaches the realm's `Object.prototype`
    /// — i.e. whether it inherits `Object.prototype`'s accessors (notably the
    /// `__proto__` getter/setter). A null-prototype object such as a module
    /// namespace exotic or `Object.create(null)` does *not*, so `obj.__proto__`
    /// on it is an ordinary (absent) property lookup yielding `undefined` rather
    /// than the `[[Prototype]]` link.
    pub fn inherits_object_proto(&self, handle: Handle) -> bool {
        let Some(target) = self.default_object_proto else {
            return true; // realm not fully wired; preserve legacy behaviour
        };
        let mut cur = self.object_proto(handle);
        let mut guard = 0u32;
        while let Some(h) = cur {
            if h == target {
                return true;
            }
            guard += 1;
            if guard > 10_000 {
                break;
            }
            cur = self.object_proto(h);
        }
        false
    }

    /// Sets the `[[Prototype]]` of the object at `handle`.
    pub fn set_object_proto(&mut self, handle: Handle, proto: Option<Handle>) -> bool {
        if let Some(obj) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            obj.set_proto(proto);
            if let Some(p) = proto {
                self.write_barrier(handle, NanBox::handle(p.to_raw()));
            }
            return true;
        }
        // A callable cell (function / native / bound native / class) or a dense
        // array has no inline object part; record its `[[Prototype]]` override in
        // the side table so `Object.getPrototypeOf` reflects it.
        if self.is_callable_cell(handle) || matches!(self.heap.get(handle), Some(Cell::Array(_))) {
            match proto {
                Some(p) => {
                    self.native_protos.insert(handle.to_raw(), p);
                    self.callable_null_protos.remove(&handle.to_raw());
                    self.write_barrier(handle, NanBox::handle(p.to_raw()));
                }
                None => {
                    // An explicit `null` prototype: a removed `native_protos` entry
                    // would re-default to `%Function.prototype%`, so track null.
                    self.native_protos.remove(&handle.to_raw());
                    self.callable_null_protos.insert(handle.to_raw());
                }
            }
            return true;
        }
        false
    }

    /// The cached `.prototype` object of the `Intl` service constructor with native
    /// dispatch id `ctor_id`, if it has been materialized.
    #[must_use]
    pub fn intl_prototype(&self, ctor_id: u16) -> Option<Handle> {
        self.intl_protos.get(&ctor_id).copied()
    }

    /// Records the `.prototype` object of the `Intl` service constructor with native
    /// dispatch id `ctor_id` (so instances can link to it and `Intl.X.prototype`
    /// reads return the same object).
    pub fn set_intl_prototype(&mut self, ctor_id: u16, proto: Handle) {
        self.intl_protos.insert(ctor_id, proto);
    }

    /// Allocates an empty object whose `[[Prototype]]` is `proto` (`Object.create`).
    pub fn new_object_with_proto(&mut self, proto: Option<Handle>) -> Handle {
        let h = self.new_object();
        self.set_object_proto(h, proto);
        h
    }

    /// Freezes the object at `handle` (`Object.freeze`); returns whether it was
    /// an object.
    pub fn freeze_object(&mut self, handle: Handle) -> bool {
        // An array carries no inline object part — track frozen-ness aside. Frozen
        // implies sealed and non-extensible.
        if self.heap.get(handle).and_then(Cell::as_array).is_some() {
            self.frozen_arrays.insert(handle.to_raw());
            self.sealed_arrays.insert(handle.to_raw());
            self.non_extensible_arrays.insert(handle.to_raw());
            return true;
        }
        match self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            Some(obj) => {
                obj.freeze();
                true
            }
            None => false,
        }
    }

    /// Whether the value at `handle` is a frozen object or array.
    #[must_use]
    pub fn is_frozen(&self, handle: Handle) -> bool {
        if self.frozen_arrays.contains(&handle.to_raw()) {
            return true;
        }
        self.heap
            .get(handle)
            .and_then(Cell::as_object)
            .is_some_and(crate::object::Object::is_frozen)
    }

    /// Whether the `length` of the array at `handle` was made non-writable via
    /// `Object.defineProperty(arr, "length", {writable:false})`. An array's
    /// `length` is writable by default.
    #[must_use]
    pub fn array_length_is_readonly(&self, handle: Handle) -> bool {
        // A frozen array's `length` is non-writable too (freeze implies it).
        self.nonwritable_array_lengths.contains(&handle.to_raw())
            || self.frozen_arrays.contains(&handle.to_raw())
    }

    /// Records (or clears) the non-writable flag for an array's `length`.
    pub fn set_array_length_readonly(&mut self, handle: Handle, readonly: bool) {
        if readonly {
            self.nonwritable_array_lengths.insert(handle.to_raw());
        } else {
            self.nonwritable_array_lengths.remove(&handle.to_raw());
        }
    }

    /// The length of the array (or typed-array view) at `handle`, or `None` if it
    /// is neither.
    #[must_use]
    pub fn array_length(&self, handle: Handle) -> Option<usize> {
        match self.heap.get(handle)? {
            Cell::Array(a) => {
                let dense = a.len();
                // A `length` set above the dense capacity (a valid uint32 beyond the
                // storage cap) is recorded as a logical override; report whichever is
                // larger so a later element write (growing the dense `Vec`) still wins.
                Some(
                    self.sparse_array_lengths
                        .get(&handle.to_raw())
                        .copied()
                        .map_or(dense, |logical| logical.max(dense)),
                )
            }
            Cell::TypedArray { length, .. } => {
                // An out-of-bounds fixed-length view reports length 0.
                if self.typed_array_out_of_bounds(handle) {
                    Some(0)
                } else {
                    Some(*length)
                }
            }
            _ => None,
        }
    }

    /// Whether the dense array slot at `index` is a *hole* (an absent index).
    /// `false` for a non-array, an out-of-range index, or a present value.
    #[must_use]
    pub fn array_hole_at(&self, handle: Handle, index: usize) -> bool {
        matches!(
            self.heap.get(handle).and_then(Cell::as_array),
            Some(a) if a.get(index).is_some_and(|e| e.is_hole())
        )
    }

    /// The *present* (non-hole) integer index keys of a dense array, ascending —
    /// the array's own enumerable indices for `Object.keys`/`for-in`/spread/etc.
    /// `None` if `handle` is not a plain array.
    #[must_use]
    pub fn array_present_indices(&self, handle: Handle) -> Option<Vec<usize>> {
        let a = self.heap.get(handle).and_then(Cell::as_array)?;
        Some((0..a.len()).filter(|&i| !a[i].is_hole()).collect())
    }

    /// The present integer index keys that are *enumerable*, ascending — for
    /// `for-in` / `Object.keys` / spread, which skip an index demoted to
    /// `enumerable: false` via `defineProperty`. Identical to
    /// [`array_present_indices`](Realm::array_present_indices) for the common array
    /// with no per-index overrides; the hidden-flag probe is skipped entirely when
    /// the array carries no aux object.
    #[must_use]
    pub fn array_enumerable_indices(&self, handle: Handle) -> Option<Vec<usize>> {
        let a = self.heap.get(handle).and_then(Cell::as_array)?;
        // No aux object ⇒ no index was ever demoted ⇒ every present index enumerates.
        let hidden_owner = self
            .aux_props
            .get(&handle.to_raw())
            .and_then(|aux| self.heap.get(*aux))
            .and_then(Cell::as_object);
        let Some(o) = hidden_owner else {
            return Some((0..a.len()).filter(|&i| !a[i].is_hole()).collect());
        };
        Some(
            (0..a.len())
                .filter(|&i| !a[i].is_hole() && !o.is_hidden(&alloc::format!("{i}")))
                .collect(),
        )
    }

    /// `arr[index]` — the element at `index`, or `undefined` if out of range or
    /// the cell is not an array (or typed-array view). Takes `&mut self` because a
    /// `BigInt64Array`/`BigUint64Array` element read allocates a `BigInt`.
    pub fn get_element(&mut self, handle: Handle, index: usize) -> NanBox {
        match self.heap.get(handle) {
            Some(Cell::Array(a)) => a.get(index).copied().unwrap_or(NanBox::undefined()),
            Some(Cell::TypedArray { .. }) => {
                self.typed_get(handle, index).unwrap_or(NanBox::undefined())
            }
            _ => NanBox::undefined(),
        }
    }

    /// Whether the array at `handle` carries *any* index override — a sparse
    /// attribute/accessor side-table entry, or being frozen/sealed. An O(1) gate
    /// so a plain dense array keeps its fast paths (e.g. `sort` reads/writes the
    /// element store directly) while an array with accessor indices takes the
    /// spec-precise `[[Get]]`/`[[Set]]` path.
    #[must_use]
    pub fn array_has_index_overrides(&self, handle: Handle) -> bool {
        let raw = handle.to_raw();
        self.frozen_arrays.contains(&raw)
            || self.sealed_arrays.contains(&raw)
            || self.aux_props.contains_key(&raw)
    }

    /// Whether the array index `i` carries a *non-default* own attribute or an
    /// accessor — i.e. a `defineProperty` once demoted its writability/enumerability/
    /// configurability or installed a getter/setter, so an `arr[i] = v` write cannot
    /// take the plain dense-store fast path and must consult the descriptor instead.
    ///
    /// Returns `false` immediately for the overwhelming common case (no aux object,
    /// not frozen), keeping the dense write path untouched and allocation-free.
    #[must_use]
    pub fn array_index_has_override(&self, handle: Handle, i: usize) -> bool {
        let raw = handle.to_raw();
        // A frozen array makes every index non-writable+non-configurable.
        if self.frozen_arrays.contains(&raw) || self.sealed_arrays.contains(&raw) {
            return true;
        }
        let Some(aux) = self.aux_props.get(&raw) else {
            return false;
        };
        let Some(o) = self.heap.get(*aux).and_then(Cell::as_object) else {
            return false;
        };
        let key = alloc::format!("{i}");
        o.is_readonly(&key)
            || o.is_hidden(&key)
            || o.is_non_configurable(&key)
            || o.accessor(&key).is_some()
    }

    /// `arr[index] = value` — grows the array with `undefined` holes if `index`
    /// is past the end (per JS). Returns `false` if the cell is not an array.
    pub fn set_element(&mut self, handle: Handle, index: usize, value: NanBox) -> bool {
        // A typed-array view writes through to its shared bytes (coercing to the
        // element kind); out-of-bounds writes are silent no-ops, per spec.
        if self.typed_len(handle).is_some() {
            return self.typed_set(handle, index, value);
        }
        // A frozen array rejects all element writes; a sealed / non-extensible array
        // rejects only writes that would grow it (a write to an existing index is
        // still allowed when merely sealed). The frozen/non-extensible registries
        // are empty for the overwhelming majority of arrays, so skip both `BTreeSet`
        // probes in that common case.
        let any_restricted =
            !self.frozen_arrays.is_empty() || !self.non_extensible_arrays.is_empty();
        if any_restricted && self.frozen_arrays.contains(&handle.to_raw()) {
            return false;
        }
        let non_ext = any_restricted && self.non_extensible_arrays.contains(&handle.to_raw());
        match self.heap.get_mut(handle).and_then(Cell::as_array_mut) {
            Some(a) => {
                if index >= a.len() {
                    if non_ext {
                        return false;
                    }
                    // C1: never grow the dense backing past the configured cap — a
                    // huge index (`a[4294967295] = 1`) would otherwise request a
                    // multi-gigabyte `Vec::resize` and abort the process. Refuse the
                    // grow (a no-op `false`, the same signal a frozen array returns)
                    // so the engine stays alive; callers can surface a catchable
                    // `RangeError("Invalid array length")`. `checked_add` guards the
                    // `usize` overflow at `index == usize::MAX`.
                    let Some(new_len) = index.checked_add(1) else {
                        return false;
                    };
                    if new_len > self.limits.max_array_len {
                        return false;
                    }
                    // The skipped slots between the old end and `index` become real
                    // holes (absent indices), not `undefined` data properties.
                    a.resize(new_len, NanBox::hole());
                }
                a[index] = value;
                self.write_barrier(handle, value);
                true
            }
            None => false,
        }
    }

    /// `arr.push(value)` — appends, returning the new length, or `None` if the
    /// cell is not an array (or the array is frozen).
    pub fn array_push(&mut self, handle: Handle, value: NanBox) -> Option<usize> {
        // A sealed / non-extensible (or frozen) array cannot grow. The registry is
        // empty for almost all arrays, so skip the `BTreeSet` probe in that case.
        if !self.non_extensible_arrays.is_empty()
            && self.non_extensible_arrays.contains(&handle.to_raw())
        {
            return None;
        }
        let max = self.limits.max_array_len;
        let a = self.heap.get_mut(handle).and_then(Cell::as_array_mut)?;
        // C1: refuse to grow past the configured cap rather than risk an unbounded
        // allocation (a single push only adds one element, but an array already at
        // the cap must not exceed it).
        if a.len() >= max {
            return None;
        }
        a.push(value);
        Some(a.len())
    }

    /// Replaces the whole contents of the array at `handle` (for in-place
    /// mutators like `splice`/`unshift`/`shift`). Returns whether it was an array.
    pub fn array_set_all(&mut self, handle: Handle, elems: Vec<NanBox>) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_array_mut) {
            Some(a) => {
                *a = elems;
                true
            }
            None => false,
        }
    }

    /// Punches a *hole* into the dense array store at `index`, growing the dense
    /// `Vec` (with holes) to `index + 1` when the index is at/beyond the current
    /// dense length — so `length` reports `index + 1`. Used by `defineProperty`
    /// when an array index becomes a user accessor (the accessor lives in the aux
    /// map; the dense slot must not shadow it with a stale data value, and the
    /// exotic `length` must still grow per ArrayDefineOwnProperty 10.4.2.1).
    /// No-op past the dense cap (the logical sparse length already covers it).
    pub fn array_index_to_hole(&mut self, handle: Handle, index: usize) {
        let cap = self.limits.max_array_len;
        if let Some(a) = self.heap.get_mut(handle).and_then(Cell::as_array_mut) {
            if index < a.len() {
                a[index] = NanBox::hole();
            } else if index < cap {
                a.resize(index + 1, NanBox::hole());
            } else {
                // Beyond the dense cap: record the logical length so `array_length`
                // reports `index + 1` without materializing billions of slots.
                let raw = handle.to_raw();
                let cur = self
                    .sparse_array_lengths
                    .get(&raw)
                    .copied()
                    .unwrap_or(0)
                    .max(a.len());
                self.sparse_array_lengths.insert(raw, cur.max(index + 1));
            }
        }
    }

    /// Sets an array's `length` (`arr.length = n`): truncates if smaller, pads
    /// with `undefined` if larger. Returns `false` only if not an array.
    ///
    /// A `len` within [`Limits::max_array_len`] resizes the dense backing `Vec`
    /// directly. A larger `len` (still a valid uint32, validated by the caller) is
    /// a *sparse* length: growing the dense `Vec` that far would request a
    /// multi-gigabyte allocation, so the dense storage is left untouched and the
    /// spec-visible `length` is recorded as a logical override (see
    /// [`array_length`](Realm::array_length)). This makes `arr.length = 4294967295`
    /// on an empty array report `4294967295` without materializing 4 billion holes.
    ///
    /// [`Limits::max_array_len`]: crate::limits::Limits::max_array_len
    pub fn set_array_length(&mut self, handle: Handle, len: usize) -> bool {
        let cap = self.limits.max_array_len;
        let raw = handle.to_raw();
        match self.heap.get_mut(handle).and_then(Cell::as_array_mut) {
            Some(a) => {
                if len > cap {
                    // Sparse: keep whatever dense elements exist (already <= cap), and
                    // record the logical length. A lowering below the dense size still
                    // truncates the dense `Vec`, but here `len > cap >= a.len()`.
                    self.sparse_array_lengths.insert(raw, len);
                } else {
                    // Within the dense cap: resize for real and drop any prior sparse
                    // override (the dense length is now authoritative). Slots added
                    // past the old end are *holes* (absent indices), not `undefined`
                    // data properties — `new Array(10)` and `arr.length = 10` create
                    // a sparse array whose indices report `HasProperty === false`.
                    a.resize(len, NanBox::hole());
                    self.sparse_array_lengths.remove(&raw);
                }
                true
            }
            None => false,
        }
    }

    /// Whether an `arr.length = new_len` set needs the descriptor-aware tree-walker
    /// rather than the VM's plain `set_array_length`. True when the array's `length`
    /// is non-writable (the set must be rejected / strict-thrown), or when a shrink
    /// could hit a non-configurable index (ArraySetLength's stop-and-fail). Returns
    /// `false` for the common unrestricted array, so the VM keeps its fast path.
    #[must_use]
    pub fn array_length_set_needs_slow_path(&self, handle: Handle, new_len: usize) -> bool {
        let raw = handle.to_raw();
        if self.array_length_is_readonly(handle) {
            return true;
        }
        // Only a shrink can hit a non-configurable index.
        let cur = self.array_length(handle).unwrap_or(0);
        if new_len >= cur {
            return false;
        }
        self.frozen_arrays.contains(&raw)
            || self.sealed_arrays.contains(&raw)
            || self
                .aux_props
                .get(&raw)
                .and_then(|a| self.heap.get(*a))
                .and_then(Cell::as_object)
                .is_some_and(|o| !o.non_configurable_is_empty())
    }

    /// `ArraySetLength` (ECMA-262 10.4.3.1) shrink semantics: lower the array's
    /// `length` to `new_len`, deleting elements from the top down. A to-be-deleted
    /// index that is **non-configurable** (a frozen/sealed array, or one demoted via
    /// `defineProperty(arr, i, {configurable:false})`) cannot be removed — deletion
    /// stops there, the length is left at `that_index + 1`, and `Ok(false)` is
    /// returned (a throwing context surfaces a TypeError). When every deletion
    /// down to `new_len` succeeds, the length becomes `new_len` and `Ok(true)` is
    /// returned. A non-shrinking `new_len` (>= current length) just sets the length.
    ///
    /// Returns `false` (the bool) only on the stop-at-non-configurable case; the
    /// outer `bool` of the tuple is whether `handle` was an array at all.
    #[must_use]
    pub fn array_set_length_truncating(&mut self, handle: Handle, new_len: usize) -> (bool, bool) {
        let cur = match self.array_length(handle) {
            Some(n) => n,
            None => return (false, false),
        };
        if new_len >= cur {
            return (true, self.set_array_length(handle, new_len));
        }
        // Find the highest present, non-configurable index in [new_len, cur). The
        // common case (no per-index flags, not frozen/sealed) has none, so the
        // override probe short-circuits and the truncation is a plain resize.
        let raw = handle.to_raw();
        let restricted = self.frozen_arrays.contains(&raw)
            || self.sealed_arrays.contains(&raw)
            || self
                .aux_props
                .get(&raw)
                .and_then(|a| self.heap.get(*a))
                .and_then(Cell::as_object)
                .is_some_and(|o| !o.non_configurable_is_empty());
        let mut stop_at: Option<usize> = None;
        if restricted {
            // Walk from the top down; the first index that is present (or has an
            // accessor) and non-configurable halts the deletion.
            let mut i = cur;
            while i > new_len {
                i -= 1;
                let present = !self.array_hole_at(handle, i)
                    && self
                        .heap
                        .get(handle)
                        .and_then(Cell::as_array)
                        .is_some_and(|a| i < a.len());
                let key = alloc::format!("{i}");
                let has_accessor = self.accessor(handle, &key).is_some();
                if (present || has_accessor) && self.property_is_non_configurable(handle, &key) {
                    stop_at = Some(i);
                    break;
                }
            }
        }
        match stop_at {
            // Deletion halted at `i`: keep everything up to and including it.
            Some(i) => {
                self.set_array_length(handle, i + 1);
                (false, true)
            }
            None => (true, self.set_array_length(handle, new_len)),
        }
    }

    /// `arr.pop()` — removes and returns the last element (`undefined` if empty
    /// or not an array).
    pub fn array_pop(&mut self, handle: Handle) -> NanBox {
        let v = self
            .heap
            .get_mut(handle)
            .and_then(Cell::as_array_mut)
            .and_then(Vec::pop)
            .unwrap_or(NanBox::undefined());
        // A popped hole reads as `undefined` (the sentinel never escapes).
        if v.is_hole() { NanBox::undefined() } else { v }
    }

    /// The `typeof` string for the heap value at `handle` (`"string"`/`"object"`),
    /// or `None` if the handle is stale.
    #[must_use]
    pub fn type_of(&self, handle: Handle) -> Option<&'static str> {
        Some(self.heap.get(handle)?.type_of())
    }

    /// The value of own property `key` on the object at `handle`, or `None` if
    /// the property is absent, the cell is not an object, or the handle is stale.
    #[must_use]
    pub fn get_property(&self, handle: Handle, key: &str) -> Option<NanBox> {
        if let Some(o) = self.heap.get(handle)?.as_object() {
            return o.get(key);
        }
        // A non-object cell (array/function): look in its auxiliary props.
        let aux = self.aux_props.get(&handle.to_raw())?;
        self.heap.get(*aux)?.as_object()?.get(key)
    }

    /// The inline-cache fast path for a plain object's `GetProp`: reads own data
    /// property `key` on the object at `handle`, consulting `cache` so a repeat
    /// access on the same shape skips the name lookup (a shape-pointer compare
    /// plus a slot load).
    ///
    /// Returns `None` — signalling the bytecode VM to take its slow path — when
    /// the cell at `handle` is *not* a plain shaped object (arrays, functions,
    /// strings, etc. carry no shape), or when the property is absent. A
    /// dictionary-mode object reads correctly through this method but never binds
    /// the cache (its sentinel shape resolves no key); see
    /// [`Object::cached_get`]. Accessors and special keys are the VM's
    /// responsibility and are filtered out before this is reached.
    pub fn object_cached_get(
        &self,
        handle: Handle,
        key: &str,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<NanBox> {
        self.heap.get(handle)?.as_object()?.cached_get(key, cache)
    }

    /// The inline-cache fast path for a plain object's `SetProp`: writes `value`
    /// to an *existing* own data property `key` on the object at `handle`,
    /// consulting `cache`, and reports whether the in-place write happened.
    ///
    /// Returns `false` — signalling the bytecode VM to take its slow path — when
    /// the cell is not a plain shaped object, when `key` is not already an own
    /// data property (a new property is a shape transition, handled by
    /// [`set_property`](Realm::set_property)), when the object is in dictionary
    /// mode, or when the property is frozen/read-only. On a `true` result the
    /// write barrier is applied, matching the slow path.
    pub fn object_cached_set(
        &mut self,
        handle: Handle,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
    ) -> bool {
        let wrote = self
            .heap
            .get_mut(handle)
            .and_then(Cell::as_object_mut)
            .is_some_and(|o| o.cached_set(key, value, cache));
        if wrote {
            self.write_barrier(handle, value);
        }
        wrote
    }

    /// Tags the object at `handle` with the class it was constructed from.
    pub fn set_class_tag(&mut self, handle: Handle, class_id: u32) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.set_class_tag(class_id);
        } else {
            // A non-object cell (a derived native instance: a real Map/Set/typed
            // array/… created by `class S extends Map {}`) keeps its class tag in a
            // side table, so `instanceof Subclass` still walks the class chain.
            self.native_class_tags.insert(handle.to_raw(), class_id);
        }
    }

    /// The class tag of the cell at `handle`, if any — the inline tag for a plain
    /// object, else the side-table tag for a derived native instance.
    #[must_use]
    pub fn class_tag(&self, handle: Handle) -> Option<u32> {
        if let Some(o) = self.heap.get(handle).and_then(Cell::as_object) {
            return o.class_tag();
        }
        self.native_class_tags.get(&handle.to_raw()).copied()
    }

    /// Deletes own property `key` from the object at `handle`; returns whether
    /// anything was removed.
    pub fn delete_property(&mut self, handle: Handle, key: &str) -> bool {
        let root = Rc::clone(&self.root_shape);
        // An array's `length` is non-configurable: `delete arr.length` always fails.
        if key == "length" && self.heap.get(handle).and_then(Cell::as_array).is_some() {
            return false;
        }
        // An array index is an element of the dense store (or a hole carrying an aux
        // accessor), not a named slot — `delete arr[i]` must punch a hole there, not
        // merely touch the aux object.
        if self.heap.get(handle).and_then(Cell::as_array).is_some()
            && key != "length"
            && let Ok(i) = key.parse::<usize>()
            && alloc::format!("{i}") == key
        {
            {
                // A present element, or an accessor installed over a hole, is the own
                // property to delete. Anything else (out of range, an already-absent
                // hole) is a no-op that still "succeeds".
                let present = !self.array_hole_at(handle, i)
                    && self
                        .heap
                        .get(handle)
                        .and_then(Cell::as_array)
                        .is_some_and(|a| i < a.len());
                let has_accessor = self.accessor(handle, key).is_some();
                if !present && !has_accessor {
                    return true;
                }
                // A non-configurable index (frozen array, or one demoted via
                // `defineProperty`) cannot be deleted.
                if self.property_is_non_configurable(handle, key) {
                    return false;
                }
                // Punch a hole in the dense store and clear every aux entry for the
                // index (accessor + attribute flags), so a later re-add starts fresh.
                if let Some(a) = self.heap.get_mut(handle).and_then(Cell::as_array_mut)
                    && i < a.len()
                {
                    a[i] = NanBox::hole();
                }
                if let Some(aux) = self.aux_props.get(&handle.to_raw()).copied()
                    && let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut)
                {
                    o.delete(root, key);
                }
                return true;
            }
            // A non-index named property on an array falls through to the aux path.
        }
        if self.heap.get(handle).and_then(Cell::as_object).is_some() {
            let o = self
                .heap
                .get_mut(handle)
                .and_then(Cell::as_object_mut)
                .expect("object cell");
            // Deleting a non-configurable property (a sealed/frozen object, or
            // one marked `configurable: false`) fails — but only if it exists;
            // deleting a missing property is a no-op that still "succeeds".
            if (o.is_sealed() || o.is_non_configurable(key)) && o.has_own_key(key) {
                return false;
            }
            o.delete(root, key);
            return true;
        }
        // A callable/array cell stores named props in its auxiliary object (e.g. a
        // built-in function's `name`/`length`); delete there so `delete fn.length`
        // actually removes the configurable own property.
        if let Some(aux) = self.aux_props.get(&handle.to_raw()).copied()
            && let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut)
        {
            if (o.is_sealed() || o.is_non_configurable(key)) && o.has_own_key(key) {
                return false;
            }
            o.delete(root, key);
        }
        // A non-object receiver with no aux: nothing to delete, which counts as
        // success.
        true
    }

    /// `Object.preventExtensions(obj)` — disallow new properties.
    pub fn prevent_extensions(&mut self, handle: Handle) {
        if self.heap.get(handle).and_then(Cell::as_array).is_some() {
            self.non_extensible_arrays.insert(handle.to_raw());
        } else if matches!(
            self.heap.get(handle),
            Some(Cell::TypedArray { .. } | Cell::RegExp { .. })
        ) {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.prevent_extensions();
            }
        } else if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.prevent_extensions();
        }
    }

    /// `Object.seal(obj)` — no new properties and no deletions.
    pub fn seal_object(&mut self, handle: Handle) {
        if self.heap.get(handle).and_then(Cell::as_array).is_some() {
            self.sealed_arrays.insert(handle.to_raw());
            self.non_extensible_arrays.insert(handle.to_raw());
        } else if matches!(
            self.heap.get(handle),
            Some(Cell::TypedArray { .. } | Cell::RegExp { .. })
        ) {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.seal();
            }
        } else if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.seal();
        }
    }

    /// Whether the object at `handle` is extensible. A plain array is extensible
    /// unless `preventExtensions`/`seal`/`freeze` marked it; functions/classes/native
    /// callables are extensible (properties may be attached to them).
    #[must_use]
    pub fn is_extensible(&self, handle: Handle) -> bool {
        if self.heap.get(handle).and_then(Cell::as_array).is_some() {
            return !self.non_extensible_arrays.contains(&handle.to_raw());
        }
        if let Some(o) = self.heap.get(handle).and_then(Cell::as_object) {
            return o.is_extensible();
        }
        // A typed array / RegExp / Date / Map / Set / Promise is extensible by
        // default; `preventExtensions`/`seal`/`freeze` records the flag on its
        // auxiliary object.
        if matches!(
            self.heap.get(handle),
            Some(
                Cell::TypedArray { .. }
                    | Cell::RegExp { .. }
                    | Cell::Date(_)
                    | Cell::Collection { .. }
                    | Cell::Promise(_)
            )
        ) {
            return self
                .aux_props
                .get(&handle.to_raw())
                .and_then(|a| self.heap.get(*a))
                .and_then(Cell::as_object)
                .is_none_or(Object::is_extensible);
        }
        matches!(
            self.heap.get(handle),
            Some(
                Cell::Function { .. }
                    | Cell::Class { .. }
                    | Cell::Native(_)
                    | Cell::HostFn(_)
                    | Cell::BoundNative { .. }
            )
        )
    }

    /// Whether the object at `handle` is sealed (or frozen).
    #[must_use]
    pub fn is_sealed(&self, handle: Handle) -> bool {
        if self.heap.get(handle).and_then(Cell::as_array).is_some() {
            return self.sealed_arrays.contains(&handle.to_raw());
        }
        self.heap
            .get(handle)
            .and_then(Cell::as_object)
            .is_some_and(Object::is_sealed)
    }

    /// Whether the cell at `handle` is a callable (any flavour of function): a
    /// user function/class, a native, or a bound native.
    #[must_use]
    pub fn is_callable_cell(&self, handle: Handle) -> bool {
        matches!(
            self.heap.get(handle),
            Some(
                Cell::Function { .. }
                    | Cell::Class { .. }
                    | Cell::Native(_)
                    | Cell::HostFn(_)
                    | Cell::BoundNative { .. }
            )
        )
    }

    /// The UTF-16 length of a String object — a `Cell::Str` primitive, or a
    /// String **wrapper** boxing one under the internal `\0prim` (PRIM_WRAP)
    /// slot — or `None` if `handle` is not a String object. Used to recognize a
    /// String exotic object's own `length` and index (`"0".."length-1"`)
    /// properties.
    fn string_object_len(&self, handle: Handle) -> Option<usize> {
        let sh = match self.get_property(handle, "\u{0}prim") {
            Some(prim) => Handle::from_raw(prim.as_handle()?),
            None => handle,
        };
        Some(crate::wtf8::utf16_len(&self.string_bytes(sh)?))
    }

    /// Whether the object at `handle` has an own property `key` (including
    /// accessors) — the `in` operator.
    #[must_use]
    pub fn has_own(&self, handle: Handle, key: &str) -> bool {
        // A **String exotic object** (a String primitive or wrapper) has own
        // `length` and index (`"0".."length-1"`) properties (StringGetOwnProperty),
        // so `"abc".hasOwnProperty(0)` / `0 in new String("abc")` are true. A
        // wrapper may also carry named own props, so fall through on a miss.
        if let Some(slen) = self.string_object_len(handle) {
            if key == "length" {
                return true;
            }
            if let Ok(i) = key.parse::<usize>()
                && alloc::format!("{i}") == key
                && i < slen
            {
                return true;
            }
        }
        if let Some(o) = self.heap.get(handle).and_then(Cell::as_object) {
            return o.contains(key) || o.accessor(key).is_some();
        }
        // An array's own properties are its in-range indices and `length` (plus any
        // aux-stored named property, checked below).
        if let Some(a) = self.heap.get(handle).and_then(Cell::as_array) {
            if key == "length" {
                return true;
            }
            if let Ok(i) = key.parse::<usize>()
                && i < a.len()
                && alloc::format!("{i}") == key
                && !a[i].is_hole()
            {
                // A present (non-hole) in-range index is an own property. A hole
                // falls through to the aux check below — `defineProperty(arr, i,
                // {get/set})` records an accessor there over an empty slot.
                return true;
            }
        }
        // A typed array's own properties are exactly its in-bounds canonical integer
        // indices (`"0".."length-1"`, ascending) — `length`/`buffer`/… are inherited
        // accessors, not own. A leading-zero / negative / fractional key is not own.
        if let Some(len) = self.typed_len(handle)
            && let Ok(i) = key.parse::<usize>()
            && i < len
            && alloc::format!("{i}") == key
        {
            return true;
        }
        // A non-object cell: check its auxiliary props (data or accessor).
        self.aux_props
            .get(&handle.to_raw())
            .and_then(|aux| self.heap.get(*aux))
            .and_then(Cell::as_object)
            .is_some_and(|o| o.contains(key) || o.accessor(key).is_some())
    }

    /// Defines an accessor (getter/setter) property on the object at `handle`.
    /// For a callable cell with no inline object part (a native, e.g. the
    /// `%TypedArray%` intrinsic), the accessor is recorded on its auxiliary
    /// object so a prototype-chain `get`/`set` still resolves it.
    pub fn define_accessor(&mut self, handle: Handle, key: &str, getter: NanBox, setter: NanBox) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.define_accessor(key, getter, setter);
            return;
        }
        let aux = self.aux_object(handle);
        if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
            o.define_accessor(key, getter, setter);
        }
    }

    /// Removes any accessor for `key` on `handle` (so a redefined data property
    /// takes precedence over a former getter/setter). For a callable/array cell with
    /// no inline object part, the accessor lives in the auxiliary object — clear it
    /// there too (e.g. converting an array index's getter/setter back to data).
    pub fn clear_accessor(&mut self, handle: Handle, key: &str) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.clear_accessor(key);
            return;
        }
        if let Some(aux) = self.aux_props.get(&handle.to_raw()).copied()
            && let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut)
        {
            o.clear_accessor(key);
        }
    }

    /// Removes the stored data slot for `key` on `handle` (without the
    /// non-configurable guard of [`delete_property`](Realm::delete_property)) — used
    /// when `defineProperty` converts an existing data property into an accessor and
    /// the old value must be discarded. Leaves any accessor side-list entry intact.
    pub fn delete_data_slot(&mut self, handle: Handle, key: &str) {
        let root = Rc::clone(&self.root_shape);
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.delete_data(root, key);
            return;
        }
        if let Some(aux) = self.aux_props.get(&handle.to_raw()).copied()
            && let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut)
        {
            o.delete_data(root, key);
        }
    }

    /// The `(getter, setter)` of accessor `key` on `handle`, if defined. Consults
    /// a callable cell's auxiliary object (see [`define_accessor`](Realm::define_accessor))
    /// so accessors installed on a native (e.g. `%TypedArray%`'s `get [Symbol.species]`)
    /// resolve.
    #[must_use]
    pub fn accessor(&self, handle: Handle, key: &str) -> Option<(NanBox, NanBox)> {
        if let Some(o) = self.heap.get(handle).and_then(Cell::as_object) {
            return o.accessor(key);
        }
        let aux = self.aux_props.get(&handle.to_raw())?;
        self.heap.get(*aux)?.as_object()?.accessor(key)
    }

    /// Sets own property `key` to `value` on the object at `handle`. Returns
    /// `false` if the handle is stale or the cell is not an object.
    pub fn set_property(&mut self, handle: Handle, key: &str, value: NanBox) -> bool {
        let dict_threshold = self.limits.object_dictionary_threshold;
        if let Some(obj) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            obj.maybe_convert_to_dict(key, dict_threshold);
            obj.set(key, value);
            self.write_barrier(handle, value);
            return true;
        }
        // Arrays, user functions, and native functions carry auxiliary named
        // properties (e.g. static methods on a built-in constructor); other
        // primitives (strings, numbers, …) reject property writes.
        let aux_eligible = self.aux_eligible(handle);
        if aux_eligible {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.maybe_convert_to_dict(key, dict_threshold);
                o.set(key, value);
            }
            self.write_barrier(aux, value);
            return true;
        }
        false
    }

    /// Like [`set_property`](Realm::set_property) but bypasses the extensibility /
    /// frozen / readonly guards (it always materializes the slot). For use by
    /// `[[DefineOwnProperty]]`, which performs its own validation and must be able
    /// to create or overwrite a data slot even on a non-extensible/frozen object
    /// (e.g. an accessor→data conversion of a configurable property).
    pub fn force_set_property(&mut self, handle: Handle, key: &str, value: NanBox) -> bool {
        let dict_threshold = self.limits.object_dictionary_threshold;
        if let Some(obj) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            obj.maybe_convert_to_dict(key, dict_threshold);
            obj.force_set(key, value);
            self.write_barrier(handle, value);
            return true;
        }
        if self.aux_eligible(handle) {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.maybe_convert_to_dict(key, dict_threshold);
                o.force_set(key, value);
            }
            self.write_barrier(aux, value);
            return true;
        }
        false
    }

    /// The object cell carrying `handle`'s named properties: the cell itself when
    /// it is a plain object, otherwise its auxiliary object (a native/function/
    /// array stores its named props — and their attribute flags — there). Returns
    /// `None` when no property-bearing cell exists yet.
    fn props_object(&self, handle: Handle) -> Option<&Object> {
        if let Some(o) = self.heap.get(handle).and_then(Cell::as_object) {
            return Some(o);
        }
        let aux = self.aux_props.get(&handle.to_raw())?;
        self.heap.get(*aux)?.as_object()
    }

    /// Whether the cell at `handle` keeps its named properties (and their
    /// attribute flags) in an auxiliary object — arrays, user functions, native
    /// functions, and bound natives (first-class prototype/static methods) all do.
    fn aux_eligible(&self, handle: Handle) -> bool {
        self.heap.get(handle).is_some_and(|c| {
            c.as_array().is_some()
                || c.as_function().is_some()
                || c.as_native().is_some()
                || c.as_bound_native().is_some()
                || c.as_class().is_some()
                // A registered host function carries auxiliary named properties
                // too (its `prototype` when `register_constructor`ed, and any own
                // props an embedder sets on it).
                || matches!(c, Cell::HostFn(_))
                || matches!(c, Cell::TypedArray { .. })
                // A RegExp instance carries no inline object part, but a script may
                // set own properties on it (`re.exec = fn`, a custom `lastIndex`
                // descriptor) — and the Symbol.* methods read a user `exec`
                // override — so it stores named props in an auxiliary object.
                || matches!(c, Cell::RegExp { .. })
                // Date / Map / Set / Promise instances are ordinary (extensible)
                // objects per spec: a script may set arbitrary own properties on
                // them (`d.value = 1`), stored in an auxiliary object.
                || matches!(c, Cell::Date(_) | Cell::Collection { .. } | Cell::Promise(_))
        })
    }

    /// Marks own property `key` of the object at `handle` non-writable
    /// (`defineProperty` with `writable: false`). For a callable cell with no
    /// inline object part (a native/function), the flag is recorded on its
    /// auxiliary object so the property descriptor reports it.
    pub fn set_readonly_property(&mut self, handle: Handle, key: &str) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.set_readonly(key);
            return;
        }
        if self.aux_eligible(handle) {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.set_readonly(key);
            }
        }
    }

    /// Clears the non-writable mark for `key` (used when `defineProperty`
    /// redefines a configurable property's attributes).
    pub fn clear_readonly_property(&mut self, handle: Handle, key: &str) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.clear_readonly(key);
            return;
        }
        if let Some(aux) = self.aux_props.get(&handle.to_raw()).copied()
            && let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut)
        {
            o.clear_readonly(key);
        }
    }

    /// Clears the non-configurable mark for `key` (used when `defineProperty`
    /// redefines a configurable property that stays configurable).
    pub fn clear_non_configurable_property(&mut self, handle: Handle, key: &str) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.clear_non_configurable(key);
            return;
        }
        if let Some(aux) = self.aux_props.get(&handle.to_raw()).copied()
            && let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut)
        {
            o.clear_non_configurable(key);
        }
    }

    /// Clears the non-enumerable mark for `key` (used when `defineProperty`
    /// redefines a configurable property to be enumerable).
    pub fn clear_hidden_property(&mut self, handle: Handle, key: &str) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.clear_hidden(key);
            return;
        }
        if let Some(aux) = self.aux_props.get(&handle.to_raw()).copied()
            && let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut)
        {
            o.clear_hidden(key);
        }
    }

    /// Marks own property `key` non-configurable (it cannot be deleted). Recorded
    /// on the auxiliary object for a callable cell with no inline object part.
    pub fn set_non_configurable_property(&mut self, handle: Handle, key: &str) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.set_non_configurable(key);
            return;
        }
        if self.aux_eligible(handle) {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.set_non_configurable(key);
            }
        }
    }

    /// Whether own property `key` is non-writable (frozen or read-only).
    #[must_use]
    pub fn property_is_readonly(&self, handle: Handle, key: &str) -> bool {
        self.props_object(handle)
            .is_some_and(|o| o.is_frozen() || o.is_readonly(key))
    }

    /// Whether own property `key` is non-configurable (frozen/sealed object, or
    /// marked `configurable: false`).
    #[must_use]
    pub fn property_is_non_configurable(&self, handle: Handle, key: &str) -> bool {
        let raw = handle.to_raw();
        // A frozen *or sealed* array makes every own index non-configurable (the
        // flag lives in the side registries, not the aux object). `length` is
        // always non-configurable for an array regardless.
        if self.frozen_arrays.contains(&raw) || self.sealed_arrays.contains(&raw) {
            return true;
        }
        self.props_object(handle)
            .is_some_and(|o| o.is_sealed() || o.is_non_configurable(key))
    }

    /// Whether own property `key` is enumerable (not marked hidden).
    #[must_use]
    pub fn property_is_enumerable(&self, handle: Handle, key: &str) -> bool {
        // A String exotic object's index properties (`"0".."length-1"`) are
        // enumerable, writable:false, configurable:false; `length` is
        // non-enumerable. Checked before the generic object branch (a String
        // wrapper is a `Cell::Object`).
        if let Some(slen) = self.string_object_len(handle) {
            if key == "length" {
                return false;
            }
            if let Ok(i) = key.parse::<usize>()
                && alloc::format!("{i}") == key
                && i < slen
            {
                return true;
            }
        }
        if let Some(o) = self.heap.get(handle).and_then(Cell::as_object) {
            return !o.is_hidden(key);
        }
        // A typed-array in-range canonical integer index is an enumerable own data
        // property (ES2021+: enumerable, writable, configurable).
        if let Some((_, _, len, _)) = self.heap.get(handle).and_then(Cell::as_typed_array)
            && let Ok(i) = key.parse::<usize>()
            && i < len
            && alloc::format!("{i}") == key
        {
            return true;
        }
        // An array's in-range indices are enumerable by default; `length` is not.
        // A per-index `enumerable: false` set via `Object.defineProperty` is
        // recorded as a hidden flag in the auxiliary object, so consult that first.
        if let Some(a) = self.heap.get(handle).and_then(Cell::as_array) {
            if key == "length" {
                return false;
            }
            if let Ok(i) = key.parse::<usize>()
                && i < a.len()
                && alloc::format!("{i}") == key
            {
                let hidden = self
                    .aux_props
                    .get(&handle.to_raw())
                    .and_then(|aux| self.heap.get(*aux))
                    .and_then(Cell::as_object)
                    .is_some_and(|o| o.is_hidden(key));
                return !hidden;
            }
        }
        // A named aux property (e.g. a custom property on an array/function) follows
        // its stored hidden flag. The property may be a data slot *or* an accessor
        // (a data property converted to a getter/setter keeps its enumerable flag),
        // so consult both — `contains` only sees data slots.
        self.aux_props
            .get(&handle.to_raw())
            .and_then(|aux| self.heap.get(*aux))
            .and_then(Cell::as_object)
            .is_some_and(|o| (o.contains(key) || o.accessor(key).is_some()) && !o.is_hidden(key))
    }

    /// Marks own property `key` non-enumerable (without changing its value).
    pub fn mark_hidden(&mut self, handle: Handle, key: &str) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.set_hidden(key);
            return;
        }
        // Arrays/functions/natives keep named properties — and their non-enumerable
        // flags — in their auxiliary object (e.g. a native's `name`/`length`).
        let aux_eligible = self.aux_eligible(handle);
        if aux_eligible {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.set_hidden(key);
            }
        }
    }

    /// Sets own property `key` to `value` but marks it **non-enumerable** — used
    /// for class methods, which are callable but must stay out of `Object.keys`,
    /// spread, `for-in`, and JSON.
    pub fn set_hidden_property(&mut self, handle: Handle, key: &str, value: NanBox) -> bool {
        let dict_threshold = self.limits.object_dictionary_threshold;
        if let Some(obj) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            obj.maybe_convert_to_dict(key, dict_threshold);
            obj.set(key, value);
            obj.set_hidden(key);
            self.write_barrier(handle, value);
            return true;
        }
        // Arrays/functions/natives carry hidden slots in their auxiliary object
        // (e.g. a VM closure's function marker), kept non-enumerable there too.
        let aux_eligible = self.aux_eligible(handle);
        if aux_eligible {
            let aux = self.aux_object(handle);
            if let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) {
                o.maybe_convert_to_dict(key, dict_threshold);
                o.set(key, value);
                o.set_hidden(key);
            }
            self.write_barrier(aux, value);
            return true;
        }
        false
    }

    /// The generational write barrier: records an old→young edge (container is
    /// in the old generation, `value` is a young heap object) in the heap's
    /// remembered set, so a minor collection keeps the young object alive.
    fn write_barrier(&mut self, container: Handle, value: NanBox) {
        if let Some(raw) = value.as_handle() {
            let target = Handle::from_raw(raw);
            self.heap.record_edge(container, target, gc::OLD_AGE);
            // Incremental (Dijkstra) barrier: shade a reference stored during a
            // marking cycle so it cannot be missed.
            if let Some(marker) = self.incremental.as_mut() {
                marker.mark_grey(target);
            }
        }
    }

    /// Interns `key`, so callers can hold a `Copy` [`Atom`] for hot property
    /// names.
    pub fn intern(&mut self, key: &str) -> Atom {
        self.atoms.intern(key)
    }

    /// The number of live objects in the heap.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.heap.len()
    }

    /// Whether `handle` still refers to a live object.
    #[must_use]
    pub fn is_live(&self, handle: Handle) -> bool {
        self.heap.is_live(handle)
    }

    /// All value handles held by the realm's value-reachable-only side-tables —
    /// auxiliary property objects (`aux_props`), lazily-created function
    /// prototypes/constructors (`fn_protos`/`fn_ctor`), and interned symbol objects
    /// (`symbols_by_id`). These are reachable only through the tables, so a minor
    /// collection (which does **not** prune the weak-key tables — see
    /// [`collect`](Self::collect)) treats every value as a root to avoid freeing
    /// state the tables still point at.
    fn gc_side_table_values(&self) -> alloc::vec::Vec<Handle> {
        let mut r = alloc::vec::Vec::new();
        r.extend(self.aux_props.values().copied());
        r.extend(self.fn_protos.values().copied());
        r.extend(self.fn_ctor.values().copied());
        r.extend(self.intl_protos.values().copied());
        r.extend(self.symbols_by_id.values().copied());
        // Host persistent handles keep their (object) values alive across calls.
        r.extend(
            self.host_persistent
                .iter()
                .flatten()
                .filter_map(|nb| nb.as_handle().map(Handle::from_raw)),
        );
        r
    }

    /// Pins `value` as a **persistent handle** and returns its slot index (see
    /// [`host_persistent`](Self::host_persistent)). The value survives GC and stays
    /// valid across compaction until [`release_persistent`](Self::release_persistent).
    /// A freed slot is reused before growing the table.
    pub fn persist(&mut self, value: NanBox) -> u32 {
        if let Some(i) = self.host_persistent.iter().position(Option::is_none) {
            self.host_persistent[i] = Some(value);
            i as u32
        } else {
            self.host_persistent.push(Some(value));
            (self.host_persistent.len() - 1) as u32
        }
    }

    /// Reads the current value of persistent handle `idx` (`None` if the slot was
    /// released or never allocated). The returned `NanBox` reflects any relocation
    /// the collector has applied.
    #[must_use]
    pub fn persistent(&self, idx: u32) -> Option<NanBox> {
        self.host_persistent.get(idx as usize).copied().flatten()
    }

    /// Releases persistent handle `idx`, so its value is no longer a GC root and the
    /// slot can be reused. A double release / unknown index is a harmless no-op.
    pub fn release_persistent(&mut self, idx: u32) {
        if let Some(slot) = self.host_persistent.get_mut(idx as usize) {
            *slot = None;
        }
    }

    /// Pushes, into `extra`, the value handles of every side-table entry whose
    /// **key is live** (reachable from the real roots) — the ephemeron rule a
    /// weak-key map needs: an entry's value is a root only while its key is alive.
    /// Handed the heap so it can find symbols still referenced as object property
    /// keys (`\0sym:{id}`), which are *not* otherwise reachable through the graph.
    /// Never pushes keys, so a dead key stays collectable.
    #[allow(clippy::too_many_arguments)] // one ref per side-table, to fit split borrows
    fn ephemeron_expand(
        heap: &Heap<Cell>,
        marked: &alloc::collections::BTreeSet<Handle>,
        aux_props: &alloc::collections::BTreeMap<u64, Handle>,
        fn_protos: &alloc::collections::BTreeMap<u32, Handle>,
        fn_ctor: &alloc::collections::BTreeMap<u32, Handle>,
        symbols_by_id: &alloc::collections::BTreeMap<u64, Handle>,
        extra: &mut Vec<Handle>,
    ) {
        // `aux_props`: keyed by the owning cell's handle. If that cell is live,
        // its aux object (and so its named properties) must survive.
        for (cell_raw, obj) in aux_props {
            if marked.contains(&Handle::from_raw(*cell_raw)) {
                extra.push(*obj);
            }
        }
        // `fn_ctor`/`fn_protos`: keyed by function id. The id is "live" iff the
        // most-recent function handle for it is live; then both its constructor
        // back-reference and its lazily-built `.prototype` must survive.
        for (id, fh) in fn_ctor {
            if marked.contains(fh) {
                extra.push(*fh);
                if let Some(proto) = fn_protos.get(id) {
                    extra.push(*proto);
                }
            }
        }
        // `symbols_by_id`: a symbol is live if its handle is reachable directly,
        // *or* if a live object still uses it as a property key (stored as the
        // string `\0sym:{id}`, which carries no handle edge). Scan the live
        // objects' keys for referenced ids and root those symbols.
        for h in marked {
            if let Some(obj) = heap.get(*h).and_then(Cell::as_object) {
                for k in obj.all_keys() {
                    if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                        && let Ok(id) = idstr.parse::<u64>()
                        && let Some(sym) = symbols_by_id.get(&id)
                    {
                        extra.push(*sym);
                    }
                }
            }
        }
    }

    /// Drops every side-table entry whose key is **not** marked-live (a dead key
    /// must not be kept alive by its own entry). `key_collectable(handle)` decides
    /// whether a given key handle is eligible for pruning this cycle — always
    /// `true` for a full collection; restricted to young-generation keys for a
    /// minor collection (an old key was not scanned, so unmarked ≠ dead). Called
    /// after the ephemeron fixpoint, so any value still kept alive by a live key
    /// is already marked.
    #[allow(clippy::too_many_arguments)] // one ref per side-table, to fit split borrows
    fn ephemeron_prune(
        marked: &alloc::collections::BTreeSet<Handle>,
        key_collectable: &dyn Fn(Handle) -> bool,
        aux_props: &mut alloc::collections::BTreeMap<u64, Handle>,
        fn_protos: &mut alloc::collections::BTreeMap<u32, Handle>,
        fn_ctor: &mut alloc::collections::BTreeMap<u32, Handle>,
        symbols_by_id: &mut alloc::collections::BTreeMap<u64, Handle>,
        frozen_arrays: &mut alloc::collections::BTreeSet<u64>,
        sealed_arrays: &mut alloc::collections::BTreeSet<u64>,
        non_extensible_arrays: &mut alloc::collections::BTreeSet<u64>,
    ) {
        let dead = |raw: u64| {
            let h = Handle::from_raw(raw);
            key_collectable(h) && !marked.contains(&h)
        };
        aux_props.retain(|cell_raw, _| !dead(*cell_raw));
        frozen_arrays.retain(|raw| !dead(*raw));
        sealed_arrays.retain(|raw| !dead(*raw));
        non_extensible_arrays.retain(|raw| !dead(*raw));
        // `fn_ctor`/`fn_protos` are id-keyed; the function handle is the key's
        // identity. Drop both when the function handle is collectable-and-dead.
        let dead_ids: alloc::vec::Vec<u32> = fn_ctor
            .iter()
            .filter(|(_, fh)| key_collectable(**fh) && !marked.contains(*fh))
            .map(|(id, _)| *id)
            .collect();
        for id in dead_ids {
            fn_ctor.remove(&id);
            fn_protos.remove(&id);
        }
        // `symbols_by_id` is keyed by the symbol's own handle; prune those whose
        // handle is collectable-and-dead (the expand pass has already kept any
        // symbol still referenced as a live object's property key).
        symbols_by_id.retain(|_, sh| !key_collectable(*sh) || marked.contains(sh));
    }

    /// Runs a full (**major**) garbage collection, keeping everything reachable
    /// from `roots` and freeing the rest (including cycles). Survivors are
    /// promoted toward the old generation. Returns the collection statistics.
    ///
    /// The realm's value-reachable side-tables (`aux_props`, `fn_protos`,
    /// `fn_ctor`, `symbols_by_id`, and the array-integrity flag sets) are treated
    /// as **weak-key (ephemeron)** maps: an entry survives iff its key is reachable
    /// from the real roots; a surviving entry's value handles are then kept alive;
    /// an entry whose key has become unreachable is dropped (so it no longer pins
    /// its dead key's handle — fixing the unbounded side-table leak). See
    /// [`gc::mark_with_ephemerons`] and [`gc::sweep_full`].
    pub fn collect(&mut self, roots: &[Handle]) -> Stats {
        let Self {
            heap,
            aux_props,
            fn_protos,
            fn_ctor,
            symbols_by_id,
            frozen_arrays,
            sealed_arrays,
            non_extensible_arrays,
            ..
        } = self;
        // Phase 1 — mark from the real roots, expanding the weak-key side-tables
        // to a fixpoint (an entry's value is a root only while its key is live).
        let marked =
            gc::mark_with_ephemerons(heap, roots.iter().copied(), |heap, marked, extra| {
                Self::ephemeron_expand(
                    heap,
                    marked,
                    aux_props,
                    fn_protos,
                    fn_ctor,
                    symbols_by_id,
                    extra,
                );
            });
        // Phase 2 — drop every entry whose key is unmarked (full collection: an
        // unmarked key is genuinely dead, so no generation guard is needed).
        Self::ephemeron_prune(
            &marked,
            &|_| true,
            aux_props,
            fn_protos,
            fn_ctor,
            symbols_by_id,
            frozen_arrays,
            sealed_arrays,
            non_extensible_arrays,
        );
        // Phase 3 — sweep the unmarked objects and promote the survivors.
        gc::sweep_full(heap, &marked)
    }

    /// Runs a **minor** (generational) collection — reclaims only short-lived
    /// objects in the young generation, treating the old generation as roots.
    /// Cheap when most allocation is short-lived. Returns the statistics.
    ///
    /// A minor cycle scans only the young generation, so an old-generation
    /// side-table key being unmarked does **not** mean it is dead. To stay sound
    /// the minor path does not prune the weak-key tables (full
    /// [`collect`](Self::collect) reclaims dead entries); it simply roots every
    /// side-table value so nothing the tables point at is freed mid-cycle.
    pub fn collect_minor(&mut self, roots: &[Handle]) -> Stats {
        let mut all = roots.to_vec();
        all.extend(self.gc_side_table_values());
        gc::collect_minor(&mut self.heap, &all)
    }

    /// Test-only: the live entry counts of the value-reachable side-tables, used
    /// to assert the weak-key pruning shrinks them back after a collection.
    #[cfg(test)]
    fn side_table_lens(&self) -> [usize; 7] {
        [
            self.aux_props.len(),
            self.fn_protos.len(),
            self.fn_ctor.len(),
            self.symbols_by_id.len(),
            self.frozen_arrays.len(),
            self.sealed_arrays.len(),
            self.non_extensible_arrays.len(),
        ]
    }

    /// Runs a **moving (compacting)** collection: keeps everything reachable from
    /// `roots`, relocates the survivors to the front of the heap (defragmenting
    /// the slot table), and rewrites every reference — including the caller's
    /// `roots`, updated in place — to the new locations. Returns the statistics.
    pub fn compact(&mut self, roots: &mut [Handle]) -> Stats {
        let n = roots.len();

        // The side-tables are weak-key (ephemeron) maps. Run the three phases
        // sequentially so each borrows the tables at a disjoint time:
        //   1. mark from the real roots + ephemeron expansion (shared borrow);
        //   2. prune dead-key entries (mutable borrow);
        //   3. compact/relocate, marking the surviving values as extra roots and
        //      forwarding every table key/value to its new handle.
        {
            let Self {
                heap,
                aux_props,
                fn_protos,
                fn_ctor,
                symbols_by_id,
                frozen_arrays,
                sealed_arrays,
                non_extensible_arrays,
                ..
            } = self;
            let marked =
                gc::mark_with_ephemerons(heap, roots.iter().copied(), |heap, marked, extra| {
                    Self::ephemeron_expand(
                        heap,
                        marked,
                        aux_props,
                        fn_protos,
                        fn_ctor,
                        symbols_by_id,
                        extra,
                    );
                });
            Self::ephemeron_prune(
                &marked,
                &|_| true,
                aux_props,
                fn_protos,
                fn_ctor,
                symbols_by_id,
                frozen_arrays,
                sealed_arrays,
                non_extensible_arrays,
            );
        }

        // The surviving entries' value handles are reachable only through the
        // tables, so they must be extra roots for the moving collector (which
        // re-marks from scratch); after pruning, every remaining key is live, so
        // this reproduces the same marked set.
        let extra = self.gc_side_table_values();
        let mut all: alloc::vec::Vec<Handle> = roots.iter().copied().chain(extra).collect();

        // Split the borrow so the moving collector (which takes `&mut heap`) can hand the
        // forwarding function to a closure that repairs the realm's out-of-heap handle tables.
        let Self {
            heap,
            frozen_arrays,
            sealed_arrays,
            non_extensible_arrays,
            aux_props,
            fn_protos,
            fn_ctor,
            symbols_by_id,
            intl_protos,
            host_persistent,
            ..
        } = self;
        let stats = gc::compact_with(heap, &mut all, &mut |forward| {
            // A typed array's backing-buffer handle lives in the `Cell::TypedArray`
            // itself and is forwarded by the moving collector with every other cell
            // reference, so the view→buffer link survives compaction intrinsically —
            // no side registry to relocate.
            // The array integrity flag sets are keyed by the array's (relocated) handle.
            for set in [
                &mut *frozen_arrays,
                &mut *sealed_arrays,
                &mut *non_extensible_arrays,
            ] {
                let old_set = core::mem::take(set);
                for raw in old_set {
                    set.insert(forward(Handle::from_raw(raw)).to_raw());
                }
            }
            // `aux_props` is handle→handle (owning cell → aux object); forward both.
            let old_aux = core::mem::take(aux_props);
            for (cell_raw, obj) in old_aux {
                aux_props.insert(forward(Handle::from_raw(cell_raw)).to_raw(), forward(obj));
            }
            // `fn_protos`/`fn_ctor`/`symbols_by_id` are id→handle: keys are stable ids, only
            // the (relocated) value handles need forwarding.
            for v in fn_protos.values_mut() {
                *v = forward(*v);
            }
            for v in fn_ctor.values_mut() {
                *v = forward(*v);
            }
            // `intl_protos` is id→handle (constructor native id → its `.prototype`):
            // the keys are stable ids, only the relocated value handles need forwarding.
            for v in intl_protos.values_mut() {
                *v = forward(*v);
            }
            for v in symbols_by_id.values_mut() {
                *v = forward(*v);
            }
            // Host persistent handles: forward each pinned *object* value so the
            // slot the embedder reads through stays valid after relocation.
            for slot in host_persistent.iter_mut().flatten() {
                if let Some(raw) = slot.as_handle() {
                    *slot = NanBox::handle(forward(Handle::from_raw(raw)).to_raw());
                }
            }
        });
        roots.copy_from_slice(&all[..n]);
        stats
    }

    /// Runs an **incremental** collection: marks in step-bounded slices of at
    /// most `step_budget` objects each (rather than one stop-the-world pass),
    /// then sweeps. Equivalent result to [`collect`](Realm::collect); the
    /// step-based [`IncrementalMarker`](gc::IncrementalMarker) is what lets the
    /// pause be bounded / interleaved with execution.
    pub fn collect_incremental(&mut self, roots: &[Handle], step_budget: usize) -> Stats {
        let before = self.heap.len();
        let mut marker = gc::IncrementalMarker::new(roots);
        while !marker.step(&self.heap, step_budget.max(1)) {}
        let swept = marker.sweep(&mut self.heap);
        Stats {
            marked: before - swept,
            swept,
        }
    }

    /// Begins an **interleaved** incremental collection: installs a marker
    /// seeded from `roots` whose write barrier is now active, so mutation
    /// (`set_property`/`set_element`/…) between steps stays sound. Pair with
    /// [`incremental_step`](Realm::incremental_step) and
    /// [`incremental_finish`](Realm::incremental_finish).
    pub fn incremental_start(&mut self, roots: &[Handle]) {
        self.incremental = Some(gc::IncrementalMarker::new(roots));
    }

    /// Advances the active interleaved marker by up to `step_budget` objects.
    /// Returns `true` when marking is complete. A no-op (`true`) if no cycle is
    /// active.
    pub fn incremental_step(&mut self, step_budget: usize) -> bool {
        // Split the borrow: the marker scans the heap immutably.
        let Some(mut marker) = self.incremental.take() else {
            return true;
        };
        let done = marker.step(&self.heap, step_budget.max(1));
        self.incremental = Some(marker);
        done
    }

    /// Finishes the interleaved cycle: sweeps everything marking did not reach,
    /// clears the active marker, and returns the statistics. A no-op (empty
    /// stats) if no cycle is active.
    pub fn incremental_finish(&mut self) -> Stats {
        let Some(marker) = self.incremental.take() else {
            return Stats::default();
        };
        let before = self.heap.len();
        let swept = marker.sweep(&mut self.heap);
        Stats {
            marked: before - swept,
            swept,
        }
    }

    // --- value operations (the VM's `+`, `ToString`, `===` over heap values) ---

    /// Whether `v` is a heap string.
    #[must_use]
    fn is_string(&self, v: NanBox) -> bool {
        v.as_handle()
            .and_then(|raw| self.heap.get(Handle::from_raw(raw)))
            .is_some_and(|c| c.as_str().is_some())
    }

    /// The rope view of `v` for concatenation: a string cell's own rope (shared,
    /// so concatenation stays O(1)), or a fresh leaf from its `ToString`.
    fn rope_of(&self, v: NanBox) -> Rope {
        if let Some(raw) = v.as_handle()
            && let Some(rope) = self.heap.get(Handle::from_raw(raw)).and_then(Cell::as_str)
        {
            return rope.clone();
        }
        Rope::from(self.to_display_string(v).as_str())
    }

    /// ECMAScript `ToString` for display: numbers/booleans/null/undefined render
    /// directly; a string yields its text; an array joins its elements with
    /// `","`; a plain object is `"[object Object]"`.
    #[must_use]
    pub fn to_display_string(&self, v: NanBox) -> alloc::string::String {
        self.to_display_string_seen(v, &mut Vec::new())
    }

    /// `to_display_string` tracking the array/proxy handles currently being
    /// rendered, so a self-referential array (`a.push(a); a.toString()`) renders
    /// the recursive element as empty — per `Array.prototype.join`'s cycle rule —
    /// instead of overflowing the stack.
    pub(crate) fn to_display_string_seen(
        &self,
        v: NanBox,
        seen: &mut Vec<Handle>,
    ) -> alloc::string::String {
        use crate::nanbox::Unpacked;
        match v.unpack() {
            Unpacked::Undefined => "undefined".into(),
            Unpacked::Null => "null".into(),
            Unpacked::Bool(b) => {
                if b {
                    "true".into()
                } else {
                    "false".into()
                }
            }
            Unpacked::Number(n) => js_number_string(n),
            Unpacked::Handle(raw) => match self.heap.get(Handle::from_raw(raw)) {
                Some(Cell::Str(r)) => r.materialize(),
                Some(Cell::Array(elems)) => {
                    let h = Handle::from_raw(raw);
                    // A circular reference back to this array renders empty; so
                    // does nesting past the depth cap, so a deep acyclic array
                    // cannot overflow the host stack here.
                    if seen.contains(&h) || seen.len() >= self.limits.max_display_depth {
                        return alloc::string::String::new();
                    }
                    seen.push(h);
                    let elems = elems.clone();
                    let parts: Vec<alloc::string::String> = elems
                        .iter()
                        .map(|e| {
                            // A hole-ish nullish element renders empty, per `Array#join`.
                            if matches!(e.unpack(), Unpacked::Undefined | Unpacked::Null) {
                                alloc::string::String::new()
                            } else {
                                self.to_display_string_seen(*e, seen)
                            }
                        })
                        .collect();
                    seen.pop();
                    parts.join(",")
                }
                Some(Cell::Object(_)) => "[object Object]".into(),
                // A callable stringifies as `Function.prototype.toString` would —
                // `function name() { [native code] }` (the engine retains no source)
                // — and a class as `class Name { }`. The name is read from the own
                // `name` property when materialized (else empty). This keeps `"" + fn`
                // / `String(fn)` consistent with `fn.toString()`.
                Some(Cell::Function { .. } | Cell::Native(_) | Cell::HostFn(_)) => {
                    let h = Handle::from_raw(raw);
                    let name = self
                        .get_property(h, "name")
                        .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                        .map(|v| self.to_display_string_seen(v, seen))
                        .unwrap_or_default();
                    alloc::format!("function {name}() {{ [native code] }}")
                }
                Some(Cell::Class { .. }) => {
                    let h = Handle::from_raw(raw);
                    let name = self
                        .get_property(h, "name")
                        .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                        .map(|v| self.to_display_string_seen(v, seen))
                        .unwrap_or_default();
                    alloc::format!("class {name} {{ }}")
                }
                Some(Cell::Collection { is_set, .. }) => {
                    if *is_set {
                        "[object Set]".into()
                    } else {
                        "[object Map]".into()
                    }
                }
                Some(Cell::BoundNative { .. }) => {
                    let h = Handle::from_raw(raw);
                    let name = self
                        .get_property(h, "name")
                        .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                        .map(|v| self.to_display_string_seen(v, seen))
                        .unwrap_or_default();
                    alloc::format!("function {name}() {{ [native code] }}")
                }
                Some(Cell::Promise(_)) => "[object Promise]".into(),
                Some(Cell::Date(ms)) => {
                    if ms.is_finite() {
                        date_to_iso(*ms)
                    } else {
                        alloc::string::String::from("Invalid Date")
                    }
                }
                Some(Cell::RegExp { source, flags, .. }) => alloc::format!("/{source}/{flags}"),
                Some(Cell::Symbol { description, .. }) => {
                    // A no-argument `Symbol()` carries a `\0`-sentinel description
                    // (an undefined `.description`); render it as `Symbol()`.
                    if description.starts_with('\u{0}') {
                        alloc::string::String::from("Symbol()")
                    } else {
                        alloc::format!("Symbol({description})")
                    }
                }
                Some(Cell::BigInt(n)) => alloc::format!("{n}"),
                // A typed-array view stringifies as its comma-joined decoded
                // elements (`Array#join` semantics over the shared bytes); the raw
                // byte backing is internal and never directly displayed. Decode
                // each element straight to its string (no heap allocation), so this
                // stays a `&self` method even for the BigInt element kinds.
                Some(Cell::TypedArray {
                    buffer,
                    byte_offset,
                    length,
                    kind,
                    ..
                }) => {
                    let (buffer, byte_offset, length, kind) =
                        (*buffer, *byte_offset, *length, *kind);
                    let size = typed_elem_size(kind);
                    let bytes = self.bytes_at(buffer).unwrap_or(&[]);
                    let parts: Vec<alloc::string::String> = (0..length)
                        .map(|i| {
                            let start = byte_offset + i * size;
                            let slice = bytes.get(start..start + size).unwrap_or(&[]);
                            if is_bigint_kind(kind) {
                                alloc::format!("{}", decode_bigint_element(kind, slice))
                            } else {
                                js_number_string(decode_typed_element(kind, slice))
                            }
                        })
                        .collect();
                    parts.join(",")
                }
                Some(Cell::Bytes(_)) => alloc::string::String::new(),
                // A proxy renders as its target would.
                Some(Cell::Proxy { target, .. }) => {
                    let h = Handle::from_raw(raw);
                    if seen.contains(&h) || seen.len() >= self.limits.max_display_depth {
                        return alloc::string::String::new();
                    }
                    seen.push(h);
                    let s = self.to_display_string_seen(NanBox::handle(target.to_raw()), seen);
                    seen.pop();
                    s
                }
                None => "undefined".into(), // stale handle
            },
        }
    }

    /// The ECMAScript `+` operator (the cases this model covers): number + number
    /// is numeric addition; if either side is a string it is string
    /// concatenation, producing a new heap string built by joining ropes (so a
    /// loop of `+` stays O(1) per step). Other combinations coerce to string for
    /// now (full `ToPrimitive` arrives with the boxed primitives).
    pub fn add(&mut self, a: NanBox, b: NanBox) -> NanBox {
        // The non-throwing wrapper (used by callers without an error channel):
        // an over-length string concatenation degrades to the spec error value's
        // text rather than corrupting a length or OOM-ing. Callers that can throw
        // use [`Realm::add_checked`] to surface the proper `RangeError`.
        self.add_checked(a, b).unwrap_or_else(|| {
            let handle = self
                .heap
                .alloc(Cell::Str(Rope::from("Invalid string length")));
            NanBox::handle(handle.to_raw())
        })
    }

    /// ECMAScript `+`, returning `None` when a string concatenation would
    /// exceed the maximum representable string length (so the caller throws a
    /// `RangeError`) instead of overflowing the cached length / OOM-ing.
    ///
    /// `None` is returned when the concatenated string would exceed
    /// [`crate::rope::MAX_STRING_LEN`].
    pub fn add_checked(&mut self, a: NanBox, b: NanBox) -> Option<NanBox> {
        if let (Some(x), Some(y)) = (a.as_number(), b.as_number()) {
            return Some(NanBox::number(x + y));
        }
        // A string operand keeps the O(1) rope concatenation path.
        if self.is_string(a) || self.is_string(b) {
            let combined = self.rope_of(a).try_concat(&self.rope_of(b))?;
            let handle = self.heap.alloc(Cell::Str(combined));
            return Some(NanBox::handle(handle.to_raw()));
        }
        // Any other heap value (array, object): `+` is string concatenation
        // after `ToPrimitive` — our arrays/objects stringify (`[1,2] + [3,4]`
        // → "1,23,4", `{} + "!"` → "[object Object]!").
        if a.as_handle().is_some() || b.as_handle().is_some() {
            let left = self.to_display_string(a);
            let right = self.to_display_string(b);
            if left
                .len()
                .checked_add(right.len())
                .is_none_or(|n| n > crate::rope::MAX_STRING_LEN)
            {
                return None;
            }
            let mut combined = left;
            combined.push_str(&right);
            let handle = self.heap.alloc(Cell::Str(Rope::from(combined.as_str())));
            return Some(NanBox::handle(handle.to_raw()));
        }
        // Primitives only (bool/null/undefined): numeric.
        Some(NanBox::number(self.to_number(a) + self.to_number(b)))
    }

    /// ECMAScript `ToNumber` (the cases this model covers): numbers pass
    /// through; `true`/`false` → `1`/`0`; `null` → `0`; `undefined` → `NaN`; a
    /// string is parsed (blank → `0`, a numeric literal → its value, else
    /// `NaN`); objects/arrays → `NaN` (full `ToPrimitive` arrives later).
    #[must_use]
    pub fn to_number(&self, v: NanBox) -> f64 {
        use crate::nanbox::Unpacked;
        match v.unpack() {
            Unpacked::Number(n) => n,
            Unpacked::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            Unpacked::Null => 0.0,
            Unpacked::Undefined => f64::NAN,
            Unpacked::Handle(raw) => {
                match self.heap.get(Handle::from_raw(raw)).and_then(Cell::as_str) {
                    Some(rope) => {
                        let s = rope.materialize();
                        let t = s.trim();
                        if t.is_empty() {
                            return 0.0;
                        }
                        // `0x`/`0o`/`0b`-prefixed integer strings parse by radix.
                        let radixed = match t.get(0..2) {
                            Some("0x" | "0X") => Some((16, &t[2..])),
                            Some("0o" | "0O") => Some((8, &t[2..])),
                            Some("0b" | "0B") => Some((2, &t[2..])),
                            _ => None,
                        };
                        if let Some((radix, body)) = radixed {
                            return i64::from_str_radix(body, radix).map_or(f64::NAN, |n| n as f64);
                        }
                        js_str_to_f64(t)
                    }
                    // A `Date` coerces to its millisecond timestamp (so `b - a`
                    // yields an elapsed-ms difference); any other object coerces
                    // via ToPrimitive (its `toString`/`valueOf`) then ToNumber —
                    // so `[5] - 2` is `3` and `{} * 1` is `NaN`.
                    None => match self.date_at(Handle::from_raw(raw)) {
                        Some(ms) => ms,
                        None => self.number_from_str(&self.to_display_string(v)),
                    },
                }
            }
        }
    }

    /// The ECMAScript abstract relational comparison `a < b`: if *both* operands
    /// are strings they compare lexicographically by code point; otherwise both
    /// are coerced with `ToNumber`. Returns `None` when the result is undefined
    /// (a `NaN` operand) — the caller turns that into `false`.
    #[must_use]
    /// Parses a string to a number with the same rules as `ToNumber` over a
    /// string (radix prefixes, trimming, empty → 0).
    fn number_from_str(&self, s: &str) -> f64 {
        let t = s.trim();
        if t.is_empty() {
            return 0.0;
        }
        let radixed = match t.get(0..2) {
            Some("0x" | "0X") => Some((16, &t[2..])),
            Some("0o" | "0O") => Some((8, &t[2..])),
            Some("0b" | "0B") => Some((2, &t[2..])),
            _ => None,
        };
        if let Some((radix, body)) = radixed {
            return i64::from_str_radix(body, radix).map_or(f64::NAN, |n| n as f64);
        }
        js_str_to_f64(t)
    }

    fn compare(&self, a: NanBox, b: NanBox) -> Option<core::cmp::Ordering> {
        // Fast path: two real string cells compare by their WTF-8 bytes, with no
        // `String` allocation (a single-leaf string borrows its bytes zero-copy).
        // WTF-8 byte order matches UTF-16 code-unit order across the BMP, the
        // accepted basis for JS string ordering here, and — unlike a lossy
        // `materialize()` — keeps lone surrogates distinct.
        if let (Some(ha), Some(hb)) = (a.as_handle(), b.as_handle()) {
            let ca = self.heap.get(Handle::from_raw(ha)).and_then(Cell::as_str);
            let cb = self.heap.get(Handle::from_raw(hb)).and_then(Cell::as_str);
            if let (Some(ra), Some(rb)) = (ca, cb) {
                return Some(rope_bytes(ra).cmp(&rope_bytes(rb)));
            }
        }
        // The abstract relational comparison applies ToPrimitive(Number) to each
        // operand: a string (or an object that stringifies, e.g. an array or plain
        // object) yields a string; everything else (numbers, booleans, and Dates —
        // whose `valueOf` is the timestamp) yields a number. If *both* sides are
        // strings they compare lexicographically; otherwise both compare as numbers.
        enum P {
            S(alloc::string::String),
            N(f64),
        }
        let prim = |this: &Self, v: NanBox| -> P {
            if this.is_string(v) {
                return P::S(this.to_display_string(v));
            }
            if let Some(raw) = v.as_handle() {
                let h = Handle::from_raw(raw);
                let is_str_cell = this.heap.get(h).and_then(Cell::as_str).is_some();
                if !is_str_cell && this.date_at(h).is_none() {
                    // array / plain object / function → ToPrimitive → toString.
                    return P::S(this.to_display_string(v));
                }
            }
            P::N(this.to_number(v))
        };
        match (prim(self, a), prim(self, b)) {
            (P::S(sa), P::S(sb)) => Some(sa.cmp(&sb)),
            (pa, pb) => {
                let n = |p: P, this: &Self| match p {
                    P::N(n) => n,
                    P::S(s) => this.number_from_str(&s),
                };
                n(pa, self).partial_cmp(&n(pb, self)) // None on NaN
            }
        }
    }

    /// `a < b` (boolean).
    #[must_use]
    pub fn less_than(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(self.compare(a, b) == Some(core::cmp::Ordering::Less))
    }

    /// `a <= b` (boolean).
    #[must_use]
    pub fn less_equal(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(matches!(
            self.compare(a, b),
            Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
        ))
    }

    /// `a > b` (boolean).
    #[must_use]
    pub fn greater_than(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(self.compare(a, b) == Some(core::cmp::Ordering::Greater))
    }

    /// `a >= b` (boolean).
    #[must_use]
    pub fn greater_equal(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(matches!(
            self.compare(a, b),
            Some(core::cmp::Ordering::Greater | core::cmp::Ordering::Equal)
        ))
    }

    /// `a - b` (numeric, both coerced with `ToNumber`).
    #[must_use]
    pub fn sub(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) - self.to_number(b))
    }

    /// `a * b` (numeric).
    #[must_use]
    pub fn mul(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) * self.to_number(b))
    }

    /// `a / b` (numeric; division by zero yields ±∞ / `NaN` per IEEE-754).
    #[must_use]
    pub fn div(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) / self.to_number(b))
    }

    /// `a % b` (the ECMAScript remainder, which follows the sign of the
    /// dividend — Rust's `%` on `f64` matches).
    #[must_use]
    pub fn rem(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) % self.to_number(b))
    }

    /// `a ** b` (exponentiation). Needs `std` for the float `powf` intrinsic
    /// (the alloc-only core omits it, like the rest of the engine's float math).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn pow(&self, a: NanBox, b: NanBox) -> NanBox {
        let base = self.to_number(a);
        let exp = self.to_number(b);
        // ECMAScript `Number::exponentiate` differs from IEEE `powf`: a NaN exponent is
        // always NaN (so `1 ** NaN` is NaN, not 1), and `(±1) ** ±Infinity` is NaN —
        // but `x ** ±0` is 1 for every `x` (including NaN).
        let r = if exp == 0.0 {
            1.0
        } else if exp.is_nan() || base.is_nan() || (base.abs() == 1.0 && exp.is_infinite()) {
            f64::NAN
        } else {
            base.powf(exp)
        };
        NanBox::number(r)
    }

    /// Unary `-a` (numeric negation).
    #[must_use]
    pub fn neg(&self, a: NanBox) -> NanBox {
        NanBox::number(-self.to_number(a))
    }

    /// Unary `!a` (logical negation via `ToBoolean`).
    #[must_use]
    pub fn logical_not(&self, a: NanBox) -> NanBox {
        NanBox::boolean(!self.truthy(a))
    }

    /// JS truthiness, heap-aware: `false`/`0`/`NaN`/`null`/`undefined` and the
    /// **empty string** are falsy; every other value (including non-empty
    /// strings and all objects) is truthy. `NanBox::to_boolean` alone can't see
    /// that a boxed string is empty, so string handles are checked here.
    #[must_use]
    pub fn truthy(&self, v: NanBox) -> bool {
        if let Some(raw) = v.as_handle() {
            let h = Handle::from_raw(raw);
            // Emptiness is O(1) on the rope (cached length) — never materialize the
            // whole string just to test it. `Cell::as_str` + `Rope::is_empty` are
            // both allocation-free, so this runs on every `if`/`while`/`&&`/`||`/
            // `??`/ternary without copying the string.
            if let Some(r) = self.heap.get(h).and_then(Cell::as_str) {
                return !r.is_empty();
            }
            if let Some(n) = self.bigint_at(h) {
                return !n.is_zero(); // `0n` is falsy
            }
            return true;
        }
        v.to_boolean()
    }

    /// The `typeof` string for any value: primitives via the box
    /// (`"undefined"`/`"boolean"`/`"number"`/`"object"` for null), and heap
    /// values via their cell (`"string"` for strings, `"object"` otherwise).
    #[must_use]
    pub fn type_of_value(&self, v: NanBox) -> &'static str {
        match v.as_handle() {
            Some(raw) => {
                let h = Handle::from_raw(raw);
                // A proxy reflects its target's `typeof` (function vs object).
                if let Some((target, _)) = self.proxy_at(h) {
                    return self.type_of(target).unwrap_or("object");
                }
                // A bound function (reserved `\0bnd_t` slot) is a function.
                if self.get_property(h, "\u{0}bnd_t").is_some() {
                    return "function";
                }
                // A bytecode-VM closure is represented as an array tagged with the
                // reserved `\0vmfn` marker; `typeof` reports it as a function.
                if self.get_property(h, "\u{0}vmfn").is_some() {
                    return "function";
                }
                self.heap.get(h).map_or("undefined", Cell::type_of)
            }
            None => v.type_of(),
        }
    }

    /// ECMAScript `ToInt32`. Needs `std` for the `trunc` float intrinsic.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn to_int32(&self, v: NanBox) -> i32 {
        let n = self.to_number(v);
        if !n.is_finite() || n == 0.0 {
            return 0;
        }
        // Reduce trunc(n) modulo 2^32 into [0, 2^32), then reinterpret as i32.
        let m = n.trunc().rem_euclid(4_294_967_296.0);
        m as u32 as i32
    }

    /// ECMAScript `ToUint32`. Needs `std` for the `trunc` float intrinsic.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn to_uint32(&self, v: NanBox) -> u32 {
        self.to_int32(v) as u32
    }

    /// `ArraySetLength` length validation for an already-`ToNumber`'d value:
    /// `ToUint32(v)` must equal `ToNumber(v)` (so `-1`, `4294967296`, `1.5`,
    /// `NaN` are rejected), returning `Some(len)` or `None` when the caller must
    /// raise a `RangeError("Invalid array length")`. (The caller is responsible
    /// for firing `ToNumber`'s `valueOf`/`toString` first when `v` may be an
    /// object.) Self-contained (no `f64::trunc`) so it is available in the
    /// `no_std`/`alloc`-only build too: a valid array length is exactly an
    /// integer in `[0, 2^32-1]`, which is its own `ToUint32`.
    #[must_use]
    pub fn array_length_uint32(&self, v: NanBox) -> Option<u32> {
        let number_len = self.to_number(v);
        // A length is valid iff it is a non-negative integer below 2^32; such a
        // value already equals its ToUint32, so no modular reduction is needed.
        // `as u32` saturates an out-of-range/negative value, so the round-trip
        // equality alone rejects `-1`/`2^32`/`1.5`/`NaN`/`Infinity`.
        if number_len.is_finite() && number_len == f64::from(number_len as u32) {
            Some(number_len as u32)
        } else {
            None
        }
    }

    /// `a & b` (bitwise AND over `ToInt32`). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_and(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) & self.to_int32(b)))
    }

    /// `a | b` (bitwise OR). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_or(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) | self.to_int32(b)))
    }

    /// `a ^ b` (bitwise XOR). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_xor(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) ^ self.to_int32(b)))
    }

    /// `~a` (bitwise NOT). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_not(&self, a: NanBox) -> NanBox {
        NanBox::number(f64::from(!self.to_int32(a)))
    }

    /// `a << b` (left shift; `b` masked to 5 bits, per spec). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn shl(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) << (self.to_uint32(b) & 31)))
    }

    /// `a >> b` (sign-propagating right shift). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn shr(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) >> (self.to_uint32(b) & 31)))
    }

    /// `a >>> b` (zero-fill right shift; result is unsigned). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn ushr(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_uint32(a) >> (self.to_uint32(b) & 31)))
    }

    /// ECMAScript abstract equality (`==`) — strict equality plus coercion:
    /// `null == undefined`; a boolean is compared as its number; a number and a
    /// string compare numerically; two references use [`strict_equals`] (string
    /// value / object identity). (Full `ToPrimitive` on objects arrives later.)
    ///
    /// [`strict_equals`]: Realm::strict_equals
    #[must_use]
    pub fn loose_equals(&self, a: NanBox, b: NanBox) -> bool {
        use crate::nanbox::Unpacked::{Bool, Handle as H, Null, Undefined};
        let (ua, ub) = (a.unpack(), b.unpack());
        let nullish = |u| matches!(u, Undefined | Null);
        // null/undefined are == to each other and to nothing else.
        if nullish(ua) || nullish(ub) {
            return nullish(ua) && nullish(ub);
        }
        // Two references: strings by value, objects by identity.
        if matches!(ua, H(_)) && matches!(ub, H(_)) {
            return self.strict_equals(a, b);
        }
        // Booleans compare as their numeric value.
        if matches!(ua, Bool(_)) {
            return self.loose_equals(NanBox::number(self.to_number(a)), b);
        }
        if matches!(ub, Bool(_)) {
            return self.loose_equals(a, NanBox::number(self.to_number(b)));
        }
        // Remaining: number vs number, or number vs string — compare numerically.
        self.to_number(a) == self.to_number(b)
    }

    /// ECMAScript strict equality (`===`) over heap values: primitives compare
    /// by value; two strings compare by *content* (strings are primitives, so
    /// distinct allocations of `"ab"` are equal); other references compare by
    /// identity.
    #[must_use]
    pub fn strict_equals(&self, a: NanBox, b: NanBox) -> bool {
        match (a.as_handle(), b.as_handle()) {
            (Some(ha), Some(hb)) => {
                if ha == hb {
                    return true; // same heap cell
                }
                let sa = self.heap.get(Handle::from_raw(ha)).and_then(Cell::as_str);
                let sb = self.heap.get(Handle::from_raw(hb)).and_then(Cell::as_str);
                match (sa, sb) {
                    // Compare the WTF-8 bytes directly: no `String` allocation, an
                    // O(1) length fast-reject, and — unlike the old lossy
                    // `materialize()` (which mapped every lone surrogate to U+FFFD,
                    // making `"\uD800" === "\uDC00"` wrongly true) — byte equality is
                    // exact.
                    (Some(ra), Some(rb)) => {
                        ra.len() == rb.len() && rope_bytes(ra) == rope_bytes(rb)
                    }
                    _ => {
                        // Two distinct BigInt cells are `===` iff their mathematical
                        // values are equal (BigInts are value types, not references).
                        match (
                            self.heap
                                .get(Handle::from_raw(ha))
                                .and_then(Cell::as_bigint),
                            self.heap
                                .get(Handle::from_raw(hb))
                                .and_then(Cell::as_bigint),
                        ) {
                            (Some(ba), Some(bb)) => ba == bb,
                            _ => false, // distinct non-string/non-bigint references
                        }
                    }
                }
            }
            // At least one primitive: decided by the boxed value itself.
            _ => a.strict_equals(b),
        }
    }
}

/// Borrows a rope's WTF-8 bytes zero-copy when it is an unconcatenated leaf, and
/// only materializes (one `Vec`) for a `Concat` tree. Used by the string-equality
/// and ordering hot paths so the overwhelmingly common single-leaf string compares
/// without allocating.
fn rope_bytes(r: &Rope) -> alloc::borrow::Cow<'_, [u8]> {
    match r.as_leaf_bytes() {
        Some(b) => alloc::borrow::Cow::Borrowed(b),
        None => alloc::borrow::Cow::Owned(r.materialize_bytes()),
    }
}

/// Parses a (decimal, already-trimmed, non-radix) `StrDecimalLiteral` to an
/// `f64` per ECMAScript `StringToNumber`. Unlike Rust's `f64::from_str`, the
/// grammar only admits the exact word `Infinity` (optionally `+`/`-`-signed) —
/// not `inf`/`infinity`/`INFINITY` — and never `NaN`/`nan` (those are a parse
/// failure → `NaN`). Everything else defers to Rust's parser.
#[must_use]
fn js_str_to_f64(t: &str) -> f64 {
    // The Infinity word is the only alphabetic literal ECMAScript accepts.
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    let neg = t.starts_with('-');
    if body == "Infinity" {
        return if neg {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    // Reject every other token containing an ASCII letter (Rust would accept
    // `inf`/`infinity`/`nan` in any case, plus hex floats via `e`/`x` — guard the
    // alphabetic ones; `e`/`E` exponents in a numeric literal are still allowed
    // because such a token also contains digits and parses below).
    if body
        .bytes()
        .any(|b| b.is_ascii_alphabetic() && !matches!(b, b'e' | b'E'))
    {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// Renders a number as ECMAScript `ToString` would for the cases the engine
/// produces: `±Infinity` (not Rust's `inf`) and `NaN`; finite values use Rust's
/// `Display` (which omits a trailing `.0` for integers).
#[must_use]
pub(crate) fn js_number_string(n: f64) -> alloc::string::String {
    if n.is_nan() {
        return "NaN".into();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    // Both `+0` and `-0` stringify to "0".
    if n == 0.0 {
        return "0".into();
    }
    let abs = n.abs();
    // JS uses exponential notation for magnitudes ≥ 1e21 or (nonzero) < 1e-6.
    if abs != 0.0 && !(1e-6..1e21).contains(&abs) {
        let s = alloc::format!("{n:e}"); // e.g. "1e21", "1.5e-7"
        if let Some(epos) = s.find('e') {
            let mant = &s[..epos];
            let exp: i64 = s[epos + 1..].parse().unwrap_or(0);
            let sign = if exp >= 0 { "+" } else { "-" };
            return alloc::format!("{mant}e{sign}{}", exp.abs());
        }
        s
    } else {
        alloc::format!("{n}")
    }
}

/// The civil date `(year, month [1-12], day [1-31])` for a day count `z` since
/// the Unix epoch — Howard Hinnant's `civil_from_days` algorithm (pure integer
/// arithmetic, so `core`-clean).
#[must_use]
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Days since the Unix epoch for a civil date (`y` full year, `m` in `1..=12`,
/// `d` day) — the inverse of [`civil_from_days`] (Howard Hinnant's algorithm).
#[must_use]
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let m = m as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Parses an ISO-8601 date/time string (`YYYY-MM-DD`, optionally
/// `THH:MM[:SS[.sss]]` and a trailing `Z`) into milliseconds since the epoch.
/// Returns `None` for anything it cannot parse (the caller yields `NaN`).
#[must_use]
pub fn parse_iso_date(s: &str) -> Option<f64> {
    let s = s.trim();
    let (date, time) = match s.split_once('T') {
        Some((d, t)) => (d, Some(t)),
        None => (s, None),
    };
    // An expanded year has a leading sign (`+YYYYYY` / `-YYYYYY`); preserve it so
    // the year split below doesn't treat a leading `-` as an empty field.
    let (year_sign, date_body): (i64, &str) = if let Some(rest) = date.strip_prefix('-') {
        (-1, rest)
    } else if let Some(rest) = date.strip_prefix('+') {
        (1, rest)
    } else {
        (1, date)
    };
    let mut dp = date_body.split('-');
    let year_field = dp.next()?;
    let y_mag: i64 = year_field.parse::<i64>().ok()?;
    // `-000000` (negative zero as an expanded year) is invalid per spec.
    if year_sign < 0 && y_mag == 0 {
        return None;
    }
    let y: i64 = year_sign * y_mag;
    // Month and day are optional: `YYYY` and `YYYY-MM` are valid ISO date forms
    // (defaulting the omitted fields to 1), per Date Time String Format.
    let mo: u32 = match dp.next() {
        Some(m) => m.parse().ok()?,
        None => 1,
    };
    let d: u32 = match dp.next() {
        Some(day) => day.parse().ok()?,
        None => 1,
    };
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || dp.next().is_some() {
        return None;
    }
    let mut ms: i64 = days_from_civil(y, mo, d) * 86_400_000;
    if let Some(t) = time {
        // A trailing timezone designator: `Z` (UTC), or a numeric `+HH:MM` / `-HH:MM`
        // offset. The offset is subtracted to convert the wall-clock time to UTC. With no
        // designator the time is taken as UTC (this engine has no local timezone).
        let (t, offset_min): (&str, i64) = if let Some(rest) = t.strip_suffix('Z') {
            (rest, 0)
        } else if let Some(pos) = t
            .char_indices()
            // The sign must not be the first character of the time component (a
            // leading `+`/`-` there is malformed, not an offset). `char_indices`
            // yields valid char-boundary byte indices, so the slices below never
            // split a multi-byte sequence on untrusted input.
            .find(|&(i, c)| i > 0 && (c == '+' || c == '-'))
            .map(|(i, _)| i)
        {
            let off = &t[pos..];
            let sign: i64 = if off.starts_with('-') { -1 } else { 1 };
            // The sign is a single ASCII byte, so `off[1..]` is a valid boundary.
            let body = &off[1..];
            let (oh, om): (i64, i64) = match body.split_once(':') {
                Some((a, b)) => (a.parse().ok()?, b.parse().ok()?),
                // `HHMM` with no separator: split after two ASCII digits. A
                // non-ASCII body fails the digit check and yields `None` (NaN)
                // rather than panicking on a char-boundary slice.
                None if body.len() == 4 && body.is_char_boundary(2) => {
                    (body[..2].parse().ok()?, body[2..].parse().ok()?)
                }
                None => (body.parse().ok()?, 0),
            };
            (&t[..pos], sign * (oh * 60 + om))
        } else {
            (t, 0)
        };
        let (hms, frac) = match t.split_once('.') {
            Some((a, b)) => (a, Some(b)),
            None => (t, None),
        };
        let mut tp = hms.split(':');
        let h: i64 = tp.next()?.parse().ok()?;
        let mi: i64 = tp.next().unwrap_or("0").parse().ok()?;
        let sec: i64 = tp.next().unwrap_or("0").parse().ok()?;
        ms += h * 3_600_000 + mi * 60_000 + sec * 1000;
        if let Some(f) = frac {
            let digits: alloc::string::String = f.chars().take(3).collect();
            // Pad to exactly three digits (milliseconds).
            let padded = alloc::format!("{digits:0<3}");
            ms += padded.parse::<i64>().ok()?;
        }
        ms -= offset_min * 60_000;
    }
    Some(ms as f64)
}

/// Parses a date string for `Date.parse`/`new Date(string)`: the ISO-8601 form
/// first, then the implementation-defined human-readable forms this engine emits
/// (`toString` and `toUTCString`, both UTC). Returns `None` on failure.
#[must_use]
pub fn parse_date_string(s: &str) -> Option<f64> {
    if let Some(ms) = parse_iso_date(s) {
        return Some(ms);
    }
    parse_human_date(s)
}

/// Parses the engine's `toString`/`toUTCString`/`toDateString` output:
/// - `"Thu, 01 Jan 1970 00:00:00 GMT"` (toUTCString)
/// - `"Thu Jan 01 1970 00:00:00 GMT+0000 (…)"` (toString)
/// - `"Thu Jan 01 1970"` (toDateString)
fn parse_human_date(s: &str) -> Option<f64> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let s = s.trim();
    // Drop a leading weekday token (`Thu` / `Thu,`).
    let rest = s.split_once(' ').map(|(_, r)| r).unwrap_or(s);
    let toks: alloc::vec::Vec<&str> = rest.split_whitespace().collect();
    // Two layouts: `Mon DD YYYY [time…]` (toString) or `DD Mon YYYY [time…]`
    // (toUTCString).
    let (mon_str, day_str, year_str, time_idx) = if toks.len() >= 3 && MONTHS.contains(&toks[0]) {
        (toks[0], toks[1], toks[2], 3)
    } else if toks.len() >= 3 && MONTHS.contains(&toks[1]) {
        (toks[1], toks[0], toks[2], 3)
    } else {
        return None;
    };
    let mo = MONTHS.iter().position(|&m| m == mon_str)? as u32 + 1;
    let day: u32 = day_str.parse().ok()?;
    let year: i64 = year_str.parse().ok()?;
    let mut ms = days_from_civil(year, mo, day) * 86_400_000;
    // Optional `HH:MM:SS` time component.
    if let Some(time) = toks.get(time_idx)
        && time.contains(':')
    {
        let mut tp = time.split(':');
        let h: i64 = tp.next()?.parse().ok()?;
        let mi: i64 = tp.next().unwrap_or("0").parse().ok()?;
        let se: i64 = tp.next().unwrap_or("0").parse().ok()?;
        ms += h * 3_600_000 + mi * 60_000 + se * 1000;
    }
    Some(ms as f64)
}

/// Renders a millisecond timestamp as an ISO-8601 UTC string.
pub(crate) fn date_to_iso(ms: f64) -> alloc::string::String {
    let total_ms = ms as i64;
    let day = total_ms.div_euclid(86_400_000);
    let tod = total_ms.rem_euclid(86_400_000); // [0, 86_400_000)
    let (y, mo, d) = civil_from_days(day);
    let (h, min, s, milli) = (
        tod / 3_600_000,
        (tod / 60_000) % 60,
        (tod / 1000) % 60,
        tod % 1000,
    );
    // Years outside 0..=9999 use the expanded `±YYYYYY` form (6 digits, signed);
    // otherwise the 4-digit form.
    let year = if (0..=9999).contains(&y) {
        alloc::format!("{y:04}")
    } else if y < 0 {
        alloc::format!("-{:06}", -y)
    } else {
        alloc::format!("+{y:06}")
    };
    alloc::format!("{year}-{mo:02}-{d:02}T{h:02}:{min:02}:{s:02}.{milli:03}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_access_properties() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        realm.set_property(obj, "x", NanBox::number(1.0));
        realm.set_property(obj, "y", NanBox::number(2.0));
        assert_eq!(realm.get_property(obj, "x"), Some(NanBox::number(1.0)));
        assert_eq!(realm.get_property(obj, "y"), Some(NanBox::number(2.0)));
        assert_eq!(realm.get_property(obj, "z"), None);
        assert_eq!(realm.object_count(), 1);
    }

    #[test]
    fn objects_share_hidden_classes_across_the_realm() {
        let mut realm = Realm::new();
        let a = realm.new_object();
        let b = realm.new_object();
        realm.set_property(a, "p", NanBox::number(1.0));
        realm.set_property(b, "p", NanBox::number(9.0));
        // Same structure built from the realm's shared root → same shape.
        let sa = Rc::clone(realm.heap.get(a).unwrap().as_object().unwrap().shape());
        let sb = Rc::clone(realm.heap.get(b).unwrap().as_object().unwrap().shape());
        assert!(Rc::ptr_eq(&sa, &sb));
    }

    #[test]
    fn gc_reclaims_unreachable_objects_including_cycles() {
        let mut realm = Realm::new();
        // root -> a; b <-> c form an unreachable cycle.
        let root = realm.new_object();
        let a = realm.new_object();
        realm.set_property(root, "a", NanBox::handle(a.to_raw()));

        let b = realm.new_object();
        let c = realm.new_object();
        realm.set_property(b, "c", NanBox::handle(c.to_raw()));
        realm.set_property(c, "b", NanBox::handle(b.to_raw()));
        assert_eq!(realm.object_count(), 4);

        let stats = realm.collect(&[root]);
        assert_eq!(stats.marked, 2); // root + a
        assert_eq!(stats.swept, 2); // the b<->c cycle
        assert!(realm.is_live(root) && realm.is_live(a));
        assert!(!realm.is_live(b) && !realm.is_live(c));
        assert_eq!(realm.object_count(), 2);
    }

    #[test]
    fn stale_handle_access_is_safe() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        realm.set_property(obj, "x", NanBox::number(1.0));
        realm.collect(&[]); // frees obj (no roots)
        assert!(!realm.is_live(obj));
        assert_eq!(realm.get_property(obj, "x"), None);
        assert!(!realm.set_property(obj, "x", NanBox::number(2.0)));
    }

    #[test]
    fn interning_is_shared() {
        let mut realm = Realm::new();
        let a = realm.intern("length");
        let b = realm.intern("length");
        assert_eq!(a, b);
    }

    #[test]
    fn strings_and_arrays_are_heap_values() {
        let mut realm = Realm::new();
        let s = realm.new_string("hello");
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        let obj = realm.new_object();
        assert_eq!(realm.string_value(s).as_deref(), Some("hello"));
        assert_eq!(realm.array_elements(arr).map(<[_]>::len), Some(2));
        assert_eq!(realm.type_of(s), Some("string"));
        assert_eq!(realm.type_of(arr), Some("object"));
        assert_eq!(realm.type_of(obj), Some("object"));
        // A property op on a string cell is rejected (not an object).
        assert!(!realm.set_property(s, "x", NanBox::number(1.0)));
        assert_eq!(realm.get_property(s, "x"), None);
    }

    #[test]
    fn add_numbers_and_concatenates_strings() {
        let mut realm = Realm::new();
        // number + number → numeric.
        let n = realm.add(NanBox::number(2.0), NanBox::number(3.0));
        assert_eq!(n.as_number(), Some(5.0));
        // string + number → concatenation (coerced).
        let hi = realm.new_string("count: ");
        let r = realm.add(NanBox::handle(hi.to_raw()), NanBox::number(42.0));
        assert_eq!(realm.to_display_string(r), "count: 42");
        // string + string.
        let a = realm.new_string("foo");
        let b = realm.new_string("bar");
        let ab = realm.add(NanBox::handle(a.to_raw()), NanBox::handle(b.to_raw()));
        assert_eq!(realm.to_display_string(ab), "foobar");
    }

    #[test]
    fn string_append_loop_builds_correctly() {
        // The `s += part` shape, now through ropes in the heap.
        let mut realm = Realm::new();
        let mut acc = NanBox::handle(realm.new_string("").to_raw());
        let mut expected = alloc::string::String::new();
        for i in 0..30 {
            let part = realm.new_string(&alloc::format!("{i},"));
            acc = realm.add(acc, NanBox::handle(part.to_raw()));
            expected.push_str(&alloc::format!("{i},"));
        }
        assert_eq!(realm.to_display_string(acc), expected);
    }

    #[test]
    fn to_display_string_covers_kinds() {
        let mut realm = Realm::new();
        assert_eq!(realm.to_display_string(NanBox::undefined()), "undefined");
        assert_eq!(realm.to_display_string(NanBox::null()), "null");
        assert_eq!(realm.to_display_string(NanBox::boolean(true)), "true");
        assert_eq!(realm.to_display_string(NanBox::number(3.5)), "3.5");
        let arr = realm.new_array(alloc::vec![
            NanBox::number(1.0),
            NanBox::number(2.0),
            NanBox::number(3.0),
        ]);
        assert_eq!(
            realm.to_display_string(NanBox::handle(arr.to_raw())),
            "1,2,3"
        );
        let obj = realm.new_object();
        assert_eq!(
            realm.to_display_string(NanBox::handle(obj.to_raw())),
            "[object Object]"
        );
    }

    #[test]
    fn strict_equals_strings_by_value_objects_by_identity() {
        let mut realm = Realm::new();
        // Primitives.
        assert!(realm.strict_equals(NanBox::number(1.0), NanBox::number(1.0)));
        assert!(!realm.strict_equals(NanBox::number(1.0), NanBox::null()));
        // Two distinct string allocations with equal content are ===.
        let a = realm.new_string("hello");
        let b = realm.new_string("hello");
        let c = realm.new_string("world");
        assert_ne!(a.to_raw(), b.to_raw()); // genuinely different cells
        assert!(realm.strict_equals(NanBox::handle(a.to_raw()), NanBox::handle(b.to_raw())));
        assert!(!realm.strict_equals(NanBox::handle(a.to_raw()), NanBox::handle(c.to_raw())));
        // Objects compare by identity.
        let o1 = realm.new_object();
        let o2 = realm.new_object();
        assert!(realm.strict_equals(NanBox::handle(o1.to_raw()), NanBox::handle(o1.to_raw())));
        assert!(!realm.strict_equals(NanBox::handle(o1.to_raw()), NanBox::handle(o2.to_raw())));
    }

    #[test]
    fn to_number_coerces() {
        let mut realm = Realm::new();
        assert_eq!(realm.to_number(NanBox::number(3.5)), 3.5);
        assert_eq!(realm.to_number(NanBox::boolean(true)), 1.0);
        assert_eq!(realm.to_number(NanBox::boolean(false)), 0.0);
        assert_eq!(realm.to_number(NanBox::null()), 0.0);
        assert!(realm.to_number(NanBox::undefined()).is_nan());
        let s = realm.new_string("  42  ");
        assert_eq!(realm.to_number(NanBox::handle(s.to_raw())), 42.0);
        let blank = realm.new_string("");
        assert_eq!(realm.to_number(NanBox::handle(blank.to_raw())), 0.0);
        let bad = realm.new_string("nope");
        assert!(realm.to_number(NanBox::handle(bad.to_raw())).is_nan());
    }

    #[test]
    fn relational_operators_on_numbers_and_strings() {
        let mut realm = Realm::new();
        let one = NanBox::number(1.0);
        let two = NanBox::number(2.0);
        assert_eq!(realm.less_than(one, two).as_boolean(), Some(true));
        assert_eq!(realm.less_than(two, one).as_boolean(), Some(false));
        assert_eq!(realm.greater_than(two, one).as_boolean(), Some(true));
        assert_eq!(realm.less_equal(two, two).as_boolean(), Some(true));
        assert_eq!(realm.greater_equal(two, two).as_boolean(), Some(true));
        // A NaN operand makes every comparison false.
        let nan = NanBox::number(f64::NAN);
        assert_eq!(realm.less_than(nan, one).as_boolean(), Some(false));
        assert_eq!(realm.greater_than(nan, one).as_boolean(), Some(false));
        // Two strings compare lexicographically: "10" < "9" (string order).
        let s10 = realm.new_string("10");
        let s9 = realm.new_string("9");
        assert_eq!(
            realm
                .less_than(NanBox::handle(s10.to_raw()), NanBox::handle(s9.to_raw()))
                .as_boolean(),
            Some(true)
        );
        // Mixed string/number coerces to numeric: 10 < "9" is false (9 < 10).
        assert_eq!(
            realm
                .less_than(NanBox::number(10.0), NanBox::handle(s9.to_raw()))
                .as_boolean(),
            Some(false)
        );
    }

    #[test]
    fn arithmetic_operators() {
        let realm = Realm::new();
        let n = NanBox::number;
        assert_eq!(realm.sub(n(5.0), n(3.0)).as_number(), Some(2.0));
        assert_eq!(realm.mul(n(4.0), n(2.5)).as_number(), Some(10.0));
        assert_eq!(realm.div(n(9.0), n(2.0)).as_number(), Some(4.5));
        assert_eq!(realm.rem(n(7.0), n(3.0)).as_number(), Some(1.0));
        assert_eq!(realm.rem(n(-7.0), n(3.0)).as_number(), Some(-1.0)); // sign of dividend
        assert_eq!(realm.neg(n(3.0)).as_number(), Some(-3.0));
        // Division by zero is ±Infinity.
        assert_eq!(realm.div(n(1.0), n(0.0)).as_number(), Some(f64::INFINITY));
        #[cfg(feature = "std")]
        assert_eq!(realm.pow(n(2.0), n(10.0)).as_number(), Some(1024.0));
    }

    #[test]
    fn abstract_equality_coerces() {
        let mut realm = Realm::new();
        let n = NanBox::number;
        // null == undefined, but not == 0.
        assert!(realm.loose_equals(NanBox::null(), NanBox::undefined()));
        assert!(!realm.loose_equals(NanBox::null(), n(0.0)));
        // number == string by numeric coercion.
        let s1 = NanBox::handle(realm.new_string("1").to_raw());
        assert!(realm.loose_equals(n(1.0), s1));
        // boolean coerces to number: true == 1, false == 0.
        assert!(realm.loose_equals(NanBox::boolean(true), n(1.0)));
        assert!(realm.loose_equals(NanBox::boolean(false), n(0.0)));
        assert!(!realm.loose_equals(NanBox::boolean(true), n(2.0)));
        // strings by value; objects by identity.
        let a = NanBox::handle(realm.new_string("x").to_raw());
        let b = NanBox::handle(realm.new_string("x").to_raw());
        assert!(realm.loose_equals(a, b));
        let o1 = NanBox::handle(realm.new_object().to_raw());
        assert!(!realm.loose_equals(o1, n(0.0))); // object != 0 (ToNumber→NaN)
    }

    #[test]
    fn typeof_and_logical_not() {
        let mut realm = Realm::new();
        assert_eq!(realm.type_of_value(NanBox::undefined()), "undefined");
        assert_eq!(realm.type_of_value(NanBox::null()), "object");
        assert_eq!(realm.type_of_value(NanBox::boolean(true)), "boolean");
        assert_eq!(realm.type_of_value(NanBox::number(1.0)), "number");
        let s = NanBox::handle(realm.new_string("hi").to_raw());
        assert_eq!(realm.type_of_value(s), "string");
        let o = NanBox::handle(realm.new_object().to_raw());
        assert_eq!(realm.type_of_value(o), "object");
        // ToBoolean-based negation.
        assert_eq!(
            realm.logical_not(NanBox::number(0.0)).as_boolean(),
            Some(true)
        );
        assert_eq!(
            realm.logical_not(NanBox::number(1.0)).as_boolean(),
            Some(false)
        );
        assert_eq!(realm.logical_not(s).as_boolean(), Some(false)); // objects truthy
    }

    #[cfg(feature = "std")]
    #[test]
    fn bitwise_operators() {
        let realm = Realm::new();
        let n = NanBox::number;
        assert_eq!(realm.bit_and(n(12.0), n(10.0)).as_number(), Some(8.0));
        assert_eq!(realm.bit_or(n(12.0), n(10.0)).as_number(), Some(14.0));
        assert_eq!(realm.bit_xor(n(12.0), n(10.0)).as_number(), Some(6.0));
        assert_eq!(realm.bit_not(n(0.0)).as_number(), Some(-1.0));
        assert_eq!(realm.shl(n(1.0), n(4.0)).as_number(), Some(16.0));
        assert_eq!(realm.shr(n(-8.0), n(1.0)).as_number(), Some(-4.0)); // sign-propagating
        assert_eq!(realm.ushr(n(-1.0), n(0.0)).as_number(), Some(4294967295.0)); // zero-fill
        // ToInt32 truncates fractional and wraps modulo 2^32.
        assert_eq!(realm.to_int32(n(3.9)), 3);
        assert_eq!(realm.to_int32(n(4294967297.0)), 1);
        assert_eq!(realm.to_int32(n(f64::NAN)), 0);
    }

    #[test]
    fn array_index_length_and_push() {
        let mut realm = Realm::new();
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        assert_eq!(realm.array_length(arr), Some(2));
        assert_eq!(realm.get_element(arr, 0).as_number(), Some(1.0));
        assert_eq!(realm.get_element(arr, 1).as_number(), Some(2.0));
        // Out of range reads undefined.
        assert!(realm.get_element(arr, 5).is_undefined());
        // Setting past the end grows with holes.
        assert!(realm.set_element(arr, 4, NanBox::number(9.0)));
        assert_eq!(realm.array_length(arr), Some(5));
        assert!(realm.get_element(arr, 3).is_undefined()); // a hole
        assert_eq!(realm.get_element(arr, 4).as_number(), Some(9.0));
        // Push returns the new length.
        assert_eq!(realm.array_push(arr, NanBox::number(7.0)), Some(6));
        assert_eq!(realm.get_element(arr, 5).as_number(), Some(7.0));
        // join renders the array (holes empty).
        assert_eq!(
            realm.to_display_string(NanBox::handle(arr.to_raw())),
            "1,2,,,9,7"
        );
        // Array ops on a non-array are rejected.
        let obj = realm.new_object();
        assert_eq!(realm.array_length(obj), None);
        assert!(!realm.set_element(obj, 0, NanBox::number(1.0)));
    }

    #[test]
    fn gc_keeps_a_mixed_object_array_string_graph() {
        let mut realm = Realm::new();
        // obj.name -> string; obj.items -> array; array[0] -> obj (a cycle).
        let obj = realm.new_object();
        let name = realm.new_string("widget");
        let items = realm.new_array(alloc::vec![NanBox::handle(obj.to_raw())]);
        realm.set_property(obj, "name", NanBox::handle(name.to_raw()));
        realm.set_property(obj, "items", NanBox::handle(items.to_raw()));
        let _unreachable = realm.new_string("garbage");
        assert_eq!(realm.object_count(), 4);

        let stats = realm.collect(&[obj]);
        assert_eq!(stats.marked, 3); // obj, name, items (cycle obj<-items kept)
        assert_eq!(stats.swept, 1); // the unreachable string
        assert!(realm.is_live(obj) && realm.is_live(name) && realm.is_live(items));
        assert_eq!(realm.string_value(name).as_deref(), Some("widget"));
    }

    #[test]
    fn compaction_defragments_and_fixes_up_the_object_graph() {
        let mut realm = Realm::new();
        // obj.name -> string; obj.items -> array; array[0] -> obj (a cycle),
        // with unreachable garbage interleaved to create gaps.
        let _g0 = realm.new_string("garbage0");
        let obj = realm.new_object();
        let name = realm.new_string("widget");
        let _g1 = realm.new_object();
        let items = realm.new_array(alloc::vec![NanBox::handle(obj.to_raw())]);
        realm.set_property(obj, "name", NanBox::handle(name.to_raw()));
        realm.set_property(obj, "items", NanBox::handle(items.to_raw()));
        assert_eq!(realm.object_count(), 5);

        let mut roots = [obj];
        let stats = realm.compact(&mut roots);
        assert_eq!(stats.marked, 3); // obj, name, items
        assert_eq!(stats.swept, 2); // the two garbage objects
        assert_eq!(realm.object_count(), 3); // slot table defragmented

        // The root was rewritten; the whole graph resolves through new handles.
        let obj2 = roots[0];
        let name2 = realm.get_property(obj2, "name").unwrap();
        assert_eq!(
            realm
                .string_value(Handle::from_raw(name2.as_handle().unwrap()))
                .as_deref(),
            Some("widget")
        );
        let items2 = realm.get_property(obj2, "items").unwrap();
        let arr = Handle::from_raw(items2.as_handle().unwrap());
        // array[0] still points back at the (relocated) object — the cycle held.
        assert_eq!(realm.get_element(arr, 0).as_handle(), Some(obj2.to_raw()));
    }

    #[test]
    fn compaction_preserves_typed_array_view_aliasing() {
        let mut realm = Realm::new();
        // A byte-backed buffer and a Uint8 view over it, with garbage interleaved before
        // them so compaction actually relocates their slots.
        let _g0 = realm.new_string("garbage0");
        let bytes = realm.new_bytes(alloc::vec![0u8; 4]);
        let abuf = realm.new_object();
        let view = realm.new_typed_array(bytes, abuf, 0, 4, 1); // kind 1 = Uint8

        let mut roots = [bytes, view, abuf];
        realm.compact(&mut roots);
        let (bytes2, view2, abuf2) = (roots[0], roots[1], roots[2]);
        // The slots moved (garbage created gaps), so the raw handles changed.
        assert_ne!(bytes2.to_raw(), bytes.to_raw());

        // The view's intrinsic buffer link was forwarded with the cell, so a write to
        // the relocated buffer is still visible through the relocated view.
        realm.bytes_at_mut(bytes2).unwrap()[1] = 200;
        assert_eq!(realm.get_element(view2, 1).as_number(), Some(200.0));
        assert_eq!(realm.typed_buffer(view2), Some(bytes2));
        // The `[[ViewedArrayBuffer]]` object link was forwarded too.
        assert_eq!(realm.typed_array_object(view2), Some(abuf2));
    }

    #[test]
    fn typed_array_view_aliases_shared_bytes() {
        let mut realm = Realm::new();
        let buf = realm.new_bytes(alloc::vec![0u8; 8]);
        let abuf = realm.new_object();
        let u8v = realm.new_typed_array(buf, abuf, 0, 8, 1); // Uint8
        let f64v = realm.new_typed_array(buf, abuf, 0, 1, 8); // Float64
        // A write through one view is decoded by the sibling (intrinsic aliasing).
        realm.typed_set(u8v, 0, NanBox::number(255.0));
        assert_eq!(realm.get_element(u8v, 0).as_number(), Some(255.0));
        // Float64 over the same first byte sees the raw bytes change.
        assert_ne!(realm.get_element(f64v, 0).as_number(), Some(0.0));
        // resize_buffer grows the bytes. Only an auto-length-tracking view
        // re-spans the resized buffer; a fixed-length view keeps its length.
        realm.mark_length_tracking(u8v);
        realm.resize_buffer(buf, 16);
        assert_eq!(realm.typed_len(u8v), Some(16), "tracking view re-spans");
        // The fixed-length Float64 view keeps its declared length of 1 (and is in
        // bounds, since the buffer grew).
        assert_eq!(
            realm.typed_len(f64v),
            Some(1),
            "fixed view keeps its length"
        );
    }

    #[test]
    fn compaction_relocates_array_flag_tables() {
        let mut realm = Realm::new();
        // A frozen array (recorded in the handle-keyed `frozen_arrays` set), with garbage
        // before it to force slot relocation.
        let _g = realm.new_string("garbage");
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        realm.freeze_object(arr);

        let mut roots = [arr];
        realm.compact(&mut roots);
        let arr2 = roots[0];
        assert_ne!(arr2.to_raw(), arr.to_raw(), "slot relocated");

        // The frozen flag followed the array to its new handle.
        assert!(
            realm.is_frozen(arr2),
            "frozen flag survived compaction via relocation"
        );
        assert!(
            !realm.is_frozen(arr),
            "the stale handle is no longer flagged"
        );
    }

    #[test]
    fn compaction_roots_and_relocates_aux_properties() {
        let mut realm = Realm::new();
        // A named property on an *array* cell is stored in the handle-keyed `aux_props` table,
        // whose value object is reachable only through that table — so the collector must root
        // it and compaction must forward both the cell key and the aux-object value.
        let _g = realm.new_string("garbage");
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        let tag = realm.new_string("tag");
        realm.set_property(arr, "label", NanBox::handle(tag.to_raw()));

        let mut roots = [arr, tag];
        realm.compact(&mut roots);
        let arr2 = roots[0];
        assert_ne!(arr2.to_raw(), arr.to_raw(), "slot relocated");

        let label = realm
            .get_property(arr2, "label")
            .expect("aux property survived compaction (rooted + relocated)");
        assert_eq!(
            realm
                .string_value(Handle::from_raw(label.as_handle().unwrap()))
                .as_deref(),
            Some("tag"),
        );
    }

    #[test]
    fn compaction_roots_and_relocates_persistent_handles() {
        let mut realm = Realm::new();
        // A persisted object is reachable ONLY through the `host_persistent` table
        // (not the explicit roots), so the collector must root it and compaction
        // must forward it. Unrooted garbage before it forces relocation.
        let _garbage = realm.new_string("garbage");
        let obj = realm.new_object();
        let tag = realm.new_string("kept");
        realm.set_property(obj, "v", NanBox::handle(tag.to_raw()));
        let idx = realm.persist(NanBox::handle(obj.to_raw()));

        // Compact with NO explicit roots: only the persistent side table keeps `obj`
        // (and, transitively, `tag`) alive.
        let mut roots: [Handle; 0] = [];
        realm.compact(&mut roots);

        let obj2 = realm
            .persistent(idx)
            .expect("persistent survived compaction");
        let h2 = Handle::from_raw(obj2.as_handle().unwrap());
        assert_ne!(obj2.as_handle().unwrap(), obj.to_raw(), "handle relocated");
        let v = realm
            .get_property(h2, "v")
            .expect("transitive value survived");
        assert_eq!(
            realm
                .string_value(Handle::from_raw(v.as_handle().unwrap()))
                .as_deref(),
            Some("kept"),
        );

        // A primitive persists too (stored inline, no rooting needed).
        let pidx = realm.persist(NanBox::number(42.0));
        assert_eq!(
            realm.persistent(pidx).and_then(|n| n.as_number()),
            Some(42.0)
        );

        // Release frees the slot and drops the root; the index reads `None`.
        realm.release_persistent(idx);
        assert!(realm.persistent(idx).is_none());
    }

    #[test]
    fn interleaved_incremental_barrier_keeps_a_mid_cycle_store() {
        let mut realm = Realm::new();
        let root = realm.new_object();

        // Start an interleaved cycle and mark the root fully (it has no edges
        // yet, so marking completes with `root` black).
        realm.incremental_start(&[root]);
        while !realm.incremental_step(1) {}

        // Mid-cycle, the mutator allocates a new object (white) and stores it on
        // the already-black root. The integrated write barrier shades it.
        let child = realm.new_object();
        realm.set_property(root, "child", NanBox::handle(child.to_raw()));

        // Drain the re-greyed work, then finish.
        while !realm.incremental_step(4) {}
        let stats = realm.incremental_finish();

        assert_eq!(stats.swept, 0, "the barrier-shaded child must survive");
        assert!(realm.is_live(root) && realm.is_live(child));
    }

    #[test]
    fn incremental_collection_matches_full_over_an_object_graph() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        let name = realm.new_string("widget");
        let items = realm.new_array(alloc::vec![NanBox::handle(obj.to_raw())]);
        realm.set_property(obj, "name", NanBox::handle(name.to_raw()));
        realm.set_property(obj, "items", NanBox::handle(items.to_raw()));
        let _garbage = realm.new_string("garbage");
        assert_eq!(realm.object_count(), 4);

        // Tiny step budget → many incremental slices; same result as a full GC.
        let stats = realm.collect_incremental(&[obj], 1);
        assert_eq!(stats.marked, 3); // obj, name, items (cycle held)
        assert_eq!(stats.swept, 1); // the unreachable string
        assert!(realm.is_live(obj) && realm.is_live(name) && realm.is_live(items));
        assert_eq!(realm.string_value(name).as_deref(), Some("widget"));
    }

    #[test]
    fn minor_collection_with_write_barrier_keeps_old_to_young_edge() {
        let mut realm = Realm::new();
        // Create an object and promote it to the old generation via a major GC.
        let parent = realm.new_object();
        realm.collect(&[parent]);

        // Now attach a freshly-allocated (young) string — the `set_property`
        // write barrier records the old→young edge.
        let child = realm.new_string("attached");
        realm.set_property(parent, "child", NanBox::handle(child.to_raw()));
        let _garbage = realm.new_string("garbage"); // young, unreferenced

        // A minor collection frees only the young garbage; `child`, reachable
        // solely through the old `parent`, survives thanks to the barrier.
        let stats = realm.collect_minor(&[parent]);
        assert_eq!(stats.swept, 1);
        assert!(realm.is_live(parent) && realm.is_live(child));
        assert_eq!(realm.string_value(child).as_deref(), Some("attached"));
    }

    // ---- C1: unbounded-array OOM guard --------------------------------------

    #[test]
    fn set_element_refuses_to_grow_past_cap_instead_of_aborting() {
        let mut realm = Realm::new();
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        let cap = realm.limits.max_array_len;
        // A huge index (`a[1e9] = 1`, well past the 100M cap) must be refused,
        // not turned into a multi-gigabyte `Vec::resize`.
        assert!(!realm.set_element(arr, 1_000_000_000, NanBox::number(2.0)));
        // The array is untouched and still tiny.
        assert_eq!(realm.array_length(arr), Some(1));
        // The boundary: exactly `cap` is past the last valid index (`cap - 1`).
        assert!(!realm.set_element(arr, cap, NanBox::number(3.0)));
        // `usize::MAX` would overflow `index + 1`; `checked_add` refuses it.
        assert!(!realm.set_element(arr, usize::MAX, NanBox::number(4.0)));
        assert_eq!(realm.array_length(arr), Some(1));
    }

    #[test]
    fn set_array_length_sparse_past_cap() {
        let mut realm = Realm::new();
        let arr = realm.new_array(Vec::new());
        // `a.length = 1e9` (> 100M cap) is a *sparse* length: the dense backing is
        // not grown (no multi-gigabyte allocation), but `length` reports the value.
        assert!(realm.set_array_length(arr, 1_000_000_000));
        assert_eq!(realm.array_length(arr), Some(1_000_000_000));
        // A modest length within the cap resizes for real and drops the override.
        assert!(realm.set_array_length(arr, 5));
        assert_eq!(realm.array_length(arr), Some(5));
        // Growing back past the cap re-arms the sparse override.
        assert!(realm.set_array_length(arr, 4_294_967_295));
        assert_eq!(realm.array_length(arr), Some(4_294_967_295));
    }

    #[test]
    fn set_element_within_cap_still_grows() {
        let mut realm = Realm::new();
        let arr = realm.new_array(Vec::new());
        assert!(realm.set_element(arr, 10, NanBox::number(7.0)));
        assert_eq!(realm.array_length(arr), Some(11));
        assert_eq!(realm.get_element(arr, 10), NanBox::number(7.0));
    }

    // ---- P4: byte-exact string equality / ordering --------------------------

    #[test]
    fn strict_equals_distinguishes_lone_surrogates() {
        let mut realm = Realm::new();
        // "\uD800" and "\uDC00" are distinct lone surrogates. The old lossy
        // `materialize()` mapped both to U+FFFD and wrongly compared them equal;
        // byte comparison keeps them distinct.
        let hi = realm.new_string_wtf8(crate::wtf8::from_utf16(&[0xD800]));
        let lo = realm.new_string_wtf8(crate::wtf8::from_utf16(&[0xDC00]));
        let hi2 = realm.new_string_wtf8(crate::wtf8::from_utf16(&[0xD800]));
        assert!(!realm.strict_equals(NanBox::handle(hi.to_raw()), NanBox::handle(lo.to_raw())));
        // Equal content in distinct allocations is still equal.
        assert!(realm.strict_equals(NanBox::handle(hi.to_raw()), NanBox::handle(hi2.to_raw())));
    }

    #[test]
    fn strict_equals_and_ordering_on_plain_strings_unchanged() {
        let mut realm = Realm::new();
        let a = realm.new_string("apple");
        let a2 = realm.new_string("apple");
        let b = realm.new_string("banana");
        let (na, na2, nb) = (
            NanBox::handle(a.to_raw()),
            NanBox::handle(a2.to_raw()),
            NanBox::handle(b.to_raw()),
        );
        assert!(realm.strict_equals(na, na2));
        assert!(!realm.strict_equals(na, nb));
        assert_eq!(realm.less_than(na, nb), NanBox::boolean(true));
        assert_eq!(realm.less_than(nb, na), NanBox::boolean(false));
        // A concatenated rope (no single leaf) compares correctly too.
        let joined = {
            let r = realm.heap.get(a).unwrap().as_str().unwrap().clone();
            let suffix = Rope::leaf("!");
            realm.heap.alloc(Cell::Str(r.concat(&suffix)))
        };
        let nj = NanBox::handle(joined.to_raw());
        assert!(!realm.strict_equals(na, nj)); // "apple" != "apple!"
        assert_eq!(realm.less_than(na, nj), NanBox::boolean(true));
    }

    // ---- H3: truthy without materializing -----------------------------------

    #[test]
    fn truthy_on_strings_is_emptiness_only() {
        let mut realm = Realm::new();
        let empty = realm.new_string("");
        let nonempty = realm.new_string("x");
        assert!(!realm.truthy(NanBox::handle(empty.to_raw())));
        assert!(realm.truthy(NanBox::handle(nonempty.to_raw())));
        // A lone surrogate is a one-code-unit, non-empty string → truthy.
        let surr = realm.new_string_wtf8(crate::wtf8::from_utf16(&[0xD800]));
        assert!(realm.truthy(NanBox::handle(surr.to_raw())));
    }

    // ---- P2: leaf-byte borrow accessor --------------------------------------

    #[test]
    fn string_leaf_bytes_borrows_when_unconcatenated() {
        let mut realm = Realm::new();
        let leaf = realm.new_string("hi");
        assert_eq!(realm.string_leaf_bytes(leaf), Some(&b"hi"[..]));
        // A concatenated rope has no single backing slice.
        let joined = {
            let r = realm.heap.get(leaf).unwrap().as_str().unwrap().clone();
            realm.heap.alloc(Cell::Str(r.concat(&Rope::leaf("!"))))
        };
        assert_eq!(realm.string_leaf_bytes(joined), None);
        assert_eq!(realm.string_bytes(joined).as_deref(), Some(&b"hi!"[..]));
        // A non-string cell yields `None`.
        let arr = realm.new_array(Vec::new());
        assert_eq!(realm.string_leaf_bytes(arr), None);
    }

    // ---- H/GC: weak-key (ephemeron) side-table pruning ----------------------

    /// Populates every value-reachable side-table with `n` short-lived entries,
    /// returning nothing (all keys go out of scope), so a full `collect` should
    /// reclaim them all.
    #[cfg(test)]
    fn populate_side_tables(realm: &mut Realm, n: usize) {
        for i in 0..n {
            // aux_props: a named property on an array cell.
            let arr = realm.new_array(alloc::vec![NanBox::number(i as f64)]);
            realm.set_property(arr, "tag", NanBox::number(i as f64));
            // fn_protos + fn_ctor: a function, with its `.prototype` materialized.
            let f = realm.new_function(1_000_000 + i as u32, crate::env::Scope::root());
            let _proto = realm.function_prototype(1_000_000 + i as u32);
            let _ = f;
            // symbols_by_id: a fresh symbol (not referenced anywhere afterwards).
            let _sym = realm.new_symbol("s");
            // frozen/sealed/non_extensible_arrays: freeze a throwaway array.
            let fa = realm.new_array(alloc::vec![NanBox::number(0.0)]);
            realm.freeze_object(fa);
        }
    }

    #[test]
    fn weak_key_side_tables_do_not_leak_across_collections() {
        let mut realm = Realm::new();
        // Baseline: nothing in the side-tables.
        assert_eq!(realm.side_table_lens(), [0; 7]);

        // Repeatedly create short-lived state, drop it, and full-collect. The
        // side-table lengths must return to baseline each iteration rather than
        // grow without bound.
        for _ in 0..5 {
            populate_side_tables(&mut realm, 50);
            // The tables actually filled up (sanity: not a no-op test).
            assert!(realm.side_table_lens().iter().any(|&l| l >= 50));
            realm.collect(&[]); // no roots: every populated key is now dead
            assert_eq!(
                realm.side_table_lens(),
                [0; 7],
                "weak-key entries for dead objects must be pruned by a full collect",
            );
        }
    }

    #[test]
    fn live_keys_keep_their_side_table_entries() {
        let mut realm = Realm::new();

        // An array with an aux property, kept rooted.
        let arr = realm.new_array(alloc::vec![NanBox::number(7.0)]);
        realm.set_property(arr, "tag", NanBox::number(42.0));

        // A frozen array, kept rooted.
        let fa = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        realm.freeze_object(fa);

        // A function with a materialized prototype + constructor back-ref, rooted.
        let fid = 7u32;
        let f = realm.new_function(fid, crate::env::Scope::root());
        let proto = realm.function_prototype(fid);

        // A symbol used as a property key on a rooted object (the symbol cell is
        // reachable only via the `\0sym:{id}` key string — the ephemeron expand
        // must keep it alive).
        let host = realm.new_object();
        let sym = realm.new_symbol("k");
        let (_d, sid) = realm.symbol_at(sym).unwrap();
        let symkey = alloc::format!("\u{0}sym:{sid}");
        realm.set_property(host, &symkey, NanBox::number(99.0));

        // Also create a pile of dead entries that SHOULD be pruned.
        populate_side_tables(&mut realm, 30);

        realm.collect(&[arr, fa, f, proto, host]);

        // The live-key entries survived with their aux state intact.
        assert!(realm.is_live(arr));
        assert_eq!(realm.get_property(arr, "tag"), Some(NanBox::number(42.0)));
        assert!(realm.is_live(fa) && realm.is_frozen(fa));
        assert!(realm.is_live(f) && realm.is_live(proto));
        // The function's prototype is still the same object, with `constructor`
        // pointing back at the (live) function.
        assert_eq!(realm.function_prototype(fid), proto);
        let ctor = realm
            .get_property(proto, "constructor")
            .and_then(|v| v.as_handle());
        assert_eq!(ctor, Some(f.to_raw()));
        // The symbol is still resolvable by id and the symbol-keyed property reads.
        assert!(realm.is_live(sym));
        assert_eq!(realm.symbol_for_id(sid), Some(sym));
        assert_eq!(
            realm.get_property(host, &symkey),
            Some(NanBox::number(99.0))
        );

        // The dead entries were pruned: only the live ones remain.
        let [aux, protos, ctors, syms, frozen, sealed, nonext] = realm.side_table_lens();
        assert_eq!(aux, 1, "only the rooted array's aux entry remains");
        assert_eq!(protos, 1);
        assert_eq!(ctors, 1);
        assert_eq!(syms, 1, "only the in-use symbol remains");
        assert_eq!(frozen, 1);
        assert_eq!(sealed, 1);
        assert_eq!(nonext, 1);
    }

    #[test]
    fn ephemeron_value_can_keep_another_entrys_key_alive() {
        // Entry A: aux_props[arr_a] = aux object whose property points at arr_b.
        // Entry B: aux_props[arr_b] = aux object with a tag.
        // While arr_a is rooted, A's value keeps arr_b reachable, so B must
        // survive (the ephemeron fixpoint), even though arr_b is not a direct root.
        let mut realm = Realm::new();
        let arr_a = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        let arr_b = realm.new_array(alloc::vec![NanBox::number(2.0)]);
        // A.value -> arr_b (a handle property on A's aux object).
        realm.set_property(arr_a, "next", NanBox::handle(arr_b.to_raw()));
        // B has its own aux entry.
        realm.set_property(arr_b, "tag", NanBox::number(99.0));

        // Only arr_a is rooted; arr_b is reachable solely through A's aux value.
        realm.collect(&[arr_a]);

        assert!(realm.is_live(arr_a), "rooted key A survives");
        assert!(
            realm.is_live(arr_b),
            "A's aux value keeps B's key alive (ephemeron fixpoint)",
        );
        // Both aux entries survived, with B's tag still readable.
        assert_eq!(realm.get_property(arr_b, "tag"), Some(NanBox::number(99.0)));
        assert_eq!(
            realm
                .get_property(arr_a, "next")
                .and_then(|v| v.as_handle()),
            Some(arr_b.to_raw()),
        );
    }

    #[test]
    fn compaction_prunes_dead_weak_keys_and_keeps_live_ones() {
        let mut realm = Realm::new();
        // A rooted frozen array with an aux property, plus dead side-table state.
        let _g = realm.new_string("garbage");
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        realm.set_property(arr, "tag", NanBox::number(5.0));
        realm.freeze_object(arr);
        populate_side_tables(&mut realm, 20); // all dead

        let mut roots = [arr];
        realm.compact(&mut roots);
        let arr2 = roots[0];

        // The live array's weak-key state followed it to its new handle; the dead
        // entries were pruned (not relocated).
        assert!(realm.is_frozen(arr2));
        assert_eq!(realm.get_property(arr2, "tag"), Some(NanBox::number(5.0)));
        let [aux, _p, _c, _s, frozen, sealed, nonext] = realm.side_table_lens();
        assert_eq!(
            aux, 1,
            "only the live array's aux entry survived compaction"
        );
        assert_eq!((frozen, sealed, nonext), (1, 1, 1));
    }

    #[test]
    fn minor_collection_does_not_free_live_side_table_values() {
        // A minor collection must not prune the weak-key tables, but it must also
        // not free the values they point at (it roots them all).
        let mut realm = Realm::new();
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0)]);
        realm.set_property(arr, "tag", NanBox::number(3.0));
        // The aux entry exists.
        assert!(realm.side_table_lens()[0] >= 1);
        // A minor collection keeps the aux object alive even though `arr` is the
        // only thing referencing it and we pass it as a root.
        realm.collect_minor(&[arr]);
        assert!(realm.is_live(arr));
        assert_eq!(realm.get_property(arr, "tag"), Some(NanBox::number(3.0)));
    }
}
