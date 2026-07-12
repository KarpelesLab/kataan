use super::*;

impl<'a> Interp<'a> {
    // --- Embedder buffer-creation API (A6, #11) -----------------------------
    //
    // These build the *JS-visible* `ArrayBuffer` object — a heap object carrying
    // the hidden `ARRAY_BUFFER_BYTES` slot, exactly like a JS-created
    // `ArrayBuffer`, so the engine's `ArrayBuffer.prototype` methods, typed-array
    // views, `instanceof`, and WASM marshaling all treat it uniformly. The
    // returned `Handle` is a live heap object; keep it rooted (e.g. install it on
    // a global, pass it to script, or hold it across a call) to keep it — and its
    // owned/external `Cell::Bytes` — alive across collection.

    /// Builds a JS-visible `ArrayBuffer` object whose contiguous `Cell::Bytes`
    /// store is an engine-owned copy of `bytes`. Round-trips through JS like any
    /// `new ArrayBuffer(n)`; mutations via a view are visible through
    /// [`realm`](Self::realm)`.bytes_at(buffer_bytes)`.
    pub fn array_buffer_from_bytes(&mut self, bytes: &[u8]) -> Handle {
        let obj = self.realm.new_object();
        let store = self.realm.new_bytes(bytes.to_vec());
        self.realm
            .set_hidden_property(obj, ARRAY_BUFFER_BYTES, NanBox::handle(store.to_raw()));
        self.link_array_buffer_proto(obj);
        obj
    }

    /// Builds a JS-visible `ArrayBuffer` object that wraps an **external**,
    /// caller-owned memory region `[ptr, ptr+len)` **zero-copy**: JS reads and
    /// writes (through typed-array/`DataView` views) hit the region in place, and
    /// `free` (if any) runs when the buffer's `Cell::Bytes` is collected.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for reads and writes of `len` bytes until
    /// `free` is invoked (or, if `free` is `None`, for as long as the resulting
    /// buffer — or any view over it — remains reachable). No other mutable alias
    /// to the region may be used while the engine holds it. See
    /// [`Realm::wrap_external_bytes`](crate::realm::Realm::wrap_external_bytes).
    #[allow(unsafe_code)]
    pub unsafe fn array_buffer_from_external(
        &mut self,
        ptr: *mut u8,
        len: usize,
        free: Option<crate::cell::ExternFree>,
    ) -> Handle {
        let obj = self.realm.new_object();
        // SAFETY: forwarded to the caller's contract documented above.
        #[allow(unsafe_code)]
        let store = unsafe { self.realm.wrap_external_bytes(ptr, len, free) };
        self.realm
            .set_hidden_property(obj, ARRAY_BUFFER_BYTES, NanBox::handle(store.to_raw()));
        self.link_array_buffer_proto(obj);
        obj
    }

    /// The contiguous `Cell::Bytes` store handle backing the `ArrayBuffer` object
    /// `buffer`, if it is one — so an embedder can read it back via
    /// [`realm`](Self::realm)`.bytes_at(..)` or mutate it via `bytes_at_mut`.
    #[must_use]
    pub fn array_buffer_bytes_handle(&self, buffer: Handle) -> Option<Handle> {
        self.array_buffer_bytes(buffer)
    }

    /// Builds a typed-array view of element-`kind` over `buffer` (an `ArrayBuffer`
    /// object from [`array_buffer_from_bytes`](Self::array_buffer_from_bytes) /
    /// [`array_buffer_from_external`](Self::array_buffer_from_external) or JS),
    /// spanning `length` elements starting at byte `offset`. `kind` is the
    /// engine's element-kind index (its element size is
    /// [`typed_elem_size`](crate::realm::typed_elem_size); e.g. `1` = `Uint8`,
    /// `8` = `Float64`). Returns `None` if `buffer` is not an `ArrayBuffer` object.
    /// `.buffer` on the view returns `buffer` itself (SameValue-stable, shared).
    pub fn typed_array_over(
        &mut self,
        buffer: Handle,
        kind: u8,
        offset: usize,
        length: usize,
    ) -> Option<Handle> {
        let bytes_h = self.array_buffer_bytes(buffer)?;
        Some(
            self.realm
                .new_typed_array(bytes_h, buffer, offset, length, kind),
        )
    }

    /// Writes `value` to element `i` of `handle`. For a typed-array view this
    /// coerces to the element kind and writes through to the shared bytes (handled
    /// intrinsically by [`Realm::set_element`]); for a plain array it is an ordinary
    /// element store.
    pub(crate) fn set_element_coerced(
        &mut self,
        handle: crate::heap::Handle,
        i: usize,
        value: NanBox,
    ) {
        self.realm.set_element(handle, i, value);
    }

    /// `fromIndex` coercion for `indexOf`/`includes` (forward search): a missing
    /// argument is `0`; otherwise `ToIntegerOrInfinity` (abrupt-propagating — a
    /// Symbol/BigInt or an abrupt `valueOf` throws), then a negative counts from
    /// `len` (floored at 0) and the result is clamped to `len`.
    pub(crate) fn array_from_index_checked(
        &mut self,
        v: NanBox,
        len: usize,
    ) -> Result<usize, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(0);
        }
        let n = self.coerce_to_integer_or_infinity(v)?;
        Ok(if n < 0.0 {
            (len as f64 + n).max(0.0) as usize
        } else if n >= len as f64 {
            len
        } else {
            n as usize
        })
    }

    /// `%TypedArray%.from(source, mapfn, thisArg)` (23.2.2.1), generic over the
    /// `this` constructor `ctor` (so `Int8Array.from(...)` and `%TypedArray%.from`
    /// share one spec-faithful path). Order: IsConstructor(ctor) → mapfn callable →
    /// `GetMethod(source, @@iterator)` (invoking a throwing getter) → iterator path
    /// (drain to a list, then `TypedArrayCreate`, then map+Set each) or array-like
    /// path (`ToObject`, `LengthOfArrayLike`, `TypedArrayCreate` *before* visiting
    /// elements, then per-index Get/map/Set). Every step propagates abruptly.
    pub(crate) fn typed_array_from(
        &mut self,
        ctor: NanBox,
        source: NanBox,
        mapfn: NanBox,
        this_arg: NanBox,
    ) -> Result<NanBox, ExecError> {
        if !self.is_constructor_value(ctor) {
            return Err(self.type_error("TypedArray.from requires a constructor this"));
        }
        let has_mapfn = !matches!(mapfn.unpack(), Unpacked::Undefined);
        if has_mapfn
            && !mapfn
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return Err(self.type_error("TypedArray.from mapfn is not a function"));
        }
        // Step 4: `usingIterator = ? GetMethod(source, @@iterator)` — reads the
        // property (invoking an accessor, so a throwing getter propagates);
        // undefined/null → array-like path; a present non-callable is a TypeError.
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        let using_iterator = match source.as_handle().map(Handle::from_raw) {
            Some(sh) => {
                let m = self.read_member(sh, &iter_key)?;
                if matches!(m.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    None
                } else if !m
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("source is not iterable"));
                } else {
                    Some(m)
                }
            }
            None => None,
        };
        let target = if let Some(method) = using_iterator {
            // Steps 5.a–5.c: drain the iterator to a list first, then
            // `TypedArrayCreate(C, « len »)` — build the target after collecting.
            let iterator = self.call_with_this(method, source, &[])?;
            let values = self.drain_iterator_values(iterator)?;
            let target = self.typed_array_create(ctor, values.len())?;
            // Step 5.d: map then `Set` each element, interleaved.
            for (k, v) in values.into_iter().enumerate() {
                let mapped = if has_mapfn {
                    self.call_with_this(mapfn, this_arg, &[v, NanBox::number(k as f64)])?
                } else {
                    v
                };
                self.typed_array_set_index_coerced(target, k, mapped)?;
            }
            target
        } else {
            // Steps 7–13 (array-like): `ToObject(source)`, then
            // `len = ? LengthOfArrayLike(arrayLike)`, `TypedArrayCreate(C, «len»)`
            // *before* visiting elements, then per-index Get / map / Set.
            let obj = self.coerce_to_object(source);
            let Some(oh) = obj.as_handle().map(Handle::from_raw) else {
                return Err(self.type_error("TypedArray.from source is not an object"));
            };
            let len_v = self.read_member(oh, "length")?;
            let len_f = self.coerce_to_integer_or_infinity(len_v)?;
            let len = if len_f <= 0.0 {
                0
            } else if len_f > self.realm.limits.max_array_len as f64 {
                let m = self.new_str("Invalid typed array length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            } else {
                len_f as usize
            };
            let target = self.typed_array_create(ctor, len)?;
            for k in 0..len {
                let kv = self.read_member(oh, &alloc::format!("{k}"))?;
                let mapped = if has_mapfn {
                    self.call_with_this(mapfn, this_arg, &[kv, NanBox::number(k as f64)])?
                } else {
                    kv
                };
                self.typed_array_set_index_coerced(target, k, mapped)?;
            }
            target
        };
        Ok(NanBox::handle(target.to_raw()))
    }

    /// `%TypedArray%.of(...items)` (23.2.2.2), generic over the `this` constructor
    /// `ctor`: `TypedArrayCreate(C, « len »)` (validating a custom `this`), then each
    /// item coerced to the element kind and written through.
    pub(crate) fn typed_array_of(
        &mut self,
        ctor: NanBox,
        items: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        if !self.is_constructor_value(ctor) {
            return Err(self.type_error("TypedArray.of requires a constructor this"));
        }
        let vh = self.typed_array_create(ctor, items.len())?;
        self.guard_view_immutable(vh)?;
        // Step 6 is a per-element `Set(newObj, k, items[k], true)`: coercion and the
        // write are **interleaved** and each `Set` re-validates the index against the
        // live length. A `valueOf` that resizes the backing buffer between elements is
        // thus observed — an out-of-bounds index at its own step is silently skipped
        // (not written from a later, grown state).
        for (k, item) in items.iter().enumerate() {
            self.typed_array_set_index_coerced(vh, k, *item)?;
        }
        Ok(NanBox::handle(vh.to_raw()))
    }

    /// `TypedArrayCreate(constructor, « length »)` (23.2.4.2): `Construct(C, «len»)`,
    /// then ValidateTypedArray (the result must have a `[[TypedArrayName]]` slot and a
    /// non-detached buffer) and — since the sole argument is the Number `len` — its
    /// `[[ArrayLength]]` must be ≥ `len`. Used by `TypedArray.from`/`of` so a custom
    /// `this` constructor that returns a non-typed-array, a detached, or an
    /// undersized instance is a TypeError.
    pub(crate) fn typed_array_create(
        &mut self,
        ctor: NanBox,
        len: usize,
    ) -> Result<Handle, ExecError> {
        let result = self.construct(ctor, &[NanBox::number(len as f64)])?;
        let Some(th) = result.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("TypedArray constructor did not return an object"));
        };
        let Some(alen) = self.realm.typed_len(th) else {
            return Err(self.type_error("TypedArray constructor did not return a TypedArray"));
        };
        if self.typed_array_detached(th) {
            return Err(self.type_error("TypedArray constructor returned a detached typed array"));
        }
        if alen < len {
            return Err(self.type_error("TypedArray constructor result is too small"));
        }
        // `TypedArrayCreateFromConstructor(C, args, write)`: `from`/`of` populate the
        // result, so a view over an *immutable* buffer is rejected here — before any
        // source element is read/mapped (the write would otherwise be the first thing
        // to fail).
        self.guard_view_immutable(th)?;
        Ok(th)
    }

    /// `Set(typedArray, index, value, true)` for an in-range integer index during
    /// `TypedArray.from`/`of` element population: coerces `value` to the element kind
    /// (ToNumber / ToBigInt — an abrupt `valueOf` propagates and interrupts
    /// population), rejects a write through an immutable buffer, then writes through
    /// the shared bytes when the index is still valid.
    pub(crate) fn typed_array_set_index_coerced(
        &mut self,
        target: Handle,
        k: usize,
        value: NanBox,
    ) -> Result<(), ExecError> {
        let coerced = if self.realm.typed_kind(target).is_some_and(is_bigint_kind) {
            self.coerce_typed_array_write(target, value)?
        } else {
            self.coerce_to_number(value)?
        };
        self.guard_view_immutable(target)?;
        if !self.typed_array_detached(target) && self.realm.typed_len(target).is_some_and(|l| k < l)
        {
            self.realm.set_element(target, k, coerced);
        }
        Ok(())
    }

    /// `subarray`'s TypedArraySpeciesCreate(O, « buffer, beginByteOffset,
    /// newLength »). Returns `Some(view)` when a *custom* `Symbol.species`
    /// constructor is used (constructing `new species(buffer, off, len)`);
    /// returns `None` when the default constructor applies, so the caller takes
    /// the fast intrinsic-view path. Errors propagate a non-object `constructor`,
    /// a non-constructor species, or a result that is not a typed array.
    pub(crate) fn typed_subarray_species(
        &mut self,
        recv: Handle,
        buffer: Handle,
        byte_offset: usize,
        new_len: usize,
        pass_length: bool,
    ) -> Result<Option<NanBox>, ExecError> {
        let ctor = self.read_member(recv, "constructor")?;
        if matches!(ctor.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        if !self.is_object_value(ctor) {
            return Err(self.type_error("constructor property is not an object"));
        }
        let ch = ctor.as_handle().map(Handle::from_raw).unwrap();
        // Default concrete TypedArray constructor → fast path.
        if self.realm.native_at(ch).is_some_and(|id| {
            (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id)
        }) {
            return Ok(None);
        }
        let species_sym = self.well_known_symbol("species");
        let species_key = self.member_key(species_sym);
        let species = self.read_member(ch, &species_key)?;
        if matches!(species.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Ok(None);
        }
        if !self.is_constructor_value(species) {
            return Err(self.type_error("Symbol.species is not a constructor"));
        }
        // Per 23.2.3.30 step 15: when O's [[ArrayLength]] is *auto* (a
        // length-tracking view) and `end` is undefined, the argument list is
        // « buffer, beginByteOffset » (no length) — the new view also length-tracks.
        // Otherwise it is « buffer, beginByteOffset, newLength ».
        let args: &[NanBox] = &[
            NanBox::handle(buffer.to_raw()),
            NanBox::number(byte_offset as f64),
            NanBox::number(new_len as f64),
        ];
        let args = if pass_length { args } else { &args[..2] };
        let result = self.construct(species, args)?;
        if result
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.typed_len(h))
            .is_none()
        {
            return Err(self.type_error("Symbol.species did not return a TypedArray"));
        }
        Ok(Some(result))
    }

    /// Spec-faithful relative-index clamp for typed-array mutators/readers:
    /// `undefined` yields
    /// `default`; otherwise `ToIntegerOrInfinity` (which **throws** for a Symbol
    /// or BigInt and propagates an abrupt `valueOf`/`toString`), then a negative
    /// counts from `len` and the result is clamped into `0..=len`. Used by the
    /// typed-array bulk mutators/readers whose relative indices must surface
    /// coercion errors (`fill`/`copyWithin`/`slice`/`indexOf`/…).
    pub(crate) fn typed_clamp_index_checked(
        &mut self,
        v: NanBox,
        default: usize,
        len: usize,
    ) -> Result<usize, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(default);
        }
        let n = self.coerce_to_integer_or_infinity(v)?;
        Ok(if n < 0.0 {
            (len as f64 + n).max(0.0) as usize
        } else {
            // `+Infinity` saturates to `len` (the clamp below caps it).
            if n >= len as f64 { len } else { n as usize }
        })
    }

    /// C1: a user-facing array element write (`arr[i] = v`). Like
    /// [`Self::set_element_coerced`], but when the index would grow the dense
    /// backing past `limits.max_array_len` the realm refuses the write (a silent
    /// no-op that would otherwise lose data invisibly); surface that as a
    /// catchable `RangeError("Invalid array length")` so `a[1e9] = 1` throws
    /// rather than vanishing. Typed-array views (fixed length, out-of-bounds writes
    /// are spec no-ops) and frozen/sealed arrays keep their existing behaviour:
    /// the throw fires only on the dense-array capacity overflow.
    pub(crate) fn set_element_checked(
        &mut self,
        handle: crate::heap::Handle,
        i: usize,
        value: NanBox,
    ) -> Result<(), ExecError> {
        // Only a plain dense array can hit the capacity cap; a typed array's
        // out-of-bounds write is a legitimate no-op, never a RangeError.
        let over_cap = self.realm.typed_len(handle).is_none()
            && self.realm.is_array(handle)
            && i >= self.realm.limits.max_array_len;
        if over_cap {
            // A valid array index (`i <= 2**32 - 2`) beyond the dense storage cap:
            // a plain `arr[i] = v` write never raises "Invalid array length" (that
            // is reserved for an explicit `.length =` past 2**32-1). Store the
            // element sparsely as a named property and bump the logical `length`
            // to `i + 1`, mirroring the array-index `defineProperty` path — the
            // backing `Vec` cannot hold billions of slots, but the property must
            // still exist and `length` grow.
            let key = alloc::format!("{i}");
            self.realm.force_set_property(handle, &key, value);
            let target_len = i + 1;
            if self.realm.array_length(handle).unwrap_or(0) < target_len {
                self.realm.set_array_length(handle, target_len);
            }
            return Ok(());
        }
        // A write to a BigInt typed-array element ToBigInt-coerces the value (a
        // Number throws TypeError) — even for an out-of-bounds index, where the
        // store itself is a no-op but the coercion's side effects/throw still run.
        let value = self.coerce_typed_array_write(handle, value)?;
        // A write through a typed-array view over an immutable buffer is a
        // TypeError (after value coercion). No-op for a plain array.
        self.guard_view_immutable(handle)?;
        self.realm.set_element(handle, i, value);
        Ok(())
    }

    /// C1: a user-facing `arr.length = n`. A length above the uint32 ceiling
    /// (2^32-1) is invalid per spec — surface a catchable
    /// `RangeError("Invalid array length")`. A valid length above the dense
    /// `limits.max_array_len` is stored as a *sparse* logical length by the realm
    /// (no multi-gigabyte allocation), so it succeeds rather than throwing.
    ///
    /// Implements `ArraySetLength`'s shrink semantics: when lowering `length`,
    /// elements are deleted from the top down, stopping at the first
    /// **non-configurable** index (a frozen/sealed array, or one demoted via
    /// `defineProperty`). Returns `Ok(true)` when the requested length was fully
    /// applied, `Ok(false)` when deletion stopped early (the length is left one
    /// above the stuck index) — the caller decides whether that is a TypeError.
    ///
    /// This does **not** check whether `length` itself is non-writable; the callers
    /// that need that (`arr.length =` and the `length` descriptor) validate it first.
    pub(crate) fn set_array_length_checked(
        &mut self,
        handle: crate::heap::Handle,
        n: usize,
    ) -> Result<bool, ExecError> {
        if n as u64 > u64::from(u32::MAX) {
            let m = self.new_str("Invalid array length");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let (all_deleted, _was_array) = self.realm.array_set_length_truncating(handle, n);
        Ok(all_deleted)
    }

    /// Builds a primitive wrapper object (`new Number`/`String`/`Boolean`,
    /// `Object(primitive)`): an object boxing `prim` behind a `\0prim` slot, with
    /// `\0wraptype` recording the constructor id (for `instanceof`).
    /// For a weak collection (`WeakMap`/`WeakSet`), throws a `TypeError` when `key`
    /// is a primitive — weak keys must be objects or symbols. A no-op for a
    /// non-weak (`Map`/`Set`) collection.
    /// Throws a `TypeError` if `buf` is a detached `ArrayBuffer` (one whose data has been
    /// moved out by `transfer()`) — every operation on a detached buffer is an error.
    pub(crate) fn guard_detached_buffer(&mut self, buf: Handle) -> Result<(), ExecError> {
        if self
            .realm
            .get_property(buf, ARRAY_BUFFER_DETACHED)
            .is_some()
        {
            let m = self.new_str("Cannot perform operation on a detached ArrayBuffer");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    /// Whether the `ArrayBuffer` `buf` is immutable (produced by
    /// `transferToImmutable` / `sliceToImmutable`). False for a non-buffer.
    pub(crate) fn is_immutable_buffer(&self, buf: Handle) -> bool {
        self.realm
            .get_property(buf, ARRAY_BUFFER_IMMUTABLE)
            .is_some()
    }

    /// Throws a `TypeError` if the `ArrayBuffer` `buf` is immutable — every
    /// operation that would modify, resize, or transfer its bytes is rejected.
    pub(crate) fn guard_immutable_buffer(&mut self, buf: Handle) -> Result<(), ExecError> {
        if self.is_immutable_buffer(buf) {
            let m = self.new_str("Cannot modify an immutable ArrayBuffer");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    /// Throws a `TypeError` if the view (`DataView` or typed array) `handle` is
    /// backed by an immutable `ArrayBuffer`. Used at the entry of every mutating
    /// view operation, *before* argument coercion, so a poisoned `valueOf` is not
    /// observed when the write is forbidden. A non-view, or a view over a mutable
    /// buffer, passes.
    pub(crate) fn guard_view_immutable(&mut self, handle: Handle) -> Result<(), ExecError> {
        // A typed array exposes its buffer via `typed_array_object`; a `DataView`
        // stores it under `DATA_VIEW_BUF`.
        let buf = self.realm.typed_array_object(handle).or_else(|| {
            self.realm
                .get_property(handle, DATA_VIEW_BUF)
                .and_then(|b| b.as_handle())
                .map(Handle::from_raw)
        });
        if let Some(buf) = buf {
            return self.guard_immutable_buffer(buf);
        }
        Ok(())
    }

    /// Builds the result array for an Array method invoked on `recv`. If `recv` is a
    /// typed array, the result is a same-kind typed array with its elements coerced
    /// to that element type; otherwise an ordinary array.
    /// Writes `elems` back into `handle` in place: per-element for a typed-array
    /// view (writing through the shared bytes, coercing each), or wholesale for a
    /// plain array. Used by in-place reorders (`sort`/`reverse`).
    pub(crate) fn write_back_elements(&mut self, handle: Handle, elems: Vec<NanBox>) {
        if self.realm.typed_kind(handle).is_some() {
            // Bulk write-through: one buffer borrow, no per-element heap lookup.
            self.realm.typed_set_from_numbers(handle, 0, &elems);
        } else {
            self.realm.array_set_all(handle, elems);
        }
    }

    pub(crate) fn typed_like(&mut self, recv: Handle, elems: Vec<NanBox>) -> NanBox {
        if let Some(kind) = self.realm.typed_kind(recv) {
            // A fresh typed array of the same kind over its own backing buffer.
            let elem_size = TYPED_ARRAY_KINDS[kind as usize].1 as usize;
            let buf = self.make_array_buffer(elems.len() * elem_size);
            let bytes_h = self.array_buffer_bytes(buf).unwrap();
            let view = self
                .realm
                .new_typed_array(bytes_h, buf, 0, elems.len(), kind);
            // Link `[[Prototype]]` to the kind's intrinsic (e.g.
            // `%Float64Array.prototype%`) so a same-kind result is a real
            // instance: `result.constructor`, `instanceof`, and prototype
            // identity all match the default constructor's products.
            if let Some(proto) = self.intrinsic_proto(TYPED_ARRAY_KINDS[kind as usize].0) {
                self.realm.set_native_proto(view, proto);
            }
            // Bulk write-through: one buffer borrow, no per-element heap lookup.
            self.realm.typed_set_from_numbers(view, 0, &elems);
            NanBox::handle(view.to_raw())
        } else {
            NanBox::handle(self.realm.new_array(elems).to_raw())
        }
    }

    /// `TypedArraySpeciesCreate(exemplar, « len »)` then fill with `elems`.
    ///
    /// For a typed-array receiver this honors `Symbol.species`: `SpeciesConstructor`
    /// reads `exemplar.constructor` (a non-undefined non-object → TypeError) then
    /// its `[Symbol.species]` (null/undefined → default ctor; a non-constructor →
    /// TypeError); `TypedArrayCreate` does `Construct(C, [len])` then
    /// ValidateTypedArray (result must be a typed array of length ≥ len); finally
    /// `elems` are written in (coercing per its element kind).
    ///
    /// When `exemplar.constructor`/species resolve to the built-in default, this
    /// `TypedArraySpeciesCreate(exemplar, « len »)` returning the *result view*
    /// (zero-filled, length `len`) — used by `map`/`filter` which must allocate the
    /// destination *before* iterating (so a throwing species getter/ctor aborts
    /// before the callback runs). Resolves `exemplar.constructor` then its
    /// `[Symbol.species]` (undefined/null → default; non-constructor → TypeError),
    /// `Construct(C, [len])`, then ValidateTypedArray (a typed array of length ≥ len).
    pub(crate) fn typed_species_create(
        &mut self,
        recv: Handle,
        len: usize,
    ) -> Result<Handle, ExecError> {
        let Some(kind) = self.realm.typed_kind(recv) else {
            return Err(self.type_error("not a typed array"));
        };
        // SpeciesConstructor(exemplar, defaultConstructor): Get(O,"constructor");
        // undefined → default; a non-Object (incl. a string/symbol/bigint, which
        // are heap-backed here but are *not* Objects) → TypeError.
        let ctor = self.read_member(recv, "constructor")?;
        let species = if matches!(ctor.unpack(), Unpacked::Undefined) {
            None
        } else if !self.is_object_value(ctor) {
            return Err(self.type_error("constructor property is not an object"));
        } else {
            let ch = ctor.as_handle().map(Handle::from_raw).unwrap();
            // The fast same-kind path applies only when `constructor` is the
            // receiver's *own* kind constructor (e.g. a `Float64Array`'s
            // `%Float64Array%`). A `constructor` overridden to a **different** built-in
            // typed-array constructor must go through the species path so the result
            // has that other kind (`Construct(C, «len»)`), not the receiver's.
            let recv_kind = self.realm.typed_kind(recv).unwrap_or(0);
            let is_default =
                self.realm.native_at(ch) == Some(N_TYPED_ARRAY_BASE + recv_kind as u16);
            if is_default {
                None
            } else {
                let species_sym = self.well_known_symbol("species");
                let species_key = self.member_key(species_sym);
                let s = self.read_member(ch, &species_key)?;
                match s.unpack() {
                    Unpacked::Undefined | Unpacked::Null => None,
                    _ => Some(s),
                }
            }
        };
        let Some(species) = species else {
            // Default path: a same-kind, intrinsic-proto view over its own buffer.
            let elem_size = TYPED_ARRAY_KINDS[kind as usize].1 as usize;
            let buf = self.make_array_buffer(len * elem_size);
            let bytes_h = self.array_buffer_bytes(buf).unwrap();
            let view = self.realm.new_typed_array(bytes_h, buf, 0, len, kind);
            if let Some(proto) = self.intrinsic_proto(TYPED_ARRAY_KINDS[kind as usize].0) {
                self.realm.set_native_proto(view, proto);
            }
            return Ok(view);
        };
        if !self.is_constructor_value(species) {
            return Err(self.type_error("Symbol.species is not a constructor"));
        }
        let result = self.construct(species, &[NanBox::number(len as f64)])?;
        let Some(rh) = result.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("TypedArray species constructor did not return an object"));
        };
        let Some(rlen) = self.realm.typed_len(rh) else {
            return Err(self.type_error("Symbol.species did not return a TypedArray"));
        };
        if rlen < len {
            return Err(self.type_error("TypedArray species constructor result is too small"));
        }
        Ok(rh)
    }

    /// degenerates to [`Self::typed_like`] (a same-kind view). A plain-array
    /// receiver just builds an ordinary array (Array species is handled elsewhere).
    pub(crate) fn typed_like_species(
        &mut self,
        recv: Handle,
        elems: Vec<NanBox>,
    ) -> Result<NanBox, ExecError> {
        if self.realm.typed_kind(recv).is_none() {
            // A plain-array receiver: `map`/`filter` allocate the result via
            // `ArraySpeciesCreate(O, len)` and populate each *present* element with
            // `CreateDataPropertyOrThrow` (a hole in `elems` stays a hole). The
            // common default-Array species takes a bulk write-through fast path.
            let n = elems.len();
            let a_v = self.array_species_create(recv, n)?;
            let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
                return Err(self.type_error("Array species did not return an object"));
            };
            // The bulk `set_element` write-through is only sound when the result is
            // a *pristine* dense array (as `ArrayCreate(n)` produces): same length,
            // no frozen/sealed/per-index-descriptor overrides. A custom `@@species`
            // may hand back an array that already carries indices with non-default
            // attributes (e.g. a non-writable/non-enumerable data slot), which must
            // be overwritten via `CreateDataPropertyOrThrow` (a full DefineOwnProperty
            // that resets attributes) — not a raw value store that preserves them.
            let default_array = self.realm.is_array(a_h)
                && self.realm.array_length(a_h) == Some(n)
                && !self.realm.array_has_index_overrides(a_h);
            for (i, e) in elems.iter().enumerate() {
                if e.is_hole() {
                    continue;
                }
                if default_array {
                    self.realm.set_element(a_h, i, *e);
                } else {
                    self.create_data_property_or_throw(a_h, i, *e)?;
                }
            }
            let len_key = self.new_str("length");
            self.assign_member_value(a_h, len_key, NanBox::number(n as f64))?;
            return Ok(a_v);
        }
        // TypedArraySpeciesCreate(O, «len») then write the produced elements
        // through (coercing to the result's kind).
        let rh = self.typed_species_create(recv, elems.len())?;
        // A species ctor that returns a view over an immutable buffer makes the
        // populating writes fail — a TypeError after the result is constructed.
        self.guard_view_immutable(rh)?;
        self.realm.typed_set_from_numbers(rh, 0, &elems);
        Ok(NanBox::handle(rh.to_raw()))
    }

    /// Validates a `padStart`/`padEnd` target length: a result longer than
    /// `MAX_STRING_LEN` is an unrepresentable string, a `RangeError`. A negative,
    /// `NaN`, or zero target is clamped to 0 (the source is returned unchanged).
    pub(crate) fn pad_target(&mut self, n: f64) -> Result<usize, ExecError> {
        if n > self.realm.limits.max_string_len as f64 {
            let m = self.new_str("Invalid string length");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(if n.is_nan() || n < 0.0 { 0 } else { n as usize })
    }

    /// Validates an untrusted length/byte count `n` (from a typed-array,
    /// `ArrayBuffer`, or WASM constructor) before it drives an allocation:
    /// rejects negative, non-integer, and over-cap values with a `RangeError`,
    /// returning the value as a `usize`. The dense NanBox-backed model amplifies
    /// each slot 8×, so an uncapped length would alloc-abort the process.
    pub(crate) fn validate_alloc_len(&mut self, n: f64, what: &str) -> Result<usize, ExecError> {
        // `floor()` is std-only; once `n` is finite, non-negative, and within the
        // cap, the `usize` round-trip is a core-friendly integrality check.
        if !n.is_finite()
            || n < 0.0
            || n > self.realm.limits.max_array_len as f64
            || (n as usize as f64) != n
        {
            let m = self.new_str(what);
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(n as usize)
    }

    /// Builds an `ArrayBuffer` object of `len` zeroed bytes — a contiguous
    /// [`Cell::Bytes`] store under the hidden `ARRAY_BUFFER_BYTES` slot.
    pub(crate) fn make_array_buffer(&mut self, len: usize) -> Handle {
        let obj = self.realm.new_object();
        let bytes = self.realm.new_bytes(alloc::vec![0u8; len]);
        self.realm
            .set_hidden_property(obj, ARRAY_BUFFER_BYTES, NanBox::handle(bytes.to_raw()));
        self.link_array_buffer_proto(obj);
        obj
    }

    /// An `ArrayBuffer` whose contiguous [`Cell::Bytes`] store is a copy of `bytes`.
    pub(crate) fn make_array_buffer_from_bytes(&mut self, bytes: &[u8]) -> Handle {
        self.array_buffer_from_bytes(bytes)
    }

    /// Allocates the result of `ArrayBuffer.prototype.slice` through
    /// `SpeciesConstructor(O, %ArrayBuffer%)` and copies `data` (already the final
    /// `new_len` bytes) into it. A default / `undefined` species takes the fast
    /// `make_array_buffer_from_bytes` path; a subclass species is `Construct`ed
    /// (returning a subclass instance), then validated as a distinct ArrayBuffer
    /// of at least `new_len` bytes.
    pub(crate) fn array_buffer_species_new(
        &mut self,
        original: Handle,
        data: &[u8],
        new_len: usize,
    ) -> Result<Handle, ExecError> {
        // SpeciesConstructor(O, %ArrayBuffer%): `Get(O, "constructor")`, then
        // `C[@@species]` (undefined/null → the default constructor).
        let mut c = self.read_member(original, "constructor")?;
        // C must be an *Object* — a heap-backed primitive (string/symbol/bigint)
        // has a handle but is not an object, so filter on `is_object_value` (else
        // a String `constructor` would be treated as an object below).
        if let Some(ch) = c
            .as_handle()
            .map(Handle::from_raw)
            .filter(|_| self.is_object_value(c))
        {
            let sym = self.well_known_symbol("species");
            let key = self.member_key(sym);
            let s = self.read_member(ch, &key)?;
            c = if matches!(s.unpack(), Unpacked::Undefined | Unpacked::Null) {
                NanBox::undefined()
            } else {
                s
            };
        } else if !matches!(c.unpack(), Unpacked::Undefined) {
            return Err(
                self.type_error("ArrayBuffer.prototype.slice: constructor is not an object")
            );
        }
        let is_default = matches!(c.unpack(), Unpacked::Undefined)
            || self.current.get("ArrayBuffer").and_then(|v| v.as_handle()) == c.as_handle();
        if is_default {
            return Ok(self.make_array_buffer_from_bytes(data));
        }
        if !self.is_constructor_value(c) {
            return Err(
                self.type_error("ArrayBuffer.prototype.slice: species is not a constructor")
            );
        }
        // `Construct(C, «new_len»)` — a distinct, non-shared ArrayBuffer with at
        // least `new_len` bytes; then copy the sliced data in.
        let created = self.construct(c, &[NanBox::number(new_len as f64)])?;
        let Some(nh) = created.as_handle().map(Handle::from_raw) else {
            return Err(
                self.type_error("ArrayBuffer.prototype.slice: species did not return an object")
            );
        };
        if nh == original {
            return Err(
                self.type_error("ArrayBuffer.prototype.slice: species returned the source buffer")
            );
        }
        let Some(nb_bytes) = self.array_buffer_bytes(nh) else {
            return Err(self
                .type_error("ArrayBuffer.prototype.slice: species did not return an ArrayBuffer"));
        };
        if self.realm.bytes_at(nb_bytes).map_or(0, <[u8]>::len) < new_len {
            return Err(self.type_error("ArrayBuffer.prototype.slice: species buffer is too small"));
        }
        if let Some(dst) = self.realm.bytes_at_mut(nb_bytes) {
            let n = data.len().min(dst.len());
            dst[..n].copy_from_slice(&data[..n]);
        }
        Ok(nh)
    }

    /// `%ArrayBuffer.prototype%`, the `[[Prototype]]` every `ArrayBuffer` object
    /// inherits (so its methods, accessors, and `Symbol.toStringTag` resolve through
    /// the chain). `None` only before `install_globals` has run.
    pub(crate) fn array_buffer_proto(&mut self) -> Option<Handle> {
        self.current
            .get("ArrayBuffer")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
    }

    /// Links `buf` to `%ArrayBuffer.prototype%` (no-op if it is not yet installed).
    fn link_array_buffer_proto(&mut self, buf: Handle) {
        if let Some(proto) = self.array_buffer_proto() {
            self.realm.set_object_proto(buf, Some(proto));
        }
    }

    /// Links a freshly-built typed-array view's `[[Prototype]]` to the concrete
    /// constructor's `.prototype` — the *newTarget*'s under `Reflect.construct` /
    /// `TA.of`/`from` with a subclass, else the kind's own constructor prototype — so
    /// `result.constructor`, `Object.getPrototypeOf(result)`, and inherited members
    /// resolve. (Typed-array views are non-object cells, so the proto lives in the
    /// realm's `native_protos` side table.)
    /// `GetPrototypeFromConstructor(newTarget, %TAKind.prototype%)` for the
    /// typed-array constructor, run at *AllocateTypedArray* time — i.e. **before**
    /// any argument coercion. Unlike a raw `get_property` read of the stored
    /// `prototype` (which cannot observe a throwing accessor), this performs a real
    /// `Get(newTarget, "prototype")`, invoking an
    /// own `prototype` getter and propagating an abrupt completion. It is only
    /// consulted when `newTarget` differs from the callee (a subclass `super()`, a
    /// `Reflect.construct` newTarget). Returns the resolved object prototype, or the
    /// kind's intrinsic default when `newTarget`'s `prototype` is a non-object.
    pub(crate) fn typed_newtarget_proto(
        &mut self,
        kind: u8,
        callee: NanBox,
        new_target: NanBox,
    ) -> Result<Option<Handle>, ExecError> {
        let kind_name = TYPED_ARRAY_KINDS[kind as usize].0;
        let default = self.intrinsic_proto(kind_name);
        self.instance_proto_checked(new_target, callee, default)
    }

    /// Whether the typed-array view at `handle` is backed by a detached buffer
    /// (so an integer-indexed `[[Get]]/[[Set]]/[[Has]]/[[Delete]]` reads/writes
    /// nothing). False for a non-view.
    pub(crate) fn typed_array_detached(&self, handle: Handle) -> bool {
        self.realm.typed_array_object(handle).is_some_and(|buf| {
            self.realm
                .get_property(buf, ARRAY_BUFFER_DETACHED)
                .is_some()
        })
    }

    /// Performs the abstract `DetachArrayBuffer(buf)`: zero-lengths the backing
    /// store, empties every typed-array view over it (length 0), and flags the
    /// buffer detached so subsequent operations throw / read 0. Idempotent.
    pub(crate) fn detach_array_buffer(&mut self, buf: Handle) {
        if let Some(bh) = self.array_buffer_bytes(buf) {
            self.realm.detach_buffer_views(bh);
            self.realm.bytes_resize(bh, 0);
        }
        self.realm
            .set_hidden_property(buf, ARRAY_BUFFER_DETACHED, NanBox::boolean(true));
    }

    /// The contiguous byte store handle of the `ArrayBuffer` object `buf`, if it has one.
    pub(crate) fn array_buffer_bytes(&self, buf: Handle) -> Option<Handle> {
        self.realm
            .get_property(buf, ARRAY_BUFFER_BYTES)
            .and_then(|b| b.as_handle())
            .map(Handle::from_raw)
    }

    /// If `target` is a `BigInt64Array`/`BigUint64Array`, ToBigInt-coerce `value`
    /// (throwing `TypeError` for a Number, per spec) and return the resulting
    /// BigInt as a heap value ready to store; otherwise return `value` unchanged.
    /// The single chokepoint every typed-array element write funnels through so a
    /// Number assigned to a BigInt element throws rather than silently writing 0.
    pub(crate) fn coerce_typed_array_write(
        &mut self,
        target: Handle,
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        if self.realm.typed_kind(target).is_some_and(is_bigint_kind) {
            let big = self.coerce_to_bigint(value)?;
            return Ok(NanBox::handle(self.realm.new_bigint(big).to_raw()));
        }
        Ok(value)
    }
}
