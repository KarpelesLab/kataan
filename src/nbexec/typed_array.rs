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
            let m = self.new_str("Invalid array length");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        // A write to a BigInt typed-array element ToBigInt-coerces the value (a
        // Number throws TypeError) — even for an out-of-bounds index, where the
        // store itself is a no-op but the coercion's side effects/throw still run.
        let value = self.coerce_typed_array_write(handle, value)?;
        self.realm.set_element(handle, i, value);
        Ok(())
    }

    /// C1: a user-facing `arr.length = n`. A length above the uint32 ceiling
    /// (2^32-1) is invalid per spec — surface a catchable
    /// `RangeError("Invalid array length")`. A valid length above the dense
    /// `limits.max_array_len` is stored as a *sparse* logical length by the realm
    /// (no multi-gigabyte allocation), so it succeeds rather than throwing.
    pub(crate) fn set_array_length_checked(
        &mut self,
        handle: crate::heap::Handle,
        n: usize,
    ) -> Result<(), ExecError> {
        if n as u64 > u64::from(u32::MAX) {
            let m = self.new_str("Invalid array length");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        self.realm.set_array_length(handle, n);
        Ok(())
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
            // Bulk write-through: one buffer borrow, no per-element heap lookup.
            self.realm.typed_set_from_numbers(view, 0, &elems);
            NanBox::handle(view.to_raw())
        } else {
            NanBox::handle(self.realm.new_array(elems).to_raw())
        }
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
        obj
    }

    /// An `ArrayBuffer` whose contiguous [`Cell::Bytes`] store is a copy of `bytes`.
    pub(crate) fn make_array_buffer_from_bytes(&mut self, bytes: &[u8]) -> Handle {
        self.array_buffer_from_bytes(bytes)
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
