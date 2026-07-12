use super::*;

impl<'a> Interp<'a> {
    /// Runs an ES2025 `Iterator.prototype` helper (`map`/`filter`/`take`/`drop`/
    /// `flatMap`/`reduce`/`toArray`/`forEach`/`some`/`every`/`find`) on the
    /// receiver iterator `this_val` (GetIteratorDirect: `this` must be an Object
    /// and is stepped through its `next` method). The lazy helpers
    /// (`map`/`filter`/`take`/`drop`/`flatMap`) return a fresh
    /// `%IteratorHelperPrototype%` object that pulls from the source on demand (so
    /// they are lazy, interleave with direct `.next()`, and work on infinite
    /// iterators); the consuming helpers drive the source to completion (closing it
    /// on early exit / abrupt completion). A non-object `this` is a TypeError.
    pub(crate) fn iterator_proto_helper(
        &mut self,
        method: &str,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        // GetIteratorDirect: `this` must be an Object; its `next` method is read
        // once. (A non-object — including a primitive `this` — is a TypeError.)
        if !self.is_object_value(this_val) {
            return Err(self.type_error(&alloc::format!(
                "Iterator.prototype.{method} requires that 'this' be an Object"
            )));
        }
        let this_h = this_val.as_handle().map(Handle::from_raw).unwrap();
        let needs_fn = matches!(
            method,
            "map" | "filter" | "flatMap" | "reduce" | "forEach" | "some" | "every" | "find"
        );
        let f = args.first().copied().unwrap_or(NanBox::undefined());
        // The lazy helpers validate their argument *before* reading `next` per
        // spec; on failure of the limit/callback coercion they must close `this`.
        match method {
            "map" | "filter" | "flatMap" => {
                if !f
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    self.iterator_close(this_h)?;
                    return Err(self.type_error(&alloc::format!(
                        "Iterator.prototype.{method} called with a non-callable argument"
                    )));
                }
                return self.make_iter_helper(method, this_val, Some(f), 0.0);
            }
            "take" | "drop" => {
                // ToNumber(limit). A NaN numLimit is a RangeError; otherwise the
                // *integer* limit (ToIntegerOrInfinity, truncating toward zero) must
                // be >= 0 — so `take(-0.5)` → 0 (allowed), `take(-1)` throws. Close
                // the iterator on any throw.
                let n = match self.coerce_to_number(f) {
                    Ok(n) => self.realm.to_number(n),
                    Err(e) => {
                        let _ = self.iterator_close(this_h);
                        return Err(e);
                    }
                };
                // trunc_toward_zero maps NaN to 0, so check NaN explicitly first.
                let lim = trunc_toward_zero(n);
                if n.is_nan() || lim < 0.0 {
                    let _ = self.iterator_close(this_h);
                    let m = self.new_str(&alloc::format!(
                        "Iterator.prototype.{method} limit must be a non-negative number"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                return self.make_iter_helper(method, this_val, None, lim);
            }
            _ => {}
        }
        // The consuming helpers require a callable first argument up front.
        if needs_fn
            && !f
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            self.iterator_close(this_h)?;
            return Err(self.type_error(&alloc::format!(
                "Iterator.prototype.{method} called with a non-callable argument"
            )));
        }
        // Consuming helpers: pull from `this` lazily via its `next` method, so a
        // user iterator's side effects and closing semantics are observed.
        let next = self.read_member(this_h, "next")?;
        match method {
            "toArray" => {
                let mut out = Vec::new();
                while let Some(v) = self.iter_step(this_h, next)? {
                    out.push(v);
                }
                Ok(NanBox::handle(self.realm.new_array(out).to_raw()))
            }
            "forEach" => {
                let mut i = 0u64;
                while let Some(v) = self.iter_step(this_h, next)? {
                    let r = self.call(f, &[v, NanBox::number(i as f64)]);
                    if let Err(e) = r {
                        let _ = self.iterator_close(this_h);
                        return Err(e);
                    }
                    i += 1;
                }
                Ok(NanBox::undefined())
            }
            "some" | "every" | "find" => {
                let mut i = 0u64;
                while let Some(v) = self.iter_step(this_h, next)? {
                    let r = match self.call(f, &[v, NanBox::number(i as f64)]) {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = self.iterator_close(this_h);
                            return Err(e);
                        }
                    };
                    let t = self.realm.truthy(r);
                    match method {
                        "every" if !t => {
                            self.iterator_close(this_h)?;
                            return Ok(NanBox::boolean(false));
                        }
                        "some" if t => {
                            self.iterator_close(this_h)?;
                            return Ok(NanBox::boolean(true));
                        }
                        "find" if t => {
                            self.iterator_close(this_h)?;
                            return Ok(v);
                        }
                        _ => {}
                    }
                    i += 1;
                }
                Ok(match method {
                    "every" => NanBox::boolean(true),
                    "some" => NanBox::boolean(false),
                    _ => NanBox::undefined(),
                })
            }
            // reduce
            _ => {
                let mut acc;
                let mut i = 0u64;
                if args.len() >= 2 {
                    acc = args[1];
                } else {
                    match self.iter_step(this_h, next)? {
                        Some(v) => {
                            acc = v;
                            i = 1;
                        }
                        None => {
                            return Err(
                                self.type_error("Reduce of empty iterator with no initial value")
                            );
                        }
                    }
                }
                while let Some(v) = self.iter_step(this_h, next)? {
                    acc = match self.call(f, &[acc, v, NanBox::number(i as f64)]) {
                        Ok(a) => a,
                        Err(e) => {
                            let _ = self.iterator_close(this_h);
                            return Err(e);
                        }
                    };
                    i += 1;
                }
                Ok(acc)
            }
        }
    }

    /// One step of the iterator protocol on iterator object `it` using its cached
    /// `next` method: returns `Some(value)` for `{done:false}`, `None` once done.
    /// The result must be an object (else a TypeError).
    pub(crate) fn iter_step(
        &mut self,
        it: Handle,
        next: NanBox,
    ) -> Result<Option<NanBox>, ExecError> {
        let res = self.call_with_this(next, NanBox::handle(it.to_raw()), &[])?;
        if !self.is_object_value(res) {
            return Err(self.type_error("iterator result is not an object"));
        }
        let rh = Handle::from_raw(res.as_handle().unwrap());
        let done = self.read_member(rh, "done")?;
        if self.realm.truthy(done) {
            return Ok(None);
        }
        Ok(Some(self.read_member(rh, "value")?))
    }

    /// Builds a lazy `%IteratorHelperPrototype%`-based helper object for `map`,
    /// `filter`, `take`, `drop`, or `flatMap`, capturing the underlying iterator
    /// `source` (its `next` read once via GetIteratorDirect) and the callback /
    /// limit. The helper's `next` (see [`Self::iter_helper_next`]) drives the
    /// transformation on demand.
    fn make_iter_helper(
        &mut self,
        kind: &str,
        source: NanBox,
        f: Option<NanBox>,
        limit: f64,
    ) -> Result<NanBox, ExecError> {
        let src_h = source.as_handle().map(Handle::from_raw).unwrap();
        let next = self.read_member(src_h, "next")?;
        let proto = self.iter_helper_proto();
        let h = self.realm.new_object_with_proto(proto);
        let kind_str = self.new_str(kind);
        self.realm.set_hidden_property(h, HELPER_KIND, kind_str);
        self.realm.set_hidden_property(h, HELPER_SOURCE, source);
        self.realm.set_hidden_property(h, HELPER_NEXT, next);
        if let Some(fv) = f {
            self.realm.set_hidden_property(h, HELPER_FN, fv);
        }
        self.realm
            .set_hidden_property(h, HELPER_LIMIT, NanBox::number(limit));
        self.realm
            .set_hidden_property(h, HELPER_COUNTER, NanBox::number(0.0));
        Ok(NanBox::handle(h.to_raw()))
    }

    /// The eager-generator iterator's `next`: advances the hidden buffer cursor,
    /// surfacing the `return` value once after the yields are exhausted.
    pub(crate) fn gen_iter_next(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Generator.prototype.next called on non-object"));
        };
        // A **live** Set/Map iterator (has `GEN_COLL`) re-reads the collection.
        if let Some(coll) = self
            .realm
            .get_property(h, GEN_COLL)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return self.live_collection_iter_next(h, coll);
        }
        // A **live** typed-array iterator (has `GEN_TA`) re-reads its live length.
        if let Some(ta) = self
            .realm
            .get_property(h, GEN_TA)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return self.live_typed_iter_next(h, ta);
        }
        // A **live** plain-array iterator (has `GEN_ARR`) re-reads its live length.
        if let Some(arr) = self
            .realm
            .get_property(h, GEN_ARR)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return self.live_array_iter_next(h, arr);
        }
        // A **lazy** RegExp String Iterator (has `RSI_MATCHER`) calls RegExpExec
        // on each `next()`.
        if self.realm.get_property(h, RSI_MATCHER).is_some() {
            return self.regexp_string_iter_next(h);
        }
        let Some(buf) = self
            .realm
            .get_property(h, GEN_BUF)
            .and_then(|b| b.as_handle())
            .map(Handle::from_raw)
        else {
            return Err(self.type_error("Generator.prototype.next called on a non-generator"));
        };
        let idx = self
            .realm
            .get_property(h, GEN_IDX)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as usize;
        let elems = self.realm.array_elements(buf).map(<[_]>::to_vec);
        let len = elems.as_ref().map_or(0, Vec::len);
        let (value, done) = match elems.as_ref().and_then(|e| e.get(idx)) {
            Some(v) => {
                self.realm
                    .set_hidden_property(h, GEN_IDX, NanBox::number((idx + 1) as f64));
                (*v, false)
            }
            None => {
                let v = if idx == len {
                    self.realm
                        .set_hidden_property(h, GEN_IDX, NanBox::number((idx + 1) as f64));
                    self.realm
                        .get_property(h, GEN_RET)
                        .unwrap_or(NanBox::undefined())
                } else {
                    NanBox::undefined()
                };
                (v, true)
            }
        };
        Ok(self.iter_result(value, done))
    }

    /// `next()` for a **live** Set/Map iterator (see `make_live_collection_iterator`):
    /// re-reads the collection so a mutation mid-iteration is observed. Resumes
    /// after the last-yielded key (found by `SameValueZero`); if that key was
    /// deleted, resumes from its recorded position (the successor after the
    /// compacting delete); once the end is reached the iterator detaches (`GEN_DONE`).
    pub(crate) fn live_collection_iter_next(
        &mut self,
        h: Handle,
        coll: Handle,
    ) -> Result<NanBox, ExecError> {
        // Once detached (exhausted), stay done regardless of later mutation.
        if self.realm.get_property(h, GEN_DONE).is_some() {
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        let kind = self
            .realm
            .get_property(h, GEN_KIND)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as u8;
        let is_set = self.realm.collection_is_set(coll) == Some(true);
        let entries = self.realm.collection_entries(coll).unwrap_or_default();
        // Determine the next position: 0 if not started, else after the
        // last-yielded key (or its recorded slot if it was since deleted).
        let recorded_idx = self
            .realm
            .get_property(h, GEN_IDX)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as usize;
        let next_pos = match self.realm.get_property(h, GEN_LASTKEY) {
            None => 0,
            Some(last_key) => match entries
                .iter()
                .position(|(k, _)| self.realm.same_value_zero(*k, last_key))
            {
                // The last-yielded key is still at (or before) its recorded slot —
                // a pure delete only shifts survivors left — so advance past it.
                Some(q) if q <= recorded_idx => q + 1,
                // Found only at a *later* slot than recorded: the key was deleted
                // and re-added (a brand-new entry appended at the end). Treat the
                // original as deleted and resume from its recorded slot (now the
                // successor after the compacting delete); the re-added copy is a
                // fresh entry that the cursor reaches later.
                Some(_) => recorded_idx,
                // Deleted (and not re-added): resume from the recorded slot.
                None => recorded_idx,
            },
        };
        let Some(&(k, v)) = entries.get(next_pos) else {
            self.realm
                .set_hidden_property(h, GEN_DONE, NanBox::boolean(true));
            return Ok(self.iter_result(NanBox::undefined(), true));
        };
        self.realm.set_hidden_property(h, GEN_LASTKEY, k);
        self.realm
            .set_hidden_property(h, GEN_IDX, NanBox::number(next_pos as f64));
        let value = match kind {
            0 => k,                        // keys
            2 => self.new_iter_pair(k, v), // entries: [key, value]
            _ => {
                if is_set {
                    k
                } else {
                    v
                }
            } // values (a Set yields its element)
        };
        Ok(self.iter_result(value, false))
    }

    /// A fresh 2-element `[key, value]` array for a Map/Set `entries()` result.
    fn new_iter_pair(&mut self, k: NanBox, v: NanBox) -> NanBox {
        NanBox::handle(self.realm.new_array(alloc::vec![k, v]).to_raw())
    }

    /// `next()` for a **live** typed-array iterator (see `make_live_typed_iterator`):
    /// re-reads the live length each step, so a resizable-buffer resize or an
    /// element write mid-iteration is observed. Iterates canonical integer indices
    /// `0..length`; an index that falls out of the (possibly shrunk) view yields
    /// `undefined`.
    pub(crate) fn live_typed_iter_next(
        &mut self,
        h: Handle,
        ta: Handle,
    ) -> Result<NanBox, ExecError> {
        // Once the cursor has passed the (live) length the iterator has completed
        // (`CreateArrayIterator`'s closure returned); it stays done even if the
        // backing buffer is later grown back in bounds.
        if self.realm.get_property(h, GEN_DONE).is_some() {
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        let kind = self
            .realm
            .get_property(h, GEN_KIND)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as u8;
        let idx = self
            .realm
            .get_property(h, GEN_IDX)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as usize;
        // Spec: each `next()` re-derives the buffer witness; if the view's
        // fixed-length range now exceeds a shrunk resizable buffer (or its buffer
        // was detached), `IsTypedArrayOutOfBounds` is true → throw a `TypeError`.
        // A *length-tracking* view re-spans instead (never out of bounds unless its
        // start offset itself is past the end), so it iterates its shrunk length.
        if self.typed_array_detached(ta) || self.realm.typed_array_out_of_bounds(ta) {
            return Err(self.type_error(
                "TypedArray iterator: the backing ArrayBuffer is out of bounds or detached",
            ));
        }
        // The live length (tracks a resizable backing buffer).
        let len = self.realm.typed_len(ta).unwrap_or(0);
        if idx >= len {
            self.realm
                .set_hidden_property(h, GEN_DONE, NanBox::boolean(true));
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        self.realm
            .set_hidden_property(h, GEN_IDX, NanBox::number((idx + 1) as f64));
        let value = match kind {
            0 => NanBox::number(idx as f64), // keys
            2 => {
                // entries: [index, element]
                let e = self
                    .realm
                    .typed_get(ta, idx)
                    .unwrap_or_else(NanBox::undefined);
                self.new_iter_pair(NanBox::number(idx as f64), e)
            }
            _ => self
                .realm
                .typed_get(ta, idx)
                .unwrap_or_else(NanBox::undefined),
        };
        Ok(self.iter_result(value, false))
    }

    /// `next()` for a **live** plain-array iterator (see `make_live_array_iterator`):
    /// re-reads `length` each step and `Get`s the element at the cursor, so a value
    /// appended/assigned after the iterator was created is observed. Once the cursor
    /// reaches the current length the iterator is exhausted (and stays done even if
    /// the array later grows again — matching the spec's monotonic index cursor).
    pub(crate) fn live_array_iter_next(
        &mut self,
        h: Handle,
        arr: Handle,
    ) -> Result<NanBox, ExecError> {
        // Once the cursor first reaches the (then-current) length, the underlying
        // generator completes; it stays done thereafter even if the array later
        // grows (a monotonic, latched cursor — matching `CreateArrayIterator`).
        if self.realm.get_property(h, GEN_DONE).is_some() {
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        let kind = self
            .realm
            .get_property(h, GEN_KIND)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as u8;
        let idx = self
            .realm
            .get_property(h, GEN_IDX)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as usize;
        let len = self.realm.array_length(arr).unwrap_or(0);
        if idx >= len {
            self.realm
                .set_hidden_property(h, GEN_DONE, NanBox::boolean(true));
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        self.realm
            .set_hidden_property(h, GEN_IDX, NanBox::number((idx + 1) as f64));
        let value = match kind {
            0 => NanBox::number(idx as f64), // keys
            2 => {
                // entries: [index, element]
                let e = self.read_member(arr, &alloc::format!("{idx}"))?;
                self.new_iter_pair(NanBox::number(idx as f64), e)
            }
            _ => self.read_member(arr, &alloc::format!("{idx}"))?, // values
        };
        Ok(self.iter_result(value, false))
    }

    /// The eager-generator iterator's `return(v)`: mark exhausted, report done.
    pub(crate) fn gen_iter_return(&mut self, this: NanBox, v: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw)
            && let Some(buf) = self
                .realm
                .get_property(h, GEN_BUF)
                .and_then(|b| b.as_handle())
                .map(Handle::from_raw)
        {
            let len = self.realm.array_elements(buf).map_or(0, <[_]>::len);
            self.realm
                .set_hidden_property(h, GEN_IDX, NanBox::number(len as f64));
        }
        Ok(self.iter_result(v, true))
    }

    /// Builds an iterator-result object `{ value, done }`.
    pub(crate) fn iter_result(&mut self, value: NanBox, done: bool) -> NanBox {
        let r = self.realm.new_object();
        self.realm.set_property(r, "value", value);
        self.realm.set_property(r, "done", NanBox::boolean(done));
        NanBox::handle(r.to_raw())
    }

    /// `%IteratorHelperPrototype%.next` — advances a lazy helper one step,
    /// pulling from the underlying iterator on demand.
    pub(crate) fn iter_helper_next(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        // Reentrancy guard (GeneratorValidate: if the helper is already executing,
        // throw a TypeError). Set the running flag, run the body, and clear it on
        // *every* path — capture the result before clearing (no `?`) so a thrown
        // body cannot leave the helper stuck "running".
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return self.iter_helper_next_body(this);
        };
        // `Iterator.zip`/`zipKeyed` results share `%IteratorHelperPrototype%` but
        // drive several underlying iterators — route to the dedicated stepper.
        if self.realm.get_property(h, ZIP_ITERS).is_some() {
            return self.iter_zip_next(this);
        }
        if self
            .realm
            .get_property(h, HELPER_RUNNING)
            .is_some_and(|v| self.realm.truthy(v))
        {
            return Err(self.type_error("Iterator Helper is already running"));
        }
        self.realm
            .set_hidden_property(h, HELPER_RUNNING, NanBox::boolean(true));
        let result = self.iter_helper_next_body(this);
        self.realm.delete_property(h, HELPER_RUNNING);
        result
    }

    fn iter_helper_next_body(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Iterator Helper next called on non-object"));
        };
        // A helper marked done returns `{ value: undefined, done: true }`.
        if self.realm.get_property(h, HELPER_DONE).is_some() {
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        let kind = self
            .realm
            .get_property(h, HELPER_KIND)
            .and_then(|k| k.as_handle())
            .map(Handle::from_raw)
            .and_then(|kh| self.realm.string_value(kh))
            .unwrap_or_default();
        let src = self
            .realm
            .get_property(h, HELPER_SOURCE)
            .unwrap_or(NanBox::undefined());
        let src_h = match src.as_handle().map(Handle::from_raw) {
            Some(s) => s,
            None => {
                self.mark_helper_done(h);
                return Ok(self.iter_result(NanBox::undefined(), true));
            }
        };
        let next = self
            .realm
            .get_property(h, HELPER_NEXT)
            .unwrap_or(NanBox::undefined());
        let f = self.realm.get_property(h, HELPER_FN);
        let result = self.iter_helper_step(h, &kind, src_h, next, f);
        match result {
            Ok(Some(v)) => Ok(self.iter_result(v, false)),
            Ok(None) => {
                self.mark_helper_done(h);
                Ok(self.iter_result(NanBox::undefined(), true))
            }
            Err(e) => {
                // An abrupt completion from the body marks the helper done (the
                // underlying iterator is treated as closed by the throw).
                self.mark_helper_done(h);
                Err(e)
            }
        }
    }

    fn mark_helper_done(&mut self, h: Handle) {
        self.realm
            .set_hidden_property(h, HELPER_DONE, NanBox::boolean(true));
    }

    /// Produces the next yielded value for a lazy helper (or `None` at end).
    fn iter_helper_step(
        &mut self,
        h: Handle,
        kind: &str,
        src_h: Handle,
        next: NanBox,
        f: Option<NanBox>,
    ) -> Result<Option<NanBox>, ExecError> {
        match kind {
            "take" => {
                let rem = self
                    .realm
                    .get_property(h, HELPER_LIMIT)
                    .and_then(|n| n.as_number())
                    .unwrap_or(0.0);
                if rem <= 0.0 {
                    // Limit reached: close the underlying iterator.
                    self.iterator_close(src_h)?;
                    return Ok(None);
                }
                self.realm
                    .set_hidden_property(h, HELPER_LIMIT, NanBox::number(rem - 1.0));
                self.iter_step(src_h, next)
            }
            "drop" => {
                let mut rem = self
                    .realm
                    .get_property(h, HELPER_LIMIT)
                    .and_then(|n| n.as_number())
                    .unwrap_or(0.0);
                while rem > 0.0 {
                    if self.iter_step(src_h, next)?.is_none() {
                        return Ok(None);
                    }
                    rem -= 1.0;
                }
                self.realm
                    .set_hidden_property(h, HELPER_LIMIT, NanBox::number(0.0));
                self.iter_step(src_h, next)
            }
            "map" => match self.iter_step(src_h, next)? {
                Some(v) => {
                    let c = self.helper_counter_incr(h);
                    let r = self.call_helper_cb(src_h, f, &[v, NanBox::number(c)])?;
                    Ok(Some(r))
                }
                None => Ok(None),
            },
            "filter" => loop {
                match self.iter_step(src_h, next)? {
                    Some(v) => {
                        let c = self.helper_counter_incr(h);
                        let keep = self.call_helper_cb(src_h, f, &[v, NanBox::number(c)])?;
                        if self.realm.truthy(keep) {
                            return Ok(Some(v));
                        }
                    }
                    None => return Ok(None),
                }
            },
            // flatMap
            _ => {
                loop {
                    // Drain the current inner iterator first, if any.
                    if let Some(inner) = self
                        .realm
                        .get_property(h, HELPER_INNER)
                        .and_then(|v| v.as_handle())
                        .map(Handle::from_raw)
                    {
                        let inner_next = self
                            .realm
                            .get_property(h, HELPER_INNER_NEXT)
                            .unwrap_or(NanBox::undefined());
                        match self.iter_step(inner, inner_next)? {
                            Some(v) => return Ok(Some(v)),
                            None => {
                                self.realm.set_hidden_property(
                                    h,
                                    HELPER_INNER,
                                    NanBox::undefined(),
                                );
                                self.realm.delete_property(h, HELPER_INNER);
                            }
                        }
                    }
                    // Pull the next outer value and open its inner iterator.
                    match self.iter_step(src_h, next)? {
                        Some(v) => {
                            let c = self.helper_counter_incr(h);
                            let mapped = self.call_helper_cb(src_h, f, &[v, NanBox::number(c)])?;
                            let inner = self.get_iterator_flattenable(mapped, src_h)?;
                            let inner_next = self.read_member(inner, "next")?;
                            self.realm.set_hidden_property(
                                h,
                                HELPER_INNER,
                                NanBox::handle(inner.to_raw()),
                            );
                            self.realm
                                .set_hidden_property(h, HELPER_INNER_NEXT, inner_next);
                        }
                        None => return Ok(None),
                    }
                }
            }
        }
    }

    fn helper_counter_incr(&mut self, h: Handle) -> f64 {
        let c = self
            .realm
            .get_property(h, HELPER_COUNTER)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0);
        self.realm
            .set_hidden_property(h, HELPER_COUNTER, NanBox::number(c + 1.0));
        c
    }

    /// Invokes a helper callback; on an abrupt completion the underlying iterator
    /// is closed (then the error re-propagates).
    fn call_helper_cb(
        &mut self,
        src_h: Handle,
        f: Option<NanBox>,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let f = f.unwrap_or(NanBox::undefined());
        match self.call(f, args) {
            Ok(v) => Ok(v),
            Err(e) => {
                let _ = self.iterator_close(src_h);
                Err(e)
            }
        }
    }

    /// `GetIteratorFlattenable(value, REJECT_PRIMITIVES)` for flatMap's inner step:
    /// the mapped value must be an object; if it has `[Symbol.iterator]` use it,
    /// else treat the object itself as an iterator. On a non-object the source is
    /// closed and a TypeError thrown.
    fn get_iterator_flattenable(
        &mut self,
        value: NanBox,
        src_h: Handle,
    ) -> Result<Handle, ExecError> {
        // GetIteratorFlattenable with REJECT_PRIMITIVES: a primitive (including a
        // string) is a TypeError; the source is closed first.
        if !self.is_object_value(value) {
            let _ = self.iterator_close(src_h);
            return Err(self.type_error("flatMap mapper must return an object"));
        }
        let vh = value.as_handle().map(Handle::from_raw).unwrap();
        // GetMethod(value, @@iterator).
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        let mut method = match self.read_member(vh, &iter_key) {
            Ok(m) => m,
            Err(e) => {
                let _ = self.iterator_close(src_h);
                return Err(e);
            }
        };
        if matches!(method.unpack(), Unpacked::Undefined | Unpacked::Null)
            && let Ok(Some(m)) = self.class_iterator_method(vh)
        {
            method = m;
        }
        match method.unpack() {
            // No `@@iterator`: a built-in iterable (array / Map / Set / generator)
            // drains; any other object is used directly as the iterator.
            Unpacked::Undefined | Unpacked::Null => {
                let is_builtin_iterable = self.realm.array_elements(vh).is_some()
                    || self.realm.collection_entries(vh).is_some()
                    || self.realm.get_property(vh, GEN_BUF).is_some();
                if is_builtin_iterable {
                    match self.get_iter_object(value) {
                        Ok(ih) => Ok(ih),
                        Err(e) => {
                            let _ = self.iterator_close(src_h);
                            Err(e)
                        }
                    }
                } else {
                    Ok(vh)
                }
            }
            // Present but not callable → close src + TypeError (GetMethod step 3).
            _ => {
                if !method
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    let _ = self.iterator_close(src_h);
                    return Err(self.type_error("flatMap: Symbol.iterator is not a function"));
                }
                let iterator = match self.call_with_this(method, value, &[]) {
                    Ok(it) => it,
                    Err(e) => {
                        let _ = self.iterator_close(src_h);
                        return Err(e);
                    }
                };
                if self.is_object_value(iterator) {
                    Ok(iterator.as_handle().map(Handle::from_raw).unwrap())
                } else {
                    let _ = self.iterator_close(src_h);
                    Err(self.type_error("flatMap: the iterator is not an object"))
                }
            }
        }
    }

    /// `%IteratorHelperPrototype%.return` — closes the helper (and its source).
    pub(crate) fn iter_helper_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw) {
            // Zip/zipKeyed results share this prototype — close all sub-iterators.
            if self.realm.get_property(h, ZIP_ITERS).is_some() {
                return self.iter_zip_return(this);
            }
            let already_done = self.realm.get_property(h, HELPER_DONE).is_some();
            self.mark_helper_done(h);
            if !already_done {
                // For `flatMap`, an inner iterator may still be open — close it
                // first (its `return` is forwarded), then close the source.
                if let Some(inner) = self
                    .realm
                    .get_property(h, HELPER_INNER)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                {
                    self.realm.delete_property(h, HELPER_INNER);
                    self.iterator_close(inner)?;
                }
                if let Some(src) = self
                    .realm
                    .get_property(h, HELPER_SOURCE)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                {
                    self.iterator_close(src)?;
                }
            }
        }
        Ok(self.iter_result(NanBox::undefined(), true))
    }

    /// The cached `%IteratorHelperPrototype%`, read from the `Iterator` ctor slot.
    fn iter_helper_proto(&mut self) -> Option<Handle> {
        self.iter_ctor_slot(ITER_HELPER_PROTO_SLOT)
    }

    fn iter_ctor_slot(&mut self, slot: &str) -> Option<Handle> {
        self.current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, slot))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
    }

    /// `Iterator.from(O)`: if `O` is a string, iterate it; if `O` is already an
    /// object with a callable `next` *and* it inherits `%IteratorPrototype%`,
    /// return it unchanged; otherwise wrap it in a `%WrapForValidIterator%`.
    pub(crate) fn iterator_from(&mut self, src: NanBox) -> Result<NanBox, ExecError> {
        // GetIteratorFlattenable(src, iterate-string-primitives): `src` must be an
        // Object or a primitive String; any other primitive is a TypeError.
        let src_h = src.as_handle().map(Handle::from_raw);
        let is_string_prim = src_h.is_some_and(|h| self.realm.string_value(h).is_some());
        if !self.is_object_value(src) && !is_string_prim {
            let m = self.new_str("Iterator.from called on a non-object");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // Both an Object and a primitive String flow through the same GetMethod
        // path below: for a primitive String, `src` is a heap string cell used as
        // its own receiver, so `read_member`/`call_with_this` fire a redefined
        // `String.prototype[@@iterator]` getter/method with a *string* receiver
        // (`GetV` primitive-receiver semantics), while a `String` wrapper observes
        // an object receiver.
        let h = src_h.unwrap();
        // method = GetMethod(src, @@iterator): the getter fires with `this` = src
        // (so a `String.prototype[@@iterator]` getter observes a string receiver
        // for a primitive, an object receiver for a wrapper).
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        let mut method = if is_string_prim {
            // `GetV(src, @@iterator)` for a *primitive* String: walk the wrapper's
            // prototype chain (`String.prototype`) for the property, firing an
            // accessor getter with the **primitive** as the receiver — so a
            // redefined `String.prototype[@@iterator]` getter observes
            // `typeof this === "string"`. The built-in `@@iterator` is not a
            // materialized property, so an unmodified chain yields `undefined`
            // here and falls through to the built-in string iteration below.
            let wrapper = self.coerce_to_object(src);
            let mut cur = wrapper.as_handle().map(Handle::from_raw);
            let mut m = NanBox::undefined();
            while let Some(o) = cur {
                if let Some((getter, _)) = self.realm.accessor(o, &iter_key) {
                    if !matches!(getter.unpack(), Unpacked::Undefined) {
                        m = self.call_with_this(getter, src, &[])?;
                    }
                    break;
                }
                if self.realm.has_own(o, &iter_key) {
                    m = self
                        .realm
                        .get_property(o, &iter_key)
                        .unwrap_or_else(NanBox::undefined);
                    break;
                }
                cur = self.realm.object_proto(o);
            }
            m
        } else {
            self.read_member(h, &iter_key)?
        };
        // A class computed-key `[Symbol.iterator]() {}` may not surface as a
        // readable property; fall back to scanning the class body.
        if matches!(method.unpack(), Unpacked::Undefined | Unpacked::Null)
            && let Some(m) = self.class_iterator_method(h)?
        {
            method = m;
        }
        let iterator = match method.unpack() {
            Unpacked::Undefined | Unpacked::Null => {
                // No `@@iterator` (GetMethod → undefined): a built-in iterable
                // (array / string-wrapper / Map / Set / generator) drains to a
                // generator; any other object is used directly as the iterator
                // (GetIteratorDirect — no `next` validation at this step).
                let is_string_wrapper = self
                    .realm
                    .get_property(h, PRIM_WRAP)
                    .and_then(|p| p.as_handle())
                    .map(Handle::from_raw)
                    .is_some_and(|ph| self.realm.string_value(ph).is_some());
                let is_builtin_iterable = is_string_wrapper
                    || is_string_prim
                    || self.realm.array_elements(h).is_some()
                    || self.realm.collection_entries(h).is_some()
                    || self.realm.get_property(h, GEN_BUF).is_some();
                if is_builtin_iterable {
                    let vals = self.iterate_values(src)?;
                    self.make_generator(vals)
                } else {
                    src
                }
            }
            _ => {
                // Present but not callable → TypeError (GetMethod step 3).
                if !method
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("Iterator.from: Symbol.iterator is not a function"));
                }
                self.call_with_this(method, src, &[])?
            }
        };
        // Step 5: the iterator must be an Object.
        if !self.is_object_value(iterator) {
            return Err(self.type_error("Iterator.from: not an iterator"));
        }
        let ih = iterator.as_handle().map(Handle::from_raw).unwrap();
        // If the iterator already inherits `%IteratorPrototype%`, return it as-is.
        if self.inherits_iterator_proto(ih) {
            return Ok(iterator);
        }
        self.wrap_iterator_value(iterator)
    }

    /// Wraps an iterator object in a fresh `%WrapForValidIterator%` whose
    /// `next`/`return` forward to the wrapped iterator.
    fn wrap_iterator_value(&mut self, iterator: NanBox) -> Result<NanBox, ExecError> {
        let ih = iterator.as_handle().map(Handle::from_raw).unwrap();
        let next = self.read_member(ih, "next")?;
        let proto = self.iter_ctor_slot(ITER_WRAP_PROTO_SLOT);
        let h = self.realm.new_object_with_proto(proto);
        self.realm.set_hidden_property(h, HELPER_SOURCE, iterator);
        self.realm.set_hidden_property(h, HELPER_NEXT, next);
        Ok(NanBox::handle(h.to_raw()))
    }

    /// Whether `h` inherits the generic `Array.prototype` methods through its
    /// `[[Prototype]]` chain — true when an actual `Array` (e.g. an object whose
    /// prototype was set to `[...]`) or the realm's `Array.prototype` itself is
    /// in the chain. Used to decide, for a direct `obj.reduce(...)` call, whether
    /// `obj` should be treated as an array-like (its inherited array method runs)
    /// rather than reporting "reduce is not a function".
    pub(crate) fn inherits_array_proto(&mut self, h: Handle) -> bool {
        let array_proto = self
            .current
            .get("Array")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        let mut cur = self.realm.object_proto(h);
        while let Some(c) = cur {
            if Some(c) == array_proto || self.realm.is_array(c) {
                return true;
            }
            cur = self.realm.object_proto(c);
        }
        false
    }

    /// Whether `h`'s prototype chain includes `%IteratorPrototype%`.
    pub(crate) fn inherits_iterator_proto(&mut self, h: Handle) -> bool {
        let Some(iter_proto) = self
            .current
            .get("Iterator")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        else {
            return false;
        };
        let mut cur = self.realm.object_proto(h);
        while let Some(c) = cur {
            if c == iter_proto {
                return true;
            }
            cur = self.realm.object_proto(c);
        }
        false
    }

    /// `%WrapForValidIteratorPrototype%.next` — forwards to the wrapped `next`.
    /// Requires the `[[Iterated]]` internal slot (`HELPER_SOURCE`).
    pub(crate) fn iter_wrap_next(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("next called on non-object"));
        };
        // RequireInternalSlot(O, [[Iterated]]).
        let Some(src) = self.realm.get_property(h, HELPER_SOURCE) else {
            return Err(self
                .type_error("%WrapForValidIteratorPrototype%.next requires an [[Iterated]] slot"));
        };
        let next = self
            .realm
            .get_property(h, HELPER_NEXT)
            .unwrap_or(NanBox::undefined());
        self.call_with_this(next, src, &[])
    }

    /// `%WrapForValidIteratorPrototype%.return` — forwards to the wrapped iterator's
    /// `return` (if callable), else returns `{ value: undefined, done: true }`.
    /// Requires the `[[Iterated]]` internal slot (`HELPER_SOURCE`).
    pub(crate) fn iter_wrap_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(
                self.type_error("%WrapForValidIteratorPrototype%.return requires an object")
            );
        };
        // RequireInternalSlot(O, [[Iterated]]) — a plain object (no wrapped
        // iterator) throws a TypeError *before* any user `return` is read/called.
        let Some(src) = self
            .realm
            .get_property(h, HELPER_SOURCE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return Err(self.type_error(
                "%WrapForValidIteratorPrototype%.return requires an [[Iterated]] slot",
            ));
        };
        let ret = self.read_member(src, "return")?;
        if ret
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return self.call_with_this(ret, NanBox::handle(src.to_raw()), &[]);
        }
        Ok(self.iter_result(NanBox::undefined(), true))
    }

    /// `Iterator.concat(...items)`: each item must be an object with a
    /// `[Symbol.iterator]` method (read eagerly, in order); the result is a lazy
    /// iterator that yields all of the first's values, then the second's, etc.
    pub(crate) fn iterator_concat(&mut self, items: &[NanBox]) -> Result<NanBox, ExecError> {
        // Validate every argument up front (per spec: each item's `@@iterator`
        // method is read via GetMethod *once*, in order, before producing the
        // result). Store the captured methods so iteration re-invokes them instead
        // of re-reading `@@iterator` (a getter must fire exactly once).
        let mut sources: Vec<NanBox> = Vec::with_capacity(items.len());
        let mut methods: Vec<NanBox> = Vec::with_capacity(items.len());
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        for it in items {
            if !self.is_object_value(*it) {
                return Err(self.type_error("Iterator.concat argument is not an object"));
            }
            let h = it.as_handle().map(Handle::from_raw).unwrap();
            // GetMethod(item, @@iterator): the getter fires here, once.
            let mut method = self.read_member(h, &iter_key)?;
            if matches!(method.unpack(), Unpacked::Undefined | Unpacked::Null)
                && let Some(m) = self.class_iterator_method(h)?
            {
                method = m;
            }
            let has_method = match method.unpack() {
                Unpacked::Undefined | Unpacked::Null => false,
                _ => {
                    // Present but not callable → TypeError (GetMethod step 3).
                    if !method
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                    {
                        return Err(
                            self.type_error("Iterator.concat: Symbol.iterator is not a function")
                        );
                    }
                    true
                }
            };
            // A built-in iterable (array/string/Map/Set/generator) exposes no
            // readable `@@iterator`; it is drained directly at iteration time.
            let is_builtin_iterable = self.realm.array_elements(h).is_some()
                || self.realm.string_value(h).is_some()
                || self.realm.collection_entries(h).is_some()
                || self.realm.get_property(h, GEN_BUF).is_some();
            if !has_method && !is_builtin_iterable {
                return Err(self.type_error("Iterator.concat argument is not iterable"));
            }
            sources.push(*it);
            methods.push(if has_method {
                method
            } else {
                NanBox::undefined()
            });
        }
        let arr = self.realm.new_array(sources);
        let methods_arr = self.realm.new_array(methods);
        let proto = self.iter_ctor_slot(ITER_CONCAT_PROTO_SLOT);
        let h = self.realm.new_object_with_proto(proto);
        self.realm
            .set_hidden_property(h, HELPER_SOURCE, NanBox::handle(arr.to_raw()));
        self.realm
            .set_hidden_property(h, HELPER_METHODS, NanBox::handle(methods_arr.to_raw()));
        self.realm
            .set_hidden_property(h, HELPER_COUNTER, NanBox::number(0.0));
        Ok(NanBox::handle(h.to_raw()))
    }

    /// Returns an iterator *object* for an iterable `value`: invokes its
    /// `[Symbol.iterator]` if callable, else (for a built-in iterable) drains it
    /// into a fresh generator. Errors if `value` is not iterable.
    pub(crate) fn get_iter_object(&mut self, value: NanBox) -> Result<Handle, ExecError> {
        let Some(vh) = value.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("value is not iterable"));
        };
        let iter_fn = self.find_iterator_fn(vh)?;
        if let Some(fv) = iter_fn
            && fv
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            let it = self.call_with_this(fv, value, &[])?;
            return match it.as_handle().map(Handle::from_raw) {
                Some(ih) => Ok(ih),
                None => Err(self.type_error("[Symbol.iterator] did not return an object")),
            };
        }
        if self.realm.array_elements(vh).is_some()
            || self.realm.string_value(vh).is_some()
            || self.realm.collection_entries(vh).is_some()
            || self.realm.get_property(vh, GEN_BUF).is_some()
        {
            let vals = self.iterate_values(value)?;
            let g = self.make_generator(vals);
            return Ok(g.as_handle().map(Handle::from_raw).unwrap());
        }
        Err(self.type_error("value is not iterable"))
    }

    /// `Iterator.zip(iterables, options)` / `Iterator.zipKeyed(iterables, options)`
    /// (the `joint-iteration` proposal). Reads the options (`mode`, and — only for
    /// `"longest"` — `padding`) *before* touching the iterables, opens each
    /// underlying iterator in order (`GetIteratorFlattenable`, interleaved with the
    /// iteration of `iterables` itself for the positional form), then returns a lazy
    /// `%IteratorHelperPrototype%` result whose `next` drives all sub-iterators one
    /// step at a time honoring the shortest/longest/strict mode. On any abrupt
    /// completion while opening, the already-opened iterators are closed in reverse.
    pub(crate) fn iterator_zip(
        &mut self,
        iterables: NanBox,
        options: NanBox,
        keyed: bool,
    ) -> Result<NanBox, ExecError> {
        let what = if keyed {
            "Iterator.zipKeyed"
        } else {
            "Iterator.zip"
        };
        // 1. iterables must be an Object.
        if !self.is_object_value(iterables) {
            return Err(self.type_error(&alloc::format!("{what}: iterables is not an object")));
        }
        // 2. GetOptionsObject(options): undefined → treated as absent; else Object.
        let opts_h = match options.unpack() {
            Unpacked::Undefined => None,
            _ => {
                if !self.is_object_value(options) {
                    return Err(
                        self.type_error(&alloc::format!("{what}: options is not an object"))
                    );
                }
                Some(options.as_handle().map(Handle::from_raw).unwrap())
            }
        };
        // 3-5. mode = Get(options, "mode"); default "shortest". The value must be
        // exactly one of the three primitive strings (no coercion, String wrappers
        // rejected) or undefined.
        let mode_val = match opts_h {
            Some(oh) => self.read_member(oh, "mode")?,
            None => NanBox::undefined(),
        };
        let mode = if matches!(mode_val.unpack(), Unpacked::Undefined) {
            0u8
        } else if !self.is_object_value(mode_val)
            && let Some(s) = mode_val
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|mh| self.realm.string_value(mh))
        {
            match s.as_str() {
                "shortest" => 0,
                "longest" => 1,
                "strict" => 2,
                _ => {
                    return Err(self.type_error(&alloc::format!(
                        "{what}: mode must be 'shortest', 'longest', or 'strict'"
                    )));
                }
            }
        } else {
            return Err(self.type_error(&alloc::format!(
                "{what}: mode must be 'shortest', 'longest', or 'strict'"
            )));
        };
        // 6-7. paddingOption = Get(options, "padding") — read *before* the iterables,
        // and only when mode is "longest". Must be undefined or an Object.
        let mut padding_opt = NanBox::undefined();
        if mode == 1
            && let Some(oh) = opts_h
        {
            padding_opt = self.read_member(oh, "padding")?;
            if !matches!(padding_opt.unpack(), Unpacked::Undefined)
                && !self.is_object_value(padding_opt)
            {
                return Err(self.type_error(&alloc::format!("{what}: padding is not an object")));
            }
        }
        // 8-12. Open each underlying iterator (in order). The positional form
        // iterates `iterables` itself; the keyed form walks its own enumerable keys.
        let mut iters: Vec<NanBox> = Vec::new();
        let mut nexts: Vec<NanBox> = Vec::new();
        let mut keys: Vec<NanBox> = Vec::new();
        if keyed {
            let ih = iterables.as_handle().map(Handle::from_raw).unwrap();
            self.zip_open_keyed(ih, &mut iters, &mut nexts, &mut keys)?;
        } else {
            self.zip_open_positional(iterables, &mut iters, &mut nexts)?;
        }
        let iter_count = iters.len();
        // 14. padding (longest mode only).
        let mut padding: Vec<NanBox> = Vec::new();
        if mode == 1 {
            if matches!(padding_opt.unpack(), Unpacked::Undefined) {
                padding.resize(iter_count, NanBox::undefined());
            } else if keyed {
                // Per-key Get(paddingOption, key).
                let ph = padding_opt.as_handle().map(Handle::from_raw).unwrap();
                for key in &keys {
                    let name = self.member_key(*key);
                    match self.read_member(ph, &name) {
                        Ok(v) => padding.push(v),
                        Err(e) => return Err(self.zip_close_throw(&iters, e)),
                    }
                }
            } else {
                self.zip_iterate_padding(padding_opt, &iters, &mut padding, iter_count)?;
            }
        }
        while padding.len() < iter_count {
            padding.push(NanBox::undefined());
        }
        // Build the lazy result object (a `%IteratorHelperPrototype%` instance).
        let proto = self.iter_helper_proto();
        let h = self.realm.new_object_with_proto(proto);
        let iters_arr = self.realm.new_array(iters);
        let nexts_arr = self.realm.new_array(nexts);
        let pad_arr = self.realm.new_array(padding);
        let fin: Vec<NanBox> = (0..iter_count).map(|_| NanBox::boolean(false)).collect();
        let fin_arr = self.realm.new_array(fin);
        self.realm
            .set_hidden_property(h, ZIP_ITERS, NanBox::handle(iters_arr.to_raw()));
        self.realm
            .set_hidden_property(h, ZIP_NEXTS, NanBox::handle(nexts_arr.to_raw()));
        self.realm
            .set_hidden_property(h, ZIP_MODE, NanBox::number(f64::from(mode)));
        self.realm
            .set_hidden_property(h, ZIP_PADDING, NanBox::handle(pad_arr.to_raw()));
        self.realm
            .set_hidden_property(h, ZIP_FINISHED, NanBox::handle(fin_arr.to_raw()));
        if keyed {
            let keys_arr = self.realm.new_array(keys);
            self.realm
                .set_hidden_property(h, ZIP_KEYS, NanBox::handle(keys_arr.to_raw()));
        }
        Ok(NanBox::handle(h.to_raw()))
    }

    /// GetIterator(iterables) then, interleaved, `IteratorStepValue` +
    /// `GetIteratorFlattenable` for each element (positional `Iterator.zip`). On an
    /// abrupt step the opened iterators are closed (reverse); on an abrupt flatten
    /// the opened iterators *and* the input iterator are closed (reverse of
    /// « inputIter » ++ iters).
    fn zip_open_positional(
        &mut self,
        iterables: NanBox,
        iters: &mut Vec<NanBox>,
        nexts: &mut Vec<NanBox>,
    ) -> Result<(), ExecError> {
        let input_ih = self.get_iter_object(iterables)?;
        let input_next = self.read_member(input_ih, "next")?;
        loop {
            match self.iter_step(input_ih, input_next) {
                Ok(None) => break,
                Ok(Some(v)) => match self.zip_get_iterator_flattenable(v) {
                    Ok((ih, next)) => {
                        iters.push(NanBox::handle(ih.to_raw()));
                        nexts.push(next);
                    }
                    Err(e) => {
                        // « inputIter » ++ iters, reverse: iters (reverse) then input.
                        let err = self.zip_close_throw(iters, e);
                        let _ = self.iterator_close(input_ih);
                        return Err(err);
                    }
                },
                Err(e) => return Err(self.zip_close_throw(iters, e)),
            }
        }
        Ok(())
    }

    /// Walks `iterables`' own keys ([[OwnPropertyKeys]] order, strings then symbols),
    /// and for each *enumerable* key whose value is not undefined opens an iterator
    /// (`GetIteratorFlattenable`) — the keyed `Iterator.zipKeyed` form. On any abrupt
    /// completion the already-opened iterators are closed in reverse.
    fn zip_open_keyed(
        &mut self,
        iterables: Handle,
        iters: &mut Vec<NanBox>,
        nexts: &mut Vec<NanBox>,
        keys: &mut Vec<NanBox>,
    ) -> Result<(), ExecError> {
        let all_keys = self.own_property_keys_values(iterables)?;
        for key in all_keys {
            let name = self.member_key(key);
            let desc = match self.descriptor_of(iterables, &name) {
                Ok(d) => d,
                Err(e) => return Err(self.zip_close_throw(iters, e)),
            };
            if matches!(desc.unpack(), Unpacked::Undefined) {
                continue;
            }
            let enumerable = desc
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|dh| self.realm.get_property(dh, "enumerable"))
                .is_some_and(|v| self.realm.truthy(v));
            if !enumerable {
                continue;
            }
            let value = match self.read_member(iterables, &name) {
                Ok(v) => v,
                Err(e) => return Err(self.zip_close_throw(iters, e)),
            };
            if matches!(value.unpack(), Unpacked::Undefined) {
                continue;
            }
            match self.zip_get_iterator_flattenable(value) {
                Ok((ih, next)) => {
                    keys.push(key);
                    iters.push(NanBox::handle(ih.to_raw()));
                    nexts.push(next);
                }
                Err(e) => return Err(self.zip_close_throw(iters, e)),
            }
        }
        Ok(())
    }

    /// `GetIteratorFlattenable(value, reject-primitives)`: the value must be an
    /// Object; use its `[Symbol.iterator]` if present (call it), else the object
    /// itself is the iterator. Returns the iterator and its (once-read) `next`.
    fn zip_get_iterator_flattenable(
        &mut self,
        value: NanBox,
    ) -> Result<(Handle, NanBox), ExecError> {
        if !self.is_object_value(value) {
            return Err(self.type_error("Iterator.zip: an iterable is not an object"));
        }
        let vh = value.as_handle().map(Handle::from_raw).unwrap();
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        let mut method = self.read_member(vh, &iter_key)?;
        if matches!(method.unpack(), Unpacked::Undefined | Unpacked::Null)
            && let Some(m) = self.class_iterator_method(vh)?
        {
            method = m;
        }
        let iterator = match method.unpack() {
            Unpacked::Undefined | Unpacked::Null => value,
            _ => {
                if !method
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("Iterator.zip: Symbol.iterator is not a function"));
                }
                self.call_with_this(method, value, &[])?
            }
        };
        if !self.is_object_value(iterator) {
            return Err(self.type_error("Iterator.zip: the iterator is not an object"));
        }
        let ih = iterator.as_handle().map(Handle::from_raw).unwrap();
        let next = self.read_member(ih, "next")?;
        Ok((ih, next))
    }

    /// Iterates the `padding` iterable (positional longest mode) exactly `iter_count`
    /// times, filling any remaining slots (and every slot after it is exhausted) with
    /// undefined; if the padding iterator was not exhausted it is closed. An abrupt
    /// completion closes the opened sub-iterators (reverse) and propagates.
    fn zip_iterate_padding(
        &mut self,
        padding_opt: NanBox,
        iters: &[NanBox],
        padding: &mut Vec<NanBox>,
        iter_count: usize,
    ) -> Result<(), ExecError> {
        let pad_ih = match self.get_iter_object(padding_opt) {
            Ok(h) => h,
            Err(e) => return Err(self.zip_close_throw(iters, e)),
        };
        let pad_next = match self.read_member(pad_ih, "next") {
            Ok(n) => n,
            Err(e) => return Err(self.zip_close_throw(iters, e)),
        };
        let mut using = true;
        for _ in 0..iter_count {
            if using {
                match self.iter_step(pad_ih, pad_next) {
                    Ok(Some(v)) => padding.push(v),
                    Ok(None) => {
                        using = false;
                        padding.push(NanBox::undefined());
                    }
                    Err(e) => return Err(self.zip_close_throw(iters, e)),
                }
            } else {
                padding.push(NanBox::undefined());
            }
        }
        if using && let Err(e) = self.iterator_close(pad_ih) {
            return Err(self.zip_close_throw(iters, e));
        }
        Ok(())
    }

    /// `IteratorCloseAll(list, ThrowCompletion(e))`: closes a plain list of iterator
    /// handles in reverse order, swallowing every `return()` error, and returns the
    /// original throw `e` (which always propagates).
    fn zip_close_throw(&mut self, iters: &[NanBox], e: ExecError) -> ExecError {
        for it in iters.iter().rev() {
            if let Some(ih) = it.as_handle().map(Handle::from_raw) {
                let _ = self.iterator_close(ih);
            }
        }
        e
    }

    /// Whether the `fin` flag at index `i` is set (a two-step read to avoid holding
    /// an immutable `realm` borrow across the mutable `get_element`).
    fn zip_fin(&mut self, fin: Handle, i: usize) -> bool {
        let v = self.realm.get_element(fin, i);
        self.realm.truthy(v)
    }

    /// `IteratorCloseAll(openIters, NormalCompletion)` over a zip result's recorded
    /// state: closes every still-open iterator (finished flag false) in reverse index
    /// order (= reverse of the insertion-ordered `openIters` list), marking each
    /// finished. The first thrown `return()` becomes the completion.
    fn zip_close_open_normal(&mut self, iters: Handle, fin: Handle) -> Result<(), ExecError> {
        let mut completion = Ok(());
        let count = self.realm.array_elements(iters).map_or(0, <[_]>::len);
        for i in (0..count).rev() {
            if self.zip_fin(fin, i) {
                continue;
            }
            self.realm.set_element(fin, i, NanBox::boolean(true));
            let Some(ih) = self
                .realm
                .get_element(iters, i)
                .as_handle()
                .map(Handle::from_raw)
            else {
                continue;
            };
            match &completion {
                Ok(()) => {
                    if let Err(e) = self.iterator_close(ih) {
                        completion = Err(e);
                    }
                }
                Err(_) => {
                    let _ = self.iterator_close(ih);
                }
            }
        }
        completion
    }

    /// `IteratorCloseAll(openIters, ThrowCompletion(e))`: closes every still-open
    /// iterator (reverse index order), swallows their errors, returns the throw `e`.
    fn zip_close_open_throw(&mut self, iters: Handle, fin: Handle, e: ExecError) -> ExecError {
        let count = self.realm.array_elements(iters).map_or(0, <[_]>::len);
        for i in (0..count).rev() {
            if self.zip_fin(fin, i) {
                continue;
            }
            self.realm.set_element(fin, i, NanBox::boolean(true));
            if let Some(ih) = self
                .realm
                .get_element(iters, i)
                .as_handle()
                .map(Handle::from_raw)
            {
                let _ = self.iterator_close(ih);
            }
        }
        e
    }

    /// `%IteratorHelperPrototype%.next` for an `Iterator.zip`/`zipKeyed` result:
    /// runs one closure step (with the shared reentrancy guard), wrapping the result.
    pub(crate) fn iter_zip_next(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Zip Iterator next called on non-object"));
        };
        // Reentrancy guard (GeneratorValidate: already executing → TypeError).
        if self
            .realm
            .get_property(h, HELPER_RUNNING)
            .is_some_and(|v| self.realm.truthy(v))
        {
            return Err(self.type_error("Iterator Helper is already running"));
        }
        if self.realm.get_property(h, ZIP_DONE).is_some() {
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        self.realm
            .set_hidden_property(h, HELPER_RUNNING, NanBox::boolean(true));
        let result = self.zip_step(h);
        self.realm.delete_property(h, HELPER_RUNNING);
        match result {
            Ok(Some(v)) => {
                // A yielded value moves the generator to "suspended-yield".
                self.realm
                    .set_hidden_property(h, ZIP_STARTED, NanBox::boolean(true));
                Ok(self.iter_result(v, false))
            }
            Ok(None) => {
                self.realm
                    .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
                Ok(self.iter_result(NanBox::undefined(), true))
            }
            Err(e) => {
                self.realm
                    .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
                Err(e)
            }
        }
    }

    /// One `IteratorZip` closure step: produces the next zipped array (`zip`) or
    /// null-prototype object (`zipKeyed`), or `None` when the whole zip is done.
    fn zip_step(&mut self, h: Handle) -> Result<Option<NanBox>, ExecError> {
        let iters = self
            .realm
            .get_property(h, ZIP_ITERS)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .unwrap();
        let nexts = self
            .realm
            .get_property(h, ZIP_NEXTS)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .unwrap();
        let fin = self
            .realm
            .get_property(h, ZIP_FINISHED)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .unwrap();
        let padding = self
            .realm
            .get_property(h, ZIP_PADDING)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .unwrap();
        let mode = self
            .realm
            .get_property(h, ZIP_MODE)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as u8;
        let count = self.realm.array_elements(iters).map_or(0, <[_]>::len);
        // If openIters is empty, the zip is done.
        if (0..count).all(|i| self.zip_fin(fin, i)) {
            return Ok(None);
        }
        let mut results: Vec<NanBox> = Vec::with_capacity(count);
        for i in 0..count {
            if self.zip_fin(fin, i) {
                // A finished iterator (longest) contributes its padding value.
                results.push(self.realm.get_element(padding, i));
                continue;
            }
            let ih = self
                .realm
                .get_element(iters, i)
                .as_handle()
                .map(Handle::from_raw)
                .unwrap();
            let next = self.realm.get_element(nexts, i);
            match self.iter_step(ih, next) {
                Ok(Some(v)) => results.push(v),
                Err(e) => {
                    self.realm.set_element(fin, i, NanBox::boolean(true));
                    return Err(self.zip_close_open_throw(iters, fin, e));
                }
                Ok(None) => {
                    // Remove this iterator from openIters.
                    self.realm.set_element(fin, i, NanBox::boolean(true));
                    match mode {
                        0 => {
                            // shortest: close the remaining open iterators, finish.
                            self.zip_close_open_normal(iters, fin)?;
                            return Ok(None);
                        }
                        2 => {
                            // strict: a later iterator finishing first is a mismatch.
                            if i != 0 {
                                let te = self.type_error(
                                    "Iterator.zip strict mode: iterators have different lengths",
                                );
                                return Err(self.zip_close_open_throw(iters, fin, te));
                            }
                            // The first finished: every other iterator must also be
                            // done on this step (IteratorStep, value not read).
                            for k in 1..count {
                                let kh = self
                                    .realm
                                    .get_element(iters, k)
                                    .as_handle()
                                    .map(Handle::from_raw)
                                    .unwrap();
                                let knext = self.realm.get_element(nexts, k);
                                match self.iter_step_done(kh, knext) {
                                    Ok(true) => {
                                        self.realm.set_element(fin, k, NanBox::boolean(true));
                                    }
                                    Ok(false) => {
                                        let te = self.type_error(
                                            "Iterator.zip strict mode: iterators have different lengths",
                                        );
                                        return Err(self.zip_close_open_throw(iters, fin, te));
                                    }
                                    Err(e) => {
                                        self.realm.set_element(fin, k, NanBox::boolean(true));
                                        return Err(self.zip_close_open_throw(iters, fin, e));
                                    }
                                }
                            }
                            return Ok(None);
                        }
                        _ => {
                            // longest: when the last live iterator finishes, we are
                            // done; otherwise this slot contributes its padding.
                            if (0..count).all(|j| self.zip_fin(fin, j)) {
                                return Ok(None);
                            }
                            results.push(self.realm.get_element(padding, i));
                        }
                    }
                }
            }
        }
        Ok(Some(self.zip_finish_results(h, results)))
    }

    /// One `IteratorStep` (done-only, does not read `value`): returns whether the
    /// iterator is done. Used by strict-mode "all remaining are done" verification.
    fn iter_step_done(&mut self, it: Handle, next: NanBox) -> Result<bool, ExecError> {
        let res = self.call_with_this(next, NanBox::handle(it.to_raw()), &[])?;
        if !self.is_object_value(res) {
            return Err(self.type_error("iterator result is not an object"));
        }
        let rh = Handle::from_raw(res.as_handle().unwrap());
        let done = self.read_member(rh, "done")?;
        Ok(self.realm.truthy(done))
    }

    /// Builds the yielded value from a step's per-iterator results: a fresh Array
    /// (`zip`) or a null-prototype object keyed by the recorded keys (`zipKeyed`).
    fn zip_finish_results(&mut self, h: Handle, results: Vec<NanBox>) -> NanBox {
        if let Some(keys) = self
            .realm
            .get_property(h, ZIP_KEYS)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            let obj = self.realm.new_object_with_proto(None);
            for (i, v) in results.into_iter().enumerate() {
                let key = self.realm.get_element(keys, i);
                let name = self.member_key(key);
                self.realm.set_property(obj, &name, v);
            }
            NanBox::handle(obj.to_raw())
        } else {
            NanBox::handle(self.realm.new_array(results).to_raw())
        }
    }

    /// `%IteratorHelperPrototype%.return` for an `Iterator.zip`/`zipKeyed` result —
    /// closes every still-open underlying iterator (reverse order), under the
    /// shared reentrancy guard, and propagates the first thrown `return()`.
    pub(crate) fn iter_zip_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw) {
            if self
                .realm
                .get_property(h, HELPER_RUNNING)
                .is_some_and(|v| self.realm.truthy(v))
            {
                return Err(self.type_error("Iterator Helper is already running"));
            }
            let already = self.realm.get_property(h, ZIP_DONE).is_some();
            if !already {
                self.realm
                    .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
                let iters = self
                    .realm
                    .get_property(h, ZIP_ITERS)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw);
                let fin = self
                    .realm
                    .get_property(h, ZIP_FINISHED)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw);
                if let (Some(iters), Some(fin)) = (iters, fin) {
                    // Only a *started* (suspended-yield) generator closes as
                    // "executing" (reentrant next/return throw). A suspended-start
                    // generator is already "completed", so no running guard.
                    let started = self.realm.get_property(h, ZIP_STARTED).is_some();
                    if started {
                        self.realm
                            .set_hidden_property(h, HELPER_RUNNING, NanBox::boolean(true));
                    }
                    let r = self.zip_close_open_normal(iters, fin);
                    if started {
                        self.realm.delete_property(h, HELPER_RUNNING);
                    }
                    r?;
                }
            }
        }
        Ok(self.iter_result(NanBox::undefined(), true))
    }

    /// `%ConcatIteratorPrototype%.return` — closes the active inner iterator (if
    /// any) and marks the concat result done.
    pub(crate) fn iter_concat_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw) {
            // Reentrancy guard: closing the inner iterator invokes its `return`,
            // which may re-enter this `return` (the underlying generator is
            // "executing") — that is a TypeError.
            if self
                .realm
                .get_property(h, HELPER_RUNNING)
                .is_some_and(|v| self.realm.truthy(v))
            {
                return Err(self.type_error("Iterator Helper is already running"));
            }
            let already = self.realm.get_property(h, HELPER_DONE).is_some();
            self.mark_helper_done(h);
            if !already
                && let Some(inner) = self
                    .realm
                    .get_property(h, HELPER_INNER)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
            {
                self.realm
                    .set_hidden_property(h, HELPER_RUNNING, NanBox::boolean(true));
                let r = self.iterator_close(inner);
                self.realm.delete_property(h, HELPER_RUNNING);
                r?;
            }
        }
        Ok(self.iter_result(NanBox::undefined(), true))
    }

    /// `%ConcatIteratorPrototype%.next` — advances through the queued iterables.
    pub(crate) fn iter_concat_next(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        // Reentrancy guard, as in `iter_helper_next` (state=executing → TypeError).
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return self.iter_concat_next_body(this);
        };
        if self
            .realm
            .get_property(h, HELPER_RUNNING)
            .is_some_and(|v| self.realm.truthy(v))
        {
            return Err(self.type_error("Iterator Helper is already running"));
        }
        self.realm
            .set_hidden_property(h, HELPER_RUNNING, NanBox::boolean(true));
        let result = self.iter_concat_next_body(this);
        self.realm.delete_property(h, HELPER_RUNNING);
        result
    }

    fn iter_concat_next_body(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("next called on non-object"));
        };
        if self.realm.get_property(h, HELPER_DONE).is_some() {
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        let arr = self
            .realm
            .get_property(h, HELPER_SOURCE)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        let Some(arr) = arr else {
            return Ok(self.iter_result(NanBox::undefined(), true));
        };
        loop {
            // Drain the current inner iterator, if open.
            if let Some(inner) = self
                .realm
                .get_property(h, HELPER_INNER)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
            {
                let inner_next = self
                    .realm
                    .get_property(h, HELPER_INNER_NEXT)
                    .unwrap_or(NanBox::undefined());
                match self.iter_step(inner, inner_next) {
                    Ok(Some(v)) => return Ok(self.iter_result(v, false)),
                    Ok(None) => {
                        self.realm.delete_property(h, HELPER_INNER);
                    }
                    Err(e) => {
                        self.mark_helper_done(h);
                        return Err(e);
                    }
                }
            }
            // Open the next iterable in the queue.
            let idx = self
                .realm
                .get_property(h, HELPER_COUNTER)
                .and_then(|n| n.as_number())
                .unwrap_or(0.0) as usize;
            let len = self.realm.array_elements(arr).map_or(0, |e| e.len());
            if idx >= len {
                self.mark_helper_done(h);
                return Ok(self.iter_result(NanBox::undefined(), true));
            }
            self.realm
                .set_hidden_property(h, HELPER_COUNTER, NanBox::number((idx + 1) as f64));
            let src = self.realm.get_element(arr, idx);
            // Re-invoke the `@@iterator` method captured at `concat` time (do NOT
            // re-read `@@iterator` — its getter must fire only once). A stored
            // `undefined` means a built-in iterable drained directly.
            let method = self
                .realm
                .get_property(h, HELPER_METHODS)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                .map(|ma| self.realm.get_element(ma, idx))
                .unwrap_or(NanBox::undefined());
            let ith = if method
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                match self.call_with_this(method, src, &[]) {
                    Ok(it) if self.is_object_value(it) => {
                        it.as_handle().map(Handle::from_raw).unwrap()
                    }
                    Ok(_) => {
                        self.mark_helper_done(h);
                        return Err(
                            self.type_error("Iterator.concat: the iterator is not an object")
                        );
                    }
                    Err(e) => {
                        self.mark_helper_done(h);
                        return Err(e);
                    }
                }
            } else {
                match self.get_iter_object(src) {
                    Ok(ih) => ih,
                    Err(e) => {
                        self.mark_helper_done(h);
                        return Err(e);
                    }
                }
            };
            let inner_next = self.read_member(ith, "next")?;
            self.realm
                .set_hidden_property(h, HELPER_INNER, NanBox::handle(ith.to_raw()));
            self.realm
                .set_hidden_property(h, HELPER_INNER_NEXT, inner_next);
        }
    }

    /// `Array.fromAsync(asyncItems, mapFn?, thisArg?)` — the synchronous core.
    /// Eagerly drives the (a)sync iterable / array-like (awaiting each value),
    /// applies `mapFn` (awaiting its result), and builds the result array. The
    /// dispatch site wraps the returned value — or a thrown value — into the
    /// promise `fromAsync` returns, so every failure here becomes a rejection.
    pub(crate) fn array_from_async_core(
        &mut self,
        items_box: NanBox,
        map_fn: NanBox,
        this_arg: NanBox,
        this_ctor: NanBox,
    ) -> Result<NanBox, ExecError> {
        let has_map = !matches!(map_fn.unpack(), Unpacked::Undefined);
        if has_map {
            self.require_callable(map_fn, "Array.fromAsync mapFn")?;
        }
        if matches!(items_box.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Err(self.type_error(
                "Array.fromAsync requires an array-like or iterable object, not null/undefined",
            ));
        }
        // An (a)sync iterable is drained through the async-iterator protocol
        // (`for_await_values`); a bare array-like (a `length` + indices, no
        // iterator) has each element `Get` then awaited.
        let h = items_box.as_handle().map(Handle::from_raw);
        let iterable = match h {
            Some(h) => {
                // GetMethod(items, @@asyncIterator): a present, non-null value that
                // is not callable is a TypeError (rejects the returned promise);
                // undefined/null means "no async iterator", so fall through to the
                // sync iterator / array-like path.
                let sym = self.well_known_symbol("asyncIterator");
                let akey = self.member_key(sym);
                let async_m = self.read_member(h, &akey)?;
                let has_async = if matches!(async_m.unpack(), Unpacked::Undefined | Unpacked::Null)
                {
                    false
                } else if self.is_callable_value(async_m) {
                    true
                } else {
                    return Err(
                        self.type_error("Array.fromAsync: @@asyncIterator is not a function")
                    );
                };
                // GetMethod(items, @@iterator): same non-callable → TypeError rule.
                let has_sync = if has_async {
                    false
                } else {
                    match self.find_iterator_fn(h)? {
                        Some(f) if self.is_callable_value(f) => true,
                        Some(f) if matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) => {
                            false
                        }
                        Some(_) => {
                            return Err(
                                self.type_error("Array.fromAsync: @@iterator is not a function")
                            );
                        }
                        None => false,
                    }
                };
                has_async
                    || has_sync
                    || self.realm.is_array(h)
                    || self.realm.string_value(h).is_some()
                    || self.realm.collection_is_set(h).is_some()
                    || self.realm.get_property(h, GEN_BUF).is_some()
                    || self.gen_frame_id(h).is_some()
            }
            None => false,
        };
        let raw_items = if iterable {
            self.for_await_values(items_box)?
        } else {
            // Array-like: ToLength(Get(O, "length")), then await Get(O, k).
            let mut out = Vec::new();
            if let Some(h) = h {
                let len_val = self.read_member(h, "length")?;
                let len_num = self.coerce_to_number(len_val)?;
                let len_raw = self.realm.to_number(len_num);
                if len_raw > self.realm.limits.max_array_len as f64 {
                    let m = self.new_str("Invalid array length");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                let len = if len_raw.is_nan() || len_raw <= 0.0 {
                    0
                } else {
                    len_raw as usize
                };
                for i in 0..len {
                    let v = self.read_member(h, &alloc::format!("{i}"))?;
                    out.push(self.await_value(v)?);
                }
            }
            out
        };
        // Apply `mapFn` (awaiting each result) in index order.
        let items = if has_map {
            let mut out = Vec::with_capacity(raw_items.len());
            for (i, e) in raw_items.iter().enumerate() {
                let mapped =
                    self.call_with_this(map_fn, this_arg, &[*e, NanBox::number(i as f64)])?;
                out.push(self.await_value(mapped)?);
            }
            out
        } else {
            raw_items
        };
        // A subclass / constructor `this` (`C.fromAsync(...)`) is `Construct`ed
        // and populated; the default `%Array%` builds a plain dense array.
        let is_array_ctor =
            self.current.get("Array").and_then(|v| v.as_handle()) == this_ctor.as_handle();
        if self.is_constructor_value(this_ctor) && !is_array_ctor {
            let len = items.len();
            let target = self.construct(this_ctor, &[NanBox::number(len as f64)])?;
            let Some(th) = target.as_handle().map(Handle::from_raw) else {
                return Err(self.type_error("Array.fromAsync constructor did not return an object"));
            };
            for (i, e) in items.iter().enumerate() {
                // `CreateDataPropertyOrThrow(A, i, e)` routes array indices into the
                // dense element store (a raw `set_property` stashes them as named
                // props that `join`/dense reads miss — as in `C.from`).
                self.create_data_property_or_throw(th, i, *e)?;
            }
            let len_key = self.new_str("length");
            self.assign_member_value(th, len_key, NanBox::number(len as f64))?;
            Ok(target)
        } else {
            Ok(NanBox::handle(self.realm.new_array(items).to_raw()))
        }
    }

    /// Drains an iterable for `for await (… of …)`. An *async* iterator — an
    /// `async function*` generator object, or any object with a callable
    /// `[Symbol.asyncIterator]` — yields **promises of iterator results**, so each
    /// `next()` result is awaited before its `done`/`value` is read. Any other
    /// iterable uses the ordinary synchronous protocol with each yielded value
    /// awaited (the `AsyncFromSyncIterator` wrapping).
    pub(crate) fn for_await_values(&mut self, v: NanBox) -> Result<Vec<NanBox>, ExecError> {
        if let Some(ih) = self.async_iterator_of(v)? {
            let mut out = Vec::new();
            loop {
                let next_fn = self.read_member(ih, "next")?;
                let res = self.call_with_this(next_fn, NanBox::handle(ih.to_raw()), &[])?;
                let res = self.await_value(res)?;
                if !self.is_object_value(res) {
                    return Err(self.type_error("iterator result is not an object"));
                }
                let rh = Handle::from_raw(res.as_handle().unwrap());
                let done = self.read_member(rh, "done")?;
                if self.realm.truthy(done) {
                    break;
                }
                out.push(self.read_member(rh, "value")?);
                if out.len() > GEN_CAP {
                    return Err(self.type_error("iterator did not terminate"));
                }
            }
            return Ok(out);
        }
        // A sync iterable in `for await`: drain synchronously, then await each
        // yielded value (a non-promise passes through unchanged).
        let mut values = self.iterate_values(v)?;
        for val in &mut values {
            *val = self.await_value(*val)?;
        }
        Ok(values)
    }

    /// The async iterator to drive for `for await`, if `v` is an async iterable:
    /// the async-generator object itself, or the result of calling its callable
    /// `[Symbol.asyncIterator]`. `None` for a sync iterable (the caller falls back
    /// to the synchronous protocol).
    pub(crate) fn async_iterator_of(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        // A lazy generator object: an `async function*` is its own async iterator;
        // a plain `function*` is a *sync* iterator (drained synchronously).
        if let Some(is_async) = self.lazy_gen_is_async(h) {
            return Ok(if is_async { Some(h) } else { None });
        }
        // Otherwise, an object whose `[Symbol.asyncIterator]` is callable.
        let sym = self.well_known_symbol("asyncIterator");
        let key = self.member_key(sym);
        let f = self.read_member(h, &key)?;
        if f.as_handle()
            .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
        {
            let it = self.call_with_this(f, v, &[])?;
            let Some(ih) = it.as_handle().map(Handle::from_raw) else {
                return Err(self.type_error("iterator is not an object"));
            };
            return Ok(Some(ih));
        }
        Ok(None)
    }

    /// Drains an **already-obtained** iterator object (the result of calling
    /// `source[@@iterator]()`) to the `Vec` of its values, propagating a throwing
    /// `next` / `next().value` (`IteratorStep` / `IteratorValue`). A generator
    /// iterator drains from its value buffer. Unlike [`Self::iterate_values`], this
    /// does *not* re-read `@@iterator` — the caller has already invoked it (so a
    /// `GetMethod`/`@@iterator`-getter side effect is observed exactly once, as
    /// `TypedArray.from`/`Array.from` require).
    pub(crate) fn drain_iterator_values(
        &mut self,
        iterator: NanBox,
    ) -> Result<Vec<NanBox>, ExecError> {
        let Some(ih) = iterator.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("iterator is not an object"));
        };
        // A generator iterator (its `next` is a built-in, not a readable property)
        // is drained directly from its buffer.
        if self.realm.get_property(ih, GEN_BUF).is_some() {
            return self.iterate_values(iterator);
        }
        // `GetIteratorFromMethod` reads `next` **once** (`iteratorRecord.[[NextMethod]]`)
        // and reuses it for every `IteratorStep` — re-reading it per step would rerun a
        // `next` *accessor* (which may hand back a fresh, self-resetting iterator each
        // read, i.e. never terminate).
        let next_fn = self.read_member(ih, "next")?;
        let mut out = Vec::new();
        loop {
            let res = self.call_with_this(next_fn, iterator, &[])?;
            if !self.is_object_value(res) {
                return Err(self.type_error("iterator result is not an object"));
            }
            let rh = Handle::from_raw(res.as_handle().unwrap());
            let done = self.read_member(rh, "done")?;
            if self.realm.truthy(done) {
                break;
            }
            out.push(self.read_member(rh, "value")?);
            if out.len() > GEN_CAP {
                return Err(self.type_error("iterator did not terminate"));
            }
        }
        Ok(out)
    }

    pub(crate) fn iterate_values(&mut self, v: NanBox) -> Result<Vec<NanBox>, ExecError> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            let m = self.new_str("is not iterable");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        // A `String` wrapper object iterates its characters (a `Number`/`Boolean`
        // wrapper is not iterable — falls through to the error).
        if let Some(prim) = self.realm.get_property(h, PRIM_WRAP)
            && let Some(ph) = prim.as_handle().map(Handle::from_raw)
            && self.realm.string_value(ph).is_some()
        {
            return self.iterate_values(prim);
        }
        if let Some(mut elems) = self.realm.elements_vec(h) {
            // The %ArrayIteratorPrototype% `next` does `Get(array, index)`, so a hole
            // reads as `undefined` (not the internal hole sentinel). Normalize so
            // for-of / spread / `Array.from` over a sparse array yield real
            // `undefined` values.
            for e in &mut elems {
                if e.is_hole() {
                    *e = NanBox::undefined();
                }
            }
            return Ok(elems);
        }
        if let Some(bytes) = self.realm.string_bytes(h) {
            // `for…of` yields one string per Unicode code point; a lone surrogate
            // is a single item (its own one-unit string).
            let mut out = Vec::new();
            for cp in crate::wtf8::code_points(&bytes) {
                let mut buf = Vec::new();
                crate::wtf8::encode_code_point(cp, &mut buf);
                out.push(self.new_str_bytes(buf));
            }
            return Ok(out);
        }
        // `Map`/`Set` iterate their entries; `WeakMap`/`WeakSet` are not iterable
        // (they fall through to the not-iterable TypeError below).
        if !self.realm.collection_is_weak(h)
            && let Some(entries) = self.realm.collection_entries(h)
        {
            if self.realm.collection_is_set(h) == Some(true) {
                return Ok(entries.iter().map(|(k, _)| *k).collect());
            }
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                out.push(NanBox::handle(
                    self.realm.new_array(alloc::vec![k, v]).to_raw(),
                ));
            }
            return Ok(out);
        }
        // A generator iterator: its remaining buffered values.
        if let Some(buf) = self
            .realm
            .get_property(h, GEN_BUF)
            .and_then(|b| b.as_handle())
            .map(Handle::from_raw)
        {
            let idx = self
                .realm
                .get_property(h, GEN_IDX)
                .and_then(|n| n.as_number())
                .unwrap_or(0.0) as usize;
            let elems = self
                .realm
                .array_elements(buf)
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            let len = elems.len();
            let result: Vec<NanBox> = elems.into_iter().skip(idx).collect();
            // Draining the iterator (for-of/spread) consumes it: advance to the end so a
            // later `.next()` reports `{ done: true }` rather than restarting.
            self.realm
                .set_property(h, GEN_IDX, NanBox::number(len as f64));
            return Ok(result);
        }
        // A custom iterable: call `obj[Symbol.iterator]()` and drain `.next()`.
        // The method may be an own/inherited property (anywhere on the prototype
        // chain) or a class method whose computed key is `Symbol.iterator`
        // (`class C { *[Symbol.iterator]() {…} }`).
        let iter_fn = self.find_iterator_fn(h)?;
        if let Some(f) = iter_fn
            && f.as_handle()
                .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
        {
            let iterator = self.call_with_this(f, v, &[])?;
            let Some(ih) = iterator.as_handle().map(Handle::from_raw) else {
                // GetIterator: a `[Symbol.iterator]()` result that is not an Object
                // is a TypeError (so `e instanceof TypeError` holds).
                return Err(self.type_error("iterator is not an object"));
            };
            // A generator iterator (its `next` is a built-in method, not a
            // readable property) is drained directly from its buffer.
            if self.realm.get_property(ih, GEN_BUF).is_some() {
                return self.iterate_values(iterator);
            }
            let mut out = Vec::new();
            loop {
                let next_fn = self.read_member(ih, "next")?;
                let res = self.call_with_this(next_fn, iterator, &[])?;
                if !self.is_object_value(res) {
                    return Err(self.type_error("iterator result is not an object"));
                }
                let rh = Handle::from_raw(res.as_handle().unwrap());
                let done = self.read_member(rh, "done")?;
                if self.realm.truthy(done) {
                    break;
                }
                out.push(self.read_member(rh, "value")?);
                if out.len() > GEN_CAP {
                    return Err(self.type_error("iterator did not terminate"));
                }
            }
            return Ok(out);
        }
        let m = self.new_str("is not iterable");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    /// Finds a class instance's `[Symbol.iterator]` method (a method whose
    /// computed key evaluates to the well-known iterator symbol), walking the
    /// `extends` chain. Returns the bound method value, or `None`.
    pub(crate) fn class_iterator_method(
        &mut self,
        h: crate::heap::Handle,
    ) -> Result<Option<NanBox>, ExecError> {
        let Some(tag) = self.realm.class_tag(h) else {
            return Ok(None);
        };
        let iter_sym = self.well_known_symbol("iterator");
        let mut cur = Some(tag);
        while let Some(cid) = cur {
            let class = self.classes[cid as usize];
            let env = self.class_envs[cid as usize].clone();
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && !m.is_static
                    && m.kind == MethodKind::Method
                    && let PropertyKey::Computed(ke) = &m.key
                {
                    let saved = core::mem::replace(&mut self.current, env.clone());
                    let key = self.eval(ke);
                    self.current = saved;
                    if self.realm.strict_equals(key?, iter_sym) {
                        let saved = core::mem::replace(&mut self.current, env.clone());
                        let f = self.make_method(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            false,
                            m.value.is_generator,
                            Some(cid),
                            false,
                        );
                        self.current = saved;
                        return Ok(Some(f));
                    }
                }
            }
            cur = self.resolve_super(class, &env)?.map(|(p, _)| p);
        }
        Ok(None)
    }

    /// Resolves an object's `[Symbol.iterator]` method (`GetMethod`), looking up
    /// the property through the *entire* prototype chain — so an iterable whose
    /// `Symbol.iterator` is inherited (`Object.create(iterable)`, a subclass of
    /// `Iterator`, a class instance whose method lives on its prototype) is found.
    /// Falls back to the class-method scan for class instances whose computed
    /// `[Symbol.iterator]` method is not yet materialized as a prototype property.
    /// Returns `None` only when no iterator method exists anywhere on the chain.
    pub(crate) fn find_iterator_fn(
        &mut self,
        h: crate::heap::Handle,
    ) -> Result<Option<NanBox>, ExecError> {
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        // `read_member` walks the prototype chain (and fires inherited accessors),
        // so an inherited `Symbol.iterator` resolves here.
        let fn_val = self.read_member(h, &iter_key)?;
        if !matches!(fn_val.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Ok(Some(fn_val));
        }
        // A class instance whose `[Symbol.iterator]` is defined with a computed
        // key may not surface as a readable prototype property; scan the class body.
        self.class_iterator_method(h)
    }

    /// The keys iterated by `for-in`: object property names or array indices,
    /// as strings.
    pub(crate) fn iterate_keys(&mut self, v: NanBox) -> Vec<NanBox> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Vec::new();
        };
        // A proxy with no `ownKeys` trap (the trap case is handled by the caller)
        // enumerates its target's keys.
        let h = self.proxy_key_target(h);
        // `for-in` enumerates own enumerable keys, then enumerable keys inherited
        // through the prototype chain — each name only once, own keys first.
        let mut seen = alloc::collections::BTreeSet::new();
        let mut out = Vec::new();
        // An array's own keys lead with its integer indices (a VM closure's backing
        // cells are not enumerable).
        if !self.realm.is_vm_function(h)
            && let Some(indices) = self.realm.array_enumerable_indices(h)
        {
            for i in indices {
                let k = alloc::format!("{i}");
                if seen.insert(k.clone()) {
                    out.push(self.new_str(&k));
                }
            }
        }
        // A String object's own enumerable keys are its indices `"0".."length-1"`
        // (`length` is non-enumerable), so `for (k in "abc")` yields "0","1","2".
        if let Some(slen) = self.string_index_count(h) {
            for i in 0..slen {
                let k = alloc::format!("{i}");
                if seen.insert(k.clone()) {
                    out.push(self.new_str(&k));
                }
            }
        }
        let mut cur = Some(h);
        while let Some(c) = cur {
            // Plain objects keep keys in the cell; arrays/functions keep named
            // properties in their auxiliary object.
            let named = self
                .realm
                .object_keys(c)
                .unwrap_or_else(|| self.realm.aux_named_keys(c));
            for k in named {
                if seen.insert(k.clone()) {
                    out.push(self.new_str(&k));
                }
            }
            cur = self.realm.object_proto(c);
        }
        out
    }

    /// `GetIterator` step `GetMethod(obj, @@iterator)` for an *array* — the one
    /// built-in iterable that exposes a real, deletable `Symbol.iterator` method
    /// (the array-destructuring/spread fast path would otherwise iterate its
    /// backing store directly). Throws a `TypeError` when that method has been
    /// deleted or replaced by a non-callable (`delete Array.prototype[Symbol.iterator]`).
    /// Strings, Maps, and Sets iterate via engine special-casing (no readable
    /// `Symbol.iterator` property) and are left to the regular machinery.
    pub(crate) fn require_iterator_method(&mut self, v: NanBox) -> Result<(), ExecError> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Ok(());
        };
        if self.realm.array_elements(h).is_none() {
            return Ok(());
        }
        let callable = self
            .find_iterator_fn(h)?
            .and_then(|f| f.as_handle().map(Handle::from_raw))
            .is_some_and(|r| self.is_callable(r));
        if !callable {
            let m = self.new_str("is not iterable");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    /// Obtains a *user* iterable's iterator object (calling `[Symbol.iterator]`
    /// once), for the lazy `for-of` path. Returns `None` for built-in iterables
    /// (arrays/strings/Maps/Sets) and generator values, which `iterate_values`
    /// drains eagerly, and for non-iterables.
    pub(crate) fn for_of_get_iterator(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        self.for_of_get_iterator_ext(v, false)
    }

    /// As [`Self::for_of_get_iterator`], but when `array_live` is set a plain array
    /// with the intrinsic `[Symbol.iterator]` returns a **live** `%ArrayIterator%`
    /// (re-reads `length` and `Get`s each element per step) instead of `None` — so a
    /// `for…of` observes `push`/`pop`/getter side effects mid-iteration, matching
    /// `CreateArrayIterator`. Destructuring/spread callers pass `false` and keep the
    /// eager snapshot path.
    pub(crate) fn for_of_get_iterator_ext(
        &mut self,
        v: NanBox,
        array_live: bool,
    ) -> Result<Option<Handle>, ExecError> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        // A non-weak Map/Set gets a **live** iterator so `for-of` observes a
        // mutation mid-iteration (a Set yields values; a Map yields entries). A
        // weak collection is not iterable — fall through to the TypeError path.
        if !self.realm.collection_is_weak(h) && self.realm.collection_entries(h).is_some() {
            let is_set = self.realm.collection_is_set(h) == Some(true);
            let tag = if is_set {
                "Set Iterator"
            } else {
                "Map Iterator"
            };
            let kind = if is_set { 1 } else { 2 };
            let it = self.make_live_collection_iterator(h, kind, tag);
            return Ok(it.as_handle().map(Handle::from_raw));
        }
        // A typed array gets a **live** iterator (values), so `for-of` observes a
        // resizable-buffer resize or element write mid-iteration.
        if self.realm.typed_kind(h).is_some() {
            let it = self.make_live_typed_iterator(h, 1);
            return Ok(it.as_handle().map(Handle::from_raw));
        }
        // A real array takes the fast path (direct backing-store read) *only* when
        // its `[Symbol.iterator]` is still the intrinsic `Array.prototype.values`.
        // A user override (`Array.prototype[Symbol.iterator] = function* …` or a
        // per-instance one) must go through the iterator protocol, per spec.
        if self.realm.array_elements(h).is_some() {
            let resolved = self.find_iterator_fn(h)?;
            let default_iter = self
                .realm
                .array_proto_intrinsic()
                .and_then(|p| self.realm.get_property(p, "values"));
            let is_default = matches!((resolved, default_iter), (Some(r), Some(d))
                if r.as_handle().is_some() && r.as_handle() == d.as_handle());
            if is_default {
                if array_live {
                    let it = self.make_live_array_iterator(h, 1);
                    return Ok(it.as_handle().map(Handle::from_raw));
                }
                return Ok(None);
            }
            if let Some(f) = resolved
                && f.as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|fh| self.is_callable(fh))
            {
                let iterator = self.call_with_this(f, v, &[])?;
                return match iterator.as_handle().map(Handle::from_raw) {
                    Some(ih) => Ok(Some(ih)),
                    None => Err(self.type_error("iterator is not an object")),
                };
            }
            return Ok(None);
        }
        if self.realm.string_value(h).is_some()
            || self.realm.collection_entries(h).is_some()
            || self.realm.get_property(h, GEN_BUF).is_some()
        {
            return Ok(None);
        }
        let Some(f) = self.find_iterator_fn(h)? else {
            return Ok(None);
        };
        if !f
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return Ok(None);
        }
        let iterator = self.call_with_this(f, v, &[])?;
        match iterator.as_handle().map(Handle::from_raw) {
            Some(ih) => Ok(Some(ih)),
            None => Err(self.type_error("iterator is not an object")),
        }
    }

    /// `IteratorClose`: invoke the iterator's `return()` method (if any) on an early
    /// exit, so the iterator can release resources. Errors from `return()` propagate.
    ///
    /// Per spec, when closing on a *normal* completion (this method's `?` callers),
    /// a `return` that is present but whose call yields a non-Object result must throw
    /// a `TypeError`. Error-completion callers discard this via `let _ = ...`, matching
    /// the spec rule that the original throw takes precedence over `IteratorClose`.
    pub(crate) fn iterator_close(&mut self, ih: Handle) -> Result<(), ExecError> {
        let ret = self.read_member(ih, "return")?;
        // `GetMethod` treats `undefined`/`null` as an absent method: nothing to close.
        if ret.is_undefined() || ret.is_null() {
            return Ok(());
        }
        if !ret
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return Err(self.type_error("iterator return is not a function"));
        }
        let result = self.call_with_this(ret, NanBox::handle(ih.to_raw()), &[])?;
        if result.as_handle().is_none() {
            return Err(self.type_error("iterator return must return an object"));
        }
        Ok(())
    }
}
