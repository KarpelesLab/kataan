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
        let Some(rh) = res.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("iterator result is not an object"));
        };
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
        let next_pos = match self.realm.get_property(h, GEN_LASTKEY) {
            None => 0,
            Some(last_key) => match entries
                .iter()
                .position(|(k, _)| self.realm.same_value_zero(*k, last_key))
            {
                Some(q) => q + 1,
                None => self
                    .realm
                    .get_property(h, GEN_IDX)
                    .and_then(|n| n.as_number())
                    .unwrap_or(0.0) as usize,
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
        // The live length (tracks a resizable backing buffer); `None`/0 for a
        // detached or out-of-bounds view ends the iteration.
        let len = self.realm.typed_len(ta).unwrap_or(0);
        if idx >= len {
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
    fn iter_result(&mut self, value: NanBox, done: bool) -> NanBox {
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
        // If `value[Symbol.iterator]` is callable, call it; if it's a built-in
        // iterable, drain it; otherwise (a plain object carrying its own `next`)
        // use the object itself as the iterator.
        let iter_fn = match self.find_iterator_fn(vh) {
            Ok(f) => f,
            Err(e) => {
                let _ = self.iterator_close(src_h);
                return Err(e);
            }
        };
        if iter_fn.is_some_and(|fv| {
            fv.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        }) || self.realm.array_elements(vh).is_some()
            || self.realm.collection_entries(vh).is_some()
            || self.realm.get_property(vh, GEN_BUF).is_some()
        {
            return match self.get_iter_object(value) {
                Ok(ih) => Ok(ih),
                Err(e) => {
                    let _ = self.iterator_close(src_h);
                    Err(e)
                }
            };
        }
        Ok(vh)
    }

    /// `%IteratorHelperPrototype%.return` — closes the helper (and its source).
    pub(crate) fn iter_helper_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw) {
            let already_done = self.realm.get_property(h, HELPER_DONE).is_some();
            self.mark_helper_done(h);
            if !already_done
                && let Some(src) = self
                    .realm
                    .get_property(h, HELPER_SOURCE)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
            {
                self.iterator_close(src)?;
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
        // A string source iterates its code points via `%StringIteratorPrototype%`.
        if let Some(h) = src.as_handle().map(Handle::from_raw)
            && self.realm.string_value(h).is_some()
        {
            // Build a lazy generator-ish wrapper: reuse the eager string iteration
            // but expose it through the wrap protocol so helpers attach.
            let values = self.iterate_values(src)?;
            let gen_iter = self.make_generator(values);
            return self.wrap_iterator_value(gen_iter);
        }
        // A non-string primitive (null/undefined/number/bigint/boolean/symbol) is
        // a TypeError (`Iterator.from` only accepts objects and strings).
        if !self.is_object_value(src) {
            let m = self.new_str("Iterator.from called on a non-object");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // GetIteratorFlattenable(O, iterate-string-primitives): if `O` has
        // `[Symbol.iterator]`, call it to obtain the iterator; else if `O` is
        // already an iterator (has a `next`), use it; else if it's a built-in
        // iterable, drain it into a generator iterator.
        let h = src.as_handle().map(Handle::from_raw).unwrap();
        let iter_fn = self.find_iterator_fn(h)?;
        let iterator = if iter_fn.is_some_and(|fv| {
            fv.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        }) {
            let fv = iter_fn.unwrap();
            self.call_with_this(fv, src, &[])?
        } else {
            // No `[Symbol.iterator]`: a built-in iterable drains to a generator,
            // an object with a `next` is used directly.
            let is_string_wrapper = self
                .realm
                .get_property(h, PRIM_WRAP)
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw)
                .is_some_and(|ph| self.realm.string_value(ph).is_some());
            let is_builtin_iterable = is_string_wrapper
                || self.realm.array_elements(h).is_some()
                || self.realm.collection_entries(h).is_some()
                || self.realm.get_property(h, GEN_BUF).is_some();
            if is_builtin_iterable {
                let vals = self.iterate_values(src)?;
                self.make_generator(vals)
            } else {
                src
            }
        };
        let Some(ih) = iterator.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Iterator.from: not an iterator"));
        };
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
    pub(crate) fn iter_wrap_next(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("next called on non-object"));
        };
        let src = self
            .realm
            .get_property(h, HELPER_SOURCE)
            .unwrap_or(NanBox::undefined());
        let next = self
            .realm
            .get_property(h, HELPER_NEXT)
            .unwrap_or(NanBox::undefined());
        self.call_with_this(next, src, &[])
    }

    /// `%WrapForValidIteratorPrototype%.return` — forwards to the wrapped iterator's
    /// `return` (if callable), else returns `{ value: undefined, done: true }`.
    pub(crate) fn iter_wrap_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw)
            && let Some(src) = self
                .realm
                .get_property(h, HELPER_SOURCE)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
        {
            let ret = self.read_member(src, "return")?;
            if ret
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                return self.call_with_this(ret, NanBox::handle(src.to_raw()), &[]);
            }
        }
        Ok(self.iter_result(NanBox::undefined(), true))
    }

    /// `Iterator.concat(...items)`: each item must be an object with a
    /// `[Symbol.iterator]` method (read eagerly, in order); the result is a lazy
    /// iterator that yields all of the first's values, then the second's, etc.
    pub(crate) fn iterator_concat(&mut self, items: &[NanBox]) -> Result<NanBox, ExecError> {
        // Validate every argument up front (per spec: collect iterator-getters in
        // order before producing the result).
        let mut sources: Vec<NanBox> = Vec::with_capacity(items.len());
        for it in items {
            if !self.is_object_value(*it) {
                return Err(self.type_error("Iterator.concat argument is not an object"));
            }
            let h = it.as_handle().map(Handle::from_raw).unwrap();
            let iter_fn = self.find_iterator_fn(h)?;
            let has_iter = iter_fn.is_some_and(|fv| {
                fv.as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            });
            let is_builtin_iterable = self.realm.array_elements(h).is_some()
                || self.realm.string_value(h).is_some()
                || self.realm.collection_entries(h).is_some()
                || self.realm.get_property(h, GEN_BUF).is_some();
            if !has_iter && !is_builtin_iterable {
                return Err(self.type_error("Iterator.concat argument is not iterable"));
            }
            sources.push(*it);
        }
        let arr = self.realm.new_array(sources);
        let proto = self.iter_ctor_slot(ITER_CONCAT_PROTO_SLOT);
        let h = self.realm.new_object_with_proto(proto);
        self.realm
            .set_hidden_property(h, HELPER_SOURCE, NanBox::handle(arr.to_raw()));
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
    /// (the `joint-iteration` proposal). Reads the options, opens each underlying
    /// iterator (in order), and returns a lazy `%ZipIteratorPrototype%` result.
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
        if !self.is_object_value(iterables) {
            return Err(self.type_error(&alloc::format!("{what}: iterables is not an object")));
        }
        // options: undefined → {}; else must be an object.
        let opts_present = !matches!(options.unpack(), Unpacked::Undefined);
        if opts_present && !self.is_object_value(options) {
            return Err(self.type_error(&alloc::format!("{what}: options is not an object")));
        }
        // mode (default "shortest").
        let mode_val = if opts_present {
            self.read_member(options.as_handle().map(Handle::from_raw).unwrap(), "mode")?
        } else {
            NanBox::undefined()
        };
        let mode = if matches!(mode_val.unpack(), Unpacked::Undefined) {
            0u8
        } else {
            let s = self.realm.to_display_string(mode_val);
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
        };
        // Collect (key?, iterable) pairs (before padding, since keyed padding is
        // read per key).
        let pairs: Vec<(Option<String>, NanBox)> = if keyed {
            let h = iterables.as_handle().map(Handle::from_raw).unwrap();
            let keys = self.realm.object_keys_with_symbols(h);
            let mut out = Vec::new();
            for k in keys {
                // Only string keys participate (per zipKeyed's own enumerable keys);
                // symbol keys are included too in the spec, but we key by string.
                let v = self.read_member(h, &k)?;
                out.push((Some(k), v));
            }
            out
        } else {
            // iterables is itself iterated to a list.
            let vals = self.iterate_values(iterables)?;
            vals.into_iter().map(|v| (None, v)).collect()
        };
        // padding (only for longest mode). For `zip` it is an iterable of per-position
        // fillers; for `zipKeyed` it is an object read per key (matching `pairs`).
        let mut padding: Vec<NanBox> = Vec::new();
        if mode == 1 && opts_present {
            let pad_val = self.read_member(
                options.as_handle().map(Handle::from_raw).unwrap(),
                "padding",
            )?;
            if !matches!(pad_val.unpack(), Unpacked::Undefined) {
                if !self.is_object_value(pad_val) {
                    return Err(
                        self.type_error(&alloc::format!("{what}: padding is not an object"))
                    );
                }
                if keyed {
                    let ph = pad_val.as_handle().map(Handle::from_raw).unwrap();
                    for (k, _) in &pairs {
                        let pv = self.read_member(ph, k.as_deref().unwrap_or(""))?;
                        padding.push(pv);
                    }
                } else {
                    // padding is collected per-iterator (indexed by position).
                    padding = self.zip_collect_padding(pad_val)?;
                }
            }
        }
        // Open each iterator (GetIteratorFlattenable), caching its `next`. On an
        // error opening iterator N, close the already-opened ones.
        let mut iters: Vec<NanBox> = Vec::with_capacity(pairs.len());
        let mut nexts: Vec<NanBox> = Vec::with_capacity(pairs.len());
        let mut keys_vec: Vec<NanBox> = Vec::with_capacity(pairs.len());
        for (k, iterable) in &pairs {
            let ih = match self.zip_open_iterator(*iterable) {
                Ok(h) => h,
                Err(e) => {
                    for opened in &iters {
                        if let Some(oh) = opened.as_handle().map(Handle::from_raw) {
                            let _ = self.iterator_close(oh);
                        }
                    }
                    return Err(e);
                }
            };
            let next = self.read_member(ih, "next")?;
            iters.push(NanBox::handle(ih.to_raw()));
            nexts.push(next);
            if let Some(ks) = k {
                let kb = self.new_str(ks);
                keys_vec.push(kb);
            }
        }
        // Build the zip-iterator object.
        let proto = self.iter_ctor_slot(ITER_ZIP_PROTO_SLOT);
        let h = self.realm.new_object_with_proto(proto);
        let iters_arr = self.realm.new_array(iters);
        let nexts_arr = self.realm.new_array(nexts);
        self.realm
            .set_hidden_property(h, ZIP_ITERS, NanBox::handle(iters_arr.to_raw()));
        self.realm
            .set_hidden_property(h, ZIP_NEXTS, NanBox::handle(nexts_arr.to_raw()));
        self.realm
            .set_hidden_property(h, ZIP_MODE, NanBox::number(f64::from(mode)));
        // Pad the padding list to the iterator count with undefined.
        while padding.len() < pairs.len() {
            padding.push(NanBox::undefined());
        }
        let pad_arr = self.realm.new_array(padding);
        self.realm
            .set_hidden_property(h, ZIP_PADDING, NanBox::handle(pad_arr.to_raw()));
        // Per-iterator finished flags (longest mode), initially all false.
        let fin: Vec<NanBox> = (0..pairs.len()).map(|_| NanBox::boolean(false)).collect();
        let fin_arr = self.realm.new_array(fin);
        self.realm
            .set_hidden_property(h, ZIP_FINISHED, NanBox::handle(fin_arr.to_raw()));
        if keyed {
            let keys_arr = self.realm.new_array(keys_vec);
            self.realm
                .set_hidden_property(h, ZIP_KEYS, NanBox::handle(keys_arr.to_raw()));
        }
        Ok(NanBox::handle(h.to_raw()))
    }

    /// Collects `padding[0..]` (the per-iterator filler values) by iterating the
    /// padding iterable into a list.
    fn zip_collect_padding(&mut self, padding: NanBox) -> Result<Vec<NanBox>, ExecError> {
        self.iterate_values(padding)
    }

    /// GetIteratorFlattenable(value, REJECT_PRIMITIVES) for zip: the value must be
    /// an object; use its `[Symbol.iterator]` if callable, else (a built-in
    /// iterable) drain it, else use the object itself as the iterator (it must
    /// carry its own `next`).
    fn zip_open_iterator(&mut self, value: NanBox) -> Result<Handle, ExecError> {
        if !self.is_object_value(value) {
            return Err(self.type_error("Iterator.zip: an iterable is not an object"));
        }
        let vh = value.as_handle().map(Handle::from_raw).unwrap();
        let iter_fn = self.find_iterator_fn(vh)?;
        if iter_fn.is_some_and(|fv| {
            fv.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        }) || self.realm.array_elements(vh).is_some()
            || self.realm.collection_entries(vh).is_some()
            || self.realm.get_property(vh, GEN_BUF).is_some()
        {
            return self.get_iter_object(value);
        }
        // No `[Symbol.iterator]`: the object itself is the iterator.
        Ok(vh)
    }

    /// Closes every still-open iterator recorded on a zip-iterator object.
    fn zip_close_all(&mut self, iters_arr: Handle, skip: usize) -> Result<(), ExecError> {
        let iters = self
            .realm
            .array_elements(iters_arr)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        // Close the still-open iterators in REVERSE index order (per Iterator.zip:
        // the abrupt path runs IteratorClose from the last opened down to the
        // first), skipping the one that already aborted.
        for (i, it) in iters.iter().enumerate().rev() {
            if i == skip {
                continue;
            }
            if let Some(ih) = it.as_handle().map(Handle::from_raw) {
                let _ = self.iterator_close(ih);
            }
        }
        Ok(())
    }

    /// `%ZipIteratorPrototype%.next` — produces one zipped array (or null-proto
    /// object for zipKeyed), honoring the shortest/longest/strict mode.
    pub(crate) fn iter_zip_next(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Zip Iterator next called on non-object"));
        };
        if self.realm.get_property(h, ZIP_DONE).is_some() {
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        let iters_arr = self
            .realm
            .get_property(h, ZIP_ITERS)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .unwrap();
        let nexts_arr = self
            .realm
            .get_property(h, ZIP_NEXTS)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .unwrap();
        let mode = self
            .realm
            .get_property(h, ZIP_MODE)
            .and_then(|n| n.as_number())
            .unwrap_or(0.0) as u8;
        let count = self.realm.array_elements(iters_arr).map_or(0, |e| e.len());
        if count == 0 {
            // Zero iterators: zip is immediately done.
            self.realm
                .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        let mut values: Vec<NanBox> = Vec::with_capacity(count);
        let mut any_live = false;
        let mut all_done_strict = true;
        for i in 0..count {
            let fin_arr = self
                .realm
                .get_property(h, ZIP_FINISHED)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                .unwrap();
            let already_fin = self.realm.get_element(fin_arr, i);
            if mode == 1 && self.realm.truthy(already_fin) {
                // Longest mode: a finished iterator contributes its padding value.
                let pad_arr = self
                    .realm
                    .get_property(h, ZIP_PADDING)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                    .unwrap();
                values.push(self.realm.get_element(pad_arr, i));
                continue;
            }
            let it = self.realm.get_element(iters_arr, i);
            let next = self.realm.get_element(nexts_arr, i);
            let ih = it.as_handle().map(Handle::from_raw).unwrap();
            let step = self.iter_step(ih, next);
            match step {
                Ok(Some(v)) => {
                    values.push(v);
                    any_live = true;
                    all_done_strict = false;
                }
                Ok(None) => {
                    match mode {
                        0 => {
                            // shortest: this iterator is done → close the others, finish.
                            self.realm
                                .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
                            self.zip_close_all(iters_arr, i)?;
                            return Ok(self.iter_result(NanBox::undefined(), true));
                        }
                        2 => {
                            // strict: if this is the first iterator's first done, the
                            // others must also be done; otherwise a TypeError.
                            if i == 0 {
                                // Verify all remaining are also done.
                                let ok =
                                    self.zip_strict_verify_rest(iters_arr, nexts_arr, count)?;
                                self.realm
                                    .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
                                if ok {
                                    return Ok(self.iter_result(NanBox::undefined(), true));
                                }
                                return Err(self.type_error(
                                    "Iterator.zip strict mode: iterators have different lengths",
                                ));
                            }
                            // A later iterator finished before the first → length mismatch.
                            self.realm
                                .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
                            self.zip_close_all(iters_arr, i)?;
                            return Err(self.type_error(
                                "Iterator.zip strict mode: iterators have different lengths",
                            ));
                        }
                        _ => {
                            // longest: mark finished, contribute padding.
                            let fin_arr2 = self
                                .realm
                                .get_property(h, ZIP_FINISHED)
                                .and_then(|v| v.as_handle())
                                .map(Handle::from_raw)
                                .unwrap();
                            self.realm.set_element(fin_arr2, i, NanBox::boolean(true));
                            let pad_arr = self
                                .realm
                                .get_property(h, ZIP_PADDING)
                                .and_then(|v| v.as_handle())
                                .map(Handle::from_raw)
                                .unwrap();
                            values.push(self.realm.get_element(pad_arr, i));
                        }
                    }
                }
                Err(e) => {
                    self.realm
                        .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
                    self.zip_close_all(iters_arr, i)?;
                    return Err(e);
                }
            }
        }
        let _ = all_done_strict;
        // Longest mode: when every iterator is finished, the zip is done.
        if mode == 1 && !any_live {
            self.realm
                .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
            return Ok(self.iter_result(NanBox::undefined(), true));
        }
        // Build the result: an array (zip) or a null-proto object keyed by the
        // recorded keys (zipKeyed).
        let result = if let Some(keys_arr) = self
            .realm
            .get_property(h, ZIP_KEYS)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            let obj = self.realm.new_object_with_proto(None);
            for (i, v) in values.into_iter().enumerate() {
                let k = self.realm.get_element(keys_arr, i);
                let ks = self.realm.to_display_string(k);
                self.realm.set_property(obj, &ks, v);
            }
            NanBox::handle(obj.to_raw())
        } else {
            NanBox::handle(self.realm.new_array(values).to_raw())
        };
        Ok(self.iter_result(result, false))
    }

    /// Strict-mode helper: after the first iterator reports done, verify every
    /// other iterator is also done (a live value is a length mismatch). Closes any
    /// iterator that is *not* done. Returns Ok(true) if all are done.
    fn zip_strict_verify_rest(
        &mut self,
        iters_arr: Handle,
        nexts_arr: Handle,
        count: usize,
    ) -> Result<bool, ExecError> {
        for i in 1..count {
            let it = self.realm.get_element(iters_arr, i);
            let next = self.realm.get_element(nexts_arr, i);
            let ih = it.as_handle().map(Handle::from_raw).unwrap();
            match self.iter_step(ih, next)? {
                None => {}
                Some(_) => {
                    // This one still had a value → mismatch; close the rest.
                    for j in (i + 1)..count {
                        let oj = self.realm.get_element(iters_arr, j);
                        if let Some(oh) = oj.as_handle().map(Handle::from_raw) {
                            let _ = self.iterator_close(oh);
                        }
                    }
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// `%ZipIteratorPrototype%.return` — closes every open underlying iterator.
    pub(crate) fn iter_zip_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw) {
            let already = self.realm.get_property(h, ZIP_DONE).is_some();
            self.realm
                .set_hidden_property(h, ZIP_DONE, NanBox::boolean(true));
            if !already
                && let Some(iters_arr) = self
                    .realm
                    .get_property(h, ZIP_ITERS)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
            {
                let count = self.realm.array_elements(iters_arr).map_or(0, |e| e.len());
                self.zip_close_all(iters_arr, count + 1)?;
            }
        }
        Ok(self.iter_result(NanBox::undefined(), true))
    }

    /// `%ConcatIteratorPrototype%.return` — closes the active inner iterator (if
    /// any) and marks the concat result done.
    pub(crate) fn iter_concat_return(&mut self, this: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw) {
            let already = self.realm.get_property(h, HELPER_DONE).is_some();
            self.mark_helper_done(h);
            if !already
                && let Some(inner) = self
                    .realm
                    .get_property(h, HELPER_INNER)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
            {
                self.iterator_close(inner)?;
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
            let ith = match self.get_iter_object(src) {
                Ok(ih) => ih,
                Err(e) => {
                    self.mark_helper_done(h);
                    return Err(e);
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
                let sym = self.well_known_symbol("asyncIterator");
                let akey = self.member_key(sym);
                let has_async = self
                    .read_member(h, &akey)?
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
                has_async
                    || self
                        .find_iterator_fn(h)?
                        .filter(|f| {
                            f.as_handle()
                                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                        })
                        .is_some()
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
                let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("iterator result is not an object"));
                };
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
    fn async_iterator_of(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
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
            let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                return Err(self.type_error("iterator result is not an object"));
            };
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
        if let Some(elems) = self.realm.elements_vec(h) {
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
                let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("iterator result is not an object"));
                };
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
                    None => Err(ExecError::Throw(self.new_str("iterator is not an object"))),
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
            None => Err(ExecError::Throw(self.new_str("iterator is not an object"))),
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
