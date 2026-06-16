use super::*;

impl<'a> Interp<'a> {
    /// Runs an ES2025 `Iterator.prototype` helper (`map`/`filter`/`take`/`drop`/
    /// `flatMap`/`reduce`/`toArray`/`forEach`/`some`/`every`/`find`) on the
    /// receiver iterator `this_val`. The receiver is drained through the iterator
    /// protocol (`iterate_values`); the lazy/element-returning helpers return a
    /// fresh generator-backed iterator, the rest a direct value. A non-iterator
    /// `this` is a TypeError (`requires that 'this' be an Iterator`).
    pub(crate) fn iterator_proto_helper(
        &mut self,
        method: &str,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let values = self.iterate_values(this_val).map_err(|_| {
            self.type_error(&alloc::format!(
                "Iterator.prototype.{method} requires that 'this' be an Iterator"
            ))
        })?;
        let f = args.first().copied().unwrap_or(NanBox::undefined());
        // The callback-taking helpers require a callable first argument.
        let needs_fn = matches!(
            method,
            "map" | "filter" | "flatMap" | "reduce" | "forEach" | "some" | "every" | "find"
        );
        if needs_fn
            && !f
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return Err(self.type_error(&alloc::format!(
                "Iterator.prototype.{method} called with a non-callable argument"
            )));
        }
        Ok(match method {
            "toArray" => NanBox::handle(self.realm.new_array(values).to_raw()),
            "map" => {
                let mut out = Vec::with_capacity(values.len());
                for (i, v) in values.into_iter().enumerate() {
                    out.push(self.call(f, &[v, NanBox::number(i as f64)])?);
                }
                self.make_generator(out)
            }
            "filter" => {
                let mut out = Vec::new();
                for (i, v) in values.into_iter().enumerate() {
                    let keep = self.call(f, &[v, NanBox::number(i as f64)])?;
                    if self.realm.truthy(keep) {
                        out.push(v);
                    }
                }
                self.make_generator(out)
            }
            "flatMap" => {
                let mut out = Vec::new();
                for (i, v) in values.into_iter().enumerate() {
                    let r = self.call(f, &[v, NanBox::number(i as f64)])?;
                    out.extend(self.iterate_values(r).unwrap_or_else(|_| alloc::vec![r]));
                }
                self.make_generator(out)
            }
            "take" => {
                let n = self.realm.to_number(f).max(0.0) as usize;
                self.make_generator(values.into_iter().take(n).collect())
            }
            "drop" => {
                let n = self.realm.to_number(f).max(0.0) as usize;
                self.make_generator(values.into_iter().skip(n).collect())
            }
            "forEach" => {
                for (i, v) in values.into_iter().enumerate() {
                    self.call(f, &[v, NanBox::number(i as f64)])?;
                }
                NanBox::undefined()
            }
            "some" | "every" | "find" => {
                for (i, v) in values.into_iter().enumerate() {
                    let r = self.call(f, &[v, NanBox::number(i as f64)])?;
                    let t = self.realm.truthy(r);
                    match method {
                        "every" if !t => return Ok(NanBox::boolean(false)),
                        "some" if t => return Ok(NanBox::boolean(true)),
                        "find" if t => return Ok(v),
                        _ => {}
                    }
                }
                match method {
                    "every" => NanBox::boolean(true),
                    "some" => NanBox::boolean(false),
                    _ => NanBox::undefined(), // find → undefined
                }
            }
            // reduce
            _ => {
                let mut it = values.into_iter();
                let mut acc = if args.len() >= 2 {
                    args[1]
                } else {
                    match it.next() {
                        Some(v) => v,
                        None => {
                            return Err(
                                self.type_error("Reduce of empty iterator with no initial value")
                            );
                        }
                    }
                };
                for v in it {
                    acc = self.call(f, &[acc, v])?;
                }
                acc
            }
        })
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
                return Err(ExecError::Throw(self.new_str("iterator is not an object")));
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
                    return Err(ExecError::Throw(
                        self.new_str("iterator result is not an object"),
                    ));
                };
                let done = self.read_member(rh, "done")?;
                if self.realm.truthy(done) {
                    break;
                }
                out.push(self.read_member(rh, "value")?);
                if out.len() > GEN_CAP {
                    return Err(ExecError::Throw(self.new_str("iterator did not terminate")));
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

    /// Whether `v` is iterable — a string, array, typed array, or any object with
    /// a resolvable `[Symbol.iterator]` method. Used to distinguish a genuine
    /// throw inside the iterator protocol (propagate) from a non-iterable
    /// array-like source (fall back to index reads).
    pub(crate) fn value_is_iterable(&mut self, v: NanBox) -> bool {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return false;
        };
        if self.realm.string_value(h).is_some()
            || self.realm.is_array(h)
            || self.realm.typed_kind(h).is_some()
        {
            return true;
        }
        matches!(self.find_iterator_fn(h), Ok(Some(_)))
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
            && let Some(len) = self.realm.array_length(h)
        {
            for i in 0..len {
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

    /// Runs `body` once per `item`, binding the loop variable (a fresh scope per
    /// iteration for a declared head).
    /// Obtains a *user* iterable's iterator object (calling `[Symbol.iterator]`
    /// once), for the lazy `for-of` path. Returns `None` for built-in iterables
    /// (arrays/strings/Maps/Sets) and generator values, which `iterate_values`
    /// drains eagerly, and for non-iterables.
    pub(crate) fn for_of_get_iterator(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        if self.realm.array_elements(h).is_some()
            || self.realm.string_value(h).is_some()
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
    pub(crate) fn iterator_close(&mut self, ih: Handle) -> Result<(), ExecError> {
        let ret = self.read_member(ih, "return")?;
        if ret
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            self.call_with_this(ret, NanBox::handle(ih.to_raw()), &[])?;
        }
        Ok(())
    }
}
