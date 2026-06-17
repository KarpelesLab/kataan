use super::*;

impl<'a> Interp<'a> {
    pub(crate) fn is_callable(&self, handle: Handle) -> bool {
        self.realm.native_at(handle).is_some()
            || self.realm.function_at(handle).is_some()
            || self.realm.bound_native_at(handle).is_some()
            // A bound function (`fn.bind(...)`) is callable.
            || self.realm.get_property(handle, BOUND_TARGET).is_some()
            // A proxy is callable iff its target is.
            || self
                .realm
                .proxy_at(handle)
                .is_some_and(|(t, _)| self.is_callable(t))
    }

    /// `IsCallable(v)` for a [`NanBox`] value: true iff `v` is a callable object.
    pub(crate) fn is_callable_value(&self, v: NanBox) -> bool {
        v.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.is_callable(h))
    }

    /// Throws a `TypeError("<what> is not a function")` unless `v` is callable;
    /// the upfront IsCallable guard the iteration built-ins (`forEach`/`map`/
    /// `reduce`/`find`/…) apply to their callback before any element processing.
    pub(crate) fn require_callable(&mut self, v: NanBox, what: &str) -> Result<(), ExecError> {
        if self.is_callable_value(v) {
            Ok(())
        } else {
            Err(self.type_error(&alloc::format!("{what} is not a function")))
        }
    }

    /// The abstract operation `IsConstructor(value)` — whether `value` has a
    /// `[[Construct]]` internal method (i.e. `new value` / `Reflect.construct`
    /// would dispatch rather than throw "is not a constructor"). Mirrors the
    /// acceptance set of [`Self::construct`].
    pub(crate) fn is_constructor_value(&self, value: NanBox) -> bool {
        let Some(handle) = value.as_handle().map(Handle::from_raw) else {
            return false;
        };
        // A bound function constructs iff its target does.
        if let Some(target) = self.realm.get_property(handle, BOUND_TARGET) {
            return self.is_constructor_value(target);
        }
        // A proxy constructs iff its target does.
        if let Some((target, _)) = self.realm.proxy_at(handle) {
            return self.is_constructor_value(NanBox::handle(target.to_raw()));
        }
        // A user class always constructs.
        if self.realm.class_at(handle).is_some() {
            return true;
        }
        // A user function constructs unless it is an arrow / generator / async
        // function (and, per spec, a concise method — which this model does not
        // distinguish, so an object-literal method is treated as a constructor).
        if let Some((func_id, _)) = self.realm.function_at(handle) {
            let def = self.functions[func_id as usize];
            return !(def.is_arrow || def.is_generator || def.is_async);
        }
        // `Object` / `Array` are namespace objects callable as constructors.
        if self.current.get("Object").and_then(|v| v.as_handle()) == value.as_handle()
            || self.current.get("Array").and_then(|v| v.as_handle()) == value.as_handle()
        {
            return true;
        }
        // A built-in native: constructs iff its dispatch id is a recognised
        // constructor (the abstract `%TypedArray%` intrinsic and all method/utility
        // natives are *not* constructors).
        if let Some(id) = self.realm.native_at(handle) {
            if id == N_TYPED_ARRAY_ABSTRACT {
                return false;
            }
            return is_native_constructor(id);
        }
        false
    }

    /// `IsConstructor(value)` — alias for [`Self::is_constructor_value`], kept for
    /// call sites that use this shorter name.
    pub(crate) fn is_constructor(&self, value: NanBox) -> bool {
        self.is_constructor_value(value)
    }

    /// Builds a bound function (`Function.prototype.bind`): an object recording
    /// the target, the bound `this`, and the leading bound arguments under
    /// reserved hidden keys. Calling it forwards to the target.
    pub(crate) fn make_bound_function(
        &mut self,
        target: NanBox,
        this_val: NanBox,
        bound: Vec<NanBox>,
    ) -> NanBox {
        let obj = self.realm.new_object();
        // A bound function's `[[Prototype]]` is the target function's
        // `[[Prototype]]` (ordinarily `%Function.prototype%`), not
        // `Object.prototype` (which `new_object` defaults to).
        let target_proto = target
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|t| self.realm.object_proto(t));
        self.realm.set_object_proto(obj, target_proto);
        self.realm.set_hidden_property(obj, BOUND_TARGET, target);
        self.realm.set_hidden_property(obj, BOUND_THIS, this_val);
        let arr = self.realm.new_array(bound);
        self.realm
            .set_hidden_property(obj, BOUND_ARGS, NanBox::handle(arr.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Calls `callee` with an explicit `this` (a method receiver, or `undefined`
    /// for a plain call).
    pub(crate) fn call_with_this(
        &mut self,
        callee: NanBox,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let Some(raw) = callee.as_handle() else {
            // Calling a non-object (a primitive `undefined`/`null`/number/…) is a
            // JS `TypeError` — catchable by user `try/catch` — not an internal
            // engine error.
            let m = self.new_str("is not a function");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        let handle = Handle::from_raw(raw);
        // `Array(...)` without `new` behaves like `new Array(...)`.
        if self.current.get("Array").and_then(|v| v.as_handle()) == callee.as_handle() {
            return self.construct(callee, args);
        }
        // `Object(value)` (ToObject): `null`/`undefined` → a new object; an object is
        // returned as-is; a primitive is boxed in its wrapper.
        if self.current.get("Object").and_then(|v| v.as_handle()) == callee.as_handle() {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            return Ok(self.coerce_to_object(v));
        }
        // A bound function: prepend the bound `this`/args and forward.
        if let Some(target) = self.realm.get_property(handle, BOUND_TARGET) {
            let bthis = self
                .realm
                .get_property(handle, BOUND_THIS)
                .unwrap_or(NanBox::undefined());
            let mut all = self
                .realm
                .get_property(handle, BOUND_ARGS)
                .and_then(|a| a.as_handle())
                .map(Handle::from_raw)
                .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                .unwrap_or_default();
            all.extend_from_slice(args);
            return self.call_with_this(target, bthis, &all);
        }
        // A callable proxy: route through its `apply` trap, or call the target.
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            let trap = self
                .realm
                .get_property(handler, "apply")
                .unwrap_or(NanBox::undefined());
            // A present but non-callable `apply` trap is a TypeError.
            if !matches!(trap.unpack(), Unpacked::Undefined | Unpacked::Null) {
                if !trap
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("proxy apply trap is not a function"));
                }
                let arr = self.realm.new_array(args.to_vec());
                let handler_box = NanBox::handle(handler.to_raw());
                return self.call_with_this(
                    trap,
                    handler_box,
                    &[
                        NanBox::handle(target.to_raw()),
                        this_val,
                        NanBox::handle(arr.to_raw()),
                    ],
                );
            }
            return self.call_with_this(NanBox::handle(target.to_raw()), this_val, args);
        }
        // A built-in function dispatches directly, with the receiver available as
        // `this` (for the `Object.prototype.*` methods called via `.call`).
        if let Some(id) = self.realm.native_at(handle) {
            let saved = core::mem::replace(&mut self.this_val, this_val);
            let r = self.call_native(id, args);
            self.this_val = saved;
            return r;
        }
        // A bound native (promise resolve/reject) carries its target.
        if let Some((id, target)) = self.realm.bound_native_at(handle) {
            // A first-class `ArrayBuffer.prototype.<method>`: reject a `this` that is
            // not an Object with an `[[ArrayBufferData]]` slot, then dispatch.
            if id == N_AB_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let ok = this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.get_property(h, ARRAY_BUFFER_BYTES).is_some());
                if !ok {
                    return Err(self.type_error(&alloc::format!(
                        "ArrayBuffer.prototype.{name} called on a non-ArrayBuffer object"
                    )));
                }
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `BigInt.prototype.<method>`: `thisBigIntValue(this)`
            // must yield a BigInt (the `this` is a BigInt or a BigInt wrapper
            // object), else a `TypeError`.
            if id == N_BIGINT_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let big = self.this_bigint_value(this_val);
                let Some(big) = big else {
                    return Err(self.type_error(&alloc::format!(
                        "BigInt.prototype.{name} requires that 'this' be a BigInt"
                    )));
                };
                let prim = NanBox::handle(self.realm.new_bigint(big).to_raw());
                return Ok(self
                    .call_method(prim, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `Number.prototype.<method>`: `thisNumberValue(this)`
            // must yield a Number (a Number primitive or a Number wrapper), else a
            // `TypeError`. The recovered primitive is then dispatched, so
            // `Number.prototype.toString.call(255, 16)` is `"ff"`.
            if id == N_NUMBER_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let Some(n) = self.this_number_value(this_val) else {
                    return Err(self.type_error(&alloc::format!(
                        "Number.prototype.{name} requires that 'this' be a Number"
                    )));
                };
                return Ok(self
                    .call_method(NanBox::number(n), &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `Boolean.prototype.<method>`: `thisBooleanValue(this)`
            // must yield a Boolean (a Boolean primitive or a Boolean wrapper), else
            // a `TypeError`.
            if id == N_BOOLEAN_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let Some(b) = self.this_boolean_value(this_val) else {
                    return Err(self.type_error(&alloc::format!(
                        "Boolean.prototype.{name} requires that 'this' be a Boolean"
                    )));
                };
                return Ok(self
                    .call_method(NanBox::boolean(b), &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `String.prototype.<method>`: RequireObjectCoercible +
            // ToString the call's `this`, then dispatch on the resulting string.
            // `toString`/`valueOf` instead require an actual String value.
            if id == N_STRING_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                if name == "toString" || name == "valueOf" {
                    // thisStringValue: a string primitive or a String wrapper.
                    let s = if let Some(s) = this_val
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.string_value(h))
                    {
                        Some(s)
                    } else {
                        this_val
                            .as_handle()
                            .map(Handle::from_raw)
                            .and_then(|h| self.realm.get_property(h, PRIM_WRAP))
                            .and_then(|p| p.as_handle())
                            .map(Handle::from_raw)
                            .and_then(|ph| self.realm.string_value(ph))
                    };
                    let Some(s) = s else {
                        return Err(self.type_error(&alloc::format!(
                            "String.prototype.{name} requires that 'this' be a String"
                        )));
                    };
                    return Ok(self.new_str(&s));
                }
                // RequireObjectCoercible: `undefined`/`null` `this` is a TypeError.
                if matches!(this_val.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(self.type_error(&alloc::format!(
                        "String.prototype.{name} called on null or undefined"
                    )));
                }
                let s = self.coerce_to_string(this_val)?;
                let str_recv = self.new_str(&s);
                return Ok(self
                    .call_method(str_recv, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `Date.prototype.<method>`: the call's `this` must have
            // a `[[DateValue]]` (be a Date), else a `TypeError`.
            if id == N_DATE_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let is_date = this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.date_at(h).is_some());
                if !is_date {
                    return Err(self.type_error(&alloc::format!(
                        "Date.prototype.{name} is not a Date object"
                    )));
                }
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `Iterator.prototype.<helper>` (map/filter/take/…) run
            // on the call's `this` iterator.
            if id == N_ITERATOR_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                return self.iterator_proto_helper(&name, this_val, args);
            }
            // A first-class `RegExp.prototype.<method>` (`exec`/`test`/`compile`/
            // `toString` or a `@@match`/… symbol method): brand-validation is per
            // method inside the dispatcher.
            if id == N_REGEXP_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                return self.regexp_proto_dispatch(&name, this_val, args);
            }
            // A `get RegExp.prototype.<accessor>` getter (source/flags/flag getters).
            if id == N_REGEXP_ACCESSOR {
                let name = self.realm.string_value(target).unwrap_or_default();
                return self.regexp_accessor_dispatch(&name, this_val);
            }
            // `get RegExp[Symbol.species]` — returns the `this` receiver.
            if id == N_REGEXP_SPECIES {
                return Ok(this_val);
            }
            // A RegExp legacy static getter (Annex B.2.5): `target` carries the
            // field selector. Brand-check `this === %RegExp%` (only the original
            // constructor passes; a subclass / instance / primitive throws).
            if id == N_REGEXP_LEGACY_GET {
                let selector = self.realm.string_value(target).unwrap_or_default();
                if this_val.as_handle().map(Handle::from_raw)
                    != Some(self.regexp_constructor_handle()?)
                {
                    return Err(self.type_error(
                        "RegExp legacy static property getter called on a non-RegExp receiver",
                    ));
                }
                let st = self.realm.legacy_regexp();
                let bytes = match selector.as_str() {
                    "input" => st.input.clone(),
                    "lastMatch" => st.last_match.clone(),
                    "lastParen" => st.last_paren.clone(),
                    "leftContext" => st.left_context.clone(),
                    "rightContext" => st.right_context.clone(),
                    s if s.starts_with('$') => {
                        let n: usize = s[1..].parse().unwrap_or(0);
                        st.parens
                            .get(n.wrapping_sub(1))
                            .cloned()
                            .unwrap_or_default()
                    }
                    _ => alloc::vec::Vec::new(),
                };
                return Ok(self.new_str_bytes(bytes));
            }
            // A `get %TypedArray%.prototype.<accessor>` getter: compute the
            // buffer/byteLength/byteOffset/length of the `this` typed array (or a
            // TypeError if `this` is not a typed array).
            if id == N_TYPED_ARRAY_ACCESSOR {
                let name = self.realm.string_value(target).unwrap_or_default();
                let Some(h) = this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.typed_kind(*h).is_some())
                else {
                    return Err(self.type_error(&alloc::format!(
                        "get TypedArray.prototype.{name} called on a non-TypedArray object"
                    )));
                };
                let kind = self.realm.typed_kind(h).unwrap();
                let bpe = f64::from(TYPED_ARRAY_KINDS[kind as usize].1);
                return Ok(match name.as_str() {
                    "buffer" => self
                        .realm
                        .typed_array_object(h)
                        .map_or(NanBox::undefined(), |b| NanBox::handle(b.to_raw())),
                    "byteOffset" => {
                        NanBox::number(self.realm.typed_byte_offset(h).unwrap_or(0) as f64)
                    }
                    "byteLength" => {
                        NanBox::number(self.realm.typed_len(h).unwrap_or(0) as f64 * bpe)
                    }
                    _ => NanBox::number(self.realm.typed_len(h).unwrap_or(0) as f64),
                });
            }
            // A `get DataView.prototype.<accessor>` getter (buffer/byteLength/
            // byteOffset): the `this` must have a `[[DataView]]` internal slot,
            // else a TypeError (RequireInternalSlot). A view over a detached
            // buffer reports `byteLength`/`byteOffset` 0 only after the detach
            // throw — here a detached buffer is a TypeError for byteLength.
            if id == N_DATA_VIEW_ACCESSOR {
                let name = self.realm.string_value(target).unwrap_or_default();
                let Some(h) = this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.get_property(*h, DATA_VIEW_BUF).is_some())
                else {
                    return Err(self.type_error(&alloc::format!(
                        "get DataView.prototype.{name} called on a non-DataView object"
                    )));
                };
                let buf = self.realm.get_property(h, DATA_VIEW_BUF).unwrap();
                let buf_h = buf.as_handle().map(Handle::from_raw);
                // A detached buffer makes `byteLength`/`byteOffset` a TypeError; `buffer`
                // still returns the (detached) buffer.
                if name != "buffer"
                    && let Some(bh) = buf_h
                {
                    self.guard_detached_buffer(bh)?;
                }
                return Ok(match name.as_str() {
                    "buffer" => buf,
                    "byteOffset" => self
                        .realm
                        .get_property(h, DATA_VIEW_OFF)
                        .unwrap_or(NanBox::number(0.0)),
                    _ => {
                        // byteLength: an explicit recorded length, else the rest of
                        // the (live) buffer past the view's byte offset.
                        if let Some(len) = self
                            .realm
                            .get_property(h, DATA_VIEW_LEN)
                            .and_then(|n| n.as_number())
                        {
                            NanBox::number(len)
                        } else {
                            let total = buf_h
                                .and_then(|bh| self.array_buffer_bytes(bh))
                                .and_then(|bh| self.realm.bytes_len(bh))
                                .unwrap_or(0);
                            let off = self
                                .realm
                                .get_property(h, DATA_VIEW_OFF)
                                .and_then(|n| n.as_number())
                                .unwrap_or(0.0) as usize;
                            NanBox::number(total.saturating_sub(off) as f64)
                        }
                    }
                });
            }
            // A first-class `DataView.prototype.<method>` (getInt8/setFloat64/…):
            // the `this` must have a `[[DataView]]` internal slot, else a
            // TypeError; then dispatch through `call_method`.
            if id == N_DATA_VIEW_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let ok = this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.get_property(h, DATA_VIEW_BUF).is_some());
                if !ok {
                    return Err(self.type_error(&alloc::format!(
                        "DataView.prototype.{name} called on a non-DataView object"
                    )));
                }
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A `get ArrayBuffer.prototype.<accessor>` getter (byteLength/
            // maxByteLength/resizable/detached): the `this` must have an
            // `[[ArrayBufferData]]` internal slot, else a TypeError.
            if id == N_AB_ACCESSOR {
                let name = self.realm.string_value(target).unwrap_or_default();
                let Some(h) = this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.get_property(*h, ARRAY_BUFFER_BYTES).is_some())
                else {
                    return Err(self.type_error(&alloc::format!(
                        "get ArrayBuffer.prototype.{name} called on a non-ArrayBuffer object"
                    )));
                };
                let detached = self.realm.get_property(h, ARRAY_BUFFER_DETACHED).is_some();
                let max = self.realm.get_property(h, ARRAY_BUFFER_MAXLEN);
                return Ok(match name.as_str() {
                    "detached" => NanBox::boolean(detached),
                    "resizable" => NanBox::boolean(max.is_some()),
                    "byteLength" => {
                        if detached {
                            NanBox::number(0.0)
                        } else {
                            let len = self
                                .array_buffer_bytes(h)
                                .and_then(|bh| self.realm.bytes_len(bh))
                                .unwrap_or(0);
                            NanBox::number(len as f64)
                        }
                    }
                    // maxByteLength: the recorded max for a resizable buffer; else
                    // the current byteLength (0 once detached).
                    _ => match max {
                        Some(m) => m,
                        None => {
                            if detached {
                                NanBox::number(0.0)
                            } else {
                                let len = self
                                    .array_buffer_bytes(h)
                                    .and_then(|bh| self.realm.bytes_len(bh))
                                    .unwrap_or(0);
                                NanBox::number(len as f64)
                            }
                        }
                    },
                });
            }
            // A first-class `%TypedArray%.prototype.<method>`: reject a `this`
            // without a `[[TypedArrayName]]` internal slot, then dispatch directly
            // (no plain-Array conversion — the typed-array method returns a
            // same-kind view where the spec requires).
            if id == N_TYPED_ARRAY_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let th = this_val.as_handle().map(Handle::from_raw);
                let ok = th.is_some_and(|h| self.realm.typed_kind(h).is_some());
                if !ok {
                    return Err(self.type_error(&alloc::format!(
                        "TypedArray.prototype.{name} called on a non-TypedArray object"
                    )));
                }
                // (ValidateTypedArray — the detached-buffer TypeError — is applied
                // centrally in `call_method` for the data-accessing methods.)
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `Set.prototype.<method>`: the receiver must be a
            // non-weak Set (`[[SetData]]`), else a TypeError — so
            // `Set.prototype.add.call(new Map(), …)` / `.call({})` reject.
            if id == N_SET_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let ok = this_val.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.collection_is_set(h) == Some(true)
                        && !self.realm.collection_is_weak(h)
                });
                if !ok {
                    return Err(self.type_error(&alloc::format!(
                        "Set.prototype.{name} requires that 'this' be a Set"
                    )));
                }
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `Map.prototype.<method>`: the receiver must be a
            // non-weak Map (`[[MapData]]`), else a TypeError.
            if id == N_MAP_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let ok = this_val.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.collection_is_set(h) == Some(false)
                        && !self.realm.collection_is_weak(h)
                });
                if !ok {
                    return Err(self.type_error(&alloc::format!(
                        "Map.prototype.{name} requires that 'this' be a Map"
                    )));
                }
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `WeakMap.prototype.<method>`: the receiver must be a
            // WeakMap, else a TypeError.
            if id == N_WEAKMAP_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let ok = this_val.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.collection_is_set(h) == Some(false)
                        && self.realm.collection_is_weak(h)
                });
                if !ok {
                    return Err(self.type_error(&alloc::format!(
                        "WeakMap.prototype.{name} requires that 'this' be a WeakMap"
                    )));
                }
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `WeakSet.prototype.<method>`: the receiver must be a
            // WeakSet, else a TypeError.
            if id == N_WEAKSET_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                let ok = this_val.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.collection_is_set(h) == Some(true)
                        && self.realm.collection_is_weak(h)
                });
                if !ok {
                    return Err(self.type_error(&alloc::format!(
                        "WeakSet.prototype.{name} requires that 'this' be a WeakSet"
                    )));
                }
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A first-class `Array.prototype.<method>`: run that array method on the
            // call's `this` (e.g. `Array.prototype.slice.call(arguments)`).
            if id == N_ARRAY_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                // Every `Array.prototype` method begins with `ToObject(this value)`,
                // which throws a TypeError for `null`/`undefined`.
                if matches!(this_val.unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error(&alloc::format!(
                        "Array.prototype.{name} called on null or undefined"
                    )));
                }
                // A *generic* `Array.prototype.<m>` whose `this` is a typed array views
                // it as an array-like and builds a plain `Array` — unlike
                // `%TypedArray%.prototype.<m>` (a direct `ta.<m>()`), which returns a
                // same-kind typed array. For the collection-returning methods we
                // materialize the view's elements into a plain array first, so e.g.
                // `Array.prototype.slice.call(u8)` returns a real, concat-spreadable
                // `Array`. We do NOT convert for typed-array-specific/mutating methods
                // (`set`, `subarray`, `sort`, `fill`, …) — those must see the live view.
                const PLAIN_ARRAY_RESULT: &[&str] = &[
                    "slice",
                    "map",
                    "filter",
                    "flat",
                    "flatMap",
                    "concat",
                    "toReversed",
                    "toSorted",
                    "with",
                ];
                // ToObject(this): a primitive `this` (a boolean/number/string/
                // symbol/bigint, e.g. `Array.prototype.reduce.call("abc", …)`) is
                // boxed to its wrapper object so its array-like indexed properties
                // and `length` are read generically.
                let this_obj = if this_val.as_handle().is_none() {
                    self.coerce_to_object(this_val)
                } else {
                    this_val
                };
                let this_eff = match this_obj.as_handle().map(Handle::from_raw) {
                    Some(h)
                        if PLAIN_ARRAY_RESULT.contains(&name.as_str())
                            && self.realm.typed_kind(h).is_some() =>
                    {
                        let elems = self.realm.typed_elements(h).unwrap_or_default();
                        NanBox::handle(self.realm.new_array(elems).to_raw())
                    }
                    _ => this_obj,
                };
                return Ok(self
                    .call_method(this_eff, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A readable static method: `target` is `[constructor, name]`. Route to the
            // constructor's static dispatch regardless of the call's `this` (so a detached
            // `var f = Number.isInteger; f(x)` works like `Number.isInteger(x)`).
            if id == N_STATIC_METHOD {
                let pair = self.realm.array_elements(target).map(<[_]>::to_vec);
                if let Some(pair) = pair
                    && let Some(ctor) = pair.first().copied()
                    && let Some(name_v) = pair.get(1).copied()
                    && let Some(name) = name_v
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.string_value(h))
                {
                    return Ok(self
                        .call_method(ctor, &name, args)?
                        .unwrap_or(NanBox::undefined()));
                }
                return Ok(NanBox::undefined());
            }
            // A WASM export wrapper: decode the carried module, instantiate, and
            // invoke the named export through the JS-value boundary.
            if id == N_WASM_CALL {
                return self.call_wasm_export(target, args);
            }
            // `WebAssembly.Global` `.value` getter / setter (bound to the global).
            if id == N_WASM_GLOBAL_GET {
                return Ok(self
                    .realm
                    .get_property(target, WASM_GLOBAL_VALUE)
                    .unwrap_or(NanBox::undefined()));
            }
            if id == N_WASM_GLOBAL_SET {
                if !self
                    .realm
                    .get_property(target, WASM_GLOBAL_MUTABLE)
                    .is_some_and(|v| self.realm.truthy(v))
                {
                    return Err(self.wasm_type_error("WebAssembly.Global is immutable"));
                }
                let new_val = args.first().copied().unwrap_or(NanBox::undefined());
                let ty = self
                    .realm
                    .get_property(target, WASM_GLOBAL_TYPE)
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                let coerced = self.wasm_coerce_global(&ty, new_val);
                self.realm
                    .set_hidden_property(target, WASM_GLOBAL_VALUE, coerced);
                return Ok(NanBox::undefined());
            }
            // `WebAssembly.Memory.prototype.buffer` getter.
            if id == N_WASM_MEM_BUFFER_GET {
                return Ok(self
                    .realm
                    .get_property(target, WASM_MEM_BUFFER)
                    .unwrap_or(NanBox::undefined()));
            }
            // `WebAssembly.Memory.prototype.grow(delta)` → old page count. The
            // SAME `ArrayBuffer` object is kept; its canonical `Cell::Bytes` store
            // is resized in place (zero-extended) and every typed-array/`DataView`
            // view over it is re-lengthened, so `Memory.buffer` is stable across
            // grow and the store stays shared with wasm (A5, #11).
            if id == N_WASM_MEM_GROW {
                let delta = args
                    .first()
                    .map_or(0.0, |v| self.realm.to_number(*v))
                    .max(0.0) as usize;
                let old_pages = self
                    .realm
                    .get_property(target, WASM_MEM_PAGES)
                    .and_then(|v| v.as_number())
                    .unwrap_or(0.0) as usize;
                let new_pages = old_pages + delta;
                if let Some(max) = self
                    .realm
                    .get_property(target, WASM_MEM_MAX)
                    .and_then(|v| v.as_number())
                    && new_pages as f64 > max
                {
                    let m = self.new_str("memory.grow exceeds the declared maximum");
                    return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                }
                if let Some(bytes_h) = self
                    .realm
                    .get_property(target, WASM_MEM_BUFFER)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                    .and_then(|ob| self.array_buffer_bytes(ob))
                {
                    // `resize_buffer` zero-extends the store and re-lengthens views.
                    self.realm.resize_buffer(bytes_h, new_pages * WASM_PAGE);
                }
                self.realm.set_hidden_property(
                    target,
                    WASM_MEM_PAGES,
                    NanBox::number(new_pages as f64),
                );
                return Ok(NanBox::number(old_pages as f64));
            }
            // `WebAssembly.Table` `.length` getter and `get`/`set`/`grow` methods
            // (bound to the table `target`), over its function-ref element array.
            if matches!(
                id,
                N_WASM_TABLE_LEN | N_WASM_TABLE_GET | N_WASM_TABLE_SET | N_WASM_TABLE_GROW
            ) {
                let Some(elems) = self
                    .realm
                    .get_property(target, WASM_TABLE_ELEMS)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                else {
                    return Ok(NanBox::undefined());
                };
                let len = self.realm.array_length(elems).unwrap_or(0);
                let idx = self
                    .realm
                    .to_number(args.first().copied().unwrap_or(NanBox::undefined()));
                match id {
                    N_WASM_TABLE_LEN => return Ok(NanBox::number(len as f64)),
                    N_WASM_TABLE_GET | N_WASM_TABLE_SET => {
                        if idx < 0.0 || idx as usize >= len {
                            let m = self.new_str("WebAssembly.Table index out of bounds");
                            return Err(ExecError::Throw(
                                self.make_error(N_ERROR_BASE + 2, Some(m)),
                            ));
                        }
                        let i = idx as usize;
                        if id == N_WASM_TABLE_GET {
                            return Ok(self.realm.get_element(elems, i));
                        }
                        let v = args.get(1).copied().unwrap_or(NanBox::null());
                        self.realm.set_element(elems, i, v);
                        return Ok(NanBox::undefined());
                    }
                    _ => {
                        // grow(delta, init?) → prior length.
                        let new_len = len + idx.max(0.0) as usize;
                        if let Some(max) = self
                            .realm
                            .get_property(target, WASM_TABLE_MAX)
                            .and_then(|v| v.as_number())
                            && new_len as f64 > max
                        {
                            let m =
                                self.new_str("WebAssembly.Table.grow exceeds the declared maximum");
                            return Err(ExecError::Throw(
                                self.make_error(N_ERROR_BASE + 2, Some(m)),
                            ));
                        }
                        let init = args.get(1).copied().unwrap_or(NanBox::null());
                        for i in len..new_len {
                            self.realm.set_element(elems, i, init);
                        }
                        return Ok(NanBox::number(len as f64));
                    }
                }
            }
            let arg0 = args.first().copied().unwrap_or(NanBox::undefined());
            match id {
                N_RESOLVE => self.resolve_with(target, arg0),
                N_REJECT => self.settle(target, arg0, false),
                // The `revoke` function from `Proxy.revocable`.
                N_PROXY_REVOKE => self.realm.revoke_proxy(target),
                _ => {}
            }
            return Ok(NanBox::undefined());
        }
        let Some((func_id, captured)) = self.realm.function_at(handle) else {
            // A handle that is not any kind of callable (an ordinary object, an
            // array, …): calling it is a catchable JS `TypeError`.
            let m = self.new_str("is not a function");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        let def = self.functions[func_id as usize];
        // An object-literal concise method carries its `[[HomeObject]]`; bind it for
        // the duration of the call so `super.x` in the body resolves through it. An
        // arrow has no own home object — it inherits the enclosing one (so an arrow
        // inside a concise method can use `super`), matching its lexical `this`.
        let saved_home_obj = if def.is_arrow {
            self.current_home_object
        } else {
            let home_obj = self
                .realm
                .get_property(handle, HOME_OBJECT)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
            core::mem::replace(&mut self.current_home_object, home_obj)
        };
        let r = self.invoke(def, captured, this_val, args);
        self.current_home_object = saved_home_obj;
        r
    }

    /// Runs a function body with `this` and the parameters bound in a fresh
    /// child of `captured`.
    /// Invokes a function, guarding against unbounded recursion: beyond
    /// `MAX_CALL_DEPTH` nested calls it throws a `RangeError` instead of letting
    /// the host stack overflow.
    pub(crate) fn invoke(
        &mut self,
        def: FnDef<'a>,
        captured: Scope,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        if self.call_depth >= self.realm.limits.max_call_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            // A proper `RangeError` object (id 2 in `ERROR_NAMES`) so `instanceof
            // RangeError`/`Error` and `.name` work on the caught value.
            let err = self.make_error(N_ERROR_BASE + 2, Some(msg));
            return Err(ExecError::Throw(err));
        }
        self.call_depth += 1;
        let r = self.invoke_inner(def, captured, this_val, args);
        self.call_depth -= 1;
        r
    }

    pub(crate) fn invoke_inner(
        &mut self,
        def: FnDef<'a>,
        captured: Scope,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let call_scope = captured.child();
        let saved = core::mem::replace(&mut self.current, call_scope);
        // An arrow has no own `this` — it inherits the enclosing one lexically,
        // so leave `self.this_val` unchanged.
        let saved_this = if def.is_arrow {
            self.this_val
        } else {
            // Sloppy-mode `this` coercion: an `undefined`/`null` receiver becomes
            // the global object. Strict functions keep it as-is.
            let bound = if !def.is_strict
                && matches!(this_val.unpack(), Unpacked::Undefined | Unpacked::Null)
            {
                self.global_this
            } else {
                this_val
            };
            core::mem::replace(&mut self.this_val, bound)
        };
        // An arrow has no own home: like `this`, it inherits the enclosing
        // method's `super` binding (home class/static and object-literal home),
        // so `() => super.m()` inside a method works. A non-arrow establishes its
        // own home from its `FnDef`.
        let (saved_home, saved_home_static) = if def.is_arrow {
            (self.current_home, self.current_home_static)
        } else {
            (
                core::mem::replace(&mut self.current_home, def.home_class),
                core::mem::replace(&mut self.current_home_static, def.home_static),
            )
        };
        // A non-arrow invocation establishes its own `new.target`: the constructor
        // when reached via `new` (passed through the one-shot `pending_new_target`),
        // else `undefined`. An arrow inherits the enclosing `new.target`.
        let saved_target = if def.is_arrow {
            self.new_target
        } else {
            let nt = self
                .pending_new_target
                .take()
                .unwrap_or(NanBox::undefined());
            core::mem::replace(&mut self.new_target, nt)
        };
        // A generator body runs eagerly into a fresh yield buffer.
        let saved_sink = if def.is_generator {
            Some(self.gen_sink.replace(Vec::new()))
        } else {
            None
        };
        // C2: the tree-walk depth counter measures native recursion *within* one
        // function frame; reset it for the callee's body (deep function-call
        // recursion is bounded separately by `call_depth`) so genuine recursion is
        // not penalised by the depth accumulated in the caller's expressions.
        let saved_eval_depth = core::mem::replace(&mut self.eval_depth, 0);
        // Strict mode is lexical: a strict function (a class member, or one with a
        // `"use strict"` prologue, or defined in strict code) runs its whole body —
        // including parameter-default evaluation — in strict mode. An arrow
        // inherits the enclosing mode (already reflected in its `is_strict`).
        let saved_strict = self.strict;
        if def.is_strict {
            self.strict = true;
        }
        let result = (|| {
            // A non-arrow function gets an `arguments` array-like of its call
            // arguments. (Arrows inherit the enclosing `arguments`.) Bound *before*
            // the parameters so a parameter default can reference `arguments`
            // (`function f(x = arguments[0]) {}`).
            if !def.is_arrow {
                let arr = self.realm.new_array(args.to_vec());
                self.current
                    .declare("arguments", NanBox::handle(arr.to_raw()));
            }
            for (i, param) in def.params.iter().enumerate() {
                let value = if param.rest {
                    let rest = args[i.min(args.len())..].to_vec();
                    NanBox::handle(self.realm.new_array(rest).to_raw())
                } else {
                    let mut v = args.get(i).copied().unwrap_or(NanBox::undefined());
                    if matches!(v.unpack(), Unpacked::Undefined)
                        && let Some(d) = &param.default
                    {
                        v = self.eval(d)?;
                        self.infer_binding_name(&param.target, d, v);
                    }
                    v
                };
                self.bind_pattern(&param.target, value)?;
            }
            self.run_body(def.body)
        })();
        self.current = saved;
        self.this_val = saved_this;
        self.current_home = saved_home;
        self.current_home_static = saved_home_static;
        self.new_target = saved_target;
        self.eval_depth = saved_eval_depth;
        self.strict = saved_strict;
        // A generator call returns an iterator over the values it yielded.
        if def.is_generator {
            let collected = self.gen_sink.take().unwrap_or_default();
            self.gen_sink = saved_sink.flatten();
            let ret = result?; // a throw during collection propagates at call time
            return Ok(self.make_generator_with_return(collected, ret));
        }
        // An `async` function returns a promise of its result (rejected on throw).
        if def.is_async {
            let promise = self.fresh_promise();
            match result {
                Ok(v) => self.resolve_with(promise, v),
                Err(ExecError::Throw(e)) => self.settle(promise, e, false),
                Err(other) => return Err(other),
            }
            return Ok(NanBox::handle(promise.to_raw()));
        }
        result
    }

    /// `new Callee(args)` — supports the built-in `Map`/`Set` constructors
    /// (optionally seeded from an iterable argument).
    pub(crate) fn construct(
        &mut self,
        callee: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let Some(raw) = callee.as_handle() else {
            // `new` on a primitive is a catchable JS `TypeError`.
            let m = self.new_str("is not a constructor");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        let handle = Handle::from_raw(raw);
        // `new someProxy(...)`: route through the `construct` trap, or construct
        // the target.
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            let trap = self
                .realm
                .get_property(handler, "construct")
                .unwrap_or(NanBox::undefined());
            // A present but non-callable `construct` trap is a TypeError.
            if !matches!(trap.unpack(), Unpacked::Undefined | Unpacked::Null) {
                if !trap
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("proxy construct trap is not a function"));
                }
                let arr = self.realm.new_array(args.to_vec());
                let target_box = NanBox::handle(target.to_raw());
                let handler_box = NanBox::handle(handler.to_raw());
                // newTarget is an explicit `Reflect.construct` newTarget if present,
                // else the proxy itself.
                let new_target = self.reflect_new_target.take().unwrap_or(callee);
                let result = self.call_with_this(
                    trap,
                    handler_box,
                    &[target_box, NanBox::handle(arr.to_raw()), new_target],
                )?;
                // The `construct` trap must return an Object (ECMA-262 step 9).
                if !self.is_object_value(result) {
                    return Err(self.type_error("proxy [[Construct]] must return an object"));
                }
                return Ok(result);
            }
            return self.construct(NanBox::handle(target.to_raw()), args);
        }
        // `new boundFn(...)`: construct the bound target with the bound arguments
        // prepended (the bound `this` is ignored when constructing).
        if let Some(target) = self.realm.get_property(handle, BOUND_TARGET) {
            let mut all = Vec::new();
            if let Some(ba) = self.realm.get_property(handle, BOUND_ARGS)
                && let Some(bh) = ba.as_handle().map(Handle::from_raw)
                && let Some(elems) = self.realm.array_elements(bh)
            {
                all.extend_from_slice(elems);
            }
            all.extend_from_slice(args);
            return self.construct(target, &all);
        }
        // `new UserClass(...)`.
        if let Some((class_id, env)) = self.realm.class_at(handle) {
            // `new.target` inside the class constructor is the class itself.
            self.pending_new_target = Some(self.reflect_new_target.take().unwrap_or(callee));
            let inst = self.instantiate(class_id, &env, args)?;
            // `instance.constructor === TheClass` (non-enumerable back-reference).
            if let Some(ih) = inst.as_handle().map(Handle::from_raw) {
                self.realm.set_hidden_property(ih, "constructor", callee);
            }
            return Ok(inst);
        }
        // `new constructorFunction(...)`: bind a fresh object as `this`, run the
        // body, and return it — unless the function explicitly returned an object
        // (the spec's constructor return rule).
        if let Some((func_id, _)) = self.realm.function_at(handle) {
            // Arrow, generator, and async functions are not constructors.
            let def = self.functions[func_id as usize];
            if def.is_arrow || def.is_generator || def.is_async {
                let m = self.new_str("is not a constructor");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // The instance's `[[Prototype]]` is the *newTarget*'s `.prototype`
            // (the callee's, except under `Reflect.construct(target, args, newTarget)`
            // with a function newTarget), so inherited methods resolve correctly.
            let proto = match self.reflect_new_target {
                Some(nt)
                    if nt
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.function_at(h))
                        .is_some() =>
                {
                    let nt_fid = self
                        .realm
                        .function_at(Handle::from_raw(nt.as_handle().unwrap()))
                        .unwrap()
                        .0;
                    self.realm.function_prototype(nt_fid)
                }
                _ => self.realm.function_prototype(func_id),
            };
            let instance = self.realm.new_object_with_proto(Some(proto));
            let this = NanBox::handle(instance.to_raw());
            // Record the constructor for `instanceof` (hidden, GC-traced slot).
            self.realm.set_hidden_property(instance, CTOR_KEY, callee);
            // `new.target` inside the constructor body is the constructor itself.
            self.pending_new_target = Some(self.reflect_new_target.take().unwrap_or(callee));
            let ret = self.call_with_this(callee, this, args)?;
            // The constructor return rule: if the body returns an Object, that
            // object is the result; otherwise the freshly-bound `this`. The object
            // forms recognized are plain objects, arrays, and exotic
            // slot-bearing objects (typed arrays, DataViews, ArrayBuffers, Maps,
            // …) — so a constructor (or `Symbol.species`) that hands back a typed
            // array is honored. (A returned *function* keeps the legacy lenient
            // `this` result to preserve the curated `new.target` gate.)
            if let Some(rh) = ret.as_handle().map(Handle::from_raw)
                && self.is_object_value(ret)
                && !self.is_callable(rh)
            {
                return Ok(ret);
            }
            return Ok(this);
        }
        // `new Object(value)` — Object is a namespace object, matched by identity.
        // With no/`null`/`undefined` argument it makes a fresh object; otherwise it
        // is ToObject(value) (the same as calling `Object(value)`).
        if self.current.get("Object").and_then(|v| v.as_handle()) == callee.as_handle() {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            return Ok(self.coerce_to_object(v));
        }
        // `new Array(...)` — Array is a namespace object, matched by identity.
        // A single number argument is the length; otherwise the elements.
        if self.current.get("Array").and_then(|v| v.as_handle()) == callee.as_handle() {
            let elems = if args.len() == 1
                && let Some(n) = args[0].as_number()
            {
                // A single number is the length: a non-negative integer fitting
                // uint32 (capped here to avoid OOM in this dense model). Otherwise a
                // `RangeError`.
                if n < 0.0
                    || n > f64::from(u32::MAX)
                    || n > 100_000_000.0
                    || n != f64::from(n as u32)
                {
                    let m = self.new_str("Invalid array length");
                    return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                }
                alloc::vec![NanBox::undefined(); n as usize]
            } else {
                args.to_vec()
            };
            return Ok(NanBox::handle(self.realm.new_array(elems).to_raw()));
        }
        // `new Object(value)` — the `Object` namespace object is a constructor.
        // With no `newTarget` subclassing, it behaves like calling `Object(value)`:
        // an object value is returned as-is, a primitive is wrapped (ToObject), and
        // null/undefined/none yield a fresh ordinary object.
        if self.current.get("Object").and_then(|v| v.as_handle()) == callee.as_handle() {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            return Ok(self.coerce_to_object(v));
        }
        let Some(id) = self.realm.native_at(handle) else {
            // A callable that is not a constructor — e.g. a built-in method such as
            // `Function.prototype.apply`/`call` (a first-class bound native), or any
            // value reaching here without a `[[Construct]]`. `new` on it is a
            // TypeError (catchable), not an internal "unsupported".
            let m = self.new_str("is not a constructor");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        // The abstract `%TypedArray%` intrinsic cannot be constructed directly.
        if id == N_TYPED_ARRAY_ABSTRACT {
            let m = self.new_str("Abstract class TypedArray not directly constructable");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // `new WebAssembly.Module(bytes)` — decode/validate, keep the bytes so a
        // later `new WebAssembly.Instance(module)` can instantiate it.
        if id == N_WASM_MODULE {
            return self.make_wasm_module(args.first().copied().unwrap_or(NanBox::undefined()));
        }
        // `new WebAssembly.Instance(module, importObject?)` → `{ exports: {…} }`.
        if id == N_WASM_INSTANCE {
            let module = args
                .first()
                .copied()
                .and_then(|m| m.as_handle())
                .map(Handle::from_raw)
                .filter(|m| self.realm.get_property(*m, WASM_IS_MODULE).is_some())
                .ok_or_else(|| {
                    self.wasm_type_error(
                        "WebAssembly.Instance argument must be a WebAssembly.Module",
                    )
                })?;
            let bytes_arr = self
                .realm
                .get_property(module, WASM_BYTES)
                .unwrap_or(NanBox::undefined());
            let imports = args.get(1).copied().unwrap_or(NanBox::undefined());
            let instance = self.build_wasm_instance(bytes_arr, imports)?;
            return Ok(instance);
        }
        // `new WebAssembly.Global({ value: "i32"|…, mutable }, init)` — a typed
        // value cell exposing a `.value` accessor (settable only if mutable).
        if id == N_WASM_GLOBAL {
            let desc = args.first().copied().unwrap_or(NanBox::undefined());
            let dh = desc.as_handle().map(Handle::from_raw);
            let ty = dh
                .and_then(|h| self.realm.get_property(h, "value"))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("i32"));
            let mutable = dh
                .and_then(|h| self.realm.get_property(h, "mutable"))
                .is_some_and(|v| self.realm.truthy(v));
            let init = args.get(1).copied().unwrap_or(NanBox::undefined());
            let value = self.wasm_coerce_global(&ty, init);
            return Ok(self.make_wasm_global(value, &ty, mutable));
        }
        // `new Proxy(target, handler)`.
        if id == N_PROXY {
            let target = args.first().copied().unwrap_or(NanBox::undefined());
            let h = args.get(1).copied().unwrap_or(NanBox::undefined());
            // Both the target and the handler must be Objects (a string/symbol/
            // bigint primitive or an immediate is a TypeError).
            let (Some(tr), Some(hr)) = (
                target.as_handle().filter(|_| self.is_object_value(target)),
                h.as_handle().filter(|_| self.is_object_value(h)),
            ) else {
                let msg = self.new_str("Cannot create proxy with a non-object target or handler");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(msg))));
            };
            let p = self
                .realm
                .new_proxy(Handle::from_raw(tr), Handle::from_raw(hr));
            return Ok(NanBox::handle(p.to_raw()));
        }
        // `new Intl.NumberFormat(locales, options)` / `Intl.DateTimeFormat(...)`.
        if id == N_INTL_NUMBER_FORMAT || id == N_INTL_DATETIME_FORMAT {
            return Ok(self.make_intl_formatter(id, args));
        }
        // `new Intl.Collator(...)` → an object whose `compare` is a bound function
        // (so `arr.sort(new Intl.Collator().compare)` works); code-point order, no
        // locale tailoring (matching `localeCompare`).
        if id == N_INTL_COLLATOR {
            return Ok(self.make_collator(args));
        }
        // `new Intl.PluralRules(...)` → an object with a `select(n)` method.
        if id == N_INTL_PLURAL_RULES {
            return Ok(self.make_plural_rules(args));
        }
        // `new Intl.ListFormat(locale, { type, style })` → an object with a `format(list)`.
        if id == N_INTL_LIST_FORMAT {
            return Ok(self.make_list_format(args));
        }
        // `new Intl.RelativeTimeFormat(locale, { numeric, style })` → an object with `format`.
        if id == N_INTL_REL_TIME {
            return Ok(self.make_relative_time_format(args));
        }
        // `new Intl.DisplayNames(locale, { type })` → an object with an `of(code)` method.
        if id == N_INTL_DISPLAY_NAMES {
            return Ok(self.make_display_names(args));
        }
        // `new Intl.Segmenter(locale, { granularity })` → an object with a `segment(s)` method.
        if id == N_INTL_SEGMENTER {
            return Ok(self.make_segmenter(args));
        }
        // `new Promise(executor)`: run executor(resolve, reject).
        if id == N_PROMISE {
            let promise = self.fresh_promise();
            let resolve = self.realm.new_bound_native(N_RESOLVE, promise);
            let reject = self.realm.new_bound_native(N_REJECT, promise);
            let executor = args.first().copied().unwrap_or(NanBox::undefined());
            let r = self.call(
                executor,
                &[
                    NanBox::handle(resolve.to_raw()),
                    NanBox::handle(reject.to_raw()),
                ],
            );
            if let Err(ExecError::Throw(e)) = r {
                self.settle(promise, e, false);
            } else {
                r?;
            }
            return Ok(NanBox::handle(promise.to_raw()));
        }
        // `new Date(ms)` (or `new Date()` for "now").
        if id == N_DATE {
            let ms = if args.len() >= 2 {
                // `new Date(year, month, day?, h?, m?, s?, ms?)` (local ≈ UTC here).
                // Every supplied argument is ToNumber'd in order (Symbol → TypeError,
                // a throwing `valueOf` propagates).
                let mut nums = Vec::with_capacity(args.len());
                for a in args {
                    let v = self.coerce_to_number(*a)?;
                    nums.push(self.realm.to_number(v));
                }
                let getn = |i: usize, dflt: f64| nums.get(i).copied().unwrap_or(dflt);
                let year_n = getn(0, 1970.0);
                let month = getn(1, 0.0);
                let day = getn(2, 1.0);
                let hours = getn(3, 0.0);
                let mins = getn(4, 0.0);
                let secs = getn(5, 0.0);
                let millis = getn(6, 0.0);
                // Any NaN component yields an invalid date.
                if [year_n, month, day, hours, mins, secs, millis]
                    .iter()
                    .any(|v| v.is_nan() || !v.is_finite())
                {
                    f64::NAN
                } else {
                    // A two-digit year (0..=99) maps to 1900+year.
                    let yi = year_n as i64;
                    let year = if (0..=99).contains(&yi) {
                        1900 + yi
                    } else {
                        yi
                    };
                    let total_months = year * 12 + month as i64;
                    let y = total_months.div_euclid(12);
                    let mo = total_months.rem_euclid(12) as u32 + 1; // 1..=12
                    // Measure the day as an offset from the 1st so an out-of-range
                    // (incl. negative) day rolls over via integer arithmetic.
                    let days = crate::realm::days_from_civil(y, mo, 1) + (day as i64 - 1);
                    time_clip(
                        (days * 86_400_000
                            + hours as i64 * 3_600_000
                            + mins as i64 * 60_000
                            + secs as i64 * 1_000
                            + millis as i64) as f64,
                    )
                }
            } else {
                match args.first().copied() {
                    Some(a) => {
                        // A Date argument copies its time value directly.
                        if let Some(existing) = a
                            .as_handle()
                            .map(Handle::from_raw)
                            .and_then(|h| self.realm.date_at(h))
                        {
                            time_clip(existing)
                        } else {
                            // ToPrimitive(value): a string is parsed as a date,
                            // anything else is ToNumber'd → TimeClip.
                            let prim = self.coerce_primitive(a, "default")?;
                            if let Some(h) = prim.as_handle().map(Handle::from_raw)
                                && let Some(s) = self.realm.string_value(h)
                            {
                                crate::realm::parse_date_string(&s).map_or(f64::NAN, time_clip)
                            } else {
                                let v = self.coerce_to_number(prim)?;
                                time_clip(self.realm.to_number(v))
                            }
                        }
                    }
                    None => now_ms(),
                }
            };
            let d = self.realm.new_date(ms);
            // Register the instance's `[[Prototype]]` as `Date.prototype` so
            // `Object.getPrototypeOf(date)`, `instanceof`, and
            // `Date.prototype.isPrototypeOf(date)` resolve correctly.
            if let Some(proto) = self
                .current
                .get("Date")
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                .and_then(|c| self.realm.get_property(c, "prototype"))
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw)
            {
                self.realm.set_native_proto(d, proto);
            }
            return Ok(NanBox::handle(d.to_raw()));
        }
        // `new RegExp(pattern, flags)` / `RegExp(pattern, flags)`.
        if id == N_REGEXP {
            let pattern = args.first().copied().unwrap_or(NanBox::undefined());
            let flags_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
            // If `pattern` is a RegExp instance, copy its source and (absent an
            // explicit `flags` argument) its flags too — `new RegExp(/x/i)` clones
            // `/x/i`, while `new RegExp(/x/i, "g")` keeps the source but uses "g".
            let pat_h = pattern.as_handle().map(Handle::from_raw);
            let (pat, flags) = if let Some((src, fl)) = pat_h.and_then(|h| self.realm.regexp_at(h))
            {
                let flags = if matches!(flags_arg.unpack(), Unpacked::Undefined) {
                    fl
                } else {
                    self.coerce_to_string(flags_arg)?
                };
                (src, flags)
            } else if let Some(ph) = pat_h.filter(|_| self.is_regexp_arg(pattern)) {
                // A non-RegExp object with a truthy `@@match` (IsRegExp): use its
                // `.source`/`.flags` when no flags argument is supplied.
                let src_v = self.read_member(ph, "source")?;
                let src = self.coerce_to_string(src_v)?;
                let flags = if matches!(flags_arg.unpack(), Unpacked::Undefined) {
                    let fv = self.read_member(ph, "flags")?;
                    self.coerce_to_string(fv)?
                } else {
                    self.coerce_to_string(flags_arg)?
                };
                (src, flags)
            } else {
                let pat = if matches!(pattern.unpack(), Unpacked::Undefined) {
                    String::new()
                } else {
                    self.coerce_to_string(pattern)?
                };
                let flags = if matches!(flags_arg.unpack(), Unpacked::Undefined) {
                    String::new()
                } else {
                    self.coerce_to_string(flags_arg)?
                };
                (pat, flags)
            };
            // Validate the pattern/flags up front: an invalid regular expression is
            // a `SyntaxError` at construction, not a silent broken object.
            #[cfg(feature = "regex")]
            if crate::regex::Regex::new(&pat, &flags).is_err() {
                let m = self.new_str(&alloc::format!(
                    "Invalid regular expression: /{pat}/{flags}"
                ));
                return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
            }
            let r = self.new_regexp_instance(&pat, &flags);
            return Ok(NanBox::handle(r.to_raw()));
        }
        // `new Error(message, { cause })` and friends → `{ name, message }` plus
        // the ES2022 `cause` option. `AggregateError(errors, message, { cause })`
        // takes its message second and exposes `.errors`.
        if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
            let is_aggregate = id == N_ERROR_BASE + 5;
            let (msg_arg, opts_arg) = if is_aggregate {
                (args.get(1).copied(), args.get(2))
            } else {
                (args.first().copied(), args.get(1))
            };
            let err = self.make_error(id, msg_arg);
            if is_aggregate && let Some(eh) = err.as_handle() {
                let errors = args.first().copied().unwrap_or(NanBox::undefined());
                let list = self.iterate_values(errors).unwrap_or_default();
                let arr = self.realm.new_array(list);
                self.realm.set_property(
                    Handle::from_raw(eh),
                    "errors",
                    NanBox::handle(arr.to_raw()),
                );
            }
            if let Some(opts) = opts_arg.and_then(|v| v.as_handle()).map(Handle::from_raw)
                && let Some(cause) = self.realm.get_property(opts, "cause")
                && let Some(eh) = err.as_handle()
            {
                self.realm
                    .set_property(Handle::from_raw(eh), "cause", cause);
                self.realm.mark_hidden(Handle::from_raw(eh), "cause");
            }
            return Ok(err);
        }
        // `new WeakRef(target)` — holds the target. `deref()` always returns it
        // (sound because GC is never driven mid-execution).
        if id == N_WEAKREF {
            let target = args.first().copied().unwrap_or(NanBox::undefined());
            let obj = self.realm.new_object();
            self.realm.set_hidden_property(obj, WEAKREF_TARGET, target);
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new FinalizationRegistry(cb)` — bounded: with no mid-execution GC the
        // cleanup callback never fires, so `register`/`unregister` are inert.
        if id == N_FINALIZATION_REGISTRY {
            let obj = self.realm.new_object();
            self.realm
                .set_hidden_property(obj, FINREG_TAG, NanBox::boolean(true));
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new ArrayBuffer(n)` — a zeroed byte store of length `n`.
        if id == N_ARRAY_BUFFER {
            let raw = args.first().map_or(0.0, |v| self.realm.to_number(*v));
            let n = self.validate_alloc_len(raw, "Invalid ArrayBuffer length")?;
            let buf = self.make_array_buffer(n);
            // `new ArrayBuffer(n, { maxByteLength })` makes the buffer resizable up to `max`.
            if let Some(opts) = args
                .get(1)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                && let Some(maxv) = self.realm.get_property(opts, "maxByteLength")
            {
                let max = self.realm.to_number(maxv).max(0.0) as usize;
                if max < n {
                    let m = self.new_str("ArrayBuffer maxByteLength is smaller than its length");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                self.realm.set_hidden_property(
                    buf,
                    ARRAY_BUFFER_MAXLEN,
                    NanBox::number(max as f64),
                );
            }
            return Ok(NanBox::handle(buf.to_raw()));
        }
        // `new WebAssembly.Memory({ initial, maximum? })` — linear memory backed by
        // an `ArrayBuffer` of `initial` 64 KiB pages, exposing `.buffer` + `grow()`.
        if id == N_WASM_MEMORY {
            let dh = args
                .first()
                .copied()
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
            let initial_raw = dh
                .and_then(|h| self.realm.get_property(h, "initial"))
                .map_or(0.0, |v| self.realm.to_number(v));
            let maximum = dh
                .and_then(|h| self.realm.get_property(h, "maximum"))
                .map(|v| self.realm.to_number(v).max(0.0) as usize);
            // Validate the *byte* size (initial pages × page size) before allocating.
            let byte_len =
                self.validate_alloc_len(initial_raw * WASM_PAGE as f64, "Invalid memory size")?;
            let initial = (initial_raw.max(0.0)) as usize;
            let buf = self.make_array_buffer(byte_len);
            let mem = self.make_wasm_memory_object(buf, initial, maximum);
            return Ok(NanBox::handle(mem.to_raw()));
        }
        // `new WebAssembly.Table({ element, initial, maximum? }, init?)` — a fixed
        // table of function references, exposing `.length` + `get`/`set`/`grow`.
        if id == N_WASM_TABLE {
            let dh = args
                .first()
                .copied()
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
            let initial_raw = dh
                .and_then(|h| self.realm.get_property(h, "initial"))
                .map_or(0.0, |v| self.realm.to_number(v));
            let initial = self.validate_alloc_len(initial_raw, "Invalid table length")?;
            let maximum = dh
                .and_then(|h| self.realm.get_property(h, "maximum"))
                .map(|v| self.realm.to_number(v).max(0.0) as usize);
            // Slots start at the init value (a function) or null.
            let init = args.get(1).copied().unwrap_or(NanBox::null());
            let elems = self.realm.new_array(alloc::vec![init; initial]);
            let table = self.realm.new_object();
            self.realm
                .set_hidden_property(table, WASM_TABLE_ELEMS, NanBox::handle(elems.to_raw()));
            self.realm.set_hidden_property(
                table,
                WASM_TABLE_MAX,
                maximum.map_or(NanBox::undefined(), |m| NanBox::number(m as f64)),
            );
            let len_get = self.realm.new_bound_native(N_WASM_TABLE_LEN, table);
            self.realm.define_accessor(
                table,
                "length",
                NanBox::handle(len_get.to_raw()),
                NanBox::undefined(),
            );
            for (name, nid) in [
                ("get", N_WASM_TABLE_GET),
                ("set", N_WASM_TABLE_SET),
                ("grow", N_WASM_TABLE_GROW),
            ] {
                let f = self.realm.new_bound_native(nid, table);
                self.realm
                    .set_property(table, name, NanBox::handle(f.to_raw()));
            }
            return Ok(NanBox::handle(table.to_raw()));
        }
        // `new DataView(buffer, byteOffset?)` — a view onto an ArrayBuffer.
        if id == N_DATA_VIEW {
            // The instance's `[[Prototype]]` is `DataView.prototype`, so inherited
            // members (the get*/set* methods, the accessors, `Symbol.toStringTag`)
            // resolve through the chain.
            let dv_proto = self
                .current
                .get("DataView")
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                .and_then(|c| self.realm.get_property(c, "prototype"))
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw);
            let obj = self.realm.new_object_with_proto(dv_proto);
            let buf = args.first().copied().unwrap_or(NanBox::undefined());
            let mut buf_len = 0usize;
            if let Some(bh) = buf.as_handle().map(Handle::from_raw) {
                self.guard_detached_buffer(bh)?;
                buf_len = self
                    .array_buffer_bytes(bh)
                    .and_then(|h| self.realm.bytes_len(h))
                    .unwrap_or(0);
            }
            let off = args.get(1).map_or(0.0, |v| self.realm.to_number(*v));
            // M1: a negative/non-integer offset, or one past the buffer, is a
            // RangeError — never trusted blindly into the access path.
            if !off.is_finite() || off < 0.0 || (off as usize as f64) != off {
                let m = self.new_str("Invalid DataView offset");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            let byte_off = off as usize;
            if byte_off > buf_len {
                let m = self.new_str("Start offset is outside the bounds of the buffer");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            self.realm.set_hidden_property(obj, DATA_VIEW_BUF, buf);
            self.realm
                .set_hidden_property(obj, DATA_VIEW_OFF, NanBox::number(off));
            // An explicit byteLength (3rd arg) is honored; otherwise the view spans
            // the rest of the buffer from `byteOffset`.
            if let Some(len) = args.get(2)
                && !matches!(len.unpack(), Unpacked::Undefined)
            {
                let raw = self.realm.to_number(*len);
                // M1: validate `byteOffset + byteLength <= buffer.byteLength` with
                // checked arithmetic (a saturated length must not wrap past the end).
                let view_len = self.validate_alloc_len(raw, "Invalid DataView length")?;
                let fits = byte_off
                    .checked_add(view_len)
                    .is_some_and(|end| end <= buf_len);
                if !fits {
                    let m = self.new_str("Invalid DataView length: exceeds buffer bounds");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                self.realm
                    .set_hidden_property(obj, DATA_VIEW_LEN, NanBox::number(view_len as f64));
            }
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new Int8Array(n)` / `new Uint8Array([…])` / `new T(buffer, off?, len?)` — a
        // typed array is a `Cell::TypedArray` *view* over a contiguous `ArrayBuffer`
        // byte store. Element reads/writes go straight through the shared bytes, so
        // sibling views and `DataView`s alias the same storage intrinsically.
        if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id) {
            let kind = (id - N_TYPED_ARRAY_BASE) as u8;
            let elem_size = TYPED_ARRAY_KINDS[kind as usize].1 as usize;
            // `new T(buffer, byteOffset?, length?)` — a view over an existing ArrayBuffer.
            if let Some(v) = args.first()
                && let Some(bh) = v.as_handle().map(Handle::from_raw)
                && self.realm.get_property(bh, ARRAY_BUFFER_BYTES).is_some()
            {
                self.guard_detached_buffer(bh)?;
                let bytes_h = self.array_buffer_bytes(bh).unwrap();
                let total = self.realm.bytes_len(bytes_h).unwrap_or(0);
                // H2/T1: validate the byteOffset. It must be a non-negative integer
                // that is a multiple of the element size, else a RangeError.
                let byte_off = match args
                    .get(1)
                    .filter(|a| !matches!(a.unpack(), Unpacked::Undefined))
                {
                    Some(a) => {
                        let raw = self.realm.to_number(*a);
                        if !raw.is_finite() || raw < 0.0 || (raw as usize as f64) != raw {
                            let m = self.new_str("Invalid typed array offset");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        let off = raw as usize;
                        if !off.is_multiple_of(elem_size) {
                            let m =
                                self.new_str("start offset must be a multiple of the element size");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        off
                    }
                    None => 0,
                };
                if byte_off > total {
                    let m = self.new_str("Start offset is outside the bounds of the buffer");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                let avail = total - byte_off;
                let length = match args
                    .get(2)
                    .filter(|a| !matches!(a.unpack(), Unpacked::Undefined))
                {
                    Some(a) => {
                        let raw = self.realm.to_number(*a);
                        let len = self.validate_alloc_len(raw, "Invalid typed array length")?;
                        // H2/T1: `byteOffset + length*elem_size` must fit the buffer.
                        // Checked arithmetic — a saturated length must not wrap.
                        let fits = len
                            .checked_mul(elem_size)
                            .and_then(|need| byte_off.checked_add(need))
                            .is_some_and(|end| end <= total);
                        if !fits {
                            let m =
                                self.new_str("Invalid typed array length: exceeds buffer bounds");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        len
                    }
                    None => {
                        // No explicit length: the view spans the rest of the buffer,
                        // which must divide evenly into elements.
                        if !avail.is_multiple_of(elem_size) {
                            let m = self.new_str(
                                "buffer length minus the byteOffset is not a multiple of the element size",
                            );
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        avail / elem_size
                    }
                };
                let view = self
                    .realm
                    .new_typed_array(bytes_h, bh, byte_off, length, kind);
                self.link_typed_array_proto(view, kind, callee);
                return Ok(NanBox::handle(view.to_raw()));
            }
            // Otherwise allocate a fresh backing buffer and view it from offset 0.
            // `new T(arrayLike)` copies+coerces the source's elements into the buffer;
            // `new T(length)` / `new T()` zero-fill.
            let mut src: Option<Vec<NanBox>> = args
                .first()
                .copied()
                .and_then(|v| v.as_handle().map(Handle::from_raw))
                .and_then(|h| self.realm.elements_vec(h));
            // InitializeTypedArrayFromArrayLike: a *plain object* with a `length`
            // property and no `Symbol.iterator` (real arrays / typed arrays / buffers
            // were already handled above; the iterable path is handled elsewhere).
            // Read `length` (ToLength), then Get each index 0..length in order; the
            // per-element ToNumber / ToBigInt coercion happens on the write below.
            if src.is_none()
                && let Some(h) = args
                    .first()
                    .copied()
                    .and_then(|v| v.as_handle().map(Handle::from_raw))
                && self.realm.object_keys(h).is_some()
                && self.realm.string_value(h).is_none()
            {
                let iter_sym = self.well_known_symbol("iterator");
                let iter_key = self.member_key(iter_sym);
                let has_iter = self
                    .realm
                    .get_property(h, &iter_key)
                    .is_some_and(|f| !matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null));
                if has_iter {
                    // InitializeTypedArrayFromList: the source is iterable. Drain
                    // its iterator (calling `[Symbol.iterator]()` then `.next()`),
                    // then coerce each value to the element type.
                    let bigint = is_bigint_kind(kind);
                    let values = self.iterate_values(args[0])?;
                    let mut elems = Vec::with_capacity(values.len());
                    for v in values {
                        let v = if bigint { v } else { self.coerce_to_number(v)? };
                        elems.push(v);
                    }
                    src = Some(elems);
                } else {
                    let len_val = self.read_member(h, "length")?;
                    // ToLength: ToNumber (fallible — a Symbol length is a TypeError),
                    // then validate it as an allocatable length.
                    let len_num = self.coerce_to_number(len_val)?;
                    let raw = self.realm.to_number(len_num);
                    let len = self.validate_alloc_len(raw, "Invalid typed array length")?;
                    let bigint = is_bigint_kind(kind);
                    let mut elems = Vec::with_capacity(len);
                    for i in 0..len {
                        let v = self.read_member(h, &alloc::format!("{i}"))?;
                        // Coerce each element eagerly so a throwing `valueOf` / a Symbol
                        // / (for a BigInt array) a non-BigInt value surfaces here rather
                        // than being swallowed by the infallible bulk write below.
                        let v = if bigint { v } else { self.coerce_to_number(v)? };
                        elems.push(v);
                    }
                    src = Some(elems);
                }
            }
            let length = match (&src, args.first().copied()) {
                (Some(s), _) => s.len(),
                (None, Some(v)) => {
                    let raw = self.realm.to_number(v);
                    self.validate_alloc_len(raw, "Invalid typed array length")?
                }
                (None, None) => 0,
            };
            let buf = self.make_array_buffer(length * elem_size);
            let bytes_h = self.array_buffer_bytes(buf).unwrap();
            let view = self.realm.new_typed_array(bytes_h, buf, 0, length, kind);
            if let Some(s) = src {
                // For a BigInt typed array, ToBigInt every source element up front
                // (a Number element throws TypeError, per spec); other kinds write
                // the values straight through.
                let s = if is_bigint_kind(kind) {
                    let mut coerced = Vec::with_capacity(s.len());
                    for v in s {
                        coerced.push(self.coerce_typed_array_write(view, v)?);
                    }
                    coerced
                } else {
                    s
                };
                // Bulk write-through: one buffer borrow, no per-element heap lookup.
                self.realm.typed_set_from_numbers(view, 0, &s);
            }
            // Link the view's `[[Prototype]]` to the concrete constructor's
            // `.prototype` (the *newTarget*'s under `Reflect.construct` /
            // `TA.of`/`from` with a subclass), so `result.constructor`,
            // `Object.getPrototypeOf(result)`, and inherited members resolve.
            self.link_typed_array_proto(view, kind, callee);
            return Ok(NanBox::handle(view.to_raw()));
        }
        // `new Number(x)` / `new String(x)` / `new Boolean(x)`: a primitive
        // wrapper object boxing the coerced primitive (`valueOf` recovers it).
        if matches!(id, N_NUMBER | N_STRING | N_BOOLEAN) {
            let prim = match id {
                N_NUMBER => {
                    let n = args.first().map_or(0.0, |v| self.realm.to_number(*v));
                    NanBox::number(n)
                }
                N_STRING => {
                    let s = args
                        .first()
                        .map_or_else(String::new, |v| self.realm.to_display_string(*v));
                    self.new_str(&s)
                }
                _ => NanBox::boolean(
                    self.realm
                        .truthy(args.first().copied().unwrap_or(NanBox::undefined())),
                ),
            };
            return Ok(self.make_primitive_wrapper(prim, id));
        }
        // `new Function(...)` builds an anonymous function from runtime source —
        // identical to calling `Function(...)` as a plain function.
        if id == N_FUNCTION {
            return self.build_function_constructor(args);
        }
        // `WeakMap`/`WeakSet` reuse the collection cell (no true weak refs here).
        let is_set = match id {
            N_SET | N_WEAKSET => true,
            N_MAP | N_WEAKMAP => false,
            // `new Symbol()` / `new BigInt()` and other non-constructor natives throw a
            // catchable TypeError rather than aborting the engine.
            _ => {
                let m = self.new_str("is not a constructor");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        };
        let handle = self.realm.new_collection(is_set);
        // Link the instance to its constructor's `.prototype` so
        // `Object.getPrototypeOf(new Map) === Map.prototype`, `instanceof`, and
        // inherited `Symbol.toStringTag`/`constructor` lookups resolve.
        let ctor_name = match id {
            N_MAP => "Map",
            N_SET => "Set",
            N_WEAKMAP => "WeakMap",
            N_WEAKSET => "WeakSet",
            _ => "",
        };
        if let Some(proto) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            // A collection is a non-object cell: its `[[Prototype]]` link is kept
            // in the realm's native-proto table (read by `object_proto`).
            self.realm.set_native_proto(handle, proto);
        }
        // A weak collection rejects primitive keys (its keys must be objects/symbols).
        if matches!(id, N_WEAKMAP | N_WEAKSET) {
            self.realm.set_collection_weak(handle);
        }
        // Seed from an iterable: a `Set` from array elements, a `Map` from
        // `[key, value]` pairs.
        // Seed from any iterable (array, string, Set, Map, …): a `Set` from each
        // value, a `Map` from each `[key, value]` pair.
        let first = args.first().copied().unwrap_or(NanBox::undefined());
        if !matches!(first.unpack(), Unpacked::Undefined | Unpacked::Null) {
            for item in self.iterate_values(first)? {
                if is_set {
                    self.guard_weak_key(handle, item)?;
                    self.realm.collection_set(handle, item, item);
                } else if let Some(pr) = item
                    .as_handle()
                    .and_then(|r| self.realm.array_elements(Handle::from_raw(r)))
                    .map(<[_]>::to_vec)
                {
                    let k = pr.first().copied().unwrap_or(NanBox::undefined());
                    let v = pr.get(1).copied().unwrap_or(NanBox::undefined());
                    self.guard_weak_key(handle, k)?;
                    self.realm.collection_set(handle, k, v);
                }
            }
        }
        Ok(NanBox::handle(handle.to_raw()))
    }

    /// Builds an error object `{ name, message }` for the constructor `id`.
    /// Applies a native superclass constructor's effect to `instance` for
    /// `super(...)` in a class that `extends` a native (e.g. `extends Error`).
    pub(crate) fn apply_native_super(&mut self, native_id: u16, instance: Handle, args: &[NanBox]) {
        // Error family: set `message` and the default `name` (a `this.name = …`
        // after `super()` may override it).
        if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&native_id) {
            let name = ERROR_NAMES[(native_id - N_ERROR_BASE) as usize];
            let name_v = self.new_str(name);
            self.realm.set_property(instance, "name", name_v);
            let msg = match args.first() {
                Some(m) if !matches!(m.unpack(), Unpacked::Undefined) => {
                    let s = self.realm.to_display_string(*m);
                    self.new_str(&s)
                }
                _ => self.new_str(""),
            };
            self.realm.set_property(instance, "message", msg);
            // `name`/`message` are non-enumerable (out of `Object.keys`/JSON).
            self.realm.mark_hidden(instance, "name");
            self.realm.mark_hidden(instance, "message");
            // ES2022 `cause`: `new Error(msg, { cause })` installs a non-enumerable
            // `cause` when the options argument has such a property (even if undefined).
            if let Some(opts) = args.get(1)
                && let Some(raw) = opts.as_handle()
                && self.realm.has_own(Handle::from_raw(raw), "cause")
            {
                let cause = self
                    .realm
                    .get_property(Handle::from_raw(raw), "cause")
                    .unwrap_or(NanBox::undefined());
                self.realm.set_property(instance, "cause", cause);
                self.realm.mark_hidden(instance, "cause");
            }
        }
    }

    /// Sorts `elems` with a JS comparator (a negative result orders `a` before
    /// `b`); without one, by the elements' string forms. Insertion sort, so the
    /// comparator can call back into the interpreter.
    pub(crate) fn sort_array(
        &mut self,
        elems: Vec<NanBox>,
        cmp: NanBox,
        numeric: bool,
    ) -> Result<Vec<NanBox>, ExecError> {
        // `sort(comparefn)`: a non-undefined comparefn that is not callable is a
        // TypeError (per spec, observed before any element comparison).
        if !matches!(cmp.unpack(), Unpacked::Undefined) && !self.is_callable_value(cmp) {
            return Err(self.type_error("comparefn must be a function"));
        }
        let has_cmp = self.is_callable_value(cmp);
        // `undefined` elements always sort to the end and are never passed to the
        // comparator; only defined values are ordered against each other.
        let undefined_count = elems
            .iter()
            .filter(|e| matches!(e.unpack(), Unpacked::Undefined))
            .count();
        let mut elems: Vec<NanBox> = elems
            .into_iter()
            .filter(|e| !matches!(e.unpack(), Unpacked::Undefined))
            .collect();
        for i in 1..elems.len() {
            let mut j = i;
            while j > 0 {
                let order = if has_cmp {
                    let r = self.call(cmp, &[elems[j - 1], elems[j]])?;
                    self.realm.to_number(r)
                } else if numeric {
                    // A typed array's default comparison is numeric ascending (with
                    // `NaN` sorting to the end).
                    let a = self.realm.to_number(elems[j - 1]);
                    let b = self.realm.to_number(elems[j]);
                    if a < b || b.is_nan() {
                        -1.0
                    } else if a > b || a.is_nan() {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let a = self.realm.to_display_string(elems[j - 1]);
                    let b = self.realm.to_display_string(elems[j]);
                    if a < b {
                        -1.0
                    } else if a > b {
                        1.0
                    } else {
                        0.0
                    }
                };
                if order > 0.0 {
                    elems.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
        // Re-append the `undefined` holes after the ordered defined values.
        elems.extend(core::iter::repeat_n(NanBox::undefined(), undefined_count));
        Ok(elems)
    }
}
