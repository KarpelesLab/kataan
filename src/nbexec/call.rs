use super::*;

impl<'a> Interp<'a> {
    pub(crate) fn is_callable(&self, handle: Handle) -> bool {
        self.realm.native_at(handle).is_some()
            || self.realm.host_fn_at(handle).is_some()
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
        // A registered host function constructs iff it was `register_constructor`ed.
        if let Some(id) = self.realm.host_fn_at(handle) {
            return self.host_fn_is_constructor(id);
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
        // A dynamically-registered host function (`register_fn`, ROADMAP §4.0):
        // invoke its Rust closure through the host boundary.
        if let Some(id) = self.realm.host_fn_at(handle) {
            return self.call_host_fn(id, this_val, args);
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
                // `replaceAll`/`matchAll`: per spec, when `searchValue`/`regexp` is
                // a RegExp it must be global — `IsRegExp` then `Get(flags)` must
                // include "g", else a TypeError — and this is checked *before*
                // `ToString(this)` runs (so a poisoned `this.toString` is not
                // called). The string-receiver path validates it again redundantly.
                if (name == "replaceAll" || name == "matchAll")
                    && args.first().copied().is_some_and(|a| self.is_regexp_arg(a))
                    && let Some(argh) = args
                        .first()
                        .and_then(|a| a.as_handle())
                        .map(Handle::from_raw)
                {
                    let flags_v = self.read_member(argh, "flags")?;
                    if matches!(flags_v.unpack(), Unpacked::Undefined | Unpacked::Null) {
                        return Err(self.type_error(&alloc::format!(
                            "String.prototype.{name} called with a non-global RegExp argument"
                        )));
                    }
                    let flags_s = self.coerce_to_string(flags_v)?;
                    if !flags_s.contains('g') {
                        return Err(self.type_error(&alloc::format!(
                            "String.prototype.{name} called with a non-global RegExp argument"
                        )));
                    }
                }
                // match/matchAll/replace/replaceAll/search/split: if the
                // searchValue/separator is not undefined/null and defines the
                // matching well-known-symbol method, delegate to
                // `searchValue[@@method](O, …rest)` with the *original* `this`
                // (`O` = RequireObjectCoercible(this), NOT ToString'd) — so a String
                // wrapper receiver reaches the replacer as the object it is, and the
                // searchValue's own `toString` is never called.
                if let Some(sym_name) = match name.as_str() {
                    "match" => Some("match"),
                    "matchAll" => Some("matchAll"),
                    "search" => Some("search"),
                    "replace" | "replaceAll" => Some("replace"),
                    "split" => Some("split"),
                    _ => None,
                } && let Some(arg0) = args.first().copied()
                    && !matches!(arg0.unpack(), Unpacked::Undefined | Unpacked::Null)
                    && let Some(argh) = arg0.as_handle().map(Handle::from_raw)
                {
                    let sym = self.well_known_symbol(sym_name);
                    let key = self.member_key(sym);
                    let m = self.read_member(argh, &key)?;
                    if m.as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                    {
                        let mut call_args = alloc::vec![this_val];
                        call_args.extend_from_slice(&args[1.min(args.len())..]);
                        return self.call_with_this(m, arg0, &call_args);
                    }
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
            // A branded `Intl.X.prototype` data method (resolvedOptions/formatToParts/
            // select/of/segment/ListFormat+RelativeTimeFormat format): RequireInternalSlot
            // on `this`, then delegate to the underlying method native.
            if id == N_INTL_PROTO_METHOD {
                return self.intl_proto_method_dispatch(this_val, target, args);
            }
            // A `get Intl.NumberFormat.prototype.format` / `…DateTimeFormat….format` /
            // `get Intl.Collator.prototype.compare` accessor: returns the per-instance
            // bound function.
            if id == N_INTL_BOUND_GETTER {
                return self.intl_bound_getter_dispatch(this_val, target);
            }
            // The per-instance bound `format`/`compare` function itself.
            if id == N_INTL_BOUND_CALL {
                return self.intl_bound_call_dispatch(target, args);
            }
            // A brand-checked `DisposableStack`/`AsyncDisposableStack` prototype
            // method (`use`/`adopt`/`defer`/`dispose`/`disposeAsync`/`move`): the
            // receiver carries the internal-slot brand. The dispatch reads
            // `self.this_val`, so set it for the duration of the call.
            if id == N_DISPOSABLE_STACK_PROTO || id == N_ASYNC_DISPOSABLE_STACK_PROTO {
                let is_async = id == N_ASYNC_DISPOSABLE_STACK_PROTO;
                let saved = core::mem::replace(&mut self.this_val, this_val);
                let r = self.dstack_proto_dispatch(is_async, target, args);
                self.this_val = saved;
                return r;
            }
            // The `get DisposableStack.prototype.disposed` /
            // `get AsyncDisposableStack.prototype.disposed` accessor — handled via
            // the native id below (it brand-checks `this_val`).
            // The dispose callback recorded by `adopt(value, onDispose)`.
            if id == N_DSTACK_ADOPT_CALL {
                return self.dstack_adopt_call(target);
            }
            // A `ShadowRealm.prototype.<method>` (`evaluate`/`importValue`).
            if id == N_SHADOW_REALM_PROTO {
                let saved = core::mem::replace(&mut self.this_val, this_val);
                let r = self.shadow_realm_dispatch(target, args);
                self.this_val = saved;
                return r;
            }
            // A wrapped callable returned across a `ShadowRealm` boundary.
            if id == N_SHADOW_REALM_WRAPPED {
                return self.shadow_realm_wrapped_call(target, args);
            }
            // An `Intl.Locale.prototype` `get` accessor (language/script/region/…).
            if id == N_INTL_LOCALE_ACCESSOR {
                let name = self.realm.string_value(target).unwrap_or_default();
                return self.intl_locale_accessor_dispatch(this_val, &name);
            }
            // An `Intl.Locale.prototype` method (maximize/minimize/toString).
            if id == N_INTL_LOCALE_METHOD {
                let name = self.realm.string_value(target).unwrap_or_default();
                return self.intl_locale_method_dispatch(this_val, &name);
            }
            // An `Intl.DurationFormat.prototype` method (format/formatToParts/resolvedOptions).
            if id == N_INTL_DURATION_METHOD {
                let name = self.realm.string_value(target).unwrap_or_default();
                return self.intl_duration_method_dispatch(this_val, &name, args);
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
                        // A detached or out-of-bounds view reports byteOffset 0.
                        let off = if self.typed_array_detached(h)
                            || self.realm.typed_array_out_of_bounds(h)
                        {
                            0
                        } else {
                            self.realm.typed_byte_offset(h).unwrap_or(0)
                        };
                        NanBox::number(off as f64)
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
                    // IsViewOutOfBounds: a resizable buffer shrank under the view —
                    // `byteLength`/`byteOffset` then throw a TypeError too.
                    let total = self
                        .array_buffer_bytes(bh)
                        .and_then(|b| self.realm.bytes_len(b))
                        .unwrap_or(0);
                    let off = self
                        .realm
                        .get_property(h, DATA_VIEW_OFF)
                        .and_then(|n| n.as_number())
                        .unwrap_or(0.0) as usize;
                    let recorded = self
                        .realm
                        .get_property(h, DATA_VIEW_LEN)
                        .and_then(|n| n.as_number())
                        .map(|n| n as usize);
                    let oob = match recorded {
                        Some(len) => off.checked_add(len).is_none_or(|end| end > total),
                        None => off > total,
                    };
                    if oob {
                        return Err(self.type_error(&alloc::format!(
                            "get DataView.prototype.{name} on an out-of-bounds view"
                        )));
                    }
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
                    // `immutable`: true iff the buffer is immutable and not detached.
                    "immutable" => NanBox::boolean(
                        !detached && self.realm.get_property(h, ARRAY_BUFFER_IMMUTABLE).is_some(),
                    ),
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
                // and `length` are read generically. A string primitive is a heap
                // value (`Cell::Str`) but still needs boxing into a `String`
                // wrapper so its per-unit indices are read as array-like.
                let is_string_prim = this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.string_value(h).is_some());
                let this_obj = if this_val.as_handle().is_none() || is_string_prim {
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
                // Mark this as a generic `Array.prototype.<m>` application so a
                // primitive-wrapper `this` is treated as an array-like object
                // rather than unwrapped to its boxed primitive.
                self.array_proto_generic = true;
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
                    // The `this`-aware statics (`Promise.resolve`/`reject` and the
                    // combinators) are generic over their receiver `C`: a call via
                    // `Promise.all.call(C, …)` must use `C`, not the bound
                    // constructor. Route the real `this_val` through `call_method`
                    // when it is an object; fall back to the bound constructor for a
                    // plain `Promise.all(…)` call (`this` = the constructor itself).
                    let this_aware = matches!(
                        name.as_str(),
                        "all"
                            | "race"
                            | "allSettled"
                            | "any"
                            | "allKeyed"
                            | "allSettledKeyed"
                            | "resolve"
                            | "reject"
                            | "try"
                    );
                    // For a `this`-aware static, a non-object receiver (number,
                    // string, boolean, symbol, null, undefined — including a detached
                    // `const f = Promise.all; f([])` where `this` is undefined) is a
                    // TypeError up front (NewPromiseCapability / OrdinaryToObject would
                    // reject it). Otherwise the real `this` is the constructor `C`.
                    if this_aware && !self.is_object_value(this_val) {
                        return Err(self
                            .type_error(&alloc::format!("Promise.{name} called on a non-object")));
                    }
                    // The combinators run `NewPromiseCapability(C)` unconditionally,
                    // which requires `C` (the receiver) to be a constructor — a
                    // non-constructor object (e.g. `Promise.all.call(eval)`) is a
                    // TypeError. We special-case it here because we know this is the
                    // genuine `Promise` combinator (not an unrelated same-named method,
                    // which the name-dispatch guard in `call_method` cannot tell apart).
                    let is_combinator = matches!(
                        name.as_str(),
                        "all" | "race" | "allSettled" | "any" | "allKeyed" | "allSettledKeyed"
                    );
                    if is_combinator && !self.is_constructor(this_val) {
                        return Err(self.type_error(&alloc::format!(
                            "Promise.{name} called on a non-constructor"
                        )));
                    }
                    let recv = if this_aware { this_val } else { ctor };
                    return Ok(self
                        .call_method(recv, &name, args)?
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
                // `Segments.prototype.containing(index)`: `target` is the segments
                // array; return the segment-data object whose `[index, index+len)`
                // range covers `index` (undefined out of range / non-integer).
                N_INTL_SEGMENTS_CONTAINING => {
                    // ToIntegerOrInfinity(index): a Symbol/BigInt throws; NaN → 0.
                    let idx = self.coerce_to_integer_or_infinity(arg0)?;
                    let mut result = NanBox::undefined();
                    if idx.is_finite() && idx >= 0.0 {
                        let idx = idx as usize;
                        if let Some(elems) = self.realm.array_elements(target).map(<[_]>::to_vec) {
                            // Segments are contiguous and ordered; the containing one
                            // is the last whose `index <= idx`, provided `idx` is
                            // within the input length.
                            let input_len = elems
                                .first()
                                .and_then(|e| e.as_handle())
                                .map(Handle::from_raw)
                                .and_then(|h| self.realm.get_property(h, "input"))
                                .and_then(|v| v.as_handle())
                                .map(Handle::from_raw)
                                .and_then(|ih| self.realm.string_value(ih))
                                .map_or(0, |s| s.encode_utf16().count());
                            if idx < input_len {
                                for e in &elems {
                                    let start = e
                                        .as_handle()
                                        .map(Handle::from_raw)
                                        .and_then(|h| self.realm.get_property(h, "index"))
                                        .and_then(|v| v.as_number())
                                        .unwrap_or(-1.0);
                                    if start >= 0.0 && start as usize <= idx {
                                        result = *e;
                                    }
                                }
                            }
                        }
                    }
                    return Ok(result);
                }
                N_RESOLVE => self.resolve_with(target, arg0),
                N_REJECT => self.settle(target, arg0, false),
                // The `revoke` function from `Proxy.revocable`.
                N_PROXY_REVOKE => self.realm.revoke_proxy(target),
                // An async coroutine resume reaction: `target` is the controller
                // object; resume the parked body with the settled value (fulfil) or
                // by throwing the rejection reason at the `await` point.
                N_ASYNC_RESUME_FULFILL => {
                    if let Some(fid) = self.async_frame_id(target) {
                        self.async_step(fid, target, generator::Resumption::Next(arg0));
                    }
                }
                N_ASYNC_RESUME_REJECT => {
                    if let Some(fid) = self.async_frame_id(target) {
                        self.async_step(fid, target, generator::Resumption::Throw(arg0));
                    }
                }
                // `NewPromiseCapability` executor: capture (resolve, reject) into the
                // bound state object. Per spec, each may be set only once.
                N_PROMISE_CAPABILITY_EXECUTOR => {
                    let resolve = args.first().copied().unwrap_or(NanBox::undefined());
                    let reject = args.get(1).copied().unwrap_or(NanBox::undefined());
                    // GetCapabilitiesExecutor (27.2.1.5.1): it is an error only if a
                    // *non-undefined* resolve/reject was already captured (a prior
                    // call with `undefined`s is allowed and overwritten).
                    let prev_resolve = self
                        .realm
                        .get_property(target, PCAP_RESOLVE)
                        .unwrap_or(NanBox::undefined());
                    let prev_reject = self
                        .realm
                        .get_property(target, PCAP_REJECT)
                        .unwrap_or(NanBox::undefined());
                    if !matches!(prev_resolve.unpack(), Unpacked::Undefined)
                        || !matches!(prev_reject.unpack(), Unpacked::Undefined)
                    {
                        return Err(self.type_error(
                            "Promise capability executor already invoked with non-undefined values",
                        ));
                    }
                    self.realm
                        .set_hidden_property(target, PCAP_RESOLVE, resolve);
                    self.realm.set_hidden_property(target, PCAP_REJECT, reject);
                    return Ok(NanBox::undefined());
                }
                // Promise combinator resolve/reject element closures.
                N_PROMISE_ALL_ELEMENT => return self.promise_all_element(target, arg0),
                N_PROMISE_ALLSETTLED_FULFILL => {
                    return self.promise_allsettled_element(target, arg0, true);
                }
                N_PROMISE_ALLSETTLED_REJECT => {
                    return self.promise_allsettled_element(target, arg0, false);
                }
                N_PROMISE_ANY_ELEMENT => return self.promise_any_element(target, arg0),
                N_PROMISE_ALLKEYED_ELEMENT => {
                    return self.promise_all_keyed_element(target, arg0);
                }
                N_PROMISE_ALLSETTLEDKEYED_FULFILL => {
                    return self.promise_allsettled_keyed_element(target, arg0, true);
                }
                N_PROMISE_ALLSETTLEDKEYED_REJECT => {
                    return self.promise_allsettled_keyed_element(target, arg0, false);
                }
                N_PROMISE_THEN_FINALLY => return self.promise_finally_thunk(target, arg0, true),
                N_PROMISE_CATCH_FINALLY => return self.promise_finally_thunk(target, arg0, false),
                // Value/throw thunks: ignore the (one) argument and return/throw the
                // captured value.
                N_PROMISE_VALUE_THUNK => {
                    return Ok(self
                        .realm
                        .get_property(target, PFIN_VALUE)
                        .unwrap_or(NanBox::undefined()));
                }
                N_PROMISE_THROW_THUNK => {
                    let v = self
                        .realm
                        .get_property(target, PFIN_VALUE)
                        .unwrap_or(NanBox::undefined());
                    return Err(ExecError::Throw(v));
                }
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
        if def.is_arrow {
            // An arrow restores the *lexical* `this`/`new.target`/home it captured at
            // definition (hidden slots), so a call from any site — including via
            // `call`/`apply`/`bind` — sees the defining environment. `invoke_inner`
            // then inherits these `self.*` fields for the arrow body.
            let saved_this = self.this_val;
            let saved_nt = self.new_target;
            let saved_home_obj = self.current_home_object;
            let saved_home = self.current_home;
            let saved_home_static = self.current_home_static;
            if let Some(t) = self.realm.get_property(handle, ARROW_THIS) {
                self.this_val = t;
            }
            if let Some(nt) = self.realm.get_property(handle, ARROW_NEW_TARGET) {
                self.new_target = nt;
            }
            // Restore the captured home context only when one was recorded at
            // definition; an arrow defined without a home (e.g. some field-initializer
            // / direct-eval contexts) keeps inheriting the caller's home so its
            // `super` still resolves.
            if let Some(home_obj) = self.realm.get_property(handle, ARROW_HOME_OBJ) {
                self.current_home_object = home_obj.as_handle().map(Handle::from_raw);
            }
            if let Some(hc) = self
                .realm
                .get_property(handle, ARROW_HOME_CLASS)
                .and_then(|v| v.as_number())
            {
                self.current_home = Some(hc as u32);
                self.current_home_static = self
                    .realm
                    .get_property(handle, ARROW_HOME_STATIC)
                    .is_some_and(|v| self.realm.truthy(v));
            }
            let r = self.invoke(
                def,
                captured,
                this_val,
                args,
                NanBox::handle(handle.to_raw()),
            );
            self.this_val = saved_this;
            self.new_target = saved_nt;
            self.current_home_object = saved_home_obj;
            self.current_home = saved_home;
            self.current_home_static = saved_home_static;
            return r;
        }
        let home_obj = self
            .realm
            .get_property(handle, HOME_OBJECT)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        let saved_home_obj = core::mem::replace(&mut self.current_home_object, home_obj);
        let r = self.invoke(
            def,
            captured,
            this_val,
            args,
            NanBox::handle(handle.to_raw()),
        );
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
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        if self.call_depth >= self.realm.limits.max_call_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            // A proper `RangeError` object (id 2 in `ERROR_NAMES`) so `instanceof
            // RangeError`/`Error` and `.name` work on the caught value.
            let err = self.make_error(N_ERROR_BASE + 2, Some(msg));
            return Err(ExecError::Throw(err));
        }
        self.call_depth += 1;
        let r = self.invoke_inner(def, captured, this_val, args, callee);
        self.call_depth -= 1;
        r
    }

    pub(crate) fn invoke_inner(
        &mut self,
        def: FnDef<'a>,
        captured: Scope,
        this_val: NanBox,
        args: &[NanBox],
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        let call_scope = captured.child();
        let saved = core::mem::replace(&mut self.current, call_scope);
        // A function defined in a module carries that module's import aliases in
        // its captured scope chain; restore them so a named import read inside the
        // body resolves even when the function is called from another module (or
        // across an import cycle). Only swap when the closure is a module function.
        #[cfg(all(feature = "module", feature = "std"))]
        let saved_module_imports = captured
            .module_imports()
            .map(|mi| core::mem::replace(&mut self.module_imports, mi));
        // The callee body opens a new variable environment (set by `hoist_with`);
        // remember the caller's so it is restored on return.
        let saved_var_scope = self.var_scope.clone();
        let saved_annexb = core::mem::take(&mut self.annexb_block_fns);
        // An arrow has no own `this` — it inherits the enclosing one lexically,
        // so leave `self.this_val` unchanged.
        let saved_this = if def.is_arrow {
            self.this_val
        } else {
            // Sloppy-mode `this` coercion (OrdinaryCallBindThis): a strict
            // function keeps `this` as-is; a sloppy function maps `undefined`/
            // `null` to the global object and `ToObject`-boxes a primitive
            // receiver (a number/string/boolean/symbol/bigint) into its wrapper,
            // so `(function(){ this.x = 1; return this; }).apply(1)` mutates and
            // returns a `Number` wrapper.
            let bound = if def.is_strict {
                this_val
            } else if matches!(this_val.unpack(), Unpacked::Undefined | Unpacked::Null) {
                self.global_this
            } else if this_val.as_handle().is_none() {
                // A primitive immediate (number/boolean) — box it.
                self.coerce_to_object(this_val)
            } else if self
                .realm
                .string_value(this_val.as_handle().map(Handle::from_raw).unwrap())
                .is_some()
                || self
                    .realm
                    .symbol_at(this_val.as_handle().map(Handle::from_raw).unwrap())
                    .is_some()
                || self
                    .realm
                    .bigint_at(this_val.as_handle().map(Handle::from_raw).unwrap())
                    .is_some()
            {
                // A primitive stored as a heap cell (string/symbol/bigint) — box it.
                self.coerce_to_object(this_val)
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
        // The lexical class for private-name resolution: an arrow inherits the
        // enclosing one (left untouched); any other function establishes its own
        // captured `lexical_class` — so `#x` inside a nested ordinary function
        // still resolves to its textually-enclosing class even though `super`
        // (driven by `current_home` above) is `None` there.
        let saved_lexical_home = if def.is_arrow {
            self.current_lexical_home
        } else {
            core::mem::replace(&mut self.current_lexical_home, def.lexical_class)
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
        // A non-arrow body brings `new.target` into lexical scope; an arrow is
        // transparent and inherits the enclosing flag (so an arrow defined at the
        // top level still has no `new.target` in scope).
        let saved_nt_scope = if def.is_arrow {
            self.new_target_in_scope
        } else {
            core::mem::replace(&mut self.new_target_in_scope, true)
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
        // Cleared for this call so a nested call / the body never inherits an
        // outer function's parameter set; restored after the call returns.
        let saved_eval_param_names = core::mem::take(&mut self.eval_param_names);
        let result = (|| {
            // A non-arrow function gets an `arguments` array-like of its call
            // arguments. (Arrows inherit the enclosing `arguments`.) Bound *before*
            // the parameters so a parameter default can reference `arguments`
            // (`function f(x = arguments[0]) {}`).
            if !def.is_arrow {
                let arguments = self.make_arguments_object(args, callee);
                self.current.declare("arguments", arguments);
            }
            // While evaluating parameter defaults, expose the formal parameter
            // BoundNames (+ `arguments`) so a sloppy direct `eval("var X")` whose
            // `X` collides with one is an EvalDeclarationInstantiation early error
            // (see `eval_string`). Only relevant when some parameter has a default.
            if def.params.iter().any(|p| p.default.is_some()) {
                let mut names: Vec<String> = Vec::new();
                if !def.is_arrow {
                    names.push(String::from("arguments"));
                }
                for p in def.params {
                    let mut refs: Vec<&str> = Vec::new();
                    collect_binding_idents(&p.target, &mut refs);
                    names.extend(refs.into_iter().map(String::from));
                }
                self.eval_param_names = Some(names);
            }
            // Formal-parameter TDZ: when some parameter has a default (the only
            // way a default can reference another parameter), declare every
            // simple-ident parameter as uninitialized *before* any default runs,
            // so a self or forward reference throws ReferenceError (`(a = a)`,
            // `(a = b, b)`). `bind_pattern` below overwrites each with its real
            // value left to right, lifting it out of the dead zone — so an earlier
            // parameter is already initialized when a later default reads it.
            if def.params.iter().any(|p| p.default.is_some()) {
                for param in def.params {
                    if !param.rest
                        && let BindingTarget::Ident(id) = &param.target
                    {
                        self.current.declare(&id.name, NanBox::tdz());
                    }
                }
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
            // Parameter binding is done; the body must not see the parameter set.
            self.eval_param_names = None;
            // FunctionDeclarationInstantiation: when the formal parameters
            // *contain expressions* (a default, or a computed key in a pattern),
            // the body runs in a separate variable environment nested inside the
            // parameter environment. Body `var`/lexical declarations then live in
            // that child, invisible to closures created by parameter defaults
            // (`((p = () => x) => { var x = 'inner'; })()` — `p` still sees the
            // outer `x`). Per spec the new env is *seeded* with the parameter
            // (and `arguments`) values so a body `var` that names a parameter
            // starts from its value; the two environments are then independent.
            if params_contain_expression(def.params) {
                let body_scope = self.current.child();
                for (name, value, is_const) in self.current.local_bindings() {
                    if is_const {
                        body_scope.declare_const(&name, value);
                    } else {
                        body_scope.declare(&name, value);
                    }
                }
                self.current = body_scope;
            }
            // A generator call does NOT run its body: it captures the
            // parameter-bound scope and ambient state into a suspended
            // [`generator::GenFrame`], returning a lazy generator object. The
            // body runs incrementally on each `next()`.
            if def.is_generator {
                let body: &'a [crate::ast::Stmt] = match def.body {
                    Body::Block(stmts) => stmts,
                    // A generator always has a block body; an (impossible) concise
                    // body yields an empty generator.
                    Body::Expr(_) => &[],
                };
                // A body-level `"use strict"` prologue applies to the whole body.
                if let Body::Block(stmts) = def.body
                    && has_use_strict(stmts)
                {
                    self.strict = true;
                }
                let scope = self.current.clone();
                return Ok(self.make_lazy_generator(body, scope, def.is_async));
            }
            // An `async` (non-generator) function whose body may `await`: do NOT
            // run its body synchronously. Capture the parameter-bound scope into a
            // suspendable coroutine, return its promise immediately, and (after the
            // caller's ambient state is restored, below) run the body up to the
            // first `await` (or completion). At each `await` the coroutine parks
            // and a microtask resumes it on the awaited promise's settlement — so
            // post-`await` code runs as a microtask, not inline.
            //
            // An async function that *never* awaits runs synchronously via
            // `run_body` (its result wrapped in a settled promise below): this is
            // observationally identical (no suspension point to reorder) and reuses
            // the ordinary walker, which fully supports forms — `with`, etc. — that
            // the coroutine lowering only reifies on the suspending path.
            if def.is_async && generator::body_has_await(&def.body) {
                // A body-level `"use strict"` prologue applies to the whole body.
                if let Body::Block(stmts) = def.body
                    && has_use_strict(stmts)
                {
                    self.strict = true;
                }
                let scope = self.current.clone();
                let (id, promise, controller) = self.make_async_frame(def.body, scope);
                self.pending_async_start = Some((id, controller));
                return Ok(NanBox::handle(promise.to_raw()));
            }
            self.run_body(def.body)
        })();
        self.current = saved;
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some(mi) = saved_module_imports {
            self.module_imports = mi;
        }
        self.var_scope = saved_var_scope;
        self.annexb_block_fns = saved_annexb;
        self.this_val = saved_this;
        self.current_home = saved_home;
        self.current_lexical_home = saved_lexical_home;
        self.current_home_static = saved_home_static;
        self.new_target = saved_target;
        self.new_target_in_scope = saved_nt_scope;
        self.eval_depth = saved_eval_depth;
        self.eval_param_names = saved_eval_param_names;
        self.strict = saved_strict;
        // An `async` (non-generator) function: now that the caller's ambient state
        // is restored, drive the coroutine's first synchronous burst (the body up
        // to the first `await` or completion). The coroutine captured its own
        // scope/`this` into its frame, so it runs independently of the restored
        // ambient state. `async_step` settles the returned promise on completion or
        // parks it on the first awaited value.
        if let Some((id, controller)) = self.pending_async_start.take() {
            self.async_step(
                id,
                controller,
                generator::Resumption::Next(NanBox::undefined()),
            );
            return result;
        }
        // An `async` (non-generator) function reaching here did NOT take the
        // coroutine path: either its body never awaits (run synchronously via
        // `run_body`) or its parameter binding threw before the frame was built.
        // Either way the call returns a *promise*: resolved with the body's result
        // (adopting a returned promise) or rejected with a thrown value (including
        // a throwing/TDZ parameter default — AsyncFunctionStart wraps argument
        // binding too), never throwing synchronously.
        if def.is_async && !def.is_generator {
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

    /// `GetPrototypeFromConstructor(newTarget, default)` / the prototype half of
    /// `OrdinaryCreateFromConstructor`: the `[[Prototype]]` for a freshly built
    /// built-in instance. `new_target` is the original constructor for a plain
    /// `new C()`, the subclass for `new Subclass()` (via `super()`), or the
    /// explicit 3rd argument of `Reflect.construct`. When `new_target` is an
    /// object with an object `prototype`, that prototype is used; otherwise the
    /// intrinsic `default_proto` (the constructor's own `.prototype`).
    ///
    /// `callee` is the constructor actually invoked; when `new_target` is the same
    /// callee (the common `new C()` case) the default is returned without a
    /// redundant `prototype` read.
    pub(crate) fn instance_proto(
        &mut self,
        new_target: NanBox,
        callee: NanBox,
        default_proto: Option<Handle>,
    ) -> Option<Handle> {
        // A `new.target` distinct from the callee (a subclass via `super()` or a
        // `Reflect.construct` newTarget) supplies the instance prototype from its
        // own `.prototype` — when that is an object.
        if new_target.as_handle().is_some()
            && new_target.as_handle() != callee.as_handle()
            && let Some(nt) = new_target.as_handle().map(Handle::from_raw)
            && let Some(p) = self.constructor_prototype(nt)
        {
            return Some(p);
        }
        default_proto
    }

    /// `Get(constructor, "prototype")` reduced to a `[[Prototype]]`-eligible object
    /// handle. Resolves a class's prototype (stored in the realm's class-prototype
    /// table), an ordinary function's `.prototype` (the lazily-materialized
    /// `fn_protos` object), and an aux-stored `prototype` data property (a native
    /// constructor, or a function whose `prototype` was reassigned). `None` when the
    /// constructor has no object prototype (so the caller falls back to the
    /// intrinsic default).
    pub(crate) fn constructor_prototype(&mut self, ctor: Handle) -> Option<Handle> {
        // A class: its prototype object (materialized on demand).
        if let Some((class_id, _)) = self.realm.class_at(ctor) {
            return Some(self.class_prototype_by_id(class_id));
        }
        // An ordinary function: its `.prototype` (an aux reassignment wins over the
        // default `fn_protos` object, matching `read_member`).
        if let Some((func_id, _)) = self.realm.function_at(ctor) {
            if let Some(p) = self
                .realm
                .get_property(ctor, "prototype")
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw)
            {
                return Some(p);
            }
            return Some(self.realm.function_prototype(func_id));
        }
        // A native constructor / bound native / any other callable: its `prototype`
        // data property, if it carries one that is an object.
        self.realm
            .get_property(ctor, "prototype")
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
    }

    /// The intrinsic `.prototype` object handle for a constructor bound at the
    /// given global `name` (e.g. `"Map"` → `Map.prototype`).
    pub(crate) fn intrinsic_proto(&mut self, name: &str) -> Option<Handle> {
        self.current
            .get(name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
    }

    /// Builds the cell-bearing base instance for a class that `extends` a native
    /// (`class S extends Map {}` → a real `Map` cell). Runs the native constructor
    /// (resolved from `root_id`) with `args` and `new.target` = the subclass
    /// (`class_handle`), so the resulting cell carries the native internal slots and
    /// its `[[Prototype]]` is the subclass's `.prototype` (via the `newTarget` path
    /// in `construct`). The returned handle becomes the derived instance.
    pub(crate) fn construct_native_base(
        &mut self,
        root_id: u16,
        args: &[NanBox],
        class_handle: NanBox,
    ) -> Result<Handle, ExecError> {
        // The namespace-object constructors (`Array`/`Object`) carry sentinel ids;
        // every other base resolves to its callable native constructor.
        let ctor = self
            .native_ctor_by_id(root_id)
            .map(|h| NanBox::handle(h.to_raw()))
            .ok_or(ExecError::Unsupported(
                "native superclass constructor not found",
            ))?;
        // `new.target` for the base construction is the subclass.
        let saved = self.reflect_new_target.replace(class_handle);
        let result = self.construct(ctor, args);
        self.reflect_new_target = saved;
        let value = result?;
        value
            .as_handle()
            .map(Handle::from_raw)
            .ok_or(ExecError::Unsupported(
                "native superclass produced a non-object",
            ))
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
            let nt = self.reflect_new_target.unwrap_or(callee);
            // `Object(value)` with an actual object value returns it as-is, ignoring
            // newTarget; only the fresh-object path (no/null/undefined arg, or a
            // primitive being wrapped) honors a subclass's `newTarget.prototype`.
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            let result = self.coerce_to_object(v);
            if nt.as_handle() != callee.as_handle()
                && !self.is_object_value(v)
                && let Some(rh) = result.as_handle().map(Handle::from_raw)
                && let Some(proto) = self.instance_proto(nt, callee, None)
            {
                self.realm.set_object_proto(rh, Some(proto));
            }
            return Ok(result);
        }
        // `new Array(...)` — Array is a namespace object, matched by identity.
        // A single number argument is the length; otherwise the elements.
        if self.current.get("Array").and_then(|v| v.as_handle()) == callee.as_handle() {
            let single_len = if args.len() == 1
                && let Some(n) = args[0].as_number()
            {
                // A single number is the length: a non-negative integer fitting
                // uint32. Otherwise a `RangeError`. The array is *sparse* (its
                // indices are holes, not `undefined` data properties); a length
                // beyond the dense cap is recorded as a logical length via
                // `set_array_length` rather than materialized.
                if n < 0.0 || n > f64::from(u32::MAX) || n != f64::from(n as u32) {
                    let m = self.new_str("Invalid array length");
                    return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                }
                Some(n as usize)
            } else {
                None
            };
            let elems = if single_len.is_some() {
                alloc::vec![]
            } else {
                args.to_vec()
            };
            let arr = self.realm.new_array(elems);
            if let Some(len) = single_len {
                self.realm.set_array_length(arr, len);
            }
            // A subclass (`class S extends Array {}` / `Reflect.construct`) links the
            // dense array to `newTarget.prototype` (default `%Array.prototype%`).
            let nt = self.reflect_new_target.unwrap_or(callee);
            if nt.as_handle() != callee.as_handle()
                && let Some(proto) = self.instance_proto(nt, callee, None)
            {
                self.realm.set_object_proto(arr, Some(proto));
            }
            return Ok(NanBox::handle(arr.to_raw()));
        }
        // `new hostCtor(...)` — a host function registered via
        // `register_constructor`: bind a fresh object (whose `[[Prototype]]` is
        // newTarget's `.prototype`) as `this`, run the closure, and apply the
        // constructor return rule (an object result wins; else the fresh `this`).
        if let Some(hid) = self.realm.host_fn_at(handle)
            && self.host_fn_is_constructor(hid)
        {
            let nt = self.reflect_new_target.take().unwrap_or(callee);
            let nt_handle = nt.as_handle().map(Handle::from_raw).unwrap_or(handle);
            let proto = self
                .realm
                .get_property(nt_handle, "prototype")
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
            let instance = self.realm.new_object_with_proto(proto);
            let this = NanBox::handle(instance.to_raw());
            self.realm.set_hidden_property(instance, CTOR_KEY, callee);
            let ret = self.call_with_this(callee, this, args)?;
            if let Some(rh) = ret.as_handle().map(Handle::from_raw)
                && self.is_object_value(ret)
                && !self.is_callable(rh)
            {
                return Ok(ret);
            }
            return Ok(this);
        }
        let Some(id) = self.realm.native_at(handle) else {
            // A callable that is not a constructor — e.g. a built-in method such as
            // `Function.prototype.apply`/`call` (a first-class bound native), or any
            // value reaching here without a `[[Construct]]`. `new` on it is a
            // TypeError (catchable), not an internal "unsupported".
            let m = self.new_str("is not a constructor");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        // The constructor under which the instance is created (`new.target`): the
        // explicit `Reflect.construct` newTarget, the subclass reached via
        // `super()`, or — for a plain `new C()` — the callee itself. Consumed (not
        // peeked) so a `Reflect.construct(NativeCtor, args, NT)` newTarget cannot
        // leak into a subsequent unrelated construction. The typed-array path reads
        // `reflect_new_target` directly, so for those ids we leave it in place and
        // let `link_typed_array_proto` consume the value.
        let is_typed_array_id =
            (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id);
        let native_new_target = if is_typed_array_id {
            self.reflect_new_target.unwrap_or(callee)
        } else {
            self.reflect_new_target.take().unwrap_or(callee)
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
            let result = self.make_intl_formatter(id, args)?;
            // OrdinaryCreateFromConstructor(newTarget): a subclass /
            // `Reflect.construct` newTarget supplies the instance prototype (the
            // formatter defaults to `%Intl.X.prototype%`).
            if let Some(oh) = result.as_handle().map(Handle::from_raw)
                && let Some(proto) = self.instance_proto(native_new_target, callee, None)
            {
                self.realm.set_object_proto(oh, Some(proto));
            }
            return Ok(result);
        }
        // `new Intl.Collator(...)` → an object whose `compare` is a bound function
        // (so `arr.sort(new Intl.Collator().compare)` works); code-point order, no
        // locale tailoring (matching `localeCompare`).
        if id == N_INTL_COLLATOR {
            return Ok(self.make_collator(args));
        }
        // `new Intl.PluralRules(...)` → an object with a `select(n)` method.
        if id == N_INTL_PLURAL_RULES {
            return self.make_plural_rules(args);
        }
        // `new Intl.ListFormat(locale, { type, style })` → an object with a `format(list)`.
        if id == N_INTL_LIST_FORMAT {
            return self.make_list_format(args);
        }
        // `new Intl.RelativeTimeFormat(locale, { numeric, style })` → an object with `format`.
        if id == N_INTL_REL_TIME {
            return self.make_relative_time_format(args);
        }
        // `new Intl.DisplayNames(locale, { type })` → an object with an `of(code)` method.
        if id == N_INTL_DISPLAY_NAMES {
            return self.make_display_names(args);
        }
        // `new Intl.Segmenter(locale, { granularity })` → an object with a `segment(s)` method.
        if id == N_INTL_SEGMENTER {
            return Ok(self.make_segmenter(args));
        }
        // `new Intl.Locale(tag, options)` → a branded Locale with `language`/… accessors.
        if id == N_INTL_LOCALE {
            return self.make_locale(args);
        }
        // `new Intl.DurationFormat(locales, options)` → a branded DurationFormat.
        if id == N_INTL_DURATION_FORMAT {
            return self.make_duration_format(args);
        }
        // `new DisposableStack()` / `new AsyncDisposableStack()`. The instance's
        // `[[Prototype]]` comes from `native_new_target.prototype` (already
        // resolved above, consuming any `Reflect.construct` newTarget).
        if id == N_DISPOSABLE_STACK || id == N_ASYNC_DISPOSABLE_STACK {
            return self.construct_disposable_stack(
                id == N_ASYNC_DISPOSABLE_STACK,
                callee,
                native_new_target,
            );
        }
        // `new ShadowRealm()`.
        if id == N_SHADOW_REALM {
            return self.construct_shadow_realm(callee, native_new_target);
        }
        // `new SuppressedError(error, suppressed, message)`.
        if id == N_SUPPRESSED_ERROR {
            return self.construct_suppressed_error(args, callee, native_new_target);
        }
        // `new Promise(executor)`: run executor(resolve, reject).
        if id == N_PROMISE {
            // The executor must be callable (else a TypeError, before any promise
            // is observably created beyond the allocation).
            let executor = args.first().copied().unwrap_or(NanBox::undefined());
            if !self.is_callable_value(executor) {
                return Err(self.type_error("Promise executor is not a function"));
            }
            let promise = self.fresh_promise();
            let resolve = self.realm.new_bound_native(N_RESOLVE, promise);
            let reject = self.realm.new_bound_native(N_REJECT, promise);
            self.install_fn_name_length(resolve, "", 1);
            self.install_fn_name_length(reject, "", 1);
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
            // Register the instance's `[[Prototype]]`: `newTarget.prototype` for a
            // subclass (`class S extends Date {}` / `Reflect.construct`), else the
            // intrinsic `Date.prototype` — so `Object.getPrototypeOf(date)`,
            // `instanceof`, and `Date.prototype.isPrototypeOf(date)` resolve.
            let default = self.intrinsic_proto("Date");
            if let Some(proto) = self.instance_proto(native_new_target, callee, default) {
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
            // A subclass (`class S extends RegExp {}` / `Reflect.construct`) links
            // the instance to `newTarget.prototype` instead of `RegExp.prototype`.
            let default = self.intrinsic_proto("RegExp");
            if let Some(proto) = self.instance_proto(native_new_target, callee, default) {
                self.realm.set_native_proto(r, proto);
            }
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
        // (sound because GC is never driven mid-execution). `target` must be
        // weakly holdable (an object or a non-registered symbol).
        if id == N_WEAKREF {
            let target = args.first().copied().unwrap_or(NanBox::undefined());
            if !self.can_be_held_weakly(target) {
                return Err(
                    self.type_error("WeakRef: target must be an object or a non-registered symbol")
                );
            }
            let obj = self.realm.new_object();
            self.realm.set_hidden_property(obj, WEAKREF_TARGET, target);
            // Link to `WeakRef.prototype` (or `newTarget.prototype` for a subclass)
            // so `instanceof`, `Object.getPrototypeOf`, and the brand-checking
            // `deref` / `[Symbol.toStringTag]` resolve.
            let default = self.intrinsic_proto("WeakRef");
            if let Some(proto) = self.instance_proto(native_new_target, callee, default) {
                self.realm.set_object_proto(obj, Some(proto));
            }
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new FinalizationRegistry(cb)` — `cb` must be callable. Bounded: with no
        // mid-execution GC the cleanup callback never fires, so the recorded cells
        // are never visited; `register`/`unregister` still maintain `[[Cells]]`.
        if id == N_FINALIZATION_REGISTRY {
            let cb = args.first().copied().unwrap_or(NanBox::undefined());
            if !cb
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.is_callable(h))
            {
                return Err(
                    self.type_error("FinalizationRegistry: cleanup callback must be callable")
                );
            }
            let obj = self.realm.new_object();
            self.realm
                .set_hidden_property(obj, FINREG_TAG, NanBox::boolean(true));
            let cells = self.realm.new_array(Vec::new());
            self.realm
                .set_hidden_property(obj, FINREG_CELLS, NanBox::handle(cells.to_raw()));
            let default = self.intrinsic_proto("FinalizationRegistry");
            if let Some(proto) = self.instance_proto(native_new_target, callee, default) {
                self.realm.set_object_proto(obj, Some(proto));
            }
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new ArrayBuffer(n)` — a zeroed byte store of length `n`.
        if id == N_ARRAY_BUFFER {
            let raw = args.first().map_or(0.0, |v| self.realm.to_number(*v));
            let n = self.validate_alloc_len(raw, "Invalid ArrayBuffer length")?;
            let buf = self.make_array_buffer(n);
            // A subclass (`class S extends ArrayBuffer {}` / `Reflect.construct`)
            // re-links the buffer to `newTarget.prototype` (the default
            // `%ArrayBuffer.prototype%` was installed by `make_array_buffer`).
            if native_new_target.as_handle() != callee.as_handle()
                && let Some(proto) = self.instance_proto(native_new_target, callee, None)
            {
                self.realm.set_object_proto(buf, Some(proto));
            }
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
            // The instance's `[[Prototype]]` is `DataView.prototype` (or
            // `newTarget.prototype` for a subclass / `Reflect.construct`), so
            // inherited members (the get*/set* methods, the accessors,
            // `Symbol.toStringTag`) resolve through the chain.
            let default = self.intrinsic_proto("DataView");
            let dv_proto = self.instance_proto(native_new_target, callee, default);
            let obj = self.realm.new_object_with_proto(dv_proto);
            let buf = args.first().copied().unwrap_or(NanBox::undefined());
            // RequireInternalSlot(buffer, [[ArrayBufferData]]): the first argument
            // must be an ArrayBuffer (has the bytes slot), else a TypeError — *before*
            // the offset coercion.
            let Some(bh) = buf
                .as_handle()
                .map(Handle::from_raw)
                .filter(|h| self.realm.get_property(*h, ARRAY_BUFFER_BYTES).is_some())
            else {
                return Err(self.type_error("DataView buffer must be an ArrayBuffer"));
            };
            // ToIndex(byteOffset) (a Symbol / abrupt valueOf throws; negative /
            // over-2^53 is a RangeError) — runs before the detached check.
            let byte_off =
                self.coerce_to_index(args.get(1).copied().unwrap_or(NanBox::undefined()))? as usize;
            // IsDetachedBuffer → TypeError (after the offset coercion).
            self.guard_detached_buffer(bh)?;
            let buf_len = self
                .array_buffer_bytes(bh)
                .and_then(|h| self.realm.bytes_len(h))
                .unwrap_or(0);
            if byte_off > buf_len {
                let m = self.new_str("Start offset is outside the bounds of the buffer");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            self.realm.set_hidden_property(obj, DATA_VIEW_BUF, buf);
            self.realm
                .set_hidden_property(obj, DATA_VIEW_OFF, NanBox::number(byte_off as f64));
            // An explicit byteLength (3rd arg) is honored; otherwise the view spans
            // the rest of the buffer from `byteOffset`.
            if let Some(len) = args.get(2)
                && !matches!(len.unpack(), Unpacked::Undefined)
            {
                // ToIndex(byteLength) (a Symbol / abrupt valueOf throws).
                let view_len = self.coerce_to_index(*len)? as usize;
                // M1: validate `byteOffset + byteLength <= buffer.byteLength` with
                // checked arithmetic (a saturated length must not wrap past the end).
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
                // Spec order (InitializeTypedArrayFromArrayBuffer):
                //   1. offset = ? ToIndex(byteOffset)   — Symbol/abrupt → throw here
                //   2. if offset % elementSize ≠ 0 → RangeError
                //   3. if length ≠ undefined: newLength = ? ToIndex(length)
                //   4. **then** IsDetachedBuffer(buffer) → TypeError
                // So the index coercions (which may run user `valueOf`, possibly
                // detaching the buffer) happen *before* the detached check.
                let byte_off = match args
                    .get(1)
                    .filter(|a| !matches!(a.unpack(), Unpacked::Undefined))
                {
                    Some(a) => {
                        // ToIndex: a Symbol / abrupt valueOf throws; a negative or
                        // over-2^53 value is a RangeError.
                        let off = self.coerce_to_index(*a)? as usize;
                        if !off.is_multiple_of(elem_size) {
                            let m =
                                self.new_str("start offset must be a multiple of the element size");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        off
                    }
                    None => 0,
                };
                let explicit_length = args
                    .get(2)
                    .is_some_and(|a| !matches!(a.unpack(), Unpacked::Undefined));
                // ToIndex(length) (if present) before the detached check.
                let explicit_len_val = match args
                    .get(2)
                    .filter(|a| !matches!(a.unpack(), Unpacked::Undefined))
                {
                    Some(a) => Some(self.coerce_to_index(*a)? as usize),
                    None => None,
                };
                // Now the buffer must not be detached (a coercion above may have
                // detached it) and its (possibly changed) byte length is re-read.
                self.guard_detached_buffer(bh)?;
                let bytes_h = self.array_buffer_bytes(bh).unwrap();
                let total = self.realm.bytes_len(bytes_h).unwrap_or(0);
                if byte_off > total {
                    let m = self.new_str("Start offset is outside the bounds of the buffer");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                let avail = total - byte_off;
                let length = match explicit_len_val {
                    Some(len) => {
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
                // A view created with no explicit length over a *resizable* buffer
                // auto-tracks the buffer's length on resize (per spec). A
                // fixed-length view (or one over a non-resizable buffer) does not.
                if !explicit_length && self.realm.get_property(bh, ARRAY_BUFFER_MAXLEN).is_some() {
                    self.realm.mark_length_tracking(view);
                }
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
                && let Some(arg0) = args.first().copied()
                && self.is_object_value(arg0)
                && let Some(h) = arg0.as_handle().map(Handle::from_raw)
            {
                // `Get(O, @@iterator)` through `read_member` so an accessor /
                // inherited iterator is observed. A present-but-not-callable
                // `@@iterator` is a TypeError (GetIterator step 4).
                let iter_sym = self.well_known_symbol("iterator");
                let iter_key = self.member_key(iter_sym);
                let iter_fn = self.read_member(h, &iter_key)?;
                let has_iter = !matches!(iter_fn.unpack(), Unpacked::Undefined | Unpacked::Null);
                if has_iter
                    && !iter_fn
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    let m = self.new_str("typed array source is not iterable");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
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
                    // `len = ? LengthOfArrayLike(O)` = ToLength(Get(O, "length")):
                    // ToIntegerOrInfinity (a Symbol length is a TypeError; a
                    // throwing `valueOf` propagates), then a NaN / negative / absent
                    // `length` clamps to **0** (so `new Uint8Array({})` is an empty
                    // array, not a RangeError). A length beyond the allocatable cap
                    // is a RangeError.
                    let len_val = self.read_member(h, "length")?;
                    let len_f = self.coerce_to_integer_or_infinity(len_val)?;
                    let len = if len_f <= 0.0 {
                        0
                    } else if len_f > self.realm.limits.max_array_len as f64 {
                        let m = self.new_str("Invalid typed array length");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    } else {
                        len_f as usize
                    };
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
                    // `new TA(length)`: ToIndex(length) — a Symbol / abrupt valueOf
                    // throws a TypeError (not a downstream RangeError); the result is
                    // then validated against the allocation cap.
                    let n = self.coerce_to_index(v)? as f64;
                    self.validate_alloc_len(n, "Invalid typed array length")?
                }
                (None, None) => 0,
            };
            let buf = self.make_array_buffer(length * elem_size);
            let bytes_h = self.array_buffer_bytes(buf).unwrap();
            let view = self.realm.new_typed_array(bytes_h, buf, 0, length, kind);
            if let Some(s) = src {
                // For a BigInt typed array, ToBigInt every source element up front
                // (a Number element throws TypeError, per spec). For a numeric kind,
                // ToNumber each element (so a wrapper / a `valueOf`-bearing object /
                // a numeric string coerces, rather than being written as NaN by the
                // infallible bulk write). When the source was a real array taken via
                // `elements_vec` (not the already-coerced array-like/iterable path)
                // these values are still raw, so the coercion is required here.
                let s = if is_bigint_kind(kind) {
                    let mut coerced = Vec::with_capacity(s.len());
                    for v in s {
                        coerced.push(self.coerce_typed_array_write(view, v)?);
                    }
                    coerced
                } else {
                    let mut coerced = Vec::with_capacity(s.len());
                    for v in s {
                        coerced.push(self.coerce_to_number(v)?);
                    }
                    coerced
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
                    // `new Number(value)`: ToNumber(value) (number hint) — a Symbol or
                    // an abrupt `valueOf`/`toString` throws (was an infallible
                    // `to_number` that swallowed these). A BigInt converts to a double.
                    let n = match args.first().copied() {
                        None => 0.0,
                        Some(v) => {
                            if let Some(big) = v
                                .as_handle()
                                .and_then(|r| self.realm.bigint_at(Handle::from_raw(r)))
                            {
                                big.to_f64()
                            } else {
                                let p = self.coerce_to_number(v)?;
                                self.realm.to_number(p)
                            }
                        }
                    };
                    NanBox::number(n)
                }
                N_STRING => {
                    // `new String(value)`: ToString(value) — an object runs its
                    // `toString`/`valueOf`/`@@toPrimitive` (abrupt-propagating), and a
                    // Symbol is a TypeError (unlike `String(symbol)` as a function).
                    let s = match args.first().copied() {
                        None => String::new(),
                        Some(v) => self.coerce_to_string(v)?,
                    };
                    self.new_str(&s)
                }
                _ => NanBox::boolean(
                    self.realm
                        .truthy(args.first().copied().unwrap_or(NanBox::undefined())),
                ),
            };
            let wrapper = self.make_primitive_wrapper(prim, id);
            // A subclass (`class S extends Number {}` / `Reflect.construct`) links
            // the wrapper to `newTarget.prototype` (the default intrinsic was set by
            // `make_primitive_wrapper`).
            if native_new_target.as_handle() != callee.as_handle()
                && let Some(wh) = wrapper.as_handle().map(Handle::from_raw)
                && let Some(proto) = self.instance_proto(native_new_target, callee, None)
            {
                self.realm.set_object_proto(wh, Some(proto));
            }
            return Ok(wrapper);
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
        let default = self.intrinsic_proto(ctor_name);
        if let Some(proto) = self.instance_proto(native_new_target, callee, default) {
            // A collection is a non-object cell: its `[[Prototype]]` link is kept
            // in the realm's native-proto table (read by `object_proto`). A subclass
            // (`class S extends Map {}` / `Reflect.construct`) links to
            // `newTarget.prototype`.
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
    pub(crate) fn apply_native_super(
        &mut self,
        native_id: u16,
        instance: Handle,
        args: &[NanBox],
    ) -> Result<(), ExecError> {
        // `class S extends DisposableStack {}` (or the async variant): `super()`
        // stamps the internal-slot brand + an empty disposer list and
        // `disposed = false` onto the (already-allocated, class-proto-linked)
        // instance object — the analog of the native `construct_*` initialization.
        if native_id == N_DISPOSABLE_STACK || native_id == N_ASYNC_DISPOSABLE_STACK {
            self.init_disposable_stack_super(native_id == N_ASYNC_DISPOSABLE_STACK, instance);
            return Ok(());
        }
        // `class S extends ShadowRealm {}`: brand + allocate the persistent scope.
        if native_id == N_SHADOW_REALM {
            self.init_shadow_realm_super(instance);
            return Ok(());
        }
        // `class S extends Intl.NumberFormat {}` / `Intl.DateTimeFormat`: `super()`
        // initializes the formatter's internal slots on the (already subclass-proto-
        // linked) instance, so `format`/`resolvedOptions`/`formatToParts` resolve.
        if native_id == N_INTL_NUMBER_FORMAT || native_id == N_INTL_DATETIME_FORMAT {
            self.init_intl_formatter_state(instance, native_id, args)?;
            return Ok(());
        }
        // `Symbol` and `BigInt` are callable but have no `[[Construct]]`: extending
        // them is allowed, but `new Subclass()` runs `super()` into a non-existent
        // constructor — a TypeError (`Symbol is not a constructor`).
        if native_id == N_SYMBOL {
            return Err(self.type_error("Symbol is not a constructor"));
        }
        if native_id == N_BIGINT {
            return Err(self.type_error("BigInt is not a constructor"));
        }
        // `class W extends WeakRef {}`: `super(target)` validates the (weakly
        // holdable) target and stamps the internal slot onto the (class-proto-
        // linked) instance, so `deref()` and the brand check resolve.
        if native_id == N_WEAKREF {
            let target = args.first().copied().unwrap_or(NanBox::undefined());
            if !self.can_be_held_weakly(target) {
                return Err(
                    self.type_error("WeakRef: target must be an object or a non-registered symbol")
                );
            }
            self.realm
                .set_hidden_property(instance, WEAKREF_TARGET, target);
            return Ok(());
        }
        // `class F extends FinalizationRegistry {}`: `super(cleanupCallback)`
        // validates the callback and brands the instance with an empty cell list.
        if native_id == N_FINALIZATION_REGISTRY {
            let cb = args.first().copied().unwrap_or(NanBox::undefined());
            if !cb
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.is_callable(h))
            {
                return Err(
                    self.type_error("FinalizationRegistry: cleanup callback must be callable")
                );
            }
            self.realm
                .set_hidden_property(instance, FINREG_TAG, NanBox::boolean(true));
            let cells = self.realm.new_array(Vec::new());
            self.realm
                .set_hidden_property(instance, FINREG_CELLS, NanBox::handle(cells.to_raw()));
            return Ok(());
        }
        // `class S extends SuppressedError {}`: apply error/suppressed/message.
        if native_id == N_SUPPRESSED_ERROR {
            self.init_suppressed_error_super(instance, args);
            return Ok(());
        }
        // `class P extends Promise {}`: `super(executor)` initializes the promise
        // machinery on the (already class-proto-linked) instance. The instance is
        // an ordinary object, so it can't hold the `Rc<PromiseState>` directly —
        // back it with a fresh `Cell::Promise` stored in a hidden `[[PromiseState]]`
        // slot that `Realm::promise_state` follows, then bind resolve/reject to the
        // backing cell and run the executor (a throw rejects, per the spec).
        if native_id == N_PROMISE {
            let executor = args.first().copied().unwrap_or(NanBox::undefined());
            if !self.is_callable_value(executor) {
                return Err(self.type_error("Promise executor is not a function"));
            }
            let backing = self.fresh_promise();
            self.realm.set_hidden_property(
                instance,
                crate::realm::PROMISE_STATE_SLOT,
                NanBox::handle(backing.to_raw()),
            );
            let resolve = self.realm.new_bound_native(N_RESOLVE, backing);
            let reject = self.realm.new_bound_native(N_REJECT, backing);
            self.install_fn_name_length(resolve, "", 1);
            self.install_fn_name_length(reject, "", 1);
            let r = self.call(
                executor,
                &[
                    NanBox::handle(resolve.to_raw()),
                    NanBox::handle(reject.to_raw()),
                ],
            );
            if let Err(ExecError::Throw(e)) = r {
                self.settle(backing, e, false);
            } else {
                r?;
            }
            return Ok(());
        }
        // Error family: set `message` and the default `name` (a `this.name = …`
        // after `super()` may override it).
        if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&native_id) {
            // A subclass `super()` into an Error base inherits `[[ErrorData]]`:
            // stamp the brand onto the (class-proto-linked) instance.
            self.realm
                .set_hidden_property(instance, ERROR_DATA, NanBox::boolean(true));
            let name = ERROR_NAMES[(native_id - N_ERROR_BASE) as usize];
            let name_v = self.new_str(name);
            self.realm.set_property(instance, "name", name_v);
            // `AggregateError(errors, message, options)` takes its message *second*
            // and exposes an own `.errors` array drained from the first argument;
            // every other error takes `(message, options)`.
            let is_aggregate = native_id == N_ERROR_BASE + 5;
            let (msg_arg, opts_arg) = if is_aggregate {
                (args.get(1).copied(), args.get(2))
            } else {
                (args.first().copied(), args.get(1))
            };
            if is_aggregate {
                let errors = args.first().copied().unwrap_or(NanBox::undefined());
                let list = self.iterate_values(errors)?;
                let arr = self.realm.new_array(list);
                self.realm
                    .set_property(instance, "errors", NanBox::handle(arr.to_raw()));
                self.realm.mark_hidden(instance, "errors");
            }
            let msg = match msg_arg {
                Some(m) if !matches!(m.unpack(), Unpacked::Undefined) => {
                    let s = self.realm.to_display_string(m);
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
            if let Some(opts) = opts_arg
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
        Ok(())
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
                    // The default comparator orders by `ToString` of each element
                    // (which invokes a user `toString`/`valueOf`, so abrupt
                    // completions propagate), compared by code units.
                    let a = self.coerce_to_string(elems[j - 1])?;
                    let b = self.coerce_to_string(elems[j])?;
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

/// ECMAScript `ContainsExpression` for a formal parameter list: whether any
/// parameter has a default initializer or a computed property key in a
/// destructuring pattern. When true, a function's body runs in a variable
/// environment distinct from its parameter environment (FunctionDeclaration-
/// Instantiation), so a parameter-default closure and a body `var` of the same
/// name are independent.
fn params_contain_expression(params: &[crate::ast::Param]) -> bool {
    params
        .iter()
        .any(|p| p.default.is_some() || target_contains_expression(&p.target))
}

/// `ContainsExpression` for a single binding target (recurses through nested
/// array/object destructuring patterns).
fn target_contains_expression(target: &crate::ast::BindingTarget) -> bool {
    use crate::ast::{ArrayPatternElement, BindingTarget, PropertyKey};
    match target {
        BindingTarget::Ident(_) => false,
        BindingTarget::Array(arr) => arr.elements.iter().any(|el| match el {
            ArrayPatternElement::Hole => false,
            ArrayPatternElement::Item {
                target, default, ..
            } => default.is_some() || target_contains_expression(target),
            ArrayPatternElement::Rest { target, .. } => target_contains_expression(target),
        }),
        BindingTarget::Object(obj) => {
            obj.properties.iter().any(|p| {
                p.default.is_some()
                    || matches!(p.key, PropertyKey::Computed(_))
                    || target_contains_expression(&p.value)
            }) || obj
                .rest
                .as_ref()
                .is_some_and(|r| target_contains_expression(r))
        }
    }
}
