use super::*;

impl<'a> Interp<'a> {
    /// Invokes a built-in by id.
    pub(crate) fn call_native(&mut self, id: u16, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        Ok(match id {
            // `Date.prototype[Symbol.toPrimitive](hint)`: requires an object `this`,
            // maps the hint to a preferred type, then runs OrdinaryToPrimitive.
            N_DATE_TO_PRIMITIVE => {
                let this = self.this_val;
                if !self.is_object_value(this) {
                    return Err(
                        self.type_error("Date.prototype[Symbol.toPrimitive] called on non-object")
                    );
                }
                let hint = self.realm.to_display_string(arg(0));
                let try_hint = match hint.as_str() {
                    "string" | "default" => "string",
                    "number" => "number",
                    _ => {
                        return Err(self.type_error("invalid hint for Symbol.toPrimitive"));
                    }
                };
                return self.ordinary_to_primitive(this, try_hint);
            }
            // `Date.prototype.toJSON(key)` — generic: ToPrimitive(this, number); a
            // non-finite Number result is `null`; otherwise call `this.toISOString()`.
            N_DATE_TO_JSON => {
                let this = self.this_val;
                let obj = self.coerce_to_object(this);
                let tv = self.coerce_primitive(obj, "number")?;
                if let Some(n) = tv.as_number()
                    && !n.is_finite()
                {
                    return Ok(NanBox::null());
                }
                let Some(oh) = obj.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("Date.prototype.toJSON called on non-object"));
                };
                let iso = self.read_member(oh, "toISOString")?;
                if !iso
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("toISOString is not callable"));
                }
                return self.call_with_this(iso, obj, &[]);
            }
            N_MATH_MAX => {
                // ToNumber every argument first (in order, propagating any abrupt
                // completion), then reduce — so each element's `valueOf` runs even
                // when an earlier element is `NaN`.
                let mut coerced = Vec::with_capacity(args.len());
                for a in args {
                    let num = self.coerce_to_number(*a)?;
                    coerced.push(self.realm.to_number(num));
                }
                let mut m = f64::NEG_INFINITY;
                for n in coerced {
                    if n.is_nan() {
                        m = f64::NAN;
                    } else if !m.is_nan()
                        && (n > m || (n == 0.0 && m == 0.0 && n.is_sign_positive()))
                    {
                        // `+0` is treated as greater than `-0`.
                        m = n;
                    }
                }
                NanBox::number(m)
            }
            N_MATH_MIN => {
                let mut coerced = Vec::with_capacity(args.len());
                for a in args {
                    let num = self.coerce_to_number(*a)?;
                    coerced.push(self.realm.to_number(num));
                }
                let mut m = f64::INFINITY;
                for n in coerced {
                    if n.is_nan() {
                        m = f64::NAN;
                    } else if !m.is_nan()
                        && (n < m || (n == 0.0 && m == 0.0 && n.is_sign_negative()))
                    {
                        // `-0` is treated as less than `+0`.
                        m = n;
                    }
                }
                NanBox::number(m)
            }
            N_MATH_ABS => NanBox::number(self.realm.to_number(arg(0)).abs()),
            N_STRING => {
                // `String(obj)` runs the object through ToString (string hint),
                // honoring a custom `toString`.
                let p = self.coerce_object(arg(0), "string")?;
                let s = self.realm.to_display_string(p);
                NanBox::handle(self.realm.new_string(&s).to_raw())
            }
            N_NUMBER => {
                // `Number()` with no arguments is `+0`.
                if args.is_empty() {
                    NanBox::number(0.0)
                } else if let Some(big) = arg(0)
                    .as_handle()
                    .and_then(|r| self.realm.bigint_at(Handle::from_raw(r)))
                {
                    // `Number(bigint)` converts to the nearest double.
                    NanBox::number(big.to_f64())
                } else {
                    // `Number(value)` runs `value` through ToNumber (number hint),
                    // honoring a custom `valueOf`/`Symbol.toPrimitive` and throwing
                    // a TypeError for a Symbol.
                    let p = self.coerce_to_number(arg(0))?;
                    NanBox::number(self.realm.to_number(p))
                }
            }
            N_BOOLEAN => NanBox::boolean(self.realm.truthy(arg(0))),
            N_SYMBOL => {
                // A no-argument `Symbol()` has an `undefined` description, marked
                // with a reserved sentinel (distinct from `Symbol("")`).
                let desc = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                    String::from(SYMBOL_NO_DESC)
                } else {
                    self.realm.to_display_string(arg(0))
                };
                NanBox::handle(self.realm.new_symbol(&desc).to_raw())
            }
            N_BIGINT => {
                // `BigInt(value)`: ToPrimitive(value, number); if the resulting
                // primitive is a Number, apply NumberToBigInt (a RangeError for a
                // non-integer or non-finite value); otherwise apply ToBigInt.
                let v = arg(0);
                let prim = self.coerce_primitive(v, "number")?;
                let n = match prim.unpack() {
                    Unpacked::Number(num) => {
                        // NumberToBigInt: only an exact integer converts; a
                        // fractional or non-finite value is a `RangeError`.
                        if !num.is_finite() || num != trunc_toward_zero(num) {
                            let m = self.new_str("The number is not a safe integer");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        if num.abs() < 1.7e38 {
                            crate::bignum::BigInt::from_i128(num as i128)
                        } else {
                            // Exact integers beyond i128's range are reconstructed
                            // from their decimal text (Rust prints integer-valued
                            // f64 exactly with the default formatter).
                            parse_bigint(&alloc::format!("{num}"))
                        }
                    }
                    _ => {
                        // ToBigInt on the primitive (string/boolean/bigint/etc.).
                        // Guard the O(n²) string parse against pathological input.
                        if let Some(raw) = prim.as_handle() {
                            let h = Handle::from_raw(raw);
                            if let Some(s) = self.realm.string_value(h)
                                && s.trim().len() as u64 > self.realm.limits.max_bigint_bits
                            {
                                let m = self.new_str("Maximum BigInt size exceeded");
                                return Err(ExecError::Throw(
                                    self.make_error(N_RANGE_ERROR, Some(m)),
                                ));
                            }
                        }
                        self.coerce_to_bigint(prim)?
                    }
                };
                NanBox::handle(self.realm.new_bigint(n).to_raw())
            }
            N_FUNCTION => {
                // `Function(args…, body)` called as a plain function behaves the
                // same as `new Function(…)`: it builds and returns a fresh
                // anonymous function from the supplied parameter/body source.
                return self.build_function_constructor(args);
            }
            N_EVAL => {
                // Reaching `eval` through `call_native` means an *indirect* eval
                // (the callee wasn't the literal identifier `eval` — e.g.
                // `(0, eval)(s)`, `var e = eval; e(s)`, `globalThis.eval(s)`). A
                // direct eval is intercepted at the call site (see `Expr::Call`).
                // `eval(x)` returns `x` unchanged when it isn't a string.
                let v = arg(0);
                let Some(source) = v
                    .as_handle()
                    .and_then(|raw| self.realm.string_value(Handle::from_raw(raw)))
                else {
                    return Ok(v);
                };
                return self.eval_string(&source, false);
            }
            N_PARSE_INT => {
                let s = self.realm.to_display_string(arg(0));
                let radix = match args.get(1) {
                    Some(r) if !matches!(r.unpack(), Unpacked::Undefined) => {
                        let n = self.realm.to_number(*r);
                        // Keep the sign (a `… as u32` cast saturates a negative
                        // radix to 0, which would wrongly default to base 10).
                        if n.is_finite() { n as i64 } else { 0 }
                    }
                    _ => 0,
                };
                // A nonzero radix outside [2, 36] is invalid → NaN; 0 means "infer".
                if radix != 0 && !(2..=36).contains(&radix) {
                    NanBox::number(f64::NAN)
                } else {
                    NanBox::number(parse_int(&s, radix as u32))
                }
            }
            N_CONSOLE_LOG => {
                let line: Vec<String> = args
                    .iter()
                    .map(|a| self.realm.to_display_string(*a))
                    .collect();
                self.output.push_str(&line.join(" "));
                self.output.push('\n');
                NanBox::undefined()
            }
            N_JSON_STRINGIFY => {
                // Optional `replacer` (arg 1): a function transforms each value,
                // an array allowlists object keys.
                let mut value = arg(0);
                if let Some(rh) = arg(1).as_handle().map(Handle::from_raw) {
                    if self.is_callable(rh) {
                        let holder = self.realm.new_object();
                        self.realm.set_property(holder, "", value);
                        value = self.json_apply_replacer(holder, "", value, arg(1))?;
                    } else if self.realm.is_array(rh) {
                        let allow: Vec<String> = self
                            .realm
                            .array_elements(rh)
                            .map(<[_]>::to_vec)
                            .unwrap_or_default()
                            .iter()
                            .map(|e| self.realm.to_display_string(*e))
                            .collect();
                        value = self.json_filter_keys(value, &allow);
                    }
                }
                // Optional `space` (arg 2): a number → that many spaces, a string
                // → that string (both capped at 10), else compact output.
                let space = arg(2);
                let indent = if let Some(n) = space.as_number() {
                    " ".repeat((n.max(0.0) as usize).min(10))
                } else if let Some(s) = space
                    .as_handle()
                    .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
                {
                    s.chars().take(10).collect()
                } else {
                    String::new()
                };
                let result = if indent.is_empty() {
                    // Interpreter-aware: honors `toJSON` and invokes getters.
                    self.json_to_string(value)?
                } else {
                    match crate::json::try_stringify_pretty(&self.realm, value, &indent) {
                        Ok(r) => r,
                        Err(crate::json::Circular) => {
                            let m = self.new_str("Converting circular structure to JSON");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                    }
                };
                match result {
                    Some(s) => NanBox::handle(self.realm.new_string(&s).to_raw()),
                    None => NanBox::undefined(),
                }
            }
            N_JSON_PARSE => {
                let text = self.realm.to_display_string(arg(0));
                let chars: Vec<char> = text.chars().collect();
                let mut pos = 0;
                let value = self.json_parse(&chars, &mut pos, 0)?;
                skip_ws(&chars, &mut pos);
                if pos != chars.len() {
                    return Err(self.json_error("Unexpected token in JSON"));
                }
                // An optional `reviver` transforms each value bottom-up.
                let reviver = arg(1);
                if reviver
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    let holder = self.realm.new_object();
                    self.realm.set_property(holder, "", value);
                    return self.json_revive(holder, "", reviver);
                }
                value
            }
            N_OBJECT_KEYS => {
                // ToObject(O): `null`/`undefined` throws (a primitive coerces to a
                // wrapper with no own enumerable keys, so it is left as-is here).
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Object.keys called on null or undefined"));
                }
                // A proxy with an `ownKeys` trap drives `Object.keys` itself.
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    let boxed: Vec<NanBox> = keys.iter().map(|k| self.new_str(k)).collect();
                    return Ok(NanBox::handle(self.realm.new_array(boxed).to_raw()));
                }
                let target = arg(0)
                    .as_handle()
                    .map(|raw| self.proxy_key_target(Handle::from_raw(raw)));
                let mut keys: Vec<alloc::string::String> = Vec::new();
                if let Some(h) = target {
                    // An array's own enumerable keys are its integer indices (in
                    // ascending order) — stored as elements, not named properties.
                    // A VM closure backs onto an array but is a function, so its
                    // "indices" (captured cells) are not enumerable keys.
                    if let Some(len) = self.realm.array_length(h)
                        && !self.realm.is_vm_function(h)
                    {
                        for i in 0..len {
                            keys.push(alloc::format!("{i}"));
                        }
                    }
                    if let Some(named) = self.realm.object_keys(h) {
                        keys.extend(named);
                    } else {
                        // An array/function/native/class keeps named properties in
                        // its auxiliary object (e.g. `arr.custom`, a match result's
                        // `index`/`input`, a class's enumerable static fields —
                        // static methods/accessors are non-enumerable and excluded).
                        keys.extend(self.realm.aux_named_keys(h));
                    }
                }
                let boxed: Vec<NanBox> = keys.iter().map(|k| self.new_str(k)).collect();
                NanBox::handle(self.realm.new_array(boxed).to_raw())
            }
            N_OBJECT_FREEZE => {
                if let Some(raw) = arg(0).as_handle() {
                    self.realm.freeze_object(Handle::from_raw(raw));
                }
                arg(0) // returns the (now frozen) object
            }
            N_OBJECT_SEAL => {
                if let Some(raw) = arg(0).as_handle() {
                    self.realm.seal_object(Handle::from_raw(raw));
                }
                arg(0)
            }
            N_OBJECT_PREVENT_EXT => {
                if let Some(raw) = arg(0).as_handle() {
                    self.realm.prevent_extensions(Handle::from_raw(raw));
                }
                arg(0)
            }
            N_OBJECT_IS_SEALED => {
                // A non-object argument (a primitive) is reported as sealed.
                let v = arg(0);
                let sealed = !self.is_object_value(v)
                    || v.as_handle()
                        .is_some_and(|raw| self.realm.is_sealed(Handle::from_raw(raw)));
                NanBox::boolean(sealed)
            }
            N_OBJECT_IS_EXTENSIBLE => {
                if let Some(obj) = arg(0).as_handle().map(Handle::from_raw) {
                    self.is_extensible_of(obj)?
                } else {
                    NanBox::boolean(false)
                }
            }
            // `Object.create(proto)` — a new object with the given prototype
            // (`null` → no prototype).
            N_OBJECT_CREATE => {
                // The prototype argument must be an Object or `null` (ECMA-262
                // step 1) — any other value (incl. `undefined` and primitives) is a
                // TypeError.
                if !matches!(arg(0).unpack(), Unpacked::Null) && !self.is_object_value(arg(0)) {
                    return Err(self.type_error("Object prototype may only be an Object or null"));
                }
                let proto = arg(0).as_handle().map(Handle::from_raw);
                let obj = self.realm.new_object_with_proto(proto);
                // Optional second argument (Properties): when present (not
                // `undefined`), it is ToObject'd — `null`/a primitive throws a
                // TypeError — then each own enumerable descriptor is applied.
                if !matches!(arg(1).unpack(), Unpacked::Undefined) {
                    let descs = self.require_object_coercible_to_object(arg(1), "Object.create")?;
                    self.apply_property_descriptors(obj, descs)?;
                }
                NanBox::handle(obj.to_raw())
            }
            N_OBJECT_GET_PROTO => {
                // ToObject(O): `null`/`undefined` throws; a primitive is boxed and
                // its prototype returned.
                let obj =
                    self.require_object_coercible_to_object(arg(0), "Object.getPrototypeOf")?;
                self.get_proto_of(obj)?
            }
            N_OBJECT_SET_PROTO => {
                // RequireObjectCoercible(O): `null`/`undefined` throws.
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(
                        self.type_error("Object.setPrototypeOf called on null or undefined")
                    );
                }
                // The proto must be an Object or `null` (else a TypeError).
                if !matches!(arg(1).unpack(), Unpacked::Null) && !self.is_object_value(arg(1)) {
                    return Err(self.type_error("Object prototype may only be an Object or null"));
                }
                if let Some(raw) = arg(0).as_handle() {
                    let proto = arg(1).as_handle().map(Handle::from_raw);
                    // A failed [[SetPrototypeOf]] (a non-extensible object) throws.
                    if !self.set_proto_of(Handle::from_raw(raw), proto)? {
                        return Err(self.type_error(
                            "Object.setPrototypeOf: cannot set prototype of a non-extensible object",
                        ));
                    }
                }
                arg(0)
            }
            // `Object.defineProperty(obj, key, descriptor)` — a `value`
            // descriptor sets the property; a `get`/`set` descriptor defines an
            // accessor.
            N_OBJECT_DEFINE_PROP => {
                // `Object.defineProperty` requires an Object target (ECMA-262 step 1):
                // a primitive (incl. `undefined`/`null`) is a TypeError.
                let Some(oraw) = arg(0).as_handle().filter(|_| self.is_object_value(arg(0))) else {
                    let m = self.new_str("Object.defineProperty called on non-object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                };
                // The descriptor must be an object (`ToPropertyDescriptor` of a
                // primitive is a TypeError).
                let Some(draw) = arg(2).as_handle().filter(|_| self.is_object_value(arg(2))) else {
                    let m = self.new_str("Property description must be an object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                };
                let obj = Handle::from_raw(oraw);
                let key = self.coerce_property_key(arg(1))?;
                self.apply_descriptor(obj, &key, Handle::from_raw(draw), false)?;
                arg(0)
            }
            // `Object.defineProperties(obj, { k: descriptor, … })`.
            N_OBJECT_DEFINE_PROPS => {
                // `Object.defineProperties` requires an Object target.
                let Some(oraw) = arg(0).as_handle().filter(|_| self.is_object_value(arg(0))) else {
                    let m = self.new_str("Object.defineProperties called on non-object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                };
                // The Properties argument is ToObject'd (`null`/a primitive throws a
                // TypeError per ECMA-262 step 1 of ObjectDefineProperties).
                let descs =
                    self.require_object_coercible_to_object(arg(1), "Object.defineProperties")?;
                self.apply_property_descriptors(Handle::from_raw(oraw), descs)?;
                arg(0)
            }
            // `Object.is(a, b)` — SameValue: like `===` but `NaN` is equal to
            // itself and `+0`/`-0` differ.
            N_OBJECT_IS => {
                let (a, b) = (arg(0), arg(1));
                let same = match (a.as_number(), b.as_number()) {
                    (Some(x), Some(y)) => {
                        (x == y && (x != 0.0 || x.is_sign_positive() == y.is_sign_positive()))
                            || (x.is_nan() && y.is_nan())
                    }
                    _ => self.realm.strict_equals(a, b),
                };
                NanBox::boolean(same)
            }
            // `Object.hasOwn(obj, key)` — own-property check (incl. array index).
            N_OBJECT_HAS_OWN => {
                // ToObject(O) first (`null`/`undefined` throws), then ToPropertyKey.
                let target = self.require_object_coercible_to_object(arg(0), "Object.hasOwn")?;
                let key = self.coerce_property_key(arg(1))?;
                let owned = Some(target).is_some_and(|h| {
                    self.realm.has_own(h, &key)
                        || self
                            .realm
                            .array_length(h)
                            .is_some_and(|len| key.parse::<usize>().is_ok_and(|i| i < len))
                });
                NanBox::boolean(owned)
            }
            // `Object.groupBy(items, cb)` — groups each item by `cb(item, i)` into
            // an object of arrays keyed by the (stringified) group.
            N_OBJECT_GROUP_BY => {
                let items = self.iterate_values(arg(0))?;
                let cb = arg(1);
                let out = self.realm.new_object();
                for (i, item) in items.iter().enumerate() {
                    let key = self.call(cb, &[*item, NanBox::number(i as f64)])?;
                    let k = self.realm.to_display_string(key);
                    let bucket = match self
                        .realm
                        .get_property(out, &k)
                        .and_then(NanBox::as_handle)
                        .map(Handle::from_raw)
                    {
                        Some(h) => h,
                        None => {
                            let arr = self.realm.new_array(Vec::new());
                            self.realm
                                .set_property(out, &k, NanBox::handle(arr.to_raw()));
                            arr
                        }
                    };
                    self.realm.array_push(bucket, *item);
                }
                NanBox::handle(out.to_raw())
            }
            // --- Reflect.* ---
            N_REFLECT_GET => {
                let h = self.reflect_object_target(arg(0), "get")?;
                let key = self.coerce_property_key(arg(1))?;
                // With an explicit `receiver` (3rd arg), a getter found on the
                // prototype chain runs with `receiver` as its `this` (a data
                // property ignores the receiver — handled by `read_member`).
                if args.len() > 2 {
                    let receiver = arg(2);
                    let mut cur = Some(h);
                    while let Some(c) = cur {
                        if let Some((getter, _)) = self.realm.accessor(c, &key) {
                            if matches!(getter.unpack(), Unpacked::Undefined) {
                                return Ok(NanBox::undefined());
                            }
                            return self.call_with_this(getter, receiver, &[]);
                        }
                        if self.realm.has_own(c, &key) {
                            break;
                        }
                        cur = self.realm.object_proto(c);
                    }
                }
                return self.read_member(h, &key);
            }
            N_REFLECT_SET => {
                {
                    let h = self.reflect_object_target(arg(0), "set")?;
                    let key = self.coerce_property_key(arg(1))?;
                    let value = arg(2);
                    // The receiver defaults to the target; an explicit one (4th arg)
                    // receives the write / is the setter's `this`.
                    let receiver = if args.len() > 3 {
                        arg(3)
                    } else {
                        NanBox::handle(h.to_raw())
                    };
                    // A setter accessor found on the chain runs with `receiver` as
                    // `this` (an accessor with no setter fails).
                    let mut cur = Some(h);
                    while let Some(c) = cur {
                        if let Some((_, setter)) = self.realm.accessor(c, &key) {
                            if matches!(setter.unpack(), Unpacked::Undefined) {
                                return Ok(NanBox::boolean(false));
                            }
                            self.call_with_this(setter, receiver, &[value])?;
                            return Ok(NanBox::boolean(true));
                        }
                        if self.realm.has_own(c, &key) {
                            break;
                        }
                        cur = self.realm.object_proto(c);
                    }
                    // No setter: write the data property on the receiver (via
                    // `assign_member_value`, so array indices/`length` behave right).
                    // A disallowed write (read-only / non-extensible) returns false.
                    let Some(rh) = receiver.as_handle() else {
                        return Ok(NanBox::boolean(false));
                    };
                    let rh = Handle::from_raw(rh);
                    if !self.can_write_property(rh, &key) {
                        return Ok(NanBox::boolean(false));
                    }
                    self.assign_member_value(rh, arg(1), value)?;
                }
                NanBox::boolean(true)
            }
            N_REFLECT_HAS => {
                // Like the `in` operator: own property or anywhere on the
                // prototype chain (array indices bounds-checked).
                let target = self.reflect_object_target(arg(0), "has")?;
                let key = self.coerce_property_key(arg(1))?;
                let mut present = false;
                let mut cur = Some(target);
                while let Some(c) = cur {
                    let here = if let Some(len) = self.realm.array_length(c) {
                        key == "length"
                            || key.parse::<usize>().is_ok_and(|i| i < len)
                            || self.realm.has_own(c, &key)
                    } else {
                        self.realm.has_own(c, &key)
                    };
                    if here {
                        present = true;
                        break;
                    }
                    cur = self.realm.object_proto(c);
                }
                NanBox::boolean(present)
            }
            N_REFLECT_DELETE => {
                // Returns the [[Delete]] result: false for a non-configurable property.
                let target = self.reflect_object_target(arg(0), "deleteProperty")?;
                let key = self.coerce_property_key(arg(1))?;
                let ok = self.realm.delete_property(target, &key);
                NanBox::boolean(ok)
            }
            N_REFLECT_OWN_KEYS => {
                // String keys (integer-indexed then insertion order), then own
                // symbol keys — matching `[[OwnPropertyKeys]]`.
                let h = self.reflect_object_target(arg(0), "ownKeys")?;
                let mut boxed = Vec::new();
                {
                    for k in self.realm.own_property_names(h).unwrap_or_default() {
                        boxed.push(self.new_str(&k));
                    }
                    // All own symbol keys (including non-enumerable ones) come last.
                    for k in self.realm.object_all_keys(h) {
                        if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                            && let Ok(id) = idstr.parse::<u64>()
                            && let Some(sh) = self.realm.symbol_for_id(id)
                        {
                            boxed.push(NanBox::handle(sh.to_raw()));
                        }
                    }
                }
                NanBox::handle(self.realm.new_array(boxed).to_raw())
            }
            // `Reflect.defineProperty(obj, key, desc)` → bool.
            N_REFLECT_DEFINE_PROP => {
                let obj = self.reflect_object_target(arg(0), "defineProperty")?;
                // The key is ToPropertyKey'd (may throw) before the descriptor; the
                // attributes argument must be an Object (ToPropertyDescriptor — a
                // primitive is a TypeError).
                let key = self.coerce_property_key(arg(1))?;
                let Some(desc) = arg(2)
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|_| self.is_object_value(arg(2)))
                else {
                    return Err(self.type_error("Property description must be an object"));
                };
                // Reflect.defineProperty returns the boolean result (false on a failed
                // definition) rather than throwing.
                let done = self.apply_descriptor(obj, &key, desc, true)?;
                NanBox::boolean(done)
            }
            // `Reflect.getOwnPropertyDescriptor(obj, key)`.
            N_REFLECT_GET_OWN_DESC => {
                let obj = self.reflect_object_target(arg(0), "getOwnPropertyDescriptor")?;
                let key = self.coerce_property_key(arg(1))?;
                self.descriptor_of(obj, &key)?
            }
            // `Reflect.getPrototypeOf(obj)` (honors a proxy `getPrototypeOf` trap).
            N_REFLECT_GET_PROTO => {
                let obj = self.reflect_object_target(arg(0), "getPrototypeOf")?;
                self.get_proto_of(obj)?
            }
            // `Reflect.setPrototypeOf(target, proto)` → boolean success.
            N_REFLECT_SET_PROTO => {
                let obj = self.reflect_object_target(arg(0), "setPrototypeOf")?;
                // The proto must be an Object or `null`, else a TypeError.
                if !matches!(arg(1).unpack(), Unpacked::Null) && !self.is_object_value(arg(1)) {
                    return Err(self.type_error("Object prototype may only be an Object or null"));
                }
                let proto = arg(1).as_handle().map(Handle::from_raw);
                // Returns the boolean [[SetPrototypeOf]] result (false on a
                // non-extensible target whose prototype would change).
                NanBox::boolean(self.set_proto_of(obj, proto)?)
            }
            // `Reflect.preventExtensions(target)` → boolean success.
            N_REFLECT_PREVENT_EXT => {
                let obj = self.reflect_object_target(arg(0), "preventExtensions")?;
                self.realm.prevent_extensions(obj);
                NanBox::boolean(true)
            }
            // `Reflect.isExtensible(target)` → boolean (target must be an Object).
            N_REFLECT_IS_EXTENSIBLE => {
                let obj = self.reflect_object_target(arg(0), "isExtensible")?;
                self.is_extensible_of(obj)?
            }
            N_REFLECT_APPLY => {
                let list = match arg(2).as_handle().map(Handle::from_raw) {
                    Some(h) => self
                        .realm
                        .array_elements(h)
                        .map(<[_]>::to_vec)
                        .unwrap_or_default(),
                    None => Vec::new(),
                };
                return self.call_with_this(arg(0), arg(1), &list);
            }
            N_REFLECT_CONSTRUCT => {
                // `target` must be a constructor.
                if !self.is_constructor_value(arg(0)) {
                    return Err(self.type_error("Reflect.construct target is not a constructor"));
                }
                // An explicit `newTarget` (3rd arg), if present, must also be a
                // constructor (`Reflect.construct(t, a, newTarget)`); else `newTarget`
                // defaults to `target`.
                let has_new_target =
                    args.len() > 2 && !matches!(arg(2).unpack(), Unpacked::Undefined);
                if has_new_target && !self.is_constructor_value(arg(2)) {
                    return Err(self.type_error("Reflect.construct newTarget is not a constructor"));
                }
                let list = match arg(1).as_handle().map(Handle::from_raw) {
                    Some(h) => self
                        .realm
                        .array_elements(h)
                        .map(<[_]>::to_vec)
                        .unwrap_or_default(),
                    None => Vec::new(),
                };
                // `target` must be a constructor.
                if !self.is_constructor(arg(0)) {
                    let m = self.new_str("Reflect.construct target is not a constructor");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // An explicit `newTarget` (3rd arg) becomes `new.target` inside the
                // constructor (else it is the target itself); it too must be a
                // constructor.
                if args.len() > 2 && !matches!(arg(2).unpack(), Unpacked::Undefined) {
                    if !self.is_constructor(arg(2)) {
                        let m = self.new_str("Reflect.construct newTarget is not a constructor");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    self.reflect_new_target = Some(arg(2));
                }
                return self.construct(arg(0), &list);
            }
            N_OBJECT_GET_OWN_DESC => {
                // ToObject(O) (a primitive is boxed; `null`/`undefined` throws),
                // then ToPropertyKey(P) — honoring a user `toString`/symbol.
                let obj = self.require_object_coercible_to_object(
                    arg(0),
                    "Object.getOwnPropertyDescriptor",
                )?;
                let key = self.coerce_property_key(arg(1))?;
                self.descriptor_of(obj, &key)?
            }
            // `Object.getOwnPropertyDescriptors(obj)` → a map of all descriptors.
            N_OBJECT_GET_OWN_DESCS => {
                let out = self.realm.new_object();
                if let Some(obj) = arg(0).as_handle().map(Handle::from_raw) {
                    let mut keys = self.realm.own_property_names(obj).unwrap_or_default();
                    keys.extend(self.realm.object_accessor_keys(obj));
                    // Symbol-keyed properties (stored under their `\0sym:` internal name)
                    // get a descriptor too, set under the symbol key on the result.
                    keys.extend(
                        self.realm
                            .object_all_keys(obj)
                            .into_iter()
                            .filter(|k| k.starts_with("\u{0}sym:")),
                    );
                    for k in keys {
                        if let Some(d) = self.build_descriptor(obj, &k) {
                            self.realm.set_property(out, &k, d);
                        }
                    }
                }
                NanBox::handle(out.to_raw())
            }
            N_OBJECT_IS_FROZEN => {
                // A non-object argument (a primitive) is reported as frozen.
                let v = arg(0);
                let frozen = !self.is_object_value(v)
                    || v.as_handle()
                        .is_some_and(|raw| self.realm.is_frozen(Handle::from_raw(raw)));
                NanBox::boolean(frozen)
            }
            N_OBJECT_GET_OWN_NAMES => {
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(
                        self.type_error("Object.getOwnPropertyNames called on null or undefined")
                    );
                }
                let names = arg(0)
                    .as_handle()
                    .and_then(|raw| self.realm.own_property_names(Handle::from_raw(raw)))
                    .unwrap_or_default();
                let boxed: Vec<NanBox> = names.iter().map(|k| self.new_str(k)).collect();
                NanBox::handle(self.realm.new_array(boxed).to_raw())
            }
            // `Object.getOwnPropertySymbols(obj)` — the own symbol-keyed
            // properties (recovered from their `\0sym:{id}` internal names).
            N_OBJECT_GET_OWN_SYMBOLS => {
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(
                        self.type_error("Object.getOwnPropertySymbols called on null or undefined")
                    );
                }
                let mut syms = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = Handle::from_raw(raw);
                    // All own symbol keys, including non-enumerable ones (e.g. a
                    // symbol defined via `Object.defineProperty`).
                    for k in self.realm.object_all_keys(h) {
                        if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                            && let Ok(id) = idstr.parse::<u64>()
                            && let Some(sh) = self.realm.symbol_for_id(id)
                        {
                            syms.push(NanBox::handle(sh.to_raw()));
                        }
                    }
                }
                NanBox::handle(self.realm.new_array(syms).to_raw())
            }
            N_OBJECT_VALUES => {
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Object.values called on null or undefined"));
                }
                // A proxy with an `ownKeys` trap: its enumerable keys, each value
                // read through the proxy (so a `get` trap fires).
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    let ph = Handle::from_raw(raw);
                    let mut vals = Vec::with_capacity(keys.len());
                    for k in keys {
                        vals.push(self.read_member(ph, &k)?);
                    }
                    return Ok(NanBox::handle(self.realm.new_array(vals).to_raw()));
                }
                let mut vals = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = self.proxy_key_target(Handle::from_raw(raw));
                    // Array index values come from element access (ascending) first
                    // — but a VM closure's backing cells are not enumerable values.
                    if !self.realm.is_vm_function(h)
                        && let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec)
                    {
                        vals.extend(elems);
                    }
                    let named = self
                        .realm
                        .object_keys(h)
                        .unwrap_or_else(|| self.realm.aux_named_keys(h));
                    for k in named {
                        vals.push(
                            self.realm
                                .get_property(h, &k)
                                .unwrap_or(NanBox::undefined()),
                        );
                    }
                    // A class constructor's enumerable static fields are already
                    // mirrored as own enumerable aux properties (covered by `named`).
                }
                NanBox::handle(self.realm.new_array(vals).to_raw())
            }
            N_ARRAY_IS_ARRAY => NanBox::boolean(self.is_array_unwrap_proxy(arg(0))?),
            // `ArrayBuffer.isView(x)` — true iff `x` is a typed array or a DataView
            // (anything with a `[[ViewedArrayBuffer]]`).
            N_ARRAY_BUFFER_IS_VIEW => NanBox::boolean(arg(0).as_handle().is_some_and(|raw| {
                let h = Handle::from_raw(raw);
                self.realm.typed_kind(h).is_some()
                    || self.realm.get_property(h, DATA_VIEW_BUF).is_some()
            })),
            N_OBJECT_ASSIGN => {
                // ToObject(target): `null`/`undefined` throws.
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Object.assign target is null or undefined"));
                }
                let target = arg(0);
                if let Some(t) = target.as_handle().map(Handle::from_raw) {
                    for src in &args[1.min(args.len())..] {
                        // A primitive string source contributes its character indices.
                        if let Some(s) = src
                            .as_handle()
                            .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
                            && !self
                                .realm
                                .is_array(Handle::from_raw(src.as_handle().unwrap()))
                        {
                            for (i, ch) in s.chars().enumerate() {
                                let cv = self.new_str(&alloc::format!("{ch}"));
                                let kb = self.new_str(&alloc::format!("{i}"));
                                self.assign_member_value(t, kb, cv)?;
                            }
                            continue;
                        }
                        if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                            // An array (or typed-array view) source contributes its
                            // indexed elements.
                            if let Some(elems) = self.realm.elements_vec(sh) {
                                for (i, e) in elems.iter().enumerate() {
                                    let kb = self.new_str(&alloc::format!("{i}"));
                                    self.assign_member_value(t, kb, *e)?;
                                }
                                continue;
                            }
                            // Own enumerable string *and* symbol keys; values read via
                            // `read_member` (so getters fire) and written via
                            // `assign_member_value` ([[Set]], so the target's setters
                            // run and a frozen/read-only property is honored).
                            let keys = self.realm.object_keys_with_symbols(sh);
                            for k in keys {
                                let v = self.read_member(sh, &k)?;
                                let kb = if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                                    && let Ok(id) = idstr.parse::<u64>()
                                    && let Some(sym) = self.realm.symbol_for_id(id)
                                {
                                    NanBox::handle(sym.to_raw())
                                } else {
                                    self.new_str(&k)
                                };
                                self.assign_member_value(t, kb, v)?;
                            }
                        }
                    }
                }
                target
            }
            N_OBJECT_ENTRIES => {
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Object.entries called on null or undefined"));
                }
                // A proxy with an `ownKeys` trap drives the entry list (values read
                // through the proxy so a `get` trap fires).
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    let ph = Handle::from_raw(raw);
                    let mut pairs = Vec::with_capacity(keys.len());
                    for k in keys {
                        let v = self.read_member(ph, &k)?;
                        let key = self.new_str(&k);
                        pairs.push(NanBox::handle(
                            self.realm.new_array(alloc::vec![key, v]).to_raw(),
                        ));
                    }
                    return Ok(NanBox::handle(self.realm.new_array(pairs).to_raw()));
                }
                let mut entries: Vec<(alloc::string::String, NanBox)> = Vec::new();
                if let Some(h) = arg(0).as_handle().map(Handle::from_raw) {
                    let h = self.proxy_key_target(h);
                    // Array index entries (ascending) before named ones — but a VM
                    // closure's backing cells are not enumerable entries.
                    if !self.realm.is_vm_function(h)
                        && let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec)
                    {
                        for (i, v) in elems.into_iter().enumerate() {
                            entries.push((alloc::format!("{i}"), v));
                        }
                    }
                    let named = self
                        .realm
                        .object_keys(h)
                        .unwrap_or_else(|| self.realm.aux_named_keys(h));
                    for k in named {
                        let v = self
                            .realm
                            .get_property(h, &k)
                            .unwrap_or(NanBox::undefined());
                        entries.push((k, v));
                    }
                    // A class constructor's enumerable static fields are already
                    // mirrored as own enumerable aux properties (covered by `named`).
                }
                let pairs: Vec<NanBox> = entries
                    .into_iter()
                    .map(|(k, v)| {
                        let key = self.new_str(&k);
                        NanBox::handle(self.realm.new_array(alloc::vec![key, v]).to_raw())
                    })
                    .collect();
                NanBox::handle(self.realm.new_array(pairs).to_raw())
            }
            N_ARRAY_FROM => {
                // Iterable → array (arrays, strings, Sets, Maps), with an
                // optional map callback applied to each element. A non-iterable
                // array-like (an object with a `length`) is read by index.
                let items = match self.iterate_values(arg(0)) {
                    Ok(v) => v,
                    Err(_) => {
                        let mut out = Vec::new();
                        if let Some(h) = arg(0).as_handle().map(Handle::from_raw) {
                            let len_raw = self
                                .realm
                                .get_property(h, "length")
                                .map(|v| self.realm.to_number(v))
                                .unwrap_or(0.0);
                            // Cap the array-like length against `max_array_len`
                            // BEFORE allocating, so `from({length: 2**32-1})` throws
                            // a catchable RangeError instead of attempting a
                            // multi-gigabyte allocation (an unbounded-memory bug).
                            if len_raw > self.realm.limits.max_array_len as f64 {
                                let m = self.new_str("Invalid array length");
                                return Err(ExecError::Throw(
                                    self.make_error(N_RANGE_ERROR, Some(m)),
                                ));
                            }
                            let len = len_raw.max(0.0) as usize;
                            for i in 0..len {
                                let k = alloc::format!("{i}");
                                out.push(
                                    self.realm
                                        .get_property(h, &k)
                                        .unwrap_or(NanBox::undefined()),
                                );
                            }
                        }
                        out
                    }
                };
                let items = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                    items
                } else {
                    let f = arg(1);
                    let this_arg = arg(2); // `Array.from(items, mapFn, thisArg)`
                    let mut out = Vec::with_capacity(items.len());
                    for (i, e) in items.iter().enumerate() {
                        out.push(self.call_with_this(
                            f,
                            this_arg,
                            &[*e, NanBox::number(i as f64)],
                        )?);
                    }
                    out
                };
                NanBox::handle(self.realm.new_array(items).to_raw())
            }
            N_ARRAY_OF => NanBox::handle(self.realm.new_array(args.to_vec()).to_raw()),
            // `%IteratorPrototype%[Symbol.iterator]()` — an iterator is its own
            // iterable: return the receiver.
            N_ITERATOR_PROTO_SELF => self.this_val,
            // The abstract `%Iterator%` constructor is not callable as a plain
            // function (and `new Iterator()` is a TypeError too — handled in
            // `construct`).
            N_ITERATOR => {
                let m = self.new_str("Abstract class Iterator not directly constructable");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // `Iterator.from(obj)` — wrap an iterable (or an iterator) so the
            // ES2025 iterator-helper methods (`map`/`filter`/`take`/…) are
            // available. An object that is already an iterator (has a `next`
            // method) is driven through its protocol; any other iterable is
            // gathered via the standard iteration. The result is a generator-
            // backed iterator, which carries the helper methods and `next()`.
            N_ITERATOR_FROM => {
                let src = arg(0);
                // RequireObjectCoercible-ish: a null/undefined source is a TypeError.
                if matches!(src.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Iterator.from called on null or undefined");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let values = self.iterate_values(src)?;
                self.make_generator(values)
            }
            // The abstract `%TypedArray%` intrinsic is not callable directly.
            N_TYPED_ARRAY_ABSTRACT => {
                let m = self.new_str("Abstract class TypedArray not directly constructable");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // `%TypedArray%.from(source, mapFn?, thisArg?)` — generic over the
            // `this` constructor (`Int8Array.from(...)` builds an `Int8Array`).
            N_TYPED_ARRAY_FROM => {
                let ctor = self.this_val;
                // Step 2: IsConstructor(C) — `%TypedArray%.from` called with a `this`
                // that is not a constructor throws a TypeError.
                if !self.is_constructor_value(ctor) {
                    return Err(self.type_error("TypedArray.from requires a constructor this"));
                }
                // Step 3: if `mapfn` is not undefined and not callable, throw a
                // TypeError — *before* accessing `source[@@iterator]` / `length`.
                let mapfn = arg(1);
                let has_mapfn = !matches!(mapfn.unpack(), Unpacked::Undefined);
                if has_mapfn
                    && !mapfn
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("TypedArray.from mapfn is not a function"));
                }
                let items = match self.iterate_values(arg(0)) {
                    Ok(v) => v,
                    Err(_) => {
                        let mut out = Vec::new();
                        if let Some(h) = arg(0).as_handle().map(Handle::from_raw) {
                            let len_raw = self
                                .realm
                                .get_property(h, "length")
                                .map(|v| self.realm.to_number(v))
                                .unwrap_or(0.0);
                            // Cap the array-like length against `max_array_len`
                            // BEFORE allocating, so `from({length: 2**32-1})` throws
                            // a catchable RangeError instead of attempting a
                            // multi-gigabyte allocation (an unbounded-memory bug).
                            if len_raw > self.realm.limits.max_array_len as f64 {
                                let m = self.new_str("Invalid array length");
                                return Err(ExecError::Throw(
                                    self.make_error(N_RANGE_ERROR, Some(m)),
                                ));
                            }
                            let len = len_raw.max(0.0) as usize;
                            for i in 0..len {
                                let k = alloc::format!("{i}");
                                out.push(
                                    self.realm
                                        .get_property(h, &k)
                                        .unwrap_or(NanBox::undefined()),
                                );
                            }
                        }
                        out
                    }
                };
                let items = if !has_mapfn {
                    items
                } else {
                    let f = mapfn;
                    let this_arg = arg(2);
                    let mut out = Vec::with_capacity(items.len());
                    for (i, e) in items.iter().enumerate() {
                        out.push(self.call_with_this(
                            f,
                            this_arg,
                            &[*e, NanBox::number(i as f64)],
                        )?);
                    }
                    out
                };
                let view = self.construct(ctor, &[NanBox::number(items.len() as f64)])?;
                if let Some(vh) = view.as_handle().map(Handle::from_raw) {
                    self.realm.typed_set_from_numbers(vh, 0, &items);
                }
                view
            }
            // `%TypedArray%.of(...items)` — generic over the `this` constructor.
            N_TYPED_ARRAY_OF => {
                let ctor = self.this_val;
                let view = self.construct(ctor, &[NanBox::number(args.len() as f64)])?;
                if let Some(vh) = view.as_handle().map(Handle::from_raw) {
                    self.realm.typed_set_from_numbers(vh, 0, args);
                }
                view
            }
            // `get %TypedArray%[Symbol.species]` — returns the receiver constructor.
            N_TYPED_ARRAY_SPECIES => self.this_val,
            // `get %TypedArray%.prototype[Symbol.toStringTag]` — the concrete view
            // name (e.g. "Uint8Array") when `this` is a typed array, else
            // `undefined` (no exception per spec).
            N_TYPED_ARRAY_TO_STRING_TAG => match self
                .this_val
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.typed_kind(h))
            {
                Some(kind) => self.new_str(TYPED_ARRAY_KINDS[kind as usize].0),
                None => NanBox::undefined(),
            },
            N_OBJECT_FROM_ENTRIES => {
                // RequireObjectCoercible + GetIterator: `null`/`undefined`/a
                // non-iterable throws a TypeError (propagated, not swallowed).
                let obj = self.realm.new_object();
                // Accepts any iterable of `[key, value]` pairs (arrays, a Map, …).
                let pairs = self.iterate_values(arg(0))?;
                for pair in pairs {
                    if let Some(kv) = pair
                        .as_handle()
                        .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                        .map(<[_]>::to_vec)
                    {
                        let k = self
                            .realm
                            .to_display_string(kv.first().copied().unwrap_or(NanBox::undefined()));
                        let v = kv.get(1).copied().unwrap_or(NanBox::undefined());
                        self.realm.set_property(obj, &k, v);
                    }
                }
                NanBox::handle(obj.to_raw())
            }
            #[cfg(feature = "std")]
            N_MATH_FLOOR => NanBox::number(self.realm.to_number(arg(0)).floor()),
            #[cfg(feature = "std")]
            N_MATH_CEIL => NanBox::number(self.realm.to_number(arg(0)).ceil()),
            #[cfg(feature = "std")]
            // JS `Math.round` rounds half toward +Infinity (`floor(x + 0.5)`),
            // unlike Rust's round-half-away-from-zero.
            N_MATH_ROUND => {
                let n = self.realm.to_number(arg(0));
                // A magnitude ≥ 2^52 is already an integer (the f64 spacing is ≥ 1),
                // so it is returned unchanged — adding 0.5 and flooring would lose
                // precision (`Math.round(2^53 − 1)` must be `2^53 − 1`, not `2^53`).
                if n.abs() >= 4_503_599_627_370_496.0 || !n.is_finite() {
                    NanBox::number(n)
                } else {
                    NanBox::number(crate::common::js_round(n))
                }
            }
            #[cfg(feature = "std")]
            N_MATH_SQRT => NanBox::number(self.realm.to_number(arg(0)).sqrt()),
            #[cfg(not(feature = "std"))]
            N_MATH_FLOOR | N_MATH_CEIL | N_MATH_ROUND | N_MATH_SQRT => {
                return Err(ExecError::Unsupported("Math float ops need std"));
            }
            #[cfg(feature = "std")]
            N_MATH_POW => self.realm.pow(arg(0), arg(1)),
            #[cfg(not(feature = "std"))]
            N_MATH_POW => return Err(ExecError::Unsupported("Math.pow needs std")),
            N_MATH_SIGN => {
                let n = self.realm.to_number(arg(0));
                NanBox::number(if n.is_nan() {
                    f64::NAN
                } else if n > 0.0 {
                    1.0
                } else if n < 0.0 {
                    -1.0
                } else {
                    n // ±0
                })
            }
            #[cfg(feature = "std")]
            N_MATH_TRUNC => NanBox::number(self.realm.to_number(arg(0)).trunc()),
            #[cfg(not(feature = "std"))]
            N_MATH_TRUNC => return Err(ExecError::Unsupported("Math.trunc needs std")),
            #[cfg(feature = "std")]
            N_MATH_HYPOT => {
                // If any argument is ±Infinity the result is +Infinity, even when
                // another argument is NaN (NaN only wins if no argument is infinite).
                // ToNumber every argument first (propagating any abrupt completion).
                let mut nums = Vec::with_capacity(args.len());
                for a in args {
                    let num = self.coerce_to_number(*a)?;
                    nums.push(self.realm.to_number(num));
                }
                let mut any_inf = false;
                let mut any_nan = false;
                let mut sum = 0.0;
                for n in nums {
                    if n.is_infinite() {
                        any_inf = true;
                    } else if n.is_nan() {
                        any_nan = true;
                    } else {
                        sum += n * n;
                    }
                }
                NanBox::number(if any_inf {
                    f64::INFINITY
                } else if any_nan {
                    f64::NAN
                } else {
                    sum.sqrt()
                })
            }
            #[cfg(feature = "std")]
            N_MATH_CBRT => NanBox::number(self.realm.to_number(arg(0)).cbrt()),
            #[cfg(feature = "std")]
            N_MATH_LOG2 => NanBox::number(self.realm.to_number(arg(0)).log2()),
            #[cfg(feature = "std")]
            N_MATH_LOG10 => NanBox::number(self.realm.to_number(arg(0)).log10()),
            #[cfg(feature = "std")]
            N_MATH_EXP => NanBox::number(self.realm.to_number(arg(0)).exp()),
            #[cfg(feature = "std")]
            N_MATH_LOG => NanBox::number(self.realm.to_number(arg(0)).ln()),
            // `Math.random()` ∈ [0, 1) — Vigna's xorshift128+ (period 2^128-1),
            // the generator family used by V8/SpiderMonkey: a strict upgrade over
            // xorshift64 in period and statistical quality. The output is the sum
            // of the two state words (the "+"); its top 53 bits form the mantissa.
            N_MATH_RANDOM => {
                let mut s1 = self.rng_state[0];
                let s0 = self.rng_state[1];
                let result = s0.wrapping_add(s1);
                self.rng_state[0] = s0;
                s1 ^= s1 << 23;
                self.rng_state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
                NanBox::number((result >> 11) as f64 / (1u64 << 53) as f64)
            }
            // Trig / hyperbolic / inverse — single-argument f64 functions.
            #[cfg(feature = "std")]
            N_MATH_SIN..=N_MATH_LOG1P => {
                let n = self.realm.to_number(arg(0));
                let r = match id {
                    N_MATH_SIN => n.sin(),
                    N_MATH_COS => n.cos(),
                    N_MATH_TAN => n.tan(),
                    N_MATH_ASIN => n.asin(),
                    N_MATH_ACOS => n.acos(),
                    N_MATH_ATAN => n.atan(),
                    N_MATH_ATAN2 => n.atan2(self.realm.to_number(arg(1))),
                    N_MATH_SINH => n.sinh(),
                    N_MATH_COSH => n.cosh(),
                    N_MATH_TANH => n.tanh(),
                    N_MATH_ASINH => n.asinh(),
                    N_MATH_ACOSH => n.acosh(),
                    N_MATH_ATANH => n.atanh(),
                    N_MATH_EXPM1 => n.exp_m1(),
                    _ => n.ln_1p(), // N_MATH_LOG1P
                };
                NanBox::number(r)
            }
            #[cfg(not(feature = "std"))]
            N_MATH_SIN..=N_MATH_LOG1P => {
                return Err(ExecError::Unsupported("Math fns need std"));
            }
            // `Math.fround(x)` — round to the nearest single-precision float.
            N_MATH_FROUND => NanBox::number(self.realm.to_number(arg(0)) as f32 as f64),
            // `Math.f16round(x)` — round to the nearest IEEE-754 binary16 value and
            // back to a double (no stable Rust `f16`, so the conversion is explicit).
            #[cfg(feature = "std")]
            N_MATH_F16ROUND => {
                NanBox::number(f16_to_f64(f64_to_f16_bits(self.realm.to_number(arg(0)))))
            }
            #[cfg(not(feature = "std"))]
            N_MATH_F16ROUND => return Err(ExecError::Unsupported("Math.f16round needs std")),
            // `Math.clz32(x)` — count leading zeros of the ToUint32 value. ToUint32
            // maps a non-finite/NaN value to 0 (so `clz32(Infinity)` is 32). The
            // `as i64 as u32` cast performs trunc-toward-zero mod 2^32 without the
            // `f64::trunc` intrinsic (so this stays available in `no_std`).
            N_MATH_CLZ32 => {
                let n = self.realm.to_number(arg(0));
                let u = if n.is_finite() { n as i64 as u32 } else { 0 };
                NanBox::number(u.leading_zeros() as f64)
            }
            // `Math.imul(a, b)` — 32-bit integer multiplication.
            N_MATH_IMUL => {
                let an = self.realm.to_number(arg(0));
                let bn = self.realm.to_number(arg(1));
                let a = if an.is_finite() { an as i64 as i32 } else { 0 };
                let b = if bn.is_finite() { bn as i64 as i32 } else { 0 };
                NanBox::number(a.wrapping_mul(b) as f64)
            }
            #[cfg(not(feature = "std"))]
            N_MATH_HYPOT | N_MATH_CBRT | N_MATH_LOG2 | N_MATH_LOG10 | N_MATH_EXP | N_MATH_LOG => {
                return Err(ExecError::Unsupported("Math fns need std"));
            }
            N_PARSE_FLOAT => {
                let s = self.realm.to_display_string(arg(0));
                NanBox::number(parse_float_prefix(s.trim()))
            }
            // URI encoding/decoding. `encodeURI` preserves the URI reserved set
            // on top of the unreserved set that `encodeURIComponent` keeps.
            N_ENCODE_URI_COMPONENT | N_ENCODE_URI => {
                let s = self.realm.to_display_string(arg(0));
                let extra = if id == N_ENCODE_URI {
                    ";,/?:@&=+$#"
                } else {
                    ""
                };
                let out = uri_encode(&s, extra);
                let h = self.realm.new_string(&out);
                NanBox::handle(h.to_raw())
            }
            N_DECODE_URI_COMPONENT | N_DECODE_URI => {
                let s = self.realm.to_display_string(arg(0));
                match uri_decode(&s) {
                    Some(out) => {
                        let h = self.realm.new_string(&out);
                        NanBox::handle(h.to_raw())
                    }
                    None => {
                        let m = self.new_str("URI malformed");
                        return Err(ExecError::Throw(self.make_error(N_URI_ERROR, Some(m))));
                    }
                }
            }
            N_STRUCTURED_CLONE => {
                let mut seen: Vec<(u64, NanBox)> = Vec::new();
                self.structured_clone(arg(0), &mut seen)?
            }
            // `Intl.NumberFormat(...)` / `Intl.DateTimeFormat(...)` called without
            // `new` build the same formatter object.
            N_INTL_NUMBER_FORMAT | N_INTL_DATETIME_FORMAT => self.make_intl_formatter(id, args),
            // `Intl.Collator(...)` / `Intl.PluralRules(...)` without `new`.
            N_INTL_COLLATOR => self.make_collator(args),
            N_INTL_PLURAL_RULES => self.make_plural_rules(args),
            // `new Intl.ListFormat(locale, { type, style })` — an object with `.format`.
            N_INTL_LIST_FORMAT => self.make_list_format(args),
            // `Intl.ListFormat.prototype.format(list)` — joins with en-US conjunction /
            // disjunction / unit patterns (Oxford comma for 3+ items).
            N_INTL_LIST_FORMAT_FORMAT => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let list_type = fmt
                    .and_then(|h| self.realm.get_property(h, "type"))
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_else(|| String::from("conjunction"));
                let items: Vec<String> = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                    .unwrap_or_default()
                    .iter()
                    .map(|e| self.realm.to_display_string(*e))
                    .collect();
                // The crate handles conjunction/disjunction with locale-aware connectors;
                // `type:"unit"` (no crate `ListStyle`) falls through to the hand-rolled join.
                #[cfg(feature = "intl")]
                {
                    let style = match list_type.as_str() {
                        "disjunction" => Some(intl::list::ListStyle::Or),
                        "conjunction" => Some(intl::list::ListStyle::And),
                        _ => None,
                    };
                    if let Some(style) = style {
                        let locale = fmt
                            .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                            .map(|v| self.realm.to_display_string(v))
                            .unwrap_or_else(|| String::from("en"));
                        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
                        return Ok(self.new_str(&intl::list::format_list(&locale, &refs, style)));
                    }
                }
                let word = match list_type.as_str() {
                    "disjunction" => "or",
                    "unit" => "",
                    _ => "and",
                };
                let out = match items.len() {
                    0 => String::new(),
                    1 => items[0].clone(),
                    2 if word.is_empty() => alloc::format!("{}, {}", items[0], items[1]),
                    2 => alloc::format!("{} {word} {}", items[0], items[1]),
                    n => {
                        let init = items[..n - 1].join(", ");
                        let last = &items[n - 1];
                        if word.is_empty() {
                            alloc::format!("{init}, {last}")
                        } else {
                            alloc::format!("{init}, {word} {last}")
                        }
                    }
                };
                self.new_str(&out)
            }
            // `Intl.RelativeTimeFormat(...)` without `new`.
            N_INTL_REL_TIME => self.make_relative_time_format(args),
            // `Intl.RelativeTimeFormat.prototype.format(value, unit)`.
            N_INTL_REL_TIME_FORMAT => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let numeric = fmt
                    .and_then(|h| self.realm.get_property(h, "numeric"))
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_else(|| String::from("always"));
                let value = self.realm.to_number(arg(0));
                let unit = self.realm.to_display_string(arg(1));
                let s = relative_time_string(value, &unit, &numeric);
                self.new_str(&s)
            }
            // `Intl.DisplayNames(...)` without `new`.
            N_INTL_DISPLAY_NAMES => self.make_display_names(args),
            // `Intl.DisplayNames.prototype.of(code)`.
            N_INTL_DISPLAY_NAMES_OF => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let ty = fmt
                    .and_then(|h| self.realm.get_property(h, "type"))
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                let code = self.realm.to_display_string(arg(0));
                #[cfg(feature = "intl")]
                {
                    let locale = fmt
                        .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                        .map(|v| self.realm.to_display_string(v))
                        .unwrap_or_else(|| String::from("en"));
                    // The crate has CLDR language/region names; currency/script use the
                    // hand-rolled table (and any code the crate doesn't know falls back).
                    let primary = code.split(['-', '_']).next().unwrap_or(&code);
                    let crate_name = match ty.as_str() {
                        "language" => intl::display::language_name(&locale, primary),
                        "region" => intl::display::region_name(&locale, &code),
                        _ => None,
                    };
                    match crate_name {
                        Some(n) => self.new_str(n),
                        None => {
                            let s = display_name(&ty, &code);
                            self.new_str(&s)
                        }
                    }
                }
                #[cfg(not(feature = "intl"))]
                {
                    let s = display_name(&ty, &code);
                    self.new_str(&s)
                }
            }
            // `Intl.Segmenter(...)` without `new`.
            N_INTL_SEGMENTER => self.make_segmenter(args),
            // `Intl.Segmenter.prototype.segment(input)` → an (iterable) array of segment
            // data objects `{ segment, index, input, isWordLike? }`.
            N_INTL_SEGMENTER_SEGMENT => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let gran = fmt
                    .and_then(|h| self.realm.get_property(h, "granularity"))
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_else(|| String::from("grapheme"));
                let input = self.realm.to_display_string(arg(0));
                let segs = segment_text(&input, &gran);
                let mut elems = Vec::with_capacity(segs.len());
                for (index, seg, is_word_like) in segs {
                    let o = self.realm.new_object();
                    let sv = self.new_str(&seg);
                    self.realm.set_property(o, "segment", sv);
                    self.realm
                        .set_property(o, "index", NanBox::number(index as f64));
                    let iv = self.new_str(&input);
                    self.realm.set_property(o, "input", iv);
                    if let Some(w) = is_word_like {
                        self.realm.set_property(o, "isWordLike", NanBox::boolean(w));
                    }
                    elems.push(NanBox::handle(o.to_raw()));
                }
                NanBox::handle(self.realm.new_array(elems).to_raw())
            }
            // `Intl.Collator.prototype.compare(a, b)` — code-point order (no locale
            // tailoring), so a negative/zero/positive result orders `a` vs `b`.
            N_INTL_COMPARE => {
                let a = self.realm.to_display_string(arg(0));
                let b = self.realm.to_display_string(arg(1));
                // With the `intl` crate, real UCA collation honoring `sensitivity`
                // (→ strength) and `numeric`; otherwise code-point order.
                #[cfg(feature = "intl")]
                let ord = {
                    use intl::unicode::collate::{AlternateHandling, Collator, Strength};
                    let fmt = self.this_val.as_handle().map(Handle::from_raw);
                    let strength = match fmt
                        .and_then(|h| self.realm.get_property(h, "sensitivity"))
                        .map(|v| self.realm.to_display_string(v))
                        .as_deref()
                    {
                        Some("base") => Strength::Primary,
                        Some("accent") => Strength::Secondary,
                        _ => Strength::Tertiary,
                    };
                    let numeric = matches!(
                        fmt.and_then(|h| self.realm.get_property(h, "numeric"))
                            .map(|v| v.unpack()),
                        Some(Unpacked::Bool(true))
                    );
                    Collator::new(AlternateHandling::Shifted)
                        .with_strength(strength)
                        .with_numeric(numeric)
                        .compare(&a, &b)
                };
                #[cfg(not(feature = "intl"))]
                let ord = a.cmp(&b);
                NanBox::number(match ord {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Equal => 0.0,
                    core::cmp::Ordering::Greater => 1.0,
                })
            }
            // `Intl.PluralRules.prototype.select(n)` — the English plural category:
            // `1` is "one", everything else "other".
            N_INTL_PLURAL_SELECT => {
                let n = self.realm.to_number(arg(0));
                #[cfg(feature = "intl")]
                {
                    let fmt = self.this_val.as_handle().map(Handle::from_raw);
                    let locale = fmt
                        .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                        .map(|v| self.realm.to_display_string(v))
                        .unwrap_or_else(|| String::from("en"));
                    let ordinal = fmt
                        .and_then(|h| self.realm.get_property(h, "type"))
                        .map(|v| self.realm.to_display_string(v))
                        .as_deref()
                        == Some("ordinal");
                    let ops = if n == (n as i64) as f64 {
                        intl::plural::PluralOperands::from_int(n as i64)
                    } else {
                        intl::plural::PluralOperands::parse(&alloc::format!("{n}"))
                            .unwrap_or_else(|| intl::plural::PluralOperands::from_int(n as i64))
                    };
                    let cat = if ordinal {
                        intl::plural::ordinal_category(&locale, &ops)
                    } else {
                        intl::plural::plural_category(&locale, &ops)
                    };
                    use intl::plural::PluralCategory::*;
                    let s = match cat {
                        Zero => "zero",
                        One => "one",
                        Two => "two",
                        Few => "few",
                        Many => "many",
                        Other => "other",
                    };
                    self.new_str(s)
                }
                #[cfg(not(feature = "intl"))]
                {
                    let cat = if n == 1.0 { "one" } else { "other" };
                    self.new_str(cat)
                }
            }
            // `nf.format(x)` read as a value then called: format against the `this`
            // formatter (a detached call with no formatter falls back to ToString).
            N_INTL_FORMAT => {
                if let Some(h) = self.this_val.as_handle().map(Handle::from_raw)
                    && self.realm.get_property(h, "\u{0}intl").is_some()
                {
                    let s = self.intl_format_value(h, arg(0));
                    self.new_str(&s)
                } else {
                    let s = self.realm.to_display_string(arg(0));
                    self.new_str(&s)
                }
            }
            // `nf.resolvedOptions()` — the resolved configuration of the formatter.
            N_INTL_RESOLVED_OPTIONS => {
                let out = self.realm.new_object();
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let kind = fmt
                    .and_then(|h| self.realm.get_property(h, "\u{0}intl"))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_else(|| String::from("number"));
                let getp = |this: &Self, key: &str| -> Option<NanBox> {
                    fmt.and_then(|h| this.realm.get_property(h, key))
                        .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                };
                let locale = getp(self, "\u{0}locale")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_else(|| String::from("en-US"));
                let lv = self.new_str(&locale);
                self.realm.set_property(out, "locale", lv);
                let ns = self.new_str("latn");
                self.realm.set_property(out, "numberingSystem", ns);
                if kind == "number" {
                    let style = getp(self, "style")
                        .map(|v| self.realm.to_display_string(v))
                        .unwrap_or_else(|| String::from("decimal"));
                    let sv = self.new_str(&style);
                    self.realm.set_property(out, "style", sv);
                    if style == "currency"
                        && let Some(c) = getp(self, "currency")
                    {
                        self.realm.set_property(out, "currency", c);
                    }
                    let (def_min, def_max): (f64, f64) = match style.as_str() {
                        "currency" => (2.0, 2.0),
                        "percent" => (0.0, 0.0),
                        _ => (0.0, 3.0),
                    };
                    let min = getp(self, "minimumFractionDigits")
                        .map_or(def_min, |v| self.realm.to_number(v));
                    let max = getp(self, "maximumFractionDigits")
                        .map_or(def_max.max(min), |v| self.realm.to_number(v));
                    self.realm
                        .set_property(out, "minimumIntegerDigits", NanBox::number(1.0));
                    self.realm
                        .set_property(out, "minimumFractionDigits", NanBox::number(min));
                    self.realm
                        .set_property(out, "maximumFractionDigits", NanBox::number(max));
                    let ug = getp(self, "useGrouping").unwrap_or(NanBox::boolean(true));
                    self.realm.set_property(out, "useGrouping", ug);
                    // `signDisplay` ("auto" | "always" | "never" | "exceptZero").
                    let sd = getp(self, "signDisplay")
                        .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                        .unwrap_or_else(|| self.new_str("auto"));
                    self.realm.set_property(out, "signDisplay", sd);
                } else {
                    let cal = self.new_str("gregory");
                    self.realm.set_property(out, "calendar", cal);
                    let tz = self.new_str("UTC");
                    self.realm.set_property(out, "timeZone", tz);
                }
                NanBox::handle(out.to_raw())
            }
            // `Intl.X.supportedLocalesOf(locales)` — the requested locales this engine
            // can serve. With no real locale data, every requested locale is accepted;
            // the result is a fresh array of the (string) requests.
            N_INTL_SUPPORTED_LOCALES => {
                let mut out = Vec::new();
                let req = arg(0);
                if let Some(rh) = req.as_handle().map(Handle::from_raw) {
                    if let Some(elems) = self.realm.array_elements(rh).map(<[_]>::to_vec) {
                        for e in elems {
                            if e.as_handle().is_some_and(|r| {
                                self.realm.string_value(Handle::from_raw(r)).is_some()
                            }) {
                                out.push(e);
                            }
                        }
                    } else if self.realm.string_value(rh).is_some() {
                        out.push(req); // a single locale string
                    }
                }
                NanBox::handle(self.realm.new_array(out).to_raw())
            }
            // `nf.formatToParts(x)` — the formatted number split into `{type, value}`
            // parts (minusSign/currency/integer/group/decimal/fraction/percent, plus
            // nan/infinity). en-US-ish; mirrors the `format` output's structure.
            N_INTL_FORMAT_TO_PARTS => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                // A DateTimeFormat breaks into typed date/time parts (weekday/month/day/year/
                // hour/minute/second/dayPeriod/era with `literal` separators).
                if let Some(h) = fmt
                    && self
                        .realm
                        .get_property(h, "\u{0}intl")
                        .map(|v| self.realm.to_display_string(v))
                        .as_deref()
                        == Some("datetime")
                {
                    let ms = match arg(0).as_handle().map(Handle::from_raw) {
                        Some(dh) if self.realm.date_at(dh).is_some() => {
                            self.realm.date_at(dh).unwrap()
                        }
                        _ => self.realm.to_number(arg(0)),
                    };
                    let parts = self.datetime_parts(h, ms);
                    let mut arr_elems = Vec::with_capacity(parts.len());
                    for (ty, val) in parts {
                        let o = self.realm.new_object();
                        let tv = self.new_str(ty);
                        self.realm.set_property(o, "type", tv);
                        let vv = self.new_str(&val);
                        self.realm.set_property(o, "value", vv);
                        arr_elems.push(NanBox::handle(o.to_raw()));
                    }
                    return Ok(NanBox::handle(self.realm.new_array(arr_elems).to_raw()));
                }
                // Number formatters use the `intl` crate's typed parts (CLDR, locale-aware).
                #[cfg(feature = "intl")]
                if let Some(h) = fmt
                    && self
                        .realm
                        .get_property(h, "\u{0}intl")
                        .map(|v| self.realm.to_display_string(v))
                        .as_deref()
                        == Some("number")
                    && !self.number_uses_handrolled(h)
                {
                    let n = self.realm.to_number(arg(0));
                    let locale = self
                        .realm
                        .get_property(h, "\u{0}locale")
                        .map(|v| self.realm.to_display_string(v))
                        .unwrap_or_else(|| String::from("en"));
                    let opts = self.number_format_options(h);
                    let parts = intl::number::format_to_parts(&locale, n, &opts);
                    let mut arr_elems = Vec::with_capacity(parts.len());
                    for p in parts {
                        let o = self.realm.new_object();
                        let tv = self.new_str(p.kind.as_str());
                        self.realm.set_property(o, "type", tv);
                        let vv = self.new_str(&p.value);
                        self.realm.set_property(o, "value", vv);
                        arr_elems.push(NanBox::handle(o.to_raw()));
                    }
                    return Ok(NanBox::handle(self.realm.new_array(arr_elems).to_raw()));
                }
                let formatted = match fmt {
                    Some(h) if self.realm.get_property(h, "\u{0}intl").is_some() => {
                        self.intl_format_value(h, arg(0))
                    }
                    _ => self.realm.to_display_string(arg(0)),
                };
                let style = fmt
                    .and_then(|h| self.realm.get_property(h, "style"))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_else(|| String::from("decimal"));
                let currency_sym = if style == "currency" {
                    let code = fmt
                        .and_then(|h| self.realm.get_property(h, "currency"))
                        .map(|v| self.realm.to_display_string(v))
                        .unwrap_or_default();
                    currency_symbol(&code)
                } else {
                    String::new()
                };
                // Build (type, value) entries from the formatted string's structure.
                let mut entries: Vec<(&'static str, String)> = Vec::new();
                let mut s = formatted.as_str();
                if let Some(rest) = s.strip_prefix('-') {
                    entries.push(("minusSign", String::from("-")));
                    s = rest;
                }
                if !currency_sym.is_empty() && s.starts_with(currency_sym.as_str()) {
                    entries.push(("currency", currency_sym.clone()));
                    s = &s[currency_sym.len()..];
                }
                // A trailing percent sign is stripped first, so the core (∞/NaN/digits)
                // is classified correctly; the percent part is appended at the end.
                let mut percent = false;
                if style == "percent" && s.ends_with('%') {
                    percent = true;
                    s = &s[..s.len() - '%'.len_utf8()];
                }
                if s == "NaN" {
                    entries.push(("nan", String::from("NaN")));
                } else if s == "∞" {
                    entries.push(("infinity", String::from("∞")));
                } else {
                    let (int_part, frac_part) = match s.split_once('.') {
                        Some((i, f)) => (i, Some(f)),
                        None => (s, None),
                    };
                    for (gi, grp) in int_part.split(',').enumerate() {
                        if gi > 0 {
                            entries.push(("group", String::from(",")));
                        }
                        entries.push(("integer", String::from(grp)));
                    }
                    if let Some(f) = frac_part {
                        entries.push(("decimal", String::from(".")));
                        entries.push(("fraction", String::from(f)));
                    }
                }
                if percent {
                    entries.push(("percentSign", String::from("%")));
                }
                let mut arr_elems = Vec::with_capacity(entries.len());
                for (ty, val) in entries {
                    let o = self.realm.new_object();
                    let tv = self.new_str(ty);
                    self.realm.set_property(o, "type", tv);
                    let vv = self.new_str(&val);
                    self.realm.set_property(o, "value", vv);
                    arr_elems.push(NanBox::handle(o.to_raw()));
                }
                NanBox::handle(self.realm.new_array(arr_elems).to_raw())
            }
            // `setTimeout(cb, delay?, ...args)` — queues `cb(...args)` as a macrotask
            // and returns a numeric timer id (usable with `clearTimeout`).
            N_SET_TIMEOUT => {
                let callback = arg(0);
                let delay = self.realm.to_number(arg(1)).max(0.0);
                let extra: Vec<NanBox> = args.iter().skip(2).copied().collect();
                let id = self.timer_next_id;
                self.timer_next_id += 1;
                let seq = self.timer_seq;
                self.timer_seq += 1;
                self.macrotasks.push(Timer {
                    id,
                    delay: if delay.is_finite() { delay } else { 0.0 },
                    seq,
                    callback,
                    args: extra,
                });
                NanBox::number(id as f64)
            }
            // `clearTimeout(id)` — cancels a pending `setTimeout`.
            N_CLEAR_TIMEOUT => {
                if let Some(id) = arg(0).as_number() {
                    self.macrotasks.retain(|t| (t.id as f64) != id);
                }
                NanBox::undefined()
            }
            // `queueMicrotask(cb)` — schedules `cb()` on the microtask queue.
            N_QUEUE_MICROTASK => {
                let callback = arg(0);
                let result = self.fresh_promise();
                self.microtasks.push(Job {
                    handler: callback,
                    value: NanBox::undefined(),
                    result,
                    fulfilled: true,
                    finally: false,
                });
                NanBox::undefined()
            }
            // `WebAssembly.validate(bytes)` — true iff `bytes` decodes to a
            // well-formed module. Accepts an `ArrayBuffer` or a byte array.
            N_WASM_VALIDATE => {
                let limits = self.realm.limits.wasm;
                let ok = self.wasm_bytes(arg(0)).is_some_and(|b| {
                    crate::wasm_rt::Module::decode_with_limits(&b, &limits).is_ok()
                });
                NanBox::boolean(ok)
            }
            // `WebAssembly.Module.exports(module)` / `.imports(module)` — arrays of
            // `{ name, kind }` / `{ module, name, kind }` descriptors.
            N_WASM_MODULE_EXPORTS | N_WASM_MODULE_IMPORTS => {
                let bytes = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.get_property(h, WASM_BYTES))
                    .and_then(|v| self.wasm_bytes(v))
                    .ok_or_else(|| self.wasm_type_error("expected a WebAssembly.Module"))?;
                let module =
                    crate::wasm_rt::Module::decode_with_limits(&bytes, &self.realm.limits.wasm)
                        .map_err(|e| self.wasm_compile_error(e.0))?;
                let mut out = Vec::new();
                if id == N_WASM_MODULE_EXPORTS {
                    let descs: Vec<(String, u8)> = module
                        .export_descriptors()
                        .iter()
                        .map(|(n, k)| ((*n).into(), *k))
                        .collect();
                    for (name, kind) in descs {
                        let obj = self.realm.new_object();
                        let nv = self.new_str(&name);
                        self.realm.set_property(obj, "name", nv);
                        let kv = self.new_str(wasm_extern_kind(kind));
                        self.realm.set_property(obj, "kind", kv);
                        out.push(NanBox::handle(obj.to_raw()));
                    }
                } else {
                    let descs: Vec<(String, String, u8)> = module
                        .import_descriptors()
                        .iter()
                        .map(|(m, f, k)| ((*m).into(), (*f).into(), *k))
                        .collect();
                    for (m, f, kind) in descs {
                        let obj = self.realm.new_object();
                        let mv = self.new_str(&m);
                        self.realm.set_property(obj, "module", mv);
                        let nv = self.new_str(&f);
                        self.realm.set_property(obj, "name", nv);
                        let kv = self.new_str(wasm_extern_kind(kind));
                        self.realm.set_property(obj, "kind", kv);
                        out.push(NanBox::handle(obj.to_raw()));
                    }
                }
                NanBox::handle(self.realm.new_array(out).to_raw())
            }
            // `WebAssembly.instantiate(x)` → a `Promise`: given source bytes it
            // resolves to `{ module, instance }`; given a `Module` it resolves to the
            // `Instance` alone. Each export is a callable wrapper. (A stateful module
            // re-instantiates per call.)
            N_WASM_INSTANTIATE => {
                let module_handle = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.get_property(*h, WASM_IS_MODULE).is_some());
                let given_module = module_handle.is_some();
                // `build_wasm_instance` consumes source bytes; a `Module` argument
                // carries them under `WASM_BYTES`.
                let source =
                    match module_handle.and_then(|h| self.realm.get_property(h, WASM_BYTES)) {
                        Some(bytes) => bytes,
                        None => arg(0),
                    };
                let p = self.fresh_promise();
                match self.build_wasm_instance(source, arg(1)) {
                    Ok(instance) => {
                        let resolved = if given_module {
                            instance
                        } else {
                            let result = self.realm.new_object();
                            self.realm.set_property(result, "instance", instance);
                            let module = self.realm.new_object();
                            self.realm.set_property(module, WASM_BYTES, arg(0));
                            self.realm.mark_hidden(module, WASM_BYTES);
                            self.realm.set_hidden_property(
                                module,
                                WASM_IS_MODULE,
                                NanBox::boolean(true),
                            );
                            self.realm.set_property(
                                result,
                                "module",
                                NanBox::handle(module.to_raw()),
                            );
                            NanBox::handle(result.to_raw())
                        };
                        self.settle(p, resolved, true);
                    }
                    Err(ExecError::Throw(err)) => self.settle(p, err, false),
                    Err(other) => return Err(other),
                }
                NanBox::handle(p.to_raw())
            }
            // `WebAssembly.compile(bytes)` → `Promise<Module>` (rejected, not thrown,
            // on a bad module).
            N_WASM_COMPILE => {
                let p = self.fresh_promise();
                match self.make_wasm_module(arg(0)) {
                    Ok(module) => self.settle(p, module, true),
                    Err(ExecError::Throw(err)) => self.settle(p, err, false),
                    Err(other) => return Err(other),
                }
                NanBox::handle(p.to_raw())
            }
            // `Object.prototype.*` methods — the receiver is `self.this_val`.
            N_OBJ_PROTO_TOSTRING => {
                let this = self.this_val;
                let s = match this.unpack() {
                    Unpacked::Undefined => String::from("[object Undefined]"),
                    Unpacked::Null => String::from("[object Null]"),
                    // A primitive number/boolean (an immediate) reports its class
                    // (ToObject would box it to a Number/Boolean wrapper).
                    Unpacked::Number(_) => String::from("[object Number]"),
                    Unpacked::Bool(_) => String::from("[object Boolean]"),
                    _ => match this.as_handle().map(Handle::from_raw) {
                        Some(h) => alloc::format!("[object {}]", self.object_string_tag(h)?),
                        None => String::from("[object Object]"),
                    },
                };
                self.new_str(&s)
            }
            N_OBJ_PROTO_VALUEOF => self.this_val,
            // `Error.prototype.toString` — the receiver must be an Object (a
            // string/symbol/bigint primitive or null/undefined is a TypeError);
            // reads `name`/`message` (each ToString'd, defaulting to `"Error"`/`""`)
            // and renders `"name: message"` (or one part when the other is empty).
            N_ERROR_PROTO_TOSTRING => {
                let this = self.this_val;
                let Some(h) = this.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("Error.prototype.toString called on non-object"));
                };
                // A boxed primitive (string/symbol/bigint) is not an ordinary Object.
                if !self.is_object_value(this) {
                    return Err(self.type_error("Error.prototype.toString called on non-object"));
                }
                let name_v = self.read_member(h, "name")?;
                let name = if matches!(name_v.unpack(), Unpacked::Undefined) {
                    String::from("Error")
                } else {
                    self.coerce_to_string(name_v)?
                };
                let msg_v = self.read_member(h, "message")?;
                let msg = if matches!(msg_v.unpack(), Unpacked::Undefined) {
                    String::new()
                } else {
                    self.coerce_to_string(msg_v)?
                };
                let s = if name.is_empty() {
                    msg
                } else if msg.is_empty() {
                    name
                } else {
                    alloc::format!("{name}: {msg}")
                };
                self.new_str(&s)
            }
            N_OBJ_PROTO_HASOWN => {
                let key = self.member_key(arg(0));
                match self.this_val.as_handle().map(Handle::from_raw) {
                    Some(h) => NanBox::boolean(self.realm.has_own(h, &key)),
                    None => NanBox::boolean(false),
                }
            }
            N_OBJ_PROTO_PROPISENUM => {
                let key = self.member_key(arg(0));
                // An own *and* enumerable property. `property_is_enumerable` works
                // for inline objects *and* aux-backed cells (arrays/functions/
                // classes), where `object_keys` returns `None` and would wrongly
                // report every aux property non-enumerable.
                let enumerable = self
                    .this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| {
                        self.realm.has_own(h, &key) && self.realm.property_is_enumerable(h, &key)
                    });
                NanBox::boolean(enumerable)
            }
            N_OBJ_PROTO_ISPROTOTYPEOF => {
                // True if `this` appears in arg(0)'s prototype chain.
                let target = self.this_val.as_handle().map(Handle::from_raw);
                let mut cur = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.object_proto(h));
                let mut found = false;
                while let Some(p) = cur {
                    if Some(p) == target {
                        found = true;
                        break;
                    }
                    cur = self.realm.object_proto(p);
                }
                NanBox::boolean(found)
            }
            // `btoa(s)`: each code unit must be a byte (0–255) → base64.
            N_BTOA => {
                let s = self.realm.to_display_string(arg(0));
                let mut bytes = Vec::with_capacity(s.chars().count());
                for ch in s.chars() {
                    if (ch as u32) > 0xff {
                        let m = self.new_str("string contains a non-Latin1 character");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    bytes.push(ch as u8);
                }
                let h = self.realm.new_string(&base64_encode(&bytes));
                NanBox::handle(h.to_raw())
            }
            // `atob(s)`: base64 → a string of bytes (each a code unit 0–255).
            N_ATOB => {
                let s = self.realm.to_display_string(arg(0));
                match base64_decode(&s) {
                    Some(bytes) => {
                        let decoded: String = bytes.iter().map(|b| *b as char).collect();
                        let h = self.realm.new_string(&decoded);
                        NanBox::handle(h.to_raw())
                    }
                    None => {
                        let m = self.new_str("invalid base64");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                }
            }
            N_IS_NAN => NanBox::boolean(self.realm.to_number(arg(0)).is_nan()),
            N_IS_FINITE => NanBox::boolean(self.realm.to_number(arg(0)).is_finite()),
            // `Error(msg)` / `new Error(msg, { cause })` (the ES2022 cause option).
            id if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) => {
                let err = self.make_error(id, args.first().copied());
                if let Some(opts) = args
                    .get(1)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                    && let Some(cause) = self.realm.get_property(opts, "cause")
                    && let Some(eh) = err.as_handle()
                {
                    self.realm
                        .set_property(Handle::from_raw(eh), "cause", cause);
                    self.realm.mark_hidden(Handle::from_raw(eh), "cause");
                }
                err
            }
            // An unrecognized native dispatch id: the value is not a callable the
            // engine can invoke. Surface a catchable JS `TypeError` rather than an
            // internal error so user `try/catch` can handle it.
            _ => {
                let m = self.new_str("is not a function");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        })
    }
}
