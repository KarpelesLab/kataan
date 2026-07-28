use super::*;

impl<'a> Interp<'a> {
    /// ToPrimitive of an object/array for loose equality: an array becomes its
    /// `join` string; a plain object uses the default-hint ToPrimitive.
    pub(crate) fn coerce_for_eq(&mut self, v: NanBox) -> Result<NanBox, ExecError> {
        self.coerce_object(v, "default")
    }

    /// ToPrimitive of an object/array with `hint`: an array becomes its `join`
    /// string (arrays have no readable `valueOf`/`toString`); a plain object goes
    /// through `coerce_primitive`. Non-objects pass through.
    /// `ToBigInt(v)` (ES2020 7.1.13) — the coercion for writing a
    /// `BigInt64Array`/`BigUint64Array` element (and the value of
    /// `DataView.prototype.setBig*`). A BigInt passes through; a Boolean maps to
    /// `0n`/`1n`; a String parses (an invalid one is a `SyntaxError`); an object is
    /// taken through ToPrimitive(number) and re-coerced. A **Number**, Symbol,
    /// `undefined`, or `null` is a `TypeError` (notably: assigning a Number to a
    /// BigInt typed-array element throws).
    /// `thisBigIntValue(value)`: the BigInt of `value` if it is a BigInt or a
    /// BigInt wrapper object (`Object(1n)`), else `None` (the caller throws a
    /// TypeError). A BigInt wrapper carries its primitive in `PRIM_WRAP` with a
    /// `PRIM_WRAP_TYPE` of `N_BIGINT`.
    pub(crate) fn this_bigint_value(&self, value: NanBox) -> Option<crate::bignum::BigInt> {
        let h = Handle::from_raw(value.as_handle()?);
        if let Some(big) = self.realm.bigint_at(h) {
            return Some(big);
        }
        let ty = self.realm.get_property(h, PRIM_WRAP_TYPE)?;
        if ty.as_number() == Some(f64::from(N_BIGINT)) {
            let prim = self.realm.get_property(h, PRIM_WRAP)?;
            return self.realm.bigint_at(Handle::from_raw(prim.as_handle()?));
        }
        None
    }

    /// `thisNumberValue(value)`: the Number primitive `value` *is* (an immediate
    /// number), or the `[[NumberData]]` of a Number wrapper object. `None` for any
    /// other value (the caller then throws a `TypeError`).
    pub(crate) fn this_number_value(&self, value: NanBox) -> Option<f64> {
        if let Some(n) = value.as_number() {
            return Some(n);
        }
        let h = Handle::from_raw(value.as_handle()?);
        let ty = self.realm.get_property(h, PRIM_WRAP_TYPE)?;
        if ty.as_number() == Some(f64::from(N_NUMBER)) {
            return self
                .realm
                .get_property(h, PRIM_WRAP)
                .and_then(|p| p.as_number());
        }
        None
    }

    /// `thisBooleanValue(value)`: the Boolean primitive `value` *is*, or the
    /// `[[BooleanData]]` of a Boolean wrapper object. `None` otherwise (the caller
    /// then throws a `TypeError`).
    pub(crate) fn this_boolean_value(&self, value: NanBox) -> Option<bool> {
        if let Unpacked::Bool(b) = value.unpack() {
            return Some(b);
        }
        let h = Handle::from_raw(value.as_handle()?);
        let ty = self.realm.get_property(h, PRIM_WRAP_TYPE)?;
        if ty.as_number() == Some(f64::from(N_BOOLEAN)) {
            return match self.realm.get_property(h, PRIM_WRAP)?.unpack() {
                Unpacked::Bool(b) => Some(b),
                _ => None,
            };
        }
        None
    }

    pub(crate) fn coerce_to_bigint(
        &mut self,
        v: NanBox,
    ) -> Result<crate::bignum::BigInt, ExecError> {
        match v.unpack() {
            Unpacked::Bool(b) => Ok(if b {
                crate::bignum::BigInt::from_i128(1)
            } else {
                crate::bignum::BigInt::zero()
            }),
            Unpacked::Number(_) => {
                let m = self.new_str("Cannot convert a Number to a BigInt");
                Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
            }
            Unpacked::Undefined | Unpacked::Null => {
                let m = self.new_str("Cannot convert undefined or null to a BigInt");
                Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
            }
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if let Some(big) = self.realm.bigint_at(h) {
                    return Ok(big);
                }
                if let Some(s) = self.realm.string_value(h) {
                    // StringToBigInt: an empty/whitespace string is `0n`; an
                    // otherwise-invalid string is a SyntaxError.
                    let t = s.trim_matches(crate::realm::is_js_whitespace);
                    if t.is_empty() {
                        return Ok(crate::bignum::BigInt::zero());
                    }
                    let (radix, body) = match t.get(0..2) {
                        Some("0x" | "0X") => (16, &t[2..]),
                        Some("0o" | "0O") => (8, &t[2..]),
                        Some("0b" | "0B") => (2, &t[2..]),
                        _ => (10, t),
                    };
                    return match crate::bignum::BigInt::from_str_radix(body, radix) {
                        Some(b) => Ok(b),
                        None => {
                            let m = self.new_str("Cannot convert string to a BigInt");
                            Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))))
                        }
                    };
                }
                if self.realm.symbol_at(h).is_some() {
                    let m = self.new_str("Cannot convert a Symbol value to a BigInt");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // An array's ToPrimitive(number) is its `toString` (its `valueOf`
                // returns the array itself), i.e. the joined string.
                if self.realm.is_array(h) {
                    let s = self.realm.to_display_string(v);
                    let str_val = self.new_str(&s);
                    return self.coerce_to_bigint(str_val);
                }
                // An object: ToPrimitive(number), then ToBigInt on the primitive
                // (which is no longer an object, so the recursion terminates).
                let prim = self.coerce_primitive(v, "number")?;
                if prim.as_handle() == v.as_handle() {
                    // ToPrimitive produced no primitive (still the same object):
                    // treat as a non-coercible value.
                    let m = self.new_str("Cannot convert object to a BigInt");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                self.coerce_to_bigint(prim)
            }
        }
    }

    /// `ToNumber(value)` as a fallible operation: an object is taken through
    /// ToPrimitive(number) (running `valueOf`/`toString`, which may throw), and a
    /// Symbol is a TypeError. Returns the resulting `Number` NanBox. Used where a
    /// later infallible `Realm::to_number` would silently swallow these errors —
    /// e.g. coercing an array-like object's elements during typed-array construction.
    pub(crate) fn coerce_to_number(&mut self, value: NanBox) -> Result<NanBox, ExecError> {
        let prim = self.coerce_primitive(value, "number")?;
        if let Some(h) = prim.as_handle().map(Handle::from_raw) {
            if self.realm.symbol_at(h).is_some() {
                let m = self.new_str("Cannot convert a Symbol value to a number");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // ToNumber on a BigInt throws a TypeError (no implicit conversion).
            if self.realm.bigint_at(h).is_some() {
                let m = self.new_str("Cannot convert a BigInt value to a number");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        Ok(NanBox::number(self.realm.to_number(prim)))
    }

    /// `ToIntegerOrInfinity(value)`: ToNumber then truncate toward zero, mapping
    /// `NaN` to `0` and preserving `±Infinity`. Propagates the TypeError that
    /// ToNumber raises for a Symbol or BigInt.
    pub(crate) fn coerce_to_integer_or_infinity(
        &mut self,
        value: NanBox,
    ) -> Result<f64, ExecError> {
        let num = self.coerce_to_number(value)?;
        let n = self.realm.to_number(num);
        Ok(if n.is_nan() {
            0.0
        } else {
            trunc_toward_zero(n)
        })
    }

    /// `ToIndex(value)`: ToIntegerOrInfinity, then a `RangeError` unless the
    /// result is an integer in `[0, 2^53 − 1]`. Returns the index as `u64`.
    pub(crate) fn coerce_to_index(&mut self, value: NanBox) -> Result<u64, ExecError> {
        let n = self.coerce_to_integer_or_infinity(value)?;
        if !(0.0..=9_007_199_254_740_991.0).contains(&n) {
            let m = self.new_str("Invalid index");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(n as u64)
    }

    pub(crate) fn coerce_object(&mut self, v: NanBox, hint: &str) -> Result<NanBox, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(h) {
                let s = self.realm.to_display_string(v);
                return Ok(self.new_str(&s));
            }
            // A plain object, or a primitive-wrapper (`Object(Symbol())`,
            // `Object(1n)`, `new Number(1)`, …). Keyless wrappers (Symbol/BigInt
            // boxes) have no `object_keys` but must still run ToPrimitive so their
            // boxed primitive surfaces (a bare return-as-is would leave the wrapper
            // object, coercing to `NaN` and skipping the Symbol/BigInt TypeError).
            if self.realm.object_keys(h).is_some()
                || self.realm.get_property(h, PRIM_WRAP).is_some()
            {
                return self.coerce_primitive(v, hint);
            }
        }
        Ok(v)
    }

    /// ToPrimitive for a plain object with the given hint: `[Symbol.toPrimitive]`
    /// first, then `valueOf`/`toString` (order depends on the hint), accepting
    /// the first non-object result. Non-objects (and strings/arrays) pass through.
    /// `OrdinaryToPrimitive(O, hint)`: tries `valueOf`/`toString` in the order set
    /// by `hint` (`"string"` → toString first; otherwise valueOf first), returning
    /// the first call that yields a primitive, else a TypeError. Unlike
    /// [`Self::coerce_primitive`] this does NOT consult `@@toPrimitive` (it is the
    /// fallback those callers reach), so it is safe to invoke from a
    /// `@@toPrimitive` method without infinite recursion.
    pub(crate) fn ordinary_to_primitive(
        &mut self,
        v: NanBox,
        hint: &str,
    ) -> Result<NanBox, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(v);
        };
        let h = Handle::from_raw(raw);
        let order = if hint == "string" {
            ["toString", "valueOf"]
        } else {
            ["valueOf", "toString"]
        };
        for method in order {
            let m = self.read_member(h, method)?;
            if m.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let r = self.call_with_this(m, v, &[])?;
                if !self.is_object_value(r) {
                    return Ok(r);
                }
            }
        }
        let m = self.new_str("Cannot convert object to primitive value");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    pub(crate) fn coerce_primitive(&mut self, v: NanBox, hint: &str) -> Result<NanBox, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(v);
        };
        let h = Handle::from_raw(raw);
        // A primitive wrapper (`new Number/String/Boolean(...)`, a boxed
        // Symbol/BigInt) is an ordinary object for ToPrimitive: it runs
        // `@@toPrimitive` / OrdinaryToPrimitive so a user-overridden
        // `valueOf`/`toString` is honored (the default methods just return the
        // boxed value). It must NOT short-circuit to the boxed primitive.
        let is_wrapper = self.realm.get_property(h, PRIM_WRAP).is_some();
        // Strings and arrays are handled by the arithmetic path directly. A
        // typed-array view is an exotic object (no `object_keys`) but still has a
        // `valueOf`/`toString` (and may carry a user-overridden one or an
        // `@@toPrimitive`), so it must go through OrdinaryToPrimitive below — not
        // be returned as-is.
        // A `Date` is an exotic object with no `object_keys`, but it has a
        // meaningful `[Symbol.toPrimitive]` / `valueOf` (its timestamp) and so must
        // run OrdinaryToPrimitive — not be returned as-is (which would make
        // `Date.prototype.toJSON.call(date)` see the Date object rather than its
        // numeric time, and call `toISOString` even on an invalid date).
        if !is_wrapper
            && (self.realm.is_string_handle(h)
                || self.realm.is_array(h)
                || (self.realm.object_keys(h).is_none()
                    // A function is an ordinary object for ToPrimitive: it must run
                    // OrdinaryToPrimitive so a user-installed `valueOf`/`toString`
                    // is honored (`f.valueOf = () => 1; 1 + f` is `2`, not string
                    // concatenation). Only genuinely key-less *non-callable* exotics
                    // fall through to the return-as-is fast path.
                    && !self.is_callable(h)
                    && self.realm.typed_kind(h).is_none()
                    && self.realm.date_at(h).is_none()
                    // A Temporal instance is an ordinary object for ToPrimitive:
                    // it must run OrdinaryToPrimitive so its `valueOf` (which
                    // always throws a TypeError) fires under `<`/`+`/etc.
                    && self.realm.temporal_at(h).is_none()))
        {
            return Ok(v);
        }
        if let Some(r) = self.symbol_to_primitive(v, hint)? {
            return Ok(r);
        }
        // String hint tries `toString` first; number/default try `valueOf` first.
        let order = if hint == "string" {
            ["toString", "valueOf"]
        } else {
            ["valueOf", "toString"]
        };
        for method in order {
            let m = self.read_member(h, method)?;
            if m.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let r = self.call_with_this(m, v, &[])?;
                if !self.is_object_value(r) {
                    return Ok(r);
                }
            }
        }
        // Neither `valueOf` nor `toString` produced a primitive — a TypeError.
        let m = self.new_str("Cannot convert object to primitive value");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    /// Coerces `v` to a string, invoking `[Symbol.toPrimitive]("string")` or a
    /// callable `toString` when present (else the default form).
    pub(crate) fn coerce_to_string(&mut self, v: NanBox) -> Result<String, ExecError> {
        if let Some(raw) = v.as_handle() {
            let h = Handle::from_raw(raw);
            // A Symbol has no implicit string conversion (e.g. in a template).
            if self.realm.symbol_at(h).is_some() {
                let m = self.new_str("Cannot convert a Symbol value to a string");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // Any object that is not already a primitive string runs
            // `ToPrimitive(v, "string")` first — `@@toPrimitive`, then
            // `OrdinaryToPrimitive` (`toString` before `valueOf`) — so a
            // prototype-inherited or overridden `toString`/`valueOf` is honored
            // (e.g. `String(fn)` with a custom `Function.prototype.toString`). The
            // default methods simply reproduce the intrinsic display form, so
            // ordinary cases are unchanged.
            //
            // Arrays and typed arrays also run `ToPrimitive("string")`: their
            // `toString` (`Array.prototype.toString` → `join`, or a user override)
            // is a callable first-class value that `ordinary_to_primitive` invokes,
            // so `String([1,2])` honors an overridden `Array.prototype.toString`
            // (and the default reproduces the same `join`-based display).
            if !self.realm.is_string_handle(h) && self.is_object_value(v) {
                let p = if let Some(r) = self.symbol_to_primitive(v, "string")? {
                    r
                } else {
                    self.ordinary_to_primitive(v, "string")?
                };
                // `ToString` of the resulting primitive: a Symbol has no string
                // conversion (e.g. an object whose `toString` returns a Symbol) — a
                // TypeError, not its descriptive string.
                if p.as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|ph| self.realm.symbol_at(ph).is_some())
                {
                    let m = self.new_str("Cannot convert a Symbol value to a string");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                return Ok(self.realm.to_display_string(p));
            }
        }
        Ok(self.realm.to_display_string(v))
    }

    /// Like [`Self::coerce_to_string`] but returns **WTF-8 bytes**, preserving
    /// lone surrogates when `v` is already a string. Other kinds (numbers,
    /// objects via `toString`) never carry surrogates, so their lossy `String`
    /// form is byte-identical to its UTF-8.
    pub(crate) fn coerce_to_string_bytes(&mut self, v: NanBox) -> Result<Vec<u8>, ExecError> {
        if let Some(raw) = v.as_handle()
            && let Some(bytes) = self.realm.string_bytes(Handle::from_raw(raw))
        {
            return Ok(bytes);
        }
        Ok(self.coerce_to_string(v)?.into_bytes())
    }
}
