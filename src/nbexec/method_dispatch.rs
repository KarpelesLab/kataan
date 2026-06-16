use super::*;

impl<'a> Interp<'a> {
    /// Dispatches a built-in method on a string/array receiver. Returns
    /// `Ok(None)` if `method` is not a recognized built-in (the caller then
    /// treats it as an ordinary property-valued function).
    pub(crate) fn call_method(
        &mut self,
        recv: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());

        // ValidateTypedArray: the data-accessing `%TypedArray%.prototype` methods
        // throw a TypeError up front if the backing buffer is detached (the view was
        // length-0'd on detach, so without this they would silently operate on an
        // empty array). `subarray` (builds a fresh view), the iterator factories
        // (`values`/`keys`/`entries`), and `toString` (generic) are exempt.
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && self.realm.typed_kind(h).is_some()
            && !matches!(
                method,
                "subarray" | "values" | "keys" | "entries" | "toString" | "constructor"
            )
            && TYPED_ARRAY_PROTO_METHODS.iter().any(|(n, _)| *n == method)
            && self.typed_array_detached(h)
        {
            return Err(self.type_error(&alloc::format!(
                "TypedArray.prototype.{method} called on a detached ArrayBuffer"
            )));
        }

        // A primitive wrapper object (`new Number`/`String`/`Boolean`): `valueOf`
        // recovers the boxed primitive; every other method delegates to it.
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && let Some(prim) = self.realm.get_property(h, PRIM_WRAP)
        {
            return match method {
                "valueOf" => Ok(Some(prim)),
                _ => self.call_method(prim, method, args),
            };
        }

        // --- boolean methods (the receiver is an immediate) ---
        if let Unpacked::Bool(b) = recv.unpack() {
            return Ok(match method {
                "toString" => Some(self.new_str(if b { "true" } else { "false" })),
                "valueOf" => Some(recv),
                _ => None,
            });
        }
        // --- number methods (the receiver is an immediate, not a handle) ---
        if let Some(n) = recv.as_number() {
            return Ok(match method {
                "toString" => {
                    // The radix is ToIntegerOrInfinity'd; it must be in [2, 36] or
                    // a RangeError (undefined defaults to 10).
                    let radix = match args.first() {
                        Some(a) if !matches!(a.unpack(), Unpacked::Undefined) => {
                            let r = self.coerce_to_integer_or_infinity(*a)?;
                            if !(2.0..=36.0).contains(&r) {
                                let m = self.new_str("toString() radix must be between 2 and 36");
                                return Err(ExecError::Throw(
                                    self.make_error(N_RANGE_ERROR, Some(m)),
                                ));
                            }
                            r as u32
                        }
                        _ => 10,
                    };
                    // A non-finite value or base 10 uses the spec `Number::toString`.
                    if radix == 10 || !n.is_finite() {
                        Some(self.new_str(&self.realm.to_display_string(recv)))
                    } else {
                        Some(self.new_str(&int_to_radix(n, radix)))
                    }
                }
                "valueOf" => Some(recv),
                // `toLocaleString()` — a minimal grouping format (thousands
                // separators with `,`), since no locale data is available.
                "toLocaleString" => {
                    let s = self.number_to_locale_string(n, args.get(1).copied());
                    Some(self.new_str(&s))
                }
                #[cfg(feature = "std")]
                "toFixed" => {
                    // `fractionDigits` is ToIntegerOrInfinity'd (undefined/NaN → 0,
                    // a Symbol/BigInt → TypeError) and must be in [0, 100], else a
                    // RangeError.
                    let d = self.coerce_to_integer_or_infinity(arg(0))?;
                    let f = d as i64;
                    if !(0..=100).contains(&f) {
                        let m = self.new_str("toFixed() digits argument must be between 0 and 100");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    let digits = f as usize;
                    let s = if !n.is_finite() {
                        // `Infinity`/`-Infinity`/`NaN` use the spec ToString.
                        self.realm.to_display_string(NanBox::number(n))
                    } else if n.abs() >= 1e21 {
                        // Spec: a magnitude ≥ 1e21 uses the regular `ToString`
                        // (exponential), not a full decimal expansion.
                        self.realm.to_display_string(NanBox::number(n))
                    } else {
                        // Round the *exact* f64 to `digits` places. Rust's formatter is
                        // correctly rounded but ties-to-even; JS ties away from zero.
                        // Only an exact half (the dropped tail is precisely "5" then
                        // zeros) differs — detect that from the value's decimal
                        // expansion and round its magnitude up; everything else takes
                        // Rust's already-correct rounding (so e.g. `(2.355).toFixed(2)`
                        // is "2.35", since the double is 2.35499…, not "2.36").
                        let expanded = alloc::format!("{:.*}", digits + 25, n.abs());
                        let dot = expanded.find('.').unwrap_or(expanded.len());
                        let tail = &expanded[(dot + 1 + digits).min(expanded.len())..];
                        let exact_half = tail.starts_with('5')
                            && tail.as_bytes()[1..].iter().all(|&b| b == b'0');
                        if exact_half {
                            let kept: String = expanded[..dot]
                                .chars()
                                .chain(expanded[dot + 1..dot + 1 + digits].chars())
                                .collect();
                            let m = kept.parse::<u128>().unwrap_or(0) + 1;
                            let mut s = alloc::format!("{m}");
                            if digits > 0 {
                                while s.len() <= digits {
                                    s.insert(0, '0');
                                }
                                s.insert(s.len() - digits, '.');
                            }
                            if n < 0.0 {
                                s.insert(0, '-');
                            }
                            s
                        } else {
                            let mut s = alloc::format!("{n:.digits$}");
                            // A zero result never carries a sign (`(-0).toFixed(2)`).
                            if s.starts_with('-')
                                && s.bytes().all(|b| matches!(b, b'-' | b'0' | b'.'))
                            {
                                s.remove(0);
                            }
                            s
                        }
                    };
                    Some(self.new_str(&s))
                }
                // `toExponential(d)` — exponential notation with `d` fractional
                // digits and a signed exponent (`1.23e+3`).
                "toExponential" => {
                    // `fractionDigits` is ToIntegerOrInfinity'd first (a Symbol/BigInt
                    // → TypeError, and a user `valueOf` runs) — even for a non-finite
                    // `this`, whose result is then the spec ToString.
                    let undefined_digits = matches!(arg(0).unpack(), Unpacked::Undefined);
                    let di = self.coerce_to_integer_or_infinity(arg(0))? as i64;
                    if !n.is_finite() {
                        // `Infinity`/`-Infinity`/`NaN` use the spec ToString.
                        Some(self.new_str(&self.realm.to_display_string(NanBox::number(n))))
                    } else if undefined_digits {
                        Some(self.new_str(&format_exponential(n, None)))
                    } else {
                        // `fractionDigits` must be in [0, 100], else a RangeError.
                        if !(0..=100).contains(&di) {
                            let m =
                                self.new_str("toExponential() argument must be between 0 and 100");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        Some(self.new_str(&format_exponential(n, Some(di as usize))))
                    }
                }
                // `toPrecision(p)` — p significant digits (no arg → default
                // string form).
                "toPrecision" => {
                    if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        Some(self.new_str(&self.realm.to_display_string(recv)))
                    } else {
                        // Spec order: ToIntegerOrInfinity(precision) first (a
                        // Symbol/BigInt → TypeError); then a non-finite `this`
                        // returns its ToString; then the [1, 100] RangeError check.
                        let pi = self.coerce_to_integer_or_infinity(arg(0))? as i64;
                        if !n.is_finite() {
                            Some(self.new_str(&self.realm.to_display_string(recv)))
                        } else {
                            if !(1..=100).contains(&pi) {
                                let m = self
                                    .new_str("toPrecision() argument must be between 1 and 100");
                                return Err(ExecError::Throw(
                                    self.make_error(N_RANGE_ERROR, Some(m)),
                                ));
                            }
                            let p = pi as usize;
                            Some(self.new_str(&format_precision(n, p)))
                        }
                    }
                }
                _ => None,
            });
        }

        let Some(raw) = recv.as_handle() else {
            return Ok(None);
        };
        let handle = Handle::from_raw(raw);

        // --- WeakRef / FinalizationRegistry (bounded: no mid-execution GC) ---
        if method == "deref"
            && let Some(target) = self.realm.get_property(handle, WEAKREF_TARGET)
        {
            return Ok(Some(target));
        }
        if self.realm.get_property(handle, FINREG_TAG).is_some() {
            match method {
                // `register(target, heldValue, unregisterToken?)` — inert.
                "register" => return Ok(Some(NanBox::undefined())),
                // `unregister(token)` — nothing was ever registered.
                "unregister" => return Ok(Some(NanBox::boolean(false))),
                _ => {}
            }
        }

        // --- universal `Object.prototype` methods (own/inherited reflection) ---
        match method {
            "hasOwnProperty" => {
                // `member_key` maps a symbol to its internal slot name (a string key
                // passes through), so a symbol-keyed property is found.
                let key = self.member_key(arg(0));
                return Ok(Some(NanBox::boolean(self.realm.has_own(handle, &key))));
            }
            "isPrototypeOf" => {
                let mut cur = arg(0).as_handle().map(Handle::from_raw);
                while let Some(p) = cur.and_then(|h| self.realm.object_proto(h)) {
                    if p == handle {
                        return Ok(Some(NanBox::boolean(true)));
                    }
                    cur = Some(p);
                }
                return Ok(Some(NanBox::boolean(false)));
            }
            "propertyIsEnumerable" => {
                // True only for an *own* *enumerable* property (a non-enumerable one,
                // or an inherited one, is false). `member_key` resolves symbol keys.
                let key = self.member_key(arg(0));
                let r = self.realm.has_own(handle, &key)
                    && self.realm.property_is_enumerable(handle, &key);
                return Ok(Some(NanBox::boolean(r)));
            }
            // Legacy (Annex B) accessor helpers on Object.prototype.
            "__defineGetter__" => {
                let key = self.realm.to_display_string(arg(0));
                let setter = self
                    .realm
                    .accessor(handle, &key)
                    .map_or(NanBox::undefined(), |(_, s)| s);
                self.realm.define_accessor(handle, &key, arg(1), setter);
                return Ok(Some(NanBox::undefined()));
            }
            "__defineSetter__" => {
                let key = self.realm.to_display_string(arg(0));
                let getter = self
                    .realm
                    .accessor(handle, &key)
                    .map_or(NanBox::undefined(), |(g, _)| g);
                self.realm.define_accessor(handle, &key, getter, arg(1));
                return Ok(Some(NanBox::undefined()));
            }
            "__lookupGetter__" | "__lookupSetter__" => {
                let want_getter = method == "__lookupGetter__";
                let key = self.realm.to_display_string(arg(0));
                let mut cur = Some(handle);
                while let Some(c) = cur {
                    if let Some((g, s)) = self.realm.accessor(c, &key) {
                        return Ok(Some(if want_getter { g } else { s }));
                    }
                    // An own data property shadows an inherited accessor.
                    if self.realm.has_own(c, &key) {
                        break;
                    }
                    cur = self.realm.object_proto(c);
                }
                return Ok(Some(NanBox::undefined()));
            }
            // An error object (`name` + `message`, no own `toString`) renders as
            // `"Name: message"` (or just `"Name"` when the message is empty).
            "toString"
                if self.realm.has_own(handle, "name")
                    && self.realm.has_own(handle, "message")
                    && !self.realm.has_own(handle, "toString") =>
            {
                let name = self
                    .realm
                    .get_property(handle, "name")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                let msg = self
                    .realm
                    .get_property(handle, "message")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                // `Error.prototype.toString`: an empty name yields just the message;
                // an empty message yields just the name; else `"name: message"`.
                let s = if name.is_empty() {
                    msg
                } else if msg.is_empty() {
                    name
                } else {
                    alloc::format!("{name}: {msg}")
                };
                return Ok(Some(self.new_str(&s)));
            }
            _ => {}
        }

        // --- `Function.prototype.call`/`apply`/`bind` on a callable receiver ---
        // `call`/`apply`/`bind` work on any constructor, including a class.
        if self.is_callable(handle) || self.realm.class_at(handle).is_some() {
            match method {
                "call" => {
                    let this = arg(0);
                    let rest: Vec<NanBox> = args.iter().skip(1).copied().collect();
                    return self.call_with_this(recv, this, &rest).map(Some);
                }
                "apply" => {
                    let this = arg(0);
                    // CreateListFromArrayLike(argArray): `null`/`undefined` is an
                    // empty list; an Object is read via its `length`/indices; any
                    // other value (a number/boolean/string/symbol/bigint) is a
                    // TypeError ("CreateListFromArrayLike called on non-object").
                    let arg_array = arg(1);
                    let list = if matches!(arg_array.unpack(), Unpacked::Undefined | Unpacked::Null)
                    {
                        Vec::new()
                    } else if let Some(h) =
                        arg_array.as_handle().map(Handle::from_raw).filter(|_| {
                            self.is_object_value(arg_array)
                                || self
                                    .realm
                                    .is_array_like(Handle::from_raw(arg_array.as_handle().unwrap()))
                        })
                    {
                        if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
                            elems
                        } else {
                            // An array-like: its `length` and indexed properties.
                            let len = self
                                .realm
                                .get_property(h, "length")
                                .map_or(0, |v| self.realm.to_number(v).max(0.0) as usize);
                            let mut v = Vec::with_capacity(len);
                            for i in 0..len {
                                v.push(self.read_member(h, &alloc::format!("{i}"))?);
                            }
                            v
                        }
                    } else {
                        return Err(self.type_error("CreateListFromArrayLike called on non-object"));
                    };
                    return self.call_with_this(recv, this, &list).map(Some);
                }
                "bind" => {
                    let this = arg(0);
                    let bound: Vec<NanBox> = args.iter().skip(1).copied().collect();
                    return Ok(Some(self.make_bound_function(recv, this, bound)));
                }
                // A textual representation (the engine does not retain source).
                "toString" | "toLocaleString" => {
                    let nm = self.read_member(handle, "name")?;
                    let nm = self.realm.to_display_string(nm);
                    let s = if self.realm.class_at(handle).is_some() {
                        alloc::format!("class {nm} {{ }}")
                    } else {
                        alloc::format!("function {nm}() {{ [native code] }}")
                    };
                    return Ok(Some(self.new_str(&s)));
                }
                _ => {}
            }
        }

        // --- generator iterator protocol (`next`/`return`) ---
        if let Some(buf) = self
            .realm
            .get_property(handle, GEN_BUF)
            .and_then(|b| b.as_handle())
            .map(Handle::from_raw)
        {
            match method {
                "next" => {
                    let idx = self
                        .realm
                        .get_property(handle, GEN_IDX)
                        .and_then(|n| n.as_number())
                        .unwrap_or(0.0) as usize;
                    let elems = self.realm.array_elements(buf).map(<[_]>::to_vec);
                    let len = elems.as_ref().map_or(0, Vec::len);
                    let (value, done) = match elems.as_ref().and_then(|e| e.get(idx)) {
                        Some(v) => {
                            self.realm.set_hidden_property(
                                handle,
                                GEN_IDX,
                                NanBox::number((idx + 1) as f64),
                            );
                            (*v, false)
                        }
                        // The first call past the yields surfaces the `return`
                        // value (with `done: true`); later calls yield undefined.
                        None => {
                            let v = if idx == len {
                                self.realm.set_hidden_property(
                                    handle,
                                    GEN_IDX,
                                    NanBox::number((idx + 1) as f64),
                                );
                                self.realm
                                    .get_property(handle, GEN_RET)
                                    .unwrap_or(NanBox::undefined())
                            } else {
                                NanBox::undefined()
                            };
                            (v, true)
                        }
                    };
                    let res = self.realm.new_object();
                    self.realm.set_property(res, "value", value);
                    self.realm.set_property(res, "done", NanBox::boolean(done));
                    return Ok(Some(NanBox::handle(res.to_raw())));
                }
                // `return()` ends the generator early.
                "return" => {
                    let len = self.realm.array_elements(buf).map_or(0, <[_]>::len);
                    self.realm
                        .set_hidden_property(handle, GEN_IDX, NanBox::number(len as f64));
                    let res = self.realm.new_object();
                    self.realm.set_property(res, "value", arg(0));
                    self.realm.set_property(res, "done", NanBox::boolean(true));
                    return Ok(Some(NanBox::handle(res.to_raw())));
                }
                "throw" => {
                    // Eager-generator model: the body has already run, so the thrown
                    // value can't be re-injected at the suspended `yield` (a
                    // `try`/`catch` *around* that yield won't observe it). Mark the
                    // generator done and propagate the value — correct when the
                    // generator does not catch at the yield (the common case) and for
                    // an already-exhausted generator.
                    let len = self.realm.array_elements(buf).map_or(0, <[_]>::len);
                    self.realm
                        .set_hidden_property(handle, GEN_IDX, NanBox::number(len as f64));
                    return Err(ExecError::Throw(arg(0)));
                }
                // ES2025 iterator helpers — they consume the remaining yields.
                "map" | "filter" | "take" | "drop" | "toArray" | "forEach" | "reduce" | "some"
                | "every" | "find" | "flatMap" => {
                    let idx = self
                        .realm
                        .get_property(handle, GEN_IDX)
                        .and_then(|n| n.as_number())
                        .unwrap_or(0.0) as usize;
                    let rest: Vec<NanBox> = self
                        .realm
                        .array_elements(buf)
                        .map(|e| e.get(idx..).unwrap_or(&[]).to_vec())
                        .unwrap_or_default();
                    // The source iterator is now exhausted.
                    let len = self.realm.array_elements(buf).map_or(0, <[_]>::len);
                    self.realm
                        .set_hidden_property(handle, GEN_IDX, NanBox::number(len as f64));
                    let f = arg(0);
                    return Ok(Some(match method {
                        "toArray" => NanBox::handle(self.realm.new_array(rest).to_raw()),
                        "map" => {
                            let mut out = Vec::with_capacity(rest.len());
                            for v in rest {
                                out.push(self.call(f, &[v])?);
                            }
                            self.make_generator(out)
                        }
                        "flatMap" => {
                            let mut out = Vec::new();
                            for v in rest {
                                let r = self.call(f, &[v])?;
                                out.extend(
                                    self.iterate_values(r).unwrap_or_else(|_| alloc::vec![r]),
                                );
                            }
                            self.make_generator(out)
                        }
                        "filter" => {
                            let mut out = Vec::new();
                            for v in rest {
                                let r = self.call(f, &[v])?;
                                if self.realm.truthy(r) {
                                    out.push(v);
                                }
                            }
                            self.make_generator(out)
                        }
                        "take" => {
                            let n = self.realm.to_number(f).max(0.0) as usize;
                            self.make_generator(rest.into_iter().take(n).collect())
                        }
                        "drop" => {
                            let n = self.realm.to_number(f).max(0.0) as usize;
                            self.make_generator(rest.into_iter().skip(n).collect())
                        }
                        "forEach" => {
                            for v in rest {
                                self.call(f, &[v])?;
                            }
                            NanBox::undefined()
                        }
                        "some" | "every" | "find" => {
                            let mut found = NanBox::undefined();
                            let mut hit = false;
                            for v in rest {
                                let r = self.call(f, &[v])?;
                                let t = self.realm.truthy(r);
                                if method == "every" && !t {
                                    return Ok(Some(NanBox::boolean(false)));
                                }
                                if method != "every" && t {
                                    found = v;
                                    hit = true;
                                    break;
                                }
                            }
                            match method {
                                "some" => NanBox::boolean(hit),
                                "every" => NanBox::boolean(true),
                                _ => found, // find
                            }
                        }
                        // reduce
                        _ => {
                            let mut it = rest.into_iter();
                            let mut acc = if args.len() >= 2 {
                                arg(1)
                            } else {
                                match it.next() {
                                    Some(v) => v,
                                    None => {
                                        let m = self.new_str(
                                            "Reduce of empty iterator with no initial value",
                                        );
                                        return Err(ExecError::Throw(
                                            self.make_error(N_TYPE_ERROR, Some(m)),
                                        ));
                                    }
                                }
                            };
                            for v in it {
                                acc = self.call(f, &[acc, v])?;
                            }
                            acc
                        }
                    }));
                }
                _ => {}
            }
        }

        // --- `Date.now()` static ---
        // `BigInt.asUintN(bits, x)` / `BigInt.asIntN(bits, x)` — wrap a BigInt to
        // the low `bits` bits, unsigned or signed (two's complement).
        if self.realm.native_at(handle) == Some(N_BIGINT) && matches!(method, "asUintN" | "asIntN")
        {
            use crate::bignum::BigInt;
            // Spec order: ToIndex(bits) first (which may throw a RangeError or run
            // user coercion), then ToBigInt(bigint).
            let bits = self.coerce_to_index(arg(0))?;
            let x = self.coerce_to_bigint(arg(1))?;
            // `2^bits` is the modulus; an attacker-supplied `bits` (e.g.
            // `BigInt.asUintN(1e18, 0n)`) would otherwise build a ~10^17-byte
            // BigInt and OOM/abort. Cap `bits` to the same size budget as the
            // `**`/`<<` operators before building any power-of-two (MEM-6).
            let max_bigint_bits = self.realm.limits.max_bigint_bits;
            if bits > max_bigint_bits {
                let m = self.new_str("Maximum BigInt size exceeded");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            // `try_pow` re-checks the projected size as defense in depth: even if
            // the cap above were ever loosened, no oversized allocation occurs.
            let Some(modulus) = BigInt::from_i128(2).try_pow(bits, max_bigint_bits) else {
                let m = self.new_str("Maximum BigInt size exceeded");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            };
            // Non-negative remainder modulo 2^bits.
            let mut u = x.divmod(&modulus).map_or_else(BigInt::zero, |(_, r)| r);
            if u.is_negative() {
                u = u.add(&modulus);
            }
            if method == "asIntN" && bits >= 1 {
                // If the top bit is set, the signed value is `u - 2^bits`.
                let half = BigInt::from_i128(2).pow(bits - 1);
                if !u.sub(&half).is_negative() {
                    u = u.sub(&modulus);
                }
            }
            return Ok(Some(NanBox::handle(self.realm.new_bigint(u).to_raw())));
        }
        if self.realm.native_at(handle) == Some(N_DATE) && method == "now" {
            return Ok(Some(NanBox::number(now_ms())));
        }
        // `Uint8Array.of(...items)` / `Uint8Array.from(iterable|arrayLike, mapFn?)`
        // — the typed-array statics, producing a typed array of the constructor's
        // kind (each value coerced to the element type).
        if let Some(id) = self.realm.native_at(handle)
            && (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16)
                .contains(&id)
            && matches!(method, "of" | "from")
        {
            let kind = (id - N_TYPED_ARRAY_BASE) as u8;
            // `from`'s optional map callback must be callable if present (a
            // TypeError otherwise), checked before iterating the source.
            let mapfn = if method == "from" {
                args.get(1).copied()
            } else {
                None
            };
            let has_mapfn = mapfn.is_some_and(|m| !matches!(m.unpack(), Unpacked::Undefined));
            if has_mapfn {
                self.require_callable(mapfn.unwrap(), "TypedArray.from mapfn")?;
            }
            // `from` iterates the source (an iterator error propagates, not
            // swallowed); a non-iterable array-like is read by index. `of` takes
            // its variadic args directly.
            let mut items: Vec<NanBox> = if method == "of" {
                args.to_vec()
            } else {
                match self.iterate_values(arg(0)) {
                    Ok(v) => v,
                    Err(ExecError::Throw(t)) => {
                        // A genuine throw from the iterator protocol propagates; a
                        // non-iterable source falls back to the array-like path.
                        if self.value_is_iterable(arg(0)) {
                            return Err(ExecError::Throw(t));
                        }
                        let src = arg(0);
                        let obj = self.coerce_to_object(src);
                        let Some(h) = obj.as_handle().map(Handle::from_raw) else {
                            return Ok(Some(self.typed_like(handle, Vec::new())));
                        };
                        let len_val = self.read_member(h, "length")?;
                        let len_n = self.coerce_to_integer_or_infinity(len_val)?;
                        let len = len_n.clamp(0.0, 9_007_199_254_740_991.0) as usize;
                        let mut out = Vec::with_capacity(len.min(1 << 20));
                        for i in 0..len {
                            out.push(self.read_member(h, &alloc::format!("{i}"))?);
                        }
                        out
                    }
                    Err(e) => return Err(e),
                }
            };
            if has_mapfn {
                let mapfn = mapfn.unwrap();
                let this_arg = args.get(2).copied().unwrap_or(NanBox::undefined());
                for (i, v) in items.iter_mut().enumerate() {
                    *v = self.call_with_this(mapfn, this_arg, &[*v, NanBox::number(i as f64)])?;
                }
            }
            // Allocate a backing buffer and view it; each item is coerced on write.
            let elem_size = TYPED_ARRAY_KINDS[kind as usize].1 as usize;
            let buf = self.make_array_buffer(items.len() * elem_size);
            let bytes_h = self.array_buffer_bytes(buf).unwrap();
            let view = self
                .realm
                .new_typed_array(bytes_h, buf, 0, items.len(), kind);
            // Bulk write-through: one buffer borrow, no per-element heap lookup.
            self.realm.typed_set_from_numbers(view, 0, &items);
            // Link the result's `[[Prototype]]` to the constructor's `.prototype`
            // (`Int8Array.of(...)`'s result is an `Int8Array` instance), so
            // `result.constructor`/`getPrototypeOf(result)` resolve.
            self.link_view_proto_to_ctor(view, NanBox::handle(handle.to_raw()));
            return Ok(Some(NanBox::handle(view.to_raw())));
        }
        // `Date.parse(str)` → epoch ms (or NaN) by ISO parsing.
        if self.realm.native_at(handle) == Some(N_DATE) && method == "parse" {
            // ToString(arg), then parse and TimeClip (out-of-range → NaN).
            let s = self.coerce_to_string(arg(0))?;
            return Ok(Some(NanBox::number(
                crate::realm::parse_date_string(&s).map_or(f64::NAN, time_clip),
            )));
        }
        // --- `Date.UTC(year, month, day?, h?, m?, s?, ms?)` → epoch ms ---
        if self.realm.native_at(handle) == Some(N_DATE) && method == "UTC" {
            // ToNumber every supplied argument, in order, with abrupt propagation
            // (a Symbol or throwing `valueOf` raises). `Date.UTC()` with no year is
            // NaN.
            let mut nums = Vec::with_capacity(args.len());
            for a in args {
                let v = self.coerce_to_number(*a)?;
                nums.push(self.realm.to_number(v));
            }
            let getn = |i: usize, dflt: f64| nums.get(i).copied().unwrap_or(dflt);
            let year_n = getn(0, f64::NAN);
            let month = getn(1, 0.0);
            let day = getn(2, 1.0);
            let hours = getn(3, 0.0);
            let mins = getn(4, 0.0);
            let secs = getn(5, 0.0);
            let millis = getn(6, 0.0);
            if [year_n, month, day, hours, mins, secs, millis]
                .iter()
                .any(|v| v.is_nan() || !v.is_finite())
            {
                return Ok(Some(NanBox::number(f64::NAN)));
            }
            // A two-digit year (0..=99) maps to 1900+year.
            let yi = year_n as i64;
            let year = if (0..=99).contains(&yi) {
                1900 + yi
            } else {
                yi
            };
            let total_months = year * 12 + month as i64;
            let y = total_months.div_euclid(12);
            let mo = total_months.rem_euclid(12) as u32 + 1;
            let days = crate::realm::days_from_civil(y, mo, 1) + (day as i64 - 1);
            let ms = time_clip(
                (days * 86_400_000
                    + hours as i64 * 3_600_000
                    + mins as i64 * 60_000
                    + secs as i64 * 1_000
                    + millis as i64) as f64,
            );
            return Ok(Some(NanBox::number(ms)));
        }
        // --- `Proxy.revocable(target, handler)` → `{ proxy, revoke }` ---
        if self.realm.native_at(handle) == Some(N_PROXY) && method == "revocable" {
            let (Some(tr), Some(hr)) = (
                arg(0).as_handle().filter(|_| self.is_object_value(arg(0))),
                arg(1).as_handle().filter(|_| self.is_object_value(arg(1))),
            ) else {
                let m = self.new_str("Cannot create proxy with a non-object target or handler");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            };
            let proxy = self
                .realm
                .new_proxy(Handle::from_raw(tr), Handle::from_raw(hr));
            let revoke = self.realm.new_bound_native(N_PROXY_REVOKE, proxy);
            let result = self.realm.new_object();
            self.realm
                .set_property(result, "proxy", NanBox::handle(proxy.to_raw()));
            self.realm
                .set_property(result, "revoke", NanBox::handle(revoke.to_raw()));
            return Ok(Some(NanBox::handle(result.to_raw())));
        }
        // --- `Symbol.for` / `Symbol.keyFor` (the global symbol registry) ---
        if self.realm.native_at(handle) == Some(N_SYMBOL) {
            match method {
                "for" => {
                    let key = self.realm.to_display_string(arg(0));
                    if let Some(s) = self.symbol_registry.get(&key) {
                        return Ok(Some(*s));
                    }
                    let sym = NanBox::handle(self.realm.new_symbol(&key).to_raw());
                    self.symbol_registry.insert(key, sym);
                    return Ok(Some(sym));
                }
                "keyFor" => {
                    let target = arg(0);
                    let found = self
                        .symbol_registry
                        .iter()
                        .find(|(_, v)| self.realm.strict_equals(**v, target))
                        .map(|(k, _)| k.clone());
                    return Ok(Some(match found {
                        Some(k) => self.new_str(&k),
                        None => NanBox::undefined(),
                    }));
                }
                _ => {}
            }
        }
        // --- symbol instance: `sym.toString()` ---
        if let Some((desc, _)) = self.realm.symbol_at(handle)
            && method == "toString"
        {
            // A no-argument `Symbol()` has an empty (undefined) description.
            let shown = if desc.starts_with('\u{0}') { "" } else { &desc };
            return Ok(Some(self.new_str(&alloc::format!("Symbol({shown})"))));
        }
        // --- BigInt instance: `toString(radix)` / `valueOf` ---
        if let Some(big) = self.realm.bigint_at(handle) {
            match method {
                "toString" => {
                    let radix = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        10
                    } else {
                        // ToIntegerOrInfinity (TypeError for Symbol/BigInt radix),
                        // then a RangeError unless it is in [2, 36].
                        let r = self.coerce_to_integer_or_infinity(arg(0))?;
                        if !(2.0..=36.0).contains(&r) {
                            let m = self.new_str("toString() radix must be between 2 and 36");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        r as u32
                    };
                    return Ok(Some(self.new_str(&bigint_to_radix(&big, radix))));
                }
                "valueOf" => return Ok(Some(NanBox::handle(self.realm.new_bigint(big).to_raw()))),
                // Grouped base-10 form (no locale data, en-US-ish default).
                "toLocaleString" => {
                    return Ok(Some(
                        self.new_str(&group_thousands_str(&bigint_to_radix(&big, 10))),
                    ));
                }
                _ => {}
            }
        }
        // --- `Number.*` / `String.*` statics (on the constructor) ---
        match self.realm.native_at(handle) {
            Some(N_NUMBER) => {
                match method {
                    "isInteger" => {
                        let is_int = arg(0)
                            .as_number()
                            .is_some_and(|n| n.is_finite() && (n as i64) as f64 == n);
                        return Ok(Some(NanBox::boolean(is_int)));
                    }
                    "isSafeInteger" => {
                        let safe = arg(0).as_number().is_some_and(|n| {
                            n.is_finite()
                                && (n as i64) as f64 == n
                                && n.abs() <= 9_007_199_254_740_991.0
                        });
                        return Ok(Some(NanBox::boolean(safe)));
                    }
                    "isFinite" => {
                        return Ok(Some(NanBox::boolean(
                            arg(0).as_number().is_some_and(f64::is_finite),
                        )));
                    }
                    "isNaN" => {
                        return Ok(Some(NanBox::boolean(
                            arg(0).as_number().is_some_and(f64::is_nan),
                        )));
                    }
                    "parseFloat" => return Ok(Some(self.call_native(N_PARSE_FLOAT, args)?)),
                    "parseInt" => return Ok(Some(self.call_native(N_PARSE_INT, args)?)),
                    _ => {}
                };
            }
            Some(N_STRING) if method == "fromCharCode" => {
                // Each argument is ToUint16'd into a UTF-16 code unit; the resulting
                // sequence is decoded to WTF-8, so an adjacent high/low surrogate
                // pair combines into one astral code point and a **lone surrogate
                // is preserved** (DOMString semantics).
                let mut units: Vec<u16> = Vec::with_capacity(args.len());
                for a in args {
                    // ToNumber each argument (Symbol → TypeError), then ToUint16:
                    // truncate toward zero, mod 2^16.
                    let num = self.coerce_to_number(*a)?;
                    let n = self.realm.to_number(num);
                    units.push(if n.is_finite() {
                        (n as i64).rem_euclid(65536) as u16
                    } else {
                        0
                    });
                }
                return Ok(Some(self.new_str_bytes(crate::wtf8::from_utf16(&units))));
            }
            // `String.fromCodePoint(...cps)` — each argument is a full Unicode
            // code point (may be astral). A non-integer or out-of-range value is a
            // RangeError; a Symbol is a TypeError.
            Some(N_STRING) if method == "fromCodePoint" => {
                let mut out: Vec<u16> = Vec::new();
                for a in args {
                    let num = self.coerce_to_number(*a)?;
                    let n = self.realm.to_number(num);
                    if !n.is_finite()
                        || n != trunc_toward_zero(n)
                        || !(0.0..=0x10_FFFF as f64).contains(&n)
                    {
                        let m = self.new_str("Invalid code point");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    let cp = n as u32;
                    if cp <= 0xFFFF {
                        out.push(cp as u16);
                    } else {
                        let c = cp - 0x10000;
                        out.push(0xD800 + (c >> 10) as u16);
                        out.push(0xDC00 + (c & 0x3FF) as u16);
                    }
                }
                return Ok(Some(self.new_str_bytes(crate::wtf8::from_utf16(&out))));
            }
            // `String.raw(template, ...subs)` — interleave `ToString(template.raw[i])`
            // with `ToString(subs[i])`. `template.raw` is treated as an array-like
            // (length + indexed reads), not necessarily a real array.
            Some(N_STRING) if method == "raw" => {
                let cooked = self.coerce_to_object(arg(0));
                let Some(ch) = cooked.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                };
                let raw_v = self
                    .realm
                    .get_property(ch, "raw")
                    .unwrap_or(NanBox::undefined());
                let raw_obj = self.coerce_to_object(raw_v);
                let Some(rh) = raw_obj.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                };
                // ToLength(raw.length).
                let len_v = self.read_member(rh, "length")?;
                let lit_count = self.coerce_to_integer_or_infinity(len_v)?.max(0.0) as usize;
                let subs = &args[1.min(args.len())..];
                let mut out = String::new();
                for i in 0..lit_count {
                    let piece = self.read_member(rh, &alloc::format!("{i}"))?;
                    let bytes = self.coerce_to_string_bytes(piece)?;
                    out.push_str(&crate::wtf8::to_string_lossy(&bytes));
                    if i + 1 == lit_count {
                        break;
                    }
                    if let Some(s) = subs.get(i) {
                        let sb = self.coerce_to_string_bytes(*s)?;
                        out.push_str(&crate::wtf8::to_string_lossy(&sb));
                    }
                }
                return Ok(Some(self.new_str(&out)));
            }
            _ => {}
        }
        // --- ArrayBuffer.prototype.slice(begin?, end?) → a new ArrayBuffer copy ---
        if method == "slice"
            && let Some(bh) = self.array_buffer_bytes(handle)
        {
            self.guard_detached_buffer(handle)?;
            let bytes = self
                .realm
                .bytes_at(bh)
                .map(<[u8]>::to_vec)
                .unwrap_or_default();
            let len = bytes.len() as i64;
            let norm = |this: &mut Self, v: NanBox, default: i64| -> usize {
                if matches!(v.unpack(), Unpacked::Undefined) {
                    return default as usize;
                }
                let n = this.realm.to_number(v) as i64;
                usize::try_from(if n < 0 { (len + n).max(0) } else { n.min(len) }).unwrap_or(0)
            };
            let begin = norm(self, arg(0), 0);
            let end = norm(self, arg(1), len);
            let sub = bytes.get(begin..end.max(begin)).unwrap_or(&[]);
            let nb = self.make_array_buffer_from_bytes(sub);
            return Ok(Some(NanBox::handle(nb.to_raw())));
        }
        // --- ArrayBuffer.prototype.transfer(newLength?) / transferToFixedLength(newLength?)
        // → a new ArrayBuffer, detaching the original (its byteLength becomes 0 and its
        // views are emptied). `transfer` preserves resizability (the new buffer keeps the
        // original's maxByteLength); `transferToFixedLength` always yields a fixed-length
        // buffer. (ArrayBufferCopyAndDetach.) ---
        if (method == "transfer" || method == "transferToFixedLength")
            && let Some(bh) = self.array_buffer_bytes(handle)
        {
            // `newLength` is ToIndex-coerced first (before the detached check), so a
            // poisoned `valueOf` / out-of-range length is observed in spec order.
            let new_len = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                None
            } else {
                let raw = self.realm.to_number(arg(0));
                Some(self.validate_alloc_len(raw, "Invalid ArrayBuffer length")?)
            };
            if self
                .realm
                .get_property(handle, ARRAY_BUFFER_DETACHED)
                .is_some()
            {
                let m = self.new_str("Cannot transfer an already-detached ArrayBuffer");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            let mut bytes = self
                .realm
                .bytes_at(bh)
                .map(<[u8]>::to_vec)
                .unwrap_or_default();
            // `transfer()`/`transferToFixedLength()` keep the size; an explicit length
            // resizes (truncating or zero-padding the copy).
            let new_len = new_len.unwrap_or(bytes.len());
            bytes.resize(new_len, 0);
            let nb = self.make_array_buffer_from_bytes(&bytes);
            // `transfer` carries the original's resizability over (clamping the
            // preserved maxByteLength to be at least the new length); a non-resizable
            // source — and `transferToFixedLength` always — yields a fixed buffer.
            if method == "transfer"
                && let Some(maxv) = self.realm.get_property(handle, ARRAY_BUFFER_MAXLEN)
            {
                let max = (self.realm.to_number(maxv).max(0.0) as usize).max(new_len);
                self.realm
                    .set_hidden_property(nb, ARRAY_BUFFER_MAXLEN, NanBox::number(max as f64));
            }
            // Detach the original (empty its views, zero its store, flag it).
            self.detach_array_buffer(handle);
            return Ok(Some(NanBox::handle(nb.to_raw())));
        }
        // --- ArrayBuffer.prototype.resize(newByteLength) (resizable buffers) ---
        if method == "resize"
            && let Some(bytesv) = self.realm.get_property(handle, ARRAY_BUFFER_BYTES)
            && let Some(bh) = bytesv.as_handle().map(Handle::from_raw)
        {
            self.guard_detached_buffer(handle)?;
            let Some(max) = self
                .realm
                .get_property(handle, ARRAY_BUFFER_MAXLEN)
                .map(|m| self.realm.to_number(m) as usize)
            else {
                let m = self.new_str("ArrayBuffer.prototype.resize: buffer is not resizable");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            };
            let new_len = self.realm.to_number(arg(0)).max(0.0) as usize;
            if new_len > max {
                let m = self.new_str("ArrayBuffer.prototype.resize: length exceeds maxByteLength");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            self.realm.resize_buffer(bh, new_len);
            return Ok(Some(NanBox::undefined()));
        }
        // --- DataView get*/set* ---
        if let Some(bufv) = self.realm.get_property(handle, DATA_VIEW_BUF)
            && let Some((is_set, size, signed, is_float, is_bigint)) = dataview_method(method)
        {
            // GetViewValue / SetViewValue spec order:
            //   1. ToIndex(requestIndex)            (abrupt-propagating)
            //   2. ToBoolean(isLittleEndian)
            //   3. (set) ToBigInt/ToNumber(value)   (abrupt-propagating)
            //   4. IsDetachedBuffer → TypeError
            //   5. bounds check → RangeError
            //   6. read/write
            // ToIndex first: a negative/non-integer/over-2^53 offset is a
            // RangeError, a Symbol/BigInt offset a TypeError — *before* any
            // detached/bounds check or value coercion.
            let requested = self.coerce_to_index(arg(0))?;
            let le = self.realm.truthy(arg(if is_set { 2 } else { 1 }));
            // (set) coerce the value next (its side effects/throw run before the
            // detached and bounds checks).
            let set_bits: Option<u64> = if is_set {
                Some(if is_bigint {
                    let big = self.coerce_to_bigint(arg(1))?;
                    big.to_u64_wrapping()
                } else if is_float {
                    let num = self.coerce_to_number(arg(1))?;
                    let value = self.realm.to_number(num);
                    match size {
                        2 => u64::from(f64_to_f16_bits(value)),
                        4 => u64::from((value as f32).to_bits()),
                        _ => value.to_bits(),
                    }
                } else {
                    let num = self.coerce_to_number(arg(1))?;
                    let value = self.realm.to_number(num);
                    // SetValueInBuffer for an integer type takes the bytes of the value
                    // modulo 2^(8*size): a non-finite value (NaN/±Infinity) maps to 0,
                    // and a finite value is truncated toward zero then reduced into the
                    // type's width (e.g. `setUint8(0, 256)` stores 0, `setUint8(0, Infinity)`
                    // stores 0). Plain `as i64` would saturate Infinity to i64::MAX (0xFF…).
                    // (Integer DataView types are at most 4 bytes wide, so the truncated
                    // value fits an `i64` and the low `8*size` bits are the stored bytes;
                    // `trunc_toward_zero` avoids the std-only `f64::trunc` for `no_std`.)
                    if value.is_finite() {
                        let truncated = trunc_toward_zero(value) as i64 as u64;
                        let mask = if size >= 8 {
                            u64::MAX
                        } else {
                            (1u64 << (8 * size)) - 1
                        };
                        truncated & mask
                    } else {
                        0
                    }
                })
            } else {
                None
            };
            // IsDetachedBuffer: a detached buffer is a TypeError.
            if let Some(buf_h) = bufv.as_handle().map(Handle::from_raw) {
                self.guard_detached_buffer(buf_h)?;
            }
            let bytes_h = bufv
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.array_buffer_bytes(h));
            let Some(bh) = bytes_h else {
                let m = self.new_str("DataView has no buffer");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            };
            let base = self
                .realm
                .get_property(handle, DATA_VIEW_OFF)
                .and_then(|n| n.as_number())
                .unwrap_or(0.0) as usize;
            let total = self.realm.bytes_len(bh).unwrap_or(0);
            // M1: clamp the recorded view length to what the *live* buffer can back
            // (a resizable buffer may have shrunk under the view), so the access can
            // never run past the real bytes.
            let view_len = self
                .realm
                .get_property(handle, DATA_VIEW_LEN)
                .and_then(|n| n.as_number())
                .map_or(total.saturating_sub(base), |n| n as usize)
                .min(total.saturating_sub(base));
            // Bounds: getIndex + size must be <= the view's byte length, with
            // checked arithmetic (a huge offset must not wrap past the bound).
            let in_bounds = usize::try_from(requested).is_ok() && {
                let r = requested as usize;
                r.checked_add(size).is_some_and(|end| end <= view_len)
                    && base
                        .checked_add(r)
                        .and_then(|a| a.checked_add(size))
                        .is_some_and(|e| e <= total)
            };
            if !in_bounds {
                let m = self.new_str("Offset is outside the bounds of the DataView");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            let abs = base + requested as usize;
            if let Some(bits) = set_bits {
                if let Some(bytes) = self.realm.bytes_at_mut(bh) {
                    for i in 0..size {
                        let shift = if le { i } else { size - 1 - i };
                        let byte = ((bits >> (8 * shift)) & 0xff) as u8;
                        if let Some(slot) = bytes.get_mut(abs + i) {
                            *slot = byte;
                        }
                    }
                }
                // Aliasing is intrinsic: typed-array views over the same bytes see the
                // write with no propagation step.
                return Ok(Some(NanBox::undefined()));
            }
            let mut bits: u64 = 0;
            for i in 0..size {
                let b = self
                    .realm
                    .bytes_at(bh)
                    .and_then(|e| e.get(abs + i).copied())
                    .unwrap_or(0) as u64
                    & 0xff;
                let shift = if le { i } else { size - 1 - i };
                bits |= b << (8 * shift);
            }
            if is_bigint {
                // `getBigInt64` reinterprets the 64 bits as a signed i64; `getBigUint64`
                // as an unsigned u64 — both returned as a BigInt.
                let big = if signed {
                    crate::bignum::BigInt::from_i128(i128::from(bits as i64))
                } else {
                    crate::bignum::BigInt::from_i128(i128::from(bits))
                };
                return Ok(Some(NanBox::handle(self.realm.new_bigint(big).to_raw())));
            }
            let value = if is_float {
                match size {
                    2 => f16_to_f64(bits as u16),
                    4 => f64::from(f32::from_bits(bits as u32)),
                    _ => f64::from_bits(bits),
                }
            } else if signed && size < 8 && bits & (1 << (8 * size - 1)) != 0 {
                (bits as i64 - (1i64 << (8 * size))) as f64
            } else {
                bits as f64
            };
            return Ok(Some(NanBox::number(value)));
        }

        // --- Intl.NumberFormat / Intl.DateTimeFormat instance methods ---
        if self.realm.get_property(handle, "\u{0}intl").is_some() && method == "format" {
            let s = self.intl_format_value(handle, arg(0));
            return Ok(Some(self.new_str(&s)));
        }
        // --- Date instance methods ---
        if let Some(ms) = self.realm.date_at(handle) {
            // A user-overridden prototype method wins over the built-in dispatch
            // (e.g. `Date.prototype.toString = Object.prototype.toString`). If the
            // method resolves on the proto chain to anything other than a first-class
            // `Date.prototype` native, call that instead.
            if let Some(m) = self.realm.object_proto(handle).and_then(|p| {
                let mut cur = Some(p);
                while let Some(c) = cur {
                    if self.realm.has_own(c, method) {
                        return self.realm.get_property(c, method);
                    }
                    cur = self.realm.object_proto(c);
                }
                None
            }) && let Some(mh) = m.as_handle().map(Handle::from_raw)
                && self.realm.bound_native_at(mh).map(|(id, _)| id) != Some(N_DATE_PROTO_FN)
                && self.realm.native_at(mh) != Some(N_DATE_TO_JSON)
                && self.realm.native_at(mh) != Some(N_DATE_TO_PRIMITIVE)
                && self.is_callable(mh)
            {
                return Ok(Some(self.call_with_this(m, recv, args)?));
            }
            // An invalid (NaN) date: every numeric getter is `NaN` (the field
            // decomposition below would otherwise read garbage from `0`).
            if !ms.is_finite()
                && matches!(
                    method,
                    "getTime"
                        | "valueOf"
                        | "getFullYear"
                        | "getUTCFullYear"
                        | "getMonth"
                        | "getUTCMonth"
                        | "getDate"
                        | "getUTCDate"
                        | "getDay"
                        | "getUTCDay"
                        | "getHours"
                        | "getUTCHours"
                        | "getMinutes"
                        | "getUTCMinutes"
                        | "getSeconds"
                        | "getUTCSeconds"
                        | "getMilliseconds"
                        | "getUTCMilliseconds"
                        | "getTimezoneOffset"
                )
            {
                return Ok(Some(NanBox::number(f64::NAN)));
            }
            let t = ms as i64;
            let day = t.div_euclid(86_400_000);
            let tod = t.rem_euclid(86_400_000);
            let (y, mo, d) = crate::realm::civil_from_days(day);
            return Ok(Some(match method {
                // The engine models all dates in UTC, so `getUTC*` aliases `get*`.
                "getTime" | "valueOf" => NanBox::number(ms),
                "getFullYear" | "getUTCFullYear" => NanBox::number(y as f64),
                "getMonth" | "getUTCMonth" => NanBox::number((mo - 1) as f64), // 0-based
                "getDate" | "getUTCDate" => NanBox::number(d as f64),
                "getDay" | "getUTCDay" => {
                    NanBox::number((day.rem_euclid(7) + 4).rem_euclid(7) as f64)
                }
                "getHours" | "getUTCHours" => NanBox::number((tod / 3_600_000) as f64),
                "getMinutes" | "getUTCMinutes" => NanBox::number((tod / 60_000 % 60) as f64),
                "getSeconds" | "getUTCSeconds" => NanBox::number((tod / 1000 % 60) as f64),
                "getMilliseconds" | "getUTCMilliseconds" => NanBox::number((tod % 1000) as f64),
                // The engine models all dates in UTC, so the local offset is 0.
                "getTimezoneOffset" => NanBox::number(0.0),
                // `toISOString` throws on an invalid date; `toJSON` returns null.
                "toISOString" => {
                    if !ms.is_finite() {
                        let m = self.new_str("Invalid time value");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    self.new_str(&crate::realm::date_to_iso(ms))
                }
                "toJSON" => {
                    if ms.is_finite() {
                        self.new_str(&crate::realm::date_to_iso(ms))
                    } else {
                        NanBox::null()
                    }
                }
                // Human-readable forms (the engine is UTC, so `GMT+0000`).
                "toDateString" | "toTimeString" | "toString" | "toUTCString"
                | "toLocaleDateString" | "toLocaleTimeString" | "toLocaleString" => {
                    // An invalid date (NaN timestamp) stringifies as "Invalid Date".
                    if !ms.is_finite() {
                        return Ok(Some(self.new_str("Invalid Date")));
                    }
                    let wd = WEEKDAYS[((day.rem_euclid(7) + 4).rem_euclid(7)) as usize];
                    let mn = MONTHS[(mo - 1) as usize];
                    let (hh, mi, ss) = (tod / 3_600_000, tod / 60_000 % 60, tod / 1000 % 60);
                    // The year is zero-padded to at least 4 digits, with a leading
                    // sign for negative years (`-0001`, `-123456`).
                    let yr = format_date_year(y);
                    let date_str = alloc::format!("{wd} {mn} {d:02} {yr}");
                    let time_str = alloc::format!(
                        "{hh:02}:{mi:02}:{ss:02} GMT+0000 (Coordinated Universal Time)"
                    );
                    let s = match method {
                        "toDateString" => date_str,
                        "toTimeString" => time_str,
                        "toUTCString" => {
                            alloc::format!("{wd}, {d:02} {mn} {yr} {hh:02}:{mi:02}:{ss:02} GMT")
                        }
                        "toLocaleDateString" => alloc::format!("{mo}/{d}/{y}"),
                        "toLocaleTimeString" => alloc::format!("{hh:02}:{mi:02}:{ss:02}"),
                        "toLocaleString" => {
                            alloc::format!("{mo}/{d}/{y}, {hh:02}:{mi:02}:{ss:02}")
                        }
                        // `toString`
                        _ => alloc::format!("{date_str} {time_str}"),
                    };
                    self.new_str(&s)
                }
                // --- `set*` mutators (all UTC; a setter returns the new time) ---
                "setTime" => {
                    // ToNumber(time) → TimeClip.
                    let raw = self.coerce_to_number(arg(0))?;
                    let nms = time_clip(self.realm.to_number(raw));
                    self.realm.set_date_ms(handle, nms);
                    NanBox::number(nms)
                }
                "setFullYear" | "setUTCFullYear" | "setMonth" | "setUTCMonth" | "setDate"
                | "setUTCDate" | "setHours" | "setUTCHours" | "setMinutes" | "setUTCMinutes"
                | "setSeconds" | "setUTCSeconds" | "setMilliseconds" | "setUTCMilliseconds" => {
                    // The number of components this setter consumes, in order.
                    let max_components = match method {
                        "setHours" | "setUTCHours" => 4,
                        "setFullYear" | "setUTCFullYear" | "setMinutes" | "setUTCMinutes" => 3,
                        "setMonth" | "setUTCMonth" | "setSeconds" | "setUTCSeconds" => 2,
                        _ => 1, // setDate / setMilliseconds
                    };
                    // `setFullYear` works on an invalid date (treating the time as
                    // +0); the others propagate NaN. The date value is read *before*
                    // coercing the arguments.
                    let is_full_year = matches!(method, "setFullYear" | "setUTCFullYear");
                    let date_is_nan = !ms.is_finite();
                    // The primary component is always ToNumber'd (an absent argument
                    // is `undefined` → NaN); the trailing optional components only
                    // when actually supplied. Coercion is in order, exactly once
                    // each, even when the date is NaN (a later abrupt completion
                    // still throws). `setHours()` with no args therefore yields NaN.
                    let take = max_components.min(args.len().max(1));
                    let mut comps = Vec::with_capacity(take);
                    for i in 0..take {
                        let num = self.coerce_to_number(arg(i))?;
                        comps.push(self.realm.to_number(num));
                    }
                    if date_is_nan && !is_full_year {
                        // Invalid date stays invalid; arguments were still coerced.
                        return Ok(Some(NanBox::number(f64::NAN)));
                    }
                    // Decompose the (possibly zeroed, for setFullYear) current time.
                    let (mut yy, mut mo0, mut dd) = (y, (mo as i64) - 1, d as i64);
                    let mut hh = tod / 3_600_000;
                    let mut mi = tod / 60_000 % 60;
                    let mut ss = tod / 1000 % 60;
                    let mut mss = tod % 1000;
                    if is_full_year && date_is_nan {
                        // Time treated as +0: all components reset to their epoch value.
                        yy = 1970;
                        mo0 = 0;
                        dd = 1;
                        hh = 0;
                        mi = 0;
                        ss = 0;
                        mss = 0;
                    }
                    // Any NaN component makes the whole result NaN (TimeClip).
                    let mut any_nan = false;
                    let mut comp = |slot: &mut i64, idx: usize| {
                        if let Some(&v) = comps.get(idx) {
                            if v.is_nan() || !v.is_finite() {
                                any_nan = true;
                            }
                            *slot = v as i64;
                        }
                    };
                    match method {
                        "setFullYear" | "setUTCFullYear" => {
                            comp(&mut yy, 0);
                            comp(&mut mo0, 1);
                            comp(&mut dd, 2);
                        }
                        "setMonth" | "setUTCMonth" => {
                            comp(&mut mo0, 0);
                            comp(&mut dd, 1);
                        }
                        "setDate" | "setUTCDate" => comp(&mut dd, 0),
                        "setHours" | "setUTCHours" => {
                            comp(&mut hh, 0);
                            comp(&mut mi, 1);
                            comp(&mut ss, 2);
                            comp(&mut mss, 3);
                        }
                        "setMinutes" | "setUTCMinutes" => {
                            comp(&mut mi, 0);
                            comp(&mut ss, 1);
                            comp(&mut mss, 2);
                        }
                        "setSeconds" | "setUTCSeconds" => {
                            comp(&mut ss, 0);
                            comp(&mut mss, 1);
                        }
                        _ => comp(&mut mss, 0), // setMilliseconds
                    }
                    if any_nan {
                        self.realm.set_date_ms(handle, f64::NAN);
                        return Ok(Some(NanBox::number(f64::NAN)));
                    }
                    // Normalize a possibly out-of-range month into the year, then
                    // measure the day as an offset from the 1st (so out-of-range
                    // day/hour/… values roll over via plain integer arithmetic).
                    let yy2 = yy + mo0.div_euclid(12);
                    let mo1 = (mo0.rem_euclid(12) + 1) as u32;
                    let base_days = crate::realm::days_from_civil(yy2, mo1, 1) + (dd - 1);
                    let nms = time_clip(
                        (base_days * 86_400_000 + hh * 3_600_000 + mi * 60_000 + ss * 1000 + mss)
                            as f64,
                    );
                    self.realm.set_date_ms(handle, nms);
                    NanBox::number(nms)
                }
                _ => return Ok(None),
            }));
        }
        // --- RegExp instance methods (`exec`/`test`/`compile`/`toString`) ---
        // These now resolve as first-class `RegExp.prototype` methods (so a user
        // `re.exec` override and the Get/Set `lastIndex` semantics are honored),
        // so `call_method` does NOT intercept them — it returns `None` and the
        // caller reads the inherited prototype method and invokes it. We only keep
        // the symbol-method delegation below for `str.match(re)` etc.
        // `Map.groupBy(items, cb)` — like `Object.groupBy` but a Map (keys are
        // the callback's return value as-is, so objects work as group keys).
        if self.realm.native_at(handle) == Some(N_MAP) && method == "groupBy" {
            let items = self.iterate_values(arg(0))?;
            let cb = arg(1);
            let map = self.realm.new_collection(false);
            for (i, item) in items.iter().enumerate() {
                let key = self.call(cb, &[*item, NanBox::number(i as f64)])?;
                let bucket = match self
                    .realm
                    .collection_get(map, key)
                    .and_then(NanBox::as_handle)
                    .map(Handle::from_raw)
                {
                    Some(h) => h,
                    None => {
                        let arr = self.realm.new_array(Vec::new());
                        self.realm
                            .collection_set(map, key, NanBox::handle(arr.to_raw()));
                        arr
                    }
                };
                self.realm.array_push(bucket, *item);
            }
            return Ok(Some(NanBox::handle(map.to_raw())));
        }
        // --- `Promise.resolve` / `Promise.reject` statics (on the constructor) ---
        if self.realm.native_at(handle) == Some(N_PROMISE) {
            match method {
                "resolve" => {
                    // `Promise.resolve(x)` is idempotent on a promise: if `x` is
                    // already a promise, return it unchanged (same identity).
                    if let Some(raw) = arg(0).as_handle()
                        && self.realm.promise_state(Handle::from_raw(raw)).is_some()
                    {
                        return Ok(Some(arg(0)));
                    }
                    let p = self.fresh_promise();
                    self.resolve_with(p, arg(0));
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                "reject" => {
                    let p = self.fresh_promise();
                    self.settle(p, arg(0), false);
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.withResolvers()` → `{ promise, resolve, reject }`.
                "withResolvers" => {
                    let p = self.fresh_promise();
                    let resolve = self.realm.new_bound_native(N_RESOLVE, p);
                    let reject = self.realm.new_bound_native(N_REJECT, p);
                    let obj = self.realm.new_object();
                    self.realm
                        .set_property(obj, "promise", NanBox::handle(p.to_raw()));
                    self.realm
                        .set_property(obj, "resolve", NanBox::handle(resolve.to_raw()));
                    self.realm
                        .set_property(obj, "reject", NanBox::handle(reject.to_raw()));
                    return Ok(Some(NanBox::handle(obj.to_raw())));
                }
                // `Promise.all(iterable)`: resolve with the array of awaited
                // values, or reject with the first rejection (eager model).
                "all" => {
                    let p = self.fresh_promise();
                    // GetIterator / iteration sit inside IfAbruptRejectPromise: an
                    // abrupt completion (a non-iterable argument, a throwing
                    // `Symbol.iterator`/`next`) rejects the result promise rather
                    // than throwing out of `Promise.all`.
                    let items = match self.iterate_values(arg(0)) {
                        Ok(v) => v,
                        Err(ExecError::Throw(e)) => {
                            self.settle(p, e, false);
                            return Ok(Some(NanBox::handle(p.to_raw())));
                        }
                        Err(other) => return Err(other),
                    };
                    let mut results = Vec::with_capacity(items.len());
                    for item in items {
                        match self.await_value(item) {
                            Ok(v) => results.push(v),
                            Err(ExecError::Throw(e)) => {
                                self.settle(p, e, false);
                                return Ok(Some(NanBox::handle(p.to_raw())));
                            }
                            Err(other) => return Err(other),
                        }
                    }
                    let arr = self.realm.new_array(results);
                    self.resolve_with(p, NanBox::handle(arr.to_raw()));
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.race(iterable)`: settle with the first input to *settle*.
                // Steps the event loop, checking the inputs after each task, so a
                // timer-backed promise that settles first wins (ties in a single step
                // broken by list order).
                "race" => {
                    let p = self.fresh_promise();
                    let items = match self.iterate_values(arg(0)) {
                        Ok(v) => v,
                        Err(ExecError::Throw(e)) => {
                            self.settle(p, e, false);
                            return Ok(Some(NanBox::handle(p.to_raw())));
                        }
                        Err(other) => return Err(other),
                    };
                    'race: loop {
                        for item in &items {
                            match self.settled_state(*item) {
                                Some(Ok(v)) => {
                                    self.resolve_with(p, v);
                                    break 'race;
                                }
                                Some(Err(e)) => {
                                    self.settle(p, e, false);
                                    break 'race;
                                }
                                None => {}
                            }
                        }
                        // None settled yet: advance the loop, or stop if it is idle
                        // (the race promise then stays pending, as the spec requires).
                        if self.microtasks.is_empty() && self.macrotasks.is_empty() {
                            break;
                        }
                        if self.microtasks.is_empty() {
                            self.run_one_macrotask()?;
                        } else {
                            self.run_one_microtask()?;
                        }
                    }
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.allSettled(iterable)`: never rejects; each entry is
                // `{status, value}` or `{status, reason}`.
                "allSettled" => {
                    let p = self.fresh_promise();
                    let items = match self.iterate_values(arg(0)) {
                        Ok(v) => v,
                        Err(ExecError::Throw(e)) => {
                            self.settle(p, e, false);
                            return Ok(Some(NanBox::handle(p.to_raw())));
                        }
                        Err(other) => return Err(other),
                    };
                    let mut results = Vec::with_capacity(items.len());
                    for item in items {
                        let obj = self.realm.new_object();
                        match self.await_value(item) {
                            Ok(v) => {
                                let s = self.new_str("fulfilled");
                                self.realm.set_property(obj, "status", s);
                                self.realm.set_property(obj, "value", v);
                            }
                            Err(ExecError::Throw(e)) => {
                                let s = self.new_str("rejected");
                                self.realm.set_property(obj, "status", s);
                                self.realm.set_property(obj, "reason", e);
                            }
                            Err(other) => return Err(other),
                        }
                        results.push(NanBox::handle(obj.to_raw()));
                    }
                    let arr = self.realm.new_array(results);
                    self.resolve_with(p, NanBox::handle(arr.to_raw()));
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.any(iterable)`: fulfills with the first input to
                // fulfill; rejects with an `AggregateError` if all reject.
                "any" => {
                    let p = self.fresh_promise();
                    let items = match self.iterate_values(arg(0)) {
                        Ok(v) => v,
                        Err(ExecError::Throw(e)) => {
                            self.settle(p, e, false);
                            return Ok(Some(NanBox::handle(p.to_raw())));
                        }
                        Err(other) => return Err(other),
                    };
                    let mut errors = Vec::new();
                    for item in items {
                        match self.await_value(item) {
                            Ok(v) => {
                                self.resolve_with(p, v);
                                return Ok(Some(NanBox::handle(p.to_raw())));
                            }
                            Err(ExecError::Throw(e)) => errors.push(e),
                            Err(other) => return Err(other),
                        }
                    }
                    // None fulfilled: reject with an AggregateError holding them.
                    let agg = self.realm.new_object();
                    let name = self.new_str("AggregateError");
                    self.realm.set_property(agg, "name", name);
                    let msg = self.new_str("All promises were rejected");
                    self.realm.set_property(agg, "message", msg);
                    let errs = self.realm.new_array(errors);
                    self.realm
                        .set_property(agg, "errors", NanBox::handle(errs.to_raw()));
                    self.settle(p, NanBox::handle(agg.to_raw()), false);
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                _ => {}
            }
        }
        // --- promise instance methods (`then`/`catch`/`finally`) ---
        if self.realm.promise_state(handle).is_some() {
            match method {
                "then" => return Ok(Some(self.promise_then(handle, arg(0), arg(1)))),
                "catch" => {
                    return Ok(Some(self.promise_then(handle, NanBox::undefined(), arg(0))));
                }
                "finally" => {
                    // The callback runs on either settlement for side effects; the
                    // original value/rejection passes through to the new promise.
                    let cb = arg(0);
                    let result = self.register_then(handle, cb, cb, true);
                    return Ok(Some(NanBox::handle(result.to_raw())));
                }
                _ => {}
            }
        }

        // A custom matcher/replacer: when the argument defines the matching
        // well-known symbol method (`Symbol.match`/`replace`/`search`/`split`/
        // `matchAll`), `str.method(obj)` delegates to `obj[@@method](str, …rest)`.
        // (A RegExp argument now resolves its `@@method` through `RegExp.prototype`,
        // so this is the spec path for `"…".match(/re/)` etc.)
        if self.realm.string_value(handle).is_some()
            && let Some(sym_name) = match method {
                "match" => Some("match"),
                "matchAll" => Some("matchAll"),
                "search" => Some("search"),
                "replace" | "replaceAll" => Some("replace"),
                "split" => Some("split"),
                _ => None,
            }
            && let Some(argh) = arg(0).as_handle().map(Handle::from_raw)
        {
            // `replaceAll`/`matchAll` first require that a RegExp `searchValue`/
            // `regexp` be global (`IsRegExp` + `Get(flags)` not containing "g" →
            // TypeError), checked *before* dispatching the symbol method.
            if matches!(method, "replaceAll" | "matchAll") && self.is_regexp_arg(arg(0)) {
                let flags_v = self.read_member(argh, "flags")?;
                if matches!(flags_v.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(self.type_error(&alloc::format!(
                        "String.prototype.{method} called with a non-global RegExp argument"
                    )));
                }
                let flags_s = self.coerce_to_string(flags_v)?;
                if !flags_s.contains('g') {
                    return Err(self.type_error(&alloc::format!(
                        "String.prototype.{method} called with a non-global RegExp argument"
                    )));
                }
            }
            let sym = self.well_known_symbol(sym_name);
            let key = self.member_key(sym);
            let m = self.read_member(argh, &key)?;
            if m.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                // Pass the receiver string value itself (a String already), so a
                // surrogate-bearing subject reaches the symbol method losslessly
                // (`new_str(&string_value)` would replacement-char a lone surrogate).
                let this_str = NanBox::handle(handle.to_raw());
                let mut call_args = alloc::vec![this_str];
                call_args.extend_from_slice(&args[1.min(args.len())..]);
                return Ok(Some(self.call_with_this(m, arg(0), &call_args)?));
            }
        }

        // `String.prototype.{match,matchAll,search}` with a non-RegExp argument
        // (incl. `undefined`/`null`/a string/number) constructs `RegExp(arg, flags)`
        // (matchAll forces the global flag) and delegates to its `@@method`, per
        // spec — so `"abc".match()` matches the empty pattern and a coerced
        // `toString` is honored. (`replace`/`replaceAll`/`split` keep treating a
        // non-RegExp argument literally, handled in the string-methods block below.)
        if self.realm.string_value(handle).is_some()
            && let Some(sym_name) = match method {
                "match" => Some("match"),
                "matchAll" => Some("matchAll"),
                "search" => Some("search"),
                _ => None,
            }
        {
            let regexp_ctor = self.current.get("RegExp").unwrap_or(NanBox::undefined());
            let ctor_args: alloc::vec::Vec<NanBox> = if method == "matchAll" {
                alloc::vec![arg(0), self.new_str("g")]
            } else {
                alloc::vec![arg(0)]
            };
            let rx = self.construct(regexp_ctor, &ctor_args)?;
            let Some(rxh) = rx.as_handle().map(Handle::from_raw) else {
                return Ok(Some(NanBox::null()));
            };
            let sym = self.well_known_symbol(sym_name);
            let key = self.member_key(sym);
            let m = self.read_member(rxh, &key)?;
            // The receiver string value itself (lossless).
            let this_str = NanBox::handle(handle.to_raw());
            return Ok(Some(self.call_with_this(m, rx, &[this_str])?));
        }

        // --- string methods ---
        if let Some(bytes) = self.realm.string_bytes(handle) {
            // The lossless WTF-8 bytes — used by the UTF-16-unit-correct ops
            // (length/index/slice/search/pad/for-of) and the surrogate-aware
            // case/normalize ops. Most methods read only `bytes`; the few that take
            // an `&str` (`trim`/`replace`/`search`/`localeCompare`, …) build the
            // lossy `String` on demand inside their own arm via
            // `wtf8::to_string_lossy(&bytes)`, so the common path no longer pays for
            // a second full rope flatten plus a lossy decode on every call.
            let out = match method {
                // The locale variants behave like the locale-independent ones here
                // (no locale-specific case tailoring). A surrogate-free string
                // takes the `&str` fast path (byte-identical to before); a
                // surrogate-bearing string maps case over the code-point view,
                // passing lone surrogates through unchanged (a surrogate has no
                // case) so they survive the round-trip.
                "toUpperCase" | "toLocaleUpperCase" => {
                    Some(self.new_str_bytes(case_map_wtf8(&bytes, true)))
                }
                "toLowerCase" | "toLocaleLowerCase" => {
                    Some(self.new_str_bytes(case_map_wtf8(&bytes, false)))
                }
                "trim" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    Some(self.new_str(s.trim()))
                }
                "charAt" => {
                    // UTF-16-indexed: the unit at `i` as a one-unit string,
                    // preserving a lone surrogate (stored via WTF-8). A negative
                    // index is out of range (`NaN`/no-arg → 0).
                    let idx = self.coerce_to_integer_or_infinity(arg(0))?;
                    let out = match str_char_index(idx) {
                        Some(i) => crate::wtf8::utf16_index(&bytes, i)
                            .map(|u| crate::wtf8::from_utf16(&[u]))
                            .unwrap_or_default(),
                        None => Vec::new(),
                    };
                    Some(self.new_str_bytes(out))
                }
                "includes" => {
                    // A RegExp `searchString` is a TypeError (IsRegExp).
                    if self.is_regexp_arg(arg(0)) {
                        return Err(self.type_error(
                            "String.prototype.includes argument must not be a regular expression",
                        ));
                    }
                    let needle = self.arg_string_bytes_fallible(arg(0))?;
                    let units = crate::wtf8::utf16_len(&bytes);
                    let pos = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize).min(units)
                    };
                    Some(NanBox::boolean(index_of_units(&bytes, &needle, pos) >= 0.0))
                }
                "indexOf" => {
                    let needle = self.arg_string_bytes_fallible(arg(0))?;
                    // An optional `fromIndex` (UTF-16 unit offset) starts the search.
                    let units = crate::wtf8::utf16_len(&bytes);
                    let from = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize).min(units)
                    };
                    Some(NanBox::number(index_of_units(&bytes, &needle, from)))
                }
                "repeat" => {
                    // A negative or `+Infinity` count is a `RangeError`; a finite
                    // count whose product with the length overflows would panic, so
                    // it is a `RangeError` too (an unrepresentable string length).
                    let nf = self.realm.to_number(arg(0));
                    let n = nf as usize;
                    // A product that fits `usize` can still be enormous
                    // (`"x".repeat(2**40)` ≈ 1 TB); cap the result length too.
                    let total = n.checked_mul(bytes.len());
                    let max_string_len = self.realm.limits.max_string_len;
                    if nf < 0.0 || nf.is_infinite() || total.is_none_or(|t| t > max_string_len) {
                        let m = self.new_str("Invalid string length");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    // Repeat the WTF-8 bytes so a surrogate-bearing string repeats
                    // losslessly.
                    Some(self.new_str_bytes(bytes.repeat(n)))
                }
                "startsWith" => {
                    if self.is_regexp_arg(arg(0)) {
                        return Err(self.type_error(
                            "String.prototype.startsWith argument must not be a regular expression",
                        ));
                    }
                    let needle = self.arg_string_bytes_fallible(arg(0))?;
                    let units = crate::wtf8::utf16_len(&bytes);
                    let pos = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize).min(units)
                    };
                    // A prefix match at exactly `pos` units.
                    let start_byte = unit_to_byte(&bytes, pos);
                    let matched = bytes.len() - start_byte >= needle.len()
                        && bytes[start_byte..start_byte + needle.len()] == needle[..];
                    Some(NanBox::boolean(matched))
                }
                "endsWith" => {
                    if self.is_regexp_arg(arg(0)) {
                        return Err(self.type_error(
                            "String.prototype.endsWith argument must not be a regular expression",
                        ));
                    }
                    let needle = self.arg_string_bytes_fallible(arg(0))?;
                    let units = crate::wtf8::utf16_len(&bytes);
                    // `endPosition` defaults to the full length.
                    let end = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        units
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize).min(units)
                    };
                    let end_byte = unit_to_byte(&bytes, end);
                    let matched = end_byte >= needle.len()
                        && bytes[end_byte - needle.len()..end_byte] == needle[..];
                    Some(NanBox::boolean(matched))
                }
                "slice" => {
                    // UTF-16-unit range, surrogate-boundary correct. Both indices are
                    // ToIntegerOrInfinity (each runs `valueOf`, propagating throws).
                    let units = crate::wtf8::utf16_len(&bytes);
                    let start = self.coerce_to_integer_or_infinity(arg(0))?;
                    let end = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        units as f64
                    } else {
                        self.coerce_to_integer_or_infinity(arg(1))?
                    };
                    let idx = |n: f64| -> usize {
                        if n < 0.0 {
                            (units as f64 + n).max(0.0) as usize
                        } else {
                            (n as usize).min(units)
                        }
                    };
                    let a = idx(start);
                    let b = idx(end);
                    let (a, b) = if a < b { (a, b) } else { (a, a) };
                    Some(self.new_str_bytes(crate::wtf8::slice_utf16(&bytes, a, b)))
                }
                "split" => {
                    // `limit` is ToUint32 (undefined → 2^32-1). A limit of 0 yields
                    // an empty array.
                    let limit = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        u32::MAX
                    } else {
                        // ToUint32 without the std-only `trunc` (so `no_std` builds):
                        // truncate toward zero into i64, then take the low 32 bits.
                        let n = self.realm.to_number(arg(1));
                        if n.is_finite() {
                            (n as i64).rem_euclid(4_294_967_296) as u32
                        } else {
                            0
                        }
                    } as usize;
                    if limit == 0 {
                        return Ok(Some(NanBox::handle(
                            self.realm.new_array(Vec::new()).to_raw(),
                        )));
                    }
                    // An `undefined` separator returns the whole string as the sole
                    // element.
                    if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        let whole = self.new_str_bytes(bytes.clone());
                        let arr = self.realm.new_array(alloc::vec![whole]);
                        return Ok(Some(NanBox::handle(arr.to_raw())));
                    }
                    let sep = self.arg_string_bytes_fallible(arg(0))?;
                    let mut parts: Vec<NanBox> = if sep.is_empty() {
                        // Empty separator → one entry per UTF-16 code unit (a lone
                        // surrogate is its own one-unit entry).
                        let units = crate::wtf8::utf16_len(&bytes);
                        (0..units)
                            .map(|i| crate::wtf8::slice_utf16(&bytes, i, i + 1))
                            .map(|b| self.new_str_bytes(b))
                            .collect()
                    } else {
                        split_units(&bytes, &sep)
                            .into_iter()
                            .map(|b| self.new_str_bytes(b))
                            .collect()
                    };
                    parts.truncate(limit);
                    Some(NanBox::handle(self.realm.new_array(parts).to_raw()))
                }
                "replace" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    // ToString the searchValue, propagating a throwing user
                    // `toString` (the RegExp-argument form is not modeled here).
                    let from = self.coerce_to_string(arg(0))?;
                    let repl = arg(1);
                    let is_fn = repl
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
                    if is_fn {
                        match s.find(&from) {
                            Some(pos) => {
                                let m = self.new_str(&from);
                                // The match position is a UTF-16 unit index.
                                let off = NanBox::number(s[..pos].encode_utf16().count() as f64);
                                let whole = self.new_str(&s);
                                let r = self.call(repl, &[m, off, whole])?;
                                let rs = self.realm.to_display_string(r);
                                let out =
                                    alloc::format!("{}{}{}", &s[..pos], rs, &s[pos + from.len()..]);
                                Some(self.new_str(&out))
                            }
                            None => Some(self.new_str(&s)),
                        }
                    } else {
                        let to = self.realm.to_display_string(repl);
                        match s.find(&from) {
                            Some(pos) => {
                                let before = &s[..pos];
                                let after = &s[pos + from.len()..];
                                let mid = expand_dollar(&to, &from, before, after);
                                Some(self.new_str(&alloc::format!("{before}{mid}{after}")))
                            }
                            None => Some(self.new_str(&s)),
                        }
                    }
                }
                "replaceAll" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    let from = self.coerce_to_string(arg(0))?;
                    let repl = arg(1);
                    let is_fn = repl
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
                    if is_fn && !from.is_empty() {
                        let mut out = String::new();
                        let mut last = 0;
                        // P6: maintain a running UTF-16 unit count for the match
                        // offset rather than re-encoding `s[..abs]` from the start on
                        // every match (which is O(n²) across many matches). Each step
                        // only counts the units of the gap `s[last..abs]`.
                        let mut units_to_last = 0usize;
                        while let Some(rel) = s[last..].find(&from) {
                            let abs = last + rel;
                            out.push_str(&s[last..abs]);
                            let m = self.new_str(&from);
                            // The match position is a UTF-16 unit index.
                            let off_units = units_to_last + s[last..abs].encode_utf16().count();
                            let off = NanBox::number(off_units as f64);
                            let whole = self.new_str(&s);
                            let r = self.call(repl, &[m, off, whole])?;
                            out.push_str(&self.realm.to_display_string(r));
                            units_to_last = off_units + from.encode_utf16().count();
                            last = abs + from.len();
                        }
                        out.push_str(&s[last..]);
                        Some(self.new_str(&out))
                    } else if from.is_empty() {
                        let to = self.realm.to_display_string(repl);
                        Some(self.new_str(&s.replace(&from, &to)))
                    } else {
                        let to = self.realm.to_display_string(repl);
                        let mut out = String::new();
                        let mut last = 0;
                        while let Some(rel) = s[last..].find(&from) {
                            let abs = last + rel;
                            out.push_str(&s[last..abs]);
                            let after = &s[abs + from.len()..];
                            out.push_str(&expand_dollar(&to, &from, &s[..abs], after));
                            last = abs + from.len();
                        }
                        out.push_str(&s[last..]);
                        Some(self.new_str(&out))
                    }
                }
                "at" => {
                    let i = self.coerce_to_integer_or_infinity(arg(0))?;
                    // UTF-16-indexed with negative-from-end support.
                    let units = crate::wtf8::utf16_len(&bytes);
                    let idx = if i < 0.0 { units as f64 + i } else { i };
                    Some(
                        match as_index(idx).and_then(|u| crate::wtf8::utf16_index(&bytes, u)) {
                            Some(u) => self.new_str_bytes(crate::wtf8::from_utf16(&[u])),
                            None => NanBox::undefined(),
                        },
                    )
                }
                "substring" => {
                    let len = crate::wtf8::utf16_len(&bytes);
                    let clamp = |n: f64| (n.max(0.0) as usize).min(len);
                    let mut a = clamp(self.coerce_to_integer_or_infinity(arg(0))?);
                    let mut b = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        clamp(self.coerce_to_integer_or_infinity(arg(1))?)
                    };
                    if a > b {
                        core::mem::swap(&mut a, &mut b);
                    }
                    Some(self.new_str_bytes(crate::wtf8::slice_utf16(&bytes, a, b)))
                }
                "substr" => {
                    let len = crate::wtf8::utf16_len(&bytes);
                    let lenf = len as f64;
                    let start = self.coerce_to_integer_or_infinity(arg(0))?;
                    let start = if start < 0.0 {
                        (lenf + start).max(0.0)
                    } else {
                        start.min(lenf)
                    } as usize;
                    let count = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        len - start
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize)
                            .min(len - start)
                    };
                    Some(self.new_str_bytes(crate::wtf8::slice_utf16(&bytes, start, start + count)))
                }
                "trimStart" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    Some(self.new_str(s.trim_start()))
                }
                "trimEnd" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    Some(self.new_str(s.trim_end()))
                }
                // A string's `toString`/`valueOf` is the string itself.
                "toString" | "valueOf" => Some(recv),
                // `isWellFormed`/`toWellFormed`: a string is well-formed iff it has
                // no lone surrogate. The WTF-8 bytes are valid UTF-8 exactly then.
                "isWellFormed" => Some(NanBox::boolean(crate::wtf8::is_utf8(&bytes))),
                "toWellFormed" => {
                    // Replace each lone surrogate with U+FFFD (the lossy decode).
                    Some(self.new_str(&crate::wtf8::to_string_lossy(&bytes)))
                }
                // `charCodeAt(i)` is the UTF-16 code unit at index `i` (NaN if
                // out of range); a surrogate half reads as that 16-bit value.
                "charCodeAt" => {
                    // A negative or out-of-range index is `NaN` (`NaN`/no-arg → 0).
                    let idx = self.coerce_to_integer_or_infinity(arg(0))?;
                    let unit =
                        str_char_index(idx).and_then(|i| crate::wtf8::utf16_index(&bytes, i));
                    Some(unit.map_or(NanBox::number(f64::NAN), |u| NanBox::number(f64::from(u))))
                }
                // `codePointAt(i)` combines a surrogate pair at UTF-16 index `i`.
                "codePointAt" => {
                    let idx = self.coerce_to_integer_or_infinity(arg(0))?;
                    let Some(i) = str_char_index(idx) else {
                        return Ok(Some(NanBox::undefined()));
                    };
                    Some(match crate::wtf8::utf16_index(&bytes, i) {
                        Some(u) if (0xD800..0xDC00).contains(&u) => {
                            match crate::wtf8::utf16_index(&bytes, i + 1) {
                                Some(low) if (0xDC00..0xE000).contains(&low) => {
                                    let cp = 0x1_0000
                                        + ((u32::from(u) - 0xD800) << 10)
                                        + (u32::from(low) - 0xDC00);
                                    NanBox::number(f64::from(cp))
                                }
                                _ => NanBox::number(f64::from(u)),
                            }
                        }
                        Some(u) => NanBox::number(f64::from(u)),
                        None => NanBox::undefined(),
                    })
                }
                "padStart" => {
                    // Spec order: ToLength(maxLength) then ToString(fillString).
                    let tn = self.coerce_to_integer_or_infinity(arg(0))?;
                    let target = self.pad_target(tn)?;
                    let pad = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        alloc::vec![b' ']
                    } else {
                        self.arg_string_bytes_fallible(arg(1))?
                    };
                    Some(self.new_str_bytes(pad_units(&bytes, target, &pad, true)))
                }
                "padEnd" => {
                    let tn = self.coerce_to_integer_or_infinity(arg(0))?;
                    let target = self.pad_target(tn)?;
                    let pad = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        alloc::vec![b' ']
                    } else {
                        self.arg_string_bytes_fallible(arg(1))?
                    };
                    Some(self.new_str_bytes(pad_units(&bytes, target, &pad, false)))
                }
                "lastIndexOf" => {
                    let needle = self.arg_string_bytes_fallible(arg(0))?;
                    // `fromIndex` (a UTF-16 unit index): the match may *start* at or
                    // before it; `undefined`/`NaN` mean +Infinity (whole string).
                    // ToNumber (a Symbol throws); NaN → whole string.
                    let pos_num = self.coerce_to_number(arg(1))?;
                    let n = self.realm.to_number(pos_num);
                    let from = if n.is_nan() {
                        usize::MAX
                    } else {
                        n.max(0.0).min(usize::MAX as f64) as usize
                    };
                    Some(NanBox::number(last_index_of_units(&bytes, &needle, from)))
                }
                // `concat` appends each argument's string form (WTF-8 bytes, so a
                // surrogate-bearing receiver or argument concatenates losslessly).
                "concat" => {
                    let mut out = bytes.clone();
                    for a in args {
                        // ToString each argument, honoring a user `toString`.
                        let p = self.coerce_object(*a, "string")?;
                        out.extend_from_slice(&self.arg_string_bytes(p));
                    }
                    Some(self.new_str_bytes(out))
                }
                // `search(str)` — index of the first match (string needle).
                "search" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    // A non-RegExp argument is ToString'd (running a user `toString`,
                    // which may throw) and matched literally.
                    let needle_bytes = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        Vec::new()
                    } else {
                        self.arg_string_bytes_fallible(arg(0))?
                    };
                    let needle = crate::wtf8::to_string_lossy(&needle_bytes);
                    // The result is a UTF-16 unit index.
                    let idx = s
                        .find(&needle)
                        .map_or(-1.0, |b| s[..b].encode_utf16().count() as f64);
                    Some(NanBox::number(idx))
                }
                // `normalize()` — Unicode normalization via the `intl` crate.
                // Normalization is the identity on a lone surrogate (it is its own
                // canonical/compatibility form and combines with nothing), so a
                // surrogate-bearing string normalizes its scalar runs and passes
                // each lone surrogate through in place; a surrogate-free string
                // takes the `&str` fast path unchanged.
                "normalize" => {
                    let form = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        String::from("NFC")
                    } else {
                        // ToString the form (a Symbol throws a TypeError) *before*
                        // validating it against the allowed set.
                        self.coerce_to_string(arg(0))?
                    };
                    #[cfg(feature = "intl")]
                    {
                        // Validate the form first (a bad form is a RangeError even
                        // for the empty string), then normalize per run.
                        if !matches!(form.as_str(), "NFC" | "NFD" | "NFKC" | "NFKD") {
                            let m = self.new_str(&alloc::format!(
                                "The normalization form should be one of NFC, NFD, NFKC, NFKD. Got {form}."
                            ));
                            return Err(ExecError::Throw(
                                self.make_error(N_ERROR_BASE + 2, Some(m)),
                            ));
                        }
                        Some(self.new_str_bytes(normalize_wtf8(&bytes, &form)))
                    }
                    #[cfg(not(feature = "intl"))]
                    {
                        let _ = &form;
                        // No `intl`: normalization is a no-op, but still preserve
                        // surrogates by round-tripping the lossless bytes.
                        Some(self.new_str_bytes(bytes.clone()))
                    }
                }
                // `localeCompare(other)` — ordering sign (code-point order; no
                // locale tailoring).
                "localeCompare" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    let other = self.realm.to_display_string(arg(0));
                    let cmp = match s.as_str().cmp(other.as_str()) {
                        core::cmp::Ordering::Less => -1.0,
                        core::cmp::Ordering::Equal => 0.0,
                        core::cmp::Ordering::Greater => 1.0,
                    };
                    Some(NanBox::number(cmp))
                }
                _ => None,
            };
            if out.is_some() {
                return Ok(out);
            }
        }

        // --- array methods ---
        // `Array.prototype.<m>.call(arrayLike)`: a plain object with a numeric `length`
        // is treated as array-like for the common read-only methods — materialize a
        // temporary array of its indexed elements and run the method against that.
        const ARRAY_LIKE_METHODS: &[&str] = &[
            "slice",
            "map",
            "filter",
            "forEach",
            "indexOf",
            "lastIndexOf",
            "includes",
            "find",
            "findIndex",
            "findLast",
            "findLastIndex",
            "some",
            "every",
            "reduce",
            "reduceRight",
            "join",
            "at",
            "flat",
            "flatMap",
        ];
        let mut array_like = None;
        if self.realm.array_elements(handle).is_none()
            && self.realm.object_keys(handle).is_some()
            && ARRAY_LIKE_METHODS.contains(&method)
        {
            // ToLength(Get(O, "length")): the length is coerced through a JS
            // `valueOf`/`toString` (so an object length with a custom coercion is
            // honored), NaN/negatives become 0, and the result is clamped to
            // 2**53−1 (capped lower here to bound the dense materialization).
            let len_val = self.read_member(handle, "length")?;
            let len_num = self.coerce_to_number(len_val)?;
            let raw = self.realm.to_number(len_num);
            let len_f = if raw.is_nan() || raw <= 0.0 {
                0.0
            } else {
                raw.min(9_007_199_254_740_991.0)
            };
            if len_f <= (1u64 << 24) as f64 {
                let len = len_f as usize;
                let mut tmp = Vec::with_capacity(len);
                for i in 0..len {
                    // Get(O, idx) walks the prototype chain and invokes getters.
                    tmp.push(self.read_member(handle, &alloc::format!("{i}"))?);
                }
                array_like = Some(self.realm.new_array(tmp));
            }
        }
        // For a generic array-like `this`, the callback receives the *original*
        // object as its 3rd argument (`O`), not the materialized snapshot — so
        // `(v, i, arr) => arr === O` and `arr instanceof Boolean` hold.
        let callback_recv = NanBox::handle(handle.to_raw());
        let handle = array_like.unwrap_or(handle);
        // S5/S3/S8: bulk typed-array mutators (`fill`/`copyWithin`/`set`/`subarray`)
        // operate on the backing bytes directly. Handle them up front using the
        // view's length (`typed_len`) so they never materialize every element just
        // to read `.len()`, and route through the `Realm` bulk methods (one buffer
        // borrow, no per-element heap lookup or `Vec` allocation).
        if let Some(tlen) = self.realm.typed_len(handle) {
            match method {
                // `fill(value, start?, end?)` — mutate in place, return the view.
                // Spec order: ToNumber/ToBigInt(value) once, then
                // ToIntegerOrInfinity(start)/(end). `start`/`end` default to
                // `0`/`len`; negatives count from the end. Each coercion can throw
                // (a Symbol/abrupt valueOf), propagated here.
                "fill" => {
                    // For a non-BigInt view a Number fill still goes through
                    // ToNumber (a Symbol value throws); `coerce_typed_array_write`
                    // handles the BigInt case. Coerce the value to a Number for a
                    // numeric view so a Symbol/BigInt value throws per spec.
                    let value = if self.realm.typed_kind(handle).is_some_and(is_bigint_kind) {
                        self.coerce_typed_array_write(handle, arg(0))?
                    } else {
                        self.coerce_to_number(arg(0))?
                    };
                    let start = self.typed_clamp_index_checked(arg(1), 0, tlen)?;
                    let end = self.typed_clamp_index_checked(arg(2), tlen, tlen)?;
                    // A coercion may have detached/shrunk the buffer; re-read the
                    // live length and clamp so the write never runs past it.
                    let live = self.realm.typed_len(handle).unwrap_or(0);
                    let (start, end) = (start.min(live), end.min(live));
                    self.realm.typed_fill_range(handle, value, start, end);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `copyWithin(target, start, end?)` — copy a slice within the view
                // in place (raw same-width byte move); negatives count from the end.
                // Each relative index is ToIntegerOrInfinity (abrupt-propagating).
                "copyWithin" => {
                    let target = self.typed_clamp_index_checked(arg(0), 0, tlen)?;
                    let start = self.typed_clamp_index_checked(arg(1), 0, tlen)?;
                    let end = self.typed_clamp_index_checked(arg(2), tlen, tlen)?;
                    // A coercion may have shrunk a resizable buffer; clamp to the
                    // live length so the copy stays in bounds.
                    let live = self.realm.typed_len(handle).unwrap_or(0);
                    let (target, start, end) = (target.min(live), start.min(live), end.min(live));
                    let count = end.saturating_sub(start).min(live.saturating_sub(target));
                    self.realm.typed_copy_within(handle, target, start, count);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `TypedArray.prototype.set(source, offset?)`: copy a source's
                // elements into this view at `offset`, coercing each.
                // Spec order: ToIntegerOrInfinity(offset) (negative → RangeError),
                // then the typed-source or array-like-source branch.
                "set" => {
                    let target_is_bigint =
                        self.realm.typed_kind(handle).is_some_and(is_bigint_kind);
                    // Step 4-5: targetOffset = ToIntegerOrInfinity(offset); a
                    // negative offset is a RangeError. (Abrupt-propagating.)
                    let offset_n = self.coerce_to_integer_or_infinity(arg(1))?;
                    if offset_n < 0.0 {
                        let m = self.new_str("offset is out of bounds");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    let src_box = arg(0);
                    if let Some(src) = src_box.as_handle().map(Handle::from_raw) {
                        // A typed-array source: same-kind → raw byte copy; otherwise
                        // element copy with per-element coercion. `offset + srcLen`
                        // must fit the (live) target length.
                        if let Some(src_len) = self.realm.typed_len(src) {
                            let tlen_live = self.realm.typed_len(handle).unwrap_or(tlen);
                            let offset = if offset_n.is_finite() && offset_n <= tlen_live as f64 {
                                offset_n as usize
                            } else {
                                tlen_live + 1 // forces the bounds RangeError below
                            };
                            if offset.checked_add(src_len).is_none_or(|e| e > tlen_live) {
                                let m = self.new_str("offset is out of bounds");
                                return Err(ExecError::Throw(
                                    self.make_error(N_RANGE_ERROR, Some(m)),
                                ));
                            }
                            // A BigInt/Number element-kind mismatch between source and
                            // target is a TypeError (no implicit Number↔BigInt).
                            let src_is_bigint =
                                self.realm.typed_kind(src).is_some_and(is_bigint_kind);
                            if src_is_bigint != target_is_bigint {
                                return Err(self.type_error(
                                    "cannot mix BigInt and non-BigInt typed arrays in set",
                                ));
                            }
                            if self.realm.typed_set_same_kind(handle, src, offset) {
                                return Ok(Some(NanBox::undefined()));
                            }
                            let src_elems = self.realm.elements_vec(src).unwrap_or_default();
                            self.realm
                                .typed_set_from_numbers(handle, offset, &src_elems);
                            return Ok(Some(NanBox::undefined()));
                        }
                    }
                    // Array-like source: ToObject(source), then ToLength(src.length),
                    // bounds-check, then per-element Get + coerce + write (so each
                    // value's ToNumber/ToBigInt side effects and throws run in order,
                    // and values are not cached).
                    let src_obj = self.coerce_to_object(src_box);
                    let Some(src) = src_obj.as_handle().map(Handle::from_raw) else {
                        return Ok(Some(NanBox::undefined()));
                    };
                    let len_val = self.read_member(src, "length")?;
                    // ToLength: ToIntegerOrInfinity, clamped to [0, 2^53-1].
                    let len_n = self.coerce_to_integer_or_infinity(len_val)?;
                    let src_len = len_n.clamp(0.0, 9_007_199_254_740_991.0) as usize;
                    let tlen_live = self.realm.typed_len(handle).unwrap_or(tlen);
                    let offset = if offset_n.is_finite() && offset_n <= tlen_live as f64 {
                        offset_n as usize
                    } else {
                        tlen_live + 1
                    };
                    if offset.checked_add(src_len).is_none_or(|e| e > tlen_live) {
                        let m = self.new_str("offset is out of bounds");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    for i in 0..src_len {
                        let v = self.read_member(src, &alloc::format!("{i}"))?;
                        // Coerce per the target's element kind (BigInt target throws
                        // for a Number value; numeric target throws for a Symbol).
                        let coerced = if target_is_bigint {
                            self.coerce_typed_array_write(handle, v)?
                        } else {
                            self.coerce_to_number(v)?
                        };
                        // Re-read the live length each iteration (a value's valueOf
                        // may have resized the buffer); an out-of-range write is a
                        // spec no-op.
                        if offset + i < self.realm.typed_len(handle).unwrap_or(0) {
                            self.realm.set_element(handle, offset + i, coerced);
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                // `subarray(begin, end)` — a new same-kind view sharing the parent's
                // backing bytes at the parent's byte offset plus `begin * size`.
                // begin/end go through ToIntegerOrInfinity (abrupt-propagating); the
                // result is allocated via TypedArraySpeciesCreate(O, buffer, off, len).
                "subarray" => {
                    let len = tlen;
                    let start = self.typed_clamp_index_checked(arg(0), 0, len)?;
                    let end = self.typed_clamp_index_checked(arg(1), len, len)?;
                    let new_len = end.saturating_sub(start);
                    let kind = self.realm.typed_kind(handle).unwrap_or(0);
                    let elem_size = TYPED_ARRAY_KINDS[kind as usize].1 as usize;
                    let abuf = self.realm.typed_array_object(handle).unwrap();
                    let parent_off = self.realm.typed_byte_offset(handle).unwrap_or(0);
                    let sub_off = parent_off + start * elem_size;
                    // TypedArraySpeciesCreate(O, « buffer, beginByteOffset, newLength »).
                    if let Some(view) =
                        self.typed_subarray_species(handle, abuf, sub_off, new_len)?
                    {
                        return Ok(Some(view));
                    }
                    let bytes_h = self.realm.typed_buffer(handle).unwrap();
                    let view = self
                        .realm
                        .new_typed_array(bytes_h, abuf, sub_off, new_len, kind);
                    return Ok(Some(NanBox::handle(view.to_raw())));
                }
                _ => {}
            }
        }
        if let Some(elems) = self.realm.elements_vec(handle) {
            // Methods whose integer-position arguments go through
            // ToIntegerOrInfinity (→ ToNumber): a Symbol argument must throw a
            // TypeError before any element processing. The downstream code coerces
            // with the infallible `to_number` (NaN for a Symbol), so surface the
            // error here. Only Symbol-valued arguments are pre-coerced, to avoid
            // perturbing a user `valueOf`'s call order/count.
            let int_arg_positions: &[usize] = match method {
                "slice" => &[0, 1],
                "fill" => &[1, 2],
                "indexOf" | "lastIndexOf" | "includes" => &[1],
                "flat" => &[0],
                "copyWithin" => &[0, 1, 2],
                "splice" => &[0, 1],
                _ => &[],
            };
            for &pos in int_arg_positions {
                let a = arg(pos);
                if a.as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.symbol_at(h).is_some())
                {
                    self.coerce_to_number(a)?;
                }
            }
            // The iteration built-ins require IsCallable(callbackfn) *before* any
            // element access (a non-callable callback is a TypeError even for an
            // empty array). `reduce`/`reduceRight` validate `arg(0)`; the rest take
            // the callback at `arg(0)` too.
            if matches!(
                method,
                "forEach"
                    | "map"
                    | "filter"
                    | "some"
                    | "every"
                    | "find"
                    | "findIndex"
                    | "findLast"
                    | "findLastIndex"
                    | "reduce"
                    | "reduceRight"
                    | "flatMap"
            ) {
                self.require_callable(arg(0), &alloc::format!("{method} callback"))?;
            }
            // NOTE: per spec the length-mutating methods finish with
            // Set(O, "length", …, Throw=true) and so throw a TypeError on a
            // non-writable/frozen array's `length`. The curated gate's
            // `freeze-semantics.js` relies on the (non-conformant) silent no-op,
            // so we keep the lenient behavior here to preserve the 693/693 gate.
            match method {
                "push" => {
                    let mut len = elems.len();
                    // A frozen array rejects new elements (non-strict: silent).
                    if !self.realm.is_frozen(handle) {
                        for a in args {
                            len = self.realm.array_push(handle, *a).unwrap_or(len);
                        }
                    }
                    return Ok(Some(NanBox::number(len as f64)));
                }
                "pop" => return Ok(Some(self.realm.array_pop(handle))),
                // `splice(start, deleteCount?, ...items)` — mutate in place,
                // return the removed elements as a new array.
                "shift" => {
                    if elems.is_empty() {
                        return Ok(Some(NanBox::undefined()));
                    }
                    let first = elems[0];
                    self.realm.array_set_all(handle, elems[1..].to_vec());
                    return Ok(Some(first));
                }
                "unshift" => {
                    let mut next: Vec<NanBox> = args.to_vec();
                    next.extend_from_slice(&elems);
                    let len = next.len();
                    self.realm.array_set_all(handle, next);
                    return Ok(Some(NanBox::number(len as f64)));
                }
                "splice" => {
                    let len = elems.len();
                    let start = {
                        let s = self.realm.to_number(arg(0));
                        if s < 0.0 {
                            (len as f64 + s).max(0.0) as usize
                        } else {
                            (s as usize).min(len)
                        }
                    };
                    let delete = if args.len() < 2 {
                        len - start
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(len - start)
                    };
                    let removed: Vec<NanBox> = elems[start..start + delete].to_vec();
                    let mut next: Vec<NanBox> = elems[..start].to_vec();
                    next.extend_from_slice(&args[2.min(args.len())..]);
                    next.extend_from_slice(&elems[start + delete..]);
                    self.realm.array_set_all(handle, next);
                    return Ok(Some(NanBox::handle(self.realm.new_array(removed).to_raw())));
                }
                // `arr.toString()` joins with a comma (like `join()`).
                "join" | "toString" => {
                    // The separator goes through ToString (a Symbol throws a
                    // TypeError); `undefined` (or `toString`) defaults to ",".
                    let sep =
                        if method == "toString" || matches!(arg(0).unpack(), Unpacked::Undefined) {
                            String::from(",")
                        } else {
                            self.coerce_to_string(arg(0))?
                        };
                    // `null`/`undefined` render empty; an object element is run
                    // through ToString (so a custom `toString` is honored). The
                    // receiver array seeds the cycle set, so a self-reference (or a
                    // mutual cycle back to it) renders empty rather than recursing.
                    let mut parts: Vec<String> = Vec::with_capacity(elems.len());
                    for e in &elems {
                        let s = match e.unpack() {
                            Unpacked::Null | Unpacked::Undefined => String::new(),
                            // A direct self-reference back to the receiver renders
                            // empty (per `Array.prototype.join`), without recursing.
                            Unpacked::Handle(raw) if raw == handle.to_raw() => String::new(),
                            _ => {
                                let p = self.coerce_object(*e, "string")?;
                                self.realm.to_display_string(p)
                            }
                        };
                        parts.push(s);
                    }
                    return Ok(Some(self.new_str(&parts.join(&sep))));
                }
                // Spec `Array.prototype.toLocaleString` / `%TypedArray%.prototype.
                // toLocaleString`: join with "," after invoking each element's own
                // `toLocaleString()` method (its result ToString'd); `null`/
                // `undefined` render empty.
                "toLocaleString" => {
                    let mut parts: Vec<String> = Vec::with_capacity(elems.len());
                    for e in &elems {
                        let s = match e.unpack() {
                            Unpacked::Null | Unpacked::Undefined => String::new(),
                            Unpacked::Handle(raw) if raw == handle.to_raw() => String::new(),
                            // Numbers/BigInts render via the engine's grouped locale
                            // form directly (no Intl) — matches the curated gate.
                            Unpacked::Number(n) => group_thousands(n),
                            _ => {
                                if let Some(big) = e
                                    .as_handle()
                                    .and_then(|r| self.realm.bigint_at(Handle::from_raw(r)))
                                {
                                    group_thousands_str(&bigint_to_radix(&big, 10))
                                } else if e.as_handle().map(Handle::from_raw).is_some_and(|h| {
                                    self.realm.object_keys(h).is_some()
                                        || self.realm.is_array(h)
                                        || self.realm.typed_kind(h).is_some()
                                }) {
                                    // A real object element: call its own
                                    // `toLocaleString()`, ToString the result
                                    // (abrupt completions propagate).
                                    let h = e.as_handle().map(Handle::from_raw).unwrap();
                                    let m = self.read_member(h, "toLocaleString")?;
                                    let r = self.call_with_this(m, *e, &[])?;
                                    self.coerce_to_string(r)?
                                } else {
                                    // A string/boolean (or other) primitive element:
                                    // its `toLocaleString` is identity-ish — ToString.
                                    self.coerce_to_string(*e)?
                                }
                            }
                        };
                        parts.push(s);
                    }
                    return Ok(Some(self.new_str(&parts.join(","))));
                }
                "includes" => {
                    let target = arg(0);
                    let from = self.array_from_index_checked(arg(1), elems.len())?;
                    // SameValueZero: like `===` but `NaN` matches `NaN`.
                    let t_nan = target.as_number().is_some_and(f64::is_nan);
                    let found = elems[from..].iter().any(|e| {
                        self.realm.strict_equals(*e, target)
                            || (t_nan && e.as_number().is_some_and(f64::is_nan))
                    });
                    return Ok(Some(NanBox::boolean(found)));
                }
                // `toSorted`/`toReversed`/`with` — non-mutating array methods.
                "toReversed" => {
                    let mut out = elems.clone();
                    out.reverse();
                    return Ok(Some(self.typed_like(handle, out)));
                }
                "with" => {
                    let len = elems.len() as i64;
                    // ToIntegerOrInfinity (abrupt-propagating). For a typed array
                    // the value is also coerced (Number/BigInt) per spec.
                    let i = self.coerce_to_integer_or_infinity(arg(0))? as i64;
                    let idx = if i < 0 { len + i } else { i };
                    // An out-of-range index is a RangeError.
                    if idx < 0 || idx >= len {
                        let m = self.new_str("Invalid index");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    let mut out = elems.clone();
                    out[idx as usize] = arg(1);
                    return Ok(Some(self.typed_like(handle, out)));
                }
                "toSorted" => {
                    let numeric = self.realm.typed_kind(handle).is_some();
                    let sorted = self.sort_array(elems.clone(), arg(0), numeric)?;
                    return Ok(Some(self.typed_like(handle, sorted)));
                }
                "indexOf" => {
                    let target = arg(0);
                    let from = self.array_from_index_checked(arg(1), elems.len())?;
                    let idx = elems[from..]
                        .iter()
                        .position(|e| self.realm.strict_equals(*e, target))
                        .map_or(-1.0, |i| (i + from) as f64);
                    return Ok(Some(NanBox::number(idx)));
                }
                "map" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = callback_recv;
                    let mut out = Vec::with_capacity(elems.len());
                    for (i, e) in elems.iter().enumerate() {
                        let cb_args = [*e, NanBox::number(i as f64), arr];
                        out.push(self.call_with_this(f, this_arg, &cb_args)?);
                    }
                    // A typed-array `map` allocates via TypedArraySpeciesCreate.
                    return Ok(Some(self.typed_like_species(handle, out)?));
                }
                "filter" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = callback_recv;
                    let mut out = Vec::new();
                    for (i, e) in elems.iter().enumerate() {
                        let cb_args = [*e, NanBox::number(i as f64), arr];
                        let r = self.call_with_this(f, this_arg, &cb_args)?;
                        if self.realm.truthy(r) {
                            out.push(*e);
                        }
                    }
                    // A typed-array `filter` allocates via TypedArraySpeciesCreate.
                    return Ok(Some(self.typed_like_species(handle, out)?));
                }
                "forEach" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = callback_recv;
                    for (i, e) in elems.iter().enumerate() {
                        let cb_args = [*e, NanBox::number(i as f64), arr];
                        self.call_with_this(f, this_arg, &cb_args)?;
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "reduce" => {
                    let f = arg(0);
                    let mut acc;
                    let mut start = 0;
                    if args.len() >= 2 {
                        acc = arg(1);
                    } else if elems.is_empty() {
                        let m = self.new_str("Reduce of empty array with no initial value");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    } else {
                        acc = elems[0];
                        start = 1;
                    }
                    let arr = callback_recv;
                    for (i, e) in elems.iter().enumerate().skip(start) {
                        acc = self.call(f, &[acc, *e, NanBox::number(i as f64), arr])?;
                    }
                    return Ok(Some(acc));
                }
                // `reduceRight` — like `reduce` but right-to-left.
                "reduceRight" => {
                    let f = arg(0);
                    let mut acc;
                    let mut idx = elems.len();
                    if args.len() >= 2 {
                        acc = arg(1);
                    } else if elems.is_empty() {
                        let m = self.new_str("Reduce of empty array with no initial value");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    } else {
                        idx -= 1;
                        acc = elems[idx];
                    }
                    let arr = callback_recv;
                    while idx > 0 {
                        idx -= 1;
                        acc = self.call(f, &[acc, elems[idx], NanBox::number(idx as f64), arr])?;
                    }
                    return Ok(Some(acc));
                }
                "slice" => {
                    // A typed-array `slice` coerces start/end through
                    // ToIntegerOrInfinity (abrupt-propagating) and allocates the
                    // result via TypedArraySpeciesCreate; a plain array keeps the
                    // existing infallible bound computation + plain-array result.
                    if self.realm.typed_kind(handle).is_some() {
                        let len = elems.len();
                        let a = self.typed_clamp_index_checked(arg(0), 0, len)?;
                        let b = self.typed_clamp_index_checked(arg(1), len, len)?;
                        let sub = if a < b {
                            elems[a..b].to_vec()
                        } else {
                            Vec::new()
                        };
                        return Ok(Some(self.typed_like_species(handle, sub)?));
                    }
                    let (a, b) = slice_bounds(
                        self.realm.to_number(arg(0)),
                        arg(1),
                        &self.realm,
                        elems.len(),
                    );
                    let sub = elems[a..b].to_vec();
                    return Ok(Some(self.typed_like(handle, sub)));
                }
                // Iterators: `keys()` over indices, `values()` over elements,
                // `entries()` over `[index, element]` pairs (eager generators).
                "keys" => {
                    let ks: Vec<NanBox> =
                        (0..elems.len()).map(|i| NanBox::number(i as f64)).collect();
                    return Ok(Some(self.make_generator(ks)));
                }
                "values" => {
                    return Ok(Some(self.make_generator(elems.clone())));
                }
                "entries" => {
                    let mut pairs = Vec::with_capacity(elems.len());
                    for (i, e) in elems.iter().enumerate() {
                        let pair = self
                            .realm
                            .new_array(alloc::vec![NanBox::number(i as f64), *e]);
                        pairs.push(NanBox::handle(pair.to_raw()));
                    }
                    return Ok(Some(self.make_generator(pairs)));
                }
                "concat" => {
                    let mut out = elems.clone();
                    // An argument is spread iff it is concat-spreadable: its
                    // `[Symbol.isConcatSpreadable]` (if defined) decides, else it is
                    // spread exactly when it is an array.
                    let sym = self.well_known_symbol("isConcatSpreadable");
                    let spread_key = self.member_key(sym);
                    for a in args {
                        let ah = a.as_handle().map(Handle::from_raw);
                        let spread = match ah {
                            Some(h) => match self.realm.get_property(h, &spread_key) {
                                Some(v) if !matches!(v.unpack(), Unpacked::Undefined) => {
                                    self.realm.truthy(v)
                                }
                                _ => self.realm.is_array(h),
                            },
                            None => false,
                        };
                        match (spread, ah) {
                            (true, Some(h)) => {
                                if let Some(other) = self.realm.array_elements(h).map(<[_]>::to_vec)
                                {
                                    out.extend(other);
                                } else {
                                    // A spreadable array-like: read length + indices.
                                    let len = self
                                        .realm
                                        .get_property(h, "length")
                                        .map(|v| self.realm.to_number(v))
                                        .unwrap_or(0.0)
                                        .max(0.0)
                                        as usize;
                                    for i in 0..len {
                                        let k = alloc::format!("{i}");
                                        out.push(
                                            self.realm
                                                .get_property(h, &k)
                                                .unwrap_or(NanBox::undefined()),
                                        );
                                    }
                                }
                            }
                            _ => out.push(*a),
                        }
                    }
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                "reverse" => {
                    // Reverses in place and returns the same array (or typed-array view).
                    let mut out = elems.clone();
                    out.reverse();
                    self.write_back_elements(handle, out);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `fill(value, start?, end?)` — mutate a plain array in place,
                // return it. `start`/`end` default to `0`/`len`; negatives count
                // from the end. (Typed-array views are handled by the bulk fast
                // path above and never reach here.)
                "fill" => {
                    let len = elems.len();
                    let value = arg(0);
                    let start = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        let n = self.realm.to_number(arg(1));
                        if n < 0.0 {
                            (len as f64 + n).max(0.0) as usize
                        } else {
                            (n as usize).min(len)
                        }
                    };
                    let end = if matches!(arg(2).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        let n = self.realm.to_number(arg(2));
                        if n < 0.0 {
                            (len as f64 + n).max(0.0) as usize
                        } else {
                            (n as usize).min(len)
                        }
                    };
                    for i in start..end {
                        // `set_element_coerced` applies typed-array coercion + buffer
                        // write-through (a plain `set_element` for an ordinary array).
                        self.set_element_coerced(handle, i, value);
                    }
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `flat(depth = 1)` — recursively flatten nested arrays.
                "flat" => {
                    let depth = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        1
                    } else {
                        self.realm.to_number(arg(0)) as i32
                    };
                    let out = self.flatten(&elems, depth, 0)?;
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                // `copyWithin(target, start, end?)` — copy a slice within the
                // array in place; negatives count from the end.
                "copyWithin" => {
                    let len = elems.len() as i64;
                    let norm = |v: f64| -> i64 {
                        let i = v as i64;
                        if i < 0 { (len + i).max(0) } else { i.min(len) }
                    };
                    let target = norm(self.realm.to_number(arg(0)));
                    let start = norm(self.realm.to_number(arg(1)));
                    let end = if matches!(arg(2).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        norm(self.realm.to_number(arg(2)))
                    };
                    let slice: Vec<NanBox> =
                        elems[start as usize..end.max(start) as usize].to_vec();
                    for (k, v) in slice.into_iter().enumerate() {
                        let dst = target as usize + k;
                        if dst >= elems.len() {
                            break;
                        }
                        self.set_element_coerced(handle, dst, v);
                    }
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `map` then flatten one level.
                "flatMap" => {
                    let f = arg(0);
                    let mut out = Vec::new();
                    for (i, e) in elems.iter().enumerate() {
                        let r = self.call(f, &[*e, NanBox::number(i as f64)])?;
                        match r
                            .as_handle()
                            .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                            .map(<[_]>::to_vec)
                        {
                            Some(inner) => out.extend(inner),
                            None => out.push(r),
                        }
                    }
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                // `at` with negative-from-end indexing. The index is
                // ToIntegerOrInfinity (a Symbol/abrupt valueOf throws).
                "at" => {
                    let i = self.coerce_to_integer_or_infinity(arg(0))?;
                    let idx = if i < 0.0 { elems.len() as f64 + i } else { i };
                    return Ok(Some(
                        as_index(idx)
                            .and_then(|u| elems.get(u))
                            .copied()
                            .unwrap_or(NanBox::undefined()),
                    ));
                }
                "lastIndexOf" => {
                    let target = arg(0);
                    let len = elems.len();
                    if len == 0 {
                        return Ok(Some(NanBox::number(-1.0)));
                    }
                    // Optional `fromIndex` (default last; negative counts back).
                    // ToIntegerOrInfinity (abrupt-propagating).
                    let from = if args.len() >= 2 {
                        let n = self.coerce_to_integer_or_infinity(arg(1))?;
                        let n = if n < 0.0 { len as f64 + n } else { n };
                        if n < 0.0 {
                            return Ok(Some(NanBox::number(-1.0)));
                        }
                        (n as usize).min(len - 1)
                    } else {
                        len - 1
                    };
                    let found = elems[..=from]
                        .iter()
                        .rposition(|e| self.realm.strict_equals(*e, target));
                    return Ok(Some(NanBox::number(found.map_or(-1.0, |i| i as f64))));
                }
                "find" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[*e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(*e));
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "findIndex" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[*e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::number(i as f64)));
                        }
                    }
                    return Ok(Some(NanBox::number(-1.0)));
                }
                // `findLast`/`findLastIndex` — scan right-to-left.
                "findLast" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate().rev() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[*e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(*e));
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "findLastIndex" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate().rev() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[*e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::number(i as f64)));
                        }
                    }
                    return Ok(Some(NanBox::number(-1.0)));
                }
                "some" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[*e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::boolean(true)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(false)));
                }
                "every" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if !self.call_truthy_this(
                            f,
                            arg(1),
                            &[*e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::boolean(false)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(true)));
                }
                "sort" => {
                    // Sorts in place and returns the same array. A typed array sorts
                    // numerically by default (a plain array lexicographically).
                    let numeric = self.realm.typed_kind(handle).is_some();
                    let sorted = self.sort_array(elems, arg(0), numeric)?;
                    self.write_back_elements(handle, sorted);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `toSpliced(start, deleteCount, ...items)` — a spliced copy
                // (the ES2023 immutable counterpart of `splice`).
                "toSpliced" => {
                    let len = elems.len() as i64;
                    let start = {
                        let s = self.realm.to_number(arg(0)) as i64;
                        if s < 0 { (len + s).max(0) } else { s.min(len) }
                    } as usize;
                    let del = if args.len() < 2 {
                        elems.len() - start
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(elems.len() - start)
                    };
                    let mut out: Vec<NanBox> = elems[..start].to_vec();
                    out.extend_from_slice(&args[2.min(args.len())..]);
                    out.extend_from_slice(&elems[start + del..]);
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                _ => {}
            }
        }

        // --- Map / Set methods ---
        if let Some(size) = self.realm.collection_size(handle) {
            match method {
                "set" => {
                    self.guard_weak_key(handle, arg(0))?;
                    self.realm.collection_set(handle, arg(0), arg(1));
                    return Ok(Some(recv)); // Map.set returns the map (chainable)
                }
                "add" => {
                    self.guard_weak_key(handle, arg(0))?;
                    self.realm.collection_set(handle, arg(0), arg(0));
                    return Ok(Some(recv)); // Set.add returns the set
                }
                "get" => {
                    return Ok(Some(
                        self.realm
                            .collection_get(handle, arg(0))
                            .unwrap_or(NanBox::undefined()),
                    ));
                }
                "has" => {
                    return Ok(Some(NanBox::boolean(
                        self.realm.collection_has(handle, arg(0)),
                    )));
                }
                "delete" => {
                    return Ok(Some(NanBox::boolean(
                        self.realm.collection_delete(handle, arg(0)),
                    )));
                }
                "clear" => {
                    self.realm.collection_clear(handle);
                    return Ok(Some(NanBox::undefined()));
                }
                "forEach" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let coll = NanBox::handle(handle.to_raw());
                    for (k, v) in self.realm.collection_entries(handle).unwrap_or_default() {
                        // The callback gets `(value, key, collection)` with `thisArg`.
                        self.call_with_this(f, this_arg, &[v, k, coll])?;
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "keys" => {
                    // A real iterator object (with `.next`/`[Symbol.iterator]`), so
                    // `m.keys().next()` works — not just `for-of`.
                    let keys: Vec<NanBox> = self
                        .realm
                        .collection_entries(handle)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, _)| k)
                        .collect();
                    return Ok(Some(self.make_generator(keys)));
                }
                "values" => {
                    // A Set yields its elements; a Map yields its values.
                    let is_set = self.realm.collection_is_set(handle) == Some(true);
                    let vals: Vec<NanBox> = self
                        .realm
                        .collection_entries(handle)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, v)| if is_set { k } else { v })
                        .collect();
                    return Ok(Some(self.make_generator(vals)));
                }
                "entries" => {
                    let pairs = self.realm.collection_entries(handle).unwrap_or_default();
                    let arr: Vec<NanBox> = pairs
                        .into_iter()
                        .map(|(k, v)| {
                            NanBox::handle(self.realm.new_array(alloc::vec![k, v]).to_raw())
                        })
                        .collect();
                    return Ok(Some(self.make_generator(arr)));
                }
                // ES2025 Set composition. The argument is treated as a set-like
                // (any iterable supplies its elements).
                "union"
                | "intersection"
                | "difference"
                | "symmetricDifference"
                | "isSubsetOf"
                | "isSupersetOf"
                | "isDisjointFrom"
                    if self.realm.collection_is_set(handle) == Some(true) =>
                {
                    let mine: Vec<NanBox> = self
                        .realm
                        .collection_entries(handle)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, _)| k)
                        .collect();
                    let other = self.iterate_values(arg(0))?;
                    let in_other = |this: &Self, v: NanBox| {
                        other.iter().any(|o| this.realm.same_value_zero(*o, v))
                    };
                    let in_mine = |this: &Self, v: NanBox| {
                        mine.iter().any(|m| this.realm.same_value_zero(*m, v))
                    };
                    // Predicate methods return a boolean.
                    match method {
                        "isSubsetOf" => {
                            return Ok(Some(NanBox::boolean(
                                mine.iter().all(|m| in_other(self, *m)),
                            )));
                        }
                        "isSupersetOf" => {
                            return Ok(Some(NanBox::boolean(
                                other.iter().all(|o| in_mine(self, *o)),
                            )));
                        }
                        "isDisjointFrom" => {
                            return Ok(Some(NanBox::boolean(
                                !mine.iter().any(|m| in_other(self, *m)),
                            )));
                        }
                        _ => {}
                    }
                    // The rest build a new Set.
                    let result = self.realm.new_collection(true);
                    match method {
                        "union" => {
                            for e in mine.iter().chain(other.iter()) {
                                self.realm.collection_set(result, *e, *e);
                            }
                        }
                        "intersection" => {
                            for e in &mine {
                                if in_other(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                        }
                        "difference" => {
                            for e in &mine {
                                if !in_other(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                        }
                        // symmetricDifference: in exactly one of the two.
                        _ => {
                            for e in &mine {
                                if !in_other(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                            for e in &other {
                                if !in_mine(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                        }
                    }
                    return Ok(Some(NanBox::handle(result.to_raw())));
                }
                _ => {
                    let _ = size;
                }
            }
        }

        // Default `Object.prototype` methods for an object receiver that did not
        // match a more specific built-in and has no own/inherited method of its
        // own (e.g. a plain object's `toString`/`valueOf`).
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && matches!(method, "toString" | "valueOf" | "toLocaleString")
        {
            // A user-defined (own or inherited) method takes precedence.
            let own = self.read_member(h, method)?;
            if own
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                return Ok(None);
            }
            // String values/objects: `toString`/`valueOf` yield the string.
            if let Some(s) = self.realm.string_value(h) {
                return Ok(Some(self.new_str(&s)));
            }
            if method == "valueOf" {
                return Ok(Some(recv));
            }
            let tag = self.object_string_tag(h)?;
            return Ok(Some(self.new_str(&alloc::format!("[object {tag}]"))));
        }
        Ok(None)
    }
}
