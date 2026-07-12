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
            // `Atomics.*` — single-agent read-modify-write / load / store over an
            // integer `TypedArray`. `isLockFree(n)` is a pure size check; the rest
            // validate the array kind + in-range index, then act atomically (trivial
            // without concurrent agents).
            // `Atomics.pause([N])` — a spin-loop hint. Single-agent: a no-op that
            // returns `undefined`. `N`, if present, must be an integral Number
            // (else a TypeError); it is otherwise ignored.
            N_ATOMICS_PAUSE => {
                let a = arg(0);
                if !matches!(a.unpack(), Unpacked::Undefined) {
                    let n = match a.unpack() {
                        Unpacked::Number(n) => n,
                        _ => f64::NAN,
                    };
                    if !(n.is_finite() && n == (n as i64) as f64) {
                        return Err(
                            self.type_error("Atomics.pause: iterationNumber must be an integer")
                        );
                    }
                }
                NanBox::undefined()
            }
            N_ATOMICS_IS_LOCK_FREE => {
                // `size` is ToIntegerOrInfinity'd — an object's `valueOf` runs
                // (`isLockFree({valueOf:()=>1})` === `isLockFree(1)`).
                let n = self.coerce_to_integer_or_infinity(arg(0))?;
                // An integer byte width of 1/2/4/8 is lock-free.
                let is_int = n.is_finite() && n == (n as i64) as f64;
                NanBox::boolean(is_int && matches!(n as i64, 1 | 2 | 4 | 8))
            }
            N_ATOMICS_ADD
            | N_ATOMICS_SUB
            | N_ATOMICS_AND
            | N_ATOMICS_OR
            | N_ATOMICS_XOR
            | N_ATOMICS_EXCHANGE
            | N_ATOMICS_COMPARE_EXCHANGE
            | N_ATOMICS_LOAD
            | N_ATOMICS_STORE => {
                let Some(ta) = arg(0).as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("Atomics operand must be an integer TypedArray"));
                };
                let Some(kind) = self.realm.typed_kind(ta) else {
                    return Err(self.type_error("Atomics operand must be an integer TypedArray"));
                };
                // Allowed: Int8/Uint8/Int16/Uint16/Int32/Uint32/BigInt64/BigUint64
                // (not Uint8Clamped=2, Float32=7, Float64=8, Float16=11).
                if matches!(kind, 2 | 7 | 8 | 11) {
                    return Err(self.type_error("Atomics operand must be an integer TypedArray"));
                }
                // A *writing* atomic op (every one but `load`) on an immutable
                // ArrayBuffer is a TypeError, raised before the index/value
                // `valueOf` coercions run (the immutable-arraybuffer proposal).
                if id != N_ATOMICS_LOAD {
                    self.guard_view_immutable(ta)?;
                }
                let len = self.realm.array_length(ta).unwrap_or(0);
                let idx_f = self.coerce_to_integer_or_infinity(arg(1))?;
                if !(idx_f.is_finite() && idx_f >= 0.0 && (idx_f as usize) < len) {
                    let m = self.new_str("Atomics index out of range");
                    return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                }
                let idx = idx_f as usize;
                // BigInt64Array / BigUint64Array: the element and the operands are
                // BigInts (ToBigInt, not ToIntegerOrInfinity). Work in the low 64
                // bits (what the element encoding keeps); `store` returns the
                // ToBigInt operand, every other writing op returns the old value.
                if crate::nbexec::is_bigint_kind(kind) {
                    use crate::bignum::BigInt;
                    let old_box = self.realm.typed_get(ta, idx).unwrap_or_else(|| {
                        NanBox::handle(self.realm.new_bigint(BigInt::zero()).to_raw())
                    });
                    let old = self
                        .this_bigint_value(old_box)
                        .unwrap_or_else(BigInt::zero)
                        .to_u64_wrapping();
                    if id == N_ATOMICS_LOAD {
                        return Ok(old_box);
                    }
                    if id == N_ATOMICS_COMPARE_EXCHANGE {
                        let expected = self.coerce_to_bigint(arg(2))?.to_u64_wrapping();
                        let replacement = self.coerce_to_bigint(arg(3))?.to_u64_wrapping();
                        if old == expected {
                            let nb = NanBox::handle(
                                self.realm
                                    .new_bigint(BigInt::from_i128(replacement as i64 as i128))
                                    .to_raw(),
                            );
                            self.realm.typed_set(ta, idx, nb);
                        }
                        return Ok(old_box);
                    }
                    let v_big = self.coerce_to_bigint(arg(2))?;
                    let v = v_big.to_u64_wrapping();
                    let new_u = match id {
                        N_ATOMICS_STORE | N_ATOMICS_EXCHANGE => v,
                        N_ATOMICS_ADD => old.wrapping_add(v),
                        N_ATOMICS_SUB => old.wrapping_sub(v),
                        N_ATOMICS_AND => old & v,
                        N_ATOMICS_OR => old | v,
                        N_ATOMICS_XOR => old ^ v,
                        _ => old,
                    };
                    let nb = NanBox::handle(
                        self.realm
                            .new_bigint(BigInt::from_i128(new_u as i64 as i128))
                            .to_raw(),
                    );
                    self.realm.typed_set(ta, idx, nb);
                    // `store` returns the ToBigInt operand; the rest return old.
                    return Ok(if id == N_ATOMICS_STORE {
                        NanBox::handle(self.realm.new_bigint(v_big).to_raw())
                    } else {
                        old_box
                    });
                }
                let old = self
                    .realm
                    .typed_get(ta, idx)
                    .and_then(|v| v.as_number())
                    .unwrap_or(0.0);
                if id == N_ATOMICS_LOAD {
                    return Ok(NanBox::number(old));
                }
                // The value operand (index 2 for RMW/store; 2=expected,3=replacement
                // for compareExchange) is ToIntegerOrInfinity'd.
                if id == N_ATOMICS_COMPARE_EXCHANGE {
                    let expected = self.coerce_to_integer_or_infinity(arg(2))?;
                    let replacement = self.coerce_to_integer_or_infinity(arg(3))?;
                    // Compare against the stored (already type-coerced) value.
                    let expected_stored = coerce_typed(u16::from(kind), expected);
                    if old == expected_stored {
                        self.realm.typed_set(ta, idx, NanBox::number(replacement));
                    }
                    return Ok(NanBox::number(old));
                }
                let v = self.coerce_to_integer_or_infinity(arg(2))?;
                let new = match id {
                    N_ATOMICS_STORE | N_ATOMICS_EXCHANGE => v,
                    N_ATOMICS_ADD => old + v,
                    N_ATOMICS_SUB => old - v,
                    // Bitwise ops act on the two's-complement integer value.
                    N_ATOMICS_AND => ((old as i64) & (v as i64)) as f64,
                    N_ATOMICS_OR => ((old as i64) | (v as i64)) as f64,
                    N_ATOMICS_XOR => ((old as i64) ^ (v as i64)) as f64,
                    _ => old,
                };
                self.realm.typed_set(ta, idx, NanBox::number(new));
                // `store` returns `ToInteger(value)` (the *un-truncated* operand —
                // `typed_set` truncated the stored element, but the return is `v`);
                // the rest return the old value.
                if id == N_ATOMICS_STORE {
                    NanBox::number(v)
                } else {
                    NanBox::number(old)
                }
            }
            N_ATOMICS_NOTIFY | N_ATOMICS_WAIT | N_ATOMICS_WAIT_ASYNC => {
                // `notify`/`wait`/`waitAsync` operate only on a *waitable* integer
                // TypedArray — `Int32Array` (kind 5) or `BigInt64Array` (kind 9)
                // (ValidateIntegerTypedArray with `waitable = true`).
                let Some(ta) = arg(0).as_handle().map(Handle::from_raw) else {
                    return Err(
                        self.type_error("Atomics operand must be an Int32Array or BigInt64Array")
                    );
                };
                let Some(kind) = self.realm.typed_kind(ta) else {
                    return Err(
                        self.type_error("Atomics operand must be an Int32Array or BigInt64Array")
                    );
                };
                if !matches!(kind, 5 | 9) {
                    return Err(
                        self.type_error("Atomics operand must be an Int32Array or BigInt64Array")
                    );
                }
                // A detached buffer ([[ArrayBufferData]] is null) is a TypeError,
                // raised before any index/count `valueOf` coercion runs.
                if self.typed_array_detached(ta) {
                    return Err(self.type_error("Atomics called on a detached ArrayBuffer"));
                }
                // `wait`/`waitAsync` require a *shared* buffer; a non-shared buffer
                // is a TypeError raised BEFORE the index/value/timeout coercions
                // (their `valueOf` must not run — the poisoned-args tests).
                // `notify` allows a non-shared buffer (it returns 0).
                if id == N_ATOMICS_WAIT || id == N_ATOMICS_WAIT_ASYNC {
                    let shared = self.realm.typed_array_object(ta).is_some_and(|buf| {
                        self.realm
                            .get_property(buf, SHARED_ARRAY_BUFFER_BRAND)
                            .is_some()
                    });
                    if !shared {
                        return Err(self.type_error("Atomics.wait requires a SharedArrayBuffer"));
                    }
                }
                // ValidateAtomicAccess: the length is read *before* the index's
                // `valueOf` runs, and an out-of-range index is a RangeError.
                let len = self.realm.array_length(ta).unwrap_or(0);
                let idx_f = self.coerce_to_integer_or_infinity(arg(1))?;
                if !(idx_f.is_finite() && idx_f >= 0.0 && (idx_f as usize) < len) {
                    let m = self.new_str("Atomics index out of range");
                    return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                }
                let idx = idx_f as usize;
                if id == N_ATOMICS_NOTIFY {
                    // Coerce `count` (default +Infinity, else ToIntegerOrInfinity
                    // clamped to ≥ 0) — its `valueOf` runs and may throw. Then wake
                    // up to `count` matching `waitAsync` waiters (single-agent: no
                    // *blocking* `wait` can ever be parked, but same-agent
                    // `waitAsync` promises can), returning the number woken.
                    let count = if matches!(arg(2).unpack(), crate::nanbox::Unpacked::Undefined) {
                        f64::INFINITY
                    } else {
                        self.coerce_to_integer_or_infinity(arg(2))?.max(0.0)
                    };
                    let woken = self.atomics_notify(ta, idx, kind, count);
                    NanBox::number(woken as f64)
                } else {
                    // `wait`/`waitAsync` on a shell-like host ([[CanBlock]] = true):
                    // coerce `value` then `timeout` (running their `valueOf`) and
                    // read the current element. `wait` blocks; this engine is
                    // single-agent, so no `notify` can arrive mid-block — a matching
                    // value resolves "timed-out" (nothing wakes it), a mismatch
                    // "not-equal". `waitAsync` instead parks an async waiter that a
                    // later same-agent `notify` can resolve "ok". Tests needing
                    // [[CanBlock]] = false carry flags:[CanBlockIsFalse] (skipped).
                    let (equal, timeout) = if crate::nbexec::is_bigint_kind(kind) {
                        let v = self.coerce_to_bigint(arg(2))?.to_u64_wrapping();
                        let timeout =
                            if matches!(arg(3).unpack(), crate::nanbox::Unpacked::Undefined) {
                                f64::INFINITY
                            } else {
                                let tn = self.coerce_to_number(arg(3))?;
                                self.realm.to_number(tn)
                            };
                        let w = match self.realm.typed_get(ta, idx) {
                            Some(b) => self.coerce_to_bigint(b)?.to_u64_wrapping(),
                            None => 0,
                        };
                        (w == v, timeout)
                    } else {
                        let v = self.coerce_to_integer_or_infinity(arg(2))? as i64 as i32;
                        let timeout =
                            if matches!(arg(3).unpack(), crate::nanbox::Unpacked::Undefined) {
                                f64::INFINITY
                            } else {
                                let tn = self.coerce_to_number(arg(3))?;
                                self.realm.to_number(tn)
                            };
                        let w = self
                            .realm
                            .typed_get(ta, idx)
                            .and_then(|x| x.as_number())
                            .unwrap_or(0.0) as i64 as i32;
                        (w == v, timeout)
                    };
                    if id == N_ATOMICS_WAIT_ASYNC {
                        return Ok(self.atomics_wait_async(ta, idx, kind, equal, timeout));
                    }
                    // Synchronous `Atomics.wait`. Each agent runs cooperatively to
                    // completion, so a blocking wait cannot be woken by a *later*
                    // cross-agent `notify` (that agent has already finished). Model
                    // the outcomes on the virtual clock:
                    //   * value mismatch -> "not-equal" (returns at once, no time
                    //     passes);
                    //   * value match, finite timeout -> the wait blocks for the
                    //     whole timeout and then times out: advance the virtual clock
                    //     by `timeout` so a `monotonicNow()`-measured duration around
                    //     the wait observes the elapsed time, and return "timed-out";
                    //   * value match, infinite timeout -> "timed-out" (the model
                    //     cannot block forever, and no notify can reach a finished
                    //     agent).
                    if equal {
                        let t = if timeout.is_nan() {
                            f64::INFINITY
                        } else {
                            timeout.max(0.0)
                        };
                        if t.is_finite() && t > 0.0 {
                            self.virtual_now += t;
                        }
                        return Ok(self.new_str("timed-out"));
                    }
                    return Ok(self.new_str("not-equal"));
                }
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
                // `String()` with no argument is the empty string.
                if args.is_empty() {
                    return Ok(NanBox::handle(self.realm.new_string("").to_raw()));
                }
                // `String(symbol)` is *not* a TypeError (unlike `new String(sym)`
                // or an implicit ToString): it yields `SymbolDescriptiveString`.
                if arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.symbol_at(h).is_some())
                {
                    let s = self.realm.to_display_string(arg(0));
                    return Ok(NanBox::handle(self.realm.new_string(&s).to_raw()));
                }
                // Otherwise run the value through ToString (string hint), honoring a
                // custom/inherited `toString`/`valueOf`/`@@toPrimitive`.
                let s = self.coerce_to_string(arg(0))?;
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
            // `Date(...)` called as a *function* (not `new`) ignores its arguments
            // and returns a string for the current time — equivalent to
            // `new Date().toString()`, per spec.
            N_DATE => {
                let d = self.realm.new_date(now_ms());
                self.call_method(NanBox::handle(d.to_raw()), "toString", &[])?
                    .unwrap_or_else(NanBox::undefined)
            }
            // `RegExp(pattern, flags)` called as a *function* behaves like
            // `new RegExp(...)` here (the spec's species/this-is-RegExp short-circuit
            // returns the same `pattern` only when it is a RegExp whose
            // `.constructor` is `RegExp` and `flags` is undefined; for that case we
            // also return `pattern` unchanged).
            N_REGEXP => {
                let pattern = arg(0);
                let flags_arg = arg(1);
                // `RegExp(re)` (no flags) where `re` is a RegExp with the default
                // constructor returns `re` itself.
                if matches!(flags_arg.unpack(), Unpacked::Undefined)
                    && self.is_regexp_arg(pattern)
                    && let Some(ph) = pattern.as_handle().map(Handle::from_raw)
                {
                    let ctor = self.current.get("RegExp").unwrap_or(NanBox::undefined());
                    let ctor_of = self.read_member(ph, "constructor")?;
                    if self.realm.same_value(ctor, ctor_of) {
                        return Ok(pattern);
                    }
                }
                let regexp_ctor = self.current.get("RegExp").unwrap_or(NanBox::undefined());
                return self.construct(regexp_ctor, args);
            }
            N_SYMBOL => {
                // A no-argument `Symbol()` has an `undefined` description, marked
                // with a reserved sentinel (distinct from `Symbol("")`).
                let desc = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                    String::from(SYMBOL_NO_DESC)
                } else {
                    // `Symbol(description)` does `ToString(description)` — a user
                    // `toString` runs (propagating) and a Symbol argument is a
                    // TypeError, unlike the raw `to_display_string`.
                    self.coerce_to_string(arg(0))?
                };
                NanBox::handle(self.realm.new_symbol(&desc).to_raw())
            }
            // `WeakRef.prototype.deref()` — RequireInternalSlot(this, [[Target]]),
            // then return the held target (never collected here).
            N_WEAKREF_DEREF => {
                let this = self.this_val;
                let target = this
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.get_property(h, WEAKREF_TARGET));
                match target {
                    Some(t) => t,
                    None => {
                        return Err(self.type_error(
                            "WeakRef.prototype.deref requires that 'this' be a WeakRef",
                        ));
                    }
                }
            }
            // `FinalizationRegistry.prototype.register(target, heldValue [, token])`.
            N_FINREG_REGISTER => {
                let this = self.this_val;
                let Some(cells) = this
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.get_property(*h, FINREG_TAG).is_some())
                    .and_then(|h| self.realm.get_property(h, FINREG_CELLS))
                    .and_then(|c| c.as_handle())
                    .map(Handle::from_raw)
                else {
                    return Err(self.type_error(
                        "FinalizationRegistry.prototype.register requires that 'this' be a FinalizationRegistry",
                    ));
                };
                let target = arg(0);
                let held = arg(1);
                let token = arg(2);
                if !self.can_be_held_weakly(target) {
                    return Err(self.type_error(
                        "FinalizationRegistry.register: target must be an object or a non-registered symbol",
                    ));
                }
                if self.realm.same_value(target, held) {
                    return Err(self.type_error(
                        "FinalizationRegistry.register: target and heldValue must differ",
                    ));
                }
                // `unregisterToken`, when present, must also be weakly holdable;
                // an `undefined` token records ~empty~ (stored as `undefined`).
                let token = if matches!(token.unpack(), Unpacked::Undefined) {
                    NanBox::undefined()
                } else if self.can_be_held_weakly(token) {
                    token
                } else {
                    return Err(self.type_error(
                        "FinalizationRegistry.register: unregisterToken must be an object or a non-registered symbol",
                    ));
                };
                let cell = self.realm.new_array(alloc::vec![target, held, token]);
                self.realm.array_push(cells, NanBox::handle(cell.to_raw()));
                NanBox::undefined()
            }
            // `FinalizationRegistry.prototype.unregister(token)` — remove every cell
            // whose unregister token SameValue-matches; report whether any removed.
            N_FINREG_UNREGISTER => {
                let this = self.this_val;
                let Some(cells) = this
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.get_property(*h, FINREG_TAG).is_some())
                    .and_then(|h| self.realm.get_property(h, FINREG_CELLS))
                    .and_then(|c| c.as_handle())
                    .map(Handle::from_raw)
                else {
                    return Err(self.type_error(
                        "FinalizationRegistry.prototype.unregister requires that 'this' be a FinalizationRegistry",
                    ));
                };
                let token = arg(0);
                if !self.can_be_held_weakly(token) {
                    return Err(self.type_error(
                        "FinalizationRegistry.unregister: token must be an object or a non-registered symbol",
                    ));
                }
                let existing = self
                    .realm
                    .array_elements(cells)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                let mut kept = Vec::with_capacity(existing.len());
                let mut removed = false;
                for cell in existing {
                    let cell_token = cell
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.array_elements(h))
                        .and_then(|e| e.get(2).copied())
                        .unwrap_or(NanBox::undefined());
                    if !matches!(cell_token.unpack(), Unpacked::Undefined)
                        && self.realm.same_value(cell_token, token)
                    {
                        removed = true;
                    } else {
                        kept.push(cell);
                    }
                }
                if removed {
                    let registry = this.as_handle().map(Handle::from_raw).unwrap();
                    let fresh = self.realm.new_array(kept);
                    self.realm.set_hidden_property(
                        registry,
                        FINREG_CELLS,
                        NanBox::handle(fresh.to_raw()),
                    );
                }
                NanBox::boolean(removed)
            }
            // `Symbol.prototype.toString()` — `thisSymbolValue` then `SymbolDescriptiveString`.
            N_SYMBOL_PROTO_TOSTRING => {
                let sym = self.this_symbol_value()?;
                let s = self.realm.to_display_string(sym);
                self.new_str(&s)
            }
            // `Symbol.prototype.valueOf()` — return the symbol primitive.
            N_SYMBOL_PROTO_VALUEOF => self.this_symbol_value()?,
            // `get Symbol.prototype.description`.
            N_SYMBOL_PROTO_DESC_GET => {
                let sym = self.this_symbol_value()?;
                let h = sym.as_handle().map(Handle::from_raw);
                match h.and_then(|h| self.realm.symbol_at(h)) {
                    Some((desc, _)) if &*desc == SYMBOL_NO_DESC => NanBox::undefined(),
                    Some((desc, _)) => self.new_str(&desc),
                    None => NanBox::undefined(),
                }
            }
            // `Symbol.prototype[Symbol.toPrimitive](hint)` — return the symbol.
            N_SYMBOL_PROTO_TOPRIMITIVE => self.this_symbol_value()?,
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
                // (Plain call → no `newTarget`; the current realm's default proto
                // applies.)
                return self.build_function_constructor(
                    args,
                    NanBox::undefined(),
                    NanBox::undefined(),
                );
            }
            N_GENERATOR_FUNCTION_CTOR => {
                return self.build_function_constructor_kw(
                    args,
                    "function*",
                    NanBox::undefined(),
                    NanBox::undefined(),
                );
            }
            N_ASYNC_GENERATOR_FUNCTION_CTOR => {
                return self.build_function_constructor_kw(
                    args,
                    "async function*",
                    NanBox::undefined(),
                    NanBox::undefined(),
                );
            }
            N_ASYNC_FUNCTION_CTOR => {
                return self.build_function_constructor_kw(
                    args,
                    "async function",
                    NanBox::undefined(),
                    NanBox::undefined(),
                );
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
                // `ToString(string)` then `ToInt32(radix)` both run user
                // `toString`/`valueOf` (whose abrupt completion propagates); the raw
                // `to_display_string`/`to_number` skip a user coercion for objects.
                let s = self.coerce_to_string(arg(0))?;
                let radix = match args.get(1) {
                    Some(r) if !matches!(r.unpack(), Unpacked::Undefined) => {
                        let n = self.coerce_to_number(*r)?;
                        let n = self.realm.to_number(n);
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
                let value = arg(0);
                let result = self.json_stringify(value, arg(1), arg(2))?;
                match result {
                    Some(s) => NanBox::handle(self.realm.new_string(&s).to_raw()),
                    None => NanBox::undefined(),
                }
            }
            N_JSON_PARSE => {
                // `JSON.parse(text)` does `ToString(text)` first — a user
                // `toString`/`valueOf` runs (propagating), unlike the raw
                // `to_display_string` which stringifies an object as "[object …]".
                let text = self.coerce_to_string(arg(0))?;
                let chars: Vec<char> = text.chars().collect();
                let mut pos = 0;
                // An optional `reviver` transforms each value bottom-up. When one is
                // present, parse with source spans so the reviver's `context.source`
                // (the json-parse-with-source proposal) can surface original text.
                let reviver = arg(1);
                let has_reviver = reviver
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
                if has_reviver {
                    let (value, src) = self.json_parse_src(&chars, &mut pos, 0)?;
                    skip_ws(&chars, &mut pos);
                    if pos != chars.len() {
                        return Err(self.json_error("Unexpected token in JSON"));
                    }
                    let holder = self.realm.new_object();
                    self.realm.set_property(holder, "", value);
                    return self.json_revive_ctx(holder, "", reviver, Some(&src));
                }
                let value = self.json_parse(&chars, &mut pos, 0)?;
                skip_ws(&chars, &mut pos);
                if pos != chars.len() {
                    return Err(self.json_error("Unexpected token in JSON"));
                }
                value
            }
            // `JSON.rawJSON(text)` — validates `ToString(text)` as a single JSON
            // primitive (number/string/boolean/null, no surrounding whitespace) and
            // returns a frozen, null-prototype object `{ rawJSON: text }` carrying an
            // internal RawJSON brand. A BigInt argument is a TypeError (ToString of a
            // BigInt is allowed, but the resulting "123" parses fine — the spec only
            // forbids non-primitive/whitespace text).
            N_JSON_RAW => {
                // ToString(value); a Symbol throws (ToString of a Symbol is a
                // TypeError), handled by `coerce_to_string`.
                let text = self.coerce_to_string(arg(0))?;
                // Reject empty text or any leading/trailing JSON whitespace.
                let is_ws = |c: char| matches!(c, '\u{20}' | '\u{09}' | '\u{0A}' | '\u{0D}');
                let first = text.chars().next();
                let last = text.chars().next_back();
                if text.is_empty() || first.is_some_and(is_ws) || last.is_some_and(is_ws) {
                    return Err(self.json_error("Invalid raw JSON text"));
                }
                // Must parse as exactly one JSON value, and that value must be a
                // primitive (not an object/array).
                let chars: Vec<char> = text.chars().collect();
                let mut pos = 0;
                let parsed = self.json_parse(&chars, &mut pos, 0)?;
                skip_ws(&chars, &mut pos);
                if pos != chars.len() {
                    return Err(self.json_error("Invalid raw JSON text"));
                }
                if parsed
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.is_array(h) || self.realm.object_keys(h).is_some())
                {
                    return Err(self.json_error("Raw JSON must be a primitive value"));
                }
                // A null-prototype object with a single own data property `rawJSON`
                // = the text, plus the internal brand; then frozen.
                let obj = self.realm.new_object();
                self.realm.set_object_proto(obj, None);
                let text_box = self.new_str(&text);
                self.realm.set_property(obj, "rawJSON", text_box);
                self.realm
                    .set_hidden_property(obj, RAW_JSON_BRAND, NanBox::boolean(true));
                self.realm.freeze_object(obj);
                NanBox::handle(obj.to_raw())
            }
            // `JSON.isRawJSON(value)` — whether `value` is an object carrying the
            // internal RawJSON brand.
            N_JSON_IS_RAW => {
                let is_raw = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.get_property(h, RAW_JSON_BRAND))
                    .is_some_and(|v| self.realm.truthy(v));
                NanBox::boolean(is_raw)
            }
            N_ERROR_IS_ERROR => {
                // `Error.isError(arg)` (ES2025): `true` iff `arg` is an object
                // carrying the `[[ErrorData]]` brand (see `ERROR_DATA`) — a
                // genuine Error instance. The brand is an own (hidden) property,
                // so a primitive, a plain `Object.create(Error.prototype)`, or a
                // `Proxy` wrapping an Error (the slot is not forwarded) all yield
                // `false`. Never throws.
                let is_err = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.has_own(h, ERROR_DATA));
                NanBox::boolean(is_err)
            }
            N_ERROR_PROTO_STACK_GET => {
                // `get Error.prototype.stack` (error-stack-accessor):
                //   1. If `this` is not an Object, throw a TypeError.
                //   2. If `this` lacks an `[[ErrorData]]` slot, return undefined.
                //   3. Else return an implementation string for its stack trace.
                let this = self.this_val;
                if !self.is_object_value(this) {
                    return Err(self.type_error("get Error.prototype.stack called on non-object"));
                }
                let h = Handle::from_raw(this.as_handle().unwrap());
                if !self.realm.has_own(h, ERROR_DATA) {
                    return Ok(NanBox::undefined());
                }
                // The proposal leaves the format implementation-defined; render the
                // error's `"name: message"` first line (the same shape as
                // `Error.prototype.toString`), which the conformance tests accept as
                // a non-empty String.
                let name = self.read_member(h, "name")?;
                let name = self.realm.to_display_string(name);
                let msg = self.read_member(h, "message")?;
                let msg = self.realm.to_display_string(msg);
                let line = if msg.is_empty() {
                    name
                } else if name.is_empty() {
                    msg
                } else {
                    alloc::format!("{name}: {msg}")
                };
                self.new_str(&line)
            }
            N_ERROR_PROTO_STACK_SET => {
                // `set Error.prototype.stack` (error-stack-accessor):
                //   1. If `this` is not an Object, throw a TypeError.
                //   2. If `v` is not a String, throw a TypeError.
                //   3. SetterThatIgnoresPrototypeProperties(this, %Error.prototype%,
                //      "stack", v): throw if `this` IS %Error.prototype%; else update
                //      an existing own "stack" via [[Set]] (Throw=true) or create a
                //      fresh writable/enumerable/configurable own data property.
                let this = self.this_val;
                if !self.is_object_value(this) {
                    return Err(self.type_error("set Error.prototype.stack called on non-object"));
                }
                let v = arg(0);
                let is_string = v
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.string_value(h).is_some());
                if !is_string {
                    return Err(
                        self.type_error("set Error.prototype.stack requires a String value")
                    );
                }
                let h = Handle::from_raw(this.as_handle().unwrap());
                // SameValue(this, %Error.prototype%) — assignment to the home object
                // emulates a non-writable data property and throws.
                let error_proto = self
                    .current
                    .get("Error")
                    .and_then(|c| c.as_handle())
                    .map(Handle::from_raw)
                    .and_then(|c| self.realm.get_property(c, "prototype"))
                    .and_then(|p| p.as_handle());
                if error_proto == Some(h.to_raw()) {
                    return Err(self.type_error(
                        "Cannot assign to read only property 'stack' of Error.prototype",
                    ));
                }
                if self.realm.has_own(h, "stack") {
                    // [[Set]] with Throw=true. An own accessor with no setter cannot
                    // be written: throw (the assignment path would silently no-op).
                    if let Some((_, setter)) = self.realm.accessor(h, "stack")
                        && matches!(setter.unpack(), Unpacked::Undefined)
                    {
                        return Err(
                            self.type_error("Cannot set property 'stack' which has only a getter")
                        );
                    }
                    // A getter-only own accessor is handled above; a non-writable own
                    // data property must also throw even in sloppy code, so force
                    // strict semantics for this write.
                    let key = self.new_str("stack");
                    let saved = self.strict;
                    self.strict = true;
                    let r = self.assign_member_value(h, key, v);
                    self.strict = saved;
                    r?;
                } else {
                    // CreateDataPropertyOrThrow: fails (TypeError) on a non-extensible
                    // receiver.
                    let desc = self.realm.new_object();
                    self.realm.set_property(desc, "value", v);
                    self.realm
                        .set_property(desc, "writable", NanBox::boolean(true));
                    self.realm
                        .set_property(desc, "enumerable", NanBox::boolean(true));
                    self.realm
                        .set_property(desc, "configurable", NanBox::boolean(true));
                    if !self.apply_descriptor(h, "stack", desc, true)? {
                        return Err(self.type_error(
                            "Cannot create property 'stack' on a non-extensible object",
                        ));
                    }
                }
                NanBox::undefined()
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
                // `Object.keys` (EnumerableOwnPropertyNames) calls `[[GetOwnProperty]]`
                // per key; a module namespace with any TDZ export throws.
                #[cfg(all(feature = "module", feature = "std"))]
                if let Some(h) = target {
                    self.namespace_enumeration_tdz(h)?;
                }
                let mut keys: Vec<alloc::string::String> = Vec::new();
                if let Some(h) = target {
                    // A String (primitive or wrapper): ToObject exposes index
                    // properties `"0".."length-1"` (each a UTF-16 code unit).
                    if let Some(n) = self.string_index_count(h) {
                        for i in 0..n {
                            keys.push(alloc::format!("{i}"));
                        }
                    }
                    // An array's own enumerable keys are its integer indices (in
                    // ascending order) — stored as elements, not named properties.
                    // A VM closure backs onto an array but is a function, so its
                    // "indices" (captured cells) are not enumerable keys.
                    if let Some(indices) = self.realm.array_enumerable_indices(h)
                        && !self.realm.is_vm_function(h)
                    {
                        for i in indices {
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
                // A proxy routes SetIntegrityLevel through its traps (ownKeys +
                // getOwnPropertyDescriptor + defineProperty, per key).
                if let Some(raw) = arg(0).as_handle()
                    && self.realm.proxy_at(Handle::from_raw(raw)).is_some()
                {
                    self.proxy_set_integrity_level(Handle::from_raw(raw), true)?;
                    return Ok(arg(0));
                }
                if let Some(raw) = arg(0).as_handle() {
                    // SetIntegrityLevel(frozen) calls `DefinePropertyOrThrow(k,
                    // {writable:false, configurable:false})` per own key. A module
                    // namespace's string exports are writable data properties whose
                    // `[[DefineOwnProperty]]` rejects `writable:false`, so freezing a
                    // namespace with any string export throws a TypeError.
                    #[cfg(all(feature = "module", feature = "std"))]
                    if self
                        .module_namespaces
                        .get(&raw)
                        .is_some_and(|m| !m.is_empty())
                    {
                        return Err(self.type_error(
                            "Cannot freeze a module namespace object (its exports are writable)",
                        ));
                    }
                    self.realm.freeze_object(Handle::from_raw(raw));
                }
                arg(0) // returns the (now frozen) object
            }
            N_OBJECT_SEAL => {
                if let Some(raw) = arg(0).as_handle()
                    && self.realm.proxy_at(Handle::from_raw(raw)).is_some()
                {
                    self.proxy_set_integrity_level(Handle::from_raw(raw), false)?;
                    return Ok(arg(0));
                }
                if let Some(raw) = arg(0).as_handle() {
                    self.realm.seal_object(Handle::from_raw(raw));
                }
                arg(0)
            }
            N_OBJECT_PREVENT_EXT => {
                // Object.preventExtensions returns its argument (a primitive is a
                // no-op pass-through); a proxy routes through its trap, and a `false`
                // [[PreventExtensions]] result is a TypeError.
                if let Some(raw) = arg(0).as_handle().filter(|_| self.is_object_value(arg(0)))
                    && !self.prevent_extensions_of(Handle::from_raw(raw))?
                {
                    return Err(self.type_error("Object.preventExtensions failed"));
                }
                arg(0)
            }
            N_OBJECT_IS_SEALED => {
                // A non-object argument (a primitive) is reported as sealed.
                let v = arg(0);
                if let Some(raw) = v.as_handle()
                    && self.realm.proxy_at(Handle::from_raw(raw)).is_some()
                {
                    let r = self.proxy_test_integrity_level(Handle::from_raw(raw), false)?;
                    return Ok(NanBox::boolean(r));
                }
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
                // Only an actual Object's prototype is set; a primitive `O` (a
                // String/Symbol/BigInt heap value passes RequireObjectCoercible but
                // is not an Object) is returned unchanged.
                if let Some(raw) = arg(0).as_handle().filter(|_| self.is_object_value(arg(0))) {
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
                #[cfg(all(feature = "module", feature = "std"))]
                self.trigger_deferred_namespace(obj, &key)?;
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
                // A module namespace's `[[GetOwnProperty]]` of a TDZ export throws.
                #[cfg(all(feature = "module", feature = "std"))]
                self.namespace_binding_tdz(target, &key)?;
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
                // RequireObjectCoercible(items): a `null`/`undefined` is a TypeError.
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Object.groupBy called on null or undefined"));
                }
                // IsCallable(callbackfn) is checked *before* iterating.
                let cb = arg(1);
                self.require_callable(cb, "Object.groupBy callback")?;
                let items = self.iterate_values(arg(0))?;
                // The result is a null-prototype object; group keys are property keys
                // (ToPropertyKey — a Symbol key stays a Symbol).
                let out = self.realm.new_object();
                self.realm.set_object_proto(out, None);
                for (i, item) in items.iter().enumerate() {
                    let key = self.call(cb, &[*item, NanBox::number(i as f64)])?;
                    let k = self.coerce_property_key(key)?;
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
                    // A module namespace exotic object's `[[Set]]` always fails.
                    #[cfg(all(feature = "module", feature = "std"))]
                    if self.module_namespaces.contains_key(&h.to_raw()) {
                        return Ok(NanBox::boolean(false));
                    }
                    let key = self.coerce_property_key(arg(1))?;
                    let value = arg(2);
                    // The receiver defaults to the target; an explicit one (4th arg)
                    // receives the write / is the setter's `this`.
                    let receiver = if args.len() > 3 {
                        arg(3)
                    } else {
                        NanBox::handle(h.to_raw())
                    };
                    // Integer-indexed exotic `[[Set]]`: a canonical numeric index on a
                    // typed-array target never consults the prototype chain.
                    //   - SameValue(O, Receiver): IntegerIndexedElementSet, return true
                    //     (an invalid/out-of-bounds index is a no-op).
                    //   - different Receiver, *invalid* index: return true (no write).
                    //   - different Receiver, *valid* index: fall to the ordinary
                    //     [[Set]] on the receiver (handled below).
                    if self.realm.typed_kind(h).is_some()
                        && let Some(n) = canonical_numeric_index(&key)
                    {
                        let is_neg_zero = n == 0.0 && n.is_sign_negative();
                        let index_ok = !is_neg_zero && n == (n as i64) as f64 && n >= 0.0;
                        let valid = index_ok
                            && !self.typed_array_detached(h)
                            && self
                                .realm
                                .typed_len(h)
                                .is_some_and(|len| (n as usize) < len);
                        let same_receiver = receiver.as_handle() == Some(h.to_raw());
                        if same_receiver {
                            // IntegerIndexedElementSet: coerce the value FIRST (its
                            // `valueOf`/`toString` side effects — which may detach or
                            // resize the buffer — run unconditionally), then write only
                            // if the index is *still* a valid integer index. A write
                            // whose coercion detached the buffer is dropped, but the
                            // whole [[Set]] still reports success (`true`).
                            let coerced = if self.realm.typed_kind(h).is_some_and(is_bigint_kind) {
                                self.coerce_typed_array_write(h, value)?
                            } else {
                                self.coerce_to_number(value)?
                            };
                            let still_valid = index_ok
                                && !self.typed_array_detached(h)
                                && self
                                    .realm
                                    .typed_len(h)
                                    .is_some_and(|len| (n as usize) < len);
                            if still_valid {
                                self.guard_view_immutable(h)?;
                                self.realm.set_element(h, n as usize, coerced);
                            }
                            return Ok(NanBox::boolean(true));
                        }
                        if !valid {
                            // Different receiver + invalid index: a no-op success
                            // (never reaches the prototype chain).
                            return Ok(NanBox::boolean(true));
                        }
                        // Different receiver + valid index: ordinary set on receiver.
                    }
                    // A setter accessor found on the chain runs with `receiver` as
                    // `this` (an accessor with no setter fails).
                    let mut cur = Some(h);
                    while let Some(c) = cur {
                        // Integer-indexed exotic `[[Set]]` at a *typed-array* chain
                        // node `O = c` (10.4.5.5): a canonical numeric index never
                        // consults an inherited setter and is governed by O's bounds.
                        //   - SameValue(O, Receiver): TypedArraySetElement — coerce V
                        //     (side effects run), write only if still a valid index,
                        //     and always report success.
                        //   - O ≠ Receiver, *invalid* index: a silent success (no
                        //     coercion, no write) — the property never reaches the
                        //     receiver.
                        //   - O ≠ Receiver, *valid* index: fall through to OrdinarySet,
                        //     which creates the data property on the *receiver* below.
                        if self.realm.typed_kind(c).is_some()
                            && let Some(n) = canonical_numeric_index(&key)
                        {
                            let index_ok = n == (n as i64) as f64
                                && n >= 0.0
                                && !(n == 0.0 && n.is_sign_negative());
                            let valid = index_ok
                                && !self.typed_array_detached(c)
                                && self
                                    .realm
                                    .typed_len(c)
                                    .is_some_and(|len| (n as usize) < len);
                            if receiver.as_handle() == Some(c.to_raw()) {
                                let coerced =
                                    if self.realm.typed_kind(c).is_some_and(is_bigint_kind) {
                                        self.coerce_typed_array_write(c, value)?
                                    } else {
                                        self.coerce_to_number(value)?
                                    };
                                let still_valid = index_ok
                                    && !self.typed_array_detached(c)
                                    && self
                                        .realm
                                        .typed_len(c)
                                        .is_some_and(|len| (n as usize) < len);
                                if still_valid {
                                    self.guard_view_immutable(c)?;
                                    self.realm.set_element(c, n as usize, coerced);
                                }
                                return Ok(NanBox::boolean(true));
                            }
                            if !valid {
                                return Ok(NanBox::boolean(true));
                            }
                            break;
                        }
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
                    // No setter: write the data property on the receiver.
                    // A disallowed write (read-only / non-extensible) returns false.
                    let Some(rh) = receiver.as_handle() else {
                        return Ok(NanBox::boolean(false));
                    };
                    let rh = Handle::from_raw(rh);
                    // A **proxy** receiver: the cell-level `can_write_property` gate
                    // reads it as non-extensible and would reject every write, so
                    // route through the proxy protocol per OrdinarySetWithOwnDescriptor.
                    if self.realm.proxy_at(rh).is_some() {
                        if receiver.as_handle() == Some(h.to_raw()) {
                            // receiver == target: `[[Set]]` on the proxy *is* the
                            // set trap (or trapless forward) — return its actual
                            // boolean result (a falsy trap result is `false`, not a
                            // throw, for `Reflect.set`).
                            let ok = self.proxy_set_bool(rh, &key, value, receiver)?;
                            return Ok(NanBox::boolean(ok));
                        }
                        // receiver != target (the canonical passthrough
                        // `set(t,k,v,r){ Reflect.set(t,k,v,r) }`): the write goes to
                        // the receiver via `[[DefineOwnProperty]]` (CreateDataProperty
                        // / value update) — NOT `[[Set]]`, which would re-enter the
                        // set trap and recurse forever.
                        let existing = self.descriptor_of(rh, &key)?;
                        let desc = self.realm.new_object();
                        self.realm.set_property(desc, "value", value);
                        if let Some(dh) = existing.as_handle().map(Handle::from_raw) {
                            // An existing accessor, or a non-writable data property,
                            // rejects the set.
                            let is_accessor = self.realm.get_property(dh, "get").is_some()
                                || self.realm.get_property(dh, "set").is_some();
                            let writable = self
                                .realm
                                .get_property(dh, "writable")
                                .is_some_and(|v| self.realm.truthy(v));
                            if is_accessor || !writable {
                                return Ok(NanBox::boolean(false));
                            }
                            // Update just the value (leave the other attributes).
                        } else {
                            // CreateDataProperty: value + writable/enumerable/
                            // configurable = true.
                            self.realm
                                .set_property(desc, "writable", NanBox::boolean(true));
                            self.realm
                                .set_property(desc, "enumerable", NanBox::boolean(true));
                            self.realm
                                .set_property(desc, "configurable", NanBox::boolean(true));
                        }
                        let ok = self.apply_descriptor(rh, &key, desc, true)?;
                        return Ok(NanBox::boolean(ok));
                    }
                    // OrdinarySetWithOwnDescriptor final steps for a (non-proxy)
                    // receiver distinct from the target: the target's own
                    // descriptor is a *data* descriptor (an accessor on the target
                    // chain was already handled above). If the receiver already
                    // has an own *accessor* for the key, the write fails (the
                    // setter is NOT invoked); a non-writable own data property also
                    // fails; otherwise the value is written / a data property is
                    // created on the receiver.
                    if receiver.as_handle() != Some(h.to_raw()) {
                        // CreateDataProperty(Receiver, P, V) when the receiver is a
                        // *typed array* and P is a canonical numeric index: the
                        // receiver's integer-indexed [[DefineOwnProperty]] governs.
                        // An index that is not a valid integer index for the receiver
                        // fails the define (returns false) and the value is **not**
                        // coerced; a valid index coerces and writes.
                        if self.realm.typed_kind(rh).is_some()
                            && let Some(rn) = canonical_numeric_index(&key)
                        {
                            let r_valid = rn == (rn as i64) as f64
                                && rn >= 0.0
                                && !(rn == 0.0 && rn.is_sign_negative())
                                && !self.typed_array_detached(rh)
                                && self
                                    .realm
                                    .typed_len(rh)
                                    .is_some_and(|len| (rn as usize) < len);
                            if !r_valid {
                                return Ok(NanBox::boolean(false));
                            }
                            let coerced = if self.realm.typed_kind(rh).is_some_and(is_bigint_kind) {
                                self.coerce_typed_array_write(rh, value)?
                            } else {
                                self.coerce_to_number(value)?
                            };
                            self.guard_view_immutable(rh)?;
                            self.realm.set_element(rh, rn as usize, coerced);
                            return Ok(NanBox::boolean(true));
                        }
                        if self.realm.accessor(rh, &key).is_some() {
                            return Ok(NanBox::boolean(false));
                        }
                        if self.realm.has_own(rh, &key) {
                            if !self.can_write_property(rh, &key) {
                                return Ok(NanBox::boolean(false));
                            }
                        } else if !self.realm.is_extensible(rh) {
                            return Ok(NanBox::boolean(false));
                        }
                        self.assign_member_value(rh, arg(1), value)?;
                        return Ok(NanBox::boolean(true));
                    }
                    if !self.can_write_property(rh, &key) {
                        return Ok(NanBox::boolean(false));
                    }
                    self.assign_member_value(rh, arg(1), value)?;
                }
                NanBox::boolean(true)
            }
            N_REFLECT_HAS => {
                // `Reflect.has(target, key)` = HasProperty — own or anywhere on the
                // prototype chain, honoring a proxy `has` trap at any chain step.
                let target = self.reflect_object_target(arg(0), "has")?;
                let key = self.coerce_property_key(arg(1))?;
                NanBox::boolean(self.has_property_proxied(target, &key)?)
            }
            N_REFLECT_DELETE => {
                // Returns the [[Delete]] result (a proxy routes through its
                // deleteProperty trap; false for a non-configurable property).
                let target = self.reflect_object_target(arg(0), "deleteProperty")?;
                let key = self.coerce_property_key(arg(1))?;
                NanBox::boolean(self.delete_property_of(target, &key)?)
            }
            N_REFLECT_OWN_KEYS => {
                // `[[OwnPropertyKeys]]`: String keys (integer-indexed then
                // insertion order) then own symbol keys. `own_property_keys_values`
                // drives a proxy's `ownKeys` trap — or, for a trapless proxy,
                // forwards to the target — where the raw cell has no physical keys.
                let h = self.reflect_object_target(arg(0), "ownKeys")?;
                #[cfg(all(feature = "module", feature = "std"))]
                self.force_deferred_namespace(h)?;
                let keys = self.own_property_keys_values(h)?;
                NanBox::handle(self.realm.new_array(keys).to_raw())
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
                NanBox::boolean(self.prevent_extensions_of(obj)?)
            }
            // `Reflect.isExtensible(target)` → boolean (target must be an Object).
            N_REFLECT_IS_EXTENSIBLE => {
                let obj = self.reflect_object_target(arg(0), "isExtensible")?;
                self.is_extensible_of(obj)?
            }
            N_REFLECT_APPLY => {
                // The target must be callable; the argumentsList is read via
                // CreateListFromArrayLike (an array-like object, not just an Array).
                if !arg(0)
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(self.type_error("Reflect.apply target is not a function"));
                }
                let list = self.create_list_from_array_like(arg(2))?;
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
                let list = self.create_list_from_array_like(arg(1))?;
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
                // `Let obj be ? ToObject(O)`: null/undefined throws, a primitive is
                // boxed (so its own indices/length get descriptors).
                let obj = self.require_object_coercible_to_object(
                    arg(0),
                    "Object.getOwnPropertyDescriptors",
                )?;
                let out = self.realm.new_object();
                {
                    if self.realm.proxy_at(obj).is_some() {
                        // A proxy: `[[OwnPropertyKeys]]` (ownKeys trap or trapless
                        // forward) then `[[GetOwnProperty]]` per key via
                        // `descriptor_of` (the `getOwnPropertyDescriptor` trap +
                        // FromPropertyDescriptor). Both String and Symbol keys.
                        for key in self.own_property_keys_values(obj)? {
                            let name = self.member_key(key);
                            let d = self.descriptor_of(obj, &name)?;
                            if !matches!(d.unpack(), Unpacked::Undefined) {
                                self.realm.set_property(out, &name, d);
                            }
                        }
                    } else {
                        let mut keys = self.realm.own_property_names(obj).unwrap_or_default();
                        keys.extend(self.realm.object_accessor_keys(obj));
                        // Symbol-keyed properties (stored under their `\0sym:` internal
                        // name) get a descriptor too, set under the symbol key on the
                        // result.
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
                }
                NanBox::handle(out.to_raw())
            }
            N_OBJECT_IS_FROZEN => {
                // A non-object argument (a primitive) is reported as frozen.
                let v = arg(0);
                if let Some(raw) = v.as_handle()
                    && self.realm.proxy_at(Handle::from_raw(raw)).is_some()
                {
                    let r = self.proxy_test_integrity_level(Handle::from_raw(raw), true)?;
                    return Ok(NanBox::boolean(r));
                }
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
                #[cfg(all(feature = "module", feature = "std"))]
                if let Some(raw) = arg(0).as_handle() {
                    self.force_deferred_namespace(Handle::from_raw(raw))?;
                }
                // A proxy drives `[[OwnPropertyKeys]]` through its `ownKeys` trap;
                // keep the String keys.
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_keys_raw(Handle::from_raw(raw))?
                {
                    let boxed: Vec<NanBox> = keys
                        .into_iter()
                        .filter(|k| {
                            k.as_handle()
                                .map(Handle::from_raw)
                                .is_some_and(|h| self.realm.string_value(h).is_some())
                        })
                        .collect();
                    return Ok(NanBox::handle(self.realm.new_array(boxed).to_raw()));
                }
                // A proxy *without* an `ownKeys` trap forwards `[[OwnPropertyKeys]]`
                // to its target (recursing through nested proxies) — so its own
                // string keys are the target's, not the (empty) proxy object's.
                let mut src = arg(0).as_handle().map(Handle::from_raw);
                while let Some(h) = src {
                    match self.realm.proxy_at(h) {
                        Some((target, _)) => src = Some(target),
                        None => break,
                    }
                }
                let names = src
                    .and_then(|h| self.realm.own_property_names(h))
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
                #[cfg(all(feature = "module", feature = "std"))]
                if let Some(raw) = arg(0).as_handle() {
                    self.force_deferred_namespace(Handle::from_raw(raw))?;
                }
                // A proxy drives `[[OwnPropertyKeys]]` through its `ownKeys` trap;
                // keep the Symbol keys.
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_keys_raw(Handle::from_raw(raw))?
                {
                    let boxed: Vec<NanBox> = keys
                        .into_iter()
                        .filter(|k| {
                            k.as_handle()
                                .map(Handle::from_raw)
                                .is_some_and(|h| self.realm.symbol_at(h).is_some())
                        })
                        .collect();
                    return Ok(NanBox::handle(self.realm.new_array(boxed).to_raw()));
                }
                let mut syms = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = Handle::from_raw(raw);
                    // All own symbol keys, including non-enumerable ones (e.g. a
                    // symbol defined via `Object.defineProperty`). Symbol keys on a
                    // genuine object cell live in `object_all_keys`; on a non-object
                    // exotic (array, RegExp, function) they live in the auxiliary
                    // object, so consult both (a cell has at most one, so no
                    // duplication) — otherwise a symbol on an array/RegExp is dropped.
                    let mut seen: Vec<u64> = Vec::new();
                    for k in self
                        .realm
                        .object_all_keys(h)
                        .into_iter()
                        .chain(self.realm.aux_all_keys(h))
                    {
                        if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                            && let Ok(id) = idstr.parse::<u64>()
                            && !seen.contains(&id)
                            && let Some(sh) = self.realm.symbol_for_id(id)
                        {
                            seen.push(id);
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
                // A proxy: per-key `[[GetOwnProperty]]` (descriptor trap) interleaved
                // with the enumerable key's `[[Get]]` (get trap), in spec order.
                if let Some(raw) = arg(0).as_handle()
                    && let Some(entries) = self.proxy_enumerable_entries(Handle::from_raw(raw))?
                {
                    let vals: Vec<NanBox> = entries.into_iter().map(|(_, v)| v).collect();
                    return Ok(NanBox::handle(self.realm.new_array(vals).to_raw()));
                }
                let mut vals = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = self.proxy_key_target(Handle::from_raw(raw));
                    // A String exposes its code units as index values.
                    if let Some(n) = self.string_index_count(h) {
                        for i in 0..n {
                            vals.push(self.read_member(h, &alloc::format!("{i}"))?);
                        }
                    }
                    // Array index values come from element access (ascending) first
                    // — but a VM closure's backing cells are not enumerable values.
                    // Only *present* (non-hole) indices enumerate: a sparse array's
                    // holes are absent own properties, so `[[Get]]` never runs for
                    // them (matching `Object.keys`, which uses the same index set).
                    if !self.realm.is_vm_function(h)
                        && let Some(indices) = self.realm.array_enumerable_indices(h)
                    {
                        for i in indices {
                            vals.push(self.read_member(h, &alloc::format!("{i}"))?);
                        }
                    }
                    let named = self
                        .realm
                        .object_keys(h)
                        .unwrap_or_else(|| self.realm.aux_named_keys(h));
                    for k in named {
                        // EnumerableOwnProperties re-runs `[[GetOwnProperty]]` per
                        // key: an earlier key's getter may have deleted this key or
                        // made it non-enumerable (then it is skipped), and `[[Get]]`
                        // must invoke the getter (a raw read would miss it).
                        if !self.realm.has_own(h, &k) || !self.realm.property_is_enumerable(h, &k) {
                            continue;
                        }
                        vals.push(self.read_member(h, &k)?);
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
                // ToObject(target): `null`/`undefined` throws; a primitive is boxed
                // to its wrapper object (the return value is that object).
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Object.assign target is null or undefined"));
                }
                let target = self.coerce_to_object(arg(0));
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
                                let name = alloc::format!("{i}");
                                let kb = self.new_str(&name);
                                self.set_or_throw(t, kb, &name, cv)?;
                            }
                            continue;
                        }
                        if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                            // Spec CopyDataProperties: take `from.[[OwnPropertyKeys]]()`
                            // in String-then-Symbol order (a proxy's `ownKeys` trap is
                            // honored), and for each key do `from.[[GetOwnProperty]]`
                            // (skip a key reported absent, or non-enumerable), then
                            // `Get` (so a getter / proxy `get` trap fires and its throw
                            // propagates) and `Set(to, key, value, true)` onto the
                            // target (which runs the target's setters and throws on a
                            // read-only / frozen / non-extensible write).
                            let keys = self.own_property_keys_values(sh)?;
                            for key in keys {
                                let name = self.member_key(key);
                                let desc = self.descriptor_of(sh, &name)?;
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
                                let v = self.read_member(sh, &name)?;
                                self.set_or_throw(t, key, &name, v)?;
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
                // A proxy: per-key `[[GetOwnProperty]]` (descriptor trap) interleaved
                // with the enumerable key's `[[Get]]` (get trap), in spec order.
                if let Some(raw) = arg(0).as_handle()
                    && let Some(entries) = self.proxy_enumerable_entries(Handle::from_raw(raw))?
                {
                    let mut pairs = Vec::with_capacity(entries.len());
                    for (k, v) in entries {
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
                    // A String exposes its code units as index entries.
                    if let Some(n) = self.string_index_count(h) {
                        for i in 0..n {
                            let v = self.read_member(h, &alloc::format!("{i}"))?;
                            entries.push((alloc::format!("{i}"), v));
                        }
                    }
                    // Array index entries (ascending) before named ones — but a VM
                    // closure's backing cells are not enumerable entries. Holes are
                    // absent own properties, so a sparse array's missing indices
                    // contribute no entry (same index set as `Object.keys`).
                    if !self.realm.is_vm_function(h)
                        && let Some(indices) = self.realm.array_enumerable_indices(h)
                    {
                        for i in indices {
                            let v = self.read_member(h, &alloc::format!("{i}"))?;
                            entries.push((alloc::format!("{i}"), v));
                        }
                    }
                    let named = self
                        .realm
                        .object_keys(h)
                        .unwrap_or_else(|| self.realm.aux_named_keys(h));
                    for k in named {
                        // EnumerableOwnProperties re-runs `[[GetOwnProperty]]` per
                        // key (an earlier getter may have deleted it or turned off
                        // enumerability) and `[[Get]]` must invoke the getter.
                        if !self.realm.has_own(h, &k) || !self.realm.property_is_enumerable(h, &k) {
                            continue;
                        }
                        let v = self.read_member(h, &k)?;
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
                // `Array.from(items, mapFn?, thisArg?)`. Step 1-2: a defined
                // `mapFn` must be callable (a TypeError otherwise, checked first).
                let map_fn = arg(1);
                let has_map = !matches!(map_fn.unpack(), Unpacked::Undefined);
                if has_map {
                    self.require_callable(map_fn, "Array.from mapFn")?;
                }
                let this_arg = arg(2);
                // Step 3: items is ToObject'd — `null`/`undefined` is a TypeError.
                if matches!(arg(0).unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(self.type_error(
                        "Array.from requires an array-like or iterable object, not null/undefined",
                    ));
                }
                // GetMethod(items, @@iterator): the getter fires (a throw
                // propagates). A present, callable iterator method → iterate;
                // otherwise fall back to the array-like (length + indices) read.
                // ToObject the primitive first (strings/objects are already heap
                // handles; number/boolean/bigint/symbol box to their wrapper) so a
                // custom `Number.prototype[@@iterator]` etc. is found — matching
                // spec's `GetMethod` on the ToObject'd value.
                let items_box = arg(0);
                let items_box = if items_box.as_handle().is_some() {
                    items_box
                } else {
                    self.coerce_to_object(items_box)
                };
                let is_iterable = match items_box.as_handle().map(Handle::from_raw) {
                    Some(h) => {
                        // A user/inherited callable `@@iterator` (its getter fires,
                        // a throw propagating), OR a built-in iterable whose
                        // iteration `iterate_values` handles directly without a
                        // readable `@@iterator` method (arrays, strings, Set/Map,
                        // generators).
                        let has_iter = self
                            .find_iterator_fn(h)?
                            .filter(|f| {
                                f.as_handle()
                                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                            })
                            .is_some();
                        has_iter
                            || self.realm.is_array_like(h)
                            || self.realm.string_value(h).is_some()
                            || self.realm.collection_is_set(h).is_some()
                            || self.realm.get_property(h, GEN_BUF).is_some()
                    }
                    None => false,
                };
                // When `Array.from` is invoked with a constructor `this` (e.g.
                // `Array.from.call(C, …)` or a subclass `C.from(…)`), the result is
                // `Construct(C, …)`. Per spec (23.1.2.1) the target is constructed
                // *before* iteration begins (a throwing constructor aborts before the
                // iterator is even obtained). The iterator path constructs with **no
                // arguments** (`Construct(C)`); the array-like path constructs with
                // `«len»`. Only the default `%Array%` / a non-constructor `this`
                // builds a plain dense array.
                let this_c = self.this_val;
                let is_array_ctor =
                    self.current.get("Array").and_then(|v| v.as_handle()) == this_c.as_handle();
                let use_ctor = self.is_constructor_value(this_c) && !is_array_ctor;
                // Helper: unwrap a freshly-`Construct`ed target to its handle.
                if is_iterable {
                    // Iterator path: `A = IsConstructor(C) ? Construct(C) : ArrayCreate(0)`
                    // is performed **before** `GetIterator` — a throwing constructor
                    // aborts before the iterator is obtained, and `Construct(C)` takes
                    // no arguments (so `Array.from.call(Object, [])` yields `{}`, not
                    // `new Object(0)`).
                    let target_h = if use_ctor {
                        let target = self.construct(this_c, &[])?;
                        match target.as_handle().map(Handle::from_raw) {
                            Some(th) => Some(th),
                            None => {
                                return Err(self.type_error(
                                    "Array.from constructor did not return an object",
                                ));
                            }
                        }
                    } else {
                        None
                    };
                    // Iterate LAZILY, applying mapFn per element interleaved with
                    // IteratorStep, so an abrupt mapFn completion IteratorCloses the
                    // still-open iterator (a `next`/getter throw propagates directly).
                    let it = self.get_iter_object(items_box)?;
                    let next = self.read_member(it, "next")?;
                    let mut out = Vec::new();
                    let mut k = 0usize;
                    loop {
                        let val = match self.iter_step(it, next)? {
                            Some(v) => v,
                            None => break,
                        };
                        let mapped = if has_map {
                            match self.call_with_this(
                                map_fn,
                                this_arg,
                                &[val, NanBox::number(k as f64)],
                            ) {
                                Ok(m) => m,
                                Err(e) => {
                                    let _ = self.iterator_close(it);
                                    return Err(e);
                                }
                            }
                        } else {
                            val
                        };
                        // A `CreateDataPropertyOrThrow` failure also closes the iterator.
                        if let Some(th) = target_h {
                            if let Err(e) = self.create_data_property_or_throw(th, k, mapped) {
                                let _ = self.iterator_close(it);
                                return Err(e);
                            }
                        } else {
                            out.push(mapped);
                        }
                        k += 1;
                    }
                    if let Some(th) = target_h {
                        let len_key = self.new_str("length");
                        self.assign_member_value(th, len_key, NanBox::number(k as f64))?;
                        NanBox::handle(th.to_raw())
                    } else {
                        NanBox::handle(self.realm.new_array(out).to_raw())
                    }
                } else {
                    // Array-like: `len = LengthOfArrayLike(items)`, then
                    // `A = IsConstructor(C) ? Construct(C, «len») : ArrayCreate(len)`,
                    // then `Get`/mapFn/`CreateDataPropertyOrThrow` each index.
                    let mut out = Vec::new();
                    let mut target_h = None;
                    let mut final_len = 0usize;
                    if let Some(h) = items_box.as_handle().map(Handle::from_raw) {
                        let len_val = self.read_member(h, "length")?;
                        let len_num = self.coerce_to_number(len_val)?;
                        let len_raw = self.realm.to_number(len_num);
                        // Cap the array-like length against `max_array_len` BEFORE
                        // allocating, so `from({length: 2**32-1})` throws a catchable
                        // RangeError instead of a multi-gigabyte allocation.
                        if len_raw > self.realm.limits.max_array_len as f64 {
                            let m = self.new_str("Invalid array length");
                            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                        }
                        let len = if len_raw.is_nan() || len_raw <= 0.0 {
                            0
                        } else {
                            len_raw as usize
                        };
                        final_len = len;
                        if use_ctor {
                            let target = self.construct(this_c, &[NanBox::number(len as f64)])?;
                            target_h = Some(match target.as_handle().map(Handle::from_raw) {
                                Some(th) => th,
                                None => {
                                    return Err(self.type_error(
                                        "Array.from constructor did not return an object",
                                    ));
                                }
                            });
                        }
                        for i in 0..len {
                            let v = self.read_member(h, &alloc::format!("{i}"))?;
                            let v = if has_map {
                                self.call_with_this(
                                    map_fn,
                                    this_arg,
                                    &[v, NanBox::number(i as f64)],
                                )?
                            } else {
                                v
                            };
                            match target_h {
                                Some(th) => self.create_data_property_or_throw(th, i, v)?,
                                None => out.push(v),
                            }
                        }
                    }
                    if let Some(th) = target_h {
                        // `Set(A, "length", len, true)`.
                        let len_key = self.new_str("length");
                        self.assign_member_value(th, len_key, NanBox::number(final_len as f64))?;
                        NanBox::handle(th.to_raw())
                    } else {
                        NanBox::handle(self.realm.new_array(out).to_raw())
                    }
                }
            }
            // `Array.fromAsync(asyncItems, mapFn?, thisArg?)` — returns a promise.
            // Its body runs eagerly (awaiting each value through the event loop, per
            // the engine's eager-async model); the result array fulfills the promise
            // and any thrown value (bad mapFn, null items, an iteration/await throw)
            // rejects it, so `fromAsync` never throws synchronously.
            N_ARRAY_FROM_ASYNC => {
                let this_ctor = self.this_val;
                let computed = self.array_from_async_core(arg(0), arg(1), arg(2), this_ctor);
                let p = self.fresh_promise();
                match computed {
                    Ok(v) => self.resolve_with(p, v),
                    Err(ExecError::Throw(e)) => self.settle(p, e, false),
                    Err(other) => return Err(other),
                }
                NanBox::handle(p.to_raw())
            }
            // `RegExp.escape(S)` (ES2025): `S` must be a String (no coercion);
            // returns it escaped so it matches literally inside a pattern.
            N_REGEXP_ESCAPE => {
                let s = arg(0);
                let Some(bytes) = s
                    .as_handle()
                    .and_then(|r| self.realm.string_bytes(Handle::from_raw(r)))
                else {
                    return Err(self.type_error("RegExp.escape requires a string argument"));
                };
                let out = regexp_escape_wtf8(&bytes);
                self.new_str_bytes(out)
            }
            // The ES2025 `uint8array-base64` proposal: `this`/argument validation,
            // option-bag reads, and the codec all live in the `base64` module.
            N_UINT8_TO_BASE64 => self.uint8_to_base64(arg(0))?,
            N_UINT8_TO_HEX => self.uint8_to_hex()?,
            N_UINT8_SET_FROM_BASE64 => self.uint8_set_from_base64(arg(0), arg(1))?,
            N_UINT8_SET_FROM_HEX => self.uint8_set_from_hex(arg(0))?,
            N_UINT8_FROM_BASE64 => self.uint8_from_base64(arg(0), arg(1))?,
            N_UINT8_FROM_HEX => self.uint8_from_hex(arg(0))?,
            N_ARRAY_OF => {
                // `Array.of(...items)`: when called with a constructor `this`
                // (`Array.of.call(C, …)` or a subclass `C.of(…)`), the result is
                // `Construct(C, «len»)` populated with `CreateDataPropertyOrThrow`
                // then `Set(A, "length", len)`; a default `%Array%` /
                // non-constructor `this` builds a plain dense array. Mirrors the
                // `Array.from` receiver handling.
                let this_c = self.this_val;
                let is_array_ctor =
                    self.current.get("Array").and_then(|v| v.as_handle()) == this_c.as_handle();
                if self.is_constructor_value(this_c) && !is_array_ctor {
                    let len = args.len();
                    let target = self.construct(this_c, &[NanBox::number(len as f64)])?;
                    let Some(th) = target.as_handle().map(Handle::from_raw) else {
                        return Err(self
                            .type_error("Array.of requires its 'this' value to return an object"));
                    };
                    for (i, e) in args.iter().enumerate() {
                        self.create_data_property_or_throw(th, i, *e)?;
                    }
                    let len_key = self.new_str("length");
                    self.assign_member_value(th, len_key, NanBox::number(len as f64))?;
                    return Ok(target);
                }
                NanBox::handle(self.realm.new_array(args.to_vec()).to_raw())
            }
            // `%IteratorPrototype%[Symbol.iterator]()` — an iterator is its own
            // iterable: return the receiver.
            N_ITERATOR_PROTO_SELF => self.this_val,
            // `String.prototype[Symbol.iterator]()` — RequireObjectCoercible +
            // ToString the receiver (a poisoned `toString` or a null/undefined
            // `this` throws), then a String Iterator over its code points. The
            // WTF-8 byte form is preserved so a lone surrogate iterates as one
            // code unit.
            N_STRING_PROTO_ITER => {
                let this = self.this_val;
                if matches!(this.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(self.type_error(
                        "String.prototype[Symbol.iterator] called on null or undefined",
                    ));
                }
                let bytes = self.coerce_to_string_bytes(this)?;
                let sval = self.new_str_bytes(bytes);
                let vals = self.iterate_values(sval)?;
                self.make_builtin_iterator(vals, "String Iterator")
            }
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
                self.iterator_from(src)?
            }
            // `%IteratorHelperPrototype%.next` — lazy helper step.
            N_ITER_HELPER_NEXT => {
                let this = self.this_val;
                self.iter_helper_next(this)?
            }
            // `%IteratorHelperPrototype%.return` — close the helper + source.
            N_ITER_HELPER_RETURN => {
                let this = self.this_val;
                self.iter_helper_return(this)?
            }
            // `%WrapForValidIteratorPrototype%.next` — forward to the wrapped
            // iterator's `next`.
            N_ITER_WRAP_NEXT => {
                let this = self.this_val;
                self.iter_wrap_next(this)?
            }
            // `%WrapForValidIteratorPrototype%.return` — forward to the wrapped
            // iterator's `return` (if any), else `{ value: undefined, done: true }`.
            N_ITER_WRAP_RETURN => {
                let this = self.this_val;
                self.iter_wrap_return(this)?
            }
            // `Iterator.concat(...iterables)` — lazy sequencing.
            N_ITERATOR_CONCAT => self.iterator_concat(args)?,
            // `%ConcatIteratorPrototype%.next` — advance the concat result.
            N_ITER_CONCAT_NEXT => {
                let this = self.this_val;
                self.iter_concat_next(this)?
            }
            // `%ConcatIteratorPrototype%.return` — close the active inner iterator.
            N_ITER_CONCAT_RETURN => {
                let this = self.this_val;
                self.iter_concat_return(this)?
            }
            // `%IteratorPrototype%[Symbol.dispose]()` — call the iterator's `return`
            // (GetMethod, so undefined/null is a no-op), then return undefined.
            N_ITERATOR_DISPOSE => {
                let this = self.this_val;
                if let Some(h) = this.as_handle().map(Handle::from_raw) {
                    let ret = self.read_member(h, "return")?;
                    if ret
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                    {
                        self.call_with_this(ret, this, &[])?;
                    } else if !matches!(ret.unpack(), Unpacked::Undefined | Unpacked::Null) {
                        return Err(self.type_error("Iterator dispose: 'return' is not callable"));
                    }
                }
                NanBox::undefined()
            }
            // `%AsyncIteratorPrototype%[@@asyncDispose]()` — invokes the iterator's
            // `return`, wraps its result with `PromiseResolve`, and returns
            // `resultWrapper.then(() => undefined)` so the dispose promise fulfills
            // with `undefined` (and rejects if `return` yields/throws a rejection). A
            // synchronously-abrupt read/call, or a non-callable `return`, rejects.
            N_ASYNC_ITERATOR_DISPOSE => {
                let this = self.this_val;
                let mut rejection: Option<NanBox> = None;
                let mut wrap_result: Option<NanBox> = None;
                if let Some(h) = this.as_handle().map(Handle::from_raw) {
                    match self.read_member(h, "return") {
                        Ok(ret) => {
                            if ret
                                .as_handle()
                                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                            {
                                match self.call_with_this(ret, this, &[]) {
                                    Ok(res) => wrap_result = Some(res),
                                    Err(ExecError::Throw(v)) => rejection = Some(v),
                                    Err(e) => return Err(e),
                                }
                            } else if !matches!(ret.unpack(), Unpacked::Undefined | Unpacked::Null)
                            {
                                let m = self.new_str(
                                    "AsyncIterator asyncDispose: 'return' is not callable",
                                );
                                rejection = Some(self.make_error(N_TYPE_ERROR, Some(m)));
                            }
                        }
                        Err(ExecError::Throw(v)) => rejection = Some(v),
                        Err(e) => return Err(e),
                    }
                }
                if let Some(v) = rejection {
                    let promise = self.fresh_promise();
                    let reject = self.realm.new_bound_native(N_REJECT, promise);
                    self.call_with_this(
                        NanBox::handle(reject.to_raw()),
                        NanBox::undefined(),
                        &[v],
                    )?;
                    NanBox::handle(promise.to_raw())
                } else if let Some(result) = wrap_result {
                    let wrapper = self.promise_resolve(result);
                    let onf = self.realm.new_native(N_RETURN_UNDEFINED);
                    let then = self.read_member(wrapper, "then")?;
                    self.call_with_this(
                        then,
                        NanBox::handle(wrapper.to_raw()),
                        &[NanBox::handle(onf.to_raw())],
                    )?
                } else {
                    let p = self.promise_resolve(NanBox::undefined());
                    NanBox::handle(p.to_raw())
                }
            }
            // `onFulfilled` of the `@@asyncDispose` `.then`-chain: ignore, yield undefined.
            N_RETURN_UNDEFINED => NanBox::undefined(),
            // `Iterator.zip(iterables, options)` (joint-iteration).
            N_ITERATOR_ZIP => self.iterator_zip(arg(0), arg(1), false)?,
            // `Iterator.zipKeyed(iterables, options)`.
            N_ITERATOR_ZIP_KEYED => self.iterator_zip(arg(0), arg(1), true)?,
            // `%ZipIteratorPrototype%.next` / `.return`.
            N_ITER_ZIP_NEXT => {
                let this = self.this_val;
                self.iter_zip_next(this)?
            }
            N_ITER_ZIP_RETURN => {
                let this = self.this_val;
                self.iter_zip_return(this)?
            }
            // `Object.prototype.__defineGetter__(P, getter)` (Annex B).
            N_OBJ_DEFINE_GETTER | N_OBJ_DEFINE_SETTER => {
                let this = self.this_val;
                let obj = self.require_object_coercible_to_object(this, "__defineGetter__")?;
                let f = arg(1);
                if !f
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Err(
                        self.type_error("Object.prototype.__defineGetter__: Expecting function")
                    );
                }
                let key = self.coerce_property_key(arg(0))?;
                // DefinePropertyOrThrow(O, key, { [get|set]: f, enumerable: true,
                // configurable: true }). Building a real descriptor object and
                // routing through `apply_descriptor` gives the full semantics:
                // a redefine on a non-extensible object or a non-configurable
                // property throws a TypeError, and the half not named by the
                // descriptor (an existing setter for `__defineGetter__`, or getter
                // for `__defineSetter__`) is preserved by ValidateAndApply.
                let desc = self.realm.new_object();
                let field = if id == N_OBJ_DEFINE_GETTER {
                    "get"
                } else {
                    "set"
                };
                self.realm.set_property(desc, field, f);
                self.realm
                    .set_property(desc, "enumerable", NanBox::boolean(true));
                self.realm
                    .set_property(desc, "configurable", NanBox::boolean(true));
                self.apply_descriptor(obj, &key, desc, false)?;
                NanBox::undefined()
            }
            // `Object.prototype.__lookupGetter__(P)` / `__lookupSetter__(P)`.
            N_OBJ_LOOKUP_GETTER | N_OBJ_LOOKUP_SETTER => {
                let this = self.this_val;
                let obj = self.require_object_coercible_to_object(this, "__lookupGetter__")?;
                let key = self.coerce_property_key(arg(0))?;
                let want_getter = id == N_OBJ_LOOKUP_GETTER;
                // Walk the prototype chain via the trap-aware `[[GetOwnProperty]]`
                // and `[[GetPrototypeOf]]` (so a proxy's throwing trap propagates):
                // at each level take its own descriptor; an accessor descriptor
                // returns the requested half, a data descriptor (or chain end)
                // returns undefined.
                let mut cur = obj;
                let mut result = NanBox::undefined();
                loop {
                    let desc = self.descriptor_of(cur, &key)?;
                    if !matches!(desc.unpack(), Unpacked::Undefined)
                        && let Some(dh) = desc.as_handle().map(Handle::from_raw)
                    {
                        // An accessor descriptor carries `get`/`set` keys; a data
                        // descriptor (`value`/`writable`) shadows inherited accessors.
                        if self.realm.has_own(dh, "get") || self.realm.has_own(dh, "set") {
                            let half = if want_getter { "get" } else { "set" };
                            result = self
                                .realm
                                .get_property(dh, half)
                                .unwrap_or(NanBox::undefined());
                        }
                        break;
                    }
                    let proto = self.get_proto_of(cur)?;
                    match proto
                        .as_handle()
                        .map(Handle::from_raw)
                        .filter(|_| self.is_object_value(proto))
                    {
                        Some(p) => cur = p,
                        None => break,
                    }
                }
                result
            }
            // `set RegExp.input` / `set RegExp.$_` (Annex B.2.5): brand-check
            // `this === %RegExp%`, then store ToString(value) as the legacy input.
            N_REGEXP_LEGACY_SET => {
                let this = self.this_val;
                if this.as_handle().map(Handle::from_raw) != Some(self.regexp_constructor_handle()?)
                {
                    return Err(self.type_error(
                        "RegExp legacy static property setter called on a non-RegExp receiver",
                    ));
                }
                let bytes = self.coerce_to_string_bytes(arg(0))?;
                let mut st = self.realm.legacy_regexp().clone();
                st.input = bytes;
                self.realm.set_legacy_regexp(st);
                NanBox::undefined()
            }
            // `get Object.prototype.__proto__` (Annex B).
            N_OBJ_PROTO_GET => {
                let this = self.this_val;
                let obj = self.require_object_coercible_to_object(this, "get __proto__")?;
                self.get_proto_of(obj)?
            }
            // `set Object.prototype.__proto__` (Annex B).
            N_OBJ_PROTO_SET => {
                let this = self.this_val;
                // RequireObjectCoercible(this); a primitive `this` is left unchanged
                // (only objects have a settable prototype).
                if matches!(this.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Object.prototype.__proto__ called on null or undefined");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let v = arg(0);
                // Only `null` or an Object are valid; any other value is ignored.
                if let Some(oh) = this
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|_| self.is_object_value(this))
                {
                    // Per spec, a failed [[SetPrototypeOf]] (a non-extensible
                    // object, or a change that would create a prototype cycle)
                    // throws a TypeError. The setter only acts when V is Null or an
                    // Object; any other value is silently ignored.
                    let proto = match v.unpack() {
                        Unpacked::Null => Some(None),
                        _ if self.is_object_value(v) => Some(v.as_handle().map(Handle::from_raw)),
                        _ => None,
                    };
                    if let Some(p) = proto
                        && !self.set_proto_of(oh, p)?
                    {
                        return Err(self.type_error(
                            "Object.prototype.__proto__: cannot set prototype of this object",
                        ));
                    }
                }
                NanBox::undefined()
            }
            // The eager-generator iterator's `next` — buffer cursor advance.
            N_GEN_ITER_NEXT => {
                let this = self.this_val;
                self.gen_iter_next(this)?
            }
            // The eager-generator iterator's `return` — exhaust + report done.
            N_GEN_ITER_RETURN => {
                let this = self.this_val;
                let v = arg(0);
                self.gen_iter_return(this, v)?
            }
            // A lazy generator's `next`/`return`/`throw` — resume the suspended
            // frame with the appropriate resumption.
            N_GEN_NEXT => {
                let this = self.this_val;
                self.lazy_gen_resume(this, generator::Resumption::Next(arg(0)))?
            }
            N_GEN_RETURN => {
                let this = self.this_val;
                self.lazy_gen_resume(this, generator::Resumption::Return(arg(0)))?
            }
            N_GEN_THROW => {
                let this = self.this_val;
                self.lazy_gen_resume(this, generator::Resumption::Throw(arg(0)))?
            }
            // Async-generator `next`/`return`/`throw`: these ALWAYS return a
            // promise, so a brand-check failure rejects rather than throwing.
            N_ASYNC_GEN_NEXT => {
                let this = self.this_val;
                self.async_gen_resume(this, generator::Resumption::Next(arg(0)))
            }
            N_ASYNC_GEN_RETURN => {
                let this = self.this_val;
                self.async_gen_resume(this, generator::Resumption::Return(arg(0)))
            }
            N_ASYNC_GEN_THROW => {
                let this = self.this_val;
                self.async_gen_resume(this, generator::Resumption::Throw(arg(0)))
            }
            // The abstract `%TypedArray%` intrinsic is not callable directly.
            N_TYPED_ARRAY_ABSTRACT => {
                let m = self.new_str("Abstract class TypedArray not directly constructable");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // `%TypedArray%.from(source, mapFn?, thisArg?)` — generic over the
            // `this` constructor (`Int8Array.from(...)` builds an `Int8Array`).
            N_TYPED_ARRAY_FROM => self.typed_array_from(self.this_val, arg(0), arg(1), arg(2))?,
            // `%TypedArray%.of(...items)` — generic over the `this` constructor.
            N_TYPED_ARRAY_OF => self.typed_array_of(self.this_val, args)?,
            // `get %TypedArray%[Symbol.species]` — returns the receiver constructor.
            // (Also shared by `Map`/`Set`/… species getters, which all return `this`.)
            N_TYPED_ARRAY_SPECIES => self.this_val,
            // `Function.prototype.toString` reached *indirectly* (`.call`/`.apply`,
            // ToPrimitive, `String(fn)`, `fn + ""`). A direct `fn.toString()` is
            // handled by the method-name shortcut. Throws a TypeError on a
            // non-callable `this` (per 20.2.3.5).
            N_FUNCTION_TO_STRING => {
                let this = self.this_val;
                let h = this.as_handle().map(Handle::from_raw).filter(|h| {
                    // `Function.prototype.toString` requires `this` to have a
                    // `[[Call]]` (any callable, or a class — whose `[[Call]]`
                    // throws — or a proxy wrapping one). A proxy wrapping a class
                    // is callable even though the bare class is modeled as a
                    // non-`Function` cell, so accept it explicitly.
                    self.is_callable(*h)
                        || self.realm.class_at(*h).is_some()
                        || self
                            .realm
                            .proxy_at(*h)
                            .is_some_and(|(t, _)| self.realm.class_at(t).is_some())
                });
                let Some(h) = h else {
                    let m = self
                        .new_str("Function.prototype.toString requires that 'this' be a Function");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                };
                let s = self.function_to_string_repr(h)?;
                self.new_str(&s)
            }
            // `get %IteratorPrototype%[Symbol.toStringTag]` → always "Iterator".
            N_ITERATOR_TAG_GET => self.new_str("Iterator"),
            // `set %IteratorPrototype%[Symbol.toStringTag]`: SetterThatIgnores-
            // PrototypeProperties. A non-object `this`, or `this` being the home
            // prototype itself (detected as owning the accessor), is a TypeError;
            // otherwise the value is written as an own property of `this`.
            N_ITERATOR_TAG_SET => {
                let Some(this_h) = self.this_val.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error(
                        "Iterator.prototype[Symbol.toStringTag] setter requires an object",
                    ));
                };
                let tag_sym = self.well_known_symbol("toStringTag");
                let tag_key = self.member_key(tag_sym);
                if self.realm.accessor(this_h, &tag_key).is_some() {
                    return Err(
                        self.type_error("cannot set Symbol.toStringTag on %Iterator.prototype%")
                    );
                }
                self.realm.set_property(this_h, &tag_key, arg(0));
                NanBox::undefined()
            }
            // `get %IteratorPrototype%.constructor` → `%Iterator%`.
            N_ITERATOR_CTOR_GET => self.current.get("Iterator").unwrap_or(NanBox::undefined()),
            // `set %IteratorPrototype%.constructor`: SetterThatIgnoresPrototype-
            // Properties(%Iterator.prototype%, "constructor", v). A non-object
            // `this`, or `this` being the home prototype itself (detected as owning
            // the accessor), is a TypeError; otherwise CreateDataProperty/Set on
            // `this`.
            N_ITERATOR_CTOR_SET => {
                let Some(this_h) = self.this_val.as_handle().map(Handle::from_raw) else {
                    return Err(
                        self.type_error("Iterator.prototype.constructor setter requires an object")
                    );
                };
                if self.realm.accessor(this_h, "constructor").is_some() {
                    return Err(self.type_error("cannot set constructor on %Iterator.prototype%"));
                }
                self.realm.set_property(this_h, "constructor", arg(0));
                NanBox::undefined()
            }
            // `get Map.prototype.size` / `get Set.prototype.size` — brand-checked.
            // `this` must be a non-weak Map (resp. Set); else a TypeError.
            N_MAP_SIZE | N_SET_SIZE => {
                let want_set = id == N_SET_SIZE;
                let ok = self
                    .this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| {
                        self.realm.collection_is_set(h) == Some(want_set)
                            && !self.realm.collection_is_weak(h)
                    });
                if !ok {
                    let what = if want_set { "Set" } else { "Map" };
                    return Err(self.type_error(&alloc::format!(
                        "get {what}.prototype.size called on an incompatible receiver"
                    )));
                }
                let h = Handle::from_raw(self.this_val.as_handle().unwrap());
                NanBox::number(self.realm.collection_size(h).unwrap_or(0) as f64)
            }
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
                if matches!(arg(0).unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Object.fromEntries called on null or undefined"));
                }
                let obj = self.realm.new_object();
                // AddEntriesFromIterable: iterate lazily; each entry must be an
                // Object, and key/value are read via `[[Get]]`. A non-object entry
                // (or a throwing read) closes the iterator (IteratorClose).
                let it = self.get_iter_object(arg(0))?;
                let next = self.read_member(it, "next")?;
                while let Some(entry) = self.iter_step(it, next)? {
                    if !self.is_object_value(entry) {
                        let _ = self.iterator_close(it);
                        return Err(self.type_error("Object.fromEntries entry is not an object"));
                    }
                    let eh = entry.as_handle().map(Handle::from_raw).unwrap();
                    let k = match self.read_member(eh, "0") {
                        Ok(k) => k,
                        Err(e) => {
                            let _ = self.iterator_close(it);
                            return Err(e);
                        }
                    };
                    let v = match self.read_member(eh, "1") {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = self.iterator_close(it);
                            return Err(e);
                        }
                    };
                    // ToPropertyKey(k) may throw (a poisoned `toString`/
                    // `@@toPrimitive`); IfAbruptCloseIterator closes the iterator.
                    let key = match self.coerce_property_key(k) {
                        Ok(key) => key,
                        Err(e) => {
                            let _ = self.iterator_close(it);
                            return Err(e);
                        }
                    };
                    self.realm.set_property(obj, &key, v);
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
            // The binary16 round-trip is pure bit manipulation (no std float
            // intrinsics), so it is available in the no_std core too.
            N_MATH_F16ROUND => {
                NanBox::number(f16_to_f64(f64_to_f16_bits(self.realm.to_number(arg(0)))))
            }
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
            // `Math.sumPrecise(items)` (ES2025) — correctly-rounded exact sum of a
            // sequence of Numbers. The argument must be iterable; each element must
            // already be a Number (no ToNumber coercion — a non-Number element throws
            // a TypeError *and* closes the iterator). Runs the spec's
            // Infinity/NaN/-0 state machine; finite values are accumulated with an
            // exact (error-free) Shewchuk partials list, rounded once at the end.
            // The empty sum (and an all-`-0` sum) is `-0`.
            N_MATH_SUM_PRECISE => {
                let it = self.get_iter_object(arg(0))?;
                let next = self.read_member(it, "next")?;
                // Spec state machine. `state`: 0=minus-zero, 1=finite,
                // 2=plus-infinity, 3=minus-infinity, 4=not-a-number.
                let mut state: u8 = 0;
                let mut acc = ExactSum::new();
                loop {
                    let step = self.iter_step(it, next);
                    let v = match step {
                        Ok(Some(v)) => v,
                        Ok(None) => break,
                        Err(e) => return Err(e),
                    };
                    let Some(n) = v.as_number() else {
                        // Non-Number element: close the iterator, then throw.
                        let _ = self.iterator_close(it);
                        return Err(self.type_error("Math.sumPrecise expects only Numbers"));
                    };
                    if state == 4 {
                        // Already NaN: keep draining but ignore values.
                        continue;
                    }
                    if n.is_nan() {
                        state = 4;
                    } else if n == f64::INFINITY {
                        state = if state == 3 { 4 } else { 2 };
                    } else if n == f64::NEG_INFINITY {
                        state = if state == 2 { 4 } else { 3 };
                    } else if n == 0.0 && n.is_sign_negative() {
                        // -0 does not change state (an all-`-0` sum stays `-0`),
                        // and adds nothing to the exact accumulator.
                    } else if state == 0 || state == 1 {
                        // Any finite value other than -0 (including +0) transitions
                        // minus-zero → finite, so the result becomes +0 not -0.
                        // Adding +0 to the accumulator is a no-op (`add` ignores 0).
                        state = 1;
                        if n != 0.0 {
                            acc.add(n);
                        }
                    }
                }
                if state == 1 && acc.overflowed {
                    // The spec's count/overflow guard: not expected in practice.
                    let m = self.new_str("Math.sumPrecise overflow");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                NanBox::number(match state {
                    4 => f64::NAN,
                    2 => f64::INFINITY,
                    3 => f64::NEG_INFINITY,
                    1 => acc.finish(),
                    // minus-zero: empty sum or all-`-0` input.
                    _ => -0.0,
                })
            }
            N_PARSE_FLOAT => {
                let s = self.coerce_to_string(arg(0))?;
                NanBox::number(parse_float_prefix(s.trim()))
            }
            // URI encoding/decoding. `encodeURI` preserves the URI reserved set
            // on top of the unreserved set that `encodeURIComponent` keeps.
            N_ENCODE_URI_COMPONENT | N_ENCODE_URI => {
                let bytes = self.coerce_to_string_bytes(arg(0))?;
                let extra = if id == N_ENCODE_URI {
                    ";,/?:@&=+$#"
                } else {
                    ""
                };
                match uri_encode(&bytes, extra) {
                    Some(out) => self.new_str_bytes(out),
                    None => {
                        let m = self.new_str("URI malformed");
                        return Err(ExecError::Throw(self.make_error(N_URI_ERROR, Some(m))));
                    }
                }
            }
            N_DECODE_URI_COMPONENT | N_DECODE_URI => {
                let bytes = self.coerce_to_string_bytes(arg(0))?;
                match uri_decode(&bytes, id == N_DECODE_URI) {
                    Some(out) => self.new_str_bytes(out),
                    None => {
                        let m = self.new_str("URI malformed");
                        return Err(ExecError::Throw(self.make_error(N_URI_ERROR, Some(m))));
                    }
                }
            }
            // `escape(string)` (Annex B.2.1): the legacy escaper. Each UTF-16
            // code unit that is not in the unescaped set (`A-Za-z0-9@*_+-./`) is
            // emitted as `%XX` (units < 256) or `%uXXXX` (units >= 256). The
            // argument is ToString-coerced first (its error propagates).
            N_ESCAPE => {
                let bytes = self.coerce_to_string_bytes(arg(0))?;
                self.new_str_bytes(legacy_escape(&bytes))
            }
            // `unescape(string)` (Annex B.2.2): the inverse of `escape`. Reads
            // `%uXXXX` (4 hex digits) and `%XX` (2 hex digits) escapes; any
            // malformed escape is left verbatim.
            N_UNESCAPE => {
                let bytes = self.coerce_to_string_bytes(arg(0))?;
                self.new_str_bytes(legacy_unescape(&bytes))
            }
            N_STRUCTURED_CLONE => {
                let mut seen: Vec<(u64, NanBox)> = Vec::new();
                self.structured_clone(arg(0), &mut seen)?
            }
            // `$262_detachArrayBuffer(buffer)` — the Test262 `$262.detachArrayBuffer`
            // host hook. Detaches the buffer (empties its store + every view, flags it
            // detached) and returns `null` per the abstract `DetachArrayBuffer`.
            // `%ThrowTypeError%`: the poisoned accessor for a strict `arguments`
            // object's `callee` and `Function.prototype.caller`/`.arguments`
            // (get or set) — a TypeError for strict/bound functions and the
            // strict `arguments` object. A legacy *non-strict, non-bound* function
            // keeps the implementation-defined `.caller`/`.arguments` reads benign
            // (Test262 `caller` feature): reading them yields `null` rather than
            // throwing, matching mainstream engines.
            N_THROW_TYPE_ERROR => {
                if let Some(h) = self.this_val.as_handle().map(Handle::from_raw)
                    && self.realm.get_property(h, BOUND_TARGET).is_none()
                    && self.realm.get_property(h, DYN_FN_MARKER).is_none()
                    && let Some((func_id, _)) = self.realm.function_at(h)
                {
                    let def = &self.functions[func_id as usize];
                    // Only an *ordinary*, source-declared non-strict function keeps
                    // the legacy benign read (returns `null`); strict, generator,
                    // async, arrow, bound, and dynamically-built functions are
                    // restricted and their `.caller`/`.arguments` always throw a
                    // TypeError (an arrow has no own `caller`/`arguments` and
                    // inherits the poisoned accessor).
                    if !def.is_strict && !def.is_generator && !def.is_async && !def.is_arrow {
                        return Ok(NanBox::null());
                    }
                }
                return Err(self.type_error(
                    "'caller', 'callee', and 'arguments' may not be accessed on strict mode \
                     functions or the arguments objects for calls to them",
                ));
            }
            // `Function.prototype[Symbol.hasInstance](V)`: OrdinaryHasInstance for
            // `this` (the function). A non-callable `this` reports `false`.
            N_FN_HAS_INSTANCE => {
                let this = self.this_val;
                let v = arg(0);
                return Ok(NanBox::boolean(self.ordinary_has_instance(this, v)?));
            }
            N_DETACH_ARRAY_BUFFER => {
                let buf = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.get_property(*h, ARRAY_BUFFER_BYTES).is_some());
                let Some(buf) = buf else {
                    return Err(self
                        .type_error("$262.detachArrayBuffer called on a non-ArrayBuffer object"));
                };
                self.detach_array_buffer(buf);
                NanBox::null()
            }
            // `$262_IsHTMLDDA()` — the Test262 `$262.IsHTMLDDA` host hook. Returns
            // a fresh [[IsHTMLDDA]] exotic object: an ordinary extensible object
            // (with `Object.prototype` as its `[[Prototype]]`) branded with the
            // hidden `HTMLDDA_SLOT`. The brand drives the exotic `typeof`
            // (`"undefined"`), `ToBoolean` (`false`), loose-equality-with-nullish
            // (`true`), and `[[Call]]` (returns `null`) behaviours.
            N_HTMLDDA => {
                let obj = self.realm.new_object();
                self.realm.set_hidden_property(
                    obj,
                    crate::realm::HTMLDDA_SLOT,
                    NanBox::boolean(true),
                );
                NanBox::handle(obj.to_raw())
            }
            // `$262_createRealm()` — the Test262 `$262.createRealm` host hook
            // (also `realm.createRealm()`). Builds a second realm with distinct
            // intrinsics and returns its `$262`-shaped realm object.
            N_262_CREATE_REALM => return self.create_realm(),
            // `realm.evalScript(src)` — evaluate `src` in the receiver realm.
            N_262_EVAL_SCRIPT => return self.eval_script_in_realm(arg(0)),
            N_262_EVAL_SCRIPT_MAIN => return self.eval_script_current_realm(arg(0)),
            // Test262 `$262.agent` cooperative scheduler (see the `agent` module).
            N_262_AGENT_START => return self.agent_start(arg(0)),
            N_262_AGENT_BROADCAST => return self.agent_broadcast(arg(0)),
            N_262_AGENT_GET_REPORT => return Ok(self.agent_get_report()),
            N_262_AGENT_GET_REPORT_ASYNC => return self.agent_get_report_async(),
            N_262_AGENT_REPORT => return self.agent_report(arg(0)),
            N_262_AGENT_RECEIVE_BROADCAST => {
                return self.agent_receive_broadcast(arg(0));
            }
            // `sleep(ms)` — advance the virtual clock by `ms` (models the agent
            // sleeping). This is only reachable through `$262.agent`, so it cannot
            // affect ordinary timer/Promise code. `leaving()`/`tryYield` are handled
            // in JS (the worker prelude / atomicsHelper), so need no native.
            N_262_AGENT_SLEEP => {
                let ms = self.realm.to_number(arg(0));
                if ms.is_finite() && ms > 0.0 {
                    self.virtual_now += ms;
                }
                NanBox::undefined()
            }
            N_262_AGENT_MONOTONIC_NOW => NanBox::number(self.agent_monotonic_now()),
            // `DisposableStack()` / `AsyncDisposableStack()` / `ShadowRealm()`
            // called without `new` is a TypeError (they require a `[[Construct]]`).
            N_DISPOSABLE_STACK | N_ASYNC_DISPOSABLE_STACK | N_SHADOW_REALM => {
                return Err(self.type_error("constructor requires 'new'"));
            }
            // `SuppressedError(...)` without `new` builds the same instance
            // (newTarget = the constructor itself → `SuppressedError.prototype`).
            N_SUPPRESSED_ERROR => {
                let callee = self
                    .current
                    .get("SuppressedError")
                    .unwrap_or(NanBox::undefined());
                return self.construct_suppressed_error(args, callee, callee);
            }
            // `get DisposableStack.prototype.disposed` /
            // `get AsyncDisposableStack.prototype.disposed`: a brand-checked boolean.
            N_DISPOSABLE_STACK_DISPOSED | N_ASYNC_DISPOSABLE_STACK_DISPOSED => {
                return self.dstack_disposed_getter(id == N_ASYNC_DISPOSABLE_STACK_DISPOSED);
            }
            // `Intl.getCanonicalLocales(locales)` — canonical locale-tag list.
            N_INTL_GET_CANONICAL_LOCALES => return self.intl_get_canonical_locales(arg(0)),
            // `Intl.supportedValuesOf(key)` — supported identifiers for a key.
            N_INTL_SUPPORTED_VALUES_OF => return self.intl_supported_values_of(arg(0)),
            // `Intl.NumberFormat(...)` / `Intl.DateTimeFormat(...)` called without
            // `new` build the same formatter object.
            N_INTL_NUMBER_FORMAT | N_INTL_DATETIME_FORMAT => {
                return self.make_intl_formatter(id, args);
            }
            // `Intl.Collator(...)` without `new` builds the same collator object.
            N_INTL_COLLATOR => self.make_collator(args)?,
            // `Intl.PluralRules(...)` without `new` is a TypeError (ECMA-402
            // sec-intl.pluralrules: "If NewTarget is undefined, throw a TypeError").
            N_INTL_PLURAL_RULES => {
                return Err(self.type_error("Constructor Intl.PluralRules requires 'new'"));
            }
            // `Intl.ListFormat(...)` without `new` is a TypeError (ECMA-402
            // sec-intl.listformat: "If NewTarget is undefined, throw a TypeError").
            N_INTL_LIST_FORMAT => {
                return Err(self.type_error("Constructor Intl.ListFormat requires 'new'"));
            }
            // `Intl.ListFormat.prototype.format(list)` — `StringListFromIterable`
            // (every element must be a String), then joined per locale/type/style.
            N_INTL_LIST_FORMAT_FORMAT => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let (list_type, style) = self.list_format_type_style(fmt);
                let items = self.string_list_from_iterable(arg(0))?;
                // The `intl` crate supplies locale-aware conjunction/disjunction
                // connectors (e.g. Spanish "y"/"o"); `unit` (and the non-intl
                // build) use the hardcoded en list patterns.
                #[cfg(feature = "intl")]
                {
                    // The crate is locale-aware only for conjunction/disjunction
                    // in the *long* style; short/narrow and `unit` route through
                    // the (English) CLDR pattern table below so `format` and
                    // `formatToParts` agree.
                    let crate_style = match (list_type.as_str(), style.as_str()) {
                        ("disjunction", "long") => Some(intl::list::ListStyle::Or),
                        ("conjunction", "long") => Some(intl::list::ListStyle::And),
                        _ => None,
                    };
                    if let Some(crate_style) = crate_style {
                        let locale = fmt
                            .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                            .map(|v| self.realm.to_display_string(v))
                            .unwrap_or_else(|| String::from("en"));
                        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
                        return Ok(self.new_str(&intl::list::format_list(
                            &locale,
                            &refs,
                            crate_style,
                        )));
                    }
                }
                let parts = self.list_format_parts(&items, &list_type, &style);
                let out: String = parts.into_iter().map(|(_, v)| v).collect();
                self.new_str(&out)
            }
            // `Intl.RelativeTimeFormat(...)` without `new` — a TypeError (the
            // constructor requires `new`, ECMA-402 sec-intl.relativetimeformat).
            N_INTL_REL_TIME => {
                return Err(self.type_error("Constructor Intl.RelativeTimeFormat requires 'new'"));
            }
            // `Intl.RelativeTimeFormat.prototype.format(value, unit)`.
            N_INTL_REL_TIME_FORMAT => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let (numeric, style) = self.rel_time_numeric_style(fmt);
                // PartitionRelativeTimePattern: ToNumber(value) (a Symbol throws a
                // TypeError), then SingularRelativeTimeUnit(unit) which ToString's
                // `unit` (Symbol → TypeError) and validates it (else RangeError);
                // a non-finite value is a RangeError.
                let nv = self.coerce_to_number(arg(0))?;
                let value = self.realm.to_number(nv);
                let unit = self.singular_relative_time_unit(arg(1))?;
                if !value.is_finite() {
                    let m = self.new_str("value must be finite");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                let parts = rel_time_parts(value, &unit, &numeric, &style);
                let out: String = parts.into_iter().map(|(_, v, _)| v).collect();
                // PartitionRelativeTimePattern formats the magnitude through the
                // instance's [[NumberFormat]], so the resolved numbering system's
                // digits apply (e.g. an `-u-nu-arab` locale renders Arabic-Indic).
                let out = match fmt {
                    Some(h) => self.apply_numbering_digits(h, out),
                    None => out,
                };
                self.new_str(&out)
            }
            // `Intl.DisplayNames(...)` without `new`.
            N_INTL_DISPLAY_NAMES => self.make_display_names(args)?,
            // `Intl.DisplayNames.prototype.of(code)`.
            N_INTL_DISPLAY_NAMES_OF => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let ty = fmt
                    .and_then(|h| self.realm.get_property(h, "type"))
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                // `code = ? ToString(code)`, then validate it against the instance
                // `[[Type]]` grammar — a mismatch is a RangeError. The original
                // (non-canonicalized) code is kept for the name lookup / `code`
                // fallback (the canonical form is not observable via `of`).
                let code = self.coerce_to_string(arg(0))?;
                if crate::nbexec::intl_fmt::validate_display_code(&ty, &code).is_none() {
                    let m = self.new_str(&alloc::format!(
                        "invalid {ty} code for Intl.DisplayNames.of"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
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
                        // A crate-backed type (language/region) with no match honors
                        // `fallback`: "none" → undefined, "code" → the code itself.
                        None if matches!(ty.as_str(), "language" | "region") => {
                            let fallback = fmt
                                .and_then(|h| self.realm.get_property(h, "fallback"))
                                .map(|v| self.realm.to_display_string(v))
                                .unwrap_or_else(|| String::from("code"));
                            if fallback == "none" {
                                NanBox::undefined()
                            } else {
                                self.new_str(&code)
                            }
                        }
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
            // `Intl.Segmenter` / `Intl.Locale` / `Intl.DurationFormat` require
            // `new` — a plain call (no `new`) is a TypeError (ECMA-402
            // sec-intl.segmenter / -.locale / -.durationformat: "If NewTarget is
            // undefined, throw a TypeError exception.").
            N_INTL_SEGMENTER => {
                return Err(self.type_error("Constructor Intl.Segmenter requires 'new'"));
            }
            N_INTL_LOCALE => {
                return Err(self.type_error("Constructor Intl.Locale requires 'new'"));
            }
            N_INTL_DURATION_FORMAT => {
                return Err(self.type_error("Constructor Intl.DurationFormat requires 'new'"));
            }
            // `Intl.Segmenter.prototype.segment(input)` → an (iterable) array of segment
            // data objects `{ segment, index, input, isWordLike? }`.
            N_INTL_SEGMENTER_SEGMENT => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                let gran = fmt
                    .and_then(|h| self.realm.get_property(h, "granularity"))
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_else(|| String::from("grapheme"));
                // `segment(string)` does `? ToString(string)` (a Symbol throws a
                // TypeError; a missing argument yields "undefined").
                let input = self.coerce_to_string(arg(0))?;
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
                let arr = self.realm.new_array(elems);
                // Brand the Segments object (spec `[[SegmentsSegmenter]]`) so
                // `%Segments.prototype%.containing` can `RequireInternalSlot` its
                // `this` value and reject foreign receivers with a TypeError.
                self.realm
                    .set_hidden_property(arr, "\u{0}segments", NanBox::boolean(true));
                // Attach `containing(index)` (a bound native over this segments
                // array) so `segments.containing(i)` works, alongside iteration.
                let containing = self.realm.new_bound_native(N_INTL_SEGMENTS_CONTAINING, arr);
                self.install_fn_name_length(containing, "containing", 1);
                self.realm
                    .set_property(arr, "containing", NanBox::handle(containing.to_raw()));
                NanBox::handle(arr.to_raw())
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
                    let fmt = self.this_val.as_handle().map(Handle::from_raw);
                    self.collator_ordering(fmt, &a, &b)
                };
                #[cfg(not(feature = "intl"))]
                let ord = a.cmp(&b);
                NanBox::number(match ord {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Equal => 0.0,
                    core::cmp::Ordering::Greater => 1.0,
                })
            }
            // `Intl.PluralRules.prototype.select(value)` — ToNumber(value), then the
            // locale+type plural category of the resulting number.
            N_INTL_PLURAL_SELECT => {
                let nv = self.coerce_to_number(arg(0))?;
                let n = self.realm.to_number(nv);
                let cat = self.plural_select_category(n);
                self.new_str(cat)
            }
            // `Intl.PluralRules.prototype.selectRange(start, end)`: ToNumber both
            // (Symbol → TypeError); `undefined` start/end → TypeError; a NaN value →
            // RangeError; else the range plural category. With only the English
            // cardinal/ordinal rules implemented, the range category is "other".
            N_INTL_PLURAL_SELECT_RANGE => {
                let start = arg(0);
                let end = arg(1);
                if matches!(start.unpack(), Unpacked::Undefined)
                    || matches!(end.unpack(), Unpacked::Undefined)
                {
                    return Err(self.type_error("selectRange requires both start and end"));
                }
                let sv = self.coerce_to_number(start)?;
                let ev = self.coerce_to_number(end)?;
                let s = self.realm.to_number(sv);
                let e = self.realm.to_number(ev);
                if s.is_nan() || e.is_nan() {
                    let m = self.new_str("selectRange argument is NaN");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                // ResolvePluralRange: the plural category for the range. Deferred to
                // the end-value category (English ranges resolve to "other").
                let cat = self.plural_select_category(e);
                self.new_str(cat)
            }
            // `nf.format(x)` read as a value then called: format against the `this`
            // formatter (a detached call with no formatter falls back to ToString).
            N_INTL_FORMAT => {
                if let Some(h) = self.this_val.as_handle().map(Handle::from_raw)
                    && self.realm.get_property(h, "\u{0}intl").is_some()
                {
                    // A DateTimeFormat coerces its argument via ToNumber + TimeClip
                    // (a non-finite/out-of-range date is a RangeError); a
                    // NumberFormat formats any numeric value (incl. ±Infinity).
                    let s = self.intl_format_checked(h, arg(0))?;
                    self.new_str(&s)
                } else {
                    let s = self.realm.to_display_string(arg(0));
                    self.new_str(&s)
                }
            }
            // `nf.formatRange(x, y)` / `dtf.formatRange(x, y)` — a formatted numeric
            // (or date) range. Both endpoints are required and must be finite.
            N_INTL_FORMAT_RANGE => {
                let this = self.this_val;
                let s = self.intl_format_range(this, arg(0), arg(1))?;
                self.new_str(&s)
            }
            N_INTL_FORMAT_RANGE_TO_PARTS => {
                let this = self.this_val;
                return self.intl_format_range_to_parts(this, arg(0), arg(1));
            }
            // `nf.resolvedOptions()` — the resolved configuration of the formatter.
            N_INTL_RESOLVED_OPTIONS => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                self.intl_resolved_options(fmt)
            }
            // `Intl.X.supportedLocalesOf(locales [, options])` — the requested
            // locales this engine can serve. `CanonicalizeLocaleList(locales)`
            // (a malformed tag is a RangeError); then `GetOptionsObject(options)`
            // and `GetOption(options, "localeMatcher", …)` (RangeError on an
            // invalid value). With no real CLDR available-locale data, every
            // canonical request is reported as supported.
            N_INTL_SUPPORTED_LOCALES => {
                let requested = self.canonicalize_locale_list(arg(0))?;
                // `options = ? CoerceOptionsToObject(options)`: `undefined` → no
                // options; `null` (→ ToObject) is a TypeError; a primitive is
                // boxed. `localeMatcher` is then read + validated (RangeError on an
                // invalid value) even though the result does not depend on it.
                let opts_arg = arg(1);
                if matches!(opts_arg.unpack(), Unpacked::Null) {
                    return Err(self.type_error("supportedLocalesOf options must not be null"));
                }
                if !matches!(opts_arg.unpack(), Unpacked::Undefined) {
                    let opts = self
                        .coerce_to_object(opts_arg)
                        .as_handle()
                        .map(Handle::from_raw);
                    let _ = self.get_string_option(
                        opts,
                        "localeMatcher",
                        &["lookup", "best fit"],
                        Some("best fit"),
                    )?;
                }
                // LookupSupportedLocales: keep only locales this engine can
                // actually serve. It serves every structurally valid locale
                // *except* the "no linguistic content" subtag `zxx` (and the
                // undetermined `und`), which carry no data and must be dropped.
                let mut out = Vec::with_capacity(requested.len());
                for tag in requested {
                    let primary = tag.split(['-', '_']).next().unwrap_or("");
                    if primary.eq_ignore_ascii_case("zxx") || primary.eq_ignore_ascii_case("und") {
                        continue;
                    }
                    let v = self.new_str(&tag);
                    out.push(v);
                }
                NanBox::handle(self.realm.new_array(out).to_raw())
            }
            // `nf.formatToParts(x)` — the formatted number split into `{type, value}`
            // parts (minusSign/currency/integer/group/decimal/fraction/percent, plus
            // nan/infinity). en-US-ish; mirrors the `format` output's structure.
            N_INTL_FORMAT_TO_PARTS => {
                let fmt = self.this_val.as_handle().map(Handle::from_raw);
                // `Intl.ListFormat.prototype.formatToParts(list)` → an array of
                // `{ type: "element" | "literal", value }` parts.
                if let Some(h) = fmt
                    && self
                        .realm
                        .get_property(h, "\u{0}intl")
                        .map(|v| self.realm.to_display_string(v))
                        .as_deref()
                        == Some("list")
                {
                    let (list_type, style) = self.list_format_type_style(Some(h));
                    let items = self.string_list_from_iterable(arg(0))?;
                    let parts = self.list_format_parts(&items, &list_type, &style);
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
                // `Intl.RelativeTimeFormat.prototype.formatToParts(value, unit)` → an
                // array of `{ type, value, unit? }` parts: the numeric substring is
                // split into integer/group/decimal/fraction parts (each carrying the
                // singular `unit`), surrounded by `{ type: "literal" }` text.
                if let Some(h) = fmt
                    && self
                        .realm
                        .get_property(h, "\u{0}intl")
                        .map(|v| self.realm.to_display_string(v))
                        .as_deref()
                        == Some("rtf")
                {
                    let (numeric, style) = self.rel_time_numeric_style(Some(h));
                    let nv = self.coerce_to_number(arg(0))?;
                    let value = self.realm.to_number(nv);
                    let unit = self.singular_relative_time_unit(arg(1))?;
                    if !value.is_finite() {
                        let m = self.new_str("value must be finite");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    let parts = rel_time_parts(value, &unit, &numeric, &style);
                    let mut arr_elems = Vec::with_capacity(parts.len());
                    for (ty, val, with_unit) in parts {
                        let o = self.realm.new_object();
                        let tv = self.new_str(ty);
                        self.realm.set_property(o, "type", tv);
                        // Substitute the resolved numbering system's digits into the
                        // numeric parts (integer/group/decimal/fraction), matching the
                        // instance's [[NumberFormat]] output.
                        let val = self.apply_numbering_digits(h, val);
                        let vv = self.new_str(&val);
                        self.realm.set_property(o, "value", vv);
                        if with_unit {
                            let uv = self.new_str(&unit);
                            self.realm.set_property(o, "unit", uv);
                        }
                        arr_elems.push(NanBox::handle(o.to_raw()));
                    }
                    return Ok(NanBox::handle(self.realm.new_array(arr_elems).to_raw()));
                }
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
                    #[cfg(feature = "intl")]
                    let parts = match self.temporal_format_parts(h, arg(0), false)? {
                        Some(p) => p,
                        None => {
                            let ms = self.datetime_operand(arg(0))?;
                            self.datetime_parts(h, ms)
                        }
                    };
                    #[cfg(not(feature = "intl"))]
                    let parts = {
                        let ms = self.datetime_operand(arg(0))?;
                        self.datetime_parts(h, ms)
                    };
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
                // Number formatters (`intl`-crate CLDR parts, or the hand-rolled
                // sign/integer-group/decimal/fraction split) via the shared helper.
                let entries: Vec<(&'static str, String)> = match fmt {
                    Some(h) if self.realm.get_property(h, "\u{0}intl").is_some() => {
                        // ToIntlMathematicalValue: coerce (ToPrimitive/ToNumber, BigInt
                        // via its value) before splitting into parts.
                        let n = self.coerce_intl_number(arg(0))?;
                        self.number_handle_parts(h, NanBox::number(n))
                    }
                    _ => {
                        // A plain (non-Intl) receiver: classify the coerced display string.
                        let formatted = self.realm.to_display_string(arg(0));
                        let mut entries: Vec<(&'static str, String)> = Vec::new();
                        let mut s = formatted.as_str();
                        if let Some(rest) = s.strip_prefix('-') {
                            entries.push(("minusSign", String::from("-")));
                            s = rest;
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
                        entries
                    }
                };
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
                let id = self.schedule_timer(delay, callback, extra);
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
                    thenable: None,
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
                    // Every other `this` is ToObject'd (a primitive number/boolean/
                    // string/symbol/bigint boxes to its wrapper) and then reports its
                    // builtinTag — but a string `Symbol.toStringTag` on the (boxed)
                    // object's prototype chain OVERRIDES it, so e.g. setting
                    // `Boolean.prototype[@@toStringTag]` changes `toString.call(true)`.
                    _ => {
                        let recv = self.coerce_to_object(this);
                        match recv.as_handle().map(Handle::from_raw) {
                            Some(h) => alloc::format!("[object {}]", self.object_string_tag(h)?),
                            None => String::from("[object Object]"),
                        }
                    }
                };
                self.new_str(&s)
            }
            // `Object.prototype.toLocaleString()` → `Invoke(this, "toString")`: box a
            // primitive `this` to resolve the (possibly overridden) `toString`, then
            // call it with the *original* `this` value (so a strict override reading
            // `typeof this` sees the primitive, not the wrapper).
            N_OBJ_PROTO_TOLOCALESTRING => {
                // `Return ? Invoke(O, "toString")`. Invoke → GetV(O, "toString")
                // boxes a primitive for the method lookup but a null/undefined
                // receiver throws a TypeError; the resolved `toString` is then
                // called with the *original* this value.
                let this = self.this_val;
                if matches!(this.unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error(
                        "Object.prototype.toLocaleString called on null or undefined",
                    ));
                }
                let recv = self.coerce_to_object(this);
                let Some(h) = recv.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error(
                        "Object.prototype.toLocaleString called on null or undefined",
                    ));
                };
                // `GetV(O, "toString")` resolves the method through the boxed
                // wrapper's prototype chain but with Receiver = the *original* `this`
                // — so an overriding `toString` accessor getter runs against the
                // primitive (a strict getter reading `typeof this` sees "boolean",
                // not "object").
                let m = self.get_with_receiver(h, "toString", this)?;
                self.call_with_this(m, this, args)?
            }
            // `Object.prototype.valueOf` → `Return ? ToObject(this value)`: a
            // null/undefined receiver throws a TypeError, and a primitive is boxed
            // (so `valueOf.call(true)` is a Boolean *object*, `typeof` `"object"`).
            N_OBJ_PROTO_VALUEOF => {
                let this = self.this_val;
                let h =
                    self.require_object_coercible_to_object(this, "Object.prototype.valueOf")?;
                NanBox::handle(h.to_raw())
            }
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
                // ToPropertyKey(V) then ToObject(this) — the key coercion runs
                // *first* (a throwing `toString`/`@@toPrimitive` propagates before
                // the receiver's ToObject), then ToObject throws for a null/undefined
                // receiver and boxes a primitive.
                let key = self.coerce_property_key(arg(0))?;
                let this = self.this_val;
                let h = self.require_object_coercible_to_object(this, "hasOwnProperty")?;
                // A proxy has no ordinary own-property table: `[[GetOwnProperty]]`
                // must route through its `getOwnPropertyDescriptor` trap (or forward
                // to the target). HasOwnProperty is `desc is not undefined`.
                // A module namespace's `[[GetOwnProperty]]` of a TDZ export throws.
                #[cfg(all(feature = "module", feature = "std"))]
                self.namespace_binding_tdz(h, &key)?;
                if self.realm.proxy_at(h).is_some() {
                    let d = self.descriptor_of(h, &key)?;
                    NanBox::boolean(!matches!(d.unpack(), Unpacked::Undefined))
                } else {
                    NanBox::boolean(self.realm.has_own(h, &key))
                }
            }
            N_OBJ_PROTO_PROPISENUM => {
                // ToPropertyKey(V) runs before ToObject(this) (spec step order).
                let key = self.coerce_property_key(arg(0))?;
                let this = self.this_val;
                let h = self.require_object_coercible_to_object(this, "propertyIsEnumerable")?;
                // A module namespace's `[[GetOwnProperty]]` of a TDZ export throws.
                #[cfg(all(feature = "module", feature = "std"))]
                self.namespace_binding_tdz(h, &key)?;
                // A proxy routes `[[GetOwnProperty]]` through its trap; the property
                // is enumerable iff the returned descriptor exists and is enumerable.
                if self.realm.proxy_at(h).is_some() {
                    let d = self.descriptor_of(h, &key)?;
                    let enumerable = d
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|dh| self.realm.get_property(dh, "enumerable"))
                        .is_some_and(|v| self.realm.truthy(v));
                    NanBox::boolean(enumerable)
                } else {
                    // An own *and* enumerable property. `property_is_enumerable` works
                    // for inline objects *and* aux-backed cells (arrays/functions/
                    // classes), where `object_keys` returns `None` and would wrongly
                    // report every aux property non-enumerable.
                    let enumerable =
                        self.realm.has_own(h, &key) && self.realm.property_is_enumerable(h, &key);
                    NanBox::boolean(enumerable)
                }
            }
            N_OBJ_PROTO_ISPROTOTYPEOF => {
                // Spec order: if V is not an object, return false; *then* ToObject
                // (this) (which throws for a null/undefined receiver). So
                // `isPrototypeOf.call(null, 5)` is false, but
                // `isPrototypeOf.call(null, {})` throws.
                let found = if let Some(v) = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|_| self.is_object_value(arg(0)))
                {
                    let this = self.this_val;
                    let target = self.require_object_coercible_to_object(this, "isPrototypeOf")?;
                    let mut cur = self.realm.object_proto(v);
                    let mut f = false;
                    while let Some(p) = cur {
                        if p == target {
                            f = true;
                            break;
                        }
                        cur = self.realm.object_proto(p);
                    }
                    f
                } else {
                    false
                };
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
            N_IS_NAN => {
                // `isNaN(x)` = ToNumber(x) is NaN; ToNumber runs a `valueOf`/
                // `toString` (whose abrupt completion propagates), not the raw
                // non-coercing `to_number` that returns NaN for objects.
                let num = self.coerce_to_number(arg(0))?;
                NanBox::boolean(self.realm.to_number(num).is_nan())
            }
            N_IS_FINITE => {
                let num = self.coerce_to_number(arg(0))?;
                NanBox::boolean(self.realm.to_number(num).is_finite())
            }
            // `Error(msg)` / `new Error(msg, { cause })` (the ES2022 cause option).
            id if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) => {
                // `Error(message)`: if message is not undefined it is `ToString`'d
                // (running a user `toString`/`valueOf`, propagating an abrupt one,
                // and throwing a TypeError for a Symbol) before it becomes the
                // non-enumerable `message` property — unlike the raw
                // `to_display_string` `make_error` falls back to.
                let msg = match args.first().copied() {
                    Some(m) if !matches!(m.unpack(), Unpacked::Undefined) => {
                        let s = self.coerce_to_string(m)?;
                        Some(self.new_str(&s))
                    }
                    _ => None,
                };
                let err = self.make_error(id, msg);
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

/// `RegExp.escape` core (ES2025 `EncodeForRegExpEscape`), over WTF-8 bytes.
/// Iterates code points: a leading ASCII digit/letter is `\xHH`; syntax
/// characters and `/` are backslash-escaped; control escapes map to
/// `\t\n\v\f\r`; whitespace / line terminators / "other punctuators" become
/// `\xHH` (≤ 0xFF) or `\uHHHH` per UTF-16 code unit; everything else is copied
/// verbatim.
fn regexp_escape_wtf8(bytes: &[u8]) -> alloc::vec::Vec<u8> {
    // JS WhiteSpace / LineTerminator code points NOT already covered by a
    // control escape (TAB/LF/VT/FF/CR), which need a hex escape.
    fn is_ws_or_lt(c: u32) -> bool {
        matches!(
            c,
            0x20 | 0xA0 | 0x1680 | 0x2000
                ..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000 | 0xFEFF
        )
    }
    const SYNTAX: &[u8] = b"^$\\.*+?()[]{}|";
    const OTHER_PUNCT: &[u8] = b",-=<>#&!%:;@~'`\"";
    let mut out: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    let mut first = true;
    for c in crate::wtf8::code_points(bytes) {
        let leading_alnum = first && matches!(c, 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A);
        if leading_alnum {
            out.extend_from_slice(alloc::format!("\\x{c:02x}").as_bytes());
        } else if (c < 0x80 && SYNTAX.contains(&(c as u8))) || c == 0x2F {
            out.push(b'\\');
            crate::wtf8::encode_code_point(c, &mut out);
        } else if let Some(ce) = match c {
            0x09 => Some(b't'),
            0x0A => Some(b'n'),
            0x0B => Some(b'v'),
            0x0C => Some(b'f'),
            0x0D => Some(b'r'),
            _ => None,
        } {
            out.push(b'\\');
            out.push(ce);
        } else if (c < 0x80 && OTHER_PUNCT.contains(&(c as u8))) || is_ws_or_lt(c) {
            // Hex-escape per UTF-16 code unit.
            let mut units = [0u16; 2];
            for &u in char::from_u32(c)
                .map(|ch| ch.encode_utf16(&mut units) as &[u16])
                .unwrap_or(&[c as u16])
            {
                if u <= 0xFF {
                    out.extend_from_slice(alloc::format!("\\x{u:02x}").as_bytes());
                } else {
                    out.extend_from_slice(alloc::format!("\\u{u:04x}").as_bytes());
                }
            }
        } else {
            crate::wtf8::encode_code_point(c, &mut out);
        }
        first = false;
    }
    out
}

/// Two-sum error-free transform. Prerequisite: `x.abs() >= y.abs()`. Returns
/// the rounded sum `hi` and the exact roundoff `lo`, so `x + y == hi + lo`
/// mathematically.
#[inline]
fn twosum(x: f64, y: f64) -> (f64, f64) {
    let hi = x + y;
    let lo = y - (hi - x);
    (hi, lo)
}

/// Error-free (exact) summation accumulator for `Math.sumPrecise`.
///
/// A direct port of the TC39 `Math.sumPrecise` reference polyfill, which adapts
/// Shewchuk's algorithm (as implemented in CPython's `math.fsum`) to handle
/// intermediate overflow via a separate biased `overflow` partial conceptually
/// scaled by 2**1024. Maintains a list of non-overlapping `f64` partials whose
/// exact mathematical sum equals the sum of all values fed in, then resolves the
/// correctly-rounded (round-half-to-even) result in a single final reduction.
///
/// Only *finite, nonzero* values are ever passed to [`Self::add`] — the
/// Infinity/NaN/±0 cases are handled by the spec state machine in the caller.
struct ExactSum {
    partials: alloc::vec::Vec<f64>,
    /// Conceptually 2**1024 times this value (the biased high-order partial).
    overflow: f64,
    /// Set if `overflow` ever exceeds 2**53 in magnitude (spec: RangeError).
    overflowed: bool,
}

const TWO_1023: f64 = 8.98846567431158e307; // exactly 2**1023 (bits 0x7fe0_0000_0000_0000)
const MAX_DOUBLE: f64 = f64::MAX; // 2**1024 - 2**(1023-52), significand all 1s
const PENULTIMATE_DOUBLE: f64 = 1.797_693_134_862_315_5e308; // 2**1024 - 2*2**(1023-52)
const MAX_ULP: f64 = MAX_DOUBLE - PENULTIMATE_DOUBLE; // 2**(1023-52)

impl ExactSum {
    fn new() -> Self {
        ExactSum {
            partials: alloc::vec::Vec::new(),
            overflow: 0.0,
            overflowed: false,
        }
    }

    /// Fold one finite, nonzero value into the running expansion, keeping every
    /// nonzero rounding remainder so the stored partials remain non-overlapping
    /// and their exact sum is preserved. Mirrors the polyfill's main-loop body,
    /// including the overflow-to-biased-partial rescaling.
    fn add(&mut self, mut x: f64) {
        let mut i = 0;
        for j in 0..self.partials.len() {
            let mut y = self.partials[j];
            if x.abs() < y.abs() {
                core::mem::swap(&mut x, &mut y);
            }
            let (mut hi, mut lo) = twosum(x, y);
            if hi.is_infinite() {
                let sign = if hi == f64::INFINITY { 1.0 } else { -1.0 };
                self.overflow += sign;
                if self.overflow.abs() >= 9_007_199_254_740_992.0 {
                    self.overflowed = true;
                }
                x = (x - sign * TWO_1023) - sign * TWO_1023;
                if x.abs() < y.abs() {
                    core::mem::swap(&mut x, &mut y);
                }
                let r = twosum(x, y);
                hi = r.0;
                lo = r.1;
            }
            if lo != 0.0 {
                self.partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        self.partials.truncate(i);
        if x != 0.0 {
            self.partials.push(x);
        }
    }

    /// Round the exact expansion to the nearest `f64` (ties to even). Direct port
    /// of the polyfill's final reduction: the biased-overflow handling, the
    /// MAX_DOUBLE rounding edge case, and the half-ULP tie correction.
    fn finish(&self) -> f64 {
        // Work on a local mutable copy so we can inject a partial in the overflow
        // path exactly as the polyfill does (`partials[n + 1] = lo`).
        let mut partials = self.partials.clone();
        let mut n: isize = partials.len() as isize - 1;
        let mut hi = 0.0_f64;
        let mut lo = 0.0_f64;

        if self.overflow != 0.0 {
            let next = if n >= 0 { partials[n as usize] } else { 0.0 };
            n -= 1;
            if self.overflow.abs() > 1.0
                || (self.overflow > 0.0 && next > 0.0)
                || (self.overflow < 0.0 && next < 0.0)
            {
                return if self.overflow > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
            }
            // |overflow| == 1: drop a factor of 2 to avoid overflowing.
            let r = twosum(self.overflow * TWO_1023, next / 2.0);
            hi = r.0;
            lo = r.1 * 2.0;
            if (2.0 * hi).is_infinite() {
                // Rounding right at the maximum representable value.
                if hi > 0.0 {
                    if hi == TWO_1023
                        && lo == -(MAX_ULP / 2.0)
                        && n >= 0
                        && partials[n as usize] < 0.0
                    {
                        return MAX_DOUBLE;
                    }
                    return f64::INFINITY;
                }
                if hi == -TWO_1023 && lo == MAX_ULP / 2.0 && n >= 0 && partials[n as usize] > 0.0 {
                    return -MAX_DOUBLE;
                }
                return f64::NEG_INFINITY;
            }
            if lo != 0.0 {
                // Re-insert `lo` as the (n+1)-th partial; the next loop consumes it.
                let slot = (n + 1) as usize;
                if slot < partials.len() {
                    partials[slot] = lo;
                } else {
                    partials.push(lo);
                }
                n += 1;
                lo = 0.0;
            }
            hi *= 2.0;
        }

        while n >= 0 {
            let x = hi;
            let y = partials[n as usize];
            n -= 1;
            let r = twosum(x, y);
            hi = r.0;
            lo = r.1;
            if lo != 0.0 {
                break;
            }
        }

        // Half-ULP tie correction (the polyfill's "handle rounding" tail).
        if n >= 0
            && ((lo < 0.0 && partials[n as usize] < 0.0)
                || (lo > 0.0 && partials[n as usize] > 0.0))
        {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
        hi
    }
}
