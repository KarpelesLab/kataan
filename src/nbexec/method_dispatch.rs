use super::*;

/// The ECMAScript `TrimString` white-space set: every Unicode `White_Space`
/// code point **plus** U+FEFF (ZERO WIDTH NO-BREAK SPACE / BOM), which the spec
/// includes but Rust's `char::is_whitespace` (current Unicode) does not.
fn is_js_trim_ws(c: char) -> bool {
    c.is_whitespace() || c == '\u{FEFF}'
}

impl<'a> Interp<'a> {
    /// `Function.prototype.toString` source-representation for the callable
    /// `handle`: `class <Name> { }` for a class, else NativeFunction syntax
    /// (`function <name>() { [native code] }` — the engine retains no source).
    /// The `name` segment is emitted only when it is a valid IdentifierName
    /// (optionally with a `get `/`set ` prefix); a `#private`, symbol-bracketed,
    /// or punctuated name is dropped rather than produce an unparsable string.
    pub(crate) fn function_to_string_repr(
        &mut self,
        handle: Handle,
    ) -> Result<alloc::string::String, ExecError> {
        // A function/method/arrow/class defined from source reproduces its exact
        // literal text (ECMA-262 20.2.3.5). A Proxy has no `[[SourceText]]`, so it
        // renders the NativeFunction form regardless of what it wraps — never the
        // wrapped source.
        if self.realm.proxy_at(handle).is_none()
            && let Some(src) = self.realm.fn_source(handle)
        {
            return Ok(src.into());
        }
        let nm = self.read_member(handle, "name")?;
        let nm = self.realm.to_display_string(nm);
        // A class with no retained source (e.g. a native/subclassed intrinsic)
        // still uses the `class …` form; a Proxy or ordinary function uses the
        // NativeFunction form.
        Ok(if self.realm.class_at(handle).is_some() {
            alloc::format!("class {nm} {{ }}")
        } else {
            let seg = crate::realm::native_fn_name_segment(&nm);
            alloc::format!("function {seg}() {{ [native code] }}")
        })
    }

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

        // Temporal statics: `Temporal.<Type>.from(...)` / `.compare(...)` — a
        // Temporal instance is not a native, so this only matches a constructor
        // receiver (mirrors how `Date.now` is recognised). Instance methods route
        // through the bound-native `N_TEMPORAL_PROTO_FN` handler in `call.rs`.
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && let Some(cid) = self.realm.native_at(h)
            && let Some(kind) = crate::nbexec::temporal::kind_for_ctor_id(cid)
            && let Some(v) = self.temporal_static(kind, recv, method, args)?
        {
            return Ok(Some(v));
        }

        // ValidateTypedArray: the data-accessing `%TypedArray%.prototype` methods
        // throw a TypeError up front if the backing buffer is detached (the view was
        // length-0'd on detach, so without this they would silently operate on an
        // empty array). `subarray` (builds a fresh view) is exempt. `toString` is
        // *not* exempt: it is `Array.prototype.toString`, which delegates to
        // `%TypedArray%.prototype.join`, so ValidateTypedArray applies to it too.
        //
        // This validation is specific to the *branded* `%TypedArray%.prototype`
        // entry points; a **generic** `Array.prototype.<m>.call(ta)` (the
        // `array_proto_generic` one-shot) runs the ordinary array-like algorithm,
        // which reads the view's (0-for-out-of-bounds) length instead of throwing.
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && !self.array_proto_generic
            && self.realm.typed_kind(h).is_some()
            && !matches!(method, "subarray" | "constructor")
            && TYPED_ARRAY_PROTO_METHODS.iter().any(|(n, _)| *n == method)
            && self.typed_array_detached(h)
        {
            return Err(self.type_error(&alloc::format!(
                "TypedArray.prototype.{method} called on a detached ArrayBuffer"
            )));
        }
        // ValidateTypedArray also rejects a view that is *out of bounds* (a
        // fixed-length view whose resizable buffer shrank below its declared
        // extent) with a TypeError, up front — same exempt set as the detached
        // guard above (and likewise skipped for the generic `Array.prototype` path).
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && !self.array_proto_generic
            && self.realm.typed_kind(h).is_some()
            && !matches!(method, "subarray" | "constructor")
            && TYPED_ARRAY_PROTO_METHODS.iter().any(|(n, _)| *n == method)
            && self.realm.typed_array_out_of_bounds(h)
        {
            return Err(self.type_error(&alloc::format!(
                "TypedArray.prototype.{method} called on an out-of-bounds typed array"
            )));
        }

        // When reached through a *generic* `Array.prototype.<m>` call, a
        // primitive-wrapper `this` must be handled as an array-like object (not
        // unwrapped) — consume the one-shot flag so it applies only to this call.
        let array_proto_generic = core::mem::take(&mut self.array_proto_generic);

        // `Array.prototype.toString` (23.1.3.36) is receiver-agnostic: it does
        // `O = ToObject(this)`, `func = Get(O, "join")`, and calls `func` if it is
        // callable, else falls back to `%Object.prototype.toString%` (yielding e.g.
        // `"[object Boolean]"` for a boxed primitive or `"[object Object]"` for a
        // plain array-like whose chain has no `join`). Intercept the *genuine*
        // `Array.prototype.toString` (the `array_proto_generic` flag) up front so a
        // boxed-primitive / non-array receiver is handled per spec rather than
        // unwrapped. A real array's `join` resolves to the callable
        // `Array.prototype.join`, so this preserves ordinary `[1,2].toString()`.
        if method == "toString"
            && array_proto_generic
            && let Some(h) = recv.as_handle().map(Handle::from_raw)
        {
            let join = self.read_member(h, "join")?;
            if self.is_callable_value(join) {
                return self.call_with_this(join, recv, args).map(Some);
            }
            let tag = self.object_string_tag(h)?;
            return Ok(Some(self.new_str(&alloc::format!("[object {tag}]"))));
        }
        // `Array.prototype.toReversed` (23.1.3.33): `len = LengthOfArrayLike(O)`,
        // then read `from = len-1 … 0` via `[[Get]]` — so index accessors fire in
        // **descending** order — building a fresh dense array. Intercept up front
        // (a real array, a `.call` on an array-like, or an inheriting object) so the
        // access order and length re-read are spec-exact; typed arrays keep their
        // own branded `toReversed` (a same-kind view) below.
        if method == "toReversed"
            && let Some(h) = recv.as_handle().map(Handle::from_raw)
            && self.realm.typed_kind(h).is_none()
            && (self.realm.is_array(h) || array_proto_generic || self.inherits_array_proto(h))
        {
            let len = self.array_like_length(h)?;
            // `ArrayCreate(len)` (the result) throws a RangeError when `len` exceeds
            // the array-length limit — *before* any source element is read.
            if len > self.realm.limits.max_array_len {
                let m = self.new_str("Invalid array length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            let mut out = Vec::with_capacity(len.min(1 << 16));
            for k in 0..len {
                let from = len - k - 1;
                out.push(self.read_member(h, &alloc::format!("{from}"))?);
            }
            return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
        }
        // `Array.prototype.toSpliced(start, skipCount, ...items)` (23.1.3.35):
        // builds a fresh dense array = the receiver with `actualSkipCount` elements
        // removed at `actualStart` and `items` inserted. `newLen > 2^53-1` is a
        // TypeError (checked before `ArrayCreate`, which itself RangeErrors above
        // `2^32-1`); elements are `[[Get]]` in the exact spec order (`0..start`,
        // then the tail `r..`), so a skipped index is never read.
        if method == "toSpliced"
            && let Some(h) = recv.as_handle().map(Handle::from_raw)
            && self.realm.typed_kind(h).is_none()
            && (self.realm.is_array(h) || array_proto_generic || self.inherits_array_proto(h))
        {
            let len = self.array_like_length(h)?;
            let rel_start = self.coerce_to_integer_or_infinity(arg(0))?;
            let actual_start = if rel_start == f64::NEG_INFINITY {
                0
            } else if rel_start < 0.0 {
                ((len as f64) + rel_start).max(0.0) as usize
            } else {
                (rel_start as usize).min(len)
            };
            let (insert_count, skip_count) = if args.is_empty() {
                (0usize, 0usize)
            } else if args.len() == 1 {
                (0usize, len - actual_start)
            } else {
                let dc = self.coerce_to_integer_or_infinity(arg(1))?;
                let dc = dc.max(0.0).min((len - actual_start) as f64) as usize;
                (args.len() - 2, dc)
            };
            let new_len = len + insert_count - skip_count;
            // Step 12: `newLen > 2^53-1` → TypeError (before ArrayCreate).
            if new_len as f64 > 9_007_199_254_740_991.0 {
                return Err(self.type_error("Array.prototype.toSpliced result exceeds 2**53-1"));
            }
            // ArrayCreate(newLen) → RangeError above the array-length limit.
            if new_len > self.realm.limits.max_array_len {
                let m = self.new_str("Invalid array length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            let mut out = Vec::with_capacity(new_len.min(1 << 16));
            for k in 0..actual_start {
                out.push(self.read_member(h, &alloc::format!("{k}"))?);
            }
            for item in args.iter().skip(2) {
                out.push(*item);
            }
            let mut r = actual_start + skip_count;
            while out.len() < new_len {
                out.push(self.read_member(h, &alloc::format!("{r}"))?);
                r += 1;
            }
            return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
        }
        // `Array.prototype.slice` on a *generic* array-like (a plain object / proxy
        // reached via `.call`, not a real Array): read only the `[k, final)` window
        // lazily via `HasProperty`/`[[Get]]` — so a huge `length` (e.g. 2^53+2)
        // slices a small range without materializing every index (which would
        // RangeError), and only the sliced indices' getters fire. A non-Array source
        // means `ArraySpeciesCreate` yields a plain dense array.
        if method == "slice"
            && let Some(h) = recv.as_handle().map(Handle::from_raw)
            && self.realm.is_generic_array_like_target(h)
            && (array_proto_generic || self.inherits_array_proto(h))
            && !self.inherits_iterator_proto(h)
        {
            let len = self.array_like_length(h)?;
            let rel_start = self.coerce_to_integer_or_infinity(arg(0))?;
            let k = if rel_start == f64::NEG_INFINITY {
                0
            } else if rel_start < 0.0 {
                ((len as f64) + rel_start).max(0.0) as usize
            } else {
                (rel_start as usize).min(len)
            };
            let rel_end = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                len as f64
            } else {
                self.coerce_to_integer_or_infinity(arg(1))?
            };
            let final_i = if rel_end == f64::NEG_INFINITY {
                0
            } else if rel_end < 0.0 {
                ((len as f64) + rel_end).max(0.0) as usize
            } else {
                (rel_end as usize).min(len)
            };
            let count = final_i.saturating_sub(k);
            if count > self.realm.limits.max_array_len {
                let m = self.new_str("Invalid array length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            // `A = ArraySpeciesCreate(O, count)` — for a *proxy* whose target is an
            // Array, `IsArray(O)` is true, so this reads `O.constructor`/`@@species`
            // (through the proxy's traps) and may build an exotic result rather than a
            // plain dense Array. A genuinely non-Array array-like yields a plain Array.
            let a_v = self.array_species_create(h, count)?;
            let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
                return Err(self.type_error("Array species did not return an object"));
            };
            let default_array = self.realm.is_array(a_h)
                && self.realm.array_length(a_h) == Some(count)
                && !self.realm.array_has_index_overrides(a_h);
            for (n, i) in (k..final_i).enumerate() {
                let key = alloc::format!("{i}");
                if self.has_property(h, &key) {
                    let v = self.read_member(h, &key)?;
                    if default_array {
                        self.realm.set_element(a_h, n, v);
                    } else {
                        self.create_data_property_or_throw(a_h, n, v)?;
                    }
                }
            }
            let len_key = self.new_str("length");
            self.assign_member_value(a_h, len_key, NanBox::number(count as f64))?;
            return Ok(Some(a_v));
        }
        // `Array.prototype.concat` is receiver-agnostic (ECMA-262 23.1.3.1: it
        // begins with `ToObject(this)`), so intercept it up front for *any*
        // receiver — a real array, a non-array array-like, or a boxed primitive —
        // before the primitive early-returns and the real-array element block
        // below. Gate exactly like the other generic `Array.prototype` methods: a
        // receiver only runs this when the call is an explicit
        // `Array.prototype.concat.call(o)` (the flag) or `o` actually inherits the
        // array methods through its prototype chain (so a plain object with no
        // `concat` in its chain still reports "concat is not a function").
        if method == "concat"
            && (array_proto_generic
                || recv
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.inherits_array_proto(h)))
        {
            return Ok(Some(self.array_concat(recv, args)?));
        }
        // `Array.prototype.<m>.call(strOrStrWrapper, …)`: the receiver must be
        // read as an array-like (each UTF-16 unit an element), so the String
        // primitive/wrapper method handlers below must NOT intercept this method.
        let force_array_like = array_proto_generic && ARRAY_LIKE_METHODS.contains(&method);
        // `Array.prototype.{push,pop,shift,unshift}.call(str)`: ToObject(str) is a
        // String exotic object whose `length` (and character indices) are
        // non-writable / non-configurable, so the operation's trailing
        // `Set(O, "length", …, true)` (and any element `Set`/`Delete`) always fails
        // → a TypeError. A String *primitive* is not otherwise treated as a generic
        // array-like target (its UTF-16 units are invisible to `HasProperty`), so
        // intercept these length-mutating forms here.
        if matches!(method, "push" | "pop" | "shift" | "unshift")
            && recv.as_handle().map(Handle::from_raw).is_some_and(|h| {
                // A String primitive (`Cell::Str`) or a String wrapper (an object
                // boxing a string under `PRIM_WRAP` — a primitive `this` is boxed via
                // ToObject before the method dispatch runs). Strings never own these
                // methods, so reaching here with a string receiver is necessarily a
                // generic `Array.prototype.<m>.call(str)` application.
                self.realm.string_value(h).is_some()
                    || self
                        .realm
                        .get_property(h, PRIM_WRAP)
                        .and_then(|p| p.as_handle())
                        .map(Handle::from_raw)
                        .is_some_and(|ph| self.realm.string_value(ph).is_some())
            })
        {
            return Err(self.type_error(&alloc::format!(
                "Array.prototype.{method} cannot set the non-writable length of a String"
            )));
        }

        // A primitive wrapper object (`new Number`/`String`/`Boolean`): `valueOf`
        // recovers the boxed primitive; every other method delegates to it. Skip
        // this when a generic `Array.prototype.<m>`/`Function.prototype.<m>` is
        // being applied to the wrapper (the boxed `this`): the array-like methods
        // read the wrapper itself, and `call`/`apply`/`bind` must observe the
        // wrapper as a non-callable `this` and throw rather than unwrap.
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && let Some(prim) = self.realm.get_property(h, PRIM_WRAP)
            && !(array_proto_generic
                && (ARRAY_LIKE_METHODS.contains(&method)
                    || MUTATING_GENERIC.contains(&method)
                    || matches!(method, "call" | "apply" | "bind")))
            // The generic `Object.prototype` methods operate on the wrapper *object*
            // itself (its prototype chain / own properties), so they must NOT unwrap
            // to the boxed primitive — e.g. `String.prototype.isPrototypeOf(x)` (a
            // String exotic whose `[[StringData]]` is `""`) must test the object, not
            // the empty string.
            && !matches!(
                method,
                "hasOwnProperty" | "isPrototypeOf" | "propertyIsEnumerable"
            )
        {
            // `toString`/`toLocaleString` only unwrap while the name still resolves
            // to the wrapper-proto built-in (e.g. `String.prototype.toString`). If it
            // was deleted/shadowed, the name resolves to `Object.prototype.toString`,
            // which must run on the wrapper *object* (yielding `"[object String]"`
            // via its `[[StringData]]` builtin tag) — so defer to real resolution.
            // A `new String(...)` receiver delegates `match`/`replace`/`replaceAll`/
            // `search`/`split` to a searchValue object's `@@method` with `O` = the
            // wrapper **object** (its `this` value), so `O` keeps its identity —
            // must run *before* unwrapping to the boxed primitive, which would
            // otherwise pass the primitive string as `O`.
            if prim
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|ph| self.realm.string_value(ph).is_some())
                && matches!(
                    method,
                    "match" | "matchAll" | "search" | "replace" | "replaceAll" | "split"
                )
                && let Some(r) = self.string_symbol_delegate(recv, method, args)?
            {
                return Ok(Some(r));
            }
            if matches!(method, "toString" | "toLocaleString") {
                let resolved = self.read_member(h, method)?;
                let is_wrapper_proto_builtin = resolved
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|fh| self.realm.native_at(fh))
                    .is_some_and(|nid| {
                        matches!(
                            nid,
                            N_STRING_PROTO_FN | N_NUMBER_PROTO_FN | N_BOOLEAN_PROTO_FN
                        )
                    });
                if !is_wrapper_proto_builtin {
                    return Ok(None);
                }
            }
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
                // `toLocaleString(locales, options)` — build a real NumberFormat and
                // format through it so *every* option (notation/unit/significant
                // digits/signDisplay/…) applies, not just the minimal grouping path.
                "toLocaleString" => {
                    #[cfg(feature = "intl")]
                    let s = {
                        let fmt_args = [
                            args.first().copied().unwrap_or(NanBox::undefined()),
                            args.get(1).copied().unwrap_or(NanBox::undefined()),
                        ];
                        let inst = self.make_intl_formatter(N_INTL_NUMBER_FORMAT, &fmt_args)?;
                        match inst.as_handle().map(Handle::from_raw) {
                            Some(h) => self.intl_format_number(h, n),
                            None => self.number_to_locale_string(n, args.get(1).copied()),
                        }
                    };
                    #[cfg(not(feature = "intl"))]
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

        // `WeakRef.prototype.deref` / `FinalizationRegistry.prototype.{register,
        // unregister}` are real, brand-checking natives on their prototypes (set up
        // in `setup`), reached through ordinary member lookup — no fast-path here.

        // --- universal `Object.prototype` methods (own/inherited reflection) ---
        match method {
            "hasOwnProperty" => {
                // ToPropertyKey(V) runs a user `toString`/`valueOf`/`@@toPrimitive`
                // (which may return a Symbol); the resulting key resolves symbol-keyed
                // properties via its internal slot name.
                let key = self.coerce_property_key(arg(0))?;
                return Ok(Some(NanBox::boolean(self.realm.has_own(handle, &key))));
            }
            "isPrototypeOf" => {
                // Only an object argument has a prototype chain to walk; a primitive
                // (or missing) argument is `false`. Walk via `[[GetPrototypeOf]]`
                // (proxy-aware — a proxy fires its `getPrototypeOf` trap).
                if !self.is_object_value(arg(0)) {
                    return Ok(Some(NanBox::boolean(false)));
                }
                let Some(v) = arg(0).as_handle().map(Handle::from_raw) else {
                    return Ok(Some(NanBox::boolean(false)));
                };
                let mut cur = self.get_proto_of(v)?;
                for _ in 0..1_000_000 {
                    let Some(p) = cur.as_handle().map(Handle::from_raw) else {
                        return Ok(Some(NanBox::boolean(false)));
                    };
                    if p == handle {
                        return Ok(Some(NanBox::boolean(true)));
                    }
                    cur = self.get_proto_of(p)?;
                }
                return Ok(Some(NanBox::boolean(false)));
            }
            "propertyIsEnumerable" => {
                // True only for an *own* *enumerable* property (a non-enumerable one,
                // or an inherited one, is false). ToPropertyKey resolves symbol keys
                // and runs a user `toString`/`valueOf`/`@@toPrimitive`.
                let key = self.coerce_property_key(arg(0))?;
                let r = self.realm.has_own(handle, &key)
                    && self.realm.property_is_enumerable(handle, &key);
                return Ok(Some(NanBox::boolean(r)));
            }
            // The legacy (Annex B) accessor helpers `__defineGetter__` /
            // `__defineSetter__` / `__lookupGetter__` / `__lookupSetter__` are NOT
            // shortcut here: their spec semantics (callable check before key
            // coercion, DefinePropertyOrThrow honoring extensibility /
            // configurability, the lookup callable-half filter) live in the native
            // `N_OBJ_*` handlers, so they fall through to the ordinary
            // member-lookup + native-call path.

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
                            // An array-like: ToLength(Get(O, "length")) fires a
                            // getter (whose abrupt completion propagates) and is
                            // coerced through `valueOf`/`toString`, then each index
                            // is read via Get.
                            let len_val = self.read_member(h, "length")?;
                            let len_num = self.coerce_to_number(len_val)?;
                            let raw = self.realm.to_number(len_num);
                            let len = if raw.is_nan() || raw <= 0.0 {
                                0
                            } else {
                                raw.min(9_007_199_254_740_991.0) as usize
                            };
                            let mut v = Vec::with_capacity(len.min(1 << 16));
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
                    return Ok(Some(self.make_bound_function(recv, this, bound)?));
                }
                // A textual representation (the engine does not retain source).
                "toString" | "toLocaleString" => {
                    let s = self.function_to_string_repr(handle)?;
                    return Ok(Some(self.new_str(&s)));
                }
                _ => {}
            }
        } else if array_proto_generic && matches!(method, "call" | "apply" | "bind" | "toString") {
            // The *genuine* `Function.prototype.{call,apply,bind,toString}` (reached
            // through the first-class bound-native dispatch, which set
            // `array_proto_generic`) requires an `IsCallable` `this`: a non-callable
            // receiver (`bind.call(5)`, `toString.call(new Proxy({},{}))`,
            // `obj.bind = Function.prototype.bind; obj.bind()`) is a TypeError.
            // We gate on the flag so a *user* method named call/apply/bind/toString
            // inherited on a non-callable object (`new M().call()`,
            // `({}).toString()`) still resolves through the normal property lookup.
            return Err(self.type_error(&alloc::format!(
                "Function.prototype.{method} called on non-callable receiver"
            )));
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
                // ES2025 iterator helpers — drive the receiver generator lazily
                // through the shared `iterator_proto_helper` (which reads the
                // generator's real `next`/`return` methods on demand). This keeps
                // map/filter/take/drop/flatMap lazy and the consuming helpers
                // spec-faithful (calling order, closing, infinite iterators).
                "map" | "filter" | "take" | "drop" | "toArray" | "forEach" | "reduce" | "some"
                | "every" | "find" | "flatMap" => {
                    let this = NanBox::handle(handle.to_raw());
                    return Ok(Some(self.iterator_proto_helper(method, this, args)?));
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
            // `Int8Array.of/from` share the same spec-faithful path as
            // `%TypedArray%.of/from` (the constructor `handle` is the `this` value);
            // routing both through one helper keeps the two engine tiers in step and
            // gets the `GetMethod(@@iterator)` / `TypedArrayCreate` ordering right.
            let ctor = NanBox::handle(handle.to_raw());
            let result = if method == "of" {
                self.typed_array_of(ctor, args)?
            } else {
                self.typed_array_from(
                    ctor,
                    arg(0),
                    args.get(1).copied().unwrap_or(NanBox::undefined()),
                    args.get(2).copied().unwrap_or(NanBox::undefined()),
                )?
            };
            return Ok(Some(result));
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
            // MakeDate(MakeDay(y, m, d), MakeTime(h, min, s, milli)) in IEEE-754
            // `f64` (the two-digit-year mapping is folded into the helper).
            let ms = crate::nbexec::make_date_ms(year_n, month, day, hours, mins, secs, millis);
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
            // The revocation function is an anonymous (`name === ""`) length-0
            // function.
            self.install_fn_name_length(revoke, "", 0);
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
                    // `Symbol.for(key)` does `ToString(key)` — a user `toString`/
                    // `valueOf` runs (propagating) and a Symbol argument is a
                    // TypeError, unlike the raw `to_display_string`.
                    let key = self.coerce_to_string(arg(0))?;
                    if let Some(s) = self.symbol_registry.get(&key) {
                        return Ok(Some(*s));
                    }
                    let sym = NanBox::handle(self.realm.new_symbol(&key).to_raw());
                    self.symbol_registry.insert(key, sym);
                    return Ok(Some(sym));
                }
                "keyFor" => {
                    let target = arg(0);
                    // `Symbol.keyFor(sym)`: if `sym` is not a Symbol, throw a
                    // TypeError (ECMA-262 20.4.2.6 step 1).
                    if !target
                        .as_handle()
                        .map(Handle::from_raw)
                        .is_some_and(|h| self.realm.symbol_at(h).is_some())
                    {
                        return Err(self.type_error("Symbol.keyFor requires a symbol argument"));
                    }
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
                // `toLocaleString(locales, options)` — build a real NumberFormat and
                // format the BigInt *value* through it, so every option
                // (notation/style/significant digits/signDisplay/…) applies and the
                // result is byte-identical to `Intl.NumberFormat.prototype.format`.
                // Passing the BigInt NanBox (not an f64) also routes it through the
                // exact-decimal path, preserving precision beyond 2^53.
                "toLocaleString" => {
                    #[cfg(feature = "intl")]
                    {
                        let fmt_args = [
                            args.first().copied().unwrap_or(NanBox::undefined()),
                            args.get(1).copied().unwrap_or(NanBox::undefined()),
                        ];
                        let inst = self.make_intl_formatter(N_INTL_NUMBER_FORMAT, &fmt_args)?;
                        let value = NanBox::handle(handle.to_raw());
                        let s = match inst.as_handle().map(Handle::from_raw) {
                            Some(h) => self.intl_format_checked(h, value)?,
                            None => group_thousands_str(&bigint_to_radix(&big, 10)),
                        };
                        return Ok(Some(self.new_str(&s)));
                    }
                    #[cfg(not(feature = "intl"))]
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
                // `cooked = ToObject(template)` — throwing ToObject: a `null` /
                // `undefined` template is a TypeError (not `Object()`'s fresh
                // object). A primitive boxes to its wrapper.
                if matches!(arg(0).unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                }
                let cooked = self.coerce_to_object(arg(0));
                let Some(ch) = cooked.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                };
                // `raw = ToObject(Get(cooked, "raw"))`: `Get` fires an inherited
                // getter (a throw propagates), and a `null`/`undefined` `raw` is a
                // TypeError.
                let raw_v = self.read_member(ch, "raw")?;
                if matches!(raw_v.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                }
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
        if (method == "slice" || method == "sliceToImmutable")
            && let Some(bh) = self.array_buffer_bytes(handle)
        {
            self.guard_detached_buffer(handle)?;
            // `len` is the byteLength captured *before* coercing the relative
            // indices (a `valueOf` may resize/detach a resizable buffer).
            let len = self.realm.bytes_len(bh).unwrap_or(0) as i64;
            // ToIntegerOrInfinity(start)/(end) run user code (their `valueOf`),
            // resolved against `len`; `end` defaults to `len`.
            let rel = |n: f64| -> usize {
                let n = n as i64;
                usize::try_from(if n < 0 { (len + n).max(0) } else { n.min(len) }).unwrap_or(0)
            };
            let start_n = self.coerce_to_integer_or_infinity(arg(0))?;
            let begin = rel(start_n);
            let end = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                len as usize
            } else {
                let end_n = self.coerce_to_integer_or_infinity(arg(1))?;
                rel(end_n)
            };
            // A coercion may have detached the source — re-validate (TypeError).
            self.guard_detached_buffer(handle)?;
            let new_len = end.saturating_sub(begin);
            let cur = self
                .realm
                .bytes_at(bh)
                .map(<[u8]>::to_vec)
                .unwrap_or_default();
            // `sliceToImmutable` requires the resolved range to lie within the
            // *current* byteLength (a resize during coercion that drops below the
            // resolved end is a RangeError); plain `slice` instead clamps the copy
            // and zero-fills any tail.
            let count = if method == "sliceToImmutable" {
                if end > cur.len() {
                    let m = self.new_str(
                        "ArrayBuffer.prototype.sliceToImmutable: range exceeds byteLength",
                    );
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                new_len
            } else {
                new_len.min(cur.len().saturating_sub(begin))
            };
            // The result is always `new_len` bytes; copy at most `count` bytes from
            // the current store, leaving any remainder zero.
            let mut sub = alloc::vec![0u8; new_len];
            sub[..count].copy_from_slice(cur.get(begin..begin + count).unwrap_or(&[]));
            // Plain `slice` allocates the result via `SpeciesConstructor(O,
            // %ArrayBuffer%)` (so a subclass gets a subclass instance);
            // `sliceToImmutable` always yields a fresh immutable `%ArrayBuffer%`.
            let nb = if method == "slice" {
                self.array_buffer_species_new(handle, &sub, new_len)?
            } else {
                self.make_array_buffer_from_bytes(&sub)
            };
            // `sliceToImmutable` yields an immutable buffer.
            if method == "sliceToImmutable" {
                self.realm
                    .set_hidden_property(nb, ARRAY_BUFFER_IMMUTABLE, NanBox::boolean(true));
            }
            // `SharedArrayBuffer.prototype.slice` yields a *SharedArrayBuffer*: carry
            // the brand over and link the copy to `%SharedArrayBuffer.prototype%`.
            if self
                .realm
                .get_property(handle, SHARED_ARRAY_BUFFER_BRAND)
                .is_some()
            {
                self.realm.set_hidden_property(
                    nb,
                    SHARED_ARRAY_BUFFER_BRAND,
                    NanBox::boolean(true),
                );
                if let Some(proto) = self
                    .current
                    .get("SharedArrayBuffer")
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                    .and_then(|c| self.realm.get_property(c, "prototype"))
                    .and_then(|p| p.as_handle())
                    .map(Handle::from_raw)
                {
                    self.realm.set_object_proto(nb, Some(proto));
                }
            }
            return Ok(Some(NanBox::handle(nb.to_raw())));
        }
        // --- ArrayBuffer.prototype.transfer(newLength?) / transferToFixedLength(newLength?)
        // → a new ArrayBuffer, detaching the original (its byteLength becomes 0 and its
        // views are emptied). `transfer` preserves resizability (the new buffer keeps the
        // original's maxByteLength); `transferToFixedLength` always yields a fixed-length
        // buffer. (ArrayBufferCopyAndDetach.) ---
        if (method == "transfer"
            || method == "transferToFixedLength"
            || method == "transferToImmutable")
            && let Some(bh) = self.array_buffer_bytes(handle)
        {
            // `newLength` is ToIndex-coerced first — before the immutable and
            // detached checks — so a poisoned `valueOf` / out-of-range length is
            // observed in spec order (ArrayBufferCopyAndDetach reads newLength
            // before verifying mutability).
            let new_len = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                None
            } else {
                Some(usize::try_from(self.coerce_to_index(arg(0))?).unwrap_or(usize::MAX))
            };
            // An immutable buffer can never be transferred (it has no transferable
            // data).
            self.guard_immutable_buffer(handle)?;
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
            // `transferToImmutable` marks the new (fixed-length) buffer immutable.
            if method == "transferToImmutable" {
                self.realm
                    .set_hidden_property(nb, ARRAY_BUFFER_IMMUTABLE, NanBox::boolean(true));
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
            self.guard_immutable_buffer(handle)?;
            let Some(max) = self
                .realm
                .get_property(handle, ARRAY_BUFFER_MAXLEN)
                .map(|m| self.realm.to_number(m) as usize)
            else {
                let m = self.new_str("ArrayBuffer.prototype.resize: buffer is not resizable");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            };
            // newLength is ToIndex'd: a negative / non-integer-out-of-range / huge
            // value is a RangeError (not silently clamped to 0). The coercion runs
            // *before* the detached check (spec 25.1.6.x steps 3-4), so a `valueOf`
            // that detaches the buffer is still observed and the argument evaluated.
            let new_f = self.coerce_to_integer_or_infinity(arg(0))?;
            if !new_f.is_finite() || !(0.0..9_007_199_254_740_992.0).contains(&new_f) {
                let m = self.new_str("ArrayBuffer.prototype.resize: invalid length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            // Step 4: a single IsDetachedBuffer check *after* argument coercion.
            self.guard_detached_buffer(handle)?;
            let new_len = new_f as usize;
            if new_len > max {
                let m = self.new_str("ArrayBuffer.prototype.resize: length exceeds maxByteLength");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            self.realm.resize_buffer(bh, new_len);
            return Ok(Some(NanBox::undefined()));
        }
        // --- SharedArrayBuffer.prototype.grow(newByteLength) (growable SABs only) ---
        // `grow` is a SharedArrayBuffer-only buffer method (ArrayBuffer uses
        // `resize`), so calling it on a plain ArrayBuffer receiver is a TypeError.
        if method == "grow"
            && self
                .realm
                .get_property(handle, ARRAY_BUFFER_BYTES)
                .is_some()
        {
            if self
                .realm
                .get_property(handle, SHARED_ARRAY_BUFFER_BRAND)
                .is_none()
            {
                let m = self
                    .new_str("SharedArrayBuffer.prototype.grow called on an incompatible receiver");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            let Some(bh) = self.array_buffer_bytes(handle) else {
                return Ok(None);
            };
            let Some(max) = self
                .realm
                .get_property(handle, ARRAY_BUFFER_MAXLEN)
                .map(|m| self.realm.to_number(m) as usize)
            else {
                let m = self.new_str("SharedArrayBuffer.prototype.grow: buffer is not growable");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            };
            let cur = self.realm.bytes_len(bh).unwrap_or(0);
            // `newLength` is ToIndex'd: a Symbol / non-coercible is a TypeError, an
            // object's `valueOf` runs, a non-integer / negative is a RangeError.
            let new_f = self.coerce_to_integer_or_infinity(arg(0))?;
            let new_len = self.validate_alloc_len(new_f, "SharedArrayBuffer.prototype.grow")?;
            // `grow` only ever increases the length (a shrink is a RangeError), up to
            // maxByteLength.
            if new_len < cur || new_len > max {
                let m = self.new_str("SharedArrayBuffer.prototype.grow: invalid new length");
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
            // A `set*` on a view over an immutable buffer is a TypeError verified
            // *before* any argument coercion (the immutable-buffer tests assert no
            // `valueOf` runs).
            if is_set {
                self.guard_view_immutable(handle)?;
            }
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
            // IsViewOutOfBounds (resizable buffer shrank under the view): a
            // fixed-length view whose `base + recorded_len` exceeds the live buffer,
            // or a length-tracking view whose `base` is past the end, is a TypeError
            // (checked before the RangeError bounds check below).
            let recorded_len = self
                .realm
                .get_property(handle, DATA_VIEW_LEN)
                .and_then(|n| n.as_number())
                .map(|n| n as usize);
            let dv_oob = match recorded_len {
                Some(len) => base.checked_add(len).is_none_or(|end| end > total),
                None => base > total,
            };
            if dv_oob {
                return Err(self
                    .type_error("DataView access on an out-of-bounds view (buffer was resized)"));
            }
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
        // `Intl.ListFormat` (kind "list") and `Intl.RelativeTimeFormat` (kind "rtf")
        // carry the same `\0intl` marker but their `format`/`formatToParts` have
        // bespoke signatures (an iterable of strings / a `(value, unit)` pair) — let
        // those fall through to the branded prototype methods below.
        if let Some(kind) = self.realm.get_property(handle, "\u{0}intl")
            && !matches!(self.realm.to_display_string(kind).as_str(), "list" | "rtf")
            && method == "format"
        {
            // DateTimeFormat validates the argument via ToNumber + TimeClip (a
            // non-finite / out-of-range date is a RangeError); NumberFormat formats
            // any numeric value.
            let s = self.intl_format_checked(handle, arg(0))?;
            return Ok(Some(self.new_str(&s)));
        }
        // --- Date instance methods ---
        if let Some(ms) = self.realm.date_at(handle).filter(|_| !force_array_like) {
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
            // `Date.prototype.toTemporalInstant()` — a Temporal.Instant at the
            // Date's epoch (an invalid Date is a RangeError).
            if method == "toTemporalInstant" {
                if !ms.is_finite() {
                    let msg = self.new_str("toTemporalInstant called on an invalid Date");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(msg))));
                }
                let data = crate::temporal_iso::TemporalData {
                    kind: crate::temporal_iso::TemporalKind::Instant,
                    epoch_ns: (ms as i128) * 1_000_000,
                    ..Default::default()
                };
                let ih = self.realm.new_temporal(data);
                if let Some(p) = self.temporal_proto(crate::temporal_iso::TemporalKind::Instant) {
                    self.realm.set_native_proto(ih, p);
                }
                return Ok(Some(NanBox::handle(ih.to_raw())));
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
                        | "getYear"
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
                // Annex B.2.4.1 `getYear`: `YearFromTime(LocalTime(t)) - 1900`
                // (the engine is UTC, so `LocalTime(t) == t`).
                "getYear" => NanBox::number((y - 1900) as f64),
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
                    // The locale-aware variants route through a real DateTimeFormat
                    // (with ToDateTimeOptions per-method defaults), so every option
                    // applies — dateStyle/timeStyle, explicit components, hourCycle, …
                    #[cfg(feature = "intl")]
                    if matches!(
                        method,
                        "toLocaleDateString" | "toLocaleTimeString" | "toLocaleString"
                    ) {
                        let (want_date, want_time) = match method {
                            "toLocaleDateString" => (true, false),
                            "toLocaleTimeString" => (false, true),
                            _ => (true, true),
                        };
                        let opts = self.date_time_options(arg(1), want_date, want_time)?;
                        let fmt_args = [arg(0), NanBox::handle(opts.to_raw())];
                        let inst = self.make_intl_formatter(N_INTL_DATETIME_FORMAT, &fmt_args)?;
                        if let Some(h) = inst.as_handle().map(Handle::from_raw) {
                            let s = self.format_intl_datetime(h, ms);
                            return Ok(Some(self.new_str(&s)));
                        }
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
                // Annex B.2.5.1 `setYear(year)`: like `setFullYear` but maps a
                // two-digit year argument (integer part in `0..=99`) to `1900 + y`.
                // Works on an invalid date (treats the time as +0).
                "setYear" => {
                    let num = self.coerce_to_number(arg(0))?;
                    let yf = self.realm.to_number(num);
                    if yf.is_nan() {
                        self.realm.set_date_ms(handle, f64::NAN);
                        return Ok(Some(NanBox::number(f64::NAN)));
                    }
                    // MakeFullYear: 0..=99 (integer part) becomes 1900-relative.
                    // (`trunc_toward_zero` is the no_std-safe `f64::trunc`.)
                    let yint = trunc_toward_zero(yf);
                    let yy = if (0.0..=99.0).contains(&yint) {
                        1900 + yint as i64
                    } else {
                        yint as i64
                    };
                    // Decompose the current time (or the epoch when invalid).
                    let date_is_nan = !ms.is_finite();
                    let (mo1, dd, hh, mi, ss, mss) = if date_is_nan {
                        (1, 1, 0, 0, 0, 0)
                    } else {
                        (
                            mo,
                            d as i64,
                            tod / 3_600_000,
                            tod / 60_000 % 60,
                            tod / 1000 % 60,
                            tod % 1000,
                        )
                    };
                    let base_days = crate::realm::days_from_civil(yy, mo1, 1) + (dd - 1);
                    let nms = time_clip(
                        (base_days * 86_400_000 + hh * 3_600_000 + mi * 60_000 + ss * 1000 + mss)
                            as f64,
                    );
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
            // GroupBy: IsCallable(callbackfn) is checked *before* iterating the
            // items (so `Map.groupBy([], null)` throws even for an empty list).
            let cb = arg(1);
            self.require_callable(cb, "Map.groupBy callback")?;
            let items = self.iterate_values(arg(0))?;
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
        // `Promise.withResolvers()` — generic over the receiver `C`:
        // `NewPromiseCapability(C)`, then package `{ promise, resolve, reject }`.
        // `C` must be a constructor (the this-aware static guard in `call.rs`
        // already throws a `TypeError` for a non-constructor / non-object receiver
        // of the genuine static). For a subclass this constructs `new C(executor)`,
        // so `withResolvers.call(SubPromise).promise` is a `SubPromise` instance.
        if method == "withResolvers" && self.is_constructor(recv) {
            let cap = self.new_promise_capability(recv)?;
            let obj = self.realm.new_object();
            self.realm.set_property(obj, "promise", cap.promise);
            self.realm.set_property(obj, "resolve", cap.resolve);
            self.realm.set_property(obj, "reject", cap.reject);
            return Ok(Some(NanBox::handle(obj.to_raw())));
        }
        // `Promise.resolve` / `Promise.reject` invoked with a *custom* constructor
        // receiver (`Promise.resolve.call(C)`, a subclass, or a foreign thenable
        // constructor) go through `NewPromiseCapability(C)` per spec — so `C` is
        // constructed with a spec-shaped executor (length 2, name ""). The native
        // `%Promise%` keeps its fast path below.
        if matches!(method, "resolve" | "reject")
            && self.realm.native_at(handle) != Some(N_PROMISE)
            && self.is_constructor(recv)
        {
            // PromiseResolve is idempotent: a promise whose `.constructor` is `C`
            // is returned unchanged.
            if method == "resolve"
                && let Some(raw) = arg(0).as_handle()
                && self.realm.promise_state(Handle::from_raw(raw)).is_some()
            {
                let ctor = self.read_member(Handle::from_raw(raw), "constructor")?;
                if ctor.as_handle() == recv.as_handle() {
                    return Ok(Some(arg(0)));
                }
            }
            let cap = self.new_promise_capability(recv)?;
            if method == "resolve" {
                self.call(cap.resolve, &[arg(0)])?;
            } else {
                self.call(cap.reject, &[arg(0)])?;
            }
            return Ok(Some(cap.promise));
        }
        // --- `Promise.resolve` / `Promise.reject` statics (on the constructor) ---
        if self.realm.native_at(handle) == Some(N_PROMISE) {
            match method {
                "resolve" => {
                    // `PromiseResolve(%Promise%, x)` is idempotent on a promise only
                    // when `SameValue(x.constructor, %Promise%)`: if `x` is already a
                    // promise *and* its `constructor` is this receiver, return it
                    // unchanged (same identity). A promise whose `constructor` was
                    // reassigned (e.g. to `null`) is instead wrapped in a fresh one.
                    if let Some(raw) = arg(0).as_handle()
                        && self.realm.promise_state(Handle::from_raw(raw)).is_some()
                    {
                        let ctor = self.read_member(Handle::from_raw(raw), "constructor")?;
                        if ctor.as_handle() == recv.as_handle() {
                            return Ok(Some(arg(0)));
                        }
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
                _ => {}
            }
        }
        // `Promise.try(fn, ...args)` — generic over the receiver `C` (`this`), which
        // must be a constructor (`NewPromiseCapability(C)` throws otherwise, so
        // `Promise.try.call(nonCtorObject, …)` is a TypeError; a non-object `this` is
        // already rejected by the `this_aware` static dispatch). Calls `fn(...args)`
        // synchronously: a returned value resolves the capability (adopting a
        // thenable), a throw (incl. a non-callable `fn`) rejects it. Gated to a
        // non-promise receiver so a `p.try` on a thenable named "try" is not hijacked.
        if method == "try" && self.realm.promise_state(handle).is_none() {
            if !self.is_constructor(recv) {
                return Err(self.type_error("Promise.try called on a non-constructor"));
            }
            let callback = arg(0);
            let rest: Vec<NanBox> = if args.len() > 1 {
                args[1..].to_vec()
            } else {
                Vec::new()
            };
            // Fast path for the intrinsic `%Promise%`; a subclass uses its capability.
            if self.realm.native_at(handle) == Some(N_PROMISE) {
                let p = self.fresh_promise();
                match self.call_with_this(callback, NanBox::undefined(), &rest) {
                    Ok(v) => self.resolve_with(p, v),
                    Err(ExecError::Throw(e)) => self.settle(p, e, false),
                    Err(other) => return Err(other),
                }
                return Ok(Some(NanBox::handle(p.to_raw())));
            }
            let cap = self.new_promise_capability(recv)?;
            match self.call_with_this(callback, NanBox::undefined(), &rest) {
                Ok(v) => {
                    self.call_with_this(cap.resolve, NanBox::undefined(), &[v])?;
                }
                Err(ExecError::Throw(e)) => {
                    self.call_with_this(cap.reject, NanBox::undefined(), &[e])?;
                }
                Err(other) => return Err(other),
            }
            return Ok(Some(cap.promise));
        }
        // --- Promise combinators (`all`/`race`/`allSettled`/`any`) ---
        // These are generic over the receiver `C` (`this`): `Promise.all`,
        // `Subclass.all`, and `Promise.all.call(C, …)` all dispatch here. They run
        // the spec algorithm (NewPromiseCapability(C) + Invoke(C,"resolve") +
        // Invoke(p,"then")), so call counts, resolve-function identity, species,
        // and AggregateError all hold. Gated on the receiver being a constructor so
        // an unrelated object's same-named method is not hijacked.
        if matches!(
            method,
            "all" | "race" | "allSettled" | "any" | "allKeyed" | "allSettledKeyed"
        ) && self.is_constructor(recv)
        {
            return match method {
                "all" => Ok(Some(self.perform_promise_all(recv, arg(0))?)),
                "allSettled" => Ok(Some(self.perform_promise_all_settled(recv, arg(0))?)),
                "race" => Ok(Some(self.perform_promise_race(recv, arg(0))?)),
                "any" => Ok(Some(self.perform_promise_any(recv, arg(0))?)),
                "allKeyed" => Ok(Some(self.perform_promise_all_keyed(recv, arg(0), false)?)),
                "allSettledKeyed" => {
                    Ok(Some(self.perform_promise_all_keyed(recv, arg(0), true)?))
                }
                _ => unreachable!(),
            };
        }
        // --- promise instance methods (`then`/`catch`/`finally`) ---
        // `then` is brand-checked + species-aware. `catch`/`finally` are generic:
        // they delegate to `Invoke(this, "then", …)`, so they work on any thenable
        // receiver and surface a poisoned/throwing/non-callable `then`.
        if method == "then" {
            return Ok(Some(self.perform_promise_then_method(
                handle,
                arg(0),
                arg(1),
            )?));
        }
        if method == "catch" {
            // `return Invoke(promise, "then", [undefined, onRejected])`.
            let then = self.read_member(handle, "then")?;
            return Ok(Some(self.call_with_this(
                then,
                recv,
                &[NanBox::undefined(), arg(0)],
            )?));
        }
        if method == "finally" {
            return Ok(Some(self.promise_finally(handle, recv, arg(0))?));
        }

        // A custom matcher/replacer: when the argument defines the matching
        // well-known symbol method (`Symbol.match`/`replace`/`search`/`split`/
        // `matchAll`), `str.method(obj)` delegates to `obj[@@method](str, …rest)`.
        // (A RegExp argument now resolves its `@@method` through `RegExp.prototype`,
        // so this is the spec path for `"…".match(/re/)` etc.) The `@@method`
        // lookup happens ONLY when the argument **is an Object** — a primitive
        // regexp (a string, or a `BigInt`/`Symbol` whose prototype might carry a
        // `@@match`) is never inspected; it is coerced to a string pattern.
        if self.realm.string_value(handle).is_some() {
            // `O` (the `this` value passed to the `@@method`) is the primitive
            // string receiver here; a `String` *wrapper* receiver is intercepted
            // earlier (before unwrapping) so its object identity is preserved.
            let o = NanBox::handle(handle.to_raw());
            if let Some(r) = self.string_symbol_delegate(o, method, args)? {
                return Ok(Some(r));
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
        // (Skipped when a generic `Array.prototype.<m>` is being applied to a
        // String primitive/wrapper, which must run the array-like path instead.)
        if let Some(bytes) = self
            .realm
            .string_bytes(handle)
            .filter(|_| !force_array_like)
        {
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
                "toUpperCase" => Some(self.new_str_bytes(case_map_wtf8(&bytes, true))),
                "toLowerCase" => Some(self.new_str_bytes(case_map_wtf8(&bytes, false))),
                "toLocaleUpperCase" | "toLocaleLowerCase" => {
                    // The `locales` argument is `CanonicalizeLocaleList`-validated
                    // (a malformed tag is a RangeError).
                    let locales = self.canonicalize_locale_list(arg(0))?;
                    let upper = method == "toLocaleUpperCase";
                    // Turkic (`tr`/`az`) and Lithuanian (`lt`) locales tailor the
                    // case mapping (dotted/dotless I, combining-dot handling); other
                    // locales use the default (locale-independent) mapping. Only the
                    // valid-UTF-8 fast path is tailored (lone surrogates are
                    // case-neutral and keep the WTF-8 mapping).
                    #[cfg(feature = "intl")]
                    {
                        let lang = locales.first().map(String::as_str).unwrap_or("");
                        let primary = lang.split('-').next().unwrap_or("");
                        let special = matches!(primary, "tr" | "az" | "lt");
                        if special && let Ok(s) = core::str::from_utf8(&bytes) {
                            let out = if upper {
                                intl::unicode::uppercase_str_lang(s, lang)
                            } else {
                                intl::unicode::lowercase_str_lang(s, lang)
                            };
                            Some(self.new_str(&out))
                        } else {
                            Some(self.new_str_bytes(case_map_wtf8(&bytes, upper)))
                        }
                    }
                    #[cfg(not(feature = "intl"))]
                    {
                        let _ = &locales;
                        Some(self.new_str_bytes(case_map_wtf8(&bytes, upper)))
                    }
                }
                "trim" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    Some(self.new_str(s.trim_matches(is_js_trim_ws)))
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
                    // A RegExp `searchString` is a TypeError (IsRegExp reads
                    // `@@match` via `[[Get]]`, so a throwing getter propagates).
                    if self.try_is_regexp(arg(0))? {
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
                    // ToIntegerOrInfinity(count) (a Symbol / abrupt valueOf throws).
                    // A negative or `+Infinity` count is a `RangeError`; a finite
                    // count whose product with the length overflows would panic, so
                    // it is a `RangeError` too (an unrepresentable string length).
                    let nf = self.coerce_to_integer_or_infinity(arg(0))?;
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
                    if self.try_is_regexp(arg(0))? {
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
                    if self.try_is_regexp(arg(0))? {
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
                    // Spec order: ToUint32(limit) runs *before* ToString(separator),
                    // and its ToNumber may run a user `valueOf`/`@@toPrimitive` (an
                    // abrupt one throws here). Computed inline so the no-`std` build
                    // (no `Realm::to_uint32`) still compiles.
                    let limit = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        u32::MAX
                    } else {
                        let nv = self.coerce_to_number(arg(1))?;
                        let n = self.realm.to_number(nv);
                        if n.is_finite() {
                            (n as i64).rem_euclid(4_294_967_296) as u32
                        } else {
                            0
                        }
                    } as usize;
                    // `R = ToString(separator)` (spec step 7) runs *before* the
                    // `lim === 0` short-circuit (step 8), so a throwing separator
                    // `toString` throws even when `limit` is 0. An `undefined`
                    // separator skips the coercion (its ToString is harmless) and
                    // returns the whole string below.
                    let sep = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        None
                    } else {
                        Some(self.arg_string_bytes_fallible(arg(0))?)
                    };
                    if limit == 0 {
                        return Ok(Some(NanBox::handle(
                            self.realm.new_array(Vec::new()).to_raw(),
                        )));
                    }
                    let Some(sep) = sep else {
                        let whole = self.new_str_bytes(bytes.clone());
                        let arr = self.realm.new_array(alloc::vec![whole]);
                        return Ok(Some(NanBox::handle(arr.to_raw())));
                    };
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
                                // `ToString` the replacer result (custom `toString`/
                                // `@@toPrimitive` runs; a throw propagates).
                                let rs = self.coerce_to_string(r)?;
                                let out =
                                    alloc::format!("{}{}{}", &s[..pos], rs, &s[pos + from.len()..]);
                                Some(self.new_str(&out))
                            }
                            None => Some(self.new_str(&s)),
                        }
                    } else {
                        // A non-callable replaceValue is `ToString`'d (spec step): a
                        // custom `toString`/`@@toPrimitive` runs and a throw
                        // propagates, not "[object Object]".
                        let to = self.coerce_to_string(repl)?;
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
                    if is_fn && from.is_empty() {
                        // An empty search string matches at every UTF-16 boundary
                        // `0..=length` (advanceBy = max(1, 0) = 1). The functional
                        // replacer runs at each: emit `replacer("", pos, string)`,
                        // then the next source char, then the next replacer, … so a
                        // subject "ab" yields `f(0) a f(1) b f(2)` and an empty
                        // subject yields a single `f(0)`.
                        let whole = self.new_str(&s);
                        let mut out = String::new();
                        let mut off_units = 0usize;
                        let empty = self.new_str("");
                        let r = self.call(repl, &[empty, NanBox::number(0.0), whole])?;
                        out.push_str(&self.coerce_to_string(r)?);
                        for ch in s.chars() {
                            out.push(ch);
                            off_units += ch.len_utf16();
                            let empty = self.new_str("");
                            let off = NanBox::number(off_units as f64);
                            let r = self.call(repl, &[empty, off, whole])?;
                            out.push_str(&self.coerce_to_string(r)?);
                        }
                        Some(self.new_str(&out))
                    } else if is_fn {
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
                            // `ToString` the replacer result (a custom `toString`/
                            // `@@toPrimitive` runs, a throw propagates).
                            let rs = self.coerce_to_string(r)?;
                            out.push_str(&rs);
                            units_to_last = off_units + from.encode_utf16().count();
                            last = abs + from.len();
                        }
                        out.push_str(&s[last..]);
                        Some(self.new_str(&out))
                    } else if from.is_empty() {
                        // A non-callable replaceValue is `ToString`'d once (spec
                        // step 6): a custom `toString`/`@@toPrimitive` runs and a
                        // throw propagates, rather than rendering "[object Object]".
                        let to = self.coerce_to_string(repl)?;
                        Some(self.new_str(&s.replace(&from, &to)))
                    } else {
                        let to = self.coerce_to_string(repl)?;
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
                    Some(self.new_str(s.trim_start_matches(is_js_trim_ws)))
                }
                "trimEnd" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    Some(self.new_str(s.trim_end_matches(is_js_trim_ws)))
                }
                // Annex B.2.3: `trimLeft`/`trimRight` are legacy aliases of
                // `trimStart`/`trimEnd`.
                "trimLeft" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    Some(self.new_str(s.trim_start_matches(is_js_trim_ws)))
                }
                "trimRight" => {
                    let s = crate::wtf8::to_string_lossy(&bytes);
                    Some(self.new_str(s.trim_end_matches(is_js_trim_ws)))
                }
                // Annex B.2.3 legacy HTML wrapper methods. `CreateHTML(S, tag,
                // attribute, value)`: wraps the receiver string `S` in
                // `<tag …>S</tag>`, optionally emitting `attribute="value"` with
                // the value's `"` escaped as `&quot;`. The attribute value is
                // ToString-coerced (its error must propagate).
                "anchor" => Some(self.create_html(&bytes, "a", "name", Some(arg(0)))?),
                "big" => Some(self.create_html(&bytes, "big", "", None)?),
                "blink" => Some(self.create_html(&bytes, "blink", "", None)?),
                "bold" => Some(self.create_html(&bytes, "b", "", None)?),
                "fixed" => Some(self.create_html(&bytes, "tt", "", None)?),
                "fontcolor" => Some(self.create_html(&bytes, "font", "color", Some(arg(0)))?),
                "fontsize" => Some(self.create_html(&bytes, "font", "size", Some(arg(0)))?),
                "italics" => Some(self.create_html(&bytes, "i", "", None)?),
                "link" => Some(self.create_html(&bytes, "a", "href", Some(arg(0)))?),
                "small" => Some(self.create_html(&bytes, "small", "", None)?),
                "strike" => Some(self.create_html(&bytes, "strike", "", None)?),
                "sub" => Some(self.create_html(&bytes, "sub", "", None)?),
                "sup" => Some(self.create_html(&bytes, "sup", "", None)?),
                // A string's `toString`/`valueOf` is the string itself.
                "toString" | "valueOf" => Some(recv),
                // `isWellFormed`/`toWellFormed`: a string is well-formed iff it has
                // no lone surrogate. The WTF-8 bytes are valid UTF-8 exactly then.
                "isWellFormed" => Some(NanBox::boolean(crate::wtf8::is_well_formed_utf16(&bytes))),
                "toWellFormed" => {
                    // Scan UTF-16 units: keep valid surrogate pairs (even when they
                    // span WTF-8 leaves), replace only *unpaired* surrogates with
                    // U+FFFD. A plain lossy decode would mangle a pair split across
                    // leaves, so rebuild from the code-unit sequence.
                    let units: Vec<u16> = crate::wtf8::utf16_units(&bytes).collect();
                    let mut out: Vec<u16> = Vec::with_capacity(units.len());
                    let mut i = 0;
                    while i < units.len() {
                        let u = units[i];
                        if (0xD800..=0xDBFF).contains(&u) {
                            if i + 1 < units.len() && (0xDC00..=0xDFFF).contains(&units[i + 1]) {
                                out.push(u);
                                out.push(units[i + 1]);
                                i += 2;
                                continue;
                            }
                            out.push(0xFFFD); // lone high surrogate
                        } else if (0xDC00..=0xDFFF).contains(&u) {
                            out.push(0xFFFD); // lone low surrogate
                        } else {
                            out.push(u);
                        }
                        i += 1;
                    }
                    Some(self.new_str_bytes(crate::wtf8::from_utf16(&out)))
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
                    // ToString(that) — unwraps a String wrapper and runs a user
                    // toString (abrupt-propagating), rather than the raw display form.
                    let other = self.coerce_to_string(arg(0))?;
                    // Per ECMA-402, `localeCompare` initializes an `Intl.Collator`
                    // from (locales, options): this validates both arguments —
                    // throwing the same TypeError/RangeError as `new Intl.Collator`
                    // — and drives real UCA collation (honoring sensitivity /
                    // numeric / ignorePunctuation, and the default lowercase-first
                    // ordering) rather than raw code-point comparison.
                    #[cfg(feature = "intl")]
                    let ord = {
                        let collator = self.make_collator(&[arg(1), arg(2)])?;
                        let ch = collator.as_handle().map(Handle::from_raw);
                        self.collator_ordering(ch, &s, &other)
                    };
                    #[cfg(not(feature = "intl"))]
                    let ord = s.as_str().cmp(other.as_str());
                    let cmp = match ord {
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
            "values",
            "keys",
            "entries",
            "toLocaleString",
            // The ES2023 immutable copies read the array-like by index and return
            // a fresh dense array, so they work on a generic array-like receiver.
            "with",
            "toReversed",
            "toSorted",
            "toSpliced",
        ];
        // The *mutating* `Array.prototype` methods are intentionally generic: on a
        // non-array array-like receiver they read `length`, then `[[Get]]`/`[[Set]]`/
        // `[[Delete]]` indices and finally `[[Set]]` the new `length`. Run them
        // directly against the original object (no materialize-into-temp, which would
        // discard the writes) when reached via `Array.prototype.<m>.call(o)` or an
        // inherited array prototype.
        const MUTATING_GENERIC: &[&str] = &[
            "push",
            "pop",
            "shift",
            "unshift",
            "reverse",
            "fill",
            "copyWithin",
            "splice",
            "sort",
        ];
        if self.realm.is_generic_array_like_target(handle)
            && MUTATING_GENERIC.contains(&method)
            && (array_proto_generic || self.inherits_array_proto(handle))
            && !self.inherits_iterator_proto(handle)
        {
            return Ok(Some(self.array_like_mutate(method, handle, args)?));
        }
        // The pure *scanning* methods (no result array proportional to `length`)
        // run lazily over a generic array-like via `[[Get]]`/`HasProperty` with
        // early exit — so a huge `{length:"Infinity"}` receiver scans without
        // materializing (and without the RangeError the dense path would raise).
        const LAZY_SCAN_GENERIC: &[&str] = &[
            "indexOf",
            "lastIndexOf",
            "includes",
            "some",
            "every",
            "find",
            "findIndex",
            "findLast",
            "findLastIndex",
            "forEach",
            "reduce",
            "reduceRight",
        ];
        // A String primitive/wrapper receiver exposes its UTF-16 units as own
        // data properties that `HasProperty` (used by the lazy scan) cannot see;
        // keep those on the materialization path (which has the `is_string_like`
        // special case).
        let is_string_receiver = self.realm.string_value(handle).is_some()
            || self
                .realm
                .get_property(handle, PRIM_WRAP)
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw)
                .is_some_and(|p| self.realm.string_value(p).is_some());
        if self.realm.is_generic_array_like_target(handle)
            && LAZY_SCAN_GENERIC.contains(&method)
            && (array_proto_generic || self.inherits_array_proto(handle))
            && !self.inherits_iterator_proto(handle)
            && !is_string_receiver
        {
            // The callback-taking forms validate IsCallable(callback) after reading
            // `length` but before any element access (spec order).
            if matches!(
                method,
                "some"
                    | "every"
                    | "find"
                    | "findIndex"
                    | "findLast"
                    | "findLastIndex"
                    | "forEach"
                    | "reduce"
                    | "reduceRight"
            ) {
                // Read `length` first (its getter/coercion side effects happen),
                // then check the callback.
                let _ = self.array_like_length(handle)?;
                self.require_callable(arg(0), &alloc::format!("{method} callback"))?;
            }
            return Ok(Some(self.array_iter_sparse(method, handle, args)?));
        }
        // `flat`/`flatMap` on a generic array-like (a plain object *or* a proxy
        // whose target is an array) run FlattenIntoArray *live*: `LengthOfArrayLike`,
        // then `ArraySpeciesCreate` (reads `constructor`/`@@species`), then per index
        // `HasProperty` → `Get` with recursion into nested arrays. This is the exact
        // spec MOP order/count a proxy observes — the materialize path would instead
        // pre-read every element (wrong access count) and read `constructor` late. A
        // real dense array keeps its fast `elems` path below.
        if self.realm.is_generic_array_like_target(handle)
            && matches!(method, "flat" | "flatMap")
            && (array_proto_generic || self.inherits_array_proto(handle))
            && !self.inherits_iterator_proto(handle)
        {
            let source_len = self.array_like_length(handle)?;
            let (depth, mapper, this_arg) = if method == "flat" {
                let depth = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                    1
                } else {
                    self.coerce_to_integer_or_infinity(arg(0))? as i32
                };
                (depth, None, NanBox::undefined())
            } else {
                self.require_callable(arg(0), "flatMap callback")?;
                (1, Some(arg(0)), arg(1))
            };
            let a_v = self.array_species_create(handle, 0)?;
            let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
                return Err(self.type_error("Array species did not return an object"));
            };
            self.flatten_into_array(a_h, handle, source_len, 0, depth, mapper, this_arg)?;
            return Ok(Some(a_v));
        }
        // `map`/`filter` on a *plain* generic array-like (not a proxy whose target is
        // an array — those still need `@@species` from the materialize path) read each
        // element live per iteration (`HasProperty` then `Get`) so a callback that
        // mutates a later index is observed; the result is built via
        // `ArraySpeciesCreate` → `CreateDataPropertyOrThrow` (an out-of-range `length`
        // throws a RangeError before any callback, matching `ArrayCreate`).
        if self.realm.is_generic_array_like_target(handle)
            && matches!(method, "map" | "filter")
            && (array_proto_generic || self.inherits_array_proto(handle))
            && !self.inherits_iterator_proto(handle)
            && !is_string_receiver
            && !self.is_array_unwrap_proxy(NanBox::handle(handle.to_raw()))?
        {
            return Ok(Some(self.array_map_filter_generic(method, handle, args)?));
        }
        let mut array_like = None;
        if self.realm.is_generic_array_like_target(handle)
            && ARRAY_LIKE_METHODS.contains(&method)
            // Only treat the receiver as a generic array-like when the call was
            // an explicit `Array.prototype.<m>.call(o)` (the flag) OR `o` actually
            // inherits the array methods through its prototype chain. A plain
            // object whose chain has no array method must report "<m> is not a
            // function" via the normal property lookup (return `None` below),
            // not be silently coerced.
            && (array_proto_generic || self.inherits_array_proto(handle))
            // An iterator (a value inheriting `%IteratorPrototype%`, e.g. a lazy
            // iterator-helper) must run its *own* `map`/`filter`/… helper, not be
            // treated as an array-like — so skip the array-like coercion for it.
            && !self.inherits_iterator_proto(handle)
        {
            // `Array.prototype.toSorted`: IsCallable(comparefn) is validated
            // *before* the `length` read (23.1.3.34 step 1), so a bad comparefn
            // throws even when the receiver's `length` getter would throw.
            if method == "toSorted"
                && !matches!(arg(0).unpack(), Unpacked::Undefined)
                && !self.is_callable_value(arg(0))
            {
                return Err(self.type_error("comparefn must be a function"));
            }
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
            // Spec order for the callback-taking methods: IsCallable(callbackfn)
            // is checked *after* reading `length` but *before* any element access
            // (so `reduceRight.call({…length getter…}, undefined)` reads length,
            // then throws, without touching the indices/getters).
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
            // A length beyond the engine's array cap (e.g. `{length: Infinity}`,
            // whose ToLength is 2^53-1) cannot be materialized/allocated — throw a
            // catchable RangeError rather than silently skipping (and never attempt
            // a multi-gigabyte allocation).
            if len_f > self.realm.limits.max_array_len as f64 {
                let m = self.new_str("Invalid array length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            if len_f <= (1u64 << 24) as f64 {
                let len = len_f as usize;
                // A String primitive/wrapper `this` (its boxed string) exposes every
                // in-range index as a present own data property — `HasProperty`
                // wouldn't see them, so treat them all as present.
                let is_string_like = self.realm.string_value(handle).is_some()
                    || self
                        .realm
                        .get_property(handle, PRIM_WRAP)
                        .and_then(|p| p.as_handle())
                        .map(Handle::from_raw)
                        .is_some_and(|p| self.realm.string_value(p).is_some());
                let mut tmp = Vec::with_capacity(len);
                let mut present = Vec::with_capacity(len);
                for i in 0..len {
                    let key = alloc::format!("{i}");
                    // HasProperty(O, idx) — a hole (absent index) is skipped by the
                    // iteration methods; Get(O, idx) walks the prototype chain and
                    // invokes getters only for a present index.
                    let here = is_string_like || self.has_property(handle, &key);
                    present.push(here);
                    tmp.push(if here {
                        self.read_member(handle, &key)?
                    } else {
                        NanBox::undefined()
                    });
                }
                array_like = Some(self.realm.new_array(tmp));
                self.array_like_present = Some(present);
            }
        }
        // A *real* array receiver (no generic mask) that contains at least one
        // hole is materialized like a generic array-like for the callback methods:
        // presence is `HasProperty` (own or inherited) and each present value is
        // `[[Get]]` (so an inherited prototype index is observed). This keeps the
        // dense fast path (no holes) untouched while making sparse arrays
        // spec-conformant for `forEach`/`map`/`reduce`/etc.
        const HOLE_AWARE_ITER: &[&str] = &[
            "forEach",
            "map",
            "filter",
            "some",
            "every",
            "find",
            "findIndex",
            "findLast",
            "findLastIndex",
            "reduce",
            "reduceRight",
            "indexOf",
            "lastIndexOf",
        ];
        if array_like.is_none()
            && self.array_like_present.is_none()
            && HOLE_AWARE_ITER.contains(&method)
            && self
                .realm
                .array_elements(handle)
                .is_some_and(|a| a.iter().any(|e| e.is_hole()))
        {
            // A sparse array uses the conformant *live* iteration: `len` is read
            // once, then each index is probed with `HasProperty` and read with
            // `Get` at that step — so a callback/getter that fills a hole or
            // deletes an inherited index mid-iteration is observed.
            return Ok(Some(self.array_iter_sparse(method, handle, args)?));
        }
        // Take the per-index presence mask (set above for a materialized generic
        // array-like); the iteration arms below consult it to skip holes.
        let array_like_present = self.array_like_present.take();
        // For a *real* array receiver (no generic mask), record which dense slots
        // are genuine holes so the iteration arms skip them too (HasProperty is
        // false for a hole). `None` when the receiver carries a generic mask.
        let real_holes: Option<Vec<bool>> = if array_like_present.is_some() {
            None
        } else {
            self.realm
                .array_elements(handle)
                .map(|a| a.iter().map(|e| e.is_hole()).collect())
        };
        // Whether the receiver reaching the element arms is a *real* dense array
        // (not a materialized generic array-like snapshot, nor a typed array). Only
        // such a receiver may be read *live* via `[[Get]]`/`HasProperty` (so a
        // callback mutating a later index is observed); a materialized generic
        // snapshot must instead consult the recorded presence mask, since its absent
        // indices were filled with `undefined` (which `HasProperty` would report as
        // present).
        let receiver_is_real_array = real_holes.is_some();
        // `true` if index `i` is *present* (not a hole): the recorded
        // `HasProperty` for a materialized generic array-like, or the dense
        // hole check for a real array (and unconditionally `true` for any other
        // receiver kind, e.g. a typed array, which has no holes).
        let is_present = |i: usize| -> bool {
            if let Some(m) = array_like_present.as_ref() {
                return m.get(i).copied().unwrap_or(false);
            }
            real_holes
                .as_ref()
                .is_none_or(|h| !h.get(i).copied().unwrap_or(false))
        };
        // For a generic array-like `this`, the callback receives the *original*
        // object as its 3rd argument (`O`), not the materialized snapshot — so
        // `(v, i, arr) => arr === O` and `arr instanceof Boolean` hold.
        let callback_recv = NanBox::handle(handle.to_raw());
        // The *original* receiver (before any generic-array-like / sparse-array
        // materialization). `ArraySpeciesCreate` must read `constructor`/`@@species`
        // from this — a Proxy receiver is materialized into a plain-array snapshot
        // for iteration, but species selection must still see the proxy's traps /
        // the target's `constructor`.
        let species_recv = handle;
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
                    // A fill onto a view over an immutable buffer is a TypeError,
                    // verified before any argument coercion runs.
                    self.guard_view_immutable(handle)?;
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
                    // A value/start/end coercion may have detached the buffer, or
                    // shrunk a resizable one below this (fixed-length) view's extent —
                    // both make the re-derived buffer witness out of bounds, a
                    // TypeError (re-ValidateTypedArray after the coercions). A
                    // length-tracking view instead re-spans (never out of bounds on a
                    // plain shrink) and its write is just clamped to the new length.
                    // Only the *branded* `%TypedArray%.prototype.fill` re-validates;
                    // a generic `Array.prototype.fill.call(ta)` runs the ordinary
                    // array-like algorithm (a `Set` to an out-of-bounds index is a
                    // silent no-op, never a TypeError).
                    if self.typed_array_detached(handle)
                        || (!array_proto_generic && self.realm.typed_array_out_of_bounds(handle))
                    {
                        return Err(self.type_error(
                            "TypedArray.prototype.fill called on a detached or out-of-bounds ArrayBuffer",
                        ));
                    }
                    // A coercion may have shrunk a resizable buffer; re-read the live
                    // length and clamp so the write never runs past it.
                    let live = self.realm.typed_len(handle).unwrap_or(0);
                    let (start, end) = (start.min(live), end.min(live));
                    self.realm.typed_fill_range(handle, value, start, end);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `copyWithin(target, start, end?)` — copy a slice within the view
                // in place (raw same-width byte move); negatives count from the end.
                // Each relative index is ToIntegerOrInfinity (abrupt-propagating).
                "copyWithin" => {
                    // Immutable backing buffer → TypeError before argument coercion.
                    self.guard_view_immutable(handle)?;
                    let target = self.typed_clamp_index_checked(arg(0), 0, tlen)?;
                    let start = self.typed_clamp_index_checked(arg(1), 0, tlen)?;
                    let end = self.typed_clamp_index_checked(arg(2), tlen, tlen)?;
                    // `count` (elements to copy) is fixed against the *original* length
                    // captured before argument coercion — `min(end - start, len - to)`.
                    let count = end.saturating_sub(start).min(tlen.saturating_sub(target));
                    // A target/start/end coercion may have detached the buffer, or
                    // shrunk a resizable one below this (fixed-length) view's extent:
                    // the re-derived witness is out of bounds → TypeError. A
                    // length-tracking view re-spans instead (only its live length
                    // shrinks) and the copy is truncated to the current buffer below.
                    // Only the branded `%TypedArray%.prototype.copyWithin` re-validates;
                    // a generic `Array.prototype.copyWithin.call(ta)` runs the ordinary
                    // array-like algorithm (out-of-bounds `Set`s are silent no-ops).
                    if self.typed_array_detached(handle)
                        || (!array_proto_generic && self.realm.typed_array_out_of_bounds(handle))
                    {
                        return Err(self.type_error(
                            "TypedArray.prototype.copyWithin called on a detached or out-of-bounds ArrayBuffer",
                        ));
                    }
                    // A coercion may have shrunk a resizable buffer; clamp the copy so
                    // neither the source nor destination range runs past the live end
                    // (the spec's per-byte `bufferByteLimit` stop).
                    let live = self.realm.typed_len(handle).unwrap_or(0);
                    let count = count
                        .min(live.saturating_sub(target))
                        .min(live.saturating_sub(start));
                    self.realm.typed_copy_within(handle, target, start, count);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `TypedArray.prototype.set(source, offset?)`: copy a source's
                // elements into this view at `offset`, coercing each.
                // Spec order: ToIntegerOrInfinity(offset) (negative → RangeError),
                // then the typed-source or array-like-source branch.
                "set" => {
                    // A `set` onto a view over an immutable buffer is a TypeError,
                    // verified before reading `source`/`offset`.
                    self.guard_view_immutable(handle)?;
                    let target_is_bigint =
                        self.realm.typed_kind(handle).is_some_and(is_bigint_kind);
                    // Step 4-5: targetOffset = ToIntegerOrInfinity(offset); a
                    // negative offset is a RangeError. (Abrupt-propagating.)
                    let offset_n = self.coerce_to_integer_or_infinity(arg(1))?;
                    if offset_n < 0.0 {
                        let m = self.new_str("offset is out of bounds");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    // The offset coercion may have run user code that detached (or
                    // shrank out of bounds) the target buffer — ValidateTypedArray
                    // the target *after* it, so that is a TypeError (not a downstream
                    // RangeError from a now-zero length).
                    if self.typed_array_detached(handle)
                        || self.realm.typed_array_out_of_bounds(handle)
                    {
                        return Err(self.type_error(
                            "TypedArray.prototype.set called on a detached or out-of-bounds typed array",
                        ));
                    }
                    // `targetLength` is fixed here (after the offset coercion, before
                    // any source access): a `length` getter on an array-like source
                    // that resizes the target buffer must **not** be observed by the
                    // `srcLength + targetOffset > targetLength` bounds check (per-element
                    // writes below re-validate against the live length instead).
                    let target_len = self.realm.typed_len(handle).unwrap_or(0);
                    let src_box = arg(0);
                    if let Some(src) = src_box.as_handle().map(Handle::from_raw) {
                        // A typed-array source: same-kind → raw byte copy; otherwise
                        // element copy with per-element coercion. `offset + srcLen`
                        // must fit the (live) target length.
                        if let Some(src_len) = self.realm.typed_len(src) {
                            // A typed-array source detached/out-of-bounds (e.g. by the
                            // offset's valueOf) is a TypeError.
                            if self.typed_array_detached(src)
                                || self.realm.typed_array_out_of_bounds(src)
                            {
                                return Err(self.type_error(
                                    "TypedArray.prototype.set source is detached or out of bounds",
                                ));
                            }
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
                    // Array-like source: `src = ? ToObject(source)` — a `null`/
                    // `undefined` source is a TypeError (the engine's `coerce_to_object`
                    // otherwise substitutes a fresh empty object).
                    if matches!(src_box.unpack(), Unpacked::Null | Unpacked::Undefined) {
                        return Err(self.type_error(
                            "TypedArray.prototype.set source cannot be converted to an object",
                        ));
                    }
                    // Then ToLength(src.length), bounds-check, then per-element Get +
                    // coerce + write (so each value's ToNumber/ToBigInt side effects and
                    // throws run in order, and values are not cached).
                    let src_obj = self.coerce_to_object(src_box);
                    let Some(src) = src_obj.as_handle().map(Handle::from_raw) else {
                        return Ok(Some(NanBox::undefined()));
                    };
                    let len_val = self.read_member(src, "length")?;
                    // ToLength: ToIntegerOrInfinity, clamped to [0, 2^53-1].
                    let len_n = self.coerce_to_integer_or_infinity(len_val)?;
                    let src_len = len_n.clamp(0.0, 9_007_199_254_740_991.0) as usize;
                    // Bounds-check against the target length captured *before* the
                    // `length` getter ran (which may have resized the buffer).
                    let offset = if offset_n.is_finite() && offset_n <= target_len as f64 {
                        offset_n as usize
                    } else {
                        target_len + 1
                    };
                    if offset.checked_add(src_len).is_none_or(|e| e > target_len) {
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
                    // srcLength is 0 when the source is already out of bounds (the
                    // relative indices then clamp into an empty range).
                    let len = if self.realm.typed_array_out_of_bounds(handle) {
                        0
                    } else {
                        tlen
                    };
                    let start = self.typed_clamp_index_checked(arg(0), 0, len)?;
                    let end = self.typed_clamp_index_checked(arg(1), len, len)?;
                    let new_len = end.saturating_sub(start);
                    let kind = self.realm.typed_kind(handle).unwrap_or(0);
                    let elem_size = TYPED_ARRAY_KINDS[kind as usize].1 as usize;
                    let abuf = self.realm.typed_array_object(handle).unwrap();
                    let parent_off = self.realm.typed_byte_offset(handle).unwrap_or(0);
                    let sub_off = parent_off + start * elem_size;
                    // 23.2.3.30 step 15: an auto-length (length-tracking) view with
                    // `end` undefined passes only « buffer, beginByteOffset » to the
                    // species constructor (the new view length-tracks too); otherwise
                    // « buffer, beginByteOffset, newLength ».
                    let pass_length = !(self.realm.is_length_tracking(handle)
                        && matches!(arg(1).unpack(), Unpacked::Undefined));
                    // TypedArraySpeciesCreate(O, argumentsList).
                    if let Some(view) =
                        self.typed_subarray_species(handle, abuf, sub_off, new_len, pass_length)?
                    {
                        return Ok(Some(view));
                    }
                    // The default constructor `new TA(buffer, off, len)` validates the
                    // buffer: a detached buffer (e.g. by the begin/end coercion) is a
                    // TypeError. The coercions above already ran (observably).
                    if self.typed_array_detached(handle) {
                        return Err(self.type_error(
                            "TypedArray.prototype.subarray called on a detached ArrayBuffer",
                        ));
                    }
                    let bytes_h = self.realm.typed_buffer(handle).unwrap();
                    // The default `new TA(buffer, beginByteOffset, newLength)` also
                    // validates the range against the *current* buffer byte length
                    // (the source may be out of bounds, or a coercion may have shrunk
                    // a resizable buffer): a begin offset past the end — or a fixed
                    // sub-range that no longer fits — is a RangeError.
                    let buf_len = self.realm.bytes_len(bytes_h).unwrap_or(0);
                    if sub_off > buf_len
                        || sub_off
                            .checked_add(new_len.saturating_mul(elem_size))
                            .is_none_or(|e| e > buf_len)
                    {
                        let m =
                            self.new_str("TypedArray.prototype.subarray: offset is out of bounds");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    let view = self
                        .realm
                        .new_typed_array(bytes_h, abuf, sub_off, new_len, kind);
                    // Link `[[Prototype]]` to the kind's intrinsic so the result is
                    // a real instance (`toString`/`valueOf`/`instanceof`/`.constructor`
                    // resolve, and ToPrimitive does not fall through to
                    // `Function.prototype.toString`).
                    if let Some(proto) = self.intrinsic_proto(TYPED_ARRAY_KINDS[kind as usize].0) {
                        self.realm.set_native_proto(view, proto);
                    }
                    // A subarray with no explicit length over a length-tracking
                    // parent is itself length-tracking.
                    if self.realm.is_length_tracking(handle)
                        && matches!(arg(1).unpack(), Unpacked::Undefined)
                    {
                        self.realm.mark_length_tracking(view);
                        // `new_len` was computed from the parent's length at method
                        // entry, which may be stale if a begin/end coercion resized the
                        // buffer afterwards (e.g. an initially out-of-bounds parent that
                        // was grown back). Re-derive every length-tracking view's span
                        // from the current buffer (a no-op byte resize) so the fresh
                        // view reports the live length.
                        let cur = self.realm.bytes_len(bytes_h).unwrap_or(0);
                        self.realm.resize_buffer(bytes_h, cur);
                    }
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
            // Per spec the length-mutating methods finish with
            // Set(O, "length", …, Throw=true), which throws a TypeError when the
            // array's `length` is non-writable (explicitly demoted or frozen). This
            // `Set` is *unconditional* — it fires even for a no-arg `push()`/
            // `unshift()` or a `pop()`/`shift()` on an empty array (the spec still
            // performs `Set(O, "length", +0𝔽, true)`), so a non-writable length
            // throws in every one of these cases.
            let length_readonly_throw = |this: &mut Self| -> ExecError {
                let m = this.new_str(
                    "Cannot assign to read only property 'length' of object '[object Array]'",
                );
                ExecError::Throw(this.make_error(N_TYPE_ERROR, Some(m)))
            };
            // A *real* array whose dense store has holes, whose indices carry
            // attribute/accessor overrides, or whose default prototype chain has an
            // integer-indexed accessor cannot use the dense fast path: element reads
            // must `[[Get]]` through the prototype (a hole may resolve to an inherited
            // value / getter), writes must `[[Set]]` (firing inherited setters) and
            // deletes must `DeletePropertyOrThrow`. This mirrors the precise gating
            // already used by `sort`/`reverse`/`copyWithin`. A typed array (which has
            // no holes and its own exotic element methods) never takes this path.
            let precise_array = self.realm.typed_kind(handle).is_none()
                && (self.realm.array_has_index_overrides(handle)
                    || elems.iter().any(|e| e.is_hole())
                    || self.realm.proto_index_accessor_dirty());
            match method {
                "push" => {
                    // Precise: `Set(O, ToString(len+i), E, true)` fires an inherited
                    // setter, and the final `Set(O, "length", …, true)` throws on a
                    // frozen / non-writable length. Also take the precise generic path
                    // when the *logical* length exceeds the dense store (a sparse array
                    // whose `length` was widened, e.g. `x.length = 2**32-1`): the fast
                    // path would push at the wrong (dense) index, and the final
                    // `Set(O,"length", len+argCount)` must throw a RangeError once it
                    // crosses the 2^32 array-length ceiling.
                    if precise_array || self.realm.array_length(handle) != Some(elems.len()) {
                        return Ok(Some(self.array_like_mutate(method, handle, args)?));
                    }
                    if self.realm.array_length_is_readonly(handle) {
                        return Err(length_readonly_throw(self));
                    }
                    let mut len = elems.len();
                    for a in args {
                        len = self.realm.array_push(handle, *a).unwrap_or(len);
                    }
                    return Ok(Some(NanBox::number(len as f64)));
                }
                "pop" => {
                    // Precise: `Get(O, len-1)` (an inherited getter fires),
                    // `DeletePropertyOrThrow`, then `Set(O, "length", len-1, true)`
                    // (throws on a frozen / non-writable length — evaluated *after*
                    // the getter, which may itself freeze the array).
                    if precise_array {
                        return Ok(Some(self.array_like_mutate(method, handle, args)?));
                    }
                    if self.realm.array_length_is_readonly(handle) {
                        return Err(length_readonly_throw(self));
                    }
                    return Ok(Some(self.realm.array_pop(handle)));
                }
                // `splice(start, deleteCount?, ...items)` — mutate in place,
                // return the removed elements as a new array.
                "shift" => {
                    // Precise: read/move each index via `[[Get]]`/`HasProperty`/
                    // `[[Set]]`/`Delete` (inherited getters/setters fire, holes
                    // resolve through the prototype), then `Set(O, "length", …, true)`.
                    if precise_array {
                        return Ok(Some(self.array_like_mutate(method, handle, args)?));
                    }
                    if self.realm.array_length_is_readonly(handle) {
                        return Err(length_readonly_throw(self));
                    }
                    if elems.is_empty() {
                        return Ok(Some(NanBox::undefined()));
                    }
                    // The removed first element: a hole is read as `undefined`
                    // (the sentinel never escapes). The remaining elements keep
                    // their holes (shifting preserves absent indices).
                    let first = elems[0];
                    let first = if first.is_hole() {
                        NanBox::undefined()
                    } else {
                        first
                    };
                    self.realm.array_set_all(handle, elems[1..].to_vec());
                    return Ok(Some(first));
                }
                "unshift" => {
                    // Precise: shift the existing elements up via `[[Get]]`/`[[Set]]`/
                    // `Delete` (inherited accessors fire, holes preserved), write the
                    // prepended items, then `Set(O, "length", …, true)`.
                    if precise_array {
                        return Ok(Some(self.array_like_mutate(method, handle, args)?));
                    }
                    if self.realm.array_length_is_readonly(handle) {
                        return Err(length_readonly_throw(self));
                    }
                    let mut next: Vec<NanBox> = args.to_vec();
                    next.extend_from_slice(&elems);
                    let len = next.len();
                    self.realm.array_set_all(handle, next);
                    return Ok(Some(NanBox::number(len as f64)));
                }
                "splice" => {
                    let len = elems.len();
                    // `relativeStart = ToIntegerOrInfinity(start)` then
                    // `ToIntegerOrInfinity(deleteCount)` — object args coerce via
                    // `valueOf`/`@@toPrimitive` (not the string form), and a throwing
                    // coercion propagates. `f64 as usize` saturates so `Infinity`
                    // clamps to the length.
                    let start = {
                        let s = self.coerce_to_integer_or_infinity(arg(0))?;
                        if s < 0.0 {
                            (len as f64 + s).max(0.0) as usize
                        } else {
                            (s as usize).min(len)
                        }
                    };
                    let delete = if args.len() < 2 {
                        len - start
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize)
                            .min(len - start)
                    };
                    let removed: Vec<NanBox> = elems[start..start + delete].to_vec();
                    // The removed array is `ArraySpeciesCreate(O, deleteCount)`
                    // populated by `CreateDataPropertyOrThrow` (holes preserved).
                    let rem_v = self.array_species_create(species_recv, delete)?;
                    let Some(rem_h) = rem_v.as_handle().map(Handle::from_raw) else {
                        return Err(self.type_error("Array species did not return an object"));
                    };
                    let default_rem = self.realm.is_array(rem_h)
                        && self.realm.array_length(rem_h) == Some(delete)
                        && !self.realm.array_has_index_overrides(rem_h);
                    for (i, e) in removed.iter().enumerate() {
                        if e.is_hole() {
                            continue;
                        }
                        if default_rem {
                            self.realm.set_element(rem_h, i, *e);
                        } else {
                            self.create_data_property_or_throw(rem_h, i, *e)?;
                        }
                    }
                    let len_key = self.new_str("length");
                    self.assign_member_value(rem_h, len_key, NanBox::number(delete as f64))?;
                    // Spec closes with `Set(O, "length", newLen, true)`, which throws a
                    // TypeError when `length` is non-writable and the length changes.
                    let insert_count = args.len().saturating_sub(2);
                    let new_len = len - delete + insert_count;
                    if new_len != len && self.realm.array_length_is_readonly(handle) {
                        return Err(length_readonly_throw(self));
                    }
                    let mut next: Vec<NanBox> = elems[..start].to_vec();
                    next.extend_from_slice(&args[2.min(args.len())..]);
                    next.extend_from_slice(&elems[start + delete..]);
                    self.realm.array_set_all(handle, next);
                    return Ok(Some(rem_v));
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
                    // Typed array: `len` is cached, the separator's ToString may have
                    // detached/shrunk the buffer, after which each element reads as
                    // `undefined` → rendered empty. Re-read live by index.
                    if self.realm.typed_kind(handle).is_some() {
                        let len = elems.len();
                        let mut parts: Vec<String> = Vec::with_capacity(len);
                        for i in 0..len {
                            let e = self
                                .realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined);
                            parts.push(match e.unpack() {
                                Unpacked::Null | Unpacked::Undefined => String::new(),
                                Unpacked::Number(n) => crate::realm::js_number_string(n),
                                _ => self.coerce_to_string(e)?,
                            });
                        }
                        return Ok(Some(self.new_str(&parts.join(&sep))));
                    }
                    // `null`/`undefined` render empty; an object element is run
                    // through ToString (so a custom `toString` is honored). The
                    // receiver array seeds the cycle set, so a self-reference (or a
                    // mutual cycle back to it) renders empty rather than recursing.
                    // A hole/accessor/prototype-polluted array reads each element via
                    // `[[Get]]` (spec `Get(O, ToString(k))`): a hole resolves through
                    // the prototype chain (an inherited value or getter), rather than
                    // rendering empty. `sep` was already ToString'd above (spec order).
                    let mut parts: Vec<String> = Vec::with_capacity(elems.len());
                    for (k, dense) in elems.iter().enumerate() {
                        let e = if precise_array {
                            self.read_member(handle, &alloc::format!("{k}"))?
                        } else {
                            *dense
                        };
                        let s = match e.unpack() {
                            Unpacked::Null | Unpacked::Undefined => String::new(),
                            // A direct self-reference back to the receiver renders
                            // empty (per `Array.prototype.join`), without recursing.
                            Unpacked::Handle(raw) if raw == handle.to_raw() => String::new(),
                            _ => {
                                let p = self.coerce_object(e, "string")?;
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
                    // `%TypedArray%.prototype.toLocaleString`: per spec each element
                    // is `Invoke(element, "toLocaleString")` (boxing the primitive
                    // Number/BigInt so a user-overridden `Number.prototype.
                    // toLocaleString` is honored), the result ToString'd, joined by
                    // ",". Elements are re-read live from the (length-validated)
                    // buffer. (Array's path below keeps the no-Intl grouped form to
                    // preserve the curated gate.)
                    if let Some(len) = self.realm.typed_len(handle) {
                        let mut parts: Vec<String> = Vec::with_capacity(len);
                        for i in 0..len {
                            let e = self
                                .realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined);
                            // `element = Get(array, k)`; only a non-null/undefined
                            // element is `Invoke`d. A resizable buffer shrunk mid-loop
                            // (by a user `toLocaleString` override) makes later indices
                            // out of bounds → `Get` returns `undefined`, which renders
                            // as the empty string rather than throwing.
                            if matches!(e.unpack(), Unpacked::Null | Unpacked::Undefined) {
                                parts.push(String::new());
                                continue;
                            }
                            // Invoke(element, "toLocaleString"): box the Number/BigInt
                            // element (ToObject), read the (possibly user-overridden)
                            // method off the wrapper's prototype, and call it with the
                            // primitive as `this` — so a replaced Number.prototype /
                            // BigInt.prototype `toLocaleString` is honored. Abrupt
                            // completions (a throwing override) propagate.
                            let boxed = self.coerce_to_object(e);
                            let bh = boxed.as_handle().map(Handle::from_raw).unwrap();
                            let m = self.read_member(bh, "toLocaleString")?;
                            let r = self.call_with_this(m, e, &[])?;
                            parts.push(self.coerce_to_string(r)?);
                        }
                        return Ok(Some(self.new_str(&parts.join(","))));
                    }
                    // A hole/accessor/prototype-polluted array reads each element via
                    // `[[Get]]` (spec `Get(O, k)`): a hole resolves through the
                    // prototype chain, so an inherited element's `toLocaleString` is
                    // invoked too (rather than rendering empty).
                    let mut parts: Vec<String> = Vec::with_capacity(elems.len());
                    for (k, dense) in elems.iter().enumerate() {
                        let e = if precise_array {
                            self.read_member(handle, &alloc::format!("{k}"))?
                        } else {
                            *dense
                        };
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
                                    let r = self.call_with_this(m, e, &[])?;
                                    self.coerce_to_string(r)?
                                } else {
                                    // A string/boolean/symbol (or other) primitive
                                    // element: `Invoke(element, "toLocaleString")` —
                                    // box it (ToObject) to resolve the possibly
                                    // user-overridden method, then call it with the
                                    // *primitive* as `this` (so an override observing a
                                    // strict primitive `this` — e.g. `typeof this` —
                                    // sees `"boolean"`/`"string"`, not the wrapper).
                                    let boxed = self.coerce_to_object(e);
                                    let bh = boxed.as_handle().map(Handle::from_raw).unwrap();
                                    let m = self.read_member(bh, "toLocaleString")?;
                                    let r = self.call_with_this(m, e, &[])?;
                                    self.coerce_to_string(r)?
                                }
                            }
                        };
                        parts.push(s);
                    }
                    return Ok(Some(self.new_str(&parts.join(","))));
                }
                "includes" => {
                    let target = arg(0);
                    // For a typed array, `len` is cached at the start; the fromIndex
                    // coercion may detach the buffer, after which element reads are
                    // `undefined` (so `includes(0)` is false but `includes(undefined)`
                    // is true). Re-read elements live rather than from the snapshot.
                    if self.realm.typed_kind(handle).is_some() {
                        let len = elems.len();
                        // Length checked before ToInteger(fromIndex): empty → false.
                        if len == 0 {
                            return Ok(Some(NanBox::boolean(false)));
                        }
                        let from = self.array_from_index_checked(arg(1), len)?;
                        let t_nan = target.as_number().is_some_and(f64::is_nan);
                        let mut found = false;
                        for i in from..len {
                            let e = self
                                .realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined);
                            if self.realm.strict_equals(e, target)
                                || (t_nan && e.as_number().is_some_and(f64::is_nan))
                            {
                                found = true;
                                break;
                            }
                        }
                        return Ok(Some(NanBox::boolean(found)));
                    }
                    // Length is checked before ToInteger(fromIndex): an empty array
                    // returns false without coercing the (side-effecting) fromIndex.
                    if elems.is_empty() {
                        return Ok(Some(NanBox::boolean(false)));
                    }
                    let from = self.array_from_index_checked(arg(1), elems.len())?;
                    // SameValueZero: like `===` but `NaN` matches `NaN`. `includes`
                    // does *not* skip holes — a hole reads as `undefined`, so
                    // `[,].includes(undefined)` is `true`.
                    let t_nan = target.as_number().is_some_and(f64::is_nan);
                    let t_undef = matches!(target.unpack(), Unpacked::Undefined);
                    let found = elems[from..].iter().any(|e| {
                        (e.is_hole() && t_undef)
                            || self.realm.strict_equals(*e, target)
                            || (t_nan && e.as_number().is_some_and(f64::is_nan))
                    });
                    return Ok(Some(NanBox::boolean(found)));
                }
                // `toSorted`/`toReversed`/`with`/`toSpliced` — the ES2023 immutable
                // copies. On a real array these `Get(O, k)` each index after caching
                // `len`, so an accessor-at-index getter runs and a hole resolves
                // through the prototype chain. Materialize via `[[Get]]` when the
                // dense store has any hole (which includes an accessor-punched index).
                "toReversed"
                    if self.realm.typed_kind(handle).is_none()
                        && elems.iter().any(|e| e.is_hole()) =>
                {
                    let len = elems.len();
                    let mut out = Vec::with_capacity(len);
                    for k in 0..len {
                        out.push(self.read_member(handle, &alloc::format!("{k}"))?);
                    }
                    out.reverse();
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                "toSorted"
                    if self.realm.typed_kind(handle).is_none()
                        && elems.iter().any(|e| e.is_hole()) =>
                {
                    let len = elems.len();
                    let mut materialized = Vec::with_capacity(len);
                    for k in 0..len {
                        materialized.push(self.read_member(handle, &alloc::format!("{k}"))?);
                    }
                    let sorted = self.sort_array(materialized, arg(0), false)?;
                    return Ok(Some(NanBox::handle(self.realm.new_array(sorted).to_raw())));
                }
                "with"
                    if self.realm.typed_kind(handle).is_none()
                        && elems.iter().any(|e| e.is_hole()) =>
                {
                    let len = elems.len() as i64;
                    let i = self.coerce_to_integer_or_infinity(arg(0))? as i64;
                    let idx = if i < 0 { len + i } else { i };
                    if idx < 0 || idx >= len {
                        let m = self.new_str("Invalid index");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    let mut out = Vec::with_capacity(len as usize);
                    for k in 0..len as usize {
                        out.push(if k == idx as usize {
                            arg(1)
                        } else {
                            self.read_member(handle, &alloc::format!("{k}"))?
                        });
                    }
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                // `toSorted`/`toReversed`/`with` — non-mutating array methods.
                "toReversed" => {
                    let mut out = elems.clone();
                    out.reverse();
                    return Ok(Some(self.typed_like(handle, out)));
                }
                "with" => {
                    // Typed-array `%TypedArray%.prototype.with(index, value)`: spec
                    // order is ToIntegerOrInfinity(index), then ToNumber/ToBigInt(value)
                    // (which runs *even for an out-of-range index*, so its side effects
                    // happen), then IsValidIntegerIndex against the *current* length
                    // (a resize during coercion is observed) — out of range is a
                    // RangeError.
                    if self.realm.typed_kind(handle).is_some() {
                        // `len` is fixed at step 3 (before any coercion): it sizes the
                        // result and anchors a negative index. A resize during value
                        // coercion changes only the *current* length used by the
                        // IsValidIntegerIndex check below.
                        let len = elems.len() as i64;
                        let i = self.coerce_to_integer_or_infinity(arg(0))? as i64;
                        let value = if self.realm.typed_kind(handle).is_some_and(is_bigint_kind) {
                            self.coerce_typed_array_write(handle, arg(1))?
                        } else {
                            self.coerce_to_number(arg(1))?
                        };
                        let idx = if i < 0 { len + i } else { i };
                        // IsValidIntegerIndex(O, idx): validated against the *current*
                        // length (a resize during coercion is observed) and false when
                        // the view is now detached or out of bounds → RangeError.
                        let cur = self.realm.typed_len(handle).unwrap_or(0) as i64;
                        if idx < 0
                            || idx >= cur
                            || self.typed_array_detached(handle)
                            || self.realm.typed_array_out_of_bounds(handle)
                        {
                            let m = self.new_str("Invalid typed array index");
                            return Err(ExecError::Throw(
                                self.make_error(N_ERROR_BASE + 2, Some(m)),
                            ));
                        }
                        // `with` uses TypedArrayCreateSameType (NOT species), sized to
                        // the *original* `len`. `idx` (validated against the possibly
                        // larger current length) may fall outside `0..len`, in which
                        // case the value is simply not placed (loop stops at `len`).
                        let mut out = Vec::with_capacity(len as usize);
                        for k in 0..len as usize {
                            out.push(if k as i64 == idx {
                                value
                            } else {
                                self.realm
                                    .typed_get(handle, k)
                                    .unwrap_or_else(NanBox::undefined)
                            });
                        }
                        return Ok(Some(self.typed_like(handle, out)));
                    }
                    let len = elems.len() as i64;
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
                    // Typed array: re-read live after the fromIndex coercion (a detach
                    // makes every read undefined → not found).
                    if self.realm.typed_kind(handle).is_some() {
                        let len = elems.len();
                        // Length is checked before ToInteger(fromIndex): a zero-length
                        // array returns -1 without coercing fromIndex.
                        if len == 0 {
                            return Ok(Some(NanBox::number(-1.0)));
                        }
                        let from = self.array_from_index_checked(arg(1), len)?;
                        // Per spec, if the fromIndex coercion left the array detached /
                        // out of bounds, `indexOf` returns -1 (no search).
                        if self.typed_array_detached(handle)
                            || self.realm.typed_array_out_of_bounds(handle)
                        {
                            return Ok(Some(NanBox::number(-1.0)));
                        }
                        let mut idx = -1.0;
                        for i in from..len {
                            let e = self
                                .realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined);
                            if self.realm.strict_equals(e, target) {
                                idx = i as f64;
                                break;
                            }
                        }
                        return Ok(Some(NanBox::number(idx)));
                    }
                    // A sparse array (logical length beyond the dense store) is
                    // scanned by its *present* indices in ascending order from
                    // `from` — iterating every index up to `len` would be billions
                    // of steps. Present = dense non-hole slots plus any element
                    // stored past the dense cap as an aux integer-key property; a
                    // named key `>= len` (a boundary/2**32+ property) is not an
                    // element and is excluded.
                    let dense_len = elems.len();
                    let logical = self.realm.array_length(handle).unwrap_or(dense_len);
                    if logical > dense_len {
                        let from = self.array_from_index_checked(arg(1), logical)?;
                        let mut present: Vec<usize> = (from..dense_len.min(logical))
                            .filter(|&i| !elems[i].is_hole())
                            .collect();
                        for k in self.realm.aux_all_keys(handle) {
                            if let Ok(i) = k.parse::<usize>()
                                && i >= from
                                && i < logical
                                && alloc::format!("{i}") == k
                            {
                                present.push(i);
                            }
                        }
                        present.sort_unstable();
                        for &i in &present {
                            let e = self.read_member(handle, &alloc::format!("{i}"))?;
                            if self.realm.strict_equals(e, target) {
                                return Ok(Some(NanBox::number(i as f64)));
                            }
                        }
                        return Ok(Some(NanBox::number(-1.0)));
                    }
                    // Length is checked before ToInteger(fromIndex): an empty array
                    // returns -1 without coercing the (possibly side-effecting)
                    // fromIndex argument.
                    if elems.is_empty() {
                        return Ok(Some(NanBox::number(-1.0)));
                    }
                    let from = self.array_from_index_checked(arg(1), elems.len())?;
                    let mut idx = -1.0;
                    for (i, e) in elems.iter().enumerate().skip(from) {
                        // `indexOf` skips holes (HasProperty is false).
                        if is_present(i) && self.realm.strict_equals(*e, target) {
                            idx = i as f64;
                            break;
                        }
                    }
                    return Ok(Some(NanBox::number(idx)));
                }
                "map" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = callback_recv;
                    // Typed-array `%TypedArray%.prototype.map`: per spec the result
                    // is allocated via TypedArraySpeciesCreate(O, «len») *before*
                    // the loop (a throwing species getter/ctor must abort before
                    // any callback runs), `len` is the *initial* length, and each
                    // kValue is read live from the (possibly resized) buffer — not
                    // from a cached snapshot.
                    if let Some(len) = self.realm.typed_len(handle) {
                        self.require_callable(f, "map callback")?;
                        let dest = self.typed_species_create(handle, len)?;
                        // A species result over an immutable buffer fails before any
                        // callback runs (the writes could never succeed).
                        self.guard_view_immutable(dest)?;
                        for i in 0..len {
                            // ToNumber/ToBigInt of an out-of-bounds (shrunk) index
                            // yields `undefined`; the callback still runs per spec.
                            let kv = self
                                .realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined);
                            let cb_args = [kv, NanBox::number(i as f64), arr];
                            let mapped = self.call_with_this(f, this_arg, &cb_args)?;
                            // Set(A, k, mappedValue) — coerce per the result kind
                            // (a BigInt-result write of a Number throws TypeError).
                            self.set_element_checked(dest, i, mapped)?;
                        }
                        return Ok(Some(NanBox::handle(dest.to_raw())));
                    }
                    // A real array reads each `kValue` *live* via `[[Get]]` (a
                    // callback that mutates a later index is observed); holes are
                    // skipped and preserved as holes in the dense result. A
                    // materialized generic array-like reads its snapshot + mask.
                    let live = receiver_is_real_array;
                    let mut out = Vec::with_capacity(elems.len());
                    for i in 0..elems.len() {
                        let present = is_present(i);
                        match self.array_cb_read(
                            handle,
                            i,
                            false,
                            live,
                            &elems,
                            present,
                            true,
                            array_proto_generic,
                        )? {
                            None => out.push(NanBox::hole()),
                            Some(e) => {
                                let cb_args = [e, NanBox::number(i as f64), arr];
                                out.push(self.call_with_this(f, this_arg, &cb_args)?);
                            }
                        }
                    }
                    // A typed-array `map` allocates via TypedArraySpeciesCreate.
                    return Ok(Some(self.typed_like_species(species_recv, out)?));
                }
                "filter" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = callback_recv;
                    // Typed array: `len` is cached, each kValue read live (a callback
                    // resize/detach is observed), kept values collected, then the
                    // result is allocated via TypedArraySpeciesCreate(O, «kept»).
                    if self.realm.typed_kind(handle).is_some() {
                        self.require_callable(f, "filter callback")?;
                        let len = elems.len();
                        let mut out = Vec::new();
                        for i in 0..len {
                            let e = self
                                .realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined);
                            let cb_args = [e, NanBox::number(i as f64), arr];
                            let r = self.call_with_this(f, this_arg, &cb_args)?;
                            if self.realm.truthy(r) {
                                out.push(e);
                            }
                        }
                        return Ok(Some(self.typed_like_species(species_recv, out)?));
                    }
                    // A real array reads each `kValue` *live* via `[[Get]]` (a
                    // callback that mutates a later index is observed); holes are
                    // skipped. A materialized generic array-like reads snapshot + mask.
                    let live = receiver_is_real_array;
                    let mut out = Vec::new();
                    for i in 0..elems.len() {
                        let present = is_present(i);
                        let Some(e) = self.array_cb_read(
                            handle,
                            i,
                            false,
                            live,
                            &elems,
                            present,
                            true,
                            array_proto_generic,
                        )?
                        else {
                            continue; // holes are skipped
                        };
                        let cb_args = [e, NanBox::number(i as f64), arr];
                        let r = self.call_with_this(f, this_arg, &cb_args)?;
                        if self.realm.truthy(r) {
                            out.push(e);
                        }
                    }
                    return Ok(Some(self.typed_like_species(species_recv, out)?));
                }
                "forEach" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = callback_recv;
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    // A typed array / real array re-reads each element live by index
                    // (a callback mutation is observed); a materialized generic
                    // array-like reads its snapshot. Holes are skipped.
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..elems.len() {
                        let present = is_present(i);
                        let Some(e) = self.array_cb_read(
                            handle,
                            i,
                            typed,
                            live,
                            &elems,
                            present,
                            true,
                            array_proto_generic,
                        )?
                        else {
                            continue;
                        };
                        let cb_args = [e, NanBox::number(i as f64), arr];
                        self.call_with_this(f, this_arg, &cb_args)?;
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "reduce" => {
                    let f = arg(0);
                    let arr = callback_recv;
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    let mut acc;
                    let mut start = 0;
                    if args.len() >= 2 {
                        acc = arg(1);
                    } else {
                        // Seed from the first *present* element; a holes-only (or
                        // empty) array with no initial value is a TypeError.
                        let mut seed = None;
                        let mut k = 0;
                        while k < elems.len() {
                            let present = is_present(k);
                            if let Some(v) = self.array_cb_read(
                                handle,
                                k,
                                typed,
                                live,
                                &elems,
                                present,
                                true,
                                array_proto_generic,
                            )? {
                                seed = Some(v);
                                start = k + 1;
                                break;
                            }
                            k += 1;
                        }
                        match seed {
                            Some(v) => acc = v,
                            None => {
                                let m = self.new_str("Reduce of empty array with no initial value");
                                return Err(ExecError::Throw(
                                    self.make_error(N_TYPE_ERROR, Some(m)),
                                ));
                            }
                        }
                    }
                    for i in start..elems.len() {
                        let present = is_present(i);
                        let Some(e) = self.array_cb_read(
                            handle,
                            i,
                            typed,
                            live,
                            &elems,
                            present,
                            true,
                            array_proto_generic,
                        )?
                        else {
                            continue; // holes are skipped
                        };
                        acc = self.call(f, &[acc, e, NanBox::number(i as f64), arr])?;
                    }
                    return Ok(Some(acc));
                }
                // `reduceRight` — like `reduce` but right-to-left.
                "reduceRight" => {
                    let f = arg(0);
                    let arr = callback_recv;
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    let mut acc;
                    let mut idx = elems.len();
                    if args.len() >= 2 {
                        acc = arg(1);
                    } else {
                        // Seed from the last present element.
                        let mut seed = None;
                        let mut k = elems.len();
                        while k > 0 {
                            k -= 1;
                            let present = is_present(k);
                            if let Some(v) = self.array_cb_read(
                                handle,
                                k,
                                typed,
                                live,
                                &elems,
                                present,
                                true,
                                array_proto_generic,
                            )? {
                                seed = Some(v);
                                idx = k;
                                break;
                            }
                        }
                        match seed {
                            Some(v) => acc = v,
                            None => {
                                let m = self.new_str("Reduce of empty array with no initial value");
                                return Err(ExecError::Throw(
                                    self.make_error(N_TYPE_ERROR, Some(m)),
                                ));
                            }
                        }
                    }
                    while idx > 0 {
                        idx -= 1;
                        let present = is_present(idx);
                        let Some(e) = self.array_cb_read(
                            handle,
                            idx,
                            typed,
                            live,
                            &elems,
                            present,
                            true,
                            array_proto_generic,
                        )?
                        else {
                            continue; // holes are skipped
                        };
                        acc = self.call(f, &[acc, e, NanBox::number(idx as f64), arr])?;
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
                        let count = b.saturating_sub(a);
                        // TypedArraySpeciesCreate(O, «count») runs first (it may itself
                        // read a custom constructor that detaches the source).
                        let dest = self.typed_species_create(handle, count)?;
                        // Spec: if count > 0, re-check the source — a start/end valueOf
                        // or the species constructor may have detached / shrunk it; that
                        // is a TypeError (the snapshot `elems` would otherwise copy
                        // stale/zeroed data).
                        if count > 0
                            && (self.typed_array_detached(handle)
                                || self.realm.typed_array_out_of_bounds(handle))
                        {
                            return Err(self.type_error(
                                "TypedArray.prototype.slice called on a detached or out-of-bounds typed array",
                            ));
                        }
                        // Copy live elements. A resizable buffer may have shrunk
                        // during the start/end coercion (a length-tracking view stays
                        // valid but shorter); only indices still within the live length
                        // are copied — the rest keep the destination's zero fill (a
                        // read past the end would coerce to NaN, which is wrong).
                        let live = self.realm.typed_len(handle).unwrap_or(0);
                        for (k, i) in (a..b).enumerate() {
                            if i >= live {
                                break;
                            }
                            let v = self
                                .realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined);
                            self.set_element_checked(dest, k, v)?;
                        }
                        return Ok(Some(NanBox::handle(dest.to_raw())));
                    }
                    // `start`/`end` are `ToIntegerOrInfinity` (object args coerce
                    // via `valueOf`, throws propagate); `end` defaults to `len`.
                    let start_n = self.coerce_to_integer_or_infinity(arg(0))?;
                    let end_n = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        None
                    } else {
                        Some(self.coerce_to_integer_or_infinity(arg(1))?)
                    };
                    let (a, b) = slice_bounds(start_n, end_n, elems.len());
                    let count = b.saturating_sub(a);
                    // `A = ArraySpeciesCreate(O, count)`, then
                    // `CreateDataPropertyOrThrow` each *present* element (holes are
                    // skipped, leaving a hole in the copy), then `Set length`.
                    let a_v = self.array_species_create(species_recv, count)?;
                    let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
                        return Err(self.type_error("Array species did not return an object"));
                    };
                    // Only a pristine dense species result (no per-index overrides)
                    // may take the raw write-through; a custom species with existing
                    // non-default index attributes needs CreateDataPropertyOrThrow.
                    let default_array = self.realm.is_array(a_h)
                        && self.realm.array_length(a_h) == Some(count)
                        && !self.realm.array_has_index_overrides(a_h);
                    for k in 0..count {
                        // Spec `slice`: `kPresent = HasProperty(O, k)`; only a present
                        // index is `Get` and `CreateDataProperty`d into the copy (an
                        // absent index leaves a hole, `n`/`k` still advancing). A
                        // hole/accessor/prototype-polluted array resolves presence and
                        // value through the prototype chain (so an inherited element
                        // becomes an own property in the copy); a dense array reads the
                        // snapshot directly and skips genuine holes.
                        let e = if precise_array {
                            if !self.has_property(handle, &alloc::format!("{}", a + k)) {
                                continue;
                            }
                            self.read_member(handle, &alloc::format!("{}", a + k))?
                        } else {
                            let e = elems[a + k];
                            if e.is_hole() {
                                continue;
                            }
                            e
                        };
                        if default_array {
                            self.realm.set_element(a_h, k, e);
                        } else {
                            self.create_data_property_or_throw(a_h, k, e)?;
                        }
                    }
                    let len_key = self.new_str("length");
                    self.assign_member_value(a_h, len_key, NanBox::number(count as f64))?;
                    return Ok(Some(a_v));
                }
                // Iterators: `keys()` over indices, `values()` over elements,
                // `entries()` over `[index, element]` pairs. A **typed array** gets
                // a live iterator (re-reads its length/elements, observing a
                // resizable-buffer resize or element write mid-iteration); a plain
                // array uses a snapshot.
                "keys" | "values" | "entries" if self.realm.typed_kind(handle).is_some() => {
                    let kind = match method {
                        "keys" => 0,
                        "entries" => 2,
                        _ => 1,
                    };
                    return Ok(Some(self.make_live_typed_iterator(handle, kind)));
                }
                // A **real** array yields a live iterator (`CreateArrayIterator`
                // re-reads `length`/`Get(k)` each `next()`, so an element pushed or
                // assigned after `.values()`/`.keys()`/`.entries()` is observed); a
                // materialized generic array-like keeps its snapshot.
                "keys" if self.realm.is_array(handle) => {
                    return Ok(Some(self.make_live_array_iterator(handle, 0)));
                }
                "values" if self.realm.is_array(handle) => {
                    return Ok(Some(self.make_live_array_iterator(handle, 1)));
                }
                "entries" if self.realm.is_array(handle) => {
                    return Ok(Some(self.make_live_array_iterator(handle, 2)));
                }
                "keys" => {
                    let ks: Vec<NanBox> =
                        (0..elems.len()).map(|i| NanBox::number(i as f64)).collect();
                    return Ok(Some(self.make_builtin_iterator(ks, "Array Iterator")));
                }
                "values" => {
                    return Ok(Some(
                        self.make_builtin_iterator(elems.clone(), "Array Iterator"),
                    ));
                }
                "entries" => {
                    let mut pairs = Vec::with_capacity(elems.len());
                    for (i, e) in elems.iter().enumerate() {
                        let pair = self
                            .realm
                            .new_array(alloc::vec![NanBox::number(i as f64), *e]);
                        pairs.push(NanBox::handle(pair.to_raw()));
                    }
                    return Ok(Some(self.make_builtin_iterator(pairs, "Array Iterator")));
                }
                "concat" => {
                    // A real-array receiver: run the shared, spec-conformant
                    // `Array.prototype.concat` over `O = this` (already an object).
                    return Ok(Some(self.array_concat(recv, args)?));
                }
                "reverse" => {
                    // A typed-array view over an immutable buffer cannot be reordered.
                    self.guard_view_immutable(handle)?;
                    // A plain array with hole/accessor-override indices runs the
                    // spec-precise reverse over `HasProperty`/`[[Get]]`/`[[Set]]`/
                    // `DeletePropertyOrThrow` (so index getters/setters fire and
                    // holes are preserved by move-and-delete); a dense array / typed
                    // array reverses its element store directly.
                    if self.realm.typed_kind(handle).is_none()
                        && let Some(len) = self.realm.array_length(handle)
                        && (self.realm.array_has_index_overrides(handle)
                            || elems.iter().any(|e| e.is_hole()))
                    {
                        for lower in 0..len / 2 {
                            let upper = len - lower - 1;
                            let (lk, uk) = (alloc::format!("{lower}"), alloc::format!("{upper}"));
                            let lo_exists = self.has_property(handle, &lk);
                            let lo = if lo_exists {
                                self.read_member(handle, &lk)?
                            } else {
                                NanBox::undefined()
                            };
                            let up_exists = self.has_property(handle, &uk);
                            let up = if up_exists {
                                self.read_member(handle, &uk)?
                            } else {
                                NanBox::undefined()
                            };
                            let lkb = self.new_str(&lk);
                            let ukb = self.new_str(&uk);
                            match (lo_exists, up_exists) {
                                (true, true) => {
                                    self.set_or_throw(handle, lkb, &lk, up)?;
                                    self.set_or_throw(handle, ukb, &uk, lo)?;
                                }
                                (false, true) => {
                                    self.set_or_throw(handle, lkb, &lk, up)?;
                                    self.delete_property_of(handle, &uk)?;
                                }
                                (true, false) => {
                                    self.delete_property_of(handle, &lk)?;
                                    self.set_or_throw(handle, ukb, &uk, lo)?;
                                }
                                (false, false) => {}
                            }
                        }
                        return Ok(Some(NanBox::handle(handle.to_raw())));
                    }
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
                        let n = self.coerce_to_integer_or_infinity(arg(1))?;
                        if n < 0.0 {
                            (len as f64 + n).max(0.0) as usize
                        } else {
                            (n as usize).min(len)
                        }
                    };
                    let end = if matches!(arg(2).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        let n = self.coerce_to_integer_or_infinity(arg(2))?;
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
                        self.coerce_to_integer_or_infinity(arg(0))? as i32
                    };
                    // `A = ArraySpeciesCreate(O, 0)` — throws a TypeError for a
                    // non-constructor `@@species`, *before* any element access.
                    let a_v = self.array_species_create(species_recv, 0)?;
                    let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
                        return Err(self.type_error("Array species did not return an object"));
                    };
                    // FlattenIntoArray processes only *present* elements
                    // (HasProperty). Mark an absent generic-array-like index as a
                    // hole so `flatten` skips it (real-array holes are already
                    // `is_hole()`), e.g. `[].flat.call({length:3,0:1,2:[2,3]})`.
                    let src: Vec<NanBox> = elems
                        .iter()
                        .enumerate()
                        .map(|(i, e)| if is_present(i) { *e } else { NanBox::hole() })
                        .collect();
                    let out = self.flatten(&src, depth, 0)?;
                    // `CreateDataPropertyOrThrow` each element into the target — a
                    // non-extensible / non-configurable target throws. The default
                    // fresh array takes the fast element write.
                    let default_array = self.realm.is_array(a_h) && !self.realm.is_frozen(a_h);
                    for (k, e) in out.into_iter().enumerate() {
                        if default_array {
                            self.realm.set_element(a_h, k, e);
                        } else {
                            self.create_data_property_or_throw(a_h, k, e)?;
                        }
                    }
                    return Ok(Some(a_v));
                }
                // `copyWithin(target, start, end?)` — copy a slice within the
                // array in place; negatives count from the end.
                "copyWithin" => {
                    let len = elems.len() as i64;
                    let norm = |v: f64| -> i64 {
                        let i = v as i64;
                        if i < 0 { (len + i).max(0) } else { i.min(len) }
                    };
                    let target = norm(self.coerce_to_integer_or_infinity(arg(0))?);
                    let start = norm(self.coerce_to_integer_or_infinity(arg(1))?);
                    let end = if matches!(arg(2).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        norm(self.coerce_to_integer_or_infinity(arg(2))?)
                    };
                    // A hole/accessor array copies through `[[Get]]`/`[[Set]]`/
                    // `Delete` (getters/setters fire, holes propagate), choosing the
                    // copy *direction* so overlapping ranges are not clobbered. The
                    // precise path is also taken when coercing `target`/`start`/`end`
                    // (each ToInteger may run a user `valueOf`) mutated the array —
                    // e.g. shrank its length — so the copy must consult the *live*
                    // array (`HasProperty`/`[[Get]]`/`Delete`), not the stale
                    // pre-coercion `elems` snapshot (`len` stays the captured value).
                    if self.realm.array_has_index_overrides(handle)
                        || elems.iter().any(|e| e.is_hole())
                        || self.realm.array_length(handle) != Some(elems.len())
                        || self.realm.array_dense_len(handle) != Some(elems.len())
                    {
                        let count = (end - start).min(len - target).max(0);
                        let (mut from, mut to, dir) = if start < target && target < start + count {
                            (start + count - 1, target + count - 1, -1i64)
                        } else {
                            (start, target, 1i64)
                        };
                        let mut c = count;
                        while c > 0 {
                            let (fk, tk) = (alloc::format!("{from}"), alloc::format!("{to}"));
                            if self.has_property(handle, &fk) {
                                let v = self.read_member(handle, &fk)?;
                                let tkb = self.new_str(&tk);
                                self.set_or_throw(handle, tkb, &tk, v)?;
                            } else {
                                self.delete_property_of(handle, &tk)?;
                            }
                            from += dir;
                            to += dir;
                            c -= 1;
                        }
                        return Ok(Some(NanBox::handle(handle.to_raw())));
                    }
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
                    // `A = ArraySpeciesCreate(O, 0)` first — a non-constructor
                    // `@@species` is a TypeError before any mapping.
                    let a_v = self.array_species_create(species_recv, 0)?;
                    let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
                        return Err(self.type_error("Array species did not return an object"));
                    };
                    let default_array = self.realm.is_array(a_h) && !self.realm.is_frozen(a_h);
                    // `flatMap(callbackfn, thisArg)` — the optional second argument
                    // is the `this` value for each callback invocation.
                    let this_arg = arg(1);
                    let mut k = 0usize;
                    for (i, e) in elems.iter().enumerate() {
                        // Only *present* elements are mapped (HasProperty); an
                        // absent generic-array-like index (or a real-array hole) is
                        // skipped, so a poisoned getter past `length` is never read.
                        if !is_present(i) {
                            continue;
                        }
                        let r = self.call_with_this(
                            f,
                            this_arg,
                            &[*e, NanBox::number(i as f64), callback_recv],
                        )?;
                        // Flatten one level: a mapped array spreads its *present*
                        // elements (holes skipped), a non-array is appended.
                        let items: Vec<NanBox> = match r
                            .as_handle()
                            .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                            .map(<[_]>::to_vec)
                        {
                            Some(inner) => inner.into_iter().filter(|v| !v.is_hole()).collect(),
                            None => alloc::vec![r],
                        };
                        for it in items {
                            if default_array {
                                self.realm.set_element(a_h, k, it);
                            } else {
                                self.create_data_property_or_throw(a_h, k, it)?;
                            }
                            k += 1;
                        }
                    }
                    return Ok(Some(a_v));
                }
                // `at` with negative-from-end indexing. The index is
                // ToIntegerOrInfinity (a Symbol/abrupt valueOf throws).
                "at" => {
                    let i = self.coerce_to_integer_or_infinity(arg(0))?;
                    let idx = if i < 0.0 { elems.len() as f64 + i } else { i };
                    // Typed array: the index coercion may have resized the buffer, so
                    // read live by index against the current length (an out-of-range
                    // or now-detached read is `undefined`).
                    if self.realm.typed_kind(handle).is_some() {
                        let cur = self.realm.typed_len(handle).unwrap_or(0);
                        return Ok(Some(
                            as_index(idx)
                                .filter(|&u| u < cur)
                                .map(|u| {
                                    self.realm
                                        .typed_get(handle, u)
                                        .unwrap_or_else(NanBox::undefined)
                                })
                                .unwrap_or(NanBox::undefined()),
                        ));
                    }
                    // Plain array: `at` reads via `[[Get]]`, so a hole reads `undefined`.
                    let v = as_index(idx)
                        .and_then(|u| elems.get(u))
                        .copied()
                        .unwrap_or(NanBox::undefined());
                    return Ok(Some(if v.is_hole() { NanBox::undefined() } else { v }));
                }
                "lastIndexOf" => {
                    let target = arg(0);
                    let typed = self.realm.typed_kind(handle).is_some();
                    let dense_len = elems.len();
                    // The *logical* length (a sparse array's may exceed the dense
                    // snapshot: `arr[2**32-2] = v` leaves the dense store empty).
                    let len = if typed {
                        dense_len
                    } else {
                        self.realm.array_length(handle).unwrap_or(dense_len)
                    };
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
                    // Per spec, a fromIndex coercion that detached / OOB'd the typed
                    // array makes `lastIndexOf` return -1 (no search).
                    if typed
                        && (self.typed_array_detached(handle)
                            || self.realm.typed_array_out_of_bounds(handle))
                    {
                        return Ok(Some(NanBox::number(-1.0)));
                    }
                    // A sparse array (logical length beyond the dense store) is walked
                    // by its *present* indices in descending order — iterating every
                    // index down from `from` would be billions of steps. Present =
                    // dense non-hole slots plus any element stored past the dense cap
                    // as an aux integer-key named property.
                    if !typed && len > dense_len {
                        let mut present: Vec<usize> = (0..dense_len.min(from + 1))
                            .filter(|&i| !elems[i].is_hole())
                            .collect();
                        for k in self.realm.aux_all_keys(handle) {
                            if let Ok(i) = k.parse::<usize>()
                                && i <= from
                                && alloc::format!("{i}") == k
                            {
                                present.push(i);
                            }
                        }
                        present.sort_unstable();
                        for &i in present.iter().rev() {
                            let e = self.read_member(handle, &alloc::format!("{i}"))?;
                            if self.realm.strict_equals(e, target) {
                                return Ok(Some(NanBox::number(i as f64)));
                            }
                        }
                        return Ok(Some(NanBox::number(-1.0)));
                    }
                    let mut found = -1.0;
                    for i in (0..=from).rev() {
                        // Typed array: read live (a fromIndex detach makes every read
                        // undefined). A plain array skips holes.
                        let e = if typed {
                            self.realm
                                .typed_get(handle, i)
                                .unwrap_or_else(NanBox::undefined)
                        } else if is_present(i) {
                            elems[i]
                        } else {
                            continue;
                        };
                        if self.realm.strict_equals(e, target) {
                            found = i as f64;
                            break;
                        }
                    }
                    return Ok(Some(NanBox::number(found)));
                }
                "find" => {
                    let f = arg(0);
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..elems.len() {
                        let e = self
                            .array_cb_read(
                                handle,
                                i,
                                typed,
                                live,
                                &elems,
                                false,
                                false,
                                array_proto_generic,
                            )?
                            .unwrap_or_else(NanBox::undefined);
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(e));
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "findIndex" => {
                    let f = arg(0);
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..elems.len() {
                        let e = self
                            .array_cb_read(
                                handle,
                                i,
                                typed,
                                live,
                                &elems,
                                false,
                                false,
                                array_proto_generic,
                            )?
                            .unwrap_or_else(NanBox::undefined);
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::number(i as f64)));
                        }
                    }
                    return Ok(Some(NanBox::number(-1.0)));
                }
                // `findLast`/`findLastIndex` — scan right-to-left.
                "findLast" => {
                    let f = arg(0);
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    #[allow(clippy::needless_range_loop)]
                    for i in (0..elems.len()).rev() {
                        let e = self
                            .array_cb_read(
                                handle,
                                i,
                                typed,
                                live,
                                &elems,
                                false,
                                false,
                                array_proto_generic,
                            )?
                            .unwrap_or_else(NanBox::undefined);
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(e));
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "findLastIndex" => {
                    let f = arg(0);
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    #[allow(clippy::needless_range_loop)]
                    for i in (0..elems.len()).rev() {
                        let e = self
                            .array_cb_read(
                                handle,
                                i,
                                typed,
                                live,
                                &elems,
                                false,
                                false,
                                array_proto_generic,
                            )?
                            .unwrap_or_else(NanBox::undefined);
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::number(i as f64)));
                        }
                    }
                    return Ok(Some(NanBox::number(-1.0)));
                }
                "some" => {
                    let f = arg(0);
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..elems.len() {
                        // Typed array: re-read live. Real array: live `[[Get]]` (a
                        // callback mutation is observed). Both skip absent indices.
                        let present = is_present(i);
                        let Some(e) = self.array_cb_read(
                            handle,
                            i,
                            typed,
                            live,
                            &elems,
                            present,
                            true,
                            array_proto_generic,
                        )?
                        else {
                            continue;
                        };
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::boolean(true)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(false)));
                }
                "every" => {
                    let f = arg(0);
                    let typed = self.realm.typed_kind(handle).is_some();
                    let live = !typed && self.realm.is_array(handle);
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..elems.len() {
                        let present = is_present(i);
                        let Some(e) = self.array_cb_read(
                            handle,
                            i,
                            typed,
                            live,
                            &elems,
                            present,
                            true,
                            array_proto_generic,
                        )?
                        else {
                            continue;
                        };
                        if !self.call_truthy_this(
                            f,
                            arg(1),
                            &[e, NanBox::number(i as f64), callback_recv],
                        )? {
                            return Ok(Some(NanBox::boolean(false)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(true)));
                }
                "sort" => {
                    // A typed-array view over an immutable buffer cannot be sorted
                    // in place (checked before the comparator runs).
                    self.guard_view_immutable(handle)?;
                    // Sorts in place and returns the same array. A typed array sorts
                    // numerically by default (a plain array lexicographically).
                    let numeric = self.realm.typed_kind(handle).is_some();
                    // A plain array with hole or accessor-override indices runs the
                    // spec-precise `SortIndexedProperties`: collect every *present*
                    // element via `[[Get]]` (index getters fire), sort, write the
                    // sorted values back via `[[Set]]` (setters fire), then
                    // `DeletePropertyOrThrow` the trailing indices. An ordinary
                    // dense array (and every typed array) keeps the fast in-memory
                    // sort over its element store.
                    if !numeric
                        && let Some(len) = self.realm.array_length(handle)
                        && (self.realm.array_has_index_overrides(handle)
                            || elems.iter().any(|e| e.is_hole()))
                    {
                        // Validate the comparefn up front (before any element access).
                        if !matches!(arg(0).unpack(), Unpacked::Undefined)
                            && !self.is_callable_value(arg(0))
                        {
                            return Err(self.type_error("comparefn must be a function"));
                        }
                        // Collect present elements: `HasProperty` then `[[Get]]`.
                        let mut items = Vec::new();
                        for i in 0..len {
                            let key = alloc::format!("{i}");
                            if self.has_property(handle, &key) {
                                items.push(self.read_member(handle, &key)?);
                            }
                        }
                        let count = items.len();
                        let sorted = self.sort_array(items, arg(0), false)?;
                        // Write the sorted values back via `[[Set]]` (fires setters),
                        // then delete the trailing indices.
                        for (i, v) in sorted.into_iter().enumerate() {
                            let key = alloc::format!("{i}");
                            let kb = self.new_str(&key);
                            self.set_or_throw(handle, kb, &key, v)?;
                        }
                        for i in count..len {
                            let key = alloc::format!("{i}");
                            self.delete_property_of(handle, &key)?;
                        }
                        return Ok(Some(NanBox::handle(handle.to_raw())));
                    }
                    let sorted = self.sort_array(elems, arg(0), numeric)?;
                    if numeric {
                        // A typed array: the comparator may have shrunk/detached a
                        // resizable backing buffer. The write-back is a sequence of
                        // `Set`s that re-validate each index, so only the still-in-bounds
                        // prefix is written (a fixed-length view now out of bounds has a
                        // live length of 0 → nothing is written back).
                        let live = self.realm.typed_len(handle).unwrap_or(0);
                        let mut sorted = sorted;
                        sorted.truncate(live);
                        self.realm.typed_set_from_numbers(handle, 0, &sorted);
                    } else {
                        self.write_back_elements(handle, sorted);
                    }
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `toSpliced(start, deleteCount, ...items)` — a spliced copy
                // (the ES2023 immutable counterpart of `splice`). On a real array
                // with holes/accessor indices, retained elements are read via
                // `[[Get]]` (so getters fire and holes resolve through the proto).
                "toSpliced"
                    if self.realm.typed_kind(handle).is_none()
                        && elems.iter().any(|e| e.is_hole()) =>
                {
                    let len = elems.len() as i64;
                    let start = {
                        let s = self.coerce_to_integer_or_infinity(arg(0))? as i64;
                        if s < 0 { (len + s).max(0) } else { s.min(len) }
                    } as usize;
                    let del = if args.len() < 2 {
                        elems.len() - start
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize)
                            .min(elems.len() - start)
                    };
                    let mut out: Vec<NanBox> = Vec::new();
                    for k in 0..start {
                        out.push(self.read_member(handle, &alloc::format!("{k}"))?);
                    }
                    out.extend_from_slice(&args[2.min(args.len())..]);
                    for k in (start + del)..elems.len() {
                        out.push(self.read_member(handle, &alloc::format!("{k}"))?);
                    }
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                "toSpliced" => {
                    let len = elems.len() as i64;
                    let start = {
                        let s = self.coerce_to_integer_or_infinity(arg(0))? as i64;
                        if s < 0 { (len + s).max(0) } else { s.min(len) }
                    } as usize;
                    let del = if args.len() < 2 {
                        elems.len() - start
                    } else {
                        (self.coerce_to_integer_or_infinity(arg(1))?.max(0.0) as usize)
                            .min(elems.len() - start)
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
                // `Map.prototype.getOrInsert(key, value)` /
                // `WeakMap.prototype.getOrInsert(key, value)` (upsert proposal):
                // return the existing value if `key` is present, else insert
                // `value` and return it. The receiver is brand-checked by the
                // first-class dispatch (`N_MAP_PROTO_FN`/`N_WEAKMAP_PROTO_FN`), so
                // here we only canonicalize the key, validate weak-holdability for
                // a WeakMap, then read/insert.
                "getOrInsert" if self.realm.collection_is_set(handle) == Some(false) => {
                    self.guard_get_or_insert_key(handle, arg(0))?;
                    let key = Self::canonicalize_collection_key(arg(0));
                    if let Some(v) = self.realm.collection_get(handle, key) {
                        return Ok(Some(v));
                    }
                    self.realm.collection_set(handle, key, arg(1));
                    return Ok(Some(arg(1)));
                }
                // `Map.prototype.getOrInsertComputed(key, callbackfn)` /
                // `WeakMap.prototype.getOrInsertComputed(key, callbackfn)`: return
                // the existing value, else call `callbackfn(canonicalKey)`, insert
                // the result, and return it. `callbackfn` must be callable — that
                // check happens *before* probing for the key (so it throws even
                // when the key is present). The presence is re-checked *after* the
                // callback (which may have mutated the map); the spec overwrites
                // with the computed value either way.
                "getOrInsertComputed" if self.realm.collection_is_set(handle) == Some(false) => {
                    self.guard_get_or_insert_key(handle, arg(0))?;
                    self.require_callable(arg(1), "getOrInsertComputed callback")?;
                    let key = Self::canonicalize_collection_key(arg(0));
                    if let Some(v) = self.realm.collection_get(handle, key) {
                        return Ok(Some(v));
                    }
                    // The callback receives the *canonical* key as its sole argument.
                    let computed = self.call(arg(1), &[key])?;
                    self.realm.collection_set(handle, key, computed);
                    return Ok(Some(computed));
                }
                "forEach" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    // IsCallable(callbackfn) is checked before any iteration, so
                    // `new Map().forEach({})` on an empty collection still throws.
                    self.require_callable(f, "Map.prototype.forEach callback")?;
                    let coll = NanBox::handle(handle.to_raw());
                    let is_set = self.realm.collection_is_set(handle) == Some(true);
                    // LIVE iteration (mirrors the keys/values/entries iterator): a
                    // value added during the callback is visited, a deleted one is
                    // skipped, and a delete-then-re-add is revisited. Track the last
                    // key and re-read entries each step, resuming after the last key's
                    // current position (or the recorded index if it was deleted).
                    let mut last_key: Option<NanBox> = None;
                    let mut idx: usize = 0;
                    loop {
                        let entries = self.realm.collection_entries(handle).unwrap_or_default();
                        let next_pos = match last_key {
                            None => 0,
                            Some(lk) => match entries
                                .iter()
                                .position(|(k, _)| self.realm.same_value_zero(*k, lk))
                            {
                                // Still at (or before) its recorded slot — a pure
                                // delete only shifts survivors left — so advance past.
                                Some(q) if q <= idx => q + 1,
                                // Found only at a *later* slot: the key was deleted and
                                // re-added (a fresh entry at the end). Treat the old
                                // occurrence as deleted, resuming from the recorded slot
                                // (its successor post-compaction); the re-added copy is
                                // reached later as a new entry.
                                Some(_) => idx,
                                // Deleted (and not re-added): resume from recorded slot.
                                None => idx,
                            },
                        };
                        let Some(&(k, v)) = entries.get(next_pos) else {
                            break;
                        };
                        last_key = Some(k);
                        idx = next_pos;
                        // The callback gets `(value, key, collection)` with `thisArg`
                        // (a Set yields its element as both value and key).
                        let val = if is_set { k } else { v };
                        self.call_with_this(f, this_arg, &[val, k, coll])?;
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                // `keys`/`values`/`entries` return a **live** iterator (re-reads the
                // collection on each `next()`), so a mutation mid-iteration is
                // observed and `m.keys().next()` works — not just `for-of`.
                "keys" | "values" | "entries" => {
                    let tag = if self.realm.collection_is_set(handle) == Some(true) {
                        "Set Iterator"
                    } else {
                        "Map Iterator"
                    };
                    let kind = match method {
                        "keys" => 0,
                        "entries" => 2,
                        _ => 1,
                    };
                    return Ok(Some(self.make_live_collection_iterator(handle, kind, tag)));
                }
                // ES2025 Set composition (24.2.4). Each method reads a *Set Record*
                // from its argument via `GetSetRecord` — `size` (a number, not NaN),
                // a callable `has`, and a callable `keys` — and never treats the
                // argument as a bare iterable (a string/array is a TypeError).
                "union"
                | "intersection"
                | "difference"
                | "symmetricDifference"
                | "isSubsetOf"
                | "isSupersetOf"
                | "isDisjointFrom"
                    if self.realm.collection_is_set(handle) == Some(true) =>
                {
                    return Ok(Some(self.set_composition(method, handle, arg(0))?));
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

    /// `CreateHTML(string, tag, attribute, value)` (Annex B.2.3): wraps the WTF-8
    /// receiver `s_bytes` in `<tag …>S</tag>`. When `attribute` is non-empty and
    /// `value` is `Some`, emits `attribute="V"` where `V` is `ToString(value)`
    /// with every `"` replaced by `&quot;`. The receiver bytes pass through
    /// verbatim (so lone surrogates survive).
    /// Conformant *live* iteration for a sparse array's callback / scan methods
    /// (ECMA-262 23.1.3.*): the length is read once, then each index is probed
    /// with `HasProperty` (own-or-inherited, skipping holes) and read with `Get`
    /// at that step — so mutations made by the callback / a getter mid-iteration
    /// are observed. The callback receives `(value, index, O)` with `thisArg`.
    /// `LengthOfArrayLike(O)` — `ToLength(Get(O, "length"))`, clamped to a `usize`
    /// the engine can index (the 2**53-1 spec cap is far beyond a materializable
    /// array, so a tighter cap is applied to keep the operations bounded).
    /// Per-iteration element read for an `Array.prototype` callback / scan method
    /// (`forEach`/`some`/`every`/`reduce`/`reduceRight`/`find*`). ECMA-262 has each
    /// step do `HasProperty(O, k)` then `Get(O, k)` **live**, so a callback (or an
    /// index getter) that mutates the array mid-iteration is observed.
    ///
    /// - A typed-array view reads its live bytes.
    /// - A *real* `Array` reads live through `[[Get]]`/`HasProperty` (`live`).
    /// - A materialized *generic array-like* keeps the pre-read `snapshot`
    ///   (`snapshot_present` = whether that index was present) — the lazy-scan path
    ///   already handles the common generic-array-like case; this preserves the
    ///   existing behaviour for the materialize path.
    ///
    /// Returns `Ok(None)` when `skip_holes` is set and the index is absent (the
    /// hole-skipping methods `forEach`/`some`/`every`/`reduce`/`reduceRight`);
    /// `find*` pass `skip_holes = false` and read an absent index as `undefined`.
    #[allow(clippy::too_many_arguments)]
    fn array_cb_read(
        &mut self,
        handle: Handle,
        i: usize,
        typed: bool,
        live: bool,
        snapshot: &[NanBox],
        snapshot_present: bool,
        skip_holes: bool,
        array_proto_generic: bool,
    ) -> Result<Option<NanBox>, ExecError> {
        if typed {
            // A *generic* `Array.prototype.<m>.call(ta)` reads the view as an
            // ordinary array-like: `HasProperty(O, i)` is false for an index whose
            // resizable buffer shrank below it (or a detached view), so a
            // hole-skipping iterator (`every`/`some`/`forEach`/`reduce`/`find*`)
            // skips it. The *branded* `%TypedArray%.prototype.<m>` (a direct
            // `ta.<m>()`) instead visits every index in `0..len` and reads
            // `undefined` for an out-of-bounds one (no `HasProperty` gate) — so only
            // apply the skip on the generic path.
            if skip_holes && array_proto_generic {
                let cur_len = self.realm.typed_len(handle).unwrap_or(0);
                if i >= cur_len {
                    return Ok(None);
                }
            }
            return Ok(Some(
                self.realm
                    .typed_get(handle, i)
                    .unwrap_or_else(NanBox::undefined),
            ));
        }
        if live {
            let key = alloc::format!("{i}");
            if skip_holes && !self.has_property(handle, &key) {
                return Ok(None);
            }
            return Ok(Some(self.read_member(handle, &key)?));
        }
        if skip_holes && !snapshot_present {
            return Ok(None);
        }
        Ok(Some(snapshot.get(i).copied().unwrap_or_else(NanBox::hole)))
    }

    /// `ArraySpeciesCreate(originalArray, length)` (ECMA-262 23.1.3.13.1): if
    /// `originalArray` is an Array, consult `Get(O,"constructor")` then
    /// `Get(C, @@species)`; an `undefined`/`null` species (or a non-Array
    /// original) builds a plain dense array of `length` holes; a constructor
    /// species is `Construct(S, «length»)`. Returns the result object.
    pub(crate) fn array_species_create(
        &mut self,
        original: Handle,
        length: usize,
    ) -> Result<NanBox, ExecError> {
        // A non-Array original (the generic-array-like concat path) → ArrayCreate.
        // IsArray is proxy-aware (a proxy whose target is an Array *is* an Array, and
        // a revoked proxy throws), so unwrap the proxy chain here. The subsequent
        // `Get(original, "constructor")` also runs through any proxy `get` trap.
        if !self.is_array_unwrap_proxy(NanBox::handle(original.to_raw()))? {
            return self.array_create_holes(length);
        }
        let mut c = self.read_member(original, "constructor")?;
        // Step 6 (cross-realm): if C is a constructor from a *different* realm than
        // the current Realm Record and `SameValue(C, realmC.[[%Array%]])`, then C is
        // set to undefined — so `[].map()` on an array whose `.constructor` was set
        // to another realm's `Array` builds the result with *this* realm's `%Array%`
        // (and never reads that other realm's `@@species`). `cur_realm` is the realm
        // of the executing `Array.prototype.*` method (the current Realm Record).
        if self.is_constructor_value(c)
            && let Some(ch) = c.as_handle().map(Handle::from_raw)
            && let Some(realm_c) = self.get_function_realm(ch)
            && Some(realm_c) != self.cur_realm
            && realm_c < self.created_realms.len()
            && self.created_realms[realm_c]
                .global_this
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|gt| self.realm.get_property(gt, "Array"))
                .and_then(|a| a.as_handle())
                == c.as_handle()
        {
            c = NanBox::undefined();
        }
        // Step 5: "If C is an Object" — read C[@@species]; null → default. Note a
        // primitive that happens to be heap-allocated (string / symbol / bigint) is
        // NOT an Object, so it must fall through to the IsConstructor(C) check (step
        // 7) and throw, rather than being probed for `@@species`.
        if self.is_object_value(c) {
            let ch = Handle::from_raw(c.as_handle().expect("object value is a handle"));
            let species_sym = self.well_known_symbol("species");
            let species_key = self.member_key(species_sym);
            let s = self.read_member(ch, &species_key)?;
            c = if matches!(s.unpack(), Unpacked::Undefined | Unpacked::Null) {
                NanBox::undefined()
            } else {
                s
            };
        }
        // Default Array constructor (or `constructor`/species undefined) → an
        // ordinary array of `length` holes.
        let is_default = matches!(c.unpack(), Unpacked::Undefined)
            || self.current.get("Array").and_then(|v| v.as_handle()) == c.as_handle();
        if is_default {
            return self.array_create_holes(length);
        }
        if !self.is_constructor_value(c) {
            return Err(self.type_error("Array species is not a constructor"));
        }
        self.construct(c, &[NanBox::number(length as f64)])
    }

    /// `ArrayCreate(length)` — a plain array of `length` holes. A `length` above
    /// the uint32 ceiling (2**32-1) is a `RangeError("Invalid array length")`. A
    /// length beyond the dense storage cap is left *sparse* (an empty backing
    /// `Vec` with a logical-length override) rather than materialized, so a valid
    /// but enormous species length (e.g. the default-species result of a method on
    /// a `length === 2**32-1` array) neither aborts nor OOMs.
    fn array_create_holes(&mut self, length: usize) -> Result<NanBox, ExecError> {
        if length as u64 > u64::from(u32::MAX) {
            let m = self.new_str("Invalid array length");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let h = self.realm.new_array(alloc::vec::Vec::new());
        if length > 0 {
            self.realm.set_array_length(h, length);
        }
        Ok(NanBox::handle(h.to_raw()))
    }

    /// `CreateDataPropertyOrThrow(O, ToString(index), value)`: defines a
    /// writable/enumerable/configurable data property at the integer index, and
    /// throws a TypeError if the (possibly exotic / non-configurable target)
    /// `[[DefineOwnProperty]]` reports failure.
    pub(crate) fn create_data_property_or_throw(
        &mut self,
        obj: Handle,
        index: usize,
        value: NanBox,
    ) -> Result<(), ExecError> {
        let desc = self.realm.new_object();
        self.realm.set_property(desc, "value", value);
        self.realm
            .set_property(desc, "writable", NanBox::boolean(true));
        self.realm
            .set_property(desc, "enumerable", NanBox::boolean(true));
        self.realm
            .set_property(desc, "configurable", NanBox::boolean(true));
        let key = alloc::format!("{index}");
        // `reflect = true` returns `Ok(false)` on a failed define (instead of
        // throwing the `Object.defineProperty` TypeError); CreateDataPropertyOrThrow
        // turns that failure into its own TypeError.
        if !self.apply_descriptor(obj, &key, desc, true)? {
            return Err(self.type_error(&alloc::format!(
                "Cannot create property '{index}' on the species result"
            )));
        }
        Ok(())
    }

    /// `Array.prototype.concat(...args)` (ECMA-262 23.1.3.1), receiver-agnostic.
    ///
    /// 1. `O = ToObject(this)`; `A = ArraySpeciesCreate(O, 0)` (reads `O`'s
    ///    `constructor`/`@@species` *before* any element). `n = 0`.
    /// 2. For each `E` in `[O, ...args]`: `IsConcatSpreadable(E)` — non-object →
    ///    false; else `Get(E, @@isConcatSpreadable)` (a getter fires); if defined,
    ///    `ToBoolean`; else `IsArray(E)` (unwrapping proxies, rejecting functions).
    ///    - Spreadable: `len = ToLength(Get(E, "length"))`; for each `k in 0..len`,
    ///      `HasProperty(E, k)` ? `CreateDataProperty(A, n, Get(E, k))` : leave a
    ///      hole; `n++`. `n + len > 2^53-1` → TypeError.
    ///    - Else: `n >= 2^53-1` → TypeError; `CreateDataProperty(A, n, E)`; `n++`.
    /// 3. `Set(A, "length", n)`; return `A`.
    ///
    /// Both the receiver and every argument are treated uniformly as `items`, so a
    /// non-array array-like `this` (`Array.prototype.concat.call(obj)`) and a boxed
    /// primitive `this` (`.call(101)` → a `Number` wrapper added as one element)
    /// behave per spec.
    /// `FlattenIntoArray(target, source, sourceLen, start, depth [, mapper, thisArg])`
    /// (ECMA-262 23.1.3.13.1) run *live* against a generic array-like `source`: for
    /// each index `HasProperty(source, P)` then `Get(source, P)` (a proxy observes
    /// each trap in spec order/count), optionally mapped, recursing one level into a
    /// nested array (proxy-aware `IsArray`) while `depth > 0`. Returns the next
    /// `targetIndex`.
    #[allow(clippy::too_many_arguments)]
    fn flatten_into_array(
        &mut self,
        target: Handle,
        source: Handle,
        source_len: usize,
        start: usize,
        depth: i32,
        mapper: Option<NanBox>,
        this_arg: NanBox,
    ) -> Result<usize, ExecError> {
        const MAX_SAFE: usize = 9_007_199_254_740_991; // 2^53 - 1
        let mut target_index = start;
        for source_index in 0..source_len {
            let key = alloc::format!("{source_index}");
            if !self.has_property(source, &key) {
                continue;
            }
            let mut element = self.read_member(source, &key)?;
            if let Some(m) = mapper {
                element = self.call_with_this(
                    m,
                    this_arg,
                    &[
                        element,
                        NanBox::number(source_index as f64),
                        NanBox::handle(source.to_raw()),
                    ],
                )?;
            }
            let should_flatten = depth > 0 && self.is_array_unwrap_proxy(element)?;
            if should_flatten {
                let element_h = Handle::from_raw(element.as_handle().expect("array is an object"));
                let element_len = self.array_like_length(element_h)?;
                // Nested flattening never carries the mapper (per spec step iv.2).
                target_index = self.flatten_into_array(
                    target,
                    element_h,
                    element_len,
                    target_index,
                    depth - 1,
                    None,
                    NanBox::undefined(),
                )?;
            } else {
                if target_index >= MAX_SAFE {
                    return Err(self.type_error("flatten result exceeds maximum array length"));
                }
                self.create_data_property_or_throw(target, target_index, element)?;
                target_index += 1;
            }
        }
        Ok(target_index)
    }

    /// `Array.prototype.map`/`filter` over a *plain* (non-array, non-proxy-of-array)
    /// generic array-like: `LengthOfArrayLike`, then `IsCallable(callbackfn)`, then
    /// `ArraySpeciesCreate` (an out-of-range length throws before any callback), then
    /// per index `HasProperty` → `Get` → callback, reading each element *live* so a
    /// callback that mutates a later index is observed.
    fn array_map_filter_generic(
        &mut self,
        method: &str,
        handle: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let f = args.first().copied().unwrap_or_else(NanBox::undefined);
        let this_arg = args.get(1).copied().unwrap_or_else(NanBox::undefined);
        let len = self.array_like_length(handle)?;
        self.require_callable(f, &alloc::format!("{method} callback"))?;
        let is_map = method == "map";
        let a_v = self.array_species_create(handle, if is_map { len } else { 0 })?;
        let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Array species did not return an object"));
        };
        let o = NanBox::handle(handle.to_raw());
        let mut to = 0usize;
        for i in 0..len {
            let key = alloc::format!("{i}");
            if !self.has_property(handle, &key) {
                continue;
            }
            let kv = self.read_member(handle, &key)?;
            let mapped = self.call_with_this(f, this_arg, &[kv, NanBox::number(i as f64), o])?;
            if is_map {
                self.create_data_property_or_throw(a_h, i, mapped)?;
            } else if self.realm.truthy(mapped) {
                self.create_data_property_or_throw(a_h, to, kv)?;
                to += 1;
            }
        }
        Ok(a_v)
    }

    pub(crate) fn array_concat(
        &mut self,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        const MAX_SAFE: f64 = 9_007_199_254_740_991.0; // 2^53 - 1
        // ToObject(this): a primitive receiver is boxed in its wrapper (added as a
        // single non-spreadable element); an object is used as-is.
        let o_v = self.coerce_to_object(this);
        let Some(o_h) = o_v.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Array.prototype.concat called on a non-object"));
        };
        // ArraySpeciesCreate(O, 0) — must run (reading `O.constructor`) before the
        // `@@isConcatSpreadable` lookups, per spec step order.
        let a_v = self.array_species_create(o_h, 0)?;
        let Some(a_h) = a_v.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Array species did not return an object"));
        };
        let sym = self.well_known_symbol("isConcatSpreadable");
        let spread_key = self.member_key(sym);
        // `n` (the running result length) is tracked as an `f64` so the spec's
        // 2^53-1 cap is meaningful even though no such array is materializable.
        let mut n: f64 = 0.0;
        // Items = [O, ...args].
        for item in core::iter::once(o_v).chain(args.iter().copied()) {
            let ih = item.as_handle().map(Handle::from_raw);
            // IsConcatSpreadable(E): "If Type(E) is not Object, return false" — so a
            // *primitive* string/symbol/bigint (a heap value, but not an Object) is
            // never spread even when `@@isConcatSpreadable` is inherited from a
            // primitive prototype (e.g. `String.prototype`). Otherwise a defined
            // `@@isConcatSpreadable` (read via the accessor path so a getter fires /
            // may throw) decides via ToBoolean, else `IsArray` (which unwraps
            // proxies and throws on a revoked one).
            let spread = match ih {
                Some(h) if self.is_object_value(item) => {
                    let v = self.read_member(h, &spread_key)?;
                    if matches!(v.unpack(), Unpacked::Undefined) {
                        self.is_array_unwrap_proxy(item)?
                    } else {
                        self.realm.truthy(v)
                    }
                }
                _ => false,
            };
            if spread {
                let h = ih.expect("spreadable implies an object");
                // len = ToLength(Get(E, "length")): the getter fires (abrupt
                // completion propagates) and the value is coerced through
                // `valueOf`/`toString`; NaN/negative clamps to 0, capped at 2^53-1.
                let len_val = self.read_member(h, "length")?;
                let len_num = self.coerce_to_number(len_val)?;
                let raw = self.realm.to_number(len_num);
                // ToLength: truncate toward zero (`as u64` on an already-clamped,
                // finite value avoids the std-only `f64::trunc`).
                let len_int = if raw.is_nan() || raw <= 0.0 {
                    0.0
                } else {
                    raw.min(MAX_SAFE) as u64 as f64
                };
                if n + len_int > MAX_SAFE {
                    return Err(self
                        .type_error("Array.prototype.concat result exceeds maximum array length"));
                }
                let len = len_int as usize;
                for k in 0..len {
                    let key = alloc::format!("{k}");
                    // Only define a result element when the source HasProperty(k);
                    // an absent index leaves a hole in `A` (but `n` still advances).
                    // `Get`/`HasProperty` walk the prototype chain and fire getters.
                    if self.has_property(h, &key) {
                        let v = self.read_member(h, &key)?;
                        self.create_data_property_or_throw(a_h, n as usize, v)?;
                    }
                    n += 1.0;
                }
            } else {
                // Non-spreadable: added as a single element (a `2^53-1` overflow is
                // a TypeError).
                if n >= MAX_SAFE {
                    return Err(self
                        .type_error("Array.prototype.concat result exceeds maximum array length"));
                }
                self.create_data_property_or_throw(a_h, n as usize, item)?;
                n += 1.0;
            }
        }
        // Set(A, "length", n) — `CreateDataProperty` already grew it, but the spec
        // sets it explicitly (and a custom species may not auto-track length).
        let len_key = self.new_str("length");
        self.assign_member_value(a_h, len_key, NanBox::number(n))?;
        Ok(a_v)
    }

    pub(crate) fn array_like_length(&mut self, handle: Handle) -> Result<usize, ExecError> {
        let len_val = self.read_member(handle, "length")?;
        let len_num = self.coerce_to_number(len_val)?;
        let raw = self.realm.to_number(len_num);
        let len = if raw.is_nan() || raw <= 0.0 {
            0.0
        } else {
            raw.min(9_007_199_254_740_991.0)
        };
        Ok(len as usize)
    }

    fn array_like_set_length(&mut self, handle: Handle, len: usize) -> Result<(), ExecError> {
        let key = self.new_str("length");
        self.assign_member_value(handle, key, NanBox::number(len as f64))
    }

    fn array_like_get(&mut self, handle: Handle, i: usize) -> Result<NanBox, ExecError> {
        self.read_member(handle, &alloc::format!("{i}"))
    }

    fn array_like_set(&mut self, handle: Handle, i: usize, v: NanBox) -> Result<(), ExecError> {
        let key = self.new_str(&alloc::format!("{i}"));
        self.assign_member_value(handle, key, v)
    }

    fn array_like_delete(&mut self, handle: Handle, i: usize) -> Result<(), ExecError> {
        // The generic `Array.prototype` mutators delete via `DeletePropertyOrThrow`
        // (ECMA-262 23.1.3): a failed `[[Delete]]` (a non-configurable index) is a
        // TypeError, not a silent drop.
        let key = alloc::format!("{i}");
        if !self.delete_property_of(handle, &key)? {
            return Err(self.type_error(&alloc::format!("Cannot delete property '{key}'")));
        }
        Ok(())
    }

    /// Moves the element from `from` to `to` if present, else deletes `to`
    /// (`CreateDataPropertyOrThrow`/`DeletePropertyOrThrow` per the spec's
    /// hole-preserving copy loops in `unshift`/`splice`/etc.).
    fn array_like_move(&mut self, handle: Handle, from: usize, to: usize) -> Result<(), ExecError> {
        if self.has_property(handle, &alloc::format!("{from}")) {
            let v = self.array_like_get(handle, from)?;
            self.array_like_set(handle, to, v)?;
        } else {
            self.array_like_delete(handle, to)?;
        }
        Ok(())
    }

    /// Resolves a spec "relative index" argument against `len`: ToIntegerOrInfinity,
    /// then a negative value counts from the end (clamped to 0) and a positive value
    /// is clamped to `len`. `default` is used when the argument is `undefined`.
    fn relative_index(
        &mut self,
        v: NanBox,
        len: usize,
        default: usize,
    ) -> Result<usize, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(default);
        }
        let n = self.coerce_to_integer_or_infinity(v)?;
        let len_f = len as f64;
        let idx = if n < 0.0 {
            (len_f + n).max(0.0)
        } else {
            n.min(len_f)
        };
        Ok(idx as usize)
    }

    /// The *mutating* generic `Array.prototype` methods over a non-array array-like
    /// `O`: each operates by `Get`/`Set`/`Delete` on integer-index keys and the
    /// `length` property (ECMA-262 23.1.3 — these are "intentionally generic").
    ///
    /// Every `Set`/`Delete` these methods perform carries `Throw=true` (they use
    /// `Set(O, P, V, true)` / `DeletePropertyOrThrow`), independent of the caller's
    /// strict mode — e.g. `Array.prototype.push.call("")` must throw because a
    /// String exotic's `length` is non-writable. `assign_member_value` /
    /// `delete_property_of` key their throw-on-failure off `self.strict`, so force
    /// a strict context for the duration of the (self-contained) mutation. A user
    /// setter invoked mid-operation runs with its own function strictness, so this
    /// only governs the failed-[[Set]]/[[Delete]] throw here.
    fn array_like_mutate(
        &mut self,
        method: &str,
        handle: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let saved_strict = self.strict;
        self.strict = true;
        let r = self.array_like_mutate_impl(method, handle, args);
        self.strict = saved_strict;
        r
    }

    fn array_like_mutate_impl(
        &mut self,
        method: &str,
        handle: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        // `Array.prototype.sort`: IsCallable(comparefn) is validated *before* the
        // `length` read (23.1.3.30 step 1), so a bad comparefn throws even when the
        // receiver's `length` getter would throw.
        if method == "sort"
            && !matches!(arg(0).unpack(), Unpacked::Undefined)
            && !self.is_callable_value(arg(0))
        {
            return Err(self.type_error("comparefn must be a function"));
        }
        let len = self.array_like_length(handle)?;
        match method {
            "push" => {
                // `len + argCount` may not exceed 2**53-1 (SetLength would fail);
                // the check happens before any element is stored.
                if (len as f64) + (args.len() as f64) > 9_007_199_254_740_991.0 {
                    let m = self.new_str("Invalid array length");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // Set each argument at the next index, then update length.
                let mut n = len;
                for a in args {
                    self.array_like_set(handle, n, *a)?;
                    n += 1;
                }
                self.array_like_set_length(handle, n)?;
                Ok(NanBox::number(n as f64))
            }
            "pop" => {
                if len == 0 {
                    self.array_like_set_length(handle, 0)?;
                    return Ok(NanBox::undefined());
                }
                let v = self.array_like_get(handle, len - 1)?;
                self.array_like_delete(handle, len - 1)?;
                self.array_like_set_length(handle, len - 1)?;
                Ok(v)
            }
            "shift" => {
                if len == 0 {
                    self.array_like_set_length(handle, 0)?;
                    return Ok(NanBox::undefined());
                }
                let first = self.array_like_get(handle, 0)?;
                for to in 1..len {
                    self.array_like_move(handle, to, to - 1)?;
                }
                self.array_like_delete(handle, len - 1)?;
                self.array_like_set_length(handle, len - 1)?;
                Ok(first)
            }
            "unshift" => {
                let count = args.len();
                // `len + argCount` may not exceed 2**53-1 (checked before moving).
                if count > 0 && (len as f64) + (count as f64) > 9_007_199_254_740_991.0 {
                    let m = self.new_str("Invalid array length");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                if count > 0 {
                    // Shift existing elements up by `count`, high index first.
                    for k in (0..len).rev() {
                        self.array_like_move(handle, k, k + count)?;
                    }
                    for (j, a) in args.iter().enumerate() {
                        self.array_like_set(handle, j, *a)?;
                    }
                }
                let new_len = len + count;
                self.array_like_set_length(handle, new_len)?;
                Ok(NanBox::number(new_len as f64))
            }
            "reverse" => {
                let mid = len / 2;
                for lower in 0..mid {
                    let upper = len - lower - 1;
                    // Spec order per pair (23.1.3.26): HasProperty(lower), then
                    // Get(lower) *only if present*, then HasProperty(upper), then
                    // Get(upper) *only if present* — so an absent index is never read
                    // and a throwing getter fires at the exact point a proxy expects.
                    let lower_exists = self.has_property(handle, &alloc::format!("{lower}"));
                    let lower_val = if lower_exists {
                        Some(self.array_like_get(handle, lower)?)
                    } else {
                        None
                    };
                    let upper_exists = self.has_property(handle, &alloc::format!("{upper}"));
                    let upper_val = if upper_exists {
                        Some(self.array_like_get(handle, upper)?)
                    } else {
                        None
                    };
                    match (lower_val, upper_val) {
                        (Some(lv), Some(uv)) => {
                            self.array_like_set(handle, lower, uv)?;
                            self.array_like_set(handle, upper, lv)?;
                        }
                        (None, Some(uv)) => {
                            self.array_like_set(handle, lower, uv)?;
                            self.array_like_delete(handle, upper)?;
                        }
                        (Some(lv), None) => {
                            self.array_like_delete(handle, lower)?;
                            self.array_like_set(handle, upper, lv)?;
                        }
                        (None, None) => {}
                    }
                }
                Ok(NanBox::handle(handle.to_raw()))
            }
            "fill" => {
                let value = arg(0);
                let start = self.relative_index(arg(1), len, 0)?;
                let end = self.relative_index(arg(2), len, len)?;
                for k in start..end {
                    self.array_like_set(handle, k, value)?;
                }
                Ok(NanBox::handle(handle.to_raw()))
            }
            "copyWithin" => {
                let to = self.relative_index(arg(0), len, 0)?;
                let from = self.relative_index(arg(1), len, 0)?;
                let fin = self.relative_index(arg(2), len, len)?;
                let count = fin.saturating_sub(from).min(len.saturating_sub(to));
                // Copy direction matters when ranges overlap (spec uses a direction
                // flag); collect-then-write avoids clobbering source elements.
                let mut buf: Vec<Option<NanBox>> = Vec::with_capacity(count);
                for k in 0..count {
                    let idx = from + k;
                    // `fromPresent = HasProperty(O, fromKey)` then ReturnIfAbrupt — a
                    // throwing `has` trap propagates (must not collapse to `false`).
                    if self.has_property_proxied(handle, &alloc::format!("{idx}"))? {
                        buf.push(Some(self.array_like_get(handle, idx)?));
                    } else {
                        buf.push(None);
                    }
                }
                for (k, slot) in buf.into_iter().enumerate() {
                    let dst = to + k;
                    match slot {
                        Some(v) => self.array_like_set(handle, dst, v)?,
                        None => self.array_like_delete(handle, dst)?,
                    }
                }
                Ok(NanBox::handle(handle.to_raw()))
            }
            "sort" => {
                // `comparefn` must be undefined or callable (checked first).
                let cmp = arg(0);
                if !matches!(cmp.unpack(), Unpacked::Undefined) && !self.is_callable_value(cmp) {
                    return Err(self.type_error("comparefn must be a function"));
                }
                // SortIndexedProperties: gather the *present* elements (holes and
                // absent indices excluded), sort them, then write the sorted run
                // back to `0..count` and delete `count..len` (holes float to the end).
                let mut items: Vec<NanBox> = Vec::new();
                for i in 0..len {
                    if self.has_property(handle, &alloc::format!("{i}")) {
                        items.push(self.array_like_get(handle, i)?);
                    }
                }
                let count = items.len();
                let sorted = self.sort_array(items, cmp, false)?;
                for (i, v) in sorted.into_iter().enumerate() {
                    self.array_like_set(handle, i, v)?;
                }
                for i in count..len {
                    self.array_like_delete(handle, i)?;
                }
                Ok(NanBox::handle(handle.to_raw()))
            }
            "splice" => {
                let start = self.relative_index(arg(0), len, 0)?;
                let insert_count = args.len().saturating_sub(2);
                let delete_count = if args.is_empty() {
                    0
                } else if args.len() == 1 {
                    len - start
                } else {
                    let dc = self.coerce_to_integer_or_infinity(arg(1))?;
                    (dc.max(0.0) as usize).min(len - start)
                };
                // Step 8: if the resulting length `len + insertCount - deleteCount`
                // would exceed 2**53-1, throw a TypeError — *before* creating the
                // result array or moving any element (this also bounds the work).
                if (len as f64) + (insert_count as f64) - (delete_count as f64)
                    > 9_007_199_254_740_991.0
                {
                    let m = self.new_str("Invalid array length");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // `A = ArraySpeciesCreate(O, actualDeleteCount)` — for a *proxy* whose
                // target is an Array, `IsArray(O)` is true, so this reads
                // `O.constructor`/`@@species` (through the proxy's traps) and may build
                // an exotic result. Collect the removed elements via
                // `CreateDataPropertyOrThrow` (holes preserved by skipping absent keys).
                let removed_arr_v = self.array_species_create(handle, delete_count)?;
                let Some(removed_arr) = removed_arr_v.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("Array species did not return an object"));
                };
                let default_removed = self.realm.is_array(removed_arr)
                    && self.realm.array_length(removed_arr) == Some(delete_count)
                    && !self.realm.array_has_index_overrides(removed_arr);
                for k in 0..delete_count {
                    let idx = start + k;
                    if self.has_property(handle, &alloc::format!("{idx}")) {
                        let v = self.array_like_get(handle, idx)?;
                        if default_removed {
                            self.realm.set_element(removed_arr, k, v);
                        } else {
                            self.create_data_property_or_throw(removed_arr, k, v)?;
                        }
                    }
                }
                let rem_len_key = self.new_str("length");
                self.assign_member_value(
                    removed_arr,
                    rem_len_key,
                    NanBox::number(delete_count as f64),
                )?;
                // Shift the tail to its new position.
                if insert_count < delete_count {
                    for k in start..(len - delete_count) {
                        self.array_like_move(handle, k + delete_count, k + insert_count)?;
                    }
                    for k in ((len - delete_count + insert_count)..len).rev() {
                        self.array_like_delete(handle, k)?;
                    }
                } else if insert_count > delete_count {
                    for k in (start..(len - delete_count)).rev() {
                        self.array_like_move(handle, k + delete_count, k + insert_count)?;
                    }
                }
                // Write the inserted items.
                for (j, item) in args.iter().skip(2).enumerate() {
                    self.array_like_set(handle, start + j, *item)?;
                }
                self.array_like_set_length(handle, len - delete_count + insert_count)?;
                Ok(removed_arr_v)
            }
            _ => Ok(NanBox::undefined()),
        }
    }

    fn array_iter_sparse(
        &mut self,
        method: &str,
        handle: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        let f = arg(0);
        let this_arg = arg(1);
        let o = NanBox::handle(handle.to_raw());
        // `LengthOfArrayLike(O)` — works for a real array *and* a generic
        // array-like (a huge `{length:"Infinity"}` scans lazily with early exit
        // rather than materializing).
        let len = self.array_like_length(handle)?;
        // `indexOf`/`lastIndexOf` take a search target + optional fromIndex; the
        // rest require a callable callback (already validated upstream, but a
        // sparse path is reached after that check).
        match method {
            "indexOf" => {
                let target = arg(0);
                let from = self.array_from_index_checked(arg(1), len)?;
                for i in from..len {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        if self.realm.strict_equals(v, target) {
                            return Ok(NanBox::number(i as f64));
                        }
                    }
                }
                return Ok(NanBox::number(-1.0));
            }
            "lastIndexOf" => {
                let target = arg(0);
                if len == 0 {
                    return Ok(NanBox::number(-1.0));
                }
                let from = if args.len() >= 2 {
                    let n = self.coerce_to_integer_or_infinity(arg(1))?;
                    if n < 0.0 {
                        (len as f64 + n) as i64
                    } else {
                        (n as i64).min(len as i64 - 1)
                    }
                } else {
                    len as i64 - 1
                };
                let mut i = from;
                while i >= 0 {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        if self.realm.strict_equals(v, target) {
                            return Ok(NanBox::number(i as f64));
                        }
                    }
                    i -= 1;
                }
                return Ok(NanBox::number(-1.0));
            }
            "includes" => {
                // SameValueZero (NaN matches NaN); an absent index reads as
                // `undefined` and is *not* skipped — `[,].includes(undefined)` is
                // true. `Get` walks the prototype for an absent own index.
                let target = arg(0);
                let from = self.array_from_index_checked(arg(1), len)?;
                let t_nan = target.as_number().is_some_and(f64::is_nan);
                for i in from..len {
                    let v = self.read_member(handle, &alloc::format!("{i}"))?;
                    if self.realm.strict_equals(v, target)
                        || (t_nan && v.as_number().is_some_and(f64::is_nan))
                    {
                        return Ok(NanBox::boolean(true));
                    }
                }
                return Ok(NanBox::boolean(false));
            }
            _ => {}
        }
        self.require_callable(f, &alloc::format!("{method} callback"))?;
        match method {
            "forEach" => {
                for i in 0..len {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        self.call_with_this(f, this_arg, &[v, NanBox::number(i as f64), o])?;
                    }
                }
                Ok(NanBox::undefined())
            }
            "map" => {
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        out.push(self.call_with_this(
                            f,
                            this_arg,
                            &[v, NanBox::number(i as f64), o],
                        )?);
                    } else {
                        out.push(NanBox::hole());
                    }
                }
                Ok(NanBox::handle(self.realm.new_array(out).to_raw()))
            }
            "filter" => {
                let mut out = Vec::new();
                for i in 0..len {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        let r =
                            self.call_with_this(f, this_arg, &[v, NanBox::number(i as f64), o])?;
                        if self.realm.truthy(r) {
                            out.push(v);
                        }
                    }
                }
                Ok(NanBox::handle(self.realm.new_array(out).to_raw()))
            }
            "some" => {
                for i in 0..len {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        let r =
                            self.call_with_this(f, this_arg, &[v, NanBox::number(i as f64), o])?;
                        if self.realm.truthy(r) {
                            return Ok(NanBox::boolean(true));
                        }
                    }
                }
                Ok(NanBox::boolean(false))
            }
            "every" => {
                for i in 0..len {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        let r =
                            self.call_with_this(f, this_arg, &[v, NanBox::number(i as f64), o])?;
                        if !self.realm.truthy(r) {
                            return Ok(NanBox::boolean(false));
                        }
                    }
                }
                Ok(NanBox::boolean(true))
            }
            // `find`/`findIndex`/`findLast`/`findLastIndex` do NOT skip holes — they
            // visit every index `[0,len)` (or reverse), reading a hole as undefined.
            "find" | "findIndex" => {
                for i in 0..len {
                    let v = self.read_member(handle, &alloc::format!("{i}"))?;
                    let r = self.call_with_this(f, this_arg, &[v, NanBox::number(i as f64), o])?;
                    if self.realm.truthy(r) {
                        return Ok(if method == "find" {
                            v
                        } else {
                            NanBox::number(i as f64)
                        });
                    }
                }
                Ok(if method == "find" {
                    NanBox::undefined()
                } else {
                    NanBox::number(-1.0)
                })
            }
            "findLast" | "findLastIndex" => {
                let mut i = len as i64 - 1;
                while i >= 0 {
                    let v = self.read_member(handle, &alloc::format!("{i}"))?;
                    let r = self.call_with_this(f, this_arg, &[v, NanBox::number(i as f64), o])?;
                    if self.realm.truthy(r) {
                        return Ok(if method == "findLast" {
                            v
                        } else {
                            NanBox::number(i as f64)
                        });
                    }
                    i -= 1;
                }
                Ok(if method == "findLast" {
                    NanBox::undefined()
                } else {
                    NanBox::number(-1.0)
                })
            }
            "reduce" => {
                let mut acc;
                let mut start = 0usize;
                if args.len() >= 2 {
                    acc = arg(1);
                } else {
                    let mut seed = None;
                    while start < len {
                        let key = alloc::format!("{start}");
                        if self.has_property(handle, &key) {
                            seed = Some(self.read_member(handle, &key)?);
                            start += 1;
                            break;
                        }
                        start += 1;
                    }
                    match seed {
                        Some(s) => acc = s,
                        None => {
                            let m = self.new_str("Reduce of empty array with no initial value");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                    }
                }
                for i in start..len {
                    let key = alloc::format!("{i}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        acc = self.call(f, &[acc, v, NanBox::number(i as f64), o])?;
                    }
                }
                Ok(acc)
            }
            "reduceRight" => {
                let mut acc;
                let mut idx = len as i64 - 1;
                if args.len() >= 2 {
                    acc = arg(1);
                } else {
                    let mut seed = None;
                    while idx >= 0 {
                        let key = alloc::format!("{idx}");
                        if self.has_property(handle, &key) {
                            seed = Some(self.read_member(handle, &key)?);
                            idx -= 1;
                            break;
                        }
                        idx -= 1;
                    }
                    match seed {
                        Some(s) => acc = s,
                        None => {
                            let m = self.new_str("Reduce of empty array with no initial value");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                    }
                }
                while idx >= 0 {
                    let key = alloc::format!("{idx}");
                    if self.has_property(handle, &key) {
                        let v = self.read_member(handle, &key)?;
                        acc = self.call(f, &[acc, v, NanBox::number(idx as f64), o])?;
                    }
                    idx -= 1;
                }
                Ok(acc)
            }
            _ => Ok(NanBox::undefined()),
        }
    }

    /// `GetSetRecord(obj)` (ECMA-262 24.2.1.2): validates `obj` is a set-like —
    /// an Object with a numeric, non-NaN `size`, a callable `has`, and a callable
    /// `keys` — returning `(obj, intSize, has, keys)`. A non-object, a NaN `size`,
    /// or a non-callable `has`/`keys` is a TypeError.
    fn get_set_record(
        &mut self,
        other: NanBox,
    ) -> Result<(Handle, f64, NanBox, NanBox), ExecError> {
        let Some(obj) = other
            .as_handle()
            .map(Handle::from_raw)
            .filter(|_| self.is_object_value(other))
        else {
            return Err(self.type_error("Set method argument must be an object"));
        };
        // Get(obj, "size") → ToNumber; a NaN (incl. `undefined`) is a TypeError.
        let size_raw = self.read_member(obj, "size")?;
        let size_num = self.coerce_to_number(size_raw)?;
        let size = self.realm.to_number(size_num);
        if size.is_nan() {
            return Err(self.type_error("set-like 'size' must not be NaN"));
        }
        // intSize = ToIntegerOrInfinity(size), but a negative size is a RangeError.
        // (`trunc_toward_zero` is the no_std-safe `f64::trunc`; `size` is finite or
        // ±Infinity here, never NaN.)
        if size < 0.0 {
            let m = self.new_str("set-like 'size' must not be negative");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let int_size = if size.is_infinite() {
            size
        } else {
            trunc_toward_zero(size)
        };
        let has = self.read_member(obj, "has")?;
        self.require_callable(has, "set-like 'has'")?;
        let keys = self.read_member(obj, "keys")?;
        self.require_callable(keys, "set-like 'keys'")?;
        Ok((obj, int_size, has, keys))
    }

    /// Drives the set-like record's `keys()` iterator to completion, returning
    /// every yielded value (used by the composition methods that must iterate the
    /// argument rather than probe it with `has`). `-0` is canonicalized to `+0`.
    /// Opens the set-like record's `keys()` iterator, returning `(iterator, next)`
    /// so a caller can drive it **lazily** one step at a time (via
    /// [`Self::iter_step`]) and short-circuit + [`Self::iterator_close`] — required
    /// by the composition methods that may finish before draining the argument
    /// (`isSupersetOf`, `isDisjointFrom`).
    fn open_set_keys(&mut self, obj: Handle, keys: NanBox) -> Result<(Handle, NanBox), ExecError> {
        let iter = self.call_with_this(keys, NanBox::handle(obj.to_raw()), &[])?;
        let Some(ih) = iter.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("set-like 'keys' did not return an iterator"));
        };
        let next = self.read_member(ih, "next")?;
        Ok((ih, next))
    }

    fn set_record_keys(&mut self, obj: Handle, keys: NanBox) -> Result<Vec<NanBox>, ExecError> {
        let (ih, next) = self.open_set_keys(obj, keys)?;
        let mut out = Vec::new();
        while let Some(v) = self.iter_step(ih, next)? {
            // CanonicalizeKeyedCollectionKey: `-0` is stored/compared as `+0`.
            let v = if v.as_number() == Some(0.0) {
                NanBox::number(0.0)
            } else {
                v
            };
            out.push(v);
        }
        Ok(out)
    }

    /// `String.prototype.{match,matchAll,search,replace,replaceAll,split}` delegate
    /// to a `searchValue`/`separator` object's `@@match`/`@@replace`/… method when
    /// present: `Return ? Call(replacer, searchValue, « O, replaceValue »)`. `o` is
    /// the `this` value (`RequireObjectCoercible(this)`) passed as the method's
    /// first argument — a primitive string for a primitive receiver, or the wrapper
    /// **object** for a `new String(...)` receiver (so `O` keeps its identity).
    /// Returns `Some(result)` when the delegation fired, `None` to fall through to
    /// the literal-string handling.
    fn string_symbol_delegate(
        &mut self,
        o: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let sym_name = match method {
            "match" => "match",
            "matchAll" => "matchAll",
            "search" => "search",
            "replace" | "replaceAll" => "replace",
            "split" => "split",
            _ => return Ok(None),
        };
        let search = args.first().copied().unwrap_or_else(NanBox::undefined);
        if !self.is_object_value(search) {
            return Ok(None);
        }
        let argh = Handle::from_raw(search.as_handle().unwrap());
        // `replaceAll`/`matchAll` first require that a RegExp `searchValue` be
        // global (`IsRegExp` + `Get(flags)` not containing "g" → TypeError),
        // checked *before* dispatching the symbol method.
        if matches!(method, "replaceAll" | "matchAll") && self.is_regexp_arg(search) {
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
            let mut call_args = alloc::vec![o];
            call_args.extend_from_slice(&args[1.min(args.len())..]);
            return Ok(Some(self.call_with_this(m, search, &call_args)?));
        }
        Ok(None)
    }

    /// One step of a **live**, delete-tolerant cursor over collection `coll`'s
    /// keys (used by the Set-composition methods whose argument callback may
    /// mutate `this` mid-iteration). `last` is the previously-yielded
    /// `(key, index)`; returns the next `(key, index)` or `None` at the end.
    /// Mirrors the `forEach`/live-iterator resume logic: a pure delete only shifts
    /// survivors left, so a key found *past* its recorded slot was deleted and
    /// re-added (skip it), and a missing key resumes from the recorded slot.
    fn collection_live_step(
        &self,
        coll: Handle,
        last: Option<(NanBox, usize)>,
    ) -> Option<(NanBox, usize)> {
        let entries = self.realm.collection_entries(coll).unwrap_or_default();
        let next_pos = match last {
            None => 0,
            Some((lk, idx)) => match entries
                .iter()
                .position(|(k, _)| self.realm.same_value_zero(*k, lk))
            {
                Some(q) if q <= idx => q + 1,
                Some(_) => idx,
                None => idx,
            },
        };
        entries.get(next_pos).map(|(k, _)| (*k, next_pos))
    }

    /// The ES2025 Set composition methods over a `GetSetRecord` argument.
    fn set_composition(
        &mut self,
        method: &str,
        handle: Handle,
        other: NanBox,
    ) -> Result<NanBox, ExecError> {
        let (obj, other_size, has, keys) = self.get_set_record(other)?;
        let mine: Vec<NanBox> = self
            .realm
            .collection_entries(handle)
            .unwrap_or_default()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        let my_size = mine.len() as f64;
        // `other.has(v)` (with `obj` as `this`), coerced to a boolean.
        let other_has = |this: &mut Self, v: NanBox| -> Result<bool, ExecError> {
            let r = this.call_with_this(has, NanBox::handle(obj.to_raw()), &[v])?;
            Ok(this.realm.truthy(r))
        };
        let in_mine =
            |this: &Self, v: NanBox| mine.iter().any(|m| this.realm.same_value_zero(*m, v));

        match method {
            "isSubsetOf" => {
                if my_size > other_size {
                    return Ok(NanBox::boolean(false));
                }
                // Iterate `this` LIVE (not the `mine` snapshot): `other.has` may
                // delete elements of `this`, and a deleted element must not be
                // passed to `has` (ECMA-262 iterates `O.[[SetData]]` by index,
                // skipping now-empty slots).
                let mut last = None;
                while let Some((e, idx)) = self.collection_live_step(handle, last) {
                    last = Some((e, idx));
                    if !other_has(self, e)? {
                        return Ok(NanBox::boolean(false));
                    }
                }
                Ok(NanBox::boolean(true))
            }
            "isSupersetOf" => {
                if my_size < other_size {
                    return Ok(NanBox::boolean(false));
                }
                // Iterate the argument's keys LAZILY: return false (and
                // IteratorClose) as soon as a key is not in this set — the rest of
                // the iterator must not be drained.
                let (ih, next) = self.open_set_keys(obj, keys)?;
                while let Some(k) = self.iter_step(ih, next)? {
                    let k = if k.as_number() == Some(0.0) {
                        NanBox::number(0.0)
                    } else {
                        k
                    };
                    if !in_mine(self, k) {
                        self.iterator_close(ih)?;
                        return Ok(NanBox::boolean(false));
                    }
                }
                Ok(NanBox::boolean(true))
            }
            "isDisjointFrom" => {
                if my_size <= other_size {
                    // Iterate `this` LIVE: `other.has` may delete elements of
                    // `this`, and a since-deleted element must not be passed to
                    // `has` (see `isSubsetOf`).
                    let mut last = None;
                    while let Some((e, idx)) = self.collection_live_step(handle, last) {
                        last = Some((e, idx));
                        if other_has(self, e)? {
                            return Ok(NanBox::boolean(false));
                        }
                    }
                } else {
                    // Lazily iterate the argument's keys: return false (and
                    // IteratorClose) on the first key found in this set.
                    let (ih, next) = self.open_set_keys(obj, keys)?;
                    while let Some(k) = self.iter_step(ih, next)? {
                        let k = if k.as_number() == Some(0.0) {
                            NanBox::number(0.0)
                        } else {
                            k
                        };
                        if in_mine(self, k) {
                            self.iterator_close(ih)?;
                            return Ok(NanBox::boolean(false));
                        }
                    }
                }
                Ok(NanBox::boolean(true))
            }
            "union" => {
                let result = self.realm.new_collection(true);
                for m in &mine {
                    self.realm.collection_set(result, *m, *m);
                }
                for k in self.set_record_keys(obj, keys)? {
                    self.realm.collection_set(result, k, k);
                }
                Ok(NanBox::handle(result.to_raw()))
            }
            "intersection" => {
                let result = self.realm.new_collection(true);
                if my_size <= other_size {
                    for m in &mine {
                        if other_has(self, *m)? {
                            self.realm.collection_set(result, *m, *m);
                        }
                    }
                } else {
                    for k in self.set_record_keys(obj, keys)? {
                        if in_mine(self, k) {
                            self.realm.collection_set(result, k, k);
                        }
                    }
                }
                Ok(NanBox::handle(result.to_raw()))
            }
            "difference" => {
                let result = self.realm.new_collection(true);
                for m in &mine {
                    self.realm.collection_set(result, *m, *m);
                }
                if my_size <= other_size {
                    for m in &mine {
                        if other_has(self, *m)? {
                            self.realm.collection_delete(result, *m);
                        }
                    }
                } else {
                    for k in self.set_record_keys(obj, keys)? {
                        if in_mine(self, k) {
                            self.realm.collection_delete(result, k);
                        }
                    }
                }
                Ok(NanBox::handle(result.to_raw()))
            }
            // symmetricDifference: in exactly one of the two. `resultSetData` is a
            // copy of `this` taken now; membership is tested against the LIVE
            // `this` (which the argument's `keys` iterator may mutate), per
            // ECMA-262 24.2.4.19 — not against the `resultSetData` copy.
            _ => {
                let result = self.realm.new_collection(true);
                for m in &mine {
                    self.realm.collection_set(result, *m, *m);
                }
                for k in self.set_record_keys(obj, keys)? {
                    let in_o = self.realm.collection_has(handle, k);
                    let in_result = self.realm.collection_has(result, k);
                    if in_o {
                        if in_result {
                            self.realm.collection_delete(result, k);
                        }
                    } else if !in_result {
                        self.realm.collection_set(result, k, k);
                    }
                }
                Ok(NanBox::handle(result.to_raw()))
            }
        }
    }

    fn create_html(
        &mut self,
        s_bytes: &[u8],
        tag: &str,
        attribute: &str,
        value: Option<NanBox>,
    ) -> Result<NanBox, ExecError> {
        let mut out: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
        out.push(b'<');
        out.extend_from_slice(tag.as_bytes());
        if !attribute.is_empty()
            && let Some(v) = value
        {
            // ToString first (errors propagate), then escape `"` → `&quot;`.
            let v = self.coerce_to_string(v)?;
            let escaped = v.replace('"', "&quot;");
            out.push(b' ');
            out.extend_from_slice(attribute.as_bytes());
            out.extend_from_slice(b"=\"");
            out.extend_from_slice(escaped.as_bytes());
            out.push(b'"');
        }
        out.push(b'>');
        out.extend_from_slice(s_bytes);
        out.extend_from_slice(b"</");
        out.extend_from_slice(tag.as_bytes());
        out.push(b'>');
        Ok(self.new_str_bytes(out))
    }
}
