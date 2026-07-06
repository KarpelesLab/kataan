use super::*;

impl<'a> Interp<'a> {
    /// `InternalizeJSONProperty(holder, name, reviver)` (25.5.1.1) — the
    /// `JSON.parse` reviver walk. Reads `val = Get(holder, name)` via `[[Get]]`
    /// (so an inherited value is observed and a getter fires); if `val` is an
    /// array its indices are recursed (CreateDataProperty / Delete the result),
    /// else its *snapshot* of own enumerable keys is recursed; finally returns
    /// `reviver.call(holder, name, val)`. A `undefined` child result deletes the
    /// member; any other re-creates it as an own data property.
    pub(crate) fn json_revive(
        &mut self,
        holder: crate::heap::Handle,
        key: &str,
        reviver: NanBox,
    ) -> Result<NanBox, ExecError> {
        // `val = ? Get(holder, name)` — through `[[Get]]` (prototype chain + getters).
        let value = self.read_member(holder, key)?;
        if let Some(vh) = value.as_handle().map(Handle::from_raw) {
            // `IsArray(val)` unwraps proxy chains — a Proxy whose target is an array
            // takes the array branch (its traps then fire on the reads/writes
            // below). A revoked proxy would already have thrown at the `[[Get]]`
            // above.
            if self.realm.is_array(self.proxy_key_target(vh)) {
                // `len = ? LengthOfArrayLike(val)` = ToLength(? Get(val,"length")):
                // both the `[[Get]]` (a proxy trap) and the numeric coercion (a
                // `valueOf`) run user code whose abrupt completion must propagate.
                let len_v = self.read_member(vh, "length")?;
                let len_f = self.coerce_to_integer_or_infinity(len_v)?;
                let len = len_f.max(0.0).min(9_007_199_254_740_991.0) as usize;
                for i in 0..len {
                    let ks = alloc::format!("{i}");
                    let nv = self.json_revive(vh, &ks, reviver)?;
                    if matches!(nv.unpack(), Unpacked::Undefined) {
                        // `? val.[[Delete]](P)` — proxy-aware, so a throwing
                        // `deleteProperty` trap propagates (a non-abrupt `false` is
                        // ignored).
                        self.delete_property_of(vh, &ks)?;
                    } else {
                        // `? CreateDataProperty(val, P, newElement)` — proxy-aware
                        // [[DefineOwnProperty]]: a throwing `defineProperty` trap
                        // propagates, a non-abrupt failure (e.g. a non-configurable
                        // index) is ignored (plain CreateDataProperty, not …OrThrow).
                        let desc = self.realm.new_object();
                        self.realm.set_property(desc, "value", nv);
                        self.realm
                            .set_property(desc, "writable", NanBox::boolean(true));
                        self.realm
                            .set_property(desc, "enumerable", NanBox::boolean(true));
                        self.realm
                            .set_property(desc, "configurable", NanBox::boolean(true));
                        self.apply_descriptor(vh, &ks, desc, true)?;
                    }
                }
            } else if self.realm.object_keys(vh).is_some() || self.realm.proxy_at(vh).is_some() {
                // `keys = ? EnumerableOwnPropertyNames(val, key)` — snapshot now (the
                // reviver may mutate `val` during the walk). A proxy drives its
                // `ownKeys`/`getOwnPropertyDescriptor` traps (errors propagate).
                let keys = if let Some(pk) = self.proxy_own_enumerable_keys(vh)? {
                    pk
                } else {
                    // A proxy with no `ownKeys` trap enumerates its target's own
                    // enumerable keys (proxy_key_target is identity for a plain
                    // object).
                    self.realm
                        .object_keys(self.proxy_key_target(vh))
                        .unwrap_or_default()
                };
                for k in keys {
                    let nv = self.json_revive(vh, &k, reviver)?;
                    if matches!(nv.unpack(), Unpacked::Undefined) {
                        // `? val.[[Delete]](P)` — proxy-aware; a throwing
                        // `deleteProperty` trap propagates, a non-abrupt `false`
                        // (e.g. a non-configurable own property) is ignored.
                        self.delete_property_of(vh, &k)?;
                    } else {
                        // `? CreateDataProperty(val, P, newElement)` — proxy-aware
                        // [[DefineOwnProperty]]: a throwing `defineProperty` trap
                        // propagates, a non-abrupt failure (a non-configurable own
                        // property) is ignored (plain CreateDataProperty).
                        let desc = self.realm.new_object();
                        self.realm.set_property(desc, "value", nv);
                        self.realm
                            .set_property(desc, "writable", NanBox::boolean(true));
                        self.realm
                            .set_property(desc, "enumerable", NanBox::boolean(true));
                        self.realm
                            .set_property(desc, "configurable", NanBox::boolean(true));
                        self.apply_descriptor(vh, &k, desc, true)?;
                    }
                }
            }
        }
        let kb = self.new_str(key);
        self.call_with_this(reviver, NanBox::handle(holder.to_raw()), &[kb, value])
    }

    /// `JSON.stringify(value, replacer, space)` — the unified spec algorithm
    /// (25.5.2): a single recursive `SerializeJSONProperty` that applies the
    /// replacer (function or property-list array) and `toJSON` at each node,
    /// detects cycles, and honors the `space` gap. Returns `None` when the top
    /// value serializes to nothing (`undefined`/function/symbol).
    pub(crate) fn json_stringify(
        &mut self,
        value: NanBox,
        replacer: NanBox,
        space: NanBox,
    ) -> Result<Option<String>, ExecError> {
        // ReplacerFunction / PropertyList from `replacer`.
        let mut replacer_fn = NanBox::undefined();
        let mut property_list: Option<Vec<String>> = None;
        if let Some(rh) = replacer.as_handle().map(Handle::from_raw) {
            if self.is_callable(rh) {
                replacer_fn = replacer;
            } else if self.realm.is_array(rh) {
                // Build the PropertyList: ToString each element that is a String, a
                // Number, or a String/Number wrapper object; dedupe, preserve order.
                let elems = self
                    .realm
                    .array_elements(rh)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                let mut list: Vec<String> = Vec::new();
                for e in elems {
                    let item = match e.unpack() {
                        Unpacked::Number(_) => Some(self.realm.to_display_string(e)),
                        Unpacked::Handle(raw) => {
                            let h = Handle::from_raw(raw);
                            if self.realm.string_value(h).is_some() {
                                Some(self.realm.to_display_string(e))
                            } else if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
                                // A String/Number wrapper contributes its key via
                                // ToString/ToNumber (honoring a user valueOf/toString).
                                // A Number *or* String wrapper's key is `ToString(v)`
                                // (per PropertyList construction) — not ToNumber.
                                match prim.unpack() {
                                    Unpacked::Number(_) => Some(self.coerce_to_string(e)?),
                                    Unpacked::Handle(pr)
                                        if self
                                            .realm
                                            .string_value(Handle::from_raw(pr))
                                            .is_some() =>
                                    {
                                        Some(self.coerce_to_string(e)?)
                                    }
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    if let Some(k) = item
                        && !list.contains(&k)
                    {
                        list.push(k);
                    }
                }
                property_list = Some(list);
            }
        }
        // The `space` gap: a Number (or Number wrapper) → that many spaces (clamped
        // to 0..=10); a String (or String wrapper) → its first 10 code units; else
        // empty (compact).
        let space = self.json_unwrap_wrapper(space)?;
        let gap = if let Some(n) = space.as_number() {
            // ToIntegerOrInfinity then `min(10)` spaces; a non-positive count is 0.
            // The cast to `usize` truncates toward zero (ToInteger) and a NaN maps
            // to 0, matching `min(MIN(ToInteger(space), 10), 0)`-style clamping.
            let n = if n >= 1.0 { (n as usize).min(10) } else { 0 };
            " ".repeat(n)
        } else if let Some(s) = space
            .as_handle()
            .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
        {
            s.chars().take(10).collect()
        } else {
            String::new()
        };
        // The wrapper holder `{ "": value }`.
        let holder = self.realm.new_object();
        self.realm.set_property(holder, "", value);
        let mut stack: Vec<Handle> = Vec::new();
        self.serialize_json_property(
            holder,
            "",
            &replacer_fn,
            property_list.as_deref(),
            &gap,
            "",
            &mut stack,
        )
    }

    /// The `space` argument's ToPrimitive: a `[[NumberData]]` wrapper becomes
    /// `ToNumber(space)` and a `[[StringData]]` wrapper `ToString(space)` (both
    /// honoring a user `valueOf`/`toString`); anything else is returned unchanged.
    fn json_unwrap_wrapper(&mut self, v: NanBox) -> Result<NanBox, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw)
            && let Some(prim) = self.realm.get_property(h, PRIM_WRAP)
        {
            return Ok(match prim.unpack() {
                Unpacked::Number(_) => self.coerce_to_number(v)?,
                Unpacked::Handle(r) if self.realm.string_value(Handle::from_raw(r)).is_some() => {
                    let s = self.coerce_to_string(v)?;
                    self.new_str(&s)
                }
                _ => prim,
            });
        }
        Ok(v)
    }

    /// `SerializeJSONProperty(key, holder)` — serializes `holder[key]`, applying
    /// `toJSON`, then the replacer function; returns `None` if the value drops
    /// (`undefined`/callable/symbol at a non-array position).
    #[allow(clippy::too_many_arguments)]
    fn serialize_json_property(
        &mut self,
        holder: Handle,
        key: &str,
        replacer_fn: &NanBox,
        property_list: Option<&[String]>,
        gap: &str,
        indent: &str,
        stack: &mut Vec<Handle>,
    ) -> Result<Option<String>, ExecError> {
        // value = Get(holder, key) — through read_member so getters fire.
        let mut value = self.read_member(holder, key)?;
        // If value is an Object (or BigInt) with a callable `toJSON`, call it.
        if let Some(h) = value.as_handle().map(Handle::from_raw) {
            let tj = self.read_member(h, "toJSON")?;
            if self.is_callable_value(tj) {
                let kb = self.new_str(key);
                value = self.call_with_this(tj, value, &[kb])?;
            }
        }
        // ReplacerFunction: value = replacer.call(holder, key, value).
        if self.is_callable_value(*replacer_fn) {
            let kb = self.new_str(key);
            value =
                self.call_with_this(*replacer_fn, NanBox::handle(holder.to_raw()), &[kb, value])?;
        }
        self.serialize_json_value(value, replacer_fn, property_list, gap, indent, stack)
    }

    /// The type-dispatch tail of SerializeJSONProperty: serialize an already-
    /// (toJSON/replacer-)transformed `value`.
    #[allow(clippy::too_many_arguments)]
    fn serialize_json_value(
        &mut self,
        value: NanBox,
        replacer_fn: &NanBox,
        property_list: Option<&[String]>,
        gap: &str,
        indent: &str,
        stack: &mut Vec<Handle>,
    ) -> Result<Option<String>, ExecError> {
        // Unwrap a primitive-wrapper object to its boxed primitive first (so a
        // `new Number(1)` serializes as `1`, `new String("x")` as `"x"`).
        let value = if let Some(h) = value.as_handle().map(Handle::from_raw) {
            // RawJSON object: emit the stored source verbatim.
            if self.realm.get_property(h, RAW_JSON_BRAND).is_some()
                && let Some(raw) = self.realm.get_property(h, "rawJSON")
            {
                return Ok(Some(self.realm.to_display_string(raw)));
            }
            if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
                // SerializeJSONProperty step 4: a `[[NumberData]]` wrapper is
                // `ToNumber(value)` and a `[[StringData]]` wrapper is
                // `ToString(value)` — both applied to the *wrapper*, so a custom
                // `valueOf`/`toString` is honored. Boolean/BigInt wrappers use the
                // boxed primitive directly.
                match prim.unpack() {
                    Unpacked::Number(_) => self.coerce_to_number(value)?,
                    Unpacked::Handle(r)
                        if self.realm.string_value(Handle::from_raw(r)).is_some() =>
                    {
                        let s = self.coerce_to_string(value)?;
                        self.new_str(&s)
                    }
                    _ => prim,
                }
            } else {
                value
            }
        } else {
            value
        };
        match value.unpack() {
            Unpacked::Null => Ok(Some(String::from("null"))),
            Unpacked::Bool(b) => Ok(Some(String::from(if b { "true" } else { "false" }))),
            Unpacked::Number(n) => Ok(Some(if n.is_finite() {
                self.realm.to_display_string(value)
            } else {
                String::from("null")
            })),
            Unpacked::Undefined => Ok(None),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if let Some(bytes) = self.realm.string_bytes(h) {
                    return Ok(Some(json_quote_wtf8(&bytes)));
                }
                if self.realm.bigint_at(h).is_some() {
                    let m = self.new_str("Do not know how to serialize a BigInt");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A callable value (function or class constructor) serializes to
                // nothing (`typeof` is "function").
                if self.is_callable(h) || self.realm.class_at(h).is_some() {
                    return Ok(None);
                }
                // Array vs object, via `IsArray` (which unwraps a proxy chain to
                // its target — a proxy over an array serializes as an array). A
                // typed array is NOT an Array exotic, so it serializes as an
                // object keyed by its indices (`{"0":1,…}`), per `JSON.stringify`.
                if self.is_array_unwrap_proxy(value)? {
                    self.serialize_json_array(h, replacer_fn, property_list, gap, indent, stack)
                        .map(Some)
                } else {
                    self.serialize_json_object(h, replacer_fn, property_list, gap, indent, stack)
                        .map(Some)
                }
            }
        }
    }

    /// `SerializeJSONObject` — `{ … }` with the PropertyList (or own enumerable
    /// keys), recursing per member; cycle-checked via `stack`.
    #[allow(clippy::too_many_arguments)]
    fn serialize_json_object(
        &mut self,
        h: Handle,
        replacer_fn: &NanBox,
        property_list: Option<&[String]>,
        gap: &str,
        indent: &str,
        stack: &mut Vec<Handle>,
    ) -> Result<String, ExecError> {
        if stack.contains(&h) {
            let m = self.new_str("Converting circular structure to JSON");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        stack.push(h);
        let new_indent = alloc::format!("{indent}{gap}");
        // The key set: the explicit PropertyList, else the object's own enumerable
        // string keys (a typed array enumerates its integer indices).
        let keys: Vec<String> = match property_list {
            Some(list) => list.to_vec(),
            None => {
                if let Some(len) = self.realm.typed_len(h) {
                    (0..len).map(|i| alloc::format!("{i}")).collect()
                } else if self.realm.proxy_at(h).is_some() {
                    // A proxy drives `[[OwnPropertyKeys]]` (the `ownKeys` trap, or
                    // trapless forwarding to the target) then filters to own
                    // enumerable **string** keys via `[[GetOwnProperty]]` (symbols
                    // never appear in JSON). Values are read later through the
                    // proxy's `get` trap by `serialize_json_property`.
                    let mut ks = Vec::new();
                    for key in self.own_property_keys_values(h)? {
                        let name = self.member_key(key);
                        if name.starts_with('\u{0}') {
                            continue; // a symbol key (internal `\0sym:` form)
                        }
                        let desc = self.descriptor_of(h, &name)?;
                        if matches!(desc.unpack(), Unpacked::Undefined) {
                            continue;
                        }
                        let enumerable = desc
                            .as_handle()
                            .map(Handle::from_raw)
                            .and_then(|dh| self.realm.get_property(dh, "enumerable"))
                            .is_some_and(|v| self.realm.truthy(v));
                        if enumerable {
                            ks.push(name);
                        }
                    }
                    ks
                } else {
                    self.realm.object_keys(h).unwrap_or_default()
                }
            }
        };
        let mut parts: Vec<String> = Vec::new();
        for k in keys {
            if let Some(s) = self.serialize_json_property(
                h,
                &k,
                replacer_fn,
                property_list,
                gap,
                &new_indent,
                stack,
            )? {
                let sep = if gap.is_empty() { ":" } else { ": " };
                parts.push(alloc::format!("{}{sep}{s}", json_quote(&k)));
            }
        }
        stack.pop();
        let out = if parts.is_empty() {
            String::from("{}")
        } else if gap.is_empty() {
            alloc::format!("{{{}}}", parts.join(","))
        } else {
            alloc::format!(
                "{{\n{new_indent}{}\n{indent}}}",
                parts.join(&alloc::format!(",\n{new_indent}"))
            )
        };
        Ok(out)
    }

    /// `SerializeJSONArray` — `[ … ]`, each element serialized (a dropped element
    /// becomes `null`); cycle-checked via `stack`.
    #[allow(clippy::too_many_arguments)]
    fn serialize_json_array(
        &mut self,
        h: Handle,
        replacer_fn: &NanBox,
        _property_list: Option<&[String]>,
        gap: &str,
        indent: &str,
        stack: &mut Vec<Handle>,
    ) -> Result<String, ExecError> {
        if stack.contains(&h) {
            let m = self.new_str("Converting circular structure to JSON");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        stack.push(h);
        let new_indent = alloc::format!("{indent}{gap}");
        let len = if self.realm.typed_kind(h).is_some() {
            self.realm.typed_len(h).unwrap_or(0)
        } else if self.realm.proxy_at(h).is_some() {
            // A proxy over an array: `LengthOfArrayLike` = ToLength(Get(h,"length")),
            // read through the proxy's `get` trap.
            let lv = self.read_member(h, "length")?;
            let ln = self.coerce_to_number(lv)?;
            let n = self.realm.to_number(ln);
            if n.is_finite() && n > 0.0 {
                (n as usize).min(self.realm.limits.max_array_len)
            } else {
                0
            }
        } else {
            self.realm.array_length(h).unwrap_or(0)
        };
        let mut parts: Vec<String> = Vec::with_capacity(len);
        for i in 0..len {
            let ks = alloc::format!("{i}");
            // The element replacer/PropertyList still applies via property serialize;
            // a PropertyList does NOT filter array indices (arrays ignore it).
            let s = self
                .serialize_json_property(h, &ks, replacer_fn, None, gap, &new_indent, stack)?
                .unwrap_or_else(|| String::from("null"));
            parts.push(s);
        }
        stack.pop();
        let out = if parts.is_empty() {
            String::from("[]")
        } else if gap.is_empty() {
            alloc::format!("[{}]", parts.join(","))
        } else {
            alloc::format!(
                "[\n{new_indent}{}\n{indent}]",
                parts.join(&alloc::format!(",\n{new_indent}"))
            )
        };
        Ok(out)
    }

    /// A `SyntaxError` for a malformed `JSON.parse` input (the spec error type, so
    /// `catch (e) { e instanceof SyntaxError }` works).
    pub(crate) fn json_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_ERROR_BASE + 3, Some(m)))
    }

    pub(crate) fn json_parse(
        &mut self,
        c: &[char],
        pos: &mut usize,
        depth: usize,
    ) -> Result<NanBox, ExecError> {
        skip_ws(c, pos);
        let err = |s: &mut Self| s.json_error("Unexpected end of JSON input");
        let Some(&ch) = c.get(*pos) else {
            return Err(err(self));
        };
        if matches!(ch, '[' | '{') && depth >= self.realm.limits.max_json_depth {
            return Err(self.json_error("Maximum JSON nesting depth exceeded"));
        }
        match ch {
            'n' => self.json_lit(c, pos, "null", NanBox::null()),
            't' => self.json_lit(c, pos, "true", NanBox::boolean(true)),
            'f' => self.json_lit(c, pos, "false", NanBox::boolean(false)),
            '"' => {
                let s = self.json_string(c, pos)?;
                Ok(self.new_str_bytes(s))
            }
            '[' => {
                *pos += 1;
                let mut elems = Vec::new();
                skip_ws(c, pos);
                if c.get(*pos) == Some(&']') {
                    *pos += 1;
                    return Ok(NanBox::handle(self.realm.new_array(elems).to_raw()));
                }
                loop {
                    let v = self.json_parse(c, pos, depth + 1)?;
                    elems.push(v);
                    skip_ws(c, pos);
                    match c.get(*pos) {
                        Some(',') => *pos += 1,
                        Some(']') => {
                            *pos += 1;
                            break;
                        }
                        _ => return Err(self.json_error("Expected ',' or ']'")),
                    }
                }
                Ok(NanBox::handle(self.realm.new_array(elems).to_raw()))
            }
            '{' => {
                *pos += 1;
                let obj = self.realm.new_object();
                skip_ws(c, pos);
                if c.get(*pos) == Some(&'}') {
                    *pos += 1;
                    return Ok(NanBox::handle(obj.to_raw()));
                }
                loop {
                    skip_ws(c, pos);
                    if c.get(*pos) != Some(&'"') {
                        return Err(self.json_error("Expected property name"));
                    }
                    // Keys live in the `&str`-keyed object layer; a lone surrogate
                    // in a *key* (an exotic edge) decodes lossily.
                    let key = crate::wtf8::to_string_lossy(&self.json_string(c, pos)?);
                    skip_ws(c, pos);
                    if c.get(*pos) != Some(&':') {
                        return Err(self.json_error("Expected ':'"));
                    }
                    *pos += 1;
                    let v = self.json_parse(c, pos, depth + 1)?;
                    self.realm.set_property(obj, &key, v);
                    skip_ws(c, pos);
                    match c.get(*pos) {
                        Some(',') => *pos += 1,
                        Some('}') => {
                            *pos += 1;
                            break;
                        }
                        _ => return Err(self.json_error("Expected ',' or '}'")),
                    }
                }
                Ok(NanBox::handle(obj.to_raw()))
            }
            '-' | '0'..='9' => {
                let start = *pos;
                if c.get(*pos) == Some(&'-') {
                    *pos += 1;
                }
                while c
                    .get(*pos)
                    .is_some_and(|d| d.is_ascii_digit() || matches!(d, '.' | 'e' | 'E' | '+' | '-'))
                {
                    *pos += 1;
                }
                let text: String = c[start..*pos].iter().collect();
                text.parse::<f64>()
                    .map(NanBox::number)
                    .map_err(|_| self.json_error("Invalid number in JSON"))
            }
            _ => Err(self.json_error("Unexpected token in JSON")),
        }
    }

    pub(crate) fn json_lit(
        &mut self,
        c: &[char],
        pos: &mut usize,
        word: &str,
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        if c[*pos..].iter().take(word.len()).copied().eq(word.chars()) {
            *pos += word.len();
            Ok(value)
        } else {
            Err(self.json_error("Unexpected token in JSON"))
        }
    }

    /// Parses a JSON string literal (the opening `"` is at `pos`) into **WTF-8
    /// bytes**, preserving lone surrogates (a `\uXXXX` surrogate with no valid
    /// partner is kept) and combining `\uXXXX\uXXXX` pairs into the astral scalar.
    pub(crate) fn json_string(
        &mut self,
        c: &[char],
        pos: &mut usize,
    ) -> Result<Vec<u8>, ExecError> {
        *pos += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            match c.get(*pos) {
                None => {
                    return Err(self.json_error("Unterminated string in JSON"));
                }
                Some('"') => {
                    *pos += 1;
                    return Ok(out);
                }
                Some('\\') => {
                    *pos += 1;
                    match c.get(*pos) {
                        Some('"') => out.push(b'"'),
                        Some('\\') => out.push(b'\\'),
                        Some('/') => out.push(b'/'),
                        Some('n') => out.push(b'\n'),
                        Some('t') => out.push(b'\t'),
                        Some('r') => out.push(b'\r'),
                        Some('b') => out.push(0x08),
                        Some('f') => out.push(0x0C),
                        Some('u') => {
                            let hi = json_hex4(c, *pos + 1)
                                .ok_or_else(|| self.json_error("Invalid \\u escape in JSON"))?;
                            *pos += 4;
                            // A high surrogate may pair with a following `\uXXXX`.
                            if (0xD800..=0xDBFF).contains(&hi)
                                && c.get(*pos + 1) == Some(&'\\')
                                && c.get(*pos + 2) == Some(&'u')
                                && let Some(lo) = json_hex4(c, *pos + 3)
                                && (0xDC00..=0xDFFF).contains(&lo)
                            {
                                let cp = 0x1_0000
                                    + ((u32::from(hi) - 0xD800) << 10)
                                    + (u32::from(lo) - 0xDC00);
                                crate::wtf8::encode_code_point(cp, &mut out);
                                *pos += 6;
                            } else {
                                crate::wtf8::encode_utf16_unit(hi, &mut out);
                            }
                        }
                        _ => return Err(self.json_error("Invalid escape in JSON")),
                    }
                    *pos += 1;
                }
                Some(&ch) => {
                    // A JSONString may not contain an unescaped control character
                    // (U+0000–U+001F); they must be written as `\n`, `\uXXXX`, etc.
                    if (ch as u32) < 0x20 {
                        return Err(
                            self.json_error("Bad control character in string literal in JSON")
                        );
                    }
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    *pos += 1;
                }
            }
        }
    }
}
