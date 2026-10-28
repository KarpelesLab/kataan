use super::*;

impl Interp {
    /// `InternalizeJSONProperty` — the `JSON.parse` reviver walk *with source-text
    /// access* (the `json-parse-with-source` proposal). Reads `val = Get(holder,
    /// name)` via `[[Get]]`; if `val` is an array its indices are recursed, else
    /// its snapshot of own enumerable keys, bottom-up. It additionally passes the
    /// reviver a third `context` argument: a plain
    /// object that, for a value that is a primitive **still equal to the one the
    /// parser produced at this position**, carries a `source` data property with
    /// that value's exact JSON source text. Structural values (objects/arrays) and
    /// values a reviver has forward-substituted get a bare `{}` context.
    ///
    /// `snapshot` is the parse-time source node for the value currently at
    /// `holder[key]` (or `None` for a member the reviver added).
    pub(crate) fn json_revive_ctx(
        &mut self,
        holder: crate::heap::Handle,
        key: &str,
        reviver: NanBox,
        snapshot: Option<&JsonSrc>,
    ) -> Result<NanBox, ExecError> {
        let value = self.read_member(holder, key)?;
        if let Some(vh) = value.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(self.proxy_key_target(vh)) {
                let len_v = self.read_member(vh, "length")?;
                let len_f = self.coerce_to_integer_or_infinity(len_v)?;
                let len = len_f.clamp(0.0, 9_007_199_254_740_991.0) as usize;
                for i in 0..len {
                    let ks = alloc::format!("{i}");
                    let child = match snapshot {
                        Some(JsonSrc::Array(v)) => v.get(i),
                        _ => None,
                    };
                    let nv = self.json_revive_ctx(vh, &ks, reviver, child)?;
                    if matches!(nv.unpack(), Unpacked::Undefined) {
                        self.delete_property_of(vh, &ks)?;
                    } else {
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
            } else {
                // EnumerableOwnPropertyNames(val, KEY) for any other object. A
                // typed array is *not* a plain object cell but does own its
                // integer indices, so a reviver that substitutes one mid-walk
                // (`this[1] = new Int8Array(1)`) must still be re-entered for
                // each of its elements.
                let keys = if let Some(pk) = self.proxy_own_enumerable_keys(vh)? {
                    pk
                } else if let Some(len) = self.realm.typed_len(vh) {
                    (0..len).map(|i| alloc::format!("{i}")).collect()
                } else {
                    self.realm
                        .object_keys(self.proxy_key_target(vh))
                        .unwrap_or_default()
                };
                for k in keys {
                    let child = match snapshot {
                        Some(JsonSrc::Object(pairs)) => {
                            pairs.iter().find(|(pk, _)| pk == &k).map(|(_, s)| s)
                        }
                        _ => None,
                    };
                    let nv = self.json_revive_ctx(vh, &k, reviver, child)?;
                    if matches!(nv.unpack(), Unpacked::Undefined) {
                        self.delete_property_of(vh, &k)?;
                    } else {
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
        // Build the `context` object. Only a primitive value that is *still* the
        // one the parser produced here (SameValue) exposes its `source`.
        let context = self.realm.new_object();
        if !self.is_object_value(value)
            && let Some(JsonSrc::Prim { value: pv, source }) = snapshot
            && self.realm.same_value(*pv, value)
        {
            let src_box = self.new_str(source);
            self.realm.set_property(context, "source", src_box);
        }
        let kb = self.new_str(key);
        self.call_with_this(
            reviver,
            NanBox::handle(holder.to_raw()),
            &[kb, value, NanBox::handle(context.to_raw())],
        )
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
            } else if self.realm.is_array(self.proxy_key_target(rh)) {
                // Build the PropertyList via `[[Get]]` (length + each index) so a
                // Proxy replacer drives its traps and an element getter's abrupt
                // completion propagates. ToString each element that is a String, a
                // Number, or a String/Number wrapper; dedupe, preserve order.
                let len_v = self.read_member(rh, "length")?;
                let len = self
                    .coerce_to_integer_or_infinity(len_v)?
                    .clamp(0.0, 9_007_199_254_740_991.0) as usize;
                let mut list: Vec<String> = Vec::new();
                for i in 0..len {
                    let e = self.read_member(rh, &alloc::format!("{i}"))?;
                    let item = match e.unpack() {
                        Unpacked::Number(_) => Some(self.realm.to_display_string(e)),
                        Unpacked::Handle(raw) => {
                            let h = Handle::from_raw(raw);
                            if self.realm.is_string_handle(h) {
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
        let mut cx = JsonOut {
            replacer_fn,
            property_list: property_list.as_deref(),
            gap: &gap,
            stack: Vec::new(),
            out: String::new(),
        };
        let value = self.read_member(holder, "")?;
        let wrote = self.json_ser_got(&mut cx, holder, JsonKey::Str(""), value, "")?;
        Ok(wrote.then_some(cx.out))
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
                Unpacked::Handle(r) if self.realm.is_string_handle(Handle::from_raw(r)) => {
                    let s = self.coerce_to_string(v)?;
                    self.new_str(&s)
                }
                _ => prim,
            });
        }
        Ok(v)
    }

    /// SerializeJSONProperty after its `Get` (`value` is `holder[key]`): applies
    /// `toJSON`, then the replacer function, then appends the serialization to
    /// `cx.out`. Returns `false` (having written nothing) when the value drops
    /// (`undefined`/callable/symbol). The key is only turned into a string when
    /// `toJSON` or a replacer function observes it, so an array element costs no
    /// index formatting.
    fn json_ser_got(
        &mut self,
        cx: &mut JsonOut<'_>,
        holder: Handle,
        key: JsonKey<'_>,
        mut value: NanBox,
        indent: &str,
    ) -> Result<bool, ExecError> {
        // If value is an Object (or BigInt) with a callable `toJSON`, call it.
        if let Some(h) = value.as_handle().map(Handle::from_raw) {
            let tj = self.read_member(h, "toJSON")?;
            if self.is_callable_value(tj) {
                let kb = key.to_value(self);
                value = self.call_with_this(tj, value, &[kb])?;
            }
        }
        // ReplacerFunction: value = replacer.call(holder, key, value).
        if self.is_callable_value(cx.replacer_fn) {
            let kb = key.to_value(self);
            value = self.call_with_this(
                cx.replacer_fn,
                NanBox::handle(holder.to_raw()),
                &[kb, value],
            )?;
        }
        self.json_ser_value(cx, value, indent)
    }

    /// The type-dispatch tail of SerializeJSONProperty: appends an already-
    /// (toJSON/replacer-)transformed `value` to `cx.out`; `false` if it drops.
    fn json_ser_value(
        &mut self,
        cx: &mut JsonOut<'_>,
        value: NanBox,
        indent: &str,
    ) -> Result<bool, ExecError> {
        // Unwrap a primitive-wrapper object to its boxed primitive first (so a
        // `new Number(1)` serializes as `1`, `new String("x")` as `"x"`).
        let value = if let Some(h) = value.as_handle().map(Handle::from_raw) {
            // RawJSON object: emit the stored source verbatim.
            if self.realm.get_property(h, RAW_JSON_BRAND).is_some()
                && let Some(raw) = self.realm.get_property(h, "rawJSON")
            {
                cx.out.push_str(&self.realm.to_display_string(raw));
                return Ok(true);
            }
            if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
                // SerializeJSONProperty step 4: a `[[NumberData]]` wrapper is
                // `ToNumber(value)` and a `[[StringData]]` wrapper is
                // `ToString(value)` — both applied to the *wrapper*, so a custom
                // `valueOf`/`toString` is honored. Boolean/BigInt wrappers use the
                // boxed primitive directly.
                match prim.unpack() {
                    Unpacked::Number(_) => self.coerce_to_number(value)?,
                    Unpacked::Handle(r) if self.realm.is_string_handle(Handle::from_raw(r)) => {
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
            Unpacked::Null => cx.out.push_str("null"),
            Unpacked::Bool(b) => cx.out.push_str(if b { "true" } else { "false" }),
            Unpacked::Number(n) => {
                if n.is_finite() {
                    cx.out.push_str(&self.realm.to_display_string(value));
                } else {
                    cx.out.push_str("null");
                }
            }
            Unpacked::Undefined => return Ok(false),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if self.realm.is_string_handle(h) {
                    // Collapse a rope once so the quoting borrows one leaf.
                    self.realm.flatten_string(h);
                    if let Some(bytes) = self.realm.string_leaf_bytes(h) {
                        json_quote_wtf8_into(&mut cx.out, bytes);
                    } else if let Some(bytes) = self.realm.string_bytes(h) {
                        json_quote_wtf8_into(&mut cx.out, &bytes);
                    }
                    return Ok(true);
                }
                if self.realm.bigint_at(h).is_some() {
                    let m = self.new_str("Do not know how to serialize a BigInt");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A Symbol value serializes to nothing (`typeof` is "symbol"), both
                // as a value and (via the object key set) as a key.
                if self.realm.symbol_at(h).is_some() {
                    return Ok(false);
                }
                // A callable value (function or class constructor) serializes to
                // nothing (`typeof` is "function").
                if self.is_callable(h) {
                    return Ok(false);
                }
                // Array vs object, via `IsArray` (which unwraps a proxy chain to
                // its target — a proxy over an array serializes as an array). A
                // typed array is NOT an Array exotic, so it serializes as an
                // object keyed by its indices (`{"0":1,…}`), per `JSON.stringify`.
                if self.is_array_unwrap_proxy(value)? {
                    self.json_ser_array(cx, h, indent)?;
                } else {
                    self.json_ser_object(cx, h, indent)?;
                }
            }
        }
        Ok(true)
    }

    /// The cycle check shared by objects and arrays: pushes `h` on the stack,
    /// or throws if it is already being serialized.
    fn json_enter(&mut self, cx: &mut JsonOut<'_>, h: Handle) -> Result<(), ExecError> {
        if cx.stack.contains(&h) {
            let m = self.new_str("Converting circular structure to JSON");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        cx.stack.push(h);
        Ok(())
    }

    /// `SerializeJSONObject` — `{ … }` with the PropertyList (or own enumerable
    /// keys), recursing per member; cycle-checked via the stack.
    fn json_ser_object(
        &mut self,
        cx: &mut JsonOut<'_>,
        h: Handle,
        indent: &str,
    ) -> Result<(), ExecError> {
        self.json_enter(cx, h)?;
        let new_indent = alloc::format!("{indent}{}", cx.gap);
        // The key set: the explicit PropertyList, else the object's own enumerable
        // string keys (a typed array enumerates its integer indices).
        let keys: Vec<String> = match cx.property_list {
            Some(list) => list.to_vec(),
            None => {
                if let Some(len) = self.realm.typed_len(h) {
                    (0..len).map(|i| alloc::format!("{i}")).collect()
                } else if self.realm.proxy_at(h).is_some() {
                    // A proxy drives `[[OwnPropertyKeys]]` (the `ownKeys` trap, or
                    // trapless forwarding to the target) then filters to own
                    // enumerable **string** keys via `[[GetOwnProperty]]` (symbols
                    // never appear in JSON). Values are read later through the
                    // proxy's `get` trap.
                    let mut ks = Vec::new();
                    for key in self.own_property_keys_values(h)? {
                        let name = self.member_key(key);
                        if crate::realm::is_internal_key(&name) {
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
        let pretty = !cx.gap.is_empty();
        cx.out.push('{');
        let mut any = false;
        for k in keys {
            let mark = cx.out.len();
            if any {
                cx.out.push(',');
            }
            if pretty {
                cx.out.push('\n');
                cx.out.push_str(&new_indent);
            }
            json_quote_wtf8_into(&mut cx.out, k.as_bytes());
            cx.out.push_str(if pretty { ": " } else { ":" });
            // `Get(h, k)`: an own data property of an ordinary object is its
            // slot; everything else takes the full `[[Get]]`.
            let value = match self.json_plain_own_data(h, &k) {
                Some(v) => v,
                None => self.read_member(h, &k)?,
            };
            if self.json_ser_got(cx, h, JsonKey::Str(&k), value, &new_indent)? {
                any = true;
            } else {
                cx.out.truncate(mark);
            }
        }
        cx.stack.pop();
        if any && pretty {
            cx.out.push('\n');
            cx.out.push_str(indent);
        }
        cx.out.push('}');
        Ok(())
    }

    /// `SerializeJSONArray` — `[ … ]`, each element serialized (a dropped element
    /// becomes `null`); cycle-checked via the stack.
    fn json_ser_array(
        &mut self,
        cx: &mut JsonOut<'_>,
        h: Handle,
        indent: &str,
    ) -> Result<(), ExecError> {
        self.json_enter(cx, h)?;
        let new_indent = alloc::format!("{indent}{}", cx.gap);
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
        let pretty = !cx.gap.is_empty();
        cx.out.push('[');
        for i in 0..len {
            if i > 0 {
                cx.out.push(',');
            }
            if pretty {
                cx.out.push('\n');
                cx.out.push_str(&new_indent);
            }
            // `Get(h, i)`: a present slot of a plain array with no per-index side
            // table is the element itself; anything else (a hole, an accessor, a
            // proxy, a typed array) goes through the full `[[Get]]`.
            let direct = if self.realm.array_has_index_side_table(h) {
                None
            } else {
                self.realm
                    .array_elements(h)
                    .and_then(|a| a.get(i).copied())
                    .filter(|v| !v.is_hole())
            };
            let value = match direct {
                Some(v) => v,
                None => self.read_member(h, &alloc::format!("{i}"))?,
            };
            // A PropertyList does NOT filter array indices (arrays ignore it for
            // their own elements), but it MUST propagate to any nested objects,
            // which it does through `cx`.
            if !self.json_ser_got(cx, h, JsonKey::Index(i), value, &new_indent)? {
                cx.out.push_str("null");
            }
        }
        cx.stack.pop();
        if len > 0 && pretty {
            cx.out.push('\n');
            cx.out.push_str(indent);
        }
        cx.out.push(']');
        Ok(())
    }

    /// `[[Get]]` of `key` on `h` when that is simply an own data slot of an
    /// ordinary object — no proxy, typed array, module namespace, mapped
    /// `arguments` or accessor involved. `None` means "take the full
    /// `read_member` path", never "absent".
    fn json_plain_own_data(&self, h: Handle, key: &str) -> Option<NanBox> {
        if self.realm.proxy_at(h).is_some()
            || self.realm.typed_kind(h).is_some()
            || self.realm.is_host_exotic(h)
            || self.arg_maps.contains_key(&h.to_raw())
        {
            return None;
        }
        self.realm.plain_own_data(h, key)
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
                // The JSON `Number` production, exactly:
                //   `-`? ( `0` | [1-9][0-9]* ) ( `.` [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
                // Scanning "any run of digits/`.`/`e`/`+`/`-`" and handing it to
                // Rust's `f64` parser accepted forms JSON forbids — a leading zero
                // (`00`, `013`), a bare trailing point (`1.`), and a leading point
                // (`.1`) all parse in Rust but must be SyntaxErrors here.
                let start = *pos;
                let digit = |c: &[char], p: usize| c.get(p).is_some_and(char::is_ascii_digit);
                if c.get(*pos) == Some(&'-') {
                    *pos += 1;
                }
                if c.get(*pos) == Some(&'0') {
                    *pos += 1;
                } else if digit(c, *pos) {
                    while digit(c, *pos) {
                        *pos += 1;
                    }
                } else {
                    return Err(self.json_error("Invalid number in JSON"));
                }
                if c.get(*pos) == Some(&'.') {
                    *pos += 1;
                    if !digit(c, *pos) {
                        return Err(self.json_error("Invalid number in JSON"));
                    }
                    while digit(c, *pos) {
                        *pos += 1;
                    }
                }
                if matches!(c.get(*pos), Some('e' | 'E')) {
                    *pos += 1;
                    if matches!(c.get(*pos), Some('+' | '-')) {
                        *pos += 1;
                    }
                    if !digit(c, *pos) {
                        return Err(self.json_error("Invalid number in JSON"));
                    }
                    while digit(c, *pos) {
                        *pos += 1;
                    }
                }
                let text: String = c[start..*pos].iter().collect();
                text.parse::<f64>()
                    .map(NanBox::number)
                    .map_err(|_| self.json_error("Invalid number in JSON"))
            }
            _ => Err(self.json_error("Unexpected token in JSON")),
        }
    }

    /// Like [`Self::json_parse`] but also returns a parallel [`JsonSrc`] tree that
    /// records, for every primitive leaf, the value produced and its exact source
    /// substring — the raw material for the `json-parse-with-source` reviver
    /// `context`. Only invoked when `JSON.parse` is given a callable reviver.
    pub(crate) fn json_parse_src(
        &mut self,
        c: &[char],
        pos: &mut usize,
        depth: usize,
    ) -> Result<(NanBox, JsonSrc), ExecError> {
        skip_ws(c, pos);
        let Some(&ch) = c.get(*pos) else {
            return Err(self.json_error("Unexpected end of JSON input"));
        };
        if matches!(ch, '[' | '{') && depth >= self.realm.limits.max_json_depth {
            return Err(self.json_error("Maximum JSON nesting depth exceeded"));
        }
        // Helper: the source substring `c[start..*pos]` collected as a String.
        match ch {
            'n' | 't' | 'f' | '"' | '-' | '0'..='9' => {
                let start = *pos;
                let value = self.json_parse(c, pos, depth)?;
                let source: String = c[start..*pos].iter().collect();
                Ok((value, JsonSrc::Prim { value, source }))
            }
            '[' => {
                *pos += 1;
                let mut elems = Vec::new();
                let mut srcs = Vec::new();
                skip_ws(c, pos);
                if c.get(*pos) == Some(&']') {
                    *pos += 1;
                    return Ok((
                        NanBox::handle(self.realm.new_array(elems).to_raw()),
                        JsonSrc::Array(srcs),
                    ));
                }
                loop {
                    let (v, s) = self.json_parse_src(c, pos, depth + 1)?;
                    elems.push(v);
                    srcs.push(s);
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
                Ok((
                    NanBox::handle(self.realm.new_array(elems).to_raw()),
                    JsonSrc::Array(srcs),
                ))
            }
            '{' => {
                *pos += 1;
                let obj = self.realm.new_object();
                let mut pairs = Vec::new();
                skip_ws(c, pos);
                if c.get(*pos) == Some(&'}') {
                    *pos += 1;
                    return Ok((NanBox::handle(obj.to_raw()), JsonSrc::Object(pairs)));
                }
                loop {
                    skip_ws(c, pos);
                    if c.get(*pos) != Some(&'"') {
                        return Err(self.json_error("Expected property name"));
                    }
                    let key = crate::wtf8::to_string_lossy(&self.json_string(c, pos)?);
                    skip_ws(c, pos);
                    if c.get(*pos) != Some(&':') {
                        return Err(self.json_error("Expected ':'"));
                    }
                    *pos += 1;
                    let (v, s) = self.json_parse_src(c, pos, depth + 1)?;
                    self.realm.set_property(obj, &key, v);
                    // A duplicate key keeps the last value (matching `set_property`);
                    // record the last source for it too.
                    if let Some(slot) = pairs
                        .iter_mut()
                        .find(|(k, _): &&mut (String, JsonSrc)| k == &key)
                    {
                        slot.1 = s;
                    } else {
                        pairs.push((key, s));
                    }
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
                Ok((NanBox::handle(obj.to_raw()), JsonSrc::Object(pairs)))
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

/// A parse-time source-text snapshot mirroring a parsed JSON value's structure,
/// used by the `json-parse-with-source` reviver `context`. Each primitive leaf
/// records the value the parser produced and its exact source substring so the
/// reviver can be handed the original text (unless a reviver has since replaced
/// the value at that position — checked by SameValue).
pub(crate) enum JsonSrc {
    Prim { value: NanBox, source: String },
    Array(Vec<JsonSrc>),
    Object(Vec<(String, JsonSrc)>),
}

/// The state of one `JSON.stringify` run: the replacer, the gap, the cycle
/// stack and the single output buffer every value is appended to.
struct JsonOut<'a> {
    replacer_fn: NanBox,
    property_list: Option<&'a [String]>,
    gap: &'a str,
    stack: Vec<Handle>,
    out: String,
}

/// The key of a property being serialized: a named key, or an array index whose
/// string form is only built when user code (`toJSON`, a replacer) observes it.
#[derive(Clone, Copy)]
enum JsonKey<'a> {
    Str(&'a str),
    Index(usize),
}

impl JsonKey<'_> {
    fn to_value(self, interp: &mut Interp) -> NanBox {
        match self {
            JsonKey::Str(k) => interp.new_str(k),
            JsonKey::Index(i) => interp.new_str(&alloc::format!("{i}")),
        }
    }
}
