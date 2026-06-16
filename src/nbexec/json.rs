use super::*;

impl<'a> Interp<'a> {
    /// Serializes a value to JSON (`None` when the value is `undefined` or a
    /// function — which `JSON.stringify` omits / drops).
    /// Recursive-descent `JSON.parse` over a char slice, advancing `pos`.
    /// `JSON.parse` reviver: transforms `holder[key]` bottom-up — children first,
    /// then `reviver.call(holder, key, value)` (a `undefined` result deletes the
    /// member). Mirrors `InternalizeJSONProperty`.
    pub(crate) fn json_revive(
        &mut self,
        holder: crate::heap::Handle,
        key: &str,
        reviver: NanBox,
    ) -> Result<NanBox, ExecError> {
        let value = if self.realm.is_array(holder)
            && let Ok(i) = key.parse::<usize>()
        {
            self.realm.get_element(holder, i)
        } else {
            self.realm
                .get_property(holder, key)
                .unwrap_or(NanBox::undefined())
        };
        if let Some(vh) = value.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(vh) {
                let len = self.realm.array_length(vh).unwrap_or(0);
                for i in 0..len {
                    let ks = alloc::format!("{i}");
                    let nv = self.json_revive(vh, &ks, reviver)?;
                    self.realm.set_element(vh, i, nv);
                }
            } else if let Some(keys) = self.realm.object_keys(vh) {
                for k in keys {
                    let nv = self.json_revive(vh, &k, reviver)?;
                    if matches!(nv.unpack(), Unpacked::Undefined) {
                        self.realm.delete_property(vh, &k);
                    } else {
                        self.realm.set_property(vh, &k, nv);
                    }
                }
            }
        }
        let kb = self.new_str(key);
        self.call_with_this(reviver, NanBox::handle(holder.to_raw()), &[kb, value])
    }

    /// `JSON.stringify` function replacer: returns a fresh value tree where each
    /// node is `replacer.call(holder, key, value)`, recursing into the result's
    /// own properties/elements (does not mutate the input).
    pub(crate) fn json_apply_replacer(
        &mut self,
        holder: crate::heap::Handle,
        key: &str,
        value: NanBox,
        replacer: NanBox,
    ) -> Result<NanBox, ExecError> {
        let kb = self.new_str(key);
        let v = self.call_with_this(replacer, NanBox::handle(holder.to_raw()), &[kb, value])?;
        if let Some(vh) = v.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(vh) {
                let elems = self
                    .realm
                    .array_elements(vh)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                let mut out = Vec::with_capacity(elems.len());
                for (i, e) in elems.iter().enumerate() {
                    let ks = alloc::format!("{i}");
                    out.push(self.json_apply_replacer(vh, &ks, *e, replacer)?);
                }
                return Ok(NanBox::handle(self.realm.new_array(out).to_raw()));
            }
            if self.realm.string_value(vh).is_none()
                && let Some(keys) = self.realm.object_keys(vh)
            {
                let no = self.realm.new_object();
                for k in keys {
                    let pv = self
                        .realm
                        .get_property(vh, &k)
                        .unwrap_or(NanBox::undefined());
                    let nv = self.json_apply_replacer(vh, &k, pv, replacer)?;
                    if !matches!(nv.unpack(), Unpacked::Undefined) {
                        self.realm.set_property(no, &k, nv);
                    }
                }
                return Ok(NanBox::handle(no.to_raw()));
            }
        }
        Ok(v)
    }

    /// `JSON.stringify` array replacer: a fresh value tree keeping only object
    /// properties whose key is in `allow` (array elements are always kept).
    pub(crate) fn json_filter_keys(&mut self, value: NanBox, allow: &[String]) -> NanBox {
        if let Some(vh) = value.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(vh) {
                let elems = self
                    .realm
                    .array_elements(vh)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                let out: Vec<NanBox> = elems
                    .iter()
                    .map(|e| self.json_filter_keys(*e, allow))
                    .collect();
                return NanBox::handle(self.realm.new_array(out).to_raw());
            }
            if self.realm.string_value(vh).is_none()
                && let Some(keys) = self.realm.object_keys(vh)
            {
                let no = self.realm.new_object();
                // Keys are emitted in allowlist order (deduplicated, own keys only).
                let mut emitted: Vec<&String> = Vec::new();
                for k in allow {
                    if keys.contains(k) && !emitted.contains(&k) {
                        emitted.push(k);
                        let pv = self
                            .realm
                            .get_property(vh, k)
                            .unwrap_or(NanBox::undefined());
                        let nv = self.json_filter_keys(pv, allow);
                        self.realm.set_property(no, k, nv);
                    }
                }
                return NanBox::handle(no.to_raw());
            }
        }
        value
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
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    *pos += 1;
                }
            }
        }
    }

    /// Interpreter-aware `JSON.stringify` (compact): honors a `toJSON` method and
    /// invokes getters, unlike the realm-only `json_stringify`.
    pub(crate) fn json_to_string(&mut self, v: NanBox) -> Result<Option<String>, ExecError> {
        self.json_to_string_seen(v, "", &mut Vec::new())
    }

    /// `JSON.stringify` serialization tracking the ancestor handles in `seen`, so a
    /// circular structure throws a `TypeError` rather than overflowing the stack.
    /// `key` is the property name under which `v` appears in its parent (`""` at the
    /// top level), passed to a `toJSON(key)` method.
    pub(crate) fn json_to_string_seen(
        &mut self,
        v: NanBox,
        key: &str,
        seen: &mut Vec<Handle>,
    ) -> Result<Option<String>, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            // A primitive-wrapper object (`new Number`/`String`/`Boolean`) serializes
            // as its boxed primitive.
            if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
                return self.json_to_string_seen(prim, key, seen);
            }
            // A `Date` serializes as its ISO string (its built-in `toJSON`).
            if let Some(ms) = self.realm.date_at(h) {
                return Ok(Some(if ms.is_finite() {
                    json_quote(&crate::realm::date_to_iso(ms))
                } else {
                    String::from("null")
                }));
            }
            // A `BigInt` cannot be serialized to JSON.
            if self.realm.bigint_at(h).is_some() {
                let m = self.new_str("Do not know how to serialize a BigInt");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        // A `toJSON` method replaces the value before serialization.
        if let Some(h) = v.as_handle().map(Handle::from_raw)
            && self.realm.string_value(h).is_none()
            && !self.realm.is_array(h)
            && self.realm.object_keys(h).is_some()
        {
            let tj = self.read_member(h, "toJSON")?;
            if tj
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let key_box = self.new_str(key);
                let r = self.call_with_this(tj, v, &[key_box])?;
                return self.json_to_string_seen(r, key, seen);
            }
        }
        match v.unpack() {
            Unpacked::Undefined => Ok(None),
            Unpacked::Null => Ok(Some(String::from("null"))),
            Unpacked::Bool(b) => Ok(Some(String::from(if b { "true" } else { "false" }))),
            // Use the spec ToString (`0` for `-0`, exponential for ≥ 1e21, …);
            // non-finite numbers serialize as `null`.
            Unpacked::Number(n) => Ok(Some(if n.is_finite() {
                self.realm.to_display_string(v)
            } else {
                String::from("null")
            })),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if let Some(bytes) = self.realm.string_bytes(h) {
                    return Ok(Some(json_quote_wtf8(&bytes)));
                }
                // A container that is already an ancestor is a cycle → TypeError.
                if (self.realm.array_elements(h).is_some() || self.realm.object_keys(h).is_some())
                    && seen.contains(&h)
                {
                    let m = self.new_str("Converting circular structure to JSON");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A typed array serializes as an object keyed by its indices
                // (`{"0":1,…}`), per ES `JSON.stringify` (it is not an Array exotic).
                if let Some(elems) = self.realm.typed_elements(h) {
                    let mut parts = Vec::with_capacity(elems.len());
                    for (i, e) in elems.into_iter().enumerate() {
                        if let Some(s) =
                            self.json_to_string_seen(e, &alloc::format!("{i}"), seen)?
                        {
                            parts.push(alloc::format!(
                                "{}:{}",
                                json_quote(&alloc::format!("{i}")),
                                s
                            ));
                        }
                    }
                    return Ok(Some(alloc::format!("{{{}}}", parts.join(","))));
                }
                if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
                    seen.push(h);
                    let mut parts = Vec::with_capacity(elems.len());
                    for (i, e) in elems.into_iter().enumerate() {
                        parts.push(
                            self.json_to_string_seen(e, &alloc::format!("{i}"), seen)?
                                .unwrap_or_else(|| String::from("null")),
                        );
                    }
                    seen.pop();
                    return Ok(Some(alloc::format!("[{}]", parts.join(","))));
                }
                if self.realm.object_keys(h).is_some() {
                    // Enumerable keys (incl. accessors), read via read_member so
                    // getters are invoked.
                    let keys = self.realm.object_keys(h).unwrap_or_default();
                    seen.push(h);
                    let mut parts = Vec::new();
                    for k in keys {
                        let val = self.read_member(h, &k)?;
                        if let Some(s) = self.json_to_string_seen(val, &k, seen)? {
                            parts.push(alloc::format!("{}:{}", json_quote(&k), s));
                        }
                    }
                    seen.pop();
                    return Ok(Some(alloc::format!("{{{}}}", parts.join(","))));
                }
                Ok(None) // a function
            }
        }
    }
}
