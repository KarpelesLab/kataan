use super::*;
use crate::realm::json_fast::{SHORT_TEXT, push_padded};
use alloc::rc::Rc;

impl Interp {
    /// A typed array's own enumerable *named* String keys (`ta.foo = 1`), which
    /// follow its integer indices in `EnumerableOwnProperties`.
    fn typed_array_named_keys(&self, h: Handle) -> Vec<String> {
        self.realm
            .object_keys(h)
            .unwrap_or_else(|| self.realm.aux_named_keys(h))
            .into_iter()
            .filter(|k| !crate::realm::is_internal_key(k))
            .collect()
    }

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
    /// `snapshot` is the parse-time source node (its spans into the parsed
    /// `cx.text`) for the value currently at
    /// `holder[key]` (or `None` for a member the reviver added).
    pub(crate) fn json_revive_ctx(
        &mut self,
        holder: crate::heap::Handle,
        key: &str,
        reviver: NanBox,
        snapshot: Option<&JsonSrc>,
        cx: &mut ReviveCx<'_>,
    ) -> Result<NanBox, ExecError> {
        // The walk recurses once per nesting level of the (possibly
        // reviver-grown) value: bounded by the real stack headroom.
        if crate::stack::exhausted() {
            return Err(self.json_too_deep());
        }
        // `Get(holder, name)`: an own plain data property is read straight
        // from its slot (no user code can observe that), anything else takes
        // the full `[[Get]]`.
        let value = self.get_own_fast(holder, key)?;
        // An object walked is rooted while its members are revived (a reviver
        // may detach it from `holder`): the walk runs in `JSON.parse`'s root
        // scope, and each reviver call is a `call_cb`, so an allocating
        // reviver may collect (see `host_roots`). Its own reviver call is
        // passed it (the VM roots a call's arguments).
        let mut mark = None;
        if let Some(vh) = value.as_handle().map(Handle::from_raw) {
            mark = Some(self.root_slot(value));
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
                    let nv = self.json_revive_ctx(vh, &ks, reviver, child, cx)?;
                    self.json_revive_store(vh, &ks, Some(i), nv)?;
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
                    let mut ks: Vec<String> = (0..len).map(|i| alloc::format!("{i}")).collect();
                    ks.extend(self.typed_array_named_keys(vh));
                    ks
                } else {
                    self.realm
                        .object_keys(self.proxy_key_target(vh))
                        .unwrap_or_default()
                };
                // Each key's snapshot member is its first occurrence in the
                // source; a wide object indexes them once (a scan of every
                // member per key was quadratic).
                let index: Option<alloc::collections::BTreeMap<&str, usize>> = match snapshot {
                    Some(JsonSrc::Object(pairs)) if pairs.len() > 8 => {
                        let mut m = alloc::collections::BTreeMap::new();
                        for (i, (pk, _)) in pairs.iter().enumerate() {
                            m.entry(pk.as_str()).or_insert(i);
                        }
                        Some(m)
                    }
                    _ => None,
                };
                for k in keys {
                    let child = match snapshot {
                        Some(JsonSrc::Object(pairs)) => match &index {
                            Some(m) => m.get(k.as_str()).map(|&i| &pairs[i].1),
                            None => pairs.iter().find(|(pk, _)| pk == &k).map(|(_, s)| s),
                        },
                        _ => None,
                    };
                    let nv = self.json_revive_ctx(vh, &k, reviver, child, cx)?;
                    self.json_revive_store(vh, &k, None, nv)?;
                }
            }
        }
        // Build the `context` object. Only a primitive value that is *still* the
        // one the parser produced here (SameValue) exposes its `source`.
        let context = self.realm.new_object();
        if !self.is_object_value(value)
            && let Some(&JsonSrc::Prim {
                value: pv,
                span: (a, b),
            }) = snapshot
            && let Some(source) = cx.text.get(a..b)
            && self.json_src_same(pv, value, source)
        {
            let src_box = self.new_str_bytes(source.to_vec());
            if !self
                .realm
                .init_data_property_cached(context, "source", src_box, &mut cx.source_ic)
            {
                self.realm.set_property(context, "source", src_box);
            }
        }
        let kb = self.new_key_str(key);
        if let Some(m) = mark {
            self.root_truncate(m);
        }
        let args = [kb, value, NanBox::handle(context.to_raw())];
        let holder = NanBox::handle(holder.to_raw());
        if cx.reviver_vm && self.created_realms.is_empty() {
            return self.call_vm_cb(reviver, holder, &args);
        }
        self.call_cb(reviver, holder, &args)
    }

    /// `SameValue(pv, value)` of the primitive `pv` a source record holds
    /// (whose JSON text is `source`) and the current `value`. A string the
    /// parser made is not rooted during the walk (a reviver may drop the
    /// last reference to it, and collect): it is used only when it is the
    /// very value at hand, else its text is parsed again.
    fn json_src_same(&mut self, pv: NanBox, value: NanBox, source: &[u8]) -> bool {
        if pv.to_bits() == value.to_bits() {
            return true;
        }
        if pv.as_handle().is_none() {
            return self.realm.same_value(pv, value);
        }
        if !value
            .as_handle()
            .is_some_and(|h| self.realm.is_string_handle(Handle::from_raw(h)))
        {
            return false;
        }
        match self.json_parse_wtf8(source) {
            Ok(fresh) => self.realm.same_value(fresh, value),
            Err(_) => false,
        }
    }

    /// The step of `InternalizeJSONProperty` that puts a revived element
    /// back: `newElement` undefined deletes `key` from `holder`, anything else
    /// is `CreateDataProperty(holder, key, newElement)` (whose failure is
    /// ignored). `index` is `key` as an array index when `holder` was walked
    /// as an array.
    fn json_revive_store(
        &mut self,
        holder: Handle,
        key: &str,
        index: Option<usize>,
        value: NanBox,
    ) -> Result<(), ExecError> {
        if matches!(value.unpack(), Unpacked::Undefined) {
            self.delete_property_of(holder, key)?;
            return Ok(());
        }
        if self.json_store_existing_plain(holder, key, index, value) {
            return Ok(());
        }
        let desc = self.realm.new_object();
        self.realm.set_property(desc, "value", value);
        self.realm
            .set_property(desc, "writable", NanBox::boolean(true));
        self.realm
            .set_property(desc, "enumerable", NanBox::boolean(true));
        self.realm
            .set_property(desc, "configurable", NanBox::boolean(true));
        self.apply_descriptor(holder, key, desc, true)?;
        Ok(())
    }

    /// `CreateDataProperty(holder, key, value)` when it is just a store into
    /// an existing slot: `key` names a present writable, enumerable,
    /// configurable own data property of an ordinary object or an ordinary
    /// array (no proxy, no exotic or host-observed object, no global object,
    /// no mapped `arguments`, no tracked built-in prototype), so the
    /// `[[DefineOwnProperty]]` only replaces the value. `false`, having
    /// changed nothing, for anything else.
    fn json_store_existing_plain(
        &mut self,
        holder: Handle,
        key: &str,
        index: Option<usize>,
        value: NanBox,
    ) -> bool {
        if self.realm.is_host_exotic(holder) || self.realm.is_dispatch_tracked_proto(holder) {
            return false;
        }
        if let Some(i) = index
            && self.realm.is_plain_array_cell(holder)
        {
            return !self.realm.array_has_index_overrides(holder)
                && self
                    .realm
                    .array_elements(holder)
                    .is_some_and(|a| a.get(i).is_some_and(|v| !v.is_hole()))
                && self.realm.set_element(holder, i, value);
        }
        if !self.realm.is_plain_object(holder)
            || self.is_mapped_arguments(holder)
            || crate::nbvm::VmHost::is_global_object(self, NanBox::handle(holder.to_raw()))
        {
            return false;
        }
        self.realm.store_existing_plain_data(holder, key, value)
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
    ) -> Result<Option<Vec<u8>>, ExecError> {
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
                        Unpacked::Number(_) => Some(self.realm.display_key(e)),
                        Unpacked::Handle(raw) => {
                            let h = Handle::from_raw(raw);
                            if self.realm.is_string_handle(h) {
                                self.realm.string_key(h)
                            } else if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
                                // A String/Number wrapper contributes its key via
                                // ToString/ToNumber (honoring a user valueOf/toString).
                                // A Number *or* String wrapper's key is `ToString(v)`
                                // (per PropertyList construction) — not ToNumber.
                                match prim.unpack() {
                                    Unpacked::Number(_) => Some(crate::realm::user_key_wtf8_owned(
                                        self.coerce_to_string_bytes(e)?,
                                    )),
                                    Unpacked::Handle(pr)
                                        if self
                                            .realm
                                            .string_value(Handle::from_raw(pr))
                                            .is_some() =>
                                    {
                                        Some(crate::realm::user_key_wtf8_owned(
                                            self.coerce_to_string_bytes(e)?,
                                        ))
                                    }
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    // The list holds storage keys (see `realm::is_internal_key`).
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
        // The wrapper holder `{ "": value }`: only a replacer function sees it
        // (as its `this`), and its `[[Get]]` of `""` is `value` itself. With
        // none, no step reads the holder, so none is made.
        let holder = if self.is_callable_value(replacer_fn) {
            let holder = self.realm.new_object();
            self.realm.set_property(holder, "", value);
            Some(holder)
        } else {
            None
        };
        // The arguments and the holder are rooted for the whole walk, and
        // every object or array being serialized (the cycle stack) before any
        // `toJSON`, replacer or getter call (`json_root_stack`): those calls
        // are `call_cb`s, so an allocating one may collect (see `host_roots`).
        // A value such a call returns needs no root of its own: it is passed
        // to the next call (whose arguments the VM roots) or entered on the
        // stack before any further call.
        let holder_v = holder.map_or(NanBox::undefined(), |h| NanBox::handle(h.to_raw()));
        let mut sc = self.root_scope(&[value, replacer, space, holder_v]);
        let this = &mut *sc;
        let stack_base = this.realm.host_roots.vals.len();
        let mut cx = JsonOut {
            replacer_fn,
            property_list: property_list.as_deref(),
            gap: &gap,
            stack: Vec::new(),
            deep: alloc::collections::BTreeSet::new(),
            out: Vec::with_capacity(this.json_out_hint),
            shape_keys: core::mem::take(&mut this.json_shape_memo),
            tojson_obj: crate::ic::PropertyCache::new(),
            tojson_arr: Default::default(),
            fast: replacer_fn.as_handle().is_none() && property_list.is_none() && gap.is_empty(),
            max_len: this.realm.limits.max_string_len,
            slow: 0,
            proofs: this.json_proofs,
            last_plan: None,
            stack_base,
            stack_rooted: 0,
        };
        let wrote = match holder {
            Some(holder) => match this.read_member(holder, "") {
                Ok(value) => this.json_ser_got(&mut cx, holder, JsonKey::Str(""), value, ""),
                Err(e) => Err(e),
            },
            // No step dereferences the holder without a replacer function:
            // a handle to no cell stands in for it.
            None => this.json_ser_got(&mut cx, NO_HOLDER, JsonKey::Str(""), value, ""),
        };
        this.json_shape_memo = core::mem::take(&mut cx.shape_keys);
        this.json_proofs = cx.proofs;
        // The next output buffer starts at this one's size (small outputs
        // then grow in place at most rarely; a huge one is not kept).
        this.json_out_hint = cx.out.len().clamp(64, 4096);
        Ok(wrote?.then_some(cx.out))
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
        value: NanBox,
        indent: &str,
    ) -> Result<bool, ExecError> {
        // With no replacer function, a primitive that is no handle (no
        // `toJSON`, nothing to unwrap) is its text at once, as
        // `json_ser_value` writes it, and a String primitive (no `toJSON`
        // lookup, no wrapper or RawJSON probes) is quoted: here, ahead of
        // the general steps (out of line, so these cheap cases do not pay
        // for their frame). In the fast mode (no property list or gap
        // either), plain objects and arrays are written here too.
        if cx.fast {
            if self.realm.json_write_prim(&mut cx.out, value, cx.max_len) {
                return Ok(true);
            }
            if let Some(raw) = value.as_handle()
                && let Some(done) = self.json_fast_container(cx, Handle::from_raw(raw))
            {
                return done.map(|()| true);
            }
        } else if cx.replacer_fn.as_handle().is_none() {
            match value.unpack() {
                Unpacked::Number(n) => {
                    if n.is_finite() {
                        crate::realm::builtins_fast::push_number_bytes(&mut cx.out, n);
                    } else {
                        cx.out.extend_from_slice(b"null");
                    }
                    return Ok(true);
                }
                Unpacked::Bool(b) => {
                    cx.out.extend_from_slice(if b { b"true" } else { b"false" });
                    return Ok(true);
                }
                Unpacked::Null => {
                    cx.out.extend_from_slice(b"null");
                    return Ok(true);
                }
                Unpacked::Handle(raw) if self.realm.typeof_is_string(Handle::from_raw(raw)) => {
                    self.json_quote_string(cx, Handle::from_raw(raw))?;
                    return Ok(true);
                }
                _ => {}
            }
        }
        self.json_ser_got_full(cx, holder, key, value, indent)
    }

    /// [`json_ser_got`](Self::json_ser_got)'s general steps.
    #[inline(never)]
    fn json_ser_got_full(
        &mut self,
        cx: &mut JsonOut<'_>,
        holder: Handle,
        key: JsonKey<'_>,
        mut value: NanBox,
        indent: &str,
    ) -> Result<bool, ExecError> {
        cx.slow += 1;
        // With no replacer function, an ordinary dense array with no named
        // own properties (so no own `toJSON`) whose prototype chain is known
        // to lack `toJSON` is serialized as an array — exactly what the
        // steps below conclude for it.
        if cx.replacer_fn.as_handle().is_none()
            && let Some(h) = value.as_handle().map(Handle::from_raw)
            && self.realm.is_plain_array_cell(h)
            && !self.realm.is_host_exotic(h)
            && !self.realm.may_own_named(h)
            && self
                .realm
                .object_proto(h)
                .is_some_and(|proto| self.realm.chain_absent_cached(proto, &cx.tojson_arr))
        {
            self.json_ser_array(cx, h, indent)?;
            return Ok(true);
        }
        // If value is an Object (or BigInt) with a callable `toJSON`, call it.
        if let Some(h) = value.as_handle().map(Handle::from_raw) {
            let tj = self.json_to_json_of(cx, h)?;
            if self.is_callable_value(tj) {
                let kb = key.to_value(self);
                self.json_root_stack(cx);
                value = self.call_cb(tj, value, &[kb])?;
            }
        }
        // ReplacerFunction: value = replacer.call(holder, key, value).
        if self.is_callable_value(cx.replacer_fn) {
            let kb = key.to_value(self);
            self.json_root_stack(cx);
            value = self.call_cb(
                cx.replacer_fn,
                NanBox::handle(holder.to_raw()),
                &[kb, value],
            )?;
        }
        self.json_ser_value(cx, value, indent)
    }

    /// `QuoteJSONString` of the string cell `h`, appended to `cx.out`.
    /// Throws the string-length RangeError when the quoted text would take
    /// the output past the cap (checked before the string is copied in).
    fn json_quote_string(&mut self, cx: &mut JsonOut<'_>, h: Handle) -> Result<(), ExecError> {
        let n = self.realm.string_byte_len(h).unwrap_or(0);
        self.check_str_len(cx.out.len() + n + 2)?;
        // A flat string is quoted from its leaf at once.
        if let Some(bytes) = self.realm.string_leaf_bytes(h) {
            json_quote_wtf8_into(&mut cx.out, bytes);
            return self.check_str_len(cx.out.len());
        }
        // Collapse a rope once so the quoting borrows one leaf.
        self.realm.flatten_string(h);
        if let Some(bytes) = self.realm.string_leaf_bytes(h) {
            json_quote_wtf8_into(&mut cx.out, bytes);
        } else if let Some(bytes) = self.realm.string_bytes(h) {
            json_quote_wtf8_into(&mut cx.out, &bytes);
        }
        self.check_str_len(cx.out.len())
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
        // A plain data object (one with a key plan: an ordinary shaped object
        // with no internal key, so no RawJSON brand, no boxed primitive, no
        // callable marker, and not an array or proxy) is an object to serialize.
        if let Some(h) = value.as_handle().map(Handle::from_raw)
            && let Some(plan) = self.json_shape_keys(cx, h)
        {
            if cx.property_list.is_none() {
                self.json_ser_planned(cx, h, &plan, indent)?;
            } else {
                self.json_ser_object(cx, h, indent)?;
            }
            return Ok(true);
        }
        let value = if let Some(h) = value.as_handle().map(Handle::from_raw) {
            // RawJSON object: emit the stored source verbatim.
            if self.realm.get_property(h, RAW_JSON_BRAND).is_some()
                && let Some(raw) = self.realm.get_property(h, "rawJSON")
            {
                match raw
                    .as_handle()
                    .and_then(|r| self.realm.string_bytes(Handle::from_raw(r)))
                {
                    Some(bytes) => cx.out.extend_from_slice(&bytes),
                    None => cx
                        .out
                        .extend_from_slice(self.realm.to_display_string(raw).as_bytes()),
                }
                return Ok(true);
            }
            if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
                // SerializeJSONProperty step 4: a `[[NumberData]]` wrapper is
                // `ToNumber(value)` and a `[[StringData]]` wrapper is
                // `ToString(value)` — both applied to the *wrapper*, so a custom
                // `valueOf`/`toString` is honored. Boolean/BigInt wrappers use the
                // boxed primitive directly. A Symbol wrapper has none of those
                // slots: it is an ordinary object (`{}`), not a dropped symbol.
                match prim.unpack() {
                    Unpacked::Number(_) => self.coerce_to_number(value)?,
                    Unpacked::Handle(r) if self.realm.is_string_handle(Handle::from_raw(r)) => {
                        let s = self.coerce_to_string(value)?;
                        self.new_str(&s)
                    }
                    Unpacked::Handle(r) if self.realm.symbol_at(Handle::from_raw(r)).is_some() => {
                        value
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
            Unpacked::Null => cx.out.extend_from_slice(b"null"),
            Unpacked::Bool(b) => cx.out.extend_from_slice(if b { b"true" } else { b"false" }),
            Unpacked::Number(n) => {
                if n.is_finite() {
                    crate::realm::builtins_fast::push_number_bytes(&mut cx.out, n);
                } else {
                    cx.out.extend_from_slice(b"null");
                }
            }
            Unpacked::Undefined => return Ok(false),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if self.realm.is_string_handle(h) {
                    self.json_quote_string(cx, h)?;
                    return Ok(true);
                }
                if self.realm.is_bigint(h) {
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

    /// `GetV(value, "toJSON")` for a value at handle `h`, with two validated
    /// caches in front of the full `[[Get]]`:
    /// - an ordinary shaped object uses the prototype-chain cache the VM's
    ///   property reads use (`ic::ChainCache`), whose hit is exactly the
    ///   ordinary `[[Get]]` walk over a chain of plain shaped objects;
    /// - an array (whose own named properties live outside any shape)
    ///   checks it has no own `toJSON` and then asks an
    ///   [`ChainKeyCache`](crate::realm::builtins_fast::ChainKeyCache)
    ///   whether its prototype chain still lacks the key.
    fn json_to_json_of(&mut self, cx: &mut JsonOut<'_>, h: Handle) -> Result<NanBox, ExecError> {
        const KEY: &str = "toJSON";
        // SerializeJSONProperty step 2 looks `toJSON` up only on an Object or a
        // BigInt: a String or Symbol primitive never consults it.
        if self.realm.typeof_is_string(h) || self.realm.symbol_at(h).is_some() {
            return Ok(NanBox::undefined());
        }
        if self.realm.is_plain_array_cell(h)
            && !self.realm.is_host_exotic(h)
            && !self.realm.has_own(h, KEY)
        {
            let Some(proto) = self.realm.object_proto(h) else {
                return Ok(NanBox::undefined());
            };
            let cache = &mut cx.tojson_arr;
            if self.realm.chain_absent_cached(proto, cache) {
                return Ok(NanBox::undefined());
            }
            self.realm.chain_key_fill(proto, KEY, cache);
            return self.read_member(h, KEY);
        }
        if self.realm.plain_keyed_shape(h).is_none() || self.is_mapped_arguments(h) {
            return self.read_member(h, KEY);
        }
        let cache = &mut cx.tojson_obj;
        match self.realm.chain_cached_get(h, KEY, cache) {
            Some(crate::realm::ChainHit::Data(v)) => Ok(v),
            Some(crate::realm::ChainHit::Absent) => Ok(NanBox::undefined()),
            Some(crate::realm::ChainHit::Accessor(getter, _, own)) => {
                if getter.as_handle().is_some() {
                    self.call_with_this(getter, NanBox::handle(h.to_raw()), &[])
                } else if !own {
                    Ok(NanBox::undefined())
                } else {
                    self.read_member(h, KEY)
                }
            }
            None => {
                self.realm.chain_cache_fill(h, KEY, cache);
                self.read_member(h, KEY)
            }
        }
    }

    /// The fast path of [`json_ser_got`](Self::json_ser_got) (no replacer,
    /// no property list, no gap) for an object or array whose own steps run
    /// no user code: a plain shaped data object owning no `toJSON`, or an
    /// array with no named or exotic own properties, either with a
    /// prototype chain shown to lack `toJSON`. `Get(value, "toJSON")` is
    /// then `undefined` with no side effect, and the value is no wrapper,
    /// RawJSON object, callable or proxy, so SerializeJSONProperty goes
    /// straight to SerializeJSONObject / SerializeJSONArray. `Some` once
    /// written, `None` (nothing written) for anything else.
    fn json_fast_container(
        &mut self,
        cx: &mut JsonOut<'_>,
        h: Handle,
    ) -> Option<Result<(), ExecError>> {
        if let Some((shape, proto)) = self.realm.json_plain_object(h) {
            let hit = cx
                .last_plan
                .as_ref()
                .filter(|p| Rc::ptr_eq(&p.shape, shape))
                .map(Rc::clone);
            let plan = match hit {
                Some(p) => p,
                None => {
                    let p = self.json_shape_keys(cx, h)?;
                    cx.last_plan = Some(Rc::clone(&p));
                    p
                }
            };
            if plan.own_tojson
                || self.is_mapped_arguments(h)
                || self.realm.is_function_proto_intrinsic(h)
                || !self.json_chain_lacks_tojson(cx, proto)
            {
                return None;
            }
            return Some(self.json_fast_object(cx, h, &plan));
        }
        let proto = self.realm.json_plain_array(h)?;
        if !self.json_chain_lacks_tojson(cx, proto) {
            return None;
        }
        Some(self.json_fast_array(cx, h))
    }

    /// Whether the `[[Get]]` of `toJSON` from the prototype `proto` finds
    /// nothing, remembered per prototype for as long as the prototype epoch
    /// it was shown at is current (every link of the chain is watched, so
    /// any change to one moves the epoch).
    fn json_chain_lacks_tojson(&mut self, cx: &mut JsonOut<'_>, proto: Option<Handle>) -> bool {
        let Some(p) = proto else {
            return true;
        };
        let raw = p.to_raw();
        let epoch = crate::object::proto_epoch();
        if cx.proofs.iter().any(|&(r, e)| r == raw && e == epoch) {
            return true;
        }
        if !self.realm.prove_chain_lacks(Some(p), "toJSON") {
            return false;
        }
        let epoch = crate::object::proto_epoch();
        let slot = cx.proofs.iter().position(|&(_, e)| e != epoch).unwrap_or(0);
        cx.proofs[slot] = (raw, epoch);
        true
    }

    /// SerializeJSONObject (compact, no property list) for an object of
    /// `plan`'s shape (see [`json_fast_container`](Self::json_fast_container)).
    /// Its values are copied out of the slots a chunk at a time and stay
    /// valid while no general step has run since (`cx.slow`): only those can
    /// run user code that changes the object. After one has, the next value
    /// is re-read the way [`json_ser_planned`](Self::json_ser_planned) does.
    fn json_fast_object(
        &mut self,
        cx: &mut JsonOut<'_>,
        h: Handle,
        plan: &JsonShapeKeys,
    ) -> Result<(), ExecError> {
        const CHUNK: usize = 16;
        self.json_enter(cx, h)?;
        cx.out.push(b'{');
        let mut any = false;
        let n = plan.keys.len();
        let mut buf = [NanBox::undefined(); CHUNK];
        // `buf[..cnt]` holds the values of keys `base..base + cnt`, read
        // when the general-step count was `at`.
        let (mut base, mut cnt, mut at) = (0, 0, cx.slow);
        for i in 0..n {
            let (k, quoted, _) = &plan.keys[i];
            let value = if i < base + cnt && cx.slow == at {
                buf[i - base]
            } else {
                let c = CHUNK.min(n - i);
                if self
                    .realm
                    .json_slots_under(h, &plan.shape, &plan.slots[i..i + c], &mut buf[..c])
                {
                    (base, cnt, at) = (i, c, cx.slow);
                    buf[0]
                } else {
                    // Reshaped by user code: the full `[[Get]]` (which may
                    // itself run a getter).
                    cnt = 0;
                    cx.slow += 1;
                    self.json_get_cb(cx, h, k)?
                }
            };
            let mark = cx.out.len();
            match plan.texts[i] {
                (ref first, ref next, len) if len > 0 => {
                    if any {
                        push_padded(&mut cx.out, next, len + 1);
                    } else {
                        push_padded(&mut cx.out, first, len);
                    }
                }
                _ => {
                    if any {
                        cx.out.push(b',');
                    }
                    cx.out.extend_from_slice(quoted);
                }
            }
            if cx.out.len() > cx.max_len {
                return Err(self.string_length_error());
            }
            if self.realm.json_write_prim(&mut cx.out, value, cx.max_len)
                || self.json_ser_got(cx, h, JsonKey::Str(k), value, "")?
            {
                any = true;
            } else {
                cx.out.truncate(mark);
            }
        }
        self.json_leave(cx);
        cx.out.push(b'}');
        Ok(())
    }

    /// SerializeJSONArray (compact) for a plain array (see
    /// [`json_fast_container`](Self::json_fast_container)): runs of
    /// elements that need nothing but their text are written straight from
    /// the element store ([`Realm::json_write_array_run`]); any other
    /// element takes the steps of [`json_ser_array`](Self::json_ser_array).
    fn json_fast_array(&mut self, cx: &mut JsonOut<'_>, h: Handle) -> Result<(), ExecError> {
        self.json_enter(cx, h)?;
        let len = self.realm.array_length(h).unwrap_or(0);
        cx.out.push(b'[');
        let mut i = 0;
        while i < len {
            i = self
                .realm
                .json_write_array_run(h, i, len, &mut cx.out, cx.max_len);
            if i >= len {
                break;
            }
            if i > 0 {
                cx.out.push(b',');
                self.check_str_len(cx.out.len())?;
            }
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
                None => {
                    cx.slow += 1;
                    self.read_member(h, &alloc::format!("{i}"))?
                }
            };
            if !self.json_ser_got(cx, h, JsonKey::Index(i), value, "")? {
                cx.out.extend_from_slice(b"null");
            }
            i += 1;
        }
        self.json_leave(cx);
        cx.out.push(b']');
        self.check_str_len(cx.out.len())
    }

    /// The memoized key plan for the object at `h` (see [`JsonShapeKeys`]),
    /// built on the first object of each shape; `None` when `h` is not a plain
    /// keyed object or its shape holds an engine-internal key.
    fn json_shape_keys(&mut self, cx: &mut JsonOut<'_>, h: Handle) -> Option<Rc<JsonShapeKeys>> {
        if self.is_mapped_arguments(h) || self.realm.is_function_proto_intrinsic(h) {
            return None;
        }
        let shape = self.realm.plain_keyed_shape(h)?;
        if let Some(m) = cx
            .shape_keys
            .iter()
            .rev()
            .find(|m| Rc::ptr_eq(&m.shape, shape))
        {
            return Some(Rc::clone(m));
        }
        let shape = Rc::clone(shape);
        // An internal key (a wrapper's boxed primitive, a brand, a private
        // element, a symbol) marks an object the generic path must handle.
        if shape
            .keys()
            .iter()
            .any(|k| crate::realm::is_internal_key(k))
        {
            return None;
        }
        let names = self.realm.object_keys(h)?;
        let mut keys = Vec::with_capacity(names.len());
        for k in names {
            let slot = shape.lookup(&k)?;
            let mut quoted = Vec::with_capacity(k.len() + 3);
            json_quote_wtf8_into(&mut quoted, &crate::realm::key_wtf8(&k));
            quoted.push(b':');
            keys.push((k, quoted, slot));
        }
        let slots = keys.iter().map(|k| k.2).collect();
        let own_tojson = keys.iter().any(|k| k.0 == "toJSON");
        let texts = keys
            .iter()
            .map(|(_, q, _)| {
                let (mut first, mut next) = ([0; SHORT_TEXT], [0; SHORT_TEXT]);
                if q.len() < SHORT_TEXT {
                    first[..q.len()].copy_from_slice(q);
                    next[0] = b',';
                    next[1..=q.len()].copy_from_slice(q);
                    (first, next, q.len())
                } else {
                    (first, next, 0)
                }
            })
            .collect();
        let m = Rc::new(JsonShapeKeys {
            shape,
            keys,
            slots,
            texts,
            own_tojson,
        });
        if cx.shape_keys.len() >= JSON_SHAPE_MEMO {
            cx.shape_keys.remove(0);
        }
        cx.shape_keys.push(Rc::clone(&m));
        Some(m)
    }

    /// Roots the objects on the cycle stack not rooted yet, before a call
    /// made through `call_cb` (see `json_stringify`). They are rooted in
    /// stack order on top of the walk's root scope, so leaving one drops its
    /// root (`json_leave`).
    fn json_root_stack(&mut self, cx: &mut JsonOut<'_>) {
        let vals = &mut self.realm.host_roots.vals;
        debug_assert_eq!(vals.len(), cx.stack_base + cx.stack_rooted);
        vals.extend(
            cx.stack[cx.stack_rooted..]
                .iter()
                .map(|h| NanBox::handle(h.to_raw())),
        );
        cx.stack_rooted = cx.stack.len();
    }

    /// `Get(h, key)` of an object being serialized (on the cycle stack, see
    /// `json_root_stack`): an own data property of an ordinary object is its
    /// slot; an own accessor's getter is called through `call_cb` (an
    /// allocating getter may collect, see `host_roots`); anything else takes
    /// the full `[[Get]]`.
    fn json_get_cb(
        &mut self,
        cx: &mut JsonOut<'_>,
        h: Handle,
        key: &str,
    ) -> Result<NanBox, ExecError> {
        if let Some(v) = self.ordinary_own_data(h, key) {
            return Ok(v);
        }
        if self.realm.proxy_at(h).is_none()
            && self.realm.typed_kind(h).is_none()
            && !self.realm.is_host_exotic(h)
            && !self.is_mapped_arguments(h)
            && let Some((getter, _)) = self.realm.accessor(h, key)
            && getter
                .as_handle()
                .is_some_and(|g| self.realm.is_vm_function(Handle::from_raw(g)))
        {
            self.json_root_stack(cx);
            return self.call_cb(getter, NanBox::handle(h.to_raw()), &[]);
        }
        self.read_member(h, key)
    }

    /// The cycle check shared by objects and arrays: pushes `h` on the stack,
    /// or throws if it is already being serialized. The nesting is bounded by
    /// `max_json_depth` (a RangeError past it, as for any runaway recursion):
    /// a replacer or `toJSON` that returns a fresh object at every level never
    /// closes a cycle, and would otherwise recurse without end.
    ///
    /// The membership test is a scan of the stack while it is shallow and a
    /// set lookup past [`JSON_STACK_SCAN`] levels, so a deep structure costs
    /// linear time, not a scan of every enclosing level at each one.
    fn json_enter(&mut self, cx: &mut JsonOut<'_>, h: Handle) -> Result<(), ExecError> {
        let seen = if cx.stack.len() <= JSON_STACK_SCAN {
            cx.stack.contains(&h)
        } else {
            cx.stack[..JSON_STACK_SCAN].contains(&h) || cx.deep.contains(&h.to_raw())
        };
        if seen {
            let m = self.new_str("Converting circular structure to JSON");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        if cx.stack.len() >= self.realm.limits.max_json_depth || crate::stack::exhausted() {
            let m = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        if cx.stack.len() >= JSON_STACK_SCAN {
            cx.deep.insert(h.to_raw());
        }
        cx.stack.push(h);
        Ok(())
    }

    /// Pops the object [`json_enter`](Self::json_enter) pushed.
    fn json_leave(&mut self, cx: &mut JsonOut<'_>) {
        if let Some(h) = cx.stack.pop()
            && cx.stack.len() >= JSON_STACK_SCAN
        {
            cx.deep.remove(&h.to_raw());
        }
        // (Its root, if `json_root_stack` made one.)
        if cx.stack_rooted > cx.stack.len() {
            cx.stack_rooted = cx.stack.len();
            self.root_truncate(cx.stack_base + cx.stack_rooted);
        }
    }

    /// `SerializeJSONObject` — `{ … }` with the PropertyList (or own enumerable
    /// keys), recursing per member; cycle-checked via the stack.
    fn json_ser_object(
        &mut self,
        cx: &mut JsonOut<'_>,
        h: Handle,
        indent: &str,
    ) -> Result<(), ExecError> {
        if cx.property_list.is_none()
            && let Some(plan) = self.json_shape_keys(cx, h)
        {
            return self.json_ser_planned(cx, h, &plan, indent);
        }
        self.json_enter(cx, h)?;
        // (No indent to build without a gap: `String::new()` does not allocate.)
        let new_indent = if cx.gap.is_empty() {
            String::new()
        } else {
            alloc::format!("{indent}{}", cx.gap)
        };
        // The key set: the explicit PropertyList, else the object's own enumerable
        // string keys (a typed array enumerates its integer indices).
        let keys: Vec<String> = match cx.property_list {
            Some(list) => list.to_vec(),
            None => {
                if let Some(len) = self.realm.typed_len(h) {
                    // Its integer indices, then its own enumerable named keys.
                    let mut ks: Vec<String> = (0..len).map(|i| alloc::format!("{i}")).collect();
                    ks.extend(self.typed_array_named_keys(h));
                    ks
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
        cx.out.push(b'{');
        let mut any = false;
        for k in keys {
            let mark = cx.out.len();
            if any {
                cx.out.push(b',');
                self.check_str_len(cx.out.len())?;
            }
            if pretty {
                cx.out.push(b'\n');
                cx.out.extend_from_slice(new_indent.as_bytes());
            }
            json_quote_wtf8_into(&mut cx.out, &crate::realm::key_wtf8(&k));
            cx.out.extend_from_slice(if pretty { b": " } else { b":" });
            // `Get(h, k)`: an own data property of an ordinary object is its
            // slot; everything else takes the full `[[Get]]`.
            let value = self.json_get_cb(cx, h, &k)?;
            if self.json_ser_got(cx, h, JsonKey::Str(&k), value, &new_indent)? {
                any = true;
            } else {
                cx.out.truncate(mark);
            }
        }
        self.json_leave(cx);
        if any && pretty {
            cx.out.push(b'\n');
            cx.out.extend_from_slice(indent.as_bytes());
        }
        cx.out.push(b'}');
        Ok(())
    }

    /// [`json_ser_object`](Self::json_ser_object) for an object with a key
    /// plan: the same steps, with the keys and their quoted forms from the plan
    /// and each `Get(h, k)` a slot read while `h` still has the plan's shape
    /// (user code run by a nested `toJSON`/replacer may reshape it, after which
    /// the remaining keys take the generic `[[Get]]`).
    fn json_ser_planned(
        &mut self,
        cx: &mut JsonOut<'_>,
        h: Handle,
        plan: &JsonShapeKeys,
        indent: &str,
    ) -> Result<(), ExecError> {
        self.json_enter(cx, h)?;
        let pretty = !cx.gap.is_empty();
        let new_indent = if pretty {
            alloc::format!("{indent}{}", cx.gap)
        } else {
            String::new()
        };
        cx.out.push(b'{');
        let mut any = false;
        for (k, quoted, slot) in &plan.keys {
            let mark = cx.out.len();
            if any {
                cx.out.push(b',');
                self.check_str_len(cx.out.len())?;
            }
            if pretty {
                cx.out.push(b'\n');
                cx.out.extend_from_slice(new_indent.as_bytes());
            }
            if pretty {
                cx.out.extend_from_slice(&quoted[..quoted.len() - 1]);
                cx.out.extend_from_slice(b": ");
            } else {
                cx.out.extend_from_slice(quoted);
            }
            let value = match self.realm.plain_slot_under(h, &plan.shape, *slot) {
                Some(v) => v,
                None => self.json_get_cb(cx, h, k)?,
            };
            if self.json_ser_got(cx, h, JsonKey::Str(k), value, &new_indent)? {
                any = true;
            } else {
                cx.out.truncate(mark);
            }
        }
        self.json_leave(cx);
        if any && pretty {
            cx.out.push(b'\n');
            cx.out.extend_from_slice(indent.as_bytes());
        }
        cx.out.push(b'}');
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
        let new_indent = if cx.gap.is_empty() {
            String::new()
        } else {
            alloc::format!("{indent}{}", cx.gap)
        };
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
        cx.out.push(b'[');
        for i in 0..len {
            if i > 0 {
                cx.out.push(b',');
                self.check_str_len(cx.out.len())?;
            }
            if pretty {
                cx.out.push(b'\n');
                cx.out.extend_from_slice(new_indent.as_bytes());
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
                cx.out.extend_from_slice(b"null");
            }
        }
        self.json_leave(cx);
        if len > 0 && pretty {
            cx.out.push(b'\n');
            cx.out.extend_from_slice(indent.as_bytes());
        }
        cx.out.push(b']');
        Ok(())
    }

    /// A `SyntaxError` for a malformed `JSON.parse` input (the spec error type, so
    /// `catch (e) { e instanceof SyntaxError }` works).
    pub(crate) fn json_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_ERROR_BASE + 3, Some(m)))
    }

    /// The RangeError of a recursive JSON walk that ran out of native stack
    /// (the input itself may be valid, so it is not a SyntaxError).
    pub(super) fn json_too_deep(&mut self) -> ExecError {
        let m = self.new_str("Maximum call stack size exceeded");
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
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
        if matches!(ch, '[' | '{')
            && (depth >= self.realm.limits.max_json_depth || crate::stack::exhausted())
        {
            return Err(self.json_too_deep());
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
                    // The storage key, lossless (a lone surrogate keeps its identity).
                    let key = crate::realm::user_key_wtf8(&self.json_string(c, pos)?);
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
                    push_json_source_char(ch, self.json_lone_base, &mut out);
                    *pos += 1;
                }
            }
        }
    }
}

/// The text of a reviver-aware `JSON.parse` as `char`s, which cannot hold a lone
/// surrogate: each one becomes a code point of a private-use block the text does
/// not otherwise use. Returns the chars and that block's base (0: no lone
/// surrogate). [`push_json_source_char`] turns the stand-ins back.
pub(crate) fn json_text_chars(text: &[u8]) -> (Vec<char>, u32) {
    if let Ok(s) = core::str::from_utf8(text) {
        return (s.chars().collect(), 0);
    }
    let mut cps: Vec<u32> = Vec::with_capacity(text.len());
    let mut it = crate::wtf8::code_points(text).peekable();
    while let Some(cp) = it.next() {
        if (0xD800..0xDC00).contains(&cp)
            && let Some(&lo) = it.peek()
            && (0xDC00..0xE000).contains(&lo)
        {
            it.next();
            cps.push(0x1_0000 + ((cp - 0xD800) << 10) + (lo - 0xDC00));
        } else {
            cps.push(cp);
        }
    }
    // Planes 15 and 16 hold 64 blocks of the surrogates' size; a text that uses
    // every one of them keeps U+FFFD for its lone surrogates.
    let base = (0..64u32)
        .map(|k| 0xF_0000 + k * 0x1000 / 2)
        .find(|b| !cps.iter().any(|cp| cp.wrapping_sub(*b) < 0x800))
        .unwrap_or(0);
    let chars = cps
        .into_iter()
        .map(|cp| {
            if (0xD800..0xE000).contains(&cp) && base != 0 {
                char::from_u32(base + cp - 0xD800).unwrap_or('\u{FFFD}')
            } else {
                char::from_u32(cp).unwrap_or('\u{FFFD}')
            }
        })
        .collect();
    (chars, base)
}

/// Appends one char of a [`json_text_chars`] text as WTF-8.
fn push_json_source_char(ch: char, lone_base: u32, out: &mut Vec<u8>) {
    let cp = ch as u32;
    if lone_base != 0 && cp.wrapping_sub(lone_base) < 0x800 {
        crate::wtf8::encode_utf16_unit((0xD800 + cp - lone_base) as u16, out);
    } else {
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    }
}

/// A parse-time source-text snapshot mirroring a parsed JSON value's structure,
/// used by the `json-parse-with-source` reviver `context`. Each primitive leaf
/// records the value the parser produced and where its exact source text is so the
/// reviver can be handed the original text (unless a reviver has since replaced
/// the value at that position — checked by SameValue).
pub(crate) enum JsonSrc {
    /// The value made and the byte range of its source in the parsed text.
    Prim {
        value: NanBox,
        span: (usize, usize),
    },
    Array(Vec<JsonSrc>),
    Object(Vec<(String, JsonSrc)>),
}

/// The state of one reviver walk ([`Interp::json_revive_ctx`]): the parsed
/// text the [`JsonSrc`] spans point into, and the transition cache that adds
/// `source` to each fresh `context` object.
pub(crate) struct ReviveCx<'a> {
    pub(crate) text: &'a [u8],
    pub(crate) source_ic: crate::ic::PropertyCache,
    /// The walk's root scope is active and the reviver is a VM function
    /// (`Interp::cb_is_vm`): each call may collect, made at once.
    pub(crate) reviver_vm: bool,
}

/// The state of one `JSON.stringify` run: the replacer, the gap, the cycle
/// stack and the single output buffer every value is appended to.
struct JsonOut<'a> {
    replacer_fn: NanBox,
    property_list: Option<&'a [String]>,
    gap: &'a str,
    stack: Vec<Handle>,
    /// The raw handles of `stack[JSON_STACK_SCAN..]`, for the cycle check of
    /// a deep structure.
    deep: alloc::collections::BTreeSet<u64>,
    out: Vec<u8>,
    /// Key lists memoized per shape (see [`JsonShapeKeys`]); a few entries,
    /// most recently added last.
    shape_keys: Vec<Rc<JsonShapeKeys>>,
    /// Prototype-chain caches for the `toJSON` probe, one per receiver kind
    /// (an ordinary object, an array's prototype).
    tojson_obj: crate::ic::PropertyCache,
    tojson_arr: crate::realm::builtins_fast::ChainKeyCache,
    /// No replacer, no property list and no gap: plain objects and arrays
    /// take [`Interp::json_fast_object`] / [`Interp::json_fast_array`].
    fast: bool,
    /// The string length cap, in bytes of `out`.
    max_len: usize,
    /// How many times the general steps ([`Interp::json_ser_got_full`]),
    /// the only ones that can run user code, were entered: values read
    /// from the heap before the count last moved are still current.
    slow: u64,
    /// Prototypes shown (by [`Realm::prove_chain_lacks`]) to have no
    /// `toJSON` on their chain, each with the prototype epoch it holds at
    /// (kept from one call to the next: a proof is only as good as its
    /// epoch, and every link it covers is watched).
    proofs: [(u64, u64); 4],
    /// The plan [`Interp::json_fast_object`] used last.
    last_plan: Option<Rc<JsonShapeKeys>>,
    /// Where the roots of `stack` start on the realm's host root stack, and
    /// how many of its objects are rooted there (`Interp::json_root_stack`).
    stack_base: usize,
    stack_rooted: usize,
}

/// The holder `JSON.stringify` passes along when there is no replacer
/// function (the only step that observes the holder): a handle naming no
/// heap cell.
const NO_HOLDER: Handle = Handle::from_raw(u64::MAX >> 16);

/// How many enclosing objects the `JSON.stringify` cycle check scans before
/// it keeps the deeper ones in a set (see `Interp::json_enter`).
const JSON_STACK_SCAN: usize = 32;

/// How many shapes' key lists one `JSON.stringify` run remembers.
const JSON_SHAPE_MEMO: usize = 8;

/// The serialization plan for every object of one shape whose own properties
/// are exactly its shape's enumerable data keys
/// ([`Realm::plain_keyed_shape`](crate::realm::Realm::plain_keyed_shape)): the
/// keys in `[[OwnPropertyKeys]]` order, each with its JSON-quoted form and its
/// slot. Shape identity fixes the keys and slots, so the plan holds for any
/// object that still has `shape` when a key is read.
pub(super) struct JsonShapeKeys {
    shape: Rc<crate::shape::Shape>,
    /// `(key, "key": quoted with its colon, slot)`.
    keys: Vec<(String, Vec<u8>, u32)>,
    /// The slots of `keys`, in order.
    slots: Vec<u32>,
    /// For each key whose `"key":` text fits a short buffer: that text
    /// staged for [`push_padded`](crate::realm::json_fast::push_padded),
    /// alone and after a `,`, and its length (`0` when it does not fit).
    texts: Vec<([u8; SHORT_TEXT], [u8; SHORT_TEXT], usize)>,
    /// Whether one of the keys is `toJSON`.
    own_tojson: bool,
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
            JsonKey::Str(k) => interp.new_key_str(k),
            JsonKey::Index(i) => interp.new_str(&alloc::format!("{i}")),
        }
    }
}
