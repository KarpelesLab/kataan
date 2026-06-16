use super::*;

impl<'a> Interp<'a> {
    /// Builds a match-result array from a `Captures` whose spans are **code-unit**
    /// indices into the pre-collected `&[u16]` subject (the native regex model).
    /// Element `i` is capture group `i` (group 0 = whole match), sliced from the
    /// unit buffer and re-encoded to WTF-8 so astral characters and lone
    /// surrogates survive. The result is a real Array (so `Array.isArray`,
    /// `JSON.stringify`, `.length`, and array methods all behave), with `index` /
    /// `input` / `groups` as enumerable own properties. `.index` is a code-unit
    /// index; the `input` argument carries the original subject string.
    /// Builds the match-result Array, optionally attaching an `indices` property
    /// (the ES2022 `d`-flag `MakeIndicesArray`) when `has_indices` is true:
    /// `indices[i]` is `[start, end]` (code-unit offsets) of capture group `i`, or
    /// `undefined` if it did not participate, and `indices.groups` maps each named
    /// group to its `[start, end]` pair (a null-prototype object, or `undefined`
    /// when there are no named groups).
    #[cfg(feature = "regex")]
    pub(crate) fn regex_match_object_u16_indices(
        &mut self,
        units: &[u16],
        input: NanBox,
        caps: &crate::regex::Captures,
        group_names: &[(usize, String)],
        has_indices: bool,
    ) -> NanBox {
        let elems: Vec<NanBox> = caps
            .groups
            .iter()
            .map(|g| match g {
                Some((s, e)) => self.new_str_bytes(u16_slice(units, *s, *e)),
                None => NanBox::undefined(),
            })
            .collect();
        let obj = self.realm.new_array(elems);
        let index = caps.groups.first().and_then(|g| *g).map_or(0, |(s, _)| s);
        self.realm
            .set_property(obj, "index", NanBox::number(index as f64));
        self.realm.set_property(obj, "input", input);
        let groups = if group_names.is_empty() {
            NanBox::undefined()
        } else {
            // Named-group container is a null-prototype object (per spec).
            let g = self.realm.new_object_with_proto(None);
            for (idx, name) in group_names {
                let v = match caps.groups.get(*idx).and_then(|x| *x) {
                    Some((s, e)) => self.new_str_bytes(u16_slice(units, s, e)),
                    None => NanBox::undefined(),
                };
                self.realm.set_property(g, name, v);
            }
            NanBox::handle(g.to_raw())
        };
        self.realm.set_property(obj, "groups", groups);
        if has_indices {
            // `indices[i]` = `[start, end]` per group (undefined when absent).
            let pair = |this: &mut Self, span: Option<(usize, usize)>| -> NanBox {
                match span {
                    Some((s, e)) => {
                        let a = this.realm.new_array(alloc::vec![
                            NanBox::number(s as f64),
                            NanBox::number(e as f64),
                        ]);
                        NanBox::handle(a.to_raw())
                    }
                    None => NanBox::undefined(),
                }
            };
            let idx_elems: Vec<NanBox> = caps.groups.iter().map(|g| pair(self, *g)).collect();
            let indices = self.realm.new_array(idx_elems);
            let idx_groups = if group_names.is_empty() {
                NanBox::undefined()
            } else {
                let g = self.realm.new_object_with_proto(None);
                for (idx, name) in group_names {
                    let v = pair(self, caps.groups.get(*idx).copied().flatten());
                    self.realm.set_property(g, name, v);
                }
                NanBox::handle(g.to_raw())
            };
            self.realm.set_property(indices, "groups", idx_groups);
            self.realm
                .set_property(obj, "indices", NanBox::handle(indices.to_raw()));
        }
        NanBox::handle(obj.to_raw())
    }

    /// `IsRegExp(value)`: a value with a truthy `@@match` property, or (absent that
    /// property) a RegExp instance. A non-object is not a RegExp.
    pub(crate) fn is_regexp_arg(&mut self, v: NanBox) -> bool {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return false;
        };
        let sym = self.well_known_symbol("match");
        let key = self.member_key(sym);
        if let Some(m) = self.realm.get_property(h, &key)
            && !matches!(m.unpack(), Unpacked::Undefined)
        {
            return self.realm.truthy(m);
        }
        self.realm.regexp_at(h).is_some()
    }

    /// Builds `RegExp.prototype` as a real object: first-class `exec`/`test`/
    /// `compile`/`toString` methods, the `@@match`/`@@matchAll`/`@@replace`/
    /// `@@search`/`@@split` symbol methods, the `source`/`flags`/flag-getter
    /// accessor properties, and a `constructor` back-link. RegExp instances created
    /// by [`new_regexp_instance`](Self::new_regexp_instance) inherit it, so
    /// `re.exec`, `RegExp.prototype.test.call(re, s)`, the property descriptors,
    /// and `instanceof` all behave per spec.
    pub(crate) fn setup_regexp_prototype(&mut self) {
        let Some(ctor) = self
            .current
            .get("RegExp")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        // Mark the prototype so its accessor getters recognise the sentinel
        // receiver (source → "(?:)", flags → "", flag getters → undefined).
        self.realm
            .set_hidden_property(proto, REGEXP_PROTO_BRAND, NanBox::boolean(true));
        // First-class data-property methods.
        for &name in REGEXP_PROTO_METHODS {
            let name_h = self.realm.new_string(name);
            let f = self.realm.new_bound_native(N_REGEXP_PROTO_FN, name_h);
            let arity = if name == "exec" || name == "test" || name == "compile" {
                1
            } else {
                0
            };
            self.install_fn_name_length(f, name, arity);
            self.realm
                .set_property(proto, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, name);
        }
        // Well-known-symbol methods (`@@match`, …).
        for &(sym_name, _) in REGEXP_SYMBOL_METHODS {
            let label = alloc::format!("[Symbol.{sym_name}]");
            let target = self.realm.new_string(sym_name);
            let f = self.realm.new_bound_native(N_REGEXP_PROTO_FN, target);
            // `@@replace`/`@@split` have length 2; the rest length 1.
            let arity = if sym_name == "replace" || sym_name == "split" {
                2
            } else {
                1
            };
            self.install_fn_name_length(f, &label, arity);
            let sym = self.well_known_symbol(sym_name);
            let key = self.member_key(sym);
            self.realm
                .set_property(proto, &key, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, &key);
        }
        // Accessor getters (`get source`, `get flags`, the flag getters).
        for &name in REGEXP_ACCESSORS {
            let label = alloc::format!("get {name}");
            let target = self.realm.new_string(name);
            let getter = self.realm.new_bound_native(N_REGEXP_ACCESSOR, target);
            self.install_fn_name_length(getter, &label, 0);
            self.realm.define_accessor(
                proto,
                name,
                NanBox::handle(getter.to_raw()),
                NanBox::undefined(),
            );
            self.realm.mark_hidden(proto, name);
        }
        // `RegExp.prototype.constructor === RegExp` (non-enumerable).
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        // Link the constructor's `prototype` (non-enumerable, non-writable,
        // non-configurable — a built-in constructor's `prototype`).
        self.realm
            .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(ctor, "prototype");
        self.realm.set_readonly_property(ctor, "prototype");
        self.realm.set_non_configurable_property(ctor, "prototype");
        // Record it so instances and species lookups can find it cheaply.
        self.regexp_proto = Some(proto);
        // `RegExp[Symbol.species]` — a getter returning `this`.
        let species_getter = self.realm.new_bound_native(N_REGEXP_SPECIES, ctor);
        self.install_fn_name_length(species_getter, "get [Symbol.species]", 0);
        let species_sym = self.well_known_symbol("species");
        let species_key = self.member_key(species_sym);
        self.realm.define_accessor(
            ctor,
            &species_key,
            NanBox::handle(species_getter.to_raw()),
            NanBox::undefined(),
        );
        self.realm.mark_hidden(ctor, &species_key);
    }

    /// Allocates a RegExp instance and links its `[[Prototype]]` to
    /// `RegExp.prototype`. `lastIndex` starts at 0.
    pub(crate) fn new_regexp_instance(&mut self, source: &str, flags: &str) -> Handle {
        let h = self.realm.new_regexp(source, flags);
        if let Some(proto) = self.regexp_proto {
            self.realm.set_native_proto(h, proto);
        }
        h
    }

    /// Writes a RegExp's `lastIndex` (the `re.lastIndex = v` assignment path),
    /// honoring a non-writable descriptor installed via `defineProperty`. The
    /// engine stores `lastIndex` as a non-negative `usize` in the cell; when an
    /// aux `lastIndex` data property exists (a user descriptor), the spec value is
    /// also mirrored there so a later read sees it. A non-writable target throws a
    /// TypeError in strict mode and is silently dropped in sloppy mode.
    pub(crate) fn regex_write_last_index(
        &mut self,
        handle: Handle,
        new: NanBox,
    ) -> Result<(), ExecError> {
        // A user `Object.defineProperty(re, "lastIndex", { writable: false })`
        // lands in the aux object; honor its writability.
        if self.realm.has_own(handle, "lastIndex")
            && (self.realm.property_is_readonly(handle, "lastIndex")
                || self.realm.is_frozen(handle))
        {
            if self.strict {
                return Err(self.type_error("Cannot assign to read only property 'lastIndex'"));
            }
            return Ok(());
        }
        let n = self.realm.to_number(new);
        // Mirror into the aux slot (so a `defineProperty`-created data slot reads
        // back the written value) and the cell field (the search position).
        if self.realm.has_own(handle, "lastIndex") {
            self.realm.set_property(handle, "lastIndex", new);
        }
        self.realm.set_regex_last_index(
            handle,
            if n.is_finite() && n >= 0.0 {
                n as usize
            } else {
                0
            },
        );
        Ok(())
    }

    /// `ToString(value)` for a regex subject, returned as both the lossy `String`
    /// (used for `.input` and new-string construction in the common
    /// surrogate-free case) and the **lossless** UTF-16 code units (so a
    /// surrogate-bearing subject indexes/slices correctly). Runs a user `toString`/
    /// `@@toPrimitive` exactly once.
    fn subject_units(&mut self, value: NanBox) -> Result<(String, Vec<u16>), ExecError> {
        let bytes = self.coerce_to_string_bytes(value)?;
        let units: Vec<u16> = crate::wtf8::utf16_units(&bytes).collect();
        let lossy = crate::wtf8::to_string_lossy(&bytes);
        Ok((lossy, units))
    }

    /// The receiver of a `RegExp.prototype` accessor / method: `Some(handle)` for
    /// an actual RegExp instance, else `None`.
    fn this_regexp(&self, this: NanBox) -> Option<Handle> {
        let h = this.as_handle().map(Handle::from_raw)?;
        self.realm.regexp_at(h).map(|_| h)
    }

    /// Whether `this` is the `RegExp.prototype` sentinel object.
    fn is_regexp_proto(&self, this: NanBox) -> bool {
        this.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.get_property(h, REGEXP_PROTO_BRAND).is_some())
    }

    /// Dispatch for a first-class `RegExp.prototype.<method>` (id
    /// [`N_REGEXP_PROTO_FN`]). `name` is the method name (`exec`/`test`/`compile`/
    /// `toString`) or a well-known symbol name (`match`/`matchAll`/`replace`/
    /// `search`/`split`). `this_val` is the call's `this`.
    pub(crate) fn regexp_proto_dispatch(
        &mut self,
        name: &str,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match name {
            "exec" => {
                let Some(h) = self.this_regexp(this_val) else {
                    return Err(
                        self.type_error("RegExp.prototype.exec called on a non-RegExp object")
                    );
                };
                #[cfg(feature = "regex")]
                {
                    self.regexp_builtin_exec(h, arg(0))
                }
                #[cfg(not(feature = "regex"))]
                {
                    let _ = h;
                    Err(ExecError::Unsupported("RegExp needs the regex feature"))
                }
            }
            "test" => {
                if this_val.as_handle().map(Handle::from_raw).is_none() {
                    return Err(self.type_error("RegExp.prototype.test called on a non-object"));
                }
                let s = self.coerce_to_string(arg(0))?;
                let str_v = self.new_str(&s);
                let r = self.regexp_exec(this_val, str_v)?;
                Ok(NanBox::boolean(!matches!(r.unpack(), Unpacked::Null)))
            }
            "compile" => self.regexp_compile(this_val, arg(0), arg(1)),
            "toString" => {
                let Some(h) = this_val.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("RegExp.prototype.toString called on a non-object"));
                };
                // `"/" + source + "/" + flags`, reading both via Get (so a
                // subclass overriding `source`/`flags` is honored).
                let src = self.read_member(h, "source")?;
                let src_s = self.coerce_to_string(src)?;
                let fl = self.read_member(h, "flags")?;
                let fl_s = self.coerce_to_string(fl)?;
                Ok(self.new_str(&alloc::format!("/{src_s}/{fl_s}")))
            }
            "match" => self.regexp_symbol_match(this_val, arg(0)),
            "matchAll" => self.regexp_symbol_match_all(this_val, arg(0)),
            "replace" => self.regexp_symbol_replace(this_val, arg(0), arg(1)),
            "search" => self.regexp_symbol_search(this_val, arg(0)),
            "split" => self.regexp_symbol_split(this_val, arg(0), arg(1)),
            _ => Ok(NanBox::undefined()),
        }
    }

    /// Dispatch for a `get RegExp.prototype.<accessor>` getter (id
    /// [`N_REGEXP_ACCESSOR`]).
    pub(crate) fn regexp_accessor_dispatch(
        &mut self,
        name: &str,
        this_val: NanBox,
    ) -> Result<NanBox, ExecError> {
        // `flags` is generic: it reads each flag getter off `this` (Get), so it
        // works on any object (incl. the prototype sentinel), not just a RegExp.
        if name == "flags" {
            if this_val.as_handle().map(Handle::from_raw).is_none() {
                return Err(self.type_error("RegExp.prototype.flags getter called on a non-object"));
            }
            return self.regexp_flags_string(this_val);
        }
        // The `RegExp.prototype` sentinel: `source` → "(?:)", each flag getter →
        // undefined.
        if self.is_regexp_proto(this_val) && self.this_regexp(this_val).is_none() {
            return Ok(match name {
                "source" => self.new_str("(?:)"),
                _ => NanBox::undefined(),
            });
        }
        let Some(h) = self.this_regexp(this_val) else {
            return Err(self.type_error(&alloc::format!(
                "RegExp.prototype.{name} getter called on a non-RegExp object"
            )));
        };
        let (source, flags) = self.realm.regexp_at(h).unwrap_or_default();
        Ok(match name {
            "source" => {
                if source.is_empty() {
                    self.new_str("(?:)")
                } else {
                    self.new_str(&escape_regexp_source(&source))
                }
            }
            "global" => NanBox::boolean(flags.contains('g')),
            "ignoreCase" => NanBox::boolean(flags.contains('i')),
            "multiline" => NanBox::boolean(flags.contains('m')),
            "dotAll" => NanBox::boolean(flags.contains('s')),
            "sticky" => NanBox::boolean(flags.contains('y')),
            "unicode" => NanBox::boolean(flags.contains('u')),
            "unicodeSets" => NanBox::boolean(flags.contains('v')),
            "hasIndices" => NanBox::boolean(flags.contains('d')),
            _ => NanBox::undefined(),
        })
    }

    /// `get RegExp.prototype.flags`: the ordered concatenation of the flag letters
    /// for which the matching flag getter on `this` is truthy (`hasIndices` `d`,
    /// `global` `g`, `ignoreCase` `i`, `multiline` `m`, `dotAll` `s`, `unicode`
    /// `u`, `unicodeSets` `v`, `sticky` `y`). Reads each via Get, so a subclass
    /// overriding a flag getter is honored.
    fn regexp_flags_string(&mut self, this_val: NanBox) -> Result<NanBox, ExecError> {
        let h = this_val.as_handle().map(Handle::from_raw).unwrap();
        let mut out = String::new();
        for (getter, letter) in [
            ("hasIndices", 'd'),
            ("global", 'g'),
            ("ignoreCase", 'i'),
            ("multiline", 'm'),
            ("dotAll", 's'),
            ("unicode", 'u'),
            ("unicodeSets", 'v'),
            ("sticky", 'y'),
        ] {
            let v = self.read_member(h, getter)?;
            if self.realm.truthy(v) {
                out.push(letter);
            }
        }
        Ok(self.new_str(&out))
    }

    /// `RegExp.prototype.compile(pattern, flags)`: recompiles the RegExp in place.
    /// `pattern` may be a RegExp (then `flags` must be undefined). Resets
    /// `lastIndex` to 0.
    fn regexp_compile(
        &mut self,
        this_val: NanBox,
        pattern: NanBox,
        flags: NanBox,
    ) -> Result<NanBox, ExecError> {
        let Some(h) = self.this_regexp(this_val) else {
            return Err(self.type_error("RegExp.prototype.compile called on a non-RegExp object"));
        };
        // `compile(re)` copies `re`'s source/flags; supplying flags too is a
        // TypeError.
        let (pat_s, flags_s) = if let Some(ph) = pattern.as_handle().map(Handle::from_raw)
            && let Some((ps, fs)) = self.realm.regexp_at(ph)
        {
            if !matches!(flags.unpack(), Unpacked::Undefined) {
                return Err(self
                    .type_error("Cannot supply flags when constructing one RegExp from another"));
            }
            (ps, fs)
        } else {
            let p = if matches!(pattern.unpack(), Unpacked::Undefined) {
                String::new()
            } else {
                self.coerce_to_string(pattern)?
            };
            let f = if matches!(flags.unpack(), Unpacked::Undefined) {
                String::new()
            } else {
                self.coerce_to_string(flags)?
            };
            (p, f)
        };
        #[cfg(feature = "regex")]
        if crate::regex::Regex::new(&pat_s, &flags_s).is_err() {
            let e = self.regexp_syntax_error(&pat_s, &flags_s);
            return Err(ExecError::Throw(e));
        }
        self.realm.recompile_regexp(h, &pat_s, &flags_s);
        Ok(this_val)
    }

    /// `RegExpBuiltinExec(R, S)` — the native matcher. Reads `lastIndex` via Get,
    /// runs the engine, writes `lastIndex` back via Set (so observable on stateful
    /// regexes), and returns a match array or null. `s` is the subject (already
    /// ToString'd); the lossless WTF-8 path is used internally.
    #[cfg(feature = "regex")]
    pub(crate) fn regexp_builtin_exec(
        &mut self,
        h: Handle,
        s_value: NanBox,
    ) -> Result<NanBox, ExecError> {
        // ToString the subject losslessly (a lone surrogate survives), and keep
        // the string value as the match result's `.input`.
        let bytes = self.coerce_to_string_bytes(s_value)?;
        let units: Vec<u16> = crate::wtf8::utf16_units(&bytes).collect();
        let input = self.new_str_bytes(bytes);
        let (_source, flags) = self.realm.regexp_at(h).unwrap_or_default();
        let global = flags.contains('g');
        let sticky = flags.contains('y');
        let unicode = flags.contains('u') || flags.contains('v');
        // `lastIndex` is read via Get and ToLength'd (so a throwing valueOf
        // surfaces). For a non-global, non-sticky regex it is ignored (start 0).
        let last_index_v = self.read_member(h, "lastIndex")?;
        let li = self.coerce_to_integer_or_infinity(last_index_v)?;
        let li = if li < 0.0 { 0usize } else { li as usize };
        let start = if global || sticky { li } else { 0 };

        let Some(re) = self.realm.regex_compiled(h) else {
            return Ok(NanBox::null());
        };
        let len = units.len();
        if start > len {
            if global || sticky {
                self.set_last_index(h, 0)?;
            }
            return Ok(NanBox::null());
        }
        // Sticky must match exactly at `start`; non-sticky scans forward.
        let caps = if sticky {
            re.captures_in_u16(&units, start)
                .filter(|c| c.whole().0 == start)
        } else {
            re.captures_in_u16(&units, start)
        };
        let _ = unicode;
        match caps {
            None => {
                if global || sticky {
                    self.set_last_index(h, 0)?;
                }
                Ok(NanBox::null())
            }
            Some(c) => {
                if global || sticky {
                    self.set_last_index(h, c.whole().1)?;
                }
                let has_indices = flags.contains('d');
                Ok(self.regex_match_object_u16_indices(
                    &units,
                    input,
                    &c,
                    re.group_names(),
                    has_indices,
                ))
            }
        }
    }

    /// The spec's internal `Set(rx, "lastIndex", n, true)` (Throw = true): always
    /// throws a TypeError when `lastIndex` is non-writable, regardless of the
    /// caller's strictness (this is the abstract operation the `@@match`/`@@split`/
    /// `RegExpBuiltinExec` algorithms use, not the user-facing `re.lastIndex = v`).
    fn set_last_index(&mut self, h: Handle, n: usize) -> Result<(), ExecError> {
        self.set_last_index_value(h, NanBox::number(n as f64))
    }

    /// As [`set_last_index`](Self::set_last_index) but with an arbitrary value (the
    /// `@@search` save/restore preserves a non-integer/object `lastIndex`).
    fn set_last_index_value(&mut self, h: Handle, v: NanBox) -> Result<(), ExecError> {
        if self.realm.has_own(h, "lastIndex")
            && (self.realm.property_is_readonly(h, "lastIndex") || self.realm.is_frozen(h))
        {
            return Err(self.type_error("Cannot assign to read only property 'lastIndex'"));
        }
        if self.realm.has_own(h, "lastIndex") {
            self.realm.set_property(h, "lastIndex", v);
        }
        let n = self.realm.to_number(v);
        self.realm.set_regex_last_index(
            h,
            if n.is_finite() && n >= 0.0 {
                n as usize
            } else {
                0
            },
        );
        Ok(())
    }

    /// `RegExpExec(R, S)` — calls `R.exec` if it is callable (honoring a
    /// user/subclass override), validating its result is Object-or-null; otherwise
    /// falls back to `RegExpBuiltinExec` (which requires `R` to be a RegExp).
    pub(crate) fn regexp_exec(&mut self, r: NanBox, s: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = r.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("RegExpExec called on a non-object"));
        };
        let exec = self.read_member(h, "exec")?;
        if exec
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|fh| self.is_callable(fh))
        {
            let result = self.call_with_this(exec, r, &[s])?;
            if !matches!(result.unpack(), Unpacked::Null) && result.as_handle().is_none() {
                return Err(self.type_error(
                    "RegExp exec method returned something other than an Object or null",
                ));
            }
            return Ok(result);
        }
        #[cfg(feature = "regex")]
        {
            if self.realm.regexp_at(h).is_none() {
                return Err(self.type_error(
                    "RegExpExec called on a non-RegExp object without a callable exec",
                ));
            }
            self.regexp_builtin_exec(h, s)
        }
        #[cfg(not(feature = "regex"))]
        {
            Err(ExecError::Unsupported("RegExp needs the regex feature"))
        }
    }

    /// A catchable `SyntaxError` for an invalid `/pattern/flags`.
    fn regexp_syntax_error(&mut self, pat: &str, flags: &str) -> NanBox {
        let m = self.new_str(&alloc::format!(
            "Invalid regular expression: /{pat}/{flags}"
        ));
        self.make_error(N_SYNTAX_ERROR, Some(m))
    }

    /// `RegExp.prototype[@@search](string)`.
    fn regexp_symbol_search(&mut self, rx: NanBox, string: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = rx.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("RegExp.prototype[Symbol.search] called on a non-object"));
        };
        let s_bytes = self.coerce_to_string_bytes(string)?;
        let str_v = self.new_str_bytes(s_bytes);
        // Save/restore lastIndex around the search (per spec; `Set(…, true)`).
        let prev = self.read_member(h, "lastIndex")?;
        if !self.same_value(prev, NanBox::number(0.0)) {
            self.set_last_index(h, 0)?;
        }
        let result = self.regexp_exec(rx, str_v)?;
        let cur = self.read_member(h, "lastIndex")?;
        if !self.same_value(cur, prev) {
            self.set_last_index_value(h, prev)?;
        }
        if matches!(result.unpack(), Unpacked::Null) {
            return Ok(NanBox::number(-1.0));
        }
        match result.as_handle().map(Handle::from_raw) {
            Some(r) => self.read_member(r, "index"),
            None => Ok(NanBox::number(-1.0)),
        }
    }

    /// `RegExp.prototype[@@match](string)`.
    fn regexp_symbol_match(&mut self, rx: NanBox, string: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = rx.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("RegExp.prototype[Symbol.match] called on a non-object"));
        };
        // ToString(string) losslessly; `str_v` (the spec's `S`) is reused on every
        // `RegExpExec` so a surrogate-bearing subject matches and slices correctly.
        let s_bytes = self.coerce_to_string_bytes(string)?;
        let str_v = self.new_str_bytes(s_bytes.clone());
        let s = crate::wtf8::to_string_lossy(&s_bytes);
        let flags_v = self.read_member(h, "flags")?;
        let flags = self.coerce_to_string(flags_v)?;
        let global = flags.contains('g');
        if !global {
            return self.regexp_exec(rx, str_v);
        }
        let unicode = flags.contains('u') || flags.contains('v');
        // Reset lastIndex to 0 (`Set(rx, "lastIndex", 0, true)`).
        self.set_last_index(h, 0)?;
        let mut results: Vec<NanBox> = Vec::new();
        loop {
            let result = self.regexp_exec(rx, str_v)?;
            if matches!(result.unpack(), Unpacked::Null) {
                break;
            }
            let Some(rh) = result.as_handle().map(Handle::from_raw) else {
                break;
            };
            let match0 = self.read_member(rh, "0")?;
            let m_str = self.coerce_to_string(match0)?;
            results.push(self.new_str(&m_str));
            if m_str.is_empty() {
                // Advance lastIndex to avoid an infinite loop.
                let li_v = self.read_member(h, "lastIndex")?;
                let li = self.coerce_to_integer_or_infinity(li_v)?.max(0.0) as usize;
                let next = self.advance_string_index(&s, li, unicode);
                self.set_last_index(h, next)?;
            }
        }
        if results.is_empty() {
            Ok(NanBox::null())
        } else {
            Ok(NanBox::handle(self.realm.new_array(results).to_raw()))
        }
    }

    /// `RegExp.prototype[@@matchAll](string)` — returns a RegExp String Iterator
    /// (here, an eager generator over the match-result objects).
    fn regexp_symbol_match_all(&mut self, rx: NanBox, string: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = rx.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("RegExp.prototype[Symbol.matchAll] called on a non-object"));
        };
        let s_bytes = self.coerce_to_string_bytes(string)?;
        let s = crate::wtf8::to_string_lossy(&s_bytes);
        let str_v = self.new_str_bytes(s_bytes);
        let flags_v = self.read_member(h, "flags")?;
        let flags = self.coerce_to_string(flags_v)?;
        let global = flags.contains('g');
        let unicode = flags.contains('u') || flags.contains('v');
        // Construct a copy of `rx` (same source/flags, current lastIndex) so
        // iteration does not disturb the original.
        let li_v = self.read_member(h, "lastIndex")?;
        let li = self.coerce_to_integer_or_infinity(li_v)?.max(0.0) as usize;
        let (src, _) = self.realm.regexp_at(h).unwrap_or_default();
        let matcher = self.new_regexp_instance(&src, &flags);
        self.realm.set_regex_last_index(matcher, li);
        let matcher_v = NanBox::handle(matcher.to_raw());
        let mut out: Vec<NanBox> = Vec::new();
        loop {
            let result = self.regexp_exec(matcher_v, str_v)?;
            if matches!(result.unpack(), Unpacked::Null) {
                break;
            }
            let Some(rh) = result.as_handle().map(Handle::from_raw) else {
                break;
            };
            out.push(result);
            if !global {
                break;
            }
            let match0 = self.read_member(rh, "0")?;
            let m_str = self.coerce_to_string(match0)?;
            if m_str.is_empty() {
                let cli_v = self.read_member(matcher, "lastIndex")?;
                let cli = self.coerce_to_integer_or_infinity(cli_v)?.max(0.0) as usize;
                let next = self.advance_string_index(&s, cli, unicode);
                self.realm.set_regex_last_index(matcher, next);
            }
        }
        Ok(self.make_generator(out))
    }

    /// `AdvanceStringIndex(S, index, unicode)` over the `&str` subject. Returns the
    /// next code-unit index, stepping a whole surrogate pair under `unicode`.
    fn advance_string_index(&self, s: &str, index: usize, unicode: bool) -> usize {
        if !unicode {
            return index + 1;
        }
        let units: Vec<u16> = s.encode_utf16().collect();
        advance_index_u16(&units, index, true)
    }

    /// SameValue for two NanBoxes (used by the lastIndex save/restore dance).
    fn same_value(&self, a: NanBox, b: NanBox) -> bool {
        self.realm.same_value(a, b)
    }

    /// `RegExp.prototype[@@split](string, limit)`.
    fn regexp_symbol_split(
        &mut self,
        rx: NanBox,
        string: NanBox,
        limit: NanBox,
    ) -> Result<NanBox, ExecError> {
        let Some(h) = rx.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("RegExp.prototype[Symbol.split] called on a non-object"));
        };
        let s_bytes = self.coerce_to_string_bytes(string)?;
        let units: Vec<u16> = crate::wtf8::utf16_units(&s_bytes).collect();
        let s = crate::wtf8::to_string_lossy(&s_bytes);
        let str_v = self.new_str_bytes(s_bytes);
        let flags_v = self.read_member(h, "flags")?;
        let flags = self.coerce_to_string(flags_v)?;
        let unicode = flags.contains('u') || flags.contains('v');
        // The splitter is a copy of `rx` forced sticky (`y`).
        let new_flags = if flags.contains('y') {
            flags.clone()
        } else {
            alloc::format!("{flags}y")
        };
        let (src, _) = self.realm.regexp_at(h).unwrap_or_default();
        let splitter = self.new_regexp_instance(&src, &new_flags);
        let splitter_v = NanBox::handle(splitter.to_raw());
        let lim = if matches!(limit.unpack(), Unpacked::Undefined) {
            u32::MAX as usize
        } else {
            // ToUint32(limit) — ToNumber (may run a user valueOf / throw), then
            // the modulo-2^32 wrap.
            let n = self.coerce_to_number(limit)?;
            self.realm.to_uint32(n) as usize
        };
        let mut parts: Vec<NanBox> = Vec::new();
        if lim == 0 {
            return Ok(NanBox::handle(self.realm.new_array(parts).to_raw()));
        }
        let size = units.len();
        if size == 0 {
            // An empty string: one element unless a match of "" is found.
            let z = self.regexp_exec(splitter_v, str_v)?;
            if matches!(z.unpack(), Unpacked::Null) {
                parts.push(str_v);
            }
            return Ok(NanBox::handle(self.realm.new_array(parts).to_raw()));
        }
        let mut p = 0usize; // start of the current segment
        let mut q = 0usize; // search cursor
        while q < size {
            self.realm.set_regex_last_index(splitter, q);
            let z = self.regexp_exec(splitter_v, str_v)?;
            if matches!(z.unpack(), Unpacked::Null) {
                q = self.advance_string_index(&s, q, unicode);
                continue;
            }
            let li_v = self.read_member(splitter, "lastIndex")?;
            let e = (self.coerce_to_integer_or_infinity(li_v)?.max(0.0) as usize).min(size);
            if e == p {
                q = self.advance_string_index(&s, q, unicode);
                continue;
            }
            // Push S[p..q].
            parts.push(self.new_str_bytes(u16_slice(&units, p, q)));
            if parts.len() == lim {
                return Ok(NanBox::handle(self.realm.new_array(parts).to_raw()));
            }
            // Splice capture groups.
            if let Some(zh) = z.as_handle().map(Handle::from_raw) {
                let n = self.read_member(zh, "length")?;
                let ncaps =
                    (self.coerce_to_integer_or_infinity(n)?.max(1.0) as usize).saturating_sub(1);
                for i in 1..=ncaps {
                    let cap = self.read_member(zh, &alloc::format!("{i}"))?;
                    parts.push(cap);
                    if parts.len() == lim {
                        return Ok(NanBox::handle(self.realm.new_array(parts).to_raw()));
                    }
                }
            }
            p = e;
            q = p;
        }
        parts.push(self.new_str_bytes(u16_slice_from(&units, p)));
        Ok(NanBox::handle(self.realm.new_array(parts).to_raw()))
    }

    /// `RegExp.prototype[@@replace](string, replaceValue)`.
    fn regexp_symbol_replace(
        &mut self,
        rx: NanBox,
        string: NanBox,
        replace: NanBox,
    ) -> Result<NanBox, ExecError> {
        let Some(h) = rx.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("RegExp.prototype[Symbol.replace] called on a non-object"));
        };
        let s_bytes = self.coerce_to_string_bytes(string)?;
        let units: Vec<u16> = crate::wtf8::utf16_units(&s_bytes).collect();
        let s = crate::wtf8::to_string_lossy(&s_bytes);
        // The lossless subject string value passed to every `RegExpExec`.
        let str_v = self.new_str_bytes(s_bytes);
        let length = units.len();
        let func_replace = replace
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|fh| self.is_callable(fh));
        let repl_str = if func_replace {
            String::new()
        } else {
            self.coerce_to_string(replace)?
        };
        let flags_v = self.read_member(h, "flags")?;
        let flags = self.coerce_to_string(flags_v)?;
        let global = flags.contains('g');
        let unicode = flags.contains('u') || flags.contains('v');
        if global {
            self.set_last_index(h, 0)?;
        }
        // Collect all matches first (so user `exec` runs in the spec order).
        let mut results: Vec<NanBox> = Vec::new();
        loop {
            let result = self.regexp_exec(rx, str_v)?;
            if matches!(result.unpack(), Unpacked::Null) {
                break;
            }
            results.push(result);
            if !global {
                break;
            }
            let m0 = match result.as_handle().map(Handle::from_raw) {
                Some(r) => self.read_member(r, "0")?,
                None => NanBox::undefined(),
            };
            let m_str = self.coerce_to_string(m0)?;
            if m_str.is_empty() {
                let li_v = self.read_member(h, "lastIndex")?;
                let li = self.coerce_to_integer_or_infinity(li_v)?.max(0.0) as usize;
                let next = self.advance_string_index(&s, li, unicode);
                self.set_last_index(h, next)?;
            }
        }
        // Build the output.
        let mut accumulated: Vec<u8> = Vec::new();
        let mut next_source_position = 0usize;
        for result in results {
            let Some(rh) = result.as_handle().map(Handle::from_raw) else {
                continue;
            };
            let n_v = self.read_member(rh, "length")?;
            let n_captures =
                (self.coerce_to_integer_or_infinity(n_v)?.max(0.0) as usize).saturating_sub(1);
            let matched_v = self.read_member(rh, "0")?;
            let (matched, matched_units) = self.subject_units(matched_v)?;
            let m_len = matched_units.len();
            let pos_v = self.read_member(rh, "index")?;
            let pos = (self.coerce_to_integer_or_infinity(pos_v)?.max(0.0) as usize).min(length);
            // Collect captures.
            let mut captures: Vec<Option<String>> = Vec::with_capacity(n_captures);
            for i in 1..=n_captures {
                let cap = self.read_member(rh, &alloc::format!("{i}"))?;
                if matches!(cap.unpack(), Unpacked::Undefined) {
                    captures.push(None);
                } else {
                    captures.push(Some(self.coerce_to_string(cap)?));
                }
            }
            let named = self.read_member(rh, "groups")?;
            let replacement = if func_replace {
                // Call(replace, undefined, [matched, ...captures, position, S, groups?]).
                let mut call_args: Vec<NanBox> = Vec::with_capacity(n_captures + 4);
                call_args.push(self.new_str(&matched));
                for c in &captures {
                    call_args.push(match c {
                        Some(cs) => self.new_str(cs),
                        None => NanBox::undefined(),
                    });
                }
                call_args.push(NanBox::number(pos as f64));
                call_args.push(str_v);
                if !matches!(named.unpack(), Unpacked::Undefined) {
                    call_args.push(named);
                }
                let r = self.call(replace, &call_args)?;
                self.coerce_to_string(r)?
            } else {
                self.get_substitution(&matched_units, &units, pos, &captures, named, &repl_str)?
            };
            // If position >= next_source_position, append S[next..pos] then the
            // replacement and advance next to pos + m_len.
            if pos >= next_source_position {
                accumulated.extend_from_slice(&u16_slice(&units, next_source_position, pos));
                accumulated.extend_from_slice(replacement.as_bytes());
                next_source_position = pos + m_len;
            }
        }
        if next_source_position < length {
            accumulated.extend_from_slice(&u16_slice_from(&units, next_source_position));
        }
        Ok(self.new_str_bytes(accumulated))
    }

    /// `GetSubstitution(matched, str, position, captures, namedCaptures, replacement)`
    /// — expands `$$`, `$&`, `$\``, `$'`, `$n`, `$nn`, and `$<name>` in the
    /// replacement template. Indices are code-unit positions into `str_units`.
    fn get_substitution(
        &mut self,
        matched_units: &[u16],
        str_units: &[u16],
        position: usize,
        captures: &[Option<String>],
        named: NanBox,
        replacement: &str,
    ) -> Result<String, ExecError> {
        let m_len = matched_units.len();
        let tail_pos = position + m_len;
        let t: Vec<char> = replacement.chars().collect();
        let mut out = String::new();
        let mut i = 0;
        while i < t.len() {
            if t[i] == '$' && i + 1 < t.len() {
                let c = t[i + 1];
                match c {
                    '$' => {
                        out.push('$');
                        i += 2;
                        continue;
                    }
                    '&' => {
                        out.push_str(&crate::wtf8::to_string_lossy(&crate::wtf8::from_utf16(
                            matched_units,
                        )));
                        i += 2;
                        continue;
                    }
                    '`' => {
                        out.push_str(&crate::wtf8::to_string_lossy(&u16_slice(
                            str_units, 0, position,
                        )));
                        i += 2;
                        continue;
                    }
                    '\'' => {
                        out.push_str(&crate::wtf8::to_string_lossy(&u16_slice_from(
                            str_units, tail_pos,
                        )));
                        i += 2;
                        continue;
                    }
                    '<' => {
                        // `$<name>` named-capture reference. With no named
                        // captures it is a literal `$<`.
                        if matches!(named.unpack(), Unpacked::Undefined) {
                            out.push('$');
                            i += 1;
                            continue;
                        }
                        if let Some(close) = t[i + 2..].iter().position(|&ch| ch == '>') {
                            let name: String = t[i + 2..i + 2 + close].iter().collect();
                            if let Some(nh) = named.as_handle().map(Handle::from_raw) {
                                let v = self.read_member(nh, &name)?;
                                if !matches!(v.unpack(), Unpacked::Undefined) {
                                    let vs = self.coerce_to_string(v)?;
                                    out.push_str(&vs);
                                }
                            }
                            i += 2 + close + 1;
                            continue;
                        }
                        out.push('$');
                        i += 1;
                        continue;
                    }
                    '0'..='9' => {
                        // `$n` or `$nn` (1..=99). Prefer two digits when valid.
                        let d1 = c as usize - '0' as usize;
                        let two = if i + 2 < t.len() && t[i + 2].is_ascii_digit() {
                            Some(d1 * 10 + (t[i + 2] as usize - '0' as usize))
                        } else {
                            None
                        };
                        let n = captures.len();
                        if let Some(two_n) = two
                            && two_n >= 1
                            && two_n <= n
                        {
                            if let Some(Some(cs)) = captures.get(two_n - 1) {
                                out.push_str(cs);
                            }
                            i += 3;
                            continue;
                        }
                        if d1 >= 1 && d1 <= n {
                            if let Some(Some(cs)) = captures.get(d1 - 1) {
                                out.push_str(cs);
                            }
                            i += 2;
                            continue;
                        }
                        // Not a valid group reference: literal `$`.
                        out.push('$');
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            out.push(t[i]);
            i += 1;
        }
        Ok(out)
    }
}

/// `EscapeRegExpPattern` (a faithful subset): escapes `/` and line terminators so
/// the result is a valid pattern between two `/` delimiters and re-parses to the
/// same source. Other characters are passed through unchanged.
#[cfg(feature = "regex")]
fn escape_regexp_source(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                // Keep an existing escape as-is (don't double it).
                out.push('\\');
                if i + 1 < chars.len() {
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
            }
            '/' => out.push_str("\\/"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

#[cfg(not(feature = "regex"))]
fn escape_regexp_source(source: &str) -> String {
    String::from(source)
}
