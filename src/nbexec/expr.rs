use super::*;

impl Interp {
    // --- expressions ---

    /// Resolves an identifier *reference* and returns its value (`GetValue`):
    /// the predeclared globals (`undefined`/`NaN`/`Infinity`), a `with`-object
    /// property, a lexical binding, or a global-object own property — throwing a
    /// catchable `ReferenceError` when the reference is unresolvable. Shared by a
    /// bare-identifier read and the read step of a compound assignment.
    pub(crate) fn read_ident_ref(&mut self, name: &str) -> Result<NanBox, ExecError> {
        // An imported binding (`import { x } from "m"`) resolves *live* through
        // the exporting module's own scope, so a later mutation of the export is
        // observed here. A reference before the source module has run leaves the
        // slot absent (TDZ) and throws a ReferenceError.
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some((src_scope, src_name)) = self.module_imports.get(name).cloned() {
            return match src_scope.get(&src_name) {
                // The slot is either absent (source module not yet run) or holds
                // the TDZ sentinel (the source `let`/`const`/`class` is hoisted but
                // its initializer has not run): both are an uninitialized binding.
                Some(v) if !v.is_tdz() => Ok(v),
                _ => {
                    let msg = self.new_str(&alloc::format!(
                        "Cannot access '{name}' before initialization"
                    ));
                    Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(msg)),
                    ))
                }
            };
        }
        // A bare identifier inside `with (obj)` first resolves against the
        // with-object's properties (via `[[Get]]`, so accessors fire) — this
        // shadows even the `undefined`/`NaN`/`Infinity` global identifiers when
        // the with-object provides them (`with ({ NaN: 1 }) { NaN }` is 1).
        if let Some(h) = self.with_binding_result(name)? {
            // `GetBindingValue(N, S)` for an object environment record re-checks
            // `? HasProperty(bindingObject, N)` (a second proxy `has` trap) *after*
            // the `HasBinding` resolution above — so a binding deleted by the
            // `@@unscopables` getter is observed: strict → ReferenceError, sloppy →
            // undefined.
            if !self.has_property_proxied(h, name)? {
                if self.strict {
                    let msg = self.new_str(&alloc::format!("{name} is not defined"));
                    return Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(msg)),
                    ));
                }
                return Ok(NanBox::undefined());
            }
            return self.read_member(h, name);
        }
        self.read_ident_lexical(name)
    }

    /// The non-`with` portion of `GetValue` for a bare identifier: the
    /// predeclared globals, a lexical binding, or a global-object own property.
    /// Split out so a caller that has *already* resolved (and rejected) the `with`
    /// object frames — e.g. a bare-identifier **call**, whose callee reference and
    /// `this`-base must be resolved by a single `HasBinding` — can finish the read
    /// without re-consulting the `with` chain (which would re-run its `has` trap).
    pub(crate) fn read_ident_lexical(&mut self, name: &str) -> Result<NanBox, ExecError> {
        // A live module-import binding (as in `read_ident_ref`) — preserved here so
        // callers using this non-`with` path (e.g. a bare-identifier call) still
        // resolve imported functions.
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some((src_scope, src_name)) = self.module_imports.get(name).cloned() {
            return match src_scope.get(&src_name) {
                Some(v) if !v.is_tdz() => Ok(v),
                _ => {
                    let msg = self.new_str(&alloc::format!(
                        "Cannot access '{name}' before initialization"
                    ));
                    Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(msg)),
                    ))
                }
            };
        }
        match name {
            "undefined" => return Ok(NanBox::undefined()),
            "NaN" => return Ok(NanBox::number(f64::NAN)),
            "Infinity" => return Ok(NanBox::number(f64::INFINITY)),
            _ => {}
        }
        match self.current.get(name) {
            // A binding still in its temporal dead zone (a formal parameter
            // referenced by its own / an earlier parameter's default before it is
            // initialized — `(a = a) =>`, `(a = b, b) =>`) throws a ReferenceError.
            Some(v) if v.is_tdz() => {
                let msg = self.new_str(&alloc::format!(
                    "Cannot access '{name}' before initialization"
                ));
                Err(ExecError::Throw(
                    self.make_error(N_REFERENCE_ERROR, Some(msg)),
                ))
            }
            Some(v) => Ok(v),
            // Not in the lexical scope chain: the global environment record's
            // *object* record still binds every name the global object has — a
            // property added directly to it (`this.x = …` / `globalThis.x = …` at
            // script level) and, since `HasBinding` is `HasProperty`, an inherited
            // one such as `%Object.prototype%`'s `toString` / `valueOf` /
            // `hasOwnProperty`.
            None => {
                if let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
                    && self.global_object_provides(name)
                {
                    return self.read_member(g, name);
                }
                let msg = self.new_str(&alloc::format!("{name} is not defined"));
                Err(ExecError::Throw(
                    self.make_error(N_REFERENCE_ERROR, Some(msg)),
                ))
            }
        }
    }

    /// Returns the cached well-known symbol `name` (e.g. `iterator`), creating it
    /// on first use. Each is a stable, unique symbol for the realm's lifetime.
    pub(crate) fn well_known_symbol(&mut self, name: &'static str) -> NanBox {
        if let Some(s) = self.well_known_symbols.get(name) {
            return *s;
        }
        let sym = NanBox::handle(
            self.realm
                .new_symbol(&alloc::format!("Symbol.{name}"))
                .to_raw(),
        );
        self.well_known_symbols.insert(name, sym);
        sym
    }

    /// Calls `f(args)` and returns the result's truthiness.
    /// Calls `f` with an explicit `this` and returns whether the result is truthy
    /// (for array predicates with a `thisArg`).
    pub(crate) fn call_truthy_this(
        &mut self,
        f: NanBox,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<bool, ExecError> {
        let r = self.call_with_this(f, this, args)?;
        Ok(self.realm.truthy(r))
    }

    /// The storage key for a property access value: a symbol becomes a unique,
    /// non-enumerable `"\0sym:<id>"` key (so symbol-keyed properties keep their
    /// identity and stay out of string enumeration); anything else is its string
    /// form.
    pub(crate) fn member_key(&self, k: NanBox) -> String {
        if let Some(raw) = k.as_handle()
            && let Some((_, id)) = self.realm.symbol_at(Handle::from_raw(raw))
        {
            return alloc::format!("\u{0}sym:{id}");
        }
        self.realm.to_display_string(k)
    }

    /// Inverse of [`member_key`] for handing a property key to a Proxy trap: a
    /// `"\0sym:<id>"` storage key becomes the real Symbol *value* (so the trap sees
    /// `Symbol(Symbol.iterator)`, not the internal sentinel string); any other key
    /// becomes a String.
    pub(crate) fn key_to_value(&mut self, name: &str) -> NanBox {
        if let Some(idstr) = name.strip_prefix("\u{0}sym:")
            && let Ok(id) = idstr.parse::<u64>()
            && let Some(sh) = self.realm.symbol_for_id(id)
        {
            return NanBox::handle(sh.to_raw());
        }
        self.new_str(name)
    }

    /// `ToPropertyKey(k)`: like `member_key`, but a non-string, non-symbol object
    /// key is coerced with ToPrimitive(String) so a user `toString` is honored
    /// (`obj[{toString(){return "x"}}]` keys on `"x"`).
    pub(crate) fn coerce_property_key(&mut self, k: NanBox) -> Result<String, ExecError> {
        let is_object_key = k.as_handle().is_some_and(|raw| {
            let h = Handle::from_raw(raw);
            self.realm.symbol_at(h).is_none() && !self.realm.is_string_handle(h)
        });
        if is_object_key {
            // ToPrimitive(k, string) in full: `@@toPrimitive` if present, else
            // OrdinaryToPrimitive (`toString` then `valueOf`). Deliberately *not*
            // `coerce_object`, whose fast paths return exotics (RegExp, Date, Map,
            // a function, …) unchanged and then stringify them internally — which
            // silently skips a user-visible `toString`, so
            // `RegExp.prototype.toString = () => { throw 42 }; ({ [/re/]: 0 })`
            // must throw 42 rather than key on `"/re/"`.
            let p = match self.symbol_to_primitive(k, "string")? {
                Some(v) => v,
                None => self.ordinary_to_primitive(k, "string")?,
            };
            // ToPropertyKey: if ToPrimitive produced a Symbol, it is the key as-is
            // (do NOT ToString it). Otherwise ToString the primitive.
            if let Some(raw) = p.as_handle()
                && self.realm.symbol_at(Handle::from_raw(raw)).is_some()
            {
                return Ok(self.member_key(p));
            }
            return Ok(self.realm.to_display_string(p));
        }
        Ok(self.member_key(k))
    }

    /// Invokes a plain object's `[Symbol.toPrimitive](hint)` method, if it has a
    /// callable one. Returns `None` to fall back to `valueOf`/`toString`.
    pub(crate) fn symbol_to_primitive(
        &mut self,
        v: NanBox,
        hint: &str,
    ) -> Result<Option<NanBox>, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(None);
        };
        let h = Handle::from_raw(raw);
        let sym = self.well_known_symbol("toPrimitive");
        let key = self.member_key(sym);
        // `Get(O, @@toPrimitive)` — through `read_member` so an *accessor*
        // `[Symbol.toPrimitive]` getter actually runs (and is observed), and an
        // inherited method resolves. A bare `get_property` would skip getters.
        let f = self.read_member(h, &key)?;
        if !matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) {
            // A non-undefined/null `@@toPrimitive` that is not callable is a
            // TypeError (per ToPrimitive step 2.c.i).
            if !f
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let m = self.new_str("Symbol.toPrimitive is not a function");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            let hint_box = self.new_str(hint);
            let r = self.call_with_this(f, v, &[hint_box])?;
            // `[Symbol.toPrimitive]` must return a primitive, else a TypeError.
            if self.is_object_value(r) {
                let m = self.new_str("Cannot convert object to primitive value");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(Some(r));
        }
        Ok(None)
    }

    /// A `TypeError` for a branded built-in accessor (`get DataView.prototype.buffer`,
    /// `get %TypedArray%.prototype.length`, …) read off a receiver that lacks the
    /// matching internal slot. The accessor is inherited, so `GetFunctionRealm` of
    /// its getter — not the running realm — supplies the `%TypeError%`: a receiver
    /// whose `[[Prototype]]` is *another realm's* view must throw that realm's
    /// `TypeError`, exactly as calling the getter function directly would.
    fn branded_accessor_type_error(
        &mut self,
        handle: Handle,
        brand: &str,
        name: &str,
        msg: &str,
    ) -> ExecError {
        let getter = self
            .brand_owner_on_chain(handle, brand)
            .and_then(|owner| self.realm.accessor(owner, name))
            .and_then(|(g, _)| g.as_handle())
            .map(Handle::from_raw);
        let saved = self.cur_realm;
        if let Some(g) = getter {
            self.cur_realm = self.get_function_realm(g);
        }
        let e = self.type_error(msg);
        self.cur_realm = saved;
        e
    }

    /// Whether `v` is an object (a non-primitive heap value: object/array/function/…)
    /// rather than a string/symbol/bigint primitive or an immediate.
    pub(crate) fn is_object_value(&self, v: NanBox) -> bool {
        self.as_object_handle(v).is_some()
    }

    /// `v` as an **object** handle. Unlike a bare `as_handle`, this rejects the
    /// heap-backed *primitives* (string, symbol, BigInt), so a spec step phrased
    /// "if Type(x) is not Object, throw a TypeError" can be written directly.
    pub(crate) fn as_object_handle(&self, v: NanBox) -> Option<Handle> {
        let h = v.as_handle().map(Handle::from_raw)?;
        (!self.realm.is_string_handle(h)
            && self.realm.symbol_at(h).is_none()
            && self.realm.bigint_at(h).is_none())
        .then_some(h)
    }

    /// The frozen template object (strings plus a frozen, non-enumerable `.raw`)
    /// of the tagged-template site `site`, created on its first evaluation and
    /// reused on every later one — its identity is observable to the tag. A
    /// quasi with an invalid escape has no cooked value (`undefined`), while its
    /// `.raw` is still preserved (ES2018).
    pub(crate) fn template_object_for_site(
        &mut self,
        site: usize,
        cooked: &[Option<Vec<u8>>],
        raw: &[String],
    ) -> NanBox {
        let cache_key = (site, self.eval_site_epoch);
        if let Some(cached) = self.tagged_template_cache.get(&cache_key) {
            return *cached;
        }
        let strings: Vec<NanBox> = cooked
            .iter()
            .map(|c| match c {
                Some(b) => self.new_str_bytes(b.clone()),
                None => NanBox::undefined(),
            })
            .collect();
        let raw: Vec<NanBox> = raw.iter().map(|r| self.new_str(r)).collect();
        let strings_h = self.realm.new_array(strings);
        // Both arrays are frozen, per spec — freeze `.raw` first and `strings`
        // last so the property write lands; `raw` is non-enumerable.
        let raw_h = self.realm.new_array(raw);
        self.realm.freeze_object(raw_h);
        self.realm
            .set_property(strings_h, "raw", NanBox::handle(raw_h.to_raw()));
        self.realm.mark_hidden(strings_h, "raw");
        self.realm.freeze_object(strings_h);
        let arr = NanBox::handle(strings_h.to_raw());
        self.tagged_template_cache.insert(cache_key, arr);
        arr
    }

    /// `recv.name(...args)` for a *named* key. An own property of that name wins over
    /// the built-in by-name dispatch (so a reassigned method keeps its own
    /// `this`-validation), a user-patched `Promise.prototype.then`/`catch`/
    /// `finally` is honoured, a primitive receiver is boxed to find the method
    /// but stays `this`, and anything else is an ordinary `[[Get]]` + call.
    pub(crate) fn call_member_named(
        &mut self,
        recv: NanBox,
        name: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        if let Some(rh) = recv.as_handle().map(Handle::from_raw)
            && self.realm.has_own(rh, name)
        {
            let f = self.read_member(rh, name)?;
            if f.as_handle()
                .map(Handle::from_raw)
                .is_some_and(|fh| self.is_callable(fh))
            {
                return self.call_with_this(f, recv, args);
            }
        }
        if matches!(name, "then" | "catch" | "finally")
            && let Some(rh) = recv.as_handle().map(Handle::from_raw)
            && self.realm.promise_state(rh).is_some()
        {
            let f = self.read_member(rh, name)?;
            if let Some(fh) = f.as_handle().map(Handle::from_raw)
                && self.is_callable(fh)
                && self.realm.native_at(fh).is_none()
            {
                return self.call_with_this(f, recv, args);
            }
        }
        if let Some(result) = self.call_method(recv, name, args)? {
            return Ok(result);
        }
        let Some(raw) = recv.as_handle() else {
            if matches!(recv.unpack(), Unpacked::Undefined | Unpacked::Null) {
                let m = self.new_str("cannot read property of null or undefined");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            let boxed = self.coerce_to_object(recv);
            if let Some(bh) = boxed.as_handle().map(Handle::from_raw) {
                let f = self.read_member(bh, name)?;
                if f.as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return self.call_with_this(f, recv, args);
                }
            }
            let m = self.new_str("is not a function");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        let f = self.read_member(Handle::from_raw(raw), name)?;
        self.call_with_this(f, recv, args)
    }

    /// `delete name` for an identifier in sloppy code: a `with`-object property,
    /// a deletable local (an `eval`-introduced `var`), or a configurable global
    /// property; anything else is not deletable. Returns `(result,
    /// is_property_delete)` — the latter decides whether a strict-mode `false`
    /// would be a `TypeError`.
    pub(crate) fn delete_identifier(&mut self, name: &str) -> (bool, bool) {
        let mut result = true;
        let mut is_property_delete = false;
        // Resolve the bare name the way the spec's environment
        // chain does: an enclosing `with` object's environment
        // record sits *between* the reference and any outer
        // declarative binding, so it is consulted first.
        // `with_binding` already reports `None` when an inner
        // lexical/var binding shadows the object, so the
        // declarative arms below still win in that case. (Only
        // the object-record arm can actually delete anything —
        // `with (o) { delete arguments }` removes `o.arguments`
        // rather than reporting `false` for the function's own
        // `arguments` binding.)
        if let Some(h) = self.with_binding(name) {
            // A bare name that resolves through a `with` object's
            // environment deletes that object's property — not the
            // similarly-named global (`with (o) { delete p }`
            // removes `o.p`, leaving any global `p` intact).
            result = self.realm.delete_property(h, name);
            is_property_delete = true;
        } else if let Some(frame) = self.current.owner_frame(name) {
            // A resolvable lexical/var binding is non-deletable
            // (a no-op returning `false`) EXCEPT a binding a
            // sloppy `eval` introduced as deletable into a
            // non-global variable environment
            // (EvalDeclarationInstantiation
            // `CreateMutableBinding(name, true)`): those are
            // removed and return `true`, after which the name
            // resolves to a ReferenceError.
            if !frame.ptr_eq(&self.global_scope) && frame.is_local_deletable(name) {
                frame.delete_local(name);
                result = true;
            } else if frame.ptr_eq(&self.global_scope)
                && let Some(g) = self.global_object()
                && (self.realm.has_own(g, name) || self.realm.accessor(g, name).is_some())
                && !self.realm.property_is_non_configurable(g, name)
            {
                // A built-in global (`JSON`, `Math`, a constructor,
                // …) is a *configurable* property of the global
                // object that this engine mirrors as a global-scope
                // binding. `delete JSON` removes the property, so
                // the mirror must go too. A global `var`/function
                // declaration is non-configurable and stays.
                result = self.realm.delete_property(g, name);
                if result {
                    frame.delete_local(name);
                }
                is_property_delete = true;
            } else {
                result = false;
            }
        } else if let Some(g) = self.global_object()
            && (self.realm.has_own(g, name) || self.realm.accessor(g, name).is_some())
        {
            // `delete name` where `name` resolves to a property of the
            // global object: succeeds only if that property is
            // configurable (e.g. `delete NaN`/`Infinity`/`undefined`
            // — non-configurable — returns `false`).
            result = self.realm.delete_property(g, name);
            is_property_delete = true;
        }
        // An unresolvable name (`delete notDefined`) returns `true`.

        (result, is_property_delete)
    }

    /// `delete h[name]` on an object handle: a proxy's `deleteProperty` trap
    /// (with its invariant checks), a typed array's integer index, or an
    /// ordinary property (breaking a mapped `arguments` alias). Returns the
    /// operation's boolean result; the strict-mode `TypeError` for `false` is
    /// the caller's.
    pub(crate) fn delete_named_on_handle(
        &mut self,
        h: Handle,
        name: &str,
    ) -> Result<bool, ExecError> {
        let result;
        let name = String::from(name);
        // A Deferred Module Namespace (`import defer`)
        // evaluates its target on a `[[Delete]]` with a
        // String (non-"then") key.
        #[cfg(all(feature = "module", feature = "std"))]
        self.trigger_deferred_namespace(h, &name)?;
        // Proxy `deleteProperty` trap, or forward.
        if let Some((target, handler)) = self.realm.proxy_at(h) {
            self.guard_revoked(h)?;
            if let Some(trap) = self.proxy_trap(handler, "deleteProperty")? {
                let kb = self.key_to_value(&name);
                let handler_box = NanBox::handle(handler.to_raw());
                let r =
                    self.call_with_this(trap, handler_box, &[NanBox::handle(target.to_raw()), kb])?;
                result = self.realm.truthy(r);
                // Invariant (10.5.10): a true result is
                // illegal if the property exists as a
                // non-configurable own property of the
                // target, or the target is non-extensible
                // and the property is present.
                if result {
                    let present = self.realm.has_own(target, &name)
                        || self.realm.accessor(target, &name).is_some();
                    if present && self.realm.property_is_non_configurable(target, &name) {
                        return Err(self.type_error(
                            "proxy 'deleteProperty' trap removed a non-configurable property",
                        ));
                    }
                    if present && !self.realm.is_extensible(target) {
                        return Err(self.type_error(
                            "proxy 'deleteProperty' trap removed a property of a non-extensible target",
                        ));
                    }
                }
            } else {
                // No `deleteProperty` trap: forward
                // `[[Delete]]` to the target — which may
                // itself be a proxy, so recurse rather than
                // doing an ordinary delete on it.
                result = self.delete_property_of(target, &name)?;
            }
        } else if self.realm.typed_kind(h).is_some()
            && let Some(n) = canonical_numeric_index(&name)
        {
            // Integer-indexed exotic `[[Delete]]`: deleting a
            // *valid* index fails (`false`); any other
            // canonical numeric index succeeds (`true`), and
            // the prototype chain is never consulted.
            let is_neg_zero = n == 0.0 && n.is_sign_negative();
            let detached = self.typed_array_detached(h);
            let valid = !detached
                && !is_neg_zero
                && n == (n as i64) as f64
                && n >= 0.0
                && self
                    .realm
                    .typed_len(h)
                    .is_some_and(|len| (n as usize) < len);
            result = !valid;
        } else {
            // `delete arr[i]` punches a hole in the dense
            // store (and rejects a non-configurable index
            // or `length`); all other deletes route the
            // same way. `delete_property` handles arrays,
            // objects, and aux-bearing cells uniformly.
            result = self.realm.delete_property(h, &name);
            // A successful delete of a mapped `arguments`
            // index breaks its aliasing (10.4.4.5).
            if result {
                self.arg_map_break(h, &name);
            }
        }

        Ok(result)
    }

    /// [`Self::write_primitive_member`] with the property key already computed
    /// (a computed-member target evaluates its key before the RHS).
    pub(crate) fn write_primitive_member_key(
        &mut self,
        prim: NanBox,
        key: &str,
        new: NanBox,
    ) -> Result<(), ExecError> {
        let wrapper = self.coerce_to_object(prim);
        let Some(wh) = wrapper.as_handle().map(Handle::from_raw) else {
            return Ok(());
        };
        // `[[Set]]` runs on the transient wrapper, but the *Receiver* is the
        // primitive — an inherited (strict) setter must see `this` as the
        // primitive value, matching the getter path.
        if self
            .set_through_proto_chain_for(wh, prim, key, new)?
            .is_some()
        {
            return Ok(());
        }
        if self.strict {
            return Err(self.type_error(&alloc::format!(
                "Cannot create property '{key}' on a primitive value"
            )));
        }
        Ok(())
    }

    /// Reads a member by an already-evaluated key value (an array index when the
    /// key is a numeric index and the receiver is an array, else a named read).
    pub(crate) fn read_member_value(
        &mut self,
        handle: crate::heap::Handle,
        key: NanBox,
    ) -> Result<NanBox, ExecError> {
        if let Some(i) = key.as_number().and_then(as_index)
            && self.realm.is_array_like(handle)
            // A plain Array's element keys are [0, 2**32−1); the boundary value
            // 2**32−1 is an ordinary named property. Typed arrays accept any index.
            && (self.realm.typed_kind(handle).is_some() || (i as u64) < u64::from(u32::MAX))
        {
            // A typed array reads directly (no holes, no prototype indices). A plain
            // array reads the element only when the index is a present own slot; a
            // hole or an out-of-range index falls through to the named `[[Get]]`
            // (which walks the prototype chain).
            if self.realm.typed_kind(handle).is_some() {
                return Ok(self.realm.get_element(handle, i));
            }
            if i < self.realm.array_length(handle).unwrap_or(0) {
                let v = self.realm.get_element(handle, i);
                if !v.is_hole() {
                    return Ok(v);
                }
            }
        }
        let name = self.member_key(key);
        self.read_member(handle, &name)
    }

    /// OrdinarySet's *parent* walk for a computed write when the receiver has no
    /// own binding for `key`: an inherited **setter**, or a **proxy** on the
    /// prototype chain, performs the write via `parent.[[Set]]` (the setter runs,
    /// or the proxy's `set` trap fires, with Receiver = the original object).
    /// Returns `Some(())` if the chain handled the write (the caller must NOT
    /// create an own property), or `None` to fall through to the ordinary
    /// own-property write. Mirrors the `assign_member` (dot-key) prototype walk so
    /// the computed-key path (`o[k] = v`, `arr[i] = v`) matches it.
    pub(crate) fn set_through_proto_chain(
        &mut self,
        receiver: crate::heap::Handle,
        key: &str,
        new: NanBox,
    ) -> Result<Option<()>, ExecError> {
        let recv_value = NanBox::handle(receiver.to_raw());
        self.set_through_proto_chain_for(receiver, recv_value, key, new)
    }

    /// [`Self::set_through_proto_chain`] with an explicit **Receiver value**: the
    /// `this` an inherited setter (or the proxy `set` trap's fourth argument) sees.
    /// It differs from the walked object only for a write through a primitive
    /// receiver (`sym.prop = v`), where `[[Set]]` runs on the transient wrapper but
    /// Receiver is the *primitive* — a strict setter must see `typeof this ===
    /// "symbol"`, not the box.
    pub(crate) fn set_through_proto_chain_for(
        &mut self,
        receiver: crate::heap::Handle,
        recv_value: NanBox,
        key: &str,
        new: NanBox,
    ) -> Result<Option<()>, ExecError> {
        let mut cur = self.realm.object_proto(receiver);
        while let Some(c) = cur {
            // A proxy above the receiver handles the write through its own
            // `[[Set]]` (trap, or trapless forward to an inherited setter, else the
            // own-property creation on the receiver).
            if let Some((target, p_handler)) = self.realm.proxy_at(c) {
                self.guard_revoked(c)?;
                if let Some(trap) = self.proxy_trap(p_handler, "set")? {
                    let key_box = self.new_str(key);
                    let handler_box = NanBox::handle(p_handler.to_raw());
                    let r = self.call_with_this(
                        trap,
                        handler_box,
                        &[NanBox::handle(target.to_raw()), key_box, new, recv_value],
                    )?;
                    if self.strict && !self.realm.truthy(r) {
                        let m = self.new_str(&alloc::format!(
                            "'set' on proxy: trap returned falsish for property '{key}'"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    return Ok(Some(()));
                }
                if let Some((_, setter)) = self.realm.accessor(target, key)
                    && !matches!(setter.unpack(), Unpacked::Undefined)
                {
                    self.call_with_this(setter, recv_value, &[new])?;
                    return Ok(Some(()));
                }
                return Ok(None);
            }
            // Integer-indexed exotic `[[Set]]` reached via the prototype chain: a
            // *canonical numeric index* on a typed array in the chain never delegates
            // to a prototype accessor (10.4.5.5). An **invalid** index (out of bounds /
            // fractional / `-0` / negative / detached) is a silent no-op success — the
            // write is dropped and the chain is *not* walked further (so a getter/setter
            // defined on `%TypedArray.prototype%[key]` is unreachable). A **valid** index
            // falls through to the `has_own` shadow-break below (the element shadows any
            // prototype accessor; the caller then writes an own property on the receiver).
            if self.realm.typed_kind(c).is_some()
                && let Some(n) = canonical_numeric_index(key)
            {
                let is_neg_zero = n == 0.0 && n.is_sign_negative();
                let valid = !self.typed_array_detached(c)
                    && !is_neg_zero
                    && n == (n as i64) as f64
                    && n >= 0.0
                    && self
                        .realm
                        .typed_len(c)
                        .is_some_and(|len| (n as usize) < len);
                if !valid {
                    return Ok(Some(()));
                }
            }
            if let Some((_, setter)) = self.realm.accessor(c, key) {
                if !matches!(setter.unpack(), Unpacked::Undefined) {
                    self.call_with_this(setter, recv_value, &[new])?;
                } else if self.strict {
                    // OrdinarySetWithOwnDescriptor: an accessor descriptor whose
                    // [[Set]] is undefined makes the whole `[[Set]]` return false —
                    // the throwing form raises a TypeError, sloppy drops the write.
                    // (The dot-key path already did this; the computed-key path
                    // silently dropped it.)
                    let m = self.new_str(&alloc::format!(
                        "Cannot assign to read only property '{key}' (accessor has no setter)"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A getter-only inherited accessor shadows the data write.
                return Ok(Some(()));
            }
            // An own data property below shadows an inherited accessor/proxy.
            if self.realm.has_own(c, key) {
                // OrdinarySetWithOwnDescriptor recursion: a *non-writable* inherited
                // data property makes the whole [[Set]] fail — strict throws, sloppy
                // silently drops — and no shadowing own property is created on the
                // receiver. A writable inherited data property allows shadowing (fall
                // through to the own-property write on the receiver). The walk starts
                // above the receiver, so `c` is always an ancestor here.
                if !self.can_write_property(c, key) {
                    if self.strict {
                        let m = self.new_str(&alloc::format!(
                            "Cannot assign to read only property '{key}'"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    return Ok(Some(()));
                }
                break;
            }
            cur = self.realm.object_proto(c);
        }
        Ok(None)
    }

    /// A proxy's `[[Set]]` returning the **boolean** result (for `Reflect.set`,
    /// which reports success/failure rather than throwing on a falsy trap
    /// result): invokes the `set` trap with `receiver`, or forwards trapless to
    /// the target's `[[Set]]` (recursing if the target is itself a proxy). A
    /// truthy trap result is subject to the success invariants (which *do*
    /// throw). An ordinary (non-proxy) forward target performs the set and
    /// reports success.
    pub(crate) fn proxy_set_bool(
        &mut self,
        handle: crate::heap::Handle,
        key: &str,
        value: NanBox,
        receiver: NanBox,
    ) -> Result<bool, ExecError> {
        let Some((target, handler)) = self.realm.proxy_at(handle) else {
            // Reached an ordinary object `O = handle` via a trapless forward:
            // OrdinarySet(O, key, value, Receiver) returning the boolean. An
            // inherited **getter-only** accessor fails (`false`); a setter runs
            // (with the Receiver as `this`) and succeeds; otherwise the write lands
            // on the *Receiver* (OrdinarySetWithOwnDescriptor).
            let mut cur = Some(handle);
            // The object whose own data property terminated the chain walk (`O` in
            // OrdinarySetWithOwnDescriptor), if any.
            let mut owner: Option<crate::heap::Handle> = None;
            while let Some(c) = cur {
                // A **proxy** reached while walking the prototype chain: its own
                // `[[Set]]` internal method takes over (OrdinarySetWithOwnDescriptor
                // delegates to `parent.[[Set]](P, V, Receiver)` when the property is
                // absent on the descendant). This fires the proxy's `set` trap (or
                // forwards to its target, possibly another proxy) with the ORIGINAL
                // Receiver preserved. `handle` itself is never a proxy here (that case
                // takes the trap path below), so this only triggers for an ancestor.
                if self.realm.proxy_at(c).is_some() {
                    return self.proxy_set_bool(c, key, value, receiver);
                }
                // Integer-indexed exotic `[[Set]]` (10.4.5.5): a canonical numeric
                // index on a **TypedArray** reached in the chain is governed by that
                // view's bounds and NEVER consults an inherited setter/data property
                // (the prototype chain past it is unreachable for such a key).
                //   - SameValue(O, Receiver): TypedArraySetElement — coerce V (its
                //     side effects run), write only if the index is still valid, and
                //     always report success.
                //   - O ≠ Receiver, *invalid* index: a silent success (no write) —
                //     terminal, so an inherited setter is unreachable.
                //   - O ≠ Receiver, *valid* index: fall through to OrdinarySet, which
                //     creates the data property on the Receiver below.
                if self.realm.typed_kind(c).is_some()
                    && let Some(n) = canonical_numeric_index(key)
                {
                    let index_ok =
                        n == (n as i64) as f64 && n >= 0.0 && !(n == 0.0 && n.is_sign_negative());
                    let valid = index_ok
                        && !self.typed_array_detached(c)
                        && self
                            .realm
                            .typed_len(c)
                            .is_some_and(|len| (n as usize) < len);
                    if receiver.as_handle() == Some(c.to_raw()) {
                        let coerced = if self.realm.typed_kind(c).is_some_and(is_bigint_kind) {
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
                        return Ok(true);
                    }
                    if !valid {
                        return Ok(true);
                    }
                    break;
                }
                if let Some((_, setter)) = self.realm.accessor(c, key) {
                    if matches!(setter.unpack(), Unpacked::Undefined) {
                        return Ok(false);
                    }
                    self.call_with_this(setter, receiver, &[value])?;
                    return Ok(true);
                }
                if self.realm.has_own(c, key) {
                    owner = Some(c);
                    break;
                }
                cur = self.realm.object_proto(c);
            }
            // No accessor on the chain: the own descriptor (if any) is a data
            // descriptor. OrdinarySetWithOwnDescriptor step 2.a rejects outright
            // when *that* descriptor is non-writable — before the Receiver is even
            // consulted — so `super.x = v` through a non-writable inherited data
            // property fails (a TypeError in strict code) rather than shadowing it
            // on the Receiver.
            if let Some(o) = owner
                && !self.can_write_property(o, key)
            {
                return Ok(false);
            }
            // The value is written to the **Receiver**, not to `O`.
            let Some(recv_h) = receiver.as_handle().map(Handle::from_raw) else {
                return Ok(false);
            };
            if recv_h == handle {
                // Receiver === O: ordinary own write (honoring read-only /
                // non-extensible gates).
                if !self.can_write_property(recv_h, key) {
                    return Ok(false);
                }
                let key_box = self.new_str(key);
                self.assign_member_value(recv_h, key_box, value)?;
                return Ok(true);
            }
            // Receiver differs from O (a trapless proxy forwarded here with the
            // original Receiver): OrdinarySetWithOwnDescriptor writes to the
            // Receiver via `[[DefineOwnProperty]]` — for a proxy Receiver this runs
            // its `getOwnPropertyDescriptor` + `defineProperty` traps.
            if self.realm.proxy_at(recv_h).is_some() {
                let existing = self.descriptor_of(recv_h, key)?;
                let desc = self.realm.new_object();
                self.realm.set_property(desc, "value", value);
                if let Some(dh) = existing.as_handle().map(Handle::from_raw) {
                    let is_accessor = self.realm.get_property(dh, "get").is_some()
                        || self.realm.get_property(dh, "set").is_some();
                    let writable = self
                        .realm
                        .get_property(dh, "writable")
                        .is_some_and(|v| self.realm.truthy(v));
                    if is_accessor || !writable {
                        return Ok(false);
                    }
                } else {
                    self.realm
                        .set_property(desc, "writable", NanBox::boolean(true));
                    self.realm
                        .set_property(desc, "enumerable", NanBox::boolean(true));
                    self.realm
                        .set_property(desc, "configurable", NanBox::boolean(true));
                }
                let ok = self.apply_descriptor(recv_h, key, desc, true)?;
                return Ok(ok);
            }
            // Ordinary Receiver distinct from O: an own accessor / non-writable
            // own data property rejects; otherwise create/update the own data
            // property on the Receiver. Only the Receiver's **own** property
            // matters — OrdinarySetWithOwnDescriptor finishes with
            // `CreateDataProperty(Receiver, P, V)` / a value-only
            // `[[DefineOwnProperty]]`, never another `[[Set]]` — so an inherited
            // non-writable data property or setter of the Receiver is irrelevant
            // here (this is what makes `super.x = v` able to shadow a
            // non-writable property inherited from the *derived* prototype).
            if self.realm.accessor(recv_h, key).is_some() {
                return Ok(false);
            }
            let recv_has_own = self.realm.has_own(recv_h, key);
            if recv_has_own {
                if !self.can_write_property(recv_h, key) {
                    return Ok(false);
                }
            } else if !self.realm.is_extensible(recv_h) {
                return Ok(false);
            }
            let desc = self.realm.new_object();
            self.realm.set_property(desc, "value", value);
            if !recv_has_own {
                self.realm
                    .set_property(desc, "writable", NanBox::boolean(true));
                self.realm
                    .set_property(desc, "enumerable", NanBox::boolean(true));
                self.realm
                    .set_property(desc, "configurable", NanBox::boolean(true));
            }
            let ok = self.apply_descriptor(recv_h, key, desc, true)?;
            return Ok(ok);
        };
        self.guard_revoked(handle)?;
        if let Some(trap) = self.proxy_trap(handler, "set")? {
            let key_box = self.key_to_value(key);
            let handler_box = NanBox::handle(handler.to_raw());
            let r = self.call_with_this(
                trap,
                handler_box,
                &[NanBox::handle(target.to_raw()), key_box, value, receiver],
            )?;
            if !self.realm.truthy(r) {
                return Ok(false);
            }
            self.proxy_set_invariant_check(target, key, value)?;
            return Ok(true);
        }
        // No `set` trap: forward `[[Set]]` to the target with the same receiver.
        self.proxy_set_bool(target, key, value, receiver)
    }

    /// Assigns a member by an already-evaluated key value (used when the target's
    /// computed key must be resolved before the RHS, per spec evaluation order).
    /// Mirrors `assign_member`'s proxy / array-index / setter / length handling.
    pub(crate) fn assign_member_value(
        &mut self,
        handle: crate::heap::Handle,
        key: NanBox,
        new: NanBox,
    ) -> Result<(), ExecError> {
        // Proxy `[[Set]]`: route through the receiver-aware `proxy_set_bool`
        // (shared with `Reflect.set`), passing the proxy itself as the Receiver.
        // This preserves the Receiver across a trapless forward — so an inherited
        // accessor setter (e.g. `Object.prototype.__proto__`) runs with `this` =
        // the proxy, and a nested proxy target re-enters its own trap. A `false`
        // result is a failed [[Set]]: strict code throws, sloppy code is silent.
        // A private element is the exception: PrivateSet never performs `[[Set]]`,
        // so a proxy is transparent to it and the element lives on the proxy
        // object itself, where the field initializer stamped it. Forwarding it to
        // the target would also hand the write to a *different* object, which a
        // frozen target then refuses.
        if self.realm.proxy_at(handle).is_some()
            && !crate::realm::is_private_key(&self.member_key(key))
        {
            let name = self.member_key(key);
            let recv = NanBox::handle(handle.to_raw());
            let ok = self.proxy_set_bool(handle, &name, new, recv)?;
            if !ok && self.strict {
                let m = self.new_str(&alloc::format!(
                    "'set' on proxy: trap returned falsish for property '{name}'"
                ));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(());
        }
        // A module namespace exotic object's `[[Set]]` (§10.4.6.9) always returns
        // false: a write is a silent no-op in sloppy code and a TypeError in strict
        // code (all module code is strict). The property table stays authoritative
        // for the live read-through; only user-level assignment is rejected here
        // (engine-internal refreshes go through `realm.set_property`).
        #[cfg(all(feature = "module", feature = "std"))]
        if self.module_namespaces.contains_key(&handle.to_raw()) {
            if self.strict {
                let name = self.member_key(key);
                let m = self.new_str(&alloc::format!(
                    "Cannot assign to read only property '{name}' of a module namespace object"
                ));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(());
        }
        // Integer-indexed exotic `[[Set]]`: for a typed array, a *canonical numeric
        // index* key writes the element (after coercing the value — whose side
        // effects/throw still run for an out-of-bounds index) and is a no-op when the
        // index is invalid; it never creates an own property or reaches a prototype
        // setter. Handles negative / fractional / `-0` / out-of-bounds canonical keys
        // that the integer-index path below (which only accepts `usize`) would miss.
        if self.realm.typed_kind(handle).is_some() {
            let s = self.member_key(key);
            if let Some(n) = canonical_numeric_index(&s) {
                // Coerce the value first (a BigInt view ToBigInt-coerces, a numeric
                // view ToNumber-coerces) so its observable effects run regardless.
                let coerced = if self.realm.typed_kind(handle).is_some_and(is_bigint_kind) {
                    self.coerce_typed_array_write(handle, new)?
                } else {
                    self.coerce_to_number(new)?
                };
                // A write through a view over an immutable buffer is a TypeError
                // (after the value coercion, per TypedArraySetElement).
                self.guard_view_immutable(handle)?;
                let is_neg_zero = n == 0.0 && n.is_sign_negative();
                if !is_neg_zero
                    && n == (n as i64) as f64
                    && n >= 0.0
                    && self
                        .realm
                        .typed_len(handle)
                        .is_some_and(|len| (n as usize) < len)
                    && !self.typed_array_detached(handle)
                {
                    self.realm.set_element(handle, n as usize, coerced);
                }
                return Ok(());
            }
        }
        // A numeric index — a number, or a canonical numeric string ("1", not "01"
        // or "1.0") as produced by `Reflect.set`/`arr["1"]=` — addresses array (or
        // typed-array view) element storage.
        if self.realm.is_array_like(handle) {
            let idx = key.as_number().and_then(as_index).or_else(|| {
                key.as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.string_value(h))
                    .and_then(|s| {
                        s.parse::<usize>()
                            .ok()
                            .filter(|i| alloc::format!("{i}") == s)
                    })
            });
            // For a plain Array, a valid array index is in [0, 2**32−1) — the
            // boundary value 2**32−1 is an ordinary named property, not an element
            // (and must not trigger ArraySetLength). Typed-array views accept any
            // in-bounds integer key here.
            let idx = idx.filter(|&i| {
                self.realm.typed_kind(handle).is_some() || (i as u64) < u64::from(u32::MAX)
            });
            if let Some(i) = idx {
                // For a plain array, `store_array_index` takes the dense fast path
                // unless the index carries a descriptor override (accessor / readonly
                // / frozen), which it then honors. A typed-array view writes through
                // its bytes via `set_element_checked`.
                if self.realm.typed_kind(handle).is_none() {
                    // OrdinarySet: when the index has no own property (a hole or past
                    // the end) an inherited setter / proxy on the chain handles the
                    // write. This walk is skipped for the common case — a pristine
                    // `%Array.prototype%` chain (no inherited index setters) unless one
                    // was installed (`proto_index_accessor_dirty`), e.g.
                    // `Array.prototype[0] = set…` or `Object.setPrototypeOf(arr, proxy)`.
                    // An *own* accessor at the index shadows any inherited one, so it
                    // is left to `store_array_index` (which fires the own setter).
                    let absent_own = self
                        .realm
                        .array_length(handle)
                        .is_none_or(|len| i >= len || self.realm.get_element(handle, i).is_hole());
                    if absent_own
                        && (self.realm.object_proto(handle) != self.realm.array_proto_intrinsic()
                            || self.realm.proto_index_accessor_dirty())
                        && self
                            .realm
                            .accessor(handle, &alloc::format!("{i}"))
                            .is_none()
                        && let Some(()) =
                            self.set_through_proto_chain(handle, &alloc::format!("{i}"), new)?
                    {
                        return Ok(());
                    }
                    self.store_array_index(handle, i, new)?;
                } else {
                    self.set_element_checked(handle, i, new)?;
                }
                return Ok(());
            }
        }
        let name = self.coerce_property_key(key)?;
        // A **mapped `arguments` index** (10.4.4.4 `[[Set]]`): also write the live
        // parameter binding it aliases (`arguments[i] = v` updates the i-th
        // parameter). Fall through to the ordinary store so the own property's
        // value stays in sync for a subsequent `getOwnPropertyDescriptor`.
        if let Some(r) = self.arg_map_binding(handle, &name) {
            self.arg_ref_set(&r, new);
        }
        // A typed array's `length` is an accessor on `%TypedArray%.prototype` with
        // no setter (an integer-indexed exotic object has no own `length`), so
        // `[[Set]]` reports failure: strict code throws, sloppy code drops the
        // write. The view's stored length is never changed either way.
        if name == "length" && self.realm.typed_len(handle).is_some() {
            if self.strict {
                let m = self.new_str(
                    "Cannot assign to read only property 'length' (accessor has no setter)",
                );
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(());
        }
        // `regex.lastIndex = n` updates the RegExp's stateful search position
        // (honoring a non-writable descriptor installed via `defineProperty`).
        if name == "lastIndex" && self.realm.regexp_at(handle).is_some() {
            return self.regex_write_last_index(handle, new);
        }
        // An own accessor setter takes precedence.
        if let Some((_, setter)) = self.realm.accessor(handle, &name) {
            if !matches!(setter.unpack(), Unpacked::Undefined) {
                let this = NanBox::handle(handle.to_raw());
                self.call_with_this(setter, this, &[new])?;
            } else if self.strict {
                // A getter-only accessor cannot be written: the throwing form of
                // `[[Set]]` raises a TypeError; sloppy assignment drops the write.
                let m = self.new_str(&alloc::format!(
                    "Cannot assign to read only property '{name}' (accessor has no setter)"
                ));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(());
        }
        // No own property: an *inherited* accessor **or a proxy** on the prototype
        // chain handles the write via `parent.[[Set]]` (its setter runs, or the
        // proxy's `set` trap fires, with `this`/Receiver = the receiver). An
        // inherited data property, or none, falls through to creating an own data
        // property.
        if !self.realm.has_own(handle, &name)
            && let Some(()) = self.set_through_proto_chain(handle, &name, new)?
        {
            return Ok(());
        }
        // `arr.length = n` resizes the array (with ToUint32 + RangeError check).
        if name == "length" && self.realm.is_array(handle) {
            // ToUint32(value) is coerced first (it may RangeError), *before* the
            // non-writable check — matching the descriptor path's ordering.
            let n = self.array_length_from_value(new)?;
            self.write_array_length(handle, n)?;
        } else if self.allow_property_write(handle, &name)? {
            // Honor a non-writable own data property / non-extensible object:
            // strict mode throws, sloppy mode silently drops the write (this is
            // the computed-key `obj[k] = v` path, e.g. a Symbol-keyed write to a
            // `writable: false` property).
            // A writable array index that reached here (it carries a non-default
            // attribute override, so it skipped the dense fast path) stores into the
            // element store, not a shadowing aux slot.
            // Only a real array index `[0, 2**32−1)` addresses element storage; the
            // boundary `2**32−1` and above are ordinary named properties.
            let array_index = self.realm.is_array(handle).then(|| {
                name.parse::<usize>()
                    .ok()
                    .filter(|i| alloc::format!("{i}") == name && (*i as u64) < u64::from(u32::MAX))
            });
            if let Some(Some(i)) = array_index {
                self.set_element_checked(handle, i, new)?;
            } else {
                self.realm.set_property(handle, &name, new);
                self.sync_global_object_write(handle, &name, new);
            }
        }
        Ok(())
    }

    /// Mirrors a write to the **global object** (`globalThis.X = v`, or the object
    /// `Function("return this")()` returns) into the global *binding* `X`, so a
    /// bare `X` afterwards reads the new value.
    ///
    /// Kataan's global scope is a declarative record and `globalThis` is an object
    /// that mirrors it, so the two would otherwise drift apart. Syncing on the
    /// **write** side keeps identifier *reads* on the plain binding path: the
    /// alternative — resolving every bare identifier through the global object —
    /// also routes the interpreter's own intrinsic lookups through it, so tampering
    /// with a global would leak into engine-internal construction (`%Promise%`,
    /// species constructors, …). Only an existing binding is updated; a brand-new
    /// `globalThis.foo = 1` is created by the ordinary global-object fallback that
    /// identifier resolution already consults.
    fn sync_global_object_write(&mut self, handle: Handle, name: &str, new: NanBox) {
        if self.global_this.as_handle() == Some(handle.to_raw()) {
            if self.global_scope.get(name).is_some() {
                self.global_scope.set(name, new);
            }
            return;
        }
        // The **main** realm's global object, written from *inside* another realm
        // (a `$262.createRealm()` realm is routinely handed the parent's
        // `globalThis`, as `h.mainGlobal = this; h.eval("mainGlobal.x = 1")`).
        // `self.global_this`/`global_scope` are the *running* realm's while that
        // code runs, so the main realm needs its own arm.
        if self.main_global_this.as_handle() == Some(handle.to_raw()) {
            if self.main_global_scope.get(name).is_some() {
                self.main_global_scope.set(name, new);
            }
            return;
        }
        // The same mirroring for **another realm's** global object: a
        // `$262.createRealm()` realm hands its `global` back to this one, and
        // `g.x = v` has to reach *that* realm's binding `x` — code running inside
        // it reads the declarative binding, not the object, so without this the
        // write is invisible there.
        if let Some(r) = self
            .created_realms
            .iter()
            .find(|r| r.global_this.as_handle() == Some(handle.to_raw()))
            && r.global_scope.get(name).is_some()
        {
            r.global_scope.set(name, new);
        }
    }

    /// `arr[i] = v` for an array index: the dense fast path unless the index carries
    /// a non-default attribute override or accessor (or the array is frozen/sealed),
    /// in which case the descriptor is honored — an accessor's setter runs, a
    /// non-writable index drops the write (strict → TypeError). Mirrors the inline
    /// logic of the primary computed-assignment path.
    pub(crate) fn store_array_index(
        &mut self,
        handle: Handle,
        i: usize,
        new: NanBox,
    ) -> Result<(), ExecError> {
        if self.realm.typed_kind(handle).is_none() && self.realm.array_index_has_override(handle, i)
        {
            let key = alloc::format!("{i}");
            // An accessor setter takes precedence. A getter-only accessor (no
            // setter) cannot be written: strict mode throws, sloppy drops.
            if let Some((_, setter)) = self.realm.accessor(handle, &key) {
                if !matches!(setter.unpack(), Unpacked::Undefined) {
                    let this = NanBox::handle(handle.to_raw());
                    self.call_with_this(setter, this, &[new])?;
                } else if self.strict {
                    let m = self.new_str(&alloc::format!(
                        "Cannot assign to read only property '{key}' (accessor has no setter)"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                return Ok(());
            }
            // A non-writable / frozen index: strict throws, sloppy drops.
            if self.allow_property_write(handle, &key)? {
                self.set_element_checked(handle, i, new)?;
            }
            return Ok(());
        }
        self.set_element_checked(handle, i, new)
    }

    /// `arr.length = n` (the assignment path of `ArraySetLength`, ECMA-262
    /// 10.4.3.1): applies the (already ToUint32-coerced) `n`. A non-writable
    /// `length` rejects any change — silently in sloppy mode, with a TypeError in
    /// strict mode (a same-value assignment is a no-op either way). When shrinking
    /// hits a non-configurable index, the truncation stops there; strict mode then
    /// throws (the length is left one above the stuck index in both modes).
    pub(crate) fn write_array_length(&mut self, handle: Handle, n: usize) -> Result<(), ExecError> {
        if self.realm.array_length_is_readonly(handle) {
            // Ordinary `[[Set]]` of a non-writable data property returns `false`
            // whether or not the new value equals the current one — the same-value
            // exception lives only in `[[DefineOwnProperty]]`/ValidateAndApply, not
            // in `[[Set]]`. So `Set(O, "length", V, true)` on a frozen / non-writable
            // -length array (e.g. the closing `Set` of `pop`/`push` on an empty
            // frozen array) throws in strict mode; a sloppy assignment drops silently.
            if self.strict {
                let m = self.new_str("Cannot assign to read only property 'length'");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(()); // sloppy: silently dropped
        }
        let all_deleted = self.set_array_length_checked(handle, n)?;
        if !all_deleted && self.strict {
            let m = self.new_str("Cannot delete non-configurable array element");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    /// `ArraySetLength` length coercion: `ToUint32(v)` must equal `ToNumber(v)`
    /// (so `-1`, `4294967296`, `1.5`, `NaN` are RangeErrors), and the `ToNumber`
    /// coercion fires `valueOf`/`toString` (a Symbol throws). Returns the
    /// validated `u32` length.
    ///
    /// Steps 3 and 4 coerce `v` *twice* — once for `ToUint32`, once for
    /// `ToNumber` — and both are observable, so `v.valueOf` runs twice even when
    /// the two agree.
    pub(crate) fn array_length_from_value(&mut self, v: NanBox) -> Result<usize, ExecError> {
        // ToNumber(v) — abrupt-propagating (a Symbol/throwing valueOf).
        let first = self.coerce_to_number(v)?;
        let new_len = self.realm.to_number(first) as u32;
        let second = self.coerce_to_number(v)?;
        let number_len = self.realm.to_number(second);
        if number_len.is_finite() && number_len == f64::from(new_len) {
            Ok(new_len as usize)
        } else {
            let m = self.new_str("Invalid array length");
            Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))))
        }
    }

    /// The proxy `[[Get]]` success invariants (10.5.8): a non-configurable,
    /// non-writable data property of the target must be reported with its actual
    /// value; a non-configurable accessor with no getter must report `undefined`.
    pub(crate) fn proxy_get_invariant_check(
        &mut self,
        target: crate::heap::Handle,
        name: &str,
        result: NanBox,
    ) -> Result<(), ExecError> {
        if let Some((getter, _)) = self.realm.accessor(target, name) {
            if self.realm.property_is_non_configurable(target, name)
                && matches!(getter.unpack(), Unpacked::Undefined)
                && !matches!(result.unpack(), Unpacked::Undefined)
            {
                return Err(self.type_error(
                    "proxy 'get' returned a value for a non-configurable accessor with no getter",
                ));
            }
        } else if self.realm.has_own(target, name)
            && self.realm.property_is_non_configurable(target, name)
            && self.realm.property_is_readonly(target, name)
        {
            let actual = self
                .realm
                .get_property(target, name)
                .unwrap_or(NanBox::undefined());
            if !self.realm.strict_equals(result, actual) {
                return Err(self.type_error(
                    "proxy 'get' returned a different value for a non-configurable non-writable property",
                ));
            }
        }
        Ok(())
    }

    /// `[[Get]](P, Receiver)` on `obj`, threading an explicit Receiver so that an
    /// inherited accessor getter (or a proxy `get` trap) runs with `this` =
    /// `receiver` — the piece the receiver-less `read_member` drops when it
    /// forwards a trapless proxy to its target or descends into a proxy on the
    /// prototype chain. Data / exotic properties are receiver-independent, so those
    /// defer to `read_member`.
    pub(crate) fn get_with_receiver(
        &mut self,
        obj: crate::heap::Handle,
        name: &str,
        receiver: NanBox,
    ) -> Result<NanBox, ExecError> {
        // A proxy: its `get` trap (with the Receiver), or a trapless forward to the
        // target that keeps the Receiver (recursing so a proxy target runs its own
        // trap / chain).
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "get")? {
                let key = self.key_to_value(name);
                let handler_box = NanBox::handle(handler.to_raw());
                let result = self.call_with_this(
                    trap,
                    handler_box,
                    &[NanBox::handle(target.to_raw()), key, receiver],
                )?;
                self.proxy_get_invariant_check(target, name, result)?;
                return Ok(result);
            }
            return self.get_with_receiver(target, name, receiver);
        }
        // An ordinary object: walk own → prototype chain. An accessor getter runs
        // with the Receiver; a proxy on the chain delegates its `[[Get]]` with the
        // same Receiver; an own data property (or reaching a non-proxy end) defers
        // to `read_member` for the receiver-independent read of `obj`.
        let mut cur = Some(obj);
        while let Some(c) = cur {
            if c != obj && self.realm.proxy_at(c).is_some() {
                return self.get_with_receiver(c, name, receiver);
            }
            if let Some((getter, _)) = self.realm.accessor(c, name) {
                if matches!(getter.unpack(), Unpacked::Undefined) {
                    return Ok(NanBox::undefined());
                }
                return self.call_with_this(getter, receiver, &[]);
            }
            if self.realm.has_own(c, name) {
                break;
            }
            cur = self.realm.object_proto(c);
        }
        self.read_member(obj, name)
    }

    /// Whether `f` is one of the *legacy* callables for which the
    /// implementation-defined `fn.caller` / `fn.arguments` reads stay benign: an
    /// ordinary, source-declared, **non-strict** FunctionDeclaration or
    /// FunctionExpression.
    ///
    /// ECMA-262 16.2 (Forbidden Extensions) restricts every other function form —
    /// strict functions, generators, async functions, arrows, class
    /// constructors/methods, concise methods and `get`/`set` accessors, bound
    /// functions, and built-ins — so those must reach the inherited
    /// `%ThrowTypeError%` accessor on `Function.prototype` and throw a TypeError.
    /// (A `new Function(...)` body is also excluded here; it is a legacy-eligible
    /// shape in mainstream engines but this engine keeps it restricted.)
    pub(crate) fn is_legacy_fn(&self, f: crate::heap::Handle) -> bool {
        if self.realm.get_property(f, BOUND_TARGET).is_some()
            || self.realm.get_property(f, DYN_FN_MARKER).is_some()
        {
            return false;
        }
        self.realm.vm_function(f).is_some_and(|(vm_id, _)| {
            self.vm_table
                .as_ref()
                .and_then(|t| t.get(vm_id as usize))
                .is_some_and(|p| p.legacy)
        })
    }

    /// The legacy `fn.caller` / `fn.arguments` value the host reports for an
    /// *ordinary, source-declared, non-strict* function: `null` (the VM answers
    /// for a function it is running). `None` for every other callable — bound,
    /// dynamically built, strict, generator, async, arrow, or a method — so those
    /// keep the spec's poisoned `%ThrowTypeError%` accessor and throw (16.2
    /// Forbidden Extensions).
    fn legacy_caller(&mut self, f: crate::heap::Handle) -> Option<NanBox> {
        self.is_legacy_fn(f).then(NanBox::null)
    }

    pub(crate) fn read_member(
        &mut self,
        handle: crate::heap::Handle,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        // A Deferred Module Namespace (`import defer`) evaluates its target the
        // first time one of its exports is read — directly or as a prototype /
        // `super` home object (import-defer proposal).
        #[cfg(all(feature = "module", feature = "std"))]
        self.trigger_deferred_in_chain(handle, name)?;
        // A **module namespace** export is a *live* binding: read the current
        // value from its backing slot (so a mutation in the exporting module that
        // happens after the namespace was materialised is observed). The
        // refreshed value is also written back so `getOwnPropertyDescriptor`
        // reports it.
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some((scope, local)) = self
            .module_namespaces
            .get(&handle.to_raw())
            .and_then(|m| m.get(name))
            .map(|(s, l)| (s.clone(), l.clone()))
        {
            let value = scope.get(&local).unwrap_or_else(NanBox::undefined);
            // A namespace binding whose source `let`/`const`/`class`/`function*`
            // has not yet run its initializer is in its Temporal Dead Zone: the
            // [[Get]] (GetBindingValue with Strict=true) throws a ReferenceError
            // rather than returning `undefined`.
            if value.is_tdz() {
                let msg = self.new_str(&alloc::format!(
                    "Cannot access '{name}' before initialization"
                ));
                return Err(ExecError::Throw(
                    self.make_error(N_REFERENCE_ERROR, Some(msg)),
                ));
            }
            // Refresh the stored data property (it is non-configurable but
            // writable, so the engine-internal write is permitted).
            self.realm.set_property(handle, name, value);
            return Ok(value);
        }
        // A **mapped `arguments` index** (10.4.4.3 `[[Get]]`): the value is the live
        // parameter binding it aliases. Refresh the stored data property too so a
        // later `getOwnPropertyDescriptor` reports the current value.
        if let Some(r) = self.arg_map_binding(handle, name) {
            let value = self.arg_ref_get(&r);
            self.realm.set_property(handle, name, value);
            return Ok(value);
        }
        // String index access (`"abc"[1]`) → the UTF-16 code unit at the index
        // (a lone surrogate preserved as a one-unit string).
        //
        // P3: read the unit through the *borrowing* `string_leaf_bytes` when the
        // rope is a single leaf (the overwhelmingly common case) so that
        // `for (i…) c = s[i]` is O(1) per read instead of flattening the whole
        // rope into an owned `Vec` every time (which made the loop O(n²)). A
        // `Concat` tree (no contiguous leaf) falls back to the owned
        // `string_bytes`; a non-string receiver makes both return `None`, so the
        // fast numeric-index path is skipped without any allocation.
        if let Ok(i) = name.parse::<usize>()
            && self.realm.is_string_handle(handle)
        {
            // Collapse a `Concat` once so repeated `s[i]` on a `+=`-built string
            // stops re-walking the tree per read.
            self.realm.flatten_string(handle);
            if let Some(u) = self.realm.string_unit_at(handle, i) {
                return Ok(self.new_str_bytes(crate::wtf8::from_utf16(&[u])));
            }
            // Out of range: a String *wrapper* object can still carry an
            // ordinary own property at that index (`Object.defineProperty(new
            // String("s"), "4", …)`) — String-exotic `[[GetOwnProperty]]` falls
            // back to OrdinaryGetOwnProperty. Only shortcut to `undefined` when
            // there is no such own property (the common primitive-string case).
            if !self.realm.has_own(handle, name) {
                return Ok(NanBox::undefined());
            }
        }
        // A canonical numeric string key on an array (`arr["0"]`) reads the
        // element, exactly like `arr[0]` — but only for a valid array index
        // [0, 2**32−1); the boundary value 2**32−1 is an ordinary named property
        // (handled by the aux lookup below).
        if self.realm.is_array(handle)
            && let Ok(i) = name.parse::<usize>()
            && alloc::format!("{i}") == name
            && (i as u64) < u64::from(u32::MAX)
            && i < self.realm.array_dense_len(handle).unwrap_or(0)
        {
            let v = self.realm.get_element(handle, i);
            // A genuine hole (absent index) is not an own property: the lookup
            // continues up the `[[Prototype]]` chain (handled by the generic walk
            // below) instead of resolving to `undefined` here. An out-of-range
            // index (`i >= length`) likewise falls through (guarded above).
            if !v.is_hole() {
                return Ok(v);
            }
        }
        // Integer-indexed exotic `[[Get]]`: when `handle` is a typed array and `name`
        // is a *canonical numeric index*, the result is the element if the index is
        // valid (an in-bounds non-negative integer, `-0` excluded, buffer attached),
        // else `undefined` — and the prototype chain is **never** consulted (so a
        // throwing getter at `TypedArray.prototype["-1"]` is not invoked).
        if self.realm.typed_kind(handle).is_some()
            && let Some(n) = canonical_numeric_index(name)
        {
            // IsValidIntegerIndex: a detached buffer, `-0`, a non-integer, or an
            // out-of-bounds index all read `undefined`.
            if self.typed_array_detached(handle) {
                return Ok(NanBox::undefined());
            }
            let is_neg_zero = n == 0.0 && n.is_sign_negative();
            if !is_neg_zero
                && n == (n as i64) as f64
                && n >= 0.0
                && let Some(len) = self.realm.typed_len(handle)
                && (n as usize) < len
            {
                return Ok(self.realm.get_element(handle, n as usize));
            }
            return Ok(NanBox::undefined());
        }
        // Proxy `[[Get]]`: the `get` trap, or a trapless forward to the target that
        // preserves the Receiver (so an inherited accessor getter runs with `this`
        // = the proxy). Routed through `get_with_receiver` with Receiver = the
        // proxy itself.
        // A private element is the exception, symmetrically with PrivateSet:
        // PrivateGet never performs `[[Get]]`, so the element is read off the
        // proxy object itself rather than forwarded to the target (which does not
        // have it — the field initializer stamped the proxy).
        if self.realm.proxy_at(handle).is_some() && !crate::realm::is_private_key(name) {
            return self.get_with_receiver(handle, name, NanBox::handle(handle.to_raw()));
        }
        // An error object's `.constructor` is its specific error global — its
        // prototype otherwise reports a generic `Object`. Recognized by an own
        // `name` in the error family plus a `message`. This is a *fallback* only:
        // it fires when nothing before `Object.prototype` defines `constructor`,
        // so a subclass instance (`class E extends Error {}`, whose `constructor`
        // resolves to `E` through its own/prototype chain) is never overridden.
        if name == "constructor" {
            let mut cur = Some(handle);
            let obj_proto = self.realm.default_object_proto();
            let mut resolved = false;
            while let Some(c) = cur {
                if Some(c) == obj_proto {
                    break;
                }
                if self.realm.has_own(c, "constructor") {
                    resolved = true;
                    break;
                }
                cur = self.realm.object_proto(c);
            }
            if !resolved {
                let nm = self
                    .realm
                    .get_property(handle, "name")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                if ERROR_NAMES.contains(&nm.as_str())
                    && self.realm.get_property(handle, "message").is_some()
                    && let Some(ctor) = self.current.get(&nm)
                {
                    return Ok(ctor);
                }
            }
        }
        // Well-known `Symbol.iterator` / `Symbol.asyncIterator` (lazily created).
        if self.realm.native_at(handle) == Some(N_SYMBOL)
            && matches!(
                name,
                "iterator"
                    | "asyncIterator"
                    | "hasInstance"
                    | "toPrimitive"
                    | "toStringTag"
                    | "species"
                    | "isConcatSpreadable"
                    | "match"
                    | "matchAll"
                    | "replace"
                    | "search"
                    | "split"
                    | "unscopables"
                    | "dispose"
                    | "asyncDispose"
            )
        {
            // The name is the well-known symbol's key.
            let key: &'static str = match name {
                "iterator" => "iterator",
                "asyncIterator" => "asyncIterator",
                "hasInstance" => "hasInstance",
                "toPrimitive" => "toPrimitive",
                "toStringTag" => "toStringTag",
                "species" => "species",
                "isConcatSpreadable" => "isConcatSpreadable",
                "match" => "match",
                "matchAll" => "matchAll",
                "replace" => "replace",
                "search" => "search",
                "split" => "split",
                "dispose" => "dispose",
                "asyncDispose" => "asyncDispose",
                _ => "unscopables",
            };
            return Ok(self.well_known_symbol(key));
        }
        // A symbol's `description` (`undefined` for a no-argument `Symbol()`).
        if let Some((desc, _)) = self.realm.symbol_at(handle)
            && name == "description"
        {
            return Ok(if &*desc == SYMBOL_NO_DESC {
                NanBox::undefined()
            } else {
                self.new_str(&desc)
            });
        }
        // A bound function's `name` is `"bound " + target.name` (recursing so a
        // re-bound function reads `"bound bound …"`); its `length` is the target's
        // length minus the bound arguments (floored at 0).
        if matches!(name, "name" | "length")
            && self.fn_meta_synthesizable(handle, name)
            && let Some(target) = self.realm.get_property(handle, BOUND_TARGET)
        {
            let th = target.as_handle().map(Handle::from_raw);
            if name == "name" {
                let tname = match th {
                    Some(t) => {
                        let v = self.read_member(t, "name")?;
                        self.realm.to_display_string(v)
                    }
                    None => String::new(),
                };
                return Ok(self.new_str(&alloc::format!("bound {tname}")));
            }
            // `length`: the same `Function.prototype.bind` steps 5-8 that
            // `make_bound_function` runs eagerly, for a bound function whose
            // physical slot is absent.
            let bound = self
                .realm
                .get_property(handle, BOUND_ARGS)
                .and_then(|a| a.as_handle().map(Handle::from_raw))
                .and_then(|bh| self.realm.array_length(bh))
                .unwrap_or(0);
            let len = self.bound_function_length(th, bound)?;
            return Ok(NanBox::number(len));
        }
        // `obj.__proto__` reads the prototype link (unless shadowed by an own
        // data property of that name).
        // The `__proto__` magic only applies when the object actually inherits
        // `Object.prototype`'s accessor; a null-proto object (module namespace,
        // `Object.create(null)`) reads it as an ordinary absent property.
        if name == "__proto__"
            && !self.realm.has_own(handle, "__proto__")
            && self.realm.proto_accessor_installed()
            && self.realm.inherits_object_proto(handle)
        {
            return Ok(match self.realm.object_proto(handle) {
                Some(p) => NanBox::handle(p.to_raw()),
                None => NanBox::null(),
            });
        }
        // A dynamically-registered host function (`register_fn`, ROADMAP §4.0)
        // reports the declared `name`/`length` its registry entry carries.
        if matches!(name, "length" | "name")
            && self.fn_meta_synthesizable(handle, name)
            && let Some(id) = self.realm.host_fn_at(handle)
            && let Some((fn_name, len)) = self.host_fn_meta(id)
        {
            return Ok(if name == "length" {
                NanBox::number(f64::from(len))
            } else {
                let fn_name = String::from(fn_name);
                self.new_str(&fn_name)
            });
        }
        // A built-in function's `name` and `length`. Plain natives carry `name` in
        // their aux object (resolved above / via `member_value`) but no physical
        // `length`; first-class prototype/static methods (bound natives) carry
        // neither. Synthesize both from the dispatch identity so every built-in
        // function exposes the spec-mandated own `name`/`length` data properties.
        if matches!(name, "length" | "name") && self.fn_meta_synthesizable(handle, name) {
            if let Some((id, target)) = self.realm.bound_native_at(handle) {
                let method = if id == N_ARRAY_PROTO_FN
                    || id == N_AB_PROTO_FN
                    || id == N_SAB_PROTO_FN
                    || id == N_TYPED_ARRAY_PROTO_FN
                {
                    self.realm.string_value(target)
                } else if id == N_STATIC_METHOD {
                    self.realm
                        .array_elements(target)
                        .and_then(|p| p.get(1).copied())
                        .and_then(|v| v.as_handle().map(Handle::from_raw))
                        .and_then(|h| self.realm.string_value(h))
                } else {
                    None
                };
                if let Some(method) = method {
                    return Ok(if name == "name" {
                        self.new_str(&method)
                    } else {
                        NanBox::number(builtin_method_arity(&method) as f64)
                    });
                }
            }
            if let Some(id) = self.realm.native_at(handle) {
                // `Function.prototype[Symbol.hasInstance].name` is the spec's
                // bracketed symbol description.
                if id == N_FN_HAS_INSTANCE && name == "name" {
                    return Ok(self.new_str("[Symbol.hasInstance]"));
                }
                if name == "length" {
                    return Ok(NanBox::number(builtin_native_arity(id) as f64));
                }
            }
        }
        // `Number.*` static constants.
        if self.realm.native_at(handle) == Some(N_NUMBER) {
            match name {
                "MAX_SAFE_INTEGER" => return Ok(NanBox::number(9_007_199_254_740_991.0)),
                "MIN_SAFE_INTEGER" => return Ok(NanBox::number(-9_007_199_254_740_991.0)),
                "MAX_VALUE" => return Ok(NanBox::number(f64::MAX)),
                // The smallest positive value is the least *subnormal* (5e-324),
                // not Rust's `MIN_POSITIVE` (the smallest *normal*, 2.2e-308).
                "MIN_VALUE" => return Ok(NanBox::number(f64::from_bits(1))),
                "EPSILON" => return Ok(NanBox::number(f64::EPSILON)),
                "POSITIVE_INFINITY" => return Ok(NanBox::number(f64::INFINITY)),
                "NEGATIVE_INFINITY" => return Ok(NanBox::number(f64::NEG_INFINITY)),
                "NaN" => return Ok(NanBox::number(f64::NAN)),
                _ => {}
            }
        }
        // The legacy `fn.caller` extension (Annex B "normative optional"): for an
        // ordinary, source-declared *non-strict* function, reading `caller`
        // reports the function currently invoking it instead of reaching the
        // poisoned `%ThrowTypeError%` accessor inherited from
        // `Function.prototype`. Every other callable (strict / generator / async /
        // arrow / bound / dynamic, and `Function.prototype` itself) still throws.
        if name == "caller"
            && !self.realm.has_own(handle, "caller")
            && let Some(v) = self.legacy_caller(handle)
        {
            return Ok(v);
        }
        // The matching legacy `fn.arguments` extension: an ordinary non-strict
        // function reports the `arguments` object of its nearest live activation
        // (`null` when it is not executing) instead of throwing.
        if name == "arguments"
            && !self.realm.has_own(handle, "arguments")
            && let Some(v) = self.legacy_caller(handle)
        {
            return Ok(v);
        }
        if let Some((getter, _)) = self.realm.accessor(handle, name) {
            if matches!(getter.unpack(), Unpacked::Undefined) {
                return Ok(NanBox::undefined());
            }
            let this = NanBox::handle(handle.to_raw());
            return self.call_with_this(getter, this, &[]);
        }
        // `RegExp.prototype.lastIndex` — a real own *data* property of every
        // RegExp instance, stored in the cell (not in the shape), so it is read
        // here directly. Unless overridden by an own aux slot (a user
        // `Object.defineProperty(re,"lastIndex",…)` would land in aux), the cell
        // value is authoritative. `source`/`flags`/the flag getters are spec
        // *accessor* properties on `RegExp.prototype` and resolve through the
        // prototype walk below (so they escape the source, validate the brand, and
        // honor a subclass override).
        if name == "lastIndex"
            && self.realm.regexp_at(handle).is_some()
            && !self.realm.regex_aux_last_index_defined(handle)
        {
            return Ok(NanBox::number(self.realm.regex_last_index(handle) as f64));
        }
        // Branded-prototype accessors. `ArrayBuffer.prototype.byteLength`,
        // `DataView.prototype.buffer`, `%TypedArray%.prototype.buffer`, … are spec
        // accessor properties whose getter requires the matching internal slot on
        // its receiver (RequireInternalSlot). When the receiver inherits the
        // branded prototype but lacks the slot — most visibly the prototype object
        // itself (`ArrayBuffer.prototype.byteLength`) — the getter throws a
        // TypeError instead of returning `undefined`. The slot-bearing instance
        // paths below are reached first for real buffers/views/typed arrays (they
        // have the `ARRAY_BUFFER_BYTES`/`DATA_VIEW_BUF`/typed-kind tags), so this
        // only fires for slot-less receivers.
        if self
            .realm
            .get_property(handle, ARRAY_BUFFER_BYTES)
            .is_none()
            && matches!(
                name,
                "byteLength" | "detached" | "maxByteLength" | "resizable"
            )
            && self.brand_on_chain(handle, ARRAY_BUFFER_PROTO_BRAND)
        {
            return Err(self.branded_accessor_type_error(
                handle,
                ARRAY_BUFFER_PROTO_BRAND,
                name,
                "ArrayBuffer.prototype accessor called on a non-ArrayBuffer object",
            ));
        }
        if self.realm.get_property(handle, DATA_VIEW_BUF).is_none()
            && matches!(name, "buffer" | "byteLength" | "byteOffset")
            && self.brand_on_chain(handle, DATA_VIEW_PROTO_BRAND)
        {
            return Err(self.branded_accessor_type_error(
                handle,
                DATA_VIEW_PROTO_BRAND,
                name,
                "DataView.prototype accessor called on a non-DataView object",
            ));
        }
        if self.realm.typed_kind(handle).is_none()
            && matches!(name, "buffer" | "byteLength" | "byteOffset" | "length")
            // An own property on the receiver shadows the inherited branded accessor
            // (ordinary [[Get]] finds the own property first). Most visibly, an Array
            // or String-wrapper receiver whose `[[Prototype]]` was set to a typed
            // array (`Object.setPrototypeOf([], ta)`) still reads its *own* `length`.
            && !self.realm.has_own(handle, name)
            && !(name == "length"
                && (self.realm.is_array(handle)
                    || self.realm.string_object_len(handle).is_some()))
            && self.brand_on_chain(handle, TYPED_ARRAY_PROTO_BRAND)
        {
            return Err(self.branded_accessor_type_error(
                handle,
                TYPED_ARRAY_PROTO_BRAND,
                name,
                "TypedArray.prototype accessor called on a non-TypedArray object",
            ));
        }
        // `ArrayBuffer.prototype` methods (`slice`/`resize`/`transfer`/
        // `transferToFixedLength`) are installed as real first-class own properties on
        // the prototype (with proper name/length), and every `ArrayBuffer` instance
        // inherits the prototype — so a read of `ab.slice` resolves them through the
        // chain (and a user write to `ArrayBuffer.prototype.slice` is honored). No
        // special case needed here.
        // `ArrayBuffer.prototype.resizable` / `.maxByteLength` (ES2024 resizable buffers).
        if matches!(name, "resizable" | "maxByteLength")
            && self
                .realm
                .get_property(handle, ARRAY_BUFFER_BYTES)
                .is_some()
        {
            let max = self.realm.get_property(handle, ARRAY_BUFFER_MAXLEN);
            if name == "resizable" {
                return Ok(NanBox::boolean(max.is_some()));
            }
            // `maxByteLength` is the recorded max, or — for a non-resizable buffer — its
            // current `byteLength`.
            return Ok(match max {
                Some(m) => m,
                None => self.read_member(handle, "byteLength")?,
            });
        }
        // `ArrayBuffer.prototype.detached` — true once `transfer()` has emptied it.
        if name == "detached"
            && self
                .realm
                .get_property(handle, ARRAY_BUFFER_BYTES)
                .is_some()
        {
            let detached = self
                .realm
                .get_property(handle, ARRAY_BUFFER_DETACHED)
                .is_some();
            return Ok(NanBox::boolean(detached));
        }
        // `ArrayBuffer.byteLength` (the byte store's length; 0 once detached).
        if name == "byteLength"
            && let Some(b) = self.realm.get_property(handle, ARRAY_BUFFER_BYTES)
            && let Some(bh) = b.as_handle().map(Handle::from_raw)
        {
            if self
                .realm
                .get_property(handle, ARRAY_BUFFER_DETACHED)
                .is_some()
            {
                return Ok(NanBox::number(0.0));
            }
            return Ok(NanBox::number(self.realm.bytes_len(bh).unwrap_or(0) as f64));
        }
        // `DataView.prototype` get*/set* methods are installed as real first-class
        // own properties on the prototype (with proper name/length), so a read of
        // `dv.getInt8` resolves them through the prototype chain — no special case.
        // `DataView.byteLength` / `.buffer` / `.byteOffset`.
        if matches!(name, "byteLength" | "buffer" | "byteOffset")
            && let Some(buf) = self.realm.get_property(handle, DATA_VIEW_BUF)
        {
            // `get DataView.prototype.byteLength`/`.byteOffset` throw a TypeError when
            // the viewed buffer is detached (`.buffer` does not — it returns it).
            if matches!(name, "byteLength" | "byteOffset")
                && let Some(bh) = buf.as_handle().map(Handle::from_raw)
                && self.realm.get_property(bh, ARRAY_BUFFER_DETACHED).is_some()
            {
                return Err(
                    self.type_error("Cannot perform DataView operation on a detached ArrayBuffer")
                );
            }
            // IsViewOutOfBounds: a resizable buffer shrank under the view — its
            // `byteLength`/`byteOffset` getters then throw a TypeError. A
            // length-tracking DataView (no recorded length) is out of bounds only
            // when its offset alone is past the current end; a fixed-length view
            // when its offset+length no longer fits.
            if matches!(name, "byteLength" | "byteOffset")
                && let Some(bh) = buf.as_handle().map(Handle::from_raw)
            {
                let total = self
                    .array_buffer_bytes(bh)
                    .and_then(|b| self.realm.bytes_len(b))
                    .unwrap_or(0);
                let off = self
                    .realm
                    .get_property(handle, DATA_VIEW_OFF)
                    .and_then(|n| n.as_number())
                    .unwrap_or(0.0) as usize;
                let recorded = self
                    .realm
                    .get_property(handle, DATA_VIEW_LEN)
                    .and_then(|n| n.as_number())
                    .map(|n| n as usize);
                let oob = match recorded {
                    Some(len) => off.checked_add(len).is_none_or(|end| end > total),
                    None => off > total,
                };
                if oob {
                    return Err(
                        self.type_error("get DataView.prototype accessor on an out-of-bounds view")
                    );
                }
            }
            return Ok(match name {
                "buffer" => buf,
                "byteOffset" => self
                    .realm
                    .get_property(handle, DATA_VIEW_OFF)
                    .unwrap_or(NanBox::number(0.0)),
                _ => {
                    // An explicit byteLength wins; else the rest of the buffer.
                    if let Some(len) = self
                        .realm
                        .get_property(handle, DATA_VIEW_LEN)
                        .and_then(|n| n.as_number())
                    {
                        return Ok(NanBox::number(len));
                    }
                    let total = buf
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.array_buffer_bytes(h))
                        .and_then(|bh| self.realm.bytes_len(bh))
                        .unwrap_or(0);
                    let off = self
                        .realm
                        .get_property(handle, DATA_VIEW_OFF)
                        .and_then(|n| n.as_number())
                        .unwrap_or(0.0) as usize;
                    NanBox::number(total.saturating_sub(off) as f64)
                }
            });
        }
        // Static `<TypedArray>.BYTES_PER_ELEMENT` (on the constructor itself).
        if name == "BYTES_PER_ELEMENT"
            && let Some(id) = self.realm.native_at(handle)
            && (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16)
                .contains(&id)
        {
            return Ok(NanBox::number(f64::from(
                TYPED_ARRAY_KINDS[(id - N_TYPED_ARRAY_BASE) as usize].1,
            )));
        }
        // A typed array's `.buffer` — its `[[ViewedArrayBuffer]]` object, returned
        // directly so it is SameValue-stable and shared with sibling views.
        if name == "buffer"
            && let Some(buf) = self.realm.typed_array_object(handle)
        {
            return Ok(NanBox::handle(buf.to_raw()));
        }
        // Typed-array-specific methods that aren't shared with `Array.prototype`
        // (`set`/`subarray`), exposed as readable methods.
        if matches!(name, "set" | "subarray") && self.realm.typed_kind(handle).is_some() {
            return Ok(self.readable_native_method(name));
        }
        // Typed-array introspection (`byteLength`, `BYTES_PER_ELEMENT`, `byteOffset`).
        if matches!(name, "byteLength" | "BYTES_PER_ELEMENT" | "byteOffset")
            && let Some(kind) = self.realm.typed_kind(handle)
        {
            let bpe = f64::from(TYPED_ARRAY_KINDS[kind as usize].1);
            // A detached or out-of-bounds view reports byteOffset 0 (and typed_len,
            // used for byteLength, already collapses to 0).
            let oob =
                self.typed_array_detached(handle) || self.realm.typed_array_out_of_bounds(handle);
            return Ok(NanBox::number(match name {
                "BYTES_PER_ELEMENT" => bpe,
                "byteOffset" if oob => 0.0,
                "byteOffset" => self.realm.typed_byte_offset(handle).unwrap_or(0) as f64,
                _ => self.realm.typed_len(handle).unwrap_or(0) as f64 * bpe,
            }));
        }
        // A String wrapper delegates `length` and indexed reads to its boxed
        // string (`new String("hi").length`, `wrapper[0]`). P3: take the borrowing
        // leaf path for `length`/indexed reads (the hot ones) and fall back to the
        // owned bytes only for a `Concat` rope.
        if let Some(prim) = self.realm.get_property(handle, PRIM_WRAP)
            && let Some(ph) = prim.as_handle().map(Handle::from_raw)
            && self.realm.is_string_handle(ph)
        {
            if name == "length" {
                let len = self.realm.string_utf16_len(ph).unwrap_or(0);
                return Ok(NanBox::number(len as f64));
            }
            if let Ok(i) = name.parse::<usize>() {
                let unit = if let Some(leaf) = self.realm.string_leaf_bytes(ph) {
                    crate::wtf8::utf16_index(leaf, i)
                } else {
                    crate::wtf8::utf16_index(&self.realm.string_bytes(ph).unwrap_or_default(), i)
                };
                if let Some(u) = unit {
                    return Ok(self.new_str_bytes(crate::wtf8::from_utf16(&[u])));
                }
                // Out of range: String-exotic `[[GetOwnProperty]]` falls back to
                // OrdinaryGetOwnProperty, so an own property defined at that index on
                // the *wrapper* (`Object.defineProperty(new String("s"), "4", …)`) is
                // still read. Only shortcut to `undefined` when there is none.
                if !self.realm.has_own(handle, name) {
                    return Ok(NanBox::undefined());
                }
            }
            let v = self.member_value(ph, name);
            if !matches!(v.unpack(), Unpacked::Undefined) {
                return Ok(v);
            }
        }
        // Own property (or a built-in like `length`) wins.
        let direct = self.member_value(handle, name);
        if !matches!(direct.unpack(), Unpacked::Undefined) || self.realm.has_own(handle, name) {
            return Ok(direct);
        }
        // Otherwise walk the `[[Prototype]]` chain for an inherited property or
        // accessor (the receiver stays `handle`).
        let mut cur = self.realm.object_proto(handle);
        // A built-in primitive/exotic cell (a string, an array, a function, a
        // Map/Set) carries no explicit `[[Prototype]]` link — its chain starts at
        // the matching intrinsic prototype. Seeding the walk there (rather than
        // leaving it to the own-property-only `builtin_proto_method` fallback
        // below) makes an inherited *accessor* run with the primitive as `this`
        // and lets the walk continue up to `%Object.prototype%`, so
        // `String.prototype.p` defined as a getter, or a method installed on
        // `Object.prototype`, is visible on `"str"`.
        if cur.is_none() {
            cur = self.builtin_proto_of(handle);
        }
        while let Some(p) = cur {
            // A proxy in the prototype chain handles the read via its own `[[Get]]`
            // (a `get` trap, or forwarding to the target and its prototype chain),
            // which is terminal for the lookup. The Receiver stays the original
            // object so an inherited accessor getter runs with the right `this`.
            if self.realm.proxy_at(p).is_some() {
                return self.get_with_receiver(p, name, NanBox::handle(handle.to_raw()));
            }
            if let Some((getter, _)) = self.realm.accessor(p, name) {
                if matches!(getter.unpack(), Unpacked::Undefined) {
                    return Ok(NanBox::undefined());
                }
                let this = NanBox::handle(handle.to_raw());
                return self.call_with_this(getter, this, &[]);
            }
            // A prototype that is itself an Array (or typed array) exposes its
            // elements and `length` as inherited indexed/`length` properties —
            // so `Object.create([1,2,3])[0]`/`.length` resolve when the chain
            // reaches the backing array (`get_property` only reads an array's
            // *aux* named props, never its elements).
            if self.realm.is_array_like(p) {
                if let Ok(i) = name.parse::<usize>()
                    && alloc::format!("{i}") == name
                {
                    if i < self.realm.array_length(p).unwrap_or(0) {
                        let v = self.realm.get_element(p, i);
                        // A hole on a prototype array is also absent — keep walking.
                        if !v.is_hole() {
                            return Ok(v);
                        }
                    }
                } else if name == "length"
                    && let Some(len) = self.realm.array_length(p)
                {
                    return Ok(NanBox::number(len as f64));
                }
            }
            if self.realm.has_own(p, name) {
                return Ok(self
                    .realm
                    .get_property(p, name)
                    .unwrap_or(NanBox::undefined()));
            }
            cur = self.realm.object_proto(p);
        }
        // A built-in value with no own/inherited `constructor` reports its global
        // constructor (`[].constructor === Array`); user functions/classes resolve
        // theirs through the prototype walk above and never reach here.
        if name == "constructor"
            && let Some(ctor) = self.builtin_constructor_for(handle)
        {
            return Ok(ctor);
        }
        // A built-in array/string/function exposes its prototype's methods as
        // first-class values — so feature detection (`if (arr.flat)`,
        // `typeof str.padStart`) and detached-method access resolve. (Ordinary
        // `recv.m(args)` calls dispatch via `call_method` and never reach here.)
        if let Some(m) = self.builtin_proto_method(handle, name) {
            return Ok(m);
        }
        Ok(direct)
    }

    /// For a built-in array/string/function value, the first-class method `name`
    /// from its constructor's prototype (`Array.prototype` etc.), or `None`.
    pub(crate) fn builtin_proto_method(&mut self, handle: Handle, name: &str) -> Option<NanBox> {
        let proto = self.builtin_proto_of(handle)?;
        let m = self.realm.get_property(proto, name)?;
        (!matches!(m.unpack(), Unpacked::Undefined)).then_some(m)
    }

    /// The intrinsic prototype a built-in cell with no explicit `[[Prototype]]`
    /// link inherits from (`%String.prototype%` for a string cell,
    /// `%Array.prototype%` for an array, …), or `None` for anything else.
    pub(crate) fn builtin_proto_of(&mut self, handle: Handle) -> Option<Handle> {
        let ctor_name = if self.realm.is_string_handle(handle) {
            "String"
        } else if self.realm.is_array_like(handle) {
            "Array"
        } else if let Some(is_set) = self.realm.collection_is_set(handle) {
            // A *weak* collection inherits from `%WeakMap/WeakSet.prototype%`, not
            // the strong `%Map/Set.prototype%` — conflating them resolved a WeakMap's
            // first-class members (e.g. a `Symbol.toStringTag` fallback) from
            // `Map.prototype`, wrongly reporting `[object Map]`.
            match (self.realm.collection_is_weak(handle), is_set) {
                (true, true) => "WeakSet",
                (true, false) => "WeakMap",
                (false, true) => "Set",
                (false, false) => "Map",
            }
        } else if self.realm.native_at(handle).is_some()
            || self.realm.bound_native_at(handle).is_some()
        {
            "Function"
        } else {
            return None;
        };
        self.current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|ns| self.realm.get_property(ns, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
    }

    pub(crate) fn unary(&mut self, op: UnaryOp, v: NanBox) -> Result<NanBox, ExecError> {
        // BigInt negation / bitwise-not stay BigInt.
        if let Some(big) = v
            .as_handle()
            .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)))
        {
            match op {
                UnaryOp::Minus => {
                    return Ok(NanBox::handle(self.realm.new_bigint(big.neg()).to_raw()));
                }
                UnaryOp::BitNot => {
                    // `~x` on a BigInt is `-(x + 1)`.
                    let one = crate::bignum::BigInt::from_i128(1);
                    let nx = big.add(&one).neg();
                    return Ok(NanBox::handle(self.realm.new_bigint(nx).to_raw()));
                }
                UnaryOp::Not => return Ok(NanBox::boolean(big.is_zero())),
                _ => {}
            }
        }
        // A Symbol cannot be converted to a number (unary `+`/`-`/`~`).
        if matches!(op, UnaryOp::Plus | UnaryOp::Minus | UnaryOp::BitNot)
            && v.as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.realm.symbol_at(h).is_some())
        {
            let m = self.new_str("Cannot convert a Symbol value to a number");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // For the numeric unary operators, ToPrimitive(number) may surface a boxed
        // Symbol/BigInt (e.g. `+Object(Symbol())`, `-Object(1n)`). ToNumber then
        // throws a TypeError for a Symbol and for a BigInt under `+`; `-`/`~` on a
        // BigInt stay BigInt (ToNumeric).
        if matches!(op, UnaryOp::Plus | UnaryOp::Minus | UnaryOp::BitNot) {
            let p = self.coerce_object(v, "number")?;
            if let Some(h) = p.as_handle().map(Handle::from_raw) {
                if self.realm.symbol_at(h).is_some() {
                    let m = self.new_str("Cannot convert a Symbol value to a number");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                if let Some(big) = self.realm.bigint_at(h) {
                    return match op {
                        UnaryOp::Minus => {
                            Ok(NanBox::handle(self.realm.new_bigint(big.neg()).to_raw()))
                        }
                        #[cfg(feature = "std")]
                        UnaryOp::BitNot => {
                            let one = crate::bignum::BigInt::from_i128(1);
                            let nx = big.add(&one).neg();
                            Ok(NanBox::handle(self.realm.new_bigint(nx).to_raw()))
                        }
                        _ => {
                            let m = self.new_str("Cannot convert a BigInt value to a number");
                            Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
                        }
                    };
                }
            }
            return Ok(match op {
                UnaryOp::Plus => NanBox::number(self.realm.to_number(p)),
                UnaryOp::Minus => self.realm.neg(p),
                #[cfg(feature = "std")]
                UnaryOp::BitNot => self.realm.bit_not(p),
                #[cfg(not(feature = "std"))]
                UnaryOp::BitNot => return Err(ExecError::Unsupported("~ needs std")),
                _ => unreachable!(),
            });
        }
        Ok(match op {
            UnaryOp::Not => self.realm.logical_not(v),
            UnaryOp::Typeof => {
                let t = self.realm.type_of_value(v);
                NanBox::handle(self.realm.new_string(t).to_raw())
            }
            UnaryOp::Void => NanBox::undefined(),
            UnaryOp::Plus | UnaryOp::Minus | UnaryOp::BitNot => unreachable!(),
            UnaryOp::Delete => return Err(ExecError::Unsupported("delete")),
        })
    }

    /// The BigInt operator path. Returns `None` to fall through (e.g. `bigint +
    /// string` is string concatenation). Both operands BigInt → i128 arithmetic;
    /// a mix with a Number throws a `TypeError` for arithmetic but compares
    /// numerically for `<`/`==`.
    pub(crate) fn bigint_binary(
        &mut self,
        op: BinaryOp,
        abig: Option<crate::bignum::BigInt>,
        bbig: Option<crate::bignum::BigInt>,
        a: NanBox,
        b: NanBox,
    ) -> Result<Option<NanBox>, ExecError> {
        // Strict equality: equal only if both are BigInt with the same value.
        match op {
            BinaryOp::EqEqEq => return Ok(Some(NanBox::boolean(abig.is_some() && abig == bbig))),
            BinaryOp::NotEqEq => {
                return Ok(Some(NanBox::boolean(!(abig.is_some() && abig == bbig))));
            }
            _ => {}
        }
        if let (Some(x), Some(y)) = (abig.clone(), bbig.clone()) {
            use core::cmp::Ordering;
            let val = |this: &mut Self, n: crate::bignum::BigInt| {
                NanBox::handle(this.realm.new_bigint(n).to_raw())
            };
            let throw = |this: &mut Self, msg: &str| {
                let m = this.new_str(msg);
                ExecError::Throw(this.make_error(N_TYPE_ERROR, Some(m)))
            };
            // BigInt division/remainder by zero and a negative exponent are
            // RangeErrors (not TypeErrors) per BigInt::divide/remainder/exponentiate.
            let range_throw = |this: &mut Self, msg: &str| {
                let m = this.new_str(msg);
                ExecError::Throw(this.make_error(N_RANGE_ERROR, Some(m)))
            };
            let r = match op {
                BinaryOp::Add => val(self, x.add(&y)),
                BinaryOp::Sub => val(self, x.sub(&y)),
                BinaryOp::Mul => val(self, x.mul(&y)),
                BinaryOp::Div => match x.divmod(&y) {
                    Some((q, _)) => val(self, q),
                    None => return Err(range_throw(self, "Division by zero")),
                },
                BinaryOp::Mod => match x.divmod(&y) {
                    Some((_, rem)) => val(self, rem),
                    None => return Err(range_throw(self, "Division by zero")),
                },
                BinaryOp::Exp => {
                    if y.is_negative() {
                        return Err(range_throw(self, "Exponent must be non-negative"));
                    }
                    let e = y.to_i128().and_then(|v| u64::try_from(v).ok()).unwrap_or(0);
                    // Projected result size ≈ bit_len(x) × e. `try_pow` rejects
                    // before the (possibly multi-GB) allocation, else `2n ** 1e10n`
                    // OOMs. Belt and suspenders: the same cap is enforced here so
                    // the error path is unmistakable.
                    let Some(p) = x.try_pow(e, self.realm.limits.max_bigint_bits) else {
                        let m = self.new_str("Maximum BigInt size exceeded");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    };
                    val(self, p)
                }
                // Two's-complement bitwise ops at arbitrary precision.
                BinaryOp::BitAnd => val(self, x.bitand(&y)),
                BinaryOp::BitOr => val(self, x.bitor(&y)),
                BinaryOp::BitXor => val(self, x.bitxor(&y)),
                // `<<`/`>>` as multiply/floor-divide by `2^n` (a negative shift
                // count reverses direction). BigInts have no unsigned `>>>`.
                BinaryOp::Shl | BinaryOp::Shr => {
                    let two = crate::bignum::BigInt::from_i128(2);
                    let count = y.to_i128().unwrap_or(0);
                    // `>>` is `<<` by the negated count, and vice versa.
                    let left = (op == BinaryOp::Shl) == (count >= 0);
                    let mag = u64::try_from(count.unsigned_abs()).unwrap_or(0);
                    // A left shift grows the result to ≈ bit_len(x) + mag bits;
                    // reject an attacker count before building `2^mag`. (A right
                    // shift only shrinks, so it needs no bound — but `2^mag` is
                    // still built, so cap the exponent itself.)
                    let projected = if left {
                        x.bit_len().saturating_add(mag)
                    } else {
                        mag
                    };
                    if projected > self.realm.limits.max_bigint_bits {
                        let m = self.new_str("Maximum BigInt size exceeded");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                    let pow2 = two.pow(mag);
                    if left {
                        val(self, x.mul(&pow2))
                    } else {
                        match x.divmod(&pow2) {
                            // Arithmetic shift floors; truncating divmod needs a
                            // `-1` correction for a negative value with a remainder.
                            Some((q, rem)) => {
                                if x.is_negative() && !rem.is_zero() {
                                    val(self, q.sub(&crate::bignum::BigInt::from_i128(1)))
                                } else {
                                    val(self, q)
                                }
                            }
                            None => val(self, crate::bignum::BigInt::zero()),
                        }
                    }
                }
                BinaryOp::Ushr => {
                    return Err(throw(self, "BigInts have no unsigned right shift"));
                }
                BinaryOp::Lt => NanBox::boolean(x.cmp(&y) == Ordering::Less),
                BinaryOp::Gt => NanBox::boolean(x.cmp(&y) == Ordering::Greater),
                BinaryOp::LtEq => NanBox::boolean(x.cmp(&y) != Ordering::Greater),
                BinaryOp::GtEq => NanBox::boolean(x.cmp(&y) != Ordering::Less),
                BinaryOp::EqEq => NanBox::boolean(x == y),
                BinaryOp::NotEq => NanBox::boolean(x != y),
                _ => return Ok(None),
            };
            return Ok(Some(r));
        }
        // Mixed: `bigint + string` (either side a string) → string concat.
        if matches!(op, BinaryOp::Add) {
            let is_str = |this: &Self, v: NanBox| {
                v.as_handle()
                    .is_some_and(|raw| this.realm.is_string_handle(Handle::from_raw(raw)))
            };
            if is_str(self, a) || is_str(self, b) {
                return Ok(None);
            }
        }
        // BigInt vs a non-BigInt primitive: exactly one operand is a BigInt (the
        // both-BigInt case returned above). Equality and the relational operators
        // compare per spec — a String coerces via StringToBigInt (an invalid string
        // is "undefined", i.e. never equal / an undefined ordering → `false`), a
        // Number/Boolean/null compares *mathematically exactly* (no lossy `f64`
        // round-trip), a Symbol throws for a relational compare (and is unequal for
        // `==`), and `undefined` is incomparable.
        if matches!(
            op,
            BinaryOp::EqEq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::LtEq
                | BinaryOp::GtEq
        ) {
            use core::cmp::Ordering;
            let is_equality = matches!(op, BinaryOp::EqEq | BinaryOp::NotEq);
            let is_relational = !is_equality;
            // The single BigInt operand and whether it is the left-hand side.
            let (big, big_left) = match (&abig, &bbig) {
                (Some(x), _) => (x.clone(), true),
                (_, Some(y)) => (y.clone(), false),
                _ => return Ok(None),
            };
            let other = if big_left { b } else { a };
            // Resolve `other` to something comparable to a BigInt.
            enum Rhs {
                Big(crate::bignum::BigInt),
                Num(f64),
                Incomparable,
            }
            let resolved = match other.unpack() {
                Unpacked::Number(n) => Rhs::Num(n),
                Unpacked::Bool(bl) => Rhs::Num(if bl { 1.0 } else { 0.0 }),
                // `==` null/undefined → not equal; a relational compares numerically
                // (ToNumeric(null) = 0, ToNumeric(undefined) = NaN → incomparable).
                Unpacked::Null => {
                    if is_equality {
                        Rhs::Incomparable
                    } else {
                        Rhs::Num(0.0)
                    }
                }
                Unpacked::Undefined => Rhs::Incomparable,
                Unpacked::Handle(raw) => {
                    let h = Handle::from_raw(raw);
                    if let Some(s) = self.realm.string_value(h) {
                        match string_to_bigint_opt(&s) {
                            Some(nb) => Rhs::Big(nb),
                            None => Rhs::Incomparable,
                        }
                    } else if self.realm.symbol_at(h).is_some() {
                        if is_relational {
                            let m = self.new_str("Cannot convert a Symbol value to a number");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        Rhs::Incomparable
                    } else {
                        // A coercible object is excluded upstream; anything else is
                        // not a BigInt-comparable primitive — defer.
                        return Ok(None);
                    }
                }
            };
            // `ord` is `big` compared against `other`; flip when the BigInt is on
            // the right so it reflects the source (left-vs-right) order.
            let ord = match resolved {
                Rhs::Incomparable => None,
                Rhs::Big(ob) => Some(big.cmp(&ob)),
                Rhs::Num(n) => bigint_cmp_f64(&big, n),
            };
            let ord = if big_left {
                ord
            } else {
                ord.map(Ordering::reverse)
            };
            let r = match op {
                BinaryOp::EqEq => ord == Some(Ordering::Equal),
                BinaryOp::NotEq => ord != Some(Ordering::Equal),
                BinaryOp::Lt => ord == Some(Ordering::Less),
                BinaryOp::Gt => ord == Some(Ordering::Greater),
                BinaryOp::LtEq => matches!(ord, Some(Ordering::Less | Ordering::Equal)),
                _ => matches!(ord, Some(Ordering::Greater | Ordering::Equal)),
            };
            return Ok(Some(NanBox::boolean(r)));
        }
        // Mixed arithmetic is a TypeError.
        let m = self.new_str("Cannot mix BigInt and other types");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    /// `ToNumber`'s Symbol guard: a Symbol primitive has no numeric conversion, so
    /// `ToNumeric`/`ToNumber` on one is a `TypeError`. Used to reject a lhs Symbol
    /// mid-`ToNumeric` before the rhs is converted (spec operand order).
    fn throw_if_symbol_to_number(&mut self, v: NanBox) -> Result<(), ExecError> {
        if v.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.symbol_at(h).is_some())
        {
            let m = self.new_str("Cannot convert a Symbol value to a number");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    pub(crate) fn binary(
        &mut self,
        op: BinaryOp,
        a: NanBox,
        b: NanBox,
    ) -> Result<NanBox, ExecError> {
        // An operand that is a wrapper/plain object (not a bigint/string primitive
        // or symbol) must be ToPrimitive-coerced *before* the BigInt path, so a
        // BigInt wrapper (`Object(1n)`) or a `Symbol.toPrimitive` yielding a BigInt
        // is unwrapped first. Defer the BigInt check in that case.
        let is_coercible_object = |this: &Self, v: NanBox| {
            v.as_handle().map(Handle::from_raw).is_some_and(|h| {
                this.realm.bigint_at(h).is_none()
                    && !this.realm.is_string_handle(h)
                    && this.realm.symbol_at(h).is_none()
            })
        };
        // BigInt operands take a dedicated path (i128 arithmetic; mixing with
        // other numeric types throws, per the spec).
        let abig = a
            .as_handle()
            .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
        let bbig = b
            .as_handle()
            .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
        if (abig.is_some() || bbig.is_some())
            && !is_coercible_object(self, a)
            && !is_coercible_object(self, b)
            && let Some(r) = self.bigint_binary(op, abig, bbig, a, b)?
        {
            return Ok(r);
        }
        // IsLooselyEqual with exactly one **Object** operand (the other a
        // non-nullish primitive) runs `ToPrimitive(obj)` with the *default* hint and
        // retries, so a user `valueOf`/`toString`/`@@toPrimitive` is honored
        // (`new Number() == 0`, `new Date == true` → the Date's `toString`).
        // `Realm::loose_equals` is `&self` and cannot call into JS, so it is done
        // here. An Object-vs-Object comparison is identity (no coercion), and
        // `null`/`undefined` never coerce — that keeps the `IsHTMLDDA` special case
        // in `loose_equals` reachable.
        if matches!(op, BinaryOp::EqEq | BinaryOp::NotEq) {
            let nullish = |v: NanBox| matches!(v.unpack(), Unpacked::Undefined | Unpacked::Null);
            let (a, b) = if self.is_object_value(a) && !self.is_object_value(b) && !nullish(b) {
                (self.coerce_primitive(a, "default")?, b)
            } else if self.is_object_value(b) && !self.is_object_value(a) && !nullish(a) {
                (a, self.coerce_primitive(b, "default")?)
            } else {
                (a, b)
            };
            // The coercion may have produced a String (or Number) facing a BigInt —
            // `0n == {toString(){return "0"}}`. That pairing is StringToBigInt /
            // mathematical-value equality, which `loose_equals` (a `&self` reference
            // comparison) cannot do; the BigInt path above ran too early to see it.
            let abig = a
                .as_handle()
                .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
            let bbig = b
                .as_handle()
                .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
            if (abig.is_some() || bbig.is_some())
                && let Some(r) = self.bigint_binary(op, abig, bbig, a, b)?
            {
                return Ok(r);
            }
            let eq = self.realm.loose_equals(a, b);
            return Ok(NanBox::boolean(if matches!(op, BinaryOp::EqEq) {
                eq
            } else {
                !eq
            }));
        }
        // Arithmetic and relational operators apply ToPrimitive to object
        // operands (`valueOf`/`toString`); equality/`instanceof`/`in` do not.
        let coerces = matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Mod
                | BinaryOp::Exp
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::LtEq
                | BinaryOp::GtEq
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::Ushr
                | BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor
        );
        // `+` uses the "default" hint; the other numeric operators use "number".
        let hint = if matches!(op, BinaryOp::Add) {
            "default"
        } else {
            "number"
        };
        let (a, b) = if coerces && (a.as_handle().is_some() || b.as_handle().is_some()) {
            // A multiplicative/additive/bitwise/shift operator applies
            // `ToNumeric(lhs)` *fully* — ToPrimitive **and** ToNumber, the latter
            // throwing for a Symbol — before touching the rhs, so a lhs whose
            // conversion throws never evaluates the rhs's `valueOf`
            // (`order-of-evaluation`). `+` and the relational operators instead
            // ToPrimitive *both* operands first (a Symbol only throws at the later
            // ToNumeric/ToString step), so they coerce as a pair.
            let sequential = !matches!(
                op,
                BinaryOp::Add | BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq
            );
            if sequential {
                let a = self.coerce_primitive(a, hint)?;
                self.throw_if_symbol_to_number(a)?;
                let b = self.coerce_primitive(b, hint)?;
                self.throw_if_symbol_to_number(b)?;
                (a, b)
            } else {
                (
                    self.coerce_primitive(a, hint)?,
                    self.coerce_primitive(b, hint)?,
                )
            }
        } else {
            (a, b)
        };
        // ToPrimitive may have unwrapped a BigInt wrapper object (`Object(1n)`) or a
        // `Symbol.toPrimitive` returning a BigInt; retry the BigInt path now that the
        // operands are primitives (`Object(5n) & 3n` → `1n`).
        if coerces {
            let abig = a
                .as_handle()
                .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
            let bbig = b
                .as_handle()
                .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
            if (abig.is_some() || bbig.is_some())
                && let Some(r) = self.bigint_binary(op, abig, bbig, a, b)?
            {
                return Ok(r);
            }
        }
        // `==`/`!=` between an object/array and a number/string primitive coerces
        // the object side (arrays via their join; plain objects via ToPrimitive).
        let (a, b) = if matches!(op, BinaryOp::EqEq | BinaryOp::NotEq) {
            // True for a real object/array — a heap value that is *not* itself a
            // primitive (string / Symbol / BigInt cells are primitives, and
            // ToPrimitive on them is a no-op, so they are not the "object" side).
            let obj = |this: &Self, v: NanBox| {
                v.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    !this.realm.is_string_handle(h)
                        && this.realm.symbol_at(h).is_none()
                        && this.realm.bigint_at(h).is_none()
                })
            };
            // True for any primitive against which an object is converted with
            // ToPrimitive per the `==` algorithm — a Number, Boolean, String,
            // Symbol, or BigInt (so `0n == Object(0n)` and `sym == Object(sym)`
            // coerce the object side and then compare as primitives).
            let prim = |this: &Self, v: NanBox| {
                v.as_number().is_some()
                    || matches!(v.unpack(), crate::nanbox::Unpacked::Bool(_))
                    || v.as_handle().map(Handle::from_raw).is_some_and(|h| {
                        this.realm.is_string_handle(h)
                            || this.realm.symbol_at(h).is_some()
                            || this.realm.bigint_at(h).is_some()
                    })
            };
            let (a, b) = if obj(self, a) && prim(self, b) {
                (self.coerce_for_eq(a)?, b)
            } else if obj(self, b) && prim(self, a) {
                (a, self.coerce_for_eq(b)?)
            } else {
                (a, b)
            };
            // ToPrimitive of the object side may have produced a BigInt (a BigInt
            // wrapper / a `valueOf` returning a BigInt) or a String to compare
            // against a BigInt: re-run the dedicated BigInt equality path so
            // `bigintN == { toString(){ return "N" } }` applies StringToBigInt
            // rather than a mismatched cross-cell `strict_equals`.
            let abig = a
                .as_handle()
                .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
            let bbig = b
                .as_handle()
                .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
            if (abig.is_some() || bbig.is_some())
                && let Some(r) = self.bigint_binary(op, abig, bbig, a, b)?
            {
                return Ok(r);
            }
            (a, b)
        } else {
            (a, b)
        };
        // A Symbol cannot be implicitly converted to a number or string, so any
        // arithmetic/relational operator on one throws a TypeError.
        if coerces {
            let is_sym = |this: &Self, v: NanBox| {
                v.as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| this.realm.symbol_at(h).is_some())
            };
            if is_sym(self, a) || is_sym(self, b) {
                let msg = if matches!(op, BinaryOp::Add) {
                    "Cannot convert a Symbol value to a string"
                } else {
                    "Cannot convert a Symbol value to a number"
                };
                let m = self.new_str(msg);
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        Ok(match op {
            BinaryOp::Add => match self.realm.add_checked(a, b) {
                Some(v) => v,
                None => {
                    let m = self.new_str("Invalid string length");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
            },
            BinaryOp::Sub => self.realm.sub(a, b),
            BinaryOp::Mul => self.realm.mul(a, b),
            BinaryOp::Div => self.realm.div(a, b),
            BinaryOp::Mod => self.realm.rem(a, b),
            BinaryOp::Lt => self.realm.less_than(a, b),
            BinaryOp::Gt => self.realm.greater_than(a, b),
            BinaryOp::LtEq => self.realm.less_equal(a, b),
            BinaryOp::GtEq => self.realm.greater_equal(a, b),
            BinaryOp::EqEq => NanBox::boolean(self.realm.loose_equals(a, b)),
            BinaryOp::NotEq => NanBox::boolean(!self.realm.loose_equals(a, b)),
            BinaryOp::EqEqEq => NanBox::boolean(self.realm.strict_equals(a, b)),
            BinaryOp::NotEqEq => NanBox::boolean(!self.realm.strict_equals(a, b)),
            #[cfg(feature = "std")]
            BinaryOp::Exp => self.realm.pow(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Shl => self.realm.shl(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Shr => self.realm.shr(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Ushr => self.realm.ushr(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitAnd => self.realm.bit_and(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitOr => self.realm.bit_or(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitXor => self.realm.bit_xor(a, b),
            #[cfg(not(feature = "std"))]
            BinaryOp::Exp
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::Ushr
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor => return Err(ExecError::Unsupported("** / bitwise need std")),
            BinaryOp::In => {
                // The right operand must be an object (a primitive is a TypeError).
                if !self.is_object_value(b) {
                    let m = self.new_str("Cannot use 'in' operator to search in a non-object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // `ToPropertyKey(a)`: an object left operand goes through
                // ToPrimitive, so a Symbol *wrapper* (`Object(sym) in obj`) keys on
                // the wrapped symbol, and `new String("x") in obj` (or any object
                // with a `toString`) keys on the converted value — `member_key`
                // alone would key on the display string.
                let key = self.coerce_property_key(a)?;
                // A Deferred Module Namespace (`import defer`) evaluates its target
                // on a `[[HasProperty]]` with a String (non-"then") key — directly
                // or anywhere in the prototype chain.
                #[cfg(all(feature = "module", feature = "std"))]
                if let Some(h) = b.as_handle().map(Handle::from_raw) {
                    self.trigger_deferred_in_chain(h, &key)?;
                }
                // The full `[[HasProperty]]`: a proxy `has` trap (or forwarding to
                // the target — which may itself be a proxy), typed-array integer
                // indices, and an ordinary own-or-inherited (accessor-aware) chain
                // walk. Delegating keeps the `in` operator consistent with member
                // lookup instead of re-deriving (and previously mis-deriving) it.
                let present = match b.as_handle().map(Handle::from_raw) {
                    Some(h) => self.has_property_proxied(h, &key)?,
                    None => false,
                };
                NanBox::boolean(present)
            }
            BinaryOp::Instanceof => NanBox::boolean(self.instance_of(a, b)?),
        })
    }

    /// `obj instanceof Ctor`: true when `obj` was constructed from `Ctor`'s
    /// class or one of its subclasses (via the instance's class tag and the
    /// `extends` chain).
    /// `OrdinaryHasInstance(C, O)` for `Function.prototype[Symbol.hasInstance]`:
    /// `false` if `C` is not callable; a bound function defers to its target;
    /// otherwise walk `O`'s `[[Prototype]]` chain for `C.prototype`. `instance_of`
    /// already implements this (and skips the default `@@hasInstance` to avoid
    /// recursion), so delegate with the arguments in instanceof order.
    pub(crate) fn ordinary_has_instance(
        &mut self,
        c: NanBox,
        o: NanBox,
    ) -> Result<bool, ExecError> {
        // IsCallable(C): a non-callable `this` reports `false` (no throw). The
        // `Get(C,"prototype")` must-be-Object check (a TypeError otherwise) is
        // performed inside `instance_of`'s ordinary path.
        let Some(ch) = c.as_handle().map(Handle::from_raw) else {
            return Ok(false);
        };
        if !self.is_callable(ch) {
            return Ok(false);
        }
        self.instance_of(o, c)
    }

    pub(crate) fn instance_of(&mut self, obj: NanBox, ctor: NanBox) -> Result<bool, ExecError> {
        // A custom `[Symbol.hasInstance]` on the right-hand side overrides the
        // ordinary prototype/cell-kind check (and applies even to a primitive
        // left-hand side, e.g. `4 instanceof Even`). Read via `read_member` so a
        // `static [Symbol.hasInstance]` on a class is found.
        if let Some(ch) = ctor.as_handle().map(Handle::from_raw) {
            let sym = self.well_known_symbol("hasInstance");
            let key = self.member_key(sym);
            let method = self.read_member(ch, &key)?;
            let callable_h = method
                .as_handle()
                .map(Handle::from_raw)
                .filter(|mh| self.is_callable(*mh));
            if let Some(mh) = callable_h {
                // Skip the *default* `Function.prototype[Symbol.hasInstance]`
                // (every function inherits it): it just performs OrdinaryHasInstance,
                // which is exactly the ordinary path below — calling it here would
                // recurse. Only a *user* `[Symbol.hasInstance]` override is honored.
                if self.realm.native_at(mh) != Some(N_FN_HAS_INSTANCE) {
                    let result = self.call_with_this(method, ctor, &[obj])?;
                    return Ok(self.realm.truthy(result));
                }
            } else if !matches!(method.unpack(), Unpacked::Undefined | Unpacked::Null) {
                // `GetMethod(target, @@hasInstance)`: a *present* but non-callable
                // value is a TypeError — it is never silently ignored (a proxy `get`
                // trap returning a RegExp, say).
                let m = self.new_str("Symbol.hasInstance is not a function");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        // The RHS must be a callable object (without a `[Symbol.hasInstance]`); a
        // primitive or a non-constructor object is a TypeError.
        let Some(ch) = ctor.as_handle().map(Handle::from_raw) else {
            let m = self.new_str("Right-hand side of 'instanceof' is not an object");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        // A bound function tests `instanceof` against its target function.
        if let Some(target) = self.realm.get_property(ch, BOUND_TARGET) {
            return self.instance_of(obj, target);
        }
        let is_ctor = self.realm.native_at(ch).is_some()
            || self.realm.host_fn_at(ch).is_some()
            || self.realm.bound_native_at(ch).is_some()
            // Any callable is a valid `instanceof` RHS per OrdinaryHasInstance's
            // IsCallable test — notably `%Function.prototype%` itself, which is a
            // callable object but not a native/user function (so `[] instanceof
            // Function.prototype` reads its `.prototype` and walks, rather than
            // wrongly throwing "not callable").
            || self.is_callable(ch)
            || self.current.get("Array").and_then(|v| v.as_handle()) == ctor.as_handle()
            || self.current.get("Object").and_then(|v| v.as_handle()) == ctor.as_handle();
        if !is_ctor {
            let m = self.new_str("Right-hand side of 'instanceof' is not callable");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // A primitive left-hand side is not an instance of anything. As well as the
        // NanBox primitives (number/boolean/null/undefined), the heap-cell
        // primitives — String, Symbol, BigInt — are values, not objects, so
        // OrdinaryHasInstance returns false for them (e.g. `Symbol() instanceof
        // Symbol` is false). A primitive *wrapper* object is a plain `Cell::Object`
        // and is unaffected.
        let Some(oh) = obj.as_handle().map(Handle::from_raw) else {
            return Ok(false);
        };
        if self.realm.symbol_at(oh).is_some()
            || self.realm.is_string_handle(oh)
            || self.realm.bigint_at(oh).is_some()
        {
            return Ok(false);
        }
        // Built-in constructors. Only the few kinds whose instances carry *no*
        // usable `[[Prototype]]` link are still matched by their cell kind or marker
        // slot (`WebAssembly.*`, `Temporal.*`, `Function`); everything else falls
        // through to the OrdinaryHasInstance walk at the end of this block, because a
        // brand says what an object *is*, never which realm's constructor made it.
        if let Some(id) = self.realm.native_at(ch) {
            // A primitive wrapper (`new Number(…)`) is matched by OrdinaryHasInstance
            // like everything else: its `[[NumberData]]` brand says *what* it wraps,
            // not which realm's `Number` produced it, so
            // `otherRealm.eval("new Number(1)") instanceof Number` was wrongly true.
            // A typed array is matched by OrdinaryHasInstance (the generic
            // `[[Prototype]]`-chain walk at the end of this block), *not* by its
            // element-kind brand: a brand match ignores which realm the constructor
            // belongs to, so `otherRealm.Int8Array.of() instanceof Int8Array` was
            // wrongly true — and it also ignores an explicit
            // `Object.setPrototypeOf(ta, null)`.
            // The `WebAssembly.*` boundary objects match by their marker slot.
            let wasm_marker = match id {
                N_WASM_GLOBAL => Some(WASM_GLOBAL_VALUE),
                N_WASM_MEMORY => Some(WASM_MEM_BUFFER),
                N_WASM_TABLE => Some(WASM_TABLE_ELEMS),
                N_WASM_MODULE => Some(WASM_IS_MODULE),
                N_WASM_INSTANCE => Some(WASM_INSTANCE_ID),
                _ => None,
            };
            if let Some(slot) = wasm_marker
                && self.realm.get_property(oh, slot).is_some()
            {
                return Ok(true);
            }
            // `ArrayBuffer` / `DataView` are matched by OrdinaryHasInstance too: the
            // `ARRAY_BUFFER_BYTES` / `DATA_VIEW_BUF` marker slot records the internal
            // slot, not the owning realm, so a buffer or view built in a
            // `$262.createRealm()` realm was an instance of *every* realm's
            // constructor.
            // The `Error` family: an error instance now links to its constructor's
            // `.prototype`, so OrdinaryHasInstance (the prototype-chain walk) is the
            // authoritative check — robust against `name` being reassigned.
            if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
                if let Some(proto) = self
                    .realm
                    .get_property(ch, "prototype")
                    .and_then(|p| p.as_handle())
                    .map(Handle::from_raw)
                {
                    let mut cur = oh;
                    for _ in 0..100_000 {
                        let next = self.get_proto_of(cur)?;
                        let Some(p) = next.as_handle().map(Handle::from_raw) else {
                            break;
                        };
                        if p == proto {
                            return Ok(true);
                        }
                        cur = p;
                    }
                }
                // No `name`-property fallback: `name` is an ordinary inherited
                // property, identical in every realm, so matching on it made an
                // error from a `$262.createRealm()` realm an instance of *this*
                // realm's `Error` (and made any `{ name: "TypeError" }` one too).
                return Ok(false);
            }
            // `RegExp`, the four collections, `Date` and `Promise` are matched by
            // OrdinaryHasInstance (the `[[Prototype]]`-chain walk at the end of this
            // block), *not* by their cell kind. A brand match ignores which realm the
            // constructor belongs to (`otherRealm.eval("new Map()") instanceof Map`
            // was wrongly true), cannot tell the four collections apart from one
            // another (`new Map() instanceof Set` was true — they share
            // `Cell::Collection`), and ignores both an explicit
            // `Object.setPrototypeOf(x, null)` and an `Object.create(Map.prototype)`.
            match id {
                id if crate::nbexec::temporal::is_temporal_ctor_id(id) => {
                    // `x instanceof Temporal.<Type>` — a branded instance of that
                    // exact kind.
                    return Ok(self.realm.temporal_at(oh).map(|d| d.kind)
                        == crate::nbexec::temporal::kind_for_ctor_id(id));
                }
                // Every callable (function, class, native, bound) is a `Function`.
                N_FUNCTION => return Ok(self.is_callable(oh)),
                _ => {}
            }
            // OrdinaryHasInstance fallback for any other built-in constructor (e.g.
            // `%Iterator%`, whose instances are recognized only by their prototype
            // chain): walk `obj`'s `[[Prototype]]` chain for the ctor's `.prototype`.
            if let Some(proto) = self
                .realm
                .get_property(ch, "prototype")
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw)
            {
                let mut cur = oh;
                for _ in 0..100_000 {
                    let next = self.get_proto_of(cur)?;
                    let Some(p) = next.as_handle().map(Handle::from_raw) else {
                        return Ok(false);
                    };
                    if p == proto {
                        return Ok(true);
                    }
                    cur = p;
                }
            }
            return Ok(false);
        }
        // `Array`/`Object` are namespace objects (not natives), matched by the
        // identity of the global binding.
        if self.current.get("Array").and_then(|v| v.as_handle()) == ctor.as_handle() {
            return Ok(self.realm.is_array(oh));
        }
        if self.current.get("Object").and_then(|v| v.as_handle()) == ctor.as_handle() {
            // Heap primitives (string/symbol/bigint values) are not objects.
            if self.realm.is_string_handle(oh)
                || self.realm.symbol_at(oh).is_some()
                || self.realm.bigint_at(oh).is_some()
            {
                return Ok(false);
            }
            // OrdinaryHasInstance: an object is `instanceof Object` iff its
            // `[[Prototype]]` chain reaches `Object.prototype`. A null-prototype
            // object (module namespace, `Object.create(null)`) is therefore *not*
            // an instance of `Object`.
            return Ok(self.realm.inherits_object_proto(oh));
        }
        // Any other callable RHS (a VM function or class, a registered host
        // constructor, `%Function.prototype%`): OrdinaryHasInstance. `Get(C,
        // "prototype")` (a proxy `get` trap fires) must be an Object — otherwise a
        // TypeError (`C.prototype = undefined`) — then walk `obj`'s `[[Prototype]]`
        // chain via `get_proto_of`, so a proxy's `getPrototypeOf` trap is honored
        // at each step (bounded to guard against a trap returning a cycle).
        let proto_val = self.read_member(ch, "prototype")?;
        let Some(proto) = proto_val
            .as_handle()
            .map(Handle::from_raw)
            .filter(|_| self.is_object_value(proto_val))
        else {
            return Err(self.type_error("Function has non-object prototype in instanceof check"));
        };
        let mut cur = oh;
        for _ in 0..100_000 {
            let next = self.get_proto_of(cur)?;
            let Some(p) = next.as_handle().map(Handle::from_raw) else {
                return Ok(false);
            };
            if p == proto {
                return Ok(true);
            }
            cur = p;
        }
        Ok(false)
    }
}

/// `StringToBigInt(str)` (ES2020 7.1.14) as a fallible parse: a trimmed, empty
/// (or all-whitespace) string is `0n`; a `0x`/`0o`/`0b` prefix selects the radix;
/// otherwise a decimal (optionally signed) integer literal. Returns `None` — the
/// spec's `undefined` — for any string that is not a valid `StringIntegerLiteral`
/// (e.g. `"0."`, `"1.5"`, `"x"`), which the comparison operators treat as an
/// unequal / undefined-ordering result rather than a throw.
fn string_to_bigint_opt(s: &str) -> Option<crate::bignum::BigInt> {
    let t = s.trim_matches(crate::realm::is_js_whitespace);
    if t.is_empty() {
        return Some(crate::bignum::BigInt::zero());
    }
    let (radix, body) = match t.get(0..2) {
        Some("0x" | "0X") => (16, &t[2..]),
        Some("0o" | "0O") => (8, &t[2..]),
        Some("0b" | "0B") => (2, &t[2..]),
        _ => (10, t),
    };
    crate::bignum::BigInt::from_str_radix(body, radix)
}

/// Exact comparison of a `BigInt` against an IEEE-754 double, with **no** loss of
/// precision (the mathematical values are compared, so `2n**60n` vs a nearby
/// `f64`, or `Number.MAX_VALUE` vs a 1024-bit BigInt, order correctly). Returns
/// `None` iff `f` is `NaN` (an undefined comparison).
fn bigint_cmp_f64(big: &crate::bignum::BigInt, f: f64) -> Option<core::cmp::Ordering> {
    use core::cmp::Ordering;
    if f.is_nan() {
        return None;
    }
    if f == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if f == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }
    // Decompose `f` into integer `mantissa * 2^exp` (exact for every finite f64).
    let bits = f.to_bits();
    let sign_neg = bits >> 63 == 1;
    let raw_exp = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    let (mantissa, exp) = if raw_exp == 0 {
        (frac, -1074i64) // subnormal (or zero)
    } else {
        (frac | 0x0010_0000_0000_0000, raw_exp - 1075)
    };
    if mantissa == 0 {
        return Some(big.cmp(&crate::bignum::BigInt::zero())); // f is ±0
    }
    let m = crate::bignum::BigInt::from_i128(i128::from(mantissa));
    let m = if sign_neg { m.neg() } else { m };
    let two = crate::bignum::BigInt::from_i128(2);
    // Compare `big` against `m * 2^exp` by clearing the power of two exactly.
    Some(if exp >= 0 {
        big.cmp(&m.mul(&two.pow(exp as u64)))
    } else {
        big.mul(&two.pow((-exp) as u64)).cmp(&m)
    })
}
