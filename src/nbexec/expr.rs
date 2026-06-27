use super::*;

impl<'a> Interp<'a> {
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
                Some(v) => Ok(v),
                None => {
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
        if let Some(h) = self.with_binding(name) {
            return self.read_member(h, name);
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
            // Not in the lexical scope chain: a property added directly to the
            // global object (`this.x = …` / `globalThis.x = …` at script level) is
            // a global binding, so fall back to a global-object own property.
            None => {
                if let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
                    && self.realm.has_own(g, name)
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

    /// Evaluates `e` and returns its JS truthiness (heap-aware, so an empty
    /// string is falsy).
    pub(crate) fn eval_truthy(&mut self, e: &'a Expr) -> Result<bool, ExecError> {
        let v = self.eval(e)?;
        Ok(self.realm.truthy(v))
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

    /// Resolves an object/class property key to its string name, evaluating a
    /// `[computed]` key expression where present (a symbol maps to its identity
    /// key, any other value to its string form).
    pub(crate) fn eval_prop_key(&mut self, key: &'a PropertyKey) -> Result<String, ExecError> {
        match key {
            PropertyKey::Computed(e) => {
                let v = self.eval(e)?;
                // ToPropertyKey: a symbol keeps its identity; any other object is
                // coerced via ToPrimitive(string) so a user `toString` runs and an
                // uncoercible key (e.g. `Object.create(null)` or a non-callable
                // `Symbol.toPrimitive`) throws a TypeError.
                self.coerce_property_key(v)
            }
            // A private name (`#x`) resolves to the storage key of the `#x`
            // declared in the lexically-enclosing class of this access site.
            PropertyKey::Private(s) => Ok(self.private_access_key(s)),
            _ => static_key(key),
        }
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

    /// `ToPropertyKey(k)`: like `member_key`, but a non-string, non-symbol object
    /// key is coerced with ToPrimitive(String) so a user `toString` is honored
    /// (`obj[{toString(){return "x"}}]` keys on `"x"`).
    pub(crate) fn coerce_property_key(&mut self, k: NanBox) -> Result<String, ExecError> {
        let is_object_key = k.as_handle().is_some_and(|raw| {
            let h = Handle::from_raw(raw);
            self.realm.symbol_at(h).is_none() && self.realm.string_value(h).is_none()
        });
        if is_object_key {
            let p = self.coerce_object(k, "string")?;
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

    /// Whether `v` is an object (a non-primitive heap value: object/array/function/…)
    /// rather than a string/symbol/bigint primitive or an immediate.
    pub(crate) fn is_object_value(&self, v: NanBox) -> bool {
        v.as_handle().map(Handle::from_raw).is_some_and(|h| {
            self.realm.string_value(h).is_none()
                && self.realm.symbol_at(h).is_none()
                && self.realm.bigint_at(h).is_none()
        })
    }

    pub(crate) fn eval(&mut self, expr: &'a Expr) -> Result<NanBox, ExecError> {
        // C2: guard the native recursion that `eval` performs on nested
        // expressions (a deep `a + a + … + a` is shallow in the AST but recurses
        // here once per term). Throw a catchable `RangeError` past the limit
        // instead of overflowing the host stack. Bounded by the dedicated
        // `max_eval_depth` knob (separate from `max_call_depth`).
        if self.eval_depth >= self.realm.limits.max_eval_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            let err = self.make_error(N_ERROR_BASE + 2, Some(msg));
            return Err(ExecError::Throw(err));
        }
        self.eval_depth += 1;
        let r = self.eval_inner(expr);
        self.eval_depth -= 1;
        r
    }

    pub(crate) fn eval_inner(&mut self, expr: &'a Expr) -> Result<NanBox, ExecError> {
        match expr {
            Expr::Null(_) => Ok(NanBox::null()),
            Expr::Bool { value, .. } => Ok(NanBox::boolean(*value)),
            Expr::Number { value, .. } => Ok(NanBox::number(*value)),
            Expr::BigInt { digits, .. } => {
                let n = parse_bigint(digits);
                Ok(NanBox::handle(self.realm.new_bigint(n).to_raw()))
            }
            Expr::Str { value, .. } => {
                // The cooked value is WTF-8 bytes; preserve any lone surrogates.
                let h = self.realm.new_string_wtf8(value.to_vec());
                Ok(NanBox::handle(h.to_raw()))
            }
            Expr::Ident(id) => self.read_ident_ref(&id.name),
            Expr::Regex { pattern, flags, .. } => Ok(NanBox::handle(
                self.new_regexp_instance(pattern, flags).to_raw(),
            )),
            // A template literal: interleave cooked quasis with interpolations.
            // Built as WTF-8 bytes so a surrogate-bearing quasi (`` `\uD800` ``)
            // round-trips.
            Expr::Template(t) => {
                let mut out: Vec<u8> = Vec::new();
                for (i, quasi) in t.quasis.iter().enumerate() {
                    match &quasi.cooked {
                        Some(cooked) => out.extend_from_slice(cooked),
                        // An invalid escape is allowed only in a *tagged* template; in a
                        // plain template literal it is a SyntaxError.
                        None => {
                            let m = self.new_str("Invalid escape sequence in template literal");
                            return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
                        }
                    }
                    if let Some(e) = t.expressions.get(i) {
                        let v = self.eval(e)?;
                        out.extend_from_slice(&self.coerce_to_string_bytes(v)?);
                    }
                }
                Ok(self.new_str_bytes(out))
            }
            // The comma operator: evaluate all, yield the last.
            Expr::Sequence { expressions, .. } => {
                let mut last = NanBox::undefined();
                for e in expressions {
                    last = self.eval(e)?;
                }
                Ok(last)
            }
            // A tagged template: `tag(stringsArray, ...interpolatedValues)`.
            Expr::TaggedTemplate { tag, quasi, .. } => {
                // The frozen strings object is created once per template-literal site
                // and reused on every evaluation (its identity is observable to the tag).
                let cache_key = core::ptr::from_ref(quasi) as usize;
                let strings_arr = if let Some(cached) = self.tagged_template_cache.get(&cache_key) {
                    *cached
                } else {
                    // A quasi with an invalid escape sequence has no cooked value
                    // (`undefined`), while its `.raw` is still preserved (ES2018).
                    let strings: Vec<NanBox> = quasi
                        .quasis
                        .iter()
                        .map(|q| match q.cooked.as_deref() {
                            Some(s) => self.new_str_bytes(s.to_vec()),
                            None => NanBox::undefined(),
                        })
                        .collect();
                    let raw: Vec<NanBox> =
                        quasi.quasis.iter().map(|q| self.new_str(&q.raw)).collect();
                    let strings_h = self.realm.new_array(strings);
                    // The strings object carries a `.raw` array (for `String.raw` and
                    // tags reading `strings.raw`). Both arrays are frozen, per spec —
                    // freeze `.raw` first and `strings` last so the property write lands.
                    let raw_h = self.realm.new_array(raw);
                    self.realm.freeze_object(raw_h);
                    self.realm
                        .set_property(strings_h, "raw", NanBox::handle(raw_h.to_raw()));
                    self.realm.freeze_object(strings_h);
                    let arr = NanBox::handle(strings_h.to_raw());
                    self.tagged_template_cache.insert(cache_key, arr);
                    arr
                };
                let mut args = alloc::vec![strings_arr];
                for e in &quasi.expressions {
                    args.push(self.eval(e)?);
                }
                // A `recv.tag` tag (e.g. `String.raw`) is dispatched as a method
                // call, so a built-in tag works even if it isn't a readable value.
                if let Expr::Member {
                    object, property, ..
                } = &**tag
                    && let PropertyKey::Ident(name) | PropertyKey::Str(name) = property
                {
                    let recv = self.eval(object)?;
                    if let Some(result) = self.call_method(recv, name, &args)? {
                        return Ok(result);
                    }
                    // Fall back to a property-valued tag function. A primitive
                    // receiver has no callable tag here — a catchable TypeError.
                    let Some(raw) = recv.as_handle() else {
                        let m = self.new_str("is not a function");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    };
                    let f = self.member(Handle::from_raw(raw), property)?;
                    return self.call_with_this(f, recv, &args);
                }
                let tagf = self.eval(tag)?;
                self.call(tagf, &args)
            }
            Expr::This(_) => {
                // In a derived constructor, `this` is in its temporal dead zone
                // until `super(...)` runs (ReferenceError if accessed before).
                if self.this_val.is_tdz() {
                    let m = self.new_str(
                        "Must call super constructor before accessing 'this' or returning from derived constructor",
                    );
                    return Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(m)),
                    ));
                }
                Ok(self.this_val)
            }
            Expr::NewTarget(_) => Ok(self.new_target),
            Expr::Await { argument, .. } => {
                let v = self.eval(argument)?;
                self.await_value(v)
            }
            // Eager generators: `yield x` appends `x` to the active buffer;
            // `yield* it` appends each value of the iterable. The expression's
            // own value is `undefined` (we cannot thread `next()` arguments back).
            Expr::Yield {
                argument, delegate, ..
            } => {
                let v = match argument {
                    Some(e) => self.eval(e)?,
                    None => NanBox::undefined(),
                };
                if *delegate {
                    let vals = self.iterate_values(v)?;
                    if let Some(sink) = self.gen_sink.as_mut() {
                        if sink.len() + vals.len() > GEN_CAP {
                            return Err(ExecError::Throw(self.new_str("generator yield limit")));
                        }
                        sink.extend(vals);
                    }
                    // `yield* iterable` evaluates to the iterator's final value — a
                    // delegated generator's `return` value (else `undefined`).
                    let ret = v
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.get_property(h, GEN_RET))
                        .unwrap_or(NanBox::undefined());
                    return Ok(ret);
                } else if let Some(sink) = self.gen_sink.as_mut() {
                    if sink.len() >= GEN_CAP {
                        return Err(ExecError::Throw(self.new_str("generator yield limit")));
                    }
                    sink.push(v);
                }
                Ok(NanBox::undefined())
            }
            Expr::Function(func) => Ok(self.eval_fn_expr(func)),
            Expr::Arrow(arrow) => Ok(self.eval_arrow(arrow)),
            Expr::Class(class) => self.make_class(class),
            Expr::Unary { op, argument, .. } => {
                // `delete obj.x` removes a property; `typeof undefinedVar` must
                // not throw — both inspect the operand rather than its value.
                match op {
                    UnaryOp::Delete => {
                        // `delete` returns `false` when the property is
                        // non-configurable (sealed/frozen); `true` otherwise.
                        let mut result = true;
                        let mut is_property_delete = false;
                        // `delete a?.b` unwraps the optional-chain target; a nullish base
                        // short-circuits the whole `delete` to a no-op returning `true`.
                        let argument: &Expr = match &**argument {
                            Expr::OptChain { expr, .. } => expr,
                            other => other,
                        };
                        if let Expr::Member {
                            object, property, ..
                        } = argument
                        {
                            is_property_delete = true;
                            // `delete super.prop` / `delete super[expr]` is a runtime
                            // ReferenceError (a super reference is never deletable).
                            if matches!(&**object, Expr::Super(_)) {
                                let m = self.new_str("Cannot delete a super property");
                                return Err(ExecError::Throw(
                                    self.make_error(N_REFERENCE_ERROR, Some(m)),
                                ));
                            }
                            // A nullish link in the base (`delete a?.b.c` with nullish `a`)
                            // short-circuits the whole `delete` to a no-op returning `true`.
                            let obj = match self.eval(object) {
                                Ok(v) => v,
                                Err(ExecError::OptShortCircuit) => {
                                    return Ok(NanBox::boolean(true));
                                }
                                Err(e) => return Err(e),
                            };
                            if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                                return Ok(NanBox::boolean(true));
                            }
                            if let Some(raw) = obj.as_handle() {
                                let h = Handle::from_raw(raw);
                                let name = match property {
                                    PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                                        Some(String::from(&**s))
                                    }
                                    PropertyKey::Computed(e) => {
                                        let k = self.eval(e)?;
                                        Some(self.member_key(k))
                                    }
                                    _ => None,
                                };
                                if let Some(name) = name {
                                    // A Deferred Module Namespace (`import defer`)
                                    // evaluates its target on a `[[Delete]]` with a
                                    // String (non-"then") key.
                                    #[cfg(all(feature = "module", feature = "std"))]
                                    self.trigger_deferred_namespace(h, &name)?;
                                    // Proxy `deleteProperty` trap, or forward.
                                    if let Some((target, handler)) = self.realm.proxy_at(h) {
                                        self.guard_revoked(h)?;
                                        if let Some(trap) =
                                            self.proxy_trap(handler, "deleteProperty")?
                                        {
                                            let kb = self.new_str(&name);
                                            let handler_box = NanBox::handle(handler.to_raw());
                                            let r = self.call_with_this(
                                                trap,
                                                handler_box,
                                                &[NanBox::handle(target.to_raw()), kb],
                                            )?;
                                            result = self.realm.truthy(r);
                                            // Invariant (10.5.10): a true result is
                                            // illegal if the property exists as a
                                            // non-configurable own property of the
                                            // target, or the target is non-extensible
                                            // and the property is present.
                                            if result {
                                                let present = self.realm.has_own(target, &name)
                                                    || self.realm.accessor(target, &name).is_some();
                                                if present
                                                    && self
                                                        .realm
                                                        .property_is_non_configurable(target, &name)
                                                {
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
                                            self.realm.delete_property(target, &name);
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
                                    }
                                }
                            }
                        } else if let Expr::Ident(id) = argument {
                            if self.current.get(&id.name).is_some() {
                                // Deleting a resolvable lexical/var binding is a no-op
                                // that returns `false` (bindings are non-deletable).
                                result = false;
                            } else if let Some(h) = self.with_binding(&id.name) {
                                // A bare name that resolves through a `with` object's
                                // environment deletes that object's property — not the
                                // similarly-named global (`with (o) { delete p }`
                                // removes `o.p`, leaving any global `p` intact).
                                result = self.realm.delete_property(h, &id.name);
                                is_property_delete = true;
                            } else if let Some(g) = self.global_object()
                                && (self.realm.has_own(g, &id.name)
                                    || self.realm.accessor(g, &id.name).is_some())
                            {
                                // `delete name` where `name` resolves to a property of the
                                // global object: succeeds only if that property is
                                // configurable (e.g. `delete NaN`/`Infinity`/`undefined`
                                // — non-configurable — returns `false`).
                                result = self.realm.delete_property(g, &id.name);
                                is_property_delete = true;
                            }
                            // An unresolvable name (`delete notDefined`) returns `true`.
                        } else {
                            // `delete <non-Reference>` (e.g. `delete foo()`): the operand
                            // is still evaluated for its side effects, then `true` is
                            // returned (there is no binding/property to remove).
                            self.eval(argument)?;
                        }
                        // A failed delete of a non-configurable property throws in strict
                        // mode (rather than silently returning `false`).
                        if self.strict && is_property_delete && !result {
                            let m =
                                self.new_str("Cannot delete property of a non-configurable object");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        return Ok(NanBox::boolean(result));
                    }
                    UnaryOp::Typeof => {
                        // `typeof importedBinding` is *not* the unresolved-reference
                        // shortcut: an imported binding exists (resolving live, and
                        // possibly in TDZ), so `typeof` must read it (and may throw
                        // for a `let`/`const`/`class` export not yet initialised).
                        #[cfg(all(feature = "module", feature = "std"))]
                        let is_import = if let Expr::Ident(id) = &**argument {
                            self.module_imports.contains_key(&*id.name)
                        } else {
                            false
                        };
                        #[cfg(not(all(feature = "module", feature = "std")))]
                        let is_import = false;
                        if let Expr::Ident(id) = &**argument
                            && !is_import
                            && self.current.get(&id.name).is_none()
                            && self.with_binding(&id.name).is_none()
                            && !matches!(&*id.name, "undefined" | "NaN" | "Infinity")
                            // A binding may live only as a global-object own property
                            // (e.g. `globalThis.x = …`, or a built-in declared onto the
                            // global object rather than the lexical scope) — `typeof`
                            // must see it, not report "undefined".
                            && !self
                                .global_this
                                .as_handle()
                                .map(Handle::from_raw)
                                .is_some_and(|g| self.realm.has_own(g, &id.name))
                        {
                            return Ok(self.new_str("undefined"));
                        }
                    }
                    _ => {}
                }
                let v = self.eval(argument)?;
                self.unary(*op, v)
            }
            // `x++` / `++x` / `x--` / `--x` on an identifier or member.
            Expr::Update {
                op,
                prefix,
                argument,
                ..
            } => {
                let current = self.read_target(argument)?;
                // A BigInt operand increments/decrements by one BigInt.
                if let Some(big) = current
                    .as_handle()
                    .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)))
                {
                    let one = crate::bignum::BigInt::from_i128(1);
                    let next = match op {
                        crate::ast::UpdateOp::Inc => big.add(&one),
                        crate::ast::UpdateOp::Dec => big.sub(&one),
                    };
                    let next_box = NanBox::handle(self.realm.new_bigint(next).to_raw());
                    self.assign_to(argument, next_box)?;
                    let old_box = NanBox::handle(self.realm.new_bigint(big).to_raw());
                    return Ok(if *prefix { next_box } else { old_box });
                }
                // `ToNumber(GetValue(arg))` runs ToPrimitive(number) on an object
                // operand (its `valueOf`/`toString`, which may throw) — e.g.
                // `(new Boolean(true))++` is `2`. The infallible `to_number` routes
                // an object through `toString` only, so go via the spec path.
                let coerced = self.coerce_to_number(current)?;
                let old = self.realm.to_number(coerced);
                let next = match op {
                    crate::ast::UpdateOp::Inc => old + 1.0,
                    crate::ast::UpdateOp::Dec => old - 1.0,
                };
                self.assign_to(argument, NanBox::number(next))?;
                Ok(NanBox::number(if *prefix { next } else { old }))
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                // `#x in obj` — the ergonomic brand check (private fields are
                // stored under a `#`-prefixed key).
                if matches!(op, BinaryOp::In)
                    && let Expr::PrivateName(name, _) = &**left
                {
                    let obj = self.eval(right)?;
                    let key = self.private_access_key(name);
                    let present = obj.as_handle().map(Handle::from_raw).is_some_and(|h| {
                        self.realm.has_own(h, &key) || self.realm.accessor(h, &key).is_some()
                    });
                    return Ok(NanBox::boolean(present));
                }
                let a = self.eval(left)?;
                let b = self.eval(right)?;
                self.binary(*op, a, b)
            }
            Expr::Logical {
                op, left, right, ..
            } => {
                let l = self.eval(left)?;
                let take_right = match op {
                    LogicalOp::And => self.realm.truthy(l),
                    LogicalOp::Or => !self.realm.truthy(l),
                    LogicalOp::Nullish => {
                        matches!(l.unpack(), Unpacked::Undefined | Unpacked::Null)
                    }
                };
                if take_right { self.eval(right) } else { Ok(l) }
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval_truthy(test)? {
                    self.eval(consequent)
                } else {
                    self.eval(alternate)
                }
            }
            Expr::Assign {
                op, target, value, ..
            } => self.eval_assign(*op, target, value),
            Expr::Call {
                callee,
                arguments,
                optional: call_optional,
                ..
            } => {
                // Dynamic `import(specifier)`. The parser desugars it to a call of
                // the bare `import` reference; intercept it here (before that
                // reference would throw) and return a promise of the requested
                // module's namespace object. Works in scripts and modules alike.
                #[cfg(all(feature = "module", feature = "std"))]
                if let Expr::Ident(id) = &**callee
                    && id.name.as_ref() == "import"
                {
                    return self.dynamic_import(arguments);
                }
                // `import.source(x)` / `import.defer(x)` — the source-phase and
                // deferred-import proposals, unimplemented. Per the tests, ToString
                // the specifier (a throw rejects with that), then return a promise
                // rejected with a SyntaxError — NOT a plain dynamic import that would
                // load the module.
                #[cfg(all(feature = "module", feature = "std"))]
                if let Expr::Member {
                    object, property, ..
                } = &**callee
                    && matches!(&**object, Expr::Ident(id) if id.name.as_ref() == "import")
                    && matches!(property, PropertyKey::Ident(p) if matches!(&**p, "source" | "defer"))
                {
                    let p = self.fresh_promise();
                    let arg0 = arguments.first().map(|a| match a {
                        crate::ast::Argument::Item(e) | crate::ast::Argument::Spread(e) => e,
                    });
                    let rejection = match arg0 {
                        Some(e) => match self.eval(e).and_then(|v| self.coerce_to_string(v)) {
                            Ok(_) => {
                                let m =
                                    self.new_str("source-phase / deferred import is not supported");
                                self.make_error(N_SYNTAX_ERROR, Some(m))
                            }
                            Err(ExecError::Throw(t)) => t,
                            Err(other) => return Err(other),
                        },
                        None => {
                            let m = self.new_str("source-phase / deferred import is not supported");
                            self.make_error(N_SYNTAX_ERROR, Some(m))
                        }
                    };
                    self.settle(p, rejection, false);
                    return Ok(NanBox::handle(p.to_raw()));
                }
                // `super(args)` — invoke the base constructor on the current
                // instance.
                if matches!(&**callee, Expr::Super(_)) {
                    let args = self.eval_args(arguments)?;
                    // The instance + this class's id are stashed in
                    // `pending_this_init` while `this` is in its TDZ (set by the
                    // derived constructor). Calling `super()` a second time leaves
                    // it `None` → ReferenceError.
                    let Some((inst_val, derived_cid)) = self.pending_this_init else {
                        let m = self.new_str("Super constructor may only be called once");
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    };
                    let inst = inst_val.as_handle().map(Handle::from_raw);
                    // Bind `this` and clear the pending marker BEFORE invoking the
                    // parent constructor — the parent's body (a base class, or a
                    // further-derived one after its own `super`) reads `this`, and
                    // a second `super()` must now see `None` and error.
                    self.this_val = inst_val;
                    self.pending_this_init = None;
                    if let Some((pid, penv)) = self.pending_super.clone() {
                        if let Some(h) = inst {
                            self.run_constructor(pid, &penv, h, &args)?;
                        }
                    } else if let Some(nid) = self.pending_super_native {
                        // `super(...)` reaching a native constructor (`extends Error`).
                        if let Some(h) = inst {
                            self.apply_native_super(nid, h, &args);
                        }
                    } else if let Some(fnp) = self.pending_super_fn {
                        // `super(...)` reaching an ordinary-function superclass.
                        self.call_with_this(fnp, inst_val, &args)?;
                    } else {
                        return Err(ExecError::Unsupported(
                            "super outside a derived constructor",
                        ));
                    }
                    // This class's field initializers run *after* `super()` returns.
                    if let Some(h) = inst {
                        self.init_instance_fields(derived_cid, h)?;
                    }
                    return Ok(NanBox::undefined());
                }
                // `super.method(args)` — invoke the base-class method with the
                // current `this`.
                if let Expr::Member {
                    object, property, ..
                } = &**callee
                    && matches!(&**object, Expr::Super(_))
                {
                    // `super.m(args)` and `super[expr](args)` — resolve the method
                    // name (a computed key is evaluated to a property key) and invoke
                    // it with the current `this`.
                    let name = match property {
                        PropertyKey::Ident(name) | PropertyKey::Str(name) => {
                            alloc::string::String::from(&**name)
                        }
                        PropertyKey::Number(n) => self.realm.to_display_string(NanBox::number(*n)),
                        PropertyKey::Computed(e) => {
                            let k = self.eval(e)?;
                            self.coerce_property_key(k)?
                        }
                        PropertyKey::Private(_) => {
                            return Err(ExecError::Unsupported("private super member"));
                        }
                    };
                    let args = self.eval_args(arguments)?;
                    let f = self.resolve_super_method(&name)?;
                    return self.call_with_this(f, self.this_val, &args);
                }
                // A `recv.method(args)` call: try a built-in method on the
                // receiver before falling back to a property-valued function.
                if let Expr::Member {
                    object,
                    property,
                    optional,
                    ..
                } = &**callee
                {
                    let recv = self.eval(object)?;
                    if matches!(recv.unpack(), Unpacked::Undefined | Unpacked::Null) {
                        if *optional {
                            return Err(ExecError::OptShortCircuit);
                        }
                        // Resolving the callee member (`obj.m`) on a nullish base is a
                        // TypeError, thrown *before* the arguments are evaluated (spec
                        // reference order): `o.bar.gar(foo())` throws before `foo()`.
                        let key = match property {
                            PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                                alloc::string::String::from(&**s)
                            }
                            PropertyKey::Number(n) => {
                                self.realm.to_display_string(NanBox::number(*n))
                            }
                            PropertyKey::Computed(e) => {
                                let k = self.eval(e)?;
                                self.coerce_property_key(k)?
                            }
                            PropertyKey::Private(s) => alloc::format!("#{s}"),
                        };
                        return Err(self.type_error(&alloc::format!(
                            "Cannot read properties of {} (reading '{key}')",
                            self.realm.to_display_string(recv)
                        )));
                    }
                    let args = self.eval_args(arguments)?;
                    // The built-in name-based dispatch (`call_method`) is an
                    // optimization for *unshadowed* built-in methods. If the
                    // receiver carries an *own* property of this name (e.g.
                    // `s.valueOf = Number.prototype.valueOf`), that property is the
                    // method to invoke — resolving and calling the function value
                    // preserves its own `this`-validation (so a cross-type
                    // `Number.prototype.valueOf` call on a String wrapper throws),
                    // rather than the receiver's built-in behavior.
                    if let PropertyKey::Ident(name) | PropertyKey::Str(name) = property
                        && recv
                            .as_handle()
                            .map(Handle::from_raw)
                            .is_some_and(|h| self.realm.has_own(h, name))
                    {
                        let rh = recv.as_handle().map(Handle::from_raw).unwrap();
                        let f = self.read_member(rh, name)?;
                        if f.as_handle()
                            .map(Handle::from_raw)
                            .is_some_and(|fh| self.is_callable(fh))
                        {
                            return self.call_with_this(f, recv, &args);
                        }
                    }
                    if let PropertyKey::Ident(name) | PropertyKey::Str(name) = property
                        && let Some(result) = self.call_method(recv, name, &args)?
                    {
                        return Ok(result);
                    }
                    // `obj[Symbol.iterator]()` → an iterator over the receiver.
                    if let PropertyKey::Computed(e) = property {
                        let key = self.eval(e)?;
                        let iter_sym = self.well_known_symbol("iterator");
                        if self.realm.strict_equals(key, iter_sym) {
                            // A generator/iterator is its own iterator (identity) —
                            // both the eager built-in iterables (`GEN_BUF`) and a
                            // lazy generator (`GEN_FRAME`).
                            if recv.as_handle().map(Handle::from_raw).is_some_and(|h| {
                                self.realm.get_property(h, GEN_BUF).is_some()
                                    || self.realm.get_property(h, GEN_FRAME).is_some()
                            }) {
                                return Ok(recv);
                            }
                            let vals = self.iterate_values(recv)?;
                            // Tag the iterator with the receiver's kind so its
                            // prototype is the real `%ArrayIteratorPrototype%` /
                            // `%StringIteratorPrototype%` / `%Map|SetIteratorPrototype%`.
                            let tag = recv.as_handle().map(Handle::from_raw).and_then(|h| {
                                if self.realm.array_elements(h).is_some()
                                    || self.realm.is_array(h)
                                    || self.realm.typed_kind(h).is_some()
                                {
                                    Some("Array Iterator")
                                } else if self.realm.string_value(h).is_some()
                                    || self
                                        .realm
                                        .get_property(h, PRIM_WRAP_TYPE)
                                        .and_then(|t| t.as_number())
                                        == Some(f64::from(N_STRING))
                                {
                                    // A primitive string cell, or a boxed `String`
                                    // wrapper (`new String("…")`) whose `[[StringData]]`
                                    // lives in the `PRIM_WRAP` slot.
                                    Some("String Iterator")
                                } else {
                                    match self.realm.collection_is_set(h) {
                                        Some(true) => Some("Set Iterator"),
                                        Some(false) => Some("Map Iterator"),
                                        None => None,
                                    }
                                }
                            });
                            return Ok(match tag {
                                Some(t) => self.make_builtin_iterator(vals, t),
                                None => self.make_generator(vals),
                            });
                        }
                    }
                    // Not a built-in method: read the member and call it.
                    let Some(raw) = recv.as_handle() else {
                        if *call_optional {
                            return Err(ExecError::OptShortCircuit);
                        }
                        // The receiver is a primitive. `null`/`undefined` cannot be
                        // coerced, so any member access is a catchable TypeError.
                        if matches!(recv.unpack(), Unpacked::Undefined | Unpacked::Null) {
                            let m = self.new_str("cannot read property of null or undefined");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        // For a number/boolean primitive, an inherited (or
                        // prototype-assigned) method is found by boxing the value and
                        // walking its prototype chain — then invoked with the original
                        // primitive as `this` (e.g.
                        // `Number.prototype.toLowerCase = String.prototype.toLowerCase`,
                        // or a computed key like `false["toString"]()`).
                        let name = match property {
                            PropertyKey::Ident(name) | PropertyKey::Str(name) => {
                                Some(alloc::string::String::from(&**name))
                            }
                            PropertyKey::Number(n) => {
                                Some(self.realm.to_display_string(NanBox::number(*n)))
                            }
                            PropertyKey::Computed(e) => {
                                let k = self.eval(e)?;
                                Some(self.coerce_property_key(k)?)
                            }
                            PropertyKey::Private(_) => None,
                        };
                        if let Some(name) = name {
                            let boxed = self.coerce_to_object(recv);
                            if let Some(bh) = boxed.as_handle().map(Handle::from_raw) {
                                let f = self.read_member(bh, &name)?;
                                if *call_optional
                                    && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null)
                                {
                                    return Err(ExecError::OptShortCircuit);
                                }
                                if f.as_handle()
                                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                                {
                                    return self.call_with_this(f, recv, &args);
                                }
                            }
                        }
                        let m = self.new_str("is not a function");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    };
                    let f = self.member(Handle::from_raw(raw), property)?;
                    // `f?.()` short-circuits when `f` is nullish.
                    if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null)
                    {
                        return Err(ExecError::OptShortCircuit);
                    }
                    // Method call: `this` is the receiver.
                    return self.call_with_this(f, recv, &args);
                }
                // A bare-identifier callee resolved through a `with (obj)` binding is
                // called with `obj` as the `this` value (the with-object is the
                // reference's base), so `with (o) { m(); }` calls `o.m` with `this`=`o`.
                if let Expr::Ident(id) = &**callee
                    && let Some(h) = self.with_binding(&id.name)
                {
                    let f = self.read_member(h, &id.name)?;
                    if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null)
                    {
                        return Err(ExecError::OptShortCircuit);
                    }
                    let args = self.eval_args(arguments)?;
                    return self.call_with_this(f, NanBox::handle(h.to_raw()), &args);
                }
                let f = self.eval(callee)?;
                if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(ExecError::OptShortCircuit);
                }
                let args = self.eval_args(arguments)?;
                // Direct eval: the callee is the literal identifier `eval` and it
                // still resolves to the built-in `eval`. Such a call runs in the
                // caller's scope (so it can read/modify locals and hoist `var`s),
                // inheriting the caller's strictness — unlike an indirect eval,
                // which `call`/`call_native` route through the global scope.
                if let Expr::Ident(id) = &**callee
                    && id.name.as_ref() == "eval"
                    && f.as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.native_at(h))
                        == Some(N_EVAL)
                {
                    let arg0 = args.first().copied().unwrap_or(NanBox::undefined());
                    let Some(source) = arg0
                        .as_handle()
                        .and_then(|raw| self.realm.string_value(Handle::from_raw(raw)))
                    else {
                        // A non-string argument is returned unchanged (per spec).
                        return Ok(arg0);
                    };
                    return self.eval_string(&source, true);
                }
                self.call(f, &args)
            }
            // The optional-chain boundary: a `?.` short-circuit inside becomes
            // `undefined` here (the rest of the chain was skipped).
            Expr::OptChain { expr, .. } => match self.eval(expr) {
                Err(ExecError::OptShortCircuit) => Ok(NanBox::undefined()),
                other => other,
            },
            Expr::New {
                callee, arguments, ..
            } => {
                let f = self.eval(callee)?;
                let args = self.eval_args(arguments)?;
                self.construct(f, &args)
            }
            Expr::Array { elements, .. } => {
                let mut items = Vec::new();
                for el in elements {
                    match el {
                        ArrayElement::Hole => items.push(NanBox::hole()),
                        ArrayElement::Item(e) => items.push(self.eval(e)?),
                        ArrayElement::Spread(e) => {
                            let v = self.eval(e)?;
                            items.extend(self.iterate_values(v)?);
                        }
                    }
                }
                let h = self.realm.new_array(items);
                Ok(NanBox::handle(h.to_raw()))
            }
            Expr::Object { members, .. } => {
                let handle = self.realm.new_object();
                for m in members {
                    match m {
                        ObjectMember::Property {
                            key,
                            value,
                            shorthand,
                            ..
                        } => {
                            // `{ __proto__: obj }` — only the *unquoted identifier*
                            // form (not `"__proto__":`, computed, shorthand, or a
                            // method) sets the prototype; a quoted/computed key makes
                            // an ordinary own `__proto__` data property.
                            if !shorthand
                                && !matches!(&**value, Expr::Function(_))
                                && let PropertyKey::Ident(s) = key
                                && &**s == "__proto__"
                            {
                                // Per spec, the `__proto__` property name in an object
                                // literal sets `[[Prototype]]` only when the value is an
                                // Object or `null`; any other primitive (string, number,
                                // boolean, undefined, symbol, bigint) is ignored — the
                                // object keeps `%Object.prototype%` and gains *no* own
                                // `__proto__` property.
                                let v = self.eval(value)?;
                                if matches!(v.unpack(), Unpacked::Null) {
                                    self.realm.set_object_proto(handle, None);
                                } else if self.is_object_value(v)
                                    && let Some(p) = v.as_handle().map(Handle::from_raw)
                                {
                                    self.realm.set_object_proto(handle, Some(p));
                                }
                                continue;
                            }
                            let k = self.eval_prop_key(key)?;
                            let v = self.eval(value)?;
                            // A method / function-valued property is named after its
                            // key when otherwise anonymous. A computed key that is a
                            // Symbol names the method `[description]` (or `""`); a
                            // static identifier/string key names it directly.
                            if matches!(
                                &**value,
                                Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_)
                            ) {
                                match key {
                                    PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                                        self.set_fn_name(v, s);
                                    }
                                    PropertyKey::Computed(_) => {
                                        // `k` is the storage key (a `\0sym:` key for a
                                        // Symbol); `method_display_name` renders the
                                        // spec name. Install it if the value is still
                                        // anonymous (an anonymous class included).
                                        let params: &[Param] = match &**value {
                                            Expr::Function(f) => &f.params,
                                            _ => &[],
                                        };
                                        if let Some(name) =
                                            self.method_display_name(&k, MethodKind::Method)
                                            && v.as_handle()
                                                .map(Handle::from_raw)
                                                .is_some_and(|h| !self.realm.has_own(h, "name"))
                                        {
                                            if matches!(&**value, Expr::Class(_)) {
                                                // A class already has its `length`; only
                                                // its `name` is set by NamedEvaluation.
                                                let nm = self.new_str(&name);
                                                if let Some(h) = v.as_handle().map(Handle::from_raw)
                                                {
                                                    self.realm.set_property(h, "name", nm);
                                                    self.realm.mark_hidden(h, "name");
                                                    self.realm.set_readonly_property(h, "name");
                                                }
                                            } else {
                                                self.install_method_meta(v, &name, params);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            // A concise method (`{ m() {} }`, not an arrow) records
                            // this object as its `[[HomeObject]]`, so `super.x`
                            // inside it resolves through the object's prototype.
                            if matches!(&**value, Expr::Function(_))
                                && let Some(fv) = v.as_handle().map(Handle::from_raw)
                            {
                                self.realm.set_hidden_property(
                                    fv,
                                    HOME_OBJECT,
                                    NanBox::handle(handle.to_raw()),
                                );
                            }
                            self.realm.set_property(handle, &k, v);
                        }
                        // `{ ...src }` — copy own enumerable properties.
                        ObjectMember::Spread { value, .. } => {
                            let src = self.eval(value)?;
                            if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                                // Spreading an array/string copies its indexed
                                // elements as `"0"`, `"1"`, … properties.
                                if let Some(elems) =
                                    self.realm.array_elements(sh).map(<[_]>::to_vec)
                                {
                                    for (i, e) in elems.iter().enumerate() {
                                        self.realm.set_property(handle, &alloc::format!("{i}"), *e);
                                    }
                                } else if let Some(s) = self.realm.string_value(sh) {
                                    for (i, c) in s.chars().enumerate() {
                                        let cv = self.new_str(&alloc::string::String::from(c));
                                        self.realm.set_property(handle, &alloc::format!("{i}"), cv);
                                    }
                                } else {
                                    // Own enumerable string + symbol keys (and
                                    // accessor getters); the raw key preserves
                                    // symbol identity.
                                    let keys = self.realm.object_keys_with_symbols(sh);
                                    for key in keys {
                                        // `read_member` invokes a getter where present.
                                        let pv = self.read_member(sh, &key)?;
                                        self.realm.set_property(handle, &key, pv);
                                    }
                                }
                            }
                        }
                        // `{ get x() {} }` / `{ set x(v) {} }`.
                        ObjectMember::Accessor {
                            key,
                            is_getter,
                            value,
                            ..
                        } => {
                            let k = self.eval_prop_key(key)?;
                            let f = self.make_function(
                                &value.params,
                                Body::Block(&value.body),
                                false,
                                false,
                            );
                            // An object-literal accessor's `[[HomeObject]]` is this
                            // object, so `super.x` inside it resolves via the proto.
                            if let Some(fh) = f.as_handle().map(Handle::from_raw) {
                                self.realm.set_hidden_property(
                                    fh,
                                    HOME_OBJECT,
                                    NanBox::handle(handle.to_raw()),
                                );
                                // The accessor's `name` is `"get <key>"` / `"set <key>"`
                                // (a symbol key → `"get [desc]"`), per SetFunctionName.
                                let kind = if *is_getter {
                                    MethodKind::Get
                                } else {
                                    MethodKind::Set
                                };
                                if let Some(nm) = self.method_display_name(&k, kind)
                                    && !self.realm.has_own(fh, "name")
                                {
                                    self.install_fn_name_length(fh, &nm, value.params.len() as u32);
                                }
                            }
                            if *is_getter {
                                self.realm
                                    .define_accessor(handle, &k, f, NanBox::undefined());
                            } else {
                                self.realm
                                    .define_accessor(handle, &k, NanBox::undefined(), f);
                            }
                        }
                    }
                }
                Ok(NanBox::handle(handle.to_raw()))
            }
            Expr::Member {
                object,
                property,
                optional,
                ..
            } => {
                // `import.meta` — the module meta-property. The parser desugars it
                // to `(import).meta`; resolve it to the current module's meta
                // object (set up by the module evaluator) here, before the bare
                // `import` reference would throw "import is not defined".
                #[cfg(all(feature = "module", feature = "std"))]
                if let Expr::Ident(id) = &**object
                    && id.name.as_ref() == "import"
                    && matches!(property, PropertyKey::Ident(p) if &**p == "meta")
                {
                    return Ok(self.import_meta.unwrap_or_else(NanBox::undefined));
                }
                // `super.name` reads a super getter/method (not via `this`).
                if matches!(&**object, Expr::Super(_)) {
                    // `super[expr]` — a computed super member. Outside any method
                    // (no `[[HomeObject]]`), `super` is a SyntaxError and the key
                    // expression must NOT be evaluated; throw before evaluating.
                    if let PropertyKey::Computed(key_expr) = property {
                        if self.current_home.is_none() && self.current_home_object.is_none() {
                            let m = self.new_str("'super' keyword unexpected here");
                            return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
                        }
                        let key = self.eval(key_expr)?;
                        let name = self.realm.to_display_string(key);
                        return self.resolve_super_member(&name);
                    }
                    let name = match property {
                        PropertyKey::Ident(name) | PropertyKey::Str(name) => {
                            alloc::string::String::from(&**name)
                        }
                        PropertyKey::Number(n) => self.realm.to_display_string(NanBox::number(*n)),
                        PropertyKey::Private(_) => {
                            return Err(ExecError::Unsupported("private super member"));
                        }
                        // Computed handled above.
                        PropertyKey::Computed(_) => unreachable!(),
                    };
                    return self.resolve_super_member(&name);
                }
                let obj = self.eval(object)?;
                if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    if *optional {
                        // Short-circuit the rest of the enclosing optional chain.
                        return Err(ExecError::OptShortCircuit);
                    }
                    // `null.x` / `undefined.x` throws a catchable TypeError.
                    let msg = self.new_str("cannot read property of null or undefined");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(msg))));
                }
                let Some(raw) = obj.as_handle() else {
                    // A number/boolean primitive reports its wrapper constructor
                    // (`(5).constructor === Number`); other reads are `undefined`
                    // here (method calls go through the call path).
                    if let PropertyKey::Ident(n) | PropertyKey::Str(n) = property
                        && n.as_ref() == "constructor"
                    {
                        let name = if obj.as_number().is_some() {
                            "Number"
                        } else if matches!(obj.unpack(), Unpacked::Bool(_)) {
                            "Boolean"
                        } else {
                            return Ok(NanBox::undefined());
                        };
                        return Ok(self.current.get(name).unwrap_or(NanBox::undefined()));
                    }
                    return Ok(NanBox::undefined());
                };
                let handle = crate::heap::Handle::from_raw(raw);
                self.member(handle, property)
            }
            _ => Err(ExecError::Unsupported("expression")),
        }
    }

    pub(crate) fn eval_fn_expr(&mut self, func: &'a Function) -> NanBox {
        // A named function expression binds its own name in an intermediate scope
        // that the closure captures, so the body can recurse by that name.
        if let Some(id) = &func.id {
            let inner = self.current.child();
            let saved = core::mem::replace(&mut self.current, inner);
            let f = self.make_function(
                &func.params,
                Body::Block(&func.body),
                func.is_async,
                func.is_generator,
            );
            self.set_fn_name(f, &id.name);
            self.current.declare(&id.name, f);
            self.current = saved;
            return f;
        }
        self.make_function(
            &func.params,
            Body::Block(&func.body),
            func.is_async,
            func.is_generator,
        )
    }

    pub(crate) fn eval_arrow(&mut self, arrow: &'a Arrow) -> NanBox {
        let body = match &arrow.body {
            ArrowBody::Expr(e) => Body::Expr(e),
            ArrowBody::Block(b) => Body::Block(b),
        };
        let f = self.make_function(&arrow.params, body, arrow.is_async, false);
        // Arrows have no own `arguments` binding (they inherit the enclosing one).
        if let Some(raw) = f.as_handle()
            && let Some((func_id, _)) = self.realm.function_at(Handle::from_raw(raw))
        {
            self.functions[func_id as usize].is_arrow = true;
            // Capture the *lexical* `this`/`new.target`/home at the definition site
            // (hidden slots), so a later call (including via `call`/`apply`/`bind`)
            // resolves them from here rather than the call site.
            let h = Handle::from_raw(raw);
            self.realm.set_hidden_property(h, ARROW_THIS, self.this_val);
            self.realm
                .set_hidden_property(h, ARROW_NEW_TARGET, self.new_target);
            if let Some(home) = self.current_home_object {
                self.realm
                    .set_hidden_property(h, ARROW_HOME_OBJ, NanBox::handle(home.to_raw()));
            }
            if let Some(hc) = self.current_home {
                self.realm
                    .set_hidden_property(h, ARROW_HOME_CLASS, NanBox::number(f64::from(hc)));
            }
            self.realm.set_hidden_property(
                h,
                ARROW_HOME_STATIC,
                NanBox::boolean(self.current_home_static),
            );
        }
        f
    }

    /// Records a function value's name (`fn.name`).
    pub(crate) fn set_fn_name(&mut self, value: NanBox, name: &'a str) {
        if let Some(raw) = value.as_handle()
            && let Some((func_id, _)) = self.realm.function_at(Handle::from_raw(raw))
            // Don't clobber a name the function already has (a named function
            // expression keeps its own name over the binding/key name).
            && self.functions[func_id as usize].name.is_empty()
        {
            self.functions[func_id as usize].name = name;
            // Materialize `name`/`length` as own, non-enumerable, non-writable,
            // configurable data properties so `f.hasOwnProperty("name")`,
            // `getOwnPropertyDescriptor`, and `verifyProperty` behave per spec.
            let handle = Handle::from_raw(raw);
            if !self.realm.has_own(handle, "name") {
                let len = self.functions[func_id as usize]
                    .params
                    .iter()
                    .take_while(|p| p.default.is_none() && !p.rest)
                    .count() as u32;
                self.install_fn_name_length(handle, name, len);
            }
            return;
        }
        // NamedEvaluation of an anonymous class: `let C = class {}` gives the
        // class constructor an own `name` of `"C"` (its `length` was already
        // installed at class creation). A class with a declared id keeps it.
        if let Some(raw) = value.as_handle() {
            let handle = Handle::from_raw(raw);
            if let Some((cid, _)) = self.realm.class_at(handle)
                && self.classes[cid as usize].id.is_none()
                && !self.realm.has_own(handle, "name")
            {
                let name_v = self.new_str(name);
                self.realm.set_property(handle, "name", name_v);
                self.realm.mark_hidden(handle, "name");
                self.realm.set_readonly_property(handle, "name");
            }
        }
    }

    /// `[[Get]]` of integer index `i` on an array-like receiver, returning
    /// `Some(value)` when the index is a *present* own element (a typed-array
    /// in-bounds element, or a plain-array in-range non-hole slot), or `None`
    /// when the read must fall through to the named `[[Get]]` (a hole or an
    /// out-of-range index, which consults the prototype chain).
    pub(crate) fn array_element_get(
        &mut self,
        handle: crate::heap::Handle,
        i: usize,
    ) -> Option<NanBox> {
        if self.realm.typed_kind(handle).is_some() {
            return Some(self.realm.get_element(handle, i));
        }
        if i < self.realm.array_length(handle).unwrap_or(0) {
            let v = self.realm.get_element(handle, i);
            if !v.is_hole() {
                return Some(v);
            }
        }
        None
    }

    pub(crate) fn member(
        &mut self,
        handle: crate::heap::Handle,
        key: &'a PropertyKey,
    ) -> Result<NanBox, ExecError> {
        match key {
            PropertyKey::Number(n)
                if as_index(*n).is_some() && self.realm.is_array_like(handle) =>
            {
                let i = as_index(*n).unwrap();
                if let Some(v) = self.array_element_get(handle, i) {
                    return Ok(v);
                }
                // A hole / out-of-range index on a plain array consults the prototype.
                self.read_member(handle, &alloc::format!("{i}"))
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                if let Some(i) = k.as_number().and_then(as_index)
                    && self.realm.is_array_like(handle)
                {
                    if let Some(v) = self.array_element_get(handle, i) {
                        return Ok(v);
                    }
                    return self.read_member(handle, &alloc::format!("{i}"));
                }
                let name = self.coerce_property_key(k)?;
                self.read_member(handle, &name)
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => self.read_member(handle, s),
            PropertyKey::Number(n) => self.read_member(handle, &alloc::format!("{n}")),
            // Private names (`this.#x`) are stored under a `#`-prefixed key.
            PropertyKey::Private(s) => {
                // `obj.#x` where obj's class did not declare `#x` is a TypeError, not
                // `undefined`. An instance holder carries the brand as an own private
                // element (field or method) or a private accessor. A *class* receiver
                // (`Class.#static`) is resolved by read_member's separate per-class storage,
                // so it is not brand-checked here.
                let key = self.private_access_key(s);
                if !self.is_callable(handle)
                    && self.realm.class_at(handle).is_none()
                    && !self.realm.has_own(handle, &key)
                    && self.realm.accessor(handle, &key).is_none()
                {
                    let m = self.new_str(&alloc::format!(
                        "Cannot read private member #{s} from an object whose class did not declare it"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // Reading a private accessor declared with only a setter
                // (`set #g(v) {}`) is a TypeError — there is no getter.
                if let Some((getter, _)) = self.realm.accessor(handle, &key)
                    && matches!(getter.unpack(), Unpacked::Undefined)
                {
                    let m = self.new_str(&alloc::format!(
                        "Cannot read private member #{s} which has only a setter"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                self.read_member(handle, &key)
            }
        }
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

    /// Assigns a member by an already-evaluated key value (used when the target's
    /// computed key must be resolved before the RHS, per spec evaluation order).
    /// Mirrors `assign_member`'s proxy / array-index / setter / length handling.
    pub(crate) fn assign_member_value(
        &mut self,
        handle: crate::heap::Handle,
        key: NanBox,
        new: NanBox,
    ) -> Result<(), ExecError> {
        // Proxy `set` trap (or forward to the target).
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            if let Some(trap) = self.proxy_trap(handler, "set")? {
                let name = self.member_key(key);
                let key_box = self.new_str(&name);
                let recv = NanBox::handle(handle.to_raw());
                let handler_box = NanBox::handle(handler.to_raw());
                let r = self.call_with_this(
                    trap,
                    handler_box,
                    &[NanBox::handle(target.to_raw()), key_box, new, recv],
                )?;
                // A `set` trap returning a falsy value is a failed [[Set]]: a strict-mode
                // assignment then throws a TypeError (sloppy mode fails silently).
                if !self.realm.truthy(r) {
                    if self.strict {
                        let m = self.new_str(&alloc::format!(
                            "'set' on proxy: trap returned falsish for property '{name}'"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    return Ok(());
                }
                // A truthy result is subject to the [[Set]] success invariants.
                self.proxy_set_invariant_check(target, &name, new)?;
                return Ok(());
            }
            return self.assign_member_value(target, key, new);
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
                    self.store_array_index(handle, i, new)?;
                } else {
                    self.set_element_checked(handle, i, new)?;
                }
                return Ok(());
            }
        }
        let name = self.coerce_property_key(key)?;
        // A typed array's `length` is fixed (non-writable): ignore the assignment.
        if name == "length" && self.realm.typed_len(handle).is_some() {
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
            }
            return Ok(());
        }
        // No own property: an *inherited* accessor on the prototype chain handles the
        // write (its setter runs with `this` = the receiver). An inherited data
        // property, or none, falls through to creating an own data property.
        if !self.realm.has_own(handle, &name) {
            let mut cur = self.realm.object_proto(handle);
            while let Some(p) = cur {
                if let Some((_, setter)) = self.realm.accessor(p, &name) {
                    if !matches!(setter.unpack(), Unpacked::Undefined) {
                        let this = NanBox::handle(handle.to_raw());
                        self.call_with_this(setter, this, &[new])?;
                    }
                    return Ok(());
                }
                if self.realm.has_own(p, &name) {
                    break;
                }
                cur = self.realm.object_proto(p);
            }
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
            }
        }
        Ok(())
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
            // A non-writable `length`: reject a real change.
            let cur = self.realm.array_length(handle).unwrap_or(0);
            if n != cur {
                if self.strict {
                    let m = self.new_str("Cannot assign to read only property 'length'");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                return Ok(()); // sloppy: silently dropped
            }
            return Ok(()); // same-value: no-op
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
    pub(crate) fn array_length_from_value(&mut self, v: NanBox) -> Result<usize, ExecError> {
        // ToNumber(v) — abrupt-propagating (a Symbol/throwing valueOf).
        let num = self.coerce_to_number(v)?;
        match self.realm.array_length_uint32(num) {
            Some(n) => Ok(n as usize),
            None => {
                let m = self.new_str("Invalid array length");
                Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))))
            }
        }
    }

    pub(crate) fn read_member(
        &mut self,
        handle: crate::heap::Handle,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        // A Deferred Module Namespace (`import defer`) evaluates its target the
        // first time one of its exports is read (import-defer proposal).
        #[cfg(all(feature = "module", feature = "std"))]
        self.trigger_deferred_namespace(handle, name)?;
        // A **module namespace** export is a *live* binding: read the current
        // value from its backing slot (so a mutation in the exporting module that
        // happens after the namespace was materialised is observed). The
        // refreshed value is also written back so `getOwnPropertyDescriptor`
        // reports it.
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some(map) = self.module_namespaces.get(&handle.to_raw())
            && let Some((scope, local)) = map.get(name)
        {
            let value = scope.get(local).unwrap_or_else(NanBox::undefined);
            // Refresh the stored data property (it is non-configurable but
            // writable, so the engine-internal write is permitted).
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
        if let Ok(i) = name.parse::<usize>() {
            if let Some(leaf) = self.realm.string_leaf_bytes(handle) {
                let unit = crate::wtf8::utf16_index(leaf, i);
                return Ok(match unit {
                    Some(u) => self.new_str_bytes(crate::wtf8::from_utf16(&[u])),
                    None => NanBox::undefined(),
                });
            }
            if let Some(bytes) = self.realm.string_bytes(handle) {
                return Ok(match crate::wtf8::utf16_index(&bytes, i) {
                    Some(u) => self.new_str_bytes(crate::wtf8::from_utf16(&[u])),
                    None => NanBox::undefined(),
                });
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
            && i < self.realm.array_length(handle).unwrap_or(0)
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
        // Proxy `get` trap (or forward the read to the target).
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            if let Some(trap) = self.proxy_trap(handler, "get")? {
                let key = self.new_str(name);
                let recv = NanBox::handle(handle.to_raw());
                let handler_box = NanBox::handle(handler.to_raw());
                let result = self.call_with_this(
                    trap,
                    handler_box,
                    &[NanBox::handle(target.to_raw()), key, recv],
                )?;
                // Invariants (10.5.8): a non-configurable, non-writable data
                // property of the target must be reported with its actual value; a
                // non-configurable accessor with no getter must report undefined.
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
                return Ok(result);
            }
            return self.read_member(target, name);
        }
        // An error object's `.constructor` is its specific error global — its
        // prototype otherwise reports a generic `Object`. Recognized by an own
        // `name` in the error family plus a `message` (so a user `new Foo()`,
        // whose constructor resolves through its prototype, is never matched).
        if name == "constructor" {
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
        // A constructor function's `.prototype` (lazily created), so
        // `Fn.prototype.method = …` and prototype-chain inheritance work.
        if name == "prototype"
            && let Some((func_id, _)) = self.realm.function_at(handle)
        {
            let proto = self.realm.function_prototype(func_id);
            return Ok(NanBox::handle(proto.to_raw()));
        }
        // A class's `.prototype` (lazily materialized with its instance
        // methods/accessors and a `constructor` back-link).
        if name == "prototype"
            && let Some((class_id, _)) = self.realm.class_at(handle)
        {
            let proto = self.class_prototype(class_id, handle);
            return Ok(NanBox::handle(proto.to_raw()));
        }
        // A bound function's `name` is `"bound " + target.name` (recursing so a
        // re-bound function reads `"bound bound …"`); its `length` is the target's
        // length minus the bound arguments (floored at 0).
        if matches!(name, "name" | "length")
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
            // `length`: target.length − number of pre-bound arguments.
            let tlen = match th {
                Some(t) => {
                    let v = self.read_member(t, "length")?;
                    self.realm.to_number(v)
                }
                None => 0.0,
            };
            let bound = self
                .realm
                .get_property(handle, BOUND_ARGS)
                .and_then(|a| a.as_handle().map(Handle::from_raw))
                .and_then(|bh| self.realm.array_length(bh))
                .unwrap_or(0);
            return Ok(NanBox::number((tlen - bound as f64).max(0.0)));
        }
        // `obj.__proto__` reads the prototype link (unless shadowed by an own
        // data property of that name).
        // The `__proto__` magic only applies when the object actually inherits
        // `Object.prototype`'s accessor; a null-proto object (module namespace,
        // `Object.create(null)`) reads it as an ordinary absent property.
        if name == "__proto__"
            && !self.realm.has_own(handle, "__proto__")
            && self.realm.inherits_object_proto(handle)
        {
            return Ok(match self.realm.object_proto(handle) {
                Some(p) => NanBox::handle(p.to_raw()),
                None => NanBox::null(),
            });
        }
        // A class's `name` is its declared identifier (`class C {}` → `"C"`), or
        // the name bound by NamedEvaluation (`let C = class {}`), which is stored
        // as an own property — so an own `name` takes precedence over the (empty)
        // declared id of an anonymous class.
        if name == "name"
            && self.realm.class_at(handle).is_some()
            && !self.realm.has_own(handle, "name")
        {
            let cname = self
                .realm
                .class_at(handle)
                .and_then(|(cid, _)| self.classes[cid as usize].id.as_ref())
                .map_or("", |i| &i.name);
            return Ok(self.new_str(cname));
        }
        // A function's `length` (params before a default/rest) and `name`.
        if matches!(name, "length" | "name")
            && !self.realm.has_own(handle, name)
            && let Some((func_id, _)) = self.realm.function_at(handle)
        {
            let def = self.functions[func_id as usize];
            return Ok(if name == "length" {
                let len = def
                    .params
                    .iter()
                    .take_while(|p| p.default.is_none() && !p.rest)
                    .count();
                NanBox::number(len as f64)
            } else {
                self.new_str(def.name)
            });
        }
        // A built-in function's `name` and `length`. Plain natives carry `name` in
        // their aux object (resolved above / via `member_value`) but no physical
        // `length`; first-class prototype/static methods (bound natives) carry
        // neither. Synthesize both from the dispatch identity so every built-in
        // function exposes the spec-mandated own `name`/`length` data properties.
        if matches!(name, "length" | "name") && !self.realm.has_own(handle, name) {
            if let Some((id, target)) = self.realm.bound_native_at(handle) {
                let method = if id == N_ARRAY_PROTO_FN
                    || id == N_AB_PROTO_FN
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
        // A class static — walking the `extends` chain for inherited statics. The
        // own level is mirrored as a real own property (so `delete`/`defineProperty`
        // take effect); only fall through to the side tables for *inherited*
        // statics, which live on the superclass and are not mirrored on `handle`.
        if let Some((cid, _)) = self.realm.class_at(handle) {
            // The own level is mirrored as a real own property of the constructor.
            // An own accessor falls through to the generic accessor path below
            // (invoked with `this` = the class); an own data property is
            // authoritative here (so `delete`/`defineProperty` are honored). Only
            // when the name is *not* an own property do we walk the superclass
            // chain via the side tables for an inherited static.
            let has_own_accessor = self.realm.accessor(handle, name).is_some_and(|(g, _)| {
                g.as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            });
            if !has_own_accessor {
                if self.realm.has_own(handle, name) {
                    if let Some(v) = self.realm.get_property(handle, name) {
                        return Ok(v);
                    }
                } else {
                    // Inherited statics: walk the superclass chain.
                    let class = self.classes[cid as usize];
                    let env = self.class_envs[cid as usize].clone();
                    let mut cur = self.resolve_super(class, &env)?.map(|(pid, _)| pid);
                    while let Some(c) = cur {
                        if let Some(v) = self.class_statics[c as usize].get(name) {
                            return Ok(*v);
                        }
                        if let Some(getter) = self.class_static_get[c as usize].get(name).copied() {
                            let this = NanBox::handle(handle.to_raw());
                            return self.call_with_this(getter, this, &[]);
                        }
                        let class = self.classes[c as usize];
                        let env = self.class_envs[c as usize].clone();
                        cur = self.resolve_super(class, &env)?.map(|(pid, _)| pid);
                    }
                }
            }
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
            && !self.realm.has_own(handle, "lastIndex")
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
            return Err(self
                .type_error("ArrayBuffer.prototype accessor called on a non-ArrayBuffer object"));
        }
        if self.realm.get_property(handle, DATA_VIEW_BUF).is_none()
            && matches!(name, "buffer" | "byteLength" | "byteOffset")
            && self.brand_on_chain(handle, DATA_VIEW_PROTO_BRAND)
        {
            return Err(
                self.type_error("DataView.prototype accessor called on a non-DataView object")
            );
        }
        if self.realm.typed_kind(handle).is_none()
            && matches!(name, "buffer" | "byteLength" | "byteOffset" | "length")
            && self.brand_on_chain(handle, TYPED_ARRAY_PROTO_BRAND)
        {
            return Err(
                self.type_error("TypedArray.prototype accessor called on a non-TypedArray object")
            );
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
            && self.realm.string_bytes(ph).is_some()
        {
            if name == "length" {
                let len = if let Some(leaf) = self.realm.string_leaf_bytes(ph) {
                    crate::wtf8::utf16_len(leaf)
                } else {
                    crate::wtf8::utf16_len(&self.realm.string_bytes(ph).unwrap_or_default())
                };
                return Ok(NanBox::number(len as f64));
            }
            if let Ok(i) = name.parse::<usize>() {
                let unit = if let Some(leaf) = self.realm.string_leaf_bytes(ph) {
                    crate::wtf8::utf16_index(leaf, i)
                } else {
                    crate::wtf8::utf16_index(&self.realm.string_bytes(ph).unwrap_or_default(), i)
                };
                return Ok(match unit {
                    Some(u) => self.new_str_bytes(crate::wtf8::from_utf16(&[u])),
                    None => NanBox::undefined(),
                });
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
        while let Some(p) = cur {
            // A proxy in the prototype chain handles the read via its own `[[Get]]`
            // (a `get` trap, or forwarding to the target and its prototype chain),
            // which is terminal for the lookup.
            if self.realm.proxy_at(p).is_some() {
                return self.read_member(p, name);
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
        let ctor_name = if self.realm.string_value(handle).is_some() {
            "String"
        } else if self.realm.is_array_like(handle) {
            "Array"
        } else if let Some(is_set) = self.realm.collection_is_set(handle) {
            if is_set { "Set" } else { "Map" }
        } else if self.realm.function_at(handle).is_some()
            || self.realm.native_at(handle).is_some()
            || self.realm.bound_native_at(handle).is_some()
        {
            "Function"
        } else {
            return None;
        };
        let proto = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|ns| self.realm.get_property(ns, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)?;
        let m = self.realm.get_property(proto, name)?;
        (!matches!(m.unpack(), Unpacked::Undefined)).then_some(m)
    }

    pub(crate) fn eval_assign(
        &mut self,
        op: AssignOp,
        target: &'a Expr,
        value: &'a Expr,
    ) -> Result<NanBox, ExecError> {
        // Logical assignment (`&&=`/`||=`/`??=`) short-circuits: the right side
        // is evaluated and stored only when the current value warrants it.
        if matches!(
            op,
            AssignOp::AndAssign | AssignOp::OrAssign | AssignOp::NullishAssign
        ) {
            let current = self.read_target(target)?;
            let assign = match op {
                AssignOp::AndAssign => self.realm.truthy(current),
                AssignOp::OrAssign => !self.realm.truthy(current),
                _ => matches!(current.unpack(), Unpacked::Undefined | Unpacked::Null),
            };
            if !assign {
                return Ok(current);
            }
            let rhs = self.eval(value)?;
            self.assign_to(target, rhs)?;
            return Ok(rhs);
        }
        // A computed-member target evaluates the object and key *before* the RHS
        // (spec order): `arr[i] = i = 1` writes the original `arr[i]`. A computed
        // `super[expr]` target is excluded here — it has no evaluable base object
        // and is handled by the `super` assignment arm below.
        if let Expr::Member {
            object,
            property: PropertyKey::Computed(key_expr),
            ..
        } = target
            && !matches!(&**object, Expr::Super(_))
        {
            let obj = self.eval(object)?;
            // Spec reference order: evaluate the base, then the key expression,
            // then (for a plain assignment) the RHS — *before* PutValue's
            // RequireObjectCoercible. So a `null`/`undefined` base still evaluates
            // the key and RHS, and only then throws a TypeError (not before).
            let key = self.eval(key_expr)?;
            let Some(raw) = obj.as_handle() else {
                // `null`/`undefined` (or a number/boolean) base: a number/boolean
                // is a primitive whose write is silently ignored in sloppy mode.
                // For a *compound* op the LHS `GetValue` (RequireObjectCoercible)
                // runs before the RHS, so a `null`/`undefined` base throws *before*
                // the RHS is evaluated; a plain `=` defers the throw past the RHS.
                let is_nullish = matches!(obj.unpack(), Unpacked::Null | Unpacked::Undefined);
                if op != AssignOp::Assign && is_nullish {
                    return Err(self.type_error("Cannot read property of null or undefined"));
                }
                let rhs = self.eval(value)?;
                if is_nullish {
                    return Err(self.type_error("Cannot set property of null or undefined"));
                }
                return Ok(rhs);
            };
            let handle = crate::heap::Handle::from_raw(raw);
            let mut key = key;
            let new = if op == AssignOp::Assign {
                // Plain `=`: ToPropertyKey is deferred to PutValue, i.e. *after* the
                // RHS — so the key's `toString` runs after the RHS is evaluated.
                self.eval(value)?
            } else {
                // Compound `op=`: the LHS reference's GetValue runs before the RHS
                // and performs ToPropertyKey on the key exactly once. For an object
                // key, coerce now (a throwing or observable `toString` therefore
                // runs before the RHS, and only once) and reuse the resulting
                // primitive key for both the read and the write. Primitive keys
                // (number / string / symbol) are left as-is so the array-index and
                // typed-array fast paths in `read_member_value` still apply.
                if key.as_handle().is_some_and(|raw| {
                    let h = Handle::from_raw(raw);
                    self.realm.symbol_at(h).is_none() && self.realm.string_value(h).is_none()
                }) {
                    let pk = self.coerce_property_key(key)?;
                    key = self.new_str(&pk);
                }
                let current = self.read_member_value(handle, key)?;
                let rhs = self.eval(value)?;
                self.binary(compound_op(op)?, current, rhs)?
            };
            self.assign_member_value(handle, key, new)?;
            return Ok(new);
        }
        // A computed `super[expr] = …` target: the key expression is evaluated
        // before the RHS (spec reference order), then the inherited setter is
        // invoked with the current `this`.
        if let Expr::Member {
            object,
            property: PropertyKey::Computed(key_expr),
            ..
        } = target
            && matches!(&**object, Expr::Super(_))
        {
            // Evaluate the key *expression* first; for a plain assignment the RHS
            // is evaluated before the key is ToPropertyKey-coerced, so
            // `super[obj] = rhs()` runs `rhs` before `obj.toString` (the spec
            // defers a super reference's key coercion past the RHS). A compound op
            // must read `super[key]` first, so it coerces the key up front.
            let k = self.eval(key_expr)?;
            let (name, new) = if op == AssignOp::Assign {
                let rhs = self.eval(value)?;
                (self.coerce_property_key(k)?, rhs)
            } else {
                let name = self.coerce_property_key(k)?;
                let current = self.resolve_super_member(&name)?;
                let rhs = self.eval(value)?;
                (name, self.binary(compound_op(op)?, current, rhs)?)
            };
            self.assign_super_member(&name, new)?;
            return Ok(new);
        }
        // A *compound* assignment to a static (non-computed, non-super) member
        // target follows spec reference order: evaluate the base (`lref`), read the
        // current value (`lval = GetValue(lref)`), *then* evaluate the RHS, apply the
        // op, and write back. So `obj.x op= rhs()` reads `obj.x` before running
        // `rhs()`, and a nullish base throws *before* the RHS is evaluated.
        // (Computed-key targets are handled by the branch above.)
        if op != AssignOp::Assign
            && let Expr::Member {
                object, property, ..
            } = target
            && !matches!(&**object, Expr::Super(_))
            && !matches!(property, PropertyKey::Computed(_))
        {
            let obj = self.eval(object)?;
            let Some(raw) = obj.as_handle() else {
                if matches!(obj.unpack(), Unpacked::Null | Unpacked::Undefined) {
                    return Err(self.type_error("Cannot read property of null or undefined"));
                }
                // A primitive base: read the (boxed) current value, evaluate the RHS
                // for side effects, then ignore the write (sloppy mode).
                let key = static_key(property)?;
                let boxed = self.coerce_to_object(obj);
                let current = match boxed.as_handle() {
                    Some(br) => self.read_member(crate::heap::Handle::from_raw(br), &key)?,
                    None => NanBox::undefined(),
                };
                let rhs = self.eval(value)?;
                return self.binary(compound_op(op)?, current, rhs);
            };
            let handle = crate::heap::Handle::from_raw(raw);
            let current = self.member(handle, property)?;
            let rhs = self.eval(value)?;
            let new = self.binary(compound_op(op)?, current, rhs)?;
            self.assign_member(handle, property, new)?;
            return Ok(new);
        }
        // For `name = class {}`, hand the LHS name to `make_class` so an anonymous
        // class's `name` is set before its static initializers run.
        if op == AssignOp::Assign
            && let Expr::Ident(id) = target
            && let Expr::Class(c) = value
            && c.id.is_none()
        {
            self.pending_class_name = Some(&id.name);
        }
        let rhs = self.eval(value)?;
        self.pending_class_name = None;
        // Destructuring assignment: `[a, b] = …` / `({ x } = …)`.
        if op == AssignOp::Assign && matches!(target, Expr::Array { .. } | Expr::Object { .. }) {
            self.assign_destructure(target, rhs)?;
            return Ok(rhs);
        }
        match target {
            Expr::Ident(id) => {
                let name = &*id.name;
                // An imported binding (`import { x } from "m"`) is an immutable
                // indirect binding: assigning to it is a TypeError (module code is
                // strict). The alias only applies when the name is not shadowed by
                // a binding in the current scope chain — a same-named *local* of
                // another module (e.g. a callee defined in a different module whose
                // own `x` happens to match this module's import alias) is a normal,
                // mutable binding.
                #[cfg(all(feature = "module", feature = "std"))]
                if self.module_imports.contains_key(name) && self.current.get(name).is_none() {
                    let m = self.new_str("Assignment to constant variable.");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A bare identifier inside `with (obj)` reads/writes the with-object's
                // property when it provides the name (so `with(o){ x op= v }` and
                // setters/getters work).
                if let Some(h) = self.with_binding(name) {
                    let new = if op == AssignOp::Assign {
                        rhs
                    } else {
                        let current = self.read_member(h, name)?;
                        self.binary(compound_op(op)?, current, rhs)?
                    };
                    let key = self.new_str(name);
                    self.assign_member_value(h, key, new)?;
                    return Ok(new);
                }
                // Reassigning a `const` binding is a TypeError.
                if self.current.is_const(name) {
                    let m = self.new_str("Assignment to constant variable.");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let new = if op == AssignOp::Assign {
                    // NamedEvaluation: `x = function(){}` / `x = () => {}` /
                    // `x = class {}` names the anonymous definition after the LHS
                    // identifier (only for a plain `=`, and only when the RHS is an
                    // anonymous function/arrow/class).
                    if matches!(value, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_)) {
                        self.set_fn_name(rhs, name);
                    }
                    rhs
                } else {
                    // A compound assignment reads the LHS first (`GetValue`); an
                    // unresolvable reference throws a catchable ReferenceError (matching
                    // a bare-identifier read), not an internal error.
                    let current = self.read_ident_ref(name)?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                if !self.current.set(name, new) {
                    if self.strict {
                        let m = self.new_str(&alloc::format!("{name} is not defined"));
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    }
                    // Sloppy implicit global: bind on the global scope + object.
                    self.declare_sloppy_global(name, new);
                }
                Ok(new)
            }
            Expr::Member {
                object, property, ..
            } if matches!(&**object, Expr::Super(_)) => {
                // `super.x = v` (and `super.x op= v`) invokes the inherited setter with
                // the current `this`; a compound op reads through `super.x` first.
                let name = self.eval_prop_key(property)?;
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self.resolve_super_member(&name)?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                self.assign_super_member(&name, new)?;
                Ok(new)
            }
            Expr::Member {
                object, property, ..
            } => {
                let obj = self.eval(object)?;
                let Some(raw) = obj.as_handle() else {
                    // A `null`/`undefined` base throws a TypeError; another primitive
                    // (number/boolean) silently ignores the write in sloppy mode.
                    if matches!(obj.unpack(), Unpacked::Null | Unpacked::Undefined) {
                        return Err(self.type_error("Cannot set property of null or undefined"));
                    }
                    return Ok(rhs);
                };
                let handle = crate::heap::Handle::from_raw(raw);
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self.member(handle, property)?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                self.assign_member(handle, property, new)?;
                Ok(new)
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
    }

    pub(crate) fn assign_member(
        &mut self,
        handle: crate::heap::Handle,
        property: &'a PropertyKey,
        new: NanBox,
    ) -> Result<(), ExecError> {
        // A **module namespace exotic object**'s `[[Set]]` always fails (§28.3.6):
        // its bindings are not assignable through the namespace. In strict code
        // (module code always is) the failed Set is a TypeError.
        #[cfg(all(feature = "module", feature = "std"))]
        if self.module_namespaces.contains_key(&handle.to_raw()) {
            if self.strict {
                return Err(self.type_error(
                    "cannot assign to a read-only property of a module namespace object",
                ));
            }
            return Ok(());
        }
        // `regex.lastIndex = n` updates the RegExp's stateful search position
        // (honoring a non-writable descriptor installed via `defineProperty`).
        if let PropertyKey::Ident(s) | PropertyKey::Str(s) = property
            && &**s == "lastIndex"
            && self.realm.regexp_at(handle).is_some()
        {
            return self.regex_write_last_index(handle, new);
        }
        // `obj.__proto__ = proto` updates the prototype link (like
        // `Object.setPrototypeOf`); a non-object, non-null value is ignored.
        if let PropertyKey::Ident(s) | PropertyKey::Str(s) = property
            && &**s == "__proto__"
        {
            match new.unpack() {
                Unpacked::Null => {
                    self.realm.set_object_proto(handle, None);
                }
                _ => {
                    if let Some(p) = new.as_handle().map(Handle::from_raw) {
                        self.realm.set_object_proto(handle, Some(p));
                    }
                }
            }
            return Ok(());
        }
        // Writing a static on a class (`C.field = v`, `++C.field`). Statics are
        // mirrored as real own properties on the constructor, so an own accessor is
        // invoked through that mirror and an own data write lands on the mirror —
        // keeping reflection and the fast read path (which now reads the mirror)
        // in sync. An *inherited* static setter (on a superclass) is still
        // dispatched via the side tables.
        if let Some((cid, _)) = self.realm.class_at(handle) {
            let key = self.eval_prop_key(property)?;
            // Own accessor (getter/setter installed on this constructor's mirror).
            if let Some((_, setter)) = self.realm.accessor(handle, &key) {
                if !matches!(setter.unpack(), Unpacked::Undefined) {
                    let this = NanBox::handle(handle.to_raw());
                    self.call_with_this(setter, this, &[new])?;
                }
                // A getter-only own accessor: the write is silently ignored
                // (non-strict) — matching ordinary accessor semantics.
                return Ok(());
            }
            // Inherited static setter (walk the superclass chain).
            let class = self.classes[cid as usize];
            let env = self.class_envs[cid as usize].clone();
            let mut cur = self.resolve_super(class, &env)?.map(|(pid, _)| pid);
            while let Some(c) = cur {
                if let Some(setter) = self.class_static_set[c as usize].get(&key).copied() {
                    let this = NanBox::handle(handle.to_raw());
                    self.call_with_this(setter, this, &[new])?;
                    return Ok(());
                }
                if self.class_static_get[c as usize].contains_key(&key) {
                    // Inherited getter-only: write ignored (non-strict).
                    return Ok(());
                }
                let class = self.classes[c as usize];
                let env = self.class_envs[c as usize].clone();
                cur = self.resolve_super(class, &env)?.map(|(pid, _)| pid);
            }
            // Plain own data static: update both the mirror (authoritative for
            // reflection/reads) and the side table (kept consistent for any
            // remaining side-table consumer).
            self.realm.set_property(handle, &key, new);
            self.class_statics[cid as usize].insert(key, new);
            return Ok(());
        }
        // Proxy `set` trap (or forward the write to the target).
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            if let Some(trap) = self.proxy_trap(handler, "set")? {
                let key = self.eval_prop_key(property)?;
                let key_box = self.new_str(&key);
                let recv = NanBox::handle(handle.to_raw());
                let handler_box = NanBox::handle(handler.to_raw());
                let r = self.call_with_this(
                    trap,
                    handler_box,
                    &[NanBox::handle(target.to_raw()), key_box, new, recv],
                )?;
                // A `set` trap returning a falsy value is a failed [[Set]]: a strict-mode
                // assignment then throws a TypeError (sloppy mode fails silently).
                if !self.realm.truthy(r) {
                    if self.strict {
                        let m = self.new_str(&alloc::format!(
                            "'set' on proxy: trap returned falsish for property '{key}'"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    return Ok(());
                }
                // A truthy result is subject to the [[Set]] success invariants.
                self.proxy_set_invariant_check(target, &key, new)?;
                return Ok(());
            }
            return self.assign_member(target, property, new);
        }
        // An accessor setter — own or inherited via the prototype chain — takes
        // precedence over creating a data property. A private accessor
        // (`set #x() {…}`) is stored under the `#`-prefixed key, so resolve that.
        let setter_key: Option<alloc::string::String> = match property {
            PropertyKey::Ident(s) | PropertyKey::Str(s) => Some(String::from(&**s)),
            PropertyKey::Private(s) => Some(self.private_access_key(s)),
            _ => None,
        };
        if let Some(skey) = setter_key {
            let mut cur = Some(handle);
            while let Some(c) = cur {
                // OrdinarySet: a proxy *on the prototype chain* (above the original
                // receiver) handles the write via its own `[[Set]]` — invoke its
                // `set` trap with Receiver = the original object, then stop.
                if c != handle
                    && let Some((target, p_handler)) = self.realm.proxy_at(c)
                {
                    self.guard_revoked(c)?;
                    if let Some(trap) = self.proxy_trap(p_handler, "set")? {
                        let key_box = self.new_str(&skey);
                        let recv = NanBox::handle(handle.to_raw());
                        let handler_box = NanBox::handle(p_handler.to_raw());
                        let r = self.call_with_this(
                            trap,
                            handler_box,
                            &[NanBox::handle(target.to_raw()), key_box, new, recv],
                        )?;
                        if self.strict && !self.realm.truthy(r) {
                            let m = self.new_str(&alloc::format!(
                                "'set' on proxy: trap returned falsish for property '{skey}'"
                            ));
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        return Ok(());
                    }
                    // No `set` trap: the proxy's `[[Set]]` is OrdinarySet on the
                    // target with the *original* receiver. If the target has an
                    // inherited accessor it would fire, but the common case is a
                    // data property (or absent), which creates/updates an OWN data
                    // property on the original receiver. Stop the prototype walk and
                    // fall through to the own-property write on `handle` — unless the
                    // target itself has a *setter* for this key, which must run.
                    if let Some((_, setter)) = self.realm.accessor(target, &skey)
                        && !matches!(setter.unpack(), Unpacked::Undefined)
                    {
                        let this = NanBox::handle(handle.to_raw());
                        self.call_with_this(setter, this, &[new])?;
                        return Ok(());
                    }
                    break;
                }
                if let Some((_, setter)) = self.realm.accessor(c, &skey) {
                    if !matches!(setter.unpack(), Unpacked::Undefined) {
                        let this = NanBox::handle(handle.to_raw());
                        self.call_with_this(setter, this, &[new])?;
                    } else if self.strict || matches!(property, PropertyKey::Private(_)) {
                        // Writing a getter-only accessor is a TypeError in strict
                        // mode; for a *private* accessor it always throws (there is
                        // no silent-failure path for private references).
                        let m = self.new_str(&alloc::format!(
                            "Cannot set property {skey} which has only a getter"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    // A getter-only accessor still shadows a data assignment.
                    return Ok(());
                }
                // An own data property below shadows an inherited accessor.
                if self.realm.has_own(c, &skey) {
                    break;
                }
                cur = self.realm.object_proto(c);
            }
        }
        match property {
            PropertyKey::Number(n) if as_index(*n).is_some() && self.realm.is_array(handle) => {
                self.store_array_index(handle, as_index(*n).unwrap(), new)?;
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                // A numeric index only addresses array storage; on an object a
                // numeric key is the equivalent string property.
                if let Some(i) = k.as_number().and_then(as_index)
                    && self.realm.is_array(handle)
                {
                    self.store_array_index(handle, i, new)?;
                } else {
                    let name = self.coerce_property_key(k)?;
                    if self.allow_property_write(handle, &name)? {
                        self.realm.set_property(handle, &name, new);
                    }
                }
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                // `arr.length = n` resizes the array (truncate/pad), rather than
                // storing a `length` property.
                if &**s == "length" && self.realm.is_array(handle) {
                    let n = self.array_length_from_value(new)?;
                    self.write_array_length(handle, n)?;
                } else if &**s == "prototype"
                    && let Some((func_id, _)) = self.realm.function_at(handle)
                    && let Some(praw) = new.as_handle()
                {
                    // `Fn.prototype = obj` reassigns the constructor's prototype.
                    self.realm
                        .set_function_prototype(func_id, Handle::from_raw(praw));
                } else if self.allow_property_write(handle, s)? {
                    self.realm.set_property(handle, s, new);
                }
            }
            PropertyKey::Number(n) => {
                self.realm.set_property(handle, &alloc::format!("{n}"), new);
            }
            PropertyKey::Private(s) => {
                // Writing `obj.#x` where obj's class did not declare `#x` is a TypeError.
                // (Field initialization writes via `set_property` directly, not this path,
                // so the initial creation of a field is exempt; a class receiver, for
                // static privates, is resolved via separate per-class storage.)
                let key = self.private_access_key(s);
                if !self.is_callable(handle)
                    && self.realm.class_at(handle).is_none()
                    && !self.realm.has_own(handle, &key)
                    && self.realm.accessor(handle, &key).is_none()
                {
                    let m = self.new_str(&alloc::format!(
                        "Cannot write private member #{s} to an object whose class did not declare it"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // PrivateSet on a private *method* is a TypeError (methods are
                // non-writable). Such a property is installed read-only, so an
                // own read-only private key here is a method, not a field.
                if self.realm.has_own(handle, &key) && self.realm.property_is_readonly(handle, &key)
                {
                    let m = self.new_str(&alloc::format!(
                        "Cannot write to private method or accessor #{s}"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                self.realm.set_property(handle, &key, new);
            }
        }
        Ok(())
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
        Ok(match op {
            UnaryOp::Plus => {
                let p = self.coerce_object(v, "number")?;
                NanBox::number(self.realm.to_number(p))
            }
            UnaryOp::Minus => {
                let p = self.coerce_object(v, "number")?;
                self.realm.neg(p)
            }
            UnaryOp::Not => self.realm.logical_not(v),
            UnaryOp::Typeof => {
                let t = self.realm.type_of_value(v);
                NanBox::handle(self.realm.new_string(t).to_raw())
            }
            UnaryOp::Void => NanBox::undefined(),
            #[cfg(feature = "std")]
            UnaryOp::BitNot => {
                // ToPrimitive(Number) first, so `~obj` honors a user `valueOf`.
                let p = self.coerce_object(v, "number")?;
                self.realm.bit_not(p)
            }
            #[cfg(not(feature = "std"))]
            UnaryOp::BitNot => return Err(ExecError::Unsupported("~ needs std")),
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
            let r = match op {
                BinaryOp::Add => val(self, x.add(&y)),
                BinaryOp::Sub => val(self, x.sub(&y)),
                BinaryOp::Mul => val(self, x.mul(&y)),
                BinaryOp::Div => match x.divmod(&y) {
                    Some((q, _)) => val(self, q),
                    None => return Err(throw(self, "Division by zero")),
                },
                BinaryOp::Mod => match x.divmod(&y) {
                    Some((_, rem)) => val(self, rem),
                    None => return Err(throw(self, "Division by zero")),
                },
                BinaryOp::Exp => {
                    if y.is_negative() {
                        return Err(throw(self, "Exponent must be non-negative"));
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
                    .is_some_and(|raw| this.realm.string_value(Handle::from_raw(raw)).is_some())
            };
            if is_str(self, a) || is_str(self, b) {
                return Ok(None);
            }
        }
        // BigInt vs Number: compare numerically (`<`/`==` only).
        if matches!(
            op,
            BinaryOp::EqEq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::LtEq
                | BinaryOp::GtEq
        ) {
            let to_f = |n: &crate::bignum::BigInt| n.to_i128().map_or(f64::NAN, |v| v as f64);
            let xn = abig.as_ref().map_or_else(|| self.realm.to_number(a), to_f);
            let yn = bbig.as_ref().map_or_else(|| self.realm.to_number(b), to_f);
            let r = match op {
                BinaryOp::EqEq => xn == yn,
                BinaryOp::NotEq => xn != yn,
                BinaryOp::Lt => xn < yn,
                BinaryOp::Gt => xn > yn,
                BinaryOp::LtEq => xn <= yn,
                _ => xn >= yn,
            };
            return Ok(Some(NanBox::boolean(r)));
        }
        // Mixed arithmetic is a TypeError.
        let m = self.new_str("Cannot mix BigInt and other types");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
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
                    && this.realm.string_value(h).is_none()
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
            (
                self.coerce_primitive(a, hint)?,
                self.coerce_primitive(b, hint)?,
            )
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
            // True for a non-string heap value (object or array).
            let obj = |this: &Self, v: NanBox| {
                v.as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| this.realm.string_value(h).is_none())
            };
            // True for a number, boolean, or string primitive — the operands
            // against which an object is converted with ToPrimitive (a boolean is
            // first coerced to a number per the `==` algorithm).
            let prim = |this: &Self, v: NanBox| {
                v.as_number().is_some()
                    || matches!(v.unpack(), crate::nanbox::Unpacked::Bool(_))
                    || v.as_handle()
                        .map(Handle::from_raw)
                        .is_some_and(|h| this.realm.string_value(h).is_some())
            };
            if obj(self, a) && prim(self, b) {
                (self.coerce_for_eq(a)?, b)
            } else if obj(self, b) && prim(self, a) {
                (a, self.coerce_for_eq(b)?)
            } else {
                (a, b)
            }
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
                let key = self.member_key(a);
                // A Deferred Module Namespace (`import defer`) evaluates its target
                // on a `[[HasProperty]]` with a String (non-"then") key.
                #[cfg(all(feature = "module", feature = "std"))]
                if let Some(h) = b.as_handle().map(Handle::from_raw) {
                    self.trigger_deferred_namespace(h, &key)?;
                }
                let present = match b.as_handle().map(Handle::from_raw) {
                    // Proxy `has` trap, or forward to the target.
                    Some(h) if self.realm.proxy_at(h).is_some() => {
                        let (target, handler) = self.realm.proxy_at(h).unwrap();
                        self.guard_revoked(h)?;
                        if let Some(trap) = self.proxy_trap(handler, "has")? {
                            let kb = self.new_str(&key);
                            let r = self.call(trap, &[NanBox::handle(target.to_raw()), kb])?;
                            self.realm.truthy(r)
                        } else {
                            self.realm.has_own(target, &key) || self.realm.is_array(target)
                        }
                    }
                    // Integer-indexed exotic `[[HasProperty]]`: for a typed array, a
                    // canonical numeric index reports `IsValidIntegerIndex` and the
                    // prototype chain is **not** consulted; any other key falls through
                    // to ordinary `[[HasProperty]]` (own + inherited).
                    Some(h)
                        if self.realm.typed_kind(h).is_some()
                            && canonical_numeric_index(&key).is_some() =>
                    {
                        let n = canonical_numeric_index(&key).unwrap();
                        let detached = self.typed_array_detached(h);
                        let is_neg_zero = n == 0.0 && n.is_sign_negative();
                        !detached
                            && !is_neg_zero
                            && n == (n as i64) as f64
                            && n >= 0.0
                            && self
                                .realm
                                .typed_len(h)
                                .is_some_and(|len| (n as usize) < len)
                    }
                    Some(h) => {
                        // `key in obj` is true for an own *or inherited* property
                        // (walk the prototype chain); arrays also report in-bounds
                        // indices and `length`.
                        let in_chain = || {
                            let mut cur = Some(h);
                            while let Some(c) = cur {
                                if self.realm.has_own(c, &key) {
                                    return true;
                                }
                                cur = self.realm.object_proto(c);
                            }
                            false
                        };
                        // `has_own` already reports an array's in-range non-hole
                        // indices and `length`, so the chain walk suffices (a hole
                        // is absent unless inherited).
                        in_chain()
                    }
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
        if !(self.is_callable(ch) || self.realm.class_at(ch).is_some()) {
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
            if let Some(mh) = method.as_handle().map(Handle::from_raw)
                && self.is_callable(mh)
                // Skip the *default* `Function.prototype[Symbol.hasInstance]`
                // (every function inherits it): it just performs OrdinaryHasInstance,
                // which is exactly the ordinary path below — calling it here would
                // recurse. Only a *user* `[Symbol.hasInstance]` override is honored.
                && self.realm.native_at(mh) != Some(N_FN_HAS_INSTANCE)
            {
                let result = self.call_with_this(method, ctor, &[obj])?;
                return Ok(self.realm.truthy(result));
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
            || self.realm.function_at(ch).is_some()
            || self.realm.class_at(ch).is_some()
            || self.realm.bound_native_at(ch).is_some()
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
            || self.realm.string_value(oh).is_some()
            || self.realm.bigint_at(oh).is_some()
        {
            return Ok(false);
        }
        // Built-in constructors: check the cell kind directly.
        if let Some(id) = self.realm.native_at(ch) {
            // A primitive wrapper (`new Number(…)`) matches its constructor.
            if let Some(wt) = self.realm.get_property(oh, PRIM_WRAP_TYPE)
                && wt.as_number() == Some(f64::from(id))
            {
                return Ok(true);
            }
            // A typed array matches its constructor (kind index == id − base).
            if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16)
                .contains(&id)
                && self.realm.typed_kind(oh) == Some((id - N_TYPED_ARRAY_BASE) as u8)
            {
                return Ok(true);
            }
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
            // `ArrayBuffer` / `DataView` match by their marker slot. (A typed array is
            // a `Cell::TypedArray`, not an object with `ARRAY_BUFFER_BYTES`, so
            // `typedArray instanceof ArrayBuffer` is correctly false.)
            if id == N_ARRAY_BUFFER && self.realm.get_property(oh, ARRAY_BUFFER_BYTES).is_some() {
                return Ok(true);
            }
            if id == N_DATA_VIEW && self.realm.get_property(oh, DATA_VIEW_BUF).is_some() {
                return Ok(true);
            }
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
                let want = ERROR_NAMES[(id - N_ERROR_BASE) as usize];
                // A user class extending a native error: walk its class chain for
                // a native error super (so `customErr instanceof Error` holds even
                // when the subclass overrides `this.name`).
                if let Some(tag) = self.realm.class_tag(oh) {
                    let mut cur = Some(tag);
                    while let Some(cid) = cur {
                        if let Some(nsup) = self.class_native_super[cid as usize]
                            && (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16)
                                .contains(&nsup)
                        {
                            let have = ERROR_NAMES[(nsup - N_ERROR_BASE) as usize];
                            if want == "Error" || want == have {
                                return Ok(true);
                            }
                        }
                        cur = self
                            .resolve_super(
                                self.classes[cid as usize],
                                &self.class_envs[cid as usize].clone(),
                            )?
                            .map(|(p, _)| p);
                    }
                }
                // Plain error objects: match by the `name` property.
                let obj_name = self
                    .realm
                    .get_property(oh, "name")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                if !ERROR_NAMES.contains(&obj_name.as_str()) {
                    return Ok(false);
                }
                return Ok(want == "Error" || obj_name == want);
            }
            match id {
                N_REGEXP => return Ok(self.realm.regexp_at(oh).is_some()),
                N_MAP | N_SET | N_WEAKMAP | N_WEAKSET => {
                    return Ok(self.realm.collection_is_set(oh).is_some());
                }
                N_DATE => return Ok(self.realm.date_at(oh).is_some()),
                N_PROMISE => return Ok(self.realm.promise_state(oh).is_some()),
                // Every callable (function, native, bound) and every class is a
                // `Function`.
                N_FUNCTION => {
                    return Ok(self.is_callable(oh) || self.realm.class_at(oh).is_some());
                }
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
            if self.realm.string_value(oh).is_some()
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
        // Plain function constructors: walk the instance's `[[Prototype]]` chain for
        // the constructor's current `.prototype` (so `Object.create(C.prototype)` is an
        // instance, and reassigning `C.prototype` is reflected). `Get(C,"prototype")`
        // must be an Object — otherwise OrdinaryHasInstance is a TypeError (e.g.
        // `C.prototype = undefined`).
        if self.realm.function_at(ch).is_some() {
            let proto_val = self.read_member(ch, "prototype")?;
            let Some(proto) = proto_val
                .as_handle()
                .map(Handle::from_raw)
                .filter(|_| self.is_object_value(proto_val))
            else {
                return Err(
                    self.type_error("Function has non-object prototype in instanceof check")
                );
            };
            // Walk via `get_proto_of` so a proxy's `getPrototypeOf` trap is honored at
            // each step (bounded to guard against a trap returning a cycle).
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
            return Ok(false);
        }
        let (Some(tag), Some((target_id, _))) = (self.realm.class_tag(oh), self.realm.class_at(ch))
        else {
            return Ok(false);
        };
        // Walk the instance's class chain (its class, then each `extends`).
        let mut cur = Some(tag);
        while let Some(cid) = cur {
            if cid == target_id {
                return Ok(true);
            }
            let class = self.classes[cid as usize];
            // Resolve the superclass in the class's own captured scope.
            let env = self.class_envs[cid as usize].clone();
            cur = self.resolve_super(class, &env)?.map(|(pid, _)| pid);
        }
        Ok(false)
    }
}
