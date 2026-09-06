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

    /// The `(old, next)` pair for a `++`/`--` on `current`: `old` is the numeric (or
    /// BigInt) value to yield for a postfix update, `next` the incremented/decremented
    /// value to store. Runs `ToNumeric` (a BigInt stays BigInt; an object operand goes
    /// through ToPrimitive, whose `valueOf`/`toString` may throw).
    fn update_value(
        &mut self,
        op: crate::ast::UpdateOp,
        current: NanBox,
    ) -> Result<(NanBox, NanBox), ExecError> {
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
            let old_box = NanBox::handle(self.realm.new_bigint(big).to_raw());
            return Ok((old_box, next_box));
        }
        let coerced = self.coerce_to_number(current)?;
        let old = self.realm.to_number(coerced);
        let next = match op {
            crate::ast::UpdateOp::Inc => old + 1.0,
            crate::ast::UpdateOp::Dec => old - 1.0,
        };
        Ok((NanBox::number(old), NanBox::number(next)))
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

    /// Whether `v` is an object (a non-primitive heap value: object/array/function/…)
    /// rather than a string/symbol/bigint primitive or an immediate.
    pub(crate) fn is_object_value(&self, v: NanBox) -> bool {
        v.as_handle().map(Handle::from_raw).is_some_and(|h| {
            !self.realm.is_string_handle(h)
                && self.realm.symbol_at(h).is_none()
                && self.realm.bigint_at(h).is_none()
        })
    }

    /// Builds a tagged template's argument list `[stringsObject, ...substitutions]`.
    /// The frozen strings object (with its `.raw` array) is created once per
    /// template-literal site and reused on every evaluation — its identity is
    /// observable to the tag. Shared by the ordinary tagged-template evaluation
    /// and the proper-tail-call path (a tagged template's tag is called in tail
    /// position).
    pub(crate) fn tagged_template_args(
        &mut self,
        quasi: &'a crate::ast::TemplateLiteral,
    ) -> Result<Vec<NanBox>, ExecError> {
        let cache_key = (core::ptr::from_ref(quasi) as usize, self.eval_site_epoch);
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
            let raw: Vec<NanBox> = quasi.quasis.iter().map(|q| self.new_str(&q.raw)).collect();
            let strings_h = self.realm.new_array(strings);
            // The strings object carries a `.raw` array (for `String.raw` and tags
            // reading `strings.raw`). Both arrays are frozen, per spec — freeze
            // `.raw` first and `strings` last so the property write lands.
            let raw_h = self.realm.new_array(raw);
            self.realm.freeze_object(raw_h);
            self.realm
                .set_property(strings_h, "raw", NanBox::handle(raw_h.to_raw()));
            // Per spec the template object's `raw` is
            // `{ writable:false, enumerable:false, configurable:false }` — mark it
            // non-enumerable *before* freezing (freeze then locks writable /
            // configurable). Without this it enumerates in `for-in`/`Object.keys`.
            self.realm.mark_hidden(strings_h, "raw");
            self.realm.freeze_object(strings_h);
            let arr = NanBox::handle(strings_h.to_raw());
            self.tagged_template_cache.insert(cache_key, arr);
            arr
        };
        let mut args = alloc::vec![strings_arr];
        for e in &quasi.expressions {
            args.push(self.eval(e)?);
        }
        Ok(args)
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
                // Joined with the same O(1) rope concatenation `+` uses, rather
                // than copied into one flat buffer: accumulating a template
                // (`s = `${s}x`` in a loop) copied the whole substitution every
                // iteration, which is quadratic. Empty quasis — the common
                // `${a}${b}` shape — are skipped so this costs no extra cells.
                let mut acc = self.new_str_bytes(Vec::new());
                for (i, quasi) in t.quasis.iter().enumerate() {
                    match &quasi.cooked {
                        Some(cooked) if cooked.is_empty() => {}
                        Some(cooked) => {
                            let part = self.new_str_bytes(cooked.to_vec());
                            acc = self.concat_or_throw(acc, part)?;
                        }
                        // An invalid escape is allowed only in a *tagged* template; in a
                        // plain template literal it is a SyntaxError.
                        None => {
                            let m = self.new_str("Invalid escape sequence in template literal");
                            return Err(ExecError::Throw(self.make_error(N_SYNTAX_ERROR, Some(m))));
                        }
                    }
                    if let Some(e) = t.expressions.get(i) {
                        let v = self.eval(e)?;
                        // A value that is *already* a string is concatenated as-is.
                        // Going through `coerce_to_string_bytes` would materialize
                        // its rope — O(n) per substitution, which is what kept the
                        // accumulating form quadratic even after the join became
                        // O(1). ToString is the identity here, so nothing is
                        // skipped.
                        let part = if self.realm.is_string(v) {
                            v
                        } else {
                            let bytes = self.coerce_to_string_bytes(v)?;
                            if bytes.is_empty() {
                                continue;
                            }
                            self.new_str_bytes(bytes)
                        };
                        acc = self.concat_or_throw(acc, part)?;
                    }
                }
                Ok(acc)
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
                let args = self.tagged_template_args(quasi)?;
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
                        // A *plain* member whose base is nullish is NOT a short-circuit —
                        // `delete u.x` / `delete n[0]` does ToObject(base), which throws a
                        // TypeError — so track whether the target was optional.
                        let argument: &Expr = match &**argument {
                            Expr::OptChain { expr, .. } => expr,
                            other => other,
                        };
                        // Only the *last* link's own `?.` short-circuits the
                        // delete: in `delete [1]?.r[k]` the chain does not
                        // short-circuit (the base is non-nullish), so the final
                        // `[k]` reference is formed on `undefined` and throws.
                        let base_optional = matches!(argument, Expr::Member { optional: true, .. });
                        if let Expr::Member {
                            object, property, ..
                        } = argument
                        {
                            is_property_delete = true;
                            // `delete super.prop` / `delete super[expr]` is a runtime
                            // ReferenceError (a super reference is never deletable) —
                            // but the *reference* is evaluated first: GetThisBinding,
                            // then the key expression, whose side effects therefore
                            // still run (`delete super[sideEffect = 1]`).
                            if matches!(&**object, Expr::Super(_)) {
                                self.require_super_this()?;
                                if let PropertyKey::Computed(key_expr) = property {
                                    self.eval(key_expr)?;
                                }
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
                                // An optional target (`delete a?.b` with nullish `a`)
                                // short-circuits to `true`; a plain member delete on a
                                // nullish base throws a TypeError (ToObject fails).
                                if base_optional {
                                    return Ok(NanBox::boolean(true));
                                }
                                let m = self.new_str("Cannot convert undefined or null to object");
                                return Err(ExecError::Throw(
                                    self.make_error(N_TYPE_ERROR, Some(m)),
                                ));
                            }
                            if let Some(raw) = obj.as_handle() {
                                let h = Handle::from_raw(raw);
                                let name = match property {
                                    PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                                        Some(String::from(&**s))
                                    }
                                    PropertyKey::Computed(e) => {
                                        // ToPropertyKey, not ToString: an object key
                                        // runs ToPrimitive (so a user `toString`
                                        // fires), and a Symbol result keeps its
                                        // identity — `delete o[{toString(){return
                                        // sym}}]` removes `o[sym]`.
                                        let k = self.eval(e)?;
                                        Some(self.coerce_property_key(k)?)
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
                                            let kb = self.key_to_value(&name);
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
                                }
                            }
                        } else if let Expr::Ident(id) = argument {
                            if let Some(frame) = self.current.owner_frame(&id.name) {
                                // A resolvable lexical/var binding is non-deletable
                                // (a no-op returning `false`) EXCEPT a binding a
                                // sloppy `eval` introduced as deletable into a
                                // non-global variable environment
                                // (EvalDeclarationInstantiation
                                // `CreateMutableBinding(name, true)`): those are
                                // removed and return `true`, after which the name
                                // resolves to a ReferenceError.
                                if !frame.ptr_eq(&self.global_scope)
                                    && frame.is_local_deletable(&id.name)
                                {
                                    frame.delete_local(&id.name);
                                    result = true;
                                } else if frame.ptr_eq(&self.global_scope)
                                    && let Some(g) = self.global_object()
                                    && (self.realm.has_own(g, &id.name)
                                        || self.realm.accessor(g, &id.name).is_some())
                                    && !self.realm.property_is_non_configurable(g, &id.name)
                                {
                                    // A built-in global (`JSON`, `Math`, a constructor,
                                    // …) is a *configurable* property of the global
                                    // object that this engine mirrors as a global-scope
                                    // binding. `delete JSON` removes the property, so
                                    // the mirror must go too. A global `var`/function
                                    // declaration is non-configurable and stays.
                                    result = self.realm.delete_property(g, &id.name);
                                    if result {
                                        frame.delete_local(&id.name);
                                    }
                                    is_property_delete = true;
                                } else {
                                    result = false;
                                }
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
                            // returned (there is no binding/property to remove). A
                            // short-circuiting optional call (`delete null?.()`,
                            // `delete o.f?.()`) yields `undefined`, which is likewise
                            // not a Reference — so the `delete` still returns `true`.
                            match self.eval(argument) {
                                Ok(_) => {}
                                Err(ExecError::OptShortCircuit) => {
                                    return Ok(NanBox::boolean(true));
                                }
                                Err(e) => return Err(e),
                            }
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
                            // A binding may live only on the global object (e.g.
                            // `globalThis.x = …`, a built-in declared onto the global
                            // object rather than the lexical scope, or a member
                            // *inherited* from `%Object.prototype%`) — `typeof` must
                            // see it, not report "undefined".
                            && !self.global_object_provides(&id.name)
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
                // AnnexB web-compat: a direct CallExpression operand of `++`/`--`
                // parses in sloppy code but is a runtime ReferenceError (the call is
                // evaluated first for its side effects). Strict mode rejected it at
                // parse time.
                if argument.is_web_compat_call_target() {
                    self.eval(argument)?;
                    let m = self
                        .new_str("Invalid left-hand side expression in prefix/postfix operation");
                    return Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(m)),
                    ));
                }
                // `++super[key]` / `super.x--`: the SuperProperty reference is
                // evaluated once (GetThisBinding, then — for a computed key — the key
                // expression + GetSuperBase, captured before ToPropertyKey), then
                // GetValue and PutValue share it.
                if let Expr::Member {
                    object, property, ..
                } = &**argument
                    && matches!(&**object, Expr::Super(_))
                {
                    self.require_super_this()?;
                    let (name, obj_base) = match property {
                        PropertyKey::Computed(key_expr) => {
                            let k = self.eval(key_expr)?;
                            let obj_base = self.object_super_base();
                            (self.coerce_property_key(k)?, obj_base)
                        }
                        _ => {
                            let name = self.eval_prop_key(property)?;
                            (name, self.object_super_base())
                        }
                    };
                    let current = match obj_base {
                        Some(Some(proto)) => self.read_super_member_object(proto, &name)?,
                        Some(None) => {
                            return Err(self.type_error("Cannot read property of null (super)"));
                        }
                        None => self.resolve_super_member(&name)?,
                    };
                    let (old, next) = self.update_value(*op, current)?;
                    match obj_base {
                        Some(Some(proto)) => self.assign_super_member_object(proto, &name, next)?,
                        Some(None) => {
                            return Err(self.type_error("Cannot set property on null (super)"));
                        }
                        None => self.assign_super_member(&name, next)?,
                    }
                    return Ok(if *prefix { next } else { old });
                }
                // For a member target, the reference is evaluated exactly once: the
                // base and (computed) key run once, then GetValue + PutValue share
                // them — so `obj[keyWithSideEffect()]++` does not run the key twice.
                if let Expr::Member {
                    object, property, ..
                } = &**argument
                    && !matches!(&**object, Expr::Super(_))
                {
                    let obj = self.eval(object)?;
                    // Evaluate the computed key *expression* now (observable side
                    // effects: `base[f()]--` runs `f()`), but defer ToPropertyKey
                    // (its `toString`) until after the null/undefined-base check —
                    // so `null[objWithThrowingToString]--` is a TypeError, not the
                    // key's `toString` error.
                    let raw_key = match property {
                        PropertyKey::Computed(e) => Some(self.eval(e)?),
                        _ => None,
                    };
                    if matches!(obj.unpack(), Unpacked::Null | Unpacked::Undefined) {
                        return Err(self.type_error("Cannot read properties of null or undefined"));
                    }
                    // A primitive base is boxed for the read; the write then lands on
                    // the throwaway wrapper (a no-op, as for any primitive property set).
                    let handle = match obj.as_handle() {
                        Some(raw) => Handle::from_raw(raw),
                        None => Handle::from_raw(
                            self.coerce_to_object(obj)
                                .as_handle()
                                .ok_or_else(|| self.type_error("cannot convert to object"))?,
                        ),
                    };
                    let key = match raw_key {
                        Some(kv) => self.coerce_property_key(kv)?,
                        None => self.eval_prop_key(property)?,
                    };
                    let current = self.read_member(handle, &key)?;
                    let (old, next) = self.update_value(*op, current)?;
                    let key_box = self.new_str(&key);
                    self.assign_member_value(handle, key_box, next)?;
                    return Ok(if *prefix { next } else { old });
                }
                // A bare identifier resolving to a `with`-object binding: resolve
                // the object ONCE, so a read through a self-mutating getter (e.g.
                // `get x(){ delete this.x; return 2 }`) does not change where the
                // write lands — GetValue and PutValue share the same reference.
                if let Expr::Ident(id) = &**argument
                    && let Some(h) = self.with_binding(&id.name)
                {
                    let name = &*id.name;
                    // Both `GetBindingValue` (the read) and `SetMutableBinding` (the
                    // write) of an object Environment Record re-run
                    // `? HasProperty(bindingObject, N)` after the `HasBinding`
                    // resolution. A getter that deletes its own property
                    // (`get x(){ delete this.x; return 2 }`) makes the second check
                    // fail, which is a ReferenceError for a strict reference.
                    let current = if self.has_property_proxied(h, name)? {
                        self.read_member(h, name)?
                    } else if self.strict {
                        let m = self.new_str(&alloc::format!("{name} is not defined"));
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    } else {
                        NanBox::undefined()
                    };
                    let (old, next) = self.update_value(*op, current)?;
                    if !self.has_property_proxied(h, name)? && self.strict {
                        let m = self.new_str(&alloc::format!("{name} is not defined"));
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    }
                    let key = self.new_str(name);
                    self.assign_member_value(h, key, next)?;
                    return Ok(if *prefix { next } else { old });
                }
                let current = self.read_target(argument)?;
                let (old, next) = self.update_value(*op, current)?;
                self.assign_to(argument, next)?;
                Ok(if *prefix { next } else { old })
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
                    // §13.10.1: `PrivateIdentifier in ShiftExpression` throws a
                    // TypeError when the right-hand value is not an Object (rather
                    // than reporting the brand as absent).
                    if !self.is_object_value(obj) {
                        let m = self.new_str(
                            "Cannot use 'in' operator to check for a private name in a non-object",
                        );
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
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
                op,
                target,
                value,
                paren_target,
                ..
            } => self.eval_assign(*op, target, value, *paren_target),
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
                // `import.defer(x)` — the import-defer proposal: load + link but do
                // not evaluate, returning a promise of the Deferred Module
                // Namespace (which evaluates lazily on first access).
                #[cfg(all(feature = "module", feature = "std"))]
                if let Expr::Member {
                    object, property, ..
                } = &**callee
                    && matches!(&**object, Expr::Ident(id) if id.name.as_ref() == "import")
                    && matches!(property, PropertyKey::Ident(p) if &**p == "defer")
                {
                    return self.dynamic_import_deferred(arguments);
                }
                // `import.source(x)` — the source-phase proposal, unimplemented.
                // ToString the specifier (a throw rejects with that), then return a
                // promise rejected with a SyntaxError — NOT a plain dynamic import.
                #[cfg(all(feature = "module", feature = "std"))]
                if let Expr::Member {
                    object, property, ..
                } = &**callee
                    && matches!(&**object, Expr::Ident(id) if id.name.as_ref() == "import")
                    && matches!(property, PropertyKey::Ident(p) if &**p == "source")
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
                // `import.meta(…)` — the meta-property is an ordinary (non-callable)
                // object, so this must be a TypeError, not a ReferenceError for the
                // bare `import` reference. Evaluate the meta-property, then call it.
                #[cfg(all(feature = "module", feature = "std"))]
                if let Expr::Member {
                    object, property, ..
                } = &**callee
                    && matches!(&**object, Expr::Ident(id) if id.name.as_ref() == "import")
                    && matches!(property, PropertyKey::Ident(p) if &**p == "meta")
                {
                    let f = self.eval(callee)?;
                    let args = self.eval_args(arguments)?;
                    return self.call(f, &args);
                }
                // `super(args)` — invoke the base constructor on the current
                // instance.
                if matches!(&**callee, Expr::Super(_)) {
                    // SuperCall evaluation order (ECMA-262 13.3.7.1):
                    // `func` = GetSuperConstructor() — the *live*
                    // `[[GetPrototypeOf]]` of the running constructor's function
                    // object — is captured FIRST, then ArgumentListEvaluation, then
                    // the IsConstructor check, then
                    // `Construct(func, args, newTarget)`, and only *then*
                    // `BindThisValue` — whose "already initialized" check (a second
                    // `super()`) is a ReferenceError thrown *after* the base
                    // constructor has run. So even a doomed second `super()` still
                    // evaluates its arguments and invokes the base constructor (whose
                    // side effects therefore happen); its result is then discarded.
                    //
                    // The instance + this class's id are stashed in
                    // `pending_this_init` while `this` is in its TDZ (set by the
                    // derived constructor). A second `super()` leaves it `None` — the
                    // running class is then recovered from the lexical home (an arrow
                    // `() => super()` may outlive its constructor).
                    let home_cid = self.pending_this_init.map(|(_, c)| c).or(self.current_home);
                    let live_super = home_cid
                        .and_then(|c| self.class_handles.get(c as usize).copied())
                        .and_then(|cv| cv.as_handle().map(Handle::from_raw))
                        .map(|ch| {
                            self.realm
                                .object_proto(ch)
                                .map_or(NanBox::null(), |p| NanBox::handle(p.to_raw()))
                        });
                    let args = self.eval_args(arguments)?;
                    // Re-read the pending this-binding *after* the arguments: a
                    // nested `super()` among them (`super(super())`) already
                    // consumed it, which makes this the second SuperCall.
                    let pending = self.pending_this_init;
                    // GetSuperConstructor's IsConstructor check — *after* the
                    // arguments, so an argument that throws (or that re-points the
                    // constructor's `[[Prototype]]`) is observed first.
                    if let Some(sc) = live_super
                        && !self.is_constructor_value(sc)
                    {
                        let m = self.new_str("Super constructor is not a constructor");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    // Dispatch off the captured live super constructor, so
                    // `Object.setPrototypeOf(Derived, Other)` really redirects
                    // `super()` (the declaration-time binding below is only a
                    // fallback for the shapes this cannot classify).
                    let live_dispatch = live_super
                        .and_then(|sc| sc.as_handle().map(Handle::from_raw))
                        .and_then(|sh| {
                            if let Some((cid, cenv)) = self.realm.class_at(sh) {
                                Some((Some((cid, cenv)), None, None))
                            } else if let Some(nid) = self.native_base_kind(sh) {
                                Some((None, Some(nid), None))
                            } else if self.is_callable(sh) {
                                Some((None, None, Some(NanBox::handle(sh.to_raw()))))
                            } else {
                                None
                            }
                        });
                    // Resolve the super-constructor binding. Normally it is the
                    // transient `pending_super*` set while the enclosing derived
                    // constructor runs. But an arrow `() => super()` can be invoked
                    // *after* that constructor has returned (those transients are then
                    // cleared) — recover the binding from the arrow's lexical home
                    // class so the base still runs (SuperCall step 7) before
                    // BindThisValue throws.
                    let have_pending = self.pending_super.is_some()
                        || self.pending_super_native.is_some()
                        || self.pending_super_fn.is_some();
                    let (eff_super, eff_native, eff_fn) = if let Some(d) = live_dispatch {
                        d
                    } else if have_pending {
                        (
                            self.pending_super.clone(),
                            self.pending_super_native,
                            self.pending_super_fn,
                        )
                    } else if let Some(home) = self.current_home {
                        let native = self
                            .class_native_super
                            .get(home as usize)
                            .copied()
                            .flatten();
                        let fnp = self.class_fn_super.get(home as usize).copied().flatten();
                        let classp = if native.is_none()
                            && fnp.is_none()
                            && let Some(class) = self.classes.get(home as usize).copied()
                        {
                            let cenv = self.current.clone();
                            self.resolve_super(class, &cenv)?
                        } else {
                            None
                        };
                        (classp, native, fnp)
                    } else {
                        (None, None, None)
                    };
                    // The object the base constructor initializes: the derived
                    // instance on the first `super()`; a throwaway on a second one
                    // (the base still runs, but its result is discarded and
                    // BindThisValue then throws).
                    let (inst_val, first_call) = match pending {
                        Some((iv, _)) => (iv, true),
                        None => (NanBox::handle(self.realm.new_object().to_raw()), false),
                    };
                    let inst = inst_val.as_handle().map(Handle::from_raw);
                    // Point `this` at the object under construction and clear the
                    // pending marker BEFORE invoking the base constructor — the base's
                    // body reads `this`, and a nested second `super()` must now see
                    // `None`. On a second `super()` the already-bound `this` is saved
                    // and restored after the (discarded) base construction.
                    let saved_this_val = self.this_val;
                    self.this_val = inst_val;
                    self.pending_this_init = None;
                    // `Construct(superCtor, args, newTarget)`. An Object return rebinds
                    // `this` (`BindThisValue`), replacing the allocated instance.
                    let base_result = (|| -> Result<Option<NanBox>, ExecError> {
                        if let Some((pid, penv)) = eff_super {
                            match inst {
                                Some(h) => self.run_constructor(pid, &penv, h, &args),
                                None => Ok(None),
                            }
                        } else if let Some(nid) = eff_native {
                            // `super(...)` reaching a native constructor (`extends Error`).
                            if let Some(h) = inst {
                                self.apply_native_super(nid, h, &args)?;
                            }
                            Ok(None)
                        } else if let Some(fnp) = eff_fn {
                            // `super(...)` reaching an ordinary-function superclass:
                            // `[[Construct]](args, newTarget)` — the SuperCall's
                            // newTarget is the derived constructor's own `new.target`
                            // (so a `Reflect.construct(Derived, …, NT)` threads `NT`
                            // into the base function's `new.target`). Its object return
                            // overrides `this`.
                            //
                            // A **Proxy** superclass needs the real
                            // `[[Construct]]`: a `[[Call]]` never consults the
                            // `construct` trap and, for a trapless proxy wrapping a
                            // class, lands on the "class constructor cannot be
                            // invoked without 'new'" guard. `Construct` allocates
                            // the instance itself (from `newTarget.prototype`), and
                            // its object result rebinds `this` below.
                            if fnp
                                .as_handle()
                                .map(Handle::from_raw)
                                .is_some_and(|h| self.realm.proxy_at(h).is_some())
                            {
                                self.reflect_new_target = Some(self.new_target);
                                return self.construct(fnp, &args).map(Some);
                            }
                            self.pending_new_target = Some(self.new_target);
                            self.call_with_this(fnp, inst_val, &args).map(Some)
                        } else {
                            Err(ExecError::Unsupported(
                                "super outside a derived constructor",
                            ))
                        }
                    })();
                    if !first_call {
                        // Second `super()`: the base ran (side effects happened) on the
                        // throwaway; a base error still precedes BindThisValue, so
                        // propagate it first, then throw the "already initialized"
                        // ReferenceError. `this` and the field initializers are
                        // untouched (fields run exactly once, on the first `super()`).
                        base_result?;
                        self.this_val = saved_this_val;
                        let m = self.new_str("Super constructor may only be called once");
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    }
                    let returned = base_result?;
                    // A returned *object* (not a primitive wrapper handle) rebinds
                    // `this`; the field initializers below then target it.
                    let this_handle = match self.constructor_return_handle(returned) {
                        Some(h) => {
                            self.this_val = NanBox::handle(h.to_raw());
                            Some(h)
                        }
                        None => inst,
                    };
                    // `this` is now initialized (BindThisValue): publish it into the
                    // this-binding cell so an arrow captured before `super()` (whose
                    // lexical `this` was in its TDZ) observes the bound value. Done
                    // *after* the base constructor so such an arrow still sees TDZ
                    // while the base's own body runs.
                    if let Some(cell) = self.this_cell {
                        self.realm
                            .set_hidden_property(cell, THIS_CELL_SLOT, self.this_val);
                    }
                    // This class's field initializers run *after* `super()` returns —
                    // exactly once (a second `super()` took the branch above).
                    if let Some((_, derived_cid)) = pending
                        && let Some(h) = this_handle
                    {
                        self.init_instance_fields(derived_cid, h)?;
                    }
                    // `SuperCall` evaluates to the newly-bound `this` value
                    // (`thisER.BindThisValue(result)`), which a base constructor's
                    // object return may have overridden — so `x = super()` observes it.
                    return Ok(self.this_val);
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
                    // MakeSuperPropertyReference does GetThisBinding, which throws a
                    // ReferenceError if `this` is uninitialized (a derived
                    // constructor before `super(...)`).
                    self.require_super_this()?;
                    // EvaluateCall: the *callee* is read from the super reference
                    // (GetSuperBase + `[[Get]]`) before ArgumentListEvaluation, so
                    // an argument that mutates the home object's prototype cannot
                    // affect which method is invoked.
                    let f = self.resolve_super_method(&name)?;
                    let args = self.eval_args(arguments)?;
                    return self.call_with_this(f, self.this_val, &args);
                }
                // A **parenthesized** optional chain used as a call target, e.g.
                // `(a?.b)()` / `(a?.b)?.()`. A ParenthesizedExpression is
                // reference-transparent, so the call keeps `this` = the member's
                // base — unlike a bare `a?.b()` (one chain), the outer `()` is *not*
                // part of the chain, so a `?.` short-circuit inside yields
                // `undefined` which the outer call then invokes (a TypeError, unless
                // the outer call is itself `?.()`).
                if let Expr::OptChain { expr, .. } = &**callee
                    && let Expr::Member {
                        object,
                        property,
                        optional,
                        ..
                    } = &**expr
                {
                    let recv = match self.eval(object) {
                        Ok(v) => v,
                        // The inner chain short-circuited (`object` was nullish at a
                        // `?.`): the parenthesized value is `undefined`.
                        Err(ExecError::OptShortCircuit) => {
                            if *call_optional {
                                return Err(ExecError::OptShortCircuit);
                            }
                            let args = self.eval_args(arguments)?;
                            return self.call(NanBox::undefined(), &args);
                        }
                        Err(e) => return Err(e),
                    };
                    if *optional && matches!(recv.unpack(), Unpacked::Undefined | Unpacked::Null) {
                        if *call_optional {
                            return Err(ExecError::OptShortCircuit);
                        }
                        let args = self.eval_args(arguments)?;
                        return self.call(NanBox::undefined(), &args);
                    }
                    self.method_recv_check(recv, property, *optional)?;
                    let args = self.eval_args(arguments)?;
                    return self.call_member_dispatch(recv, property, *call_optional, &args);
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
                    // Resolving the callee member (`obj.m`) on a nullish base is a
                    // TypeError (or, for `obj?.m()`, an optional short-circuit),
                    // thrown *before* the arguments are evaluated (spec reference
                    // order): `o.bar.gar(foo())` throws before `foo()`.
                    self.method_recv_check(recv, property, *optional)?;
                    let args = self.eval_args(arguments)?;
                    return self.call_member_dispatch(recv, property, *call_optional, &args);
                }
                // A bare-identifier callee: resolve the reference **once** (a single
                // `HasBinding`) so the `has` trap of a `with (proxy)` frame fires
                // exactly once. If a `with` object provides the name, the call's
                // `this` is that object (`with (o) { m(); }` calls `o.m` with
                // `this`=`o`) and the callee read re-checks `HasProperty`
                // (`GetBindingValue`); otherwise finish the read against the lexical
                // / global scope without re-consulting the `with` chain.
                // Set when a `with` object supplied the callee *and* it is the
                // built-in `eval`: the reference is still the syntactic direct-eval
                // form, so the call must fall through to the direct-eval path below
                // rather than being dispatched as an ordinary method of the `with`
                // object (which would run it as an *indirect* eval, in global scope).
                let mut with_eval: Option<NanBox> = None;
                if let Expr::Ident(id) = &**callee {
                    let name = &*id.name;
                    if let Some(h) = self.with_binding_result(name)? {
                        // `GetBindingValue`: a binding deleted after `HasBinding`
                        // (via the `@@unscopables` getter) is a strict ReferenceError,
                        // else `undefined` (→ not-callable TypeError below).
                        let f = if self.has_property_proxied(h, name)? {
                            self.read_member(h, name)?
                        } else if self.strict {
                            let m = self.new_str(&alloc::format!("{name} is not defined"));
                            return Err(ExecError::Throw(
                                self.make_error(N_REFERENCE_ERROR, Some(m)),
                            ));
                        } else {
                            NanBox::undefined()
                        };
                        if *call_optional
                            && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null)
                        {
                            return Err(ExecError::OptShortCircuit);
                        }
                        let is_eval = !*call_optional
                            && name == "eval"
                            && f.as_handle().map(Handle::from_raw).is_some_and(|fh| {
                                self.realm.native_at(fh) == Some(N_EVAL)
                                    && self.get_function_realm(fh) == self.cur_realm
                            });
                        if is_eval {
                            with_eval = Some(f);
                        } else {
                            let args = self.eval_args(arguments)?;
                            return self.call_with_this(f, NanBox::handle(h.to_raw()), &args);
                        }
                    }
                }
                let f = if let Some(f) = with_eval {
                    f
                } else if let Expr::Ident(id) = &**callee {
                    // The `with` chain was already consulted above (no match); finish
                    // against the lexical / global scope so its `has` trap is not
                    // re-run.
                    self.read_ident_lexical(&id.name)?
                } else {
                    self.eval(callee)?
                };
                if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(ExecError::OptShortCircuit);
                }
                let args = self.eval_args(arguments)?;
                // Direct eval: the callee is the literal identifier `eval` and it
                // still resolves to the built-in `eval`. Such a call runs in the
                // caller's scope (so it can read/modify locals and hoist `var`s),
                // inheriting the caller's strictness — unlike an indirect eval,
                // which `call`/`call_native` route through the global scope.
                //
                // An *optional* call `eval?.(x)` is an OptionalChain, not the
                // direct-eval syntactic form (`CallExpression : MemberExpression
                // Arguments`), so it is always an *indirect* eval — fall through to
                // `self.call` below.
                // `SameValue(func, %eval%)` compares against the *running* realm's
                // eval intrinsic, so another realm's eval (`otherRealm.global.eval`)
                // is an ordinary — indirect — call even when spelled `eval(…)`.
                let is_current_realm_eval = |this: &mut Self, f: NanBox| {
                    f.as_handle().map(Handle::from_raw).is_some_and(|h| {
                        this.realm.native_at(h) == Some(N_EVAL)
                            && this.get_function_realm(h) == this.cur_realm
                    })
                };
                if let Expr::Ident(id) = &**callee
                    && !*call_optional
                    && id.name.as_ref() == "eval"
                    && is_current_realm_eval(self, f)
                {
                    let arg0 = args.first().copied().unwrap_or(NanBox::undefined());
                    // WTF-8 bytes, not a lossy `&str`: the program text is a JS
                    // string and may hold lone surrogates that a literal in it must
                    // reproduce (`eval("/" + String.fromCharCode(0xD800) + "/")`).
                    let Some(source) = arg0
                        .as_handle()
                        .and_then(|raw| self.realm.string_bytes(Handle::from_raw(raw)))
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
                            method,
                            span: member_span,
                        } => {
                            // `{ __proto__: obj }` — only the *unquoted identifier*
                            // form (not `"__proto__":`, computed, shorthand, or a
                            // method) sets the prototype; a quoted/computed key makes
                            // an ordinary own `__proto__` data property.
                            // The exclusion is *method definitions* (`__proto__() {}`,
                            // which is not the `PropertyName : AssignmentExpression`
                            // production) — NOT every function-valued right-hand side:
                            // `__proto__: function () {}` is still the prototype
                            // setter, and its function is therefore left anonymous.
                            if !shorthand
                                && !*method
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
                                    // A *numeric* literal key names the function after
                                    // its ToString form (`{5: function(){}}` → "5",
                                    // `{0.4: …}` → "0.4"); `k` already holds it.
                                    PropertyKey::Number(_) => {
                                        let name = k.clone();
                                        self.set_fn_name_owned(v, &name);
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
                                                .is_some_and(|h| self.fn_name_unset(h))
                                        {
                                            if matches!(&**value, Expr::Class(_)) {
                                                // A class already has its `length`; only
                                                // its `name` is set by NamedEvaluation.
                                                let nm = self.new_str(&name);
                                                if let Some(h) = v.as_handle().map(Handle::from_raw)
                                                {
                                                    // The class carries a readonly `name`
                                                    // `""` placeholder from creation, and
                                                    // `set_property` no-ops on a readonly
                                                    // property — clear the flag first so the
                                                    // NamedEvaluation name actually lands.
                                                    self.realm.clear_readonly_property(h, "name");
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
                            // inside it resolves through the object's prototype, and
                            // is flagged as a method (no `[[Construct]]`).
                            if *method
                                && matches!(&**value, Expr::Function(_))
                                && let Some(fv) = v.as_handle().map(Handle::from_raw)
                            {
                                // A concise method's source text (ECMA-262
                                // 20.2.3.5) is its whole MethodDefinition — the key,
                                // any `*`/`async`/`get`/`set` prefix, params and
                                // body — i.e. the object member's span, not the
                                // inner function-expression span `eval` stamped.
                                self.set_fn_source(v, *member_span);
                                self.realm.set_hidden_property(
                                    fv,
                                    HOME_OBJECT,
                                    NanBox::handle(handle.to_raw()),
                                );
                                if let Some((fid, _)) = self.realm.function_at(fv) {
                                    self.functions[fid as usize].is_method = true;
                                    // A concise method is non-constructable and has
                                    // no `prototype` — except a *generator* method,
                                    // which does. Strip the tentatively-materialized
                                    // property for the non-generator case.
                                    if !self.functions[fid as usize].is_generator {
                                        self.demote_fn_prototype(fv);
                                    }
                                }
                            }
                            // PropertyDefinitionEvaluation uses
                            // CreateDataPropertyOrThrow — a *define*, not a set — so
                            // a later data member replaces an accessor defined
                            // earlier in the same literal
                            // (`{ get x() { … }, ['x']: null }` ends up with a data
                            // `x`), rather than invoking (or being swallowed by) it.
                            self.realm.clear_accessor(handle, &k);
                            self.realm.set_property(handle, &k, v);
                        }
                        // `{ ...src }` — copy own enumerable properties.
                        ObjectMember::Spread { value, .. } => {
                            let src = self.eval(value)?;
                            self.object_spread_into(handle, src)?;
                        }
                        // `{ get x() {} }` / `{ set x(v) {} }`.
                        ObjectMember::Accessor {
                            key,
                            is_getter,
                            value,
                            span: member_span,
                        } => {
                            let k = self.eval_prop_key(key)?;
                            let f = self.make_function(
                                &value.params,
                                Body::Block(&value.body),
                                false,
                                false,
                            );
                            // The accessor's source text (ECMA-262 20.2.3.5) is its
                            // whole `get`/`set` MethodDefinition — the member span.
                            self.set_fn_source(f, *member_span);
                            // An object-literal accessor's `[[HomeObject]]` is this
                            // object, so `super.x` inside it resolves via the proto;
                            // an accessor is a method (no `[[Construct]]`).
                            if let Some(fh) = f.as_handle().map(Handle::from_raw) {
                                if let Some((fid, _)) = self.realm.function_at(fh) {
                                    self.functions[fid as usize].is_method = true;
                                }
                                // An accessor has no `prototype`.
                                self.demote_fn_prototype(fh);
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
                                    && self.fn_name_unset(fh)
                                {
                                    // `length` is the ExpectedArgumentCount: params
                                    // before the first one with a default / rest. A
                                    // setter with a defaulted param (`set m(x = 42)`)
                                    // therefore has `length` 0, not 1.
                                    let len = value
                                        .params
                                        .iter()
                                        .take_while(|p| p.default.is_none() && !p.rest)
                                        .count()
                                        as u32;
                                    self.install_fn_name_length(fh, &nm, len);
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
                        // GetThisBinding precedes evaluating the key expression: in a
                        // derived constructor before `super()`, `this` is uninitialized
                        // → ReferenceError, and the key expression is never evaluated.
                        self.require_super_this()?;
                        let key = self.eval(key_expr)?;
                        // For an object-literal method, GetSuperBase is captured here —
                        // *before* ToPropertyKey — so a key whose `toString` mutates the
                        // home object's prototype still reads from the original base.
                        if let Some(base) = self.object_super_base() {
                            let name = self.coerce_property_key(key)?;
                            let Some(proto) = base else {
                                return Err(self.type_error("Cannot read property of null (super)"));
                            };
                            return self.read_super_member_object(proto, &name);
                        }
                        // `ToPropertyKey`: a Symbol key must become its sentinel
                        // form (so `super[Symbol.x]` reads the real symbol-keyed
                        // property — and, for a deferred namespace, does *not*
                        // trigger evaluation), not a `"Symbol(…)"` display string.
                        let name = self.coerce_property_key(key)?;
                        return self.resolve_super_member(&name);
                    }
                    self.require_super_this()?;
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
                self.read_member_of(obj, property, *optional)
            }
            _ => Err(ExecError::Unsupported("expression")),
        }
    }

    /// Phase A of a method call `recv.property(...)`: the nullish-base check that
    /// runs *before* argument evaluation. Returns `Err(OptShortCircuit)` when the
    /// base is nullish and the member access is optional (`obj?.m()`), a
    /// `TypeError` when it is nullish and not optional, and `Ok(())` otherwise.
    /// Factored out so the generator/async step-machine can perform it eagerly
    /// (correct pre-argument order) before stepping suspending arguments.
    pub(crate) fn method_recv_check(
        &mut self,
        recv: NanBox,
        property: &'a PropertyKey,
        member_optional: bool,
    ) -> Result<(), ExecError> {
        if matches!(recv.unpack(), Unpacked::Undefined | Unpacked::Null) {
            if member_optional {
                return Err(ExecError::OptShortCircuit);
            }
            let key = match property {
                PropertyKey::Ident(s) | PropertyKey::Str(s) => alloc::string::String::from(&**s),
                PropertyKey::Number(n) => self.realm.to_display_string(NanBox::number(*n)),
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
        Ok(())
    }

    /// Phase C of a method call `recv.property(args)`: the built-in/own-property/
    /// primitive dispatch that runs *after* the receiver (already checked
    /// non-nullish by [`Self::method_recv_check`]) and the arguments have been
    /// evaluated. `call_optional` is the *call*'s own `?.()` flag. Factored out
    /// verbatim from the eager `Expr::Call` path so the generator/async
    /// step-machine can reify a method call whose *arguments* contain an
    /// `await`/`yield`: evaluate the receiver eagerly, step the arguments (so a
    /// suspension parks), then complete here with identical semantics (built-in
    /// dispatch, own-property shadowing, `this` binding, primitive boxing).
    pub(crate) fn call_member_dispatch(
        &mut self,
        recv: NanBox,
        property: &'a PropertyKey,
        call_optional: bool,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
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
                return self.call_with_this(f, recv, args);
            }
        }
        // A monkey-patched *inherited* `Promise.prototype.{then,catch,finally}` must
        // be honored on a direct `promise.then(…)` call: the native fast-path below
        // (`call_method`) bypasses the prototype chain, so a user reassignment of
        // these methods would otherwise be invisible (observable in reaction/species
        // call-counting tests). Only when the resolved method is a *user* function
        // (no native backing) do we route through it; the pristine intrinsic keeps
        // the fast path.
        {
            let pname = match property {
                PropertyKey::Ident(n) | PropertyKey::Str(n) => Some(&**n),
                _ => None,
            };
            if let Some(name) = pname
                && matches!(name, "then" | "catch" | "finally")
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
        }
        if let PropertyKey::Ident(name) | PropertyKey::Str(name) = property
            && let Some(result) = self.call_method(recv, name, args)?
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
                        || self.realm.get_property(h, GEN_COLL).is_some()
                        || self.realm.get_property(h, GEN_TA).is_some()
                }) {
                    return Ok(recv);
                }
                // A typed array yields a **live** values iterator.
                if let Some(h) = recv.as_handle().map(Handle::from_raw)
                    && self.realm.typed_kind(h).is_some()
                {
                    return Ok(self.make_live_typed_iterator(h, 1));
                }
                // A real array yields a **live** `%ArrayIterator%` (each `next()`
                // re-reads `length` and `Get`s the element at the cursor, so a
                // `push`/length change after `[Symbol.iterator]()` is observed —
                // `CreateArrayIterator`).
                if let Some(h) = recv.as_handle().map(Handle::from_raw)
                    && (self.realm.array_elements(h).is_some() || self.realm.is_array(h))
                {
                    return Ok(self.make_live_array_iterator(h, 1));
                }
                // A generic array-like object whose `@@iterator` is the intrinsic
                // `%Array.prototype.values%` (e.g. an `arguments` exotic object)
                // also iterates **live** over its `length` property.
                if let Some(h) = recv.as_handle().map(Handle::from_raw)
                    && self.realm.get_property(h, "length").is_some()
                {
                    let iter_key = self.member_key(iter_sym);
                    let own_iter = self.realm.get_property(h, &iter_key);
                    let arr_values = self
                        .realm
                        .array_proto_intrinsic()
                        .and_then(|p| self.realm.get_property(p, "values"));
                    if let (Some(oi), Some(av)) = (own_iter, arr_values)
                        && oi.as_handle().is_some()
                        && oi.as_handle() == av.as_handle()
                    {
                        return Ok(self.make_live_array_iterator(h, 1));
                    }
                }
                // A non-weak Map/Set yields a **live** iterator (a Set
                // over its values, a Map over its entries), so
                // `s[Symbol.iterator]()` observes mutation mid-iteration.
                if let Some(h) = recv.as_handle().map(Handle::from_raw)
                    && !self.realm.collection_is_weak(h)
                    && self.realm.collection_entries(h).is_some()
                {
                    let is_set = self.realm.collection_is_set(h) == Some(true);
                    let tag = if is_set {
                        "Set Iterator"
                    } else {
                        "Map Iterator"
                    };
                    let kind = if is_set { 1 } else { 2 };
                    return Ok(self.make_live_collection_iterator(h, kind, tag));
                }
                // None of the built-in iterable shapes above matched. Before
                // re-deriving iteration from the receiver's contents, honour a
                // *user* `[Symbol.iterator]()` — a class or object-literal method,
                // or a monkey-patched prototype entry: `obj[Symbol.iterator]()` is
                // an ordinary method call and must run that body (the derived
                // fallback would silently ignore it).
                if let Some(rh) = recv.as_handle().map(Handle::from_raw) {
                    let iter_key = self.member_key(iter_sym);
                    let f = self.read_member(rh, &iter_key)?;
                    if let Some(fh) = f.as_handle().map(Handle::from_raw)
                        && self.is_callable(fh)
                        && self.realm.native_at(fh).is_none()
                    {
                        return self.call_with_this(f, recv, args);
                    }
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
                    } else if self.realm.is_string_handle(h)
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
            if call_optional {
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
                PropertyKey::Number(n) => Some(self.realm.to_display_string(NanBox::number(*n))),
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
                    if call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) {
                        return Err(ExecError::OptShortCircuit);
                    }
                    if f.as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                    {
                        return self.call_with_this(f, recv, args);
                    }
                }
            }
            let m = self.new_str("is not a function");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        let f = self.member(Handle::from_raw(raw), property)?;
        // `f?.()` short-circuits when `f` is nullish.
        if call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Err(ExecError::OptShortCircuit);
        }
        // Method call: `this` is the receiver.
        self.call_with_this(f, recv, args)
    }

    /// Reads `property` off the already-evaluated member base `obj` — the tail of
    /// a (non-`super`) `Expr::Member` evaluation, factored out so the generator/
    /// async step-machine can reify a member read whose *object* contains an
    /// `await`/`yield` (evaluating the base step-by-step, then completing the read
    /// here with identical semantics — getters, computed keys, primitive bases).
    pub(crate) fn read_member_of(
        &mut self,
        obj: NanBox,
        property: &'a PropertyKey,
        optional: bool,
    ) -> Result<NanBox, ExecError> {
        if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
            if optional {
                // Short-circuit the rest of the enclosing optional chain.
                return Err(ExecError::OptShortCircuit);
            }
            // `null.x` / `undefined.x` throws a catchable TypeError.
            let msg = self.new_str("cannot read property of null or undefined");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(msg))));
        }
        let Some(raw) = obj.as_handle() else {
            // PrivateFieldGet / PrivateMethodOrAccessorGet step 2: if the receiver
            // is not an object (a primitive `this`, e.g. `method.call(15)` reaching
            // `this.#p`), throw a TypeError — a primitive can never carry a private
            // brand.
            if let PropertyKey::Private(s) = property {
                let m = self.new_str(&alloc::format!(
                    "Cannot read private member #{s} from a non-object"
                ));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // Reading a property of a number/boolean primitive follows
            // GetValue → ToObject → [[Get]]: box the primitive into its wrapper
            // (whose [[Prototype]] is the intrinsic `Number.prototype` /
            // `Boolean.prototype`) and read through it, so an inherited property
            // — a built-in like `constructor`/`toFixed` *or* a user-added
            // `Number.prototype.foo` — resolves instead of reporting `undefined`.
            if matches!(obj.unpack(), Unpacked::Number(_) | Unpacked::Bool(_)) {
                let wrapper = self.coerce_to_object(obj);
                if let Some(wh) = wrapper.as_handle().map(Handle::from_raw) {
                    // GetValue's `ToObject` supplies the *object* for `[[Get]]`,
                    // but `GetThisValue(V)` is still the primitive — so an
                    // inherited accessor's getter runs with the primitive as its
                    // receiver. A strict getter therefore sees `5`; a sloppy one
                    // is boxed by OrdinaryCallBindThis, as for any primitive
                    // `this`.
                    let key = self.eval_prop_key(property)?;
                    return self.get_with_receiver(wh, &key, obj);
                }
            }
            return Ok(NanBox::undefined());
        };
        let handle = crate::heap::Handle::from_raw(raw);
        self.member(handle, property)
    }

    /// PutValue for a reference whose base is a **primitive** (number, boolean,
    /// string, symbol, or BigInt): ToObject the primitive into its wrapper
    /// (`Number.prototype` / `String.prototype` / … on its chain) and perform
    /// `[[Set]]` with the primitive as the conceptual receiver. An inherited
    /// setter or a Proxy on the wrapper's prototype chain handles the write;
    /// otherwise the write would create an own data property on the non-object
    /// receiver, which fails — a strict-mode TypeError, a sloppy silent no-op.
    pub(crate) fn write_primitive_member(
        &mut self,
        prim: NanBox,
        property: &'a PropertyKey,
        new: NanBox,
    ) -> Result<(), ExecError> {
        let key = self.eval_prop_key(property)?;
        self.write_primitive_member_key(prim, &key, new)
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
            self.set_fn_source(f, func.span);
            // The name is an immutable binding: reassigning it inside the body
            // throws in strict mode and is a silent no-op in sloppy mode.
            self.current.declare_soft_const(&id.name, f);
            self.current = saved;
            return f;
        }
        let f = self.make_function(
            &func.params,
            Body::Block(&func.body),
            func.is_async,
            func.is_generator,
        );
        self.set_fn_source(f, func.span);
        f
    }

    pub(crate) fn eval_arrow(&mut self, arrow: &'a Arrow) -> NanBox {
        let body = match &arrow.body {
            ArrowBody::Expr(e) => Body::Expr(e),
            ArrowBody::Block(b) => Body::Block(b),
        };
        let f = self.make_function(&arrow.params, body, arrow.is_async, false);
        self.set_fn_source(f, arrow.span);
        // Arrows have no own `arguments` binding (they inherit the enclosing one).
        if let Some(raw) = f.as_handle()
            && let Some((func_id, _)) = self.realm.function_at(Handle::from_raw(raw))
        {
            self.functions[func_id as usize].is_arrow = true;
            // An arrow is not constructable: strip the `prototype` own property
            // `make_method` tentatively materialized (before `is_arrow` was set).
            self.demote_fn_prototype(Handle::from_raw(raw));
            // Capture the *lexical* `this`/`new.target`/home at the definition site
            // (hidden slots), so a later call (including via `call`/`apply`/`bind`)
            // resolves them from here rather than the call site.
            let h = Handle::from_raw(raw);
            self.realm.set_hidden_property(h, ARROW_THIS, self.this_val);
            // Inside a derived constructor before `super(...)`, the lexical `this`
            // is still in its temporal dead zone. Snapshotting `tdz()` would make
            // the arrow throw forever; instead capture the constructor's this-binding
            // *cell* so a call resolves the (later-bound) value live.
            if self.this_val.is_tdz()
                && let Some(cell) = self.this_cell
            {
                self.realm
                    .set_hidden_property(h, ARROW_THIS_CELL, NanBox::handle(cell.to_raw()));
            }
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
            // Private names are lexically scoped and survive paths that establish
            // no home class (a class's computed ClassElementName), so capture the
            // lexical class separately.
            if let Some(lc) = self.current_lexical_home {
                self.realm.set_hidden_property(
                    h,
                    ARROW_LEXICAL_CLASS,
                    NanBox::number(f64::from(lc)),
                );
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
            // Overwrite the `name` "" placeholder materialized at creation with the
            // NamedEvaluation name (own, non-enumerable, non-writable, configurable),
            // so `hasOwnProperty`/`getOwnPropertyDescriptor`/`verifyProperty` see it.
            let handle = Handle::from_raw(raw);
            let len = self.functions[func_id as usize]
                .params
                .iter()
                .take_while(|p| p.default.is_none() && !p.rest)
                .count() as u32;
            self.install_fn_name_length(handle, name, len);
            return;
        }
        // NamedEvaluation of an anonymous class: `let C = class {}` gives the
        // class constructor an own `name` of `"C"` (its `length` was already
        // installed at class creation). A class with a declared id keeps it.
        if let Some(raw) = value.as_handle() {
            let handle = Handle::from_raw(raw);
            if let Some((cid, _)) = self.realm.class_at(handle)
                && self.classes[cid as usize].id.is_none()
                // The class carries a default `name === ""` placeholder unless its
                // own body declares a `static name` element (which set the real
                // value). Only overwrite the placeholder — never an explicit one.
                && !self.class_declares_static_name(cid)
                && !self.class_has_own_name_element(handle)
            {
                let name_v = self.new_str(name);
                self.realm.clear_readonly_property(handle, "name");
                self.realm.set_property(handle, "name", name_v);
                self.realm.mark_hidden(handle, "name");
                self.realm.set_readonly_property(handle, "name");
            }
        }
    }

    /// `SetFunctionName` for a name known only at runtime (a `String`, not a
    /// source `&'a str`) — used by class field initializers, whose field name is
    /// a computed/private key resolved during evaluation. Mirrors [`set_fn_name`]
    /// but materializes only the `name` own property (the anonymous function's
    /// internal `name` stays `""`; `.name` reads resolve to the own property).
    ///
    /// [`set_fn_name`]: Self::set_fn_name
    pub(crate) fn set_fn_name_owned(&mut self, value: NanBox, name: &str) {
        if let Some(raw) = value.as_handle()
            && let Some((func_id, _)) = self.realm.function_at(Handle::from_raw(raw))
            // Don't clobber a name the function already has (a named function
            // expression keeps its own name over the field name).
            && self.functions[func_id as usize].name.is_empty()
        {
            let handle = Handle::from_raw(raw);
            let len = self.functions[func_id as usize]
                .params
                .iter()
                .take_while(|p| p.default.is_none() && !p.rest)
                .count() as u32;
            self.install_fn_name_length(handle, name, len);
            return;
        }
        // An anonymous class initializer (`#f = class {}`) takes the field name.
        if let Some(raw) = value.as_handle() {
            let handle = Handle::from_raw(raw);
            if let Some((cid, _)) = self.realm.class_at(handle)
                && self.classes[cid as usize].id.is_none()
                // The class carries a default `name === ""` placeholder unless its
                // own body declares a `static name` element (which set the real
                // value). Only overwrite the placeholder — never an explicit one.
                && !self.class_declares_static_name(cid)
                && !self.class_has_own_name_element(handle)
            {
                let name_v = self.new_str(name);
                self.realm.clear_readonly_property(handle, "name");
                self.realm.set_property(handle, "name", name_v);
                self.realm.mark_hidden(handle, "name");
                self.realm.set_readonly_property(handle, "name");
            }
        }
    }

    /// Whether the class constructor object *already carries* a `name` supplied by
    /// one of its own elements — the runtime counterpart of
    /// [`class_declares_static_name`](Self::class_declares_static_name), which sees
    /// only literal keys. A **computed** `static [k]() {}` with `k === "name"`
    /// installs the method before NamedEvaluation runs, and must not be clobbered
    /// by it. An anonymous class's untouched placeholder is exactly the empty
    /// string (or an accessor, which is always a real element).
    fn class_has_own_name_element(&self, handle: crate::heap::Handle) -> bool {
        if self.realm.accessor(handle, "name").is_some() {
            return true;
        }
        match self.realm.get_property(handle, "name") {
            Some(v) => !v
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|nh| self.realm.string_value(nh))
                .is_some_and(|s| s.is_empty()),
            None => false,
        }
    }

    /// Whether class `cid`'s body declares a `static name` member (a method,
    /// accessor, or field with the literal key `name`) — which supplies the
    /// constructor's `name` own property and therefore blocks NamedEvaluation
    /// from overwriting it. A computed `static [x]` key is not statically known,
    /// so it is conservatively ignored here (see
    /// [`class_has_own_name_element`](Self::class_has_own_name_element), which
    /// catches it at runtime).
    fn class_declares_static_name(&self, cid: u32) -> bool {
        self.classes[cid as usize].body.iter().any(|m| {
            let (is_static, key) = match m {
                crate::ast::ClassMember::Method(mm) => (mm.is_static, &mm.key),
                crate::ast::ClassMember::Field(f) => (f.is_static, &f.key),
                crate::ast::ClassMember::StaticBlock { .. } => return false,
            };
            is_static
                && matches!(key, PropertyKey::Ident(s) | PropertyKey::Str(s) if &**s == "name")
        })
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
        // Only an index within the *dense* backing can be a present element; an
        // index in `[dense_len, length)` is a hole or a sparse aux-stored element,
        // so fall through (`None`) to the named `[[Get]]` (which consults the aux
        // property table, then the prototype chain).
        if i < self.realm.array_dense_len(handle).unwrap_or(0) {
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
                // `undefined`. The holder carries the brand as an OWN private element:
                // an instance field/method/accessor, or — for `Class.#static` — a
                // static private in the class's own statics. Static privates are
                // **not inherited**, so a subclass constructor (whose `[[Prototype]]`
                // is the base class) that lacks the own element throws even though a
                // plain `read_member` would walk up to the base's static private.
                let key = self.private_access_key(s);
                // A private element is an OWN internal slot, never inherited and
                // never routed through a proxy's `get` trap. A private accessor is
                // invoked directly; a private field/method is read raw from the
                // holder's own storage (its auxiliary object for an exotic holder
                // such as a Proxy), bypassing `read_member`'s prototype walk and
                // proxy traps.
                if let Some((getter, _)) = self.realm.accessor(handle, &key) {
                    // A private accessor declared with only a setter (`set #g(v) {}`)
                    // has no getter — reading it is a TypeError.
                    if matches!(getter.unpack(), Unpacked::Undefined) {
                        let m = self.new_str(&alloc::format!(
                            "Cannot read private member #{s} which has only a setter"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    return self.call_with_this(getter, NanBox::handle(handle.to_raw()), &[]);
                }
                if !self.realm.has_own(handle, &key) {
                    let m = self.new_str(&alloc::format!(
                        "Cannot read private member #{s} from an object whose class did not declare it"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                Ok(self
                    .realm
                    .get_property(handle, &key)
                    .unwrap_or_else(NanBox::undefined))
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

    /// `{ ...src }` — copy `src`'s own enumerable properties onto `target`
    /// (CopyDataProperties). Spreading an array/string copies its indexed elements
    /// as `"0"`, `"1"`, … properties; any other object copies its own enumerable
    /// string + symbol keys (invoking getters); a primitive is a no-op. Shared by
    /// the object-literal evaluator and the generator step-machine.
    pub(crate) fn object_spread_into(
        &mut self,
        target: crate::heap::Handle,
        src: NanBox,
    ) -> Result<(), ExecError> {
        if let Some(sh) = src.as_handle().map(Handle::from_raw) {
            if let Some(elems) = self.realm.array_elements(sh).map(<[_]>::to_vec) {
                for (i, e) in elems.iter().enumerate() {
                    self.realm.set_property(target, &alloc::format!("{i}"), *e);
                }
            } else if let Some(s) = self.realm.string_value(sh) {
                for (i, c) in s.chars().enumerate() {
                    let cv = self.new_str(&alloc::string::String::from(c));
                    self.realm.set_property(target, &alloc::format!("{i}"), cv);
                }
            } else if self.realm.proxy_at(sh).is_some() {
                // A **proxy** source runs full CopyDataProperties through the
                // proxy protocol (`ownKeys` trap → per-key enumerable check →
                // `get` trap). The plain `object_keys_with_symbols` path below
                // reads the proxy *cell's* keys (none), which is why spread
                // otherwise saw `{}`.
                self.copy_data_properties(target, sh, &[])?;
            } else {
                let keys = self.realm.object_keys_with_symbols(sh);
                for key in keys {
                    // `read_member` invokes a getter where present.
                    let pv = self.read_member(sh, &key)?;
                    self.realm.set_property(target, &key, pv);
                }
            }
        }
        Ok(())
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
        if self.realm.proxy_at(handle).is_some() {
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
        if let Some((scope, param)) = self.arg_map_binding(handle, &name) {
            scope.set(&param, new);
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

    /// The legacy `fn.caller` value for an *ordinary, source-declared, non-strict*
    /// function: the function that is currently invoking `f` (its nearest live
    /// caller on the invocation stack), or `null` when there is none, when `f` is
    /// not executing, or when the caller is strict (a strict frame is never
    /// exposed). `None` for every other callable — bound, dynamically built,
    /// strict, generator, async, or arrow — so those keep the spec's poisoned
    /// `%ThrowTypeError%` accessor and throw.
    fn legacy_caller(&mut self, f: crate::heap::Handle) -> Option<NanBox> {
        if self.realm.get_property(f, BOUND_TARGET).is_some()
            || self.realm.get_property(f, DYN_FN_MARKER).is_some()
        {
            return None;
        }
        let (func_id, _) = self.realm.function_at(f)?;
        let def = &self.functions[func_id as usize];
        if def.is_strict || def.is_generator || def.is_async || def.is_arrow {
            return None;
        }
        let raw = f.to_raw();
        let Some(idx) = self
            .fn_stack
            .iter()
            .rposition(|v| v.as_handle() == Some(raw))
        else {
            return Some(NanBox::null());
        };
        let Some(caller) = self.fn_stack[..idx].last().copied() else {
            return Some(NanBox::null());
        };
        let caller_is_strict = caller
            .as_handle()
            .map(crate::heap::Handle::from_raw)
            .and_then(|h| self.realm.function_at(h))
            .is_some_and(|(id, _)| self.functions[id as usize].is_strict);
        Some(if caller_is_strict {
            NanBox::null()
        } else {
            caller
        })
    }

    /// `a + b` for two string values, throwing the spec `RangeError` when the
    /// result would exceed the maximum string length. Shares the O(1) rope
    /// concatenation `+` uses.
    fn concat_or_throw(&mut self, a: NanBox, b: NanBox) -> Result<NanBox, ExecError> {
        match self.realm.add_checked(a, b) {
            Some(v) => Ok(v),
            None => {
                let m = self.new_str("Invalid string length");
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
        if let Some((scope, param)) = self.arg_map_binding(handle, name) {
            let value = scope.get(&param).unwrap_or_else(NanBox::undefined);
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
        if self.realm.proxy_at(handle).is_some() {
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
        // A constructor function's `.prototype` (lazily created), so
        // `Fn.prototype.method = …` and prototype-chain inheritance work. Skip the
        // synthesis when an own `prototype` property is present — a *non-object*
        // assignment (`Fn.prototype = undefined`) can't enter the `fn_protos`
        // side-table (it holds Handles) and is stored as an own property, which
        // must be honored (else the synthesized default wrongly shadows it).
        if name == "prototype"
            && let Some((func_id, _)) = self.realm.function_at(handle)
            && !self.realm.has_own(handle, "prototype")
            // Only constructable functions have a `prototype`; a non-constructable
            // one (arrow / `async` / concise method / accessor) reads it as absent.
            // Constructable functions normally carry a materialized own property
            // (so `has_own` is true and this is skipped); this synthesis remains a
            // safety net for any constructable function created off the main path.
            && self.fn_has_prototype(func_id)
        {
            let proto = self.realm.function_prototype(func_id);
            return Ok(NanBox::handle(proto.to_raw()));
        }
        // A class's `.prototype` (lazily materialized with its instance
        // methods/accessors and a `constructor` back-link).
        if name == "prototype"
            && let Some((class_id, _)) = self.realm.class_at(handle)
            && !self.realm.has_own(handle, "prototype")
        {
            let proto = self.class_prototype(class_id, handle);
            return Ok(NanBox::handle(proto.to_raw()));
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
            && self.fn_meta_synthesizable(handle, "name")
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
            && self.fn_meta_synthesizable(handle, name)
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
        } else if self.realm.function_at(handle).is_some()
            || self.realm.native_at(handle).is_some()
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

    pub(crate) fn eval_assign(
        &mut self,
        op: AssignOp,
        target: &'a Expr,
        value: &'a Expr,
        paren_target: bool,
    ) -> Result<NanBox, ExecError> {
        // AnnexB "Runtime Errors for Function Call Assignment Targets": a direct
        // CallExpression LHS parses in sloppy code but is a runtime ReferenceError.
        // The call itself is evaluated (its side effects run), then the assignment
        // fails *before* the RHS is evaluated — matching the spec's ordering
        // (`f() = g()` calls `f`, never `g`). Strict mode rejected this at parse
        // time, so only sloppy `=`/`op=` reach here.
        if target.is_web_compat_call_target() {
            self.eval(target)?;
            let m = self.new_str("Invalid left-hand side in assignment");
            return Err(ExecError::Throw(
                self.make_error(N_REFERENCE_ERROR, Some(m)),
            ));
        }
        // Logical assignment (`&&=`/`||=`/`??=`) short-circuits: the right side
        // is evaluated and stored only when the current value warrants it.
        if matches!(
            op,
            AssignOp::AndAssign | AssignOp::OrAssign | AssignOp::NullishAssign
        ) {
            // A `super.x &&= …` / `super[k] ??= …` target: the SuperProperty
            // reference is evaluated once (GetThisBinding, then — for a computed
            // key — the key expression + GetSuperBase, captured before
            // ToPropertyKey), and GetValue/PutValue share it. The RHS runs only
            // when the short circuit does not take.
            if let Expr::Member {
                object, property, ..
            } = target
                && matches!(&**object, Expr::Super(_))
            {
                self.require_super_this()?;
                let (name, obj_base) = match property {
                    PropertyKey::Computed(key_expr) => {
                        let k = self.eval(key_expr)?;
                        let obj_base = self.object_super_base();
                        (self.coerce_property_key(k)?, obj_base)
                    }
                    _ => {
                        let name = self.eval_prop_key(property)?;
                        (name, self.object_super_base())
                    }
                };
                let current = match obj_base {
                    Some(Some(proto)) => self.read_super_member_object(proto, &name)?,
                    Some(None) => {
                        return Err(self.type_error("Cannot read property of null (super)"));
                    }
                    None => self.resolve_super_member(&name)?,
                };
                let assign = match op {
                    AssignOp::AndAssign => self.realm.truthy(current),
                    AssignOp::OrAssign => !self.realm.truthy(current),
                    _ => matches!(current.unpack(), Unpacked::Undefined | Unpacked::Null),
                };
                if !assign {
                    return Ok(current);
                }
                let rhs = self.eval(value)?;
                match obj_base {
                    Some(Some(proto)) => self.assign_super_member_object(proto, &name, rhs)?,
                    Some(None) => {
                        return Err(self.type_error("Cannot set property on null (super)"));
                    }
                    None => self.assign_super_member(&name, rhs)?,
                }
                return Ok(rhs);
            }
            // A computed-member target (non-super): evaluate base + key ONCE,
            // shared by the read and the write — this both avoids the key
            // double-eval (`obj[f()] &&= g()` calls `f()` once) and evaluates the
            // key even on a null base, before that base's GetValue TypeError
            // (`null[f()] &&= g()` runs `f()` first).
            if let Expr::Member {
                object,
                property: PropertyKey::Computed(key_expr),
                ..
            } = target
                && !matches!(&**object, Expr::Super(_))
            {
                let obj = self.eval(object)?;
                let mut key = self.eval(key_expr)?;
                let Some(raw) = obj.as_handle() else {
                    // A null/undefined base is a TypeError (the key was already
                    // evaluated); a number/boolean primitive reads `undefined` and
                    // its write is a sloppy no-op.
                    if matches!(obj.unpack(), Unpacked::Null | Unpacked::Undefined) {
                        return Err(self.type_error("Cannot read property of null or undefined"));
                    }
                    if matches!(op, AssignOp::AndAssign) {
                        return Ok(NanBox::undefined());
                    }
                    return self.eval(value);
                };
                let handle = crate::heap::Handle::from_raw(raw);
                // Coerce an object key to a primitive property key once (its
                // `toString` runs once, before the RHS), reused for read and write.
                if key.as_handle().is_some_and(|r| {
                    let h = crate::heap::Handle::from_raw(r);
                    self.realm.symbol_at(h).is_none() && !self.realm.is_string_handle(h)
                }) {
                    let pk = self.coerce_property_key(key)?;
                    key = self.new_str(&pk);
                }
                let current = self.read_member_value(handle, key)?;
                let assign = match op {
                    AssignOp::AndAssign => self.realm.truthy(current),
                    AssignOp::OrAssign => !self.realm.truthy(current),
                    _ => matches!(current.unpack(), Unpacked::Undefined | Unpacked::Null),
                };
                if !assign {
                    return Ok(current);
                }
                let rhs = self.eval(value)?;
                self.assign_member_value(handle, key, rhs)?;
                return Ok(rhs);
            }
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
            // NamedEvaluation: `x &&= function(){}` / `x ||= () => {}` /
            // `x ??= class {}` names the anonymous RHS after the LHS *identifier*
            // (only a simple identifier target, only an anonymous fn/arrow/class).
            // …and only for a bare `IdentifierReference`: `IsIdentifierRef` of a
            // parenthesized target is false, so `(a) ??= function(){}` leaves the
            // function anonymous.
            if !paren_target
                && let Expr::Ident(id) = target
                && matches!(value, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_))
            {
                self.set_fn_name(rhs, &id.name);
            }
            self.assign_to(target, rhs)?;
            return Ok(rhs);
        }
        // A `super.x` / `super[expr]` target: MakeSuperPropertyReference performs
        // GetThisBinding, evaluates the key expression, and captures GetSuperBase —
        // all *before* the RHS. So `super[x] = (() => (x = "other", 0))()` writes
        // the original key, and `super["p"] = ruin()` still writes through the base
        // captured before `ruin()` re-pointed the home object's prototype.
        if let Expr::Member {
            object, property, ..
        } = target
            && matches!(&**object, Expr::Super(_))
        {
            self.require_super_this()?;
            // The key *expression* is evaluated to a value here; `ToPropertyKey`
            // is part of GetValue/PutValue and therefore runs after the RHS for a
            // plain assignment (and after GetSuperBase in every case).
            let key_val = match property {
                PropertyKey::Computed(key_expr) => Some(self.eval(key_expr)?),
                _ => None,
            };
            let base = self.super_base()?;
            let to_key = |this: &mut Self| -> Result<String, ExecError> {
                match key_val {
                    Some(k) => this.coerce_property_key(k),
                    None => this.eval_prop_key(property),
                }
            };
            let (name, new) = if op == AssignOp::Assign {
                let rhs = self.eval(value)?;
                let name = to_key(self)?;
                (name, rhs)
            } else {
                // A compound assignment's `GetValue` on the super reference runs
                // before the RHS (and needs a non-null base); its `ToPropertyKey`
                // is part of that GetValue.
                let name = to_key(self)?;
                let Some(b) = base else {
                    return Err(self.type_error("Cannot read property of null (super)"));
                };
                let current = self.read_super_member_object(b, &name)?;
                let rhs = self.eval(value)?;
                (name, self.binary(compound_op(op)?, current, rhs)?)
            };
            let Some(b) = base else {
                return Err(self.type_error("Cannot set property on null (super)"));
            };
            self.assign_super_member_object(b, &name, new)?;
            return Ok(new);
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
                // A number/boolean primitive base: PutValue boxes it and performs
                // `[[Set]](key, v, primitiveReceiver)` — an inherited setter runs,
                // otherwise creating an own property on a non-object receiver fails
                // (a strict TypeError, a sloppy no-op). This is the same rule the
                // static-member path applies via `write_primitive_member`; skipping
                // it here made `(function(){"use strict"; true[k] = 1})()` silent.
                let k = self.coerce_property_key(key)?;
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let boxed = self.coerce_to_object(obj);
                    let current = match boxed.as_handle() {
                        Some(br) => self.read_member(crate::heap::Handle::from_raw(br), &k)?,
                        None => NanBox::undefined(),
                    };
                    self.binary(compound_op(op)?, current, rhs)?
                };
                self.write_primitive_member_key(obj, &k, new)?;
                return Ok(new);
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
                    self.realm.symbol_at(h).is_none() && !self.realm.is_string_handle(h)
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
            // GetThisBinding precedes evaluating the key expression: a derived
            // constructor before `super()` throws ReferenceError here, never running
            // the key or the RHS.
            self.require_super_this()?;
            // Evaluate the key *expression* first; for a plain assignment the RHS
            // is evaluated before the key is ToPropertyKey-coerced, so
            // `super[obj] = rhs()` runs `rhs` before `obj.toString` (the spec
            // defers a super reference's key coercion past the RHS). A compound op
            // must read `super[key]` first, so it coerces the key up front.
            let k = self.eval(key_expr)?;
            // For an object-literal method, GetSuperBase is captured now — before
            // ToPropertyKey — so a key whose `toString` mutates the home object's
            // prototype still targets the original base for both read and write.
            let obj_base = self.object_super_base();
            let (name, new) = if op == AssignOp::Assign {
                let rhs = self.eval(value)?;
                (self.coerce_property_key(k)?, rhs)
            } else {
                let name = self.coerce_property_key(k)?;
                let current = match obj_base {
                    Some(Some(proto)) => self.read_super_member_object(proto, &name)?,
                    Some(None) => {
                        return Err(self.type_error("Cannot read property of null (super)"));
                    }
                    None => self.resolve_super_member(&name)?,
                };
                let rhs = self.eval(value)?;
                (name, self.binary(compound_op(op)?, current, rhs)?)
            };
            match obj_base {
                Some(Some(proto)) => self.assign_super_member_object(proto, &name, new)?,
                Some(None) => {
                    return Err(self.type_error("Cannot set property on null (super)"));
                }
                None => self.assign_super_member(&name, new)?,
            }
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
        // A *compound* assignment to a bare identifier follows spec reference
        // order: evaluate `lref` and read `lval = GetValue(lref)` *before* the
        // RHS, then `PutValue(lref, …)` using that same reference. Capturing the
        // binding's scope frame up front matters when the RHS has a side effect
        // that introduces a more-local binding of the same name — e.g. a direct
        // `eval("var x = …")` inside the RHS: the write must still target the
        // originally-resolved (outer) binding, and the new local only shows
        // through to *later* reads.
        if op != AssignOp::Assign
            && let Expr::Ident(id) = target
        {
            let name = &*id.name;
            // A `with`-object binding (captured before the RHS so the object's
            // current value is read first and a setter fires on write). Both
            // `GetBindingValue` (the read) and `SetMutableBinding` (the write)
            // re-run `? HasProperty` (a proxy `has` trap) after the `HasBinding`
            // resolution — a binding deleted mid-operation is a strict-mode
            // ReferenceError.
            if let Some(h) = self.with_binding_result(name)? {
                let current = if self.has_property_proxied(h, name)? {
                    self.read_member(h, name)?
                } else if self.strict {
                    let m = self.new_str(&alloc::format!("{name} is not defined"));
                    return Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(m)),
                    ));
                } else {
                    NanBox::undefined()
                };
                let rhs = self.eval(value)?;
                let new = self.binary(compound_op(op)?, current, rhs)?;
                if !self.has_property_proxied(h, name)? && self.strict {
                    let m = self.new_str(&alloc::format!("{name} is not defined"));
                    return Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(m)),
                    ));
                }
                let key = self.new_str(name);
                self.assign_member_value(h, key, new)?;
                return Ok(new);
            }
            // An imported binding is immutable (module code is strict): error
            // before running the RHS, matching the plain-assign path.
            #[cfg(all(feature = "module", feature = "std"))]
            if self.module_imports.contains_key(name) && self.current.get(name).is_none() {
                let m = self.new_str("Assignment to constant variable.");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            if self.current.is_const(name) {
                let m = self.new_str("Assignment to constant variable.");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // A named-function-expression name is a soft immutable binding: strict
            // reassignment throws, sloppy is a silent no-op (the expression still
            // evaluates to the RHS).
            if self.current.is_soft_const(name) {
                let rhs = self.eval(value)?;
                if self.strict {
                    let m = self.new_str("Assignment to constant variable.");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let current = self.read_ident_ref(name)?;
                return Ok(if op == AssignOp::Assign {
                    rhs
                } else {
                    self.binary(compound_op(op)?, current, rhs)?
                });
            }
            // Capture the declarative reference (owning scope frame) *now*.
            let frame = self.current.owner_frame(name);
            let current = self.read_ident_ref(name)?;
            let rhs = self.eval(value)?;
            let new = self.binary(compound_op(op)?, current, rhs)?;
            if let Some(fr) = frame {
                fr.declare(name, new);
                // Keep a global `var`'s global-object mirror in step.
                self.sync_global_var(name, new);
            } else if !self.current.set(name, new) {
                if self.strict {
                    let m = self.new_str(&alloc::format!("{name} is not defined"));
                    return Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(m)),
                    ));
                }
                self.declare_sloppy_global(name, new)?;
            }
            return Ok(new);
        }
        // For `name = class {}`, hand the LHS name to `make_class` so an anonymous
        // class's `name` is set before its static initializers run.
        if op == AssignOp::Assign
            && !paren_target
            && let Expr::Ident(id) = target
            && let Expr::Class(c) = value
            && c.id.is_none()
        {
            self.pending_class_name = Some(&id.name);
        }
        // PutValue resolves the LHS reference *before* the RHS runs. For a plain
        // `x = rhs` naming a bare identifier with no binding, capture whether `x`
        // is already an own global-object property NOW, so a strict assignment to
        // an unresolvable reference still throws even when the RHS creates that
        // property (`undeclared = (this.undeclared = 5)` must throw).
        let ident_pre_own_global = if let Expr::Ident(id) = target {
            self.global_this
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|g| self.realm.has_own(g, &id.name))
        } else {
            false
        };
        // `ObjectEnvironmentRecord.HasBinding` for the global object record is
        // `HasProperty(globalObj, N)` — the global object's **prototype chain**
        // counts too, so `Object.setPrototypeOf(globalThis, new Proxy(…, { has }))`
        // makes a bare name a *resolvable* reference (no strict ReferenceError) and
        // the write goes through that chain's `set` trap. Practically every program
        // leaves the global object's `[[Prototype]]` at `%Object.prototype%`, where
        // no engine treats an inherited name as a global binding, so the chain walk
        // is gated on that prototype having been replaced — the ordinary path keeps
        // the single own-property probe.
        let ident_pre_proto_global = if let Expr::Ident(id) = target
            && !ident_pre_own_global
            && let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
            && self.realm.object_proto(g) != self.realm.default_object_proto()
        {
            self.has_property_proxied(g, &id.name)?
        } else {
            false
        };
        // PutValue also resolves *which* binding the LHS names before the RHS runs.
        // Capture that base now — the `with`-object frame that provides the name, or
        // else the scope frame that owns it — so a RHS that mutates the binding
        // structure (a `with`-object `delete`, or a direct-eval `var` that creates a
        // shadowing local) cannot redirect the write to a different binding
        // (test262 assignment/S11.13.1_A5*/A6*). The `with` walk is gated on there
        // being a `with` scope at all, keeping the common case a single frame walk.
        let (ident_with_ref, ident_owner_scope) = if let Expr::Ident(id) = target {
            let with_ref = if self.in_with_scope() {
                self.with_binding_result(&id.name)?
            } else {
                None
            };
            let owner = if with_ref.is_none() {
                self.current.owner_frame(&id.name)
            } else {
                None
            };
            (with_ref, owner)
        } else {
            (None, None)
        };
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
                // setters/getters work). The providing object was resolved *before*
                // the RHS (`ident_with_ref`), per PutValue's reference order.
                if let Some(h) = ident_with_ref {
                    let new = if op == AssignOp::Assign {
                        // NamedEvaluation applies to *any* IdentifierReference target,
                        // an object-environment (`with`) one included:
                        // `with (o) { dynamic = function(){} }` names it "dynamic".
                        if !paren_target
                            && matches!(value, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_))
                        {
                            self.set_fn_name(rhs, name);
                        }
                        rhs
                    } else {
                        let current = self.read_member(h, name)?;
                        self.binary(compound_op(op)?, current, rhs)?
                    };
                    // `SetMutableBinding` re-checks `? HasProperty` (a second `has`
                    // trap) after `HasBinding`: a strict write to a now-missing
                    // binding is a ReferenceError; sloppy still `[[Set]]`s.
                    if !self.has_property_proxied(h, name)? && self.strict {
                        let m = self.new_str(&alloc::format!("{name} is not defined"));
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    }
                    let key = self.new_str(name);
                    self.assign_member_value(h, key, new)?;
                    return Ok(new);
                }
                // Assigning to a lexical binding still in its temporal dead zone
                // (a `let`/`const`/`class` referenced by a bare `x = v` before its
                // declaration executes) is a ReferenceError — PutValue on an
                // uninitialized binding throws, exactly as reading it does. The
                // compound path additionally reads first (read_ident_ref, which
                // also throws TDZ); this guards the plain `=` case where no read
                // occurs before the write.
                if self.current.get(name).is_some_and(|v| v.is_tdz()) {
                    let msg = self.new_str(&alloc::format!(
                        "Cannot access '{name}' before initialization"
                    ));
                    return Err(ExecError::Throw(
                        self.make_error(N_REFERENCE_ERROR, Some(msg)),
                    ));
                }
                // Reassigning a `const` binding is a TypeError.
                if self.current.is_const(name) {
                    let m = self.new_str("Assignment to constant variable.");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A named-function-expression name is a soft immutable binding:
                // strict reassignment throws, sloppy is a silent no-op (the
                // expression still evaluates to the RHS / compound result).
                if self.current.is_soft_const(name) {
                    if self.strict {
                        let m = self.new_str("Assignment to constant variable.");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    // NamedEvaluation happens while evaluating the *right-hand side*,
                    // before PutValue — so the function is named even though the write
                    // itself is a silent no-op (`namedLambda = function(){}` inside
                    // `namedLambda` yields a function named "namedLambda").
                    if !paren_target
                        && op == AssignOp::Assign
                        && matches!(value, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_))
                    {
                        self.set_fn_name(rhs, name);
                    }
                    return Ok(if op == AssignOp::Assign {
                        rhs
                    } else {
                        let current = self.read_ident_ref(name)?;
                        self.binary(compound_op(op)?, current, rhs)?
                    });
                }
                let new = if op == AssignOp::Assign {
                    // NamedEvaluation: `x = function(){}` / `x = () => {}` /
                    // `x = class {}` names the anonymous definition after the LHS
                    // identifier (only for a plain `=`, and only when the RHS is an
                    // anonymous function/arrow/class).
                    // …but only for a bare `IdentifierReference` LHS: `IsIdentifierRef`
                    // of a parenthesized target is false, so `(fn) = function(){}`
                    // leaves the function anonymous.
                    if !paren_target
                        && matches!(value, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_))
                    {
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
                // Write through the frame resolved *before* the RHS
                // (`ident_owner_scope`) so a direct-eval `var` created by the RHS
                // cannot capture the assignment; in the common case this is the same
                // frame `self.current.set` would find.
                let stored = match &ident_owner_scope {
                    Some(sc) => sc.set(name, new),
                    None => self.current.set(name, new),
                };
                if stored {
                    // A global `var` is also an own property of the global object;
                    // keep the mirror in step (`var b; b = 1` → `this.b === 1`).
                    self.sync_global_var(name, new);
                }
                if !stored {
                    // A property on the global object (created via `this.x = …` /
                    // `globalThis.x = …`, or a global `var`) is a *resolvable*
                    // reference — assignment updates it, in strict mode too. Only a
                    // truly unresolvable reference is a strict-mode ReferenceError.
                    // Mirrors the read path's global-object own-property fallback.
                    // Skipped inside a `with` scope (object-first resolution): a
                    // deleted `with` binding must still reach the strict throw.
                    // Uses the pre-RHS resolvability (`ident_pre_own_global`): the
                    // reference is resolved before the RHS, so a property the RHS
                    // itself creates does NOT make the reference resolvable
                    // (`undeclared = (this.undeclared = 5)` throws in strict). But
                    // SetMutableBinding for the global object env record also re-checks
                    // HasProperty at PutValue time: if the RHS deleted the property, a
                    // strict write throws ReferenceError while a sloppy write recreates
                    // it (`x = (delete global.x, 2)`).
                    if !self.in_with_scope()
                        && ident_pre_own_global
                        && let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
                        && (self.realm.has_own(g, name) || !self.strict)
                    {
                        // …unless the property is non-writable (`NaN`,
                        // `Infinity`, `undefined`): `[[Set]]` returns false, which
                        // `PutValue` turns into a TypeError for a strict
                        // reference and silently ignores for a sloppy one.
                        if self.realm.property_is_readonly(g, name) {
                            if self.strict {
                                let m = self.new_str(&alloc::format!(
                                    "Cannot assign to read only property '{name}'"
                                ));
                                return Err(ExecError::Throw(
                                    self.make_error(N_TYPE_ERROR, Some(m)),
                                ));
                            }
                            return Ok(new);
                        }
                        self.realm.set_property(g, name, new);
                        return Ok(new);
                    }
                    // Resolvable only through the global object's (replaced)
                    // prototype chain: `SetMutableBinding` is `Set(globalObj, N, V,
                    // strict)` — an OrdinarySet with the global object as the
                    // Receiver, so the inherited setter / proxy `set` trap runs and
                    // the value lands as an own property of the global object. The
                    // sloppy path reaches the same code via `declare_sloppy_global`.
                    if ident_pre_proto_global
                        && let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
                    {
                        let recv = NanBox::handle(g.to_raw());
                        let ok = self.proxy_set_bool(g, name, new, recv)?;
                        if !ok && self.strict {
                            let m = self.new_str(&alloc::format!(
                                "Cannot assign to read only property '{name}'"
                            ));
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        return Ok(new);
                    }
                    // A strict write whose global-object property the RHS deleted
                    // (or an unresolvable reference) falls through to the throw.
                    if self.strict {
                        let m = self.new_str(&alloc::format!("{name} is not defined"));
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    }
                    // Sloppy implicit global: bind on the global scope + object.
                    self.declare_sloppy_global(name, new)?;
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
                    // PrivateFieldSet step 2: a private write requires an object
                    // receiver — a primitive `this` (`method.call(15)` reaching
                    // `this.#p = …`) is a TypeError, never a silent no-op.
                    if let PropertyKey::Private(s) = property {
                        let m = self.new_str(&alloc::format!(
                            "Cannot write private member #{s} to a non-object"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    // Writing a property of a number/boolean primitive follows
                    // PutValue → ToObject → [[Set]] (see `write_primitive_member`):
                    // an inherited setter / prototype Proxy fires, else it is a
                    // strict-mode TypeError / sloppy no-op.
                    if matches!(obj.unpack(), Unpacked::Number(_) | Unpacked::Bool(_)) {
                        let new = if op == AssignOp::Assign {
                            rhs
                        } else {
                            let current = self.read_member_of(obj, property, false)?;
                            self.binary(compound_op(op)?, current, rhs)?
                        };
                        self.write_primitive_member(obj, property, new)?;
                        return Ok(new);
                    }
                    return Ok(rhs);
                };
                // A primitive base that is a *heap* value (a String / Symbol /
                // BigInt primitive) is ToObject'd before `[[Set]]`: an inherited
                // setter or a Proxy on the wrapper's prototype chain fires, else
                // the write would create an own data property on the non-object
                // primitive receiver — a strict-mode TypeError, a sloppy no-op
                // (see `write_primitive_member`). An ordinary object base writes
                // directly.
                if !self.is_object_value(obj) {
                    let new = if op == AssignOp::Assign {
                        rhs
                    } else {
                        let current = self.read_member_of(obj, property, false)?;
                        self.binary(compound_op(op)?, current, rhs)?
                    };
                    self.write_primitive_member(obj, property, new)?;
                    return Ok(new);
                }
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
        // `obj.__proto__ = proto` invokes the inherited `set __proto__` accessor
        // (Annex B), which performs `O.[[SetPrototypeOf]]` like
        // `Object.setPrototypeOf` — a non-object, non-null value is ignored, and a
        // failed set (non-extensible object, or a prototype cycle) throws a
        // TypeError. The magic only applies when the object actually inherits
        // `Object.prototype`'s accessor and has no own `__proto__` data property;
        // otherwise the write falls through to an ordinary property assignment.
        if let PropertyKey::Ident(s) | PropertyKey::Str(s) = property
            && &**s == "__proto__"
            && !self.realm.has_own(handle, "__proto__")
            && self.realm.inherits_object_proto(handle)
        {
            let proto = match new.unpack() {
                Unpacked::Null => Some(None),
                _ if self.is_object_value(new) => Some(new.as_handle().map(Handle::from_raw)),
                _ => None,
            };
            if let Some(p) = proto
                && !self.set_proto_of(handle, p)?
            {
                return Err(self.type_error(
                    "Object.prototype.__proto__: cannot set prototype of this object",
                ));
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
                } else if let PropertyKey::Private(s) = property {
                    // A getter-only *private* accessor always throws on set (there
                    // is no silent-failure path for private references).
                    let m = self.new_str(&alloc::format!(
                        "Cannot write private member #{s} which has only a getter"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A getter-only own (public) accessor: the write is silently
                // ignored (non-strict) — matching ordinary accessor semantics.
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
            // PrivateSet brand check on a class receiver: writing `this.#x` where
            // this class does not carry `#x` (a distinct per-class brand) is a
            // TypeError — e.g. `C1.access.call(C2)` writing `C1`'s static `#m` on an
            // unrelated class `C2`. An own accessor was already dispatched above, so
            // reaching here without an own key means the element is genuinely absent.
            if let PropertyKey::Private(s) = property {
                if !self.realm.has_own(handle, &key) {
                    let m = self.new_str(&alloc::format!(
                        "Cannot write private member #{s} to an object whose class did not declare it"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                // A static private *method* is a non-writable own key: PrivateSet
                // on it is a TypeError (methods and getter-only accessors can't be
                // assigned).
                if self.realm.property_is_readonly(handle, &key) {
                    let m = self.new_str(&alloc::format!(
                        "Cannot write to private method or accessor #{s}"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
            }
            // An accessor inherited through the *ordinary* `[[Prototype]]` chain
            // (not the class-static side tables) handles the write — most notably
            // the poisoned `Function.prototype.caller`/`.arguments` accessors that a
            // class constructor inherits. `set_through_proto_chain` invokes the
            // setter (a poisoned one throws) and stops; it returns `None` for a plain
            // inherited/absent data property, which falls through to the data write.
            if !self.realm.has_own(handle, &key)
                && let Some(()) = self.set_through_proto_chain(handle, &key, new)?
            {
                return Ok(());
            }
            // Plain own data static: update both the mirror (authoritative for
            // reflection/reads) and the side table (kept consistent for any
            // remaining side-table consumer).
            self.realm.set_property(handle, &key, new);
            self.class_statics[cid as usize].insert(key, new);
            return Ok(());
        }
        // Proxy `[[Set]]`: route through the receiver-aware `proxy_set_bool`
        // (shared with `Reflect.set` and `assign_member_value`), passing the proxy
        // itself as the Receiver. A trapless forward keeps the Receiver, so an
        // inherited accessor setter (e.g. `Object.prototype.__proto__`) runs with
        // `this` = the proxy and a nested proxy target re-enters its own trap. A
        // `false` result is a failed [[Set]]: strict code throws, sloppy is silent.
        //
        // A **private** reference is exempt: `PrivateSet` (7.3.32) operates on the
        // object's `[[PrivateElements]]` list directly and is never routed through
        // `[[Set]]`, so a proxy is fully transparent to it — the element lives on
        // the proxy object itself (that is where the field initializer stamped it),
        // never on the target, and no trap is consulted.
        if self.realm.proxy_at(handle).is_some() && !matches!(property, PropertyKey::Private(_)) {
            let key = self.eval_prop_key(property)?;
            let recv = NanBox::handle(handle.to_raw());
            let ok = self.proxy_set_bool(handle, &key, new, recv)?;
            if !ok && self.strict {
                let m = self.new_str(&alloc::format!(
                    "'set' on proxy: trap returned falsish for property '{key}'"
                ));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(());
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
                    // OrdinarySetWithOwnDescriptor: an *inherited* (`c != handle`)
                    // non-writable data property makes the whole [[Set]] fail — strict
                    // throws, sloppy drops — with no shadowing own property created on
                    // the receiver. The receiver's own property (`c == handle`) falls
                    // through to the normal write, which honors its own writability.
                    if c != handle && !self.can_write_property(c, &skey) {
                        if self.strict {
                            let m = self.new_str(&alloc::format!(
                                "Cannot assign to read only property '{skey}'"
                            ));
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        return Ok(());
                    }
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
                    // OrdinarySet: when the index has no own property, honor an
                    // inherited index setter reached through the prototype chain
                    // (e.g. `Array.prototype[1]`'s setter, for a `for (arr[1] in …)`
                    // / `[arr[1]] = …` target) before the dense write. Guarded like
                    // `assign_member_value` so a pristine `%Array.prototype%` chain
                    // keeps the fast path.
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
                    let name = self.coerce_property_key(k)?;
                    if self.allow_property_write(handle, &name)? {
                        self.realm.set_property(handle, &name, new);
                        self.sync_global_object_write(handle, &name, new);
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
                    && !self.realm.property_is_readonly(handle, "prototype")
                {
                    // `Fn.prototype = obj` reassigns the constructor's prototype.
                    // Keep the side table (drives the `new`/`Reflect.construct`
                    // read) and the materialized own data property in sync.
                    self.realm
                        .set_function_prototype(func_id, Handle::from_raw(praw));
                    self.realm.set_property(handle, "prototype", new);
                } else if self.allow_property_write(handle, s)? {
                    self.realm.set_property(handle, s, new);
                    self.sync_global_object_write(handle, s, new);
                }
            }
            PropertyKey::Number(n) => {
                // Canonical `ToString(Number)` so a non-canonical literal write
                // (`obj[0.0000001] = v`) keys identically to the read (`"1e-7"`).
                self.realm
                    .set_property(handle, &crate::realm::js_number_string(*n), new);
            }
            PropertyKey::Private(s) => {
                // Writing `obj.#x` where obj's class did not declare `#x` is a TypeError.
                // (Field initialization writes via `set_property` directly, not this path,
                // so the initial creation of a field is exempt; a class receiver, for
                // static privates, is resolved via separate per-class storage.)
                let key = self.private_access_key(s);
                // PrivateSet requires the receiver to actually carry the private
                // element — as a field (own data key) or an accessor. This holds for
                // *static* privates too: `C1.access.call(C2)` writing `this.#m` throws
                // a TypeError because `C2` lacks `C1`'s `#m` (its distinct per-class
                // brand). A found accessor was already invoked by the prototype-chain
                // walk above, so reaching here means the element is genuinely absent.
                // (First-time field creation writes via `set_property` directly, not
                // this path, so a fresh field is exempt.)
                if !self.realm.has_own(handle, &key) && self.realm.accessor(handle, &key).is_none()
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
            || self.realm.function_at(ch).is_some()
            || self.realm.class_at(ch).is_some()
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
                id if crate::nbexec::temporal::is_temporal_ctor_id(id) => {
                    // `x instanceof Temporal.<Type>` — a branded instance of that
                    // exact kind.
                    return Ok(self.realm.temporal_at(oh).map(|d| d.kind)
                        == crate::nbexec::temporal::kind_for_ctor_id(id));
                }
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
        // Plain function constructors: walk the instance's `[[Prototype]]` chain for
        // the constructor's current `.prototype` (so `Object.create(C.prototype)` is an
        // instance, and reassigning `C.prototype` is reflected). `Get(C,"prototype")`
        // must be an Object — otherwise OrdinaryHasInstance is a TypeError (e.g.
        // `C.prototype = undefined`).
        // A registered host constructor (`register_constructor`) walks the same
        // way: its instances have `[[Prototype]] = hostFn.prototype`.
        if self.realm.function_at(ch).is_some() || self.realm.host_fn_at(ch).is_some() {
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
        // User class RHS: OrdinaryHasInstance is the authoritative prototype-chain
        // walk against the class's `.prototype` — so an instance built with a
        // distinct `newTarget` (`Reflect.construct(C, args, D)`, whose instance has
        // `D.prototype` on its chain) is `instanceof D` and `instanceof C`. `Get(ch,
        // "prototype")` fires a proxy `get` trap; `get_proto_of` honors a
        // `getPrototypeOf` trap at each step.
        let proto_val = self.read_member(ch, "prototype")?;
        let proto = proto_val
            .as_handle()
            .map(Handle::from_raw)
            .filter(|_| self.is_object_value(proto_val));
        // A non-class callable RHS (e.g. `%Function.prototype%`) follows
        // OrdinaryHasInstance strictly: `Get(C, "prototype")` must be an Object,
        // otherwise it is a TypeError (`prototype-getter-with-primitive`).
        if self.realm.class_at(ch).is_none() {
            let Some(proto) = proto else {
                return Err(
                    self.type_error("Function has non-object prototype in instanceof check")
                );
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
            return Ok(false);
        }
        if let Some(proto) = proto {
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
        // Fallback: the instance's class-tag chain (its class, then each `extends`)
        // — robust when an instance's prototype was detached but its class identity
        // is still recorded.
        let (Some(tag), Some((target_id, _))) = (self.realm.class_tag(oh), self.realm.class_at(ch))
        else {
            return Ok(false);
        };
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
