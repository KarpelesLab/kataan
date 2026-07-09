use super::*;

impl<'a> Interp<'a> {
    /// The spec `name` of a method/accessor whose property key (already evaluated
    /// to its storage form) is `key`, given the accessor `kind`. A string/number
    /// key yields itself; a symbol key (stored as `"\0sym:<id>"`) yields
    /// `[description]`; getters/setters are prefixed with `get `/`set `. Returns
    /// `None` for a symbol whose description cannot be recovered (rare), so the
    /// caller leaves the name unset rather than storing the internal key.
    pub(crate) fn method_display_name(&self, key: &str, kind: MethodKind) -> Option<String> {
        let base = if let Some(rest) = key.strip_prefix("\u{0}sym:") {
            let id: u64 = rest.parse().ok()?;
            let h = self.realm.symbol_for_id(id)?;
            let (desc, _) = self.realm.symbol_at(h)?;
            // A symbol with no description (`Symbol()`) names the method `""`;
            // a described symbol names it `[description]`.
            if desc == SYMBOL_NO_DESC {
                String::new()
            } else {
                alloc::format!("[{desc}]")
            }
        } else if let Some(rest) = key.strip_prefix('\u{0}') {
            // A private element's internal storage key is `\0#name@<scope>`; its
            // spec `name` is the visible `#name` (the `@<scope>` suffix that ties
            // the key to its declaring class is dropped).
            let visible = rest.rsplit_once('@').map_or(rest, |(n, _)| n);
            String::from(visible)
        } else {
            String::from(key)
        };
        Some(match kind {
            MethodKind::Get => alloc::format!("get {base}"),
            MethodKind::Set => alloc::format!("set {base}"),
            _ => base,
        })
    }

    /// The spec `name` of a private method/accessor — `#name` (with `get `/`set `
    /// prefix for accessors). Returns `None` for a non-private key.
    pub(crate) fn private_method_display_name(
        &self,
        key: &PropertyKey,
        kind: MethodKind,
    ) -> Option<String> {
        let PropertyKey::Private(name) = key else {
            return None;
        };
        let base = alloc::format!("#{name}");
        Some(match kind {
            MethodKind::Get => alloc::format!("get {base}"),
            MethodKind::Set => alloc::format!("set {base}"),
            _ => base,
        })
    }

    /// Resolves a private reference `#name` at the current execution site to the
    /// id of the class that *declares* it: the nearest lexically-enclosing class
    /// (starting from the running method's home class) whose body declares
    /// `#name`. Returns `None` only outside any class (which the parser rejects
    /// for a real private reference) — callers then build a key that matches no
    /// stored private element, producing the spec TypeError.
    pub(crate) fn private_scope_id(&self, name: &str) -> Option<u32> {
        // Private names are lexically scoped: resolve from the running function's
        // lexical class (which, unlike `current_home`, survives nested ordinary
        // functions), falling back to `current_home` for any path that sets the
        // home but not the lexical home.
        let mut cur = self.current_lexical_home.or(self.current_home);
        while let Some(cid) = cur {
            if self.class_private_names[cid as usize].contains(name) {
                return Some(cid);
            }
            cur = self.class_lexical_parent[cid as usize];
        }
        None
    }

    /// The storage key for a private reference `#name` at the current execution
    /// site, resolving it to its lexically-enclosing declaring class. When the
    /// name does not resolve (no enclosing class declares it), returns a key
    /// guaranteed not to match any stored private element, so the brand check at
    /// the access site throws.
    pub(crate) fn private_access_key(&self, name: &str) -> String {
        match self.private_scope_id(name) {
            Some(scope) => crate::nbexec::private_storage_key(name, scope),
            // `u32::MAX` is never a real class id (ids are dense from 0), so this
            // key cannot collide with any declared private element.
            None => crate::nbexec::private_storage_key(name, u32::MAX),
        }
    }

    /// Evaluates a class member's property key to its storage key, given the
    /// class `cid` whose body *declares* the member. A private name (`#x`) keys
    /// on `cid` (the declaration site), not on the lexically-enclosing class of
    /// the surrounding code — so a static private or an installed private method
    /// is stored under the same key its in-class references resolve to.
    pub(crate) fn eval_member_key_for_class(
        &mut self,
        key: &'a PropertyKey,
        cid: u32,
    ) -> Result<String, ExecError> {
        if let PropertyKey::Private(s) = key {
            return Ok(crate::nbexec::private_storage_key(s, cid));
        }
        self.eval_prop_key(key)
    }

    /// The property key for the member at `class.body[idx]` of class `cid`. A
    /// *computed* key was pre-evaluated once at class definition (stored in
    /// `class_member_keys`), so it is read back here rather than re-evaluated
    /// (re-evaluation would repeat side effects); a static-string or private key
    /// is evaluated directly (deterministic). Used by the lazily-built prototype
    /// and the static-member installer so a computed key is evaluated exactly
    /// once, in source order, at definition time.
    pub(crate) fn class_member_key(
        &mut self,
        cid: u32,
        idx: usize,
        key: &'a PropertyKey,
    ) -> Result<String, ExecError> {
        if matches!(key, PropertyKey::Computed(_))
            && let Some(k) = self.class_member_keys[cid as usize].get(&idx)
        {
            return Ok(k.clone());
        }
        self.eval_member_key_for_class(key, cid)
    }

    /// Registers a class and allocates a class value capturing the current scope.
    pub(crate) fn make_class(&mut self, class: &'a Class) -> Result<NanBox, ExecError> {
        let class_id = self.classes.len() as u32;
        // The home class of the code evaluating this definition is this class's
        // *lexical parent* — captured now, before the static-member loop below may
        // set `current_home` to this class while building nested definitions.
        let lexical_parent = self.current_home;
        self.classes.push(class);
        // Reserve this class's per-id side-table slots *before* evaluating any
        // static member, because a computed key or static field initializer may
        // itself define a nested class — which would push its own slots and shift
        // the indices, leaving `class_statics[class_id]` (etc.) misaligned with
        // `classes[class_id]`. We fill the reserved slots in place below.
        let class_env = self.current.child();
        let handle = self.realm.new_class(class_id, class_env.clone());
        let class_val = NanBox::handle(handle.to_raw());
        self.class_member_keys
            .push(alloc::collections::BTreeMap::new());
        self.class_statics.push(alloc::collections::BTreeMap::new());
        self.class_static_fields.push(Vec::new());
        self.class_static_get
            .push(alloc::collections::BTreeMap::new());
        self.class_static_set
            .push(alloc::collections::BTreeMap::new());
        self.class_envs.push(class_env.clone());
        self.class_native_super.push(None);
        self.class_fn_super.push(None);
        self.class_super_id.push(None);
        self.class_handles.push(class_val);
        self.class_lexical_parent.push(lexical_parent);
        // Record the bare private names this class declares (`#x` → `x`), so a
        // private reference can be resolved to its declaring class.
        let mut private_names = alloc::collections::BTreeSet::new();
        for member in &class.body {
            let key = match member {
                ClassMember::Method(m) => &m.key,
                ClassMember::Field(field) => &field.key,
                ClassMember::StaticBlock { .. } => continue,
            };
            if let PropertyKey::Private(s) = key {
                private_names.insert(alloc::boxed::Box::<str>::from(&**s));
            }
        }
        self.class_private_names.push(private_names);
        // ClassDefinitionEvaluation: evaluate every *computed* member key once, in
        // source order, BEFORE installing any member — so an undeclared / throwing
        // key (`get [zzqq]() {}`, `[Symbol.foo]` where it's unresolvable) is a
        // class-definition-time error and side effects run exactly once. The
        // results are reused by the (otherwise lazy) prototype / private / static
        // builders below instead of re-evaluating the expression.
        {
            let saved = core::mem::replace(&mut self.current, class_env.clone());
            let r = (|| -> Result<(), ExecError> {
                for (idx, member) in class.body.iter().enumerate() {
                    let (key, is_static) = match member {
                        ClassMember::Method(m) => (&m.key, m.is_static),
                        ClassMember::Field(field) => (&field.key, field.is_static),
                        ClassMember::StaticBlock { .. } => continue,
                    };
                    if matches!(key, PropertyKey::Computed(_)) {
                        let k = self.eval_prop_key(key)?;
                        // A *static* class element whose computed key evaluates to
                        // "prototype" is a TypeError (the constructor's `prototype` is
                        // a non-configurable own property that a class element may not
                        // redefine). The literal `static prototype` form is rejected
                        // earlier, at parse time.
                        if is_static && k == "prototype" {
                            let m = self.new_str(
                                "Classes may not have a static property named 'prototype'",
                            );
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        self.class_member_keys[class_id as usize].insert(idx, k);
                    }
                }
                Ok(())
            })();
            self.current = saved;
            r?;
        }
        // Build the static members (`static foo() {}` / `static x = …`) with the
        // class scope current, so a static method captures `class_env` — the same
        // scope the class name is bound into below — and can therefore reference
        // the class by name (`static m() { return NamedExpr.other() }`), matching
        // instance methods (which are built over `class_env` in `class_prototype`).
        let saved_static_env = core::mem::replace(&mut self.current, class_env.clone());
        let mut statics = alloc::collections::BTreeMap::new();
        let mut static_fields = Vec::new();
        let mut static_getters = alloc::collections::BTreeMap::new();
        let mut static_setters = alloc::collections::BTreeMap::new();
        let static_build = (|| -> Result<(), ExecError> {
            for (idx, member) in class.body.iter().enumerate() {
                match member {
                    ClassMember::Method(m) if m.is_static && m.kind == MethodKind::Method => {
                        // The computed key was pre-evaluated in source order above; a
                        // throw from it already propagated (it does not silently skip
                        // the member).
                        let key = self.class_member_key(class_id, idx, &m.key)?;
                        // A static method's home is this class, entered statically, so
                        // `super.x` resolves against the superclass's static members.
                        let f = self.make_method(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            m.value.is_async,
                            m.value.is_generator,
                            Some(class_id),
                            true,
                        );
                        if let Some(n) = self.method_display_name(&key, MethodKind::Method) {
                            self.install_method_meta(f, &n, &m.value.params);
                        }
                        statics.insert(key, f);
                    }
                    ClassMember::Field(field) if field.is_static => {
                        // A static field is installed as an enumerable own key, but its
                        // initializer is evaluated *later* (in source order with static
                        // blocks, after the constructor object — with its name/methods —
                        // exists), with `this` = the class. Install a placeholder now.
                        let key = self.class_member_key(class_id, idx, &field.key)?;
                        if !static_fields.contains(&key) {
                            static_fields.push(key.clone());
                        }
                        // Install a placeholder, but do NOT clobber an earlier static
                        // *method*/accessor of the same name: per spec all methods are
                        // installed before any static field initializer runs, so a field
                        // initializer like `static g = this.g()` must still see the
                        // method `g` until the field's own initializer overwrites it.
                        statics.entry(key).or_insert(NanBox::undefined());
                    }
                    // `static get x() {}` / `static set x(v) {}` — accessors.
                    ClassMember::Method(m)
                        if m.is_static && matches!(m.kind, MethodKind::Get | MethodKind::Set) =>
                    {
                        let key = self.class_member_key(class_id, idx, &m.key)?;
                        let f = self.make_method(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            false,
                            false,
                            Some(class_id),
                            true,
                        );
                        if let Some(n) = self.method_display_name(&key, m.kind) {
                            self.install_method_meta(f, &n, &m.value.params);
                        }
                        if m.kind == MethodKind::Get {
                            static_getters.insert(key, f);
                        } else {
                            static_setters.insert(key, f);
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        })();
        self.current = saved_static_env;
        static_build?;
        // Fill the reserved side-table slots (created above, in place so nested
        // classes defined during member evaluation cannot shift the indices).
        self.class_statics[class_id as usize] = statics;
        self.class_static_fields[class_id as usize] = static_fields;
        self.class_static_get[class_id as usize] = static_getters;
        self.class_static_set[class_id as usize] = static_setters;
        // Record a native-constructor superclass (`extends Error`) or an ordinary
        // user-function superclass (`extends fn`), so construction, `super(...)`,
        // the prototype chain, and `instanceof` can reach it (neither has a class
        // id). A class superclass is handled via `resolve_super`'s class chain.
        let (native_super, fn_super) = if let Some(expr) = &class.super_class {
            // A class definition is strict code, so the heritage (`extends <expr>`)
            // is evaluated in strict mode — a function expression there is strict
            // (its `.caller`/`.arguments` are the poisoned accessors).
            let saved_heritage_strict = core::mem::replace(&mut self.strict, true);
            // The heritage is evaluated in the class scope (`class_env`), so the
            // inner class-name binding is in scope: a closure created here
            // (`class C extends (f = () => C, …)`) captures `class_env` and, when
            // called after the class is defined, resolves `C` to the class.
            let saved_heritage_env = core::mem::replace(&mut self.current, class_env.clone());
            let sval = self.eval(expr);
            self.current = saved_heritage_env;
            self.strict = saved_heritage_strict;
            let sval = sval?;
            // `extends null` makes a base-ish class with a null prototype; any other
            // non-object, or a non-constructor object (arrow/generator/async fn,
            // a plain object), is a TypeError (the superclass must be a constructor).
            if matches!(sval.unpack(), Unpacked::Null) {
                (None, None)
            } else {
                match sval.as_handle().map(|r| (sval, Handle::from_raw(r))) {
                    // `class D extends C {}` → `getPrototypeOf(D) === C` (the
                    // constructor inherits the superclass's static members).
                    Some((_, h)) if self.realm.class_at(h).is_some() => {
                        self.realm.set_native_proto(handle, h);
                        // Cache the resolved super class id so the eager prototype
                        // build does not re-evaluate the `extends` expression.
                        self.class_super_id[class_id as usize] =
                            self.realm.class_at(h).map(|(pid, _)| pid);
                        (None, None)
                    }
                    // A native constructor superclass: a callable native (Map, Set,
                    // Date, RegExp, typed arrays, wrappers, ArrayBuffer, DataView,
                    // Error, Promise, …) *or* a namespace-object constructor
                    // (`Array`/`Object`, recognized by identity). `getPrototypeOf(D)`
                    // is the superclass; the native base id drives `super()` and the
                    // instance's internal slots.
                    Some((_, h)) if self.native_base_kind(h).is_some() => {
                        self.realm.set_native_proto(handle, h);
                        (self.native_base_kind(h), None)
                    }
                    // A callable ordinary function used as a superclass — only if it
                    // is actually a constructor (its prototype is linked below).
                    Some((v, h)) if self.is_callable(h) && self.is_constructor_value(v) => {
                        (None, Some(v))
                    }
                    _ => {
                        let m = self.new_str("Class extends value is not a constructor or null");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                }
            }
        } else {
            (None, None)
        };
        self.class_native_super[class_id as usize] = native_super;
        self.class_fn_super[class_id as usize] = fn_super;
        // `class D extends fn {}` makes `Object.getPrototypeOf(D) === fn` (the
        // constructor inherits static members from its function superclass).
        if let Some(fnp) = fn_super
            && let Some(sh) = fnp.as_handle().map(Handle::from_raw)
        {
            self.realm.set_native_proto(handle, sh);
        }
        // Install the constructor's own `length` (its declared param count up to
        // the first default/rest; 0 with no explicit constructor) and, for a named
        // class, its own `name` — both `{ w:false, e:false, c:true }` per spec.
        let ctor_len = class
            .body
            .iter()
            .find_map(|m| match m {
                ClassMember::Method(m) if m.kind == MethodKind::Constructor => Some(
                    m.value
                        .params
                        .iter()
                        .take_while(|p| p.default.is_none() && !p.rest)
                        .count() as u32,
                ),
                _ => None,
            })
            .unwrap_or(0);
        // An anonymous class expression undergoing NamedEvaluation (`var C = class
        // {}`) receives the binding name *now*, before static initializers run (so
        // `static x = this.name` sees it). A declared id always wins.
        let pending_name = self.pending_class_name.take();
        let class_name = class
            .id
            .as_ref()
            .map(|id| String::from(&*id.name))
            .or_else(|| pending_name.map(String::from));
        // Every class constructor has an own `name` (a bare anonymous `class {}`
        // keeps `name === ""`, own/non-writable/non-enumerable/configurable). A
        // later NamedEvaluation (`x = class {}`) overwrites this default placeholder
        // via `set_fn_name`, which recognizes it because the body declares no
        // `static name` member (see `set_fn_name`).
        self.install_fn_name_length(handle, class_name.as_deref().unwrap_or(""), ctor_len);
        // Mirror static members as real own properties of the constructor so
        // reflection (`hasOwnProperty`, `getOwnPropertyDescriptor`, `Object.keys`,
        // `verifyProperty`) sees them. The side tables above still drive the fast
        // read path and `super`-static resolution. Static methods are
        // `{ w:true, e:false, c:true }`; static fields are `{ w:true, e:true,
        // c:true }`; accessors install a getter/setter pair (`e:false, c:true`).
        let static_keys: Vec<(String, NanBox)> = self.class_statics[class_id as usize]
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let field_keys: alloc::collections::BTreeSet<String> = self.class_static_fields
            [class_id as usize]
            .iter()
            .cloned()
            .collect();
        for (k, v) in static_keys {
            // An explicit static member named `name`/`length` *overrides* the
            // default constructor `name`/`length` installed above (e.g.
            // `class { static name() {} }` makes `C.name` that method, writable
            // and non-enumerable — not the read-only default). Replace the
            // placeholder so it carries the member's own attributes.
            let is_name_len = k == "name" || k == "length";
            if is_name_len {
                self.realm.delete_property(handle, &k);
            }
            self.realm.set_property(handle, &k, v);
            if !field_keys.contains(&k) {
                // A static method is non-enumerable; a static field is enumerable.
                self.realm.mark_hidden(handle, &k);
                // A static *private method* (`\0#…` key that is not a field) is
                // non-writable, so `this.#m = v` is a TypeError (PrivateSet on a
                // method). Public static methods stay writable.
                if k.starts_with("\u{0}#") {
                    self.realm.set_readonly_property(handle, &k);
                }
            }
        }
        let getters: Vec<(String, NanBox)> = self.class_static_get[class_id as usize]
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let setters: alloc::collections::BTreeMap<String, NanBox> =
            self.class_static_set[class_id as usize].clone();
        for (k, getter) in getters {
            let setter = setters.get(&k).copied().unwrap_or(NanBox::undefined());
            // A static accessor named `name`/`length` overrides the default data
            // property installed above; drop the data slot so the getter wins.
            if k == "name" || k == "length" {
                self.realm.delete_data_slot(handle, &k);
            }
            self.realm.define_accessor(handle, &k, getter, setter);
            self.realm.mark_hidden(handle, &k);
        }
        for (k, setter) in &setters {
            if !self.class_static_get[class_id as usize].contains_key(k) {
                if k == "name" || k == "length" {
                    self.realm.delete_data_slot(handle, k);
                }
                self.realm
                    .define_accessor(handle, k, NanBox::undefined(), *setter);
                self.realm.mark_hidden(handle, k);
            }
        }
        // Materialize the class's own `prototype` data property — per
        // MakeClassConstructor it is `{ writable: false, enumerable: false,
        // configurable: false }`. Building it now (before static initializers)
        // installs the instance methods and the `constructor` back-link, and lets
        // a `static x = C.prototype` initializer observe it as an own property.
        let class_proto = self.class_prototype(class_id, handle);
        self.install_fn_prototype(handle, class_proto, false);
        // Bind the class's own name in its methods' scope (a named class
        // expression sees itself). The inner binding is an immutable `const`, so
        // reassigning it inside the class body (`class C { m() { C = 1; } }`) is a
        // TypeError (the outer declaration binding, if any, stays mutable).
        if let Some(id) = &class.id {
            class_env.declare_const(&id.name, class_val);
        }
        // Run static initialization — `static field = …` initializers and
        // `static { … }` blocks — in source order, *after* the constructor object
        // (with its name and methods) exists, with `this` = the class and the class
        // name bound (so an initializer/block can reference the class, its name,
        // and its other statics). Class bodies are strict code.
        let has_static_init = class.body.iter().any(|m| {
            matches!(m, ClassMember::StaticBlock { .. })
                || matches!(m, ClassMember::Field(f) if f.is_static && f.value.is_some())
        });
        if has_static_init {
            let scope = self.current.child();
            if let Some(id) = &class.id {
                // The class-name binding a static initializer/block sees is the
                // same immutable inner `const`.
                scope.declare_const(&id.name, class_val);
            }
            let saved = core::mem::replace(&mut self.current, scope);
            let saved_this = core::mem::replace(&mut self.this_val, class_val);
            let saved_strict = core::mem::replace(&mut self.strict, true);
            // Static initializers and static blocks run with the class as their
            // home object so `super.x` resolves against the superclass's static
            // side (`[[HomeObject]]` is the constructor, `IsStatic` is true).
            let saved_home = self.current_home.replace(class_id);
            let saved_lexical_home = self.current_lexical_home.replace(class_id);
            let saved_home_static = core::mem::replace(&mut self.current_home_static, true);
            let saved_home_obj = core::mem::take(&mut self.current_home_object);
            // A static block / static field initializer is function code with a
            // `[[NewTarget]]` of undefined: `new.target` inside (e.g. via a direct
            // eval) is a valid token that evaluates to undefined.
            let saved_nt_scope = core::mem::replace(&mut self.new_target_in_scope, true);
            let saved_target = core::mem::replace(&mut self.new_target, NanBox::undefined());
            let r = (|| {
                for (idx, member) in class.body.iter().enumerate() {
                    match member {
                        ClassMember::StaticBlock { body, .. } => {
                            for stmt in body {
                                self.exec(stmt)?;
                            }
                        }
                        ClassMember::Field(field) if field.is_static => {
                            let key = self.class_member_key(class_id, idx, &field.key)?;
                            let v = match &field.value {
                                Some(e) => self.eval(e)?,
                                None => continue,
                            };
                            // DefineField step 7: an anonymous function/arrow/class
                            // static-field initializer takes the field name.
                            if let Some(init) = &field.value
                                && matches!(
                                    init,
                                    Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_)
                                )
                                && let Some(disp) =
                                    self.method_display_name(&key, MethodKind::Method)
                            {
                                self.set_fn_name_owned(v, &disp);
                            }
                            if let Some(h) = class_val.as_handle().map(Handle::from_raw) {
                                self.realm.set_property(h, &key, v);
                            }
                        }
                        _ => {}
                    }
                }
                Ok(())
            })();
            self.current = saved;
            self.this_val = saved_this;
            self.strict = saved_strict;
            self.current_home = saved_home;
            self.current_lexical_home = saved_lexical_home;
            self.current_home_static = saved_home_static;
            self.current_home_object = saved_home_obj;
            self.new_target_in_scope = saved_nt_scope;
            self.new_target = saved_target;
            r?;
        }
        Ok(class_val)
    }

    /// Resolves a class's `extends` superclass to `(class_id, env)`, if any.
    pub(crate) fn resolve_super(
        &mut self,
        class: &'a Class,
        env: &Scope,
    ) -> Result<Option<(u32, Scope)>, ExecError> {
        let Some(expr) = &class.super_class else {
            return Ok(None);
        };
        let saved = core::mem::replace(&mut self.current, env.clone());
        let value = self.eval(expr);
        self.current = saved;
        let resolved = value?;
        // `extends null` is valid (a base class with a null prototype).
        if matches!(resolved.unpack(), Unpacked::Null) {
            return Ok(None);
        }
        let raw = resolved
            .as_handle()
            .ok_or(ExecError::Unsupported("extends a non-class"))?;
        let h = Handle::from_raw(raw);
        if let Some(parent) = self.realm.class_at(h) {
            Ok(Some(parent))
        } else if self.native_base_kind(h).is_some() || self.is_callable(h) {
            // A native superclass (`extends Error|Map|Array|…`) or an ordinary-
            // function superclass (`extends fn`) has no class chain; both are
            // tracked separately (`class_native_super` / `class_fn_super`). The
            // namespace-object constructors (`Array`/`Object`) are recognized by
            // `native_base_kind` (they are not callable cells / native ids).
            Ok(None)
        } else {
            Err(ExecError::Unsupported("extends a non-class"))
        }
    }

    /// Instantiates `new Class(args)`: creates the object, installs the methods
    /// of the whole `extends` chain (derived overriding base), then runs the
    /// constructor (with `super(...)` reaching the base).
    /// `GetPrototypeFromConstructor(newTarget, ...)` for a class instance: when the
    /// pending `new.target` differs from the class being constructed (an explicit
    /// `Reflect.construct(C, args, D)` or a proxy forwarding a distinct newTarget),
    /// the instance's `[[Prototype]]` is `D.[[Get]]("prototype")` when that is an
    /// Object. Returns `None` to fall back to the class's own prototype (a same
    /// newTarget, or a non-object `prototype`).
    fn new_target_instance_proto(
        &mut self,
        class_handle: NanBox,
    ) -> Result<Option<Handle>, ExecError> {
        let Some(nt) = self.pending_new_target else {
            return Ok(None);
        };
        let Some(nt_h) = nt.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        if class_handle.as_handle().map(Handle::from_raw) == Some(nt_h) {
            return Ok(None);
        }
        let proto = self.read_member(nt_h, "prototype")?;
        Ok(proto
            .as_handle()
            .map(Handle::from_raw)
            .filter(|_| self.is_object_value(proto)))
    }

    pub(crate) fn instantiate(
        &mut self,
        class_id: u32,
        env: &Scope,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        // Build the chain (this class, then each `extends`) up front: it both
        // drives private-method installation below and lets us find the root native
        // base — the deepest class's native superclass, if any.
        let mut chain: Vec<(u32, Scope)> = Vec::new();
        let mut cur = Some((class_id, env.clone()));
        while let Some((cid, cenv)) = cur {
            chain.push((cid, cenv.clone()));
            cur = self.resolve_super(self.classes[cid as usize], &cenv)?;
        }
        // The class prototype carries the public methods/accessors of the whole
        // `extends` chain — the instance's `[[Prototype]]`, so methods are
        // *inherited* (`instance.m === C.prototype.m`) and the chain resolves
        // derived-over-base.
        let proto = self.class_prototype_by_id(class_id);

        // A class extending a *cell-bearing* native (Map/Set/typed array/Date/
        // RegExp/wrapper/ArrayBuffer/DataView/Array) must produce a real native
        // instance (with the native internal slots), not a plain object. The base
        // native constructor builds it with `newTarget` = this class, so its
        // `[[Prototype]]` is the class prototype and the existing per-id
        // construction (storage, seeding, length, …) runs exactly once. (The
        // implicit/explicit `super(...)` for such a class is then a no-op — the
        // cell already exists; see `run_constructor`.)
        let class_handle = self.class_handles[class_id as usize];
        let native_root = chain
            .iter()
            .find_map(|(cid, _)| self.class_native_super[*cid as usize])
            .filter(|id| Self::native_base_is_cell(*id));
        let instance = if let Some(root_id) = native_root {
            // The base constructor links the cell to `class_handle.prototype` (which
            // is `proto`, the class prototype) via the `newTarget` path.
            self.construct_native_base(root_id, args, class_handle)?
        } else {
            let obj = self.realm.new_object();
            // GetPrototypeFromConstructor(newTarget, "%Object.prototype%"): the
            // instance's [[Prototype]] is `newTarget.prototype` when it is an
            // object. For a plain `new C()` newTarget is `C` (so this is the class
            // prototype), but `Reflect.construct(C, args, D)` — or a proxy whose
            // target is `C` with a distinct newTarget — supplies `D.prototype`.
            let inst_proto = self
                .new_target_instance_proto(class_handle)?
                .unwrap_or(proto);
            self.realm.set_object_proto(obj, Some(inst_proto));
            obj
        };
        let this_val = NanBox::handle(instance.to_raw());

        // NOTE: private methods/accessors are NOT installed here. Per spec they are
        // added by `InitializeInstanceElements` — after each class's `super()`
        // returns — so `install_private_methods` runs per-class from
        // `init_instance_fields`, keeping a private method unreachable while a base
        // constructor is still executing.

        self.realm.set_class_tag(instance, class_id);
        let saved_this = core::mem::replace(&mut self.this_val, this_val);
        // `new.target` (the class reached via `new`, passed through the one-shot)
        // holds for the whole constructor, incl. a base reached via `super(...)`.
        let nt = self
            .pending_new_target
            .take()
            .unwrap_or(NanBox::undefined());
        let saved_target = core::mem::replace(&mut self.new_target, nt);
        // A constructor body (and its field initializers) is function code, so a
        // direct `eval` inside it may use `new.target`. Construction does not flow
        // through `invoke`, so mark `new.target` as lexically in scope here.
        let saved_nt_scope = core::mem::replace(&mut self.new_target_in_scope, true);
        let result = self.run_constructor(class_id, env, instance, args);
        self.new_target_in_scope = saved_nt_scope;
        self.this_val = saved_this;
        self.new_target = saved_target;
        let ret = result?;
        // A constructor that `return`s an *object* makes `new` yield that object
        // instead of the freshly-built instance.
        let returned_object = match ret {
            Some(v) => v.as_handle().map(Handle::from_raw).is_some_and(|h| {
                self.realm.string_value(h).is_none()
                    && self.realm.bigint_at(h).is_none()
                    && self.realm.symbol_at(h).is_none()
            }),
            None => false,
        };
        if returned_object {
            return Ok(ret.unwrap());
        }
        // For a *derived* class, a constructor returning a non-`undefined` value
        // that is not an Object is a TypeError (ECMA-262 — derived constructors
        // must return an Object or undefined). A *base* class ignores a
        // primitive return and yields the new instance.
        let is_derived = self.classes[class_id as usize].super_class.is_some();
        if is_derived
            && let Some(v) = ret
            && !matches!(v.unpack(), Unpacked::Undefined)
        {
            let m = self.new_str("Derived constructors may only return object or undefined");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(this_val)
    }

    /// Materializes (and caches) the `.prototype` object for a class, populated
    /// with the class's instance methods/accessors over the whole `extends` chain
    /// (derived overriding base), a non-enumerable `constructor` back-link to the
    /// class handle, and a prototype link to the superclass's `.prototype` so that
    /// `Object.getPrototypeOf(C.prototype) === Base.prototype`.
    ///
    /// This mirrors the per-instance method installation in [`Self::instantiate`];
    /// the engine copies methods directly onto instances at `new` time, but a real
    /// prototype object is still required for `C.prototype.m`, `C.prototype[k]`,
    /// accessor reads, and prototype-chain reflection.
    /// The global constructor handle for a native built-in id (an `extends`-able
    /// native superclass: the Error family, `Iterator`, …). The Error family
    /// resolves by `ERROR_NAMES`; any other native is found by scanning the global
    /// bindings for the callable whose native id matches.
    pub(crate) fn native_ctor_by_id(&mut self, id: u16) -> Option<Handle> {
        // Temporal constructors live on the `Temporal` namespace object, not as
        // global bindings, so `class X extends Temporal.PlainDate {}` resolves the
        // base through `Temporal[<Type>]`.
        if let Some(kind) = crate::nbexec::temporal::kind_for_ctor_id(id) {
            return self
                .current
                .get("Temporal")
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                .and_then(|t| self.realm.get_property(t, kind.type_name()))
                .and_then(|c| c.as_handle())
                .map(Handle::from_raw);
        }
        if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
            let name = ERROR_NAMES[(id - N_ERROR_BASE) as usize];
            return self
                .current
                .get(name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
        }
        // The namespace-object constructors (`Array`/`Object`) carry sentinel base
        // ids — resolve them by their global binding.
        let sentinel = match id {
            N_BASE_ARRAY => Some("Array"),
            N_BASE_OBJECT => Some("Object"),
            _ => None,
        };
        if let Some(name) = sentinel {
            return self
                .current
                .get(name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
        }
        // A typed-array kind resolves by its concrete constructor name.
        if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id) {
            let name = TYPED_ARRAY_KINDS[(id - N_TYPED_ARRAY_BASE) as usize].0;
            return self
                .current
                .get(name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
        }
        // Other single-name natives: resolve by scanning the well-known global
        // bindings for the callable whose native id matches.
        for name in [
            "Iterator",
            "Map",
            "Set",
            "WeakMap",
            "WeakSet",
            "WeakRef",
            "FinalizationRegistry",
            "Date",
            "RegExp",
            "Number",
            "String",
            "Boolean",
            "ArrayBuffer",
            "SharedArrayBuffer",
            "DataView",
            "Promise",
            "DisposableStack",
            "AsyncDisposableStack",
            "ShadowRealm",
            "SuppressedError",
        ] {
            if let Some(h) = self
                .current
                .get(name)
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw)
                && self.realm.native_at(h) == Some(id)
            {
                return Some(h);
            }
        }
        // `class M extends Intl.NumberFormat {}` resolves the base by id (the Intl
        // constructors live under the `Intl` namespace, not the global scope) so
        // `M.prototype` links to `Intl.NumberFormat.prototype`. Only NumberFormat /
        // DateTimeFormat are resolved here — the ones whose `super()` init is wired
        // in `apply_native_super`; the other Intl services would link a prototype
        // without initializing their internal slots.
        let intl_name = match id {
            N_INTL_NUMBER_FORMAT => "NumberFormat",
            N_INTL_DATETIME_FORMAT => "DateTimeFormat",
            N_INTL_PLURAL_RULES => "PluralRules",
            N_INTL_LIST_FORMAT => "ListFormat",
            N_INTL_REL_TIME => "RelativeTimeFormat",
            N_INTL_SEGMENTER => "Segmenter",
            N_INTL_LOCALE => "Locale",
            _ => return None,
        };
        self.intl_ctor_handle(intl_name)
    }

    /// The "native base kind" for a superclass `handle` used as `extends` heritage:
    /// the native id for a callable native constructor (Map/Set/Date/RegExp/typed
    /// arrays/wrappers/ArrayBuffer/DataView/Error/…), or the `N_BASE_ARRAY` /
    /// `N_BASE_OBJECT` sentinel for the namespace-object constructors. `None` for an
    /// ordinary user function (handled as a function superclass).
    pub(crate) fn native_base_kind(&mut self, handle: Handle) -> Option<u16> {
        if let Some(id) = self.realm.native_at(handle) {
            return Some(id);
        }
        // `Array`/`Object` are namespace objects (no native id), matched by the
        // identity of their global binding.
        let hv = NanBox::handle(handle.to_raw());
        if self.current.get("Array").and_then(|v| v.as_handle()) == hv.as_handle() {
            return Some(N_BASE_ARRAY);
        }
        if self.current.get("Object").and_then(|v| v.as_handle()) == hv.as_handle() {
            return Some(N_BASE_OBJECT);
        }
        None
    }

    /// Whether a native base id denotes a constructor whose instances are
    /// *cell-bearing* — a real Map/Set/typed array/Date/RegExp/wrapper/ArrayBuffer/
    /// DataView/Array cell that must be created by the base constructor (so the
    /// derived instance carries the native internal slots), as opposed to the
    /// Error/Object families whose instances are ordinary objects decorated in
    /// place.
    pub(crate) fn native_base_is_cell(id: u16) -> bool {
        matches!(
            id,
            N_MAP
                | N_SET
                | N_WEAKMAP
                | N_WEAKSET
                | N_DATE
                | N_REGEXP
                | N_NUMBER
                | N_STRING
                | N_BOOLEAN
                | N_ARRAY_BUFFER
                | N_SHARED_ARRAY_BUFFER
                | N_DATA_VIEW
                | N_BASE_ARRAY
        ) || (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id)
            || crate::nbexec::temporal::is_temporal_ctor_id(id)
    }

    pub(crate) fn class_prototype(&mut self, class_id: u32, class_handle: Handle) -> Handle {
        if let Some(p) = self.realm.class_prototype_cached(class_id) {
            return p;
        }
        let proto = self.realm.new_object();
        // Cache immediately so a self-referential computed key cannot recurse.
        self.realm.set_class_prototype(class_id, proto);

        let env = self.class_envs[class_id as usize].clone();
        // Link to the superclass prototype (class chain or function superclass).
        // Use the *cached* class-super id (resolved once at definition time) so
        // building the prototype eagerly does not re-evaluate `extends`.
        let class = self.classes[class_id as usize];
        if let Some(super_id) = self.class_super_id[class_id as usize] {
            let super_proto = self.class_prototype_by_id(super_id);
            self.realm.set_object_proto(proto, Some(super_proto));
        } else if let Some(fn_super) = self.class_fn_super[class_id as usize]
            && let Some(sh) = fn_super.as_handle().map(Handle::from_raw)
        {
            // `class D extends fn {}`: `D.prototype.[[Prototype]]` is
            // `fn.prototype` (an object — created on demand).
            if let Ok(sp) = self.read_member(sh, "prototype")
                && let Some(spp) = sp.as_handle().map(Handle::from_raw)
            {
                self.realm.set_object_proto(proto, Some(spp));
            }
        } else if let Some(super_id) = self.class_native_super[class_id as usize] {
            // `class D extends Error|Iterator|… {}` (a native superclass):
            // `D.prototype.[[Prototype]]` is `NativeCtor.prototype`. Resolve the
            // constructor by its global name (Error family by ERROR_NAMES, else by
            // scanning the global bindings for the native with this id).
            let super_proto = self
                .native_ctor_by_id(super_id)
                .and_then(|c| self.realm.get_property(c, "prototype"))
                .and_then(|p| p.as_handle())
                .map(Handle::from_raw);
            if let Some(sp) = super_proto {
                self.realm.set_object_proto(proto, Some(sp));
            }
        }

        // Install this class's own instance methods/accessors.
        for (idx, member) in class.body.iter().enumerate() {
            let ClassMember::Method(m) = member else {
                continue;
            };
            if m.is_static || m.kind == MethodKind::Constructor {
                continue;
            }
            // Private methods/accessors are not public prototype members; they
            // brand the instance directly (installed in `instantiate`).
            if matches!(&m.key, PropertyKey::Private(_)) {
                continue;
            }
            let saved = core::mem::replace(&mut self.current, env.clone());
            // The computed key was pre-evaluated (and any throw raised) at class
            // definition; read it back rather than re-evaluate. A non-computed key
            // is a deterministic name lookup.
            let key = self.class_member_key(class_id, idx, &m.key);
            let f = self.make_method(
                &m.value.params,
                Body::Block(&m.value.body),
                m.value.is_async,
                m.value.is_generator,
                Some(class_id),
                false,
            );
            self.current = saved;
            let Ok(key) = key else { continue };
            if let Some(n) = self.method_display_name(&key, m.kind) {
                self.install_method_meta(f, &n, &m.value.params);
            }
            match m.kind {
                MethodKind::Method => {
                    self.realm.set_hidden_property(proto, &key, f);
                }
                MethodKind::Get => {
                    self.realm
                        .define_accessor(proto, &key, f, NanBox::undefined());
                    self.realm.mark_hidden(proto, &key);
                }
                MethodKind::Set => {
                    self.realm
                        .define_accessor(proto, &key, NanBox::undefined(), f);
                    self.realm.mark_hidden(proto, &key);
                }
                MethodKind::Constructor => {}
            }
        }
        // Non-enumerable `constructor` back-link.
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(class_handle.to_raw()));
        proto
    }

    /// Materializes a class's prototype by id, recovering its constructor handle
    /// from `class_handles`.
    pub(crate) fn class_prototype_by_id(&mut self, class_id: u32) -> Handle {
        if let Some(p) = self.realm.class_prototype_cached(class_id) {
            return p;
        }
        let handle = self
            .class_handles
            .get(class_id as usize)
            .and_then(|v| v.as_handle().map(Handle::from_raw));
        match handle {
            Some(handle) => self.class_prototype(class_id, handle),
            None => {
                let proto = self.realm.new_object();
                self.realm.set_class_prototype(class_id, proto);
                proto
            }
        }
    }

    /// Runs one class's field initializers and constructor on `instance` (with
    /// `this` already bound). `super(args)` reaches the base via `pending_super`.
    /// Applies a class's own (non-static) instance field initializers to
    /// `instance`. Run before the constructor body (base class) / after the
    /// implicit super for a constructor-less derived class.
    /// Installs *one class's* private methods/accessors (`#m`, `get #x`, `set #x`)
    /// on `instance` as own hidden private-keyed members. Per spec these are added
    /// by `InitializeInstanceElements` — i.e. **after `super()` returns**, together
    /// with (and just before) that class's field initializers — so a private
    /// method is NOT reachable while a base constructor is still running (called
    /// from `init_instance_fields`, not at allocation).
    pub(crate) fn install_private_methods(
        &mut self,
        class_id: u32,
        instance: Handle,
    ) -> Result<(), ExecError> {
        let cenv = self.class_envs[class_id as usize].clone();
        let class = self.classes[class_id as usize];
        for member in &class.body {
            let ClassMember::Method(m) = member else {
                continue;
            };
            if m.is_static || !matches!(&m.key, PropertyKey::Private(_)) {
                continue;
            }
            let key = {
                let saved = core::mem::replace(&mut self.current, cenv.clone());
                let k = self.eval_member_key_for_class(&m.key, class_id);
                self.current = saved;
                k?
            };
            // A private method/accessor is defined once per class and shared by
            // every instance (`c1.#m === c2.#m`); cache by (class, kind, key).
            let cache_key = (
                class_id,
                alloc::format!("{}\u{0}{key}", method_kind_tag(m.kind)),
            );
            let f = if let Some(f) = self.private_method_cache.get(&cache_key) {
                *f
            } else {
                let saved = core::mem::replace(&mut self.current, cenv.clone());
                let f = self.make_method(
                    &m.value.params,
                    Body::Block(&m.value.body),
                    m.value.is_async,
                    m.value.is_generator,
                    Some(class_id),
                    false,
                );
                self.current = saved;
                if let Some(n) = self.private_method_display_name(&m.key, m.kind) {
                    self.install_method_meta(f, &n, &m.value.params);
                }
                self.private_method_cache.insert(cache_key, f);
                f
            };
            match m.kind {
                MethodKind::Method => {
                    self.realm.set_hidden_property(instance, &key, f);
                    // A private method is non-writable (and non-configurable):
                    // `obj.#method = …` is a TypeError in PrivateSet.
                    self.realm.set_readonly_property(instance, &key);
                }
                MethodKind::Get => {
                    self.realm
                        .define_accessor(instance, &key, f, NanBox::undefined());
                    self.realm.mark_hidden(instance, &key);
                }
                MethodKind::Set => {
                    self.realm
                        .define_accessor(instance, &key, NanBox::undefined(), f);
                    self.realm.mark_hidden(instance, &key);
                }
                MethodKind::Constructor => {}
            }
        }
        Ok(())
    }

    pub(crate) fn init_instance_fields(
        &mut self,
        class_id: u32,
        instance: Handle,
    ) -> Result<(), ExecError> {
        // Private methods/accessors install first (before this class's field
        // initializers, which may reference them), and only now — after `super()`.
        self.install_private_methods(class_id, instance)?;
        let class = self.classes[class_id as usize];
        // Instance field initializers run with the class as their home object, so
        // `super.x` (e.g. inside an arrow stored in a field) resolves against the
        // superclass prototype with `IsStatic` false.
        let saved_home = self.current_home.replace(class_id);
        let saved_lexical_home = self.current_lexical_home.replace(class_id);
        let saved_home_static = core::mem::replace(&mut self.current_home_static, false);
        let saved_home_obj = core::mem::take(&mut self.current_home_object);
        // A field initializer is evaluated as its own function-like with
        // `[[NewTarget]]` of undefined — so a `new.target` inside (e.g. via a
        // direct eval) sees undefined, not the constructor reached via `new`.
        let saved_target = core::mem::replace(&mut self.new_target, NanBox::undefined());
        let result = (|| {
            for (idx, member) in class.body.iter().enumerate() {
                if let ClassMember::Field(field) = member
                    && !field.is_static
                {
                    // A computed field name (`[expr]`) was evaluated exactly once, in
                    // source order, at class-definition time (its side effects and
                    // ToPropertyKey coercion already ran) and stored in
                    // `class_member_keys`; read it back here rather than re-evaluating
                    // per construction. Only the *value* is (re)computed below.
                    let is_private = matches!(&field.key, PropertyKey::Private(_));
                    let key = self.class_member_key(class_id, idx, &field.key)?;
                    let v = match &field.value {
                        Some(e) => self.eval(e)?,
                        None => NanBox::undefined(),
                    };
                    // DefineField step 7: an anonymous function/arrow/class
                    // initializer takes the field name (`x = () => {}` → name "x";
                    // `#f = () => {}` → name "#f").
                    if let Some(init) = &field.value
                        && matches!(init, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_))
                        && let Some(disp) = self.method_display_name(&key, MethodKind::Method)
                    {
                        self.set_fn_name_owned(v, &disp);
                    }
                    if is_private {
                        // `PrivateFieldAdd` on a **non-extensible** object is a
                        // TypeError (the `nonextensible-applies-to-private`
                        // semantics) — e.g. a base constructor returned a frozen
                        // object or a module namespace. Private fields are *not*
                        // string-keyed properties, so they never force a deferred
                        // module namespace.
                        if !self.realm.is_extensible(instance) {
                            let m =
                                self.new_str("Cannot add private field to a non-extensible object");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                        self.realm.set_property(instance, &key, v);
                        continue;
                    }
                    // A class field is a CreateDataPropertyOrThrow ([[DefineOwnProperty]])
                    // on the instance; if the instance is a Deferred Module Namespace
                    // (e.g. a base constructor returned one) that forces evaluation.
                    #[cfg(all(feature = "module", feature = "std"))]
                    self.trigger_deferred_namespace(instance, &key)?;
                    self.realm.set_property(instance, &key, v);
                }
            }
            Ok(())
        })();
        self.current_home = saved_home;
        self.current_lexical_home = saved_lexical_home;
        self.current_home_static = saved_home_static;
        self.current_home_object = saved_home_obj;
        self.new_target = saved_target;
        result
    }

    /// A constructor's `return value` overrides the new instance only when it is
    /// an **Object** (per `[[Construct]]` / `SuperCall`). Returns that object's
    /// handle, or `None` for `undefined` / a primitive (incl. string/bigint/symbol
    /// wrapper handles, which are primitives here).
    pub(crate) fn constructor_return_handle(&self, ret: Option<NanBox>) -> Option<Handle> {
        ret.and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .filter(|h| {
                self.realm.string_value(*h).is_none()
                    && self.realm.bigint_at(*h).is_none()
                    && self.realm.symbol_at(*h).is_none()
            })
    }

    pub(crate) fn run_constructor(
        &mut self,
        class_id: u32,
        env: &Scope,
        instance: Handle,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let class = self.classes[class_id as usize];
        let parent = self.resolve_super(class, env)?;
        let saved_super = core::mem::replace(&mut self.pending_super, parent.clone());
        let native_parent = self.class_native_super[class_id as usize];
        let saved_super_native = core::mem::replace(&mut self.pending_super_native, native_parent);
        let fn_parent = self.class_fn_super[class_id as usize];
        let saved_super_fn = core::mem::replace(&mut self.pending_super_fn, fn_parent);
        let saved_scope = core::mem::replace(&mut self.current, env.child());
        // A class constructor body (and its field initializers) is strict code.
        let saved_strict = core::mem::replace(&mut self.strict, true);
        // A constructor body is *never* a proper-tail-call context: `[[Construct]]`
        // must survive to apply the constructor-return rule (a derived constructor
        // returning a non-object throws), so a `return f()` here must not reuse the
        // frame. Clear the inherited `tail_pos` for the whole body.
        let saved_tail_pos = core::mem::replace(&mut self.tail_pos, false);
        // The constructor body's home class is this class, so a `this.#x` /
        // `super.x` inside it resolves the private name (and super members)
        // against this class. (`init_instance_fields` re-establishes the same
        // home for the field initializers it runs.)
        let saved_home = self.current_home.replace(class_id);
        let saved_lexical_home = self.current_lexical_home.replace(class_id);
        let saved_home_static = core::mem::replace(&mut self.current_home_static, false);
        let saved_home_obj = self.current_home_object.replace(instance);
        let result = (|| {
            let ctor = class.body.iter().find_map(|m| match m {
                ClassMember::Method(m) if m.kind == MethodKind::Constructor => Some(m),
                _ => None,
            });
            match (ctor, &parent) {
                (Some(ctor), _) => {
                    // A *derived* constructor (any superclass — class, native, or
                    // function) runs its body with `this` in a temporal dead zone:
                    // `super(...)` initializes `this` and runs this class's field
                    // initializers. A *base* constructor binds `this` and runs its
                    // fields up front, before the body.
                    let is_derived = parent.is_some()
                        || self.pending_super_native.is_some()
                        || self.pending_super_fn.is_some();
                    let (poisoned_this, saved_pending) = if is_derived {
                        let inst = NanBox::handle(instance.to_raw());
                        let sp = self.pending_this_init.replace((inst, class_id));
                        (
                            Some(core::mem::replace(&mut self.this_val, NanBox::tdz())),
                            sp,
                        )
                    } else {
                        // Own fields initialize before the body, so a constructor
                        // write isn't clobbered by a later field decl.
                        self.init_instance_fields(class_id, instance)?;
                        (None, None)
                    };
                    let scope = self.current.child();
                    let saved = core::mem::replace(&mut self.current, scope);
                    let r: Result<Option<NanBox>, ExecError> = (|| {
                        // `arguments` is available in the constructor (incl. its
                        // parameter defaults), bound before the parameters.
                        let arg_arr = self.realm.new_array(args.to_vec());
                        self.current
                            .declare("arguments", NanBox::handle(arg_arr.to_raw()));
                        // Bind parameters (rest/default/destructuring supported).
                        for (i, param) in ctor.value.params.iter().enumerate() {
                            let value = if param.rest {
                                let rest = args[i.min(args.len())..].to_vec();
                                NanBox::handle(self.realm.new_array(rest).to_raw())
                            } else {
                                let mut v = args.get(i).copied().unwrap_or(NanBox::undefined());
                                if matches!(v.unpack(), Unpacked::Undefined)
                                    && let Some(d) = &param.default
                                {
                                    v = self.eval(d)?;
                                    self.infer_binding_name(&param.target, d, v);
                                }
                                v
                            };
                            self.bind_pattern(&param.target, value)?;
                        }
                        // The constructor's `return value` (if an object) overrides
                        // the new instance; captured here.
                        let mut returned = None;
                        for stmt in &ctor.value.body {
                            if let Flow::Return(v) = self.exec(stmt)? {
                                returned = Some(v);
                                break;
                            }
                        }
                        Ok(returned)
                    })();
                    self.current = saved;
                    let r = r?;
                    if is_derived {
                        // `super()` clears `pending_this_init` (and sets `this`); if
                        // it is still set, the body returned without calling super.
                        let super_called = self.pending_this_init.is_none();
                        self.this_val = poisoned_this.expect("derived poisoned this");
                        self.pending_this_init = saved_pending;
                        // A derived constructor must call `super()` before it
                        // returns — but only when its completion value is empty
                        // (no return, or `return undefined`): the `this` binding is
                        // then required. A return of an object becomes the result
                        // (bypassing `this`), and a return of a non-undefined
                        // non-object is a TypeError handled by `construct` — neither
                        // is a "must call super" ReferenceError.
                        let ret_empty = r.is_none_or(|v| matches!(v.unpack(), Unpacked::Undefined));
                        if !super_called && ret_empty {
                            let m = self.new_str(
                                "Must call super constructor before accessing 'this' or returning from derived constructor",
                            );
                            return Err(ExecError::Throw(
                                self.make_error(N_REFERENCE_ERROR, Some(m)),
                            ));
                        }
                    }
                    Ok(r)
                }
                // No own constructor but a base: implicit `super(args)`, then
                // this class's own field initializers. A super constructor that
                // returns an Object rebinds `this`, so the fields target it.
                (None, Some((pid, penv))) => {
                    let ret = self.run_constructor(*pid, &penv.clone(), instance, args)?;
                    let target = self.constructor_return_handle(ret).unwrap_or(instance);
                    self.init_instance_fields(class_id, target)?;
                    Ok(ret)
                }
                (None, None) => {
                    // A constructor-less class extending a *native* superclass
                    // (`class X extends Error {}`) performs the implicit
                    // `super(...args)` into the native constructor, so e.g. the
                    // error message is forwarded.
                    if let Some(nid) = native_parent {
                        self.apply_native_super(nid, instance, args)?;
                        self.init_instance_fields(class_id, instance)?;
                        Ok(None)
                    } else if let Some(fnp) = fn_parent {
                        // Constructor-less class extending a function: implicit
                        // `super(...args)` calls the function with `this` = instance.
                        // A returned Object becomes the result and the field target.
                        let ret =
                            self.call_with_this(fnp, NanBox::handle(instance.to_raw()), args)?;
                        let returned = self.constructor_return_handle(Some(ret));
                        let target = returned.unwrap_or(instance);
                        self.init_instance_fields(class_id, target)?;
                        Ok(returned.map(|h| NanBox::handle(h.to_raw())))
                    } else {
                        self.init_instance_fields(class_id, instance)?;
                        Ok(None)
                    }
                }
            }
        })();
        self.current = saved_scope;
        self.pending_super = saved_super;
        self.pending_super_native = saved_super_native;
        self.pending_super_fn = saved_super_fn;
        self.strict = saved_strict;
        self.tail_pos = saved_tail_pos;
        self.current_home = saved_home;
        self.current_lexical_home = saved_lexical_home;
        self.current_home_static = saved_home_static;
        self.current_home_object = saved_home_obj;
        result
    }

    /// Finds `name` as a method in the superclass chain of the currently-running
    /// method's home class, returning a callable bound to the base definition.
    pub(crate) fn resolve_super_method(&mut self, name: &str) -> Result<NanBox, ExecError> {
        // An object-literal method: `super.m()` is `HomeObject.[[Prototype]].m`,
        // called (by the caller) with the current `this`.
        if self.current_home.is_none()
            && let Some(home) = self.current_home_object
        {
            if let Some(proto) = self.realm.object_proto(home) {
                let f = self.read_member(proto, name)?;
                if f.as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Ok(f);
                }
            }
            return Err(ExecError::Throw(
                self.new_str(&alloc::format!("super method {name} not found")),
            ));
        }
        let home = self
            .current_home
            .ok_or(ExecError::Unsupported("super outside a method"))?;
        let mut cur = self.resolve_super(
            self.classes[home as usize],
            &self.class_envs[home as usize].clone(),
        )?;
        while let Some((pid, penv)) = cur {
            let class = self.classes[pid as usize];
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && m.is_static == self.current_home_static
                    && m.kind == MethodKind::Method
                    && static_key(&m.key).ok().as_deref() == Some(name)
                {
                    let saved = core::mem::replace(&mut self.current, penv.clone());
                    let f = self.make_method(
                        &m.value.params,
                        Body::Block(&m.value.body),
                        m.value.is_async,
                        m.value.is_generator,
                        Some(pid),
                        self.current_home_static,
                    );
                    self.current = saved;
                    return Ok(f);
                }
            }
            // A function superclass (`extends fn`) is not in the class chain;
            // resolve `super.m` through `fn.prototype` (and its chain).
            if let Some(fn_super) = self.class_fn_super[pid as usize]
                && let Some(sh) = fn_super.as_handle().map(Handle::from_raw)
                && let Ok(sp) = self.read_member(sh, "prototype")
                && let Some(spp) = sp.as_handle().map(Handle::from_raw)
            {
                let f = self.read_member(spp, name)?;
                if f.as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Ok(f);
                }
            }
            cur = self.resolve_super(class, &penv)?;
        }
        // The home class itself may directly extend a function (no intermediate
        // class level), so check its function super's prototype too.
        if let Some(fn_super) = self.class_fn_super[home as usize]
            && let Some(sh) = fn_super.as_handle().map(Handle::from_raw)
            && let Ok(sp) = self.read_member(sh, "prototype")
            && let Some(spp) = sp.as_handle().map(Handle::from_raw)
        {
            let f = self.read_member(spp, name)?;
            if f.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                return Ok(f);
            }
        }
        // Fallback: resolve the method off the real prototype chain of the
        // HomeObject (the defining class's `.prototype`). Catches a method added
        // dynamically to a superclass prototype AND a native superclass's method
        // (`class MyMap extends Map { … super.set() }` → `Map.prototype.set`),
        // neither of which is a declared class-body method.
        if !self.current_home_static {
            let home_proto = self.class_prototype_by_id(home);
            if let Some(super_base) = self.realm.object_proto(home_proto)
                && self.has_property(super_base, name)
            {
                let f = self.read_member(super_base, name)?;
                if f.as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Ok(f);
                }
            }
        }
        Err(ExecError::Throw(
            self.new_str(&alloc::format!("super method {name} not found")),
        ))
    }

    /// `super.name` as a value read: a super getter is invoked (with the current
    /// `this`); a super method is returned as a bound function.
    /// `super.name = value`: invoke an inherited setter (found on the home's parent
    /// chain) with `this` = the current receiver; if there is none, assign the property
    /// directly on the receiver.
    pub(crate) fn assign_super_member(
        &mut self,
        name: &str,
        value: NanBox,
    ) -> Result<(), ExecError> {
        // An object-literal method: `super.x = v` uses `HomeObject.[[Prototype]]`.
        if self.current_home.is_none()
            && let Some(home) = self.current_home_object
        {
            if let Some(proto) = self.realm.object_proto(home)
                && let Some((_, setter)) = self.realm.accessor(proto, name)
                && !matches!(setter.unpack(), Unpacked::Undefined)
            {
                self.call_with_this(setter, self.this_val, &[value])?;
                return Ok(());
            }
            if let Some(th) = self.this_val.as_handle().map(Handle::from_raw) {
                self.realm.set_property(th, name, value);
            }
            return Ok(());
        }
        let home = self
            .current_home
            .ok_or(ExecError::Unsupported("super outside a method"))?;
        // GetSuperBase = HomeObject.[[GetPrototypeOf]](). For a class the home object
        // is the constructor (static method) or the class prototype (instance method).
        // PutValue on a SuperProperty performs ToObject(GetSuperBase), so a null super
        // base — e.g. after `Object.setPrototypeOf(C, null)` — is a TypeError, thrown
        // *after* the RHS has been evaluated and regardless of whether an inherited
        // setter exists.
        let home_obj = if self.current_home_static {
            self.class_handles
                .get(home as usize)
                .and_then(|v| v.as_handle().map(Handle::from_raw))
        } else {
            Some(self.class_prototype_by_id(home))
        };
        if let Some(ho) = home_obj
            && self.realm.object_proto(ho).is_none()
        {
            return Err(self.type_error("Cannot set property on null (super)"));
        }
        let mut cur = self.resolve_super(
            self.classes[home as usize],
            &self.class_envs[home as usize].clone(),
        )?;
        while let Some((pid, penv)) = cur {
            let class = self.classes[pid as usize];
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && m.is_static == self.current_home_static
                    && m.kind == MethodKind::Set
                    && static_key(&m.key).ok().as_deref() == Some(name)
                {
                    let saved = core::mem::replace(&mut self.current, penv.clone());
                    let f = self.make_method(
                        &m.value.params,
                        Body::Block(&m.value.body),
                        m.value.is_async,
                        m.value.is_generator,
                        Some(pid),
                        self.current_home_static,
                    );
                    self.current = saved;
                    self.call_with_this(f, self.this_val, &[value])?;
                    return Ok(());
                }
            }
            cur = self.resolve_super(class, &penv)?;
        }
        // No inherited setter — the write lands on the receiver (`this`). A
        // Deferred Module Namespace receiver forces evaluation ([[Set]]/[[Define]]).
        if let Some(th) = self.this_val.as_handle().map(Handle::from_raw) {
            #[cfg(all(feature = "module", feature = "std"))]
            self.trigger_deferred_namespace(th, name)?;
            // `super.x = v` is `superBase.[[Set]](x, v, this)`; OrdinarySet reads
            // `Receiver.[[GetOwnProperty]](x)` before writing, so a namespace
            // receiver whose export `x` is still in its TDZ throws a ReferenceError.
            #[cfg(all(feature = "module", feature = "std"))]
            self.namespace_binding_tdz(th, name)?;
            // `super.x = v` is `superBase.[[Set]](x, v, this)` — the write lands on
            // the receiver. Route it through the strict-aware member-assignment path
            // so a failed set (frozen/non-extensible/non-writable receiver) throws a
            // TypeError, since a class method body is always strict.
            let key = self.new_str(name);
            self.assign_member_value(th, key, value)?;
        }
        Ok(())
    }

    pub(crate) fn resolve_super_member(&mut self, name: &str) -> Result<NanBox, ExecError> {
        // An object-literal method: `super.x` reads `HomeObject.[[Prototype]].x`.
        // A data property is returned directly; an inherited *getter* is invoked
        // with the current `this` (the receiver), not the prototype — so
        // `super.accessor` in `obj.method()` sees `obj` as `this`.
        if self.current_home.is_none()
            && let Some(home) = self.current_home_object
        {
            // GetSuperBase = HomeObject.[[GetPrototypeOf]](); ? RequireObjectCoercible
            // throws a TypeError when the home object's prototype is null (e.g.
            // `Object.setPrototypeOf(obj, null)` before `super.x` in `obj.method`).
            let Some(proto) = self.realm.object_proto(home) else {
                return Err(self.type_error("Cannot read property of null (super)"));
            };
            if let Some((getter, _)) = self.realm.accessor(proto, name) {
                if matches!(getter.unpack(), Unpacked::Undefined) {
                    return Ok(NanBox::undefined());
                }
                return self.call_with_this(getter, self.this_val, &[]);
            }
            return self.read_member(proto, name);
        }
        let home = self
            .current_home
            .ok_or(ExecError::Unsupported("super outside a method"))?;
        let mut cur = self.resolve_super(
            self.classes[home as usize],
            &self.class_envs[home as usize].clone(),
        )?;
        while let Some((pid, penv)) = cur {
            let class = self.classes[pid as usize];
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && m.is_static == self.current_home_static
                    && matches!(m.kind, MethodKind::Method | MethodKind::Get)
                    && static_key(&m.key).ok().as_deref() == Some(name)
                {
                    let saved = core::mem::replace(&mut self.current, penv.clone());
                    let f = self.make_method(
                        &m.value.params,
                        Body::Block(&m.value.body),
                        m.value.is_async,
                        m.value.is_generator,
                        Some(pid),
                        self.current_home_static,
                    );
                    self.current = saved;
                    return if m.kind == MethodKind::Get {
                        self.call_with_this(f, self.this_val, &[])
                    } else {
                        Ok(f)
                    };
                }
            }
            // A function superclass at this level: for a static member, `super.x`
            // reads from the superclass *constructor* object directly; for an
            // instance member, through `fn.prototype`.
            if self.current_home_static {
                if let Some(v) = self.fn_super_static_member(pid, name)? {
                    return Ok(v);
                }
            } else if let Some(v) = self.fn_super_member(pid, name)? {
                return Ok(v);
            }
            cur = self.resolve_super(class, &penv)?;
        }
        // The home class may directly extend a function (no class parent level).
        if self.current_home_static {
            if let Some(v) = self.fn_super_static_member(home, name)? {
                return Ok(v);
            }
        } else if let Some(v) = self.fn_super_member(home, name)? {
            return Ok(v);
        }
        // Fallback for a non-static member: the class-body walk above only matches
        // *declared* methods/getters, so a property added dynamically to a
        // superclass prototype (`A.prototype.p = …`) is missed. Resolve it the
        // spec way — GetPrototypeOf(HomeObject).[[Get]](name, this) — over the real
        // prototype chain (HomeObject = the defining class's `.prototype`).
        if !self.current_home_static {
            let home_proto = self.class_prototype_by_id(home);
            if let Some(super_base) = self.realm.object_proto(home_proto)
                && self.has_property(super_base, name)
            {
                if let Some((getter, _)) = self.realm.accessor(super_base, name) {
                    if matches!(getter.unpack(), Unpacked::Undefined) {
                        return Ok(NanBox::undefined());
                    }
                    return self.call_with_this(getter, self.this_val, &[]);
                }
                return self.read_member(super_base, name);
            }
        }
        Ok(NanBox::undefined())
    }

    /// Reads `super.name` for a *static* member through class `cid`'s function
    /// superclass *constructor object* (`class C extends fn { static {...} }`): a
    /// data property is returned directly; a getter anywhere on the constructor's
    /// prototype chain is invoked with the current `this`. `None` if `cid` has no
    /// function super or the constructor lacks the name.
    fn fn_super_static_member(
        &mut self,
        cid: u32,
        name: &str,
    ) -> Result<Option<NanBox>, ExecError> {
        let Some(fn_super) = self.class_fn_super[cid as usize] else {
            return Ok(None);
        };
        let Some(sh) = fn_super.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        if !self.has_property(sh, name) {
            return Ok(None);
        }
        let mut cur = Some(sh);
        while let Some(h) = cur {
            if let Some((getter, _)) = self.realm.accessor(h, name) {
                if matches!(getter.unpack(), Unpacked::Undefined) {
                    return Ok(Some(NanBox::undefined()));
                }
                return Ok(Some(self.call_with_this(getter, self.this_val, &[])?));
            }
            if self.realm.has_own(h, name) {
                return Ok(Some(self.read_member(h, name)?));
            }
            cur = self.realm.object_proto(h);
        }
        Ok(Some(self.read_member(sh, name)?))
    }

    /// Reads `super.name` through class `cid`'s function superclass prototype
    /// (`class C extends fn {}`): a data property is returned directly; a getter
    /// is invoked with the current `this`. `None` if `cid` has no function super
    /// or the prototype lacks the name.
    fn fn_super_member(&mut self, cid: u32, name: &str) -> Result<Option<NanBox>, ExecError> {
        let Some(fn_super) = self.class_fn_super[cid as usize] else {
            return Ok(None);
        };
        let Some(sh) = fn_super.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        let Ok(sp) = self.read_member(sh, "prototype") else {
            return Ok(None);
        };
        let Some(spp) = sp.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        if !self.has_property(spp, name) {
            return Ok(None);
        }
        // An accessor anywhere on the prototype chain is invoked with the current
        // `this`; otherwise the data property is read directly.
        let mut cur = Some(spp);
        while let Some(h) = cur {
            if let Some((getter, _)) = self.realm.accessor(h, name) {
                if matches!(getter.unpack(), Unpacked::Undefined) {
                    return Ok(Some(NanBox::undefined()));
                }
                return Ok(Some(self.call_with_this(getter, self.this_val, &[])?));
            }
            if self.realm.has_own(h, name) {
                break;
            }
            cur = self.realm.object_proto(h);
        }
        Ok(Some(self.read_member(spp, name)?))
    }
}

/// A short tag distinguishing a method kind, used to key the private-method cache
/// (a getter and setter can share a storage key, so the kind must be part of the
/// cache identity).
fn method_kind_tag(kind: MethodKind) -> &'static str {
    match kind {
        MethodKind::Get => "g",
        MethodKind::Set => "s",
        MethodKind::Method => "m",
        MethodKind::Constructor => "c",
    }
}
