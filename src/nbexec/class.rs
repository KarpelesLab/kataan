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

    /// Registers a class and allocates a class value capturing the current scope.
    pub(crate) fn make_class(&mut self, class: &'a Class) -> Result<NanBox, ExecError> {
        let class_id = self.classes.len() as u32;
        self.classes.push(class);
        // Build the static members (`static foo() {}` / `static x = …`).
        let mut statics = alloc::collections::BTreeMap::new();
        let mut static_fields = Vec::new();
        let mut static_getters = alloc::collections::BTreeMap::new();
        let mut static_setters = alloc::collections::BTreeMap::new();
        for member in &class.body {
            match member {
                ClassMember::Method(m) if m.is_static && m.kind == MethodKind::Method => {
                    // A computed key is evaluated at class-definition time; a throw
                    // from it propagates (it does not silently skip the member).
                    let key = self.eval_prop_key(&m.key)?;
                    // A static method's home is this class, entered statically, so
                    // `super.x` resolves against the superclass's static members.
                    let f = self.make_method(
                        &m.value.params,
                        Body::Block(&m.value.body),
                        false,
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
                    let key = self.eval_prop_key(&field.key)?;
                    let v = match &field.value {
                        Some(e) => self.eval(e).unwrap_or(NanBox::undefined()),
                        None => NanBox::undefined(),
                    };
                    // Static fields are enumerable own keys of the constructor.
                    if !static_fields.contains(&key) {
                        static_fields.push(key.clone());
                    }
                    statics.insert(key, v);
                }
                // `static get x() {}` / `static set x(v) {}` — accessors.
                ClassMember::Method(m)
                    if m.is_static && matches!(m.kind, MethodKind::Get | MethodKind::Set) =>
                {
                    let key = self.eval_prop_key(&m.key)?;
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
        self.class_statics.push(statics);
        self.class_static_fields.push(static_fields);
        self.class_static_get.push(static_getters);
        self.class_static_set.push(static_setters);
        // The methods' captured scope; a named class binds its own name here (so
        // `class C { m() { return C; } }` can self-reference), filled in below.
        let class_env = self.current.child();
        self.class_envs.push(class_env.clone());
        // Record a native-constructor superclass (`extends Error`), if any, so
        // construction and `instanceof` can reach it (it has no class id).
        let native_super = if let Some(expr) = &class.super_class {
            self.eval(expr).ok().and_then(|v| {
                let h = Handle::from_raw(v.as_handle()?);
                if self.realm.class_at(h).is_some() {
                    None
                } else {
                    self.realm.native_at(h)
                }
            })
        } else {
            None
        };
        self.class_native_super.push(native_super);
        let handle = self.realm.new_class(class_id, class_env.clone());
        let class_val = NanBox::handle(handle.to_raw());
        self.class_handles.push(class_val);
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
        let class_name = class.id.as_ref().map(|id| String::from(&*id.name));
        self.install_fn_name_length(handle, class_name.as_deref().unwrap_or(""), ctor_len);
        if class_name.is_none() {
            // Anonymous: `name` becomes own only via NamedEvaluation (`let C = class
            // {}`); remove the placeholder so `set_fn_name` can set it later, but
            // keep `length` (always own).
            self.realm.delete_property(handle, "name");
        }
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
            // `name`/`length` are already installed with their own attributes.
            if k == "name" || k == "length" {
                continue;
            }
            self.realm.set_property(handle, &k, v);
            if !field_keys.contains(&k) {
                // A static method is non-enumerable; a static field is enumerable.
                self.realm.mark_hidden(handle, &k);
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
            self.realm.define_accessor(handle, &k, getter, setter);
            self.realm.mark_hidden(handle, &k);
        }
        for (k, setter) in &setters {
            if !self.class_static_get[class_id as usize].contains_key(k) {
                self.realm
                    .define_accessor(handle, k, NanBox::undefined(), *setter);
                self.realm.mark_hidden(handle, k);
            }
        }
        // Bind the class's own name in its methods' scope (a named class
        // expression sees itself; the binding is read-only in spec but not
        // enforced here).
        if let Some(id) = &class.id {
            class_env.declare(&id.name, class_val);
        }
        // Run `static { … }` initialization blocks with `this` = the class and the
        // class name bound (so the block can reference the class and its statics).
        if class
            .body
            .iter()
            .any(|m| matches!(m, ClassMember::StaticBlock { .. }))
        {
            let scope = self.current.child();
            if let Some(id) = &class.id {
                scope.declare(&id.name, class_val);
            }
            let saved = core::mem::replace(&mut self.current, scope);
            let saved_this = core::mem::replace(&mut self.this_val, class_val);
            let r = (|| {
                for member in &class.body {
                    if let ClassMember::StaticBlock { body, .. } = member {
                        for stmt in body {
                            self.exec(stmt)?;
                        }
                    }
                }
                Ok(())
            })();
            self.current = saved;
            self.this_val = saved_this;
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
        let raw = value?
            .as_handle()
            .ok_or(ExecError::Unsupported("extends a non-class"))?;
        let h = Handle::from_raw(raw);
        if let Some(parent) = self.realm.class_at(h) {
            Ok(Some(parent))
        } else if self.realm.native_at(h).is_some() {
            // A native superclass (e.g. `extends Error`) has no class chain;
            // it is tracked separately in `class_native_super`.
            Ok(None)
        } else {
            Err(ExecError::Unsupported("extends a non-class"))
        }
    }

    /// Instantiates `new Class(args)`: creates the object, installs the methods
    /// of the whole `extends` chain (derived overriding base), then runs the
    /// constructor (with `super(...)` reaching the base).
    pub(crate) fn instantiate(
        &mut self,
        class_id: u32,
        env: &Scope,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let instance = self.realm.new_object();
        let this_val = NanBox::handle(instance.to_raw());

        // Link the instance to the class's `.prototype` object (which carries the
        // public methods/accessors of the whole `extends` chain), so methods are
        // *inherited* — `instance.m === C.prototype.m`, `instance` has no own `m`,
        // and the prototype chain resolves derived-over-base.
        let proto = self.class_prototype_by_id(class_id);
        self.realm.set_object_proto(instance, Some(proto));

        // *Private* methods/accessors (`#m`) are not public prototype members:
        // they brand the instance directly, so install them as own (hidden)
        // private-keyed members over the whole chain (base-first so a derived
        // private overrides — though shadowing across the chain is rare).
        let mut chain: Vec<(u32, Scope)> = Vec::new();
        let mut cur = Some((class_id, env.clone()));
        while let Some((cid, cenv)) = cur {
            chain.push((cid, cenv.clone()));
            cur = self.resolve_super(self.classes[cid as usize], &cenv)?;
        }
        for (cid, cenv) in chain.iter().rev() {
            let class = self.classes[*cid as usize];
            for member in &class.body {
                let ClassMember::Method(m) = member else {
                    continue;
                };
                if m.is_static {
                    continue;
                }
                // Only private members are installed on the instance; public ones
                // live on the prototype.
                if !matches!(&m.key, PropertyKey::Private(_)) {
                    continue;
                }
                let saved = core::mem::replace(&mut self.current, cenv.clone());
                let key = self.eval_prop_key(&m.key)?;
                let f = self.make_method(
                    &m.value.params,
                    Body::Block(&m.value.body),
                    false,
                    m.value.is_generator,
                    Some(*cid),
                    false,
                );
                self.current = saved;
                if let Some(n) = self.private_method_display_name(&m.key, m.kind) {
                    self.install_method_meta(f, &n, &m.value.params);
                }
                match m.kind {
                    MethodKind::Method => {
                        self.realm.set_hidden_property(instance, &key, f);
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
        }

        self.realm.set_class_tag(instance, class_id);
        let saved_this = core::mem::replace(&mut self.this_val, this_val);
        // `new.target` (the class reached via `new`, passed through the one-shot)
        // holds for the whole constructor, incl. a base reached via `super(...)`.
        let nt = self
            .pending_new_target
            .take()
            .unwrap_or(NanBox::undefined());
        let saved_target = core::mem::replace(&mut self.new_target, nt);
        let result = self.run_constructor(class_id, env, instance, args);
        self.this_val = saved_this;
        self.new_target = saved_target;
        let ret = result?;
        // A constructor that `return`s an *object* makes `new` yield that object
        // instead of the freshly-built instance; a primitive return is ignored.
        match ret {
            Some(v)
                if v.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.string_value(h).is_none()
                        && self.realm.bigint_at(h).is_none()
                        && self.realm.symbol_at(h).is_none()
                }) =>
            {
                Ok(v)
            }
            _ => Ok(this_val),
        }
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
    pub(crate) fn class_prototype(&mut self, class_id: u32, class_handle: Handle) -> Handle {
        if let Some(p) = self.realm.class_prototype_cached(class_id) {
            return p;
        }
        let proto = self.realm.new_object();
        // Cache immediately so a self-referential computed key cannot recurse.
        self.realm.set_class_prototype(class_id, proto);

        let env = self.class_envs[class_id as usize].clone();
        // Link to the superclass prototype (class chain or native superclass).
        let class = self.classes[class_id as usize];
        if let Ok(Some((super_id, _))) = self.resolve_super(class, &env) {
            let super_proto = self.class_prototype_by_id(super_id);
            self.realm.set_object_proto(proto, Some(super_proto));
        }

        // Install this class's own instance methods/accessors.
        for member in &class.body {
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
            let key = self.eval_prop_key(&m.key);
            let f = self.make_method(
                &m.value.params,
                Body::Block(&m.value.body),
                false,
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
    pub(crate) fn init_instance_fields(
        &mut self,
        class_id: u32,
        instance: Handle,
    ) -> Result<(), ExecError> {
        let class = self.classes[class_id as usize];
        for member in &class.body {
            if let ClassMember::Field(field) = member
                && !field.is_static
            {
                // A computed field name (`[expr] = v`) is evaluated here.
                let key = match &field.key {
                    PropertyKey::Computed(e) => {
                        let k = self.eval(e)?;
                        self.member_key(k)
                    }
                    other => static_key(other)?,
                };
                let v = match &field.value {
                    Some(e) => self.eval(e)?,
                    None => NanBox::undefined(),
                };
                self.realm.set_property(instance, &key, v);
            }
        }
        Ok(())
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
        let saved_scope = core::mem::replace(&mut self.current, env.child());
        let result = (|| {
            let ctor = class.body.iter().find_map(|m| match m {
                ClassMember::Method(m) if m.kind == MethodKind::Constructor => Some(m),
                _ => None,
            });
            match (ctor, &parent) {
                (Some(ctor), _) => {
                    // Own fields initialize before the constructor body, so a
                    // constructor write isn't clobbered by a later field decl.
                    self.init_instance_fields(class_id, instance)?;
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
                    r
                }
                // No own constructor but a base: implicit `super(args)`, then
                // this class's own field initializers.
                (None, Some((pid, penv))) => {
                    let ret = self.run_constructor(*pid, &penv.clone(), instance, args)?;
                    self.init_instance_fields(class_id, instance)?;
                    Ok(ret)
                }
                (None, None) => {
                    // A constructor-less class extending a *native* superclass
                    // (`class X extends Error {}`) performs the implicit
                    // `super(...args)` into the native constructor, so e.g. the
                    // error message is forwarded.
                    if let Some(nid) = native_parent {
                        self.apply_native_super(nid, instance, args);
                    }
                    self.init_instance_fields(class_id, instance)?;
                    Ok(None)
                }
            }
        })();
        self.current = saved_scope;
        self.pending_super = saved_super;
        self.pending_super_native = saved_super_native;
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
                        false,
                        m.value.is_generator,
                        Some(pid),
                        self.current_home_static,
                    );
                    self.current = saved;
                    return Ok(f);
                }
            }
            cur = self.resolve_super(class, &penv)?;
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
                        false,
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
        // No inherited setter — the write lands on the receiver (`this`).
        if let Some(th) = self.this_val.as_handle().map(Handle::from_raw) {
            self.realm.set_property(th, name, value);
        }
        Ok(())
    }

    pub(crate) fn resolve_super_member(&mut self, name: &str) -> Result<NanBox, ExecError> {
        // An object-literal method: `super.x` reads `HomeObject.[[Prototype]].x`
        // (a data property, or a getter — invoked through the proto here).
        if self.current_home.is_none()
            && let Some(home) = self.current_home_object
        {
            return match self.realm.object_proto(home) {
                Some(proto) => self.read_member(proto, name),
                None => Ok(NanBox::undefined()),
            };
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
                        false,
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
            cur = self.resolve_super(class, &penv)?;
        }
        Ok(NanBox::undefined())
    }
}
