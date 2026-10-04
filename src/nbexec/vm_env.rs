//! The host side of the bytecode VM's dynamic scoping (`ROADMAP.md` §2.0).
//!
//! A VM function containing a direct `eval` or a `with` statement keeps its
//! bindings in interpreter [`Scope`]s, held by the VM as `Cell::Env` values, so
//! eval code and object environments resolve names exactly as the
//! tree-walker does. Each request runs with `current` (and `strict`) switched to
//! the VM's environment for its duration.

use super::*;
use crate::nbvm::{
    ENV_INIT_CONST, ENV_INIT_LET, ENV_INIT_MARK_CONST, ENV_INIT_SOFT_CONST, ENV_INIT_TDZ,
    ENV_INIT_TDZ_LEXICAL, ENV_INIT_VAR, ENV_INIT_VAR_SET, EnvReq,
};

impl<'a> Interp<'a> {
    /// A VM-held environment value for `scope`.
    pub(crate) fn env_value(&mut self, scope: Scope) -> NanBox {
        NanBox::handle(self.realm.new_env(scope).to_raw())
    }

    /// The scope an environment value holds.
    fn env_scope(&self, v: NanBox) -> Result<Scope, ExecError> {
        v.as_handle()
            .and_then(|raw| self.realm.env_at(Handle::from_raw(raw)))
            .ok_or(ExecError::Unsupported("not a VM environment"))
    }

    fn not_defined(&mut self, name: &str) -> ExecError {
        let m = self.new_str(&alloc::format!("{name} is not defined"));
        ExecError::Throw(self.make_error(N_REFERENCE_ERROR, Some(m)))
    }

    fn uninitialized(&mut self, name: &str) -> ExecError {
        let m = self.new_str(&alloc::format!(
            "Cannot access '{name}' before initialization"
        ));
        ExecError::Throw(self.make_error(N_REFERENCE_ERROR, Some(m)))
    }

    fn const_assign(&mut self) -> ExecError {
        let m = self.new_str("Assignment to constant variable.");
        ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m)))
    }

    /// Serves one [`EnvReq`].
    pub(crate) fn vm_env_op(&mut self, req: EnvReq<'_>) -> Result<(NanBox, NanBox), ExecError> {
        let undef = NanBox::undefined();
        match req {
            EnvReq::Root { name } => {
                #[cfg(all(feature = "module", feature = "std"))]
                if let Some((m, _)) = crate::nbvm::split_module_name(name) {
                    let s = self.vm_module_scope(m)?;
                    return Ok((self.env_value(s), undef));
                }
                let _ = name;
                let g = self.global_scope.clone();
                Ok((self.env_value(g), undef))
            }
            EnvReq::Child { parent, catch } => {
                let p = self.env_scope(parent)?;
                let c = if catch { p.child_catch() } else { p.child() };
                Ok((self.env_value(c), undef))
            }
            EnvReq::With { parent, obj } => {
                let p = self.env_scope(parent)?;
                if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(self.type_error("Cannot convert undefined or null to object"));
                }
                let o = self.coerce_to_object(obj);
                let c = p.child_with(o);
                Ok((self.env_value(c), undef))
            }
            EnvReq::Clone { src } => {
                let s = self.env_scope(src)?;
                let c = match s.parent() {
                    Some(p) => p.child(),
                    None => Scope::root(),
                };
                for (name, value, is_const) in s.local_bindings() {
                    if is_const {
                        c.declare_const(&name, value);
                    } else {
                        c.declare(&name, value);
                    }
                }
                Ok((self.env_value(c), undef))
            }
            EnvReq::Seed { src } => {
                let s = self.env_scope(src)?;
                let c = s.child();
                for (name, value, is_const) in s.local_bindings() {
                    if is_const {
                        c.declare_const(&name, value);
                    } else {
                        c.declare(&name, value);
                    }
                }
                Ok((self.env_value(c), undef))
            }
            EnvReq::Init {
                env,
                name,
                value,
                mode,
            } => {
                let s = self.env_scope(env)?;
                match mode {
                    ENV_INIT_LET => s.declare(name, value),
                    ENV_INIT_CONST => s.declare_const(name, value),
                    ENV_INIT_TDZ => s.declare(name, NanBox::tdz()),
                    ENV_INIT_TDZ_LEXICAL => {
                        s.declare(name, NanBox::tdz());
                        s.mark_lexical(name);
                    }
                    ENV_INIT_VAR => {
                        if !s.has_local(name) {
                            s.declare(name, undef);
                        }
                    }
                    ENV_INIT_SOFT_CONST => s.declare_soft_const(name, value),
                    ENV_INIT_MARK_CONST => s.mark_const(name),
                    ENV_INIT_VAR_SET => {
                        s.declare(name, value);
                        if s.ptr_eq(&self.global_scope)
                            && let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
                        {
                            self.realm.set_property(g, name, value);
                        }
                    }
                    _ => return Err(ExecError::Unsupported("environment init mode")),
                }
                Ok((undef, undef))
            }
            EnvReq::Load { env, name, strict } => {
                let s = self.env_scope(env)?;
                self.in_env(s, strict, |this| this.read_ident_ref(name))
                    .map(|v| (v, undef))
            }
            EnvReq::Store {
                env,
                name,
                value,
                strict,
            } => {
                let s = self.env_scope(env)?;
                self.in_env(s, strict, |this| this.assign_to_name(name, value))?;
                Ok((undef, undef))
            }
            EnvReq::Typeof { env, name } => {
                let s = self.env_scope(env)?;
                let t = self.in_env(s, false, |this| {
                    if this.current.get(name).is_none()
                        && this.with_binding(name).is_none()
                        && !matches!(name, "undefined" | "NaN" | "Infinity")
                        && !this.global_object_provides(name)
                    {
                        return Ok(this.new_str("undefined"));
                    }
                    let v = this.read_ident_ref(name)?;
                    this.unary(UnaryOp::Typeof, v)
                })?;
                Ok((t, undef))
            }
            EnvReq::Delete { env, name } => {
                let s = self.env_scope(env)?;
                let (r, _) = self.in_env(s, false, |this| Ok(this.delete_identifier(name)))?;
                Ok((NanBox::boolean(r), undef))
            }
            EnvReq::Callee { env, name, strict } => {
                let s = self.env_scope(env)?;
                self.in_env(s, strict, |this| {
                    if let Some(h) = this.with_binding_result(name)? {
                        let f = if this.has_property_proxied(h, name)? {
                            this.read_member(h, name)?
                        } else if this.strict {
                            return Err(this.not_defined(name));
                        } else {
                            NanBox::undefined()
                        };
                        return Ok((f, NanBox::handle(h.to_raw())));
                    }
                    Ok((this.read_ident_lexical(name)?, NanBox::undefined()))
                })
            }
            EnvReq::Ref { env, name, strict } => {
                let s = self.env_scope(env)?;
                self.in_env(s, strict, |this| {
                    if this.in_with_scope()
                        && let Some(h) = this.with_binding_result(name)?
                    {
                        return Ok((NanBox::handle(h.to_raw()), NanBox::undefined()));
                    }
                    if let Some(fr) = this.current.owner_frame(name) {
                        return Ok((this.env_value(fr), NanBox::undefined()));
                    }
                    // Unresolvable: remember whether the global object had the
                    // property when the reference was resolved.
                    let own = this
                        .global_this
                        .as_handle()
                        .map(Handle::from_raw)
                        .is_some_and(|g| this.realm.has_own(g, name));
                    Ok((
                        NanBox::number(f64::from(u8::from(own))),
                        NanBox::undefined(),
                    ))
                })
            }
            EnvReq::RefGet {
                env,
                reference,
                name,
                strict,
            } => {
                let s = self.env_scope(env)?;
                self.in_env(s, strict, |this| this.ref_get(reference, name))
                    .map(|v| (v, undef))
            }
            EnvReq::RefPut {
                env,
                reference,
                name,
                value,
                strict,
            } => {
                let s = self.env_scope(env)?;
                self.in_env(s, strict, |this| this.ref_put(reference, name, value))?;
                Ok((undef, undef))
            }
            EnvReq::Eval {
                env,
                var_env,
                callee,
                this_call,
                args,
                strict,
                this,
                new_target,
                new_target_in_scope,
                param_names,
                home,
                field_init,
                private_names,
                private_keys,
            } => {
                let is_eval = callee.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.native_at(h) == Some(N_EVAL)
                        && self.get_function_realm(h) == self.cur_realm
                });
                if !is_eval {
                    return self
                        .call_with_this(callee, this_call, args)
                        .map(|v| (v, undef));
                }
                let arg0 = args.first().copied().unwrap_or(undef);
                let Some(source) = arg0
                    .as_handle()
                    .and_then(|raw| self.realm.string_bytes(Handle::from_raw(raw)))
                else {
                    return Ok((arg0, undef));
                };
                let lex = self.env_scope(env)?;
                let var = self.env_scope(var_env)?;
                #[cfg(all(feature = "module", feature = "std"))]
                let saved_imports = lex
                    .module_imports()
                    .map(|mi| core::mem::replace(&mut self.module_imports, mi));
                let saved_current = core::mem::replace(&mut self.current, lex);
                let saved_var = core::mem::replace(&mut self.var_scope, var);
                let saved_strict = core::mem::replace(&mut self.strict, strict);
                let saved_this = core::mem::replace(&mut self.this_val, this);
                let saved_nt = core::mem::replace(&mut self.new_target, new_target);
                let saved_nt_scope =
                    core::mem::replace(&mut self.new_target_in_scope, new_target_in_scope);
                let saved_sc_scope = core::mem::replace(&mut self.super_call_in_scope, false);
                let saved_home = self.current_home.take();
                let saved_home_object = core::mem::replace(
                    &mut self.current_home_object,
                    home.and_then(|h| h.as_handle()).map(Handle::from_raw),
                );
                let saved_lexical_home = self.current_lexical_home.take();
                let saved_field_init =
                    core::mem::replace(&mut self.in_field_initializer, field_init);
                let saved_vm_home = core::mem::replace(&mut self.vm_eval_home, home);
                let privates: Vec<(String, NanBox)> = private_names
                    .iter()
                    .zip(private_keys)
                    .map(|(n, k)| (String::from(*n), *k))
                    .collect();
                let saved_privates = core::mem::replace(&mut self.vm_eval_privates, privates);
                let saved_param_names = core::mem::replace(
                    &mut self.eval_param_names,
                    param_names.map(|ns| ns.iter().map(|n| String::from(*n)).collect()),
                );
                let r = self.eval_string(&source, true);
                self.current = saved_current;
                self.var_scope = saved_var;
                self.strict = saved_strict;
                self.this_val = saved_this;
                self.new_target = saved_nt;
                self.new_target_in_scope = saved_nt_scope;
                self.super_call_in_scope = saved_sc_scope;
                self.current_home = saved_home;
                self.current_home_object = saved_home_object;
                self.current_lexical_home = saved_lexical_home;
                self.in_field_initializer = saved_field_init;
                self.vm_eval_home = saved_vm_home;
                self.vm_eval_privates = saved_privates;
                self.eval_param_names = saved_param_names;
                #[cfg(all(feature = "module", feature = "std"))]
                if let Some(mi) = saved_imports {
                    self.module_imports = mi;
                }
                r.map(|v| (v, undef))
            }
            EnvReq::DeclFn {
                var_env,
                name,
                value,
                script,
            } => {
                let vs = self.env_scope(var_env)?;
                self.bind_eval_function(&vs, name, value, !script);
                Ok((undef, undef))
            }
            EnvReq::IsEval { callee } => {
                let is_eval = callee.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.native_at(h) == Some(N_EVAL)
                        && self.get_function_realm(h) == self.cur_realm
                });
                Ok((NanBox::boolean(is_eval), undef))
            }
            EnvReq::MapArgs {
                args_obj,
                env,
                names,
            } => {
                let s = self.env_scope(env)?;
                let Some(obj) = args_obj.as_handle().map(Handle::from_raw) else {
                    return Ok((undef, undef));
                };
                let argc = self
                    .realm
                    .get_property(obj, "length")
                    .and_then(|l| l.as_number())
                    .unwrap_or(0.0) as usize;
                let bound = argc.min(names.len());
                let mut slots = alloc::collections::BTreeMap::new();
                for i in 0..bound {
                    if !names[i + 1..].contains(&names[i]) {
                        slots.insert(i, String::from(names[i]));
                    }
                }
                if !slots.is_empty() {
                    self.arg_maps.insert(
                        obj.to_raw(),
                        ArgMap {
                            scope: s,
                            slots,
                            cells: alloc::collections::BTreeMap::new(),
                        },
                    );
                }
                Ok((undef, undef))
            }
        }
    }

    /// The newest function table of the VM run this interpreter hosts: the
    /// running one, or the one eval code extended from it.
    /// (Every table of one interpreter is a prefix of the next: eval code and
    /// dynamically imported modules only ever append to the longest.)
    pub(crate) fn vm_newest_table(&self) -> Option<alloc::rc::Rc<[crate::nbvm::FnProto]>> {
        let mut best = self.vm_table.clone();
        let mut consider = |t: alloc::rc::Rc<[crate::nbvm::FnProto]>| {
            if best.as_ref().is_none_or(|b| t.len() > b.len()) {
                best = Some(t);
            }
        };
        if let Some(ext) = &self.vm_ext_table {
            consider(alloc::rc::Rc::clone(ext));
        }
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some(m) = self.module_vm_table() {
            consider(m);
        }
        best
    }

    /// Eval code `program` (a direct or indirect eval's, or a `$262.evalScript`
    /// script's when `script`) compiled to and run on the VM, when this
    /// interpreter hosts a VM run — with the eval's environments already set up
    /// (`current` its lexical, `eval_var_scope`/`current` its variable one) and
    /// `new_target` the `new.target` in scope, if any. `None` when there is no
    /// VM run, or the code does not compile (the caller tree-walks it then).
    pub(crate) fn vm_eval_program(
        &mut self,
        program: &'a Program,
        strict: bool,
        new_target: Option<NanBox>,
        script: bool,
    ) -> Option<Result<NanBox, ExecError>> {
        let mut flags = 0u8;
        if script {
            flags |= crate::nbvm::EVAL_SCRIPT;
        }
        if new_target.is_some() {
            flags |= crate::nbvm::EVAL_NEW_TARGET;
        }
        let home = self.vm_eval_home.filter(|_| !script);
        if home.is_some() {
            flags |= crate::nbvm::EVAL_HOME;
        }
        if self.in_field_initializer && !script {
            flags |= crate::nbvm::EVAL_FIELD_INIT;
        }
        let privates: Vec<(String, NanBox)> = if script {
            Vec::new()
        } else {
            self.vm_eval_privates.clone()
        };
        let names: Vec<&str> = privates.iter().map(|(n, _)| n.as_str()).collect();
        let (table, proto) = self.vm_eval_proto(program, strict, flags, &names)?;
        // EvalDeclarationInstantiation's bindings (the VM code instantiates the
        // function declarations itself).
        let saved_gc = core::mem::replace(&mut self.gc_ok, false);
        let saved_src = core::mem::replace(&mut self.src, &program.source);
        let saved_epoch = self.eval_site_epoch;
        self.eval_site_counter += 1;
        self.eval_site_epoch = self.eval_site_counter;
        // A Script's GlobalDeclarationInstantiation checks.
        let checked = if self.script_eval_globals && self.var_scope.ptr_eq(&self.global_scope) {
            self.global_declaration_checks(program)
        } else {
            Ok(())
        };
        self.hoist_skip_fns = true;
        let hoisted = checked.and_then(|()| self.hoist_with_kind(&program.body, true, true));
        self.hoist_skip_fns = false;
        let result = hoisted.and_then(|()| {
            let lex = self.env_value(self.current.clone());
            let var = self.env_value(self.var_scope.clone());
            let mut caps = alloc::vec![lex, var, self.this_val];
            caps.extend(new_target);
            caps.extend(home);
            if flags & crate::nbvm::EVAL_FIELD_INIT != 0 {
                caps.push(NanBox::boolean(true));
            }
            caps.extend(privates.iter().map(|(_, k)| *k));
            let table = self.vm_newest_table().unwrap_or(table);
            crate::nbvm::run_eval_code(self, &table, &proto, &caps).map_err(vm_to_exec)
        });
        self.eval_site_epoch = saved_epoch;
        self.src = saved_src;
        self.gc_ok = saved_gc;
        Some(result)
    }

    /// CreateDynamicFunction's function, built on the VM from the parsed
    /// `(function anonymous(…) {…})` wrapper `program` in the global
    /// environment; `None` when there is no VM run or it does not compile.
    pub(crate) fn vm_dynamic_function(
        &mut self,
        program: &'a Program,
        strict: bool,
    ) -> Option<Result<NanBox, ExecError>> {
        let (table, proto) = self.vm_eval_proto(program, strict, crate::nbvm::EVAL_DYN_FN, &[])?;
        let g = self.global_scope.clone();
        let env = self.env_value(g);
        let caps = [env, env, self.global_this];
        Some(crate::nbvm::run_eval_code(self, &table, &proto, &caps).map_err(vm_to_exec))
    }

    /// Eval code compiled for the VM (cached per program and flags), with the
    /// newest function table — extended when the code defines functions.
    fn vm_eval_proto(
        &mut self,
        program: &'a Program,
        strict: bool,
        flags: u8,
        privates: &[&str],
    ) -> Option<(
        alloc::rc::Rc<[crate::nbvm::FnProto]>,
        alloc::rc::Rc<crate::nbvm::FnProto>,
    )> {
        let table = self.vm_newest_table()?;
        let key = (
            core::ptr::from_ref(program) as usize,
            u16::from(flags) | (u16::from(strict) << 8),
            privates.join("\n"),
        );
        let proto = match self.vm_eval_cache.get(&key) {
            Some(p) => alloc::rc::Rc::clone(p),
            None => {
                // Without nested functions the body needs no table slots.
                let mut scratch = Vec::new();
                let first =
                    crate::nbvm::compile_eval_code(program, &mut scratch, strict, flags, privates);
                let proto = match first {
                    Err(_) => return None,
                    Ok(p) if scratch.is_empty() => p,
                    Ok(_) => {
                        let mut full: Vec<crate::nbvm::FnProto> = table.to_vec();
                        let base = full.len();
                        let p = crate::nbvm::compile_eval_code(
                            program, &mut full, strict, flags, privates,
                        )
                        .ok()?;
                        crate::nbvm::resolve_source_text(&mut full[base..], &program.source);
                        // One growth chain with scripts and dynamic imports.
                        self.install_vm_table(full.into());
                        p
                    }
                };
                let rc = alloc::rc::Rc::new(proto);
                self.vm_eval_cache.insert(key, alloc::rc::Rc::clone(&rc));
                rc
            }
        };
        let table = self.vm_newest_table().unwrap_or(table);
        Some((table, proto))
    }

    /// Binds eval code's (`eval_code`) or a Script's function declaration
    /// `name` = `value` in the variable environment `vs` (CreateGlobalFunctionBinding
    /// at the global one).
    fn bind_eval_function(&mut self, vs: &Scope, name: &str, value: NanBox, eval_code: bool) {
        let at_global = vs.ptr_eq(&self.global_scope);
        if eval_code && !at_global {
            vs.declare_deletable(name, value);
        } else {
            vs.declare(name, value);
        }
        if at_global && let Some(g) = self.global_this.as_handle().map(Handle::from_raw) {
            let deletable = eval_code;
            let has_own = self.realm.has_own(g, name) || self.realm.accessor(g, name).is_some();
            let redefine_attrs = !has_own || !self.realm.property_is_non_configurable(g, name);
            if redefine_attrs {
                self.realm.clear_accessor(g, name);
            }
            self.realm.force_set_property(g, name, value);
            if redefine_attrs {
                self.realm.clear_readonly_property(g, name);
                self.realm.clear_hidden_property(g, name);
                if deletable {
                    self.realm.clear_non_configurable_property(g, name);
                } else {
                    self.realm.set_non_configurable_property(g, name);
                }
            }
        }
    }

    /// Runs `f` with `scope` as the current lexical environment in strict or
    /// sloppy code.
    fn in_env<T>(
        &mut self,
        scope: Scope,
        strict: bool,
        f: impl FnOnce(&mut Self) -> Result<T, ExecError>,
    ) -> Result<T, ExecError> {
        // Module code's environments see its imports.
        #[cfg(all(feature = "module", feature = "std"))]
        let saved_imports = scope
            .module_imports()
            .map(|mi| core::mem::replace(&mut self.module_imports, mi));
        let saved = core::mem::replace(&mut self.current, scope);
        let saved_strict = core::mem::replace(&mut self.strict, strict);
        let r = f(self);
        self.current = saved;
        self.strict = saved_strict;
        #[cfg(all(feature = "module", feature = "std"))]
        if let Some(mi) = saved_imports {
            self.module_imports = mi;
        }
        r
    }

    /// `GetValue` of a reference resolved by [`EnvReq::Ref`].
    fn ref_get(&mut self, reference: NanBox, name: &str) -> Result<NanBox, ExecError> {
        if let Some(h) = reference.as_handle().map(Handle::from_raw) {
            if let Some(fr) = self.realm.env_at(h) {
                return match fr.has_local(name).then(|| fr.get(name)).flatten() {
                    Some(v) if v.is_tdz() => Err(self.uninitialized(name)),
                    Some(v) => Ok(v),
                    None => self.read_ident_ref(name),
                };
            }
            // An object environment record's binding, re-checked by
            // GetBindingValue's `HasProperty`.
            if self.has_property_proxied(h, name)? {
                return self.read_member(h, name);
            }
            if self.strict {
                return Err(self.not_defined(name));
            }
            return Ok(NanBox::undefined());
        }
        self.read_ident_lexical(name)
    }

    /// `PutValue` of a reference resolved by [`EnvReq::Ref`].
    fn ref_put(&mut self, reference: NanBox, name: &str, value: NanBox) -> Result<(), ExecError> {
        if let Some(h) = reference.as_handle().map(Handle::from_raw) {
            if let Some(fr) = self.realm.env_at(h) {
                if !fr.has_local(name) {
                    return self.assign_to_name(name, value);
                }
                if fr.get(name).is_some_and(|v| v.is_tdz()) {
                    return Err(self.uninitialized(name));
                }
                if fr.is_const(name) {
                    return Err(self.const_assign());
                }
                if fr.is_soft_const(name) {
                    if self.strict {
                        return Err(self.const_assign());
                    }
                    return Ok(());
                }
                fr.declare(name, value);
                if fr.ptr_eq(&self.global_scope) {
                    self.sync_global_var(name, value);
                }
                return Ok(());
            }
            // SetMutableBinding of an object environment record re-checks
            // `HasProperty`: a strict write to a vanished binding throws.
            if !self.has_property_proxied(h, name)? && self.strict {
                return Err(self.not_defined(name));
            }
            let key = self.new_str(name);
            self.assign_member_value(h, key, value)?;
            return Ok(());
        }
        // Unresolvable when resolved (a binding the right-hand side created —
        // a direct eval's `var` — still takes the write, as in the
        // tree-walker).
        if self.current.set(name, value) {
            self.sync_global_var(name, value);
            return Ok(());
        }
        let was_own = reference.as_number().is_some_and(|n| n != 0.0);
        if let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
            && was_own
            && (self.realm.has_own(g, name) || !self.strict)
        {
            if self.realm.property_is_readonly(g, name) {
                if self.strict {
                    let m = self.new_str(&alloc::format!(
                        "Cannot assign to read only property '{name}'"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                return Ok(());
            }
            let key = self.new_str(name);
            return self.assign_member_value(g, key, value);
        }
        if self.strict {
            return Err(self.not_defined(name));
        }
        self.declare_sloppy_global(name, value)
    }
}
