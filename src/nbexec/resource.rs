//! The ES2025 explicit-resource-management classes (`DisposableStack`,
//! `AsyncDisposableStack`) and the `ShadowRealm` constructor.
//!
//! These are modeled with the same "branded native object" pattern the `Intl`
//! services use (see `intl_fmt.rs`): a real `.prototype` object carrying
//! first-class, brand-checked methods (each a `new_bound_native` whose target is
//! the method-name string), a hidden internal-slot marker on every instance, and
//! a `Symbol.toStringTag`.
//!
//! What is intentionally *not* here: the `using` / `await using` block-scope
//! disposal *runtime* (a statement-level feature owned by the execution model).
//! Only the classes are provided, so `new DisposableStack()` and its methods work
//! as first-class values.

use super::*;

/// Hidden internal-slot brand marking a `DisposableStack` instance (and gating
/// its prototype methods' RequireInternalSlot checks).
const DSTACK_BRAND: &str = "\u{0}dstack";
/// Hidden slot holding the disposer-entry array of a `DisposableStack`. Each
/// entry is a two-element array `[value, method]` (the dispose receiver and its
/// callback); entries are disposed in reverse (LIFO) order.
const DSTACK_LIST: &str = "\u{0}dstack_list";
/// Hidden boolean slot: whether a `DisposableStack` has been disposed/moved.
const DSTACK_DISPOSED: &str = "\u{0}dstack_disposed";

/// Hidden internal-slot brand marking an `AsyncDisposableStack` instance.
const ADSTACK_BRAND: &str = "\u{0}adstack";
/// Hidden slot holding an `AsyncDisposableStack`'s disposer-entry array.
const ADSTACK_LIST: &str = "\u{0}adstack_list";
/// Hidden boolean slot: whether an `AsyncDisposableStack` has been disposed/moved.
const ADSTACK_DISPOSED: &str = "\u{0}adstack_disposed";

/// Hidden slot on a `$262.createRealm()` realm object: the index of its distinct
/// global environment in `Interp::created_realms`.
const CREATED_REALM_IDX: &str = "\u{0}created_realm";

/// Hidden internal-slot brand marking a `ShadowRealm` instance.
const SHADOWREALM_BRAND: &str = "\u{0}shadowrealm";
/// Hidden slot on a `ShadowRealm` instance: the index of its persistent global
/// scope in `Interp::shadow_realm_scopes`.
const SHADOWREALM_SCOPE_IDX: &str = "\u{0}shadowrealm_scope";

/// `DisposableStack.prototype` method names (each a brand-checked bound native).
const DSTACK_METHODS: &[&str] = &["use", "adopt", "defer", "dispose", "move"];
/// `AsyncDisposableStack.prototype` method names.
const ADSTACK_METHODS: &[&str] = &["use", "adopt", "defer", "disposeAsync", "move"];

impl<'a> Interp<'a> {
    /// Installs the `DisposableStack`, `AsyncDisposableStack`, and `ShadowRealm`
    /// global constructors with their real `.prototype` objects. Invoked from
    /// realm setup after `Object.prototype` exists.
    pub(crate) fn install_resource_management(&mut self) {
        self.install_suppressed_error();
        self.install_disposable_stack(false);
        self.install_disposable_stack(true);
        self.install_shadow_realm();
    }

    /// Installs the `SuppressedError` constructor (ES2025) with a `.prototype`
    /// inheriting `Error.prototype` (so `e instanceof SuppressedError` and the
    /// inherited `Error` behavior work). Its constructor inherits the `Error`
    /// constructor's static side.
    fn install_suppressed_error(&mut self) {
        let ctor = self.new_named_native("SuppressedError", N_SUPPRESSED_ERROR);
        self.realm
            .set_hidden_property(ctor, "length", NanBox::number(3.0));
        self.realm.set_readonly_property(ctor, "length");
        // `Object.getPrototypeOf(SuppressedError) === Error`.
        if let Some(error_ctor) = self
            .current
            .get("Error")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_native_proto(ctor, error_ctor);
        }
        // `SuppressedError.prototype` inherits `Error.prototype`.
        let error_proto = self
            .current
            .get("Error")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            .or_else(|| self.object_prototype());
        let proto = self.realm.new_object_with_proto(error_proto);
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        let name = self.new_str("SuppressedError");
        self.realm.set_property(proto, "name", name);
        self.realm.mark_hidden(proto, "name");
        let msg = self.new_str("");
        self.realm.set_property(proto, "message", msg);
        self.realm.mark_hidden(proto, "message");
        self.realm
            .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(ctor, "prototype");
        self.realm.set_readonly_property(ctor, "prototype");
        self.realm.set_non_configurable_property(ctor, "prototype");
        self.current
            .declare("SuppressedError", NanBox::handle(ctor.to_raw()));
    }

    /// `new SuppressedError(error, suppressed, message)` / `SuppressedError(...)`.
    /// Builds an instance linked to `SuppressedError.prototype` with
    /// non-enumerable `error`/`suppressed` data properties (and `message` when the
    /// `message` argument is not `undefined`).
    pub(crate) fn construct_suppressed_error(
        &mut self,
        args: &[NanBox],
        callee: NanBox,
        new_target: NanBox,
    ) -> Result<NanBox, ExecError> {
        let error = args.first().copied().unwrap_or(NanBox::undefined());
        let suppressed = args.get(1).copied().unwrap_or(NanBox::undefined());
        let message = args.get(2).copied().unwrap_or(NanBox::undefined());
        let obj = self.realm.new_object();
        let default = self.intrinsic_proto("SuppressedError");
        if let Some(proto) = self.resource_instance_proto(new_target, callee, default)? {
            self.realm.set_object_proto(obj, Some(proto));
        }
        if !matches!(message.unpack(), Unpacked::Undefined) {
            let s = self.coerce_to_string(message)?;
            let m = self.new_str(&s);
            self.realm.set_property(obj, "message", m);
            self.realm.mark_hidden(obj, "message");
        }
        self.realm.set_property(obj, "error", error);
        self.realm.mark_hidden(obj, "error");
        self.realm.set_property(obj, "suppressed", suppressed);
        self.realm.mark_hidden(obj, "suppressed");
        // `SuppressedError` is an Error subclass: stamp the `[[ErrorData]]` brand
        // (hidden; see `ERROR_DATA`) so `Error.isError` recognizes it.
        self.realm
            .set_hidden_property(obj, ERROR_DATA, NanBox::boolean(true));
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Builds one of the two disposable-stack constructors (`is_async` selects
    /// `AsyncDisposableStack`) and its prototype.
    fn install_disposable_stack(&mut self, is_async: bool) {
        let (name, ctor_id, proto_id, disposed_id, methods, tag, dispose_sym_name) = if is_async {
            (
                "AsyncDisposableStack",
                N_ASYNC_DISPOSABLE_STACK,
                N_ASYNC_DISPOSABLE_STACK_PROTO,
                N_ASYNC_DISPOSABLE_STACK_DISPOSED,
                ADSTACK_METHODS,
                "AsyncDisposableStack",
                "asyncDispose",
            )
        } else {
            (
                "DisposableStack",
                N_DISPOSABLE_STACK,
                N_DISPOSABLE_STACK_PROTO,
                N_DISPOSABLE_STACK_DISPOSED,
                DSTACK_METHODS,
                "DisposableStack",
                "dispose",
            )
        };
        let ctor = self.new_named_native(name, ctor_id);
        self.realm
            .set_hidden_property(ctor, "length", NanBox::number(0.0));
        self.realm.set_readonly_property(ctor, "length");
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        // First-class, brand-checked prototype methods.
        for &m in methods {
            let name_h = self.realm.new_string(m);
            let f = self.realm.new_bound_native(proto_id, name_h);
            let arity = match m {
                "adopt" => 2,
                "use" | "defer" => 1,
                _ => 0,
            };
            self.install_fn_name_length(f, m, arity);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        // The `disposed` getter accessor (`length` 0, `name` "get disposed").
        let disposed_get = self.realm.new_native(disposed_id);
        self.install_fn_name_length(disposed_get, "get disposed", 0);
        self.realm.define_accessor(
            proto,
            "disposed",
            NanBox::handle(disposed_get.to_raw()),
            NanBox::undefined(),
        );
        self.realm.mark_hidden(proto, "disposed");
        // `[Symbol.dispose]` / `[Symbol.asyncDispose]` aliases `dispose` /
        // `disposeAsync` — the *same* function object per spec.
        let dispose_method_name = if is_async { "disposeAsync" } else { "dispose" };
        if let Some(dm) = self.realm.get_property(proto, dispose_method_name) {
            let sym = self.well_known_symbol(dispose_sym_name);
            let key = self.member_key(sym);
            self.realm.set_property(proto, &key, dm);
            self.realm.mark_hidden(proto, &key);
        }
        // `prototype[Symbol.toStringTag]`.
        self.install_to_string_tag(proto, tag);
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.realm
            .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(ctor, "prototype");
        self.realm.set_readonly_property(ctor, "prototype");
        self.realm.set_non_configurable_property(ctor, "prototype");
        self.current.declare(name, NanBox::handle(ctor.to_raw()));
    }

    /// Builds the `ShadowRealm` constructor and its prototype (`evaluate`,
    /// `importValue`, `Symbol.toStringTag`).
    fn install_shadow_realm(&mut self) {
        let ctor = self.new_named_native("ShadowRealm", N_SHADOW_REALM);
        self.realm
            .set_hidden_property(ctor, "length", NanBox::number(0.0));
        self.realm.set_readonly_property(ctor, "length");
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &(m, arity) in &[("evaluate", 1u32), ("importValue", 2)] {
            let name_h = self.realm.new_string(m);
            let f = self.realm.new_bound_native(N_SHADOW_REALM_PROTO, name_h);
            self.install_fn_name_length(f, m, arity);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        self.install_to_string_tag(proto, "ShadowRealm");
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.realm
            .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(ctor, "prototype");
        self.realm.set_readonly_property(ctor, "prototype");
        self.realm.set_non_configurable_property(ctor, "prototype");
        self.current
            .declare("ShadowRealm", NanBox::handle(ctor.to_raw()));
    }

    /// `new DisposableStack()` / `new AsyncDisposableStack()` — a fresh branded
    /// instance with an empty disposer list and `disposed = false`. `new_target`
    /// (the `Reflect.construct`/subclass newTarget, else the constructor itself)
    /// supplies the `[[Prototype]]`.
    pub(crate) fn construct_disposable_stack(
        &mut self,
        is_async: bool,
        callee: NanBox,
        new_target: NanBox,
    ) -> Result<NanBox, ExecError> {
        let (ctor_name, brand, list_key, disposed_key) = if is_async {
            (
                "AsyncDisposableStack",
                ADSTACK_BRAND,
                ADSTACK_LIST,
                ADSTACK_DISPOSED,
            )
        } else {
            (
                "DisposableStack",
                DSTACK_BRAND,
                DSTACK_LIST,
                DSTACK_DISPOSED,
            )
        };
        let obj = self.realm.new_object();
        let default = self.intrinsic_proto(ctor_name);
        if let Some(proto) = self.resource_instance_proto(new_target, callee, default)? {
            self.realm.set_object_proto(obj, Some(proto));
        }
        self.realm
            .set_hidden_property(obj, brand, NanBox::boolean(true));
        let list = self.realm.new_array(Vec::new());
        self.realm
            .set_hidden_property(obj, list_key, NanBox::handle(list.to_raw()));
        self.realm
            .set_hidden_property(obj, disposed_key, NanBox::boolean(false));
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// `get DisposableStack.prototype.disposed` /
    /// `get AsyncDisposableStack.prototype.disposed`: a brand-checked boolean
    /// reading the instance's `disposed` slot.
    pub(crate) fn dstack_disposed_getter(&mut self, is_async: bool) -> Result<NanBox, ExecError> {
        let (brand, disposed_key, what) = if is_async {
            (
                ADSTACK_BRAND,
                ADSTACK_DISPOSED,
                "get AsyncDisposableStack.prototype.disposed",
            )
        } else {
            (
                DSTACK_BRAND,
                DSTACK_DISPOSED,
                "get DisposableStack.prototype.disposed",
            )
        };
        let obj = self.require_dstack(brand, what)?;
        Ok(NanBox::boolean(self.dstack_is_disposed(obj, disposed_key)))
    }

    /// `GetPrototypeFromConstructor(newTarget, default)` done spec-correctly for
    /// these classes: reads `Get(newTarget, "prototype")` (invoking an accessor),
    /// and uses it only when it is a real (non-callable) Object; otherwise the
    /// intrinsic `default`. This differs from the shared `instance_proto` helper,
    /// which falls back to a *function's* own `.prototype` object when `newTarget`
    /// is a plain function whose `prototype` was set to a non-object.
    fn resource_instance_proto(
        &mut self,
        new_target: NanBox,
        callee: NanBox,
        default: Option<Handle>,
    ) -> Result<Option<Handle>, ExecError> {
        // The common `new C()` case: newTarget is the callee → the default.
        if new_target.as_handle().is_none() || new_target.as_handle() == callee.as_handle() {
            return Ok(default);
        }
        let Some(nt) = new_target.as_handle().map(Handle::from_raw) else {
            return Ok(default);
        };
        // `Get(newTarget, "prototype")`. An *own accessor* `prototype` (e.g. a bound
        // function with a `prototype` getter) is invoked via `read_member_value`; an
        // own *data* `prototype` (e.g. `nt.prototype = undefined`) is read straight
        // from the aux store; otherwise `read_member` yields the function's
        // intrinsic `.prototype`. (`read_member` special-cases a function's
        // `prototype` to its intrinsic *before* the aux store, so it cannot observe a
        // `nt.prototype = <non-object>` override — hence the explicit checks here.)
        let proto = if self.realm.accessor(nt, "prototype").is_some() {
            let key = self.new_str("prototype");
            self.read_member_value(nt, key)?
        } else if self.realm.has_own(nt, "prototype") {
            self.realm
                .get_property(nt, "prototype")
                .unwrap_or(NanBox::undefined())
        } else {
            self.read_member(nt, "prototype")?
        };
        // Use it only when `Type(proto)` is Object (`is_object_value` excludes
        // string/symbol/bigint primitives; numbers/booleans/undefined/null are not
        // handles); otherwise the intrinsic default.
        if self.is_object_value(proto) {
            return Ok(proto.as_handle().map(Handle::from_raw));
        }
        // Non-object `prototype` → intrinsic default from
        // `GetFunctionRealm(newTarget)` (spec step 4): a cross-realm newTarget uses
        // its own realm's intrinsic.
        Ok(self.realm_default_proto(default, nt))
    }

    /// Stamps the disposable-stack internal slots onto an instance built by a
    /// subclass's `super()` (`class S extends DisposableStack {}`): the brand, an
    /// empty disposer list, and `disposed = false`.
    pub(crate) fn init_disposable_stack_super(&mut self, is_async: bool, instance: Handle) {
        let (brand, list_key, disposed_key) = if is_async {
            (ADSTACK_BRAND, ADSTACK_LIST, ADSTACK_DISPOSED)
        } else {
            (DSTACK_BRAND, DSTACK_LIST, DSTACK_DISPOSED)
        };
        self.realm
            .set_hidden_property(instance, brand, NanBox::boolean(true));
        let list = self.realm.new_array(Vec::new());
        self.realm
            .set_hidden_property(instance, list_key, NanBox::handle(list.to_raw()));
        self.realm
            .set_hidden_property(instance, disposed_key, NanBox::boolean(false));
    }

    /// `$262.createRealm()` — the Test262 cross-realm host hook. Builds a second
    /// global environment with a *distinct* set of intrinsics on the shared heap
    /// (its `Array`/`Object`/`TypeError`/… are separate heap cells from the current
    /// realm's) and returns a `$262`-shaped realm object: `{ global, evalScript,
    /// createRealm }`. `.global` is the new global object (its own constructors),
    /// so `other.Array !== Array`, `[] instanceof other.Array === false`, etc.
    ///
    /// The heap, atom table, well-known symbols, and a handful of lazily-shared
    /// prototypes (iterator/regexp/intl) are shared with the host realm — a
    /// best-effort model that lands the large cross-realm *identity* bulk. The
    /// deep `proto-from-ctor-realm` subset (which needs per-function realm tagging
    /// so `GetPrototypeFromConstructor` picks the *newTarget's* realm's intrinsic)
    /// is intentionally left for a follow-up.
    /// Builds a fresh, genuinely distinct realm environment (its own
    /// `install_globals` populating a brand-new root scope, so its intrinsics are
    /// separate heap cells from every other realm's), records it in
    /// `created_realms`, and returns its index. The active realm's environment
    /// (scope, `globalThis`, intrinsic prototype pointers) is saved and restored,
    /// so the caller keeps running in its original realm. Shared by
    /// `$262.createRealm` and `$262.agent.start` (each worker gets its own realm).
    pub(crate) fn create_realm_env(&mut self) -> usize {
        // Save the active realm's environment.
        let saved_current = self.current.clone();
        let saved_global_scope = self.global_scope.clone();
        let saved_var_scope = self.var_scope.clone();
        let saved_global_this = self.global_this;
        let saved_this = self.this_val;
        let saved_new_target = self.new_target;
        let saved_strict = self.strict;
        let saved_intrinsics = self.realm.intrinsics_snapshot();

        // Install a fresh global environment into a brand-new root scope. Because
        // `install_globals` allocates a fresh cell for every intrinsic, the new
        // realm's globals are distinct handles from the current realm's.
        let root = Scope::root();
        self.current = root.clone();
        self.global_scope = root.clone();
        self.var_scope = root.clone();
        self.this_val = NanBox::undefined();
        self.new_target = NanBox::undefined();
        self.strict = false;
        // The `Intl` service `.prototype` cache is keyed by ctor id, so it cannot
        // hold two realms' prototypes at once — clear it (stashing the active realm's)
        // so the child's `install_intl_prototypes` builds and wires its own distinct
        // `%Intl.X.prototype%` objects onto the child's constructors rather than
        // short-circuiting on the parent realm's cached entries.
        let saved_intl_protos = self.realm.take_intl_protos();
        self.install_globals();
        let new_global_this = self.global_this;
        let new_global_scope = self.global_scope.clone();
        let new_intrinsics = self.realm.intrinsics_snapshot();
        // Capture the child's freshly-built Intl prototypes, then restore the active
        // realm's cache so its own `Intl.X.prototype` reads stay correct.
        let new_intl_protos = self.realm.replace_intl_protos(saved_intl_protos);

        // Restore the active realm's environment — including its intrinsic
        // prototype pointers, so `[]`/`{}` literals created afterward in the
        // original realm keep the original realm's prototypes.
        self.current = saved_current;
        self.global_scope = saved_global_scope;
        self.var_scope = saved_var_scope;
        self.global_this = saved_global_this;
        self.this_val = saved_this;
        self.new_target = saved_new_target;
        self.strict = saved_strict;
        self.realm.restore_intrinsics(saved_intrinsics);

        let idx = self.created_realms.len();
        self.created_realms.push(CreatedRealm {
            global_scope: new_global_scope,
            global_this: new_global_this,
            intrinsics: new_intrinsics,
            intl_protos: new_intl_protos,
        });
        // `GetFunctionRealm` tagging: every callable reachable from this realm's
        // global (its `Object`/`Function`/`Array`/… constructors AND their
        // prototype methods / accessor getters+setters / static methods, each a
        // distinct heap cell from every other realm's) belongs to realm `idx`.
        // Recording them lets `GetPrototypeFromConstructor` pick *this* realm's
        // intrinsic default for a cross-realm `newTarget`, lets a cross-realm
        // intrinsic method throw *this* realm's `%TypeError%` (see `make_error`),
        // and lets `ArraySpeciesCreate` recognize `realmC.[[%Array%]]`.
        //
        // A *bounded breadth-first* walk over own data properties (recursing into
        // object values) and accessor getters/setters, guarded by a `visited` set,
        // so the shared iterator/regexp/intl prototypes and any cycles terminate.
        if let Some(gt) = new_global_this.as_handle().map(Handle::from_raw) {
            let mut visited = alloc::collections::BTreeSet::new();
            let mut queue = alloc::vec::Vec::new();
            queue.push((gt, 0u32));
            visited.insert(gt.to_raw());
            while let Some((obj, depth)) = queue.pop() {
                if depth > 3 {
                    continue;
                }
                // A native cell (a constructor such as `String`/`Array`) stores its
                // own properties — including `prototype` and static methods — in an
                // *auxiliary* object, not the cell itself, so both key stores must
                // be walked to reach e.g. `String.prototype` (and thence its methods).
                let mut keys = self.realm.object_all_keys(obj);
                keys.extend(self.realm.aux_all_keys(obj));
                for key in keys {
                    // Accessor (getter/setter) — tag both halves; they never carry
                    // a nested object graph to recurse into.
                    if let Some((g, s)) = self.realm.accessor(obj, &key) {
                        for f in [g, s] {
                            if let Some(h) = f.as_handle().map(Handle::from_raw)
                                && self.is_callable(h)
                            {
                                self.fn_realm.entry(h.to_raw()).or_insert(idx);
                            }
                        }
                        continue;
                    }
                    let Some(v) = self.realm.get_property(obj, &key) else {
                        continue;
                    };
                    let Some(h) = v.as_handle().map(Handle::from_raw) else {
                        continue;
                    };
                    if self.is_callable(h) {
                        self.fn_realm.entry(h.to_raw()).or_insert(idx);
                    }
                    // Recurse into object values (a constructor's `.prototype`, a
                    // namespace object such as `Math`/`JSON`/`Reflect`, …) once, so
                    // prototype methods are tagged. Bounded by `visited` + depth.
                    if self.is_object_value(v) && visited.insert(h.to_raw()) {
                        queue.push((h, depth + 1));
                    }
                }
            }
        }
        idx
    }

    pub(crate) fn create_realm(&mut self) -> Result<NanBox, ExecError> {
        let idx = self.create_realm_env();
        let new_global_this = self.created_realms[idx].global_this;
        let realm_obj = self.realm.new_object();
        self.realm
            .set_hidden_property(realm_obj, CREATED_REALM_IDX, NanBox::number(idx as f64));
        self.realm
            .set_property(realm_obj, "global", new_global_this);
        // `.evalScript(src)` — a plain native dispatched with `this` = realm_obj.
        let eval_fn = self.new_named_native("evalScript", N_262_EVAL_SCRIPT);
        self.realm
            .set_property(realm_obj, "evalScript", NanBox::handle(eval_fn.to_raw()));
        // Nested `.createRealm()` (some tests build realms recursively).
        let cr_fn = self.new_named_native("createRealm", N_262_CREATE_REALM);
        self.realm
            .set_property(realm_obj, "createRealm", NanBox::handle(cr_fn.to_raw()));
        Ok(NanBox::handle(realm_obj.to_raw()))
    }

    /// `$262.AbstractModuleSource` — build the `%AbstractModuleSource%` intrinsic
    /// (source-phase-imports proposal §28) and return its constructor. Only the
    /// *intrinsic shape* is materialized (no loadable module-source objects exist
    /// yet): the abstract constructor (throws on `[[Call]]`/`[[Construct]]`,
    /// `name` = "AbstractModuleSource", `length` = 0, `[[Prototype]]` =
    /// %FunctionPrototype%) and `%AbstractModuleSource.prototype%` (its
    /// `[[Prototype]]` is %Object.prototype%, a `constructor` data property back to
    /// the ctor, and a `@@toStringTag` accessor getter returning the receiver's
    /// `[[ModuleSourceClassName]]` — always `undefined` here since no receiver
    /// carries that slot).
    pub(crate) fn abstract_module_source(&mut self) -> Result<NanBox, ExecError> {
        // The abstract constructor: `new`/call both throw a TypeError.
        let ctor = self.realm.new_native(N_ABSTRACT_MODULE_SOURCE_CTOR);
        self.install_fn_name_length(ctor, "AbstractModuleSource", 0);
        // `%AbstractModuleSource.prototype%` — a plain object (so its
        // `[[Prototype]]` is %Object.prototype%).
        let proto = self.realm.new_object();
        // `%AbstractModuleSource.prototype%.constructor` — a data property
        // { writable: true, enumerable: false, configurable: true }.
        self.realm
            .set_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.realm.mark_hidden(proto, "constructor");
        // `get %AbstractModuleSource%.prototype[@@toStringTag]` — an accessor with
        // no setter, { enumerable: false, configurable: true }.
        let tag_get = self.realm.new_native(N_ABSTRACT_MODULE_SOURCE_TAG_GET);
        self.install_fn_name_length(tag_get, "get [Symbol.toStringTag]", 0);
        let tag_sym = self.well_known_symbol("toStringTag");
        let tag_key = self.member_key(tag_sym);
        self.realm.define_accessor(
            proto,
            &tag_key,
            NanBox::handle(tag_get.to_raw()),
            NanBox::undefined(),
        );
        self.realm.mark_hidden(proto, &tag_key);
        // `%AbstractModuleSource%.prototype` — { writable: false, enumerable:
        // false, configurable: false }.
        self.install_fn_prototype(ctor, proto, false);
        Ok(NanBox::handle(ctor.to_raw()))
    }

    /// `realm.evalScript(src)` — evaluate `src` in the receiver realm's global
    /// environment (swapping in its scope + intrinsics), returning the completion
    /// value. `this` must be a realm object built by [`create_realm`].
    pub(crate) fn eval_script_in_realm(&mut self, src: NanBox) -> Result<NanBox, ExecError> {
        let idx = self
            .this_val
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.get_property(h, CREATED_REALM_IDX))
            .and_then(|v| v.as_number())
            .map(|n| n as usize);
        let Some(idx) = idx.filter(|i| *i < self.created_realms.len()) else {
            return Err(self.type_error("evalScript called on an object that is not a realm"));
        };
        let source = self.coerce_to_string(src)?;
        self.eval_source_in_realm(idx, &source)
    }

    /// `$262.evalScript(src)` — evaluate `src` as a **Script** in the *current*
    /// realm's global environment. Unlike an indirect `eval` (which runs its
    /// lexical declarations in a fresh, discarded declarative environment), a
    /// script's top-level `let`/`const`/`class` become persistent global lexical
    /// bindings, and a sloppy script's `var`/function bindings persist too. A
    /// parse failure throws a SyntaxError.
    pub(crate) fn eval_script_current_realm(&mut self, src: NanBox) -> Result<NanBox, ExecError> {
        let source = self.coerce_to_string(src)?;
        if self.eval_depth >= self.realm.limits.max_eval_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(msg))));
        }
        // Parse first so a SyntaxError surfaces before any environment swap.
        let program = self.parse_eval_program(&source, false, false, false, false, &[], false)?;

        let saved_current = self.current.clone();
        let saved_var_scope = self.var_scope.clone();
        let saved_this = self.this_val;
        let saved_new_target = self.new_target;
        let saved_strict = self.strict;

        self.strict = has_use_strict(&program.body);
        // A strict script gets its own child so its lexical declarations don't
        // leak into the persistent global scope; a sloppy one runs in the global
        // scope directly so `var`/function declarations persist.
        self.current = if self.strict {
            self.global_scope.child()
        } else {
            self.global_scope.clone()
        };
        self.var_scope = self.current.clone();
        self.this_val = self.global_this;
        self.new_target = NanBox::undefined();
        self.eval_depth += 1;
        // `$262.evalScript` runs a *Script*, so its global `var`/function
        // bindings are non-configurable (GlobalDeclarationInstantiation), unlike
        // an indirect `eval`'s deletable ones — even though both reuse
        // `run_eval_body`.
        let saved_script_eval = self.script_eval_globals;
        self.script_eval_globals = true;
        let result = self.run_eval_body(program);
        self.script_eval_globals = saved_script_eval;
        self.eval_depth -= 1;

        self.current = saved_current;
        self.var_scope = saved_var_scope;
        self.this_val = saved_this;
        self.new_target = saved_new_target;
        self.strict = saved_strict;
        result
    }

    /// Evaluates `source` (as a Script) in the created realm at `idx`, swapping in
    /// that realm's global scope + intrinsics for the duration and restoring the
    /// caller's afterward. Shared by `evalScript` and `$262.agent.start` (which
    /// runs a worker's whole source in its own realm).
    pub(crate) fn eval_source_in_realm(
        &mut self,
        idx: usize,
        source: &str,
    ) -> Result<NanBox, ExecError> {
        if self.eval_depth >= self.realm.limits.max_eval_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(msg))));
        }
        let scope = self.created_realms[idx].global_scope.clone();
        let global_this = self.created_realms[idx].global_this;
        let intrinsics = self.created_realms[idx].intrinsics;

        // Parse first so a SyntaxError surfaces before any environment swap.
        let program = self.parse_eval_program(source, false, false, false, false, &[], false)?;

        let saved_current = self.current.clone();
        let saved_global_scope = self.global_scope.clone();
        let saved_var_scope = self.var_scope.clone();
        let saved_global_this = self.global_this;
        let saved_this = self.this_val;
        let saved_new_target = self.new_target;
        let saved_strict = self.strict;
        let saved_intrinsics = self.realm.intrinsics_snapshot();

        self.strict = has_use_strict(&program.body);
        // A strict program gets its own child so its lexical declarations don't
        // leak into the realm's persistent global scope; a sloppy one runs in it
        // directly so `var`/function declarations persist across evalScript calls.
        self.current = if self.strict {
            scope.child()
        } else {
            scope.clone()
        };
        self.global_scope = scope;
        self.var_scope = self.current.clone();
        self.global_this = global_this;
        self.this_val = global_this;
        self.new_target = NanBox::undefined();
        self.realm.restore_intrinsics(intrinsics);
        // Swap in the child realm's `Intl` service `.prototype` cache so same-realm
        // `new Intl.X()` inside the evaluated source links to *this* realm's
        // prototypes (and any lazily-built ones are written back below).
        let child_intl = core::mem::take(&mut self.created_realms[idx].intl_protos);
        let saved_intl = self.realm.replace_intl_protos(child_intl);
        self.eval_depth += 1;
        let result = self.run_eval_body(program);
        self.eval_depth -= 1;

        self.created_realms[idx].intl_protos = self.realm.replace_intl_protos(saved_intl);
        self.current = saved_current;
        self.global_scope = saved_global_scope;
        self.var_scope = saved_var_scope;
        self.global_this = saved_global_this;
        self.this_val = saved_this;
        self.new_target = saved_new_target;
        self.strict = saved_strict;
        self.realm.restore_intrinsics(saved_intrinsics);
        result
    }

    /// Stamps the `ShadowRealm` internal slots onto a subclass-built instance.
    pub(crate) fn init_shadow_realm_super(&mut self, instance: Handle) {
        self.realm
            .set_hidden_property(instance, SHADOWREALM_BRAND, NanBox::boolean(true));
        let idx = self.shadow_realm_scopes.len();
        let scope = self.global_scope.child();
        self.shadow_realm_scopes.push(scope);
        self.realm
            .set_hidden_property(instance, SHADOWREALM_SCOPE_IDX, NanBox::number(idx as f64));
    }

    /// Applies `SuppressedError(error, suppressed, message)` semantics to a
    /// subclass-built instance.
    pub(crate) fn init_suppressed_error_super(&mut self, instance: Handle, args: &[NanBox]) {
        let error = args.first().copied().unwrap_or(NanBox::undefined());
        let suppressed = args.get(1).copied().unwrap_or(NanBox::undefined());
        let message = args.get(2).copied().unwrap_or(NanBox::undefined());
        if !matches!(message.unpack(), Unpacked::Undefined) {
            let s = self
                .coerce_to_string(message)
                .unwrap_or_else(|_| String::new());
            let m = self.new_str(&s);
            self.realm.set_property(instance, "message", m);
            self.realm.mark_hidden(instance, "message");
        }
        self.realm.set_property(instance, "error", error);
        self.realm.mark_hidden(instance, "error");
        self.realm.set_property(instance, "suppressed", suppressed);
        self.realm.mark_hidden(instance, "suppressed");
        // A `class S extends SuppressedError {}` instance inherits `[[ErrorData]]`.
        self.realm
            .set_hidden_property(instance, ERROR_DATA, NanBox::boolean(true));
    }

    /// RequireInternalSlot for a disposable-stack receiver: returns the receiver
    /// handle when `this` carries the brand, else a TypeError.
    fn require_dstack(&mut self, brand: &str, what: &str) -> Result<Handle, ExecError> {
        let this = self.this_val;
        if let Some(h) = this.as_handle().map(Handle::from_raw)
            && self.realm.get_property(h, brand).is_some()
        {
            return Ok(h);
        }
        Err(self.type_error(&alloc::format!("{what} called on an incompatible receiver")))
    }

    /// The current disposer-entry list of a disposable-stack instance (a JS array
    /// handle), creating one lazily if missing.
    fn dstack_list(&mut self, obj: Handle, list_key: &str) -> Handle {
        if let Some(h) = self
            .realm
            .get_property(obj, list_key)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            return h;
        }
        let h = self.realm.new_array(Vec::new());
        self.realm
            .set_hidden_property(obj, list_key, NanBox::handle(h.to_raw()));
        h
    }

    /// Pushes a disposer entry array onto the instance's list.
    fn dstack_push(&mut self, obj: Handle, list_key: &str, entry: Vec<NanBox>) {
        let list = self.dstack_list(obj, list_key);
        let mut elems = self
            .realm
            .array_elements(list)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let entry_arr = self.realm.new_array(entry);
        elems.push(NanBox::handle(entry_arr.to_raw()));
        let new_list = self.realm.new_array(elems);
        self.realm
            .set_hidden_property(obj, list_key, NanBox::handle(new_list.to_raw()));
    }

    /// Whether a disposable-stack instance is disposed.
    fn dstack_is_disposed(&mut self, obj: Handle, disposed_key: &str) -> bool {
        self.realm
            .get_property(obj, disposed_key)
            .is_some_and(|v| self.realm.truthy(v))
    }

    /// Reads a symbol-keyed method from `value` (down its prototype chain). A
    /// `null`/`undefined` `value` (or a primitive without the method) yields
    /// `None`; a present-but-not-callable method is a TypeError.
    fn get_dispose_method(
        &mut self,
        value: NanBox,
        sym_name: &'static str,
    ) -> Result<Option<NanBox>, ExecError> {
        if matches!(value.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Ok(None);
        }
        let obj = self.coerce_to_object(value);
        let Some(oh) = obj.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        let sym = self.well_known_symbol(sym_name);
        let key = self.member_key(sym);
        let m = self.read_member(oh, &key)?;
        if matches!(m.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Ok(None);
        }
        if !self.is_callable_value(m) {
            return Err(self.type_error(&alloc::format!("{sym_name} is not callable")));
        }
        Ok(Some(m))
    }

    /// Dispatches a `DisposableStack.prototype.<method>` / `AsyncDisposableStack`
    /// method. `target` is the method-name string; `is_async` selects the family.
    pub(crate) fn dstack_proto_dispatch(
        &mut self,
        is_async: bool,
        target: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let method = self.realm.string_value(target).unwrap_or_default();
        let (brand, list_key, disposed_key, what) = if is_async {
            (
                ADSTACK_BRAND,
                ADSTACK_LIST,
                ADSTACK_DISPOSED,
                "AsyncDisposableStack.prototype method",
            )
        } else {
            (
                DSTACK_BRAND,
                DSTACK_LIST,
                DSTACK_DISPOSED,
                "DisposableStack.prototype method",
            )
        };
        // `disposeAsync` is fully promise-returning: a failed RequireInternalSlot
        // (`this` not an AsyncDisposableStack) *rejects* the returned promise
        // rather than throwing synchronously.
        if method == "disposeAsync" {
            let p = self.fresh_promise();
            match self.require_dstack(brand, what) {
                Ok(obj) => match self.dstack_run_dispose(obj, list_key, disposed_key, true) {
                    Ok(_) => self.resolve_with(p, NanBox::undefined()),
                    Err(ExecError::Throw(e)) => self.settle(p, e, false),
                    Err(other) => return Err(other),
                },
                Err(ExecError::Throw(e)) => self.settle(p, e, false),
                Err(other) => return Err(other),
            }
            return Ok(NanBox::handle(p.to_raw()));
        }
        let obj = self.require_dstack(brand, what)?;
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method.as_str() {
            "use" => {
                if self.dstack_is_disposed(obj, disposed_key) {
                    return Err(self.reference_disposed());
                }
                let value = arg(0);
                // `null`/`undefined` is added with no dispose method (a no-op on
                // disposal), and returned as-is.
                if !matches!(value.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.resource_dispose_method(value, is_async)?;
                    self.dstack_push(obj, list_key, alloc::vec![value, m]);
                }
                Ok(value)
            }
            "adopt" => {
                if self.dstack_is_disposed(obj, disposed_key) {
                    return Err(self.reference_disposed());
                }
                let value = arg(0);
                let on_dispose = arg(1);
                if !self.is_callable_value(on_dispose) {
                    return Err(self.type_error("adopt onDispose is not callable"));
                }
                // The recorded dispose callback ignores `this` and calls
                // `onDispose(value)` at disposal time.
                let wrapper = self.make_adopt_wrapper(value, on_dispose);
                self.dstack_push(obj, list_key, alloc::vec![NanBox::undefined(), wrapper]);
                Ok(value)
            }
            "defer" => {
                if self.dstack_is_disposed(obj, disposed_key) {
                    return Err(self.reference_disposed());
                }
                let on_dispose = arg(0);
                if !self.is_callable_value(on_dispose) {
                    return Err(self.type_error("defer onDispose is not callable"));
                }
                self.dstack_push(obj, list_key, alloc::vec![NanBox::undefined(), on_dispose]);
                Ok(NanBox::undefined())
            }
            "move" => {
                if self.dstack_is_disposed(obj, disposed_key) {
                    return Err(self.reference_disposed());
                }
                // Build a fresh stack of the same kind, transfer the entries, then
                // mark this one disposed (without running disposers).
                let new_obj = self.realm.new_object();
                let default = self.intrinsic_proto(if is_async {
                    "AsyncDisposableStack"
                } else {
                    "DisposableStack"
                });
                if let Some(proto) = default {
                    self.realm.set_object_proto(new_obj, Some(proto));
                }
                self.realm
                    .set_hidden_property(new_obj, brand, NanBox::boolean(true));
                let list = self.dstack_list(obj, list_key);
                self.realm
                    .set_hidden_property(new_obj, list_key, NanBox::handle(list.to_raw()));
                self.realm
                    .set_hidden_property(new_obj, disposed_key, NanBox::boolean(false));
                // Empty this stack and mark it disposed.
                let empty = self.realm.new_array(Vec::new());
                self.realm
                    .set_hidden_property(obj, list_key, NanBox::handle(empty.to_raw()));
                self.realm
                    .set_hidden_property(obj, disposed_key, NanBox::boolean(true));
                Ok(NanBox::handle(new_obj.to_raw()))
            }
            "dispose" => self.dstack_run_dispose(obj, list_key, disposed_key, false),
            // `disposeAsync` is handled before the brand check above.
            _ => Err(self.type_error("unknown DisposableStack method")),
        }
    }

    /// Records a `using` / `await using` declaration's resource in the current
    /// lexical scope's dispose list (run, LIFO, when the scope is exited). Mirrors
    /// `CreateDisposableResource` + `AddDisposableResource`:
    ///
    /// * A `null`/`undefined` resource adds a no-op disposer (`method` is
    ///   `undefined`) — allowed for both hints (the sync-dispose null/undefined
    ///   short-circuit and the async path both record nothing to call).
    /// * Otherwise the resource must be an Object (a non-object primitive — number,
    ///   string, boolean, symbol, bigint — is a **TypeError** at the declaration),
    ///   and `GetDisposeMethod(V, hint)` is read **once**: for `await using` the
    ///   `@@asyncDispose` method, falling back to `@@dispose`; for `using` the
    ///   `@@dispose` method. A missing or non-callable method is a TypeError here
    ///   (at the declaration), per spec.
    pub(crate) fn record_using_resource(
        &mut self,
        value: NanBox,
        is_await: bool,
    ) -> Result<(), ExecError> {
        // `null`/`undefined`: record a no-op disposer (no method read).
        if matches!(value.unpack(), Unpacked::Undefined | Unpacked::Null) {
            self.current
                .add_disposer(value, NanBox::undefined(), is_await);
            return Ok(());
        }
        // A non-object primitive cannot be a disposable resource.
        if !self.is_object_value(value) {
            return Err(self.type_error("using declaration value is not an object"));
        }
        let method = self.using_dispose_method(value, is_await)?;
        self.current.add_disposer(value, method, is_await);
        Ok(())
    }

    /// `GetDisposeMethod(V, hint)` for a `using`/`await using` declaration whose
    /// resource is a non-null Object. For `await using` (async-dispose hint) the
    /// `@@asyncDispose` method is read first, falling back to `@@dispose`; for
    /// `using` (sync-dispose hint) only `@@dispose`. The chosen method must be
    /// callable, else a TypeError. The relevant symbol property is read exactly
    /// once (a getter fires once).
    fn using_dispose_method(&mut self, value: NanBox, is_await: bool) -> Result<NanBox, ExecError> {
        let Some(oh) = value.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("using declaration value is not an object"));
        };
        if is_await {
            // `@@asyncDispose`, falling back to `@@dispose`.
            let async_sym = self.well_known_symbol("asyncDispose");
            let async_key = self.member_key(async_sym);
            let m = self.read_member(oh, &async_key)?;
            if !matches!(m.unpack(), Unpacked::Undefined | Unpacked::Null) {
                if !self.is_callable_value(m) {
                    return Err(self.type_error("@@asyncDispose is not callable"));
                }
                return Ok(m);
            }
            let sym = self.well_known_symbol("dispose");
            let key = self.member_key(sym);
            let m = self.read_member(oh, &key)?;
            if matches!(m.unpack(), Unpacked::Undefined | Unpacked::Null) {
                return Err(self.type_error("value is not an async disposable"));
            }
            if !self.is_callable_value(m) {
                return Err(self.type_error("@@dispose is not callable"));
            }
            return Ok(m);
        }
        let sym = self.well_known_symbol("dispose");
        let key = self.member_key(sym);
        let m = self.read_member(oh, &key)?;
        if matches!(m.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Err(self.type_error("value is not disposable"));
        }
        if !self.is_callable_value(m) {
            return Err(self.type_error("@@dispose is not callable"));
        }
        Ok(m)
    }

    /// Runs the recorded `using` disposers of `disposers` (the list taken from a
    /// just-exited scope, in declaration order) in **reverse** (LIFO) order,
    /// threading `completion` through them: `Ok(_)` if the scope is leaving
    /// normally, `Err(ExecError::Throw)` if it is already unwinding a throw. A
    /// throwing disposer is aggregated with the in-flight completion into a
    /// `SuppressedError` chain (the newer error becomes `.error`, the prior
    /// completion `.suppressed`); a non-throw abrupt completion from a disposer is
    /// propagated as-is. An `await using` disposer's result is awaited **eagerly**
    /// (driving the event loop via `await_value`) — see the module note.
    pub(crate) fn dispose_resources(
        &mut self,
        disposers: alloc::vec::Vec<(NanBox, NanBox, bool)>,
        completion: Result<NanBox, ExecError>,
    ) -> Result<NanBox, ExecError> {
        // Start from the incoming completion: a normal one, or the in-flight throw.
        let (ok_value, mut pending): (NanBox, Option<NanBox>) = match completion {
            Ok(v) => (v, None),
            Err(ExecError::Throw(e)) => (NanBox::undefined(), Some(e)),
            // A non-throw abrupt completion (engine-internal) is propagated without
            // running disposers' aggregation against it.
            Err(other) => return Err(other),
        };
        for (value, method, is_async) in disposers.into_iter().rev() {
            if matches!(method.unpack(), Unpacked::Undefined | Unpacked::Null) {
                continue;
            }
            let result = self.call_with_this(method, value, &[]);
            // For an `await using`, await the dispose result (eagerly).
            let result = match result {
                Ok(v) if is_async => self.await_value(v),
                other => other,
            };
            if let Err(ExecError::Throw(e)) = result {
                pending = Some(match pending {
                    None => e,
                    Some(prev) => self.make_suppressed_error(e, prev),
                });
            } else {
                // Propagate a non-throw abrupt completion as-is.
                result?;
            }
        }
        match pending {
            None => Ok(ok_value),
            Some(e) => Err(ExecError::Throw(e)),
        }
    }

    /// Resolves the dispose method for a value added via `use`: for the sync
    /// stack the `[Symbol.dispose]` method; for the async stack the
    /// `[Symbol.asyncDispose]` method, falling back to `[Symbol.dispose]`. A value
    /// with neither is a TypeError.
    fn resource_dispose_method(
        &mut self,
        value: NanBox,
        is_async: bool,
    ) -> Result<NanBox, ExecError> {
        if is_async {
            if let Some(m) = self.get_dispose_method(value, "asyncDispose")? {
                return Ok(m);
            }
            if let Some(m) = self.get_dispose_method(value, "dispose")? {
                return Ok(m);
            }
            return Err(self.type_error("value is not an async disposable"));
        }
        if let Some(m) = self.get_dispose_method(value, "dispose")? {
            return Ok(m);
        }
        Err(self.type_error("value is not disposable"))
    }

    /// Builds the dispose callback for `adopt(value, onDispose)`: a bound native
    /// carrying `[value, onDispose]` that, when invoked at disposal, calls
    /// `onDispose(value)`.
    fn make_adopt_wrapper(&mut self, value: NanBox, on_dispose: NanBox) -> NanBox {
        let pair = self.realm.new_array(alloc::vec![value, on_dispose]);
        let f = self.realm.new_bound_native(N_DSTACK_ADOPT_CALL, pair);
        NanBox::handle(f.to_raw())
    }

    /// Dispatches an `N_DSTACK_ADOPT_CALL` callback: `target` is `[value,
    /// onDispose]`; calls `onDispose(value)`.
    pub(crate) fn dstack_adopt_call(&mut self, target: Handle) -> Result<NanBox, ExecError> {
        let pair = self
            .realm
            .array_elements(target)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let value = pair.first().copied().unwrap_or(NanBox::undefined());
        let on_dispose = pair.get(1).copied().unwrap_or(NanBox::undefined());
        self.call_with_this(on_dispose, NanBox::undefined(), &[value])
    }

    /// Runs the disposer entries of `obj` in reverse (LIFO) order, then marks the
    /// stack disposed. Idempotent: a disposed stack is a no-op. Throwing disposers
    /// are aggregated into a chain of `SuppressedError`s (the most recent error is
    /// `.error`, the prior accumulated completion is `.suppressed`).
    fn dstack_run_dispose(
        &mut self,
        obj: Handle,
        list_key: &str,
        disposed_key: &str,
        is_async: bool,
    ) -> Result<NanBox, ExecError> {
        if self.dstack_is_disposed(obj, disposed_key) {
            return Ok(NanBox::undefined());
        }
        self.realm
            .set_hidden_property(obj, disposed_key, NanBox::boolean(true));
        let list = self.dstack_list(obj, list_key);
        let entries = self
            .realm
            .array_elements(list)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        // Clear the list so later state inspection sees an empty stack.
        let empty = self.realm.new_array(Vec::new());
        self.realm
            .set_hidden_property(obj, list_key, NanBox::handle(empty.to_raw()));
        // A completion threaded through the reverse iteration: `None` = normal so
        // far; `Some(err)` = a pending thrown value to be (possibly) suppressed.
        let mut pending: Option<NanBox> = None;
        for entry in entries.into_iter().rev() {
            let pair = entry
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                .unwrap_or_default();
            let value = pair.first().copied().unwrap_or(NanBox::undefined());
            let m = pair.get(1).copied().unwrap_or(NanBox::undefined());
            if matches!(m.unpack(), Unpacked::Undefined | Unpacked::Null) {
                continue;
            }
            let result = self.call_with_this(m, value, &[]);
            // For the async stack, await a returned promise (eagerly).
            let result = match result {
                Ok(v) if is_async => self.await_value(v),
                other => other,
            };
            if let Err(ExecError::Throw(e)) = result {
                pending = Some(match pending {
                    None => e,
                    Some(prev) => self.make_suppressed_error(e, prev),
                });
            } else {
                // A non-throw abrupt completion (engine-internal) propagates as-is.
                result?;
            }
        }
        match pending {
            None => Ok(NanBox::undefined()),
            Some(e) => Err(ExecError::Throw(e)),
        }
    }

    /// A `ReferenceError` for operating on a disposed stack.
    fn reference_disposed(&mut self) -> ExecError {
        let m = self.new_str("DisposableStack already disposed");
        ExecError::Throw(self.make_error(N_REFERENCE_ERROR, Some(m)))
    }

    /// Builds a `SuppressedError` object (ES2025) wrapping `error` (the most
    /// recent throw) and `suppressed` (the prior completion). Per the spec the
    /// constructor is called with `(error, suppressed, "An error was suppressed
    /// during disposal.")`, so the instance is a real `SuppressedError` (with the
    /// proper `[[Prototype]]`, `error`/`suppressed`, and message).
    fn make_suppressed_error(&mut self, error: NanBox, suppressed: NanBox) -> NanBox {
        let msg = self.new_str("An error was suppressed during disposal.");
        let callee = self
            .current
            .get("SuppressedError")
            .unwrap_or(NanBox::undefined());
        self.construct_suppressed_error(&[error, suppressed, msg], callee, callee)
            .unwrap_or(error)
    }

    /// `new ShadowRealm()` — a fresh branded instance. The "realm" itself is
    /// modeled lazily: `evaluate` runs source in a fresh global child scope.
    pub(crate) fn construct_shadow_realm(
        &mut self,
        callee: NanBox,
        new_target: NanBox,
    ) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        let default = self.intrinsic_proto("ShadowRealm");
        if let Some(proto) = self.resource_instance_proto(new_target, callee, default)? {
            self.realm.set_object_proto(obj, Some(proto));
        }
        self.realm
            .set_hidden_property(obj, SHADOWREALM_BRAND, NanBox::boolean(true));
        // Allocate a persistent per-instance global scope (a child of the host
        // global scope) so successive `evaluate` calls share declarations.
        let idx = self.shadow_realm_scopes.len();
        let scope = self.global_scope.child();
        self.shadow_realm_scopes.push(scope);
        self.realm
            .set_hidden_property(obj, SHADOWREALM_SCOPE_IDX, NanBox::number(idx as f64));
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Dispatches a `ShadowRealm.prototype.<method>`. `target` is the method-name
    /// string.
    pub(crate) fn shadow_realm_dispatch(
        &mut self,
        target: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let method = self.realm.string_value(target).unwrap_or_default();
        // RequireInternalSlot([[ShadowRealm]]).
        let this = self.this_val;
        if this
            .as_handle()
            .map(Handle::from_raw)
            .is_none_or(|h| self.realm.get_property(h, SHADOWREALM_BRAND).is_none())
        {
            return Err(self.type_error(&alloc::format!(
                "ShadowRealm.prototype.{method} called on an incompatible receiver"
            )));
        }
        let realm_obj = this.as_handle().map(Handle::from_raw);
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method.as_str() {
            "evaluate" => self.shadow_realm_evaluate(realm_obj, arg(0)),
            "importValue" => {
                // Per spec these argument validations happen *synchronously* (they
                // throw, not reject): ToString(specifier) — which may run a user
                // `toString`/`valueOf` and throw — then a TypeError if `exportName`
                // is not a String.
                let _specifier = self.coerce_to_string(arg(0))?;
                let export_name = arg(1);
                if export_name
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_none_or(|h| self.realm.string_value(h).is_none())
                {
                    return Err(
                        self.type_error("ShadowRealm importValue exportName must be a string")
                    );
                }
                // Modules are unsupported: return a rejected promise (the
                // constructor + `evaluate` are the conformance focus).
                let p = self.fresh_promise();
                let m = self.new_str("ShadowRealm.prototype.importValue is not supported");
                let err = self.make_error(N_TYPE_ERROR, Some(m));
                self.settle(p, err, false);
                Ok(NanBox::handle(p.to_raw()))
            }
            _ => Err(self.type_error("unknown ShadowRealm method")),
        }
    }

    /// `ShadowRealm.prototype.evaluate(sourceText)` — `sourceText` must be a
    /// string (else a TypeError). Evaluates it in a fresh global child scope and
    /// returns a primitive directly, wraps a callable in a new function, or throws
    /// a TypeError for any other object result.
    fn shadow_realm_evaluate(
        &mut self,
        realm_obj: Option<Handle>,
        source_arg: NanBox,
    ) -> Result<NanBox, ExecError> {
        // The argument must be a String (no coercion of objects).
        let Some(source) = source_arg
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("ShadowRealm.prototype.evaluate expects a string"));
        };
        // Parse first: a SyntaxError from parsing is surfaced to the caller realm
        // *as a SyntaxError* (per the ShadowRealm spec), not wrapped. ShadowRealm
        // code is global-scoped, so no inherited `super`.
        let program = self.parse_eval_program(&source, false, false, false, false, &[], false)?;
        // Evaluate the parsed program in the instance's persistent global scope (a
        // best-effort isolated environment that shares the intrinsics). A *runtime*
        // throw is wrapped as a TypeError per the spec.
        let result = self.shadow_realm_run_program(realm_obj, program);
        let value = match result {
            Ok(v) => v,
            Err(ExecError::Throw(_)) => {
                return Err(self.type_error("ShadowRealm evaluate threw"));
            }
            Err(other) => return Err(other),
        };
        // A primitive passes through directly.
        if !self.is_object_value(value) {
            return Ok(value);
        }
        // A callable is wrapped in a new function in the caller's realm. Any error
        // raised while copying `name`/`length` (e.g. a throwing accessor) is
        // wrapped as a TypeError per the spec.
        if self.is_callable_value(value) {
            return match self.make_wrapped_function(value) {
                Ok(w) => Ok(w),
                Err(ExecError::Throw(_)) => {
                    Err(self.type_error("ShadowRealm wrapped function creation threw"))
                }
                Err(other) => Err(other),
            };
        }
        // Any other object is a TypeError (only primitives/callables cross the
        // realm boundary).
        Err(self.type_error("ShadowRealm.prototype.evaluate result is not a primitive or callable"))
    }

    /// Runs an already-parsed `program` in the `ShadowRealm` instance's persistent
    /// global scope, returning its completion value. Declarations persist across
    /// successive `evaluate` calls on the same instance. Mirrors indirect-eval
    /// scoping (global `this`, sloppy unless the source self-declares strict).
    fn shadow_realm_run_program(
        &mut self,
        realm_obj: Option<Handle>,
        program: &'a Program,
    ) -> Result<NanBox, ExecError> {
        if self.eval_depth >= self.realm.limits.max_eval_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(msg))));
        }
        // The instance's persistent scope (a fresh child fallback if absent).
        let scope = realm_obj
            .and_then(|h| self.realm.get_property(h, SHADOWREALM_SCOPE_IDX))
            .and_then(|v| v.as_number())
            .and_then(|n| self.shadow_realm_scopes.get(n as usize).cloned())
            .unwrap_or_else(|| self.global_scope.child());
        let saved_strict = self.strict;
        let saved_scope = self.current.clone();
        let (saved_this, saved_new_target) = (self.this_val, self.new_target);
        self.strict = has_use_strict(&program.body);
        // A strict program gets its own child so its lexical declarations don't
        // leak into the shared instance scope; a sloppy one runs directly in it so
        // `var`/function declarations persist (matching global-scope eval).
        self.current = if self.strict { scope.child() } else { scope };
        self.this_val = self.global_this;
        self.new_target = NanBox::undefined();
        self.eval_depth += 1;
        let result = self.run_eval_body(program);
        self.eval_depth -= 1;
        self.current = saved_scope;
        self.strict = saved_strict;
        self.this_val = saved_this;
        self.new_target = saved_new_target;
        result
    }

    /// Wraps a callable `target` from the shadow realm in a new function exposed in
    /// the caller's realm: an `N_SHADOW_REALM_WRAPPED` bound native carrying the
    /// target, with `length`/`name` copied from the target.
    fn make_wrapped_function(&mut self, target: NanBox) -> Result<NanBox, ExecError> {
        let Some(th) = target.as_handle().map(Handle::from_raw) else {
            return Ok(NanBox::undefined());
        };
        let f = self.realm.new_bound_native(N_SHADOW_REALM_WRAPPED, th);
        // `length` is `ToIntegerOrInfinity(Get(target, "length"))` clamped to
        // [0, +∞): `read_member` invokes a `length` accessor if present.
        let raw_len = self.read_member(th, "length")?;
        let n = self.realm.to_number(raw_len);
        // `ToIntegerOrInfinity` then clamped to [0, +∞]: NaN → 0, +∞ stays +∞,
        // anything ≤ 0 → 0, otherwise truncated toward zero.
        let length = if n.is_nan() || n <= 0.0 {
            0.0
        } else if n.is_infinite() {
            f64::INFINITY
        } else {
            trunc_toward_zero(n)
        };
        self.realm
            .set_hidden_property(f, "length", NanBox::number(length));
        self.realm.set_readonly_property(f, "length");
        // `name` is `ToString(Get(target, "name"))` (a non-string defaults to "").
        let raw_name = self.read_member(th, "name")?;
        let name = if raw_name
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.string_value(h).is_some())
        {
            self.realm.to_display_string(raw_name)
        } else {
            String::new()
        };
        let name_v = self.new_str(&name);
        self.realm.set_property(f, "name", name_v);
        self.realm.mark_hidden(f, "name");
        self.realm.set_readonly_property(f, "name");
        Ok(NanBox::handle(f.to_raw()))
    }

    /// Dispatches a call to a wrapped shadow-realm function
    /// (`N_SHADOW_REALM_WRAPPED`): `target` is the wrapped callable. Arguments must
    /// be primitives or callables (else a TypeError); the result is likewise
    /// wrapped/checked.
    pub(crate) fn shadow_realm_wrapped_call(
        &mut self,
        target: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        // Each argument must be a primitive or a callable (which is re-wrapped).
        let mut call_args = Vec::with_capacity(args.len());
        for &a in args {
            if !self.is_object_value(a) {
                call_args.push(a);
            } else if self.is_callable_value(a) {
                let w = self.make_wrapped_function(a)?;
                call_args.push(w);
            } else {
                return Err(self.type_error("wrapped function arguments must be primitives"));
            }
        }
        let target_box = NanBox::handle(target.to_raw());
        let result = match self.call_with_this(target_box, NanBox::undefined(), &call_args) {
            Ok(v) => v,
            Err(ExecError::Throw(_)) => {
                return Err(self.type_error("wrapped function threw"));
            }
            Err(other) => return Err(other),
        };
        if !self.is_object_value(result) {
            return Ok(result);
        }
        if self.is_callable_value(result) {
            return self.make_wrapped_function(result);
        }
        Err(self.type_error("wrapped function result is not a primitive or callable"))
    }
}
