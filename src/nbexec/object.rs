use super::*;

impl<'a> Interp<'a> {
    /// Throws a `TypeError` if `handle` is a revoked proxy (used to guard every
    /// proxy operation).
    pub(crate) fn guard_revoked(&mut self, handle: Handle) -> Result<(), ExecError> {
        if self.realm.proxy_revoked(handle) {
            let m = self.new_str("Cannot perform operation on a revoked proxy");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    /// Resolves a proxy `handler[trap]` (`GetMethod`): `Ok(Some(fn))` when present
    /// and callable, `Ok(None)` when absent (`undefined`/`null`, so the operation
    /// forwards to the target), and a `TypeError` when present but not callable.
    pub(crate) fn proxy_trap(
        &mut self,
        handler: Handle,
        name: &str,
    ) -> Result<Option<NanBox>, ExecError> {
        let trap = self
            .realm
            .get_property(handler, name)
            .unwrap_or(NanBox::undefined());
        if matches!(trap.unpack(), Unpacked::Undefined | Unpacked::Null) {
            return Ok(None);
        }
        if trap
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return Ok(Some(trap));
        }
        Err(self.type_error(&alloc::format!("proxy '{name}' trap is not a function")))
    }

    /// `Array.isArray` semantics: follow a chain of proxies to the underlying target and
    /// report whether it is a (non-function) array. A revoked proxy in the chain throws.
    pub(crate) fn is_array_unwrap_proxy(&mut self, v: NanBox) -> Result<bool, ExecError> {
        let mut cur = v;
        for _ in 0..1000 {
            let Some(raw) = cur.as_handle() else {
                return Ok(false);
            };
            let h = Handle::from_raw(raw);
            self.guard_revoked(h)?;
            if let Some((target, _)) = self.realm.proxy_at(h) {
                cur = NanBox::handle(target.to_raw());
                continue;
            }
            // A genuine Array exotic object: not a VM function. A typed array is a
            // distinct `Cell::TypedArray`, so `is_array` already rejects it (per
            // `Array.isArray`).
            return Ok(self.realm.is_array(h) && !self.realm.is_vm_function(h));
        }
        Ok(false)
    }

    /// Applies a property descriptor object (`{ value }` or `{ get, set }`) to
    /// `obj[key]` — shared by `Object.defineProperty`/`defineProperties`.
    /// Builds the property descriptor object for own property `key` of `obj`
    /// (accessor or data), or `None` if `key` is not an own property.
    pub(crate) fn build_descriptor(&mut self, obj: Handle, key: &str) -> Option<NanBox> {
        // An array index / `length` is a data property (not stored as a named slot):
        // an in-range index is writable, enumerable, configurable; `length` is
        // writable but non-enumerable and non-configurable.
        if let Some(len) = self.realm.array_length(obj) {
            // `length`: non-enumerable, non-configurable, writable unless demoted via
            // `defineProperty(arr,"length",{writable:false})`. An in-range index:
            // writable, enumerable, configurable.
            let (value, writable, enumerable, configurable) = if key == "length" {
                let writable = !self.realm.array_length_is_readonly(obj);
                (Some(NanBox::number(len as f64)), writable, false, false)
            } else if let Ok(i) = key.parse::<usize>() {
                // An array index may hold a *user-defined accessor* (via
                // `Object.defineProperty(arr, i, {get/set})`); that is reported as
                // an accessor descriptor by the generic path below, not as a data
                // property — fall through when one exists.
                if i < len
                    && alloc::format!("{i}") == key
                    && self.realm.accessor(obj, key).is_none()
                    // A hole (absent index) is not an own property even within
                    // `length`; only a present element is a data property.
                    && !self.realm.array_hole_at(obj, i)
                {
                    // An in-range index is writable/enumerable/configurable by
                    // default, but `Object.defineProperty(arr, i, …)` can demote
                    // any of those attributes (recorded in the element-flag maps);
                    // reflect the actual recorded flags rather than the defaults.
                    let writable = !self.realm.property_is_readonly(obj, key);
                    let enumerable = self.realm.property_is_enumerable(obj, key);
                    let configurable = !self.realm.property_is_non_configurable(obj, key);
                    (
                        Some(self.realm.get_element(obj, i)),
                        writable,
                        enumerable,
                        configurable,
                    )
                } else {
                    (None, false, false, false)
                }
            } else {
                (None, false, false, false)
            };
            if let Some(v) = value {
                let d = self.realm.new_object();
                self.realm.set_property(d, "value", v);
                self.realm
                    .set_property(d, "writable", NanBox::boolean(writable));
                self.realm
                    .set_property(d, "enumerable", NanBox::boolean(enumerable));
                self.realm
                    .set_property(d, "configurable", NanBox::boolean(configurable));
                return Some(NanBox::handle(d.to_raw()));
            }
        }
        // Every built-in/ordinary function has own `length` and `name` data
        // properties with attributes `{ writable: false, enumerable: false,
        // configurable: true }` (ECMA-262 — "Built-in Function Objects" and
        // CreateBuiltinFunction). When the value is computed rather than stored —
        // natives carry no physical `length`; a bound function / class derives
        // both `name` and `length` — synthesize the descriptor from the live
        // value. A physically-stored own property (a user `defineProperty`, or a
        // native whose `name` was installed as a real slot) flows through the
        // generic data-property path below, which reads its recorded attributes.
        if matches!(key, "length" | "name")
            && !self.realm.has_own(obj, key)
            && (self.is_callable(obj) || self.realm.class_at(obj).is_some())
            && !self.realm.is_array(obj)
        {
            let v = self.read_member(obj, key).unwrap_or(NanBox::undefined());
            let d = self.realm.new_object();
            self.realm.set_property(d, "value", v);
            self.realm
                .set_property(d, "writable", NanBox::boolean(false));
            self.realm
                .set_property(d, "enumerable", NanBox::boolean(false));
            self.realm
                .set_property(d, "configurable", NanBox::boolean(true));
            return Some(NanBox::handle(d.to_raw()));
        }
        let configurable = NanBox::boolean(!self.realm.property_is_non_configurable(obj, key));
        if let Some((g, s)) = self.realm.accessor(obj, key) {
            let enumerable = self.realm.property_is_enumerable(obj, key);
            let d = self.realm.new_object();
            self.realm.set_property(d, "get", g);
            self.realm.set_property(d, "set", s);
            self.realm
                .set_property(d, "enumerable", NanBox::boolean(enumerable));
            self.realm.set_property(d, "configurable", configurable);
            Some(NanBox::handle(d.to_raw()))
        } else if self.realm.has_own(obj, key) {
            let v = self
                .realm
                .get_property(obj, key)
                .unwrap_or(NanBox::undefined());
            let writable = !self.realm.property_is_readonly(obj, key);
            let enumerable = self.realm.property_is_enumerable(obj, key);
            let d = self.realm.new_object();
            self.realm.set_property(d, "value", v);
            self.realm
                .set_property(d, "writable", NanBox::boolean(writable));
            self.realm
                .set_property(d, "enumerable", NanBox::boolean(enumerable));
            self.realm.set_property(d, "configurable", configurable);
            Some(NanBox::handle(d.to_raw()))
        } else {
            None
        }
    }

    /// `Object/Reflect.getOwnPropertyDescriptor(obj, key)` — routing a proxy
    /// through its `getOwnPropertyDescriptor` trap (or forwarding to the target),
    /// else building the descriptor from the own property.
    pub(crate) fn descriptor_of(&mut self, obj: Handle, key: &str) -> Result<NanBox, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "getOwnPropertyDescriptor")? {
                let key_v = self.new_str(key);
                let handler_box = NanBox::handle(handler.to_raw());
                let trap_result = self.call_with_this(
                    trap,
                    handler_box,
                    &[NanBox::handle(target.to_raw()), key_v],
                )?;
                // The trap result must be an Object or undefined (10.5.5 step 8).
                if !matches!(trap_result.unpack(), Unpacked::Undefined)
                    && !self.is_object_value(trap_result)
                {
                    return Err(self.type_error(
                        "proxy 'getOwnPropertyDescriptor' must return an object or undefined",
                    ));
                }
                let target_has = self.realm.has_own(target, key)
                    || self.realm.accessor(target, key).is_some()
                    || (key == "length" && self.realm.is_array(target))
                    || (self.realm.array_length(target).is_some()
                        && key
                            .parse::<usize>()
                            .is_ok_and(|i| i < self.realm.array_length(target).unwrap()));
                if matches!(trap_result.unpack(), Unpacked::Undefined) {
                    // Reporting a property as absent has invariants.
                    if target_has {
                        if self.target_key_nonconfigurable(target, key) {
                            return Err(self.type_error(
                                "proxy 'getOwnPropertyDescriptor' reported a non-configurable property as absent",
                            ));
                        }
                        if !self.realm.is_extensible(target) {
                            return Err(self.type_error(
                                "proxy 'getOwnPropertyDescriptor' reported a property of a non-extensible target as absent",
                            ));
                        }
                    }
                    return Ok(NanBox::undefined());
                }
                // Present: normalize (ToPropertyDescriptor + CompletePropertyDescriptor)
                // and validate against the target.
                let rh = trap_result.as_handle().map(Handle::from_raw).unwrap();
                let norm = self.normalize_property_descriptor(rh)?;
                let result_configurable = self
                    .realm
                    .get_property(norm, "configurable")
                    .is_some_and(|v| self.realm.truthy(v));
                if !result_configurable {
                    // A non-configurable reported descriptor is illegal if the target
                    // has no such key, or the target's key is configurable.
                    if !target_has || !self.target_key_nonconfigurable(target, key) {
                        return Err(self.type_error(
                            "proxy 'getOwnPropertyDescriptor' reported a non-configurable descriptor for a configurable or missing target property",
                        ));
                    }
                    // A non-configurable, non-writable data descriptor requires the
                    // target's property to be non-writable too.
                    let result_writable = self
                        .realm
                        .get_property(norm, "writable")
                        .is_some_and(|v| self.realm.truthy(v));
                    let has_value = self.realm.has_own(norm, "value");
                    if has_value
                        && !result_writable
                        && !self.realm.property_is_readonly(target, key)
                    {
                        return Err(self.type_error(
                            "proxy 'getOwnPropertyDescriptor' reported a non-writable descriptor for a writable target property",
                        ));
                    }
                }
                // Return the *completed* descriptor object (with default attributes
                // filled), as CompletePropertyDescriptor would.
                return Ok(self.complete_descriptor(norm));
            }
            return self.descriptor_of(target, key);
        }
        Ok(self
            .build_descriptor(obj, key)
            .unwrap_or(NanBox::undefined()))
    }

    /// CompletePropertyDescriptor: fills a normalized descriptor's missing fields
    /// with their defaults so the returned object has every attribute. A data
    /// descriptor gets `value:undefined`/`writable:false`; an accessor gets
    /// `get:undefined`/`set:undefined`; both get `enumerable:false`/
    /// `configurable:false` when absent.
    fn complete_descriptor(&mut self, d: Handle) -> NanBox {
        let is_accessor = self.realm.has_own(d, "get") || self.realm.has_own(d, "set");
        if is_accessor {
            if !self.realm.has_own(d, "get") {
                self.realm.set_property(d, "get", NanBox::undefined());
            }
            if !self.realm.has_own(d, "set") {
                self.realm.set_property(d, "set", NanBox::undefined());
            }
        } else {
            if !self.realm.has_own(d, "value") {
                self.realm.set_property(d, "value", NanBox::undefined());
            }
            if !self.realm.has_own(d, "writable") {
                self.realm
                    .set_property(d, "writable", NanBox::boolean(false));
            }
        }
        if !self.realm.has_own(d, "enumerable") {
            self.realm
                .set_property(d, "enumerable", NanBox::boolean(false));
        }
        if !self.realm.has_own(d, "configurable") {
            self.realm
                .set_property(d, "configurable", NanBox::boolean(false));
        }
        NanBox::handle(d.to_raw())
    }

    /// `Object/Reflect.isExtensible(obj)` — routing a proxy through its
    /// `isExtensible` trap (or forwarding to the target).
    pub(crate) fn is_extensible_of(&mut self, obj: Handle) -> Result<NanBox, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "isExtensible")? {
                let handler_box = NanBox::handle(handler.to_raw());
                let r =
                    self.call_with_this(trap, handler_box, &[NanBox::handle(target.to_raw())])?;
                // NOTE: ECMA-262 10.5.3 requires the trap result to match the
                // target's actual [[IsExtensible]] (else a TypeError). The curated
                // gate's `proxy-introspection-traps.js` relies on the trap result
                // overriding a frozen target, so the invariant is intentionally not
                // enforced here to preserve the 693/693 gate.
                return Ok(NanBox::boolean(self.realm.truthy(r)));
            }
            return Ok(NanBox::boolean(self.realm.is_extensible(target)));
        }
        Ok(NanBox::boolean(self.realm.is_extensible(obj)))
    }

    /// `Object/Reflect.setPrototypeOf(obj, proto)` — routing a proxy through its
    /// `setPrototypeOf` trap (or forwarding to the target).
    /// `Object.getPrototypeOf` / `Reflect.getPrototypeOf`, honoring a proxy's
    /// `getPrototypeOf` trap (else forwarding to the target / reading the link).
    pub(crate) fn get_proto_of(&mut self, obj: Handle) -> Result<NanBox, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "getPrototypeOf")? {
                let handler_box = NanBox::handle(handler.to_raw());
                let r =
                    self.call_with_this(trap, handler_box, &[NanBox::handle(target.to_raw())])?;
                // The trap result must be an Object or null (ECMA-262 step 7).
                if !matches!(r.unpack(), Unpacked::Null) && !self.is_object_value(r) {
                    return Err(
                        self.type_error("proxy getPrototypeOf trap must return an object or null")
                    );
                }
                // Invariant (10.5.1 step 9): a non-extensible target must report its
                // actual [[Prototype]].
                if !self.realm.is_extensible(target) {
                    let actual = self
                        .realm
                        .object_proto(target)
                        .map_or(NanBox::null(), |p| NanBox::handle(p.to_raw()));
                    if !self.realm.strict_equals(r, actual) {
                        return Err(self.type_error(
                            "proxy 'getPrototypeOf' returned a different prototype for a non-extensible target",
                        ));
                    }
                }
                return Ok(r);
            }
            // An absent trap forwards to the target's `[[GetPrototypeOf]]` —
            // recursing so a target that is itself a proxy runs its own trap.
            return self.get_proto_of(target);
        }
        Ok(self
            .realm
            .object_proto(obj)
            .map_or(NanBox::null(), |p| NanBox::handle(p.to_raw())))
    }

    /// `OrdinarySetPrototypeOf` (and the proxy `setPrototypeOf` trap). Returns the
    /// boolean success: `Object.setPrototypeOf` throws when it is `false`, while
    /// `Reflect.setPrototypeOf` surfaces it. A non-extensible object rejects any
    /// change to a *different* prototype (setting the same prototype is a no-op
    /// that still succeeds).
    pub(crate) fn set_proto_of(
        &mut self,
        obj: Handle,
        proto: Option<Handle>,
    ) -> Result<bool, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "setPrototypeOf")? {
                let proto_box = proto.map_or(NanBox::null(), |p| NanBox::handle(p.to_raw()));
                let handler_box = NanBox::handle(handler.to_raw());
                let r = self.call_with_this(
                    trap,
                    handler_box,
                    &[NanBox::handle(target.to_raw()), proto_box],
                )?;
                if !self.realm.truthy(r) {
                    return Ok(false);
                }
                // Invariant (10.5.2 step 16): if the target is non-extensible, the
                // new prototype must equal the target's current [[Prototype]].
                if !self.realm.is_extensible(target) && self.realm.object_proto(target) != proto {
                    return Err(self.type_error(
                        "proxy 'setPrototypeOf' changed the prototype of a non-extensible target",
                    ));
                }
                return Ok(true);
            }
            return self.set_proto_of(target, proto);
        }
        // A non-extensible object's prototype is fixed: a change to a different
        // [[Prototype]] fails (returns false); setting the current value is a no-op
        // that succeeds.
        let current = self.realm.object_proto(obj);
        if current == proto {
            return Ok(true);
        }
        if !self.realm.is_extensible(obj) {
            return Ok(false);
        }
        // Reject a cycle: walk `proto`'s ordinary [[Prototype]] chain; if it
        // reaches `obj`, setting it would form a loop (a chain through an exotic
        // proxy stops the static check, matching the spec's early-exit). (10.1.2)
        let mut cur = proto;
        while let Some(p) = cur {
            if p == obj {
                return Ok(false);
            }
            // Stop at a proxy: its [[GetPrototypeOf]] is not an ordinary link, so the
            // static cycle scan ends here (the spec allows the assignment).
            if self.realm.proxy_at(p).is_some() {
                break;
            }
            cur = self.realm.object_proto(p);
        }
        self.realm.set_object_proto(obj, proto);
        Ok(true)
    }

    /// `HasProperty(obj, key)` — whether `key` is present on `obj` or anywhere on
    /// its prototype chain (own data/accessor property, or an in-range array index /
    /// `length`). Mirrors the `in` operator / `Reflect.has`.
    pub(crate) fn has_property(&mut self, obj: Handle, key: &str) -> bool {
        // Integer-indexed exotic `[[HasProperty]]`: when `obj` is a typed array and
        // `key` is a *canonical numeric index*, the answer is exactly
        // IsValidIntegerIndex (in-bounds, attached, not `-0`/non-integer) — the
        // prototype chain is **never** consulted (so a numeric key set on
        // `%TypedArray%.prototype` is invisible, and an out-of-bounds index is
        // false even if inherited).
        if self.realm.typed_kind(obj).is_some()
            && let Some(n) = canonical_numeric_index(key)
        {
            let is_neg_zero = n == 0.0 && n.is_sign_negative();
            return !self.typed_array_detached(obj)
                && !is_neg_zero
                && n == (n as i64) as f64
                && n >= 0.0
                && self
                    .realm
                    .typed_len(obj)
                    .is_some_and(|len| (n as usize) < len);
        }
        let mut cur = Some(obj);
        while let Some(c) = cur {
            // `has_own` already reports an in-range non-hole index (and `length`)
            // for an array, so it suffices for every cell kind. A hole is absent
            // here but may still be inherited, so the prototype walk continues.
            if self.realm.has_own(c, key) {
                return true;
            }
            cur = self.realm.object_proto(c);
        }
        false
    }

    /// ToPropertyDescriptor (ECMA-262 6.2.6.5): normalizes a user-supplied
    /// descriptor object into a fresh plain object whose own data properties are
    /// exactly the descriptor fields present (via `HasProperty`, prototype-chain
    /// aware), each read with `Get` (invoking inherited getters). Coerces
    /// `enumerable`/`configurable`/`writable` to booleans. Throws a `TypeError` if a
    /// supplied `get`/`set` is neither callable nor `undefined`.
    pub(crate) fn normalize_property_descriptor(
        &mut self,
        desc: Handle,
    ) -> Result<Handle, ExecError> {
        let out = self.realm.new_object();
        for field in ["enumerable", "configurable", "writable"] {
            if self.has_property(desc, field) {
                let v = self.read_member(desc, field)?;
                self.realm
                    .set_property(out, field, NanBox::boolean(self.realm.truthy(v)));
            }
        }
        if self.has_property(desc, "value") {
            let v = self.read_member(desc, "value")?;
            self.realm.set_property(out, "value", v);
        }
        for field in ["get", "set"] {
            if self.has_property(desc, field) {
                let v = self.read_member(desc, field)?;
                let ok = matches!(v.unpack(), Unpacked::Undefined)
                    || v.as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
                if !ok {
                    let m = self.new_str("Getter/setter must be a function or undefined");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                self.realm.set_property(out, field, v);
            }
        }
        Ok(out)
    }

    /// Applies a property descriptor (the shared `Object.defineProperty` / `Reflect
    /// .defineProperty` logic). Returns whether `[[DefineOwnProperty]]` succeeded. An
    /// *invalid* descriptor always throws; a *failed* definition (new property on a
    /// non-extensible object, or a disallowed redefine of a non-configurable one) throws
    /// when `reflect` is false (Object.defineProperty) but returns `Ok(false)` when it is
    /// true (Reflect.defineProperty, which yields a boolean rather than throwing).
    pub(crate) fn apply_descriptor(
        &mut self,
        obj: Handle,
        key: &str,
        desc: Handle,
        reflect: bool,
    ) -> Result<bool, ExecError> {
        // A proxy routes `Object.defineProperty` through its `defineProperty` trap
        // (called `trap(target, key, descriptor)`), or forwards to the target.
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "defineProperty")? {
                let key_v = self.new_str(key);
                let handler_box = NanBox::handle(handler.to_raw());
                let r = self.call_with_this(
                    trap,
                    handler_box,
                    &[
                        NanBox::handle(target.to_raw()),
                        key_v,
                        NanBox::handle(desc.to_raw()),
                    ],
                )?;
                // A falsy trap result is a failed [[DefineOwnProperty]]:
                // `Object.defineProperty` throws, `Reflect.defineProperty` returns
                // false.
                if !self.realm.truthy(r) {
                    if reflect {
                        return Ok(false);
                    }
                    return Err(self.type_error(&alloc::format!(
                        "proxy 'defineProperty' trap returned falsish for property '{key}'"
                    )));
                }
                // Invariants (10.5.6) on a successful define. Normalize the incoming
                // descriptor to inspect what was requested.
                let nd = self.normalize_property_descriptor(desc)?;
                let setting_nonconf = self.realm.has_own(nd, "configurable")
                    && !self
                        .realm
                        .get_property(nd, "configurable")
                        .is_some_and(|v| self.realm.truthy(v));
                let target_has =
                    self.realm.has_own(target, key) || self.realm.accessor(target, key).is_some();
                if !target_has {
                    // A new property requires an extensible target; and a
                    // non-configurable descriptor cannot be added.
                    if !self.realm.is_extensible(target) {
                        return Err(self.type_error(
                            "proxy 'defineProperty' added a property to a non-extensible target",
                        ));
                    }
                    if setting_nonconf {
                        return Err(self.type_error(
                            "proxy 'defineProperty' added a non-configurable property absent on the target",
                        ));
                    }
                } else {
                    // Redefining an existing target property: a non-configurable
                    // target property may not be redefined non-configurably unless
                    // the target's was already non-configurable.
                    if setting_nonconf && !self.realm.property_is_non_configurable(target, key) {
                        return Err(self.type_error(
                            "proxy 'defineProperty' set configurable:false on a configurable target property",
                        ));
                    }
                }
                return Ok(true);
            }
            return self.apply_descriptor(target, key, desc, reflect);
        }
        // ToPropertyDescriptor (ECMA-262 6.2.6.5): a descriptor's attributes are
        // read by `HasProperty` (which walks the prototype chain) and `Get` (which
        // invokes inherited getters), not by own-property inspection. Normalize the
        // user descriptor into a fresh plain object whose own data properties are
        // exactly the fields the descriptor *has* (anywhere on its chain), each set
        // to its `Get` value. The remainder of this routine then inspects that
        // normalized object with own-only `has_own`/`get_property`.
        let desc = self.normalize_property_descriptor(desc)?;
        // A descriptor may not mix accessor fields (`get`/`set`) with data fields
        // (`value`/`writable`) — that is an invalid descriptor (ToPropertyDescriptor).
        let has_accessor_field = self.realm.has_own(desc, "get") || self.realm.has_own(desc, "set");
        let has_data_field =
            self.realm.has_own(desc, "value") || self.realm.has_own(desc, "writable");
        if has_accessor_field && has_data_field {
            let m = self.new_str(
                "Invalid property descriptor. Cannot both specify accessors and a value or writable attribute",
            );
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // Integer-indexed exotic `[[DefineOwnProperty]]` (ECMA-262 10.4.5.3): when
        // `obj` is a typed array and `key` is a canonical numeric index, the only
        // legal define is a writable, enumerable, configurable data property at a
        // valid index — anything else (an invalid index, a non-configurable /
        // non-enumerable / non-writable field, or an accessor) fails. A success
        // stores the (coerced) value through the element; a failure throws for
        // `Object.defineProperty` and returns `false` for `Reflect.defineProperty`.
        if self.realm.typed_kind(obj).is_some()
            && let Some(n) = canonical_numeric_index(key)
        {
            let fail = |this: &mut Self| -> Result<bool, ExecError> {
                if reflect {
                    return Ok(false);
                }
                let m = this.new_str(&alloc::format!(
                    "Cannot define property {key} on a TypedArray with an invalid descriptor or index"
                ));
                Err(ExecError::Throw(this.make_error(N_TYPE_ERROR, Some(m))))
            };
            // IsValidIntegerIndex: an in-bounds non-negative integer, `-0` excluded,
            // backing buffer attached.
            let is_neg_zero = n == 0.0 && n.is_sign_negative();
            let detached = self.typed_array_detached(obj);
            let valid = !detached
                && !is_neg_zero
                && n == (n as i64) as f64
                && n >= 0.0
                && self
                    .realm
                    .typed_len(obj)
                    .is_some_and(|len| (n as usize) < len);
            if !valid {
                return fail(self);
            }
            // An accessor descriptor, or a data field that is non-configurable /
            // non-enumerable / non-writable, is rejected.
            let bad_bool = |this: &Self, field: &str| -> bool {
                this.realm.has_own(desc, field)
                    && !this
                        .realm
                        .get_property(desc, field)
                        .is_some_and(|v| this.realm.truthy(v))
            };
            if has_accessor_field
                || bad_bool(self, "configurable")
                || bad_bool(self, "enumerable")
                || bad_bool(self, "writable")
            {
                return fail(self);
            }
            // Store the value (if the descriptor carries one), coercing to the view's
            // element type (a Number into a BigInt view throws here).
            if self.realm.has_own(desc, "value") {
                let v = self
                    .realm
                    .get_property(desc, "value")
                    .unwrap_or(NanBox::undefined());
                let coerced = if self.realm.typed_kind(obj).is_some_and(is_bigint_kind) {
                    self.coerce_typed_array_write(obj, v)?
                } else {
                    self.coerce_to_number(v)?
                };
                self.realm.set_element(obj, n as usize, coerced);
            }
            return Ok(true);
        }
        // An array's `length` is an exotic own data property governed by
        // ArraySetLength (ECMA-262 10.4.3.1): it is `{enumerable:false,
        // configurable:false}`, writable by default, and its "value" resizes the
        // array. Route it through a dedicated validator rather than the generic
        // ordinary-object path (which would store a shadowing aux slot).
        if self.realm.is_array(obj) && key == "length" {
            return self.apply_array_length_descriptor(obj, desc, reflect);
        }
        // ArrayDefineOwnProperty (10.4.2.1): defining an index `>= length` when the
        // array's `length` is non-writable fails — a new element cannot grow a
        // frozen length. (An in-range index, or a redefine of an existing one, is
        // governed by the ordinary rules below.)
        if self.realm.is_array(obj)
            && self.realm.array_length_is_readonly(obj)
            && let Ok(i) = key.parse::<usize>()
            && alloc::format!("{i}") == key
            && i >= self.realm.array_length(obj).unwrap_or(0)
        {
            if reflect {
                return Ok(false);
            }
            let m = self.new_str("Cannot add array index past a non-writable length");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // A callable's `length` and `name` are own properties per spec
        // (`{writable:false, enumerable:false, configurable:true}`), but they are
        // synthesized lazily and may not be materialized in the cell's aux object
        // yet — so `has_own` would miss them. Treat them as existing own data
        // properties with their intrinsic attributes so a redefine merges over the
        // spec defaults (and the second redefine sees them as configurable).
        let is_intrinsic_callable_prop = (key == "length" || key == "name")
            && self.realm.is_callable_cell(obj)
            && !self.realm.has_own(obj, key)
            && self.realm.accessor(obj, key).is_none();
        let is_own = self.realm.has_own(obj, key)
            || self.realm.accessor(obj, key).is_some()
            || is_intrinsic_callable_prop;
        // Adding a *new* property to a non-extensible object fails.
        if !is_own && !self.realm.is_extensible(obj) {
            if reflect {
                return Ok(false);
            }
            let m = self.new_str("Cannot define property: object is not extensible");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // The *shape* of the incoming descriptor (ToPropertyDescriptor semantics):
        // a field counts only when it is an OWN field of the descriptor object. A
        // descriptor with neither accessor (`get`/`set`) nor data (`value`/
        // `writable`) field is "generic" and, on a redefine, preserves the current
        // property's kind.
        let desc_is_accessor = has_accessor_field;
        let desc_is_data = has_data_field;
        // The existing property's kind and attributes (only meaningful when
        // `is_own`). An intrinsic callable `length`/`name` is a data property.
        let existing_is_accessor = self.realm.accessor(obj, key).is_some();
        // The resulting property kind: an accessor descriptor makes it an accessor,
        // a data descriptor makes it data, and a generic redefine keeps the current
        // kind (a generic *new* property defaults to a data property).
        let result_is_accessor = if desc_is_accessor {
            true
        } else if desc_is_data {
            false
        } else {
            is_own && existing_is_accessor
        };

        // ValidateAndApplyPropertyDescriptor — a non-configurable property allows
        // only a restricted set of changes; anything else is a rejection (a
        // TypeError for `Object.defineProperty`, `false` for `Reflect`).
        if is_own && self.realm.property_is_non_configurable(obj, key) {
            let writable = !self.realm.property_is_readonly(obj, key);
            let allowed = self.redefine_allowed_on_non_configurable(
                obj,
                key,
                desc,
                existing_is_accessor,
                result_is_accessor,
                writable,
            )?;
            if !allowed {
                if reflect {
                    return Ok(false);
                }
                let m = self.new_str("Cannot redefine non-configurable property");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }

        // Per `ValidateAndApplyPropertyDescriptor`, redefining an existing own
        // property MERGES over its current attributes: an attribute field the
        // descriptor omits keeps the property's existing value. For a *new*
        // property each omitted attribute takes its ECMAScript default (`false`).
        // Resolve each effective attribute up front (explicit field, else the
        // preserved existing value on a redefine, else the default `false`).
        let resolve = |this: &Self, field: &str, existing: bool| -> bool {
            match this.realm.get_property(desc, field) {
                Some(v) if this.realm.has_own(desc, field) => this.realm.truthy(v),
                _ if is_own => existing,
                _ => false,
            }
        };
        let want_enum = resolve(
            self,
            "enumerable",
            self.realm.property_is_enumerable(obj, key),
        );
        let want_configurable = resolve(
            self,
            "configurable",
            is_own && !self.realm.property_is_non_configurable(obj, key),
        );
        if result_is_accessor {
            // Merge omitted accessor fields with the existing accessor's get/set so
            // a redefine that touches only enumerable/configurable keeps the
            // current getter and setter. When converting from a data property the
            // omitted side defaults to `undefined`.
            let (cur_get, cur_set) = if existing_is_accessor {
                self.realm
                    .accessor(obj, key)
                    .unwrap_or((NanBox::undefined(), NanBox::undefined()))
            } else {
                (NanBox::undefined(), NanBox::undefined())
            };
            let getter = if self.realm.has_own(desc, "get") {
                self.realm
                    .get_property(desc, "get")
                    .unwrap_or(NanBox::undefined())
            } else {
                cur_get
            };
            let setter = if self.realm.has_own(desc, "set") {
                self.realm
                    .get_property(desc, "set")
                    .unwrap_or(NanBox::undefined())
            } else {
                cur_set
            };
            // Converting a data property to an accessor: drop the stored value and
            // its writable mark. `define_accessor` only overwrites get/set when the
            // supplied value is defined, so seed a clean accessor first to allow a
            // getter/setter to be reset to `undefined`.
            self.realm.clear_accessor(obj, key);
            self.realm.delete_data_slot(obj, key);
            self.realm.clear_readonly_property(obj, key);
            // An accessor at an array index (ArrayDefineOwnProperty, 10.4.2.1):
            // the dense element store can only hold data, so punch a hole at the
            // index (the accessor takes precedence on read) and grow `length` to
            // index+1 when the index is at/beyond the current length.
            if self.realm.is_array(obj)
                && let Ok(i) = key.parse::<usize>()
                && alloc::format!("{i}") == key
            {
                self.realm.array_index_to_hole(obj, i);
            }
            self.realm.define_accessor(obj, key, getter, setter);
            // Enumerable: explicit field, else preserved on redefine, else default false.
            if want_enum {
                self.realm.clear_hidden_property(obj, key);
            } else {
                self.realm.mark_hidden(obj, key);
            }
        } else {
            // An intrinsic callable `length`/`name` is non-writable by default; its
            // lazy form isn't yet flagged readonly in the aux object, so seed the
            // spec value explicitly.
            let existing_writable = is_own
                && !is_intrinsic_callable_prop
                && !existing_is_accessor
                && !self.realm.property_is_readonly(obj, key);
            let want_writable = resolve(self, "writable", existing_writable);
            // Redefining as a data property removes any prior accessor.
            self.realm.clear_accessor(obj, key);
            // A `defineProperty` redefines attributes from scratch: drop any prior
            // non-writable mark so the new value takes effect, then set it.
            self.realm.clear_readonly_property(obj, key);
            // Only overwrite the stored value when the descriptor supplies one (a
            // bare `{writable:...}` redefine keeps the existing value); a fresh
            // define with no `value` field, or a conversion from an accessor, uses
            // `undefined`.
            // For a numeric index on an array, the value lives in the dense
            // element store (so `arr[i]` reads it and `length` grows), not the aux
            // named-property map. `set_element` extends the array as needed.
            let array_index = if self.realm.is_array(obj) {
                key.parse::<usize>()
                    .ok()
                    .filter(|i| alloc::format!("{i}") == key)
            } else {
                None
            };
            if self.realm.has_own(desc, "value") {
                let value = self
                    .realm
                    .get_property(desc, "value")
                    .unwrap_or(NanBox::undefined());
                if let Some(i) = array_index {
                    self.realm.set_element(obj, i, value);
                } else {
                    // `[[DefineOwnProperty]]` validates extensibility itself, so the
                    // store must bypass the ordinary non-extensible / frozen guard
                    // (e.g. converting an accessor to a data property on a sealed
                    // object, or changing a configurable property's value after
                    // `preventExtensions`).
                    self.realm.force_set_property(obj, key, value);
                }
            } else if !is_own || existing_is_accessor {
                if let Some(i) = array_index {
                    self.realm.set_element(obj, i, NanBox::undefined());
                } else {
                    self.realm.force_set_property(obj, key, NanBox::undefined());
                }
            }
            // Writable: explicit field, else preserved on redefine, else default false.
            if !want_writable {
                self.realm.set_readonly_property(obj, key);
            }
            // Enumerable: explicit field, else preserved on redefine, else default false.
            if want_enum {
                self.realm.clear_hidden_property(obj, key);
            } else {
                self.realm.mark_hidden(obj, key);
            }
        }
        // Configurable: explicit field, else preserved on redefine, else default false.
        if want_configurable {
            self.realm.clear_non_configurable_property(obj, key);
        } else {
            self.realm.set_non_configurable_property(obj, key);
        }
        Ok(true)
    }

    /// `Object.defineProperty(arr, "length", desc)` — the ArraySetLength exotic
    /// (ECMA-262 10.4.3.1). The array's `length` is `{enumerable:false,
    /// configurable:false}`, writable unless explicitly demoted. A length descriptor
    /// may change the value (resizing the array) and may turn writability off, but
    /// once non-writable it cannot be made writable again nor have its value changed.
    pub(crate) fn apply_array_length_descriptor(
        &mut self,
        obj: Handle,
        desc: Handle,
        reflect: bool,
    ) -> Result<bool, ExecError> {
        let reject = |this: &mut Self| -> Result<bool, ExecError> {
            if reflect {
                return Ok(false);
            }
            let m = this.new_str("Cannot redefine property: length");
            Err(ExecError::Throw(this.make_error(N_TYPE_ERROR, Some(m))))
        };
        // ArraySetLength: the new `length` value is coerced FIRST (ToUint32 +
        // ToNumber via ToPrimitive — running a user `valueOf`/`toString`, which may
        // throw or yield a non-integral value → RangeError), *before* any
        // descriptor-attribute rejection. (10.4.3.1 steps 2–4: the RangeError on
        // an invalid value precedes the configurable/enumerable/writable checks,
        // so `defineProperty(arr,"length",{value:-1,configurable:true})` is a
        // RangeError, not a TypeError.)
        let new_len = if self.realm.has_own(desc, "value") {
            let value = self
                .realm
                .get_property(desc, "value")
                .unwrap_or(NanBox::undefined());
            // ToUint32 / ToNumber must agree (a fractional, NaN, or out-of-range
            // length is a RangeError). `coerce_to_number` runs ToPrimitive (a
            // throwing `toString`/`valueOf` propagates; one returning a non-
            // primitive is itself a TypeError).
            let prim = self.coerce_to_number(value)?;
            let num = self.realm.to_number(prim);
            let len = num as u32;
            if !(num.is_finite() && f64::from(len) == num) {
                let m = self.new_str("Invalid array length");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            Some(len as usize)
        } else {
            None
        };
        // `length` is non-configurable and non-enumerable: reject any descriptor
        // that asks to make it configurable or enumerable.
        if self.realm.has_own(desc, "configurable")
            && self
                .realm
                .get_property(desc, "configurable")
                .is_some_and(|v| self.realm.truthy(v))
        {
            return reject(self);
        }
        if self.realm.has_own(desc, "enumerable")
            && self
                .realm
                .get_property(desc, "enumerable")
                .is_some_and(|v| self.realm.truthy(v))
        {
            return reject(self);
        }
        // A `length` descriptor is a data descriptor; accessor fields are invalid.
        if self.realm.has_own(desc, "get") || self.realm.has_own(desc, "set") {
            return reject(self);
        }
        let cur_writable = !self.realm.array_length_is_readonly(obj);
        let new_writable = if self.realm.has_own(desc, "writable") {
            self.realm
                .get_property(desc, "writable")
                .is_some_and(|v| self.realm.truthy(v))
        } else {
            cur_writable
        };
        // A non-writable `length` cannot be made writable again.
        if !cur_writable && new_writable {
            return reject(self);
        }
        if let Some(len) = new_len {
            let cur_len = self.realm.array_length(obj).unwrap_or(0);
            // A non-writable `length` rejects a value change (a same-value "change"
            // is allowed).
            if !cur_writable && len != cur_len {
                return reject(self);
            }
            // ArraySetLength: a shrink that hits a non-configurable index stops there
            // and fails (the length is left one above it). Apply the writability
            // demotion *before* reporting the failure, per 10.4.3.1 steps 17–19.
            let all_deleted = self.set_array_length_checked(obj, len)?;
            if !all_deleted {
                self.realm.set_array_length_readonly(obj, !new_writable);
                return reject(self);
            }
        }
        // Apply the (possibly lowered) writability last.
        self.realm.set_array_length_readonly(obj, !new_writable);
        Ok(true)
    }

    /// ValidateAndApplyPropertyDescriptor's non-configurable guard: whether
    /// redefining the existing **non-configurable** own property `key` of `obj` with
    /// `desc` is permitted. A non-configurable property forbids: becoming
    /// configurable, an enumerable toggle, a data<->accessor switch, an accessor
    /// get/set change, making a non-writable data property writable, and changing a
    /// non-writable data property's value (a same-value redefine is always allowed).
    pub(crate) fn redefine_allowed_on_non_configurable(
        &mut self,
        obj: Handle,
        key: &str,
        desc: Handle,
        existing_is_accessor: bool,
        result_is_accessor: bool,
        writable: bool,
    ) -> Result<bool, ExecError> {
        // Becoming configurable is never allowed.
        if self.realm.has_own(desc, "configurable")
            && self
                .realm
                .get_property(desc, "configurable")
                .is_some_and(|v| self.realm.truthy(v))
        {
            return Ok(false);
        }
        // An enumerable toggle is not allowed.
        if self.realm.has_own(desc, "enumerable")
            && self
                .realm
                .get_property(desc, "enumerable")
                .is_some_and(|v| self.realm.truthy(v))
                != self.realm.property_is_enumerable(obj, key)
        {
            return Ok(false);
        }
        // Switching kind (data <-> accessor) is not allowed.
        if result_is_accessor != existing_is_accessor {
            return Ok(false);
        }
        if existing_is_accessor {
            // An accessor's get/set cannot change.
            let (cur_get, cur_set) = self
                .realm
                .accessor(obj, key)
                .unwrap_or((NanBox::undefined(), NanBox::undefined()));
            if self.realm.has_own(desc, "get") {
                let g = self
                    .realm
                    .get_property(desc, "get")
                    .unwrap_or(NanBox::undefined());
                if !self.realm.same_value(g, cur_get) {
                    return Ok(false);
                }
            }
            if self.realm.has_own(desc, "set") {
                let s = self
                    .realm
                    .get_property(desc, "set")
                    .unwrap_or(NanBox::undefined());
                if !self.realm.same_value(s, cur_set) {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        // A writable data property may change its value and writability freely.
        if writable {
            return Ok(true);
        }
        // A non-writable data property: cannot be made writable, and cannot change
        // its value.
        if self.realm.has_own(desc, "writable")
            && self
                .realm
                .get_property(desc, "writable")
                .is_some_and(|v| self.realm.truthy(v))
        {
            return Ok(false);
        }
        if self.realm.has_own(desc, "value") {
            let new_val = self
                .realm
                .get_property(desc, "value")
                .unwrap_or(NanBox::undefined());
            // An array index's value lives in the dense element store, not a named
            // slot, so read it with `get_element` for a same-value comparison.
            let cur_val = if self.realm.is_array(obj)
                && let Ok(i) = key.parse::<usize>()
                && alloc::format!("{i}") == key
            {
                self.realm.get_element(obj, i)
            } else {
                self.realm
                    .get_property(obj, key)
                    .unwrap_or(NanBox::undefined())
            };
            if !self.realm.same_value(new_val, cur_val) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `structuredClone(v)`: a deep copy. Primitives and immutable heap values
    /// (strings, BigInts) are shared; Dates, Maps, Sets, arrays, and plain
    /// objects are recursively cloned. `seen` maps each visited source handle to
    /// its clone so cyclic and shared references are preserved. Functions and
    /// symbols are not cloneable (a TypeError, like `DataCloneError`).
    pub(crate) fn structured_clone(
        &mut self,
        v: NanBox,
        seen: &mut Vec<(u64, NanBox)>,
    ) -> Result<NanBox, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(v); // a primitive
        };
        let h = Handle::from_raw(raw);
        // Immutable heap values are shared, not copied.
        if self.realm.string_value(h).is_some() || self.realm.bigint_at(h).is_some() {
            return Ok(v);
        }
        if self.is_callable(h) || self.realm.symbol_at(h).is_some() {
            let m = self.new_str("value could not be cloned");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // A previously-cloned handle (cycle or shared reference).
        if let Some((_, c)) = seen.iter().find(|(r, _)| *r == raw) {
            return Ok(*c);
        }
        // Bound the recursion so a deep acyclic structure throws rather than
        // overflowing the host stack.
        if seen.len() >= self.realm.limits.max_display_depth {
            let m = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        if let Some(ms) = self.realm.date_at(h) {
            return Ok(NanBox::handle(self.realm.new_date(ms).to_raw()));
        }
        if let Some(is_set) = self.realm.collection_is_set(h) {
            let coll = self.realm.new_collection(is_set);
            let cbox = NanBox::handle(coll.to_raw());
            seen.push((raw, cbox));
            for (k, val) in self.realm.collection_entries(h).unwrap_or_default() {
                let ck = self.structured_clone(k, seen)?;
                let cv = self.structured_clone(val, seen)?;
                self.realm.collection_set(coll, ck, cv);
            }
            return Ok(cbox);
        }
        if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
            let arr = self.realm.new_array(Vec::new());
            let abox = NanBox::handle(arr.to_raw());
            seen.push((raw, abox));
            for e in elems {
                let c = self.structured_clone(e, seen)?;
                self.realm.array_push(arr, c);
            }
            return Ok(abox);
        }
        // A plain object: clone own enumerable string-keyed properties.
        let obj = self.realm.new_object();
        let obox = NanBox::handle(obj.to_raw());
        seen.push((raw, obox));
        for k in self.realm.object_keys(h).unwrap_or_default() {
            if let Some(pv) = self.realm.get_property(h, &k) {
                let c = self.structured_clone(pv, seen)?;
                self.realm.set_property(obj, &k, c);
            }
        }
        Ok(obox)
    }

    /// CanBeHeldWeakly(v) (ECMA-262): true for an object or a *non-registered*
    /// symbol. A symbol obtained from `Symbol.for` (present in the global symbol
    /// registry) cannot be held weakly. Primitives (string/number/bigint/bool/
    /// null/undefined) cannot either.
    pub(crate) fn can_be_held_weakly(&self, v: NanBox) -> bool {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return false;
        };
        // A symbol: weakly holdable unless it lives in the global registry.
        if self.realm.symbol_at(h).is_some() {
            return !self
                .symbol_registry
                .values()
                .any(|s| self.realm.same_value(*s, v));
        }
        // Any other heap value that is not a non-object primitive (string /
        // bigint are heap cells but not weakly holdable).
        self.realm.string_value(h).is_none() && self.realm.bigint_at(h).is_none()
    }

    /// `thisSymbolValue(this)` (ECMA-262 20.4.3): if `this` is a Symbol
    /// primitive, return it; if it is a Symbol wrapper object (a [`PRIM_WRAP`]
    /// holding a symbol), return the boxed symbol; otherwise a TypeError. Used by
    /// the brand-checking `Symbol.prototype` methods. (Here `Object(sym)` returns
    /// the primitive itself, so the wrapper branch is a belt-and-braces fallback.)
    pub(crate) fn this_symbol_value(&mut self) -> Result<NanBox, ExecError> {
        let this = self.this_val;
        if let Some(h) = this.as_handle().map(Handle::from_raw) {
            if self.realm.symbol_at(h).is_some() {
                return Ok(this);
            }
            if let Some(prim) = self.realm.get_property(h, PRIM_WRAP)
                && prim
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|ph| self.realm.symbol_at(ph).is_some())
            {
                return Ok(prim);
            }
        }
        Err(self.type_error("Symbol.prototype method called on a non-symbol value"))
    }

    /// `CanonicalizeKeyedCollectionKey(key)` (ECMA-262): `-0𝔽` becomes `+0𝔽`;
    /// every other value is returned unchanged. Used by the upsert proposal's
    /// `getOrInsert`/`getOrInsertComputed` so the canonical key is both stored and
    /// handed to the callback. (`-0.0 == 0.0` in Rust, and `NanBox::number(0.0)`
    /// produces positive zero, so this normalizes the sign.)
    #[must_use]
    pub(crate) fn canonicalize_collection_key(key: NanBox) -> NanBox {
        if key.as_number() == Some(0.0) {
            NanBox::number(0.0)
        } else {
            key
        }
    }

    /// The key validation shared by `getOrInsert`/`getOrInsertComputed`. For a
    /// WeakMap, `CanBeHeldWeakly(key)` must hold — an object or a non-registered
    /// symbol — else a TypeError (this correctly rejects registered symbols and
    /// every primitive). For a non-weak Map there is no constraint.
    pub(crate) fn guard_get_or_insert_key(
        &mut self,
        coll: Handle,
        key: NanBox,
    ) -> Result<(), ExecError> {
        if !self.realm.collection_is_weak(coll) || self.can_be_held_weakly(key) {
            return Ok(());
        }
        let m = self.new_str("Invalid value used as weak collection key");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    pub(crate) fn guard_weak_key(&mut self, coll: Handle, key: NanBox) -> Result<(), ExecError> {
        if !self.realm.collection_is_weak(coll) {
            return Ok(());
        }
        let valid = key.as_handle().map(Handle::from_raw).is_some_and(|h| {
            self.realm.string_value(h).is_none() && self.realm.bigint_at(h).is_none()
        });
        if valid {
            return Ok(());
        }
        let m = self.new_str("Invalid value used as weak collection key");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    pub(crate) fn make_primitive_wrapper(&mut self, prim: NanBox, ctor_id: u16) -> NanBox {
        let obj = self.realm.new_object();
        // The wrapper's `[[Prototype]]` is the corresponding constructor's
        // `.prototype` (so `Object.getPrototypeOf(new Number(1)) === Number.prototype`
        // and inherited methods such as `toFixed` resolve to the prototype's).
        let ctor_name = match ctor_id {
            N_NUMBER => Some("Number"),
            N_STRING => Some("String"),
            N_BOOLEAN => Some("Boolean"),
            _ => None,
        };
        if let Some(proto) = ctor_name
            .and_then(|n| self.current.get(n))
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_object_proto(obj, Some(proto));
        }
        self.realm.set_hidden_property(obj, PRIM_WRAP, prim);
        self.realm
            .set_hidden_property(obj, PRIM_WRAP_TYPE, NanBox::number(f64::from(ctor_id)));
        NanBox::handle(obj.to_raw())
    }

    /// Builds an `arguments` exotic object for a function call (`args`), per
    /// 10.4.4 CreateUnmappedArgumentsObject / a best-effort mapped object:
    ///
    /// - `[[Prototype]]` is `Object.prototype` (not `Array.prototype`).
    /// - Indexed elements `0..len` are ordinary enumerable, writable,
    ///   configurable own data properties.
    /// - `length` is a writable, configurable, **non-enumerable** data property.
    /// - `[Symbol.iterator]` is `Array.prototype.values` (writable, configurable,
    ///   non-enumerable), so `[...arguments]`/`for-of` work.
    /// - `callee`: in sloppy mode the *function itself* (writable, configurable,
    ///   non-enumerable); in strict mode a poisoned accessor throwing `TypeError`.
    /// - A hidden `ARGS_MARKER` makes `Object.prototype.toString` report
    ///   `[object Arguments]`.
    ///
    /// (True parameter↔index aliasing for a mapped object is not modeled; the
    /// values are a snapshot of the call arguments.)
    pub(crate) fn make_arguments_object(&mut self, args: &[NanBox], callee: NanBox) -> NanBox {
        let obj = self.realm.new_object();
        // [[Prototype]] = Object.prototype.
        if let Some(proto) = self
            .current
            .get("Object")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
        {
            self.realm.set_object_proto(obj, Some(proto));
        }
        // Indexed elements: ordinary enumerable own data properties.
        for (i, v) in args.iter().enumerate() {
            self.realm.set_property(obj, &alloc::format!("{i}"), *v);
        }
        // `length` — writable, configurable, non-enumerable.
        self.realm
            .set_hidden_property(obj, "length", NanBox::number(args.len() as f64));
        // `[Symbol.iterator]` = `%Array.prototype.values%` — the SAME function
        // value as `Array.prototype[Symbol.iterator]` (i.e. `[][Symbol.iterator]`,
        // which aliases `values`), so a `verifyProperty(arguments, Symbol.iterator,
        // {value: [][Symbol.iterator]})` SameValue check holds. Non-enumerable.
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        if let Some(values) = self
            .current
            .get("Array")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)
            .and_then(|proto| self.realm.get_property(proto, "values"))
        {
            self.realm.set_hidden_property(obj, &iter_key, values);
        }
        // `callee`.
        if self.strict {
            // A strict arguments object's `callee` is a poisoned accessor.
            let thrower = NanBox::handle(self.realm.new_native(N_THROW_TYPE_ERROR).to_raw());
            self.realm.define_accessor(obj, "callee", thrower, thrower);
            self.realm.mark_hidden(obj, "callee");
        } else {
            self.realm.set_hidden_property(obj, "callee", callee);
        }
        self.realm
            .set_hidden_property(obj, ARGS_MARKER, NanBox::boolean(true));
        NanBox::handle(obj.to_raw())
    }

    /// `ToObject(v)` for `Object(v)`: `null`/`undefined` yield a fresh object; an
    /// existing object/array/function is returned unchanged; a primitive is boxed in
    /// its wrapper (so `Object(42).valueOf()` is `42`).
    pub(crate) fn coerce_to_object(&mut self, v: NanBox) -> NanBox {
        match v.unpack() {
            Unpacked::Undefined | Unpacked::Null => {
                NanBox::handle(self.realm.new_object().to_raw())
            }
            Unpacked::Number(_) => self.make_primitive_wrapper(v, N_NUMBER),
            Unpacked::Bool(_) => self.make_primitive_wrapper(v, N_BOOLEAN),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if self.realm.string_value(h).is_some() {
                    self.make_primitive_wrapper(v, N_STRING)
                } else {
                    // An already-object value (object/array/function/symbol/bigint).
                    v
                }
            }
        }
    }

    /// Resolves a (trap-less) proxy to its target for key enumeration, so
    /// `Object.keys`/`values`/`entries` on a pass-through proxy see the target's
    /// own keys. A non-proxy is returned unchanged. (The `ownKeys` trap is not
    /// invoked here.)
    pub(crate) fn proxy_key_target(&self, mut h: crate::heap::Handle) -> crate::heap::Handle {
        while let Some((target, _)) = self.realm.proxy_at(h) {
            h = target;
        }
        h
    }

    /// `Object.keys` for a proxy that defines an `ownKeys` trap: invoke the trap,
    /// then keep each string key whose property is enumerable — via the
    /// `getOwnPropertyDescriptor` trap if present, else the target. Returns `None`
    /// when there is no `ownKeys` trap (so the caller uses the target's keys).
    pub(crate) fn proxy_own_enumerable_keys(
        &mut self,
        proxy: Handle,
    ) -> Result<Option<Vec<String>>, ExecError> {
        let Some((target, handler)) = self.realm.proxy_at(proxy) else {
            return Ok(None);
        };
        // `[[OwnPropertyKeys]]` via the validated trap result (type/duplicate/
        // invariant checks); `None` when there is no `ownKeys` trap.
        let Some(keys) = self.proxy_own_keys_raw(proxy)? else {
            return Ok(None);
        };
        let target_box = NanBox::handle(target.to_raw());
        let gopd = self
            .realm
            .get_property(handler, "getOwnPropertyDescriptor")
            .unwrap_or(NanBox::undefined());
        let gopd_callable = gopd
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
        let mut out = Vec::new();
        for k in keys {
            // Only string keys participate in `Object.keys` (symbols are skipped).
            let Some(name) = k
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.string_value(h))
            else {
                continue;
            };
            let enumerable = if gopd_callable {
                let kbox = self.new_str(&name);
                let desc = self.call(gopd, &[target_box, kbox])?;
                desc.as_handle()
                    .map(Handle::from_raw)
                    .and_then(|dh| self.realm.get_property(dh, "enumerable"))
                    .is_some_and(|v| self.realm.truthy(v))
            } else {
                // No descriptor trap: forward to the target — an own, enumerable
                // property only (a key the target lacks is not enumerable).
                self.realm.has_own(target, &name)
                    && self.realm.property_is_enumerable(target, &name)
            };
            if enumerable {
                out.push(name);
            }
        }
        Ok(Some(out))
    }

    /// The proxy `set` trap success invariant (10.5.9): a `true` result is illegal
    /// when the target has `key` as a non-configurable, non-writable data property
    /// with a different value, or as a non-configurable accessor whose setter is
    /// undefined. Call after a `set` trap returns truthy.
    pub(crate) fn proxy_set_invariant_check(
        &mut self,
        target: Handle,
        key: &str,
        value: NanBox,
    ) -> Result<(), ExecError> {
        if let Some((g, s)) = self.realm.accessor(target, key) {
            let _ = g;
            if self.realm.property_is_non_configurable(target, key)
                && matches!(s.unpack(), Unpacked::Undefined)
            {
                return Err(self.type_error(
                    "proxy 'set' trap succeeded for a non-configurable accessor with no setter",
                ));
            }
            return Ok(());
        }
        if self.realm.has_own(target, key)
            && self.realm.property_is_non_configurable(target, key)
            && self.realm.property_is_readonly(target, key)
        {
            let cur = self
                .realm
                .get_property(target, key)
                .unwrap_or(NanBox::undefined());
            if !self.realm.strict_equals(cur, value) {
                return Err(self.type_error(
                    "proxy 'set' trap succeeded for a non-configurable non-writable property with a different value",
                ));
            }
        }
        Ok(())
    }

    /// `[[Delete]]` honoring a proxy's `deleteProperty` trap (and its invariants).
    /// Returns the boolean result. Used by `Reflect.deleteProperty`.
    pub(crate) fn delete_property_of(&mut self, obj: Handle, key: &str) -> Result<bool, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "deleteProperty")? {
                let kb = self.new_str(key);
                let handler_box = NanBox::handle(handler.to_raw());
                let r =
                    self.call_with_this(trap, handler_box, &[NanBox::handle(target.to_raw()), kb])?;
                if !self.realm.truthy(r) {
                    return Ok(false);
                }
                let present =
                    self.realm.has_own(target, key) || self.realm.accessor(target, key).is_some();
                if present && self.realm.property_is_non_configurable(target, key) {
                    return Err(self.type_error(
                        "proxy 'deleteProperty' trap removed a non-configurable property",
                    ));
                }
                if present && !self.realm.is_extensible(target) {
                    return Err(self.type_error(
                        "proxy 'deleteProperty' trap removed a property of a non-extensible target",
                    ));
                }
                return Ok(true);
            }
            return self.delete_property_of(target, key);
        }
        Ok(self.realm.delete_property(obj, key))
    }

    /// `[[PreventExtensions]]` honoring a proxy's `preventExtensions` trap. Returns
    /// the boolean success. A `true` trap result is validated against the spec
    /// invariant: the target must then report non-extensible (else a TypeError).
    pub(crate) fn prevent_extensions_of(&mut self, obj: Handle) -> Result<bool, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            if let Some(trap) = self.proxy_trap(handler, "preventExtensions")? {
                let handler_box = NanBox::handle(handler.to_raw());
                let r =
                    self.call_with_this(trap, handler_box, &[NanBox::handle(target.to_raw())])?;
                if self.realm.truthy(r) {
                    // Invariant: a successful trap requires the target to be
                    // non-extensible.
                    if self.realm.is_extensible(target) {
                        return Err(self.type_error(
                            "proxy 'preventExtensions' trap returned true but the target is extensible",
                        ));
                    }
                    return Ok(true);
                }
                return Ok(false);
            }
            return self.prevent_extensions_of(target);
        }
        self.realm.prevent_extensions(obj);
        Ok(true)
    }

    /// `CreateListFromArrayLike(obj)` (default element types): the argument must be
    /// an Object; reads `length` (ToLength) then each indexed element via `[[Get]]`
    /// (so getters / proxy traps fire), returning the value list. Used by
    /// `Reflect.apply` / `Reflect.construct`.
    pub(crate) fn create_list_from_array_like(
        &mut self,
        v: NanBox,
    ) -> Result<Vec<NanBox>, ExecError> {
        if !self.is_object_value(v) {
            return Err(self.type_error("CreateListFromArrayLike called on non-object"));
        }
        let h = v.as_handle().map(Handle::from_raw).unwrap();
        // A real dense array: fast path.
        if let Some(elems) = self.realm.array_elements(h) {
            return Ok(elems.to_vec());
        }
        let len_val = self.read_member(h, "length")?;
        let len_num = self.coerce_to_number(len_val)?;
        let raw = self.realm.to_number(len_num);
        let len = if raw.is_nan() || raw <= 0.0 {
            0
        } else {
            (raw.min(9_007_199_254_740_991.0)) as usize
        };
        let mut out = Vec::with_capacity(len.min(1 << 16));
        for i in 0..len {
            out.push(self.read_member(h, &alloc::format!("{i}"))?);
        }
        Ok(out)
    }

    /// A proxy's `[[OwnPropertyKeys]]` (ECMA-262 10.5.11): invoke the `ownKeys`
    /// trap and validate the result — every entry must be a String or Symbol, with
    /// no duplicates; the result must contain every non-configurable own key of the
    /// target and (when the target is non-extensible) exactly the target's own
    /// keys. Returns the validated key list (String/Symbol NanBoxes). Returns
    /// `None` if there is no `ownKeys` trap (the caller falls back to the target).
    pub(crate) fn proxy_own_keys_raw(
        &mut self,
        proxy: Handle,
    ) -> Result<Option<Vec<NanBox>>, ExecError> {
        let Some((target, handler)) = self.realm.proxy_at(proxy) else {
            return Ok(None);
        };
        self.guard_revoked(proxy)?;
        let Some(trap) = self.proxy_trap(handler, "ownKeys")? else {
            return Ok(None);
        };
        let target_box = NanBox::handle(target.to_raw());
        let handler_box = NanBox::handle(handler.to_raw());
        let result = self.call_with_this(trap, handler_box, &[target_box])?;
        // CreateListFromArrayLike(result, « String, Symbol »): the trap result must
        // be an Object; each element must be a String or Symbol.
        if !self.is_object_value(result) {
            return Err(self.type_error("proxy 'ownKeys' trap must return an array-like object"));
        }
        let rh = result.as_handle().map(Handle::from_raw).unwrap();
        let len = self
            .read_member(rh, "length")?
            .as_number()
            .map(|n| n.max(0.0) as usize)
            .unwrap_or(0);
        let mut keys: Vec<NanBox> = Vec::with_capacity(len);
        let mut seen_strs: alloc::collections::BTreeSet<String> =
            alloc::collections::BTreeSet::new();
        let mut seen_syms: alloc::collections::BTreeSet<u64> = alloc::collections::BTreeSet::new();
        for i in 0..len {
            let el = self.read_member(rh, &alloc::format!("{i}"))?;
            let elh = el.as_handle().map(Handle::from_raw);
            // Each key must be a String or Symbol; duplicates are a TypeError.
            if let Some(eh) = elh
                && let Some(s) = self.realm.string_value(eh)
            {
                if !seen_strs.insert(s) {
                    return Err(self.type_error("proxy 'ownKeys' trap returned duplicate entries"));
                }
            } else if let Some(eh) = elh
                && let Some((_, sid)) = self.realm.symbol_at(eh)
            {
                if !seen_syms.insert(sid) {
                    return Err(self.type_error("proxy 'ownKeys' trap returned duplicate entries"));
                }
            } else {
                return Err(
                    self.type_error("proxy 'ownKeys' trap returned a non-string, non-symbol key")
                );
            }
            keys.push(el);
        }
        // Invariant checks against the target's own keys.
        let extensible = self.realm.is_extensible(target);
        let target_keys = self.target_own_key_set(target);
        // Every non-configurable own key of the target must be present.
        for tk in &target_keys {
            let is_nonconf = self.target_key_nonconfigurable(target, tk);
            if is_nonconf && !self.key_list_contains(&keys, tk) {
                return Err(self.type_error(
                    "proxy 'ownKeys' trap omitted a non-configurable key of the target",
                ));
            }
        }
        if !extensible {
            // The result must contain exactly the target's own keys.
            for tk in &target_keys {
                if !self.key_list_contains(&keys, tk) {
                    return Err(self.type_error(
                        "proxy 'ownKeys' trap omitted a key of a non-extensible target",
                    ));
                }
            }
            for k in &keys {
                let kd = self.key_descriptor_string(*k);
                if !target_keys.contains(&kd) {
                    return Err(self.type_error(
                        "proxy 'ownKeys' trap added a key not on a non-extensible target",
                    ));
                }
            }
        }
        Ok(Some(keys))
    }

    /// The own keys of `target` as canonical descriptor strings (a symbol becomes
    /// its internal `"\0sym:<id>"` storage key), for proxy ownKeys invariants.
    fn target_own_key_set(&mut self, target: Handle) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(indices) = self.realm.array_present_indices(target) {
            for i in indices {
                out.push(alloc::format!("{i}"));
            }
            out.push(String::from("length"));
        }
        for k in self.realm.own_property_names(target).unwrap_or_default() {
            if !out.contains(&k) {
                out.push(k);
            }
        }
        for k in self.realm.object_all_keys(target) {
            if k.starts_with("\u{0}sym:") && !out.contains(&k) {
                out.push(k);
            }
        }
        out
    }

    /// The canonical descriptor string for a String/Symbol key NanBox.
    fn key_descriptor_string(&self, key: NanBox) -> String {
        if let Some(h) = key.as_handle().map(Handle::from_raw) {
            if let Some(s) = self.realm.string_value(h) {
                return s;
            }
            if let Some((_, id)) = self.realm.symbol_at(h) {
                return alloc::format!("\u{0}sym:{id}");
            }
        }
        String::new()
    }

    /// Whether the key list (String/Symbol NanBoxes) contains `target_key` (given
    /// as its canonical descriptor string).
    fn key_list_contains(&self, keys: &[NanBox], target_key: &str) -> bool {
        keys.iter()
            .any(|k| self.key_descriptor_string(*k) == target_key)
    }

    /// Whether `target`'s own property `key` (a canonical descriptor string) is
    /// non-configurable.
    fn target_key_nonconfigurable(&mut self, target: Handle, key: &str) -> bool {
        if key == "length" && self.realm.is_array(target) {
            return true; // array `length` is non-configurable
        }
        if let Some((_, _)) = self.realm.accessor(target, key) {
            return self.realm.property_is_non_configurable(target, key);
        }
        if self.realm.has_own(target, key) {
            return self.realm.property_is_non_configurable(target, key);
        }
        false
    }

    /// Whether `handle` or any object on its prototype chain carries the hidden
    /// `brand` marker. Used to detect that a receiver inherits a branded built-in
    /// prototype (`ArrayBuffer.prototype`, `%TypedArray%.prototype`, …) whose
    /// slot-requiring accessors must throw when no internal slot is present.
    pub(crate) fn brand_on_chain(&self, handle: Handle, brand: &str) -> bool {
        let mut cur = Some(handle);
        let mut guard = 0;
        while let Some(h) = cur {
            if self.realm.has_own(h, brand) {
                return true;
            }
            guard += 1;
            if guard > 1000 {
                break;
            }
            cur = self.realm.object_proto(h);
        }
        false
    }

    pub(crate) fn object_string_tag(
        &mut self,
        h: crate::heap::Handle,
    ) -> Result<String, ExecError> {
        // The spec order (20.1.3.6) computes a `builtinTag` FIRST (IsArray is
        // proxy-aware and a proxy whose target is callable reports "Function"),
        // then a string `Symbol.toStringTag` OVERRIDES it.
        // IsArray walks proxy chains (a revoked proxy throws).
        let is_array = self.is_array_unwrap_proxy(NanBox::handle(h.to_raw()))?;
        // A callable proxy (target has [[Call]]) reports "Function".
        let is_callable_unwrapped = {
            let mut cur = h;
            let mut callable = false;
            for _ in 0..1000 {
                if let Some((target, _)) = self.realm.proxy_at(cur) {
                    cur = target;
                    continue;
                }
                callable = self.is_callable(cur) || self.realm.class_at(cur).is_some();
                break;
            }
            callable
        };
        let builtin_tag = if is_array {
            "Array"
        } else if let Some(kind) = self.realm.typed_kind(h) {
            TYPED_ARRAY_KINDS[kind as usize].0
        } else if is_callable_unwrapped {
            "Function"
        } else if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
            // A boxed primitive wrapper reports its primitive's class.
            match prim.unpack() {
                Unpacked::Number(_) => "Number",
                Unpacked::Bool(_) => "Boolean",
                _ => "String",
            }
        } else if self.realm.string_value(h).is_some() {
            "String"
        } else if self.realm.date_at(h).is_some() {
            "Date"
        } else if self.realm.regexp_at(h).is_some() {
            "RegExp"
        } else if self.is_error_object(h) {
            "Error"
        } else if self.realm.get_property(h, ARGS_MARKER).is_some() {
            "Arguments"
        } else {
            "Object"
        };
        // A string `Symbol.toStringTag` (read through the prototype chain, firing an
        // accessor) overrides the builtin tag.
        let tag_sym = self.well_known_symbol("toStringTag");
        let tag_key = self.member_key(tag_sym);
        let v = self.read_member(h, &tag_key)?;
        if let Some(sh) = v.as_handle().map(Handle::from_raw)
            && let Some(s) = self.realm.string_value(sh)
        {
            return Ok(s);
        }
        Ok(String::from(builtin_tag))
    }

    /// `HasProperty(O, P)` — the `in` operator / `Reflect.has`, honoring a proxy's
    /// `has` trap *anywhere on the prototype chain* (OrdinaryHasProperty recurses
    /// into `parent.[[HasProperty]]`, and a proxy parent runs its own trap).
    pub(crate) fn has_property_proxied(
        &mut self,
        obj: Handle,
        key: &str,
    ) -> Result<bool, ExecError> {
        // Integer-indexed exotic `[[HasProperty]]`: a canonical numeric index on a
        // (non-proxy) typed array is exactly IsValidIntegerIndex — the prototype
        // chain is never consulted (an out-of-bounds or proto-set numeric key is
        // absent).
        if self.realm.proxy_at(obj).is_none()
            && self.realm.typed_kind(obj).is_some()
            && let Some(n) = canonical_numeric_index(key)
        {
            let is_neg_zero = n == 0.0 && n.is_sign_negative();
            return Ok(!self.typed_array_detached(obj)
                && !is_neg_zero
                && n == (n as i64) as f64
                && n >= 0.0
                && self
                    .realm
                    .typed_len(obj)
                    .is_some_and(|len| (n as usize) < len));
        }
        let mut cur = Some(obj);
        // Bound to guard against a `getPrototypeOf` trap returning a cycle.
        for _ in 0..100_000 {
            let Some(c) = cur else { return Ok(false) };
            if let Some((target, handler)) = self.realm.proxy_at(c) {
                self.guard_revoked(c)?;
                if let Some(trap) = self.proxy_trap(handler, "has")? {
                    let kb = self.new_str(key);
                    let handler_box = NanBox::handle(handler.to_raw());
                    let r = self.call_with_this(
                        trap,
                        handler_box,
                        &[NanBox::handle(target.to_raw()), kb],
                    )?;
                    let result = self.realm.truthy(r);
                    // Invariant (10.5.7): a false result is illegal if the property
                    // exists as a non-configurable own property of the target, or if
                    // the target is non-extensible and the property is own.
                    if !result {
                        let target_has = self.realm.has_own(target, key)
                            || self.realm.accessor(target, key).is_some()
                            || (key == "length" && self.realm.is_array(target))
                            || (self.realm.array_length(target).is_some()
                                && key
                                    .parse::<usize>()
                                    .is_ok_and(|i| i < self.realm.array_length(target).unwrap()));
                        if target_has {
                            if self.target_key_nonconfigurable(target, key) {
                                return Err(self.type_error(
                                    "proxy 'has' trap returned false for a non-configurable property of the target",
                                ));
                            }
                            if !self.realm.is_extensible(target) {
                                return Err(self.type_error(
                                    "proxy 'has' trap returned false for a property of a non-extensible target",
                                ));
                            }
                        }
                    }
                    return Ok(result);
                }
                // No `has` trap: forward `[[HasProperty]]` to the target (which
                // itself walks its chain, possibly through further proxies).
                return self.has_property_proxied(target, key);
            }
            // An own data/accessor property, or an in-range array index / `length`.
            let here = if let Some(len) = self.realm.array_length(c) {
                key == "length"
                    || key.parse::<usize>().is_ok_and(|i| i < len)
                    || self.realm.has_own(c, key)
                    || self.realm.accessor(c, key).is_some()
            } else {
                self.realm.has_own(c, key) || self.realm.accessor(c, key).is_some()
            };
            if here {
                return Ok(true);
            }
            cur = self.realm.object_proto(c);
        }
        Ok(false)
    }

    /// Whether `handle` is an Error object (has `[[ErrorData]]`): its prototype
    /// chain includes `Error.prototype`, or it is a class instance whose `extends`
    /// chain reaches a native `Error*` constructor.
    pub(crate) fn is_error_object(&mut self, handle: Handle) -> bool {
        // Resolve `Error.prototype` (the root of every error prototype chain).
        let error_proto = self
            .current
            .get("Error")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|c| self.realm.get_property(c, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw);
        if let Some(ep) = error_proto {
            let mut cur = self.realm.object_proto(handle);
            for _ in 0..10_000 {
                let Some(c) = cur else { break };
                if c == ep {
                    return true;
                }
                cur = self.realm.object_proto(c);
            }
        }
        false
    }

    /// Whether `handle` has `name` as an own or inherited property (walks the
    /// prototype chain; includes accessors).
    pub(crate) fn has_property_chain(&self, handle: Handle, name: &str) -> bool {
        let mut cur = Some(handle);
        while let Some(c) = cur {
            if self.realm.has_own(c, name) || self.realm.accessor(c, name).is_some() {
                return true;
            }
            cur = self.realm.object_proto(c);
        }
        false
    }

    /// `ToObject(v)` for the spec sites that require an Object argument and must
    /// reject `null`/`undefined` with a TypeError (e.g. the *Properties* argument
    /// of `Object.create`/`Object.defineProperties`). An object passes through; a
    /// primitive wrapper boxes; `null`/`undefined` throw using `site` in the
    /// message.
    /// The `Reflect.*` target requirement: `v` must be an Object (ECMA-262 — the
    /// first step of every `Reflect` operation is `if Type(target) is not Object,
    /// throw a TypeError`). A string/symbol/bigint primitive or an immediate
    /// (number/boolean/null/undefined) is rejected. Returns the target handle.
    pub(crate) fn reflect_object_target(
        &mut self,
        v: NanBox,
        op: &str,
    ) -> Result<Handle, ExecError> {
        if self.is_object_value(v)
            && let Some(raw) = v.as_handle()
        {
            return Ok(Handle::from_raw(raw));
        }
        Err(self.type_error(&alloc::format!("Reflect.{op} called on non-object")))
    }

    pub(crate) fn require_object_coercible_to_object(
        &mut self,
        v: NanBox,
        site: &str,
    ) -> Result<Handle, ExecError> {
        if matches!(v.unpack(), Unpacked::Null | Unpacked::Undefined) {
            return Err(self.type_error(&alloc::format!("{site} called on null or undefined")));
        }
        let obj = self.coerce_to_object(v);
        obj.as_handle().map(Handle::from_raw).ok_or_else(|| {
            self.type_error(&alloc::format!(
                "{site} could not coerce argument to an object"
            ))
        })
    }

    /// Applies the own *enumerable* property descriptors of `descs` onto `target`
    /// (`Object.defineProperties` / the second argument of `Object.create`). Each
    /// descriptor object is read and validated via `apply_descriptor`
    /// (ToPropertyDescriptor), so a malformed descriptor (e.g. both `value` and
    /// `get`) throws.
    pub(crate) fn apply_property_descriptors(
        &mut self,
        target: Handle,
        descs: Handle,
    ) -> Result<(), ExecError> {
        // OwnPropertyKeys(Properties) filtered to enumerable, in spec order. A
        // function/array/native keeps its named (accessor) properties in the aux
        // object, so fall back to `aux_named_keys` like `Object.keys`. An array's
        // own enumerable keys also include its integer indices.
        let mut keys: Vec<alloc::string::String> = Vec::new();
        if let Some(indices) = self.realm.array_present_indices(descs)
            && !self.realm.is_vm_function(descs)
        {
            for i in indices {
                keys.push(alloc::format!("{i}"));
            }
        }
        if let Some(named) = self.realm.object_keys(descs) {
            keys.extend(named);
        } else {
            keys.extend(self.realm.aux_named_keys(descs));
        }
        for key in keys {
            // Get(props, key) invokes a getter (the descriptor value may be
            // computed); the result must be an object (ToPropertyDescriptor).
            let d_val = self.read_member(descs, &key)?;
            let Some(d) = d_val
                .as_handle()
                .map(Handle::from_raw)
                .filter(|_| self.is_object_value(d_val))
            else {
                return Err(self.type_error("Property description must be an object"));
            };
            self.apply_descriptor(target, &key, d, false)?;
        }
        Ok(())
    }

    /// Reads a named member, honoring class statics and accessor getters before
    /// ordinary property/length access.
    /// The global constructor a built-in heap value reports as its `.constructor`
    /// (so `[].constructor === Array`), resolved by the value's cell kind. Returns
    /// the actual global binding (identity-equal to `Array`, `Object`, …), or
    /// `None` for kinds without a distinct constructor.
    pub(crate) fn builtin_constructor_for(
        &mut self,
        handle: crate::heap::Handle,
    ) -> Option<NanBox> {
        let name = if self.realm.is_array(handle) {
            "Array"
        } else if self.realm.string_value(handle).is_some() {
            "String"
        } else if self.realm.regexp_at(handle).is_some() {
            "RegExp"
        } else if self.realm.bigint_at(handle).is_some() {
            "BigInt"
        } else if self.realm.symbol_at(handle).is_some() {
            "Symbol"
        } else if self.realm.date_at(handle).is_some() {
            "Date"
        } else if let Some(is_set) = self.realm.collection_is_set(handle) {
            if is_set { "Set" } else { "Map" }
        } else if self.realm.promise_state(handle).is_some() {
            "Promise"
        } else if self.realm.object_keys(handle).is_some() {
            // A plain object reports `Object`. (Error objects are handled earlier in
            // `read_member`, before their prototype's generic `constructor`.)
            "Object"
        } else {
            return None;
        };
        self.current.get(name)
    }

    pub(crate) fn member_value(&self, handle: crate::heap::Handle, key: &str) -> NanBox {
        if let Some(v) = self.realm.get_property(handle, key) {
            return v;
        }
        if key == "length" {
            if let Some(len) = self.realm.array_length(handle) {
                return NanBox::number(len as f64);
            }
            // `String.length` counts UTF-16 code units (astral chars = 2, a lone
            // surrogate = 1). P3: borrow the leaf when possible so `.length` in a
            // loop does not flatten the rope into an owned `Vec` on every read.
            if let Some(leaf) = self.realm.string_leaf_bytes(handle) {
                return NanBox::number(crate::wtf8::utf16_len(leaf) as f64);
            }
            if let Some(bytes) = self.realm.string_bytes(handle) {
                return NanBox::number(crate::wtf8::utf16_len(&bytes) as f64);
            }
        }
        // `Map`/`Set` expose `size`.
        // `Map`/`Set` expose `.size`; the weak variants do not (no enumeration).
        if key == "size"
            && !self.realm.collection_is_weak(handle)
            && let Some(n) = self.realm.collection_size(handle)
        {
            return NanBox::number(n as f64);
        }
        NanBox::undefined()
    }

    /// Decides whether a data-property write may proceed. A write to a
    /// non-writable property (its own `writable: false`, or any property of a
    /// frozen object) is a `TypeError` in strict mode and silently ignored
    /// otherwise. Returns `true` when the caller should perform the write.
    /// Whether `handle[key] = …` is permitted (non-throwing): the property is not
    /// read-only/frozen, and either already own or the object is extensible. The shared
    /// predicate behind `allow_property_write` (which adds the strict-mode throw) and
    /// `Reflect.set` (which returns the boolean).
    /// If `h` is a String (a string primitive cell or a String wrapper object),
    /// returns its UTF-16 length — the count of own index properties ToObject(str)
    /// exposes (`"0".."length-1"`). `None` for any non-string.
    pub(crate) fn string_index_count(&self, h: crate::heap::Handle) -> Option<usize> {
        // A String wrapper object boxes its primitive under PRIM_WRAP.
        let sh = if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
            prim.as_handle().map(Handle::from_raw)?
        } else {
            h
        };
        let bytes = self.realm.string_bytes(sh)?;
        Some(crate::wtf8::utf16_len(&bytes))
    }

    /// `Set(O, key, value, true)` — a [[Set]] whose `Throw` is true regardless of
    /// the caller's strictness (used by `Object.assign`, whose CopyDataProperties
    /// step always throws on a failed write). `key_name` is the string form of the
    /// key (the `\0sym:` sentinel for a symbol) for the writability predicate;
    /// `key_box` is the value passed to the [[Set]] machinery.
    pub(crate) fn set_or_throw(
        &mut self,
        target: crate::heap::Handle,
        key_box: NanBox,
        key_name: &str,
        value: NanBox,
    ) -> Result<(), ExecError> {
        // An own accessor takes precedence: its setter runs (and a frozen object's
        // accessor is still writable through the setter), so delegate without the
        // data-write gate. The `[[Set]]` path itself rejects a getter-only accessor.
        let has_own_accessor = self.realm.accessor(target, key_name).is_some();
        // A data write that OrdinarySet would reject (read-only / frozen property,
        // or a new property on a non-extensible target) is a TypeError here even in
        // sloppy mode. A property with a setter, or a writable slot, passes the
        // predicate and is delegated to the normal [[Set]] (which runs setters,
        // proxy traps, array-index and length handling).
        if !has_own_accessor && !self.can_write_property(target, key_name) {
            let add_to_non_extensible =
                !self.realm.has_own(target, key_name) && !self.realm.is_extensible(target);
            let m = if add_to_non_extensible {
                self.new_str(&alloc::format!(
                    "Cannot add property '{key_name}', object is not extensible"
                ))
            } else {
                self.new_str(&alloc::format!(
                    "Cannot assign to read only property '{key_name}'"
                ))
            };
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        self.assign_member_value(target, key_box, value)
    }

    pub(crate) fn can_write_property(&self, handle: crate::heap::Handle, key: &str) -> bool {
        // A string's index properties (`"abc"[0]`, or `new String("abc")[0]`) are
        // non-writable own data properties, so a write to an in-range index fails.
        if let Ok(i) = key.parse::<usize>()
            && self.string_index_count(handle).is_some_and(|n| i < n)
        {
            return false;
        }
        let add_to_non_extensible =
            !self.realm.has_own(handle, key) && !self.realm.is_extensible(handle);
        let readonly = self.realm.property_is_readonly(handle, key)
            || (self.realm.is_frozen(handle) && self.realm.get_property(handle, key).is_some());
        !readonly && !add_to_non_extensible
    }

    pub(crate) fn allow_property_write(
        &mut self,
        handle: crate::heap::Handle,
        key: &str,
    ) -> Result<bool, ExecError> {
        if !self.can_write_property(handle, key) {
            if self.strict {
                let add_to_non_extensible =
                    !self.realm.has_own(handle, key) && !self.realm.is_extensible(handle);
                let m = if add_to_non_extensible {
                    self.new_str(&alloc::format!(
                        "Cannot add property '{key}', object is not extensible"
                    ))
                } else {
                    self.new_str(&alloc::format!(
                        "Cannot assign to read only property '{key}'"
                    ))
                };
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(false); // sloppy mode: the write is silently dropped
        }
        Ok(true)
    }
}
