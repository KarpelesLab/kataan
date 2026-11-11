//! `structuredClone(value, { transfer })` — the HTML structured clone
//! algorithm (StructuredSerializeInternal + StructuredDeserialize, run as one
//! pass) over the engine's own values.
//!
//! What is cloned, and how, follows the HTML spec:
//! - primitives are copied (a Symbol is a `DataCloneError`);
//! - Boolean / Number / BigInt / String wrapper objects, Dates and RegExps
//!   (source and flags; `lastIndex` starts at 0) get a fresh object of the
//!   same kind;
//! - an ArrayBuffer is copied (keeping a resizable buffer's
//!   `maxByteLength`), a SharedArrayBuffer is shared, and a view (typed
//!   array / DataView) is rebuilt over its cloned buffer at the same offset
//!   and length, so two views over one buffer still share a buffer;
//! - Map / Set entries are cloned deeply; an Error keeps its standard name
//!   (`Error` for any other), its own data `message` (as a string) and its
//!   own data `cause`;
//! - an Array keeps its length and holes, and an Array or ordinary object
//!   gets each own enumerable string-keyed property, read with `[[Get]]` (so
//!   getters run) when it is reached, skipping one a getter deleted;
//! - everything else — functions, proxies, and any object with internal
//!   slots the algorithm does not know (WeakMap, Promise, generators,
//!   iterators, `arguments`, Intl objects, …) — is a `DataCloneError`.
//!
//! The result never shares mutable state with the input (SharedArrayBuffers
//! aside); shared and cyclic references are preserved through a memory map.
//! Buffers in `transfer` are moved: the clone takes their bytes and the
//! originals are detached once the whole value has been cloned.

use super::*;
use crate::realm::is_internal_key;

/// Per-call state of one `structuredClone`.
struct CloneState {
    /// Source handle to its clone (cycles and shared references).
    memory: alloc::collections::BTreeMap<u64, NanBox>,
    /// The ArrayBuffers listed in `transfer`, in order.
    transfer: Vec<Handle>,
}

impl Interp {
    /// `structuredClone(value, options)`; `argc` is the call's argument count
    /// (the value is a required argument).
    pub(crate) fn structured_clone_call(
        &mut self,
        argc: usize,
        value: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        if argc == 0 {
            return Err(self.type_error("structuredClone requires a value to clone"));
        }
        let mut st = CloneState {
            memory: alloc::collections::BTreeMap::new(),
            transfer: Vec::new(),
        };
        // `StructuredSerializeOptions { sequence<object> transfer = [] }`.
        if !matches!(options.unpack(), Unpacked::Undefined | Unpacked::Null) {
            if !self.is_object_value(options) {
                return Err(self.type_error("structuredClone options must be an object"));
            }
            let list = self.get_v(options, "transfer")?;
            if !list.is_undefined() {
                // (Only the arguments and the list are held while a custom
                // list is stepped: a generator body it resumes may collect.)
                for t in self.iterate_values_keep(list, &[value, options, list])? {
                    if !self.is_object_value(t) {
                        return Err(self.type_error("a transferable must be an object"));
                    }
                    let h = Handle::from_raw(t.as_handle().unwrap_or_default());
                    if !self.is_plain_array_buffer(h) {
                        return Err(self.data_clone_error("Value not transferable"));
                    }
                    if st.transfer.contains(&h) {
                        return Err(
                            self.data_clone_error("An ArrayBuffer is listed in transfer twice")
                        );
                    }
                    if self.array_buffer_is_detached(h) {
                        return Err(self.data_clone_error("An ArrayBuffer is detached"));
                    }
                    st.transfer.push(h);
                }
            }
        }
        let out = self.sc_value(value, &mut st)?;
        for h in core::mem::take(&mut st.transfer) {
            self.detach_array_buffer(h);
        }
        Ok(out)
    }

    /// A `DataCloneError` (a `DOMException` in browsers; here an Error whose
    /// non-enumerable own `name` is `"DataCloneError"`).
    fn data_clone_error(&mut self, message: &str) -> ExecError {
        let m = self.new_str(message);
        let e = self.make_error(N_ERROR_BASE, Some(m));
        if let Some(h) = e.as_handle().map(Handle::from_raw) {
            let n = self.new_str("DataCloneError");
            self.realm.set_property(h, "name", n);
            self.realm.mark_hidden(h, "name");
        }
        ExecError::Throw(e)
    }

    /// An `ArrayBuffer` object (not a SharedArrayBuffer).
    fn is_plain_array_buffer(&self, h: Handle) -> bool {
        self.realm.proxy_at(h).is_none()
            && self.array_buffer_bytes(h).is_some()
            && self.realm.typed_kind(h).is_none()
            && self
                .realm
                .get_property(h, SHARED_ARRAY_BUFFER_BRAND)
                .is_none()
    }

    fn array_buffer_is_detached(&self, h: Handle) -> bool {
        self.realm.get_property(h, ARRAY_BUFFER_DETACHED).is_some()
    }

    /// `new <intrinsic name>(...args)` — the realm's own constructor, never a
    /// rebound global.
    fn sc_construct(&mut self, name: &str, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let Some(ctor) = self.intrinsic_global(name) else {
            return Err(self.data_clone_error("value could not be cloned"));
        };
        self.construct(ctor, args)
    }

    fn sc_value(&mut self, v: NanBox, st: &mut CloneState) -> Result<NanBox, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(v); // number, boolean, null, undefined
        };
        let h = Handle::from_raw(raw);
        if self.realm.is_string_handle(h) || self.realm.is_bigint(h) {
            return Ok(v);
        }
        if self.realm.symbol_at(h).is_some() {
            return Err(self.data_clone_error("A Symbol could not be cloned"));
        }
        if let Some(c) = st.memory.get(&raw) {
            return Ok(*c);
        }
        if crate::stack::exhausted() {
            let m = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        if self.realm.proxy_at(h).is_some() {
            return Err(self.data_clone_error("A Proxy could not be cloned"));
        }
        if self.is_callable(h) {
            return Err(self.data_clone_error("A function could not be cloned"));
        }
        // Primitive wrappers: a fresh wrapper of the same primitive.
        if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
            let symbol = prim
                .as_handle()
                .is_some_and(|p| self.realm.symbol_at(Handle::from_raw(p)).is_some());
            if symbol {
                return Err(self.data_clone_error("A Symbol object could not be cloned"));
            }
            let c = self.coerce_to_object(prim);
            st.memory.insert(raw, c);
            return Ok(c);
        }
        if let Some(ms) = self.realm.date_at(h) {
            let c = self.sc_construct("Date", &[NanBox::number(ms)])?;
            st.memory.insert(raw, c);
            return Ok(c);
        }
        if let Some(flags) = self.realm.regexp_flags(h).map(String::from) {
            let source = self.realm.regexp_source_bytes(h).unwrap_or_default();
            let s = self.new_str_bytes(source);
            let f = self.new_str(&flags);
            let c = self.sc_construct("RegExp", &[s, f])?;
            st.memory.insert(raw, c);
            return Ok(c);
        }
        if self.realm.typed_kind(h).is_some() {
            return self.sc_typed_array(h, st);
        }
        if let Some(buf) = self.realm.get_property(h, DATA_VIEW_BUF) {
            return self.sc_data_view(h, buf, st);
        }
        if self.array_buffer_bytes(h).is_some() {
            return self.sc_array_buffer(h, st);
        }
        if let Some(is_set) = self.realm.collection_is_set(h) {
            if self.realm.collection_is_weak(h) {
                return Err(self.data_clone_error("A weak collection could not be cloned"));
            }
            let entries = self.realm.collection_entries(h).unwrap_or_default();
            let coll = self.new_collection_object(is_set);
            let c = NanBox::handle(coll.to_raw());
            st.memory.insert(raw, c);
            for (k, val) in entries {
                let ck = self.sc_value(k, st)?;
                let cv = self.sc_value(val, st)?;
                self.realm.collection_set(coll, ck, cv);
            }
            return Ok(c);
        }
        if self.is_error_object(h) {
            return self.sc_error(h, st);
        }
        if self.realm.is_array(h) {
            let len = self.realm.array_length(h).unwrap_or(0);
            let arr = self.realm.new_array(Vec::new());
            if len > 0 {
                self.realm.set_array_length(arr, len);
            }
            let c = NanBox::handle(arr.to_raw());
            st.memory.insert(raw, c);
            self.sc_properties(h, arr, true, st)?;
            return Ok(c);
        }
        // An object with an internal slot the algorithm does not know (a
        // promise, generator, iterator, `arguments`, WeakRef, Intl object, …)
        // is not cloneable; only private fields and symbol keys may be stored
        // as internal keys on an ordinary object.
        let ordinary = self.realm.object_keys(h).is_some()
            && !self.realm.object_all_keys(h).iter().any(|k| {
                is_internal_key(k) && !k.starts_with("\u{0}#") && !k.starts_with("\u{0}sym:")
            });
        if !ordinary {
            return Err(self.data_clone_error("The object could not be cloned"));
        }
        let obj = self.realm.new_object();
        let c = NanBox::handle(obj.to_raw());
        st.memory.insert(raw, c);
        self.sc_properties(h, obj, false, st)?;
        Ok(c)
    }

    /// Copies `src`'s own enumerable string-keyed properties onto `dst`,
    /// cloning each value. The keys are collected first; a key whose property
    /// is gone (or no longer enumerable) when it is reached is skipped.
    fn sc_properties(
        &mut self,
        src: Handle,
        dst: Handle,
        dst_is_array: bool,
        st: &mut CloneState,
    ) -> Result<(), ExecError> {
        let keys: Vec<String> = self
            .own_property_keys_values(src)?
            .into_iter()
            .filter(|k| {
                k.as_handle()
                    .is_some_and(|r| self.realm.is_string_handle(Handle::from_raw(r)))
            })
            .map(|k| self.member_key(k))
            .collect();
        for key in keys {
            let desc = self.descriptor_of(src, &key)?;
            let Some(dh) = desc.as_handle().map(Handle::from_raw) else {
                continue;
            };
            let enumerable = self
                .realm
                .get_property(dh, "enumerable")
                .is_some_and(|e| self.realm.truthy(e));
            if !enumerable {
                continue;
            }
            let v = self.read_member(src, &key)?;
            let cv = self.sc_value(v, st)?;
            if dst_is_array
                && let Ok(i) = key.parse::<usize>()
                && crate::realm::is_canonical_decimal(&key)
                && i < u32::MAX as usize
            {
                self.create_data_property_or_throw(dst, i, cv)?;
            } else {
                self.realm.set_property(dst, &key, cv);
            }
        }
        Ok(())
    }

    fn sc_error(&mut self, h: Handle, st: &mut CloneState) -> Result<NanBox, ExecError> {
        const NAMES: [&str; 7] = [
            "Error",
            "EvalError",
            "RangeError",
            "ReferenceError",
            "SyntaxError",
            "TypeError",
            "URIError",
        ];
        let name_v = self.read_member(h, "name")?;
        let name = name_v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|n| self.realm.string_value(n))
            .and_then(|n| NAMES.iter().find(|x| **x == n).copied())
            .unwrap_or("Error");
        // Own *data* `message` / `cause` only (a getter is not run).
        let own_data = |this: &mut Self, key: &str| -> Result<Option<NanBox>, ExecError> {
            let desc = this.descriptor_of(h, key)?;
            Ok(desc
                .as_handle()
                .map(Handle::from_raw)
                .filter(|d| this.realm.has_own(*d, "value"))
                .and_then(|d| this.realm.get_property(d, "value")))
        };
        let message = match own_data(self, "message")? {
            Some(m) => {
                let s = self.coerce_to_string(m)?;
                self.new_str(&s)
            }
            None => NanBox::undefined(),
        };
        let cause = own_data(self, "cause")?;
        let e = self.sc_construct(name, &[message])?;
        st.memory.insert(h.to_raw(), e);
        // The clone is in memory before its cause is cloned, so a cause chain
        // that cycles back to this error reaches the clone.
        if let Some(cv) = cause
            && let Some(eh) = e.as_handle().map(Handle::from_raw)
        {
            let cc = self.sc_value(cv, st)?;
            self.realm.set_property(eh, "cause", cc);
            self.realm.mark_hidden(eh, "cause");
        }
        Ok(e)
    }

    fn sc_array_buffer(&mut self, h: Handle, st: &mut CloneState) -> Result<NanBox, ExecError> {
        let raw = h.to_raw();
        let bh = self.array_buffer_bytes(h);
        // A SharedArrayBuffer clone shares the original's data block.
        if self
            .realm
            .get_property(h, SHARED_ARRAY_BUFFER_BRAND)
            .is_some()
        {
            let obj = self.realm.new_object();
            if let Some(bh) = bh {
                self.realm.set_hidden_property(
                    obj,
                    ARRAY_BUFFER_BYTES,
                    NanBox::handle(bh.to_raw()),
                );
            }
            self.realm
                .set_hidden_property(obj, SHARED_ARRAY_BUFFER_BRAND, NanBox::boolean(true));
            if let Some(max) = self.realm.get_property(h, ARRAY_BUFFER_MAXLEN) {
                self.realm
                    .set_hidden_property(obj, ARRAY_BUFFER_MAXLEN, max);
            }
            if let Some(proto) = self.intrinsic_proto("SharedArrayBuffer") {
                self.realm.set_object_proto(obj, Some(proto));
            }
            let c = NanBox::handle(obj.to_raw());
            st.memory.insert(raw, c);
            return Ok(c);
        }
        if self.array_buffer_is_detached(h) {
            return Err(self.data_clone_error("An ArrayBuffer is detached and could not be cloned"));
        }
        let bytes = bh
            .and_then(|b| self.realm.bytes_at(b))
            .map(<[u8]>::to_vec)
            .unwrap_or_default();
        let nb = self.make_array_buffer_from_bytes(&bytes);
        if let Some(max) = self.realm.get_property(h, ARRAY_BUFFER_MAXLEN) {
            self.realm.set_hidden_property(nb, ARRAY_BUFFER_MAXLEN, max);
        }
        let c = NanBox::handle(nb.to_raw());
        st.memory.insert(raw, c);
        Ok(c)
    }

    /// The clone of a view's buffer object (detached → `DataCloneError`).
    fn sc_view_buffer(
        &mut self,
        buf: Option<Handle>,
        st: &mut CloneState,
    ) -> Result<NanBox, ExecError> {
        let Some(buf) = buf else {
            return Err(self.data_clone_error("The view could not be cloned"));
        };
        if self.array_buffer_is_detached(buf) {
            return Err(self.data_clone_error("An ArrayBuffer is detached and could not be cloned"));
        }
        self.sc_value(NanBox::handle(buf.to_raw()), st)
    }

    fn sc_typed_array(&mut self, h: Handle, st: &mut CloneState) -> Result<NanBox, ExecError> {
        let kind = self.realm.typed_kind(h).unwrap_or(0) as usize;
        let name = TYPED_ARRAY_KINDS.get(kind).map_or("Uint8Array", |k| k.0);
        let buf = self.sc_view_buffer(self.realm.typed_array_object(h), st)?;
        let off = self.realm.typed_byte_offset(h).unwrap_or(0);
        let mut args = alloc::vec![buf, NanBox::number(off as f64)];
        if !self.realm.is_length_tracking(h) {
            let len = self.realm.typed_len(h).unwrap_or(0);
            args.push(NanBox::number(len as f64));
        }
        let c = self.sc_construct(name, &args)?;
        st.memory.insert(h.to_raw(), c);
        Ok(c)
    }

    fn sc_data_view(
        &mut self,
        h: Handle,
        buf: NanBox,
        st: &mut CloneState,
    ) -> Result<NanBox, ExecError> {
        let cbuf = self.sc_view_buffer(buf.as_handle().map(Handle::from_raw), st)?;
        let off = self
            .realm
            .get_property(h, DATA_VIEW_OFF)
            .unwrap_or(NanBox::number(0.0));
        let mut args = alloc::vec![cbuf, off];
        if let Some(len) = self.realm.get_property(h, DATA_VIEW_LEN) {
            args.push(len);
        }
        let c = self.sc_construct("DataView", &args)?;
        st.memory.insert(h.to_raw(), c);
        Ok(c)
    }
}
