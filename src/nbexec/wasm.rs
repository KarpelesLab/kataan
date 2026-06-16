use super::*;

impl<'a> Interp<'a> {
    /// The tag used by `Object.prototype.toString` (`"[object <tag>]"`): a
    /// `Symbol.toStringTag` string property if present, else the built-in tag.
    /// Invokes a WASM export wrapper: `data` carries the module bytes and the
    /// export name. Decodes the module *once* (cached per instance id in
    /// `wasm_modules` — S1: a repeat call reuses the parsed+validated module rather
    /// than re-`decode_with_limits`-ing the raw bytes), then marshals `args` across
    /// the JS↔WASM boundary and returns the (first) result as a JS value.
    pub(crate) fn call_wasm_export(
        &mut self,
        data: crate::heap::Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let name = self
            .realm
            .get_property(data, WASM_EXPORT)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .ok_or_else(|| self.wasm_compile_error("missing export name"))?;
        let imports_obj = self
            .realm
            .get_property(data, WASM_IMPORTS)
            .unwrap_or(NanBox::undefined());
        let inst_id = self
            .realm
            .get_property(data, WASM_INSTANCE_ID)
            .and_then(|v| v.as_number())
            .map(|n| n as u32);

        // S1: reuse the decoded module across calls. Take the cached `Module` out
        // for the duration of this call (so the import-dispatch closure below can
        // borrow the engine mutably without aliasing it); decode on the first call.
        let mut module = match inst_id.and_then(|id| self.wasm_modules.remove(&id)) {
            Some(m) => m,
            None => {
                let bytes = self
                    .realm
                    .get_property(data, WASM_BYTES)
                    .and_then(|v| self.wasm_bytes(v))
                    .ok_or_else(|| self.wasm_compile_error("missing module bytes"))?;
                crate::wasm_rt::Module::decode_with_limits(&bytes, &self.realm.limits.wasm)
                    .map_err(|_| self.wasm_compile_error("invalid module"))?
            }
        };
        // Run the call against the (cached) module, then put the module back into the
        // cache regardless of outcome so the next call reuses it.
        let result = self.call_wasm_export_with_module(&module, &name, imports_obj, inst_id, args);
        if let Some(id) = inst_id {
            self.wasm_modules.insert(id, core::mem::take(&mut module));
        }
        result
    }

    /// The body of [`call_wasm_export`] over an already-decoded `module` (cached by
    /// the caller). Resolves imports, marshals arguments, syncs the JS-shared linear
    /// memory in/out, runs the export, and persists the instance's mutable state.
    pub(crate) fn call_wasm_export_with_module(
        &mut self,
        module: &crate::wasm_rt::Module,
        name: &str,
        imports_obj: NanBox,
        inst_id: Option<u32>,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        // Resolve each function import to a JS function: importObject[mod][field].
        let import_names: Vec<(String, String)> = module
            .import_names()
            .iter()
            .map(|(m, f)| ((*m).into(), (*f).into()))
            .collect();
        let mut import_fns: Vec<NanBox> = Vec::with_capacity(import_names.len());
        for (m, f) in &import_names {
            let ns = imports_obj
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, m))
                .unwrap_or(NanBox::undefined());
            let func = ns
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, f))
                .unwrap_or(NanBox::undefined());
            // A required function import that is absent or not callable is a LinkError.
            if !func
                .as_handle()
                .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
            {
                let msg = self.new_str(&alloc::format!("import {m}.{f} is not a function"));
                return Err(ExecError::Throw(
                    self.make_error(N_WASM_LINK_ERROR, Some(msg)),
                ));
            }
            import_fns.push(func);
        }
        // The result type of each import (to marshal the JS return back to a Val).
        let import_results: Vec<Vec<crate::wasm_rt::ValType>> = (0..import_names.len())
            .map(|i| {
                module
                    .func_type(i as u32)
                    .map(|t| t.results.clone())
                    .unwrap_or_default()
            })
            .collect();

        // Marshal the export's arguments per its parameter types.
        let export_idx = module
            .export(name)
            .ok_or_else(|| self.wasm_compile_error("no such export"))?;
        let params = module
            .func_type(export_idx)
            .map(|t| t.params.clone())
            .unwrap_or_default();
        if args.len() != params.len() {
            return Err(self.wasm_compile_error("argument count mismatch"));
        }
        let mut val_args: Vec<crate::wasm_rt::Val> = Vec::with_capacity(params.len());
        for (t, v) in params.iter().zip(args) {
            // An `i64` parameter takes a BigInt (its low 64 bits); other types take a
            // Number/Bool via `from_nanbox`.
            let val = if *t == crate::wasm_rt::ValType::I64 {
                match v
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.bigint_at(h))
                    .and_then(|b| b.to_i128())
                {
                    Some(n) => crate::wasm_rt::Val::I64(n as i64),
                    None => {
                        return Err(self.wasm_compile_error("argument not coercible to wasm value"));
                    }
                }
            } else {
                crate::wasm_rt::Val::from_nanbox(*v, *t).ok_or_else(|| {
                    self.wasm_compile_error("argument not coercible to wasm value")
                })?
            };
            val_args.push(val);
        }

        // Resolve imported globals: importObject[mod][field] is a
        // `WebAssembly.Global` (its `.value`) or a plain Number/BigInt, coerced to
        // the imported global's declared type.
        let global_imports: Vec<(String, String, crate::wasm_rt::ValType)> = module
            .global_import_names()
            .iter()
            .map(|(m, f, t)| ((*m).into(), (*f).into(), *t))
            .collect();
        let mut import_global_vals = Vec::with_capacity(global_imports.len());
        for (m, f, ty) in &global_imports {
            let ns = imports_obj
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, m))
                .unwrap_or(NanBox::undefined());
            let entry = ns
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, f))
                .unwrap_or(NanBox::undefined());
            // A `WebAssembly.Global` carries its value in a hidden slot; otherwise
            // the entry is itself the value (a Number/BigInt).
            let raw_val = entry
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, WASM_GLOBAL_VALUE))
                .unwrap_or(entry);
            let v = crate::wasm_rt::Val::from_nanbox(raw_val, *ty)
                .ok_or_else(|| self.wasm_compile_error("imported global not coercible"))?;
            import_global_vals.push(v);
        }

        let mut inst = if !global_imports.is_empty() {
            crate::wasm_rt::Instance::with_host_imports_and_globals(module, import_global_vals)
        } else if import_names.is_empty() {
            crate::wasm_rt::Instance::new(module)
        } else {
            crate::wasm_rt::Instance::with_host_imports(module)
        }
        .map_err(|e| self.wasm_compile_error(e.0))?;

        // Resume this instance's persistent memory/globals from its prior call, so
        // mutable state (a counter global, written linear memory, …) carries over.
        if let Some(id) = inst_id
            && let Some(state) = self.wasm_states.get(&id)
        {
            inst.import_state(state);
        }
        // S2: only sync the JS-shared linear memory when the instance actually
        // exports a `WebAssembly.Memory` (otherwise there is nothing JS can observe
        // and no copy is needed). The canonical `Memory.buffer` byte handle is
        // resolved once here and reused for the post-call copy-out below.
        let mem_bytes_h = inst_id
            .and_then(|id| self.wasm_mem_objs.get(&id).copied())
            .and_then(|mem_obj| {
                self.realm
                    .get_property(mem_obj, WASM_MEM_BUFFER)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
            })
            .and_then(|ab| self.array_buffer_bytes(ab));
        // Copy the canonical, JS-shared linear-memory store INTO the instance
        // before the call, so wasm reads any JS writes made through a view over
        // `Memory.buffer` (A5, #11). The Memory's `Cell::Bytes` is authoritative
        // (it tracks `mem.grow`), so it overrides the resumed `import_state` mem.
        // Skipped entirely when the store is empty (no memory / zero-length).
        if let Some(bytes_h) = mem_bytes_h
            && self.realm.bytes_len(bytes_h).is_some_and(|n| n != 0)
            && let Some(canon) = self.realm.bytes_at(bytes_h).map(<[u8]>::to_vec)
        {
            inst.set_memory(canon);
        }

        // The import dispatcher: marshal Vals → JS, call the JS import, marshal the
        // result back. Borrows `self` (the engine) directly — sound because the
        // instance state (`inst`) is a separate object.
        let mut thrown: Option<ExecError> = None;
        let results = {
            let me: &mut Self = self;
            let mut host = |i: usize, wargs: &[crate::wasm_rt::Val]| {
                let nbargs: Vec<NanBox> = wargs.iter().map(|v| v.to_nanbox()).collect();
                match me.call(import_fns[i], &nbargs) {
                    Ok(r) => import_results[i]
                        .iter()
                        .map(|t| {
                            crate::wasm_rt::Val::from_nanbox(r, *t)
                                .ok_or(crate::wasm_rt::WasmRtError("import result not coercible"))
                        })
                        .collect(),
                    Err(e) => {
                        thrown = Some(e);
                        Err(crate::wasm_rt::WasmRtError("host import threw"))
                    }
                }
            };
            inst.call_export_with_host(name, &val_args, &mut host)
        };
        if let Some(e) = thrown {
            // A JS exception thrown by an import unwinds the whole call; the
            // instance's mutable state is discarded (not persisted), matching the
            // prior behavior, but the cached module is still restored by the caller.
            return Err(e);
        }
        // An error *executing* an export is a runtime trap (`unreachable`, div-by-zero,
        // out-of-bounds, an indirect-call type mismatch, …) → `WebAssembly.RuntimeError`,
        // not a compile error (which is reserved for decode/validation at `Module`).
        let results = results.map_err(|e| self.wasm_runtime_error(e.0))?;
        // T6 (grow-during-call): a wasm `memory.grow` inside the call grows only the
        // instance's `Store.mem`; the canonical JS-shared store still has its old
        // size. Grow the `Cell::Bytes` to match *before* copying out, so writes the
        // export made into the newly-grown pages aren't truncated. Also re-length any
        // views (resize_buffer) so a JS `Uint8Array` over `Memory.buffer` can read
        // the new region, and keep the Memory's page count in sync.
        if let Some(bytes_h) = mem_bytes_h {
            let grown = inst.memory().len();
            let canon = self.realm.bytes_len(bytes_h).unwrap_or(0);
            if grown > canon {
                self.realm.resize_buffer(bytes_h, grown);
                if let Some(id) = inst_id
                    && let Some(mem_obj) = self.wasm_mem_objs.get(&id).copied()
                {
                    let pages = inst.memory_pages();
                    self.realm.set_hidden_property(
                        mem_obj,
                        WASM_MEM_PAGES,
                        NanBox::number(pages as f64),
                    );
                }
            }
        }
        // S2: move the instance's post-call state out (no clone) so we can persist
        // it and reuse its memory buffer for the copy-out below.
        let state = inst.into_state();
        // Copy the instance's post-call linear memory BACK into the canonical,
        // JS-shared store, so a `Uint8Array`/`DataView` over `Memory.buffer`
        // observes the writes the export made (A5, #11). JS and wasm never run
        // concurrently within a call, so this snapshot is observably live. Only the
        // instances that export memory have a canonical store to mirror into.
        if let Some(bytes_h) = mem_bytes_h
            && let Some(dst) = self.realm.bytes_at_mut(bytes_h)
        {
            let n = state.mem.len().min(dst.len());
            dst[..n].copy_from_slice(&state.mem[..n]);
        }
        // Persist the post-call memory/globals so the next call sees them. (For a
        // memory-exporting instance the canonical store is authoritative and is
        // re-read into `set_memory` next call, but persisting here keeps globals,
        // dropped segments, and the no-exported-memory case correct.)
        if let Some(id) = inst_id {
            self.wasm_states.insert(id, state);
        }
        // An `i64` result becomes a BigInt (other types map directly to a Number).
        Ok(match results.first() {
            Some(crate::wasm_rt::Val::I64(n)) => {
                let big = crate::bignum::BigInt::from_i128(i128::from(*n));
                NanBox::handle(self.realm.new_bigint(big).to_raw())
            }
            Some(v) => v.to_nanbox(),
            None => NanBox::undefined(),
        })
    }

    /// A thrown `WebAssembly`-style `TypeError` for compile/instantiate failures.
    pub(crate) fn wasm_compile_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_WASM_COMPILE_ERROR, Some(m)))
    }

    /// A thrown `WebAssembly.RuntimeError` for a trap raised while executing a module.
    pub(crate) fn wasm_runtime_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_WASM_RUNTIME_ERROR, Some(m)))
    }

    pub(crate) fn wasm_type_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m)))
    }

    /// Builds a `WebAssembly.Memory` object over `buf` (an ArrayBuffer): a `.buffer`
    /// accessor + `.grow`, with the page count and (optional) maximum recorded.
    pub(crate) fn make_wasm_memory_object(
        &mut self,
        buf: Handle,
        pages: usize,
        maximum: Option<usize>,
    ) -> Handle {
        let mem = self.realm.new_object();
        self.realm
            .set_hidden_property(mem, WASM_MEM_BUFFER, NanBox::handle(buf.to_raw()));
        self.realm
            .set_hidden_property(mem, WASM_MEM_PAGES, NanBox::number(pages as f64));
        self.realm.set_hidden_property(
            mem,
            WASM_MEM_MAX,
            maximum.map_or(NanBox::undefined(), |m| NanBox::number(m as f64)),
        );
        let getter = self.realm.new_bound_native(N_WASM_MEM_BUFFER_GET, mem);
        self.realm.define_accessor(
            mem,
            "buffer",
            NanBox::handle(getter.to_raw()),
            NanBox::undefined(),
        );
        let grow = self.realm.new_bound_native(N_WASM_MEM_GROW, mem);
        self.realm
            .set_property(mem, "grow", NanBox::handle(grow.to_raw()));
        mem
    }

    /// Builds a `WebAssembly.Global` object wrapping the already-coerced `value`
    /// of type `ty`, with a `.value` accessor (settable only when `mutable`).
    pub(crate) fn make_wasm_global(&mut self, value: NanBox, ty: &str, mutable: bool) -> NanBox {
        let g = self.realm.new_object();
        self.realm.set_hidden_property(g, WASM_GLOBAL_VALUE, value);
        let ty_v = self.new_str(ty);
        self.realm.set_hidden_property(g, WASM_GLOBAL_TYPE, ty_v);
        self.realm
            .set_hidden_property(g, WASM_GLOBAL_MUTABLE, NanBox::boolean(mutable));
        let getter = self.realm.new_bound_native(N_WASM_GLOBAL_GET, g);
        let setter = self.realm.new_bound_native(N_WASM_GLOBAL_SET, g);
        self.realm.define_accessor(
            g,
            "value",
            NanBox::handle(getter.to_raw()),
            NanBox::handle(setter.to_raw()),
        );
        self.realm.mark_hidden(g, "value"); // a prototype accessor in spec: non-enumerable
        // `valueOf()` returns the value (so a Global coerces numerically), reusing
        // the same getter native; it is non-enumerable.
        let value_of = self.realm.new_bound_native(N_WASM_GLOBAL_GET, g);
        self.realm
            .set_property(g, "valueOf", NanBox::handle(value_of.to_raw()));
        self.realm.mark_hidden(g, "valueOf");
        NanBox::handle(g.to_raw())
    }

    /// Coerces a JS value to a WASM `Global`'s value type: `i32` via ToInt32,
    /// `f32`/`f64` via ToNumber, `i64` to a `BigInt`.
    pub(crate) fn wasm_coerce_global(&mut self, ty: &str, v: NanBox) -> NanBox {
        match ty {
            "i32" => {
                // ToInt32 without the std-only `trunc`: truncate toward zero (the
                // `as i64` cast) then take the low 32 bits.
                let n = self.realm.to_number(v);
                let i = if n.is_finite() { n as i64 as i32 } else { 0 };
                NanBox::number(f64::from(i))
            }
            "i64" => {
                if let Some(h) = v.as_handle().map(Handle::from_raw)
                    && self.realm.bigint_at(h).is_some()
                {
                    v
                } else {
                    let n = crate::bignum::BigInt::from_i128(self.realm.to_number(v) as i128);
                    NanBox::handle(self.realm.new_bigint(n).to_raw())
                }
            }
            // f32 (and f64) keep the JS number; f32 precision is not narrowed here.
            _ => NanBox::number(self.realm.to_number(v)),
        }
    }

    /// Decodes/validates `bytes_arr` and builds a `WebAssembly.Module` object that
    /// retains the source bytes (for later instantiation), or throws a `TypeError`.
    pub(crate) fn make_wasm_module(&mut self, bytes_arr: NanBox) -> Result<NanBox, ExecError> {
        let bytes = self
            .wasm_bytes(bytes_arr)
            .ok_or_else(|| self.wasm_compile_error("invalid module source"))?;
        crate::wasm_rt::Module::decode_with_limits(&bytes, &self.realm.limits.wasm)
            .map_err(|_| self.wasm_compile_error("invalid module"))?;
        let module = self.realm.new_object();
        self.realm.set_property(module, WASM_BYTES, bytes_arr);
        self.realm.mark_hidden(module, WASM_BYTES);
        self.realm
            .set_hidden_property(module, WASM_IS_MODULE, NanBox::boolean(true));
        Ok(NanBox::handle(module.to_raw()))
    }

    /// Builds an instance object (`{ exports: {…} }`) from module `bytes_arr` (the
    /// original `BufferSource`, kept for per-call re-decode) and an optional import
    /// object. Each export is a callable wrapper bound to `N_WASM_CALL`. Shared by
    /// `WebAssembly.instantiate` and `new WebAssembly.Instance`.
    pub(crate) fn build_wasm_instance(
        &mut self,
        bytes_arr: NanBox,
        imports_obj: NanBox,
    ) -> Result<NanBox, ExecError> {
        let bytes = self
            .wasm_bytes(bytes_arr)
            .ok_or_else(|| self.wasm_compile_error("invalid module source"))?;
        let module = crate::wasm_rt::Module::decode_with_limits(&bytes, &self.realm.limits.wasm)
            .map_err(|_| self.wasm_compile_error("invalid module"))?;
        // Validate function imports eagerly: a required import that is absent or not
        // callable makes `new WebAssembly.Instance` fail with a LinkError (rather than the
        // failure surfacing only when an export is first called).
        let import_names: Vec<(String, String)> = module
            .import_names()
            .iter()
            .map(|(m, f)| ((*m).into(), (*f).into()))
            .collect();
        for (m, f) in &import_names {
            let func = imports_obj
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, m))
                .and_then(|ns| ns.as_handle())
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, f))
                .unwrap_or(NanBox::undefined());
            if !func
                .as_handle()
                .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
            {
                let msg = self.new_str(&alloc::format!("import {m}.{f} is not a function"));
                return Err(ExecError::Throw(
                    self.make_error(N_WASM_LINK_ERROR, Some(msg)),
                ));
            }
        }
        let names: Vec<String> = module.export_names().iter().map(|s| (*s).into()).collect();
        // A fresh instance id ties every export wrapper to one persistent state
        // entry, so the instance's memory/globals survive across export calls.
        let inst_id = self.wasm_next_id;
        self.wasm_next_id = self.wasm_next_id.wrapping_add(1);
        let exports = self.realm.new_object();
        for name in names {
            let data = self.realm.new_object();
            self.realm.set_property(data, WASM_BYTES, bytes_arr);
            self.realm.set_property(data, WASM_IMPORTS, imports_obj);
            let name_v = self.new_str(&name);
            self.realm.set_property(data, WASM_EXPORT, name_v);
            self.realm
                .set_property(data, WASM_INSTANCE_ID, NanBox::number(f64::from(inst_id)));
            let f = self.realm.new_bound_native(N_WASM_CALL, data);
            self.realm
                .set_property(exports, &name, NanBox::handle(f.to_raw()));
        }
        // Exported globals become `WebAssembly.Global` objects holding the
        // instance's value, read at instantiation. Supported for modules with no
        // imports (so a fresh `Instance::new` reflects the real initial values).
        let global_exports: Vec<(String, u32)> = module
            .global_exports()
            .iter()
            .map(|(n, i)| ((*n).into(), *i))
            .collect();
        if !global_exports.is_empty()
            && module.import_names().is_empty()
            && module.global_import_names().is_empty()
            && let Ok(inst) = crate::wasm_rt::Instance::new(&module)
        {
            for (gname, gidx) in &global_exports {
                if let Some(val) = inst.global_value(*gidx) {
                    let (value, ty) = self.wasm_global_export_value(val);
                    let mutable = module.global_is_mutable(*gidx);
                    let g = self.make_wasm_global(value, ty, mutable);
                    self.realm.set_property(exports, gname, g);
                }
            }
        }
        // Exported memory becomes a `WebAssembly.Memory` whose `ArrayBuffer`'s
        // `Cell::Bytes` is the *canonical, shared* linear-memory store (A5, #11):
        // initialized from the instance's post-instantiation memory (data segments
        // + start), then copied in/out of the instance's `Store.mem` around each
        // export call so JS and wasm observe each other's writes. A wasm module has
        // at most one memory; if several names export it they all alias the same
        // `Memory` object. (Modules with no imports, as for the global-export case.)
        let memory_exports: Vec<String> = module
            .memory_exports()
            .iter()
            .map(|(n, _)| (*n).into())
            .collect();
        if !memory_exports.is_empty()
            && module.import_names().is_empty()
            && module.global_import_names().is_empty()
            && let Ok(inst) = crate::wasm_rt::Instance::new(&module)
        {
            let mem_bytes = inst.memory().to_vec();
            let pages = mem_bytes.len() / WASM_PAGE;
            let buf = self.make_array_buffer_from_bytes(&mem_bytes);
            let mem = self.make_wasm_memory_object(buf, pages, None);
            // Tag the Memory with its instance id so `grow` resizes the canonical
            // store and the call boundary can find it; register it for sharing.
            self.realm.set_hidden_property(
                mem,
                WASM_INSTANCE_ID,
                NanBox::number(f64::from(inst_id)),
            );
            self.wasm_mem_objs.insert(inst_id, mem);
            for mname in &memory_exports {
                self.realm
                    .set_property(exports, mname, NanBox::handle(mem.to_raw()));
            }
        }
        let instance = self.realm.new_object();
        self.realm
            .set_property(instance, "exports", NanBox::handle(exports.to_raw()));
        // A (non-enumerable) marker so `instance instanceof WebAssembly.Instance`
        // matches, like the other `WebAssembly.*` boundary objects.
        self.realm.set_hidden_property(
            instance,
            WASM_INSTANCE_ID,
            NanBox::number(f64::from(inst_id)),
        );
        Ok(NanBox::handle(instance.to_raw()))
    }

    /// Converts a WASM global's `Val` to a `(JS value, type string)` pair for
    /// building a `WebAssembly.Global` (an `i64` becomes a `BigInt`).
    pub(crate) fn wasm_global_export_value(
        &mut self,
        val: crate::wasm_rt::Val,
    ) -> (NanBox, &'static str) {
        match val {
            crate::wasm_rt::Val::I32(_) => (val.to_nanbox(), "i32"),
            crate::wasm_rt::Val::F32(_) => (val.to_nanbox(), "f32"),
            crate::wasm_rt::Val::F64(_) => (val.to_nanbox(), "f64"),
            crate::wasm_rt::Val::I64(n) => {
                let big = crate::bignum::BigInt::from_i128(i128::from(n));
                (NanBox::handle(self.realm.new_bigint(big).to_raw()), "i64")
            }
        }
    }

    /// Extracts a byte vector from a JS `BufferSource`-ish value for the WASM
    /// builtins, in priority order:
    /// (a) an `ArrayBuffer` object → the bytes of its `Cell::Bytes` store;
    /// (b) a typed-array view → the underlying buffer's bytes over the view's span
    ///     `[byte_offset .. byte_offset + length * elem_size]` (module sources are a
    ///     `Uint8Array`, so this reproduces the bytes exactly);
    /// (c) a raw `Cell::Bytes` store directly;
    /// (d) a plain JS array of byte numbers.
    /// Returns `None` if `v` isn't byte-like.
    pub(crate) fn wasm_bytes(&self, v: NanBox) -> Option<Vec<u8>> {
        let h = Handle::from_raw(v.as_handle()?);
        // (a) An ArrayBuffer object → the bytes of its backing store.
        if let Some(bytes_h) = self.array_buffer_bytes(h)
            && let Some(bytes) = self.realm.bytes_at(bytes_h)
        {
            return Some(bytes.to_vec());
        }
        // (b) A typed-array view → the underlying buffer's bytes for the view's span.
        if let Some(kind) = self.realm.typed_kind(h) {
            let buffer = self.realm.typed_buffer(h)?;
            let off = self.realm.typed_byte_offset(h).unwrap_or(0);
            let len = self.realm.typed_len(h).unwrap_or(0);
            let span = len * crate::realm::typed_elem_size(kind);
            let bytes = self.realm.bytes_at(buffer)?;
            return Some(bytes.get(off..off + span).unwrap_or(&[]).to_vec());
        }
        // (c) A raw byte store directly.
        if let Some(bytes) = self.realm.bytes_at(h) {
            return Some(bytes.to_vec());
        }
        // (d) A plain JS array of byte numbers.
        let elems = self.realm.array_elements(h)?;
        Some(
            elems
                .iter()
                .map(|e| e.as_number().unwrap_or(0.0) as u8)
                .collect(),
        )
    }
}
