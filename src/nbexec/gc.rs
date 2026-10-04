//! The interpreter's half of a garbage collection during a hosted bytecode-VM
//! run (`ROADMAP.md` §2.0): what the interpreter keeps alive, and when its own
//! state is simple enough for a collection to be sound.
//!
//! [`Realm::collect`](crate::realm::Realm::collect) is a non-moving mark-sweep:
//! it frees only what the root set cannot reach, and surviving [`Handle`]s stay
//! valid. So the *only* correctness question is whether the root set is
//! complete at the moment it runs.
//!
//! The VM decides *when*: its safepoint (`vm_safepoint` in `crate::nbvm`)
//! collects only while every Rust frame between the outermost VM activation
//! and the current one has published its registers — a descent through a
//! native, a getter, a `valueOf` or an iterator step does not publish, so the
//! VM declines. It then hands its live registers to
//! [`collect_with_roots`](Interp::collect_with_roots) (through
//! `VmHost::collect_garbage`), which adds everything the interpreter keeps
//! alive ([`gc_roots`](Interp::gc_roots)): the global, module and eval
//! environments, ambient values, intrinsics, registries and job queues.
//!
//! [`gc_world_is_simple`](Interp::gc_world_is_simple) refuses to collect at
//! all while the program has state this pass does not trace — pending
//! jobs/timers, extra realms, modules, host functions, WASM instances, or
//! `$262.agent` workers. Those are conservative bail-outs, not claims that
//! collection would be wrong: each one is a root source that would have to be
//! enumerated first.

use super::{Interp, Job, Timer};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::vec::Vec;

impl Interp {
    /// Runs a collection now (if allocation pressure warrants one and the
    /// interpreter's world is simple), rooting everything the interpreter keeps
    /// alive plus `extra` — the live values of a hosted bytecode-VM run
    /// (`ROADMAP.md` §2.0), which only the VM can enumerate. Returns whether the
    /// interpreter's state allowed it.
    ///
    /// Only sound when no interpreter frame holds unpublished values: a VM
    /// safepoint whose descent is fully published (no delegated host call in
    /// flight).
    pub(crate) fn collect_with_roots(&mut self, extra: &[Handle]) -> bool {
        if !self.gc_world_is_simple() {
            return false;
        }
        let mut roots: Vec<Handle> = extra.to_vec();
        self.gc_roots(&mut roots);
        // `arg_maps` and `fn_realm` are keyed by an object handle and are **weak**:
        // an entry lives only while its arguments object / callable does. Moved out
        // for the cycle so the realm can run them through the ephemeron fixpoint
        // while it holds `&mut self.realm`.
        // (`RefCell` because the expand hook reads the table and the prune hook
        // rewrites it, and both are handed to the collector at once.)
        let arg_maps = core::cell::RefCell::new(core::mem::take(&mut self.arg_maps));
        let fn_realm = core::cell::RefCell::new(core::mem::take(&mut self.fn_realm));
        self.realm.maybe_collect_with(
            &roots,
            &mut |marked, extra| {
                // A *live* mapped-`arguments` object keeps the parameter scope it
                // aliases alive; a dead one keeps nothing (and is pruned below).
                for (key, map) in arg_maps.borrow().iter() {
                    if marked.contains(&Handle::from_raw(*key)) {
                        map.scope.for_each_handle(&mut |h| extra.push(h));
                        extra.extend(map.cells.values().copied());
                    }
                }
            },
            &mut |marked| {
                arg_maps
                    .borrow_mut()
                    .retain(|key, _| marked.contains(&Handle::from_raw(*key)));
                fn_realm
                    .borrow_mut()
                    .retain(|key, _| marked.contains(&Handle::from_raw(*key)));
            },
        );
        self.arg_maps = arg_maps.into_inner();
        self.fn_realm = fn_realm.into_inner();
        true
    }

    /// Whether the interpreter's state is confined to what [`gc_roots`](Self::gc_roots)
    /// enumerates. Each `false` case names a root source this pass does not trace;
    /// refusing to collect is the safe answer, and the memory is reclaimed later
    /// (or not at all) rather than incorrectly.
    fn gc_world_is_simple(&self) -> bool {
        // Pending jobs and timers hold handler/value pairs; extra realms hold a
        // whole second set of globals and intrinsics.
        if !self.microtasks.is_empty()
            || !self.macrotasks.is_empty()
            || !self.created_realms.is_empty()
        {
            return false;
        }
        // Host closures capture Rust state the collector cannot see; WASM
        // instances alias linear memory through a handle-keyed table.
        if !self.host_fns.is_empty() || !self.wasm_mem_objs.is_empty() {
            return false;
        }
        // A built-in prototype method dispatch is in flight (its receiver lives in
        // a Rust local).
        if !self.replaced_dispatch.is_empty() {
            return false;
        }
        // `$262.agent`: reports, broadcasts and waiters are cross-agent state.
        if !self.agent.reports.is_empty()
            || !self.agent.broadcasts.is_empty()
            || !self.agent.waiters.is_empty()
        {
            return false;
        }
        #[cfg(feature = "std")]
        if self.agent.pool.is_some() || !self.agent.pool_waiters.is_empty() {
            return false;
        }
        // Module graphs: namespace objects, live import aliases and the registry
        // itself all hold scopes outside the ordinary chain.
        #[cfg(all(feature = "module", feature = "std"))]
        if !self.module_imports.is_empty()
            || !self.module_namespaces.is_empty()
            || !self.deferred_namespaces.is_empty()
            || self.import_meta.is_some()
            || self.active_module_key.is_some()
            || !self.modules.is_empty()
        {
            return false;
        }
        true
    }

    /// Every [`Handle`] the interpreter itself keeps alive, for the safepoint's
    /// root set. Over-approximates freely — a spurious root only delays
    /// reclamation, a missing one frees a live object.
    ///
    /// Only sound in combination with [`gc_world_is_simple`](Self::gc_world_is_simple),
    /// which rules out the state deliberately not enumerated here.
    fn gc_roots(&self, out: &mut Vec<Handle>) {
        let push = |out: &mut Vec<Handle>, v: NanBox| {
            if let Some(raw) = v.as_handle() {
                out.push(Handle::from_raw(raw));
            }
        };

        // --- scope chains (each walks to its root, so enclosing frames are covered) ---
        let visit_scope = |s: &crate::env::Scope, out: &mut Vec<Handle>| {
            s.for_each_handle(&mut |h| out.push(h));
        };
        visit_scope(&self.current, out);
        visit_scope(&self.var_scope, out);
        visit_scope(&self.global_scope, out);
        visit_scope(&self.main_global_scope, out);
        if let Some(s) = &self.eval_var_scope {
            visit_scope(s, out);
        }

        // --- ambient values ---
        for v in [
            self.this_val,
            self.new_target,
            self.global_this,
            self.main_global_this,
        ] {
            push(out, v);
        }
        if let Some(v) = self.reflect_new_target {
            push(out, v);
        }
        // A VM direct eval's context (see `vm_env`).
        for v in self
            .vm_eval_home
            .into_iter()
            .chain(self.vm_eval_privates.iter().map(|(_, k)| *k))
            .chain(self.vm_eval_derived.into_iter().flat_map(|(a, b)| [a, b]))
        {
            push(out, v);
        }
        out.extend(
            [
                self.regexp_proto,
                self.main_regexp_proto,
                self.regexp_ctor,
                self.current_home_object,
            ]
            .into_iter()
            .flatten(),
        );
        // The main realm's intrinsic slots. While a cross-realm call is running,
        // `Realm`'s own slots hold the *callee* realm's, so this snapshot is the
        // only field naming the main realm's — trace it rather than rely on them
        // also being reachable from `main_global_scope`. (Today `gc_world_is_simple`
        // refuses to collect at all once a second realm exists, so this is
        // belt-and-braces for when it stops doing so.)
        let m = &self.main_intrinsics;
        out.extend(
            [
                m.default_object_proto,
                m.array_proto,
                m.promise_proto,
                m.function_proto,
                m.symbol_proto,
                m.bigint_proto,
                m.typed_array,
                m.throw_type_error,
            ]
            .into_iter()
            .flatten(),
        );
        out.extend(self.builtin_iter_protos.values().copied());
        // `%AbstractModuleSource%` + its prototype: memoized on the interpreter,
        // so they must stay live even after the program drops every reference the
        // host hook handed out.
        if let Some((ctor, proto)) = self.module_source_intrinsic {
            out.push(ctor);
            out.push(proto);
        }
        out.extend(self.temporal_protos.iter().copied().flatten());
        out.extend(self.wasm_mem_objs.values().copied());

        // --- interned/registry values ---
        for v in self
            .symbol_registry
            .values()
            .chain(self.well_known_symbols.values())
            .chain(self.tagged_template_cache.values())
        {
            push(out, *v);
        }

        // --- job queues (empty at a real safepoint; rooted anyway) ---
        for j in &self.microtasks {
            gc_root_job(j, out);
        }
        for t in &self.macrotasks {
            gc_root_timer(t, out);
        }
    }
}

/// Roots one pending promise-reaction job.
fn gc_root_job(j: &Job, out: &mut Vec<Handle>) {
    for v in [j.handler, j.value] {
        if let Some(raw) = v.as_handle() {
            out.push(Handle::from_raw(raw));
        }
    }
    out.push(j.result);
    if let Some((a, b)) = j.thenable {
        for v in [a, b] {
            if let Some(raw) = v.as_handle() {
                out.push(Handle::from_raw(raw));
            }
        }
    }
}

/// Roots one pending `setTimeout` macrotask.
fn gc_root_timer(t: &Timer, out: &mut Vec<Handle>) {
    for v in core::iter::once(t.callback).chain(t.args.iter().copied()) {
        if let Some(raw) = v.as_handle() {
            out.push(Handle::from_raw(raw));
        }
    }
}
