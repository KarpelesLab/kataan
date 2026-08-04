//! The tree-walker's **GC safepoint**: where an allocation-triggered collection
//! may run, what it roots, and why that is sound (`ROADMAP.md` §3, the GC).
//!
//! # Why a safepoint at all
//!
//! [`Realm::collect`](crate::realm::Realm::collect) is a non-moving mark-sweep:
//! it frees only what the root set cannot reach, and surviving [`Handle`]s stay
//! valid. So the *only* correctness question is whether the root set is
//! complete at the moment it runs.
//!
//! In this interpreter it usually is not. The tree-walker evaluates
//! sub-expressions into ordinary Rust locals (`let a = self.eval(x)?;` then
//! `self.eval(y)?`, which can run arbitrary user code and allocate), and native
//! builtins build `Vec<NanBox>` argument and result buffers on the Rust heap.
//! None of that is reachable from the interpreter's fields, so collecting at an
//! arbitrary point would free live objects.
//!
//! # What is sound
//!
//! There is one place where the live set *is* enumerable: a **statement
//! boundary in the top-level script body**, reached only through statement
//! executors. At such a point the Rust stack holds nothing but statement frames,
//! and each of those has been audited (see the `gc_root` calls in `stmt.rs`) to
//! publish its live [`NanBox`] locals — a loop's completion value, a `for-of`
//! iterator, a `for-in` key list, a `try`'s pending completion — into
//! [`Interp::gc_shadow`].
//!
//! Everything else is fenced off by [`Interp::gc_ok`], which is `false` by
//! default and only `true` while the top-level statement chain is running. Every
//! entry into a *function*, `eval`, module, generator, or class body clears it
//! for that body's dynamic extent, so a statement boundary nested inside any of
//! them never collects — regardless of what its callers hold.
//!
//! # What this deliberately does not reclaim
//!
//! Garbage produced inside function bodies is not reclaimed *while the function
//! runs*; it is reclaimed at the next top-level safepoint. And
//! [`Interp::gc_world_is_simple`] refuses to collect at all while the program
//! has state this pass does not trace — suspended generators, pending
//! jobs/timers, extra realms, modules, host functions, mapped `arguments`
//! WASM instances, or `$262.agent` workers. Those are conservative
//! bail-outs, not claims that collection would be wrong: each one is a root
//! source that would have to be enumerated first.

use super::{Interp, Job, Timer};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::vec::Vec;

/// A [`gc_root`](Interp::gc_root) mark meaning "nothing was pushed" — returned
/// when the safepoint is fenced off anyway, so the registration is skipped
/// entirely and costs one predictable branch.
const NO_MARK: usize = usize::MAX;

impl Interp<'_> {
    /// Publishes `vals` as GC roots for the dynamic extent of a statement
    /// executor's recursion. Returns a mark to hand to
    /// [`gc_unroot`](Self::gc_unroot); pair the two on **every** exit path
    /// (including `?`) or the shadow stack grows without bound.
    ///
    /// A no-op (and free) while collection is fenced off, which is the case for
    /// all code inside a function body — the hot path.
    pub(crate) fn gc_root(&mut self, vals: &[NanBox]) -> usize {
        if !self.gc_ok {
            return NO_MARK;
        }
        let mark = self.gc_shadow.len();
        self.gc_shadow.extend_from_slice(vals);
        mark
    }

    /// Replaces the values published at `mark` (for a loop local that changes
    /// each iteration, such as the accumulated completion value).
    pub(crate) fn gc_reroot(&mut self, mark: usize, vals: &[NanBox]) {
        if mark == NO_MARK {
            return;
        }
        self.gc_shadow.truncate(mark);
        self.gc_shadow.extend_from_slice(vals);
    }

    /// Whether a collection could fire at the next statement boundary — i.e.
    /// whether building a root publication is worth anything at all. Lets a
    /// statement executor skip copying a whole item list on the hot (fenced) path.
    pub(crate) const fn gc_can_collect(&self) -> bool {
        self.gc_ok
    }

    /// Drops everything published since `mark`.
    pub(crate) fn gc_unroot(&mut self, mark: usize) {
        if mark != NO_MARK {
            self.gc_shadow.truncate(mark);
        }
    }

    /// Runs a collection if allocation pressure warrants one and this point is
    /// safe. Called at every statement boundary; the fenced-off and
    /// under-pressure cases are two loads and a compare.
    pub(crate) fn gc_safepoint(&mut self) {
        if !self.gc_ok || self.realm.gc_pressure() < self.realm.gc_next_threshold() {
            return;
        }
        if !self.gc_world_is_simple() {
            return;
        }
        let mut roots: Vec<Handle> = Vec::new();
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
    }

    /// Whether the interpreter's state is confined to what [`gc_roots`](Self::gc_roots)
    /// enumerates. Each `false` case names a root source this pass does not trace;
    /// refusing to collect is the safe answer, and the memory is reclaimed later
    /// (or not at all) rather than incorrectly.
    fn gc_world_is_simple(&self) -> bool {
        // Suspended coroutine activations hold AST cursors, scopes and operand
        // values that are not reachable from any traced field.
        if !self.gen_frames.iter().all(Option::is_none) || self.gen_sink.is_some() {
            return false;
        }
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

        // --- the audited Rust-local publications of the statement executors ---
        for v in &self.gc_shadow {
            push(out, *v);
        }

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
        for s in &self.class_envs {
            visit_scope(s, out);
        }
        if let Some((_, s)) = &self.pending_super {
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
        for v in [
            self.pending_new_target,
            self.reflect_new_target,
            self.pending_super_fn,
            self.pending_this_init.map(|(v, _)| v),
        ]
        .into_iter()
        .flatten()
        {
            push(out, v);
        }
        out.extend(
            [
                self.regexp_proto,
                self.main_regexp_proto,
                self.regexp_ctor,
                self.current_home_object,
                self.this_cell,
                self.pending_async_start.map(|(_, h)| h),
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

        // --- class tables (parallel to `classes`, keyed by class id) ---
        for m in self
            .class_statics
            .iter()
            .chain(&self.class_static_get)
            .chain(&self.class_static_set)
        {
            for v in m.values() {
                push(out, *v);
            }
        }
        for v in self
            .class_handles
            .iter()
            .copied()
            .chain(self.class_fn_super.iter().copied().flatten())
            .chain(self.class_proto_parent.iter().copied().flatten())
        {
            push(out, v);
        }
        for v in self.private_method_cache.values() {
            push(out, *v);
        }

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
