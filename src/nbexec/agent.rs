//! The Test262 `$262.agent` host hooks and `Atomics.waitAsync`.
//!
//! # The agent model
//!
//! With the `std` feature, `$262.agent.start(src)` spawns a **real OS thread
//! running its own [`Interp`]** — its own heap, realm and intrinsics, matching
//! the spec's agent isolation. The agents share only what the spec says they
//! share, all of it owned by the [`AgentPool`](super::agent_pool::AgentPool):
//! the Shared Data Blocks of broadcast `SharedArrayBuffer`s, the report queue,
//! and the `(block, byte index)` waiter lists. A single execution baton means at
//! most one agent runs JS at a time, so those blocks are never touched
//! concurrently and the interleaving is deterministic; see `agent_pool` for the
//! full argument.
//!
//! That buys the thing a cooperative single-stack model cannot express: a worker
//! can **block inside `Atomics.wait`**, mid-callback, and be resumed later by
//! another agent's `Atomics.notify`.
//!
//! - **`start(src)`** spawns the agent and returns once it has evaluated `src`
//!   and reached its first blocking point (so a `broadcast` issued next is never
//!   lost).
//! - **`broadcast(sab)` / `safeBroadcast(sab)`** hands the buffer's Shared Data
//!   Block to every worker, which re-materializes a `SharedArrayBuffer` object
//!   over the *same* block in its own heap and invokes its `receiveBroadcast`
//!   callback.
//! - **`report(msg)` / `getReport()`** are the pool's shared FIFO.
//! - **`Atomics.wait`** parks the calling agent on the pool's waiter list;
//!   **`Atomics.notify`** wakes up to `count` of them in FIFO order;
//!   **`Atomics.waitAsync`** adds a waiter its owner settles a promise for when
//!   its event loop next runs.
//!
//! Without `std` there are no threads: `$262.agent.start` keeps the older
//! cooperative approximation (the worker runs eagerly to completion in a fresh
//! realm of the *same* `Interp`), which lands every test that does not need a
//! worker to block.

use super::*;

/// The worker-side JS prelude `$262.agent.start` prepends to the worker source,
/// building a `$262.agent` bound to the host natives installed in every realm.
const WORKER_AGENT_PRELUDE: &str = r#"
var $262 = {
  global: this,
  agent: {
    report: function (m) { $262_agent_report(m); },
    receiveBroadcast: function (f) { $262_agent_receiveBroadcast(f); },
    leaving: function () {},
    sleep: function (ms) { $262_agent_sleep(ms); },
    monotonicNow: function () { return $262_agent_monotonicNow(); }
  }
};
"#;

/// Native stack for a worker agent's thread. A worker runs a full `Interp` and
/// the tree-walker recurses per AST node, so it needs real headroom — but the
/// reservation counts against the process's *address space*, and a program may
/// start many agents, so this is deliberately well below what an embedder gives
/// the main thread.
#[cfg(feature = "std")]
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// The body of a worker agent's thread: build an `Interp` of its own, evaluate
/// the worker source in it, then serve broadcasts until the pool shuts down.
///
/// Nothing from the parent agent crosses the thread boundary except the pool
/// `Arc` and the source text — the `Interp`, its heap and every `NanBox` are
/// created and dropped here.
#[cfg(feature = "std")]
fn worker_main(pool: alloc::sync::Arc<super::agent_pool::AgentPool>, id: usize, source: String) {
    pool.acquire();
    // `program` must outlive `interp` (which borrows it); locals drop in reverse
    // declaration order, so declaring it first is what makes this compile.
    let program = match crate::parser::Parser::parse_program(&source) {
        Ok(p) => p,
        Err(_) => {
            pool.finish(id);
            pool.release();
            return;
        }
    };
    let mut interp = Interp::new();
    interp.agent.pool = Some(pool.clone());
    interp.agent.id = id;
    // A throw escaping the worker source is the worker's own business — a real
    // agent runs independently and the main agent observes it only as a missing
    // report.
    let _ = interp.run(&program);
    // Serve broadcasts. `recv_broadcast` releases the baton while idle and
    // returns `None` once the pool is shutting down.
    while let Some(block) = pool.recv_broadcast(id) {
        let _ = interp.agent_receive_block(block);
    }
    pool.finish(id);
}

impl<'a> Interp<'a> {
    /// The shared scheduler, creating it (and taking the baton) on first use.
    /// Only the main agent reaches the creating branch: a worker's `Interp` is
    /// handed the pool before it runs any code.
    #[cfg(feature = "std")]
    fn agent_pool(&mut self) -> alloc::sync::Arc<super::agent_pool::AgentPool> {
        if let Some(p) = &self.agent.pool {
            return p.clone();
        }
        let pool = alloc::sync::Arc::new(super::agent_pool::AgentPool::new());
        // The main agent runs as agent 0 and must hold the baton like any other.
        pool.acquire();
        self.agent.pool = Some(pool.clone());
        self.agent.id = 0;
        pool
    }

    /// A scheduling point: hand the baton to the next agent in line, then take it
    /// back. Called on every loop back-edge and after each macrotask, so an agent
    /// that spins on shared memory (`$262.agent.waitUntil`, or a worker's
    /// `while (Atomics.load(…) === 0);`) still lets the others run. A no-op — one
    /// `Option` test — until a `$262.agent` has actually been started.
    ///
    /// # Errors
    /// Returns [`ExecError::Unsupported`] once the pool is shutting down, which
    /// unwinds the agent's whole stack (a `catch` never sees it) so its thread
    /// can exit even from an unbounded spin loop.
    #[inline]
    /// Observes a host interrupt at a loop back-edge (see [`crate::interrupt`]).
    ///
    /// Returns the non-catchable [`ExecError::Interrupted`], so a `try`/`catch`
    /// around the loop body cannot swallow the deadline. Costs a null test on an
    /// `Option` when no watchdog is installed.
    pub(crate) fn check_interrupt(&self) -> Result<(), ExecError> {
        if self
            .realm
            .interrupt
            .as_ref()
            .is_some_and(crate::interrupt::Interrupt::is_tripped)
        {
            return Err(ExecError::Interrupted);
        }
        Ok(())
    }

    pub(crate) fn agent_tick(&mut self) -> Result<(), ExecError> {
        self.check_interrupt()?;
        #[cfg(feature = "std")]
        if let Some(pool) = &self.agent.pool
            && pool.yield_baton()
        {
            return Err(ExecError::Unsupported("$262.agent shutdown"));
        }
        Ok(())
    }

    /// `$262.agent.start(src)` — spawn a worker agent.
    pub(crate) fn agent_start(&mut self, src: NanBox) -> Result<NanBox, ExecError> {
        let source = self.coerce_to_string(src)?;
        #[cfg(feature = "std")]
        let started = self.agent_start_threaded(&source);
        #[cfg(not(feature = "std"))]
        let started = self.agent_start_cooperative(&source);
        started
    }

    /// `$262.agent.start` with threads: the agent gets its own OS thread and its
    /// own `Interp`, and `start` returns only once it has evaluated `source` and
    /// reached its first blocking point — a host delivers a later `broadcast` to
    /// every agent already `start`ed, so returning before it registered
    /// `receiveBroadcast` would drop that broadcast on the floor.
    #[cfg(feature = "std")]
    fn agent_start_threaded(&mut self, source: &str) -> Result<NanBox, ExecError> {
        let pool = self.agent_pool();
        let id = pool.register();
        let full = alloc::format!("{WORKER_AGENT_PRELUDE}{source}");
        let worker_pool = pool.clone();
        match std::thread::Builder::new()
            .stack_size(WORKER_STACK_BYTES)
            .spawn(move || worker_main(worker_pool, id, full))
        {
            Ok(h) => self.agent.workers.push(h),
            Err(_) => return Err(self.type_error("$262.agent.start: cannot spawn an agent")),
        }
        pool.await_started(id);
        Ok(NanBox::undefined())
    }

    /// `$262.agent.start` without threads: run the worker eagerly to completion
    /// in a fresh realm of this same `Interp`. A worker that only registers
    /// `receiveBroadcast` defers its work to `broadcast`; one that would *block*
    /// in `Atomics.wait` cannot be expressed at all on this path.
    #[cfg(not(feature = "std"))]
    fn agent_start_cooperative(&mut self, source: &str) -> Result<NanBox, ExecError> {
        let idx = self.create_realm_env();
        let full = alloc::format!("{WORKER_AGENT_PRELUDE}{source}");
        let saved = self.agent.current_agent_realm;
        self.agent.current_agent_realm = Some(idx);
        let result = self.eval_source_in_realm(idx, &full);
        self.agent.current_agent_realm = saved;
        match result {
            Ok(_) | Err(ExecError::Throw(_)) => Ok(NanBox::undefined()),
            Err(other) => Err(other),
        }
    }

    /// `$262.agent.broadcast(sab)` / `safeBroadcast(sab)` — deliver `sab` to every
    /// worker agent's registered `receiveBroadcast` callback.
    pub(crate) fn agent_broadcast(&mut self, sab: NanBox) -> Result<NanBox, ExecError> {
        #[cfg(feature = "std")]
        if let Some(pool) = self.agent.pool.clone() {
            // Hand over the buffer's Shared Data Block, not the object: each agent
            // builds its own `SharedArrayBuffer` over the same bytes.
            if let Some(block) = sab
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.shared_block_of_buffer(h))
            {
                pool.broadcast(&block);
            }
            // Give the workers a first slice immediately, so the usual
            // `waitUntil(…)` spin that follows has something to observe.
            self.agent_tick()?;
            return Ok(NanBox::undefined());
        }
        let cbs = core::mem::take(&mut self.agent.broadcasts);
        for (idx, cb) in cbs {
            if idx < self.created_realms.len() {
                let saved_intrinsics = self.realm.intrinsics_snapshot();
                let saved_gt = self.global_this;
                self.realm
                    .restore_intrinsics(self.created_realms[idx].intrinsics);
                self.global_this = self.created_realms[idx].global_this;
                let child_intl = core::mem::take(&mut self.created_realms[idx].intl_protos);
                let saved_intl = self.realm.replace_intl_protos(child_intl);
                // A worker callback is an independent agent: its faults are only
                // ever observed by the main agent through (missing) reports.
                let _ = self.call(cb, &[sab]);
                self.created_realms[idx].intl_protos = self.realm.replace_intl_protos(saved_intl);
                self.realm.restore_intrinsics(saved_intrinsics);
                self.global_this = saved_gt;
            } else {
                let _ = self.call(cb, &[sab]);
            }
        }
        Ok(NanBox::undefined())
    }

    /// The receiving half of a broadcast, in a worker agent: build a
    /// `SharedArrayBuffer` over the broadcast block *in this agent's own heap*
    /// and run every registered `receiveBroadcast` callback with it, then drain
    /// this agent's event loop (an `async` callback parks on `await` and finishes
    /// there).
    #[cfg(feature = "std")]
    pub(crate) fn agent_receive_block(
        &mut self,
        block: alloc::sync::Arc<crate::cell::SharedBlock>,
    ) -> Result<(), ExecError> {
        let buf = self.make_array_buffer_over_shared(block);
        let arg = NanBox::handle(buf.to_raw());
        for (_, cb) in core::mem::take(&mut self.agent.broadcasts) {
            // As above: a worker's throw is not the main agent's to observe.
            match self.call(cb, &[arg]) {
                Ok(_) | Err(ExecError::Throw(_)) => {}
                Err(other) => return Err(other),
            }
        }
        self.run_event_loop()
    }

    /// The Shared Data Block backing the `ArrayBuffer` object `buffer`, if it has
    /// one (i.e. it is a `SharedArrayBuffer`).
    #[cfg(feature = "std")]
    fn shared_block_of_buffer(
        &self,
        buffer: Handle,
    ) -> Option<alloc::sync::Arc<crate::cell::SharedBlock>> {
        // A typed array stands in for its `[[ViewedArrayBuffer]]`: `broadcast` is
        // documented to take the buffer, but hosts accept the view too.
        let buffer = match self.realm.typed_array_object(buffer) {
            Some(b) => b,
            None => buffer,
        };
        let bytes = self
            .realm
            .get_property(buffer, ARRAY_BUFFER_BYTES)?
            .as_handle()
            .map(Handle::from_raw)?;
        self.realm.shared_block_at(bytes)
    }

    /// `$262.agent.report(msg)` — push `ToString(msg)` onto the shared queue.
    pub(crate) fn agent_report(&mut self, msg: NanBox) -> Result<NanBox, ExecError> {
        let s = self.coerce_to_string(msg)?;
        #[cfg(feature = "std")]
        if let Some(pool) = &self.agent.pool {
            pool.push_report(s);
            return Ok(NanBox::undefined());
        }
        self.agent.reports.push_back(s);
        Ok(NanBox::undefined())
    }

    /// `$262.agent.getReport()` — pop the front of the shared queue, or `null`.
    pub(crate) fn agent_get_report(&mut self) -> NanBox {
        #[cfg(feature = "std")]
        if let Some(pool) = self.agent.pool.clone() {
            return match pool.pop_report() {
                Some(s) => self.new_str(&s),
                None => NanBox::null(),
            };
        }
        match self.agent.reports.pop_front() {
            Some(s) => self.new_str(&s),
            None => NanBox::null(),
        }
    }

    /// `$262.agent.getReportAsync()` — the report (string / `null`), in a promise.
    pub(crate) fn agent_get_report_async(&mut self) -> Result<NanBox, ExecError> {
        let v = self.agent_get_report();
        let p = self.fresh_promise();
        self.resolve_with(p, v);
        Ok(NanBox::handle(p.to_raw()))
    }

    /// `$262.agent.receiveBroadcast(cb)` — register `cb` to receive the next
    /// `broadcast`.
    pub(crate) fn agent_receive_broadcast(&mut self, cb: NanBox) -> Result<NanBox, ExecError> {
        let idx = self.agent.current_agent_realm.unwrap_or(usize::MAX);
        self.agent.broadcasts.push((idx, cb));
        Ok(NanBox::undefined())
    }

    /// `$262.agent.sleep(ms)` — yield to the other agents for `ms`.
    ///
    /// With a pool this is a *real* sleep with the baton released, so the agents
    /// it is meant to let run actually run (`$262.agent.tryYield` and the
    /// `getReport` poll loop are built on it). The virtual clock advances by at
    /// least `ms` either way, so `monotonicNow()` sees the sleep.
    pub(crate) fn agent_sleep(&mut self, ms: f64) {
        let ms = if ms.is_finite() && ms > 0.0 { ms } else { 0.0 };
        #[cfg(feature = "std")]
        if let Some(pool) = self.agent.pool.clone() {
            let elapsed = pool.sleep(ms);
            self.virtual_now += elapsed.max(ms);
            return;
        }
        self.virtual_now += ms;
    }

    /// `$262.agent.monotonicNow()` — reads the virtual clock (ms). It advances
    /// when macrotasks (timeouts) fire and, with a pool, by the real time an
    /// agent spends sleeping or parked in `Atomics.wait`/`waitAsync` — so a
    /// worker that measures `monotonicNow()` around a wait observes the block.
    pub(crate) fn agent_monotonic_now(&mut self) -> f64 {
        self.virtual_now
    }

    // --- Atomics.wait / notify / waitAsync ---

    /// The wait-location key `(backing-buffer handle, absolute byte index)` for
    /// element `idx` of the waitable typed array `ta`.
    fn atomics_wait_key(&self, ta: Handle, idx: usize, kind: u8) -> (u64, usize) {
        let buffer = self.realm.typed_array_object(ta).map_or(0, |h| h.to_raw());
        let (_, byte_index) = self.atomics_byte_index(ta, idx, kind);
        (buffer, byte_index)
    }

    /// The absolute byte offset of element `idx` of `ta` within its buffer, plus
    /// the element size — the spec's `indexedPosition = (i × elementSize) + offset`.
    fn atomics_byte_index(&self, ta: Handle, idx: usize, kind: u8) -> (usize, usize) {
        let size = if crate::nbexec::is_bigint_kind(kind) {
            8
        } else {
            4
        };
        let offset = self.realm.typed_byte_offset(ta).unwrap_or(0);
        (size, offset + idx * size)
    }

    /// The pool wait-location key `(Shared Data Block identity, byte index)` —
    /// the spec's `GetWaiterList(block, i)` pair — or `None` when this agent has
    /// no pool or the buffer is not a shared one.
    #[cfg(feature = "std")]
    fn atomics_pool_key(&self, ta: Handle, idx: usize, kind: u8) -> Option<(usize, usize)> {
        let buffer = self.realm.typed_array_object(ta)?;
        let block = self.shared_block_of_buffer(buffer)?;
        let (_, byte_index) = self.atomics_byte_index(ta, idx, kind);
        Some((super::agent_pool::block_id(&block), byte_index))
    }

    /// A blocking `Atomics.wait` whose value comparison already succeeded: park
    /// on the pool's waiter list until a matching `Atomics.notify` (`"ok"`) or
    /// the timeout (`"timed-out"`). Returns `None` when there is no pool or the
    /// buffer is not shared, leaving the caller on the single-agent model.
    #[cfg(feature = "std")]
    pub(crate) fn atomics_wait_blocking(
        &mut self,
        ta: Handle,
        idx: usize,
        kind: u8,
        timeout: f64,
    ) -> Option<&'static str> {
        let pool = self.agent.pool.clone()?;
        let (block, byte_index) = self.atomics_pool_key(ta, idx, kind)?;
        let (outcome, elapsed) = pool.wait(self.agent.id, block, byte_index, timeout);
        // A completed timeout must be visible on the virtual clock even if the
        // condvar returned a hair early.
        self.virtual_now += match outcome {
            super::agent_pool::WaitOutcome::TimedOut if timeout.is_finite() => elapsed.max(timeout),
            _ => elapsed,
        };
        Some(match outcome {
            super::agent_pool::WaitOutcome::Ok => "ok",
            super::agent_pool::WaitOutcome::TimedOut => "timed-out",
        })
    }

    /// `Atomics.notify(view, idx, count)` core: wake up to `count` waiters parked
    /// on this location and return the number woken. With a pool that is the
    /// cross-agent waiter list (blocking `wait` *and* `waitAsync` waiters, in
    /// FIFO order); the same-agent list below is the no-pool path.
    pub(crate) fn atomics_notify(&mut self, ta: Handle, idx: usize, kind: u8, count: f64) -> usize {
        #[cfg(feature = "std")]
        if let Some(pool) = self.agent.pool.clone()
            && let Some((block, byte_index)) = self.atomics_pool_key(ta, idx, kind)
        {
            return pool.notify(block, byte_index, count);
        }
        let (buffer, byte_index) = self.atomics_wait_key(ta, idx, kind);
        let mut to_resolve = Vec::new();
        let mut remaining = count;
        let mut i = 0;
        while i < self.agent.waiters.len() {
            if remaining <= 0.0 {
                break;
            }
            let w = &self.agent.waiters[i];
            if w.buffer == buffer && w.byte_index == byte_index {
                to_resolve.push(self.agent.waiters.remove(i).promise);
                remaining -= 1.0;
            } else {
                i += 1;
            }
        }
        let woken = to_resolve.len();
        for p in to_resolve {
            let ok = self.new_str("ok");
            self.settle(p, ok, true);
        }
        woken
    }

    /// Adds an `Atomics.waitAsync` waiter to the *pool's* waiter list, where
    /// every agent's `Atomics.notify` can see it, keeping only the `waiter id →
    /// promise` mapping here. Returns whether it did — `false` (no pool, or a
    /// non-shared buffer) leaves the caller on the same-agent waiter list.
    #[cfg(feature = "std")]
    fn atomics_park_async_shared(
        &mut self,
        ta: Handle,
        idx: usize,
        kind: u8,
        timeout: f64,
        promise: Handle,
    ) -> bool {
        let Some(pool) = self.agent.pool.clone() else {
            return false;
        };
        let Some((block, byte_index)) = self.atomics_pool_key(ta, idx, kind) else {
            return false;
        };
        let id = pool.park_async(self.agent.id, block, byte_index, timeout);
        self.agent.pool_waiters.push((id, promise));
        true
    }

    /// Without threads there is no cross-agent waiter list to park on.
    #[cfg(not(feature = "std"))]
    fn atomics_park_async_shared(
        &mut self,
        _ta: Handle,
        _idx: usize,
        _kind: u8,
        _timeout: f64,
        _promise: Handle,
    ) -> bool {
        false
    }

    /// `Atomics.waitAsync(view, idx, value, timeout)` core (after validation and
    /// the `value`/`timeout` coercions). Returns the spec `{ async, value }`
    /// result object: a value mismatch is `{ false, "not-equal" }`, a non-positive
    /// timeout `{ false, "timed-out" }`, otherwise `{ true, <promise> }` parked
    /// until a matching `notify` (→ `"ok"`) or the timeout (→ `"timed-out"`).
    pub(crate) fn atomics_wait_async(
        &mut self,
        ta: Handle,
        idx: usize,
        kind: u8,
        equal: bool,
        timeout: f64,
    ) -> NanBox {
        let result = self.realm.new_object();
        let resolved = |slf: &mut Self, s: &str| -> NanBox {
            let v = slf.new_str(s);
            slf.realm
                .set_property(result, "async", NanBox::boolean(false));
            slf.realm.set_property(result, "value", v);
            NanBox::handle(result.to_raw())
        };
        if !equal {
            return resolved(self, "not-equal");
        }
        // ToNumber(timeout) with NaN → +Infinity, negatives clamped to 0.
        let t = if timeout.is_nan() {
            f64::INFINITY
        } else {
            timeout.max(0.0)
        };
        if t <= 0.0 {
            return resolved(self, "timed-out");
        }
        let promise = self.fresh_promise();
        if !self.atomics_park_async_shared(ta, idx, kind, t, promise) {
            let (buffer, byte_index) = self.atomics_wait_key(ta, idx, kind);
            self.agent.waiters.push(AtomicsWaiter {
                buffer,
                byte_index,
                promise,
            });
            // A finite timeout eventually times out: schedule a macrotask (which
            // runs after the microtask queue drains, so a synchronous `notify`
            // wins first).
            if t.is_finite() {
                let cb = self
                    .realm
                    .new_bound_native(N_ATOMICS_ASYNC_TIMEOUT, promise);
                self.schedule_timer(t, NanBox::handle(cb.to_raw()), Vec::new());
            }
        }
        self.realm
            .set_property(result, "async", NanBox::boolean(true));
        self.realm
            .set_property(result, "value", NanBox::handle(promise.to_raw()));
        NanBox::handle(result.to_raw())
    }

    /// A finite-timeout waiter's macrotask fired: if `promise` is still parked,
    /// remove it and settle `"timed-out"` (a prior `notify` would have removed it,
    /// making this a no-op). No-pool path only.
    pub(crate) fn atomics_wait_async_timeout(&mut self, promise: Handle) {
        if let Some(pos) = self.agent.waiters.iter().position(|w| w.promise == promise) {
            self.agent.waiters.remove(pos);
            let v = self.new_str("timed-out");
            self.settle(promise, v, true);
        }
    }

    /// Settles every `Atomics.waitAsync` promise the pool has woken (or retired)
    /// for this agent. Returns whether any settled — the event loop then drains
    /// the microtasks they queued before deciding it has nothing left to do.
    #[cfg(feature = "std")]
    pub(crate) fn agent_settle_async_wakes(&mut self) -> bool {
        let Some(pool) = self.agent.pool.clone() else {
            return false;
        };
        let wakes = pool.take_async_wakes(self.agent.id);
        if wakes.is_empty() {
            return false;
        }
        for w in wakes {
            let Some(pos) = self.agent.pool_waiters.iter().position(|(i, _)| *i == w.id) else {
                continue;
            };
            let (_, promise) = self.agent.pool_waiters.remove(pos);
            let v = self.new_str(if w.ok { "ok" } else { "timed-out" });
            self.settle(promise, v, true);
        }
        true
    }

    /// This agent's event loop is out of work but still has `Atomics.waitAsync`
    /// promises parked: release the baton and block until one is woken or its
    /// timeout expires. Returns whether it is worth looping again.
    #[cfg(feature = "std")]
    pub(crate) fn agent_idle_for_waiters(&mut self) -> bool {
        let Some(pool) = self.agent.pool.clone() else {
            return false;
        };
        if self.agent.pool_waiters.is_empty() {
            return false;
        }
        let (elapsed, progressed) = pool.idle(self.agent.id);
        self.virtual_now += elapsed;
        progressed
    }
}
