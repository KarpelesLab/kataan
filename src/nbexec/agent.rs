//! The Test262 `$262.agent` cooperative scheduler and `Atomics.waitAsync`.
//!
//! This engine is single-threaded (the heap is `Rc`, not `Send`), so the
//! Test262 agent model — a main agent that spawns worker agents which
//! communicate through a `SharedArrayBuffer` and `Atomics.wait`/`notify` — is
//! approximated *cooperatively*, without OS threads or whole-script suspension:
//!
//! - **`start(src)`** creates a fresh realm (its own intrinsics, via
//!   [`create_realm_env`](Interp::create_realm_env)), installs a worker-side
//!   `$262.agent`, and runs `src` to completion **eagerly** in it. A worker that
//!   registers a `receiveBroadcast` callback instead defers its work until the
//!   main agent broadcasts (see below). This lands the large class of tests where
//!   the worker computes-and-reports without ever *blocking* on the main agent.
//! - **`report(msg)`** pushes `ToString(msg)` onto a shared FIFO queue;
//!   **`getReport()`** pops its front (or `null`), **`getReportAsync()`** wraps
//!   that in a resolved promise.
//! - **`broadcast(sab)` / `safeBroadcast(sab)`** delivers the SharedArrayBuffer to
//!   every worker's registered `receiveBroadcast` callback (invoked in its realm).
//! - **`Atomics.waitAsync`** parks an async waiter keyed by `(buffer, byte index)`;
//!   a matching **`Atomics.notify`** settles it `"ok"`, and a finite timeout is
//!   settled `"timed-out"` by a macrotask. `notify` returns the number woken.
//!
//! **Out of scope (ledgered as failing):** tests requiring *true* interleaving —
//! the main agent *blocks* in `Atomics.wait` while a worker runs and `notify`s,
//! then the main resumes. That needs whole-script suspension the engine lacks; in
//! the cooperative model such a `wait` simply times out.

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

impl<'a> Interp<'a> {
    /// `$262.agent.start(src)` — spawn a worker agent. Creates a fresh realm,
    /// installs the worker-side `$262.agent`, and runs `src` eagerly to completion
    /// in that realm. A throw inside the worker is swallowed (a real agent runs
    /// independently — the main agent only observes it through reports); a
    /// resource error (stack overflow) still propagates.
    pub(crate) fn agent_start(&mut self, src: NanBox) -> Result<NanBox, ExecError> {
        let source = self.coerce_to_string(src)?;
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
    /// worker `receiveBroadcast` callback registered so far (drained in order).
    /// Each callback runs with its own realm's intrinsics + `globalThis` swapped
    /// in; a throw from a worker callback is swallowed.
    pub(crate) fn agent_broadcast(&mut self, sab: NanBox) -> Result<NanBox, ExecError> {
        let cbs = core::mem::take(&mut self.agent.broadcasts);
        for (idx, cb) in cbs {
            if idx < self.created_realms.len() {
                let saved_intrinsics = self.realm.intrinsics_snapshot();
                let saved_gt = self.global_this;
                self.realm
                    .restore_intrinsics(self.created_realms[idx].intrinsics);
                self.global_this = self.created_realms[idx].global_this;
                // A worker callback is an independent agent: its faults are only
                // ever observed by the main agent through (missing) reports.
                let _ = self.call(cb, &[sab]);
                self.realm.restore_intrinsics(saved_intrinsics);
                self.global_this = saved_gt;
            } else {
                let _ = self.call(cb, &[sab]);
            }
        }
        Ok(NanBox::undefined())
    }

    /// `$262.agent.report(msg)` — push `ToString(msg)` onto the shared queue.
    pub(crate) fn agent_report(&mut self, msg: NanBox) -> Result<NanBox, ExecError> {
        let s = self.coerce_to_string(msg)?;
        self.agent.reports.push_back(s);
        Ok(NanBox::undefined())
    }

    /// `$262.agent.getReport()` — pop the front of the shared queue, or `null`.
    pub(crate) fn agent_get_report(&mut self) -> NanBox {
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

    /// `$262.agent.receiveBroadcast(cb)` — register `cb` (tagged with the running
    /// worker's realm) to receive the next `broadcast`.
    pub(crate) fn agent_receive_broadcast(&mut self, cb: NanBox) -> Result<NanBox, ExecError> {
        let idx = self.agent.current_agent_realm.unwrap_or(usize::MAX);
        self.agent.broadcasts.push((idx, cb));
        Ok(NanBox::undefined())
    }

    /// `$262.agent.monotonicNow()` — reads the virtual clock (ms). Advancing when
    /// macrotasks (timeouts) fire, this lets a worker that parks in
    /// `Atomics.waitAsync(…, timeout)` and is released by the timeout observe the
    /// elapsed `monotonicNow()` grow by ~`timeout`, as a real clock would.
    pub(crate) fn agent_monotonic_now(&mut self) -> f64 {
        self.virtual_now
    }

    // --- Atomics.waitAsync support ---

    /// The wait-location key `(backing-buffer handle, absolute byte index)` for
    /// element `idx` of the waitable typed array `ta`.
    fn atomics_wait_key(&self, ta: Handle, idx: usize, kind: u8) -> (u64, usize) {
        let buffer = self.realm.typed_array_object(ta).map_or(0, |h| h.to_raw());
        let size = if crate::nbexec::is_bigint_kind(kind) {
            8
        } else {
            4
        };
        let offset = self.realm.typed_byte_offset(ta).unwrap_or(0);
        (buffer, offset + idx * size)
    }

    /// `Atomics.notify(view, idx, count)` core: wake up to `count` async
    /// `waitAsync` waiters parked on this location (settling each `"ok"`), and
    /// return the number woken. (No *blocking* `wait` can ever be parked in this
    /// single-agent model, so the async waiters are the only wakeable ones.)
    pub(crate) fn atomics_notify(&mut self, ta: Handle, idx: usize, kind: u8, count: f64) -> usize {
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

    /// `Atomics.waitAsync(view, idx, value, timeout)` core (after validation and
    /// the `value`/`timeout` coercions). Returns the spec `{ async, value }`
    /// result object: a value mismatch is `{ false, "not-equal" }`, a non-positive
    /// timeout `{ false, "timed-out" }`, otherwise `{ true, <promise> }` parked
    /// until a matching `notify` (→ `"ok"`) or, for a finite timeout, a macrotask
    /// (→ `"timed-out"`).
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
        let (buffer, byte_index) = self.atomics_wait_key(ta, idx, kind);
        let promise = self.fresh_promise();
        self.agent.waiters.push(AtomicsWaiter {
            buffer,
            byte_index,
            promise,
        });
        // A finite timeout eventually times out: schedule a macrotask (which runs
        // after the microtask queue drains, so a synchronous `notify` wins first).
        if t.is_finite() {
            let cb = self
                .realm
                .new_bound_native(N_ATOMICS_ASYNC_TIMEOUT, promise);
            self.schedule_timer(t, NanBox::handle(cb.to_raw()), Vec::new());
        }
        self.realm
            .set_property(result, "async", NanBox::boolean(true));
        self.realm
            .set_property(result, "value", NanBox::handle(promise.to_raw()));
        NanBox::handle(result.to_raw())
    }

    /// A finite-timeout waiter's macrotask fired: if `promise` is still parked,
    /// remove it and settle `"timed-out"` (a prior `notify` would have removed it,
    /// making this a no-op).
    pub(crate) fn atomics_wait_async_timeout(&mut self, promise: Handle) {
        if let Some(pos) = self.agent.waiters.iter().position(|w| w.promise == promise) {
            self.agent.waiters.remove(pos);
            let v = self.new_str("timed-out");
            self.settle(promise, v, true);
        }
    }
}
