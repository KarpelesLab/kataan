//! §4.1 — the event loop & timer scheduling (`ROADMAP.md` §4.1).
//!
//! This is the **macrotask / timer** layer of the host runtime, built entirely
//! on the §4.0 embedding API ([`Interp::register_global_fn`], `Ctx`, and the
//! `Ctx::persist` / [`Interp::persistent`] / [`Interp::release_persistent`]
//! handle-scope so a moving GC cannot drop a pinned callback). The engine
//! already owns a Promise **microtask** queue; here we add:
//!
//! * `setTimeout` / `setInterval` / `setImmediate` (returning a numeric id) and
//!   `clearTimeout` / `clearInterval` / `clearImmediate` (cancel by id),
//! * `queueMicrotask` (enqueues onto the engine's existing promise-job queue via
//!   `Promise.resolve().then(cb)` — the same queue promise reactions use),
//! * `process.nextTick` (a separate queue drained *before* microtasks each turn),
//! * `AbortController` / `AbortSignal` (with `AbortSignal.timeout` / `.abort`),
//!
//! and the drain loop [`run_event_loop`].
//!
//! # The loop model (single-threaded)
//!
//! A `TimerStore` lives in a thread-local (the engine is single-threaded, so a
//! thread-local is a natural side table keyed by the running thread's `Interp`).
//! Each scheduled callback and its extra arguments are pinned with
//! `Ctx::persist`, so the store holds only stable `u32` persistent-handle
//! indices — never a bare `NanBox` the GC could relocate. One-shot timers and
//! `nextTick` entries release their handles when they fire; intervals keep theirs
//! until cleared.
//!
//! [`run_event_loop`] advances a **virtual clock** (no real sleeping — callbacks
//! fire in due order): it repeatedly drains the nextTick queue, then the
//! microtask queue, then — if timers remain — pops the earliest-due timer,
//! advances the clock to its due time, fires it, and re-drains, stopping only
//! when nextTick + microtasks + timers are all empty (a clean drain). Every
//! callback is invoked through the engine's own call path, so the engine's
//! `max_call_depth` limit guard applies unchanged.

use crate::nbexec::{ExecError, Interp};

/// Installs the timer / event-loop globals (and `AbortController` /
/// `AbortSignal`, `process.nextTick`) into `interp`. Additive: an existing
/// `process` object is extended, not replaced.
pub fn install(interp: &mut Interp<'_>) {
    #[cfg(feature = "std")]
    imp::install(interp);
    #[cfg(not(feature = "std"))]
    let _ = interp;
}

/// Runs the host event loop to a clean drain: nextTick jobs, then microtasks,
/// then the earliest-due timer (virtual clock), repeating until every queue is
/// empty. Intervals are rescheduled; a runaway (never-cleared) interval is
/// bounded by an internal safety cap.
///
/// # Errors
/// Propagates the first uncaught [`ExecError`] a callback throws (mirroring an
/// uncaught exception aborting the loop).
pub fn run_event_loop(interp: &mut Interp<'_>) -> Result<(), ExecError> {
    #[cfg(feature = "std")]
    {
        imp::run_event_loop(interp)
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = interp;
        Ok(())
    }
}

#[cfg(feature = "std")]
mod imp {
    use super::{ExecError, Interp};
    use crate::NanBox;
    use crate::nbexec::Ctx;
    use crate::parser::Parser;
    use alloc::collections::VecDeque;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    /// A guard against a never-cleared `setInterval` hanging the loop forever.
    /// Far above any realistic test/program's macrotask count; a conformant loop
    /// with a live interval genuinely never terminates, so this is a safety valve
    /// (like a fuel budget), not a semantic limit.
    const SAFETY_CAP: u64 = 50_000_000;

    /// A scheduled timer: `setTimeout` (`period == None`), `setInterval`
    /// (`period == Some(ms)`), or `setImmediate` (a `period == None`, `due == now`
    /// entry). `cb` / `args` are persistent-handle indices (`Ctx::persist`).
    struct TimerEntry {
        id: u64,
        /// Virtual-clock time at which the callback becomes due.
        due: f64,
        /// Insertion order, breaking ties between equal-`due` timers (FIFO).
        seq: u64,
        cb: u32,
        args: Vec<u32>,
        /// `Some(ms)` for an interval (reschedule `due += ms` after firing).
        period: Option<f64>,
    }

    /// A queued `process.nextTick` callback (persistent-handle indices).
    struct Tick {
        cb: u32,
        args: Vec<u32>,
    }

    /// The host timer store — a side table for the running thread's interpreter.
    #[derive(Default)]
    struct TimerStore {
        /// Monotonic timer-id source (ids start at 1; 0 is never handed out).
        next_id: u64,
        /// Monotonic insertion counter for equal-`due` tie-breaking.
        seq: u64,
        /// The virtual clock (advanced to each fired timer's due time).
        clock: f64,
        timers: Vec<TimerEntry>,
        ticks: VecDeque<Tick>,
    }

    std::thread_local! {
        static STORE: RefCell<TimerStore> = RefCell::new(TimerStore::default());
    }

    /// The `AbortController` / `AbortSignal` classes and the `process.nextTick`
    /// wiring, expressed in JS over the host primitives installed below
    /// (`setTimeout`, `__kataan_next_tick`). Runs once at [`install`] time; it
    /// assigns onto `globalThis` (additively for `process`) so the bindings
    /// survive as ordinary global properties.
    const PRELUDE: &str = r#"
(function () {
  var g = globalThis;

  // process.nextTick — additive: extend an existing `process`, never clobber it.
  var proc = g.process;
  if (!proc || typeof proc !== 'object') { proc = {}; g.process = proc; }
  proc.nextTick = g.__kataan_next_tick;

  function makeError(msg, name) {
    var e = new Error(msg);
    e.name = name;
    return e;
  }

  class AbortSignal {
    constructor() {
      this.aborted = false;
      this.reason = undefined;
      this.onabort = null;
      this._listeners = [];
    }
    addEventListener(type, cb) {
      if (type === 'abort' && typeof cb === 'function') this._listeners.push(cb);
    }
    removeEventListener(type, cb) {
      if (type === 'abort') {
        var i = this._listeners.indexOf(cb);
        if (i >= 0) this._listeners.splice(i, 1);
      }
    }
    dispatchEvent(ev) { return true; }
    throwIfAborted() { if (this.aborted) throw this.reason; }
    _signalAbort(reason) {
      if (this.aborted) return;
      this.aborted = true;
      this.reason = (reason !== undefined)
        ? reason
        : makeError('This operation was aborted', 'AbortError');
      var ev = { type: 'abort', target: this, currentTarget: this };
      if (typeof this.onabort === 'function') { try { this.onabort(ev); } catch (e) {} }
      var ls = this._listeners.slice();
      for (var i = 0; i < ls.length; i++) { try { ls[i].call(this, ev); } catch (e) {} }
    }
    static abort(reason) {
      var s = new AbortSignal();
      s._signalAbort((reason !== undefined)
        ? reason
        : makeError('This operation was aborted', 'AbortError'));
      return s;
    }
    static timeout(ms) {
      var s = new AbortSignal();
      g.setTimeout(function () {
        s._signalAbort(makeError('The operation was aborted due to timeout', 'TimeoutError'));
      }, ms);
      return s;
    }
  }

  class AbortController {
    constructor() { this.signal = new AbortSignal(); }
    abort(reason) { this.signal._signalAbort(reason); }
  }

  g.AbortSignal = AbortSignal;
  g.AbortController = AbortController;
  try { delete g.__kataan_next_tick; } catch (e) {}
})();
"#;

    pub(super) fn install(interp: &mut Interp<'_>) {
        // Fresh store per install: persistent-handle indices are only valid for
        // the interpreter that minted them, so never carry them across installs.
        STORE.with(|s| *s.borrow_mut() = TimerStore::default());

        interp.register_global_fn("setTimeout", 1, |cx, _this, args| {
            schedule_timer(cx, args, false)
        });
        interp.register_global_fn("setInterval", 1, |cx, _this, args| {
            schedule_timer(cx, args, true)
        });
        interp.register_global_fn("setImmediate", 1, |cx, _this, args| {
            schedule_immediate(cx, args)
        });
        interp.register_global_fn("clearTimeout", 1, |cx, _this, args| clear_timer(cx, args));
        interp.register_global_fn("clearInterval", 1, |cx, _this, args| clear_timer(cx, args));
        interp.register_global_fn("clearImmediate", 1, |cx, _this, args| clear_timer(cx, args));
        interp.register_global_fn("queueMicrotask", 1, |cx, _this, args| {
            queue_microtask(cx, args)
        });
        interp.register_global_fn("__kataan_next_tick", 1, |cx, _this, args| {
            next_tick(cx, args)
        });

        // Wire process.nextTick + install AbortController/AbortSignal in JS. The
        // parsed program is leaked to `'static` (the interpreter stores `&'a`
        // references into the class/function AST it defines) — a one-time cost,
        // like the engine's own `eval`/`Function` program cache.
        let boxed = alloc::boxed::Box::new(
            Parser::parse_program(PRELUDE).expect("kataan timers prelude must parse"),
        );
        let leaked = alloc::boxed::Box::leak(boxed);
        interp
            .run(leaked)
            .expect("kataan timers prelude must evaluate");
    }

    /// `setTimeout(cb, delay?, ...args)` / `setInterval(cb, delay?, ...args)`.
    /// Returns the numeric timer id.
    fn schedule_timer(
        cx: &mut Ctx<'_, '_>,
        args: &[NanBox],
        is_interval: bool,
    ) -> Result<NanBox, NanBox> {
        let cb = args.first().copied().unwrap_or_else(|| cx.undefined());
        let delay = match args.get(1).copied() {
            Some(v) => cx.to_number(v)?,
            None => 0.0,
        };
        // Clamp to a sane, finite delay. Intervals floor at 1ms so the virtual
        // clock always advances (a 0ms interval would spin at a fixed clock).
        let delay = if delay.is_finite() && delay > 0.0 {
            delay
        } else {
            0.0
        };
        let period = if is_interval {
            Some(delay.max(1.0))
        } else {
            None
        };
        let extra: Vec<u32> = args.iter().skip(2).map(|v| cx.persist(*v)).collect();
        let cb_idx = cx.persist(cb);
        let id = STORE.with(|s| {
            let mut s = s.borrow_mut();
            s.next_id += 1;
            let id = s.next_id;
            let seq = s.seq;
            s.seq += 1;
            let due = s.clock + delay;
            s.timers.push(TimerEntry {
                id,
                due,
                seq,
                cb: cb_idx,
                args: extra,
                period,
            });
            id
        });
        Ok(cx.number(id as f64))
    }

    /// `setImmediate(cb, ...args)` — a zero-delay one-shot macrotask.
    fn schedule_immediate(cx: &mut Ctx<'_, '_>, args: &[NanBox]) -> Result<NanBox, NanBox> {
        let cb = args.first().copied().unwrap_or_else(|| cx.undefined());
        let extra: Vec<u32> = args.iter().skip(1).map(|v| cx.persist(*v)).collect();
        let cb_idx = cx.persist(cb);
        let id = STORE.with(|s| {
            let mut s = s.borrow_mut();
            s.next_id += 1;
            let id = s.next_id;
            let seq = s.seq;
            s.seq += 1;
            let due = s.clock;
            s.timers.push(TimerEntry {
                id,
                due,
                seq,
                cb: cb_idx,
                args: extra,
                period: None,
            });
            id
        });
        Ok(cx.number(id as f64))
    }

    /// `clearTimeout` / `clearInterval` / `clearImmediate` — cancel by id and
    /// release the pinned callback + argument handles.
    fn clear_timer(cx: &mut Ctx<'_, '_>, args: &[NanBox]) -> Result<NanBox, NanBox> {
        let Some(id) = args.first().and_then(|v| v.as_number()) else {
            return Ok(cx.undefined());
        };
        let removed = STORE.with(|s| {
            let mut s = s.borrow_mut();
            s.timers
                .iter()
                .position(|t| (t.id as f64) == id)
                .map(|pos| s.timers.remove(pos))
        });
        if let Some(t) = removed {
            cx.release_persistent(t.cb);
            for a in t.args {
                cx.release_persistent(a);
            }
        }
        Ok(cx.undefined())
    }

    /// `queueMicrotask(cb)` — schedule `cb` onto the engine's existing promise-job
    /// (microtask) queue by adopting `Promise.resolve().then(cb)`.
    fn queue_microtask(cx: &mut Ctx<'_, '_>, args: &[NanBox]) -> Result<NanBox, NanBox> {
        let cb = args.first().copied().unwrap_or_else(|| cx.undefined());
        if !cx.is_callable(cb) {
            return Err(cx.type_error("queueMicrotask requires a callable callback"));
        }
        let undef = cx.undefined();
        let promise = cx.resolved_promise(undef);
        let then = cx.get(promise, "then")?;
        cx.call(then, promise, &[cb])?;
        Ok(cx.undefined())
    }

    /// `process.nextTick(cb, ...args)` — enqueue onto the nextTick queue, drained
    /// before microtasks each turn.
    fn next_tick(cx: &mut Ctx<'_, '_>, args: &[NanBox]) -> Result<NanBox, NanBox> {
        let cb = args.first().copied().unwrap_or_else(|| cx.undefined());
        if !cx.is_callable(cb) {
            return Err(cx.type_error("process.nextTick requires a callable callback"));
        }
        let extra: Vec<u32> = args.iter().skip(1).map(|v| cx.persist(*v)).collect();
        let cb_idx = cx.persist(cb);
        STORE.with(|s| {
            s.borrow_mut().ticks.push_back(Tick {
                cb: cb_idx,
                args: extra,
            });
        });
        Ok(cx.undefined())
    }

    pub(super) fn run_event_loop(interp: &mut Interp<'_>) -> Result<(), ExecError> {
        let mut budget = 0u64;
        loop {
            // 1. nextTick jobs, then microtasks — to a joint fixpoint.
            drain_jobs(interp)?;

            // 2. The earliest-due timer, if any remain.
            let Some(idx) = STORE.with(|s| {
                let s = s.borrow();
                s.timers
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.due.total_cmp(&b.due).then(a.seq.cmp(&b.seq)))
                    .map(|(i, _)| i)
            }) else {
                break;
            };

            // Advance the virtual clock and pull the callback out. An interval is
            // rescheduled *in place* before firing, so a `clearInterval(id)` from
            // within the callback (which removes the entry) wins over the reschedule.
            let (cb, arg_idxs, one_shot) = STORE.with(|s| {
                let mut s = s.borrow_mut();
                let due = s.timers[idx].due;
                if s.clock < due {
                    s.clock = due;
                }
                let cb = s.timers[idx].cb;
                let arg_idxs = s.timers[idx].args.clone();
                match s.timers[idx].period {
                    Some(period) => {
                        let seq = s.seq;
                        s.seq += 1;
                        s.timers[idx].due = due + period;
                        s.timers[idx].seq = seq;
                        (cb, arg_idxs, false)
                    }
                    None => {
                        let entry = s.timers.remove(idx);
                        (entry.cb, entry.args, true)
                    }
                }
            });

            let callee = interp.persistent(cb).unwrap_or_else(NanBox::undefined);
            let call_args: Vec<NanBox> = arg_idxs
                .iter()
                .map(|i| interp.persistent(*i).unwrap_or_else(NanBox::undefined))
                .collect();
            let outcome = interp.call_with_this(callee, NanBox::undefined(), &call_args);
            // Release a one-shot's pinned handles once it has fired (an interval
            // keeps its handles for the next tick; `clearInterval` frees them).
            if one_shot {
                interp.release_persistent(cb);
                for i in &arg_idxs {
                    interp.release_persistent(*i);
                }
            }
            outcome?;

            budget += 1;
            if budget >= SAFETY_CAP {
                break;
            }
        }
        Ok(())
    }

    /// Drains the nextTick queue (highest priority) and then the microtask queue,
    /// repeating until both are empty — a nextTick may enqueue microtasks and a
    /// microtask may enqueue nextTicks, so this runs to a joint fixpoint with
    /// nextTick always taking precedence.
    fn drain_jobs(interp: &mut Interp<'_>) -> Result<(), ExecError> {
        loop {
            // All currently-queued nextTicks (draining ones they enqueue too).
            while let Some(tick) = STORE.with(|s| s.borrow_mut().ticks.pop_front()) {
                let callee = interp.persistent(tick.cb).unwrap_or_else(NanBox::undefined);
                let call_args: Vec<NanBox> = tick
                    .args
                    .iter()
                    .map(|i| interp.persistent(*i).unwrap_or_else(NanBox::undefined))
                    .collect();
                let outcome = interp.call_with_this(callee, NanBox::undefined(), &call_args);
                interp.release_persistent(tick.cb);
                for i in &tick.args {
                    interp.release_persistent(*i);
                }
                outcome?;
            }
            // Then the promise-job (microtask) queue.
            interp.drain_microtasks()?;
            // If a microtask enqueued a nextTick, loop; otherwise we are quiescent.
            if STORE.with(|s| s.borrow().ticks.is_empty()) {
                break;
            }
        }
        Ok(())
    }
}
