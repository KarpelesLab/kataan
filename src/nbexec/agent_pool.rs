//! The cross-agent scheduler behind the Test262 `$262.agent` model: a set of
//! genuinely-parallel-capable agents that nonetheless execute one at a time.
//!
//! # Why a pool at all
//!
//! `Atomics.wait` must *block* the calling agent until another agent's
//! `Atomics.notify` wakes it. Blocking means the waiting agent's whole call stack
//! is suspended mid-native-call, which the tree-walker cannot express (its
//! generator step-machine only reifies the `yield`-bearing spine of a generator
//! body). The only construct that can hold an arbitrary suspended Rust stack is a
//! real OS thread, so **every agent is an OS thread with its own
//! [`Interp`](crate::nbexec::Interp)** — its own heap, realm and intrinsics, which
//! is also what the spec's agent isolation actually says.
//!
//! # What is shared, and how it stays sound
//!
//! Agents share exactly three things, all of them owned by [`AgentPool`] (an
//! `Arc` every agent holds):
//!
//! * the **Shared Data Blocks** of `SharedArrayBuffer`s
//!   ([`SharedBlock`](crate::cell::SharedBlock)) — carried by
//!   `$262.agent.broadcast`;
//! * the **report queue** (`$262.agent.report` / `getReport`);
//! * the **waiter list**, keyed by `(block, byte index)` exactly like the spec's
//!   `GetWaiterList(block, i)`, in FIFO order.
//!
//! Nothing else crosses a thread: no `Rc`, no handle, no `NanBox`.
//!
//! Soundness of the shared blocks rests on the **baton**: a single FIFO token
//! that an agent must hold to run JS at all. It is handed over only at explicit
//! points — a loop back-edge ([`Interp::agent_tick`](crate::nbexec::Interp)),
//! `$262.agent.sleep`, a blocking `Atomics.wait`, an idle event loop, and
//! `$262.agent.start`/`broadcast` — and each handoff is a mutex release/acquire
//! pair, so accesses to a shared block are both mutually exclusive and ordered.
//! The result is *deterministic* cooperative interleaving rather than the
//! timing-dependent preemption a real host has, which is what makes these tests
//! reproducible.
//!
//! # Deadlock
//!
//! A test that parks every agent in `Atomics.wait` with nobody left to notify
//! would hang the process. The pool detects exactly that state — no agent
//! runnable and no waiter already marked woken — and releases the parked waiters
//! with `"timed-out"`, which is the outcome an infinitely-patient host would
//! reach anyway and cannot manufacture a spurious `"ok"`.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// A `SharedArrayBuffer`'s data block as it travels between agents.
pub(crate) type Block = Arc<crate::cell::SharedBlock>;

/// A JS millisecond count as a [`Duration`], clamped to a day.
///
/// `Duration::from_secs_f64` *panics* past `u64::MAX` seconds, and a JS timeout
/// is an arbitrary Number (`Atomics.wait(ta, 0, 0, 1e300)`), so the conversion
/// has to saturate. A day is indistinguishable from "forever" for any caller —
/// the alternative is an infinite timeout, which this already models as `None`.
fn ms_to_duration(ms: f64) -> Duration {
    const MAX_SECS: f64 = 86_400.0;
    Duration::from_secs_f64(if ms.is_nan() {
        0.0
    } else {
        (ms / 1000.0).clamp(0.0, MAX_SECS)
    })
}

/// The identity of a Shared Data Block — the spec's `block` half of the
/// `(block, i)` waiter-list key. Address identity is stable: an `Arc`'s payload
/// never moves, and the pool holds a clone for every live block it has seen.
#[must_use]
pub(crate) fn block_id(block: &Block) -> usize {
    Arc::as_ptr(block) as usize
}

/// What an agent is doing when it is not the one running.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    /// Executing, queued for the baton, or sleeping — it will make progress on
    /// its own.
    Running,
    /// Parked in a blocking `Atomics.wait`.
    WaitingAtomics,
    /// Its event loop has nothing to run but a pending `Atomics.waitAsync`.
    Idle,
    /// A worker waiting for a `$262.agent.broadcast` that has not arrived.
    WaitingBroadcast,
    /// Ran to completion.
    Finished,
}

/// One entry of a waiter list (`(block, byte index)` → FIFO of waiters).
struct Waiter {
    id: u64,
    agent: usize,
    block: usize,
    byte_index: usize,
    /// Set by `Atomics.notify`; the waiter then resolves `"ok"`.
    woken: bool,
    /// Set when the deadlock breaker or a deadline retires the waiter.
    timed_out: bool,
    /// `Atomics.waitAsync`: the owning agent settles a promise for it rather than
    /// being parked on it.
    is_async: bool,
    /// An async waiter's real deadline (`None` for an infinite timeout). A
    /// blocking waiter's deadline is held by the parked thread instead.
    deadline: Option<Instant>,
}

/// The outcome of a blocking wait.
pub(crate) enum WaitOutcome {
    /// Woken by a matching `Atomics.notify`.
    Ok,
    /// The timeout expired (or the pool broke a deadlock / shut down).
    TimedOut,
}

/// The wake state of an `Atomics.waitAsync` waiter, as its owner collects it.
pub(crate) struct AsyncWake {
    /// The waiter id returned by [`AgentPool::park_async`].
    pub id: u64,
    /// `true` for a matching `notify`, `false` for a timeout.
    pub ok: bool,
}

struct PoolState {
    /// Whether some agent currently holds the baton.
    baton_held: bool,
    /// FIFO of tickets queued for the baton (fair handoff, so a yielding agent
    /// cannot starve the others by immediately re-acquiring).
    queue: VecDeque<u64>,
    next_ticket: u64,
    /// Per-agent status, indexed by agent id (0 is the main agent).
    agents: Vec<Status>,
    /// Per-agent queue of broadcast blocks not yet picked up.
    inbox: Vec<VecDeque<Block>>,
    /// The shared FIFO report queue (`report` pushes, `getReport` pops).
    reports: VecDeque<String>,
    /// Every waiter list, flattened; order within a `(block, byte_index)` group
    /// is the FIFO order `AddWaiter` requires.
    waiters: Vec<Waiter>,
    next_waiter_id: u64,
    /// Set at teardown: every parked agent unblocks and its thread exits.
    shutdown: bool,
}

impl PoolState {
    /// The pool is wedged when nobody can make progress on its own and no waiter
    /// has already been marked woken (that wake is progress waiting to happen).
    fn is_deadlocked(&self) -> bool {
        if self.waiters.iter().any(|w| w.woken || w.timed_out) {
            return false;
        }
        // `Running` is the only status from which an agent proceeds on its own
        // (it is executing, queued for the baton, or sleeping); every other one
        // needs some *other* agent to act first.
        !self.agents.contains(&Status::Running)
    }

    /// Retires every parked waiter with `"timed-out"` (deadlock breaker).
    fn break_deadlock(&mut self) {
        for w in &mut self.waiters {
            w.timed_out = true;
        }
    }
}

/// The shared scheduler. Every agent holds an `Arc` of one.
pub(crate) struct AgentPool {
    state: Mutex<PoolState>,
    cv: Condvar,
}

impl AgentPool {
    /// A pool with the main agent (id 0) registered and *not yet* holding the
    /// baton — the caller takes it with [`AgentPool::acquire`].
    pub(crate) fn new() -> Self {
        AgentPool {
            state: Mutex::new(PoolState {
                baton_held: false,
                queue: VecDeque::new(),
                next_ticket: 0,
                agents: alloc::vec![Status::Running],
                inbox: alloc::vec![VecDeque::new()],
                reports: VecDeque::new(),
                waiters: Vec::new(),
                next_waiter_id: 1,
                shutdown: false,
            }),
            cv: Condvar::new(),
        }
    }

    /// Registers a new worker agent, returning its id.
    pub(crate) fn register(&self) -> usize {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.agents.push(Status::Running);
        s.inbox.push(VecDeque::new());
        s.agents.len() - 1
    }

    // --- the baton ---

    /// Takes the baton, blocking (FIFO) until it is free.
    pub(crate) fn acquire(&self) {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        drop(self.acquire_locked(s));
    }

    /// The body of [`AgentPool::acquire`], for callers already holding the lock.
    fn acquire_locked<'g>(
        &'g self,
        mut s: std::sync::MutexGuard<'g, PoolState>,
    ) -> std::sync::MutexGuard<'g, PoolState> {
        let ticket = s.next_ticket;
        s.next_ticket += 1;
        s.queue.push_back(ticket);
        while s.baton_held || s.queue.front() != Some(&ticket) {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.queue.pop_front();
        s.baton_held = true;
        s
    }

    /// Releases the baton.
    pub(crate) fn release(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.baton_held = false;
        self.cv.notify_all();
    }

    /// Hands the baton to the next agent in line and takes it back afterwards —
    /// the scheduling point a loop back-edge and `$262.agent.broadcast` use.
    /// Returns `true` once the pool is shutting down, which tells the caller to
    /// unwind (its thread is expected to exit).
    pub(crate) fn yield_baton(&self) -> bool {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.shutdown {
            return true;
        }
        // Nobody else is queued: keep running rather than pay a futex round-trip.
        if s.queue.is_empty() {
            return false;
        }
        s.baton_held = false;
        self.cv.notify_all();
        let s = self.acquire_locked(s);
        s.shutdown
    }

    /// Blocks (baton released) until the freshly-`register`ed agent `id` has run
    /// its source far enough to stop being the runner — it registered
    /// `receiveBroadcast` and is waiting, parked in a wait, or finished. This is
    /// what makes `$262.agent.start` followed by `broadcast` deterministic.
    pub(crate) fn await_started(&self, id: usize) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.baton_held = false;
        self.cv.notify_all();
        while s.agents[id] == Status::Running && !s.shutdown {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        drop(self.acquire_locked(s));
    }

    /// `$262.agent.sleep(ms)` — release the baton for `ms` of real time so other
    /// agents run, then take it back. Returns the elapsed milliseconds.
    pub(crate) fn sleep(&self, ms: f64) -> f64 {
        let start = Instant::now();
        self.release();
        if ms > 0.0 && ms.is_finite() {
            std::thread::sleep(ms_to_duration(ms));
        } else {
            std::thread::yield_now();
        }
        self.acquire();
        start.elapsed().as_secs_f64() * 1000.0
    }

    // --- reports ---

    /// `$262.agent.report(msg)`.
    pub(crate) fn push_report(&self, msg: String) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.reports.push_back(msg);
    }

    /// `$262.agent.getReport()` — the front of the queue, or `None`.
    pub(crate) fn pop_report(&self) -> Option<String> {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.reports.pop_front()
    }

    // --- broadcast ---

    /// `$262.agent.broadcast(sab)` — deliver `block` to every *worker* agent.
    pub(crate) fn broadcast(&self, block: &Block) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for (id, inbox) in s.inbox.iter_mut().enumerate() {
            if id != 0 {
                inbox.push_back(block.clone());
            }
        }
        self.cv.notify_all();
    }

    /// A worker waits for its next broadcast, releasing the baton meanwhile.
    /// `None` means the pool is shutting down and the worker should exit.
    pub(crate) fn recv_broadcast(&self, agent: usize) -> Option<Block> {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Shutdown wins over a queued broadcast: a teardown must not start new
        // work in an agent whose thread is being wound down.
        if s.shutdown {
            s.agents[agent] = Status::Finished;
            s.baton_held = false;
            self.cv.notify_all();
            return None;
        }
        if let Some(b) = s.inbox[agent].pop_front() {
            return Some(b);
        }
        s.agents[agent] = Status::WaitingBroadcast;
        s.baton_held = false;
        self.cv.notify_all();
        let block = loop {
            if let Some(b) = s.inbox[agent].pop_front() {
                break Some(b);
            }
            if s.shutdown {
                break None;
            }
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        };
        s.agents[agent] = if block.is_some() {
            Status::Running
        } else {
            Status::Finished
        };
        if block.is_some() {
            drop(self.acquire_locked(s));
        }
        block
    }

    /// Marks an agent finished (its thread is about to exit).
    pub(crate) fn finish(&self, agent: usize) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.agents[agent] = Status::Finished;
        self.cv.notify_all();
    }

    // --- the waiter lists ---

    /// A blocking `Atomics.wait`: park on `(block, byte_index)` until a matching
    /// `Atomics.notify`, the deadline, or a detected deadlock. Must be called
    /// with the baton held; the baton is released while parked and re-taken
    /// before returning. Also returns the real milliseconds spent parked, which
    /// the caller adds to its virtual clock so `$262.agent.monotonicNow()`
    /// measures the block.
    pub(crate) fn wait(
        &self,
        agent: usize,
        block: usize,
        byte_index: usize,
        timeout_ms: f64,
    ) -> (WaitOutcome, f64) {
        let start = Instant::now();
        let deadline = if timeout_ms.is_finite() {
            Some(start + ms_to_duration(timeout_ms))
        } else {
            None
        };
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let id = s.next_waiter_id;
        s.next_waiter_id += 1;
        s.waiters.push(Waiter {
            id,
            agent,
            block,
            byte_index,
            woken: false,
            timed_out: false,
            is_async: false,
            deadline,
        });
        s.agents[agent] = Status::WaitingAtomics;
        s.baton_held = false;
        self.cv.notify_all();
        let outcome = loop {
            let me = s.waiters.iter().position(|w| w.id == id);
            match me.map(|i| (s.waiters[i].woken, s.waiters[i].timed_out)) {
                Some((true, _)) => break WaitOutcome::Ok,
                Some((_, true)) | None => break WaitOutcome::TimedOut,
                _ => {}
            }
            if s.shutdown {
                break WaitOutcome::TimedOut;
            }
            if s.is_deadlocked() {
                s.break_deadlock();
                self.cv.notify_all();
                continue;
            }
            s = match deadline {
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        break WaitOutcome::TimedOut;
                    }
                    self.cv
                        .wait_timeout(s, d - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0
                }
                // An unbounded wait still re-checks periodically: the deadlock
                // predicate can only be evaluated by an agent that is awake.
                None => {
                    self.cv
                        .wait_timeout(s, Duration::from_millis(20))
                        .unwrap_or_else(|e| e.into_inner())
                        .0
                }
            };
        };
        s.waiters.retain(|w| w.id != id);
        s.agents[agent] = Status::Running;
        drop(self.acquire_locked(s));
        (outcome, start.elapsed().as_secs_f64() * 1000.0)
    }

    /// `Atomics.waitAsync`: add a waiter the owner does *not* park on. Returns
    /// its id, which the owner maps back to the promise to settle. Must be called
    /// with the baton held.
    pub(crate) fn park_async(
        &self,
        agent: usize,
        block: usize,
        byte_index: usize,
        timeout_ms: f64,
    ) -> u64 {
        let deadline = if timeout_ms.is_finite() {
            Some(Instant::now() + ms_to_duration(timeout_ms))
        } else {
            None
        };
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let id = s.next_waiter_id;
        s.next_waiter_id += 1;
        s.waiters.push(Waiter {
            id,
            agent,
            block,
            byte_index,
            woken: false,
            timed_out: false,
            is_async: true,
            deadline,
        });
        id
    }

    /// `Atomics.notify(view, i, count)` — wake up to `count` waiters on
    /// `(block, byte_index)` in FIFO order, returning how many were woken.
    /// Must be called with the baton held.
    pub(crate) fn notify(&self, block: usize, byte_index: usize, count: f64) -> usize {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut woken = 0usize;
        for w in &mut s.waiters {
            if woken as f64 >= count {
                break;
            }
            if !w.woken && !w.timed_out && w.block == block && w.byte_index == byte_index {
                w.woken = true;
                woken += 1;
            }
        }
        if woken > 0 {
            self.cv.notify_all();
        }
        woken
    }

    /// Collects this agent's `waitAsync` waiters that a `notify` has woken (or
    /// that the deadlock breaker retired), removing them from the waiter list.
    pub(crate) fn take_async_wakes(&self, agent: usize) -> Vec<AsyncWake> {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        s.waiters.retain(|w| {
            if w.is_async && w.agent == agent && (w.woken || w.timed_out) {
                out.push(AsyncWake {
                    id: w.id,
                    ok: w.woken,
                });
                false
            } else {
                true
            }
        });
        out
    }

    /// The event loop has nothing to run but this agent has parked `waitAsync`
    /// waiters: release the baton and block until one of them is woken, its
    /// `deadline` passes, or the pool wedges. Returns the real milliseconds spent
    /// idle and whether a waiter is now resolvable (`false` means the pool is
    /// shutting down and the event loop should stop).
    pub(crate) fn idle(&self, agent: usize) -> (f64, bool) {
        let start = Instant::now();
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.agents[agent] = Status::Idle;
        s.baton_held = false;
        self.cv.notify_all();
        let progressed = loop {
            // Retire any of this agent's async waiters whose timeout has expired.
            let now = Instant::now();
            for w in &mut s.waiters {
                if w.agent == agent && w.is_async && w.deadline.is_some_and(|d| now >= d) {
                    w.timed_out = true;
                }
            }
            if s.waiters
                .iter()
                .any(|w| w.agent == agent && (w.woken || w.timed_out))
            {
                break true;
            }
            if s.shutdown {
                break false;
            }
            if s.is_deadlocked() {
                s.break_deadlock();
                self.cv.notify_all();
                continue;
            }
            // Wake up regularly regardless: the deadlock predicate can only be
            // evaluated by an agent that is awake, and a deadline may expire.
            let slice = s
                .waiters
                .iter()
                .filter(|w| w.agent == agent && w.is_async)
                .filter_map(|w| w.deadline)
                .min()
                .map_or(Duration::from_millis(20), |d| {
                    d.saturating_duration_since(now)
                        .min(Duration::from_millis(20))
                });
            s = self
                .cv
                .wait_timeout(s, slice)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        };
        s.agents[agent] = Status::Running;
        drop(self.acquire_locked(s));
        (start.elapsed().as_secs_f64() * 1000.0, progressed)
    }

    /// Tears the pool down: every parked agent unblocks and its thread exits.
    /// The caller must not be holding the baton.
    pub(crate) fn shutdown(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.shutdown = true;
        s.baton_held = false;
        self.cv.notify_all();
    }
}
