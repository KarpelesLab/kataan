//! What lets a collection run inside JS code that a **host built-in** re-entered
//! (`Array.from(gen)`, `new Set(gen)`, an iterator helper draining a
//! generator...): the values the built-in's Rust frame holds across the
//! re-entry, and the bookkeeping that tells the nested VM run whether every
//! frame below it is rooted.
//!
//! A VM safepoint collects only when it can enumerate the whole root set. A
//! built-in that steps an iterator keeps its iterator record, its callback and
//! the values gathered so far in Rust locals, so the nested run it starts
//! (a generator resumption) was always refused collection, and a long
//! allocating generator body drained by a built-in grew without bound.
//!
//! The protocol, end to end:
//!
//! * Every VM-to-host crossing (`with_host` in `crate::nbvm`) counts itself in
//!   [`crossings`](HostRoots::crossings). One made from a context whose every
//!   frame is published (or lent: `with_host_regs`) and whose collector is on
//!   also pushes a [`HostLink`] naming that context, so a collection further
//!   in can root its frames. A crossing that *is* a call of a host function
//!   (a call, construct or iterator-step site) is marked a **relay**.
//! * Every host function call (`Interp::call_with_this`, `Interp::construct`)
//!   takes the next [`serial`](HostRoots::serial) number.
//! * A built-in that wants collection inside what it re-enters opens a
//!   **root scope** (`Interp::root_scope`): it pushes the values it holds onto
//!   [`vals`](HostRoots::vals) (a root of every collection). The scope is
//!   *active* only when no host frame between the VM and the built-in holds
//!   values of its own: the innermost crossing is a linked relay and the
//!   built-in's own call is the first host call made since it (its serial is
//!   the link's plus one), or an active scope stepping an iterator relayed to
//!   it the same way ([`relay`](HostRoots::relay)). An active scope records
//!   its crossing in [`scope`](HostRoots::scope). The scope closes (values
//!   popped, state restored) when its guard drops, on every path including a
//!   panic.
//! * Right before one audited re-entry (a generator resumption or a VM `next`
//!   call, with nothing in between that holds unrooted values) an active
//!   scope **arms** it: [`armed`](HostRoots::armed) = the current crossing.
//! * The nested VM entry takes the arm ([`HostRoots::take_arm`]): only when it
//!   names the current crossing, and only once, so whatever else the host
//!   segment re-enters (a getter, a proxy trap) stays uncollectable, as before.
//!   The nested context then collects as an outermost one would, adding the
//!   linked contexts' frames to its roots.
//!
//! * The collecting event loop's own job runs under a scope of the loop's
//!   ([`LOOP`]): its handler, when a VM function, is an armed re-entry, and
//!   a host handler (a built-in's continuation) may open an active scope as
//!   if relayed to. A nested entry armed at the loop level collects with no
//!   link: nothing but the interpreter's own roots, its job queues and the
//!   root stack is live below it.
//! * A built-in's callback (`Interp::call_cb`) is armed the same way as a
//!   step: a VM function is the re-entry; anything else is relayed to.
//!
//! The collector does not move objects, so a copy of a value is as good a
//! root as the original, and nothing needs updating afterwards.

use crate::nanbox::NanBox;
use alloc::vec::Vec;

/// One VM-to-host crossing made from a fully published VM context.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostLink {
    /// The crossing's number (its [`HostRoots::crossings`] count once made).
    pub(crate) crossing: u32,
    /// The address of the VM context that crossed (a `crate::nbvm` `Ctx`),
    /// alive and suspended while the link is on the stack.
    pub(crate) ctx: usize,
    /// The register file of that context's running frame, when it was not
    /// published but lent to the crossing (address and length; 0 when none).
    pub(crate) regs: usize,
    /// See [`regs`](Self::regs).
    pub(crate) regs_len: usize,
    /// [`HostRoots::serial`] when the crossing was made.
    pub(crate) serial: u32,
    /// Whether the crossing is a call of a host function (a relay: the
    /// first host call it makes may open an active root scope).
    pub(crate) relay: bool,
}

/// See the module docs.
#[derive(Default, Debug)]
pub(crate) struct HostRoots {
    /// Values host frames hold across a re-entry: roots of every collection.
    pub(crate) vals: Vec<NanBox>,
    /// VM-to-host crossings currently on the Rust stack.
    pub(crate) crossings: u32,
    /// The crossings among them made from a fully published context,
    /// innermost last.
    pub(crate) links: Vec<HostLink>,
    /// The serial number of the last host function call begun.
    pub(crate) serial: u32,
    /// An active scope's step through a host `next` (or call of a host
    /// callback): `(scope, serial)` when it began, so the call it makes may
    /// open an active scope too.
    pub(crate) relay: Option<(u32, u32)>,
    /// The crossing whose host segment runs inside an active root scope (0:
    /// none).
    pub(crate) scope: u32,
    /// The crossing at which the next nested VM entry may collect (0: none).
    pub(crate) armed: u32,
}

/// The flag of a [`HostRoots::scope`] / [`HostRoots::armed`] value (over the
/// crossing count it was opened at) of a scope at the level of the
/// collecting event loop: what is below it is the loop's frames, which root
/// what they hold (the loop's keep list and its queues, and the job being
/// run under a scope of its own), and the outermost hosted run whose body
/// has returned (its completion value is the loop's keep, its own job queue
/// is empty), crossed by no relay: exactly the state a VM resumption job of
/// the loop collects in (`Interp::job_gc`).
pub(crate) const LOOP: u32 = 1 << 31;

impl HostRoots {
    /// Whether a root scope opened now by the host call numbered `call` may
    /// be active (see the module docs): its crossing when so, else 0. (Call
    /// numbers are never reused, so only the one call made first after the
    /// crossing or relay qualifies, however many scopes it opens.)
    #[inline]
    pub(crate) fn activation(&self, call: u32) -> u32 {
        let c = self.crossings;
        // The event loop's level: a call an active scope of that level (the
        // loop's own, over the job it runs) relayed to.
        if let Some((rc, rs)) = self.relay
            && rc == c | LOOP
            && call == rs.wrapping_add(1)
        {
            return c | LOOP;
        }
        let Some(link) = self.links.last().filter(|l| c != 0 && l.crossing == c) else {
            return 0;
        };
        if link.relay && call == link.serial.wrapping_add(1) {
            return c;
        }
        if let Some((rc, rs)) = self.relay
            && rc == c
            && call == rs.wrapping_add(1)
        {
            return c;
        }
        0
    }

    /// Arms the next nested VM entry when the running host segment holds an
    /// active root scope; returns whether it did.
    #[inline]
    pub(crate) fn arm(&mut self) -> bool {
        if self.live() {
            self.armed = self.scope;
            true
        } else {
            false
        }
    }

    /// Whether the running host segment holds an active root scope: one
    /// opened at the current crossing, or at the event loop's level with no
    /// crossing made since.
    #[inline]
    pub(crate) fn live(&self) -> bool {
        self.scope != 0 && self.scope & !LOOP == self.crossings
    }

    /// Takes the arm for a nested VM entry made now: whether the entry may
    /// collect (the arm names the current crossing, which is linked); the
    /// arm is spent either way.
    #[inline]
    pub(crate) fn take_arm(&mut self) -> bool {
        let armed = core::mem::take(&mut self.armed);
        if armed & LOOP != 0 {
            return armed & !LOOP == self.crossings;
        }
        armed != 0
            && armed == self.crossings
            && self.links.last().is_some_and(|l| l.crossing == armed)
    }
}
