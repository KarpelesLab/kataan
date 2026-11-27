//! Proxy internal methods whose trap lookups can be decided without running
//! code.
//!
//! A proxy's internal method begins with `GetMethod(handler, trapName)`: a
//! full `[[Get]]` on the handler. When the handler is an ordinary object and
//! neither it nor anything on its prototype chain has an accessor (or a proxy)
//! for that name, the lookup runs no code, so finding it absent (or
//! `undefined` / `null`) is unobservable and the internal method is exactly
//! the target's. The helpers here take that path and hand everything else to
//! the full protocol.

use super::*;

impl Interp {
    /// The write of OrdinarySetWithOwnDescriptor's final steps
    /// (`Receiver.[[GetOwnProperty]](P)` then a value-only define or
    /// `CreateDataProperty(Receiver, P, V)`) for a **proxy** Receiver, when it
    /// needs no code: every proxy on the way is live with neither a
    /// `getOwnPropertyDescriptor` nor a `defineProperty` trap (so both internal
    /// methods are the target's), and the final ordinary target takes the
    /// write as a plain store. `Some(true)` once written; `None` (nothing
    /// done, nothing observed) for the full protocol.
    pub(crate) fn proxy_receiver_define_quiet(
        &mut self,
        receiver: Handle,
        key: &str,
        value: NanBox,
    ) -> Option<bool> {
        let mut r = receiver;
        for _ in 0..64 {
            let Some((target, handler)) = self.realm.proxy_at(r) else {
                return self
                    .realm
                    .create_data_property_plain(r, key, value)
                    .then_some(true);
            };
            if self.realm.proxy_revoked(r)
                || !self
                    .realm
                    .proxy_trap_absent_quiet(handler, "getOwnPropertyDescriptor")
                || !self
                    .realm
                    .proxy_trap_absent_quiet(handler, "defineProperty")
            {
                return None;
            }
            r = target;
        }
        None
    }
}
