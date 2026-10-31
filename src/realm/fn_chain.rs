//! The prototype-chain read cache for a VM closure receiver.
//!
//! A static class member lives on the constructor, a [`Cell::VmFunction`]
//! whose own named properties sit in its auxiliary object and whose
//! `[[Prototype]]` is a side-table link: `class C extends B {}` makes `C`'s
//! `[[Prototype]]` `B`, so `C.kind` for a `static get kind()` of `B` walks from
//! one function to the next. The shape-based [`crate::ic::ChainCache`] only
//! describes ordinary objects, so such a read used to take the general walk on
//! every execution. [`FnChain`] records the walk instead, as the links the
//! method-call caches use (`builtins_fast::MethodSiteCache`): each link's
//! handle, its auxiliary object and its key epoch, the holder last.
//!
//! A hit is exact. Every link is watched (`Realm::watch_link`), so while the
//! prototype epoch is the one the walk was validated at, no link has changed
//! its own keys, its auxiliary object or its `[[Prototype]]`: the receiver
//! being the recorded first link then proves the whole walk, and only the
//! holder's property is read (live). Under a newer epoch the walk is checked
//! link by link (identity, auxiliary object, key epoch) before it is trusted
//! again. Only VM closures and ordinary non-wrapper objects are links (never a
//! proxy, a host-exotic object, a built-in function whose properties the host
//! may synthesize), and only for a named key that no closure synthesizes or
//! keeps elsewhere.

use super::{ChainHit, Realm};
use crate::cell::Cell;
use crate::heap::Handle;
use crate::ic::PropertyCache;
use alloc::vec::Vec;

/// A recorded walk from a VM closure to the holder of a key (see the module
/// docs).
pub struct FnChain {
    /// `(handle, auxiliary object, key epoch)` per link, receiver first and
    /// holder last.
    links: Vec<(u64, Option<u64>, u32)>,
    /// The object holding the holder's named properties (raw handle).
    holder_props: u64,
    /// Whether the holder's property is an accessor (else a data property).
    accessor: bool,
    /// The prototype epoch the walk was last validated at with every link
    /// watched (`0`: never).
    stamp: u64,
    /// `Realm::cell_swaps` when recorded.
    swaps: u32,
}

/// Whether `key` may be cached on a closure chain: a named key no VM
/// closure synthesizes (`name`, `length`), keeps elsewhere (`prototype`) or
/// resolves specially (the legacy `caller`/`arguments`), and not internal.
fn key_ok(key: &str) -> bool {
    !matches!(
        key,
        "name" | "length" | "prototype" | "caller" | "arguments" | "__proto__"
    ) && key
        .as_bytes()
        .first()
        .is_some_and(|b| *b != 0 && !b.is_ascii_digit() && *b != b'-')
}

impl Realm {
    /// One link of a closure chain: its auxiliary object and key epoch, and
    /// the next link. `None` for anything but a VM closure or an ordinary
    /// non-wrapper object, and for a host-exotic one.
    fn fn_link(&self, h: Handle) -> Option<(Option<u64>, u32, Option<Handle>)> {
        if self.is_host_exotic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) if !o.is_prim_wrapper() => Some((None, o.key_epoch(), o.proto())),
            Cell::VmFunction { .. } => {
                let (aux, epoch) = match self.aux_props.get(&h.to_raw()) {
                    Some(a) => (
                        Some(a.to_raw()),
                        self.heap.get(*a)?.as_object()?.key_epoch(),
                    ),
                    None => (None, 0),
                };
                Some((aux, epoch, self.object_proto(h)))
            }
            _ => None,
        }
    }

    /// The object holding a link's named properties: the link itself, or
    /// its auxiliary object `aux`.
    fn fn_link_props(&self, raw: u64, aux: Option<u64>) -> Option<&crate::object::Object> {
        self.heap
            .get(Handle::from_raw(aux.unwrap_or(raw)))?
            .as_object()
    }

    /// The site cache's answer for the read of `key` from the VM closure at
    /// `recv` (see the module docs): the holder's data value or accessor
    /// pair. On a miss, re-arms the cache (within its fill budget) and
    /// returns `None`; the caller then takes the general path.
    pub fn fn_chain_get(
        &mut self,
        recv: Handle,
        key: &str,
        cache: &mut PropertyCache,
    ) -> Option<ChainHit> {
        if let Some(c) = cache
            .chain
            .as_deref_mut()
            .and_then(|c| c.fn_links.as_deref_mut())
            && let Some(hit) = self.fn_chain_hit(recv, key, c)
        {
            return Some(hit);
        }
        self.fn_chain_fill(recv, key, cache);
        None
    }

    /// A hit of the recorded walk `c` from `recv`, revalidating (and
    /// restamping) it under a newer prototype epoch.
    fn fn_chain_hit(&self, recv: Handle, key: &str, c: &mut FnChain) -> Option<ChainHit> {
        if c.links.first()?.0 != recv.to_raw() || c.swaps != self.cell_swaps {
            return None;
        }
        let epoch = crate::object::proto_epoch();
        if c.stamp != epoch {
            let mut cur = Some(recv);
            for &(raw, aux, e) in &c.links {
                let h = cur?;
                if h.to_raw() != raw {
                    return None;
                }
                let (a, k, next) = self.fn_link(h)?;
                if a != aux || k != e {
                    return None;
                }
                cur = next;
            }
            c.stamp = if c
                .links
                .iter()
                .all(|&(raw, aux, _)| self.link_watched(raw, aux))
            {
                epoch
            } else {
                0
            };
        }
        let o = self
            .heap
            .get(Handle::from_raw(c.holder_props))?
            .as_object()?;
        if c.accessor {
            let (g, s) = o.accessor(key)?;
            Some(ChainHit::Accessor(g, s, c.links.len() == 1))
        } else if o.accessor(key).is_some() {
            None
        } else {
            o.get(key).map(ChainHit::Data)
        }
    }

    /// Records the walk of `key` from the VM closure at `recv` in `cache`
    /// when every link up to the holder is one [`fn_link`](Self::fn_link)
    /// accepts (at most eight); leaves it unarmed otherwise.
    fn fn_chain_fill(&mut self, recv: Handle, key: &str, cache: &mut PropertyCache) {
        if cache.chain_fills >= crate::ic::CHAIN_FILL_LIMIT {
            cache.chain = None;
            return;
        }
        cache.chain_fills += 1;
        cache.chain = None;
        if !key_ok(key) || !matches!(self.heap.get(recv), Some(Cell::VmFunction { .. })) {
            return;
        }
        let mut links = Vec::new();
        let mut cur = Some(recv);
        for _ in 0..8 {
            let Some(h) = cur else {
                return;
            };
            let Some((aux, epoch, next)) = self.fn_link(h) else {
                return;
            };
            links.push((h.to_raw(), aux, epoch));
            let Some(o) = self.fn_link_props(h.to_raw(), aux) else {
                // A closure without an auxiliary object owns no named key.
                cur = next;
                continue;
            };
            let accessor = o.accessor(key).is_some();
            if !accessor && !o.contains(key) {
                cur = next;
                continue;
            }
            let holder_props = aux.unwrap_or(h.to_raw());
            for &(raw, aux, _) in &links {
                self.watch_link(raw, aux);
            }
            cache.chain = Some(alloc::boxed::Box::new(crate::ic::ChainCache {
                shapes: Vec::new(),
                slot: 0,
                accessor,
                absent: false,
                fn_links: Some(alloc::boxed::Box::new(FnChain {
                    links,
                    holder_props,
                    accessor,
                    stamp: crate::object::proto_epoch(),
                    swaps: self.cell_swaps,
                })),
            }));
            return;
        }
    }
}
