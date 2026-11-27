//! A match's named-groups object: the realm's one-step builder
//! (`Realm::named_groups_object`, which remembers each RegExp's layout).

use super::*;

impl Interp {
    /// The named-groups object of a match of `re` (see
    /// `Realm::named_groups_object`): `vals` indexed by group,
    /// `participated(i)` whether group `i` took part.
    pub(crate) fn named_groups_object(
        &mut self,
        re: &alloc::rc::Rc<crate::regex::Regex>,
        vals: &[NanBox],
        participated: impl Fn(usize) -> bool,
    ) -> NanBox {
        self.realm.named_groups_object(re, vals, participated)
    }
}
