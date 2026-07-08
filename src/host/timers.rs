//! Host runtime submodule (ROADMAP §4). Installed via [`install`].
use crate::nbexec::Interp;

/// Install this area's globals into `interp`.
pub fn install(_interp: &mut Interp<'_>) {
    // Implemented incrementally — see ROADMAP.md §4.
}
