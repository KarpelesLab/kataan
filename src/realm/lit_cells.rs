//! Compiled code's **string-literal cells**.
//!
//! Evaluating a string literal (`Op::NewString`) yields a String whose
//! identity is not observable: two Strings with the same contents are the
//! same value to every operation (`===`, `SameValue`, property keys,
//! `Map`/`Set` keys). So every evaluation of one literal may hand back one
//! cell, and compiled code keeps that cell's NanBox in a word of its own
//! ([`LitCells`], one word per literal site, owned by the code): the
//! literal is then a load of that word, with a call only the first time,
//! when the word is still empty (`0`, never a valid NanBox of a String).
//!
//! The words hold heap handles, so they are GC roots for as long as the code
//! lives: a filled [`LitCells`] registers itself (a [`Weak`]) with the realm,
//! which roots the handles of every live one ([`Realm::lit_cell_roots`]) and
//! forgets the dead ones. A compaction moves every handle: it empties every
//! word first ([`Realm::lit_cells_retire`]), and the code refills them.

use super::Realm;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::boxed::Box;
use alloc::rc::{Rc, Weak};
use alloc::vec::Vec;

/// The string-literal words of one compiled body: word `i` is `0` (not yet
/// evaluated) or the NanBox of the String that literal site `i` evaluates
/// to. The words live at a stable address (the code holds it).
pub struct LitCells {
    words: Box<[core::cell::Cell<u64>]>,
    /// Whether the realm's registry holds this table already.
    registered: core::cell::Cell<bool>,
}

#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
impl LitCells {
    /// `n` empty words.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            words: (0..n).map(|_| core::cell::Cell::new(0)).collect(),
            registered: core::cell::Cell::new(false),
        }
    }

    /// The address of word `i` (stable for the table's life).
    pub(crate) fn word_addr(&self, i: usize) -> u64 {
        self.words[i].as_ptr() as usize as u64
    }

    /// The handles the filled words hold.
    fn roots(&self, out: &mut Vec<Handle>) {
        out.extend(
            self.words
                .iter()
                .filter_map(|w| NanBox::from_bits(w.get()).as_handle().map(Handle::from_raw)),
        );
    }
}

#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
impl Realm {
    /// The value of literal site `i` of `cells` whose text is `text`: the
    /// literal's String (`new_literal_string`), now also kept in word `i`
    /// for the code to load, and the table registered as a GC root.
    pub(crate) fn fill_lit_cell(&mut self, cells: &Rc<LitCells>, i: usize, text: &str) -> NanBox {
        let v = NanBox::handle(self.new_literal_string(text).to_raw());
        let Some(w) = cells.words.get(i) else {
            return v;
        };
        if !cells.registered.get() {
            cells.registered.set(true);
            // (Forget the dead tables now and then: amortized O(1).)
            if self.lit_cells.len() >= self.lit_cells_prune {
                self.lit_cells.retain(|t| t.strong_count() > 0);
                self.lit_cells_prune = (self.lit_cells.len() * 2).max(16);
            }
            self.lit_cells.push(Rc::downgrade(cells));
        }
        w.set(v.to_bits());
        v
    }

    /// Fills word `i` of `cells` with `v` (a heap value the code keeps:
    /// `crate::jit::sink`'s stand-ins), the table registered as a GC root.
    pub(crate) fn fill_lit_word(&mut self, cells: &Rc<LitCells>, i: usize, v: NanBox) {
        let Some(w) = cells.words.get(i) else {
            return;
        };
        if !cells.registered.get() {
            cells.registered.set(true);
            if self.lit_cells.len() >= self.lit_cells_prune {
                self.lit_cells.retain(|t| t.strong_count() > 0);
                self.lit_cells_prune = (self.lit_cells.len() * 2).max(16);
            }
            self.lit_cells.push(Rc::downgrade(cells));
        }
        w.set(v.to_bits());
    }

    /// The handles every live table's filled words hold (GC roots).
    pub(crate) fn lit_cell_roots(&self, out: &mut Vec<Handle>) {
        for t in self.lit_cells.iter().filter_map(Weak::upgrade) {
            t.roots(out);
        }
    }

    /// Before a compaction moves every handle: empties every word (the code
    /// refills them) and forgets the tables.
    pub(crate) fn lit_cells_retire(&mut self) {
        for t in self.lit_cells.drain(..).filter_map(|t| t.upgrade()) {
            for w in &t.words {
                w.set(0);
            }
            t.registered.set(false);
        }
    }
}
