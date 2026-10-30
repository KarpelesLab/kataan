//! Text for engine error messages: what a property key or a call's callee
//! looks like to the reader of `TypeError: Cannot read properties of
//! undefined (reading 'x')` or `TypeError: u is not a function`. Messages
//! are not specified; these only ever render a value that converts to a
//! string without running code (a primitive), and never change which error
//! is thrown.

use super::{Handle, NanBox, Realm};
use alloc::string::String;

/// The most characters of a key a message quotes.
const KEY_TEXT_MAX: usize = 100;

impl Realm {
    /// `key` as a message quotes it: a string's text, a symbol's
    /// `Symbol(description)`, any other primitive's `ToString`; `None` for an
    /// object (converting it could run code).
    #[must_use]
    pub fn message_key_text(&self, key: NanBox) -> Option<String> {
        let text = match key.as_handle().map(Handle::from_raw) {
            Some(h) => {
                if let Some((desc, _)) = self.symbol_at(h) {
                    alloc::format!("Symbol({desc})")
                } else {
                    self.string_value(h)?
                }
            }
            None => self.to_display_string(key),
        };
        Some(match text.char_indices().nth(KEY_TEXT_MAX) {
            Some((i, _)) => alloc::format!("{}...", &text[..i]),
            None => text,
        })
    }

    /// `Cannot read properties of null (reading 'k')` (or `set` … `setting`)
    /// for a nullish `recv` and the text of the key (see
    /// [`message_key_text`](Self::message_key_text); the key part is left out
    /// when there is none to show).
    #[must_use]
    pub fn nullish_access_message(recv: NanBox, key: Option<&str>, set: bool) -> String {
        let what = if recv.is_null() { "null" } else { "undefined" };
        let (verb, ing) = if set {
            ("set", "setting")
        } else {
            ("read", "reading")
        };
        match key {
            Some(k) => alloc::format!("Cannot {verb} properties of {what} ({ing} '{k}')"),
            None => alloc::format!("Cannot {verb} properties of {what}"),
        }
    }
}
