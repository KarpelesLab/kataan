//! Unicode character properties and case mapping, pinned to the UCD version of
//! the generated tables in `unicode_data` (17.0.0).
//!
//! JavaScript semantics must not depend on the Rust toolchain: `char::to_uppercase`,
//! `char::is_whitespace`, `char::is_alphabetic` and friends follow whatever
//! Unicode version the compiler's `core` ships, so a toolchain bump silently
//! changes `"ɷ".toUpperCase()`. Everything JS-observable — `String.prototype`
//! case conversion, identifier and white-space classification, the legacy regex
//! `Canonicalize` — goes through this module instead. Works without `std`; the
//! string mappers need `alloc`.

use crate::unicode_data as d;

pub use crate::unicode_data::UNICODE_VERSION;

/// Unicode `White_Space` (what `char::is_whitespace` answers, but pinned). Not
/// the ECMAScript *WhiteSpace* production; see `realm::is_js_whitespace`.
#[inline]
#[must_use]
pub fn is_white_space(c: char) -> bool {
    match c {
        '\t'..='\r' | ' ' => true,
        _ if (c as u32) < 0x85 => false,
        _ => d::in_ranges(d::WHITE_SPACE, c as u32),
    }
}

/// General category `Zs` (space separator).
#[inline]
#[must_use]
pub fn is_space_separator(c: char) -> bool {
    c == ' ' || (c as u32 >= 0xA0 && d::in_ranges(d::SPACE_SEPARATOR, c as u32))
}

/// Unicode `ID_Start` (ECMAScript `UnicodeIDStart`).
#[inline]
#[must_use]
pub fn is_id_start(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    d::in_ranges(d::ID_START, c as u32)
}

/// Unicode `ID_Continue` (ECMAScript `UnicodeIDContinue`).
#[inline]
#[must_use]
pub fn is_id_continue(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    d::in_ranges(d::ID_CONTINUE, c as u32)
}

/// Unicode `Cased` (a code point; lone surrogates are not cased).
#[inline]
#[must_use]
pub fn is_cased(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_alphabetic();
    }
    d::in_ranges(d::CASED, cp)
}

/// Unicode `Case_Ignorable` (a code point; lone surrogates are not).
#[inline]
#[must_use]
pub fn is_case_ignorable(cp: u32) -> bool {
    if cp < 0x80 {
        return matches!(cp as u8, b'\'' | b'.' | b':' | b'^' | b'`');
    }
    d::in_ranges(d::CASE_IGNORABLE, cp)
}

/// Unicode `Alphabetic` (builds without the `intl` crate, which otherwise
/// answers this from the same UCD version).
#[cfg(not(feature = "intl"))]
#[inline]
#[must_use]
pub fn is_alphabetic(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    d::in_ranges(d::ALPHABETIC, c as u32)
}

/// Unicode `Lowercase` (builds without the `intl` crate).
#[cfg(not(feature = "intl"))]
#[inline]
#[must_use]
pub fn is_lowercase(c: char) -> bool {
    d::in_ranges(d::LOWERCASE, c as u32)
}

/// Unicode `Uppercase` (builds without the `intl` crate).
#[cfg(not(feature = "intl"))]
#[inline]
#[must_use]
pub fn is_uppercase(c: char) -> bool {
    d::in_ranges(d::UPPERCASE, c as u32)
}

/// General category `N` — `Nd`, `Nl` or `No` (builds without the `intl` crate).
#[cfg(not(feature = "intl"))]
#[inline]
#[must_use]
pub fn is_numeric(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    d::in_ranges(d::NUMERIC, c as u32)
}

/// `Alphabetic` or general category `N` (builds without the `intl` crate).
#[cfg(not(feature = "intl"))]
#[inline]
#[must_use]
pub fn is_alphanumeric(c: char) -> bool {
    is_alphabetic(c) || is_numeric(c)
}

/// General category `Cc` (closed form, fixed by Unicode stability policy).
#[inline]
#[must_use]
pub fn is_control(c: char) -> bool {
    matches!(c, '\0'..='\x1F' | '\x7F'..='\u{9F}')
}

/// The simple (one-to-one) mapping of `cp` through a delta-run table.
#[inline]
fn simple_map(runs: &[(u32, u32, i32, u32)], cp: u32) -> u32 {
    let i = runs.partition_point(|&(lo, _, _, _)| lo <= cp);
    if i == 0 {
        return cp;
    }
    let (lo, hi, delta, step) = runs[i - 1];
    if cp <= hi && (cp - lo).is_multiple_of(step) {
        cp.wrapping_add_signed(delta)
    } else {
        cp
    }
}

/// The unconditional multi-code-point mapping of `cp`, if any.
#[inline]
fn special(table: &'static [(u32, [char; 3])], cp: u32) -> Option<&'static [char; 3]> {
    table
        .binary_search_by_key(&cp, |&(c, _)| c)
        .ok()
        .map(|i| &table[i].1)
}

/// The full case mapping of one character: up to three characters.
#[derive(Clone, Debug)]
pub struct CaseMapping {
    chars: [char; 3],
    pos: u8,
}

impl CaseMapping {
    fn one(c: char) -> Self {
        Self {
            chars: [c, '\0', '\0'],
            pos: 0,
        }
    }
}

impl Iterator for CaseMapping {
    type Item = char;
    #[inline]
    fn next(&mut self) -> Option<char> {
        let c = *self.chars.get(usize::from(self.pos))?;
        // Only the first slot may legitimately hold U+0000 (`"\0".toUpperCase()`).
        if c == '\0' && self.pos > 0 {
            return None;
        }
        self.pos += 1;
        Some(c)
    }
}

fn full_map(c: char, upper: bool) -> CaseMapping {
    let cp = c as u32;
    if cp < 0x80 {
        let b = cp as u8;
        let m = if upper {
            b.to_ascii_uppercase()
        } else {
            b.to_ascii_lowercase()
        };
        return CaseMapping::one(char::from(m));
    }
    let (sp, runs) = if upper {
        (d::SPECIAL_UPPER, d::UPPER_RUNS)
    } else {
        (d::SPECIAL_LOWER, d::LOWER_RUNS)
    };
    if let Some(&chars) = special(sp, cp) {
        return CaseMapping { chars, pos: 0 };
    }
    CaseMapping::one(char::from_u32(simple_map(runs, cp)).unwrap_or(c))
}

/// The full (context-free) uppercase mapping of `c`: `SpecialCasing.txt`'s
/// unconditional entries, else the simple mapping (`ß` → `SS`).
#[inline]
#[must_use]
pub fn to_uppercase(c: char) -> CaseMapping {
    full_map(c, true)
}

/// The full (context-free) lowercase mapping of `c` (`İ` → `i̇`). The
/// context-sensitive `Final_Sigma` rule needs the surrounding text; the string
/// mappers below apply it.
#[inline]
#[must_use]
pub fn to_lowercase(c: char) -> CaseMapping {
    full_map(c, false)
}

/// Case-maps a WTF-8 string with the locale-independent full mappings
/// (ECMAScript `String.prototype.toUpperCase`/`toLowerCase`), including the
/// `Final_Sigma` rule when lowercasing. Lone surrogate code points have no case
/// and pass through unchanged. Valid UTF-8 in gives valid UTF-8 out.
#[cfg(feature = "alloc")]
#[must_use]
pub fn case_map_wtf8(bytes: &[u8], upper: bool) -> alloc::vec::Vec<u8> {
    if bytes.is_ascii() {
        return if upper {
            bytes.to_ascii_uppercase()
        } else {
            bytes.to_ascii_lowercase()
        };
    }
    let mut out = alloc::vec::Vec::with_capacity(bytes.len());
    case_map_each(bytes, upper, |cp| {
        if cp < 0x80 {
            out.push(cp as u8);
        } else {
            crate::wtf8::encode_code_point(cp, &mut out);
        }
    });
    out
}

/// [`case_map_wtf8`] over a `&str`: the full uppercase of `s`.
#[cfg(feature = "alloc")]
#[must_use]
pub fn to_uppercase_str(s: &str) -> alloc::string::String {
    case_map_str(s, true)
}

/// [`case_map_wtf8`] over a `&str`: the full lowercase of `s` (with `Final_Sigma`).
#[cfg(feature = "alloc")]
#[must_use]
pub fn to_lowercase_str(s: &str) -> alloc::string::String {
    case_map_str(s, false)
}

#[cfg(feature = "alloc")]
fn case_map_str(s: &str, upper: bool) -> alloc::string::String {
    if s.is_ascii() {
        return if upper {
            s.to_ascii_uppercase()
        } else {
            s.to_ascii_lowercase()
        };
    }
    let mut out = alloc::string::String::with_capacity(s.len());
    // A `&str` holds no surrogates and every mapping target is a scalar value,
    // so the fallback is unreachable.
    case_map_each(s.as_bytes(), upper, |cp| {
        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
    });
    out
}

/// The case-mapping walk shared by the WTF-8 and `&str` mappers: hands each
/// output code point to `emit`.
#[cfg(feature = "alloc")]
#[inline]
fn case_map_each(bytes: &[u8], upper: bool, mut emit: impl FnMut(u32)) {
    // `Final_Sigma` lookbehind: whether the last code point that is not
    // case-ignorable was cased.
    let mut cased_before = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            emit(u32::from(if upper {
                b.to_ascii_uppercase()
            } else {
                b.to_ascii_lowercase()
            }));
            if !upper && !is_case_ignorable(u32::from(b)) {
                cased_before = b.is_ascii_alphabetic();
            }
            i += 1;
            continue;
        }
        let (cp, len) = crate::wtf8::decode_code_point(bytes, i);
        i += len;
        let Some(c) = char::from_u32(cp) else {
            // A lone surrogate: no case, neither cased nor case-ignorable.
            emit(cp);
            cased_before = false;
            continue;
        };
        if !upper && cp == 0x03A3 {
            // Final_Sigma: preceded by a cased letter (skipping case-ignorables)
            // and not followed by one.
            let fin = cased_before && !cased_after(bytes, i);
            emit(if fin { 0x03C2 } else { 0x03C3 });
            cased_before = true;
            continue;
        }
        for m in full_map(c, upper) {
            emit(m as u32);
        }
        if !upper && !is_case_ignorable(cp) {
            cased_before = is_cased(cp);
        }
    }
}

/// Whether a cased code point follows byte offset `i`, skipping case-ignorables.
#[cfg(feature = "alloc")]
fn cased_after(bytes: &[u8], mut i: usize) -> bool {
    while i < bytes.len() {
        let (cp, len) = crate::wtf8::decode_code_point(bytes, i);
        if (0xD800..=0xDFFF).contains(&cp) {
            return false;
        }
        if !is_case_ignorable(cp) {
            return is_cased(cp);
        }
        i += len;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    fn up(c: char) -> String {
        to_uppercase(c).collect()
    }
    fn low(c: char) -> String {
        to_lowercase(c).collect()
    }

    #[test]
    fn pinned_to_unicode_17() {
        assert_eq!(UNICODE_VERSION, (17, 0, 0));
        // U+0277 gained an uppercase (U+A7DD) only after Unicode 17.0.
        assert_eq!(up('\u{277}'), "\u{277}");
    }

    #[cfg(feature = "intl")]
    #[test]
    fn same_unicode_version_as_intl() {
        assert_eq!(UNICODE_VERSION, intl::unicode::UNICODE_VERSION);
    }

    #[test]
    fn case_mappings() {
        assert_eq!(up('a'), "A");
        assert_eq!(up('\0'), "\0");
        assert_eq!(low('\0'), "\0");
        assert_eq!(up('ß'), "SS");
        assert_eq!(up('\u{1F0}'), "J\u{30C}");
        assert_eq!(up('\u{390}'), "\u{399}\u{308}\u{301}");
        assert_eq!(low('\u{130}'), "i\u{307}");
        assert_eq!(up('\u{1C5}'), "\u{1C4}");
        assert_eq!(low('\u{1C5}'), "\u{1C6}");
        assert_eq!(up('\u{101}'), "\u{100}");
        assert_eq!(up('\u{100}'), "\u{100}");
        assert_eq!(low('\u{100}'), "\u{101}");
        assert_eq!(low('\u{10400}'), "\u{10428}");
        assert_eq!(up('\u{10428}'), "\u{10400}");
        assert_eq!(up('µ'), "\u{39C}");
        assert_eq!(up('ÿ'), "\u{178}");
        assert_eq!(low('K'), "k");
        assert_eq!(low('\u{212A}'), "k");
    }

    #[test]
    fn final_sigma() {
        assert_eq!(to_lowercase_str("ὈΔΥΣΣΕΎΣ"), "ὀδυσσεύς");
        assert_eq!(to_lowercase_str("Σ"), "σ");
        assert_eq!(to_lowercase_str("AΣ"), "aς");
        assert_eq!(to_lowercase_str("AΣB"), "aσb");
        assert_eq!(to_lowercase_str("A.Σ"), "a.ς");
        assert_eq!(to_lowercase_str("AΣ.b"), "aσ.b");
        assert_eq!(to_lowercase_str("AΣ "), "aς ");
        assert_eq!(to_uppercase_str("straße"), "STRASSE");
        assert_eq!(to_lowercase_str("ABC"), "abc");
    }

    #[test]
    fn surrogates_pass_through() {
        let mut s = alloc::vec::Vec::new();
        crate::wtf8::encode_code_point(u32::from('a'), &mut s);
        crate::wtf8::encode_code_point(0xD800, &mut s);
        crate::wtf8::encode_code_point(u32::from('ß'), &mut s);
        let mut want = alloc::vec::Vec::new();
        crate::wtf8::encode_code_point(u32::from('A'), &mut want);
        crate::wtf8::encode_code_point(0xD800, &mut want);
        want.extend_from_slice(b"SS");
        assert_eq!(case_map_wtf8(&s, true), want);
    }

    #[test]
    fn properties() {
        assert!(is_white_space('\u{85}'));
        assert!(!is_white_space('\u{FEFF}'));
        assert!(is_white_space('\u{3000}'));
        assert!(is_space_separator('\u{202F}'));
        assert!(!is_space_separator('\t'));
        assert!(is_id_start('\u{37A}'));
        assert!(!is_id_start('\u{2E2F}'));
        assert!(is_id_continue('\u{B7}'));
        assert!(is_cased(u32::from('ª')));
        assert!(is_case_ignorable(0x345));
    }
}
