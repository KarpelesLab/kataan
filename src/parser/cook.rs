//! "Cooking" — turning a literal token's raw source text into its runtime
//! value: numeric literals into `f64`, string/template bodies into decoded
//! text, and `BigInt` literals into normalized digit strings.
//!
//! The lexer has already validated literal *shape* (so, e.g., a `\x` escape is
//! known to be followed by two hex digits), which keeps these routines focused
//! on decoding rather than re-validating.

use crate::common::Span;
use crate::error::{Error, Result};
use alloc::string::String;

/// Decodes a numeric-literal token's text into an `f64`.
///
/// Handles decimal integers and floats with exponents, the `0x` / `0o` / `0b`
/// radix prefixes, and `_` digit separators. Radix integers are accumulated in
/// floating point, which is exact up to 2^53 and correctly rounds larger
/// magnitudes to the nearest representable double for typical inputs.
#[must_use]
pub(super) fn number(text: &str) -> f64 {
    // Separators are syntactically constrained by the lexer; we can strip them
    // unconditionally here.
    let cleaned: String = text.chars().filter(|&c| c != '_').collect();

    if let Some(rest) = strip_radix(&cleaned, ['x', 'X']) {
        return from_radix(rest, 16);
    }
    if let Some(rest) = strip_radix(&cleaned, ['o', 'O']) {
        return from_radix(rest, 8);
    }
    if let Some(rest) = strip_radix(&cleaned, ['b', 'B']) {
        return from_radix(rest, 2);
    }

    // Decimal / float / exponent. Rust's float parser is strict about a leading
    // or trailing `.`, which JS allows (`.5`, `5.`), so normalize those.
    let mut s = cleaned;
    if s.starts_with('.') {
        s.insert(0, '0');
    }
    if let Some(dot) = s.find('.') {
        // A `.` immediately followed by end-of-string or the exponent marker
        // needs a `0` inserted after it.
        let after = &s[dot + 1..];
        if after.is_empty() || after.starts_with(['e', 'E']) {
            s.insert(dot + 1, '0');
        }
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// If `s` is `0` followed by one of `markers`, returns the digits after the
/// marker.
fn strip_radix(s: &str, markers: [char; 2]) -> Option<&str> {
    let bytes = s.as_bytes();
    if bytes.len() >= 2
        && bytes[0] == b'0'
        && (s[1..2].starts_with(markers[0]) || s[1..2].starts_with(markers[1]))
    {
        Some(&s[2..])
    } else {
        None
    }
}

fn from_radix(digits: &str, radix: u32) -> f64 {
    let mut acc = 0.0_f64;
    let r = f64::from(radix);
    for c in digits.chars() {
        if let Some(d) = c.to_digit(radix) {
            acc = acc * r + f64::from(d);
        }
    }
    acc
}

/// Normalizes a `BigInt` literal token (`123n`, `0xFFn`, `1_000n`) into its
/// digit string: separators and the trailing `n` removed, any radix prefix
/// retained (so `0xFFn` → `0xFF`).
#[must_use]
pub(super) fn bigint(text: &str) -> String {
    text.chars().filter(|&c| c != '_' && c != 'n').collect()
}

/// Decodes a string-literal token (including its surrounding quotes) into its
/// runtime value.
pub(super) fn string(raw: &str, span: Span) -> Result<String> {
    // The first and last bytes are the ASCII quote characters.
    let inner = &raw[1..raw.len() - 1];
    decode_escapes(inner, span)
}

/// Decodes the escape sequences in a string or template-cooked segment.
///
/// `body` is the text *between* the delimiters. Returns the cooked value.
///
/// Lone UTF-16 surrogates (which JS strings can legally contain but Rust's
/// `String` cannot) are decoded to U+FFFD for now; the engine's eventual
/// WTF-16-capable string type will preserve them. This only affects string
/// literals that deliberately encode unpaired surrogates via `\u`.
pub(super) fn decode_escapes(body: &str, span: Span) -> Result<String> {
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some(esc) = chars.next() else {
            return Err(Error::syntax("unterminated escape sequence", span));
        };
        match esc {
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            'b' => out.push('\u{08}'),
            'f' => out.push('\u{0C}'),
            'v' => out.push('\u{0B}'),
            '0' if !chars.peek().is_some_and(|c| c.is_ascii_digit()) => out.push('\0'),
            'x' => {
                let hi = hex_digit(chars.next(), span)?;
                let lo = hex_digit(chars.next(), span)?;
                out.push(char::from(hi * 16 + lo));
            }
            'u' => decode_unicode_escape(&mut chars, &mut out, span)?,
            // Line continuation: a backslash before a line terminator is elided.
            '\n' | '\u{2028}' | '\u{2029}' => {}
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
            }
            // Any other escaped character stands for itself (covers `\\`, `\'`,
            // `\"`, `` \` ``, `\$`, `\/`, and the identity escapes).
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Decodes a `\u` escape body — either `\uXXXX` (already past the `u`) or
/// `\u{ … }` — appending the result to `out`. Handles `\uXXXX\uXXXX` surrogate
/// pairs.
fn decode_unicode_escape(
    chars: &mut core::iter::Peekable<core::str::Chars<'_>>,
    out: &mut String,
    span: Span,
) -> Result<()> {
    if chars.peek() == Some(&'{') {
        chars.next(); // `{`
        let mut value: u32 = 0;
        while let Some(&c) = chars.peek() {
            if c == '}' {
                break;
            }
            value = value
                .saturating_mul(16)
                .saturating_add(u32::from(hex_digit(chars.next(), span)?));
        }
        chars.next(); // `}`
        // A code point above U+10FFFF is an invalid escape (no cooked value).
        if value > 0x10_FFFF {
            return Err(Error::syntax("invalid escape sequence", span));
        }
        out.push(char::from_u32(value).unwrap_or('\u{FFFD}'));
        return Ok(());
    }

    let hi = read_u16_hex(chars, span)?;
    if (0xD800..=0xDBFF).contains(&hi) {
        // Possible surrogate pair: look for a following `\uXXXX` low surrogate.
        let mut clone = chars.clone();
        if clone.next() == Some('\\') && clone.next() == Some('u') {
            let lo = read_u16_hex(&mut clone, span)?;
            if (0xDC00..=0xDFFF).contains(&lo) {
                *chars = clone;
                let cp = 0x10000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
                out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                return Ok(());
            }
        }
    }
    out.push(char::from_u32(u32::from(hi)).unwrap_or('\u{FFFD}'));
    Ok(())
}

/// Reads exactly four hex digits into a `u16`.
fn read_u16_hex(chars: &mut core::iter::Peekable<core::str::Chars<'_>>, span: Span) -> Result<u16> {
    let mut v: u16 = 0;
    for _ in 0..4 {
        v = v * 16 + u16::from(hex_digit(chars.next(), span)?);
    }
    Ok(v)
}

/// Converts an expected hex-digit char to its value, erroring if missing or
/// non-hex.
fn hex_digit(c: Option<char>, span: Span) -> Result<u8> {
    match c.and_then(|c| c.to_digit(16)) {
        Some(d) => Ok(d as u8),
        None => Err(Error::syntax("invalid escape sequence", span)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp() -> Span {
        Span::new(0, 0)
    }

    #[test]
    fn decimal_numbers() {
        assert_eq!(number("0"), 0.0);
        assert_eq!(number("42"), 42.0);
        assert_eq!(number("2.5"), 2.5);
        assert_eq!(number(".5"), 0.5);
        assert_eq!(number("5."), 5.0);
        assert_eq!(number("1e3"), 1000.0);
        assert_eq!(number("1.5E-3"), 0.0015);
        assert_eq!(number("1_000_000"), 1_000_000.0);
    }

    #[test]
    fn radix_numbers() {
        assert_eq!(number("0xFF"), 255.0);
        assert_eq!(number("0xdead_beef"), 0xDEAD_BEEFu32 as f64);
        assert_eq!(number("0o17"), 15.0);
        assert_eq!(number("0b1010"), 10.0);
    }

    #[test]
    fn bigint_normalization() {
        assert_eq!(bigint("123n"), "123");
        assert_eq!(bigint("1_000n"), "1000");
        assert_eq!(bigint("0xFFn"), "0xFF");
    }

    #[test]
    fn string_escapes() {
        assert_eq!(string(r#""hello""#, sp()).unwrap(), "hello");
        assert_eq!(string(r#""a\tb\nc""#, sp()).unwrap(), "a\tb\nc");
        assert_eq!(string(r#""\x41\x42""#, sp()).unwrap(), "AB");
        assert_eq!(string(r#""A""#, sp()).unwrap(), "A");
        assert_eq!(string(r#""\u{1F600}""#, sp()).unwrap(), "\u{1F600}");
        assert_eq!(string(r"'it\'s'", sp()).unwrap(), "it's");
        // Surrogate pair for U+1F600.
        assert_eq!(string(r#""😀""#, sp()).unwrap(), "\u{1F600}");
        // Line continuation.
        assert_eq!(string("\"a\\\nb\"", sp()).unwrap(), "ab");
    }
}
