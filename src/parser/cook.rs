//! "Cooking" — turning a literal token's raw source text into its runtime
//! value: numeric literals into `f64`, string/template bodies into decoded
//! text, and `BigInt` literals into normalized digit strings.
//!
//! The lexer has already validated literal *shape* (so, e.g., a `\x` escape is
//! known to be followed by two hex digits), which keeps these routines focused
//! on decoding rather than re-validating.

use crate::common::Span;
use crate::error::{Error, Result};
use crate::wtf8;
use alloc::string::String;
use alloc::vec::Vec;

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

    // A `LegacyOctalIntegerLiteral` — a leading `0` followed by one or more
    // octal digits (0–7), with no fraction/exponent — is interpreted as octal in
    // sloppy mode. A leading-zero run containing an `8`/`9`
    // (`NonOctalDecimalIntegerLiteral`, e.g. `09`, `0189`) stays decimal. The
    // lexer has already rejected separators and BigInt suffixes on these forms.
    if let Some(rest) = cleaned.strip_prefix('0')
        && !rest.is_empty()
        && rest.bytes().all(|b| (b'0'..=b'7').contains(&b))
    {
        return from_radix(rest, 8);
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

/// Whether a (lexer-validated) numeric-literal token is a
/// `LegacyOctalIntegerLiteral` (`0123`) or `NonOctalDecimalIntegerLiteral`
/// (`08`, `0189`) — i.e. a `0`-prefixed integer whose second character is a
/// decimal digit. (`0x`/`0o`/`0b` carry a letter; `0.5`/`0e5` a `.`/`e`; `0n` an
/// `n`; a bare `0` has no second digit — none of these match.) Such literals are
/// an early Syntax Error in strict mode.
pub(super) fn is_legacy_octal_literal(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() >= 2 && b[0] == b'0' && b[1].is_ascii_digit()
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

/// Decodes the `\uXXXX` / `\u{…}` escapes in an `IdentifierName`'s raw source
/// text into its `StringValue` (the cooked identifier name). Identifiers may
/// only contain Unicode escapes (no `\x`, no string-style escapes), and the
/// lexer has already validated each escape's shape and that every code point is
/// a valid identifier char, so this only reassembles the scalar values.
#[must_use]
pub(super) fn identifier_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        // `\u` (the lexer guarantees the `u` and a well-formed body follow).
        chars.next(); // `u`
        let mut value: u32 = 0;
        if chars.peek() == Some(&'{') {
            chars.next(); // `{`
            while let Some(&d) = chars.peek() {
                if d == '}' {
                    break;
                }
                value = value
                    .saturating_mul(16)
                    .saturating_add(d.to_digit(16).unwrap_or(0));
                chars.next();
            }
            chars.next(); // `}`
        } else {
            for _ in 0..4 {
                if let Some(d) = chars.next().and_then(|d| d.to_digit(16)) {
                    value = value * 16 + d;
                }
            }
        }
        if let Some(ch) = char::from_u32(value) {
            out.push(ch);
        }
    }
    out
}

/// Normalizes a `BigInt` literal token (`123n`, `0xFFn`, `1_000n`) into its
/// digit string: separators and the trailing `n` removed, any radix prefix
/// retained (so `0xFFn` → `0xFF`).
#[must_use]
pub(super) fn bigint(text: &str) -> String {
    text.chars().filter(|&c| c != '_' && c != 'n').collect()
}

/// Decodes a string-literal token (including its surrounding quotes) into its
/// runtime value — WTF-8 bytes preserving any lone UTF-16 surrogates.
pub(super) fn string(raw: &str, span: Span) -> Result<Vec<u8>> {
    // The first and last bytes are the ASCII quote characters.
    let inner = &raw[1..raw.len() - 1];
    decode_escapes(inner, span)
}

/// Decodes a string-literal token used as a **property key**, returning a
/// `String`. Property keys are stored in the `&str`-keyed object/shape layer, so
/// a lone surrogate in a key is decoded lossily (→ U+FFFD); a non-surrogate key
/// — the overwhelmingly common case — is unchanged. (Surrogate-correct *string
/// values* go through [`string`], which keeps the WTF-8 bytes.)
pub(super) fn string_key(raw: &str, span: Span) -> Result<String> {
    Ok(wtf8::to_string_lossy(&string(raw, span)?))
}

/// Decodes the escape sequences in a string or template-cooked segment.
///
/// `body` is the text *between* the delimiters. Returns the cooked value as
/// **WTF-8 bytes**.
///
/// Lone UTF-16 surrogates (which a JS DOMString can legally contain but Rust's
/// `String` cannot) are preserved as WTF-8 surrogate code points (via
/// [`wtf8::encode_utf16_unit`]); an adjacent `\u` high+low pair is combined into
/// the astral scalar it denotes. A string with no surrogates is byte-identical
/// to its UTF-8, so the common case is unchanged.
pub(super) fn decode_escapes(body: &str, span: Span) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(body.len());
    let mut chars = body.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '\\' {
            push_char(&mut out, c);
            continue;
        }
        let Some(esc) = chars.next() else {
            return Err(Error::syntax("unterminated escape sequence", span));
        };
        match esc {
            'n' => out.push(b'\n'),
            't' => out.push(b'\t'),
            'r' => out.push(b'\r'),
            'b' => out.push(0x08),
            'f' => out.push(0x0C),
            'v' => out.push(0x0B),
            '0' if !chars.peek().is_some_and(|c| c.is_ascii_digit()) => out.push(0),
            'x' => {
                let hi = hex_digit(chars.next(), span)?;
                let lo = hex_digit(chars.next(), span)?;
                push_char(&mut out, char::from(hi * 16 + lo));
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
            other => push_char(&mut out, other),
        }
    }
    Ok(out)
}

/// Appends a scalar `char`'s UTF-8 bytes to `out`. (A `char` is never a
/// surrogate, so this is plain UTF-8; surrogates only enter via `\u` escapes,
/// handled by [`decode_unicode_escape`].)
fn push_char(out: &mut Vec<u8>, c: char) {
    let mut buf = [0u8; 4];
    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
}

/// Decodes a `\u` escape body — either `\uXXXX` (already past the `u`) or
/// `\u{ … }` — appending the result to `out` as WTF-8 bytes. Handles
/// `\uXXXX\uXXXX` surrogate pairs (combined into one astral scalar) and
/// preserves a lone surrogate as a surrogate code point.
fn decode_unicode_escape(
    chars: &mut core::iter::Peekable<core::str::Chars<'_>>,
    out: &mut Vec<u8>,
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
        // `\u{D800}` is a lone surrogate code point — preserved, not replaced.
        wtf8::encode_code_point(value, out);
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
                wtf8::encode_code_point(cp, out);
                return Ok(());
            }
        }
    }
    // A BMP scalar, or a lone surrogate kept as a surrogate code point.
    wtf8::encode_utf16_unit(hi, out);
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
        // Cooked values are WTF-8 bytes; a non-surrogate string is byte-identical
        // to its UTF-8.
        assert_eq!(string(r#""hello""#, sp()).unwrap(), b"hello");
        assert_eq!(string(r#""a\tb\nc""#, sp()).unwrap(), b"a\tb\nc");
        assert_eq!(string(r#""\x41\x42""#, sp()).unwrap(), b"AB");
        assert_eq!(string(r#""A""#, sp()).unwrap(), b"A");
        assert_eq!(
            string(r#""\u{1F600}""#, sp()).unwrap(),
            "\u{1F600}".as_bytes()
        );
        assert_eq!(string(r"'it\'s'", sp()).unwrap(), b"it's");
        // Surrogate pair for U+1F600.
        assert_eq!(string(r#""😀""#, sp()).unwrap(), "\u{1F600}".as_bytes());
        // Line continuation.
        assert_eq!(string("\"a\\\nb\"", sp()).unwrap(), b"ab");
    }

    #[test]
    fn lone_surrogates_preserved() {
        // A lone high surrogate via `\uXXXX` is kept as a WTF-8 surrogate code
        // point (3 bytes ED A0 80), not collapsed to U+FFFD.
        assert_eq!(
            string(r#""\uD800""#, sp()).unwrap(),
            wtf8::from_utf16(&[0xD800])
        );
        // `\u{D800}` (brace form) likewise.
        assert_eq!(
            string(r#""\u{D800}""#, sp()).unwrap(),
            wtf8::from_utf16(&[0xD800])
        );
        // A lone low surrogate.
        assert_eq!(
            string(r#""\uDC00""#, sp()).unwrap(),
            wtf8::from_utf16(&[0xDC00])
        );
        // High + low across two escapes pair into the astral scalar.
        assert_eq!(string(r#""😀""#, sp()).unwrap(), "😀".as_bytes());
        // A high surrogate followed by a non-low stays lone, then the next char.
        let mut expected = wtf8::from_utf16(&[0xD800]);
        expected.push(b'x');
        assert_eq!(string(r#""\uD800x""#, sp()).unwrap(), expected);
    }
}
