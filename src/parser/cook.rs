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

/// Whether a string-literal token's source (quotes included) contains a
/// `LegacyOctalEscapeSequence` (`\1`–`\7`, or `\0` immediately followed by a
/// decimal digit, e.g. `\00`) or a `NonOctalDecimalEscapeSequence` (`\8` /
/// `\9`). Both are accepted in sloppy code (Annex B) but are early Syntax Errors
/// in strict-mode code; the strict check is applied by the validation pass.
///
/// Scanning is byte-wise: a backslash, the escape selector, and the decimal
/// digits are all ASCII, and a UTF-8 continuation byte is always ≥ 0x80, so a
/// multi-byte scalar inside the literal can never be mistaken for an escape.
#[must_use]
pub(super) fn string_has_legacy_octal_escape(raw: &str) -> bool {
    let b = raw.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' {
            i += 1;
            continue;
        }
        // `b[i]` is a backslash; inspect the escaped byte (if any).
        match b.get(i + 1) {
            // `\1`–`\9`: a legacy octal (`1`–`7`) or non-octal decimal (`8`/`9`).
            Some(d @ b'1'..=b'9') => {
                let _ = d;
                return true;
            }
            // `\0` is a legal `NUL` escape only when *not* followed by a digit; a
            // following digit makes it a legacy octal escape (`\00`, `\012`).
            Some(b'0') => {
                if b.get(i + 2).is_some_and(u8::is_ascii_digit) {
                    return true;
                }
                i += 2; // consume `\0`
            }
            // Any other escape (`\x`, `\u`, `\n`, `\\`, `\"`, …): skip the
            // backslash and the escaped byte so an escaped backslash (`\\`) does
            // not let the following char be misread as starting a new escape.
            Some(_) => i += 2,
            None => i += 1,
        }
    }
    false
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
            // Legacy octal escape (Annex B B.1.2). Only reached in sloppy mode — a
            // strict-mode string with such an escape is rejected before cooking. A
            // leading `0`–`3` admits up to three octal digits; a leading `4`–`7`,
            // two (value ≤ 255). `\8` / `\9` are not octal (the identity branch).
            '0'..='7' => {
                let mut val = esc as u32 - '0' as u32;
                let max_more = if esc <= '3' { 2 } else { 1 };
                for _ in 0..max_more {
                    match chars.peek() {
                        Some(&d @ '0'..='7') => {
                            chars.next();
                            val = val * 8 + (d as u32 - '0' as u32);
                        }
                        _ => break,
                    }
                }
                // `val` ≤ 0o377 = 255, so it is a valid Latin-1 code point.
                push_char(&mut out, char::from_u32(val).unwrap_or('\0'));
            }
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

/// Validates the escape sequences of an *untagged* template segment, returning
/// a Syntax Error for any `NotEscapeSequence` (a `\` followed by something that
/// is not a legal `TemplateEscapeSequence`).
///
/// Per the grammar (`sec-template-literal-lexical-components`), a template
/// segment admits only:
///   - `CharacterEscapeSequence` (`\n`, `\\`, `\` + any non-escape char, line
///     continuations) — always valid,
///   - `\0` provided it is **not** followed by a decimal digit,
///   - `HexEscapeSequence` `\xHH` with exactly two hex digits,
///   - `UnicodeEscapeSequence` `\uHHHH` or `\u{ CodePoint }` (≤ U+10FFFF, no
///     numeric separators).
///
/// In particular — and unlike a sloppy-mode string, where Annex B allows them —
/// a legacy octal escape (`\1`–`\7`, `\00`, …) and `\8` / `\9` are *never* legal
/// in a template and so are rejected here regardless of strict mode. (Tagged
/// templates skip this check: their cooked value is simply `undefined`.)
pub(super) fn validate_template_escapes(body: &str, span: Span) -> Result<()> {
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            continue;
        }
        let Some(esc) = chars.next() else {
            // A trailing `\` cannot occur: the lexer requires the closing
            // delimiter, so a `\` is always followed by at least one char.
            return Err(Error::syntax("invalid template escape sequence", span));
        };
        match esc {
            // `\0` is only an escape when not followed by a decimal digit; a
            // following digit makes it a legacy octal escape (`\00`), which is
            // not a TemplateEscapeSequence.
            '0' => {
                if chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return Err(Error::syntax(
                        "octal escape sequences are not allowed in template literals",
                        span,
                    ));
                }
            }
            // Legacy octal (`\1`–`\7`) and the non-octal decimals `\8` / `\9`.
            '1'..='9' => {
                return Err(Error::syntax(
                    "octal escape sequences are not allowed in template literals",
                    span,
                ));
            }
            'x' => {
                if !next_is_hex(&mut chars) || !next_is_hex(&mut chars) {
                    return Err(Error::syntax(
                        "invalid hexadecimal escape sequence in template literal",
                        span,
                    ));
                }
            }
            'u' => validate_template_unicode_escape(&mut chars, span)?,
            // Any other escaped char (incl. line continuations) is valid.
            _ => {}
        }
    }
    Ok(())
}

/// Validates a `\u` escape body in a template (the `u` already consumed):
/// either `\uHHHH` or `\u{ CodePoint }` with `CodePoint` ≤ U+10FFFF and no
/// numeric separators.
fn validate_template_unicode_escape(
    chars: &mut core::iter::Peekable<core::str::Chars<'_>>,
    span: Span,
) -> Result<()> {
    let invalid = || Error::syntax("invalid Unicode escape sequence in template literal", span);
    if chars.peek() == Some(&'{') {
        chars.next(); // `{`
        let mut value: u32 = 0;
        let mut any = false;
        loop {
            match chars.peek() {
                Some('}') => break,
                Some(&d) if d.is_ascii_hexdigit() => {
                    any = true;
                    value = value
                        .saturating_mul(16)
                        .saturating_add(d.to_digit(16).unwrap_or(0));
                    chars.next();
                }
                // A numeric separator (`_`) or any other char is invalid.
                _ => return Err(invalid()),
            }
        }
        chars.next(); // `}`
        if !any || value > 0x10_FFFF {
            return Err(invalid());
        }
    } else {
        for _ in 0..4 {
            if !next_is_hex(chars) {
                return Err(invalid());
            }
        }
    }
    Ok(())
}

/// Consumes one char if it is an ASCII hex digit, reporting whether it was.
fn next_is_hex(chars: &mut core::iter::Peekable<core::str::Chars<'_>>) -> bool {
    if chars.peek().is_some_and(char::is_ascii_hexdigit) {
        chars.next();
        true
    } else {
        false
    }
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
    fn template_escape_validation() {
        // Valid TemplateEscapeSequences.
        for ok in [
            "",
            "abc",
            r"\n\t\\",
            r"\x41",
            r"A",
            r"\u{1F600}",
            r"\u{D800}",
            r"\0",
            r"\w",
        ] {
            assert!(
                validate_template_escapes(ok, sp()).is_ok(),
                "should accept: {ok:?}"
            );
        }
        // NotEscapeSequences — early errors in an untagged template.
        for bad in [
            r"\x0",        // truncated \x
            r"\xZZ",       // non-hex \x
            r"\u0",        // truncated \u
            r"\u",         // \u with no digits
            r"\u{}",       // empty code point
            r"\u{110000}", // out of range
            r"\u{1F_639}", // separator in \u{}
            r"\00",        // legacy octal
            r"\1",         // legacy octal
            r"\8",         // \8
            r"\9",         // \9
        ] {
            assert!(
                validate_template_escapes(bad, sp()).is_err(),
                "should reject: {bad:?}"
            );
        }
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
