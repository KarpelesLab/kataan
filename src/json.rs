//! Pure, `alloc`-only `JSON.parse` / `JSON.stringify` over realm values.
//!
//! These are free functions operating on a `Realm` and `NanBox` values so
//! both the tree-walker and the bytecode VM can share one implementation
//! (`ROADMAP.md` stdlib). `stringify` mirrors `JSON.stringify` (dropping
//! `undefined`/functions, rendering non-finite numbers as `null`); `parse` is a
//! recursive-descent reader returning a realm value, or an error message.

use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use crate::realm::Realm;
use alloc::string::String;
use alloc::vec::Vec;

/// Returned when `JSON.stringify` hits a circular structure (the caller throws a
/// `TypeError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Circular;

/// Serializes `v` to a JSON string, or `None` when it has no JSON form
/// (`undefined` or a function — which `JSON.stringify` omits).
#[must_use]
pub fn stringify(realm: &Realm, v: NanBox) -> Option<String> {
    stringify_seen(realm, v, &mut Vec::new()).unwrap_or(None)
}

/// `JSON.stringify` that reports a circular reference as `Err(())` (so the caller
/// can throw the spec's `TypeError`) instead of recursing forever.
///
/// # Errors
/// `Err(())` when `v` contains a cycle.
pub fn try_stringify(realm: &Realm, v: NanBox) -> Result<Option<String>, Circular> {
    stringify_seen(realm, v, &mut Vec::new())
}

/// Like [`stringify`], but tracking the ancestor handles in `seen` so a cycle is
/// detected (returning `Err(())`) rather than overflowing the stack.
fn stringify_seen(
    realm: &Realm,
    v: NanBox,
    seen: &mut Vec<Handle>,
) -> Result<Option<String>, Circular> {
    match v.unpack() {
        Unpacked::Undefined => Ok(None),
        Unpacked::Null => Ok(Some(String::from("null"))),
        Unpacked::Bool(b) => Ok(Some(String::from(if b { "true" } else { "false" }))),
        // Spec `ToString` (`0` for `-0`, exponential for ≥ 1e21).
        Unpacked::Number(n) => Ok(Some(if n.is_finite() {
            realm.to_display_string(v)
        } else {
            String::from("null")
        })),
        Unpacked::Handle(raw) => {
            let h = Handle::from_raw(raw);
            if let Some(bytes) = realm.string_bytes(h) {
                return Ok(Some(quote_wtf8(&bytes)));
            }
            // A bytecode-VM closure is a tagged array but a function — omitted.
            if realm.is_vm_function(h) {
                return Ok(None);
            }
            let is_container = realm.array_elements(h).is_some() || realm.object_keys(h).is_some();
            if is_container {
                // A repeated handle is a cycle; depth past the cap is treated the
                // same (the caller throws) so a deep acyclic structure cannot
                // overflow the native stack.
                if seen.contains(&h) || seen.len() >= realm.limits.max_json_depth {
                    return Err(Circular); // circular or too-deeply-nested structure
                }
                seen.push(h);
            }
            let result = if let Some(elems) = realm.array_elements(h).map(<[_]>::to_vec) {
                let mut parts = Vec::with_capacity(elems.len());
                for e in &elems {
                    parts.push(
                        stringify_seen(realm, *e, seen)?.unwrap_or_else(|| String::from("null")),
                    );
                }
                Some(alloc::format!("[{}]", parts.join(",")))
            } else if let Some(keys) = realm.object_keys(h) {
                let mut parts = Vec::new();
                for k in keys {
                    let val = realm.get_property(h, &k).unwrap_or(NanBox::undefined());
                    if let Some(s) = stringify_seen(realm, val, seen)? {
                        parts.push(alloc::format!("{}:{}", quote(&k), s));
                    }
                }
                Some(alloc::format!("{{{}}}", parts.join(",")))
            } else {
                None // a function
            };
            if is_container {
                seen.pop();
            }
            Ok(result)
        }
    }
}

/// Serializes `v` with `indent` (the `JSON.stringify` `space` argument) applied
/// per nesting level — newlines and indentation between members.
#[must_use]
pub fn stringify_pretty(realm: &Realm, v: NanBox, indent: &str) -> Option<String> {
    stringify_at(realm, v, indent, "", &mut Vec::new()).unwrap_or(None)
}

/// Cycle-checked variant of [`stringify_pretty`] (see [`try_stringify`]).
///
/// # Errors
/// `Err(())` when `v` contains a cycle.
pub fn try_stringify_pretty(
    realm: &Realm,
    v: NanBox,
    indent: &str,
) -> Result<Option<String>, Circular> {
    stringify_at(realm, v, indent, "", &mut Vec::new())
}

fn stringify_at(
    realm: &Realm,
    v: NanBox,
    indent: &str,
    cur: &str,
    seen: &mut Vec<Handle>,
) -> Result<Option<String>, Circular> {
    match v.unpack() {
        Unpacked::Handle(raw) => {
            let h = Handle::from_raw(raw);
            if let Some(bytes) = realm.string_bytes(h) {
                return Ok(Some(quote_wtf8(&bytes)));
            }
            // A bytecode-VM closure is a tagged array but a function — omitted.
            if realm.is_vm_function(h) {
                return Ok(None);
            }
            let inner = alloc::format!("{cur}{indent}");
            let is_container = realm.array_elements(h).is_some() || realm.object_keys(h).is_some();
            if is_container {
                if seen.contains(&h) || seen.len() >= realm.limits.max_json_depth {
                    return Err(Circular); // circular or too-deeply-nested structure
                }
                seen.push(h);
            }
            let result = if let Some(elems) = realm.array_elements(h).map(<[_]>::to_vec) {
                if elems.is_empty() {
                    Some(String::from("[]"))
                } else {
                    let mut parts = Vec::with_capacity(elems.len());
                    for e in &elems {
                        let s = stringify_at(realm, *e, indent, &inner, seen)?
                            .unwrap_or_else(|| String::from("null"));
                        parts.push(alloc::format!("{inner}{s}"));
                    }
                    Some(alloc::format!("[\n{}\n{cur}]", parts.join(",\n")))
                }
            } else if let Some(keys) = realm.object_keys(h) {
                let mut parts = Vec::new();
                for k in keys {
                    let val = realm.get_property(h, &k).unwrap_or(NanBox::undefined());
                    if let Some(s) = stringify_at(realm, val, indent, &inner, seen)? {
                        parts.push(alloc::format!("{inner}{}: {}", quote(&k), s));
                    }
                }
                if parts.is_empty() {
                    Some(String::from("{}"))
                } else {
                    Some(alloc::format!("{{\n{}\n{cur}}}", parts.join(",\n")))
                }
            } else {
                None
            };
            if is_container {
                seen.pop();
            }
            Ok(result)
        }
        // Primitives render the same with or without indentation.
        _ => stringify_seen(realm, v, seen),
    }
}

/// Quotes a string as a JSON string literal.
///
/// Iterates UTF-16 code units so a **lone surrogate** is escaped as `\uXXXX`
/// (per the spec, `JSON.stringify` emits well-formed JSON for any DOMString); a
/// valid surrogate pair emits the astral character directly. A non-surrogate
/// string is unchanged.
#[must_use]
pub fn quote(s: &str) -> String {
    quote_wtf8(s.as_bytes())
}

/// Like [`quote`] but over raw **WTF-8 bytes**, so lone surrogates round-trip
/// (escaped as `\uXXXX`). This is the form `JSON.stringify` uses for string
/// values; [`quote`] is the `&str` convenience wrapper.
///
/// Iterates code points: a scalar (BMP or astral) emits its character; a **lone
/// surrogate** is escaped as `\uXXXX` (valid pairs are already scalars here, so
/// they emit the astral character directly, matching the spec).
#[must_use]
pub fn quote_wtf8(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 2);
    out.push('"');
    for cp in crate::wtf8::code_points(bytes) {
        match cp {
            0x22 => out.push_str("\\\""),
            0x5C => out.push_str("\\\\"),
            0x0A => out.push_str("\\n"),
            0x0D => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            // C0 controls and lone surrogates → `\uXXXX`.
            cp if cp < 0x20 || crate::wtf8::is_surrogate(cp) => {
                out.push_str(&alloc::format!("\\u{cp:04x}"));
            }
            // A scalar value (BMP or astral): emit the character.
            cp => {
                if let Some(c) = char::from_u32(cp) {
                    out.push(c);
                }
            }
        }
    }
    out.push('"');
    out
}

/// Parses `src` as JSON into a realm value.
///
/// # Errors
/// Returns a message describing the first syntax error.
pub fn parse(realm: &mut Realm, src: &str) -> Result<NanBox, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut pos = 0;
    let v = parse_value(realm, &chars, &mut pos, 0)?;
    skip_ws(&chars, &mut pos);
    if pos != chars.len() {
        return Err(String::from("Unexpected trailing characters in JSON"));
    }
    Ok(v)
}

fn parse_value(
    realm: &mut Realm,
    c: &[char],
    pos: &mut usize,
    depth: usize,
) -> Result<NanBox, String> {
    skip_ws(c, pos);
    let Some(&ch) = c.get(*pos) else {
        return Err(String::from("Unexpected end of JSON input"));
    };
    if matches!(ch, '[' | '{') && depth >= realm.limits.max_json_depth {
        return Err(String::from("Maximum JSON nesting depth exceeded"));
    }
    match ch {
        'n' => lit(c, pos, "null", NanBox::null()),
        't' => lit(c, pos, "true", NanBox::boolean(true)),
        'f' => lit(c, pos, "false", NanBox::boolean(false)),
        '"' => {
            let s = parse_string(c, pos)?;
            Ok(NanBox::handle(realm.new_string_wtf8(s).to_raw()))
        }
        '[' => {
            *pos += 1;
            let mut elems = Vec::new();
            skip_ws(c, pos);
            if c.get(*pos) == Some(&']') {
                *pos += 1;
                return Ok(NanBox::handle(realm.new_array(elems).to_raw()));
            }
            loop {
                let v = parse_value(realm, c, pos, depth + 1)?;
                elems.push(v);
                skip_ws(c, pos);
                match c.get(*pos) {
                    Some(',') => *pos += 1,
                    Some(']') => {
                        *pos += 1;
                        break;
                    }
                    _ => return Err(String::from("Expected ',' or ']' in JSON")),
                }
            }
            Ok(NanBox::handle(realm.new_array(elems).to_raw()))
        }
        '{' => {
            *pos += 1;
            let obj = realm.new_object();
            skip_ws(c, pos);
            if c.get(*pos) == Some(&'}') {
                *pos += 1;
                return Ok(NanBox::handle(obj.to_raw()));
            }
            loop {
                skip_ws(c, pos);
                if c.get(*pos) != Some(&'"') {
                    return Err(String::from("Expected property name in JSON"));
                }
                // Property keys live in the `&str`-keyed object layer; a lone
                // surrogate in a *key* (an exotic edge) is decoded lossily.
                let key = crate::wtf8::to_string_lossy(&parse_string(c, pos)?);
                skip_ws(c, pos);
                if c.get(*pos) != Some(&':') {
                    return Err(String::from("Expected ':' in JSON"));
                }
                *pos += 1;
                let v = parse_value(realm, c, pos, depth + 1)?;
                realm.set_property(obj, &key, v);
                skip_ws(c, pos);
                match c.get(*pos) {
                    Some(',') => *pos += 1,
                    Some('}') => {
                        *pos += 1;
                        break;
                    }
                    _ => return Err(String::from("Expected ',' or '}' in JSON")),
                }
            }
            Ok(NanBox::handle(obj.to_raw()))
        }
        '-' | '0'..='9' => {
            // The JSON `Number` production, exactly:
            //   `-`? ( `0` | [1-9][0-9]* ) ( `.` [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
            // A loose digit run handed to Rust's `f64` parser would accept forms JSON
            // forbids — a leading zero (`00`, `013`), a bare trailing point (`1.`) —
            // all of which must be SyntaxErrors.
            let start = *pos;
            let digit = |c: &[char], p: usize| c.get(p).is_some_and(char::is_ascii_digit);
            if c.get(*pos) == Some(&'-') {
                *pos += 1;
            }
            if c.get(*pos) == Some(&'0') {
                *pos += 1;
            } else if digit(c, *pos) {
                while digit(c, *pos) {
                    *pos += 1;
                }
            } else {
                return Err(String::from("Invalid number in JSON"));
            }
            if c.get(*pos) == Some(&'.') {
                *pos += 1;
                if !digit(c, *pos) {
                    return Err(String::from("Invalid number in JSON"));
                }
                while digit(c, *pos) {
                    *pos += 1;
                }
            }
            if matches!(c.get(*pos), Some('e' | 'E')) {
                *pos += 1;
                if matches!(c.get(*pos), Some('+' | '-')) {
                    *pos += 1;
                }
                if !digit(c, *pos) {
                    return Err(String::from("Invalid number in JSON"));
                }
                while digit(c, *pos) {
                    *pos += 1;
                }
            }
            let text: String = c[start..*pos].iter().collect();
            text.parse::<f64>()
                .map(NanBox::number)
                .map_err(|_| String::from("Invalid number in JSON"))
        }
        _ => Err(String::from("Unexpected token in JSON")),
    }
}

fn lit(c: &[char], pos: &mut usize, word: &str, value: NanBox) -> Result<NanBox, String> {
    if c[*pos..].iter().take(word.len()).copied().eq(word.chars()) {
        *pos += word.len();
        Ok(value)
    } else {
        Err(String::from("Unexpected token in JSON"))
    }
}

/// Parses a JSON string literal into **WTF-8 bytes**, preserving lone surrogates
/// (a `\uXXXX` for a surrogate code point that has no valid partner is kept as a
/// surrogate, not collapsed to U+FFFD). Adjacent `\uXXXX\uXXXX` halves of a valid
/// pair combine into the astral scalar. A non-surrogate string is byte-identical
/// to its UTF-8.
fn parse_string(c: &[char], pos: &mut usize) -> Result<Vec<u8>, String> {
    *pos += 1; // opening quote
    let mut out: Vec<u8> = Vec::new();
    let push_char = |out: &mut Vec<u8>, ch: char| {
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    };
    loop {
        match c.get(*pos) {
            None => return Err(String::from("Unterminated string in JSON")),
            Some('"') => {
                *pos += 1;
                return Ok(out);
            }
            Some('\\') => {
                *pos += 1;
                match c.get(*pos) {
                    Some('"') => out.push(b'"'),
                    Some('\\') => out.push(b'\\'),
                    Some('/') => out.push(b'/'),
                    Some('n') => out.push(b'\n'),
                    Some('r') => out.push(b'\r'),
                    Some('t') => out.push(b'\t'),
                    Some('b') => out.push(0x08),
                    Some('f') => out.push(0x0C),
                    Some('u') => {
                        let hi = parse_hex4(c, *pos + 1)?;
                        *pos += 4;
                        // A high surrogate may be followed by `\uXXXX` low half.
                        if (0xD800..=0xDBFF).contains(&hi)
                            && c.get(*pos + 1) == Some(&'\\')
                            && c.get(*pos + 2) == Some(&'u')
                            && let Ok(lo) = parse_hex4(c, *pos + 3)
                            && (0xDC00..=0xDFFF).contains(&lo)
                        {
                            let cp = 0x1_0000
                                + ((u32::from(hi) - 0xD800) << 10)
                                + (u32::from(lo) - 0xDC00);
                            crate::wtf8::encode_code_point(cp, &mut out);
                            *pos += 6;
                        } else {
                            // A BMP scalar, or a lone surrogate kept as-is.
                            crate::wtf8::encode_utf16_unit(hi, &mut out);
                        }
                    }
                    _ => return Err(String::from("Invalid escape in JSON")),
                }
                *pos += 1;
            }
            Some(&ch) => {
                push_char(&mut out, ch);
                *pos += 1;
            }
        }
    }
}

/// Reads exactly four hex digits starting at `at` into a `u16`.
fn parse_hex4(c: &[char], at: usize) -> Result<u16, String> {
    let hex: String = c.get(at..at + 4).unwrap_or(&[]).iter().collect();
    if hex.len() == 4 {
        u16::from_str_radix(&hex, 16).map_err(|_| String::from("Invalid \\u escape in JSON"))
    } else {
        Err(String::from("Invalid \\u escape in JSON"))
    }
}

fn skip_ws(c: &[char], pos: &mut usize) {
    while c
        .get(*pos)
        .is_some_and(|ch| matches!(ch, ' ' | '\t' | '\n' | '\r'))
    {
        *pos += 1;
    }
}
