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
            if let Some(s) = realm.string_value(h) {
                return Ok(Some(quote(&s)));
            }
            // A bytecode-VM closure is a tagged array but a function — omitted.
            if realm.is_vm_function(h) {
                return Ok(None);
            }
            let is_container = realm.array_elements(h).is_some() || realm.object_keys(h).is_some();
            if is_container {
                if seen.contains(&h) {
                    return Err(Circular); // circular structure
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
            if let Some(s) = realm.string_value(h) {
                return Ok(Some(quote(&s)));
            }
            // A bytecode-VM closure is a tagged array but a function — omitted.
            if realm.is_vm_function(h) {
                return Ok(None);
            }
            let inner = alloc::format!("{cur}{indent}");
            let is_container = realm.array_elements(h).is_some() || realm.object_keys(h).is_some();
            if is_container {
                if seen.contains(&h) {
                    return Err(Circular); // circular structure
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
#[must_use]
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&alloc::format!("\\u{:04x}", c as u32)),
            c => out.push(c),
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
    let v = parse_value(realm, &chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos != chars.len() {
        return Err(String::from("Unexpected trailing characters in JSON"));
    }
    Ok(v)
}

fn parse_value(realm: &mut Realm, c: &[char], pos: &mut usize) -> Result<NanBox, String> {
    skip_ws(c, pos);
    let Some(&ch) = c.get(*pos) else {
        return Err(String::from("Unexpected end of JSON input"));
    };
    match ch {
        'n' => lit(c, pos, "null", NanBox::null()),
        't' => lit(c, pos, "true", NanBox::boolean(true)),
        'f' => lit(c, pos, "false", NanBox::boolean(false)),
        '"' => {
            let s = parse_string(c, pos)?;
            Ok(NanBox::handle(realm.new_string(&s).to_raw()))
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
                let v = parse_value(realm, c, pos)?;
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
                let key = parse_string(c, pos)?;
                skip_ws(c, pos);
                if c.get(*pos) != Some(&':') {
                    return Err(String::from("Expected ':' in JSON"));
                }
                *pos += 1;
                let v = parse_value(realm, c, pos)?;
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
            let start = *pos;
            if c.get(*pos) == Some(&'-') {
                *pos += 1;
            }
            while c
                .get(*pos)
                .is_some_and(|d| d.is_ascii_digit() || matches!(d, '.' | 'e' | 'E' | '+' | '-'))
            {
                *pos += 1;
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

fn parse_string(c: &[char], pos: &mut usize) -> Result<String, String> {
    *pos += 1; // opening quote
    let mut out = String::new();
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
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('b') => out.push('\u{0008}'),
                    Some('f') => out.push('\u{000c}'),
                    Some('u') => {
                        let hex: String = c.get(*pos + 1..*pos + 5).unwrap_or(&[]).iter().collect();
                        let code = u32::from_str_radix(&hex, 16)
                            .ok()
                            .and_then(char::from_u32)
                            .ok_or_else(|| String::from("Invalid \\u escape in JSON"))?;
                        out.push(code);
                        *pos += 4;
                    }
                    _ => return Err(String::from("Invalid escape in JSON")),
                }
                *pos += 1;
            }
            Some(&ch) => {
                out.push(ch);
                *pos += 1;
            }
        }
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
