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

/// Serializes `v` to a JSON string, or `None` when it has no JSON form
/// (`undefined` or a function — which `JSON.stringify` omits).
#[must_use]
pub fn stringify(realm: &Realm, v: NanBox) -> Option<String> {
    match v.unpack() {
        Unpacked::Undefined => None,
        Unpacked::Null => Some(String::from("null")),
        Unpacked::Bool(b) => Some(String::from(if b { "true" } else { "false" })),
        Unpacked::Number(n) => Some(if n.is_finite() {
            alloc::format!("{n}")
        } else {
            String::from("null")
        }),
        Unpacked::Handle(raw) => {
            let h = Handle::from_raw(raw);
            if let Some(s) = realm.string_value(h) {
                return Some(quote(&s));
            }
            if let Some(elems) = realm.array_elements(h).map(<[_]>::to_vec) {
                let parts: Vec<String> = elems
                    .iter()
                    .map(|e| stringify(realm, *e).unwrap_or_else(|| String::from("null")))
                    .collect();
                return Some(alloc::format!("[{}]", parts.join(",")));
            }
            if let Some(keys) = realm.object_keys(h) {
                let mut parts = Vec::new();
                for k in keys {
                    let val = realm.get_property(h, &k).unwrap_or(NanBox::undefined());
                    if let Some(s) = stringify(realm, val) {
                        parts.push(alloc::format!("{}:{}", quote(&k), s));
                    }
                }
                return Some(alloc::format!("{{{}}}", parts.join(",")));
            }
            None // a function
        }
    }
}

/// Serializes `v` with `indent` (the `JSON.stringify` `space` argument) applied
/// per nesting level — newlines and indentation between members.
#[must_use]
pub fn stringify_pretty(realm: &Realm, v: NanBox, indent: &str) -> Option<String> {
    stringify_at(realm, v, indent, "")
}

fn stringify_at(realm: &Realm, v: NanBox, indent: &str, cur: &str) -> Option<String> {
    match v.unpack() {
        Unpacked::Handle(raw) => {
            let h = Handle::from_raw(raw);
            if let Some(s) = realm.string_value(h) {
                return Some(quote(&s));
            }
            let inner = alloc::format!("{cur}{indent}");
            if let Some(elems) = realm.array_elements(h).map(<[_]>::to_vec) {
                if elems.is_empty() {
                    return Some(String::from("[]"));
                }
                let parts: Vec<String> = elems
                    .iter()
                    .map(|e| {
                        let s = stringify_at(realm, *e, indent, &inner)
                            .unwrap_or_else(|| String::from("null"));
                        alloc::format!("{inner}{s}")
                    })
                    .collect();
                return Some(alloc::format!("[\n{}\n{cur}]", parts.join(",\n")));
            }
            if let Some(keys) = realm.object_keys(h) {
                let mut parts = Vec::new();
                for k in keys {
                    let val = realm.get_property(h, &k).unwrap_or(NanBox::undefined());
                    if let Some(s) = stringify_at(realm, val, indent, &inner) {
                        parts.push(alloc::format!("{inner}{}: {}", quote(&k), s));
                    }
                }
                if parts.is_empty() {
                    return Some(String::from("{}"));
                }
                return Some(alloc::format!("{{\n{}\n{cur}}}", parts.join(",\n")));
            }
            None
        }
        // Primitives render the same with or without indentation.
        _ => stringify(realm, v),
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
