//! `JSON.parse` without a reviver: a single pass over the text's UTF-8 bytes.
//!
//! It accepts exactly the grammar (and reports the same errors) as the
//! reviver-aware parser in `json.rs`, but borrows unescaped strings and keys
//! straight from the input, reads integers without a temporary string, and
//! adds each object's properties through transition caches keyed by the
//! property's nesting depth and position, so a run of records with the same
//! keys builds every one of them along one shape path with no name lookups.

use super::*;
use crate::ic::PropertyCache;
use alloc::boxed::Box;

/// Nesting depths and property positions that get a transition cache; deeper
/// or later properties use the ordinary add.
const CACHED_DEPTHS: usize = 16;
const CACHED_POSITIONS: usize = 32;

/// The JSON string escape byte.
const BACKSLASH: u8 = 0x5C;

/// Per-parse state: the transition caches, each remembering the key it was
/// armed for (a cache is only ever offered the key it learned).
struct JsonParser<'a> {
    text: &'a str,
    src: &'a [u8],
    pos: usize,
    caches: Vec<Option<(Box<str>, PropertyCache)>>,
    /// Scratch for strings with escapes.
    buf: Vec<u8>,
}

impl JsonParser<'_> {
    fn skip_ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.src.get(self.pos) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    /// When the input at `pos` is the quoted key that the transition cache
    /// of (`depth`, `ordinal`) was armed for, written without escapes (the
    /// key holds no quote, backslash or control character, so the token can
    /// only be that key): consumes it and returns the cache's index.
    fn cached_key_at(&mut self, depth: usize, ordinal: usize) -> Option<usize> {
        if depth >= CACHED_DEPTHS || ordinal >= CACHED_POSITIONS {
            return None;
        }
        let idx = depth * CACHED_POSITIONS + ordinal;
        let (k, _) = self.caches.get(idx)?.as_ref()?;
        let k = k.as_bytes();
        let rest = self.src.get(self.pos..)?;
        if rest.len() < k.len() + 2
            || rest[0] != b'"'
            || rest[k.len() + 1] != b'"'
            || &rest[1..=k.len()] != k
            || k.iter().any(|&b| b == b'"' || b == BACKSLASH || b < 0x20)
        {
            return None;
        }
        self.pos += k.len() + 2;
        Some(idx)
    }

    /// Four hex digits at `at` as a code unit.
    fn hex4(&self, at: usize) -> Option<u16> {
        let digits = self.src.get(at..at + 4)?;
        let mut v: u16 = 0;
        for &d in digits {
            v = (v << 4) | u16::from((d as char).to_digit(16)? as u8);
        }
        Some(v)
    }
}

/// A container [`Interp::json_fast_value`] has open: an array's elements so
/// far, or an object with the count of properties added and the key whose
/// value is being parsed.
enum Open {
    Array(Vec<NanBox>),
    Object {
        obj: Handle,
        ordinal: usize,
        key: JsonKey,
    },
}

/// A property name: the key a transition cache was armed for (by the
/// cache's index), a range of the input (no escapes), or decoded.
enum JsonKey {
    Cached(usize),
    Borrowed(usize, usize),
    Owned(alloc::string::String),
}

/// Where a string token ended up: borrowed from the input (no escapes) or
/// decoded into the parser's scratch buffer (WTF-8).
enum JsonStr {
    Borrowed(usize, usize),
    Scratch,
}

impl Interp {
    /// `JSON.parse(text)` with no reviver: the whole text must be one JSON
    /// value plus optional whitespace.
    pub(crate) fn json_parse_text(&mut self, text: &str) -> Result<NanBox, ExecError> {
        let mut p = JsonParser {
            text,
            src: text.as_bytes(),
            pos: 0,
            caches: Vec::new(),
            buf: Vec::new(),
        };
        let value = self.json_fast_value(&mut p)?;
        p.skip_ws();
        if p.pos != p.src.len() {
            return Err(self.json_error("Unexpected token in JSON"));
        }
        Ok(value)
    }

    /// One JSON value at the cursor. Iterative: the open arrays and objects
    /// live on an explicit stack rather than the native one, so nesting is
    /// bounded by memory, not by stack (`JSON.parse("[".repeat(1e6) + …)`
    /// works). The finished value is heap cells, which the collector marks
    /// from a worklist, so nothing about it recurses either.
    fn json_fast_value(&mut self, p: &mut JsonParser<'_>) -> Result<NanBox, ExecError> {
        let mut open: Vec<Open> = Vec::new();
        'value: loop {
            p.skip_ws();
            let Some(ch) = p.peek() else {
                return Err(self.json_error("Unexpected end of JSON input"));
            };
            let mut v = match ch {
                b'n' => self.json_fast_lit(p, b"null", NanBox::null())?,
                b't' => self.json_fast_lit(p, b"true", NanBox::boolean(true))?,
                b'f' => self.json_fast_lit(p, b"false", NanBox::boolean(false))?,
                b'"' => match self.json_fast_string(p)? {
                    // A one-byte (ASCII) string is the realm's shared unit
                    // string (strings are values: sharing is invisible).
                    JsonStr::Borrowed(a, b) if b == a + 1 => {
                        NanBox::handle(self.realm.unit_string(u16::from(p.src[a])).to_raw())
                    }
                    JsonStr::Borrowed(a, b) => {
                        // The input is a `&str`, and an unescaped run between
                        // quotes starts and ends on character boundaries.
                        let text = p.text.get(a..b).unwrap_or_default();
                        NanBox::handle(self.realm.new_string(text).to_raw())
                    }
                    JsonStr::Scratch => {
                        let bytes = p.buf.clone();
                        self.new_str_bytes(bytes)
                    }
                },
                b'[' => {
                    p.pos += 1;
                    p.skip_ws();
                    if p.peek() == Some(b']') {
                        p.pos += 1;
                        NanBox::handle(self.realm.new_array(Vec::new()).to_raw())
                    } else {
                        open.push(Open::Array(Vec::new()));
                        continue 'value;
                    }
                }
                b'{' => {
                    p.pos += 1;
                    let obj = self.realm.new_object();
                    p.skip_ws();
                    if p.peek() == Some(b'}') {
                        p.pos += 1;
                        NanBox::handle(obj.to_raw())
                    } else {
                        let key = self.json_fast_key(p, open.len(), 0)?;
                        open.push(Open::Object {
                            obj,
                            ordinal: 0,
                            key,
                        });
                        continue 'value;
                    }
                }
                b'-' | b'0'..=b'9' => self.json_fast_number(p)?,
                _ => return Err(self.json_error("Unexpected token in JSON")),
            };
            // `v` is complete: hand it to the innermost open container, and
            // close every container it completes.
            loop {
                let depth = open.len().saturating_sub(1);
                match open.last_mut() {
                    None => return Ok(v),
                    Some(Open::Array(elems)) => {
                        elems.push(v);
                        p.skip_ws();
                        match p.peek() {
                            Some(b',') => {
                                p.pos += 1;
                                continue 'value;
                            }
                            Some(b']') => {
                                p.pos += 1;
                                let Some(Open::Array(elems)) = open.pop() else {
                                    unreachable!("the innermost container is this array");
                                };
                                v = NanBox::handle(self.realm.new_array(elems).to_raw());
                            }
                            _ => return Err(self.json_error("Expected ',' or ']'")),
                        }
                    }
                    Some(Open::Object { obj, ordinal, key }) => {
                        let (obj, n) = (*obj, *ordinal);
                        *ordinal += 1;
                        let text = p.text;
                        match key {
                            JsonKey::Cached(idx) => {
                                let added = match &mut p.caches[*idx] {
                                    Some((k, cache)) => {
                                        self.realm.init_data_property_cached(obj, k, v, cache)
                                    }
                                    None => false,
                                };
                                if !added && let Some((k, _)) = &p.caches[*idx] {
                                    self.realm.set_property(obj, k, v);
                                }
                            }
                            JsonKey::Borrowed(a, b) => {
                                let name = text.get(*a..*b).unwrap_or_default();
                                self.json_fast_add(p, obj, depth, n, name, v);
                            }
                            JsonKey::Owned(name) => self.json_fast_add(p, obj, depth, n, name, v),
                        }
                        p.skip_ws();
                        match p.peek() {
                            Some(b',') => {
                                p.pos += 1;
                                let next = self.json_fast_key(p, depth, n + 1)?;
                                if let Some(Open::Object { key, .. }) = open.last_mut() {
                                    *key = next;
                                }
                                continue 'value;
                            }
                            Some(b'}') => {
                                p.pos += 1;
                                open.pop();
                                v = NanBox::handle(obj.to_raw());
                            }
                            _ => return Err(self.json_error("Expected ',' or '}'")),
                        }
                    }
                }
            }
        }
    }

    /// The name of property `ordinal` of an object at nesting `depth`, and its
    /// `:`, the cursor at the whitespace before the name. Keys live in the
    /// `&str`-keyed object layer; a lone surrogate in a *key* (an exotic edge)
    /// decodes lossily.
    fn json_fast_key(
        &mut self,
        p: &mut JsonParser<'_>,
        depth: usize,
        ordinal: usize,
    ) -> Result<JsonKey, ExecError> {
        p.skip_ws();
        // The key this position's transition cache was armed for, when the
        // input spells it out verbatim (no escapes): then the property name
        // needs no scan and no compare.
        let key = if let Some(idx) = p.cached_key_at(depth, ordinal) {
            JsonKey::Cached(idx)
        } else {
            if p.peek() != Some(b'"') {
                return Err(self.json_error("Expected property name"));
            }
            match self.json_fast_string(p)? {
                JsonStr::Borrowed(a, b) => JsonKey::Borrowed(a, b),
                // (An unescaped run cannot hold a raw NUL; an escaped
                // `\u0000...` key is escaped into the user namespace, see
                // `realm::is_internal_key`.)
                JsonStr::Scratch => JsonKey::Owned(crate::realm::user_key_owned(
                    crate::wtf8::to_string_lossy(&p.buf),
                )),
            }
        };
        p.skip_ws();
        if p.peek() != Some(b':') {
            return Err(self.json_error("Expected ':'"));
        }
        p.pos += 1;
        Ok(key)
    }

    /// `CreateDataProperty(obj, key, v)` on the parser's fresh object, through
    /// the transition cache for (`depth`, `ordinal`) when that cache was armed
    /// for this same key.
    fn json_fast_add(
        &mut self,
        p: &mut JsonParser<'_>,
        obj: Handle,
        depth: usize,
        ordinal: usize,
        key: &str,
        v: NanBox,
    ) {
        if depth < CACHED_DEPTHS && ordinal < CACHED_POSITIONS {
            let idx = depth * CACHED_POSITIONS + ordinal;
            if p.caches.len() <= idx {
                p.caches.resize_with(idx + 1, || None);
            }
            let entry = &mut p.caches[idx];
            if !entry.as_ref().is_some_and(|(k, _)| &**k == key) {
                *entry = Some((Box::from(key), PropertyCache::new()));
            }
            if let Some((_, cache)) = entry
                && self.realm.init_data_property_cached(obj, key, v, cache)
            {
                return;
            }
        }
        self.realm.set_property(obj, key, v);
    }

    fn json_fast_lit(
        &mut self,
        p: &mut JsonParser<'_>,
        word: &[u8],
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        if p.src[p.pos..].starts_with(word) {
            p.pos += word.len();
            Ok(value)
        } else {
            Err(self.json_error("Unexpected token in JSON"))
        }
    }

    /// The JSON `Number` production, exactly:
    /// `-`? ( `0` | [1-9][0-9]* ) ( `.` [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
    fn json_fast_number(&mut self, p: &mut JsonParser<'_>) -> Result<NanBox, ExecError> {
        let start = p.pos;
        let digit = |p: &JsonParser<'_>| p.peek().is_some_and(|b| b.is_ascii_digit());
        let neg = p.peek() == Some(b'-');
        if neg {
            p.pos += 1;
        }
        let int_start = p.pos;
        if p.peek() == Some(b'0') {
            p.pos += 1;
        } else if digit(p) {
            while digit(p) {
                p.pos += 1;
            }
        } else {
            return Err(self.json_error("Invalid number in JSON"));
        }
        let int_end = p.pos;
        let mut simple = true;
        if p.peek() == Some(b'.') {
            simple = false;
            p.pos += 1;
            if !digit(p) {
                return Err(self.json_error("Invalid number in JSON"));
            }
            while digit(p) {
                p.pos += 1;
            }
        }
        if matches!(p.peek(), Some(b'e' | b'E')) {
            simple = false;
            p.pos += 1;
            if matches!(p.peek(), Some(b'+' | b'-')) {
                p.pos += 1;
            }
            if !digit(p) {
                return Err(self.json_error("Invalid number in JSON"));
            }
            while digit(p) {
                p.pos += 1;
            }
        }
        // An integer of at most 15 digits is exact in an f64.
        if simple && int_end - int_start <= 15 {
            let mut u: u64 = 0;
            for &d in &p.src[int_start..int_end] {
                u = u * 10 + u64::from(d - b'0');
            }
            let n = u as f64;
            return Ok(NanBox::number(if neg { -n } else { n }));
        }
        p.text
            .get(start..p.pos)
            .and_then(|t| t.parse::<f64>().ok())
            .map(NanBox::number)
            .ok_or_else(|| self.json_error("Invalid number in JSON"))
    }

    /// Scans a string token at `p.pos` (the opening quote). Without escapes the
    /// result is the input range between the quotes; otherwise it is decoded
    /// into `p.buf` as WTF-8 (a `\uXXXX` high surrogate pairs with a directly
    /// following `\uXXXX` low surrogate; a lone one is kept as a code unit).
    fn json_fast_string(&mut self, p: &mut JsonParser<'_>) -> Result<JsonStr, ExecError> {
        p.pos += 1; // opening quote
        let start = p.pos;
        loop {
            match p.peek() {
                None => return Err(self.json_error("Unterminated string in JSON")),
                Some(b'"') => {
                    p.pos += 1;
                    return Ok(JsonStr::Borrowed(start, p.pos - 1));
                }
                Some(b'\\') => break,
                Some(b) if b < 0x20 => {
                    return Err(self.json_error("Bad control character in string literal in JSON"));
                }
                Some(_) => p.pos += 1,
            }
        }
        p.buf.clear();
        p.buf.extend_from_slice(&p.src[start..p.pos]);
        loop {
            match p.peek() {
                None => return Err(self.json_error("Unterminated string in JSON")),
                Some(b'"') => {
                    p.pos += 1;
                    return Ok(JsonStr::Scratch);
                }
                Some(b'\\') => {
                    p.pos += 1;
                    match p.peek() {
                        Some(b'"') => p.buf.push(b'"'),
                        Some(b'\\') => p.buf.push(b'\\'),
                        Some(b'/') => p.buf.push(b'/'),
                        Some(b'n') => p.buf.push(b'\n'),
                        Some(b't') => p.buf.push(b'\t'),
                        Some(b'r') => p.buf.push(b'\r'),
                        Some(b'b') => p.buf.push(0x08),
                        Some(b'f') => p.buf.push(0x0C),
                        Some(b'u') => {
                            let hi = p
                                .hex4(p.pos + 1)
                                .ok_or_else(|| self.json_error("Invalid \\u escape in JSON"))?;
                            p.pos += 4;
                            if (0xD800..=0xDBFF).contains(&hi)
                                && p.src.get(p.pos + 1) == Some(&b'\\')
                                && p.src.get(p.pos + 2) == Some(&b'u')
                                && let Some(lo) = p.hex4(p.pos + 3)
                                && (0xDC00..=0xDFFF).contains(&lo)
                            {
                                let cp = 0x1_0000
                                    + ((u32::from(hi) - 0xD800) << 10)
                                    + (u32::from(lo) - 0xDC00);
                                crate::wtf8::encode_code_point(cp, &mut p.buf);
                                p.pos += 6;
                            } else {
                                crate::wtf8::encode_utf16_unit(hi, &mut p.buf);
                            }
                        }
                        _ => return Err(self.json_error("Invalid escape in JSON")),
                    }
                    p.pos += 1;
                }
                Some(b) if b < 0x20 => {
                    return Err(self.json_error("Bad control character in string literal in JSON"));
                }
                Some(b) => {
                    p.buf.push(b);
                    p.pos += 1;
                }
            }
        }
    }
}
