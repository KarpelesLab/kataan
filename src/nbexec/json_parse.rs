//! `JSON.parse` without a reviver: a single pass over the text's WTF-8 bytes.
//!
//! It accepts exactly the grammar (and reports the same errors) as the
//! reviver-aware parser in `json.rs`. The text is read as the string's own
//! bytes (a flat string is not copied, and a lone surrogate in it survives).
//! Unescaped strings and keys are taken straight from the input; integers and
//! short decimals are read without a temporary string.
//!
//! Containers are built bottom-up: the values of every open array and object
//! wait on one shared value stack, and a container is made in one step when it
//! closes. An object's keys are matched, as they are read, against the
//! *template* recorded for the object's position in the document (its path
//! of property ordinals): a run of records with the same keys then compares
//! each key's bytes with the template's and makes each record directly in the
//! template's final shape with its slots filled, with no name lookups and no
//! transitions. An object that does not match is built the ordinary way, one
//! `CreateDataProperty` at a time through transition caches keyed by nesting
//! depth and position, and may become its position's new template.

use super::*;
use crate::ic::PropertyCache;
use crate::rope::Rope;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::Rc;

/// Nesting depths and property positions that get a transition cache; deeper
/// or later properties use the ordinary add.
const CACHED_DEPTHS: usize = 16;
const CACHED_POSITIONS: usize = 32;

/// The number of template slots; an object's slot is a hash of its path.
const TEMPLATE_SITES: usize = 256;

/// The misses in a row after which a site's template is replaced by the
/// shape of the object that missed (one stray record keeps the template).
const TEMPLATE_REPLACE_MISSES: u8 = 2;

/// The JSON string escape byte.
const BACKSLASH: u8 = 0x5C;

/// The keys, in order, of the objects one document position has been
/// producing, and the shape adding them to an empty object leads to. Every
/// key is spelled in the input exactly as stored (no quote, backslash or
/// control character, so the raw bytes between quotes can only be that key).
struct Template {
    keys: Box<[Box<str>]>,
    shape: Rc<Shape>,
}

/// The templates a site records before it must earn each further one with
/// [`TEMPLATE_HITS_PER_LEARN`] hits (a position whose objects keep changing
/// keys stops paying for templates nobody uses).
const TEMPLATE_FREE_LEARNS: u32 = 8;
const TEMPLATE_HITS_PER_LEARN: u32 = 8;

/// One template slot: the template, how many objects in a row missed it, and
/// how many templates it has recorded and objects it has made.
#[derive(Default)]
struct Site {
    template: Option<Rc<Template>>,
    misses: u8,
    learns: u32,
    hits: u32,
    /// The parse (see [`JsonParseMemo::parse`]) `learns` and `hits` count
    /// for: the template itself outlives a parse, its learning budget does
    /// not.
    parse: u32,
}

impl Site {
    /// Starts this site's learning budget afresh when it was last counted
    /// by an earlier parse than `parse`.
    fn enter(&mut self, parse: u32) {
        if self.parse != parse {
            self.parse = parse;
            self.learns = 0;
            self.hits = 0;
        }
    }

    /// Whether a missed object should become this site's template.
    fn should_learn(&self) -> bool {
        (self.template.is_none() || self.misses >= TEMPLATE_REPLACE_MISSES)
            && (self.learns < TEMPLATE_FREE_LEARNS
                || self.hits / TEMPLATE_HITS_PER_LEARN >= self.learns - TEMPLATE_FREE_LEARNS)
    }
}

/// Per-parse state.
struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
    /// The transition caches of the ordinary object build, each remembering
    /// the key it was armed for (a cache is only ever offered that key).
    caches: Vec<Option<(Box<str>, PropertyCache)>>,
    /// The record templates (see [`Template`]), `TEMPLATE_SITES` once used.
    sites: Vec<Site>,
    /// The values of every open container, innermost last.
    vals: Vec<NanBox>,
    /// The keys of every open object's values, innermost last.
    keys: Vec<JsonKey>,
    /// The one-unit ASCII strings made so far this parse, by unit.
    units: [Option<NanBox>; 128],
    /// Scratch for strings with escapes.
    buf: Vec<u8>,
    /// The realm's root shape (every template grew from it).
    root: Rc<Shape>,
    /// This parse's number (see [`Site::enter`]).
    parse: u32,
}

/// The parts of [`JsonParser`] kept from one parse to the next: the
/// transition caches and record templates (each validated where it is used:
/// a cache by the shapes it saw, a template by the input's bytes and its
/// root shape), and the stacks' and scratch buffer's storage (left empty).
/// Documents of the same form, parsed one call each, then meet their
/// templates on the first record.
#[derive(Default)]
pub(crate) struct JsonParseMemo {
    caches: Vec<Option<(Box<str>, PropertyCache)>>,
    sites: Vec<Site>,
    vals: Vec<NanBox>,
    keys: Vec<JsonKey>,
    buf: Vec<u8>,
    /// The number of parses begun, wrapping.
    parse: u32,
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

    /// Four hex digits at `at` as a code unit.
    fn hex4(&self, at: usize) -> Option<u16> {
        let digits = self.src.get(at..at + 4)?;
        let mut v: u16 = 0;
        for &d in digits {
            v = (v << 4) | u16::from((d as char).to_digit(16)? as u8);
        }
        Some(v)
    }

    /// When the input at `pos` is `key` in quotes: consumes it.
    #[inline]
    fn eat_quoted(&mut self, key: &[u8]) -> bool {
        let n = key.len();
        match self.src.get(self.pos..self.pos + n + 2) {
            Some(t) if t[0] == b'"' && t[n + 1] == b'"' && &t[1..=n] == key => {
                self.pos += n + 2;
                true
            }
            _ => false,
        }
    }
}

/// A container [`Interp::json_fast_value`] has open. Its values so far are
/// `vals[start..]`; an object's keys are `keys[start..]` of the key stack (as
/// many as its values, plus the one whose value is being read).
enum Open {
    Array {
        start: usize,
        path: u64,
    },
    Object {
        vstart: usize,
        kstart: usize,
        path: u64,
        /// The template of this object's site when it opened, and whether
        /// every key so far has been that template's key at that position.
        template: Option<Rc<Template>>,
        matched: bool,
    },
}

impl Open {
    fn path(&self) -> u64 {
        match self {
            Open::Array { path, .. } | Open::Object { path, .. } => *path,
        }
    }
}

/// The path hash of a container that is value number `ordinal` of a container
/// with path `parent` (array elements all share ordinal 0, so the records of
/// an array share a site).
#[inline]
fn child_path(parent: u64, ordinal: usize) -> u64 {
    (parent.rotate_left(7) ^ (ordinal as u64 + 1)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[inline]
fn site_of(path: u64) -> usize {
    (path >> 56) as usize % TEMPLATE_SITES
}

/// A property name: key `i` of the object's template, a range of the input
/// (no escapes, `ascii` when every byte is), or decoded.
enum JsonKey {
    Template(usize),
    Borrowed(usize, usize, bool),
    Owned(alloc::string::String),
}

/// Where a string token ended up: borrowed from the input (no escapes) or
/// decoded into the parser's scratch buffer (WTF-8); and whether it is all
/// ASCII.
enum JsonStr {
    Borrowed(usize, usize, bool),
    Scratch(bool),
}

impl Interp {
    /// `JSON.parse(text)` with no reviver: the whole text must be one JSON
    /// value plus optional whitespace.
    pub(crate) fn json_parse_text(&mut self, text: &str) -> Result<NanBox, ExecError> {
        self.json_parse_wtf8(text.as_bytes())
    }

    /// [`json_parse_text`](Self::json_parse_text) of a string's WTF-8 bytes.
    pub(crate) fn json_parse_wtf8(&mut self, src: &[u8]) -> Result<NanBox, ExecError> {
        let mut m = self.json_parse_memo.take().unwrap_or_default();
        m.parse = m.parse.wrapping_add(1);
        let mut p = JsonParser {
            src,
            pos: 0,
            caches: core::mem::take(&mut m.caches),
            sites: core::mem::take(&mut m.sites),
            vals: core::mem::take(&mut m.vals),
            keys: core::mem::take(&mut m.keys),
            units: [None; 128],
            buf: core::mem::take(&mut m.buf),
            root: self.realm.root_shape_rc(),
            parse: m.parse,
        };
        let value = self.json_parse_run(&mut p);
        // Back into the memo, the stacks emptied (an error leaves values
        // on them).
        p.vals.clear();
        p.keys.clear();
        p.buf.clear();
        m.caches = p.caches;
        m.sites = p.sites;
        m.vals = p.vals;
        m.keys = p.keys;
        m.buf = p.buf;
        self.json_parse_memo = Some(m);
        value
    }

    /// The whole text at `p`: one JSON value plus optional whitespace.
    #[inline(never)]
    fn json_parse_run(&mut self, p: &mut JsonParser<'_>) -> Result<NanBox, ExecError> {
        let value = self.json_fast_value(p)?;
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
    /// from a worklist, so nothing about it recurses either. (Nothing here
    /// collects: the values waiting on the stacks need no rooting.)
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
                    JsonStr::Borrowed(a, b, true) if b == a + 1 => {
                        let unit = p.src[a];
                        match p.units[usize::from(unit)] {
                            Some(s) => s,
                            None => {
                                let s = NanBox::handle(
                                    self.realm.unit_string(u16::from(unit)).to_raw(),
                                );
                                p.units[usize::from(unit)] = Some(s);
                                s
                            }
                        }
                    }
                    JsonStr::Borrowed(a, b, ascii) => {
                        // A run between quotes of a well-formed WTF-8 string
                        // is itself well-formed WTF-8.
                        let bytes = &p.src[a..b];
                        let rope = if ascii {
                            Rope::from_ascii_bytes(bytes)
                        } else {
                            Rope::from_bytes(bytes)
                        };
                        NanBox::handle(self.realm.new_string_from_rope(rope).to_raw())
                    }
                    JsonStr::Scratch(ascii) => {
                        let rope = if ascii {
                            Rope::from_ascii_bytes(&p.buf)
                        } else {
                            Rope::from_bytes(&p.buf)
                        };
                        NanBox::handle(self.realm.new_string_from_rope(rope).to_raw())
                    }
                },
                b'[' => {
                    p.pos += 1;
                    p.skip_ws();
                    if p.peek() == Some(b']') {
                        p.pos += 1;
                        NanBox::handle(self.realm.new_array(Vec::new()).to_raw())
                    } else {
                        let path = Self::json_child_path(p, &open);
                        open.push(Open::Array {
                            start: p.vals.len(),
                            path,
                        });
                        continue 'value;
                    }
                }
                b'{' => {
                    p.pos += 1;
                    p.skip_ws();
                    if p.peek() == Some(b'}') {
                        p.pos += 1;
                        NanBox::handle(self.realm.new_object().to_raw())
                    } else {
                        let path = Self::json_child_path(p, &open);
                        let template = p.sites.get(site_of(path)).and_then(|s| s.template.clone());
                        let mut frame = Open::Object {
                            vstart: p.vals.len(),
                            kstart: p.keys.len(),
                            path,
                            matched: template.is_some(),
                            template,
                        };
                        self.json_fast_key(p, &mut frame)?;
                        open.push(frame);
                        continue 'value;
                    }
                }
                b'-' | b'0'..=b'9' => self.json_fast_number(p)?,
                _ => return Err(self.json_error("Unexpected token in JSON")),
            };
            // `v` is complete: hand it to the innermost open container, and
            // close every container it completes.
            loop {
                let Some(top) = open.last_mut() else {
                    return Ok(v);
                };
                p.vals.push(v);
                p.skip_ws();
                match (top, p.peek()) {
                    (Open::Array { .. }, Some(b',')) => {
                        p.pos += 1;
                        continue 'value;
                    }
                    (Open::Array { start, .. }, Some(b']')) => {
                        p.pos += 1;
                        let start = *start;
                        open.pop();
                        v = NanBox::handle(
                            self.realm.new_array_from_slice(&p.vals[start..]).to_raw(),
                        );
                        p.vals.truncate(start);
                    }
                    (Open::Array { .. }, _) => return Err(self.json_error("Expected ',' or ']'")),
                    (frame @ Open::Object { .. }, Some(b',')) => {
                        p.pos += 1;
                        self.json_fast_key(p, frame)?;
                        continue 'value;
                    }
                    (Open::Object { .. }, Some(b'}')) => {
                        p.pos += 1;
                        let depth = open.len() - 1;
                        let Some(frame) = open.pop() else {
                            unreachable!("the innermost container is this object");
                        };
                        v = self.json_close_object(p, frame, depth);
                    }
                    (Open::Object { .. }, _) => return Err(self.json_error("Expected ',' or '}'")),
                }
            }
        }
    }

    /// The path hash of a container starting at the cursor (see
    /// [`child_path`]).
    fn json_child_path(p: &JsonParser<'_>, open: &[Open]) -> u64 {
        match open.last() {
            None => 0x243F_6A88_85A3_08D3,
            Some(Open::Array { path, .. }) => child_path(*path, 0),
            Some(o @ Open::Object { vstart, .. }) => child_path(o.path(), p.vals.len() - vstart),
        }
    }

    /// The object `frame` holds, now closed: made in its template's shape in
    /// one step when every key matched, else built property by property (and
    /// then perhaps recorded as its site's template).
    fn json_close_object(&mut self, p: &mut JsonParser<'_>, frame: Open, depth: usize) -> NanBox {
        let Open::Object {
            vstart,
            kstart,
            path,
            template,
            matched,
        } = frame
        else {
            unreachable!("json_close_object of an object frame");
        };
        let n = p.vals.len() - vstart;
        let site = site_of(path);
        if matched
            && let Some(t) = &template
            && t.keys.len() == n
            && let Some(h) = self
                .realm
                .new_object_in_shape(&p.root, &t.shape, &p.vals[vstart..])
        {
            p.vals.truncate(vstart);
            p.keys.truncate(kstart);
            if let Some(s) = p.sites.get_mut(site) {
                s.enter(p.parse);
                s.misses = 0;
                s.hits = s.hits.saturating_add(1);
            }
            return NanBox::handle(h.to_raw());
        }
        let obj = self.realm.new_object();
        // Every key stays representable as a template key when it is spelled
        // in the input exactly as stored.
        let mut learnable = true;
        let src = p.src;
        for i in 0..n {
            let v = p.vals[vstart + i];
            // Taken out so the add can borrow the parser.
            let key = core::mem::replace(&mut p.keys[kstart + i], JsonKey::Template(0));
            match &key {
                JsonKey::Template(j) => {
                    let t = template.as_ref().expect("a template key has a template");
                    self.json_fast_add(p, obj, depth, i, &t.keys[*j], v);
                }
                JsonKey::Borrowed(a, b, ascii) => {
                    let bytes = &src[*a..*b];
                    if *ascii && let Ok(name) = core::str::from_utf8(bytes) {
                        self.json_fast_add(p, obj, depth, i, name, v);
                    } else {
                        learnable &= core::str::from_utf8(bytes).is_ok();
                        let name = crate::realm::user_key_wtf8(bytes);
                        self.json_fast_add(p, obj, depth, i, &name, v);
                    }
                }
                JsonKey::Owned(name) => {
                    learnable &= name
                        .bytes()
                        .all(|b| b != b'"' && b != BACKSLASH && b >= 0x20);
                    self.json_fast_add(p, obj, depth, i, name, v);
                }
            }
            p.keys[kstart + i] = key;
        }
        // Record this object as its site's template (see `Site::should_learn`).
        if p.sites.is_empty() {
            p.sites.resize_with(TEMPLATE_SITES, Site::default);
        }
        let s = &mut p.sites[site];
        s.enter(p.parse);
        s.misses = s.misses.saturating_add(1);
        if learnable
            && s.should_learn()
            && let Some(shape) = self.realm.plain_fresh_shape(obj, n)
        {
            let keys: Box<[Box<str>]> = p.keys[kstart..kstart + n]
                .iter()
                .map(|k| -> Box<str> {
                    match k {
                        JsonKey::Template(j) => template
                            .as_ref()
                            .map(|t| t.keys[*j].clone())
                            .unwrap_or_default(),
                        JsonKey::Borrowed(a, b, _) => core::str::from_utf8(&p.src[*a..*b])
                            .unwrap_or_default()
                            .into(),
                        JsonKey::Owned(name) => name.as_str().into(),
                    }
                })
                .collect();
            let s = &mut p.sites[site];
            s.template = Some(Rc::new(Template { keys, shape }));
            s.misses = 0;
            s.learns = s.learns.saturating_add(1);
        }
        p.vals.truncate(vstart);
        p.keys.truncate(kstart);
        NanBox::handle(obj.to_raw())
    }

    /// The next property name of the object `frame`, and its `:`, the cursor
    /// at the whitespace before the name. Keys live in the `&str`-keyed
    /// object layer; a lone surrogate in a key is stored in its encoded form
    /// (see `realm::user_key_wtf8`).
    fn json_fast_key(&mut self, p: &mut JsonParser<'_>, frame: &mut Open) -> Result<(), ExecError> {
        let Open::Object {
            kstart,
            template,
            matched,
            ..
        } = frame
        else {
            unreachable!("json_fast_key of an object frame");
        };
        p.skip_ws();
        let ordinal = p.keys.len() - *kstart;
        // The key the template has at this position, when the input spells
        // it out verbatim: then the property name needs no scan.
        let key = if *matched
            && let Some(t) = template.as_deref()
            && let Some(k) = t.keys.get(ordinal)
            && p.eat_quoted(k.as_bytes())
        {
            JsonKey::Template(ordinal)
        } else {
            *matched = false;
            if p.peek() != Some(b'"') {
                return Err(self.json_error("Expected property name"));
            }
            match self.json_fast_string(p)? {
                JsonStr::Borrowed(a, b, ascii) => JsonKey::Borrowed(a, b, ascii),
                // (An unescaped run cannot hold a raw NUL; an escaped
                // `\u0000...` key is escaped into the user namespace, see
                // `realm::is_internal_key`.)
                JsonStr::Scratch(_) => JsonKey::Owned(crate::realm::user_key_wtf8(&p.buf)),
            }
        };
        p.keys.push(key);
        p.skip_ws();
        if p.peek() != Some(b':') {
            return Err(self.json_error("Expected ':'"));
        }
        p.pos += 1;
        Ok(())
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

    #[inline]
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
        let src = p.src;
        let start = p.pos;
        let mut i = start;
        let neg = src.get(i) == Some(&b'-');
        if neg {
            i += 1;
        }
        // The digits of the significand (integer and fraction parts) as an
        // integer, exact while there are at most 19 of them, and how many of
        // them are fraction digits.
        let mut m: u64 = 0;
        let digits_from = |m: &mut u64, mut i: usize| {
            while let Some(&d @ b'0'..=b'9') = src.get(i) {
                *m = m.wrapping_mul(10).wrapping_add(u64::from(d - b'0'));
                i += 1;
            }
            i
        };
        let int_start = i;
        match src.get(i) {
            Some(b'0') => i += 1,
            Some(b'1'..=b'9') => i = digits_from(&mut m, i),
            _ => return Err(self.json_error("Invalid number in JSON")),
        }
        let mut ndigits = i - int_start;
        let mut frac = 0usize;
        if src.get(i) == Some(&b'.') {
            let f = i + 1;
            i = digits_from(&mut m, f);
            frac = i - f;
            if frac == 0 {
                return Err(self.json_error("Invalid number in JSON"));
            }
            ndigits += frac;
        }
        let mut has_exp = false;
        if let Some(b'e' | b'E') = src.get(i) {
            has_exp = true;
            i += 1;
            if let Some(b'+' | b'-') = src.get(i) {
                i += 1;
            }
            let e = i;
            while src.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
            if i == e {
                return Err(self.json_error("Invalid number in JSON"));
            }
        }
        p.pos = i;
        // An exactly representable significand (below 2^53) scaled by an
        // exactly representable power of ten (10^0..=10^22) is one correctly
        // rounded IEEE division. Everything else takes the full conversion.
        if !has_exp && ndigits <= 19 {
            let n = if m < (1u64 << 53) && frac <= 22 {
                Some(m as f64 / POW10[frac])
            } else {
                decimal_to_f64(m, frac)
            };
            if let Some(n) = n {
                return Ok(NanBox::number(if neg { -n } else { n }));
            }
        }
        core::str::from_utf8(&src[start..i])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .map(NanBox::number)
            .ok_or_else(|| self.json_error("Invalid number in JSON"))
    }

    /// Scans a string token at `p.pos` (the opening quote). Without escapes the
    /// result is the input range between the quotes; otherwise it is decoded
    /// into `p.buf` as WTF-8: a `\uXXXX` low surrogate right after a high one
    /// (escaped or not) joins it into one code point, and a lone one is kept as
    /// a code unit.
    fn json_fast_string(&mut self, p: &mut JsonParser<'_>) -> Result<JsonStr, ExecError> {
        const LO: u64 = 0x0101_0101_0101_0101;
        const HI: u64 = 0x8080_8080_8080_8080;
        p.pos += 1; // opening quote
        let start = p.pos;
        let mut high = 0u64;
        // Eight bytes at a time while none of them is a quote, a backslash or
        // a control character (bytes at or past 0x80 are only noted).
        while let Some(w) = p.src.get(p.pos..p.pos + 8) {
            let x = u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
            let q = x ^ (LO * u64::from(b'"'));
            let s = x ^ (LO * u64::from(BACKSLASH));
            let stop =
                ((q.wrapping_sub(LO) & !q) | (s.wrapping_sub(LO) & !s) | x.wrapping_sub(LO * 0x20))
                    & !x
                    & HI;
            if stop != 0 {
                break;
            }
            high |= x;
            p.pos += 8;
        }
        loop {
            match p.peek() {
                None => return Err(self.json_error("Unterminated string in JSON")),
                Some(b'"') => {
                    p.pos += 1;
                    return Ok(JsonStr::Borrowed(start, p.pos - 1, high & HI == 0));
                }
                Some(b'\\') => break,
                Some(b) if b < 0x20 => {
                    return Err(self.json_error("Bad control character in string literal in JSON"));
                }
                Some(b) => {
                    high |= u64::from(b);
                    p.pos += 1;
                }
            }
        }
        p.buf.clear();
        p.buf.extend_from_slice(&p.src[start..p.pos]);
        let mut run = p.pos;
        loop {
            match p.peek() {
                None => return Err(self.json_error("Unterminated string in JSON")),
                Some(b'"') => {
                    crate::wtf8::append(&mut p.buf, &p.src[run..p.pos]);
                    p.pos += 1;
                    let ascii = p.buf.is_ascii();
                    return Ok(JsonStr::Scratch(ascii));
                }
                Some(b'\\') => {
                    crate::wtf8::append(&mut p.buf, &p.src[run..p.pos]);
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
                            let unit = p
                                .hex4(p.pos + 1)
                                .ok_or_else(|| self.json_error("Invalid \\u escape in JSON"))?;
                            p.pos += 4;
                            if (0xDC00..=0xDFFF).contains(&unit)
                                && let Some(hi) = crate::wtf8::trailing_high_surrogate(&p.buf)
                            {
                                p.buf.truncate(p.buf.len() - 3);
                                let cp = 0x1_0000
                                    + ((u32::from(hi) - 0xD800) << 10)
                                    + (u32::from(unit) - 0xDC00);
                                crate::wtf8::encode_code_point(cp, &mut p.buf);
                            } else {
                                crate::wtf8::encode_utf16_unit(unit, &mut p.buf);
                            }
                        }
                        _ => return Err(self.json_error("Invalid escape in JSON")),
                    }
                    p.pos += 1;
                    run = p.pos;
                }
                Some(b) if b < 0x20 => {
                    return Err(self.json_error("Bad control character in string literal in JSON"));
                }
                Some(_) => p.pos += 1,
            }
        }
    }
}

/// `m / 10^frac`, correctly rounded (to nearest, ties to even), for a
/// significand `m` of at least 2^53 and at most 19 fraction digits (what a
/// shortest round-trip double of 16 or 17 digits usually is); `None` outside
/// that range. Exact integer arithmetic: `q = floor(m * 2^64 / 10^frac)` has
/// at least 54 significant bits, so rounding its top 53 bits, with the
/// division's remainder as the sticky part, rounds the exact quotient.
fn decimal_to_f64(m: u64, frac: usize) -> Option<f64> {
    if m < (1u64 << 53) || frac > 19 {
        return None;
    }
    let d = 10u64.pow(frac as u32);
    let num = u128::from(m) << 64;
    let q = num / u128::from(d);
    let sticky = num % u128::from(d) != 0;
    let bits = 128 - q.leading_zeros();
    // `m >= 2^53` and `d < 2^64` make `q >= 2^53`, so at least one bit goes.
    if bits <= 53 {
        return None;
    }
    let shift = bits - 53;
    let mut mant = (q >> shift) as u64;
    let rem = q & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    if rem > half || (rem == half && (sticky || mant & 1 == 1)) {
        // At most 2^53, still exact as an f64.
        mant += 1;
    }
    // The value is `mant * 2^(shift - 64)`, a power-of-two scaling of an
    // exact integer by a normal power of two (`shift - 64` is in -63..=11).
    let e = i64::from(shift) - 64;
    let scale = f64::from_bits(((1023 + e) as u64) << 52);
    Some(mant as f64 * scale)
}

/// 10^0 ..= 10^22, every one exactly representable in an f64.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

#[cfg(test)]
mod tests {
    use super::decimal_to_f64;
    use alloc::format;

    /// The exact conversion agrees with the standard library's correctly
    /// rounded one on random 16..=19-digit significands at every scale, and
    /// on the exact ties between two doubles just past 2^53 and 2^54.
    #[test]
    fn decimal_to_f64_rounds_like_std() {
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..200_000 {
            let digits = 16 + (next() % 4) as u32;
            let m = next() % 10u64.pow(digits);
            let frac = (next() % 20) as usize;
            let Some(got) = decimal_to_f64(m, frac) else {
                assert!(m < (1u64 << 53));
                continue;
            };
            let s = format!("{m}e-{frac}");
            assert_eq!(got.to_bits(), s.parse::<f64>().unwrap().to_bits(), "{s}");
        }
        for base in [1u64 << 53, 1u64 << 54, 1u64 << 60] {
            for k in 0..5_000u64 {
                for frac in 0..2 {
                    let m = base + k;
                    let got = decimal_to_f64(m, frac).unwrap();
                    let s = format!("{m}e-{frac}");
                    assert_eq!(got.to_bits(), s.parse::<f64>().unwrap().to_bits(), "{s}");
                }
            }
        }
    }
}
