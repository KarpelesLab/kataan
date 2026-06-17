use super::*;

/// Validates and canonicalizes a BCP-47 / UTS-35 `unicode_locale_id` (the
/// structural grammar, without CLDR alias/grandfathered replacement which the
/// `intl` crate doesn't expose). Returns the canonical tag, or `None` if the tag
/// is structurally invalid (the caller raises a `RangeError`).
///
/// Canonicalization performed: lowercase language; Titlecase script; UPPERCASE
/// region; lowercase variants **sorted** alphabetically; extensions ordered by
/// singleton (`x` private-use last); `-u-`/`-t-` keyword/field groups sorted by
/// key in ASCII order. Not performed (needs CLDR data the crate omits): legacy
/// language/region alias replacement, `-u-` type alias mapping (e.g. `yes`→`true`).
pub(crate) fn canonicalize_locale_id(tag: &str) -> Option<String> {
    // Structurally rejected outright: empty, non-ASCII, `_` separators, and the
    // empty subtags produced by leading/trailing/doubled `-`.
    if tag.is_empty() || !tag.is_ascii() || tag.contains('_') {
        return None;
    }
    let parts: Vec<&str> = tag.split('-').collect();
    if parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let is_alpha = |s: &str| s.bytes().all(|b| b.is_ascii_alphabetic());
    let is_digit = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    let is_alnum = |s: &str| s.bytes().all(|b| b.is_ascii_alphanumeric());

    let mut idx = 0usize;
    let n = parts.len();

    // unicode_language_subtag = alpha{2,3} | alpha{5,8}  (NOT 4, NOT extlang).
    let lang = parts[idx];
    if !((2..=3).contains(&lang.len()) || (5..=8).contains(&lang.len())) || !is_alpha(lang) {
        return None;
    }
    let language = lang.to_ascii_lowercase();
    idx += 1;
    // No extlang subtags allowed in UTS-35: a 3-alpha subtag after a 2-3 alpha
    // language would be an extlang (BCP-47) — invalid here.
    if idx < n && is_alpha(parts[idx]) && parts[idx].len() == 3 {
        return None;
    }

    // unicode_script_subtag = alpha{4}.
    let mut script = None;
    if idx < n && parts[idx].len() == 4 && is_alpha(parts[idx]) {
        let s = parts[idx];
        let mut t = String::new();
        for (i, c) in s.chars().enumerate() {
            if i == 0 {
                t.push(c.to_ascii_uppercase());
            } else {
                t.push(c.to_ascii_lowercase());
            }
        }
        script = Some(t);
        idx += 1;
    }

    // unicode_region_subtag = alpha{2} | digit{3}.
    let mut region = None;
    if idx < n
        && ((parts[idx].len() == 2 && is_alpha(parts[idx]))
            || (parts[idx].len() == 3 && is_digit(parts[idx])))
    {
        region = Some(parts[idx].to_ascii_uppercase());
        idx += 1;
    }

    // unicode_variant_subtag = (alphanum{5,8} | digit alphanum{3}).
    let mut variants: Vec<String> = Vec::new();
    while idx < n {
        let s = parts[idx];
        let is_variant = ((5..=8).contains(&s.len()) && is_alnum(s))
            || (s.len() == 4 && s.as_bytes()[0].is_ascii_digit() && is_alnum(s));
        if !is_variant {
            break;
        }
        let v = s.to_ascii_lowercase();
        if variants.contains(&v) {
            return None; // duplicate variant
        }
        variants.push(v);
        idx += 1;
    }
    variants.sort();

    // Extensions and private use: singleton (alphanum) followed by subtags.
    // Each singleton may appear once; `u`/`t` have their own subtag grammars but
    // we validate generically and canonicalize key ordering for `u` and `t`.
    let mut extensions: Vec<(char, String)> = Vec::new();
    let mut seen_singletons: Vec<char> = Vec::new();
    while idx < n {
        let sing = parts[idx];
        if sing.len() != 1 || !sing.as_bytes()[0].is_ascii_alphanumeric() {
            return None; // expected a singleton here
        }
        let singleton = sing.as_bytes()[0].to_ascii_lowercase() as char;
        if seen_singletons.contains(&singleton) {
            return None; // duplicate singleton
        }
        seen_singletons.push(singleton);
        idx += 1;
        // Gather this singleton's subtags. Private use (`x`) consumes *all* the
        // remaining subtags, including length-1 ones (so `x-u-foo` is one private
        // sequence, not the start of a `u` extension).
        let mut subs: Vec<String> = Vec::new();
        let private = singleton == 'x';
        while idx < n && (private || parts[idx].len() != 1) {
            let st = parts[idx];
            // Private-use subtags: 1..=8 alphanum. Others: 2..=8 alphanum.
            let min = if private { 1 } else { 2 };
            if !((min..=8).contains(&st.len()) && is_alnum(st)) {
                return None;
            }
            subs.push(st.to_ascii_lowercase());
            idx += 1;
        }
        // A singleton with a length-1 subtag (private use excepted) is invalid,
        // and any singleton must have at least one subtag.
        if subs.is_empty() {
            return None;
        }
        let body = canonicalize_extension(singleton, &subs)?;
        extensions.push((singleton, body));
    }
    // UTS-35 forbids a `unicode_locale_id` consisting only of private use; a
    // valid id must have a real language (we already required one), so this is
    // satisfied. Extensions sort by singleton with `x` last.
    extensions.sort_by_key(|(s, _)| (*s == 'x', *s));

    let mut out = language;
    if let Some(s) = script {
        out.push('-');
        out.push_str(&s);
    }
    if let Some(r) = region {
        out.push('-');
        out.push_str(&r);
    }
    for v in &variants {
        out.push('-');
        out.push_str(v);
    }
    for (_, body) in &extensions {
        out.push('-');
        out.push_str(body);
    }
    Some(out)
}

/// Canonicalizes one extension's subtags into its `singleton-subtag-…` body.
/// For `u`/`t` the keyword/field groups are sorted by key (ASCII order); for
/// other singletons (and private use `x`) the order is preserved.
fn canonicalize_extension(singleton: char, subs: &[String]) -> Option<String> {
    if singleton == 'u' {
        // -u- = attribute* (key type*)* — attributes (length-2..8, but a *key*
        // is exactly length 2) come first, then keyword groups keyed by a
        // 2-char key. Group the trailing keyword sequences and sort by key.
        let mut attributes: Vec<String> = Vec::new();
        let mut i = 0;
        // A `key` is exactly 2 chars: alphanum then alpha (e.g. `ca`, `nu`, `0c`).
        // An `attribute`/`type` is 3..=8 alphanum. A 2-char subtag whose 2nd char
        // is a digit (e.g. `c0`, `00`) is neither — structurally invalid.
        let is_key = |s: &str| s.len() == 2 && s.as_bytes()[1].is_ascii_alphabetic();
        let is_attr_or_type = |s: &str| (3..=8).contains(&s.len());
        while i < subs.len() && !is_key(&subs[i]) {
            if !is_attr_or_type(&subs[i]) {
                return None;
            }
            attributes.push(subs[i].clone());
            i += 1;
        }
        attributes.sort();
        let mut keywords: Vec<(String, Vec<String>)> = Vec::new();
        while i < subs.len() {
            let key = subs[i].clone();
            i += 1;
            let mut vals: Vec<String> = Vec::new();
            while i < subs.len() && !is_key(&subs[i]) {
                if !is_attr_or_type(&subs[i]) {
                    return None;
                }
                vals.push(subs[i].clone());
                i += 1;
            }
            // A `true` type value is elided in canonical form.
            if vals.len() == 1 && vals[0] == "true" {
                vals.clear();
            }
            keywords.push((key, vals));
        }
        keywords.sort_by(|a, b| a.0.cmp(&b.0));
        let mut body = String::from("u");
        for a in &attributes {
            body.push('-');
            body.push_str(a);
        }
        for (k, vals) in &keywords {
            body.push('-');
            body.push_str(k);
            for v in vals {
                body.push('-');
                body.push_str(v);
            }
        }
        return Some(body);
    }
    if singleton == 't' {
        // -t- = (tlang)? (tfield)*  where tfield = tkey tvalue+. A tkey is
        // exactly 2 chars with an alpha first and digit second (e.g. `m0`, `k0`).
        // The optional tlang (a language subtag) comes first.
        let is_tkey = |s: &str| {
            s.len() == 2
                && s.as_bytes()[0].is_ascii_alphabetic()
                && s.as_bytes()[1].is_ascii_digit()
        };
        let mut i = 0;
        let mut tlang: Vec<String> = Vec::new();
        while i < subs.len() && !is_tkey(&subs[i]) {
            tlang.push(subs[i].clone());
            i += 1;
        }
        // Canonicalize the tlang via the locale-id rules (lang/script/region/variants).
        let tlang_canon = if tlang.is_empty() {
            None
        } else {
            Some(canonicalize_locale_id(&tlang.join("-"))?)
        };
        let mut fields: Vec<(String, Vec<String>)> = Vec::new();
        while i < subs.len() {
            let key = subs[i].clone();
            i += 1;
            let mut vals: Vec<String> = Vec::new();
            while i < subs.len() && !is_tkey(&subs[i]) {
                vals.push(subs[i].clone());
                i += 1;
            }
            if vals.is_empty() {
                return None; // a tfield key must have a value
            }
            fields.push((key, vals));
        }
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        let mut body = String::from("t");
        if let Some(tl) = &tlang_canon {
            body.push('-');
            body.push_str(&tl.to_ascii_lowercase());
        }
        for (k, vals) in &fields {
            body.push('-');
            body.push_str(k);
            for v in vals {
                body.push('-');
                body.push_str(v);
            }
        }
        return Some(body);
    }
    // Other singletons / private use: subtags in given order, already lowercased.
    let mut body = String::new();
    body.push(singleton);
    for s in subs {
        body.push('-');
        body.push_str(s);
    }
    Some(body)
}

impl<'a> Interp<'a> {
    /// Builds an `Intl.NumberFormat`/`DateTimeFormat` instance — an object that
    /// captures the relevant options behind a `\0intl` kind marker. Used for both
    /// `new Intl.X(...)` and the callable-without-`new` form.
    pub(crate) fn make_intl_formatter(&mut self, id: u16, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        let kind = if id == N_INTL_NUMBER_FORMAT {
            "number"
        } else {
            "datetime"
        };
        let marker = self.new_str(kind);
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // `.format` is a readable function (so `typeof nf.format === "function"` and a
        // member call `nf.format(x)` works); it formats against its `this` formatter.
        let fmt = self.new_named_native("format", N_INTL_FORMAT);
        self.realm
            .set_property(obj, "format", NanBox::handle(fmt.to_raw()));
        // `resolvedOptions()` reports the (resolved) configuration.
        let ro = self.new_named_native("resolvedOptions", N_INTL_RESOLVED_OPTIONS);
        self.realm
            .set_property(obj, "resolvedOptions", NanBox::handle(ro.to_raw()));
        // `formatToParts(x)` returns the formatted output broken into typed parts.
        let ftp = self.new_named_native("formatToParts", N_INTL_FORMAT_TO_PARTS);
        self.realm
            .set_property(obj, "formatToParts", NanBox::handle(ftp.to_raw()));
        // The requested locale (a string first argument), defaulting to en-US.
        let locale = args
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_else(|| String::from("en-US"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            for key in [
                "style",
                "currency",
                "minimumFractionDigits",
                "maximumFractionDigits",
                "useGrouping",
                "signDisplay",
                "unit",
                "unitDisplay",
                "notation",
                // DateTimeFormat component options.
                "weekday",
                "era",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "hour12",
                "hourCycle",
                "dayPeriod",
                "fractionalSecondDigits",
                "timeZoneName",
                "dateStyle",
                "timeStyle",
                "timeZone",
            ] {
                if let Some(v) = self.realm.get_property(opts, key) {
                    self.realm.set_hidden_property(obj, key, v);
                }
            }
        }
        NanBox::handle(obj.to_raw())
    }

    /// Formats `value` per the `Intl.NumberFormat`/`DateTimeFormat` instance `handle`
    /// (a `\0intl`-marked object). Shared by `nf.format(x)` and the bound `nf.format`.
    /// `Number.prototype.toLocaleString(locale, options)` — with no options this is the
    /// grouped default; with an options object it honors `style` (decimal/percent/
    /// currency), `currency`, and `minimum`/`maximumFractionDigits` (en-US-ish, no real
    /// locale data; the rounding mode follows Rust's formatter, ~halfExpand).
    pub(crate) fn number_to_locale_string(&self, n: f64, opts: Option<NanBox>) -> String {
        let oh = match opts {
            Some(v) if !matches!(v.unpack(), Unpacked::Undefined | Unpacked::Null) => {
                match v.as_handle() {
                    Some(raw) => Handle::from_raw(raw),
                    None => return group_thousands(n),
                }
            }
            _ => return group_thousands(n),
        };
        // Non-finite values ignore the options (NaN/∞/-∞).
        if !n.is_finite() {
            return group_thousands(n);
        }
        let str_opt = |key: &str| -> Option<String> {
            self.realm
                .get_property(oh, key)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_display_string(v))
        };
        let num_opt = |key: &str| -> Option<i32> {
            self.realm
                .get_property(oh, key)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_number(v) as i32)
        };
        let style = str_opt("style").unwrap_or_else(|| String::from("decimal"));
        let (value, prefix, suffix, def_min, def_max) = match style.as_str() {
            "percent" => (n * 100.0, String::new(), String::from("%"), 0, 0),
            "currency" => {
                let sym = currency_symbol(&str_opt("currency").unwrap_or_default());
                (n, sym, String::new(), 2, 2)
            }
            _ => (n, String::new(), String::new(), 0, 3),
        };
        let min_frac = num_opt("minimumFractionDigits")
            .unwrap_or(def_min)
            .clamp(0, 100);
        let max_frac = num_opt("maximumFractionDigits")
            .unwrap_or(def_max.max(min_frac))
            .clamp(min_frac, 100);
        let neg = value.is_sign_negative() && value != 0.0;
        // Round to `max_frac` places, then trim trailing zeros down to `min_frac`.
        let formatted = alloc::format!("{:.*}", max_frac as usize, value.abs());
        let trimmed = if max_frac > min_frac && formatted.contains('.') {
            let dot = formatted.find('.').unwrap();
            let keep_min = dot + 1 + min_frac as usize;
            let mut end = formatted.len();
            while end > keep_min && formatted.as_bytes()[end - 1] == b'0' {
                end -= 1;
            }
            if end == dot + 1 {
                end = dot; // no fractional digits left → drop the '.'
            }
            String::from(&formatted[..end])
        } else {
            formatted
        };
        let grouped = group_thousands_str(&trimmed);
        let mut out = String::new();
        if neg {
            out.push('-');
        }
        out.push_str(&prefix);
        out.push_str(&grouped);
        out.push_str(&suffix);
        out
    }

    pub(crate) fn intl_format_value(&mut self, handle: Handle, value: NanBox) -> String {
        let kind = self
            .realm
            .get_property(handle, "\u{0}intl")
            .map(|k| self.realm.to_display_string(k))
            .unwrap_or_default();
        if kind == "datetime" {
            let ms = match value.as_handle().map(Handle::from_raw) {
                Some(h) if self.realm.date_at(h).is_some() => self.realm.date_at(h).unwrap(),
                _ => self.realm.to_number(value),
            };
            self.format_intl_datetime(handle, ms)
        } else {
            let n = self.realm.to_number(value);
            self.intl_format_number(handle, n)
        }
    }

    /// Builds an `Intl.RelativeTimeFormat` instance: an object capturing `numeric`/`style`
    /// with a readable `format(value, unit)` method.
    pub(crate) fn make_relative_time_format(&mut self, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            for key in ["numeric", "style"] {
                if let Some(v) = self.realm.get_property(opts, key) {
                    self.realm.set_hidden_property(obj, key, v);
                }
            }
        }
        let f = self.new_named_native("format", N_INTL_REL_TIME_FORMAT);
        self.realm
            .set_property(obj, "format", NanBox::handle(f.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Builds an `Intl.DisplayNames` instance: an object capturing `type` with a readable
    /// `of(code)` method.
    pub(crate) fn make_display_names(&mut self, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        let locale = args
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            for key in ["type", "style", "fallback"] {
                if let Some(v) = self.realm.get_property(opts, key) {
                    self.realm.set_hidden_property(obj, key, v);
                }
            }
        }
        let f = self.new_named_native("of", N_INTL_DISPLAY_NAMES_OF);
        self.realm
            .set_property(obj, "of", NanBox::handle(f.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Builds an `Intl.Collator` instance: an object capturing the locale and
    /// `sensitivity`/`numeric` options with a readable `compare` function (usable directly and
    /// as `arr.sort(collator.compare)`).
    pub(crate) fn make_collator(&mut self, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        let locale = args
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            for key in ["sensitivity", "numeric", "caseFirst"] {
                if let Some(v) = self.realm.get_property(opts, key) {
                    self.realm.set_hidden_property(obj, key, v);
                }
            }
        }
        let cmp = self.new_named_native("compare", N_INTL_COMPARE);
        self.realm
            .set_property(obj, "compare", NanBox::handle(cmp.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Builds an `Intl.ListFormat` instance: an object capturing the locale, `type`, and
    /// `style` with a readable `format(list)` method.
    pub(crate) fn make_list_format(&mut self, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        let locale = args
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            for key in ["type", "style"] {
                if let Some(v) = self.realm.get_property(opts, key) {
                    self.realm.set_hidden_property(obj, key, v);
                }
            }
        }
        let f = self.new_named_native("format", N_INTL_LIST_FORMAT_FORMAT);
        self.realm
            .set_property(obj, "format", NanBox::handle(f.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Builds an `Intl.PluralRules` instance: an object capturing the locale and `type`
    /// (cardinal/ordinal) with a readable `select(n)` method.
    pub(crate) fn make_plural_rules(&mut self, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        let locale = args
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            && let Some(v) = self.realm.get_property(opts, "type")
        {
            self.realm.set_hidden_property(obj, "type", v);
        }
        let sel = self.new_named_native("select", N_INTL_PLURAL_SELECT);
        self.realm
            .set_property(obj, "select", NanBox::handle(sel.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Builds an `Intl.Segmenter` instance: an object capturing `granularity` with a readable
    /// `segment(input)` method.
    pub(crate) fn make_segmenter(&mut self, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            && let Some(v) = self.realm.get_property(opts, "granularity")
        {
            self.realm.set_hidden_property(obj, "granularity", v);
        }
        let f = self.new_named_native("segment", N_INTL_SEGMENTER_SEGMENT);
        self.realm
            .set_property(obj, "segment", NanBox::handle(f.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Breaks a UTC millisecond timestamp into typed `(type, value)` parts per an
    /// `Intl.DateTimeFormat` instance's options, via the `intl` crate (CLDR, locale-aware).
    /// Used by both `format` and `formatToParts`.
    #[cfg(feature = "intl")]
    pub(crate) fn datetime_parts(
        &mut self,
        handle: Handle,
        ms: f64,
    ) -> Vec<(&'static str, String)> {
        use intl::datetime::{
            self, DateStyle, DateTime, DateTimeFormatOptions, HourCycle, MonthStyle, NameStyle,
            Numeric2Digit, TimeZoneNameStyle,
        };
        let opt = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let locale = opt(self, "\u{0}locale").unwrap_or_else(|| String::from("en"));
        let msi = ms as i64;
        let day = msi.div_euclid(86_400_000);
        let tod = msi.rem_euclid(86_400_000);
        let (y, mo, d) = crate::realm::civil_from_days(day);
        let dt = DateTime {
            year: y as i32,
            month: mo as u8,
            day: d as u8,
            hour: (tod / 3_600_000) as u8,
            minute: ((tod / 60_000) % 60) as u8,
            second: ((tod / 1_000) % 60) as u8,
            millisecond: (tod % 1_000) as u16,
        };
        let name = |s: &str| match s {
            "long" => Some(NameStyle::Long),
            "short" => Some(NameStyle::Short),
            "narrow" => Some(NameStyle::Narrow),
            _ => None,
        };
        let n2 = |s: &str| match s {
            "numeric" => Some(Numeric2Digit::Numeric),
            "2-digit" => Some(Numeric2Digit::TwoDigit),
            _ => None,
        };
        let dstyle = |s: &str| match s {
            "full" => Some(DateStyle::Full),
            "long" => Some(DateStyle::Long),
            "medium" => Some(DateStyle::Medium),
            "short" => Some(DateStyle::Short),
            _ => None,
        };
        let mut o = DateTimeFormatOptions::default();
        // `dateStyle`/`timeStyle` are mutually exclusive with component fields (the crate
        // errors if both are set), matching ECMA-402.
        if opt(self, "dateStyle").is_some() || opt(self, "timeStyle").is_some() {
            o.date_style = opt(self, "dateStyle").as_deref().and_then(dstyle);
            o.time_style = opt(self, "timeStyle").as_deref().and_then(dstyle);
        } else {
            o.weekday = opt(self, "weekday").as_deref().and_then(name);
            o.era = opt(self, "era").as_deref().and_then(name);
            o.year = opt(self, "year").as_deref().and_then(n2);
            o.month = opt(self, "month").as_deref().and_then(|s| match s {
                "numeric" => Some(MonthStyle::Numeric),
                "2-digit" => Some(MonthStyle::TwoDigit),
                "long" => Some(MonthStyle::Long),
                "short" => Some(MonthStyle::Short),
                "narrow" => Some(MonthStyle::Narrow),
                _ => None,
            });
            o.day = opt(self, "day").as_deref().and_then(n2);
            o.hour = opt(self, "hour").as_deref().and_then(n2);
            o.minute = opt(self, "minute").as_deref().and_then(n2);
            o.second = opt(self, "second").as_deref().and_then(n2);
            o.day_period = opt(self, "dayPeriod").as_deref().and_then(name);
            o.fractional_second_digits =
                opt(self, "fractionalSecondDigits").and_then(|s| s.parse().ok());
            // ECMA-402 default when no component is requested: a numeric date.
            if o.weekday.is_none()
                && o.era.is_none()
                && o.year.is_none()
                && o.month.is_none()
                && o.day.is_none()
                && o.hour.is_none()
                && o.minute.is_none()
                && o.second.is_none()
            {
                o.year = Some(Numeric2Digit::Numeric);
                o.month = Some(MonthStyle::Numeric);
                o.day = Some(Numeric2Digit::Numeric);
            }
        }
        o.hour12 = self
            .realm
            .get_property(handle, "hour12")
            .and_then(|v| match v.unpack() {
                Unpacked::Bool(b) => Some(b),
                _ => None,
            });
        o.hour_cycle = opt(self, "hourCycle").as_deref().and_then(|s| match s {
            "h11" => Some(HourCycle::H11),
            "h12" => Some(HourCycle::H12),
            "h23" => Some(HourCycle::H23),
            "h24" => Some(HourCycle::H24),
            _ => None,
        });
        if let Some(tzn) = opt(self, "timeZoneName") {
            o.time_zone_name = match tzn.as_str() {
                "long" => Some(TimeZoneNameStyle::Long),
                "short" => Some(TimeZoneNameStyle::Short),
                "shortOffset" => Some(TimeZoneNameStyle::ShortOffset),
                "longOffset" => Some(TimeZoneNameStyle::LongOffset),
                "shortGeneric" => Some(TimeZoneNameStyle::ShortGeneric),
                "longGeneric" => Some(TimeZoneNameStyle::LongGeneric),
                _ => None,
            };
            o.tz_offset_minutes = Some(0); // engine is UTC-only
        }
        match datetime::format_to_parts(&locale, &dt, &o) {
            Ok(parts) => parts
                .into_iter()
                .map(|p| (p.kind.as_str(), p.value))
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Hand-rolled en-US fallback for [`datetime_parts`](Self::datetime_parts) when the `intl`
    /// crate is unavailable: component options (`weekday`/`year`/`month`/`day`/`hour`/`minute`/
    /// `second`/`hour12`/`era`/`dateStyle`/`timeStyle`) with `literal` separators, UTC.
    #[cfg(not(feature = "intl"))]
    pub(crate) fn datetime_parts(
        &mut self,
        handle: Handle,
        ms: f64,
    ) -> Vec<(&'static str, String)> {
        let opt = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let msi = ms as i64;
        let day = msi.div_euclid(86_400_000);
        let tod = msi.rem_euclid(86_400_000);
        let (y, mo, d) = crate::realm::civil_from_days(day);
        let (mo, d) = (i64::from(mo), i64::from(d));
        let wd_idx = (day + 4).rem_euclid(7) as usize; // 0 = Sunday
        let hour24 = tod / 3_600_000;
        let minute = (tod / 60_000) % 60;
        let second = (tod / 1_000) % 60;

        const MONTHS: [&str; 12] = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ];
        const WEEKDAYS: [&str; 7] = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ];
        let two = |v: i64| alloc::format!("{v:02}");
        let bare = |v: i64| alloc::format!("{v}");

        // Effective component options (after expanding dateStyle/timeStyle presets).
        let mut weekday = opt(self, "weekday");
        let mut year = opt(self, "year");
        let mut month = opt(self, "month");
        let mut day_o = opt(self, "day");
        let mut hour = opt(self, "hour");
        let mut minute_o = opt(self, "minute");
        let mut second_o = opt(self, "second");
        match opt(self, "dateStyle").as_deref() {
            Some("full") => {
                weekday = Some(String::from("long"));
                year = Some(String::from("numeric"));
                month = Some(String::from("long"));
                day_o = Some(String::from("numeric"));
            }
            Some("long") => {
                year = Some(String::from("numeric"));
                month = Some(String::from("long"));
                day_o = Some(String::from("numeric"));
            }
            Some("medium") => {
                year = Some(String::from("numeric"));
                month = Some(String::from("short"));
                day_o = Some(String::from("numeric"));
            }
            Some("short") => {
                year = Some(String::from("2-digit"));
                month = Some(String::from("numeric"));
                day_o = Some(String::from("numeric"));
            }
            _ => {}
        }
        match opt(self, "timeStyle").as_deref() {
            Some("full" | "long" | "medium") => {
                hour = Some(String::from("numeric"));
                minute_o = Some(String::from("2-digit"));
                second_o = Some(String::from("2-digit"));
            }
            Some("short") => {
                hour = Some(String::from("numeric"));
                minute_o = Some(String::from("2-digit"));
            }
            _ => {}
        }
        // With no options at all, the default is a numeric date.
        if weekday.is_none()
            && year.is_none()
            && month.is_none()
            && day_o.is_none()
            && hour.is_none()
            && minute_o.is_none()
            && second_o.is_none()
        {
            year = Some(String::from("numeric"));
            month = Some(String::from("numeric"));
            day_o = Some(String::from("numeric"));
        }

        let year_str = |style: &str| -> String {
            if style == "2-digit" {
                two(y.rem_euclid(100))
            } else {
                bare(y)
            }
        };
        let named_month = matches!(month.as_deref(), Some("long" | "short" | "narrow"));

        let lit = |s: &str| ("literal", String::from(s));

        // --- Date components (typed parts, en-US order) ---
        let mut date: Vec<(&'static str, String)> = Vec::new();
        if let Some(ws) = &weekday {
            let name = WEEKDAYS[wd_idx];
            date.push((
                "weekday",
                String::from(if ws == "long" { name } else { &name[..3] }),
            ));
        }
        if named_month {
            if !date.is_empty() {
                date.push(lit(", "));
            }
            if let Some(m) = &month {
                let name = MONTHS[(mo as usize).saturating_sub(1).min(11)];
                date.push((
                    "month",
                    String::from(if m == "long" { name } else { &name[..3] }),
                ));
            }
            if let Some(ds) = &day_o {
                date.push(lit(" "));
                date.push(("day", if ds == "2-digit" { two(d) } else { bare(d) }));
            }
            if let Some(ys) = &year {
                date.push(lit(if day_o.is_some() { ", " } else { " " }));
                date.push(("year", year_str(ys)));
            }
        } else {
            // A weekday-only request has no trailing separator; add ", " only before an
            // actual numeric date.
            if !date.is_empty() && (month.is_some() || day_o.is_some() || year.is_some()) {
                date.push(lit(", "));
            }
            let mut first = true;
            if let Some(m) = &month {
                date.push(("month", if m == "2-digit" { two(mo) } else { bare(mo) }));
                first = false;
            }
            if let Some(ds) = &day_o {
                if !first {
                    date.push(lit("/"));
                }
                date.push(("day", if ds == "2-digit" { two(d) } else { bare(d) }));
                first = false;
            }
            if let Some(ys) = &year {
                if !first {
                    date.push(lit("/"));
                }
                date.push(("year", year_str(ys)));
            }
        }
        if opt(self, "era").is_some() {
            if !date.is_empty() {
                date.push(lit(" "));
            }
            date.push(("era", String::from(if y > 0 { "AD" } else { "BC" })));
        }

        // --- Time components ---
        let mut time: Vec<(&'static str, String)> = Vec::new();
        if hour.is_some() || minute_o.is_some() || second_o.is_some() {
            // en-US defaults to 12-hour unless `hour12: false`.
            let h12 = !matches!(
                self.realm.get_property(handle, "hour12"),
                Some(v) if matches!(v.unpack(), Unpacked::Bool(false))
            );
            let h = if h12 {
                let m = hour24 % 12;
                if m == 0 { 12 } else { m }
            } else {
                hour24
            };
            time.push((
                "hour",
                if hour.as_deref() == Some("2-digit") {
                    two(h)
                } else {
                    bare(h)
                },
            ));
            if minute_o.is_some() {
                time.push(lit(":"));
                time.push(("minute", two(minute)));
            }
            if second_o.is_some() {
                time.push(lit(":"));
                time.push(("second", two(second)));
            }
            if h12 {
                // CLDR separates the time from AM/PM with U+202F (narrow no-break space).
                time.push(lit("\u{202f}"));
                time.push((
                    "dayPeriod",
                    String::from(if hour24 < 12 { "AM" } else { "PM" }),
                ));
            }
        }

        // --- Combine ---
        let mut parts = date;
        if !parts.is_empty() && !time.is_empty() {
            // CLDR's standard date-time connector is ", " (the crate uses it too).
            let _ = named_month;
            parts.push(lit(", "));
        }
        parts.extend(time);
        parts
    }

    /// The `Intl.DateTimeFormat` rendering of `ms` as a flat string (joins
    /// [`datetime_parts`](Self::datetime_parts)).
    pub(crate) fn format_intl_datetime(&mut self, handle: Handle, ms: f64) -> String {
        let mut s = String::new();
        for (_, v) in self.datetime_parts(handle, ms) {
            s.push_str(&v);
        }
        s
    }

    /// Builds the `intl` crate's `NumberFormatOptions` from an `Intl.NumberFormat` instance's
    /// stored JS options.
    #[cfg(feature = "intl")]
    pub(crate) fn number_format_options(
        &mut self,
        handle: Handle,
    ) -> intl::number::NumberFormatOptions {
        use intl::number::{
            CompactDisplay, CurrencyDisplay, Notation, NumberFormatOptions, NumberStyle,
            RoundingMode, SignDisplay, UnitDisplay, UseGrouping,
        };
        let opt_str = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let opt_num = |this: &mut Self, k: &str| -> Option<u8> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_number(v) as u8)
        };
        let mut o = NumberFormatOptions {
            style: match opt_str(self, "style").as_deref() {
                Some("percent") => NumberStyle::Percent,
                Some("currency") => NumberStyle::Currency,
                Some("unit") => NumberStyle::Unit,
                _ => NumberStyle::Decimal,
            },
            notation: match opt_str(self, "notation").as_deref() {
                Some("scientific") => Notation::Scientific,
                Some("engineering") => Notation::Engineering,
                Some("compact") => Notation::Compact,
                _ => Notation::Standard,
            },
            compact_display: match opt_str(self, "compactDisplay").as_deref() {
                Some("long") => CompactDisplay::Long,
                _ => CompactDisplay::Short,
            },
            sign_display: match opt_str(self, "signDisplay").as_deref() {
                Some("always") => SignDisplay::Always,
                Some("exceptZero") => SignDisplay::ExceptZero,
                Some("negative") => SignDisplay::Negative,
                Some("never") => SignDisplay::Never,
                _ => SignDisplay::Auto,
            },
            currency_display: match opt_str(self, "currencyDisplay").as_deref() {
                Some("code") => CurrencyDisplay::Code,
                Some("name") => CurrencyDisplay::Name,
                Some("narrowSymbol") => CurrencyDisplay::NarrowSymbol,
                _ => CurrencyDisplay::Symbol,
            },
            unit_display: match opt_str(self, "unitDisplay").as_deref() {
                Some("long") => UnitDisplay::Long,
                Some("narrow") => UnitDisplay::Narrow,
                _ => UnitDisplay::Short,
            },
            // ECMA-402's default rounding is half-expand (1.25 → 1.3), not banker's rounding.
            rounding_mode: RoundingMode::HalfExpand,
            ..Default::default()
        };
        if matches!(
            self.realm
                .get_property(handle, "useGrouping")
                .map(|v| v.unpack()),
            Some(Unpacked::Bool(false))
        ) {
            o.use_grouping = UseGrouping::Never;
        }
        o.minimum_fraction_digits = opt_num(self, "minimumFractionDigits");
        o.maximum_fraction_digits = opt_num(self, "maximumFractionDigits");
        o.minimum_significant_digits = opt_num(self, "minimumSignificantDigits");
        o.maximum_significant_digits = opt_num(self, "maximumSignificantDigits");
        // Scientific/engineering cap the mantissa at 3 fraction digits by default (1.235E5).
        if o.maximum_fraction_digits.is_none()
            && o.maximum_significant_digits.is_none()
            && matches!(o.notation, Notation::Scientific | Notation::Engineering)
        {
            o.maximum_fraction_digits = Some(3);
        }
        if let Some(c) = opt_str(self, "currency") {
            o.currency = Some(self.intern_static(&c));
        }
        if let Some(u) = opt_str(self, "unit") {
            o.unit = Some(self.intern_static(&u));
        }
        o
    }

    /// Whether an `Intl.NumberFormat` instance must use the hand-rolled path rather than the
    /// `intl` crate: `style: "unit"` and `notation: "compact"` aren't yet faithfully rendered
    /// by `intl::number::format` (units are dropped; compact rounds differently from V8).
    #[cfg(feature = "intl")]
    pub(crate) fn number_uses_handrolled(&mut self, handle: Handle) -> bool {
        let get = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        get(self, "style").as_deref() == Some("unit")
            || get(self, "notation").as_deref() == Some("compact")
    }

    /// Formats `n` per an `Intl.NumberFormat` instance. With the `intl` crate, all styles
    /// except those in [`number_uses_handrolled`](Self::number_uses_handrolled) go through
    /// `intl::number::format` (CLDR, locale-aware, full ECMA-402 options); the rest, and the
    /// no-`intl` build, use the hand-rolled en-US path below.
    pub(crate) fn intl_format_number(&mut self, handle: Handle, n: f64) -> String {
        #[cfg(feature = "intl")]
        if !self.number_uses_handrolled(handle) {
            let locale = self
                .realm
                .get_property(handle, "\u{0}locale")
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("en"));
            let opts = self.number_format_options(handle);
            return intl::number::format(&locale, n, &opts);
        }
        let opt_str = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let opt_num = |this: &mut Self, k: &str| -> Option<i32> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_number(v) as i32)
        };
        let style = opt_str(self, "style").unwrap_or_else(|| String::from("decimal"));
        let currency = opt_str(self, "currency");
        // Non-finite values render as the core glyph (∞ / NaN) with the sign and
        // currency/percent affixes, but no grouping or fraction digits.
        if !n.is_finite() {
            let mut out = String::new();
            if n.is_sign_negative() && !n.is_nan() {
                out.push('-');
            }
            if style == "currency" {
                out.push_str(&currency_symbol(currency.as_deref().unwrap_or("")));
            }
            out.push_str(if n.is_nan() { "NaN" } else { "∞" });
            if style == "percent" {
                out.push('%');
            }
            return out;
        }
        let use_grouping = !matches!(
            self.realm.get_property(handle, "useGrouping"),
            Some(v) if matches!(v.unpack(), Unpacked::Bool(false))
        );
        // Default fraction digits: currency = 2 (0 for JPY), else 0..=3.
        let is_jpy = currency.as_deref() == Some("JPY");
        let (def_min, def_max) = match style.as_str() {
            "currency" if is_jpy => (0, 0),
            "currency" => (2, 2),
            "percent" => (0, 0),
            _ => (0, 3),
        };
        let min = opt_num(self, "minimumFractionDigits")
            .unwrap_or(def_min)
            .clamp(0, 20);
        let max = opt_num(self, "maximumFractionDigits")
            .unwrap_or(def_max.max(min))
            .clamp(min, 20);
        let value = if style == "percent" { n * 100.0 } else { n };
        // Round `x` to `max` digits, trimming trailing zeros down to `min`.
        let fmt_digits = |x: f64| -> String {
            let mut s = alloc::format!("{:.*}", max as usize, x);
            if max > min && s.contains('.') {
                while s.ends_with('0')
                    && s.split_once('.').map_or(0, |(_, f)| f.len()) > min as usize
                {
                    s.pop();
                }
                if s.ends_with('.') {
                    s.pop();
                }
            }
            s
        };
        // `notation: "scientific" | "engineering"` renders `mantissa E exponent` (no
        // grouping); engineering pins the exponent to a multiple of 3.
        let notation = opt_str(self, "notation").unwrap_or_default();
        let (s, do_group) = if matches!(notation.as_str(), "scientific" | "engineering") {
            let neg = value < 0.0;
            let mag = value.abs();
            let mut exp = 0i32;
            let mut p = 1.0f64; // 10^exp
            if mag >= 1.0 {
                while mag >= p * 10.0 {
                    p *= 10.0;
                    exp += 1;
                }
            } else if mag > 0.0 {
                while mag < p {
                    p /= 10.0;
                    exp -= 1;
                }
            }
            if notation == "engineering" {
                // Drop the exponent to the nearest lower multiple of 3, scaling the mantissa
                // up to compensate (p = 10^exp must shrink as exp shrinks).
                let shift = exp.rem_euclid(3);
                exp -= shift;
                for _ in 0..shift {
                    p /= 10.0;
                }
            }
            let m = if mag == 0.0 { 0.0 } else { mag / p };
            let sign = if neg { "-" } else { "" };
            (alloc::format!("{sign}{}E{exp}", fmt_digits(m)), false)
        } else if notation == "compact" {
            // `notation: "compact"` (short): divide by the largest 10^(3k) scale and append
            // its suffix (K/M/B/T), showing one fraction digit only for a single-digit
            // mantissa (`1.2M`, but `123K` and `12K`).
            let neg = value < 0.0;
            let mag = value.abs();
            let (div, suffix) = if mag >= 1e12 {
                (1e12, "T")
            } else if mag >= 1e9 {
                (1e9, "B")
            } else if mag >= 1e6 {
                (1e6, "M")
            } else if mag >= 1e3 {
                (1e3, "K")
            } else {
                (1.0, "")
            };
            let m = mag / div;
            let cmax = if m < 10.0 { 1 } else { 0 };
            let mut ms = alloc::format!("{m:.*}", cmax as usize);
            if ms.contains('.') {
                while ms.ends_with('0') {
                    ms.pop();
                }
                if ms.ends_with('.') {
                    ms.pop();
                }
            }
            let sign = if neg { "-" } else { "" };
            (alloc::format!("{sign}{ms}{suffix}"), suffix.is_empty())
        } else {
            (fmt_digits(value), use_grouping)
        };
        // Group the integer part (skipped for scientific/engineering).
        let grouped = if do_group {
            let neg = s.starts_with('-');
            let body = s.trim_start_matches('-');
            let (ip, fp) = body
                .split_once('.')
                .map_or((body, None), |(i, f)| (i, Some(f)));
            let mut g = String::new();
            let len = ip.len();
            for (i, b) in ip.bytes().enumerate() {
                if i > 0 && (len - i) % 3 == 0 {
                    g.push(',');
                }
                g.push(b as char);
            }
            if let Some(f) = fp {
                g.push('.');
                g.push_str(f);
            }
            if neg { alloc::format!("-{g}") } else { g }
        } else {
            s
        };
        // Separate the sign from the magnitude so `signDisplay` and the style affixes
        // compose with the sign outermost (e.g. `-$5.00`, `+5%`).
        let neg = grouped.starts_with('-');
        let magnitude = grouped.trim_start_matches('-');
        let styled = match style.as_str() {
            "percent" => alloc::format!("{magnitude}%"),
            "currency" => {
                let sym = match currency.as_deref() {
                    Some("USD") => "$",
                    Some("EUR") => "€",
                    Some("GBP") => "£",
                    Some("JPY" | "CNY") => "¥",
                    Some(other) => {
                        let other = String::from(other);
                        return alloc::format!(
                            "{}{other}\u{a0}{magnitude}",
                            if neg { "-" } else { "" }
                        );
                    }
                    None => "$",
                };
                alloc::format!("{sym}{magnitude}")
            }
            "unit" => {
                // `style: "unit"` appends the unit's short symbol (`5 km`); a
                // `unit-per-unit` compound joins the two with `/` (`5 km/h`).
                let unit = opt_str(self, "unit").unwrap_or_default();
                let sym = unit.split_once("-per-").map_or_else(
                    || String::from(unit_symbol(&unit)),
                    |(a, b)| alloc::format!("{}/{}", unit_symbol(a), unit_symbol(b)),
                );
                // Temperature/angle units attach with no space (`20°C`); others use a
                // (non-breaking) space (`5 km`).
                let sep = if matches!(unit.as_str(), "celsius" | "fahrenheit" | "degree") {
                    ""
                } else {
                    "\u{a0}"
                };
                alloc::format!("{magnitude}{sep}{sym}")
            }
            _ => String::from(magnitude),
        };
        let is_zero = magnitude.bytes().all(|b| matches!(b, b'0' | b'.' | b','));
        let sign = match opt_str(self, "signDisplay").as_deref() {
            Some("never") => "",
            Some("always") => {
                if neg {
                    "-"
                } else {
                    "+"
                }
            }
            Some("exceptZero") if !is_zero => {
                if neg {
                    "-"
                } else {
                    "+"
                }
            }
            // "auto" (default) and "exceptZero" on a zero: a sign only for negatives.
            _ if neg => "-",
            _ => "",
        };
        alloc::format!("{sign}{styled}")
    }

    /// ECMA-402 `CanonicalizeLocaleList(locales)`: coerces `locales` to a list of
    /// canonical locale tags (deduplicated, order preserved). `undefined` → empty
    /// list; a single string is treated as a one-element list; otherwise the
    /// argument is `ToObject`-ed and iterated by its `length`. Each element must be
    /// a String or Object (else **TypeError**), and each tag must be a structurally
    /// valid locale (else **RangeError**). A `Locale` instance contributes its
    /// already-canonical `[[Locale]]` tag.
    pub(crate) fn canonicalize_locale_list(
        &mut self,
        locales: NanBox,
    ) -> Result<Vec<String>, ExecError> {
        let mut seen: Vec<String> = Vec::new();
        if matches!(locales.unpack(), Unpacked::Undefined) {
            return Ok(seen);
        }
        // A bare string is a single-element list (no ToObject coercion of the
        // characters). A `Locale` object short-circuits below in the loop.
        let is_string = locales
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.string_value(h).is_some());
        let push_tag =
            |this: &mut Self, tag: &str, seen: &mut Vec<String>| -> Result<(), ExecError> {
                match canonicalize_locale_id(tag) {
                    Some(c) => {
                        if !seen.contains(&c) {
                            seen.push(c);
                        }
                        Ok(())
                    }
                    None => {
                        let m = this.new_str(&alloc::format!(
                            "Incorrect locale information provided: {tag}"
                        ));
                        Err(ExecError::Throw(this.make_error(N_RANGE_ERROR, Some(m))))
                    }
                }
            };
        if is_string {
            let s = self.coerce_to_string(locales)?;
            push_tag(self, &s, &mut seen)?;
            return Ok(seen);
        }
        // ToObject(null) is a TypeError (undefined was handled above).
        if matches!(locales.unpack(), Unpacked::Null) {
            let m = self.new_str("Cannot convert null to object");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // ToObject, then iterate [0, len) reading each element (getter-aware, so
        // throwing getters and inherited indices behave per spec).
        let obj = self.coerce_to_object(locales);
        let Some(oh) = obj.as_handle().map(Handle::from_raw) else {
            return Ok(seen);
        };
        let len_v = self.read_member(oh, "length")?;
        // ToLength: a Symbol `length` is a TypeError (ToNumber throws).
        let len_v = self.coerce_to_number(len_v)?;
        let len_f = self.realm.to_number(len_v);
        let len = if len_f.is_nan() || len_f <= 0.0 {
            0u64
        } else {
            len_f.min(u32::MAX as f64 * 2.0) as u64
        };
        for i in 0..len {
            let key = alloc::format!("{i}");
            // HasProperty check: skip absent indices (sparse array-likes).
            if !self.has_property(oh, &key) {
                continue;
            }
            let el = self.read_member(oh, &key)?;
            // Element must be a String or Object.
            let el_is_string = el
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.realm.string_value(h).is_some());
            let el_is_object = self.is_object_value(el) && !el_is_string;
            if !el_is_string && !el_is_object {
                let m = self.new_str("locale list element is not a string or object");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // A `Locale` instance contributes its canonical `[[Locale]]`.
            if let Some(h) = el.as_handle().map(Handle::from_raw)
                && let Some(loc) = self.realm.get_property(h, "\u{0}locale_tag")
            {
                let tag = self.realm.to_display_string(loc);
                if !seen.contains(&tag) {
                    seen.push(tag);
                }
                continue;
            }
            let s = self.coerce_to_string(el)?;
            push_tag(self, &s, &mut seen)?;
        }
        Ok(seen)
    }

    /// `Intl.getCanonicalLocales(locales)` — a fresh, mutable Array of the
    /// canonicalized tags.
    pub(crate) fn intl_get_canonical_locales(
        &mut self,
        locales: NanBox,
    ) -> Result<NanBox, ExecError> {
        let tags = self.canonicalize_locale_list(locales)?;
        let elems: Vec<NanBox> = tags.iter().map(|t| self.new_str(t)).collect();
        Ok(NanBox::handle(self.realm.new_array(elems).to_raw()))
    }

    /// `Intl.supportedValuesOf(key)` — a sorted, duplicate-free Array of the
    /// supported identifiers for `key` (`calendar`/`collation`/`currency`/
    /// `numberingSystem`/`timeZone`/`unit`). A `key` outside this set is a
    /// **RangeError**.
    pub(crate) fn intl_supported_values_of(&mut self, key: NanBox) -> Result<NanBox, ExecError> {
        let k = self.coerce_to_string(key)?;
        let values: &[&str] = match k.as_str() {
            "calendar" => &[
                "buddhist",
                "chinese",
                "coptic",
                "dangi",
                "ethioaa",
                "ethiopic",
                "gregory",
                "hebrew",
                "indian",
                "islamic",
                "islamic-civil",
                "islamic-rgsa",
                "islamic-tbla",
                "islamic-umalqura",
                "iso8601",
                "japanese",
                "persian",
                "roc",
            ],
            "collation" => &[
                "compat", "dict", "emoji", "eor", "phonebk", "phonetic", "pinyin", "searchjl",
                "stroke", "trad", "unihan", "zhuyin",
            ],
            "currency" => &[
                "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN", "BAM", "BBD",
                "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL", "BSD", "BTN", "BWP", "BYN",
                "BZD", "CAD", "CDF", "CHF", "CLP", "CNY", "COP", "CRC", "CUP", "CVE", "CZK", "DJF",
                "DKK", "DOP", "DZD", "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS",
                "GIP", "GMD", "GNF", "GTQ", "GYD", "HKD", "HNL", "HRK", "HTG", "HUF", "IDR", "ILS",
                "INR", "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR", "KMF", "KPW",
                "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL", "LYD", "MAD", "MDL",
                "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR", "MVR", "MWK", "MXN", "MYR", "MZN",
                "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR",
                "PLN", "PYG", "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK",
                "SGD", "SHP", "SLE", "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL", "THB", "TJS",
                "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU", "UZS",
                "VES", "VND", "VUV", "WST", "XAF", "XCD", "XOF", "XPF", "YER", "ZAR", "ZMW", "ZWL",
            ],
            "numberingSystem" => &[
                "adlm", "ahom", "arab", "arabext", "bali", "beng", "bhks", "brah", "cakm", "cham",
                "deva", "diak", "fullwide", "gong", "gonm", "gujr", "guru", "hanidec", "hmng",
                "hmnp", "java", "kali", "kawi", "khmr", "knda", "lana", "lanatham", "laoo", "latn",
                "lepc", "limb", "mathbold", "mathdbl", "mathmono", "mathsanb", "mathsans", "mlym",
                "modi", "mong", "mroo", "mtei", "mymr", "mymrshan", "mymrtlng", "nagm", "newa",
                "nkoo", "olck", "orya", "osma", "rohg", "saur", "segment", "shrd", "sind", "sinh",
                "sora", "sund", "takr", "talu", "tamldec", "telu", "thai", "tibt", "tirh", "tnsa",
                "vaii", "wara", "wcho",
            ],
            "timeZone" => &["UTC"],
            "unit" => &[
                "acre",
                "bit",
                "byte",
                "celsius",
                "centimeter",
                "day",
                "degree",
                "fahrenheit",
                "fluid-ounce",
                "foot",
                "gallon",
                "gigabit",
                "gigabyte",
                "gram",
                "hectare",
                "hour",
                "inch",
                "kilobit",
                "kilobyte",
                "kilogram",
                "kilometer",
                "liter",
                "megabit",
                "megabyte",
                "meter",
                "microsecond",
                "mile",
                "mile-scandinavian",
                "milliliter",
                "millimeter",
                "millisecond",
                "minute",
                "month",
                "nanosecond",
                "ounce",
                "percent",
                "petabyte",
                "pound",
                "second",
                "stone",
                "terabit",
                "terabyte",
                "week",
                "yard",
                "year",
            ],
            _ => {
                let m = self.new_str(&alloc::format!("invalid key: {k}"));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        };
        let mut sorted: Vec<&str> = values.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let elems: Vec<NanBox> = sorted.iter().map(|s| self.new_str(s)).collect();
        Ok(NanBox::handle(self.realm.new_array(elems).to_raw()))
    }
}
