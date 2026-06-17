use super::*;

/// The resolved form of `Intl.NumberFormat`'s `useGrouping` option: a boolean
/// `false`, or one of the string modes `"auto"`/`"always"`/`"min2"`.
enum UseGroupingResolved {
    Bool(bool),
    Str(&'static str),
}

/// Whether `s` matches the UTS-35 `type` value production: one or more
/// `alphanum{3,8}` subtags joined by `-` (used for `ca`/`nu` option validation).
fn is_unicode_type_value(s: &str) -> bool {
    !s.is_empty()
        && s.split('-').all(|seg| {
            (3..=8).contains(&seg.len()) && seg.bytes().all(|b| b.is_ascii_alphanumeric())
        })
}

/// Whether `code` is a well-formed ISO-4217 currency code: exactly three ASCII
/// letters (case-insensitive).
fn is_well_formed_currency(code: &str) -> bool {
    code.len() == 3 && code.bytes().all(|b| b.is_ascii_alphabetic())
}

/// The ECMA-402 sanctioned single-unit identifiers (Table: "Single units
/// sanctioned for use in ECMAScript"). Shared by `Intl.supportedValuesOf("unit")`
/// and `IsWellFormedUnitIdentifier`.
pub(crate) const SANCTIONED_UNITS: &[&str] = &[
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
];

/// `IsWellFormedUnitIdentifier(unit)`: a sanctioned single unit, or a
/// `<numerator>-per-<denominator>` compound of two sanctioned single units.
fn is_well_formed_unit(unit: &str) -> bool {
    let valid_single = |u: &str| SANCTIONED_UNITS.contains(&u);
    match unit.split_once("-per-") {
        Some((a, b)) => valid_single(a) && valid_single(b),
        None => valid_single(unit),
    }
}

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
    pub(crate) fn make_intl_formatter(
        &mut self,
        id: u16,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
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
        // CanonicalizeLocaleList(locales): the resolved locale is the first
        // requested tag (this engine serves any structurally valid locale), else
        // the default. A malformed tag raises a RangeError here.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = requested
            .into_iter()
            .next()
            .unwrap_or_else(|| String::from("en-US"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // Coerce options: `undefined` → no options; otherwise ToObject (a
        // primitive other than undefined is wrapped, exposing no option keys).
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else {
            self.coerce_to_object(opts_arg)
                .as_handle()
                .map(Handle::from_raw)
        };
        if id == N_INTL_NUMBER_FORMAT {
            self.init_number_format(obj, opts)?;
        } else {
            self.init_datetime_format(obj, opts)?;
        }
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// `GetOption(options, prop, "string", values, default)` — reads `prop` via
    /// its getter, returns the default when `undefined`, else coerces to a string
    /// (a Symbol is a TypeError) and validates membership in `values` (a
    /// **RangeError** otherwise). `None` `default` with an absent option yields
    /// `None`.
    fn get_string_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
        values: &[&str],
        default: Option<&str>,
    ) -> Result<Option<String>, ExecError> {
        let raw = match opts {
            Some(h) => self.read_member(h, prop)?,
            None => NanBox::undefined(),
        };
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(default.map(String::from));
        }
        let s = self.coerce_to_string(raw)?;
        if !values.is_empty() && !values.iter().any(|v| *v == s) {
            let m = self.new_str(&alloc::format!("invalid value '{s}' for option {prop}"));
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(Some(s))
    }

    /// `GetOption(options, prop, "boolean", …)` — reads `prop`; `undefined` →
    /// `default`, else `ToBoolean`.
    fn get_bool_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
        default: Option<bool>,
    ) -> Result<Option<bool>, ExecError> {
        let raw = match opts {
            Some(h) => self.read_member(h, prop)?,
            None => NanBox::undefined(),
        };
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(default);
        }
        Ok(Some(self.realm.truthy(raw)))
    }

    /// `GetNumberOption` / `DefaultNumberOption`: reads `prop`, coerces to a
    /// Number (throwing for a Symbol), and requires it to be a finite integer in
    /// `[min, max]` (else **RangeError**). `undefined` → `default`.
    fn get_int_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
        min: f64,
        max: f64,
        default: Option<f64>,
    ) -> Result<Option<f64>, ExecError> {
        let raw = match opts {
            Some(h) => self.read_member(h, prop)?,
            None => NanBox::undefined(),
        };
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(default);
        }
        let nv = self.coerce_to_number(raw)?;
        let n = self.realm.to_number(nv);
        if n.is_nan() || n < min || n > max {
            let m = self.new_str(&alloc::format!("value out of range for option {prop}"));
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(Some(trunc_toward_zero(n)))
    }

    /// Stores a resolved option on the formatter (under its own key) when present.
    fn store_str(&mut self, obj: Handle, key: &str, val: &Option<String>) {
        if let Some(v) = val {
            let sv = self.new_str(v);
            self.realm.set_hidden_property(obj, key, sv);
        }
    }

    /// Initializes an `Intl.NumberFormat`, reading and validating options in spec
    /// order (`InitializeNumberFormat` → `SetNumberFormatUnitOptions` →
    /// `SetNumberFormatDigitOptions`). Stores resolved options on `obj`.
    fn init_number_format(&mut self, obj: Handle, opts: Option<Handle>) -> Result<(), ExecError> {
        // localeMatcher (validated, not otherwise used).
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let nu = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(ns) = &nu {
            // numberingSystem must match `type` production (3-8 alnum, hyphen-joined).
            if !is_unicode_type_value(ns) {
                let m = self.new_str("invalid numberingSystem");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        self.store_str(obj, "numberingSystem", &nu);

        // --- SetNumberFormatUnitOptions ---
        let style = self
            .get_string_option(
                opts,
                "style",
                &["decimal", "percent", "currency", "unit"],
                Some("decimal"),
            )?
            .unwrap();
        let currency = self.get_string_option(opts, "currency", &[], None)?;
        let currency_display = self.get_string_option(
            opts,
            "currencyDisplay",
            &["code", "symbol", "narrowSymbol", "name"],
            Some("symbol"),
        )?;
        let currency_sign = self.get_string_option(
            opts,
            "currencySign",
            &["standard", "accounting"],
            Some("standard"),
        )?;
        let unit = self.get_string_option(opts, "unit", &[], None)?;
        let unit_display = self.get_string_option(
            opts,
            "unitDisplay",
            &["short", "narrow", "long"],
            Some("short"),
        )?;
        // SetNumberFormatUnitOptions: a `currency` is required (TypeError) when
        // style is "currency"; whenever present it must be well-formed (RangeError)
        // regardless of style. Same shape for `unit`.
        match &currency {
            None if style == "currency" => {
                let m = self.new_str("currency code is required with currency style");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            Some(c) if !is_well_formed_currency(c) => {
                let m = self.new_str("invalid currency code");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            _ => {}
        }
        match &unit {
            None if style == "unit" => {
                let m = self.new_str("unit is required with unit style");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            Some(u) if !is_well_formed_unit(u) => {
                let m = self.new_str("invalid unit");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            _ => {}
        }
        let style_s = Some(style.clone());
        self.store_str(obj, "style", &style_s);
        // Canonical currency code is uppercased.
        if style == "currency" {
            let cc = currency.as_ref().map(|c| c.to_ascii_uppercase());
            self.store_str(obj, "currency", &cc);
            self.store_str(obj, "currencyDisplay", &currency_display);
            self.store_str(obj, "currencySign", &currency_sign);
        }
        if style == "unit" {
            self.store_str(obj, "unit", &unit);
            self.store_str(obj, "unitDisplay", &unit_display);
        }

        // notation (read before digit options per the spec order).
        let notation = self
            .get_string_option(
                opts,
                "notation",
                &["standard", "scientific", "engineering", "compact"],
                Some("standard"),
            )?
            .unwrap();

        // --- SetNumberFormatDigitOptions ---
        let mnid = self
            .get_int_option(opts, "minimumIntegerDigits", 1.0, 21.0, Some(1.0))?
            .unwrap();
        let mnfd = self.get_int_option(opts, "minimumFractionDigits", 0.0, 100.0, None)?;
        let mxfd = self.get_int_option(opts, "maximumFractionDigits", 0.0, 100.0, None)?;
        let mnsd = self.get_int_option(opts, "minimumSignificantDigits", 1.0, 21.0, None)?;
        let mxsd = self.get_int_option(opts, "maximumSignificantDigits", 1.0, 21.0, None)?;
        // Cross-validation: an explicit minimum may not exceed an explicit maximum.
        if let (Some(a), Some(b)) = (mnfd, mxfd)
            && a > b
        {
            let m = self.new_str("minimumFractionDigits is greater than maximumFractionDigits");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        if let (Some(a), Some(b)) = (mnsd, mxsd)
            && a > b
        {
            let m =
                self.new_str("minimumSignificantDigits is greater than maximumSignificantDigits");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        // roundingIncrement ∈ a fixed allowed set.
        let rinc = self
            .get_int_option(opts, "roundingIncrement", 1.0, 5000.0, Some(1.0))?
            .unwrap();
        const ALLOWED_INC: [u32; 15] = [
            1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000,
        ];
        if !ALLOWED_INC.contains(&(rinc as u32)) {
            let m = self.new_str("invalid roundingIncrement");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let rounding_mode = self
            .get_string_option(
                opts,
                "roundingMode",
                &[
                    "ceil",
                    "floor",
                    "expand",
                    "trunc",
                    "halfCeil",
                    "halfFloor",
                    "halfExpand",
                    "halfTrunc",
                    "halfEven",
                ],
                Some("halfExpand"),
            )?
            .unwrap();
        let rounding_priority = self
            .get_string_option(
                opts,
                "roundingPriority",
                &["auto", "morePrecision", "lessPrecision"],
                Some("auto"),
            )?
            .unwrap();
        let tzd = self
            .get_string_option(
                opts,
                "trailingZeroDisplay",
                &["auto", "stripIfInteger"],
                Some("auto"),
            )?
            .unwrap();

        // compactDisplay, useGrouping, signDisplay.
        let compact_display =
            self.get_string_option(opts, "compactDisplay", &["short", "long"], Some("short"))?;
        if notation == "compact" {
            self.store_str(obj, "compactDisplay", &compact_display);
        }
        // useGrouping: ECMA-402 accepts a boolean or "min2"/"auto"/"always". Read
        // it without enum validation, then normalize.
        let ug_raw = match opts {
            Some(h) => self.read_member(h, "useGrouping")?,
            None => NanBox::undefined(),
        };
        let use_grouping_val = self.normalize_use_grouping(ug_raw)?;
        let sign_display = self
            .get_string_option(
                opts,
                "signDisplay",
                &["auto", "never", "always", "exceptZero", "negative"],
                Some("auto"),
            )?
            .unwrap();

        // Store digit/notation/sign options for resolvedOptions + formatting.
        self.realm
            .set_hidden_property(obj, "minimumIntegerDigits", NanBox::number(mnid));
        if let Some(v) = mnfd {
            self.realm
                .set_hidden_property(obj, "minimumFractionDigits", NanBox::number(v));
        }
        if let Some(v) = mxfd {
            self.realm
                .set_hidden_property(obj, "maximumFractionDigits", NanBox::number(v));
        }
        if let Some(v) = mnsd {
            self.realm
                .set_hidden_property(obj, "minimumSignificantDigits", NanBox::number(v));
        }
        if let Some(v) = mxsd {
            self.realm
                .set_hidden_property(obj, "maximumSignificantDigits", NanBox::number(v));
        }
        self.realm
            .set_hidden_property(obj, "roundingIncrement", NanBox::number(rinc));
        self.store_str(obj, "notation", &Some(notation));
        self.store_str(obj, "roundingMode", &Some(rounding_mode));
        self.store_str(obj, "roundingPriority", &Some(rounding_priority));
        self.store_str(obj, "trailingZeroDisplay", &Some(tzd));
        self.store_str(obj, "signDisplay", &Some(sign_display));
        match use_grouping_val {
            UseGroupingResolved::Bool(b) => {
                self.realm
                    .set_hidden_property(obj, "useGrouping", NanBox::boolean(b));
            }
            UseGroupingResolved::Str(s) => {
                let sv = self.new_str(s);
                self.realm.set_hidden_property(obj, "useGrouping", sv);
            }
        }
        Ok(())
    }

    /// `GetBooleanOrStringNumberFormatOption` for `useGrouping`: `undefined` →
    /// `"auto"`; `true` → `"always"`; a falsy value → `false`; the strings
    /// `"true"`/`"false"` → `"auto"`; one of `"min2"`/`"auto"`/`"always"` → that
    /// string; any other string (or non-string, via ToString) → **RangeError**.
    fn normalize_use_grouping(&mut self, raw: NanBox) -> Result<UseGroupingResolved, ExecError> {
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(UseGroupingResolved::Str("auto"));
        }
        if matches!(raw.unpack(), Unpacked::Bool(true)) {
            return Ok(UseGroupingResolved::Str("always"));
        }
        if !self.realm.truthy(raw) {
            return Ok(UseGroupingResolved::Bool(false));
        }
        let s = self.coerce_to_string(raw)?;
        match s.as_str() {
            "true" | "false" => Ok(UseGroupingResolved::Str("auto")),
            "min2" => Ok(UseGroupingResolved::Str("min2")),
            "auto" => Ok(UseGroupingResolved::Str("auto")),
            "always" => Ok(UseGroupingResolved::Str("always")),
            _ => {
                let m = self.new_str(&alloc::format!("invalid useGrouping value '{s}'"));
                Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))))
            }
        }
    }

    /// Initializes an `Intl.DateTimeFormat`, reading and validating options in
    /// spec order. Stores resolved options on `obj`.
    fn init_datetime_format(&mut self, obj: Handle, opts: Option<Handle>) -> Result<(), ExecError> {
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let ca = self.get_string_option(opts, "calendar", &[], None)?;
        if let Some(c) = &ca
            && !is_unicode_type_value(c)
        {
            let m = self.new_str("invalid calendar");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        self.store_str(obj, "calendar", &ca);
        let nu = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(n) = &nu
            && !is_unicode_type_value(n)
        {
            let m = self.new_str("invalid numberingSystem");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        self.store_str(obj, "numberingSystem", &nu);
        // hour12 (boolean) and hourCycle (enum).
        let hour12 = self.get_bool_option(opts, "hour12", None)?;
        if let Some(b) = hour12 {
            self.realm
                .set_hidden_property(obj, "hour12", NanBox::boolean(b));
        }
        let hc = self.get_string_option(opts, "hourCycle", &["h11", "h12", "h23", "h24"], None)?;
        self.store_str(obj, "hourCycle", &hc);
        let _tz = self.get_string_option(opts, "timeZone", &[], None)?;
        self.store_str(obj, "timeZone", &_tz);
        // weekday/era/year/month/day/hour/minute/second/… component options.
        let nv = ["numeric", "2-digit"];
        let nm = ["long", "short", "narrow"];
        let weekday = self.get_string_option(opts, "weekday", &nm, None)?;
        self.store_str(obj, "weekday", &weekday);
        let era = self.get_string_option(opts, "era", &nm, None)?;
        self.store_str(obj, "era", &era);
        let year = self.get_string_option(opts, "year", &nv, None)?;
        self.store_str(obj, "year", &year);
        let month = self.get_string_option(
            opts,
            "month",
            &["numeric", "2-digit", "long", "short", "narrow"],
            None,
        )?;
        self.store_str(obj, "month", &month);
        let day = self.get_string_option(opts, "day", &nv, None)?;
        self.store_str(obj, "day", &day);
        let day_period = self.get_string_option(opts, "dayPeriod", &nm, None)?;
        self.store_str(obj, "dayPeriod", &day_period);
        let hour = self.get_string_option(opts, "hour", &nv, None)?;
        self.store_str(obj, "hour", &hour);
        let minute = self.get_string_option(opts, "minute", &nv, None)?;
        self.store_str(obj, "minute", &minute);
        let second = self.get_string_option(opts, "second", &nv, None)?;
        self.store_str(obj, "second", &second);
        let fsd = self.get_int_option(opts, "fractionalSecondDigits", 1.0, 3.0, None)?;
        if let Some(v) = fsd {
            self.realm
                .set_hidden_property(obj, "fractionalSecondDigits", NanBox::number(v));
        }
        let tzn = self.get_string_option(
            opts,
            "timeZoneName",
            &[
                "long",
                "short",
                "shortOffset",
                "longOffset",
                "shortGeneric",
                "longGeneric",
            ],
            None,
        )?;
        self.store_str(obj, "timeZoneName", &tzn);
        // formatMatcher (validated, unused).
        let _ = self.get_string_option(
            opts,
            "formatMatcher",
            &["basic", "best fit"],
            Some("best fit"),
        )?;
        let date_style = self.get_string_option(
            opts,
            "dateStyle",
            &["full", "long", "medium", "short"],
            None,
        )?;
        self.store_str(obj, "dateStyle", &date_style);
        let time_style = self.get_string_option(
            opts,
            "timeStyle",
            &["full", "long", "medium", "short"],
            None,
        )?;
        self.store_str(obj, "timeStyle", &time_style);
        // dateStyle/timeStyle are mutually exclusive with explicit component fields.
        if (date_style.is_some() || time_style.is_some())
            && (weekday.is_some()
                || era.is_some()
                || year.is_some()
                || month.is_some()
                || day.is_some()
                || hour.is_some()
                || minute.is_some()
                || second.is_some()
                || day_period.is_some()
                || fsd.is_some()
                || tzn.is_some())
        {
            let m = self.new_str("dateStyle/timeStyle may not be combined with component options");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    /// `Intl.NumberFormat`/`DateTimeFormat` `resolvedOptions()` — a fresh object
    /// reporting the resolved configuration, in spec property order. `fmt` is the
    /// formatter instance (`None` → a default decimal NumberFormat shape).
    pub(crate) fn intl_resolved_options(&mut self, fmt: Option<Handle>) -> NanBox {
        let out = self.realm.new_object();
        let kind = fmt
            .and_then(|h| self.realm.get_property(h, "\u{0}intl"))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("number"));
        let get_str = |this: &Self, key: &str| -> Option<String> {
            fmt.and_then(|h| this.realm.get_property(h, key))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let get_num = |this: &Self, key: &str| -> Option<f64> {
            fmt.and_then(|h| this.realm.get_property(h, key))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_number(v))
        };
        let locale = get_str(self, "\u{0}locale").unwrap_or_else(|| String::from("en-US"));
        let lv = self.new_str(&locale);
        self.realm.set_property(out, "locale", lv);

        if kind == "number" {
            let ns = get_str(self, "numberingSystem").unwrap_or_else(|| String::from("latn"));
            let nsv = self.new_str(&ns);
            self.realm.set_property(out, "numberingSystem", nsv);
            let style = get_str(self, "style").unwrap_or_else(|| String::from("decimal"));
            let sv = self.new_str(&style);
            self.realm.set_property(out, "style", sv);
            if style == "currency" {
                if let Some(c) = get_str(self, "currency") {
                    let cv = self.new_str(&c);
                    self.realm.set_property(out, "currency", cv);
                }
                let cd = get_str(self, "currencyDisplay").unwrap_or_else(|| String::from("symbol"));
                let cdv = self.new_str(&cd);
                self.realm.set_property(out, "currencyDisplay", cdv);
                let cs = get_str(self, "currencySign").unwrap_or_else(|| String::from("standard"));
                let csv = self.new_str(&cs);
                self.realm.set_property(out, "currencySign", csv);
            }
            if style == "unit" {
                if let Some(u) = get_str(self, "unit") {
                    let uv = self.new_str(&u);
                    self.realm.set_property(out, "unit", uv);
                }
                let ud = get_str(self, "unitDisplay").unwrap_or_else(|| String::from("short"));
                let udv = self.new_str(&ud);
                self.realm.set_property(out, "unitDisplay", udv);
            }
            // Digit options. Resolve per SetNumberFormatDigitOptions: significant
            // digits (if requested) are reported; otherwise fraction digits with
            // style-derived defaults. roundingPriority "auto" with neither set
            // reports fraction digits only.
            let mnid = get_num(self, "minimumIntegerDigits").unwrap_or(1.0);
            self.realm
                .set_property(out, "minimumIntegerDigits", NanBox::number(mnid));
            let mnsd = get_num(self, "minimumSignificantDigits");
            let mxsd = get_num(self, "maximumSignificantDigits");
            let mnfd_opt = get_num(self, "minimumFractionDigits");
            let mxfd_opt = get_num(self, "maximumFractionDigits");
            let priority =
                get_str(self, "roundingPriority").unwrap_or_else(|| String::from("auto"));
            let has_sig = mnsd.is_some() || mxsd.is_some();
            let (def_min, def_max): (f64, f64) = match style.as_str() {
                "currency" => (2.0, 2.0),
                "percent" => (0.0, 0.0),
                _ => (0.0, 3.0),
            };
            let report_frac = |this: &mut Self, out: Handle| {
                let mnfd = mnfd_opt.unwrap_or(def_min);
                let mxfd = mxfd_opt.unwrap_or_else(|| def_max.max(mnfd));
                this.realm
                    .set_property(out, "minimumFractionDigits", NanBox::number(mnfd));
                this.realm
                    .set_property(out, "maximumFractionDigits", NanBox::number(mxfd));
            };
            let report_sig = |this: &mut Self, out: Handle| {
                let mnsd = mnsd.unwrap_or(1.0);
                let mxsd = mxsd.unwrap_or(21.0);
                this.realm
                    .set_property(out, "minimumSignificantDigits", NanBox::number(mnsd));
                this.realm
                    .set_property(out, "maximumSignificantDigits", NanBox::number(mxsd));
            };
            if priority == "morePrecision" || priority == "lessPrecision" {
                // Both groups present.
                report_frac(self, out);
                report_sig(self, out);
            } else if has_sig {
                report_sig(self, out);
            } else {
                // Significant digits absent → fraction digits (with or without
                // explicit values, defaulted by style).
                report_frac(self, out);
            }

            let ug = fmt
                .and_then(|h| self.realm.get_property(h, "useGrouping"))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .unwrap_or_else(|| self.new_str("auto"));
            self.realm.set_property(out, "useGrouping", ug);
            let notation = get_str(self, "notation").unwrap_or_else(|| String::from("standard"));
            let nv = self.new_str(&notation);
            self.realm.set_property(out, "notation", nv);
            if notation == "compact" {
                let cd = get_str(self, "compactDisplay").unwrap_or_else(|| String::from("short"));
                let cdv = self.new_str(&cd);
                self.realm.set_property(out, "compactDisplay", cdv);
            }
            let sd = get_str(self, "signDisplay").unwrap_or_else(|| String::from("auto"));
            let sdv = self.new_str(&sd);
            self.realm.set_property(out, "signDisplay", sdv);
            let rinc = get_num(self, "roundingIncrement").unwrap_or(1.0);
            self.realm
                .set_property(out, "roundingIncrement", NanBox::number(rinc));
            let rm = get_str(self, "roundingMode").unwrap_or_else(|| String::from("halfExpand"));
            let rmv = self.new_str(&rm);
            self.realm.set_property(out, "roundingMode", rmv);
            let rp = self.new_str(&priority);
            self.realm.set_property(out, "roundingPriority", rp);
            let tzd = get_str(self, "trailingZeroDisplay").unwrap_or_else(|| String::from("auto"));
            let tzv = self.new_str(&tzd);
            self.realm.set_property(out, "trailingZeroDisplay", tzv);
        } else {
            // DateTimeFormat resolvedOptions.
            let ns = get_str(self, "numberingSystem").unwrap_or_else(|| String::from("latn"));
            let nsv = self.new_str(&ns);
            self.realm.set_property(out, "numberingSystem", nsv);
            let cal = get_str(self, "calendar").unwrap_or_else(|| String::from("gregory"));
            let cv = self.new_str(&cal);
            self.realm.set_property(out, "calendar", cv);
            let tz = get_str(self, "timeZone").unwrap_or_else(|| String::from("UTC"));
            let tzv = self.new_str(&tz);
            self.realm.set_property(out, "timeZone", tzv);
            // Component options that were set, plus hourCycle/hour12.
            if let Some(hc) = get_str(self, "hourCycle") {
                let v = self.new_str(&hc);
                self.realm.set_property(out, "hourCycle", v);
                let h12 = matches!(hc.as_str(), "h11" | "h12");
                self.realm.set_property(out, "hour12", NanBox::boolean(h12));
            } else if let Some(h) = fmt.and_then(|h| self.realm.get_property(h, "hour12")) {
                self.realm.set_property(out, "hour12", h);
            }
            for key in [
                "weekday",
                "era",
                "year",
                "month",
                "day",
                "dayPeriod",
                "hour",
                "minute",
                "second",
                "timeZoneName",
                "dateStyle",
                "timeStyle",
            ] {
                if let Some(v) = get_str(self, key) {
                    let vv = self.new_str(&v);
                    self.realm.set_property(out, key, vv);
                }
            }
            if let Some(v) = get_num(self, "fractionalSecondDigits") {
                self.realm
                    .set_property(out, "fractionalSecondDigits", NanBox::number(v));
            }
        }
        NanBox::handle(out.to_raw())
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
            "unit" => SANCTIONED_UNITS,
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
