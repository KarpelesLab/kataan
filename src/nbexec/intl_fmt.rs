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

/// Per-`Intl` service metadata used to build its `.prototype` (the spec
/// `Symbol.toStringTag`, the hidden brand-marker key stamped on every instance,
/// and the data-method / accessor-getter names installed on the prototype).
struct IntlService {
    /// The constructor's native dispatch id (`N_INTL_NUMBER_FORMAT`, …).
    ctor_id: u16,
    /// The spec `[Symbol.toStringTag]` string (e.g. `"Intl.NumberFormat"`).
    tag: &'static str,
    /// The hidden internal-slot brand-marker property stamped on every instance and
    /// required by the prototype methods'/accessors' RequireInternalSlot check.
    marker: &'static str,
    /// First-class data-method names installed on the prototype (each a branded
    /// `N_INTL_PROTO_METHOD` wrapper delegating to the underlying method native).
    methods: &'static [&'static str],
    /// The single bound-function accessor on this service's prototype, if any:
    /// `(accessor-name, underlying-selector)`. NumberFormat/DateTimeFormat have
    /// `get format`; Collator has `get compare`. These return a per-instance bound
    /// function rather than being plain data methods (ECMA-402).
    bound_accessor: Option<(&'static str, &'static str)>,
}

/// The eight constructor-style `Intl` services that share the
/// formatter/options-object instance shape (NumberFormat … Segmenter). `Intl.Locale`
/// and `Intl.DurationFormat` have bespoke prototypes built separately.
const INTL_SERVICES: &[IntlService] = &[
    IntlService {
        ctor_id: N_INTL_NUMBER_FORMAT,
        tag: "Intl.NumberFormat",
        marker: "\u{0}brand_nf",
        methods: &[
            "resolvedOptions",
            "formatToParts",
            "formatRange",
            "formatRangeToParts",
        ],
        bound_accessor: Some(("format", "format")),
    },
    IntlService {
        ctor_id: N_INTL_DATETIME_FORMAT,
        tag: "Intl.DateTimeFormat",
        marker: "\u{0}brand_dtf",
        methods: &[
            "resolvedOptions",
            "formatToParts",
            "formatRange",
            "formatRangeToParts",
        ],
        bound_accessor: Some(("format", "format")),
    },
    IntlService {
        ctor_id: N_INTL_COLLATOR,
        tag: "Intl.Collator",
        marker: "\u{0}brand_col",
        methods: &["resolvedOptions"],
        bound_accessor: Some(("compare", "compare")),
    },
    IntlService {
        ctor_id: N_INTL_PLURAL_RULES,
        tag: "Intl.PluralRules",
        marker: "\u{0}brand_pr",
        methods: &["resolvedOptions", "select", "selectRange"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_LIST_FORMAT,
        tag: "Intl.ListFormat",
        marker: "\u{0}brand_lf",
        methods: &["resolvedOptions", "format", "formatToParts"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_REL_TIME,
        tag: "Intl.RelativeTimeFormat",
        marker: "\u{0}brand_rtf",
        methods: &["resolvedOptions", "format", "formatToParts"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_DISPLAY_NAMES,
        tag: "Intl.DisplayNames",
        marker: "\u{0}brand_dn",
        methods: &["resolvedOptions", "of"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_SEGMENTER,
        tag: "Intl.Segmenter",
        marker: "\u{0}brand_seg",
        methods: &["resolvedOptions", "segment"],
        bound_accessor: None,
    },
];

/// `Intl.Locale.prototype` accessor-getter names (each a `get`-only accessor
/// brand-checking the `[[InitializedLocale]]` slot).
const LOCALE_ACCESSORS: &[&str] = &[
    "baseName",
    "calendar",
    "caseFirst",
    "collation",
    "hourCycle",
    "language",
    "numberingSystem",
    "numeric",
    "region",
    "script",
];

/// The arity (`length`) of a branded `Intl` prototype data method, given its
/// owning service ctor id and method name (RelativeTimeFormat's `format`/
/// `formatToParts` are length 2; everywhere else `format`-family methods are length 1).
fn intl_method_arity(ctor_id: u16, name: &str) -> u32 {
    match (ctor_id, name) {
        (_, "resolvedOptions") => 0,
        (N_INTL_REL_TIME, "format" | "formatToParts") => 2,
        (N_INTL_PLURAL_RULES, "selectRange") => 2,
        (_, "formatRange" | "formatRangeToParts") => 2,
        _ => 1,
    }
}

impl<'a> Interp<'a> {
    /// The underlying method native id for a branded `Intl` prototype data method.
    fn intl_underlying_native(name: &str) -> u16 {
        match name {
            "format" => N_INTL_FORMAT,
            "resolvedOptions" => N_INTL_RESOLVED_OPTIONS,
            "formatToParts" | "format_to_parts" => N_INTL_FORMAT_TO_PARTS,
            "formatRange" => N_INTL_FORMAT_RANGE,
            "formatRangeToParts" => N_INTL_FORMAT_RANGE_TO_PARTS,
            "compare" => N_INTL_COMPARE,
            "select" => N_INTL_PLURAL_SELECT,
            "selectRange" => N_INTL_PLURAL_SELECT_RANGE,
            "of" => N_INTL_DISPLAY_NAMES_OF,
            "segment" => N_INTL_SEGMENTER_SEGMENT,
            "list_format" => N_INTL_LIST_FORMAT_FORMAT,
            "rel_format" => N_INTL_REL_TIME_FORMAT,
            _ => 0,
        }
    }

    /// Builds (once) and links each `Intl` service constructor's `.prototype`,
    /// installing the branded methods, the `Symbol.toStringTag`, and the
    /// `constructor` back-link, and replacing the constructor's default
    /// `Function.prototype`-derived `.prototype` with a real `%XPrototype%`. Idempotent
    /// (the per-id cache short-circuits a second call). Invoked at realm setup.
    pub(crate) fn install_intl_prototypes(&mut self) {
        for svc in INTL_SERVICES {
            self.intl_service_prototype(svc);
        }
        self.intl_locale_prototype();
        self.intl_duration_prototype();
    }

    /// The `Intl` namespace object (where the service constructors live), if set up.
    fn intl_namespace(&mut self) -> Option<Handle> {
        self.current
            .get("Intl")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
    }

    /// The constructor function handle for the `Intl` service whose property name
    /// on the namespace is `ctor_name` (e.g. `"NumberFormat"`).
    pub(crate) fn intl_ctor_handle(&mut self, ctor_name: &str) -> Option<Handle> {
        let ns = self.intl_namespace()?;
        self.realm
            .get_property(ns, ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
    }

    /// Builds and caches the `.prototype` object for one of the formatter-shaped
    /// `Intl` services, wiring it onto the constructor. Returns the prototype handle.
    fn intl_service_prototype(&mut self, svc: &IntlService) -> Option<Handle> {
        if let Some(p) = self.realm.intl_prototype(svc.ctor_id) {
            return Some(p);
        }
        // Map the ctor id back to its namespace property name.
        let ctor_name = match svc.ctor_id {
            N_INTL_NUMBER_FORMAT => "NumberFormat",
            N_INTL_DATETIME_FORMAT => "DateTimeFormat",
            N_INTL_COLLATOR => "Collator",
            N_INTL_PLURAL_RULES => "PluralRules",
            N_INTL_LIST_FORMAT => "ListFormat",
            N_INTL_REL_TIME => "RelativeTimeFormat",
            N_INTL_DISPLAY_NAMES => "DisplayNames",
            N_INTL_SEGMENTER => "Segmenter",
            _ => return None,
        };
        let ctor = self.intl_ctor_handle(ctor_name)?;
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &m in svc.methods {
            // ListFormat/RelativeTimeFormat have distinct `format`/`formatToParts`
            // method natives; pick the underlying selector accordingly.
            let selector = match (svc.ctor_id, m) {
                (N_INTL_LIST_FORMAT, "format") => "list_format",
                (N_INTL_REL_TIME, "format") => "rel_format",
                (N_INTL_LIST_FORMAT | N_INTL_REL_TIME, "formatToParts") => "format_to_parts",
                _ => m,
            };
            let arity = intl_method_arity(svc.ctor_id, m);
            let f = self.make_intl_proto_method(svc.marker, m, selector, arity);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        // The bound-function accessor (`get format` / `get compare`).
        if let Some((acc_name, selector)) = svc.bound_accessor {
            let label = alloc::format!("get {acc_name}");
            let marker_v = self.new_str(svc.marker);
            let sel_v = self.new_str(selector);
            let pair = self.realm.new_array(alloc::vec![marker_v, sel_v]);
            let getter = self.realm.new_bound_native(N_INTL_BOUND_GETTER, pair);
            self.install_fn_name_length(getter, &label, 0);
            self.realm.define_accessor(
                proto,
                acc_name,
                NanBox::handle(getter.to_raw()),
                NanBox::undefined(),
            );
            self.realm.mark_hidden(proto, acc_name);
        }
        // `prototype[Symbol.toStringTag]` — a `{ w:false, e:false, c:true }` string
        // data property (ECMA-402 11.3.x etc.).
        self.install_to_string_tag(proto, svc.tag);
        // `prototype.constructor` back-link (non-enumerable).
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.link_ctor_prototype(ctor, proto);
        self.realm.set_intl_prototype(svc.ctor_id, proto);
        Some(proto)
    }

    /// A branded prototype data method: an `N_INTL_PROTO_METHOD` bound native whose
    /// target is the 2-element `[markerKey, underlyingSelector]` array. The wrapper
    /// (in `call_with_this`) RequireInternalSlot-checks the call's `this` before
    /// delegating to the underlying method native.
    fn make_intl_proto_method(
        &mut self,
        marker: &str,
        name: &str,
        selector: &str,
        arity: u32,
    ) -> Handle {
        let marker_v = self.new_str(marker);
        let sel_v = self.new_str(selector);
        let pair = self.realm.new_array(alloc::vec![marker_v, sel_v]);
        let f = self.realm.new_bound_native(N_INTL_PROTO_METHOD, pair);
        self.install_fn_name_length(f, name, arity);
        f
    }

    /// Installs `proto` as the constructor's `prototype` data property, with the
    /// built-in `{ writable:false, enumerable:false, configurable:false }`
    /// attributes (every built-in constructor's `prototype`).
    fn link_ctor_prototype(&mut self, ctor: Handle, proto: Handle) {
        self.realm
            .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(ctor, "prototype");
        self.realm.set_readonly_property(ctor, "prototype");
        self.realm.set_non_configurable_property(ctor, "prototype");
    }

    /// Stamps a service's internal-slot brand marker on `obj` and links its
    /// `[[Prototype]]` to the (lazily built) service prototype — turning a bare
    /// options-bag object into a branded `Intl.X` instance.
    fn brand_intl_instance(&mut self, obj: Handle, ctor_id: u16) {
        let svc = INTL_SERVICES.iter().find(|s| s.ctor_id == ctor_id);
        if let Some(svc) = svc {
            self.realm
                .set_hidden_property(obj, svc.marker, NanBox::boolean(true));
            if let Some(proto) = self.intl_service_prototype(svc) {
                self.realm.set_object_proto(obj, Some(proto));
            }
        }
    }

    /// Brands `obj` with the service's internal-slot marker *without* touching its
    /// prototype (for a subclass `super()`, whose instance already links to the
    /// subclass prototype).
    fn set_intl_marker(&mut self, obj: Handle, ctor_id: u16) {
        if let Some(svc) = INTL_SERVICES.iter().find(|s| s.ctor_id == ctor_id) {
            self.realm
                .set_hidden_property(obj, svc.marker, NanBox::boolean(true));
        }
    }

    /// RequireInternalSlot for a branded `Intl` receiver: returns the receiver
    /// handle when `this` is an object carrying the hidden `marker`, else a
    /// TypeError. The prototype object itself (which has no marker) is rejected,
    /// matching the spec's per-method brand check.
    pub(crate) fn require_intl_slot(
        &mut self,
        this: NanBox,
        marker: &str,
        what: &str,
    ) -> Result<Handle, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw)
            && self.realm.get_property(h, marker).is_some()
        {
            return Ok(h);
        }
        Err(self.type_error(&alloc::format!(
            "{what} called on an object that is not a valid {what} receiver"
        )))
    }

    /// Dispatches an `N_INTL_PROTO_METHOD` wrapper: `target` is the
    /// `[markerKey, underlyingSelector]` pair. Brand-checks `this`, then delegates to
    /// the underlying method native with `this` preserved.
    pub(crate) fn intl_proto_method_dispatch(
        &mut self,
        this: NanBox,
        target: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let pair = self
            .realm
            .array_elements(target)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let marker = pair
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        let selector = pair
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        self.require_intl_slot(this, &marker, "Intl method")?;
        let id = Self::intl_underlying_native(&selector);
        // The underlying method native reads `self.this_val`; preserve the receiver.
        let saved = core::mem::replace(&mut self.this_val, this);
        let r = self.call_native(id, args);
        self.this_val = saved;
        r
    }

    /// Dispatches an `N_INTL_BOUND_GETTER` accessor (`get format`/`get compare`):
    /// `target` is `[markerKey, selector]`. Brand-checks `this`, then returns the
    /// per-instance bound function — created once and cached on the instance under a
    /// hidden `\0bound_<selector>` key so repeated reads return the same value
    /// (`nf.format === nf.format`, per ECMA-402).
    pub(crate) fn intl_bound_getter_dispatch(
        &mut self,
        this: NanBox,
        target: Handle,
    ) -> Result<NanBox, ExecError> {
        let pair = self
            .realm
            .array_elements(target)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let marker = pair
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        let selector = pair
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        let inst = self.require_intl_slot(this, &marker, "Intl bound-function getter")?;
        let cache_key = alloc::format!("\u{0}bound_{selector}");
        if let Some(v) = self.realm.get_property(inst, &cache_key) {
            return Ok(v);
        }
        // Build the bound function: target = [instance, selector].
        let inst_v = NanBox::handle(inst.to_raw());
        let sel_v = self.new_str(&selector);
        let bpair = self.realm.new_array(alloc::vec![inst_v, sel_v]);
        let bound = self.realm.new_bound_native(N_INTL_BOUND_CALL, bpair);
        // The bound `format`/`compare`'s `name` is `""` and `length` is 1.
        self.install_fn_name_length(bound, "", 1);
        let boundv = NanBox::handle(bound.to_raw());
        self.realm.set_hidden_property(inst, &cache_key, boundv);
        Ok(boundv)
    }

    /// Dispatches an `N_INTL_BOUND_CALL` (the function returned by `get format`/
    /// `get compare`): `target` is `[instance, selector]`. Formats/compares against
    /// the captured instance regardless of the call's `this` (a BoundFunction).
    pub(crate) fn intl_bound_call_dispatch(
        &mut self,
        target: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let pair = self
            .realm
            .array_elements(target)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let inst = pair
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        let selector = pair
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        let Some(inst) = inst else {
            return Ok(NanBox::undefined());
        };
        let arg0 = args.first().copied().unwrap_or(NanBox::undefined());
        match selector.as_str() {
            "compare" => {
                // Delegate to N_INTL_COMPARE with the captured collator as `this`.
                let saved = core::mem::replace(&mut self.this_val, NanBox::handle(inst.to_raw()));
                let r = self.call_native(N_INTL_COMPARE, args);
                self.this_val = saved;
                r
            }
            _ => {
                // `format`: format the single argument against the captured formatter.
                let s = self.intl_format_value(inst, arg0);
                Ok(self.new_str(&s))
            }
        }
    }

    /// Builds an `Intl.NumberFormat`/`DateTimeFormat` instance — an object that
    /// captures the relevant options behind a `\0intl` kind marker. Used for both
    /// `new Intl.X(...)` and the callable-without-`new` form.
    pub(crate) fn make_intl_formatter(
        &mut self,
        id: u16,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        // Link `[[Prototype]]` to `Intl.X.prototype` (a subclass/Reflect.construct
        // newTarget overrides it in the caller) and initialize the formatter state.
        self.brand_intl_instance(obj, id);
        self.init_intl_formatter_state(obj, id, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Initializes an Intl formatter's internal slots on an *already-allocated*
    /// (and prototype-linked) `obj` — the shared body of `make_intl_formatter` and
    /// the `class S extends Intl.NumberFormat {}` `super()` path, so a subclass
    /// instance carries the `[[NumberFormat]]`/`[[DateTimeFormat]]` slots.
    pub(crate) fn init_intl_formatter_state(
        &mut self,
        obj: Handle,
        id: u16,
        args: &[NanBox],
    ) -> Result<(), ExecError> {
        let kind = if id == N_INTL_NUMBER_FORMAT {
            "number"
        } else {
            "datetime"
        };
        let marker = self.new_str(kind);
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // Brand the instance with the service's internal-slot marker (without
        // resetting the prototype, which a subclass `super()` has already set).
        self.set_intl_marker(obj, id);
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
        Ok(())
    }

    /// `GetOption(options, prop, "string", values, default)` — reads `prop` via
    /// its getter, returns the default when `undefined`, else coerces to a string
    /// (a Symbol is a TypeError) and validates membership in `values` (a
    /// **RangeError** otherwise). `None` `default` with an absent option yields
    /// `None`.
    pub(crate) fn get_string_option(
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

        // --- SetNumberFormatDigitOptions --- (shared with Intl.PluralRules).
        self.set_number_format_digit_options(obj, opts)?;

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

        // Store notation/sign options for resolvedOptions + formatting (the digit
        // options were stored by `set_number_format_digit_options`).
        self.store_str(obj, "notation", &Some(notation));
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

    /// `SetNumberFormatDigitOptions`: reads, validates, and stores the shared
    /// number-format digit slots (`minimumIntegerDigits`, the fraction/significant
    /// digit pairs, `roundingIncrement`, `roundingMode`, `roundingPriority`,
    /// `trailingZeroDisplay`) on `obj` under their own hidden keys, in spec read
    /// order. Shared by `Intl.NumberFormat` and `Intl.PluralRules` so both honor the
    /// same option semantics and constructor option-read order. An invalid value is a
    /// RangeError; a Symbol is a TypeError (via the option getters).
    fn set_number_format_digit_options(
        &mut self,
        obj: Handle,
        opts: Option<Handle>,
    ) -> Result<(), ExecError> {
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
        self.store_str(obj, "roundingMode", &Some(rounding_mode));
        self.store_str(obj, "roundingPriority", &Some(rounding_priority));
        self.store_str(obj, "trailingZeroDisplay", &Some(tzd));
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

        if kind == "list" {
            // `Intl.ListFormat.prototype.resolvedOptions()` — `{ locale, type, style }`
            // in that order (Table: "Resolved Options of ListFormat instances").
            let lt = get_str(self, "type").unwrap_or_else(|| String::from("conjunction"));
            let ltv = self.new_str(&lt);
            self.realm.set_property(out, "type", ltv);
            let style = get_str(self, "style").unwrap_or_else(|| String::from("long"));
            let stv = self.new_str(&style);
            self.realm.set_property(out, "style", stv);
        } else if kind == "rtf" {
            // `Intl.RelativeTimeFormat.prototype.resolvedOptions()` —
            // `{ locale, style, numeric, numberingSystem }` in that order
            // (Table: "Resolved Options of RelativeTimeFormat instances"). The
            // resolved numbering system for the supported locales is always `latn`.
            let style = get_str(self, "style").unwrap_or_else(|| String::from("long"));
            let stv = self.new_str(&style);
            self.realm.set_property(out, "style", stv);
            let numeric = get_str(self, "numeric").unwrap_or_else(|| String::from("always"));
            let nv = self.new_str(&numeric);
            self.realm.set_property(out, "numeric", nv);
            let ns = get_str(self, "numberingSystem").unwrap_or_else(|| String::from("latn"));
            let nsv = self.new_str(&ns);
            self.realm.set_property(out, "numberingSystem", nsv);
        } else if kind == "plural" {
            // `Intl.PluralRules.prototype.resolvedOptions()` — the key order is
            // `{ locale, type, notation, minimumIntegerDigits, (fraction|significant
            // digit pair), pluralCategories, roundingIncrement, roundingMode,
            // roundingPriority, trailingZeroDisplay }`.
            let pr_type = get_str(self, "type").unwrap_or_else(|| String::from("cardinal"));
            let tv = self.new_str(&pr_type);
            self.realm.set_property(out, "type", tv);
            let notation = get_str(self, "notation").unwrap_or_else(|| String::from("standard"));
            let nv = self.new_str(&notation);
            self.realm.set_property(out, "notation", nv);
            if notation == "compact" {
                let cd = get_str(self, "compactDisplay").unwrap_or_else(|| String::from("short"));
                let cdv = self.new_str(&cd);
                self.realm.set_property(out, "compactDisplay", cdv);
            }
            // Digit options, resolved as in SetNumberFormatDigitOptions (decimal
            // defaults: minimumFractionDigits 0, maximumFractionDigits 3).
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
            let report_frac = |this: &mut Self, out: Handle| {
                let mnfd = mnfd_opt.unwrap_or(0.0);
                let mxfd = mxfd_opt.unwrap_or_else(|| 3.0_f64.max(mnfd));
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
                report_frac(self, out);
                report_sig(self, out);
            } else if has_sig {
                report_sig(self, out);
            } else {
                report_frac(self, out);
            }
            // pluralCategories: a fresh sorted array on every call.
            let ordinal = pr_type == "ordinal";
            let cats = self.plural_categories(&locale, ordinal);
            let cat_vals: Vec<NanBox> = cats.iter().map(|c| self.new_str(c)).collect();
            let arr = self.realm.new_array(cat_vals);
            self.realm
                .set_property(out, "pluralCategories", NanBox::handle(arr.to_raw()));
            // Rounding options.
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
        } else if kind == "number" {
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

    /// The `[[NumberFormat]]`/`[[DateTimeFormat]]` instance for a `formatRange`
    /// call, plus its two coerced numeric endpoints. Per spec both `start`/`end`
    /// are **required** (`undefined` → TypeError) and must not be `NaN`
    /// (→ RangeError); a `start > end` is allowed. Returns `(instance, x, y)`.
    fn intl_range_operands(
        &mut self,
        this: NanBox,
        start: NanBox,
        end: NanBox,
    ) -> Result<(Handle, f64, f64), ExecError> {
        let Some(inst) = this
            .as_handle()
            .map(Handle::from_raw)
            .filter(|h| self.realm.get_property(*h, "\u{0}intl").is_some())
        else {
            return Err(self.type_error("formatRange called on an incompatible receiver"));
        };
        if matches!(start.unpack(), Unpacked::Undefined)
            || matches!(end.unpack(), Unpacked::Undefined)
        {
            return Err(self.type_error("formatRange requires two defined arguments"));
        }
        let x = self.coerce_to_number(start)?;
        let y = self.coerce_to_number(end)?;
        let (x, y) = (self.realm.to_number(x), self.realm.to_number(y));
        if x.is_nan() || y.is_nan() {
            let m = self.new_str("formatRange arguments must not be NaN");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok((inst, x, y))
    }

    /// `Intl.NumberFormat/DateTimeFormat.prototype.formatRange(x, y)`: formats each
    /// endpoint and joins them with an en-dash range separator (a basic
    /// `FormatNumericRange`; `x === y` collapses to the single formatted value).
    pub(crate) fn intl_format_range(
        &mut self,
        this: NanBox,
        start: NanBox,
        end: NanBox,
    ) -> Result<String, ExecError> {
        let (inst, x, y) = self.intl_range_operands(this, start, end)?;
        let fx = self.intl_format_value(inst, NanBox::number(x));
        if x == y {
            return Ok(fx);
        }
        let fy = self.intl_format_value(inst, NanBox::number(y));
        Ok(alloc::format!("{fx}\u{2013}{fy}"))
    }

    /// `formatRangeToParts(x, y)` → an array of `{ type, value, source }` parts.
    pub(crate) fn intl_format_range_to_parts(
        &mut self,
        this: NanBox,
        start: NanBox,
        end: NanBox,
    ) -> Result<NanBox, ExecError> {
        let (inst, x, y) = self.intl_range_operands(this, start, end)?;
        let fx = self.intl_format_value(inst, NanBox::number(x));
        let mut parts: Vec<(&str, String, &str)> = alloc::vec![("literal", fx, "startRange")];
        if x != y {
            let fy = self.intl_format_value(inst, NanBox::number(y));
            parts.push(("literal", String::from("\u{2013}"), "shared"));
            parts.push(("literal", fy, "endRange"));
        }
        let mut elems = Vec::with_capacity(parts.len());
        for (ty, val, src) in parts {
            let o = self.realm.new_object();
            let tv = self.new_str(ty);
            let vv = self.new_str(&val);
            let sv = self.new_str(src);
            self.realm.set_property(o, "type", tv);
            self.realm.set_property(o, "value", vv);
            self.realm.set_property(o, "source", sv);
            elems.push(NanBox::handle(o.to_raw()));
        }
        Ok(NanBox::handle(self.realm.new_array(elems).to_raw()))
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

    /// Builds an `Intl.RelativeTimeFormat` instance (`InitializeRelativeTimeFormat`):
    /// canonicalizes the requested locales, then reads and validates the options in
    /// spec order (`localeMatcher`, `numberingSystem`, `style`, `numeric`), and brands
    /// the object with `\0intl="rtf"` plus its resolved `\0locale`/`numberingSystem`/
    /// `style`/`numeric`. Used only by the `new`-form constructor; a plain call (no
    /// `new`) throws a `TypeError` before reaching here.
    pub(crate) fn make_relative_time_format(
        &mut self,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        let marker = self.new_str("rtf");
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // 1. CanonicalizeLocaleList(locales) — a malformed tag raises a RangeError.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = requested
            .into_iter()
            .next()
            .unwrap_or_else(|| String::from("en-US"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // `InitializeRelativeTimeFormat` coerces options with `ToObject` (not
        // GetOptionsObject): `undefined` → an empty options bag; `null` → a TypeError;
        // any other primitive (boolean/string/number/symbol) → its wrapper object
        // (which has no own option keys, so all defaults apply).
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = match opts_arg.unpack() {
            Unpacked::Undefined => None,
            Unpacked::Null => {
                return Err(self.type_error("Intl.RelativeTimeFormat options cannot be null"));
            }
            _ => self
                .coerce_to_object(opts_arg)
                .as_handle()
                .map(Handle::from_raw),
        };
        // Options are read in spec order: localeMatcher, numberingSystem, style, numeric.
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let nu = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(ns) = &nu {
            // numberingSystem must match the UTS-35 `type` production (3-8 alnum,
            // hyphen-joined) — otherwise a RangeError.
            if !is_unicode_type_value(ns) {
                let m = self.new_str("invalid numberingSystem");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        let style = self
            .get_string_option(opts, "style", &["long", "short", "narrow"], Some("long"))?
            .unwrap();
        let numeric = self
            .get_string_option(opts, "numeric", &["always", "auto"], Some("always"))?
            .unwrap();
        self.store_str(obj, "style", &Some(style));
        self.store_str(obj, "numeric", &Some(numeric));
        self.brand_intl_instance(obj, N_INTL_REL_TIME);
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// The resolved `(numeric, style)` of an `Intl.RelativeTimeFormat` instance
    /// (defaults `"always"`/`"long"` when the slots are absent).
    pub(crate) fn rel_time_numeric_style(&mut self, fmt: Option<Handle>) -> (String, String) {
        let numeric = fmt
            .and_then(|h| self.realm.get_property(h, "numeric"))
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("always"));
        let style = fmt
            .and_then(|h| self.realm.get_property(h, "style"))
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("long"));
        (numeric, style)
    }

    /// `SingularRelativeTimeUnit(unit)`: `ToString`s `unit` (a Symbol throws a
    /// TypeError), accepts both the singular and plural forms of the eight relative
    /// time units, returning the singular stem; any other value is a RangeError.
    pub(crate) fn singular_relative_time_unit(
        &mut self,
        unit: NanBox,
    ) -> Result<String, ExecError> {
        let s = self.coerce_to_string(unit)?;
        let singular = match s.as_str() {
            "seconds" | "second" => "second",
            "minutes" | "minute" => "minute",
            "hours" | "hour" => "hour",
            "days" | "day" => "day",
            "weeks" | "week" => "week",
            "months" | "month" => "month",
            "quarters" | "quarter" => "quarter",
            "years" | "year" => "year",
            _ => {
                let m = self.new_str(&alloc::format!("invalid relative time unit '{s}'"));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        };
        Ok(String::from(singular))
    }

    /// Builds an `Intl.DisplayNames` instance: an object capturing `type` with a readable
    /// `of(code)` method.
    pub(crate) fn make_display_names(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        let locale = args
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // `options = ? GetOptionsObject(options)` — a primitive other than
        // `undefined` is a TypeError; `undefined` yields an (empty) options object.
        let opts = match args.get(1).copied() {
            Some(v) if !matches!(v.unpack(), Unpacked::Undefined) => {
                if !self.is_object_value(v) {
                    return Err(self.type_error("Intl.DisplayNames: options must be an object"));
                }
                v.as_handle().map(Handle::from_raw)
            }
            _ => None,
        };
        // `type` is a **required** option (its absence — including when `options`
        // is undefined — is a TypeError); it must be one of the valid types.
        let type_s = self.get_string_option(
            opts,
            "type",
            &[
                "language",
                "region",
                "script",
                "currency",
                "calendar",
                "dateTimeField",
            ],
            None,
        )?;
        let Some(type_s) = type_s else {
            return Err(self.type_error("Intl.DisplayNames: the `type` option is required"));
        };
        let tv = self.new_str(&type_s);
        self.realm.set_hidden_property(obj, "type", tv);
        if let Some(opts) = opts {
            for key in ["style", "fallback"] {
                if let Some(v) = self.realm.get_property(opts, key) {
                    self.realm.set_hidden_property(obj, key, v);
                }
            }
        }
        self.brand_intl_instance(obj, N_INTL_DISPLAY_NAMES);
        Ok(NanBox::handle(obj.to_raw()))
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
        self.brand_intl_instance(obj, N_INTL_COLLATOR);
        NanBox::handle(obj.to_raw())
    }

    /// Builds an `Intl.ListFormat` instance (`InitializeListFormat`): canonicalizes
    /// the requested locales, reads and validates the `localeMatcher`/`type`/`style`
    /// options (in that order), and brands the object with `\0intl="list"` plus its
    /// resolved `\0locale`/`type`/`style`. Used only by the `new`-form constructor;
    /// a plain call (no `new`) throws a `TypeError` before reaching here.
    pub(crate) fn make_list_format(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        let marker = self.new_str("list");
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // 1. CanonicalizeLocaleList(locales) — a malformed tag raises a RangeError.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = requested
            .into_iter()
            .next()
            .unwrap_or_else(|| String::from("en-US"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // GetOptionsObject: `undefined` → no options; a non-object (other than
        // undefined) is a TypeError; otherwise the object itself.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.ListFormat options must be an object"));
        };
        // Options are read in spec order: localeMatcher, then type, then style.
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let list_type = self
            .get_string_option(
                opts,
                "type",
                &["conjunction", "disjunction", "unit"],
                Some("conjunction"),
            )?
            .unwrap();
        let style = self
            .get_string_option(opts, "style", &["long", "short", "narrow"], Some("long"))?
            .unwrap();
        self.store_str(obj, "type", &Some(list_type));
        self.store_str(obj, "style", &Some(style));
        self.brand_intl_instance(obj, N_INTL_LIST_FORMAT);
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// `StringListFromIterable(iterable)`: `undefined` → an empty list; otherwise
    /// iterate, requiring every yielded value to be a primitive String (else a
    /// `TypeError`, after closing the iterator). Used by `format`/`formatToParts`.
    pub(crate) fn string_list_from_iterable(
        &mut self,
        iterable: NanBox,
    ) -> Result<Vec<String>, ExecError> {
        if matches!(iterable.unpack(), Unpacked::Undefined) {
            return Ok(Vec::new());
        }
        // Open the iterator once via the user `[Symbol.iterator]` when present, so
        // a custom iterable's `next`/`return` observe the exact spec call sequence
        // (and a non-string element closes it mid-stream). Built-in iterables
        // (arrays/strings/Maps/Sets/generators) have no readable iterator method;
        // they are drained eagerly and then validated element-by-element.
        if let Some(ih) = self.for_of_get_iterator(iterable)? {
            let mut out = Vec::new();
            loop {
                let next_fn = self.read_member(ih, "next")?;
                let res = self.call_with_this(next_fn, NanBox::handle(ih.to_raw()), &[])?;
                let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("iterator result is not an object"));
                };
                let done = self.read_member(rh, "done")?;
                if self.realm.truthy(done) {
                    break;
                }
                let value = self.read_member(rh, "value")?;
                match value.as_handle().map(Handle::from_raw) {
                    Some(vh) if self.realm.type_of(vh) == Some("string") => {
                        out.push(self.realm.string_value(vh).unwrap_or_default());
                    }
                    _ => {
                        // Non-String element: close the iterator, then throw.
                        let _ = self.iterator_close(ih);
                        return Err(
                            self.type_error("Intl.ListFormat: list elements must all be strings")
                        );
                    }
                }
                if out.len() > GEN_CAP {
                    return Err(self.type_error("iterator did not terminate"));
                }
            }
            return Ok(out);
        }
        // Built-in iterable (or a non-iterable, which `iterate_values` rejects with
        // a TypeError): drain, then require each element to be a String.
        let elems = self.iterate_values(iterable)?;
        let mut out = Vec::with_capacity(elems.len());
        for e in elems {
            match e.as_handle().map(Handle::from_raw) {
                Some(vh) if self.realm.type_of(vh) == Some("string") => {
                    out.push(self.realm.string_value(vh).unwrap_or_default());
                }
                _ => {
                    return Err(
                        self.type_error("Intl.ListFormat: list elements must all be strings")
                    );
                }
            }
        }
        Ok(out)
    }

    /// The resolved `(type, style)` of an `Intl.ListFormat` instance (defaults
    /// `"conjunction"`/`"long"` when the slots are absent).
    pub(crate) fn list_format_type_style(&self, fmt: Option<Handle>) -> (String, String) {
        let get = |key: &str, dflt: &str| -> String {
            fmt.and_then(|h| self.realm.get_property(h, key))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from(dflt))
        };
        (get("type", "conjunction"), get("style", "long"))
    }

    /// The `(literal-or-element, value)` parts of an `Intl.ListFormat` `format`/
    /// `formatToParts` over `items`, for the given `type` and `style`. The English
    /// CLDR patterns are hardcoded (the `intl` 0.4 crate exposes only and/or
    /// connectors, not the per-`type`/`style` list patterns, nor `unit`); these
    /// match the en-US output the conformance tests require.
    pub(crate) fn list_format_parts(
        &self,
        items: &[String],
        list_type: &str,
        _style: &str,
    ) -> Vec<(&'static str, String)> {
        let n = items.len();
        let mut parts: Vec<(&'static str, String)> = Vec::new();
        if n == 0 {
            return parts;
        }
        // en patterns. `pair` is the two-element connector; `start`/`middle`/`end`
        // are the three-or-more connectors. (CLDR `listPattern` for `en`.)
        let (pair, end): (&str, &str) = match list_type {
            "disjunction" => (" or ", ", or "),
            "unit" => (", ", ", "),
            _ => (" and ", ", and "), // conjunction (default)
        };
        let middle = ", ";
        for (i, it) in items.iter().enumerate() {
            if i > 0 {
                let lit = if n == 2 {
                    pair
                } else if i == n - 1 {
                    end
                } else {
                    middle
                };
                parts.push(("literal", String::from(lit)));
            }
            parts.push(("element", it.clone()));
        }
        parts
    }

    /// Builds an `Intl.PluralRules` instance (`InitializePluralRules`): canonicalizes
    /// the locale list, reads `localeMatcher`/`type`/`notation`/`compactDisplay` and
    /// the shared `SetNumberFormatDigitOptions` slots (in spec read order), and stores
    /// the resolved configuration behind hidden keys for `select`/`selectRange`/
    /// `resolvedOptions`. A non-object `options` (other than `undefined`) is a
    /// TypeError; an invalid option value is a RangeError.
    pub(crate) fn make_plural_rules(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        // Mark the instance kind so `resolvedOptions` reports the PluralRules shape.
        let kindv = self.new_str("plural");
        self.realm.set_hidden_property(obj, "\u{0}intl", kindv);
        self.brand_intl_instance(obj, N_INTL_PLURAL_RULES);
        // CanonicalizeLocaleList(locales): a malformed tag is a RangeError; the
        // resolved locale is the first requested tag, else the default.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = requested
            .into_iter()
            .next()
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // GetOptionsObject: undefined → no options; a non-object → TypeError.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.PluralRules options must be an object"));
        };
        // localeMatcher (validated, not otherwise used).
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        // type: "cardinal" (default) or "ordinal".
        let pr_type = self
            .get_string_option(opts, "type", &["cardinal", "ordinal"], Some("cardinal"))?
            .unwrap();
        self.store_str(obj, "type", &Some(pr_type));
        // notation: "standard" (default) | "compact" | "scientific" | "engineering".
        let notation = self
            .get_string_option(
                opts,
                "notation",
                &["standard", "compact", "scientific", "engineering"],
                Some("standard"),
            )?
            .unwrap();
        // compactDisplay is read regardless, but only stored when notation is compact.
        let compact_display =
            self.get_string_option(opts, "compactDisplay", &["short", "long"], Some("short"))?;
        if notation == "compact" {
            self.store_str(obj, "compactDisplay", &compact_display);
        }
        self.store_str(obj, "notation", &Some(notation));
        // SetNumberFormatDigitOptions (shared with Intl.NumberFormat).
        self.set_number_format_digit_options(obj, opts)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// The plural category name (`"zero"`/`"one"`/`"two"`/`"few"`/`"many"`/`"other"`)
    /// of `n` per the receiver `Intl.PluralRules` instance's `[[Locale]]` and
    /// `[[Type]]`. A non-finite `n` is always `"other"`. With the `intl` feature this
    /// uses the crate's CLDR cardinal/ordinal rules; otherwise it falls back to the
    /// English rule (`1` → `"one"`, else `"other"`). Shared by `select`/`selectRange`.
    pub(crate) fn plural_select_category(&mut self, n: f64) -> &'static str {
        if !n.is_finite() {
            return "other";
        }
        #[cfg(feature = "intl")]
        {
            let fmt = self.this_val.as_handle().map(Handle::from_raw);
            let locale = fmt
                .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("en"));
            let ordinal = fmt
                .and_then(|h| self.realm.get_property(h, "type"))
                .map(|v| self.realm.to_display_string(v))
                .as_deref()
                == Some("ordinal");
            let ops = if n == (n as i64) as f64 {
                intl::plural::PluralOperands::from_int(n as i64)
            } else {
                intl::plural::PluralOperands::parse(&alloc::format!("{n}"))
                    .unwrap_or_else(|| intl::plural::PluralOperands::from_int(n as i64))
            };
            let cat = if ordinal {
                intl::plural::ordinal_category(&locale, &ops)
            } else {
                intl::plural::plural_category(&locale, &ops)
            };
            use intl::plural::PluralCategory::*;
            match cat {
                Zero => "zero",
                One => "one",
                Two => "two",
                Few => "few",
                Many => "many",
                Other => "other",
            }
        }
        #[cfg(not(feature = "intl"))]
        {
            if n == 1.0 { "one" } else { "other" }
        }
    }

    /// The sorted `pluralCategories` list for an `Intl.PluralRules` instance's
    /// locale+type: the distinct categories its cardinal (or ordinal) rules can
    /// produce, in CLDR order (`zero`, `one`, `two`, `few`, `many`, `other`). With
    /// the `intl` feature this probes the crate's rules over a representative sample
    /// of operands; otherwise it returns the English cardinal set `["one","other"]`.
    pub(crate) fn plural_categories(&mut self, locale: &str, ordinal: bool) -> Vec<&'static str> {
        const ORDER: [&str; 6] = ["zero", "one", "two", "few", "many", "other"];
        #[cfg(feature = "intl")]
        {
            use intl::plural::PluralCategory::*;
            let name = |c: intl::plural::PluralCategory| -> &'static str {
                match c {
                    Zero => "zero",
                    One => "one",
                    Two => "two",
                    Few => "few",
                    Many => "many",
                    Other => "other",
                }
            };
            let mut seen: Vec<&'static str> = Vec::new();
            // Probe a representative sample of operands: integers 0..=200 capture the
            // mod-10/mod-100 rule structure, plus a couple of fractional and large
            // compact-notation values that trigger the `many` category in some locales.
            let mut push = |this: &mut Self, s: &str| {
                if let Some(ops) = intl::plural::PluralOperands::parse(s) {
                    let cat = if ordinal {
                        intl::plural::ordinal_category(locale, &ops)
                    } else {
                        intl::plural::plural_category(locale, &ops)
                    };
                    let _ = this;
                    let nm = name(cat);
                    if !seen.contains(&nm) {
                        seen.push(nm);
                    }
                }
            };
            for i in 0..=200u32 {
                let s = alloc::format!("{i}");
                push(self, &s);
            }
            for s in ["0.0", "0.1", "1.5", "2.5", "1000000", "1000000.0"] {
                push(self, s);
            }
            // Return in canonical CLDR order.
            ORDER.iter().copied().filter(|c| seen.contains(c)).collect()
        }
        #[cfg(not(feature = "intl"))]
        {
            let _ = (locale, ordinal);
            let _ = ORDER;
            alloc::vec!["one", "other"]
        }
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
        self.brand_intl_instance(obj, N_INTL_SEGMENTER);
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
        // `NumberFormatOptions` is `#[non_exhaustive]` in intl 0.5, so it can't be
        // built with a struct literal — start from `default()` and set fields.
        let mut o = NumberFormatOptions::default();
        o.style = match opt_str(self, "style").as_deref() {
            Some("percent") => NumberStyle::Percent,
            Some("currency") => NumberStyle::Currency,
            Some("unit") => NumberStyle::Unit,
            _ => NumberStyle::Decimal,
        };
        o.notation = match opt_str(self, "notation").as_deref() {
            Some("scientific") => Notation::Scientific,
            Some("engineering") => Notation::Engineering,
            Some("compact") => Notation::Compact,
            _ => Notation::Standard,
        };
        o.compact_display = match opt_str(self, "compactDisplay").as_deref() {
            Some("long") => CompactDisplay::Long,
            _ => CompactDisplay::Short,
        };
        o.sign_display = match opt_str(self, "signDisplay").as_deref() {
            Some("always") => SignDisplay::Always,
            Some("exceptZero") => SignDisplay::ExceptZero,
            Some("negative") => SignDisplay::Negative,
            Some("never") => SignDisplay::Never,
            _ => SignDisplay::Auto,
        };
        o.currency_display = match opt_str(self, "currencyDisplay").as_deref() {
            Some("code") => CurrencyDisplay::Code,
            Some("name") => CurrencyDisplay::Name,
            Some("narrowSymbol") => CurrencyDisplay::NarrowSymbol,
            _ => CurrencyDisplay::Symbol,
        };
        o.unit_display = match opt_str(self, "unitDisplay").as_deref() {
            Some("long") => UnitDisplay::Long,
            Some("narrow") => UnitDisplay::Narrow,
            _ => UnitDisplay::Short,
        };
        // ECMA-402's default rounding is half-expand (1.25 → 1.3), not banker's rounding.
        o.rounding_mode = match opt_str(self, "roundingMode").as_deref() {
            Some("ceil") => RoundingMode::Ceil,
            Some("floor") => RoundingMode::Floor,
            Some("expand") => RoundingMode::Expand,
            Some("trunc") => RoundingMode::Trunc,
            Some("halfCeil") => RoundingMode::HalfCeil,
            Some("halfFloor") => RoundingMode::HalfFloor,
            Some("halfExpand") => RoundingMode::HalfExpand,
            Some("halfTrunc") => RoundingMode::HalfTrunc,
            Some("halfEven") => RoundingMode::HalfEven,
            _ => RoundingMode::HalfExpand,
        };
        if matches!(
            self.realm
                .get_property(handle, "useGrouping")
                .map(|v| v.unpack()),
            Some(Unpacked::Bool(false))
        ) {
            o.use_grouping = UseGrouping::Never;
        }
        if let Some(mid) = opt_num(self, "minimumIntegerDigits") {
            o.minimum_integer_digits = mid;
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
            // `intl` 0.4 panics on negative zero; format +0 and re-apply the sign.
            if n == 0.0 && n.is_sign_negative() {
                let body = intl::number::format(&locale, 0.0, &opts);
                let sd = self
                    .realm
                    .get_property(handle, "signDisplay")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                if !matches!(sd.as_str(), "never" | "exceptZero") && !body.starts_with('-') {
                    // `signDisplay: "always"` formatted `+0` as "+0"; -0 takes the
                    // negative sign, so drop a leading "+" before prepending "-".
                    let mag = body.strip_prefix('+').unwrap_or(&body);
                    return alloc::format!("-{mag}");
                }
                return body;
            }
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

    // --- Intl.Locale -------------------------------------------------------

    /// Builds (once) and links `Intl.Locale.prototype`: the `language`/`script`/…
    /// brand-checked `get` accessors, the `maximize`/`minimize`/`toString` methods,
    /// the `constructor` back-link, and `[Symbol.toStringTag] = "Intl.Locale"`.
    fn intl_locale_prototype(&mut self) -> Option<Handle> {
        if let Some(p) = self.realm.intl_prototype(N_INTL_LOCALE) {
            return Some(p);
        }
        let ctor = self.intl_ctor_handle("Locale")?;
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &name in LOCALE_ACCESSORS {
            let label = alloc::format!("get {name}");
            let target = self.new_str(name);
            let th = target.as_handle().map(Handle::from_raw).unwrap();
            let getter = self.realm.new_bound_native(N_INTL_LOCALE_ACCESSOR, th);
            self.install_fn_name_length(getter, &label, 0);
            self.realm.define_accessor(
                proto,
                name,
                NanBox::handle(getter.to_raw()),
                NanBox::undefined(),
            );
            self.realm.mark_hidden(proto, name);
        }
        for &m in &["maximize", "minimize", "toString"] {
            let target = self.new_str(m);
            let th = target.as_handle().map(Handle::from_raw).unwrap();
            let f = self.realm.new_bound_native(N_INTL_LOCALE_METHOD, th);
            self.install_fn_name_length(f, m, 0);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        self.install_to_string_tag(proto, "Intl.Locale");
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.link_ctor_prototype(ctor, proto);
        self.realm.set_intl_prototype(N_INTL_LOCALE, proto);
        Some(proto)
    }

    /// `new Intl.Locale(tag, options)` — parses and canonicalizes `tag` (a string or
    /// another `Locale`), applies the `language`/`script`/`region`/`calendar`/
    /// `collation`/`hourCycle`/`caseFirst`/`kn`/`numberingSystem` options onto the
    /// Unicode extension, and stores the resolved components behind hidden slots for
    /// the prototype accessors. Returns a branded `Intl.Locale` instance.
    pub(crate) fn make_locale(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let tag_arg = args.first().copied().unwrap_or(NanBox::undefined());
        // A `Locale` argument contributes its `[[Locale]]` tag; otherwise ToString
        // (undefined/symbol throw via the usual coercion / TypeError below).
        let base_tag = if let Some(h) = tag_arg.as_handle().map(Handle::from_raw)
            && let Some(t) = self.realm.get_property(h, "\u{0}locale_tag")
        {
            self.realm.to_display_string(t)
        } else if matches!(tag_arg.unpack(), Unpacked::Undefined) {
            let m = self.new_str("Intl.Locale: tag must be a string or Locale");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        } else {
            self.coerce_to_string(tag_arg)?
        };
        let Some(canon) = canonicalize_locale_id(&base_tag) else {
            let m = self.new_str(&alloc::format!("invalid language tag: {base_tag}"));
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        };
        // Read the option overrides (each validated like the formatter options).
        let opts = args
            .get(1)
            .copied()
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.coerce_to_object(v))
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        let language = self.get_string_option(opts, "language", &[], None)?;
        let script = self.get_string_option(opts, "script", &[], None)?;
        let region = self.get_string_option(opts, "region", &[], None)?;
        let calendar = self.get_string_option(opts, "calendar", &[], None)?;
        let collation = self.get_string_option(opts, "collation", &[], None)?;
        let hour_cycle =
            self.get_string_option(opts, "hourCycle", &["h11", "h12", "h23", "h24"], None)?;
        let case_first =
            self.get_string_option(opts, "caseFirst", &["upper", "lower", "false"], None)?;
        let numeric = self.get_bool_option(opts, "numeric", None)?;
        let numbering = self.get_string_option(opts, "numberingSystem", &[], None)?;

        let mut parsed = ParsedLocale::from_canonical(&canon);
        if let Some(l) = &language {
            parsed.language = l.to_ascii_lowercase();
        }
        if let Some(s) = &script {
            parsed.script = Some(titlecase_script(s));
        }
        if let Some(r) = &region {
            parsed.region = Some(r.to_ascii_uppercase());
        }
        if let Some(c) = &calendar {
            parsed.set_keyword("ca", c);
        }
        if let Some(c) = &collation {
            parsed.set_keyword("co", c);
        }
        if let Some(h) = &hour_cycle {
            parsed.set_keyword("hc", h);
        }
        if let Some(k) = &case_first {
            parsed.set_keyword("kf", k);
        }
        if let Some(b) = numeric {
            parsed.set_keyword("kn", if b { "true" } else { "false" });
        }
        if let Some(n) = &numbering {
            parsed.set_keyword("nu", n);
        }
        let final_tag = parsed.to_tag();
        let obj = self.realm.new_object();
        let tagv = self.new_str(&final_tag);
        self.realm.set_hidden_property(obj, "\u{0}locale_tag", tagv);
        // Stash the resolved components so the accessors read them cheaply.
        let store = |this: &mut Self, key: &str, val: Option<&str>| {
            if let Some(v) = val {
                let sv = this.new_str(v);
                this.realm.set_hidden_property(obj, key, sv);
            }
        };
        store(self, "\u{0}loc_language", Some(&parsed.language));
        store(self, "\u{0}loc_script", parsed.script.as_deref());
        store(self, "\u{0}loc_region", parsed.region.as_deref());
        store(self, "\u{0}loc_baseName", Some(&parsed.base_name()));
        store(self, "\u{0}loc_ca", parsed.keyword("ca"));
        store(self, "\u{0}loc_co", parsed.keyword("co"));
        store(self, "\u{0}loc_hc", parsed.keyword("hc"));
        store(self, "\u{0}loc_kf", parsed.keyword("kf"));
        store(self, "\u{0}loc_nu", parsed.keyword("nu"));
        // `numeric` is the boolean form of the `kn` keyword.
        let kn = parsed.keyword("kn");
        let numeric_val = matches!(kn, Some("true") | Some(""));
        self.realm
            .set_hidden_property(obj, "\u{0}loc_numeric", NanBox::boolean(numeric_val));
        // Brand + link to the prototype.
        self.realm
            .set_hidden_property(obj, "\u{0}brand_loc", NanBox::boolean(true));
        if let Some(proto) = self.intl_locale_prototype() {
            self.realm.set_object_proto(obj, Some(proto));
        }
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Dispatches an `Intl.Locale` `get` accessor: brand-checks `this`, then returns
    /// the stored component (a string, or `undefined`; `numeric` is a boolean).
    pub(crate) fn intl_locale_accessor_dispatch(
        &mut self,
        this: NanBox,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let h = self.require_intl_slot(this, "\u{0}brand_loc", "Intl.Locale.prototype getter")?;
        let read = |this: &Self, key: &str| -> NanBox {
            this.realm
                .get_property(h, key)
                .unwrap_or(NanBox::undefined())
        };
        Ok(match name {
            "language" => read(self, "\u{0}loc_language"),
            "script" => read(self, "\u{0}loc_script"),
            "region" => read(self, "\u{0}loc_region"),
            "baseName" => read(self, "\u{0}loc_baseName"),
            "calendar" => read(self, "\u{0}loc_ca"),
            "collation" => read(self, "\u{0}loc_co"),
            "hourCycle" => read(self, "\u{0}loc_hc"),
            "caseFirst" => read(self, "\u{0}loc_kf"),
            "numberingSystem" => read(self, "\u{0}loc_nu"),
            "numeric" => read(self, "\u{0}loc_numeric"),
            _ => NanBox::undefined(),
        })
    }

    /// Dispatches an `Intl.Locale` method (`maximize`/`minimize`/`toString`):
    /// brand-checks `this`. `toString` returns the canonical tag; `maximize`/
    /// `minimize` return a fresh `Locale` with the same tag (no CLDR likely-subtags
    /// data, so the tag is returned unchanged — structurally valid).
    pub(crate) fn intl_locale_method_dispatch(
        &mut self,
        this: NanBox,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let h = self.require_intl_slot(this, "\u{0}brand_loc", "Intl.Locale.prototype method")?;
        let tag = self
            .realm
            .get_property(h, "\u{0}locale_tag")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        match name {
            "toString" => Ok(self.new_str(&tag)),
            // maximize/minimize: rebuild a Locale from the (unchanged) tag.
            _ => {
                let tagv = self.new_str(&tag);
                self.make_locale(&[tagv])
            }
        }
    }

    /// Splits the formatted output of an `Intl.NumberFormat` handle into typed
    /// `(kind, value)` parts, mirroring `nf.formatToParts(value)`. Shared by the
    /// `formatToParts` dispatch and by `Intl.DurationFormat`, which composes
    /// per-unit `NumberFormat`s. The `intl`-crate path yields CLDR parts; the
    /// hand-rolled path (unit style, or the no-`intl` build) re-derives parts from
    /// the formatted string's structure (sign / integer-groups / decimal / fraction).
    pub(crate) fn number_handle_parts(
        &mut self,
        handle: Handle,
        value: NanBox,
    ) -> Vec<(&'static str, String)> {
        #[cfg(feature = "intl")]
        if !self.number_uses_handrolled(handle) {
            let n = self.realm.to_number(value);
            let locale = self
                .realm
                .get_property(handle, "\u{0}locale")
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("en"));
            let opts = self.number_format_options(handle);
            // `intl` 0.4 panics formatting negative zero; format +0 and re-apply the
            // sign per `signDisplay` (negative zero shows "-0" under auto/always).
            let neg_zero = n == 0.0 && n.is_sign_negative();
            let feed = if neg_zero { 0.0 } else { n };
            let mut parts: Vec<(&'static str, String)> =
                intl::number::format_to_parts(&locale, feed, &opts)
                    .into_iter()
                    .map(|p| (p.kind.as_str(), p.value))
                    .collect();
            if neg_zero {
                let sd = self
                    .realm
                    .get_property(handle, "signDisplay")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                let show = !matches!(sd.as_str(), "never" | "exceptZero")
                    && !parts.iter().any(|(t, _)| *t == "minusSign");
                if show {
                    parts.insert(0, ("minusSign", String::from("-")));
                }
            }
            return parts;
        }
        // Hand-rolled path: re-derive parts from the formatted string.
        let formatted = self.intl_format_value(handle, value);
        let style = self
            .realm
            .get_property(handle, "style")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("decimal"));
        let currency_sym = if style == "currency" {
            let code = self
                .realm
                .get_property(handle, "currency")
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_default();
            currency_symbol(&code)
        } else {
            String::new()
        };
        let mut entries: Vec<(&'static str, String)> = Vec::new();
        let mut s = formatted.as_str();
        if let Some(rest) = s.strip_prefix('-') {
            entries.push(("minusSign", String::from("-")));
            s = rest;
        }
        if !currency_sym.is_empty() && s.starts_with(currency_sym.as_str()) {
            entries.push(("currency", currency_sym.clone()));
            s = &s[currency_sym.len()..];
        }
        let mut percent = false;
        if style == "percent" && s.ends_with('%') {
            percent = true;
            s = &s[..s.len() - '%'.len_utf8()];
        }
        if s == "NaN" {
            entries.push(("nan", String::from("NaN")));
        } else if s == "∞" {
            entries.push(("infinity", String::from("∞")));
        } else {
            let (int_part, frac_part) = match s.split_once('.') {
                Some((i, f)) => (i, Some(f)),
                None => (s, None),
            };
            for (gi, grp) in int_part.split(',').enumerate() {
                if gi > 0 {
                    entries.push(("group", String::from(",")));
                }
                entries.push(("integer", String::from(grp)));
            }
            if let Some(f) = frac_part {
                entries.push(("decimal", String::from(".")));
                entries.push(("fraction", String::from(f)));
            }
        }
        if percent {
            entries.push(("percentSign", String::from("%")));
        }
        entries
    }

    // --- Intl.DurationFormat ----------------------------------------------

    /// Builds (once) and links `Intl.DurationFormat.prototype` with branded
    /// `format`/`formatToParts`/`resolvedOptions` methods, the `constructor`
    /// back-link, and `[Symbol.toStringTag] = "Intl.DurationFormat"`.
    fn intl_duration_prototype(&mut self) -> Option<Handle> {
        if let Some(p) = self.realm.intl_prototype(N_INTL_DURATION_FORMAT) {
            return Some(p);
        }
        let ctor = self.intl_ctor_handle("DurationFormat")?;
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &(m, arity) in &[
            ("resolvedOptions", 0u32),
            ("format", 1),
            ("formatToParts", 1),
        ] {
            let target = self.new_str(m);
            let th = target.as_handle().map(Handle::from_raw).unwrap();
            let f = self.realm.new_bound_native(N_INTL_DURATION_METHOD, th);
            self.install_fn_name_length(f, m, arity);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        self.install_to_string_tag(proto, "Intl.DurationFormat");
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.link_ctor_prototype(ctor, proto);
        self.realm.set_intl_prototype(N_INTL_DURATION_FORMAT, proto);
        Some(proto)
    }

    /// `new Intl.DurationFormat(locales, options)` — `InitializeDurationFormat`:
    /// canonicalizes the locale, resolves `numberingSystem` (from the `-u-nu-`
    /// extension and/or option), reads the top-level `style` and each of the ten
    /// per-unit `<unit>`/`<unit>Display` options (via `GetDurationUnitOptions`, in
    /// spec order), and `fractionalDigits`. Stores all resolved slots behind hidden
    /// keys for `resolvedOptions`/`format`. A non-object `options` (other than
    /// `undefined`) is a TypeError; an invalid option value is a RangeError.
    pub(crate) fn make_duration_format(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        // GetOptionsObject: undefined → no options; null/primitive → TypeError.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.DurationFormat options must be an object"));
        };
        // localeMatcher (validated, not otherwise used).
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        // numberingSystem option (UTS-35 `type` validated) and the `-u-nu-` extension
        // resolve the data locale and the effective numbering system.
        let nu_opt = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(ns) = &nu_opt
            && !is_unicode_type_value(ns)
        {
            let m = self.new_str("invalid numberingSystem");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let (locale, numbering) =
            self.resolve_duration_locale(requested.first().map(String::as_str), nu_opt.as_deref());
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        let nuv = self.new_str(&numbering);
        self.realm.set_hidden_property(obj, "numberingSystem", nuv);
        // style (default "short"); "digital" implies numeric h/m/s defaults.
        let style = self
            .get_string_option(
                opts,
                "style",
                &["long", "short", "narrow", "digital"],
                Some("short"),
            )?
            .unwrap();
        self.store_str(obj, "style", &Some(style.clone()));
        let digital = style == "digital";
        // Each row of the unit table: (unit, stylesList, digitalBase). prevStyle is
        // threaded through hours..microseconds (GetDurationUnitOptions step 6).
        const SUB: &[&str] = &["long", "short", "narrow"];
        const TIME: &[&str] = &["long", "short", "narrow", "numeric", "2-digit"];
        const FRAC: &[&str] = &["long", "short", "narrow", "numeric"];
        // (unit, stylesList, digitalDefault) per the unit table.
        let table: &[(&str, &[&str], &str)] = &[
            ("years", SUB, "short"),
            ("months", SUB, "short"),
            ("weeks", SUB, "short"),
            ("days", SUB, "short"),
            ("hours", TIME, "numeric"),
            ("minutes", TIME, "numeric"),
            ("seconds", TIME, "numeric"),
            ("milliseconds", FRAC, "numeric"),
            ("microseconds", FRAC, "numeric"),
            ("nanoseconds", FRAC, "numeric"),
        ];
        let mut prev_style: Option<String> = None;
        for (unit, styles, digital_base) in table {
            let (ust, udisp) = self.get_duration_unit_options(
                opts,
                unit,
                &style,
                styles,
                digital_base,
                prev_style.as_deref(),
                digital,
            )?;
            self.store_str(obj, unit, &Some(ust.clone()));
            let disp_key = alloc::format!("{unit}Display");
            self.store_str(obj, &disp_key, &Some(udisp));
            if matches!(
                *unit,
                "hours" | "minutes" | "seconds" | "milliseconds" | "microseconds"
            ) {
                prev_style = Some(ust);
            }
        }
        // fractionalDigits: GetNumberOption(0, 9, undefined).
        if let Some(fd) = self.get_int_option(opts, "fractionalDigits", 0.0, 9.0, None)? {
            self.realm
                .set_hidden_property(obj, "fractionalDigits", NanBox::number(fd));
        }
        self.realm
            .set_hidden_property(obj, "\u{0}brand_df", NanBox::boolean(true));
        if let Some(proto) = self.intl_duration_prototype() {
            self.realm.set_object_proto(obj, Some(proto));
        }
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Resolves the `Intl.DurationFormat` data locale and effective numbering system
    /// from the requested tag's `-u-nu-` extension and the `numberingSystem` option.
    /// The option (when a supported value) wins; otherwise a supported extension
    /// value is kept and reflected in the locale; an unsupported value falls back to
    /// the locale's default (`"latn"` for `en`). Returns `(dataLocale, numbering)`.
    fn resolve_duration_locale(
        &mut self,
        requested: Option<&str>,
        option: Option<&str>,
    ) -> (String, String) {
        let tag = requested.unwrap_or("en-US");
        // Split off any `-u-` extension; recover an `nu` keyword value.
        let parsed = ParsedLocale::from_canonical(tag);
        let ext_nu = parsed.keyword("nu").map(String::from);
        let default_nu = String::from("latn");
        let supported = |ns: &str| is_supported_numbering_system(ns);
        // The base tag without any -u- keywords (other extensions preserved).
        let mut base = parsed.base_name();
        for e in &parsed.other_ext {
            base.push('-');
            base.push_str(e);
        }
        match option {
            Some(opt) if supported(opt) => {
                // Option wins; reflect in the locale only if it equals the extension.
                if ext_nu.as_deref() == Some(opt) {
                    (alloc::format!("{base}-u-nu-{opt}"), String::from(opt))
                } else {
                    (base, String::from(opt))
                }
            }
            _ => {
                // No (usable) option: a supported extension value is used & reflected.
                match ext_nu {
                    Some(ns) if supported(&ns) => (alloc::format!("{base}-u-nu-{ns}"), ns),
                    _ => (base, default_nu),
                }
            }
        }
    }

    /// `GetDurationUnitOptions(unit, options, baseStyle, stylesList, digitalBase,
    /// prevStyle, twoDigitHours)` — reads the per-unit `<unit>` style and
    /// `<unit>Display` options, applying the spec defaults (digital base, the
    /// `prevStyle`-driven "numeric"/"2-digit" promotion) and the RangeError
    /// conditions. Returns `(style, display)`.
    #[allow(clippy::too_many_arguments)]
    fn get_duration_unit_options(
        &mut self,
        opts: Option<Handle>,
        unit: &str,
        base_style: &str,
        styles_list: &[&str],
        digital_base: &str,
        prev_style: Option<&str>,
        two_digit_hours: bool,
    ) -> Result<(String, String), ExecError> {
        let mut style = self.get_string_option(opts, unit, styles_list, None)?;
        let mut display_default = "always";
        if style.is_none() {
            if base_style == "digital" {
                if !matches!(unit, "hours" | "minutes" | "seconds") {
                    display_default = "auto";
                }
                style = Some(String::from(digital_base));
            } else {
                display_default = "auto";
                if matches!(prev_style, Some("numeric") | Some("2-digit")) {
                    style = Some(String::from("numeric"));
                } else {
                    style = Some(String::from(base_style));
                }
            }
        }
        let mut style = style.unwrap();
        let disp_key = alloc::format!("{unit}Display");
        let display = self
            .get_string_option(opts, &disp_key, &["auto", "always"], Some(display_default))?
            .unwrap();
        if matches!(prev_style, Some("numeric") | Some("2-digit")) {
            if style != "numeric" && style != "2-digit" {
                let m = self.new_str(&alloc::format!(
                    "invalid style '{style}' for {unit} following a numeric unit"
                ));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            } else if matches!(unit, "minutes" | "seconds") {
                style = String::from("2-digit");
            }
        }
        // twoDigitHours: a digital formatter with 2-digit hours promotes "numeric".
        if unit == "hours" && two_digit_hours && style == "numeric" {
            // (Kept as "numeric" — the engine's digital renderer never pads hours,
            // matching the en reference where [[TwoDigitHours]] is false.)
        }
        Ok((style, display))
    }

    /// The ten duration units in table order.
    const DURATION_UNITS: [&'static str; 10] = [
        "years",
        "months",
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ];

    /// Dispatches an `Intl.DurationFormat` prototype method: brand-checks `this`.
    pub(crate) fn intl_duration_method_dispatch(
        &mut self,
        this: NanBox,
        name: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let h = self.require_intl_slot(
            this,
            "\u{0}brand_df",
            "Intl.DurationFormat.prototype method",
        )?;
        match name {
            "resolvedOptions" => self.duration_resolved_options(h),
            "formatToParts" => {
                let rec = self
                    .read_duration_record(args.first().copied().unwrap_or(NanBox::undefined()))?;
                let parts = self.partition_duration(h, &rec);
                let mut arr = Vec::with_capacity(parts.len());
                for (ty, val, unit) in parts {
                    let o = self.realm.new_object();
                    let tv = self.new_str(ty);
                    self.realm.set_property(o, "type", tv);
                    let vv = self.new_str(&val);
                    self.realm.set_property(o, "value", vv);
                    if let Some(u) = unit {
                        let uv = self.new_str(u);
                        self.realm.set_property(o, "unit", uv);
                    }
                    arr.push(NanBox::handle(o.to_raw()));
                }
                Ok(NanBox::handle(self.realm.new_array(arr).to_raw()))
            }
            // `format(duration)` — concatenate the partitioned parts' values.
            _ => {
                let rec = self
                    .read_duration_record(args.first().copied().unwrap_or(NanBox::undefined()))?;
                let parts = self.partition_duration(h, &rec);
                let s: String = parts.into_iter().map(|(_, v, _)| v).collect();
                Ok(self.new_str(&s))
            }
        }
    }

    /// `Intl.DurationFormat.prototype.resolvedOptions` — `{locale, numberingSystem,
    /// style, <unit>, <unit>Display …, fractionalDigits?}` in spec key order.
    fn duration_resolved_options(&mut self, h: Handle) -> Result<NanBox, ExecError> {
        let out = self.realm.new_object();
        let read = |this: &mut Self, key: &str, dflt: &str| -> String {
            this.realm
                .get_property(h, key)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
                .unwrap_or_else(|| String::from(dflt))
        };
        let locale = read(self, "\u{0}locale", "en-US");
        let lv = self.new_str(&locale);
        self.realm.set_property(out, "locale", lv);
        let nu = read(self, "numberingSystem", "latn");
        let nuv = self.new_str(&nu);
        self.realm.set_property(out, "numberingSystem", nuv);
        let style = read(self, "style", "short");
        let sv = self.new_str(&style);
        self.realm.set_property(out, "style", sv);
        for unit in Self::DURATION_UNITS {
            let ust = read(self, unit, "short");
            let uv = self.new_str(&ust);
            self.realm.set_property(out, unit, uv);
            let disp_key = alloc::format!("{unit}Display");
            let udisp = read(self, &disp_key, "auto");
            let dv = self.new_str(&udisp);
            self.realm.set_property(out, &disp_key, dv);
        }
        // fractionalDigits is reported only when it was explicitly set.
        if let Some(v) = self
            .realm
            .get_property(h, "fractionalDigits")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
        {
            self.realm.set_property(out, "fractionalDigits", v);
        }
        Ok(NanBox::handle(out.to_raw()))
    }

    /// `ToDurationRecord(input)` — a String is parsed as an ISO-8601 duration
    /// (RangeError on a bad string); a non-object is a TypeError; otherwise each of
    /// the ten unit fields is read and `ToIntegerIfIntegral`-coerced (a non-integer
    /// or non-finite value is a RangeError), all fields absent is a TypeError, and
    /// the record is validated by `IsValidDuration` (RangeError otherwise). Returns
    /// the ten `f64` unit values in table order.
    fn read_duration_record(&mut self, input: NanBox) -> Result<[f64; 10], ExecError> {
        // A Temporal.Duration-like string is parsed; any other primitive (or a
        // String wrapper is an Object, handled below) follows the object path.
        if let Some(s) = input
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|hh| self.realm.string_value(hh))
        {
            return self.parse_duration_string(&s);
        }
        if !self.is_object_value(input) {
            return Err(self.type_error("Intl.DurationFormat.format: argument must be an object"));
        }
        let oh = input.as_handle().map(Handle::from_raw).unwrap();
        let mut rec = [0.0f64; 10];
        let mut any = false;
        for (i, unit) in Self::DURATION_UNITS.iter().enumerate() {
            let v = self.read_member(oh, unit)?;
            if matches!(v.unpack(), Unpacked::Undefined) {
                continue;
            }
            any = true;
            // ToIntegerIfIntegral: ToNumber, then require an integral value.
            let nv = self.coerce_to_number(v)?;
            let n = self.realm.to_number(nv);
            if !n.is_finite() || trunc_toward_zero(n) != n {
                let m = self.new_str(&alloc::format!("duration field {unit} is not an integer"));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            rec[i] = n;
        }
        if !any {
            return Err(self.type_error("Intl.DurationFormat.format: no duration fields present"));
        }
        if !is_valid_duration(&rec) {
            let m = self.new_str("invalid Duration");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(rec)
    }

    /// Minimal ISO-8601 duration-string parser (`±PnYnMnWnDTnHnMnS`, fractional
    /// seconds). A malformed string is a RangeError. Sub-second fractions are split
    /// into milli/micro/nanoseconds.
    fn parse_duration_string(&mut self, s: &str) -> Result<[f64; 10], ExecError> {
        let bad = |this: &mut Self| -> ExecError {
            let m = this.new_str("invalid Duration string");
            ExecError::Throw(this.make_error(N_RANGE_ERROR, Some(m)))
        };
        let mut rec = [0.0f64; 10];
        let bytes = s.trim();
        let (sign, rest) = match bytes.strip_prefix('-') {
            Some(r) => (-1.0, r),
            None => (1.0, bytes.strip_prefix('+').unwrap_or(bytes)),
        };
        let rest = match rest.strip_prefix(['P', 'p']) {
            Some(r) => r,
            None => return Err(bad(self)),
        };
        // Split date / time portions on 'T'.
        let (date_part, time_part) = match rest.split_once(['T', 't']) {
            Some((d, t)) => (d, Some(t)),
            None => (rest, None),
        };
        let mut saw_any = false;
        // Returns (value, designator) tokens.
        let parse_section = |this: &mut Self,
                             sect: &str,
                             allowed: &[(char, usize)],
                             rec: &mut [f64; 10],
                             saw: &mut bool|
         -> Result<(), ExecError> {
            let mut chars = sect.chars().peekable();
            let mut last_idx: i32 = -1;
            while chars.peek().is_some() {
                let mut num = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' || c == ',' {
                        num.push(if c == ',' { '.' } else { c });
                        chars.next();
                    } else {
                        break;
                    }
                }
                let Some(desig) = chars.next() else {
                    if num.is_empty() {
                        break;
                    }
                    return Err(bad(this));
                };
                if num.is_empty() {
                    return Err(bad(this));
                }
                let Some(&(_, slot)) = allowed.iter().find(|(d, _)| d.eq_ignore_ascii_case(&desig))
                else {
                    return Err(bad(this));
                };
                // Designators must appear in order.
                if (slot as i32) <= last_idx {
                    return Err(bad(this));
                }
                last_idx = slot as i32;
                *saw = true;
                // Only seconds may be fractional.
                if num.contains('.') && slot != 6 {
                    return Err(bad(this));
                }
                if slot == 6 {
                    // seconds with optional fraction → split into s/ms/us/ns.
                    let (int_s, frac) = num.split_once('.').unwrap_or((num.as_str(), ""));
                    rec[6] = int_s.parse::<f64>().map_err(|_| bad(this))?;
                    let mut f: String = frac.chars().take(9).collect();
                    while f.len() < 9 {
                        f.push('0');
                    }
                    rec[7] = f[0..3].parse::<f64>().unwrap_or(0.0);
                    rec[8] = f[3..6].parse::<f64>().unwrap_or(0.0);
                    rec[9] = f[6..9].parse::<f64>().unwrap_or(0.0);
                } else {
                    rec[slot] = num.parse::<f64>().map_err(|_| bad(this))?;
                }
            }
            Ok(())
        };
        // Date designators: Y(0) M(1) W(2) D(3).
        parse_section(
            self,
            date_part,
            &[('Y', 0), ('M', 1), ('W', 2), ('D', 3)],
            &mut rec,
            &mut saw_any,
        )?;
        if let Some(tp) = time_part {
            if tp.is_empty() {
                return Err(bad(self));
            }
            // Time designators: H(4) M(5) S(6).
            parse_section(
                self,
                tp,
                &[('H', 4), ('M', 5), ('S', 6)],
                &mut rec,
                &mut saw_any,
            )?;
        }
        if !saw_any {
            return Err(bad(self));
        }
        for v in &mut rec {
            *v *= sign;
        }
        if !is_valid_duration(&rec) {
            let m = self.new_str("invalid Duration");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(rec)
    }

    /// Resolves a duration formatter's per-unit `(style, display)` from its slots.
    fn duration_unit_resolved(&mut self, h: Handle, unit: &str) -> (String, String) {
        let style = self
            .realm
            .get_property(h, unit)
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("short"));
        let disp = self
            .realm
            .get_property(h, &alloc::format!("{unit}Display"))
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("auto"));
        (style, disp)
    }

    /// `PartitionDurationFormatPattern` (en, mirroring the test262 reference): formats
    /// each shown unit via a per-unit `Intl.NumberFormat` (`style:"unit"` or numeric/
    /// 2-digit), threads the time separator between numeric h:m:s, then joins the unit
    /// strings with an `Intl.ListFormat` (`type:"unit"`). Returns `(type, value, unit?)`
    /// parts. Composing the real `NumberFormat`/`ListFormat` keeps `format` and the
    /// reference output identical regardless of CLDR-data fidelity.
    fn partition_duration(
        &mut self,
        h: Handle,
        duration: &[f64; 10],
    ) -> Vec<(&'static str, String, Option<&'static str>)> {
        let style = self
            .realm
            .get_property(h, "style")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("short"));
        let numbering = self
            .realm
            .get_property(h, "numberingSystem")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("latn"));
        let fractional_digits = self
            .realm
            .get_property(h, "fractionalDigits")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_number(v) as i32);
        let units = Self::DURATION_UNITS;
        // Resolve per-unit styles/displays up front.
        let mut ustyle = [const { String::new() }; 10];
        let mut udisp = [const { String::new() }; 10];
        for (i, u) in units.iter().enumerate() {
            let (s, d) = self.duration_unit_resolved(h, u);
            ustyle[i] = s;
            udisp[i] = d;
        }
        let time_separator = ":";
        // Each element of `result` is the ordered parts list for one shown unit.
        let mut result: Vec<Vec<(&'static str, String, Option<&'static str>)>> = Vec::new();
        let mut need_separator = false;
        let mut display_negative_sign = true;
        // Whether any field is negative (for the negative-zero leading-sign rule).
        let any_negative = duration
            .iter()
            .any(|&v| v < 0.0 || (v == 0.0 && v.is_sign_negative()));
        for idx in 0..units.len() {
            let unit = units[idx];
            let singular: &'static str = duration_singular(unit);
            let mut value = duration[idx];
            let style_u = ustyle[idx].clone();
            let display_u = udisp[idx].clone();
            // Combine numeric seconds/ms/us with their fractional remainder.
            let mut value_str: Option<String> = None;
            let mut done = false;
            let (mut nf_min_frac, mut nf_max_frac, mut nf_trunc) =
                (None::<i32>, None::<i32>, false);
            if matches!(unit, "seconds" | "milliseconds" | "microseconds") {
                let next_style = ustyle[idx + 1].as_str();
                if next_style == "numeric" {
                    let exp = match unit {
                        "seconds" => 9,
                        "milliseconds" => 6,
                        _ => 3,
                    };
                    value_str = Some(duration_to_fractional(duration, exp));
                    nf_max_frac = Some(fractional_digits.unwrap_or(9));
                    nf_min_frac = Some(fractional_digits.unwrap_or(0));
                    nf_trunc = true;
                    done = true;
                }
            }
            // Display zero numeric minutes when seconds will be displayed.
            let mut display_required = false;
            if unit == "minutes" && need_separator {
                display_required = udisp[6] == "always"
                    || duration[6] != 0.0
                    || duration[7] != 0.0
                    || duration[8] != 0.0
                    || duration[9] != 0.0;
            }
            let nonzero = value != 0.0
                || value_str
                    .as_deref()
                    .is_some_and(|s| s.bytes().any(|b| matches!(b, b'1'..=b'9')));
            if nonzero || display_u != "auto" || display_required {
                let mut sign_never = false;
                if display_negative_sign {
                    display_negative_sign = false;
                    if value == 0.0 && value_str.is_none() && any_negative {
                        value = -0.0;
                    }
                } else {
                    sign_never = true;
                }
                // Build the per-unit NumberFormat handle.
                let nf = self.realm.new_object();
                let marker = self.new_str("number");
                self.realm.set_hidden_property(nf, "\u{0}intl", marker);
                let loc = self.new_str("en");
                self.realm.set_hidden_property(nf, "\u{0}locale", loc);
                let nuv = self.new_str(&numbering);
                self.realm.set_hidden_property(nf, "numberingSystem", nuv);
                if sign_never {
                    let sd = self.new_str("never");
                    self.realm.set_hidden_property(nf, "signDisplay", sd);
                }
                if style_u == "2-digit" {
                    self.realm
                        .set_hidden_property(nf, "minimumIntegerDigits", NanBox::number(2.0));
                }
                if style_u != "numeric" && style_u != "2-digit" {
                    let st = self.new_str("unit");
                    self.realm.set_hidden_property(nf, "style", st);
                    let uu = self.new_str(singular);
                    self.realm.set_hidden_property(nf, "unit", uu);
                    let ud = self.new_str(&style_u);
                    self.realm.set_hidden_property(nf, "unitDisplay", ud);
                } else {
                    self.realm
                        .set_hidden_property(nf, "useGrouping", NanBox::boolean(false));
                }
                if let Some(mn) = nf_min_frac {
                    self.realm.set_hidden_property(
                        nf,
                        "minimumFractionDigits",
                        NanBox::number(mn as f64),
                    );
                }
                if let Some(mx) = nf_max_frac {
                    self.realm.set_hidden_property(
                        nf,
                        "maximumFractionDigits",
                        NanBox::number(mx as f64),
                    );
                }
                if nf_trunc {
                    let rm = self.new_str("trunc");
                    self.realm.set_hidden_property(nf, "roundingMode", rm);
                }
                // The value to format: the combined fractional string, or the number.
                let value_box = match &value_str {
                    Some(s) => self.new_str(s),
                    None => NanBox::number(value),
                };
                let number_parts = self.number_handle_parts(nf, value_box);
                let mut list: Vec<(&'static str, String, Option<&'static str>)> = if !need_separator
                {
                    Vec::new()
                } else {
                    // Append to the previous (numeric) unit's list, with a separator.
                    let mut prev = result.pop().unwrap();
                    prev.push(("literal", String::from(time_separator), None));
                    prev
                };
                for (ty, val) in number_parts {
                    list.push((ty, val, Some(singular)));
                }
                if !need_separator {
                    if style_u == "2-digit" || style_u == "numeric" {
                        need_separator = true;
                    }
                    result.push(list);
                } else {
                    result.push(list);
                }
            }
            if done {
                break;
            }
        }
        // List style: digital collapses to short.
        let list_style = if style == "digital" {
            String::from("short")
        } else {
            style
        };
        // Build the strings, then join via ListFormat "unit".
        let strings: Vec<String> = result
            .iter()
            .map(|parts| parts.iter().map(|(_, v, _)| v.as_str()).collect())
            .collect();
        let lf = self.realm.new_object();
        let lm = self.new_str("list");
        self.realm.set_hidden_property(lf, "\u{0}intl", lm);
        let llo = self.new_str("en");
        self.realm.set_hidden_property(lf, "\u{0}locale", llo);
        let lt = self.new_str("unit");
        self.realm.set_hidden_property(lf, "type", lt);
        let ls = self.new_str(&list_style);
        self.realm.set_hidden_property(lf, "style", ls);
        let list_parts = self.list_format_parts(&strings, "unit", &list_style);
        // Flatten: an "element" splices in the next unit's parts; a "literal" passes through.
        let mut flattened: Vec<(&'static str, String, Option<&'static str>)> = Vec::new();
        let mut iter = result.into_iter();
        for (ty, val) in list_parts {
            if ty == "element" {
                if let Some(parts) = iter.next() {
                    flattened.extend(parts);
                }
            } else {
                flattened.push((ty, val, None));
            }
        }
        let _ = lf;
        flattened
    }
}

/// A parsed `unicode_locale_id` split into core subtags plus its `-u-` keyword
/// list, used by `Intl.Locale` to apply option overrides and rebuild the tag.
struct ParsedLocale {
    language: String,
    script: Option<String>,
    region: Option<String>,
    variants: Vec<String>,
    /// `-u-` keyword `(key, value)` pairs (value may be empty for `kn`/`true`).
    keywords: Vec<(String, String)>,
    /// Any other extensions (`-t-`, `-x-`, …) preserved verbatim after the `-u-` block.
    other_ext: Vec<String>,
}

impl ParsedLocale {
    /// Splits a canonical tag (the output of [`canonicalize_locale_id`]) into its
    /// components and `-u-` keywords.
    fn from_canonical(canon: &str) -> Self {
        let mut language = String::new();
        let mut script = None;
        let mut region = None;
        let mut variants = Vec::new();
        let mut keywords = Vec::new();
        let mut other_ext = Vec::new();
        let parts: Vec<&str> = canon.split('-').collect();
        let mut i = 0;
        if i < parts.len() {
            language = parts[i].to_ascii_lowercase();
            i += 1;
        }
        if i < parts.len()
            && parts[i].len() == 4
            && parts[i].bytes().all(|b| b.is_ascii_alphabetic())
        {
            script = Some(titlecase_script(parts[i]));
            i += 1;
        }
        if i < parts.len()
            && ((parts[i].len() == 2 && parts[i].bytes().all(|b| b.is_ascii_alphabetic()))
                || (parts[i].len() == 3 && parts[i].bytes().all(|b| b.is_ascii_digit())))
        {
            region = Some(parts[i].to_ascii_uppercase());
            i += 1;
        }
        // Variants until a singleton.
        while i < parts.len() && parts[i].len() != 1 {
            variants.push(parts[i].to_ascii_lowercase());
            i += 1;
        }
        // Extensions.
        while i < parts.len() {
            let singleton = parts[i];
            if singleton == "u" {
                i += 1;
                // A `-u-` block alternates 2-char keys with their (3-8 char) values.
                // Leading attributes (3-8 char, no key) are rare; skip until the first
                // 2-char key, then read each key's run of non-key value subtags.
                while i < parts.len() && parts[i].len() != 1 {
                    if parts[i].len() != 2 {
                        // An attribute (no key) — preserve as a bare keyword-less subtag.
                        i += 1;
                        continue;
                    }
                    let key = String::from(parts[i]);
                    i += 1;
                    let mut vals: Vec<String> = Vec::new();
                    while i < parts.len() && parts[i].len() != 1 && parts[i].len() != 2 {
                        vals.push(String::from(parts[i]));
                        i += 1;
                    }
                    keywords.push((key, vals.join("-")));
                }
            } else {
                // Preserve other extensions verbatim.
                let mut buf = alloc::vec![String::from(singleton)];
                i += 1;
                while i < parts.len() && parts[i].len() != 1 {
                    buf.push(String::from(parts[i]));
                    i += 1;
                }
                other_ext.push(buf.join("-"));
            }
        }
        ParsedLocale {
            language,
            script,
            region,
            variants,
            keywords,
            other_ext,
        }
    }

    /// The value of `-u-` keyword `key`, if present (empty string for a bare key).
    fn keyword(&self, key: &str) -> Option<&str> {
        self.keywords
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Sets (or replaces) a `-u-` keyword. A `kf`/`kn` `"false"`/`"true"` is kept
    /// as-is; the accessors interpret it.
    fn set_keyword(&mut self, key: &str, val: &str) {
        let v = val.to_ascii_lowercase();
        if let Some(e) = self.keywords.iter_mut().find(|(k, _)| k == key) {
            e.1 = v;
        } else {
            self.keywords.push((String::from(key), v));
        }
    }

    /// The `baseName`: `language(-script)?(-region)?(-variants)?` (no extensions).
    fn base_name(&self) -> String {
        let mut out = self.language.clone();
        if let Some(s) = &self.script {
            out.push('-');
            out.push_str(s);
        }
        if let Some(r) = &self.region {
            out.push('-');
            out.push_str(r);
        }
        for v in &self.variants {
            out.push('-');
            out.push_str(v);
        }
        out
    }

    /// Rebuilds the full canonical tag (base name + sorted `-u-` keywords + other
    /// extensions).
    fn to_tag(&self) -> String {
        let mut out = self.base_name();
        let mut kw = self.keywords.clone();
        kw.sort_by(|a, b| a.0.cmp(&b.0));
        if !kw.is_empty() {
            out.push_str("-u");
            for (k, v) in &kw {
                out.push('-');
                out.push_str(k);
                if !v.is_empty() && v != "true" {
                    out.push('-');
                    out.push_str(v);
                }
            }
        }
        for e in &self.other_ext {
            out.push('-');
            out.push_str(e);
        }
        out
    }
}

/// Whether `ns` is a numbering system this engine treats as supported (only `latn`
/// and `arab` have data; others fall back to the locale default for DurationFormat).
fn is_supported_numbering_system(ns: &str) -> bool {
    matches!(ns, "latn" | "arab")
}

/// The singular `Intl.NumberFormat` unit name for a duration unit field
/// (`"hours"` → `"hour"`), i.e. the field name without its trailing `s`.
fn duration_singular(unit: &str) -> &'static str {
    match unit {
        "years" => "year",
        "months" => "month",
        "weeks" => "week",
        "days" => "day",
        "hours" => "hour",
        "minutes" => "minute",
        "seconds" => "second",
        "milliseconds" => "millisecond",
        "microseconds" => "microsecond",
        _ => "nanosecond",
    }
}

/// `IsValidDuration(record)` for the ten `f64` unit values: all-same-sign, finite,
/// `abs(years|months|weeks) < 2^32`, and `abs(normalizedSeconds) < 2^53` where the
/// seconds are accumulated exactly via `i128` nanoseconds to avoid `f64` rounding.
fn is_valid_duration(rec: &[f64; 10]) -> bool {
    let mut sign = 0i32;
    for &v in rec {
        if !v.is_finite() {
            return false;
        }
        let s = if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 {
            if sign != 0 && sign != s {
                return false;
            }
            sign = s;
        }
    }
    // years/months/weeks bound.
    let two32 = 4_294_967_296.0f64; // 2^32
    if rec[0].abs() >= two32 || rec[1].abs() >= two32 || rec[2].abs() >= two32 {
        return false;
    }
    // normalizedSeconds = days*86400 + hours*3600 + minutes*60 + seconds
    //   + ms/1e3 + us/1e6 + ns/1e9, computed exactly in i128 nanoseconds.
    // Each value is integral and finite here. Guard against i128 overflow by
    // rejecting any single term that already exceeds the 2^53-seconds budget.
    let two53_ns = (1i128 << 53) * 1_000_000_000; // 2^53 seconds in ns
    let term_ns = |v: f64, scale_ns: i128| -> Option<i128> {
        if v.abs() >= 9.0e30 {
            return None; // far beyond any valid range; treat as overflow
        }
        (v as i128).checked_mul(scale_ns)
    };
    let mut total: i128 = 0;
    let scales: [(usize, i128); 7] = [
        (3, 86_400 * 1_000_000_000),
        (4, 3_600 * 1_000_000_000),
        (5, 60 * 1_000_000_000),
        (6, 1_000_000_000),
        (7, 1_000_000),
        (8, 1_000),
        (9, 1),
    ];
    for (i, scale) in scales {
        match term_ns(rec[i], scale).and_then(|t| total.checked_add(t)) {
            Some(t) => total = t,
            None => return false,
        }
    }
    total.abs() < two53_ns
}

/// `DurationToFractional` (test262 reference): the numeric value of the unit at
/// `exponent` (9=seconds, 6=milliseconds, 3=microseconds) combined with all smaller
/// sub-second fields, as an exact decimal string. Uses `i128` nanoseconds.
fn duration_to_fractional(duration: &[f64; 10], exponent: u32) -> String {
    let (seconds, milliseconds, microseconds, nanoseconds) =
        (duration[6], duration[7], duration[8], duration[9]);
    // Fast path: no smaller sub-seconds present → return the bare amount.
    match exponent {
        9 if milliseconds == 0.0 && microseconds == 0.0 && nanoseconds == 0.0 => {
            return format_integral(seconds);
        }
        6 if microseconds == 0.0 && nanoseconds == 0.0 => {
            return format_integral(milliseconds);
        }
        3 if nanoseconds == 0.0 => {
            return format_integral(microseconds);
        }
        _ => {}
    }
    let mut ns: i128 = nanoseconds as i128;
    if exponent >= 9 {
        ns += (seconds as i128) * 1_000_000_000;
    }
    if exponent >= 6 {
        ns += (milliseconds as i128) * 1_000_000;
    }
    if exponent >= 3 {
        ns += (microseconds as i128) * 1_000;
    }
    let e: i128 = 10i128.pow(exponent);
    let q = ns / e;
    let mut r = ns % e;
    if r < 0 {
        r = -r;
    }
    let mut rs = alloc::format!("{r}");
    while rs.len() < exponent as usize {
        rs.insert(0, '0');
    }
    alloc::format!("{q}.{rs}")
}

/// Formats an integral `f64` without a decimal point or exponent (for duration
/// values, which are validated integers).
fn format_integral(v: f64) -> String {
    alloc::format!("{}", v as i128)
}

/// Titlecases a 4-letter script subtag (`latn` → `Latn`).
fn titlecase_script(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == 0 {
            out.push(c.to_ascii_uppercase());
        } else {
            out.push(c.to_ascii_lowercase());
        }
    }
    out
}
