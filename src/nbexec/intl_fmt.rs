use super::*;

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
}
