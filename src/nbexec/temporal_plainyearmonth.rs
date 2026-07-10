//! `Temporal.PlainYearMonth` — logic module. A fan-out unit: everything specific to
//! `PlainYearMonth` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
//!
//! A `PlainYearMonth` is a year + month in the ISO-8601 calendar with a hidden
//! *reference ISO day* (default 1). It is stored in [`TemporalData::date`] as an
//! [`IsoDate`] carrying the reference day; the `kind` is
//! [`TemporalKind::PlainYearMonth`].
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    self, IsoDate, MAX_EPOCH_DAYS, MIN_EPOCH_DAYS, Overflow, RoundMode, TemporalData, TemporalKind,
    format_iso_year, is_leap_year, iso_date_in_range, iso_days_in_month, iso_days_in_year, pad,
    regulate_iso_date, round_to_increment,
};

/// Prototype method names installed on `Temporal.PlainYearMonth.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "with",
    "add",
    "subtract",
    "until",
    "since",
    "equals",
    "toPlainDate",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
];
/// Getter-accessor names installed on `Temporal.PlainYearMonth.prototype`.
pub(crate) const GETTERS: &[&str] = &[
    "year",
    "month",
    "monthCode",
    "calendarId",
    "daysInMonth",
    "daysInYear",
    "monthsInYear",
    "inLeapYear",
    "era",
    "eraYear",
];

/// Optional `(year, month, monthCode)` fields read off a property bag; a
/// monthCode is `(monthNumber, isLeap)`.
type YearMonthFields = (Option<i64>, Option<i64>, Option<(u8, bool)>);

// ---------------------------------------------------------------------------
// Pure helpers (no engine coupling)
// ---------------------------------------------------------------------------

/// `ISOYearMonthWithinLimits`: whether a (year, month) is representable.
fn ym_within_limits(year: i64, month: i64) -> bool {
    if !(-271_821..=275_760).contains(&year) {
        return false;
    }
    if year == -271_821 && month < 4 {
        return false;
    }
    if year == 275_760 && month > 9 {
        return false;
    }
    true
}

/// `MM`-style month code for a month number, e.g. 6 → `"M06"`.
fn month_code(month: u8) -> alloc::string::String {
    alloc::format!("M{}", pad(u64::from(month), 2))
}

/// Parses the *syntax* of a month code (`M` + two digits + optional `L`).
/// Returns `(monthNumber, isLeap)`; `None` if ill-formed. Suitability (range /
/// leap-month validity) is checked separately.
fn parse_month_code_syntax(s: &str) -> Option<(u8, bool)> {
    let b = s.as_bytes();
    if b.len() != 3 && b.len() != 4 {
        return None;
    }
    if b[0] != b'M' || !b[1].is_ascii_digit() || !b[2].is_ascii_digit() {
        return None;
    }
    let leap = match b.get(3) {
        None => false,
        Some(b'L') => true,
        Some(_) => return None,
    };
    let num = (b[1] - b'0') * 10 + (b[2] - b'0');
    Some((num, leap))
}

/// Whether an annotation key is a valid lowercase key (`[a-z_][a-z0-9_-]*`).
fn valid_annotation_key(key: &str) -> bool {
    let b = key.as_bytes();
    if b.is_empty() {
        return false;
    }
    if !(b[0].is_ascii_lowercase() || b[0] == b'_') {
        return false;
    }
    b[1..]
        .iter()
        .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_' || c == b'-')
}

/// Validates the annotation tail (`[...]...`) of a Temporal string for
/// PlainYearMonth: at most one *effective* calendar (which must be `iso8601`),
/// no critical flag on any unknown annotation, and no multiple calendar
/// annotations when any is critical.
fn annotations_ok(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0usize;
    let mut cal_count = 0u32;
    let mut cal_critical = false;
    let mut tz_count = 0u32;
    let mut first_cal: Option<alloc::string::String> = None;
    while i < b.len() {
        if b[i] != b'[' {
            return false;
        }
        i += 1;
        let critical = b.get(i) == Some(&b'!');
        if critical {
            i += 1;
        }
        let start = i;
        while i < b.len() && b[i] != b']' {
            i += 1;
        }
        if i >= b.len() {
            return false;
        }
        let content = &s[start..i];
        i += 1; // ']'
        if let Some(eq) = content.find('=') {
            let key = &content[..eq];
            let val = &content[eq + 1..];
            if !valid_annotation_key(key) {
                return false;
            }
            if key == "u-ca" {
                cal_count += 1;
                if critical {
                    cal_critical = true;
                }
                if first_cal.is_none() {
                    first_cal = Some(val.to_ascii_lowercase());
                }
            } else if critical {
                return false;
            }
        } else {
            // A `=`-less annotation is a time-zone annotation.
            tz_count += 1;
        }
    }
    if tz_count >= 2 || (cal_count >= 2 && cal_critical) {
        return false;
    }
    !matches!(first_cal, Some(c) if c != "iso8601")
}

/// Parses a year-month-only string (`[±]YYYY[YY] -? MM`) fully.
fn parse_ym_only(m: &str) -> Option<(i32, u8)> {
    let b = m.as_bytes();
    let mut i = 0usize;
    let (neg, ylen) = match b.first() {
        Some(b'+') => {
            i = 1;
            (false, 6)
        }
        Some(b'-') => {
            i = 1;
            (true, 6)
        }
        Some(0xE2) if b.get(1) == Some(&0x88) && b.get(2) == Some(&0x92) => {
            i = 3;
            (true, 6)
        }
        _ => (false, 4),
    };
    let mut year: i64 = 0;
    for _ in 0..ylen {
        let c = *b.get(i)?;
        if !c.is_ascii_digit() {
            return None;
        }
        year = year * 10 + i64::from(c - b'0');
        i += 1;
    }
    if neg {
        if year == 0 {
            return None;
        }
        year = -year;
    }
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    let mut month: i64 = 0;
    for _ in 0..2 {
        let c = *b.get(i)?;
        if !c.is_ascii_digit() {
            return None;
        }
        month = month * 10 + i64::from(c - b'0');
        i += 1;
    }
    if i != b.len() || !(1..=12).contains(&month) {
        return None;
    }
    if year < i64::from(i32::MIN) || year > i64::from(i32::MAX) {
        return None;
    }
    Some((year as i32, month as u8))
}

/// Whether any `.`/`,` fractional group in `main` has more than 9 digits.
fn has_overlong_fraction(main: &str) -> bool {
    let b = main.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'.' || b[i] == b',' {
            let mut n = 0;
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                n += 1;
                i += 1;
            }
            if n > 9 {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// Parses a Temporal PlainYearMonth string into `(year, month)`. `None` if
/// malformed (does not range-check limits — the caller does).
fn parse_temporal_year_month(s: &str) -> Option<(i32, u8)> {
    // A non-ASCII minus sign (U+2212) is not accepted anywhere.
    if s.contains('\u{2212}') {
        return None;
    }
    let (main, ann) = match s.find('[') {
        Some(idx) => (&s[..idx], &s[idx..]),
        None => (s, ""),
    };
    if !annotations_ok(ann) || has_overlong_fraction(main) {
        return None;
    }
    // A full date/date-time string carries a day; extract its year+month. A
    // string that parses only as a bare time (no date) falls through to the
    // year-month form (e.g. "2000-05" is a year-month, not "20:00-05:00").
    if let Some(p) = temporal_iso::parse_iso_datetime(main)
        && let Some(d) = p.date
    {
        if p.z || (p.offset_ns.is_some() && p.time.is_none()) {
            return None;
        }
        return Some((d.year, d.month));
    }
    parse_ym_only(main)
}

/// Parses a rounding-mode string into a [`RoundMode`].
fn parse_round_mode(s: &str) -> RoundMode {
    match s {
        "ceil" => RoundMode::Ceil,
        "floor" => RoundMode::Floor,
        "expand" => RoundMode::Expand,
        "halfCeil" => RoundMode::HalfCeil,
        "halfFloor" => RoundMode::HalfFloor,
        "halfExpand" => RoundMode::HalfExpand,
        "halfTrunc" => RoundMode::HalfTrunc,
        "halfEven" => RoundMode::HalfEven,
        _ => RoundMode::Trunc,
    }
}

/// `NegateRoundingMode`: swaps ceil/floor directions for the `since` operation.
fn negate_round_mode(m: RoundMode) -> RoundMode {
    match m {
        RoundMode::Ceil => RoundMode::Floor,
        RoundMode::Floor => RoundMode::Ceil,
        RoundMode::HalfCeil => RoundMode::HalfFloor,
        RoundMode::HalfFloor => RoundMode::HalfCeil,
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Engine logic
// ---------------------------------------------------------------------------

impl<'a> Interp<'a> {
    /// A `RangeError` throw with `msg`.
    fn pym_range(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// `ToIntegerWithTruncation`: truncate toward zero. NaN and ±∞ → RangeError.
    /// Symbol/BigInt propagate a TypeError.
    fn pym_to_integer(&mut self, v: NanBox) -> Result<i64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let f = self.realm.to_number(num);
        if !f.is_finite() {
            return Err(self.pym_range("PlainYearMonth: value must be a finite integer"));
        }
        Ok(f.trunc() as i64)
    }

    /// `ToIntegerIfIntegral`: a Number that must be an integer, else RangeError.
    fn pym_to_integral(&mut self, v: NanBox) -> Result<i64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let f = self.realm.to_number(num);
        if !f.is_finite() || f.fract() != 0.0 {
            return Err(self.pym_range("PlainYearMonth: value must be an integer"));
        }
        Ok(f as i64)
    }

    fn is_undef(v: NanBox) -> bool {
        matches!(v.unpack(), Unpacked::Undefined)
    }

    /// The string content of `v` if it is a String value.
    fn as_string_value(&self, v: NanBox) -> Option<alloc::string::String> {
        v.as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
    }

    /// `CanonicalizeCalendar` for the *constructor* form: only a bare calendar id
    /// is accepted (an ISO date string is NOT a valid calendar here).
    fn canonicalize_calendar_strict(&mut self, v: NanBox) -> Result<(), ExecError> {
        if Self::is_undef(v) {
            return Ok(());
        }
        let Some(s) = self.as_string_value(v) else {
            return Err(self.type_error("PlainYearMonth: calendar must be a string"));
        };
        if s.eq_ignore_ascii_case("iso8601") {
            Ok(())
        } else {
            Err(self.pym_range("PlainYearMonth: unknown calendar"))
        }
    }

    /// `ToTemporalCalendarIdentifier` for the property-bag form: a bare id or an
    /// ISO string (whose calendar annotation, if any, must be `iso8601`).
    fn calendar_identifier(&mut self, v: NanBox) -> Result<(), ExecError> {
        if Self::is_undef(v) {
            return Ok(());
        }
        // A *calendared* Temporal object supplies its own calendar (iso8601) via
        // the fast path; non-calendared objects (`{}`, `Duration`, …) are invalid.
        if let Some(cal) = self.temporal_object_calendar(v) {
            return if cal.eq_ignore_ascii_case("iso8601") {
                Ok(())
            } else {
                Err(self.pym_range("PlainYearMonth: unknown calendar"))
            };
        }
        let Some(s) = self.as_string_value(v) else {
            return Err(self.type_error("PlainYearMonth: calendar must be a string"));
        };
        if s.eq_ignore_ascii_case("iso8601") {
            return Ok(());
        }
        if let Some(idx) = s.find("[u-ca=") {
            let rest = &s[idx + 6..];
            let val = rest.split(']').next().unwrap_or("");
            return if val.eq_ignore_ascii_case("iso8601") {
                Ok(())
            } else {
                Err(self.pym_range("PlainYearMonth: unknown calendar"))
            };
        }
        // A negative-zero extended year is not a valid ISO string.
        if s.contains('\u{2212}') || s.starts_with("-000000") {
            return Err(self.pym_range("PlainYearMonth: invalid calendar string"));
        }
        let first = s.as_bytes().first().copied();
        if matches!(first, Some(b) if b.is_ascii_digit() || b == b'+' || b == b'-')
            || s.starts_with('T')
        {
            Ok(())
        } else {
            Err(self.pym_range("PlainYearMonth: unknown calendar"))
        }
    }

    /// Builds a PlainYearMonth instance linked to the intrinsic prototype.
    fn new_year_month(&mut self, date: IsoDate) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainYearMonth,
            date,
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainYearMonth) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    fn new_plain_date(&mut self, date: IsoDate) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainDate,
            date,
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainDate) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    fn pym_new_duration(&mut self, d: crate::temporal_iso::DurationFields) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::Duration,
            duration: d,
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::Duration) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// `GetOptionsObject`: `undefined` → `None`; an object → `Some(h)`; a
    /// primitive → TypeError.
    fn pym_options(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        if Self::is_undef(v) {
            return Ok(None);
        }
        if self.is_object_value(v)
            && let Some(h) = v.as_handle().map(Handle::from_raw)
        {
            return Ok(Some(h));
        }
        Err(self.type_error("PlainYearMonth: options must be an object"))
    }

    /// Reads a string option constrained to `allowed`, with a default.
    fn pym_string_option(
        &mut self,
        opts: Option<Handle>,
        key: &str,
        allowed: &[&str],
        default: &str,
    ) -> Result<alloc::string::String, ExecError> {
        let Some(h) = opts else {
            return Ok(alloc::string::String::from(default));
        };
        let v = self.pym_get(h, key)?;
        if Self::is_undef(v) {
            return Ok(alloc::string::String::from(default));
        }
        let s = self.coerce_to_string(v)?;
        if allowed.contains(&s.as_str()) {
            Ok(s)
        } else {
            Err(self.pym_range(&alloc::format!("PlainYearMonth: invalid {key} option")))
        }
    }

    /// `GetTemporalOverflowOption`.
    fn pym_overflow(&mut self, opts: Option<Handle>) -> Result<Overflow, ExecError> {
        let s = self.pym_string_option(opts, "overflow", &["constrain", "reject"], "constrain")?;
        Ok(if s == "reject" {
            Overflow::Reject
        } else {
            Overflow::Constrain
        })
    }

    /// Reads a member property of an object handle (runs getters).
    fn pym_get(&mut self, h: Handle, key: &str) -> Result<NanBox, ExecError> {
        self.read_member(h, key)
    }

    // -- field resolution ---------------------------------------------------

    /// `CalendarYearMonthFromFields` for iso8601: resolves optional
    /// year/month/monthCode fields into a reference `IsoDate` (day 1).
    fn resolve_year_month(
        &mut self,
        year: Option<i64>,
        month: Option<i64>,
        month_code: Option<(u8, bool)>,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        let Some(year) = year else {
            return Err(self.type_error("PlainYearMonth: missing 'year'"));
        };
        if month.is_none() && month_code.is_none() {
            return Err(self.type_error("PlainYearMonth: missing 'month' or 'monthCode'"));
        }
        let resolved_month: i64 = if let Some((num, leap)) = month_code {
            if leap || !(1..=12).contains(&num) {
                return Err(self.pym_range("PlainYearMonth: invalid monthCode"));
            }
            if let Some(m) = month
                && m != i64::from(num)
            {
                return Err(self.pym_range("PlainYearMonth: month and monthCode conflict"));
            }
            i64::from(num)
        } else {
            let m = month.unwrap();
            if m < 1 {
                return Err(self.pym_range("PlainYearMonth: month out of range"));
            }
            match overflow {
                Overflow::Constrain => m.min(12),
                Overflow::Reject => {
                    if m > 12 {
                        return Err(self.pym_range("PlainYearMonth: month out of range"));
                    }
                    m
                }
            }
        };
        if !ym_within_limits(year, resolved_month) {
            return Err(self.pym_range("PlainYearMonth: out of range"));
        }
        Ok(IsoDate {
            year: year as i32,
            month: resolved_month as u8,
            day: 1,
        })
    }

    /// Reads and coerces the year/month/monthCode fields off a property bag in
    /// spec order (calendar, month, monthCode, year).
    fn read_year_month_fields(
        &mut self,
        h: Handle,
        read_calendar: bool,
    ) -> Result<YearMonthFields, ExecError> {
        // `with` reads (and rejects) the calendar in its own preceding step, so it
        // must not be read again here (observable order).
        if read_calendar {
            let cal = self.pym_get(h, "calendar")?;
            self.calendar_identifier(cal)?;
        }

        let month_v = self.pym_get(h, "month")?;
        let month = if Self::is_undef(month_v) {
            None
        } else {
            // `month` is a `ToPositiveIntegerWithTruncation`: a non-positive value
            // is a RangeError (thrown while reading fields, before options).
            let m = self.pym_to_integer(month_v)?;
            if m < 1 {
                return Err(self.pym_range("PlainYearMonth: month must be positive"));
            }
            Some(m)
        };

        let mc_v = self.pym_get(h, "monthCode")?;
        let month_code = if Self::is_undef(mc_v) {
            None
        } else {
            // A monthCode must coerce (ToPrimitive, string hint) to a String
            // primitive; a number/bigint/etc. is a TypeError.
            let prim = self.coerce_primitive(mc_v, "string")?;
            let Some(s) = self.as_string_value(prim) else {
                return Err(self.type_error("PlainYearMonth: monthCode must be a string"));
            };
            match parse_month_code_syntax(&s) {
                Some(mc) => Some(mc),
                None => return Err(self.pym_range("PlainYearMonth: malformed monthCode")),
            }
        };

        let year_v = self.pym_get(h, "year")?;
        let year = if Self::is_undef(year_v) {
            None
        } else {
            Some(self.pym_to_integer(year_v)?)
        };
        Ok((year, month, month_code))
    }

    /// `ToTemporalYearMonth`: cast an arbitrary value (instance / property bag /
    /// string) to a reference `IsoDate`.
    fn cast_year_month(&mut self, v: NanBox, options: NanBox) -> Result<IsoDate, ExecError> {
        if self.is_object_value(v) {
            let h = v.as_handle().map(Handle::from_raw).unwrap();
            // A PlainYearMonth instance is copied (only options are read).
            if let Some(data) = self.realm.temporal_at(h)
                && data.kind == TemporalKind::PlainYearMonth
            {
                let opts = self.pym_options(options)?;
                self.pym_overflow(opts)?;
                return Ok(data.date);
            }
            let (year, month, mc) = self.read_year_month_fields(h, true)?;
            let opts = self.pym_options(options)?;
            let overflow = self.pym_overflow(opts)?;
            return self.resolve_year_month(year, month, mc, overflow);
        }
        // A string is parsed; any other primitive is a TypeError. The string is
        // parsed *before* the overflow option is read (spec order).
        let Some(s) = self.as_string_value(v) else {
            return Err(self.type_error("PlainYearMonth: expected an object or string"));
        };
        let Some((year, month)) = parse_temporal_year_month(&s) else {
            return Err(self.pym_range("PlainYearMonth: invalid string"));
        };
        if !ym_within_limits(i64::from(year), i64::from(month)) {
            return Err(self.pym_range("PlainYearMonth: out of range"));
        }
        let opts = self.pym_options(options)?;
        self.pym_overflow(opts)?;
        Ok(IsoDate {
            year,
            month,
            day: 1,
        })
    }

    // -- duration reading (for add/subtract) --------------------------------

    /// Reads a Temporal.Duration-like value (instance / property bag / string).
    fn read_duration(
        &mut self,
        v: NanBox,
    ) -> Result<crate::temporal_iso::DurationFields, ExecError> {
        use crate::temporal_iso::DurationFields;
        if self.is_object_value(v) {
            let h = v.as_handle().map(Handle::from_raw).unwrap();
            if let Some(data) = self.realm.temporal_at(h)
                && data.kind == TemporalKind::Duration
            {
                return Ok(data.duration);
            }
            // Duration fields are read in alphabetical order (ToTemporalDuration).
            let names = [
                "days",
                "hours",
                "microseconds",
                "milliseconds",
                "minutes",
                "months",
                "nanoseconds",
                "seconds",
                "weeks",
                "years",
            ];
            let mut vals = [0i64; 10];
            let mut any = false;
            for (i, name) in names.iter().enumerate() {
                let fv = self.pym_get(h, name)?;
                if !Self::is_undef(fv) {
                    any = true;
                    vals[i] = self.pym_to_integral(fv)?;
                }
            }
            if !any {
                return Err(self.type_error("PlainYearMonth: invalid duration"));
            }
            let d = DurationFields {
                days: vals[0],
                hours: vals[1],
                microseconds: vals[2],
                milliseconds: vals[3],
                minutes: vals[4],
                months: vals[5],
                nanoseconds: vals[6],
                seconds: vals[7],
                weeks: vals[8],
                years: vals[9],
            };
            if !d.is_valid() {
                return Err(self.pym_range("PlainYearMonth: duration has mixed signs"));
            }
            return Ok(d);
        }
        let Some(s) = self.as_string_value(v) else {
            return Err(self.type_error("PlainYearMonth: expected a duration"));
        };
        temporal_iso::parse_iso_duration(&s)
            .ok_or_else(|| self.pym_range("PlainYearMonth: invalid duration string"))
    }

    // -- constructor --------------------------------------------------------

    /// `new Temporal.PlainYearMonth(...)`.
    pub(crate) fn plainyearmonth_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        if Self::is_undef(new_target) {
            return Err(self.type_error("Temporal.PlainYearMonth must be called with new"));
        }
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        let year = self.pym_to_integer(arg(0))?;
        let month = self.pym_to_integer(arg(1))?;
        self.canonicalize_calendar_strict(arg(2))?;
        let ref_day = if Self::is_undef(arg(3)) {
            1
        } else {
            self.pym_to_integer(arg(3))?
        };
        if !ym_within_limits(year, month) {
            return Err(self.pym_range("PlainYearMonth: out of range"));
        }
        // RejectISODate on (year, month, referenceDay).
        let Some(date) = regulate_iso_date(year as i32, month, ref_day, Overflow::Reject) else {
            return Err(self.pym_range("PlainYearMonth: invalid ISO date"));
        };
        let data = TemporalData {
            kind: TemporalKind::PlainYearMonth,
            date,
            ..Default::default()
        };
        self.finish_temporal(data, new_target, callee)
    }

    // -- methods ------------------------------------------------------------

    pub(crate) fn plainyearmonth_method(
        &mut self,
        this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method {
            "with" => self.pym_with(data, arg(0), arg(1)),
            "add" => self.pym_add_sub(data, arg(0), arg(1), false),
            "subtract" => self.pym_add_sub(data, arg(0), arg(1), true),
            "until" => self.pym_diff(data, arg(0), arg(1), false),
            "since" => self.pym_diff(data, arg(0), arg(1), true),
            "equals" => {
                let other = self.cast_year_month(arg(0), NanBox::undefined())?;
                Ok(NanBox::boolean(other == data.date))
            }
            "toPlainDate" => self.pym_to_plain_date(data, arg(0)),
            "toString" => {
                let opts = self.pym_options(arg(0))?;
                let show = self.pym_string_option(
                    opts,
                    "calendarName",
                    &["auto", "always", "never", "critical"],
                    "auto",
                )?;
                let s = self.pym_to_string(data, &show);
                Ok(self.new_str(&s))
            }
            "toJSON" | "toLocaleString" => {
                let s = self.pym_to_string(data, "auto");
                Ok(self.new_str(&s))
            }
            "valueOf" => Err(self.type_error(
                "Temporal.PlainYearMonth.prototype.valueOf: use compare() or an explicit conversion",
            )),
            _ => {
                let _ = this;
                Err(self.temporal_todo(&alloc::format!("PlainYearMonth.prototype.{method}")))
            }
        }
    }

    fn pym_with(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        if !self.is_object_value(arg) {
            return Err(self.type_error("PlainYearMonth.with: argument must be an object"));
        }
        let h = arg.as_handle().map(Handle::from_raw).unwrap();
        // `IsPartialTemporalObject`: a Temporal-branded object is not a partial bag.
        if self.realm.temporal_at(h).is_some() {
            return Err(self.type_error("PlainYearMonth.with: argument must be a plain object"));
        }
        // RejectObjectWithCalendarOrTimeZone (calendar/timeZone read here, once).
        let cal = self.pym_get(h, "calendar")?;
        if !Self::is_undef(cal) {
            return Err(self.type_error("PlainYearMonth.with: unexpected calendar field"));
        }
        let tz = self.pym_get(h, "timeZone")?;
        if !Self::is_undef(tz) {
            return Err(self.type_error("PlainYearMonth.with: unexpected timeZone field"));
        }
        let (in_year, in_month, in_mc) = self.read_year_month_fields(h, false)?;
        // At least one recognised field is required (`IsPartialTemporalObject`).
        if in_year.is_none() && in_month.is_none() && in_mc.is_none() {
            return Err(self.type_error("PlainYearMonth.with: no recognised fields"));
        }
        let opts = self.pym_options(options)?;
        let overflow = self.pym_overflow(opts)?;

        // Merge with the receiver.
        let year = in_year.or(Some(i64::from(data.date.year)));
        let (month, mc) = if in_month.is_some() || in_mc.is_some() {
            (in_month, in_mc)
        } else {
            (Some(i64::from(data.date.month)), None)
        };
        let date = self.resolve_year_month(year, month, mc, overflow)?;
        Ok(self.new_year_month(date))
    }

    fn pym_add_sub(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
        options: NanBox,
        subtract: bool,
    ) -> Result<NanBox, ExecError> {
        let mut dur = self.read_duration(arg)?;
        let opts = self.pym_options(options)?;
        // overflow is validated but has no effect for the ISO calendar.
        self.pym_overflow(opts)?;
        if subtract {
            dur = crate::temporal_iso::DurationFields {
                years: -dur.years,
                months: -dur.months,
                weeks: -dur.weeks,
                days: -dur.days,
                hours: -dur.hours,
                minutes: -dur.minutes,
                seconds: -dur.seconds,
                milliseconds: -dur.milliseconds,
                microseconds: -dur.microseconds,
                nanoseconds: -dur.nanoseconds,
            };
        }
        // A PlainYearMonth cannot represent any unit smaller than a month.
        if dur.weeks != 0
            || dur.days != 0
            || dur.hours != 0
            || dur.minutes != 0
            || dur.seconds != 0
            || dur.milliseconds != 0
            || dur.microseconds != 0
            || dur.nanoseconds != 0
        {
            return Err(self.pym_range("PlainYearMonth: cannot add units smaller than months"));
        }
        // Only years and months affect a year-month; add them directly (dropping
        // the reference day). Working in i64 avoids the i32 wrap that
        // `add_iso_date` suffers for out-of-range durations.
        let m0 = (i64::from(data.date.month) - 1) + dur.months;
        let result_year = i64::from(data.date.year) + dur.years + m0.div_euclid(12);
        let result_month = m0.rem_euclid(12) + 1;
        if !ym_within_limits(result_year, result_month) {
            return Err(self.pym_range("PlainYearMonth: result out of range"));
        }
        Ok(self.new_year_month(IsoDate {
            year: result_year as i32,
            month: result_month as u8,
            day: 1,
        }))
    }

    fn pym_diff(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
        options: NanBox,
        since: bool,
    ) -> Result<NanBox, ExecError> {
        let other = self.cast_year_month(arg, NanBox::undefined())?;
        let opts = self.pym_options(options)?;

        // GetDifferenceSettings reads *all* option properties (largestUnit,
        // roundingIncrement, roundingMode, smallestUnit) — accepting any valid
        // unit name — before any algorithmic validation of which units are
        // allowed for this operation.
        const ALL_UNITS: &[&str] = &[
            "year",
            "years",
            "month",
            "months",
            "week",
            "weeks",
            "day",
            "days",
            "hour",
            "hours",
            "minute",
            "minutes",
            "second",
            "seconds",
            "millisecond",
            "milliseconds",
            "microsecond",
            "microseconds",
            "nanosecond",
            "nanoseconds",
        ];
        let largest_allowed: alloc::vec::Vec<&str> = core::iter::once("auto")
            .chain(ALL_UNITS.iter().copied())
            .collect();
        let largest_s = self.pym_string_option(opts, "largestUnit", &largest_allowed, "auto")?;
        let increment = self.pym_rounding_increment(opts)?;
        let mode_s = self.pym_string_option(
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
            "trunc",
        )?;
        let smallest_s = self.pym_string_option(opts, "smallestUnit", ALL_UNITS, "month")?;

        // Only year / month units are meaningful for a PlainYearMonth.
        let unit_ok = |u: &str| matches!(u, "auto" | "year" | "years" | "month" | "months");
        if !unit_ok(&largest_s) || !unit_ok(&smallest_s) {
            return Err(self.pym_range("PlainYearMonth: unsupported unit for difference"));
        }
        let smallest_year = smallest_s.starts_with("year");
        let largest_year = matches!(largest_s.as_str(), "year" | "years" | "auto");
        // largestUnit must be >= smallestUnit; a year is larger than a month.
        if smallest_year && !largest_year {
            return Err(self.pym_range("PlainYearMonth: largestUnit smaller than smallestUnit"));
        }

        // The difference is measured between the *first days* of the two months,
        // so each operand's reference (day-1) date must be representable — a
        // stricter bound than ISOYearMonthWithinLimits at the boundary months.
        let this_ref = IsoDate {
            year: data.date.year,
            month: data.date.month,
            day: 1,
        };
        let other_ref = IsoDate {
            year: other.year,
            month: other.month,
            day: 1,
        };
        if !iso_date_in_range(this_ref) || !iso_date_in_range(other_ref) {
            return Err(self.pym_range("PlainYearMonth: date out of representable range"));
        }

        // Both operands are the first day of a month, so the difference is an
        // exact whole number of months (this → other). Computing it directly
        // avoids `difference_iso_date`, which can loop for dates whose
        // whole-year intermediate falls outside the representable range.
        let total_months = (i64::from(other.year) - i64::from(data.date.year)) * 12
            + (i64::from(other.month) - i64::from(data.date.month));
        let (years, months) = if largest_year {
            (total_months / 12, total_months % 12)
        } else {
            (0, total_months)
        };

        let mode = parse_round_mode(&mode_s);
        let mode = if since { negate_round_mode(mode) } else { mode };
        let (mut ry, mut rm) =
            round_year_month(years, months, smallest_year, largest_year, increment, mode);
        if since {
            ry = -ry;
            rm = -rm;
        }

        let d = crate::temporal_iso::DurationFields {
            years: ry,
            months: rm,
            ..Default::default()
        };
        Ok(self.pym_new_duration(d))
    }

    fn pym_rounding_increment(&mut self, opts: Option<Handle>) -> Result<i128, ExecError> {
        let Some(h) = opts else {
            return Ok(1);
        };
        let v = self.pym_get(h, "roundingIncrement")?;
        if Self::is_undef(v) {
            return Ok(1);
        }
        let num = self.coerce_to_number(v)?;
        let f = self.realm.to_number(num);
        if !f.is_finite() {
            return Err(self.pym_range("PlainYearMonth: invalid roundingIncrement"));
        }
        let n = f.trunc();
        if !(1.0..=1_000_000_000.0).contains(&n) {
            return Err(self.pym_range("PlainYearMonth: invalid roundingIncrement"));
        }
        Ok(n as i128)
    }

    fn pym_to_plain_date(&mut self, data: &TemporalData, arg: NanBox) -> Result<NanBox, ExecError> {
        if !self.is_object_value(arg) {
            return Err(self.type_error("PlainYearMonth.toPlainDate: argument must be an object"));
        }
        let h = arg.as_handle().map(Handle::from_raw).unwrap();
        let day_v = self.pym_get(h, "day")?;
        if Self::is_undef(day_v) {
            return Err(self.type_error("PlainYearMonth.toPlainDate: missing 'day'"));
        }
        let day = self.pym_to_integer(day_v)?;
        let Some(date) = regulate_iso_date(
            data.date.year,
            i64::from(data.date.month),
            day,
            Overflow::Constrain,
        ) else {
            return Err(self.pym_range("PlainYearMonth.toPlainDate: invalid date"));
        };
        // `ISODateWithinLimits` for a *PlainDate*: epoch days in `[MIN-1, MAX]`
        // (unlike `iso_date_in_range`, a PlainDate may not sit one day past MAX).
        let ed = crate::temporal_iso::iso_to_epoch_days(date);
        if !(MIN_EPOCH_DAYS - 1..=MAX_EPOCH_DAYS).contains(&ed) {
            return Err(self.pym_range("PlainYearMonth.toPlainDate: date out of range"));
        }
        Ok(self.new_plain_date(date))
    }

    fn pym_to_string(&mut self, data: &TemporalData, show: &str) -> alloc::string::String {
        let mut out = alloc::format!(
            "{}-{}",
            format_iso_year(data.date.year),
            pad(u64::from(data.date.month), 2)
        );
        match show {
            "always" => out.push_str(&alloc::format!(
                "-{}[u-ca=iso8601]",
                pad(u64::from(data.date.day), 2)
            )),
            "critical" => out.push_str(&alloc::format!(
                "-{}[!u-ca=iso8601]",
                pad(u64::from(data.date.day), 2)
            )),
            _ => {}
        }
        out
    }

    // -- getters ------------------------------------------------------------

    pub(crate) fn plainyearmonth_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let d = data.date;
        Ok(match name {
            "year" => NanBox::number(f64::from(d.year)),
            "month" => NanBox::number(f64::from(d.month)),
            "monthCode" => {
                let s = month_code(d.month);
                self.new_str(&s)
            }
            "calendarId" => self.new_str("iso8601"),
            "daysInMonth" => NanBox::number(f64::from(iso_days_in_month(d.year, d.month))),
            "daysInYear" => NanBox::number(f64::from(iso_days_in_year(d.year))),
            "monthsInYear" => NanBox::number(12.0),
            "inLeapYear" => NanBox::boolean(is_leap_year(d.year)),
            "era" | "eraYear" => NanBox::undefined(),
            _ => return Err(self.temporal_todo(&alloc::format!("PlainYearMonth getter {name}"))),
        })
    }

    // -- statics ------------------------------------------------------------

    pub(crate) fn plainyearmonth_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method {
            "from" => {
                let date = self.cast_year_month(arg(0), arg(1))?;
                Ok(Some(self.new_year_month(date)))
            }
            "compare" => {
                let a = self.cast_year_month(arg(0), NanBox::undefined())?;
                let b = self.cast_year_month(arg(1), NanBox::undefined())?;
                let n = match crate::temporal_iso::compare_iso_date(a, b) {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Equal => 0.0,
                    core::cmp::Ordering::Greater => 1.0,
                };
                Ok(Some(NanBox::number(n)))
            }
            _ => Ok(None),
        }
    }
}

/// Rounds a (years, months) difference per the smallest/largest units.
fn round_year_month(
    years: i64,
    months: i64,
    smallest_year: bool,
    largest_year: bool,
    increment: i128,
    mode: RoundMode,
) -> (i64, i64) {
    if smallest_year {
        // Round (years + months/12) to `increment` years, working in twelfths.
        let twelfths = i128::from(years) * 12 + i128::from(months);
        let rounded = round_to_increment(twelfths, increment * 12, mode);
        ((rounded / 12) as i64, 0)
    } else if largest_year {
        // Round the months component; keep years, carrying any overflow.
        let rm = round_to_increment(i128::from(months), increment, mode) as i64;
        let mut y = years;
        let mut m = rm;
        if m.abs() >= 12 {
            y += m / 12;
            m %= 12;
        }
        (y, m)
    } else {
        // largestUnit month, smallestUnit month: round the total months.
        let total = i128::from(years) * 12 + i128::from(months);
        (0, round_to_increment(total, increment, mode) as i64)
    }
}
