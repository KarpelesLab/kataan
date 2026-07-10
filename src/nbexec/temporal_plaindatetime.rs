//! `Temporal.PlainDateTime` — logic module. A fan-out unit: everything specific to
//! `PlainDateTime` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
//!
//! A `PlainDateTime` is a calendar date plus a wall-clock time (no time zone),
//! calendar `"iso8601"`. It stores both `TemporalData.date` (`IsoDate`) and
//! `TemporalData.time` (`IsoTime`) and is essentially `PlainDate + PlainTime`
//! combined, reusing the shared `crate::temporal_iso` helpers.
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    self as iso, DurationFields, IsoDate, IsoTime, Overflow, RoundMode, TemporalData, TemporalKind,
    Unit,
};

/// Prototype method names installed on `Temporal.PlainDateTime.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "with",
    "withPlainTime",
    "withCalendar",
    "add",
    "subtract",
    "until",
    "since",
    "round",
    "equals",
    "toPlainDate",
    "toPlainTime",
    "toZonedDateTime",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
];
/// Getter-accessor names installed on `Temporal.PlainDateTime.prototype`.
pub(crate) const GETTERS: &[&str] = &[
    "calendarId",
    "era",
    "eraYear",
    "year",
    "month",
    "monthCode",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
    "dayOfWeek",
    "dayOfYear",
    "weekOfYear",
    "yearOfWeek",
    "daysInWeek",
    "daysInMonth",
    "daysInYear",
    "monthsInYear",
    "inLeapYear",
];

/// The recognised calendar/time fields of a date-time property bag.
#[derive(Default)]
struct DtBag {
    year: Option<i64>,
    month: Option<i64>,
    /// `(month number, is-leap-suffix)` — well-formed syntax only; the ISO
    /// suitability (range 1..=12, no leap) is checked later during resolution.
    month_code: Option<(i64, bool)>,
    day: Option<i64>,
    hour: Option<i64>,
    minute: Option<i64>,
    second: Option<i64>,
    ms: Option<i64>,
    us: Option<i64>,
    ns: Option<i64>,
}

impl DtBag {
    fn any(&self) -> bool {
        self.year.is_some()
            || self.month.is_some()
            || self.month_code.is_some()
            || self.day.is_some()
            || self.hour.is_some()
            || self.minute.is_some()
            || self.second.is_some()
            || self.ms.is_some()
            || self.us.is_some()
            || self.ns.is_some()
    }

    fn any_time(&self) -> bool {
        self.hour.is_some()
            || self.minute.is_some()
            || self.second.is_some()
            || self.ms.is_some()
            || self.us.is_some()
            || self.ns.is_some()
    }
}

/// Nanoseconds in one unit of `u` (Day..Nanosecond).
fn unit_ns(u: Unit) -> i128 {
    match u {
        Unit::Day => iso::NS_PER_DAY,
        Unit::Hour => iso::NS_PER_HOUR,
        Unit::Minute => iso::NS_PER_MINUTE,
        Unit::Second => iso::NS_PER_SEC,
        Unit::Millisecond => 1_000_000,
        Unit::Microsecond => 1_000,
        _ => 1,
    }
}

/// Parses a Temporal duration/round unit name (singular or plural).
fn parse_unit(s: &str) -> Option<Unit> {
    Some(match s {
        "year" | "years" => Unit::Year,
        "month" | "months" => Unit::Month,
        "week" | "weeks" => Unit::Week,
        "day" | "days" => Unit::Day,
        "hour" | "hours" => Unit::Hour,
        "minute" | "minutes" => Unit::Minute,
        "second" | "seconds" => Unit::Second,
        "millisecond" | "milliseconds" => Unit::Millisecond,
        "microsecond" | "microseconds" => Unit::Microsecond,
        "nanosecond" | "nanoseconds" => Unit::Nanosecond,
        _ => return None,
    })
}

/// Parses a `roundingMode` option name.
fn parse_round_mode(s: &str) -> Option<RoundMode> {
    Some(match s {
        "ceil" => RoundMode::Ceil,
        "floor" => RoundMode::Floor,
        "expand" => RoundMode::Expand,
        "trunc" => RoundMode::Trunc,
        "halfCeil" => RoundMode::HalfCeil,
        "halfFloor" => RoundMode::HalfFloor,
        "halfExpand" => RoundMode::HalfExpand,
        "halfTrunc" => RoundMode::HalfTrunc,
        "halfEven" => RoundMode::HalfEven,
        _ => return None,
    })
}

/// Parses the *syntax* of an ISO month code (`"M05"`, or `"M05L"` with a leap
/// suffix), returning `(month number, is-leap)`. `None` for a malformed code.
/// The numeric range and the leap-suffix rejection (ISO forbids leap months) are
/// enforced separately, at resolution time, so a well-formed but unsuitable code
/// still throws — but only after the required-field / type checks.
fn parse_month_code(s: &str) -> Option<(i64, bool)> {
    let b = s.as_bytes();
    // `M` + exactly two digits, optionally a trailing `L` (leap month).
    if !(b.len() == 3 || b.len() == 4)
        || b[0] != b'M'
        || !b[1].is_ascii_digit()
        || !b[2].is_ascii_digit()
        || (b.len() == 4 && b[3] != b'L')
    {
        return None;
    }
    Some((
        i64::from(b[1] - b'0') * 10 + i64::from(b[2] - b'0'),
        b.len() == 4,
    ))
}

impl<'a> Interp<'a> {
    /// `new Temporal.PlainDateTime(...)`.
    pub(crate) fn plaindatetime_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        // isoYear/isoMonth/isoDay: always ToIntegerWithTruncation (undefined → NaN
        // → RangeError). hour…nanosecond: default 0 when undefined.
        let year = self.pdt_to_int(arg(0))?;
        let month = self.pdt_to_int(arg(1))?;
        let day = self.pdt_to_int(arg(2))?;
        let hour = self.pdt_to_int_opt(arg(3))?;
        let minute = self.pdt_to_int_opt(arg(4))?;
        let second = self.pdt_to_int_opt(arg(5))?;
        let ms = self.pdt_to_int_opt(arg(6))?;
        let us = self.pdt_to_int_opt(arg(7))?;
        let ns = self.pdt_to_int_opt(arg(8))?;
        // calendar (10th arg): must be a String naming "iso8601" (case-insensitive).
        self.pdt_calendar_arg(arg(9))?;

        let date = self.pdt_regulate_date(year, month, day, Overflow::Reject)?;
        let time = iso::regulate_iso_time(hour, minute, second, ms, us, ns, Overflow::Reject)
            .ok_or_else(|| self.pdt_range("invalid ISO time"))?;
        if !pdt_in_range(date, time) {
            return Err(self.pdt_range("PlainDateTime outside representable range"));
        }
        let data = TemporalData {
            kind: TemporalKind::PlainDateTime,
            date,
            time,
            ..Default::default()
        };
        Ok(self.finish_temporal(data, new_target, callee))
    }

    /// A `Temporal.PlainDateTime.prototype.<method>()` call.
    pub(crate) fn plaindatetime_method(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method {
            "with" => self.pdt_with(data, arg(0), arg(1)),
            "withPlainTime" => self.pdt_with_plain_time(data, arg(0)),
            "withCalendar" => self.pdt_with_calendar(data, arg(0)),
            "add" => self.pdt_add(data, arg(0), arg(1), 1),
            "subtract" => self.pdt_add(data, arg(0), arg(1), -1),
            "until" => self.pdt_diff(data, arg(0), arg(1), false),
            "since" => self.pdt_diff(data, arg(0), arg(1), true),
            "round" => self.pdt_round(data, arg(0)),
            "equals" => {
                let (d2, t2) = self.pdt_to_datetime(arg(0), NanBox::undefined())?;
                let eq = data.date == d2 && data.time == t2;
                Ok(NanBox::boolean(eq))
            }
            "toPlainDate" => Ok(self.pdt_make_date(data.date)),
            "toPlainTime" => Ok(self.pdt_make_time(data.time)),
            "toString" => self.pdt_to_string(data, arg(0)),
            "toJSON" | "toLocaleString" => self.pdt_to_string(data, NanBox::undefined()),
            "valueOf" => Err(self.type_error(
                "Temporal.PlainDateTime.prototype.valueOf must not be called; use compare() or an \
                 explicit conversion",
            )),
            "toZonedDateTime" => {
                // Interpret this wall-clock date-time in `timeZone` → an exact
                // instant (epoch ns = local wall ns − zone offset).
                let tz = self.temporal_tz_arg(arg(0))?;
                let local_ns = crate::temporal_iso::iso_to_epoch_days(data.date) as i128
                    * crate::temporal_iso::NS_PER_DAY
                    + crate::temporal_iso::time_to_nanos(data.time);
                let offset = self.temporal_tz_offset_ns(&tz, local_ns).unwrap_or(0);
                Ok(self.build_temporal(crate::temporal_iso::TemporalData {
                    kind: crate::temporal_iso::TemporalKind::ZonedDateTime,
                    epoch_ns: local_ns - offset,
                    tz: Some(tz),
                    ..Default::default()
                }))
            }
            _ => Err(self.temporal_todo(&alloc::format!("PlainDateTime.prototype.{method}"))),
        }
    }

    /// A `Temporal.PlainDateTime.prototype.<getter>` read.
    pub(crate) fn plaindatetime_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let d = data.date;
        let t = data.time;
        let num = |n: i64| NanBox::number(n as f64);
        Ok(match name {
            "calendarId" => self.new_str("iso8601"),
            "era" | "eraYear" => NanBox::undefined(),
            "year" => num(i64::from(d.year)),
            "month" => num(i64::from(d.month)),
            "monthCode" => {
                let s = alloc::format!("M{}", iso::pad(u64::from(d.month), 2));
                self.new_str(&s)
            }
            "day" => num(i64::from(d.day)),
            "hour" => num(i64::from(t.hour)),
            "minute" => num(i64::from(t.minute)),
            "second" => num(i64::from(t.second)),
            "millisecond" => num(i64::from(t.millisecond)),
            "microsecond" => num(i64::from(t.microsecond)),
            "nanosecond" => num(i64::from(t.nanosecond)),
            "dayOfWeek" => num(i64::from(iso::iso_day_of_week(d))),
            "dayOfYear" => num(i64::from(iso::iso_day_of_year(d))),
            "weekOfYear" => num(i64::from(iso::iso_week_of_year(d).0)),
            "yearOfWeek" => num(i64::from(iso::iso_week_of_year(d).1)),
            "daysInWeek" => num(7),
            "daysInMonth" => num(i64::from(iso::iso_days_in_month(d.year, d.month))),
            "daysInYear" => num(i64::from(iso::iso_days_in_year(d.year))),
            "monthsInYear" => num(12),
            "inLeapYear" => NanBox::boolean(iso::is_leap_year(d.year)),
            _ => return Err(self.temporal_todo(&alloc::format!("PlainDateTime getter {name}"))),
        })
    }

    /// A `Temporal.PlainDateTime.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn plaindatetime_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method {
            "from" => {
                let (date, time) = self.pdt_to_datetime(arg(0), arg(1))?;
                Ok(Some(self.pdt_make(date, time)))
            }
            "compare" => {
                let (d1, t1) = self.pdt_to_datetime(arg(0), NanBox::undefined())?;
                let (d2, t2) = self.pdt_to_datetime(arg(1), NanBox::undefined())?;
                let ord = iso::compare_iso_date(d1, d2).then(iso::compare_iso_time(t1, t2));
                Ok(Some(NanBox::number(match ord {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Greater => 1.0,
                    core::cmp::Ordering::Equal => 0.0,
                })))
            }
            _ => Ok(None),
        }
    }

    // --- error / coercion helpers ------------------------------------------

    fn pdt_range(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// `ToIntegerWithTruncation`: ToNumber then truncate; a non-finite value
    /// (`NaN`/`±∞`) throws a RangeError.
    fn pdt_to_int(&mut self, v: NanBox) -> Result<i64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.pdt_range("value must be a finite integer"));
        }
        Ok(n.trunc() as i64)
    }

    /// Like [`Self::pdt_to_int`] but `undefined` defaults to `0` (optional time
    /// components of the constructor).
    fn pdt_to_int_opt(&mut self, v: NanBox) -> Result<i64, ExecError> {
        if v.is_undefined() {
            Ok(0)
        } else {
            self.pdt_to_int(v)
        }
    }

    /// `RegulateISODate` with an in-representable-`i32`-year guard.
    fn pdt_regulate_date(
        &mut self,
        year: i64,
        month: i64,
        day: i64,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        if i32::try_from(year).is_err() {
            return Err(self.pdt_range("year outside representable range"));
        }
        iso::regulate_iso_date(year as i32, month, day, overflow)
            .ok_or_else(|| self.pdt_range("invalid ISO date"))
    }

    /// Validates a constructor/`withCalendar` calendar argument: it must be a
    /// primitive String naming `iso8601` (case-insensitive). Non-strings →
    /// TypeError; unknown calendar → RangeError. `undefined` is accepted.
    fn pdt_calendar_arg(&mut self, v: NanBox) -> Result<(), ExecError> {
        if v.is_undefined() {
            return Ok(());
        }
        if let Some(cal) = self.temporal_object_calendar(v) {
            return if cal.eq_ignore_ascii_case("iso8601") {
                Ok(())
            } else {
                Err(self.pdt_range("only the iso8601 calendar is supported"))
            };
        }
        let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        if s.eq_ignore_ascii_case("iso8601") {
            Ok(())
        } else {
            Err(self.pdt_range("only the iso8601 calendar is supported"))
        }
    }

    /// Reads and validates the optional `calendar` field of a property bag (for
    /// `from`/`compare`/`equals`/`until`/`since`). A string must name `iso8601`
    /// (case-insensitive); a non-string, non-object value → TypeError.
    fn pdt_bag_calendar(&mut self, h: Handle) -> Result<(), ExecError> {
        let Some(v) = self.pdt_field(h, "calendar")? else {
            return Ok(());
        };
        // A *calendared* Temporal object supplies its `[[Calendar]]` via the fast
        // path (no property read). Non-calendared objects (`{}`, `Duration`, …)
        // and every other non-string value are a TypeError.
        if let Some(cal) = self.temporal_object_calendar(v) {
            return if cal.eq_ignore_ascii_case("iso8601") {
                Ok(())
            } else {
                Err(self.pdt_range("only the iso8601 calendar is supported"))
            };
        }
        if let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|x| self.realm.string_value(x))
        {
            // A calendar identifier (`"iso8601"`, case-insensitive) or a date-ish
            // ISO string whose `[u-ca=…]` annotations (if any) all name `iso8601`.
            // Partial forms (`"2020-01"`, `"01-01"`) are accepted for the calendar
            // slot even though they are not full PlainDateTime strings.
            if pdt_calendar_string_ok(&s) {
                Ok(())
            } else {
                Err(self.pdt_range("only the iso8601 calendar is supported"))
            }
        } else {
            Err(self.type_error("calendar must be a string"))
        }
    }

    // --- options helpers ---------------------------------------------------

    /// `GetOptionsObject`: `undefined` → no options; an object → that object;
    /// anything else → TypeError.
    fn pdt_options(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        if v.is_undefined() {
            Ok(None)
        } else if self.is_object_value(v) {
            Ok(v.as_handle().map(Handle::from_raw))
        } else {
            Err(self.type_error("options must be an object or undefined"))
        }
    }

    /// Reads a string-valued option, validating it against `allowed`.
    fn pdt_str_option(
        &mut self,
        opts: Option<Handle>,
        key: &str,
        allowed: &[&str],
    ) -> Result<Option<alloc::string::String>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, key)?;
        if v.is_undefined() {
            return Ok(None);
        }
        let s = self.coerce_to_string(v)?;
        if allowed.contains(&s.as_str()) {
            Ok(Some(s))
        } else {
            Err(self.pdt_range(&alloc::format!("invalid value for option {key}")))
        }
    }

    fn pdt_overflow(&mut self, opts: Option<Handle>) -> Result<Overflow, ExecError> {
        Ok(
            match self
                .pdt_str_option(opts, "overflow", &["constrain", "reject"])?
                .as_deref()
            {
                Some("reject") => Overflow::Reject,
                _ => Overflow::Constrain,
            },
        )
    }

    fn pdt_rounding_mode(
        &mut self,
        opts: Option<Handle>,
        default: RoundMode,
    ) -> Result<RoundMode, ExecError> {
        let allowed = [
            "ceil",
            "floor",
            "expand",
            "trunc",
            "halfCeil",
            "halfFloor",
            "halfExpand",
            "halfTrunc",
            "halfEven",
        ];
        Ok(match self.pdt_str_option(opts, "roundingMode", &allowed)? {
            Some(s) => parse_round_mode(&s).unwrap_or(default),
            None => default,
        })
    }

    fn pdt_rounding_increment(&mut self, opts: Option<Handle>) -> Result<i64, ExecError> {
        let Some(h) = opts else { return Ok(1) };
        let v = self.read_member(h, "roundingIncrement")?;
        if v.is_undefined() {
            return Ok(1);
        }
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        // `GetRoundingIncrementOption`: finite and ≥ 1, then truncated toward zero.
        if !n.is_finite() || n < 1.0 {
            return Err(self.pdt_range("roundingIncrement must be a positive integer"));
        }
        Ok(n.trunc() as i64)
    }

    // --- property-bag reading ---------------------------------------------

    /// Reads a single field of `h`, returning `None` when it is `undefined`.
    fn pdt_field(&mut self, h: Handle, key: &str) -> Result<Option<NanBox>, ExecError> {
        let v = self.read_member(h, key)?;
        Ok((!v.is_undefined()).then_some(v))
    }

    /// `PrepareTemporalFields` for a date-time property bag, reading the recognised
    /// keys in alphabetical order (observable order-of-operations).
    fn pdt_read_bag(&mut self, h: Handle) -> Result<DtBag, ExecError> {
        let mut bag = DtBag::default();
        if let Some(v) = self.pdt_field(h, "day")? {
            bag.day = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "hour")? {
            bag.hour = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "microsecond")? {
            bag.us = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "millisecond")? {
            bag.ms = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "minute")? {
            bag.minute = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "month")? {
            bag.month = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "monthCode")? {
            // `monthCode` is `ToPrimitiveAndRequireString`: ToPrimitive(string), then
            // the result must itself be a String (a Number/Boolean/Symbol/BigInt, or
            // a `toString` that returns a non-string, → TypeError).
            let prim = self.coerce_primitive(v, "string")?;
            let Some(s) = prim
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|x| self.realm.string_value(x))
            else {
                return Err(self.type_error("monthCode must be a string"));
            };
            // Only the *syntax* is validated eagerly (before later fields are read).
            let mc = parse_month_code(&s).ok_or_else(|| self.pdt_range("invalid monthCode"))?;
            bag.month_code = Some(mc);
        }
        if let Some(v) = self.pdt_field(h, "nanosecond")? {
            bag.ns = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "second")? {
            bag.second = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "year")? {
            bag.year = Some(self.pdt_to_int(v)?);
        }
        Ok(bag)
    }

    /// Resolves `month`/`monthCode` fields into a month number (checking that they
    /// agree when both are present, and that a `monthCode` is in range). `fallback`
    /// supplies the default.
    fn pdt_resolve_month(
        &mut self,
        month: Option<i64>,
        code: Option<(i64, bool)>,
        fallback: Option<i64>,
    ) -> Result<i64, ExecError> {
        // Suitability of a well-formed `monthCode`: no leap suffix, in range 1..=12.
        let coded = match code {
            Some((c, leap)) => {
                if leap || !(1..=12).contains(&c) {
                    return Err(self.pdt_range("monthCode not valid for the ISO calendar"));
                }
                Some(c)
            }
            None => None,
        };
        match (month, coded) {
            (Some(m), Some(c)) if m != c => Err(self.pdt_range("month and monthCode disagree")),
            (Some(m), _) => Ok(m),
            (None, Some(c)) => Ok(c),
            (None, None) => fallback.ok_or_else(|| self.type_error("month or monthCode required")),
        }
    }

    // --- ToTemporalDateTime -----------------------------------------------

    /// `ToTemporalDateTime(item, options)` → the ISO date + time. Accepts a
    /// `PlainDateTime` (copy), a `PlainDate` (midnight), a property bag, or an ISO
    /// string.
    fn pdt_to_datetime(
        &mut self,
        item: NanBox,
        options: NanBox,
    ) -> Result<(IsoDate, IsoTime), ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(d) = self.realm.temporal_at(h) {
                let opts = self.pdt_options(options)?;
                self.pdt_overflow(opts)?; // validated even though ignored on copy
                return match d.kind {
                    TemporalKind::PlainDateTime => Ok((d.date, d.time)),
                    TemporalKind::PlainDate => Ok((d.date, IsoTime::default())),
                    // A ZonedDateTime yields its wall-clock date+time in its zone.
                    TemporalKind::ZonedDateTime => {
                        let tz = d.tz.as_deref().unwrap_or("UTC");
                        Ok(crate::nbexec::temporal_zoneddatetime::local_of(
                            tz, d.epoch_ns,
                        ))
                    }
                    _ => Err(self.type_error("expected a PlainDateTime")),
                };
            }
            if let Some(s) = self.realm.string_value(h) {
                // Parse (and range-check) the string *before* touching `options`, so
                // an invalid string throws without observing the options bag.
                let (date, time) = pdt_parse_datetime(&s)
                    .ok_or_else(|| self.pdt_range("invalid PlainDateTime string"))?;
                if !pdt_in_range(date, time) {
                    return Err(self.pdt_range("PlainDateTime outside representable range"));
                }
                let opts = self.pdt_options(options)?;
                self.pdt_overflow(opts)?;
                return Ok((date, time));
            }
            if self.is_object_value(item) {
                let opts = self.pdt_options(options)?;
                self.pdt_bag_calendar(h)?;
                let bag = self.pdt_read_bag(h)?;
                let overflow = self.pdt_overflow(opts)?;
                let year = bag
                    .year
                    .ok_or_else(|| self.type_error("year is required"))?;
                let day = bag.day.ok_or_else(|| self.type_error("day is required"))?;
                let month = self.pdt_resolve_month(bag.month, bag.month_code, None)?;
                if month < 1 || day < 1 {
                    return Err(self.pdt_range("month and day must be positive"));
                }
                let date = self.pdt_regulate_date(year, month, day, overflow)?;
                let time = iso::regulate_iso_time(
                    bag.hour.unwrap_or(0),
                    bag.minute.unwrap_or(0),
                    bag.second.unwrap_or(0),
                    bag.ms.unwrap_or(0),
                    bag.us.unwrap_or(0),
                    bag.ns.unwrap_or(0),
                    overflow,
                )
                .ok_or_else(|| self.pdt_range("invalid ISO time"))?;
                if !pdt_in_range(date, time) {
                    return Err(self.pdt_range("PlainDateTime outside representable range"));
                }
                return Ok((date, time));
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.PlainDateTime"))
    }

    // --- with / withPlainTime / withCalendar ------------------------------

    fn pdt_with(
        &mut self,
        data: &TemporalData,
        fields: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        // The argument must be a plain object (not a Temporal instance, which lacks
        // the string field keys).
        let is_temporal = fields
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.temporal_at(h).is_some());
        if !self.is_object_value(fields) || is_temporal {
            return Err(self.type_error("with() requires a plain fields object"));
        }
        let h = fields.as_handle().map(Handle::from_raw).unwrap();
        // RejectObjectWithCalendarOrTimeZone: a `calendar`/`timeZone` present → TypeError.
        if self.pdt_field(h, "calendar")?.is_some() {
            return Err(self.type_error("with() fields must not have a calendar property"));
        }
        if self.pdt_field(h, "timeZone")?.is_some() {
            return Err(self.type_error("with() fields must not have a timeZone property"));
        }
        let bag = self.pdt_read_bag(h)?;
        let opts = self.pdt_options(options)?;
        let overflow = self.pdt_overflow(opts)?;
        if !bag.any() {
            return Err(self.type_error("with() requires at least one recognised field"));
        }
        let cur = data.date;
        let ct = data.time;
        let year = bag.year.unwrap_or(i64::from(cur.year));
        let day = bag.day.unwrap_or(i64::from(cur.day));
        let month =
            self.pdt_resolve_month(bag.month, bag.month_code, Some(i64::from(cur.month)))?;
        if month < 1 || day < 1 {
            return Err(self.pdt_range("month and day must be positive"));
        }
        let date = self.pdt_regulate_date(year, month, day, overflow)?;
        let time = iso::regulate_iso_time(
            bag.hour.unwrap_or(i64::from(ct.hour)),
            bag.minute.unwrap_or(i64::from(ct.minute)),
            bag.second.unwrap_or(i64::from(ct.second)),
            bag.ms.unwrap_or(i64::from(ct.millisecond)),
            bag.us.unwrap_or(i64::from(ct.microsecond)),
            bag.ns.unwrap_or(i64::from(ct.nanosecond)),
            overflow,
        )
        .ok_or_else(|| self.pdt_range("invalid ISO time"))?;
        if !pdt_in_range(date, time) {
            return Err(self.pdt_range("PlainDateTime outside representable range"));
        }
        Ok(self.pdt_make(date, time))
    }

    fn pdt_with_plain_time(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
    ) -> Result<NanBox, ExecError> {
        let time = if arg.is_undefined() {
            IsoTime::default()
        } else {
            self.pdt_to_time(arg)?
        };
        Ok(self.pdt_make(data.date, time))
    }

    fn pdt_with_calendar(&mut self, data: &TemporalData, arg: NanBox) -> Result<NanBox, ExecError> {
        self.pdt_calendar_arg(arg)?;
        Ok(self.pdt_make(data.date, data.time))
    }

    /// `ToTemporalTime(item)` → an ISO time. Accepts a `PlainTime`/`PlainDateTime`
    /// (its time), an ISO string, or a property bag of time fields.
    fn pdt_to_time(&mut self, item: NanBox) -> Result<IsoTime, ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(d) = self.realm.temporal_at(h) {
                return match d.kind {
                    TemporalKind::PlainTime | TemporalKind::PlainDateTime => Ok(d.time),
                    _ => Err(self.type_error("expected a PlainTime")),
                };
            }
            if let Some(s) = self.realm.string_value(h) {
                let p = iso::parse_iso_time_string(&s)
                    .ok_or_else(|| self.pdt_range("invalid PlainTime string"))?;
                return p
                    .time
                    .ok_or_else(|| self.pdt_range("string is missing a time"));
            }
            if self.is_object_value(item) {
                let bag = self.pdt_read_bag(h)?;
                if !bag.any_time() {
                    return Err(self.type_error("no time fields present"));
                }
                return iso::regulate_iso_time(
                    bag.hour.unwrap_or(0),
                    bag.minute.unwrap_or(0),
                    bag.second.unwrap_or(0),
                    bag.ms.unwrap_or(0),
                    bag.us.unwrap_or(0),
                    bag.ns.unwrap_or(0),
                    Overflow::Constrain,
                )
                .ok_or_else(|| self.pdt_range("invalid ISO time"));
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.PlainTime"))
    }

    // --- add / subtract ----------------------------------------------------

    fn pdt_add(
        &mut self,
        data: &TemporalData,
        dur_arg: NanBox,
        options: NanBox,
        sign: i64,
    ) -> Result<NanBox, ExecError> {
        let mut dur = self.pdt_to_duration(dur_arg)?;
        if sign < 0 {
            dur = negate_duration(dur);
        }
        let opts = self.pdt_options(options)?;
        let overflow = self.pdt_overflow(opts)?;
        // AddDateTime: add the time part (yielding a day carry), then the date part.
        let (day_carry, new_time) = iso::add_time(data.time, dur.time_nanos());
        let new_date = iso::add_iso_date(
            data.date,
            dur.years,
            dur.months,
            dur.weeks,
            dur.days + day_carry,
            overflow,
        )
        .ok_or_else(|| self.pdt_range("result outside representable range"))?;
        if !pdt_in_range(new_date, new_time) {
            return Err(self.pdt_range("result outside representable range"));
        }
        Ok(self.pdt_make(new_date, new_time))
    }

    /// `ToTemporalDuration(item)`: a `Temporal.Duration`, an ISO duration string,
    /// or a property bag of duration fields.
    fn pdt_to_duration(&mut self, item: NanBox) -> Result<DurationFields, ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(d) = self.realm.temporal_at(h) {
                return if d.kind == TemporalKind::Duration {
                    Ok(d.duration)
                } else {
                    Err(self.type_error("expected a Temporal.Duration"))
                };
            }
            if let Some(s) = self.realm.string_value(h) {
                return iso::parse_iso_duration(&s)
                    .ok_or_else(|| self.pdt_range("invalid duration string"));
            }
            if self.is_object_value(item) {
                return self.pdt_read_duration_bag(h);
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.Duration"))
    }

    fn pdt_read_duration_bag(&mut self, h: Handle) -> Result<DurationFields, ExecError> {
        let keys = [
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
        let mut d = DurationFields::default();
        let mut any = false;
        for key in keys {
            if let Some(v) = self.pdt_field(h, key)? {
                let num = self.coerce_to_number(v)?;
                let n = self.realm.to_number(num);
                if !n.is_finite() || n.fract() != 0.0 {
                    return Err(self.pdt_range("duration fields must be integers"));
                }
                let val = n as i64;
                any = true;
                match key {
                    "years" => d.years = val,
                    "months" => d.months = val,
                    "weeks" => d.weeks = val,
                    "days" => d.days = val,
                    "hours" => d.hours = val,
                    "minutes" => d.minutes = val,
                    "seconds" => d.seconds = val,
                    "milliseconds" => d.milliseconds = val,
                    "microseconds" => d.microseconds = val,
                    _ => d.nanoseconds = val,
                }
            }
        }
        if !any {
            return Err(self.type_error("no recognised duration fields present"));
        }
        if !d.is_valid() {
            return Err(self.pdt_range("duration fields must share one sign"));
        }
        Ok(d)
    }

    // --- until / since -----------------------------------------------------

    fn pdt_diff(
        &mut self,
        data: &TemporalData,
        other: NanBox,
        options: NanBox,
        negate: bool,
    ) -> Result<NanBox, ExecError> {
        let (d2, t2) = self.pdt_to_datetime(other, NanBox::undefined())?;
        let opts = self.pdt_options(options)?;
        let units = [
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
        let smallest = match self.pdt_str_option(opts, "smallestUnit", &units)? {
            Some(s) => parse_unit(&s).unwrap_or(Unit::Nanosecond),
            None => Unit::Nanosecond,
        };
        let largest = match self.pdt_str_option(opts, "largestUnit", &units)? {
            Some(s) if s != "auto" => parse_unit(&s).unwrap_or(Unit::Day),
            _ => Unit::Day.min(smallest),
        };
        if largest > smallest {
            return Err(self.pdt_range("largestUnit must be at least as large as smallestUnit"));
        }
        let increment = self.pdt_rounding_increment(opts)?;
        let mode = self.pdt_rounding_mode(opts, RoundMode::Trunc)?;

        let from = (data.date, data.time);
        let to = (d2, t2);
        // The time-only path (no year/month component) supports rounding; the
        // calendar path is emitted unrounded.
        let mut dur = if largest >= Unit::Day {
            pdt_round_duration(from, to, largest, smallest, increment, mode)
        } else {
            pdt_difference(from, to, largest)
        };
        if negate {
            dur = negate_duration(dur);
        }
        Ok(self.pdt_make_duration(dur))
    }

    // --- round -------------------------------------------------------------

    fn pdt_round(&mut self, data: &TemporalData, options: NanBox) -> Result<NanBox, ExecError> {
        // `roundTo` is a required parameter: `undefined` (or a missing argument) →
        // TypeError before any option is examined.
        if options.is_undefined() {
            return Err(self.type_error("round() requires a roundTo argument"));
        }
        // `round` accepts a bare string smallestUnit or an options bag.
        let string_form = options
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.string_value(h).is_some());
        let units = [
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
        let (smallest, increment, mode) = if string_form {
            let s = self.coerce_to_string(options)?;
            let u = parse_unit(&s)
                .filter(|u| *u >= Unit::Day)
                .ok_or_else(|| self.pdt_range("invalid smallestUnit"))?;
            (u, 1, RoundMode::HalfExpand)
        } else {
            let opts = self.pdt_options(options)?;
            // Options are read increment → mode → smallestUnit (observable order).
            let increment = self.pdt_rounding_increment(opts)?;
            let mode = self.pdt_rounding_mode(opts, RoundMode::HalfExpand)?;
            let u = match self.pdt_str_option(opts, "smallestUnit", &units)? {
                Some(s) => parse_unit(&s).unwrap_or(Unit::Nanosecond),
                None => return Err(self.pdt_range("round() requires a smallestUnit")),
            };
            (u, increment, mode)
        };
        self.pdt_validate_increment(smallest, increment)?;

        let (date, time) = pdt_round_datetime(data.date, data.time, smallest, increment, mode);
        if !pdt_in_range(date, time) {
            return Err(self.pdt_range("rounded PlainDateTime outside representable range"));
        }
        Ok(self.pdt_make(date, time))
    }

    /// `ValidateTemporalRoundingIncrement` for a PlainDateTime round unit
    /// (non-inclusive: the increment must be strictly less than the unit's
    /// dividend and divide it evenly). `day` permits only an increment of 1.
    fn pdt_validate_increment(&mut self, unit: Unit, increment: i64) -> Result<(), ExecError> {
        let dividend: i64 = match unit {
            Unit::Day => return self.pdt_require_day_increment(increment),
            Unit::Hour => 24,
            Unit::Minute | Unit::Second => 60,
            _ => 1000,
        };
        if increment >= dividend || dividend % increment != 0 {
            return Err(self.pdt_range("invalid roundingIncrement for the smallestUnit"));
        }
        Ok(())
    }

    fn pdt_require_day_increment(&mut self, increment: i64) -> Result<(), ExecError> {
        if increment == 1 {
            Ok(())
        } else {
            Err(self.pdt_range("roundingIncrement must be 1 when smallestUnit is day"))
        }
    }

    // --- toString ----------------------------------------------------------

    fn pdt_to_string(&mut self, data: &TemporalData, options: NanBox) -> Result<NanBox, ExecError> {
        let opts = self.pdt_options(options)?;
        let cal = self
            .pdt_str_option(
                opts,
                "calendarName",
                &["auto", "always", "never", "critical"],
            )?
            .unwrap_or_else(|| alloc::string::String::from("auto"));
        let time_units = [
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
        let smallest = self
            .pdt_str_option(opts, "smallestUnit", &time_units)?
            .map(|s| parse_unit(&s).unwrap_or(Unit::Nanosecond));
        let frac = self.pdt_frac_digits(opts)?;
        let mode = self.pdt_rounding_mode(opts, RoundMode::Trunc)?;

        // Rounding increment (ns) + whether seconds show + fixed precision.
        let (inc_ns, seconds_shown, precision): (i128, bool, Option<u8>) = match smallest {
            Some(Unit::Minute) => (iso::NS_PER_MINUTE, false, None),
            Some(Unit::Second) => (iso::NS_PER_SEC, true, Some(0)),
            Some(Unit::Millisecond) => (1_000_000, true, Some(3)),
            Some(Unit::Microsecond) => (1_000, true, Some(6)),
            Some(Unit::Nanosecond) => (1, true, Some(9)),
            _ => match frac {
                None => (1, true, None),
                Some(n) => (10_i128.pow(u32::from(9 - n)), true, Some(n)),
            },
        };

        let total = iso::time_to_nanos(data.time);
        let rounded = iso::round_to_increment(total, inc_ns, mode);
        let (day_carry, time) = iso::balance_time_from_nanos(rounded);
        let date = iso::epoch_days_to_iso(iso::iso_to_epoch_days(data.date) + day_carry);
        if !pdt_in_range(date, time) {
            return Err(self.pdt_range("PlainDateTime outside representable range"));
        }

        let mut out = alloc::format!(
            "{}-{}-{}T{}:{}",
            iso::format_iso_year(date.year),
            iso::pad(u64::from(date.month), 2),
            iso::pad(u64::from(date.day), 2),
            iso::pad(u64::from(time.hour), 2),
            iso::pad(u64::from(time.minute), 2),
        );
        if seconds_shown {
            out.push(':');
            out.push_str(&iso::pad(u64::from(time.second), 2));
            let sub = u32::from(time.millisecond) * 1_000_000
                + u32::from(time.microsecond) * 1_000
                + u32::from(time.nanosecond);
            out.push_str(&iso::format_fraction(sub, precision));
        }
        match cal.as_str() {
            "always" => out.push_str("[u-ca=iso8601]"),
            "critical" => out.push_str("[!u-ca=iso8601]"),
            _ => {}
        }
        Ok(self.new_str(&out))
    }

    /// `GetTemporalFractionalSecondDigitsOption`: `undefined`/`"auto"` → `None`
    /// (auto), a Number in `[0, 9]` → that many digits (floored). Any other type
    /// must stringify to `"auto"`; out-of-range or `NaN` → RangeError.
    fn pdt_frac_digits(&mut self, opts: Option<Handle>) -> Result<Option<u8>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, "fractionalSecondDigits")?;
        if v.is_undefined() {
            return Ok(None);
        }
        if v.is_number() {
            let n = v.as_number().unwrap_or(f64::NAN);
            if n.is_nan() || !(0.0..=9.0).contains(&n) {
                return Err(self.pdt_range("fractionalSecondDigits out of range"));
            }
            return Ok(Some(n.floor() as u8));
        }
        let s = self.coerce_to_string(v)?;
        if s == "auto" {
            Ok(None)
        } else {
            Err(self.pdt_range("invalid fractionalSecondDigits"))
        }
    }

    // --- result builders ---------------------------------------------------

    fn pdt_make(&mut self, date: IsoDate, time: IsoTime) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainDateTime,
            date,
            time,
            ..Default::default()
        };
        self.pdt_alloc(data, TemporalKind::PlainDateTime)
    }

    fn pdt_make_date(&mut self, date: IsoDate) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainDate,
            date,
            ..Default::default()
        };
        self.pdt_alloc(data, TemporalKind::PlainDate)
    }

    fn pdt_make_time(&mut self, time: IsoTime) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainTime,
            time,
            ..Default::default()
        };
        self.pdt_alloc(data, TemporalKind::PlainTime)
    }

    fn pdt_make_duration(&mut self, dur: DurationFields) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::Duration,
            duration: dur,
            ..Default::default()
        };
        self.pdt_alloc(data, TemporalKind::Duration)
    }

    fn pdt_alloc(&mut self, data: TemporalData, kind: TemporalKind) -> NanBox {
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(kind) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }
}

/// Whether a date+time is inside the representable `PlainDateTime` range
/// (`ISODateTimeWithinLimits`: within one day of the Instant epoch-ns limits,
/// boundary-exclusive).
fn pdt_in_range(date: IsoDate, time: IsoTime) -> bool {
    let ns = iso::iso_to_epoch_days(date) as i128 * iso::NS_PER_DAY + iso::time_to_nanos(time);
    ns > iso::MIN_EPOCH_NS - iso::NS_PER_DAY && ns < iso::MAX_EPOCH_NS + iso::NS_PER_DAY
}

/// A byte cursor for the strict ISO-8601 date-time parser.
struct PdtCursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl PdtCursor<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn peek_digit(&self) -> bool {
        self.peek().is_some_and(|c| c.is_ascii_digit())
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn digits(&mut self, n: usize) -> Option<i64> {
        let mut v = 0_i64;
        for _ in 0..n {
            let c = self.peek()?;
            if !c.is_ascii_digit() {
                return None;
            }
            v = v * 10 + i64::from(c - b'0');
            self.i += 1;
        }
        Some(v)
    }
    /// Reads an optional `.`/`,` fractional-seconds group into nanoseconds. `None`
    /// on a malformed group (no digits, or more than nine).
    fn fraction(&mut self) -> Option<i64> {
        if !(self.eat(b'.') || self.eat(b',')) {
            return Some(0);
        }
        let mut ns = 0_i64;
        let mut count = 0;
        while self.peek_digit() {
            if count == 9 {
                return None; // more than nine fractional digits
            }
            ns = ns * 10 + i64::from(self.peek().unwrap() - b'0');
            self.i += 1;
            count += 1;
        }
        if count == 0 {
            return None;
        }
        for _ in count..9 {
            ns *= 10;
        }
        Some(ns)
    }
}

/// The `PlainDateTime` ISO-string grammar (strict): a required date, an optional
/// `T`/space-introduced time with an optional numeric UTC offset (a bare `Z`
/// designator is rejected), and optional `[…]` annotations. Returns the ISO
/// date+time, or `None` for any malformed / unsupported form.
fn pdt_parse_datetime(s: &str) -> Option<(IsoDate, IsoTime)> {
    // The Unicode MINUS SIGN (U+2212) is never accepted in a Temporal string.
    if s.as_bytes().windows(3).any(|w| w == [0xE2, 0x88, 0x92]) {
        return None;
    }
    let mut c = PdtCursor {
        b: s.as_bytes(),
        i: 0,
    };
    let date = pdt_parse_date(&mut c)?;
    let mut time = IsoTime::default();
    if c.eat(b'T') || c.eat(b't') || c.eat(b' ') {
        let (h, m, sec, frac) = pdt_parse_time(&mut c)?;
        if h > 23 || m > 59 || sec > 60 {
            return None;
        }
        let sec = if sec == 60 { 59 } else { sec }; // leap second → constrain
        time = IsoTime {
            hour: h as u8,
            minute: m as u8,
            second: sec as u8,
            millisecond: (frac / 1_000_000) as u16,
            microsecond: ((frac / 1_000) % 1_000) as u16,
            nanosecond: (frac % 1_000) as u16,
        };
        if pdt_parse_offset(&mut c)? {
            return None; // bare `Z` designator is invalid for a PlainDateTime
        }
    }
    pdt_parse_annotations(&mut c)?;
    (c.i == c.b.len()).then_some((date, time))
}

/// Parses the date portion, enforcing consistent separator usage (all-dashes or
/// none) so mixed forms like `202501-01` are rejected.
fn pdt_parse_date(c: &mut PdtCursor) -> Option<IsoDate> {
    let year = if c.eat(b'+') {
        c.digits(6)?
    } else if c.eat(b'-') {
        let y = c.digits(6)?;
        if y == 0 {
            return None; // -000000 is not a valid extended year
        }
        -y
    } else {
        c.digits(4)?
    };
    let dash = c.eat(b'-');
    let month = c.digits(2)?;
    if dash {
        if !c.eat(b'-') {
            return None; // inconsistent separators
        }
    } else if c.peek() == Some(b'-') {
        return None; // inconsistent separators
    }
    let day = c.digits(2)?;
    iso::regulate_iso_date(year as i32, month, day, Overflow::Reject)
}

/// Parses the wall-clock time portion (`HH[:MM[:SS[.fff]]]`), returning
/// `(hour, minute, second, sub-second-ns)`.
fn pdt_parse_time(c: &mut PdtCursor) -> Option<(i64, i64, i64, i64)> {
    let hour = c.digits(2)?;
    let mut minute = 0;
    let mut second = 0;
    let mut frac = 0;
    if c.eat(b':') {
        minute = c.digits(2)?;
        if c.eat(b':') {
            second = c.digits(2)?;
            frac = c.fraction()?;
        }
    } else if c.peek_digit() {
        minute = c.digits(2)?;
        if c.peek_digit() {
            second = c.digits(2)?;
            frac = c.fraction()?;
        }
    }
    Some((hour, minute, second, frac))
}

/// Parses an optional trailing offset, returning whether it was a bare `Z`/`z`.
/// A numeric offset is consumed and ignored; `None` on a malformed offset.
fn pdt_parse_offset(c: &mut PdtCursor) -> Option<bool> {
    if c.eat(b'Z') || c.eat(b'z') {
        return Some(true);
    }
    if c.peek() == Some(b'+') || c.peek() == Some(b'-') {
        c.i += 1;
        c.digits(2)?;
        if c.eat(b':') {
            c.digits(2)?;
            if c.eat(b':') {
                c.digits(2)?;
                c.fraction()?;
            }
        } else if c.peek_digit() {
            c.digits(2)?;
            if c.peek_digit() {
                c.digits(2)?;
                c.fraction()?;
            }
        }
    }
    Some(false)
}

/// Parses the `[…]` annotation suffixes, enforcing the ISO annotation rules
/// (lowercase keys; a single time-zone annotation; an `iso8601` calendar; no
/// conflicting-critical or unknown-critical annotations). `None` on violation.
fn pdt_parse_annotations(c: &mut PdtCursor) -> Option<()> {
    let mut cal_count = 0;
    let mut cal_critical = false;
    let mut first_cal_iso = true;
    let mut tz_count = 0;
    while c.eat(b'[') {
        let critical = c.eat(b'!');
        let start = c.i;
        while c.peek().is_some_and(|b| b != b']') {
            c.i += 1;
        }
        let inner = &c.b[start..c.i];
        if !c.eat(b']') {
            return None;
        }
        if let Some(eq) = inner.iter().position(|&b| b == b'=') {
            let key = &inner[..eq];
            let val = &inner[eq + 1..];
            if key.iter().any(u8::is_ascii_uppercase) {
                return None; // annotation keys must be lowercase
            }
            if key == b"u-ca" {
                cal_count += 1;
                cal_critical |= critical;
                if cal_count == 1 {
                    first_cal_iso = val.eq_ignore_ascii_case(b"iso8601");
                }
            } else if critical {
                return None; // unknown annotation with the critical flag
            }
        } else {
            tz_count += 1; // a time-zone annotation
        }
    }
    (!(cal_count > 1 && cal_critical) && tz_count <= 1 && first_cal_iso).then_some(())
}

/// Whether a property-bag `calendar` **string** names the ISO calendar: either
/// the bare identifier `"iso8601"` or a date-ish ISO string (leading digit/sign)
/// whose every `u-ca=` annotation value is `iso8601`.
fn pdt_calendar_string_ok(s: &str) -> bool {
    if s.eq_ignore_ascii_case("iso8601") {
        return true;
    }
    if !s
        .as_bytes()
        .first()
        .is_some_and(|&c| c.is_ascii_digit() || c == b'+' || c == b'-')
    {
        return false;
    }
    // Minus-zero is not a valid extended year (`-000000`).
    if s.starts_with("-000000") {
        return false;
    }
    let mut rest = s;
    while let Some(p) = rest.find("u-ca=") {
        let after = &rest[p + 5..];
        let end = after.find(']').unwrap_or(after.len());
        if !after[..end].eq_ignore_ascii_case("iso8601") {
            return false;
        }
        rest = &after[end..];
    }
    true
}

/// Negates every field of a duration.
fn negate_duration(d: DurationFields) -> DurationFields {
    DurationFields {
        years: -d.years,
        months: -d.months,
        weeks: -d.weeks,
        days: -d.days,
        hours: -d.hours,
        minutes: -d.minutes,
        seconds: -d.seconds,
        milliseconds: -d.milliseconds,
        microseconds: -d.microseconds,
        nanoseconds: -d.nanoseconds,
    }
}

/// `DifferenceISODateTime(from, to, largestUnit)` (no rounding).
fn pdt_difference(
    from: (IsoDate, IsoTime),
    to: (IsoDate, IsoTime),
    largest: Unit,
) -> DurationFields {
    if largest >= Unit::Day {
        // No calendar component: work entirely in nanoseconds.
        let total = (iso::iso_to_epoch_days(to.0) - iso::iso_to_epoch_days(from.0)) as i128
            * iso::NS_PER_DAY
            + (iso::time_to_nanos(to.1) - iso::time_to_nanos(from.1));
        return pdt_balance_datetime(total, largest);
    }
    // Year/Month/Week largest: split the time part off, borrowing a day when it
    // points opposite the date direction.
    let mut time_ns = iso::time_to_nanos(to.1) - iso::time_to_nanos(from.1);
    let time_sign = time_ns.signum();
    let date_sign = match iso::compare_iso_date(to.0, from.0) {
        core::cmp::Ordering::Greater => 1_i128,
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
    };
    let mut adjusted_to = to.0;
    if time_sign != 0 && time_sign == -date_sign {
        adjusted_to = iso::epoch_days_to_iso(iso::iso_to_epoch_days(to.0) + time_sign as i64);
        time_ns -= time_sign * iso::NS_PER_DAY;
    }
    let (y, mo, w, d) = iso::difference_iso_date(from.0, adjusted_to, largest);
    let mut dur = iso::balance_time_duration(time_ns, Unit::Hour);
    dur.years = y;
    dur.months = mo;
    dur.weeks = w;
    dur.days = d;
    dur
}

/// Balances a signed nanosecond total into a duration down to `largest`, where
/// `largest` is Day or a finer unit.
fn pdt_balance_datetime(total_ns: i128, largest: Unit) -> DurationFields {
    let sign = total_ns.signum() as i64;
    let mut r = total_ns.abs();
    let (mut weeks, mut days) = (0_i64, 0_i64);
    if largest <= Unit::Day {
        days = (r / iso::NS_PER_DAY) as i64;
        r %= iso::NS_PER_DAY;
        if largest == Unit::Week {
            weeks = days / 7;
            days %= 7;
        }
    }
    let mut dur = iso::balance_time_duration(r * i128::from(sign), largest);
    dur.days = days * sign;
    dur.weeks = weeks * sign;
    dur
}

/// Rounds the whole date+time difference to `smallest` (Day..Nanosecond) — the
/// time-only path used when there is no year/month component.
fn pdt_round_duration(
    from: (IsoDate, IsoTime),
    to: (IsoDate, IsoTime),
    largest: Unit,
    smallest: Unit,
    increment: i64,
    mode: RoundMode,
) -> DurationFields {
    let total = (iso::iso_to_epoch_days(to.0) - iso::iso_to_epoch_days(from.0)) as i128
        * iso::NS_PER_DAY
        + (iso::time_to_nanos(to.1) - iso::time_to_nanos(from.1));
    let inc = unit_ns(smallest) * i128::from(increment.max(1));
    let rounded = iso::round_to_increment(total, inc, mode);
    pdt_balance_datetime(rounded, largest)
}

/// `RoundISODateTime`: rounds the date+time to `smallest` with `increment`.
fn pdt_round_datetime(
    date: IsoDate,
    time: IsoTime,
    smallest: Unit,
    increment: i64,
    mode: RoundMode,
) -> (IsoDate, IsoTime) {
    let inc = unit_ns(smallest) * i128::from(increment.max(1));
    let rounded = iso::round_to_increment(iso::time_to_nanos(time), inc, mode);
    let (day_carry, new_time) = iso::balance_time_from_nanos(rounded);
    let new_date = iso::epoch_days_to_iso(iso::iso_to_epoch_days(date) + day_carry);
    (new_date, new_time)
}
