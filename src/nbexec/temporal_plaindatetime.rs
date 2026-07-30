//! `Temporal.PlainDateTime` — logic module. A fan-out unit: everything specific to
//! `PlainDateTime` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
//!
//! A `PlainDateTime` is a calendar date plus a wall-clock time (no time zone).
//! It stores both `TemporalData.date` (`IsoDate`) and `TemporalData.time`
//! (`IsoTime`) plus a calendar id in `TemporalData.calendar` (default
//! `"iso8601"`); it is essentially `PlainDate + PlainTime` combined, reusing the
//! shared `crate::temporal_iso` helpers. The calendar-dependent date fields route
//! through [`super::temporal_calendar`] for a non-ISO calendar, keeping the ISO
//! fast path byte-for-byte unchanged; the wall-clock time is calendar-independent.
use super::temporal_calendar as tcal;
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
}

/// The recognised calendar/time fields of a **non-ISO** date-time property bag —
/// like [`DtBag`] but carrying `era`/`eraYear` and the raw `monthCode` string (the
/// calendar layer judges its suitability), for `CalendarDateFromFields`.
#[derive(Default)]
struct CalDtBag {
    era: Option<String>,
    era_year: Option<i64>,
    year: Option<i64>,
    month: Option<i64>,
    month_code: Option<String>,
    day: Option<i64>,
    hour: Option<i64>,
    minute: Option<i64>,
    second: Option<i64>,
    ms: Option<i64>,
    us: Option<i64>,
    ns: Option<i64>,
}

impl CalDtBag {
    fn any(&self) -> bool {
        self.era.is_some()
            || self.era_year.is_some()
            || self.year.is_some()
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
        // calendar (10th arg): undefined → iso8601; a String → canonicalized (the
        // full Temporal calendar-id set); a non-string → TypeError.
        let calendar = self.pdt_validate_calendar(arg(9))?;

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
            calendar,
            ..Default::default()
        };
        self.finish_temporal(data, new_target, callee)
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
                let (d2, t2, cal2) = self.pdt_to_datetime(arg(0), NanBox::undefined())?;
                let eq = data.date == d2 && data.time == t2 && data.calendar == cal2;
                Ok(NanBox::boolean(eq))
            }
            "toPlainDate" => Ok(self.pdt_make_date(data.date, &data.calendar)),
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
                // `GetOptionsObject` (TypeError on a primitive), then read/validate
                // the `disambiguation` option (RangeError on a bad value).
                let opts = self.pdt_options(arg(1))?;
                let disamb = self
                    .pdt_str_option(
                        opts,
                        "disambiguation",
                        &["compatible", "earlier", "later", "reject"],
                    )?
                    .unwrap_or_else(|| alloc::string::String::from("compatible"));
                let local_ns = crate::temporal_iso::iso_to_epoch_days(data.date) as i128
                    * crate::temporal_iso::NS_PER_DAY
                    + crate::temporal_iso::time_to_nanos(data.time);
                // `GetEpochNanosecondsFor` with proper DST disambiguation — a naïve
                // offset-at-the-wall-instant conversion is wrong inside gaps/overlaps.
                let epoch = crate::nbexec::temporal_zoneddatetime::epoch_for_wall_disamb(
                    &tz, local_ns, &disamb,
                )
                .map_err(|()| {
                    self.pdt_range(
                        "wall-clock time is ambiguous or nonexistent (disambiguation: reject)",
                    )
                })?;
                // The resulting exact time must be within the Instant range.
                if !(crate::temporal_iso::MIN_EPOCH_NS..=crate::temporal_iso::MAX_EPOCH_NS)
                    .contains(&epoch)
                {
                    return Err(self.pdt_range("instant is outside the representable range"));
                }
                Ok(self.build_temporal(crate::temporal_iso::TemporalData {
                    kind: crate::temporal_iso::TemporalKind::ZonedDateTime,
                    epoch_ns: epoch,
                    tz: Some(tz),
                    calendar: data.calendar.clone(),
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
        let cal = data.calendar.as_str();
        let num = |n: i64| NanBox::number(n as f64);
        // Calendar-independent getters (the wall-clock time + the calendar id).
        match name {
            "calendarId" => return Ok(self.new_str(cal)),
            "hour" => return Ok(num(i64::from(t.hour))),
            "minute" => return Ok(num(i64::from(t.minute))),
            "second" => return Ok(num(i64::from(t.second))),
            "millisecond" => return Ok(num(i64::from(t.millisecond))),
            "microsecond" => return Ok(num(i64::from(t.microsecond))),
            "nanosecond" => return Ok(num(i64::from(t.nanosecond))),
            _ => {}
        }
        // ISO-8601 fast path — byte-for-byte the original computation.
        if tcal::is_iso(cal) {
            return Ok(match name {
                "era" | "eraYear" => NanBox::undefined(),
                "year" => num(i64::from(d.year)),
                "month" => num(i64::from(d.month)),
                "monthCode" => {
                    let s = alloc::format!("M{}", iso::pad(u64::from(d.month), 2));
                    self.new_str(&s)
                }
                "day" => num(i64::from(d.day)),
                "dayOfWeek" => num(i64::from(iso::iso_day_of_week(d))),
                "dayOfYear" => num(i64::from(iso::iso_day_of_year(d))),
                "weekOfYear" => num(i64::from(iso::iso_week_of_year(d).0)),
                "yearOfWeek" => num(i64::from(iso::iso_week_of_year(d).1)),
                "daysInWeek" => num(7),
                "daysInMonth" => num(i64::from(iso::iso_days_in_month(d.year, d.month))),
                "daysInYear" => num(i64::from(iso::iso_days_in_year(d.year))),
                "monthsInYear" => num(12),
                "inLeapYear" => NanBox::boolean(iso::is_leap_year(d.year)),
                _ => {
                    return Err(self.temporal_todo(&alloc::format!("PlainDateTime getter {name}")));
                }
            });
        }
        // Non-ISO calendar: route through the calendar abstraction layer.
        let f = tcal::iso_to_fields(cal, d);
        Ok(match name {
            "era" => match &f.era {
                Some(e) => self.new_str(e),
                None => NanBox::undefined(),
            },
            "eraYear" => match f.era_year {
                Some(y) => NanBox::number(y as f64),
                None => NanBox::undefined(),
            },
            "year" => NanBox::number(f.year as f64),
            "month" => NanBox::number(f.month as f64),
            "monthCode" => self.new_str(&f.month_code),
            "day" => NanBox::number(f.day as f64),
            "dayOfWeek" => NanBox::number(tcal::day_of_week(d) as f64),
            "dayOfYear" => NanBox::number(tcal::day_of_year(cal, d) as f64),
            "weekOfYear" => match tcal::week_of_year(cal, d) {
                Some((w, _)) => NanBox::number(w as f64),
                None => NanBox::undefined(),
            },
            "yearOfWeek" => match tcal::year_of_week(cal, d) {
                Some(y) => NanBox::number(y as f64),
                None => NanBox::undefined(),
            },
            "daysInWeek" => NanBox::number(tcal::days_in_week() as f64),
            "daysInMonth" => NanBox::number(tcal::days_in_month(cal, d) as f64),
            "daysInYear" => NanBox::number(tcal::days_in_year(cal, d) as f64),
            "monthsInYear" => NanBox::number(tcal::months_in_year(cal, d) as f64),
            "inLeapYear" => NanBox::boolean(tcal::in_leap_year(cal, d)),
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
                let (date, time, cal) = self.pdt_to_datetime(arg(0), arg(1))?;
                Ok(Some(self.pdt_make_cal(date, time, &cal)))
            }
            "compare" => {
                let (d1, t1, _) = self.pdt_to_datetime(arg(0), NanBox::undefined())?;
                let (d2, t2, _) = self.pdt_to_datetime(arg(1), NanBox::undefined())?;
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

    /// Validates a constructor/`withCalendar` calendar argument and returns its
    /// canonical id: `undefined` → `"iso8601"`; a calendared Temporal object → its
    /// id; a String → canonicalized against the full Temporal calendar-id set
    /// (unknown → RangeError); a non-string → TypeError.
    fn pdt_validate_calendar(&mut self, v: NanBox) -> Result<String, ExecError> {
        if v.is_undefined() {
            return Ok(String::from("iso8601"));
        }
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.pdt_canonicalize_calendar(&cal);
        }
        let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        // `ToTemporalCalendarIdentifier` for a calendar *argument* accepts only a
        // bare calendar id — a full ISO string (even one carrying `[u-ca=…]`) is
        // not a valid calendar identifier here (see calendar-invalid-iso-string).
        if let Some(c) = tcal::canonicalize_calendar(&s) {
            return Ok(String::from(c));
        }
        Err(self.pdt_range(&alloc::format!("invalid calendar identifier '{s}'")))
    }

    /// Canonicalizes a calendar identifier against the full Temporal calendar set
    /// (CLDR aliases, ASCII-case-insensitively); an unsupported id → RangeError.
    fn pdt_canonicalize_calendar(&mut self, s: &str) -> Result<String, ExecError> {
        match tcal::canonicalize_calendar(s) {
            Some(c) => Ok(String::from(c)),
            None => Err(self.pdt_range(&alloc::format!("invalid calendar identifier '{s}'"))),
        }
    }

    /// Reads the optional `calendar` field of a property bag (for `from`/`with`
    /// etc.) and returns its canonical id (default `"iso8601"`). A calendared
    /// Temporal object supplies its `[[Calendar]]`; a string is either a calendar
    /// id or a parseable ISO string whose `[u-ca=…]` annotation supplies one; a
    /// non-string, non-object value → TypeError; an unsupported id → RangeError.
    fn pdt_bag_calendar(&mut self, h: Handle) -> Result<String, ExecError> {
        let Some(v) = self.pdt_field(h, "calendar")? else {
            return Ok(String::from("iso8601"));
        };
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.pdt_canonicalize_calendar(&cal);
        }
        let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|x| self.realm.string_value(x))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        // A bare calendar identifier, or a date-ish ISO string whose `[u-ca=…]`
        // annotation names one. Partial forms (`"2020-01"`, `"01-01"`) are accepted
        // for the calendar slot even though they are not full PlainDateTime strings.
        if let Some(c) = tcal::canonicalize_calendar(&s) {
            return Ok(String::from(c));
        }
        if let Some(c) = pdt_calendar_string_id(&s) {
            return Ok(c);
        }
        Err(self.pdt_range(&alloc::format!("invalid calendar identifier '{s}'")))
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

    /// `PrepareTemporalFields` for a **non-ISO** date-time property bag, reading the
    /// recognised keys (including `era`/`eraYear`) in alphabetical order. Unlike the
    /// ISO reader, `monthCode` is kept as its raw well-formed string — the calendar
    /// layer decides its suitability.
    fn pdt_read_bag_cal(&mut self, h: Handle) -> Result<CalDtBag, ExecError> {
        let mut bag = CalDtBag::default();
        if let Some(v) = self.pdt_field(h, "day")? {
            bag.day = Some(self.pdt_to_int(v)?);
        }
        if let Some(v) = self.pdt_field(h, "era")? {
            bag.era = Some(self.coerce_to_string(v)?);
        }
        if let Some(v) = self.pdt_field(h, "eraYear")? {
            bag.era_year = Some(self.pdt_to_int(v)?);
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
            let prim = self.coerce_primitive(v, "string")?;
            let Some(s) = prim
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|x| self.realm.string_value(x))
            else {
                return Err(self.type_error("monthCode must be a string"));
            };
            // Well-formedness only (M + two digits + optional L); the calendar layer
            // checks whether the code occurs in the year.
            parse_month_code(&s).ok_or_else(|| self.pdt_range("invalid monthCode"))?;
            bag.month_code = Some(s);
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

    /// Regulates a non-ISO `CalDtBag`'s time fields into an [`IsoTime`], defaulting
    /// each absent component to 0.
    fn pdt_cal_bag_time(
        &mut self,
        bag: &CalDtBag,
        overflow: Overflow,
    ) -> Result<IsoTime, ExecError> {
        iso::regulate_iso_time(
            bag.hour.unwrap_or(0),
            bag.minute.unwrap_or(0),
            bag.second.unwrap_or(0),
            bag.ms.unwrap_or(0),
            bag.us.unwrap_or(0),
            bag.ns.unwrap_or(0),
            overflow,
        )
        .ok_or_else(|| self.pdt_range("invalid ISO time"))
    }

    /// Runs [`tcal::fields_to_iso`], mapping its error to the right exception.
    fn pdt_cal_fields_to_iso(
        &mut self,
        cal: &str,
        input: &tcal::FieldsInput,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        match tcal::fields_to_iso(cal, input, overflow) {
            Ok(d) => Ok(d),
            Err(tcal::CalError::Range(m)) => Err(self.pdt_range(&m)),
            Err(tcal::CalError::MissingFields(m)) => Err(self.type_error(&m)),
        }
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

    /// `ToTemporalDateTime(item, options)` → the ISO date + time + calendar id.
    /// Accepts a `PlainDateTime` (copy), a `PlainDate` (midnight), a property bag,
    /// or an ISO string.
    fn pdt_to_datetime(
        &mut self,
        item: NanBox,
        options: NanBox,
    ) -> Result<(IsoDate, IsoTime, String), ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(d) = self.realm.temporal_at(h) {
                let opts = self.pdt_options(options)?;
                self.pdt_overflow(opts)?; // validated even though ignored on copy
                return match d.kind {
                    TemporalKind::PlainDateTime => Ok((d.date, d.time, d.calendar.clone())),
                    TemporalKind::PlainDate => Ok((d.date, IsoTime::default(), d.calendar.clone())),
                    // A ZonedDateTime yields its wall-clock date+time in its zone.
                    TemporalKind::ZonedDateTime => {
                        let tz = d.tz.as_deref().unwrap_or("UTC");
                        let (date, time) =
                            crate::nbexec::temporal_zoneddatetime::local_of(tz, d.epoch_ns);
                        Ok((date, time, d.calendar.clone()))
                    }
                    _ => Err(self.type_error("expected a PlainDateTime")),
                };
            }
            if let Some(s) = self.realm.string_value(h) {
                // Parse (and range-check) the string *before* touching `options`, so
                // an invalid string throws without observing the options bag.
                let (date, time, cal) = pdt_parse_datetime(&s)
                    .ok_or_else(|| self.pdt_range("invalid PlainDateTime string"))?;
                if !pdt_in_range(date, time) {
                    return Err(self.pdt_range("PlainDateTime outside representable range"));
                }
                let opts = self.pdt_options(options)?;
                self.pdt_overflow(opts)?;
                return Ok((date, time, cal));
            }
            if self.is_object_value(item) {
                // Observable order (`ToTemporalDateTime`): read `calendar` and all
                // the date/time fields *before* `GetOptionsObject`, so a primitive
                // `options` value throws only after every field has been read.
                let cal = self.pdt_bag_calendar(h)?;
                if !tcal::is_iso(&cal) {
                    let opts = self.pdt_options(options)?;
                    return self.pdt_datetime_from_bag_cal(h, &cal, opts);
                }
                let bag = self.pdt_read_bag(h)?;
                let opts = self.pdt_options(options)?;
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
                return Ok((date, time, cal));
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.PlainDateTime"))
    }

    /// The non-ISO property-bag path of `ToTemporalDateTime`: reads the calendar
    /// fields (`era`/`eraYear`/`year`/`month`/`monthCode`/`day`) through the layer
    /// and the time fields directly.
    fn pdt_datetime_from_bag_cal(
        &mut self,
        h: Handle,
        cal: &str,
        opts: Option<Handle>,
    ) -> Result<(IsoDate, IsoTime, String), ExecError> {
        let bag = self.pdt_read_bag_cal(h)?;
        let overflow = self.pdt_overflow(opts)?;
        let day = bag.day.ok_or_else(|| self.type_error("day is required"))?;
        if bag.month.is_none() && bag.month_code.is_none() {
            return Err(self.type_error("month or monthCode is required"));
        }
        let input = tcal::FieldsInput {
            era: bag.era.clone(),
            era_year: bag.era_year,
            year: bag.year,
            month: bag.month,
            month_code: bag.month_code.clone(),
            day,
        };
        let date = self.pdt_cal_fields_to_iso(cal, &input, overflow)?;
        let time = self.pdt_cal_bag_time(&bag, overflow)?;
        if !pdt_in_range(date, time) {
            return Err(self.pdt_range("PlainDateTime outside representable range"));
        }
        Ok((date, time, String::from(cal)))
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
        if !tcal::is_iso(&data.calendar) {
            return self.pdt_with_cal(data, h, options);
        }
        let bag = self.pdt_read_bag(h)?;
        // `day`/`month` are read with `ToPositiveIntegerWithTruncation` during field
        // preparation, so a non-positive value is a RangeError *before* the options
        // object is validated (GetOptionsObject → TypeError).
        if matches!(bag.day, Some(d) if d < 1) || matches!(bag.month, Some(m) if m < 1) {
            return Err(self.pdt_range("month and day must be positive"));
        }
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
        Ok(self.pdt_make_cal(date, time, &data.calendar))
    }

    /// The non-ISO `with` path: merges the provided fields over the receiver's
    /// existing calendar fields, re-derives the ISO date through the layer, and
    /// merges the time fields over the receiver's wall-clock time.
    fn pdt_with_cal(
        &mut self,
        data: &TemporalData,
        h: Handle,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        let cal = data.calendar.as_str();
        let existing = tcal::iso_to_fields(cal, data.date);
        let bag = self.pdt_read_bag_cal(h)?;
        let opts = self.pdt_options(options)?;
        let overflow = self.pdt_overflow(opts)?;
        if !bag.any() {
            return Err(self.type_error("with() requires at least one recognised field"));
        }
        // Merge: an explicit year (or era+eraYear) wins; otherwise keep the
        // receiver's year. Likewise for month (prefer monthCode to preserve leap
        // months) and day.
        let (year, era, era_year) =
            if bag.year.is_some() || bag.era.is_some() || bag.era_year.is_some() {
                (bag.year, bag.era.clone(), bag.era_year)
            } else {
                (Some(existing.year), None, None)
            };
        let (month, month_code) = if bag.month.is_some() || bag.month_code.is_some() {
            (bag.month, bag.month_code.clone())
        } else {
            (None, Some(existing.month_code.clone()))
        };
        let day = bag.day.unwrap_or(existing.day);
        let input = tcal::FieldsInput {
            era,
            era_year,
            year,
            month,
            month_code,
            day,
        };
        let date = self.pdt_cal_fields_to_iso(cal, &input, overflow)?;
        let ct = data.time;
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
        Ok(self.pdt_make_cal(date, time, cal))
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
        // `CreateTemporalDateTime`: the combined date+time must be within the valid
        // ISO range (a boundary date with a different time can fall outside it).
        if !pdt_in_range(data.date, time) {
            return Err(self.pdt_range("combined date-time is outside the valid ISO range"));
        }
        Ok(self.pdt_make_cal(data.date, time, &data.calendar))
    }

    fn pdt_with_calendar(&mut self, data: &TemporalData, arg: NanBox) -> Result<NanBox, ExecError> {
        // `withCalendar` requires a calendar argument: `ToTemporalCalendarIdentifier`
        // of `undefined` is a TypeError (unlike the constructor's ISO default).
        if arg.is_undefined() {
            return Err(self.type_error("withCalendar requires a calendar argument"));
        }
        // `ToTemporalCalendarSlotValue`: unlike the constructor, `withCalendar`
        // accepts (via `ParseTemporalCalendarString`) a valid annotated ISO string
        // whose `[u-ca=…]` annotation supplies the calendar id (default iso8601).
        let cal = if let Some(c) = self.temporal_object_calendar(arg) {
            self.pdt_canonicalize_calendar(&c)?
        } else if let Some(s) = arg
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        {
            if let Some(c) = tcal::canonicalize_calendar(&s) {
                String::from(c)
            } else if let Some(raw) = iso::parse_calendar_string(&s)
                && let Some(c) = tcal::canonicalize_calendar(&raw)
            {
                String::from(c)
            } else {
                return Err(self.pdt_range(&alloc::format!("invalid calendar identifier '{s}'")));
            }
        } else {
            return Err(self.type_error("calendar must be a string"));
        };
        Ok(self.pdt_make_cal(data.date, data.time, &cal))
    }

    /// `ToTemporalTime(item)` → an ISO time. Accepts a `PlainTime`/`PlainDateTime`
    /// (its time), an ISO string, or a property bag of time fields.
    fn pdt_to_time(&mut self, item: NanBox) -> Result<IsoTime, ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(d) = self.realm.temporal_at(h) {
                return match d.kind {
                    TemporalKind::PlainTime | TemporalKind::PlainDateTime => Ok(d.time),
                    // ToTemporalTime on a ZonedDateTime uses its wall-clock time.
                    TemporalKind::ZonedDateTime => {
                        let tz = d.tz.clone().unwrap_or_else(|| String::from("UTC"));
                        let epoch = d.epoch_ns;
                        Ok(super::temporal_zoneddatetime::local_of(&tz, epoch).1)
                    }
                    _ => Err(self.type_error("expected a PlainTime")),
                };
            }
            if let Some(s) = self.realm.string_value(h) {
                let p = iso::parse_iso_time_string(&s)
                    .ok_or_else(|| self.pdt_range("invalid PlainTime string"))?;
                // `ParseTemporalTimeString` forbids the UTC designator (`Z`): a
                // string like "09:00:00Z" is not a valid PlainTime.
                if p.z {
                    return Err(
                        self.pdt_range("a PlainTime string must not carry a UTC designator")
                    );
                }
                return p
                    .time
                    .ok_or_else(|| self.pdt_range("string is missing a time"));
            }
            if self.is_object_value(item) {
                // ToTemporalTimeRecord reads *only* the time fields, in alphabetical
                // order (hour, microsecond, millisecond, minute, nanosecond, second),
                // through the getter/proxy-aware accessor.
                let mut t = [0_i64; 6]; // hour, minute, second, ms, us, ns
                let mut any = false;
                for (k, i) in [
                    ("hour", 0usize),
                    ("microsecond", 4),
                    ("millisecond", 3),
                    ("minute", 1),
                    ("nanosecond", 5),
                    ("second", 2),
                ] {
                    if let Some(v) = self.pdt_field(h, k)? {
                        any = true;
                        t[i] = self.pdt_to_int(v)?;
                    }
                }
                if !any {
                    return Err(self.type_error("no time fields present"));
                }
                return iso::regulate_iso_time(
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    t[5],
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
        // The time-part carry is calendar-independent; only the date-add step differs.
        let (day_carry, new_time) = iso::add_time(data.time, dur.time_nanos());
        let date_days = (dur.days + i128::from(day_carry)) as i64;
        let new_date = if tcal::is_iso(&data.calendar) {
            // ISO fast path — byte-for-byte the original computation.
            iso::add_iso_date(
                data.date,
                dur.years as i64,
                dur.months as i64,
                dur.weeks as i64,
                date_days,
                overflow,
            )
            .ok_or_else(|| self.pdt_range("result outside representable range"))?
        } else {
            // Non-ISO: add years/months in calendar terms, then weeks/days as a
            // plain day offset, via the shared calendar layer.
            match tcal::calendar_date_add(
                &data.calendar,
                data.date,
                dur.years as i64,
                dur.months as i64,
                dur.weeks as i64,
                date_days,
                overflow,
            ) {
                Ok(r) => r,
                Err(tcal::CalError::Range(m)) => return Err(self.pdt_range(&m)),
                Err(tcal::CalError::MissingFields(m)) => return Err(self.type_error(&m)),
            }
        };
        if !pdt_in_range(new_date, new_time) {
            return Err(self.pdt_range("result outside representable range"));
        }
        Ok(self.pdt_make_cal(new_date, new_time, &data.calendar))
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
                let val = n as i128;
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
        let (d2, t2, other_cal) = self.pdt_to_datetime(other, NanBox::undefined())?;
        // Both operands must share one calendar — including when the receiver is
        // ISO and the argument is not (`iso.until(gregory)` is a RangeError too).
        if !(tcal::is_iso(&data.calendar) && tcal::is_iso(&other_cal)) && other_cal != data.calendar
        {
            return Err(self
                .pdt_range("cannot compute the difference between dates of different calendars"));
        }
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
        // `GetDifferenceSettings`: read and cast *all* options first, in the order
        // largestUnit, roundingIncrement, roundingMode, smallestUnit — before any
        // algorithmic validation (largest-vs-smallest, increment maximum).
        // `largestUnit` additionally accepts `"auto"` (→ the default).
        let mut units_auto = units.to_vec();
        units_auto.push("auto");
        let largest_raw = match self.pdt_str_option(opts, "largestUnit", &units_auto)? {
            Some(s) if s != "auto" => Some(parse_unit(&s).unwrap_or(Unit::Day)),
            _ => None,
        };
        let increment = self.pdt_rounding_increment(opts)?;
        let mode = self.pdt_rounding_mode(opts, RoundMode::Trunc)?;
        let smallest = match self.pdt_str_option(opts, "smallestUnit", &units)? {
            Some(s) => parse_unit(&s).unwrap_or(Unit::Nanosecond),
            None => Unit::Nanosecond,
        };
        let largest = largest_raw.unwrap_or(Unit::Day.min(smallest));
        if largest > smallest {
            return Err(self.pdt_range("largestUnit must be at least as large as smallestUnit"));
        }
        // ValidateTemporalRoundingIncrement (non-inclusive) for time smallestUnits.
        if smallest >= Unit::Hour {
            self.pdt_validate_increment(smallest, increment)?;
        }

        let from = (data.date, data.time);
        let to = (d2, t2);
        // Day-or-finer largestUnit → pure time/day rounding; a calendar
        // smallestUnit (year/month/week) uses calendar-relative rounding
        // (NudgeToCalendarUnit); a coarser-largest with a day/time smallestUnit is
        // emitted with the calendar part unrounded.
        // A Day-or-finer largestUnit carries no year/month/week component, so it is
        // computed purely from epoch-days + wall-time nanoseconds — calendar-
        // independent, shared by every calendar. Only the coarser-largest paths,
        // whose date part spans calendar years/months/weeks, differ for non-ISO.
        let is_iso = tcal::is_iso(&data.calendar);
        // `since` rounds the reversed (from→to) difference with the rounding mode
        // negated (`NegateRoundingMode`), then negates the whole result below.
        let mode = if negate {
            pdt_negate_round_mode(mode)
        } else {
            mode
        };
        let mut dur = if largest >= Unit::Day {
            pdt_round_duration(from, to, largest, smallest, increment, mode)
        } else if matches!(smallest, Unit::Year | Unit::Month | Unit::Week) {
            let rounded = if is_iso {
                pdt_round_calendar(from, to, largest, smallest, increment, mode)
            } else {
                pdt_round_calendar_cal(&data.calendar, from, to, largest, smallest, increment, mode)
            };
            rounded.map_err(|()| self.pdt_range("rounded date is outside the valid ISO range"))?
        } else if is_iso {
            // A calendar `largestUnit` with a day/time `smallestUnit`: round the
            // day+time span (NudgeToDayOrTimeUnit + BubbleRelativeDuration). When no
            // rounding is required (nanoseconds, increment 1) the raw difference is
            // already exact.
            if smallest == Unit::Nanosecond && increment == 1 {
                pdt_difference(from, to, largest)
            } else {
                pdt_round_day_time(from, to, largest, smallest, increment, mode)
            }
        } else {
            pdt_difference_cal(&data.calendar, from, to, largest)
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
            .is_some_and(|h| self.realm.is_string_handle(h));
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
        Ok(self.pdt_make_cal(date, time, &data.calendar))
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
        // Read/cast all options first, in spec order: calendarName (above),
        // fractionalSecondDigits, roundingMode, smallestUnit.
        let frac = self.pdt_frac_digits(opts)?;
        let mode = self.pdt_rounding_mode(opts, RoundMode::Trunc)?;
        let smallest = self
            .pdt_str_option(opts, "smallestUnit", &time_units)?
            .map(|s| parse_unit(&s).unwrap_or(Unit::Nanosecond));

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
        let id = data.calendar.as_str();
        match cal.as_str() {
            "always" => out.push_str(&alloc::format!("[u-ca={id}]")),
            "critical" => out.push_str(&alloc::format!("[!u-ca={id}]")),
            "auto" if !tcal::is_iso(id) => out.push_str(&alloc::format!("[u-ca={id}]")),
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
            // GetStringOrNumberOption: floor first, *then* range-check, so e.g.
            // 9.7 → 9 (valid) and -0.6 → -1 (RangeError).
            if n.is_nan() {
                return Err(self.pdt_range("fractionalSecondDigits out of range"));
            }
            let f = n.floor();
            if !(0.0..=9.0).contains(&f) {
                return Err(self.pdt_range("fractionalSecondDigits out of range"));
            }
            return Ok(Some(f as u8));
        }
        let s = self.coerce_to_string(v)?;
        if s == "auto" {
            Ok(None)
        } else {
            Err(self.pdt_range("invalid fractionalSecondDigits"))
        }
    }

    // --- result builders ---------------------------------------------------

    /// Builds a `Temporal.PlainDateTime` carrying calendar id `cal` (default
    /// `"iso8601"`), linked to the intrinsic prototype.
    fn pdt_make_cal(&mut self, date: IsoDate, time: IsoTime, cal: &str) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainDateTime,
            date,
            time,
            calendar: String::from(cal),
            ..Default::default()
        };
        self.pdt_alloc(data, TemporalKind::PlainDateTime)
    }

    fn pdt_make_date(&mut self, date: IsoDate, cal: &str) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainDate,
            date,
            calendar: String::from(cal),
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
        // Duration fields are Numbers: quantize to float64-representable integers.
        let dur = iso::quantize_duration_fields(dur);
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
/// date+time plus the calendar id from the first `[u-ca=…]` annotation
/// (canonicalized, defaulting to `"iso8601"`), or `None` for any malformed /
/// unsupported form.
fn pdt_parse_datetime(s: &str) -> Option<(IsoDate, IsoTime, String)> {
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
    let cal = pdt_parse_annotations(&mut c)?;
    (c.i == c.b.len()).then_some((date, time, cal))
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
/// (lowercase keys; a single time-zone annotation; no conflicting-critical or
/// unknown-critical annotations). Returns the calendar id from the first
/// `[u-ca=…]` annotation (canonicalized, defaulting to `"iso8601"`), or `None` on
/// a rule violation or an unsupported calendar id.
fn pdt_parse_annotations(c: &mut PdtCursor) -> Option<String> {
    let mut cal_count = 0;
    let mut cal_critical = false;
    let mut first_cal: Option<String> = None;
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
                    first_cal = Some(core::str::from_utf8(val).ok()?.into());
                }
            } else if critical {
                return None; // unknown annotation with the critical flag
            }
        } else {
            tz_count += 1; // a time-zone annotation
        }
    }
    if (cal_count > 1 && cal_critical) || tz_count > 1 {
        return None;
    }
    match first_cal {
        Some(v) => tcal::canonicalize_calendar(&v).map(String::from),
        None => Some(String::from("iso8601")),
    }
}

/// The canonical calendar id named by a property-bag `calendar` **string** that is
/// a date-ish ISO string (leading digit/sign) whose first `u-ca=` annotation
/// supplies it (defaulting to `"iso8601"`), or `None` when the string is not a
/// date-ish form or carries an unsupported calendar id.
fn pdt_calendar_string_id(s: &str) -> Option<String> {
    if !s
        .as_bytes()
        .first()
        .is_some_and(|&c| c.is_ascii_digit() || c == b'+' || c == b'-')
    {
        return None;
    }
    // Minus-zero is not a valid extended year (`-000000`).
    if s.starts_with("-000000") {
        return None;
    }
    match s.find("u-ca=") {
        Some(p) => {
            let after = &s[p + 5..];
            let end = after.find(']').unwrap_or(after.len());
            tcal::canonicalize_calendar(&after[..end]).map(String::from)
        }
        None => Some(String::from("iso8601")),
    }
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
    dur.years = i128::from(y);
    dur.months = i128::from(mo);
    dur.weeks = i128::from(w);
    dur.days = i128::from(d);
    dur
}

/// Balances a signed nanosecond total into a duration down to `largest`, where
/// `largest` is Day or a finer unit.
fn pdt_balance_datetime(total_ns: i128, largest: Unit) -> DurationFields {
    let sign = total_ns.signum();
    let mut r = total_ns.abs();
    let (mut weeks, mut days) = (0_i128, 0_i128);
    if largest <= Unit::Day {
        days = r / iso::NS_PER_DAY;
        r %= iso::NS_PER_DAY;
        if largest == Unit::Week {
            weeks = days / 7;
            days %= 7;
        }
    }
    let mut dur = iso::balance_time_duration(r * sign, largest);
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

/// `RoundDuration` for a date+time difference rounded to a calendar
/// `smallestUnit` (Year/Month/Week). Keeps the components coarser than
/// `smallest`, then rounds the `smallest` count measured against the *local
/// nanosecond* span of one increment step (so time-of-day is folded into the
/// fraction), per `NudgeToCalendarUnit`.
/// `NegateRoundingMode`: swaps ceil/floor and halfCeil/halfFloor; symmetric modes
/// (expand/trunc/halfExpand/halfTrunc/halfEven) are unchanged.
fn pdt_negate_round_mode(mode: RoundMode) -> RoundMode {
    match mode {
        RoundMode::Ceil => RoundMode::Floor,
        RoundMode::Floor => RoundMode::Ceil,
        RoundMode::HalfCeil => RoundMode::HalfFloor,
        RoundMode::HalfFloor => RoundMode::HalfCeil,
        other => other,
    }
}

fn pdt_round_calendar(
    from: (IsoDate, IsoTime),
    to: (IsoDate, IsoTime),
    largest: Unit,
    smallest: Unit,
    increment: i64,
    mode: RoundMode,
) -> Result<DurationFields, ()> {
    let (from_date, from_time) = from;
    let base = pdt_difference(from, to, largest);
    let to_ns = iso::iso_to_epoch_days(to.0) as i128 * iso::NS_PER_DAY + iso::time_to_nanos(to.1);
    let from_ns =
        iso::iso_to_epoch_days(from_date) as i128 * iso::NS_PER_DAY + iso::time_to_nanos(from_time);
    let sign = (to_ns - from_ns).signum();
    let sign_i = sign as i64;
    let (keep_y, keep_m) = match smallest {
        Unit::Year => (0, 0),
        Unit::Month => (base.years, 0),
        _ => (base.years, base.months), // Week
    };
    let anchor = iso::add_iso_date(
        from_date,
        keep_y as i64,
        keep_m as i64,
        0,
        0,
        Overflow::Constrain,
    )
    .unwrap_or(from_date);
    // Local pseudo-epoch (ns) of `anchor + count smallest-units`, keeping `from`'s
    // wall time.
    let step_ns = |count: i64| -> i128 {
        let (y, m, w) = match smallest {
            Unit::Year => (count, 0, 0),
            Unit::Month => (0, count, 0),
            _ => (0, 0, count),
        };
        let nd = iso::add_iso_date(anchor, y, m, w, 0, Overflow::Constrain).unwrap_or(anchor);
        iso::iso_to_epoch_days(nd) as i128 * iso::NS_PER_DAY + iso::time_to_nanos(from_time)
    };
    let mut out = DurationFields {
        years: keep_y,
        months: keep_m,
        ..Default::default()
    };
    if sign == 0 {
        return Ok(out);
    }
    let seed = pdt_difference((anchor, from_time), to, smallest);
    let mut r1 = (match smallest {
        Unit::Year => seed.years,
        Unit::Month => seed.months,
        _ => seed.weeks,
    }) as i64;
    let beyond = |e: i128| if sign > 0 { e > to_ns } else { e < to_ns };
    for _ in 0..14 {
        if beyond(step_ns(r1 + sign_i)) {
            break;
        }
        r1 += sign_i;
    }
    for _ in 0..14 {
        if beyond(step_ns(r1)) {
            r1 -= sign_i;
        } else {
            break;
        }
    }
    // `NudgeToCalendarUnit` projects the ceil (`r2`) increment-multiple endpoint via
    // `CalendarDateAdd`; a date outside the representable ISO range throws.
    let inc = increment.max(1);
    let r2 = (r1 / inc) * inc + inc * sign_i;
    let (ry, rm, rw) = match smallest {
        Unit::Year => (r2, 0, 0),
        Unit::Month => (0, r2, 0),
        _ => (0, 0, r2),
    };
    if iso::add_iso_date(anchor, ry, rm, rw, 0, Overflow::Constrain).is_none() {
        return Err(());
    }
    let e1 = step_ns(r1);
    let count = if e1 == to_ns {
        // Even an exact-boundary count must still snap to `roundingIncrement`.
        iso::round_to_increment(i128::from(r1), i128::from(increment.max(1)), mode) as i64
    } else {
        let e2 = step_ns(r1 + sign_i);
        let den = (e2 - e1).abs().max(1);
        let num = (to_ns - e1).abs();
        let x = i128::from(r1) * den + sign * num;
        (iso::round_to_increment(x, i128::from(increment.max(1)) * den, mode) / den) as i64
    };
    match smallest {
        Unit::Year => out.years = i128::from(count),
        Unit::Month => {
            // `BubbleRelativeDuration`: a rounded month count can reach a whole
            // year; re-express the rounded target date as a difference in `largest`.
            let rd =
                iso::add_iso_date(anchor, 0, count, 0, 0, Overflow::Constrain).unwrap_or(anchor);
            let (by, bm, _bw, _bd) = iso::difference_iso_date(from_date, rd, largest);
            out.years = i128::from(by);
            out.months = i128::from(bm);
        }
        _ => out.weeks = i128::from(count),
    }
    Ok(out)
}

/// `RoundRelativeDuration` for a PlainDateTime difference with a calendar
/// `largestUnit` (year/month/week) but a **day-or-time** `smallestUnit`.
/// `NudgeToDayOrTimeUnit` rounds the combined whole-day + time-of-day span to the
/// smallest unit (a rounded-up time carries into the day count), then
/// `BubbleRelativeDuration` re-expresses the whole-day part in `largestUnit` terms
/// (so a carried day can cross a month/year boundary). Year boundaries that fall
/// outside the representable range are never materialised — only the actual end
/// date is, so a result that stays within a few days of the limit does not throw.
fn pdt_round_day_time(
    from: (IsoDate, IsoTime),
    to: (IsoDate, IsoTime),
    largest: Unit,
    smallest: Unit,
    increment: i64,
    mode: RoundMode,
) -> DurationFields {
    let base = pdt_difference(from, to, largest);
    // NudgeToDayOrTimeUnit: round (whole days + time-of-day), in nanoseconds, to
    // the smallest unit; `days * nsPerDay` is already a multiple of any sub-day
    // unit, so this rounds only the time part and carries into the day count.
    let total_ns = base.days * iso::NS_PER_DAY + base.time_nanos();
    let unit = unit_ns(smallest) * i128::from(increment.max(1));
    let rounded = iso::round_to_increment(total_ns, unit, mode);
    let new_days = (rounded / iso::NS_PER_DAY) as i64;
    let rem = rounded % iso::NS_PER_DAY;
    // BubbleRelativeDuration: the whole-day date part, re-expressed in largestUnit.
    let end_date = iso::add_iso_date(
        from.0,
        base.years as i64,
        base.months as i64,
        base.weeks as i64,
        new_days,
        Overflow::Constrain,
    )
    .unwrap_or(from.0);
    let (by, bm, bw, bd) = iso::difference_iso_date(from.0, end_date, largest);
    let mut out = iso::balance_time_duration(rem, Unit::Hour);
    out.years = i128::from(by);
    out.months = i128::from(bm);
    out.weeks = i128::from(bw);
    out.days = i128::from(bd);
    out
}

/// The calendar-aware analogue of [`pdt_difference`] for a non-ISO calendar:
/// `DifferenceISODateTime` with the date part measured in the calendar's own
/// year/month/week terms via [`tcal::calendar_date_until`]. The day-borrow that
/// balances a wall-time pointing opposite the date direction is identical to the
/// ISO path (it is a plain epoch-day shift); only the date-difference engine
/// differs.
fn pdt_difference_cal(
    cal: &str,
    from: (IsoDate, IsoTime),
    to: (IsoDate, IsoTime),
    largest: Unit,
) -> DurationFields {
    // Split the time part off, borrowing a day when it points opposite the date
    // direction (mirrors the ISO path byte-for-byte).
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
    let parts = tcal::calendar_date_until(cal, from.0, adjusted_to, largest);
    let mut dur = iso::balance_time_duration(time_ns, Unit::Hour);
    dur.years = i128::from(parts.years);
    dur.months = i128::from(parts.months);
    dur.weeks = i128::from(parts.weeks);
    dur.days = i128::from(parts.days);
    dur
}

/// The calendar-aware analogue of [`pdt_round_calendar`] for a non-ISO calendar:
/// the `from → to` difference is measured in the calendar's own year/month terms
/// (via [`pdt_difference_cal`] / [`tcal::calendar_date_until`]) and the coarse-
/// unit nudge steps with [`tcal::calendar_date_add`], so month lengths and leap
/// months are honoured. The wall-time is folded into the rounding fraction exactly
/// as in the ISO path (identical local-nanosecond arithmetic).
#[allow(clippy::too_many_arguments)]
fn pdt_round_calendar_cal(
    cal: &str,
    from: (IsoDate, IsoTime),
    to: (IsoDate, IsoTime),
    largest: Unit,
    smallest: Unit,
    increment: i64,
    mode: RoundMode,
) -> Result<DurationFields, ()> {
    let (from_date, from_time) = from;
    let base = pdt_difference_cal(cal, from, to, largest);
    let to_ns = iso::iso_to_epoch_days(to.0) as i128 * iso::NS_PER_DAY + iso::time_to_nanos(to.1);
    let from_ns =
        iso::iso_to_epoch_days(from_date) as i128 * iso::NS_PER_DAY + iso::time_to_nanos(from_time);
    let sign = (to_ns - from_ns).signum();
    let sign_i = sign as i64;
    let (keep_y, keep_m) = match smallest {
        Unit::Year => (0, 0),
        Unit::Month => (base.years, 0),
        _ => (base.years, base.months), // Week
    };
    // Calendar-aware add relative to a base (Constrain); `None` when the result
    // falls outside the representable ISO range.
    let cadd_opt = |base: IsoDate, y: i64, m: i64, w: i64| -> Option<IsoDate> {
        tcal::calendar_date_add(cal, base, y, m, w, 0, Overflow::Constrain).ok()
    };
    let cadd = |base: IsoDate, y: i64, m: i64, w: i64| -> IsoDate {
        cadd_opt(base, y, m, w).unwrap_or(base)
    };
    let anchor = cadd(from_date, keep_y as i64, keep_m as i64, 0);
    // Local pseudo-epoch (ns) of `anchor + count smallest-units`, keeping `from`'s
    // wall time.
    let unit_add = |count: i64| -> Option<IsoDate> {
        let (y, m, w) = match smallest {
            Unit::Year => (count, 0, 0),
            Unit::Month => (0, count, 0),
            _ => (0, 0, count),
        };
        cadd_opt(anchor, y, m, w)
    };
    let step_ns = |count: i64| -> i128 {
        let nd = unit_add(count).unwrap_or(anchor);
        iso::iso_to_epoch_days(nd) as i128 * iso::NS_PER_DAY + iso::time_to_nanos(from_time)
    };
    let mut out = DurationFields {
        years: keep_y,
        months: keep_m,
        ..Default::default()
    };
    if sign == 0 {
        return Ok(out);
    }
    let seed = pdt_difference_cal(cal, (anchor, from_time), to, smallest);
    let mut r1 = (match smallest {
        Unit::Year => seed.years,
        Unit::Month => seed.months,
        _ => seed.weeks,
    }) as i64;
    let beyond = |e: i128| if sign > 0 { e > to_ns } else { e < to_ns };
    for _ in 0..14 {
        if beyond(step_ns(r1 + sign_i)) {
            break;
        }
        r1 += sign_i;
    }
    for _ in 0..14 {
        if beyond(step_ns(r1)) {
            r1 -= sign_i;
        } else {
            break;
        }
    }
    // `NudgeToCalendarUnit`: the ceil (`r2`) increment-multiple endpoint must be
    // representable, else a RangeError.
    let inc = increment.max(1);
    let r2 = (r1 / inc) * inc + inc * sign_i;
    if unit_add(r2).is_none() {
        return Err(());
    }
    let e1 = step_ns(r1);
    let count = if e1 == to_ns {
        r1
    } else {
        let e2 = step_ns(r1 + sign_i);
        let den = (e2 - e1).abs().max(1);
        let num = (to_ns - e1).abs();
        let x = i128::from(r1) * den + sign * num;
        (iso::round_to_increment(x, i128::from(increment.max(1)) * den, mode) / den) as i64
    };
    match smallest {
        Unit::Year => out.years = i128::from(count),
        Unit::Month => out.months = i128::from(count),
        _ => out.weeks = i128::from(count),
    }
    Ok(out)
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
