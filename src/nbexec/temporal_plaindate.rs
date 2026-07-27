//! `Temporal.PlainDate` — logic module. A fan-out unit: everything specific to
//! `PlainDate` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::temporal_calendar as tcal;
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    self, DurationFields, IsoDate, MAX_EPOCH_DAYS, MIN_EPOCH_DAYS, Overflow, TemporalData,
    TemporalKind, Unit, add_iso_date, compare_iso_date, difference_iso_date, format_iso_year,
    is_leap_year, iso_day_of_week, iso_day_of_year, iso_days_in_month, iso_days_in_year,
    iso_to_epoch_days, iso_week_of_year, pad, parse_iso_datetime, parse_iso_duration,
    regulate_iso_date,
};

/// Prototype method names installed on `Temporal.PlainDate.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "add",
    "subtract",
    "with",
    "withCalendar",
    "until",
    "since",
    "equals",
    "toPlainDateTime",
    "toPlainYearMonth",
    "toPlainMonthDay",
    "toZonedDateTime",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
];
/// Getter-accessor names installed on `Temporal.PlainDate.prototype`.
pub(crate) const GETTERS: &[&str] = &[
    "calendarId",
    "era",
    "eraYear",
    "year",
    "month",
    "monthCode",
    "day",
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

impl<'a> Interp<'a> {
    /// `new Temporal.PlainDate(...)`.
    pub(crate) fn plaindate_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        // NewTarget-undefined (called as a plain function) is rejected by the
        // caller before reaching here.
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        let y = self.pd_to_integer_trunc(arg(0))?;
        let m = self.pd_to_integer_trunc(arg(1))?;
        let d = self.pd_to_integer_trunc(arg(2))?;
        // Calendar: undefined -> iso8601; a non-string -> TypeError; a string ->
        // canonicalize (the full Temporal calendar-id set, case-insensitively).
        let calendar = self.pd_validate_calendar(arg(3))?;
        let date = self.pd_reject_iso_date(y, m, d)?;
        let data = TemporalData {
            kind: TemporalKind::PlainDate,
            date,
            calendar,
            ..Default::default()
        };
        self.finish_temporal(data, new_target, callee)
    }

    /// A `Temporal.PlainDate.prototype.<method>()` call.
    pub(crate) fn plaindate_method(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        let cal = data.calendar.clone();
        match method {
            "add" => self.pd_add(data.date, &cal, arg(0), arg(1), false),
            "subtract" => self.pd_add(data.date, &cal, arg(0), arg(1), true),
            "with" => self.pd_with(data.date, &cal, arg(0), arg(1)),
            "withCalendar" => self.pd_with_calendar(data.date, arg(0)),
            "until" => self.pd_difference(data.date, &cal, arg(0), arg(1), false),
            "since" => self.pd_difference(data.date, &cal, arg(0), arg(1), true),
            "equals" => {
                let (other, ocal) = self.pd_to_temporal_date(arg(0), NanBox::undefined())?;
                Ok(NanBox::boolean(other == data.date && ocal == cal))
            }
            "toPlainDateTime" => {
                let time = self.pd_to_temporal_time(arg(0))?;
                // `ISODateTimeWithinLimits` on the combined date-time.
                let ns = crate::temporal_iso::iso_to_epoch_days(data.date) as i128
                    * crate::temporal_iso::NS_PER_DAY
                    + crate::temporal_iso::time_to_nanos(time);
                if !(ns > crate::temporal_iso::MIN_EPOCH_NS - crate::temporal_iso::NS_PER_DAY
                    && ns < crate::temporal_iso::MAX_EPOCH_NS + crate::temporal_iso::NS_PER_DAY)
                {
                    return Err(self
                        .pd_range_error("combined date-time is outside the representable range"));
                }
                Ok(self.pd_new_kind_cal(TemporalKind::PlainDateTime, data.date, time, &cal))
            }
            "toPlainYearMonth" => {
                // A PlainYearMonth stores a reference ISO day; for the ISO
                // calendar `ISOYearMonthFromFields` fixes it at 1.
                let mut d = data.date;
                if tcal::is_iso(&cal) {
                    d.day = 1;
                }
                Ok(self.pd_new_kind_cal(
                    TemporalKind::PlainYearMonth,
                    d,
                    temporal_iso::IsoTime::default(),
                    &cal,
                ))
            }
            "toPlainMonthDay" => {
                // A PlainMonthDay stores a reference ISO year; for the ISO
                // calendar the canonical reference year is 1972 (a leap year).
                let mut d = data.date;
                if tcal::is_iso(&cal) {
                    d.year = 1972;
                }
                Ok(self.pd_new_kind_cal(
                    TemporalKind::PlainMonthDay,
                    d,
                    temporal_iso::IsoTime::default(),
                    &cal,
                ))
            }
            "toString" => {
                let s = self.pd_to_string(data.date, &cal, arg(0))?;
                Ok(self.new_str(&s))
            }
            "toJSON" | "toLocaleString" => {
                let s = self.pd_format(data.date, &cal, "auto");
                Ok(self.new_str(&s))
            }
            "valueOf" => Err(self.type_error(
                "Called Temporal.PlainDate.prototype.valueOf, use compare() or equals() instead",
            )),
            "toZonedDateTime" => {
                // `date.toZonedDateTime(timeZone | { timeZone, plainTime })` → the
                // instant of the (date, plainTime|midnight) wall time in the zone.
                let item = arg(0);
                // A plain (non-Temporal) object is a `{ timeZone, plainTime }` bag:
                // read `timeZone` first (observably), then `plainTime`. When the bag
                // carries no `timeZone`, the object itself is the time-zone-like. A
                // String or `Temporal.ZonedDateTime` is itself the time-zone-like
                // (with no separate plain time).
                let is_plain_obj = self.is_object_value(item)
                    && item
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.temporal_at(h))
                        .is_none();
                // `time` is `None` when no `plainTime` was supplied — the spec then
                // uses `GetStartOfDay` (DST-gap aware), not a plain midnight.
                let (tz, time): (_, Option<temporal_iso::IsoTime>) = if is_plain_obj {
                    let h = item.as_handle().map(Handle::from_raw).unwrap();
                    let tzlike = self.read_member(h, "timeZone")?;
                    if tzlike.is_undefined() {
                        (self.temporal_tz_arg(item)?, None)
                    } else {
                        let tz = self.temporal_tz_arg(tzlike)?;
                        // `plainTime` runs the full `ToTemporalTime` (string/bag/
                        // PlainTime/PlainDateTime/ZonedDateTime); absent → start of day.
                        let ptv = self.read_member(h, "plainTime")?;
                        let time = if ptv.is_undefined() {
                            None
                        } else {
                            Some(self.pd_to_temporal_time(ptv)?)
                        };
                        (tz, time)
                    }
                } else {
                    (self.temporal_tz_arg(item)?, None)
                };
                // `ISODateTimeWithinLimits` on the combined wall date-time.
                let time_of_day = time.unwrap_or_default();
                let local_ns = crate::temporal_iso::iso_to_epoch_days(data.date) as i128
                    * crate::temporal_iso::NS_PER_DAY
                    + crate::temporal_iso::time_to_nanos(time_of_day);
                if !(local_ns > crate::temporal_iso::MIN_EPOCH_NS - crate::temporal_iso::NS_PER_DAY
                    && local_ns
                        < crate::temporal_iso::MAX_EPOCH_NS + crate::temporal_iso::NS_PER_DAY)
                {
                    return Err(self
                        .pd_range_error("combined date-time is outside the representable range"));
                }
                // `GetStartOfDay` when no time was given (DST-gap aware); otherwise
                // `GetEpochNanosecondsFor` with `compatible` disambiguation — a naïve
                // offset-at-the-wall-instant conversion is wrong near DST transitions.
                let epoch_ns = if time.is_none() {
                    crate::nbexec::temporal_zoneddatetime::start_of_day_pub(
                        &tz,
                        crate::temporal_iso::iso_to_epoch_days(data.date),
                    )
                    .filter(|e| {
                        (crate::temporal_iso::MIN_EPOCH_NS..=crate::temporal_iso::MAX_EPOCH_NS)
                            .contains(e)
                    })
                    .ok_or_else(|| {
                        self.pd_range_error("resulting instant is outside the representable range")
                    })?
                } else {
                    crate::nbexec::temporal_zoneddatetime::epoch_for_wall_disamb(
                        &tz,
                        local_ns,
                        "compatible",
                    )
                    .ok()
                    .filter(|e| {
                        (crate::temporal_iso::MIN_EPOCH_NS..=crate::temporal_iso::MAX_EPOCH_NS)
                            .contains(e)
                    })
                    .ok_or_else(|| {
                        self.pd_range_error("resulting instant is outside the representable range")
                    })?
                };
                Ok(self.build_temporal(crate::temporal_iso::TemporalData {
                    kind: crate::temporal_iso::TemporalKind::ZonedDateTime,
                    epoch_ns,
                    tz: Some(tz),
                    calendar: cal.clone(),
                    ..Default::default()
                }))
            }
            _ => Err(self.temporal_todo(&alloc::format!("PlainDate.prototype.{method}"))),
        }
    }

    /// A `Temporal.PlainDate.prototype.<getter>` read.
    pub(crate) fn plaindate_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let date = data.date;
        let cal = data.calendar.as_str();
        if name == "calendarId" {
            return Ok(self.new_str(cal));
        }
        // ISO-8601 fast path — byte-for-byte the original computation.
        if tcal::is_iso(cal) {
            return Ok(match name {
                // era / eraYear are undefined for the ISO 8601 calendar.
                "era" | "eraYear" => NanBox::undefined(),
                "year" => NanBox::number(f64::from(date.year)),
                "month" => NanBox::number(f64::from(date.month)),
                "monthCode" => {
                    let s = alloc::format!("M{}", pad(u64::from(date.month), 2));
                    self.new_str(&s)
                }
                "day" => NanBox::number(f64::from(date.day)),
                "dayOfWeek" => NanBox::number(f64::from(iso_day_of_week(date))),
                "dayOfYear" => NanBox::number(f64::from(iso_day_of_year(date))),
                "weekOfYear" => NanBox::number(f64::from(iso_week_of_year(date).0)),
                "yearOfWeek" => NanBox::number(f64::from(iso_week_of_year(date).1)),
                "daysInWeek" => NanBox::number(7.0),
                "daysInMonth" => {
                    NanBox::number(f64::from(iso_days_in_month(date.year, date.month)))
                }
                "daysInYear" => NanBox::number(f64::from(iso_days_in_year(date.year))),
                "monthsInYear" => NanBox::number(12.0),
                "inLeapYear" => NanBox::boolean(is_leap_year(date.year)),
                _ => return Err(self.temporal_todo(&alloc::format!("PlainDate getter {name}"))),
            });
        }
        // Non-ISO calendar: route through the calendar abstraction layer.
        let f = tcal::iso_to_fields(cal, date);
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
            "dayOfWeek" => NanBox::number(tcal::day_of_week(date) as f64),
            "dayOfYear" => NanBox::number(tcal::day_of_year(cal, date) as f64),
            "weekOfYear" => match tcal::week_of_year(cal, date) {
                Some((w, _)) => NanBox::number(w as f64),
                None => NanBox::undefined(),
            },
            "yearOfWeek" => match tcal::year_of_week(cal, date) {
                Some(y) => NanBox::number(y as f64),
                None => NanBox::undefined(),
            },
            "daysInWeek" => NanBox::number(tcal::days_in_week() as f64),
            "daysInMonth" => NanBox::number(tcal::days_in_month(cal, date) as f64),
            "daysInYear" => NanBox::number(tcal::days_in_year(cal, date) as f64),
            "monthsInYear" => NanBox::number(tcal::months_in_year(cal, date) as f64),
            "inLeapYear" => NanBox::boolean(tcal::in_leap_year(cal, date)),
            _ => return Err(self.temporal_todo(&alloc::format!("PlainDate getter {name}"))),
        })
    }

    /// A `Temporal.PlainDate.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn plaindate_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method {
            "from" => {
                // ToTemporalDate: the options (overflow) are read *after* the
                // item is converted (observable-order), so pass them through.
                let (date, cal) = self.pd_to_temporal_date(arg(0), arg(1))?;
                Ok(Some(self.pd_new_cal(date, &cal)))
            }
            "compare" => {
                let (a, _) = self.pd_to_temporal_date(arg(0), NanBox::undefined())?;
                let (b, _) = self.pd_to_temporal_date(arg(1), NanBox::undefined())?;
                let c = match compare_iso_date(a, b) {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Equal => 0.0,
                    core::cmp::Ordering::Greater => 1.0,
                };
                Ok(Some(NanBox::number(c)))
            }
            _ => Ok(None),
        }
    }

    // -- helpers ------------------------------------------------------------

    /// A `RangeError` throw with `message`.
    fn pd_range_error(&mut self, message: &str) -> ExecError {
        let m = self.new_str(message);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// `ToIntegerWithTruncation`: `ToNumber` (Symbol/BigInt -> TypeError), then a
    /// **RangeError** for a non-finite value, else truncate toward zero.
    fn pd_to_integer_trunc(&mut self, v: NanBox) -> Result<f64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.pd_range_error("PlainDate component must be a finite integer"));
        }
        Ok(n.trunc())
    }

    /// Validates a constructor calendar argument and returns its canonical id:
    /// `undefined` -> `"iso8601"`; a calendared Temporal object -> its id; a
    /// string -> canonicalized (RangeError if unsupported); a non-string ->
    /// TypeError.
    fn pd_validate_calendar(&mut self, v: NanBox) -> Result<String, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(String::from("iso8601"));
        }
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.pd_canonicalize_calendar(&cal);
        }
        let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        self.pd_canonicalize_calendar(&s)
    }

    /// Canonicalizes a calendar identifier string against the full Temporal
    /// calendar set (CLDR aliases, ASCII-case-insensitively). An unsupported id
    /// is a RangeError.
    fn pd_canonicalize_calendar(&mut self, s: &str) -> Result<String, ExecError> {
        match tcal::canonicalize_calendar(s) {
            Some(c) => Ok(String::from(c)),
            None => Err(self.pd_range_error(&alloc::format!("invalid calendar identifier '{s}'"))),
        }
    }

    /// Whether `date` is within the representable range of a `Temporal.PlainDate`
    /// (`ISODateWithinLimits`): epoch days in `[MIN_EPOCH_DAYS - 1, MAX_EPOCH_DAYS]`.
    fn pd_in_range(date: IsoDate) -> bool {
        let d = iso_to_epoch_days(date);
        (MIN_EPOCH_DAYS - 1..=MAX_EPOCH_DAYS).contains(&d)
    }

    /// `RejectISODate` + range check on already-truncated components.
    fn pd_reject_iso_date(&mut self, y: f64, m: f64, d: f64) -> Result<IsoDate, ExecError> {
        if !(f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&y) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        let Some(date) = regulate_iso_date(y as i32, m as i64, d as i64, Overflow::Reject) else {
            return Err(self.pd_range_error("date is outside the representable range"));
        };
        if !Self::pd_in_range(date) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        Ok(date)
    }

    /// Regulates `(year, month, day)` with `overflow` and checks the ISO range,
    /// yielding a RangeError on failure.
    fn pd_make_date(
        &mut self,
        year: i64,
        month: i64,
        day: i64,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&year) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        let Some(date) = regulate_iso_date(year as i32, month, day, overflow) else {
            return Err(self.pd_range_error("date is outside the representable range"));
        };
        if !Self::pd_in_range(date) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        Ok(date)
    }

    /// `ToPositiveIntegerWithTruncation`: like [`Self::pd_to_integer_trunc`] but a
    /// value below 1 is a RangeError (used for `month`/`day` calendar fields).
    fn pd_to_positive_integer(&mut self, v: NanBox) -> Result<i64, ExecError> {
        let n = self.pd_to_integer_trunc(v)?;
        if n < 1.0 {
            return Err(self.pd_range_error("value must be a positive integer"));
        }
        Ok(n as i64)
    }

    /// `ToIntegerIfIntegral`: `ToNumber`, then a **RangeError** unless the result
    /// is a finite integer (used for Duration property-bag fields).
    fn pd_to_integer_if_integral(&mut self, v: NanBox) -> Result<i128, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() || n.fract() != 0.0 {
            return Err(self.pd_range_error("duration field must be an integer"));
        }
        Ok(n as i128)
    }

    /// Builds a fresh `Temporal.PlainDate` linked to the intrinsic prototype.
    fn pd_new(&mut self, date: IsoDate) -> NanBox {
        self.pd_new_kind(
            TemporalKind::PlainDate,
            date,
            temporal_iso::IsoTime::default(),
        )
    }

    /// Builds a fresh `Temporal.PlainDate` carrying calendar id `cal`, linked to
    /// the intrinsic prototype. (For `"iso8601"` this is equivalent to
    /// [`Self::pd_new`], since that is the default calendar.)
    fn pd_new_cal(&mut self, date: IsoDate, cal: &str) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainDate,
            date,
            calendar: String::from(cal),
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainDate) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// Builds a fresh Temporal instance of a date/time kind linked to that kind's
    /// intrinsic prototype (used for PlainDate and the `toPlain*` conversions).
    /// ToTemporalTime for `toPlainDateTime`'s argument: undefined → midnight, a
    /// PlainTime/PlainDateTime instance → its time, an object with time fields →
    /// each `ToIntegerWithTruncation`'d + constrained, or an ISO string.
    fn pd_to_temporal_time(&mut self, v: NanBox) -> Result<temporal_iso::IsoTime, ExecError> {
        use crate::temporal_iso::{IsoTime, Overflow, parse_iso_time_string, regulate_iso_time};
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(IsoTime::default());
        }
        if self.is_object_value(v)
            && let Some(h) = v.as_handle().map(Handle::from_raw)
        {
            if let Some(td) = self.realm.temporal_at(h) {
                return match td.kind {
                    TemporalKind::PlainTime | TemporalKind::PlainDateTime => Ok(td.time),
                    // ToTemporalTime on a ZonedDateTime uses its wall-clock time
                    // (the time part of the instant in its own time zone).
                    TemporalKind::ZonedDateTime => {
                        let tz = td.tz.clone().unwrap_or_else(|| String::from("UTC"));
                        let epoch = td.epoch_ns;
                        Ok(super::temporal_zoneddatetime::local_of(&tz, epoch).1)
                    }
                    _ => Err(self.type_error("toPlainDateTime: not a time-like value")),
                };
            }
            // A property bag of time fields (ToTemporalTimeRecord); at least one
            // recognised field must be present. Fields are read in alphabetical
            // order (hour, microsecond, millisecond, minute, nanosecond, second),
            // through the proxy/getter-aware accessor so traps are observed. The
            // index is the position in `regulate_iso_time`'s argument list.
            let mut fields = [0_i64; 6];
            let mut any = false;
            for (k, i) in [
                ("hour", 0usize),
                ("microsecond", 4),
                ("millisecond", 3),
                ("minute", 1),
                ("nanosecond", 5),
                ("second", 2),
            ] {
                let pv = self.read_member(h, k)?;
                if !matches!(pv.unpack(), Unpacked::Undefined) {
                    any = true;
                    fields[i] = self.coerce_to_integer_or_infinity(pv)? as i64;
                }
            }
            if !any {
                return Err(self.type_error("toPlainDateTime: object has no time fields"));
            }
            return regulate_iso_time(
                fields[0],
                fields[1],
                fields[2],
                fields[3],
                fields[4],
                fields[5],
                Overflow::Constrain,
            )
            .ok_or_else(|| self.type_error("toPlainDateTime: invalid time"));
        }
        // Only a String parses to a time; any other primitive (number, boolean,
        // …) is a TypeError under ToTemporalTime, not a RangeError.
        let is_string = v
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.is_string_handle(h));
        if !is_string {
            return Err(self
                .type_error("toPlainDateTime: expected a time, time-like object, or ISO string"));
        }
        let s = self.coerce_to_string(v)?;
        // `ParseTemporalTimeString`: a bare `Z`/UTC designator or a date-only
        // string (no time component) is rejected — no implicit midnight.
        let Some(p) = parse_iso_time_string(&s) else {
            return Err(self.pd_range_error("toPlainDateTime: invalid time string"));
        };
        if p.z {
            return Err(self.pd_range_error("toPlainDateTime: UTC designator not allowed"));
        }
        p.time
            .ok_or_else(|| self.pd_range_error("toPlainDateTime: string has no time component"))
    }

    fn pd_new_kind(
        &mut self,
        kind: TemporalKind,
        date: IsoDate,
        time: temporal_iso::IsoTime,
    ) -> NanBox {
        self.pd_new_kind_cal(kind, date, time, "iso8601")
    }

    /// As [`pd_new_kind`](Self::pd_new_kind), carrying calendar id `cal`. The
    /// `toPlainDateTime`/`toPlainYearMonth`/`toPlainMonthDay` conversions must
    /// propagate the receiver's `[[Calendar]]` — without it a
    /// `buddhist`-calendared date produced an `iso8601` year-month, so
    /// `d.toPlainYearMonth().year` reported the ISO year instead of the
    /// calendar's.
    fn pd_new_kind_cal(
        &mut self,
        kind: TemporalKind,
        date: IsoDate,
        time: temporal_iso::IsoTime,
        cal: &str,
    ) -> NanBox {
        let data = TemporalData {
            kind,
            date,
            time,
            calendar: String::from(cal),
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(kind) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// Builds a fresh `Temporal.Duration` linked to the intrinsic prototype.
    fn pd_new_duration(&mut self, duration: DurationFields) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::Duration,
            duration,
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::Duration) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// `GetOptionsObject`: `undefined` -> no options; an object -> that object;
    /// anything else -> TypeError.
    fn pd_options(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        if self.is_object_value(v) {
            Ok(v.as_handle().map(Handle::from_raw))
        } else {
            Err(self.type_error("options must be an object or undefined"))
        }
    }

    /// Reads the `overflow` option (`"constrain"` default / `"reject"`).
    fn pd_overflow(&mut self, options: NanBox) -> Result<Overflow, ExecError> {
        let opts = self.pd_options(options)?;
        let s = self.get_string_option(
            opts,
            "overflow",
            &["constrain", "reject"],
            Some("constrain"),
        )?;
        Ok(if s.as_deref() == Some("reject") {
            Overflow::Reject
        } else {
            Overflow::Constrain
        })
    }

    /// `ToTemporalDate`: a PlainDate/PlainDateTime instance is copied; a string is
    /// parsed; a property bag is read + regulated. `options` (an options object or
    /// `undefined`) supplies the `overflow`, read *after* the item is converted.
    fn pd_to_temporal_date(
        &mut self,
        v: NanBox,
        options: NanBox,
    ) -> Result<(IsoDate, String), ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if let Some(td) = self.realm.temporal_at(h) {
                match td.kind {
                    TemporalKind::PlainDate | TemporalKind::PlainDateTime => {
                        // Options are still read + validated (observable) even
                        // though the date is copied verbatim.
                        self.pd_overflow(options)?;
                        return Ok((td.date, td.calendar.clone()));
                    }
                    TemporalKind::ZonedDateTime => {
                        // `ToTemporalDate` on a ZonedDateTime uses its internal
                        // slots (instant + time zone) directly — no field getters
                        // are observed — then reads the overflow option.
                        let tz = td.tz.clone().unwrap_or_else(|| String::from("UTC"));
                        let epoch = td.epoch_ns;
                        let cal = td.calendar.clone();
                        let (date, _time) = super::temporal_zoneddatetime::local_of(&tz, epoch);
                        self.pd_overflow(options)?;
                        return Ok((date, cal));
                    }
                    _ => {}
                }
            }
            if self.is_object_value(v) {
                return self.pd_from_fields(h, options);
            }
            // A string: parse first (may throw), then read the options.
            if let Some(s) = self.realm.string_value(h) {
                let (date, cal) = self.pd_from_string(&s)?;
                self.pd_overflow(options)?;
                return Ok((date, cal));
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.PlainDate"))
    }

    /// Parses an ISO string into a `(date, calendarId)` pair (rejecting a UTC
    /// designator and an invalid annotation set). The calendar id comes from the
    /// first `[u-ca=…]` annotation (canonicalized), defaulting to `"iso8601"`.
    fn pd_from_string(&mut self, s: &str) -> Result<(IsoDate, String), ExecError> {
        let Some(p) = parse_iso_datetime(s) else {
            return Err(self.pd_range_error(&alloc::format!("invalid PlainDate string '{s}'")));
        };
        let Some(date) = p.date else {
            return Err(self.pd_range_error(&alloc::format!("invalid PlainDate string '{s}'")));
        };
        if p.z {
            return Err(self.pd_range_error("a PlainDate string must not have a UTC designator"));
        }
        let cal = self.pd_validate_annotations(s)?;
        if !Self::pd_in_range(date) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        Ok((date, cal))
    }

    /// Validates the `[...]` annotation section of an ISO string: at most one
    /// time-zone annotation (before any key-value annotation), well-formed
    /// lowercase keys, no critical unknown annotation, at most one calendar
    /// annotation when any is critical, and an ISO first calendar annotation.
    fn pd_validate_annotations(&mut self, s: &str) -> Result<String, ExecError> {
        let Some(start) = s.find('[') else {
            return Ok(String::from("iso8601"));
        };
        let bytes = s.as_bytes();
        let mut i = start;
        let mut cal_count = 0_u32;
        let mut cal_critical = false;
        let mut tz_count = 0_u32;
        let mut seen_kv = false;
        let mut first_cal: Option<&str> = None;
        let bad = |slf: &mut Self| slf.pd_range_error("invalid annotation");
        while i < bytes.len() {
            if bytes[i] != b'[' {
                return Err(bad(self));
            }
            i += 1;
            let critical = i < bytes.len() && bytes[i] == b'!';
            if critical {
                i += 1;
            }
            let body_start = i;
            while i < bytes.len() && bytes[i] != b']' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(bad(self));
            }
            let body = &s[body_start..i];
            i += 1; // consume ']'
            if let Some(eq) = body.find('=') {
                let key = &body[..eq];
                let val = &body[eq + 1..];
                if !pd_is_annotation_key(key) {
                    return Err(bad(self));
                }
                seen_kv = true;
                if key == "u-ca" {
                    cal_count += 1;
                    cal_critical |= critical;
                    if first_cal.is_none() {
                        first_cal = Some(val);
                    }
                } else if critical {
                    return Err(bad(self));
                }
            } else {
                // A time-zone annotation; it must precede any key-value one.
                if seen_kv {
                    return Err(bad(self));
                }
                tz_count += 1;
            }
        }
        if tz_count > 1 || (cal_count > 1 && cal_critical) {
            return Err(bad(self));
        }
        match first_cal {
            Some(cal) => match tcal::canonicalize_calendar(cal) {
                Some(c) => Ok(String::from(c)),
                None => Err(bad(self)),
            },
            None => Ok(String::from("iso8601")),
        }
    }

    /// Reads a property-bag `{ year, month | monthCode, day, calendar? }`,
    /// converting each field in canonical order, then (after reading `overflow`
    /// from `options`) resolves and regulates it into a date.
    fn pd_from_fields(
        &mut self,
        h: Handle,
        options: NanBox,
    ) -> Result<(IsoDate, String), ExecError> {
        // The `calendar` field is read + canonicalized first.
        let cal_v = self.read_member(h, "calendar")?;
        let calendar = if matches!(cal_v.unpack(), Unpacked::Undefined) {
            String::from("iso8601")
        } else {
            self.pd_calendar_field(cal_v)?
        };
        if tcal::is_iso(&calendar) {
            let date = self.pd_from_fields_iso(h, options)?;
            Ok((date, calendar))
        } else {
            let date = self.pd_from_fields_cal(h, &calendar, options)?;
            Ok((date, calendar))
        }
    }

    /// The ISO-8601 property-bag path (unchanged behaviour): reads
    /// `day`/`month`/`monthCode`/`year` in alphabetical order after the calendar,
    /// then resolves + regulates into a date.
    fn pd_from_fields_iso(&mut self, h: Handle, options: NanBox) -> Result<IsoDate, ExecError> {
        let day_v = self.read_member(h, "day")?;
        let day = if matches!(day_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(day_v)?)
        };

        let month_v = self.read_member(h, "month")?;
        let month_num = if matches!(month_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(month_v)?)
        };

        // monthCode: ToPrimitive(string) then require a String; only its
        // *well-formedness* (syntax) is checked now — suitability is checked
        // after `overflow` is read.
        let month_code_v = self.read_member(h, "monthCode")?;
        let month_code = self.pd_read_month_code(month_code_v)?;

        let year_v = self.read_member(h, "year")?;
        let year = if matches!(year_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_integer_trunc(year_v)? as i64)
        };

        // Required-field presence (after all conversions, so a conversion error
        // above surfaces first).
        let (Some(year), Some(day)) = (year, day) else {
            return Err(self.type_error("PlainDate-like object is missing required fields"));
        };
        if month_num.is_none() && month_code.is_none() {
            return Err(self.type_error("PlainDate-like object is missing month/monthCode"));
        }

        // `overflow` is read before the algorithmic month/monthCode validation.
        let overflow = self.pd_overflow(options)?;
        let month = self.pd_resolve_month(month_num, month_code)?;
        self.pd_make_date(year, month, day, overflow)
    }

    /// The non-ISO property-bag path (`CalendarDateFromFields`): reads
    /// `day`/`era`/`eraYear`/`month`/`monthCode`/`year` and routes them through
    /// the calendar abstraction layer.
    fn pd_from_fields_cal(
        &mut self,
        h: Handle,
        calendar: &str,
        options: NanBox,
    ) -> Result<IsoDate, ExecError> {
        // Alphabetical field order after `calendar`: day, era, eraYear, month,
        // monthCode, year.
        let day_v = self.read_member(h, "day")?;
        let day = if matches!(day_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(day_v)?)
        };

        let era_v = self.read_member(h, "era")?;
        let era = if matches!(era_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.coerce_to_string(era_v)?)
        };

        let era_year_v = self.read_member(h, "eraYear")?;
        let era_year = if matches!(era_year_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_integer_trunc(era_year_v)? as i64)
        };

        let month_v = self.read_member(h, "month")?;
        let month = if matches!(month_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(month_v)?)
        };

        let month_code_v = self.read_member(h, "monthCode")?;
        let month_code = self.pd_read_month_code_str(month_code_v)?;

        let year_v = self.read_member(h, "year")?;
        let year = if matches!(year_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_integer_trunc(year_v)? as i64)
        };

        if day.is_none() {
            return Err(self.type_error("PlainDate-like object is missing 'day'"));
        }
        if month.is_none() && month_code.is_none() {
            return Err(self.type_error("PlainDate-like object is missing month/monthCode"));
        }

        let overflow = self.pd_overflow(options)?;
        let input = tcal::FieldsInput {
            era,
            era_year,
            year,
            month,
            month_code,
            day: day.unwrap(),
        };
        let date = self.pd_cal_fields_to_iso(calendar, &input, overflow)?;
        if !Self::pd_in_range(date) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        Ok(date)
    }

    /// Runs [`tcal::fields_to_iso`], mapping its error to the right exception.
    fn pd_cal_fields_to_iso(
        &mut self,
        calendar: &str,
        input: &tcal::FieldsInput,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        match tcal::fields_to_iso(calendar, input, overflow) {
            Ok(d) => Ok(d),
            Err(tcal::CalError::Range(m)) => Err(self.pd_range_error(&m)),
            Err(tcal::CalError::MissingFields(m)) => Err(self.type_error(&m)),
        }
    }

    /// `ToTemporalCalendarIdentifier` for a property-bag `calendar` field →
    /// canonical id. A calendared Temporal object contributes its id; a string is
    /// either a calendar id or a parseable ISO string whose annotation supplies
    /// one. A non-string is a TypeError; an unsupported id is a RangeError.
    fn pd_calendar_field(&mut self, v: NanBox) -> Result<String, ExecError> {
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.pd_canonicalize_calendar(&cal);
        }
        let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        if let Some(c) = tcal::canonicalize_calendar(&s) {
            return Ok(String::from(c));
        }
        // Otherwise it must be a valid ISO string carrying a calendar annotation.
        if parse_iso_datetime(&s).is_some()
            && let Ok(cal) = self.pd_validate_annotations(&s)
        {
            return Ok(cal);
        }
        Err(self.pd_range_error(&alloc::format!("invalid calendar identifier '{s}'")))
    }

    /// Reads a `monthCode` field as its raw well-formed string (for the non-ISO
    /// path, where suitability is judged by the calendar layer).
    fn pd_read_month_code_str(&mut self, v: NanBox) -> Result<Option<String>, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        let prim = self.coerce_primitive(v, "string")?;
        let Some(s) = prim
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("monthCode must be a string"));
        };
        // Well-formedness only (M + two digits + optional L).
        self.pd_monthcode_wellformed(&s)?;
        Ok(Some(s))
    }

    /// Reads a `monthCode` field: `undefined` -> `None`; otherwise
    /// ToPrimitive(string) which must yield a String, whose *well-formedness*
    /// (`M` + two digits + optional `L`) is validated -> `Some((number, is_leap))`.
    fn pd_read_month_code(&mut self, v: NanBox) -> Result<Option<(u8, bool)>, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        let prim = self.coerce_primitive(v, "string")?;
        let Some(s) = prim
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("monthCode must be a string"));
        };
        Ok(Some(self.pd_monthcode_wellformed(&s)?))
    }

    /// Resolves the month number from an optional `month` and optional decoded
    /// `monthCode`, checking monthCode suitability (1..12, non-leap for ISO) and
    /// agreement.
    fn pd_resolve_month(
        &mut self,
        month_num: Option<i64>,
        month_code: Option<(u8, bool)>,
    ) -> Result<i64, ExecError> {
        match (month_num, month_code) {
            (_, Some((n, is_leap))) => {
                if is_leap || !(1..=12).contains(&n) {
                    return Err(self.pd_range_error("monthCode is not valid for the ISO calendar"));
                }
                if let Some(m) = month_num
                    && m != i64::from(n)
                {
                    return Err(self.pd_range_error("month and monthCode disagree"));
                }
                Ok(i64::from(n))
            }
            (Some(m), None) => Ok(m),
            (None, None) => unreachable!(),
        }
    }

    /// Validates an ISO `monthCode` for *well-formedness* only (an `M`, two
    /// digits, then an optional leap `L`), returning `(number, is_leap)`. Bad
    /// syntax is a RangeError; range/suitability is checked by the caller.
    fn pd_monthcode_wellformed(&mut self, s: &str) -> Result<(u8, bool), ExecError> {
        let b = s.as_bytes();
        let ok = (b.len() == 3 || (b.len() == 4 && b[3] == b'L'))
            && b[0] == b'M'
            && b[1].is_ascii_digit()
            && b[2].is_ascii_digit();
        if !ok {
            return Err(self.pd_range_error(&alloc::format!("malformed monthCode '{s}'")));
        }
        let n = (b[1] - b'0') * 10 + (b[2] - b'0');
        Ok((n, b.len() == 4))
    }

    /// `add` / `subtract`: adds (or, when `negate`, subtracts) a Duration.
    ///
    /// The ISO-8601 calendar takes the shared ISO fast path (`AddISODate`); every
    /// other calendar routes through [`tcal::calendar_date_add`], which honours
    /// the calendar's variable month lengths and leap months (adding years/months
    /// in calendar terms, then weeks/days as a plain day offset).
    fn pd_add(
        &mut self,
        date: IsoDate,
        cal: &str,
        dur_arg: NanBox,
        options: NanBox,
        negate: bool,
    ) -> Result<NanBox, ExecError> {
        let mut d = self.pd_to_duration(dur_arg)?;
        if negate {
            d = DurationFields {
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
            };
        }
        let overflow = self.pd_overflow(options)?;
        // The sub-day time part contributes only its whole-day carry to a date.
        let extra_days = d.time_nanos() / temporal_iso::NS_PER_DAY;
        let days = (d.days + extra_days) as i64;
        if tcal::is_iso(cal) {
            // ISO fast path — byte-for-byte the original computation.
            // Guard against a year that would overflow the i32 in the shared
            // year/month balancing (huge `years`/`months` durations).
            let approx_year = i64::from(date.year)
                + d.years as i64
                + (i64::from(date.month) + d.months as i64 - 1).div_euclid(12);
            if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&approx_year) {
                return Err(self.pd_range_error("result is outside the representable range"));
            }
            return match add_iso_date(
                date,
                d.years as i64,
                d.months as i64,
                d.weeks as i64,
                days,
                overflow,
            ) {
                Some(result) if Self::pd_in_range(result) => Ok(self.pd_new_cal(result, cal)),
                _ => Err(self.pd_range_error("result is outside the representable range")),
            };
        }
        // Non-ISO: calendar-aware year/month addition through the calendar layer.
        let result = match tcal::calendar_date_add(
            cal,
            date,
            d.years as i64,
            d.months as i64,
            d.weeks as i64,
            days,
            overflow,
        ) {
            Ok(r) => r,
            Err(tcal::CalError::Range(m)) => return Err(self.pd_range_error(&m)),
            Err(tcal::CalError::MissingFields(m)) => return Err(self.type_error(&m)),
        };
        if !Self::pd_in_range(result) {
            return Err(self.pd_range_error("result is outside the representable range"));
        }
        Ok(self.pd_new_cal(result, cal))
    }

    /// `ToTemporalDuration`: a Duration instance, an ISO string, or a property
    /// bag of the ten duration fields.
    fn pd_to_duration(&mut self, v: NanBox) -> Result<DurationFields, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if let Some(td) = self.realm.temporal_at(h)
                && td.kind == TemporalKind::Duration
            {
                return Ok(td.duration);
            }
            if self.is_object_value(v) {
                return self.pd_duration_from_fields(h);
            }
            if let Some(s) = self.realm.string_value(h) {
                return match parse_iso_duration(&s) {
                    Some(d) => Ok(d),
                    None => {
                        Err(self.pd_range_error(&alloc::format!("invalid duration string '{s}'")))
                    }
                };
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.Duration"))
    }

    /// Reads a duration property bag (at least one recognised field required).
    fn pd_duration_from_fields(&mut self, h: Handle) -> Result<DurationFields, ExecError> {
        const FIELDS: [&str; 10] = [
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
        for f in FIELDS {
            let v = self.read_member(h, f)?;
            if matches!(v.unpack(), Unpacked::Undefined) {
                continue;
            }
            any = true;
            let n = self.pd_to_integer_if_integral(v)?;
            match f {
                "years" => d.years = n,
                "months" => d.months = n,
                "weeks" => d.weeks = n,
                "days" => d.days = n,
                "hours" => d.hours = n,
                "minutes" => d.minutes = n,
                "seconds" => d.seconds = n,
                "milliseconds" => d.milliseconds = n,
                "microseconds" => d.microseconds = n,
                _ => d.nanoseconds = n,
            }
        }
        if !any {
            return Err(self.type_error("duration-like object has no recognised fields"));
        }
        if !d.is_valid() {
            return Err(self.pd_range_error("duration fields have mixed signs"));
        }
        Ok(d)
    }

    /// `with`: returns a copy with `year`/`month`/`monthCode`/`day` replaced.
    fn pd_with(
        &mut self,
        date: IsoDate,
        cal: &str,
        fields: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        if !self.is_object_value(fields) {
            return Err(self.type_error("with() requires a fields object"));
        }
        let h = fields.as_handle().map(Handle::from_raw).unwrap();
        // `IsPartialTemporalObject`: a Temporal-branded object (PlainDate,
        // ZonedDateTime, …) is not a partial property bag → TypeError.
        if self.realm.temporal_at(h).is_some() {
            return Err(self.type_error("with() argument must be a plain object"));
        }
        // Reject a calendar / timeZone property (RejectTemporalLikeObject).
        for banned in ["calendar", "timeZone"] {
            let v = self.read_member(h, banned)?;
            if !matches!(v.unpack(), Unpacked::Undefined) {
                return Err(self.type_error(&alloc::format!(
                    "with() fields object must not have a '{banned}' property"
                )));
            }
        }
        if !tcal::is_iso(cal) {
            return self.pd_with_cal(date, cal, h, options);
        }
        let day_v = self.read_member(h, "day")?;
        let day = if matches!(day_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(day_v)?)
        };
        let month_v = self.read_member(h, "month")?;
        let month_num = if matches!(month_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(month_v)?)
        };
        let month_code_v = self.read_member(h, "monthCode")?;
        let month_code = self.pd_read_month_code(month_code_v)?;
        let year_v = self.read_member(h, "year")?;
        let year = if matches!(year_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_integer_trunc(year_v)? as i64)
        };

        if year.is_none() && day.is_none() && month_num.is_none() && month_code.is_none() {
            return Err(self.type_error("with() fields object has no recognised fields"));
        }

        let overflow = self.pd_overflow(options)?;
        // Merge with the receiver's existing fields.
        let month = if month_num.is_some() || month_code.is_some() {
            self.pd_resolve_month(month_num, month_code)?
        } else {
            i64::from(date.month)
        };
        let year = year.unwrap_or(i64::from(date.year));
        let day = day.unwrap_or(i64::from(date.day));

        let result = self.pd_make_date(year, month, day, overflow)?;
        Ok(self.pd_new(result))
    }

    /// The non-ISO `with` path: merges the provided fields over the receiver's
    /// existing calendar fields and re-derives the ISO date through the layer.
    fn pd_with_cal(
        &mut self,
        date: IsoDate,
        cal: &str,
        h: Handle,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        let existing = tcal::iso_to_fields(cal, date);

        let day_v = self.read_member(h, "day")?;
        let day = if matches!(day_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(day_v)?)
        };
        let era_v = self.read_member(h, "era")?;
        let era = if matches!(era_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.coerce_to_string(era_v)?)
        };
        let era_year_v = self.read_member(h, "eraYear")?;
        let era_year = if matches!(era_year_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_integer_trunc(era_year_v)? as i64)
        };
        let month_v = self.read_member(h, "month")?;
        let month = if matches!(month_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_positive_integer(month_v)?)
        };
        let month_code_v = self.read_member(h, "monthCode")?;
        let month_code = self.pd_read_month_code_str(month_code_v)?;
        let year_v = self.read_member(h, "year")?;
        let year = if matches!(year_v.unpack(), Unpacked::Undefined) {
            None
        } else {
            Some(self.pd_to_integer_trunc(year_v)? as i64)
        };

        if year.is_none()
            && day.is_none()
            && month.is_none()
            && month_code.is_none()
            && era.is_none()
            && era_year.is_none()
        {
            return Err(self.type_error("with() fields object has no recognised fields"));
        }

        let overflow = self.pd_overflow(options)?;
        // Merge: an explicit year (or era+eraYear) wins; otherwise keep the
        // receiver's year. Likewise for month (prefer monthCode to preserve leap
        // months) and day.
        // If the caller supplies *any* of the year group (year / era / eraYear),
        // pass exactly what they gave through to the layer, which validates the
        // combination (e.g. eraYear alone → TypeError). Only when none is present
        // do we fall back to the receiver's year.
        let (year, era, era_year) = if year.is_some() || era.is_some() || era_year.is_some() {
            (year, era, era_year)
        } else {
            (Some(existing.year), None, None)
        };
        let (month, month_code) = if month.is_some() || month_code.is_some() {
            (month, month_code)
        } else {
            (None, Some(existing.month_code.clone()))
        };
        let day = day.unwrap_or(existing.day);

        let input = tcal::FieldsInput {
            era,
            era_year,
            year,
            month,
            month_code,
            day,
        };
        let result = self.pd_cal_fields_to_iso(cal, &input, overflow)?;
        if !Self::pd_in_range(result) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        Ok(self.pd_new_cal(result, cal))
    }

    /// `withCalendar`: returns a copy of the date re-tagged with calendar `cal`.
    fn pd_with_calendar(&mut self, date: IsoDate, cal: NanBox) -> Result<NanBox, ExecError> {
        let id = if let Some(c) = self.temporal_object_calendar(cal) {
            self.pd_canonicalize_calendar(&c)?
        } else {
            let Some(s) = cal
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.string_value(h))
            else {
                return Err(self.type_error("calendar must be a string"));
            };
            // `ToTemporalCalendarSlotValue`: a bare builtin calendar id, or (via
            // `ParseTemporalCalendarString`) a valid annotated ISO date / date-time
            // / time string whose `[u-ca=…]` annotation supplies the id (default
            // `iso8601`). An unknown id or a malformed string → RangeError.
            if let Some(c) = tcal::canonicalize_calendar(&s) {
                String::from(c)
            } else if let Some(raw) = temporal_iso::parse_calendar_string(&s)
                && let Some(c) = tcal::canonicalize_calendar(&raw)
            {
                String::from(c)
            } else {
                return Err(
                    self.pd_range_error(&alloc::format!("invalid calendar identifier '{s}'"))
                );
            }
        };
        Ok(self.pd_new_cal(date, &id))
    }

    /// `until` (`since` when `negate`): the date difference as a Duration.
    fn pd_difference(
        &mut self,
        date: IsoDate,
        cal: &str,
        other: NanBox,
        options: NanBox,
        negate: bool,
    ) -> Result<NanBox, ExecError> {
        let (other_date, other_cal) = self.pd_to_temporal_date(other, NanBox::undefined())?;
        // A non-ISO receiver requires both dates to share the same calendar (the
        // ISO fast path keeps its original, calendar-agnostic behaviour).
        if !tcal::is_iso(cal) && other_cal != cal {
            return Err(self.pd_range_error(
                "cannot compute the difference between dates of different calendars",
            ));
        }
        let opts = self.pd_options(options)?;
        // `GetDifferenceSettings`: read and cast *all* options first, in the order
        // largestUnit, roundingIncrement, roundingMode, smallestUnit — before any
        // algorithmic (disallowed-unit / largest-vs-smallest) validation.
        let largest_raw = self.pd_unit_option(opts, "largestUnit", true)?;
        // `GetRoundingIncrementOption` (validated for every call; for the date
        // units used here there is no per-unit maximum, so any finite integer in
        // [1, 1e9] is accepted — an out-of-range/NaN value is a RangeError).
        let increment = self.pd_rounding_increment(opts)?;
        let mode = self.get_string_option(
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
            Some("trunc"),
        )?;
        let smallest_raw = self.pd_unit_option(opts, "smallestUnit", false)?;

        // Algorithmic validation (after all reads): a PlainDate difference admits
        // only date units (year/month/week/day); a time unit is a RangeError.
        if let Some(l) = largest_raw
            && (l as u8) > (Unit::Day as u8)
        {
            return Err(self.pd_range_error("largestUnit must be a date unit"));
        }
        if let Some(s) = smallest_raw
            && (s as u8) > (Unit::Day as u8)
        {
            return Err(self.pd_range_error("smallestUnit must be a date unit"));
        }
        let smallest = smallest_raw.unwrap_or(Unit::Day);
        // Default largestUnit ("auto") is the larger of Day and smallestUnit.
        let largest = largest_raw.unwrap_or(if (smallest as u8) < (Unit::Day as u8) {
            smallest
        } else {
            Unit::Day
        });
        if (largest as u8) > (smallest as u8) {
            return Err(self.pd_range_error("largestUnit must be larger than smallestUnit"));
        }

        // Compute in the forward (`until`) orientation and round with the
        // rounding mode negated for `since`; the whole difference is then negated
        // for `since` (matching `DifferenceTemporalPlainDate` +
        // `NegateRoundingMode`).
        let round_mode = pd_round_mode_from_str(mode.as_deref(), negate);
        let mut duration = if tcal::is_iso(cal) {
            self.pd_round_date_diff(date, other_date, largest, smallest, increment, round_mode)?
        } else {
            self.pd_round_date_diff_cal(
                cal, date, other_date, largest, smallest, increment, round_mode,
            )?
        };
        if negate {
            duration.years = -duration.years;
            duration.months = -duration.months;
            duration.weeks = -duration.weeks;
            duration.days = -duration.days;
        }
        Ok(self.pd_new_duration(duration))
    }

    /// `RoundDuration` for a date-only difference (ISO calendar). Computes the
    /// `from → to` difference in `largest` units, then rounds to `smallest`
    /// (`increment`, `mode`). For a `smallestUnit` coarser than a day the fraction
    /// is measured against the day-span of one increment step from an anchor that
    /// already carries the coarser components (`NudgeToCalendarUnit`).
    fn pd_round_date_diff(
        &mut self,
        from: IsoDate,
        to: IsoDate,
        largest: Unit,
        smallest: Unit,
        increment: i64,
        mode: crate::temporal_iso::RoundMode,
    ) -> Result<DurationFields, ExecError> {
        use crate::temporal_iso::{
            add_iso_date, iso_to_epoch_days, round_to_increment as round_inc,
        };
        let (years, months, weeks, days) = difference_iso_date(from, to, largest);
        if smallest == Unit::Day {
            let d = round_inc(i128::from(days), i128::from(increment.max(1)), mode);
            return Ok(DurationFields {
                years: i128::from(years),
                months: i128::from(months),
                weeks: i128::from(weeks),
                days: d,
                ..Default::default()
            });
        }
        // Keep components coarser than `smallest`; round the `smallest` count.
        let (keep_y, keep_m) = match smallest {
            Unit::Year => (0, 0),
            Unit::Month => (years, 0),
            _ => (years, months), // Week
        };
        let anchor = add_iso_date(from, keep_y, keep_m, 0, 0, Overflow::Constrain).unwrap_or(from);
        let anchor_e = iso_to_epoch_days(anchor);
        let to_e = iso_to_epoch_days(to);
        let sign = (to_e - anchor_e).signum();
        let mut out = DurationFields {
            years: i128::from(keep_y),
            months: i128::from(keep_m),
            ..Default::default()
        };
        if sign == 0 {
            return Ok(out);
        }
        let (sy, sm, sw, _) = difference_iso_date(anchor, to, smallest);
        let mut r1 = match smallest {
            Unit::Year => sy,
            Unit::Month => sm,
            _ => sw,
        };
        let unit_add = |count: i64| -> Option<IsoDate> {
            let (y, m, w) = match smallest {
                Unit::Year => (count, 0, 0),
                Unit::Month => (0, count, 0),
                _ => (0, 0, count),
            };
            add_iso_date(anchor, y, m, w, 0, Overflow::Constrain)
        };
        let step = |count: i64| -> i64 { iso_to_epoch_days(unit_add(count).unwrap_or(anchor)) };
        let beyond = |e: i64| if sign > 0 { e > to_e } else { e < to_e };
        for _ in 0..12 {
            if beyond(step(r1 + sign)) {
                break;
            }
            r1 += sign;
        }
        for _ in 0..12 {
            if beyond(step(r1)) {
                r1 -= sign;
            } else {
                break;
            }
        }
        // `NudgeToCalendarUnit` projects the ceil (`r2`) increment-multiple endpoint
        // via `CalendarDateAdd`; a date outside the representable ISO range throws a
        // RangeError. Only a coarse unit with a large `roundingIncrement` reaches it.
        let inc = increment.max(1);
        let r2 = (r1 / inc) * inc + inc * sign;
        if unit_add(r2).is_none() {
            return Err(self.pd_range_error("rounded date is outside the valid ISO range"));
        }
        let e1 = step(r1);
        let count = if e1 == to_e {
            // Even an exact-boundary count must still snap to `roundingIncrement`.
            round_inc(i128::from(r1), i128::from(increment.max(1)), mode) as i64
        } else {
            let e2 = step(r1 + sign);
            let den = i128::from((e2 - e1).abs().max(1));
            let num = i128::from((to_e - e1).abs());
            let x = i128::from(r1) * den + i128::from(sign) * num;
            (round_inc(x, i128::from(increment.max(1)) * den, mode) / den) as i64
        };
        match smallest {
            Unit::Year => out.years = i128::from(count),
            Unit::Month => {
                // `BubbleRelativeDuration`: a rounded month count can reach a whole
                // year (e.g. 12 months → 1 year), so re-express the rounded target
                // date as a difference in `largest` units, balancing months upward.
                let rd =
                    add_iso_date(anchor, 0, count, 0, 0, Overflow::Constrain).unwrap_or(anchor);
                let (by, bm, _bw, _bd) = difference_iso_date(from, rd, largest);
                out.years = i128::from(by);
                out.months = i128::from(bm);
            }
            _ => out.weeks = i128::from(count),
        }
        Ok(out)
    }

    /// The calendar-aware analogue of [`Self::pd_round_date_diff`] for a non-ISO
    /// calendar: the `from → to` difference is measured in the calendar's own
    /// year/month terms (via [`tcal::calendar_date_until`]) and the coarse-unit
    /// nudge steps with [`tcal::calendar_date_add`], so month lengths and leap
    /// months are honoured. The rounding structure mirrors the ISO path exactly.
    #[allow(clippy::too_many_arguments)]
    fn pd_round_date_diff_cal(
        &mut self,
        cal: &str,
        from: IsoDate,
        to: IsoDate,
        largest: Unit,
        smallest: Unit,
        increment: i64,
        mode: crate::temporal_iso::RoundMode,
    ) -> Result<DurationFields, ExecError> {
        use crate::temporal_iso::{iso_to_epoch_days, round_to_increment as round_inc};
        let parts = tcal::calendar_date_until(cal, from, to, largest);
        let (years, months, weeks, days) = (parts.years, parts.months, parts.weeks, parts.days);
        if smallest == Unit::Day {
            let d = round_inc(i128::from(days), i128::from(increment.max(1)), mode);
            return Ok(DurationFields {
                years: i128::from(years),
                months: i128::from(months),
                weeks: i128::from(weeks),
                days: d,
                ..Default::default()
            });
        }
        // Keep components coarser than `smallest`; round the `smallest` count.
        let (keep_y, keep_m) = match smallest {
            Unit::Year => (0, 0),
            Unit::Month => (years, 0),
            _ => (years, months), // Week
        };
        // Calendar-aware add relative to a base (Constrain); `None` when the result
        // falls outside the representable ISO range.
        let cadd_opt = |base: IsoDate, y: i64, m: i64, w: i64| -> Option<IsoDate> {
            tcal::calendar_date_add(cal, base, y, m, w, 0, Overflow::Constrain).ok()
        };
        let cadd = |base: IsoDate, y: i64, m: i64, w: i64| -> IsoDate {
            cadd_opt(base, y, m, w).unwrap_or(base)
        };
        let anchor = cadd(from, keep_y, keep_m, 0);
        let anchor_e = iso_to_epoch_days(anchor);
        let to_e = iso_to_epoch_days(to);
        let sign = (to_e - anchor_e).signum();
        let mut out = DurationFields {
            years: i128::from(keep_y),
            months: i128::from(keep_m),
            ..Default::default()
        };
        if sign == 0 {
            return Ok(out);
        }
        let sub = tcal::calendar_date_until(cal, anchor, to, smallest);
        let mut r1 = match smallest {
            Unit::Year => sub.years,
            Unit::Month => sub.months,
            _ => sub.weeks,
        };
        let unit_add = |count: i64| -> Option<IsoDate> {
            let (y, m, w) = match smallest {
                Unit::Year => (count, 0, 0),
                Unit::Month => (0, count, 0),
                _ => (0, 0, count),
            };
            cadd_opt(anchor, y, m, w)
        };
        let step = |count: i64| -> i64 { iso_to_epoch_days(unit_add(count).unwrap_or(anchor)) };
        let beyond = |e: i64| if sign > 0 { e > to_e } else { e < to_e };
        for _ in 0..14 {
            if beyond(step(r1 + sign)) {
                break;
            }
            r1 += sign;
        }
        for _ in 0..14 {
            if beyond(step(r1)) {
                r1 -= sign;
            } else {
                break;
            }
        }
        // `NudgeToCalendarUnit`: the ceil (`r2`) increment-multiple endpoint must be
        // representable, else a RangeError.
        let inc = increment.max(1);
        let r2 = (r1 / inc) * inc + inc * sign;
        if unit_add(r2).is_none() {
            return Err(self.pd_range_error("rounded date is outside the valid ISO range"));
        }
        let e1 = step(r1);
        let count = if e1 == to_e {
            // Even an exact-boundary count must still snap to `roundingIncrement`.
            round_inc(i128::from(r1), i128::from(increment.max(1)), mode) as i64
        } else {
            let e2 = step(r1 + sign);
            let den = i128::from((e2 - e1).abs().max(1));
            let num = i128::from((to_e - e1).abs());
            let x = i128::from(r1) * den + i128::from(sign) * num;
            (round_inc(x, i128::from(increment.max(1)) * den, mode) / den) as i64
        };
        match smallest {
            Unit::Year => out.years = i128::from(count),
            Unit::Month => {
                // `BubbleRelativeDuration` (calendar-aware): a rounded month count
                // can reach a whole calendar year; re-express the rounded target
                // date as a difference in `largest` units.
                let rd = cadd(anchor, 0, count, 0);
                let b = tcal::calendar_date_until(cal, from, rd, largest);
                out.years = i128::from(b.years);
                out.months = i128::from(b.months);
            }
            _ => out.weeks = i128::from(count),
        }
        Ok(out)
    }

    /// `GetRoundingIncrementOption`: default 1; `ToNumber` (Symbol/BigInt →
    /// TypeError), non-finite → RangeError, then truncate toward zero and require
    /// the result to lie in `[1, 1e9]`.
    fn pd_rounding_increment(&mut self, opts: Option<Handle>) -> Result<i64, ExecError> {
        let Some(h) = opts else { return Ok(1) };
        let v = self.read_member(h, "roundingIncrement")?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(1);
        }
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.pd_range_error("roundingIncrement must be a finite integer"));
        }
        let i = n.trunc() as i64;
        if !(1..=1_000_000_000).contains(&i) {
            return Err(self.pd_range_error("roundingIncrement out of range"));
        }
        Ok(i)
    }

    /// `GetTemporalUnit`: reads and casts a difference-unit option to a [`Unit`]
    /// (singular or plural spelling), accepting *any* temporal unit. `undefined`
    /// (or `"auto"` when allowed) -> `None`; an unrecognized string -> RangeError.
    /// Whether a *time* unit is legal for this operation is validated by the
    /// caller, after every option has been read.
    fn pd_unit_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
        allow_auto: bool,
    ) -> Result<Option<Unit>, ExecError> {
        let Some(s) = self.get_string_option(opts, prop, &[], None)? else {
            return Ok(None);
        };
        // `largestUnit: "auto"` means the default (→ `None`); it is invalid as a
        // smallestUnit.
        if allow_auto && s == "auto" {
            return Ok(None);
        }
        let u = match s.as_str() {
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
            _ => {
                return Err(
                    self.pd_range_error(&alloc::format!("invalid value '{s}' for option {prop}"))
                );
            }
        };
        Ok(Some(u))
    }

    /// `toString` honoring the `calendarName` option.
    fn pd_to_string(
        &mut self,
        date: IsoDate,
        cal: &str,
        options: NanBox,
    ) -> Result<String, ExecError> {
        let opts = self.pd_options(options)?;
        let calname = self
            .get_string_option(
                opts,
                "calendarName",
                &["auto", "always", "never", "critical"],
                Some("auto"),
            )?
            .unwrap_or_else(|| String::from("auto"));
        Ok(self.pd_format(date, cal, &calname))
    }

    /// Formats a date as `YYYY-MM-DD` with an optional `[u-ca=<id>]` calendar
    /// annotation (`MaybeFormatCalendarAnnotation`): `always`/`critical` always
    /// emit; `auto` emits only for a non-ISO calendar; `never` never emits.
    fn pd_format(&mut self, date: IsoDate, cal: &str, calendar_name: &str) -> String {
        let mut s = alloc::format!(
            "{}-{}-{}",
            format_iso_year(date.year),
            pad(u64::from(date.month), 2),
            pad(u64::from(date.day), 2),
        );
        match calendar_name {
            "always" => s.push_str(&alloc::format!("[u-ca={cal}]")),
            "critical" => s.push_str(&alloc::format!("[!u-ca={cal}]")),
            "auto" if !tcal::is_iso(cal) => s.push_str(&alloc::format!("[u-ca={cal}]")),
            _ => {}
        }
        s
    }
}

/// Maps a `roundingMode` option string (defaulting to `trunc`) to a [`RoundMode`],
/// applying `NegateRoundingMode` when `negate` is set (as `since` does before it
/// flips the sign of the whole difference).
fn pd_round_mode_from_str(mode: Option<&str>, negate: bool) -> crate::temporal_iso::RoundMode {
    use crate::temporal_iso::RoundMode;
    let m = match mode {
        Some("ceil") => RoundMode::Ceil,
        Some("floor") => RoundMode::Floor,
        Some("expand") => RoundMode::Expand,
        Some("halfCeil") => RoundMode::HalfCeil,
        Some("halfFloor") => RoundMode::HalfFloor,
        Some("halfExpand") => RoundMode::HalfExpand,
        Some("halfTrunc") => RoundMode::HalfTrunc,
        Some("halfEven") => RoundMode::HalfEven,
        _ => RoundMode::Trunc,
    };
    if !negate {
        return m;
    }
    match m {
        RoundMode::Ceil => RoundMode::Floor,
        RoundMode::Floor => RoundMode::Ceil,
        RoundMode::HalfCeil => RoundMode::HalfFloor,
        RoundMode::HalfFloor => RoundMode::HalfCeil,
        other => other,
    }
}

/// Whether `key` is a well-formed ISO-string annotation key (`AnnotationKey`): a
/// lowercase ASCII letter or `_` followed by lowercase letters, digits, `-`, `_`.
fn pd_is_annotation_key(key: &str) -> bool {
    let b = key.as_bytes();
    !b.is_empty()
        && (b[0].is_ascii_lowercase() || b[0] == b'_')
        && b.iter()
            .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-' || c == b'_')
}
