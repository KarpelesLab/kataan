//! `Temporal.PlainDate` — logic module. A fan-out unit: everything specific to
//! `PlainDate` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::*;
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
        // canonicalize (only "iso8601", ASCII-case-insensitively).
        self.pd_validate_calendar(arg(3))?;
        let date = self.pd_reject_iso_date(y, m, d)?;
        let data = TemporalData {
            kind: TemporalKind::PlainDate,
            date,
            ..Default::default()
        };
        Ok(self.finish_temporal(data, new_target, callee))
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
        match method {
            "add" => self.pd_add(data.date, arg(0), arg(1), false),
            "subtract" => self.pd_add(data.date, arg(0), arg(1), true),
            "with" => self.pd_with(data.date, arg(0), arg(1)),
            "withCalendar" => self.pd_with_calendar(data.date, arg(0)),
            "until" => self.pd_difference(data.date, arg(0), arg(1), false),
            "since" => self.pd_difference(data.date, arg(0), arg(1), true),
            "equals" => {
                let other = self.pd_to_temporal_date(arg(0), NanBox::undefined())?;
                Ok(NanBox::boolean(other == data.date))
            }
            "toPlainDateTime" => {
                let time = self.pd_to_temporal_time(arg(0))?;
                Ok(self.pd_new_kind(TemporalKind::PlainDateTime, data.date, time))
            }
            "toPlainYearMonth" => Ok(self.pd_new_kind(
                TemporalKind::PlainYearMonth,
                data.date,
                temporal_iso::IsoTime::default(),
            )),
            "toPlainMonthDay" => Ok(self.pd_new_kind(
                TemporalKind::PlainMonthDay,
                data.date,
                temporal_iso::IsoTime::default(),
            )),
            "toString" => {
                let s = self.pd_to_string(data.date, arg(0))?;
                Ok(self.new_str(&s))
            }
            "toJSON" | "toLocaleString" => {
                let s = self.pd_format(data.date, "auto");
                Ok(self.new_str(&s))
            }
            "valueOf" => Err(self.type_error(
                "Called Temporal.PlainDate.prototype.valueOf, use compare() or equals() instead",
            )),
            "toZonedDateTime" => {
                // `date.toZonedDateTime(timeZone | { timeZone, plainTime })` → the
                // instant of the (date, plainTime|midnight) wall time in the zone.
                let item = arg(0);
                let tz = self.temporal_tz_arg(item)?;
                let mut time = crate::temporal_iso::IsoTime::default();
                if self.is_object_value(item)
                    && let Some(h) = item.as_handle().map(Handle::from_raw)
                {
                    let ptv = self
                        .realm
                        .get_property(h, "plainTime")
                        .unwrap_or(NanBox::undefined());
                    if let Some(pt) = ptv
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|hh| self.realm.temporal_at(hh))
                        && pt.kind == crate::temporal_iso::TemporalKind::PlainTime
                    {
                        time = pt.time;
                    }
                }
                let local_ns = crate::temporal_iso::iso_to_epoch_days(data.date) as i128
                    * crate::temporal_iso::NS_PER_DAY
                    + crate::temporal_iso::time_to_nanos(time);
                let offset = self.temporal_tz_offset_ns(&tz, local_ns).unwrap_or(0);
                Ok(self.build_temporal(crate::temporal_iso::TemporalData {
                    kind: crate::temporal_iso::TemporalKind::ZonedDateTime,
                    epoch_ns: local_ns - offset,
                    tz: Some(tz),
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
        Ok(match name {
            "calendarId" => self.new_str("iso8601"),
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
            "daysInMonth" => NanBox::number(f64::from(iso_days_in_month(date.year, date.month))),
            "daysInYear" => NanBox::number(f64::from(iso_days_in_year(date.year))),
            "monthsInYear" => NanBox::number(12.0),
            "inLeapYear" => NanBox::boolean(is_leap_year(date.year)),
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
                let date = self.pd_to_temporal_date(arg(0), arg(1))?;
                Ok(Some(self.pd_new(date)))
            }
            "compare" => {
                let a = self.pd_to_temporal_date(arg(0), NanBox::undefined())?;
                let b = self.pd_to_temporal_date(arg(1), NanBox::undefined())?;
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

    /// Validates a constructor calendar argument: `undefined` (or an "iso8601"
    /// string, case-insensitively) is accepted; any other string is a RangeError
    /// and a non-string is a TypeError.
    fn pd_validate_calendar(&mut self, v: NanBox) -> Result<(), ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(());
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

    /// A calendar identifier string is valid only if it is `"iso8601"`
    /// (ASCII-case-insensitively). Anything else is a RangeError.
    fn pd_canonicalize_calendar(&mut self, s: &str) -> Result<(), ExecError> {
        if s.is_ascii() && s.eq_ignore_ascii_case("iso8601") {
            Ok(())
        } else {
            Err(self.pd_range_error(&alloc::format!("invalid calendar identifier '{s}'")))
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
    fn pd_to_integer_if_integral(&mut self, v: NanBox) -> Result<i64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() || n.fract() != 0.0 {
            return Err(self.pd_range_error("duration field must be an integer"));
        }
        Ok(n as i64)
    }

    /// Builds a fresh `Temporal.PlainDate` linked to the intrinsic prototype.
    fn pd_new(&mut self, date: IsoDate) -> NanBox {
        self.pd_new_kind(
            TemporalKind::PlainDate,
            date,
            temporal_iso::IsoTime::default(),
        )
    }

    /// Builds a fresh Temporal instance of a date/time kind linked to that kind's
    /// intrinsic prototype (used for PlainDate and the `toPlain*` conversions).
    /// ToTemporalTime for `toPlainDateTime`'s argument: undefined → midnight, a
    /// PlainTime/PlainDateTime instance → its time, an object with time fields →
    /// each `ToIntegerWithTruncation`'d + constrained, or an ISO string.
    fn pd_to_temporal_time(&mut self, v: NanBox) -> Result<temporal_iso::IsoTime, ExecError> {
        use crate::temporal_iso::{IsoTime, Overflow, parse_iso_datetime, regulate_iso_time};
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(IsoTime::default());
        }
        if self.is_object_value(v)
            && let Some(h) = v.as_handle().map(Handle::from_raw)
        {
            if let Some(td) = self.realm.temporal_at(h) {
                return match td.kind {
                    TemporalKind::PlainTime | TemporalKind::PlainDateTime => Ok(td.time),
                    _ => Err(self.type_error("toPlainDateTime: not a time-like value")),
                };
            }
            // A property bag of time fields (ToTemporalTimeRecord); at least one
            // recognised field must be present.
            let mut fields = [0_i64; 6];
            let mut any = false;
            for (i, k) in [
                "hour",
                "minute",
                "second",
                "millisecond",
                "microsecond",
                "nanosecond",
            ]
            .into_iter()
            .enumerate()
            {
                let pv = self.realm.get_property(h, k).unwrap_or(NanBox::undefined());
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
            .is_some_and(|h| self.realm.string_value(h).is_some());
        if !is_string {
            return Err(self
                .type_error("toPlainDateTime: expected a time, time-like object, or ISO string"));
        }
        let s = self.coerce_to_string(v)?;
        parse_iso_datetime(&s)
            .and_then(|p| p.time.or(Some(IsoTime::default())))
            .ok_or_else(|| self.pd_range_error("toPlainDateTime: invalid time string"))
    }

    fn pd_new_kind(
        &mut self,
        kind: TemporalKind,
        date: IsoDate,
        time: temporal_iso::IsoTime,
    ) -> NanBox {
        let data = TemporalData {
            kind,
            date,
            time,
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
    fn pd_to_temporal_date(&mut self, v: NanBox, options: NanBox) -> Result<IsoDate, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if let Some(td) = self.realm.temporal_at(h) {
                match td.kind {
                    TemporalKind::PlainDate | TemporalKind::PlainDateTime => {
                        // Options are still read + validated (observable) even
                        // though the date is copied verbatim.
                        self.pd_overflow(options)?;
                        return Ok(td.date);
                    }
                    _ => {}
                }
            }
            if self.is_object_value(v) {
                return self.pd_from_fields(h, options);
            }
            // A string: parse first (may throw), then read the options.
            if let Some(s) = self.realm.string_value(h) {
                let date = self.pd_from_string(&s)?;
                self.pd_overflow(options)?;
                return Ok(date);
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.PlainDate"))
    }

    /// Parses an ISO string into a date (rejecting a UTC designator and an
    /// invalid annotation set).
    fn pd_from_string(&mut self, s: &str) -> Result<IsoDate, ExecError> {
        let Some(p) = parse_iso_datetime(s) else {
            return Err(self.pd_range_error(&alloc::format!("invalid PlainDate string '{s}'")));
        };
        let Some(date) = p.date else {
            return Err(self.pd_range_error(&alloc::format!("invalid PlainDate string '{s}'")));
        };
        if p.z {
            return Err(self.pd_range_error("a PlainDate string must not have a UTC designator"));
        }
        self.pd_validate_annotations(s)?;
        if !Self::pd_in_range(date) {
            return Err(self.pd_range_error("date is outside the representable range"));
        }
        Ok(date)
    }

    /// Validates the `[...]` annotation section of an ISO string: at most one
    /// time-zone annotation (before any key-value annotation), well-formed
    /// lowercase keys, no critical unknown annotation, at most one calendar
    /// annotation when any is critical, and an ISO first calendar annotation.
    fn pd_validate_annotations(&mut self, s: &str) -> Result<(), ExecError> {
        let Some(start) = s.find('[') else {
            return Ok(());
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
        if let Some(cal) = first_cal
            && !(cal.is_ascii() && cal.eq_ignore_ascii_case("iso8601"))
        {
            return Err(bad(self));
        }
        Ok(())
    }

    /// Reads a property-bag `{ year, month | monthCode, day, calendar? }`,
    /// converting each field in canonical order, then (after reading `overflow`
    /// from `options`) resolves and regulates it into a date.
    fn pd_from_fields(&mut self, h: Handle, options: NanBox) -> Result<IsoDate, ExecError> {
        // Fields are read + converted in alphabetical order: calendar, day,
        // month, monthCode, year.
        let cal_v = self.read_member(h, "calendar")?;
        if !matches!(cal_v.unpack(), Unpacked::Undefined) {
            self.pd_validate_calendar_field(cal_v)?;
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

    /// `ToTemporalCalendarIdentifier` for a property-bag `calendar` field: a
    /// string that is either the `"iso8601"` id or a parseable ISO date/time
    /// string whose (defaulted) calendar is ISO. A non-string is a TypeError.
    fn pd_validate_calendar_field(&mut self, v: NanBox) -> Result<(), ExecError> {
        let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        if s.is_ascii() && s.eq_ignore_ascii_case("iso8601") {
            return Ok(());
        }
        // Otherwise it must be a valid ISO string carrying (at most) an ISO
        // calendar annotation.
        if let Some(p) = parse_iso_datetime(&s)
            && p.calendar
                .as_ref()
                .is_none_or(|c| c.is_ascii() && c.eq_ignore_ascii_case("iso8601"))
            && self.pd_validate_annotations(&s).is_ok()
        {
            return Ok(());
        }
        Err(self.pd_range_error(&alloc::format!("invalid calendar identifier '{s}'")))
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
    fn pd_add(
        &mut self,
        date: IsoDate,
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
        // Guard against a year that would overflow the i32 in the shared
        // year/month balancing (huge `years`/`months` durations).
        let approx_year =
            i64::from(date.year) + d.years + (i64::from(date.month) + d.months - 1).div_euclid(12);
        if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&approx_year) {
            return Err(self.pd_range_error("result is outside the representable range"));
        }
        // The sub-day time part contributes only its whole-day carry to a date.
        let extra_days = (d.time_nanos() / temporal_iso::NS_PER_DAY) as i64;
        let days = d.days + extra_days;
        match add_iso_date(date, d.years, d.months, d.weeks, days, overflow) {
            Some(result) if Self::pd_in_range(result) => Ok(self.pd_new(result)),
            _ => Err(self.pd_range_error("result is outside the representable range")),
        }
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
        fields: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        if !self.is_object_value(fields) {
            return Err(self.type_error("with() requires a fields object"));
        }
        let h = fields.as_handle().map(Handle::from_raw).unwrap();
        // Reject a calendar / timeZone property (RejectTemporalLikeObject).
        for banned in ["calendar", "timeZone"] {
            let v = self.read_member(h, banned)?;
            if !matches!(v.unpack(), Unpacked::Undefined) {
                return Err(self.type_error(&alloc::format!(
                    "with() fields object must not have a '{banned}' property"
                )));
            }
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

    /// `withCalendar`: only the ISO calendar is supported.
    fn pd_with_calendar(&mut self, date: IsoDate, cal: NanBox) -> Result<NanBox, ExecError> {
        let Some(s) = cal
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        self.pd_canonicalize_calendar(&s)?;
        Ok(self.pd_new(date))
    }

    /// `until` (`since` when `negate`): the date difference as a Duration.
    fn pd_difference(
        &mut self,
        date: IsoDate,
        other: NanBox,
        options: NanBox,
        negate: bool,
    ) -> Result<NanBox, ExecError> {
        let other_date = self.pd_to_temporal_date(other, NanBox::undefined())?;
        let opts = self.pd_options(options)?;
        let smallest = self.pd_unit_option(opts, "smallestUnit")?;
        let largest_raw = self.pd_unit_option(opts, "largestUnit")?;
        // roundingMode / roundingIncrement are validated but only the default
        // (trunc, 1) is fully supported here.
        let _ = self.get_string_option(
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

        let smallest = smallest.unwrap_or(Unit::Day);
        // Default largestUnit ("auto") is the larger of Day and smallestUnit.
        let largest = largest_raw.unwrap_or(if (smallest as u8) < (Unit::Day as u8) {
            smallest
        } else {
            Unit::Day
        });
        if (largest as u8) > (smallest as u8) {
            return Err(self.pd_range_error("largestUnit must be larger than smallestUnit"));
        }

        let (from, to) = if negate {
            (other_date, date)
        } else {
            (date, other_date)
        };
        let (years, months, weeks, days) = difference_iso_date(from, to, largest);
        let duration = DurationFields {
            years,
            months,
            weeks,
            days,
            ..Default::default()
        };
        Ok(self.pd_new_duration(duration))
    }

    /// Reads a date-difference unit option (`years`/`months`/`weeks`/`days`,
    /// plural or singular). `undefined` -> `None`; a time unit or unknown value
    /// -> RangeError.
    fn pd_unit_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
    ) -> Result<Option<Unit>, ExecError> {
        let Some(s) = self.get_string_option(opts, prop, &[], None)? else {
            return Ok(None);
        };
        let u = match s.as_str() {
            "year" | "years" => Unit::Year,
            "month" | "months" => Unit::Month,
            "week" | "weeks" => Unit::Week,
            "day" | "days" => Unit::Day,
            _ => {
                return Err(
                    self.pd_range_error(&alloc::format!("invalid value '{s}' for option {prop}"))
                );
            }
        };
        Ok(Some(u))
    }

    /// `toString` honoring the `calendarName` option.
    fn pd_to_string(&mut self, date: IsoDate, options: NanBox) -> Result<String, ExecError> {
        let opts = self.pd_options(options)?;
        let calname = self
            .get_string_option(
                opts,
                "calendarName",
                &["auto", "always", "never", "critical"],
                Some("auto"),
            )?
            .unwrap_or_else(|| String::from("auto"));
        Ok(self.pd_format(date, &calname))
    }

    /// Formats a date as `YYYY-MM-DD` with an optional calendar annotation.
    fn pd_format(&mut self, date: IsoDate, calendar_name: &str) -> String {
        let mut s = alloc::format!(
            "{}-{}-{}",
            format_iso_year(date.year),
            pad(u64::from(date.month), 2),
            pad(u64::from(date.day), 2),
        );
        match calendar_name {
            "always" => s.push_str("[u-ca=iso8601]"),
            "critical" => s.push_str("[!u-ca=iso8601]"),
            _ => {}
        }
        s
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
