//! `Temporal.PlainMonthDay` — logic module. A fan-out unit: everything specific to
//! `PlainMonthDay` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    IsoDate, MAX_EPOCH_DAYS, MIN_EPOCH_DAYS, Overflow, TemporalData, TemporalKind, format_iso_year,
    iso_days_in_month, iso_to_epoch_days, pad, parse_iso_datetime,
};

/// Prototype method names installed on `Temporal.PlainMonthDay.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "with",
    "equals",
    "toPlainDate",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
];
/// Getter-accessor names installed on `Temporal.PlainMonthDay.prototype`.
pub(crate) const GETTERS: &[&str] = &["monthCode", "day", "calendarId"];

/// A well-formed monthCode split into its numeric month and leap flag.
struct MonthCode {
    number: u8,
    leap: bool,
}

impl<'a> Interp<'a> {
    /// `new Temporal.PlainMonthDay(...)`.
    pub(crate) fn plainmonthday_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        // Order of operations: month, day, calendar, then referenceISOYear.
        let month = self.pmd_to_integer_with_truncation(arg_or_undef(args, 0))?;
        let day = self.pmd_to_integer_with_truncation(arg_or_undef(args, 1))?;
        // Calendar (a String primitive, canonicalised to "iso8601").
        let cal = arg_or_undef(args, 2);
        if !cal.is_undefined() {
            let Some(s) = cal.as_handle().and_then(|r| {
                let h = Handle::from_raw(r);
                self.realm.string_value(h)
            }) else {
                return Err(self.type_error("PlainMonthDay: calendar must be a string"));
            };
            if !s.eq_ignore_ascii_case("iso8601") {
                return Err(self.pmd_range_error("PlainMonthDay: unsupported calendar"));
            }
        }
        // referenceISOYear (defaults to 1972, a leap year).
        let ref_year = arg_or_undef(args, 3);
        let year = if ref_year.is_undefined() {
            1972_i32
        } else {
            let y = self.pmd_to_integer_with_truncation(ref_year)?;
            saturating_year(y)
        };
        let (m, d) = self.pmd_validate_iso(year, month, day)?;
        let data = TemporalData {
            kind: TemporalKind::PlainMonthDay,
            date: IsoDate {
                year,
                month: m,
                day: d,
            },
            ..Default::default()
        };
        Ok(self.finish_temporal(data, new_target, callee))
    }

    /// A `Temporal.PlainMonthDay.prototype.<method>()` call.
    pub(crate) fn plainmonthday_method(
        &mut self,
        this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        match method {
            "with" => self.pmd_with(data, arg_or_undef(args, 0), arg_or_undef(args, 1)),
            "equals" => self.pmd_equals(data, arg_or_undef(args, 0)),
            "toPlainDate" => self.pmd_to_plain_date(data, arg_or_undef(args, 0)),
            "toString" => {
                let s = self.pmd_to_string(data, arg_or_undef(args, 0))?;
                Ok(self.new_str(&s))
            }
            "toJSON" => {
                let s = self.pmd_format(data, "auto");
                Ok(self.new_str(&s))
            }
            "toLocaleString" => {
                // A minimal locale-independent rendering (≈ toString with defaults).
                let s = self.pmd_format(data, "auto");
                Ok(self.new_str(&s))
            }
            "valueOf" => Err(self.type_error(
                "Called Temporal.PlainMonthDay.prototype.valueOf, use equals() to compare",
            )),
            _ => {
                let _ = this;
                Err(self.temporal_todo(&alloc::format!("PlainMonthDay.prototype.{method}")))
            }
        }
    }

    /// A `Temporal.PlainMonthDay.prototype.<getter>` read.
    pub(crate) fn plainmonthday_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        match name {
            "monthCode" => {
                let s = alloc::format!("M{:02}", data.date.month);
                Ok(self.new_str(&s))
            }
            "day" => Ok(NanBox::number(f64::from(data.date.day))),
            "calendarId" => Ok(self.new_str("iso8601")),
            _ => Err(self.temporal_todo(&alloc::format!("PlainMonthDay getter {name}"))),
        }
    }

    /// A `Temporal.PlainMonthDay.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn plainmonthday_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        match method {
            "from" => {
                let iso = self.pmd_to_temporal(arg_or_undef(args, 0), arg_or_undef(args, 1))?;
                Ok(Some(self.pmd_new(iso)))
            }
            _ => Ok(None),
        }
    }

    // --- helpers -------------------------------------------------------------

    /// `ToIntegerWithTruncation`: ToNumber, reject non-finite (RangeError),
    /// then truncate toward zero.
    fn pmd_to_integer_with_truncation(&mut self, v: NanBox) -> Result<f64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.pmd_range_error("PlainMonthDay: value must be a finite integer"));
        }
        Ok(n.trunc())
    }

    /// `ToPositiveIntegerWithTruncation`: like the above but rejects values `< 1`
    /// (used for the `month`/`day` fields, whose positivity is validated as the
    /// field is read — before any options are consulted).
    fn pmd_to_positive_integer(&mut self, v: NanBox) -> Result<f64, ExecError> {
        let n = self.pmd_to_integer_with_truncation(v)?;
        if n < 1.0 {
            return Err(self.pmd_range_error("PlainMonthDay: month/day must be positive"));
        }
        Ok(n)
    }

    fn pmd_range_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// `RejectISODate` + `ISODateWithinLimits` for the constructor path.
    fn pmd_validate_iso(&mut self, year: i32, month: f64, day: f64) -> Result<(u8, u8), ExecError> {
        if !(1.0..=12.0).contains(&month) || day < 1.0 {
            return Err(self.pmd_range_error("PlainMonthDay: month/day out of range"));
        }
        let m = month as u8;
        let dim = f64::from(iso_days_in_month(year, m));
        if day > dim {
            return Err(self.pmd_range_error("PlainMonthDay: day out of range"));
        }
        let d = day as u8;
        if !pmd_date_in_range(IsoDate {
            year,
            month: m,
            day: d,
        }) {
            return Err(self.pmd_range_error("PlainMonthDay: date outside supported range"));
        }
        Ok((m, d))
    }

    /// Builds a fresh intrinsic `Temporal.PlainMonthDay` (subclassing ignored for
    /// `from`/`with`/method results).
    fn pmd_new(&mut self, iso: IsoDate) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainMonthDay,
            date: iso,
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainMonthDay) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// `GetTemporalOverflowOption(GetOptionsObject(options))`.
    fn pmd_overflow(&mut self, options: NanBox) -> Result<Overflow, ExecError> {
        if options.is_undefined() {
            return Ok(Overflow::Constrain);
        }
        if !self.is_object_value(options) {
            return Err(self.type_error("PlainMonthDay: options must be an object"));
        }
        let h = Handle::from_raw(options.as_handle().unwrap());
        let v = self.read_member(h, "overflow")?;
        if v.is_undefined() {
            return Ok(Overflow::Constrain);
        }
        let s = self.coerce_to_string(v)?;
        match s.as_str() {
            "constrain" => Ok(Overflow::Constrain),
            "reject" => Ok(Overflow::Reject),
            _ => Err(self.pmd_range_error("PlainMonthDay: invalid overflow option")),
        }
    }

    /// `ToTemporalMonthDay(item, options)` → the resulting ISO date (reference
    /// year encoded in `IsoDate::year`).
    fn pmd_to_temporal(&mut self, item: NanBox, options: NanBox) -> Result<IsoDate, ExecError> {
        if self.is_object_value(item) {
            let h = Handle::from_raw(item.as_handle().unwrap());
            // A PlainMonthDay argument is copied verbatim (options still validated).
            if let Some(td) = self.realm.temporal_at(h)
                && td.kind == TemporalKind::PlainMonthDay
            {
                let iso = td.date;
                self.pmd_overflow(options)?;
                return Ok(iso);
            }
            // Property bag: read calendar, then the date fields, then overflow.
            self.pmd_read_calendar(h)?;
            self.pmd_fields_to_iso(h, options)
        } else if let Some(s) = item
            .as_handle()
            .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
        {
            let (m, d) = self.pmd_parse_string(&s)?;
            self.pmd_overflow(options)?;
            Ok(IsoDate {
                year: 1972,
                month: m,
                day: d,
            })
        } else {
            Err(self.type_error("PlainMonthDay: expected an object or ISO string"))
        }
    }

    /// Reads and validates a property bag's `calendar` field (default iso8601).
    fn pmd_read_calendar(&mut self, h: Handle) -> Result<(), ExecError> {
        let v = self.read_member(h, "calendar")?;
        if v.is_undefined() {
            return Ok(());
        }
        // A *calendared* Temporal object supplies its own (iso8601) calendar via
        // the fast path; non-calendared objects (`{}`, `Duration`, …) are invalid.
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.pmd_check_calendar_string(&cal);
        }
        if let Some(handle) = v.as_handle().map(Handle::from_raw)
            && let Some(s) = self.realm.string_value(handle)
        {
            return self.pmd_check_calendar_string(&s);
        }
        Err(self.type_error("PlainMonthDay: calendar must be a string"))
    }

    /// A property-bag / string calendar identifier must resolve to iso8601.
    fn pmd_check_calendar_string(&mut self, s: &str) -> Result<(), ExecError> {
        if s.eq_ignore_ascii_case("iso8601") {
            return Ok(());
        }
        // May be an ISO date string carrying a `[u-ca=...]` annotation.
        if let Some(p) = parse_iso_datetime(&clamp_leap_second(s)) {
            let cal = p.calendar.unwrap_or_else(|| String::from("iso8601"));
            if cal.eq_ignore_ascii_case("iso8601") {
                return Ok(());
            }
        }
        Err(self.pmd_range_error("PlainMonthDay: unsupported calendar"))
    }

    /// `PrepareCalendarFields` + `ISOMonthDayFromFields` for a property bag.
    fn pmd_fields_to_iso(&mut self, h: Handle, options: NanBox) -> Result<IsoDate, ExecError> {
        // Fields are read in sorted order: day, month, monthCode, year.
        let day_v = self.read_member(h, "day")?;
        if day_v.is_undefined() {
            return Err(self.type_error("PlainMonthDay: 'day' is required"));
        }
        let day = self.pmd_to_positive_integer(day_v)?;

        let month_v = self.read_member(h, "month")?;
        let month = if month_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_positive_integer(month_v)?)
        };

        let mc_v = self.read_member(h, "monthCode")?;
        let month_code = if mc_v.is_undefined() {
            None
        } else {
            let s = self.coerce_to_string(mc_v)?;
            // Syntax (well-formedness) is validated here, before `year`.
            Some(self.pmd_parse_month_code(&s)?)
        };

        let year_v = self.read_member(h, "year")?;
        let year = if year_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_integer_with_truncation(year_v)?)
        };

        let overflow = self.pmd_overflow(options)?;
        self.pmd_resolve(month, month_code, day, year, overflow)
    }

    /// Resolves month/monthCode agreement and suitability (`month`/`day`
    /// positivity is validated earlier, as each field is read).
    fn pmd_resolve_month(
        &mut self,
        month: Option<f64>,
        month_code: &Option<MonthCode>,
    ) -> Result<f64, ExecError> {
        let resolved_month = match month_code {
            Some(mc) => {
                // Suitability: iso8601 has no leap months, months are 1..=12.
                if mc.leap || !(1..=12).contains(&mc.number) {
                    return Err(self.pmd_range_error("PlainMonthDay: invalid monthCode"));
                }
                let n = f64::from(mc.number);
                if let Some(m) = month
                    && m != n
                {
                    return Err(self.pmd_range_error("PlainMonthDay: month and monthCode conflict"));
                }
                n
            }
            None => match month {
                Some(m) => m,
                None => {
                    return Err(
                        self.type_error("PlainMonthDay: 'month' or 'monthCode' is required")
                    );
                }
            },
        };
        Ok(resolved_month)
    }

    /// Resolves month/monthCode/day/year fields into an ISO date with reference
    /// year 1972 (using `year` only to apply the overflow option).
    fn pmd_resolve(
        &mut self,
        month: Option<f64>,
        month_code: Option<MonthCode>,
        day: f64,
        year: Option<f64>,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        let resolved_month = self.pmd_resolve_month(month, &month_code)?;
        // Reference year for overflow: the supplied year, else 1972.
        let ref_year = year.map_or(1972, saturating_year);
        let iso = self.pmd_regulate(ref_year, resolved_month, day, overflow)?;
        Ok(IsoDate {
            year: 1972,
            month: iso.month,
            day: iso.day,
        })
    }

    /// `RegulateISODate` producing month/day (year used only for days-in-month).
    fn pmd_regulate(
        &mut self,
        year: i32,
        month: f64,
        day: f64,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        let out = crate::temporal_iso::regulate_iso_date(year, month as i64, day as i64, overflow);
        out.ok_or_else(|| self.pmd_range_error("PlainMonthDay: month/day out of range"))
    }

    /// Parses a well-formed monthCode `M` DD `L?`. `Err` = syntax error.
    fn pmd_parse_month_code(&mut self, s: &str) -> Result<MonthCode, ExecError> {
        let b = s.as_bytes();
        let ok = (b.len() == 3 || b.len() == 4)
            && b[0] == b'M'
            && b[1].is_ascii_digit()
            && b[2].is_ascii_digit()
            && (b.len() == 3 || b[3] == b'L');
        if !ok {
            return Err(self.pmd_range_error("PlainMonthDay: ill-formed monthCode"));
        }
        Ok(MonthCode {
            number: (b[1] - b'0') * 10 + (b[2] - b'0'),
            leap: b.len() == 4,
        })
    }

    /// `ParseTemporalMonthDayString` → (month, day). Accepts a MonthDay string
    /// (`MM-DD`, `--MM-DD`, basic `MMDD`) or a calendar-date/date-time string.
    fn pmd_parse_string(&mut self, s: &str) -> Result<(u8, u8), ExecError> {
        self.pmd_parse_string_inner(s)
            .ok_or_else(|| self.pmd_range_error("PlainMonthDay: invalid ISO string"))
    }

    fn pmd_parse_string_inner(&self, s: &str) -> Option<(u8, u8)> {
        // Reject the non-ASCII minus sign (U+2212) outright.
        if s.as_bytes().windows(3).any(|w| w == [0xE2, 0x88, 0x92]) {
            return None;
        }
        // Split off trailing `[...]` annotations and validate them.
        let ann_start = s.find('[').unwrap_or(s.len());
        let core = &s[..ann_start];
        let ann = &s[ann_start..];
        if !pmd_validate_annotations(ann) {
            return None;
        }
        // No more than nine fractional-second digits are permitted.
        if has_excess_fraction(core) {
            return None;
        }
        // Try the MonthDay-specific grammar first.
        if let Some(md) = pmd_parse_md_core(core) {
            return Some(md);
        }
        // Otherwise a full calendar-date / date-time string (a leap second in
        // the time is ignored for a MonthDay, so clamp `:60` → `:59`).
        let parsed = parse_iso_datetime(&clamp_leap_second(core))?;
        let date = parsed.date?;
        // A UTC designator (`Z`) is not permitted for a PlainMonthDay.
        if parsed.z {
            return None;
        }
        if let Some(cal) = &parsed.calendar
            && !cal.eq_ignore_ascii_case("iso8601")
        {
            return None;
        }
        Some((date.month, date.day))
    }

    /// `Temporal.PlainMonthDay.prototype.with`.
    fn pmd_with(
        &mut self,
        data: &TemporalData,
        fields: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        if !self.is_object_value(fields) {
            return Err(self.type_error("PlainMonthDay.with: argument must be an object"));
        }
        let h = Handle::from_raw(fields.as_handle().unwrap());
        // `IsPartialTemporalObject`: a Temporal-branded object (PlainDate,
        // ZonedDateTime, …) is not a partial property bag → TypeError.
        if self.realm.temporal_at(h).is_some() {
            return Err(self.type_error("PlainMonthDay.with: argument must be a plain object"));
        }
        // Reject a bag that carries a calendar or timeZone.
        for key in ["calendar", "timeZone"] {
            let v = self.read_member(h, key)?;
            if !v.is_undefined() {
                return Err(self.type_error("PlainMonthDay.with: calendar/timeZone not allowed"));
            }
        }
        // Read the partial fields (at least one recognised field required).
        let mut any = false;
        let day_v = self.read_member(h, "day")?;
        let day = if day_v.is_undefined() {
            None
        } else {
            any = true;
            Some(self.pmd_to_positive_integer(day_v)?)
        };
        let month_v = self.read_member(h, "month")?;
        let month = if month_v.is_undefined() {
            None
        } else {
            any = true;
            Some(self.pmd_to_positive_integer(month_v)?)
        };
        let mc_v = self.read_member(h, "monthCode")?;
        let month_code = if mc_v.is_undefined() {
            None
        } else {
            any = true;
            let s = self.coerce_to_string(mc_v)?;
            Some(self.pmd_parse_month_code(&s)?)
        };
        let year_v = self.read_member(h, "year")?;
        let year = if year_v.is_undefined() {
            None
        } else {
            any = true;
            Some(self.pmd_to_integer_with_truncation(year_v)?)
        };
        if !any {
            return Err(self.type_error("PlainMonthDay.with: no recognised fields"));
        }

        // Merge with the receiver: base contributes monthCode + day. If the
        // partial supplies month or monthCode, the base monthCode is dropped.
        let base_day = f64::from(data.date.day);
        let (m_month, m_code) = if month.is_some() || month_code.is_some() {
            (month, month_code)
        } else {
            (
                None,
                Some(MonthCode {
                    number: data.date.month,
                    leap: false,
                }),
            )
        };
        let m_day = day.unwrap_or(base_day);
        // Options (overflow) are read before the monthCode-suitability
        // validation; month/day positivity was already checked at read time.
        let overflow = self.pmd_overflow(options)?;
        let iso = self.pmd_resolve(m_month, m_code, m_day, year, overflow)?;
        Ok(self.pmd_new(iso))
    }

    /// `Temporal.PlainMonthDay.prototype.equals`.
    fn pmd_equals(&mut self, data: &TemporalData, other: NanBox) -> Result<NanBox, ExecError> {
        let iso = self.pmd_to_temporal(other, NanBox::undefined())?;
        let eq =
            iso.year == data.date.year && iso.month == data.date.month && iso.day == data.date.day;
        Ok(NanBox::boolean(eq))
    }

    /// `Temporal.PlainMonthDay.prototype.toPlainDate`.
    fn pmd_to_plain_date(
        &mut self,
        data: &TemporalData,
        item: NanBox,
    ) -> Result<NanBox, ExecError> {
        if !self.is_object_value(item) {
            return Err(self.type_error("PlainMonthDay.toPlainDate: argument must be an object"));
        }
        let h = Handle::from_raw(item.as_handle().unwrap());
        let year_v = self.read_member(h, "year")?;
        if year_v.is_undefined() {
            return Err(self.type_error("PlainMonthDay.toPlainDate: 'year' is required"));
        }
        let year = saturating_year(self.pmd_to_integer_with_truncation(year_v)?);
        // Combine with the receiver's month/day (constrain).
        let iso = self.pmd_regulate(
            year,
            f64::from(data.date.month),
            f64::from(data.date.day),
            Overflow::Constrain,
        )?;
        let iso = IsoDate {
            year,
            month: iso.month,
            day: iso.day,
        };
        if !pmd_date_in_range(iso) {
            return Err(self.pmd_range_error("PlainMonthDay.toPlainDate: date out of range"));
        }
        let date_data = TemporalData {
            kind: TemporalKind::PlainDate,
            date: iso,
            ..Default::default()
        };
        let ph = self.realm.new_temporal(date_data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainDate) {
            self.realm.set_native_proto(ph, p);
        }
        Ok(NanBox::handle(ph.to_raw()))
    }

    /// `Temporal.PlainMonthDay.prototype.toString` with its `calendarName` option.
    fn pmd_to_string(&mut self, data: &TemporalData, options: NanBox) -> Result<String, ExecError> {
        let show = self.pmd_show_calendar(options)?;
        Ok(self.pmd_format(data, &show))
    }

    /// Reads the `calendarName` show-option (auto/always/never/critical).
    fn pmd_show_calendar(&mut self, options: NanBox) -> Result<String, ExecError> {
        if options.is_undefined() {
            return Ok(String::from("auto"));
        }
        if !self.is_object_value(options) {
            return Err(self.type_error("PlainMonthDay.toString: options must be an object"));
        }
        let h = Handle::from_raw(options.as_handle().unwrap());
        let v = self.read_member(h, "calendarName")?;
        if v.is_undefined() {
            return Ok(String::from("auto"));
        }
        let s = self.coerce_to_string(v)?;
        match s.as_str() {
            "auto" | "always" | "never" | "critical" => Ok(s),
            _ => Err(self.pmd_range_error("PlainMonthDay.toString: invalid calendarName")),
        }
    }

    /// `TemporalMonthDayToString`.
    fn pmd_format(&self, data: &TemporalData, show: &str) -> String {
        let mut result = alloc::format!(
            "{}-{}",
            pad(u64::from(data.date.month), 2),
            pad(u64::from(data.date.day), 2)
        );
        let force_year = show == "always" || show == "critical";
        if force_year {
            result = alloc::format!("{}-{result}", format_iso_year(data.date.year));
        }
        let annotation = match show {
            "always" => "[u-ca=iso8601]",
            "critical" => "[!u-ca=iso8601]",
            _ => "",
        };
        result.push_str(annotation);
        result
    }
}

/// The nth argument, or `undefined`.
fn arg_or_undef(args: &[NanBox], i: usize) -> NanBox {
    args.get(i).copied().unwrap_or_else(NanBox::undefined)
}

/// Whether `date` is within the ISO plain-date range (noon-based bounds, no
/// time-of-day slop): `epoch_days ∈ [MIN_EPOCH_DAYS, MAX_EPOCH_DAYS]`.
fn pmd_date_in_range(date: IsoDate) -> bool {
    // `ISODateWithinLimits`: a `PlainDate` may sit one day below the minimum
    // instant date (to leave room for a time-of-day), i.e. `[MIN-1, MAX]`.
    (MIN_EPOCH_DAYS - 1..=MAX_EPOCH_DAYS).contains(&iso_to_epoch_days(date))
}

/// Clamps a leap-second `:60` to `:59` (the time is irrelevant to a MonthDay,
/// which only needs the calendar date).
fn clamp_leap_second(s: &str) -> String {
    s.replace(":60", ":59")
}

/// Whether any `.`/`,` fractional group in `core` has more than nine digits.
fn has_excess_fraction(core: &str) -> bool {
    let b = core.as_bytes();
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

/// Clamps a (possibly enormous) integral f64 year into `i32` range; the value is
/// only ever used to determine days-in-month, so saturation is harmless.
fn saturating_year(y: f64) -> i32 {
    if y >= f64::from(i32::MAX) {
        i32::MAX
    } else if y <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        y as i32
    }
}

/// Parses the MonthDay-specific grammar: `--`? `MM` `-`? `DD` (nothing else).
fn pmd_parse_md_core(core: &str) -> Option<(u8, u8)> {
    let b = core.as_bytes();
    let mut i = 0;
    if b.len() >= 2 && b[0] == b'-' && b[1] == b'-' {
        i = 2;
    }
    let month = two_digits(b, i)?;
    i += 2;
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    let day = two_digits(b, i)?;
    i += 2;
    if i != b.len() {
        return None;
    }
    // Reject leap-year-independent out-of-range values here.
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some((month, day))
}

fn two_digits(b: &[u8], i: usize) -> Option<u8> {
    let d0 = b.get(i)?;
    let d1 = b.get(i + 1)?;
    if !d0.is_ascii_digit() || !d1.is_ascii_digit() {
        return None;
    }
    Some((d0 - b'0') * 10 + (d1 - b'0'))
}

/// Validates a run of trailing `[...]` annotations. Rejects: malformed brackets,
/// an uppercase annotation key, a non-iso used calendar, an unknown *critical*
/// annotation, or more than one calendar annotation when any is critical.
fn pmd_validate_annotations(ann: &str) -> bool {
    let bytes = ann.as_bytes();
    let mut i = 0;
    let mut calendar_count = 0usize;
    let mut calendar_critical = false;
    let mut used_calendar: Option<&str> = None;
    let mut timezone_count = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            return false;
        }
        i += 1;
        let critical = bytes.get(i) == Some(&b'!');
        if critical {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i] != b']' {
            i += 1;
        }
        if i >= bytes.len() {
            return false; // unterminated
        }
        let inner = &ann[start..i];
        i += 1; // consume ']'
        if let Some(eq) = inner.find('=') {
            let key = &inner[..eq];
            let value = &inner[eq + 1..];
            if key.is_empty()
                || !key
                    .bytes()
                    .all(|c| matches!(c, b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_'))
            {
                return false;
            }
            if key == "u-ca" {
                calendar_count += 1;
                calendar_critical |= critical;
                if used_calendar.is_none() {
                    used_calendar = Some(value);
                }
            } else if critical {
                return false; // unknown critical annotation
            }
        } else {
            // A time-zone annotation (e.g. `[UTC]`, `[Asia/Tokyo]`): at most one.
            if inner.is_empty() {
                return false;
            }
            timezone_count += 1;
        }
    }
    if timezone_count > 1 {
        return false;
    }
    if calendar_count > 1 && calendar_critical {
        return false;
    }
    if let Some(cal) = used_calendar
        && !cal.eq_ignore_ascii_case("iso8601")
    {
        return false;
    }
    true
}
