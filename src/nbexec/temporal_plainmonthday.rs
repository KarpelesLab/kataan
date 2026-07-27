//! `Temporal.PlainMonthDay` — logic module. A fan-out unit: everything specific to
//! `PlainMonthDay` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
//!
//! A `PlainMonthDay` stores an ISO *reference date* (a reference year + month +
//! day) that identifies a recurring month/day in its `[[Calendar]]`. For the ISO
//! calendar the reference year is a fixed 1972 (a leap year). For a non-ISO
//! calendar the reference date is derived through the shared calendar-abstraction
//! layer ([`super::temporal_calendar`]): the (monthCode, day) is resolved in a
//! calendar-appropriate reference year (the latest ISO date on/before 1972-12-31
//! in which it occurs, else the earliest afterwards), so leap months and
//! 30-day-only months anchor to a real ISO date.
use super::temporal_calendar as tcal;
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
        // Calendar (a String primitive, canonicalised across the full id set).
        let calendar = self.pmd_ctor_calendar(arg_or_undef(args, 2))?;
        // referenceISOYear (defaults to 1972, a leap year). The month/day/year
        // are ISO values (the constructor takes a raw ISO reference date).
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
            calendar,
            ..Default::default()
        };
        self.finish_temporal(data, new_target, callee)
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
        let cal = data.calendar.as_str();
        if name == "calendarId" {
            return Ok(self.new_str(cal));
        }
        // ISO-8601 fast path — byte-for-byte the original computation.
        if tcal::is_iso(cal) {
            return match name {
                "monthCode" => {
                    let s = alloc::format!("M{:02}", data.date.month);
                    Ok(self.new_str(&s))
                }
                "day" => Ok(NanBox::number(f64::from(data.date.day))),
                _ => Err(self.temporal_todo(&alloc::format!("PlainMonthDay getter {name}"))),
            };
        }
        // Non-ISO calendar: route through the calendar abstraction layer.
        let f = tcal::iso_to_fields(cal, data.date);
        match name {
            "monthCode" => Ok(self.new_str(&f.month_code)),
            "day" => Ok(NanBox::number(f.day as f64)),
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
                let (iso, cal) =
                    self.pmd_to_temporal(arg_or_undef(args, 0), arg_or_undef(args, 1))?;
                Ok(Some(self.pmd_new(iso, &cal)))
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

    /// Canonicalizes a calendar id against the full Temporal calendar set; an
    /// unsupported id is a RangeError.
    fn pmd_canonicalize(&mut self, s: &str) -> Result<String, ExecError> {
        match tcal::canonicalize_calendar(s) {
            Some(c) => Ok(String::from(c)),
            None => Err(self.pmd_range_error("PlainMonthDay: unsupported calendar")),
        }
    }

    /// The *constructor* calendar argument: `undefined` → iso8601, a non-string →
    /// TypeError, a bare id → canonicalize (an ISO string is not accepted here).
    fn pmd_ctor_calendar(&mut self, v: NanBox) -> Result<String, ExecError> {
        if v.is_undefined() {
            return Ok(String::from("iso8601"));
        }
        let Some(s) = v
            .as_handle()
            .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
        else {
            return Err(self.type_error("PlainMonthDay: calendar must be a string"));
        };
        self.pmd_canonicalize(&s)
    }

    /// Maps a calendar-layer error to the correct exception type.
    fn pmd_layer_fields(
        &mut self,
        cal: &str,
        input: &tcal::FieldsInput,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        match tcal::fields_to_iso(cal, input, overflow) {
            Ok(d) => Ok(d),
            Err(tcal::CalError::Range(m)) => Err(self.pmd_range_error(&m)),
            Err(tcal::CalError::MissingFields(m)) => Err(self.type_error(&m)),
        }
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

    /// Builds a fresh intrinsic `Temporal.PlainMonthDay` carrying calendar id
    /// `cal` (subclassing ignored for `from`/`with`/method results).
    fn pmd_new(&mut self, iso: IsoDate, cal: &str) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainMonthDay,
            date: iso,
            calendar: String::from(cal),
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainMonthDay) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// Builds a fresh intrinsic `Temporal.PlainDate` carrying calendar id `cal`.
    fn pmd_new_plain_date(&mut self, iso: IsoDate, cal: &str) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainDate,
            date: iso,
            calendar: String::from(cal),
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainDate) {
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

    // --- reference-year selection (non-ISO) ---------------------------------

    /// Searches for the ISO reference date of a calendar `(monthCode, day)`: the
    /// latest ISO date on/before 1972-12-31 in which it occurs, else the earliest
    /// afterwards. `None` if the combination never occurs within the searched
    /// window (≈ ±a century around 1972). The day must exist exactly (Reject).
    fn pmd_ref_search(&self, cal: &str, mc: &str, day: i64) -> Option<IsoDate> {
        let limit = iso_to_epoch_days(IsoDate {
            year: 1972,
            month: 12,
            day: 31,
        });
        let anchor = tcal::iso_to_fields(
            cal,
            IsoDate {
                year: 1972,
                month: 12,
                day: 31,
            },
        )
        .year;
        let build = |cy: i64| tcal::FieldsInput {
            year: Some(cy),
            month_code: Some(String::from(mc)),
            day,
            ..Default::default()
        };
        // Prefer the latest ISO date on/before the 1972 anchor (descending: the
        // first match is the largest calendar year, i.e. the latest ISO date).
        let mut cy = anchor + 1;
        while cy >= anchor - 80 {
            if let Ok(iso) = tcal::fields_to_iso(cal, &build(cy), Overflow::Reject)
                && iso_to_epoch_days(iso) <= limit
            {
                return Some(iso);
            }
            cy -= 1;
        }
        // Otherwise the earliest ISO date afterwards (ascending: first crossing).
        let mut cy = anchor;
        while cy <= anchor + 130 {
            if let Ok(iso) = tcal::fields_to_iso(cal, &build(cy), Overflow::Reject)
                && iso_to_epoch_days(iso) > limit
            {
                return Some(iso);
            }
            cy += 1;
        }
        None
    }

    /// Resolves a calendar `(monthCode, day)` to its ISO reference date, honoring
    /// `overflow` for a day that does not occur in any candidate year.
    fn pmd_reference(
        &mut self,
        cal: &str,
        mc: &str,
        day: i64,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        if let Some(iso) = self.pmd_ref_search(cal, mc, day) {
            return Ok(iso);
        }
        if overflow == Overflow::Reject {
            return Err(self.pmd_range_error("PlainMonthDay: monthCode/day does not exist"));
        }
        // Constrain: reduce the *day* as little as possible (a Chinese/Hebrew month
        // is at most 30 days, so day 31 clamps to 30), preferring to keep the day
        // and — for a leap monthCode that cannot hold it — collapse onto the
        // regular month it augments rather than shrink the day further. Concretely
        // Chinese `M02L`-30 → `M02`-30 (a leap month that never reaches 30 days),
        // while `M03L`-31 → `M03L`-30 (a leap month that does) and Hebrew Adar I
        // stays Adar I. Walk candidate days from the request downward; at each day
        // try the leap month first, then its base month.
        let base_code = tcal::constrain_leap_base_code(cal, mc);
        let mut d = day.min(31);
        while d >= 1 {
            if let Some(iso) = self.pmd_ref_search(cal, mc, d) {
                return Ok(iso);
            }
            if let Some(base) = &base_code
                && let Some(iso) = self.pmd_ref_search(cal, base, d)
            {
                return Ok(iso);
            }
            d -= 1;
        }
        Err(self.pmd_range_error("PlainMonthDay: monthCode does not occur"))
    }

    // --- ToTemporalMonthDay -------------------------------------------------

    /// `ToTemporalMonthDay(item, options)` → `(reference IsoDate, calendarId)`.
    fn pmd_to_temporal(
        &mut self,
        item: NanBox,
        options: NanBox,
    ) -> Result<(IsoDate, String), ExecError> {
        if self.is_object_value(item) {
            let h = Handle::from_raw(item.as_handle().unwrap());
            // A PlainMonthDay argument is copied verbatim (options still validated).
            if let Some(td) = self.realm.temporal_at(h)
                && td.kind == TemporalKind::PlainMonthDay
            {
                let iso = td.date;
                let cal = td.calendar.clone();
                self.pmd_overflow(options)?;
                return Ok((iso, cal));
            }
            // Property bag: read calendar, then the date fields, then overflow.
            let cal = self.pmd_read_calendar(h)?;
            if tcal::is_iso(&cal) {
                let iso = self.pmd_fields_to_iso(h, options)?;
                return Ok((iso, cal));
            }
            let iso = self.pmd_fields_to_iso_cal(h, &cal, options)?;
            Ok((iso, cal))
        } else if let Some(s) = item
            .as_handle()
            .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
        {
            let (parsed, cal) = self.pmd_parse_string(&s)?;
            self.pmd_overflow(options)?;
            if tcal::is_iso(&cal) {
                Ok((
                    IsoDate {
                        year: 1972,
                        month: parsed.month,
                        day: parsed.day,
                    },
                    cal,
                ))
            } else {
                // `ISODateWithinLimits`: the parsed reference date must be
                // representable before it is re-anchored for the calendar.
                if !pmd_date_in_range(parsed) {
                    return Err(self.pmd_range_error("PlainMonthDay: reference date out of range"));
                }
                let f = tcal::iso_to_fields(&cal, parsed);
                let iso = self.pmd_reference(&cal, &f.month_code, f.day, Overflow::Constrain)?;
                Ok((iso, cal))
            }
        } else {
            Err(self.type_error("PlainMonthDay: expected an object or ISO string"))
        }
    }

    /// Reads a property bag's `calendar` field, returning its canonical id
    /// (default iso8601).
    fn pmd_read_calendar(&mut self, h: Handle) -> Result<String, ExecError> {
        let v = self.read_member(h, "calendar")?;
        if v.is_undefined() {
            return Ok(String::from("iso8601"));
        }
        // A *calendared* Temporal object supplies its own calendar id.
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.pmd_canonicalize(&cal);
        }
        let Some(s) = v
            .as_handle()
            .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
        else {
            return Err(self.type_error("PlainMonthDay: calendar must be a string"));
        };
        // A bare calendar identifier wins.
        if let Some(c) = tcal::canonicalize_calendar(&s) {
            return Ok(String::from(c));
        }
        // Otherwise it must be an ISO string carrying a `[u-ca=...]` annotation.
        if let Some(p) = parse_iso_datetime(&clamp_leap_second(&s)) {
            let cal = p.calendar.unwrap_or_else(|| String::from("iso8601"));
            return self.pmd_canonicalize(&cal);
        }
        Err(self.pmd_range_error("PlainMonthDay: unsupported calendar"))
    }

    /// `PrepareCalendarFields` + `ISOMonthDayFromFields` for an iso8601 bag.
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

    /// `PrepareCalendarFields` + `CalendarMonthDayFromFields` for a *non-ISO* bag.
    fn pmd_fields_to_iso_cal(
        &mut self,
        h: Handle,
        cal: &str,
        options: NanBox,
    ) -> Result<IsoDate, ExecError> {
        // Read + coerce fields (running getters) in alphabetical order.
        let day_v = self.read_member(h, "day")?;
        let day = if day_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_positive_integer(day_v)? as i64)
        };
        let era_v = self.read_member(h, "era")?;
        let era = if era_v.is_undefined() {
            None
        } else {
            Some(self.coerce_to_string(era_v)?)
        };
        let era_year_v = self.read_member(h, "eraYear")?;
        let era_year = if era_year_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_integer_with_truncation(era_year_v)? as i64)
        };
        let month_v = self.read_member(h, "month")?;
        let month = if month_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_positive_integer(month_v)? as i64)
        };
        let mc_v = self.read_member(h, "monthCode")?;
        let month_code = if mc_v.is_undefined() {
            None
        } else {
            let s = self.coerce_to_string(mc_v)?;
            self.pmd_parse_month_code(&s)?; // syntax validation (RangeError)
            Some(s)
        };
        let year_v = self.read_member(h, "year")?;
        let year = if year_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_integer_with_truncation(year_v)? as i64)
        };
        let overflow = self.pmd_overflow(options)?;

        // Required-field validation (all TypeErrors precede any RangeError).
        let Some(day) = day else {
            return Err(self.type_error("PlainMonthDay: 'day' is required"));
        };
        // era / eraYear only carry meaning for calendars that use eras; for
        // iso8601 / chinese / dangi they are ignored entirely (no pairing check,
        // no year contribution), so a bare `{ era, eraYear, monthCode, day }`
        // still resolves through the monthCode-only reference path.
        if tcal::has_eras(cal) && era.is_some() != era_year.is_some() {
            return Err(self.type_error("PlainMonthDay: era and eraYear must both be present"));
        }
        let has_year =
            year.is_some() || (tcal::has_eras(cal) && era.is_some() && era_year.is_some());
        if month_code.is_none() && month.is_none() {
            return Err(self.type_error("PlainMonthDay: 'month' or 'monthCode' is required"));
        }
        if month.is_some() && !has_year {
            return Err(self.type_error("PlainMonthDay: 'year' is required when 'month' is given"));
        }

        if has_year {
            // A year context resolves the ordinal and constrains the day; the
            // reference year is then re-derived from the resulting (monthCode, day).
            let input = tcal::FieldsInput {
                era: era.clone(),
                era_year,
                year,
                month,
                month_code: month_code.clone(),
                day,
            };
            let iso = self.pmd_layer_fields(cal, &input, overflow)?;
            // `CalendarMonthDayFromFields` resolves the month/day *in that year*
            // first, so the intermediate date must satisfy `ISODateWithinLimits`
            // before the reference year is re-derived. Without this an absurd
            // `year` (the corpus uses ±999999) silently fell through to the
            // reference search and produced a PlainMonthDay instead of throwing.
            let days = iso_to_epoch_days(iso);
            if !(crate::temporal_iso::MIN_EPOCH_DAYS..=crate::temporal_iso::MAX_EPOCH_DAYS)
                .contains(&days)
            {
                return Err(self.pmd_range_error("PlainMonthDay: date is out of range"));
            }
            // An explicit year and era/eraYear must agree.
            if year.is_some() && era.is_some() && era_year.is_some() {
                let input_era = tcal::FieldsInput {
                    era,
                    era_year,
                    year: None,
                    month,
                    month_code,
                    day,
                };
                let iso_era = self.pmd_layer_fields(cal, &input_era, overflow)?;
                if iso_era != iso {
                    return Err(
                        self.pmd_range_error("PlainMonthDay: year and era/eraYear conflict")
                    );
                }
            }
            let f = tcal::iso_to_fields(cal, iso);
            self.pmd_reference(cal, &f.month_code, f.day, Overflow::Reject)
        } else {
            // monthCode + day only: the reference search does the constraining.
            let mc = month_code.unwrap();
            self.pmd_reference(cal, &mc, day, overflow)
        }
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

    /// `ParseTemporalMonthDayString` → `(reference IsoDate, calendarId)`. A
    /// MonthDay-only string yields `{1972, month, day}` + iso8601; a full
    /// calendar-date/date-time string yields its parsed ISO date + the annotated
    /// calendar (which the caller re-anchors for a non-ISO calendar).
    fn pmd_parse_string(&mut self, s: &str) -> Result<(IsoDate, String), ExecError> {
        self.pmd_parse_string_inner(s)
            .ok_or_else(|| self.pmd_range_error("PlainMonthDay: invalid ISO string"))
    }

    fn pmd_parse_string_inner(&self, s: &str) -> Option<(IsoDate, String)> {
        // Reject the non-ASCII minus sign (U+2212) outright.
        if s.as_bytes().windows(3).any(|w| w == [0xE2, 0x88, 0x92]) {
            return None;
        }
        // Split off trailing `[...]` annotations and validate them.
        let ann_start = s.find('[').unwrap_or(s.len());
        let core = &s[..ann_start];
        let ann = &s[ann_start..];
        // Structural validation + the raw first `[u-ca=…]` value.
        let raw_cal = pmd_annotations_calendar(ann)?;
        // No more than nine fractional-second digits are permitted.
        if has_excess_fraction(core) {
            return None;
        }
        // Try the MonthDay-specific grammar first; it admits only iso8601.
        if let Some((m, d)) = pmd_parse_md_core(core) {
            if matches!(&raw_cal, Some(c) if !c.eq_ignore_ascii_case("iso8601")) {
                return None;
            }
            return Some((
                IsoDate {
                    year: 1972,
                    month: m,
                    day: d,
                },
                String::from("iso8601"),
            ));
        }
        // Otherwise a full calendar-date / date-time string (a leap second in
        // the time is ignored for a MonthDay, so clamp `:60` → `:59`).
        let parsed = parse_iso_datetime(&clamp_leap_second(core))?;
        let date = parsed.date?;
        // A UTC designator (`Z`) is not permitted for a PlainMonthDay.
        if parsed.z {
            return None;
        }
        let raw = raw_cal
            .or(parsed.calendar)
            .unwrap_or_else(|| String::from("iso8601"));
        let cal = tcal::canonicalize_calendar(&raw)?;
        Some((date, String::from(cal)))
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
        let cal = data.calendar.clone();
        if !tcal::is_iso(&cal) {
            return self.pmd_with_cal(data, &cal, h, options);
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
        Ok(self.pmd_new(iso, &cal))
    }

    /// The non-ISO `with` path: merges the provided calendar fields over the
    /// receiver's existing calendar fields (using the receiver's calendar year as
    /// the year context) and re-derives the reference ISO date through the layer.
    fn pmd_with_cal(
        &mut self,
        data: &TemporalData,
        cal: &str,
        h: Handle,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        let existing = tcal::iso_to_fields(cal, data.date);
        let mut any = false;
        let day_v = self.read_member(h, "day")?;
        let day = if day_v.is_undefined() {
            None
        } else {
            any = true;
            Some(self.pmd_to_positive_integer(day_v)? as i64)
        };
        let month_v = self.read_member(h, "month")?;
        let month = if month_v.is_undefined() {
            None
        } else {
            any = true;
            Some(self.pmd_to_positive_integer(month_v)? as i64)
        };
        let mc_v = self.read_member(h, "monthCode")?;
        let month_code = if mc_v.is_undefined() {
            None
        } else {
            any = true;
            let s = self.coerce_to_string(mc_v)?;
            self.pmd_parse_month_code(&s)?;
            Some(s)
        };
        let year_v = self.read_member(h, "year")?;
        let year = if year_v.is_undefined() {
            None
        } else {
            any = true;
            Some(self.pmd_to_integer_with_truncation(year_v)? as i64)
        };
        if !any {
            return Err(self.type_error("PlainMonthDay.with: no recognised fields"));
        }
        let overflow = self.pmd_overflow(options)?;

        // `ISODateToFields(calendar, isoDate, month-day)` contributes only
        // `monthCode` and `day` — never a year — and `CalendarMergeFields` drops
        // the receiver's `monthCode` as soon as the partial supplies `month` or
        // `monthCode` of its own.
        let (month, month_code) = if month.is_some() || month_code.is_some() {
            (month, month_code)
        } else {
            (None, Some(existing.month_code.clone()))
        };
        let day = day.unwrap_or(existing.day);
        // With no `monthCode` to go on, a non-ISO month-day cannot be resolved
        // from `month` alone — there is no year to interpret it in, and the
        // receiver does not supply one. (ISO is exempt: its reference year is
        // fixed at 1972, so `month` is unambiguous.) This is the
        // `CalendarResolveFields` TypeError.
        if month_code.is_none() && year.is_none() {
            return Err(self
                .type_error("PlainMonthDay.with: 'monthCode' or 'year' is required with 'month'"));
        }
        // Without a year, the reference-year search resolves (monthCode, day)
        // directly, exactly as the `from({ monthCode, day })` path does.
        let Some(year_ctx) = year else {
            let mc = month_code.expect("checked above");
            let result = self.pmd_reference(cal, &mc, day, overflow)?;
            return Ok(self.pmd_new(result, cal));
        };
        let input = tcal::FieldsInput {
            year: Some(year_ctx),
            month,
            month_code,
            day,
            ..Default::default()
        };
        let iso = self.pmd_layer_fields(cal, &input, overflow)?;
        let f = tcal::iso_to_fields(cal, iso);
        let result = self.pmd_reference(cal, &f.month_code, f.day, Overflow::Reject)?;
        Ok(self.pmd_new(result, cal))
    }

    /// `Temporal.PlainMonthDay.prototype.equals`.
    fn pmd_equals(&mut self, data: &TemporalData, other: NanBox) -> Result<NanBox, ExecError> {
        let (iso, ocal) = self.pmd_to_temporal(other, NanBox::undefined())?;
        let eq = iso.year == data.date.year
            && iso.month == data.date.month
            && iso.day == data.date.day
            && ocal == data.calendar;
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
        let cal = data.calendar.clone();
        if !tcal::is_iso(&cal) {
            return self.pmd_to_plain_date_cal(data, &cal, h);
        }
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
        Ok(self.pmd_new_plain_date(iso, &cal))
    }

    /// The non-ISO `toPlainDate` path: combines the receiver's monthCode + day
    /// with a supplied year (or era + eraYear) in the calendar.
    fn pmd_to_plain_date_cal(
        &mut self,
        data: &TemporalData,
        cal: &str,
        h: Handle,
    ) -> Result<NanBox, ExecError> {
        let existing = tcal::iso_to_fields(cal, data.date);
        let era_v = self.read_member(h, "era")?;
        let era = if era_v.is_undefined() {
            None
        } else {
            Some(self.coerce_to_string(era_v)?)
        };
        let era_year_v = self.read_member(h, "eraYear")?;
        let era_year = if era_year_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_integer_with_truncation(era_year_v)? as i64)
        };
        let year_v = self.read_member(h, "year")?;
        let year = if year_v.is_undefined() {
            None
        } else {
            Some(self.pmd_to_integer_with_truncation(year_v)? as i64)
        };
        if year.is_none() && !(era.is_some() && era_year.is_some()) {
            return Err(self
                .type_error("PlainMonthDay.toPlainDate: 'year' (or era and eraYear) is required"));
        }
        let input = tcal::FieldsInput {
            era,
            era_year,
            year,
            month: None,
            month_code: Some(existing.month_code),
            day: existing.day,
        };
        let iso = self.pmd_layer_fields(cal, &input, Overflow::Constrain)?;
        if !pmd_date_in_range(iso) {
            return Err(self.pmd_range_error("PlainMonthDay.toPlainDate: date out of range"));
        }
        Ok(self.pmd_new_plain_date(iso, cal))
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

    /// `TemporalMonthDayToString`. The ISO reference month-day is always emitted;
    /// the reference year is prepended when the calendar annotation is forced
    /// (`always`/`critical`) or the calendar is non-ISO.
    fn pmd_format(&self, data: &TemporalData, show: &str) -> String {
        let cal = data.calendar.as_str();
        let is_iso = tcal::is_iso(cal);
        let mut result = alloc::format!(
            "{}-{}",
            pad(u64::from(data.date.month), 2),
            pad(u64::from(data.date.day), 2)
        );
        let force_year = show == "always" || show == "critical" || !is_iso;
        if force_year {
            result = alloc::format!("{}-{result}", format_iso_year(data.date.year));
        }
        match show {
            "always" => result.push_str(&alloc::format!("[u-ca={cal}]")),
            "critical" => result.push_str(&alloc::format!("[!u-ca={cal}]")),
            _ if !is_iso => result.push_str(&alloc::format!("[u-ca={cal}]")),
            _ => {}
        }
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

/// Validates a run of trailing `[...]` annotations, returning the raw first
/// `[u-ca=…]` calendar value (if any) for the caller to canonicalize. `None`
/// means the tail is malformed (bad brackets/key, an unknown *critical*
/// annotation, more than one time-zone annotation, or multiple calendar
/// annotations when any is critical).
fn pmd_annotations_calendar(ann: &str) -> Option<Option<String>> {
    let b = ann.as_bytes();
    let mut i = 0usize;
    let mut cal_count = 0u32;
    let mut cal_critical = false;
    let mut tz_count = 0u32;
    let mut first_cal: Option<String> = None;
    while i < b.len() {
        if b[i] != b'[' {
            return None;
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
            return None; // unterminated
        }
        let inner = &ann[start..i];
        i += 1; // consume ']'
        if let Some(eq) = inner.find('=') {
            let key = &inner[..eq];
            let val = &inner[eq + 1..];
            if !valid_annotation_key(key) {
                return None;
            }
            if key == "u-ca" {
                cal_count += 1;
                cal_critical |= critical;
                if first_cal.is_none() {
                    first_cal = Some(String::from(val));
                }
            } else if critical {
                return None; // unknown critical annotation
            }
        } else {
            // A `=`-less annotation is a time-zone annotation: at most one.
            if inner.is_empty() {
                return None;
            }
            tz_count += 1;
        }
    }
    if tz_count > 1 || (cal_count > 1 && cal_critical) {
        return None;
    }
    Some(first_cal)
}
