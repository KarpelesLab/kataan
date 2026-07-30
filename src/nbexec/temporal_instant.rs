//! `Temporal.Instant` — logic module. A fan-out unit: everything specific to
//! `Instant` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
//!
//! An `Instant` is an exact time: a count of nanoseconds since the Unix epoch,
//! held as an `i128` in `TemporalData.epoch_ns`. Every operation reduces to
//! integer nanosecond arithmetic; wall-clock strings are derived on demand from
//! the ISO core (`epoch_days_to_iso` + `balance_time_from_nanos`).
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    self, DurationFields, IsoDate, IsoTime, RoundMode, TemporalData, TemporalKind, Unit,
    balance_time_duration, balance_time_from_nanos, epoch_days_to_iso, format_fraction,
    format_iso_year, iso_to_epoch_days, pad, parse_iso_duration, regulate_iso_date, time_to_nanos,
};

/// Prototype method names installed on `Temporal.Instant.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "add",
    "subtract",
    "until",
    "since",
    "round",
    "equals",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
    "toZonedDateTimeISO",
];
/// Getter-accessor names installed on `Temporal.Instant.prototype`.
pub(crate) const GETTERS: &[&str] = &["epochMilliseconds", "epochNanoseconds"];

/// Nanoseconds in one of the sub-day units, used for rounding increments.
fn unit_length(u: Unit) -> i128 {
    match u {
        Unit::Hour => temporal_iso::NS_PER_HOUR,
        Unit::Minute => temporal_iso::NS_PER_MINUTE,
        Unit::Second => temporal_iso::NS_PER_SEC,
        Unit::Millisecond => 1_000_000,
        Unit::Microsecond => 1_000,
        _ => 1, // Nanosecond (and any coarser unit, unused for Instant)
    }
}

/// Any Temporal unit name (with its plural), year..nanosecond. Used where the
/// spec reads and *casts* the unit before deciding whether it is allowed for the
/// operation (so a disallowed-but-valid unit like `"week"` on `Instant` must
/// still parse, then be rejected only after every option has been read).
fn parse_any_unit(s: &str) -> Option<Unit> {
    Some(match s {
        "year" | "years" => Unit::Year,
        "month" | "months" => Unit::Month,
        "week" | "weeks" => Unit::Week,
        "day" | "days" => Unit::Day,
        _ => return parse_time_unit(s),
    })
}

/// A time unit name (with its plural), restricted to hour..nanosecond.
fn parse_time_unit(s: &str) -> Option<Unit> {
    Some(match s {
        "hour" | "hours" => Unit::Hour,
        "minute" | "minutes" => Unit::Minute,
        "second" | "seconds" => Unit::Second,
        "millisecond" | "milliseconds" => Unit::Millisecond,
        "microsecond" | "microseconds" => Unit::Microsecond,
        "nanosecond" | "nanoseconds" => Unit::Nanosecond,
        _ => return None,
    })
}

fn parse_rounding_mode(s: &str) -> Option<RoundMode> {
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

/// `RoundNumberToIncrementAsIfPositive`: rounds `x` to a multiple of `inc`
/// resolving every mode as though `x` were non-negative (used by
/// `Instant.round` and `toString`, whose "down" is always toward the Big Bang).
fn round_as_if_positive(x: i128, inc: i128, mode: RoundMode) -> i128 {
    if inc <= 1 {
        return x;
    }
    let q = x.div_euclid(inc);
    let r = x.rem_euclid(inc);
    if r == 0 {
        return x;
    }
    let lower = q * inc;
    let upper = lower + inc;
    let pick_upper = match mode {
        RoundMode::Ceil | RoundMode::Expand => true,
        RoundMode::Floor | RoundMode::Trunc => false,
        RoundMode::HalfCeil | RoundMode::HalfExpand => 2 * r >= inc,
        RoundMode::HalfFloor | RoundMode::HalfTrunc => 2 * r > inc,
        RoundMode::HalfEven => {
            if 2 * r == inc {
                q % 2 != 0
            } else {
                2 * r > inc
            }
        }
    };
    if pick_upper { upper } else { lower }
}

/// `RoundNumberToIncrement` (signed): symmetric rounding where "expand"/"trunc"
/// and the half-tie modes follow the sign of `x` (used by `since`/`until`, so
/// that a negative difference mirrors its positive counterpart).
fn round_signed(x: i128, inc: i128, mode: RoundMode) -> i128 {
    if inc <= 1 {
        return x;
    }
    let q = x.div_euclid(inc);
    let r = x.rem_euclid(inc);
    if r == 0 {
        return x;
    }
    let lower = q * inc;
    let upper = lower + inc;
    let pick_upper = match mode {
        RoundMode::Ceil => true,
        RoundMode::Floor => false,
        RoundMode::Trunc => x < 0,
        RoundMode::Expand => x > 0,
        RoundMode::HalfCeil => 2 * r >= inc,
        RoundMode::HalfFloor => 2 * r > inc,
        RoundMode::HalfExpand => {
            if 2 * r == inc {
                x > 0
            } else {
                2 * r > inc
            }
        }
        RoundMode::HalfTrunc => {
            if 2 * r == inc {
                x < 0
            } else {
                2 * r > inc
            }
        }
        RoundMode::HalfEven => {
            if 2 * r == inc {
                q % 2 != 0
            } else {
                2 * r > inc
            }
        }
    };
    if pick_upper { upper } else { lower }
}

/// Whether `v` is a representable epoch-nanosecond value (±8.64e21).
fn valid_epoch(v: i128) -> bool {
    (temporal_iso::MIN_EPOCH_NS..=temporal_iso::MAX_EPOCH_NS).contains(&v)
}

impl<'a> Interp<'a> {
    /// A `RangeError` with `msg`.
    fn range_err(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// Boxes a fresh `Temporal.Instant` for `epoch_ns` on the intrinsic
    /// prototype (subclass constructors are ignored for method/static results).
    fn make_instant(&mut self, epoch_ns: i128) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::Instant,
            epoch_ns,
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::Instant) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// Boxes a fresh `Temporal.Duration` for `since`/`until` results.
    fn make_duration(&mut self, duration: DurationFields) -> NanBox {
        // Duration fields are Numbers: quantize to float64-representable integers.
        let duration = temporal_iso::quantize_duration_fields(duration);
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

    /// Boxes an `i128` as a BigInt value.
    fn make_bigint_i128(&mut self, v: i128) -> NanBox {
        let h = self.realm.new_bigint(crate::bignum::BigInt::from_i128(v));
        NanBox::handle(h.to_raw())
    }

    /// `new Temporal.Instant(epochNanoseconds)` — `epochNanoseconds` is coerced
    /// with `ToBigInt` (a Number/Symbol/undefined → TypeError, a bad string →
    /// SyntaxError), then range-checked.
    pub(crate) fn instant_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        let arg = args.first().copied().unwrap_or_else(NanBox::undefined);
        let big = self.coerce_to_bigint(arg)?;
        let epoch = match big.to_i128() {
            Some(v) if valid_epoch(v) => v,
            _ => return Err(self.range_err("epoch nanoseconds out of range")),
        };
        let data = TemporalData {
            kind: TemporalKind::Instant,
            epoch_ns: epoch,
            ..Default::default()
        };
        self.finish_temporal(data, new_target, callee)
    }

    /// A `Temporal.Instant.prototype.<getter>` read.
    pub(crate) fn instant_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        match name {
            // floor(epoch_ns / 1e6) toward negative infinity, as a Number.
            "epochMilliseconds" => Ok(NanBox::number(data.epoch_ns.div_euclid(1_000_000) as f64)),
            "epochNanoseconds" => Ok(self.make_bigint_i128(data.epoch_ns)),
            _ => Err(self.temporal_todo(&alloc::format!("Instant getter {name}"))),
        }
    }

    /// A `Temporal.Instant.prototype.<method>()` call.
    pub(crate) fn instant_method(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or_else(NanBox::undefined);
        match method {
            "add" => self.instant_add_sub(data, arg(0), 1),
            "subtract" => self.instant_add_sub(data, arg(0), -1),
            "until" => self.instant_difference(data, arg(0), arg(1), false),
            "since" => self.instant_difference(data, arg(0), arg(1), true),
            "round" => self.instant_round(data, arg(0)),
            "equals" => {
                let other = self.resolve_instant(arg(0))?;
                Ok(NanBox::boolean(other == data.epoch_ns))
            }
            "toString" => self.instant_to_string(data, arg(0), false),
            "toJSON" | "toLocaleString" => self.instant_to_string(data, NanBox::undefined(), true),
            "valueOf" => Err(self.type_error(
                "Temporal.Instant.prototype.valueOf: use compare() or an explicit conversion",
            )),
            "toZonedDateTimeISO" => {
                // The time zone is required: `undefined` is not a valid
                // `ToTemporalTimeZoneIdentifier` argument.
                if arg(0).is_undefined() {
                    return Err(self.type_error("toZonedDateTimeISO requires a time zone"));
                }
                let tz = self.temporal_tz_arg(arg(0))?;
                Ok(self.build_temporal(TemporalData {
                    kind: TemporalKind::ZonedDateTime,
                    epoch_ns: data.epoch_ns,
                    tz: Some(tz),
                    calendar: alloc::string::String::from("iso8601"),
                    ..Default::default()
                }))
            }
            _ => Err(self.temporal_todo(&alloc::format!("Instant.prototype.{method}"))),
        }
    }

    /// A `Temporal.Instant.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn instant_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or_else(NanBox::undefined);
        match method {
            "from" => {
                let epoch = self.resolve_instant(arg(0))?;
                Ok(Some(self.make_instant(epoch)))
            }
            "fromEpochMilliseconds" => {
                let num = self.coerce_to_number(arg(0))?;
                let f = self.realm.to_number(num);
                if !f.is_finite() || f.fract() != 0.0 {
                    return Err(self.range_err("epoch milliseconds must be an integer"));
                }
                let max_ms = (temporal_iso::MAX_EPOCH_NS / 1_000_000) as f64;
                if f.abs() > max_ms {
                    return Err(self.range_err("epoch milliseconds out of range"));
                }
                let epoch = (f as i128) * 1_000_000;
                if !valid_epoch(epoch) {
                    return Err(self.range_err("epoch milliseconds out of range"));
                }
                Ok(Some(self.make_instant(epoch)))
            }
            "fromEpochNanoseconds" => {
                let big = self.coerce_to_bigint(arg(0))?;
                let epoch = match big.to_i128() {
                    Some(v) if valid_epoch(v) => v,
                    _ => return Err(self.range_err("epoch nanoseconds out of range")),
                };
                Ok(Some(self.make_instant(epoch)))
            }
            "compare" => {
                let a = self.resolve_instant(arg(0))?;
                let b = self.resolve_instant(arg(1))?;
                let c = match a.cmp(&b) {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Equal => 0.0,
                    core::cmp::Ordering::Greater => 1.0,
                };
                Ok(Some(NanBox::number(c)))
            }
            _ => Ok(None),
        }
    }

    // --- ToTemporalInstant -------------------------------------------------

    /// `ToTemporalInstant(item)` → the epoch nanoseconds. An `Instant`/
    /// `ZonedDateTime` object copies its slot; any other object is taken through
    /// `ToPrimitive(string)` then parsed; a String is parsed; every other
    /// primitive is a TypeError.
    fn resolve_instant(&mut self, v: NanBox) -> Result<i128, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if let Some(d) = self.realm.temporal_at(h)
                && matches!(d.kind, TemporalKind::Instant | TemporalKind::ZonedDateTime)
            {
                return Ok(d.epoch_ns);
            }
            if self.is_object_value(v) {
                let prim = self.coerce_primitive(v, "string")?;
                let s = self.coerce_to_string(prim)?;
                return self.parse_instant_string(&s);
            }
            if let Some(s) = self.realm.string_value(h) {
                return self.parse_instant_string(&s);
            }
        }
        Err(self.type_error("Temporal.Instant: cannot convert value to an Instant"))
    }

    /// Parses an ISO instant string (date + time + `Z`/offset required) and
    /// returns its epoch nanoseconds.
    fn parse_instant_string(&mut self, s: &str) -> Result<i128, ExecError> {
        match parse_instant_epoch(s) {
            Some(epoch) if valid_epoch(epoch) => Ok(epoch),
            _ => Err(self.range_err("invalid ISO instant string")),
        }
    }

    // --- add / subtract ----------------------------------------------------

    fn instant_add_sub(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
        sign: i128,
    ) -> Result<NanBox, ExecError> {
        let d = self.resolve_duration(arg)?;
        if d.years != 0 || d.months != 0 || d.weeks != 0 || d.days != 0 {
            return Err(self
                .range_err("Temporal.Instant arithmetic: duration must not have calendar units"));
        }
        let delta = d.time_nanos() * sign;
        let epoch = data.epoch_ns + delta;
        if !valid_epoch(epoch) {
            return Err(self.range_err("resulting Instant is out of range"));
        }
        Ok(self.make_instant(epoch))
    }

    /// `ToTemporalDuration(item)` restricted to the fields Instant cares about.
    /// A `Duration` object returns its fields; a plain object is read as a
    /// duration property bag; a String is parsed; every other value is a
    /// TypeError (an unparseable string is a RangeError).
    fn resolve_duration(&mut self, v: NanBox) -> Result<DurationFields, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if let Some(d) = self.realm.temporal_at(h)
                && d.kind == TemporalKind::Duration
            {
                return Ok(d.duration);
            }
            if self.is_object_value(v) {
                return self.instant_duration_bag(h);
            }
            if let Some(s) = self.realm.string_value(h) {
                return parse_iso_duration(&s)
                    .ok_or_else(|| self.range_err("invalid ISO duration string"));
            }
        }
        Err(self.type_error("Temporal.Duration: cannot convert value to a duration"))
    }

    /// `ToTemporalDurationRecord` for a property bag: reads the ten plural
    /// duration fields in alphabetical order, each `ToIntegerIfIntegral`.
    fn instant_duration_bag(&mut self, h: Handle) -> Result<DurationFields, ExecError> {
        let fields: [&str; 10] = [
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
        for f in fields {
            let val = self.read_member(h, f)?;
            if matches!(val.unpack(), Unpacked::Undefined) {
                continue;
            }
            any = true;
            let num = self.coerce_to_number(val)?;
            let n = self.realm.to_number(num);
            if !n.is_finite() || n.fract() != 0.0 {
                return Err(self.range_err("duration fields must be integers"));
            }
            let iv = n as i128;
            match f {
                "days" => d.days = iv,
                "hours" => d.hours = iv,
                "microseconds" => d.microseconds = iv,
                "milliseconds" => d.milliseconds = iv,
                "minutes" => d.minutes = iv,
                "months" => d.months = iv,
                "nanoseconds" => d.nanoseconds = iv,
                "seconds" => d.seconds = iv,
                "weeks" => d.weeks = iv,
                _ => d.years = iv,
            }
        }
        if !any {
            return Err(self.type_error("invalid duration-like object"));
        }
        if !d.is_valid() {
            return Err(self.range_err("duration fields must not have mixed signs"));
        }
        Ok(d)
    }

    // --- since / until -----------------------------------------------------

    fn instant_difference(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
        options_arg: NanBox,
        is_since: bool,
    ) -> Result<NanBox, ExecError> {
        let other = self.resolve_instant(arg)?;
        let opts = self.get_options_object(options_arg)?;

        let mut largest: Option<Unit> = None;
        let mut smallest = Unit::Nanosecond;
        let mut mode = RoundMode::Trunc;
        let mut increment: i128 = 1;
        if let Some(h) = opts {
            // Read order per GetDifferenceSettings: largestUnit, roundingIncrement,
            // roundingMode, smallestUnit. Every option is READ and cast (accepting
            // any valid unit name) BEFORE any algorithmic validation; a
            // disallowed-for-Instant unit is rejected only afterward.
            largest = self.read_any_unit_option(h, "largestUnit", true)?;
            increment = self.read_rounding_increment(h)?;
            mode = self.read_rounding_mode(h, RoundMode::Trunc)?;
            if let Some(u) = self.read_any_unit_option(h, "smallestUnit", false)? {
                smallest = u;
            }
        }
        // Instant only supports time units (hour..nanosecond) for both bounds.
        if smallest < Unit::Hour {
            return Err(self.range_err("smallestUnit must be a time unit for Instant"));
        }
        if let Some(l) = largest
            && l < Unit::Hour
        {
            return Err(self.range_err("largestUnit must be a time unit for Instant"));
        }
        // Default largestUnit = the coarser of Second and smallestUnit.
        let largest = largest.unwrap_or_else(|| core::cmp::min(Unit::Second, smallest));
        // largestUnit must not be finer than smallestUnit.
        if largest > smallest {
            return Err(self.range_err("largestUnit must be larger than smallestUnit"));
        }
        let max = Self::max_diff_increment(smallest);
        self.validate_increment(increment, max, false)?;

        let diff = if is_since {
            data.epoch_ns - other
        } else {
            other - data.epoch_ns
        };
        let rounded = round_signed(diff, unit_length(smallest) * increment, mode);
        let fields = balance_time_duration(rounded, largest);
        Ok(self.make_duration(fields))
    }

    /// `MaximumTemporalDurationRoundingIncrement` for a time unit.
    fn max_diff_increment(u: Unit) -> i128 {
        match u {
            Unit::Hour => 24,
            Unit::Minute | Unit::Second => 60,
            _ => 1000, // millisecond / microsecond / nanosecond
        }
    }

    // --- round -------------------------------------------------------------

    fn instant_round(&mut self, data: &TemporalData, arg: NanBox) -> Result<NanBox, ExecError> {
        let (smallest, increment, mode) = if matches!(arg.unpack(), Unpacked::Undefined) {
            return Err(self.type_error("Temporal.Instant.prototype.round requires an argument"));
        } else if let Some(s) = arg
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        {
            let u = parse_time_unit(&s).ok_or_else(|| self.range_err("invalid smallestUnit"))?;
            (u, 1_i128, RoundMode::HalfExpand)
        } else if self.is_object_value(arg) {
            let h = Handle::from_raw(arg.as_handle().unwrap());
            let increment = self.read_rounding_increment(h)?;
            let mode = self.read_rounding_mode(h, RoundMode::HalfExpand)?;
            let smallest = self
                .read_unit_option(h, "smallestUnit", false)?
                .ok_or_else(|| self.range_err("smallestUnit is required"))?;
            (smallest, increment, mode)
        } else {
            return Err(self.type_error("Temporal.Instant.prototype.round: invalid options"));
        };
        // Increment must divide evenly into a solar day (inclusive of the day).
        let per_day = temporal_iso::NS_PER_DAY / unit_length(smallest);
        self.validate_increment(increment, per_day, true)?;
        let rounded = round_as_if_positive(data.epoch_ns, unit_length(smallest) * increment, mode);
        if !valid_epoch(rounded) {
            return Err(self.range_err("rounded Instant is out of range"));
        }
        Ok(self.make_instant(rounded))
    }

    // --- toString ----------------------------------------------------------

    fn instant_to_string(
        &mut self,
        data: &TemporalData,
        options_arg: NanBox,
        ignore_options: bool,
    ) -> Result<NanBox, ExecError> {
        // precision: None = minute (no seconds); Some(None) = auto fractional;
        // Some(Some(n)) = exactly n fractional digits.
        let mut minute_only = false;
        let mut frac: Option<u8> = None; // None => auto
        let mut unit = Unit::Nanosecond;
        let mut increment: i128 = 1;
        let mut mode = RoundMode::Trunc;
        let mut offset: Option<i128> = None;

        if !ignore_options && let Some(h) = self.get_options_object(options_arg)? {
            // Read order mirrors the spec: fractionalSecondDigits, roundingMode,
            // smallestUnit, then timeZone last (all options are read before any
            // algorithmic validation).
            let digits = self.read_fractional_digits(h)?;
            mode = self.read_rounding_mode(h, RoundMode::Trunc)?;
            let su = self.read_string_option(h, "smallestUnit")?;
            // timeZone is READ (raw, no coercion) last, before any algorithmic
            // validation of the units, per ToTemporalTimeZoneIdentifier.
            let tz_val = self.read_member(h, "timeZone")?;
            if let Some(s) = su {
                let u = parse_time_unit(&s)
                    .filter(|u| *u != Unit::Hour)
                    .ok_or_else(|| self.range_err("invalid smallestUnit"))?;
                match u {
                    Unit::Minute => {
                        minute_only = true;
                        unit = Unit::Minute;
                    }
                    Unit::Second => {
                        frac = Some(0);
                        unit = Unit::Second;
                    }
                    Unit::Millisecond => {
                        frac = Some(3);
                        unit = Unit::Millisecond;
                    }
                    Unit::Microsecond => {
                        frac = Some(6);
                        unit = Unit::Microsecond;
                    }
                    _ => {
                        frac = Some(9);
                        unit = Unit::Nanosecond;
                    }
                }
            } else {
                // No smallestUnit: derive from fractionalSecondDigits.
                match digits {
                    None => {
                        frac = None;
                        unit = Unit::Nanosecond;
                    }
                    Some(0) => {
                        frac = Some(0);
                        unit = Unit::Second;
                    }
                    Some(d @ 1..=3) => {
                        frac = Some(d);
                        unit = Unit::Millisecond;
                        increment = 10_i128.pow(u32::from(3 - d));
                    }
                    Some(d @ 4..=6) => {
                        frac = Some(d);
                        unit = Unit::Microsecond;
                        increment = 10_i128.pow(u32::from(6 - d));
                    }
                    Some(d) => {
                        frac = Some(d);
                        unit = Unit::Nanosecond;
                        increment = 10_i128.pow(u32::from(9 - d));
                    }
                }
            }
            // Resolve the timeZone read above via ToTemporalTimeZoneIdentifier
            // (bare id, or a datetime string carrying a `[TimeZone]`/`Z`/offset;
            // a `Temporal.ZonedDateTime` object uses its `[[TimeZone]]`; any other
            // non-string is a TypeError). Its offset at this instant determines
            // the rendered wall-clock. `undefined` renders UTC (`Z`).
            if !tz_val.is_undefined() {
                let id = self.temporal_tz_arg(tz_val)?;
                offset = Some(self.temporal_tz_offset_ns(&id, data.epoch_ns).unwrap_or(0));
            }
        }

        let rounded = round_as_if_positive(data.epoch_ns, unit_length(unit) * increment, mode);
        let off = offset.unwrap_or(0);
        let local = rounded + off;
        let (day, time) = balance_time_from_nanos(local);
        let date = epoch_days_to_iso(day);

        let mut out = alloc::format!(
            "{}-{}-{}T{}:{}",
            format_iso_year(date.year),
            pad(u64::from(date.month), 2),
            pad(u64::from(date.day), 2),
            pad(u64::from(time.hour), 2),
            pad(u64::from(time.minute), 2),
        );
        if !minute_only {
            out.push(':');
            out.push_str(&pad(u64::from(time.second), 2));
            let sub = u32::from(time.millisecond) * 1_000_000
                + u32::from(time.microsecond) * 1_000
                + u32::from(time.nanosecond);
            out.push_str(&format_fraction(sub, frac));
        }
        if offset.is_some() {
            out.push_str(&Self::format_offset(off));
        } else {
            out.push('Z');
        }
        let h = self.realm.new_string(&out);
        Ok(NanBox::handle(h.to_raw()))
    }

    /// `FormatDateTimeUTCOffsetRounded(off)` — the offset (ns east of UTC) as
    /// `±HH:MM`, **rounded to the nearest minute** (ties away from zero). A
    /// sub-minute historical offset therefore serializes as minutes:
    /// `Africa/Monrovia` at the epoch is −00:44:30 and renders `-00:45`, while the
    /// wall clock it produced (23:15:30) keeps the exact offset.
    fn format_offset(off: i128) -> alloc::string::String {
        let sign = if off < 0 { '-' } else { '+' };
        let a = off.abs();
        // Round-half-up on the absolute value == halfExpand on the signed one.
        let minutes = (a + temporal_iso::NS_PER_MINUTE / 2).div_euclid(temporal_iso::NS_PER_MINUTE);
        alloc::format!(
            "{sign}{}:{}",
            pad((minutes / 60) as u64, 2),
            pad((minutes % 60) as u64, 2)
        )
    }

    // --- option-reading helpers -------------------------------------------

    /// `GetOptionsObject`: `undefined` → no options; an object → that object; any
    /// other value → TypeError.
    fn get_options_object(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        if self.is_object_value(v) {
            return Ok(Some(Handle::from_raw(v.as_handle().unwrap())));
        }
        Err(self.type_error("options must be an object or undefined"))
    }

    /// Reads a string-valued option (running any getter). `None` if undefined.
    fn read_string_option(
        &mut self,
        h: Handle,
        key: &str,
    ) -> Result<Option<alloc::string::String>, ExecError> {
        let v = self.read_member(h, key)?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        Ok(Some(self.coerce_to_string(v)?))
    }

    /// Reads a time-unit option. `allow_auto` maps `"auto"` (and absence) to
    /// `None`; an unrecognised unit is a RangeError.
    fn read_unit_option(
        &mut self,
        h: Handle,
        key: &str,
        allow_auto: bool,
    ) -> Result<Option<Unit>, ExecError> {
        match self.read_string_option(h, key)? {
            None => Ok(None),
            Some(s) if allow_auto && s == "auto" => Ok(None),
            Some(s) => Ok(Some(
                parse_time_unit(&s).ok_or_else(|| self.range_err("invalid unit"))?,
            )),
        }
    }

    /// Reads a unit option accepting *any* valid unit name (year..nanosecond);
    /// `allow_auto` maps `"auto"`/absence to `None`. Whether the unit is allowed
    /// for the operation is validated by the caller, after all options are read.
    fn read_any_unit_option(
        &mut self,
        h: Handle,
        key: &str,
        allow_auto: bool,
    ) -> Result<Option<Unit>, ExecError> {
        match self.read_string_option(h, key)? {
            None => Ok(None),
            Some(s) if allow_auto && s == "auto" => Ok(None),
            Some(s) => Ok(Some(
                parse_any_unit(&s).ok_or_else(|| self.range_err("invalid unit"))?,
            )),
        }
    }

    /// `GetRoundingModeOption` with a default.
    fn read_rounding_mode(
        &mut self,
        h: Handle,
        default: RoundMode,
    ) -> Result<RoundMode, ExecError> {
        match self.read_string_option(h, "roundingMode")? {
            None => Ok(default),
            Some(s) => {
                parse_rounding_mode(&s).ok_or_else(|| self.range_err("invalid roundingMode"))
            }
        }
    }

    /// `GetRoundingIncrementOption`: default 1, must be a finite integer in
    /// `[1, 1e9]` (truncated toward zero).
    fn read_rounding_increment(&mut self, h: Handle) -> Result<i128, ExecError> {
        let v = self.read_member(h, "roundingIncrement")?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(1);
        }
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.range_err("roundingIncrement must be finite"));
        }
        let i = n.trunc();
        if !(1.0..=1e9).contains(&i) {
            return Err(self.range_err("roundingIncrement out of range"));
        }
        Ok(i as i128)
    }

    /// `GetTemporalFractionalSecondDigitsOption`: `undefined`/`"auto"` → auto
    /// (`None`); a Number in `[0, 9]` (floored) → that many digits; anything else
    /// → RangeError (or TypeError for a Symbol).
    fn read_fractional_digits(&mut self, h: Handle) -> Result<Option<u8>, ExecError> {
        let v = self.read_member(h, "fractionalSecondDigits")?;
        match v.unpack() {
            Unpacked::Undefined => Ok(None),
            Unpacked::Number(n) => {
                if n.is_nan() || !n.is_finite() {
                    return Err(self.range_err("fractionalSecondDigits out of range"));
                }
                let f = n.floor();
                if !(0.0..=9.0).contains(&f) {
                    return Err(self.range_err("fractionalSecondDigits out of range"));
                }
                Ok(Some(f as u8))
            }
            _ => {
                // A non-number, non-undefined value: only the string "auto" is
                // valid (Symbol → TypeError via coercion).
                let s = self.coerce_to_string(v)?;
                if s == "auto" {
                    Ok(None)
                } else {
                    Err(self.range_err("invalid fractionalSecondDigits"))
                }
            }
        }
    }

    /// `ValidateTemporalRoundingIncrement`: `increment` must divide `dividend`
    /// and not exceed the (in/exclusive) maximum.
    fn validate_increment(
        &mut self,
        increment: i128,
        dividend: i128,
        inclusive: bool,
    ) -> Result<(), ExecError> {
        let maximum = if inclusive || dividend <= 1 {
            dividend
        } else {
            dividend - 1
        };
        if increment > maximum || dividend % increment != 0 {
            return Err(self.range_err("invalid roundingIncrement for the given unit"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Strict ISO-8601 instant-string parser
// ---------------------------------------------------------------------------
//
// `Temporal.Instant` requires a self-contained parser (rather than the shared
// `parse_iso_datetime`) because its conformance corpus checks many rejection
// cases the lenient shared parser accepts: non-ASCII offset signs, out-of-range
// or basic/extended-inconsistent fields, >9 fractional digits, and the full
// annotation grammar (calendar keys must be lowercase, critical unknown
// annotations reject, sub-minute time-zone-annotation offsets reject, etc.).

/// A byte cursor over the input string.
struct P<'s> {
    b: &'s [u8],
    i: usize,
}

impl<'s> P<'s> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn is_digit(&self) -> bool {
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
    /// Reads exactly `n` ASCII digits as an integer.
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
    /// Reads an optional `.`/`,` fraction, returning nanoseconds (0 if absent);
    /// `None` (malformed) for a fraction point with zero or more than nine digits.
    fn fraction(&mut self) -> Option<u32> {
        if self.peek() == Some(b'.') || self.peek() == Some(b',') {
            self.i += 1;
            let start = self.i;
            while self.is_digit() {
                self.i += 1;
            }
            let n = self.i - start;
            if n == 0 || n > 9 {
                return None;
            }
            let mut val = 0_u32;
            for k in 0..9 {
                let d = if k < n {
                    u32::from(self.b[start + k] - b'0')
                } else {
                    0
                };
                val = val * 10 + d;
            }
            Some(val)
        } else {
            Some(0)
        }
    }
}

/// Parses a Temporal ISO instant string into epoch nanoseconds, or `None` if it
/// is malformed or lacks the required time + offset.
fn parse_instant_epoch(s: &str) -> Option<i128> {
    let mut p = P {
        b: s.as_bytes(),
        i: 0,
    };
    let date = parse_iso_date(&mut p)?;

    // A time (introduced by `T`/`t`/space) is mandatory for an Instant.
    if !(p.eat(b'T') || p.eat(b't') || p.eat(b' ')) {
        return None;
    }
    let time = parse_iso_time(&mut p)?;

    // A `Z` designator or an ASCII-signed numeric offset is mandatory.
    let offset = parse_offset(&mut p)?;

    parse_annotations(&mut p)?;
    if p.i != p.b.len() {
        return None;
    }
    Some(iso_to_epoch_days(date) as i128 * temporal_iso::NS_PER_DAY + time_to_nanos(time) - offset)
}

/// `±YYYYYY`/`YYYY` `-`? `MM` `-`? `DD`, basic/extended separators consistent.
fn parse_iso_date(p: &mut P) -> Option<IsoDate> {
    // Year: an ASCII or U+2212 sign forces the 6-digit expanded form.
    let year = if let Some(neg) = eat_year_sign(p) {
        let y = p.digits(6)?;
        if neg && y == 0 {
            return None; // -000000 is invalid
        }
        if neg { -y } else { y }
    } else {
        p.digits(4)?
    };
    let extended = p.eat(b'-');
    let month = p.digits(2)?;
    if extended && !p.eat(b'-') {
        return None; // inconsistent basic/extended separators
    }
    if !extended && p.peek() == Some(b'-') {
        return None;
    }
    let day = p.digits(2)?;
    regulate_iso_date(year as i32, month, day, temporal_iso::Overflow::Reject)
}

/// Consumes a year sign: ASCII `+`/`-` or U+2212. `Some(true)` = minus.
fn eat_year_sign(p: &mut P) -> Option<bool> {
    match p.peek() {
        Some(b'+') => {
            p.i += 1;
            Some(false)
        }
        Some(b'-') => {
            p.i += 1;
            Some(true)
        }
        Some(0xE2) if p.b.get(p.i + 1) == Some(&0x88) && p.b.get(p.i + 2) == Some(&0x92) => {
            p.i += 3;
            Some(true)
        }
        _ => None,
    }
}

/// `HH` (`:`? `MM` (`:`? `SS` fraction?)?)? with a leap-second (`60`) clamp.
fn parse_iso_time(p: &mut P) -> Option<IsoTime> {
    let hour = p.digits(2)?;
    if hour > 23 {
        return None;
    }
    let mut minute = 0;
    let mut second = 0;
    let mut frac = 0_u32;
    let colon = p.eat(b':');
    if colon || p.is_digit() {
        minute = p.digits(2)?;
        if minute > 59 {
            return None;
        }
        let has_sec = if colon { p.eat(b':') } else { p.is_digit() };
        if has_sec {
            second = p.digits(2)?;
            if second > 60 {
                return None;
            }
            frac = p.fraction()?;
        }
    }
    Some(IsoTime {
        hour: hour as u8,
        minute: minute as u8,
        second: second.min(59) as u8, // leap second → :59
        millisecond: (frac / 1_000_000) as u16,
        microsecond: (frac / 1_000 % 1_000) as u16,
        nanosecond: (frac % 1_000) as u16,
    })
}

/// A mandatory `Z`/`z` or ASCII-signed numeric offset → nanoseconds east of UTC.
fn parse_offset(p: &mut P) -> Option<i128> {
    if p.eat(b'Z') || p.eat(b'z') {
        return Some(0);
    }
    let neg = match p.peek() {
        Some(b'+') => false,
        Some(b'-') => true,
        _ => return None,
    };
    p.i += 1;
    let hour = p.digits(2)?;
    if hour > 23 {
        return None;
    }
    let mut minute = 0;
    let mut second = 0;
    let mut frac = 0_u32;
    let colon = p.eat(b':');
    if colon || p.is_digit() {
        minute = p.digits(2)?;
        if minute > 59 {
            return None;
        }
        let has_sec = if colon { p.eat(b':') } else { p.is_digit() };
        if has_sec {
            second = p.digits(2)?;
            if second > 59 {
                return None;
            }
            frac = p.fraction()?;
        }
    }
    let ns = hour * temporal_iso::NS_PER_HOUR as i64
        + minute * temporal_iso::NS_PER_MINUTE as i64
        + second * temporal_iso::NS_PER_SEC as i64
        + i64::from(frac);
    Some(if neg { -i128::from(ns) } else { i128::from(ns) })
}

/// Parses the trailing `[…]` annotation blocks, enforcing the Temporal rules:
/// at most one time-zone annotation (before any key=value), lowercase keys, no
/// critical unknown annotation, ≤1 calendar if any is critical, and no
/// sub-minute offset in a time-zone annotation.
fn parse_annotations(p: &mut P) -> Option<()> {
    let mut tz_seen = false;
    let mut kv_seen = false;
    let mut cal_count = 0_u32;
    let mut cal_critical = false;
    while p.eat(b'[') {
        let critical = p.eat(b'!');
        let start = p.i;
        while p.peek().is_some_and(|c| c != b']') {
            p.i += 1;
        }
        if !p.eat(b']') {
            return None; // unterminated annotation
        }
        let content = core::str::from_utf8(&p.b[start..p.i - 1]).ok()?;
        if let Some(eq) = content.find('=') {
            kv_seen = true;
            let key = &content[..eq];
            if key.is_empty()
                || !key
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-' || c == b'_')
            {
                return None; // keys must be lowercase
            }
            if key == "u-ca" {
                cal_count += 1;
                cal_critical |= critical;
            } else if critical {
                return None; // unknown critical annotation
            }
        } else {
            // A time-zone annotation: at most one, and before any key=value.
            if tz_seen || kv_seen || !valid_tz_annotation(content) {
                return None;
            }
            tz_seen = true;
        }
    }
    if cal_count > 1 && cal_critical {
        return None;
    }
    Some(())
}

/// Whether `s` is a valid time-zone annotation body: a named identifier, or a
/// numeric offset of at most minute precision (a seconds component is rejected).
fn valid_tz_annotation(s: &str) -> bool {
    let bytes = s.as_bytes();
    if matches!(bytes.first(), Some(b'+') | Some(b'-')) {
        let mut q = P { b: bytes, i: 1 };
        if q.digits(2).filter(|h| *h <= 23).is_none() {
            return false;
        }
        if q.eat(b':') {
            if q.digits(2).filter(|m| *m <= 59).is_none() {
                return false;
            }
        } else if q.is_digit() && q.digits(2).filter(|m| *m <= 59).is_none() {
            return false;
        }
        // No trailing content allowed (a seconds field is sub-minute → invalid).
        q.i == bytes.len()
    } else {
        !s.is_empty()
    }
}
