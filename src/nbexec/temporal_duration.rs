//! `Temporal.Duration` — logic module. A fan-out unit: everything specific to
//! `Duration` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::*;
use crate::temporal_iso::{
    DurationFields, NS_PER_DAY, NS_PER_HOUR, NS_PER_MINUTE, NS_PER_SEC, RoundMode, TemporalData,
    TemporalKind, Unit, balance_time_duration, format_fraction, round_to_increment,
};

/// Prototype method names installed on `Temporal.Duration.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "with",
    "negated",
    "abs",
    "add",
    "subtract",
    "round",
    "total",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
];
/// Getter-accessor names installed on `Temporal.Duration.prototype`.
pub(crate) const GETTERS: &[&str] = &[
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
    "sign",
    "blank",
];

/// The ten Duration field names in the alphabetical order the spec reads them
/// (ToTemporalPartialDurationRecord / ToTemporalDurationRecord), paired with the
/// index into the canonical `[years, months, weeks, days, hours, minutes,
/// seconds, milliseconds, microseconds, nanoseconds]` layout.
const ALPHA_FIELDS: [(&str, usize); 10] = [
    ("days", 3),
    ("hours", 4),
    ("microseconds", 8),
    ("milliseconds", 7),
    ("minutes", 5),
    ("months", 1),
    ("nanoseconds", 9),
    ("seconds", 6),
    ("weeks", 2),
    ("years", 0),
];

/// `2^32`: the (exclusive) magnitude limit on the years/months/weeks fields.
const TWO_POW_32: f64 = 4_294_967_296.0;
/// `2^53`: the (exclusive) magnitude limit on the normalized-seconds total.
const TWO_POW_53: i128 = 9_007_199_254_740_992;

/// Maps a unit string (singular or plural) to its [`Unit`].
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

/// Maps a `roundingMode` string to its [`RoundMode`].
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

/// Nanoseconds in one of the fixed-length units (Day..Nanosecond). `0` for a
/// calendar unit (Year/Month/Week), which has no fixed nanosecond length.
fn ns_per_unit(u: Unit) -> i128 {
    match u {
        Unit::Day => NS_PER_DAY,
        Unit::Hour => NS_PER_HOUR,
        Unit::Minute => NS_PER_MINUTE,
        Unit::Second => NS_PER_SEC,
        Unit::Millisecond => 1_000_000,
        Unit::Microsecond => 1_000,
        Unit::Nanosecond => 1,
        _ => 0,
    }
}

/// `IsValidDuration` over the ten already-integral field values: a single sign
/// across all non-zero fields, `|years|,|months|,|weeks| < 2^32`, and the
/// normalized-seconds total under `2^53` in magnitude.
fn duration_fields_valid(f: &[f64; 10]) -> bool {
    // Single consistent sign (mixed signs are invalid).
    let mut sign = 0.0_f64;
    for &v in f {
        if v != 0.0 {
            let s = v.signum();
            if sign != 0.0 && s != sign {
                return false;
            }
            sign = s;
        }
    }
    // years/months/weeks bounded by 2^32.
    if f[0].abs() >= TWO_POW_32 || f[1].abs() >= TWO_POW_32 || f[2].abs() >= TWO_POW_32 {
        return false;
    }
    // Any time-ish field beyond i64 range cannot be part of a valid duration.
    for &v in &f[3..10] {
        if v.abs() >= 9.0e18 {
            return false;
        }
    }
    let total_ns = f[3] as i128 * NS_PER_DAY
        + f[4] as i128 * NS_PER_HOUR
        + f[5] as i128 * NS_PER_MINUTE
        + f[6] as i128 * NS_PER_SEC
        + f[7] as i128 * 1_000_000
        + f[8] as i128 * 1_000
        + f[9] as i128;
    // Truncate toward zero (not `div_euclid`, which would floor a negative total
    // one unit too far and spuriously reject the min-magnitude durations).
    (total_ns / NS_PER_SEC).abs() < TWO_POW_53
}

/// Parses a Temporal ISO 8601 **duration** string (`±P…`) into raw field values
/// (as `f64`, so an overflowing 1000-digit component becomes a non-finite value
/// the caller's validation rejects). Returns `None` for a malformed string.
///
/// A fraction may appear only on the last present time unit and cascades into the
/// smaller fields (e.g. `PT0.5H` → 30 minutes), which the shared
/// `parse_iso_duration` does not do correctly for hours/minutes.
fn parse_duration_string(s: &str) -> Option<[f64; 10]> {
    let b = s.as_bytes();
    let mut i = 0usize;
    // Sign: ASCII +/- or U+2212 MINUS SIGN.
    let sign = if b.first() == Some(&b'-') {
        i = 1;
        -1.0
    } else if b.first() == Some(&b'+') {
        i = 1;
        1.0
    } else if b.len() >= 3 && b[0] == 0xE2 && b[1] == 0x88 && b[2] == 0x92 {
        i = 3;
        -1.0
    } else {
        1.0
    };
    if b.get(i).map(u8::to_ascii_uppercase) != Some(b'P') {
        return None;
    }
    i += 1;

    let mut f = [0.0_f64; 10];
    let mut any = false;

    // Reads a run of digits into an f64; `None` if there are none.
    let read_int = |i: &mut usize| -> Option<f64> {
        let start = *i;
        let mut v = 0.0_f64;
        while b.get(*i).is_some_and(u8::is_ascii_digit) {
            v = v * 10.0 + f64::from(b[*i] - b'0');
            *i += 1;
        }
        if *i == start { None } else { Some(v) }
    };
    // Reads a `.`/`,` fraction as `(scaled_value, digit_count)` scaled to 1e9.
    let read_frac = |i: &mut usize| -> Option<Option<(i128, u32)>> {
        if b.get(*i) == Some(&b'.') || b.get(*i) == Some(&b',') {
            *i += 1;
            let start = *i;
            let mut val = 0i128;
            let mut count = 0u32;
            while b.get(*i).is_some_and(u8::is_ascii_digit) {
                if count >= 9 {
                    return None; // more than 9 fractional digits is invalid
                }
                val = val * 10 + i128::from(b[*i] - b'0');
                *i += 1;
                count += 1;
            }
            if count == 0 {
                return None; // a separator with no digits is invalid
            }
            let _ = start;
            Some(Some((val, count)))
        } else {
            Some(None)
        }
    };

    // Date portion: integer Y, M, W, D in order (no fractions).
    let date_units: [(u8, usize); 4] = [(b'Y', 0), (b'M', 1), (b'W', 2), (b'D', 3)];
    let mut di = 0usize;
    while let Some(n) = read_int(&mut i) {
        let desig = b.get(i).copied()?.to_ascii_uppercase();
        let pos = date_units.iter().skip(di).position(|&(d, _)| d == desig)?;
        di += pos + 1;
        f[date_units[di - 1].1] = n * sign;
        i += 1;
        any = true;
    }

    // Time portion.
    if b.get(i).map(u8::to_ascii_uppercase) == Some(b'T') {
        i += 1;
        let time_units: [(u8, usize, i128); 3] = [
            (b'H', 4, NS_PER_HOUR),
            (b'M', 5, NS_PER_MINUTE),
            (b'S', 6, NS_PER_SEC),
        ];
        let mut ti = 0usize;
        let mut seen = false;
        while let Some(n) = read_int(&mut i) {
            let frac = read_frac(&mut i)?;
            let desig = b.get(i).copied()?.to_ascii_uppercase();
            let pos = time_units
                .iter()
                .skip(ti)
                .position(|&(d, _, _)| d == desig)?;
            ti += pos + 1;
            let (_, idx, per) = time_units[ti - 1];
            f[idx] = n * sign;
            i += 1;
            seen = true;
            any = true;
            if let Some((val, count)) = frac {
                // Distribute the fraction into the fields smaller than this unit.
                let remaining = val * (per / 10i128.pow(count));
                distribute_frac(&mut f, idx, remaining, sign);
                // A fraction must be on the last unit.
                break;
            }
        }
        if !seen {
            return None; // `T` with no time component
        }
    }

    if !any || i != b.len() {
        return None;
    }
    Some(f)
}

/// Distributes `remaining` nanoseconds (below the unit at field index `from_idx`)
/// into the smaller minute/second/subsecond fields of `f`, applying `sign`.
fn distribute_frac(f: &mut [f64; 10], from_idx: usize, remaining: i128, sign: f64) {
    let mut r = remaining;
    // (field index, nanoseconds-per-unit) for every unit strictly smaller than
    // the fractional one.
    let steps: &[(usize, i128)] = &[
        (5, NS_PER_MINUTE),
        (6, NS_PER_SEC),
        (7, 1_000_000),
        (8, 1_000),
        (9, 1),
    ];
    for &(idx, per) in steps {
        if idx <= from_idx {
            continue;
        }
        f[idx] += (r / per) as f64 * sign;
        r %= per;
    }
}

impl<'a> Interp<'a> {
    /// `new Temporal.Duration(...)`.
    pub(crate) fn duration_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        let mut f = [0.0_f64; 10];
        for (i, slot) in f.iter_mut().enumerate() {
            let arg = args.get(i).copied().unwrap_or_else(NanBox::undefined);
            if !arg.is_undefined() {
                *slot = self.dur_to_integer_if_integral(arg)?;
            }
        }
        let fields = self.dur_build(f)?;
        let data = TemporalData {
            kind: TemporalKind::Duration,
            duration: fields,
            ..Default::default()
        };
        Ok(self.finish_temporal(data, new_target, callee))
    }

    /// A `Temporal.Duration.prototype.<method>()` call.
    pub(crate) fn duration_method(
        &mut self,
        this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        if data.kind != TemporalKind::Duration {
            return Err(self
                .type_error("Temporal.Duration.prototype method called on incompatible receiver"));
        }
        let d = data.duration;
        let arg0 = args.first().copied().unwrap_or_else(NanBox::undefined);
        match method {
            "negated" => {
                let n = DurationFields {
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
                Ok(self.new_duration(n))
            }
            "abs" => {
                let n = DurationFields {
                    years: d.years.abs(),
                    months: d.months.abs(),
                    weeks: d.weeks.abs(),
                    days: d.days.abs(),
                    hours: d.hours.abs(),
                    minutes: d.minutes.abs(),
                    seconds: d.seconds.abs(),
                    milliseconds: d.milliseconds.abs(),
                    microseconds: d.microseconds.abs(),
                    nanoseconds: d.nanoseconds.abs(),
                };
                Ok(self.new_duration(n))
            }
            "with" => self.duration_with(d, arg0),
            "add" => self.duration_add_sub(d, arg0, false),
            "subtract" => self.duration_add_sub(d, arg0, true),
            "round" => self.duration_round(d, arg0),
            "total" => self.duration_total(d, arg0),
            "toString" => {
                let s = self.duration_to_string(d, arg0)?;
                Ok(self.new_str(&s))
            }
            "toJSON" | "toLocaleString" => {
                let s = self.duration_to_string(d, NanBox::undefined())?;
                Ok(self.new_str(&s))
            }
            "valueOf" => Err(self.type_error(
                "Called valueOf on a Temporal.Duration; use compare() or an explicit conversion",
            )),
            _ => {
                let _ = this;
                Err(self.temporal_todo(&alloc::format!("Duration.prototype.{method}")))
            }
        }
    }

    /// A `Temporal.Duration.prototype.<getter>` read.
    pub(crate) fn duration_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        if data.kind != TemporalKind::Duration {
            return Err(self.type_error("Temporal.Duration getter called on incompatible receiver"));
        }
        let d = data.duration;
        let v = match name {
            "years" => d.years,
            "months" => d.months,
            "weeks" => d.weeks,
            "days" => d.days,
            "hours" => d.hours,
            "minutes" => d.minutes,
            "seconds" => d.seconds,
            "milliseconds" => d.milliseconds,
            "microseconds" => d.microseconds,
            "nanoseconds" => d.nanoseconds,
            "sign" => d.sign(),
            "blank" => return Ok(NanBox::boolean(d.sign() == 0)),
            _ => return Err(self.temporal_todo(&alloc::format!("Duration getter {name}"))),
        };
        Ok(NanBox::number(v as f64))
    }

    /// A `Temporal.Duration.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn duration_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        match method {
            "from" => {
                let arg = args.first().copied().unwrap_or_else(NanBox::undefined);
                let d = self.coerce_temporal_duration(arg)?;
                Ok(Some(self.new_duration(d)))
            }
            "compare" => {
                let a = args.first().copied().unwrap_or_else(NanBox::undefined);
                let b = args.get(1).copied().unwrap_or_else(NanBox::undefined);
                let da = self.coerce_temporal_duration(a)?;
                let db = self.coerce_temporal_duration(b)?;
                let opts = args.get(2).copied().unwrap_or_else(NanBox::undefined);
                // GetOptionsObject validates the type, then `relativeTo` is read
                // (observably) before any range/algorithmic validation.
                if let Some(h) = self.dur_options_object(opts)? {
                    let _ = self.read_member(h, "relativeTo")?;
                }
                // Two durations with identical fields compare equal without a
                // relativeTo, even when they contain calendar units.
                if da == db {
                    return Ok(Some(NanBox::number(0.0)));
                }
                // Otherwise calendar units on either operand require a relativeTo,
                // which we do not support; those comparisons throw a RangeError.
                let has_calendar =
                    |d: &DurationFields| d.years != 0 || d.months != 0 || d.weeks != 0;
                if has_calendar(&da) || has_calendar(&db) {
                    return Err(
                        self.dur_range_error("compare requires relativeTo for calendar units")
                    );
                }
                let na = da.days as i128 * NS_PER_DAY + da.time_nanos();
                let nb = db.days as i128 * NS_PER_DAY + db.time_nanos();
                let r = (na - nb).signum() as f64;
                Ok(Some(NanBox::number(r)))
            }
            _ => Ok(None),
        }
    }

    // -- helpers --------------------------------------------------------------

    /// Builds a fresh `Temporal.Duration` bound to the intrinsic prototype.
    fn new_duration(&mut self, d: DurationFields) -> NanBox {
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

    /// A `RangeError` with the given message.
    fn dur_range_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// `ToIntegerIfIntegral(value)`: ToNumber, then require a finite integer
    /// (else RangeError). Symbol/BigInt propagate a TypeError from ToNumber.
    fn dur_to_integer_if_integral(&mut self, v: NanBox) -> Result<f64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() || n.fract() != 0.0 {
            return Err(self.dur_range_error("Duration field must be a finite integer"));
        }
        Ok(n)
    }

    /// Validates `f` as a well-formed duration and returns the integer fields.
    fn dur_build(&mut self, f: [f64; 10]) -> Result<DurationFields, ExecError> {
        if !duration_fields_valid(&f) {
            return Err(self.dur_range_error("Invalid Duration: out of range or mixed signs"));
        }
        Ok(DurationFields {
            years: f[0] as i64,
            months: f[1] as i64,
            weeks: f[2] as i64,
            days: f[3] as i64,
            hours: f[4] as i64,
            minutes: f[5] as i64,
            seconds: f[6] as i64,
            milliseconds: f[7] as i64,
            microseconds: f[8] as i64,
            nanoseconds: f[9] as i64,
        })
    }

    /// Returns the options handle for an options argument: `None` for undefined,
    /// `Some(handle)` for an object, and a TypeError otherwise.
    fn dur_options_object(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        if v.is_undefined() {
            return Ok(None);
        }
        if self.is_object_value(v) {
            return Ok(v.as_handle().map(Handle::from_raw));
        }
        Err(self.type_error("options must be an object or undefined"))
    }

    /// Returns the value as a Rust `String` iff it is a primitive string (used to
    /// recognise the `round`/`total` string shorthand — a non-string primitive
    /// must instead route through GetOptionsObject and become a TypeError).
    fn dur_as_string(&self, v: NanBox) -> Option<String> {
        v.as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
    }

    /// Reads a string-valued option (running getters); `None` if absent/undefined.
    fn dur_string_option(
        &mut self,
        opts: Option<Handle>,
        key: &str,
    ) -> Result<Option<String>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, key)?;
        if v.is_undefined() {
            return Ok(None);
        }
        Ok(Some(self.coerce_to_string(v)?))
    }

    /// `ToTemporalDuration(item)`: a Duration is copied, a string is parsed, an
    /// object is read as a duration record; anything else is a TypeError.
    fn coerce_temporal_duration(&mut self, item: NanBox) -> Result<DurationFields, ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(td) = self.realm.temporal_at(h)
                && td.kind == TemporalKind::Duration
            {
                return Ok(td.duration);
            }
            if self.is_object_value(item) {
                let (f, any) = self.read_partial_duration(h)?;
                if !any {
                    return Err(
                        self.type_error("Invalid duration-like object: no recognized fields")
                    );
                }
                let full = [
                    f[0].unwrap_or(0.0),
                    f[1].unwrap_or(0.0),
                    f[2].unwrap_or(0.0),
                    f[3].unwrap_or(0.0),
                    f[4].unwrap_or(0.0),
                    f[5].unwrap_or(0.0),
                    f[6].unwrap_or(0.0),
                    f[7].unwrap_or(0.0),
                    f[8].unwrap_or(0.0),
                    f[9].unwrap_or(0.0),
                ];
                return self.dur_build(full);
            }
            if let Some(s) = self.realm.string_value(h) {
                return match parse_duration_string(&s) {
                    Some(f) => self.dur_build(f),
                    None => Err(self.dur_range_error("Invalid ISO 8601 duration string")),
                };
            }
        }
        Err(self.type_error("Cannot convert value to a Temporal.Duration"))
    }

    /// Reads the ten plural duration fields off `h` in alphabetical order,
    /// applying ToIntegerIfIntegral to each present (non-undefined) property.
    /// Returns the per-field options plus whether any field was present.
    fn read_partial_duration(&mut self, h: Handle) -> Result<([Option<f64>; 10], bool), ExecError> {
        let mut out: [Option<f64>; 10] = [None; 10];
        let mut any = false;
        for (name, idx) in ALPHA_FIELDS {
            let v = self.read_member(h, name)?;
            if !v.is_undefined() {
                any = true;
                out[idx] = Some(self.dur_to_integer_if_integral(v)?);
            }
        }
        Ok((out, any))
    }

    /// `Temporal.Duration.prototype.with(temporalDurationLike)`.
    fn duration_with(&mut self, d: DurationFields, arg: NanBox) -> Result<NanBox, ExecError> {
        let Some(h) = arg
            .as_handle()
            .map(Handle::from_raw)
            .filter(|_| self.is_object_value(arg))
        else {
            return Err(self.type_error("with() argument must be an object"));
        };
        let (partial, any) = self.read_partial_duration(h)?;
        if !any {
            return Err(self.type_error("with() argument has no recognized duration fields"));
        }
        let base = [
            d.years as f64,
            d.months as f64,
            d.weeks as f64,
            d.days as f64,
            d.hours as f64,
            d.minutes as f64,
            d.seconds as f64,
            d.milliseconds as f64,
            d.microseconds as f64,
            d.nanoseconds as f64,
        ];
        let mut merged = base;
        for i in 0..10 {
            if let Some(v) = partial[i] {
                merged[i] = v;
            }
        }
        let fields = self.dur_build(merged)?;
        Ok(self.new_duration(fields))
    }

    /// `add`/`subtract`: field-wise combine with another duration (negated for
    /// subtract), then re-balance the day+time portion. Calendar units require a
    /// relativeTo (absent here) → RangeError.
    fn duration_add_sub(
        &mut self,
        d: DurationFields,
        arg: NanBox,
        subtract: bool,
    ) -> Result<NanBox, ExecError> {
        let mut other = self.coerce_temporal_duration(arg)?;
        if subtract {
            other = DurationFields {
                years: -other.years,
                months: -other.months,
                weeks: -other.weeks,
                days: -other.days,
                hours: -other.hours,
                minutes: -other.minutes,
                seconds: -other.seconds,
                milliseconds: -other.milliseconds,
                microseconds: -other.microseconds,
                nanoseconds: -other.nanoseconds,
            };
        }
        if d.years != 0
            || d.months != 0
            || d.weeks != 0
            || other.years != 0
            || other.months != 0
            || other.weeks != 0
        {
            return Err(
                self.dur_range_error("Duration arithmetic with calendar units requires relativeTo")
            );
        }
        // The result balances only up to the larger of the two operands' own
        // largest units (durations do not spontaneously balance beyond that).
        let largest = default_largest_unit(&d).min(default_largest_unit(&other));
        let total_ns =
            (d.days + other.days) as i128 * NS_PER_DAY + d.time_nanos() + other.time_nanos();
        let (days, time) = if largest <= Unit::Day {
            (
                (total_ns / NS_PER_DAY) as i64,
                balance_time_duration(total_ns % NS_PER_DAY, Unit::Hour),
            )
        } else {
            (0, balance_time_duration(total_ns, largest))
        };
        let f = [
            0.0,
            0.0,
            0.0,
            days as f64,
            time.hours as f64,
            time.minutes as f64,
            time.seconds as f64,
            time.milliseconds as f64,
            time.microseconds as f64,
            time.nanoseconds as f64,
        ];
        let fields = self.dur_build(f)?;
        Ok(self.new_duration(fields))
    }

    /// `Temporal.Duration.prototype.round(options)` — the time-unit cases only.
    fn duration_round(&mut self, d: DurationFields, arg: NanBox) -> Result<NanBox, ExecError> {
        // The `roundTo` argument is required (a missing/undefined value is a
        // TypeError, not a RangeError).
        if arg.is_undefined() {
            return Err(self.type_error("round() requires a smallestUnit or options argument"));
        }
        // `round` accepts a string shorthand (== smallestUnit) or an options bag;
        // any other non-object primitive is a TypeError (via GetOptionsObject).
        let (opts, smallest_shorthand) = if let Some(s) = self.dur_as_string(arg) {
            (None, Some(s))
        } else {
            (self.dur_options_object(arg)?, None)
        };

        // Read every option, in spec order, and fully coerce each — before any
        // algorithmic validation (GetOptionsObject read order:
        // largestUnit, relativeTo, roundingIncrement, roundingMode, smallestUnit).
        let largest_str = self.dur_string_option(opts, "largestUnit")?;
        let rel = opts
            .map(|h| self.read_member(h, "relativeTo"))
            .transpose()?
            .map(|v| !v.is_undefined())
            .unwrap_or(false);
        let increment = self.dur_rounding_increment(opts)?;
        let mode = match self.dur_string_option(opts, "roundingMode")? {
            Some(s) => {
                parse_round_mode(&s).ok_or_else(|| self.dur_range_error("invalid roundingMode"))?
            }
            None => RoundMode::HalfExpand,
        };
        let smallest_str = match smallest_shorthand {
            Some(s) => Some(s),
            None => self.dur_string_option(opts, "smallestUnit")?,
        };

        if smallest_str.is_none() && largest_str.is_none() {
            return Err(self.dur_range_error("round requires smallestUnit or largestUnit"));
        }
        let smallest = match &smallest_str {
            Some(s) => {
                Some(parse_unit(s).ok_or_else(|| self.dur_range_error("invalid smallestUnit"))?)
            }
            None => None,
        };
        let largest = match &largest_str {
            Some(s) if s == "auto" => None,
            Some(s) => {
                Some(parse_unit(s).ok_or_else(|| self.dur_range_error("invalid largestUnit"))?)
            }
            None => None,
        };

        let has_calendar = d.years != 0 || d.months != 0 || d.weeks != 0;
        let smallest_is_cal = matches!(smallest, Some(Unit::Year | Unit::Month | Unit::Week));
        let largest_is_cal = matches!(largest, Some(Unit::Year | Unit::Month | Unit::Week));
        if (has_calendar || smallest_is_cal || largest_is_cal) && !rel {
            return Err(self.dur_range_error("round with calendar units requires relativeTo"));
        }
        if has_calendar || smallest_is_cal || largest_is_cal {
            return Err(
                self.dur_range_error("round with calendar units and relativeTo is unsupported")
            );
        }

        // Default smallestUnit is nanosecond; default largestUnit is the larger
        // of the duration's own largest unit and smallestUnit.
        let smallest = smallest.unwrap_or(Unit::Nanosecond);
        let default_largest = default_largest_unit(&d);
        let largest = largest.unwrap_or_else(|| default_largest.min(smallest));

        let total_ns = d.days as i128 * NS_PER_DAY + d.time_nanos();
        let incr = ns_per_unit(smallest) * increment;
        let rounded = round_to_increment(total_ns, incr.max(1), mode);

        let (days, time) = if largest <= Unit::Day {
            let days = (rounded / NS_PER_DAY) as i64;
            let rem = rounded % NS_PER_DAY;
            (days, balance_time_duration(rem, Unit::Hour))
        } else {
            (0, balance_time_duration(rounded, largest))
        };
        let f = [
            0.0,
            0.0,
            0.0,
            days as f64,
            time.hours as f64,
            time.minutes as f64,
            time.seconds as f64,
            time.milliseconds as f64,
            time.microseconds as f64,
            time.nanoseconds as f64,
        ];
        let fields = self.dur_build(f)?;
        Ok(self.new_duration(fields))
    }

    /// Reads the `roundingIncrement` option (default 1, positive integer).
    fn dur_rounding_increment(&mut self, opts: Option<Handle>) -> Result<i128, ExecError> {
        let Some(h) = opts else { return Ok(1) };
        let v = self.read_member(h, "roundingIncrement")?;
        if v.is_undefined() {
            return Ok(1);
        }
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() || n < 1.0 || n.trunc() != n {
            return Err(self.dur_range_error("invalid roundingIncrement"));
        }
        Ok(n as i128)
    }

    /// `Temporal.Duration.prototype.total(unitOrOptions)` — time units only.
    fn duration_total(&mut self, d: DurationFields, arg: NanBox) -> Result<NanBox, ExecError> {
        let (opts, unit_str) = if let Some(s) = self.dur_as_string(arg) {
            (None, Some(s))
        } else {
            let o = self.dur_options_object(arg)?;
            let u = self.dur_string_option(o, "unit")?;
            (o, u)
        };
        let Some(unit_str) = unit_str else {
            return Err(self.dur_range_error("total requires a unit"));
        };
        let unit = parse_unit(&unit_str).ok_or_else(|| self.dur_range_error("invalid unit"))?;

        let has_calendar = d.years != 0 || d.months != 0 || d.weeks != 0;
        let unit_is_cal = matches!(unit, Unit::Year | Unit::Month | Unit::Week);
        let rel = opts
            .map(|h| self.read_member(h, "relativeTo"))
            .transpose()?
            .map(|v| !v.is_undefined())
            .unwrap_or(false);
        if (has_calendar || unit_is_cal) && !rel {
            return Err(self.dur_range_error("total with calendar units requires relativeTo"));
        }
        if has_calendar || unit_is_cal {
            return Err(
                self.dur_range_error("total with calendar units and relativeTo is unsupported")
            );
        }

        let total_ns = d.days as i128 * NS_PER_DAY + d.time_nanos();
        let per = ns_per_unit(unit);
        // Exact whole part plus fractional remainder, kept separate to preserve
        // precision when the quotient is large.
        let whole = (total_ns / per) as f64;
        let rem = (total_ns % per) as f64 / per as f64;
        Ok(NanBox::number(whole + rem))
    }

    /// `TemporalDurationToString(duration, precision)`.
    fn duration_to_string(&mut self, d: DurationFields, arg: NanBox) -> Result<String, ExecError> {
        let opts = self.dur_options_object(arg)?;
        let mode = match self.dur_string_option(opts, "roundingMode")? {
            Some(s) => {
                parse_round_mode(&s).ok_or_else(|| self.dur_range_error("invalid roundingMode"))?
            }
            None => RoundMode::Trunc,
        };

        // Determine precision (fixed digit count) / increment from smallestUnit
        // (which overrides) or fractionalSecondDigits.
        let (precision, incr_ns): (Option<u8>, i128) =
            if let Some(su) = self.dur_string_option(opts, "smallestUnit")? {
                let digits = match su.as_str() {
                    "second" | "seconds" => 0u8,
                    "millisecond" | "milliseconds" => 3,
                    "microsecond" | "microseconds" => 6,
                    "nanosecond" | "nanoseconds" => 9,
                    _ => return Err(self.dur_range_error("invalid smallestUnit for toString")),
                };
                (Some(digits), 10i128.pow(u32::from(9 - digits)))
            } else {
                match self.dur_fractional_digits(opts)? {
                    Some(p) => (Some(p), 10i128.pow(u32::from(9 - p))),
                    None => (None, 1),
                }
            };

        // Combine whole seconds + subseconds, round to the chosen increment.
        let total_subsec = d.seconds as i128 * NS_PER_SEC
            + d.milliseconds as i128 * 1_000_000
            + d.microseconds as i128 * 1_000
            + d.nanoseconds as i128;
        let rounded = round_to_increment(total_subsec, incr_ns.max(1), mode);
        let whole_seconds = rounded / NS_PER_SEC;
        let frac = (rounded % NS_PER_SEC).unsigned_abs() as u32;

        let sign = d.sign();
        let mut date_part = String::new();
        if d.years != 0 {
            date_part.push_str(&alloc::format!("{}Y", d.years.unsigned_abs()));
        }
        if d.months != 0 {
            date_part.push_str(&alloc::format!("{}M", d.months.unsigned_abs()));
        }
        if d.weeks != 0 {
            date_part.push_str(&alloc::format!("{}W", d.weeks.unsigned_abs()));
        }
        if d.days != 0 {
            date_part.push_str(&alloc::format!("{}D", d.days.unsigned_abs()));
        }
        let mut time_part = String::new();
        if d.hours != 0 {
            time_part.push_str(&alloc::format!("{}H", d.hours.unsigned_abs()));
        }
        if d.minutes != 0 {
            time_part.push_str(&alloc::format!("{}M", d.minutes.unsigned_abs()));
        }
        let include_seconds = whole_seconds != 0
            || frac != 0
            || precision.is_some()
            || (date_part.is_empty() && time_part.is_empty());
        if include_seconds {
            let frac_str = format_fraction(frac, precision);
            time_part.push_str(&alloc::format!(
                "{}{frac_str}S",
                whole_seconds.unsigned_abs()
            ));
        }

        let mut out = String::new();
        if sign < 0 {
            out.push('-');
        }
        out.push('P');
        out.push_str(&date_part);
        if !time_part.is_empty() {
            out.push('T');
            out.push_str(&time_part);
        }
        Ok(out)
    }

    /// Reads `fractionalSecondDigits` (`GetStringOrNumberOption`): `None` = auto,
    /// `Some(0..=9)` = a fixed digit count.
    fn dur_fractional_digits(&mut self, opts: Option<Handle>) -> Result<Option<u8>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, "fractionalSecondDigits")?;
        if v.is_undefined() {
            return Ok(None);
        }
        if let Some(n) = v.as_number() {
            if n.is_nan() {
                return Err(self.dur_range_error("fractionalSecondDigits out of range"));
            }
            let d = n.floor();
            if !(0.0..=9.0).contains(&d) {
                return Err(self.dur_range_error("fractionalSecondDigits out of range"));
            }
            return Ok(Some(d as u8));
        }
        let s = self.coerce_to_string(v)?;
        if s == "auto" {
            Ok(None)
        } else {
            Err(self.dur_range_error("invalid fractionalSecondDigits"))
        }
    }
}

/// `DefaultTemporalLargestUnit`: the largest unit with a non-zero value (or
/// nanosecond for a zero duration). Calendar fields are already excluded from
/// the code paths that call this, but they are still honoured here.
fn default_largest_unit(d: &DurationFields) -> Unit {
    if d.years != 0 {
        Unit::Year
    } else if d.months != 0 {
        Unit::Month
    } else if d.weeks != 0 {
        Unit::Week
    } else if d.days != 0 {
        Unit::Day
    } else if d.hours != 0 {
        Unit::Hour
    } else if d.minutes != 0 {
        Unit::Minute
    } else if d.seconds != 0 {
        Unit::Second
    } else if d.milliseconds != 0 {
        Unit::Millisecond
    } else if d.microseconds != 0 {
        Unit::Microsecond
    } else {
        Unit::Nanosecond
    }
}
