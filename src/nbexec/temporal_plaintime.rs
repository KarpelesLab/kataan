//! `Temporal.PlainTime` — logic module. A fan-out unit: everything specific to
//! `PlainTime` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    self, DurationFields, IsoTime, Overflow, RoundMode, TemporalData, TemporalKind, Unit,
};

/// Prototype method names installed on `Temporal.PlainTime.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "add",
    "subtract",
    "with",
    "until",
    "since",
    "round",
    "equals",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
];
/// Getter-accessor names installed on `Temporal.PlainTime.prototype`.
pub(crate) const GETTERS: &[&str] = &[
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
];

impl<'a> Interp<'a> {
    // --- small shared helpers -------------------------------------------------

    /// Builds a `RangeError` throw with `msg`.
    fn plaintime_range_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// `ToIntegerWithTruncation(value)`: ToNumber, then a `RangeError` for a
    /// non-finite result, else truncate toward zero. Propagates the `TypeError`
    /// ToNumber raises for a Symbol/BigInt.
    fn plaintime_to_int_trunc(&mut self, v: NanBox) -> Result<i64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.plaintime_range_error("Temporal.PlainTime: value must be finite"));
        }
        Ok(n.trunc() as i64)
    }

    /// `GetOptionsObject`: `undefined` → no options; an object → it; anything
    /// else → `TypeError`.
    fn plaintime_options(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        match v.unpack() {
            Unpacked::Undefined => Ok(None),
            _ if self.is_object_value(v) => Ok(v.as_handle().map(Handle::from_raw)),
            _ => Err(self.type_error("Temporal: options must be an object or undefined")),
        }
    }

    /// Reads a string option (`undefined` → `None`, else coerced to a `String`).
    fn plaintime_str_option(
        &mut self,
        opts: Option<Handle>,
        key: &str,
    ) -> Result<Option<String>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, key)?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        Ok(Some(self.coerce_to_string(v)?))
    }

    /// The `overflow` option (`"constrain"` default, `"reject"`).
    fn plaintime_overflow(&mut self, opts: Option<Handle>) -> Result<Overflow, ExecError> {
        match self.plaintime_str_option(opts, "overflow")?.as_deref() {
            None | Some("constrain") => Ok(Overflow::Constrain),
            Some("reject") => Ok(Overflow::Reject),
            Some(_) => Err(self.plaintime_range_error("Temporal: invalid overflow value")),
        }
    }

    /// The `roundingMode` option (`default` used when absent).
    fn plaintime_rounding_mode(
        &mut self,
        opts: Option<Handle>,
        default: RoundMode,
    ) -> Result<RoundMode, ExecError> {
        match self.plaintime_str_option(opts, "roundingMode")?.as_deref() {
            None => Ok(default),
            Some("ceil") => Ok(RoundMode::Ceil),
            Some("floor") => Ok(RoundMode::Floor),
            Some("expand") => Ok(RoundMode::Expand),
            Some("trunc") => Ok(RoundMode::Trunc),
            Some("halfCeil") => Ok(RoundMode::HalfCeil),
            Some("halfFloor") => Ok(RoundMode::HalfFloor),
            Some("halfExpand") => Ok(RoundMode::HalfExpand),
            Some("halfTrunc") => Ok(RoundMode::HalfTrunc),
            Some("halfEven") => Ok(RoundMode::HalfEven),
            Some(_) => Err(self.plaintime_range_error("Temporal: invalid roundingMode value")),
        }
    }

    /// `ToTemporalRoundingIncrement`: `undefined` → 1; must be a finite integer in
    /// `[1, 1e9]` (`RangeError` otherwise).
    fn plaintime_rounding_increment(&mut self, opts: Option<Handle>) -> Result<i128, ExecError> {
        let Some(h) = opts else { return Ok(1) };
        let v = self.read_member(h, "roundingIncrement")?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(1);
        }
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.plaintime_range_error("Temporal: roundingIncrement must be finite"));
        }
        let i = n.trunc() as i128;
        if !(1..=1_000_000_000).contains(&i) {
            return Err(self.plaintime_range_error("Temporal: roundingIncrement out of range"));
        }
        Ok(i)
    }

    /// Validates a rounding increment against a time unit's maximum (must be a
    /// strict divisor of it).
    fn plaintime_validate_increment(
        &mut self,
        increment: i128,
        unit: Unit,
    ) -> Result<(), ExecError> {
        let max = plaintime_unit_max(unit);
        if increment >= max || max % increment != 0 {
            return Err(self.plaintime_range_error("Temporal: roundingIncrement out of range"));
        }
        Ok(())
    }

    /// Builds a fresh intrinsic `Temporal.PlainTime` from an [`IsoTime`].
    fn plaintime_make(&mut self, time: IsoTime) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::PlainTime,
            time,
            ..Default::default()
        };
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(TemporalKind::PlainTime) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    /// Builds a branded `Temporal.Duration` result linked to
    /// `Temporal.Duration.prototype`. Its field getters are supplied by the
    /// sibling `temporal_duration` module: with that present the ten components
    /// read back correctly (and `instanceof Temporal.Duration`, which checks the
    /// brand, always holds).
    fn plaintime_make_duration(&mut self, d: DurationFields) -> NanBox {
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

    /// `ToTemporalTime(item)`: a `PlainTime`/`PlainDateTime` → its time; a string
    /// → parsed; a property bag → read + regulated with `overflow`. Anything else
    /// (number, boolean, …) → `TypeError`; a bad string → `RangeError`.
    fn plaintime_to_time(&mut self, item: NanBox, opts_raw: NanBox) -> Result<IsoTime, ExecError> {
        if self.is_object_value(item) {
            let h = item.as_handle().map(Handle::from_raw).unwrap();
            if let Some(data) = self.realm.temporal_at(h) {
                match data.kind {
                    TemporalKind::PlainTime | TemporalKind::PlainDateTime => {
                        // Validate the options object type even though overflow is
                        // unused for a direct copy.
                        let opts = self.plaintime_options(opts_raw)?;
                        self.plaintime_overflow(opts)?;
                        return Ok(data.time);
                    }
                    _ => {}
                }
            }
            // Property bag: read the six time fields, then the options.
            return self.plaintime_read_fields(h, IsoTime::default(), true, opts_raw);
        }
        if let Some(raw) = item.as_handle() {
            // A Symbol/BigInt is a handle but not a string → TypeError.
            let sh = Handle::from_raw(raw);
            if self.realm.symbol_at(sh).is_some() || self.realm.bigint_at(sh).is_some() {
                return Err(self.type_error("Temporal.PlainTime: invalid argument"));
            }
        }
        // A primitive string is parsed; other primitives are a TypeError.
        let is_string = item
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.string_value(h).is_some());
        if !is_string {
            return Err(self.type_error("Temporal.PlainTime: invalid argument"));
        }
        let s = self.coerce_to_string(item)?;
        // The string is parsed (RangeError on failure) *before* the options
        // object is validated (TypeError).
        let time = self.plaintime_parse(&s)?;
        let opts = self.plaintime_options(opts_raw)?;
        self.plaintime_overflow(opts)?;
        Ok(time)
    }

    /// Parses an ISO string into an [`IsoTime`] under PlainTime rules: a bare `Z`
    /// designator or a date-only string is rejected.
    fn plaintime_parse(&mut self, s: &str) -> Result<IsoTime, ExecError> {
        let Some(p) = temporal_iso::parse_iso_time_string(s) else {
            return Err(self.plaintime_range_error("Temporal.PlainTime: invalid ISO string"));
        };
        if p.z {
            return Err(
                self.plaintime_range_error("Temporal.PlainTime: UTC designator not allowed")
            );
        }
        match p.time {
            Some(t) => Ok(t),
            None => {
                Err(self.plaintime_range_error("Temporal.PlainTime: string has no time component"))
            }
        }
    }

    /// Reads the six time fields from a property-bag object `h`, starting from
    /// `base`. When `require_one` is set, at least one field must be present
    /// (`TypeError` otherwise). Fields are regulated with the `overflow` option.
    fn plaintime_read_fields(
        &mut self,
        h: Handle,
        base: IsoTime,
        require_one: bool,
        opts_raw: NanBox,
    ) -> Result<IsoTime, ExecError> {
        // Fields are read in alphabetical order (per PrepareTemporalFields), each
        // defaulting to the corresponding component of `base` when undefined.
        let mut hour = base.hour as i64;
        let mut minute = base.minute as i64;
        let mut second = base.second as i64;
        let mut ms = base.millisecond as i64;
        let mut us = base.microsecond as i64;
        let mut ns = base.nanosecond as i64;
        let mut any = false;
        for (name, slot) in [
            ("hour", &mut hour),
            ("microsecond", &mut us),
            ("millisecond", &mut ms),
            ("minute", &mut minute),
            ("nanosecond", &mut ns),
            ("second", &mut second),
        ] {
            let v = self.read_member(h, name)?;
            if !matches!(v.unpack(), Unpacked::Undefined) {
                *slot = self.plaintime_to_int_trunc(v)?;
                any = true;
            }
        }
        if require_one && !any {
            return Err(self.type_error("Temporal.PlainTime: object has no time properties"));
        }
        // The options object is validated / the `overflow` option read only
        // after all fields.
        let opts = self.plaintime_options(opts_raw)?;
        let overflow = self.plaintime_overflow(opts)?;
        match temporal_iso::regulate_iso_time(hour, minute, second, ms, us, ns, overflow) {
            Some(t) => Ok(t),
            None => Err(self.plaintime_range_error("Temporal.PlainTime: time out of range")),
        }
    }

    /// `ToTemporalDuration(item)` reduced to its raw fields: a `Duration` → copy;
    /// a string → parsed; a property bag → the ten fields (each an integral
    /// value); anything else → `TypeError`.
    fn plaintime_to_duration(&mut self, item: NanBox) -> Result<DurationFields, ExecError> {
        if self.is_object_value(item) {
            let h = item.as_handle().map(Handle::from_raw).unwrap();
            if let Some(data) = self.realm.temporal_at(h)
                && data.kind == TemporalKind::Duration
            {
                return Ok(data.duration);
            }
            let mut d = DurationFields::default();
            let mut any = false;
            // Read in alphabetical order (per ToTemporalPartialDurationRecord).
            for (name, slot) in [
                ("days", &mut d.days),
                ("hours", &mut d.hours),
                ("microseconds", &mut d.microseconds),
                ("milliseconds", &mut d.milliseconds),
                ("minutes", &mut d.minutes),
                ("months", &mut d.months),
                ("nanoseconds", &mut d.nanoseconds),
                ("seconds", &mut d.seconds),
                ("weeks", &mut d.weeks),
                ("years", &mut d.years),
            ] {
                let v = self.read_member(h, name)?;
                if matches!(v.unpack(), Unpacked::Undefined) {
                    continue;
                }
                any = true;
                let num = self.coerce_to_number(v)?;
                let n = self.realm.to_number(num);
                if !n.is_finite() || n.fract() != 0.0 {
                    return Err(self.plaintime_range_error("Temporal.Duration: non-integer field"));
                }
                *slot = n as i128;
            }
            if !any {
                return Err(self.type_error("Temporal.Duration: object has no duration properties"));
            }
            if !d.is_valid() {
                return Err(self.plaintime_range_error("Temporal.Duration: mixed-sign fields"));
            }
            self.plaintime_validate_duration(&d)?;
            return Ok(d);
        }
        let is_string = item
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.string_value(h).is_some());
        if !is_string {
            return Err(self.type_error("Temporal.Duration: invalid argument"));
        }
        let s = self.coerce_to_string(item)?;
        match temporal_iso::parse_iso_duration(&s) {
            Some(d) => {
                self.plaintime_validate_duration(&d)?;
                Ok(d)
            }
            None => Err(self.plaintime_range_error("Temporal.Duration: invalid ISO string")),
        }
    }

    /// `IsValidDuration`: the calendar components must be `< 2^32` in magnitude
    /// and the whole duration, reduced to nanoseconds, must stay below
    /// `2^53` seconds. (`RangeError` otherwise.)
    fn plaintime_validate_duration(&mut self, d: &DurationFields) -> Result<(), ExecError> {
        const MAX_CAL: u128 = 1_u128 << 32; // 2^32
        if d.years.unsigned_abs() >= MAX_CAL
            || d.months.unsigned_abs() >= MAX_CAL
            || d.weeks.unsigned_abs() >= MAX_CAL
        {
            return Err(self.plaintime_range_error("Temporal.Duration: value out of range"));
        }
        // 2^53 seconds expressed in nanoseconds.
        const MAX_TIME_NS: i128 = 9_007_199_254_740_992_i128 * 1_000_000_000;
        let total = d.days * temporal_iso::NS_PER_DAY + d.time_nanos();
        if total.abs() >= MAX_TIME_NS {
            return Err(self.plaintime_range_error("Temporal.Duration: value out of range"));
        }
        Ok(())
    }

    /// The shared `until`/`since` implementation.
    fn plaintime_difference(
        &mut self,
        this_time: IsoTime,
        args: &[NanBox],
        is_since: bool,
    ) -> Result<NanBox, ExecError> {
        // `other` (and its fields) are read before any option.
        let other = self.plaintime_to_time(
            args.first().copied().unwrap_or(NanBox::undefined()),
            NanBox::undefined(),
        )?;
        let opts = self.plaintime_options(args.get(1).copied().unwrap_or(NanBox::undefined()))?;

        // GetDifferenceSettings reads options in the fixed order: largestUnit,
        // roundingIncrement, roundingMode, smallestUnit.
        let largest_str = self.plaintime_str_option(opts, "largestUnit")?;
        let increment = self.plaintime_rounding_increment(opts)?;
        let mut mode = self.plaintime_rounding_mode(opts, RoundMode::Trunc)?;
        let smallest_str = self.plaintime_str_option(opts, "smallestUnit")?;

        let smallest = match smallest_str {
            None => Unit::Nanosecond,
            Some(s) => plaintime_parse_time_unit(&s)
                .ok_or_else(|| self.plaintime_range_error("Temporal: invalid smallestUnit"))?,
        };
        // `auto`/absent → the larger of Hour and smallestUnit (Hour is the widest
        // unit a time difference can produce).
        let largest = match largest_str {
            None => Unit::Hour.min(smallest),
            Some(s) if s == "auto" => Unit::Hour.min(smallest),
            Some(s) => plaintime_parse_time_unit(&s)
                .ok_or_else(|| self.plaintime_range_error("Temporal: invalid largestUnit"))?,
        };
        // largestUnit must be at least as large as smallestUnit.
        if largest > smallest {
            return Err(self.plaintime_range_error("Temporal: largestUnit < smallestUnit"));
        }
        self.plaintime_validate_increment(increment, smallest)?;
        if is_since {
            mode = plaintime_negate_mode(mode);
        }

        let diff = temporal_iso::time_to_nanos(other) - temporal_iso::time_to_nanos(this_time);
        let incr_ns = plaintime_unit_ns(smallest) * increment;
        let rounded = temporal_iso::round_to_increment(diff, incr_ns, mode);
        let mut d = temporal_iso::balance_time_duration(rounded, largest);
        if is_since {
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
        Ok(self.plaintime_make_duration(d))
    }

    /// `Temporal.PlainTime.prototype.round`.
    fn plaintime_round(
        &mut self,
        this_time: IsoTime,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let round_to = args.first().copied().unwrap_or(NanBox::undefined());
        if matches!(round_to.unpack(), Unpacked::Undefined) {
            return Err(self.type_error("Temporal.PlainTime.round: options required"));
        }
        // A string shorthand is `{ smallestUnit: <string> }` (no other options).
        let (increment, mode, smallest_str) = if round_to
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.string_value(h).is_some())
        {
            (
                1,
                RoundMode::HalfExpand,
                Some(self.coerce_to_string(round_to)?),
            )
        } else {
            let opts = self.plaintime_options(round_to)?;
            // Options are read in the fixed order roundingIncrement, roundingMode,
            // smallestUnit, before any algorithmic validation.
            let increment = self.plaintime_rounding_increment(opts)?;
            let mode = self.plaintime_rounding_mode(opts, RoundMode::HalfExpand)?;
            let s = self.plaintime_str_option(opts, "smallestUnit")?;
            (increment, mode, s)
        };
        let smallest = match smallest_str {
            None => {
                return Err(
                    self.plaintime_range_error("Temporal.PlainTime.round: smallestUnit required")
                );
            }
            Some(s) => plaintime_parse_time_unit(&s)
                .ok_or_else(|| self.plaintime_range_error("Temporal: invalid smallestUnit"))?,
        };
        self.plaintime_validate_increment(increment, smallest)?;

        let ns = temporal_iso::time_to_nanos(this_time);
        let incr_ns = plaintime_unit_ns(smallest) * increment;
        let rounded = temporal_iso::round_to_increment(ns, incr_ns, mode);
        let (_carry, t) = temporal_iso::balance_time_from_nanos(rounded);
        Ok(self.plaintime_make(t))
    }

    /// `Temporal.PlainTime.prototype.toString` (and `toJSON`/`toLocaleString`).
    fn plaintime_to_string(
        &mut self,
        this_time: IsoTime,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let opts = self.plaintime_options(args.first().copied().unwrap_or(NanBox::undefined()))?;
        // Options are read in the order fractionalSecondDigits, roundingMode,
        // smallestUnit (per GetTemporalShowXOptions / ToSecondsStringPrecision).
        let digits = self.plaintime_fractional_digits(opts)?;
        let mode = self.plaintime_rounding_mode(opts, RoundMode::Trunc)?;
        let smallest = self.plaintime_str_option(opts, "smallestUnit")?;

        // Determine the rounding unit and fractional-second output precision.
        // `show_seconds == false` means `smallestUnit: "minute"` (HH:MM only).
        let (unit_ns, show_seconds, precision): (i128, bool, Option<u8>) = match smallest.as_deref()
        {
            Some("minute") | Some("minutes") => (temporal_iso::NS_PER_MINUTE, false, None),
            Some("second") | Some("seconds") => (temporal_iso::NS_PER_SEC, true, Some(0)),
            Some("millisecond") | Some("milliseconds") => (1_000_000, true, Some(3)),
            Some("microsecond") | Some("microseconds") => (1_000, true, Some(6)),
            Some("nanosecond") | Some("nanoseconds") => (1, true, Some(9)),
            Some(_) => return Err(self.plaintime_range_error("Temporal: invalid smallestUnit")),
            // No smallestUnit: use the fractionalSecondDigits precision.
            None => match digits {
                None => (1, true, None),
                Some(d) => (10_i128.pow(u32::from(9 - d)), true, Some(d)),
            },
        };

        let ns = temporal_iso::time_to_nanos(this_time);
        let rounded = temporal_iso::round_to_increment(ns, unit_ns, mode);
        let (_carry, t) = temporal_iso::balance_time_from_nanos(rounded);

        let mut out = alloc::format!(
            "{}:{}",
            temporal_iso::pad(u64::from(t.hour), 2),
            temporal_iso::pad(u64::from(t.minute), 2)
        );
        if show_seconds {
            out.push(':');
            out.push_str(&temporal_iso::pad(u64::from(t.second), 2));
            let sub = u32::from(t.millisecond) * 1_000_000
                + u32::from(t.microsecond) * 1_000
                + u32::from(t.nanosecond);
            out.push_str(&temporal_iso::format_fraction(sub, precision));
        }
        Ok(self.new_str(&out))
    }

    /// The `fractionalSecondDigits` option: `"auto"`/absent → `None`; a number
    /// floored into `0..=9` (`RangeError` outside).
    fn plaintime_fractional_digits(
        &mut self,
        opts: Option<Handle>,
    ) -> Result<Option<u8>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, "fractionalSecondDigits")?;
        if matches!(v.unpack(), Unpacked::Undefined) {
            return Ok(None);
        }
        // Only a primitive Number takes the numeric path; every other type is
        // stringified and must equal "auto" (else RangeError).
        if let Some(n) = v.as_number() {
            if !n.is_finite() {
                return Err(
                    self.plaintime_range_error("Temporal: fractionalSecondDigits out of range")
                );
            }
            let d = n.floor();
            if !(0.0..=9.0).contains(&d) {
                return Err(
                    self.plaintime_range_error("Temporal: fractionalSecondDigits out of range")
                );
            }
            return Ok(Some(d as u8));
        }
        let s = self.coerce_to_string(v)?;
        if s == "auto" {
            return Ok(None);
        }
        Err(self.plaintime_range_error("Temporal: invalid fractionalSecondDigits"))
    }

    // --- the four dispatch entry points --------------------------------------

    /// `new Temporal.PlainTime(...)`.
    pub(crate) fn plaintime_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        let mut v = [0_i64; 6];
        for (i, slot) in v.iter_mut().enumerate() {
            let a = args.get(i).copied().unwrap_or(NanBox::undefined());
            // An absent/undefined component defaults to 0 (it is not coerced).
            *slot = if matches!(a.unpack(), Unpacked::Undefined) {
                0
            } else {
                self.plaintime_to_int_trunc(a)?
            };
        }
        match temporal_iso::regulate_iso_time(v[0], v[1], v[2], v[3], v[4], v[5], Overflow::Reject)
        {
            Some(time) => {
                let data = TemporalData {
                    kind: TemporalKind::PlainTime,
                    time,
                    ..Default::default()
                };
                self.finish_temporal(data, new_target, callee)
            }
            None => Err(self.plaintime_range_error("Temporal.PlainTime: time out of range")),
        }
    }

    /// A `Temporal.PlainTime.prototype.<method>()` call.
    pub(crate) fn plaintime_method(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let t = data.time;
        match method {
            "add" | "subtract" => {
                let arg = args.first().copied().unwrap_or(NanBox::undefined());
                let d = self.plaintime_to_duration(arg)?;
                let mut delta = d.time_nanos();
                if method == "subtract" {
                    delta = -delta;
                }
                let (_carry, nt) = temporal_iso::add_time(t, delta);
                Ok(self.plaintime_make(nt))
            }
            "with" => {
                let arg = args.first().copied().unwrap_or(NanBox::undefined());
                if !self.is_object_value(arg) {
                    return Err(
                        self.type_error("Temporal.PlainTime.with: argument must be an object")
                    );
                }
                let h = arg.as_handle().map(Handle::from_raw).unwrap();
                // Reject Temporal instances and property bags carrying
                // calendar/timeZone (a PlainTime-like must not).
                if self.realm.temporal_at(h).is_some() {
                    return Err(self.type_error("Temporal.PlainTime.with: invalid argument"));
                }
                for bad in ["calendar", "timeZone"] {
                    let v = self.read_member(h, bad)?;
                    if !matches!(v.unpack(), Unpacked::Undefined) {
                        return Err(self
                            .type_error("Temporal.PlainTime.with: calendar/timeZone not allowed"));
                    }
                }
                let opts_raw = args.get(1).copied().unwrap_or(NanBox::undefined());
                let nt = self.plaintime_read_fields(h, t, true, opts_raw)?;
                Ok(self.plaintime_make(nt))
            }
            "until" => self.plaintime_difference(t, args, false),
            "since" => self.plaintime_difference(t, args, true),
            "round" => self.plaintime_round(t, args),
            "equals" => {
                let other = self.plaintime_to_time(
                    args.first().copied().unwrap_or(NanBox::undefined()),
                    NanBox::undefined(),
                )?;
                Ok(NanBox::boolean(
                    temporal_iso::compare_iso_time(t, other) == core::cmp::Ordering::Equal,
                ))
            }
            "toString" | "toJSON" | "toLocaleString" => {
                // toJSON / toLocaleString take no options.
                let a: &[NanBox] = if method == "toString" { args } else { &[] };
                self.plaintime_to_string(t, a)
            }
            "valueOf" => Err(self.type_error(
                "Temporal.PlainTime.prototype.valueOf: use compare() or an explicit conversion",
            )),
            _ => Err(self.temporal_todo(&alloc::format!("PlainTime.prototype.{method}"))),
        }
    }

    /// A `Temporal.PlainTime.prototype.<getter>` read.
    pub(crate) fn plaintime_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let t = data.time;
        let v = match name {
            "hour" => t.hour as f64,
            "minute" => t.minute as f64,
            "second" => t.second as f64,
            "millisecond" => t.millisecond as f64,
            "microsecond" => t.microsecond as f64,
            "nanosecond" => t.nanosecond as f64,
            _ => return Err(self.temporal_todo(&alloc::format!("PlainTime getter {name}"))),
        };
        Ok(NanBox::number(v))
    }

    /// A `Temporal.PlainTime.<static>()` call. `Ok(None)` = not a recognised static.
    pub(crate) fn plaintime_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        match method {
            "from" => {
                let item = args.first().copied().unwrap_or(NanBox::undefined());
                let opts_raw = args.get(1).copied().unwrap_or(NanBox::undefined());
                let t = self.plaintime_to_time(item, opts_raw)?;
                Ok(Some(self.plaintime_make(t)))
            }
            "compare" => {
                let a = self.plaintime_to_time(
                    args.first().copied().unwrap_or(NanBox::undefined()),
                    NanBox::undefined(),
                )?;
                let b = self.plaintime_to_time(
                    args.get(1).copied().unwrap_or(NanBox::undefined()),
                    NanBox::undefined(),
                )?;
                let r = match temporal_iso::compare_iso_time(a, b) {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Equal => 0.0,
                    core::cmp::Ordering::Greater => 1.0,
                };
                Ok(Some(NanBox::number(r)))
            }
            _ => Ok(None),
        }
    }
}

// --- pure helpers (no `self`) ------------------------------------------------

/// Nanoseconds in one of the time units.
fn plaintime_unit_ns(unit: Unit) -> i128 {
    match unit {
        Unit::Hour => temporal_iso::NS_PER_HOUR,
        Unit::Minute => temporal_iso::NS_PER_MINUTE,
        Unit::Second => temporal_iso::NS_PER_SEC,
        Unit::Millisecond => 1_000_000,
        Unit::Microsecond => 1_000,
        _ => 1,
    }
}

/// The exclusive maximum rounding increment for a time unit.
fn plaintime_unit_max(unit: Unit) -> i128 {
    match unit {
        Unit::Hour => 24,
        Unit::Minute | Unit::Second => 60,
        _ => 1000,
    }
}

/// Parses a time-unit option string (singular or plural).
fn plaintime_parse_time_unit(s: &str) -> Option<Unit> {
    match s {
        "hour" | "hours" => Some(Unit::Hour),
        "minute" | "minutes" => Some(Unit::Minute),
        "second" | "seconds" => Some(Unit::Second),
        "millisecond" | "milliseconds" => Some(Unit::Millisecond),
        "microsecond" | "microseconds" => Some(Unit::Microsecond),
        "nanosecond" | "nanoseconds" => Some(Unit::Nanosecond),
        _ => None,
    }
}

/// `NegateRoundingMode`: swaps ceil/floor and halfCeil/halfFloor; others fixed.
fn plaintime_negate_mode(mode: RoundMode) -> RoundMode {
    match mode {
        RoundMode::Ceil => RoundMode::Floor,
        RoundMode::Floor => RoundMode::Ceil,
        RoundMode::HalfCeil => RoundMode::HalfFloor,
        RoundMode::HalfFloor => RoundMode::HalfCeil,
        other => other,
    }
}
