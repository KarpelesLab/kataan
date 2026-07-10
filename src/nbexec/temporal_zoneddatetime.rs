//! `Temporal.ZonedDateTime` — logic module. A fan-out unit: everything specific to
//! `ZonedDateTime` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
//!
//! A `ZonedDateTime` is an exact instant (`TemporalData.epoch_ns`, an `i128` count
//! of nanoseconds since the Unix epoch) plus an IANA time-zone id
//! (`TemporalData.tz`) and a calendar (always `"iso8601"`). Wall-clock fields are
//! derived on demand: `local = epoch_ns + offset(zone, epoch_ns)`, then decomposed
//! with `balance_time_from_nanos` + `epoch_days_to_iso`.
use super::temporal_calendar as tcal;
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    self as iso, DurationFields, IsoDate, IsoTime, Overflow, RoundMode, TemporalData, TemporalKind,
    Unit, balance_time_from_nanos, epoch_days_to_iso, iso_to_epoch_days, time_to_nanos,
};
use alloc::string::{String, ToString};

/// Prototype method names installed on `Temporal.ZonedDateTime.prototype`.
pub(crate) const METHODS: &[&str] = &[
    "with",
    "withPlainTime",
    "withTimeZone",
    "withCalendar",
    "add",
    "subtract",
    "until",
    "since",
    "round",
    "startOfDay",
    "getTimeZoneTransition",
    "equals",
    "toInstant",
    "toPlainDate",
    "toPlainTime",
    "toPlainDateTime",
    "toString",
    "toJSON",
    "toLocaleString",
    "valueOf",
];
/// Getter-accessor names installed on `Temporal.ZonedDateTime.prototype`.
pub(crate) const GETTERS: &[&str] = &[
    "calendarId",
    "timeZoneId",
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
    "epochMilliseconds",
    "epochNanoseconds",
    "dayOfWeek",
    "dayOfYear",
    "weekOfYear",
    "yearOfWeek",
    "hoursInDay",
    "daysInWeek",
    "daysInMonth",
    "daysInYear",
    "monthsInYear",
    "inLeapYear",
    "offset",
    "offsetNanoseconds",
];

/// How a parsed-out UTC offset should reconcile with the time zone (`ToTemporalOffset`).
#[derive(Clone, Copy, PartialEq)]
enum OffsetOpt {
    Prefer,
    Use,
    Ignore,
    Reject,
}

/// Nanoseconds in one unit (Day..Nanosecond).
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

/// Parses the *syntax* of an ISO month code (`"M05"`), returning `(month, is-leap)`.
fn parse_month_code(s: &str) -> Option<(i64, bool)> {
    let b = s.as_bytes();
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

/// Parses a bare offset *identifier* (minute precision only): `±HH`, `±HHMM`,
/// `±HH:MM`. Returns `(offset_ns, canonical "±HH:MM")`. Sub-minute forms and any
/// trailing junk are rejected (they are not valid time-zone identifiers).
fn parse_offset_id(s: &str) -> Option<(i128, String)> {
    let (neg, rest) = match s.as_bytes().first()? {
        b'+' => (false, &s[1..]),
        b'-' => (true, &s[1..]),
        _ => return None,
    };
    let rb = rest.as_bytes();
    if rb.len() < 2 || !rb[0].is_ascii_digit() || !rb[1].is_ascii_digit() {
        return None;
    }
    let hh = i128::from(rb[0] - b'0') * 10 + i128::from(rb[1] - b'0');
    if hh > 23 {
        return None;
    }
    let after = &rest[2..];
    let mm = if after.is_empty() {
        0
    } else {
        let mb = if let Some(m) = after.strip_prefix(':') {
            m
        } else {
            after
        };
        let bytes = mb.as_bytes();
        if bytes.len() != 2 || !bytes[0].is_ascii_digit() || !bytes[1].is_ascii_digit() {
            return None;
        }
        let mm = i128::from(bytes[0] - b'0') * 10 + i128::from(bytes[1] - b'0');
        if mm > 59 {
            return None;
        }
        mm
    };
    let total_min = hh * 60 + mm;
    let ns = total_min * iso::NS_PER_MINUTE;
    let neg = neg && total_min != 0;
    let canon = alloc::format!("{}{:02}:{:02}", if neg { '-' } else { '+' }, hh, mm);
    Some((if neg { -ns } else { ns }, canon))
}

/// Parses a full UTC-offset *value* (allowing seconds/fraction), for an `offset`
/// property-bag field or a string's numeric offset. Returns offset nanoseconds.
fn parse_offset_value(s: &str) -> Option<i128> {
    let (neg, rest) = match s.as_bytes().first()? {
        b'+' => (false, &s[1..]),
        b'-' => (true, &s[1..]),
        _ => return None,
    };
    let mut it = rest.split(':');
    let hh = it.next()?;
    if hh.len() != 2 || !hh.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let hh: i128 = hh.parse().ok()?;
    if hh > 23 {
        return None;
    }
    let mut mm = 0_i128;
    let mut ss = 0_i128;
    let mut frac = 0_i128;
    if let Some(m) = it.next() {
        if m.len() != 2 || !m.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        mm = m.parse().ok()?;
        if mm > 59 {
            return None;
        }
        if let Some(s) = it.next() {
            let (sec, fr) = match s.split_once(['.', ',']) {
                Some((a, b)) => (a, b),
                None => (s, ""),
            };
            if sec.len() != 2 || !sec.bytes().all(|c| c.is_ascii_digit()) {
                return None;
            }
            ss = sec.parse().ok()?;
            if ss > 59 {
                return None;
            }
            if !fr.is_empty() {
                if fr.len() > 9 || !fr.bytes().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                let mut v = 0_i128;
                for k in 0..9 {
                    v = v * 10 + i128::from(fr.as_bytes().get(k).map_or(0, |b| b - b'0'));
                }
                frac = v;
            }
        }
    }
    if it.next().is_some() {
        return None;
    }
    let ns = hh * iso::NS_PER_HOUR + mm * iso::NS_PER_MINUTE + ss * iso::NS_PER_SEC + frac;
    Some(if neg { -ns } else { ns })
}

/// Resolves an IANA time-zone name to its canonical form via the embedded db.
fn resolve_named(s: &str) -> Option<String> {
    timezone_data::load_insensitive(s)
        .ok()
        .map(|z| z.name().to_string())
}

/// The offset (ns east of UTC) of `tz` at the exact instant `epoch_ns`.
fn tz_offset_at(tz: &str, epoch_ns: i128) -> i128 {
    if let Some((ns, _)) = parse_offset_id(tz) {
        return ns;
    }
    if let Ok(z) = timezone_data::load(tz) {
        let secs = epoch_ns.div_euclid(iso::NS_PER_SEC) as i64;
        return i128::from(z.lookup(secs).offset) * iso::NS_PER_SEC;
    }
    0
}

/// The local (wall-clock) ISO date + time for an exact instant in `tz`.
pub(crate) fn local_of(tz: &str, epoch_ns: i128) -> (IsoDate, IsoTime) {
    let off = tz_offset_at(tz, epoch_ns);
    let (day, time) = balance_time_from_nanos(epoch_ns + off);
    (epoch_days_to_iso(day), time)
}

/// `GetEpochNanosecondsFor(tz, wall_ns, "compatible")`: the exact instant whose
/// local wall time is `wall_ns`. Fixed-offset zones are exact; named zones use the
/// offset at the candidate instant (a pragmatic single-step disambiguation).
fn wall_to_epoch(tz: &str, wall_ns: i128) -> i128 {
    if let Some((ns, _)) = parse_offset_id(tz) {
        return wall_ns - ns;
    }
    let o0 = tz_offset_at(tz, wall_ns);
    let cand = wall_ns - o0;
    let o1 = tz_offset_at(tz, cand);
    if o1 == o0 { cand } else { wall_ns - o1 }
}

/// Formats an offset (ns east of UTC) as `±HH:MM` (or `±HH:MM:SS[.fff]`).
fn format_offset(off: i128) -> String {
    let sign = if off < 0 { '-' } else { '+' };
    let a = off.abs();
    let h = a / iso::NS_PER_HOUR;
    let m = (a % iso::NS_PER_HOUR) / iso::NS_PER_MINUTE;
    let s = (a % iso::NS_PER_MINUTE) / iso::NS_PER_SEC;
    let frac = a % iso::NS_PER_SEC;
    if frac != 0 {
        let f = iso::format_fraction(frac as u32, None);
        alloc::format!("{sign}{h:02}:{m:02}:{s:02}{f}")
    } else if s != 0 {
        alloc::format!("{sign}{h:02}:{m:02}:{s:02}")
    } else {
        alloc::format!("{sign}{h:02}:{m:02}")
    }
}

fn valid_epoch(v: i128) -> bool {
    (iso::MIN_EPOCH_NS..=iso::MAX_EPOCH_NS).contains(&v)
}

/// `NegateRoundingMode`: swaps the directional (`ceil`↔`floor`) and half-directional
/// modes; symmetric modes are unchanged.
fn negate_round_mode(mode: RoundMode) -> RoundMode {
    match mode {
        RoundMode::Ceil => RoundMode::Floor,
        RoundMode::Floor => RoundMode::Ceil,
        RoundMode::HalfCeil => RoundMode::HalfFloor,
        RoundMode::HalfFloor => RoundMode::HalfCeil,
        other => other,
    }
}

/// `RoundNumberToIncrement` (signed): rounds `x` to a multiple of `inc`, with the
/// half-tie and directional modes following the sign of `x`.
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

/// Balances a signed nanosecond total into a duration down to `largest` (Day or finer).
fn balance_datetime(total_ns: i128, largest: Unit) -> DurationFields {
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

/// `DifferenceISODateTime(from, to, largestUnit)` (no rounding) in wall time. The
/// date portion is computed in `cal`'s own calendar (ISO takes the shared
/// `DifferenceISODate` fast path).
fn datetime_diff(
    cal: &str,
    from: (IsoDate, IsoTime),
    to: (IsoDate, IsoTime),
    largest: Unit,
) -> DurationFields {
    if largest >= Unit::Day {
        let total = (iso_to_epoch_days(to.0) - iso_to_epoch_days(from.0)) as i128 * iso::NS_PER_DAY
            + (time_to_nanos(to.1) - time_to_nanos(from.1));
        return balance_datetime(total, largest);
    }
    let mut time_ns = time_to_nanos(to.1) - time_to_nanos(from.1);
    let time_sign = time_ns.signum();
    let date_sign = match iso::compare_iso_date(to.0, from.0) {
        core::cmp::Ordering::Greater => 1_i128,
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
    };
    let mut adjusted_to = to.0;
    if time_sign != 0 && time_sign == -date_sign {
        adjusted_to = epoch_days_to_iso(iso_to_epoch_days(to.0) + time_sign as i64);
        time_ns -= time_sign * iso::NS_PER_DAY;
    }
    let (y, mo, w, d) = if tcal::is_iso(cal) {
        iso::difference_iso_date(from.0, adjusted_to, largest)
    } else {
        let p = tcal::calendar_date_until(cal, from.0, adjusted_to, largest);
        (p.years, p.months, p.weeks, p.days)
    };
    let mut dur = iso::balance_time_duration(time_ns, Unit::Hour);
    dur.years = i128::from(y);
    dur.months = i128::from(mo);
    dur.weeks = i128::from(w);
    dur.days = i128::from(d);
    dur
}

impl<'a> Interp<'a> {
    /// A `RangeError` with `msg`.
    fn zdt_range(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m)))
    }

    /// Boxes a fresh `Temporal.ZonedDateTime` carrying calendar id `cal` on the
    /// intrinsic prototype.
    fn make_zdt_cal(&mut self, epoch_ns: i128, tz: String, cal: String) -> NanBox {
        let data = TemporalData {
            kind: TemporalKind::ZonedDateTime,
            epoch_ns,
            tz: Some(tz),
            calendar: cal,
            ..Default::default()
        };
        self.zdt_alloc(data, TemporalKind::ZonedDateTime)
    }

    fn zdt_alloc(&mut self, data: TemporalData, kind: TemporalKind) -> NanBox {
        let h = self.realm.new_temporal(data);
        if let Some(p) = self.temporal_proto(kind) {
            self.realm.set_native_proto(h, p);
        }
        NanBox::handle(h.to_raw())
    }

    fn zdt_bigint_i128(&mut self, v: i128) -> NanBox {
        let h = self.realm.new_bigint(crate::bignum::BigInt::from_i128(v));
        NanBox::handle(h.to_raw())
    }

    /// `new Temporal.ZonedDateTime(epochNanoseconds, timeZone [, calendar])`.
    pub(crate) fn zoneddatetime_construct(
        &mut self,
        args: &[NanBox],
        new_target: NanBox,
        callee: NanBox,
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        // epochNanoseconds: ToBigInt, then range-check.
        let big = self.coerce_to_bigint(arg(0))?;
        let epoch = match big.to_i128() {
            Some(v) if valid_epoch(v) => v,
            _ => return Err(self.zdt_range("epoch nanoseconds out of range")),
        };
        // timeZone: must be a primitive String; parsed as a bare identifier.
        let tz_arg = arg(1);
        let Some(tzs) = tz_arg
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        else {
            return Err(self.type_error("time zone must be a string"));
        };
        let tz = self.parse_tz_identifier(&tzs)?;
        // calendar: undefined → iso8601; else must be a String naming a bare
        // calendar id (CanonicalizeCalendar — an ISO date string is NOT accepted).
        let calendar = self.zdt_calendar_arg(arg(2), false)?;
        let data = TemporalData {
            kind: TemporalKind::ZonedDateTime,
            epoch_ns: epoch,
            tz: Some(tz),
            calendar,
            ..Default::default()
        };
        self.finish_temporal(data, new_target, callee)
    }

    /// Parses a *bare* time-zone identifier (offset or IANA name) → canonical id.
    fn parse_tz_identifier(&mut self, s: &str) -> Result<String, ExecError> {
        if s.is_empty() {
            return Err(self.zdt_range("invalid time zone identifier"));
        }
        if let Some((_, canon)) = parse_offset_id(s) {
            return Ok(canon);
        }
        if let Some(name) = resolve_named(s) {
            return Ok(name);
        }
        Err(self.zdt_range("invalid time zone identifier"))
    }

    /// `ToTemporalTimeZoneIdentifier` from a string: a bare identifier, or a
    /// datetime string carrying a `[TimeZone]` annotation.
    pub(crate) fn tz_from_string(&mut self, s: &str) -> Result<String, ExecError> {
        if s.is_empty() {
            return Err(self.zdt_range("invalid time zone"));
        }
        if let Some((_, canon)) = parse_offset_id(s) {
            return Ok(canon);
        }
        if let Some(name) = resolve_named(s) {
            return Ok(name);
        }
        // A datetime string: a `[TimeZone]` annotation wins; otherwise a `Z`
        // designator means UTC and a numeric offset (minute precision only) names
        // an offset zone.
        if let Some(p) = parse_zdt_string(s) {
            return self.parse_tz_identifier(&p.tz);
        }
        if let Some(p) = iso::parse_iso_datetime(s) {
            if let Some(name) = p.tz_name {
                return self.parse_tz_identifier(&name);
            }
            if p.z {
                return Ok(String::from("UTC"));
            }
            if let Some(off) = p.offset_ns {
                if off % iso::NS_PER_MINUTE != 0 || dt_offset_subminute(s) {
                    return Err(self.zdt_range("sub-minute offset is not a valid time zone"));
                }
                return Ok(format_offset(off));
            }
        }
        Err(self.zdt_range("invalid time zone"))
    }

    /// Validates a constructor/`withCalendar` calendar argument and returns its
    /// canonical id (`undefined` → `"iso8601"`). When `allow_iso_string` is set
    /// (`withCalendar` → `ToTemporalCalendarIdentifier` → `ParseTemporalCalendarString`)
    /// a valid ISO date/time/datetime string is accepted and its `[u-ca=…]`
    /// annotation used; when clear (the constructor → `CanonicalizeCalendar`) only
    /// a bare calendar identifier is accepted.
    fn zdt_calendar_arg(&mut self, v: NanBox, allow_iso_string: bool) -> Result<String, ExecError> {
        if v.is_undefined() {
            return Ok(String::from("iso8601"));
        }
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.zdt_canonicalize_calendar(&cal);
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
        if allow_iso_string && let Some(cal) = self.zdt_calendar_from_iso_string(&s) {
            return Ok(cal);
        }
        Err(self.zdt_range(&alloc::format!("invalid calendar identifier '{s}'")))
    }

    /// Validates a property-bag `calendar` field and returns its canonical id: a
    /// primitive String naming a calendar (bare or via a date-ish ISO string), or a
    /// Temporal object (whose `[[Calendar]]` is used via the fast path). Other
    /// non-strings → TypeError; an unknown calendar → RangeError.
    fn zdt_validate_calendar_field(&mut self, v: NanBox) -> Result<String, ExecError> {
        if let Some(cal) = self.temporal_object_calendar(v) {
            return self.zdt_canonicalize_calendar(&cal);
        }
        let Some(s) = v
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|x| self.realm.string_value(x))
        else {
            return Err(self.type_error("calendar must be a string"));
        };
        if let Some(c) = tcal::canonicalize_calendar(&s) {
            return Ok(String::from(c));
        }
        if let Some(cal) = self.zdt_calendar_from_iso_string(&s) {
            return Ok(cal);
        }
        Err(self.zdt_range(&alloc::format!("invalid calendar identifier '{s}'")))
    }

    /// Canonicalizes a bare calendar identifier; an unsupported id is a RangeError.
    fn zdt_canonicalize_calendar(&mut self, s: &str) -> Result<String, ExecError> {
        match tcal::canonicalize_calendar(s) {
            Some(c) => Ok(String::from(c)),
            None => Err(self.zdt_range(&alloc::format!("invalid calendar identifier '{s}'"))),
        }
    }

    /// Extracts a canonical calendar id from a date/time/datetime ISO string's
    /// `[u-ca=…]` annotation (`ParseTemporalCalendarString`), defaulting to
    /// `"iso8601"`. Returns `None` if the string does not parse or names an
    /// unsupported calendar.
    fn zdt_calendar_from_iso_string(&mut self, s: &str) -> Option<String> {
        let p = iso::parse_iso_datetime(s).or_else(|| iso::parse_iso_time_string(s))?;
        let cal = p.calendar.as_deref().unwrap_or("iso8601");
        tcal::canonicalize_calendar(cal).map(String::from)
    }

    /// Reads an `offset` property-bag field: `ToPrimitive(string)` must yield a
    /// String (else TypeError); bad offset syntax → RangeError.
    fn zdt_read_offset_field(&mut self, v: NanBox) -> Result<i128, ExecError> {
        let prim = self.coerce_primitive(v, "string")?;
        let Some(s) = prim
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|x| self.realm.string_value(x))
        else {
            return Err(self.type_error("offset must be a string"));
        };
        parse_offset_value(&s).ok_or_else(|| self.zdt_range("invalid offset string"))
    }

    /// A `Temporal.ZonedDateTime.prototype.<getter>` read.
    pub(crate) fn zoneddatetime_getter(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let tz = data.tz.clone().unwrap_or_else(|| String::from("UTC"));
        let (d, t) = local_of(&tz, data.epoch_ns);
        let cal = data.calendar.as_str();
        let num = |n: i64| NanBox::number(n as f64);
        // Calendar-independent getters (time / offset / instant / time-zone).
        match name {
            "calendarId" => return Ok(self.new_str(cal)),
            "timeZoneId" => return Ok(self.new_str(&tz)),
            "hour" => return Ok(num(i64::from(t.hour))),
            "minute" => return Ok(num(i64::from(t.minute))),
            "second" => return Ok(num(i64::from(t.second))),
            "millisecond" => return Ok(num(i64::from(t.millisecond))),
            "microsecond" => return Ok(num(i64::from(t.microsecond))),
            "nanosecond" => return Ok(num(i64::from(t.nanosecond))),
            "epochMilliseconds" => {
                return Ok(NanBox::number(data.epoch_ns.div_euclid(1_000_000) as f64));
            }
            "epochNanoseconds" => return Ok(self.zdt_bigint_i128(data.epoch_ns)),
            "dayOfWeek" => return Ok(num(i64::from(iso::iso_day_of_week(d)))),
            "hoursInDay" => {
                let start = wall_to_epoch(&tz, iso_to_epoch_days(d) as i128 * iso::NS_PER_DAY);
                let next = wall_to_epoch(&tz, (iso_to_epoch_days(d) + 1) as i128 * iso::NS_PER_DAY);
                if !valid_epoch(start) || !valid_epoch(next) {
                    return Err(self.zdt_range("day boundary is out of range"));
                }
                return Ok(NanBox::number(
                    (next - start) as f64 / iso::NS_PER_HOUR as f64,
                ));
            }
            "daysInWeek" => return Ok(num(7)),
            "offset" => {
                let s = format_offset(tz_offset_at(&tz, data.epoch_ns));
                return Ok(self.new_str(&s));
            }
            "offsetNanoseconds" => {
                return Ok(NanBox::number(tz_offset_at(&tz, data.epoch_ns) as f64));
            }
            _ => {}
        }
        // ISO-8601 fast path — byte-for-byte the original computation, on the
        // local (wall-clock) date.
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
                "dayOfYear" => num(i64::from(iso::iso_day_of_year(d))),
                "weekOfYear" => num(i64::from(iso::iso_week_of_year(d).0)),
                "yearOfWeek" => num(i64::from(iso::iso_week_of_year(d).1)),
                "daysInMonth" => num(i64::from(iso::iso_days_in_month(d.year, d.month))),
                "daysInYear" => num(i64::from(iso::iso_days_in_year(d.year))),
                "monthsInYear" => num(12),
                "inLeapYear" => NanBox::boolean(iso::is_leap_year(d.year)),
                _ => return Err(self.temporal_todo(&alloc::format!("ZonedDateTime getter {name}"))),
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
            "dayOfYear" => NanBox::number(tcal::day_of_year(cal, d) as f64),
            "weekOfYear" => match tcal::week_of_year(cal, d) {
                Some((w, _)) => NanBox::number(w as f64),
                None => NanBox::undefined(),
            },
            "yearOfWeek" => match tcal::year_of_week(cal, d) {
                Some(y) => NanBox::number(y as f64),
                None => NanBox::undefined(),
            },
            "daysInMonth" => NanBox::number(tcal::days_in_month(cal, d) as f64),
            "daysInYear" => NanBox::number(tcal::days_in_year(cal, d) as f64),
            "monthsInYear" => NanBox::number(tcal::months_in_year(cal, d) as f64),
            "inLeapYear" => NanBox::boolean(tcal::in_leap_year(cal, d)),
            _ => return Err(self.temporal_todo(&alloc::format!("ZonedDateTime getter {name}"))),
        })
    }

    /// A `Temporal.ZonedDateTime.prototype.<method>()` call.
    pub(crate) fn zoneddatetime_method(
        &mut self,
        _this: NanBox,
        data: &TemporalData,
        method: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method {
            "with" => self.zdt_with(data, arg(0), arg(1)),
            "withPlainTime" => self.zdt_with_plain_time(data, arg(0)),
            "withTimeZone" => self.zdt_with_time_zone(data, arg(0)),
            "withCalendar" => {
                if arg(0).is_undefined() {
                    return Err(self.type_error("withCalendar requires a calendar argument"));
                }
                let cal = self.zdt_calendar_arg(arg(0), true)?;
                Ok(self.make_zdt_cal(data.epoch_ns, self.zdt_tz(data), cal))
            }
            "add" => self.zdt_add(data, arg(0), arg(1), 1),
            "subtract" => self.zdt_add(data, arg(0), arg(1), -1),
            "until" => self.zdt_diff(data, arg(0), arg(1), false),
            "since" => self.zdt_diff(data, arg(0), arg(1), true),
            "round" => self.zdt_round(data, arg(0)),
            "startOfDay" => {
                let tz = self.zdt_tz(data);
                let (d, _) = local_of(&tz, data.epoch_ns);
                let epoch = wall_to_epoch(&tz, iso_to_epoch_days(d) as i128 * iso::NS_PER_DAY);
                if !valid_epoch(epoch) {
                    return Err(self.zdt_range("start of day is out of range"));
                }
                Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
            }
            "getTimeZoneTransition" => self.zdt_get_transition(data, arg(0)),
            "equals" => {
                let (epoch, tz, cal) = self.resolve_zdt(arg(0))?;
                let eq = epoch == data.epoch_ns && tz == self.zdt_tz(data) && cal == data.calendar;
                Ok(NanBox::boolean(eq))
            }
            "toInstant" => {
                let data2 = TemporalData {
                    kind: TemporalKind::Instant,
                    epoch_ns: data.epoch_ns,
                    ..Default::default()
                };
                Ok(self.zdt_alloc(data2, TemporalKind::Instant))
            }
            "toPlainDate" => {
                let (d, _) = local_of(&self.zdt_tz(data), data.epoch_ns);
                let data2 = TemporalData {
                    kind: TemporalKind::PlainDate,
                    date: d,
                    calendar: data.calendar.clone(),
                    ..Default::default()
                };
                Ok(self.zdt_alloc(data2, TemporalKind::PlainDate))
            }
            "toPlainTime" => {
                let (_, t) = local_of(&self.zdt_tz(data), data.epoch_ns);
                let data2 = TemporalData {
                    kind: TemporalKind::PlainTime,
                    time: t,
                    ..Default::default()
                };
                Ok(self.zdt_alloc(data2, TemporalKind::PlainTime))
            }
            "toPlainDateTime" => {
                let (d, t) = local_of(&self.zdt_tz(data), data.epoch_ns);
                let data2 = TemporalData {
                    kind: TemporalKind::PlainDateTime,
                    date: d,
                    time: t,
                    calendar: data.calendar.clone(),
                    ..Default::default()
                };
                Ok(self.zdt_alloc(data2, TemporalKind::PlainDateTime))
            }
            "toString" => self.zdt_to_string(data, arg(0)),
            "toJSON" | "toLocaleString" => self.zdt_to_string(data, NanBox::undefined()),
            "valueOf" => Err(self.type_error(
                "Temporal.ZonedDateTime.prototype.valueOf must not be called; use compare() or an \
                 explicit conversion",
            )),
            _ => Err(self.temporal_todo(&alloc::format!("ZonedDateTime.prototype.{method}"))),
        }
    }

    /// A `Temporal.ZonedDateTime.<static>()` call. `Ok(None)` = not recognised.
    pub(crate) fn zoneddatetime_static(
        &mut self,
        _ctor: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        match method {
            "from" => {
                // A ZonedDateTime item is copied (after validating the options bag).
                if let Some(h) = arg(0).as_handle().map(Handle::from_raw)
                    && let Some(dd) = self.realm.temporal_at(h)
                    && dd.kind == TemporalKind::ZonedDateTime
                {
                    let opts = self.zdt_options(arg(1))?;
                    self.zdt_disambiguation(opts)?;
                    self.zdt_offset_option(opts)?;
                    self.zdt_overflow(opts)?;
                    let (epoch, tz, cal) = (
                        dd.epoch_ns,
                        dd.tz.clone().unwrap_or_default(),
                        dd.calendar.clone(),
                    );
                    return Ok(Some(self.make_zdt_cal(epoch, tz, cal)));
                }
                let (epoch, tz, cal) = self.interpret_zdt(arg(0), arg(1))?;
                Ok(Some(self.make_zdt_cal(epoch, tz, cal)))
            }
            "compare" => {
                let a = self.resolve_zdt(arg(0))?.0;
                let b = self.resolve_zdt(arg(1))?.0;
                Ok(Some(NanBox::number(match a.cmp(&b) {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Greater => 1.0,
                    core::cmp::Ordering::Equal => 0.0,
                })))
            }
            _ => Ok(None),
        }
    }

    /// The receiver's time-zone id (defaulting to `"UTC"` if somehow absent).
    fn zdt_tz(&self, data: &TemporalData) -> String {
        data.tz.clone().unwrap_or_else(|| String::from("UTC"))
    }

    // --- options helpers ---------------------------------------------------

    fn zdt_options(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        if v.is_undefined() {
            Ok(None)
        } else if self.is_object_value(v) {
            Ok(v.as_handle().map(Handle::from_raw))
        } else {
            Err(self.type_error("options must be an object or undefined"))
        }
    }

    fn zdt_str_option(
        &mut self,
        opts: Option<Handle>,
        key: &str,
        allowed: &[&str],
    ) -> Result<Option<String>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, key)?;
        if v.is_undefined() {
            return Ok(None);
        }
        let s = self.coerce_to_string(v)?;
        if allowed.contains(&s.as_str()) {
            Ok(Some(s))
        } else {
            Err(self.zdt_range(&alloc::format!("invalid value for option {key}")))
        }
    }

    fn zdt_overflow(&mut self, opts: Option<Handle>) -> Result<Overflow, ExecError> {
        Ok(
            match self
                .zdt_str_option(opts, "overflow", &["constrain", "reject"])?
                .as_deref()
            {
                Some("reject") => Overflow::Reject,
                _ => Overflow::Constrain,
            },
        )
    }

    fn zdt_offset_option(&mut self, opts: Option<Handle>) -> Result<OffsetOpt, ExecError> {
        Ok(
            match self
                .zdt_str_option(opts, "offset", &["prefer", "use", "ignore", "reject"])?
                .as_deref()
            {
                Some("use") => OffsetOpt::Use,
                Some("ignore") => OffsetOpt::Ignore,
                Some("prefer") => OffsetOpt::Prefer,
                _ => OffsetOpt::Reject,
            },
        )
    }

    fn zdt_disambiguation(&mut self, opts: Option<Handle>) -> Result<(), ExecError> {
        self.zdt_str_option(
            opts,
            "disambiguation",
            &["compatible", "earlier", "later", "reject"],
        )?;
        Ok(())
    }

    fn zdt_rounding_mode(
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
        Ok(match self.zdt_str_option(opts, "roundingMode", &allowed)? {
            Some(s) => parse_round_mode(&s).unwrap_or(default),
            None => default,
        })
    }

    fn zdt_rounding_increment(&mut self, opts: Option<Handle>) -> Result<i64, ExecError> {
        let Some(h) = opts else { return Ok(1) };
        let v = self.read_member(h, "roundingIncrement")?;
        if v.is_undefined() {
            return Ok(1);
        }
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.zdt_range("roundingIncrement must be finite"));
        }
        let i = n.trunc();
        if !(1.0..=1e9).contains(&i) {
            return Err(self.zdt_range("roundingIncrement out of range"));
        }
        Ok(i as i64)
    }

    // --- field-bag reading -------------------------------------------------

    fn zdt_field(&mut self, h: Handle, key: &str) -> Result<Option<NanBox>, ExecError> {
        let v = self.read_member(h, key)?;
        Ok((!v.is_undefined()).then_some(v))
    }

    /// `ToIntegerWithTruncation`: ToNumber, then truncate; non-finite → RangeError.
    fn zdt_to_int(&mut self, v: NanBox) -> Result<i64, ExecError> {
        let num = self.coerce_to_number(v)?;
        let n = self.realm.to_number(num);
        if !n.is_finite() {
            return Err(self.zdt_range("value must be a finite integer"));
        }
        Ok(n.trunc() as i64)
    }

    // --- ToTemporalZonedDateTime ------------------------------------------

    /// `ToTemporalZonedDateTime(item, options)` → `(epoch_ns, tz_id, calendar_id)`.
    fn interpret_zdt(
        &mut self,
        item: NanBox,
        options: NanBox,
    ) -> Result<(i128, String, String), ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(dd) = self.realm.temporal_at(h)
                && dd.kind == TemporalKind::ZonedDateTime
            {
                let opts = self.zdt_options(options)?;
                self.zdt_disambiguation(opts)?;
                self.zdt_offset_option(opts)?;
                self.zdt_overflow(opts)?;
                return Ok((
                    dd.epoch_ns,
                    dd.tz.clone().unwrap_or_default(),
                    dd.calendar.clone(),
                ));
            }
            if let Some(s) = self.realm.string_value(h) {
                return self.zdt_from_string(&s, options);
            }
            if self.is_object_value(item) {
                return self.zdt_from_bag(h, options);
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.ZonedDateTime"))
    }

    /// Like [`Self::interpret_zdt`] but with no options (used by equals/compare).
    fn resolve_zdt(&mut self, item: NanBox) -> Result<(i128, String, String), ExecError> {
        self.interpret_zdt(item, NanBox::undefined())
    }

    /// Builds a ZonedDateTime from a property bag (`year`, …, `timeZone`, `offset`).
    ///
    /// Fields are read + coerced in the alphabetical order the spec's
    /// `PrepareTemporalFields` prescribes (each `ToIntegerWithTruncation` /
    /// `ToPrimitiveAndRequireString` observable at read time); options follow, and
    /// only then does algorithmic (range/suitability) validation run.
    fn zdt_from_bag(
        &mut self,
        h: Handle,
        options: NanBox,
    ) -> Result<(i128, String, String), ExecError> {
        // calendar (a String naming a calendar, or a Temporal object's calendar).
        let calendar = match self.zdt_field(h, "calendar")? {
            Some(v) => self.zdt_validate_calendar_field(v)?,
            None => String::from("iso8601"),
        };
        if !tcal::is_iso(&calendar) {
            return self.zdt_from_bag_cal(h, &calendar, options);
        }
        let day = self.read_int_field(h, "day")?;
        let hour = self.read_int_field(h, "hour")?;
        let us = self.read_int_field(h, "microsecond")?;
        let ms = self.read_int_field(h, "millisecond")?;
        let minute = self.read_int_field(h, "minute")?;
        let month = self.read_int_field(h, "month")?;
        let month_code = self.read_month_code_field(h)?;
        let ns = self.read_int_field(h, "nanosecond")?;
        let offset_ns = match self.zdt_field(h, "offset")? {
            Some(v) => Some(self.zdt_read_offset_field(v)?),
            None => None,
        };
        let second = self.read_int_field(h, "second")?;
        let tz_val = self.zdt_field(h, "timeZone")?;
        let year = self.read_int_field(h, "year")?;

        // Options (read order: disambiguation, offset, overflow).
        let opts = self.zdt_options(options)?;
        self.zdt_disambiguation(opts)?;
        let offset_opt = self.zdt_offset_option(opts)?;
        let overflow = self.zdt_overflow(opts)?;

        // Algorithmic validation (required fields → month suitability → time zone).
        let Some(year) = year else {
            return Err(self.type_error("year is required"));
        };
        let Some(day) = day else {
            return Err(self.type_error("day is required"));
        };
        let month_num = self.combine_month(month, month_code)?;
        if month_num < 1 || day < 1 {
            return Err(self.zdt_range("month and day must be positive"));
        }
        let Some(tz_val) = tz_val else {
            return Err(self.type_error("timeZone is required"));
        };
        let tz = self.tz_from_value(tz_val)?;

        let date = iso::regulate_iso_date(
            i32::try_from(year).map_err(|_| self.zdt_range("year out of range"))?,
            month_num,
            day,
            overflow,
        )
        .ok_or_else(|| self.zdt_range("invalid ISO date"))?;
        let time = iso::regulate_iso_time(
            hour.unwrap_or(0),
            minute.unwrap_or(0),
            second.unwrap_or(0),
            ms.unwrap_or(0),
            us.unwrap_or(0),
            ns.unwrap_or(0),
            overflow,
        )
        .ok_or_else(|| self.zdt_range("invalid ISO time"))?;

        let wall = iso_to_epoch_days(date) as i128 * iso::NS_PER_DAY + time_to_nanos(time);
        let epoch = self.resolve_epoch(&tz, wall, offset_ns, false, offset_opt)?;
        Ok((epoch, tz, calendar))
    }

    /// The non-ISO property-bag path (`CalendarDateFromFields` for the date, then
    /// the wall-clock → instant disambiguation). Reads the calendar date fields
    /// (`day`/`era`/`eraYear`/`month`/`monthCode`/`year`) + time/offset/timeZone in
    /// the alphabetical order the spec prescribes and routes the date portion
    /// through the calendar abstraction layer.
    fn zdt_from_bag_cal(
        &mut self,
        h: Handle,
        calendar: &str,
        options: NanBox,
    ) -> Result<(i128, String, String), ExecError> {
        // Alphabetical read order: day, era, eraYear, hour, microsecond,
        // millisecond, minute, month, monthCode, nanosecond, offset, second,
        // timeZone, year.
        let day = self.read_int_field(h, "day")?;
        let era = match self.zdt_field(h, "era")? {
            Some(v) => Some(self.coerce_to_string(v)?),
            None => None,
        };
        let era_year = self.read_int_field(h, "eraYear")?;
        let hour = self.read_int_field(h, "hour")?;
        let us = self.read_int_field(h, "microsecond")?;
        let ms = self.read_int_field(h, "millisecond")?;
        let minute = self.read_int_field(h, "minute")?;
        let month = self.read_int_field(h, "month")?;
        let month_code = self.zdt_read_month_code_str(h)?;
        let ns = self.read_int_field(h, "nanosecond")?;
        let offset_ns = match self.zdt_field(h, "offset")? {
            Some(v) => Some(self.zdt_read_offset_field(v)?),
            None => None,
        };
        let second = self.read_int_field(h, "second")?;
        let tz_val = self.zdt_field(h, "timeZone")?;
        let year = self.read_int_field(h, "year")?;

        // Options (read order: disambiguation, offset, overflow).
        let opts = self.zdt_options(options)?;
        self.zdt_disambiguation(opts)?;
        let offset_opt = self.zdt_offset_option(opts)?;
        let overflow = self.zdt_overflow(opts)?;

        let Some(day) = day else {
            return Err(self.type_error("day is required"));
        };
        if month.is_none() && month_code.is_none() {
            return Err(self.type_error("month or monthCode is required"));
        }
        let Some(tz_val) = tz_val else {
            return Err(self.type_error("timeZone is required"));
        };
        let tz = self.tz_from_value(tz_val)?;

        let input = tcal::FieldsInput {
            era,
            era_year,
            year,
            month,
            month_code,
            day,
        };
        let date = self.zdt_cal_fields_to_iso(calendar, &input, overflow)?;
        let time = iso::regulate_iso_time(
            hour.unwrap_or(0),
            minute.unwrap_or(0),
            second.unwrap_or(0),
            ms.unwrap_or(0),
            us.unwrap_or(0),
            ns.unwrap_or(0),
            overflow,
        )
        .ok_or_else(|| self.zdt_range("invalid ISO time"))?;

        let wall = iso_to_epoch_days(date) as i128 * iso::NS_PER_DAY + time_to_nanos(time);
        let epoch = self.resolve_epoch(&tz, wall, offset_ns, false, offset_opt)?;
        Ok((epoch, tz, String::from(calendar)))
    }

    /// Runs [`tcal::fields_to_iso`], mapping its error to the right exception.
    fn zdt_cal_fields_to_iso(
        &mut self,
        calendar: &str,
        input: &tcal::FieldsInput,
        overflow: Overflow,
    ) -> Result<IsoDate, ExecError> {
        match tcal::fields_to_iso(calendar, input, overflow) {
            Ok(d) => Ok(d),
            Err(tcal::CalError::Range(m)) => Err(self.zdt_range(&m)),
            Err(tcal::CalError::MissingFields(m)) => Err(self.type_error(&m)),
        }
    }

    /// Reads a `monthCode` field as its raw well-formed string (for the non-ISO
    /// path, where suitability is judged by the calendar layer).
    fn zdt_read_month_code_str(&mut self, h: Handle) -> Result<Option<String>, ExecError> {
        let Some(v) = self.zdt_field(h, "monthCode")? else {
            return Ok(None);
        };
        let prim = self.coerce_primitive(v, "string")?;
        let Some(s) = prim
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|x| self.realm.string_value(x))
        else {
            return Err(self.type_error("monthCode must be a string"));
        };
        // Well-formedness only (M + two digits + optional L).
        if parse_month_code(&s).is_none() {
            return Err(self.zdt_range("invalid monthCode"));
        }
        Ok(Some(s))
    }

    /// Reads an integer field (`ToIntegerWithTruncation`) inline; `None` if absent.
    fn read_int_field(&mut self, h: Handle, key: &str) -> Result<Option<i64>, ExecError> {
        match self.zdt_field(h, key)? {
            Some(v) => Ok(Some(self.zdt_to_int(v)?)),
            None => Ok(None),
        }
    }

    /// Reads the `monthCode` field inline (`ToPrimitiveAndRequireString` + syntax
    /// check); suitability (ISO range / no-leap) is validated later.
    fn read_month_code_field(&mut self, h: Handle) -> Result<Option<(i64, bool)>, ExecError> {
        let Some(v) = self.zdt_field(h, "monthCode")? else {
            return Ok(None);
        };
        let prim = self.coerce_primitive(v, "string")?;
        let Some(s) = prim
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|x| self.realm.string_value(x))
        else {
            return Err(self.type_error("monthCode must be a string"));
        };
        let mc = parse_month_code(&s).ok_or_else(|| self.zdt_range("invalid monthCode"))?;
        Ok(Some(mc))
    }

    /// Combines already-coerced `month`/`monthCode`, validating suitability.
    fn combine_month(
        &mut self,
        month: Option<i64>,
        code: Option<(i64, bool)>,
    ) -> Result<i64, ExecError> {
        let coded = match code {
            Some((c, leap)) => {
                if leap || !(1..=12).contains(&c) {
                    return Err(self.zdt_range("monthCode not valid for the ISO calendar"));
                }
                Some(c)
            }
            None => None,
        };
        match (month, coded) {
            (Some(a), Some(b)) if a != b => Err(self.zdt_range("month and monthCode disagree")),
            (Some(a), _) => Ok(a),
            (None, Some(b)) => Ok(b),
            (None, None) => Err(self.type_error("month or monthCode is required")),
        }
    }

    /// Resolves a `timeZone` property value (a string or an object with a timeZone).
    fn tz_from_value(&mut self, v: NanBox) -> Result<String, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if let Some(dd) = self.realm.temporal_at(h)
                && dd.kind == TemporalKind::ZonedDateTime
            {
                return Ok(dd.tz.clone().unwrap_or_default());
            }
            if let Some(s) = self.realm.string_value(h) {
                return self.tz_from_string(&s);
            }
            if self.is_object_value(v)
                && let Some(inner) = self.zdt_field(h, "timeZone")?
            {
                // Nested { timeZone } — read once (not recursively per spec, but
                // pragmatic).
                if let Some(s) = inner
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|x| self.realm.string_value(x))
                {
                    return self.tz_from_string(&s);
                }
            }
        }
        Err(self.type_error("invalid time zone"))
    }

    /// Builds a ZonedDateTime from an ISO string with a `[TimeZone]` annotation.
    fn zdt_from_string(
        &mut self,
        s: &str,
        options: NanBox,
    ) -> Result<(i128, String, String), ExecError> {
        let p = parse_zdt_string(s).ok_or_else(|| self.zdt_range("invalid ISO string"))?;
        let tz = self.parse_tz_identifier(&p.tz)?;
        // The `[u-ca=…]` annotation (canonicalized) supplies the calendar id.
        let calendar = match p.cal.as_deref() {
            Some(c) => self.zdt_canonicalize_calendar(c)?,
            None => String::from("iso8601"),
        };

        let opts = self.zdt_options(options)?;
        self.zdt_disambiguation(opts)?;
        let offset_opt = self.zdt_offset_option(opts)?;
        self.zdt_overflow(opts)?;

        let wall = iso_to_epoch_days(p.date) as i128 * iso::NS_PER_DAY + time_to_nanos(p.time);
        if !(iso::MIN_EPOCH_NS - iso::NS_PER_DAY..=iso::MAX_EPOCH_NS + iso::NS_PER_DAY)
            .contains(&wall)
        {
            return Err(self.zdt_range("date-time is outside the representable range"));
        }
        // offset_ns from a numeric offset; z handled separately.
        let offset_ns = if p.z { None } else { p.offset_ns };
        let epoch = self.resolve_epoch(&tz, wall, offset_ns, p.z, offset_opt)?;
        Ok((epoch, tz, calendar))
    }

    /// `InterpretISODateTimeOffset`: turns a wall time + (optional) offset/`Z` into
    /// an exact instant, honouring the `offset` reconciliation option.
    fn resolve_epoch(
        &mut self,
        tz: &str,
        wall_ns: i128,
        offset_ns: Option<i128>,
        has_z: bool,
        offset_opt: OffsetOpt,
    ) -> Result<i128, ExecError> {
        let epoch = if has_z {
            wall_ns
        } else if let Some(off) = offset_ns {
            let candidate = wall_ns - off;
            match offset_opt {
                OffsetOpt::Use => candidate,
                OffsetOpt::Ignore => wall_to_epoch(tz, wall_ns),
                OffsetOpt::Prefer | OffsetOpt::Reject => {
                    if tz_offset_at(tz, candidate) == off {
                        candidate
                    } else if offset_opt == OffsetOpt::Prefer {
                        wall_to_epoch(tz, wall_ns)
                    } else {
                        return Err(self.zdt_range("offset does not match the time zone"));
                    }
                }
            }
        } else {
            wall_to_epoch(tz, wall_ns)
        };
        if !valid_epoch(epoch) {
            return Err(self.zdt_range("resulting instant is out of range"));
        }
        Ok(epoch)
    }

    // --- with / withPlainTime / withTimeZone ------------------------------

    fn zdt_with(
        &mut self,
        data: &TemporalData,
        fields: NanBox,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        let is_temporal = fields
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.temporal_at(h).is_some());
        if !self.is_object_value(fields) || is_temporal {
            return Err(self.type_error("with() requires a plain fields object"));
        }
        let h = fields.as_handle().map(Handle::from_raw).unwrap();
        if self.zdt_field(h, "calendar")?.is_some() {
            return Err(self.type_error("with() fields must not have a calendar property"));
        }
        if self.zdt_field(h, "timeZone")?.is_some() {
            return Err(self.type_error("with() fields must not have a timeZone property"));
        }
        let tz = self.zdt_tz(data);
        let (cd, ct) = local_of(&tz, data.epoch_ns);
        let cur_offset = tz_offset_at(&tz, data.epoch_ns);
        if !tcal::is_iso(&data.calendar) {
            return self.zdt_with_cal(data, h, &tz, cd, ct, cur_offset, options);
        }

        // Read + coerce the partial fields inline in alphabetical order.
        let day = self.read_int_field(h, "day")?;
        let hour = self.read_int_field(h, "hour")?;
        let us = self.read_int_field(h, "microsecond")?;
        let ms = self.read_int_field(h, "millisecond")?;
        let minute = self.read_int_field(h, "minute")?;
        let month = self.read_int_field(h, "month")?;
        let month_code = self.read_month_code_field(h)?;
        let ns = self.read_int_field(h, "nanosecond")?;
        let offset_field = match self.zdt_field(h, "offset")? {
            Some(v) => Some(self.zdt_read_offset_field(v)?),
            None => None,
        };
        let second = self.read_int_field(h, "second")?;
        let year = self.read_int_field(h, "year")?;
        let any = day.is_some()
            || hour.is_some()
            || us.is_some()
            || ms.is_some()
            || minute.is_some()
            || month.is_some()
            || month_code.is_some()
            || ns.is_some()
            || offset_field.is_some()
            || second.is_some()
            || year.is_some();
        if !any {
            return Err(self.type_error("with() requires at least one recognised field"));
        }

        // An explicitly-supplied non-positive month/day is rejected before the
        // options object is examined (monthCode *suitability* is checked later).
        if month.is_some_and(|m| m < 1) || day.is_some_and(|d| d < 1) {
            return Err(self.zdt_range("month and day must be positive"));
        }

        // Options (read order: disambiguation, offset, overflow).
        let opts = self.zdt_options(options)?;
        self.zdt_disambiguation(opts)?;
        let offset_opt = self.zdt_offset_option_default(opts, OffsetOpt::Prefer)?;
        let overflow = self.zdt_overflow(opts)?;

        let month_num = if month.is_none() && month_code.is_none() {
            i64::from(cd.month)
        } else {
            self.combine_month(month, month_code)?
        };
        let year = year.unwrap_or(i64::from(cd.year));
        let day = day.unwrap_or(i64::from(cd.day));
        if month_num < 1 || day < 1 {
            return Err(self.zdt_range("month and day must be positive"));
        }
        let hour = hour.unwrap_or(i64::from(ct.hour));
        let minute = minute.unwrap_or(i64::from(ct.minute));
        let second = second.unwrap_or(i64::from(ct.second));
        let ms = ms.unwrap_or(i64::from(ct.millisecond));
        let us = us.unwrap_or(i64::from(ct.microsecond));
        let ns = ns.unwrap_or(i64::from(ct.nanosecond));
        let offset_ns = Some(offset_field.unwrap_or(cur_offset));

        let date = iso::regulate_iso_date(
            i32::try_from(year).map_err(|_| self.zdt_range("year out of range"))?,
            month_num,
            day,
            overflow,
        )
        .ok_or_else(|| self.zdt_range("invalid ISO date"))?;
        let time = iso::regulate_iso_time(hour, minute, second, ms, us, ns, overflow)
            .ok_or_else(|| self.zdt_range("invalid ISO time"))?;
        let wall = iso_to_epoch_days(date) as i128 * iso::NS_PER_DAY + time_to_nanos(time);
        let epoch = self.resolve_epoch(&tz, wall, offset_ns, false, offset_opt)?;
        Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
    }

    /// The non-ISO `with` path: merges the provided calendar/time fields over the
    /// receiver's existing wall-clock fields, re-derives the ISO date through the
    /// calendar abstraction layer, then applies the wall-clock → instant
    /// disambiguation.
    #[allow(clippy::too_many_arguments)]
    fn zdt_with_cal(
        &mut self,
        data: &TemporalData,
        h: Handle,
        tz: &str,
        cd: IsoDate,
        ct: IsoTime,
        cur_offset: i128,
        options: NanBox,
    ) -> Result<NanBox, ExecError> {
        let cal = data.calendar.as_str();
        let existing = tcal::iso_to_fields(cal, cd);

        // Alphabetical read order: day, era, eraYear, hour, microsecond,
        // millisecond, minute, month, monthCode, nanosecond, offset, second, year.
        let day = self.read_int_field(h, "day")?;
        let era = match self.zdt_field(h, "era")? {
            Some(v) => Some(self.coerce_to_string(v)?),
            None => None,
        };
        let era_year = self.read_int_field(h, "eraYear")?;
        let hour = self.read_int_field(h, "hour")?;
        let us = self.read_int_field(h, "microsecond")?;
        let ms = self.read_int_field(h, "millisecond")?;
        let minute = self.read_int_field(h, "minute")?;
        let month = self.read_int_field(h, "month")?;
        let month_code = self.zdt_read_month_code_str(h)?;
        let ns = self.read_int_field(h, "nanosecond")?;
        let offset_field = match self.zdt_field(h, "offset")? {
            Some(v) => Some(self.zdt_read_offset_field(v)?),
            None => None,
        };
        let second = self.read_int_field(h, "second")?;
        let year = self.read_int_field(h, "year")?;
        let any = day.is_some()
            || era.is_some()
            || era_year.is_some()
            || hour.is_some()
            || us.is_some()
            || ms.is_some()
            || minute.is_some()
            || month.is_some()
            || month_code.is_some()
            || ns.is_some()
            || offset_field.is_some()
            || second.is_some()
            || year.is_some();
        if !any {
            return Err(self.type_error("with() requires at least one recognised field"));
        }
        if month.is_some_and(|m| m < 1) || day.is_some_and(|d| d < 1) {
            return Err(self.zdt_range("month and day must be positive"));
        }

        // Options (read order: disambiguation, offset, overflow).
        let opts = self.zdt_options(options)?;
        self.zdt_disambiguation(opts)?;
        let offset_opt = self.zdt_offset_option_default(opts, OffsetOpt::Prefer)?;
        let overflow = self.zdt_overflow(opts)?;

        // Merge date fields: an explicit year (or era+eraYear) wins; otherwise keep
        // the receiver's year. Prefer monthCode to preserve leap months.
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
        let date = self.zdt_cal_fields_to_iso(cal, &input, overflow)?;

        let hour = hour.unwrap_or(i64::from(ct.hour));
        let minute = minute.unwrap_or(i64::from(ct.minute));
        let second = second.unwrap_or(i64::from(ct.second));
        let ms = ms.unwrap_or(i64::from(ct.millisecond));
        let us = us.unwrap_or(i64::from(ct.microsecond));
        let ns = ns.unwrap_or(i64::from(ct.nanosecond));
        let offset_ns = Some(offset_field.unwrap_or(cur_offset));
        let time = iso::regulate_iso_time(hour, minute, second, ms, us, ns, overflow)
            .ok_or_else(|| self.zdt_range("invalid ISO time"))?;

        let wall = iso_to_epoch_days(date) as i128 * iso::NS_PER_DAY + time_to_nanos(time);
        let epoch = self.resolve_epoch(tz, wall, offset_ns, false, offset_opt)?;
        Ok(self.make_zdt_cal(epoch, tz.to_string(), data.calendar.clone()))
    }

    fn zdt_offset_option_default(
        &mut self,
        opts: Option<Handle>,
        default: OffsetOpt,
    ) -> Result<OffsetOpt, ExecError> {
        Ok(
            match self
                .zdt_str_option(opts, "offset", &["prefer", "use", "ignore", "reject"])?
                .as_deref()
            {
                Some("use") => OffsetOpt::Use,
                Some("ignore") => OffsetOpt::Ignore,
                Some("prefer") => OffsetOpt::Prefer,
                Some("reject") => OffsetOpt::Reject,
                _ => default,
            },
        )
    }

    fn zdt_with_plain_time(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
    ) -> Result<NanBox, ExecError> {
        let tz = self.zdt_tz(data);
        let (cd, _) = local_of(&tz, data.epoch_ns);
        let time = if arg.is_undefined() {
            IsoTime::default()
        } else {
            self.zdt_to_time(arg)?
        };
        let wall = iso_to_epoch_days(cd) as i128 * iso::NS_PER_DAY + time_to_nanos(time);
        let epoch = wall_to_epoch(&tz, wall);
        if !valid_epoch(epoch) {
            return Err(self.zdt_range("resulting instant is out of range"));
        }
        Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
    }

    fn zdt_to_time(&mut self, item: NanBox) -> Result<IsoTime, ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(dd) = self.realm.temporal_at(h) {
                return match dd.kind {
                    TemporalKind::PlainTime | TemporalKind::PlainDateTime => Ok(dd.time),
                    TemporalKind::ZonedDateTime => {
                        Ok(local_of(&dd.tz.clone().unwrap_or_default(), dd.epoch_ns).1)
                    }
                    _ => Err(self.type_error("expected a PlainTime")),
                };
            }
            if let Some(s) = self.realm.string_value(h) {
                return parse_plaintime_string(&s)
                    .ok_or_else(|| self.zdt_range("invalid PlainTime string"));
            }
            if self.is_object_value(item) {
                // PrepareTemporalFields order: hour, microsecond, millisecond,
                // minute, nanosecond, second (alphabetical, coerced inline).
                let hour = self.read_int_field(h, "hour")?;
                let us = self.read_int_field(h, "microsecond")?;
                let ms = self.read_int_field(h, "millisecond")?;
                let minute = self.read_int_field(h, "minute")?;
                let ns = self.read_int_field(h, "nanosecond")?;
                let second = self.read_int_field(h, "second")?;
                if hour.is_none()
                    && minute.is_none()
                    && second.is_none()
                    && ms.is_none()
                    && us.is_none()
                    && ns.is_none()
                {
                    return Err(self.type_error("no time fields present"));
                }
                return iso::regulate_iso_time(
                    hour.unwrap_or(0),
                    minute.unwrap_or(0),
                    second.unwrap_or(0),
                    ms.unwrap_or(0),
                    us.unwrap_or(0),
                    ns.unwrap_or(0),
                    Overflow::Constrain,
                )
                .ok_or_else(|| self.zdt_range("invalid ISO time"));
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.PlainTime"))
    }

    fn zdt_with_time_zone(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
    ) -> Result<NanBox, ExecError> {
        let tz = self.tz_from_value(arg)?;
        Ok(self.make_zdt_cal(data.epoch_ns, tz, data.calendar.clone()))
    }

    // --- add / subtract ----------------------------------------------------

    fn zdt_add(
        &mut self,
        data: &TemporalData,
        dur_arg: NanBox,
        options: NanBox,
        sign: i64,
    ) -> Result<NanBox, ExecError> {
        let mut dur = self.zdt_to_duration(dur_arg)?;
        if sign < 0 {
            dur = negate_duration(dur);
        }
        let opts = self.zdt_options(options)?;
        let overflow = self.zdt_overflow(opts)?;
        let tz = self.zdt_tz(data);

        let epoch = if dur.years == 0 && dur.months == 0 && dur.weeks == 0 && dur.days == 0 {
            data.epoch_ns + dur.time_nanos()
        } else {
            let (d, t) = local_of(&tz, data.epoch_ns);
            // AddZonedDateTime: add the date parts to the wall-clock date *in the
            // calendar*. ISO takes the shared fast path (byte-for-byte); every other
            // calendar routes through CalendarDateAdd (variable month lengths / leap
            // months honoured).
            let new_date = if tcal::is_iso(&data.calendar) {
                iso::add_iso_date(
                    d,
                    dur.years as i64,
                    dur.months as i64,
                    dur.weeks as i64,
                    dur.days as i64,
                    overflow,
                )
                .ok_or_else(|| self.zdt_range("result out of range"))?
            } else {
                match tcal::calendar_date_add(
                    &data.calendar,
                    d,
                    dur.years as i64,
                    dur.months as i64,
                    dur.weeks as i64,
                    dur.days as i64,
                    overflow,
                ) {
                    Ok(r) => r,
                    Err(tcal::CalError::Range(m)) => return Err(self.zdt_range(&m)),
                    Err(tcal::CalError::MissingFields(m)) => return Err(self.type_error(&m)),
                }
            };
            let wall = iso_to_epoch_days(new_date) as i128 * iso::NS_PER_DAY + time_to_nanos(t);
            let intermediate = wall_to_epoch(&tz, wall);
            intermediate + dur.time_nanos()
        };
        if !valid_epoch(epoch) {
            return Err(self.zdt_range("result out of range"));
        }
        Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
    }

    fn zdt_to_duration(&mut self, item: NanBox) -> Result<DurationFields, ExecError> {
        if let Some(h) = item.as_handle().map(Handle::from_raw) {
            if let Some(dd) = self.realm.temporal_at(h) {
                return if dd.kind == TemporalKind::Duration {
                    Ok(dd.duration)
                } else {
                    Err(self.type_error("expected a Temporal.Duration"))
                };
            }
            if let Some(s) = self.realm.string_value(h) {
                return iso::parse_iso_duration(&s)
                    .ok_or_else(|| self.zdt_range("invalid duration string"));
            }
            if self.is_object_value(item) {
                return self.zdt_duration_bag(h);
            }
        }
        Err(self.type_error("cannot convert value to a Temporal.Duration"))
    }

    fn zdt_duration_bag(&mut self, h: Handle) -> Result<DurationFields, ExecError> {
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
            if let Some(v) = self.zdt_field(h, key)? {
                let num = self.coerce_to_number(v)?;
                let n = self.realm.to_number(num);
                if !n.is_finite() || n.fract() != 0.0 {
                    return Err(self.zdt_range("duration fields must be integers"));
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
            return Err(self.zdt_range("duration fields must share one sign"));
        }
        Ok(d)
    }

    // --- until / since -----------------------------------------------------

    fn zdt_diff(
        &mut self,
        data: &TemporalData,
        other: NanBox,
        options: NanBox,
        negate: bool,
    ) -> Result<NanBox, ExecError> {
        let (other_epoch, _other_tz, other_cal) = self.resolve_zdt(other)?;
        let cal = data.calendar.clone();
        // DifferenceTemporalZonedDateTime enforces CalendarEquals before reading the
        // options bag. The ISO fast path keeps its original calendar-agnostic
        // behaviour; a non-ISO receiver requires both operands to share a calendar.
        if !tcal::is_iso(&cal) && other_cal != cal {
            return Err(self
                .zdt_range("cannot compute the difference between dates of different calendars"));
        }
        let tz = self.zdt_tz(data);
        let opts = self.zdt_options(options)?;
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
        let mut units_auto = units.to_vec();
        units_auto.push("auto");
        // GetDifferenceSettings read order: largestUnit, roundingIncrement,
        // roundingMode, smallestUnit.
        let largest_opt = self.zdt_str_option(opts, "largestUnit", &units_auto)?;
        let increment = self.zdt_rounding_increment(opts)?;
        let mut mode = self.zdt_rounding_mode(opts, RoundMode::Trunc)?;
        // `since` rounds the (other − receiver) difference with a negated mode, then
        // negates the result (NegateRoundingMode).
        if negate {
            mode = negate_round_mode(mode);
        }
        let smallest = match self.zdt_str_option(opts, "smallestUnit", &units)? {
            Some(s) => parse_unit(&s).unwrap_or(Unit::Nanosecond),
            None => Unit::Nanosecond,
        };
        let largest = match largest_opt {
            Some(s) if s != "auto" => parse_unit(&s).unwrap_or(Unit::Hour),
            _ => Unit::Hour.min(smallest),
        };
        if largest > smallest {
            return Err(self.zdt_range("largestUnit must be at least as large as smallestUnit"));
        }
        // ValidateTemporalRoundingIncrement (non-inclusive) for time smallestUnits:
        // the increment must divide evenly into, and be smaller than, the next
        // coarser unit (e.g. 11h or 24h are invalid; 29min is invalid).
        if smallest >= Unit::Hour {
            self.zdt_validate_increment(smallest, increment)?;
        }

        let mut dur = if largest >= Unit::Day {
            // Time-only difference (exact nanoseconds), with sign-aware rounding.
            let total = other_epoch - data.epoch_ns;
            let inc = unit_ns(smallest) * i128::from(increment.max(1));
            let rounded = round_signed(total, inc, mode);
            balance_datetime(rounded, largest)
        } else if largest == smallest
            && matches!(smallest, Unit::Year | Unit::Month | Unit::Week | Unit::Day)
        {
            // Calendar-unit rounding relative to the receiver (NudgeToCalendarUnit),
            // expressed in a single unit (largestUnit == smallestUnit).
            self.zdt_round_calendar(
                &cal,
                &tz,
                data.epoch_ns,
                other_epoch,
                smallest,
                increment,
                mode,
            )
        } else {
            // Calendar difference in the time zone's wall time (unrounded).
            let from = local_of(&tz, data.epoch_ns);
            let to = local_of(&tz, other_epoch);
            datetime_diff(&cal, from, to, largest)
        };
        if negate {
            dur = negate_duration(dur);
        }
        let data2 = TemporalData {
            kind: TemporalKind::Duration,
            duration: dur,
            ..Default::default()
        };
        Ok(self.zdt_alloc(data2, TemporalKind::Duration))
    }

    /// `NudgeToCalendarUnit` for a single calendar unit: the whole difference from
    /// `start_epoch` to `end_epoch` in `unit`s, rounded to `increment` under `mode`,
    /// with the fraction measured against the instant span of one `unit` step in the
    /// zone (so it is DST-aware).
    #[allow(clippy::too_many_arguments)]
    fn zdt_round_calendar(
        &self,
        cal: &str,
        tz: &str,
        start_epoch: i128,
        end_epoch: i128,
        unit: Unit,
        increment: i64,
        mode: RoundMode,
    ) -> DurationFields {
        let sign = (end_epoch - start_epoch).signum() as i64;
        let mut dur = DurationFields::default();
        if sign == 0 {
            return dur;
        }
        let (sd, st) = local_of(tz, start_epoch);
        let ed = local_of(tz, end_epoch).0;
        // Whole-unit initial guess from the date difference (calendar-aware for a
        // non-ISO calendar, so month lengths / leap months are honoured).
        let (dy, dm, dw, dd) = if tcal::is_iso(cal) {
            iso::difference_iso_date(sd, ed, unit)
        } else {
            let p = tcal::calendar_date_until(cal, sd, ed, unit);
            (p.years, p.months, p.weeks, p.days)
        };
        let mut r1 = match unit {
            Unit::Year => dy,
            Unit::Month => dm,
            Unit::Week => dw,
            _ => dd,
        };
        let add = |count: i64| -> i128 {
            let (y, m, w, d) = match unit {
                Unit::Year => (count, 0, 0, 0),
                Unit::Month => (0, count, 0, 0),
                Unit::Week => (0, 0, count, 0),
                _ => (0, 0, 0, count),
            };
            let nd = if tcal::is_iso(cal) {
                iso::add_iso_date(sd, y, m, w, d, Overflow::Constrain).unwrap_or(sd)
            } else {
                tcal::calendar_date_add(cal, sd, y, m, w, d, Overflow::Constrain).unwrap_or(sd)
            };
            let wall = iso_to_epoch_days(nd) as i128 * iso::NS_PER_DAY + time_to_nanos(st);
            wall_to_epoch(tz, wall)
        };
        let beyond = |e: i128| -> bool {
            if sign > 0 {
                e > end_epoch
            } else {
                e < end_epoch
            }
        };
        // Correct the guess: grow while the next step stays on the near side, then
        // shrink while the current step overshoots.
        for _ in 0..8 {
            if beyond(add(r1 + sign)) {
                break;
            }
            r1 += sign;
        }
        for _ in 0..8 {
            if beyond(add(r1)) {
                r1 -= sign;
            } else {
                break;
            }
        }
        let epoch1 = add(r1);
        let count = if epoch1 == end_epoch {
            r1
        } else {
            let epoch2 = add(r1 + sign);
            let den = (epoch2 - epoch1).abs().max(1);
            let num = (end_epoch - epoch1).abs();
            let x = i128::from(r1) * den + i128::from(sign) * num;
            let rounded = round_signed(x, i128::from(increment.max(1)) * den, mode);
            (rounded / den) as i64
        };
        match unit {
            Unit::Year => dur.years = i128::from(count),
            Unit::Month => dur.months = i128::from(count),
            Unit::Week => dur.weeks = i128::from(count),
            _ => dur.days = i128::from(count),
        }
        dur
    }

    // --- round -------------------------------------------------------------

    fn zdt_round(&mut self, data: &TemporalData, options: NanBox) -> Result<NanBox, ExecError> {
        if options.is_undefined() {
            return Err(self.type_error("round() requires a roundTo argument"));
        }
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
                .filter(|u| *u >= Unit::Day && *u <= Unit::Nanosecond)
                .ok_or_else(|| self.zdt_range("invalid smallestUnit"))?;
            (u, 1_i64, RoundMode::HalfExpand)
        } else {
            let opts = self.zdt_options(options)?;
            let increment = self.zdt_rounding_increment(opts)?;
            let mode = self.zdt_rounding_mode(opts, RoundMode::HalfExpand)?;
            let u = match self.zdt_str_option(opts, "smallestUnit", &units)? {
                Some(s) => parse_unit(&s).unwrap_or(Unit::Nanosecond),
                None => return Err(self.zdt_range("round() requires a smallestUnit")),
            };
            (u, increment, mode)
        };
        self.zdt_validate_increment(smallest, increment)?;

        let tz = self.zdt_tz(data);
        let epoch = if smallest == Unit::Day {
            let (d, _) = local_of(&tz, data.epoch_ns);
            let start = wall_to_epoch(&tz, iso_to_epoch_days(d) as i128 * iso::NS_PER_DAY);
            let next = wall_to_epoch(&tz, (iso_to_epoch_days(d) + 1) as i128 * iso::NS_PER_DAY);
            if !valid_epoch(start) || !valid_epoch(next) {
                return Err(self.zdt_range("day boundary is out of range"));
            }
            let day_len = next - start;
            let progress = data.epoch_ns - start;
            let rounded = iso::round_to_increment(progress, day_len, mode);
            start + rounded
        } else {
            let (d, t) = local_of(&tz, data.epoch_ns);
            let inc = unit_ns(smallest) * i128::from(increment.max(1));
            let rounded = iso::round_to_increment(time_to_nanos(t), inc, mode);
            let (carry, t2) = balance_time_from_nanos(rounded);
            let d2 = epoch_days_to_iso(iso_to_epoch_days(d) + carry);
            let wall = iso_to_epoch_days(d2) as i128 * iso::NS_PER_DAY + time_to_nanos(t2);
            wall_to_epoch(&tz, wall)
        };
        if !valid_epoch(epoch) {
            return Err(self.zdt_range("rounded instant is out of range"));
        }
        Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
    }

    fn zdt_validate_increment(&mut self, unit: Unit, increment: i64) -> Result<(), ExecError> {
        let dividend: i64 = match unit {
            Unit::Day => {
                return if increment == 1 {
                    Ok(())
                } else {
                    Err(self.zdt_range("roundingIncrement must be 1 when smallestUnit is day"))
                };
            }
            Unit::Hour => 24,
            Unit::Minute | Unit::Second => 60,
            _ => 1000,
        };
        if increment >= dividend || dividend % increment != 0 {
            return Err(self.zdt_range("invalid roundingIncrement for the smallestUnit"));
        }
        Ok(())
    }

    // --- getTimeZoneTransition ---------------------------------------------

    fn zdt_get_transition(
        &mut self,
        data: &TemporalData,
        arg: NanBox,
    ) -> Result<NanBox, ExecError> {
        // The direction is a required option: a string smallestUnit-style value or
        // an options bag with a `direction` property.
        let direction = if arg.is_undefined() {
            return Err(self.type_error("getTimeZoneTransition() requires a direction"));
        } else if let Some(s) = arg
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
        {
            s
        } else if self.is_object_value(arg) {
            let h = arg.as_handle().map(Handle::from_raw).unwrap();
            match self.zdt_field(h, "direction")? {
                Some(v) => self.coerce_to_string(v)?,
                None => return Err(self.zdt_range("direction is required")),
            }
        } else {
            return Err(self.type_error("invalid direction"));
        };
        let next = match direction.as_str() {
            "next" => true,
            "previous" => false,
            _ => return Err(self.zdt_range("direction must be \"next\" or \"previous\"")),
        };
        let tz = self.zdt_tz(data);
        // Fixed-offset zones (and UTC) have no transitions.
        if parse_offset_id(&tz).is_some() {
            return Ok(NanBox::null());
        }
        let Ok(zone) = timezone_data::load(&tz) else {
            return Ok(NanBox::null());
        };
        let secs = data.epoch_ns.div_euclid(iso::NS_PER_SEC) as i64;
        let mut best: Option<i64> = None;
        for tr in zone.transitions() {
            if next {
                if tr.when > secs && best.is_none_or(|b| tr.when < b) {
                    best = Some(tr.when);
                }
            } else if tr.when < secs && best.is_none_or(|b| tr.when > b) {
                best = Some(tr.when);
            }
        }
        match best {
            Some(w) => {
                let epoch = i128::from(w) * iso::NS_PER_SEC;
                Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
            }
            None => Ok(NanBox::null()),
        }
    }

    // --- toString ----------------------------------------------------------

    fn zdt_to_string(&mut self, data: &TemporalData, options: NanBox) -> Result<NanBox, ExecError> {
        let opts = self.zdt_options(options)?;
        // Options are read in alphabetical order: calendarName,
        // fractionalSecondDigits, offset, roundingMode, smallestUnit, timeZoneName.
        let cal = self
            .zdt_str_option(
                opts,
                "calendarName",
                &["auto", "always", "never", "critical"],
            )?
            .unwrap_or_else(|| String::from("auto"));
        let frac = self.zdt_frac_digits(opts)?;
        let offset_mode = self
            .zdt_str_option(opts, "offset", &["auto", "never"])?
            .unwrap_or_else(|| String::from("auto"));
        let mode = self.zdt_rounding_mode(opts, RoundMode::Trunc)?;
        // smallestUnit is READ (coerced to a raw string, accepting any unit name)
        // before timeZoneName; whether it is a time unit is validated only after
        // every option has been read (all options are read before validation).
        let smallest_raw = match opts {
            Some(h) => {
                let v = self.read_member(h, "smallestUnit")?;
                if v.is_undefined() {
                    None
                } else {
                    Some(self.coerce_to_string(v)?)
                }
            }
            None => None,
        };
        let tzname = self
            .zdt_str_option(opts, "timeZoneName", &["auto", "never", "critical"])?
            .unwrap_or_else(|| String::from("auto"));
        let smallest = match smallest_raw {
            None => None,
            Some(s) => {
                let u = parse_unit(&s).ok_or_else(|| self.zdt_range("invalid smallestUnit"))?;
                // toString allows only minute..nanosecond (hour and coarser are
                // date/too-coarse units here).
                if u <= Unit::Hour {
                    return Err(self.zdt_range("smallestUnit must be minute..nanosecond"));
                }
                Some(u)
            }
        };

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

        let tz = self.zdt_tz(data);
        let offset_ns = tz_offset_at(&tz, data.epoch_ns);
        // Round the wall time, carrying a whole-day overflow into the date.
        let (d0, t0) = local_of(&tz, data.epoch_ns);
        let rounded = iso::round_to_increment(time_to_nanos(t0), inc_ns, mode);
        let (carry, time) = balance_time_from_nanos(rounded);
        let date = epoch_days_to_iso(iso_to_epoch_days(d0) + carry);

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
        if offset_mode != "never" {
            out.push_str(&format_offset(offset_ns));
        }
        match tzname.as_str() {
            "never" => {}
            "critical" => {
                out.push_str("[!");
                out.push_str(&tz);
                out.push(']');
            }
            _ => {
                out.push('[');
                out.push_str(&tz);
                out.push(']');
            }
        }
        let cal_id = data.calendar.as_str();
        match cal.as_str() {
            "always" => out.push_str(&alloc::format!("[u-ca={cal_id}]")),
            "critical" => out.push_str(&alloc::format!("[!u-ca={cal_id}]")),
            "auto" if !tcal::is_iso(cal_id) => out.push_str(&alloc::format!("[u-ca={cal_id}]")),
            _ => {}
        }
        Ok(self.new_str(&out))
    }

    fn zdt_frac_digits(&mut self, opts: Option<Handle>) -> Result<Option<u8>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, "fractionalSecondDigits")?;
        if v.is_undefined() {
            return Ok(None);
        }
        if v.is_number() {
            let n = v.as_number().unwrap_or(f64::NAN);
            if n.is_nan() {
                return Err(self.zdt_range("fractionalSecondDigits out of range"));
            }
            let f = n.floor();
            if !(0.0..=9.0).contains(&f) {
                return Err(self.zdt_range("fractionalSecondDigits out of range"));
            }
            return Ok(Some(f as u8));
        }
        let s = self.coerce_to_string(v)?;
        if s == "auto" {
            Ok(None)
        } else {
            Err(self.zdt_range("invalid fractionalSecondDigits"))
        }
    }
}

/// Whether the trailing UTC offset of a datetime string (no annotation) carries a
/// sub-minute component (a seconds field), which disqualifies it as a time-zone
/// identifier even when its value is a whole minute (e.g. `-07:00:00`).
fn dt_offset_subminute(s: &str) -> bool {
    let Some(tpos) = s.find(['T', 't']) else {
        return false;
    };
    let tail = &s[tpos + 1..];
    let Some(op) = tail.find(['+', '-']) else {
        return false;
    };
    let off = &tail[op + 1..];
    let off = off.split('[').next().unwrap_or(off);
    if off.contains('.') || off.contains(',') {
        return true;
    }
    let colons = off.matches(':').count();
    if colons >= 2 {
        return true;
    }
    if colons == 0 {
        let digits = off.chars().take_while(char::is_ascii_digit).count();
        return digits > 4;
    }
    false
}

// ---------------------------------------------------------------------------
// Strict ISO-8601 ZonedDateTime-string parser
// ---------------------------------------------------------------------------
//
// A ZonedDateTime string is a Temporal date-time string that *must* carry a
// `[TimeZone]` annotation, and — when it has a `Z`/offset — also a time. The
// conformance corpus checks many rejection cases the lenient shared parser
// accepts (basic/extended-inconsistent fields, >9 fractional digits, multiple
// time-zone annotations, U+2212 minus sign, sub-minute annotation offsets, …), so
// ZonedDateTime uses this self-contained strict parser.

/// The parsed pieces of a ZonedDateTime string.
struct ParsedZdt {
    date: IsoDate,
    time: IsoTime,
    offset_ns: Option<i128>,
    z: bool,
    tz: String,
    /// The (raw, un-canonicalized) first `[u-ca=…]` calendar annotation, if any.
    cal: Option<String>,
}

struct Zp<'s> {
    b: &'s [u8],
    i: usize,
}

impl Zp<'_> {
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

/// Parses a strict ZonedDateTime string; `None` if malformed or missing a
/// required time-zone annotation.
fn parse_zdt_string(s: &str) -> Option<ParsedZdt> {
    // The Unicode MINUS SIGN (U+2212) is never accepted.
    if s.as_bytes().windows(3).any(|w| w == [0xE2, 0x88, 0x92]) {
        return None;
    }
    let mut p = Zp {
        b: s.as_bytes(),
        i: 0,
    };
    let date = zp_date(&mut p)?;
    let mut time = IsoTime::default();
    let mut offset = None;
    let mut z = false;
    if p.eat(b'T') || p.eat(b't') || p.eat(b' ') {
        time = zp_time(&mut p)?;
        let (off, is_z) = zp_offset(&mut p)?;
        offset = off;
        z = is_z;
    }
    let (tz, cal) = zp_annotations(&mut p)?;
    if p.i != p.b.len() {
        return None;
    }
    Some(ParsedZdt {
        date,
        time,
        offset_ns: offset,
        z,
        tz,
        cal,
    })
}

/// Parses a strict Temporal PlainTime string: a bare time or a date-time string
/// (time extracted), optionally with a numeric UTC offset (any precision, ignored)
/// and `[…]` annotations. A `Z` designator, a date-only string, the U+2212 minus
/// sign, >9 fractional digits, or bad annotations all reject.
fn parse_plaintime_string(s: &str) -> Option<IsoTime> {
    // Delegate to the shared ISO parser, which enforces the time-designator
    // disambiguation rule (a bare time that is also a valid month-day/year-month
    // needs a `T` prefix), strict `[...]` annotation validity, and rejection of
    // the U+2212 minus sign. A `Z`/UTC designator is not a valid PlainTime.
    let p = crate::temporal_iso::parse_iso_time_string(s)?;
    if p.z {
        return None;
    }
    p.time
}

fn zp_date(p: &mut Zp) -> Option<IsoDate> {
    let year = if p.eat(b'+') {
        p.digits(6)?
    } else if p.eat(b'-') {
        let y = p.digits(6)?;
        if y == 0 {
            return None;
        }
        -y
    } else {
        p.digits(4)?
    };
    let extended = p.eat(b'-');
    let month = p.digits(2)?;
    if extended && !p.eat(b'-') {
        return None;
    }
    if !extended && p.peek() == Some(b'-') {
        return None;
    }
    let day = p.digits(2)?;
    iso::regulate_iso_date(year as i32, month, day, Overflow::Reject)
}

fn zp_time(p: &mut Zp) -> Option<IsoTime> {
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
        second: second.min(59) as u8,
        millisecond: (frac / 1_000_000) as u16,
        microsecond: (frac / 1_000 % 1_000) as u16,
        nanosecond: (frac % 1_000) as u16,
    })
}

/// A `Z`/`z` designator or a strict numeric offset (basic/extended-consistent),
/// returning `(offset_ns, is_z)`. Absence yields `(None, false)`.
fn zp_offset(p: &mut Zp) -> Option<(Option<i128>, bool)> {
    if p.eat(b'Z') || p.eat(b'z') {
        return Some((Some(0), true));
    }
    let neg = match p.peek() {
        Some(b'+') => false,
        Some(b'-') => true,
        _ => return Some((None, false)),
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
    let ns = i128::from(hour) * iso::NS_PER_HOUR
        + i128::from(minute) * iso::NS_PER_MINUTE
        + i128::from(second) * iso::NS_PER_SEC
        + i128::from(frac);
    Some((Some(if neg { -ns } else { ns }), false))
}

/// Parses the trailing `[…]` annotations, returning the (single) time-zone
/// annotation body plus the first `[u-ca=…]` calendar annotation value (raw).
/// Enforces the Temporal annotation rules.
fn zp_annotations(p: &mut Zp) -> Option<(String, Option<String>)> {
    let mut tz: Option<String> = None;
    let mut cal: Option<String> = None;
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
            return None;
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
                return None;
            }
            if key == "u-ca" {
                cal_count += 1;
                cal_critical |= critical;
                if cal.is_none() {
                    cal = Some(content[eq + 1..].to_string());
                }
            } else if critical {
                return None;
            }
        } else {
            // A time-zone annotation: at most one, before any key=value.
            if tz.is_some() || kv_seen || content.is_empty() {
                return None;
            }
            tz = Some(content.to_string());
        }
    }
    if cal_count > 1 && cal_critical {
        return None;
    }
    tz.map(|t| (t, cal))
}
