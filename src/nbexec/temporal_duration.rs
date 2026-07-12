//! `Temporal.Duration` — logic module. A fan-out unit: everything specific to
//! `Duration` lives here (its method/getter name tables plus the construct/
//! method/getter/static logic), so it can be implemented independently of the
//! other Temporal types and of the shared wiring in `temporal.rs`.
use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::temporal_iso::{
    DurationFields, IsoDate, IsoTime, MAX_EPOCH_DAYS, MIN_EPOCH_DAYS, NS_PER_DAY, NS_PER_HOUR,
    NS_PER_MINUTE, NS_PER_SEC, Overflow, RoundMode, TemporalData, TemporalKind, Unit, add_iso_date,
    balance_time_duration, balance_time_from_nanos, difference_iso_date, epoch_days_to_iso,
    format_fraction, iso_date_in_range, iso_to_epoch_days, parse_iso_datetime, time_to_nanos,
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

/// The four ISO date-duration components `(years, months, weeks, days)`.
type DateFields = (i64, i64, i64, i64);

/// A resolved `relativeTo` anchor. `time_ns` is the wall time-of-day in
/// nanoseconds (always `0` for a `PlainDate`/`PlainDateTime` anchor, which the
/// spec reduces to a date at midnight). `tz` is `Some` only for a
/// `ZonedDateTime` anchor, whose day lengths are time-zone dependent.
#[derive(Clone)]
struct DurAnchor {
    date: IsoDate,
    time_ns: i128,
    tz: Option<String>,
}

/// The instant an anchor+duration lands on: its wall `date`, wall time-of-day
/// `tod` (ns), and `dest_rel` = signed nanoseconds from the anchor's own instant.
struct DurTarget {
    date: IsoDate,
    tod: i128,
    dest_rel: i128,
}

/// The unsigned rounding direction after folding the sign into a mode.
#[derive(Clone, Copy)]
enum UnsignedMode {
    Zero,
    Infinity,
    HalfZero,
    HalfInfinity,
    HalfEven,
}

/// `GetUnsignedRoundingMode(mode, isNegative)`.
fn unsigned_round_mode(mode: RoundMode, neg: bool) -> UnsignedMode {
    use UnsignedMode::*;
    match mode {
        RoundMode::Ceil => {
            if neg {
                Zero
            } else {
                Infinity
            }
        }
        RoundMode::Floor => {
            if neg {
                Infinity
            } else {
                Zero
            }
        }
        RoundMode::Expand => Infinity,
        RoundMode::Trunc => Zero,
        RoundMode::HalfCeil => {
            if neg {
                HalfZero
            } else {
                HalfInfinity
            }
        }
        RoundMode::HalfFloor => {
            if neg {
                HalfInfinity
            } else {
                HalfZero
            }
        }
        RoundMode::HalfExpand => HalfInfinity,
        RoundMode::HalfTrunc => HalfZero,
        RoundMode::HalfEven => HalfEven,
    }
}

/// Field-wise negation of a duration.
fn negate_duration(o: &DurationFields) -> DurationFields {
    DurationFields {
        years: -o.years,
        months: -o.months,
        weeks: -o.weeks,
        days: -o.days,
        hours: -o.hours,
        minutes: -o.minutes,
        seconds: -o.seconds,
        milliseconds: -o.milliseconds,
        microseconds: -o.microseconds,
        nanoseconds: -o.nanoseconds,
    }
}

/// Truncates a signed value toward zero to a multiple of `increment`.
fn trunc_to_increment(v: i64, increment: i64) -> i64 {
    (v / increment) * increment
}

/// Parses a fixed UTC-offset time-zone identifier (`UTC`, `±HH`, `±HHMM`,
/// `±HH:MM`, with optional `:SS`); returns the offset in nanoseconds east of UTC.
fn parse_fixed_offset(s: &str) -> Option<i128> {
    if s.eq_ignore_ascii_case("utc") {
        return Some(0);
    }
    let b = s.as_bytes();
    let sign = match b.first()? {
        b'+' => 1_i128,
        b'-' => -1,
        _ => return None,
    };
    let rest = &s[1..];
    let digits: alloc::vec::Vec<u8> = rest.bytes().filter(u8::is_ascii_digit).collect();
    // Must be all digits + optional colons, and only HH / HHMM / HHMMSS shapes.
    if rest.bytes().any(|c| c != b':' && !c.is_ascii_digit()) {
        return None;
    }
    let (h, m, sec) = match digits.len() {
        2 => (&digits[..], &b""[..], &b""[..]),
        4 => (&digits[..2], &digits[2..4], &b""[..]),
        6 => (&digits[..2], &digits[2..4], &digits[4..6]),
        _ => return None,
    };
    let to_n = |d: &[u8]| -> i128 { d.iter().fold(0_i128, |a, &c| a * 10 + i128::from(c - b'0')) };
    let hh = to_n(h);
    let mm = if m.is_empty() { 0 } else { to_n(m) };
    let ss = if sec.is_empty() { 0 } else { to_n(sec) };
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    Some(sign * (hh * NS_PER_HOUR + mm * NS_PER_MINUTE + ss * NS_PER_SEC))
}

/// Canonicalizes a minute-precision UTC-offset time-zone *identifier* (`±HH`,
/// `±HH:MM`, `±HHMM`) to `±HH:MM`. Sub-minute precision (a seconds field) or any
/// other form is rejected — a time-zone id may not carry sub-minute offset.
fn dur_offset_id_canonical(s: &str) -> Option<String> {
    if s.eq_ignore_ascii_case("utc") {
        return Some(String::from("UTC"));
    }
    let b = s.as_bytes();
    let neg = match b.first()? {
        b'+' => false,
        b'-' => true,
        _ => return None,
    };
    let rest = &s[1..];
    if rest.bytes().any(|c| c != b':' && !c.is_ascii_digit()) {
        return None;
    }
    let digits: alloc::vec::Vec<u8> = rest.bytes().filter(u8::is_ascii_digit).collect();
    let (h, m) = match digits.len() {
        2 => (&digits[..2], &b"00"[..]),
        4 => (&digits[..2], &digits[2..4]),
        _ => return None, // 6 digits (seconds) → sub-minute → not a valid id
    };
    let to_n = |d: &[u8]| -> i64 { d.iter().fold(0, |a, &c| a * 10 + i64::from(c - b'0')) };
    let hh = to_n(h);
    let mm = to_n(m);
    if hh > 23 || mm > 59 {
        return None;
    }
    Some(alloc::format!(
        "{}{:02}:{:02}",
        if neg { '-' } else { '+' },
        hh,
        mm
    ))
}

/// Extracts the trailing numeric UTC-offset substring (`±HH…`) from an ISO
/// date-time string (after its `T`), so its precision can be validated. `None`
/// if there is no numeric offset.
fn dur_offset_substr(s: &str) -> Option<&str> {
    let t = s.rfind(['T', 't'])?;
    let after = &s[t + 1..];
    for (i, c) in after.char_indices() {
        match c {
            '+' | '-' => return Some(&after[i..]),
            'Z' | 'z' | '[' => return None,
            _ => {}
        }
    }
    None
}

/// Strictly parses a UTC-offset *value* string (`±HH`, `±HH:MM`, `±HHMM`,
/// `±HH:MM:SS[.fff]`, `±HHMMSS[.fff]`) with consistent separators, returning the
/// offset in nanoseconds. `None` for any malformed form.
fn dur_parse_offset_value(s: &str) -> Option<i128> {
    let b = s.as_bytes();
    let sign = match b.first()? {
        b'+' => 1_i128,
        b'-' | 0xE2 => -1, // ASCII '-' (U+2212 not handled here)
        _ => return None,
    };
    let mut i = 1usize;
    let two = |i: &mut usize| -> Option<i128> {
        let d0 = *b.get(*i)?;
        let d1 = *b.get(*i + 1)?;
        if !d0.is_ascii_digit() || !d1.is_ascii_digit() {
            return None;
        }
        *i += 2;
        Some(i128::from(d0 - b'0') * 10 + i128::from(d1 - b'0'))
    };
    let hh = two(&mut i)?;
    if hh > 23 {
        return None;
    }
    let mut mm = 0_i128;
    let mut ss = 0_i128;
    let mut frac = 0_i128;
    if i < b.len() {
        let colon = b[i] == b':';
        if colon {
            i += 1;
        }
        mm = two(&mut i)?;
        if mm > 59 {
            return None;
        }
        if i < b.len() {
            // Seconds must use the same separator style as minutes.
            if colon {
                if b[i] != b':' {
                    return None;
                }
                i += 1;
            }
            ss = two(&mut i)?;
            if ss > 59 {
                return None;
            }
            if i < b.len() && (b[i] == b'.' || b[i] == b',') {
                i += 1;
                let start = i;
                let mut scale = 100_000_000_i128;
                while i < b.len() && b[i].is_ascii_digit() {
                    if i - start < 9 {
                        frac += i128::from(b[i] - b'0') * scale;
                        scale /= 10;
                    }
                    i += 1;
                }
                if i == start {
                    return None;
                }
            }
        }
    }
    if i != b.len() {
        return None;
    }
    Some(sign * (hh * NS_PER_HOUR + mm * NS_PER_MINUTE + ss * NS_PER_SEC + frac))
}

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
    // Normalized-seconds total: |Σ time-in-seconds| < 2^53. Accumulate in
    // nanoseconds with checked i128 arithmetic so an overflowing (hence invalid)
    // field magnitude is rejected rather than wrapping. A large-but-valid field
    // (e.g. milliseconds ≈ 9·10^18) is retained.
    let per: [(usize, i128); 7] = [
        (3, NS_PER_DAY),
        (4, NS_PER_HOUR),
        (5, NS_PER_MINUTE),
        (6, NS_PER_SEC),
        (7, 1_000_000),
        (8, 1_000),
        (9, 1),
    ];
    let mut total_ns: i128 = 0;
    for (idx, mult) in per {
        let v = f[idx];
        if !v.is_finite() || v.abs() >= 1.0e30 {
            return false;
        }
        let Some(contrib) = (v as i128).checked_mul(mult) else {
            return false;
        };
        let Some(sum) = total_ns.checked_add(contrib) else {
            return false;
        };
        total_ns = sum;
    }
    // Truncate toward zero (not `div_euclid`, which would floor a negative total
    // one unit too far and spuriously reject the min-magnitude durations).
    (total_ns / NS_PER_SEC).abs() < TWO_POW_53
}

/// `2.0_f64.powi(k)` for `0 <= k <= 1023`, built from the IEEE-754 bit pattern so
/// it is exact and available in `no_std` (no `f64::powi`).
fn exp2_u32(k: u32) -> f64 {
    debug_assert!(k <= 1023);
    f64::from_bits((1023_u64 + u64::from(k)) << 52)
}

/// Correctly-rounded (round-to-nearest, ties-to-even) value of `num / den` as an
/// `f64`, for `den > 0`. Computing the integer quotient and the fractional
/// remainder as two separate `f64`s and adding them (`whole + rem`) double-rounds
/// once the quotient nears/exceeds `2^53`; this rounds the exact rational once.
/// `DivideNormalizedTimeDuration` in the Temporal spec is an exact rational, so
/// `Duration.prototype.total` must round it a single time.
fn ratio_to_f64(num: i128, den: i128) -> f64 {
    debug_assert!(den > 0);
    if num == 0 {
        return 0.0;
    }
    let neg = num < 0;
    let n = num.unsigned_abs();
    let d = den.unsigned_abs();
    let q = n / d; // integer part of |num|/|den|
    let rem = n % d; // exact value is q + rem/d, with 0 <= rem < d
    let qbits = 128 - q.leading_zeros(); // significant bits of q (0 when q == 0)
    let result = if q == 0 {
        // |value| < 1. A single IEEE division is correctly rounded when both
        // operands are exactly representable; `den` here is a fixed
        // nanoseconds-per-unit constant (<= 3.6e12 < 2^53) and `rem < den`.
        (rem as f64) / (d as f64)
    } else if qbits <= 53 {
        // q is exact in an f64 and there is mantissa room for `53 - qbits`
        // fraction bits. Build the 54-bit integer `floor(value * 2^scale)` (a
        // guard bit below the mantissa), then round-half-even using the division
        // remainder `fr` as the sticky bit. `value = mant * 2^-fbits`.
        let fbits = 53 - qbits;
        let scale = fbits + 1; // 1..=53
        let scaled = (rem << scale) / d; // fraction bits of rem/d (fits: rem < 2^42)
        let fr = (rem << scale) % d; // sticky remainder
        let m_full = (q << scale) + scaled; // 54-bit: floor(value * 2^scale)
        let guard = m_full & 1;
        let mut mant = m_full >> 1; // 53-bit mantissa
        if guard == 1 && (fr > 0 || (mant & 1) == 1) {
            mant += 1;
        }
        (mant as f64) / exp2_u32(fbits)
    } else {
        // q has > 53 significant bits: keep the top 53, and round using the bits
        // of q we drop plus the sub-integer fraction rem/d as the sticky bit.
        let drop = qbits - 53;
        let kept = q >> drop;
        let dropped = q & ((1_u128 << drop) - 1);
        let half = 1_u128 << (drop - 1);
        let round_up = match dropped.cmp(&half) {
            core::cmp::Ordering::Greater => true,
            core::cmp::Ordering::Less => false,
            // Exactly halfway among the dropped q-bits: the fraction rem/d (if
            // non-zero) tips it up, otherwise round to even.
            core::cmp::Ordering::Equal => rem > 0 || (kept & 1) == 1,
        };
        let m = kept + u128::from(round_up);
        (m as f64) * exp2_u32(drop)
    };
    if neg { -result } else { result }
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
        self.finish_temporal(data, new_target, callee)
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
            "add" => {
                let opts = args.get(1).copied().unwrap_or_else(NanBox::undefined);
                self.duration_add_sub(d, arg0, opts, false)
            }
            "subtract" => {
                let opts = args.get(1).copied().unwrap_or_else(NanBox::undefined);
                self.duration_add_sub(d, arg0, opts, true)
            }
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
            "sign" => i128::from(d.sign()),
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
                // (observably) and coerced to an anchor before any comparison.
                let opts_h = self.dur_options_object(opts)?;
                let anchor = self.dur_relative_to(opts_h)?;
                // Two durations with identical fields compare equal.
                if da == db {
                    return Ok(Some(NanBox::number(0.0)));
                }
                let lu_a = default_largest_unit(&da);
                let lu_b = default_largest_unit(&db);
                // A date-category largest unit (day or coarser) means the day count
                // may be irregular under a zoned anchor, so those are added to the
                // anchor and compared as instants. Everything else — time-only
                // durations, or day-durations under a plain anchor — is compared as
                // a straight (24h-day) time span, WITHOUT anchoring (so a time-only
                // comparison never overflows the anchor's instant range).
                let a_date = (lu_a as usize) <= (Unit::Day as usize);
                let b_date = (lu_b as usize) <= (Unit::Day as usize);
                if let Some(a) = &anchor
                    && a.tz.is_some()
                    && (a_date || b_date)
                {
                    let ta =
                        self.dur_apply(a, da.years, da.months, da.weeks, da.days, da.time_nanos())?;
                    let tb =
                        self.dur_apply(a, db.years, db.months, db.weeks, db.days, db.time_nanos())?;
                    let r = (ta.dest_rel - tb.dest_rel).signum() as f64;
                    return Ok(Some(NanBox::number(r)));
                }
                // Fallback: compare as time durations. Calendar units (year/month/
                // week) must first be resolved to a day count against a (plain)
                // relativeTo — absent one, that is a RangeError.
                let a_cal = matches!(lu_a, Unit::Year | Unit::Month | Unit::Week);
                let b_cal = matches!(lu_b, Unit::Year | Unit::Month | Unit::Week);
                let (mut d1, mut d2) = (da.days, db.days);
                if a_cal || b_cal {
                    let Some(a) = &anchor else {
                        return Err(
                            self.dur_range_error("compare requires relativeTo for calendar units")
                        );
                    };
                    d1 = self.dur_date_days(a, &da)?;
                    d2 = self.dur_date_days(a, &db)?;
                }
                let na = d1 * NS_PER_DAY + da.time_nanos();
                let nb = d2 * NS_PER_DAY + db.time_nanos();
                // `add24HourDays`: folding the resolved day count into the time span
                // must not exceed the maximum representable time duration.
                const MAX_TIME_NS: i128 = 9_007_199_254_740_991 * NS_PER_SEC + NS_PER_SEC - 1;
                if na.abs() > MAX_TIME_NS || nb.abs() > MAX_TIME_NS {
                    return Err(self.dur_range_error("duration time span out of range"));
                }
                let r = (na - nb).signum() as f64;
                Ok(Some(NanBox::number(r)))
            }
            _ => Ok(None),
        }
    }

    // -- helpers --------------------------------------------------------------

    /// Builds a fresh `Temporal.Duration` bound to the intrinsic prototype.
    fn new_duration(&mut self, d: DurationFields) -> NanBox {
        // Duration fields are Numbers: quantize to float64-representable integers.
        let d = crate::temporal_iso::quantize_duration_fields(d);
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
            years: f[0] as i128,
            months: f[1] as i128,
            weeks: f[2] as i128,
            days: f[3] as i128,
            hours: f[4] as i128,
            minutes: f[5] as i128,
            seconds: f[6] as i128,
            milliseconds: f[7] as i128,
            microseconds: f[8] as i128,
            nanoseconds: f[9] as i128,
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
    /// relativeTo anchor.
    fn duration_add_sub(
        &mut self,
        d: DurationFields,
        arg: NanBox,
        opts_arg: NanBox,
        subtract: bool,
    ) -> Result<NanBox, ExecError> {
        let mut other = self.coerce_temporal_duration(arg)?;
        if subtract {
            other = negate_duration(&other);
        }
        let opts_h = self.dur_options_object(opts_arg)?;
        let anchor = self.dur_relative_to(opts_h)?;

        let has_calendar = |x: &DurationFields| x.years != 0 || x.months != 0 || x.weeks != 0;
        if let Some(a) = anchor {
            // Add both durations to the anchor in sequence, then express the
            // net displacement as a fresh duration balanced to the larger of the
            // two operands' largest units.
            let t1 = self.dur_apply(&a, d.years, d.months, d.weeks, d.days, d.time_nanos())?;
            let a2 = DurAnchor {
                date: t1.date,
                time_ns: t1.tod,
                tz: a.tz.clone(),
            };
            let t2 = self.dur_apply(
                &a2,
                other.years,
                other.months,
                other.weeks,
                other.days,
                other.time_nanos(),
            )?;
            let dest_rel = t1.dest_rel + t2.dest_rel;
            let largest = default_largest_unit(&d).min(default_largest_unit(&other));
            let sign = dest_rel.signum() as i64;
            let fields = self.dur_round_from_target(
                &a,
                t2.date,
                t2.tod,
                dest_rel,
                sign,
                Unit::Nanosecond,
                largest,
                1,
                RoundMode::Trunc,
            )?;
            return Ok(self.new_duration(fields));
        }

        if has_calendar(&d) || has_calendar(&other) {
            return Err(
                self.dur_range_error("Duration arithmetic with calendar units requires relativeTo")
            );
        }
        // The result balances only up to the larger of the two operands' own
        // largest units (durations do not spontaneously balance beyond that).
        let largest = default_largest_unit(&d).min(default_largest_unit(&other));
        let total_ns = (d.days + other.days) * NS_PER_DAY + d.time_nanos() + other.time_nanos();
        // Reject a combined magnitude that no longer fits a valid duration before
        // balancing (which would otherwise overflow the i64 fields silently).
        if (total_ns / NS_PER_SEC).abs() >= TWO_POW_53 {
            return Err(self.dur_range_error("Duration arithmetic result out of range"));
        }
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
        let anchor = self.dur_relative_to(opts)?;
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

        // Default smallestUnit is nanosecond; default largestUnit is the coarser
        // of the duration's own largest unit and smallestUnit.
        let smallest = smallest.unwrap_or(Unit::Nanosecond);
        let default_largest = default_largest_unit(&d);
        let largest = largest.unwrap_or_else(|| default_largest.min(smallest));
        // largestUnit must be coarser-or-equal to smallestUnit (smaller ordinal).
        if (largest as usize) > (smallest as usize) {
            return Err(self.dur_range_error("largestUnit must not be finer than smallestUnit"));
        }
        // Validate the rounding increment against the smallest unit's ceiling.
        self.dur_validate_increment(smallest, increment)?;
        // Rounding to an increment >1 of a calendar/day smallestUnit is not
        // allowed when also balancing to a coarser largestUnit.
        if increment > 1
            && matches!(smallest, Unit::Year | Unit::Month | Unit::Week | Unit::Day)
            && largest != smallest
        {
            return Err(
                self.dur_range_error("cannot round to a calendar-unit increment while balancing")
            );
        }

        let has_calendar = d.years != 0 || d.months != 0 || d.weeks != 0;
        let smallest_is_cal = matches!(smallest, Unit::Year | Unit::Month | Unit::Week);
        let largest_is_cal = matches!(largest, Unit::Year | Unit::Month | Unit::Week);
        let need_anchor = has_calendar || smallest_is_cal || largest_is_cal;

        if let Some(a) = anchor {
            let sign = d.sign();
            // A zero-length duration usually rounds to zero. But with a
            // ZonedDateTime anchor, a sub-day smallestUnit, and a date-category
            // largestUnit, the spec still probes the day boundary (start-of-day
            // and start-of-next-day) via NudgeToZonedTime — which throws a
            // RangeError when the anchor sits at the edge of the representable
            // range. Run the nudge (with sign forced to +1) in that case.
            let smallest_is_time = matches!(
                smallest,
                Unit::Hour
                    | Unit::Minute
                    | Unit::Second
                    | Unit::Millisecond
                    | Unit::Microsecond
                    | Unit::Nanosecond
            );
            // Rounding to nanosecond/increment-1 is a no-op: the spec returns the
            // difference without a RoundRelativeDuration, so the boundary is not
            // probed and a max-edge anchor does not throw.
            let is_noop = smallest == Unit::Nanosecond && increment == 1;
            let zoned_probe = sign == 0
                && a.tz.is_some()
                && smallest_is_time
                && (largest as usize) <= (Unit::Day as usize)
                && !is_noop;
            if sign == 0 && !zoned_probe {
                return Ok(self.new_duration(DurationFields::default()));
            }
            let eff_sign = if sign == 0 { 1 } else { sign };
            let t = self.dur_apply(&a, d.years, d.months, d.weeks, d.days, d.time_nanos())?;
            // A plain (non-zoned) relativeTo anchors the difference at its date's
            // midnight; both endpoints must lie within the ISO date-time range.
            // (A zero-length difference short-circuits before this check.)
            if a.tz.is_none() && sign != 0 {
                self.dur_reject_datetime(a.date, a.time_ns)?;
                self.dur_reject_datetime(t.date, t.tod)?;
            }
            let fields = self.dur_round_from_target(
                &a, t.date, t.tod, t.dest_rel, eff_sign, smallest, largest, increment, mode,
            )?;
            return Ok(self.new_duration(fields));
        }
        if need_anchor {
            return Err(self.dur_range_error("round with calendar units requires relativeTo"));
        }

        // No calendar involvement and no anchor: pure fixed-length rounding.
        let total_ns = d.days * NS_PER_DAY + d.time_nanos();
        let incr = ns_per_unit(smallest) * increment;
        let rounded = self.dur_round_increment(total_ns, incr.max(1), mode);
        if (rounded / NS_PER_SEC).abs() >= TWO_POW_53 {
            return Err(self.dur_range_error("rounded Duration is out of range"));
        }

        let (days, time) = if (largest as usize) <= (Unit::Day as usize) {
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
        // ToTemporalRoundingIncrement truncates toward zero, then range-checks.
        if !n.is_finite() {
            return Err(self.dur_range_error("invalid roundingIncrement"));
        }
        let n = n.trunc();
        if n < 1.0 {
            return Err(self.dur_range_error("invalid roundingIncrement"));
        }
        Ok(n as i128)
    }

    /// `ValidateTemporalRoundingIncrement`: a fixed-length `smallest` unit caps
    /// the increment at the count of that unit in the next larger one, and the
    /// increment must divide that count evenly. Calendar units (and `day`) are
    /// unbounded.
    fn dur_validate_increment(&mut self, smallest: Unit, increment: i128) -> Result<(), ExecError> {
        let max = match smallest {
            Unit::Hour => 24,
            Unit::Minute | Unit::Second => 60,
            Unit::Millisecond | Unit::Microsecond | Unit::Nanosecond => 1000,
            _ => return Ok(()),
        };
        if increment >= max || max % increment != 0 {
            return Err(self.dur_range_error("roundingIncrement out of range for smallestUnit"));
        }
        Ok(())
    }

    // -- relativeTo anchor + calendar rounding --------------------------------

    /// `GetTemporalRelativeToOption(options)`: reads `options.relativeTo` and
    /// coerces it into an anchor. `None` = absent/undefined; a `Temporal`
    /// date/datetime/zoned instance, an ISO string, or a property bag otherwise.
    fn dur_relative_to(&mut self, opts: Option<Handle>) -> Result<Option<DurAnchor>, ExecError> {
        let Some(h) = opts else { return Ok(None) };
        let v = self.read_member(h, "relativeTo")?;
        if v.is_undefined() {
            return Ok(None);
        }
        // A Temporal instance is used directly by brand.
        if let Some(oh) = v.as_handle().map(Handle::from_raw) {
            if let Some(td) = self.realm.temporal_at(oh) {
                return match td.kind {
                    TemporalKind::PlainDate | TemporalKind::PlainDateTime => Ok(Some(DurAnchor {
                        date: td.date,
                        time_ns: 0,
                        tz: None,
                    })),
                    TemporalKind::ZonedDateTime => {
                        let tz = td.tz.clone().unwrap_or_else(|| String::from("UTC"));
                        let (date, time) = self.dur_local_of(&tz, td.epoch_ns);
                        Ok(Some(DurAnchor {
                            date,
                            time_ns: time_to_nanos(time),
                            tz: Some(tz),
                        }))
                    }
                    _ => Err(self.type_error("relativeTo must be a date, datetime, or string")),
                };
            }
            if self.is_object_value(v) {
                return Ok(Some(self.dur_relative_bag(oh)?));
            }
            // A string.
            if let Some(s) = self.realm.string_value(oh) {
                return Ok(Some(self.dur_relative_string(&s)?));
            }
        }
        // Numbers, bigints, booleans, null, symbols: not coercible.
        Err(self.type_error("relativeTo is not a valid date, datetime, or string"))
    }

    /// Parses an ISO string `relativeTo`: a `[tz]` annotation makes it zoned.
    fn dur_relative_string(&mut self, s: &str) -> Result<DurAnchor, ExecError> {
        // A leap second (`:60`) is constrained to `:59` rather than rejected.
        let parsed =
            parse_iso_datetime(s).or_else(|| parse_iso_datetime(&s.replacen(":60", ":59", 1)));
        let Some(p) = parsed else {
            return Err(self.dur_range_error("invalid relativeTo string"));
        };
        let Some(date) = p.date else {
            return Err(self.dur_range_error("relativeTo string has no date"));
        };
        if let Some(c) = &p.calendar
            && !(c.is_ascii() && c.eq_ignore_ascii_case("iso8601"))
        {
            return Err(self.dur_range_error("relativeTo calendar must be iso8601"));
        }
        if !iso_date_in_range(date) {
            return Err(self.dur_range_error("relativeTo date out of range"));
        }
        if let Some(tz_name) = p.tz_name {
            // A zoned relativeTo: its LOCAL date must satisfy CheckISODaysRange
            // (|epochDays| ≤ MAX_EPOCH_DAYS — one day tighter than the plain ±1
            // slop), the zone must be valid, and the resolved instant must be
            // representable.
            if iso_to_epoch_days(date).abs() > MAX_EPOCH_DAYS {
                return Err(self.dur_range_error("relativeTo date out of range"));
            }
            let Some(tz) = self.dur_resolve_tz_string(&tz_name) else {
                return Err(self.dur_range_error("invalid relativeTo time zone"));
            };
            let time = p.time.unwrap_or_default();
            let wall = iso_to_epoch_days(date) as i128 * NS_PER_DAY + time_to_nanos(time);
            let epoch = self.dur_zoned_epoch(&tz, wall, p.z, p.offset_ns)?;
            if !(crate::temporal_iso::MIN_EPOCH_NS..=crate::temporal_iso::MAX_EPOCH_NS)
                .contains(&epoch)
            {
                return Err(self.dur_range_error("relativeTo instant out of range"));
            }
            let (ldate, ltime) = self.dur_local_of(&tz, epoch);
            return Ok(DurAnchor {
                date: ldate,
                time_ns: time_to_nanos(ltime),
                tz: Some(tz),
            });
        }
        // A plain (non-zoned) relativeTo string may not carry a bare UTC
        // designator without a time-zone annotation.
        if p.z {
            return Err(self.dur_range_error("relativeTo string has a UTC designator but no zone"));
        }
        // Its date-at-noon must be within the representable ISO date-time range.
        self.dur_reject_daterange(date)?;
        Ok(DurAnchor {
            date,
            time_ns: 0,
            tz: None,
        })
    }

    /// Reads a `relativeTo` property bag (`{ year, month|monthCode, day, … }`),
    /// optionally with `timeZone` (→ zoned), `offset`, and `calendar`.
    fn dur_relative_bag(&mut self, h: Handle) -> Result<DurAnchor, ExecError> {
        // calendar
        let cal_v = self.read_member(h, "calendar")?;
        if !cal_v.is_undefined() {
            let ok = if let Some(c) = self.temporal_object_calendar(cal_v) {
                c.is_ascii() && c.eq_ignore_ascii_case("iso8601")
            } else {
                let Some(s) = cal_v
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|ch| self.realm.string_value(ch))
                else {
                    return Err(self.type_error("calendar must be a string"));
                };
                (s.is_ascii() && s.eq_ignore_ascii_case("iso8601"))
                    || parse_iso_datetime(&s).is_some_and(|p| {
                        p.calendar
                            .as_ref()
                            .is_none_or(|c| c.is_ascii() && c.eq_ignore_ascii_case("iso8601"))
                    })
            };
            if !ok {
                return Err(self.dur_range_error("relativeTo calendar must be iso8601"));
            }
        }
        // Date fields, read in alphabetical order.
        let day = self.dur_bag_field(h, "day")?;
        let hour = self.dur_bag_field(h, "hour")?.unwrap_or(0);
        let microsecond = self.dur_bag_field(h, "microsecond")?.unwrap_or(0);
        let millisecond = self.dur_bag_field(h, "millisecond")?.unwrap_or(0);
        let minute = self.dur_bag_field(h, "minute")?.unwrap_or(0);
        let month = self.dur_bag_field(h, "month")?;
        let month_code_v = self.read_member(h, "monthCode")?;
        let month_code = if month_code_v.is_undefined() {
            None
        } else {
            let s = self.coerce_to_string(month_code_v)?;
            Some(self.dur_parse_month_code(&s)?)
        };
        let nanosecond = self.dur_bag_field(h, "nanosecond")?.unwrap_or(0);
        // offset (validated for format only): read, then `ToPrimitive(string)`
        // which must yield a String (per ToRelativeTemporalObject / the offset
        // field's ToOffsetString) — an object with a `toString` is accepted.
        let offset_v = self.read_member(h, "offset")?;
        let offset_ns = if offset_v.is_undefined() {
            None
        } else {
            let prim = self.coerce_primitive(offset_v, "string")?;
            let Some(s) = prim
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|oh| self.realm.string_value(oh))
            else {
                return Err(self.type_error("offset must be a string"));
            };
            match dur_parse_offset_value(&s) {
                Some(v) => Some(v),
                None => return Err(self.dur_range_error("invalid offset string")),
            }
        };
        let second = self.dur_bag_field(h, "second")?.unwrap_or(0);
        // timeZone (must be a string identifier).
        let tz_v = self.read_member(h, "timeZone")?;
        let tz = if tz_v.is_undefined() {
            None
        } else {
            let Some(s) = tz_v
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|th| self.realm.string_value(th))
            else {
                return Err(self.type_error("timeZone must be a string"));
            };
            match self.dur_resolve_tz_string(&s) {
                Some(id) => Some(id),
                None => return Err(self.dur_range_error("invalid time zone identifier")),
            }
        };
        let year = self.dur_bag_field(h, "year")?;

        let (Some(year), Some(day)) = (year, day) else {
            return Err(self.type_error("relativeTo bag missing required year/day"));
        };
        let month = self.dur_resolve_month(month, month_code)?;
        let Some(date) =
            crate::temporal_iso::regulate_iso_date(year as i32, month, day, Overflow::Constrain)
        else {
            return Err(self.dur_range_error("relativeTo bag is not a valid date"));
        };
        if !iso_date_in_range(date) {
            return Err(self.dur_range_error("relativeTo date out of range"));
        }
        let Some(time) = crate::temporal_iso::regulate_iso_time(
            hour,
            minute,
            second,
            millisecond,
            microsecond,
            nanosecond,
            Overflow::Constrain,
        ) else {
            return Err(self.dur_range_error("relativeTo bag has invalid time fields"));
        };
        if let Some(tz_name) = tz {
            let wall = iso_to_epoch_days(date) as i128 * NS_PER_DAY + time_to_nanos(time);
            let epoch = self.dur_zoned_epoch(&tz_name, wall, false, offset_ns)?;
            let (ldate, ltime) = self.dur_local_of(&tz_name, epoch);
            return Ok(DurAnchor {
                date: ldate,
                time_ns: time_to_nanos(ltime),
                tz: Some(tz_name),
            });
        }
        Ok(DurAnchor {
            date,
            time_ns: 0,
            tz: None,
        })
    }

    /// Reads one integer date/time field from a bag (ToIntegerWithTruncation);
    /// `None` if the property is absent. Non-finite → RangeError.
    fn dur_bag_field(&mut self, h: Handle, key: &str) -> Result<Option<i64>, ExecError> {
        let v = self.read_member(h, key)?;
        if v.is_undefined() {
            return Ok(None);
        }
        let n = self.coerce_to_integer_or_infinity(v)?;
        if !n.is_finite() {
            return Err(self.dur_range_error("relativeTo field must be finite"));
        }
        Ok(Some(n as i64))
    }

    /// Parses a `monthCode` (`M##` optionally with a trailing `L`) → month number.
    fn dur_parse_month_code(&mut self, s: &str) -> Result<i64, ExecError> {
        let b = s.as_bytes();
        if (b.len() == 3 || b.len() == 4)
            && b[0] == b'M'
            && b[1].is_ascii_digit()
            && b[2].is_ascii_digit()
        {
            let n = i64::from(b[1] - b'0') * 10 + i64::from(b[2] - b'0');
            let leap = b.len() == 4 && b[3] == b'L';
            if !leap && (1..=12).contains(&n) {
                return Ok(n);
            }
        }
        Err(self.dur_range_error("invalid monthCode"))
    }

    /// Reconciles an optional numeric `month` and optional `monthCode`.
    fn dur_resolve_month(
        &mut self,
        month: Option<i64>,
        month_code: Option<i64>,
    ) -> Result<i64, ExecError> {
        match (month, month_code) {
            (Some(m), Some(c)) if m == c => Ok(m),
            (Some(_), Some(_)) => Err(self.dur_range_error("month and monthCode disagree")),
            (Some(m), None) => Ok(m),
            (None, Some(c)) => Ok(c),
            (None, None) => Err(self.type_error("relativeTo bag missing month/monthCode")),
        }
    }

    /// The absolute epoch nanoseconds of the anchor's own instant.
    fn dur_anchor_epoch_abs(&self, a: &DurAnchor) -> i128 {
        let wall = iso_to_epoch_days(a.date) as i128 * NS_PER_DAY + a.time_ns;
        match &a.tz {
            Some(tz) => self.dur_wall_to_epoch(tz, wall),
            None => wall,
        }
    }

    /// Epoch nanoseconds, relative to the anchor, of `date` combined with the
    /// anchor's own wall time-of-day (used for calendar start/end boundaries).
    fn dur_wall_rel(&self, a: &DurAnchor, date: IsoDate) -> i128 {
        let wall = iso_to_epoch_days(date) as i128 * NS_PER_DAY + a.time_ns;
        let abs = match &a.tz {
            Some(tz) => self.dur_wall_to_epoch(tz, wall),
            None => wall,
        };
        abs - self.dur_anchor_epoch_abs(a)
    }

    /// `DateDurationDays`: the whole-day count obtained by adding only a
    /// duration's calendar (year/month/week/day) part to the anchor — used by
    /// `Duration.compare` to resolve calendar units against a plain relativeTo.
    fn dur_date_days(&mut self, a: &DurAnchor, d: &DurationFields) -> Result<i128, ExecError> {
        let t = self.dur_apply(a, d.years, d.months, d.weeks, d.days, 0)?;
        Ok((iso_to_epoch_days(t.date) - iso_to_epoch_days(a.date)) as i128)
    }

    /// Adds a duration's date part + normalized time to an anchor, yielding the
    /// resulting wall date, wall time-of-day, and signed ns from the anchor.
    fn dur_apply(
        &mut self,
        a: &DurAnchor,
        years: i128,
        months: i128,
        weeks: i128,
        days: i128,
        norm: i128,
    ) -> Result<DurTarget, ExecError> {
        let d1 = add_iso_date(
            a.date,
            years as i64,
            months as i64,
            weeks as i64,
            days as i64,
            Overflow::Constrain,
        )
        .ok_or_else(|| self.dur_range_error("relativeTo date arithmetic out of range"))?;
        match &a.tz {
            None => {
                let combined = a.time_ns + norm;
                let extra = combined.div_euclid(NS_PER_DAY) as i64;
                let tod = combined.rem_euclid(NS_PER_DAY);
                let target_days = iso_to_epoch_days(d1) + extra;
                if !(MIN_EPOCH_DAYS - 1..=MAX_EPOCH_DAYS + 1).contains(&target_days) {
                    return Err(self.dur_range_error("resulting date out of range"));
                }
                let date = epoch_days_to_iso(target_days);
                let dest_rel = (iso_to_epoch_days(date) - iso_to_epoch_days(a.date)) as i128
                    * NS_PER_DAY
                    + tod
                    - a.time_ns;
                Ok(DurTarget {
                    date,
                    tod,
                    dest_rel,
                })
            }
            Some(tz) => {
                let wall1 = iso_to_epoch_days(d1) as i128 * NS_PER_DAY + a.time_ns;
                let epoch1 = self.dur_wall_to_epoch(tz, wall1);
                let epoch_final = epoch1 + norm;
                if !(crate::temporal_iso::MIN_EPOCH_NS..=crate::temporal_iso::MAX_EPOCH_NS)
                    .contains(&epoch_final)
                {
                    return Err(self.dur_range_error("resulting instant out of range"));
                }
                let (date, time) = self.dur_local_of(tz, epoch_final);
                let dest_rel = epoch_final - self.dur_anchor_epoch_abs(a);
                Ok(DurTarget {
                    date,
                    tod: time_to_nanos(time),
                    dest_rel,
                })
            }
        }
    }

    /// Rounds the displacement (`target` relative to `a`) to `smallest`/`largest`.
    #[allow(clippy::too_many_arguments)]
    fn dur_round_from_target(
        &mut self,
        a: &DurAnchor,
        date: IsoDate,
        tod: i128,
        dest_rel: i128,
        sign: i64,
        smallest: Unit,
        largest: Unit,
        increment: i128,
        mode: RoundMode,
    ) -> Result<DurationFields, ExecError> {
        if sign == 0 {
            return Ok(DurationFields::default());
        }
        // Difference granularity: the coarsest of largest and day (difference is
        // only defined for calendar units + day).
        let diff_unit = if (largest as usize) <= (Unit::Day as usize) {
            largest
        } else {
            Unit::Day
        };
        // Borrow so the sub-day time-of-day agrees with the overall sign.
        let mut adj = date;
        let mut subday = tod - a.time_ns;
        if sign > 0 && subday < 0 {
            adj = epoch_days_to_iso(iso_to_epoch_days(adj) - 1);
            subday += NS_PER_DAY;
        } else if sign < 0 && subday > 0 {
            adj = epoch_days_to_iso(iso_to_epoch_days(adj) + 1);
            subday -= NS_PER_DAY;
        }
        let (cy, cm, cw, cd) = difference_iso_date(a.date, adj, diff_unit);

        let smallest_is_cal = matches!(smallest, Unit::Year | Unit::Month | Unit::Week);
        let smallest_is_time = matches!(
            smallest,
            Unit::Hour
                | Unit::Minute
                | Unit::Second
                | Unit::Millisecond
                | Unit::Microsecond
                | Unit::Nanosecond
        );
        // NudgeToZonedTime: with a ZonedDateTime anchor and a sub-day smallestUnit,
        // days keep their (possibly non-24h) calendar length — the sub-day time is
        // rounded on its own against the actual day span, rather than folding days
        // into the time as fixed 24h intervals.
        // A no-op rounding (nanosecond, increment 1) keeps the difference as-is:
        // route it through the plain fixed-length path so no day boundary is probed.
        let is_noop = smallest == Unit::Nanosecond && increment == 1;
        let zoned_time = a.tz.is_some()
            && smallest_is_time
            && (largest as usize) <= (Unit::Day as usize)
            && !is_noop;
        let (mut y, mut m, mut w, mut days, norm_time, nudged_rel, did_expand) = if smallest_is_cal
        {
            let nc = self
                .dur_nudge_calendar(a, cy, cm, cw, cd, dest_rel, sign, smallest, increment, mode)?;
            (nc.0, nc.1, nc.2, nc.3, 0_i128, nc.4, nc.5)
        } else if zoned_time {
            self.dur_nudge_zoned_time(
                a, adj, cy, cm, cw, cd, subday, sign, smallest, increment, mode,
            )?
        } else {
            // Fixed-length nudge (day/time): combine leftover days + sub-day time.
            let norm_with_days = cd as i128 * NS_PER_DAY + subday;
            let unit_ns = ns_per_unit(smallest) * increment;
            let rounded = self.dur_round_increment(norm_with_days, unit_ns.max(1), mode);
            let whole_days = norm_with_days / NS_PER_DAY;
            let rounded_days = rounded / NS_PER_DAY;
            let did_expand = rounded_days != whole_days;
            let norm_time = rounded - rounded_days * NS_PER_DAY;
            let nudged_rel = dest_rel + (rounded - norm_with_days);
            (
                cy,
                cm,
                cw,
                rounded_days as i64,
                norm_time,
                nudged_rel,
                did_expand,
            )
        };

        // Bubble a rounded calendar/day result up to coarser units if it reached
        // the next boundary.
        if did_expand && smallest != Unit::Week {
            let (by, bm, bw, bd) =
                self.dur_bubble(a, y, m, w, days, nudged_rel, sign, largest, smallest)?;
            y = by;
            m = bm;
            w = bw;
            days = bd;
        }

        // Assemble the final fields, balancing the time portion to `largest`.
        let (final_days, time) = if (largest as usize) > (Unit::Day as usize) {
            let total = days as i128 * NS_PER_DAY + norm_time;
            (0_i64, balance_time_duration(total, largest))
        } else {
            (days, balance_time_duration(norm_time, Unit::Hour))
        };
        let f = [
            y as f64,
            m as f64,
            w as f64,
            final_days as f64,
            time.hours as f64,
            time.minutes as f64,
            time.seconds as f64,
            time.milliseconds as f64,
            time.microseconds as f64,
            time.nanoseconds as f64,
        ];
        self.dur_build(f)
    }

    /// `NudgeToCalendarUnit`: rounds the `unit` calendar component toward the
    /// anchored destination. Returns `(years, months, weeks, days, nudged_rel,
    /// did_expand)`.
    #[allow(clippy::too_many_arguments)]
    fn dur_nudge_calendar(
        &mut self,
        a: &DurAnchor,
        cy: i64,
        cm: i64,
        cw: i64,
        cd: i64,
        dest_rel: i128,
        sign: i64,
        unit: Unit,
        increment: i128,
        mode: RoundMode,
    ) -> Result<(i64, i64, i64, i64, i128, bool), ExecError> {
        let inc = increment as i64;
        let step = inc * sign;
        let (r1, start, end): (i64, DateFields, DateFields) = match unit {
            Unit::Year => {
                let r = trunc_to_increment(cy, inc);
                (r, (r, 0, 0, 0), (r + step, 0, 0, 0))
            }
            Unit::Month => {
                let r = trunc_to_increment(cm, inc);
                (r, (cy, r, 0, 0), (cy, r + step, 0, 0))
            }
            _ => {
                // Week: fold the leftover whole days (from a coarser difference)
                // into the week count before truncating to the increment.
                let r = trunc_to_increment(cw + cd / 7, inc);
                (r, (cy, cm, r, 0), (cy, cm, r + step, 0))
            }
        };
        let start_date = add_iso_date(
            a.date,
            start.0,
            start.1,
            start.2,
            start.3,
            Overflow::Constrain,
        )
        .ok_or_else(|| self.dur_range_error("calendar rounding out of range"))?;
        let end_date = add_iso_date(a.date, end.0, end.1, end.2, end.3, Overflow::Constrain)
            .ok_or_else(|| self.dur_range_error("calendar rounding out of range"))?;
        let start_rel = self.dur_wall_rel(a, start_date);
        let end_rel = self.dur_wall_rel(a, end_date);
        let span = end_rel - start_rel;
        let prog = dest_rel - start_rel;
        let expand = self.dur_decide_expand(prog, span, r1, inc, sign, mode);
        let (fields, nudged) = if expand {
            (end, end_rel)
        } else {
            (start, start_rel)
        };
        Ok((fields.0, fields.1, fields.2, fields.3, nudged, expand))
    }

    /// `RejectDateRange`: a plain `relativeTo` date is representable only when its
    /// noon instant lies within the ISO date-time range (`DATETIME_NS_MIN/MAX`) —
    /// noon, not midnight, so the ±1-day date span is admitted symmetrically.
    fn dur_reject_daterange(&mut self, date: IsoDate) -> Result<(), ExecError> {
        let ns = iso_to_epoch_days(date) as i128 * NS_PER_DAY + NS_PER_DAY / 2;
        let min = crate::temporal_iso::MIN_EPOCH_NS - NS_PER_DAY + 1;
        let max = crate::temporal_iso::MAX_EPOCH_NS + NS_PER_DAY - 1;
        if ns < min || ns > max {
            return Err(self.dur_range_error(
                "date is outside the representable range for a relativeTo parameter",
            ));
        }
        Ok(())
    }

    /// `RejectDateTimeRange`: a bare (non-zoned) ISO date-time is representable
    /// only within one day of the instant limits (`DATETIME_NS_MIN/MAX`), which is
    /// one nanosecond tighter at each edge than the ±1-day ISO-date span. A plain
    /// `relativeTo` whose date-at-midnight lies just outside is rejected.
    fn dur_reject_datetime(&mut self, date: IsoDate, time_ns: i128) -> Result<(), ExecError> {
        let ns = iso_to_epoch_days(date) as i128 * NS_PER_DAY + time_ns;
        let min = crate::temporal_iso::MIN_EPOCH_NS - NS_PER_DAY + 1;
        let max = crate::temporal_iso::MAX_EPOCH_NS + NS_PER_DAY - 1;
        if ns < min || ns > max {
            return Err(self.dur_range_error(
                "date is outside the representable range for a relativeTo parameter",
            ));
        }
        Ok(())
    }

    /// Epoch ns of `date` at the anchor's wall time-of-day, relative to the
    /// anchor — but throwing a RangeError when that absolute instant falls
    /// outside the representable range (the checked analogue of `dur_wall_rel`,
    /// used where the spec's `GetEpochNanosecondsFor` would reject a boundary).
    fn dur_wall_rel_checked(&mut self, a: &DurAnchor, date: IsoDate) -> Result<i128, ExecError> {
        let wall = iso_to_epoch_days(date) as i128 * NS_PER_DAY + a.time_ns;
        let abs = match &a.tz {
            Some(tz) => self.dur_wall_to_epoch(tz, wall),
            None => wall,
        };
        if !(crate::temporal_iso::MIN_EPOCH_NS..=crate::temporal_iso::MAX_EPOCH_NS).contains(&abs) {
            return Err(self.dur_range_error("day boundary is out of range"));
        }
        Ok(abs - self.dur_anchor_epoch_abs(a))
    }

    /// `NudgeToZonedTime`: rounds a sub-day time unit within a zoned day, keeping
    /// the calendar day count (`cd`) intact. `adj`/`subday` are the target date and
    /// the sign-aligned sub-day remainder. Returns the same tuple shape as
    /// `dur_nudge_calendar`: `(years, months, weeks, days, norm_time, nudged_rel,
    /// did_expand)`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn dur_nudge_zoned_time(
        &mut self,
        a: &DurAnchor,
        adj: IsoDate,
        cy: i64,
        cm: i64,
        cw: i64,
        cd: i64,
        subday: i128,
        sign: i64,
        unit: Unit,
        increment: i128,
        mode: RoundMode,
    ) -> Result<(i64, i64, i64, i64, i128, i128, bool), ExecError> {
        // Start / end of the target's whole-day interval, as instants. These probe
        // the representable range and may throw at the edge.
        let start_rel = self.dur_wall_rel_checked(a, adj)?;
        let end_date = epoch_days_to_iso(iso_to_epoch_days(adj) + sign);
        let end_rel = self.dur_wall_rel_checked(a, end_date)?;
        let day_span = end_rel - start_rel;
        let unit_ns = (ns_per_unit(unit) * increment).max(1);
        let mut rounded = self.dur_round_increment(subday, unit_ns, mode);
        // Did the rounded time reach or cross the end of the (possibly non-24h) day?
        let beyond = rounded - day_span;
        let did_beyond = if sign >= 0 { beyond >= 0 } else { beyond <= 0 };
        let (day_delta, nudged_rel) = if did_beyond {
            // Rounded into the next day: re-round the overshoot from the day-end.
            rounded = self.dur_round_increment(beyond, unit_ns, mode);
            (sign, end_rel + rounded)
        } else {
            (0, start_rel + rounded)
        };
        Ok((cy, cm, cw, cd + day_delta, rounded, nudged_rel, did_beyond))
    }

    /// `ApplyUnsignedRoundingMode` reduced to a boolean: whether to round toward
    /// the far (`r2`) increment boundary. `prog`/`span` are the signed ns progress
    /// and total span between the two boundaries.
    #[allow(clippy::too_many_arguments)]
    fn dur_decide_expand(
        &self,
        prog: i128,
        span: i128,
        r1: i64,
        inc: i64,
        sign: i64,
        mode: RoundMode,
    ) -> bool {
        let ap = prog.abs();
        let asp = span.abs();
        if ap == 0 {
            return false;
        }
        if asp == 0 || ap >= asp {
            return true;
        }
        match unsigned_round_mode(mode, sign < 0) {
            UnsignedMode::Zero => false,
            UnsignedMode::Infinity => true,
            UnsignedMode::HalfZero => 2 * ap > asp,
            UnsignedMode::HalfInfinity => 2 * ap >= asp,
            UnsignedMode::HalfEven => {
                if 2 * ap == asp {
                    // Round toward the even multiple: r2's cardinality is even.
                    let r2_card = r1 / inc + sign;
                    r2_card % 2 == 0
                } else {
                    2 * ap > asp
                }
            }
        }
    }

    /// Rounds `x` to a multiple of `increment` with sign-correct handling of the
    /// half/expand/trunc modes (the shared `round_to_increment` resolves ties
    /// toward +∞ rather than away from zero, which is wrong for negatives).
    fn dur_round_increment(&self, x: i128, increment: i128, mode: RoundMode) -> i128 {
        if increment <= 1 {
            return x;
        }
        let sign = x.signum();
        let ax = x.abs();
        let a_lower = (ax / increment) * increment;
        let ar = ax - a_lower;
        if ar == 0 {
            return x;
        }
        let aq = a_lower / increment;
        let far = match unsigned_round_mode(mode, sign < 0) {
            UnsignedMode::Zero => false,
            UnsignedMode::Infinity => true,
            UnsignedMode::HalfZero => 2 * ar > increment,
            UnsignedMode::HalfInfinity => 2 * ar >= increment,
            UnsignedMode::HalfEven => {
                if 2 * ar == increment {
                    aq % 2 == 1
                } else {
                    2 * ar > increment
                }
            }
        };
        let mag = if far { a_lower + increment } else { a_lower };
        sign * mag
    }

    /// `BubbleRelativeDuration`: carry a rounded result up through weeks/months/
    /// years when it reached the next coarser boundary.
    #[allow(clippy::too_many_arguments)]
    fn dur_bubble(
        &mut self,
        a: &DurAnchor,
        mut y: i64,
        mut m: i64,
        mut w: i64,
        mut d: i64,
        nudged_rel: i128,
        sign: i64,
        largest: Unit,
        smallest: Unit,
    ) -> Result<(i64, i64, i64, i64), ExecError> {
        if smallest == largest {
            return Ok((y, m, w, d));
        }
        let li = largest as usize; // Year=0..Day=3
        // Candidate coarser units, from just above smallest up to largest,
        // clamped so we never treat "day" as a bubble target.
        let start = core::cmp::min((smallest as usize).saturating_sub(1), Unit::Week as usize);
        let mut ui = start;
        loop {
            if ui < li {
                break;
            }
            let unit = match ui {
                0 => Unit::Year,
                1 => Unit::Month,
                _ => Unit::Week,
            };
            let is_week = unit == Unit::Week;
            if !is_week || largest == Unit::Week {
                let end = match unit {
                    Unit::Year => (y + sign, 0, 0, 0),
                    Unit::Month => (y, m + sign, 0, 0),
                    _ => (y, m, w + sign, 0),
                };
                let end_date =
                    add_iso_date(a.date, end.0, end.1, end.2, end.3, Overflow::Constrain)
                        .ok_or_else(|| self.dur_range_error("bubble out of range"))?;
                let end_rel = self.dur_wall_rel(a, end_date);
                let beyond = if sign < 0 {
                    nudged_rel <= end_rel
                } else {
                    nudged_rel >= end_rel
                };
                if beyond {
                    y = end.0;
                    m = end.1;
                    w = end.2;
                    d = 0;
                } else {
                    break;
                }
            }
            if ui == 0 {
                break;
            }
            ui -= 1;
        }
        Ok((y, m, w, d))
    }

    /// `TotalDuration`: the exact fractional total in `unit`, anchored.
    fn dur_total_relative(
        &mut self,
        a: &DurAnchor,
        d: DurationFields,
        unit: Unit,
    ) -> Result<f64, ExecError> {
        let sign = d.sign();
        let t = self.dur_apply(a, d.years, d.months, d.weeks, d.days, d.time_nanos())?;
        // With a ZonedDateTime anchor, `day` is an irregular-length unit: even a
        // zero-length total probes the day boundary (which throws at the edge of
        // the representable range). Every other zero-length total is 0.
        let zoned_day = unit == Unit::Day && a.tz.is_some();
        if sign == 0 && !zoned_day {
            return Ok(0.0);
        }
        let eff_sign = if sign == 0 { 1 } else { sign };
        // A plain (non-zoned) relativeTo anchors at its date's midnight; both
        // endpoints of a non-empty difference must be within the ISO date-time
        // range (a zero-length total returned above, before this check).
        if a.tz.is_none() && sign != 0 {
            self.dur_reject_datetime(a.date, a.time_ns)?;
            self.dur_reject_datetime(t.date, t.tod)?;
        }
        if zoned_day {
            // Whole days toward the destination + the fractional remainder, using
            // the actual (possibly non-24h) length of the bounding zoned day.
            let mut adj = t.date;
            let subday = t.tod - a.time_ns;
            if eff_sign > 0 && subday < 0 {
                adj = epoch_days_to_iso(iso_to_epoch_days(adj) - 1);
            } else if eff_sign < 0 && subday > 0 {
                adj = epoch_days_to_iso(iso_to_epoch_days(adj) + 1);
            }
            let (_, _, _, cd) = difference_iso_date(a.date, adj, Unit::Day);
            let end_date = epoch_days_to_iso(iso_to_epoch_days(adj) + eff_sign);
            let start_rel = self.dur_wall_rel_checked(a, adj)?;
            let end_rel = self.dur_wall_rel_checked(a, end_date)?;
            let asp = (end_rel - start_rel).unsigned_abs();
            if asp == 0 {
                return Ok(cd as f64);
            }
            let ap = (t.dest_rel - start_rel).unsigned_abs() as i128;
            let num = cd as i128 * asp as i128 + eff_sign as i128 * ap;
            return Ok(ratio_to_f64(num, asp as i128));
        }
        if matches!(unit, Unit::Year | Unit::Month | Unit::Week) {
            // Whole units toward the destination + the fractional remainder.
            let mut adj = t.date;
            let subday = t.tod - a.time_ns;
            if sign > 0 && subday < 0 {
                adj = epoch_days_to_iso(iso_to_epoch_days(adj) - 1);
            } else if sign < 0 && subday > 0 {
                adj = epoch_days_to_iso(iso_to_epoch_days(adj) + 1);
            }
            let (cy, cm, cw, cd) = difference_iso_date(a.date, adj, unit);
            let inc = 1_i64;
            let step = sign;
            let (r1, start, end): (i64, DateFields, DateFields) = match unit {
                Unit::Year => (cy, (cy, 0, 0, 0), (cy + step, 0, 0, 0)),
                Unit::Month => (cm, (cy, cm, 0, 0), (cy, cm + step, 0, 0)),
                _ => (cw, (cy, cm, cw, 0), (cy, cm, cw + step, 0)),
            };
            let _ = (cd, inc);
            let start_date = add_iso_date(
                a.date,
                start.0,
                start.1,
                start.2,
                start.3,
                Overflow::Constrain,
            )
            .ok_or_else(|| self.dur_range_error("total out of range"))?;
            let end_date = add_iso_date(a.date, end.0, end.1, end.2, end.3, Overflow::Constrain)
                .ok_or_else(|| self.dur_range_error("total out of range"))?;
            let start_rel = self.dur_wall_rel(a, start_date);
            let end_rel = self.dur_wall_rel(a, end_date);
            // `r1 + sign·(|prog|/|span|)` is a single exact rational in the spec
            // (`TotalDuration`); computing the fraction as an f64 and adding the
            // whole part double-rounds. Fold it into one numerator/denominator and
            // round to a double exactly once. prog/span share the duration's sign.
            let span = end_rel - start_rel;
            let prog = t.dest_rel - start_rel;
            let asp = span.unsigned_abs();
            if asp == 0 {
                return Ok(r1 as f64);
            }
            let ap = prog.unsigned_abs() as i128;
            let asp = asp as i128;
            let num = r1 as i128 * asp + sign as i128 * ap;
            return Ok(ratio_to_f64(num, asp));
        }
        // Fixed-length unit: the exact rational displacement/unit, rounded once.
        let per = ns_per_unit(unit);
        Ok(ratio_to_f64(t.dest_rel, per))
    }

    // -- time-zone helpers (fixed-offset + IANA) ------------------------------

    /// `ToTemporalTimeZoneIdentifier` for a string: a minute-precision offset id,
    /// a named IANA zone, or an ISO date-time string whose bracket annotation /
    /// `Z` / offset designates the zone. `None` (→ RangeError) if invalid.
    fn dur_resolve_tz_string(&self, s: &str) -> Option<String> {
        if s.is_empty() {
            return None;
        }
        if let Some(id) = dur_offset_id_canonical(s) {
            return Some(id);
        }
        if timezone_data::load(s).is_ok() {
            return Some(String::from(s));
        }
        // A date-time string carrying a zone. `parse_iso_datetime` rejects an
        // invalid date (e.g. the -000000 extended year) but not a leap second,
        // so tolerate a `:60` seconds field by clamping it for the probe.
        let probe = s.replacen(":60", ":59", 1);
        let p = parse_iso_datetime(&probe)?;
        p.date?;
        if let Some(ann) = p.tz_name {
            // The annotation itself must be a valid zone identifier.
            if let Some(id) = dur_offset_id_canonical(&ann) {
                return Some(id);
            }
            if timezone_data::load(&ann).is_ok() {
                return Some(ann);
            }
            return None;
        }
        if p.z {
            return Some(String::from("UTC"));
        }
        if p.offset_ns.is_some() {
            // A bare offset designates a zone only when written at minute
            // precision (no seconds field) — validate the raw substring.
            return dur_offset_substr(&probe).and_then(dur_offset_id_canonical);
        }
        None
    }

    /// Resolves a zoned wall time to an exact instant, reconciling an explicit
    /// UTC `offset` against the zone (`Z`/`z` → exact UTC; a numeric offset must
    /// match the zone's offset, else RangeError; absent → derive from the zone).
    fn dur_zoned_epoch(
        &mut self,
        tz: &str,
        wall: i128,
        z: bool,
        offset: Option<i128>,
    ) -> Result<i128, ExecError> {
        if z {
            return Ok(wall);
        }
        if let Some(off) = offset {
            let cand = wall - off;
            let zone_off = self.dur_tz_offset_at(tz, cand);
            if zone_off != off {
                return Err(self.dur_range_error("offset does not match the time zone"));
            }
            return Ok(cand);
        }
        Ok(self.dur_wall_to_epoch(tz, wall))
    }

    /// The offset (ns east of UTC) of `tz` at the exact instant `epoch_ns`.
    fn dur_tz_offset_at(&self, tz: &str, epoch_ns: i128) -> i128 {
        if let Some(ns) = parse_fixed_offset(tz) {
            return ns;
        }
        if let Ok(z) = timezone_data::load(tz) {
            let secs = epoch_ns.div_euclid(NS_PER_SEC) as i64;
            return i128::from(z.lookup(secs).offset) * NS_PER_SEC;
        }
        0
    }

    /// The wall date + time for an exact instant in `tz`.
    fn dur_local_of(&self, tz: &str, epoch_ns: i128) -> (IsoDate, IsoTime) {
        let off = self.dur_tz_offset_at(tz, epoch_ns);
        let (day, time) = balance_time_from_nanos(epoch_ns + off);
        (epoch_days_to_iso(day), time)
    }

    /// The exact instant whose local wall time is `wall_ns`, using the default
    /// (`compatible`) disambiguation across DST gaps/overlaps — shared with the
    /// `ZonedDateTime` resolver so the two stay consistent (`AddZonedDateTime`).
    fn dur_wall_to_epoch(&self, tz: &str, wall_ns: i128) -> i128 {
        if let Some(ns) = parse_fixed_offset(tz) {
            return wall_ns - ns;
        }
        super::temporal_zoneddatetime::wall_to_epoch(tz, wall_ns)
    }

    /// `Temporal.Duration.prototype.total(unitOrOptions)` — time units only.
    fn duration_total(&mut self, d: DurationFields, arg: NanBox) -> Result<NanBox, ExecError> {
        // The `totalOf` argument is required (a missing value is a TypeError).
        if arg.is_undefined() {
            return Err(self.type_error("total() requires a unit or options argument"));
        }
        // A string shorthand supplies the unit directly; otherwise the options
        // object is read in spec order: relativeTo, then unit.
        let (opts, unit_shorthand) = if let Some(s) = self.dur_as_string(arg) {
            (None, Some(s))
        } else {
            (self.dur_options_object(arg)?, None)
        };
        let anchor = self.dur_relative_to(opts)?;
        let unit_str = match unit_shorthand {
            Some(s) => Some(s),
            None => self.dur_string_option(opts, "unit")?,
        };
        let Some(unit_str) = unit_str else {
            return Err(self.dur_range_error("total requires a unit"));
        };
        let unit = parse_unit(&unit_str).ok_or_else(|| self.dur_range_error("invalid unit"))?;

        let has_calendar = d.years != 0 || d.months != 0 || d.weeks != 0;
        let unit_is_cal = matches!(unit, Unit::Year | Unit::Month | Unit::Week);

        if let Some(a) = anchor {
            let total = self.dur_total_relative(&a, d, unit)?;
            return Ok(NanBox::number(total));
        }
        if has_calendar || unit_is_cal {
            return Err(self.dur_range_error("total with calendar units requires relativeTo"));
        }

        let total_ns = d.days * NS_PER_DAY + d.time_nanos();
        let per = ns_per_unit(unit);
        // `DivideNormalizedTimeDuration`: the exact rational total_ns/per rounded to
        // a double exactly once (a naive whole-plus-fraction split double-rounds).
        Ok(NanBox::number(ratio_to_f64(total_ns, per)))
    }

    /// `TemporalDurationToString(duration, precision)`.
    fn duration_to_string(&mut self, d: DurationFields, arg: NanBox) -> Result<String, ExecError> {
        let opts = self.dur_options_object(arg)?;
        // Option read order: fractionalSecondDigits, then roundingMode, then
        // smallestUnit (which — if present — overrides the digit count).
        let frac_digits = self.dur_fractional_digits(opts)?;
        let mode = match self.dur_string_option(opts, "roundingMode")? {
            Some(s) => {
                parse_round_mode(&s).ok_or_else(|| self.dur_range_error("invalid roundingMode"))?
            }
            None => RoundMode::Trunc,
        };
        let smallest = self.dur_string_option(opts, "smallestUnit")?;

        let (precision, incr_ns): (Option<u8>, i128) = if let Some(su) = smallest {
            let digits = match su.as_str() {
                "second" | "seconds" => 0u8,
                "millisecond" | "milliseconds" => 3,
                "microsecond" | "microseconds" => 6,
                "nanosecond" | "nanoseconds" => 9,
                _ => return Err(self.dur_range_error("invalid smallestUnit for toString")),
            };
            (Some(digits), 10i128.pow(u32::from(9 - digits)))
        } else {
            match frac_digits {
                Some(p) => (Some(p), 10i128.pow(u32::from(9 - p))),
                None => (None, 1),
            }
        };

        // Round the sub-second remainder, then re-balance the time portion up to
        // the duration's own largest unit: a rounded-up second may carry into
        // minutes/hours, and (when a date unit is present) hours into days —
        // but the carry never propagates past days (e.g. 1:59:60 → 2h).
        let total_subsec = d.seconds * NS_PER_SEC
            + d.milliseconds * 1_000_000
            + d.microseconds * 1_000
            + d.nanoseconds;
        let rounded_subsec = self.dur_round_increment(total_subsec, incr_ns.max(1), mode);
        let total_time_ns = d.hours * NS_PER_HOUR + d.minutes * NS_PER_MINUTE + rounded_subsec;
        // Validity: the whole duration's normalized seconds must stay under 2^53.
        let secs_total = d.days * NS_PER_DAY + total_time_ns;
        if (secs_total / NS_PER_SEC).abs() >= TWO_POW_53 {
            return Err(self.dur_range_error("Duration is out of range for toString"));
        }
        // When rounding actually applies (a fixed precision), a rounded-up second
        // may carry into minutes→hours→days, but only as far as the duration's
        // own largest unit. With "auto" precision the fields print verbatim.
        let dl = default_largest_unit(&d) as usize;
        let mut whole_seconds = rounded_subsec / NS_PER_SEC;
        let frac = (rounded_subsec % NS_PER_SEC).unsigned_abs() as u32;
        let mut disp_minutes = d.minutes;
        let mut disp_hours = d.hours;
        let mut disp_days = d.days;
        if precision.is_some() {
            if dl <= Unit::Minute as usize {
                disp_minutes += whole_seconds / 60;
                whole_seconds %= 60;
            }
            if dl <= Unit::Hour as usize {
                disp_hours += disp_minutes / 60;
                disp_minutes %= 60;
            }
            if dl <= Unit::Day as usize {
                disp_days += disp_hours / 24;
                disp_hours %= 24;
            }
        }

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
        if disp_days != 0 {
            date_part.push_str(&alloc::format!("{}D", disp_days.unsigned_abs()));
        }
        let mut time_part = String::new();
        if disp_hours != 0 {
            time_part.push_str(&alloc::format!("{}H", disp_hours.unsigned_abs()));
        }
        if disp_minutes != 0 {
            time_part.push_str(&alloc::format!("{}M", disp_minutes.unsigned_abs()));
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

#[cfg(test)]
mod ratio_tests {
    use super::{exp2_u32, ratio_to_f64};

    /// The reference: a single IEEE division, correctly rounded whenever both
    /// operands are exactly representable as `f64` (|num|, den < 2^53).
    fn reference(num: i128, den: i128) -> f64 {
        num as f64 / den as f64
    }

    #[test]
    fn exp2_is_exact_power_of_two() {
        assert_eq!(exp2_u32(0), 1.0);
        assert_eq!(exp2_u32(1), 2.0);
        assert_eq!(exp2_u32(43), 8_796_093_022_208.0);
        assert_eq!(exp2_u32(52), 4_503_599_627_370_496.0);
    }

    #[test]
    fn matches_single_ieee_division_for_small_operands() {
        // Every case here has |num| < 2^53 and den < 2^53, so the reference is
        // itself correctly rounded and `ratio_to_f64` must reproduce it bit-exactly.
        let dens = [1_i128, 1_000, 1_000_000, 1_000_000_000, 3_600_000_000_000];
        for &den in &dens {
            for num in [0_i128, 1, 2, 7, 816, 999_999, 2_939_649_187_497_660] {
                assert_eq!(
                    ratio_to_f64(num, den).to_bits(),
                    reference(num, den).to_bits(),
                    "num={num} den={den}"
                );
                assert_eq!(
                    ratio_to_f64(-num, den).to_bits(),
                    reference(-num, den).to_bits(),
                    "num=-{num} den={den}"
                );
            }
        }
    }

    #[test]
    fn large_quotient_rounds_once_not_twice() {
        // total("milliseconds") of {milliseconds: 2^53, microseconds: 1999}: the
        // exact value 9007199254740993.999 rounds up to 9007199254740994 (spacing 2
        // at 2^53). A whole-plus-fraction split would land on 9007199254740992.
        let total_ns = 9_007_199_254_740_992_i128 * 1_000_000 + 1999 * 1_000;
        assert_eq!(ratio_to_f64(total_ns, 1_000_000), 9_007_199_254_740_994.0);
    }

    #[test]
    fn hour_total_is_spec_exact() {
        // Duration{hours:816, nanoseconds:2049187497660}.total("hours"). The
        // numerator fits in 2^53, so `reference` is itself correctly rounded and
        // is the test262 expected value 816.56921874935. A naive whole+fraction
        // split double-rounds one ULP away from it.
        let total_ns = 816_i128 * super::NS_PER_HOUR + 2_049_187_497_660;
        assert!(total_ns < (1_i128 << 53));
        assert_eq!(
            ratio_to_f64(total_ns, super::NS_PER_HOUR).to_bits(),
            reference(total_ns, super::NS_PER_HOUR).to_bits()
        );
    }

    #[test]
    fn sub_unit_fraction_is_correct() {
        // |value| < 1 branch: 91035820 ns / 1 hour.
        let expected = 91_035_820_f64 / super::NS_PER_HOUR as f64;
        assert_eq!(
            ratio_to_f64(91_035_820, super::NS_PER_HOUR).to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn huge_microsecond_quotient() {
        // A quotient far beyond 2^53 (the > 53-bit branch): 2^53 seconds in µs is
        // exactly representable (= 2^59 * 15625), so the result is exact.
        let total_ns = 9_007_199_254_740_992_i128 * super::NS_PER_SEC;
        assert_eq!(
            ratio_to_f64(total_ns, 1_000),
            9_007_199_254_740_992_000_000.0
        );
    }
}

/// End-to-end checks of `Duration.prototype.round/total/compare` with a
/// `relativeTo` — the calendar/DST-aware rounding and range-limit semantics.
#[cfg(test)]
mod relative_tests {
    /// Runs a sloppy-mode script, returning its `print` output on success or the
    /// thrown error's name (e.g. `"RangeError"`) on a throw.
    fn run(src: &str) -> Result<alloc::string::String, alloc::string::String> {
        let prelude = "var print = function () { var s=''; for (var i=0;i<arguments.length;i++){ if(i) s+=' '; s+=arguments[i]; } console.log(s); };\n";
        let combined = alloc::format!("{prelude}{src}");
        match crate::nbvm::execute_typed(&combined, crate::limits::Limits::default()) {
            Ok((out, _)) => Ok(out),
            Err(t) => Err(alloc::string::String::from(t.name)),
        }
    }

    #[test]
    fn zoned_round_keeps_days_separate_half_even() {
        // 3 days 12 hours, smallestUnit hours, increment 8, halfEven. With a
        // ZonedDateTime relativeTo the days stay separate, so only 12h rounds:
        // 12h/8 = 1.5 → halfEven → 16h. Result: 3 days 16 hours.
        let out = run(r#"
            var d = new Temporal.Duration(0,0,0,3,12);
            var z = new Temporal.ZonedDateTime(0n, "UTC");
            var r = d.round({ smallestUnit:"hours", roundingIncrement:8, roundingMode:"halfEven", relativeTo:z });
            print(r.days, r.hours);
        "#)
        .expect("no throw");
        assert_eq!(out.trim(), "3 16");
    }

    #[test]
    fn plain_round_folds_days_half_even() {
        // The same rounding with a PlainDate relativeTo folds days at 24h:
        // 84h/8 = 10.5 → halfEven → 80h → 3 days 8 hours.
        let out = run(r#"
            var d = new Temporal.Duration(0,0,0,3,12);
            var p = new Temporal.PlainDate(1970,1,1);
            var r = d.round({ smallestUnit:"hours", roundingIncrement:8, roundingMode:"halfEven", relativeTo:p });
            print(r.days, r.hours);
        "#)
        .expect("no throw");
        assert_eq!(out.trim(), "3 8");
    }

    #[test]
    fn total_month_is_single_rounded_rational() {
        // total("months") of P5W5D relative to 1972-01-31 must equal the exact
        // rational rounded once (1.3548387096774193), not a double-rounded value.
        let out = run(r#"
            var d = new Temporal.Duration(0,0,5,5);
            print(d.total({ unit:"months", relativeTo:"1972-01-31" }));
        "#)
        .expect("no throw");
        assert_eq!(out.trim(), "1.3548387096774193");
    }

    #[test]
    fn round_next_day_boundary_out_of_range_throws() {
        // A zero duration at the max instant: rounding to days must probe the
        // next-day boundary, which is out of range → RangeError.
        let err = run(r#"
            var d = new Temporal.Duration();
            var z = new Temporal.ZonedDateTime(86400_0000_0000_000_000_000n, "UTC");
            d.round({ largestUnit:"days", smallestUnit:"minutes", relativeTo:z });
        "#)
        .unwrap_err();
        assert_eq!(err, "RangeError");
    }

    #[test]
    fn round_noop_at_max_zoned_does_not_throw() {
        // A no-op rounding (nanosecond/increment 1) does NOT probe the boundary,
        // so a max-edge ZonedDateTime relativeTo is fine.
        let out = run(r#"
            var d = new Temporal.Duration();
            var r = d.round({ largestUnit:"years", relativeTo:"-271821-04-20T00:00+00:00[UTC]" });
            print(r.years, r.days);
        "#)
        .expect("no throw");
        assert_eq!(out.trim(), "0 0");
    }

    #[test]
    fn total_days_zoned_boundary_throws_one_second_past() {
        // total("days") of a zero duration: valid exactly at the max whole day,
        // but one second later the day-end boundary is out of range → RangeError.
        let ok = run(r#"
            print(new Temporal.Duration(0).total({ unit:"days", relativeTo:"+275760-09-12T00:00:00+00:00[UTC]" }));
        "#)
        .expect("no throw");
        assert_eq!(ok.trim(), "0");
        let err = run(r#"
            new Temporal.Duration(0).total({ unit:"days", relativeTo:"+275760-09-12T00:00:01+00:00[UTC]" });
        "#)
        .unwrap_err();
        assert_eq!(err, "RangeError");
    }

    #[test]
    fn offset_zone_relativeto_local_date_out_of_range_throws() {
        // A zoned relativeTo whose LOCAL date is beyond the CheckISODaysRange
        // bound is rejected at parse (both zero and non-zero durations throw).
        let err = run(r#"
            new Temporal.Duration(0,0,0,0,0,5).round({ smallestUnit:"minutes", relativeTo:"-271821-04-19T23:00-01:00[-01:00]" });
        "#)
        .unwrap_err();
        assert_eq!(err, "RangeError");
    }

    #[test]
    fn plain_relativeto_max_date_valid_but_next_is_not() {
        // +275760-09-13 is a valid plain relativeTo; +275760-09-14 is out of range.
        run(r#"
            new Temporal.Duration(0,0,0,0,0,5).round({ smallestUnit:"minutes", relativeTo:"+275760-09-13" });
        "#)
        .expect("valid max date");
        let err = run(r#"
            new Temporal.Duration().round({ smallestUnit:"minutes", relativeTo:"+275760-09-14" });
        "#)
        .unwrap_err();
        assert_eq!(err, "RangeError");
    }

    #[test]
    fn compare_time_only_at_max_zoned_does_not_anchor() {
        // Time-only durations compare as a straight time span even with a zoned
        // relativeTo at the max instant — no AddZonedDateTime, so no overflow.
        let out = run(r#"
            var a = new Temporal.Duration(0,0,0,0,0,5);
            var b = new Temporal.Duration();
            print(Temporal.Duration.compare(a, b, { relativeTo:"+275760-09-13T00:00Z[UTC]" }));
        "#)
        .expect("no throw");
        assert_eq!(out.trim(), "1");
    }

    #[test]
    fn compare_calendar_time_overflow_throws() {
        // 1 year + (2^53-1) seconds vs 2 years, relative to a plain date: folding
        // the year's days into the huge time span overflows → RangeError.
        let err = run(r#"
            var a = Temporal.Duration.from({ years:1, seconds: 2**53 - 1 });
            var b = Temporal.Duration.from({ years:2 });
            Temporal.Duration.compare(a, b, { relativeTo: new Temporal.PlainDate(2000,1,1) });
        "#)
        .unwrap_err();
        assert_eq!(err, "RangeError");
    }
}
