//! Pure-logic ISO-8601 calendar/time/duration core for the `Temporal.*` native
//! types. No engine coupling: plain `Copy` structs and functions operating on
//! them, so it is unit-testable in isolation and usable from any of the
//! per-type Temporal modules.
//!
//! Ranges follow the Temporal spec: the ISO year is limited so that the
//! representable instant stays within ±(10^8) days of the epoch. Times are
//! nanosecond precision; epoch instants use `i128` nanoseconds.
#![allow(clippy::many_single_char_names)]
#![allow(missing_docs)] // data-heavy pure-logic module; fields are self-describing
#![allow(clippy::manual_range_contains, clippy::manual_clamp, clippy::while_let_loop)]

use alloc::string::String;

/// Max/min ISO date as days from the 1970-01-01 epoch (±10^8 days, plus a day of
/// slop for the time-of-day the spec allows at the boundary).
pub const MAX_EPOCH_DAYS: i64 = 100_000_000;
pub const MIN_EPOCH_DAYS: i64 = -100_000_000;
/// Nanoseconds in a day / hour / minute / second.
pub const NS_PER_DAY: i128 = 86_400_000_000_000;
pub const NS_PER_HOUR: i128 = 3_600_000_000_000;
pub const NS_PER_MINUTE: i128 = 60_000_000_000;
pub const NS_PER_SEC: i128 = 1_000_000_000;
/// Epoch-nanosecond limits (±10^8 days).
pub const MAX_EPOCH_NS: i128 = MAX_EPOCH_DAYS as i128 * NS_PER_DAY;
pub const MIN_EPOCH_NS: i128 = MIN_EPOCH_DAYS as i128 * NS_PER_DAY;

/// A plain ISO calendar date (proleptic Gregorian). `month` and `day` are
/// 1-based.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IsoDate {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// A plain wall-clock time with nanosecond precision.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct IsoTime {
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub millisecond: u16,
    pub microsecond: u16,
    pub nanosecond: u16,
}

/// A combined date + wall-clock time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IsoDateTime {
    pub date: IsoDate,
    pub time: IsoTime,
}

/// The ten Temporal.Duration fields (all signed; a valid Duration has a single
/// sign across all non-zero fields).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DurationFields {
    pub years: i64,
    pub months: i64,
    pub weeks: i64,
    pub days: i64,
    pub hours: i64,
    pub minutes: i64,
    pub seconds: i64,
    pub milliseconds: i64,
    pub microseconds: i64,
    pub nanoseconds: i64,
}

/// Temporal calendar/rounding unit, ordered largest→smallest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Unit {
    Year,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

/// Rounding mode (a subset sufficient for Temporal's `roundingMode` option).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RoundMode {
    Ceil,
    Floor,
    Expand,
    Trunc,
    HalfCeil,
    HalfFloor,
    HalfExpand,
    HalfTrunc,
    HalfEven,
}

/// How out-of-range date/time components are handled on construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Overflow {
    Constrain,
    Reject,
}

/// Which `Temporal.*` type a branded instance is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TemporalKind {
    PlainDate,
    PlainTime,
    PlainDateTime,
    Duration,
    Instant,
    PlainYearMonth,
    PlainMonthDay,
    ZonedDateTime,
}

impl TemporalKind {
    /// The unqualified constructor name (`"PlainDate"`, …).
    #[must_use]
    pub fn type_name(self) -> &'static str {
        match self {
            TemporalKind::PlainDate => "PlainDate",
            TemporalKind::PlainTime => "PlainTime",
            TemporalKind::PlainDateTime => "PlainDateTime",
            TemporalKind::Duration => "Duration",
            TemporalKind::Instant => "Instant",
            TemporalKind::PlainYearMonth => "PlainYearMonth",
            TemporalKind::PlainMonthDay => "PlainMonthDay",
            TemporalKind::ZonedDateTime => "ZonedDateTime",
        }
    }
}

/// The internal slots of a branded `Temporal.*` instance — one uniform record for
/// every type (unused fields stay at their defaults). Temporal objects are
/// immutable, so this is shared behind an `Rc` in the heap cell.
#[derive(Clone, Debug)]
pub struct TemporalData {
    pub kind: TemporalKind,
    /// Date-bearing types (PlainDate/PlainDateTime/PlainYearMonth/PlainMonthDay).
    pub date: IsoDate,
    /// Time-bearing types (PlainTime/PlainDateTime).
    pub time: IsoTime,
    /// Duration fields (Duration).
    pub duration: DurationFields,
    /// Epoch nanoseconds (Instant/ZonedDateTime).
    pub epoch_ns: i128,
    /// Calendar id — always `"iso8601"` for now.
    pub calendar: String,
    /// Time-zone id (ZonedDateTime only).
    pub tz: Option<String>,
}

impl Default for TemporalData {
    fn default() -> Self {
        TemporalData {
            kind: TemporalKind::PlainDate,
            date: IsoDate {
                year: 0,
                month: 1,
                day: 1,
            },
            time: IsoTime::default(),
            duration: DurationFields::default(),
            epoch_ns: 0,
            calendar: String::from("iso8601"),
            tz: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Calendar math (proleptic Gregorian, ISO 8601)
// ---------------------------------------------------------------------------

/// Whether `year` is a leap year in the proleptic Gregorian calendar.
#[must_use]
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Number of days in `month` (1-12) of `year`.
#[must_use]
pub fn iso_days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Days in `year` (365 or 366).
#[must_use]
pub fn iso_days_in_year(year: i32) -> u16 {
    if is_leap_year(year) { 366 } else { 365 }
}

/// Days from the 1970-01-01 epoch to `date` (Howard Hinnant's `days_from_civil`,
/// valid across the full proleptic Gregorian range).
#[must_use]
pub fn iso_to_epoch_days(date: IsoDate) -> i64 {
    let y = i64::from(date.year);
    let m = i64::from(date.month);
    let d = i64::from(date.day);
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Inverse of [`iso_to_epoch_days`] (Hinnant's `civil_from_days`).
#[must_use]
pub fn epoch_days_to_iso(z: i64) -> IsoDate {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    IsoDate {
        year: (if m <= 2 { y + 1 } else { y }) as i32,
        month: m as u8,
        day: d as u8,
    }
}

/// ISO day of week for `date`: 1 = Monday … 7 = Sunday.
#[must_use]
pub fn iso_day_of_week(date: IsoDate) -> u8 {
    // 1970-01-01 was a Thursday (ISO weekday 4).
    let dow = (iso_to_epoch_days(date) + 3).rem_euclid(7); // 0 = Monday
    (dow + 1) as u8
}

/// 1-based day of the year.
#[must_use]
pub fn iso_day_of_year(date: IsoDate) -> u16 {
    let jan1 = IsoDate {
        year: date.year,
        month: 1,
        day: 1,
    };
    (iso_to_epoch_days(date) - iso_to_epoch_days(jan1) + 1) as u16
}

/// ISO 8601 week-of-year and the year that week belongs to (weeks belong to the
/// year containing their Thursday).
#[must_use]
pub fn iso_week_of_year(date: IsoDate) -> (u8, i32) {
    let dow = i32::from(iso_day_of_week(date));
    let doy = i32::from(iso_day_of_year(date));
    let week = (doy - dow + 10) / 7;
    if week < 1 {
        // Belongs to the last week of the previous year.
        let prev = date.year - 1;
        let wk = if is_iso_long_year(prev) { 53 } else { 52 };
        (wk as u8, prev)
    } else if week > 52 && !is_iso_long_year(date.year) {
        (1, date.year + 1)
    } else {
        (week as u8, date.year)
    }
}

/// Whether an ISO year has 53 weeks (a "long" year).
fn is_iso_long_year(year: i32) -> bool {
    let p = |y: i32| (y + y / 4 - y / 100 + y / 400) % 7;
    p(year) == 4 || p(year - 1) == 3
}

// ---------------------------------------------------------------------------
// Validation / regulation
// ---------------------------------------------------------------------------

/// Whether `date` is inside the representable Temporal range.
#[must_use]
pub fn iso_date_in_range(date: IsoDate) -> bool {
    let d = iso_to_epoch_days(date);
    // A date is valid if some instant on that day is in range: allow the day
    // before MIN and after MAX (the time-of-day slop).
    (MIN_EPOCH_DAYS - 1..=MAX_EPOCH_DAYS + 1).contains(&d)
}

/// Validates raw date components. `Some(date)` if a real calendar date;
/// constrains or rejects an out-of-range month/day per `overflow`.
#[must_use]
pub fn regulate_iso_date(year: i32, month: i64, day: i64, overflow: Overflow) -> Option<IsoDate> {
    match overflow {
        Overflow::Constrain => {
            let month = month.clamp(1, 12) as u8;
            let dim = i64::from(iso_days_in_month(year, month));
            let day = day.clamp(1, dim) as u8;
            Some(IsoDate { year, month, day })
        }
        Overflow::Reject => {
            if !(1..=12).contains(&month) {
                return None;
            }
            let month = month as u8;
            if day < 1 || day > i64::from(iso_days_in_month(year, month)) {
                return None;
            }
            Some(IsoDate {
                year,
                month,
                day: day as u8,
            })
        }
    }
}

/// Validates raw time components (each already reduced to its field range check).
/// Constrains to the valid range or rejects per `overflow`.
#[must_use]
pub fn regulate_iso_time(
    hour: i64,
    minute: i64,
    second: i64,
    ms: i64,
    us: i64,
    ns: i64,
    overflow: Overflow,
) -> Option<IsoTime> {
    let ck = |v: i64, max: i64| -> Option<i64> {
        match overflow {
            Overflow::Constrain => Some(v.clamp(0, max)),
            Overflow::Reject => (0..=max).contains(&v).then_some(v),
        }
    };
    Some(IsoTime {
        hour: ck(hour, 23)? as u8,
        minute: ck(minute, 59)? as u8,
        second: ck(second, 59)? as u8,
        millisecond: ck(ms, 999)? as u16,
        microsecond: ck(us, 999)? as u16,
        nanosecond: ck(ns, 999)? as u16,
    })
}

// ---------------------------------------------------------------------------
// Time arithmetic / balancing
// ---------------------------------------------------------------------------

/// Nanoseconds since midnight for `t` (0 ..= 86_399_999_999_999).
#[must_use]
pub fn time_to_nanos(t: IsoTime) -> i128 {
    i128::from(t.hour) * NS_PER_HOUR
        + i128::from(t.minute) * NS_PER_MINUTE
        + i128::from(t.second) * NS_PER_SEC
        + i128::from(t.millisecond) * 1_000_000
        + i128::from(t.microsecond) * 1_000
        + i128::from(t.nanosecond)
}

/// Splits a signed nanosecond count into a whole-day carry and the wall-clock
/// time of day (BalanceTime): returns `(day_carry, time)`.
#[must_use]
pub fn balance_time_from_nanos(total_ns: i128) -> (i64, IsoTime) {
    let day = total_ns.div_euclid(NS_PER_DAY);
    let mut r = total_ns.rem_euclid(NS_PER_DAY);
    let hour = (r / NS_PER_HOUR) as u8;
    r %= NS_PER_HOUR;
    let minute = (r / NS_PER_MINUTE) as u8;
    r %= NS_PER_MINUTE;
    let second = (r / NS_PER_SEC) as u8;
    r %= NS_PER_SEC;
    let millisecond = (r / 1_000_000) as u16;
    r %= 1_000_000;
    let microsecond = (r / 1_000) as u16;
    let nanosecond = (r % 1_000) as u16;
    (
        day as i64,
        IsoTime {
            hour,
            minute,
            second,
            millisecond,
            microsecond,
            nanosecond,
        },
    )
}

/// Adds a signed nanosecond delta to a time, returning the day carry and the new
/// time (AddTime).
#[must_use]
pub fn add_time(t: IsoTime, delta_ns: i128) -> (i64, IsoTime) {
    balance_time_from_nanos(time_to_nanos(t) + delta_ns)
}

// ---------------------------------------------------------------------------
// Date arithmetic
// ---------------------------------------------------------------------------

/// Balances a year + (possibly out-of-range) month into a valid (year, month).
#[must_use]
pub fn balance_iso_year_month(year: i64, month: i64) -> (i32, u8) {
    let m0 = month - 1;
    let y = year + m0.div_euclid(12);
    let m = m0.rem_euclid(12) + 1;
    (y as i32, m as u8)
}

/// AddISODate: adds years, months, weeks, days to `date` with the given overflow
/// behaviour. `None` on overflow-reject or out-of-range result.
#[must_use]
pub fn add_iso_date(
    date: IsoDate,
    years: i64,
    months: i64,
    weeks: i64,
    days: i64,
    overflow: Overflow,
) -> Option<IsoDate> {
    // 1. Add years and months, then regulate the resulting day-of-month.
    let (y, m) =
        balance_iso_year_month(i64::from(date.year) + years, i64::from(date.month) + months);
    let intermediate = regulate_iso_date(y, i64::from(m), i64::from(date.day), overflow)?;
    // 2. Add weeks and days as a plain day offset.
    let total_days = days + weeks * 7;
    let result = epoch_days_to_iso(iso_to_epoch_days(intermediate) + total_days);
    iso_date_in_range(result).then_some(result)
}

/// DifferenceISODate down to `largest_unit` (one of Year/Month/Week/Day). Returns
/// the date-portion duration fields (years, months, weeks, days).
#[must_use]
pub fn difference_iso_date(from: IsoDate, to: IsoDate, largest: Unit) -> (i64, i64, i64, i64) {
    if largest == Unit::Day || largest == Unit::Week {
        let mut days = iso_to_epoch_days(to) - iso_to_epoch_days(from);
        let mut weeks = 0;
        if largest == Unit::Week {
            weeks = days / 7;
            days %= 7;
        }
        return (0, 0, weeks, days);
    }
    // Year/Month: count whole years then whole months, then leftover days.
    let sign = match compare_iso_date(to, from) {
        core::cmp::Ordering::Greater => 1_i64,
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => return (0, 0, 0, 0),
    };
    let mut years = i64::from(to.year) - i64::from(from.year);
    let mut mid = add_iso_date(from, years, 0, 0, 0, Overflow::Constrain).unwrap_or(from);
    // Back off until mid does not pass `to`.
    while sign > 0 && compare_iso_date(mid, to) == core::cmp::Ordering::Greater
        || sign < 0 && compare_iso_date(mid, to) == core::cmp::Ordering::Less
    {
        years -= sign;
        mid = add_iso_date(from, years, 0, 0, 0, Overflow::Constrain).unwrap_or(from);
    }
    let mut months = 0_i64;
    loop {
        let next =
            add_iso_date(from, years, months + sign, 0, 0, Overflow::Constrain).unwrap_or(mid);
        if sign > 0 && compare_iso_date(next, to) == core::cmp::Ordering::Greater
            || sign < 0 && compare_iso_date(next, to) == core::cmp::Ordering::Less
        {
            break;
        }
        months += sign;
        mid = next;
        if months.abs() > 12 {
            // Roll excess months into years (keeps months in (-12, 12)).
            years += sign;
            months -= 12 * sign;
        }
    }
    if largest == Unit::Month {
        months += years * 12;
        years = 0;
    }
    let days = iso_to_epoch_days(to) - iso_to_epoch_days(mid);
    (years, months, 0, days)
}

/// Total ordering on ISO dates.
#[must_use]
pub fn compare_iso_date(a: IsoDate, b: IsoDate) -> core::cmp::Ordering {
    (a.year, a.month, a.day).cmp(&(b.year, b.month, b.day))
}

/// Total ordering on wall-clock times.
#[must_use]
pub fn compare_iso_time(a: IsoTime, b: IsoTime) -> core::cmp::Ordering {
    time_to_nanos(a).cmp(&time_to_nanos(b))
}

// ---------------------------------------------------------------------------
// Duration helpers
// ---------------------------------------------------------------------------

impl DurationFields {
    /// The sign of a duration: -1, 0, or +1. The first non-zero field decides it
    /// (a valid duration has a consistent sign, but this tolerates any input).
    #[must_use]
    pub fn sign(&self) -> i64 {
        for v in [
            self.years,
            self.months,
            self.weeks,
            self.days,
            self.hours,
            self.minutes,
            self.seconds,
            self.milliseconds,
            self.microseconds,
            self.nanoseconds,
        ] {
            if v != 0 {
                return v.signum();
            }
        }
        0
    }

    /// Whether all non-zero fields share one sign (DurationSign well-formedness).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let mut sign = 0_i64;
        for v in [
            self.years,
            self.months,
            self.weeks,
            self.days,
            self.hours,
            self.minutes,
            self.seconds,
            self.milliseconds,
            self.microseconds,
            self.nanoseconds,
        ] {
            if v != 0 {
                let s = v.signum();
                if sign != 0 && s != sign {
                    return false;
                }
                sign = s;
            }
        }
        true
    }

    /// The sub-day portion expressed as a nanosecond count (hours…nanoseconds).
    #[must_use]
    pub fn time_nanos(&self) -> i128 {
        i128::from(self.hours) * NS_PER_HOUR
            + i128::from(self.minutes) * NS_PER_MINUTE
            + i128::from(self.seconds) * NS_PER_SEC
            + i128::from(self.milliseconds) * 1_000_000
            + i128::from(self.microseconds) * 1_000
            + i128::from(self.nanoseconds)
    }
}

/// Balances a raw nanosecond total into duration time fields down to
/// `largest_unit` (Hour..Nanosecond). Used by Duration.round / Instant / Time.
#[must_use]
pub fn balance_time_duration(total_ns: i128, largest: Unit) -> DurationFields {
    let sign = total_ns.signum();
    let mut r = total_ns.abs();
    let mut d = DurationFields::default();
    let mut set = |field: &mut i64, per: i128, active: bool| {
        if active {
            *field = (r / per) as i64 * sign as i64;
            r %= per;
        }
    };
    set(&mut d.hours, NS_PER_HOUR, largest <= Unit::Hour);
    set(&mut d.minutes, NS_PER_MINUTE, largest <= Unit::Minute);
    set(&mut d.seconds, NS_PER_SEC, largest <= Unit::Second);
    set(&mut d.milliseconds, 1_000_000, largest <= Unit::Millisecond);
    set(&mut d.microseconds, 1_000, largest <= Unit::Microsecond);
    d.nanoseconds = r as i64 * sign as i64;
    d
}

/// Rounds `x` to the nearest multiple of `increment` using `mode`.
#[must_use]
pub fn round_to_increment(x: i128, increment: i128, mode: RoundMode) -> i128 {
    if increment <= 1 {
        return x;
    }
    let q = x.div_euclid(increment);
    let r = x.rem_euclid(increment);
    if r == 0 {
        return x;
    }
    let lower = q * increment;
    let upper = lower + increment;
    let pick_upper = match mode {
        RoundMode::Ceil | RoundMode::Expand => true,
        RoundMode::Floor | RoundMode::Trunc => false,
        RoundMode::HalfCeil | RoundMode::HalfExpand => 2 * r >= increment,
        RoundMode::HalfFloor | RoundMode::HalfTrunc => 2 * r > increment,
        RoundMode::HalfEven => {
            if 2 * r == increment {
                (q % 2) != 0
            } else {
                2 * r > increment
            }
        }
    };
    // Ceil/Floor/Trunc/Expand are direction-sensitive vs sign; the div_euclid
    // above already floors, so Trunc toward zero needs adjustment for negatives.
    match mode {
        RoundMode::Trunc => {
            if x >= 0 {
                lower
            } else {
                upper
            }
        }
        RoundMode::Expand => {
            if x >= 0 {
                upper
            } else {
                lower
            }
        }
        _ => {
            if pick_upper {
                upper
            } else {
                lower
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Zero-pads `v` to at least `width` digits.
#[must_use]
pub fn pad(v: u64, width: usize) -> String {
    let s = alloc::format!("{v}");
    if s.len() >= width {
        s
    } else {
        let mut out = String::with_capacity(width);
        for _ in 0..width - s.len() {
            out.push('0');
        }
        out.push_str(&s);
        out
    }
}

/// Formats an ISO year: 4 digits, or `±NNNNNN` (6 digits, signed) outside
/// [0, 9999] (ExpandedYear).
#[must_use]
pub fn format_iso_year(year: i32) -> String {
    if (0..=9999).contains(&year) {
        pad(year as u64, 4)
    } else {
        let sign = if year < 0 { '-' } else { '+' };
        alloc::format!("{sign}{}", pad(year.unsigned_abs() as u64, 6))
    }
}

/// Formats the fractional-seconds part (`.NNN…`) for a time given its sub-second
/// nanoseconds and a `precision`: `None` = auto (trim trailing zeros, omit if
/// zero), `Some(n)` = exactly `n` fractional digits.
#[must_use]
pub fn format_fraction(sub_second_ns: u32, precision: Option<u8>) -> String {
    match precision {
        Some(0) => String::new(),
        Some(p) => {
            let full = pad(u64::from(sub_second_ns), 9);
            alloc::format!(".{}", &full[..p as usize])
        }
        None => {
            if sub_second_ns == 0 {
                return String::new();
            }
            let full = pad(u64::from(sub_second_ns), 9);
            let trimmed = full.trim_end_matches('0');
            alloc::format!(".{trimmed}")
        }
    }
}

// ---------------------------------------------------------------------------
// ISO-8601 parsing (Temporal grammar subset)
// ---------------------------------------------------------------------------

/// The parsed pieces of a Temporal ISO string. Fields absent in the input are
/// `None`; `offset` is minutes east of UTC (or `Some(i64::MIN)` sentinel for a
/// bare `Z`); `calendar`/`tz` carry annotation text when present.
#[derive(Clone, Debug, Default)]
pub struct ParsedIso {
    pub date: Option<IsoDate>,
    pub time: Option<IsoTime>,
    pub offset_ns: Option<i128>,
    pub z: bool,
    pub tz_name: Option<String>,
    pub calendar: Option<String>,
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}
impl Cursor<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    /// Consumes a sign: ASCII `+`/`-` or U+2212 MINUS SIGN (`0xE2 0x88 0x92`).
    /// Returns `Some(true)` for a minus, `Some(false)` for a plus.
    fn eat_sign(&mut self) -> Option<bool> {
        match self.peek() {
            Some(b'+') => {
                self.i += 1;
                Some(false)
            }
            Some(b'-') => {
                self.i += 1;
                Some(true)
            }
            Some(0xE2)
                if self.b.get(self.i + 1) == Some(&0x88)
                    && self.b.get(self.i + 2) == Some(&0x92) =>
            {
                self.i += 3;
                Some(true)
            }
            _ => None,
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
}

/// Parses a Temporal date/datetime/time string into its components. Accepts the
/// common grammar: `±YYYYYY`/`YYYY` `-`? `MM` `-`? `DD` (`T`|` `) time,
/// optional fractional seconds, optional `Z`/offset, optional `[tz]` and
/// `[u-ca=cal]` annotations. Returns `None` on a malformed string.
#[must_use]
pub fn parse_iso_datetime(s: &str) -> Option<ParsedIso> {
    let mut c = Cursor {
        b: s.as_bytes(),
        i: 0,
    };
    let mut out = ParsedIso::default();

    // A leading date is required unless the string is a bare time (starts with
    // `T` or looks like `HH:MM`). Try date first.
    let start = c.i;
    if let Some(date) = parse_date(&mut c) {
        out.date = Some(date);
        // Optional time, introduced by `T`/`t`/space.
        if c.eat(b'T') || c.eat(b't') || c.eat(b' ') {
            let (time, off, z) = parse_time_and_offset(&mut c)?;
            out.time = Some(time);
            out.offset_ns = off;
            out.z = z;
        }
    } else {
        c.i = start;
        // Bare time: optional leading `T`.
        let _ = c.eat(b'T') || c.eat(b't');
        let (time, off, z) = parse_time_and_offset(&mut c)?;
        out.time = Some(time);
        out.offset_ns = off;
        out.z = z;
    }

    parse_annotations(&mut c, &mut out);
    // Must be fully consumed.
    if c.i != c.b.len() {
        return None;
    }
    Some(out)
}

fn parse_date(c: &mut Cursor) -> Option<IsoDate> {
    // Year: ±YYYYYY (6-digit signed) or YYYY.
    let year = if let Some(neg) = c.eat_sign() {
        let y = c.digits(6)?;
        let y = if neg { -y } else { y };
        if neg && y == 0 {
            return None; // -000000 is invalid
        }
        y
    } else {
        c.digits(4)?
    };
    c.eat(b'-');
    let month = c.digits(2)?;
    c.eat(b'-');
    let day = c.digits(2)?;
    regulate_iso_date(year as i32, month, day, Overflow::Reject)
}

fn parse_time_and_offset(c: &mut Cursor) -> Option<(IsoTime, Option<i128>, bool)> {
    let hour = c.digits(2)?;
    let mut minute = 0;
    let mut second = 0;
    let mut frac_ns = 0_i64;
    if c.eat(b':') {
        minute = c.digits(2)?;
        if c.eat(b':') {
            second = c.digits(2)?;
            frac_ns = parse_fraction(c);
        }
    } else if c.peek().is_some_and(|b| b.is_ascii_digit()) {
        minute = c.digits(2)?;
        if c.peek().is_some_and(|b| b.is_ascii_digit()) {
            second = c.digits(2)?;
            frac_ns = parse_fraction(c);
        }
    }
    let time = regulate_iso_time(
        hour,
        minute,
        second,
        frac_ns / 1_000_000,
        (frac_ns / 1_000) % 1_000,
        frac_ns % 1_000,
        Overflow::Reject,
    )?;
    let (off, z) = parse_offset(c);
    Some((time, off, z))
}

/// Parses a `.` or `,` fractional-seconds group into nanoseconds (0 if absent).
fn parse_fraction(c: &mut Cursor) -> i64 {
    if c.eat(b'.') || c.eat(b',') {
        let mut ns = 0_i64;
        let mut count = 0;
        while count < 9 {
            match c.peek() {
                Some(b) if b.is_ascii_digit() => {
                    ns = ns * 10 + i64::from(b - b'0');
                    c.i += 1;
                    count += 1;
                }
                _ => break,
            }
        }
        // Consume any excess digits (spec allows only up to 9, but be lenient).
        while c.peek().is_some_and(|b| b.is_ascii_digit()) {
            c.i += 1;
        }
        for _ in count..9 {
            ns *= 10;
        }
        ns
    } else {
        0
    }
}

fn parse_offset(c: &mut Cursor) -> (Option<i128>, bool) {
    if c.eat(b'Z') || c.eat(b'z') {
        return (Some(0), true);
    }
    if let Some(neg) = c.eat_sign() {
        let Some(h) = c.digits(2) else {
            return (None, false);
        };
        let mut m = 0;
        let mut s = 0;
        let mut frac = 0_i64;
        if c.eat(b':') {
            m = c.digits(2).unwrap_or(0);
            if c.eat(b':') {
                s = c.digits(2).unwrap_or(0);
                frac = parse_fraction(c);
            }
        } else if c.peek().is_some_and(|b| b.is_ascii_digit()) {
            m = c.digits(2).unwrap_or(0);
        }
        let ns = i128::from(h) * NS_PER_HOUR
            + i128::from(m) * NS_PER_MINUTE
            + i128::from(s) * NS_PER_SEC
            + i128::from(frac);
        return (Some(if neg { -ns } else { ns }), false);
    }
    (None, false)
}

fn parse_annotations(c: &mut Cursor, out: &mut ParsedIso) {
    while c.eat(b'[') {
        let critical = c.eat(b'!');
        let _ = critical;
        let start = c.i;
        while c.peek().is_some_and(|b| b != b']') {
            c.i += 1;
        }
        let inner = core::str::from_utf8(&c.b[start..c.i]).unwrap_or("");
        if !c.eat(b']') {
            return;
        }
        if let Some(cal) = inner.strip_prefix("u-ca=") {
            out.calendar = Some(String::from(cal));
        } else if !inner.is_empty() {
            out.tz_name = Some(String::from(inner));
        }
    }
}

/// Parses a Temporal ISO 8601 **duration** string (`±P…`) into fields. Returns
/// `None` if malformed. Fractional values are only allowed on the smallest
/// present time unit; the fraction cascades into the smaller nanosecond fields.
#[must_use]
pub fn parse_iso_duration(s: &str) -> Option<DurationFields> {
    let mut c = Cursor {
        b: s.as_bytes(),
        i: 0,
    };
    let sign = match c.eat_sign() {
        Some(true) => -1_i64,
        _ => 1,
    };
    if !(c.eat(b'P') || c.eat(b'p')) {
        return None;
    }
    let mut d = DurationFields::default();
    let mut any = false;

    // Date portion: number+designator for Y, M, W, D.
    let mut last_date = 0;
    while let Some((n, _frac)) = peek_number(&mut c) {
        let desig = c.peek()?;
        let ord = match desig.to_ascii_uppercase() {
            b'Y' => 1,
            b'M' => 2,
            b'W' => 3,
            b'D' => 4,
            _ => break,
        };
        if ord <= last_date {
            return None; // out of order / duplicate
        }
        last_date = ord;
        c.i += 1;
        match ord {
            1 => d.years = n * sign,
            2 => d.months = n * sign,
            3 => d.weeks = n * sign,
            _ => d.days = n * sign,
        }
        any = true;
    }

    // Time portion: `T` then H, M, S (S may be fractional).
    if c.eat(b'T') || c.eat(b't') {
        let mut last_time = 0;
        let mut seen_time = false;
        while let Some((n, frac)) = peek_number(&mut c) {
            let desig = c.peek()?;
            let ord = match desig.to_ascii_uppercase() {
                b'H' => 1,
                b'M' => 2,
                b'S' => 3,
                _ => return None,
            };
            if ord <= last_time {
                return None;
            }
            last_time = ord;
            c.i += 1;
            seen_time = true;
            any = true;
            match ord {
                1 => {
                    d.hours = n * sign;
                    if let Some(f) = frac {
                        distribute_fraction(&mut d, f, NS_PER_HOUR, sign);
                        break;
                    }
                }
                2 => {
                    d.minutes = n * sign;
                    if let Some(f) = frac {
                        distribute_fraction(&mut d, f, NS_PER_MINUTE, sign);
                        break;
                    }
                }
                _ => {
                    d.seconds = n * sign;
                    if let Some(f) = frac {
                        distribute_fraction(&mut d, f, NS_PER_SEC, sign);
                    }
                }
            }
        }
        if !seen_time {
            return None; // `T` with no time components
        }
    }

    if !any || c.i != c.b.len() {
        return None;
    }
    d.is_valid().then_some(d)
}

/// Reads an integer (optionally with a `.`/`,` fraction, returned as a 0..1e9
/// nanosecond-scaled fraction) without consuming a trailing designator.
fn peek_number(c: &mut Cursor) -> Option<(i64, Option<i64>)> {
    if !c.peek().is_some_and(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut n = 0_i64;
    while c.peek().is_some_and(|b| b.is_ascii_digit()) {
        n = n * 10 + i64::from(c.peek().unwrap() - b'0');
        c.i += 1;
    }
    let frac = if c.peek() == Some(b'.') || c.peek() == Some(b',') {
        c.i += 1;
        let mut f = 0_i64;
        let mut cnt = 0;
        while cnt < 9 && c.peek().is_some_and(|b| b.is_ascii_digit()) {
            f = f * 10 + i64::from(c.peek().unwrap() - b'0');
            c.i += 1;
            cnt += 1;
        }
        while c.peek().is_some_and(|b| b.is_ascii_digit()) {
            c.i += 1;
        }
        for _ in cnt..9 {
            f *= 10;
        }
        Some(f)
    } else {
        None
    };
    Some((n, frac))
}

/// Distributes a fractional part (in 1e-9 units of `per`-nanosecond-worth) down
/// into seconds/millis/micros/nanos of `d`.
fn distribute_fraction(d: &mut DurationFields, frac_1e9: i64, per: i128, sign: i64) {
    let total_ns = i128::from(frac_1e9) * per / NS_PER_SEC;
    let mut r = total_ns;
    d.seconds += (r / NS_PER_SEC) as i64 * sign;
    r %= NS_PER_SEC;
    d.milliseconds += (r / 1_000_000) as i64 * sign;
    r %= 1_000_000;
    d.microseconds += (r / 1_000) as i64 * sign;
    r %= 1_000;
    d.nanoseconds += r as i64 * sign;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_round_trip() {
        for &(y, m, dd) in &[
            (1970, 1, 1),
            (2000, 2, 29),
            (1, 1, 1),
            (-1, 12, 31),
            (275760, 9, 13),
        ] {
            let date = IsoDate {
                year: y,
                month: m,
                day: dd,
            };
            assert_eq!(epoch_days_to_iso(iso_to_epoch_days(date)), date);
        }
        assert_eq!(
            iso_to_epoch_days(IsoDate {
                year: 1970,
                month: 1,
                day: 1
            }),
            0
        );
        assert_eq!(
            iso_to_epoch_days(IsoDate {
                year: 1970,
                month: 1,
                day: 2
            }),
            1
        );
        assert_eq!(
            iso_to_epoch_days(IsoDate {
                year: 1969,
                month: 12,
                day: 31
            }),
            -1
        );
    }

    #[test]
    fn day_of_week_known() {
        // 2020-01-01 was a Wednesday (ISO 3).
        assert_eq!(
            iso_day_of_week(IsoDate {
                year: 2020,
                month: 1,
                day: 1
            }),
            3
        );
        // 1970-01-01 was a Thursday (ISO 4).
        assert_eq!(
            iso_day_of_week(IsoDate {
                year: 1970,
                month: 1,
                day: 1
            }),
            4
        );
    }

    #[test]
    fn leap_and_dim() {
        assert!(is_leap_year(2000));
        assert!(!is_leap_year(1900));
        assert!(is_leap_year(2024));
        assert_eq!(iso_days_in_month(2024, 2), 29);
        assert_eq!(iso_days_in_month(2023, 2), 28);
    }

    #[test]
    fn parse_basic() {
        let p = parse_iso_datetime("2020-03-15T12:30:45.5").unwrap();
        assert_eq!(
            p.date.unwrap(),
            IsoDate {
                year: 2020,
                month: 3,
                day: 15
            }
        );
        let t = p.time.unwrap();
        assert_eq!(
            (t.hour, t.minute, t.second, t.millisecond),
            (12, 30, 45, 500)
        );
        let z = parse_iso_datetime("2020-03-15T12:30:45Z").unwrap();
        assert!(z.z);
    }

    #[test]
    fn parse_duration_basic() {
        let d = parse_iso_duration("P1Y2M3DT4H5M6.5S").unwrap();
        assert_eq!(
            (d.years, d.months, d.days, d.hours, d.minutes, d.seconds),
            (1, 2, 3, 4, 5, 6)
        );
        assert_eq!(d.milliseconds, 500);
        assert!(parse_iso_duration("P").is_none());
        assert_eq!(parse_iso_duration("-P1D").unwrap().days, -1);
    }

    #[test]
    fn add_date_overflow() {
        // Jan 31 + 1 month, constrain → Feb 28/29.
        let jan31 = IsoDate {
            year: 2023,
            month: 1,
            day: 31,
        };
        assert_eq!(
            add_iso_date(jan31, 0, 1, 0, 0, Overflow::Constrain).unwrap(),
            IsoDate {
                year: 2023,
                month: 2,
                day: 28
            }
        );
        assert_eq!(add_iso_date(jan31, 0, 1, 0, 0, Overflow::Reject), None);
    }
}
