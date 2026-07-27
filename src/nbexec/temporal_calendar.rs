//! Non-ISO calendar abstraction for `Temporal` (Wave 1 foundation).
//!
//! A **purely additive** layer over the ISO-8601 core in
//! [`crate::temporal_iso`]. The engine stores every date as a proleptic
//! Gregorian [`IsoDate`]; this module converts that pivot to and from a
//! *calendar's* era/year/month/day fields on demand, so the ISO fast path is
//! never disturbed. It is consulted only when a `PlainDate`'s `[[Calendar]]`
//! identifier is something other than `"iso8601"`.
//!
//! The Julian Day Number (JDN) is the pivot: `IsoDate → JDN → <calendar>` and
//! back. Proleptic-Gregorian JDN math and the pure-arithmetic calendars
//! (Coptic, Ethiopic, Indian) are implemented here directly; the lunisolar and
//! era calendars (Islamic, Persian, Hebrew, Chinese, Japanese) delegate to the
//! `intl` crate's `calendar` module and are gated on the `intl` feature. Built
//! without `intl`, only `"iso8601"` and the Gregorian-family / pure-arithmetic
//! calendars produce meaningful results; the crate-backed ones degrade to the
//! ISO date (documented — no_std / no-intl builds still compile and pass the
//! ISO suite).
//!
//! Public surface (reused by later waves for PlainDateTime / PlainYearMonth /
//! PlainMonthDay / ZonedDateTime):
//! - [`canonicalize_calendar`] — id validation + CLDR alias canonicalization.
//! - [`CalFields`] / [`iso_to_fields`] — IsoDate → calendar fields.
//! - [`FieldsInput`] / [`fields_to_iso`] — calendar fields → IsoDate.
//! - Derived accessors: [`days_in_month`], [`days_in_year`], [`months_in_year`],
//!   [`in_leap_year`], [`day_of_week`], [`day_of_year`], [`week_of_year`],
//!   [`year_of_week`], [`days_in_week`].

use crate::temporal_iso::{IsoDate, MAX_EPOCH_DAYS, MIN_EPOCH_DAYS, Overflow, Unit};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Canonical calendar-id table + validation
// ---------------------------------------------------------------------------

/// The canonical (already-lowercase) calendar identifiers Temporal accepts, with
/// their CLDR aliases. Note that the bare `"islamic"` and `"islamic-rgsa"` ids
/// are deliberately **absent**: per Test262 they are only recognised by
/// `Intl.DateTimeFormat`, not by Temporal, so they must be rejected here.
const CALENDAR_ALIASES: &[(&str, &[&str])] = &[
    ("iso8601", &[]),
    ("gregory", &["gregorian"]),
    ("buddhist", &[]),
    ("japanese", &[]),
    ("roc", &["minguo"]),
    ("persian", &[]),
    ("islamic-civil", &["islamicc"]),
    ("islamic-tbla", &[]),
    ("islamic-umalqura", &[]),
    ("hebrew", &[]),
    ("chinese", &[]),
    ("dangi", &[]),
    ("indian", &[]),
    ("coptic", &[]),
    ("ethiopic", &[]),
    ("ethioaa", &["ethiopic-amete-alem"]),
];

/// Canonicalizes a calendar identifier (ASCII-case-insensitively, resolving CLDR
/// aliases). Returns the `'static` canonical id, or `None` if the id is not a
/// Temporal-supported calendar (the caller then throws a RangeError).
#[must_use]
pub(crate) fn canonicalize_calendar(s: &str) -> Option<&'static str> {
    if !s.is_ascii() {
        return None;
    }
    for &(canon, aliases) in CALENDAR_ALIASES {
        if s.eq_ignore_ascii_case(canon) {
            return Some(canon);
        }
        for &a in aliases {
            if s.eq_ignore_ascii_case(a) {
                return Some(canon);
            }
        }
    }
    None
}

/// Whether `cal` is the ISO-8601 calendar (the engine's fast path).
#[must_use]
pub(crate) fn is_iso(cal: &str) -> bool {
    cal == "iso8601"
}

// ---------------------------------------------------------------------------
// Field records
// ---------------------------------------------------------------------------

/// Calendar fields extracted from an [`IsoDate`] for a given calendar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CalFields {
    /// Era code (e.g. `"ce"`, `"ah"`), or `None` for era-less calendars.
    pub era: Option<String>,
    /// Year within the era, or `None` for era-less calendars.
    pub era_year: Option<i64>,
    /// The calendar's arithmetic year (the `.year` getter value).
    pub year: i64,
    /// 1-based ordinal month position within the calendar year.
    pub month: i64,
    /// Month code: `"M01".."M13"`, with a trailing `L` for a leap month
    /// (`"M05L"`).
    pub month_code: String,
    /// 1-based day of month.
    pub day: i64,
}

/// Input to [`fields_to_iso`] (`CalendarDateFromFields`). Either
/// `era` + `era_year` **or** `year` supplies the year; either `month` (1-based
/// ordinal) **or** `month_code` supplies the month.
#[derive(Clone, Debug, Default)]
pub(crate) struct FieldsInput {
    pub era: Option<String>,
    pub era_year: Option<i64>,
    pub year: Option<i64>,
    pub month: Option<i64>,
    pub month_code: Option<String>,
    pub day: i64,
}

/// A calendar-math error, carrying a RangeError message for the caller to throw.
#[derive(Clone, Debug)]
pub(crate) enum CalError {
    Range(String),
    /// A required field (year/month) was absent — the caller throws a TypeError.
    MissingFields(String),
}

// ---------------------------------------------------------------------------
// JDN pivot (proleptic Gregorian) — pure arithmetic, always available
// ---------------------------------------------------------------------------

/// The Julian Day Number of a proleptic Gregorian `(year, month, day)`.
fn greg_to_jdn(year: i64, month: i64, day: i64) -> i64 {
    let a = (14 - month).div_euclid(12);
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    // Floor division (`div_euclid`) throughout: for proleptic dates far in the BC
    // era `y` goes negative, where truncating `/` would give the wrong leap-day
    // count. For any date after ~4801 BC (`y >= 0`) this is identical to `/`.
    day + (153 * m + 2).div_euclid(5) + 365 * y + y.div_euclid(4) - y.div_euclid(100)
        + y.div_euclid(400)
        - 32045
}

/// The proleptic Gregorian `(year, month, day)` of a Julian Day Number.
fn jdn_to_greg(jdn: i64) -> (i64, i64, i64) {
    // Floor division throughout: the intermediate `a = jdn + 32044` goes negative
    // for JDNs before ~4801 BC, where truncating `/` on the century terms would be
    // off by one. For any JDN >= -32044 this matches the classic truncating form.
    let a = jdn + 32044;
    let b = (4 * a + 3).div_euclid(146097);
    let c = a - (146097 * b).div_euclid(4);
    let d = (4 * c + 3).div_euclid(1461);
    let e = c - (1461 * d).div_euclid(4);
    let m = (5 * e + 2).div_euclid(153);
    let day = e - (153 * m + 2).div_euclid(5) + 1;
    let month = m + 3 - 12 * m.div_euclid(10);
    let year = 100 * b + d - 4800 + m.div_euclid(10);
    (year, month, day)
}

fn iso_to_jdn(iso: IsoDate) -> i64 {
    greg_to_jdn(
        i64::from(iso.year),
        i64::from(iso.month),
        i64::from(iso.day),
    )
}

fn jdn_to_iso(jdn: i64) -> IsoDate {
    let (y, m, d) = jdn_to_greg(jdn);
    IsoDate {
        year: y as i32,
        month: m as u8,
        day: d as u8,
    }
}

fn greg_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// ---------------------------------------------------------------------------
// Pure-arithmetic calendars: Coptic, Ethiopic (13-month, 30×12 + epagomenal)
// ---------------------------------------------------------------------------

/// JDN of Coptic 1-01-01 (1 Thout, year 1 = Anno Martyrum).
const COPTIC_EPOCH: i64 = 1_825_030;
/// JDN of Ethiopic 1-01-01 (1 Mäskäräm, Amete Mihret year 1).
const ETHIOPIC_EPOCH: i64 = 1_724_221;

/// Shared 13-month (12×30 + a 5/6-day epagomenal month) forward conversion.
fn coptic_like_to_jdn(epoch: i64, year: i64, month: i64, day: i64) -> i64 {
    epoch - 1 + 365 * (year - 1) + year.div_euclid(4) + 30 * (month - 1) + day
}

/// Shared 13-month reverse conversion → `(year, month, day)`.
fn coptic_like_from_jdn(epoch: i64, jdn: i64) -> (i64, i64, i64) {
    let mut year = (4 * (jdn - epoch) + 1463).div_euclid(1461);
    while coptic_like_to_jdn(epoch, year, 1, 1) > jdn {
        year -= 1;
    }
    while coptic_like_to_jdn(epoch, year + 1, 1, 1) <= jdn {
        year += 1;
    }
    let doy = jdn - coptic_like_to_jdn(epoch, year, 1, 1); // 0-based
    if doy < 360 {
        (year, doy / 30 + 1, doy % 30 + 1)
    } else {
        (year, 13, doy - 360 + 1)
    }
}

/// A 13-month calendar's leap year: the epagomenal month has 6 days when
/// `year mod 4 == 3`.
fn coptic_like_leap(year: i64) -> bool {
    year.rem_euclid(4) == 3
}

// ---------------------------------------------------------------------------
// Pure-arithmetic calendar: Indian national (Saka), solar, epoch 78 CE
// ---------------------------------------------------------------------------

fn indian_to_jdn(saka_year: i64, month: i64, day: i64) -> i64 {
    let greg_year = saka_year + 78;
    let leap = greg_leap(greg_year);
    let chaitra1 = greg_to_jdn(greg_year, 3, if leap { 21 } else { 22 });
    let mut jdn = chaitra1;
    if month == 1 {
        jdn += day - 1;
    } else {
        jdn += if leap { 31 } else { 30 }; // Chaitra
        let mut m = 2;
        while m < month {
            jdn += if m <= 6 { 31 } else { 30 };
            m += 1;
        }
        jdn += day - 1;
    }
    jdn
}

fn indian_from_jdn(jdn: i64) -> (i64, i64, i64) {
    let (gy, _, _) = jdn_to_greg(jdn);
    let leap_gy = greg_leap(gy);
    let chaitra1 = greg_to_jdn(gy, 3, if leap_gy { 21 } else { 22 });
    let (saka_year, mut yday, leap) = if jdn >= chaitra1 {
        (gy - 78, jdn - chaitra1, leap_gy)
    } else {
        let prev_leap = greg_leap(gy - 1);
        let prev = greg_to_jdn(gy - 1, 3, if prev_leap { 21 } else { 22 });
        (gy - 79, jdn - prev, prev_leap)
    };
    let chaitra_len = if leap { 31 } else { 30 };
    if yday < chaitra_len {
        (saka_year, 1, yday + 1)
    } else {
        yday -= chaitra_len;
        if yday < 5 * 31 {
            (saka_year, 2 + yday / 31, yday % 31 + 1)
        } else {
            yday -= 5 * 31;
            (saka_year, 7 + yday / 30, yday % 30 + 1)
        }
    }
}

fn indian_leap(saka_year: i64) -> bool {
    greg_leap(saka_year + 78)
}

// ---------------------------------------------------------------------------
// intl-crate-backed calendars (feature-gated); ISO fallback without `intl`
// ---------------------------------------------------------------------------

/// The per-variant JDN offset relative to the `intl` crate's tabular Islamic
/// algorithm. The crate implements the **civil** (Friday-epoch, Kuwaiti)
/// tabular calendar, so `islamic-civil` needs no offset; `islamic-tbla` (the
/// astronomical Thursday-epoch variant) is one day later, i.e. its epoch is one
/// day earlier — subtracting 1 from the pivot JDN advances the reported date by
/// a day. Verified against Test262: 2000-01-01 = 1420-09-24 (civil) / -09-25
/// (tbla). `islamic-umalqura` is handled separately via the intl crate's
/// dedicated Umm al-Qura table (`jdn_to_umalqura`/`umalqura_to_jdn`) and never
/// reaches this delta.
fn islamic_delta(cal: &str) -> i64 {
    match cal {
        "islamic-tbla" => -1,
        // islamic-civil (umalqura is routed to its own table below).
        _ => 0,
    }
}

#[cfg(feature = "intl")]
fn islamic_from_jdn(cal: &str, jdn: i64) -> (i64, i64, i64) {
    if cal == "islamic-umalqura" {
        // Real Umm al-Qura (Saudi) table; auto-falls-back to civil tabular
        // outside the tabulated range (AH 1300–1600).
        return intl::calendar::jdn_to_umalqura(jdn);
    }
    intl::calendar::jdn_to_islamic(jdn - islamic_delta(cal))
}
#[cfg(feature = "intl")]
fn islamic_to_jdn(cal: &str, y: i64, m: i64, d: i64) -> i64 {
    if cal == "islamic-umalqura" {
        return intl::calendar::umalqura_to_jdn(y, m, d);
    }
    intl::calendar::islamic_to_jdn(y, m, d) + islamic_delta(cal)
}
#[cfg(not(feature = "intl"))]
fn islamic_from_jdn(_cal: &str, jdn: i64) -> (i64, i64, i64) {
    jdn_to_greg(jdn)
}
#[cfg(not(feature = "intl"))]
fn islamic_to_jdn(_cal: &str, y: i64, m: i64, d: i64) -> i64 {
    greg_to_jdn(y, m, d)
}

/// JDN of 1 Farvardin, Persian (Solar Hijri) year 1.
const PERSIAN_EPOCH: i64 = 1_948_320;

/// Persian (Solar Hijri) `(year, month, day)` → JDN.
///
/// Uses the **33-year arithmetic rule** — `floor((8·year + 21) / 33)` leap days
/// before a year — which is what ICU (and therefore Temporal's `persian`
/// calendar) implements, and which reproduces the Iranian calendar authority's
/// Nowruz dates exactly across the corpus's 1206–1498 range. The 2820-year
/// (Birashk) cycle this used to use agrees with it for most years but drifts by
/// a day at cycle boundaries — it put Nowruz of 1210, 1243, 1404, 1437 and 1470
/// on the wrong ISO day.
///
/// Both the year term and the leap term are floor-divided, so the formula is
/// *proleptic*: year 0 exists as an ordinary year and non-positive years
/// round-trip, as Temporal requires (the `intl` crate's historical variant has
/// no year 0).
fn persian_to_jdn(year: i64, month: i64, day: i64) -> i64 {
    let month_days = if month <= 7 {
        (month - 1) * 31
    } else {
        (month - 1) * 30 + 6
    };
    PERSIAN_EPOCH - 1 + 365 * (year - 1) + (8 * year + 21).div_euclid(33) + month_days + day
}

/// The Persian (Solar Hijri) `(year, month, day)` of a Julian Day Number.
fn persian_from_jdn(jdn: i64) -> (i64, i64, i64) {
    let mut year = 475 + (jdn - PERSIAN_EPOCH).div_euclid(366);
    while persian_to_jdn(year, 1, 1) > jdn {
        year -= 1;
    }
    while persian_to_jdn(year + 1, 1, 1) <= jdn {
        year += 1;
    }
    let mut month = 1;
    while month < 12 && persian_to_jdn(year, month + 1, 1) <= jdn {
        month += 1;
    }
    let day = jdn - persian_to_jdn(year, month, 1) + 1;
    (year, month, day)
}

#[cfg(feature = "intl")]
fn hebrew_from_jdn(jdn: i64) -> (i64, i64, i64) {
    intl::calendar::jdn_to_hebrew(jdn)
}
#[cfg(feature = "intl")]
fn hebrew_to_jdn(y: i64, m: i64, d: i64) -> i64 {
    intl::calendar::hebrew_to_jdn(y, m, d)
}
#[cfg(not(feature = "intl"))]
fn hebrew_from_jdn(jdn: i64) -> (i64, i64, i64) {
    jdn_to_greg(jdn)
}
#[cfg(not(feature = "intl"))]
fn hebrew_to_jdn(y: i64, m: i64, d: i64) -> i64 {
    greg_to_jdn(y, m, d)
}

/// Whether a Hebrew year is a leap (13-month) year.
fn hebrew_leap(year: i64) -> bool {
    (7 * year + 1).rem_euclid(19) < 7
}

/// Both the Chinese and Korean dangi calendars are lunisolar with an identical
/// month/`leap` convention; they differ only in the meridian at which the New
/// Moon / solar term is observed (Beijing vs. Korea), so a few years carry a
/// different leap month or New-Year day. `cal` selects the underlying intl
/// table: `"dangi"` → the Korean table, anything else → the Chinese table.
#[cfg(feature = "intl")]
fn chinese_from_jdn(cal: &str, jdn: i64) -> Option<(i64, i64, i64, bool)> {
    if cal == "dangi" {
        intl::calendar::jdn_to_dangi(jdn)
    } else {
        intl::calendar::jdn_to_chinese(jdn)
    }
}
#[cfg(feature = "intl")]
fn chinese_to_jdn(cal: &str, y: i64, m: i64, d: i64, leap: bool) -> Option<i64> {
    if cal == "dangi" {
        intl::calendar::dangi_to_jdn(y, m, d, leap)
    } else {
        intl::calendar::chinese_to_jdn(y, m, d, leap)
    }
}
#[cfg(not(feature = "intl"))]
fn chinese_from_jdn(_cal: &str, jdn: i64) -> Option<(i64, i64, i64, bool)> {
    let (y, m, d) = jdn_to_greg(jdn);
    Some((y, m, d, false))
}
#[cfg(not(feature = "intl"))]
fn chinese_to_jdn(_cal: &str, y: i64, m: i64, d: i64, _leap: bool) -> Option<i64> {
    Some(greg_to_jdn(y, m, d))
}

/// The Chinese/dangi year's leap-month number (1..=12), or 0 if the year has
/// none. `cal` selects the meridian (see [`chinese_from_jdn`]).
fn chinese_leap_month(cal: &str, year: i64) -> i64 {
    for m in 1..=12 {
        if chinese_to_jdn(cal, year, m, 1, true).is_some() {
            return m;
        }
    }
    0
}

/// Whether an intercalary month numbered `num` ever occurs in the Chinese/Dangi
/// table (used to decide whether a `M<NN>L` leap code that is absent in a given
/// year is nonetheless a *valid* leap code that may be constrained onto `M<NN>`,
/// vs. a code — like `M01L`/`M12L` — that never carries a leap month and is
/// therefore invalid). A leap `M<NN>L` code is only constrainable for an `NN`
/// that is astronomically an intercalary month somewhere in the supported range.
fn chinese_leap_num_occurs(cal: &str, num: i64) -> bool {
    if !(1..=12).contains(&num) {
        return false;
    }
    // The intl table covers years 1900–2099; a leap month numbered `num`
    // that never appears there is not a real intercalary month for this model.
    (1900..=2099).any(|y| chinese_leap_month(cal, y) == num)
}

// ---------------------------------------------------------------------------
// Internal calendar "parts" — arithmetic year, ordinal month, day, code
// ---------------------------------------------------------------------------

struct Parts {
    year: i64,
    month: i64, // 1-based ordinal
    day: i64,
    code: String,
    _leap: bool,
}

/// Hebrew civil month order (Tishrei-first), returning `(intl_month, code)` for
/// each ordinal position in the given year.
fn hebrew_month_list(year: i64) -> Vec<(i64, String)> {
    let leap = hebrew_leap(year);
    // intl numbers: 7=Tishrei … 13=Adar II, then 1=Nisan … 6=Elul.
    let seq: &[i64] = if leap {
        &[7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
    } else {
        &[7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    };
    seq.iter()
        .map(|&m| {
            let code = match m {
                7 => "M01".to_string(),
                8 => "M02".to_string(),
                9 => "M03".to_string(),
                10 => "M04".to_string(),
                11 => "M05".to_string(),
                12 => {
                    if leap {
                        "M05L".to_string()
                    } else {
                        "M06".to_string()
                    }
                }
                13 => "M06".to_string(),
                1 => "M07".to_string(),
                2 => "M08".to_string(),
                3 => "M09".to_string(),
                4 => "M10".to_string(),
                5 => "M11".to_string(),
                _ => "M12".to_string(),
            };
            (m, code)
        })
        .collect()
}

/// Chinese civil month order, returning `(nominal_month, is_leap, code)` for each
/// ordinal position in the year.
fn chinese_month_list(cal: &str, year: i64) -> Vec<(i64, bool, String)> {
    let leap_m = chinese_leap_month(cal, year);
    let mut out = Vec::new();
    for m in 1..=12 {
        out.push((m, false, format!("M{m:02}")));
        if m == leap_m {
            out.push((m, true, format!("M{m:02}L")));
        }
    }
    out
}

/// Converts an [`IsoDate`] to a calendar's arithmetic-year / ordinal-month parts.
fn parts_from_iso(cal: &str, iso: IsoDate) -> Parts {
    let jdn = iso_to_jdn(iso);
    match cal {
        "gregory" | "japanese" => Parts {
            year: i64::from(iso.year),
            month: i64::from(iso.month),
            day: i64::from(iso.day),
            code: format!("M{:02}", iso.month),
            _leap: false,
        },
        "buddhist" => Parts {
            year: i64::from(iso.year) + 543,
            month: i64::from(iso.month),
            day: i64::from(iso.day),
            code: format!("M{:02}", iso.month),
            _leap: false,
        },
        "roc" => Parts {
            year: i64::from(iso.year) - 1911,
            month: i64::from(iso.month),
            day: i64::from(iso.day),
            code: format!("M{:02}", iso.month),
            _leap: false,
        },
        "persian" => {
            let (y, m, d) = persian_from_jdn(jdn);
            Parts {
                year: y,
                month: m,
                day: d,
                code: format!("M{m:02}"),
                _leap: false,
            }
        }
        "islamic-civil" | "islamic-tbla" | "islamic-umalqura" => {
            let (y, m, d) = islamic_from_jdn(cal, jdn);
            Parts {
                year: y,
                month: m,
                day: d,
                code: format!("M{m:02}"),
                _leap: false,
            }
        }
        "indian" => {
            let (y, m, d) = indian_from_jdn(jdn);
            Parts {
                year: y,
                month: m,
                day: d,
                code: format!("M{m:02}"),
                _leap: false,
            }
        }
        "coptic" => {
            let (y, m, d) = coptic_like_from_jdn(COPTIC_EPOCH, jdn);
            Parts {
                year: y,
                month: m,
                day: d,
                code: format!("M{m:02}"),
                _leap: false,
            }
        }
        "ethiopic" => {
            let (y, m, d) = coptic_like_from_jdn(ETHIOPIC_EPOCH, jdn);
            Parts {
                year: y,
                month: m,
                day: d,
                code: format!("M{m:02}"),
                _leap: false,
            }
        }
        "ethioaa" => {
            let (y, m, d) = coptic_like_from_jdn(ETHIOPIC_EPOCH, jdn);
            Parts {
                year: y + 5500,
                month: m,
                day: d,
                code: format!("M{m:02}"),
                _leap: false,
            }
        }
        "hebrew" => {
            let (y, im, d) = hebrew_from_jdn(jdn);
            let list = hebrew_month_list(y);
            let (ord, code) = list
                .iter()
                .enumerate()
                .find(|(_, (m, _))| *m == im)
                .map(|(i, (_, c))| (i as i64 + 1, c.clone()))
                .unwrap_or((im, format!("M{im:02}")));
            Parts {
                year: y,
                month: ord,
                day: d,
                code,
                _leap: false,
            }
        }
        "chinese" | "dangi" => {
            if let Some((y, nominal, d, leap)) = chinese_from_jdn(cal, jdn) {
                let list = chinese_month_list(cal, y);
                let (ord, code) = list
                    .iter()
                    .enumerate()
                    .find(|(_, (m, lp, _))| *m == nominal && *lp == leap)
                    .map(|(i, (_, _, c))| (i as i64 + 1, c.clone()))
                    .unwrap_or((nominal, format!("M{nominal:02}")));
                Parts {
                    year: y,
                    month: ord,
                    day: d,
                    code,
                    _leap: leap,
                }
            } else {
                // Out of the intl table's supported range: fall back to ISO.
                Parts {
                    year: i64::from(iso.year),
                    month: i64::from(iso.month),
                    day: i64::from(iso.day),
                    code: format!("M{:02}", iso.month),
                    _leap: false,
                }
            }
        }
        // iso8601 and unknown ids.
        _ => Parts {
            year: i64::from(iso.year),
            month: i64::from(iso.month),
            day: i64::from(iso.day),
            code: format!("M{:02}", iso.month),
            _leap: false,
        },
    }
}

/// Converts a calendar's arithmetic year + ordinal month + day back to an
/// [`IsoDate`]. Returns `None` if the value is not representable (e.g. a Chinese
/// year outside the supported table).
fn iso_from_parts(cal: &str, year: i64, ord_month: i64, day: i64) -> Option<IsoDate> {
    let iso = match cal {
        "gregory" | "japanese" => jdn_to_iso(greg_to_jdn(year, ord_month, day)),
        "buddhist" => jdn_to_iso(greg_to_jdn(year - 543, ord_month, day)),
        "roc" => jdn_to_iso(greg_to_jdn(year + 1911, ord_month, day)),
        "persian" => jdn_to_iso(persian_to_jdn(year, ord_month, day)),
        "islamic-civil" | "islamic-tbla" | "islamic-umalqura" => {
            jdn_to_iso(islamic_to_jdn(cal, year, ord_month, day))
        }
        "indian" => jdn_to_iso(indian_to_jdn(year, ord_month, day)),
        "coptic" => jdn_to_iso(coptic_like_to_jdn(COPTIC_EPOCH, year, ord_month, day)),
        "ethiopic" => jdn_to_iso(coptic_like_to_jdn(ETHIOPIC_EPOCH, year, ord_month, day)),
        "ethioaa" => jdn_to_iso(coptic_like_to_jdn(
            ETHIOPIC_EPOCH,
            year - 5500,
            ord_month,
            day,
        )),
        "hebrew" => {
            let list = hebrew_month_list(year);
            let intl_m = list.get((ord_month - 1) as usize).map(|(m, _)| *m)?;
            jdn_to_iso(hebrew_to_jdn(year, intl_m, day))
        }
        "chinese" | "dangi" => {
            let list = chinese_month_list(cal, year);
            let (nominal, leap) = list
                .get((ord_month - 1) as usize)
                .map(|(m, l, _)| (*m, *l))?;
            jdn_to_iso(chinese_to_jdn(cal, year, nominal, day, leap)?)
        }
        _ => jdn_to_iso(greg_to_jdn(year, ord_month, day)),
    };
    Some(iso)
}

// ---------------------------------------------------------------------------
// Era handling
// ---------------------------------------------------------------------------

/// Whether the calendar exposes an era / eraYear pair.
#[must_use]
pub(crate) fn has_eras(cal: &str) -> bool {
    !matches!(cal, "iso8601" | "chinese" | "dangi")
}

/// Derives `(era, eraYear)` from a calendar id + arithmetic year + IsoDate.
fn derive_era(cal: &str, year: i64, iso: IsoDate) -> (Option<String>, Option<i64>) {
    let dual = |pos: &str, neg: &str, y: i64| {
        if y >= 1 {
            (Some(pos.to_string()), Some(y))
        } else {
            (Some(neg.to_string()), Some(1 - y))
        }
    };
    match cal {
        "iso8601" | "chinese" | "dangi" => (None, None),
        "gregory" => dual("ce", "bce", i64::from(iso.year)),
        "japanese" => derive_japanese_era(iso),
        "buddhist" => (Some("be".to_string()), Some(year)),
        "roc" => dual("roc", "broc", year),
        "persian" => (Some("ap".to_string()), Some(year)),
        "islamic-civil" | "islamic-tbla" | "islamic-umalqura" => dual("ah", "bh", year),
        "hebrew" => (Some("am".to_string()), Some(year)),
        "indian" => (Some("shaka".to_string()), Some(year)),
        "coptic" => (Some("am".to_string()), Some(year)),
        "ethioaa" => (Some("aa".to_string()), Some(year)),
        "ethiopic" => {
            // `year` here is the Amete-Mihret year.
            if year >= 1 {
                (Some("am".to_string()), Some(year))
            } else {
                (Some("aa".to_string()), Some(year + 5500))
            }
        }
        _ => (None, None),
    }
}

/// The Japanese era for an IsoDate: a modern regnal era on/after Meiji,
/// else Gregorian `ce`/`bce`.
fn derive_japanese_era(iso: IsoDate) -> (Option<String>, Option<i64>) {
    // Meiji began 1868-10-23.
    let ymd = (
        i64::from(iso.year),
        i64::from(iso.month),
        i64::from(iso.day),
    );
    if ymd >= (1868, 10, 23) {
        let (name, ey) = japanese_era_name(iso);
        // ICU4X (and thus Temporal) only labels dates on/after Japan's Gregorian
        // adoption (1873-01-01) with the "meiji" era; the traditional Meiji reign
        // days of 1868-1872 read back as the gregorian "ce" era instead. The
        // eraYear anchor is unchanged (so "meiji" input still counts from 1868 and
        // 1873 reads back as Meiji 6).
        if name == "meiji" && ymd < (1873, 1, 1) {
            return (Some("ce".to_string()), Some(i64::from(iso.year)));
        }
        (Some(name), Some(ey))
    } else if iso.year >= 1 {
        (Some("ce".to_string()), Some(i64::from(iso.year)))
    } else {
        (Some("bce".to_string()), Some(1 - i64::from(iso.year)))
    }
}

/// Modern Japanese era name (lowercase) + era-year for an on/after-Meiji date.
fn japanese_era_name(iso: IsoDate) -> (String, i64) {
    #[cfg(feature = "intl")]
    {
        let (name, ey) = intl::calendar::japanese_era(
            i64::from(iso.year),
            i64::from(iso.month),
            i64::from(iso.day),
        );
        (name.to_ascii_lowercase(), ey)
    }
    #[cfg(not(feature = "intl"))]
    {
        // Inline modern-era table for the no-intl build.
        const ERAS: [(i64, i64, i64, &str); 5] = [
            (1868, 10, 23, "meiji"),
            (1912, 7, 30, "taisho"),
            (1926, 12, 25, "showa"),
            (1989, 1, 8, "heisei"),
            (2019, 5, 1, "reiwa"),
        ];
        let (y, m, d) = (
            i64::from(iso.year),
            i64::from(iso.month),
            i64::from(iso.day),
        );
        for &(sy, sm, sd, name) in ERAS.iter().rev() {
            if (y, m, d) >= (sy, sm, sd) {
                return (name.to_string(), y - sy + 1);
            }
        }
        ("ce".to_string(), y)
    }
}

/// Resolves an era + eraYear to the calendar's arithmetic year. Returns `None`
/// if the era code is not valid for this calendar (caller throws RangeError).
fn era_to_year(cal: &str, era: &str, era_year: i64) -> Option<i64> {
    // Canonicalize a handful of documented era aliases.
    let era = match era {
        "ad" => "ce",
        "bc" => "bce",
        other => other,
    };
    let dual = |pos: &str, neg: &str| {
        if era == pos {
            Some(era_year)
        } else if era == neg {
            Some(1 - era_year)
        } else {
            None
        }
    };
    match cal {
        "gregory" => dual("ce", "bce"),
        "japanese" => match era {
            "ce" => Some(era_year),
            "bce" => Some(1 - era_year),
            "meiji" => Some(1867 + era_year),
            "taisho" => Some(1911 + era_year),
            "showa" => Some(1925 + era_year),
            "heisei" => Some(1988 + era_year),
            "reiwa" => Some(2018 + era_year),
            _ => None,
        },
        "buddhist" => (era == "be").then_some(era_year),
        "roc" => dual("roc", "broc"),
        "persian" => (era == "ap").then_some(era_year),
        "islamic-civil" | "islamic-tbla" | "islamic-umalqura" => dual("ah", "bh"),
        "hebrew" => (era == "am").then_some(era_year),
        "indian" => (era == "shaka").then_some(era_year),
        "coptic" => (era == "am").then_some(era_year),
        "ethioaa" => (era == "aa").then_some(era_year),
        "ethiopic" => match era {
            "am" => Some(era_year),
            "aa" => Some(era_year - 5500),
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Public API: field extraction
// ---------------------------------------------------------------------------

/// Extracts a calendar's `era`/`eraYear`/`year`/`month`/`monthCode`/`day` from an
/// [`IsoDate`]. For `"iso8601"` this returns the ISO fields verbatim (era-less).
#[must_use]
pub(crate) fn iso_to_fields(cal: &str, iso: IsoDate) -> CalFields {
    let parts = parts_from_iso(cal, iso);
    let (era, era_year) = derive_era(cal, parts.year, iso);
    CalFields {
        era,
        era_year,
        year: parts.year,
        month: parts.month,
        month_code: parts.code,
        day: parts.day,
    }
}

// ---------------------------------------------------------------------------
// Public API: CalendarDateFromFields
// ---------------------------------------------------------------------------

/// Parses a month code (`"M05"` / `"M05L"`) into `(number, is_leap)`.
fn parse_month_code(code: &str) -> Option<(i64, bool)> {
    let b = code.as_bytes();
    let leap = b.len() == 4 && b[3] == b'L';
    if !((b.len() == 3 || leap) && b[0] == b'M' && b[1].is_ascii_digit() && b[2].is_ascii_digit()) {
        return None;
    }
    let n = i64::from(b[1] - b'0') * 10 + i64::from(b[2] - b'0');
    Some((n, leap))
}

/// The 1-based ordinal position of an exact `month_code` in `cal`'s `year`, or
/// `None` if that code does not occur in the year (e.g. a leap month absent that
/// year). This is the exact map — no constraining.
fn month_code_to_ordinal(cal: &str, year: i64, code: &str) -> Option<i64> {
    match cal {
        "hebrew" => hebrew_month_list(year)
            .iter()
            .position(|(_, c)| c == code)
            .map(|i| i as i64 + 1),
        "chinese" | "dangi" => chinese_month_list(cal, year)
            .iter()
            .position(|(_, _, c)| c == code)
            .map(|i| i as i64 + 1),
        _ => {
            // Non-leap-month calendars: the ordinal is the code number, and a
            // leap code is never valid.
            let (num, leap) = parse_month_code(code)?;
            if leap || num < 1 || num > months_in_year_by(cal, year) {
                None
            } else {
                Some(num)
            }
        }
    }
}

/// The `overflow: "constrain"` target ordinal for a leap `month_code` that does
/// **not** occur in `cal`'s `year`. Per the Temporal spec the leap month
/// collapses onto the named month it augments:
/// * Hebrew Adar I (`M05L`) → Adar (`M06`) in a common year;
/// * Chinese/Dangi `M<NN>L` → its base month `M<NN>` in a year lacking that leap.
///
/// Returns `None` for a non-leap code or a calendar without leap months (those
/// codes are simply out of range, not constrainable).
fn month_code_constrain_ordinal(cal: &str, year: i64, code: &str) -> Option<i64> {
    let base = constrain_leap_base_code(cal, code)?;
    month_code_to_ordinal(cal, year, &base)
}

/// The regular (non-leap) month code that a leap `code` collapses onto under
/// `overflow: "constrain"`, or `None` if `code` is not a constrainable leap code
/// for `cal`. Hebrew Adar I (`M05L`) → Adar (`M06`); a Chinese/Dangi intercalary
/// `M<NN>L` → its base `M<NN>` (only for an `NN` that is a real intercalary month
/// in the supported range). Exposed so `PlainMonthDay`'s reference-year search can
/// fall back to the augmented month when a day does not fit the leap month.
#[must_use]
pub(crate) fn constrain_leap_base_code(cal: &str, code: &str) -> Option<String> {
    let (num, leap) = parse_month_code(code)?;
    if !leap {
        return None;
    }
    match cal {
        "hebrew" => (num == 5).then(|| "M06".to_string()),
        "chinese" | "dangi" => chinese_leap_num_occurs(cal, num).then(|| format!("M{num:02}")),
        _ => None,
    }
}

/// Resolves the ordinal month for `year` from an optional ordinal `month` and/or
/// a `month_code`, honoring `overflow` for a code/leap that does not occur.
fn resolve_ordinal(
    cal: &str,
    year: i64,
    month: Option<i64>,
    month_code: Option<&str>,
    overflow: Overflow,
) -> Result<i64, CalError> {
    let n_months = months_in_year_by(cal, year);
    if let Some(code) = month_code {
        // Validate the code shape up front (rejects malformed codes regardless of
        // whether the named month happens to occur this year).
        parse_month_code(code)
            .ok_or_else(|| CalError::Range(format!("invalid monthCode '{code}'")))?;
        let ord = match month_code_to_ordinal(cal, year, code) {
            Some(o) => o,
            None => {
                // The code does not occur this year — only a leap month can be
                // absent. Constrain collapses it onto its named month; reject
                // throws.
                let not_found = || {
                    CalError::Range(format!(
                        "monthCode '{code}' does not occur in {cal} year {year}"
                    ))
                };
                match overflow {
                    Overflow::Constrain => {
                        month_code_constrain_ordinal(cal, year, code).ok_or_else(not_found)?
                    }
                    Overflow::Reject => return Err(not_found()),
                }
            }
        };
        if let Some(m) = month
            && m != ord
        {
            return Err(CalError::Range("month and monthCode disagree".to_string()));
        }
        Ok(ord)
    } else if let Some(m) = month {
        if m < 1 {
            return Err(CalError::Range("month must be positive".to_string()));
        }
        Ok(match overflow {
            Overflow::Constrain => m.min(n_months),
            Overflow::Reject => {
                if m > n_months {
                    return Err(CalError::Range(format!(
                        "month {m} is out of range for {cal} year {year}"
                    )));
                }
                m
            }
        })
    } else {
        Err(CalError::MissingFields(
            "month or monthCode is required".to_string(),
        ))
    }
}

/// `CalendarDateFromFields`: resolves the year (era+eraYear or year), the month
/// (ordinal or code), and the day into an [`IsoDate`], honoring `overflow`.
///
/// The returned date's *era* is not carried — the engine stores only the
/// [`IsoDate`], and [`iso_to_fields`] recomputes the era on read (so regnal-era
/// remapping such as "Reiwa 1 January → Heisei 31" falls out naturally).
pub(crate) fn fields_to_iso(
    cal: &str,
    input: &FieldsInput,
    overflow: Overflow,
) -> Result<IsoDate, CalError> {
    // Era / eraYear validation for calendars that use eras. These checks run even
    // when a plain `year` is also present (an `era` still has to be a real era for
    // the calendar, and `era`/`eraYear` must always come as a pair).
    if has_eras(cal) {
        match (input.era.as_deref(), input.era_year) {
            // A complete pair: the era code must be valid for this calendar. The
            // `eraYear` itself is left unchecked here so that out-of-bounds regnal
            // values remap leniently on read (e.g. Reiwa 1 January → Heisei 31).
            (Some(era), Some(ey)) => {
                if era_to_year(cal, era, ey).is_none() {
                    return Err(CalError::Range(format!(
                        "{era} is not a valid era in calendar {cal}"
                    )));
                }
            }
            (None, None) => {}
            // Exactly one of `era` / `eraYear` supplied → a TypeError per
            // CalendarResolveFields (they are mutually required).
            _ => {
                return Err(CalError::MissingFields(
                    "era and eraYear must be provided together".to_string(),
                ));
            }
        }
    }

    // Resolve the arithmetic year.
    let year = if let Some(y) = input.year {
        y
    } else if has_eras(cal) {
        match (input.era.as_deref(), input.era_year) {
            (Some(era), Some(ey)) => era_to_year(cal, era, ey)
                .ok_or_else(|| CalError::Range(format!("invalid era '{era}' for {cal}")))?,
            _ => {
                return Err(CalError::MissingFields(
                    "year, or era and eraYear, are required".to_string(),
                ));
            }
        }
    } else {
        return Err(CalError::MissingFields("year is required".to_string()));
    };

    let ord = resolve_ordinal(
        cal,
        year,
        input.month,
        input.month_code.as_deref(),
        overflow,
    )?;
    let day = input.day;
    if day < 1 {
        return Err(CalError::Range("day must be positive".to_string()));
    }
    // Constrain / reject the day against the resolved month's length.
    let dim = days_in_month_by(cal, year, ord);
    let day = match overflow {
        Overflow::Constrain => day.min(dim.max(1)),
        Overflow::Reject => {
            if day > dim {
                return Err(CalError::Range(format!(
                    "day {day} is out of range for {cal} {year}-M{ord:02}"
                )));
            }
            day
        }
    };
    iso_from_parts(cal, year, ord, day).ok_or_else(|| {
        CalError::Range(format!(
            "{cal} date {year}-{ord}-{day} is not representable"
        ))
    })
}

// ---------------------------------------------------------------------------
// Public API: derived accessors
// ---------------------------------------------------------------------------

/// JDN of day 1 of ordinal `month` in `year` (for month-length arithmetic).
fn month_start_jdn(cal: &str, year: i64, month: i64) -> Option<i64> {
    let n = months_in_year_by(cal, year);
    if month <= n {
        iso_from_parts(cal, year, month, 1).map(iso_to_jdn)
    } else {
        // First day of the next year.
        iso_from_parts(cal, year + 1, 1, 1).map(iso_to_jdn)
    }
}

/// Number of ordinal months in `year` for `cal`.
fn months_in_year_by(cal: &str, year: i64) -> i64 {
    match cal {
        "coptic" | "ethiopic" | "ethioaa" => 13,
        "hebrew" if hebrew_leap(year) => 13,
        "chinese" | "dangi" if chinese_leap_month(cal, year) != 0 => 13,
        _ => 12,
    }
}

/// Days in ordinal `month` of `year` for `cal`.
fn days_in_month_by(cal: &str, year: i64, month: i64) -> i64 {
    match (
        month_start_jdn(cal, year, month),
        month_start_jdn(cal, year, month + 1),
    ) {
        (Some(a), Some(b)) => (b - a).max(1),
        _ => 30,
    }
}

/// `monthsInYear` for the calendar year containing `iso`.
#[must_use]
pub(crate) fn months_in_year(cal: &str, iso: IsoDate) -> i64 {
    let p = parts_from_iso(cal, iso);
    months_in_year_by(cal, p.year)
}

/// `daysInMonth` for the calendar month containing `iso`.
#[must_use]
pub(crate) fn days_in_month(cal: &str, iso: IsoDate) -> i64 {
    let p = parts_from_iso(cal, iso);
    days_in_month_by(cal, p.year, p.month)
}

/// `daysInYear` for the calendar year containing `iso`.
#[must_use]
pub(crate) fn days_in_year(cal: &str, iso: IsoDate) -> i64 {
    let p = parts_from_iso(cal, iso);
    match (
        iso_from_parts(cal, p.year, 1, 1).map(iso_to_jdn),
        iso_from_parts(cal, p.year + 1, 1, 1).map(iso_to_jdn),
    ) {
        (Some(a), Some(b)) => (b - a).max(1),
        _ => 365,
    }
}

/// `inLeapYear` for the calendar year containing `iso`.
#[must_use]
pub(crate) fn in_leap_year(cal: &str, iso: IsoDate) -> bool {
    let p = parts_from_iso(cal, iso);
    match cal {
        "gregory" | "japanese" => greg_leap(i64::from(iso.year)),
        "buddhist" => greg_leap(i64::from(iso.year)),
        "roc" => greg_leap(i64::from(iso.year)),
        "coptic" | "ethiopic" => coptic_like_leap(p.year),
        "ethioaa" => coptic_like_leap(p.year - 5500),
        "indian" => indian_leap(p.year),
        "hebrew" => hebrew_leap(p.year),
        "chinese" | "dangi" => chinese_leap_month(cal, p.year) != 0,
        // persian / islamic: a leap year simply has more days.
        _ => days_in_year(cal, iso) > days_in_year_min(cal),
    }
}

/// The non-leap day count for the length-based `inLeapYear` fallback.
fn days_in_year_min(cal: &str) -> i64 {
    match cal {
        "islamic-civil" | "islamic-tbla" | "islamic-umalqura" => 354,
        "persian" => 365,
        _ => 365,
    }
}

/// `dayOfWeek` (calendar-independent): 1 = Monday … 7 = Sunday.
#[must_use]
pub(crate) fn day_of_week(iso: IsoDate) -> i64 {
    iso_to_jdn(iso).rem_euclid(7) + 1
}

/// `daysInWeek` — always 7.
#[must_use]
pub(crate) fn days_in_week() -> i64 {
    7
}

/// `dayOfYear` within the calendar year.
#[must_use]
pub(crate) fn day_of_year(cal: &str, iso: IsoDate) -> i64 {
    let p = parts_from_iso(cal, iso);
    match iso_from_parts(cal, p.year, 1, 1).map(iso_to_jdn) {
        Some(start) => iso_to_jdn(iso) - start + 1,
        None => 1,
    }
}

/// ISO-8601-style `(weekOfYear, yearOfWeek)`. Non-ISO calendars use the same
/// week rule as ISO (weeks belong to the year with their Thursday); a calendar
/// whose week is not well-defined returns `(0-ish)` — callers may return
/// `undefined` for such calendars.
#[must_use]
pub(crate) fn week_of_year(cal: &str, iso: IsoDate) -> Option<(i64, i64)> {
    // Per the Temporal spec, only the ISO-8601 calendar has a well-defined week
    // numbering; every non-ISO calendar reports `undefined` for
    // weekOfYear/yearOfWeek.
    if cal != "iso8601" {
        return None;
    }
    let jdn = iso_to_jdn(iso);
    let weekday = jdn.rem_euclid(7) + 1; // 1..7
    let thursday = jdn - (weekday - 4);
    let (iso_year, _, _) = jdn_to_greg(thursday);
    let jan4 = greg_to_jdn(iso_year, 1, 4);
    let jan4_weekday = jan4.rem_euclid(7) + 1;
    let week1_monday = jan4 - (jan4_weekday - 1);
    let week = (jdn - week1_monday) / 7 + 1;
    Some((week, iso_year))
}

/// `yearOfWeek` companion to [`week_of_year`].
#[must_use]
pub(crate) fn year_of_week(cal: &str, iso: IsoDate) -> Option<i64> {
    week_of_year(cal, iso).map(|(_, y)| y)
}

// ---------------------------------------------------------------------------
// Public API: calendar-aware date arithmetic (CalendarDateAdd / CalendarDateUntil)
// ---------------------------------------------------------------------------

/// The four date-portion components of a calendar difference. The caller balances
/// these (with any time components) into a `Temporal.Duration`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DateDurationParts {
    pub years: i64,
    pub months: i64,
    pub weeks: i64,
    pub days: i64,
}

/// A safe bound on a calendar arithmetic year, well outside the representable ISO
/// range (±273790 years), so the intermediate field→ISO conversion never wraps
/// its `i32` year. The caller's `ISODateWithinLimits` check is the precise gate.
const CAL_YEAR_LIMIT: i64 = 500_000;

/// Adds `months` ordinal months to `(year, ord_month)`, walking calendar years —
/// whose month counts (and leap months) vary — so the ordinal month always stays
/// in `1..=months_in_year`. Returns the new `(year, ord_month)`.
fn add_ordinal_months(cal: &str, mut year: i64, mut ord_month: i64, months: i64) -> (i64, i64) {
    if months >= 0 {
        let mut rem = months;
        while rem > 0 {
            let n = months_in_year_by(cal, year);
            let room = n - ord_month; // months addable while staying in this year
            if rem <= room {
                ord_month += rem;
                rem = 0;
            } else {
                rem -= room + 1; // step to month 1 of the next year
                year += 1;
                ord_month = 1;
            }
        }
    } else {
        let mut rem = -months;
        while rem > 0 {
            let room = ord_month - 1; // months subtractable while staying in this year
            if rem <= room {
                ord_month -= rem;
                rem = 0;
            } else {
                rem -= ord_month; // step to the last month of the previous year
                year -= 1;
                ord_month = months_in_year_by(cal, year);
            }
        }
    }
    (year, ord_month)
}

/// `CalendarDateAdd`: adds a date duration to `iso` in `cal`'s own calendar
/// (variable month lengths + leap months honoured), returning the ISO date.
///
/// Order (per the Temporal spec): add `years` to the calendar year and `months`
/// by walking the calendar, constrain the day to the target month's length
/// (`overflow`), convert back to ISO, then apply `weeks*7 + days` as a plain,
/// calendar-independent day offset.
pub(crate) fn calendar_date_add(
    cal: &str,
    iso: IsoDate,
    years: i64,
    months: i64,
    weeks: i64,
    days: i64,
    overflow: Overflow,
) -> Result<IsoDate, CalError> {
    let f = iso_to_fields(cal, iso);
    // 1. Add years to the calendar year. A calendar year's ordinal month
    //    *positions* shift as leap months appear or vanish, so the anchor month
    //    must be re-resolved from its **monthCode** (the stable named month) in
    //    the post-addition year — not carried as a raw ordinal. A leap month that
    //    is absent in the new year constrains onto its named month (Adar I → Adar,
    //    `M<NN>L` → `M<NN>`) or, under `reject`, throws.
    let year_after = f.year + years;
    let start_ord = resolve_ordinal(cal, year_after, None, Some(&f.month_code), overflow)?;
    // 2. Walk the ordinal months (a leap month is one ordinal step like any
    //    other), rolling across calendar years whose month counts vary.
    let (year, ord_month) = add_ordinal_months(cal, year_after, start_ord, months);
    if !(-CAL_YEAR_LIMIT..=CAL_YEAR_LIMIT).contains(&year) {
        return Err(CalError::Range(format!(
            "{cal} year {year} is out of the representable range"
        )));
    }
    // 3. Constrain (or reject) the day against the target month's length.
    let dim = days_in_month_by(cal, year, ord_month);
    let day = match overflow {
        Overflow::Constrain => f.day.min(dim.max(1)),
        Overflow::Reject => {
            if f.day > dim {
                return Err(CalError::Range(format!(
                    "day {} is out of range for {cal} {year}-M{ord_month:02}",
                    f.day
                )));
            }
            f.day
        }
    };
    // 4. Field → ISO, then apply the plain week/day offset in JDN space.
    let base = iso_from_parts(cal, year, ord_month, day).ok_or_else(|| {
        CalError::Range(format!(
            "{cal} date {year}-{ord_month}-{day} is not representable"
        ))
    })?;
    let jdn = iso_to_jdn(base) + weeks * 7 + days;
    // Bound the JDN so the final i32-year conversion cannot wrap (the caller's
    // ISODateWithinLimits check is the precise range gate).
    let epoch_days = jdn - greg_to_jdn(1970, 1, 1);
    if !(MIN_EPOCH_DAYS - 2..=MAX_EPOCH_DAYS + 2).contains(&epoch_days) {
        return Err(CalError::Range(
            "result is outside the representable range".to_string(),
        ));
    }
    Ok(jdn_to_iso(jdn))
}

/// The signed number of ordinal-month steps from `(y_from, o_from)` to
/// `(y_to, o_to)`, summing each intervening calendar year's variable month count
/// (which differs across leap years). Positive when the target is later. Used to
/// fold a whole-years + whole-months difference into a single month total for
/// `largestUnit: "month"`, where the month count of each year span depends on
/// exactly which months the leap month falls between — so it cannot be derived
/// from the year count alone.
fn ordinal_month_distance(cal: &str, y_from: i64, o_from: i64, y_to: i64, o_to: i64) -> i64 {
    if (y_from, o_from) == (y_to, o_to) {
        return 0;
    }
    let forward = (y_to, o_to) > (y_from, o_from);
    let (ya, oa, yb, ob) = if forward {
        (y_from, o_from, y_to, o_to)
    } else {
        (y_to, o_to, y_from, o_from)
    };
    // Steps from (ya, oa) to (yb, ob): fill out year ya from oa to its last
    // month, cross into each subsequent year, and stop at ob. Telescopes to
    // Σ months_in_year(ya..yb) − oa + ob.
    let mut sum = 0;
    for y in ya..yb {
        sum += months_in_year_by(cal, y);
    }
    sum = sum - oa + ob;
    if forward { sum } else { -sum }
}

/// `CalendarDateUntil`: the calendar-aware difference from `iso1` to `iso2`, in
/// components down to `largest_unit` (Year / Month / Week / Day). Days (and, for
/// `Week`, weeks) hold the calendar-independent remainder; the caller balances the
/// result into a `Temporal.Duration`. Signs follow `iso1 → iso2`.
pub(crate) fn calendar_date_until(
    cal: &str,
    iso1: IsoDate,
    iso2: IsoDate,
    largest_unit: Unit,
) -> DateDurationParts {
    // Day / Week: a pure ISO-day difference (calendar-independent).
    if matches!(largest_unit, Unit::Week | Unit::Day) {
        let mut days = iso_to_jdn(iso2) - iso_to_jdn(iso1);
        let mut weeks = 0;
        if largest_unit == Unit::Week {
            weeks = days / 7;
            days %= 7;
        }
        return DateDurationParts {
            weeks,
            days,
            ..Default::default()
        };
    }
    // Year / Month: count whole calendar years, then whole calendar months, then
    // the leftover days (mirroring DifferenceISODate but calendar-aware). The
    // reference algorithm adds candidate years/months and backs off if it passes
    // the target.
    let jdn1 = iso_to_jdn(iso1);
    let jdn2 = iso_to_jdn(iso2);
    if jdn1 == jdn2 {
        return DateDurationParts::default();
    }
    let sign = if jdn2 > jdn1 { 1 } else { -1 };
    let f1 = iso_to_fields(cal, iso1);
    let f2 = iso_to_fields(cal, iso2);

    let f1_day = f1.day;
    // Adds to `iso1` with Constrain; on any failure falls back to the anchor
    // (mirroring the ISO reference's `.unwrap_or(from)`).
    let add = |y: i64, m: i64| -> IsoDate {
        calendar_date_add(cal, iso1, y, m, 0, 0, Overflow::Constrain).unwrap_or(iso1)
    };
    // Whether a candidate `(years, months)` step lands strictly beyond `iso2`.
    //
    // The overshoot is measured with the *ideal* (unconstrained) day-of-month —
    // the anchor's own day, even when that day does not exist in the target month.
    // Comparing the constrained landing date instead would hide a real overshoot:
    // e.g. Jan 29 + 1 month lands on the non-existent Feb 29, which constrains down
    // to Feb 28 and would look like an exact one-month hit, when Temporal treats it
    // as 30 days (not a whole month). Because a calendar's `(year, ordinalMonth,
    // day)` triple is monotonic in real time, a lexicographic compare is valid even
    // for an out-of-range `day`.
    let passes = |y: i64, m: i64| -> bool {
        let tf = iso_to_fields(cal, add(y, m));
        let a = (tf.year, tf.month, f1_day);
        let b = (f2.year, f2.month, f2.day);
        if sign > 0 { a > b } else { a < b }
    };

    // Whole years — via the Temporal reference `untilCalendar` rule: two dates are
    // a whole year apart only when the *monthCode* ordering agrees with the year
    // ordering. Comparing monthCodes lexicographically (so a leap "M04L" sorts
    // after "M04" and before "M05", and all base codes are zero-padded to two
    // digits) is exactly what makes a leap-vs-nonleap boundary count as months and
    // not a year: adding a year to a leap-month date constrains onto a *different*
    // monthCode, which must not be counted as a full year.
    let diff_days = f2.day - f1.day;
    let diff_in_year_sign = if f2.month_code > f1.month_code {
        1
    } else if f2.month_code < f1.month_code {
        -1
    } else {
        diff_days.signum()
    };
    // When `iso1`'s month-day sits further into the year than `iso2`'s (relative to
    // the travel direction), the raw year subtraction overshoots by one.
    let mut years = if diff_in_year_sign * sign < 0 {
        (f2.year - f1.year) - sign
    } else {
        f2.year - f1.year
    };
    // Correct any residual leap-year overshoot left by the constrained year add.
    if passes(years, 0) {
        years -= sign;
    }

    // Whole months within the final year span (bounded by months_in_year).
    let mut months = 0;
    while !passes(years, months + sign) {
        months += sign;
    }

    let mid = add(years, months);
    let days = iso_to_jdn(iso2) - iso_to_jdn(mid);

    if largest_unit == Unit::Month {
        // Fold the whole-years + whole-months span into one month total by
        // counting the ordinal-month steps from the start to `mid` directly. This
        // respects leap years (13 months) and the fact that a same-monthCode year
        // step spans 12 or 13 ordinal months depending on where the leap month
        // sits relative to the month — which a per-year-count sum cannot capture.
        let mid_f = iso_to_fields(cal, mid);
        let total_months = ordinal_month_distance(cal, f1.year, f1.month, mid_f.year, mid_f.month);
        return DateDurationParts {
            months: total_months,
            days,
            ..Default::default()
        };
    }
    DateDurationParts {
        years,
        months,
        days,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iso(y: i32, m: u8, d: u8) -> IsoDate {
        IsoDate {
            year: y,
            month: m,
            day: d,
        }
    }

    #[test]
    fn canonicalize() {
        assert_eq!(canonicalize_calendar("ISO8601"), Some("iso8601"));
        assert_eq!(canonicalize_calendar("islamicc"), Some("islamic-civil"));
        assert_eq!(canonicalize_calendar("gregorian"), Some("gregory"));
        assert_eq!(canonicalize_calendar("minguo"), Some("roc"));
        assert_eq!(
            canonicalize_calendar("ethiopic-amete-alem"),
            Some("ethioaa")
        );
        // Not supported in Temporal:
        assert_eq!(canonicalize_calendar("islamic"), None);
        assert_eq!(canonicalize_calendar("islamic-rgsa"), None);
        assert_eq!(canonicalize_calendar("bogus"), None);
    }

    #[test]
    fn gregory_family() {
        let f = iso_to_fields("gregory", iso(2000, 3, 6));
        assert_eq!(f.era.as_deref(), Some("ce"));
        assert_eq!(f.era_year, Some(2000));
        assert_eq!(f.year, 2000);
        let f = iso_to_fields("buddhist", iso(2000, 1, 1));
        assert_eq!(f.year, 2543);
        assert_eq!(f.era.as_deref(), Some("be"));
        let f = iso_to_fields("roc", iso(2025, 1, 1));
        assert_eq!(f.year, 114);
        assert_eq!(f.era.as_deref(), Some("roc"));
    }

    #[cfg(feature = "intl")]
    #[test]
    fn spot_check_known_dates() {
        // 2000-01-01 anchors (see task brief).
        let h = iso_to_fields("hebrew", iso(2000, 1, 1));
        assert_eq!(h.year, 5760);
        assert_eq!(h.day, 23);
        // Hebrew month of 2000-01-01 is Tevet (civil ordinal 4 = M04).
        assert_eq!(h.month_code, "M04");

        let p = iso_to_fields("persian", iso(2000, 1, 1));
        assert_eq!(p.year, 1378);
        assert_eq!(p.month, 10);
        assert_eq!(p.day, 11);

        // Islamic variants (Test262 pins 2000-01-01 = 1420-09-24 civil / -25 tbla).
        let ic = iso_to_fields("islamic-civil", iso(2000, 1, 1));
        assert_eq!((ic.year, ic.month, ic.day), (1420, 9, 24));
        let it = iso_to_fields("islamic-tbla", iso(2000, 1, 1));
        assert_eq!((it.year, it.month, it.day), (1420, 9, 25));
        let iu = iso_to_fields("islamic-umalqura", iso(2000, 1, 1));
        assert_eq!((iu.year, iu.month, iu.day), (1420, 9, 24));

        // Japanese Heisei 12.
        let j = iso_to_fields("japanese", iso(2000, 1, 1));
        assert_eq!(j.era.as_deref(), Some("heisei"));
        assert_eq!(j.era_year, Some(12));
        assert_eq!(j.year, 2000);
    }

    #[test]
    fn roundtrip_iso_all() {
        // Every calendar must round-trip an ordinary date through fields.
        let cals = [
            "gregory", "buddhist", "roc", "japanese", "coptic", "ethiopic", "ethioaa", "indian",
        ];
        for cal in cals {
            let d = iso(2023, 6, 15);
            let f = iso_to_fields(cal, d);
            let input = FieldsInput {
                era: f.era.clone(),
                era_year: f.era_year,
                year: Some(f.year),
                month: Some(f.month),
                month_code: None,
                day: f.day,
            };
            let back = fields_to_iso(cal, &input, Overflow::Reject)
                .unwrap_or_else(|_| panic!("{cal} roundtrip failed"));
            assert_eq!(back, d, "calendar {cal} round-trip");
        }
    }

    #[test]
    fn arith_gregory_matches_iso() {
        use crate::temporal_iso::{Unit, add_iso_date, difference_iso_date};
        // The Gregorian calendar's arithmetic year == the ISO year, so
        // calendar-aware add/until must agree with the ISO reference.
        let cases = [
            (iso(2020, 1, 31), 0_i64, 1_i64, 0_i64, 0_i64), // Jan 31 + 1mo -> Feb 29 (constrain)
            (iso(2021, 1, 31), 0, 1, 0, 0),                 // -> Feb 28
            (iso(2020, 2, 29), 1, 0, 0, 0),                 // leap day + 1yr -> Feb 28
            (iso(2023, 6, 15), 2, 5, 1, 10),
            (iso(2023, 12, 15), 0, 0, 0, 40),
            (iso(2023, 3, 15), -1, -4, -2, -5),
        ];
        for (d, y, m, w, days) in cases {
            let got = calendar_date_add("gregory", d, y, m, w, days, Overflow::Constrain).unwrap();
            let want = add_iso_date(d, y, m, w, days, Overflow::Constrain).unwrap();
            assert_eq!(got, want, "gregory add {d:?} +{y}y{m}m{w}w{days}d");
        }
        // until must agree with DifferenceISODate for every date largestUnit.
        let pairs = [
            (iso(2020, 1, 15), iso(2023, 6, 20)),
            (iso(2023, 6, 20), iso(2020, 1, 15)),
            (iso(2020, 2, 29), iso(2021, 3, 1)),
            (iso(2019, 12, 31), iso(2020, 1, 1)),
        ];
        for lu in [Unit::Year, Unit::Month, Unit::Week, Unit::Day] {
            for (a, b) in pairs {
                let got = calendar_date_until("gregory", a, b, lu);
                let (wy, wm, ww, wd) = difference_iso_date(a, b, lu);
                assert_eq!(
                    (got.years, got.months, got.weeks, got.days),
                    (wy, wm, ww, wd),
                    "gregory until {a:?}->{b:?} largest {lu:?}"
                );
            }
        }
    }

    #[test]
    fn arith_add_until_inverse() {
        use crate::temporal_iso::Unit;
        // For any calendar, until(a, a+dur) recovers the duration in the same
        // units when the addition loses nothing (no day constrain).
        let cals = ["gregory", "coptic", "ethiopic", "indian", "buddhist", "roc"];
        for cal in cals {
            let a = iso(2015, 5, 10);
            let b = calendar_date_add(cal, a, 2, 3, 0, 0, Overflow::Constrain).unwrap();
            let diff = calendar_date_until(cal, a, b, Unit::Year);
            // Re-adding the reported difference must land back on b.
            let back = calendar_date_add(
                cal,
                a,
                diff.years,
                diff.months,
                diff.weeks,
                diff.days,
                Overflow::Constrain,
            )
            .unwrap();
            assert_eq!(back, b, "{cal} add/until inverse");
        }
    }

    /// Builds an ISO date from `cal` fields `(year, monthCode, day)` with reject.
    #[cfg(feature = "intl")]
    fn from_code(cal: &str, year: i64, code: &str, day: i64) -> IsoDate {
        fields_to_iso(
            cal,
            &FieldsInput {
                year: Some(year),
                month_code: Some(code.to_string()),
                day,
                ..Default::default()
            },
            Overflow::Reject,
        )
        .unwrap_or_else(|_| panic!("{cal} {year}-{code}-{day} should be representable"))
    }

    #[cfg(feature = "intl")]
    #[test]
    fn arith_hebrew_leap_month() {
        use crate::temporal_iso::Unit;
        let code_of = |d: IsoDate| iso_to_fields("hebrew", d).month_code;
        let year_of = |d: IsoDate| iso_to_fields("hebrew", d).year;
        let add =
            |d, y, m| calendar_date_add("hebrew", d, y, m, 0, 0, Overflow::Constrain).unwrap();

        // Year addition preserves the *named* month (monthCode), NOT the raw
        // ordinal: 5782 (leap) Nisan `M07` is ordinal 8, but +1yr lands on 5783
        // (common) Nisan `M07`, which is ordinal 7.
        let nisan_leap = from_code("hebrew", 5782, "M07", 1);
        let plus1 = add(nisan_leap, 1, 0);
        assert_eq!(year_of(plus1), 5783, "hebrew +1yr advances year");
        assert_eq!(code_of(plus1), "M07", "hebrew +1yr preserves monthCode");

        // Adar I (`M05L`) of leap 5784 has no counterpart in common 5785: constrain
        // collapses onto Adar (`M06`); reject throws.
        let adar1 = from_code("hebrew", 5784, "M05L", 1);
        let c = add(adar1, 1, 0);
        assert_eq!(
            (year_of(c), code_of(c).as_str()),
            (5785, "M06"),
            "Adar I +1yr → Adar"
        );
        assert!(
            matches!(
                calendar_date_add("hebrew", adar1, 1, 0, 0, 0, Overflow::Reject),
                Err(CalError::Range(_))
            ),
            "Adar I +1yr rejects when the leap month is absent"
        );

        // The leap month is a single ordinal step in the month walk: Tevet (`M04`)
        // of leap 5784 → +2 = Adar I (`M05L`), +3 = Adar II (`M06`).
        let tevet = from_code("hebrew", 5784, "M04", 1);
        assert_eq!(code_of(add(tevet, 0, 2)), "M05L", "M04 +2mo → Adar I");
        assert_eq!(code_of(add(tevet, 0, 3)), "M06", "M04 +3mo → Adar II");

        // until across a leap boundary: Shevat→Shevat leap→common is a whole year
        // yet spans 13 ordinal months (the leap year it leaves has 13).
        let leap_shevat = from_code("hebrew", 5784, "M05", 1);
        let common2_shevat = from_code("hebrew", 5785, "M05", 1);
        let y = calendar_date_until("hebrew", leap_shevat, common2_shevat, Unit::Year);
        assert_eq!((y.years, y.months), (1, 0), "M05→M05 leap→common is 1y");
        let m = calendar_date_until("hebrew", leap_shevat, common2_shevat, Unit::Month);
        assert_eq!(m.months, 13, "M05→M05 leap→common is 13mo not 12mo");

        // Calendar-specific constraining in `until`: Adar I → next common Adar
        // (`M06`) is a full year, but Adar I → next common Shevat (`M05`) is only
        // 12 months (not a year).
        let common2_adar = from_code("hebrew", 5785, "M06", 1);
        let ya = calendar_date_until("hebrew", adar1, common2_adar, Unit::Year);
        assert_eq!((ya.years, ya.months), (1, 0), "M05L→M06 is 1y");
        let ys = calendar_date_until("hebrew", adar1, common2_shevat, Unit::Year);
        assert_eq!((ys.years, ys.months), (0, 12), "M05L→M05 is 12mo not 1y");
    }

    #[cfg(feature = "intl")]
    #[test]
    fn arith_chinese_leap_month() {
        let code_of = |d: IsoDate| iso_to_fields("chinese", d).month_code;
        let year_of = |d: IsoDate| iso_to_fields("chinese", d).year;

        // Pick a Chinese leap year from the table and its leap-month number.
        let leap_year = (2000..=2030)
            .find(|&y| chinese_leap_month("chinese", y) != 0)
            .expect("a chinese leap year exists in range");
        let lm = chinese_leap_month("chinese", leap_year);
        let base = format!("M{lm:02}");
        let leap = format!("M{lm:02}L");

        // Year addition preserves the monthCode.
        let m08 = from_code("chinese", leap_year, "M08", 1);
        let plus1 = calendar_date_add("chinese", m08, 1, 0, 0, 0, Overflow::Constrain).unwrap();
        assert_eq!(year_of(plus1), leap_year + 1, "chinese +1yr advances year");
        assert_eq!(code_of(plus1), "M08", "chinese +1yr preserves monthCode");

        // The intercalary month is one ordinal step: `M<lm>` + 1 month → `M<lm>L`.
        let base_iso = from_code("chinese", leap_year, &base, 1);
        let stepped =
            calendar_date_add("chinese", base_iso, 0, 1, 0, 0, Overflow::Constrain).unwrap();
        assert_eq!(code_of(stepped), leap, "M{lm:02} +1mo → M{lm:02}L");

        // Resolving the leap code in a year that lacks it: reject throws, constrain
        // drops the leap marker onto the base month.
        let common_year = (leap_year + 1..=2035)
            .find(|&y| chinese_leap_month("chinese", y) == 0)
            .expect("a chinese common year exists in range");
        let fi = |ov| {
            fields_to_iso(
                "chinese",
                &FieldsInput {
                    year: Some(common_year),
                    month_code: Some(leap.clone()),
                    day: 1,
                    ..Default::default()
                },
                ov,
            )
        };
        assert!(
            matches!(fi(Overflow::Reject), Err(CalError::Range(_))),
            "leap code rejects in common year"
        );
        let constrained = fi(Overflow::Constrain).expect("leap code constrains in common year");
        assert_eq!(
            code_of(constrained),
            base,
            "leap code constrains onto base month"
        );
    }

    #[cfg(feature = "intl")]
    #[test]
    fn dangi_and_umalqura_use_dedicated_tables() {
        // `islamic-umalqura` routes to the real Umm al-Qura table, which diverges
        // from the civil tabular calendar: 2016-06-06 is 1 Ramadan 1437 in
        // umalqura but 29 Sha'ban 1437 in the civil calendar.
        let u = iso_to_fields("islamic-umalqura", iso(2016, 6, 6));
        assert_eq!((u.year, u.month, u.day), (1437, 9, 1));
        let c = iso_to_fields("islamic-civil", iso(2016, 6, 6));
        assert_eq!((c.year, c.month, c.day), (1437, 8, 29));

        // `dangi` routes to the Korean-meridian table, whose leap months differ
        // from the Chinese calendar's in some years (e.g. 2017: Chinese M06L vs
        // Dangi M05L).
        assert_eq!(chinese_leap_month("chinese", 2017), 6);
        assert_eq!(chinese_leap_month("dangi", 2017), 5);

        // Both crate-backed calendars round-trip an ordinary date through fields.
        for cal in ["islamic-umalqura", "dangi"] {
            let d = iso(2023, 6, 15);
            let f = iso_to_fields(cal, d);
            let input = FieldsInput {
                era: f.era.clone(),
                era_year: f.era_year,
                year: Some(f.year),
                month: Some(f.month),
                month_code: None,
                day: f.day,
            };
            let back = fields_to_iso(cal, &input, Overflow::Reject)
                .unwrap_or_else(|_| panic!("{cal} roundtrip failed"));
            assert_eq!(back, d, "calendar {cal} round-trip");
        }
    }

    #[cfg(feature = "intl")]
    #[test]
    fn leap_code_validity() {
        let fi = |cal: &str, y: i64, code: &str, ov| {
            fields_to_iso(
                cal,
                &FieldsInput {
                    year: Some(y),
                    month_code: Some(code.to_string()),
                    day: 1,
                    ..Default::default()
                },
                ov,
            )
        };
        // Hebrew: `M05L` is the only leap code. It exists in leap 5784, is absent
        // (rejects / constrains to Adar) in common 5783, and any other `M<NN>L` or
        // an out-of-range `M13` is invalid even under constrain.
        assert!(fi("hebrew", 5784, "M05L", Overflow::Reject).is_ok());
        assert!(fi("hebrew", 5783, "M05L", Overflow::Reject).is_err());
        assert!(fi("hebrew", 5783, "M05L", Overflow::Constrain).is_ok());
        assert!(fi("hebrew", 5784, "M02L", Overflow::Constrain).is_err());
        assert!(fi("hebrew", 5779, "M13", Overflow::Constrain).is_err());
        // Chinese: `M01L` and `M12L` never carry an intercalary month → invalid
        // even under constrain.
        assert!(!chinese_leap_num_occurs("chinese", 1));
        assert!(!chinese_leap_num_occurs("chinese", 12));
        assert!(chinese_leap_num_occurs("chinese", 6));
        assert!(fi("chinese", 2001, "M12L", Overflow::Constrain).is_err());
        assert!(fi("chinese", 2001, "M13", Overflow::Constrain).is_err());
    }

    #[cfg(feature = "intl")]
    #[test]
    fn arith_islamic_constrain() {
        // Islamic-civil months alternate 30/29 days. Take the 30th of a 30-day
        // month, add one month, and confirm the day is constrained to the next
        // month's length.
        // Find an islamic date on day 30.
        let mut found = None;
        for off in 0..40 {
            let d = jdn_to_iso(iso_to_jdn(iso(2000, 1, 1)) + off);
            let f = iso_to_fields("islamic-civil", d);
            if f.day == 30 {
                found = Some((d, f));
                break;
            }
        }
        let (d, f) = found.expect("a day-30 islamic-civil date exists in range");
        let next = calendar_date_add("islamic-civil", d, 0, 1, 0, 0, Overflow::Constrain).unwrap();
        let nf = iso_to_fields("islamic-civil", next);
        // Same year (unless the month wrapped) and day <= that month's length.
        let dim = days_in_month("islamic-civil", next);
        assert!(nf.day <= dim, "islamic constrain: day {} <= {dim}", nf.day);
        assert!(nf.day <= f.day, "islamic constrain never grows the day");
        // Reject must throw when the source day exceeds the target month length.
        if dim < 30 {
            let rejected = calendar_date_add("islamic-civil", d, 0, 1, 0, 0, Overflow::Reject);
            assert!(
                matches!(rejected, Err(CalError::Range(_))),
                "islamic reject overflows"
            );
        }
    }

    #[test]
    fn coptic_epoch() {
        // Coptic 1-01-01 should be a real date; year 0 of the arithmetic scheme
        // begins in ISO 283 (per Test262 epoch-year table).
        let f = iso_to_fields("coptic", iso(284, 8, 29));
        assert_eq!(f.year, 1);
        assert_eq!(f.month, 1);
        assert_eq!(f.day, 1);
    }

    #[test]
    fn greg_jdn_roundtrip_bc() {
        // Floor-division fix: proleptic Gregorian <-> JDN must round-trip for
        // dates deep in the BC era (JDN < -32044), where the classic truncating
        // formula was off by one. Also confirm nothing changed for a modern date.
        for &j in &[
            -300000i64, -284654, -50000, -32045, -32044, 0, 2_451_545, 3_000_000,
        ] {
            let (y, m, d) = jdn_to_greg(j);
            assert_eq!(greg_to_jdn(y, m, d), j, "greg roundtrip at jdn {j}");
        }
        // The J2000 epoch (2000-01-01 12:00 UT is JDN 2451545).
        assert_eq!(jdn_to_greg(2_451_545), (2000, 1, 1));
    }

    fn from_era(cal: &str, era: &str, ey: i64, code: &str) -> Result<IsoDate, CalError> {
        fields_to_iso(
            cal,
            &FieldsInput {
                era: Some(era.to_string()),
                era_year: Some(ey),
                month_code: Some(code.to_string()),
                day: 1,
                ..Default::default()
            },
            Overflow::Reject,
        )
    }

    #[cfg(feature = "intl")]
    #[test]
    fn non_positive_single_era_year_roundtrips() {
        // Single-era calendars: eraYear equals the arithmetic year and must
        // round-trip for non-positive values (regression: ethioaa +1 and persian
        // -1 collapsed to 0 before the coptic-BC / continuous-persian fixes).
        // Gated on `intl` because hebrew needs the crate's month data.
        for (cal, era) in [
            ("ethioaa", "aa"),
            ("coptic", "am"),
            ("buddhist", "be"),
            ("hebrew", "am"),
            ("indian", "shaka"),
            ("persian", "ap"),
        ] {
            for ey in [-1i64, 0, 1] {
                let d = from_era(cal, era, ey, "M01").expect("resolves");
                let f = iso_to_fields(cal, d);
                assert_eq!(f.era.as_deref(), Some(era), "{cal} era");
                assert_eq!(f.era_year, Some(ey), "{cal} eraYear {ey} round-trips");
                assert_eq!(f.year, ey, "{cal} year == eraYear for {ey}");
            }
        }
    }

    #[cfg(feature = "intl")]
    #[test]
    fn persian_year_zero_exists() {
        // Temporal's proleptic Persian has a real year 0 distinct from year -1
        // (the intl crate's historical variant collapses them).
        let jm1 = iso_to_jdn(from_era("persian", "ap", -1, "M01").unwrap());
        let j0 = iso_to_jdn(from_era("persian", "ap", 0, "M01").unwrap());
        let j1 = iso_to_jdn(from_era("persian", "ap", 1, "M01").unwrap());
        assert!(jm1 < j0 && j0 < j1, "persian years -1 < 0 < 1 are ordered");
        // Each proleptic year is a full 365- or 366-day solar year (no collapse).
        assert!((365..=366).contains(&(j1 - j0)), "persian year 0 length");
        assert!((365..=366).contains(&(j0 - jm1)), "persian year -1 length");
    }

    #[cfg(feature = "intl")]
    #[test]
    fn japanese_meiji_era_label_boundary() {
        // ICU / Temporal only label dates on/after the 1873-01-01 Gregorian
        // adoption as "meiji"; the traditional Meiji days of 1868-1872 read back
        // as the gregorian "ce" era, while 1873 reads back as Meiji 6.
        let f = iso_to_fields("japanese", iso(1868, 10, 23));
        assert_eq!(f.era.as_deref(), Some("ce"));
        assert_eq!(f.era_year, Some(1868));
        let f = iso_to_fields("japanese", iso(1872, 12, 31));
        assert_eq!(f.era.as_deref(), Some("ce"));
        let f = iso_to_fields("japanese", iso(1873, 1, 1));
        assert_eq!(f.era.as_deref(), Some("meiji"));
        assert_eq!(f.era_year, Some(6));
        // A later modern era is unaffected.
        let f = iso_to_fields("japanese", iso(2019, 5, 1));
        assert_eq!(f.era.as_deref(), Some("reiwa"));
        assert_eq!(f.era_year, Some(1));
    }

    #[test]
    fn era_erayear_pairing_and_validity() {
        // Exactly one of era / eraYear → TypeError (MissingFields); an unknown era
        // code (even alongside a valid year) → RangeError; a valid pair resolves.
        let only_era = fields_to_iso(
            "gregory",
            &FieldsInput {
                era: Some("ce".to_string()),
                year: Some(2000),
                month_code: Some("M01".to_string()),
                day: 1,
                ..Default::default()
            },
            Overflow::Reject,
        );
        assert!(matches!(only_era, Err(CalError::MissingFields(_))));

        let only_erayear = fields_to_iso(
            "gregory",
            &FieldsInput {
                era_year: Some(1),
                month_code: Some("M01".to_string()),
                day: 1,
                ..Default::default()
            },
            Overflow::Reject,
        );
        assert!(matches!(only_erayear, Err(CalError::MissingFields(_))));

        let bad_era = fields_to_iso(
            "buddhist",
            &FieldsInput {
                era: Some("xyz".to_string()),
                era_year: Some(2025),
                year: Some(2025),
                month_code: Some("M01".to_string()),
                day: 1,
                ..Default::default()
            },
            Overflow::Reject,
        );
        assert!(matches!(bad_era, Err(CalError::Range(_))));

        // A non-era calendar ignores era/eraYear entirely.
        assert!(
            fields_to_iso(
                "iso8601",
                &FieldsInput {
                    era: Some("xyz".to_string()),
                    era_year: Some(1),
                    year: Some(1970),
                    month_code: Some("M01".to_string()),
                    day: 1,
                    ..Default::default()
                },
                Overflow::Reject,
            )
            .is_ok()
        );
        // "ad" is an accepted alias for "ce".
        assert!(from_era("gregory", "ad", 2024, "M01").is_ok());
    }
}
