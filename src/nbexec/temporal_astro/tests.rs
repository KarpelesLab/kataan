//! Ephemeris + lunisolar-calendar tests. The reference values are the Chinese
//! New Year dates and leap months of record.

use super::{Lunisolar::Chinese, Lunisolar::Dangi, *};

/// Gregorian `(y, m, d)` of a JDN, for readable assertions.
fn greg(jdn: i64) -> (i64, i64, i64) {
    let a = jdn + 32044;
    let b = (4 * a + 3).div_euclid(146097);
    let c = a - (146097 * b).div_euclid(4);
    let d = (4 * c + 3).div_euclid(1461);
    let e = c - (1461 * d).div_euclid(4);
    let m = (5 * e + 2).div_euclid(153);
    (
        100 * b + d - 4800 + m.div_euclid(10),
        m + 3 - 12 * m.div_euclid(10),
        e - (153 * m + 2).div_euclid(5) + 1,
    )
}

fn jdn_of(y: i64, m: i64, d: i64) -> i64 {
    rd_from_gregorian(y, m, d) + RD_EPOCH_JDN
}

#[test]
fn chinese_new_year_dates_of_record() {
    // Chinese New Year = month 1, day 1 of the related Gregorian year.
    let cases = [
        (1900, 1900, 1, 31),
        (1949, 1949, 1, 29),
        (1970, 1970, 2, 6),
        (2000, 2000, 2, 5),
        (2020, 2020, 1, 25),
        (2023, 2023, 1, 22),
        (2024, 2024, 2, 10),
        (2025, 2025, 1, 29),
        (2033, 2033, 1, 31),
        (2100, 2100, 2, 9),
    ];
    for (year, gy, gm, gd) in cases {
        let jdn = to_jdn(Chinese, year, 1, 1, false).expect("new year exists");
        assert_eq!(greg(jdn), (gy, gm, gd), "Chinese New Year {year}");
    }
}

#[test]
fn chinese_round_trips_across_the_exact_range() {
    // 1900-2100 is the range the conformance suite pins exactly. Sweep every
    // single day of a representative stretch (which spans several leap and
    // common suis, and both sides of the 1929 meridian change is covered by the
    // stride pass below), then stride across the whole range at 13 days — a
    // period sharing no factor with either the ~29.53-day lunar month or the
    // ~365.24-day year, so it lands on every phase of both.
    let mut checked = 0u32;
    let mut leap_months = 0u32;
    let mut sweep = |from: (i64, i64, i64), to: (i64, i64, i64), stride: i64| {
        let mut jdn = jdn_of(from.0, from.1, from.2);
        let end = jdn_of(to.0, to.1, to.2);
        while jdn <= end {
            let d = from_jdn(Chinese, jdn).expect("in range");
            assert!(
                (1..=12).contains(&d.month),
                "month {} at {:?}",
                d.month,
                greg(jdn)
            );
            assert!(
                (1..=30).contains(&d.day),
                "day {} at {:?}",
                d.day,
                greg(jdn)
            );
            if d.leap {
                leap_months += 1;
            }
            let back = to_jdn(Chinese, d.year, d.month, d.day, d.leap).unwrap_or_else(|| {
                panic!(
                    "no round-trip for {:?} at {:?}",
                    (d.year, d.month, d.day, d.leap),
                    greg(jdn)
                )
            });
            assert_eq!(back, jdn, "round-trip at {:?}", greg(jdn));
            checked += 1;
            jdn += stride;
        }
    };
    sweep((2010, 1, 1), (2025, 12, 31), 1);
    sweep((1900, 1, 31), (2100, 12, 31), 13);
    assert!(checked > 11_000, "only {checked} days checked");
    // Roughly one month in 2.7 years is intercalary, so a few percent of days.
    let pct = f64::from(leap_months) * 100.0 / f64::from(checked);
    assert!(
        (2.0..5.0).contains(&pct),
        "{pct:.1}% of sampled days were in leap months"
    );
}

#[test]
fn chinese_leap_months_of_record() {
    // (related year, leap month number) — the intercalary months of record.
    let cases = [
        (2001, 4),
        (2004, 2),
        (2006, 7),
        (2009, 5),
        (2012, 4),
        (2014, 9),
        (2017, 6),
        (2020, 4),
        (2023, 2),
        (2025, 6),
    ];
    for (year, want) in cases {
        let found = (1..=12)
            .find(|&m| to_jdn(Chinese, year, m, 1, true).is_some())
            .unwrap_or(0);
        assert_eq!(found, want, "leap month of {year}");
    }
    // A year with no leap month has none at any position.
    for year in [2002, 2005, 2010, 2019, 2024] {
        assert!(
            (1..=12).all(|m| to_jdn(Chinese, year, m, 1, true).is_none()),
            "{year} should have no leap month"
        );
    }
}

#[test]
fn dangi_diverges_from_chinese_where_the_meridian_matters() {
    // Korea and China keep the same rules at different meridians, so most years
    // agree and a few do not. 1900-2050 is dangi's exact range.
    let mut differ = 0;
    for year in 1900..=2050 {
        let c = to_jdn(Chinese, year, 1, 1, false);
        let k = to_jdn(Dangi, year, 1, 1, false);
        assert!(c.is_some() && k.is_some(), "new year {year}");
        if c != k {
            differ += 1;
        }
    }
    // They must not be identical (that would mean the meridian is ignored) nor
    // wildly different (that would mean the rules diverged).
    assert!(
        (1..=12).contains(&differ),
        "{differ} of 151 dangi new years differ from Chinese"
    );
    // The 1997 divergence is the well-known one: Korea's leap month that year is
    // month 7 while China's is month 5 — different suis entirely.
    assert!(to_jdn(Dangi, 2050, 12, 29, false).is_some());
}

#[test]
fn extreme_years_are_defined_not_panicking() {
    // The contract outside the accurate range is only that a date converts *at
    // all* — the suite explicitly does not check the value ("Create dates far in
    // the past and future but don't care about the conversion").
    for cal in [Chinese, Dangi] {
        for year in [-250_000, -5738, -4098, -2173, -180, 1, 1651, 250_000] {
            let jdn = to_jdn(cal, year, 1, 1, false);
            assert!(jdn.is_some(), "{year}-01-01 should convert");
            // Whatever it produced must itself be a well-formed lunisolar date.
            let d = from_jdn(cal, jdn.unwrap()).expect("converts back");
            assert!((1..=12).contains(&d.month) && (1..=30).contains(&d.day));
        }
    }
    // Within the range where the ephemeris is meaningful the related year must
    // round-trip exactly, including well before the tabulated era.
    for year in [1, 1000, 1651, 1899, 1900, 2000, 2100, 2400] {
        let jdn = to_jdn(Chinese, year, 1, 1, false).expect("converts");
        let d = from_jdn(Chinese, jdn).expect("converts back");
        assert_eq!((d.year, d.month, d.day), (year, 1, 1), "round-trip {year}");
    }
}

#[test]
fn solar_longitude_hits_the_solstices_and_equinoxes() {
    // Solar longitude is 270° at the December solstice, 0° at the March equinox.
    // 2024: March equinox Mar 20, December solstice Dec 21 (UT).
    let march = solar_longitude(rd_from_gregorian(2024, 3, 20) as f64 + 3.0 / 24.0);
    assert!(
        !(0.5..=359.5).contains(&march),
        "March equinox longitude {march}"
    );
    let december = solar_longitude(rd_from_gregorian(2024, 12, 21) as f64 + 10.0 / 24.0);
    assert!(
        (december - 270.0).abs() < 0.5,
        "December solstice longitude {december}"
    );
}

#[test]
fn new_moons_match_published_times() {
    // Published new moons (UT). Within an hour is ample — the calendar only
    // needs the right *day* at the meridian.
    for (y, m, d, hour) in [
        (2024, 1, 11, 11.0),
        (2024, 7, 5, 22.6),
        (2000, 1, 6, 18.2),
        (1900, 1, 1, 13.9),
    ] {
        let target = rd_from_gregorian(y, m, d) as f64 + hour / 24.0;
        let moon = nth_new_moon(new_moon_index_before(target - 1.0) + 1);
        assert!(
            (moon - target).abs() < 1.0 / 24.0,
            "new moon {y}-{m}-{d}: got {} expected {target}",
            moon
        );
    }
}

#[test]
fn ephemeris_correction_is_continuous_and_plausible() {
    // Known ΔT values (seconds): ~63.8 in 2000, ~-2.7 in 1900, ~69 in 2020.
    let dt = |y| ephemeris_correction(rd_from_gregorian(y, 1, 1) as f64) * 86400.0;
    assert!((dt(2000) - 63.8).abs() < 2.0, "ΔT(2000) = {}", dt(2000));
    assert!((dt(1900) - (-2.8)).abs() < 3.0, "ΔT(1900) = {}", dt(1900));
    assert!((dt(2020) - 69.0).abs() < 3.0, "ΔT(2020) = {}", dt(2020));
    // The piecewise fits must join without a gross step. ΔT itself ranges from
    // seconds (modern) to hours (antiquity), so the tolerance is relative to the
    // magnitude there rather than absolute.
    for boundary in [500, 1600, 1700, 1800, 1900, 1987, 2006, 2051, 2151] {
        let before = dt(boundary - 1);
        let after = dt(boundary + 1);
        let scale = before.abs().max(after.abs()).max(10.0);
        assert!(
            (after - before).abs() < 0.15 * scale,
            "ΔT jumps {} s across {boundary} (from {before} to {after})",
            after - before
        );
    }
}
