//! Astronomical lunisolar calendar arithmetic — the Chinese and Korean (dangi)
//! calendars.
//!
//! Unlike every other calendar Temporal supports, these are not arithmetic: a
//! month begins on the day of an astronomical **new moon**, and the year's shape
//! is fixed by where the **winter solstice** and the twelve *major solar terms*
//! (the sun crossing each multiple of 30° of longitude) fall — all observed at a
//! specific meridian, so China and Korea genuinely disagree on a handful of
//! years. A table cannot cover the range Temporal requires, so this module
//! computes the ephemeris directly.
//!
//! The algorithms follow Reingold & Dershowitz, *Calendrical Calculations*
//! (which is also what ICU4X implements, so the results agree with the reference
//! implementation the conformance suite was generated from), over Meeus'
//! series for the new moon and the sun's apparent longitude. Accuracy is well
//! inside a day across the range the suite pins exactly — Chinese years
//! 1900–2100, dangi 1900–2050 — and degrades gracefully outside it; the series
//! stay *defined and monotonic* everywhere, which is what lets the far-future /
//! far-past dates (±250 000) convert without error even though nobody claims
//! they are meteorologically meaningful.
//!
//! Times are **RD moments**: days (and fractions) since the proleptic Gregorian
//! `0001-01-01`, i.e. `JDN - 1_721_425`. Angles are degrees throughout, matching
//! the source material; [`crate::common::sin_deg`] supplies the trigonometry so
//! the module builds without `std`.

#[cfg(not(feature = "std"))]
use crate::common::FloatExt;
use crate::common::{cos_deg, sin_deg};

/// RD 0 in Julian Day Numbers: RD 1 is Gregorian `0001-01-01` = JDN 1 721 426.
pub(crate) const RD_EPOCH_JDN: i64 = 1_721_425;

/// The mean interval between new moons, in days.
const MEAN_SYNODIC_MONTH: f64 = 29.530_588_861;
/// The mean interval between vernal equinoxes, in days.
const MEAN_TROPICAL_YEAR: f64 = 365.242_189;
/// RD moment of 2000-01-01 12:00 (J2000), the epoch of the series below.
/// RD day 730 120 *is* 2000-01-01, and RD moments run from local midnight, so
/// noon of that day is `730_120.5`.
const J2000: f64 = 730_120.5;
/// Sun's longitude at the December solstice.
const WINTER_SOLSTICE: f64 = 270.0;

/// RD of the Chinese calendar's epoch — Gregorian −2636-02-15, the start of the
/// legendary first sexagenary cycle. Only the *year numbering* depends on it.
const CHINESE_EPOCH: f64 = -963_099.0;
/// The proleptic Gregorian year of [`CHINESE_EPOCH`] (−2636).
const CHINESE_EPOCH_GREGORIAN_YEAR: i64 = -2636;

/// Julian centuries from J2000 for an RD moment.
fn julian_centuries(t: f64) -> f64 {
    (dynamical_from_universal(t) - J2000) / 36525.0
}

/// Evaluates `a[0] + a[1]·x + a[2]·x² + …` (Horner).
fn poly(x: f64, a: &[f64]) -> f64 {
    a.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

/// `x mod y` with the sign of `y` (the book's `mod`), for the degree wrap-arounds.
fn amod_f(x: f64, y: f64) -> f64 {
    x - y * (x / y).floor()
}

/// The book's `amod`: like `mod` but yielding `y` instead of `0`.
fn amod(x: i64, y: i64) -> i64 {
    let m = x.rem_euclid(y);
    if m == 0 { y } else { m }
}

// ---------------------------------------------------------------------------
// Time scales
// ---------------------------------------------------------------------------

/// ΔT — the difference between Terrestrial (dynamical) and Universal time, in
/// days. Earth's rotation is irregular, so the correction is empirical: a
/// piecewise fit (NASA/Espenak–Meeus, as tabulated by Reingold & Dershowitz)
/// over the observed record, extrapolated by the parabola outside it.
fn ephemeris_correction(t: f64) -> f64 {
    let year = gregorian_year_from_rd(t.floor() as i64);
    let y = year as f64;
    // RD of January 1 of `year`, used by the fits expressed in fractional years.
    let c = (rd_from_gregorian(year, 7, 1) - rd_from_gregorian(1900, 1, 1)) as f64 / 36525.0;
    let seconds = if (2051..=2150).contains(&year) {
        // Interpolate between the 2050 fit and the long-term parabola.
        -20.0 + 32.0 * ((y - 1820.0) / 100.0).powi(2) - 0.5628 * (2150.0 - y)
    } else if (2006..=2050).contains(&year) {
        poly(y - 2000.0, &[62.92, 0.32217, 0.005589])
    } else if (1987..=2005).contains(&year) {
        poly(
            y - 2000.0,
            &[
                63.86,
                0.3345,
                -0.060_374,
                0.001_727_5,
                0.000_651_814,
                0.000_023_73,
            ],
        )
    } else if (1900..=1986).contains(&year) {
        poly(
            c,
            &[
                -0.00002, 0.000297, 0.025184, -0.181133, 0.553040, -0.861938, 0.677066, -0.212591,
            ],
        ) * 86400.0
    } else if (1800..=1899).contains(&year) {
        poly(
            c,
            &[
                -0.000009, 0.003844, 0.083563, 0.865736, 4.867575, 15.845535, 31.332267, 38.291999,
                28.316289, 11.636204, 2.043794,
            ],
        ) * 86400.0
    } else if (1700..=1799).contains(&year) {
        poly(
            y - 1700.0,
            &[8.118780842, -0.005092142, 0.003336121, -0.0000266484],
        )
    } else if (1600..=1699).contains(&year) {
        poly(y - 1600.0, &[120.0, -0.9808, -0.01532, 0.000140272128])
    } else if (500..=1599).contains(&year) {
        poly(
            (y - 1000.0) / 100.0,
            &[
                1574.2,
                -556.01,
                71.23472,
                0.319781,
                -0.8503463,
                -0.005050998,
                0.0083572073,
            ],
        )
    } else if (-499..500).contains(&year) {
        poly(
            y / 100.0,
            &[
                10583.6,
                -1014.41,
                33.78311,
                -5.952053,
                -0.1798452,
                0.022174192,
                0.0090316521,
            ],
        )
    } else {
        // Long-term parabolic extrapolation (Morrison & Stephenson).
        -20.0 + 32.0 * ((y - 1820.0) / 100.0).powi(2)
    };
    seconds / 86400.0
}

/// Dynamical time from universal time.
fn dynamical_from_universal(t: f64) -> f64 {
    t + ephemeris_correction(t)
}

/// Universal time from dynamical time.
fn universal_from_dynamical(t: f64) -> f64 {
    t - ephemeris_correction(t)
}

// ---------------------------------------------------------------------------
// Solar longitude
// ---------------------------------------------------------------------------

/// Coefficients `(x, y, z)` of the sun's periodic perturbation terms: each
/// contributes `x · sin(y + z·c)` arc-seconds, with `c` in Julian centuries.
#[rustfmt::skip]
const SOLAR_TERMS: [(f64, f64, f64); 49] = [
    (403406.0, 270.54861, 0.9287892), (195207.0, 340.19128, 35999.1376),
    (119433.0, 63.91854, 35999.4089), (112392.0, 331.2622, 35998.7155),
    (3891.0, 317.843, 71998.20), (2819.0, 86.631, 71998.4380),
    (1721.0, 240.052, 36000.35726), (660.0, 310.26, 71997.4812),
    (350.0, 247.23, 32964.4678), (334.0, 260.87, -19.4410),
    (314.0, 297.82, 445267.1117), (268.0, 343.14, 45036.8840),
    (242.0, 166.79, 3.1008), (234.0, 81.53, 22518.4434),
    (158.0, 3.50, -19.9739), (132.0, 132.75, 65928.9345),
    (129.0, 182.95, 9038.0293), (114.0, 162.03, 3034.7684),
    (99.0, 29.8, 33718.148), (93.0, 266.4, 3034.448),
    (86.0, 249.2, -2280.773), (78.0, 157.6, 29929.992),
    (72.0, 257.8, 31556.493), (68.0, 185.1, 149.588),
    (64.0, 69.9, 9037.750), (46.0, 8.0, 107997.405),
    (38.0, 197.1, -4444.176), (37.0, 250.4, 151.771),
    (32.0, 65.3, 67555.316), (29.0, 162.7, 31556.080),
    (28.0, 341.5, -4561.540), (27.0, 291.6, 107996.706),
    (27.0, 98.5, 1221.655), (25.0, 146.7, 62894.167),
    (24.0, 110.0, 31437.369), (21.0, 5.2, 14578.298),
    (21.0, 342.6, -31931.757), (20.0, 230.9, 34777.243),
    (18.0, 256.1, 1221.999), (17.0, 45.3, 62894.511),
    (14.0, 242.9, -4442.039), (13.0, 115.2, 107997.909),
    (13.0, 151.8, 119.066), (13.0, 285.3, 16859.071),
    (12.0, 53.3, -4.578), (10.0, 126.6, 26895.292),
    (10.0, 205.7, -39.127), (10.0, 85.9, 12297.536),
    (10.0, 146.1, 90073.778),
];

/// The sun's apparent geocentric longitude, in degrees, at RD moment `t`.
fn solar_longitude(t: f64) -> f64 {
    let c = julian_centuries(t);
    let sum: f64 = SOLAR_TERMS
        .iter()
        .map(|&(x, y, z)| x * sin_deg(y + z * c))
        .sum();
    // 5.729577951308232e-6 converts the arc-second sum to degrees.
    let longitude = 282.7771834 + 36_000.769_537_44 * c + 0.000_005_729_577_951_308_232 * sum;
    amod_f(longitude + aberration(c) + nutation(c), 360.0)
}

/// Nutation in longitude (degrees) — the wobble of Earth's axis.
fn nutation(c: f64) -> f64 {
    let a = poly(c, &[124.90, -1934.134, 0.002063]);
    let b = poly(c, &[201.11, 72001.5377, 0.00057]);
    -0.004778 * sin_deg(a) - 0.0003667 * sin_deg(b)
}

/// Aberration (degrees) — the apparent displacement from Earth's own motion.
fn aberration(c: f64) -> f64 {
    0.0000974 * cos_deg(177.63 + 35999.01848 * c) - 0.005575
}

// ---------------------------------------------------------------------------
// New moons
// ---------------------------------------------------------------------------

/// `(v, w, x, y, z)` of the new-moon correction terms: each contributes
/// `v · E^w · sin(x·solar + y·lunar + z·moon_argument)` days.
#[rustfmt::skip]
const NEW_MOON_TERMS: [(f64, i32, f64, f64, f64); 24] = [
    (-0.40720, 0, 0.0, 1.0, 0.0), (0.17241, 1, 1.0, 0.0, 0.0),
    (0.01608, 0, 0.0, 2.0, 0.0), (0.01039, 0, 0.0, 0.0, 2.0),
    (0.00739, 1, -1.0, 1.0, 0.0), (-0.00514, 1, 1.0, 1.0, 0.0),
    (0.00208, 2, 2.0, 0.0, 0.0), (-0.00111, 0, 0.0, 1.0, -2.0),
    (-0.00057, 0, 0.0, 1.0, 2.0), (0.00056, 1, 1.0, 2.0, 0.0),
    (-0.00042, 0, 0.0, 3.0, 0.0), (0.00042, 1, 1.0, 0.0, 2.0),
    (0.00038, 1, 1.0, 0.0, -2.0), (-0.00024, 1, -1.0, 2.0, 0.0),
    (-0.00017, 0, 0.0, 0.0, 1.0), (-0.00007, 0, 2.0, 1.0, 0.0),
    (0.00004, 0, 0.0, 2.0, -2.0), (0.00004, 0, 3.0, 0.0, 0.0),
    (0.00003, 0, 1.0, 1.0, -2.0), (0.00003, 0, 0.0, 2.0, 2.0),
    (-0.00003, 0, 1.0, 1.0, 2.0), (0.00003, 0, -1.0, 1.0, 2.0),
    (-0.00002, 0, -1.0, 1.0, -2.0), (0.00002, 0, 1.0, 3.0, 0.0),
];

/// `(i, j, l)` of the additional planetary-perturbation terms: each contributes
/// `l · sin(i + j·k)` days.
#[rustfmt::skip]
const NEW_MOON_EXTRA: [(f64, f64, f64); 13] = [
    (251.88, 0.016321, 0.000165), (251.83, 26.651886, 0.000164),
    (349.42, 36.412478, 0.000126), (84.66, 18.206239, 0.000110),
    (141.74, 53.303771, 0.000062), (207.14, 2.453732, 0.000060),
    (154.84, 7.306860, 0.000056), (34.52, 27.261239, 0.000047),
    (207.19, 0.121824, 0.000042), (291.34, 1.844379, 0.000040),
    (161.72, 24.198154, 0.000037), (239.56, 25.513099, 0.000035),
    (331.55, 3.592518, 0.000023),
];

/// The RD moment (universal time) of the `n`-th new moon after the one of
/// 2000-01-06, per Meeus' series.
fn nth_new_moon(n: i64) -> f64 {
    let k = (n - 24724) as f64;
    let c = k / 1236.85;
    let approx = J2000
        + poly(
            c,
            &[
                5.09766,
                MEAN_SYNODIC_MONTH * 1236.85,
                0.000_154_37,
                -0.000_000_150,
                0.000_000_000_73,
            ],
        );
    let e = poly(c, &[1.0, -0.002516, -0.0000074]);
    let solar = poly(
        c,
        &[2.5534, 1236.85 * 29.105_356_70, -0.0000014, -0.00000011],
    );
    let lunar = poly(
        c,
        &[
            201.5643,
            385.816_935_28 * 1236.85,
            0.0107582,
            0.00001238,
            -0.000_000_058,
        ],
    );
    let arg = poly(
        c,
        &[
            160.7108,
            390.670_502_84 * 1236.85,
            -0.0016118,
            -0.00000227,
            0.000_000_011,
        ],
    );
    let omega = poly(
        c,
        &[124.7746, -1.563_755_88 * 1236.85, 0.0020672, 0.00000215],
    );
    let correction: f64 = -0.00017 * sin_deg(omega)
        + NEW_MOON_TERMS
            .iter()
            .map(|&(v, w, x, y, z)| v * e.powi(w) * sin_deg(x * solar + y * lunar + z * arg))
            .sum::<f64>();
    let extra = 0.000325 * sin_deg(poly(c, &[299.77, 132.8475848, -0.009173]));
    let additional: f64 = NEW_MOON_EXTRA
        .iter()
        .map(|&(i, j, l)| l * sin_deg(i + j * k))
        .sum();
    universal_from_dynamical(approx + correction + extra + additional)
}

/// The index of the new moon at or before RD moment `t`.
///
/// `nth_new_moon` is very nearly linear in `n` with slope
/// [`MEAN_SYNODIC_MONTH`], so correcting the index by the residual converges in
/// a couple of steps — even at the far extremes, where ΔT alone displaces the
/// series by ~80 lunations and a fixed-step search would never reach it. A
/// bounded exact settle then guarantees `nth_new_moon(n) <= t < nth_new_moon(n+1)`.
fn new_moon_index_before(t: f64) -> i64 {
    let mut n = ((t - nth_new_moon(0)) / MEAN_SYNODIC_MONTH).round() as i64;
    for _ in 0..8 {
        let step = ((t - nth_new_moon(n)) / MEAN_SYNODIC_MONTH).round() as i64;
        if step == 0 {
            break;
        }
        n += step;
    }
    for _ in 0..8 {
        if nth_new_moon(n) > t {
            n -= 1;
        } else if nth_new_moon(n + 1) <= t {
            n += 1;
        } else {
            break;
        }
    }
    n
}

/// The RD moment of the last new moon strictly before `t`.
fn new_moon_before(t: f64) -> f64 {
    let n = new_moon_index_before(t);
    let m = nth_new_moon(n);
    if m < t { m } else { nth_new_moon(n - 1) }
}

// ---------------------------------------------------------------------------
// Observation meridians
// ---------------------------------------------------------------------------

/// Which of the two lunisolar calendars is being computed — they share every
/// rule and differ only in *where* the new moon and solar terms are observed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lunisolar {
    /// Chinese: Beijing local mean time before 1929, then UTC+8.
    Chinese,
    /// Korean (dangi): Seoul, whose legal offset changed five times.
    Dangi,
}

/// The observation meridian's offset from UT, in days, at RD moment `t`.
fn zone_offset(cal: Lunisolar, t: f64) -> f64 {
    let day = t.floor() as i64;
    let hours = match cal {
        Lunisolar::Chinese => {
            // Before the 1929 standard-time reform, Beijing local mean time
            // (116°25′E → 7h45m40s).
            if gregorian_year_from_rd(day) < 1929 {
                1397.0 / 180.0
            } else {
                8.0
            }
        }
        Lunisolar::Dangi => {
            // Korea's legal zone moved four times; the book's `korean-location`.
            if day < rd_from_gregorian(1908, 4, 1) {
                3809.0 / 450.0
            } else if day < rd_from_gregorian(1912, 1, 1) {
                8.5
            } else if day < rd_from_gregorian(1954, 3, 21) {
                9.0
            } else if day < rd_from_gregorian(1961, 8, 10) {
                8.5
            } else {
                9.0
            }
        }
    };
    hours / 24.0
}

/// Universal time of local midnight beginning RD day `day` at the meridian.
fn midnight(cal: Lunisolar, day: i64) -> f64 {
    day as f64 - zone_offset(cal, day as f64)
}

/// The local RD day containing universal moment `t`.
fn local_day(cal: Lunisolar, t: f64) -> i64 {
    (t + zone_offset(cal, t)).floor() as i64
}

// ---------------------------------------------------------------------------
// Calendar rules
// ---------------------------------------------------------------------------

/// The RD day of the December solstice on or before RD day `day`.
///
/// The defining condition is that the sun's longitude passes 270° during the
/// local day: `longitude(midnight(d)) <= 270 < longitude(midnight(d + 1))`.
/// Rather than scan up to that day — which is unbounded if the starting estimate
/// is poor, and *is* poor a quarter-million years out, where ΔT alone reaches
/// thousands of days — iterate the estimate to convergence and then nudge by at
/// most a few days.
fn winter_solstice_on_or_before(cal: Lunisolar, day: i64) -> i64 {
    let t = midnight(cal, day + 1);
    let rate = MEAN_TROPICAL_YEAR / 360.0;
    // Fixed-point iteration on "how far is the sun past 270°, in days".
    let mut tau = t - rate * amod_f(solar_longitude(t) - WINTER_SOLSTICE, 360.0);
    for _ in 0..6 {
        let delta = amod_f(solar_longitude(tau) - WINTER_SOLSTICE + 180.0, 360.0) - 180.0;
        if delta.abs() < 1e-6 {
            break;
        }
        tau -= rate * delta;
    }
    let past = |d: i64| solar_longitude(midnight(cal, d)) > WINTER_SOLSTICE;
    let mut d = local_day(cal, tau);
    // Converged to within a day; settle onto the day that actually straddles the
    // crossing. Both nudges are bounded, so a pathological argument degrades to a
    // slightly wrong day rather than a hang.
    for _ in 0..8 {
        if past(d) {
            d -= 1;
        } else if !past(d + 1) {
            d += 1;
        } else {
            break;
        }
    }
    // "On or before" is part of the contract every caller relies on.
    d.min(day)
}

/// The local RD day on which new moon number `n` falls.
fn local_day_of_moon(cal: Lunisolar, n: i64) -> i64 {
    local_day(cal, nth_new_moon(n))
}

/// The RD day of the last new moon strictly before RD day `day`.
fn new_moon_day_before(cal: Lunisolar, day: i64) -> i64 {
    local_day(cal, new_moon_before(midnight(cal, day)))
}

/// The index (1..=12) of the major solar term in force on RD day `day`.
fn current_major_solar_term(cal: Lunisolar, day: i64) -> i64 {
    let s = solar_longitude(midnight(cal, day));
    amod(2 + (s / 30.0).floor() as i64, 12)
}

/// One month of a sui: where it starts, the number it carries (1..=12), and
/// whether it is that number's intercalary twin.
#[derive(Clone, Copy)]
struct SuiMonth {
    start: i64,
    /// The start of the following month — i.e. one past this month's last day.
    /// The walk computes it anyway, so recording it saves the caller a search.
    end: i64,
    number: i64,
    leap: bool,
}

/// Every month of the sui (solstice-to-solstice year) containing RD day `day`,
/// in order, with its number and leap flag resolved.
///
/// This is the workhorse: both directions of the conversion need the same walk,
/// and doing it once — computing each month boundary and each solar term a
/// single time — is what keeps a conversion to a few dozen ephemeris
/// evaluations instead of a few hundred. The month *numbers* come from the
/// elapsed-lunation count, shifted down by one from the intercalary month
/// onward; the intercalary month is the first in the sui with no major solar
/// term, and only exists when the sui spans thirteen lunations.
fn months_of_sui(cal: Lunisolar, day: i64) -> ([SuiMonth; 14], usize) {
    let s1 = winter_solstice_on_or_before(cal, day);
    let s2 = winter_solstice_on_or_before(cal, s1 + 370);
    // The first new moon after the solstice starts month 12 (the solstice itself
    // falls in month 11), and the last before the next solstice starts month 11
    // of the following year.
    // Resolve the lunation *index* of the sui's first month once, then step it:
    // consecutive months are consecutive new moons, so each costs one
    // `nth_new_moon` rather than a fresh index search.
    let mut moon = new_moon_index_before(midnight(cal, s1 + 1)) + 1;
    let m12 = local_day_of_moon(cal, moon);
    let next_m11 = new_moon_day_before(cal, s2 + 1);
    let leap_sui = ((next_m11 - m12) as f64 / MEAN_SYNODIC_MONTH).round() as i64 == 12;

    let mut months = [SuiMonth {
        start: 0,
        end: 0,
        number: 0,
        leap: false,
    }; 14];
    let mut len = 0usize;
    let mut start = m12;
    let mut term = current_major_solar_term(cal, start);
    let mut leap_seen = false;
    let mut index = 0i64;
    while len < 14 {
        moon += 1;
        let next = local_day_of_moon(cal, moon);
        let next_term = current_major_solar_term(cal, next);
        // A month with no major solar term is the intercalary one — but only the
        // first such month in a thirteen-lunation sui.
        let no_term = leap_sui && term == next_term;
        let is_leap = no_term && !leap_seen;
        leap_seen |= no_term;
        months[len] = SuiMonth {
            start,
            end: next,
            number: amod(index - i64::from(leap_seen), 12),
            leap: is_leap,
        };
        len += 1;
        if start >= next_m11 {
            break;
        }
        start = next;
        term = next_term;
        index += 1;
    }
    (months, len)
}

/// The New Year (start of month 1) within an already-computed sui.
fn new_year_of(months: &[SuiMonth; 14], len: usize) -> i64 {
    (0..len)
        .find(|&i| months[i].number == 1 && !months[i].leap)
        .map_or(months[0].start, |i| months[i].start)
}

/// The sui whose New Year is the latest one on or before RD day `day`, returned
/// *with* its month list so the caller does not have to walk it again — the
/// walk is by far the most expensive thing this module does.
fn sui_of_year_containing(cal: Lunisolar, day: i64) -> ([SuiMonth; 14], usize, i64) {
    let (months, len) = months_of_sui(cal, day);
    let ny = new_year_of(&months, len);
    if day >= ny {
        return (months, len, ny);
    }
    // `day` is in months 11 or 12, before this sui's New Year; the lunisolar
    // year it belongs to began one sui earlier.
    let (months, len) = months_of_sui(cal, day - 180);
    let ny = new_year_of(&months, len);
    (months, len, ny)
}

/// The *related Gregorian year* of an RD day whose lunisolar month number is
/// `month` — the Gregorian year the containing New Year falls in, which is what
/// Temporal exposes as `year`.
///
/// Closed form from the elapsed mean tropical years since the calendar epoch
/// (Reingold & Dershowitz derive the sexagenary year the same way). Searching
/// for the New Year instead would double the cost of every conversion; the
/// `1.5 - month/12` term is what puts the boundary at the New Year rather than
/// at January 1.
fn related_gregorian_year(day: i64, month: i64) -> i64 {
    let elapsed_years = (1.5 - month as f64 / 12.0
        + (day as f64 - CHINESE_EPOCH) / MEAN_TROPICAL_YEAR)
        .floor() as i64;
    elapsed_years + CHINESE_EPOCH_GREGORIAN_YEAR - 1
}

/// A lunisolar date: the *related Gregorian year* (what Temporal exposes as
/// `year`), the nominal month 1..=12, whether it is that month's intercalary
/// twin, and the day of month.
pub(crate) struct LunisolarDate {
    pub year: i64,
    pub month: i64,
    pub day: i64,
    pub leap: bool,
}

/// The month containing RD day `day`.
///
/// A sui's month list begins at the first new moon *after* the solstice, so the
/// days from the solstice up to that new moon fall in the previous sui's last
/// month (month 11, the one the solstice itself is in) — that is the retry.
fn month_containing(cal: Lunisolar, day: i64) -> SuiMonth {
    let (months, len) = months_of_sui(cal, day);
    if let Some(i) = (0..len).rev().find(|&i| months[i].start <= day) {
        return months[i];
    }
    let (prev, prev_len) = months_of_sui(cal, months[0].start - 40);
    let i = (0..prev_len)
        .rev()
        .find(|&i| prev[i].start <= day)
        .unwrap_or(prev_len - 1);
    prev[i]
}

/// Converts an RD day to its lunisolar date.
fn from_rd(cal: Lunisolar, day: i64) -> LunisolarDate {
    let m = month_containing(cal, day);
    LunisolarDate {
        year: related_gregorian_year(day, m.number),
        month: m.number,
        day: day - m.start + 1,
        leap: m.leap,
    }
}

/// The sui containing lunisolar year `year`'s New Year, with that New Year.
///
/// Finds the New Year that [`related_gregorian_year`] itself labels `year`,
/// rather than assuming December 31 of the Gregorian year lands in it. The two
/// agree throughout the accurate range, but that labelling measures in *mean
/// tropical* years, which slip a fraction of a year against the Gregorian
/// calendar over a quarter-million of them — enough to cross a boundary.
/// Aligning on the reported year makes the round-trip exact by construction
/// wherever the labelling is monotonic, which it is everywhere.
fn year_start(cal: Lunisolar, year: i64) -> Option<([SuiMonth; 14], usize, i64)> {
    let mut anchor = rd_from_gregorian(year, 12, 31);
    let (mut months, mut len, mut new_year) = sui_of_year_containing(cal, anchor);
    for _ in 0..4 {
        // `new_year` starts month 1 by construction, so its related year is a
        // closed-form read — going back through `from_rd` would re-walk a whole
        // sui for a number already determined.
        let reported = related_gregorian_year(new_year, 1);
        if reported == year {
            return Some((months, len, new_year));
        }
        anchor += (year - reported) * 366;
        (months, len, new_year) = sui_of_year_containing(cal, anchor);
    }
    Some((months, len, new_year))
}

/// Converts a lunisolar date to its RD day, or `None` if that month/day does not
/// occur in that year (a leap month the year lacks, or day 30 of a 29-day month).
fn to_rd(cal: Lunisolar, year: i64, month: i64, day: i64, leap: bool) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=30).contains(&day) {
        return None;
    }
    // Find the New Year that `from_rd` itself labels with `year`, rather than
    // assuming December 31 of the Gregorian year lands in it. The two agree
    // throughout the accurate range, but `related_gregorian_year` measures in
    // *mean tropical* years, which slip a fraction of a year against the
    // Gregorian calendar over a quarter-million of them — enough to cross a
    // boundary. Aligning on the reported year makes the round-trip exact by
    // construction wherever the labelling is monotonic, which it is everywhere.
    let (months, len, new_year) = year_start(cal, year)?;
    // A lunisolar year spans two suis — months 1..11 sit in the sui containing
    // the New Year, month 12 in the next. The first is the walk just done; the
    // second probe is 400 days on, comfortably past the next solstice.
    let (next_months, next_len) = months_of_sui(cal, new_year + 400);
    for (ms, l) in [(&months, len), (&next_months, next_len)] {
        for m in ms.iter().take(l) {
            if m.number != month || m.leap != leap || m.start < new_year {
                continue;
            }
            if related_gregorian_year(m.start, m.number) != year {
                continue;
            }
            if day > m.end - m.start {
                return None; // the month is shorter than the requested day
            }
            return Some(m.start + day - 1);
        }
    }
    None
}

/// The intercalary month number of lunisolar year `year` (1..=12), or `0` if the
/// year has none.
///
/// One walk answers this. The obvious alternative — probing all twelve
/// `to_jdn(year, m, 1, leap = true)` — costs a couple of hundred ephemeris
/// walks per query, which is what made a single date conversion take
/// milliseconds when the calendar layer called it per field access.
#[must_use]
pub(crate) fn leap_month_of_year(cal: Lunisolar, year: i64) -> i64 {
    if year.unsigned_abs() > 1 << 40 {
        return 0;
    }
    let Some((months, len, new_year)) = year_start(cal, year) else {
        return 0;
    };
    // The intercalary month may be in either sui the lunisolar year spans.
    for (ms, l) in [(months, len), {
        let (m, l) = months_of_sui(cal, new_year + 400);
        (m, l)
    }] {
        for m in ms.iter().take(l) {
            if m.leap && m.start >= new_year && related_gregorian_year(m.start, m.number) == year {
                return m.number;
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Public entry points (JDN-based, matching the calendar layer)
// ---------------------------------------------------------------------------

/// The lunisolar `(related year, month 1..=12, day, is_leap_month)` of a JDN.
#[must_use]
pub(crate) fn from_jdn(cal: Lunisolar, jdn: i64) -> Option<LunisolarDate> {
    let day = jdn.checked_sub(RD_EPOCH_JDN)?;
    // Guard the series against arguments where the `f64` day count would lose
    // integer precision; the corpus only requires *some* answer out there.
    if day.unsigned_abs() > 1 << 52 {
        return None;
    }
    Some(from_rd(cal, day))
}

/// The JDN of a lunisolar date, or `None` if it does not occur.
#[must_use]
pub(crate) fn to_jdn(cal: Lunisolar, year: i64, month: i64, day: i64, leap: bool) -> Option<i64> {
    if year.unsigned_abs() > 1 << 40 {
        return None;
    }
    to_rd(cal, year, month, day, leap).map(|d| d + RD_EPOCH_JDN)
}

// ---------------------------------------------------------------------------
// Proleptic Gregorian helpers (RD-based)
// ---------------------------------------------------------------------------

/// RD day of a proleptic Gregorian date.
fn rd_from_gregorian(year: i64, month: i64, day: i64) -> i64 {
    let a = (14 - month).div_euclid(12);
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    let jdn = day + (153 * m + 2).div_euclid(5) + 365 * y + y.div_euclid(4) - y.div_euclid(100)
        + y.div_euclid(400)
        - 32045;
    jdn - RD_EPOCH_JDN
}

/// The proleptic Gregorian year containing RD day `day` (Fliegel–Van Flandern,
/// with Euclidean division so it stays correct for the deeply negative JDNs the
/// extreme-year range reaches).
fn gregorian_year_from_rd(day: i64) -> i64 {
    let a = day + RD_EPOCH_JDN + 32044;
    let b = (4 * a + 3).div_euclid(146097);
    let c = a - (146097 * b).div_euclid(4);
    let d = (4 * c + 3).div_euclid(1461);
    let e = c - (1461 * d).div_euclid(4);
    let m = (5 * e + 2).div_euclid(153);
    100 * b + d - 4800 + m.div_euclid(10)
}
#[cfg(test)]
mod tests;
