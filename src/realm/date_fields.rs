//! The `Date.prototype` getters and component setters as pure functions of
//! a time value, shared by the host's by-name dispatch
//! (`Interp::call_method_body`) and a memoized call site's lean form
//! (`nbvm::native_call`), so the two cannot diverge. The engine presents
//! every date in UTC, so `getX` and `getUTCX` (and `setX` / `setUTCX`) are
//! the same function.

/// A `Date.prototype` getter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DateGet {
    /// `getTime` / `valueOf`.
    Time,
    FullYear,
    /// Annex B.2.4.1 `getYear`.
    Year,
    Month,
    Date,
    Day,
    Hours,
    Minutes,
    Seconds,
    Millis,
    TimezoneOffset,
}

impl DateGet {
    /// The getter a `Date.prototype` method name is.
    pub(crate) fn of(name: &str) -> Option<Self> {
        Some(match name {
            "getTime" | "valueOf" => Self::Time,
            "getFullYear" | "getUTCFullYear" => Self::FullYear,
            "getYear" => Self::Year,
            "getMonth" | "getUTCMonth" => Self::Month,
            "getDate" | "getUTCDate" => Self::Date,
            "getDay" | "getUTCDay" => Self::Day,
            "getHours" | "getUTCHours" => Self::Hours,
            "getMinutes" | "getUTCMinutes" => Self::Minutes,
            "getSeconds" | "getUTCSeconds" => Self::Seconds,
            "getMilliseconds" | "getUTCMilliseconds" => Self::Millis,
            "getTimezoneOffset" => Self::TimezoneOffset,
            _ => return None,
        })
    }

    /// The getter's result for the time value `ms` (`NaN` for an invalid
    /// date).
    pub(crate) fn get(self, ms: f64) -> f64 {
        if !ms.is_finite() {
            return f64::NAN;
        }
        let t = ms as i64;
        let day = t.div_euclid(86_400_000);
        let tod = t.rem_euclid(86_400_000);
        let ymd = || super::civil_from_days(day);
        match self {
            Self::Time => ms,
            Self::FullYear => ymd().0 as f64,
            Self::Year => (ymd().0 - 1900) as f64,
            Self::Month => (ymd().1 - 1) as f64,
            Self::Date => ymd().2 as f64,
            Self::Day => (day.rem_euclid(7) + 4).rem_euclid(7) as f64,
            Self::Hours => (tod / 3_600_000) as f64,
            Self::Minutes => (tod / 60_000 % 60) as f64,
            Self::Seconds => (tod / 1000 % 60) as f64,
            Self::Millis => (tod % 1000) as f64,
            Self::TimezoneOffset => 0.0,
        }
    }
}

/// A `Date.prototype` component setter (`setFullYear` … `setMilliseconds`,
/// and their `UTC` forms).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DateSet {
    FullYear,
    Month,
    Date,
    Hours,
    Minutes,
    Seconds,
    Millis,
}

impl DateSet {
    /// The setter a `Date.prototype` method name is.
    pub(crate) fn of(name: &str) -> Option<Self> {
        Some(match name {
            "setFullYear" | "setUTCFullYear" => Self::FullYear,
            "setMonth" | "setUTCMonth" => Self::Month,
            "setDate" | "setUTCDate" => Self::Date,
            "setHours" | "setUTCHours" => Self::Hours,
            "setMinutes" | "setUTCMinutes" => Self::Minutes,
            "setSeconds" | "setUTCSeconds" => Self::Seconds,
            "setMilliseconds" | "setUTCMilliseconds" => Self::Millis,
            _ => return None,
        })
    }

    /// How many components the setter consumes, in order.
    pub(crate) fn max_components(self) -> usize {
        match self {
            Self::Hours => 4,
            Self::FullYear | Self::Minutes => 3,
            Self::Month | Self::Seconds => 2,
            Self::Date | Self::Millis => 1,
        }
    }

    /// The new time value of a date whose time value is `ms` after the
    /// setter with the (already ToNumber'd) components `comps`: `NaN` for an
    /// invalid result (and for an invalid date, except under `setFullYear`,
    /// which takes its time as +0).
    pub(crate) fn apply(self, ms: f64, comps: &[f64]) -> f64 {
        let is_full_year = self == Self::FullYear;
        let date_is_nan = !ms.is_finite();
        if date_is_nan && !is_full_year {
            return f64::NAN;
        }
        let t = ms as i64;
        let day = t.div_euclid(86_400_000);
        let tod = t.rem_euclid(86_400_000);
        let (y, mo, d) = super::civil_from_days(day);
        let (mut yy, mut mo0, mut dd) = (y, (mo as i64) - 1, d as i64);
        let mut hh = tod / 3_600_000;
        let mut mi = tod / 60_000 % 60;
        let mut ss = tod / 1000 % 60;
        let mut mss = tod % 1000;
        if is_full_year && date_is_nan {
            // The time taken as +0: every component its epoch value.
            (yy, mo0, dd, hh, mi, ss, mss) = (1970, 0, 1, 0, 0, 0, 0);
        }
        // Any NaN component makes the whole result NaN (TimeClip). `limit`
        // bounds each component so the `i64` combination below cannot wrap:
        // every limit is far outside the representable range (|t| ≤ 8.64e15
        // ms), so exceeding one is exactly MakeDay/TimeClip's NaN.
        let mut any_nan = false;
        let mut comp = |slot: &mut i64, idx: usize, limit: f64| {
            if let Some(&v) = comps.get(idx) {
                if v.is_nan() || !v.is_finite() || v > limit || v < -limit {
                    any_nan = true;
                }
                *slot = v as i64;
            }
        };
        const YEAR_MAX: f64 = 1.0e7;
        const MONTH_MAX: f64 = 1.0e8;
        const DATE_MAX: f64 = 1.0e10;
        const HOUR_MAX: f64 = 1.0e11;
        const MIN_MAX: f64 = 1.0e13;
        const SEC_MAX: f64 = 1.0e14;
        const MS_MAX: f64 = 1.0e17;
        match self {
            Self::FullYear => {
                comp(&mut yy, 0, YEAR_MAX);
                comp(&mut mo0, 1, MONTH_MAX);
                comp(&mut dd, 2, DATE_MAX);
            }
            Self::Month => {
                comp(&mut mo0, 0, MONTH_MAX);
                comp(&mut dd, 1, DATE_MAX);
            }
            Self::Date => comp(&mut dd, 0, DATE_MAX),
            Self::Hours => {
                comp(&mut hh, 0, HOUR_MAX);
                comp(&mut mi, 1, MIN_MAX);
                comp(&mut ss, 2, SEC_MAX);
                comp(&mut mss, 3, MS_MAX);
            }
            Self::Minutes => {
                comp(&mut mi, 0, MIN_MAX);
                comp(&mut ss, 1, SEC_MAX);
                comp(&mut mss, 2, MS_MAX);
            }
            Self::Seconds => {
                comp(&mut ss, 0, SEC_MAX);
                comp(&mut mss, 1, MS_MAX);
            }
            Self::Millis => comp(&mut mss, 0, MS_MAX),
        }
        if any_nan {
            return f64::NAN;
        }
        // A possibly out-of-range month normalized into the year, then the day
        // measured from the 1st (so out-of-range day/hour/… values roll over
        // by plain integer arithmetic).
        let yy2 = yy + mo0.div_euclid(12);
        let mo1 = (mo0.rem_euclid(12) + 1) as u32;
        let base_days = super::days_from_civil(yy2, mo1, 1) + (dd - 1);
        time_clip((base_days * 86_400_000 + hh * 3_600_000 + mi * 60_000 + ss * 1000 + mss) as f64)
    }
}

/// `Date.UTC(…)` (§21.4.3.4) of its arguments after ToNumber (at most the
/// seven read): `NaN` without a year or with a non-finite component.
pub(crate) fn date_utc(nums: &[f64]) -> f64 {
    let getn = |i: usize, dflt: f64| nums.get(i).copied().unwrap_or(dflt);
    let c = [
        getn(0, f64::NAN),
        getn(1, 0.0),
        getn(2, 1.0),
        getn(3, 0.0),
        getn(4, 0.0),
        getn(5, 0.0),
        getn(6, 0.0),
    ];
    if c.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    // MakeDate(MakeDay(y, m, d), MakeTime(h, min, s, milli)) in IEEE-754
    // `f64` (the two-digit-year mapping is folded into the helper).
    crate::nbexec::make_date_ms(c[0], c[1], c[2], c[3], c[4], c[5], c[6])
}

/// TimeClip (§21.4.1.31).
pub(crate) fn time_clip(t: f64) -> f64 {
    if !t.is_finite() || !(-8.64e15..=8.64e15).contains(&t) {
        return f64::NAN;
    }
    // (|t| ≤ 8.64e15 < 2^63: the `i64` truncation is exact ToIntegerOrInfinity.)
    let truncated = t as i64 as f64;
    if truncated == 0.0 { 0.0 } else { truncated }
}

impl super::Realm {
    /// Whether the cell at `h` is an ordinary object or a Date owning no
    /// property (the receivers `nbvm::to_prim` converts), on one read of it.
    #[inline]
    pub(crate) fn to_prim_receiver(&self, h: crate::heap::Handle) -> bool {
        match self.heap.get(h) {
            Some(crate::cell::Cell::Object(_)) => true,
            Some(crate::cell::Cell::Date(_)) => !self.has_aux_object(h),
            _ => false,
        }
    }

    /// `new Date(…)` of the (single) realm's intrinsic `Date`, with
    /// `new.target` the constructor itself, when no argument conversion can
    /// run code: one Number (its TimeClip) or one Date (its time value), or
    /// two to seven leading Numbers (MakeDate of the components). The new
    /// Date with the constructor's `prototype` (non-writable and
    /// non-configurable, so `GetPrototypeFromConstructor`'s answer), as the
    /// host's construct makes it; `None`, having done nothing, otherwise
    /// (no argument: the current time, the host's).
    pub(crate) fn new_default_date(
        &mut self,
        ctor: crate::nanbox::NanBox,
        args: &[crate::nanbox::NanBox],
    ) -> Option<crate::nanbox::NanBox> {
        use crate::heap::Handle;
        use crate::nanbox::NanBox;
        if self.multi_realm || args.is_empty() {
            return None;
        }
        let c = Handle::from_raw(ctor.as_handle()?);
        if self.native_at(c) != Some(crate::nbexec::N_DATE) {
            return None;
        }
        let ms = if let [a] = args {
            match a.as_number() {
                Some(n) => time_clip(n),
                None => time_clip(self.date_at(Handle::from_raw(a.as_handle()?))?),
            }
        } else {
            let mut n = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
            for (slot, a) in n.iter_mut().zip(args) {
                *slot = a.as_number()?;
            }
            if n.iter().any(|v| !v.is_finite()) {
                f64::NAN
            } else {
                crate::nbexec::make_date_ms(n[0], n[1], n[2], n[3], n[4], n[5], n[6])
            }
        };
        let proto = self.get_property(c, "prototype")?.as_handle()?;
        let d = self.new_date(ms);
        self.set_native_proto(d, Handle::from_raw(proto));
        Some(NanBox::handle(d.to_raw()))
    }
}
