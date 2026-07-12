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

/// `disambiguation` option: how to resolve a wall-clock time that maps to zero
/// (spring-forward gap) or two (fall-back overlap) exact instants.
#[derive(Clone, Copy, PartialEq)]
enum Disamb {
    Compatible,
    Earlier,
    Later,
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
pub(crate) fn parse_offset_id(s: &str) -> Option<(i128, String)> {
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
pub(crate) fn resolve_named(s: &str) -> Option<String> {
    timezone_data::load_insensitive(s)
        .ok()
        .map(|z| z.name().to_string())
}

/// IANA "backward" link table: a non-primary time-zone identifier (a link) mapped
/// to its primary (canonical) target. Sorted by key for binary search. Derived
/// from the tzdb `backward` file; the UTC/GMT-zero family is handled separately in
/// [`tz_primary`] (their Temporal primary identifier is "UTC", not the tzdb target).
static TZ_LINKS: &[(&str, &str)] = &[
    ("Africa/Accra", "Africa/Abidjan"),
    ("Africa/Addis_Ababa", "Africa/Nairobi"),
    ("Africa/Asmara", "Africa/Nairobi"),
    ("Africa/Asmera", "Africa/Nairobi"),
    ("Africa/Bamako", "Africa/Abidjan"),
    ("Africa/Bangui", "Africa/Lagos"),
    ("Africa/Banjul", "Africa/Abidjan"),
    ("Africa/Blantyre", "Africa/Maputo"),
    ("Africa/Brazzaville", "Africa/Lagos"),
    ("Africa/Bujumbura", "Africa/Maputo"),
    ("Africa/Conakry", "Africa/Abidjan"),
    ("Africa/Dakar", "Africa/Abidjan"),
    ("Africa/Dar_es_Salaam", "Africa/Nairobi"),
    ("Africa/Djibouti", "Africa/Nairobi"),
    ("Africa/Douala", "Africa/Lagos"),
    ("Africa/Freetown", "Africa/Abidjan"),
    ("Africa/Gaborone", "Africa/Maputo"),
    ("Africa/Harare", "Africa/Maputo"),
    ("Africa/Kampala", "Africa/Nairobi"),
    ("Africa/Kigali", "Africa/Maputo"),
    ("Africa/Kinshasa", "Africa/Lagos"),
    ("Africa/Libreville", "Africa/Lagos"),
    ("Africa/Lome", "Africa/Abidjan"),
    ("Africa/Luanda", "Africa/Lagos"),
    ("Africa/Lubumbashi", "Africa/Maputo"),
    ("Africa/Lusaka", "Africa/Maputo"),
    ("Africa/Malabo", "Africa/Lagos"),
    ("Africa/Maseru", "Africa/Johannesburg"),
    ("Africa/Mbabane", "Africa/Johannesburg"),
    ("Africa/Mogadishu", "Africa/Nairobi"),
    ("Africa/Niamey", "Africa/Lagos"),
    ("Africa/Nouakchott", "Africa/Abidjan"),
    ("Africa/Ouagadougou", "Africa/Abidjan"),
    ("Africa/Porto-Novo", "Africa/Lagos"),
    ("Africa/Timbuktu", "Africa/Abidjan"),
    ("America/Anguilla", "America/Puerto_Rico"),
    ("America/Antigua", "America/Puerto_Rico"),
    (
        "America/Argentina/ComodRivadavia",
        "America/Argentina/Catamarca",
    ),
    ("America/Aruba", "America/Puerto_Rico"),
    ("America/Atikokan", "America/Panama"),
    ("America/Atka", "America/Adak"),
    ("America/Blanc-Sablon", "America/Puerto_Rico"),
    ("America/Buenos_Aires", "America/Argentina/Buenos_Aires"),
    ("America/Catamarca", "America/Argentina/Catamarca"),
    ("America/Cayman", "America/Panama"),
    ("America/Coral_Harbour", "America/Panama"),
    ("America/Cordoba", "America/Argentina/Cordoba"),
    ("America/Creston", "America/Phoenix"),
    ("America/Curacao", "America/Puerto_Rico"),
    ("America/Dominica", "America/Puerto_Rico"),
    ("America/Ensenada", "America/Tijuana"),
    ("America/Fort_Wayne", "America/Indiana/Indianapolis"),
    ("America/Godthab", "America/Nuuk"),
    ("America/Grenada", "America/Puerto_Rico"),
    ("America/Guadeloupe", "America/Puerto_Rico"),
    ("America/Indianapolis", "America/Indiana/Indianapolis"),
    ("America/Jujuy", "America/Argentina/Jujuy"),
    ("America/Knox_IN", "America/Indiana/Knox"),
    ("America/Kralendijk", "America/Puerto_Rico"),
    ("America/Louisville", "America/Kentucky/Louisville"),
    ("America/Lower_Princes", "America/Puerto_Rico"),
    ("America/Marigot", "America/Puerto_Rico"),
    ("America/Mendoza", "America/Argentina/Mendoza"),
    ("America/Montreal", "America/Toronto"),
    ("America/Montserrat", "America/Puerto_Rico"),
    ("America/Nassau", "America/Toronto"),
    ("America/Nipigon", "America/Toronto"),
    ("America/Pangnirtung", "America/Iqaluit"),
    ("America/Port_of_Spain", "America/Puerto_Rico"),
    ("America/Porto_Acre", "America/Rio_Branco"),
    ("America/Rainy_River", "America/Winnipeg"),
    ("America/Rosario", "America/Argentina/Cordoba"),
    ("America/Santa_Isabel", "America/Tijuana"),
    ("America/Shiprock", "America/Denver"),
    ("America/St_Barthelemy", "America/Puerto_Rico"),
    ("America/St_Kitts", "America/Puerto_Rico"),
    ("America/St_Lucia", "America/Puerto_Rico"),
    ("America/St_Thomas", "America/Puerto_Rico"),
    ("America/St_Vincent", "America/Puerto_Rico"),
    ("America/Thunder_Bay", "America/Toronto"),
    ("America/Tortola", "America/Puerto_Rico"),
    ("America/Virgin", "America/Puerto_Rico"),
    ("America/Yellowknife", "America/Edmonton"),
    ("Antarctica/DumontDUrville", "Pacific/Port_Moresby"),
    ("Antarctica/McMurdo", "Pacific/Auckland"),
    ("Antarctica/South_Pole", "Pacific/Auckland"),
    ("Antarctica/Syowa", "Asia/Riyadh"),
    ("Arctic/Longyearbyen", "Europe/Berlin"),
    ("Asia/Aden", "Asia/Riyadh"),
    ("Asia/Ashkhabad", "Asia/Ashgabat"),
    ("Asia/Bahrain", "Asia/Qatar"),
    ("Asia/Brunei", "Asia/Kuching"),
    ("Asia/Calcutta", "Asia/Kolkata"),
    ("Asia/Choibalsan", "Asia/Ulaanbaatar"),
    ("Asia/Chongqing", "Asia/Shanghai"),
    ("Asia/Chungking", "Asia/Shanghai"),
    ("Asia/Dacca", "Asia/Dhaka"),
    ("Asia/Harbin", "Asia/Shanghai"),
    ("Asia/Istanbul", "Europe/Istanbul"),
    ("Asia/Kashgar", "Asia/Urumqi"),
    ("Asia/Katmandu", "Asia/Kathmandu"),
    ("Asia/Kuala_Lumpur", "Asia/Singapore"),
    ("Asia/Kuwait", "Asia/Riyadh"),
    ("Asia/Macao", "Asia/Macau"),
    ("Asia/Muscat", "Asia/Dubai"),
    ("Asia/Phnom_Penh", "Asia/Bangkok"),
    ("Asia/Rangoon", "Asia/Yangon"),
    ("Asia/Saigon", "Asia/Ho_Chi_Minh"),
    ("Asia/Tel_Aviv", "Asia/Jerusalem"),
    ("Asia/Thimbu", "Asia/Thimphu"),
    ("Asia/Ujung_Pandang", "Asia/Makassar"),
    ("Asia/Ulan_Bator", "Asia/Ulaanbaatar"),
    ("Asia/Vientiane", "Asia/Bangkok"),
    ("Atlantic/Faeroe", "Atlantic/Faroe"),
    ("Atlantic/Jan_Mayen", "Europe/Berlin"),
    ("Atlantic/Reykjavik", "Africa/Abidjan"),
    ("Atlantic/St_Helena", "Africa/Abidjan"),
    ("Australia/ACT", "Australia/Sydney"),
    ("Australia/Canberra", "Australia/Sydney"),
    ("Australia/Currie", "Australia/Hobart"),
    ("Australia/LHI", "Australia/Lord_Howe"),
    ("Australia/NSW", "Australia/Sydney"),
    ("Australia/North", "Australia/Darwin"),
    ("Australia/Queensland", "Australia/Brisbane"),
    ("Australia/South", "Australia/Adelaide"),
    ("Australia/Tasmania", "Australia/Hobart"),
    ("Australia/Victoria", "Australia/Melbourne"),
    ("Australia/West", "Australia/Perth"),
    ("Australia/Yancowinna", "Australia/Broken_Hill"),
    ("Brazil/Acre", "America/Rio_Branco"),
    ("Brazil/DeNoronha", "America/Noronha"),
    ("Brazil/East", "America/Sao_Paulo"),
    ("Brazil/West", "America/Manaus"),
    ("CET", "Europe/Brussels"),
    ("CST6CDT", "America/Chicago"),
    ("Canada/Atlantic", "America/Halifax"),
    ("Canada/Central", "America/Winnipeg"),
    ("Canada/Eastern", "America/Toronto"),
    ("Canada/Mountain", "America/Edmonton"),
    ("Canada/Newfoundland", "America/St_Johns"),
    ("Canada/Pacific", "America/Vancouver"),
    ("Canada/Saskatchewan", "America/Regina"),
    ("Canada/Yukon", "America/Whitehorse"),
    ("Chile/Continental", "America/Santiago"),
    ("Chile/EasterIsland", "Pacific/Easter"),
    ("Cuba", "America/Havana"),
    ("EET", "Europe/Athens"),
    ("EST", "America/Panama"),
    ("EST5EDT", "America/New_York"),
    ("Egypt", "Africa/Cairo"),
    ("Eire", "Europe/Dublin"),
    ("Europe/Amsterdam", "Europe/Brussels"),
    ("Europe/Belfast", "Europe/London"),
    ("Europe/Bratislava", "Europe/Prague"),
    ("Europe/Busingen", "Europe/Zurich"),
    ("Europe/Copenhagen", "Europe/Berlin"),
    ("Europe/Guernsey", "Europe/London"),
    ("Europe/Isle_of_Man", "Europe/London"),
    ("Europe/Jersey", "Europe/London"),
    ("Europe/Kiev", "Europe/Kyiv"),
    ("Europe/Ljubljana", "Europe/Belgrade"),
    ("Europe/Luxembourg", "Europe/Brussels"),
    ("Europe/Mariehamn", "Europe/Helsinki"),
    ("Europe/Monaco", "Europe/Paris"),
    ("Europe/Nicosia", "Asia/Nicosia"),
    ("Europe/Oslo", "Europe/Berlin"),
    ("Europe/Podgorica", "Europe/Belgrade"),
    ("Europe/San_Marino", "Europe/Rome"),
    ("Europe/Sarajevo", "Europe/Belgrade"),
    ("Europe/Skopje", "Europe/Belgrade"),
    ("Europe/Stockholm", "Europe/Berlin"),
    ("Europe/Tiraspol", "Europe/Chisinau"),
    ("Europe/Uzhgorod", "Europe/Kyiv"),
    ("Europe/Vaduz", "Europe/Zurich"),
    ("Europe/Vatican", "Europe/Rome"),
    ("Europe/Zagreb", "Europe/Belgrade"),
    ("Europe/Zaporozhye", "Europe/Kyiv"),
    ("GB", "Europe/London"),
    ("GB-Eire", "Europe/London"),
    ("HST", "Pacific/Honolulu"),
    ("Hongkong", "Asia/Hong_Kong"),
    ("Iceland", "Africa/Abidjan"),
    ("Indian/Antananarivo", "Africa/Nairobi"),
    ("Indian/Christmas", "Asia/Bangkok"),
    ("Indian/Cocos", "Asia/Yangon"),
    ("Indian/Comoro", "Africa/Nairobi"),
    ("Indian/Kerguelen", "Indian/Maldives"),
    ("Indian/Mahe", "Asia/Dubai"),
    ("Indian/Mayotte", "Africa/Nairobi"),
    ("Indian/Reunion", "Asia/Dubai"),
    ("Iran", "Asia/Tehran"),
    ("Israel", "Asia/Jerusalem"),
    ("Jamaica", "America/Jamaica"),
    ("Japan", "Asia/Tokyo"),
    ("Kwajalein", "Pacific/Kwajalein"),
    ("Libya", "Africa/Tripoli"),
    ("MET", "Europe/Brussels"),
    ("MST", "America/Phoenix"),
    ("MST7MDT", "America/Denver"),
    ("Mexico/BajaNorte", "America/Tijuana"),
    ("Mexico/BajaSur", "America/Mazatlan"),
    ("Mexico/General", "America/Mexico_City"),
    ("NZ", "Pacific/Auckland"),
    ("NZ-CHAT", "Pacific/Chatham"),
    ("Navajo", "America/Denver"),
    ("PRC", "Asia/Shanghai"),
    ("PST8PDT", "America/Los_Angeles"),
    ("Pacific/Chuuk", "Pacific/Port_Moresby"),
    ("Pacific/Enderbury", "Pacific/Kanton"),
    ("Pacific/Funafuti", "Pacific/Tarawa"),
    ("Pacific/Johnston", "Pacific/Honolulu"),
    ("Pacific/Majuro", "Pacific/Tarawa"),
    ("Pacific/Midway", "Pacific/Pago_Pago"),
    ("Pacific/Pohnpei", "Pacific/Guadalcanal"),
    ("Pacific/Ponape", "Pacific/Guadalcanal"),
    ("Pacific/Saipan", "Pacific/Guam"),
    ("Pacific/Samoa", "Pacific/Pago_Pago"),
    ("Pacific/Truk", "Pacific/Port_Moresby"),
    ("Pacific/Wake", "Pacific/Tarawa"),
    ("Pacific/Wallis", "Pacific/Tarawa"),
    ("Pacific/Yap", "Pacific/Port_Moresby"),
    ("Poland", "Europe/Warsaw"),
    ("Portugal", "Europe/Lisbon"),
    ("ROC", "Asia/Taipei"),
    ("ROK", "Asia/Seoul"),
    ("Singapore", "Asia/Singapore"),
    ("Turkey", "Europe/Istanbul"),
    ("US/Alaska", "America/Anchorage"),
    ("US/Aleutian", "America/Adak"),
    ("US/Arizona", "America/Phoenix"),
    ("US/Central", "America/Chicago"),
    ("US/East-Indiana", "America/Indiana/Indianapolis"),
    ("US/Eastern", "America/New_York"),
    ("US/Hawaii", "Pacific/Honolulu"),
    ("US/Indiana-Starke", "America/Indiana/Knox"),
    ("US/Michigan", "America/Detroit"),
    ("US/Mountain", "America/Denver"),
    ("US/Pacific", "America/Los_Angeles"),
    ("US/Samoa", "Pacific/Pago_Pago"),
    ("W-SU", "Europe/Moscow"),
    ("WET", "Europe/Lisbon"),
];

/// The UTC/GMT-zero family, whose Temporal *primary* time-zone identifier is "UTC"
/// (used for `equals`/`until`/`since` comparison; the display id is still preserved).
fn is_utc_family(id: &str) -> bool {
    matches!(
        id,
        "UTC"
            | "Etc/UTC"
            | "Etc/UCT"
            | "UCT"
            | "Etc/Universal"
            | "Universal"
            | "Etc/Zulu"
            | "Zulu"
            | "Etc/GMT"
            | "Etc/GMT+0"
            | "Etc/GMT-0"
            | "Etc/GMT0"
            | "Etc/Greenwich"
            | "GMT"
            | "GMT+0"
            | "GMT-0"
            | "GMT0"
            | "Greenwich"
    )
}

/// `TimeZoneEquals`/primary-identifier resolution: maps a (case-normalized) IANA
/// time-zone identifier to its Temporal *primary* identifier, so that links compare
/// equal to their canonical zone (e.g. `Asia/Calcutta` → `Asia/Kolkata`, every
/// UTC/GMT-zero alias → `UTC`). Offset identifiers and unknown names are returned
/// unchanged. Only the primary is used for comparison; the stored/display id is not
/// rewritten.
pub(crate) fn tz_primary(id: &str) -> String {
    if is_utc_family(id) {
        return String::from("UTC");
    }
    if let Ok(idx) = TZ_LINKS.binary_search_by(|(k, _)| (*k).cmp(id)) {
        return String::from(TZ_LINKS[idx].1);
    }
    String::from(id)
}

/// The offset (ns east of UTC) of `tz` at the exact instant `epoch_ns`.
/// `GetOffsetNanosecondsFor(tz, epoch_ns)`.
pub(crate) fn tz_offset_at(tz: &str, epoch_ns: i128) -> i128 {
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

/// `GetPossibleEpochNanoseconds(tz, wall_ns)`: the list (0, 1, or 2 entries, sorted
/// ascending) of exact instants whose local wall time in `tz` equals `wall_ns`
/// (treated as an ISO date-time, i.e. a UTC epoch count of the wall clock). Zero
/// entries = a spring-forward gap; two = a fall-back overlap.
///
/// Follows the reference algorithm: probe the zone's offset one day before and one
/// day after the wall time (which straddles at most one transition), form a
/// candidate instant for each distinct offset, and keep those whose local time
/// round-trips back to `wall_ns`.
fn possible_instants(tz: &str, wall_ns: i128) -> ([i128; 2], usize) {
    if let Some((off, _)) = parse_offset_id(tz) {
        return ([wall_ns - off, 0], 1);
    }
    let Ok(z) = timezone_data::load(tz) else {
        return ([wall_ns, 0], 1);
    };
    let off_at = |epoch: i128| -> i128 {
        let secs = epoch.div_euclid(iso::NS_PER_SEC) as i64;
        i128::from(z.lookup(secs).offset) * iso::NS_PER_SEC
    };
    let day = iso::NS_PER_DAY;
    // Clamp the probe points to the representable range (per the spec, so the
    // ±1-day probe never overflows near the edges).
    let ns_earlier = if wall_ns - day < iso::MIN_EPOCH_NS - day {
        wall_ns
    } else {
        wall_ns - day
    };
    let ns_later = if wall_ns + day > iso::MAX_EPOCH_NS + day {
        wall_ns
    } else {
        wall_ns + day
    };
    let off_earlier = off_at(ns_earlier);
    let off_later = off_at(ns_later);

    let mut out = [0_i128; 2];
    let mut n = 0_usize;
    let consider = |off: i128, out: &mut [i128; 2], n: &mut usize| {
        let epoch = wall_ns - off;
        // Keep only if the candidate's own local time is exactly `wall_ns`.
        if epoch + off_at(epoch) == wall_ns && !(*n == 1 && out[0] == epoch) {
            out[*n] = epoch;
            *n += 1;
        }
    };
    consider(off_earlier, &mut out, &mut n);
    if off_later != off_earlier {
        consider(off_later, &mut out, &mut n);
    }
    if n == 2 && out[0] > out[1] {
        out.swap(0, 1);
    }
    (out, n)
}

/// `DisambiguatePossibleEpochNanoseconds(tz, wall_ns, disambiguation)`: resolves the
/// wall time to a single exact instant. `Err(())` signals the `reject` conflict
/// (a gap or an overlap) that the caller turns into a `RangeError`.
fn disambiguate(tz: &str, wall_ns: i128, d: Disamb) -> Result<i128, ()> {
    let (poss, n) = possible_instants(tz, wall_ns);
    if n == 1 {
        return Ok(poss[0]);
    }
    if n == 2 {
        // Fall-back overlap.
        return match d {
            Disamb::Earlier | Disamb::Compatible => Ok(poss[0]),
            Disamb::Later => Ok(poss[1]),
            Disamb::Reject => Err(()),
        };
    }
    // n == 0: a spring-forward gap.
    if d == Disamb::Reject {
        return Err(());
    }
    if let Some((off, _)) = parse_offset_id(tz) {
        // Fixed-offset zones never have gaps.
        return Ok(wall_ns - off);
    }
    // The size of the gap is (offsetAfter − offsetBefore); shift the wall time by it
    // and re-resolve into the adjacent interval.
    let day = iso::NS_PER_DAY;
    let ns_earlier = if wall_ns - day < iso::MIN_EPOCH_NS - day {
        wall_ns
    } else {
        wall_ns - day
    };
    let ns_later = if wall_ns + day > iso::MAX_EPOCH_NS + day {
        wall_ns
    } else {
        wall_ns + day
    };
    let off_before = tz_offset_at(tz, ns_earlier);
    let off_after = tz_offset_at(tz, ns_later);
    let gap = off_after - off_before;
    if d == Disamb::Earlier {
        let (p, m) = possible_instants(tz, wall_ns - gap);
        Ok(if m > 0 {
            p[0]
        } else {
            wall_ns - gap - off_before
        })
    } else {
        // Compatible or Later: resolve forward into the later interval.
        let (p, m) = possible_instants(tz, wall_ns + gap);
        Ok(if m > 0 {
            p[m - 1]
        } else {
            wall_ns + gap - off_after
        })
    }
}

/// The `offset: prefer|reject` reconciliation: scans the possible instants for
/// `wall_ns` and returns the one whose implied offset matches `off`. Under
/// `MATCH_MINUTES` (a minute-precision source offset string) an instant whose
/// offset rounds to the same minute matches; otherwise (`MATCH_EXACTLY`) the
/// offset must be identical to the nanosecond.
fn offset_match(tz: &str, wall_ns: i128, off: i128, match_minutes: bool) -> Option<i128> {
    let (poss, n) = possible_instants(tz, wall_ns);
    for &epoch in poss.iter().take(n) {
        let candidate_offset = wall_ns - epoch;
        if candidate_offset == off {
            return Some(epoch);
        }
        if match_minutes
            && round_signed(candidate_offset, iso::NS_PER_MINUTE, RoundMode::HalfExpand) == off
        {
            return Some(epoch);
        }
    }
    None
}

/// `GetEpochNanosecondsFor(tz, wall_ns, "compatible")`: the exact instant whose local
/// wall time is `wall_ns`, using the default (compatible) disambiguation, which never
/// conflicts.
pub(crate) fn wall_to_epoch(tz: &str, wall_ns: i128) -> i128 {
    disambiguate(tz, wall_ns, Disamb::Compatible)
        .unwrap_or_else(|()| wall_ns - tz_offset_at(tz, wall_ns))
}

/// `GetEpochNanosecondsFor(tz, wall_ns, disambiguation)`: resolves a wall time to a
/// single exact instant honouring the named `disambiguation` option
/// (`compatible`/`earlier`/`later`/`reject`). `Err(())` is the `reject` conflict on a
/// gap/overlap — the caller turns it into a `RangeError`. Unlike a naïve
/// offset-at-the-wall-instant conversion, this is correct near DST transitions.
pub(crate) fn epoch_for_wall_disamb(tz: &str, wall_ns: i128, disamb: &str) -> Result<i128, ()> {
    let d = match disamb {
        "earlier" => Disamb::Earlier,
        "later" => Disamb::Later,
        "reject" => Disamb::Reject,
        _ => Disamb::Compatible,
    };
    disambiguate(tz, wall_ns, d)
}

/// `GetStartOfDay(tz, isoDate)`: the first exact instant of the calendar day
/// `epoch_days` (days since the ISO epoch) in `tz`.
///
/// Normally this is local midnight, but when midnight falls inside a
/// spring-forward gap it is the instant of the transition that ends the gap (so
/// a day can start at 00:30, 01:00, …); when midnight occurs twice (a fall-back
/// straddling midnight) it is the *earlier* of the two. Returns `None` only when
/// the one-day-earlier probe epoch is outside the representable instant range —
/// the caller turns that into a `RangeError`.
pub(crate) fn start_of_day_pub(tz: &str, epoch_days: i64) -> Option<i128> {
    start_of_day(tz, epoch_days)
}

fn start_of_day(tz: &str, epoch_days: i64) -> Option<i128> {
    let midnight = epoch_days as i128 * iso::NS_PER_DAY;
    let (poss, n) = possible_instants(tz, midnight);
    if n > 0 {
        // Single instant, or (fall-back overlap) the earlier of the two.
        return Some(poss[0]);
    }
    // Midnight is skipped (a DST gap starting at/around 00:00): the start of the
    // day is the transition that ends the gap.
    let day_before = midnight - iso::NS_PER_DAY;
    if !(iso::MIN_EPOCH_NS..=iso::MAX_EPOCH_NS).contains(&day_before) {
        return None;
    }
    let z = timezone_data::load(tz).ok()?;
    zone_next_transition(&z, day_before)
}

/// `GetNamedTimeZoneNextTransition`: the first offset transition strictly after
/// `epoch_ns` (in ns), combining the stored transition table with the POSIX
/// extend rule (so future transitions past the stored range are found). `None`
/// once the zone has no further transitions (e.g. a fixed zone with no DST rule).
fn zone_next_transition(zone: &timezone_data::Zone<'_>, epoch_ns: i128) -> Option<i128> {
    let secs = epoch_ns.div_euclid(iso::NS_PER_SEC) as i64;
    let max_secs = (iso::MAX_EPOCH_NS / iso::NS_PER_SEC) as i64;
    // `transitions_for_range` yields stored + POSIX-generated transitions in
    // chronological order; the first one strictly after the instant is the answer.
    zone.transitions_for_range(secs, max_secs)
        .map(|t| i128::from(t.when) * iso::NS_PER_SEC)
        .find(|&w| w > epoch_ns)
}

/// `GetNamedTimeZonePreviousTransition`: the last offset transition strictly before
/// `epoch_ns` (in ns). Stored transitions cover history; any POSIX-generated one is
/// at most ~1 year back, so a bounded look-back window suffices past the stored range.
fn zone_prev_transition(zone: &timezone_data::Zone<'_>, epoch_ns: i128) -> Option<i128> {
    let secs = epoch_ns.div_euclid(iso::NS_PER_SEC) as i64;
    let mut best: Option<i128> = zone
        .transitions()
        .map(|t| i128::from(t.when) * iso::NS_PER_SEC)
        .filter(|&w| w < epoch_ns)
        .max();
    let last_stored = zone
        .transitions()
        .last()
        .map(|t| t.when)
        .unwrap_or(i64::MIN);
    if secs > last_stored {
        let start = (secs - 800 * 86_400).max(last_stored.saturating_add(1));
        for t in zone.transitions_for_range(start, secs.saturating_add(2)) {
            let w = i128::from(t.when) * iso::NS_PER_SEC;
            if w < epoch_ns {
                best = Some(best.map_or(w, |b| b.max(w)));
            }
        }
    }
    best
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
/// `RoundTemporalInstant`: rounds an exact epoch-nanoseconds value to `inc` ns
/// using `RoundNumberToIncrementAsIfPositive` — the rounding-mode direction is
/// applied as if the value were positive, so a negative epoch does not flip the
/// meaning of `expand`/`trunc`/`halfExpand`/`halfTrunc`.
fn round_instant(epoch: i128, inc: i128, mode: RoundMode) -> i128 {
    let m = match mode {
        RoundMode::Expand => RoundMode::Ceil,
        RoundMode::Trunc => RoundMode::Floor,
        RoundMode::HalfExpand => RoundMode::HalfCeil,
        RoundMode::HalfTrunc => RoundMode::HalfFloor,
        other => other,
    };
    iso::round_to_increment(epoch, inc, m)
}

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
    /// Reads an `offset` field string → `(offset_ns, has_seconds)`. `has_seconds`
    /// is set when the source string carried sub-minute (seconds) precision, which
    /// forces `MATCH_EXACTLY` rather than minute-rounded matching.
    fn zdt_read_offset_field(&mut self, v: NanBox) -> Result<(i128, bool), ExecError> {
        let prim = self.coerce_primitive(v, "string")?;
        let Some(s) = prim
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|x| self.realm.string_value(x))
        else {
            return Err(self.type_error("offset must be a string"));
        };
        let off = parse_offset_value(&s).ok_or_else(|| self.zdt_range("invalid offset string"))?;
        Ok((off, offset_str_has_seconds(&s)))
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
                let start = start_of_day(&tz, iso_to_epoch_days(d));
                let next = start_of_day(&tz, iso_to_epoch_days(d) + 1);
                let (Some(start), Some(next)) = (start, next) else {
                    return Err(self.zdt_range("day boundary is out of range"));
                };
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
                let epoch = start_of_day(&tz, iso_to_epoch_days(d))
                    .filter(|e| valid_epoch(*e))
                    .ok_or_else(|| self.zdt_range("start of day is out of range"))?;
                Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
            }
            "getTimeZoneTransition" => self.zdt_get_transition(data, arg(0)),
            "equals" => {
                let (epoch, tz, cal) = self.resolve_zdt(arg(0))?;
                // `TimeZoneEquals` compares *primary* (canonical) identifiers, so a
                // link equals its canonical zone (e.g. Asia/Calcutta == Asia/Kolkata).
                let eq = epoch == data.epoch_ns
                    && tz_primary(&tz) == tz_primary(&self.zdt_tz(data))
                    && cal == data.calendar;
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

    fn zdt_disambiguation(&mut self, opts: Option<Handle>) -> Result<Disamb, ExecError> {
        Ok(
            match self
                .zdt_str_option(
                    opts,
                    "disambiguation",
                    &["compatible", "earlier", "later", "reject"],
                )?
                .as_deref()
            {
                Some("earlier") => Disamb::Earlier,
                Some("later") => Disamb::Later,
                Some("reject") => Disamb::Reject,
                _ => Disamb::Compatible,
            },
        )
    }

    /// `DisambiguatePossibleEpochNanoseconds`, throwing a `RangeError` for the
    /// `reject` conflict (a gap or overlap that `reject` refuses to resolve).
    fn disamb_epoch(&mut self, tz: &str, wall_ns: i128, d: Disamb) -> Result<i128, ExecError> {
        disambiguate(tz, wall_ns, d).map_err(|()| {
            self.zdt_range("wall-clock time is ambiguous or nonexistent (disambiguation: reject)")
        })
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
        let offset_field = match self.zdt_field(h, "offset")? {
            Some(v) => Some(self.zdt_read_offset_field(v)?),
            None => None,
        };
        let offset_ns = offset_field.map(|(o, _)| o);
        // A property-bag offset is always matched exactly (never minute-rounded):
        // `matchBehaviour` is only `match-minutes` for a minute-precision offset
        // parsed from an ISO string, per `ToTemporalZonedDateTime`.
        let match_minutes = false;
        let second = self.read_int_field(h, "second")?;
        let tz_val = self.zdt_field(h, "timeZone")?;
        let year = self.read_int_field(h, "year")?;

        // Options (read order: disambiguation, offset, overflow).
        let opts = self.zdt_options(options)?;
        let disamb = self.zdt_disambiguation(opts)?;
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
        let epoch = self.resolve_epoch(
            &tz,
            wall,
            offset_ns,
            false,
            offset_opt,
            disamb,
            match_minutes,
        )?;
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
        let offset_field = match self.zdt_field(h, "offset")? {
            Some(v) => Some(self.zdt_read_offset_field(v)?),
            None => None,
        };
        let offset_ns = offset_field.map(|(o, _)| o);
        // A property-bag offset is always matched exactly (never minute-rounded):
        // `matchBehaviour` is only `match-minutes` for a minute-precision offset
        // parsed from an ISO string, per `ToTemporalZonedDateTime`.
        let match_minutes = false;
        let second = self.read_int_field(h, "second")?;
        let tz_val = self.zdt_field(h, "timeZone")?;
        let year = self.read_int_field(h, "year")?;

        // Options (read order: disambiguation, offset, overflow).
        let opts = self.zdt_options(options)?;
        let disamb = self.zdt_disambiguation(opts)?;
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
        let epoch = self.resolve_epoch(
            &tz,
            wall,
            offset_ns,
            false,
            offset_opt,
            disamb,
            match_minutes,
        )?;
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
        let disamb = self.zdt_disambiguation(opts)?;
        let offset_opt = self.zdt_offset_option(opts)?;
        self.zdt_overflow(opts)?;

        let wall = iso_to_epoch_days(p.date) as i128 * iso::NS_PER_DAY + time_to_nanos(p.time);
        if !(iso::MIN_EPOCH_NS - iso::NS_PER_DAY..=iso::MAX_EPOCH_NS + iso::NS_PER_DAY)
            .contains(&wall)
        {
            return Err(self.zdt_range("date-time is outside the representable range"));
        }
        // A date-only string (no time component, wall offset behaviour) resolves to
        // `GetStartOfDay` — DST-gap aware — per `InterpretISODateTimeOffset`.
        if !p.has_time && !p.z && p.offset_ns.is_none() {
            let epoch = start_of_day(&tz, iso_to_epoch_days(p.date))
                .filter(|e| valid_epoch(*e))
                .ok_or_else(|| self.zdt_range("start of day is out of range"))?;
            return Ok((epoch, tz, calendar));
        }
        // offset_ns from a numeric offset; z handled separately. A minute-precision
        // offset string (no seconds) enables minute-rounded matching.
        let offset_ns = if p.z { None } else { p.offset_ns };
        let match_minutes = !dt_offset_subminute(s);
        let epoch =
            self.resolve_epoch(&tz, wall, offset_ns, p.z, offset_opt, disamb, match_minutes)?;
        Ok((epoch, tz, calendar))
    }

    /// `InterpretISODateTimeOffset`: turns a wall time + (optional) offset/`Z` into an
    /// exact instant, honouring both the `offset` reconciliation option and the
    /// `disambiguation` option (used whenever the offset is ignored or does not match).
    #[allow(clippy::too_many_arguments)]
    fn resolve_epoch(
        &mut self,
        tz: &str,
        wall_ns: i128,
        offset_ns: Option<i128>,
        has_z: bool,
        offset_opt: OffsetOpt,
        disamb: Disamb,
        match_minutes: bool,
    ) -> Result<i128, ExecError> {
        // `CheckISODaysRange` (per `InterpretISODateTimeOffset`): the effective ISO
        // date must have |epoch days| ≤ 10^8. Which date is checked depends on the
        // offset behaviour: `exact`/`use` check the offset-balanced date; the
        // `option` (prefer/reject) path checks the raw wall date; `wall`/`ignore`
        // skip it (the epoch validity check below suffices). This rejects e.g.
        // "-271821-04-19T…" (day −10^8−1) whose epoch can still land exactly on the
        // representable boundary.
        let day_of = |ns: i128| ns.div_euclid(iso::NS_PER_DAY);
        let checked_days: Option<i128> = if has_z {
            Some(day_of(wall_ns))
        } else if let Some(off) = offset_ns {
            match offset_opt {
                OffsetOpt::Use => Some(day_of(wall_ns - off)),
                OffsetOpt::Prefer | OffsetOpt::Reject => Some(day_of(wall_ns)),
                OffsetOpt::Ignore => None,
            }
        } else {
            None
        };
        if let Some(days) = checked_days
            && days.abs() > 100_000_000
        {
            return Err(
                self.zdt_range("date-time is outside the representable range of ZonedDateTime")
            );
        }

        let epoch = if has_z {
            wall_ns
        } else if let Some(off) = offset_ns {
            match offset_opt {
                // `use`: honour the given offset exactly, transitions notwithstanding.
                OffsetOpt::Use => wall_ns - off,
                // `ignore`: discard the offset and disambiguate the wall time.
                OffsetOpt::Ignore => self.disamb_epoch(tz, wall_ns, disamb)?,
                // `prefer` / `reject`: keep any possible instant whose implied offset
                // matches (to the minute only when the source was minute-precision).
                OffsetOpt::Prefer | OffsetOpt::Reject => {
                    match offset_match(tz, wall_ns, off, match_minutes) {
                        Some(epoch) => epoch,
                        None if offset_opt == OffsetOpt::Prefer => {
                            self.disamb_epoch(tz, wall_ns, disamb)?
                        }
                        None => return Err(self.zdt_range("offset does not match the time zone")),
                    }
                }
            }
        } else {
            self.disamb_epoch(tz, wall_ns, disamb)?
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
        let disamb = self.zdt_disambiguation(opts)?;
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
        let offset_ns = Some(offset_field.map(|(o, _)| o).unwrap_or(cur_offset));

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
        let epoch = self.resolve_epoch(&tz, wall, offset_ns, false, offset_opt, disamb, false)?;
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
        let disamb = self.zdt_disambiguation(opts)?;
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
        let offset_ns = Some(offset_field.map(|(o, _)| o).unwrap_or(cur_offset));
        let time = iso::regulate_iso_time(hour, minute, second, ms, us, ns, overflow)
            .ok_or_else(|| self.zdt_range("invalid ISO time"))?;

        let wall = iso_to_epoch_days(date) as i128 * iso::NS_PER_DAY + time_to_nanos(time);
        let epoch = self.resolve_epoch(tz, wall, offset_ns, false, offset_opt, disamb, false)?;
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
        // With no argument, `withPlainTime()` means the start of the day
        // (`GetStartOfDay`), which is DST-gap aware — not a plain midnight.
        let epoch = if arg.is_undefined() {
            start_of_day(&tz, iso_to_epoch_days(cd))
                .filter(|e| valid_epoch(*e))
                .ok_or_else(|| self.zdt_range("resulting instant is out of range"))?
        } else {
            let time = self.zdt_to_time(arg)?;
            let wall = iso_to_epoch_days(cd) as i128 * iso::NS_PER_DAY + time_to_nanos(time);
            let epoch = wall_to_epoch(&tz, wall);
            if !valid_epoch(epoch) {
                return Err(self.zdt_range("resulting instant is out of range"));
            }
            epoch
        };
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
        let (other_epoch, other_tz, other_cal) = self.resolve_zdt(other)?;
        let cal = data.calendar.clone();
        let tz = self.zdt_tz(data);
        // DifferenceTemporalZonedDateTime enforces CalendarEquals before reading the
        // options bag. The ISO fast path keeps its original calendar-agnostic
        // behaviour; a non-ISO receiver requires both operands to share a calendar.
        if !tcal::is_iso(&cal) && other_cal != cal {
            return Err(self
                .zdt_range("cannot compute the difference between dates of different calendars"));
        }
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

        // `DifferenceZonedDateTime` (a *date* largestUnit — year/month/week/day)
        // measures whole days against the zone's own day boundaries, so both
        // operands must share a time zone (`TimeZoneEquals` on primary identifiers).
        // A time-unit largestUnit (hour or finer) is a pure epoch difference and
        // needs no such check.
        if largest <= Unit::Day && tz_primary(&tz) != tz_primary(&other_tz) {
            return Err(self
                .zdt_range("cannot compute the difference between dates in different time zones"));
        }

        let mut dur = if largest == Unit::Day {
            // `DifferenceZonedDateTime` with largestUnit day: the day count follows the
            // time zone's own (variable-length) day boundaries, not a fixed 24 hours.
            self.zdt_diff_days(&tz, data.epoch_ns, other_epoch, smallest, increment, mode)?
        } else if largest >= Unit::Day {
            // Hour or finer: an exact nanosecond difference, with sign-aware rounding.
            let total = other_epoch - data.epoch_ns;
            let inc = unit_ns(smallest) * i128::from(increment.max(1));
            let rounded = round_signed(total, inc, mode);
            balance_datetime(rounded, largest)
        } else if matches!(smallest, Unit::Year | Unit::Month | Unit::Week | Unit::Day) {
            // Calendar-unit rounding relative to the receiver (NudgeToCalendarUnit at
            // the smallestUnit), then balanced up to the (coarser or equal) calendar
            // largestUnit — e.g. 1y 11m 24d rounded to months, expanded, is 2 years.
            self.zdt_round_calendar(
                &cal,
                &tz,
                data.epoch_ns,
                other_epoch,
                smallest,
                largest,
                increment,
                mode,
            )
        } else {
            // Calendar largestUnit (years/months/weeks) with a time smallestUnit —
            // DST-aware date diff plus the time remainder rounded at smallestUnit
            // (which may bubble a whole day up into the date part).
            self.zdt_calendar_diff(
                &cal,
                &tz,
                data.epoch_ns,
                other_epoch,
                largest,
                smallest,
                increment,
                mode,
            )
        };
        if negate {
            dur = negate_duration(dur);
        }
        // Duration fields are Numbers: quantize to float64-representable integers.
        let dur = iso::quantize_duration_fields(dur);
        let data2 = TemporalData {
            kind: TemporalKind::Duration,
            duration: dur,
            ..Default::default()
        };
        Ok(self.zdt_alloc(data2, TemporalKind::Duration))
    }

    /// `AddZonedDateTime` of a date-only duration: the instant reached by adding the
    /// calendar `y/m/w/d` to the wall-clock date of `ns` (keeping its wall time), then
    /// re-resolving via the default disambiguation. Falls back to `ns` on overflow.
    #[allow(clippy::too_many_arguments)]
    fn add_zdt_date_epoch(
        &self,
        cal: &str,
        tz: &str,
        ns: i128,
        y: i64,
        m: i64,
        w: i64,
        d: i64,
    ) -> i128 {
        // `AddZonedDateTime`: a zero date portion adds nothing to the wall date, so
        // the instant is unchanged (`AddInstant` of a zero time duration). Crucially
        // this does NOT re-resolve the wall time through disambiguation — which would
        // otherwise collapse a fall-back overlap onto the *earlier* instant and lose
        // which of the two same-wall-clock instants we started from.
        if y == 0 && m == 0 && w == 0 && d == 0 {
            return ns;
        }
        let (date, t) = local_of(tz, ns);
        let nd = if tcal::is_iso(cal) {
            iso::add_iso_date(date, y, m, w, d, Overflow::Constrain)
        } else {
            tcal::calendar_date_add(cal, date, y, m, w, d, Overflow::Constrain).ok()
        };
        match nd {
            Some(nd) => {
                let wall = iso_to_epoch_days(nd) as i128 * iso::NS_PER_DAY + time_to_nanos(t);
                wall_to_epoch(tz, wall)
            }
            None => ns,
        }
    }

    /// `DifferenceZonedDateTime` for a calendar largestUnit (year/month/week): the
    /// date part is the wall-clock calendar difference, but the time part is the exact
    /// instant gap left after adding that date part back (so DST offset shifts are
    /// reflected — a 24-hour remainder does not collapse to a day inside a 25-hour day).
    #[allow(clippy::too_many_arguments)]
    fn zdt_calendar_diff(
        &self,
        cal: &str,
        tz: &str,
        ns1: i128,
        ns2: i128,
        largest: Unit,
        smallest: Unit,
        increment: i64,
        mode: RoundMode,
    ) -> DurationFields {
        let from = local_of(tz, ns1);
        let to = local_of(tz, ns2);
        let mut d = datetime_diff(cal, from, to, largest);
        let date_epoch = self.add_zdt_date_epoch(
            cal,
            tz,
            ns1,
            d.years as i64,
            d.months as i64,
            d.weeks as i64,
            d.days as i64,
        );
        // `NudgeToDayOrTime`: round the exact sub-day time remainder at the (time)
        // smallestUnit. When it rounds up to a whole day, that day bubbles into the
        // date part (`BubbleRelativeDuration`) — e.g. …23:59:59.999999999 rounded up
        // to the microsecond becomes the next day, which can carry all the way to a
        // year. A nanosecond smallestUnit with increment 1 leaves this a no-op.
        let overall_sign = (ns2 - ns1).signum();
        let inc = unit_ns(smallest) * i128::from(increment.max(1));
        let rem_ns = ns2 - date_epoch;
        let rounded = round_signed(rem_ns, inc, mode);
        let next_date_epoch = self.add_zdt_date_epoch(
            cal,
            tz,
            ns1,
            d.years as i64,
            d.months as i64,
            d.weeks as i64,
            d.days as i64 + overall_sign as i64,
        );
        let day_len = (next_date_epoch - date_epoch).abs();
        // Carry only when *rounding* pushes a genuinely sub-day remainder up to a
        // full day. When the unrounded remainder already equals the (possibly
        // DST-shortened) day length, `end` sits exactly on the next day boundary —
        // that is the "pick the smaller of two possible durations" case, and the
        // difference must stay in days/time rather than balancing up a whole day
        // (which could wrongly bubble into a month).
        let leftover = if day_len > 0 && rem_ns.abs() < day_len && rounded.abs() >= day_len {
            // The time rounded up to (at least) a full day: carry it into the date
            // part and re-difference so the extra day bubbles up through the calendar.
            let carried_days = d.days as i64 + overall_sign as i64;
            let end_date = if tcal::is_iso(cal) {
                iso::add_iso_date(
                    from.0,
                    d.years as i64,
                    d.months as i64,
                    d.weeks as i64,
                    carried_days,
                    Overflow::Constrain,
                )
            } else {
                tcal::calendar_date_add(
                    cal,
                    from.0,
                    d.years as i64,
                    d.months as i64,
                    d.weeks as i64,
                    carried_days,
                    Overflow::Constrain,
                )
                .ok()
            };
            if let Some(ed2) = end_date {
                let (by, bm, bw, bd) = if tcal::is_iso(cal) {
                    iso::difference_iso_date(from.0, ed2, largest)
                } else {
                    let p = tcal::calendar_date_until(cal, from.0, ed2, largest);
                    (p.years, p.months, p.weeks, p.days)
                };
                d.years = i128::from(by);
                d.months = i128::from(bm);
                d.weeks = i128::from(bw);
                d.days = i128::from(bd);
            } else {
                d.days += overall_sign;
            }
            rounded - overall_sign * day_len
        } else {
            rounded
        };
        let rem = iso::balance_time_duration(leftover, Unit::Hour);
        d.hours = rem.hours;
        d.minutes = rem.minutes;
        d.seconds = rem.seconds;
        d.milliseconds = rem.milliseconds;
        d.microseconds = rem.microseconds;
        d.nanoseconds = rem.nanoseconds;
        d
    }

    /// `DifferenceZonedDateTime` restricted to largestUnit day: the whole-day count
    /// between two instants measured against the time zone's own day boundaries (a
    /// day may be 23/24/25 hours across a DST transition), plus the balanced time
    /// remainder (rounded to `smallest`/`increment`). Mirrors `AddZonedDateTime`: the
    /// day part is added as wall-clock days, and the leftover is the exact instant gap.
    #[allow(clippy::too_many_arguments)]
    fn zdt_diff_days(
        &mut self,
        tz: &str,
        ns1: i128,
        ns2: i128,
        smallest: Unit,
        increment: i64,
        mode: RoundMode,
    ) -> Result<DurationFields, ExecError> {
        let sign = (ns2 - ns1).signum();
        if sign == 0 {
            return Ok(DurationFields::default());
        }
        let (sd, st) = local_of(tz, ns1);
        let ed = local_of(tz, ns2).0;
        let start_days = iso_to_epoch_days(sd);
        let start_time_ns = time_to_nanos(st);
        // The instant of (start wall date + `n` days) at the start wall time.
        let add_days = |n: i128| -> i128 {
            let nd = epoch_days_to_iso(start_days + n as i64);
            let wall = iso_to_epoch_days(nd) as i128 * iso::NS_PER_DAY + start_time_ns;
            wall_to_epoch(tz, wall)
        };
        let beyond = |e: i128| -> bool { if sign > 0 { e > ns2 } else { e < ns2 } };
        // Initial guess from the wall-clock date difference, then correct by at most a
        // day or two (a DST day differs from a calendar day by well under 24 hours).
        let mut days = (iso_to_epoch_days(ed) - start_days) as i128;
        for _ in 0..4 {
            if beyond(add_days(days + sign)) {
                break;
            }
            days += sign;
        }
        for _ in 0..4 {
            if beyond(add_days(days)) {
                days -= sign;
            } else {
                break;
            }
        }
        // The leftover exact instant gap (same sign as the overall difference, and no
        // larger than the crossed day) is the sub-day part.
        let remainder = ns2 - add_days(days);

        if smallest == Unit::Day {
            // Round the day count itself, measuring the fraction against the actual
            // (DST-aware) length of the day being crossed.
            let den = (add_days(days + sign) - add_days(days)).abs().max(1);
            let inc = i128::from(increment.max(1));
            let scaled = days * den + sign * remainder.abs();
            // The rounding brackets `scaled` between two multiples of `inc` days; the
            // far bound's instant is materialized (`AddZonedDateTime`), so if adding
            // that many days overflows the representable range it is a RangeError —
            // even when rounding would ultimately land on the nearer bound.
            let floor_units = scaled.abs() / (inc * den);
            let far_days = sign * (floor_units + 1) * inc;
            if !valid_epoch(add_days(far_days)) {
                return Err(self.zdt_range("rounded day boundary is out of range"));
            }
            let rounded = round_signed(scaled, inc * den, mode) / den;
            return Ok(DurationFields {
                days: rounded,
                ..Default::default()
            });
        }

        // largestUnit day with a sub-day smallestUnit: keep the whole days and round
        // the time remainder.
        let inc = unit_ns(smallest) * i128::from(increment.max(1));
        let mut rounded = round_signed(remainder, inc, mode);
        // If rounding the remainder reaches (or passes) the full length of the day
        // being crossed, it bubbles up into an extra day (BubbleRelativeDuration):
        // e.g. 23h59m rounded to the nearest hour becomes a whole day.
        let day_len = (add_days(days + sign) - add_days(days)).abs();
        if day_len > 0 && rounded.abs() >= day_len {
            days += sign;
            rounded -= sign * day_len;
        }
        let mut dur = iso::balance_time_duration(rounded, Unit::Hour);
        dur.days = days;
        Ok(dur)
    }

    /// `NudgeToCalendarUnit` for a calendar `unit` (the smallestUnit), rounding the
    /// difference from `start_epoch` to `end_epoch` to `increment` under `mode`,
    /// with the fraction measured against the instant span of one `unit` step in the
    /// zone (so it is DST-aware). The rounded count is then re-expressed up to
    /// `largest` (`BalanceDateDurationRelative`) — e.g. rounding to months and
    /// balancing to years turns 24 months into 2 years.
    #[allow(clippy::too_many_arguments)]
    fn zdt_round_calendar(
        &self,
        cal: &str,
        tz: &str,
        start_epoch: i128,
        end_epoch: i128,
        unit: Unit,
        largest: Unit,
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
        // Full unrounded date difference at `largest` (calendar-aware for non-ISO).
        // NudgeToCalendarUnit keeps the units coarser than `unit` and rounds only the
        // `unit` component — it does NOT re-express the whole span in `unit`s (which
        // would corrupt e.g. a whole month rounded to weeks).
        let (dy, dm, dw, dd) = if tcal::is_iso(cal) {
            iso::difference_iso_date(sd, ed, largest)
        } else {
            let p = tcal::calendar_date_until(cal, sd, ed, largest);
            (p.years, p.months, p.weeks, p.days)
        };
        // The instant of (start date + a date duration) at the start wall time.
        let add_date = |y: i64, m: i64, w: i64, d: i64| -> i128 {
            let nd = if tcal::is_iso(cal) {
                iso::add_iso_date(sd, y, m, w, d, Overflow::Constrain).unwrap_or(sd)
            } else {
                tcal::calendar_date_add(cal, sd, y, m, w, d, Overflow::Constrain).unwrap_or(sd)
            };
            let wall = iso_to_epoch_days(nd) as i128 * iso::NS_PER_DAY + time_to_nanos(st);
            wall_to_epoch(tz, wall)
        };
        // A date duration keeping the units coarser than `unit`, `unit` = `v`, finer = 0.
        let with_comp = |v: i64| -> (i64, i64, i64, i64) {
            match unit {
                Unit::Year => (v, 0, 0, 0),
                Unit::Month => (dy, v, 0, 0),
                Unit::Week => (dy, dm, v, 0),
                _ => (dy, dm, dw, v),
            }
        };
        let comp_epoch = |v: i64| -> i128 {
            let (y, m, w, d) = with_comp(v);
            add_date(y, m, w, d)
        };
        let beyond = |e: i128| -> bool {
            if sign > 0 {
                e > end_epoch
            } else {
                e < end_epoch
            }
        };
        // The current value of the `unit` component, then corrected so that its own
        // instant is on the near side of `end_epoch` and the next step overshoots
        // (time-of-day / DST can shift the wall-date guess by a step).
        let mut r1 = match unit {
            Unit::Year => dy,
            Unit::Month => dm,
            Unit::Week => dw,
            _ => dd,
        };
        for _ in 0..8 {
            if beyond(comp_epoch(r1 + sign)) {
                break;
            }
            r1 += sign;
        }
        for _ in 0..8 {
            if beyond(comp_epoch(r1)) {
                r1 -= sign;
            } else {
                break;
            }
        }
        let epoch1 = comp_epoch(r1);
        let rounded_v = if epoch1 == end_epoch {
            r1
        } else {
            let epoch2 = comp_epoch(r1 + sign);
            let den = (epoch2 - epoch1).abs().max(1);
            let num = (end_epoch - epoch1).abs();
            let x = i128::from(r1) * den + i128::from(sign) * num;
            let rounded = round_signed(x, i128::from(increment.max(1)) * den, mode);
            (rounded / den) as i64
        };
        // `BubbleRelativeDuration`: the nudged date duration (coarser units + rounded
        // `unit` component) is re-expressed up to `largest` by adding it to the start
        // date and re-differencing, carrying any full coarser unit (e.g. 12 months →
        // 1 year, or a rounded-up week that completes a month).
        let (ny, nm, nw, nd2) = with_comp(rounded_v);
        let end_date = if tcal::is_iso(cal) {
            iso::add_iso_date(sd, ny, nm, nw, nd2, Overflow::Constrain)
        } else {
            tcal::calendar_date_add(cal, sd, ny, nm, nw, nd2, Overflow::Constrain).ok()
        };
        let (by, bm, bw, bd) = match end_date {
            Some(ed2) if tcal::is_iso(cal) => iso::difference_iso_date(sd, ed2, largest),
            Some(ed2) => {
                let p = tcal::calendar_date_until(cal, sd, ed2, largest);
                (p.years, p.months, p.weeks, p.days)
            }
            None => (ny, nm, nw, nd2),
        };
        dur.years = i128::from(by);
        dur.months = i128::from(bm);
        dur.weeks = i128::from(bw);
        dur.days = i128::from(bd);
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
            let start = start_of_day(&tz, iso_to_epoch_days(d));
            let next = start_of_day(&tz, iso_to_epoch_days(d) + 1);
            let (Some(start), Some(next)) = (start, next) else {
                return Err(self.zdt_range("day boundary is out of range"));
            };
            if !valid_epoch(start) || !valid_epoch(next) {
                return Err(self.zdt_range("day boundary is out of range"));
            }
            // When a wall date "starts twice" (a fall-back straddling midnight), the
            // instant can be at or past the *next* day's start; the spec clamps it to
            // one ns before, so it still rounds within its own (long) day.
            let this_ns = if data.epoch_ns >= next {
                next - 1
            } else {
                data.epoch_ns
            };
            let day_len = next - start;
            let progress = this_ns - start;
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
        let best = if next {
            zone_next_transition(&zone, data.epoch_ns)
        } else {
            zone_prev_transition(&zone, data.epoch_ns)
        };
        match best {
            Some(epoch) if valid_epoch(epoch) => {
                Ok(self.make_zdt_cal(epoch, tz, data.calendar.clone()))
            }
            _ => Ok(NanBox::null()),
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
        // `RoundTemporalInstant` rounds the exact epoch, then the offset and wall
        // time are re-derived from the rounded instant. This is correct when the
        // rounded time lands in a DST gap (e.g. rounding 01:59:59.999… up to 02:00
        // in a spring-forward zone yields the post-transition 03:00 at the new
        // offset), which rounding the wall time in isolation cannot express.
        let rounded_epoch = round_instant(data.epoch_ns, inc_ns, mode);
        if !valid_epoch(rounded_epoch) {
            return Err(self.zdt_range("rounded instant is out of range"));
        }
        let offset_ns = tz_offset_at(&tz, rounded_epoch);
        let (date, time) = local_of(&tz, rounded_epoch);

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
/// Whether a bare offset string (e.g. `-11:20:00`, `+0530`, `-11:20`) carries a
/// seconds (sub-minute) component. Such offsets force `MATCH_EXACTLY`.
fn offset_str_has_seconds(s: &str) -> bool {
    let s = s.trim();
    let body = s.strip_prefix(['+', '-', '\u{2212}']).unwrap_or(s);
    if body.contains('.') || body.contains(',') {
        return true;
    }
    let colons = body.matches(':').count();
    if colons >= 2 {
        return true;
    }
    if colons == 0 {
        // Compact form: HHMMSS (>4 significant digits) has seconds.
        let digits = body.chars().take_while(char::is_ascii_digit).count();
        return digits > 4;
    }
    false
}

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
    /// Whether the string carried an explicit time component. A date-only string
    /// (`has_time == false`) resolves to the start of the day rather than midnight.
    has_time: bool,
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
    let mut has_time = false;
    if p.eat(b'T') || p.eat(b't') || p.eat(b' ') {
        has_time = true;
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
        has_time,
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

#[cfg(test)]
mod dst_tests {
    use super::*;

    /// Wall-clock date-time as a UTC epoch count of nanoseconds (an ISO date-time
    /// treated as if UTC), matching the input to `possible_instants`/`disambiguate`.
    fn wall(y: i32, mo: u8, d: u8, h: i128, mi: i128) -> i128 {
        iso_to_epoch_days(IsoDate {
            year: y,
            month: mo,
            day: d,
        }) as i128
            * iso::NS_PER_DAY
            + h * iso::NS_PER_HOUR
            + mi * iso::NS_PER_MINUTE
    }

    #[test]
    fn spring_forward_gap() {
        let tz = "America/New_York";
        // 2020-03-08 02:30 does not exist (clocks jump 02:00 -> 03:00 EDT).
        let w = wall(2020, 3, 8, 2, 30);
        let (_, n) = possible_instants(tz, w);
        assert_eq!(n, 0, "spring-forward wall time has no valid instant");

        // compatible / later resolve forward to 03:30 (-04:00); earlier back to 01:30.
        let later = disambiguate(tz, w, Disamb::Compatible).unwrap();
        let (ld, lt) = local_of(tz, later);
        assert_eq!((ld.month, ld.day, lt.hour, lt.minute), (3, 8, 3, 30));
        assert_eq!(tz_offset_at(tz, later), -4 * iso::NS_PER_HOUR);

        let earlier = disambiguate(tz, w, Disamb::Earlier).unwrap();
        let (_, et) = local_of(tz, earlier);
        assert_eq!((et.hour, et.minute), (1, 30));
        assert_eq!(tz_offset_at(tz, earlier), -5 * iso::NS_PER_HOUR);

        assert!(disambiguate(tz, w, Disamb::Reject).is_err());
    }

    #[test]
    fn fall_back_overlap() {
        let tz = "America/New_York";
        // 2020-11-01 01:30 occurs twice (clocks fall 02:00 -> 01:00).
        let w = wall(2020, 11, 1, 1, 30);
        let (poss, n) = possible_instants(tz, w);
        assert_eq!(n, 2, "fall-back wall time has two valid instants");
        assert_eq!(poss[1] - poss[0], iso::NS_PER_HOUR, "one hour apart");

        // earlier = -04:00 (EDT), later = -05:00 (EST); compatible = earlier.
        let earlier = disambiguate(tz, w, Disamb::Earlier).unwrap();
        let later = disambiguate(tz, w, Disamb::Later).unwrap();
        let compatible = disambiguate(tz, w, Disamb::Compatible).unwrap();
        assert_eq!(earlier, poss[0]);
        assert_eq!(later, poss[1]);
        assert_eq!(compatible, poss[0]);
        assert_eq!(tz_offset_at(tz, earlier), -4 * iso::NS_PER_HOUR);
        assert_eq!(tz_offset_at(tz, later), -5 * iso::NS_PER_HOUR);
        // Both render as the same wall clock.
        assert_eq!(local_of(tz, earlier).1.hour, 1);
        assert_eq!(local_of(tz, later).1.hour, 1);

        assert!(disambiguate(tz, w, Disamb::Reject).is_err());
    }

    #[test]
    fn ordinary_wall_time_unique() {
        let tz = "America/New_York";
        let w = wall(2020, 6, 15, 12, 0);
        let (_, n) = possible_instants(tz, w);
        assert_eq!(n, 1);
        // 12:00 EDT (-04:00) -> 16:00 UTC.
        let e = wall_to_epoch(tz, w);
        assert_eq!(tz_offset_at(tz, e), -4 * iso::NS_PER_HOUR);
        assert_eq!(local_of(tz, e).1.hour, 12);
    }

    #[test]
    fn fixed_offset_zone_has_single_instant() {
        let (_, n) = possible_instants("+05:30", wall(2020, 3, 8, 2, 30));
        assert_eq!(n, 1);
        assert_eq!(
            wall_to_epoch("+05:30", wall(2020, 1, 1, 0, 0)),
            wall(2020, 1, 1, 0, 0) - (5 * iso::NS_PER_HOUR + 30 * iso::NS_PER_MINUTE)
        );
    }

    #[test]
    fn transitions_next_and_previous() {
        let zone = timezone_data::load("America/New_York").unwrap();
        // A summer instant: previous transition is the spring-forward, next is fall-back.
        let summer = wall_to_epoch("America/New_York", wall(2020, 6, 15, 12, 0));
        let next = zone_next_transition(&zone, summer).expect("a next transition");
        let prev = zone_prev_transition(&zone, summer).expect("a previous transition");
        assert!(next > summer && prev < summer);
        // Spring-forward: offset just before is EST(-5), just after EDT(-4).
        assert_eq!(
            tz_offset_at("America/New_York", prev - 1),
            -5 * iso::NS_PER_HOUR
        );
        assert_eq!(
            tz_offset_at("America/New_York", prev),
            -4 * iso::NS_PER_HOUR
        );
        // Fall-back: offset flips from EDT(-4) back to EST(-5).
        assert_eq!(
            tz_offset_at("America/New_York", next - 1),
            -4 * iso::NS_PER_HOUR
        );
        assert_eq!(
            tz_offset_at("America/New_York", next),
            -5 * iso::NS_PER_HOUR
        );
    }

    #[test]
    fn transitions_reach_future_via_posix_rule() {
        // Well past the stored transition table (~2037): the POSIX extend rule must
        // still yield the yearly US DST transitions.
        let zone = timezone_data::load("America/New_York").unwrap();
        let far = wall_to_epoch("America/New_York", wall(2099, 6, 15, 12, 0));
        assert!(zone_next_transition(&zone, far).is_some());
        assert!(zone_prev_transition(&zone, far).is_some());
    }

    #[test]
    fn start_of_day_across_midnight_gap() {
        // America/Toronto 1919-03-31: the day starts at 00:30 because a 1-hour
        // spring-forward at 1919-03-30T23:30 skips 23:30..00:29 across midnight.
        let tz = "America/Toronto";
        let day = iso_to_epoch_days(IsoDate {
            year: 1919,
            month: 3,
            day: 31,
        });
        let epoch = start_of_day(tz, day).expect("start of day exists");
        let (d, t) = local_of(tz, epoch);
        assert_eq!((d.month, d.day, t.hour, t.minute), (3, 31, 0, 30));

        // A day whose midnight is unambiguous starts at exactly 00:00.
        let normal = iso_to_epoch_days(IsoDate {
            year: 2020,
            month: 6,
            day: 15,
        });
        let (_, nt) = local_of(tz, start_of_day(tz, normal).unwrap());
        assert_eq!((nt.hour, nt.minute), (0, 0));
    }

    #[test]
    fn start_of_day_fall_back_returns_earlier() {
        // Antarctica/Casey 2010-03-05 midnight occurs twice (a 3-hour fall-back
        // straddling midnight); GetStartOfDay returns the earlier (+11) instant.
        let tz = "Antarctica/Casey";
        let day = iso_to_epoch_days(IsoDate {
            year: 2010,
            month: 3,
            day: 5,
        });
        let epoch = start_of_day(tz, day).unwrap();
        // Earlier occurrence is at offset +11:00.
        assert_eq!(tz_offset_at(tz, epoch), 11 * iso::NS_PER_HOUR);
    }

    #[test]
    fn tz_primary_canonicalizes_links_and_utc_family() {
        // IANA "backward" links resolve to their canonical zone.
        assert_eq!(tz_primary("Asia/Calcutta"), "Asia/Kolkata");
        assert_eq!(tz_primary("Asia/Ulan_Bator"), "Asia/Ulaanbaatar");
        assert_eq!(tz_primary("America/Atka"), "America/Adak");
        assert_eq!(tz_primary("Europe/Nicosia"), "Asia/Nicosia");
        assert_eq!(tz_primary("Australia/Canberra"), "Australia/Sydney");
        // The whole UTC/GMT-zero family shares the primary "UTC".
        for id in ["UTC", "Etc/UTC", "Etc/GMT", "GMT", "Greenwich", "Zulu"] {
            assert_eq!(tz_primary(id), "UTC", "{id}");
        }
        // A canonical zone and an offset id are returned unchanged.
        assert_eq!(tz_primary("Asia/Kolkata"), "Asia/Kolkata");
        assert_eq!(tz_primary("+05:30"), "+05:30");
        // Distinct zones keep distinct primaries.
        assert_ne!(tz_primary("Asia/Colombo"), tz_primary("Asia/Kolkata"));
    }

    // The following exercise the reusable helpers that back ECMA-402
    // `Intl.DateTimeFormat` time-zone resolution and offset application (see
    // `intl_fmt::Interp::dtf_resolve_time_zone` / `dtf_zone_offset_ms`).

    #[test]
    fn offset_identifier_canonicalization_for_dtf() {
        // `IsTimeZoneOffsetString` accepts `±HH`, `±HHMM`, `±HH:MM` (minute
        // precision) and normalizes to `±HH:MM`.
        assert_eq!(parse_offset_id("+03").unwrap().1, "+03:00");
        assert_eq!(parse_offset_id("+0300").unwrap().1, "+03:00");
        assert_eq!(parse_offset_id("+01:03").unwrap().1, "+01:03");
        assert_eq!(parse_offset_id("-14").unwrap().1, "-14:00");
        assert_eq!(parse_offset_id("-2100").unwrap().1, "-21:00");
        // Negative zero collapses to `+00:00`.
        assert_eq!(parse_offset_id("-00").unwrap().1, "+00:00");
        assert_eq!(parse_offset_id("-00:00").unwrap().1, "+00:00");
        // Malformed offsets are rejected (→ RangeError in DTF).
        for bad in [
            "+3",
            "+24",
            "+23:0",
            "-10.50",
            "-1:10",
            "+15:59:00",
            "+13234",
            "-22230",
        ] {
            assert!(parse_offset_id(bad).is_none(), "{bad} should be rejected");
        }
        // A U+2212 MINUS SIGN is not an ASCII sign, so it never parses.
        assert!(parse_offset_id("\u{2212}05").is_none());
    }

    #[test]
    fn named_zone_resolution_is_case_insensitive_and_preserves_links() {
        // ASCII-case-insensitive matching returns the correctly-cased [[Identifier]].
        assert_eq!(
            resolve_named("africa/abidjan").as_deref(),
            Some("Africa/Abidjan")
        );
        assert_eq!(
            resolve_named("AMERICA/NEW_YORK").as_deref(),
            Some("America/New_York")
        );
        assert_eq!(resolve_named("utc").as_deref(), Some("UTC"));
        // Link identifiers are preserved, NOT canonicalized to their primary
        // (matches `timezone-not-canonicalized.js`).
        assert_eq!(
            resolve_named("Asia/Calcutta").as_deref(),
            Some("Asia/Calcutta")
        );
        // Legacy non-IANA abbreviations are not valid names.
        for bad in ["ACT", "PST", "EST5EDT-junk", "MEZ", "invalid"] {
            assert_eq!(resolve_named(bad), None, "{bad} should not resolve");
        }
    }

    #[test]
    fn named_zone_offset_is_dst_aware() {
        let tz = "America/New_York";
        // 2021-08-04T00:00Z is summer → Eastern Daylight Time = UTC-4.
        let summer = wall(2021, 8, 4, 0, 0);
        assert_eq!(tz_offset_at(tz, summer), -4 * iso::NS_PER_HOUR);
        // 2021-01-04T00:00Z is winter → Eastern Standard Time = UTC-5.
        let winter = wall(2021, 1, 4, 0, 0);
        assert_eq!(tz_offset_at(tz, winter), -5 * iso::NS_PER_HOUR);
        // Asia/Kolkata is a fixed +05:30 all year.
        assert_eq!(
            tz_offset_at("Asia/Kolkata", summer),
            5 * iso::NS_PER_HOUR + 30 * iso::NS_PER_MINUTE
        );
        // A bare offset identifier applies its fixed offset regardless of instant.
        assert_eq!(tz_offset_at("+03:00", winter), 3 * iso::NS_PER_HOUR);
        assert_eq!(
            tz_offset_at("-0509", summer),
            -(5 * iso::NS_PER_HOUR + 9 * iso::NS_PER_MINUTE)
        );
    }
}
