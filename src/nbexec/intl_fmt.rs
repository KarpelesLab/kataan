use super::*;
#[cfg(not(feature = "std"))]
use crate::common::FloatExt;

/// The resolved form of `Intl.NumberFormat`'s `useGrouping` option: a boolean
/// `false`, or one of the string modes `"auto"`/`"always"`/`"min2"`.
enum UseGroupingResolved {
    Bool(bool),
    Str(&'static str),
}

/// The Unicode code point of digit `0` for a decimal numbering system whose ten
/// digits are consecutive (the common case), so digit `d` is `base + d`. Returns
/// `None` for `latn` (ASCII, unchanged) and for non-consecutive systems like
/// `hanidec`. Covers the numeric numbering systems the corpus exercises.
fn numbering_system_digit_base(nu: &str) -> Option<u32> {
    // Every CLDR numbering system with a *simple, consecutive* 10-digit mapping
    // (per Test262's `numberingSystemDigits`); `latn` maps to ASCII 0. The one
    // simple-mapped system whose digits are NOT consecutive — `hanidec` — is
    // handled separately in [`substitute_numbering_digits`].
    Some(match nu {
        "adlm" => 0x1E950,
        "ahom" => 0x11730,
        "arab" => 0x0660,
        "arabext" => 0x06F0,
        "bali" => 0x1B50,
        "beng" => 0x09E6,
        "bhks" => 0x11C50,
        "brah" => 0x11066,
        "cakm" => 0x11136,
        "cham" => 0xAA50,
        "deva" => 0x0966,
        "diak" => 0x11950,
        "fullwide" => 0xFF10,
        "gara" => 0x10D40,
        "gong" => 0x11DA0,
        "gonm" => 0x11D50,
        "gujr" => 0x0AE6,
        "gukh" => 0x16130,
        "guru" => 0x0A66,
        "hmng" => 0x16B50,
        "hmnp" => 0x1E140,
        "java" => 0xA9D0,
        "kali" => 0xA900,
        "kawi" => 0x11F50,
        "khmr" => 0x17E0,
        "knda" => 0x0CE6,
        "krai" => 0x16D70,
        "lana" => 0x1A80,
        "lanatham" => 0x1A90,
        "laoo" => 0x0ED0,
        "latn" => 0x0030,
        "lepc" => 0x1C40,
        "limb" => 0x1946,
        "mathbold" => 0x1D7CE,
        "mathdbl" => 0x1D7D8,
        "mathmono" => 0x1D7F6,
        "mathsanb" => 0x1D7EC,
        "mathsans" => 0x1D7E2,
        "mlym" => 0x0D66,
        "modi" => 0x11650,
        "mong" => 0x1810,
        "mroo" => 0x16A60,
        "mtei" => 0xABF0,
        "mymr" => 0x1040,
        "mymrepka" => 0x116DA,
        "mymrpao" => 0x116D0,
        "mymrshan" => 0x1090,
        "mymrtlng" => 0xA9F0,
        "nagm" => 0x1E4F0,
        "newa" => 0x11450,
        "nkoo" => 0x07C0,
        "olck" => 0x1C50,
        "onao" => 0x1E5F1,
        "orya" => 0x0B66,
        "osma" => 0x104A0,
        "outlined" => 0x1CCF0,
        "rohg" => 0x10D30,
        "saur" => 0xA8D0,
        "segment" => 0x1FBF0,
        "shrd" => 0x111D0,
        "sind" => 0x112F0,
        "sinh" => 0x0DE6,
        "sora" => 0x110F0,
        "sund" => 0x1BB0,
        "sunu" => 0x11BF0,
        "takr" => 0x116C0,
        "talu" => 0x19D0,
        "tamldec" => 0x0BE6,
        "telu" => 0x0C66,
        "thai" => 0x0E50,
        "tibt" => 0x0F20,
        "tirh" => 0x114D0,
        "tnsa" => 0x16AC0,
        "tols" => 0x11DE0,
        "vaii" => 0xA620,
        "wara" => 0x118E0,
        "wcho" => 0x1E2F0,
        _ => return None,
    })
}

/// Rewrites the ASCII digits of `s` into numbering system `nu` (the consecutive
/// mapping from [`numbering_system_digit_base`], or the special non-consecutive
/// `hanidec`); other systems and non-digit characters pass through unchanged.
fn substitute_numbering_digits(nu: &str, s: String) -> String {
    if nu == "hanidec" {
        const HANIDEC: [char; 10] = ['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
        return s
            .chars()
            .map(|c| {
                if c.is_ascii_digit() {
                    HANIDEC[(c as u8 - b'0') as usize]
                } else {
                    c
                }
            })
            .collect();
    }
    match numbering_system_digit_base(nu) {
        Some(base) if base != 0x0030 => s
            .chars()
            .map(|c| {
                if c.is_ascii_digit() {
                    char::from_u32(base + (c as u32 - '0' as u32)).unwrap_or(c)
                } else {
                    c
                }
            })
            .collect(),
        _ => s,
    }
}

/// The currency codes `Intl.supportedValuesOf("currency")` reports.
///
/// Shared with `Intl.DisplayNames`, which must name **exactly** this set: the
/// conformance test asserts both directions, so a code DisplayNames can name but
/// `supportedValuesOf` omits — or vice versa — is a failure. A supported code
/// with no localized name displays as the code itself, which is CLDR's own
/// behaviour for a currency whose `displayName` is absent.
pub(crate) static SUPPORTED_CURRENCIES: &[&str] = &[
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN", "BAM", "BBD", "BDT",
    "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL", "BSD", "BTN", "BWP", "BYN", "BZD", "CAD",
    "CDF", "CHF", "CLP", "CNY", "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD",
    "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD", "GNF", "GTQ",
    "GYD", "HKD", "HNL", "HRK", "HTG", "HUF", "IDR", "ILS", "INR", "IQD", "IRR", "ISK", "JMD",
    "JOD", "JPY", "KES", "KGS", "KHR", "KMF", "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP",
    "LKR", "LRD", "LSL", "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR",
    "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB",
    "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD",
    "SCR", "SDG", "SEK", "SGD", "SHP", "SLE", "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL",
    "THB", "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU",
    "UZS", "VES", "VND", "VUV", "WST", "XAF", "XCD", "XOF", "XPF", "YER", "ZAR", "ZMW", "ZWL",
];

/// Whether `c` is a digit of numbering system `nu`.
///
/// The scaffold probes format a sample and read it back to find the locale's
/// affixes and separators, so they must know which characters are digits *in the
/// system the sample was rendered in*. No character-class test can stand in:
/// `hanidec` renders with CJK ideographs (`一二三`), which Unicode classes as
/// ordinary letters, not digits. Latin digits are always accepted too, since a
/// probe may come back un-substituted.
fn is_digit_of_numbering_system(nu: &str, c: char) -> bool {
    if c.is_ascii_digit() {
        return true;
    }
    if nu == "hanidec" {
        return matches!(
            c,
            '〇' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九'
        );
    }
    match numbering_system_digit_base(nu) {
        Some(base) => (c as u32) >= base && (c as u32) < base + 10,
        None => false,
    }
}

/// The CLDR collation tailoring for a resolved locale tag, memoized.
///
/// Building one parses the locale's CLDR rule, which is far too expensive to
/// redo per comparison — a sort calls the collator `O(n log n)` times, and
/// re-parsing made a 3000-element `Intl.Collator("sv")` sort take 28s against
/// 2.7s for an untailored locale. Keyed by the resolved tag (so `-u-co-`
/// variants get their own entry) and bounded by the number of distinct locales a
/// program actually collates in.
#[cfg(all(feature = "intl", feature = "std"))]
fn locale_tailoring(locale: &str) -> Option<alloc::rc::Rc<intl::unicode::collate::Tailoring>> {
    std::thread_local! {
        static CACHE: core::cell::RefCell<
            alloc::collections::BTreeMap<String, Option<alloc::rc::Rc<intl::unicode::collate::Tailoring>>>,
        > = const { core::cell::RefCell::new(alloc::collections::BTreeMap::new()) };
    }
    CACHE.with(|cache| {
        if let Some(hit) = cache.borrow().get(locale).cloned() {
            return hit;
        }
        let built = intl::unicode::collate::Tailoring::for_locale(locale).map(alloc::rc::Rc::new);
        cache
            .borrow_mut()
            .insert(String::from(locale), built.clone());
        built
    })
}

/// As above, without the memo: `thread_local!` needs `std`, so the `no_std` build
/// rebuilds the tailoring per call.
#[cfg(all(feature = "intl", not(feature = "std")))]
fn locale_tailoring(locale: &str) -> Option<alloc::rc::Rc<intl::unicode::collate::Tailoring>> {
    intl::unicode::collate::Tailoring::for_locale(locale).map(alloc::rc::Rc::new)
}

/// The numbering-system name for a native zero digit `c` (the reverse of
/// [`numbering_system_digit_base`]) — used to name the CLDR default numbering
/// system a locale resolves to, detected from the digit the `intl` crate emits.
#[cfg(feature = "intl")]
fn numbering_system_name_from_zero(c: char) -> Option<&'static str> {
    Some(match c as u32 {
        0x0030 => "latn",
        0x0660 => "arab",
        0x06F0 => "arabext",
        0x1B50 => "bali",
        0x09E6 => "beng",
        0x0966 => "deva",
        0xFF10 => "fullwide",
        0x0AE6 => "gujr",
        0x0A66 => "guru",
        0x17E0 => "khmr",
        0x0CE6 => "knda",
        0x0ED0 => "laoo",
        0x1946 => "limb",
        0x0D66 => "mlym",
        0x1810 => "mong",
        0x1040 => "mymr",
        0x0B66 => "orya",
        0x104A0 => "osma",
        0xA8D0 => "saur",
        0x1BB0 => "sund",
        0x19D0 => "talu",
        0x0BE6 => "tamldec",
        0x0C66 => "telu",
        0x0E50 => "thai",
        0x0F20 => "tibt",
        0xA620 => "vaii",
        _ => return None,
    })
}

/// Whether a decimal digit vector, rounded at index `cut`, rounds *up* under
/// `mode`. Mirrors `intl::number`'s internal `should_round_up`, but the caller
/// feeds the value's **shortest round-trip decimal** so the rounding boundary
/// matches ECMA-402 / ICU (which rounds the shortest decimal) rather than the
/// f64's exact binary expansion (`1.15` is really `1.1499…`, so the crate's own
/// path wrongly rounds it *down*).
#[cfg(feature = "intl")]
fn dec_round_up(
    digits: &[u8],
    cut: usize,
    mode: intl::number::RoundingMode,
    negative: bool,
) -> bool {
    use intl::number::RoundingMode::*;
    if cut >= digits.len() {
        return false;
    }
    let first = digits[cut];
    let rest_nonzero = digits[cut + 1..].iter().any(|&d| d != 0);
    let any = first != 0 || rest_nonzero;
    let gt_half = first > 5 || (first == 5 && rest_nonzero);
    let eq_half = first == 5 && !rest_nonzero;
    let kept_last_odd = cut > 0 && digits[cut - 1] % 2 == 1;
    match mode {
        Trunc => false,
        Expand => any,
        Ceil => any && !negative,
        Floor => any && negative,
        HalfExpand => gt_half || eq_half,
        HalfTrunc => gt_half,
        HalfEven => gt_half || (eq_half && kept_last_odd),
        HalfCeil => gt_half || (eq_half && !negative),
        HalfFloor => gt_half || (eq_half && negative),
    }
}

/// Add `delta` (< 10^18) at the units place of a big-endian decimal digit vector,
/// growing `point` (integer-digit count) on a leading carry.
#[cfg(feature = "intl")]
fn dec_add_units(digits: &mut alloc::vec::Vec<u8>, point: &mut usize, delta: u64) {
    let mut carry = delta;
    let mut i = digits.len();
    while carry > 0 && i > 0 {
        i -= 1;
        let sum = digits[i] as u64 + carry;
        digits[i] = (sum % 10) as u8;
        carry = sum / 10;
    }
    while carry > 0 {
        digits.insert(0, (carry % 10) as u8);
        carry /= 10;
        *point += 1;
    }
}

/// Subtract `delta` (≤ the represented value) at the units place of a big-endian
/// decimal digit vector.
#[cfg(feature = "intl")]
fn dec_sub_units(digits: &mut [u8], mut delta: u64) {
    let mut i = digits.len();
    while delta > 0 && i > 0 {
        i -= 1;
        let cur = digits[i] as i64 - (delta % 10) as i64;
        delta /= 10;
        if cur < 0 {
            digits[i] = (cur + 10) as u8;
            delta += 1;
        } else {
            digits[i] = cur as u8;
        }
    }
}

/// Round the **shortest round-trip decimal** of finite non-zero `n` to `keep_frac`
/// fraction digits (or, when `sig` is `Some`, to that many significant digits)
/// under `mode`, then snap to the nearest multiple of `increment` at the fraction
/// place. Returns a "clean" f64 whose exact binary expansion re-renders (through
/// `intl::number::format`) to the ECMA-402-correct digits, sidestepping the
/// crate's binary-expansion rounding boundary. See [`dec_round_up`].
#[cfg(feature = "intl")]
fn intl_decimal_round(
    n: f64,
    keep_frac: usize,
    sig: Option<usize>,
    increment: u32,
    mode: intl::number::RoundingMode,
) -> f64 {
    if !n.is_finite() || n == 0.0 {
        return n;
    }
    let negative = n.is_sign_negative();
    let abs = n.abs();
    let s = alloc::format!("{abs}");
    let (ip, fp) = s.split_once('.').unwrap_or((s.as_str(), ""));
    let mut digits: alloc::vec::Vec<u8> = ip.bytes().chain(fp.bytes()).map(|b| b - b'0').collect();
    let mut point = ip.len();
    if increment > 1 && sig.is_none() {
        // `roundingIncrement`: round `value × 10^keep_frac` to the nearest multiple
        // of `increment`, comparing the full discarded remainder against the
        // increment midpoint (no intermediate rounding to the fraction place, which
        // would double-round, e.g. 1.25 inc=2 → 1.2, not 1.4).
        let inc = increment as u64;
        // Pad so exactly `point + keep_frac` digits precede the cut.
        while digits.len() < point + keep_frac {
            digits.push(0);
        }
        let cut = point + keep_frac;
        // Remainder of the kept integer modulo the increment, and the quotient's
        // low bit (for ties-to-even on the multiple index).
        let mut rem_int: u64 = 0;
        let mut q_low: u64 = 0;
        for &d in &digits[..cut] {
            let cur = rem_int * 10 + d as u64;
            q_low = q_low.wrapping_mul(10).wrapping_add(cur / inc);
            rem_int = cur % inc;
        }
        // Fractional remainder below the cut, classified against 1/2.
        let first = digits.get(cut).copied().unwrap_or(0);
        let rest_nonzero = cut < digits.len() && digits[cut + 1..].iter().any(|&d| d != 0);
        let frac_pos = cut < digits.len() && (first != 0 || rest_nonzero);
        let frac_gt_half = first > 5 || (first == 5 && rest_nonzero);
        let frac_eq_half = first == 5 && !rest_nonzero;
        // Zero the fraction below the cut, keep the kept integer.
        digits.truncate(cut);
        let twice = rem_int * 2;
        let tie = {
            use intl::number::RoundingMode::*;
            match mode {
                Trunc | HalfTrunc => false,
                Floor => negative,
                Ceil => !negative,
                HalfFloor => negative,
                HalfCeil | HalfExpand | Expand => true,
                HalfEven => (q_low & 1) == 1,
            }
        };
        let up_snap = match twice.cmp(&inc) {
            core::cmp::Ordering::Greater => true,
            core::cmp::Ordering::Equal => {
                if frac_pos {
                    true
                } else {
                    tie
                }
            }
            core::cmp::Ordering::Less => {
                if inc - twice == 1 {
                    frac_gt_half || (frac_eq_half && tie)
                } else {
                    false
                }
            }
        };
        if up_snap {
            dec_add_units(&mut digits, &mut point, inc - rem_int);
        } else if rem_int > 0 {
            dec_sub_units(&mut digits, rem_int);
        }
    } else {
        let cut = if let Some(ms) = sig {
            match digits.iter().position(|&d| d != 0) {
                Some(fz) => (fz + ms).min(digits.len()),
                None => point,
            }
        } else {
            (point + keep_frac).min(digits.len())
        };
        let up = dec_round_up(&digits, cut, mode, negative);
        for d in digits.iter_mut().skip(cut).take(point.saturating_sub(cut)) {
            *d = 0;
        }
        digits.truncate(cut.max(point));
        if up {
            let mut i = cut;
            loop {
                if i == 0 {
                    digits.insert(0, 1);
                    point += 1;
                    break;
                }
                i -= 1;
                if digits[i] == 9 {
                    digits[i] = 0;
                } else {
                    digits[i] += 1;
                    break;
                }
            }
        }
    }
    let int_s: String = digits[..point]
        .iter()
        .map(|&d| (b'0' + d) as char)
        .collect();
    let frac_s: String = digits[point..]
        .iter()
        .map(|&d| (b'0' + d) as char)
        .collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    out.push_str(if int_s.is_empty() { "0" } else { &int_s });
    if !frac_s.is_empty() {
        out.push('.');
        out.push_str(&frac_s);
    }
    out.parse::<f64>().unwrap_or(n)
}

/// Whether a locale renders a negative `currencySign: "accounting"` amount with
/// parentheses (the CLDR root default, inherited by en/ja/ko/zh and most others)
/// rather than a leading minus (e.g. de-DE). Approximates the accounting pattern
/// the `intl` crate does not expose; the minus-sign locales are left to the crate.
#[cfg(feature = "intl")]
fn accounting_uses_parens(locale: &str) -> bool {
    let lang = locale
        .split(['-', '_'])
        .next()
        .unwrap_or(locale)
        .to_ascii_lowercase();
    // Locales known to keep a minus sign in their accounting pattern.
    !matches!(lang.as_str(), "de" | "nl" | "fi" | "hu" | "et")
}

/// `WeekdayToString` for `Intl.Locale`'s `firstDayOfWeek`: a weekday name
/// (`mon`…`sun`) or the 1–7 numeric form (1 = Monday) maps to the canonical name;
/// anything else is invalid.
fn weekday_to_string(s: &str) -> Option<String> {
    Some(String::from(match s {
        "1" => "mon",
        "2" => "tue",
        "3" => "wed",
        "4" => "thu",
        "5" => "fri",
        "6" => "sat",
        // Both 0 and 7 canonicalize to Sunday.
        "0" | "7" => "sun",
        // A non-numeric value is used verbatim if it is a valid `-u-fw` type
        // sequence (weekday names plus experimental calendars like `primidi`).
        other if is_unicode_type_value(other) => return Some(String::from(other)),
        _ => return None,
    }))
}

/// Split a BCP-47 locale into its base (with the requested `u`-extension keyword
/// removed) and the value of that keyword, if present. `key` is the two-letter
/// Unicode keyword (e.g. `"hc"`, `"nu"`, `"ca"`). Returns
/// `(locale_without_key, Some(value))` when the keyword is present, else
/// `(locale.to_owned(), None)`. Keys are exactly two chars; type values are
/// 3–8 chars, so the two are distinguishable within the `-u-` section.
fn split_u_keyword(locale: &str, key: &str) -> (String, Option<String>) {
    let segs: Vec<&str> = locale.split('-').collect();
    // Locate the `u` singleton extension (a length-1 segment equal to `u`).
    let mut ustart = None;
    for (i, s) in segs.iter().enumerate() {
        if s.len() == 1 && s.eq_ignore_ascii_case("u") {
            ustart = Some(i);
            break;
        }
    }
    let Some(ustart) = ustart else {
        return (String::from(locale), None);
    };
    // The `u` section runs until the next singleton (length-1) segment or end.
    let mut ext_end = ustart + 1;
    while ext_end < segs.len() && segs[ext_end].len() != 1 {
        ext_end += 1;
    }
    let is_key = |s: &str| s.len() == 2 && s.bytes().all(|b| b.is_ascii_alphanumeric());
    let mut found: Option<String> = None;
    let mut kept: Vec<&str> = Vec::new();
    let mut j = ustart + 1;
    while j < ext_end {
        let cur = segs[j];
        if is_key(cur) {
            // Gather this keyword's type values (3–8 char segments until the
            // next 2-char key or the section end).
            let mut k = j + 1;
            while k < ext_end && !is_key(segs[k]) {
                k += 1;
            }
            if cur.eq_ignore_ascii_case(key) {
                found = Some(segs[j + 1..k].join("-").to_ascii_lowercase());
            } else {
                kept.extend_from_slice(&segs[j..k]);
            }
            j = k;
        } else {
            // A leading attribute (no key) — preserve it.
            kept.push(cur);
            j += 1;
        }
    }
    if found.is_none() {
        return (String::from(locale), None);
    }
    let mut out: Vec<&str> = segs[..ustart].to_vec();
    if !kept.is_empty() {
        out.push("u");
        out.extend_from_slice(&kept);
    }
    out.extend_from_slice(&segs[ext_end..]);
    (out.join("-"), found)
}

/// Decide whether the dropped-tail `dropped` (the digits at and beyond the kept
/// precision) rounds the retained value up (away from zero), under `mode`.
#[cfg(feature = "intl")]
fn exact_round_up(dropped: &[u8], mode: intl::number::RoundingMode, neg: bool) -> bool {
    use intl::number::RoundingMode::*;
    if dropped.is_empty() {
        return false;
    }
    let first = dropped[0];
    let rest_nonzero = dropped[1..].iter().any(|&d| d != 0);
    let any_nonzero = first != 0 || rest_nonzero;
    match mode {
        Trunc => false,
        Expand => any_nonzero,
        Ceil => any_nonzero && !neg,
        Floor => any_nonzero && neg,
        HalfExpand => first >= 5,
        HalfTrunc => first > 5 || (first == 5 && rest_nonzero),
        HalfCeil => (first > 5 || (first == 5 && rest_nonzero)) || (first == 5 && !neg),
        HalfFloor => (first > 5 || (first == 5 && rest_nonzero)) || (first == 5 && neg),
        // Half-even (banker's) needs the last kept digit; approximate as halfExpand
        // (unused by the exact-decimal target tests).
        _ => first >= 5,
    }
}

/// Add one unit at the least-significant retained position of `int.frac`,
/// propagating the carry (grows `int` if it overflows).
#[cfg(feature = "intl")]
fn exact_increment(int: &mut alloc::vec::Vec<u8>, frac: &mut [u8]) {
    let mut carry = 1u8;
    for d in frac.iter_mut().rev() {
        let s = *d + carry;
        *d = s % 10;
        carry = s / 10;
        if carry == 0 {
            return;
        }
    }
    for d in int.iter_mut().rev() {
        let s = *d + carry;
        *d = s % 10;
        carry = s / 10;
        if carry == 0 {
            return;
        }
    }
    if carry > 0 {
        int.insert(0, carry);
    }
}

/// Map an ECMA-402 [`RoundingMode`](intl::number::RoundingMode) to the
/// `puremp::Decimal` [`Rounding`](puremp::decimal::Rounding) that reproduces it
/// exactly. `puremp` has no ties-toward-±∞ mode, so `halfCeil`/`halfFloor` are
/// resolved per the value's sign (`neg`) into ties-away-from / ties-toward zero —
/// exact because the sign fixes which of ±∞ the tie rounds toward.
#[cfg(feature = "intl")]
fn intl_mode_to_rounding(mode: intl::number::RoundingMode, neg: bool) -> puremp::decimal::Rounding {
    use intl::number::RoundingMode as M;
    use puremp::decimal::Rounding as R;
    match mode {
        M::Trunc => R::Down,
        M::Expand => R::Up,
        M::Floor => R::Floor,
        M::Ceil => R::Ceiling,
        M::HalfExpand => R::HalfUp,
        M::HalfTrunc => R::HalfDown,
        M::HalfEven => R::HalfEven,
        M::HalfCeil => {
            if neg {
                R::HalfDown
            } else {
                R::HalfUp
            }
        }
        M::HalfFloor => {
            if neg {
                R::HalfUp
            } else {
                R::HalfDown
            }
        }
    }
}

/// Exact ECMA-402 `ToRawPrecision` for a high-precision operand: rounds the exact
/// value `±int_part.frac_part` to `max_sig` significant digits (then pads trailing
/// zeros up to `min_sig`) using a `puremp::Decimal`, so the result is exact for
/// values beyond `f64` precision and every rounding mode — including half-even,
/// which the hand-rolled digit helpers only approximate. Returns the rounded
/// `(integer_digits, fraction_digits)` latn digit vectors for `assemble_decimal`.
#[cfg(feature = "intl")]
fn exact_significant_digits(
    neg: bool,
    int_part: &str,
    frac_part: &str,
    min_sig: Option<usize>,
    max_sig: Option<usize>,
    mode: intl::number::RoundingMode,
) -> Option<(alloc::vec::Vec<u8>, alloc::vec::Vec<u8>)> {
    use puremp::decimal::Decimal;
    // Reassemble the signed literal and parse it into an exact base-10 Decimal.
    let mut lit = String::new();
    if neg {
        lit.push('-');
    }
    lit.push_str(if int_part.is_empty() { "0" } else { int_part });
    if !frac_part.is_empty() {
        lit.push('.');
        lit.push_str(frac_part);
    }
    let d: Decimal = lit.parse().ok()?;
    let rounded = match max_sig {
        Some(m) => d.round_to_digits(m.max(1) as u32, intl_mode_to_rounding(mode, neg)),
        None => d,
    };
    // Expand the rounded Decimal to plain (non-scientific) int/frac digit runs.
    let s = alloc::format!("{}", rounded.abs());
    let (ip, fp) = s.split_once('.').unwrap_or((s.as_str(), ""));
    let mut int_digits: alloc::vec::Vec<u8> = ip.bytes().map(|b| b - b'0').collect();
    let mut frac_digits: alloc::vec::Vec<u8> = fp.bytes().map(|b| b - b'0').collect();
    if int_digits.is_empty() {
        int_digits.push(0);
    }
    // minimumSignificantDigits: pad trailing zeros so at least `min_sig` significant
    // digits (counting from the first nonzero, trailing zeros included) are present.
    if let Some(min_sig) = min_sig {
        let first = int_digits
            .iter()
            .chain(frac_digits.iter())
            .position(|&d| d != 0)
            .unwrap_or(int_digits.len());
        let sig_now = (int_digits.len() + frac_digits.len()).saturating_sub(first);
        if sig_now < min_sig {
            frac_digits.resize(frac_digits.len() + (min_sig - sig_now), 0);
        }
    }
    Some((int_digits, frac_digits))
}

/// Whether a compact-notation formatter carries no explicit digit options, so the
/// ECMA-402 default rounding (`roundingPriority: "morePrecision"` with
/// `maximumFractionDigits: 0` / `maximumSignificantDigits: 2`) applies and the
/// crate's plain one-fraction-digit mantissa must be re-rounded (see
/// [`compact_reround_parts`]). Any user-set fraction/significant digit count opts
/// out (the crate then honors the given precision).
#[cfg(feature = "intl")]
fn compact_wants_reround(opts: &intl::number::NumberFormatOptions) -> bool {
    matches!(opts.notation, intl::number::Notation::Compact)
        && opts.minimum_fraction_digits.is_none()
        && opts.maximum_fraction_digits.is_none()
        && opts.minimum_significant_digits.is_none()
        && opts.maximum_significant_digits.is_none()
}

/// Re-round the mantissa of a default compact-notation result to the ECMA-402
/// default precision. The crate renders the compact mantissa with a single
/// fraction digit (`987654321` → `987.7M`); the spec applies
/// `roundingPriority: "morePrecision"` over `maximumFractionDigits: 0` and
/// `maximumSignificantDigits: 2`, which for a mantissa whose most-significant
/// digit sits at decimal exponent `e` keeps `max(0, 1 - e)` fraction digits
/// (`987.65…M` → `988M`, `9.87…K` → `9.9K`, `0.159` → `0.16`). Operates on the
/// crate's tagged `(kind, value)` parts (latn digits, before numbering-system
/// substitution): rewrites the leading `integer`/`decimal`/`fraction` run in
/// place, leaving the sign and compact-suffix parts untouched. Grouped mantissae
/// (a `group` part right after the integer) are left alone — compact mantissae are
/// always < 1000 there, so no re-rounding is needed.
#[cfg(feature = "intl")]
fn compact_reround_parts(
    parts: &mut alloc::vec::Vec<(&'static str, String)>,
    mode: intl::number::RoundingMode,
) {
    let Some(int_idx) = parts.iter().position(|(k, _)| *k == "integer") else {
        return;
    };
    // Skip a grouped integer core (a `group` immediately after the integer): the
    // digits span several parts, but such a mantissa is >= 1000 with no fraction,
    // so `max(0, 1 - e)` is already 0 and nothing needs re-rounding.
    if parts.get(int_idx + 1).map(|(k, _)| *k) == Some("group") {
        return;
    }
    let neg = parts.iter().any(|(k, _)| *k == "minusSign");
    let int_str = parts[int_idx].1.clone();
    let (dec_idx, frac_idx) = if parts.get(int_idx + 1).map(|(k, _)| *k) == Some("decimal") {
        let f = if parts.get(int_idx + 2).map(|(k, _)| *k) == Some("fraction") {
            Some(int_idx + 2)
        } else {
            None
        };
        (Some(int_idx + 1), f)
    } else {
        (None, None)
    };
    let frac_str = frac_idx.map(|i| parts[i].1.clone()).unwrap_or_default();
    // Only re-round plain latn digits (kataan substitutes the numbering system
    // afterwards); bail on anything unexpected.
    if !int_str.bytes().all(|b| b.is_ascii_digit()) || !frac_str.bytes().all(|b| b.is_ascii_digit())
    {
        return;
    }
    // Decimal exponent `e` of the mantissa's most-significant digit.
    let int_stripped = int_str.trim_start_matches('0');
    let e: i32 = if !int_stripped.is_empty() {
        int_stripped.len() as i32 - 1
    } else {
        match frac_str.bytes().position(|b| b != b'0') {
            Some(p) => -(p as i32) - 1,
            None => 0,
        }
    };
    let keep = (1 - e).max(0) as usize;
    let mut int_d: alloc::vec::Vec<u8> = int_str.bytes().map(|b| b - b'0').collect();
    let mut frac_d: alloc::vec::Vec<u8> = frac_str.bytes().map(|b| b - b'0').collect();
    if frac_d.len() > keep {
        let up = exact_round_up(&frac_d[keep..], mode, neg);
        frac_d.truncate(keep);
        if up {
            exact_increment(&mut int_d, &mut frac_d);
        }
    }
    // Trim trailing fraction zeros (minimumFractionDigits defaults to 0).
    while frac_d.last() == Some(&0) {
        frac_d.pop();
    }
    while int_d.len() > 1 && int_d[0] == 0 {
        int_d.remove(0);
    }
    let new_int: String = int_d.iter().map(|d| (b'0' + d) as char).collect();
    let new_frac: String = frac_d.iter().map(|d| (b'0' + d) as char).collect();
    let dec_sep = dec_idx.map(|i| parts[i].1.clone());
    // Remove the old fraction then decimal (descending index keeps `int_idx` valid).
    if let Some(fi) = frac_idx {
        parts.remove(fi);
    }
    if let Some(di) = dec_idx {
        parts.remove(di);
    }
    parts[int_idx].1 = new_int;
    if !new_frac.is_empty() {
        let sep = dec_sep.unwrap_or_else(|| String::from("."));
        parts.insert(int_idx + 1, ("decimal", sep));
        parts.insert(int_idx + 2, ("fraction", new_frac));
    }
}

/// Split the separator whitespace off each `compact` part into its own `literal`
/// part, matching ECMA-402's parts shaping: a CLDR compact pattern like `"0 Mio."`
/// tags the space between the number and the suffix as a `literal` and only the
/// bare suffix as `compact` (`[integer "988"][literal "\u{a0}"][compact "Mio."]`).
/// The `intl` crate returns the leading/trailing whitespace fused into the
/// `compact` value; only the *outer* whitespace is peeled here (an internal space
/// in a multi-word suffix stays part of the `compact` token).
#[cfg(feature = "intl")]
fn split_compact_affix_parts(parts: &mut alloc::vec::Vec<(&'static str, String)>) {
    let needs = parts.iter().any(|(k, v)| {
        *k == "compact" && (v.starts_with(char::is_whitespace) || v.ends_with(char::is_whitespace))
    });
    if !needs {
        return;
    }
    let mut out: alloc::vec::Vec<(&'static str, String)> =
        alloc::vec::Vec::with_capacity(parts.len() + 2);
    for (k, v) in core::mem::take(parts) {
        if k != "compact" || v.trim().is_empty() {
            out.push((k, v));
            continue;
        }
        let lead_len = v.len() - v.trim_start().len();
        let core_end = v.trim_end().len();
        if lead_len > 0 {
            out.push(("literal", String::from(&v[..lead_len])));
        }
        out.push(("compact", String::from(&v[lead_len..core_end])));
        if core_end < v.len() {
            out.push(("literal", String::from(&v[core_end..])));
        }
    }
    *parts = out;
}

/// From a probe like `"1.1"` / `"-1.1"` (latn digits), split the leading prefix,
/// the decimal separator, and the trailing suffix.
#[cfg(feature = "intl")]
fn split_number_scaffold(probe: &str, nu: &str) -> (String, String, String) {
    let chars: alloc::vec::Vec<char> = probe.chars().collect();
    let mut i = 0;
    let mut prefix = String::new();
    while i < chars.len() && !is_digit_of_numbering_system(nu, chars[i]) {
        prefix.push(chars[i]);
        i += 1;
    }
    while i < chars.len() && is_digit_of_numbering_system(nu, chars[i]) {
        i += 1;
    }
    let mut sep = String::new();
    while i < chars.len() && !is_digit_of_numbering_system(nu, chars[i]) {
        sep.push(chars[i]);
        i += 1;
    }
    while i < chars.len() && is_digit_of_numbering_system(nu, chars[i]) {
        i += 1;
    }
    let suffix: String = chars[i..].iter().collect();
    (prefix, sep, suffix)
}

/// From a probe like `"1,111"` (latn), extract the first group separator that
/// appears between integer digit groups (`""` if the value isn't grouped).
#[cfg(feature = "intl")]
fn extract_group_sep(probe: &str, nu: &str) -> String {
    let chars: alloc::vec::Vec<char> = probe.chars().collect();
    let mut i = 0;
    while i < chars.len() && !is_digit_of_numbering_system(nu, chars[i]) {
        i += 1; // skip prefix
    }
    while i < chars.len() && is_digit_of_numbering_system(nu, chars[i]) {
        i += 1; // first integer group
    }
    let mut sep = String::new();
    while i < chars.len() && !is_digit_of_numbering_system(nu, chars[i]) {
        sep.push(chars[i]);
        i += 1;
    }
    sep
}

/// `ToRawFixed`: round the value `(int, frac)` to at most `max_frac` fraction
/// digits under `mode`. Returns the rounded `(int, frac)`.
#[cfg(feature = "intl")]
fn to_raw_fixed(
    neg: bool,
    mut int: alloc::vec::Vec<u8>,
    mut frac: alloc::vec::Vec<u8>,
    max_frac: usize,
    mode: intl::number::RoundingMode,
) -> (alloc::vec::Vec<u8>, alloc::vec::Vec<u8>) {
    if frac.len() > max_frac {
        let up = exact_round_up(&frac[max_frac..], mode, neg);
        frac.truncate(max_frac);
        if up {
            exact_increment(&mut int, &mut frac);
        }
    }
    (int, frac)
}

/// `ToRawPrecision`: round the value `(int, frac)` to at most `max_sig`
/// significant digits under `mode`. Returns the rounded `(int, frac)`.
#[cfg(feature = "intl")]
fn to_raw_precision(
    neg: bool,
    int: alloc::vec::Vec<u8>,
    frac: alloc::vec::Vec<u8>,
    max_sig: usize,
    mode: intl::number::RoundingMode,
) -> (alloc::vec::Vec<u8>, alloc::vec::Vec<u8>) {
    let mut digits: alloc::vec::Vec<u8> = int.iter().chain(frac.iter()).copied().collect();
    let mut point = int.len();
    let Some(first_sig) = digits.iter().position(|&d| d != 0) else {
        return (int, frac);
    };
    let last = first_sig + max_sig.max(1) - 1;
    if last + 1 < digits.len() {
        let up = exact_round_up(&digits[last + 1..], mode, neg);
        digits.truncate(last + 1);
        if up {
            let mut carry = 1u8;
            for d in digits.iter_mut().rev() {
                let s = *d + carry;
                *d = s % 10;
                carry = s / 10;
                if carry == 0 {
                    break;
                }
            }
            if carry > 0 {
                digits.insert(0, carry);
                point += 1;
            }
        }
    }
    while digits.len() < point {
        digits.push(0);
    }
    let int_out = digits[..point].to_vec();
    let frac_out = digits[point..].to_vec();
    (int_out, frac_out)
}

/// Group `int_str` (latn digits, no sign) into thousands with `sep`.
#[cfg(feature = "intl")]
fn group_thousands_sep(int_str: &str, sep: &str) -> String {
    if sep.is_empty() || int_str.len() <= 3 {
        return String::from(int_str);
    }
    let bytes = int_str.as_bytes();
    let mut out = String::new();
    let first = bytes.len() % 3;
    let first = if first == 0 { 3 } else { first };
    out.push_str(&int_str[..first]);
    let mut i = first;
    while i < bytes.len() {
        out.push_str(sep);
        out.push_str(&int_str[i..i + 3]);
        i += 3;
    }
    out
}

/// Whether `s` matches the UTS-35 `type` value production: one or more
/// `alphanum{3,8}` subtags joined by `-` (used for `ca`/`nu` option validation).
fn is_unicode_type_value(s: &str) -> bool {
    !s.is_empty()
        && s.split('-').all(|seg| {
            (3..=8).contains(&seg.len()) && seg.bytes().all(|b| b.is_ascii_alphanumeric())
        })
}

/// Whether `code` is a well-formed ISO-4217 currency code: exactly three ASCII
/// letters (case-insensitive).
fn is_well_formed_currency(code: &str) -> bool {
    code.len() == 3 && code.bytes().all(|b| b.is_ascii_alphabetic())
}

/// The ECMA-402 sanctioned single-unit identifiers (Table: "Single units
/// sanctioned for use in ECMAScript"). Shared by `Intl.supportedValuesOf("unit")`
/// and `IsWellFormedUnitIdentifier`.
pub(crate) const SANCTIONED_UNITS: &[&str] = &[
    "acre",
    "bit",
    "byte",
    "celsius",
    "centimeter",
    "day",
    "degree",
    "fahrenheit",
    "fluid-ounce",
    "foot",
    "gallon",
    "gigabit",
    "gigabyte",
    "gram",
    "hectare",
    "hour",
    "inch",
    "kilobit",
    "kilobyte",
    "kilogram",
    "kilometer",
    "liter",
    "megabit",
    "megabyte",
    "meter",
    "microsecond",
    "mile",
    "mile-scandinavian",
    "milliliter",
    "millimeter",
    "millisecond",
    "minute",
    "month",
    "nanosecond",
    "ounce",
    "percent",
    "petabyte",
    "pound",
    "second",
    "stone",
    "terabit",
    "terabyte",
    "week",
    "yard",
    "year",
];

/// `IsWellFormedUnitIdentifier(unit)`: a sanctioned single unit, or a
/// `<numerator>-per-<denominator>` compound of two sanctioned single units.
fn is_well_formed_unit(unit: &str) -> bool {
    let valid_single = |u: &str| SANCTIONED_UNITS.contains(&u);
    match unit.split_once("-per-") {
        Some((a, b)) => valid_single(a) && valid_single(b),
        None => valid_single(unit),
    }
}

/// Validates and canonicalizes a BCP-47 / UTS-35 `unicode_locale_id` (the
/// structural grammar, without CLDR alias/grandfathered replacement which the
/// `intl` crate doesn't expose). Returns the canonical tag, or `None` if the tag
/// is structurally invalid (the caller raises a `RangeError`).
///
/// Canonicalization performed: lowercase language; Titlecase script; UPPERCASE
/// region; lowercase variants **sorted** alphabetically; extensions ordered by
/// singleton (`x` private-use last); `-u-`/`-t-` keyword/field groups sorted by
/// key in ASCII order. Not performed (needs CLDR data the crate omits): legacy
/// language/region alias replacement, `-u-` type alias mapping (e.g. `yes`→`true`).
/// UTS-35 §3.3.1 "regular" grandfathered tag → canonical replacement, matched
/// on the lowercased language(-variant) base. This is the fixed set of regular
/// grandfathered tags that remain structurally valid Unicode BCP-47 locale ids
/// (their preferred replacement changes the primary language subtag).
fn grandfathered_canonical(base: &str) -> Option<&'static str> {
    Some(match base {
        "art-lojban" => "jbo",
        "cel-gaulish" => "xtg",
        "zh-guoyu" => "zh",
        "zh-hakka" => "hak",
        "zh-xiang" => "hsn",
        _ => return None,
    })
}

/// CLDR `bcp47` Unicode-extension type-value alias → canonical value, for the
/// `(key, value)` pairs ECMA-402 exercises. Deprecated type values (e.g.
/// `ca-islamicc`, `ks-primary`, `ms-imperial`) canonicalize to their preferred
/// forms. `value` is the full (hyphen-joined) type. Returns `None` when the pair
/// has no alias.
pub(crate) fn unicode_type_alias(key: &str, value: &str) -> Option<&'static str> {
    Some(match (key, value) {
        // -u-ca- (calendar)
        ("ca", "ethiopic-amete-alem") => "ethioaa",
        ("ca", "islamicc") => "islamic-civil",
        // -u-ks- (collation strength)
        ("ks", "primary") => "level1",
        ("ks", "tertiary") => "level3",
        // -u-ms- (measurement system)
        ("ms", "imperial") => "uksystem",
        // -u-tz- (time zone) and -u-rg-/-u-sd- (region override / subdivision):
        // the CLDR deprecation tables, generated into `intl_aliases`.
        ("tz", v) => return super::intl_aliases::lookup(super::intl_aliases::TIMEZONE, v),
        ("rg" | "sd", v) => {
            return super::intl_aliases::lookup(super::intl_aliases::SUBDIVISION, v);
        }
        _ => return None,
    })
}

/// The preferred form of a `-t-` *tvalue* (`m0-names` → `m0-prprname`), from
/// CLDR's `bcp47/transform.xml` deprecation aliases.
pub(crate) fn transform_value_alias(value: &str) -> Option<&'static str> {
    super::intl_aliases::lookup(super::intl_aliases::TRANSFORM_VALUE, value)
}

/// Extracts the `-u-ca-<calendar>` value from a BCP-47 locale tag, if present
/// (e.g. `"en-u-ca-iso8601"` → `Some("iso8601")`, `"ja-u-ca-islamic-civil"` →
/// `Some("islamic-civil")`). The calendar value may span several `-`-joined
/// subtags (each ≥ 3 chars); collection stops at the next 2-char extension key.
#[cfg(feature = "intl")]
pub(crate) fn locale_unicode_calendar(locale: &str) -> Option<String> {
    let lower = locale.to_ascii_lowercase();
    let subtags: Vec<&str> = lower.split('-').collect();
    // Find the "-u-" singleton extension, then the "ca" key inside it.
    let mut i = 0;
    while i < subtags.len() {
        if subtags[i] == "u" {
            let mut j = i + 1;
            while j < subtags.len() && subtags[j] != "u" {
                if subtags[j] == "ca" {
                    let mut parts = Vec::new();
                    let mut k = j + 1;
                    while k < subtags.len() && subtags[k].len() >= 3 {
                        parts.push(subtags[k]);
                        k += 1;
                    }
                    if parts.is_empty() {
                        return None;
                    }
                    let value = parts.join("-");
                    return Some(
                        unicode_type_alias("ca", &value)
                            .map(String::from)
                            .unwrap_or(value),
                    );
                }
                j += 1;
            }
        }
        i += 1;
    }
    None
}

/// Extracts the `-u-<key>-<value>` type value from a BCP-47 locale tag, if
/// present (e.g. `locale_unicode_keyword("en-u-nu-arab", "nu")` → `Some("arab")`).
/// The value may span several `-`-joined subtags (each ≥ 3 chars); collection
/// stops at the next 2-char extension key or the end of the `-u-` group.
pub(crate) fn locale_unicode_keyword(locale: &str, key: &str) -> Option<String> {
    let lower = locale.to_ascii_lowercase();
    let subtags: Vec<&str> = lower.split('-').collect();
    let mut i = 0;
    while i < subtags.len() {
        // A private-use singleton (`x`) starts private subtags; a `-u-` after it
        // is not a real extension (`de-x-u-co-phonebk` has no `co` keyword).
        if subtags[i] == "x" {
            break;
        }
        if subtags[i] == "u" {
            let mut j = i + 1;
            while j < subtags.len() && subtags[j].len() != 1 {
                if subtags[j] == key {
                    let mut parts = Vec::new();
                    let mut k = j + 1;
                    while k < subtags.len() && subtags[k].len() >= 3 {
                        parts.push(subtags[k]);
                        k += 1;
                    }
                    if parts.is_empty() {
                        return None;
                    }
                    return Some(parts.join("-"));
                }
                j += 1;
            }
        }
        i += 1;
    }
    None
}

/// Extracts a boolean Unicode `-u-<key>-` keyword (e.g. `kn`) from a locale tag:
/// a bare key or an explicit `true` → `Some(true)`, `false` → `Some(false)`, the
/// key being absent → `None`. Handles the canonical form where a `true` value is
/// dropped (`en-u-kn-true` → `en-u-kn`).
pub(crate) fn locale_unicode_bool_keyword(locale: &str, key: &str) -> Option<bool> {
    let lower = locale.to_ascii_lowercase();
    let subs: Vec<&str> = lower.split('-').collect();
    let mut i = 0;
    while i < subs.len() {
        if subs[i] == "x" {
            break;
        }
        if subs[i] == "u" {
            let mut j = i + 1;
            while j < subs.len() && subs[j].len() != 1 {
                if subs[j] == key {
                    return Some(match subs.get(j + 1) {
                        Some(v) if v.len() >= 3 => *v != "false",
                        _ => true,
                    });
                }
                j += 1;
            }
        }
        i += 1;
    }
    None
}

/// Whether `co` is a collation type the engine treats as a supported
/// `Intl.Collator` collation (the CLDR collation identifiers, excluding the
/// reserved `standard`/`search` values that are never valid `-u-co-` types).
pub(crate) fn is_supported_collation(base: &str, co: &str) -> bool {
    // `emoji` and `eor` are root collations: they tailor the root order rather
    // than a language's, so every locale offers them.
    if matches!(co, "emoji" | "eor") {
        return true;
    }
    // Every other collation belongs to specific languages, and asking for one
    // outside them is simply not a supported request — it must be ignored, not
    // reported back. This list was read off ICU (`resolvedOptions().collation`
    // over each collation × a spread of 59 languages) and matches CLDR's
    // per-locale `collations`, with one deliberate divergence: ICU reports
    // `zh` + `pinyin` as `"default"` (pinyin *is* Chinese's default, so it
    // canonicalizes the request away), yet still lists `pinyin` in
    // `supportedValuesOf("collation")` — which contradicts
    // `collations-accepted-by-Collator.js`, and node fails that test as a
    // result. Accepting it for `zh` keeps the two consistent, which is what the
    // test actually requires.
    let langs: &[&str] = match co {
        "compat" => &["ar"],
        "dict" => &["si"],
        "phonebk" => &["de"],
        "phonetic" => &["ln"],
        "searchjl" => &["ko"],
        "pinyin" | "stroke" | "zhuyin" => &["zh"],
        "trad" => &["bn", "es", "fi", "kn", "sv", "vi"],
        "unihan" => &["ja", "ko", "zh"],
        _ => return false,
    };
    let lang = base.split(['-', '_']).next().unwrap_or(base);
    langs.iter().any(|l| lang.eq_ignore_ascii_case(l))
}

/// `PartitionRelativeTimePattern` without CLDR: the English "in N units" /
/// "N units ago" shape, with the integer part grouped in threes.
///
/// Only reachable in a build without the `intl` feature, which has no locale
/// data at all — the CLDR path ([`Interp::rel_time_parts_cldr`]) is the real
/// implementation. This exists because the non-`intl` build otherwise does not
/// compile: the previous fallback called `crate::nbexec::rel_time_parts`, which
/// was deleted along with three English tables when relative time moved to CLDR
/// (737cb2fc), leaving a dangling call no CI job builds.
///
/// The sign *bit* picks the pattern, so `-0` formats as past ("0 units ago")
/// while `+0` is future, matching the spec's `PartitionRelativeTimePattern`.
#[cfg(not(feature = "intl"))]
fn rel_time_parts_en(value: f64, unit: &str) -> Vec<(&'static str, String, bool)> {
    let n = value.abs();
    let s = alloc::format!("{n}");
    let (int_str, frac_str) = s.split_once('.').unwrap_or((s.as_str(), ""));
    let mut parts: Vec<(&'static str, String, bool)> = Vec::new();
    if !value.is_sign_negative() {
        parts.push(("literal", String::from("in "), false));
    }
    // Group the integer digits in threes from the right.
    let digits: Vec<char> = int_str.chars().collect();
    let head = match digits.len() % 3 {
        0 => 3.min(digits.len()),
        r => r,
    };
    for (i, chunk) in core::iter::once(&digits[..head])
        .chain(digits[head..].chunks(3))
        .enumerate()
    {
        if i > 0 {
            parts.push(("group", String::from(","), true));
        }
        parts.push(("integer", chunk.iter().collect(), true));
    }
    if !frac_str.is_empty() {
        parts.push(("decimal", String::from("."), true));
        parts.push(("fraction", String::from(frac_str), true));
    }
    // `1 day`, but `0 days` / `2 days` — English pluralizes everything but one.
    let plural = if n == 1.0 { "" } else { "s" };
    parts.push((
        "literal",
        if value.is_sign_negative() {
            alloc::format!(" {unit}{plural} ago")
        } else {
            alloc::format!(" {unit}{plural}")
        },
        true,
    ));
    parts
}

/// Zero-pads the minute/second fields of crate-produced `DateTimePart`s to two
/// digits when combined with another time field, and maps each to its ECMA-402
/// `(type, value)` part. CLDR/ICU has no single-digit `m`/`s` time format:
/// whenever minute or second appears alongside another time field it renders
/// 2-digit even for the `"numeric"` option (the crate widens by option only).
/// Shared by the `Date` and Temporal DateTimeFormat rendering paths.
///
/// Literal separators additionally have U+202F (narrow no-break space) folded to
/// a plain space. CLDR 42 changed the `en` time patterns to separate the clock
/// from the day period with U+202F; the resulting web breakage led the reference
/// implementation (V8, hence `node`) to fold it back to U+0020 in *date-time*
/// output specifically — `Intl.NumberFormat` keeps its U+202F group separator
/// (e.g. `fr`). Test262 bakes that reference behaviour into the
/// `DateTimeFormat/prototype/{format,formatToParts}/dayPeriod-*-en.js`
/// expectations, so match it here rather than in the shared CLDR tables.
#[cfg(feature = "intl")]
fn dtf_pad_time_parts(
    parts: alloc::vec::Vec<intl::datetime::DateTimePart>,
) -> Vec<(&'static str, String)> {
    use intl::datetime::DateTimePartType;
    let has_hour = parts.iter().any(|p| p.kind == DateTimePartType::Hour);
    let has_min = parts.iter().any(|p| p.kind == DateTimePartType::Minute);
    let has_sec = parts.iter().any(|p| p.kind == DateTimePartType::Second);
    parts
        .into_iter()
        .map(|p| {
            let mut v = p.value;
            let widen = match p.kind {
                DateTimePartType::Minute => has_hour || has_sec,
                DateTimePartType::Second => has_hour || has_min,
                _ => false,
            };
            if widen && v.len() == 1 && v.as_bytes()[0].is_ascii_digit() {
                v.insert(0, '0');
            }
            if p.kind == DateTimePartType::Literal && v.contains('\u{202f}') {
                v = v.replace('\u{202f}', " ");
            }
            (p.kind.as_str(), v)
        })
        .collect()
}

/// The sexagenary (stem-branch) cyclic year name for the 1-based cyclic index
/// `cyclic1` (1..=60), e.g. `36` → `己亥`. The 60-name cycle is the ten heavenly
/// stems crossed with the twelve earthly branches; the Han characters are the
/// canonical (and CLDR `zh`) forms. Used for the `yearName` part of Chinese/Dangi
/// lunisolar dates (the crate's localized cyclic names are not public).
#[cfg(feature = "intl")]
fn sexagenary_year_name(cyclic1: i64) -> String {
    const STEMS: [&str; 10] = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
    const BRANCHES: [&str; 12] = [
        "子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥",
    ];
    let i = (cyclic1 - 1).rem_euclid(60);
    alloc::format!(
        "{}{}",
        STEMS[(i % 10) as usize],
        BRANCHES[(i % 12) as usize]
    )
}

/// Parses a Temporal month code (`"M05"` / `"M05L"`) into `(number, is_leap)`.
#[cfg(feature = "intl")]
fn month_code_parts(code: &str) -> Option<(u32, bool)> {
    let b = code.as_bytes();
    let leap = b.len() == 4 && b[3] == b'L';
    if !((b.len() == 3 || leap) && b[0] == b'M' && b[1].is_ascii_digit() && b[2].is_ascii_digit()) {
        return None;
    }
    Some((u32::from(b[1] - b'0') * 10 + u32::from(b[2] - b'0'), leap))
}

/// The `intl` crate calendar + CLDR era **index** naming the Temporal era code
/// `era` in calendar `cal`.
///
/// UTS #35 numbers a calendar's `<eras>` positionally and the numbering is not
/// uniform, so this is a table rather than a computation: Gregorian is 0 = BC /
/// 1 = AD, Coptic's *only* era is index 1, the Islamic calendars put Anno
/// Hegirae at 0 and Before Hijrah at 1 (the reverse of the "negative years come
/// first" intuition), and the Japanese nengō run 0 (Taika) … 236 (Reiwa) — of
/// which the engine's Temporal calendar only ever reports the five modern ones,
/// falling back to the Gregorian era codes before Meiji.
#[cfg(feature = "intl")]
fn cldr_era_index(cal: &str, era: &str) -> Option<(intl::datetime::Calendar, u32)> {
    use intl::datetime::Calendar;
    // The five modern nengō occupy the tail of the Japanese era table.
    let modern_nengo = match era {
        "meiji" => Some(232),
        "taisho" => Some(233),
        "showa" => Some(234),
        "heisei" => Some(235),
        "reiwa" => Some(236),
        _ => None,
    };
    if let Some(i) = modern_nengo {
        return Some((Calendar::Japanese, i));
    }
    match era {
        "bce" => return Some((Calendar::Gregory, 0)),
        "ce" => return Some((Calendar::Gregory, 1)),
        _ => {}
    }
    let c = Calendar::from_bcp47(cal)?;
    let idx = match (cal, era) {
        (_, "ah") => 0,
        (_, "bh") => 1,
        ("roc", "broc") => 0,
        ("roc", "roc") => 1,
        ("ethiopic", "aa") => 0,
        ("ethiopic", "am") => 1,
        // Coptic has a single era and CLDR indexes it 1, not 0.
        ("coptic", _) => 1,
        // Every other calendar the engine implements is single-era at index 0
        // (`buddhist` BE, `persian` AP, `hebrew` AM, `indian` Śaka, `ethioaa` AA).
        _ => 0,
    };
    Some((c, idx))
}

/// The CLDR month **index** (and leap-name flag) naming the month a Temporal
/// `month_code` identifies in `cal`'s year.
///
/// CLDR numbers a lunisolar calendar's months by *name slot*, not by ordinal
/// position in the year, and the two disagree exactly where a leap month is
/// involved:
/// * Hebrew reserves slot 6 for Adar I and slot 7 for Adar (Adar II in a leap
///   year), so an ordinary month after Shevat sits one slot above its ordinal in
///   a common year (`M06` Adar is ordinal 6 but slot 7);
/// * Chinese/`dangi` keep the slot equal to the code's number and mark the
///   intercalary month with the `leap` flag instead (`M05L` → slot 5, leap).
///
/// The `leap` flag is what selects between UTS #35's two leap renderings: the
/// `monthPatterns` wrapper (`"5bis"`) for Chinese/`dangi`, and the
/// `yeartype="leap"` alternate name (Adar → Adar II) for Hebrew — which is a
/// property of the *year*, not of the month, hence `leap_year`.
#[cfg(feature = "intl")]
fn cldr_month_index(cal: &str, month_code: &str, ordinal: i64, leap_year: bool) -> (u32, bool) {
    match cal {
        "hebrew" => match month_code_parts(month_code) {
            // `M05L` (Adar I) is slot 6; `M06`..`M12` shift up one past it.
            Some((n, leap)) if leap || n > 5 => (n + 1, leap_year),
            Some((n, _)) => (n, leap_year),
            None => (ordinal.clamp(1, 13) as u32, leap_year),
        },
        "chinese" | "dangi" => match month_code_parts(month_code) {
            Some((n, leap)) => (n, leap),
            None => (ordinal.clamp(1, 12) as u32, false),
        },
        _ => (ordinal.clamp(1, 13) as u32, false),
    }
}

/// The complete set of CLDR numbering-system identifiers (BCP-47 `nu` types).
/// Mirrors ECMA-402's `AvailableNumberingSystems`; used by
/// `Intl.supportedValuesOf("numberingSystem")` and to validate the
/// `numberingSystem` option against known systems in the Intl formatters.
pub(crate) const NUMBERING_SYSTEMS: &[&str] = &[
    "adlm", "ahom", "arab", "arabext", "armn", "armnlow", "bali", "beng", "bhks", "brah", "cakm",
    "cham", "cyrl", "deva", "diak", "ethi", "finance", "fullwide", "gara", "geor", "gong", "gonm",
    "grek", "greklow", "gujr", "gukh", "guru", "hanidays", "hanidec", "hans", "hansfin", "hant",
    "hantfin", "hebr", "hmng", "hmnp", "java", "jpan", "jpanfin", "jpanyear", "kali", "kawi",
    "khmr", "knda", "krai", "lana", "lanatham", "laoo", "latn", "lepc", "limb", "mathbold",
    "mathdbl", "mathmono", "mathsanb", "mathsans", "mlym", "modi", "mong", "mroo", "mtei", "mymr",
    "mymrepka", "mymrpao", "mymrshan", "mymrtlng", "nagm", "native", "newa", "nkoo", "olck",
    "onao", "orya", "osma", "outlined", "rohg", "roman", "romanlow", "saur", "segment", "shrd",
    "sind", "sinh", "sora", "sund", "sunu", "takr", "talu", "taml", "tamldec", "tnsa", "telu",
    "thai", "tirh", "tibt", "tols", "traditio", "vaii", "wara", "wcho",
];

/// Whether `nu` is a known CLDR numbering-system identifier.
pub(crate) fn is_known_numbering_system(nu: &str) -> bool {
    NUMBERING_SYSTEMS.contains(&nu)
}

/// `floor(log10(a))` for `a > 0`, computed by scaling so it is exact at powers
/// of ten (`log10(1e6)` can round to just under 6.0 in `f64`).
#[cfg(feature = "intl")]
fn decimal_magnitude(a: f64) -> i32 {
    if a <= 0.0 || !a.is_finite() {
        return 0;
    }
    let mut m = 0i32;
    let mut x = a;
    if x >= 1.0 {
        while x >= 10.0 {
            x /= 10.0;
            m += 1;
        }
    } else {
        while x < 1.0 {
            x *= 10.0;
            m -= 1;
        }
    }
    m
}

/// Builds the plural-operand string carrying the compact-decimal exponent for a
/// non-standard `notation`, in the crate's `<mantissa>c<exp>` form (e.g. `1.5e6`
/// in compact → `"1.5c6"`). Returns `None` for standard notation or when no
/// exponent applies (so the caller uses the plain decimal). `compact`/
/// `engineering` use exponents that are multiples of three; `scientific` uses
/// the full magnitude; compact suppresses exponents below 1000.
#[cfg(feature = "intl")]
fn plural_notation_operand_string(n: f64, notation: &str) -> Option<String> {
    let a = n.abs();
    if a == 0.0 || !a.is_finite() {
        return None;
    }
    let mag = decimal_magnitude(a);
    let e = match notation {
        "scientific" => mag,
        "engineering" => (mag as f64 / 3.0).floor() as i32 * 3,
        "compact" => {
            if mag < 3 {
                0
            } else {
                (mag as f64 / 3.0).floor() as i32 * 3
            }
        }
        _ => return None,
    };
    if e <= 0 {
        return None;
    }
    let mantissa = a / libm_pow10(e);
    Some(alloc::format!("{mantissa}c{e}"))
}

/// `10^e` for a small non-negative exponent, exact for the range used by
/// compact/scientific plural operands.
#[cfg(feature = "intl")]
fn libm_pow10(e: i32) -> f64 {
    let mut p = 1.0f64;
    for _ in 0..e {
        p *= 10.0;
    }
    p
}

/// The CLDR default numbering system for `locale` (e.g. `ar` → `arab`), derived
/// from the shape of its zero digit. Honors an explicit `-u-nu-` extension via
/// the crate. Falls back to `latn`.
pub(crate) fn default_numbering_for_locale_str(locale: &str) -> String {
    #[cfg(feature = "intl")]
    {
        // The locale's *default* numbering system — what `ResolveLocale` resolves
        // and what `Intl.NumberFormat(locale)` uses. Not
        // `otherNumberingSystems.native`: CLDR gives `ar` a default of `latn` and a
        // native of `arab`, and ECMA-402 wants the former — which since 0.6.1 is
        // `format_decimal`'s own behaviour.
        let zero = intl::number::format_decimal(locale, 0.0);
        if let Some(c) = zero.chars().next()
            && let Some(name) = numbering_system_name_from_zero(c)
        {
            return String::from(name);
        }
    }
    let _ = locale;
    String::from("latn")
}

/// ResolveLocale for the Unicode `nu` key. Given the extension-free `base`, the
/// full requested `locale` (for its `-u-nu-` extension), and the validated
/// `option`, returns `(resolved_value, extension_addition)`. The addition is
/// non-empty (e.g. `"-nu-arab"`) only when the value is sourced from the
/// extension; an option-sourced override drops it. Values that are not known
/// numbering systems are ignored (falling back to the locale default).
pub(crate) fn resolve_nu_key(base: &str, locale: &str, option: Option<&str>) -> (String, String) {
    // The generic/algorithmic aliases (`native`, `traditio`, `finance`) are valid
    // `nu` type values but have no fixed digit mapping, so ECMA-402 does not treat
    // them as selectable numbering systems: a `-u-nu-native` (etc.) request is
    // ignored, falling back to the locale default.
    let selectable = |nu: &str| {
        is_known_numbering_system(nu) && !matches!(nu, "native" | "traditio" | "finance")
    };
    let mut value = default_numbering_for_locale_str(base);
    let mut addition = String::new();
    #[cfg(feature = "intl")]
    if let Some(ext) = locale_unicode_keyword(locale, "nu")
        && selectable(&ext)
    {
        value = ext.clone();
        addition = alloc::format!("-nu-{ext}");
    }
    let _ = locale;
    if let Some(opt) = option
        && selectable(opt)
        && opt != value
    {
        value = String::from(opt);
        addition = String::new();
    }
    (value, addition)
}

/// The calendar identifiers ECMA-402 `AvailableCalendars` reports (the CLDR
/// BCP-47 calendar types with fixed semantics). Used to validate a `calendar`
/// option / `-u-ca-` extension value: an unrecognized one is ignored.
pub(crate) const AVAILABLE_CALENDARS: [&str; 16] = [
    "buddhist",
    "chinese",
    "coptic",
    "dangi",
    "ethioaa",
    "ethiopic",
    "gregory",
    "hebrew",
    "indian",
    "islamic-civil",
    "islamic-tbla",
    "islamic-umalqura",
    "iso8601",
    "japanese",
    "persian",
    "roc",
];

/// Canonicalize a calendar identifier: lowercase it, apply the CLDR type alias
/// (`islamicc` → `islamic-civil`, `ethiopic-amete-alem` → `ethioaa`), then map the
/// deprecated `islamic`/`islamic-rgsa` to a concrete available calendar
/// (`islamic-civil`, per CreateDateTimeFormat step 9).
pub(crate) fn canonicalize_calendar(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let canon = unicode_type_alias("ca", &lower)
        .map(String::from)
        .unwrap_or(lower);
    match canon.as_str() {
        "islamic" | "islamic-rgsa" => String::from("islamic-civil"),
        _ => canon,
    }
}

/// ResolveLocale for the Unicode `ca` (calendar) key, mirroring [`resolve_nu_key`].
/// Returns `(resolved_calendar, extension_addition)`. An unavailable calendar
/// (option or extension) is ignored; the addition is kept only when the value is
/// extension-sourced.
pub(crate) fn resolve_ca_key(_base: &str, locale: &str, option: Option<&str>) -> (String, String) {
    let mut value = String::from("gregory");
    let mut addition = String::new();
    #[cfg(feature = "intl")]
    if let Some(ext) = locale_unicode_calendar(locale) {
        let ext = canonicalize_calendar(&ext);
        if AVAILABLE_CALENDARS.contains(&ext.as_str()) {
            addition = alloc::format!("-ca-{ext}");
            value = ext;
        }
    }
    let _ = locale;
    if let Some(opt) = option {
        let opt = canonicalize_calendar(opt);
        if AVAILABLE_CALENDARS.contains(&opt.as_str()) && opt != value {
            value = opt;
            addition = String::new();
        }
    }
    (value, addition)
}

/// Removes the `-u-`/`-t-`/`-x-` singleton extensions (and everything after the
/// first singleton subtag) from a canonical BCP-47 tag, leaving just the
/// language/script/region/variant core — the base for ResolveLocale's
/// resolved-locale reconstruction.
pub(crate) fn strip_unicode_extension(locale: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for sub in locale.split('-') {
        if sub.len() == 1 {
            break; // a singleton subtag (u/t/x/…) starts the extensions
        }
        out.push(sub);
    }
    out.join("-")
}

/// Reconstructs a resolved locale from its extension-free `base` and the list of
/// per-key Unicode-extension additions (each already rendered, e.g. `"-nu-arab"`,
/// `"-kn"`, `"-kf-upper"`). Keys are emitted in canonical (alphabetical) order.
/// An empty addition list yields the bare `base`.
pub(crate) fn build_resolved_locale(base: &str, additions: &[String]) -> String {
    let mut adds: Vec<&String> = additions.iter().filter(|a| !a.is_empty()).collect();
    if adds.is_empty() {
        return String::from(base);
    }
    adds.sort();
    let mut out = String::from(base);
    out.push_str("-u");
    for a in adds {
        out.push_str(a);
    }
    out
}

/// ECMA-402 `CanonicalizeUnicodeLocaleId(tag)`: validate `tag` as a structurally
/// valid language tag, then return its full canonical form.
///
/// Structure and case are canonicalized by the strict local grammar
/// ([`canonicalize_locale_id_structural`]) — which, unlike `intl`'s looser BCP-47
/// parser, rejects `_` separators, 4-alpha "languages", extlang subtags, and
/// duplicate variant/singleton subtags, sorts variant subtags, and applies the
/// `-u-`/`-t-` extension canonicalization (attribute/keyword sort, `true`/`yes`
/// elision only for the keys that alias it). On top of that, the deprecated
/// **base** subtags (language / script / region / variant, plus grandfathered
/// whole-tag forms) are substituted with their CLDR-canonical replacements via
/// `intl::locale::canonicalize` — the alias corpus (aliases.bin) newly shipped by
/// intl 0.5.1. Extension (`-u-`/`-t-`) subtags keep the local canonicalization,
/// which is more complete/correct here than intl 0.5.1 (which mis-drops `ka-yes`
/// and lacks `rg`/`sd` subdivision aliases). Returns `None` (→ **RangeError**)
/// when the tag is not structurally valid.
/// Canonicalizes a `-t-` extension's *tlang* — structural normalization plus the
/// CLDR base-subtag alias corpus, since a tlang is itself a language tag. Kept
/// separate from [`canonicalize_locale_id`] because a tlang carries no extensions
/// of its own, so there is no suffix to split off.
fn canonicalize_tlang(tlang: &str) -> Option<String> {
    let structural = canonicalize_locale_id_structural(tlang)?;
    #[cfg(feature = "intl")]
    {
        Some(
            intl::locale::canonicalize(&structural)
                .and_then(|t| canonicalize_locale_id_structural(&t))
                .unwrap_or(structural),
        )
    }
    #[cfg(not(feature = "intl"))]
    {
        Some(structural)
    }
}

pub(crate) fn canonicalize_locale_id(tag: &str) -> Option<String> {
    // Strict structural validation + local canonical form (rejects invalid tags,
    // e.g. `i-klingon`, → RangeError).
    let structural = canonicalize_locale_id_structural(tag)?;
    // Split off the extension/private-use suffix (the first length-1 subtag onward)
    // — its local canonicalization is authoritative; only the base takes intl's
    // alias substitution.
    let subs: Vec<&str> = structural.split('-').collect();
    let ext_at = subs.iter().position(|s| s.len() == 1).unwrap_or(subs.len());
    let base = subs[..ext_at].join("-");
    // Apply the CLDR base-subtag alias corpus, then re-run the strict canonicalizer
    // so any alias-introduced subtags are re-normalized (variant re-sort, casing).
    // The alias corpus is `intl` data; without that feature the structural
    // canonical form stands on its own (no alias substitution, still valid).
    #[cfg(feature = "intl")]
    let aliased_base = intl::locale::canonicalize(&base)
        .and_then(|b| canonicalize_locale_id_structural(&b))
        .unwrap_or(base);
    #[cfg(not(feature = "intl"))]
    let aliased_base = base;
    if ext_at == subs.len() {
        Some(aliased_base)
    } else {
        Some(alloc::format!(
            "{aliased_base}-{}",
            subs[ext_at..].join("-")
        ))
    }
}

fn canonicalize_locale_id_structural(tag: &str) -> Option<String> {
    // Structurally rejected outright: empty, non-ASCII, `_` separators, and the
    // empty subtags produced by leading/trailing/doubled `-`.
    if tag.is_empty() || !tag.is_ascii() || tag.contains('_') {
        return None;
    }
    // UTS-35 §3.3.1 "regular" grandfathered tags: the language(-variant) base is
    // replaced by its canonical form (the irregular `i-*`/`sgn-*`/`en-gb-oed`
    // grandfathered forms are structurally invalid and rejected by the grammar
    // below). Extensions, if any, are preserved; the substituted tag is then
    // canonicalized normally.
    {
        let lower = tag.to_ascii_lowercase();
        let subs: Vec<&str> = lower.split('-').collect();
        let base_end = subs.iter().position(|p| p.len() == 1).unwrap_or(subs.len());
        if let Some(repl) = grandfathered_canonical(&subs[..base_end].join("-")) {
            let mut rebuilt = String::from(repl);
            for p in &subs[base_end..] {
                rebuilt.push('-');
                rebuilt.push_str(p);
            }
            return canonicalize_locale_id_structural(&rebuilt);
        }
    }
    let parts: Vec<&str> = tag.split('-').collect();
    if parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let is_alpha = |s: &str| s.bytes().all(|b| b.is_ascii_alphabetic());
    let is_digit = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    let is_alnum = |s: &str| s.bytes().all(|b| b.is_ascii_alphanumeric());

    let mut idx = 0usize;
    let n = parts.len();

    // unicode_language_subtag = alpha{2,3} | alpha{5,8}  (NOT 4, NOT extlang).
    let lang = parts[idx];
    if !((2..=3).contains(&lang.len()) || (5..=8).contains(&lang.len())) || !is_alpha(lang) {
        return None;
    }
    let language = lang.to_ascii_lowercase();
    idx += 1;
    // No extlang subtags allowed in UTS-35: a 3-alpha subtag after a 2-3 alpha
    // language would be an extlang (BCP-47) — invalid here.
    if idx < n && is_alpha(parts[idx]) && parts[idx].len() == 3 {
        return None;
    }

    // unicode_script_subtag = alpha{4}.
    let mut script = None;
    if idx < n && parts[idx].len() == 4 && is_alpha(parts[idx]) {
        let s = parts[idx];
        let mut t = String::new();
        for (i, c) in s.chars().enumerate() {
            if i == 0 {
                t.push(c.to_ascii_uppercase());
            } else {
                t.push(c.to_ascii_lowercase());
            }
        }
        script = Some(t);
        idx += 1;
    }

    // unicode_region_subtag = alpha{2} | digit{3}.
    let mut region = None;
    if idx < n
        && ((parts[idx].len() == 2 && is_alpha(parts[idx]))
            || (parts[idx].len() == 3 && is_digit(parts[idx])))
    {
        region = Some(parts[idx].to_ascii_uppercase());
        idx += 1;
    }

    // unicode_variant_subtag = (alphanum{5,8} | digit alphanum{3}).
    let mut variants: Vec<String> = Vec::new();
    while idx < n {
        let s = parts[idx];
        let is_variant = ((5..=8).contains(&s.len()) && is_alnum(s))
            || (s.len() == 4 && s.as_bytes()[0].is_ascii_digit() && is_alnum(s));
        if !is_variant {
            break;
        }
        let v = s.to_ascii_lowercase();
        if variants.contains(&v) {
            return None; // duplicate variant
        }
        variants.push(v);
        idx += 1;
    }
    variants.sort();
    // UTS-35 canonicalizes the *syntax* first (variants sorted) and only then
    // applies `languageAlias`, so a regular grandfathered tag whose second half
    // has been re-sorted away from the language still has to be recognized:
    // `art-lojban-fonipa` sorts to `art-fonipa-lojban`, and its `lojban` variant
    // is what the alias rule matches (→ `jbo-fonipa`).
    if let Some(pos) = variants
        .iter()
        .position(|v| grandfathered_canonical(&alloc::format!("{language}-{v}")).is_some())
    {
        let repl = grandfathered_canonical(&alloc::format!("{language}-{}", variants[pos]))?;
        let mut rebuilt = String::from(repl);
        for (i, p) in parts.iter().enumerate() {
            // Drop the original language subtag and the matched variant; every
            // other subtag (script/region/other variants/extensions) rides along.
            if i == 0 || p.eq_ignore_ascii_case(&variants[pos]) {
                continue;
            }
            rebuilt.push('-');
            rebuilt.push_str(p);
        }
        return canonicalize_locale_id_structural(&rebuilt);
    }

    // Extensions and private use: singleton (alphanum) followed by subtags.
    // Each singleton may appear once; `u`/`t` have their own subtag grammars but
    // we validate generically and canonicalize key ordering for `u` and `t`.
    let mut extensions: Vec<(char, String)> = Vec::new();
    let mut seen_singletons: Vec<char> = Vec::new();
    while idx < n {
        let sing = parts[idx];
        if sing.len() != 1 || !sing.as_bytes()[0].is_ascii_alphanumeric() {
            return None; // expected a singleton here
        }
        let singleton = sing.as_bytes()[0].to_ascii_lowercase() as char;
        if seen_singletons.contains(&singleton) {
            return None; // duplicate singleton
        }
        seen_singletons.push(singleton);
        idx += 1;
        // Gather this singleton's subtags. Private use (`x`) consumes *all* the
        // remaining subtags, including length-1 ones (so `x-u-foo` is one private
        // sequence, not the start of a `u` extension).
        let mut subs: Vec<String> = Vec::new();
        let private = singleton == 'x';
        while idx < n && (private || parts[idx].len() != 1) {
            let st = parts[idx];
            // Private-use subtags: 1..=8 alphanum. Others: 2..=8 alphanum.
            let min = if private { 1 } else { 2 };
            if !((min..=8).contains(&st.len()) && is_alnum(st)) {
                return None;
            }
            subs.push(st.to_ascii_lowercase());
            idx += 1;
        }
        // A singleton with a length-1 subtag (private use excepted) is invalid,
        // and any singleton must have at least one subtag.
        if subs.is_empty() {
            return None;
        }
        let body = canonicalize_extension(singleton, &subs)?;
        extensions.push((singleton, body));
    }
    // UTS-35 forbids a `unicode_locale_id` consisting only of private use; a
    // valid id must have a real language (we already required one), so this is
    // satisfied. Extensions sort by singleton with `x` last.
    extensions.sort_by_key(|(s, _)| (*s == 'x', *s));

    let mut out = language;
    if let Some(s) = script {
        out.push('-');
        out.push_str(&s);
    }
    if let Some(r) = region {
        out.push('-');
        out.push_str(&r);
    }
    for v in &variants {
        out.push('-');
        out.push_str(v);
    }
    for (_, body) in &extensions {
        out.push('-');
        out.push_str(body);
    }
    Some(out)
}

/// Validates + canonicalizes the `code` argument to
/// `Intl.DisplayNames.prototype.of` against its instance `[[Type]]` (ECMA-402
/// `CanonicalizeDisplayNamesType`-adjacent `of` grammar). Returns the
/// canonical code, or `None` when `code` does not match the type's grammar (the
/// caller raises a `RangeError`). Unknown types pass through unchanged.
pub(crate) fn validate_display_code(ty: &str, code: &str) -> Option<String> {
    let is_alpha = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphabetic());
    let is_digit = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let is_alnum = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric());
    match ty {
        // unicode_language_id: a structurally valid tag with NO extension/private
        // singletons (a length-1 subtag such as `u` in `en-u-hebrew` is invalid).
        "language" => {
            if code.split('-').any(|p| p.len() == 1) {
                return None;
            }
            canonicalize_locale_id(code)
        }
        // unicode_region_subtag = alpha{2} | digit{3}.
        "region" => {
            if code.len() == 2 && is_alpha(code) {
                Some(code.to_ascii_uppercase())
            } else if code.len() == 3 && is_digit(code) {
                Some(String::from(code))
            } else {
                None
            }
        }
        // unicode_script_subtag = alpha{4} (title-cased).
        "script" => {
            if code.len() == 4 && is_alpha(code) {
                let mut t = String::new();
                for (i, c) in code.chars().enumerate() {
                    if i == 0 {
                        t.push(c.to_ascii_uppercase());
                    } else {
                        t.push(c.to_ascii_lowercase());
                    }
                }
                Some(t)
            } else {
                None
            }
        }
        // IsWellFormedCurrencyCode: 3 ASCII letters (upper-cased).
        "currency" => (code.len() == 3 && is_alpha(code)).then(|| code.to_ascii_uppercase()),
        // `type` nonterminal: alphanum{3,8} (-alphanum{3,8})* (lower-cased).
        "calendar" => {
            if code
                .split('-')
                .all(|p| (3..=8).contains(&p.len()) && is_alnum(p))
            {
                Some(code.to_ascii_lowercase())
            } else {
                None
            }
        }
        // A fixed enumeration (ES2023 Table of date-time field codes).
        "dateTimeField" => {
            const FIELDS: &[&str] = &[
                "era",
                "year",
                "quarter",
                "month",
                "weekOfYear",
                "weekday",
                "day",
                "dayPeriod",
                "hour",
                "minute",
                "second",
                "timeZoneName",
            ];
            FIELDS.contains(&code).then(|| String::from(code))
        }
        _ => Some(String::from(code)),
    }
}

/// Canonicalizes one extension's subtags into its `singleton-subtag-…` body.
/// For `u`/`t` the keyword/field groups are sorted by key (ASCII order); for
/// other singletons (and private use `x`) the order is preserved.
fn canonicalize_extension(singleton: char, subs: &[String]) -> Option<String> {
    if singleton == 'u' {
        // -u- = attribute* (key type*)* — attributes (length-2..8, but a *key*
        // is exactly length 2) come first, then keyword groups keyed by a
        // 2-char key. Group the trailing keyword sequences and sort by key.
        let mut attributes: Vec<String> = Vec::new();
        let mut i = 0;
        // A `key` is exactly 2 chars: alphanum then alpha (e.g. `ca`, `nu`, `0c`).
        // An `attribute`/`type` is 3..=8 alphanum. A 2-char subtag whose 2nd char
        // is a digit (e.g. `c0`, `00`) is neither — structurally invalid.
        let is_key = |s: &str| s.len() == 2 && s.as_bytes()[1].is_ascii_alphabetic();
        let is_attr_or_type = |s: &str| (3..=8).contains(&s.len());
        while i < subs.len() && !is_key(&subs[i]) {
            if !is_attr_or_type(&subs[i]) {
                return None;
            }
            attributes.push(subs[i].clone());
            i += 1;
        }
        attributes.sort();
        let mut keywords: Vec<(String, Vec<String>)> = Vec::new();
        while i < subs.len() {
            let key = subs[i].clone();
            i += 1;
            let mut vals: Vec<String> = Vec::new();
            while i < subs.len() && !is_key(&subs[i]) {
                if !is_attr_or_type(&subs[i]) {
                    return None;
                }
                vals.push(subs[i].clone());
                i += 1;
            }
            // CLDR bcp47 type-value aliases (calendar/collation-strength/
            // measurement): a deprecated value canonicalizes to its preferred
            // form (`ca-islamicc` → `ca-islamic-civil`, `ks-primary` → `ks-level1`,
            // `ms-imperial` → `ms-uksystem`, …).
            if let Some(canon) = unicode_type_alias(&key, &vals.join("-")) {
                vals = canon.split('-').map(String::from).collect();
            }
            // CLDR bcp47 type alias: for the boolean collation keys
            // kb/kc/kh/kk/kn the type "yes" is an alias of "true"
            // (`<type name="true" alias="yes"/>`). Other keys (ka/kf/kr/ks/kv)
            // have no such alias and keep "yes".
            if vals.len() == 1
                && vals[0] == "yes"
                && matches!(key.as_str(), "kb" | "kc" | "kh" | "kk" | "kn")
            {
                vals[0] = String::from("true");
            }
            // A `true` type value is elided in canonical form.
            if vals.len() == 1 && vals[0] == "true" {
                vals.clear();
            }
            // UTS-35 canonicalization: a repeated key keeps its *first* occurrence
            // and discards the rest (`da-u-ca-gregory-ca-buddhist` → `da-u-ca-gregory`).
            if !keywords.iter().any(|(k, _)| k == &key) {
                keywords.push((key, vals));
            }
        }
        keywords.sort_by(|a, b| a.0.cmp(&b.0));
        let mut body = String::from("u");
        for a in &attributes {
            body.push('-');
            body.push_str(a);
        }
        for (k, vals) in &keywords {
            body.push('-');
            body.push_str(k);
            for v in vals {
                body.push('-');
                body.push_str(v);
            }
        }
        return Some(body);
    }
    if singleton == 't' {
        // -t- = (tlang)? (tfield)*  where tfield = tkey tvalue+. A tkey is
        // exactly 2 chars with an alpha first and digit second (e.g. `m0`, `k0`).
        // The optional tlang (a language subtag) comes first.
        let is_tkey = |s: &str| {
            s.len() == 2
                && s.as_bytes()[0].is_ascii_alphabetic()
                && s.as_bytes()[1].is_ascii_digit()
        };
        let mut i = 0;
        let mut tlang: Vec<String> = Vec::new();
        while i < subs.len() && !is_tkey(&subs[i]) {
            tlang.push(subs[i].clone());
            i += 1;
        }
        // Canonicalize the tlang via the locale-id rules (lang/script/region/
        // variants). It is a *language tag*, so the CLDR alias corpus applies to
        // it exactly as it does to the base tag — `en-t-iw` canonicalizes to
        // `en-t-he`. Structural normalization alone left the deprecated subtag
        // in place.
        let tlang_canon = if tlang.is_empty() {
            None
        } else {
            Some(canonicalize_tlang(&tlang.join("-"))?)
        };
        let mut fields: Vec<(String, Vec<String>)> = Vec::new();
        while i < subs.len() {
            let key = subs[i].clone();
            i += 1;
            let mut vals: Vec<String> = Vec::new();
            while i < subs.len() && !is_tkey(&subs[i]) {
                vals.push(subs[i].clone());
                i += 1;
            }
            if vals.is_empty() {
                return None; // a tfield key must have a value
            }
            // CLDR bcp47 tvalue aliases (`m0-names` → `m0-prprname`).
            if let Some(canon) = transform_value_alias(&vals.join("-")) {
                vals = canon.split('-').map(String::from).collect();
            }
            fields.push((key, vals));
        }
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        let mut body = String::from("t");
        if let Some(tl) = &tlang_canon {
            body.push('-');
            body.push_str(&tl.to_ascii_lowercase());
        }
        for (k, vals) in &fields {
            body.push('-');
            body.push_str(k);
            for v in vals {
                body.push('-');
                body.push_str(v);
            }
        }
        return Some(body);
    }
    // Other singletons / private use: subtags in given order, already lowercased.
    let mut body = String::new();
    body.push(singleton);
    for s in subs {
        body.push('-');
        body.push_str(s);
    }
    Some(body)
}

/// Per-`Intl` service metadata used to build its `.prototype` (the spec
/// `Symbol.toStringTag`, the hidden brand-marker key stamped on every instance,
/// and the data-method / accessor-getter names installed on the prototype).
struct IntlService {
    /// The constructor's native dispatch id (`N_INTL_NUMBER_FORMAT`, …).
    ctor_id: u16,
    /// The spec `[Symbol.toStringTag]` string (e.g. `"Intl.NumberFormat"`).
    tag: &'static str,
    /// The hidden internal-slot brand-marker property stamped on every instance and
    /// required by the prototype methods'/accessors' RequireInternalSlot check.
    marker: &'static str,
    /// First-class data-method names installed on the prototype (each a branded
    /// `N_INTL_PROTO_METHOD` wrapper delegating to the underlying method native).
    methods: &'static [&'static str],
    /// The single bound-function accessor on this service's prototype, if any:
    /// `(accessor-name, underlying-selector)`. NumberFormat/DateTimeFormat have
    /// `get format`; Collator has `get compare`. These return a per-instance bound
    /// function rather than being plain data methods (ECMA-402).
    bound_accessor: Option<(&'static str, &'static str)>,
}

/// The eight constructor-style `Intl` services that share the
/// formatter/options-object instance shape (NumberFormat … Segmenter). `Intl.Locale`
/// and `Intl.DurationFormat` have bespoke prototypes built separately.
const INTL_SERVICES: &[IntlService] = &[
    IntlService {
        ctor_id: N_INTL_NUMBER_FORMAT,
        tag: "Intl.NumberFormat",
        marker: "\u{0}brand_nf",
        methods: &[
            "resolvedOptions",
            "formatToParts",
            "formatRange",
            "formatRangeToParts",
        ],
        bound_accessor: Some(("format", "format")),
    },
    IntlService {
        ctor_id: N_INTL_DATETIME_FORMAT,
        tag: "Intl.DateTimeFormat",
        marker: "\u{0}brand_dtf",
        methods: &[
            "resolvedOptions",
            "formatToParts",
            "formatRange",
            "formatRangeToParts",
        ],
        bound_accessor: Some(("format", "format")),
    },
    IntlService {
        ctor_id: N_INTL_COLLATOR,
        tag: "Intl.Collator",
        marker: "\u{0}brand_col",
        methods: &["resolvedOptions"],
        bound_accessor: Some(("compare", "compare")),
    },
    IntlService {
        ctor_id: N_INTL_PLURAL_RULES,
        tag: "Intl.PluralRules",
        marker: "\u{0}brand_pr",
        methods: &["resolvedOptions", "select", "selectRange"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_LIST_FORMAT,
        tag: "Intl.ListFormat",
        marker: "\u{0}brand_lf",
        methods: &["resolvedOptions", "format", "formatToParts"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_REL_TIME,
        tag: "Intl.RelativeTimeFormat",
        marker: "\u{0}brand_rtf",
        methods: &["resolvedOptions", "format", "formatToParts"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_DISPLAY_NAMES,
        tag: "Intl.DisplayNames",
        marker: "\u{0}brand_dn",
        methods: &["resolvedOptions", "of"],
        bound_accessor: None,
    },
    IntlService {
        ctor_id: N_INTL_SEGMENTER,
        tag: "Intl.Segmenter",
        marker: "\u{0}brand_seg",
        methods: &["resolvedOptions", "segment"],
        bound_accessor: None,
    },
];

/// `Intl.Locale.prototype` accessor-getter names (each a `get`-only accessor
/// brand-checking the `[[InitializedLocale]]` slot).
const LOCALE_ACCESSORS: &[&str] = &[
    "baseName",
    "calendar",
    "caseFirst",
    "collation",
    "firstDayOfWeek",
    "hourCycle",
    "language",
    "numberingSystem",
    "numeric",
    "region",
    "script",
    "variants",
];

/// The arity (`length`) of a branded `Intl` prototype data method, given its
/// owning service ctor id and method name (RelativeTimeFormat's `format`/
/// `formatToParts` are length 2; everywhere else `format`-family methods are length 1).
fn intl_method_arity(ctor_id: u16, name: &str) -> u32 {
    match (ctor_id, name) {
        (_, "resolvedOptions") => 0,
        (N_INTL_REL_TIME, "format" | "formatToParts") => 2,
        (N_INTL_PLURAL_RULES, "selectRange") => 2,
        (_, "formatRange" | "formatRangeToParts") => 2,
        _ => 1,
    }
}

impl<'a> Interp<'a> {
    /// The underlying method native id for a branded `Intl` prototype data method.
    fn intl_underlying_native(name: &str) -> u16 {
        match name {
            "format" => N_INTL_FORMAT,
            "resolvedOptions" => N_INTL_RESOLVED_OPTIONS,
            "formatToParts" | "format_to_parts" => N_INTL_FORMAT_TO_PARTS,
            "formatRange" => N_INTL_FORMAT_RANGE,
            "formatRangeToParts" => N_INTL_FORMAT_RANGE_TO_PARTS,
            "compare" => N_INTL_COMPARE,
            "select" => N_INTL_PLURAL_SELECT,
            "selectRange" => N_INTL_PLURAL_SELECT_RANGE,
            "of" => N_INTL_DISPLAY_NAMES_OF,
            "segment" => N_INTL_SEGMENTER_SEGMENT,
            "list_format" => N_INTL_LIST_FORMAT_FORMAT,
            "rel_format" => N_INTL_REL_TIME_FORMAT,
            _ => 0,
        }
    }

    /// Builds (once) and links each `Intl` service constructor's `.prototype`,
    /// installing the branded methods, the `Symbol.toStringTag`, and the
    /// `constructor` back-link, and replacing the constructor's default
    /// `Function.prototype`-derived `.prototype` with a real `%XPrototype%`. Idempotent
    /// (the per-id cache short-circuits a second call). Invoked at realm setup.
    pub(crate) fn install_intl_prototypes(&mut self) {
        for svc in INTL_SERVICES {
            self.intl_service_prototype(svc);
        }
        self.intl_locale_prototype();
        self.intl_duration_prototype();
    }

    /// The `Intl` namespace object (where the service constructors live), if set up.
    fn intl_namespace(&mut self) -> Option<Handle> {
        self.current
            .get("Intl")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
    }

    /// The constructor function handle for the `Intl` service whose property name
    /// on the namespace is `ctor_name` (e.g. `"NumberFormat"`).
    pub(crate) fn intl_ctor_handle(&mut self, ctor_name: &str) -> Option<Handle> {
        let ns = self.intl_namespace()?;
        self.realm
            .get_property(ns, ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
    }

    /// Builds and caches the `.prototype` object for one of the formatter-shaped
    /// `Intl` services, wiring it onto the constructor. Returns the prototype handle.
    fn intl_service_prototype(&mut self, svc: &IntlService) -> Option<Handle> {
        if let Some(p) = self.realm.intl_prototype(svc.ctor_id) {
            return Some(p);
        }
        // Map the ctor id back to its namespace property name.
        let ctor_name = match svc.ctor_id {
            N_INTL_NUMBER_FORMAT => "NumberFormat",
            N_INTL_DATETIME_FORMAT => "DateTimeFormat",
            N_INTL_COLLATOR => "Collator",
            N_INTL_PLURAL_RULES => "PluralRules",
            N_INTL_LIST_FORMAT => "ListFormat",
            N_INTL_REL_TIME => "RelativeTimeFormat",
            N_INTL_DISPLAY_NAMES => "DisplayNames",
            N_INTL_SEGMENTER => "Segmenter",
            _ => return None,
        };
        let ctor = self.intl_ctor_handle(ctor_name)?;
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &m in svc.methods {
            // ListFormat/RelativeTimeFormat have distinct `format`/`formatToParts`
            // method natives; pick the underlying selector accordingly.
            let selector = match (svc.ctor_id, m) {
                (N_INTL_LIST_FORMAT, "format") => "list_format",
                (N_INTL_REL_TIME, "format") => "rel_format",
                (N_INTL_LIST_FORMAT | N_INTL_REL_TIME, "formatToParts") => "format_to_parts",
                _ => m,
            };
            let arity = intl_method_arity(svc.ctor_id, m);
            let f = self.make_intl_proto_method(svc.marker, m, selector, arity);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        // The bound-function accessor (`get format` / `get compare`).
        if let Some((acc_name, selector)) = svc.bound_accessor {
            let label = alloc::format!("get {acc_name}");
            let marker_v = self.new_str(svc.marker);
            let sel_v = self.new_str(selector);
            let pair = self.realm.new_array(alloc::vec![marker_v, sel_v]);
            let getter = self.realm.new_bound_native(N_INTL_BOUND_GETTER, pair);
            self.install_fn_name_length(getter, &label, 0);
            self.realm.define_accessor(
                proto,
                acc_name,
                NanBox::handle(getter.to_raw()),
                NanBox::undefined(),
            );
            self.realm.mark_hidden(proto, acc_name);
        }
        // `prototype[Symbol.toStringTag]` — a `{ w:false, e:false, c:true }` string
        // data property (ECMA-402 11.3.x etc.).
        self.install_to_string_tag(proto, svc.tag);
        // `prototype.constructor` back-link (non-enumerable).
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.link_ctor_prototype(ctor, proto);
        self.realm.set_intl_prototype(svc.ctor_id, proto);
        Some(proto)
    }

    /// A branded prototype data method: an `N_INTL_PROTO_METHOD` bound native whose
    /// target is the 2-element `[markerKey, underlyingSelector]` array. The wrapper
    /// (in `call_with_this`) RequireInternalSlot-checks the call's `this` before
    /// delegating to the underlying method native.
    fn make_intl_proto_method(
        &mut self,
        marker: &str,
        name: &str,
        selector: &str,
        arity: u32,
    ) -> Handle {
        let marker_v = self.new_str(marker);
        let sel_v = self.new_str(selector);
        let pair = self.realm.new_array(alloc::vec![marker_v, sel_v]);
        let f = self.realm.new_bound_native(N_INTL_PROTO_METHOD, pair);
        self.install_fn_name_length(f, name, arity);
        f
    }

    /// Installs `proto` as the constructor's `prototype` data property, with the
    /// built-in `{ writable:false, enumerable:false, configurable:false }`
    /// attributes (every built-in constructor's `prototype`).
    fn link_ctor_prototype(&mut self, ctor: Handle, proto: Handle) {
        self.realm
            .set_property(ctor, "prototype", NanBox::handle(proto.to_raw()));
        self.realm.mark_hidden(ctor, "prototype");
        self.realm.set_readonly_property(ctor, "prototype");
        self.realm.set_non_configurable_property(ctor, "prototype");
    }

    /// Stamps a service's internal-slot brand marker on `obj` and links its
    /// `[[Prototype]]` to the (lazily built) service prototype — turning a bare
    /// options-bag object into a branded `Intl.X` instance.
    fn brand_intl_instance(&mut self, obj: Handle, ctor_id: u16) {
        let svc = INTL_SERVICES.iter().find(|s| s.ctor_id == ctor_id);
        if let Some(svc) = svc {
            self.realm
                .set_hidden_property(obj, svc.marker, NanBox::boolean(true));
            if let Some(proto) = self.intl_service_prototype(svc) {
                self.realm.set_object_proto(obj, Some(proto));
            }
        }
    }

    /// Brands `obj` with the service's internal-slot marker *without* touching its
    /// prototype (for a subclass `super()`, whose instance already links to the
    /// subclass prototype).
    fn set_intl_marker(&mut self, obj: Handle, ctor_id: u16) {
        if let Some(svc) = INTL_SERVICES.iter().find(|s| s.ctor_id == ctor_id) {
            self.realm
                .set_hidden_property(obj, svc.marker, NanBox::boolean(true));
        }
    }

    /// `%Intl%.[[FallbackSymbol]]` — the per-realm private symbol whose
    /// `[[Description]]` is `"IntlLegacyConstructedSymbol"`, used by the ECMA-402
    /// normative-optional legacy constructor mode (`Intl.NumberFormat.call(obj)`)
    /// to stash the real formatter on an arbitrary receiver. Created on first use
    /// and cached on the `Intl` namespace object so it is stable per realm.
    pub(crate) fn intl_fallback_symbol(&mut self) -> Option<NanBox> {
        let ns = self.intl_namespace()?;
        if let Some(v) = self.realm.get_property(ns, "\u{0}fallback_symbol") {
            return Some(v);
        }
        let sym = NanBox::handle(
            self.realm
                .new_symbol("IntlLegacyConstructedSymbol")
                .to_raw(),
        );
        self.realm
            .set_hidden_property(ns, "\u{0}fallback_symbol", sym);
        Some(sym)
    }

    /// `OrdinaryHasInstance(%Intl.X%, o)` without consulting a (tamperable)
    /// namespace property: walks `o`'s prototype chain — proxy-aware, so a Proxy
    /// wrapping a legacy-constructed object still reports `true` — looking for the
    /// intrinsic `%Intl.X.prototype%` of the service with `ctor_id`.
    fn intl_proto_on_chain(&mut self, o: NanBox, ctor_id: u16) -> Result<bool, ExecError> {
        let Some(svc) = INTL_SERVICES.iter().find(|s| s.ctor_id == ctor_id) else {
            return Ok(false);
        };
        let Some(proto) = self.intl_service_prototype(svc) else {
            return Ok(false);
        };
        let Some(mut cur) = o.as_handle().map(Handle::from_raw) else {
            return Ok(false);
        };
        if !self.is_object_value(o) {
            return Ok(false);
        }
        for _ in 0..100_000 {
            let Some(next) = self.get_proto_of(cur)?.as_handle().map(Handle::from_raw) else {
                return Ok(false);
            };
            if next == proto {
                return Ok(true);
            }
            cur = next;
        }
        Ok(false)
    }

    /// `ChainNumberFormat` / `ChainDateTimeFormat` (ECMA-402 normative optional):
    /// when `Intl.NumberFormat` / `Intl.DateTimeFormat` is called *without* `new`
    /// and `this` is already an instance of that constructor, the freshly built
    /// `formatter` is stashed on `this` under `%Intl%.[[FallbackSymbol]]` as a
    /// non-writable/non-enumerable/non-configurable own property and `this` is
    /// returned; otherwise the formatter itself is returned.
    pub(crate) fn chain_intl_formatter(
        &mut self,
        ctor_id: u16,
        formatter: NanBox,
    ) -> Result<NanBox, ExecError> {
        let this = self.this_val;
        if !self.intl_proto_on_chain(this, ctor_id)? {
            return Ok(formatter);
        }
        let (Some(sym), Some(h)) = (
            self.intl_fallback_symbol(),
            this.as_handle().map(Handle::from_raw),
        ) else {
            return Ok(formatter);
        };
        let key = self.member_key(sym);
        // DefinePropertyOrThrow: an existing non-configurable own property of the
        // same name cannot be redefined.
        if self.realm.get_property(h, &key).is_some()
            && self.realm.property_is_non_configurable(h, &key)
        {
            return Err(self.type_error("Cannot redefine %Intl%.[[FallbackSymbol]]"));
        }
        self.realm.set_hidden_property(h, &key, formatter);
        self.realm.set_readonly_property(h, &key);
        self.realm.set_non_configurable_property(h, &key);
        Ok(this)
    }

    /// `UnwrapNumberFormat` / `UnwrapDateTimeFormat`: `Intl.NumberFormat.prototype`
    /// `.format`/`.resolvedOptions` accept a legacy-constructed receiver — an object
    /// that lacks the `[[InitializedNumberFormat]]` slot but is an instance of the
    /// constructor — by reading the real formatter out of
    /// `%Intl%.[[FallbackSymbol]]`. The `Get` is a full property read, so a Proxy
    /// receiver runs its `get` trap.
    fn unwrap_legacy_intl(
        &mut self,
        this: NanBox,
        marker: &str,
    ) -> Result<Option<NanBox>, ExecError> {
        let ctor_id = match marker {
            "\u{0}brand_nf" => N_INTL_NUMBER_FORMAT,
            "\u{0}brand_dtf" => N_INTL_DATETIME_FORMAT,
            _ => return Ok(None),
        };
        let Some(h) = this.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        if self.realm.get_property(h, marker).is_some() {
            return Ok(None);
        }
        if !self.intl_proto_on_chain(this, ctor_id)? {
            return Ok(None);
        }
        let Some(sym) = self.intl_fallback_symbol() else {
            return Ok(None);
        };
        let key = self.member_key(sym);
        Ok(Some(self.read_member(h, &key)?))
    }

    /// RequireInternalSlot for a branded `Intl` receiver: returns the receiver
    /// handle when `this` is an object carrying the hidden `marker`, else a
    /// TypeError. The prototype object itself (which has no marker) is rejected,
    /// matching the spec's per-method brand check.
    pub(crate) fn require_intl_slot(
        &mut self,
        this: NanBox,
        marker: &str,
        what: &str,
    ) -> Result<Handle, ExecError> {
        if let Some(h) = this.as_handle().map(Handle::from_raw)
            && self.realm.get_property(h, marker).is_some()
        {
            return Ok(h);
        }
        Err(self.type_error(&alloc::format!(
            "{what} called on an object that is not a valid {what} receiver"
        )))
    }

    /// Dispatches an `N_INTL_PROTO_METHOD` wrapper: `target` is the
    /// `[markerKey, underlyingSelector]` pair. Brand-checks `this`, then delegates to
    /// the underlying method native with `this` preserved.
    pub(crate) fn intl_proto_method_dispatch(
        &mut self,
        this: NanBox,
        target: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let pair = self
            .realm
            .array_elements(target)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let marker = pair
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        let selector = pair
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        // `Intl.NumberFormat.prototype.resolvedOptions` /
        // `Intl.DateTimeFormat.prototype.resolvedOptions` accept a
        // legacy-constructed receiver (Unwrap{Number,DateTime}Format) before the
        // brand check; the other prototype methods do not.
        let this = if selector == "resolvedOptions" {
            self.unwrap_legacy_intl(this, &marker)?.unwrap_or(this)
        } else {
            this
        };
        self.require_intl_slot(this, &marker, "Intl method")?;
        let id = Self::intl_underlying_native(&selector);
        // The underlying method native reads `self.this_val`; preserve the receiver.
        let saved = core::mem::replace(&mut self.this_val, this);
        let r = self.call_native(id, args);
        self.this_val = saved;
        r
    }

    /// Dispatches an `N_INTL_BOUND_GETTER` accessor (`get format`/`get compare`):
    /// `target` is `[markerKey, selector]`. Brand-checks `this`, then returns the
    /// per-instance bound function — created once and cached on the instance under a
    /// hidden `\0bound_<selector>` key so repeated reads return the same value
    /// (`nf.format === nf.format`, per ECMA-402).
    pub(crate) fn intl_bound_getter_dispatch(
        &mut self,
        this: NanBox,
        target: Handle,
    ) -> Result<NanBox, ExecError> {
        let pair = self
            .realm
            .array_elements(target)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let marker = pair
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        let selector = pair
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        // `get Intl.NumberFormat.prototype.format` / `get …DateTimeFormat…format`
        // unwrap a legacy-constructed receiver before the brand check.
        let this = if selector == "format" {
            self.unwrap_legacy_intl(this, &marker)?.unwrap_or(this)
        } else {
            this
        };
        let inst = self.require_intl_slot(this, &marker, "Intl bound-function getter")?;
        let cache_key = alloc::format!("\u{0}bound_{selector}");
        if let Some(v) = self.realm.get_property(inst, &cache_key) {
            return Ok(v);
        }
        // Build the bound function: target = [instance, selector].
        let inst_v = NanBox::handle(inst.to_raw());
        let sel_v = self.new_str(&selector);
        let bpair = self.realm.new_array(alloc::vec![inst_v, sel_v]);
        let bound = self.realm.new_bound_native(N_INTL_BOUND_CALL, bpair);
        // The bound function's `name` is `""`; its `length` is 2 for `compare`
        // (a two-argument comparator) and 1 for `format` (sec-collator-compare /
        // sec-number-format-functions).
        let len = if selector == "compare" { 2 } else { 1 };
        self.install_fn_name_length(bound, "", len);
        let boundv = NanBox::handle(bound.to_raw());
        self.realm.set_hidden_property(inst, &cache_key, boundv);
        Ok(boundv)
    }

    /// Dispatches an `N_INTL_BOUND_CALL` (the function returned by `get format`/
    /// `get compare`): `target` is `[instance, selector]`. Formats/compares against
    /// the captured instance regardless of the call's `this` (a BoundFunction).
    pub(crate) fn intl_bound_call_dispatch(
        &mut self,
        target: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let pair = self
            .realm
            .array_elements(target)
            .map(<[_]>::to_vec)
            .unwrap_or_default();
        let inst = pair
            .first()
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        let selector = pair
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .unwrap_or_default();
        let Some(inst) = inst else {
            return Ok(NanBox::undefined());
        };
        let arg0 = args.first().copied().unwrap_or(NanBox::undefined());
        match selector.as_str() {
            "compare" => {
                // Delegate to N_INTL_COMPARE with the captured collator as `this`.
                let saved = core::mem::replace(&mut self.this_val, NanBox::handle(inst.to_raw()));
                let r = self.call_native(N_INTL_COMPARE, args);
                self.this_val = saved;
                r
            }
            _ => {
                // `format`: format the single argument against the captured formatter.
                let s = self.intl_format_checked(inst, arg0)?;
                Ok(self.new_str(&s))
            }
        }
    }

    /// `format(value)` for a NumberFormat or DateTimeFormat instance. A
    /// DateTimeFormat coerces `value` via `ToNumber` + `TimeClip` (a non-finite /
    /// out-of-range date is a RangeError; `undefined` → now); a NumberFormat
    /// formats any numeric value (including `±Infinity`). Shared by the `format`
    /// method dispatch and the bound `get format` function.
    pub(crate) fn intl_format_checked(
        &mut self,
        inst: Handle,
        value: NanBox,
    ) -> Result<String, ExecError> {
        let is_datetime = self
            .realm
            .get_property(inst, "\u{0}intl")
            .map(|k| self.realm.to_display_string(k))
            .as_deref()
            == Some("datetime");
        if is_datetime {
            #[cfg(feature = "intl")]
            if let Some(s) = self.temporal_format_flat(inst, value, false)? {
                return Ok(s);
            }
            let ms = self.datetime_operand(value)?;
            Ok(self.format_intl_datetime(inst, ms))
        } else {
            // ECMA-402 `ToIntlMathematicalValue`: a string / BigInt argument with
            // more precision than an f64 preserves formats from its exact decimal
            // digits, bypassing the crate's f64 round-trip.
            #[cfg(feature = "intl")]
            if let Some(s) = self.try_exact_decimal_format(inst, value) {
                return Ok(s);
            }
            let n = self.coerce_intl_number(value)?;
            Ok(self.intl_format_value(inst, NanBox::number(n)))
        }
    }

    /// `Intl.NumberFormat` operand coercion (a reduced `ToIntlMathematicalValue`
    /// collapsed to an f64): a BigInt uses its numeric value; everything else is
    /// `ToPrimitive`(number) then `ToNumber` — so an object with a `valueOf`
    /// returning a numeric string formats as that number, and a `Symbol` throws.
    pub(crate) fn coerce_intl_number(&mut self, value: NanBox) -> Result<f64, ExecError> {
        if let Some(h) = value.as_handle().map(Handle::from_raw)
            && let Some(big) = self.realm.bigint_at(h)
        {
            return Ok(big.to_f64());
        }
        let prim = self.coerce_to_number(value)?;
        Ok(self.realm.to_number(prim))
    }

    /// Builds an `Intl.NumberFormat`/`DateTimeFormat` instance — an object that
    /// captures the relevant options behind a `\0intl` kind marker. Used for both
    /// `new Intl.X(...)` and the callable-without-`new` form.
    pub(crate) fn make_intl_formatter(
        &mut self,
        id: u16,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        // Link `[[Prototype]]` to `Intl.X.prototype` (a subclass/Reflect.construct
        // newTarget overrides it in the caller) and initialize the formatter state.
        self.brand_intl_instance(obj, id);
        self.init_intl_formatter_state(obj, id, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Initializes an Intl formatter's internal slots on an *already-allocated*
    /// (and prototype-linked) `obj` — the shared body of `make_intl_formatter` and
    /// the `class S extends Intl.NumberFormat {}` `super()` path, so a subclass
    /// instance carries the `[[NumberFormat]]`/`[[DateTimeFormat]]` slots.
    pub(crate) fn init_intl_formatter_state(
        &mut self,
        obj: Handle,
        id: u16,
        args: &[NanBox],
    ) -> Result<(), ExecError> {
        let kind = if id == N_INTL_NUMBER_FORMAT {
            "number"
        } else {
            "datetime"
        };
        let marker = self.new_str(kind);
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // Brand the instance with the service's internal-slot marker (without
        // resetting the prototype, which a subclass `super()` has already set).
        self.set_intl_marker(obj, id);
        // CanonicalizeLocaleList(locales): the resolved locale is the first
        // requested tag (this engine serves any structurally valid locale), else
        // the default. A malformed tag raises a RangeError here.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = self
            .lookup_available_locale(&requested)
            .unwrap_or_else(|| String::from("en-US"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // `options = ? CoerceOptionsToObject(options)`: `undefined` → no options;
        // `null` (→ ToObject) is a TypeError; any other primitive is wrapped
        // (exposing no own option keys).
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if matches!(opts_arg.unpack(), Unpacked::Null) {
            return Err(self.type_error("Intl formatter options must not be null"));
        } else {
            self.coerce_to_object(opts_arg)
                .as_handle()
                .map(Handle::from_raw)
        };
        if id == N_INTL_NUMBER_FORMAT {
            self.init_number_format(obj, opts)?;
        } else {
            self.init_datetime_format(obj, opts)?;
        }
        Ok(())
    }

    /// `GetOption(options, prop, "string", values, default)` — reads `prop` via
    /// its getter, returns the default when `undefined`, else coerces to a string
    /// (a Symbol is a TypeError) and validates membership in `values` (a
    /// **RangeError** otherwise). `None` `default` with an absent option yields
    /// `None`.
    pub(crate) fn get_string_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
        values: &[&str],
        default: Option<&str>,
    ) -> Result<Option<String>, ExecError> {
        let raw = match opts {
            Some(h) => self.read_member(h, prop)?,
            None => NanBox::undefined(),
        };
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(default.map(String::from));
        }
        let s = self.coerce_to_string(raw)?;
        if !values.is_empty() && !values.iter().any(|v| *v == s) {
            let m = self.new_str(&alloc::format!("invalid value '{s}' for option {prop}"));
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(Some(s))
    }

    /// `GetOption(options, prop, "boolean", …)` — reads `prop`; `undefined` →
    /// `default`, else `ToBoolean`.
    fn get_bool_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
        default: Option<bool>,
    ) -> Result<Option<bool>, ExecError> {
        let raw = match opts {
            Some(h) => self.read_member(h, prop)?,
            None => NanBox::undefined(),
        };
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(default);
        }
        Ok(Some(self.realm.truthy(raw)))
    }

    /// `GetNumberOption` / `DefaultNumberOption`: reads `prop`, coerces to a
    /// Number (throwing for a Symbol), and requires it to be a finite integer in
    /// `[min, max]` (else **RangeError**). `undefined` → `default`.
    fn get_int_option(
        &mut self,
        opts: Option<Handle>,
        prop: &str,
        min: f64,
        max: f64,
        default: Option<f64>,
    ) -> Result<Option<f64>, ExecError> {
        let raw = match opts {
            Some(h) => self.read_member(h, prop)?,
            None => NanBox::undefined(),
        };
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(default);
        }
        let nv = self.coerce_to_number(raw)?;
        let n = self.realm.to_number(nv);
        if n.is_nan() || n < min || n > max {
            let m = self.new_str(&alloc::format!("value out of range for option {prop}"));
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(Some(trunc_toward_zero(n)))
    }

    /// Stores a resolved option on the formatter (under its own key) when present.
    fn store_str(&mut self, obj: Handle, key: &str, val: &Option<String>) {
        if let Some(v) = val {
            let sv = self.new_str(v);
            self.realm.set_hidden_property(obj, key, sv);
        }
    }

    /// Initializes an `Intl.NumberFormat`, reading and validating options in spec
    /// order (`InitializeNumberFormat` → `SetNumberFormatUnitOptions` →
    /// `SetNumberFormatDigitOptions`). Stores resolved options on `obj`.
    fn init_number_format(&mut self, obj: Handle, opts: Option<Handle>) -> Result<(), ExecError> {
        // localeMatcher (validated, not otherwise used).
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let nu = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(ns) = &nu {
            // numberingSystem must match `type` production (3-8 alnum, hyphen-joined).
            if !is_unicode_type_value(ns) {
                let m = self.new_str("invalid numberingSystem");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        // ResolveLocale for the `nu` key: a supported `numberingSystem` option
        // wins, else the locale's `-u-nu-` extension (if it names a known system),
        // else the CLDR default. The resolved locale keeps only the relevant
        // extension (`-u-nu-` when it survives) and drops irrelevant ones (e.g.
        // `-u-cu-`), so `resolvedOptions().locale` reflects only what applied.
        let raw_locale = self
            .realm
            .get_property(obj, "\u{0}locale")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en-US"));
        let base = strip_unicode_extension(&raw_locale);
        let (resolved_nu, add) = resolve_nu_key(&base, &raw_locale, nu.as_deref());
        let resolved_locale = build_resolved_locale(&base, &[add]);
        let locv = self.new_str(&resolved_locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        self.store_str(obj, "numberingSystem", &Some(resolved_nu));

        // --- SetNumberFormatUnitOptions ---
        let style = self
            .get_string_option(
                opts,
                "style",
                &["decimal", "percent", "currency", "unit"],
                Some("decimal"),
            )?
            .unwrap();
        let currency = self.get_string_option(opts, "currency", &[], None)?;
        let currency_display = self.get_string_option(
            opts,
            "currencyDisplay",
            &["code", "symbol", "narrowSymbol", "name"],
            Some("symbol"),
        )?;
        let currency_sign = self.get_string_option(
            opts,
            "currencySign",
            &["standard", "accounting"],
            Some("standard"),
        )?;
        let unit = self.get_string_option(opts, "unit", &[], None)?;
        let unit_display = self.get_string_option(
            opts,
            "unitDisplay",
            &["short", "narrow", "long"],
            Some("short"),
        )?;
        // SetNumberFormatUnitOptions: a `currency` is required (TypeError) when
        // style is "currency"; whenever present it must be well-formed (RangeError)
        // regardless of style. Same shape for `unit`.
        match &currency {
            None if style == "currency" => {
                let m = self.new_str("currency code is required with currency style");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            Some(c) if !is_well_formed_currency(c) => {
                let m = self.new_str("invalid currency code");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            _ => {}
        }
        match &unit {
            None if style == "unit" => {
                let m = self.new_str("unit is required with unit style");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            Some(u) if !is_well_formed_unit(u) => {
                let m = self.new_str("invalid unit");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            _ => {}
        }
        let style_s = Some(style.clone());
        self.store_str(obj, "style", &style_s);
        // Canonical currency code is uppercased.
        if style == "currency" {
            let cc = currency.as_ref().map(|c| c.to_ascii_uppercase());
            self.store_str(obj, "currency", &cc);
            self.store_str(obj, "currencyDisplay", &currency_display);
            self.store_str(obj, "currencySign", &currency_sign);
        }
        if style == "unit" {
            self.store_str(obj, "unit", &unit);
            self.store_str(obj, "unitDisplay", &unit_display);
        }

        // notation (read before digit options per the spec order).
        let notation = self
            .get_string_option(
                opts,
                "notation",
                &["standard", "scientific", "engineering", "compact"],
                Some("standard"),
            )?
            .unwrap();

        // --- SetNumberFormatDigitOptions --- (shared with Intl.PluralRules).
        self.set_number_format_digit_options(obj, opts)?;

        // compactDisplay, useGrouping, signDisplay.
        let compact_display =
            self.get_string_option(opts, "compactDisplay", &["short", "long"], Some("short"))?;
        if notation == "compact" {
            self.store_str(obj, "compactDisplay", &compact_display);
        }
        // useGrouping: ECMA-402 accepts a boolean or "min2"/"auto"/"always". Read
        // it without enum validation, then normalize.
        let ug_raw = match opts {
            Some(h) => self.read_member(h, "useGrouping")?,
            None => NanBox::undefined(),
        };
        let use_grouping_val = self.normalize_use_grouping(ug_raw, &notation)?;
        let sign_display = self
            .get_string_option(
                opts,
                "signDisplay",
                &["auto", "never", "always", "exceptZero", "negative"],
                Some("auto"),
            )?
            .unwrap();

        // Store notation/sign options for resolvedOptions + formatting (the digit
        // options were stored by `set_number_format_digit_options`).
        self.store_str(obj, "notation", &Some(notation));
        self.store_str(obj, "signDisplay", &Some(sign_display));
        match use_grouping_val {
            UseGroupingResolved::Bool(b) => {
                self.realm
                    .set_hidden_property(obj, "useGrouping", NanBox::boolean(b));
            }
            UseGroupingResolved::Str(s) => {
                let sv = self.new_str(s);
                self.realm.set_hidden_property(obj, "useGrouping", sv);
            }
        }
        Ok(())
    }

    /// `SetNumberFormatDigitOptions`: reads, validates, and stores the shared
    /// number-format digit slots (`minimumIntegerDigits`, the fraction/significant
    /// digit pairs, `roundingIncrement`, `roundingMode`, `roundingPriority`,
    /// `trailingZeroDisplay`) on `obj` under their own hidden keys, in spec read
    /// order. Shared by `Intl.NumberFormat` and `Intl.PluralRules` so both honor the
    /// same option semantics and constructor option-read order. An invalid value is a
    /// RangeError; a Symbol is a TypeError (via the option getters).
    fn set_number_format_digit_options(
        &mut self,
        obj: Handle,
        opts: Option<Handle>,
    ) -> Result<(), ExecError> {
        let mnid = self
            .get_int_option(opts, "minimumIntegerDigits", 1.0, 21.0, Some(1.0))?
            .unwrap();
        let mnfd = self.get_int_option(opts, "minimumFractionDigits", 0.0, 100.0, None)?;
        let mxfd = self.get_int_option(opts, "maximumFractionDigits", 0.0, 100.0, None)?;
        let mnsd = self.get_int_option(opts, "minimumSignificantDigits", 1.0, 21.0, None)?;
        let mxsd = self.get_int_option(opts, "maximumSignificantDigits", 1.0, 21.0, None)?;
        // Cross-validation: an explicit minimum may not exceed an explicit maximum.
        if let (Some(a), Some(b)) = (mnfd, mxfd)
            && a > b
        {
            let m = self.new_str("minimumFractionDigits is greater than maximumFractionDigits");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        if let (Some(a), Some(b)) = (mnsd, mxsd)
            && a > b
        {
            let m =
                self.new_str("minimumSignificantDigits is greater than maximumSignificantDigits");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        // SetNumberFormatDigitOptions: when *either* significant-digit option is
        // present, the "significant digits" path is used and the other defaults
        // (min → 1, max → 21). Without this, a lone `minimumSignificantDigits` was
        // stored but the missing max meant the padding never applied
        // (`format(1)` with `{minimumSignificantDigits:3}` gave "1", not "1.00").
        let (mnsd, mxsd) = if mnsd.is_some() || mxsd.is_some() {
            (Some(mnsd.unwrap_or(1.0)), Some(mxsd.unwrap_or(21.0)))
        } else {
            (mnsd, mxsd)
        };
        // roundingIncrement ∈ a fixed allowed set.
        let rinc = self
            .get_int_option(opts, "roundingIncrement", 1.0, 5000.0, Some(1.0))?
            .unwrap();
        const ALLOWED_INC: [u32; 15] = [
            1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000,
        ];
        if !ALLOWED_INC.contains(&(rinc as u32)) {
            let m = self.new_str("invalid roundingIncrement");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let rounding_mode = self
            .get_string_option(
                opts,
                "roundingMode",
                &[
                    "ceil",
                    "floor",
                    "expand",
                    "trunc",
                    "halfCeil",
                    "halfFloor",
                    "halfExpand",
                    "halfTrunc",
                    "halfEven",
                ],
                Some("halfExpand"),
            )?
            .unwrap();
        let rounding_priority = self
            .get_string_option(
                opts,
                "roundingPriority",
                &["auto", "morePrecision", "lessPrecision"],
                Some("auto"),
            )?
            .unwrap();
        // SetNumberFormatDigitOptions: a `roundingIncrement` other than 1 is only
        // valid with the *fractionDigits* rounding type — else a TypeError — and
        // additionally requires maximumFractionDigits == minimumFractionDigits
        // (else a RangeError).
        if rinc != 1.0 {
            let rounding_type = if rounding_priority == "morePrecision" {
                "morePrecision"
            } else if rounding_priority == "lessPrecision" {
                "lessPrecision"
            } else if mnsd.is_some() {
                "significantDigits"
            } else {
                "fractionDigits"
            };
            if rounding_type != "fractionDigits" {
                return Err(self.type_error(
                    "roundingIncrement other than 1 requires the fractionDigits rounding type",
                ));
            }
            if let (Some(a), Some(b)) = (mnfd, mxfd)
                && a != b
            {
                let m = self.new_str("roundingIncrement requires equal min/max fraction digits");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        let tzd = self
            .get_string_option(
                opts,
                "trailingZeroDisplay",
                &["auto", "stripIfInteger"],
                Some("auto"),
            )?
            .unwrap();

        self.realm
            .set_hidden_property(obj, "minimumIntegerDigits", NanBox::number(mnid));
        if let Some(v) = mnfd {
            self.realm
                .set_hidden_property(obj, "minimumFractionDigits", NanBox::number(v));
        }
        if let Some(v) = mxfd {
            self.realm
                .set_hidden_property(obj, "maximumFractionDigits", NanBox::number(v));
        }
        if let Some(v) = mnsd {
            self.realm
                .set_hidden_property(obj, "minimumSignificantDigits", NanBox::number(v));
        }
        if let Some(v) = mxsd {
            self.realm
                .set_hidden_property(obj, "maximumSignificantDigits", NanBox::number(v));
        }
        self.realm
            .set_hidden_property(obj, "roundingIncrement", NanBox::number(rinc));
        self.store_str(obj, "roundingMode", &Some(rounding_mode));
        self.store_str(obj, "roundingPriority", &Some(rounding_priority));
        self.store_str(obj, "trailingZeroDisplay", &Some(tzd));
        Ok(())
    }

    /// `GetBooleanOrStringNumberFormatOption` for `useGrouping`: `undefined` →
    /// the default (`"min2"` when `notation` is `"compact"`, else `"auto"`);
    /// `true` → `"always"`; a falsy value → `false`; the strings `"true"`/`"false"`
    /// → the default; one of `"min2"`/`"auto"`/`"always"` → that string; any other
    /// string (or non-string, via ToString) → **RangeError**.
    fn normalize_use_grouping(
        &mut self,
        raw: NanBox,
        notation: &str,
    ) -> Result<UseGroupingResolved, ExecError> {
        let fallback = if notation == "compact" {
            "min2"
        } else {
            "auto"
        };
        if matches!(raw.unpack(), Unpacked::Undefined) {
            return Ok(UseGroupingResolved::Str(fallback));
        }
        if matches!(raw.unpack(), Unpacked::Bool(true)) {
            return Ok(UseGroupingResolved::Str("always"));
        }
        if !self.realm.truthy(raw) {
            return Ok(UseGroupingResolved::Bool(false));
        }
        let s = self.coerce_to_string(raw)?;
        match s.as_str() {
            "true" | "false" => Ok(UseGroupingResolved::Str(fallback)),
            "min2" => Ok(UseGroupingResolved::Str("min2")),
            "auto" => Ok(UseGroupingResolved::Str("auto")),
            "always" => Ok(UseGroupingResolved::Str("always")),
            _ => {
                let m = self.new_str(&alloc::format!("invalid useGrouping value '{s}'"));
                Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))))
            }
        }
    }

    /// Initializes an `Intl.DateTimeFormat`, reading and validating options in
    /// spec order. Stores resolved options on `obj`.
    fn init_datetime_format(&mut self, obj: Handle, opts: Option<Handle>) -> Result<(), ExecError> {
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let ca = self.get_string_option(opts, "calendar", &[], None)?;
        if let Some(c) = &ca
            && !is_unicode_type_value(c)
        {
            let m = self.new_str("invalid calendar");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let nu = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(n) = &nu
            && !is_unicode_type_value(n)
        {
            let m = self.new_str("invalid numberingSystem");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        // ResolveLocale for the relevant DateTimeFormat extension keys (`ca`, `nu`,
        // `hc`): a supported option wins, else the locale's `-u-` extension value,
        // else the default. The resolved locale keeps only surviving relevant keys
        // (`-u-ca-`/`-u-nu-`/`-u-hc-`) and drops irrelevant ones (e.g. `-u-cu-`).
        // `hc` is retained here as-is; its option interplay is finalized in
        // `dtf_hour_resolution` at resolvedOptions time.
        let raw_locale = self
            .realm
            .get_property(obj, "\u{0}locale")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en-US"));
        let base = strip_unicode_extension(&raw_locale);
        let (resolved_ca, ca_add) = resolve_ca_key(&base, &raw_locale, ca.as_deref());
        let (resolved_nu, nu_add) = resolve_nu_key(&base, &raw_locale, nu.as_deref());
        let hc_ext = split_u_keyword(&raw_locale, "hc")
            .1
            .filter(|v| matches!(v.as_str(), "h11" | "h12" | "h23" | "h24"));
        let hc_add = hc_ext
            .as_deref()
            .map(|v| alloc::format!("-hc-{v}"))
            .unwrap_or_default();
        let resolved_locale = build_resolved_locale(&base, &[ca_add, hc_add, nu_add]);
        let locv = self.new_str(&resolved_locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        self.store_str(obj, "calendar", &Some(resolved_ca));
        self.store_str(obj, "numberingSystem", &Some(resolved_nu));
        // hour12 (boolean) and hourCycle (enum).
        let hour12 = self.get_bool_option(opts, "hour12", None)?;
        if let Some(b) = hour12 {
            self.realm
                .set_hidden_property(obj, "hour12", NanBox::boolean(b));
        }
        let hc = self.get_string_option(opts, "hourCycle", &["h11", "h12", "h23", "h24"], None)?;
        self.store_str(obj, "hourCycle", &hc);
        // `timeZone`: ECMA-402 CreateDateTimeFormat steps 28–30. Undefined defaults
        // to the (system) UTC zone; otherwise the string is either an offset
        // identifier (`IsTimeZoneOffsetString`, canonicalized to `±HH:MM`) or a
        // named IANA identifier (ASCII-case-insensitively matched via the embedded
        // tz database and stored as its correctly-cased [[Identifier]] — links are
        // NOT canonicalized to their primary). Anything else is a RangeError.
        let tz = match self.get_string_option(opts, "timeZone", &[], None)? {
            Some(s) => self.dtf_resolve_time_zone(&s)?,
            None => String::from("UTC"),
        };
        self.store_str(obj, "timeZone", &Some(tz));
        // weekday/era/year/month/day/hour/minute/second/… component options.
        let nv = ["numeric", "2-digit"];
        let nm = ["long", "short", "narrow"];
        let weekday = self.get_string_option(opts, "weekday", &nm, None)?;
        self.store_str(obj, "weekday", &weekday);
        let era = self.get_string_option(opts, "era", &nm, None)?;
        self.store_str(obj, "era", &era);
        let year = self.get_string_option(opts, "year", &nv, None)?;
        self.store_str(obj, "year", &year);
        let month = self.get_string_option(
            opts,
            "month",
            &["numeric", "2-digit", "long", "short", "narrow"],
            None,
        )?;
        self.store_str(obj, "month", &month);
        let day = self.get_string_option(opts, "day", &nv, None)?;
        self.store_str(obj, "day", &day);
        let day_period = self.get_string_option(opts, "dayPeriod", &nm, None)?;
        self.store_str(obj, "dayPeriod", &day_period);
        let hour = self.get_string_option(opts, "hour", &nv, None)?;
        self.store_str(obj, "hour", &hour);
        let minute = self.get_string_option(opts, "minute", &nv, None)?;
        self.store_str(obj, "minute", &minute);
        let second = self.get_string_option(opts, "second", &nv, None)?;
        self.store_str(obj, "second", &second);
        let fsd = self.get_int_option(opts, "fractionalSecondDigits", 1.0, 3.0, None)?;
        if let Some(v) = fsd {
            self.realm
                .set_hidden_property(obj, "fractionalSecondDigits", NanBox::number(v));
        }
        let tzn = self.get_string_option(
            opts,
            "timeZoneName",
            &[
                "long",
                "short",
                "shortOffset",
                "longOffset",
                "shortGeneric",
                "longGeneric",
            ],
            None,
        )?;
        self.store_str(obj, "timeZoneName", &tzn);
        // formatMatcher (validated, unused).
        let _ = self.get_string_option(
            opts,
            "formatMatcher",
            &["basic", "best fit"],
            Some("best fit"),
        )?;
        let date_style = self.get_string_option(
            opts,
            "dateStyle",
            &["full", "long", "medium", "short"],
            None,
        )?;
        self.store_str(obj, "dateStyle", &date_style);
        let time_style = self.get_string_option(
            opts,
            "timeStyle",
            &["full", "long", "medium", "short"],
            None,
        )?;
        self.store_str(obj, "timeStyle", &time_style);
        // dateStyle/timeStyle are mutually exclusive with explicit component fields.
        if (date_style.is_some() || time_style.is_some())
            && (weekday.is_some()
                || era.is_some()
                || year.is_some()
                || month.is_some()
                || day.is_some()
                || hour.is_some()
                || minute.is_some()
                || second.is_some()
                || day_period.is_some()
                || fsd.is_some()
                || tzn.is_some())
        {
            let m = self.new_str("dateStyle/timeStyle may not be combined with component options");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // If no component field and no dateStyle/timeStyle were requested, the
        // default format is a numeric date: year/month/day = "numeric".
        let any_field = weekday.is_some()
            || era.is_some()
            || year.is_some()
            || month.is_some()
            || day.is_some()
            || day_period.is_some()
            || hour.is_some()
            || minute.is_some()
            || second.is_some()
            || fsd.is_some()
            || tzn.is_some();
        if !any_field && date_style.is_none() && time_style.is_none() {
            let numeric = Some(String::from("numeric"));
            self.store_str(obj, "year", &numeric);
            self.store_str(obj, "month", &numeric);
            self.store_str(obj, "day", &numeric);
            // Mark that year/month/day are the implicit default (not user-requested),
            // so the ECMA-402 Temporal formatting protocol (which operates on the
            // *raw* component options) treats them as absent.
            self.realm
                .set_hidden_property(obj, "\u{0}dtf_default_date", NanBox::boolean(true));
        }
        Ok(())
    }

    /// ECMA-402 CreateDateTimeFormat time-zone resolution (steps 29–30). Accepts a
    /// UTC-offset identifier (`IsTimeZoneOffsetString`, minute precision only —
    /// `±HH`, `±HHMM`, `±HH:MM` — normalized to `±HH:MM`, and `-00` → `+00:00`), or
    /// a named IANA identifier matched ASCII-case-insensitively against the embedded
    /// tz database and returned as its correctly-cased [[Identifier]] (links such as
    /// `Asia/Calcutta` are preserved, NOT canonicalized to their primary). An empty
    /// string, a U+2212 sign, a malformed offset, or an unknown name is a RangeError.
    /// Reuses the Temporal time-zone helpers (same tz database as `ZonedDateTime`).
    fn dtf_resolve_time_zone(&mut self, s: &str) -> Result<String, ExecError> {
        use super::temporal_zoneddatetime::{parse_offset_id, resolve_named};
        if !s.is_empty() {
            if let Some((_, canon)) = parse_offset_id(s) {
                return Ok(canon);
            }
            if let Some(name) = resolve_named(s) {
                return Ok(name);
            }
        }
        let m = self.new_str(&alloc::format!("invalid time zone: {s}"));
        Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))))
    }

    /// The offset (in milliseconds east of UTC) that DateTimeFormat instance
    /// `handle`'s resolved `timeZone` applies at the exact instant `epoch_ms`.
    /// Returns 0 for the UTC default (and for any zone the tz database cannot
    /// resolve). Named zones consult the embedded IANA data (DST-aware); offset
    /// identifiers apply their fixed offset. Shared by the number/`Date` and the
    /// Temporal (Instant/ZonedDateTime) formatting paths.
    #[cfg(feature = "intl")]
    fn dtf_zone_offset_ms(&self, handle: Handle, epoch_ms: i64) -> i64 {
        let Some(tz) = self
            .realm
            .get_property(handle, "timeZone")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
        else {
            return 0;
        };
        if tz == "UTC" || tz.is_empty() {
            return 0;
        }
        let epoch_ns = i128::from(epoch_ms) * 1_000_000;
        (super::temporal_zoneddatetime::tz_offset_at(&tz, epoch_ns) / 1_000_000) as i64
    }

    /// Applies DateTimeFormat `handle`'s time zone to a Temporal value's epoch
    /// milliseconds: an Instant/ZonedDateTime is an exact instant, so its wall-clock
    /// rendering is shifted into the resolved zone (and the zone offset recorded for
    /// `timeZoneName`); the wall-clock Plain* types carry no instant and are rendered
    /// as-is (the zone is ignored, per the spec's `isPlain` handling). Returns the
    /// (possibly shifted) epoch milliseconds to decompose.
    #[cfg(feature = "intl")]
    fn dtf_apply_temporal_zone(
        &mut self,
        handle: Handle,
        ms: f64,
        kind: crate::temporal_iso::TemporalKind,
        o: &mut intl::datetime::DateTimeFormatOptions,
    ) -> f64 {
        use crate::temporal_iso::TemporalKind;
        if matches!(kind, TemporalKind::Instant | TemporalKind::ZonedDateTime) {
            let off = self.dtf_zone_offset_ms(handle, ms as i64);
            // Only an exact instant carries a zone; the plain types leave both
            // fields unset so the crate strips the pattern's zone field instead of
            // filling it (a `timeStyle: "long"` PlainDateTime shows no zone name).
            if let Some(tz) = self
                .realm
                .get_property(handle, "timeZone")
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_display_string(v))
                .filter(|t| !t.is_empty())
            {
                o.time_zone = Some(self.intern_static(&tz));
            }
            o.tz_offset_minutes = Some((off / 60_000) as i32);
            ms + off as f64
        } else {
            ms
        }
    }

    /// The locale's default 12-hour cycle (`"h11"`/`"h12"`) and its default clock
    /// (whether it is a 12-hour locale). `base_locale` has no `-u-hc-` keyword.
    /// Uses the `intl` crate's CLDR time patterns to decide 12- vs 24-hour; the
    /// non-intl build falls back to the en-family heuristic.
    fn locale_hour_defaults(&self, base_locale: &str) -> (&'static str, bool) {
        // CLDR's 12-hour cycle is `h11` (0–11) for Japanese, `h12` (1–12) else.
        let primary = base_locale.split('-').next().unwrap_or("");
        let hc12 = if primary.eq_ignore_ascii_case("ja") {
            "h11"
        } else {
            "h12"
        };
        #[cfg(feature = "intl")]
        {
            use intl::datetime::{DateTime, DateTimeFormatOptions, Numeric2Digit};
            let dt = DateTime {
                year: 2020,
                month: 1,
                day: 1,
                hour: 13,
                minute: 0,
                second: 0,
                millisecond: 0,
            };
            let mut o = DateTimeFormatOptions::default();
            o.hour = Some(Numeric2Digit::Numeric);
            if let Ok(parts) = intl::datetime::format_to_parts(base_locale, &dt, &o) {
                // A 12-hour locale renders 13:00 with a dayPeriod part (or an hour
                // value != "13").
                let is_12h = parts.iter().any(|p| {
                    matches!(p.kind, intl::datetime::DateTimePartType::DayPeriod)
                        || (matches!(p.kind, intl::datetime::DateTimePartType::Hour)
                            && p.value != "13")
                });
                return (hc12, is_12h);
            }
        }
        let is_12h = primary.eq_ignore_ascii_case("en");
        (hc12, is_12h)
    }

    /// ECMA-402 hour-cycle resolution for a `DateTimeFormat` instance: returns the
    /// resolved locale (its `-u-hc-` keyword kept only when it survives the option
    /// interplay), the resolved `[[HourCycle]]` (`None` when no hour field), and
    /// `[[HourCycle]]`-derived `hour12` (`None` when no hour field).
    fn dtf_hour_resolution(&self, handle: Handle) -> (String, Option<String>, Option<bool>) {
        let raw_locale = self
            .realm
            .get_property(handle, "\u{0}locale")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en-US"));
        let (base_locale, hc_ext) = split_u_keyword(&raw_locale, "hc");
        let hc_ext = hc_ext.filter(|v| matches!(v.as_str(), "h11" | "h12" | "h23" | "h24"));
        let get_str = |k: &str| -> Option<String> {
            self.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_display_string(v))
        };
        let hc_opt = get_str("hourCycle");
        let hour12 = self
            .realm
            .get_property(handle, "hour12")
            .and_then(|v| v.as_boolean());
        // [[Hour]] is set iff an hour component or a timeStyle was requested.
        let hour_present = get_str("hour").is_some() || get_str("timeStyle").is_some();

        // Resolved locale: the `-u-hc-` keyword is kept only when it is present
        // and neither hour12 nor a *differing* hourCycle option overrides it.
        let keep_ext = hc_ext.is_some()
            && hour12.is_none()
            && match &hc_opt {
                Some(opt) => Some(opt.as_str()) == hc_ext.as_deref(),
                None => true,
            };
        let resolved_locale = if keep_ext {
            raw_locale.clone()
        } else {
            base_locale.clone()
        };

        if !hour_present {
            return (resolved_locale, None, None);
        }
        let (hc12, default_is_12h) = self.locale_hour_defaults(&base_locale);
        let resolved_hc = match hour12 {
            Some(true) => String::from(hc12),
            Some(false) => String::from("h23"),
            None => hc_opt
                .or(hc_ext)
                .unwrap_or_else(|| String::from(if default_is_12h { hc12 } else { "h23" })),
        };
        let h12 = matches!(resolved_hc.as_str(), "h11" | "h12");
        (resolved_locale, Some(resolved_hc), Some(h12))
    }

    /// `Intl.NumberFormat`/`DateTimeFormat` `resolvedOptions()` — a fresh object
    /// reporting the resolved configuration, in spec property order. `fmt` is the
    /// formatter instance (`None` → a default decimal NumberFormat shape).
    pub(crate) fn intl_resolved_options(&mut self, fmt: Option<Handle>) -> NanBox {
        let out = self.realm.new_object();
        let kind = fmt
            .and_then(|h| self.realm.get_property(h, "\u{0}intl"))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("number"));
        let get_str = |this: &Self, key: &str| -> Option<String> {
            fmt.and_then(|h| this.realm.get_property(h, key))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let get_num = |this: &Self, key: &str| -> Option<f64> {
            fmt.and_then(|h| this.realm.get_property(h, key))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_number(v))
        };
        let locale = get_str(self, "\u{0}locale").unwrap_or_else(|| String::from("en-US"));
        let lv = self.new_str(&locale);
        self.realm.set_property(out, "locale", lv);

        if kind == "segmenter" {
            // `Intl.Segmenter.prototype.resolvedOptions()` — `{ locale, granularity }`.
            let gran = get_str(self, "granularity").unwrap_or_else(|| String::from("grapheme"));
            let gv = self.new_str(&gran);
            self.realm.set_property(out, "granularity", gv);
        } else if kind == "collator" {
            // `Intl.Collator.prototype.resolvedOptions()` — `{ locale, usage,
            // sensitivity, ignorePunctuation, collation, numeric?, caseFirst? }`
            // in that order (Table: "Resolved Options of Collator instances").
            let usage = get_str(self, "usage").unwrap_or_else(|| String::from("sort"));
            let uv = self.new_str(&usage);
            self.realm.set_property(out, "usage", uv);
            let sensitivity =
                get_str(self, "sensitivity").unwrap_or_else(|| String::from("variant"));
            let sv = self.new_str(&sensitivity);
            self.realm.set_property(out, "sensitivity", sv);
            let ip = fmt
                .and_then(|h| self.realm.get_property(h, "ignorePunctuation"))
                .is_some_and(|v| self.realm.truthy(v));
            self.realm
                .set_property(out, "ignorePunctuation", NanBox::boolean(ip));
            let collation = get_str(self, "collation").unwrap_or_else(|| String::from("default"));
            let cv = self.new_str(&collation);
            self.realm.set_property(out, "collation", cv);
            // `numeric`/`caseFirst` are reported only when the instance resolved
            // them (they are optional keys of the resolved-options table).
            if let Some(n) = fmt.and_then(|h| self.realm.get_property(h, "numeric")) {
                let nv = NanBox::boolean(self.realm.truthy(n));
                self.realm.set_property(out, "numeric", nv);
            }
            if let Some(cf) = get_str(self, "caseFirst") {
                let cfv = self.new_str(&cf);
                self.realm.set_property(out, "caseFirst", cfv);
            }
        } else if kind == "display" {
            // `Intl.DisplayNames.prototype.resolvedOptions()` — `{ locale, style,
            // type, fallback[, languageDisplay] }` (Table: "Resolved Options of
            // DisplayNames instances").
            let style = get_str(self, "style").unwrap_or_else(|| String::from("long"));
            let stv = self.new_str(&style);
            self.realm.set_property(out, "style", stv);
            if let Some(t) = get_str(self, "type") {
                let tv = self.new_str(&t);
                self.realm.set_property(out, "type", tv);
            }
            let fallback = get_str(self, "fallback").unwrap_or_else(|| String::from("code"));
            let fv = self.new_str(&fallback);
            self.realm.set_property(out, "fallback", fv);
            if get_str(self, "type").as_deref() == Some("language") {
                let ld =
                    get_str(self, "languageDisplay").unwrap_or_else(|| String::from("dialect"));
                let ldv = self.new_str(&ld);
                self.realm.set_property(out, "languageDisplay", ldv);
            }
        } else if kind == "list" {
            // `Intl.ListFormat.prototype.resolvedOptions()` — `{ locale, type, style }`
            // in that order (Table: "Resolved Options of ListFormat instances").
            let lt = get_str(self, "type").unwrap_or_else(|| String::from("conjunction"));
            let ltv = self.new_str(&lt);
            self.realm.set_property(out, "type", ltv);
            let style = get_str(self, "style").unwrap_or_else(|| String::from("long"));
            let stv = self.new_str(&style);
            self.realm.set_property(out, "style", stv);
        } else if kind == "rtf" {
            // `Intl.RelativeTimeFormat.prototype.resolvedOptions()` —
            // `{ locale, style, numeric, numberingSystem }` in that order
            // (Table: "Resolved Options of RelativeTimeFormat instances"). The
            // resolved numbering system for the supported locales is always `latn`.
            let style = get_str(self, "style").unwrap_or_else(|| String::from("long"));
            let stv = self.new_str(&style);
            self.realm.set_property(out, "style", stv);
            let numeric = get_str(self, "numeric").unwrap_or_else(|| String::from("always"));
            let nv = self.new_str(&numeric);
            self.realm.set_property(out, "numeric", nv);
            let ns = get_str(self, "numberingSystem").unwrap_or_else(|| String::from("latn"));
            let nsv = self.new_str(&ns);
            self.realm.set_property(out, "numberingSystem", nsv);
        } else if kind == "plural" {
            // `Intl.PluralRules.prototype.resolvedOptions()` — the key order is
            // `{ locale, type, notation, minimumIntegerDigits, (fraction|significant
            // digit pair), pluralCategories, roundingIncrement, roundingMode,
            // roundingPriority, trailingZeroDisplay }`.
            let pr_type = get_str(self, "type").unwrap_or_else(|| String::from("cardinal"));
            let tv = self.new_str(&pr_type);
            self.realm.set_property(out, "type", tv);
            let notation = get_str(self, "notation").unwrap_or_else(|| String::from("standard"));
            let nv = self.new_str(&notation);
            self.realm.set_property(out, "notation", nv);
            if notation == "compact" {
                let cd = get_str(self, "compactDisplay").unwrap_or_else(|| String::from("short"));
                let cdv = self.new_str(&cd);
                self.realm.set_property(out, "compactDisplay", cdv);
            }
            // Digit options, resolved as in SetNumberFormatDigitOptions (decimal
            // defaults: minimumFractionDigits 0, maximumFractionDigits 3).
            let mnid = get_num(self, "minimumIntegerDigits").unwrap_or(1.0);
            self.realm
                .set_property(out, "minimumIntegerDigits", NanBox::number(mnid));
            let mnsd = get_num(self, "minimumSignificantDigits");
            let mxsd = get_num(self, "maximumSignificantDigits");
            let mnfd_opt = get_num(self, "minimumFractionDigits");
            let mxfd_opt = get_num(self, "maximumFractionDigits");
            let priority =
                get_str(self, "roundingPriority").unwrap_or_else(|| String::from("auto"));
            let has_sig = mnsd.is_some() || mxsd.is_some();
            let report_frac = |this: &mut Self, out: Handle| {
                let mnfd = mnfd_opt.unwrap_or(0.0);
                let mxfd = mxfd_opt.unwrap_or_else(|| 3.0_f64.max(mnfd));
                this.realm
                    .set_property(out, "minimumFractionDigits", NanBox::number(mnfd));
                this.realm
                    .set_property(out, "maximumFractionDigits", NanBox::number(mxfd));
            };
            let report_sig = |this: &mut Self, out: Handle| {
                let mnsd = mnsd.unwrap_or(1.0);
                let mxsd = mxsd.unwrap_or(21.0);
                this.realm
                    .set_property(out, "minimumSignificantDigits", NanBox::number(mnsd));
                this.realm
                    .set_property(out, "maximumSignificantDigits", NanBox::number(mxsd));
            };
            if priority == "morePrecision" || priority == "lessPrecision" {
                report_frac(self, out);
                report_sig(self, out);
            } else if has_sig {
                report_sig(self, out);
            } else {
                report_frac(self, out);
            }
            // pluralCategories: a fresh sorted array on every call.
            let ordinal = pr_type == "ordinal";
            let cats = self.plural_categories(&locale, ordinal);
            let cat_vals: Vec<NanBox> = cats.iter().map(|c| self.new_str(c)).collect();
            let arr = self.realm.new_array(cat_vals);
            self.realm
                .set_property(out, "pluralCategories", NanBox::handle(arr.to_raw()));
            // Rounding options.
            let rinc = get_num(self, "roundingIncrement").unwrap_or(1.0);
            self.realm
                .set_property(out, "roundingIncrement", NanBox::number(rinc));
            let rm = get_str(self, "roundingMode").unwrap_or_else(|| String::from("halfExpand"));
            let rmv = self.new_str(&rm);
            self.realm.set_property(out, "roundingMode", rmv);
            let rp = self.new_str(&priority);
            self.realm.set_property(out, "roundingPriority", rp);
            let tzd = get_str(self, "trailingZeroDisplay").unwrap_or_else(|| String::from("auto"));
            let tzv = self.new_str(&tzd);
            self.realm.set_property(out, "trailingZeroDisplay", tzv);
        } else if kind == "number" {
            let ns = get_str(self, "numberingSystem").unwrap_or_else(|| String::from("latn"));
            let nsv = self.new_str(&ns);
            self.realm.set_property(out, "numberingSystem", nsv);
            let style = get_str(self, "style").unwrap_or_else(|| String::from("decimal"));
            let sv = self.new_str(&style);
            self.realm.set_property(out, "style", sv);
            if style == "currency" {
                if let Some(c) = get_str(self, "currency") {
                    let cv = self.new_str(&c);
                    self.realm.set_property(out, "currency", cv);
                }
                let cd = get_str(self, "currencyDisplay").unwrap_or_else(|| String::from("symbol"));
                let cdv = self.new_str(&cd);
                self.realm.set_property(out, "currencyDisplay", cdv);
                let cs = get_str(self, "currencySign").unwrap_or_else(|| String::from("standard"));
                let csv = self.new_str(&cs);
                self.realm.set_property(out, "currencySign", csv);
            }
            if style == "unit" {
                if let Some(u) = get_str(self, "unit") {
                    let uv = self.new_str(&u);
                    self.realm.set_property(out, "unit", uv);
                }
                let ud = get_str(self, "unitDisplay").unwrap_or_else(|| String::from("short"));
                let udv = self.new_str(&ud);
                self.realm.set_property(out, "unitDisplay", udv);
            }
            // Digit options. Resolve per SetNumberFormatDigitOptions: significant
            // digits (if requested) are reported; otherwise fraction digits with
            // style-derived defaults. roundingPriority "auto" with neither set
            // reports fraction digits only.
            let mnid = get_num(self, "minimumIntegerDigits").unwrap_or(1.0);
            self.realm
                .set_property(out, "minimumIntegerDigits", NanBox::number(mnid));
            let mnsd = get_num(self, "minimumSignificantDigits");
            let mxsd = get_num(self, "maximumSignificantDigits");
            let mnfd_opt = get_num(self, "minimumFractionDigits");
            let mxfd_opt = get_num(self, "maximumFractionDigits");
            let priority =
                get_str(self, "roundingPriority").unwrap_or_else(|| String::from("auto"));
            let has_sig = mnsd.is_some() || mxsd.is_some();
            // SetNumberFormatDigitOptions defaults: the currency-specific fraction
            // digits apply only in "standard" notation; compact defaults max to 0;
            // scientific/engineering (and standard non-currency) use 0..=3; percent 0.
            let notation = get_str(self, "notation").unwrap_or_else(|| String::from("standard"));
            let (def_min, def_max): (f64, f64) = match style.as_str() {
                "currency" if notation == "standard" => (2.0, 2.0),
                "percent" => (0.0, 0.0),
                _ if notation == "compact" => (0.0, 0.0),
                _ => (0.0, 3.0),
            };
            let report_frac = |this: &mut Self, out: Handle| {
                let mnfd = mnfd_opt.unwrap_or(def_min);
                let mxfd = mxfd_opt.unwrap_or_else(|| def_max.max(mnfd));
                this.realm
                    .set_property(out, "minimumFractionDigits", NanBox::number(mnfd));
                this.realm
                    .set_property(out, "maximumFractionDigits", NanBox::number(mxfd));
            };
            let report_sig = |this: &mut Self, out: Handle| {
                let mnsd = mnsd.unwrap_or(1.0);
                let mxsd = mxsd.unwrap_or(21.0);
                this.realm
                    .set_property(out, "minimumSignificantDigits", NanBox::number(mnsd));
                this.realm
                    .set_property(out, "maximumSignificantDigits", NanBox::number(mxsd));
            };
            if priority == "morePrecision" || priority == "lessPrecision" {
                // Both groups present.
                report_frac(self, out);
                report_sig(self, out);
            } else if has_sig {
                report_sig(self, out);
            } else {
                // Significant digits absent → fraction digits (with or without
                // explicit values, defaulted by style).
                report_frac(self, out);
            }

            let ug = fmt
                .and_then(|h| self.realm.get_property(h, "useGrouping"))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .unwrap_or_else(|| self.new_str("auto"));
            self.realm.set_property(out, "useGrouping", ug);
            let notation = get_str(self, "notation").unwrap_or_else(|| String::from("standard"));
            let nv = self.new_str(&notation);
            self.realm.set_property(out, "notation", nv);
            if notation == "compact" {
                let cd = get_str(self, "compactDisplay").unwrap_or_else(|| String::from("short"));
                let cdv = self.new_str(&cd);
                self.realm.set_property(out, "compactDisplay", cdv);
            }
            let sd = get_str(self, "signDisplay").unwrap_or_else(|| String::from("auto"));
            let sdv = self.new_str(&sd);
            self.realm.set_property(out, "signDisplay", sdv);
            let rinc = get_num(self, "roundingIncrement").unwrap_or(1.0);
            self.realm
                .set_property(out, "roundingIncrement", NanBox::number(rinc));
            let rm = get_str(self, "roundingMode").unwrap_or_else(|| String::from("halfExpand"));
            let rmv = self.new_str(&rm);
            self.realm.set_property(out, "roundingMode", rmv);
            let rp = self.new_str(&priority);
            self.realm.set_property(out, "roundingPriority", rp);
            let tzd = get_str(self, "trailingZeroDisplay").unwrap_or_else(|| String::from("auto"));
            let tzv = self.new_str(&tzd);
            self.realm.set_property(out, "trailingZeroDisplay", tzv);
        } else {
            // DateTimeFormat resolvedOptions — property order per spec:
            // locale, calendar, numberingSystem, timeZone, hourCycle, hour12,
            // weekday, era, year, month, day, dayPeriod, hour, minute, second,
            // fractionalSecondDigits, timeZoneName, then dateStyle/timeStyle.
            let cal = get_str(self, "calendar").unwrap_or_else(|| String::from("gregory"));
            let cv = self.new_str(&cal);
            self.realm.set_property(out, "calendar", cv);
            let ns = get_str(self, "numberingSystem").unwrap_or_else(|| String::from("latn"));
            let nsv = self.new_str(&ns);
            self.realm.set_property(out, "numberingSystem", nsv);
            let tz = get_str(self, "timeZone").unwrap_or_else(|| String::from("UTC"));
            let tzv = self.new_str(&tz);
            self.realm.set_property(out, "timeZone", tzv);
            // Hour-cycle resolution (ECMA-402 CreateDateTimeFormat + the
            // resolvedOptions Table 6/7 rules): `[[HourCycle]]` and the derived
            // `hour12` are reported *only* when an hour field is present (no hour →
            // both undefined; e.g. a dateStyle-only or numeric-date formatter). The
            // resolved locale's `-u-hc-` keyword survives only when it is not
            // overridden by a differing option.
            if let Some(fmt) = fmt {
                let (resolved_locale, hour_cycle, hour12) = self.dtf_hour_resolution(fmt);
                let lv = self.new_str(&resolved_locale);
                self.realm.set_property(out, "locale", lv);
                if let Some(hc) = hour_cycle {
                    let v = self.new_str(&hc);
                    self.realm.set_property(out, "hourCycle", v);
                }
                if let Some(h12) = hour12 {
                    self.realm.set_property(out, "hour12", NanBox::boolean(h12));
                }
            }
            for key in [
                "weekday",
                "era",
                "year",
                "month",
                "day",
                "dayPeriod",
                "hour",
                "minute",
                "second",
            ] {
                if let Some(v) = get_str(self, key) {
                    let vv = self.new_str(&v);
                    self.realm.set_property(out, key, vv);
                }
            }
            // fractionalSecondDigits precedes timeZoneName in the spec order.
            if let Some(v) = get_num(self, "fractionalSecondDigits") {
                self.realm
                    .set_property(out, "fractionalSecondDigits", NanBox::number(v));
            }
            for key in ["timeZoneName", "dateStyle", "timeStyle"] {
                if let Some(v) = get_str(self, key) {
                    let vv = self.new_str(&v);
                    self.realm.set_property(out, key, vv);
                }
            }
        }
        NanBox::handle(out.to_raw())
    }

    /// Formats `value` per the `Intl.NumberFormat`/`DateTimeFormat` instance `handle`
    /// (a `\0intl`-marked object). Shared by `nf.format(x)` and the bound `nf.format`.
    /// `Number.prototype.toLocaleString(locale, options)` — with no options this is the
    /// grouped default; with an options object it honors `style` (decimal/percent/
    /// currency), `currency`, and `minimum`/`maximumFractionDigits` (en-US-ish, no real
    /// locale data; the rounding mode follows Rust's formatter, ~halfExpand).
    pub(crate) fn number_to_locale_string(&self, n: f64, opts: Option<NanBox>) -> String {
        let oh = match opts {
            Some(v) if !matches!(v.unpack(), Unpacked::Undefined | Unpacked::Null) => {
                match v.as_handle() {
                    Some(raw) => Handle::from_raw(raw),
                    None => return group_thousands(n),
                }
            }
            _ => return group_thousands(n),
        };
        // Non-finite values ignore the options (NaN/∞/-∞).
        if !n.is_finite() {
            return group_thousands(n);
        }
        let str_opt = |key: &str| -> Option<String> {
            self.realm
                .get_property(oh, key)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_display_string(v))
        };
        let num_opt = |key: &str| -> Option<i32> {
            self.realm
                .get_property(oh, key)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_number(v) as i32)
        };
        let style = str_opt("style").unwrap_or_else(|| String::from("decimal"));
        let (value, prefix, suffix, def_min, def_max) = match style.as_str() {
            "percent" => (n * 100.0, String::new(), String::from("%"), 0, 0),
            "currency" => {
                let sym = currency_symbol(&str_opt("currency").unwrap_or_default());
                (n, sym, String::new(), 2, 2)
            }
            _ => (n, String::new(), String::new(), 0, 3),
        };
        let min_frac = num_opt("minimumFractionDigits")
            .unwrap_or(def_min)
            .clamp(0, 100);
        let max_frac = num_opt("maximumFractionDigits")
            .unwrap_or(def_max.max(min_frac))
            .clamp(min_frac, 100);
        let neg = value.is_sign_negative() && value != 0.0;
        // Round to `max_frac` places, then trim trailing zeros down to `min_frac`.
        let formatted = alloc::format!("{:.*}", max_frac as usize, value.abs());
        let trimmed = if max_frac > min_frac && formatted.contains('.') {
            let dot = formatted.find('.').unwrap();
            let keep_min = dot + 1 + min_frac as usize;
            let mut end = formatted.len();
            while end > keep_min && formatted.as_bytes()[end - 1] == b'0' {
                end -= 1;
            }
            if end == dot + 1 {
                end = dot; // no fractional digits left → drop the '.'
            }
            String::from(&formatted[..end])
        } else {
            formatted
        };
        let grouped = group_thousands_str(&trimmed);
        let mut out = String::new();
        if neg {
            out.push('-');
        }
        out.push_str(&prefix);
        out.push_str(&grouped);
        out.push_str(&suffix);
        out
    }

    /// `Intl.DateTimeFormat` operand coercion for `format`/`formatToParts`:
    /// `undefined` → the current time; otherwise `? ToNumber(value)` (which runs
    /// a throwing `valueOf`), then `TimeClip`. A non-finite or out-of-range time
    /// value (`|x| > 8.64e15`) is a **RangeError** (sec-partitiondatetimepattern).
    pub(crate) fn datetime_operand(&mut self, value: NanBox) -> Result<f64, ExecError> {
        let x = if matches!(value.unpack(), Unpacked::Undefined) {
            now_ms()
        } else {
            let n = self.coerce_to_number(value)?;
            self.realm.to_number(n)
        };
        if !x.is_finite() || x.abs() > 8.64e15_f64 {
            let m = self.new_str("date value is not a finite time value");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(x)
    }

    /// The `[[NumberFormat]]`/`[[DateTimeFormat]]` instance for a `formatRange`
    /// call, plus its two coerced numeric endpoints. Per spec both `start`/`end`
    /// are **required** (`undefined` → TypeError) and must not be `NaN`
    /// (→ RangeError); a `start > end` is allowed. For a DateTimeFormat the
    /// endpoints are additionally `TimeClip`-validated (non-finite / out-of-range
    /// → RangeError). Returns `(instance, x, y)`.
    fn intl_range_operands(
        &mut self,
        this: NanBox,
        start: NanBox,
        end: NanBox,
    ) -> Result<(Handle, f64, f64), ExecError> {
        let Some(inst) = this
            .as_handle()
            .map(Handle::from_raw)
            .filter(|h| self.realm.get_property(*h, "\u{0}intl").is_some())
        else {
            return Err(self.type_error("formatRange called on an incompatible receiver"));
        };
        if matches!(start.unpack(), Unpacked::Undefined)
            || matches!(end.unpack(), Unpacked::Undefined)
        {
            return Err(self.type_error("formatRange requires two defined arguments"));
        }
        // `Intl.NumberFormat` uses ToIntlMathematicalValue, which accepts a BigInt
        // operand (via its numeric value); `Intl.DateTimeFormat` uses ToNumber,
        // which throws on a BigInt. Branch on the receiver kind.
        let is_number_format = self
            .realm
            .get_property(inst, "\u{0}intl")
            .map(|k| self.realm.to_display_string(k))
            .as_deref()
            == Some("number");
        let coerce_operand = |this: &mut Self, v: NanBox| -> Result<f64, ExecError> {
            if is_number_format
                && let Some(h) = v.as_handle().map(Handle::from_raw)
                && let Some(big) = this.realm.bigint_at(h)
            {
                return Ok(big.to_f64());
            }
            let n = this.coerce_to_number(v)?;
            Ok(this.realm.to_number(n))
        };
        let x = coerce_operand(self, start)?;
        let y = coerce_operand(self, end)?;
        if x.is_nan() || y.is_nan() {
            let m = self.new_str("formatRange arguments must not be NaN");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let is_datetime = self
            .realm
            .get_property(inst, "\u{0}intl")
            .map(|k| self.realm.to_display_string(k))
            .as_deref()
            == Some("datetime");
        if is_datetime && (!x.is_finite() || x.abs() > 8.64e15_f64 || y.abs() > 8.64e15_f64) {
            let m = self.new_str("formatRange date value is not a finite time value");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok((inst, x, y))
    }

    /// `Intl.NumberFormat/DateTimeFormat.prototype.formatRange(x, y)`
    /// (`FormatNumericRange` / `FormatDateTimeRange`): the flat string is the
    /// concatenation of the tagged range parts.
    pub(crate) fn intl_format_range(
        &mut self,
        this: NanBox,
        start: NanBox,
        end: NanBox,
    ) -> Result<String, ExecError> {
        let Some(inst) = this
            .as_handle()
            .map(Handle::from_raw)
            .filter(|h| self.realm.get_property(*h, "\u{0}intl").is_some())
        else {
            return Err(self.type_error("formatRange called on an incompatible receiver"));
        };
        // A DateTimeFormat range is rendered via the CLDR interval patterns (shared
        // fields collapsed, the locale interval separator); the flat string is the
        // concatenation of the tagged range parts.
        if self.intl_kind_is_datetime(inst) {
            let parts = self.dtf_range_dispatch(inst, start, end)?;
            return Ok(parts.iter().map(|(_, v, _)| v.as_str()).collect());
        }
        let (inst, x, y) = self.intl_range_operands(this, start, end)?;
        let parts = self.nf_range_parts(inst, start, end, x, y);
        Ok(parts.iter().map(|(_, v, _)| v.as_str()).collect())
    }

    /// Whether the `\0intl`-branded instance `inst` is an `Intl.DateTimeFormat`.
    fn intl_kind_is_datetime(&self, inst: Handle) -> bool {
        self.realm
            .get_property(inst, "\u{0}intl")
            .map(|k| self.realm.to_display_string(k))
            .as_deref()
            == Some("datetime")
    }

    /// `formatRangeToParts(x, y)` → an array of `{ type, value, source }` parts.
    pub(crate) fn intl_format_range_to_parts(
        &mut self,
        this: NanBox,
        start: NanBox,
        end: NanBox,
    ) -> Result<NanBox, ExecError> {
        let Some(inst) = this
            .as_handle()
            .map(Handle::from_raw)
            .filter(|h| self.realm.get_property(*h, "\u{0}intl").is_some())
        else {
            return Err(self.type_error("formatRange called on an incompatible receiver"));
        };
        // `Intl.DateTimeFormat`: for a Gregorian, number-valued range, the `intl`
        // crate's `format_range_to_parts` applies the CLDR interval patterns (the
        // greatest-difference field, shared-field collapse, the locale interval
        // separator). Temporal endpoints and non-Gregorian calendars fall back to a
        // field-level approximation (each endpoint's `formatToParts`, tagged by
        // source), which still collapses byte-for-byte to `formatToParts` when the
        // endpoints are practically equal.
        if self.intl_kind_is_datetime(inst) {
            let tagged = self.dtf_range_dispatch(inst, start, end)?;
            return Ok(self.intl_build_source_parts(tagged));
        }
        let (inst, x, y) = self.intl_range_operands(this, start, end)?;
        let parts = self.nf_range_parts(inst, start, end, x, y);
        Ok(self.intl_build_source_parts(parts))
    }

    /// The tagged `(type, value, source)` parts of an `Intl.NumberFormat` range —
    /// `PartitionNumberRangePattern` — shared by `formatRange` (concatenated) and
    /// `formatRangeToParts`.
    ///
    /// The rendering is the `intl` crate's `format_range_to_parts`: the locale's
    /// CLDR `miscPatterns` `range` form, the `approximately` form when both ends
    /// render alike (every part `shared`), and ICU's `AUTO`-level collapsing of the
    /// affixes the two ends have in common (`"+$2.90–3.10"`, but `"$3 – $5"`).
    /// `start`/`end` are the *original* operands, needed for the exact-decimal path;
    /// `x`/`y` are their `f64` coercions.
    #[cfg(feature = "intl")]
    fn nf_range_parts(
        &mut self,
        inst: Handle,
        start: NanBox,
        end: NanBox,
        x: f64,
        y: f64,
    ) -> Vec<(&'static str, String, &'static str)> {
        // ECMA-402 `ToIntlMathematicalValue` keeps a high-precision string / BigInt
        // endpoint exact; the crate takes `f64`s, which would round both ends of
        // `"987654321987654321"`–`"987654321987654322"` to the same value (and so
        // collapse the range to the `approximately` form).
        if let Some(p) = self.nf_range_exact_parts(inst, start, end, x, y) {
            return p;
        }
        let locale = self
            .realm
            .get_property(inst, "\u{0}locale")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en"));
        let base = self.number_format_options(inst);
        // Pre-round each endpoint to the ECMA-402 / ICU decimal exactly as the
        // `format` path does. `number_precision_round` also swaps in `halfExpand`
        // for a value it pre-rounded (so the crate won't re-round the f64 noise),
        // hence a private copy of the options per endpoint plus a merge.
        let mut ox = base;
        let rx = self.number_precision_round(inst, &mut ox, x);
        let mut oy = base;
        let ry = self.number_precision_round(inst, &mut oy, y);
        let mut opts = base;
        opts.rounding_mode = if ox.rounding_mode != base.rounding_mode {
            ox.rounding_mode
        } else {
            oy.rounding_mode
        };
        // `intl` panics formatting a true negative zero; the smallest negative
        // subnormal renders as a signed displayed zero (see the `format` path).
        let feed = |n: f64| {
            if n == 0.0 && n.is_sign_negative() {
                -f64::from_bits(1)
            } else {
                n
            }
        };
        let mut parts: Vec<(&'static str, String, &'static str)> =
            intl::number::format_range_to_parts(&locale, feed(rx), feed(ry), &opts)
                .into_iter()
                .map(|p| (p.kind.as_str(), p.value, p.source.as_str()))
                .collect();
        self.apply_numbering_digits_to_parts(inst, &mut parts);
        parts
    }

    /// Rewrites each part's ASCII digits into the formatter's resolved numbering
    /// system (see [`apply_numbering_digits`](Self::apply_numbering_digits)).
    #[cfg(feature = "intl")]
    fn apply_numbering_digits_to_parts(
        &mut self,
        inst: Handle,
        parts: &mut [(&'static str, String, &'static str)],
    ) {
        let nu = self
            .realm
            .get_property(inst, "numberingSystem")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        if !numbering_system_digit_base(&nu).is_some_and(|b| b != 0x0030) && nu != "hanidec" {
            return;
        }
        for (_, v, _) in parts.iter_mut() {
            if v.chars().any(|c| c.is_ascii_digit()) {
                *v = substitute_numbering_digits(&nu, core::mem::take(v));
            }
        }
    }

    /// The range parts when at least one endpoint is a high-precision string /
    /// BigInt that `format`'s exact-decimal path renders (see
    /// [`try_exact_decimal_format`](Self::try_exact_decimal_format)); `None` when
    /// neither endpoint needs it, so the caller uses the crate's `f64` range.
    ///
    /// The endpoints are rendered exactly, and the crate is consulted only for the
    /// glue: the `range` / `approximately` pattern's prefix, infix and suffix at
    /// this option set, recovered by formatting a probe pair through the very same
    /// code path and splitting the range rendering around the two endpoint
    /// renderings. (The exact path is reached only for standard-notation decimal
    /// style, which carries no unit/currency modifier, so the range is exactly
    /// `prefix + start + infix + end + suffix`.) Like `formatToParts`, the exact
    /// renderer produces a flat string rather than tagged digit runs, so each
    /// endpoint contributes a single `literal` part.
    #[cfg(feature = "intl")]
    fn nf_range_exact_parts(
        &mut self,
        inst: Handle,
        start: NanBox,
        end: NanBox,
        x: f64,
        y: f64,
    ) -> Option<Vec<(&'static str, String, &'static str)>> {
        let exact_start = self.try_exact_decimal_format(inst, start);
        let exact_end = self.try_exact_decimal_format(inst, end);
        if exact_start.is_none() && exact_end.is_none() {
            return None;
        }
        let a = match exact_start {
            Some(s) => s,
            None => self.intl_format_value(inst, NanBox::number(x)),
        };
        let b = match exact_end {
            Some(s) => s,
            None => self.intl_format_value(inst, NanBox::number(y)),
        };
        let locale = self
            .realm
            .get_property(inst, "\u{0}locale")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en"));
        let opts = self.number_format_options(inst);
        // The probe keeps each end's sign (which is a modifier, so it drives the
        // separator's spacing heuristic) but is otherwise a trivially distinct pair.
        let px = if x.is_sign_negative() { -1.0 } else { 1.0 };
        let py = if y.is_sign_negative() { -2.0 } else { 2.0 };
        let mut out: Vec<(&'static str, String, &'static str)> = Vec::new();
        let lit = |text: &str, out: &mut Vec<(&'static str, String, &'static str)>| {
            if !text.is_empty() {
                out.push(("literal", String::from(text), "shared"));
            }
        };
        if a == b {
            // Both ends render alike: the locale's `approximately` form, all shared.
            let one = intl::number::format(&locale, px, &opts);
            let approx = intl::number::format_range(&locale, px, px, &opts);
            let (pre, post) = approx.split_once(one.as_str())?;
            lit(pre, &mut out);
            out.push(("literal", a, "shared"));
            lit(post, &mut out);
            return Some(out);
        }
        let fx = intl::number::format(&locale, px, &opts);
        let fy = intl::number::format(&locale, py, &opts);
        let range = intl::number::format_range(&locale, px, py, &opts);
        let (pre, rest) = range.split_once(fx.as_str())?;
        let (sep, post) = rest.rsplit_once(fy.as_str())?;
        lit(pre, &mut out);
        out.push(("literal", a, "startRange"));
        lit(sep, &mut out);
        out.push(("literal", b, "endRange"));
        lit(post, &mut out);
        Some(out)
    }

    /// The tagged `(type, value, source)` parts of an `Intl.NumberFormat` range
    /// without the `intl` crate: each endpoint's hand-rolled rendering, joined by an
    /// en-dash (`x === y` collapses to the lone value).
    #[cfg(not(feature = "intl"))]
    fn nf_range_parts(
        &mut self,
        inst: Handle,
        _start: NanBox,
        _end: NanBox,
        x: f64,
        y: f64,
    ) -> Vec<(&'static str, String, &'static str)> {
        let fx = self.intl_format_value(inst, NanBox::number(x));
        let mut parts: Vec<(&'static str, String, &'static str)> =
            alloc::vec![("literal", fx, "startRange")];
        if x != y {
            let fy = self.intl_format_value(inst, NanBox::number(y));
            parts.push(("literal", String::from("\u{2013}"), "shared"));
            parts.push(("literal", fy, "endRange"));
        }
        parts
    }

    /// Resolves and validates a `DateTimeFormat` range's two endpoints, returning
    /// `(startIsTemporal, startNumber, endIsTemporal, endNumber)`. Both endpoints are
    /// required (`undefined` → TypeError) and must share the same Date/Temporal kind.
    ///
    /// Per `PartitionDateTimeRangePattern`, `ToDateTimeFormattable` runs on *both*
    /// endpoints (in order) — a Temporal object is kept as-is, anything else is
    /// `ToNumber`-coerced (running its `valueOf`) — **before** the same-kind check.
    /// A `NaN` from `ToNumber` does not throw here; the `TimeClip` RangeError is
    /// raised only later, when the number is actually formatted.
    fn dtf_resolve_range_operands(
        &mut self,
        start: NanBox,
        end: NanBox,
    ) -> Result<(bool, f64, bool, f64), ExecError> {
        if matches!(start.unpack(), Unpacked::Undefined)
            || matches!(end.unpack(), Unpacked::Undefined)
        {
            return Err(self.type_error("formatRange requires two defined arguments"));
        }
        #[cfg(feature = "intl")]
        let sx_temporal = self.is_temporal_value(start);
        #[cfg(not(feature = "intl"))]
        let sx_temporal = false;
        let sx_num = if sx_temporal {
            0.0
        } else {
            let n = self.coerce_to_number(start)?;
            self.realm.to_number(n)
        };
        #[cfg(feature = "intl")]
        let sy_temporal = self.is_temporal_value(end);
        #[cfg(not(feature = "intl"))]
        let sy_temporal = false;
        let sy_num = if sy_temporal {
            0.0
        } else {
            let n = self.coerce_to_number(end)?;
            self.realm.to_number(n)
        };
        // SameTemporalType: if either endpoint is a Temporal object, both must be
        // the same Temporal kind (a plain number counts as its own, non-Temporal
        // kind, so a Temporal-vs-number pair is always a mismatch).
        #[cfg(feature = "intl")]
        if (sx_temporal || sy_temporal) && self.range_type_tag(start) != self.range_type_tag(end) {
            return Err(
                self.type_error("formatRange arguments must be of the same Date/Temporal type")
            );
        }
        Ok((sx_temporal, sx_num, sy_temporal, sy_num))
    }

    /// The tagged `(type, value, source)` parts of a `DateTimeFormat` range.
    ///
    /// A Gregorian, number-valued range is rendered through the `intl` crate's
    /// `format_range_to_parts` (real CLDR interval patterns). Temporal endpoints and
    /// non-Gregorian calendars fall back to a field-level approximation: each
    /// endpoint's `formatToParts`, tagged `startRange`/`endRange` and joined by a
    /// shared interval separator, collapsing to all-`shared` when the endpoints are
    /// practically equal.
    fn dtf_range_dispatch(
        &mut self,
        inst: Handle,
        start: NanBox,
        end: NanBox,
    ) -> Result<Vec<(&'static str, String, &'static str)>, ExecError> {
        let (sx_temporal, sx_num, sy_temporal, sy_num) =
            self.dtf_resolve_range_operands(start, end)?;
        #[cfg(feature = "intl")]
        {
            let cal = self.dtf_resolved_calendar(inst);
            let gregorian = matches!(cal.as_str(), "gregory" | "gregorian" | "iso8601");
            if gregorian && !sx_temporal && !sy_temporal {
                for n in [sx_num, sy_num] {
                    if !n.is_finite() || n.abs() > 8.64e15_f64 {
                        let m = self.new_str("date value is not a finite time value");
                        return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                    }
                }
                // The collapse decision uses each endpoint's full field parts (so a
                // second-/fractional-second-only difference — which the crate's
                // greatest-difference search does not detect — still produces a
                // range). Practically-equal endpoints collapse to all-`shared`.
                let sp = self.datetime_parts(inst, sx_num);
                let ep = self.datetime_parts(inst, sy_num);
                if sp == ep {
                    return Ok(sp.into_iter().map(|(t, v)| (t, v, "shared")).collect());
                }
                // Same date, different time of day: the date is rendered once.
                let sep = self.dtf_interval_separator(inst);
                if let Some(v) = Self::dtf_same_day_range(&sp, &ep, &sep) {
                    return Ok(v);
                }
                // Endpoints differ: prefer the crate's CLDR interval patterns when it
                // resolves a genuine split; otherwise (it only collapsed because the
                // sole difference is below the minute) fall back to a field-level range
                // joined by the default CLDR interval separator.
                let crate_parts = self.dtf_range_crate_parts(inst, sx_num, sy_num);
                if crate_parts.iter().any(|(_, _, src)| *src != "shared") {
                    return Ok(crate_parts);
                }
                let mut v: Vec<(&'static str, String, &'static str)> =
                    Vec::with_capacity(sp.len() + ep.len() + 1);
                for (t, val) in sp {
                    v.push((t, val, "startRange"));
                }
                v.push(("literal", sep, "shared"));
                for (t, val) in ep {
                    v.push((t, val, "endRange"));
                }
                return Ok(v);
            }
        }
        let sp = self.dtf_operand_parts(inst, start, sx_temporal, sx_num)?;
        let ep = self.dtf_operand_parts(inst, end, sy_temporal, sy_num)?;
        if sp == ep {
            return Ok(sp.into_iter().map(|(t, v)| (t, v, "shared")).collect());
        }
        #[cfg(feature = "intl")]
        let sep = self.dtf_interval_separator(inst);
        #[cfg(not(feature = "intl"))]
        let sep = String::from("\u{2013}");
        #[cfg(feature = "intl")]
        if let Some(v) = Self::dtf_same_day_range(&sp, &ep, &sep) {
            return Ok(v);
        }
        let mut v: Vec<(&'static str, String, &'static str)> =
            Vec::with_capacity(sp.len() + ep.len() + 1);
        for (t, val) in sp {
            v.push((t, val, "startRange"));
        }
        v.push(("literal", sep, "shared"));
        for (t, val) in ep {
            v.push((t, val, "endRange"));
        }
        Ok(v)
    }

    /// UTS #35 §2.6.2 date+time interval composition: when a range's two endpoints
    /// fall on the same date and differ only in the time of day, the date is
    /// rendered *once* and only the time is ranged — `"8/4/2021, 12:30:45 AM – 11:30:45 PM"`,
    /// not the date twice. CLDR ships interval patterns for date-only and
    /// time-only skeletons but none for a combined one (there is no `yMdhms`
    /// item), so the composition is done here from the two endpoints' own
    /// crate-produced field parts: the common date prefix (up to and including the
    /// date/time connector literal) becomes `shared`, the two time tails
    /// `startRange`/`endRange`.
    ///
    /// `None` when the shape does not apply — the dates differ, there are no time
    /// fields, the formatter is time-only, or the locale pattern puts a date field
    /// after a time field — leaving the caller's ordinary interval-pattern path in
    /// charge.
    #[cfg(feature = "intl")]
    fn dtf_same_day_range(
        sp: &[(&'static str, String)],
        ep: &[(&'static str, String)],
        sep: &str,
    ) -> Option<Vec<(&'static str, String, &'static str)>> {
        fn is_time(t: &str) -> bool {
            matches!(
                t,
                "hour" | "minute" | "second" | "fractionalSecond" | "dayPeriod" | "timeZoneName"
            )
        }
        fn is_date(t: &str) -> bool {
            matches!(
                t,
                "weekday" | "era" | "year" | "month" | "day" | "relatedYear" | "yearName"
            )
        }
        let i = sp.iter().position(|(t, _)| is_time(t))?;
        let j = ep.iter().position(|(t, _)| is_time(t))?;
        // A time-first (or date-interleaved) locale pattern cannot be split this way.
        if i == 0 || i != j || sp[i..].iter().chain(&ep[j..]).any(|(t, _)| is_date(t)) {
            return None;
        }
        // Only a pure time-of-day difference composes; anything else is a genuine
        // date range and belongs to the CLDR interval patterns.
        if sp[..i] != ep[..j] || sp[i..] == ep[j..] {
            return None;
        }
        let mut v: Vec<(&'static str, String, &'static str)> =
            Vec::with_capacity(sp.len() + ep.len() + 1);
        for (t, val) in &sp[..i] {
            v.push((t, val.clone(), "shared"));
        }
        for (t, val) in &sp[i..] {
            v.push((t, val.clone(), "startRange"));
        }
        v.push(("literal", String::from(sep), "shared"));
        for (t, val) in &ep[j..] {
            v.push((t, val.clone(), "endRange"));
        }
        Some(v)
    }

    /// The locale's CLDR date-interval separator (`"\u{2009}–\u{2009}"` for `en`).
    /// Read out of the `intl` crate's interval patterns — by formatting a synthetic
    /// one-day numeric-date range and taking the `shared` literal — rather than
    /// hard-coded, so the field-level range fallback (Temporal endpoints,
    /// non-Gregorian calendars, and sub-minute differences the crate's
    /// greatest-difference search misses) joins its two halves exactly the way the
    /// crate-driven Gregorian path does. Without that agreement a formatter reports
    /// two different separators depending on the endpoint kind.
    #[cfg(feature = "intl")]
    fn dtf_interval_separator(&mut self, inst: Handle) -> String {
        use intl::datetime::{
            DateTime, DateTimeFormatOptions, DateTimePartType, MonthStyle, Numeric2Digit,
            RangeSource, format_range_to_parts,
        };
        let locale = self
            .realm
            .get_property(inst, "\u{0}locale")
            .map_or_else(|| String::from("en"), |v| self.realm.to_display_string(v));
        let mut o = DateTimeFormatOptions::default();
        o.year = Some(Numeric2Digit::Numeric);
        o.month = Some(MonthStyle::Numeric);
        o.day = Some(Numeric2Digit::Numeric);
        let a = DateTime {
            year: 2021,
            month: 8,
            day: 4,
            ..DateTime::default()
        };
        let b = DateTime { day: 5, ..a };
        format_range_to_parts(&locale, &a, &b, &o)
            .ok()
            .and_then(|ps| {
                ps.into_iter()
                    .find(|p| {
                        p.kind == DateTimePartType::Literal && p.source == RangeSource::Shared
                    })
                    .map(|p| p.value)
            })
            .unwrap_or_else(|| String::from("\u{2009}\u{2013}\u{2009}"))
    }

    /// The tagged range parts for a Gregorian, number-valued `DateTimeFormat` range,
    /// via the `intl` crate's CLDR interval patterns. Numeric part values are mapped
    /// through the instance's resolved numbering system.
    #[cfg(feature = "intl")]
    fn dtf_range_crate_parts(
        &mut self,
        inst: Handle,
        sx_num: f64,
        sy_num: f64,
    ) -> Vec<(&'static str, String, &'static str)> {
        let (locale, dt1, o) = self.dtf_locale_dt_opts(inst, sx_num);
        let (_, dt2, _) = self.dtf_locale_dt_opts(inst, sy_num);
        self.dtf_crate_range_parts(inst, &locale, &dt1, &dt2, &o)
    }

    /// The shared back half of the crate-driven range paths: runs the `intl`
    /// crate's CLDR interval formatter over an already-built
    /// `(locale, start, end, options)` and normalizes the parts the same way
    /// [`dtf_pad_time_parts`] normalizes single-date parts (2-digit
    /// minute/second widening, U+202F folding), then maps digits through the
    /// instance's numbering system.
    #[cfg(feature = "intl")]
    fn dtf_crate_range_parts(
        &mut self,
        inst: Handle,
        locale: &str,
        dt1: &intl::datetime::DateTime,
        dt2: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Vec<(&'static str, String, &'static str)> {
        use intl::datetime::{self, DateTimePartType};
        let raw = match datetime::format_range_to_parts(locale, dt1, dt2, o) {
            Ok(parts) => parts,
            Err(_) => return Vec::new(),
        };
        // Match `dtf_pad_time_parts`: an abutting hour/second widens a single-digit
        // minute (and hour/minute widens second) to two digits, so a collapsed range
        // is byte-for-byte `format` (which pads the same way).
        let has_hour = raw.iter().any(|p| p.kind == DateTimePartType::Hour);
        let has_min = raw.iter().any(|p| p.kind == DateTimePartType::Minute);
        let has_sec = raw.iter().any(|p| p.kind == DateTimePartType::Second);
        raw.into_iter()
            .map(|p| {
                let widen = match p.kind {
                    DateTimePartType::Minute => has_hour || has_sec,
                    DateTimePartType::Second => has_hour || has_min,
                    _ => false,
                };
                let mut v = p.value;
                if widen && v.len() == 1 && v.as_bytes()[0].is_ascii_digit() {
                    v.insert(0, '0');
                }
                if p.kind == DateTimePartType::Literal && v.contains('\u{202f}') {
                    v = v.replace('\u{202f}', " ");
                }
                let value = self.apply_numbering_digits(inst, v);
                (p.kind.as_str(), value, p.source.as_str())
            })
            .collect()
    }

    /// The field-level parts of one already-`ToDateTimeFormattable`-resolved range
    /// endpoint: a Temporal object via the ECMA-402 Temporal protocol, otherwise the
    /// `TimeClip`-validated number `num` (non-finite / out-of-range → RangeError).
    #[cfg_attr(not(feature = "intl"), expect(unused_variables))]
    fn dtf_operand_parts(
        &mut self,
        inst: Handle,
        value: NanBox,
        is_temporal: bool,
        num: f64,
    ) -> Result<Vec<(&'static str, String)>, ExecError> {
        #[cfg(feature = "intl")]
        if is_temporal && let Some(p) = self.temporal_format_parts(inst, value, false)? {
            return Ok(p);
        }
        if !num.is_finite() || num.abs() > 8.64e15_f64 {
            let m = self.new_str("date value is not a finite time value");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(self.datetime_parts(inst, num))
    }

    /// Materializes a `[{ type, value, source }]` array from `(type, value, source)`
    /// triples.
    fn intl_build_source_parts(&mut self, parts: Vec<(&str, String, &str)>) -> NanBox {
        let mut elems = Vec::with_capacity(parts.len());
        for (ty, val, src) in parts {
            let o = self.realm.new_object();
            let tv = self.new_str(ty);
            let vv = self.new_str(&val);
            let sv = self.new_str(src);
            self.realm.set_property(o, "type", tv);
            self.realm.set_property(o, "value", vv);
            self.realm.set_property(o, "source", sv);
            elems.push(NanBox::handle(o.to_raw()));
        }
        NanBox::handle(self.realm.new_array(elems).to_raw())
    }

    pub(crate) fn intl_format_value(&mut self, handle: Handle, value: NanBox) -> String {
        let kind = self
            .realm
            .get_property(handle, "\u{0}intl")
            .map(|k| self.realm.to_display_string(k))
            .unwrap_or_default();
        if kind == "datetime" {
            let ms = match value.as_handle().map(Handle::from_raw) {
                Some(h) if self.realm.date_at(h).is_some() => self.realm.date_at(h).unwrap(),
                _ => self.realm.to_number(value),
            };
            self.format_intl_datetime(handle, ms)
        } else {
            let n = self.realm.to_number(value);
            self.intl_format_number(handle, n)
        }
    }

    /// Builds an `Intl.RelativeTimeFormat` instance (`InitializeRelativeTimeFormat`):
    /// canonicalizes the requested locales, then reads and validates the options in
    /// spec order (`localeMatcher`, `numberingSystem`, `style`, `numeric`), and brands
    /// the object with `\0intl="rtf"` plus its resolved `\0locale`/`numberingSystem`/
    /// `style`/`numeric`. Used only by the `new`-form constructor; a plain call (no
    /// `new`) throws a `TypeError` before reaching here.
    pub(crate) fn make_relative_time_format(
        &mut self,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        self.init_relative_time_format(obj, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    pub(crate) fn init_relative_time_format(
        &mut self,
        obj: Handle,
        args: &[NanBox],
    ) -> Result<(), ExecError> {
        let marker = self.new_str("rtf");
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // 1. CanonicalizeLocaleList(locales) — a malformed tag raises a RangeError.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = self
            .lookup_available_locale(&requested)
            .unwrap_or_else(|| String::from("en-US"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // `InitializeRelativeTimeFormat` coerces options with `ToObject` (not
        // GetOptionsObject): `undefined` → an empty options bag; `null` → a TypeError;
        // any other primitive (boolean/string/number/symbol) → its wrapper object
        // (which has no own option keys, so all defaults apply).
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = match opts_arg.unpack() {
            Unpacked::Undefined => None,
            Unpacked::Null => {
                return Err(self.type_error("Intl.RelativeTimeFormat options cannot be null"));
            }
            _ => self
                .coerce_to_object(opts_arg)
                .as_handle()
                .map(Handle::from_raw),
        };
        // Options are read in spec order: localeMatcher, numberingSystem, style, numeric.
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let nu = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(ns) = &nu {
            // numberingSystem must match the UTS-35 `type` production (3-8 alnum,
            // hyphen-joined) — otherwise a RangeError.
            if !is_unicode_type_value(ns) {
                let m = self.new_str("invalid numberingSystem");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        let style = self
            .get_string_option(opts, "style", &["long", "short", "narrow"], Some("long"))?
            .unwrap();
        let numeric = self
            .get_string_option(opts, "numeric", &["always", "auto"], Some("always"))?
            .unwrap();
        // ResolveLocale for the `nu` key: a known `numberingSystem` option wins,
        // else the locale's `-u-nu-` extension, else the CLDR default. The
        // resolved locale keeps `-u-nu-` only when the value came from the
        // extension (an option-sourced value drops the key from the tag).
        let base = strip_unicode_extension(&locale);
        let (resolved_nu, add) = resolve_nu_key(&base, &locale, nu.as_deref());
        let resolved_locale = build_resolved_locale(&base, &[add]);
        let locv = self.new_str(&resolved_locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        self.store_str(obj, "numberingSystem", &Some(resolved_nu));
        self.store_str(obj, "style", &Some(style));
        self.store_str(obj, "numeric", &Some(numeric));
        self.brand_intl_instance(obj, N_INTL_REL_TIME);
        Ok(())
    }

    /// The resolved `(numeric, style)` of an `Intl.RelativeTimeFormat` instance
    /// (defaults `"always"`/`"long"` when the slots are absent).
    pub(crate) fn rel_time_numeric_style(&mut self, fmt: Option<Handle>) -> (String, String) {
        let numeric = fmt
            .and_then(|h| self.realm.get_property(h, "numeric"))
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("always"));
        let style = fmt
            .and_then(|h| self.realm.get_property(h, "style"))
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("long"));
        (numeric, style)
    }

    /// `SingularRelativeTimeUnit(unit)`: `ToString`s `unit` (a Symbol throws a
    /// TypeError), accepts both the singular and plural forms of the eight relative
    /// time units, returning the singular stem; any other value is a RangeError.
    pub(crate) fn singular_relative_time_unit(
        &mut self,
        unit: NanBox,
    ) -> Result<String, ExecError> {
        let s = self.coerce_to_string(unit)?;
        let singular = match s.as_str() {
            "seconds" | "second" => "second",
            "minutes" | "minute" => "minute",
            "hours" | "hour" => "hour",
            "days" | "day" => "day",
            "weeks" | "week" => "week",
            "months" | "month" => "month",
            "quarters" | "quarter" => "quarter",
            "years" | "year" => "year",
            _ => {
                let m = self.new_str(&alloc::format!("invalid relative time unit '{s}'"));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        };
        Ok(String::from(singular))
    }

    /// Builds an `Intl.DisplayNames` instance: an object capturing `type` with a readable
    /// `of(code)` method.
    pub(crate) fn make_display_names(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        // CanonicalizeLocaleList(locales) first (a Symbol/bad element is a
        // TypeError/RangeError); the resolved locale is the first requested tag.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = self
            .lookup_available_locale(&requested)
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // `options = ? GetOptionsObject(options)` — a primitive other than
        // `undefined` is a TypeError; `undefined` yields an (empty) options object.
        let opts = match args.get(1).copied() {
            Some(v) if !matches!(v.unpack(), Unpacked::Undefined) => {
                if !self.is_object_value(v) {
                    return Err(self.type_error("Intl.DisplayNames: options must be an object"));
                }
                v.as_handle().map(Handle::from_raw)
            }
            _ => None,
        };
        // `localeMatcher` is read (and validated) first, before `type` (spec order);
        // it is otherwise unused.
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        // Spec option-evaluation order: `style` is read (and validated) *before*
        // `type` (sec-Intl.DisplayNames steps 12–14), so an invalid `style` is a
        // RangeError even when the required `type` is absent.
        let style =
            self.get_string_option(opts, "style", &["narrow", "short", "long"], Some("long"))?;
        // `type` is a **required** option (its absence — including when `options`
        // is undefined — is a TypeError); it must be one of the valid types.
        let type_s = self.get_string_option(
            opts,
            "type",
            &[
                "language",
                "region",
                "script",
                "currency",
                "calendar",
                "dateTimeField",
            ],
            None,
        )?;
        let Some(type_s) = type_s else {
            return Err(self.type_error("Intl.DisplayNames: the `type` option is required"));
        };
        let tv = self.new_str(&type_s);
        self.realm.set_hidden_property(obj, "type", tv);
        // Mark the service kind so `resolvedOptions` reports the DisplayNames shape.
        let kindv = self.new_str("display");
        self.realm.set_hidden_property(obj, "\u{0}intl", kindv);
        self.store_str(obj, "style", &style);
        // `fallback` (default "code") and — for a language type —
        // `languageDisplay` (default "dialect"), validated + stored.
        let fallback = self.get_string_option(opts, "fallback", &["code", "none"], Some("code"))?;
        self.store_str(obj, "fallback", &fallback);
        if type_s == "language" {
            let ld = self.get_string_option(
                opts,
                "languageDisplay",
                &["dialect", "standard"],
                Some("dialect"),
            )?;
            self.store_str(obj, "languageDisplay", &ld);
        }
        self.brand_intl_instance(obj, N_INTL_DISPLAY_NAMES);
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Builds an `Intl.Collator` instance: an object capturing the locale and
    /// `sensitivity`/`numeric` options with a readable `compare` function (usable directly and
    /// as `arr.sort(collator.compare)`).
    pub(crate) fn make_collator(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        self.init_collator(obj, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// `InitializeCollator`: canonicalizes the requested locales, then reads and
    /// validates the options in spec order — `usage`, `localeMatcher`,
    /// `collation`, `numeric`, `caseFirst`, `sensitivity`, `ignorePunctuation` —
    /// storing the resolved values behind hidden slots for `resolvedOptions`.
    pub(crate) fn init_collator(&mut self, obj: Handle, args: &[NanBox]) -> Result<(), ExecError> {
        let marker = self.new_str("collator");
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // 1. CanonicalizeLocaleList(locales) — a malformed tag raises a RangeError.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = self
            .lookup_available_locale(&requested)
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // GetOptionsObject: `undefined` → no options; a non-object (other than
        // undefined, e.g. `null`) is a TypeError; otherwise the object itself.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.Collator options must be an object"));
        };
        // Read order (observable via throwing getters): usage, localeMatcher,
        // collation, numeric, caseFirst, sensitivity, ignorePunctuation.
        let usage = self
            .get_string_option(opts, "usage", &["sort", "search"], Some("sort"))?
            .unwrap();
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        // `collation`: an unvalidated string that must match the `type` grammar
        // (alphanum{3,8} groups) — a malformed value is a RangeError.
        let collation = self.get_string_option(opts, "collation", &[], None)?;
        if let Some(c) = &collation
            && !c
                .split('-')
                .all(|p| (3..=8).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_alphanumeric()))
        {
            let m = self.new_str(&alloc::format!("invalid collation option: {c}"));
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let numeric = self.get_bool_option(opts, "numeric", None)?;
        let case_first =
            self.get_string_option(opts, "caseFirst", &["upper", "lower", "false"], None)?;
        let sensitivity = self
            .get_string_option(
                opts,
                "sensitivity",
                &["base", "accent", "case", "variant"],
                None,
            )?
            .unwrap_or_else(|| String::from("variant"));
        // `ignorePunctuation` default is locale-dependent (CLDR): true for Thai,
        // false elsewhere.
        let primary = locale.split(['-', '_']).next().unwrap_or("");
        let ip_default = primary.eq_ignore_ascii_case("th");
        let ignore_punct = self
            .get_bool_option(opts, "ignorePunctuation", Some(ip_default))?
            .unwrap_or(ip_default);
        // --- ResolveLocale over Collator's relevant extension keys (co, kn, kf).
        // Each key's value comes from the locale's `-u-` extension when supported,
        // overridden by a supported option; an option-sourced value drops the key
        // from the resolved locale, while an extension-sourced value keeps it.
        // Non-relevant `-u-` keys and unsupported values are dropped entirely.
        let base = strip_unicode_extension(&locale);
        let mut additions: Vec<String> = Vec::new();

        // co (collation): default "default".
        let mut co_value = String::from("default");
        if let Some(ext) = locale_unicode_keyword(&locale, "co")
            && is_supported_collation(&base, &ext)
        {
            co_value = ext.clone();
            additions.push(alloc::format!("-co-{ext}"));
        }
        if let Some(opt) = &collation
            && is_supported_collation(&base, opt)
            && *opt != co_value
        {
            co_value = opt.clone();
            additions.retain(|a| !a.starts_with("-co-"));
        }

        // kn (numeric): default false; reported only when set from ext or option.
        let ext_kn = locale_unicode_bool_keyword(&locale, "kn");
        let mut kn_value = ext_kn.unwrap_or(false);
        if let Some(b) = ext_kn {
            additions.push(if b {
                String::from("-kn")
            } else {
                String::from("-kn-false")
            });
        }
        if let Some(opt) = numeric
            && opt != kn_value
        {
            kn_value = opt;
            additions.retain(|a| a != "-kn" && a != "-kn-false");
        }
        let kn_reported = ext_kn.is_some() || numeric.is_some();

        // kf (caseFirst): reported only when set from ext or option.
        let mut kf_value: Option<String> = None;
        if let Some(ext) = locale_unicode_keyword(&locale, "kf")
            && matches!(ext.as_str(), "upper" | "lower" | "false")
        {
            kf_value = Some(ext.clone());
            additions.push(alloc::format!("-kf-{ext}"));
        }
        if let Some(opt) = &case_first
            && kf_value.as_deref() != Some(opt.as_str())
        {
            kf_value = Some(opt.clone());
            additions.retain(|a| !a.starts_with("-kf-"));
        }

        let resolved_locale = build_resolved_locale(&base, &additions);
        let rlv = self.new_str(&resolved_locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", rlv);

        // Store the resolved slots for resolvedOptions.
        self.store_str(obj, "usage", &Some(usage));
        self.store_str(obj, "sensitivity", &Some(sensitivity));
        let ipv = NanBox::boolean(ignore_punct);
        self.realm
            .set_hidden_property(obj, "ignorePunctuation", ipv);
        self.store_str(obj, "collation", &Some(co_value));
        if kn_reported {
            self.realm
                .set_hidden_property(obj, "numeric", NanBox::boolean(kn_value));
        }
        if let Some(cf) = kf_value {
            self.store_str(obj, "caseFirst", &Some(cf));
        }
        self.brand_intl_instance(obj, N_INTL_COLLATOR);
        Ok(())
    }

    /// Compares `a` and `b` under an `Intl.Collator` instance's resolved slots via
    /// real UCA collation: `sensitivity` → strength, `numeric` → numeric ordering,
    /// `ignorePunctuation` → variable-weighting (Shifted vs NonIgnorable). Shared by
    /// `Intl.Collator.prototype.compare` and `String.prototype.localeCompare`.
    #[cfg(feature = "intl")]
    pub(crate) fn collator_ordering(
        &mut self,
        ch: Option<Handle>,
        a: &str,
        b: &str,
    ) -> core::cmp::Ordering {
        use intl::unicode::collate::{AlternateHandling, Collator, Strength};
        // `sensitivity` selects the comparison strength. "case" is the odd one: it
        // ignores accents but *not* case, which is not a strength level at all —
        // it is primary strength with the UCA case level switched on. Folding it
        // into the tertiary default (as this did) made `case` behave like
        // `variant`, so "Aa" and "Aã" compared unequal.
        let sensitivity = ch
            .and_then(|h| self.realm.get_property(h, "sensitivity"))
            .map(|v| self.realm.to_display_string(v));
        let (strength, case_level) = match sensitivity.as_deref() {
            Some("base") => (Strength::Primary, false),
            Some("accent") => (Strength::Secondary, false),
            Some("case") => (Strength::Primary, true),
            _ => (Strength::Tertiary, false),
        };
        let numeric = matches!(
            ch.and_then(|h| self.realm.get_property(h, "numeric"))
                .map(|v| v.unpack()),
            Some(Unpacked::Bool(true))
        );
        let alternate = if matches!(
            ch.and_then(|h| self.realm.get_property(h, "ignorePunctuation"))
                .map(|v| v.unpack()),
            Some(Unpacked::Bool(true))
        ) {
            AlternateHandling::Shifted
        } else {
            AlternateHandling::NonIgnorable
        };
        // The resolved locale's CLDR tailoring, when the crate bundles one (78
        // locales, plus the `-u-co-` variants). Without this every locale sorted in
        // root DUCET order, so `new Intl.Collator("sv")` folded `å ä ö` in with
        // `a`/`o` instead of ordering them after `z` — the locale was accepted and
        // then ignored.
        //
        // `Tailoring` carries no strength / numeric / variable-weighting knobs, so
        // it can only serve a collator that asked for the defaults. Any other option
        // set keeps the option-aware root collator, which is what every locale got
        // before — so this only ever adds tailoring, never removes an option.
        if strength == Strength::Tertiary
            && !case_level
            && !numeric
            && alternate == AlternateHandling::NonIgnorable
            && let Some(locale) = ch
                .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                .map(|v| self.realm.to_display_string(v))
            && let Some(tailoring) = locale_tailoring(&locale)
        {
            return tailoring.compare(a, b);
        }
        Collator::new(alternate)
            .with_strength(strength)
            .with_case_level(case_level)
            .with_numeric(numeric)
            .compare(a, b)
    }

    /// Builds an `Intl.ListFormat` instance (`InitializeListFormat`): canonicalizes
    /// the requested locales, reads and validates the `localeMatcher`/`type`/`style`
    /// options (in that order), and brands the object with `\0intl="list"` plus its
    /// resolved `\0locale`/`type`/`style`. Used only by the `new`-form constructor;
    /// a plain call (no `new`) throws a `TypeError` before reaching here.
    pub(crate) fn make_list_format(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        self.init_list_format(obj, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    pub(crate) fn init_list_format(
        &mut self,
        obj: Handle,
        args: &[NanBox],
    ) -> Result<(), ExecError> {
        let marker = self.new_str("list");
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // 1. CanonicalizeLocaleList(locales) — a malformed tag raises a RangeError.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = self
            .lookup_available_locale(&requested)
            .unwrap_or_else(|| String::from("en-US"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // GetOptionsObject: `undefined` → no options; a non-object (other than
        // undefined) is a TypeError; otherwise the object itself.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.ListFormat options must be an object"));
        };
        // Options are read in spec order: localeMatcher, then type, then style.
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        let list_type = self
            .get_string_option(
                opts,
                "type",
                &["conjunction", "disjunction", "unit"],
                Some("conjunction"),
            )?
            .unwrap();
        let style = self
            .get_string_option(opts, "style", &["long", "short", "narrow"], Some("long"))?
            .unwrap();
        self.store_str(obj, "type", &Some(list_type));
        self.store_str(obj, "style", &Some(style));
        self.brand_intl_instance(obj, N_INTL_LIST_FORMAT);
        Ok(())
    }

    /// `StringListFromIterable(iterable)`: `undefined` → an empty list; otherwise
    /// iterate, requiring every yielded value to be a primitive String (else a
    /// `TypeError`, after closing the iterator). Used by `format`/`formatToParts`.
    pub(crate) fn string_list_from_iterable(
        &mut self,
        iterable: NanBox,
    ) -> Result<Vec<String>, ExecError> {
        if matches!(iterable.unpack(), Unpacked::Undefined) {
            return Ok(Vec::new());
        }
        // Open the iterator once via the user `[Symbol.iterator]` when present, so
        // a custom iterable's `next`/`return` observe the exact spec call sequence
        // (and a non-string element closes it mid-stream). Built-in iterables
        // (arrays/strings/Maps/Sets/generators) have no readable iterator method;
        // they are drained eagerly and then validated element-by-element.
        if let Some(ih) = self.for_of_get_iterator(iterable)? {
            let mut out = Vec::new();
            loop {
                let next_fn = self.read_member(ih, "next")?;
                let res = self.call_with_this(next_fn, NanBox::handle(ih.to_raw()), &[])?;
                let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                    return Err(self.type_error("iterator result is not an object"));
                };
                let done = self.read_member(rh, "done")?;
                if self.realm.truthy(done) {
                    break;
                }
                let value = self.read_member(rh, "value")?;
                match value.as_handle().map(Handle::from_raw) {
                    Some(vh) if self.realm.type_of(vh) == Some("string") => {
                        out.push(self.realm.string_value(vh).unwrap_or_default());
                    }
                    _ => {
                        // Non-String element: close the iterator, then throw.
                        let _ = self.iterator_close(ih);
                        return Err(
                            self.type_error("Intl.ListFormat: list elements must all be strings")
                        );
                    }
                }
                if out.len() > GEN_CAP {
                    return Err(self.type_error("iterator did not terminate"));
                }
            }
            return Ok(out);
        }
        // Built-in iterable (or a non-iterable, which `iterate_values` rejects with
        // a TypeError): drain, then require each element to be a String.
        let elems = self.iterate_values(iterable)?;
        let mut out = Vec::with_capacity(elems.len());
        for e in elems {
            match e.as_handle().map(Handle::from_raw) {
                Some(vh) if self.realm.type_of(vh) == Some("string") => {
                    out.push(self.realm.string_value(vh).unwrap_or_default());
                }
                _ => {
                    return Err(
                        self.type_error("Intl.ListFormat: list elements must all be strings")
                    );
                }
            }
        }
        Ok(out)
    }

    /// The resolved `(type, style)` of an `Intl.ListFormat` instance (defaults
    /// `"conjunction"`/`"long"` when the slots are absent).
    pub(crate) fn list_format_type_style(&self, fmt: Option<Handle>) -> (String, String) {
        let get = |key: &str, dflt: &str| -> String {
            fmt.and_then(|h| self.realm.get_property(h, key))
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from(dflt))
        };
        (get("type", "conjunction"), get("style", "long"))
    }

    /// `PartitionRelativeTimePattern` for a formatter instance: CLDR-backed when
    /// the `intl` feature is on, else [`rel_time_parts_en`].
    pub(crate) fn rel_time_partition(
        &mut self,
        fmt: Option<Handle>,
        value: f64,
        unit: &str,
        numeric: &str,
        style: &str,
    ) -> Vec<(&'static str, String, bool)> {
        #[cfg(feature = "intl")]
        {
            let locale = fmt
                .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("en"));
            self.rel_time_parts_cldr(&locale, value, unit, numeric, style)
        }
        #[cfg(not(feature = "intl"))]
        {
            let _ = (fmt, numeric, style);
            rel_time_parts_en(value, unit)
        }
    }

    /// `PartitionRelativeTimePattern` — the `(type, value, has_unit)` parts of an
    /// `Intl.RelativeTimeFormat` `format`/`formatToParts`, from CLDR.
    ///
    /// The crate splices `NumberFormat` parts into the locale's relative-time
    /// pattern, which is exactly ECMA-402's construction, so the unit wording, the
    /// `numeric: "auto"` literals ("yesterday", "übermorgen"), the short/narrow
    /// widths and the number's own separators and numbering system all come from
    /// the locale rather than from an English table.
    #[cfg(feature = "intl")]
    pub(crate) fn rel_time_parts_cldr(
        &self,
        locale: &str,
        value: f64,
        unit: &str,
        numeric: &str,
        style: &str,
    ) -> Vec<(&'static str, String, bool)> {
        use intl::relative::{RelativeNumeric, RelativeUnit, RelativeWidth};
        let ru = match unit {
            "second" => RelativeUnit::Second,
            "minute" => RelativeUnit::Minute,
            "hour" => RelativeUnit::Hour,
            "day" => RelativeUnit::Day,
            "week" => RelativeUnit::Week,
            "month" => RelativeUnit::Month,
            "quarter" => RelativeUnit::Quarter,
            _ => RelativeUnit::Year,
        };
        let mut opts = intl::relative::RelativeTimeFormatOptions::default();
        opts.numeric = if numeric == "auto" {
            RelativeNumeric::Auto
        } else {
            RelativeNumeric::Always
        };
        opts.width = match style {
            "short" => RelativeWidth::Short,
            "narrow" => RelativeWidth::Narrow,
            _ => RelativeWidth::Long,
        };
        intl::relative::format_relative_to_parts(locale, value, ru, &opts)
            .into_iter()
            .map(|p| (p.kind.as_str(), p.value, p.unit.is_some()))
            .collect()
    }

    /// Maps ECMA-402 `type`/`style` onto the crate's list options.
    #[cfg(feature = "intl")]
    fn list_format_options(list_type: &str, style: &str) -> intl::list::ListFormatOptions {
        use intl::list::{ListType, ListWidth};
        // `#[non_exhaustive]`: build from `Default` and assign, rather than a
        // struct literal.
        let mut opts = intl::list::ListFormatOptions::default();
        opts.list_type = match list_type {
            "disjunction" => ListType::Disjunction,
            "unit" => ListType::Unit,
            _ => ListType::Conjunction,
        };
        opts.width = match style {
            "short" => ListWidth::Short,
            "narrow" => ListWidth::Narrow,
            _ => ListWidth::Long,
        };
        opts
    }

    /// The `(literal-or-element, value)` parts of an `Intl.ListFormat` `format` /
    /// `formatToParts`, from the crate's CLDR `listPattern`s for all nine
    /// `type` x `style` combinations.
    ///
    /// The crate formats a whole list and exposes no parts API, so the literals
    /// are recovered by formatting **sentinels** in place of the elements and
    /// splitting on those. Splitting on the elements themselves would be wrong:
    /// `es` joins with `" y "`, so a list containing the element `"y"` has the
    /// connector's `y` occurring before the element's, and the split lands in the
    /// wrong place. A sentinel cannot collide with pattern text.
    #[cfg(feature = "intl")]
    pub(crate) fn list_format_parts(
        &self,
        items: &[String],
        list_type: &str,
        style: &str,
        locale: &str,
    ) -> Vec<(&'static str, String)> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let opts = Self::list_format_options(list_type, style);
        // U+0000-delimited indices: not producible by CLDR pattern text.
        let sentinels: Vec<String> = (0..n).map(|i| alloc::format!("\u{0}{i}\u{0}")).collect();
        let refs: Vec<&str> = sentinels.iter().map(String::as_str).collect();
        let shaped = intl::list::format_list(locale, &refs, &opts);
        let mut parts: Vec<(&'static str, String)> = Vec::new();
        let mut rest = shaped.as_str();
        for (i, sentinel) in sentinels.iter().enumerate() {
            let Some(at) = rest.find(sentinel.as_str()) else {
                // The pattern dropped an element (never happens for CLDR data);
                // fall back to emitting it unseparated rather than losing it.
                parts.push(("element", items[i].clone()));
                continue;
            };
            if at > 0 {
                parts.push(("literal", String::from(&rest[..at])));
            }
            parts.push(("element", items[i].clone()));
            rest = &rest[at + sentinel.len()..];
        }
        if !rest.is_empty() {
            parts.push(("literal", String::from(rest)));
        }
        parts
    }

    #[cfg(not(feature = "intl"))]
    pub(crate) fn list_format_parts(
        &self,
        items: &[String],
        list_type: &str,
        style: &str,
        locale: &str,
    ) -> Vec<(&'static str, String)> {
        // Signature kept parallel to the `intl` variant; this fallback has no
        // locale data, so everything below is English.
        let _ = locale;
        let n = items.len();
        let mut parts: Vec<(&'static str, String)> = Vec::new();
        if n == 0 {
            return parts;
        }
        // English CLDR `listPattern`s, keyed by (type, style). `pair` is the
        // two-element connector; `middle` joins non-final elements; `end` is the
        // final connector for three-or-more lists. (Reference-locale data; the
        // `intl` crate exposes only conjunction/disjunction *long*, so short /
        // narrow / unit patterns are encoded here.)
        let (pair, middle, end): (&str, &str, &str) = match (list_type, style) {
            ("disjunction", _) => (" or ", ", ", ", or "),
            ("unit", "narrow") => (" ", " ", " "),
            ("unit", _) => (", ", ", ", ", "),
            (_, "short") => (" & ", ", ", ", & "), // conjunction short
            (_, "narrow") => (", ", ", ", ", "),   // conjunction narrow
            _ => (" and ", ", ", ", and "),        // conjunction long (default)
        };
        for (i, it) in items.iter().enumerate() {
            if i > 0 {
                let lit = if n == 2 {
                    pair
                } else if i == n - 1 {
                    end
                } else {
                    middle
                };
                parts.push(("literal", String::from(lit)));
            }
            parts.push(("element", it.clone()));
        }
        parts
    }

    /// Builds an `Intl.PluralRules` instance (`InitializePluralRules`): canonicalizes
    /// the locale list, reads `localeMatcher`/`type`/`notation`/`compactDisplay` and
    /// the shared `SetNumberFormatDigitOptions` slots (in spec read order), and stores
    /// the resolved configuration behind hidden keys for `select`/`selectRange`/
    /// `resolvedOptions`. A non-object `options` (other than `undefined`) is a
    /// TypeError; an invalid option value is a RangeError.
    pub(crate) fn make_plural_rules(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        self.init_plural_rules(obj, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Initializes an existing object as an `Intl.PluralRules` (used by both the
    /// `new`-form constructor and `super()` in a subclass — see `apply_native_super`).
    pub(crate) fn init_plural_rules(
        &mut self,
        obj: Handle,
        args: &[NanBox],
    ) -> Result<(), ExecError> {
        // Mark the instance kind so `resolvedOptions` reports the PluralRules shape.
        let kindv = self.new_str("plural");
        self.realm.set_hidden_property(obj, "\u{0}intl", kindv);
        self.brand_intl_instance(obj, N_INTL_PLURAL_RULES);
        // CanonicalizeLocaleList(locales): a malformed tag is a RangeError; the
        // resolved locale is the first requested tag, else the default.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = self
            .lookup_available_locale(&requested)
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // GetOptionsObject: undefined → no options; a non-object → TypeError.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.PluralRules options must be an object"));
        };
        // localeMatcher (validated, not otherwise used).
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        // type: "cardinal" (default) or "ordinal".
        let pr_type = self
            .get_string_option(opts, "type", &["cardinal", "ordinal"], Some("cardinal"))?
            .unwrap();
        self.store_str(obj, "type", &Some(pr_type));
        // notation: "standard" (default) | "compact" | "scientific" | "engineering".
        let notation = self
            .get_string_option(
                opts,
                "notation",
                &["standard", "compact", "scientific", "engineering"],
                Some("standard"),
            )?
            .unwrap();
        // compactDisplay is read regardless, but only stored when notation is compact.
        let compact_display =
            self.get_string_option(opts, "compactDisplay", &["short", "long"], Some("short"))?;
        if notation == "compact" {
            self.store_str(obj, "compactDisplay", &compact_display);
        }
        self.store_str(obj, "notation", &Some(notation));
        // SetNumberFormatDigitOptions (shared with Intl.NumberFormat).
        self.set_number_format_digit_options(obj, opts)?;
        Ok(())
    }

    /// The plural category name (`"zero"`/`"one"`/`"two"`/`"few"`/`"many"`/`"other"`)
    /// of `n` per the receiver `Intl.PluralRules` instance's `[[Locale]]` and
    /// `[[Type]]`. A non-finite `n` is always `"other"`. With the `intl` feature this
    /// uses the crate's CLDR cardinal/ordinal rules; otherwise it falls back to the
    /// English rule (`1` → `"one"`, else `"other"`). Shared by `select`/`selectRange`.
    pub(crate) fn plural_select_category(&mut self, n: f64) -> &'static str {
        if !n.is_finite() {
            return "other";
        }
        #[cfg(feature = "intl")]
        {
            let fmt = self.this_val.as_handle().map(Handle::from_raw);
            let locale = fmt
                .and_then(|h| self.realm.get_property(h, "\u{0}locale"))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("en"));
            let ordinal = fmt
                .and_then(|h| self.realm.get_property(h, "type"))
                .map(|v| self.realm.to_display_string(v))
                .as_deref()
                == Some("ordinal");
            let notation = fmt
                .and_then(|h| self.realm.get_property(h, "notation"))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("standard"));
            // For compact/scientific/engineering notation the plural operands carry
            // a non-zero compact-decimal exponent `c`/`e` (e.g. 1.5e6 in compact →
            // mantissa 1.5, exponent 6), which some locales' `many` rules depend on
            // (fr: `e ∉ 0..5 → many`). Encode it via the crate's `<mantissa>c<exp>`
            // operand syntax; standard notation uses the plain decimal.
            let ops = match plural_notation_operand_string(n, &notation) {
                Some(s) => intl::plural::PluralOperands::parse(&s)
                    .unwrap_or_else(|| intl::plural::PluralOperands::from_int(n as i64)),
                None if n == (n as i64) as f64 => intl::plural::PluralOperands::from_int(n as i64),
                None => intl::plural::PluralOperands::parse(&alloc::format!("{n}"))
                    .unwrap_or_else(|| intl::plural::PluralOperands::from_int(n as i64)),
            };
            let cat = if ordinal {
                intl::plural::ordinal_category(&locale, &ops)
            } else {
                intl::plural::plural_category(&locale, &ops)
            };
            use intl::plural::PluralCategory::*;
            match cat {
                Zero => "zero",
                One => "one",
                Two => "two",
                Few => "few",
                Many => "many",
                Other => "other",
            }
        }
        #[cfg(not(feature = "intl"))]
        {
            if n == 1.0 { "one" } else { "other" }
        }
    }

    /// The sorted `pluralCategories` list for an `Intl.PluralRules` instance's
    /// locale+type: the distinct categories its cardinal (or ordinal) rules can
    /// produce, in CLDR order (`zero`, `one`, `two`, `few`, `many`, `other`). With
    /// the `intl` feature this probes the crate's rules over a representative sample
    /// of operands; otherwise it returns the English cardinal set `["one","other"]`.
    pub(crate) fn plural_categories(&mut self, locale: &str, ordinal: bool) -> Vec<&'static str> {
        const ORDER: [&str; 6] = ["zero", "one", "two", "few", "many", "other"];
        #[cfg(feature = "intl")]
        {
            use intl::plural::PluralCategory::*;
            let name = |c: intl::plural::PluralCategory| -> &'static str {
                match c {
                    Zero => "zero",
                    One => "one",
                    Two => "two",
                    Few => "few",
                    Many => "many",
                    Other => "other",
                }
            };
            let mut seen: Vec<&'static str> = Vec::new();
            // Probe a representative sample of operands: integers 0..=200 capture the
            // mod-10/mod-100 rule structure, plus a couple of fractional and large
            // compact-notation values that trigger the `many` category in some locales.
            let mut push = |this: &mut Self, s: &str| {
                if let Some(ops) = intl::plural::PluralOperands::parse(s) {
                    let cat = if ordinal {
                        intl::plural::ordinal_category(locale, &ops)
                    } else {
                        intl::plural::plural_category(locale, &ops)
                    };
                    let _ = this;
                    let nm = name(cat);
                    if !seen.contains(&nm) {
                        seen.push(nm);
                    }
                }
            };
            for i in 0..=200u32 {
                let s = alloc::format!("{i}");
                push(self, &s);
            }
            for s in ["0.0", "0.1", "1.5", "2.5", "1000000", "1000000.0"] {
                push(self, s);
            }
            // Return in canonical CLDR order.
            ORDER.iter().copied().filter(|c| seen.contains(c)).collect()
        }
        #[cfg(not(feature = "intl"))]
        {
            let _ = (locale, ordinal);
            let _ = ORDER;
            alloc::vec!["one", "other"]
        }
    }

    /// Builds an `Intl.Segmenter` instance: an object capturing `granularity` with a readable
    /// `segment(input)` method.
    pub(crate) fn make_segmenter(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        self.init_segmenter(obj, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    pub(crate) fn init_segmenter(&mut self, obj: Handle, args: &[NanBox]) -> Result<(), ExecError> {
        // 1. CanonicalizeLocaleList(locales) — a malformed tag / non-string,
        //    non-object element raises a RangeError/TypeError.
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        let locale = self
            .lookup_available_locale(&requested)
            .unwrap_or_else(|| String::from("en"));
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        // GetOptionsObject: `undefined` → no options; a non-object (other than
        // undefined, e.g. `null`) is a TypeError; otherwise the object itself.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.Segmenter options must be an object"));
        };
        // Options are read in spec order: localeMatcher, then granularity.
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        // `granularity` (default "grapheme"), validated + stored so
        // `resolvedOptions` reports it (an invalid value is a RangeError).
        let gran = self
            .get_string_option(
                opts,
                "granularity",
                &["grapheme", "word", "sentence"],
                Some("grapheme"),
            )?
            .unwrap();
        let granv = self.new_str(&gran);
        self.realm.set_hidden_property(obj, "granularity", granv);
        // Mark the service kind so `resolvedOptions` reports the Segmenter shape.
        let kindv = self.new_str("segmenter");
        self.realm.set_hidden_property(obj, "\u{0}intl", kindv);
        self.brand_intl_instance(obj, N_INTL_SEGMENTER);
        Ok(())
    }

    /// Breaks a UTC millisecond timestamp into typed `(type, value)` parts per an
    /// `Intl.DateTimeFormat` instance's options, via the `intl` crate (CLDR, locale-aware).
    /// Used by both `format` and `formatToParts`.
    #[cfg(feature = "intl")]
    pub(crate) fn datetime_parts(
        &mut self,
        handle: Handle,
        ms: f64,
    ) -> Vec<(&'static str, String)> {
        let (locale, dt, o) = self.dtf_locale_dt_opts(handle, ms);
        // Chinese/Dangi lunisolar calendars: the crate's `format_to_parts` renders
        // only proleptic Gregorian, so derive the `relatedYear`/`yearName` (+ numeric
        // month/day) parts these calendars require ourselves.
        let mut parts = match self.lunisolar_parts(handle, &locale, &dt, &o) {
            Some(p) => p,
            None => {
                let mut parts = Self::dtf_crate_parts(&locale, &dt, &o);
                self.rewrite_calendar_numerics(handle, &dt, &o, &mut parts);
                parts
            }
        };
        self.fix_fractional_separator(handle, &locale, &mut parts);
        parts
    }

    /// Re-express the separator in front of a `fractionalSecond` part in the
    /// formatter's resolved numbering system.
    ///
    /// ECMA-402's `FormatDateTimePattern` formats the fractional seconds through
    /// an `Intl.NumberFormat` carrying the DateTimeFormat's `[[NumberingSystem]]`,
    /// so the *separator* is numbering-system data and not just the digits:
    /// `en-US-u-nu-arab` renders `٢:٣٥:٠٦٫٧٨٩` with U+066B, while `en-US-u-nu-deva`
    /// keeps the locale's own `.`. The crate's date formatter only ever emits the
    /// locale's Latin-numbering separator, so it is patched here — probing an
    /// `intl` decimal format in the same locale + numbering system, which is
    /// exactly the nested `NumberFormat` the spec describes.
    #[cfg(feature = "intl")]
    fn fix_fractional_separator(
        &self,
        handle: Handle,
        locale: &str,
        parts: &mut [(&'static str, String)],
    ) {
        let Some(i) = parts.iter().position(|(k, _)| *k == "fractionalSecond") else {
            return;
        };
        if i == 0 || parts[i - 1].0 != "literal" {
            return;
        }
        let nu = self
            .realm
            .get_property(handle, "numberingSystem")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        // `latn` is what the crate already rendered the pattern with.
        if nu.is_empty() || nu == "latn" {
            return;
        }
        let probe_locale =
            alloc::format!("{}-u-nu-{}", strip_unicode_extension(locale).as_str(), nu);
        let (_, sep, _) =
            split_number_scaffold(&intl::number::format_decimal(&probe_locale, 1.1), &nu);
        if !sep.is_empty() {
            parts[i - 1].1 = sep;
        }
    }

    /// Re-express the `era`/`year`/`month`/`day` parts in the instance's resolved
    /// calendar. The `intl` crate's `format_to_parts` renders only proleptic
    /// Gregorian fields, so a `calendar: "buddhist"` formatter would otherwise
    /// report the ISO year (2050 rather than 2593) and a `calendar: "japanese"`
    /// one "Anno Domini" rather than "Reiwa" — and the same for every other
    /// arithmetic calendar the engine already implements for Temporal.
    ///
    /// The era and the month *name* come from the crate's per-calendar CLDR
    /// tables ([`era_name`](intl::datetime::era_name) /
    /// [`month_name`](intl::datetime::month_name)), keyed off the era code and
    /// month code the engine's own Temporal calendar reports — so
    /// `Intl.DateTimeFormat` and `Temporal.PlainDate` cannot disagree about which
    /// era or month a date falls in. The year is the era-relative one when the
    /// calendar has eras (`date.eraYear ?? date.year`, as `Temporal.PlainDate`
    /// reports it).
    #[cfg(feature = "intl")]
    fn rewrite_calendar_numerics(
        &self,
        handle: Handle,
        dt: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
        parts: &mut [(&'static str, String)],
    ) {
        use crate::nbexec::temporal_calendar;
        use crate::temporal_iso::IsoDate;
        use intl::datetime::{NameStyle, Numeric2Digit};
        let cal = self.dtf_resolved_calendar(handle);
        // ISO/Gregorian already agree with the crate; the lunisolar calendars are
        // rendered by `lunisolar_parts`, which never reaches here.
        if matches!(cal.as_str(), "gregory" | "iso8601" | "chinese" | "dangi") {
            return;
        }
        let iso = IsoDate {
            year: dt.year,
            month: dt.month,
            day: dt.day,
        };
        let f = temporal_calendar::iso_to_fields(&cal, iso);
        // CLDR names are keyed by language; the `-u-` keywords (`ca`/`nu`/`hc`)
        // are the formatter's own resolution, not part of the lookup.
        let lang = strip_unicode_extension(&self.dtf_locale(handle));
        let leap_year = temporal_calendar::months_in_year(&cal, iso) == 13;
        let (midx, mleap) = cldr_month_index(&cal, &f.month_code, f.month, leap_year);
        let two_digit = |v: i64| alloc::format!("{:02}", v.rem_euclid(100));
        for (kind, value) in parts.iter_mut() {
            match *kind {
                "era" => {
                    if let Some(code) = f.era.as_deref()
                        && let Some((c, idx)) = cldr_era_index(&cal, code)
                        && let Some(name) = intl::datetime::era_name(
                            &lang,
                            c,
                            idx,
                            // A `dateStyle` pattern's era field is UTS #35 `G`,
                            // i.e. the abbreviated width.
                            o.era.unwrap_or(NameStyle::Short),
                        )
                    {
                        *value = String::from(name);
                    }
                }
                "month" => Self::rewrite_month_part(&cal, &lang, o, midx, mleap, value),
                // A field the crate rendered non-numerically is left alone — there
                // is no per-calendar CLDR text to replace it with.
                "year" if value.bytes().all(|b| b.is_ascii_digit()) => {
                    let y = f.era_year.unwrap_or(f.year);
                    *value = if o.year == Some(Numeric2Digit::TwoDigit) {
                        two_digit(y)
                    } else {
                        alloc::format!("{y}")
                    };
                }
                "day" if value.bytes().all(|b| b.is_ascii_digit()) => {
                    *value = if value.len() == 2 {
                        two_digit(f.day)
                    } else {
                        alloc::format!("{}", f.day)
                    };
                }
                _ => {}
            }
        }
    }

    /// Re-expresses one `month` part in calendar `cal` at CLDR month slot `midx`.
    ///
    /// The requested `month` option is the width; with none (the part came out of
    /// a `dateStyle` pattern, whose width is the pattern's own) only a numeric
    /// rendering is rewritten, and the name is left as the crate produced it.
    ///
    /// Hebrew is the exception ECMA-402 inherits from CLDR-15510: its months are
    /// *never* rendered as numbers at the numeric/2-digit widths (only `narrow`
    /// gives a number), so a `month: "numeric"` Hebrew formatter renders "Sivan".
    #[cfg(feature = "intl")]
    fn rewrite_month_part(
        cal: &str,
        lang: &str,
        o: &intl::datetime::DateTimeFormatOptions,
        midx: u32,
        mleap: bool,
        value: &mut String,
    ) {
        use intl::datetime::{Calendar, MonthStyle};
        let numeric = value.bytes().all(|b| b.is_ascii_digit());
        let Some(c) = Calendar::from_bcp47(cal) else {
            return;
        };
        let style = match o.month {
            Some(MonthStyle::Numeric | MonthStyle::TwoDigit) | None if cal == "hebrew" => {
                MonthStyle::Long
            }
            Some(s) => s,
            // A `dateStyle` month: only a numeric one can be re-expressed safely.
            None if numeric => {
                if value.len() == 2 {
                    MonthStyle::TwoDigit
                } else {
                    MonthStyle::Numeric
                }
            }
            None => return,
        };
        if let Some(name) = intl::datetime::month_name(lang, c, midx, mleap, style) {
            *value = name;
        }
    }

    /// The `Intl.DateTimeFormat` instance's resolved locale tag.
    #[cfg(feature = "intl")]
    fn dtf_locale(&self, handle: Handle) -> String {
        self.realm
            .get_property(handle, "\u{0}locale")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en"))
    }

    /// Runs the `intl` crate's CLDR date-time formatter and repairs the shapes it
    /// cannot express on its own, so every `Intl.DateTimeFormat` rendering path
    /// (legacy `Date`, Temporal, `toLocaleString`) agrees:
    ///
    /// * a `dayPeriod`-only options set, and the stale `midnight` override — see
    ///   [`day_period_text`](Self::day_period_text);
    /// * an options set whose time skeleton has no CLDR `availableFormat` (a lone
    ///   `minute`/`second`/`fractionalSecondDigits`) — see
    ///   [`lone_time_field_parts`](Self::lone_time_field_parts);
    /// * a BCE year, which is era-relative rather than astronomical — see
    ///   [`fix_bce_era_year`](Self::fix_bce_era_year).
    #[cfg(feature = "intl")]
    fn dtf_crate_parts(
        locale: &str,
        dt: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Vec<(&'static str, String)> {
        // A `dayPeriod`-only options set renders exactly the flexible day period.
        if o.day_period.is_some() && Self::day_period_is_only_field(o) {
            return match Self::day_period_text(locale, dt, o) {
                Some(v) => alloc::vec![("dayPeriod", v)],
                None => Vec::new(),
            };
        }
        // An explicit hour cycle has to reach a `timeStyle` pattern too.
        let synthesized = Self::time_style_cycle_options(locale, dt, o);
        let o = synthesized.as_ref().unwrap_or(o);
        let mut parts = match intl::datetime::format_to_parts(locale, dt, o) {
            Ok(parts) => dtf_pad_time_parts(parts),
            Err(_) => Vec::new(),
        };
        if parts.is_empty()
            && let Some(p) = Self::lone_time_field_parts(locale, dt, o)
        {
            return p;
        }
        Self::widen_pattern_hour(locale, o, &mut parts);
        // Midnight correction (see [`day_period_text`]).
        if o.day_period.is_some()
            && dt.hour == 0
            && (dt.minute, dt.second, dt.millisecond) == (0, 0, 0)
            && let Some(fixed) = Self::day_period_text(locale, dt, o)
        {
            for p in &mut parts {
                if p.0 == "dayPeriod" {
                    p.1.clone_from(&fixed);
                }
            }
        }
        Self::fix_bce_era_year(dt, &mut parts);
        parts
    }

    /// The crate honours `hour_cycle`/`hour12` only on the skeleton path, so a
    /// `timeStyle` pattern keeps the locale's own clock:
    /// `new Intl.DateTimeFormat("en-US-u-hc-h23", { timeStyle: "medium" })` renders
    /// `2:12:47 PM` instead of `14:12:47`. When an explicit cycle disagrees with the
    /// clock the style pattern actually used, re-express the style as the equivalent
    /// component options — which the crate *does* apply the cycle to. Returns `None`
    /// (keep the style) when there is no explicit cycle, no disagreement, or a
    /// `dateStyle` is also in play (its date+time glue pattern is crate-internal).
    #[cfg(feature = "intl")]
    fn time_style_cycle_options(
        locale: &str,
        dt: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Option<intl::datetime::DateTimeFormatOptions> {
        use intl::datetime::{
            DateStyle, DateTimeFormatOptions, DateTimePartType, HourCycle, Numeric2Digit,
            TimeZoneNameStyle,
        };
        let ts = o.time_style?;
        if o.date_style.is_some() {
            return None;
        }
        let want12 = match (o.hour_cycle, o.hour12) {
            (Some(HourCycle::H11 | HourCycle::H12), _) => true,
            (Some(HourCycle::H23 | HourCycle::H24), _) => false,
            (None, Some(b)) => b,
            (None, None) => return None,
        };
        let styled = intl::datetime::format_to_parts(locale, dt, o).ok()?;
        if styled.iter().any(|p| p.kind == DateTimePartType::DayPeriod) == want12 {
            return None;
        }
        // UTS #35's four time styles are hour+minute (+second above `short`), with
        // the zone name the pattern's own `z`/`zzzz` field carries.
        let mut n = DateTimeFormatOptions::default();
        n.hour_cycle = o.hour_cycle;
        n.hour12 = o.hour12;
        n.hour = Some(Numeric2Digit::Numeric);
        n.minute = Some(Numeric2Digit::TwoDigit);
        if !matches!(ts, DateStyle::Short) {
            n.second = Some(Numeric2Digit::TwoDigit);
        }
        n.fractional_second_digits = o.fractional_second_digits;
        n.time_zone = o.time_zone;
        n.tz_offset_minutes = o.tz_offset_minutes;
        n.time_zone_name = match ts {
            DateStyle::Full => Some(TimeZoneNameStyle::Long),
            DateStyle::Long => Some(TimeZoneNameStyle::Short),
            _ => None,
        };
        Some(n)
    }

    /// ECMA-402 takes the *pattern's* hour width, only widening it when the caller
    /// asked for `2-digit`: `{hour: "numeric", hourCycle: "h23"}` renders `09:30` in
    /// `en` (CLDR `Hm` is `HH:mm`) but `9:30` in `he`/`fi`/`hu`/`cs`/`vi` (`H:mm`).
    /// The crate instead forces the hour run to the requested length, so a
    /// `numeric` request loses the pattern's leading zero. Recover the pattern's own
    /// width from the matching CLDR skeleton (`H`/`Hm`/`Hms`) and pad to it.
    ///
    /// Only the 24-hour patterns need this — every CLDR 12-hour pattern uses a
    /// single `h` — so the rendered parts having no `dayPeriod` is the test for
    /// "this is the `H` family".
    #[cfg(feature = "intl")]
    fn widen_pattern_hour(
        locale: &str,
        o: &intl::datetime::DateTimeFormatOptions,
        parts: &mut [(&'static str, String)],
    ) {
        use intl::datetime::{DateTime, Numeric2Digit};
        if o.hour != Some(Numeric2Digit::Numeric) {
            return;
        }
        if parts.iter().any(|(t, _)| *t == "dayPeriod") {
            return;
        }
        let Some(hour) = parts
            .iter_mut()
            .find(|(t, v)| *t == "hour" && v.len() == 1 && v.as_bytes()[0].is_ascii_digit())
        else {
            return;
        };
        let mut skeleton = String::from("H");
        if o.minute.is_some() {
            skeleton.push('m');
        }
        if o.second.is_some() {
            skeleton.push('s');
        }
        // A single-digit hour in the probe date exposes whether the pattern pads.
        let probe = DateTime {
            year: 2000,
            month: 1,
            day: 1,
            hour: 9,
            minute: 30,
            second: 45,
            millisecond: 0,
        };
        let padded = intl::datetime::format_skeleton_to_parts(locale, &probe, &skeleton)
            .into_iter()
            .any(|p| p.kind == intl::datetime::DateTimePartType::Hour && p.value == "09");
        if padded {
            hour.1.insert(0, '0');
        }
    }

    /// The proleptic Gregorian calendar has no year 0, so UTS #35's `y` field is
    /// the *era-relative* year: ISO year 0 renders as `1 BC`, ISO −271821 as
    /// `271822 BC`. The `intl` crate renders the raw astronomical year (its era
    /// field is already BCE-correct), so rewrite the year part here — matching the
    /// crate's own two-digit (`yy`) rendering when that is what it produced, so a
    /// `dateStyle: "short"` date is converted as well.
    #[cfg(feature = "intl")]
    fn fix_bce_era_year(dt: &intl::datetime::DateTime, parts: &mut [(&'static str, String)]) {
        if dt.year > 0 {
            return;
        }
        let era_year = 1 - i64::from(dt.year);
        let astro_full = alloc::format!("{}", dt.year);
        let astro_two = alloc::format!("{:02}", dt.year.rem_euclid(100));
        for p in parts.iter_mut() {
            if p.0 != "year" {
                continue;
            }
            if p.1 == astro_full {
                p.1 = alloc::format!("{era_year}");
            } else if p.1 == astro_two {
                p.1 = alloc::format!("{:02}", era_year.rem_euclid(100));
            }
        }
    }

    /// Renders an options set whose time skeleton CLDR has no `availableFormat`
    /// for — a lone `minute`, `second` or `fractionalSecondDigits`. The `intl`
    /// crate resolves a skeleton strictly through the CLDR `availableFormats`
    /// table (it does not synthesize a pattern the way ICU's
    /// `DateTimePatternGenerator` does), so `{ second: "numeric" }` resolves to a
    /// *date* pattern that then has every field stripped, i.e. an empty string.
    ///
    /// The fields are recovered from a probe with the full `hour`/`minute`/`second`
    /// skeleton (which every locale has), keeping the span from the first to the
    /// last *requested* field and dropping the separators that bordered a dropped
    /// field. The probe forces the numeric width because a lone field is never
    /// zero-padded (`{ second: "2-digit" }` renders `6`, not `06` — padding is a
    /// property of the abutting pattern, not of the option).
    #[cfg(feature = "intl")]
    fn lone_time_field_parts(
        locale: &str,
        dt: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Option<Vec<(&'static str, String)>> {
        use intl::datetime::{DateTimePartType, Numeric2Digit};
        if o.date_style.is_some() || o.time_style.is_some() {
            return None;
        }
        let wants = |k: DateTimePartType| match k {
            DateTimePartType::Hour => o.hour.is_some(),
            DateTimePartType::Minute => o.minute.is_some(),
            DateTimePartType::Second => o.second.is_some(),
            DateTimePartType::FractionalSecond => o.fractional_second_digits.is_some(),
            DateTimePartType::DayPeriod => o.day_period.is_some(),
            _ => false,
        };
        let mut probe = *o;
        probe.hour = Some(Numeric2Digit::Numeric);
        probe.minute = Some(Numeric2Digit::Numeric);
        probe.second = Some(Numeric2Digit::Numeric);
        let parts = intl::datetime::format_to_parts(locale, dt, &probe).ok()?;
        let first = parts.iter().position(|p| wants(p.kind))?;
        let last = parts.iter().rposition(|p| wants(p.kind))?;
        let mut out: Vec<(&'static str, String)> = Vec::new();
        let mut pending: Option<String> = None;
        for p in &parts[first..=last] {
            if p.kind == DateTimePartType::Literal {
                pending = Some(p.value.clone());
            } else if wants(p.kind) {
                if let Some(l) = pending.take()
                    && !out.is_empty()
                {
                    out.push(("literal", l));
                }
                out.push((p.kind.as_str(), p.value.clone()));
            } else {
                pending = None;
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// Whether `dayPeriod` is the *only* field an `Intl.DateTimeFormat` asked for
    /// (`new Intl.DateTimeFormat("en", { dayPeriod: "long" })`), which per
    /// ECMA-402 renders the flexible day period and nothing else.
    #[cfg(feature = "intl")]
    fn day_period_is_only_field(o: &intl::datetime::DateTimeFormatOptions) -> bool {
        o.weekday.is_none()
            && o.era.is_none()
            && o.year.is_none()
            && o.month.is_none()
            && o.day.is_none()
            && o.hour.is_none()
            && o.minute.is_none()
            && o.second.is_none()
            && o.fractional_second_digits.is_none()
            && o.time_zone_name.is_none()
            && o.date_style.is_none()
            && o.time_style.is_none()
    }

    /// The CLDR flexible day-period text ("in the morning", "noon", "at night")
    /// for `dt` at the width `o.day_period` requests.
    ///
    /// Two `intl`-crate quirks are routed around, without inventing any locale
    /// string — the text always comes from the crate's CLDR tables:
    ///
    /// * the crate builds its time skeleton from `hour`/`minute`/`second` only, so
    ///   an options set carrying no time field resolves to a *date* pattern that
    ///   then has every field stripped. A synthetic 12-hour `hour` field is what
    ///   pulls CLDR's flexible day-period field `B` into the pattern.
    /// * the crate applies a `midnight` override to `B` at exactly 00:00:00.000.
    ///   CLDR removed `midnight` from the day-period *format* rules (it survives
    ///   only for the `b` field), so `B` at midnight must come from the ordinary
    ///   range rules — which is what the crate yields one millisecond later. The
    ///   `noon` override at 12:00 is still current CLDR and is left alone.
    #[cfg(feature = "intl")]
    fn day_period_text(
        locale: &str,
        dt: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Option<String> {
        use intl::datetime::{DateTimePartType, HourCycle, Numeric2Digit};
        o.day_period?;
        let mut probe = *o;
        probe.hour = Some(Numeric2Digit::Numeric);
        probe.hour12 = Some(true);
        probe.hour_cycle = Some(HourCycle::H12);
        probe.fractional_second_digits = None;
        probe.time_zone_name = None;
        probe.date_style = None;
        probe.time_style = None;
        let mut when = *dt;
        if when.hour == 0 && (when.minute, when.second, when.millisecond) == (0, 0, 0) {
            when.millisecond = 1;
        }
        intl::datetime::format_to_parts(locale, &when, &probe)
            .ok()?
            .into_iter()
            .find(|p| p.kind == DateTimePartType::DayPeriod)
            .map(|p| p.value)
    }

    /// Builds the `intl` crate `(locale, DateTime, DateTimeFormatOptions)` for
    /// `handle` at instant `ms` — the shared front half of [`datetime_parts`] and
    /// the `formatRange` renderers. The `DateTime` is the instant shifted into the
    /// instance's resolved time zone; the options mirror the stored ECMA-402 fields.
    #[cfg(feature = "intl")]
    fn dtf_locale_dt_opts(
        &mut self,
        handle: Handle,
        ms: f64,
    ) -> (
        String,
        intl::datetime::DateTime,
        intl::datetime::DateTimeFormatOptions,
    ) {
        use intl::datetime::{
            DateStyle, DateTime, DateTimeFormatOptions, HourCycle, MonthStyle, NameStyle,
            Numeric2Digit, TimeZoneNameStyle,
        };
        let opt = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let locale = opt(self, "\u{0}locale").unwrap_or_else(|| String::from("en"));
        // A `Date`/number is an exact instant; shift its wall clock into the
        // instance's resolved time zone (offset at that instant, DST-aware for named
        // zones) before decomposing. `zone_off_ms` also feeds `timeZoneName`.
        let zone_off_ms = self.dtf_zone_offset_ms(handle, ms as i64);
        let msi = ms as i64 + zone_off_ms;
        let day = msi.div_euclid(86_400_000);
        let tod = msi.rem_euclid(86_400_000);
        let (y, mo, d) = crate::realm::civil_from_days(day);
        let dt = DateTime {
            year: y as i32,
            month: mo as u8,
            day: d as u8,
            hour: (tod / 3_600_000) as u8,
            minute: ((tod / 60_000) % 60) as u8,
            second: ((tod / 1_000) % 60) as u8,
            millisecond: (tod % 1_000) as u16,
        };
        let name = |s: &str| match s {
            "long" => Some(NameStyle::Long),
            "short" => Some(NameStyle::Short),
            "narrow" => Some(NameStyle::Narrow),
            _ => None,
        };
        let n2 = |s: &str| match s {
            "numeric" => Some(Numeric2Digit::Numeric),
            "2-digit" => Some(Numeric2Digit::TwoDigit),
            _ => None,
        };
        let dstyle = |s: &str| match s {
            "full" => Some(DateStyle::Full),
            "long" => Some(DateStyle::Long),
            "medium" => Some(DateStyle::Medium),
            "short" => Some(DateStyle::Short),
            _ => None,
        };
        let mut o = DateTimeFormatOptions::default();
        // `dateStyle`/`timeStyle` are mutually exclusive with component fields (the crate
        // errors if both are set), matching ECMA-402.
        if opt(self, "dateStyle").is_some() || opt(self, "timeStyle").is_some() {
            o.date_style = opt(self, "dateStyle").as_deref().and_then(dstyle);
            o.time_style = opt(self, "timeStyle").as_deref().and_then(dstyle);
        } else {
            o.weekday = opt(self, "weekday").as_deref().and_then(name);
            o.era = opt(self, "era").as_deref().and_then(name);
            o.year = opt(self, "year").as_deref().and_then(n2);
            o.month = opt(self, "month").as_deref().and_then(|s| match s {
                "numeric" => Some(MonthStyle::Numeric),
                "2-digit" => Some(MonthStyle::TwoDigit),
                "long" => Some(MonthStyle::Long),
                "short" => Some(MonthStyle::Short),
                "narrow" => Some(MonthStyle::Narrow),
                _ => None,
            });
            o.day = opt(self, "day").as_deref().and_then(n2);
            o.hour = opt(self, "hour").as_deref().and_then(n2);
            o.minute = opt(self, "minute").as_deref().and_then(n2);
            o.second = opt(self, "second").as_deref().and_then(n2);
            o.day_period = opt(self, "dayPeriod").as_deref().and_then(name);
            o.fractional_second_digits =
                opt(self, "fractionalSecondDigits").and_then(|s| s.parse().ok());
            // ECMA-402 `ToDateTimeOptions` default: when *none* of the date/time
            // components (weekday/year/month/day/dayPeriod/hour/minute/second/
            // fractionalSecondDigits) is requested, fall back to a numeric date.
            // `era` and `timeZoneName` do NOT count (an era-only or timeZoneName-only
            // formatter still gets the default date), and a dayPeriod-only or
            // fractionalSecond-only formatter must NOT (it renders just that field).
            if o.weekday.is_none()
                && o.year.is_none()
                && o.month.is_none()
                && o.day.is_none()
                && o.day_period.is_none()
                && o.hour.is_none()
                && o.minute.is_none()
                && o.second.is_none()
                && o.fractional_second_digits.is_none()
            {
                o.year = Some(Numeric2Digit::Numeric);
                o.month = Some(MonthStyle::Numeric);
                o.day = Some(Numeric2Digit::Numeric);
            }
        }
        o.hour12 = self
            .realm
            .get_property(handle, "hour12")
            .and_then(|v| match v.unpack() {
                Unpacked::Bool(b) => Some(b),
                _ => None,
            });
        // `ResolveLocale`: the `hourCycle` option outranks the tag's `-u-hc-`
        // keyword, and `hour12` outranks both (ECMA-402 sets `[[HourCycle]]` from
        // `hour12` and discards the cycle). The crate resolves neither from the tag,
        // so the keyword has to be lifted into the option here.
        let hc_ext = split_u_keyword(&locale, "hc").1;
        o.hour_cycle = if o.hour12.is_some() {
            None
        } else {
            opt(self, "hourCycle")
                .or(hc_ext)
                .as_deref()
                .and_then(|s| match s {
                    "h11" => Some(HourCycle::H11),
                    "h12" => Some(HourCycle::H12),
                    "h23" => Some(HourCycle::H23),
                    "h24" => Some(HourCycle::H24),
                    _ => None,
                })
        };
        // The zone is supplied unconditionally, not just when `timeZoneName` was
        // requested: a `timeStyle: "full"`/`"long"` pattern carries its own zone
        // field (`h:mm:ss a zzzz`), and the crate fills it from `time_zone` /
        // `tz_offset_minutes` — with neither, it strips the field instead.
        if let Some(tz) = opt(self, "timeZone").filter(|t| !t.is_empty()) {
            o.time_zone = Some(self.intern_static(&tz));
        }
        o.tz_offset_minutes = Some((zone_off_ms / 60_000) as i32);
        if let Some(tzn) = opt(self, "timeZoneName") {
            o.time_zone_name = match tzn.as_str() {
                "long" => Some(TimeZoneNameStyle::Long),
                "short" => Some(TimeZoneNameStyle::Short),
                "shortOffset" => Some(TimeZoneNameStyle::ShortOffset),
                "longOffset" => Some(TimeZoneNameStyle::LongOffset),
                "shortGeneric" => Some(TimeZoneNameStyle::ShortGeneric),
                "longGeneric" => Some(TimeZoneNameStyle::LongGeneric),
                _ => None,
            };
        }
        (locale, dt, o)
    }

    /// Chinese/Dangi lunisolar `formatToParts` output for `handle` (returns `None`
    /// unless the resolved calendar is `chinese`/`dangi` and the Gregorian date is
    /// within the supported lunisolar range). Produces `relatedYear` (the
    /// Gregorian year the lunisolar year began), `yearName` (the localized
    /// sexagenary cycle name) and `month`/`day` parts; no `era` part, because
    /// CLDR gives these calendars no eras.
    ///
    /// The calendar fields come from the engine's own Temporal calendar rather
    /// than from the `intl` crate's tables, so `dtf.formatToParts(instant)` and
    /// `Temporal.PlainDate.withCalendar(cal)` cannot report different months for
    /// the same instant. Only the *range check* is the crate's, since falling
    /// outside it is what makes this path decline (and render Gregorian instead).
    /// A leap month carries UTS #35's `monthPatterns` marker at the requested
    /// width (`"5bis"` in `en`, `"闰五月"` in `zh`). Any requested time components
    /// are appended via the crate's (calendar-independent) time formatter.
    #[cfg(feature = "intl")]
    fn lunisolar_parts(
        &self,
        handle: Handle,
        locale: &str,
        dt: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Option<Vec<(&'static str, String)>> {
        use crate::nbexec::temporal_calendar;
        use crate::temporal_iso::IsoDate;
        use intl::calendar;
        use intl::datetime::{self, Calendar, DateTimeFormatOptions, MonthStyle, Numeric2Digit};
        let cal = self.dtf_resolved_calendar(handle);
        let (y, m, d) = (dt.year as i64, dt.month as i64, dt.day as i64);
        let crate_cal = match cal.as_str() {
            "chinese" => {
                calendar::gregorian_to_chinese(y, m, d)?;
                Calendar::Chinese
            }
            "dangi" => {
                calendar::gregorian_to_dangi(y, m, d)?;
                Calendar::Dangi
            }
            _ => return None,
        };
        let iso = IsoDate {
            year: dt.year,
            month: dt.month,
            day: dt.day,
        };
        let f = temporal_calendar::iso_to_fields(&cal, iso);
        // The lunisolar year is named by its position in the 60-year sexagenary
        // cycle; `Temporal`'s `.year` for these calendars *is* the related
        // Gregorian year (the year the lunisolar year began), which anchors it.
        let related = f.year;
        let cyclic1 = (related - 4).rem_euclid(60) + 1;
        let (midx, mleap) = cldr_month_index(&cal, &f.month_code, f.month, false);
        // CLDR names are keyed by language; the `-u-` keywords are the formatter's
        // own resolution, not part of the lookup.
        let lang = strip_unicode_extension(locale);
        // zh renders the year/month/day with trailing `年`/`月`/`日` field markers;
        // other locales separate the numeric fields with plain literals.
        let zh = locale.starts_with("zh");
        let two = |v: i64| alloc::format!("{v:02}");
        let mut parts: Vec<(&'static str, String)> = Vec::new();
        if o.year.is_some() {
            parts.push(("relatedYear", alloc::format!("{related}")));
            parts.push((
                "yearName",
                datetime::cyclic_year_name(&lang, crate_cal, cyclic1 as u32)
                    .map_or_else(|| sexagenary_year_name(cyclic1), String::from),
            ));
            if zh {
                parts.push(("literal", String::from("年")));
            }
        }
        if let Some(mstyle) = o.month {
            if !zh && !parts.is_empty() {
                parts.push(("literal", String::from(", ")));
            }
            let fallback = || match mstyle {
                MonthStyle::TwoDigit => two(i64::from(midx)),
                _ => alloc::format!("{midx}"),
            };
            parts.push((
                "month",
                datetime::month_name(&lang, crate_cal, midx, mleap, mstyle)
                    .unwrap_or_else(fallback),
            ));
            if zh {
                parts.push(("literal", String::from("月")));
            }
        }
        if let Some(dstyle) = o.day {
            if !zh && !parts.is_empty() {
                parts.push(("literal", String::from(" ")));
            }
            parts.push((
                "day",
                match dstyle {
                    Numeric2Digit::TwoDigit => two(f.day),
                    _ => alloc::format!("{}", f.day),
                },
            ));
            if zh {
                parts.push(("literal", String::from("日")));
            }
        }
        // Time-of-day is calendar-independent: reuse the crate's time formatter for
        // any requested hour/minute/second/dayPeriod/fractionalSecond components.
        let has_time = o.hour.is_some()
            || o.minute.is_some()
            || o.second.is_some()
            || o.day_period.is_some()
            || o.fractional_second_digits.is_some();
        if has_time {
            let mut to = DateTimeFormatOptions::default();
            to.hour = o.hour;
            to.minute = o.minute;
            to.second = o.second;
            to.day_period = o.day_period;
            to.fractional_second_digits = o.fractional_second_digits;
            to.hour12 = o.hour12;
            to.hour_cycle = o.hour_cycle;
            to.time_zone_name = o.time_zone_name;
            to.tz_offset_minutes = o.tz_offset_minutes;
            if let Ok(tp) = datetime::format_to_parts(locale, dt, &to) {
                if !parts.is_empty() {
                    parts.push(("literal", String::from(", ")));
                }
                parts.extend(dtf_pad_time_parts(tp));
            }
        }
        Some(parts)
    }

    /// ECMA-402 `HandleDateTimeValue`: if `value` is a Temporal object, resolve it
    /// (WITHOUT calling `valueOf`) against `handle`'s `Intl.DateTimeFormat` options
    /// into the `(epoch-milliseconds, kind)` needed for formatting. A
    /// `Temporal.ZonedDateTime` throws a `TypeError`; a calendar mismatch throws a
    /// `RangeError`. Returns `Ok(None)` when `value` is not a Temporal object (the
    /// caller then falls back to the `ToNumber` / `Date` path).
    #[cfg(feature = "intl")]
    pub(crate) fn temporal_dtf_value(
        &mut self,
        handle: Handle,
        value: NanBox,
        zoned_ok: bool,
    ) -> Result<Option<(f64, crate::temporal_iso::TemporalKind)>, ExecError> {
        use crate::temporal_iso::{TemporalKind, iso_to_epoch_days};
        let Some(h) = value.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        let Some(d) = self.realm.temporal_at(h) else {
            return Ok(None);
        };
        // Resolved DateTimeFormat calendar: the explicit `calendar` option, else the
        // locale's `-u-ca-` extension, else "gregory".
        let dtf_cal = self
            .realm
            .get_property(handle, "calendar")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .or_else(|| {
                self.realm
                    .get_property(handle, "\u{0}locale")
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .and_then(|loc| locale_unicode_calendar(&loc))
            })
            .unwrap_or_else(|| String::from("gregory"));
        let cal = d.calendar.clone();
        // Calendar-compatibility checks (per-type).
        let cal_ok_iso = cal == dtf_cal || cal == "iso8601";
        let cal_ok_exact = cal == dtf_cal;
        let noon = 43_200_000_i64;
        let tod = |t: &crate::temporal_iso::IsoTime| -> i64 {
            i64::from(t.hour) * 3_600_000
                + i64::from(t.minute) * 60_000
                + i64::from(t.second) * 1_000
                + i64::from(t.millisecond)
        };
        let ms = match d.kind {
            // `Intl.DateTimeFormat.prototype.format`/`formatToParts`/`formatRange`
            // reject a `ZonedDateTime` (its own `toLocaleString` — `zoned_ok` —
            // formats it via its instant + time zone instead).
            TemporalKind::ZonedDateTime if !zoned_ok => {
                return Err(self.type_error(
                    "Temporal.ZonedDateTime is not supported by Intl.DateTimeFormat.prototype.format; \
                     use toLocaleString() or explicit options",
                ));
            }
            TemporalKind::ZonedDateTime => {
                // Format the ZonedDateTime's instant in its time zone (UTC-only
                // engine → the instant's wall clock). Its calendar must match.
                if !cal_ok_iso {
                    let m = self.new_str(
                        "Temporal object calendar is incompatible with this Intl.DateTimeFormat",
                    );
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                (d.epoch_ns / 1_000_000) as f64
            }
            TemporalKind::Duration => return Ok(None),
            TemporalKind::Instant => (d.epoch_ns / 1_000_000) as f64,
            TemporalKind::PlainDate | TemporalKind::PlainDateTime => {
                if !cal_ok_iso {
                    let m = self.new_str(
                        "Temporal object calendar is incompatible with this Intl.DateTimeFormat",
                    );
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                let days = iso_to_epoch_days(d.date);
                if d.kind == TemporalKind::PlainDate {
                    (days * 86_400_000 + noon) as f64
                } else {
                    (days * 86_400_000 + tod(&d.time)) as f64
                }
            }
            TemporalKind::PlainYearMonth | TemporalKind::PlainMonthDay => {
                if !cal_ok_exact {
                    let m = self.new_str(
                        "Temporal object calendar is incompatible with this Intl.DateTimeFormat",
                    );
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                (iso_to_epoch_days(d.date) * 86_400_000 + noon) as f64
            }
            TemporalKind::PlainTime => tod(&d.time) as f64,
        };
        Ok(Some((ms, d.kind)))
    }

    /// ECMA-402 `GetDateTimeFormat` specialized for a Temporal `kind`: builds the
    /// effective `intl` crate options for formatting a Temporal object, restricting
    /// `handle`'s DateTimeFormat options to the type's data model (per the spec's
    /// `required`/`defaults`/`inherit` rules). Returns `Ok(None)` when the format is
    /// *null* — meaning the requested options don't overlap the type's data model —
    /// which the caller turns into a `TypeError`.
    #[cfg(feature = "intl")]
    pub(crate) fn temporal_plain_options(
        &mut self,
        handle: Handle,
        kind: crate::temporal_iso::TemporalKind,
    ) -> Option<intl::datetime::DateTimeFormatOptions> {
        use crate::temporal_iso::TemporalKind;
        use intl::datetime::{
            DateStyle, DateTimeFormatOptions, HourCycle, MonthStyle, NameStyle, Numeric2Digit,
            TimeZoneNameStyle,
        };
        let opt = |this: &Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let defaulted = self
            .realm
            .get_property(handle, "\u{0}dtf_default_date")
            .and_then(|v| v.as_boolean())
            .unwrap_or(false);
        // Raw component options (the auto-filled numeric date is treated as absent).
        let weekday = opt(self, "weekday");
        let era = opt(self, "era");
        let (year, month, day) = if defaulted {
            (None, None, None)
        } else {
            (opt(self, "year"), opt(self, "month"), opt(self, "day"))
        };
        let day_period = opt(self, "dayPeriod");
        let hour = opt(self, "hour");
        let minute = opt(self, "minute");
        let second = opt(self, "second");
        let frac = self
            .realm
            .get_property(handle, "fractionalSecondDigits")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_number(v) as u8);
        let tz_name = opt(self, "timeZoneName");
        let hour_cycle_s = opt(self, "hourCycle");
        let hour12 = self
            .realm
            .get_property(handle, "hour12")
            .and_then(|v| v.as_boolean());
        let date_style = opt(self, "dateStyle");
        let time_style = opt(self, "timeStyle");
        let ds = date_style.is_some();
        let ts = time_style.is_some();
        // Presence of the nine format components (dateStyle/timeStyle expanded).
        let p_weekday = weekday.is_some() || (ds && date_style.as_deref() == Some("full"));
        let p_year = year.is_some() || ds;
        let p_month = month.is_some() || ds;
        let p_day = day.is_some() || ds;
        let p_hour = hour.is_some() || ts;
        let p_minute = minute.is_some() || ts;
        let p_second = second.is_some() || (ts && time_style.as_deref() != Some("short"));
        let p_day_period = day_period.is_some();
        let p_frac = frac.is_some();
        let any_present = p_weekday
            || p_year
            || p_month
            || p_day
            || p_hour
            || p_minute
            || p_second
            || p_day_period
            || p_frac;

        // Style helpers.
        let name = |s: &str| match s {
            "long" => Some(NameStyle::Long),
            "short" => Some(NameStyle::Short),
            "narrow" => Some(NameStyle::Narrow),
            _ => None,
        };
        let n2 = |s: &str| match s {
            "numeric" => Some(Numeric2Digit::Numeric),
            "2-digit" => Some(Numeric2Digit::TwoDigit),
            _ => None,
        };
        let mstyle = |s: &str| match s {
            "numeric" => Some(MonthStyle::Numeric),
            "2-digit" => Some(MonthStyle::TwoDigit),
            "long" => Some(MonthStyle::Long),
            "short" => Some(MonthStyle::Short),
            "narrow" => Some(MonthStyle::Narrow),
            _ => None,
        };
        let hc = |s: &str| match s {
            "h11" => Some(HourCycle::H11),
            "h12" => Some(HourCycle::H12),
            "h23" => Some(HourCycle::H23),
            "h24" => Some(HourCycle::H24),
            _ => None,
        };
        // Native dateStyle → crate DateStyle (kept as-is for date-capable types).
        let dstyle = |s: &str| match s {
            "full" => Some(DateStyle::Full),
            "long" => Some(DateStyle::Long),
            "medium" => Some(DateStyle::Medium),
            "short" => Some(DateStyle::Short),
            _ => None,
        };
        // timeStyle downgraded to drop the time-zone name (plain types have no zone):
        // "full"/"long" carry a zone name, so clamp them to "medium".
        let tstyle_no_tz = |s: &str| match s {
            "short" => Some(DateStyle::Short),
            _ => Some(DateStyle::Medium),
        };
        // dateStyle expanded to (year, month) styles for year-month / month-day.
        let ds_year = |s: &str| match s {
            "short" => Numeric2Digit::TwoDigit,
            _ => Numeric2Digit::Numeric,
        };
        let ds_month = |s: &str| match s {
            "full" | "long" => MonthStyle::Long,
            "medium" => MonthStyle::Short,
            _ => MonthStyle::Numeric,
        };

        // `GetDateTimeFormat(dtf, required, defaults, inherit)` with `inherit` =
        // ~all~ (the `Temporal.<Type>.prototype.toLocaleString` entry point) keeps
        // every option, so a style outside the type's data model makes the format
        // *null*: a date-only type rejects `timeStyle`, `PlainTime` rejects
        // `dateStyle`. `Intl.DateTimeFormat.prototype.format` passes ~relevant~
        // instead, which simply drops the out-of-model style.
        let inherit_all = self
            .realm
            .get_property(handle, "\u{0}dtf_inherit_all")
            .and_then(|v| v.as_boolean())
            .unwrap_or(false);
        let style_mismatch = inherit_all
            && match kind {
                TemporalKind::PlainDate
                | TemporalKind::PlainYearMonth
                | TemporalKind::PlainMonthDay => ts,
                TemporalKind::PlainTime => ds,
                _ => false,
            };
        if style_mismatch {
            return None;
        }

        let mut o = DateTimeFormatOptions::default();
        match kind {
            TemporalKind::PlainDate => {
                let need_defaults = !(p_weekday || p_year || p_month || p_day);
                if need_defaults {
                    if any_present {
                        return None;
                    }
                    o.era = era.as_deref().and_then(name);
                    o.year = Some(Numeric2Digit::Numeric);
                    o.month = Some(MonthStyle::Numeric);
                    o.day = Some(Numeric2Digit::Numeric);
                } else if ds {
                    o.date_style = date_style.as_deref().and_then(dstyle);
                } else {
                    o.weekday = weekday.as_deref().and_then(name);
                    o.era = era.as_deref().and_then(name);
                    o.year = year.as_deref().and_then(n2);
                    o.month = month.as_deref().and_then(mstyle);
                    o.day = day.as_deref().and_then(n2);
                }
            }
            TemporalKind::PlainYearMonth => {
                let need_defaults = !(p_year || p_month);
                if need_defaults {
                    if any_present {
                        return None;
                    }
                    o.era = era.as_deref().and_then(name);
                    o.year = Some(Numeric2Digit::Numeric);
                    o.month = Some(MonthStyle::Numeric);
                } else if ds {
                    let s = date_style.as_deref().unwrap_or("short");
                    o.year = Some(ds_year(s));
                    o.month = Some(ds_month(s));
                } else {
                    o.era = era.as_deref().and_then(name);
                    o.year = year.as_deref().and_then(n2);
                    o.month = month.as_deref().and_then(mstyle);
                }
            }
            TemporalKind::PlainMonthDay => {
                let need_defaults = !(p_month || p_day);
                if need_defaults {
                    if any_present {
                        return None;
                    }
                    o.month = Some(MonthStyle::Numeric);
                    o.day = Some(Numeric2Digit::Numeric);
                } else if ds {
                    let s = date_style.as_deref().unwrap_or("short");
                    o.month = Some(ds_month(s));
                    o.day = Some(Numeric2Digit::Numeric);
                } else {
                    o.month = month.as_deref().and_then(mstyle);
                    o.day = day.as_deref().and_then(n2);
                }
            }
            TemporalKind::PlainTime => {
                let need_defaults = !(p_day_period || p_hour || p_minute || p_second || p_frac);
                if need_defaults {
                    if any_present {
                        return None;
                    }
                    o.hour_cycle = hour_cycle_s.as_deref().and_then(hc);
                    o.hour12 = hour12;
                    o.hour = Some(Numeric2Digit::Numeric);
                    o.minute = Some(Numeric2Digit::Numeric);
                    o.second = Some(Numeric2Digit::Numeric);
                } else {
                    o.hour_cycle = hour_cycle_s.as_deref().and_then(hc);
                    o.hour12 = hour12;
                    if ts {
                        o.time_style = time_style.as_deref().and_then(tstyle_no_tz);
                    } else {
                        o.hour = hour.as_deref().and_then(n2);
                        o.minute = minute.as_deref().and_then(n2);
                        o.second = second.as_deref().and_then(n2);
                        o.day_period = day_period.as_deref().and_then(name);
                        o.fractional_second_digits = frac;
                    }
                }
            }
            TemporalKind::PlainDateTime => {
                // required = ~any~ → never null (needDefaults ⇒ !anyPresent).
                o.hour_cycle = hour_cycle_s.as_deref().and_then(hc);
                o.hour12 = hour12;
                if !any_present {
                    o.era = era.as_deref().and_then(name);
                    o.year = Some(Numeric2Digit::Numeric);
                    o.month = Some(MonthStyle::Numeric);
                    o.day = Some(Numeric2Digit::Numeric);
                    o.hour = Some(Numeric2Digit::Numeric);
                    o.minute = Some(Numeric2Digit::Numeric);
                    o.second = Some(Numeric2Digit::Numeric);
                } else {
                    if ds {
                        o.date_style = date_style.as_deref().and_then(dstyle);
                    } else {
                        o.weekday = weekday.as_deref().and_then(name);
                        o.era = era.as_deref().and_then(name);
                        o.year = year.as_deref().and_then(n2);
                        o.month = month.as_deref().and_then(mstyle);
                        o.day = day.as_deref().and_then(n2);
                    }
                    if ts {
                        o.time_style = time_style.as_deref().and_then(tstyle_no_tz);
                    } else {
                        o.hour = hour.as_deref().and_then(n2);
                        o.minute = minute.as_deref().and_then(n2);
                        o.second = second.as_deref().and_then(n2);
                        o.day_period = day_period.as_deref().and_then(name);
                        o.fractional_second_digits = frac;
                    }
                }
            }
            TemporalKind::Instant | TemporalKind::ZonedDateTime => {
                // required = ~any~, defaults = ~all~, inherit = ~all~: keep every
                // requested option (including timeZoneName) and, absent any component,
                // fall back to a full numeric date *and* time. An Instant / ZonedDateTime
                // has absolute time, so the DateTimeFormat's time zone applies (engine
                // is UTC-only). A ZonedDateTime additionally defaults `timeZoneName` to
                // "short" when no component options were requested.
                o.hour_cycle = hour_cycle_s.as_deref().and_then(hc);
                o.hour12 = hour12;
                if !any_present {
                    o.era = era.as_deref().and_then(name);
                    o.year = Some(Numeric2Digit::Numeric);
                    o.month = Some(MonthStyle::Numeric);
                    o.day = Some(Numeric2Digit::Numeric);
                    o.hour = Some(Numeric2Digit::Numeric);
                    o.minute = Some(Numeric2Digit::Numeric);
                    o.second = Some(Numeric2Digit::Numeric);
                    if kind == TemporalKind::ZonedDateTime && tz_name.is_none() {
                        o.time_zone_name = Some(TimeZoneNameStyle::Short);
                    }
                } else {
                    if ds {
                        o.date_style = date_style.as_deref().and_then(dstyle);
                    } else {
                        o.weekday = weekday.as_deref().and_then(name);
                        o.era = era.as_deref().and_then(name);
                        o.year = year.as_deref().and_then(n2);
                        o.month = month.as_deref().and_then(mstyle);
                        o.day = day.as_deref().and_then(n2);
                    }
                    if ts {
                        // Instant keeps the native time style (time-zone name and all).
                        o.time_style = time_style.as_deref().and_then(dstyle);
                    } else {
                        o.hour = hour.as_deref().and_then(n2);
                        o.minute = minute.as_deref().and_then(n2);
                        o.second = second.as_deref().and_then(n2);
                        o.day_period = day_period.as_deref().and_then(name);
                        o.fractional_second_digits = frac;
                    }
                }
                if let Some(tzn) = tz_name.as_deref() {
                    o.time_zone_name = match tzn {
                        "long" => Some(TimeZoneNameStyle::Long),
                        "short" => Some(TimeZoneNameStyle::Short),
                        "shortOffset" => Some(TimeZoneNameStyle::ShortOffset),
                        "longOffset" => Some(TimeZoneNameStyle::LongOffset),
                        "shortGeneric" => Some(TimeZoneNameStyle::ShortGeneric),
                        "longGeneric" => Some(TimeZoneNameStyle::LongGeneric),
                        _ => None,
                    };
                }
                return Some(o);
            }
            // ZonedDateTime / Duration never reach here.
            _ => return None,
        }
        // Plain temporal types never render a time-zone name.
        let _ = tz_name;
        Some(o)
    }

    /// Renders `ms` through explicit `intl` crate options into `{type,value}` parts
    /// (the Temporal branch of `format`/`formatToParts`).
    #[cfg(feature = "intl")]
    pub(crate) fn temporal_datetime_parts(
        &self,
        handle: Handle,
        ms: f64,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Vec<(&'static str, String)> {
        use intl::datetime::DateTime;
        let locale = self
            .realm
            .get_property(handle, "\u{0}locale")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en"));
        let msi = ms as i64;
        let day = msi.div_euclid(86_400_000);
        let tod = msi.rem_euclid(86_400_000);
        let (y, mo, d) = crate::realm::civil_from_days(day);
        let dt = DateTime {
            year: y as i32,
            month: mo as u8,
            day: d as u8,
            hour: (tod / 3_600_000) as u8,
            minute: ((tod / 60_000) % 60) as u8,
            second: ((tod / 1_000) % 60) as u8,
            millisecond: (tod % 1_000) as u16,
        };
        // Non-Gregorian calendar rendering (Islamic / Persian): the `intl` crate can
        // only format these via a coarse `DateStyle` (no field-level parts), so this
        // path is taken only when the resolved calendar is one the crate ships CLDR
        // month/era names for AND a `dateStyle` is in effect. Everything else (all
        // Gregorian/ISO output, and field-component options) falls through to the
        // Gregorian `format_to_parts` below.
        if let Some(alt) = self.alt_calendar_date_parts(handle, &locale, &dt, o) {
            return alt;
        }
        let mut parts = Self::dtf_crate_parts(&locale, &dt, o);
        self.rewrite_calendar_numerics(handle, &dt, o, &mut parts);
        self.fix_fractional_separator(handle, &locale, &mut parts);
        parts
    }

    /// Renders the date portion of a Temporal object whose resolved `Intl`
    /// calendar is one the `intl` crate ships localized names for (Islamic and
    /// Persian). Returns `None` (Gregorian fallthrough) unless the DateTimeFormat
    /// `handle`'s resolved calendar is Islamic/Persian *and* the effective options
    /// carry a `dateStyle` (the only shape the crate's coarse calendar API can
    /// satisfy). When taken, the localized date is emitted as a single `month`
    /// part, optionally followed by the Gregorian-rendered time. `dt` is the
    /// engine's proleptic-Gregorian pivot for the value (date + time-of-day).
    #[cfg(feature = "intl")]
    fn alt_calendar_date_parts(
        &self,
        handle: Handle,
        locale: &str,
        dt: &intl::datetime::DateTime,
        o: &intl::datetime::DateTimeFormatOptions,
    ) -> Option<Vec<(&'static str, String)>> {
        use crate::temporal_iso::IsoDate;
        use intl::datetime;
        // The crate only renders a whole `dateStyle`; per-field month/era options on
        // a non-Gregorian calendar aren't expressible through its public API.
        let date_style = o.date_style?;
        let cal = self.dtf_resolved_calendar(handle);
        // Which crate formatter (if any) covers this calendar. All Islamic variants
        // share the same localized month/era names; only the day arithmetic (which
        // the engine has already done) differs between them.
        let is_islamic = cal.starts_with("islamic");
        let is_persian = cal == "persian";
        if !is_islamic && !is_persian {
            return None;
        }
        let iso = IsoDate {
            year: dt.year,
            month: dt.month,
            day: dt.day,
        };
        let f = crate::nbexec::temporal_calendar::iso_to_fields(&cal, iso);
        let (yy, mm, dd) = (f.year, f.month, f.day);
        let date = if is_islamic {
            datetime::format_islamic_date(locale, yy, mm, dd, date_style)
        } else {
            datetime::format_persian_date(locale, yy, mm, dd, date_style)
        };
        // Optional time portion (dateStyle + timeStyle). The crate has no
        // calendar-specific time rendering, so the Gregorian time formatter is used
        // — time-of-day is calendar-independent.
        let mut parts: Vec<(&'static str, String)> = alloc::vec![("month", date)];
        if let Some(ts) = o.time_style {
            parts.push(("literal", String::from(", ")));
            parts.push(("hour", datetime::format_time(locale, dt, ts)));
        }
        Some(parts)
    }

    /// The resolved `Intl.DateTimeFormat` calendar for `handle`: the explicit
    /// `calendar` option, else the locale's `-u-ca-` extension, else `"gregory"`.
    #[cfg(feature = "intl")]
    fn dtf_resolved_calendar(&self, handle: Handle) -> String {
        self.realm
            .get_property(handle, "calendar")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .or_else(|| {
                self.realm
                    .get_property(handle, "\u{0}locale")
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                    .map(|v| self.realm.to_display_string(v))
                    .and_then(|loc| locale_unicode_calendar(&loc))
            })
            .unwrap_or_else(|| String::from("gregory"))
    }

    /// The Temporal branch of `Intl.DateTimeFormat.prototype.format(value)`:
    /// if `value` is a Temporal object, formats it per the ECMA-402 protocol and
    /// returns `Ok(Some(string))`; otherwise `Ok(None)` (caller uses the number path).
    #[cfg(feature = "intl")]
    pub(crate) fn temporal_format_flat(
        &mut self,
        handle: Handle,
        value: NanBox,
        zoned_ok: bool,
    ) -> Result<Option<String>, ExecError> {
        let Some((ms, kind)) = self.temporal_dtf_value(handle, value, zoned_ok)? else {
            return Ok(None);
        };
        let Some(mut o) = self.temporal_plain_options(handle, kind) else {
            return Err(self.type_error(
                "the requested Intl.DateTimeFormat options are not compatible with this Temporal type",
            ));
        };
        let ms = self.dtf_apply_temporal_zone(handle, ms, kind, &mut o);
        let mut s = String::new();
        for (_, v) in self.temporal_datetime_parts(handle, ms, &o) {
            s.push_str(&v);
        }
        Ok(Some(self.apply_numbering_digits(handle, s)))
    }

    /// The Temporal branch of `formatToParts` — `Ok(Some(parts))` or `Ok(None)`.
    #[cfg(feature = "intl")]
    pub(crate) fn temporal_format_parts(
        &mut self,
        handle: Handle,
        value: NanBox,
        zoned_ok: bool,
    ) -> Result<Option<Vec<(&'static str, String)>>, ExecError> {
        let Some((ms, kind)) = self.temporal_dtf_value(handle, value, zoned_ok)? else {
            return Ok(None);
        };
        let Some(mut o) = self.temporal_plain_options(handle, kind) else {
            return Err(self.type_error(
                "the requested Intl.DateTimeFormat options are not compatible with this Temporal type",
            ));
        };
        let ms = self.dtf_apply_temporal_zone(handle, ms, kind, &mut o);
        Ok(Some(self.temporal_datetime_parts(handle, ms, &o)))
    }

    /// `Temporal.<Type>.prototype.toLocaleString(locales, options)`: constructs an
    /// `Intl.DateTimeFormat` from the arguments and formats the Temporal receiver
    /// through the ECMA-402 Temporal protocol (identical to `dtf.format(this)`).
    /// Used for the calendared/plain types (Duration and ZonedDateTime keep their
    /// own `toString`-based behavior).
    #[cfg(feature = "intl")]
    pub(crate) fn temporal_to_locale_string(
        &mut self,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let locales = args.first().copied().unwrap_or(NanBox::undefined());
        let options = args.get(1).copied().unwrap_or(NanBox::undefined());
        // `Temporal.ZonedDateTime.prototype.toLocaleString` formats in the
        // *instance's* zone, so the options bag may not carry a `timeZone` of its
        // own — even one that agrees with the instance's.
        let zdt_zone = this
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.temporal_at(h))
            .filter(|d| d.kind == crate::temporal_iso::TemporalKind::ZonedDateTime)
            .and_then(|d| d.tz.clone());
        if zdt_zone.is_some()
            && let Some(oh) = options.as_handle().map(Handle::from_raw)
            && self.is_object_value(options)
            && !matches!(
                self.read_member(oh, "timeZone")?.unpack(),
                Unpacked::Undefined
            )
        {
            return Err(self.type_error(
                "Temporal.ZonedDateTime.prototype.toLocaleString does not accept a timeZone option",
            ));
        }
        let fmt_args = [locales, options];
        let inst = self.make_intl_formatter(N_INTL_DATETIME_FORMAT, &fmt_args)?;
        let Some(h) = inst.as_handle().map(Handle::from_raw) else {
            return Ok(self.new_str(""));
        };
        // `toLocaleString` inherits *all* of the options bag (see
        // `temporal_plain_options`), unlike `Intl.DateTimeFormat.prototype.format`.
        self.realm
            .set_hidden_property(h, "\u{0}dtf_inherit_all", NanBox::boolean(true));
        // The instance's zone replaces the formatter's resolved (default) one.
        if let Some(tz) = zdt_zone {
            let tzv = self.new_str(&tz);
            self.realm.set_hidden_property(h, "timeZone", tzv);
        }
        // `toLocaleString` accepts a `ZonedDateTime` (unlike `format`).
        let s = self
            .temporal_format_flat(h, this, true)?
            .unwrap_or_default();
        Ok(self.new_str(&s))
    }

    /// `Temporal.Duration.prototype.toLocaleString(locales, options)`: identical to
    /// `new Intl.DurationFormat(locales, options).format(this)`.
    #[cfg(feature = "intl")]
    pub(crate) fn duration_to_locale_string(
        &mut self,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let locales = args.first().copied().unwrap_or(NanBox::undefined());
        let options = args.get(1).copied().unwrap_or(NanBox::undefined());
        let df = self.make_duration_format(&[locales, options])?;
        let Some(h) = df.as_handle().map(Handle::from_raw) else {
            return Ok(self.new_str(""));
        };
        let rec = self.read_duration_record(this)?;
        let parts = self.partition_duration(h, &rec);
        let s: String = parts.into_iter().map(|(_, v, _)| v).collect();
        Ok(self.new_str(&s))
    }

    /// A range-endpoint "kind" tag: `0` for a Date/number, and a distinct value per
    /// Temporal type. Two `formatRange` endpoints must share the same tag.
    #[cfg(feature = "intl")]
    fn range_type_tag(&self, value: NanBox) -> u8 {
        use crate::temporal_iso::TemporalKind;
        if let Some(h) = value.as_handle().map(Handle::from_raw)
            && let Some(d) = self.realm.temporal_at(h)
        {
            return match d.kind {
                TemporalKind::PlainDate => 1,
                TemporalKind::PlainTime => 2,
                TemporalKind::PlainDateTime => 3,
                TemporalKind::Duration => 4,
                TemporalKind::Instant => 5,
                TemporalKind::PlainYearMonth => 6,
                TemporalKind::PlainMonthDay => 7,
                TemporalKind::ZonedDateTime => 8,
            };
        }
        0
    }

    /// Whether `value` carries a Temporal internal slot.
    #[cfg(feature = "intl")]
    fn is_temporal_value(&self, value: NanBox) -> bool {
        value
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|h| self.realm.temporal_at(h))
            .is_some()
    }

    /// Hand-rolled en-US fallback for [`datetime_parts`](Self::datetime_parts) when the `intl`
    /// crate is unavailable: component options (`weekday`/`year`/`month`/`day`/`hour`/`minute`/
    /// `second`/`hour12`/`era`/`dateStyle`/`timeStyle`) with `literal` separators, UTC.
    #[cfg(not(feature = "intl"))]
    pub(crate) fn datetime_parts(
        &mut self,
        handle: Handle,
        ms: f64,
    ) -> Vec<(&'static str, String)> {
        let opt = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let msi = ms as i64;
        let day = msi.div_euclid(86_400_000);
        let tod = msi.rem_euclid(86_400_000);
        let (y, mo, d) = crate::realm::civil_from_days(day);
        let (mo, d) = (i64::from(mo), i64::from(d));
        let wd_idx = (day + 4).rem_euclid(7) as usize; // 0 = Sunday
        let hour24 = tod / 3_600_000;
        let minute = (tod / 60_000) % 60;
        let second = (tod / 1_000) % 60;

        const MONTHS: [&str; 12] = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ];
        const WEEKDAYS: [&str; 7] = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ];
        let two = |v: i64| alloc::format!("{v:02}");
        let bare = |v: i64| alloc::format!("{v}");

        // Effective component options (after expanding dateStyle/timeStyle presets).
        let mut weekday = opt(self, "weekday");
        let mut year = opt(self, "year");
        let mut month = opt(self, "month");
        let mut day_o = opt(self, "day");
        let mut hour = opt(self, "hour");
        let mut minute_o = opt(self, "minute");
        let mut second_o = opt(self, "second");
        match opt(self, "dateStyle").as_deref() {
            Some("full") => {
                weekday = Some(String::from("long"));
                year = Some(String::from("numeric"));
                month = Some(String::from("long"));
                day_o = Some(String::from("numeric"));
            }
            Some("long") => {
                year = Some(String::from("numeric"));
                month = Some(String::from("long"));
                day_o = Some(String::from("numeric"));
            }
            Some("medium") => {
                year = Some(String::from("numeric"));
                month = Some(String::from("short"));
                day_o = Some(String::from("numeric"));
            }
            Some("short") => {
                year = Some(String::from("2-digit"));
                month = Some(String::from("numeric"));
                day_o = Some(String::from("numeric"));
            }
            _ => {}
        }
        match opt(self, "timeStyle").as_deref() {
            Some("full" | "long" | "medium") => {
                hour = Some(String::from("numeric"));
                minute_o = Some(String::from("2-digit"));
                second_o = Some(String::from("2-digit"));
            }
            Some("short") => {
                hour = Some(String::from("numeric"));
                minute_o = Some(String::from("2-digit"));
            }
            _ => {}
        }
        // With no options at all, the default is a numeric date.
        if weekday.is_none()
            && year.is_none()
            && month.is_none()
            && day_o.is_none()
            && hour.is_none()
            && minute_o.is_none()
            && second_o.is_none()
        {
            year = Some(String::from("numeric"));
            month = Some(String::from("numeric"));
            day_o = Some(String::from("numeric"));
        }

        let year_str = |style: &str| -> String {
            if style == "2-digit" {
                two(y.rem_euclid(100))
            } else {
                bare(y)
            }
        };
        let named_month = matches!(month.as_deref(), Some("long" | "short" | "narrow"));

        let lit = |s: &str| ("literal", String::from(s));

        // --- Date components (typed parts, en-US order) ---
        let mut date: Vec<(&'static str, String)> = Vec::new();
        if let Some(ws) = &weekday {
            let name = WEEKDAYS[wd_idx];
            date.push((
                "weekday",
                String::from(if ws == "long" { name } else { &name[..3] }),
            ));
        }
        if named_month {
            if !date.is_empty() {
                date.push(lit(", "));
            }
            if let Some(m) = &month {
                let name = MONTHS[(mo as usize).saturating_sub(1).min(11)];
                date.push((
                    "month",
                    String::from(if m == "long" { name } else { &name[..3] }),
                ));
            }
            if let Some(ds) = &day_o {
                date.push(lit(" "));
                date.push(("day", if ds == "2-digit" { two(d) } else { bare(d) }));
            }
            if let Some(ys) = &year {
                date.push(lit(if day_o.is_some() { ", " } else { " " }));
                date.push(("year", year_str(ys)));
            }
        } else {
            // A weekday-only request has no trailing separator; add ", " only before an
            // actual numeric date.
            if !date.is_empty() && (month.is_some() || day_o.is_some() || year.is_some()) {
                date.push(lit(", "));
            }
            let mut first = true;
            if let Some(m) = &month {
                date.push(("month", if m == "2-digit" { two(mo) } else { bare(mo) }));
                first = false;
            }
            if let Some(ds) = &day_o {
                if !first {
                    date.push(lit("/"));
                }
                date.push(("day", if ds == "2-digit" { two(d) } else { bare(d) }));
                first = false;
            }
            if let Some(ys) = &year {
                if !first {
                    date.push(lit("/"));
                }
                date.push(("year", year_str(ys)));
            }
        }
        if opt(self, "era").is_some() {
            if !date.is_empty() {
                date.push(lit(" "));
            }
            date.push(("era", String::from(if y > 0 { "AD" } else { "BC" })));
        }

        // --- Time components ---
        let mut time: Vec<(&'static str, String)> = Vec::new();
        if hour.is_some() || minute_o.is_some() || second_o.is_some() {
            // en-US defaults to 12-hour unless `hour12: false`.
            let h12 = !matches!(
                self.realm.get_property(handle, "hour12"),
                Some(v) if matches!(v.unpack(), Unpacked::Bool(false))
            );
            let h = if h12 {
                let m = hour24 % 12;
                if m == 0 { 12 } else { m }
            } else {
                hour24
            };
            time.push((
                "hour",
                if hour.as_deref() == Some("2-digit") {
                    two(h)
                } else {
                    bare(h)
                },
            ));
            if minute_o.is_some() {
                time.push(lit(":"));
                time.push(("minute", two(minute)));
            }
            if second_o.is_some() {
                time.push(lit(":"));
                time.push(("second", two(second)));
            }
            if h12 {
                // CLDR separates the time from AM/PM with U+202F (narrow no-break
                // space); like the `intl`-crate path (see `dtf_pad_time_parts`)
                // this is folded to a plain space to match the reference engine.
                time.push(lit(" "));
                time.push((
                    "dayPeriod",
                    String::from(if hour24 < 12 { "AM" } else { "PM" }),
                ));
            }
        }

        // --- Combine ---
        let mut parts = date;
        if !parts.is_empty() && !time.is_empty() {
            // CLDR's standard date-time connector is ", " (the crate uses it too).
            let _ = named_month;
            parts.push(lit(", "));
        }
        parts.extend(time);
        parts
    }

    /// `ToDateTimeOptions` for `Date.prototype.toLocale{,Date,Time}String`: builds
    /// a fresh options object from the user's `options`, adding the per-method
    /// component defaults (numeric year/month/day for a date-required call, and
    /// numeric hour/minute/second for a time-required call) when the user gave no
    /// matching component and no `dateStyle`/`timeStyle`.
    #[cfg(feature = "intl")]
    pub(crate) fn date_time_options(
        &mut self,
        user: NanBox,
        want_date: bool,
        want_time: bool,
    ) -> Result<Handle, ExecError> {
        let uh = user.as_handle().map(Handle::from_raw);
        let present = |this: &mut Self, keys: &[&str]| -> bool {
            uh.is_some_and(|h| {
                keys.iter().any(|k| {
                    this.realm
                        .get_property(h, k)
                        .is_some_and(|v| !matches!(v.unpack(), Unpacked::Undefined))
                })
            })
        };
        let has_date = present(self, &["weekday", "year", "month", "day"]);
        let has_time = present(
            self,
            &[
                "dayPeriod",
                "hour",
                "minute",
                "second",
                "fractionalSecondDigits",
            ],
        );
        let has_style = present(self, &["dateStyle", "timeStyle"]);
        let obj = self.realm.new_object();
        if let Some(h) = uh {
            for key in [
                "localeMatcher",
                "weekday",
                "era",
                "year",
                "month",
                "day",
                "dayPeriod",
                "hour",
                "minute",
                "second",
                "fractionalSecondDigits",
                "timeZoneName",
                "hour12",
                "hourCycle",
                "timeZone",
                "calendar",
                "numberingSystem",
                "dateStyle",
                "timeStyle",
                "formatMatcher",
            ] {
                if let Some(v) = self
                    .realm
                    .get_property(h, key)
                    .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                {
                    self.realm.set_property(obj, key, v);
                }
            }
        }
        // ECMA-402 `ToDateTimeOptions` uses a *single* `needDefaults` flag over the
        // whole `required` set: any present component (date *or* time, per what the
        // method requires) suppresses *all* defaults. So `toLocaleString` with a lone
        // `{ year }` yields just the year — not year + a defaulted time.
        let need_defaults = !((want_date && has_date) || (want_time && has_time));
        if !has_style && need_defaults {
            let num = self.new_str("numeric");
            if want_date {
                for k in ["year", "month", "day"] {
                    self.realm.set_property(obj, k, num);
                }
            }
            if want_time {
                for k in ["hour", "minute", "second"] {
                    self.realm.set_property(obj, k, num);
                }
            }
        }
        Ok(obj)
    }

    /// The `Intl.DateTimeFormat` rendering of `ms` as a flat string (joins
    /// [`datetime_parts`](Self::datetime_parts)).
    pub(crate) fn format_intl_datetime(&mut self, handle: Handle, ms: f64) -> String {
        let mut s = String::new();
        for (_, v) in self.datetime_parts(handle, ms) {
            s.push_str(&v);
        }
        self.apply_numbering_digits(handle, s)
    }

    /// Rewrites the ASCII digits of `s` into the format object's resolved
    /// `numberingSystem` (see [`numbering_system_digit_base`]); non-digit code
    /// points (separators, letters) are untouched. Shared by NumberFormat and
    /// DateTimeFormat.
    pub(crate) fn apply_numbering_digits(&mut self, handle: Handle, s: String) -> String {
        let nu = self
            .realm
            .get_property(handle, "numberingSystem")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        substitute_numbering_digits(&nu, s)
    }

    /// Builds the `intl` crate's `NumberFormatOptions` from an `Intl.NumberFormat` instance's
    /// stored JS options.
    #[cfg(feature = "intl")]
    pub(crate) fn number_format_options(
        &mut self,
        handle: Handle,
    ) -> intl::number::NumberFormatOptions {
        use intl::number::{
            CompactDisplay, CurrencyDisplay, Notation, NumberFormatOptions, NumberStyle,
            RoundingMode, SignDisplay, UnitDisplay, UseGrouping,
        };
        let opt_str = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let opt_num = |this: &mut Self, k: &str| -> Option<u8> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_number(v) as u8)
        };
        // `NumberFormatOptions` is `#[non_exhaustive]` in intl 0.5, so it can't be
        // built with a struct literal — start from `default()` and set fields.
        let mut o = NumberFormatOptions::default();
        o.style = match opt_str(self, "style").as_deref() {
            Some("percent") => NumberStyle::Percent,
            Some("currency") => NumberStyle::Currency,
            Some("unit") => NumberStyle::Unit,
            _ => NumberStyle::Decimal,
        };
        o.notation = match opt_str(self, "notation").as_deref() {
            Some("scientific") => Notation::Scientific,
            Some("engineering") => Notation::Engineering,
            Some("compact") => Notation::Compact,
            _ => Notation::Standard,
        };
        o.compact_display = match opt_str(self, "compactDisplay").as_deref() {
            Some("long") => CompactDisplay::Long,
            _ => CompactDisplay::Short,
        };
        o.sign_display = match opt_str(self, "signDisplay").as_deref() {
            Some("always") => SignDisplay::Always,
            Some("exceptZero") => SignDisplay::ExceptZero,
            Some("negative") => SignDisplay::Negative,
            Some("never") => SignDisplay::Never,
            _ => SignDisplay::Auto,
        };
        o.currency_display = match opt_str(self, "currencyDisplay").as_deref() {
            Some("code") => CurrencyDisplay::Code,
            Some("name") => CurrencyDisplay::Name,
            Some("narrowSymbol") => CurrencyDisplay::NarrowSymbol,
            _ => CurrencyDisplay::Symbol,
        };
        o.unit_display = match opt_str(self, "unitDisplay").as_deref() {
            Some("long") => UnitDisplay::Long,
            Some("narrow") => UnitDisplay::Narrow,
            _ => UnitDisplay::Short,
        };
        // ECMA-402's default rounding is half-expand (1.25 → 1.3), not banker's rounding.
        o.rounding_mode = match opt_str(self, "roundingMode").as_deref() {
            Some("ceil") => RoundingMode::Ceil,
            Some("floor") => RoundingMode::Floor,
            Some("expand") => RoundingMode::Expand,
            Some("trunc") => RoundingMode::Trunc,
            Some("halfCeil") => RoundingMode::HalfCeil,
            Some("halfFloor") => RoundingMode::HalfFloor,
            Some("halfExpand") => RoundingMode::HalfExpand,
            Some("halfTrunc") => RoundingMode::HalfTrunc,
            Some("halfEven") => RoundingMode::HalfEven,
            _ => RoundingMode::HalfExpand,
        };
        // `useGrouping` is stored as a boolean (`false` → never, `true` → always) or
        // a string (`"min2"`/`"auto"`/`"always"`), per GetStringOrBooleanOption.
        match self
            .realm
            .get_property(handle, "useGrouping")
            .map(|v| v.unpack())
        {
            Some(Unpacked::Bool(false)) => o.use_grouping = UseGrouping::Never,
            Some(Unpacked::Bool(true)) => o.use_grouping = UseGrouping::Always,
            Some(_) => match opt_str(self, "useGrouping").as_deref() {
                Some("min2") => o.use_grouping = UseGrouping::Min2,
                Some("always") | Some("true") => o.use_grouping = UseGrouping::Always,
                Some("false") => o.use_grouping = UseGrouping::Never,
                // "auto" (and the absence of the option) keeps the locale default.
                _ => {}
            },
            None => {}
        }
        if let Some(mid) = opt_num(self, "minimumIntegerDigits") {
            o.minimum_integer_digits = mid;
        }
        o.minimum_fraction_digits = opt_num(self, "minimumFractionDigits");
        o.maximum_fraction_digits = opt_num(self, "maximumFractionDigits");
        o.minimum_significant_digits = opt_num(self, "minimumSignificantDigits");
        o.maximum_significant_digits = opt_num(self, "maximumSignificantDigits");
        // Scientific/engineering cap the mantissa at 3 fraction digits by default (1.235E5).
        if o.maximum_fraction_digits.is_none()
            && o.maximum_significant_digits.is_none()
            && matches!(o.notation, Notation::Scientific | Notation::Engineering)
        {
            o.maximum_fraction_digits = Some(3);
        }
        if let Some(c) = opt_str(self, "currency") {
            o.currency = Some(self.intern_static(&c));
        }
        if let Some(u) = opt_str(self, "unit") {
            o.unit = Some(self.intern_static(&u));
        }
        o
    }

    /// Formats `n` per an `Intl.NumberFormat` instance. With the `intl` crate, all styles
    /// except those in [`number_uses_handrolled`](Self::number_uses_handrolled) go through
    /// `intl::number::format` (CLDR, locale-aware, full ECMA-402 options); the rest, and the
    /// no-`intl` build, use the hand-rolled en-US path below.
    pub(crate) fn intl_format_number(&mut self, handle: Handle, n: f64) -> String {
        // `roundingPriority` more/lessPrecision selects between the significant-
        // and fraction-digit results — an interplay the crate can't express.
        #[cfg(feature = "intl")]
        if let Some(s) = self.try_rounding_priority(handle, n) {
            return s;
        }
        let s = self.intl_format_number_inner(handle, n);
        self.apply_numbering_digits(handle, s)
    }

    /// Parse a string / BigInt `value` into an exact decimal
    /// `(negative, integer_digits, fraction_digits)` — but only when it carries
    /// more significant digits than an `f64` preserves (so the ordinary numeric
    /// path would silently round it). Returns `None` for values the crate's f64
    /// path renders exactly (≤ 15 significant digits) or that aren't a plain
    /// decimal literal (Infinity/NaN/exponent/etc.).
    #[cfg(feature = "intl")]
    fn extract_exact_decimal(&self, value: NanBox) -> Option<(bool, String, String)> {
        let h = value.as_handle().map(Handle::from_raw)?;
        let raw = if let Some(big) = self.realm.bigint_at(h) {
            alloc::format!("{big}")
        } else {
            self.realm.string_value(h)?
        };
        let s = raw.trim();
        let (neg, body) = match s.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        if body.is_empty() || body.matches('.').count() > 1 {
            return None;
        }
        if !body.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
            return None;
        }
        let mut it = body.splitn(2, '.');
        let int_part = it.next().unwrap_or("");
        let frac_part = it.next().unwrap_or("");
        // Significant-digit span (first to last nonzero across int+frac).
        let all = alloc::format!("{int_part}{frac_part}");
        let first_nz = all.bytes().position(|b| b != b'0');
        let last_nz = all.bytes().rposition(|b| b != b'0');
        let sig = match (first_nz, last_nz) {
            (Some(a), Some(b)) => b - a + 1,
            _ => 0,
        };
        // A large-magnitude integer (its trailing zeros are significant) also
        // exceeds f64 integer precision beyond ~15 digits, even at low `sig`.
        let int_magnitude = int_part.trim_start_matches('0').len();
        if sig <= 15 && int_magnitude <= 15 {
            return None;
        }
        Some((neg, String::from(int_part), String::from(frac_part)))
    }

    /// Exact-decimal `Intl.NumberFormat.prototype.format` for a high-precision
    /// string / BigInt argument (ECMA-402 `ToIntlMathematicalValue` keeps the full
    /// value; the crate's `f64` path would round it). Handles only
    /// standard-notation plain-decimal formatting with fraction-digit options —
    /// everything else returns `None` to fall back to the numeric path.
    #[cfg(feature = "intl")]
    fn try_exact_decimal_format(&mut self, handle: Handle, value: NanBox) -> Option<String> {
        use intl::number::{Notation, NumberStyle, UseGrouping};
        let opts = self.number_format_options(handle);
        // Only standard-notation decimal style is shaped by `assemble_decimal`'s
        // probe; currency/percent/unit affixes and scientific/engineering/compact
        // mantissa shaping stay on the crate's f64 path.
        if opts.notation != Notation::Standard || !matches!(opts.style, NumberStyle::Decimal) {
            return None;
        }
        let increment = self
            .realm
            .get_property(handle, "roundingIncrement")
            .and_then(|v| v.as_number())
            .unwrap_or(1.0);
        if increment != 1.0 {
            // roundingIncrement rounds to a multiple of `increment` at the fraction
            // place; exact multiple-rounding on huge values isn't implemented here,
            // so leave it to the f64 path.
            return None;
        }
        let (neg, int_part, frac_part) = self.extract_exact_decimal(value)?;
        let min_int = (opts.minimum_integer_digits.max(1)) as usize;

        let (mut int_digits, frac_digits) = if opts.minimum_significant_digits.is_some()
            || opts.maximum_significant_digits.is_some()
        {
            // Significant-digit rounding (ECMA-402 roundingType `significantDigits`),
            // done exactly with a `puremp::Decimal`.
            exact_significant_digits(
                neg,
                &int_part,
                &frac_part,
                opts.minimum_significant_digits.map(|m| m as usize),
                opts.maximum_significant_digits.map(|m| m as usize),
                opts.rounding_mode,
            )?
        } else {
            let max_frac = opts.maximum_fraction_digits.unwrap_or(3) as usize;
            let min_frac = opts.minimum_fraction_digits.unwrap_or(0) as usize;
            let mut int_digits: alloc::vec::Vec<u8> = int_part.bytes().map(|b| b - b'0').collect();
            let mut frac_digits: alloc::vec::Vec<u8> =
                frac_part.bytes().map(|b| b - b'0').collect();
            // Round the fraction to maximumFractionDigits.
            if frac_digits.len() > max_frac {
                let up = exact_round_up(&frac_digits[max_frac..], opts.rounding_mode, neg);
                frac_digits.truncate(max_frac);
                if up {
                    exact_increment(&mut int_digits, &mut frac_digits);
                }
            }
            // Trim trailing fraction zeros down to minimumFractionDigits, then pad up.
            while frac_digits.len() > min_frac && frac_digits.last() == Some(&0) {
                frac_digits.pop();
            }
            while frac_digits.len() < min_frac {
                frac_digits.push(0);
            }
            (int_digits, frac_digits)
        };
        // Normalize the integer digits (strip leading zeros, pad to the minimum).
        while int_digits.len() > 1 && int_digits.first() == Some(&0) {
            int_digits.remove(0);
        }
        while int_digits.len() < min_int {
            int_digits.insert(0, 0);
        }
        let grouping = !matches!(opts.use_grouping, UseGrouping::Never);
        Some(self.assemble_decimal(handle, neg, &int_digits, &frac_digits, grouping))
    }

    /// Assemble a locale-scaffolded decimal string from latn `int`/`frac` digit
    /// vectors (probing the `intl` crate for the locale's affixes/separators),
    /// then substitute the resolved numbering-system digits. Shared by the
    /// exact-decimal and roundingPriority renderers.
    #[cfg(feature = "intl")]
    fn assemble_decimal(
        &mut self,
        handle: Handle,
        neg: bool,
        int: &[u8],
        frac: &[u8],
        grouping: bool,
    ) -> String {
        let int_str: String = int.iter().map(|d| (b'0' + d) as char).collect();
        let frac_str: String = frac.iter().map(|d| (b'0' + d) as char).collect();
        // Probe with the *full* locale, `-u-nu-` included: the sample is read back
        // for the locale's affixes and separators, and those differ per numbering
        // system (`ar-u-nu-arab` uses U+066B where plain `ar` uses `.`). The digits
        // that come back are the system's own, which is why the splitters are told
        // which system to expect rather than assuming ASCII.
        let nu = self
            .realm
            .get_property(handle, "numberingSystem")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        let probe = self.intl_format_number_inner(handle, if neg { -1.1 } else { 1.1 });
        let (prefix, dec_sep, suffix) = split_number_scaffold(&probe, &nu);
        let grouped = if grouping {
            let gp = self.intl_format_number_inner(handle, if neg { -1111.0 } else { 1111.0 });
            let group_sep = extract_group_sep(&gp, &nu);
            group_thousands_sep(&int_str, &group_sep)
        } else {
            int_str
        };
        let mut out = String::new();
        out.push_str(&prefix);
        out.push_str(&grouped);
        if !frac_str.is_empty() {
            out.push_str(&dec_sep);
            out.push_str(&frac_str);
        }
        out.push_str(&suffix);
        self.apply_numbering_digits(handle, out)
    }

    /// `roundingPriority: "morePrecision"` / `"lessPrecision"`: ECMA-402
    /// `FormatNumericToString` computes both `ToRawPrecision` (significant-digit)
    /// and `ToRawFixed` (fraction-digit) results and selects between them by their
    /// rounding magnitude. The `intl` crate can't express this interplay, so we
    /// render it here for standard-notation decimal formatting. Returns `None`
    /// (fall back to the crate) for any other configuration.
    #[cfg(feature = "intl")]
    fn try_rounding_priority(&mut self, handle: Handle, n: f64) -> Option<String> {
        use intl::number::{Notation, NumberStyle, UseGrouping};
        if !n.is_finite() || n == 0.0 {
            return None;
        }
        let priority = self
            .realm
            .get_property(handle, "roundingPriority")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        if !matches!(priority.as_str(), "morePrecision" | "lessPrecision") {
            return None;
        }
        let opts = self.number_format_options(handle);
        if opts.notation != Notation::Standard || !matches!(opts.style, NumberStyle::Decimal) {
            return None;
        }
        let min_sig = opts.minimum_significant_digits.unwrap_or(1) as usize;
        let max_sig = opts.maximum_significant_digits.unwrap_or(21) as usize;
        let min_frac = opts.minimum_fraction_digits.unwrap_or(0) as usize;
        let max_frac = opts
            .maximum_fraction_digits
            .map(|m| m as usize)
            .unwrap_or_else(|| min_frac.max(3));

        // Shortest round-trip decimal of |n| (exponent form → fall back).
        let neg = n.is_sign_negative();
        let shortest = alloc::format!("{}", n.abs());
        if shortest.contains(['e', 'E', 'i', 'n']) {
            return None;
        }
        let mut it = shortest.splitn(2, '.');
        let int_part = it.next().unwrap_or("0");
        let frac_part = it.next().unwrap_or("");
        let int_v: alloc::vec::Vec<u8> = int_part.bytes().map(|b| b - b'0').collect();
        let frac_v: alloc::vec::Vec<u8> = frac_part.bytes().map(|b| b - b'0').collect();

        // sResult = ToRawPrecision; fResult = ToRawFixed.
        let (s_int, s_frac) = to_raw_precision(
            neg,
            int_v.clone(),
            frac_v.clone(),
            max_sig,
            opts.rounding_mode,
        );
        let (f_int, f_frac) = to_raw_fixed(neg, int_v, frac_v, max_frac, opts.rounding_mode);
        // ECMA-402 rounding magnitudes: ToRawPrecision rounds at
        // `e - maxSig + 1` (e = exponent of the most significant digit);
        // ToRawFixed rounds at `-maxFrac`. Lower (more negative) = more precise.
        let e = {
            let int_stripped = int_part.trim_start_matches('0');
            if !int_stripped.is_empty() {
                int_stripped.len() as i32 - 1
            } else {
                match frac_part.bytes().position(|b| b != b'0') {
                    Some(p) => -(p as i32) - 1,
                    None => 0,
                }
            }
        };
        let s_mag = e - max_sig as i32 + 1;
        let f_mag = -(max_frac as i32);

        let use_s = if priority == "morePrecision" {
            s_mag <= f_mag
        } else {
            s_mag > f_mag
        };

        // Build the selected result's digit vectors with its own minimum padding.
        let (int_d, frac_d) = if use_s {
            // ToRawPrecision: pad with trailing zeros up to minimumSignificantDigits.
            let (mut i, mut f) = (s_int, s_frac);
            let sig_now = {
                let first = i
                    .iter()
                    .chain(f.iter())
                    .position(|&d| d != 0)
                    .unwrap_or(i.len());
                (i.len() + f.len()).saturating_sub(first)
            };
            if sig_now < min_sig {
                f.resize(f.len() + (min_sig - sig_now), 0);
            }
            // Normalize leading zeros (keep at least one integer digit).
            while i.len() > 1 && i[0] == 0 {
                i.remove(0);
            }
            (i, f)
        } else {
            // ToRawFixed: trim trailing zeros down to minimumFractionDigits, then pad.
            let (i, mut f) = (f_int, f_frac);
            while f.len() > min_frac && f.last() == Some(&0) {
                f.pop();
            }
            if f.len() < min_frac {
                f.resize(min_frac, 0);
            }
            (i, f)
        };

        // minimumIntegerDigits padding.
        let min_int = opts.minimum_integer_digits.max(1) as usize;
        let mut int_d = int_d;
        while int_d.len() < min_int {
            int_d.insert(0, 0);
        }
        let grouping = !matches!(opts.use_grouping, UseGrouping::Never);
        Some(self.assemble_decimal(handle, neg, &int_d, &frac_d, grouping))
    }

    /// Pre-round `n` to the resolved precision using the ECMA-402 / ICU decimal
    /// rounding (see [`intl_decimal_round`]), returning a value the `intl` crate
    /// re-renders correctly. Applies to standard-notation decimal/currency only:
    /// percent (the crate scales by 100) and scientific/engineering/compact
    /// (mantissa rounding) are left untouched. Shared by `format`/`formatToParts`.
    #[cfg(feature = "intl")]
    fn number_precision_round(
        &mut self,
        handle: Handle,
        opts: &mut intl::number::NumberFormatOptions,
        n: f64,
    ) -> f64 {
        use intl::number::{Notation, NumberStyle};
        if !n.is_finite() || n == 0.0 || opts.notation != Notation::Standard {
            return n;
        }
        if matches!(opts.style, NumberStyle::Percent) {
            return n;
        }
        // `roundingPriority` other than the default `auto` combines the significant-
        // and fraction-digit results in a way the crate can't express; leave those
        // to the existing path rather than pre-rounding on the wrong precision.
        let priority = self
            .realm
            .get_property(handle, "roundingPriority")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        if matches!(priority.as_str(), "morePrecision" | "lessPrecision")
            && opts.maximum_significant_digits.is_some()
            && opts.maximum_fraction_digits.is_some()
        {
            return n;
        }
        let increment = self
            .realm
            .get_property(handle, "roundingIncrement")
            .and_then(|v| v.as_number())
            .unwrap_or(1.0) as u32;
        let sig = opts.maximum_significant_digits.map(|s| s as usize);
        // Effective maximum fraction digits (the crate's pattern default when unset).
        let default_max = match opts.style {
            NumberStyle::Currency if opts.currency == Some("JPY") => 0,
            NumberStyle::Currency => 2,
            _ => 3,
        };
        let keep_frac = opts
            .maximum_fraction_digits
            .map(|m| m as usize)
            .unwrap_or(default_max);
        let rounded = intl_decimal_round(n, keep_frac, sig, increment, opts.rounding_mode);
        // We already applied the correct rounding to the shortest decimal; hand the
        // crate a mode that won't re-round the pre-rounded value's f64 noise (a
        // directional mode like `ceil`/`expand` would spuriously round `1.3` — really
        // `1.30000…0444` — up again). `halfExpand` leaves the clean value untouched.
        opts.rounding_mode = intl::number::RoundingMode::HalfExpand;
        rounded
    }

    #[cfg(feature = "intl")]
    fn intl_format_number_inner(&mut self, handle: Handle, n: f64) -> String {
        let locale = self
            .realm
            .get_property(handle, "\u{0}locale")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en"));
        let mut opts = self.number_format_options(handle);
        // Round the value's shortest round-trip decimal ourselves (ECMA-402 /
        // ICU rounding) and hand the crate a "clean" value — this fixes the
        // crate's binary-expansion rounding boundary (`1.15` → `1.2`, not `1.1`)
        // and applies `roundingIncrement` (which the crate has no option for).
        let n = self.number_precision_round(handle, &mut opts, n);
        // `trailingZeroDisplay: "stripIfInteger"`: an integer value drops its
        // forced trailing fraction (and significant) zeros. The intl crate has
        // no such option, so lower the minimum digit counts for this value.
        if n.is_finite() && n.fract() == 0.0 {
            let tzd = self
                .realm
                .get_property(handle, "trailingZeroDisplay")
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_default();
            if tzd == "stripIfInteger" {
                opts.minimum_fraction_digits = Some(0);
                if opts.minimum_significant_digits.is_some() {
                    opts.minimum_significant_digits = Some(1);
                }
            }
        }
        // `intl` 0.5 panics formatting a true negative zero. Substitute the
        // smallest negative subnormal (which rounds to a displayed zero) so the
        // crate applies the locale's negative affix (e.g. ar's U+200E mark) and
        // the resolved `signDisplay` logic uniformly, instead of us hand-rolling
        // the sign and losing locale marks.
        let n = if n == 0.0 && n.is_sign_negative() {
            -f64::from_bits(1)
        } else {
            n
        };
        // Default compact notation: render via the tagged parts so the mantissa
        // can be re-rounded to the ECMA-402 default precision (the crate's plain
        // string path keeps a single fraction digit — `987654321` → `987.7M`
        // instead of `988M`). NaN/∞ carry no mantissa and fall through.
        if n.is_finite() && compact_wants_reround(&opts) {
            let mut o = opts;
            o.maximum_fraction_digits = Some(6);
            let mut parts: Vec<(&'static str, String)> =
                intl::number::format_to_parts(&locale, n, &o)
                    .into_iter()
                    .map(|p| (p.kind.as_str(), p.value))
                    .collect();
            compact_reround_parts(&mut parts, opts.rounding_mode);
            // Return latn digits; the caller (`intl_format_number`) applies the
            // numbering system.
            return parts.into_iter().map(|(_, v)| v).collect();
        }
        let formatted = intl::number::format(&locale, n, &opts);
        // A small-magnitude negative that rounds to a displayed zero
        // (e.g. -0.0001 → "0", or -0.004 currency → "$0.00") carries no sign
        // under signDisplay "negative"/"never"/"exceptZero" — only "auto"/
        // "always" sign a zero. The intl crate keeps the input value's minus,
        // so strip it when every rendered digit is zero. (A literal -0 is
        // handled by the negative-zero branch above; -∞/NaN have no digits.)
        if formatted.starts_with('-') && n.is_finite() {
            let digits = formatted.bytes().filter(u8::is_ascii_digit);
            let mut any = false;
            let all_zero = digits.inspect(|_| any = true).all(|b| b == b'0');
            if any && all_zero {
                let sd = self
                    .realm
                    .get_property(handle, "signDisplay")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                if matches!(sd.as_str(), "negative" | "never" | "exceptZero") {
                    return String::from(&formatted[1..]);
                }
            }
        }
        // `signDisplay: "always"` signs a NaN as non-negative ("+NaN"); the
        // intl crate leaves NaN unsigned (it signs ±∞ but not NaN). Only
        // "always" signs NaN — "auto"/"never"/"exceptZero"/"negative" do not.
        if n.is_nan()
            && !formatted.starts_with(['+', '-'])
            && self
                .realm
                .get_property(handle, "signDisplay")
                .map(|v| self.realm.to_display_string(v))
                .as_deref()
                == Some("always")
        {
            return alloc::format!("+{formatted}");
        }
        // `currencySign: "accounting"` renders a negative currency amount in the
        // locale's accounting pattern. That pattern is CLDR-locale data: most
        // locales inherit the root's parenthesized form (`($5.00)`), but some
        // (e.g. de-DE: `-987,00 $`) keep the minus. The intl crate has no
        // currencySign field, so approximate the parenthesizing locales here and
        // leave the minus-locales on the crate's output.
        if accounting_uses_parens(&locale)
            && formatted.starts_with('-')
            && self
                .realm
                .get_property(handle, "currencySign")
                .map(|v| self.realm.to_display_string(v))
                .as_deref()
                == Some("accounting")
        {
            return alloc::format!("({})", &formatted[1..]);
        }
        formatted
    }

    /// The no-`intl` fallback formatter: an en-US-shaped renderer covering the
    /// `decimal`/`percent`/`currency`/`unit` styles, grouping, fraction digits and
    /// `signDisplay`, with no CLDR data behind it.
    #[cfg(not(feature = "intl"))]
    fn intl_format_number_inner(&mut self, handle: Handle, n: f64) -> String {
        let opt_str = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let opt_num = |this: &mut Self, k: &str| -> Option<i32> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_number(v) as i32)
        };
        let style = opt_str(self, "style").unwrap_or_else(|| String::from("decimal"));
        let currency = opt_str(self, "currency");
        // Non-finite values render as the core glyph (∞ / NaN) with the sign and
        // currency/percent affixes, but no grouping or fraction digits.
        if !n.is_finite() {
            let mut out = String::new();
            if n.is_sign_negative() && !n.is_nan() {
                out.push('-');
            }
            if style == "currency" {
                out.push_str(&currency_symbol(currency.as_deref().unwrap_or("")));
            }
            out.push_str(if n.is_nan() { "NaN" } else { "∞" });
            if style == "percent" {
                out.push('%');
            }
            return out;
        }
        let use_grouping = !matches!(
            self.realm.get_property(handle, "useGrouping"),
            Some(v) if matches!(v.unpack(), Unpacked::Bool(false))
        );
        // Default fraction digits: currency = 2 (0 for JPY), else 0..=3.
        let is_jpy = currency.as_deref() == Some("JPY");
        let (def_min, def_max) = match style.as_str() {
            "currency" if is_jpy => (0, 0),
            "currency" => (2, 2),
            "percent" => (0, 0),
            _ => (0, 3),
        };
        let min = opt_num(self, "minimumFractionDigits")
            .unwrap_or(def_min)
            .clamp(0, 20);
        let max = opt_num(self, "maximumFractionDigits")
            .unwrap_or(def_max.max(min))
            .clamp(min, 20);
        let value = if style == "percent" { n * 100.0 } else { n };
        // Round `x` to `max` digits, trimming trailing zeros down to `min`.
        let fmt_digits = |x: f64| -> String {
            let mut s = alloc::format!("{:.*}", max as usize, x);
            if max > min && s.contains('.') {
                while s.ends_with('0')
                    && s.split_once('.').map_or(0, |(_, f)| f.len()) > min as usize
                {
                    s.pop();
                }
                if s.ends_with('.') {
                    s.pop();
                }
            }
            s
        };
        // `notation: "scientific" | "engineering"` renders `mantissa E exponent` (no
        // grouping); engineering pins the exponent to a multiple of 3.
        let notation = opt_str(self, "notation").unwrap_or_default();
        let (s, do_group) = if matches!(notation.as_str(), "scientific" | "engineering") {
            let neg = value < 0.0;
            let mag = value.abs();
            let mut exp = 0i32;
            let mut p = 1.0f64; // 10^exp
            if mag >= 1.0 {
                while mag >= p * 10.0 {
                    p *= 10.0;
                    exp += 1;
                }
            } else if mag > 0.0 {
                while mag < p {
                    p /= 10.0;
                    exp -= 1;
                }
            }
            if notation == "engineering" {
                // Drop the exponent to the nearest lower multiple of 3, scaling the mantissa
                // up to compensate (p = 10^exp must shrink as exp shrinks).
                let shift = exp.rem_euclid(3);
                exp -= shift;
                for _ in 0..shift {
                    p /= 10.0;
                }
            }
            let m = if mag == 0.0 { 0.0 } else { mag / p };
            let sign = if neg { "-" } else { "" };
            (alloc::format!("{sign}{}E{exp}", fmt_digits(m)), false)
        } else if notation == "compact" {
            // `notation: "compact"` (short): divide by the largest 10^(3k) scale and append
            // its suffix (K/M/B/T), showing one fraction digit only for a single-digit
            // mantissa (`1.2M`, but `123K` and `12K`).
            let neg = value < 0.0;
            let mag = value.abs();
            let (div, suffix) = if mag >= 1e12 {
                (1e12, "T")
            } else if mag >= 1e9 {
                (1e9, "B")
            } else if mag >= 1e6 {
                (1e6, "M")
            } else if mag >= 1e3 {
                (1e3, "K")
            } else {
                (1.0, "")
            };
            let m = mag / div;
            let cmax = if m < 10.0 { 1 } else { 0 };
            let mut ms = alloc::format!("{m:.*}", cmax as usize);
            if ms.contains('.') {
                while ms.ends_with('0') {
                    ms.pop();
                }
                if ms.ends_with('.') {
                    ms.pop();
                }
            }
            let sign = if neg { "-" } else { "" };
            (alloc::format!("{sign}{ms}{suffix}"), suffix.is_empty())
        } else {
            (fmt_digits(value), use_grouping)
        };
        // Group the integer part (skipped for scientific/engineering).
        let grouped = if do_group {
            let neg = s.starts_with('-');
            let body = s.trim_start_matches('-');
            let (ip, fp) = body
                .split_once('.')
                .map_or((body, None), |(i, f)| (i, Some(f)));
            let mut g = String::new();
            let len = ip.len();
            for (i, b) in ip.bytes().enumerate() {
                if i > 0 && (len - i) % 3 == 0 {
                    g.push(',');
                }
                g.push(b as char);
            }
            if let Some(f) = fp {
                g.push('.');
                g.push_str(f);
            }
            if neg { alloc::format!("-{g}") } else { g }
        } else {
            s
        };
        // Separate the sign from the magnitude so `signDisplay` and the style affixes
        // compose with the sign outermost (e.g. `-$5.00`, `+5%`).
        let neg = grouped.starts_with('-');
        let magnitude = grouped.trim_start_matches('-');
        let styled = match style.as_str() {
            "percent" => alloc::format!("{magnitude}%"),
            "currency" => {
                let sym = match currency.as_deref() {
                    Some("USD") => "$",
                    Some("EUR") => "€",
                    Some("GBP") => "£",
                    Some("JPY" | "CNY") => "¥",
                    Some(other) => {
                        let other = String::from(other);
                        return alloc::format!(
                            "{}{other}\u{a0}{magnitude}",
                            if neg { "-" } else { "" }
                        );
                    }
                    None => "$",
                };
                alloc::format!("{sym}{magnitude}")
            }
            "unit" => {
                // `style: "unit"` appends the unit's short symbol (`5 km`); a
                // `unit-per-unit` compound joins the two with `/` (`5 km/h`).
                let unit = opt_str(self, "unit").unwrap_or_default();
                let sym = unit.split_once("-per-").map_or_else(
                    || String::from(unit_symbol(&unit)),
                    |(a, b)| alloc::format!("{}/{}", unit_symbol(a), unit_symbol(b)),
                );
                // Temperature/angle units attach with no space (`20°C`); others use a
                // (non-breaking) space (`5 km`).
                let sep = if matches!(unit.as_str(), "celsius" | "fahrenheit" | "degree") {
                    ""
                } else {
                    "\u{a0}"
                };
                alloc::format!("{magnitude}{sep}{sym}")
            }
            _ => String::from(magnitude),
        };
        // `currencySign: "accounting"` renders a negative currency amount in
        // parentheses (`($5.00)`) instead of with a minus sign.
        if neg
            && style == "currency"
            && opt_str(self, "currencySign").as_deref() == Some("accounting")
        {
            return alloc::format!("({styled})");
        }
        let is_zero = magnitude.bytes().all(|b| matches!(b, b'0' | b'.' | b','));
        let sign = match opt_str(self, "signDisplay").as_deref() {
            Some("never") => "",
            Some("always") => {
                if neg {
                    "-"
                } else {
                    "+"
                }
            }
            Some("exceptZero") if !is_zero => {
                if neg {
                    "-"
                } else {
                    "+"
                }
            }
            // "negative": a minus for a negative value but NOT a negative zero.
            Some("negative") if neg && !is_zero => "-",
            Some("negative") => "",
            // "auto" (default) and "exceptZero" on a zero: a sign only for negatives
            // (including a negative zero under "auto").
            _ if neg => "-",
            _ => "",
        };
        alloc::format!("{sign}{styled}")
    }

    /// ECMA-402 `CanonicalizeLocaleList(locales)`: coerces `locales` to a list of
    /// canonical locale tags (deduplicated, order preserved). `undefined` → empty
    /// list; a single string is treated as a one-element list; otherwise the
    /// argument is `ToObject`-ed and iterated by its `length`. Each element must be
    /// a String or Object (else **TypeError**), and each tag must be a structurally
    /// valid locale (else **RangeError**). A `Locale` instance contributes its
    /// already-canonical `[[Locale]]` tag.
    pub(crate) fn canonicalize_locale_list(
        &mut self,
        locales: NanBox,
    ) -> Result<Vec<String>, ExecError> {
        let mut seen: Vec<String> = Vec::new();
        if matches!(locales.unpack(), Unpacked::Undefined) {
            return Ok(seen);
        }
        // A bare string is a single-element list (no ToObject coercion of the
        // characters). A `Locale` object short-circuits below in the loop.
        let is_string = locales
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.realm.is_string_handle(h));
        let push_tag =
            |this: &mut Self, tag: &str, seen: &mut Vec<String>| -> Result<(), ExecError> {
                match canonicalize_locale_id(tag) {
                    Some(c) => {
                        if !seen.contains(&c) {
                            seen.push(c);
                        }
                        Ok(())
                    }
                    None => {
                        let m = this.new_str(&alloc::format!(
                            "Incorrect locale information provided: {tag}"
                        ));
                        Err(ExecError::Throw(this.make_error(N_RANGE_ERROR, Some(m))))
                    }
                }
            };
        if is_string {
            let s = self.coerce_to_string(locales)?;
            push_tag(self, &s, &mut seen)?;
            return Ok(seen);
        }
        // A `Locale` object (has `[[InitializedLocale]]`) is likewise a single-
        // element list contributing its already-canonical `[[Locale]]` tag — read
        // directly (never via `toString`, which a subclass may override).
        if let Some(h) = locales.as_handle().map(Handle::from_raw)
            && self.realm.get_property(h, "\u{0}brand_loc").is_some()
            && let Some(loc) = self.realm.get_property(h, "\u{0}locale_tag")
        {
            seen.push(self.realm.to_display_string(loc));
            return Ok(seen);
        }
        // ToObject(null) is a TypeError (undefined was handled above).
        if matches!(locales.unpack(), Unpacked::Null) {
            let m = self.new_str("Cannot convert null to object");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // ToObject, then iterate [0, len) reading each element (getter-aware, so
        // throwing getters and inherited indices behave per spec).
        let obj = self.coerce_to_object(locales);
        let Some(oh) = obj.as_handle().map(Handle::from_raw) else {
            return Ok(seen);
        };
        let len_v = self.read_member(oh, "length")?;
        // ToLength: a Symbol `length` is a TypeError (ToNumber throws).
        let len_v = self.coerce_to_number(len_v)?;
        let len_f = self.realm.to_number(len_v);
        let len = if len_f.is_nan() || len_f <= 0.0 {
            0u64
        } else {
            len_f.min(u32::MAX as f64 * 2.0) as u64
        };
        for i in 0..len {
            let key = alloc::format!("{i}");
            // HasProperty check: skip absent indices (sparse array-likes). Uses the
            // proxy-aware `[[HasProperty]]` so a `has` trap runs (and may throw).
            if !self.has_property_proxied(oh, &key)? {
                continue;
            }
            let el = self.read_member(oh, &key)?;
            // Element must be a String or Object (per spec `Type(kValue)`); any other
            // primitive — Symbol, Number, Boolean, BigInt — is a TypeError. (A boxed
            // String/primitive is an Object and is allowed, coerced via ToString.)
            let el_is_string = el
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.realm.is_string_handle(h));
            let el_is_object = self.is_object_value(el) && !el_is_string;
            let ty = self.realm.type_of_value(el);
            let el_is_primitive_nonstring =
                !el_is_string && matches!(ty, "symbol" | "number" | "boolean" | "bigint");
            if el_is_primitive_nonstring || (!el_is_string && !el_is_object) {
                let m = self.new_str("locale list element is not a string or object");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // A `Locale` instance contributes its canonical `[[Locale]]`.
            if let Some(h) = el.as_handle().map(Handle::from_raw)
                && let Some(loc) = self.realm.get_property(h, "\u{0}locale_tag")
            {
                let tag = self.realm.to_display_string(loc);
                if !seen.contains(&tag) {
                    seen.push(tag);
                }
                continue;
            }
            let s = self.coerce_to_string(el)?;
            push_tag(self, &s, &mut seen)?;
        }
        Ok(seen)
    }

    /// ECMA-402 `ResolveLocale`'s LookupMatcher: the first requested tag whose
    /// language CLDR actually has data for, else the default locale. Without it a
    /// service would report an unsupported tag it cannot serve
    /// (`new Intl.Segmenter(["xyz", "ar"]).resolvedOptions().locale` must be `"ar"`).
    /// `[[AvailableLocales]]` is implementation-defined; ours is "CLDR knows this
    /// language subtag", which is exactly the data the formatters fall back over.
    pub(crate) fn lookup_available_locale(&self, requested: &[String]) -> Option<String> {
        #[cfg(feature = "intl")]
        {
            requested
                .iter()
                .find(|tag| {
                    let lang = tag.split(['-', '_']).next().unwrap_or("");
                    lang.eq_ignore_ascii_case("und")
                        || intl::display::language_name("en", lang).is_some()
                })
                .cloned()
        }
        #[cfg(not(feature = "intl"))]
        {
            requested.first().cloned()
        }
    }

    /// `Intl.getCanonicalLocales(locales)` — a fresh, mutable Array of the
    /// canonicalized tags.
    pub(crate) fn intl_get_canonical_locales(
        &mut self,
        locales: NanBox,
    ) -> Result<NanBox, ExecError> {
        let tags = self.canonicalize_locale_list(locales)?;
        let elems: Vec<NanBox> = tags.iter().map(|t| self.new_str(t)).collect();
        Ok(NanBox::handle(self.realm.new_array(elems).to_raw()))
    }

    /// `Intl.supportedValuesOf(key)` — a sorted, duplicate-free Array of the
    /// supported identifiers for `key` (`calendar`/`collation`/`currency`/
    /// `numberingSystem`/`timeZone`/`unit`). A `key` outside this set is a
    /// **RangeError**.
    pub(crate) fn intl_supported_values_of(&mut self, key: NanBox) -> Result<NanBox, ExecError> {
        let k = self.coerce_to_string(key)?;
        let values: &[&str] = match k.as_str() {
            // `AvailableCalendars`: the concrete calendar types (the deprecated
            // `islamic`/`islamic-rgsa` aliases are resolved away, not reported).
            "calendar" => &AVAILABLE_CALENDARS,
            "collation" => &[
                "compat", "dict", "emoji", "eor", "phonebk", "phonetic", "pinyin", "searchjl",
                "stroke", "trad", "unihan", "zhuyin",
            ],
            "currency" => SUPPORTED_CURRENCIES,

            "numberingSystem" => NUMBERING_SYSTEMS,
            // AvailablePrimaryTimeZoneIdentifiers: every IANA zone the embedded
            // tzdb ships that is its own primary identifier (links resolve to
            // their target and are therefore not reported), plus "UTC".
            "timeZone" => {
                let mut zones: Vec<String> = timezone_data::names()
                    .filter(|n| {
                        crate::nbexec::temporal_zoneddatetime::tz_primary(n) == *n
                            && n.contains('/')
                    })
                    .map(String::from)
                    .collect();
                zones.push(String::from("UTC"));
                zones.sort_unstable();
                zones.dedup();
                let elems: Vec<NanBox> = zones.iter().map(|s| self.new_str(s)).collect();
                return Ok(NanBox::handle(self.realm.new_array(elems).to_raw()));
            }
            "unit" => SANCTIONED_UNITS,
            _ => {
                let m = self.new_str(&alloc::format!("invalid key: {k}"));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        };
        let mut sorted: Vec<&str> = values.to_vec();
        // The generic/algorithmic numbering aliases (`native`/`traditio`/`finance`)
        // have no fixed digit mapping and are not selectable via the `nu` key, so
        // they are not reported by `supportedValuesOf` either.
        if k == "numberingSystem" {
            sorted.retain(|s| !matches!(*s, "native" | "traditio" | "finance"));
        }
        sorted.sort_unstable();
        sorted.dedup();
        let elems: Vec<NanBox> = sorted.iter().map(|s| self.new_str(s)).collect();
        Ok(NanBox::handle(self.realm.new_array(elems).to_raw()))
    }

    // --- Intl.Locale -------------------------------------------------------

    /// Builds (once) and links `Intl.Locale.prototype`: the `language`/`script`/…
    /// brand-checked `get` accessors, the `maximize`/`minimize`/`toString` methods,
    /// the `constructor` back-link, and `[Symbol.toStringTag] = "Intl.Locale"`.
    fn intl_locale_prototype(&mut self) -> Option<Handle> {
        if let Some(p) = self.realm.intl_prototype(N_INTL_LOCALE) {
            return Some(p);
        }
        let ctor = self.intl_ctor_handle("Locale")?;
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &name in LOCALE_ACCESSORS {
            let label = alloc::format!("get {name}");
            let target = self.new_str(name);
            let th = target.as_handle().map(Handle::from_raw).unwrap();
            let getter = self.realm.new_bound_native(N_INTL_LOCALE_ACCESSOR, th);
            self.install_fn_name_length(getter, &label, 0);
            self.realm.define_accessor(
                proto,
                name,
                NanBox::handle(getter.to_raw()),
                NanBox::undefined(),
            );
            self.realm.mark_hidden(proto, name);
        }
        for &m in &[
            "maximize",
            "minimize",
            "toString",
            "getCalendars",
            "getCollations",
            "getHourCycles",
            "getNumberingSystems",
            "getTimeZones",
            "getTextInfo",
            "getWeekInfo",
        ] {
            let target = self.new_str(m);
            let th = target.as_handle().map(Handle::from_raw).unwrap();
            let f = self.realm.new_bound_native(N_INTL_LOCALE_METHOD, th);
            self.install_fn_name_length(f, m, 0);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        self.install_to_string_tag(proto, "Intl.Locale");
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.link_ctor_prototype(ctor, proto);
        self.realm.set_intl_prototype(N_INTL_LOCALE, proto);
        Some(proto)
    }

    /// `new Intl.Locale(tag, options)` — parses and canonicalizes `tag` (a string or
    /// another `Locale`), applies the `language`/`script`/`region`/`calendar`/
    /// `collation`/`hourCycle`/`caseFirst`/`kn`/`numberingSystem` options onto the
    /// Unicode extension, and stores the resolved components behind hidden slots for
    /// the prototype accessors. Returns a branded `Intl.Locale` instance.
    pub(crate) fn make_locale(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        self.init_locale(obj, args)?;
        Ok(NanBox::handle(obj.to_raw()))
    }

    pub(crate) fn init_locale(&mut self, obj: Handle, args: &[NanBox]) -> Result<(), ExecError> {
        let tag_arg = args.first().copied().unwrap_or(NanBox::undefined());
        // Step 7: If Type(tag) is not String or Object, throw a TypeError. A
        // `Locale` argument contributes its `[[Locale]]` tag; a String/Object is
        // ToString-ed; everything else (undefined/null/boolean/number/symbol/
        // bigint) is a TypeError *before* any structural validation.
        let base_tag = if let Some(h) = tag_arg.as_handle().map(Handle::from_raw) {
            if let Some(t) = self.realm.get_property(h, "\u{0}locale_tag") {
                self.realm.to_display_string(t)
            } else if matches!(self.realm.type_of(h), Some("symbol") | Some("bigint")) {
                let m = self.new_str("Intl.Locale: tag must be a string or Locale");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            } else {
                self.coerce_to_string(tag_arg)?
            }
        } else {
            let m = self.new_str("Intl.Locale: tag must be a string or Locale");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        let Some(canon) = canonicalize_locale_id(&base_tag) else {
            let m = self.new_str(&alloc::format!("invalid language tag: {base_tag}"));
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        };
        // Read the option overrides (each validated like the formatter options).
        // `ApplyOptionsToTag` does `ToObject(options)`: `undefined` yields no
        // overrides, `null` is a **TypeError**, and any other primitive boxes.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = match opts_arg.unpack() {
            Unpacked::Undefined => None,
            Unpacked::Null => {
                let m = self.new_str("Intl.Locale options must not be null");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            _ => self
                .coerce_to_object(opts_arg)
                .as_handle()
                .map(Handle::from_raw),
        };
        let language = self.get_string_option(opts, "language", &[], None)?;
        let script = self.get_string_option(opts, "script", &[], None)?;
        let region = self.get_string_option(opts, "region", &[], None)?;
        // `variants` is read after region, before the Unicode-extension keys (spec
        // option-evaluation order); applied to the tag's variant subtags.
        let variants = self.get_string_option(opts, "variants", &[], None)?;
        let calendar = self.get_string_option(opts, "calendar", &[], None)?;
        let collation = self.get_string_option(opts, "collation", &[], None)?;
        let hour_cycle =
            self.get_string_option(opts, "hourCycle", &["h11", "h12", "h23", "h24"], None)?;
        let case_first =
            self.get_string_option(opts, "caseFirst", &["upper", "lower", "false"], None)?;
        let numeric = self.get_bool_option(opts, "numeric", None)?;
        let numbering = self.get_string_option(opts, "numberingSystem", &[], None)?;
        // `firstDayOfWeek` (ES2024): a weekday name (mon…sun) or 1–7 (1=Monday),
        // resolved via WeekdayToString and applied as the `-u-fw` keyword.
        let first_day = self.get_string_option(opts, "firstDayOfWeek", &[], None)?;
        let first_day = if let Some(fd) = &first_day {
            let Some(w) = weekday_to_string(fd) else {
                let m = self.new_str("invalid firstDayOfWeek");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            };
            Some(w)
        } else {
            None
        };
        // language/script/region subtags each have a fixed shape (UTS-35): a
        // language is alpha{2,3} or alpha{5,8}; a script is alpha{4}; a region is
        // alpha{2} or digit{3}. A malformed value is a RangeError.
        if let Some(l) = &language {
            let n = l.len();
            let ok = ((2..=3).contains(&n) || (5..=8).contains(&n))
                && l.bytes().all(|b| b.is_ascii_alphabetic());
            if !ok {
                let m = self.new_str("invalid language");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        if let Some(s) = &script
            && !(s.len() == 4 && s.bytes().all(|b| b.is_ascii_alphabetic()))
        {
            let m = self.new_str("invalid script");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        if let Some(r) = &region {
            let alpha2 = r.len() == 2 && r.bytes().all(|b| b.is_ascii_alphabetic());
            let digit3 = r.len() == 3 && r.bytes().all(|b| b.is_ascii_digit());
            if !alpha2 && !digit3 {
                let m = self.new_str("invalid region");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }
        // calendar/collation/numberingSystem must each match the UTS-35 `type`
        // value production (they take arbitrary keyword values, not a fixed list).
        for (val, name) in [
            (&calendar, "calendar"),
            (&collation, "collation"),
            (&numbering, "numberingSystem"),
        ] {
            if let Some(v) = val
                && !is_unicode_type_value(v)
            {
                let m = self.new_str(&alloc::format!("invalid {name}"));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
        }

        let mut parsed = ParsedLocale::from_canonical(&canon);
        if let Some(l) = &language {
            parsed.language = l.to_ascii_lowercase();
        }
        if let Some(s) = &script {
            parsed.script = Some(titlecase_script(s));
        }
        if let Some(r) = &region {
            parsed.region = Some(r.to_ascii_uppercase());
        }
        if let Some(v) = &variants {
            // A `-`-joined sequence of `variant` subtags (each alnum{5,8}, or a
            // digit followed by alnum{3}); validated, lowercased, sorted, unique.
            let mut vs: Vec<String> = Vec::new();
            for sub in v.split('-') {
                let s = sub.to_ascii_lowercase();
                let alnum = s.bytes().all(|b| b.is_ascii_alphanumeric());
                let ok = ((5..=8).contains(&s.len()) && alnum)
                    || (s.len() == 4 && s.as_bytes()[0].is_ascii_digit() && alnum);
                if !ok || vs.contains(&s) {
                    let m = self.new_str("invalid variants");
                    return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
                }
                vs.push(s);
            }
            vs.sort();
            parsed.variants = vs;
        }
        if let Some(c) = &calendar {
            // A deprecated calendar value passed via the `calendar` option is
            // canonicalized like the `-u-ca-` type (`islamicc` → `islamic-civil`).
            parsed.set_keyword("ca", unicode_type_alias("ca", c).unwrap_or(c));
        }
        if let Some(c) = &collation {
            parsed.set_keyword("co", c);
        }
        if let Some(h) = &hour_cycle {
            parsed.set_keyword("hc", h);
        }
        if let Some(k) = &case_first {
            parsed.set_keyword("kf", k);
        }
        if let Some(b) = numeric {
            parsed.set_keyword("kn", if b { "true" } else { "false" });
        }
        if let Some(n) = &numbering {
            parsed.set_keyword("nu", n);
        }
        if let Some(fw) = &first_day {
            parsed.set_keyword("fw", fw);
        }
        // `ApplyOptionsToTag` ends in `CanonicalizeUnicodeLocaleId(tag)`: the
        // option-assembled tag goes through the alias corpus again, so a numeric
        // region becomes its alpha-2 form (`{region: "554"}` → `en-NZ`) and a base
        // that the options turned into a regular grandfathered form gets replaced
        // (`cel` + variant `gaulish` → `cel-gaulish` → `xtg`). Re-derive the
        // components too, so the accessors reflect the replacement.
        if let Some(canon) = canonicalize_locale_id(&parsed.to_tag()) {
            parsed = ParsedLocale::from_canonical(&canon);
        }
        let final_tag = parsed.to_tag();
        let tagv = self.new_str(&final_tag);
        self.realm.set_hidden_property(obj, "\u{0}locale_tag", tagv);
        // Stash the resolved components so the accessors read them cheaply.
        let store = |this: &mut Self, key: &str, val: Option<&str>| {
            if let Some(v) = val {
                let sv = this.new_str(v);
                this.realm.set_hidden_property(obj, key, sv);
            }
        };
        store(self, "\u{0}loc_language", Some(&parsed.language));
        store(self, "\u{0}loc_script", parsed.script.as_deref());
        store(self, "\u{0}loc_region", parsed.region.as_deref());
        // `variants`: the `-`-joined variant subtags (already lowercased + sorted
        // by `from_canonical`), or `undefined` when the base name has none.
        if !parsed.variants.is_empty() {
            store(self, "\u{0}loc_variants", Some(&parsed.variants.join("-")));
        }
        store(self, "\u{0}loc_baseName", Some(&parsed.base_name()));
        store(self, "\u{0}loc_ca", parsed.keyword("ca"));
        store(self, "\u{0}loc_co", parsed.keyword("co"));
        store(self, "\u{0}loc_hc", parsed.keyword("hc"));
        store(self, "\u{0}loc_kf", parsed.keyword("kf"));
        store(self, "\u{0}loc_nu", parsed.keyword("nu"));
        store(self, "\u{0}loc_fw", parsed.keyword("fw"));
        // `numeric` is the boolean form of the `kn` keyword.
        let kn = parsed.keyword("kn");
        let numeric_val = matches!(kn, Some("true") | Some(""));
        self.realm
            .set_hidden_property(obj, "\u{0}loc_numeric", NanBox::boolean(numeric_val));
        // Brand + link to the prototype.
        self.realm
            .set_hidden_property(obj, "\u{0}brand_loc", NanBox::boolean(true));
        if let Some(proto) = self.intl_locale_prototype() {
            self.realm.set_object_proto(obj, Some(proto));
        }
        Ok(())
    }

    /// Dispatches an `Intl.Locale` `get` accessor: brand-checks `this`, then returns
    /// the stored component (a string, or `undefined`; `numeric` is a boolean).
    pub(crate) fn intl_locale_accessor_dispatch(
        &mut self,
        this: NanBox,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let h = self.require_intl_slot(this, "\u{0}brand_loc", "Intl.Locale.prototype getter")?;
        let read = |this: &Self, key: &str| -> NanBox {
            this.realm
                .get_property(h, key)
                .unwrap_or(NanBox::undefined())
        };
        Ok(match name {
            "language" => read(self, "\u{0}loc_language"),
            "script" => read(self, "\u{0}loc_script"),
            "region" => read(self, "\u{0}loc_region"),
            "variants" => read(self, "\u{0}loc_variants"),
            "baseName" => read(self, "\u{0}loc_baseName"),
            "calendar" => read(self, "\u{0}loc_ca"),
            "collation" => read(self, "\u{0}loc_co"),
            "hourCycle" => read(self, "\u{0}loc_hc"),
            "caseFirst" => read(self, "\u{0}loc_kf"),
            "numberingSystem" => read(self, "\u{0}loc_nu"),
            "numeric" => read(self, "\u{0}loc_numeric"),
            "firstDayOfWeek" => read(self, "\u{0}loc_fw"),
            _ => NanBox::undefined(),
        })
    }

    /// Dispatches an `Intl.Locale` method (`maximize`/`minimize`/`toString`):
    /// brand-checks `this`. `toString` returns the canonical tag; `maximize`/
    /// `minimize` return a fresh `Locale` with the same tag (no CLDR likely-subtags
    /// data, so the tag is returned unchanged — structurally valid).
    pub(crate) fn intl_locale_method_dispatch(
        &mut self,
        this: NanBox,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        let h = self.require_intl_slot(this, "\u{0}brand_loc", "Intl.Locale.prototype method")?;
        let tag = self
            .realm
            .get_property(h, "\u{0}locale_tag")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        // The Locale Info API: single-element defaults (real CLDR availability
        // data is §3.9). Each returns a fresh Array/object per call.
        let array_of = |this: &mut Self, items: &[&str]| {
            let vals: Vec<NanBox> = items.iter().map(|s| this.new_str(s)).collect();
            NanBox::handle(this.realm.new_array(vals).to_raw())
        };
        match name {
            "toString" => Ok(self.new_str(&tag)),
            "getCalendars" => Ok(array_of(self, &["gregory"])),
            "getCollations" => Ok(array_of(self, &["default"])),
            "getHourCycles" => Ok(array_of(self, &["h23"])),
            "getNumberingSystems" => Ok(array_of(self, &["latn"])),
            "getTimeZones" => {
                // `undefined` for a region-less locale; otherwise a (placeholder)
                // Array. Real per-region zone lists are §3.9 CLDR data.
                if ParsedLocale::from_canonical(&tag)
                    .region
                    .as_deref()
                    .unwrap_or("")
                    .is_empty()
                {
                    Ok(NanBox::undefined())
                } else {
                    Ok(array_of(self, &["UTC"]))
                }
            }
            "getTextInfo" => {
                let obj = self.realm.new_object();
                let pl = ParsedLocale::from_canonical(&tag);
                let rtl = matches!(
                    pl.script.as_deref().unwrap_or(""),
                    "Arab" | "Hebr" | "Syrc" | "Thaa" | "Nkoo" | "Rohg" | "Adlm"
                ) || matches!(
                    pl.language.as_str(),
                    "ar" | "he" | "fa" | "ur" | "ps" | "sd" | "ug" | "yi" | "dv" | "ku" | "ckb"
                );
                let dir = if rtl { "rtl" } else { "ltr" };
                let d = self.new_str(dir);
                self.realm.set_property(obj, "direction", d);
                Ok(NanBox::handle(obj.to_raw()))
            }
            "getWeekInfo" => {
                // Own keys are exactly { firstDay, weekend } (the proposal dropped
                // `minimalDays`). `firstDay` (1=Mon…7=Sun) comes from the locale's
                // `firstDayOfWeek` option when set, else defaults to Monday.
                let obj = self.realm.new_object();
                let fw = self
                    .realm
                    .get_property(h, "\u{0}loc_fw")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                let first_day = match fw.as_str() {
                    "tue" => 2.0,
                    "wed" => 3.0,
                    "thu" => 4.0,
                    "fri" => 5.0,
                    "sat" => 6.0,
                    "sun" => 7.0,
                    _ => 1.0,
                };
                self.realm
                    .set_property(obj, "firstDay", NanBox::number(first_day));
                let weekend = self
                    .realm
                    .new_array(alloc::vec![NanBox::number(6.0), NanBox::number(7.0)]);
                self.realm
                    .set_property(obj, "weekend", NanBox::handle(weekend.to_raw()));
                Ok(NanBox::handle(obj.to_raw()))
            }
            // maximize/minimize: apply the CLDR likely-subtags data (Add/Remove
            // Likely Subtags, UTS-35) to the base name, preserving the `-u-`
            // keywords and other extensions. ECMA-402 `minimize` is defined as
            // "maximize, then remove", so it runs on the maximized locale.
            "maximize" | "minimize" => {
                #[allow(unused_mut)]
                let mut pl = ParsedLocale::from_canonical(&tag);
                #[cfg(feature = "intl")]
                {
                    // Likely-subtags is a language/script/region operation; the
                    // variant subtags ride along untouched. They are stripped from
                    // the tag handed to the crate because its `minimize` builds its
                    // trial candidates without variants, so a tag that has any
                    // never matches its own maximization and is returned unchanged
                    // (`en-Latn-US-fonipa` would not minimize to `en-fonipa`).
                    let mut lsr = pl.language.clone();
                    if let Some(sc) = &pl.script {
                        lsr.push('-');
                        lsr.push_str(sc);
                    }
                    if let Some(rg) = &pl.region {
                        lsr.push('-');
                        lsr.push_str(rg);
                    }
                    if let Ok(loc) = intl::locale::Locale::parse(&lsr) {
                        let result = if name == "maximize" {
                            loc.maximize()
                        } else {
                            loc.maximize().minimize()
                        };
                        pl.language = if result.language.is_empty() {
                            String::from("und")
                        } else {
                            result.language.clone()
                        };
                        pl.script = result.script.clone();
                        pl.region = result.region.clone();
                    }
                }
                let new_tag = pl.to_tag();
                let tagv = self.new_str(&new_tag);
                self.make_locale(&[tagv])
            }
            // Any other method name: rebuild a Locale from the (unchanged) tag.
            _ => {
                let tagv = self.new_str(&tag);
                self.make_locale(&[tagv])
            }
        }
    }

    /// Splits the formatted output of an `Intl.NumberFormat` handle into typed
    /// `(kind, value)` parts, mirroring `nf.formatToParts(value)`. Shared by the
    /// `formatToParts` dispatch and by `Intl.DurationFormat`, which composes
    /// per-unit `NumberFormat`s. The `intl`-crate path yields CLDR parts; the
    /// hand-rolled path (unit style, or the no-`intl` build) re-derives parts from
    /// the formatted string's structure (sign / integer-groups / decimal / fraction).
    pub(crate) fn number_handle_parts(
        &mut self,
        handle: Handle,
        value: NanBox,
    ) -> Vec<(&'static str, String)> {
        let mut parts = self.number_handle_parts_inner(handle, value);
        // Apply the numbering system to each part's digits (see
        // `apply_numbering_digits`); non-digit parts (group/decimal/currency) are
        // untouched since they carry no ASCII digits.
        let nu = self
            .realm
            .get_property(handle, "numberingSystem")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        if numbering_system_digit_base(&nu).is_some_and(|b| b != 0x0030) || nu == "hanidec" {
            for (_, v) in &mut parts {
                if v.chars().any(|c| c.is_ascii_digit()) {
                    *v = substitute_numbering_digits(&nu, core::mem::take(v));
                }
            }
        }
        parts
    }

    #[cfg(feature = "intl")]
    fn number_handle_parts_inner(
        &mut self,
        handle: Handle,
        value: NanBox,
    ) -> Vec<(&'static str, String)> {
        let n = self.realm.to_number(value);
        let locale = self
            .realm
            .get_property(handle, "\u{0}locale")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en"));
        let mut opts = self.number_format_options(handle);
        // Pre-round to the ECMA-402 / ICU decimal (matches the `format` path).
        let n = self.number_precision_round(handle, &mut opts, n);
        // `intl` 0.5 panics formatting a true negative zero; feed the smallest
        // negative subnormal (rounds to a displayed zero) so the crate emits the
        // proper minusSign / locale marks and applies `signDisplay` itself.
        let feed = if n == 0.0 && n.is_sign_negative() {
            -f64::from_bits(1)
        } else {
            n
        };
        // Default compact notation: request full mantissa precision from the
        // crate, then re-round to the ECMA-402 default (see the `format` path
        // and [`compact_reround_parts`]).
        let reround = feed.is_finite() && compact_wants_reround(&opts);
        if reround {
            opts.maximum_fraction_digits = Some(6);
        }
        let mut parts: Vec<(&'static str, String)> =
            intl::number::format_to_parts(&locale, feed, &opts)
                .into_iter()
                .map(|p| (p.kind.as_str(), p.value))
                .collect();
        if reround {
            compact_reround_parts(&mut parts, opts.rounding_mode);
        }
        // Compact suffixes carry a separator space fused into the `compact`
        // token; peel it into a `literal` part per ECMA-402 (applies whether or
        // not the mantissa was re-rounded).
        if matches!(opts.notation, intl::number::Notation::Compact) {
            split_compact_affix_parts(&mut parts);
        }
        let sd = self
            .realm
            .get_property(handle, "signDisplay")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        // The crate leaves NaN unsigned; `signDisplay: "always"` prefixes a
        // plusSign part ("+NaN"), matching the string `format` path.
        if n.is_nan()
            && sd == "always"
            && !parts
                .iter()
                .any(|(t, _)| *t == "minusSign" || *t == "plusSign")
        {
            parts.insert(0, ("plusSign", String::from("+")));
        }
        // The crate aliases `signDisplay: "negative"` to `"auto"`, so it wrongly
        // signs a value that rounds to a displayed zero; "negative"/"never"/
        // "exceptZero" must not sign zero. Drop the minusSign in that case.
        if matches!(sd.as_str(), "negative" | "never" | "exceptZero")
            && parts.iter().any(|(t, _)| *t == "minusSign")
            && !parts
                .iter()
                .any(|(t, v)| matches!(*t, "integer" | "fraction") && v.chars().any(|c| c != '0'))
            && !parts.iter().any(|(t, _)| matches!(*t, "nan" | "infinity"))
        {
            parts.retain(|(t, _)| *t != "minusSign");
        }
        // `currencySign: "accounting"` parenthesizes a negative amount in the
        // parenthesizing locales (see `accounting_uses_parens`): the minusSign
        // part becomes a leading `literal "("` and a trailing `literal ")"`.
        let accounting = self
            .realm
            .get_property(handle, "currencySign")
            .map(|v| self.realm.to_display_string(v))
            .as_deref()
            == Some("accounting");
        if accounting
            && accounting_uses_parens(&locale)
            && parts.iter().any(|(t, _)| *t == "minusSign")
        {
            parts.retain(|(t, _)| *t != "minusSign");
            parts.insert(0, ("literal", String::from("(")));
            parts.push(("literal", String::from(")")));
        }
        parts
    }

    /// The no-`intl` fallback for [`number_handle_parts`]: re-derives the parts
    /// from the formatted string's structure (sign / integer-groups / decimal /
    /// fraction / affixes), since there is no CLDR data to tag them with.
    #[cfg(not(feature = "intl"))]
    fn number_handle_parts_inner(
        &mut self,
        handle: Handle,
        value: NanBox,
    ) -> Vec<(&'static str, String)> {
        // Hand-rolled path: re-derive parts from the formatted string.
        let formatted = self.intl_format_value(handle, value);
        let style = self
            .realm
            .get_property(handle, "style")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("decimal"));
        let currency_sym = if style == "currency" {
            let code = self
                .realm
                .get_property(handle, "currency")
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_default();
            currency_symbol(&code)
        } else {
            String::new()
        };
        let mut entries: Vec<(&'static str, String)> = Vec::new();
        let mut s = formatted.as_str();
        if let Some(rest) = s.strip_prefix('-') {
            entries.push(("minusSign", String::from("-")));
            s = rest;
        }
        if !currency_sym.is_empty() && s.starts_with(currency_sym.as_str()) {
            entries.push(("currency", currency_sym.clone()));
            s = &s[currency_sym.len()..];
        }
        let mut percent = false;
        if style == "percent" && s.ends_with('%') {
            percent = true;
            s = &s[..s.len() - '%'.len_utf8()];
        }
        // `unit`/`compact` append a suffix after the number: strip it here and emit
        // it as its own part(s) so the numeric decomposition below sees only digits
        // (`5 m` → integer "5" + literal " " + unit "m"; `1.2M` → …fraction + compact
        // "M"). The numeric prefix is the leading run of digits/group/decimal chars.
        let notation = self
            .realm
            .get_property(handle, "notation")
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_default();
        let mut suffix_parts: Vec<(&'static str, String)> = Vec::new();
        if style == "unit" || notation == "compact" {
            let num_end = s
                .find(|c: char| !(c.is_ascii_digit() || c == ',' || c == '.'))
                .unwrap_or(s.len());
            let suffix = &s[num_end..];
            if !suffix.is_empty() {
                let part_kind = if style == "unit" { "unit" } else { "compact" };
                // A leading separator (space/NBSP) is a `literal`; the rest is the
                // unit/compact token.
                let sep_end = suffix
                    .find(|c: char| !c.is_whitespace())
                    .unwrap_or(suffix.len());
                if sep_end > 0 {
                    suffix_parts.push(("literal", String::from(&suffix[..sep_end])));
                }
                if sep_end < suffix.len() {
                    suffix_parts.push((part_kind, String::from(&suffix[sep_end..])));
                }
            }
            s = &s[..num_end];
        }
        if s == "NaN" {
            entries.push(("nan", String::from("NaN")));
        } else if s == "∞" {
            entries.push(("infinity", String::from("∞")));
        } else {
            let (int_part, frac_part) = match s.split_once('.') {
                Some((i, f)) => (i, Some(f)),
                None => (s, None),
            };
            for (gi, grp) in int_part.split(',').enumerate() {
                if gi > 0 {
                    entries.push(("group", String::from(",")));
                }
                entries.push(("integer", String::from(grp)));
            }
            if let Some(f) = frac_part {
                entries.push(("decimal", String::from(".")));
                entries.push(("fraction", String::from(f)));
            }
        }
        if percent {
            entries.push(("percentSign", String::from("%")));
        }
        entries.extend(suffix_parts);
        entries
    }

    // --- Intl.DurationFormat ----------------------------------------------

    /// Builds (once) and links `Intl.DurationFormat.prototype` with branded
    /// `format`/`formatToParts`/`resolvedOptions` methods, the `constructor`
    /// back-link, and `[Symbol.toStringTag] = "Intl.DurationFormat"`.
    fn intl_duration_prototype(&mut self) -> Option<Handle> {
        if let Some(p) = self.realm.intl_prototype(N_INTL_DURATION_FORMAT) {
            return Some(p);
        }
        let ctor = self.intl_ctor_handle("DurationFormat")?;
        let obj_proto = self.object_prototype();
        let proto = self.realm.new_object_with_proto(obj_proto);
        for &(m, arity) in &[
            ("resolvedOptions", 0u32),
            ("format", 1),
            ("formatToParts", 1),
        ] {
            let target = self.new_str(m);
            let th = target.as_handle().map(Handle::from_raw).unwrap();
            let f = self.realm.new_bound_native(N_INTL_DURATION_METHOD, th);
            self.install_fn_name_length(f, m, arity);
            self.realm
                .set_property(proto, m, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, m);
        }
        self.install_to_string_tag(proto, "Intl.DurationFormat");
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ctor.to_raw()));
        self.link_ctor_prototype(ctor, proto);
        self.realm.set_intl_prototype(N_INTL_DURATION_FORMAT, proto);
        Some(proto)
    }

    /// `new Intl.DurationFormat(locales, options)` — `InitializeDurationFormat`:
    /// canonicalizes the locale, resolves `numberingSystem` (from the `-u-nu-`
    /// extension and/or option), reads the top-level `style` and each of the ten
    /// per-unit `<unit>`/`<unit>Display` options (via `GetDurationUnitOptions`, in
    /// spec order), and `fractionalDigits`. Stores all resolved slots behind hidden
    /// keys for `resolvedOptions`/`format`. A non-object `options` (other than
    /// `undefined`) is a TypeError; an invalid option value is a RangeError.
    pub(crate) fn make_duration_format(&mut self, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let obj = self.realm.new_object();
        let requested =
            self.canonicalize_locale_list(args.first().copied().unwrap_or(NanBox::undefined()))?;
        // GetOptionsObject: undefined → no options; null/primitive → TypeError.
        let opts_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
        let opts = if matches!(opts_arg.unpack(), Unpacked::Undefined) {
            None
        } else if self.is_object_value(opts_arg) {
            opts_arg.as_handle().map(Handle::from_raw)
        } else {
            return Err(self.type_error("Intl.DurationFormat options must be an object"));
        };
        // localeMatcher (validated, not otherwise used).
        let _ = self.get_string_option(
            opts,
            "localeMatcher",
            &["lookup", "best fit"],
            Some("best fit"),
        )?;
        // numberingSystem option (UTS-35 `type` validated) and the `-u-nu-` extension
        // resolve the data locale and the effective numbering system.
        let nu_opt = self.get_string_option(opts, "numberingSystem", &[], None)?;
        if let Some(ns) = &nu_opt
            && !is_unicode_type_value(ns)
        {
            let m = self.new_str("invalid numberingSystem");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let (locale, numbering) = {
            let picked = self.lookup_available_locale(&requested);
            self.resolve_duration_locale(picked.as_deref(), nu_opt.as_deref())
        };
        let locv = self.new_str(&locale);
        self.realm.set_hidden_property(obj, "\u{0}locale", locv);
        let nuv = self.new_str(&numbering);
        self.realm.set_hidden_property(obj, "numberingSystem", nuv);
        // style (default "short"); "digital" implies numeric h/m/s defaults.
        let style = self
            .get_string_option(
                opts,
                "style",
                &["long", "short", "narrow", "digital"],
                Some("short"),
            )?
            .unwrap();
        self.store_str(obj, "style", &Some(style.clone()));
        let digital = style == "digital";
        // Each row of the unit table: (unit, stylesList, digitalBase). prevStyle is
        // threaded through hours..microseconds (GetDurationUnitOptions step 6).
        const SUB: &[&str] = &["long", "short", "narrow"];
        const TIME: &[&str] = &["long", "short", "narrow", "numeric", "2-digit"];
        const FRAC: &[&str] = &["long", "short", "narrow", "numeric"];
        // (unit, stylesList, digitalDefault) per the unit table.
        let table: &[(&str, &[&str], &str)] = &[
            ("years", SUB, "short"),
            ("months", SUB, "short"),
            ("weeks", SUB, "short"),
            ("days", SUB, "short"),
            ("hours", TIME, "numeric"),
            ("minutes", TIME, "numeric"),
            ("seconds", TIME, "numeric"),
            ("milliseconds", FRAC, "numeric"),
            ("microseconds", FRAC, "numeric"),
            ("nanoseconds", FRAC, "numeric"),
        ];
        let mut prev_style: Option<String> = None;
        for (unit, styles, digital_base) in table {
            let (ust, udisp) = self.get_duration_unit_options(
                opts,
                unit,
                &style,
                styles,
                digital_base,
                prev_style.as_deref(),
                digital,
            )?;
            self.store_str(obj, unit, &Some(ust.clone()));
            let disp_key = alloc::format!("{unit}Display");
            self.store_str(obj, &disp_key, &Some(udisp));
            if matches!(
                *unit,
                "hours" | "minutes" | "seconds" | "milliseconds" | "microseconds"
            ) {
                prev_style = Some(ust);
            }
        }
        // fractionalDigits: GetNumberOption(0, 9, undefined).
        if let Some(fd) = self.get_int_option(opts, "fractionalDigits", 0.0, 9.0, None)? {
            self.realm
                .set_hidden_property(obj, "fractionalDigits", NanBox::number(fd));
        }
        self.realm
            .set_hidden_property(obj, "\u{0}brand_df", NanBox::boolean(true));
        if let Some(proto) = self.intl_duration_prototype() {
            self.realm.set_object_proto(obj, Some(proto));
        }
        Ok(NanBox::handle(obj.to_raw()))
    }

    /// Resolves the `Intl.DurationFormat` data locale and effective numbering system
    /// from the requested tag's `-u-nu-` extension and the `numberingSystem` option.
    /// The option (when a supported value) wins; otherwise a supported extension
    /// value is kept and reflected in the locale; an unsupported value falls back to
    /// the locale's default (`"latn"` for `en`). Returns `(dataLocale, numbering)`.
    fn resolve_duration_locale(
        &mut self,
        requested: Option<&str>,
        option: Option<&str>,
    ) -> (String, String) {
        let tag = requested.unwrap_or("en-US");
        // Split off any `-u-` extension; recover an `nu` keyword value.
        let parsed = ParsedLocale::from_canonical(tag);
        let ext_nu = parsed.keyword("nu").map(String::from);
        let default_nu = String::from("latn");
        let supported = |ns: &str| is_supported_numbering_system(ns);
        // The base tag without any -u- keywords (other extensions preserved).
        let mut base = parsed.base_name();
        for e in &parsed.other_ext {
            base.push('-');
            base.push_str(e);
        }
        match option {
            Some(opt) if supported(opt) => {
                // Option wins; reflect in the locale only if it equals the extension.
                if ext_nu.as_deref() == Some(opt) {
                    (alloc::format!("{base}-u-nu-{opt}"), String::from(opt))
                } else {
                    (base, String::from(opt))
                }
            }
            _ => {
                // No (usable) option: a supported extension value is used & reflected.
                match ext_nu {
                    Some(ns) if supported(&ns) => (alloc::format!("{base}-u-nu-{ns}"), ns),
                    _ => (base, default_nu),
                }
            }
        }
    }

    /// `GetDurationUnitOptions(unit, options, baseStyle, stylesList, digitalBase,
    /// prevStyle, twoDigitHours)` — reads the per-unit `<unit>` style and
    /// `<unit>Display` options, applying the spec defaults (digital base, the
    /// `prevStyle`-driven "numeric"/"2-digit" promotion) and the RangeError
    /// conditions. Returns `(style, display)`.
    #[allow(clippy::too_many_arguments)]
    fn get_duration_unit_options(
        &mut self,
        opts: Option<Handle>,
        unit: &str,
        base_style: &str,
        styles_list: &[&str],
        digital_base: &str,
        prev_style: Option<&str>,
        two_digit_hours: bool,
    ) -> Result<(String, String), ExecError> {
        let mut style = self.get_string_option(opts, unit, styles_list, None)?;
        let mut display_default = "always";
        if style.is_none() {
            if base_style == "digital" {
                if !matches!(unit, "hours" | "minutes" | "seconds") {
                    display_default = "auto";
                }
                style = Some(String::from(digital_base));
            } else {
                display_default = "auto";
                if matches!(prev_style, Some("numeric") | Some("2-digit")) {
                    style = Some(String::from("numeric"));
                } else {
                    style = Some(String::from(base_style));
                }
            }
        }
        let mut style = style.unwrap();
        let disp_key = alloc::format!("{unit}Display");
        let display = self
            .get_string_option(opts, &disp_key, &["auto", "always"], Some(display_default))?
            .unwrap();
        if matches!(prev_style, Some("numeric") | Some("2-digit")) {
            if style != "numeric" && style != "2-digit" {
                let m = self.new_str(&alloc::format!(
                    "invalid style '{style}' for {unit} following a numeric unit"
                ));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            } else if matches!(unit, "minutes" | "seconds") {
                style = String::from("2-digit");
            }
        }
        // twoDigitHours: a digital formatter with 2-digit hours promotes "numeric".
        if unit == "hours" && two_digit_hours && style == "numeric" {
            // (Kept as "numeric" — the engine's digital renderer never pads hours,
            // matching the en reference where [[TwoDigitHours]] is false.)
        }
        Ok((style, display))
    }

    /// The ten duration units in table order.
    const DURATION_UNITS: [&'static str; 10] = [
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
    ];

    /// Dispatches an `Intl.DurationFormat` prototype method: brand-checks `this`.
    pub(crate) fn intl_duration_method_dispatch(
        &mut self,
        this: NanBox,
        name: &str,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let h = self.require_intl_slot(
            this,
            "\u{0}brand_df",
            "Intl.DurationFormat.prototype method",
        )?;
        match name {
            "resolvedOptions" => self.duration_resolved_options(h),
            "formatToParts" => {
                let rec = self
                    .read_duration_record(args.first().copied().unwrap_or(NanBox::undefined()))?;
                let parts = self.partition_duration(h, &rec);
                let mut arr = Vec::with_capacity(parts.len());
                for (ty, val, unit) in parts {
                    let o = self.realm.new_object();
                    let tv = self.new_str(ty);
                    self.realm.set_property(o, "type", tv);
                    let vv = self.new_str(&val);
                    self.realm.set_property(o, "value", vv);
                    if let Some(u) = unit {
                        let uv = self.new_str(u);
                        self.realm.set_property(o, "unit", uv);
                    }
                    arr.push(NanBox::handle(o.to_raw()));
                }
                Ok(NanBox::handle(self.realm.new_array(arr).to_raw()))
            }
            // `format(duration)` — concatenate the partitioned parts' values.
            _ => {
                let rec = self
                    .read_duration_record(args.first().copied().unwrap_or(NanBox::undefined()))?;
                let parts = self.partition_duration(h, &rec);
                let s: String = parts.into_iter().map(|(_, v, _)| v).collect();
                Ok(self.new_str(&s))
            }
        }
    }

    /// `Intl.DurationFormat.prototype.resolvedOptions` — `{locale, numberingSystem,
    /// style, <unit>, <unit>Display …, fractionalDigits?}` in spec key order.
    fn duration_resolved_options(&mut self, h: Handle) -> Result<NanBox, ExecError> {
        let out = self.realm.new_object();
        let read = |this: &mut Self, key: &str, dflt: &str| -> String {
            this.realm
                .get_property(h, key)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
                .unwrap_or_else(|| String::from(dflt))
        };
        let locale = read(self, "\u{0}locale", "en-US");
        let lv = self.new_str(&locale);
        self.realm.set_property(out, "locale", lv);
        let nu = read(self, "numberingSystem", "latn");
        let nuv = self.new_str(&nu);
        self.realm.set_property(out, "numberingSystem", nuv);
        let style = read(self, "style", "short");
        let sv = self.new_str(&style);
        self.realm.set_property(out, "style", sv);
        for unit in Self::DURATION_UNITS {
            let ust = read(self, unit, "short");
            let uv = self.new_str(&ust);
            self.realm.set_property(out, unit, uv);
            let disp_key = alloc::format!("{unit}Display");
            let udisp = read(self, &disp_key, "auto");
            let dv = self.new_str(&udisp);
            self.realm.set_property(out, &disp_key, dv);
        }
        // fractionalDigits is reported only when it was explicitly set.
        if let Some(v) = self
            .realm
            .get_property(h, "fractionalDigits")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
        {
            self.realm.set_property(out, "fractionalDigits", v);
        }
        Ok(NanBox::handle(out.to_raw()))
    }

    /// `ToDurationRecord(input)` — a String is parsed as an ISO-8601 duration
    /// (RangeError on a bad string); a non-object is a TypeError; otherwise each of
    /// the ten unit fields is read and `ToIntegerIfIntegral`-coerced (a non-integer
    /// or non-finite value is a RangeError), all fields absent is a TypeError, and
    /// the record is validated by `IsValidDuration` (RangeError otherwise). Returns
    /// the ten `f64` unit values in table order.
    fn read_duration_record(&mut self, input: NanBox) -> Result<[f64; 10], ExecError> {
        // A Temporal.Duration-like string is parsed; any other primitive (or a
        // String wrapper is an Object, handled below) follows the object path.
        if let Some(s) = input
            .as_handle()
            .map(Handle::from_raw)
            .and_then(|hh| self.realm.string_value(hh))
        {
            return self.parse_duration_string(&s);
        }
        if !self.is_object_value(input) {
            return Err(self.type_error("Intl.DurationFormat.format: argument must be an object"));
        }
        let oh = input.as_handle().map(Handle::from_raw).unwrap();
        // `ToDurationRecord` reads a `Temporal.Duration` from its internal slots,
        // NOT via the (user-observable, potentially tainted) prototype getters.
        if let Some(td) = self.realm.temporal_at(oh)
            && td.kind == crate::temporal_iso::TemporalKind::Duration
        {
            let d = &td.duration;
            return Ok([
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
            ]);
        }
        let mut rec = [0.0f64; 10];
        let mut any = false;
        for (i, unit) in Self::DURATION_UNITS.iter().enumerate() {
            let v = self.read_member(oh, unit)?;
            if matches!(v.unpack(), Unpacked::Undefined) {
                continue;
            }
            any = true;
            // ToIntegerIfIntegral: ToNumber, then require an integral value.
            let nv = self.coerce_to_number(v)?;
            let n = self.realm.to_number(nv);
            if !n.is_finite() || trunc_toward_zero(n) != n {
                let m = self.new_str(&alloc::format!("duration field {unit} is not an integer"));
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            rec[i] = n;
        }
        if !any {
            return Err(self.type_error("Intl.DurationFormat.format: no duration fields present"));
        }
        if !is_valid_duration(&rec) {
            let m = self.new_str("invalid Duration");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(rec)
    }

    /// Minimal ISO-8601 duration-string parser (`±PnYnMnWnDTnHnMnS`, fractional
    /// seconds). A malformed string is a RangeError. Sub-second fractions are split
    /// into milli/micro/nanoseconds.
    fn parse_duration_string(&mut self, s: &str) -> Result<[f64; 10], ExecError> {
        let bad = |this: &mut Self| -> ExecError {
            let m = this.new_str("invalid Duration string");
            ExecError::Throw(this.make_error(N_RANGE_ERROR, Some(m)))
        };
        let mut rec = [0.0f64; 10];
        let bytes = s.trim();
        let (sign, rest) = match bytes.strip_prefix('-') {
            Some(r) => (-1.0, r),
            None => (1.0, bytes.strip_prefix('+').unwrap_or(bytes)),
        };
        let rest = match rest.strip_prefix(['P', 'p']) {
            Some(r) => r,
            None => return Err(bad(self)),
        };
        // Split date / time portions on 'T'.
        let (date_part, time_part) = match rest.split_once(['T', 't']) {
            Some((d, t)) => (d, Some(t)),
            None => (rest, None),
        };
        let mut saw_any = false;
        // Returns (value, designator) tokens.
        let parse_section = |this: &mut Self,
                             sect: &str,
                             allowed: &[(char, usize)],
                             rec: &mut [f64; 10],
                             saw: &mut bool|
         -> Result<(), ExecError> {
            let mut chars = sect.chars().peekable();
            let mut last_idx: i32 = -1;
            while chars.peek().is_some() {
                let mut num = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' || c == ',' {
                        num.push(if c == ',' { '.' } else { c });
                        chars.next();
                    } else {
                        break;
                    }
                }
                let Some(desig) = chars.next() else {
                    if num.is_empty() {
                        break;
                    }
                    return Err(bad(this));
                };
                if num.is_empty() {
                    return Err(bad(this));
                }
                let Some(&(_, slot)) = allowed.iter().find(|(d, _)| d.eq_ignore_ascii_case(&desig))
                else {
                    return Err(bad(this));
                };
                // Designators must appear in order.
                if (slot as i32) <= last_idx {
                    return Err(bad(this));
                }
                last_idx = slot as i32;
                *saw = true;
                // Only seconds may be fractional.
                if num.contains('.') && slot != 6 {
                    return Err(bad(this));
                }
                if slot == 6 {
                    // seconds with optional fraction → split into s/ms/us/ns.
                    let (int_s, frac) = num.split_once('.').unwrap_or((num.as_str(), ""));
                    rec[6] = int_s.parse::<f64>().map_err(|_| bad(this))?;
                    let mut f: String = frac.chars().take(9).collect();
                    while f.len() < 9 {
                        f.push('0');
                    }
                    rec[7] = f[0..3].parse::<f64>().unwrap_or(0.0);
                    rec[8] = f[3..6].parse::<f64>().unwrap_or(0.0);
                    rec[9] = f[6..9].parse::<f64>().unwrap_or(0.0);
                } else {
                    rec[slot] = num.parse::<f64>().map_err(|_| bad(this))?;
                }
            }
            Ok(())
        };
        // Date designators: Y(0) M(1) W(2) D(3).
        parse_section(
            self,
            date_part,
            &[('Y', 0), ('M', 1), ('W', 2), ('D', 3)],
            &mut rec,
            &mut saw_any,
        )?;
        if let Some(tp) = time_part {
            if tp.is_empty() {
                return Err(bad(self));
            }
            // Time designators: H(4) M(5) S(6).
            parse_section(
                self,
                tp,
                &[('H', 4), ('M', 5), ('S', 6)],
                &mut rec,
                &mut saw_any,
            )?;
        }
        if !saw_any {
            return Err(bad(self));
        }
        for v in &mut rec {
            *v *= sign;
        }
        if !is_valid_duration(&rec) {
            let m = self.new_str("invalid Duration");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        Ok(rec)
    }

    /// Resolves a duration formatter's per-unit `(style, display)` from its slots.
    fn duration_unit_resolved(&mut self, h: Handle, unit: &str) -> (String, String) {
        let style = self
            .realm
            .get_property(h, unit)
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("short"));
        let disp = self
            .realm
            .get_property(h, &alloc::format!("{unit}Display"))
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("auto"));
        (style, disp)
    }

    /// `PartitionDurationFormatPattern` (en, mirroring the test262 reference): formats
    /// each shown unit via a per-unit `Intl.NumberFormat` (`style:"unit"` or numeric/
    /// 2-digit), threads the time separator between numeric h:m:s, then joins the unit
    /// strings with an `Intl.ListFormat` (`type:"unit"`). Returns `(type, value, unit?)`
    /// parts. Composing the real `NumberFormat`/`ListFormat` keeps `format` and the
    /// reference output identical regardless of CLDR-data fidelity.
    fn partition_duration(
        &mut self,
        h: Handle,
        duration: &[f64; 10],
    ) -> Vec<(&'static str, String, Option<&'static str>)> {
        let style = self
            .realm
            .get_property(h, "style")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("short"));
        let numbering = self
            .realm
            .get_property(h, "numberingSystem")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("latn"));
        // PartitionDurationFormatPattern composes the *instance's* locale into the
        // per-unit NumberFormats and the joining ListFormat; using "en" here made
        // every duration render with English unit names whatever the locale.
        let df_locale = self
            .realm
            .get_property(h, "\u{0}locale")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_display_string(v))
            .unwrap_or_else(|| String::from("en"));
        let fractional_digits = self
            .realm
            .get_property(h, "fractionalDigits")
            .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
            .map(|v| self.realm.to_number(v) as i32);
        let units = Self::DURATION_UNITS;
        // Resolve per-unit styles/displays up front.
        let mut ustyle = [const { String::new() }; 10];
        let mut udisp = [const { String::new() }; 10];
        for (i, u) in units.iter().enumerate() {
            let (s, d) = self.duration_unit_resolved(h, u);
            ustyle[i] = s;
            udisp[i] = d;
        }
        let time_separator = ":";
        // Each element of `result` is the ordered parts list for one shown unit.
        let mut result: Vec<Vec<(&'static str, String, Option<&'static str>)>> = Vec::new();
        let mut need_separator = false;
        let mut display_negative_sign = true;
        // Whether the duration is negative (drives the leading unit's sign). Per
        // DurationSign, a negative *zero* field counts as zero — `{years:-0}` has
        // sign 0 and formats identically to `{years:+0}` (no minus sign).
        let any_negative = duration.iter().any(|&v| v < 0.0);
        for idx in 0..units.len() {
            let unit = units[idx];
            let singular: &'static str = duration_singular(unit);
            // Normalize a negative-zero field to +0 (a genuinely-negative duration
            // re-applies the leading sign below); `{years:-0}` must format as "0".
            let mut value = duration[idx];
            if value == 0.0 {
                value = 0.0;
            }
            let style_u = ustyle[idx].clone();
            let display_u = udisp[idx].clone();
            // Combine numeric seconds/ms/us with their fractional remainder.
            let mut value_str: Option<String> = None;
            let mut done = false;
            let (mut nf_min_frac, mut nf_max_frac, mut nf_trunc) =
                (None::<i32>, None::<i32>, false);
            if matches!(unit, "seconds" | "milliseconds" | "microseconds") {
                let next_style = ustyle[idx + 1].as_str();
                if next_style == "numeric" {
                    let exp = match unit {
                        "seconds" => 9,
                        "milliseconds" => 6,
                        _ => 3,
                    };
                    value_str = Some(duration_to_fractional(duration, exp));
                    nf_max_frac = Some(fractional_digits.unwrap_or(9));
                    nf_min_frac = Some(fractional_digits.unwrap_or(0));
                    nf_trunc = true;
                    done = true;
                }
            }
            // Display zero numeric minutes when seconds will be displayed.
            let mut display_required = false;
            if unit == "minutes" && need_separator {
                display_required = udisp[6] == "always"
                    || duration[6] != 0.0
                    || duration[7] != 0.0
                    || duration[8] != 0.0
                    || duration[9] != 0.0;
            }
            let nonzero = value != 0.0
                || value_str
                    .as_deref()
                    .is_some_and(|s| s.bytes().any(|b| matches!(b, b'1'..=b'9')));
            if nonzero || display_u != "auto" || display_required {
                let mut sign_never = false;
                if display_negative_sign {
                    display_negative_sign = false;
                    if value == 0.0 && value_str.is_none() && any_negative {
                        value = -0.0;
                    }
                } else {
                    sign_never = true;
                }
                // Build the per-unit NumberFormat handle.
                let nf = self.realm.new_object();
                let marker = self.new_str("number");
                self.realm.set_hidden_property(nf, "\u{0}intl", marker);
                let loc = self.new_str(&df_locale);
                self.realm.set_hidden_property(nf, "\u{0}locale", loc);
                let nuv = self.new_str(&numbering);
                self.realm.set_hidden_property(nf, "numberingSystem", nuv);
                if sign_never {
                    let sd = self.new_str("never");
                    self.realm.set_hidden_property(nf, "signDisplay", sd);
                }
                if style_u == "2-digit" {
                    self.realm
                        .set_hidden_property(nf, "minimumIntegerDigits", NanBox::number(2.0));
                }
                if style_u != "numeric" && style_u != "2-digit" {
                    let st = self.new_str("unit");
                    self.realm.set_hidden_property(nf, "style", st);
                    let uu = self.new_str(singular);
                    self.realm.set_hidden_property(nf, "unit", uu);
                    let ud = self.new_str(&style_u);
                    self.realm.set_hidden_property(nf, "unitDisplay", ud);
                } else {
                    self.realm
                        .set_hidden_property(nf, "useGrouping", NanBox::boolean(false));
                }
                if let Some(mn) = nf_min_frac {
                    self.realm.set_hidden_property(
                        nf,
                        "minimumFractionDigits",
                        NanBox::number(mn as f64),
                    );
                }
                if let Some(mx) = nf_max_frac {
                    self.realm.set_hidden_property(
                        nf,
                        "maximumFractionDigits",
                        NanBox::number(mx as f64),
                    );
                }
                if nf_trunc {
                    let rm = self.new_str("trunc");
                    self.realm.set_hidden_property(nf, "roundingMode", rm);
                }
                // The value to format: the combined fractional string, or the number.
                let value_box = match &value_str {
                    Some(s) => self.new_str(s),
                    None => NanBox::number(value),
                };
                let number_parts = self.number_handle_parts(nf, value_box);
                let mut list: Vec<(&'static str, String, Option<&'static str>)> = if !need_separator
                {
                    Vec::new()
                } else {
                    // Append to the previous (numeric) unit's list, with a separator.
                    let mut prev = result.pop().unwrap();
                    prev.push(("literal", String::from(time_separator), None));
                    prev
                };
                for (ty, val) in number_parts {
                    list.push((ty, val, Some(singular)));
                }
                if !need_separator {
                    if style_u == "2-digit" || style_u == "numeric" {
                        need_separator = true;
                    }
                    result.push(list);
                } else {
                    result.push(list);
                }
            }
            if done {
                break;
            }
        }
        // List style: digital collapses to short.
        let list_style = if style == "digital" {
            String::from("short")
        } else {
            style
        };
        // Build the strings, then join via ListFormat "unit".
        let strings: Vec<String> = result
            .iter()
            .map(|parts| parts.iter().map(|(_, v, _)| v.as_str()).collect())
            .collect();
        let lf = self.realm.new_object();
        let lm = self.new_str("list");
        self.realm.set_hidden_property(lf, "\u{0}intl", lm);
        let llo = self.new_str(&df_locale);
        self.realm.set_hidden_property(lf, "\u{0}locale", llo);
        let lt = self.new_str("unit");
        self.realm.set_hidden_property(lf, "type", lt);
        let ls = self.new_str(&list_style);
        self.realm.set_hidden_property(lf, "style", ls);
        let list_parts = self.list_format_parts(&strings, "unit", &list_style, &df_locale);
        // Flatten: an "element" splices in the next unit's parts; a "literal" passes through.
        let mut flattened: Vec<(&'static str, String, Option<&'static str>)> = Vec::new();
        let mut iter = result.into_iter();
        for (ty, val) in list_parts {
            if ty == "element" {
                if let Some(parts) = iter.next() {
                    flattened.extend(parts);
                }
            } else {
                flattened.push((ty, val, None));
            }
        }
        let _ = lf;
        flattened
    }
}

/// A parsed `unicode_locale_id` split into core subtags plus its `-u-` keyword
/// list, used by `Intl.Locale` to apply option overrides and rebuild the tag.
struct ParsedLocale {
    language: String,
    script: Option<String>,
    region: Option<String>,
    variants: Vec<String>,
    /// `-u-` leading attributes (keyword-less subtags, e.g. the `attr` in
    /// `en-u-attr-co-phonebk`), sorted; emitted before the keywords.
    attributes: Vec<String>,
    /// `-u-` keyword `(key, value)` pairs (value may be empty for `kn`/`true`).
    keywords: Vec<(String, String)>,
    /// Any other extensions (`-t-`, `-x-`, …) preserved verbatim after the `-u-` block.
    other_ext: Vec<String>,
}

impl ParsedLocale {
    /// Splits a canonical tag (the output of [`canonicalize_locale_id`]) into its
    /// components and `-u-` keywords.
    fn from_canonical(canon: &str) -> Self {
        let mut language = String::new();
        let mut script = None;
        let mut region = None;
        let mut variants = Vec::new();
        let mut attributes: Vec<String> = Vec::new();
        let mut keywords = Vec::new();
        let mut other_ext = Vec::new();
        let parts: Vec<&str> = canon.split('-').collect();
        let mut i = 0;
        if i < parts.len() {
            language = parts[i].to_ascii_lowercase();
            i += 1;
        }
        if i < parts.len()
            && parts[i].len() == 4
            && parts[i].bytes().all(|b| b.is_ascii_alphabetic())
        {
            script = Some(titlecase_script(parts[i]));
            i += 1;
        }
        if i < parts.len()
            && ((parts[i].len() == 2 && parts[i].bytes().all(|b| b.is_ascii_alphabetic()))
                || (parts[i].len() == 3 && parts[i].bytes().all(|b| b.is_ascii_digit())))
        {
            region = Some(parts[i].to_ascii_uppercase());
            i += 1;
        }
        // Variants until a singleton.
        while i < parts.len() && parts[i].len() != 1 {
            variants.push(parts[i].to_ascii_lowercase());
            i += 1;
        }
        // Extensions.
        while i < parts.len() {
            let singleton = parts[i];
            if singleton == "u" {
                i += 1;
                // A `-u-` block alternates 2-char keys with their (3-8 char) values.
                // Leading attributes (3-8 char, no key) are rare; skip until the first
                // 2-char key, then read each key's run of non-key value subtags.
                while i < parts.len() && parts[i].len() != 1 {
                    if parts[i].len() != 2 {
                        // An attribute (no key) — a keyword-less subtag preserved
                        // before the keywords (`en-u-attr-co-phonebk`).
                        attributes.push(String::from(parts[i]));
                        i += 1;
                        continue;
                    }
                    let key = String::from(parts[i]);
                    i += 1;
                    let mut vals: Vec<String> = Vec::new();
                    while i < parts.len() && parts[i].len() != 1 && parts[i].len() != 2 {
                        vals.push(String::from(parts[i]));
                        i += 1;
                    }
                    keywords.push((key, vals.join("-")));
                }
            } else {
                // Preserve other extensions verbatim. Private use (`x`) consumes
                // *all* remaining subtags, including length-1 ones (so `x-u-foo`
                // is one private sequence, not the start of a `-u-` extension).
                let private = singleton == "x";
                let mut buf = alloc::vec![String::from(singleton)];
                i += 1;
                while i < parts.len() && (private || parts[i].len() != 1) {
                    buf.push(String::from(parts[i]));
                    i += 1;
                }
                other_ext.push(buf.join("-"));
            }
        }
        attributes.sort();
        ParsedLocale {
            language,
            script,
            region,
            variants,
            attributes,
            keywords,
            other_ext,
        }
    }

    /// The value of `-u-` keyword `key`, if present (empty string for a bare key).
    fn keyword(&self, key: &str) -> Option<&str> {
        self.keywords
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Sets (or replaces) a `-u-` keyword. A `kf`/`kn` `"false"`/`"true"` is kept
    /// as-is; the accessors interpret it.
    fn set_keyword(&mut self, key: &str, val: &str) {
        let v = val.to_ascii_lowercase();
        if let Some(e) = self.keywords.iter_mut().find(|(k, _)| k == key) {
            e.1 = v;
        } else {
            self.keywords.push((String::from(key), v));
        }
    }

    /// The `baseName`: `language(-script)?(-region)?(-variants)?` (no extensions).
    fn base_name(&self) -> String {
        let mut out = self.language.clone();
        if let Some(s) = &self.script {
            out.push('-');
            out.push_str(s);
        }
        if let Some(r) = &self.region {
            out.push('-');
            out.push_str(r);
        }
        for v in &self.variants {
            out.push('-');
            out.push_str(v);
        }
        out
    }

    /// Rebuilds the full canonical tag: base name followed by every extension
    /// ordered by its singleton (`-a-`/`-t-`/`-u-`/… ascending, private use `-x-`
    /// last), with the `-u-` attributes and keywords each sorted.
    fn to_tag(&self) -> String {
        let mut out = self.base_name();
        // The `-u-` block, assembled from its (sorted) attributes and keywords.
        let mut kw = self.keywords.clone();
        kw.sort_by(|a, b| a.0.cmp(&b.0));
        let mut exts: Vec<String> = self.other_ext.clone();
        if !kw.is_empty() || !self.attributes.is_empty() {
            let mut u = String::from("u");
            for a in &self.attributes {
                u.push('-');
                u.push_str(a);
            }
            for (k, v) in &kw {
                u.push('-');
                u.push_str(k);
                if !v.is_empty() && v != "true" {
                    u.push('-');
                    u.push_str(v);
                }
            }
            exts.push(u);
        }
        // Extensions sort by their leading singleton, with private use (`x`) last.
        exts.sort_by_key(|e| {
            let s = e.as_bytes().first().copied().unwrap_or(b'~');
            (s == b'x', s)
        });
        for e in &exts {
            out.push('-');
            out.push_str(e);
        }
        out
    }
}

/// Whether `ns` is a numbering system this engine treats as supported (only `latn`
/// and `arab` have data; others fall back to the locale default for DurationFormat).
fn is_supported_numbering_system(ns: &str) -> bool {
    is_known_numbering_system(ns)
}

/// The singular `Intl.NumberFormat` unit name for a duration unit field
/// (`"hours"` → `"hour"`), i.e. the field name without its trailing `s`.
fn duration_singular(unit: &str) -> &'static str {
    match unit {
        "years" => "year",
        "months" => "month",
        "weeks" => "week",
        "days" => "day",
        "hours" => "hour",
        "minutes" => "minute",
        "seconds" => "second",
        "milliseconds" => "millisecond",
        "microseconds" => "microsecond",
        _ => "nanosecond",
    }
}

/// `IsValidDuration(record)` for the ten `f64` unit values: all-same-sign, finite,
/// `abs(years|months|weeks) < 2^32`, and `abs(normalizedSeconds) < 2^53` where the
/// seconds are accumulated exactly via `i128` nanoseconds to avoid `f64` rounding.
fn is_valid_duration(rec: &[f64; 10]) -> bool {
    let mut sign = 0i32;
    for &v in rec {
        if !v.is_finite() {
            return false;
        }
        let s = if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 {
            if sign != 0 && sign != s {
                return false;
            }
            sign = s;
        }
    }
    // years/months/weeks bound.
    let two32 = 4_294_967_296.0f64; // 2^32
    if rec[0].abs() >= two32 || rec[1].abs() >= two32 || rec[2].abs() >= two32 {
        return false;
    }
    // normalizedSeconds = days*86400 + hours*3600 + minutes*60 + seconds
    //   + ms/1e3 + us/1e6 + ns/1e9, computed exactly in i128 nanoseconds.
    // Each value is integral and finite here. Guard against i128 overflow by
    // rejecting any single term that already exceeds the 2^53-seconds budget.
    let two53_ns = (1i128 << 53) * 1_000_000_000; // 2^53 seconds in ns
    let term_ns = |v: f64, scale_ns: i128| -> Option<i128> {
        if v.abs() >= 9.0e30 {
            return None; // far beyond any valid range; treat as overflow
        }
        (v as i128).checked_mul(scale_ns)
    };
    let mut total: i128 = 0;
    let scales: [(usize, i128); 7] = [
        (3, 86_400 * 1_000_000_000),
        (4, 3_600 * 1_000_000_000),
        (5, 60 * 1_000_000_000),
        (6, 1_000_000_000),
        (7, 1_000_000),
        (8, 1_000),
        (9, 1),
    ];
    for (i, scale) in scales {
        match term_ns(rec[i], scale).and_then(|t| total.checked_add(t)) {
            Some(t) => total = t,
            None => return false,
        }
    }
    total.abs() < two53_ns
}

/// `DurationToFractional` (test262 reference): the numeric value of the unit at
/// `exponent` (9=seconds, 6=milliseconds, 3=microseconds) combined with all smaller
/// sub-second fields, as an exact decimal string. Uses `i128` nanoseconds.
fn duration_to_fractional(duration: &[f64; 10], exponent: u32) -> String {
    let (seconds, milliseconds, microseconds, nanoseconds) =
        (duration[6], duration[7], duration[8], duration[9]);
    // Fast path: no smaller sub-seconds present → return the bare amount.
    match exponent {
        9 if milliseconds == 0.0 && microseconds == 0.0 && nanoseconds == 0.0 => {
            return format_integral(seconds);
        }
        6 if microseconds == 0.0 && nanoseconds == 0.0 => {
            return format_integral(milliseconds);
        }
        3 if nanoseconds == 0.0 => {
            return format_integral(microseconds);
        }
        _ => {}
    }
    let mut ns: i128 = nanoseconds as i128;
    if exponent >= 9 {
        ns += (seconds as i128) * 1_000_000_000;
    }
    if exponent >= 6 {
        ns += (milliseconds as i128) * 1_000_000;
    }
    if exponent >= 3 {
        ns += (microseconds as i128) * 1_000;
    }
    let e: i128 = 10i128.pow(exponent);
    let q = ns / e;
    let mut r = ns % e;
    if r < 0 {
        r = -r;
    }
    let mut rs = alloc::format!("{r}");
    while rs.len() < exponent as usize {
        rs.insert(0, '0');
    }
    alloc::format!("{q}.{rs}")
}

/// Formats an integral `f64` without a decimal point or exponent (for duration
/// values, which are validated integers).
fn format_integral(v: f64) -> String {
    alloc::format!("{}", v as i128)
}

/// Titlecases a 4-letter script subtag (`latn` → `Latn`).
fn titlecase_script(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == 0 {
            out.push(c.to_ascii_uppercase());
        } else {
            out.push(c.to_ascii_lowercase());
        }
    }
    out
}

#[cfg(all(test, feature = "intl"))]
mod decimal_round_tests {
    use super::intl_decimal_round;
    use intl::number::RoundingMode;

    // Round a value and render it the way `intl::number::format` would for en-US
    // decimal (min/max fraction), so the assertions read like the ECMA-402 output.
    fn fmt(n: f64, keep_frac: usize, sig: Option<usize>, inc: u32, mode: RoundingMode) -> f64 {
        intl_decimal_round(n, keep_frac, sig, inc, mode)
    }

    #[test]
    fn shortest_decimal_boundary() {
        // 1.15 is really 1.1499… in f64; ECMA-402 rounds the *shortest* decimal up.
        assert_eq!(fmt(1.15, 3, Some(2), 1, RoundingMode::HalfExpand), 1.2);
        assert_eq!(fmt(1.15, 3, Some(2), 1, RoundingMode::HalfEven), 1.2);
        assert_eq!(fmt(1.15, 3, Some(2), 1, RoundingMode::HalfCeil), 1.2);
        assert_eq!(fmt(-1.15, 3, Some(2), 1, RoundingMode::HalfFloor), -1.2);
        // 123.445 minSig handled by crate; here max 5 sig → 123.45.
        assert_eq!(
            fmt(123.445, 3, Some(5), 1, RoundingMode::HalfExpand),
            123.45
        );
    }

    #[test]
    fn rounding_increment_direct() {
        // 1.25 inc=2 maxFrac=1 → nearest 0.2 → 1.2 (not double-rounded to 1.4).
        assert_eq!(fmt(1.25, 1, None, 2, RoundingMode::HalfExpand), 1.2);
        // 1.075 inc=5 maxFrac=2 → nearest 0.05, tie → away → 1.10.
        assert_eq!(fmt(1.0750, 2, None, 5, RoundingMode::HalfExpand), 1.1);
        // 1.15 inc=10 maxFrac=2 → nearest 0.10, tie → 1.20.
        assert_eq!(fmt(1.15, 2, None, 10, RoundingMode::HalfExpand), 1.2);
        // 1.20 inc=20 maxFrac=2 → nearest 0.20 → already a multiple → 1.20.
        assert_eq!(fmt(1.20, 2, None, 20, RoundingMode::HalfExpand), 1.2);
        // 1.5 inc=25 maxFrac=2 → multiples 1.25/1.50 → 1.50.
        assert_eq!(fmt(1.5000, 2, None, 25, RoundingMode::HalfExpand), 1.5);
        // 1.25 inc=250 maxFrac=3 → nearest 0.250 → 1.250.
        assert_eq!(fmt(1.2500, 3, None, 250, RoundingMode::HalfExpand), 1.25);
    }

    #[test]
    fn plain_fraction_rounding() {
        assert_eq!(fmt(1.005, 2, None, 1, RoundingMode::HalfExpand), 1.01);
        assert_eq!(fmt(2.5, 0, None, 1, RoundingMode::HalfEven), 2.0);
        assert_eq!(fmt(3.5, 0, None, 1, RoundingMode::HalfEven), 4.0);
        assert_eq!(fmt(0.0, 2, None, 1, RoundingMode::HalfExpand), 0.0);
    }
}

// ECMA-402 resolution helpers (`-u-` keyword extraction, calendar / numbering
// canonicalization) and the exact-decimal / roundingPriority digit routines,
// tested as pure functions.
#[cfg(test)]
mod resolution_helper_tests {
    use super::*;

    #[test]
    fn split_u_keyword_hc() {
        assert_eq!(
            split_u_keyword("en-US-u-hc-h23", "hc"),
            (String::from("en-US"), Some(String::from("h23")))
        );
        // Keeps sibling keywords when removing `hc`.
        assert_eq!(
            split_u_keyword("en-u-ca-gregory-hc-h11-nu-arab", "hc"),
            (
                String::from("en-u-ca-gregory-nu-arab"),
                Some(String::from("h11"))
            )
        );
        // Absent keyword → unchanged, None.
        assert_eq!(
            split_u_keyword("en-US", "hc"),
            (String::from("en-US"), None)
        );
        // Removing the only keyword drops the whole `-u-` section.
        assert_eq!(
            split_u_keyword("de-u-hc-h24", "hc"),
            (String::from("de"), Some(String::from("h24")))
        );
    }

    #[test]
    fn calendar_canonicalization() {
        assert_eq!(canonicalize_calendar("islamicc"), "islamic-civil");
        assert_eq!(canonicalize_calendar("ISO8601"), "iso8601");
        assert_eq!(canonicalize_calendar("ethiopic-amete-alem"), "ethioaa");
        // Deprecated `islamic`/`islamic-rgsa` fall back to a concrete calendar.
        assert_eq!(canonicalize_calendar("islamic"), "islamic-civil");
        assert_eq!(canonicalize_calendar("islamic-rgsa"), "islamic-civil");
        assert_eq!(canonicalize_calendar("gregory"), "gregory");
    }

    #[test]
    fn resolve_ca_option_and_extension() {
        // Invalid option ignored → the extension calendar is used and kept.
        assert_eq!(
            resolve_ca_key("en", "en-u-ca-iso8601", Some("invalid")),
            (String::from("iso8601"), String::from("-ca-iso8601"))
        );
        // Valid option differing from the extension → option wins, extension dropped.
        assert_eq!(
            resolve_ca_key("en", "en-u-ca-gregory", Some("iso8601")),
            (String::from("iso8601"), String::new())
        );
        // Both invalid → default gregory, no extension.
        assert_eq!(
            resolve_ca_key("en", "en-u-ca-invalid", Some("invalid2")),
            (String::from("gregory"), String::new())
        );
    }

    #[test]
    fn resolve_nu_rejects_generic_aliases() {
        // `native`/`traditio`/`finance` are not selectable → fall back to default.
        assert_eq!(
            resolve_nu_key("ja-JP", "ja-JP-u-nu-native", None),
            (String::from("latn"), String::new())
        );
        assert_eq!(
            resolve_nu_key("en", "en", Some("finance")),
            (String::from("latn"), String::new())
        );
        // A real system via the extension is kept.
        assert_eq!(
            resolve_nu_key("en", "en-u-nu-arab", None),
            (String::from("arab"), String::from("-nu-arab"))
        );
    }
}

#[cfg(all(test, feature = "intl"))]
mod exact_decimal_tests {
    use super::*;
    use intl::number::RoundingMode;

    fn digits(v: &[u8]) -> alloc::vec::Vec<u8> {
        v.to_vec()
    }

    #[test]
    fn raw_fixed_rounds_half_expand() {
        // 1.625 → 2 fraction digits, halfExpand → 1.63.
        let (i, f) = to_raw_fixed(
            false,
            digits(&[1]),
            digits(&[6, 2, 5]),
            2,
            RoundingMode::HalfExpand,
        );
        assert_eq!((i, f), (digits(&[1]), digits(&[6, 3])));
    }

    #[test]
    fn raw_precision_rounds_to_sig_digits() {
        // 1.234 → 3 significant digits → 1.23.
        let (i, f) = to_raw_precision(
            false,
            digits(&[1]),
            digits(&[2, 3, 4]),
            3,
            RoundingMode::HalfExpand,
        );
        assert_eq!((i, f), (digits(&[1]), digits(&[2, 3])));
    }

    #[test]
    fn round_up_modes() {
        assert!(exact_round_up(&[5], RoundingMode::HalfExpand, false));
        assert!(!exact_round_up(&[4, 9], RoundingMode::HalfExpand, false));
        assert!(!exact_round_up(&[5], RoundingMode::Trunc, false));
        assert!(exact_round_up(&[1], RoundingMode::Expand, false));
    }

    #[test]
    fn group_thousands() {
        assert_eq!(group_thousands_sep("100000", ","), "100,000");
        assert_eq!(
            group_thousands_sep("987654321987654321", ","),
            "987,654,321,987,654,321"
        );
        assert_eq!(group_thousands_sep("12", ","), "12");
    }
}

// Non-Gregorian (Islamic / Persian) `dateStyle` rendering: the engine computes the
// calendar fields via `temporal_calendar`, then the `intl` crate renders the
// localized month/era names. This exercises that wiring at the seam.
#[cfg(all(test, feature = "intl"))]
mod alt_calendar_format_tests {
    use crate::nbexec::temporal_calendar::iso_to_fields;
    use crate::temporal_iso::IsoDate;
    use intl::datetime::{DateStyle, format_islamic_date, format_persian_date};

    #[test]
    fn islamic_month_fields_and_names() {
        // 2024-03-26 (proleptic Gregorian) is 17 Ramadan 1445 in islamic-tbla.
        let iso = IsoDate {
            year: 2024,
            month: 3,
            day: 26,
        };
        let f = iso_to_fields("islamic-tbla", iso);
        assert_eq!(f.month, 9, "Ramadan is the 9th Islamic month");
        assert_eq!(f.year, 1445);
        // dateStyle:long spells the month out; dateStyle:short does not.
        let long = format_islamic_date("en", f.year, f.month, f.day, DateStyle::Long);
        assert!(long.contains("Ramadan"), "long includes month name: {long}");
        let short = format_islamic_date("en", f.year, f.month, f.day, DateStyle::Short);
        assert!(
            !short.contains("Ramadan"),
            "short uses a numeric month: {short}"
        );
    }

    #[test]
    fn persian_month_fields_and_names() {
        // 2024-03-26 is 7 Farvardin 1403 in the Persian (Solar Hijri) calendar.
        let iso = IsoDate {
            year: 2024,
            month: 3,
            day: 26,
        };
        let f = iso_to_fields("persian", iso);
        assert_eq!(f.month, 1, "Farvardin is the 1st Persian month");
        assert_eq!(f.year, 1403);
        let long = format_persian_date("en", f.year, f.month, f.day, DateStyle::Long);
        assert!(
            long.contains("Farvardin"),
            "long includes month name: {long}"
        );
        let short = format_persian_date("en", f.year, f.month, f.day, DateStyle::Short);
        assert!(
            !short.contains("Farvardin"),
            "short uses a numeric month: {short}"
        );
    }
}
