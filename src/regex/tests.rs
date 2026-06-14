//! Regex engine tests.

use super::Regex;

fn re(pattern: &str, flags: &str) -> Regex {
    Regex::new(pattern, flags).expect("compile ok")
}

#[test]
fn literals_and_anchors() {
    assert!(re("abc", "").is_match("xxabcyy"));
    assert!(!re("abc", "").is_match("ab c"));
    assert!(re("^abc$", "").is_match("abc"));
    assert!(!re("^abc$", "").is_match("abcd"));
    assert_eq!(re("abc", "").find_from("xxabc", 0), Some((2, 5)));
}

#[test]
fn unicode_and_hex_escapes() {
    // `\uHHHH` and `\xHH` resolve to the code point.
    assert!(re(r"A", "").is_match("A"));
    assert!(re(r"\x41", "").is_match("A"));
    assert!(re(r"σ", "").is_match("\u{03c3}"));
    assert!(!re(r"A", "").is_match("B"));
    // `\uHHHH` 4-digit form (the pattern contains a backslash-u escape).
    assert!(re("\\u0041", "").is_match("A"));
    assert!(re("\\u03c3", "").is_match("\u{03c3}"));
    assert!(!re("\\u0041", "").is_match("B"));
    // `\u{…}` code-point form (supplementary planes).
    assert!(re(r"\u{1F600}", "").is_match("\u{1F600}"));
    // Inside a character class.
    assert!(re(r"[A-Z]+", "").is_match("HELLO"));
    assert!(re(r"[\x61\x62]", "").is_match("b"));
    // `\t` via `\x09`.
    assert!(re(r"\x09", "").is_match("a\tb"));
    // A malformed escape is a compile error.
    assert!(Regex::new(r"\u00", "").is_err());
}

#[test]
fn dot_and_classes() {
    assert!(re("a.c", "").is_match("axc"));
    assert!(!re("a.c", "").is_match("a\nc")); // `.` excludes newline
    assert!(re("a.c", "s").is_match("a\nc")); // dotall
    assert!(re("[abc]+", "").is_match("cab"));
    assert!(re("[a-z]+", "").is_match("hello"));
    assert!(!re("[a-z]+", "").is_match("123"));
    assert!(re("[^0-9]", "").is_match("a"));
    assert!(!re("[^0-9]", "").is_match("5"));
    assert!(re(r"\d{3}", "").is_match("a123b"));
    assert!(re(r"\w+", "").is_match("foo_bar"));
    assert!(re(r"\s", "").is_match("a b"));
}

#[test]
fn quantifiers() {
    assert!(re("a*", "").is_match(""));
    assert!(re("ab+c", "").is_match("abbbc"));
    assert!(!re("ab+c", "").is_match("ac"));
    assert!(re("colou?r", "").is_match("color"));
    assert!(re("colou?r", "").is_match("colour"));
    assert!(re("a{2,4}", "").is_match("aaa"));
    assert!(!re("^a{2,4}$", "").is_match("a"));
    assert!(!re("^a{2,4}$", "").is_match("aaaaa"));
    // Greedy vs lazy.
    assert_eq!(
        re("a.*b", "").captures_from("axxbxxb", 0).unwrap().whole(),
        (0, 7)
    );
    assert_eq!(
        re("a.*?b", "").captures_from("axxbxxb", 0).unwrap().whole(),
        (0, 4)
    );
}

#[test]
fn groups_and_alternation() {
    assert!(re("cat|dog", "").is_match("hotdog"));
    assert!(!re("cat|dog", "").is_match("fish"));
    let caps = re(r"(\d+)-(\d+)", "")
        .captures_from("x 12-34 y", 0)
        .unwrap();
    assert_eq!(caps.group(1), Some((2, 4)));
    assert_eq!(caps.group(2), Some((5, 7)));
    // Non-capturing group still groups for quantification.
    assert!(re("(?:ab)+", "").is_match("ababab"));
    assert_eq!(re("(?:ab)+", "").group_count(), 0);
    assert_eq!(re("(a)(b)", "").group_count(), 2);
}

#[test]
fn word_boundaries() {
    assert!(re(r"\bword\b", "").is_match("a word here"));
    assert!(!re(r"\bword\b", "").is_match("wordy"));
    assert!(re(r"\Bord", "").is_match("word"));
}

#[test]
fn case_insensitive() {
    assert!(re("hello", "i").is_match("HELLO"));
    assert!(re("[a-z]+", "i").is_match("ABC"));
    assert!(!re("hello", "").is_match("HELLO"));
}

#[test]
fn multiline() {
    assert!(re("^bar", "m").is_match("foo\nbar"));
    assert!(!re("^bar", "").is_match("foo\nbar"));
}

#[test]
fn replace() {
    assert_eq!(re("o", "g").replace("foo boo", "0"), "f00 b00");
    assert_eq!(re("o", "").replace("foo", "0"), "f0o"); // first only
    assert_eq!(
        re(r"(\w+)@(\w+)", "").replace("user@host", "$2.$1"),
        "host.user"
    );
    assert_eq!(re(r"\d+", "g").replace("a1b22c333", "#"), "a#b#c#");
}

#[test]
fn lookaround_backref_named() {
    // Lookahead / negative lookahead.
    assert_eq!(re("foo(?=bar)", "").find_from("foobar", 0), Some((0, 3)));
    assert!(!re("foo(?=bar)", "").is_match("foobaz"));
    assert!(re("foo(?!bar)", "").is_match("foobaz"));
    // Lookbehind / negative lookbehind.
    assert_eq!(re("(?<=\\$)\\d+", "").find_from("$100", 0), Some((1, 4)));
    assert!(re("(?<!\\$)\\d+", "").is_match("100"));
    // Backreference.
    assert!(re("(ab)\\1", "").is_match("abab"));
    assert!(!re("(ab)\\1", "").is_match("abcd"));
    // Named group exposes its index.
    let r = re("(?<year>\\d{4})", "");
    assert_eq!(r.group_names(), &[(1, alloc::string::String::from("year"))]);
}

#[test]
fn unicode_property_escapes() {
    assert!(re("\\p{L}", "").is_match("a"));
    assert!(!re("\\p{L}", "").is_match("5"));
    assert!(re("\\p{L}", "").is_match("Ω")); // Unicode-aware, not just ASCII
    assert!(re("^\\p{N}+$", "").is_match("123"));
    assert!(re("\\p{Lu}", "").is_match("A"));
    assert!(!re("\\p{Lu}", "").is_match("a"));
    assert!(re("\\P{L}", "").is_match("5")); // negated
    assert!(re("^[\\p{L}\\p{N}]+$", "").is_match("abc123")); // in a class
    assert!(Regex::new("\\p{Nonsense}", "").is_err()); // unknown property
}

#[test]
fn sticky_flag() {
    // Sticky must match at exactly the start position.
    assert!(re("\\d", "y").find_from("1a", 0).is_some());
    assert!(re("\\d", "y").find_from("a1", 0).is_none());
    // Non-sticky scans forward.
    assert!(re("\\d", "").find_from("a1", 0).is_some());
    // Sticky from a later start matches only there.
    assert_eq!(re("\\d", "y").find_from("a1", 1), Some((1, 2)));
    assert!(re("abc", "y").find_from("xabc", 0).is_none());
}

#[test]
fn errors() {
    assert!(Regex::new("(unterminated", "").is_err());
    assert!(Regex::new("[abc", "").is_err());
    assert!(Regex::new("a", "z").is_err()); // unknown flag
    assert!(Regex::new("*abc", "").is_err()); // nothing to repeat
}

#[test]
fn redos_catastrophic_terminates() {
    let subject: alloc::string::String = "a".repeat(40) + "!";
    assert!(!re("(a+)+$", "").is_match(&subject));
}

#[test]
fn redos_linear_depth_terminates() {
    let subject: alloc::string::String = "a".repeat(200_000);
    assert!(re("a*", "").is_match(&subject));
    assert_eq!(
        re("a+", "").captures_from(&subject, 0).unwrap().whole().1,
        200_000
    );
}

#[test]
fn redos_zero_width_terminates() {
    assert!(re("()*", "").is_match("abc"));
    assert!(re("(a*)*", "").is_match("aaa"));
    assert!(re("(a*)*", "").is_match(""));
    assert!(re("(|a)*", "").is_match("aa"));
}

#[test]
fn compile_blowup_rejected() {
    assert!(Regex::new("a{99999999999}", "").is_err());
    assert!(Regex::new("a{5,2}", "").is_err());
    assert!(Regex::new("(a{1000}){1000}", "").is_err());
    assert!(Regex::new("a{100}", "").is_ok());
    assert!(Regex::new("a{2,4}", "").is_ok());
}

// --- UTF-16 code-unit API (`*_in_u16`) ---

/// Encodes a `&str` to UTF-16 code units for the u16 entry points.
fn u16s(s: &str) -> alloc::vec::Vec<u16> {
    s.encode_utf16().collect()
}

#[test]
fn u16_dot_non_unicode_matches_one_code_unit() {
    // "😀" is U+1F600 → two code units. Without `u`, `.` matches one code unit,
    // so the first match is length 1 and there are two matches over the string.
    let units = u16s("😀");
    assert_eq!(units.len(), 2);
    let r = re(".", "");
    let m1 = r.find_in_u16(&units, 0).unwrap();
    assert_eq!(m1, (0, 1));
    let m2 = r.find_in_u16(&units, 1).unwrap();
    assert_eq!(m2, (1, 2));
    assert!(r.find_in_u16(&units, 2).is_none());
}

#[test]
fn u16_dot_unicode_matches_astral_as_one() {
    // With `u`, `.` matches the whole astral character (a surrogate pair) as one
    // code point, but the reported span is in code-unit indices (0..2).
    let units = u16s("😀");
    let r = re(".", "u");
    let m = r.find_in_u16(&units, 0).unwrap();
    assert_eq!(m, (0, 2));
    // Only one match: after consuming both units we're at the end.
    assert!(r.find_in_u16(&units, 2).is_none());
}

#[test]
fn u16_lone_surrogate_matches() {
    // A lone high surrogate (0xD83D) is a matchable code unit in both modes.
    let units: alloc::vec::Vec<u16> = alloc::vec![0xD83D];
    assert_eq!(re(".", "").find_in_u16(&units, 0), Some((0, 1)));
    assert_eq!(re(".", "u").find_in_u16(&units, 0), Some((0, 1)));
    // A class can match the specific lone surrogate via a `\u` escape.
    let r = re(r"\uD83D", "");
    assert_eq!(r.find_in_u16(&units, 0), Some((0, 1)));
}

#[test]
fn u16_unicode_escape_astral_in_u_mode() {
    // `\u{1F600}` in `u` mode matches the astral character as one code point.
    let units = u16s("😀");
    let r = re(r"\u{1F600}", "u");
    assert_eq!(r.find_in_u16(&units, 0), Some((0, 2)));
    // The non-u engine matches it via the surrogate-pair code units too.
    let r2 = re(r"\u{1F600}", "");
    assert_eq!(r2.find_in_u16(&units, 0), Some((0, 2)));
}

#[test]
fn u16_capture_indices_are_code_unit_based() {
    // "x😀y" → units: x(1) hi(1) lo(1) y(1) = indices 0,1,2,3.
    let units = u16s("x😀y");
    assert_eq!(units.len(), 4);
    // Capture the astral char in u mode; its span is code units 1..3.
    let r = re(r"x(.)y", "u");
    let caps = r.captures_in_u16(&units, 0).unwrap();
    assert_eq!(caps.whole(), (0, 4));
    assert_eq!(caps.group(1), Some((1, 3)));
}

#[test]
fn u16_astral_quantifier_unicode() {
    // `.+` in u mode over two astral chars consumes 4 code units.
    let units = u16s("😀😁");
    assert_eq!(units.len(), 4);
    assert_eq!(re(".+", "u").find_in_u16(&units, 0), Some((0, 4)));
    // Astral class range works in u mode.
    let r = re(r"[\u{1F600}-\u{1F610}]+", "u");
    assert_eq!(r.find_in_u16(&units, 0), Some((0, 4)));
}

#[test]
fn u16_backtracking_bomb_terminates() {
    // The step budget still bounds a catastrophic pattern over the u16 API.
    let subject: alloc::string::String = "a".repeat(40) + "!";
    let units = u16s(&subject);
    assert!(re("(a+)+$", "").find_in_u16(&units, 0).is_none());
}

#[test]
fn parser_deep_nesting_rejected() {
    let pat: alloc::string::String = "(".repeat(100_000) + "a" + &")".repeat(100_000);
    assert!(Regex::new(&pat, "").is_err());
    let ok: alloc::string::String = "(".repeat(50) + "a" + &")".repeat(50);
    assert!(Regex::new(&ok, "").is_ok());
}
