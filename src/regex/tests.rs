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
    // Without the `u` flag, an ill-formed `\u` is a valid AnnexB IdentityEscape:
    // `\u00` matches the literal `u00` (`\u` → `u`, then `00`).
    assert!(re(r"\u00", "").is_match("u00"));
    // With the `u` flag it IS a compile error (strict unicode mode).
    assert!(Regex::new(r"\u00", "u").is_err());
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
    // Property escapes require the `u` flag. The `&str`/`&[char]` adapters always
    // run the scalar (unicode-mode) program, so these match as property escapes.
    assert!(re("\\p{L}", "u").is_match("a"));
    assert!(!re("\\p{L}", "u").is_match("5"));
    assert!(re("\\p{L}", "u").is_match("Ω")); // Unicode-aware, not just ASCII
    assert!(re("^\\p{N}+$", "u").is_match("123"));
    assert!(re("\\p{Lu}", "u").is_match("A"));
    assert!(!re("\\p{Lu}", "u").is_match("a"));
    assert!(re("\\P{L}", "u").is_match("5")); // negated
    assert!(re("^[\\p{L}\\p{N}]+$", "u").is_match("abc123")); // in a class

    // `Property=Value` forms: General_Category, Script, Script_Extensions, with
    // canonical names, aliases (`gc`/`sc`/`scx`), and ISO 15924 short codes.
    assert!(re("^\\p{General_Category=Letter}$", "u").is_match("a"));
    assert!(re("^\\p{gc=L}$", "u").is_match("a"));
    assert!(re("^\\p{Script=Greek}+$", "u").is_match("αβ"));
    assert!(re("^\\p{sc=Latn}$", "u").is_match("a"));
    assert!(!re("^\\p{Script=Greek}$", "u").is_match("a"));
    assert!(re("^\\p{Script_Extensions=Latin}$", "u").is_match("a"));
    // Synthetic GC values: `LC`/`Cased_Letter` and POSIX-style aliases.
    assert!(re("^\\p{LC}$", "u").is_match("A"));
    assert!(re("^\\p{gc=digit}$", "u").is_match("7"));
    // Closed-form binary properties.
    assert!(re("^\\p{ASCII}$", "u").is_match("a"));
    assert!(!re("^\\p{ASCII}$", "u").is_match("é"));
    assert!(re("^\\p{ASCII_Hex_Digit}+$", "u").is_match("0aF"));
    assert!(re("^\\p{White_Space}$", "u").is_match(" "));
    assert!(re("^\\p{Alphabetic}$", "u").is_match("a"));

    // --- Validation (→ SyntaxError at compile) ---
    // Unknown property / value names must be rejected.
    assert!(Regex::new("\\p{Nonsense}", "u").is_err());
    assert!(Regex::new("\\p{Script=Nonsense}", "u").is_err());
    // A non-binary property used as a lone name must be rejected.
    assert!(Regex::new("\\p{General_Category}", "u").is_err());
    assert!(Regex::new("\\p{Script}", "u").is_err());
    // A binary property with an explicit value must be rejected.
    assert!(Regex::new("\\p{ASCII=Invalid}", "u").is_err());
    assert!(Regex::new("\\p{Alphabetic=Yes}", "u").is_err());
    // Malformed grammar.
    assert!(Regex::new("\\p{}", "u").is_err());
    assert!(Regex::new("\\p{=}", "u").is_err());
    assert!(Regex::new("\\p{=L}", "u").is_err());
    assert!(Regex::new("\\p{^L}", "u").is_err());
    assert!(Regex::new("\\p{ L }", "u").is_err()); // no whitespace folding
    assert!(Regex::new("\\pL", "u").is_err()); // braces required under `u`
    assert!(Regex::new("\\p", "u").is_err());
    assert!(Regex::new("\\p{", "u").is_err());
    // Valid binary property names with no local data still parse (no SyntaxError).
    assert!(Regex::new("\\p{Emoji}", "u").is_ok());
    assert!(Regex::new("\\p{ID_Start}", "u").is_ok());

    // Without the `u` flag, `\p`/`\P` are an Annex B IdentityEscape: the literal
    // `p`/`P`, so any "property name" text is just literal characters and never a
    // SyntaxError. (Compiled in non-unicode mode via the native program.)
    assert!(Regex::new("\\p{Nonsense}", "").is_ok());
    assert!(Regex::new("\\pL", "").is_ok());
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

#[test]
fn lazy_scalar_prog_reused_is_consistent() {
    // RE-P2: the scalar adapter program is built lazily on first `&str` use; a
    // single compiled `Regex` reused across many adapter calls must keep
    // returning identical results (the `OnceCell` is filled once, not rebuilt).
    let r = re(r"(\d+)", "");
    for _ in 0..5 {
        assert_eq!(r.captures_from("a12b", 0).unwrap().whole(), (1, 3));
        assert_eq!(r.find_from("xx99", 0), Some((2, 4)));
        assert!(r.is_match("z7"));
    }

    // The native u16 path (what the interpreter uses) and the scalar adapter path
    // agree for the same reused regex, and stay consistent over repeated calls.
    let g = re(r"(\d+)", "g");
    for _ in 0..3 {
        let units = u16s("a1b22c333");
        let mut pos = 0;
        let mut found = alloc::vec::Vec::new();
        while let Some((s, e)) = g.find_in_u16(&units, pos) {
            found.push((s, e));
            pos = if e > s { e } else { e + 1 };
        }
        assert_eq!(found, alloc::vec![(1, 2), (3, 5), (6, 9)]);
    }

    // Astral scalar atomicity through the lazily-built scalar program: an astral
    // char is one atom for `.` on the `&str` adapter path.
    let dot = re(".", "");
    assert_eq!(dot.find_from("😀x", 0), Some((0, 1)));
}

#[test]
fn positive_lookahead_captures_propagate() {
    // A positive lookahead contributes the groups its sub-match captured.
    let units = u16s("123");
    let caps = re(r"(?=(\d+))", "").captures_in_u16(&units, 0).unwrap();
    assert_eq!(caps.whole(), (0, 0)); // zero-width
    assert_eq!(caps.group(1), Some((0, 3))); // captured "123"

    // The captured group is usable by a later backreference.
    let units = u16s("abcabc");
    assert!(re(r"(?=(\w{3}))\1", "").find_in_u16(&units, 0).is_some());

    // A negative lookahead contributes nothing.
    let units = u16s("ac");
    let caps = re(r"a(?!(b))", "").captures_in_u16(&units, 0).unwrap();
    assert_eq!(caps.group(1), None);
}

#[test]
fn positive_lookbehind_captures_propagate() {
    // A positive lookbehind reports the groups of the matched substring.
    let units = u16s("foobar");
    let caps = re(r"(?<=(o)(o))bar", "")
        .captures_in_u16(&units, 0)
        .unwrap();
    assert_eq!(caps.group(1), Some((1, 2)));
    assert_eq!(caps.group(2), Some((2, 3)));

    // A negative lookbehind contributes nothing.
    let units = u16s("za");
    let caps = re(r"(?<!(x))a", "").captures_in_u16(&units, 0).unwrap();
    assert_eq!(caps.group(1), None);
}

#[test]
fn group_name_validation() {
    // Valid identifier-name groups compile.
    assert!(Regex::new(r"(?<a>x)", "").is_ok());
    assert!(Regex::new(r"(?<$_>x)", "").is_ok());
    assert!(Regex::new(r"(?<A>x)", "").is_ok());
    // Invalid group names are a Syntax Error.
    assert!(Regex::new(r"(?<1a>x)", "").is_err()); // digit start
    assert!(Regex::new(r"(?<a b>x)", "").is_err()); // space
    assert!(Regex::new(r"(?<>x)", "").is_err()); // empty
}

#[test]
fn duplicate_group_names() {
    // Same alternative → Syntax Error.
    assert!(Regex::new(r"(?<a>x)(?<a>y)", "").is_err());
    // Different (mutually exclusive) alternatives → allowed (ES2025).
    assert!(Regex::new(r"(?<a>x)|(?<a>y)", "").is_ok());
    assert!(Regex::new(r"(?:(?<a>x)|(?<a>y))", "").is_ok());
    // Nested in the same alternative → Syntax Error.
    assert!(Regex::new(r"(?<a>(?<a>y))", "").is_err());
}

#[test]
fn named_backreference_validation() {
    // A reference to a declared name is fine (even forward).
    assert!(Regex::new(r"(?<a>x)\k<a>", "").is_ok());
    assert!(Regex::new(r"\k<a>(?<a>x)", "").is_ok());
    // A reference to an undefined name (when named groups exist) is an error.
    assert!(Regex::new(r"(?<a>x)\k<b>", "").is_err());
    // In `u` mode any `\k<…>` requires a matching group.
    assert!(Regex::new(r"\k<a>", "u").is_err());
    // Annex B: with no named groups (non-`u`), `\k<a>` is the literal `k<a>`.
    let r = re(r"\k<a>", "");
    assert!(r.is_match("k<a>"));
    assert!(!r.is_match("xyz"));
}

#[test]
fn unicode_mode_strict_syntax() {
    // Out-of-range / legacy numeric escapes are errors under `u`.
    assert!(Regex::new(r"\1", "u").is_err());
    assert!(Regex::new(r"\8", "u").is_err());
    assert!(Regex::new(r"(a)\1", "u").is_ok());
    // Invalid identity escapes are errors under `u`, fine under Annex B.
    assert!(Regex::new(r"\M", "u").is_err());
    assert!(re(r"\M", "").is_match("M"));
    // Lone `{`, `}`, `]` are errors under `u`, literal under Annex B.
    assert!(Regex::new(r"{", "u").is_err());
    assert!(Regex::new(r"}", "u").is_err());
    assert!(Regex::new(r"]", "u").is_err());
    assert!(re(r"}", "").is_match("}"));
    // Control and character escapes remain valid under `u`.
    assert!(Regex::new(r"\cA", "u").is_ok());
    assert!(Regex::new(r"\n\t\r\f\v\0", "u").is_ok());
    // SyntaxCharacter identity escapes are valid under `u`.
    assert!(Regex::new(r"\.\*\+\?\(\)\[\]\{\}\|\^\$\\\/", "u").is_ok());
}

#[test]
fn quantifier_on_assertion() {
    // A lookbehind is never quantifiable (both modes).
    assert!(Regex::new(r"(?<=a)?b", "").is_err());
    assert!(Regex::new(r"(?<=a)?b", "u").is_err());
    assert!(Regex::new(r"(?<=a){2}b", "u").is_err());
    // A lookahead is quantifiable under Annex B (non-`u`) but not under `u`.
    assert!(Regex::new(r"(?=a)?b", "").is_ok());
    assert!(Regex::new(r"(?=a)*b", "").is_ok());
    assert!(Regex::new(r"(?=a)?b", "u").is_err());
    // `^`, `$`, `\b` are unquantifiable under `u`.
    assert!(Regex::new(r"^?a", "u").is_err());
    assert!(Regex::new(r"\b?a", "u").is_err());
    // A quantified group around an assertion is fine.
    assert!(Regex::new(r"(?:^)+", "u").is_ok());
}

#[test]
fn inline_modifier_groups() {
    // Add/remove i/m/s scoped to the group.
    assert!(re("(?i:A)", "").is_match("a"));
    assert!(!re("(?-i:A)", "i").is_match("a"));
    assert!(re("(?m:^a$)", "").is_match("x\na\ny"));
    assert!(re("(?s:.)", "").is_match("\n"));
    // `i` applies only inside the scope.
    assert!(re("(?i:a)b", "").is_match("Ab"));
    assert!(!re("(?i:a)b", "").is_match("AB"));
    // Nested modifiers: inner remove overrides outer add.
    assert!(!re("(?i:(?-i:a))", "").is_match("A"));
    assert!(re("(?i:(?-i:a))", "").is_match("a"));
    // Valid combined / single-sided forms stay accepted.
    assert!(Regex::new("(?i-m:a)", "").is_ok());
    assert!(Regex::new("(?-i:a)", "").is_ok());
    assert!(Regex::new("(?ims:a)", "").is_ok());
    // RemoveFlags after `-` may be empty: add-only with a trailing dash is valid.
    assert!(Regex::new("(?i-:a)", "").is_ok());
    // Syntax errors — each must be a *hard* error (not an `unsupported` deferral),
    // so that a regex *literal* is rejected at parse phase. `is_unsupported()` must
    // be false for every one.
    for src in [
        "(?-:a)",     // both sets empty
        "(?ii:a)",    // duplicate within add
        "(?i-mm:a)",  // duplicate within remove
        "(?ims-m:a)", // `m` in both add and remove
        "(?i-i:a)",   // single flag in both
        "(?d:a)",     // invalid flag (g/y/u/d/v/uppercase/etc.)
        "(?g:a)",
        "(?u:a)",
        "(?y:a)",
        "(?I:a)",
        "(?Q:a)",
        "(?1:a)",
        "(?i)", // modifier flags with no `:Disjunction`
        "(?ms-i)",
        "(?-s)",
        "(?i-)", // no colon
    ] {
        match Regex::new(src, "") {
            Ok(_) => panic!("{src} should be a SyntaxError"),
            Err(e) => assert!(!e.is_unsupported(), "{src} should be a hard SyntaxError"),
        }
    }
}

#[test]
fn ignore_case_unicode_word_carveout() {
    // Under `iu`, U+017F (ſ) and U+212A (K) count as word chars.
    assert!(re(r"\w", "iu").is_match("\u{017F}"));
    assert!(re(r"\w", "iu").is_match("\u{212A}"));
    // Without both flags, they do not.
    assert!(!re(r"\w", "u").is_match("\u{017F}"));
}

#[test]
fn case_insensitive_property() {
    // `\p{Lu}` matches lowercase under `i`; `\P{Lu}` matches uppercase under `i`.
    assert!(re(r"\p{Lu}", "iu").is_match("a"));
    assert!(re(r"\P{Lu}", "iu").is_match("A"));
    assert!(!re(r"\p{Lu}", "u").is_match("a"));
}

#[test]
fn v_flag_set_operations() {
    // Intersection, difference, union, nested classes.
    assert!(re("^[[0-9]&&[0-9]]+$", "v").is_match("123"));
    assert!(!re("^[[0-9]&&[a-z]]+$", "v").is_match("1"));
    assert!(re(r"^[\d--[0-5]]+$", "v").is_match("789"));
    assert!(!re(r"^[\d--[0-5]]+$", "v").is_match("3"));
    assert!(re(r"^[\p{ASCII}&&\p{L}]+$", "v").is_match("abc"));
    assert!(!re(r"^[\p{ASCII}&&\p{L}]+$", "v").is_match("a1"));
}

#[test]
fn v_flag_string_literals() {
    // `\q{…}` string alternatives, longest-match preferred.
    assert!(re(r"^[\q{abc|de}]+$", "v").is_match("abcde"));
    assert!(!re(r"^[\q{abc|de}]+$", "v").is_match("abccd"));
    assert!(re(r"^[[0-9]\q{ab}]+$", "v").is_match("ab9"));
}

#[test]
fn group_name_surrogate_pairs_non_unicode() {
    // In a non-`u` regex a group name may use `\u` surrogate pairs, `\u{…}` code
    // point escapes, or literal astral characters (all name U+1D453 `𝑓` …).
    // Previously a `\u`-surrogate-pair name was rejected as an invalid code point.
    assert!(Regex::new(r"(?<𝑓>fox)", "").is_ok());
    assert!(Regex::new(r"(?<\u{1d453}>fox)", "").is_ok());
    assert!(Regex::new("(?<\u{1d453}>fox)", "").is_ok());
    // Named backreference to an astral-named group.
    assert!(re(r"(?<𝑓>dog)(.*?)(\k<𝑓>)", "").is_match("dog eat dog"));
    // A lone `\u` surrogate that does not pair is still an invalid name.
    assert!(Regex::new(r"(?<\ud835x>fox)", "").is_err());
}

#[test]
fn v_flag_property_of_strings_keycap() {
    // `\p{Emoji_Keycap_Sequence}` matches the twelve `<base>U+FE0F U+20E3`
    // keycap strings (multi-code-point), and nothing else.
    let kc = "0\u{FE0F}\u{20E3}"; // 0️⃣
    let hashkc = "#\u{FE0F}\u{20E3}"; // #️⃣
    assert!(re(r"^\p{Emoji_Keycap_Sequence}$", "v").is_match(kc));
    assert!(re(r"^\p{Emoji_Keycap_Sequence}$", "v").is_match(hashkc));
    assert!(!re(r"^\p{Emoji_Keycap_Sequence}$", "v").is_match("0"));
    // As a class operand, combined with a string literal via union.
    let re1 = re(r"^[\p{Emoji_Keycap_Sequence}\q{0|2}]+$", "v");
    assert!(re1.is_match(kc));
    assert!(re1.is_match("0"));
    let mut combo = alloc::string::String::new();
    combo.push_str(kc);
    combo.push_str(hashkc);
    combo.push('0');
    assert!(re1.is_match(&combo));
    // Difference: `[0-9]` minus the keycap strings still matches bare digits
    // (a bare `0` is not a keycap sequence) but never a keycap sequence.
    let d = re(r"^[[0-9]--\p{Emoji_Keycap_Sequence}]+$", "v");
    assert!(d.is_match("019"));
    assert!(!d.is_match(kc));
    // Intersection with a string literal set keeps only the shared strings.
    let i = re(r"^[\p{Emoji_Keycap_Sequence}&&\q{0️⃣|zz}]+$", "v");
    assert!(i.is_match(kc));
    assert!(!i.is_match(hashkc));
}

#[test]
fn v_flag_property_of_strings_errors() {
    // A property of strings may not be negated…
    assert!(Regex::new(r"\P{Emoji_Keycap_Sequence}", "v").is_err());
    // …nor appear inside a negated class…
    assert!(Regex::new(r"[^\p{Emoji_Keycap_Sequence}]", "v").is_err());
    // …and unsupported string properties stay a hard error when matched.
    assert!(Regex::new(r"[\p{RGI_Emoji}]", "v").is_err());
}

#[test]
fn v_flag_syntax_errors() {
    assert!(Regex::new(r"[^\q{ab}]", "v").is_err()); // negated class with string
    assert!(Regex::new("[a~~b]", "v").is_err()); // reserved double punctuator
    assert!(Regex::new("[a&&]", "v").is_err()); // trailing operator
    assert!(Regex::new("[(]", "v").is_err()); // unescaped reserved char
    assert!(Regex::new(r"[\(]", "v").is_ok()); // escaped is fine
}
