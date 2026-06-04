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
fn errors() {
    assert!(Regex::new("(unterminated", "").is_err());
    assert!(Regex::new("[abc", "").is_err());
    assert!(Regex::new("a", "z").is_err()); // unknown flag
    assert!(Regex::new("*abc", "").is_err()); // nothing to repeat
}
