//! Lexer tests.

use super::{Keyword, Lexer, TokenKind};
use alloc::vec::Vec;

/// Lexes `src` and returns the token kinds, excluding the trailing `Eof`.
fn kinds(src: &str) -> Vec<TokenKind> {
    let toks = Lexer::new(src).tokenize().expect("lex ok");
    assert_eq!(toks.last().map(|t| t.kind), Some(TokenKind::Eof));
    toks[..toks.len() - 1].iter().map(|t| t.kind).collect()
}

/// Lexes `src`, asserting it fails, and returns the error message.
fn lex_err(src: &str) -> alloc::string::String {
    use alloc::string::ToString;
    Lexer::new(src).tokenize().unwrap_err().to_string()
}

#[test]
fn empty_and_whitespace() {
    assert_eq!(kinds(""), &[]);
    assert_eq!(kinds("   \t\n  "), &[]);
}

#[test]
fn simple_var_decl() {
    use TokenKind::*;
    assert_eq!(
        kinds("let x = 42;"),
        &[
            Keyword(self::Keyword::Let),
            Identifier,
            Eq,
            Number,
            Semicolon
        ]
    );
}

#[test]
fn punctuators_maximal_munch() {
    use TokenKind::*;
    assert_eq!(kinds(">>>="), &[UshrEq]);
    assert_eq!(kinds(">>>"), &[Ushr]);
    assert_eq!(kinds(">>="), &[ShrEq]);
    assert_eq!(kinds("==="), &[EqEqEq]);
    assert_eq!(kinds("!=="), &[BangEqEq]);
    assert_eq!(kinds("=>"), &[Arrow]);
    assert_eq!(kinds("**="), &[StarStarEq]);
    assert_eq!(kinds("??="), &[QuestionQuestionEq]);
    assert_eq!(kinds("?.x"), &[QuestionDot, Identifier]);
    assert_eq!(kinds("..."), &[DotDotDot]);
    assert_eq!(kinds("&&="), &[AmpAmpEq]);
    assert_eq!(kinds("||="), &[PipePipeEq]);
}

#[test]
fn optional_chain_vs_ternary_number() {
    use TokenKind::*;
    // `?.5` is `?` `.5`, not optional-chaining, per spec.
    assert_eq!(
        kinds("a?.5:1"),
        &[Identifier, Question, Number, Colon, Number]
    );
    // `a?.b` is optional chaining.
    assert_eq!(kinds("a?.b"), &[Identifier, QuestionDot, Identifier]);
}

#[test]
fn numbers() {
    use TokenKind::*;
    assert_eq!(kinds("0"), &[Number]);
    assert_eq!(kinds("3.14"), &[Number]);
    assert_eq!(kinds(".5"), &[Number]);
    assert_eq!(kinds("1e10"), &[Number]);
    assert_eq!(kinds("1.5E-3"), &[Number]);
    assert_eq!(kinds("0xFF"), &[Number]);
    assert_eq!(kinds("0o17"), &[Number]);
    assert_eq!(kinds("0b1010"), &[Number]);
    assert_eq!(kinds("1_000_000"), &[Number]);
    assert_eq!(kinds("0xDEAD_BEEF"), &[Number]);
    assert_eq!(kinds("123n"), &[BigInt]);
    assert_eq!(kinds("0xFFn"), &[BigInt]);
}

#[test]
fn bad_numbers() {
    assert!(lex_err("1_").contains("separator"));
    assert!(lex_err("1__2").contains("separator"));
    assert!(lex_err("0x").contains("missing digits"));
    assert!(lex_err("1e").contains("exponent"));
    assert!(lex_err("3in").contains("identifier directly after"));
}

#[test]
fn strings_and_escapes() {
    use TokenKind::*;
    assert_eq!(kinds(r#""hello""#), &[String]);
    assert_eq!(kinds(r"'it\'s'"), &[String]);
    assert_eq!(kinds(r#""tab\tnewline\n""#), &[String]);
    assert_eq!(kinds(r#""\x41B\u{1F600}""#), &[String]);
    // Line continuation.
    assert_eq!(kinds("'a\\\nb'"), &[String]);
}

#[test]
fn bad_strings() {
    assert!(lex_err("\"unterminated").contains("unterminated string"));
    assert!(lex_err("'line\nbreak'").contains("line terminator"));
    assert!(lex_err(r#""\xZZ""#).contains("hexadecimal escape"));
    assert!(lex_err(r#""\u{}""#).contains("code-point escape"));
}

#[test]
fn templates_no_substitution() {
    use TokenKind::*;
    assert_eq!(kinds("`hello world`"), &[NoSubstitutionTemplate]);
    assert_eq!(kinds("`a\\`b`"), &[NoSubstitutionTemplate]);
}

#[test]
fn templates_with_substitution() {
    use TokenKind::*;
    // `a${x}b`
    assert_eq!(kinds("`a${x}b`"), &[TemplateHead, Identifier, TemplateTail]);
    // `a${x}b${y}c`
    assert_eq!(
        kinds("`a${x}b${y}c`"),
        &[
            TemplateHead,
            Identifier,
            TemplateMiddle,
            Identifier,
            TemplateTail
        ]
    );
}

#[test]
fn templates_nested() {
    use TokenKind::*;
    // `${`${x}`}` — a template inside a substitution inside a template.
    assert_eq!(
        kinds("`${`${x}`}`"),
        &[
            TemplateHead,
            TemplateHead,
            Identifier,
            TemplateTail,
            TemplateTail
        ]
    );
}

#[test]
fn object_braces_not_confused_with_templates() {
    use TokenKind::*;
    // The `}` here closes an object literal, not a template substitution.
    assert_eq!(
        kinds("`${ {a:1} }`"),
        &[
            TemplateHead,
            LBrace,
            Identifier,
            Colon,
            Number,
            RBrace,
            TemplateTail
        ]
    );
}

#[test]
fn regex_vs_division() {
    use TokenKind::*;
    // Division: after an identifier.
    assert_eq!(
        kinds("a / b / c"),
        &[Identifier, Slash, Identifier, Slash, Identifier]
    );
    // Regex: at expression position (after `=`).
    assert_eq!(kinds("x = /ab+c/g"), &[Identifier, Eq, Regex]);
    // Regex: after `return`.
    assert_eq!(
        kinds("return /x/"),
        &[Keyword(self::Keyword::Return), Regex]
    );
    // Regex with a character class containing a slash.
    assert_eq!(kinds("= /[/]/"), &[Eq, Regex]);
    // Division after `)`.
    assert_eq!(
        kinds("(a) / b"),
        &[LParen, Identifier, RParen, Slash, Identifier]
    );
}

#[test]
fn comments_are_trivia() {
    use TokenKind::*;
    assert_eq!(kinds("a // line comment\nb"), &[Identifier, Identifier]);
    assert_eq!(kinds("a /* block */ b"), &[Identifier, Identifier]);
    assert_eq!(kinds("/* only a comment */"), &[]);
}

#[test]
fn newline_before_flag_for_asi() {
    let toks = Lexer::new("a\nb c").tokenize().unwrap();
    // a, b, c, Eof
    assert!(!toks[0].newline_before); // a
    assert!(toks[1].newline_before); // b — newline before
    assert!(!toks[2].newline_before); // c — same line as b
    // A block comment containing a newline counts for ASI.
    let toks = Lexer::new("a /*\n*/ b").tokenize().unwrap();
    assert!(toks[1].newline_before);
}

#[test]
fn private_names() {
    use TokenKind::*;
    assert_eq!(
        kinds("this.#count"),
        &[Keyword(self::Keyword::This), Dot, PrivateName]
    );
    assert!(lex_err("# ").contains("expected an identifier after `#`"));
}

#[test]
fn unicode_identifiers() {
    use TokenKind::*;
    assert_eq!(kinds("café"), &[Identifier]);
    assert_eq!(kinds("π = 3.14"), &[Identifier, Eq, Number]);
    let toks = Lexer::new("café").tokenize().unwrap();
    assert_eq!(toks[0].text("café".as_bytes()), "café".as_bytes());
    // Letters from various scripts and combining marks continue identifiers.
    assert_eq!(
        kinds("日本語 λ naïve _x$"),
        &[Identifier, Identifier, Identifier, Identifier]
    );
}

// `ID_Start` precision needs the Unicode property tables (the `intl` feature):
// a currency symbol is not a valid identifier and is rejected (not swallowed as
// a one-char identifier).
#[cfg(feature = "intl")]
#[test]
fn rejects_non_identifier_unicode() {
    assert!(Lexer::new("let € = 1;").tokenize().is_err());
    assert!(Lexer::new("°").tokenize().is_err());
    // But a valid identifier with a trailing combining mark is fine.
    assert!(Lexer::new("e\u{0301}").tokenize().is_ok());
}

#[test]
fn spans_point_at_source() {
    let src = "let answer = 42;";
    let toks = Lexer::new(src).tokenize().unwrap();
    assert_eq!(toks[0].text(src.as_bytes()), "let".as_bytes());
    assert_eq!(toks[1].text(src.as_bytes()), "answer".as_bytes());
    assert_eq!(toks[2].text(src.as_bytes()), "=".as_bytes());
    assert_eq!(toks[3].text(src.as_bytes()), "42".as_bytes());
    assert_eq!(toks[4].text(src.as_bytes()), ";".as_bytes());
}

#[test]
fn keywords_classified() {
    use TokenKind::*;
    assert_eq!(
        kinds("function f() { return null; }"),
        &[
            Keyword(self::Keyword::Function),
            Identifier,
            LParen,
            RParen,
            LBrace,
            Keyword(self::Keyword::Return),
            Keyword(self::Keyword::Null),
            Semicolon,
            RBrace
        ]
    );
}

#[test]
fn escaped_identifier_is_identifier_not_keyword() {
    use TokenKind::*;
    // `class` cooks to `class` but is an Identifier, not the keyword, and
    // the token records that it contained an escape.
    let toks = Lexer::new(r"cl\u0061ss").tokenize().expect("lex ok");
    assert_eq!(toks[0].kind, Identifier);
    assert!(toks[0].had_escape);
    // A plain keyword has no escape flag and the keyword kind.
    let toks2 = Lexer::new("class").tokenize().expect("lex ok");
    assert_eq!(toks2[0].kind, Keyword(self::Keyword::Class));
    assert!(!toks2[0].had_escape);
    // An escape that is not a valid identifier start (here U+0020 space) is
    // rejected.
    assert!(Lexer::new(r"\u0020abc").tokenize().is_err());
}

#[test]
fn hashbang_comment_at_start() {
    use TokenKind::*;
    // A leading `#!` line is a comment.
    assert_eq!(kinds("#!/usr/bin/env node\nx"), &[Identifier]);
    // `#!` not at the very start is not a hashbang (here `#` begins a private
    // name, which is an error outside a class — surfaced as a lex error).
    assert!(Lexer::new(" #!/x").tokenize().is_err());
}

#[test]
fn other_id_start_continue_chars() {
    use TokenKind::*;
    // U+2118 (℘) is `Other_ID_Start`; U+00B7 (·) is `Other_ID_Continue`.
    assert_eq!(kinds("\u{2118}"), &[Identifier]);
    assert_eq!(kinds("a\u{00B7}b"), &[Identifier]);
}
