//! Token kinds, keywords, and the [`Token`] type produced by the [`Lexer`].
//!
//! [`Lexer`]: super::Lexer

use crate::common::Span;

/// A single lexical token: its [`TokenKind`], the source [`Span`] it covers,
/// and whether a line terminator preceded it (the signal Automatic Semicolon
/// Insertion needs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    /// What kind of token this is.
    pub kind: TokenKind,
    /// The byte range in the source this token covers.
    pub span: Span,
    /// Whether at least one line terminator (or a block comment containing
    /// one) appeared in the trivia immediately before this token.
    pub newline_before: bool,
}

impl Token {
    /// The raw source text of this token.
    #[inline]
    #[must_use]
    pub fn text<'s>(&self, source: &'s str) -> &'s str {
        self.span.slice(source)
    }
}

/// The lexical category of a [`Token`].
///
/// Literal tokens ([`Number`](Self::Number), [`String`](Self::String),
/// [`Regex`](Self::Regex), the template parts, …) carry no decoded value: the
/// value is recovered from the token's source [`Span`] by the parser. This
/// keeps `TokenKind` a cheap `Copy` enum and defers cooking strings/numbers to
/// the stage that needs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TokenKind {
    // --- end of input ---
    /// End of the source text.
    Eof,

    // --- names & literals ---
    /// An identifier or a contextual keyword used as a name.
    Identifier,
    /// A reserved word; the specific keyword is carried inline.
    Keyword(Keyword),
    /// A private class member name, e.g. `#count`.
    PrivateName,
    /// A numeric literal (decimal, hex/octal/binary, float, exponent).
    Number,
    /// A `BigInt` literal (a numeric literal with the `n` suffix).
    BigInt,
    /// A string literal (single- or double-quoted), escapes not yet decoded.
    String,
    /// A regular-expression literal `/pattern/flags`.
    Regex,
    /// A template with no substitutions: `` `text` ``.
    NoSubstitutionTemplate,
    /// The head of a template up to the first `${`: `` `text${ ``.
    TemplateHead,
    /// A template part between two substitutions: `` }text${ ``.
    TemplateMiddle,
    /// The tail of a template after the last substitution: `` }text` ``.
    TemplateTail,

    // --- brackets ---
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,

    // --- punctuation ---
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `...`
    DotDotDot,
    /// `:`
    Colon,
    /// `?`
    Question,
    /// `?.`
    QuestionDot,
    /// `??`
    QuestionQuestion,
    /// `=>`
    Arrow,

    // --- operators ---
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    LtEq,
    /// `>=`
    GtEq,
    /// `==`
    EqEq,
    /// `!=`
    BangEq,
    /// `===`
    EqEqEq,
    /// `!==`
    BangEqEq,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `**`
    StarStar,
    /// `++`
    PlusPlus,
    /// `--`
    MinusMinus,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `>>>`
    Ushr,
    /// `&`
    Amp,
    /// `|`
    Pipe,
    /// `^`
    Caret,
    /// `!`
    Bang,
    /// `~`
    Tilde,
    /// `&&`
    AmpAmp,
    /// `||`
    PipePipe,

    // --- assignment ---
    /// `=`
    Eq,
    /// `+=`
    PlusEq,
    /// `-=`
    MinusEq,
    /// `*=`
    StarEq,
    /// `/=`
    SlashEq,
    /// `%=`
    PercentEq,
    /// `**=`
    StarStarEq,
    /// `<<=`
    ShlEq,
    /// `>>=`
    ShrEq,
    /// `>>>=`
    UshrEq,
    /// `&=`
    AmpEq,
    /// `|=`
    PipeEq,
    /// `^=`
    CaretEq,
    /// `&&=`
    AmpAmpEq,
    /// `||=`
    PipePipeEq,
    /// `??=`
    QuestionQuestionEq,
}

impl TokenKind {
    /// Whether this token is a template part that introduces or continues a
    /// substitution context (`` `…${ `` or `` }…${ ``).
    #[must_use]
    pub fn is_template_open(self) -> bool {
        matches!(self, TokenKind::TemplateHead | TokenKind::TemplateMiddle)
    }
}

/// The ECMAScript reserved words and contextual keywords the lexer recognizes.
///
/// This includes the always-reserved words, the strict-mode-reserved words,
/// and the common contextual keywords (`async`, `await`, `let`, `of`,
/// `yield`, `static`, `get`, `set`, …). Whether a contextual keyword acts as a
/// keyword or an identifier in a given position is the parser's job; the lexer
/// classifies the spelling and lets the parser decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)] // each variant is its own keyword; the name is the doc
pub enum Keyword {
    // Always reserved.
    Await,
    Break,
    Case,
    Catch,
    Class,
    Const,
    Continue,
    Debugger,
    Default,
    Delete,
    Do,
    Else,
    Enum,
    Export,
    Extends,
    False,
    Finally,
    For,
    Function,
    If,
    Import,
    In,
    Instanceof,
    New,
    Null,
    Return,
    Super,
    Switch,
    This,
    Throw,
    True,
    Try,
    Typeof,
    Var,
    Void,
    While,
    With,
    // Strict-mode reserved.
    Implements,
    Interface,
    Let,
    Package,
    Private,
    Protected,
    Public,
    Static,
    Yield,
    // Common contextual keywords.
    As,
    Async,
    From,
    Get,
    Of,
    Set,
    Target,
    Accessor,
}

impl Keyword {
    /// Maps a spelling to a [`Keyword`], or `None` if it is an ordinary
    /// identifier. (Not `FromStr`: a non-keyword is `None`, not an error.)
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Keyword> {
        use Keyword::*;
        Some(match s {
            "await" => Await,
            "break" => Break,
            "case" => Case,
            "catch" => Catch,
            "class" => Class,
            "const" => Const,
            "continue" => Continue,
            "debugger" => Debugger,
            "default" => Default,
            "delete" => Delete,
            "do" => Do,
            "else" => Else,
            "enum" => Enum,
            "export" => Export,
            "extends" => Extends,
            "false" => False,
            "finally" => Finally,
            "for" => For,
            "function" => Function,
            "if" => If,
            "import" => Import,
            "in" => In,
            "instanceof" => Instanceof,
            "new" => New,
            "null" => Null,
            "return" => Return,
            "super" => Super,
            "switch" => Switch,
            "this" => This,
            "throw" => Throw,
            "true" => True,
            "try" => Try,
            "typeof" => Typeof,
            "var" => Var,
            "void" => Void,
            "while" => While,
            "with" => With,
            "implements" => Implements,
            "interface" => Interface,
            "let" => Let,
            "package" => Package,
            "private" => Private,
            "protected" => Protected,
            "public" => Public,
            "static" => Static,
            "yield" => Yield,
            "as" => As,
            "async" => Async,
            "from" => From,
            "get" => Get,
            "of" => Of,
            "set" => Set,
            "target" => Target,
            "accessor" => Accessor,
            _ => return None,
        })
    }

    /// The canonical spelling of this keyword.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        use Keyword::*;
        match self {
            Await => "await",
            Break => "break",
            Case => "case",
            Catch => "catch",
            Class => "class",
            Const => "const",
            Continue => "continue",
            Debugger => "debugger",
            Default => "default",
            Delete => "delete",
            Do => "do",
            Else => "else",
            Enum => "enum",
            Export => "export",
            Extends => "extends",
            False => "false",
            Finally => "finally",
            For => "for",
            Function => "function",
            If => "if",
            Import => "import",
            In => "in",
            Instanceof => "instanceof",
            New => "new",
            Null => "null",
            Return => "return",
            Super => "super",
            Switch => "switch",
            This => "this",
            Throw => "throw",
            True => "true",
            Try => "try",
            Typeof => "typeof",
            Var => "var",
            Void => "void",
            While => "while",
            With => "with",
            Implements => "implements",
            Interface => "interface",
            Let => "let",
            Package => "package",
            Private => "private",
            Protected => "protected",
            Public => "public",
            Static => "static",
            Yield => "yield",
            As => "as",
            Async => "async",
            From => "from",
            Get => "get",
            Of => "of",
            Set => "set",
            Target => "target",
            Accessor => "accessor",
        }
    }

    /// Whether, when this keyword is the previous significant token, a
    /// following `/` should be read as the start of a regex literal rather
    /// than division. Keywords that can end an expression (`this`, `super`,
    /// `true`, `false`, `null`) are followed by division; keywords that
    /// introduce an expression (`return`, `typeof`, `case`, `in`, …) are
    /// followed by a regex.
    #[must_use]
    pub fn before_expression(self) -> bool {
        use Keyword::*;
        !matches!(self, This | Super | True | False | Null)
    }
}

#[cfg(test)]
mod tests {
    use super::{Keyword, TokenKind};

    #[test]
    fn keyword_roundtrip() {
        for kw in [
            Keyword::Await,
            Keyword::Function,
            Keyword::Yield,
            Keyword::Of,
        ] {
            assert_eq!(Keyword::from_str(kw.as_str()), Some(kw));
        }
        assert_eq!(Keyword::from_str("notakeyword"), None);
        assert_eq!(Keyword::from_str("Function"), None); // case-sensitive
    }

    #[test]
    fn template_open_classification() {
        assert!(TokenKind::TemplateHead.is_template_open());
        assert!(TokenKind::TemplateMiddle.is_template_open());
        assert!(!TokenKind::TemplateTail.is_template_open());
        assert!(!TokenKind::NoSubstitutionTemplate.is_template_open());
    }

    #[test]
    fn regex_after_keyword() {
        assert!(Keyword::Return.before_expression());
        assert!(Keyword::Typeof.before_expression());
        assert!(!Keyword::This.before_expression());
        assert!(!Keyword::True.before_expression());
    }
}
