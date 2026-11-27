//! Whether this thread's code can reach the legacy `f.caller` /
//! `f.arguments` of a sloppy ordinary function.
//!
//! ECMA-262 does not specify these two properties on non-strict functions
//! (17.1 Forbidden Extensions only restricts strict, arrow, method,
//! generator and async functions), but engines answer them and so does this
//! one: `f.caller` names the function that invoked `f`'s innermost live
//! activation, `f.arguments` an arguments object over that activation's
//! arguments (`nbvm::legacy_activations`, the one reader). Answering
//! needs every live sloppy activation to be on record, which a compiled call
//! pays for with a native-call header (`jit::call_seq`): about a tenth of a
//! JS-to-JS call.
//!
//! Almost no program ever reads them, and a read needs the property key
//! `"caller"` or `"arguments"`. So the engine keeps a per-thread flag,
//! **reachable**, that turns on — and then stays on — as soon as
//!
//! * program text of any kind is lexed (scripts, modules, `eval`,
//!   `new Function`, …: every path that turns text into bytecode goes
//!   through the lexer, see `note_token`) that names `caller` anywhere, an
//!   `arguments` property (`x.arguments`, an object literal's or pattern's
//!   `arguments` key), a `with` statement, a string or template literal
//!   containing either word (code to `eval`, or a computed key), or `eval`,
//!   `Function` or `evalScript` (text that becomes code later, while this
//!   text's activations run);
//! * a portable bytecode artifact is loaded (its source was never lexed
//!   here);
//! * the legacy reader runs at all ([`note`] is called before it looks).
//!
//! While the flag is off, a compiled call enters a sloppy ordinary callee
//! **headerless**, as it enters a strict one; turning it on stops that for
//! every later call (`jit::legacy_hl::invalidate` drops the headerless ways
//! already cached at call sites). The one thing it cannot do is give back
//! the records of headerless activations that were already running when it
//! turned on. Today the reader is only reached through a property name
//! spelled in lexed text (a computed `f[k]` or `Reflect.get` answers from
//! the host, which never looks at activations), so that takes text lexed
//! while they run that the rules above did not see coming: code built at
//! run time and compiled through a route not named `eval` / `Function` /
//! `evalScript` (a function's `.constructor`, a generator function's).
//! Such a read sees those activations as a strict one's (`null`).
//!
//! The flag is per thread because compiled code and its native register
//! stack are (`jit::call_seq::JitStack`); without `std` it is always on.

#[cfg(feature = "std")]
std::thread_local! {
    static REACHABLE: core::cell::Cell<bool> = core::cell::Cell::new(!enabled());
}

/// Whether headerless sloppy calls are allowed at all
/// (`KATAAN_JIT_LEGACYHL=0` turns them off; read once per process).
#[cfg(feature = "std")]
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("KATAAN_JIT_LEGACYHL").map_or(true, |v| v != "0"))
}

/// Whether this thread's code may read a sloppy function's legacy
/// `caller` / `arguments` (see the module docs). Sticky once `true`.
#[inline]
#[must_use]
pub fn reachable() -> bool {
    #[cfg(feature = "std")]
    {
        REACHABLE.with(core::cell::Cell::get)
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// Turns [`reachable`] on (and keeps it on), dropping the headerless ways
/// compiled call sites cached for sloppy callees.
#[cold]
pub fn note() {
    #[cfg(feature = "std")]
    if !REACHABLE.with(|r| r.replace(true)) {
        #[cfg(all(feature = "jit", jit_native))]
        crate::jit::legacy_hl::invalidate();
    }
}

/// Whether `s` contains `caller` or `arguments`.
fn mentions(s: &[u8]) -> bool {
    fn has(s: &[u8], w: &[u8]) -> bool {
        s.len() >= w.len() && s.windows(w.len()).any(|x| x == w)
    }
    has(s, b"caller") || has(s, b"arguments")
}

/// The lexer's hook, for each token it makes (see the module docs):
/// `kind`, its raw `text`, whether an identifier was written with escapes,
/// the previous significant token and whether the innermost open brace is
/// an object literal's, class body's or pattern's (not a block's).
pub(crate) fn note_token(
    kind: crate::lexer::TokenKind,
    text: &[u8],
    had_escape: bool,
    prev: Option<crate::lexer::TokenKind>,
    in_object_brace: bool,
) {
    use crate::lexer::{Keyword, TokenKind as K};
    if reachable() {
        return;
    }
    match kind {
        K::Identifier => {
            let cooked;
            let name: &[u8] = if had_escape {
                cooked =
                    crate::parser::cook_identifier_name(crate::wtf8::as_str(text).unwrap_or(""));
                cooked.as_bytes()
            } else {
                text
            };
            if matches!(name, b"caller" | b"eval" | b"Function" | b"evalScript")
                || (name == b"arguments"
                    && (in_object_brace || matches!(prev, Some(K::Dot | K::QuestionDot))))
            {
                note();
            }
        }
        K::Keyword(Keyword::With) => note(),
        K::String
        | K::NoSubstitutionTemplate
        | K::TemplateHead
        | K::TemplateMiddle
        | K::TemplateTail => {
            if mentions(text) {
                note();
            } else if text.contains(&b'\\') {
                // (Escapes may spell either word.)
                match crate::parser::cook_escapes(text) {
                    Some(v) if !mentions(&v) => {}
                    _ => note(),
                }
            }
        }
        _ => {}
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    /// Whether lexing `src` on a fresh thread turns the flag on.
    fn turns_on(src: &'static str) -> bool {
        std::thread::spawn(move || {
            assert!(!super::reachable());
            let _ = crate::lexer::Lexer::new(src).tokenize();
            super::reachable()
        })
        .join()
        .expect("lexer thread")
    }

    #[test]
    fn names_that_reach_the_legacy_properties() {
        if !super::enabled() {
            return;
        }
        for src in [
            "f.caller",
            "f?.caller",
            "f.\\u0063aller",
            "var caller = 1",
            "f.arguments",
            "f?.arguments",
            "const { arguments: a } = f",
            "var { arguments } = f",
            "with (f) x",
            "f['caller']",
            "f[\"argu\\x6dents\"]",
            "f[`caller`]",
            "f[`a${b}caller`]",
            "eval(s)",
            "new Function(s)",
            "$262.evalScript(s)",
            "x = 'g.caller'",
        ] {
            assert!(turns_on(src), "{src}");
        }
    }

    #[test]
    fn code_that_does_not() {
        if !super::enabled() {
            return;
        }
        for src in [
            "function f(a) { return arguments.length + arguments[0] + a }",
            "function f() { return g.apply(this, arguments) }",
            "var call = 1, er = 2; f.call(null)",
            "x = 'call' + 'er'",
            "x = `a${b}c`",
        ] {
            assert!(!turns_on(src), "{src}");
        }
    }
}
