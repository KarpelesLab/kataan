//! The engine-level error type.
//!
//! This is the *host-facing* error returned by the public Rust API (parsing a
//! script, evaluating it, driving the runtime). It is distinct from a thrown
//! JavaScript value: a JS `throw` produces a runtime [`Value`](crate) handled
//! inside the VM, whereas [`Error`] represents an engine outcome surfaced to
//! the embedder — a syntax error with its source location, a resource limit,
//! or an internal invariant violation.

use crate::common::Span;
use alloc::boxed::Box;
use alloc::string::String;
use core::fmt;

/// A specialized [`Result`](core::result::Result) for engine operations.
pub type Result<T> = core::result::Result<T, Error>;

/// An error surfaced from the engine to the embedder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The source text was not valid ECMAScript. Carries a human-readable
    /// message and the source [`Span`] the error was detected at.
    Syntax {
        /// What was wrong (e.g. `"unterminated string literal"`).
        message: Box<str>,
        /// Where in the source the problem is.
        span: Span,
    },

    /// A configured resource limit (stack depth, heap size, instruction
    /// budget, …) was exceeded.
    ResourceLimit(&'static str),

    /// An engine invariant was violated — a bug in Kataan, not in the input.
    /// Reaching this is never the script's fault.
    Internal(&'static str),
}

impl Error {
    /// Builds a [`Error::Syntax`] from any displayable message and a span.
    pub fn syntax(message: impl Into<String>, span: Span) -> Self {
        Error::Syntax {
            message: message.into().into_boxed_str(),
            span,
        }
    }

    /// The source span this error refers to, if it is positional.
    #[must_use]
    pub fn span(&self) -> Option<Span> {
        match self {
            Error::Syntax { span, .. } => Some(*span),
            Error::ResourceLimit(_) | Error::Internal(_) => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Syntax { message, span } => {
                write!(f, "SyntaxError: {message} (at {span:?})")
            }
            Error::ResourceLimit(what) => write!(f, "resource limit exceeded: {what}"),
            Error::Internal(what) => write!(f, "internal error: {what}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
