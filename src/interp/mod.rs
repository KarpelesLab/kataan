//! A tree-walking interpreter — the Phase-C semantics MVP.
//!
//! This evaluator runs the AST directly to pin down ECMAScript semantics before
//! the performance-oriented bytecode VM, NaN-boxed values, hidden classes, and
//! GC replace it (see `ROADMAP.md`). It currently supports primitives, the full
//! operator/coercion set, control flow, and functions/closures (including
//! arrows); objects, arrays, and member access are the next increment.
//!
//! ```
//! use kataan::parser::Parser;
//! use kataan::interp::{Interp, Value};
//!
//! let program = Parser::parse_program("let x = 20; x * 2 + 2").unwrap();
//! let mut interp = Interp::new();
//! let result = interp.run(&program).unwrap();
//! assert_eq!(result.to_js_string(), "42");
//! ```

mod builtins;
mod env;
mod eval;
mod value;

#[cfg(test)]
mod tests;

pub use env::{Env, Scope};
pub use eval::{Completion, Interp};
pub use value::{Callable, ClassValue, Closure, NativeFn, Obj, Value, loose_equals, strict_equals};
