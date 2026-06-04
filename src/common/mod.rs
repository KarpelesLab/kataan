//! Cross-cutting building blocks shared by every layer of the engine: source
//! positions ([`Span`]) and, as the engine grows, the string interner, the
//! diagnostic types, and the bump arena.

mod span;

pub use span::Span;
