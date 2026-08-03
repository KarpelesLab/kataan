//! Source spans.

use core::fmt;

/// A half-open byte range `[start, end)` into the source text a token or AST
/// node came from.
///
/// Offsets are byte offsets (not UTF-16 code units or character counts) so
/// they map directly onto the `&str` source for slicing. They are `u32`,
/// which caps a single compilation unit at 4 GiB of source — far beyond any
/// real script and half the memory footprint of `usize` spans.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// Byte offset of the first byte of the range.
    pub start: u32,
    /// Byte offset one past the last byte of the range.
    pub end: u32,
}

impl Span {
    /// A span covering the byte range `[start, end)`.
    #[inline]
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// An empty span at `offset`, useful for synthesized nodes and for marking
    /// an insertion point (e.g. an ASI-inserted semicolon).
    #[inline]
    #[must_use]
    pub const fn point(offset: u32) -> Self {
        Self {
            start: offset,
            end: offset,
        }
    }

    /// The length of the span in bytes.
    #[inline]
    #[must_use]
    pub const fn len(self) -> u32 {
        self.end - self.start
    }

    /// Whether the span covers zero bytes.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// The smallest span covering both `self` and `other`.
    #[inline]
    #[must_use]
    pub fn to(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Slices `source` to the bytes this span covers.
    ///
    /// # Panics
    ///
    /// Panics if the span is out of bounds for `source` or does not fall on
    /// UTF-8 character boundaries — which cannot happen for spans the lexer
    /// produces, since they are always derived from char boundaries.
    #[inline]
    #[must_use]
    pub fn slice(self, source: &str) -> &str {
        &source[self.start as usize..self.end as usize]
    }

    /// Slices the **WTF-8** `source` bytes this span covers.
    ///
    /// Program text is scanned as WTF-8 (a JS string handed to `eval` may hold
    /// lone surrogates, which `str` cannot represent), so the byte form is the
    /// lossless one. Spans always fall on code-point boundaries.
    ///
    /// # Panics
    ///
    /// Panics if the span is out of bounds for `source`.
    #[inline]
    #[must_use]
    pub fn slice_bytes(self, source: &[u8]) -> &[u8] {
        &source[self.start as usize..self.end as usize]
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::Span;

    #[test]
    fn len_and_empty() {
        assert_eq!(Span::new(3, 8).len(), 5);
        assert!(Span::point(4).is_empty());
        assert!(!Span::new(3, 8).is_empty());
    }

    #[test]
    fn merge() {
        assert_eq!(Span::new(2, 5).to(Span::new(7, 9)), Span::new(2, 9));
        assert_eq!(Span::new(7, 9).to(Span::new(2, 5)), Span::new(2, 9));
    }

    #[test]
    fn slice() {
        let src = "let x = 42;";
        assert_eq!(Span::new(4, 5).slice(src), "x");
        assert_eq!(Span::new(8, 10).slice(src), "42");
    }
}
