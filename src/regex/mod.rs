//! A small, in-house regular-expression engine (no foreign code).
//!
//! The pattern is parsed to an AST, compiled to a flat instruction list, and
//! executed by a backtracking virtual machine (the classic Thompson/Pike
//! "regex VM" shape — see Russ Cox, *Regular Expression Matching: the Virtual
//! Machine Approach*). Backtracking gives JavaScript's greedy/lazy and
//! capturing-group semantics directly.
//!
//! Supported today: literals, `.`, character classes `[...]` (ranges and the
//! `\d \w \s` shorthands, negation), anchors `^ $`, word boundaries `\b \B`,
//! capturing `( )`, non-capturing `(?: )` and named `(?<name> )` groups,
//! alternation `|`, the quantifiers `* + ? {n} {n,} {n,m}` (greedy and lazy),
//! backreferences `\1`, lookahead `(?= )`/`(?! )`, lookbehind `(?<= )`/`(?<! )`,
//! the Unicode property escapes `\p{…}`/`\P{…}` (under the `u` flag), and the
//! `i` (case-insensitive), `m` (multiline), `s` (dotall), and `y` (sticky) flags.
//! Positions are Unicode scalar (`char`) indices.
//!
//! Property escapes (see the `props` module) follow the ECMAScript grammar: a lone binary
//! property (`\p{Alphabetic}`, `\p{ASCII}`, …), a lone `General_Category` value
//! (`\p{Lu}`, `\p{Letter}`), or a `name=value` form for the non-binary
//! properties `General_Category`/`gc`, `Script`/`sc`, and `Script_Extensions`/
//! `scx` — with canonical names, the standard aliases, and ISO 15924 script
//! short codes. Names/values are validated exactly (an unknown one is a
//! `SyntaxError`). Matching is exact for `General_Category`, `Script`,
//! `Script_Extensions`, and the binary properties expressible from the `intl`
//! tables or a closed form; a handful of binary properties whose UCD tables this
//! crate's `intl` feature set does not include (the emoji family, the
//! `ID`/`Grapheme`/`Ideographic`/… group) are recognised as valid names but
//! match no code point. Without the `intl` feature only the categories/properties
//! expressible via `char` methods resolve. Without the `u` flag, `\p`/`\P` are an
//! Annex B IdentityEscape (the literal `p`/`P`), never a property escape.
//!
//! # Subject model
//!
//! The matching core (the `vm` module) operates on a subject of **UTF-16 code
//! units** (`&[u16]`), so positions are code-unit indices and a lone surrogate
//! is a matchable unit — this is JavaScript string semantics. The `u` (unicode)
//! flag selects code-*point* operation (a surrogate pair is one character, `.`
//! and classes span the whole astral character, `\u{…}` ranges work) while still
//! reporting code-unit indices. The new public entry points `captures_in_u16` /
//! `find_in_u16` take and return code-unit positions.
//!
//! The historical `&str` / `&[char]` entry points (`is_match`, `captures_from`,
//! `captures_in`, …) are preserved as thin **adapters**:
//! they encode the input to UTF-16, run the u16 core, and translate the returned
//! unit indices back to scalar (`char`) indices, so their observable behavior is
//! exactly as before. New callers should migrate to the u16 API.

mod compile;
mod parser;
mod props;
mod props_data;
mod props_emoji;
mod vm;

#[cfg(test)]
mod tests;

use alloc::string::String;
use alloc::vec::Vec;

pub use parser::RegexError;

/// A compiled regular expression.
pub struct Regex {
    /// The program for the native UTF-16 entry points, compiled per `flags`
    /// (astral literals split into surrogate units when the `u` flag is off).
    prog: Vec<vm::Inst>,
    /// A program for the legacy `&str`/`&[char]` adapters, always compiled in
    /// code-point (unicode) mode so each Unicode scalar is one atom — exactly the
    /// pre-UTF-16 behavior those adapters must preserve. Identical to `prog` when
    /// the `u` flag is set, so it is only a distinct allocation for non-`u`
    /// patterns. The adapters run it with code-point reads over the scalar
    /// subject, so an astral `char` matches `.`/a literal atomically as before.
    ///
    /// Built **lazily** on the first adapter call (RE-P2): the interpreter never
    /// uses the `&str`/`&[char]` adapters (it scans via `captures_in_u16` /
    /// `find_in_u16`), so for every real compile this second parse+compile is
    /// skipped entirely. Only the in-crate tests exercise the adapters, and they
    /// pay the cost once per `Regex`. `OnceCell` keeps `Regex` single-threaded
    /// (it lives behind an `Rc`, never shared across threads).
    scalar_prog: core::cell::OnceCell<Vec<vm::Inst>>,
    /// The original pattern source, retained so [`scalar_prog`](Self::scalar_prog)
    /// can be parsed+compiled lazily on first adapter use.
    source: alloc::string::String,
    /// Whether every path through the program passes a `^` assertion before it
    /// can consume input or match — so a non-multiline match can only begin at
    /// one position and the scan must not retry at every offset. See
    /// [`program_is_anchored`].
    anchored: bool,
    /// The one code unit every match must begin with, when the program has one.
    /// A scan can then skip straight to the next occurrence instead of running
    /// the matcher at every offset. See [`program_first_unit`].
    first_unit: Option<u16>,
    /// The Latin-1 code units that can begin a match, when no *single* unit is
    /// required — the class-led case (`/\s+/`, `/[0-9]+/`) that `first_unit`
    /// cannot express. See [`program_start_set`].
    start_set: Option<StartSet>,
    /// Number of capturing groups (excluding the whole-match group 0).
    group_count: usize,
    /// `(group index, name)` pairs for named capture groups (`(?<name>…)`).
    group_names: alloc::vec::Vec<(usize, alloc::string::String)>,
    flags: Flags,
}

/// The active regex flags.
#[derive(Clone, Copy, Default)]
pub struct Flags {
    /// `i` — case-insensitive matching.
    pub ignore_case: bool,
    /// `g` — global (affects iteration by the caller, not a single match).
    pub global: bool,
    /// `m` — multiline (`^`/`$` match at line terminators).
    pub multiline: bool,
    /// `s` — dotall (`.` matches line terminators).
    pub dotall: bool,
    /// `y` — sticky: a match must begin exactly at the start position.
    pub sticky: bool,
    /// `u` — unicode: `.`/classes/quantifiers operate on code points (a surrogate
    /// pair is one character) and astral `\u{…}`/ranges are honored. Positions
    /// are still reported as code-unit indices.
    pub unicode: bool,
    /// `v` — unicodeSets: a stricter superset of `u` enabling extended character
    /// classes (set operations `&&`/`--`, nested classes, `\q{…}` string
    /// literals, properties of strings). Implies code-point (`unicode`) operation.
    pub unicode_sets: bool,
}

impl Flags {
    /// Parses a flags string (e.g. `"gi"`), erroring on unknown flags, on a
    /// repeated flag character, or on the incompatible `u`+`v` combination.
    ///
    /// Per `RegExp.prototype` flag validation (and the regex-literal early error)
    /// each flag character may appear at most once and the `u` (unicode) and `v`
    /// (unicodeSets) flags are mutually exclusive — both are Syntax Errors.
    pub fn parse(s: &str) -> Result<Flags, RegexError> {
        let mut f = Flags::default();
        // Tracks which flag characters have already been seen so a duplicate
        // (e.g. `gg`) is rejected as a Syntax Error.
        let mut seen = [false; 128];
        for c in s.chars() {
            // A flag is always an ASCII letter; an out-of-range char is an unknown
            // flag and a non-ASCII char can never be a valid flag. Index the seen
            // table only for ASCII so the bound is safe.
            let idx = c as usize;
            if idx < 128 {
                if seen[idx] {
                    return Err(RegexError::new(alloc::format!("duplicate flag `{c}`")));
                }
                seen[idx] = true;
            }
            match c {
                'i' => f.ignore_case = true,
                'g' => f.global = true,
                'm' => f.multiline = true,
                's' => f.dotall = true,
                'y' => f.sticky = true,
                'u' => f.unicode = true,
                // `v` (unicodeSets) is a superset of `u`: it turns on code-point
                // operation and additionally enables extended character classes.
                'v' => {
                    f.unicode = true;
                    f.unicode_sets = true;
                }
                'd' => {} // accepted but not yet acted on (hasIndices)
                other => return Err(RegexError::new(alloc::format!("unknown flag `{other}`"))),
            }
        }
        // `u` and `v` may not both be set: `v` is a distinct, stricter mode, not
        // an addition to `u`. (`f.unicode` is implied by `v`, so check the raw
        // presence of both letters.)
        if f.unicode_sets && s.contains('u') {
            return Err(RegexError::new(
                "the `u` and `v` flags are mutually exclusive",
            ));
        }
        Ok(f)
    }
}

/// A single match: the whole-match span plus any capture-group spans (each
/// `Some((start, end))` in `char` indices, or `None` if the group did not
/// participate). Index 0 is the whole match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Captures {
    /// The group spans; `groups[0]` is the whole match.
    pub groups: Vec<Option<(usize, usize)>>,
}

impl Captures {
    /// The whole match span.
    #[must_use]
    pub fn whole(&self) -> (usize, usize) {
        self.groups[0].expect("group 0 is always set on a successful match")
    }

    /// The span of capture group `i` (1-based for user groups), if it
    /// participated.
    #[must_use]
    pub fn group(&self, i: usize) -> Option<(usize, usize)> {
        self.groups.get(i).copied().flatten()
    }
}

impl Regex {
    /// Compiles `pattern` with the given `flags` string.
    pub fn new(pattern: &str, flags: &str) -> Result<Regex, RegexError> {
        let flags = Flags::parse(flags)?;
        // The native u16 program follows the `u` flag (splitting astral literals
        // in non-`u` mode). The scalar-atomic adapter program is built lazily on
        // first `&str`/`&[char]` use (RE-P2) — the interpreter never takes that
        // path, so this is the only compile that runs for real callers.
        let (ast, _, group_names) = parser::parse(pattern, flags.unicode, flags.unicode_sets)?;
        let (prog, group_count) = compile::compile(&ast, &group_names, flags.unicode)?;
        let first_unit = program_first_unit(&prog);
        Ok(Regex {
            anchored: program_is_anchored(&prog),
            // The single-unit filter is a specialized equality scan, so it stays
            // the fast path; the set only covers what it cannot express.
            start_set: if first_unit.is_some() {
                None
            } else {
                program_start_set(&prog, flags)
            },
            first_unit,
            prog,
            scalar_prog: core::cell::OnceCell::new(),
            source: alloc::string::String::from(pattern),
            group_count,
            group_names,
            flags,
        })
    }

    /// The scalar-atomic adapter program, compiled on first use (RE-P2).
    ///
    /// Compiled in unicode (code-point) mode so each Unicode scalar is one atom —
    /// the pre-UTF-16 behavior the `&str`/`&[char]` adapters must preserve. When
    /// the `u` flag is already set this is the same compile as `prog`; we still
    /// build a fresh copy here, but only the first adapter call pays for it.
    fn scalar_prog(&self) -> &[vm::Inst] {
        self.scalar_prog.get_or_init(|| {
            // Parse in the regex's *own* unicode mode (so a pattern accepted in
            // `new` re-parses identically — the `u`-flag syntax strictness must
            // not turn a valid non-`u` pattern into a parse error here), then
            // compile with `unicode = true` for scalar-atomic (code-point) reads,
            // which is what the `&str`/`&[char]` adapters require. The pattern
            // compiled cleanly in `new`, so this cannot fail; fall back to an
            // empty program in the impossible error case rather than panicking.
            let Ok((ast, _, gn)) =
                parser::parse(&self.source, self.flags.unicode, self.flags.unicode_sets)
            else {
                return Vec::new();
            };
            compile::compile(&ast, &gn, true)
                .map(|(sp, _)| sp)
                .unwrap_or_default()
        })
    }

    /// The number of capturing groups (excluding group 0).
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// The `(group index, name)` pairs of named capture groups (`(?<name>…)`).
    #[must_use]
    pub fn group_names(&self) -> &[(usize, alloc::string::String)] {
        &self.group_names
    }

    /// The flags this regex was compiled with.
    #[must_use]
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Whether the pattern matches anywhere in `text`.
    #[must_use]
    pub fn is_match(&self, text: &str) -> bool {
        // Scalar-atomic path (legacy semantics), so an astral char is one atom.
        self.captures_from(text, 0).is_some()
    }

    /// The first match at or after `char` index `start`, with captures (in `char`
    /// indices).
    ///
    /// Collects `text` into a `Vec<char>` on every call. Callers that scan a
    /// subject repeatedly (match/matchAll/replace/split loops) must instead
    /// collect once and use [`captures_in`](Self::captures_in), or the
    /// per-match re-collection makes the whole loop O(n²) (RE-7).
    #[must_use]
    pub fn captures_from(&self, text: &str, start: usize) -> Option<Captures> {
        let chars: Vec<char> = text.chars().collect();
        self.captures_at(&chars, start)
    }

    /// The first match's whole span at or after `start` (char indices).
    #[must_use]
    pub fn find_from(&self, text: &str, start: usize) -> Option<(usize, usize)> {
        self.captures_from(text, start).map(|c| c.whole())
    }

    /// Like [`captures_from`](Self::captures_from), but over a pre-collected
    /// `&[char]` so repeated scans of one subject don't re-collect it each call
    /// (RE-7). `start` is a char offset, as are the returned capture indices.
    #[must_use]
    pub fn captures_in(&self, chars: &[char], start: usize) -> Option<Captures> {
        self.captures_at(chars, start)
    }

    /// Like [`find_from`](Self::find_from), but over a pre-collected `&[char]`
    /// (RE-7). `start` and the returned span are char offsets.
    #[must_use]
    pub fn find_in(&self, chars: &[char], start: usize) -> Option<(usize, usize)> {
        self.captures_at(chars, start).map(|c| c.whole())
    }

    // --- UTF-16 code-unit entry points (the native subject model) ---

    /// The first match at or after code-unit index `start`, with captures
    /// reported as **code-unit** spans. This is the engine's native entry point;
    /// the `&str`/`&[char]` methods are adapters over it.
    #[must_use]
    pub fn captures_in_u16(&self, units: &[u16], start: usize) -> Option<Captures> {
        self.captures_at_u16(units, start)
    }

    /// Like [`captures_in_u16`](Self::captures_in_u16) but returns only the whole
    /// match span (code-unit indices).
    #[must_use]
    pub fn find_in_u16(&self, units: &[u16], start: usize) -> Option<(usize, usize)> {
        self.captures_at_u16(units, start).map(|c| c.whole())
    }

    /// Runs the native u16 program, scanning forward from `start` (unless
    /// sticky). All positions are code-unit indices and the `u` flag drives
    /// code-point vs code-unit semantics.
    fn captures_at_u16(&self, units: &[u16], start: usize) -> Option<Captures> {
        self.scan(&self.prog, units, start, self.flags)
    }

    /// `&[char]` adapter: encodes the scalar subject to UTF-16, runs the
    /// scalar-atomic program ([`scalar_prog`](Self::scalar_prog)) with code-point
    /// reads so each Unicode scalar is one atom (preserving the pre-UTF-16
    /// behavior), then translates the returned code-unit indices back to `char`
    /// (scalar) indices. `start` and the returned spans are `char` offsets.
    fn captures_at(&self, chars: &[char], start: usize) -> Option<Captures> {
        // Build the UTF-16 subject plus a per-char unit offset table, so we can
        // map unit indices back to char indices. `char_to_unit[i]` is the unit
        // index where `chars[i]` begins; the final entry is the total unit len.
        let mut units: Vec<u16> = Vec::with_capacity(chars.len());
        let mut char_to_unit: Vec<usize> = Vec::with_capacity(chars.len() + 1);
        let mut buf = [0u16; 2];
        for &c in chars {
            char_to_unit.push(units.len());
            units.extend_from_slice(c.encode_utf16(&mut buf));
        }
        char_to_unit.push(units.len());
        let unit_start = *char_to_unit.get(start).unwrap_or(&units.len());

        // Run the scalar-atomic program with code-point reads (unicode = true),
        // so an astral subject char is read as one code point and matched against
        // the unsplit literal/`.`/class — exactly the legacy `&[char]` behavior.
        let mut adapter_flags = self.flags;
        adapter_flags.unicode = true;
        let caps = self.scan(self.scalar_prog(), &units, unit_start, adapter_flags)?;

        // Translate every span from unit indices back to char indices. A scalar
        // subject only ever has matches on whole-char boundaries, so each unit
        // index is the start of some char (binary search on the offset table).
        let to_char = |u: usize| -> usize {
            match char_to_unit.binary_search(&u) {
                Ok(i) => i,
                // Should not happen on a scalar subject, but stay total: round
                // down to the enclosing char.
                Err(i) => i.saturating_sub(1),
            }
        };
        let groups = caps
            .groups
            .into_iter()
            .map(|g| g.map(|(s, e)| (to_char(s), to_char(e))))
            .collect();
        Some(Captures { groups })
    }

    /// Scans `prog` forward from unit index `start` (honoring the sticky flag),
    /// sharing one step budget across all start positions (RE-8). All positions
    /// are code-unit indices. Used by both the native u16 path and the legacy
    /// adapters (which differ only in which program and flags they pass).
    fn scan(
        &self,
        prog: &[vm::Inst],
        units: &[u16],
        start: usize,
        flags: Flags,
    ) -> Option<Captures> {
        // A sticky match must begin exactly at `start`. So must an anchored one
        // (`^…`) when `m` is off: `^` holds only at position 0, so if `start` is 0
        // there is exactly one candidate, and if it is not, no position from
        // `start` on can match — and either way one attempt settles it. Without
        // this the scan retried every offset in the subject and let the VM reject
        // each on the leading assertion, making a guaranteed-failing anchored
        // `test()` cost O(subject length) instead of O(1).
        let anchored = self.anchored && !flags.multiline;
        let last = if flags.sticky || anchored {
            start
        } else {
            units.len()
        };
        let steps = core::cell::Cell::new(0u64);
        let budget = vm::budget_for(units.len());
        // When every match must begin with one known code unit, offsets where the
        // subject does not hold it cannot match, so skip to the next occurrence
        // rather than starting the matcher and having it fail on its first step.
        // Under `i` the unit stands for a whole case-folding class, so the filter
        // does not apply.
        let filter = if flags.ignore_case {
            None
        } else {
            self.first_unit
        };
        // One set of working buffers for the whole scan (see `vm::Scratch`).
        let mut scratch = vm::Scratch::default();
        let mut s = start;
        while s <= last {
            if let Some(want) = filter {
                s += units[s..].iter().position(|&u| u == want)?;
                if s > last {
                    return None;
                }
                // In `u` mode a match can only begin on a character boundary, so a
                // hit in the middle of a surrogate pair is not a candidate — step
                // over it and look for the next occurrence.
                if flags.unicode && !is_char_boundary(units, s) {
                    s += 1;
                    continue;
                }
            } else if let Some(set) = &self.start_set {
                // No single required unit, but a known set of possible ones (a
                // class-led pattern): skip offsets the set rules out.
                s += units[s..].iter().position(|&u| set.allows(u))?;
                if s > last {
                    return None;
                }
                if flags.unicode && !is_char_boundary(units, s) {
                    s += 1;
                    continue;
                }
            }
            if let Some(groups) = vm::run_with(
                &mut scratch,
                prog,
                units,
                s,
                self.group_count,
                flags,
                &steps,
                budget,
            ) {
                return Some(Captures { groups });
            }
            s = advance_string_index(units, s, flags.unicode);
        }
        None
    }

    /// Replaces matches in `text` with `replacement`, honoring the `g` flag
    /// (otherwise only the first match). `$&` in the replacement inserts the
    /// whole match and `$1`..`$9` insert capture groups.
    #[must_use]
    pub fn replace(&self, text: &str, replacement: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::new();
        let mut pos = 0;
        while pos <= chars.len() {
            let Some(caps) = self.captures_at(&chars, pos) else {
                break;
            };
            let (ms, me) = caps.whole();
            out.extend(&chars[pos..ms]);
            expand_replacement(replacement, &chars, &caps, &mut out);
            if me > ms {
                pos = me;
            } else {
                // Empty match: emit one char and advance so we make progress.
                if me < chars.len() {
                    out.push(chars[me]);
                }
                pos = me + 1;
            }
            if !self.flags.global {
                break;
            }
        }
        out.extend(&chars[pos.min(chars.len())..]);
        out
    }
}

/// `AdvanceStringIndex(S, index, fullUnicode)`: the next position a match may
/// begin at after a failed attempt at `s`. With `u` set a surrogate pair is one
/// character, so the scan steps over both units and never starts inside a pair.
fn advance_string_index(units: &[u16], s: usize, unicode: bool) -> usize {
    if unicode
        && let Some(&hi) = units.get(s)
        && (0xD800..=0xDBFF).contains(&hi)
        && units
            .get(s + 1)
            .is_some_and(|lo| (0xDC00..=0xDFFF).contains(lo))
    {
        return s + 2;
    }
    s + 1
}

/// Whether unit index `s` is the start of a character — false only for the low
/// half of a surrogate pair. Used in `u` mode, where a match may not begin inside
/// a pair.
fn is_char_boundary(units: &[u16], s: usize) -> bool {
    s == 0 || !(0xDC00..=0xDFFF).contains(&units[s]) || !(0xD800..=0xDBFF).contains(&units[s - 1])
}

/// Whether `prog` can only match at the position the scan starts from: every
/// path from the entry reaches a `^` assertion before it consumes input or
/// matches.
///
/// Walks the control-flow graph, following `Jmp`/`Split` and passing through the
/// zero-width bookkeeping instructions. Reaching a consuming instruction — or
/// `Match` — without having passed `Assert::Start` proves the pattern can match
/// somewhere other than the start, so the answer is `false`. Anything unhandled
/// answers `false` too, which only costs the optimization.
///
/// Deliberately conservative in three places: an `Assert::Start` *inside* a
/// lookaround does not count (the lookaround is zero-width and may be negated),
/// a revisited instruction contributes no new path (any real counterexample is
/// found on some first visit), and an inline-modifier scope gives up entirely
/// since `(?m:^)` would make the assertion match at every line start.
///
/// The result holds only when the `m` flag is off — the caller checks that.
fn program_is_anchored(prog: &[vm::Inst]) -> bool {
    use vm::{Assert, Inst};
    let mut seen = alloc::vec![false; prog.len()];
    let mut stack = alloc::vec![0usize];
    while let Some(pc) = stack.pop() {
        let Some(inst) = prog.get(pc) else {
            return false;
        };
        if core::mem::replace(&mut seen[pc], true) {
            continue;
        }
        match inst {
            // Pinned to the subject start: this path cannot match anywhere else.
            Inst::Assert(Assert::Start) => {}
            Inst::Jmp(t) => stack.push(*t),
            Inst::Split(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            // Zero-width and position-agnostic: keep looking for the `^`.
            Inst::Save(_)
            | Inst::ClearCaptures { .. }
            | Inst::Mark
            | Inst::EmptyFail
            | Inst::Assert(Assert::End | Assert::WordBoundary | Assert::NotWordBoundary)
            | Inst::Look { .. }
            | Inst::LookBehind { .. } => stack.push(pc + 1),
            // Never matches, so this path proves nothing.
            Inst::Fail => {}
            // Consumes input, matches, or changes the meaning of `^`.
            _ => return false,
        }
    }
    true
}

/// The single code unit every match of `prog` must start with, or `None` when
/// there is no such unit.
///
/// Same walk as [`program_is_anchored`]: follow control flow to the first
/// input-consuming instruction on each path. If every path reaches the *same*
/// `Inst::Char`, that unit is required and a scan can skip offsets without it.
/// Anything else — two different literals, a class, `.`, a backreference, or a
/// reachable `Match` (the pattern matches empty) — means no single unit is
/// required.
///
/// Astral literals are excluded: in `u` mode one `Char` spans a surrogate pair,
/// and filtering on the high surrogate alone would be a different predicate than
/// the one the matcher applies. Only costs the optimization.
///
/// The result holds only when `i` is off — the caller checks that, since under
/// case folding the literal stands for a class rather than one unit.
/// The set of Latin-1 code units that can begin a match.
///
/// `first_unit` only fires when *one* literal is required, so a class-led
/// pattern (`/\s+/`) filtered nothing and started the matcher at every offset —
/// which is what made `"word ".repeat(120_000).split(/\s+/)` ~95× slower than
/// node. This covers the class case with a 256-bit map.
///
/// Only units below 256 are represented. A code unit at or above 256 may be a
/// lone surrogate, or the lead of a pair standing for an astral code point, so
/// deciding it would mean reasoning about the decode as well as the class;
/// those units are conservatively always allowed. That costs nothing on the
/// ASCII-dominated subjects this exists to speed up, and keeps the filter a
/// pure subset of what the matcher accepts.
struct StartSet {
    /// Bit `u` set iff code unit `u` can begin a match.
    latin1: [u64; 4],
}

impl StartSet {
    fn allows(&self, unit: u16) -> bool {
        // Units >= 256 are not represented and are always candidates.
        unit >= 256 || self.latin1[usize::from(unit >> 6)] & (1u64 << (unit & 63)) != 0
    }
}

/// Builds the [`StartSet`] for a program, or `None` when no useful filter
/// exists (a reachable `Match`, a backreference, or a first instruction that
/// accepts every Latin-1 unit anyway).
///
/// Walks the same CFG as [`program_first_unit`], but instead of demanding one
/// shared literal it unions the accepted units of every first-consuming
/// instruction on any path. Membership comes from
/// [`vm::single_consume_matches`] — the matcher's own predicate — so the filter
/// cannot disagree with it and skip an offset that would have matched. That
/// also means `i` needs no special case here: the folding is inside the
/// predicate, unlike the raw equality `first_unit` does.
fn program_start_set(prog: &[vm::Inst], flags: Flags) -> Option<StartSet> {
    use vm::Inst;
    let mut set = StartSet { latin1: [0; 4] };
    let mut seen = alloc::vec![false; prog.len()];
    let mut stack = alloc::vec![0usize];
    while let Some(pc) = stack.pop() {
        let inst = prog.get(pc)?;
        if core::mem::replace(&mut seen[pc], true) {
            continue;
        }
        match inst {
            Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::ClassSet { .. } => {
                for u in 0u32..256 {
                    if vm::single_consume_matches(inst, u, flags) {
                        set.latin1[(u >> 6) as usize] |= 1u64 << (u & 63);
                    }
                }
            }
            Inst::Jmp(t) => stack.push(*t),
            Inst::Split(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            // Zero-width: whatever a match must start with is still ahead.
            Inst::Save(_)
            | Inst::ClearCaptures { .. }
            | Inst::Mark
            | Inst::EmptyFail
            | Inst::Assert(_)
            | Inst::Look { .. }
            | Inst::LookBehind { .. } => stack.push(pc + 1),
            // Never matches: contributes no path.
            Inst::Fail => {}
            // A reachable `Match` means the empty string matches anywhere, and a
            // backreference's text is not known until run time.
            _ => return None,
        }
    }
    // The filter costs a bitmap test per offset, so it only pays if it rejects
    // a worthwhile share of them. A `.`-led pattern accepts all but the line
    // terminators — technically a filter, practically pure overhead. Require it
    // to rule out at least a quarter of the Latin-1 range.
    let accepted: u32 = set.latin1.iter().map(|w| w.count_ones()).sum();
    (accepted <= 192).then_some(set)
}

fn program_first_unit(prog: &[vm::Inst]) -> Option<u16> {
    use vm::Inst;
    let mut seen = alloc::vec![false; prog.len()];
    let mut stack = alloc::vec![0usize];
    let mut required: Option<u16> = None;
    while let Some(pc) = stack.pop() {
        let inst = prog.get(pc)?;
        if core::mem::replace(&mut seen[pc], true) {
            continue;
        }
        match inst {
            // The first thing this path consumes. Every path must agree.
            Inst::Char(c) => {
                let unit = u16::try_from(*c).ok()?;
                match required {
                    None => required = Some(unit),
                    Some(prev) if prev == unit => {}
                    Some(_) => return None,
                }
            }
            Inst::Jmp(t) => stack.push(*t),
            Inst::Split(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            // Zero-width: the required unit, if any, is still ahead.
            Inst::Save(_)
            | Inst::ClearCaptures { .. }
            | Inst::Mark
            | Inst::EmptyFail
            | Inst::Assert(_)
            | Inst::Look { .. }
            | Inst::LookBehind { .. } => stack.push(pc + 1),
            // Never matches: contributes no path, so no required unit either.
            Inst::Fail => {}
            _ => return None,
        }
    }
    required
}

/// Expands a replacement template (`$&`, `$1`..`$9`) for one match.
fn expand_replacement(template: &str, chars: &[char], caps: &Captures, out: &mut String) {
    let t: Vec<char> = template.chars().collect();
    let mut i = 0;
    while i < t.len() {
        if t[i] == '$' && i + 1 < t.len() {
            match t[i + 1] {
                '&' => {
                    let (s, e) = caps.whole();
                    out.extend(&chars[s..e]);
                    i += 2;
                    continue;
                }
                d @ '1'..='9' => {
                    let idx = d as usize - '0' as usize;
                    if let Some((s, e)) = caps.group(idx) {
                        out.extend(&chars[s..e]);
                    }
                    i += 2;
                    continue;
                }
                '$' => {
                    out.push('$');
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }
        out.push(t[i]);
        i += 1;
    }
}
