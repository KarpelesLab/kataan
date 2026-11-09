//! Rope strings — lazy-concatenation string values (`ROADMAP.md` §3, item 4).
//!
//! Building a string by repeated concatenation (`s += chunk` in a loop, a
//! `join`, a template assembled piecewise) is quadratic with a flat buffer:
//! each `+` copies the whole left side. A **rope** stores a concatenation as a
//! small tree of shared segments, so joining two ropes is O(1) — it allocates
//! one interior node pointing at both sides — and the characters are only
//! copied once, when the final string is materialized.
//!
//! `Rope` is a cheap-to-clone handle (`Rc` inside). Concatenation is O(1) and
//! length is O(1) (cached per node); materializing flattens to a `String`
//! in O(n) with an explicit work stack, so even a deeply left-nested rope — the
//! exact shape `s += x` in a loop produces — flattens without recursing.
//!
//! Pure, safe `alloc`-only Rust.
//!
//! # Small strings
//!
//! A flat string of at most [`SMALL_CAP`](crate::rope::SMALL_CAP) bytes is stored *inline* in its node
//! (`Node::Small`) instead of in a separately allocated `Box<[u8]>`: the node
//! is the size of a `Concat` anyway, so the bytes ride in space that would
//! otherwise be padding, and creating such a string costs one allocation
//! instead of two. Most strings a program makes (keys, numbers, words, single
//! characters) are this short.
//!
//! # Storage: WTF-8 bytes
//!
//! Leaves hold **WTF-8 bytes** (`Box<[u8]>`), not `Box<str>`, so a string may
//! carry lone UTF-16 surrogates (DOMString semantics — see [`crate::wtf8`]). A
//! string with no surrogates is byte-identical to its UTF-8, so the common case
//! is unchanged. `Rope::from`/`Rope::leaf` store a `&str`'s (valid-WTF-8)
//! bytes; `Rope::from_wtf8`/`Rope::from_bytes` take surrogate-bearing bytes.
//! `Rope::materialize_bytes` returns the WTF-8 bytes; `Rope::materialize`
//! and `Display` stay `String`-typed and are **lossy** (surrogates → U+FFFD),
//! so existing `&str`/`String` callers compile and behave unchanged for
//! non-surrogate strings. Surrogate-correct accessors land in a later milestone.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::wtf8;

/// Maximum length, in bytes, of a string the engine will build by
/// concatenation. A `+` whose result would exceed this is a `RangeError`
/// ("Invalid string length") rather than a `usize` overflow / OOM bomb.
pub const MAX_STRING_LEN: usize = crate::limits::DEFAULT_MAX_STRING_LEN;

/// The identity of a rope's root node that does not keep its contents alive
/// (a weak reference): a cache keyed on "the same string as last time" can
/// hold one without pinning a huge string. While it exists the node's
/// allocation is not reused, so [`RopeId::is`] never mistakes a later rope
/// for the dropped one.
pub struct RopeId(alloc::rc::Weak<Node>);

impl RopeId {
    /// Whether `rope` is the rope this id was taken from.
    #[must_use]
    pub fn is(&self, rope: &Rope) -> bool {
        core::ptr::eq(self.0.as_ptr(), Rc::as_ptr(&rope.0))
    }
}

/// A string value represented as a tree of concatenated segments. Cloning is a
/// reference-count bump; concatenation and length are O(1).
#[derive(Clone)]
pub struct Rope(Rc<Node>);

enum Node {
    /// A short contiguous run of WTF-8 bytes held inline: `buf[..len]`, with
    /// its UTF-16 length memoized in `u16` on first use (`SMALL_U16_UNKNOWN`
    /// until then). The bytes past `len` are always zero (every constructor
    /// starts from a zeroed buffer): compiled code reads a short key's text
    /// as whole words of `buf` (`crate::jit::key_probe`).
    Small {
        len: u8,
        u16: core::cell::Cell<u8>,
        buf: [u8; SMALL_CAP],
    },
    /// A contiguous run of WTF-8 bytes, with its UTF-16 length memoized on
    /// first use (see [`Rope::utf16_len`]).
    Leaf(Box<[u8]>, core::cell::Cell<usize>),
    /// A contiguous run of another string's bytes: `base`'s bytes
    /// `[start, start + len)`, shared rather than copied (see
    /// [`Rope::slice_units`]). `base` is always a `Leaf` or a `Build`, so a
    /// slice of a slice points at the original buffer and reads stay one hop. The
    /// UTF-16 length is memoized like a leaf's.
    Slice {
        base: Rope,
        start: usize,
        len: usize,
        u16_len: core::cell::Cell<usize>,
    },
    /// The concatenation of two ropes, with the total byte length cached and the
    /// UTF-16 length memoized on first use.
    Concat {
        left: Rope,
        right: Rope,
        len: usize,
        u16_len: core::cell::Cell<usize>,
    },
    /// The first `len` bytes of a shared append buffer (see [`BuildBuf`]):
    /// a flat string that a `s += x` loop extends in place, with its UTF-16
    /// length memoized like a leaf's.
    Build {
        buf: Rc<BuildBuf>,
        len: usize,
        u16_len: core::cell::Cell<usize>,
    },
}

/// The append-only byte buffer behind [`Node::Build`] ropes.
///
/// `s += x` in a loop makes each new string by appending to the last one. A
/// tree of `Concat` nodes makes that O(1) per step, but the result is a long
/// left spine that every later read walks and that has to be flattened (a
/// full copy) before most uses. Here the strings of such a chain share one
/// buffer instead: each holds a prefix of it (`len` bytes), and appending to
/// the string that ends where the buffer's written part ends (`used`) writes
/// the new bytes into the spare capacity behind it and makes a string of the
/// longer prefix. Bytes below `used` are never written again, so every
/// string's text is fixed; an append to any other string (an older prefix)
/// or one that does not fit goes elsewhere (a fresh, larger buffer or a
/// `Concat`). Growth doubles the capacity, so a loop's copying is amortized
/// O(1) per appended byte, and every string of the chain is flat: its
/// length, indexed reads and slices need no walk and no flattening.
pub(crate) struct BuildBuf {
    /// How many leading bytes of `data` are written (and immutable).
    used: core::cell::Cell<usize>,
    /// The buffer; `data.len()` is its capacity.
    data: Box<[core::cell::UnsafeCell<u8>]>,
}

impl BuildBuf {
    /// A buffer of capacity `cap` holding `a` then `b` (`a.len() + b.len()
    /// <= cap`).
    fn with(cap: usize, a: &[u8], b: &[u8]) -> Rc<BuildBuf> {
        let used = a.len() + b.len();
        debug_assert!(used <= cap);
        let mut v: Vec<u8> = Vec::with_capacity(cap);
        v.extend_from_slice(a);
        v.extend_from_slice(b);
        v.resize(cap, 0);
        let raw = Box::into_raw(v.into_boxed_slice()) as *mut [core::cell::UnsafeCell<u8>];
        // SAFETY: `UnsafeCell<u8>` is `repr(transparent)` over `u8`, so the
        // boxed slice's layout (and its allocation's) is unchanged.
        #[allow(unsafe_code)]
        let data = unsafe { Box::from_raw(raw) };
        Rc::new(BuildBuf {
            used: core::cell::Cell::new(used),
            data,
        })
    }

    /// The first `len` bytes (`len <= used`).
    #[inline]
    fn bytes(&self, len: usize) -> &[u8] {
        let len = len.min(self.used.get());
        // SAFETY: `UnsafeCell<u8>` is `repr(transparent)` over `u8`, and the
        // bytes below `used` are never written again (`append_at` writes only
        // at and past `used`), so nothing mutates what this borrow covers.
        #[allow(unsafe_code)]
        unsafe {
            core::slice::from_raw_parts(
                core::cell::UnsafeCell::raw_get(self.data.as_ptr()).cast_const(),
                len,
            )
        }
    }

    /// Appends `parts` at `at` when `at` is where the written part ends and
    /// they fit the spare capacity: `true` then (`used` moves past them).
    #[inline]
    fn append_at(&self, at: usize, parts: &[&[u8]]) -> bool {
        let n: usize = parts.iter().map(|p| p.len()).sum();
        if at != self.used.get() || self.data.len() - at < n {
            return false;
        }
        let mut to = at;
        for p in parts {
            // SAFETY: `[to, to + p.len())` is inside `data` (checked above)
            // and at or past `used`, which no borrow from `bytes` reaches; `p`
            // is some string's bytes, which lie below its own buffer's `used`
            // (or in another allocation), so the ranges are disjoint.
            #[allow(unsafe_code)]
            unsafe {
                core::ptr::copy_nonoverlapping(
                    p.as_ptr(),
                    core::cell::UnsafeCell::raw_get(self.data.as_ptr().add(to)),
                    p.len(),
                );
            }
            to += p.len();
        }
        self.used.set(to);
        true
    }

    /// [`append_at`](Self::append_at) of one short inline string, `buf[..n]`
    /// of a [`Node::Small`]'s buffer: when the spare capacity holds the whole
    /// buffer it is copied as one fixed-size block (no length-dependent copy
    /// call for a two-byte append). The bytes past the appended `n` land in
    /// spare capacity, which nothing reads (`bytes` stops at `used`).
    #[inline]
    fn append_small(&self, at: usize, buf: &[u8; SMALL_CAP], n: usize) -> bool {
        let cap = self.data.len();
        if at != self.used.get() || cap - at < n {
            return false;
        }
        if cap - at >= SMALL_CAP {
            // SAFETY: `[at, at + SMALL_CAP)` is inside `data` (checked) and at
            // or past `used`, which no borrow from `bytes` reaches; `buf` is
            // another node's inline array, so the ranges are disjoint. (A
            // whole-array store: a fixed-size copy, which the compiler inlines.)
            #[allow(unsafe_code)]
            unsafe {
                core::ptr::write_unaligned(
                    core::cell::UnsafeCell::raw_get(self.data.as_ptr().add(at))
                        .cast::<[u8; SMALL_CAP]>(),
                    *buf,
                );
            }
            self.used.set(at + n);
            return true;
        }
        self.append_tail(at, &buf[..n])
    }

    /// [`append_small`](Self::append_small) near the end of the buffer: an
    /// exact copy. (Out of line, so that the fixed-size copy above is not
    /// merged with this one into a call.)
    #[cold]
    #[inline(never)]
    fn append_tail(&self, at: usize, b: &[u8]) -> bool {
        self.append_at(at, &[b])
    }
}

/// The longest flat left side [`Rope::try_concat`] copies into a new append
/// buffer to start a [`Node::Build`] chain (a longer one is shared by a
/// `Concat` instead, so `big + x` made over and over never copies `big`).
const BUILD_SEED_MAX: usize = 1024;

/// The longest right side that starts a [`Node::Build`] chain (with a left
/// side of at most [`BUILD_SEED_MAX`]): a new chain is a guess that more
/// appends follow, so what it copies up front stays small.
const BUILD_SEED_RIGHT_MAX: usize = 256;

/// Right sides up to this long are always copied into an append buffer; a
/// longer one only when it is no longer than the left side (so the copy is
/// amortized by the doubling, not a duplicate of a huge shared string).
const BUILD_APPEND_MIN: usize = 4096;

/// The smallest capacity of a new append buffer.
const BUILD_MIN_CAP: usize = 128;

/// Byte offsets within a rope's node, from the pointer a [`Rope`] holds, for
/// compiled code reading a flat string (`jit::props`). Probed on live nodes.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RopeLayout {
    /// The node's tag byte.
    pub(crate) tag: i32,
    /// That byte in a [`Node::Leaf`].
    pub(crate) leaf_tag: u8,
    /// A leaf's memoized UTF-16 length ([`U16_LEN_UNKNOWN`] until first asked).
    pub(crate) leaf_u16: i32,
    /// A leaf's byte pointer and byte length.
    pub(crate) leaf_ptr: i32,
    pub(crate) leaf_len: i32,
    /// The tag byte of a [`Node::Small`], and its memoized UTF-16 length (a
    /// byte, [`SMALL_U16_UNKNOWN`] until first asked).
    pub(crate) small_tag: u8,
    pub(crate) small_u16: i32,
    /// A [`Node::Small`]'s byte length (a byte) and its inline bytes.
    pub(crate) small_len: i32,
    pub(crate) small_buf: i32,
    /// The tag byte of a [`Node::Concat`], and its memoized UTF-16 length
    /// ([`U16_LEN_UNKNOWN`] until first asked).
    pub(crate) concat_tag: u8,
    pub(crate) concat_u16: i32,
    /// The tag byte of a [`Node::Build`], its byte length, its memoized
    /// UTF-16 length ([`U16_LEN_UNKNOWN`] until first asked) and its buffer's
    /// pointer word (an `Rc`'s, as the node holds it).
    pub(crate) build_tag: u8,
    pub(crate) build_len: i32,
    pub(crate) build_u16: i32,
    pub(crate) build_buf: i32,
    /// The buffer's byte pointer, from that pointer word's value.
    pub(crate) build_data: i32,
}

/// Derives [`RopeLayout`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub(crate) fn jit_rope_layout() -> RopeLayout {
    let long = [b'x'; SMALL_CAP + 9];
    let leaf = Rope::from_bytes(&long);
    // (Longer than an append buffer copies, so the two make a `Concat`.)
    let other = Rope::from_bytes(&[b'y'; BUILD_APPEND_MIN + 1]);
    let small = Rope::from_bytes(b"s");
    let concat = leaf.concat(&other);
    let value = Rc::as_ptr(&leaf.0) as usize;
    // SAFETY: a `Rope` is exactly the `Rc`'s one word, in bounds of the local.
    #[allow(unsafe_code)]
    let raw = unsafe { *core::ptr::addr_of!(leaf).cast::<usize>() };
    let base = i32::try_from(value - raw).expect("an Rc's value offset");
    // SAFETY: the first byte of each live node; `Node`'s tag lives there (the
    // assertions below reject any other layout).
    #[allow(unsafe_code)]
    let tag_of = |r: &Rope| unsafe { *Rc::as_ptr(&r.0).cast::<u8>() };
    let leaf_tag = tag_of(&leaf);
    assert!(
        tag_of(&other) == leaf_tag && tag_of(&small) != leaf_tag && tag_of(&concat) != leaf_tag,
        "a rope node's tag must be its first byte"
    );
    let Node::Leaf(bytes, memo) = &*leaf.0 else {
        unreachable!("longer than SMALL_CAP")
    };
    let boxed = core::ptr::addr_of!(*bytes) as usize;
    // SAFETY: the two words of the live `Box<[u8]>`, in bounds.
    #[allow(unsafe_code)]
    let (w0, w1) = unsafe {
        (
            *(boxed as *const usize),
            *((boxed + core::mem::size_of::<usize>()) as *const usize),
        )
    };
    let off = |a: usize| base + i32::try_from(a - value).expect("a node field's offset");
    let (ptr_at, len_at) = if w0 == bytes.as_ptr() as usize && w1 == bytes.len() {
        (boxed, boxed + core::mem::size_of::<usize>())
    } else {
        assert!(w1 == bytes.as_ptr() as usize && w0 == bytes.len());
        (boxed + core::mem::size_of::<usize>(), boxed)
    };
    // The other flat-length memos, at their own nodes' offsets (each from its
    // node's value, as `off` is from the leaf's).
    let field_off = |r: &Rope, a: usize| {
        base + i32::try_from(a - Rc::as_ptr(&r.0) as usize).expect("a node field's offset")
    };
    let Node::Small {
        u16: small_memo,
        len: small_len,
        buf: small_buf,
    } = &*small.0
    else {
        unreachable!("shorter than SMALL_CAP")
    };
    let Node::Concat {
        u16_len: concat_memo,
        ..
    } = &*concat.0
    else {
        unreachable!("two leaves concatenate into a concat node")
    };
    let small_tag = tag_of(&small);
    let concat_tag = tag_of(&concat);
    assert!(small_tag != concat_tag, "distinct rope node tags");
    // An append buffer's string: a short leaf and more.
    let build = Rope::from_bytes(&[b'z'; SMALL_JOIN + 1]).concat(&small);
    let Node::Build {
        buf,
        len: build_len,
        u16_len: build_memo,
    } = &*build.0
    else {
        unreachable!("a short leaf and more start an append buffer")
    };
    let build_tag = tag_of(&build);
    assert!(
        build_tag != leaf_tag && build_tag != small_tag && build_tag != concat_tag,
        "distinct rope node tags"
    );
    let buf_word = core::ptr::addr_of!(*buf) as usize;
    // SAFETY: the live `Rc`'s one word, in bounds of the node.
    #[allow(unsafe_code)]
    let buf_raw = unsafe { *(buf_word as *const usize) };
    let data = core::ptr::addr_of!(buf.data) as usize;
    let data_ptr = buf.bytes(0).as_ptr() as usize;
    // SAFETY: the two words of the live boxed slice, in bounds.
    #[allow(unsafe_code)]
    let (d0, d1) = unsafe {
        (
            *(data as *const usize),
            *((data + core::mem::size_of::<usize>()) as *const usize),
        )
    };
    let data_at = if d0 == data_ptr {
        data
    } else {
        assert!(d1 == data_ptr && d0 == buf.data.len());
        data + core::mem::size_of::<usize>()
    };
    RopeLayout {
        tag: base,
        leaf_tag,
        leaf_u16: off(core::ptr::addr_of!(*memo) as usize),
        leaf_ptr: off(ptr_at),
        leaf_len: off(len_at),
        small_tag,
        small_u16: field_off(&small, core::ptr::addr_of!(*small_memo) as usize),
        small_len: field_off(&small, core::ptr::addr_of!(*small_len) as usize),
        small_buf: field_off(&small, core::ptr::addr_of!(*small_buf) as usize),
        concat_tag,
        concat_u16: field_off(&concat, core::ptr::addr_of!(*concat_memo) as usize),
        build_tag,
        build_len: field_off(&build, core::ptr::addr_of!(*build_len) as usize),
        build_u16: field_off(&build, core::ptr::addr_of!(*build_memo) as usize),
        build_buf: field_off(&build, buf_word),
        build_data: i32::try_from(data_at - buf_raw).expect("an Rc's field offset"),
    }
}

/// The longest flat string stored inline in its node (see the module docs).
/// Chosen so that `Node::Small` (a tag, a length byte and the buffer) is no
/// larger than `Node::Concat`, keeping every node the same size.
pub const SMALL_CAP: usize = 37;

/// `Node::Small`'s "UTF-16 length not computed yet" marker (a real one is at
/// most [`SMALL_CAP`]).
const SMALL_U16_UNKNOWN: u8 = u8::MAX;

/// `"00" "01" … "99"`: the two decimal digits of each value below 100.
const DIGIT_PAIRS: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[2 * i] = b'0' + (i / 10) as u8;
        t[2 * i + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

/// `DIGIT_PAIRS` as little-endian words: the text of `i` (below 100) as
/// its two bytes in memory order.
const DIGIT_WORDS: [u16; 100] = {
    let mut t = [0u16; 100];
    let mut i = 0;
    while i < 100 {
        t[i] = DIGIT_PAIRS[2 * i] as u16 | (DIGIT_PAIRS[2 * i + 1] as u16) << 8;
        i += 1;
    }
    t
};

/// The decimal digits of `u` (below 10^16) packed into a word in memory
/// order (the first digit in the lowest byte, zero past the last), and
/// their count. Built from the right, two digits per division, in a
/// register: stored with one 16-byte write.
#[inline]
fn digits_le(mut u: u64) -> (u128, usize) {
    debug_assert!(u < 10_000_000_000_000_000);
    let mut acc: u128 = 0;
    let mut n = 0;
    while u >= 100 {
        acc = (acc << 16) | u128::from(DIGIT_WORDS[(u % 100) as usize]);
        u /= 100;
        n += 2;
    }
    if u >= 10 {
        acc = (acc << 16) | u128::from(DIGIT_WORDS[u as usize]);
        n += 2;
    } else {
        acc = (acc << 8) | u128::from(b'0' + u as u8);
        n += 1;
    }
    (acc, n)
}

/// Writes the decimal text of `i` (magnitude below 2^53; `n` digits packed
/// in `acc` by [`digits_le`]) at `out[at..]`, which has room for 17 bytes
/// past `at`; the bytes after the text that the write covers become zero.
#[inline]
fn put_int(out: &mut [u8], at: usize, i: i64, acc: u128) {
    let mut at = at;
    if i < 0 {
        out[at] = b'-';
        at += 1;
    }
    out[at..at + 16].copy_from_slice(&acc.to_le_bytes());
}

impl Node {
    /// A flat node holding a copy of `bytes`: inline when short.
    #[inline]
    fn flat(bytes: &[u8]) -> Node {
        if bytes.len() <= SMALL_CAP {
            let mut buf = [0u8; SMALL_CAP];
            buf[..bytes.len()].copy_from_slice(bytes);
            Node::Small {
                len: bytes.len() as u8,
                u16: core::cell::Cell::new(SMALL_U16_UNKNOWN),
                buf,
            }
        } else {
            Node::Leaf(Box::from(bytes), core::cell::Cell::new(U16_LEN_UNKNOWN))
        }
    }

    /// The bytes of a flat node (any but a `Concat`), or `None` for a
    /// `Concat`. (The two commonest kinds inline, the others out of line.)
    #[inline]
    fn flat_bytes(&self) -> Option<&[u8]> {
        match self {
            Node::Small { len, buf, .. } => Some(&buf[..usize::from(*len)]),
            Node::Leaf(s, _) => Some(s),
            _ => self.flat_bytes_other(),
        }
    }

    /// [`flat_bytes`](Self::flat_bytes) of a `Slice`, a `Build` or a `Concat`.
    #[inline(never)]
    fn flat_bytes_other(&self) -> Option<&[u8]> {
        match self {
            Node::Small { len, buf, .. } => Some(&buf[..usize::from(*len)]),
            Node::Leaf(s, _) => Some(s),
            Node::Slice {
                base, start, len, ..
            } => match &*base.0 {
                Node::Leaf(s, _) => s.get(*start..*start + *len),
                Node::Build { buf, len: blen, .. } => buf.bytes(*blen).get(*start..*start + *len),
                _ => None,
            },
            Node::Build { buf, len, .. } => Some(buf.bytes(*len)),
            Node::Concat { .. } => None,
        }
    }
}

/// The shortest result of [`Rope::slice_units`] that shares its receiver's
/// buffer: anything that fits a `Small` node is copied into one.
const SLICE_SHARE_MIN: usize = SMALL_CAP + 1;

/// [`Rope::slice_units`] shares the receiver's buffer only when the result is
/// at least `1 / SLICE_SHARE_RATIO` of that buffer, which bounds how much a
/// kept slice can pin: a short piece of a huge string is copied (its cost is
/// proportional to the piece anyway), so the huge one can still be freed.
const SLICE_SHARE_RATIO: usize = 16;

/// The most `Concat` nodes [`Rope::slice_units_tree`] visits before leaving
/// the slice to a flatten.
const TREE_SLICE_STEPS: usize = 64;

/// The most `Concat` nodes [`Rope::unit_at_tree`] descends through before
/// leaving the read to a flatten: a string read in a loop after a few
/// appends stays a short walk per read.
const UNIT_WALK_STEPS: usize = 32;

/// The longest non-ASCII leaf [`Rope::unit_at_tree`] scans for a unit.
const UNIT_WALK_SCAN: usize = 256;

/// Sentinel for "the UTF-16 length has not been computed yet". A real length is
/// bounded by [`MAX_STRING_LEN`], so it can never collide.
const U16_LEN_UNKNOWN: usize = usize::MAX;

/// The longest result (in bytes) of a concatenation of two leaves that is
/// stored as one new leaf rather than a `Concat` node.
const SMALL_JOIN: usize = 64;

impl Rope {
    /// An empty rope.
    #[must_use]
    pub fn new() -> Self {
        Self::leaf("")
    }

    /// A leaf rope holding `s` (stored as its UTF-8 bytes, which are valid
    /// WTF-8).
    #[must_use]
    pub fn leaf(s: &str) -> Self {
        Rope(Rc::new(Node::flat(s.as_bytes())))
    }

    /// A leaf rope holding raw WTF-8 `bytes` — the surrogate-bearing path. The
    /// bytes are taken as-is (assumed already well-formed WTF-8, e.g. from
    /// [`crate::wtf8::from_utf16`] or another rope's `Rope::materialize_bytes`).
    #[must_use]
    pub fn from_wtf8(bytes: Vec<u8>) -> Self {
        if bytes.len() <= SMALL_CAP {
            return Rope(Rc::new(Node::flat(&bytes)));
        }
        Rope(Rc::new(Node::Leaf(
            bytes.into_boxed_slice(),
            core::cell::Cell::new(U16_LEN_UNKNOWN),
        )))
    }

    /// A flat rope of `len` bytes, at most [`SMALL_CAP`], that `fill` writes
    /// straight into its inline buffer (`fill` must leave canonical WTF-8);
    /// `None` when `len` is too long to be held inline.
    #[must_use]
    #[inline]
    pub fn small_with(len: usize, fill: impl FnOnce(&mut [u8])) -> Option<Self> {
        if len > SMALL_CAP {
            return None;
        }
        let mut buf = [0u8; SMALL_CAP];
        fill(&mut buf[..len]);
        Some(Rope(Rc::new(Node::Small {
            len: len as u8,
            u16: core::cell::Cell::new(SMALL_U16_UNKNOWN),
            buf,
        })))
    }

    /// A leaf rope copying raw WTF-8 `bytes`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Rope(Rc::new(Node::flat(bytes)))
    }

    /// [`from_bytes`](Self::from_bytes) of bytes the caller knows are all
    /// ASCII: the UTF-16 length (the byte length) is recorded up front, so a
    /// later `.length` or indexed read needs no scan.
    #[must_use]
    pub fn from_ascii_bytes(bytes: &[u8]) -> Self {
        debug_assert!(bytes.is_ascii());
        let node = Node::flat(bytes);
        Self::with_known_utf16(node, bytes.len())
    }

    /// A flat `node` with its UTF-16 length memo set to `units`.
    #[inline]
    fn with_known_utf16(node: Node, units: usize) -> Self {
        match &node {
            Node::Small { u16, .. } => u16.set(units as u8),
            Node::Leaf(_, cell) => cell.set(units),
            _ => {}
        }
        Rope(Rc::new(node))
    }

    /// The total length in bytes (O(1) — cached).
    #[must_use]
    pub fn len(&self) -> usize {
        match &*self.0 {
            Node::Small { len, .. } => usize::from(*len),
            Node::Leaf(s, _) => s.len(),
            Node::Slice { len, .. } | Node::Concat { len, .. } | Node::Build { len, .. } => *len,
        }
    }

    /// Whether the rope is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The length in **UTF-16 code units** — what JavaScript's `String.length`
    /// reports (an astral character counts 2, a lone surrogate 1).
    ///
    /// Memoized per node, because this is not derivable from the byte length and
    /// the naive form is a full scan: `str.length` was materializing the whole
    /// string and counting units on *every* read, so `for (i = 0; i < s.length;
    /// i++)` was quadratic (40 000 reads of a 40 000-char string: 4.6 s). The
    /// first call costs one pass, every later call on the same node is O(1).
    ///
    /// Sound to cache because a rope node is immutable once built — `concat`
    /// makes a new node rather than mutating either side.
    ///
    /// Walks with an **explicit stack**, like [`Rope::materialize_bytes`] and
    /// `Drop`. A `+=`-built string is a left-leaning tree whose depth is its
    /// append count, so the obvious recursive sum overflows the native stack on
    /// exactly the strings this exists to speed up — test262's `buildString`
    /// aborted several `RegExp/property-escapes` workers that way.
    #[must_use]
    pub fn utf16_len(&self) -> usize {
        let root_cell = match &*self.0 {
            Node::Small { len, u16, buf } => {
                let cached = u16.get();
                if cached != SMALL_U16_UNKNOWN {
                    return usize::from(cached);
                }
                let n = wtf8::utf16_len(&buf[..usize::from(*len)]);
                u16.set(n as u8);
                return n;
            }
            Node::Leaf(bytes, cell) => {
                let cached = cell.get();
                if cached != U16_LEN_UNKNOWN {
                    return cached;
                }
                let n = wtf8::utf16_len(bytes);
                cell.set(n);
                return n;
            }
            Node::Slice { u16_len, .. } | Node::Build { u16_len, .. } => {
                let cached = u16_len.get();
                if cached != U16_LEN_UNKNOWN {
                    return cached;
                }
                let n = wtf8::utf16_len(self.0.flat_bytes().unwrap_or_default());
                u16_len.set(n);
                return n;
            }
            Node::Concat { u16_len, .. } => u16_len,
        };
        let cached = root_cell.get();
        if cached != U16_LEN_UNKNOWN {
            return cached;
        }
        let mut total = 0usize;
        let mut stack: Vec<&Rope> = alloc::vec![self];
        while let Some(node) = stack.pop() {
            // An already-known subtree contributes its total without a descent,
            // so a repeated read over a growing rope stays cheap.
            let known = node
                .u16_len_cell()
                .map_or(U16_LEN_UNKNOWN, core::cell::Cell::get);
            if known != U16_LEN_UNKNOWN {
                total += known;
                continue;
            }
            match &*node.0 {
                Node::Small { len, buf, .. } => total += wtf8::utf16_len(&buf[..usize::from(*len)]),
                Node::Leaf(bytes, cell) => {
                    let n = wtf8::utf16_len(bytes);
                    cell.set(n);
                    total += n;
                }
                Node::Slice { .. } | Node::Build { .. } => total += node.utf16_len(),
                Node::Concat { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        root_cell.set(total);
        total
    }

    /// Whether every code point is ASCII — in which case byte index, code-point
    /// index and UTF-16 unit index all coincide, so an indexed read is O(1)
    /// instead of a scan from the start of the string.
    ///
    /// The test is exact and costs nothing after the first call: a UTF-8 byte
    /// length equals the UTF-16 unit length precisely when every character
    /// encodes as one byte (2- and 3-byte forms are 1 unit, the 4-byte form is
    /// 2), and both lengths are memoized.
    #[must_use]
    pub fn is_ascii(&self) -> bool {
        self.len() == self.utf16_len()
    }

    /// The node's UTF-16 length cache slot (see [`Rope::utf16_len`]); an
    /// inline node has none (its length is recounted).
    fn u16_len_cell(&self) -> Option<&core::cell::Cell<usize>> {
        match &*self.0 {
            Node::Small { .. } => None,
            Node::Leaf(_, u16_len)
            | Node::Slice { u16_len, .. }
            | Node::Concat { u16_len, .. }
            | Node::Build { u16_len, .. } => Some(u16_len),
        }
    }

    /// Concatenates `self` and `other` in O(1) when the combined length is
    /// representable (≤ [`MAX_STRING_LEN`]), sharing both sides. Returns `None`
    /// when the result would overflow `usize` or exceed the cap — the caller
    /// turns that into a `RangeError("Invalid string length")` rather than
    /// corrupting the cached length or OOM-ing on materialization.
    ///
    /// The concatenation of a program's `+`: a short right side may be
    /// copied into an append buffer (see [`BuildBuf`]), which a growing
    /// chain of such steps amortizes. Code assembling a new string from
    /// pieces of existing ones (a slice of a tree, `repeat`) uses
    /// [`try_concat_shared`](Self::try_concat_shared), which never copies.
    #[must_use]
    pub fn try_concat(&self, other: &Rope) -> Option<Rope> {
        self.concat_with(other, true)
    }

    /// [`try_concat`](Self::try_concat) without append buffers: long sides
    /// are always shared by a `Concat`, so building a string from `k`
    /// pieces costs O(k) however long they are (`try_concat` could copy a
    /// right side up to the left's length into the left's buffer, which a
    /// one-off assembly repeated per step, `s = s.slice(1)` over a tree,
    /// would pay in full every time).
    #[must_use]
    pub fn try_concat_shared(&self, other: &Rope) -> Option<Rope> {
        self.concat_with(other, false)
    }

    /// [`try_concat`](Self::try_concat), through an append buffer when
    /// `build`.
    #[inline]
    fn concat_with(&self, other: &Rope, build: bool) -> Option<Rope> {
        // The commonest step of a `s += x` loop, `x` a short inline string
        // (a literal, a character, a number's digits): as the general one
        // below, with the inline buffer copied as one fixed-size block and
        // its UTF-16 length read from its memo.
        if build
            && let Node::Build {
                buf,
                len: at,
                u16_len: lu,
            } = &*self.0
            && let Node::Small {
                len: n,
                u16: nu,
                buf: sb,
            } = &*other.0
            && *n > 0
            && sb[0] != 0xED
            && *at + usize::from(*n) <= MAX_STRING_LEN
            && buf.append_small(*at, sb, usize::from(*n))
        {
            let n = usize::from(*n);
            let known = lu.get();
            let u16_len = if known == U16_LEN_UNKNOWN {
                U16_LEN_UNKNOWN
            } else {
                match nu.get() {
                    SMALL_U16_UNKNOWN => known + other.utf16_len(),
                    m => known + usize::from(m),
                }
            };
            return Some(Rope(Rc::new(Node::Build {
                buf: Rc::clone(buf),
                len: *at + n,
                u16_len: core::cell::Cell::new(u16_len),
            })));
        }
        // The step of a `s += x` loop: `self` the last string of its append
        // buffer and `other` a short flat string that fits behind it and does
        // not begin with a surrogate (which could pair at the seam), so the
        // general checks below all pass and `build_append` writes in place.
        if build
            && let Node::Build {
                buf,
                len: at,
                u16_len: lu,
            } = &*self.0
            && let Some(b) = other.0.flat_bytes()
            && let [first, ..] = *b
            && first != 0xED
            && b.len() <= BUILD_APPEND_MIN
            && *at + b.len() <= MAX_STRING_LEN
            && buf.append_at(*at, &[b])
        {
            let known = lu.get();
            let u16_len = match other.utf16_len_if_cheap() {
                Some(n) if known != U16_LEN_UNKNOWN => known + n,
                _ => U16_LEN_UNKNOWN,
            };
            return Some(Rope(Rc::new(Node::Build {
                buf: Rc::clone(buf),
                len: *at + b.len(),
                u16_len: core::cell::Cell::new(u16_len),
            })));
        }
        // Concatenating with empty just shares the non-empty side.
        if self.is_empty() {
            return Some(other.clone());
        }
        if other.is_empty() {
            return Some(self.clone());
        }
        let len = self.len().checked_add(other.len())?;
        if len > MAX_STRING_LEN {
            return None;
        }
        // WTF-8 canonicalization at the boundary: if `self` ends with a lone high
        // surrogate and `other` begins with a lone low surrogate, they denote a
        // single astral scalar (exactly as UTF-16 decoding would pair them). Coalesce
        // them into the 4-byte astral form so the two byte encodings of the same
        // UTF-16 sequence stay canonical — and therefore compare byte-equal, iterate
        // as one code point, and are "well-formed". This only materializes in the
        // rare case that a boundary pair actually exists; the common concat stays
        // O(1). The peek walks to the rightmost leaf of `self` / leftmost of `other`.
        // (The appended side first: usually a short leaf, and rarely a match.)
        if let Some(lo) = wtf8::leading_low_surrogate(other.first_leaf_bytes())
            && let Some(hi) = wtf8::trailing_high_surrogate(self.last_leaf_bytes())
        {
            let mut bytes = self.materialize_bytes();
            bytes.truncate(bytes.len() - 3); // drop the lone high surrogate
            let cp = 0x1_0000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
            wtf8::encode_code_point(cp, &mut bytes);
            let other_bytes = other.materialize_bytes();
            bytes.extend_from_slice(&other_bytes[3..]); // drop the lone low surrogate
            return Some(Rope::from_wtf8(bytes));
        }
        // No surrogate pair forms at the boundary, so the result's UTF-16
        // length is the sum of the sides': carried over when both are known
        // already (or, for a short flat side, counted now), so a `s += c`
        // loop that reads `s.length` never rescans what it built.
        let u16_len = self
            .utf16_len_if_cheap()
            .and_then(|x| Some(x + other.utf16_len_if_cheap()?));
        // Two short leaves (`'item' + i`, a template's pieces) join into one
        // leaf: copying a few bytes costs less than a `Concat` node that every
        // later read has to walk and that eventually has to be flattened anyway.
        if len <= SMALL_CAP
            && let (
                Node::Small {
                    len: la, buf: ba, ..
                },
                Node::Small { buf: bb, .. },
            ) = (&*self.0, &*other.0)
        {
            // Two inline buffers, each copied whole as one fixed-size block
            // (their bytes past their lengths are zero, so the result's are).
            let la = usize::from(*la);
            let memo = core::cell::Cell::new(u16_len.map_or(SMALL_U16_UNKNOWN, |n| n as u8));
            if len - la <= 16 && la + 16 <= SMALL_CAP {
                // A right side of one word: written in the new node itself
                // (see `small_with_int`).
                let mut rc = Rc::new(Node::Small {
                    len: len as u8,
                    u16: memo,
                    buf: *ba,
                });
                if let Some(Node::Small { buf: out, .. }) = Rc::get_mut(&mut rc) {
                    out[la..la + 16].copy_from_slice(&bb[..16]);
                }
                return Some(Rope(rc));
            }
            let mut tmp = [0u8; 2 * SMALL_CAP];
            tmp[..SMALL_CAP].copy_from_slice(ba);
            tmp[la..la + SMALL_CAP].copy_from_slice(bb);
            let mut buf = [0u8; SMALL_CAP];
            buf.copy_from_slice(&tmp[..SMALL_CAP]);
            return Some(Rope(Rc::new(Node::Small {
                len: len as u8,
                u16: memo,
                buf,
            })));
        }
        if len <= SMALL_JOIN
            && let (Some(a), Some(b)) = (self.0.flat_bytes(), other.0.flat_bytes())
        {
            if len <= SMALL_CAP {
                let mut buf = [0u8; SMALL_CAP];
                buf[..a.len()].copy_from_slice(a);
                buf[a.len()..len].copy_from_slice(b);
                return Some(Rope(Rc::new(Node::Small {
                    len: len as u8,
                    u16: core::cell::Cell::new(u16_len.map_or(SMALL_U16_UNKNOWN, |n| n as u8)),
                    buf,
                })));
            }
            let mut bytes = Vec::with_capacity(len);
            bytes.extend_from_slice(a);
            bytes.extend_from_slice(b);
            return Some(Rope(Rc::new(Node::Leaf(
                bytes.into_boxed_slice(),
                core::cell::Cell::new(u16_len.unwrap_or(U16_LEN_UNKNOWN)),
            ))));
        }
        if build && let Some(r) = self.build_append(other, len, u16_len) {
            return Some(r);
        }
        Some(Rope(Rc::new(Node::Concat {
            left: self.clone(),
            right: other.clone(),
            len,
            u16_len: core::cell::Cell::new(u16_len.unwrap_or(U16_LEN_UNKNOWN)),
        })))
    }

    /// `self + other` (`len` bytes, past [`SMALL_JOIN`], no surrogate pair
    /// at the seam, UTF-16 length `u16_len` when known) through an append
    /// buffer (see [`BuildBuf`]): in place behind `self` when it ends the
    /// written part of its buffer and `other` fits, in a fresh buffer of
    /// twice the length when `self` is that last string but the buffer is
    /// full or `self` is a short flat string, and as a `Concat` whose right
    /// side is so extended when `self` is a `Concat` ending in such a
    /// string. `None` (a plain `Concat`) otherwise, or when `other` is not
    /// flat or is long (see [`BUILD_APPEND_MIN`]).
    fn build_append(&self, other: &Rope, len: usize, u16_len: Option<usize>) -> Option<Rope> {
        let b = other.0.flat_bytes()?;
        if b.len() > BUILD_APPEND_MIN.max(self.len()) {
            return None;
        }
        let u16_len = core::cell::Cell::new(u16_len.unwrap_or(U16_LEN_UNKNOWN));
        let fresh = |a: &[u8], u16_len| {
            let cap = len
                .saturating_mul(2)
                .max(BUILD_MIN_CAP)
                .min(MAX_STRING_LEN.max(len));
            Rope(Rc::new(Node::Build {
                buf: BuildBuf::with(cap, a, b),
                len,
                u16_len,
            }))
        };
        match &*self.0 {
            Node::Build { buf, len: at, .. } => {
                if buf.append_at(*at, &[b]) {
                    return Some(Rope(Rc::new(Node::Build {
                        buf: Rc::clone(buf),
                        len,
                        u16_len,
                    })));
                }
                // The last string of a full buffer: a bigger one. (An older
                // prefix is shared by a `Concat`.)
                (*at == buf.used.get()).then(|| fresh(buf.bytes(*at), u16_len))
            }
            Node::Small { .. } | Node::Leaf(..) | Node::Slice { .. } => {
                let a = self.0.flat_bytes()?;
                (a.len() <= BUILD_SEED_MAX && b.len() <= BUILD_SEED_RIGHT_MAX)
                    .then(|| fresh(a, u16_len))
            }
            Node::Concat {
                left,
                right,
                u16_len: _,
                len: _,
            } => {
                // (`right` + `other` cannot pair surrogates: `self` + `other`
                // did not.)
                let rlen = right.len() + b.len();
                let ru16 = right
                    .utf16_len_if_cheap()
                    .and_then(|x| Some(x + other.utf16_len_if_cheap()?));
                let joined = match &*right.0 {
                    Node::Build { .. } => right.build_append(other, rlen, ru16)?,
                    Node::Small { .. } | Node::Leaf(..) if right.len() <= BUILD_SEED_MAX => {
                        right.build_append(other, rlen, ru16)?
                    }
                    _ => return None,
                };
                Some(Rope(Rc::new(Node::Concat {
                    left: left.clone(),
                    right: joined,
                    len,
                    u16_len,
                })))
            }
        }
    }

    /// The decimal text of an integer `i` of magnitude below 2^53 (`-0`
    /// passed as `0`), as `Number::toString` prints it, held inline with its
    /// UTF-16 length known.
    #[must_use]
    #[inline]
    pub fn from_int(i: i64) -> Rope {
        let (acc, n) = digits_le(i.unsigned_abs());
        let len = n + usize::from(i < 0);
        let mut buf = [0u8; SMALL_CAP];
        put_int(&mut buf, 0, i, acc);
        Rope(Rc::new(Node::Small {
            len: len as u8,
            u16: core::cell::Cell::new(len as u8),
            buf,
        }))
    }

    /// `self + i` (or `i + self` when `number_first`) for an inline string
    /// `self` and an integer `i` of magnitude below 2^53 (which prints as
    /// plain decimal digits; `-0` is passed as `0`), when the result fits
    /// inline too: that string, built in one step with no digit string of
    /// its own. `None` for a longer result or a string not held inline.
    #[must_use]
    #[inline]
    pub fn small_with_int(&self, i: i64, number_first: bool) -> Option<Rope> {
        let Node::Small { len, u16, buf } = &*self.0 else {
            return None;
        };
        let la = usize::from(*len);
        let (acc, n) = digits_le(i.unsigned_abs());
        let digits = n + usize::from(i < 0);
        let total = la + digits;
        if total > SMALL_CAP {
            return None;
        }
        // Digits are one UTF-16 unit each. (The text's own count is taken
        // once, and memoized on it: usually a literal, met again.)
        let units = match u16.get() {
            SMALL_U16_UNKNOWN => self.utf16_len() as u8,
            m => m,
        };
        let memo = units + digits as u8;
        // From fixed-size pieces: the text's whole inline buffer (its bytes
        // past its length are zero, so the result's are) and the digits'
        // word (zero past the digits), the later write over the earlier's
        // zeros. Text first, with room for the word: written in the new
        // node itself (bytes assembled in a scratch buffer by overlapping
        // writes and then read back as a whole stall the processor).
        if !number_first && la + 17 <= SMALL_CAP {
            let mut rc = Rc::new(Node::Small {
                len: total as u8,
                u16: core::cell::Cell::new(memo),
                buf: *buf,
            });
            if let Some(Node::Small { buf: out, .. }) = Rc::get_mut(&mut rc) {
                put_int(out, la, i, acc);
            }
            return Some(Rope(rc));
        }
        let mut tmp = [0u8; SMALL_CAP + 17];
        if number_first {
            put_int(&mut tmp, 0, i, acc);
            tmp[digits..digits + SMALL_CAP].copy_from_slice(buf);
        } else {
            tmp[..SMALL_CAP].copy_from_slice(buf);
            put_int(&mut tmp, la, i, acc);
        }
        let mut out = [0u8; SMALL_CAP];
        out.copy_from_slice(&tmp[..SMALL_CAP]);
        Some(Rope(Rc::new(Node::Small {
            len: total as u8,
            u16: core::cell::Cell::new(memo),
            buf: out,
        })))
    }

    /// The UTF-16 length when it costs (nearly) nothing: memoized already, or
    /// an inline run counted now (at most [`SMALL_CAP`] bytes). `None`
    /// otherwise.
    #[inline]
    fn utf16_len_if_cheap(&self) -> Option<usize> {
        if let Node::Small { u16, .. } = &*self.0 {
            return Some(match u16.get() {
                SMALL_U16_UNKNOWN => self.utf16_len(),
                m => usize::from(m),
            });
        }
        let n = self.u16_len_cell()?.get();
        (n != U16_LEN_UNKNOWN).then_some(n)
    }

    /// Concatenates `self` and `other` in O(1), sharing both sides. A result
    /// past [`MAX_STRING_LEN`] (or `usize` overflow) is clamped to sharing only
    /// the left side — a bounded, panic-free degradation. Callers that need to
    /// surface the spec's `RangeError` use [`Rope::try_concat`] instead.
    #[must_use]
    pub fn concat(&self, other: &Rope) -> Rope {
        self.try_concat(other).unwrap_or_else(|| self.clone())
    }

    /// Appends `s` to the rope (O(1)), returning the new rope.
    #[must_use]
    pub fn push_str(&self, s: &str) -> Rope {
        self.concat(&Rope::leaf(s))
    }

    /// Whether the two ropes hold the same WTF-8 bytes (strict equality of
    /// two strings): two short inline strings by their first word, else
    /// their lengths then their bytes.
    #[must_use]
    #[inline]
    pub fn same_bytes(&self, other: &Rope) -> bool {
        if let (Node::Small { len: a, buf: x, .. }, Node::Small { len: b, buf: y, .. }) =
            (&*self.0, &*other.0)
        {
            if a != b {
                return false;
            }
            // The bytes past `len` are zero in both: one word decides a
            // short string (most keys and names), with no `memcmp` call.
            let word = |s: &[u8; SMALL_CAP]| {
                u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
            };
            if *a <= 8 {
                return word(x) == word(y);
            }
            let n = usize::from(*a);
            return x[..n] == y[..n];
        }
        if self.len() != other.len() {
            return false;
        }
        match (self.as_leaf_bytes(), other.as_leaf_bytes()) {
            (Some(p), Some(q)) => p == q,
            _ => self.materialize_bytes() == other.materialize_bytes(),
        }
    }

    /// Borrows the rope's WTF-8 bytes directly when it is a single, unconcatenated
    /// `Leaf` (the common case — a string that has never been `+`-joined), returning
    /// `None` for a `Concat` tree. This is a zero-copy fast path: callers that only
    /// *read* the bytes (equality, ordering, emptiness) avoid the owned `Vec` that
    /// [`Rope::materialize_bytes`] allocates. Fall back to `materialize_bytes` when
    /// this returns `None`.
    #[must_use]
    #[inline]
    pub fn as_leaf_bytes(&self) -> Option<&[u8]> {
        self.0.flat_bytes()
    }

    /// Borrows the rope as a `&str` when it is a single flat leaf holding
    /// well-formed UTF-8 (no lone surrogates) — the zero-copy counterpart of
    /// [`Rope::materialize`] for the common case.
    #[must_use]
    pub fn as_leaf_str(&self) -> Option<&str> {
        self.0.flat_bytes().and_then(wtf8::as_str)
    }

    /// The UTF-16 code units `[start, end)` of a flat rope (see
    /// [`Rope::as_leaf_bytes`]) as a new string — `String.prototype.slice`'s
    /// and `substring`'s result. Indices are clamped to the length, and
    /// `start >= end` is empty. A boundary inside an astral character splits
    /// it into lone surrogates, as JavaScript does.
    ///
    /// A long result **shares** the receiver's buffer (a `Slice` node) rather
    /// than copying it, so `s = s.slice(1)` or `s.substring(i)` in a loop is
    /// linear, not quadratic in the string's length; see [`SLICE_SHARE_MIN`]
    /// and [`SLICE_SHARE_RATIO`] for when it copies instead. `None` when the
    /// rope is a `Concat` (flatten it first).
    #[must_use]
    pub fn slice_units(&self, start: usize, end: usize) -> Option<Rope> {
        self.slice_units_from((0, 0), start, end).map(|(r, _)| r)
    }

    /// [`Rope::slice_units`] of a `Concat` rope **without flattening it**: the
    /// UTF-16 range `[start, end)` (clamped, `start < end` expected) is found
    /// by walking down the tree, so a slice inside one leaf costs a few steps
    /// plus that leaf's scan rather than a copy (and, for a non-ASCII string,
    /// a unit scan) of the whole rope. A range spanning several nodes shares
    /// the subtrees it covers and slices the leaves at its ends.
    ///
    /// `None` for a flat rope (use [`Rope::slice_units`]) and whenever the
    /// walk would be long, past [`TREE_SLICE_STEPS`] nodes (a deep `+=`-built
    /// tree, where flattening once makes every later read O(1)). The caller
    /// then flattens as before, so the walk adds a bounded cost at worst.
    #[must_use]
    pub fn slice_units_tree(&self, start: usize, end: usize) -> Option<Rope> {
        if !matches!(&*self.0, Node::Concat { .. }) {
            return None;
        }
        let total = self.utf16_len();
        let end = end.min(total);
        let start = start.min(end);
        if start == end {
            return Some(Rope::new());
        }
        // `(node, its UTF-16 length)`, descending while one child holds the range.
        let (mut node, mut units) = (self, total);
        let (mut a, mut b) = (start, end);
        let mut steps = 0usize;
        loop {
            let Node::Concat { left, right, .. } = &*node.0 else {
                return node.slice_units(a, b);
            };
            steps += 1;
            if steps > TREE_SLICE_STEPS {
                return None;
            }
            let (lu, ru) = Self::child_units(left, right, units);
            if b <= lu {
                (node, units) = (left, lu);
            } else if a >= lu {
                (node, units) = (right, ru);
                a -= lu;
                b -= lu;
            } else {
                break;
            }
        }
        // The range spans `node`'s children: join the overlapped pieces left
        // to right, a fully covered subtree shared whole and a partly covered
        // leaf sliced, so the result costs O(steps) however long it is
        // (`try_concat` pairs a surrogate split at a seam).
        let mut out = Rope::new();
        let mut stack: Vec<(&Rope, usize, usize, usize)> = alloc::vec![(node, units, a, b)];
        while let Some((n, nu, a, b)) = stack.pop() {
            let piece = match &*n.0 {
                _ if a == 0 && b == nu => n.clone(),
                Node::Concat { left, right, .. } => {
                    steps += 1;
                    if steps > TREE_SLICE_STEPS {
                        return None;
                    }
                    let (lu, ru) = Self::child_units(left, right, nu);
                    if b > lu {
                        stack.push((right, ru, a.saturating_sub(lu), b - lu));
                    }
                    if a < lu {
                        stack.push((left, lu, a, b.min(lu)));
                    }
                    continue;
                }
                _ => n.slice_units(a, b)?,
            };
            out = out.try_concat_shared(&piece)?;
        }
        Some(out)
    }

    /// The UTF-16 code unit at `i` of a `Concat` rope, read by walking down
    /// its tree rather than flattening it: `Some(None)` past the end. `None`
    /// (flatten and read the flat string instead) for a flat rope, when the
    /// walk passes [`UNIT_WALK_STEPS`] nodes, or when the leaf reached is
    /// long and not ASCII (a unit index into it is a scan, which the flat
    /// string's memoized cursor makes O(1) for a loop).
    #[must_use]
    pub fn unit_at_tree(&self, i: usize) -> Option<Option<u16>> {
        if !matches!(&*self.0, Node::Concat { .. }) {
            return None;
        }
        let total = self.utf16_len();
        if i >= total {
            return Some(None);
        }
        let (mut node, mut units, mut i) = (self, total, i);
        for _ in 0..=UNIT_WALK_STEPS {
            let Node::Concat { left, right, .. } = &*node.0 else {
                let bytes = node.0.flat_bytes()?;
                if units == bytes.len() {
                    return Some(bytes.get(i).map(|b| u16::from(*b)));
                }
                if bytes.len() > UNIT_WALK_SCAN {
                    return None;
                }
                return Some(wtf8::utf16_index(bytes, i));
            };
            let (lu, ru) = Self::child_units(left, right, units);
            if i < lu {
                (node, units) = (left, lu);
            } else {
                (node, units) = (right, ru);
                i -= lu;
            }
        }
        None
    }

    /// Visits the leaves' WTF-8 bytes in order from one end (the right end
    /// when `from_right`) while `visit` returns `true`, walking at most
    /// [`TREE_SLICE_STEPS`] `Concat` nodes: `false` when the walk was cut
    /// short by that bound (the caller then flattens instead).
    pub fn walk_edge_leaves(&self, from_right: bool, mut visit: impl FnMut(&[u8]) -> bool) -> bool {
        let mut stack: Vec<&Rope> = alloc::vec![self];
        let mut steps = 0usize;
        while let Some(node) = stack.pop() {
            match &*node.0 {
                Node::Concat { left, right, .. } => {
                    steps += 1;
                    if steps > TREE_SLICE_STEPS {
                        return false;
                    }
                    if from_right {
                        stack.push(left);
                        stack.push(right);
                    } else {
                        stack.push(right);
                        stack.push(left);
                    }
                }
                flat => {
                    if !visit(flat.flat_bytes().unwrap_or_default()) {
                        return true;
                    }
                }
            }
        }
        true
    }

    /// Whether this is a `Concat` (not a flat string).
    #[must_use]
    pub fn is_concat(&self) -> bool {
        matches!(&*self.0, Node::Concat { .. })
    }

    /// The bytes `[b0, b1)` of this flat rope (both on character
    /// boundaries) as a string, sharing its buffer as
    /// [`Rope::slice_units`] does; `None` for a `Concat`.
    #[must_use]
    pub fn share_byte_range(&self, b0: usize, b1: usize) -> Option<Rope> {
        let bytes = self.0.flat_bytes()?;
        let sub = bytes.get(b0..b1)?;
        let units = if self.is_ascii() {
            sub.len()
        } else {
            wtf8::utf16_len(sub)
        };
        Some(self.share_bytes(b0, b1, units))
    }

    /// The address of this rope's node: two ropes with the same address are
    /// the same text, while some [`RopeId`] of it or a clone stands.
    #[must_use]
    #[inline]
    pub fn addr(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// A non-owning identity for this rope (see [`RopeId`]).
    #[must_use]
    pub fn id(&self) -> RopeId {
        RopeId(Rc::downgrade(&self.0))
    }

    /// The UTF-16 lengths of a `Concat`'s children given the node's own
    /// (`units`): an all-ASCII node's children are their byte lengths, and
    /// otherwise only the shorter child is counted (memoized), the other being
    /// the difference — so a walk down a lopsided tree never rescans its big side.
    fn child_units(left: &Rope, right: &Rope, units: usize) -> (usize, usize) {
        if units == left.len() + right.len() {
            return (left.len(), right.len());
        }
        if left.len() <= right.len() {
            let lu = left.utf16_len();
            (lu, units - lu)
        } else {
            let ru = right.utf16_len();
            (units - ru, ru)
        }
    }

    /// [`Rope::slice_units`] scanning a non-ASCII string from `hint`, a
    /// `(byte, unit)` character boundary of it (see [`wtf8::utf16_range`]),
    /// rather than from its start; also returns the first whole character's
    /// `(byte, unit)` position, the next call's hint.
    #[must_use]
    pub fn slice_units_from(
        &self,
        hint: (usize, usize),
        start: usize,
        end: usize,
    ) -> Option<(Rope, (usize, usize))> {
        let bytes = self.0.flat_bytes()?;
        let units = self.utf16_len();
        let end = end.min(units);
        let start = start.min(end);
        if start == end {
            return Some((Rope::new(), hint));
        }
        if self.is_ascii() {
            return Some((self.share_bytes(start, end, end - start), (start, start)));
        }
        let r = wtf8::utf16_range(bytes, hint, start, end, units);
        let (lead, trail) = (
            usize::from(r.lead.is_some()),
            usize::from(r.trail.is_some()),
        );
        let mid = self.share_bytes(r.start, r.end, end - start - lead - trail);
        let at = (r.start, start + lead);
        if lead + trail == 0 {
            return Some((mid, at));
        }
        // A boundary splits an astral character: its half is a lone
        // surrogate beside the whole characters (the concatenation keeps the
        // canonical form; nothing here can pair).
        let lone = |u: Option<u16>| u.map(|u| Rope::from_wtf8(wtf8::from_utf16(&[u])));
        let join = |a: &Rope, b: &Rope| a.try_concat_shared(b).unwrap_or_else(|| a.clone());
        let mut out = join(&lone(r.lead).unwrap_or_default(), &mid);
        if let Some(t) = lone(r.trail) {
            out = join(&out, &t);
        }
        Some((out, at))
    }

    /// The bytes `[b0, b1)` of this flat rope, holding `units` UTF-16 units,
    /// as a rope: shared or copied (see [`Rope::slice_units`]).
    fn share_bytes(&self, b0: usize, b1: usize, units: usize) -> Rope {
        let bytes = self.0.flat_bytes().unwrap_or_default();
        let len = b1 - b0;
        if len == bytes.len() {
            return self.clone();
        }
        let (base, base_start) = match &*self.0 {
            Node::Slice { base, start, .. } => (base, *start),
            Node::Leaf(..) | Node::Build { .. } => (self, 0),
            _ => return Rope::from_bytes(&bytes[b0..b1]),
        };
        if len < SLICE_SHARE_MIN || len.saturating_mul(SLICE_SHARE_RATIO) < base.len() {
            return Rope::from_bytes(&bytes[b0..b1]);
        }
        Rope(Rc::new(Node::Slice {
            base: base.clone(),
            start: base_start + b0,
            len,
            u16_len: core::cell::Cell::new(units),
        }))
    }

    /// The rightmost leaf's bytes (O(depth)) — used to peek the trailing code unit
    /// for boundary surrogate coalescing without materializing the whole rope.
    #[must_use]
    fn last_leaf_bytes(&self) -> &[u8] {
        let mut node = &self.0;
        loop {
            match &**node {
                Node::Concat { right, .. } => node = &right.0,
                flat => return flat.flat_bytes().unwrap_or_default(),
            }
        }
    }

    /// The leftmost leaf's bytes (O(depth)).
    #[must_use]
    fn first_leaf_bytes(&self) -> &[u8] {
        let mut node = &self.0;
        loop {
            match &**node {
                Node::Concat { left, .. } => node = &left.0,
                flat => return flat.flat_bytes().unwrap_or_default(),
            }
        }
    }

    /// Whether `self` and `other` share one backing node — an `Rc` pointer
    /// comparison, so it is O(1) and says nothing about equal *content* held in
    /// two allocations. Callers holding a clone of one side keep that allocation
    /// alive, which is what makes the identity meaningful: a live node's address
    /// cannot be reused by a different node.
    #[must_use]
    pub fn ptr_eq(&self, other: &Rope) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// Materializes the rope into a flat WTF-8 byte buffer in O(n), iteratively
    /// (so a deeply nested rope cannot overflow the stack). This is the
    /// lossless form — lone surrogates are preserved. Callers wanting a real
    /// `&str`/`String` use `Rope::materialize` (lossy).
    #[must_use]
    pub fn materialize_bytes(&self) -> Vec<u8> {
        if let Some(b) = self.0.flat_bytes() {
            return b.to_vec();
        }
        let mut out = alloc::vec![0u8; self.len()];
        self.fill_bytes(&mut out);
        out
    }

    /// Writes the rope's bytes into `out`, which is exactly [`len`](Self::len)
    /// bytes long. Every node's length is known, so each piece goes straight
    /// to its offset and the tree can be visited in any order: of a
    /// concatenation's two sides the shorter is done first and the longer
    /// waits on the stack, which so never holds more than about log2(len)
    /// entries however lopsided the tree (a `s += c` loop builds a left
    /// spine as long as the string).
    fn fill_bytes(&self, out: &mut [u8]) {
        let mut stack: Vec<(&Rope, usize)> = alloc::vec![(self, 0)];
        while let Some((mut node, mut at)) = stack.pop() {
            loop {
                match &*node.0 {
                    Node::Concat { left, right, .. } => {
                        let mid = at + left.len();
                        let (small, big) = if left.len() <= right.len() {
                            ((left, at), (right, mid))
                        } else {
                            ((right, mid), (left, at))
                        };
                        stack.push(big);
                        (node, at) = small;
                    }
                    flat => {
                        let b = flat.flat_bytes().unwrap_or_default();
                        if let [one] = *b {
                            out[at] = one;
                        } else {
                            out[at..at + b.len()].copy_from_slice(b);
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Appends the rope's WTF-8 bytes (lossless) to `out`, iteratively — the
    /// in-place form of [`Rope::materialize_bytes`] for a caller assembling a
    /// larger buffer, so no intermediate `Vec` is made.
    pub fn append_bytes_to(&self, out: &mut Vec<u8>) {
        if let Some(b) = self.0.flat_bytes() {
            out.extend_from_slice(b);
            return;
        }
        let start = out.len();
        out.resize(start + self.len(), 0);
        self.fill_bytes(&mut out[start..]);
    }

    /// Materializes the rope into a flat `String` in O(n). **Lossy**: any stored
    /// lone surrogate is replaced with U+FFFD (via [`crate::wtf8::to_string_lossy`]),
    /// so the signature stays `String` for existing `&str`/`String` callers. A
    /// non-surrogate string is unchanged. Use `Rope::materialize_bytes` for the
    /// lossless WTF-8 bytes.
    #[must_use]
    pub fn materialize(&self) -> String {
        if let Some(b) = self.0.flat_bytes() {
            return wtf8::to_string_lossy(b);
        }
        // Flattened bytes without surrogates are already UTF-8: reuse the
        // buffer rather than copying it into a second one.
        match String::from_utf8(self.materialize_bytes()) {
            Ok(s) => s,
            Err(e) => wtf8::to_string_lossy(e.as_bytes()),
        }
    }
}

impl Drop for Node {
    /// Dismantles a `Concat` chain iteratively. A deeply nested rope would
    /// otherwise free recursively (each `Node` drops its child `Rc<Node>`, which
    /// drops the next…), overflowing the stack on platforms with a small default
    /// (e.g. Windows' 1 MB test threads) for the pathological left-/right-deep
    /// shape that O(1) concatenation builds.
    ///
    /// The test is inline (most dying nodes are not a `Concat`, and most
    /// that are own no `Concat` child alone); the dismantling is out of line.
    #[inline]
    fn drop(&mut self) {
        let Node::Concat { left, right, .. } = self else {
            return; // a leaf owns nothing recursive
        };
        // Only a child this node owns alone *and* that is itself a `Concat`
        // continues the chain. Without one, the fields' ordinary drop is a
        // refcount decrement or a leaf free, never a recursion, so the
        // iterative dismantling (and its allocations) is skipped.
        let continues =
            |r: &Rope| Rc::strong_count(&r.0) == 1 && matches!(*r.0, Node::Concat { .. });
        if continues(left) || continues(right) {
            dismantle_concat(left, right);
        }
    }
}

/// The iterative part of a `Concat` node's [`Drop`]: takes its children
/// `left` and `right` out (leaving shared empty leaves) and frees the chain
/// below them without recursing.
#[cold]
#[inline(never)]
fn dismantle_concat(left: &mut Rope, right: &mut Rope) {
    // A cheap leaf to swap into a child slot so we can take ownership of the
    // real `Rc<Node>` without recursing through its `Drop`. One is shared by
    // every slot (each swap is a refcount increment, not an allocation).
    let empty = Rc::new(Node::Small {
        len: 0,
        u16: core::cell::Cell::new(0),
        buf: [0; SMALL_CAP],
    });
    let sentinel = || Rc::clone(&empty);
    let mut stack = alloc::vec![
        core::mem::replace(&mut left.0, sentinel()),
        core::mem::replace(&mut right.0, sentinel()),
    ];
    while let Some(rc) = stack.pop() {
        // Only the last owner can free; otherwise just drop the `Rc` (a
        // refcount decrement, no recursion).
        if let Ok(mut node) = Rc::try_unwrap(rc)
            && let Node::Concat { left, right, .. } = &mut node
        {
            stack.push(core::mem::replace(&mut left.0, sentinel()));
            stack.push(core::mem::replace(&mut right.0, sentinel()));
            // `node` now holds sentinel leaves, so its own drop is shallow.
        }
    }
}

impl Default for Rope {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for Rope {
    fn from(s: &str) -> Self {
        Rope::leaf(s)
    }
}

impl core::fmt::Display for Rope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Stream segments. Leaves are usually plain UTF-8 (the `as_str` fast
        // path), written without allocation; a leaf bearing lone surrogates is
        // written lossily (surrogates → U+FFFD), matching `materialize`.
        let mut stack: Vec<&Rope> = alloc::vec![self];
        while let Some(node) = stack.pop() {
            match &*node.0 {
                Node::Concat { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
                flat => {
                    let s = flat.flat_bytes().unwrap_or_default();
                    match wtf8::as_str(s) {
                        Some(valid) => f.write_str(valid)?,
                        None => f.write_str(&wtf8::to_string_lossy(s))?,
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// `a` and `b` joined by a `Concat` node whatever their lengths (what
    /// `concat` makes of long or non-flat sides), for tests of tree walks.
    fn cat(a: &Rope, b: &Rope) -> Rope {
        Rope(Rc::new(Node::Concat {
            left: a.clone(),
            right: b.clone(),
            len: a.len() + b.len(),
            u16_len: core::cell::Cell::new(U16_LEN_UNKNOWN),
        }))
    }

    #[test]
    fn slices_share_long_results_and_stay_exact() {
        let base = Rope::leaf(&"0123456789".repeat(20));
        let tail = base.slice_units(5, 200).unwrap();
        assert!(matches!(*tail.0, Node::Slice { .. }));
        assert_eq!(tail.materialize(), base.materialize()[5..]);
        assert_eq!(tail.utf16_len(), 195);
        // A slice of a slice points at the original leaf.
        let inner = tail.slice_units(10, 150).unwrap();
        match &*inner.0 {
            Node::Slice { base: b, start, .. } => {
                assert!(b.ptr_eq(&base));
                assert_eq!(*start, 15);
            }
            _ => panic!("expected a shared slice"),
        }
        assert_eq!(inner.materialize(), base.materialize()[15..155]);
        // Short results are copied inline; so is a small piece of a big base.
        assert!(matches!(
            *base.slice_units(3, 9).unwrap().0,
            Node::Small { .. }
        ));
        let big = Rope::leaf(&"x".repeat(10_000));
        assert!(matches!(
            *big.slice_units(0, 100).unwrap().0,
            Node::Leaf(..)
        ));
        // A whole-range slice is the receiver itself; a Concat has no slice.
        assert!(base.slice_units(0, 999).unwrap().ptr_eq(&base));
        assert!(cat(&base, &base).slice_units(0, 1).is_none());
        // Non-ASCII and split astral pairs.
        let s = Rope::from_wtf8(crate::wtf8::from_utf16(
            &"a\u{e9}\u{1F600}b"
                .repeat(20)
                .encode_utf16()
                .collect::<Vec<_>>(),
        ));
        let units: Vec<u16> = "a\u{e9}\u{1F600}b".repeat(20).encode_utf16().collect();
        for (a, b) in [(0, 100), (2, 100), (3, 50), (1, 3), (4, 99), (7, 7)] {
            let got = s.slice_units(a, b).unwrap();
            let want = crate::wtf8::from_utf16(&units[a..b.min(units.len())]);
            assert_eq!(got.materialize_bytes(), want, "{a}..{b}");
            assert_eq!(got.utf16_len(), b.min(units.len()) - a);
        }
    }

    /// A `Concat` slices by walking its tree, exactly (non-ASCII, astral
    /// characters split at a seam and at the range ends), sharing a leaf it
    /// stays inside and the whole subtrees a spanning range covers, and gives
    /// up (leaving the caller to flatten) on a walk past `TREE_SLICE_STEPS`.
    #[test]
    fn tree_slices_walk_without_flattening() {
        let unit_rope = |s: &str| {
            Rope::from_wtf8(crate::wtf8::from_utf16(
                &s.encode_utf16().collect::<Vec<_>>(),
            ))
        };
        let a = unit_rope(&"a\u{e9}\u{1F600}b".repeat(50));
        let b = Rope::leaf(&"0123456789".repeat(30));
        // `\u{1F601}` split across a seam: the halves are lone surrogates
        // in their own leaves, joined into one astral by `try_concat` (which
        // flattens that subtree).
        let hi = Rope::from_wtf8(crate::wtf8::from_utf16(&[0xD83D]));
        let lo = Rope::from_wtf8(crate::wtf8::from_utf16(&[0xDE01]));
        let r = cat(&cat(&a, &b), &a.concat(&hi).concat(&lo.concat(&b)));
        let flat = r.materialize_bytes();
        let units: Vec<u16> = crate::wtf8::utf16_units(&flat).collect();
        assert_eq!(r.utf16_len(), units.len());
        let n = units.len();
        for (s, e) in [
            (0, n),
            (0, 1),
            (5, 40),
            (2, 3),
            (249, 260),
            (250, 550),
            (299, 900),
            (n - 310, n - 300),
            (n - 302, n - 299),
            (n - 301, n),
            (1, n - 1),
            (n, n),
        ] {
            let got = r.slice_units_tree(s, e).expect("shallow tree");
            let want = crate::wtf8::from_utf16(&units[s..e]);
            assert_eq!(got.materialize_bytes(), want, "{s}..{e}");
            assert_eq!(got.utf16_len(), e - s, "{s}..{e}");
        }
        // Inside one leaf: that leaf's slice (shared when long).
        let inner = r.slice_units_tree(560, 1000).unwrap();
        assert!(matches!(*inner.0, Node::Slice { .. }));
        // The whole range is the rope itself.
        assert!(r.slice_units_tree(0, n).unwrap().ptr_eq(&r));
        // Flat ropes have no tree; a deep tree is left to a flatten.
        assert!(a.slice_units_tree(0, 3).is_none());
        let mut deep = Rope::leaf(&"x".repeat(100));
        for _ in 0..200 {
            deep = cat(&deep, &Rope::leaf(&"y".repeat(100)));
        }
        assert!(deep.slice_units_tree(0, 3).is_none());
        assert_eq!(
            deep.slice_units_tree(deep.len() - 150, deep.len() - 40)
                .unwrap()
                .materialize(),
            "y".repeat(110)
        );
    }

    #[test]
    fn append_buffers_extend_in_place_and_keep_every_prefix() {
        // A `+=` loop: every step flat, the buffer shared, older strings fixed.
        let mut s = Rope::new();
        let mut kept = Vec::new();
        let mut want = String::new();
        for i in 0..2000 {
            let piece = alloc::format!("{i},");
            s = s.concat(&Rope::leaf(&piece));
            want.push_str(&piece);
            assert_eq!(s.len(), want.len());
            if i % 97 == 0 {
                kept.push((s.clone(), want.clone()));
            }
        }
        assert!(s.as_leaf_bytes().is_some());
        assert!(matches!(*s.0, Node::Build { .. }));
        assert_eq!(s.materialize(), want);
        assert_eq!(s.utf16_len(), want.len());
        // Appending to an older prefix leaves the newer strings alone.
        for (k, w) in &kept {
            assert_eq!(k.materialize(), *w);
            let forked = k.concat(&Rope::leaf("FORK"));
            assert_eq!(forked.materialize(), alloc::format!("{w}FORK"));
            assert_eq!(k.materialize(), *w);
        }
        assert_eq!(s.materialize(), want);
        // The newest string's buffer, appended to twice: the second shares.
        let a = s.concat(&Rope::leaf("A"));
        let b = s.concat(&Rope::leaf("B"));
        assert_eq!(a.materialize(), alloc::format!("{want}A"));
        assert_eq!(b.materialize(), alloc::format!("{want}B"));
        // Self-append (of the newest string, `a`) and slices of a buffer's
        // string.
        let d = a.concat(&a);
        let dd = alloc::format!("{want}A{want}A");
        assert_eq!(d.materialize(), dd);
        assert!(matches!(*d.0, Node::Build { .. }));
        let sl = d.slice_units(3, d.len() - 3).unwrap();
        assert_eq!(sl.materialize(), dd[3..d.len() - 3]);
        assert!(matches!(*sl.0, Node::Slice { .. }));
        // A long prefix shared by a `Concat` whose right side is extended.
        let long = Rope::leaf(&"L".repeat(5000));
        let mut t = long.concat(&Rope::leaf("x"));
        for _ in 0..500 {
            t = t.concat(&Rope::leaf("yz"));
        }
        assert!(t.is_concat());
        let Node::Concat { left, right, .. } = &*t.0 else {
            unreachable!()
        };
        assert!(left.ptr_eq(&long) && matches!(*right.0, Node::Build { .. }));
        assert_eq!(
            t.materialize(),
            alloc::format!("{}x{}", "L".repeat(5000), "yz".repeat(500))
        );
        // Surrogates pair at a buffer's seam.
        let hi = s.concat(&Rope::from_wtf8(crate::wtf8::from_utf16(&[0xD83D])));
        let paired = hi.concat(&Rope::from_wtf8(crate::wtf8::from_utf16(&[0xDE00])));
        assert_eq!(paired.utf16_len(), want.len() + 2);
        assert!(paired.materialize().ends_with('\u{1F600}'));
    }

    #[test]
    fn leaf_basics() {
        let r = Rope::leaf("hello");
        assert_eq!(r.len(), 5);
        assert!(!r.is_empty());
        assert_eq!(r.materialize(), "hello");
        assert!(Rope::new().is_empty());
        assert_eq!(Rope::new().materialize(), "");
    }

    #[test]
    fn concat_is_o1_and_flattens() {
        let a = Rope::leaf("foo");
        let b = Rope::leaf("bar");
        let ab = a.concat(&b);
        assert_eq!(ab.len(), 6);
        assert_eq!(ab.materialize(), "foobar");
        // Originals are untouched (persistent).
        assert_eq!(a.materialize(), "foo");
        assert_eq!(b.materialize(), "bar");
    }

    #[test]
    fn concat_with_empty_shares_the_other_side() {
        let a = Rope::leaf("x");
        let empty = Rope::new();
        assert_eq!(a.concat(&empty).materialize(), "x");
        assert_eq!(empty.concat(&a).materialize(), "x");
    }

    #[test]
    fn repeated_append_builds_correctly() {
        // The `s += part` loop shape: a deeply left-nested rope.
        let mut r = Rope::new();
        let mut expected = String::new();
        for i in 0..50 {
            let part = i.to_string();
            r = r.push_str(&part);
            expected.push_str(&part);
        }
        assert_eq!(r.len(), expected.len());
        assert_eq!(r.materialize(), expected);
    }

    #[test]
    fn deeply_nested_rope_flattens_without_recursion() {
        // A pathologically deep rope (10k deep) must flatten iteratively.
        let mut r = Rope::leaf("a");
        for _ in 0..10_000 {
            r = r.push_str("b");
        }
        let s = r.materialize();
        assert_eq!(s.len(), 10_001);
        assert!(s.starts_with("ab"));
        assert!(s.ends_with("bb"));
    }

    /// A deeply nested rope must also *drop* iteratively — a recursive free
    /// overflows small stacks (this regressed CI on Windows' 1 MB test
    /// threads). Run on a deliberately tiny stack to catch it everywhere.
    #[cfg(feature = "std")]
    #[test]
    fn deeply_nested_rope_drops_without_overflow() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let mut r = Rope::leaf("a");
                for _ in 0..100_000 {
                    r = r.push_str("b");
                }
                assert_eq!(r.materialize().len(), 100_001);
                // `r` drops here, on the 256 KB stack.
            })
            .expect("spawn")
            .join()
            .expect("the deep rope dropped without overflowing the stack");
    }

    #[test]
    fn display_matches_to_string() {
        let r = Rope::leaf("a").push_str("b").concat(&Rope::leaf("cd"));
        assert_eq!(alloc::format!("{r}"), r.materialize());
        assert_eq!(r.materialize(), "abcd");
    }

    #[test]
    fn non_surrogate_string_is_byte_identical() {
        // The common case: a normal string round-trips byte-for-byte through the
        // rope, exactly as before the WTF-8 storage change.
        for s in ["", "hello", "héllo 中 😀 mix"] {
            let r = Rope::from(s);
            assert_eq!(r.materialize_bytes(), s.as_bytes());
            assert_eq!(r.materialize(), s);
            assert_eq!(r.len(), s.len());
        }
    }

    #[test]
    fn lone_surrogate_round_trips_through_rope() {
        // "a\uD800b" stored via from_wtf8 survives losslessly in the bytes,
        // while the lossy String form replaces the surrogate with U+FFFD.
        let bytes = crate::wtf8::from_utf16(&[0x61, 0xD800, 0x62]);
        let r = Rope::from_wtf8(bytes.clone());
        assert_eq!(r.materialize_bytes(), bytes);
        assert_eq!(r.materialize(), "a\u{FFFD}b");
        assert_eq!(alloc::format!("{r}"), "a\u{FFFD}b");
        assert_eq!(r.len(), bytes.len()); // byte length, not unit length
    }

    #[test]
    fn concat_preserves_surrogate_bytes() {
        let left = Rope::from("x");
        let right = Rope::from_wtf8(crate::wtf8::from_utf16(&[0xD800]));
        let joined = left.concat(&right);
        let mut expected = alloc::vec![b'x'];
        expected.extend_from_slice(&crate::wtf8::from_utf16(&[0xD800]));
        assert_eq!(joined.materialize_bytes(), expected);
    }

    #[test]
    fn from_bytes_matches_from_wtf8() {
        let bytes = crate::wtf8::from_utf16(&[0xDC00, 0x41]);
        assert_eq!(
            Rope::from_bytes(&bytes).materialize_bytes(),
            Rope::from_wtf8(bytes.clone()).materialize_bytes()
        );
    }

    #[test]
    fn as_leaf_bytes_borrows_unconcatenated_leaf() {
        // A plain leaf exposes its bytes zero-copy.
        let leaf = Rope::leaf("hello");
        assert_eq!(leaf.as_leaf_bytes(), Some(&b"hello"[..]));
        // A surrogate-bearing leaf is exposed losslessly.
        let bytes = crate::wtf8::from_utf16(&[0xD800]);
        let surr = Rope::from_wtf8(bytes.clone());
        assert_eq!(surr.as_leaf_bytes(), Some(&bytes[..]));
        // Two short leaves join into one leaf…
        let small = Rope::leaf("foo").concat(&Rope::leaf("bar"));
        assert_eq!(small.as_leaf_bytes(), Some(&b"foobar"[..]));
        // …a short one and more share an append buffer, still one slice…
        let (a, b) = ("f".repeat(40), "b".repeat(40));
        let built = Rope::leaf(&a).concat(&Rope::leaf(&b));
        assert_eq!(
            built.as_leaf_bytes(),
            Some(alloc::format!("{a}{b}").as_bytes())
        );
        // …but a long concatenation has no single backing slice.
        let b = "b".repeat(BUILD_APPEND_MIN + 1);
        let joined = Rope::leaf(&a).concat(&Rope::leaf(&b));
        assert_eq!(joined.as_leaf_bytes(), None);
        // …though its materialized bytes still agree with the borrow-or-copy path.
        assert_eq!(
            joined.materialize_bytes(),
            alloc::format!("{a}{b}").as_bytes()
        );
    }

    #[test]
    fn small_strings_are_inline_and_nodes_stay_compact() {
        assert!(core::mem::size_of::<Node>() <= 40);
        for n in [0, 1, SMALL_CAP - 1, SMALL_CAP, SMALL_CAP + 1, 100] {
            let s = "x".repeat(n);
            let r = Rope::leaf(&s);
            assert_eq!(matches!(*r.0, Node::Small { .. }), n <= SMALL_CAP);
            assert_eq!(r.len(), n);
            assert_eq!(r.utf16_len(), n);
            assert_eq!(r.as_leaf_str(), Some(s.as_str()));
            assert_eq!(r.materialize(), s);
            let rr = r.concat(&Rope::leaf("\u{e9}"));
            assert_eq!(rr.materialize(), alloc::format!("{s}\u{e9}"));
            assert_eq!(rr.utf16_len(), n + 1);
        }
        // A deep rope over inline leaves.
        let mut r = Rope::leaf("");
        for _ in 0..1000 {
            r = r.concat(&Rope::leaf(&"y".repeat(70)));
        }
        assert_eq!(r.utf16_len(), 70_000);
        assert_eq!(r.materialize().len(), 70_000);
    }

    /// An inline node's bytes past its length are zero, however it was made
    /// (compiled code reads them as a key's padded text).
    #[test]
    fn small_tails_are_zero() {
        let check = |r: &Rope| {
            let Node::Small { len, buf, .. } = &*r.0 else {
                panic!("expected an inline node");
            };
            assert!(buf[usize::from(*len)..].iter().all(|&b| b == 0));
        };
        for n in 0..=SMALL_CAP {
            check(&Rope::leaf(&"x".repeat(n)));
            check(&Rope::from_bytes(&alloc::vec![b'q'; n]));
            check(&Rope::small_with(n, |b| b.fill(b'z')).expect("short"));
        }
        for n in 0..SMALL_CAP {
            check(&Rope::leaf(&"k".repeat(n)).concat(&Rope::leaf("1")));
        }
        for i in [0i64, 7, -7, 12345, -9_007_199_254_740_991] {
            check(&Rope::from_int(i));
            for n in 0..SMALL_CAP {
                for first in [false, true] {
                    if let Some(r) = Rope::leaf(&"k".repeat(n)).small_with_int(i, first) {
                        check(&r);
                    }
                }
            }
        }
        for a in 0..SMALL_CAP {
            for b in 0..=SMALL_CAP - a {
                check(&Rope::leaf(&"a".repeat(a)).concat(&Rope::leaf(&"b".repeat(b))));
            }
        }
    }

    #[test]
    fn integers_print_as_their_decimal_text() {
        let vals = [
            0i64,
            1,
            -1,
            9,
            10,
            99,
            100,
            -100,
            12_345,
            2_147_483_647,
            -2_147_483_648,
            999_999_999_999_999,
            1_000_000_000_000_000,
            9_007_199_254_740_991,
            -9_007_199_254_740_991,
        ];
        for i in vals {
            let text = alloc::format!("{i}");
            let r = Rope::from_int(i);
            assert_eq!(r.materialize(), text);
            assert_eq!(r.utf16_len(), text.len());
            for pre in [
                "",
                "k",
                "é",
                "a_prefix_of_twenty_b",
                "a_prefix_of_exactly_thirty_bytes__",
            ] {
                let p = Rope::leaf(pre);
                match p.small_with_int(i, false) {
                    Some(r) => {
                        assert_eq!(r.materialize(), alloc::format!("{pre}{i}"));
                        assert_eq!(r.utf16_len(), pre.chars().count() + text.len());
                    }
                    None => assert!(pre.len() + text.len() > SMALL_CAP),
                }
                match p.small_with_int(i, true) {
                    Some(r) => assert_eq!(r.materialize(), alloc::format!("{i}{pre}")),
                    None => assert!(pre.len() + text.len() > SMALL_CAP),
                }
            }
        }
    }

    #[test]
    fn short_appends_to_an_append_buffer_keep_every_prefix() {
        let mut s = Rope::from_bytes(&[b'z'; SMALL_JOIN + 1]).concat(&Rope::leaf("é"));
        let mut kept = alloc::vec::Vec::new();
        for i in 0..2000 {
            let piece = match i % 4 {
                0 => "ab",
                1 => "é",
                2 => "thirty-seven-bytes-of-plain-ascii-txt",
                _ => "中",
            };
            let next = s.concat(&Rope::leaf(piece));
            assert_eq!(next.len(), s.len() + piece.len());
            if i % 97 == 0 {
                kept.push((s.clone(), s.materialize()));
            }
            s = next;
        }
        for (r, text) in &kept {
            assert_eq!(&r.materialize(), text);
            assert_eq!(r.utf16_len(), text.encode_utf16().count());
        }
        assert_eq!(s.utf16_len(), s.materialize().encode_utf16().count());
    }

    #[test]
    fn clone_is_cheap_and_shares() {
        let a = Rope::leaf("shared");
        let b = a.clone();
        // Same underlying allocation.
        assert!(Rc::ptr_eq(&a.0, &b.0));
        assert_eq!(b.materialize(), "shared");
    }
}

#[cfg(test)]
mod utf16_len_tests {
    use super::*;

    /// `utf16_len` must be memoized: `String.length` is read in loop conditions,
    /// so an O(n) recount per read makes `for (i = 0; i < s.length; i++)`
    /// quadratic.
    #[test]
    fn utf16_len_memoizes() {
        let leaf = Rope::from_bytes(&b"x".repeat(1000));
        assert_eq!(leaf.utf16_len(), 1000);
        assert_eq!(leaf.utf16_len(), 1000);
        // Astral characters count 2 units each.
        let astral = Rope::from_wtf8(crate::wtf8::from_utf16(&[0xD83D, 0xDE00]));
        assert_eq!(astral.utf16_len(), 2);
        assert_eq!(astral.utf16_len(), 2);
        // A concat sums its sides, and caches at every level it walks.
        let cat = leaf.concat(&astral);
        assert_eq!(cat.utf16_len(), 1002);
        assert_eq!(cat.utf16_len(), 1002);
    }

    /// A `+=`-built string is a left-leaning tree whose depth is its append
    /// count, so a *recursive* length walk overflows the native stack. That is
    /// not hypothetical: it aborted test262 workers on the
    /// `RegExp/property-escapes` tests, whose `buildString` helper appends in a
    /// loop. Deep enough here to blow a recursive implementation.
    #[test]
    fn utf16_len_survives_a_deep_rope() {
        let mut r = Rope::leaf("");
        for _ in 0..200_000 {
            r = r.concat(&Rope::leaf("ab"));
        }
        assert_eq!(r.utf16_len(), 400_000);
        // Second read comes from the root cache.
        assert_eq!(r.utf16_len(), 400_000);
    }
}
