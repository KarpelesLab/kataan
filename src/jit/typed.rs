//! Inline **typed-array element access** for the generic tier: `ta[i]` and
//! `ta[i] = n` on an `Int8Array` .. `Float64Array` view without a helper call.
//!
//! The fast path does exactly what `Realm::typed_get_fast` /
//! `Realm::typed_set_fast` (`src/realm/typed_fast.rs`) do, and answers to
//! the same cases: a view (`Cell::TypedArray`) over an `Owned` byte store
//! whose whole extent `[byteOffset, byteOffset + length * size)` lies inside
//! the store's current bytes, an index below the view's `length`, and a kind
//! below Float16 that is not a BigInt kind. Such a view is in bounds whatever
//! kind of view it is (a fixed-length view goes out of bounds only when that
//! extent passes the end of a shrunk resizable buffer, a length-tracking one
//! only when its offset does), so the index names bytes that exist and
//! `IsValidIntegerIndex` holds: a read is the element, a write stores it, no
//! prototype is consulted (integer-indexed exotic objects never do for a
//! numeric key). Every other case — out of bounds (`undefined` / a no-op), a
//! detached buffer (its views have length 0 and its store none), a shared or
//! external store, a BigInt or Float16 view — goes to the helper, which runs
//! the general algorithm. Nothing is cached across accesses: the view's
//! length, offset and kind and the store's pointer and length are read on
//! every access, so a resize, a detach or a transfer between two accesses
//! is seen by the next one.
//!
//! A write takes a Number only (any other value's `ToNumber` may run code,
//! which the helper does before anything else); the float kinds store any
//! Number (Float32 rounding as the interpreter's `as f32` does, with the
//! same instruction), the integer kinds one below 2^63 in magnitude
//! (`ToInt8` .. `ToUint32` of one is its truncation's low bits;
//! Uint8Clamped rounds it half to even, then clamps it to `0..=255`): a
//! NaN, an infinity or a larger Number goes to the helper.
//!
//! A read of a Float32 / Float64 element that holds a NaN gives the
//! canonical NaN, as `NanBox::number` does (any NaN pattern can be written
//! into the bytes through another view).

use super::call_seq::{cc, r};
use super::{JitLayout, Label, X64Assembler};

/// The typed-array element layout compiled code reads, as offsets from a
/// heap slot (see [`crate::cell::TypedCellLayout`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TypedLayout {
    /// `Cell::TypedArray`'s discriminant byte.
    pub typed_disc: u8,
    /// A view's buffer handle: its arena index (`u32`) and generation (`u16`).
    pub off_buf_index: i32,
    pub off_buf_gen: i32,
    /// A view's `byte_offset` and `length` (`usize`) and `kind` (`u8`).
    pub off_byte_offset: i32,
    pub off_length: i32,
    pub off_kind: i32,
    /// `Cell::Bytes`'s discriminant byte.
    pub bytes_disc: u8,
    /// The byte store's tag byte, and `ByteStore::Owned`'s tag.
    pub off_store_tag: i32,
    pub owned_tag: u8,
    /// An `Owned` store's data pointer and length.
    pub off_data_ptr: i32,
    pub off_data_len: i32,
}

/// The process's [`TypedLayout`] (probed once).
pub(crate) fn typed_layout() -> &'static TypedLayout {
    use std::sync::OnceLock;
    static LAYOUT: OnceLock<TypedLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        let t = crate::cell::jit_typed_layout();
        let slot = crate::heap::jit_slot_layout(crate::cell::Cell::Box(
            crate::nanbox::NanBox::undefined(),
        ));
        let base = slot.value_off;
        let at = |v: usize| i32::try_from(base + v).expect("typed layout offset fits in i32");
        // A rooted walk proves its slot occupied by the cell-tag compare
        // alone (`super::handle`): neither tag may be a vacant slot's.
        assert_ne!(t.typed_disc, slot.vacant_tag);
        assert_ne!(t.bytes_disc, slot.vacant_tag);
        assert_eq!(
            i32::try_from(base).ok(),
            Some(super::compute_jit_layout().off_cell_tag)
        );
        TypedLayout {
            typed_disc: t.typed_disc,
            off_buf_index: at(t.buffer_index_off),
            off_buf_gen: at(t.buffer_gen_off),
            off_byte_offset: at(t.byte_offset_off),
            off_length: at(t.length_off),
            off_kind: at(t.kind_off),
            bytes_disc: t.bytes_disc,
            off_store_tag: at(t.store_tag_off),
            owned_tag: t.owned_tag,
            off_data_ptr: at(t.data_ptr_off),
            off_data_len: at(t.data_len_off),
        }
    })
}

/// Whether a later site of the same iteration takes op `pc`'s way as filled
/// by it (`recv_cache::SitePlan::known`): `pc`'s typed-array path, which
/// fills nothing, must then not continue in the fast version.
pub(super) fn relied_on(rc: &super::recv_cache::RecvPlan, pc: usize) -> bool {
    rc.sites
        .iter()
        .any(|s| s.is_some_and(|p| p.known && p.pred == Some(pc)))
}

/// Takes every array-family site of `rc` out of the receiver cache: a way
/// only an array fills is no help to a typed-array view (a compare, a miss,
/// a walk to a failing family check, then the view's own walk), and no site
/// may take a way as filled by an earlier one (see [`relied_on`]).
pub(super) fn drop_array_sites(rc: &mut super::recv_cache::RecvPlan) {
    for s in &mut rc.sites {
        if s.is_some_and(|p| p.family == super::recv_cache::Family::Arr) {
            *s = None;
        }
    }
}

/// A **hoisted view**: the frame slots (`rbp`-relative) where a speculative
/// loop's fast version keeps what its entry check found about a typed-array
/// view held in a register (`numeric::NumGen`'s
/// `hoisted_view`): the view's kind (a byte, [`NO_VIEW`] when the register
/// holds anything the inline path does not serve), its length, and the
/// address of its first element.
///
/// They stay right from that check to the loop's back-edge: a view's
/// length, offset and kind never change, and its store's bytes move or
/// shrink (a resize, a detach, a transfer) only in Rust code, which the
/// fast version never runs — every one of its slow paths leaves for the
/// plain copy, and every way back into it (the loop's entry, a back-edge
/// poll's return, an inner header's entry from the copy) checks afresh
/// ([`t_hoist`](X64Assembler::t_hoist)). A register the loop never writes
/// holds the same view throughout; for one it may write, `held` keeps the
/// value the facts are about, and an access compares the register with it
/// first (a different value takes the access's ordinary path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ViewSlots {
    pub(crate) kind: i32,
    pub(crate) len: i32,
    pub(crate) base: i32,
    pub(crate) held: Option<i32>,
}

/// Where [`t_hoist`](X64Assembler::t_hoist) finds the value to check: a
/// register (frame displacement), or a global binding whose validated
/// address is in a frame slot.
#[derive(Clone, Copy, Debug)]
pub(crate) enum HoistFrom {
    Reg(i32),
    Binding(i32),
}

/// [`ViewSlots::kind`] of a register that is not a view the inline path
/// serves (above every served kind).
pub(crate) const NO_VIEW: u8 = 0xff;

/// Where an element access finds its view's facts: walked from the receiver
/// (`rcx` / `r9` from [`t_view`](X64Assembler::t_view)), or in a hoisted
/// view's frame slots.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Src {
    View,
    Hoisted(ViewSlots),
}

/// The element kinds the fast path serves (`TYPED_ARRAY_KINDS` indices), in
/// the order a site tests them, with their element size's log2.
pub(crate) const INT8: u8 = 0;
pub(crate) const UINT8: u8 = 1;
pub(crate) const UINT8_CLAMPED: u8 = 2;
pub(crate) const INT16: u8 = 3;
pub(crate) const UINT16: u8 = 4;
pub(crate) const INT32: u8 = 5;
pub(crate) const UINT32: u8 = 6;
pub(crate) const FLOAT32: u8 = 7;
pub(crate) const FLOAT64: u8 = 8;

pub(crate) const fn size_log2(kind: u8) -> u8 {
    match kind {
        INT8 | UINT8 | UINT8_CLAMPED => 0,
        INT16 | UINT16 => 1,
        INT32 | UINT32 | FLOAT32 => 2,
        _ => 3,
    }
}

/// x86 condition codes beyond [`cc`]'s.
const CC_O: u8 = 0x0;
const CC_NP: u8 = 0xb;
const CC_NS: u8 = 0x9;
const CC_S: u8 = 0x8;
const CC_LE: u8 = 0xe;

impl X64Assembler {
    /// With `rcx` the slot of a rooted cell (its walk done, the arena
    /// loaded): to `miss` unless the cell is a typed-array view over an
    /// `Owned` byte store, whose slot is then in `r9`. Clobbers `rax`,
    /// `r8` (audit mode) and `r11`.
    pub(crate) fn t_view(&mut self, l: &JitLayout, t: &TypedLayout, miss: Label) {
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, t.typed_disc);
        self.g_jcc(cc::NE, miss);
        // The buffer handle (a field of a rooted cell, so rooted itself):
        // its index in `eax`, and in audit mode its generation above it.
        self.g_mem(false, &[0x8b], r::RAX, r::RCX, t.off_buf_index); // mov eax, [rcx+idx]
        if super::handle::handle_audit() {
            self.g_mem(false, &[0x0f, 0xb7], r::R11, r::RCX, t.off_buf_gen); // movzx r11d, word
            self.raw(&[0x49, 0xc1, 0xe3, 0x20]); // shl r11, 32
            self.raw(&[0x4c, 0x09, 0xd8]); // or rax, r11
        }
        self.p_slot_rooted(l, r::R9, miss);
        self.g_cmp_byte_imm(r::R9, l.off_cell_tag, t.bytes_disc);
        self.g_jcc(cc::NE, miss);
        self.g_cmp_byte_imm(r::R9, t.off_store_tag, t.owned_tag);
        self.g_jcc(cc::NE, miss);
    }

    /// With `rax` an integer index (and, for [`Src::View`], `rcx` / `r9`
    /// from [`t_view`](Self::t_view)): to `miss` unless it is below the
    /// view's length; then `r8` = the address of the view's first element.
    fn t_index(&mut self, t: &TypedLayout, src: Src, miss: Label) {
        match src {
            Src::View => {
                self.g_cmp_mem(r::RAX, r::RCX, t.off_length);
                self.g_jcc(cc::AE, miss); // past the length, or negative
                self.g_load(r::R8, r::R9, t.off_data_ptr);
                self.g_add_mem(r::R8, r::RCX, t.off_byte_offset);
            }
            Src::Hoisted(v) => {
                self.g_cmp_mem(r::RAX, r::RBP, v.len);
                self.g_jcc(cc::AE, miss); // past the length, or negative
                self.g_load(r::R8, r::RBP, v.base);
            }
        }
    }

    /// To `other` unless the view's kind is `kind`; then (a walked view)
    /// to `miss` unless its whole extent lies in its store's bytes, which
    /// a hoisted view's entry check proved. Clobbers `r11`.
    fn t_kind(&mut self, t: &TypedLayout, src: Src, kind: u8, other: Label, miss: Label) {
        match src {
            Src::View => {
                self.g_cmp_byte_imm(r::RCX, t.off_kind, kind);
                self.g_jcc(cc::NE, other);
                self.t_extent(t, size_log2(kind), miss);
            }
            Src::Hoisted(v) => {
                self.g_cmp_byte_imm(r::RBP, v.kind, kind);
                self.g_jcc(cc::NE, other);
            }
        }
    }

    /// To `miss` unless `byteOffset + length << s` (`s` the element size's
    /// log2) is within the store's length. Clobbers `r11`.
    fn t_extent(&mut self, t: &TypedLayout, s: u8, miss: Label) {
        self.g_load(r::R11, r::RCX, t.off_length);
        if s > 0 {
            self.raw(&[0x49, 0xc1, 0xe3, s]); // shl r11, s
        }
        self.g_add_mem(r::R11, r::RCX, t.off_byte_offset);
        self.g_cmp_mem(r::R11, r::R9, t.off_data_len);
        self.g_jcc(cc::A, miss);
    }

    /// `[prefix] REX op reg, [r8 + rax << scale]` (`w`: REX.W).
    fn t_elem(&mut self, prefix: Option<u8>, w: bool, opcode: &[u8], reg: u8, scale: u8) {
        if let Some(p) = prefix {
            self.code.push(p);
        }
        self.code
            .push(0x41 | (u8::from(w) << 3) | (((reg >> 3) & 1) << 2));
        self.code.extend_from_slice(opcode);
        // ModRM mod=00 rm=100 (SIB); SIB index=rax, base=r8 (with REX.B).
        self.code.push(((reg & 7) << 3) | 4);
        self.code.push(scale << 6);
    }

    /// `[pfx] [REX] 0F opc /r` over two registers (`reg`, `rm`: register
    /// numbers, `w`: REX.W).
    fn t_rr(&mut self, pfx: u8, w: bool, opc: u8, reg: u8, rm: u8) {
        if pfx != 0 {
            self.code.push(pfx);
        }
        let rex = 0x40 | (u8::from(w) << 3) | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code
            .extend_from_slice(&[0x0f, opc, 0xc0 | ((reg & 7) << 3) | (rm & 7)]);
    }

    /// The element read: with `rax` an integer index (and, for
    /// [`Src::View`], `rcx` / `r9` from [`t_view`](Self::t_view)), `rax` =
    /// the element (a `NanBox` Number's bits) and on to `got`, else to
    /// `miss`. With `to = Some((x, past))` (`x` an SSE register other than
    /// `xmm0`), an element whose Number is no NaN goes into `xmm x` and on
    /// to `past` instead, skipping the round trip through `rax` that
    /// `got`'s code (`xmm x = rax`, then `past`) would make; a NaN still
    /// goes through `got`. Clobbers `r8`, `r11`, `xmm0`.
    pub(crate) fn t_get(
        &mut self,
        t: &TypedLayout,
        src: Src,
        miss: Label,
        got: Label,
        to: Option<(u8, Label)>,
    ) {
        self.t_index(t, src, miss);
        // (The kind slot is loop-invariant for a hoisted view: the chain's
        // branches are predicted; the float kinds after the commonest
        // integer ones.)
        for kind in [
            UINT8,
            INT32,
            FLOAT64,
            FLOAT32,
            UINT16,
            INT16,
            INT8,
            UINT32,
            UINT8_CLAMPED,
        ] {
            let next = self.new_label();
            self.t_kind(t, src, kind, next, miss);
            let s = size_log2(kind);
            // Where the Number goes: `xmm x`, or `xmm0` (then `rax`).
            let x = to.map_or(0, |(x, _)| x);
            match kind {
                INT8 => self.t_elem(None, false, &[0x0f, 0xbe], r::RAX, s), // movsx eax, byte
                UINT8 | UINT8_CLAMPED => self.t_elem(None, false, &[0x0f, 0xb6], r::RAX, s),
                INT16 => self.t_elem(None, false, &[0x0f, 0xbf], r::RAX, s), // movsx eax, word
                UINT16 => self.t_elem(None, false, &[0x0f, 0xb7], r::RAX, s), // movzx eax, word
                INT32 | UINT32 => self.t_elem(None, false, &[0x8b], r::RAX, s), // mov eax, dword
                FLOAT32 => self.t_elem(Some(0xf3), false, &[0x0f, 0x5a], x, s), // cvtss2sd xmm x
                _ => self.t_elem(Some(0xf2), false, &[0x0f, 0x10], x, s),    // movsd xmm x
            }
            if !matches!(kind, FLOAT32 | FLOAT64) {
                // (`cvtsi2sd` writes only the low lane: without this, each
                // conversion would wait for the last write of `xmm x`.)
                self.t_rr(0, false, 0x57, x, x); // xorps xmm x, xmm x
            }
            match kind {
                // cvtsi2sd xmm x, rax (the zero-extended `u32`).
                UINT32 => self.t_rr(0xf2, true, 0x2a, x, r::RAX),
                FLOAT32 | FLOAT64 => {
                    // A NaN is the canonical one.
                    let num = self.new_label();
                    self.t_rr(0x66, false, 0x2e, x, x); // ucomisd xmm x, xmm x
                    self.g_jcc(CC_NP, num);
                    self.g_movabs(r::RAX, crate::nanbox::NanBox::number(f64::NAN).to_bits());
                    self.jmp(got);
                    self.bind(num);
                }
                // cvtsi2sd xmm x, eax (the sign- or zero-extended value).
                _ => self.t_rr(0xf2, false, 0x2a, x, r::RAX),
            }
            match to {
                Some((_, past)) => self.jmp(past),
                None => {
                    self.raw(&[0x66, 0x48, 0x0f, 0x7e, 0xc0]); // movq rax, xmm0
                    self.jmp(got);
                }
            }
            self.bind(next);
        }
        self.jmp(miss);
    }

    /// The element read of the integer version (`numeric::int32`): with
    /// `rax` an integer index (and, for [`Src::View`], `rcx` / `r9` from
    /// [`t_view`](Self::t_view)), `eax` = the element of an integer kind,
    /// an int32 (a Uint32 one past `i32::MAX` only when `wrap`: its low 32
    /// bits, for a reader that takes `ToInt32` of it), and on to `got`;
    /// else to `miss`. Clobbers `r8`, `r11`.
    pub(crate) fn t_get_i32(
        &mut self,
        t: &TypedLayout,
        src: Src,
        wrap: bool,
        miss: Label,
        got: Label,
    ) {
        self.t_index(t, src, miss);
        for kind in [UINT8, INT32, UINT16, INT16, INT8, UINT32, UINT8_CLAMPED] {
            let next = self.new_label();
            self.t_kind(t, src, kind, next, miss);
            let s = size_log2(kind);
            match kind {
                INT8 => self.t_elem(None, false, &[0x0f, 0xbe], r::RAX, s), // movsx eax, byte
                UINT8 | UINT8_CLAMPED => self.t_elem(None, false, &[0x0f, 0xb6], r::RAX, s),
                INT16 => self.t_elem(None, false, &[0x0f, 0xbf], r::RAX, s), // movsx eax, word
                UINT16 => self.t_elem(None, false, &[0x0f, 0xb7], r::RAX, s), // movzx eax, word
                _ => self.t_elem(None, false, &[0x8b], r::RAX, s),           // mov eax, dword
            }
            if kind == UINT32 && !wrap {
                self.raw(&[0x85, 0xc0]); // test eax, eax
                self.g_jcc(CC_S, miss); // past i32::MAX
            }
            self.jmp(got);
            self.bind(next);
        }
        self.jmp(miss);
    }

    /// [`t_get_i32`](Self::t_get_i32) of a hoisted view (a speculative
    /// region's integer version, `numeric::int32`), touching no register
    /// but `rax` and `r11`: with `rax` an integer index, to `miss` unless
    /// the register at frame displacement `recv` holds view `v` and the
    /// index is below its length, then `eax` = the element of an integer
    /// kind (a Uint32 one past `i32::MAX` only when `wrap`), else to `miss`.
    pub(crate) fn t_hoisted_get_i32(&mut self, v: ViewSlots, recv: i32, wrap: bool, miss: Label) {
        if let Some(h) = v.held {
            // (The value the facts are about: the index rides in `r11`.)
            self.g_mov(r::R11, r::RAX);
            self.load_rax(recv);
            self.g_cmp_mem(r::RAX, r::RBP, h);
            self.g_jcc(cc::NE, miss);
            self.g_mov(r::RAX, r::R11);
        }
        self.g_cmp_mem(r::RAX, r::RBP, v.len);
        self.g_jcc(cc::AE, miss); // past the length, or negative
        // (No view, `NO_VIEW`, is no kind below: to `miss` with no load.)
        self.g_load(r::R11, r::RBP, v.base);
        let got = self.new_label();
        for kind in [UINT8, INT32, UINT16, INT16, INT8, UINT32, UINT8_CLAMPED] {
            let next = self.new_label();
            self.g_cmp_byte_imm(r::RBP, v.kind, kind);
            self.g_jcc(cc::NE, next);
            let sib = (size_log2(kind) << 6) | 0x03; // [r11 + rax << s]
            match kind {
                INT8 => self.raw(&[0x41, 0x0f, 0xbe, 0x04, sib]), // movsx eax, byte
                UINT8 | UINT8_CLAMPED => self.raw(&[0x41, 0x0f, 0xb6, 0x04, sib]),
                INT16 => self.raw(&[0x41, 0x0f, 0xbf, 0x04, sib]), // movsx eax, word
                UINT16 => self.raw(&[0x41, 0x0f, 0xb7, 0x04, sib]), // movzx eax, word
                _ => self.raw(&[0x41, 0x8b, 0x04, sib]),           // mov eax, dword
            }
            if kind == UINT32 && !wrap {
                self.raw(&[0x85, 0xc0]); // test eax, eax
                self.g_jcc(CC_S, miss); // past i32::MAX
            }
            self.jmp(got);
            self.bind(next);
        }
        self.jmp(miss);
        self.bind(got);
    }

    /// A hoisted view's entry check ([`ViewSlots`]): when the register at
    /// frame displacement `recv` holds a view the inline path serves (as
    /// [`t_view`](Self::t_view) and a kind below Float16), wholly inside its
    /// store's bytes, its kind, length and first element's address go to
    /// `v`'s slots; else its kind slot says [`NO_VIEW`]. Clobbers `rax`,
    /// `rcx`, `r8`, `r9`, `r11` (and, unpinned, `rdx` / `r10`).
    pub(crate) fn t_hoist(
        &mut self,
        l: &JitLayout,
        t: &TypedLayout,
        realm: i32,
        from: HoistFrom,
        v: ViewSlots,
    ) {
        let none = self.new_label();
        let done = self.new_label();
        let ok = self.new_label();
        self.p_arena(realm, &l.props);
        match from {
            HoistFrom::Reg(d) => self.load_rax(d),
            HoistFrom::Binding(slot) => {
                self.g_load(r::RAX, r::RBP, slot);
                self.g_load(r::RAX, r::RAX, 0);
            }
        }
        if let Some(h) = v.held {
            self.g_store(r::RBP, h, r::RAX);
        }
        self.p_is_handle(r::RCX, none);
        self.p_slot_rooted(l, r::RCX, none);
        self.t_view(l, t, none);
        for (last, s) in [(UINT8_CLAMPED, 0), (UINT16, 1), (FLOAT32, 2), (FLOAT64, 3)] {
            let next = self.new_label();
            self.g_cmp_byte_imm(r::RCX, t.off_kind, last);
            self.g_jcc(cc::A, next);
            self.t_extent(t, s, none);
            self.jmp(ok);
            self.bind(next);
        }
        self.jmp(none);
        self.bind(ok);
        self.g_load(r::RAX, r::RCX, t.off_length);
        self.g_store(r::RBP, v.len, r::RAX);
        self.g_load(r::RAX, r::R9, t.off_data_ptr);
        self.g_add_mem(r::RAX, r::RCX, t.off_byte_offset);
        self.g_store(r::RBP, v.base, r::RAX);
        self.g_mem(false, &[0x0f, 0xb6], r::RAX, r::RCX, t.off_kind); // movzx eax, byte
        // Bit 8 (above the byte every access compares): `.length` may not be
        // the length slot (see `t_hoisted_len`).
        let intact = self.new_label();
        self.t_len_intact(realm, r::R8, intact);
        self.raw(&[0x0d, 0x00, 0x01, 0x00, 0x00]); // or eax, 0x100
        self.bind(intact);
        self.g_store(r::RBP, v.kind, r::RAX);
        self.jmp(done);
        self.bind(none);
        self.g_store_imm32(r::RBP, v.kind, i32::from(NO_VIEW));
        if let Some(h) = v.held {
            // (No register holds the throw sentinel: no access matches.)
            self.g_movabs(r::RAX, crate::nanbox::NanBox::jit_throw_bits());
            self.g_store(r::RBP, h, r::RAX);
        }
        self.bind(done);
    }

    /// To `other` unless hoisted view `v` holds a view ([`NO_VIEW`]) and,
    /// for a register the loop may write, the register at frame
    /// displacement `recv` holds the very value its facts are about.
    /// Clobbers `rax`.
    pub(crate) fn t_hoisted(&mut self, v: ViewSlots, recv: i32, other: Label) {
        match v.held {
            Some(h) => {
                self.load_rax(recv);
                self.g_cmp_mem(r::RAX, r::RBP, h);
                self.g_jcc(cc::NE, other);
            }
            None => {
                self.g_cmp_byte_imm(r::RBP, v.kind, NO_VIEW);
                self.g_jcc(cc::E, other);
            }
        }
    }

    /// To `intact` when a typed array's `.length` is its length slot: no
    /// typed array has an own property and nothing on a typed array's
    /// prototype chain shadows or redefines the built-in getter, as
    /// `Realm::jit_typed_len_refresh` proved at the current prototype epoch
    /// (else on, for its helper, which re-proves it). `realm` is the realm
    /// pointer's `ctx` (`r15`) displacement. Clobbers `tmp`, `r11`.
    fn t_len_intact(&mut self, realm: i32, tmp: u8, intact: Label) {
        let stamp =
            i32::try_from(crate::realm::Realm::jit_typed_len_stamp_offset()).unwrap_or(i32::MAX);
        self.g_load(r::R11, r::R15, realm);
        self.g_movabs(tmp, crate::object::proto_epoch_addr() as u64);
        self.g_load(tmp, tmp, 0);
        self.g_cmp_mem(tmp, r::R11, stamp);
        self.g_jcc(cc::E, intact);
    }

    /// A hoisted view's `length` (see [`t_len`](Self::t_len)) when the
    /// register at frame displacement `recv` holds it (as
    /// [`t_hoisted`](Self::t_hoisted)): `rax`, else to `miss` — also when
    /// its entry check found `.length` maybe not the length slot (bit 8 of
    /// the kind slot). The entry check suffices: anything that could change
    /// that runs Rust code, after which the loop re-enters through it.
    /// Clobbers `rax`.
    pub(crate) fn t_hoisted_len(&mut self, v: ViewSlots, recv: i32, miss: Label) {
        match v.held {
            // One compare: a served kind is below `NO_VIEW`, with bit 8 clear.
            None => {
                self.g_mem(true, &[0x81], 7, r::RBP, v.kind); // cmp qword [..], imm32
                self.code
                    .extend_from_slice(&i32::from(NO_VIEW).to_le_bytes());
                self.g_jcc(cc::AE, miss);
            }
            Some(_) => {
                self.t_hoisted(v, recv, miss);
                self.g_cmp_byte_imm(r::RBP, v.kind + 1, 0);
                self.g_jcc(cc::NE, miss);
            }
        }
        self.g_load(r::RAX, r::RBP, v.len);
    }

    /// The view's `length` (what `Realm::array_length` answers for a view
    /// wholly inside its store, so in bounds, while
    /// [`t_len_intact`](Self::t_len_intact) holds): with `rcx` / `r9` from
    /// [`t_view`](Self::t_view), `rax` = the length (an integer) and on to
    /// `got`, else to `miss`. Clobbers `r11`.
    pub(crate) fn t_len(&mut self, t: &TypedLayout, realm: i32, miss: Label, got: Label) {
        let intact = self.new_label();
        self.t_len_intact(realm, r::RAX, intact);
        self.jmp(miss);
        self.bind(intact);
        for (last, s) in [(UINT8_CLAMPED, 0), (UINT16, 1), (FLOAT32, 2), (FLOAT64, 3)] {
            let next = self.new_label();
            self.g_cmp_byte_imm(r::RCX, t.off_kind, last);
            self.g_jcc(cc::A, next);
            self.t_extent(t, s, miss);
            self.g_load(r::RAX, r::RCX, t.off_length);
            self.jmp(got);
            self.bind(next);
        }
        self.jmp(miss);
    }

    /// The element write of the Number in `xmm0`: with `rax` an integer
    /// index (and, for [`Src::View`], `rcx` / `r9` from
    /// [`t_view`](Self::t_view)), stores it and on to `done`, else (nothing
    /// written) to `miss`. Clobbers `rdx`, `r8`, `r11`, `xmm1`.
    pub(crate) fn t_set(&mut self, t: &TypedLayout, src: Src, miss: Label, done: Label) {
        self.t_index(t, src, miss);
        // The view's kind byte compared with `imm`.
        macro_rules! cmp_kind {
            ($imm:expr) => {
                match src {
                    Src::View => self.g_cmp_byte_imm(r::RCX, t.off_kind, $imm),
                    Src::Hoisted(v) => self.g_cmp_byte_imm(r::RBP, v.kind, $imm),
                }
            };
        }
        // (A walked view's whole extent in its store's bytes, for an
        // element of size `1 << s`; a hoisted view's entry check proved it.)
        macro_rules! extent {
            ($s:expr) => {
                if let Src::View = src {
                    self.t_extent(t, $s, miss);
                }
            };
        }
        // The kinds told apart by ranges, the integer ones first (with no
        // kind past Float64, [`NO_VIEW`] included, ever stored):
        // Int8 / Uint8 (0, 1) store a byte, Int16 / Uint16 (3, 4) a word,
        // Int32 / Uint32 (5, 6) a dword — the truncation's low bits, the
        // same for both signs — and Uint8Clamped (2) its clamped byte.
        let floats = self.new_label();
        let clamped = self.new_label();
        cmp_kind!(UINT32);
        self.g_jcc(cc::A, floats);
        cmp_kind!(UINT8_CLAMPED);
        self.g_jcc(cc::E, clamped);
        // `ToInt8` .. `ToUint32` of a Number is its truncation's low bits,
        // which `cvttsd2si` makes for any one below 2^63 in magnitude; a
        // NaN, an infinity or a larger Number gives the "integer
        // indefinite" `i64::MIN` (as does -2^63 itself), the only value
        // `cmp rdx, 1` overflows on.
        self.raw(&[0xf2, 0x48, 0x0f, 0x2c, 0xd0]); // cvttsd2si rdx, xmm0
        self.raw(&[0x48, 0x83, 0xfa, 0x01]); // cmp rdx, 1
        self.g_jcc(CC_O, miss);
        let (byte, dword) = (self.new_label(), self.new_label());
        cmp_kind!(UINT8);
        self.g_jcc(cc::BE, byte);
        cmp_kind!(INT32);
        self.g_jcc(cc::AE, dword);
        extent!(1);
        self.t_elem(Some(0x66), false, &[0x89], r::RDX, 1); // mov word [..], dx
        self.jmp(done);
        self.bind(dword);
        extent!(2);
        self.t_elem(None, false, &[0x89], r::RDX, 2); // mov dword [..], edx
        self.jmp(done);
        // Uint8Clamped's rounds half to even (`ToUint8Clamp`), as
        // `cvtsd2si` does in the default rounding mode, before the clamp
        // (the same indefinite integer for the same Numbers).
        self.bind(clamped);
        self.raw(&[0xf2, 0x48, 0x0f, 0x2d, 0xd0]); // cvtsd2si rdx, xmm0
        self.raw(&[0x48, 0x83, 0xfa, 0x01]); // cmp rdx, 1
        self.g_jcc(CC_O, miss);
        let low = self.new_label();
        let high = self.new_label();
        self.raw(&[0x48, 0x85, 0xd2]); // test rdx, rdx
        self.g_jcc(CC_NS, low);
        self.raw(&[0x31, 0xd2]); // xor edx, edx
        self.bind(low);
        self.raw(&[0x48, 0x81, 0xfa, 0xff, 0, 0, 0]); // cmp rdx, 255
        self.g_jcc(CC_LE, high);
        self.raw(&[0xba, 0xff, 0, 0, 0]); // mov edx, 255
        self.bind(high);
        self.bind(byte);
        extent!(0);
        self.t_elem(None, false, &[0x88], r::RDX, 0); // mov byte [..], dl
        self.jmp(done);
        // The float kinds store any Number.
        self.bind(floats);
        let not64 = self.new_label();
        cmp_kind!(FLOAT64);
        self.g_jcc(cc::NE, not64);
        extent!(3);
        self.t_elem(Some(0xf2), false, &[0x0f, 0x11], 0, 3); // movsd [..], xmm0
        self.jmp(done);
        self.bind(not64);
        cmp_kind!(FLOAT32);
        self.g_jcc(cc::NE, miss);
        extent!(2);
        self.raw(&[0xf2, 0x0f, 0x5a, 0xc8]); // cvtsd2ss xmm1, xmm0
        self.t_elem(Some(0xf3), false, &[0x0f, 0x11], 1, 2); // movss [..], xmm1
        self.jmp(done);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probed layout is self-consistent: distinct tags, fields inside a
    /// slot, the store's words inside its cell.
    #[test]
    fn typed_layout_probes() {
        let t = typed_layout();
        let l = super::super::compute_jit_layout();
        assert_ne!(t.typed_disc, t.bytes_disc);
        assert_ne!(t.typed_disc, l.cell_array_disc);
        assert_ne!(t.typed_disc, l.cell_object_disc);
        for off in [
            t.off_buf_index,
            t.off_buf_gen,
            t.off_byte_offset,
            t.off_length,
            t.off_kind,
            t.off_store_tag,
            t.off_data_ptr,
            t.off_data_len,
        ] {
            assert!(off > l.off_cell_tag && off < l.slot_stride, "{off}");
        }
        assert_ne!(t.off_data_ptr, t.off_data_len);
    }

    /// The SIB element operand encodes `[r8 + rax*scale]`.
    #[test]
    fn element_operand_encoding() {
        let mut a = X64Assembler::new();
        a.t_elem(None, false, &[0x0f, 0xb6], r::RAX, 0);
        a.t_elem(Some(0xf2), false, &[0x0f, 0x10], 0, 3);
        a.t_elem(Some(0x66), false, &[0x89], r::RDX, 1);
        assert_eq!(
            a.code(),
            &[
                0x41, 0x0f, 0xb6, 0x04, 0x00, // movzx eax, byte [r8+rax]
                0xf2, 0x41, 0x0f, 0x10, 0x04, 0xc0, // movsd xmm0, [r8+rax*8]
                0x66, 0x41, 0x89, 0x14, 0x40, // mov [r8+rax*2], dx
            ]
        );
    }
}
