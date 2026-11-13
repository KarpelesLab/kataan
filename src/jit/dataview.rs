//! A compiled `view.getUint32(off, le)` / `view.setFloat64(off, v, le)`
//! with no call.
//!
//! A `DataView` keeps its `[[ViewedArrayBuffer]]`, `[[ByteOffset]]` and
//! `[[ByteLength]]` in hidden slots that never change once it is made, and
//! the general accessor reads them by name on every call
//! (`nbexec::dataview_fast`). The site's memo ([`DvMemo`], filled by
//! `MethodSite::fill_dv` from the fill helper) instead records, for the last
//! view the site saw, the byte store of its buffer, its offset and the end of
//! its extent, together with the proof `fill_math` makes for `Math.f`: while
//! the prototype epoch is the memo's stamp, the lookup of the site's key on
//! that very view finds a data property whose value is at `ptr`, and that
//! value was the built-in accessor when it was `method`'s bits (every object
//! of the chain, the view included, is watched, so a property added to any
//! of them, a change of the view's `[[Prototype]]`, or a cell swap or
//! compaction, moves the epoch; the value is read live). A call whose
//! method is already read (`Op::CallWithMethod`) compares that method with
//! the memo's instead.
//!
//! With that, the call is what `GetViewValue` / `SetViewValue` do when none
//! of their steps can run code or throw, under the same conditions as
//! `dataview_fast`: an offset that is a Number and already an integer index
//! (so `ToIndex` is the identity), a `littleEndian` that is absent or a
//! Boolean, a set's value a Number (`ToNumber` the identity; an integer
//! type's needs a truncation below 2^63 in magnitude), and an access that
//! lies inside the view, whose extent lies inside the store's **current**
//! bytes (read on every access: a resize resizes the store in place, and a
//! detach empties it, so a view over a detached buffer is never in bounds
//! here). Anything else goes to the general path, which raises the spec's
//! errors in the spec's order. Only `Owned` stores are read in place; an
//! immutable buffer's view is never memoized for a `set*`, and BigInt and
//! Float16 accessors are not served.

use super::call_seq::{cc, r};
use super::props::MethodSite;
use super::typed::TypedLayout;
use super::{JitLayout, Label, X64Assembler};
use crate::nbexec::DvAccess;

/// See the module docs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DvMemo {
    /// The view's bits, or [`super::CallSite::EMPTY`].
    recv: u64,
    /// The prototype epoch it was made under (`0`: never; the epoch starts
    /// at `1`).
    stamp: u64,
    /// The address holding the method, and the method's bits.
    ptr: u64,
    method: u64,
    /// The raw handle of the view's buffer's `Owned` byte store.
    bytes: u64,
    /// The view's `[[ByteOffset]]`, and the end of its extent (`u64::MAX`:
    /// length-tracking, the extent ends with the store).
    base: u64,
    end: u64,
    /// How many fills, and how many inline accesses, the site made (a site
    /// whose views keep changing stops filling, see `fill_dv`).
    fills: u64,
    hits: u64,
}

impl DvMemo {
    /// The empty memo.
    pub(crate) const EMPTY: Self = Self {
        recv: super::CallSite::EMPTY,
        stamp: 0,
        ptr: 0,
        method: 0,
        bytes: 0,
        base: 0,
        end: 0,
        fills: 0,
        hits: 0,
    };
}

/// Fills past which a site whose memo rarely hits stops filling.
const DV_FILLS_FREE: u64 = 64;

impl MethodSite {
    /// For an inline `DataView` accessor site: records the memo for the view
    /// `recv` (see the module docs), when the lookup of the site's key meets
    /// only objects whose links can be watched and finds a shaped data
    /// property whose value `proves` (the built-in accessor of the site's
    /// key), and `facts` (`nbexec::dv_view_facts`) describes the view.
    /// Otherwise the memo is emptied. Whether it was filled. Runs no code.
    pub(crate) fn fill_dv(
        &mut self,
        realm: &mut crate::realm::Realm,
        recv: crate::heap::Handle,
        facts: Option<(crate::heap::Handle, u64, u64)>,
        proves: impl FnOnce(&crate::realm::Realm, crate::nanbox::NanBox) -> bool,
    ) -> bool {
        let (fills, hits) = (self.dv.fills.saturating_add(1), self.dv.hits);
        self.dv = DvMemo {
            fills,
            hits,
            ..DvMemo::EMPTY
        };
        // A site that keeps seeing new views (each used for a call or two)
        // would pay the fill on top of the general path every time.
        if fills > DV_FILLS_FREE && hits < fills.saturating_mul(4) {
            return false;
        }
        let Some((bytes, base, end)) = facts else {
            return false;
        };
        if realm.multi_realm || !realm.is_ordinary_get_cell(recv) {
            return false;
        }
        let key = self.key();
        let Some(Some((holder, shape, slot))) = realm.watch_get_chain(recv, key) else {
            return false;
        };
        let Some(o) = realm.jit_object(crate::heap::Handle::from_raw(holder)) else {
            return false;
        };
        if !o
            .shaped_shape()
            .is_some_and(|s| alloc::rc::Rc::ptr_eq(s, &shape))
        {
            return false;
        }
        let (Some(ptr), Some(v)) = (o.jit_slot_ptr(slot), o.slot(slot)) else {
            return false;
        };
        if !proves(realm, v) {
            return false;
        }
        self.dv = DvMemo {
            recv: crate::nanbox::NanBox::handle(recv.to_raw()).to_bits(),
            stamp: crate::object::proto_epoch(),
            ptr: ptr as u64,
            method: v.to_bits(),
            bytes: bytes.to_raw(),
            base,
            end,
            fills,
            hits,
        };
        true
    }

    /// The offsets, from a site, of its [`DvMemo`]'s fields: `recv`,
    /// `stamp`, `ptr`, `method`, `bytes`, `base`, `end`, `hits`.
    fn dv_offsets() -> [i32; 8] {
        use core::mem::offset_of;
        let m = offset_of!(MethodSite, dv);
        let i = |v: usize| i32::try_from(m + v).expect("site offset fits in i32");
        [
            i(offset_of!(DvMemo, recv)),
            i(offset_of!(DvMemo, stamp)),
            i(offset_of!(DvMemo, ptr)),
            i(offset_of!(DvMemo, method)),
            i(offset_of!(DvMemo, bytes)),
            i(offset_of!(DvMemo, base)),
            i(offset_of!(DvMemo, end)),
            i(offset_of!(DvMemo, hits)),
        ]
    }
}

/// The operands of one inline accessor call: frame displacements of the
/// receiver, the offset, the value (a `set*`) and `littleEndian` (absent:
/// `false`), and of the destination; and, for a call whose method is
/// already read (`Op::CallWithMethod`), of that method, which is then
/// compared with the memo's instead of the holder's slot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DvOperands {
    pub(crate) callee: Option<i32>,
    pub(crate) recv: i32,
    pub(crate) off: i32,
    pub(crate) val: Option<i32>,
    pub(crate) le: Option<i32>,
    pub(crate) dst: i32,
}

/// `NanBox` bits the code compares with.
const TRUE_BITS: u64 = crate::nanbox::NanBox::boolean(true).to_bits();
const FALSE_BITS: u64 = crate::nanbox::NanBox::boolean(false).to_bits();
const UNDEF_BITS: u64 = crate::nanbox::NanBox::undefined().to_bits();
/// A Number's bits are those whose AND with this is not this.
const QNAN: u64 = 0x7ffc_0000_0000_0000;

impl X64Assembler {
    /// The accessor `acc` on the operands `o` (see the module docs): the
    /// result stored to `o.dst` and on to `done`; to `fill` when the site's
    /// memo is not for this receiver at the current epoch, to `general`
    /// for anything else, with nothing done. Clobbers `rax`, `rcx`, `rdx`,
    /// `rsi`, `rdi`, `r8`–`r11`, `xmm0`, `xmm1`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dv_access(
        &mut self,
        l: &JitLayout,
        t: &TypedLayout,
        realm_off: i32,
        site: *const MethodSite,
        acc: DvAccess,
        o: DvOperands,
        (fill, general, done): (Label, Label, Label),
    ) {
        let [recv, stamp, ptr, method, bytes, base, end, hits] = MethodSite::dv_offsets();
        let size = i32::from(acc.size);
        // The store's slot (the view is in a register, so its buffer and
        // store are live): `rcx`. (First: the arena walk clobbers `rax`,
        // `rdx`, `r10`, `r11`.)
        self.p_arena(realm_off, &l.props);
        self.g_movabs(r::R11, site as u64);
        // The memo is this receiver's, at the current epoch ...
        self.dv_slot(r::RAX, o.recv);
        self.g_cmp_mem(r::RAX, r::R11, recv);
        self.g_jcc(cc::NE, fill);
        self.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
        self.g_cmp_mem(r::R9, r::R11, stamp);
        self.g_jcc(cc::NE, fill);
        // ... and the method's slot still holds the built-in (or the method
        // read is that built-in).
        match o.callee {
            Some(c) => self.dv_slot(r::R9, c),
            None => {
                self.g_load(r::R9, r::R11, ptr);
                self.g_load(r::R9, r::R9, 0);
            }
        }
        self.g_cmp_mem(r::R9, r::R11, method);
        self.g_jcc(cc::NE, general);
        self.g_load(r::RAX, r::R11, bytes);
        self.p_slot_rooted(l, r::RCX, general);
        self.g_cmp_byte_imm(r::RCX, l.off_cell_tag, t.bytes_disc);
        self.g_jcc(cc::NE, general);
        self.g_cmp_byte_imm(r::RCX, t.off_store_tag, t.owned_tag);
        self.g_jcc(cc::NE, general);
        // `littleEndian`: `esi` = 1 for `true`, 0 for `false` or absent.
        self.g_movabs(r::RSI, 0);
        if let Some(le) = o.le {
            let set = self.new_label();
            let decided = self.new_label();
            self.dv_slot(r::RAX, le);
            self.g_movabs(r::R9, TRUE_BITS);
            self.g_cmp(r::RAX, r::R9);
            self.g_jcc(cc::E, set);
            self.g_movabs(r::R9, FALSE_BITS);
            self.g_cmp(r::RAX, r::R9);
            self.g_jcc(cc::E, decided);
            self.g_movabs(r::R9, UNDEF_BITS);
            self.g_cmp(r::RAX, r::R9);
            self.g_jcc(cc::E, decided);
            self.jmp(general);
            self.bind(set);
            self.g_movabs(r::RSI, 1);
            self.bind(decided);
        }
        // A set's value, as the bits to store: `rdi`.
        if let Some(v) = o.val {
            self.dv_slot(r::RAX, v);
            self.dv_number_check(general);
            self.raw(&[0x66, 0x48, 0x0f, 0x6e, 0xc0]); // movq xmm0, rax
            if acc.float && acc.size == 8 {
                self.g_mov(r::RDI, r::RAX);
            } else if acc.float {
                self.raw(&[0xf2, 0x0f, 0x5a, 0xc0]); // cvtsd2ss xmm0, xmm0
                self.raw(&[0x66, 0x0f, 0x7e, 0xc7]); // movd edi, xmm0
            } else {
                // The truncation's low bits (ToInt8 .. ToUint32), unless it
                // is the "indefinite" integer (NaN, ±∞, |v| >= 2^63).
                self.cvttsd2si_rax_xmm0();
                self.g_movabs(r::R9, i64::MIN as u64);
                self.g_cmp(r::RAX, r::R9);
                self.g_jcc(cc::E, general);
                self.g_mov(r::RDI, r::RAX);
            }
        }
        // The offset: a Number that is an integer in [0, 2^63) (an index
        // past the extent is a miss below), in `rax`. (`-0` is index 0.)
        self.dv_slot(r::RAX, o.off);
        self.dv_number_check(general);
        self.raw(&[0x66, 0x48, 0x0f, 0x6e, 0xc0]); // movq xmm0, rax
        self.cvttsd2si_rax_xmm0();
        self.g_test(r::RAX);
        self.g_jcc(CC_S, general); // negative, or the indefinite integer
        self.raw(&[0x0f, 0x57, 0xc9]); // xorps xmm1, xmm1
        self.raw(&[0xf2, 0x48, 0x0f, 0x2a, 0xc8]); // cvtsi2sd xmm1, rax
        self.raw(&[0x66, 0x0f, 0x2e, 0xc1]); // ucomisd xmm0, xmm1
        self.g_jcc(cc::NE, general);
        self.g_jcc(CC_P, general);
        // The limit (`r8`): the extent's end, which must lie in the store's
        // current bytes (a length-tracking view's is the store's end).
        let fits = self.new_label();
        self.g_load(r::R9, r::RCX, t.off_data_len);
        self.g_load(r::R8, r::R11, end);
        self.g_cmp(r::R8, r::R9);
        self.g_jcc(cc::BE, fits);
        self.g_cmp_imm(r::R8, -1);
        self.g_jcc(cc::NE, general); // out of bounds: the general path's TypeError
        self.g_mov(r::R8, r::R9);
        self.bind(fits);
        // r10 = the access's end in the store (offsets are below 2^53 and
        // the index below 2^63: no wrap), within the limit.
        self.g_load(r::R10, r::R11, base);
        self.g_add(r::R10, r::RAX);
        self.g_lea(r::R10, r::R10, size);
        self.g_cmp(r::R10, r::R8);
        self.g_jcc(cc::A, general);
        // r8 = the address just past the access.
        self.g_load(r::R8, r::RCX, t.off_data_ptr);
        self.g_add(r::R8, r::R10);
        self.g_inc_mem(r::R11, hits);
        if o.val.is_some() {
            self.dv_store(acc);
            self.g_movabs(r::RAX, UNDEF_BITS);
        } else {
            self.dv_load(acc);
        }
        self.store_rax(o.dst);
        self.jmp(done);
    }

    /// `mov reg, <the register at frame displacement disp>`.
    fn dv_slot(&mut self, reg: u8, disp: i32) {
        let base = self.frame_base(disp);
        self.g_load(reg, base, disp);
    }

    /// To `miss` unless `rax` holds a Number's bits. Clobbers `r9`, `rdx`.
    fn dv_number_check(&mut self, miss: Label) {
        self.g_movabs(r::R9, QNAN);
        self.g_and(r::R9, r::RAX);
        self.g_movabs(r::RDX, QNAN);
        self.g_cmp(r::R9, r::RDX);
        self.g_jcc(cc::E, miss);
    }

    /// The element's bytes just below `r8`, in the byte order `esi` says
    /// (1: little-endian), as a Number's bits in `rax`. Clobbers `xmm0`.
    fn dv_load(&mut self, acc: DvAccess) {
        let s = -i32::from(acc.size);
        let le = self.new_label();
        let got = self.new_label();
        // The raw bits in `rax` (zero-extended), byte-swapped when
        // big-endian.
        match acc.size {
            1 => self.g_mem(false, &[0x0f, 0xb6], r::RAX, r::R8, s), // movzx eax, byte
            2 => {
                self.g_mem(false, &[0x0f, 0xb7], r::RAX, r::R8, s); // movzx eax, word
                self.g_test(r::RSI);
                self.g_jcc(cc::NE, le);
                self.raw(&[0x66, 0xc1, 0xc0, 0x08]); // rol ax, 8
            }
            4 => {
                self.g_mem(false, &[0x8b], r::RAX, r::R8, s); // mov eax, dword
                self.g_test(r::RSI);
                self.g_jcc(cc::NE, le);
                self.raw(&[0x0f, 0xc8]); // bswap eax
            }
            _ => {
                self.g_load(r::RAX, r::R8, s);
                self.g_test(r::RSI);
                self.g_jcc(cc::NE, le);
                self.raw(&[0x48, 0x0f, 0xc8]); // bswap rax
            }
        }
        self.bind(le);
        if acc.float {
            if acc.size == 4 {
                self.raw(&[0x66, 0x0f, 0x6e, 0xc0]); // movd xmm0, eax
                self.raw(&[0xf3, 0x0f, 0x5a, 0xc0]); // cvtss2sd xmm0, xmm0
            } else {
                self.raw(&[0x66, 0x48, 0x0f, 0x6e, 0xc0]); // movq xmm0, rax
            }
            // A NaN is the canonical one.
            let num = self.new_label();
            self.raw(&[0x66, 0x0f, 0x2e, 0xc0]); // ucomisd xmm0, xmm0
            self.g_jcc(CC_NP, num);
            self.g_movabs(r::RAX, crate::nanbox::NanBox::number(f64::NAN).to_bits());
            self.jmp(got);
            self.bind(num);
        } else {
            if acc.signed {
                match acc.size {
                    1 => self.raw(&[0x48, 0x0f, 0xbe, 0xc0]), // movsx rax, al
                    2 => self.raw(&[0x48, 0x0f, 0xbf, 0xc0]), // movsx rax, ax
                    _ => self.raw(&[0x48, 0x63, 0xc0]),       // movsxd rax, eax
                }
            }
            self.cvtsi2sd_xmm0_rax();
        }
        self.raw(&[0x66, 0x48, 0x0f, 0x7e, 0xc0]); // movq rax, xmm0
        self.bind(got);
    }

    /// Stores the low `acc.size` bytes of `rdi` just below `r8`, in the byte
    /// order `esi` says (1: little-endian). Clobbers `rax`.
    fn dv_store(&mut self, acc: DvAccess) {
        let s = -i32::from(acc.size);
        let le = self.new_label();
        self.g_mov(r::RAX, r::RDI);
        if acc.size > 1 {
            self.g_test(r::RSI);
            self.g_jcc(cc::NE, le);
            match acc.size {
                2 => self.raw(&[0x66, 0xc1, 0xc0, 0x08]), // rol ax, 8
                4 => self.raw(&[0x0f, 0xc8]),             // bswap eax
                _ => self.raw(&[0x48, 0x0f, 0xc8]),       // bswap rax
            }
        }
        self.bind(le);
        match acc.size {
            1 => self.g_mem(false, &[0x88], r::RAX, r::R8, s), // mov byte, al
            2 => {
                self.raw(&[0x66]);
                self.g_mem(false, &[0x89], r::RAX, r::R8, s); // mov word, ax
            }
            4 => self.g_mem(false, &[0x89], r::RAX, r::R8, s), // mov dword, eax
            _ => self.g_mem(true, &[0x89], r::RAX, r::R8, s),  // mov qword, rax
        }
    }
}

/// x86 condition codes beyond [`cc`]'s.
const CC_S: u8 = 0x8;
const CC_P: u8 = 0xa;
const CC_NP: u8 = 0xb;
