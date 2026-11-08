//! Direct calls of small built-ins from the generic tier (`ROADMAP.md` §2.1).
//!
//! A static built-in call — `Array.isArray(x)`, `Number.isInteger(x)`,
//! `Object.hasOwn(o, k)`, `Object.getPrototypeOf(o)`, `Object.is(a, b)`,
//! `String.fromCharCode(c)` — otherwise reaches the built-in through the
//! site's method helper, the interpreter's call op and the per-site memo's
//! by-kind dispatch (`nbvm::native_call::try_call`): several hundred
//! instructions for an answer that is often one compare.
//!
//! Here the call op of such a site (its key and argument count name the
//! built-in, [`kind_of`]) first asks the site's [`MethodSite`] memo of a
//! built-in function receiver's own data property (`NativeOwn`, the one
//! compiled property reads use): the receiver is the one it recorded (its
//! bits, which carry the handle's generation), the prototype epoch is the
//! memo's stamp (so the receiver still owns the key as a data property in
//! the same slot), one realm — and the method value in that slot is the very
//! value the memo's proof (`MethodSite::fill_direct`) found to be the
//! built-in. The call then is that built-in's, and it is computed here: in
//! line when it is a test of the argument's bits (the `Number` predicates;
//! `Array.isArray` of an array or an ordinary object), else through one
//! narrow helper of fixed signature (`(ctx, kind, a0, a1) -> bits`,
//! `nbvm::native_call::direct_run`), which declines — the throw sentinel,
//! having done nothing — whatever input could run code, throw or allocate
//! (a proxy, a key that is an object, a unit not yet interned, …). Every
//! miss, and every decline, falls through to the site's general call,
//! nothing done yet. So a replaced, deleted or shadowed built-in, another
//! receiver, or a second realm is the general path's, exactly as before.
//!
//! The helper runs no JavaScript, allocates nothing and moves no heap cell,
//! so it is no GC point and leaves every compiled cache valid: it is called
//! as a pure leaf (`X64Assembler::call_rax_pure`, which keeps the
//! receiver-cache keys and only spills and reloads the promoted SSE
//! registers around it). The miss path's fill helper, which watches cells,
//! is an ordinary helper call.

use super::call_seq::{cc, r};
use super::props::{DIRECT_TRIES, MethodSite};
use super::{JitLayout, Label, X64Assembler};
use crate::nanbox::NanBox;

/// `Array.isArray(x)`.
pub(crate) const K_IS_ARRAY: u8 = 1;
/// `Number.isInteger(x)`.
pub(crate) const K_IS_INTEGER: u8 = 2;
/// `Number.isSafeInteger(x)`.
pub(crate) const K_IS_SAFE_INTEGER: u8 = 3;
/// `Number.isFinite(x)`.
pub(crate) const K_IS_FINITE: u8 = 4;
/// `Number.isNaN(x)`.
pub(crate) const K_IS_NAN: u8 = 5;
/// `Object.hasOwn(o, k)`.
pub(crate) const K_HAS_OWN: u8 = 6;
/// `String.fromCharCode(c)` (one argument).
pub(crate) const K_FROM_CHAR_CODE: u8 = 7;
/// `Object.getPrototypeOf(o)`.
pub(crate) const K_GET_PROTO: u8 = 8;
/// `Object.is(a, b)`.
pub(crate) const K_OBJECT_IS: u8 = 9;

/// The direct kind of a method call of `key` with `argc` arguments, when
/// that names one of the built-ins served here (whether the site's method
/// really is that built-in is its run-time proof's to say).
pub(crate) fn kind_of(key: &str, argc: usize) -> Option<u8> {
    Some(match (key, argc) {
        ("isArray", 1..=2) => K_IS_ARRAY,
        ("isInteger", 1..=2) => K_IS_INTEGER,
        ("isSafeInteger", 1..=2) => K_IS_SAFE_INTEGER,
        ("isFinite", 1..=2) => K_IS_FINITE,
        ("isNaN", 1..=2) => K_IS_NAN,
        ("hasOwn", 2) => K_HAS_OWN,
        ("fromCharCode", 1) => K_FROM_CHAR_CODE,
        ("getPrototypeOf", 1..=2) => K_GET_PROTO,
        ("is", 2..=3) => K_OBJECT_IS,
        _ => return None,
    })
}

/// What [`emit`] needs of the call op and the compiled body.
pub(crate) struct DirectEmit<'a> {
    /// The built-in ([`kind_of`]).
    pub kind: u8,
    /// The destination register.
    pub dst: u16,
    /// The receiver register.
    pub recv: u16,
    /// The argument registers.
    pub args: &'a [u16],
    /// The call op's method site (its `NativeOwn` memo and proof).
    pub site: *const MethodSite,
    /// The context's `hosted` byte, realm pointer and the realm's
    /// `multi_realm` byte (`CallLayout`).
    pub hosted: i32,
    pub realm: i32,
    pub multi_realm: i32,
    /// `jit_helper_direct_fill` / `jit_helper_direct_call`.
    pub fill: u64,
    pub call: u64,
    /// The heap layout (the inline `Array.isArray`'s cell walk).
    pub layout: &'a JitLayout,
    /// Where the op's code ends (the result stored).
    pub done: Label,
}

/// A register's slot in the register file (`[rbx + 8 r]`).
fn disp(r: u16) -> i32 {
    i32::from(r) * 8
}

/// Emits the direct form of a call op: on a proved site and an input the
/// built-in answers without running code, the result stored to `dst` and a
/// jump to `done`; otherwise falls through past the emitted code with
/// nothing done (the receiver-cache keys maybe emptied by a helper call) to
/// the op's general call.
pub(crate) fn emit(a: &mut X64Assembler, e: &DirectEmit<'_>) {
    let general = a.new_label();
    let fill = a.new_label();
    let (direct_off, tries_off) = MethodSite::direct_offsets();
    // The site's proof: hosted, the recorded receiver at the memo's epoch in
    // one realm, its slot holding the proved built-in.
    a.g_cmp_byte_imm(r::R15, e.hosted, 0);
    a.g_jcc(cc::E, general);
    a.p_native_own_hit(
        (e.realm, e.multi_realm),
        (r::RBX, disp(e.recv)),
        e.site,
        fill,
    );
    a.g_cmp_mem(r::RAX, r::R11, direct_off);
    a.g_jcc(cc::NE, fill);
    emit_body(a, e, general);
    // A miss: the memo refilled and proved (up to `DIRECT_TRIES` times),
    // and this call made by the general path.
    let cold = a.cold_begin();
    a.bind(fill);
    a.g_movabs(r::R11, e.site as u64);
    a.p_cmp32_imm8(r::R11, tries_off, DIRECT_TRIES as i8);
    a.g_jcc(cc::AE, general);
    a.mov_rdi_r15();
    a.load_argreg(1, disp(e.recv));
    a.movabs_argreg(2, e.site as i64);
    a.movabs_argreg(3, i64::from(e.kind));
    a.movabs_rax(e.fill as i64);
    a.call_rax();
    a.jmp(general);
    a.cold_end(cold);
    a.bind(general);
}

/// What a `CallWithMethod` site's identity memo (`call_seq::IdentMemo`)
/// holds as its value when the built-in its helper last served is the
/// direct kind `kind`: above every native id + 1 (the `Math` functions'
/// form, `jit::emit_got_ident`), so neither is taken for the other.
pub(crate) const fn got_value(kind: u8) -> u64 {
    0x2_0000 + kind as u64
}

/// [`emit`] for a `CallWithMethod` (`EvaluateCall` read the callee before
/// the arguments, which the compiler could not guard): the callee value at
/// `callee` is proved the built-in by the call site `site`'s identity memo
/// — the very value its helper last served, which it recorded with
/// [`got_value`] of this kind (and the stamp `u64::MAX`, no epoch: a
/// built-in function cell is what it is for life, and its bits carry its
/// generation). The receiver plays no part: every built-in served here
/// ignores its `this`.
pub(crate) fn emit_got(
    a: &mut X64Assembler,
    e: &DirectEmit<'_>,
    callee: u16,
    site: *const super::call_seq::CallSite,
) {
    use super::call_seq::{IDENT_KEY, IDENT_STAMP, IDENT_VALUE};
    let general = a.new_label();
    // (Hosted and one realm, as the method helper's own fast form.)
    a.g_cmp_byte_imm(r::R15, e.hosted, 0);
    a.g_jcc(cc::E, general);
    a.g_load(r::R9, r::R15, e.realm);
    a.g_cmp_byte_imm(r::R9, e.multi_realm, 0);
    a.g_jcc(cc::NE, general);
    a.load_rax(disp(callee));
    a.g_movabs(r::R11, site as u64);
    a.g_cmp_mem(r::RAX, r::R11, IDENT_KEY);
    a.g_jcc(cc::NE, general);
    a.g_mem(true, &[0x83], 7, r::R11, IDENT_STAMP);
    a.raw(&[0xff]); // cmp qword [..], -1 (u64::MAX)
    a.g_jcc(cc::NE, general);
    a.g_mem(true, &[0x81], 7, r::R11, IDENT_VALUE);
    a.raw(&(got_value(e.kind) as i32).to_le_bytes()); // cmp qword [..], value
    a.g_jcc(cc::NE, general);
    emit_body(a, e, general);
    a.bind(general);
}

/// The built-in's computation once a guard proved the callee (see
/// [`emit`]): the result stored and a jump to `e.done`, or to `general`
/// with nothing done.
fn emit_body(a: &mut X64Assembler, e: &DirectEmit<'_>, general: Label) {
    let t = NanBox::boolean(true).to_bits() as i64;
    let f = NanBox::boolean(false).to_bits() as i64;
    let x = e.args.first().copied();
    match e.kind {
        K_IS_INTEGER | K_IS_SAFE_INTEGER | K_IS_FINITE | K_IS_NAN => {
            let yes = a.new_label();
            let no = a.new_label();
            if let Some(x) = x {
                emit_num_pred(a, e.kind, disp(x), yes, no);
            } else {
                a.jmp(no);
            }
            a.bind(yes);
            a.movabs_rax(t);
            a.store_rax(disp(e.dst));
            a.jmp(e.done);
            a.bind(no);
            a.movabs_rax(f);
            a.store_rax(disp(e.dst));
            a.jmp(e.done);
        }
        _ => {
            let helper = a.new_label();
            if let (K_OBJECT_IS, &[x, y, ..]) = (e.kind, e.args) {
                // SameValue: the same bits are the same value; two Numbers
                // otherwise only when both are NaN; a heap value (a String
                // or BigInt compares by content) is the helper's; any other
                // pair differs.
                const QNAN: i64 = 0x7ffc_0000_0000_0000;
                let yes = a.new_label();
                let no = a.new_label();
                let num_x = a.new_label();
                a.load_rax(disp(x));
                a.g_load(r::RCX, r::RBX, disp(y));
                a.g_cmp(r::RAX, r::RCX);
                a.je(yes);
                a.p_cmp_handle_floor(r::RAX);
                a.g_jcc(cc::AE, helper);
                a.p_cmp_handle_floor(r::RCX);
                a.g_jcc(cc::AE, helper);
                a.movabs_r11(QNAN);
                a.and_rax_r11();
                a.cmp_rax_r11();
                a.jne(num_x);
                a.jmp(no); // `x` is no Number (nor heap value): `y` differs
                a.bind(num_x);
                a.g_load(r::RAX, r::RBX, disp(y));
                a.and_rax_r11();
                a.cmp_rax_r11();
                a.je(no);
                a.movsd_xmm0_mem(disp(x));
                a.raw(&[0x66, 0x0f, 0x2e, 0xc0]); // ucomisd xmm0, xmm0
                a.g_jcc(0xb, no); // jnp: `x` is no NaN
                a.movsd_xmm0_mem(disp(y));
                a.raw(&[0x66, 0x0f, 0x2e, 0xc0]); // ucomisd xmm0, xmm0
                a.jp(yes);
                a.bind(no);
                a.movabs_rax(f);
                a.store_rax(disp(e.dst));
                a.jmp(e.done);
                a.bind(yes);
                a.movabs_rax(t);
                a.store_rax(disp(e.dst));
                a.jmp(e.done);
            }
            if let (K_IS_ARRAY, Some(x)) = (e.kind, x) {
                // An array is one, a primitive or an ordinary object is
                // not; any other cell (a proxy among them) is the
                // helper's.
                let yes = a.new_label();
                let no = a.new_label();
                let l = e.layout;
                a.p_arena(e.realm, &l.props);
                a.load_rax(disp(x));
                a.p_is_handle(r::RCX, no);
                a.p_slot_rooted(l, r::RCX, helper);
                a.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_array_disc);
                a.je(yes);
                a.cmp_byte_at_rcx_imm(l.off_cell_tag, l.cell_object_disc);
                a.jne(helper);
                a.bind(no);
                a.movabs_rax(f);
                a.store_rax(disp(e.dst));
                a.jmp(e.done);
                a.bind(yes);
                a.movabs_rax(t);
                a.store_rax(disp(e.dst));
                a.jmp(e.done);
            }
            a.bind(helper);
            let undef = NanBox::undefined().to_bits() as i64;
            a.mov_rdi_r15();
            a.movabs_argreg(1, i64::from(e.kind));
            for (i, reg) in [2usize, 3].into_iter().enumerate() {
                match e.args.get(i) {
                    Some(&m) => a.load_argreg(reg, disp(m)),
                    None => a.movabs_argreg(reg, undef),
                }
            }
            a.movabs_rax(e.call as i64);
            a.call_rax_pure();
            a.p_cmp_throw_rax();
            a.g_jcc(cc::E, general);
            a.store_rax(disp(e.dst));
            a.jmp(e.done);
        }
    }
}

/// The `Number` predicate `kind` of the value at `[rbx + x]`, to `yes` or
/// `no` (anything but a Number is `false`; nothing is converted).
fn emit_num_pred(a: &mut X64Assembler, kind: u8, x: i32, yes: Label, no: Label) {
    const QNAN: i64 = 0x7ffc_0000_0000_0000;
    a.load_rax(x);
    a.mov_rcx_rax();
    a.movabs_r11(QNAN);
    a.and_rcx_r11();
    a.cmp_rcx_r11();
    a.je(no); // not a Number
    if kind == K_IS_NAN {
        a.movsd_xmm0_mem(x);
        a.raw(&[0x66, 0x0f, 0x2e, 0xc0]); // ucomisd xmm0, xmm0
        a.jp(yes);
        a.jmp(no);
        return;
    }
    // Finite: the bits shifted past the sign are below the infinities'.
    a.raw(&[0x48, 0xd1, 0xe0]); // shl rax, 1
    a.movabs_r11((0x7ff0_0000_0000_0000_u64 << 1) as i64);
    a.g_cmp(r::RAX, r::R11);
    a.g_jcc(cc::AE, no);
    if kind == K_IS_FINITE {
        a.jmp(yes);
        return;
    }
    // `rax` = |x|'s bits.
    a.raw(&[0x48, 0xd1, 0xe8]); // shr rax, 1
    if kind == K_IS_SAFE_INTEGER {
        a.movabs_r11(9_007_199_254_740_991.0_f64.to_bits() as i64);
        a.g_cmp(r::RAX, r::R11);
        a.g_jcc(cc::A, no);
    }
    // At least 2^52: no fraction bits. Below, the truncation is exact.
    a.movabs_r11(4_503_599_627_370_496.0_f64.to_bits() as i64);
    a.g_cmp(r::RAX, r::R11);
    a.g_jcc(cc::AE, yes);
    a.movsd_xmm0_mem(x);
    a.cvttsd2si_rax_xmm0();
    a.cvtsi2sd_xmm0_rax();
    a.movsd_xmm1_mem(x);
    a.ucomisd_xmm0_xmm1();
    a.je(yes);
    a.jmp(no);
}
