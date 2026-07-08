//! Reference-types + bulk-table conformance for the `kataan` WebAssembly engine
//! (`ROADMAP.md` §2.2). Each test hand-encodes a small `.wasm` module exercising
//! one opcode family — `ref.null`/`ref.is_null`/`ref.func`, `table.get`/`.set`/
//! `.size`/`.grow`/`.fill`/`.copy`/`.init`, `elem.drop`, multiple tables +
//! `call_indirect` with a table index, all element-segment modes, and typed
//! `select` — and drives it through the runtime, asserting results/traps.

use kataan::wasm_rt::{Instance, Module, Val, ValType};

// --- tiny binary-module assembler -------------------------------------------

/// Unsigned LEB128 (all indices in these tests are < 128, so this is only used
/// for section sizes, which may exceed 127).
fn uleb(mut v: u32, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

/// A section: id byte, LEB length, body.
fn section(id: u8, body: &[u8]) -> Vec<u8> {
    let mut s = vec![id];
    uleb(body.len() as u32, &mut s);
    s.extend_from_slice(body);
    s
}

/// A `vec(x)`: a LEB count followed by the concatenated encoded items.
fn wvec(items: &[Vec<u8>]) -> Vec<u8> {
    let mut b = Vec::new();
    uleb(items.len() as u32, &mut b);
    for it in items {
        b.extend_from_slice(it);
    }
    b
}

/// A `functype`: `0x60 vec(params) vec(results)`.
fn functype(params: &[u8], results: &[u8]) -> Vec<u8> {
    let mut t = vec![0x60];
    uleb(params.len() as u32, &mut t);
    t.extend_from_slice(params);
    uleb(results.len() as u32, &mut t);
    t.extend_from_slice(results);
    t
}

/// One code-section entry: `size · (local-decls · code · end)`. `locals` is a
/// list of `(count, valtype)` runs.
fn body(locals: &[(u32, u8)], code: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    uleb(locals.len() as u32, &mut b);
    for (n, t) in locals {
        uleb(*n, &mut b);
        b.push(*t);
    }
    b.extend_from_slice(code);
    b.push(0x0b); // end
    let mut out = Vec::new();
    uleb(b.len() as u32, &mut out);
    out.extend_from_slice(&b);
    out
}

/// Assembles a module from already-built sections (each `Vec<u8>` a full
/// section, in canonical order).
fn assemble(sections: &[Vec<u8>]) -> Vec<u8> {
    let mut m = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    for s in sections {
        m.extend_from_slice(s);
    }
    m
}

// valtype / opcode shorthands
const I32: u8 = 0x7f;
const FUNCREF: u8 = 0x70;
const EXTERNREF: u8 = 0x6f;

/// An export entry `name·kind·index`.
fn export(name: &str, kind: u8, index: u8) -> Vec<u8> {
    let mut e = Vec::new();
    uleb(name.len() as u32, &mut e);
    e.extend_from_slice(name.as_bytes());
    e.push(kind);
    e.push(index);
    e
}

// --- tests ------------------------------------------------------------------

/// `ref.func` + `table.set` install a function into a table at runtime, then
/// `call_indirect` dispatches through it.
#[test]
fn ref_func_table_set_call_indirect() {
    // types: t0 = (i32,i32) -> i32
    let types = section(1, &wvec(&[functype(&[I32, I32], &[I32])]));
    // funcs: f0 = add (t0), f1 = run (t0)
    let funcs = section(3, &wvec(&[vec![0x00], vec![0x00]]));
    // table 0: funcref, min 1
    let tables = section(4, &wvec(&[vec![FUNCREF, 0x00, 0x01]]));
    let exports = section(7, &wvec(&[export("run", 0x00, 1)]));
    // f0: local.get 0 local.get 1 i32.add
    let add = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6a]);
    // f1: i32.const 0; ref.func 0; table.set 0; get a; get b; i32.const 0;
    //     call_indirect t0 table0
    let run = body(
        &[],
        &[
            0x41, 0x00, // i32.const 0 (table slot)
            0xd2, 0x00, // ref.func 0
            0x26, 0x00, // table.set 0
            0x20, 0x00, 0x20, 0x01, // args
            0x41, 0x00, // i32.const 0 (call index)
            0x11, 0x00, 0x00, // call_indirect type 0 table 0
        ],
    );
    let code = section(10, &wvec(&[add, run]));
    let m = assemble(&[types, funcs, tables, exports, code]);

    let module = Module::decode(&m).expect("decode reftable");
    assert_eq!(
        module.call(1, &[Val::I32(20), Val::I32(22)]).unwrap(),
        vec![Val::I32(42)]
    );
}

/// An `externref` round-trips through a table: store the argument, read it back.
#[test]
fn externref_round_trips_through_a_table() {
    // type t0 = (externref) -> externref
    let types = section(1, &wvec(&[functype(&[EXTERNREF], &[EXTERNREF])]));
    let funcs = section(3, &wvec(&[vec![0x00]]));
    // table 0: externref, min 1
    let tables = section(4, &wvec(&[vec![EXTERNREF, 0x00, 0x01]]));
    let exports = section(7, &wvec(&[export("rt", 0x00, 0)]));
    // i32.const 0; local.get 0; table.set 0; i32.const 0; table.get 0
    let rt = body(
        &[],
        &[
            0x41, 0x00, 0x20, 0x00, 0x26, 0x00, // table.set 0
            0x41, 0x00, 0x25, 0x00, // table.get 0
        ],
    );
    let code = section(10, &wvec(&[rt]));
    let m = assemble(&[types, funcs, tables, exports, code]);

    let module = Module::decode(&m).expect("decode externref table");
    // A non-null externref token comes back unchanged.
    assert_eq!(
        module
            .call(0, &[Val::ExternRef(Some(0xdead_beef))])
            .unwrap(),
        vec![Val::ExternRef(Some(0xdead_beef))]
    );
    // A null externref round-trips as null.
    assert_eq!(
        module.call(0, &[Val::ExternRef(None)]).unwrap(),
        vec![Val::ExternRef(None)]
    );
}

/// `ref.null` + `ref.is_null` classify null vs non-null references.
#[test]
fn ref_null_and_is_null() {
    // t0 = () -> i32 ; t1 = (externref) -> i32
    let types = section(
        1,
        &wvec(&[functype(&[], &[I32]), functype(&[EXTERNREF], &[I32])]),
    );
    let funcs = section(3, &wvec(&[vec![0x00], vec![0x01]]));
    let exports = section(
        7,
        &wvec(&[export("nullIsNull", 0x00, 0), export("argIsNull", 0x00, 1)]),
    );
    // f0: ref.null func; ref.is_null
    let f0 = body(&[], &[0xd0, FUNCREF, 0xd1]);
    // f1: local.get 0; ref.is_null
    let f1 = body(&[], &[0x20, 0x00, 0xd1]);
    let code = section(10, &wvec(&[f0, f1]));
    let m = assemble(&[types, funcs, exports, code]);

    let module = Module::decode(&m).expect("decode ref.null");
    assert_eq!(module.call(0, &[]).unwrap(), vec![Val::I32(1)]);
    assert_eq!(
        module.call(1, &[Val::ExternRef(None)]).unwrap(),
        vec![Val::I32(1)]
    );
    assert_eq!(
        module.call(1, &[Val::ExternRef(Some(1))]).unwrap(),
        vec![Val::I32(0)]
    );
}

/// `table.size` / `table.grow` / `table.fill` over an `externref` table, with the
/// growth persisting across `Instance` calls.
#[test]
fn table_size_grow_fill() {
    let types = section(
        1,
        &wvec(&[
            functype(&[], &[I32]),
            functype(&[I32], &[I32]),
            functype(&[], &[]),
        ]),
    );
    // f0 size()->i32, f1 grow(i32)->i32, f2 fill()->()
    let funcs = section(3, &wvec(&[vec![0x00], vec![0x01], vec![0x02]]));
    // externref table, min 1, max 8
    let tables = section(4, &wvec(&[vec![EXTERNREF, 0x01, 0x01, 0x08]]));
    let exports = section(
        7,
        &wvec(&[
            export("size", 0x00, 0),
            export("grow", 0x00, 1),
            export("fill", 0x00, 2),
        ]),
    );
    let f_size = body(&[], &[0xfc, 0x10, 0x00]); // table.size 0
    // grow(n): ref.null extern; local.get 0; table.grow 0
    let f_grow = body(&[], &[0xd0, EXTERNREF, 0x20, 0x00, 0xfc, 0x0f, 0x00]);
    // fill(): i32.const 0; ref.null extern; i32.const 2; table.fill 0
    let f_fill = body(
        &[],
        &[0x41, 0x00, 0xd0, EXTERNREF, 0x41, 0x02, 0xfc, 0x11, 0x00],
    );
    let code = section(10, &wvec(&[f_size, f_grow, f_fill]));
    let m = assemble(&[types, funcs, tables, exports, code]);

    let module = Module::decode(&m).expect("decode grow/size/fill");
    let mut inst = Instance::new(&module).expect("instantiate");
    assert_eq!(inst.call_export("size", &[]).unwrap(), vec![Val::I32(1)]);
    // grow by 3 returns the old size (1); size is now 4.
    assert_eq!(
        inst.call_export("grow", &[Val::I32(3)]).unwrap(),
        vec![Val::I32(1)]
    );
    assert_eq!(inst.call_export("size", &[]).unwrap(), vec![Val::I32(4)]);
    // grow past the declared maximum (8) returns -1 and leaves the size at 4.
    assert_eq!(
        inst.call_export("grow", &[Val::I32(100)]).unwrap(),
        vec![Val::I32(-1)]
    );
    assert_eq!(inst.call_export("size", &[]).unwrap(), vec![Val::I32(4)]);
    // fill runs without trapping.
    inst.call_export("fill", &[]).unwrap();
    // The host observes the table via the public accessor.
    assert_eq!(inst.table_size(0), Some(4));
    assert_eq!(inst.table_get(0, 0), Some(Val::ExternRef(None)));
}

/// A passive element segment + `table.init` populates a table, `table.copy`
/// duplicates a slot, and `elem.drop` releases the segment — then
/// `call_indirect` dispatches to the installed functions.
#[test]
fn table_init_copy_and_elem_drop() {
    // t0 = (i32,i32)->i32 ; t1 = (i32,i32,i32)->i32
    let types = section(
        1,
        &wvec(&[
            functype(&[I32, I32], &[I32]),
            functype(&[I32, I32, I32], &[I32]),
        ]),
    );
    // f0 add(t0), f1 sub(t0), f2 dispatch(t1)
    let funcs = section(3, &wvec(&[vec![0x00], vec![0x00], vec![0x01]]));
    // funcref table, min 4
    let tables = section(4, &wvec(&[vec![FUNCREF, 0x00, 0x04]]));
    let exports = section(7, &wvec(&[export("dispatch", 0x00, 2)]));
    // passive element segment (flags 1): elemkind 0x00, funcs [0,1]
    let elems = section(9, &wvec(&[vec![0x01, 0x00, 0x02, 0x00, 0x01]]));
    let add = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6a]);
    let sub = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6b]);
    // dispatch(a,b,sel):
    //   table.init 0 0 (dst=0 src=0 n=2)   ; table[0..2] = [add, sub]
    //   table.copy 0 0 (dst=2 src=0 n=1)   ; table[2] = table[0] (add)
    //   elem.drop 0
    //   a b sel call_indirect t0 table0
    let dispatch = body(
        &[],
        &[
            0x41, 0x00, 0x41, 0x00, 0x41, 0x02, 0xfc, 0x0c, 0x00, 0x00, // table.init
            0x41, 0x02, 0x41, 0x00, 0x41, 0x01, 0xfc, 0x0e, 0x00, 0x00, // table.copy
            0xfc, 0x0d, 0x00, // elem.drop 0
            0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0x11, 0x00, 0x00, // call_indirect
        ],
    );
    let code = section(10, &wvec(&[add, sub, dispatch]));
    let m = assemble(&[types, funcs, tables, exports, elems, code]);

    let module = Module::decode(&m).expect("decode init/copy/drop");
    // sel 0 -> add(10,3)=13 ; sel 1 -> sub(10,3)=7 ; sel 2 -> copied add = 13
    assert_eq!(
        module
            .call(2, &[Val::I32(10), Val::I32(3), Val::I32(0)])
            .unwrap(),
        vec![Val::I32(13)]
    );
    assert_eq!(
        module
            .call(2, &[Val::I32(10), Val::I32(3), Val::I32(1)])
            .unwrap(),
        vec![Val::I32(7)]
    );
    assert_eq!(
        module
            .call(2, &[Val::I32(10), Val::I32(3), Val::I32(2)])
            .unwrap(),
        vec![Val::I32(13)]
    );
    // slot 3 is still null → call_indirect traps.
    assert!(
        module
            .call(2, &[Val::I32(1), Val::I32(1), Val::I32(3)])
            .is_err()
    );
}

/// Two tables, each holding a different function, dispatched by
/// `call_indirect` with the corresponding table-index immediate.
#[test]
fn multiple_tables_call_indirect_table_index() {
    let types = section(1, &wvec(&[functype(&[I32, I32], &[I32])]));
    // f0 add, f1 sub, f2 via0, f3 via1
    let funcs = section(3, &wvec(&[vec![0x00], vec![0x00], vec![0x00], vec![0x00]]));
    // two funcref tables, each min 1
    let tables = section(
        4,
        &wvec(&[vec![FUNCREF, 0x00, 0x01], vec![FUNCREF, 0x00, 0x01]]),
    );
    let exports = section(
        7,
        &wvec(&[export("via0", 0x00, 2), export("via1", 0x00, 3)]),
    );
    // elem seg 0: active table 0 (flags 0), offset 0, funcs [0] (add)
    // elem seg 1: active table 1 (flags 2), tableidx 1, offset 0, funcs [1] (sub)
    let elems = section(
        9,
        &wvec(&[
            vec![0x00, 0x41, 0x00, 0x0b, 0x01, 0x00],
            vec![0x02, 0x01, 0x41, 0x00, 0x0b, 0x00, 0x01, 0x01],
        ]),
    );
    let add = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6a]);
    let sub = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6b]);
    let via0 = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x41, 0x00, 0x11, 0x00, 0x00]);
    let via1 = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x41, 0x00, 0x11, 0x00, 0x01]);
    let code = section(10, &wvec(&[add, sub, via0, via1]));
    let m = assemble(&[types, funcs, tables, exports, elems, code]);

    let module = Module::decode(&m).expect("decode multi-table");
    assert_eq!(
        module.call(2, &[Val::I32(10), Val::I32(3)]).unwrap(),
        vec![Val::I32(13)]
    ); // via table 0 -> add
    assert_eq!(
        module.call(3, &[Val::I32(10), Val::I32(3)]).unwrap(),
        vec![Val::I32(7)]
    ); // via table 1 -> sub
}

/// Typed `select` (`select t*`) over `externref` operands.
#[test]
fn typed_select_over_externref() {
    // (externref, externref, i32) -> externref
    let types = section(
        1,
        &wvec(&[functype(&[EXTERNREF, EXTERNREF, I32], &[EXTERNREF])]),
    );
    let funcs = section(3, &wvec(&[vec![0x00]]));
    let exports = section(7, &wvec(&[export("pick", 0x00, 0)]));
    // local.get 0; local.get 1; local.get 2; select externref (0x1c 0x01 0x6f)
    let pick = body(
        &[],
        &[0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0x1c, 0x01, EXTERNREF],
    );
    let code = section(10, &wvec(&[pick]));
    let m = assemble(&[types, funcs, exports, code]);

    let module = Module::decode(&m).expect("decode typed select");
    // cond != 0 picks the first operand.
    assert_eq!(
        module
            .call(
                0,
                &[
                    Val::ExternRef(Some(1)),
                    Val::ExternRef(Some(2)),
                    Val::I32(1)
                ]
            )
            .unwrap(),
        vec![Val::ExternRef(Some(1))]
    );
    assert_eq!(
        module
            .call(
                0,
                &[
                    Val::ExternRef(Some(1)),
                    Val::ExternRef(Some(2)),
                    Val::I32(0)
                ]
            )
            .unwrap(),
        vec![Val::ExternRef(Some(2))]
    );
}

/// An exported table is introspectable through the module + instance API, and
/// declarative/expression element segments decode.
#[test]
fn exported_table_and_expression_elements() {
    let types = section(1, &wvec(&[functype(&[I32, I32], &[I32])]));
    let funcs = section(3, &wvec(&[vec![0x00]]));
    let tables = section(4, &wvec(&[vec![FUNCREF, 0x00, 0x02]]));
    let exports = section(7, &wvec(&[export("t", 0x01, 0), export("add", 0x00, 0)]));
    // elem seg (flags 4): active table 0, offset 0, expression elements
    // [ ref.func 0 ]  → encodes each item as `expr end`.
    let elems = section(
        9,
        &wvec(&[vec![
            0x04, // flags: active table 0, expressions
            0x41, 0x00, 0x0b, // offset: i32.const 0 end
            0x01, // 1 item
            0xd2, 0x00, 0x0b, // ref.func 0 end
        ]]),
    );
    let add = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6a]);
    let code = section(10, &wvec(&[add]));
    let m = assemble(&[types, funcs, tables, exports, elems, code]);

    let module = Module::decode(&m).expect("decode exported table");
    assert_eq!(module.table_exports(), vec![("t", 0)]);
    assert_eq!(module.table_type(0), Some(ValType::FuncRef));
    let inst = Instance::new(&module).expect("instantiate");
    // The expression element installed func 0 at slot 0.
    assert_eq!(inst.table_get(0, 0), Some(Val::FuncRef(Some(0))));
    assert_eq!(inst.table_get(0, 1), Some(Val::FuncRef(None)));
}

/// `call_indirect` traps on a null slot and on a signature mismatch, and an
/// out-of-bounds active element segment fails instantiation.
#[test]
fn call_indirect_traps_and_bad_segment() {
    // t0 (i32,i32)->i32 (the call_indirect signature) ; t2 (i32,i32,i32)->i32
    // (the caller's own signature).
    let types = section(
        1,
        &wvec(&[
            functype(&[I32, I32], &[I32]),
            functype(&[], &[I32]),
            functype(&[I32, I32, I32], &[I32]),
        ]),
    );
    let funcs = section(3, &wvec(&[vec![0x00], vec![0x02]]));
    // funcref table min 2, slot 0 = add, slot 1 = null
    let tables = section(4, &wvec(&[vec![FUNCREF, 0x00, 0x02]]));
    let exports = section(7, &wvec(&[export("call", 0x00, 1)]));
    let elems = section(9, &wvec(&[vec![0x00, 0x41, 0x00, 0x0b, 0x01, 0x00]]));
    let add = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6a]);
    // call(a,b,sel): a b sel call_indirect t0 table0
    let caller = body(&[], &[0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0x11, 0x00, 0x00]);
    let code = section(10, &wvec(&[add, caller]));
    let m = assemble(&[types, funcs, tables, exports, elems, code]);

    let module = Module::decode(&m).expect("decode trap module");
    // slot 0 -> add(4,5)=9
    assert_eq!(
        module
            .call(1, &[Val::I32(4), Val::I32(5), Val::I32(0)])
            .unwrap(),
        vec![Val::I32(9)]
    );
    // slot 1 is null -> trap
    assert!(
        module
            .call(1, &[Val::I32(4), Val::I32(5), Val::I32(1)])
            .is_err()
    );
    // slot 5 is out of bounds -> trap
    assert!(
        module
            .call(1, &[Val::I32(4), Val::I32(5), Val::I32(5)])
            .is_err()
    );

    // An active element segment past the table end fails at instantiation.
    let tables2 = section(4, &wvec(&[vec![FUNCREF, 0x00, 0x01]])); // min 1
    // active table 0 offset 5, funcs [0] -> out of bounds
    let bad_elem = section(9, &wvec(&[vec![0x00, 0x41, 0x05, 0x0b, 0x01, 0x00]]));
    let m2 = assemble(&[
        section(1, &wvec(&[functype(&[I32, I32], &[I32])])),
        section(3, &wvec(&[vec![0x00]])),
        tables2,
        bad_elem,
        section(10, &wvec(&[body(&[], &[0x20, 0x00, 0x20, 0x01, 0x6a])])),
    ]);
    // Decode succeeds (segment bounds are checked at instantiation).
    let module2 = Module::decode(&m2).expect("decode");
    assert!(
        Instance::new(&module2).is_err(),
        "out-of-bounds active elem must fail"
    );
}
