/*---
description: WASM call_indirect dispatches through a funcref table; out-of-bounds and type-mismatch trap
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, payload) { return [id].concat(uleb(payload.length), payload); }
// func0 ()->i32 = 10, func1 ()->i32 = 20, func2 (i32)->i32 (wrong type for the table call),
// func3 (i32)->i32 = call_indirect type0 table[idx]. Table holds [func0, func1, func2]. Export func3 "f".
function mkmod() {
  var type0 = [0x60, 0, 1, 0x7f];      // ()->i32
  var type1 = [0x60, 1, 0x7f, 1, 0x7f]; // (i32)->i32
  var typesec = sec(1, [2].concat(type0, type1));
  var funcsec = sec(3, [4, 0, 0, 1, 1]); // func0:t0 func1:t0 func2:t1 func3:t1
  var tablesec = sec(4, [1, 0x70, 0x00, 3]);
  var exportsec = sec(7, [1, 1, 0x66, 0, 3]); // "f" -> func 3
  var elemsec = sec(9, [1, 0, 0x41, 0, 0x0b, 3, 0, 1, 2]); // table[0..3] = func0,func1,func2
  var body0 = [0, 0x41, 10, 0x0b];
  var body1 = [0, 0x41, 20, 0x0b];
  var body2 = [0, 0x20, 0, 0x0b];                 // (i32)->i32 identity (unused result type)
  var body3 = [0, 0x20, 0, 0x11, 0, 0x00, 0x0b];  // local.get 0; call_indirect type0 table0
  var codesec = sec(10, [4].concat(
    uleb(body0.length), body0, uleb(body1.length), body1, uleb(body2.length), body2, uleb(body3.length), body3));
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(typesec, funcsec, tablesec, exportsec, elemsec, codesec));
}
var f = new WebAssembly.Instance(new WebAssembly.Module(mkmod())).exports.f;

// Indirect dispatch through the table.
assert.sameValue(f(0), 10, "call_indirect table[0] -> func0");
assert.sameValue(f(1), 20, "call_indirect table[1] -> func1");

// table[2] is a (i32)->i32 function but call_indirect expects ()->i32: a runtime type mismatch traps.
var typeTrap = false;
try { f(2); } catch (e) { typeTrap = true; }
assert.sameValue(typeTrap, true, "type-signature mismatch traps");

// An out-of-bounds table index traps.
var oobTrap = false;
try { f(5); } catch (e) { oobTrap = true; }
assert.sameValue(oobTrap, true, "out-of-bounds index traps");
