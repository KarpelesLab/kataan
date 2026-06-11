/*---
description: WASM linear memory load/store (widths, signedness, offsets), memory.size/grow, OOB trap
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
// Module with (memory 1), one func exported "f".
function memmod(params, results, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var typesec = [1].concat(uleb(1 + type.length), [1], type);
  var funcsec = [3, 2, 1, 0];
  var memsec = [5, 3, 1, 0, 1];
  var exportsec = [7, 5, 1, 1, 0x66, 0, 0];
  var code = [0].concat(body, [0x0b]);
  var codesec = [10].concat(uleb(1 + uleb(code.length).length + code.length), [1], uleb(code.length), code);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(typesec, funcsec, memsec, exportsec, codesec));
}
function call(m) {
  var inst = new WebAssembly.Instance(new WebAssembly.Module(m));
  return inst.exports.f.apply(null, Array.prototype.slice.call(arguments, 1));
}
// i32 store then load at the same address.
assert.sameValue(call(memmod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x36, 2, 0, 0x20, 0, 0x28, 2, 0]), 0, 42), 42, "i32 store/load");

// store8 / load8_u (the high bits are dropped on store, zero-extended on load).
assert.sameValue(call(memmod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x3a, 0, 0, 0x20, 0, 0x2d, 0, 0]), 0, 200), 200, "store8/load8_u");
// load8_s sign-extends (200 -> -56).
assert.sameValue(call(memmod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x3a, 0, 0, 0x20, 0, 0x2c, 0, 0]), 0, 200), -56, "load8_s sign-extends");
// store16 / load16_u.
assert.sameValue(call(memmod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x3b, 1, 0, 0x20, 0, 0x2f, 1, 0]), 0, 1000), 1000, "store16/load16_u");

// A static load/store offset is honored.
assert.sameValue(call(memmod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x36, 2, 4, 0x20, 0, 0x28, 2, 4]), 0, 77), 77, "store/load with offset 4");

// i64 store/load round-trips a BigInt.
assert.sameValue(call(memmod([0x7f, 0x7e], [0x7e], [0x20, 0, 0x20, 1, 0x37, 3, 0, 0x20, 0, 0x29, 3, 0]), 0, 123456789012n), 123456789012n, "i64 store/load");

// memory.size reports the page count; memory.grow returns the prior size.
assert.sameValue(call(memmod([], [0x7f], [0x3f, 0])), 1, "memory.size = 1 page");
assert.sameValue(call(memmod([0x7f], [0x7f], [0x20, 0, 0x40, 0]), 2), 1, "memory.grow returns old size");

// An out-of-bounds store traps.
var trapped = false;
try { call(memmod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x36, 2, 0, 0x41, 0, 0x0b]), 100000, 1); } catch (e) { trapped = true; }
assert.sameValue(trapped, true, "out-of-bounds store traps");
