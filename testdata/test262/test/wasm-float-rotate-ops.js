/*---
description: WASM f64 min/max/copysign/sqrt/ceil/floor/trunc/nearest and i32 rotl/rotr
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }
function mod(params, results, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var code = [0].concat(body, [0x0b]);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1].concat(type)), sec(3, [1, 0]), sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(code.length), code))));
}
function call(m) { var inst = new WebAssembly.Instance(new WebAssembly.Module(m)); return inst.exports.f.apply(null, Array.prototype.slice.call(arguments, 1)); }
function f64bin(op) { return mod([0x7c, 0x7c], [0x7c], [0x20, 0, 0x20, 1, op]); }
function f64un(op) { return mod([0x7c], [0x7c], [0x20, 0, op]); }
function i32bin(op) { return mod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, op]); }

// f64 binary.
assert.sameValue(call(f64bin(0xa4), 3.5, 2.5), 2.5, "f64.min");
assert.sameValue(call(f64bin(0xa5), 3.5, 2.5), 3.5, "f64.max");
assert.sameValue(call(f64bin(0xa6), 3.5, -1), -3.5, "f64.copysign");
assert.sameValue(Number.isNaN(call(f64bin(0xa4), NaN, 1)), true, "f64.min propagates NaN");
assert.sameValue(Number.isNaN(call(f64bin(0xa5), 1, NaN)), true, "f64.max propagates NaN");

// f64 unary.
assert.sameValue(call(f64un(0x9f), 16), 4, "f64.sqrt");
assert.sameValue(call(f64un(0x99), -7), 7, "f64.abs");
assert.sameValue(call(f64un(0x9a), 5), -5, "f64.neg");
assert.sameValue(call(f64un(0x9b), 2.1), 3, "f64.ceil");
assert.sameValue(call(f64un(0x9c), 2.9), 2, "f64.floor");
assert.sameValue(call(f64un(0x9d), -2.7), -2, "f64.trunc");
assert.sameValue(call(f64un(0x9e), 2.5), 2, "f64.nearest rounds half to even (2.5 -> 2)");
assert.sameValue(call(f64un(0x9e), 3.5), 4, "f64.nearest (3.5 -> 4)");

// i32 rotate (with 32-bit wraparound).
assert.sameValue(call(i32bin(0x77), 1, 4), 16, "i32.rotl");
assert.sameValue(call(i32bin(0x78), 16, 4), 1, "i32.rotr");
assert.sameValue(call(i32bin(0x77), -2147483648, 1) | 0, 1, "i32.rotl wraps the top bit around");
assert.sameValue(call(i32bin(0x78), 1, 1) | 0, -2147483648, "i32.rotr wraps the low bit to the top");
