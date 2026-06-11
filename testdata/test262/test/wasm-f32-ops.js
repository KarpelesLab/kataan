/*---
description: WASM f32 arithmetic, min/max, sqrt/abs/neg, comparisons, rounding, and single-precision rounding
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
function call(m) { var i = new WebAssembly.Instance(new WebAssembly.Module(m)); return i.exports.f.apply(null, Array.prototype.slice.call(arguments, 1)); }
var F32 = 0x7d, I32 = 0x7f;
function bin(op) { return mod([F32, F32], [F32], [0x20, 0, 0x20, 1, op]); }
function un(op) { return mod([F32], [F32], [0x20, 0, op]); }
function cmp(op) { return mod([F32, F32], [I32], [0x20, 0, 0x20, 1, op]); }

// Arithmetic.
assert.sameValue(call(bin(0x92), 1.5, 2.5), 4, "f32.add");
assert.sameValue(call(bin(0x93), 5, 3), 2, "f32.sub");
assert.sameValue(call(bin(0x94), 3, 4), 12, "f32.mul");
assert.sameValue(call(bin(0x95), 10, 4), 2.5, "f32.div");
assert.sameValue(call(bin(0x96), 3, 2), 2, "f32.min");
assert.sameValue(call(bin(0x97), 3, 2), 3, "f32.max");

// Unary.
assert.sameValue(call(un(0x91), 9), 3, "f32.sqrt");
assert.sameValue(call(un(0x8b), -5), 5, "f32.abs");
assert.sameValue(call(un(0x8c), 3), -3, "f32.neg");
assert.sameValue(call(un(0x8d), 2.3), 3, "f32.ceil");
assert.sameValue(call(un(0x8e), 2.7), 2, "f32.floor");
assert.sameValue(call(un(0x8f), -2.3), -2, "f32.trunc");
assert.sameValue(call(un(0x90), 2.5), 2, "f32.nearest (round half to even)");

// Comparisons (i32 0/1 result).
assert.sameValue(call(cmp(0x5b), 1, 1), 1, "f32.eq");
assert.sameValue(call(cmp(0x5c), 1, 2), 1, "f32.ne");
assert.sameValue(call(cmp(0x5d), 1, 2), 1, "f32.lt");
assert.sameValue(call(cmp(0x5e), 2, 1), 1, "f32.gt");
assert.sameValue(call(cmp(0x5f), 1, 1), 1, "f32.le");
assert.sameValue(call(cmp(0x60), 1, 1), 1, "f32.ge");

// The arguments and result are genuinely single-precision: 0.1 + 0.2 rounds to the f32 value.
assert.sameValue(call(bin(0x92), 0.1, 0.2), 0.30000001192092896, "f32 single-precision rounding");
