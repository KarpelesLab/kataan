/*---
description: WASM i64 comparison, bitwise, shift and rotate opcodes (BigInt operands)
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
var I64 = 0x7e, I32 = 0x7f;
function cmp(op) { return mod([I64, I64], [I32], [0x20, 0, 0x20, 1, op]); }
function bit(op) { return mod([I64, I64], [I64], [0x20, 0, 0x20, 1, op]); }

// Signed comparisons (i32 result).
assert.sameValue(call(cmp(0x51), 5n, 5n), 1, "i64.eq");
assert.sameValue(call(cmp(0x52), 5n, 6n), 1, "i64.ne");
assert.sameValue(call(cmp(0x53), 3n, 5n), 1, "i64.lt_s");
assert.sameValue(call(cmp(0x55), 5n, 3n), 1, "i64.gt_s");
assert.sameValue(call(cmp(0x57), 5n, 5n), 1, "i64.le_s");
assert.sameValue(call(cmp(0x59), 5n, 5n), 1, "i64.ge_s");
assert.sameValue(call(cmp(0x53), -1n, 1n), 1, "i64.lt_s respects sign");

// Unsigned comparisons.
assert.sameValue(call(cmp(0x54), 3n, 5n), 1, "i64.lt_u");
assert.sameValue(call(cmp(0x56), 5n, 3n), 1, "i64.gt_u");
assert.sameValue(call(cmp(0x58), 5n, 5n), 1, "i64.le_u");
assert.sameValue(call(cmp(0x5a), 5n, 5n), 1, "i64.ge_u");
// -1 as unsigned is the max value, so it is NOT < 1 unsigned.
assert.sameValue(call(cmp(0x54), -1n, 1n), 0, "i64.lt_u treats -1 as max");

// Bitwise and shifts (i64 result).
assert.sameValue(call(bit(0x83), 0xFFn, 0x0Fn), 0x0Fn, "i64.and");
assert.sameValue(call(bit(0x84), 0xF0n, 0x0Fn), 0xFFn, "i64.or");
assert.sameValue(call(bit(0x85), 0xFFn, 0x0Fn), 0xF0n, "i64.xor");
assert.sameValue(call(bit(0x86), 1n, 4n), 16n, "i64.shl");
assert.sameValue(call(bit(0x87), 256n, 2n), 64n, "i64.shr_s");
assert.sameValue(call(bit(0x88), 256n, 2n), 64n, "i64.shr_u");
assert.sameValue(call(bit(0x87), -16n, 2n), -4n, "i64.shr_s is arithmetic (sign-preserving)");
assert.sameValue(call(bit(0x89), 1n, 4n), 16n, "i64.rotl");
assert.sameValue(call(bit(0x8a), 16n, 4n), 1n, "i64.rotr");
