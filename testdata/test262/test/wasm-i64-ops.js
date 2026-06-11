/*---
description: WASM i64 arithmetic/bitwise/comparison/unary ops and constants (full BigInt precision)
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sleb64(n) {
  var b = [], more = true;
  while (more) {
    var byte = Number(n & 0x7fn); n >>= 7n;
    if ((n === 0n && (byte & 0x40) === 0) || (n === -1n && (byte & 0x40) !== 0)) more = false; else byte |= 0x80;
    b.push(byte);
  }
  return b;
}
function mod(params, results, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var typesec = [1].concat(uleb(1 + type.length), [1], type);
  var funcsec = [3, 2, 1, 0];
  var exportsec = [7, 5, 1, 1, 0x66, 0, 0];
  var code = [0].concat(body, [0x0b]);
  var codesec = [10].concat(uleb(1 + uleb(code.length).length + code.length), [1], uleb(code.length), code);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(typesec, funcsec, exportsec, codesec));
}
function call(m) {
  var inst = new WebAssembly.Instance(new WebAssembly.Module(m));
  return inst.exports.f.apply(null, Array.prototype.slice.call(arguments, 1));
}
// i64 (0x7e) binary op -> i64 (BigInt).
function i64op(op) { return mod([0x7e, 0x7e], [0x7e], [0x20, 0, 0x20, 1, op]); }
assert.sameValue(call(i64op(0x7c), 10n, 20n), 30n, "i64.add");
assert.sameValue(call(i64op(0x7d), 50n, 8n), 42n, "i64.sub");
assert.sameValue(call(i64op(0x7e), 1000000n, 1000000n), 1000000000000n, "i64.mul");
assert.sameValue(call(i64op(0x7f), 100n, 7n), 14n, "i64.div_s");
assert.sameValue(call(i64op(0x81), 100n, 7n), 2n, "i64.rem_s");
assert.sameValue(call(i64op(0x83), 0xffn, 0x0fn), 0x0fn, "i64.and");
assert.sameValue(call(i64op(0x84), 0xf0n, 0x0fn), 0xffn, "i64.or");
assert.sameValue(call(i64op(0x85), 0xffn, 0x0fn), 0xf0n, "i64.xor");
assert.sameValue(call(i64op(0x86), 1n, 40n), 1099511627776n, "i64.shl past 32 bits");
assert.sameValue(call(i64op(0x87), 1024n, 2n), 256n, "i64.shr_s");

// Full 64-bit precision (beyond 2^53).
assert.sameValue(call(i64op(0x7c), 9007199254740993n, 1000n), 9007199254741993n, "above 2^53");
assert.sameValue(call(i64op(0x7c), 9223372036854775000n, 7n), 9223372036854775007n, "near i64 max");
assert.sameValue(call(i64op(0x7d), 5n, 10n), -5n, "negative result");

// Comparisons -> i32.
function i64cmp(op) { return mod([0x7e, 0x7e], [0x7f], [0x20, 0, 0x20, 1, op]); }
assert.sameValue(call(i64cmp(0x51), 5n, 5n), 1, "i64.eq");
assert.sameValue(call(i64cmp(0x53), 3n, 5n), 1, "i64.lt_s");
assert.sameValue(call(i64cmp(0x55), 5n, 3n), 1, "i64.gt_s");

// i64.eqz -> i32; clz/popcnt -> i64.
assert.sameValue(call(mod([0x7e], [0x7f], [0x20, 0, 0x50]), 0n), 1, "i64.eqz 0");
assert.sameValue(call(mod([0x7e], [0x7f], [0x20, 0, 0x50]), 5n), 0, "i64.eqz nonzero");
assert.sameValue(call(mod([0x7e], [0x7e], [0x20, 0, 0x79]), 1n), 63n, "i64.clz");
assert.sameValue(call(mod([0x7e], [0x7e], [0x20, 0, 0x7b]), 0xffn), 8n, "i64.popcnt");

// A large i64 constant (signed LEB128).
assert.sameValue(call(mod([], [0x7e], [0x42].concat(sleb64(1000000000000n)))), 1000000000000n, "i64.const");
assert.sameValue(call(mod([], [0x7e], [0x42].concat(sleb64(-42n)))), -42n, "negative i64.const");
