/*---
description: WASM f32/f64 arithmetic, comparisons, unary ops, and numeric conversions
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
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
// f64 (0x7c) binary ops -> f64.
function f64op(op) { return mod([0x7c, 0x7c], [0x7c], [0x20, 0, 0x20, 1, op]); }
assert.sameValue(call(f64op(0xa0), 1.5, 2.5), 4, "f64.add");
assert.sameValue(call(f64op(0xa1), 5.5, 2.0), 3.5, "f64.sub");
assert.sameValue(call(f64op(0xa2), 3.0, 4.0), 12, "f64.mul");
assert.sameValue(call(f64op(0xa3), 10.0, 4.0), 2.5, "f64.div");
assert.sameValue(call(f64op(0xa4), 3.0, 5.0), 3, "f64.min");
assert.sameValue(call(f64op(0xa5), 3.0, 5.0), 5, "f64.max");
assert.sameValue(call(f64op(0xa6), 5.0, -1.0), -5, "f64.copysign");

// f64 comparisons -> i32.
function f64cmp(op) { return mod([0x7c, 0x7c], [0x7f], [0x20, 0, 0x20, 1, op]); }
assert.sameValue(call(f64cmp(0x61), 1.5, 1.5), 1, "f64.eq");
assert.sameValue(call(f64cmp(0x63), 1.0, 2.0), 1, "f64.lt");
assert.sameValue(call(f64cmp(0x64), 2.0, 1.0), 1, "f64.gt");

// f64 unary -> f64.
function f64un(op) { return mod([0x7c], [0x7c], [0x20, 0, op]); }
assert.sameValue(call(f64un(0x9f), 16.0), 4, "f64.sqrt");
assert.sameValue(call(f64un(0x99), -5.0), 5, "f64.abs");
assert.sameValue(call(f64un(0x9a), 5.0), -5, "f64.neg");
assert.sameValue(call(f64un(0x9b), 2.3), 3, "f64.ceil");
assert.sameValue(call(f64un(0x9c), 2.7), 2, "f64.floor");
assert.sameValue(call(f64un(0x9d), -2.7), -2, "f64.trunc");
assert.sameValue(call(f64un(0x9e), 2.5), 2, "f64.nearest (ties to even)");

// f32 (0x7d) ops.
function f32op(op) { return mod([0x7d, 0x7d], [0x7d], [0x20, 0, 0x20, 1, op]); }
assert.sameValue(call(f32op(0x92), 1.5, 2.5), 4, "f32.add");
assert.sameValue(call(f32op(0x94), 2.0, 3.0), 6, "f32.mul");

// Conversions.
assert.sameValue(call(mod([0x7f], [0x7c], [0x20, 0, 0xb7]), 42), 42, "f64.convert_i32_s");
assert.sameValue(call(mod([0x7c], [0x7f], [0x20, 0, 0xaa]), 42.9), 42, "i32.trunc_f64_s");
assert.sameValue(call(mod([0x7f], [0x7e], [0x20, 0, 0xac]), 5), 5n, "i64.extend_i32_s -> BigInt");
assert.sameValue(call(mod([0x7e], [0x7f], [0x20, 0, 0xa7]), 5n), 5, "i32.wrap_i64");

// reinterpret: the IEEE-754 bits of 1.0 as an i64.
assert.sameValue(call(mod([0x7c], [0x7e], [0x20, 0, 0xbd]), 1.0), 4607182418800017408n, "f64.reinterpret -> bits of 1.0");
