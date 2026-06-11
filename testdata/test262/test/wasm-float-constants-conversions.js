/*---
description: WASM float constants, promote/demote, signed/unsigned conversions, reinterpret, nop/drop
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
function f64bytes(x) { var b = new ArrayBuffer(8); new DataView(b).setFloat64(0, x, true); return Array.prototype.slice.call(new Uint8Array(b)); }
function f32bytes(x) { var b = new ArrayBuffer(4); new DataView(b).setFloat32(0, x, true); return Array.prototype.slice.call(new Uint8Array(b)); }

// Float constants (immediate IEEE-754 operands).
assert.sameValue(call(mod([], [0x7c], [0x44].concat(f64bytes(3.14159)))), 3.14159, "f64.const");
assert.sameValue(call(mod([], [0x7d], [0x43].concat(f32bytes(2.5)))), 2.5, "f32.const");
assert.sameValue(call(mod([0x7c], [0x7c], [0x44].concat(f64bytes(10.0), [0x20, 0, 0xa0])), 5.5), 15.5, "f64.const + add");
assert.sameValue(call(mod([], [0x7c], [0x44].concat(f64bytes(Infinity)))), Infinity, "f64.const Infinity");
assert.sameValue(Number.isNaN(call(mod([], [0x7c], [0x44].concat(f64bytes(NaN))))), true, "f64.const NaN");

// promote / demote between f32 and f64.
assert.sameValue(call(mod([0x7d], [0x7c], [0x20, 0, 0xbb]), 1.5), 1.5, "f64.promote_f32");
assert.sameValue(call(mod([0x7c], [0x7d], [0x20, 0, 0xb6]), 1.5), 1.5, "f32.demote_f64");

// Signed/unsigned truncation and conversion.
assert.sameValue(call(mod([0x7c], [0x7f], [0x20, 0, 0xab]), 42.9), 42, "i32.trunc_f64_u");
assert.sameValue(call(mod([0x7f], [0x7c], [0x20, 0, 0xb8]), -1), 4294967295, "f64.convert_i32_u (unsigned)");
assert.sameValue(call(mod([0x7c], [0x7e], [0x20, 0, 0xb0]), 123.9), 123n, "i64.trunc_f64_s -> BigInt");
assert.sameValue(call(mod([0x7e], [0x7c], [0x20, 0, 0xb9]), 1000000000000n), 1000000000000, "f64.convert_i64_s");

// reinterpret: the IEEE-754 bits of f32 1.0 are 0x3f800000.
assert.sameValue(call(mod([0x7d], [0x7f], [0x20, 0, 0xbc]), 1.0), 1065353216, "i32.reinterpret_f32");

// nop and drop.
assert.sameValue(call(mod([], [0x7f], [0x41, 5, 0x41, 99, 0x1a, 0x01])), 5, "drop + nop");
