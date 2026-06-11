/*---
description: WASM locals of mixed types (i32/i64/f32/f64) with local.get/set/tee
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }
// `locals` is the raw local-declaration vector (group-count, then (count, valtype) groups).
function mod(params, results, locals, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var code = locals.concat(body, [0x0b]);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1].concat(type)), sec(3, [1, 0]), sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(code.length), code))));
}
function call(m) { var i = new WebAssembly.Instance(new WebAssembly.Module(m)); return i.exports.f.apply(null, Array.prototype.slice.call(arguments, 1)); }
var F64 = 0x7c, I64 = 0x7e, F32 = 0x7d, I32 = 0x7f;

// f64 local + local.tee: tee leaves the value on the stack while storing it, so this is 2*x.
// body: local.get 0; local.tee 1; local.get 1; f64.add
var tee = mod([F64], [F64], [1, 1, F64], [0x20, 0, 0x22, 1, 0x20, 1, 0xa0]);
assert.sameValue(call(tee, 3.5), 7, "f64 local.tee");
assert.sameValue(call(tee, 10), 20, "f64 local.tee again");

// Mixed locals: (param i32) with one i64 and one f32 local. Computes 2 + param:
// f32.const 2.5; local.set 2; local.get 2; i64.trunc_f32_s; local.set 1;
// local.get 1; local.get 0; i64.extend_i32_s; i64.add
var multi = mod([I32], [I64], [2, 1, I64, 1, F32],
  [0x43, 0, 0, 0x20, 0x40, 0x21, 2, 0x20, 2, 0xae, 0x21, 1, 0x20, 1, 0x20, 0, 0xac, 0x7c]);
assert.sameValue(call(multi, 10), 12n, "mixed i64/f32 locals: 2 + 10");
assert.sameValue(call(multi, -3), -1n, "mixed locals with a negative param");

// i64 local with local.set / local.get: local0 = 42; return local0 + local0.
var i64loc = mod([], [I64], [1, 1, I64], [0x42, 0x2a, 0x21, 0, 0x20, 0, 0x20, 0, 0x7c]);
assert.sameValue(call(i64loc), 84n, "i64 local set/get");

// Locals default-initialize to zero (an unwritten i32 local read back).
var zero = mod([], [I32], [1, 1, I32], [0x20, 0]);
assert.sameValue(call(zero), 0, "unwritten local defaults to 0");
