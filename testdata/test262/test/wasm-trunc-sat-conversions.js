/*---
description: WASM non-trapping (saturating) float-to-int conversions, i32/i64.trunc_sat_*
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
// op is the 0xfc sub-opcode; pt is the param valtype, rt the result valtype.
function conv(pt, rt, op) { return mod([pt], [rt], [0x20, 0, 0xfc, op]); }
var F64 = 0x7c, F32 = 0x7d, I32 = 0x7f, I64 = 0x7e;

// i32.trunc_sat_f64_s (0xfc 0x02): truncates toward zero, saturates, NaN -> 0.
var s32 = conv(F64, I32, 0x02);
assert.sameValue(call(s32, 3.7), 3, "3.7 -> 3");
assert.sameValue(call(s32, -3.7), -3, "-3.7 -> -3");
assert.sameValue(call(s32, NaN), 0, "NaN -> 0");
assert.sameValue(call(s32, 1e20) | 0, 2147483647, "+overflow -> INT32_MAX");
assert.sameValue(call(s32, -1e20) | 0, -2147483648, "-overflow -> INT32_MIN");

// i32.trunc_sat_f64_u (0xfc 0x03): unsigned saturation.
var u32 = conv(F64, I32, 0x03);
assert.sameValue(call(u32, 5.9) | 0, 5, "5.9 -> 5");
assert.sameValue(call(u32, -1) | 0, 0, "negative -> 0");
assert.sameValue(call(u32, 1e20) >>> 0, 4294967295, "+overflow -> UINT32_MAX");

// i32.trunc_sat_f32_s (0xfc 0x00).
var s32f = conv(F32, I32, 0x00);
assert.sameValue(call(s32f, -2.9), -2, "f32 -2.9 -> -2");
assert.sameValue(call(s32f, NaN), 0, "f32 NaN -> 0");

// i64.trunc_sat_f64_s (0xfc 0x06): BigInt result.
var s64 = conv(F64, I64, 0x06);
assert.sameValue(call(s64, 42.9), 42n, "i64 42.9 -> 42n");
assert.sameValue(call(s64, -42.9), -42n, "i64 -42.9 -> -42n");
assert.sameValue(call(s64, NaN), 0n, "i64 NaN -> 0n");
assert.sameValue(call(s64, 1e30), 9223372036854775807n, "i64 +overflow -> INT64_MAX");
