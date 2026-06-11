/*---
description: WASM sign-extension operators (i32.extend8_s/16_s, i64.extend8_s/16_s/32_s)
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
function un32(op) { return mod([0x7f], [0x7f], [0x20, 0, op]); }
function un64(op) { return mod([0x7e], [0x7e], [0x20, 0, op]); }

// i32.extend8_s: sign-extends the low byte.
assert.sameValue(call(un32(0xc0), 0xff), -1, "extend8_s 0xff -> -1");
assert.sameValue(call(un32(0xc0), 0x7f), 127, "extend8_s 0x7f -> 127");
assert.sameValue(call(un32(0xc0), 0x80), -128, "extend8_s 0x80 -> -128");
assert.sameValue(call(un32(0xc0), 0x1ff), -1, "extend8_s ignores high bits");

// i32.extend16_s: sign-extends the low 16 bits.
assert.sameValue(call(un32(0xc1), 0xffff), -1, "extend16_s 0xffff -> -1");
assert.sameValue(call(un32(0xc1), 0x7fff), 32767, "extend16_s 0x7fff -> 32767");
assert.sameValue(call(un32(0xc1), 0x8000), -32768, "extend16_s 0x8000 -> -32768");

// i64 variants (BigInt results).
assert.sameValue(call(un64(0xc2), 255n), -1n, "i64.extend8_s");
assert.sameValue(call(un64(0xc3), 65535n), -1n, "i64.extend16_s");
assert.sameValue(call(un64(0xc4), 0xFFFFFFFFn), -1n, "i64.extend32_s");
assert.sameValue(call(un64(0xc4), 0x7FFFFFFFn), 0x7FFFFFFFn, "i64.extend32_s positive stays");
