/*---
description: WASM sub-width memory ops (i32.load8/16_s/u, i32.store8/16) — sign/zero extension and store truncation
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }
// Module with one page of memory exporting f.
function mmod(params, results, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var code = [0].concat(body, [0x0b]);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1].concat(type)), sec(3, [1, 0]), sec(5, [1, 0x00, 0x01]),
    sec(7, [1, 1, 0x66, 0, 0]), sec(10, [1].concat(uleb(code.length), code))));
}
function call(m) { var i = new WebAssembly.Instance(new WebAssembly.Module(m)); return i.exports.f.apply(null, Array.prototype.slice.call(arguments, 1)); }
// f(v): store v at address 0 with storeOp, then load it back with loadOp.
//   i32.const 0; local.get 0; <store> 0 0; i32.const 0; <load> 0 0
function sl(storeOp, loadOp) { return mmod([0x7f], [0x7f], [0x41, 0, 0x20, 0, storeOp, 0x00, 0x00, 0x41, 0, loadOp, 0x00, 0x00]); }

// i32.store8 (0x3a) then i32.load8_s (0x2c) sign-extends the byte.
assert.sameValue(call(sl(0x3a, 0x2c), 0xFF), -1, "load8_s 0xFF -> -1");
assert.sameValue(call(sl(0x3a, 0x2c), 0x7F), 127, "load8_s 0x7F -> 127");
assert.sameValue(call(sl(0x3a, 0x2c), 0x80), -128, "load8_s 0x80 -> -128");
// i32.load8_u (0x2d) zero-extends.
assert.sameValue(call(sl(0x3a, 0x2d), 0xFF), 255, "load8_u 0xFF -> 255");
assert.sameValue(call(sl(0x3a, 0x2d), 0x80), 128, "load8_u 0x80 -> 128");
// i32.store8 truncates the stored value to 8 bits.
assert.sameValue(call(sl(0x3a, 0x2d), 0x1FF), 255, "store8 truncates 0x1FF -> 0xFF");

// i32.store16 (0x3b) + i32.load16_s (0x2e) / load16_u (0x2f).
assert.sameValue(call(sl(0x3b, 0x2e), 0xFFFF), -1, "load16_s 0xFFFF -> -1");
assert.sameValue(call(sl(0x3b, 0x2e), 0x7FFF), 32767, "load16_s 0x7FFF -> 32767");
assert.sameValue(call(sl(0x3b, 0x2e), 0x8000), -32768, "load16_s 0x8000 -> -32768");
assert.sameValue(call(sl(0x3b, 0x2f), 0xFFFF), 65535, "load16_u 0xFFFF -> 65535");
assert.sameValue(call(sl(0x3b, 0x2f), 0x1FFFF), 65535, "store16 truncates 0x1FFFF -> 0xFFFF");
