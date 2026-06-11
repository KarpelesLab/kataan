/*---
description: WASM f32 param/result marshalling to/from JS, and f32<->f64 promote/demote
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
var F32 = 0x7d, F64 = 0x7c;

// An (f32)->(f32) identity rounds the JS argument to single precision on the way in.
var id = mod([F32], [F32], [0x20, 0]);
assert.sameValue(call(id, 1.5), 1.5, "1.5 is exact in f32");
assert.sameValue(call(id, -2.25), -2.25, "-2.25 exact");
assert.sameValue(call(id, 0.1), 0.10000000149011612, "0.1 rounds to the nearest f32");

// f32.add keeps single precision.
assert.sameValue(call(mod([F32, F32], [F32], [0x20, 0, 0x20, 1, 0x92]), 0.1, 0.2), 0.30000001192092896, "f32.add");

// f32.const round-trips its 4 little-endian bytes (1.5 = 0x3fc00000).
assert.sameValue(call(mod([], [F32], [0x43, 0, 0, 0xc0, 0x3f])), 1.5, "f32.const 1.5");

// f64.promote_f32 (0xbb) widens an f32 to f64 — the value is the rounded-to-f32 0.1.
assert.sameValue(call(mod([F32], [F64], [0x20, 0, 0xbb]), 0.1), 0.10000000149011612, "promote f32->f64");

// f32.demote_f64 (0xb6) narrows an f64 to f32.
assert.sameValue(call(mod([F64], [F32], [0x20, 0, 0xb6]), 0.1), 0.10000000149011612, "demote f64->f32");
// A value exactly representable in f32 demotes losslessly.
assert.sameValue(call(mod([F64], [F32], [0x20, 0, 0xb6]), 0.5), 0.5, "demote exact 0.5");
