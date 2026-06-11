/*---
description: WASM br_table (the switch/jump-table branch) dispatches by index with a default
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
function call(m, a) { return new WebAssembly.Instance(new WebAssembly.Module(m)).exports.f(a); }

// Four nested blocks; br_table [0,1,2] default 3 branches out by the i32 operand, then each
// landing pad returns a distinct constant (10/20/30 for indices 0/1/2, 40 for the default).
var body = [
  0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40,
  0x20, 0,                          // local.get 0
  0x0e, 0x03, 0x00, 0x01, 0x02, 0x03, // br_table 0 1 2 (default 3)
  0x0b, 0x41, 10, 0x0f,             // end b0; i32.const 10; return
  0x0b, 0x41, 20, 0x0f,             // end b1; i32.const 20; return
  0x0b, 0x41, 30, 0x0f,             // end b2; i32.const 30; return
  0x0b, 0x41, 40,                   // end b3; i32.const 40 (default fallthrough)
];
var f = mod([0x7f], [0x7f], body);
assert.sameValue(call(f, 0), 10, "index 0");
assert.sameValue(call(f, 1), 20, "index 1");
assert.sameValue(call(f, 2), 30, "index 2");
assert.sameValue(call(f, 3), 40, "index 3 -> default");
assert.sameValue(call(f, 5), 40, "out-of-range -> default");
assert.sameValue(call(f, 100), 40, "large index -> default");
