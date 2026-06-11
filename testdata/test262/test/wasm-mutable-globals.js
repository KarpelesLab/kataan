/*---
description: WASM module-defined mutable globals (i64/f64) accumulate state across calls
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }

// (global (mut i64) i64.const 0) ; f(x) { g += x; return g }
function i64Mod() {
  var body = [0, 0x23, 0, 0x20, 0, 0x7c, 0x24, 0, 0x23, 0, 0x0b];
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 1, 0x7e, 1, 0x7e]),
    sec(3, [1, 0]),
    sec(6, [1, 0x7e, 0x01, 0x42, 0x00, 0x0b]),
    sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(body.length), body))));
}
var g = new WebAssembly.Instance(new WebAssembly.Module(i64Mod())).exports.f;
assert.sameValue(g(5n), 5n, "global 0 + 5 = 5");
assert.sameValue(g(3n), 8n, "global persists: 5 + 3 = 8");
assert.sameValue(g(100n), 108n, "8 + 100 = 108");

// A second instance has independent global state.
var g2 = new WebAssembly.Instance(new WebAssembly.Module(i64Mod())).exports.f;
assert.sameValue(g2(1n), 1n, "fresh instance starts at 0");
assert.sameValue(g(0n), 108n, "first instance unaffected");

// (global (mut f64) f64.const 0) ; f(x) { g += x; return g }
function f64Mod() {
  var body = [0, 0x23, 0, 0x20, 0, 0xa0, 0x24, 0, 0x23, 0, 0x0b];
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 1, 0x7c, 1, 0x7c]),
    sec(3, [1, 0]),
    sec(6, [1, 0x7c, 0x01, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0x0b]),
    sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(body.length), body))));
}
var f = new WebAssembly.Instance(new WebAssembly.Module(f64Mod())).exports.f;
assert.sameValue(f(1.5), 1.5, "f64 global + 1.5");
assert.sameValue(f(2.5), 4, "f64 global persists: 1.5 + 2.5 = 4");
