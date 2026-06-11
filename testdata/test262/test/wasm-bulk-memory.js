/*---
description: WASM bulk-memory ops memory.fill and memory.copy (0xFC-prefix), including OOB traps
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sleb(n) {
  var b = [], more = true;
  while (more) { var x = n & 0x7f; n >>= 7; if ((n === 0 && (x & 0x40) === 0) || (n === -1 && (x & 0x40) !== 0)) more = false; else x |= 0x80; b.push(x); }
  return b;
}
function sec(id, p) { return [id].concat(uleb(p.length), p); }
function memmod(body) {
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]), sec(3, [1, 0]), sec(5, [1, 0, 1]), sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(body.length), body))));
}
function f(m) { var inst = new WebAssembly.Instance(new WebAssembly.Module(m)); return inst.exports.f; }
var k = function (n) { return [0x41].concat(sleb(n)); }; // i32.const n

// memory.fill: fill mem[0..10] with 200, then read mem[addr] (load8_u zero-extends).
var fill = f(memmod([0].concat(k(0), k(200), k(10), [0xfc, 0x0b, 0x00], [0x20, 0, 0x2d, 0, 0], [0x0b])));
assert.sameValue(fill(3), 200, "filled byte");
assert.sameValue(fill(9), 200, "last filled byte");
assert.sameValue(fill(10), 0, "byte past the fill is zero");

// memory.copy: store 42 at mem[0], copy 1 byte from mem[0] to mem[20], read it back.
var copyBody = [0]
  .concat(k(0), k(42), [0x3a, 0, 0])          // mem[0] = 42 (i32.store8)
  .concat(k(20), k(0), k(1), [0xfc, 0x0a, 0x00, 0x00]) // memory.copy dst=20 src=0 len=1
  .concat([0x20, 0, 0x2d, 0, 0], [0x0b]);     // load8_u mem[addr]
var copy = f(memmod(copyBody));
assert.sameValue(copy(0), 42, "source preserved");
assert.sameValue(copy(20), 42, "byte copied to destination");
assert.sameValue(copy(21), 0, "only one byte copied");

// memory.fill with an out-of-range range traps.
var oob = f(memmod([0].concat(k(0xffffff), k(7), k(0xffffff), [0xfc, 0x0b, 0x00], [0x20, 0, 0x2d, 0, 0], [0x0b])));
var trapped = false;
try { oob(0); } catch (e) { trapped = true; }
assert.sameValue(trapped, true, "out-of-bounds memory.fill traps");
