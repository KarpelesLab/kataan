/*---
description: WASM in-module memory.size / memory.grow ops (grow returns previous size; memory persists)
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }

// Three exported funcs over a 1-page memory:
//   grow(delta:i32) -> i32   (memory.grow, yields the previous page count)
//   size() -> i32            (memory.size)
//   store(addr:i32, v:i32)   then load(addr:i32)->i32   to prove the grown memory is usable.
function buildModule() {
  var tGrow = [0x60, 1, 0x7f, 1, 0x7f];        // (i32)->i32
  var tSize = [0x60, 0, 1, 0x7f];              // ()->i32
  var tStore = [0x60, 2, 0x7f, 0x7f, 0];       // (i32,i32)->()
  var tLoad = [0x60, 1, 0x7f, 1, 0x7f];        // (i32)->i32
  var growBody = [0, 0x20, 0, 0x40, 0, 0x0b];
  var sizeBody = [0, 0x3f, 0, 0x0b];
  // store: i32.store at addr (align 2, offset 0): local.get 0; local.get 1; i32.store
  var storeBody = [0, 0x20, 0, 0x20, 1, 0x36, 0x02, 0, 0x0b];
  // load: i32.load at addr: local.get 0; i32.load
  var loadBody = [0, 0x20, 0, 0x28, 0x02, 0, 0x0b];
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [4].concat(tGrow, tSize, tStore, tLoad)),
    sec(3, [4, 0, 1, 2, 3]),
    sec(5, [1, 0x00, 0x01]),
    sec(7, [4,
      4, 0x67, 0x72, 0x6f, 0x77, 0, 0,                 // "grow" -> f0
      4, 0x73, 0x69, 0x7a, 0x65, 0, 1,                 // "size" -> f1
      5, 0x73, 0x74, 0x6f, 0x72, 0x65, 0, 2,           // "store" -> f2
      4, 0x6c, 0x6f, 0x61, 0x64, 0, 3]),               // "load" -> f3
    sec(10, [4]
      .concat(uleb(growBody.length), growBody)
      .concat(uleb(sizeBody.length), sizeBody)
      .concat(uleb(storeBody.length), storeBody)
      .concat(uleb(loadBody.length), loadBody))));
}
var x = new WebAssembly.Instance(new WebAssembly.Module(buildModule())).exports;

assert.sameValue(x.size(), 1, "initial size is 1 page");
assert.sameValue(x.grow(2), 1, "grow(2) returns the previous page count");
assert.sameValue(x.size(), 3, "size reflects the growth");
assert.sameValue(x.grow(1), 3, "grow again returns previous count");
assert.sameValue(x.size(), 4, "size is now 4 pages");

// The newly grown region (page 3, well past the original 64KiB) is addressable.
var addr = 3 * 65536 + 16;
x.store(addr, 0x12345678);
assert.sameValue(x.load(addr) | 0, 0x12345678, "grown memory is readable/writable");
// And distinct from address 0.
x.store(0, 42);
assert.sameValue(x.load(0), 42, "low memory still works");
assert.sameValue(x.load(addr) | 0, 0x12345678, "high write undisturbed");
