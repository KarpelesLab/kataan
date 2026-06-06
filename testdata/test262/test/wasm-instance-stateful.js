/*---
description: a WebAssembly.Instance keeps mutable global/memory state across calls
features: [WebAssembly]
---*/
// (module (global $c (mut i32) i32.const 0)
//        (func (export "inc") (result i32) (global.set $c (i32.add (global.get $c) (i32.const 1))) (global.get $c)))
var bytes = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 5, 1, 0x60, 0, 1, 0x7f,
  3, 2, 1, 0,
  6, 6, 1, 0x7f, 1, 0x41, 0, 0x0b,
  7, 7, 1, 3, 105, 110, 99, 0, 0,
  10, 13, 1, 11, 0, 0x23, 0, 0x41, 1, 0x6a, 0x24, 0, 0x23, 0, 0x0b,
]);

var inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
assert.sameValue(inst.exports.inc(), 1, "first call");
assert.sameValue(inst.exports.inc(), 2, "global persists");
assert.sameValue(inst.exports.inc(), 3, "and again");

// A separate instance has its own independent state.
var inst2 = new WebAssembly.Instance(new WebAssembly.Module(bytes));
assert.sameValue(inst2.exports.inc(), 1, "second instance starts fresh");
assert.sameValue(inst.exports.inc(), 4, "first instance unaffected");
