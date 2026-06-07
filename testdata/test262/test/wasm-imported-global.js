/*---
description: a WASM module's imported global is supplied from a WebAssembly.Global or a number
features: [WebAssembly]
---*/
// (module (import "env" "g" (global i32))
//         (func (export "get") (result i32) (global.get 0)))
var bytes = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 5, 1, 0x60, 0, 1, 0x7f,
  2, 0x0a, 1, 3, 101, 110, 118, 1, 103, 3, 0x7f, 0,
  3, 2, 1, 0,
  7, 7, 1, 3, 103, 101, 116, 0, 0,
  0x0a, 6, 1, 4, 0, 0x23, 0, 0x0b,
]);
var mod = new WebAssembly.Module(bytes);

// Supplied via a WebAssembly.Global.
var inst = new WebAssembly.Instance(mod, { env: { g: new WebAssembly.Global({ value: "i32" }, 42) } });
assert.sameValue(inst.exports.get(), 42, "imported global from WebAssembly.Global");

// Supplied as a plain number.
var inst2 = new WebAssembly.Instance(mod, { env: { g: 7 } });
assert.sameValue(inst2.exports.get(), 7, "imported global from a number");
