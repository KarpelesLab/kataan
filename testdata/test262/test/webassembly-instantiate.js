/*---
description: WebAssembly.instantiate builds an instance whose exports are callable from JS
features: [WebAssembly]
---*/
// (module (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add)
//         (func (export "mul") (param i32 i32) (result i32) local.get 0 local.get 1 i32.mul))
var bytes = [
  0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
  1, 7, 1, 0x60, 2, 0x7f, 0x7f, 1, 0x7f,
  3, 3, 2, 0, 0,
  7, 13, 2, 3, 0x61, 0x64, 0x64, 0, 0, 3, 0x6d, 0x75, 0x6c, 0, 1,
  0x0a, 17, 2,
  7, 0, 0x20, 0, 0x20, 1, 0x6a, 0x0b,
  7, 0, 0x20, 0, 0x20, 1, 0x6c, 0x0b
];
assert.sameValue(typeof WebAssembly.instantiate, "function", "instantiate is a function");
var result = WebAssembly.instantiate(bytes);
assert.sameValue(typeof result.instance, "object", "result.instance is an object");
var ex = result.instance.exports;
assert.sameValue(typeof ex.add, "function", "exports.add is callable");
assert.sameValue(typeof ex.mul, "function", "exports.mul is callable");
assert.sameValue(ex.add(20, 22), 42, "add(20,22)");
assert.sameValue(ex.add(-5, 8), 3, "add(-5,8)");
assert.sameValue(ex.mul(6, 7), 42, "mul(6,7)");
