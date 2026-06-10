/*---
description: WebAssembly modules can import and call JS functions
features: [WebAssembly]
---*/
// (import "env" "triple" (func (param i32) (result i32)))
// (func (export "run") (param i32) (result i32) local.get 0 call 0 i32.const 1 i32.add)
var bytes = [
  0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
  1, 6, 1, 0x60, 1, 0x7f, 1, 0x7f,
  2, 14, 1, 3, 0x65, 0x6e, 0x76, 6, 0x74, 0x72, 0x69, 0x70, 0x6c, 0x65, 0, 0,
  3, 2, 1, 0,
  7, 7, 1, 3, 0x72, 0x75, 0x6e, 0, 1,
  0x0a, 11, 1, 9, 0, 0x20, 0, 0x10, 0, 0x41, 1, 0x6a, 0x0b
];
var calls = 0;
var imports = { env: { triple: function (x) { calls = calls + 1; return x * 3; } } };
// instantiate() is async (Promise); use the synchronous Instance path with imports.
var inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), imports);
assert.sameValue(inst.exports.run(10), 31, "triple(10) + 1");
assert.sameValue(inst.exports.run(-2), -5, "triple(-2) + 1");
assert.sameValue(calls, 2, "the JS import was actually called twice");
