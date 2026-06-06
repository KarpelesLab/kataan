/*---
description: WebAssembly.validate decodes a module and reports well-formedness
features: [WebAssembly]
---*/
assert.sameValue(typeof WebAssembly, "object", "WebAssembly namespace exists");
assert.sameValue(typeof WebAssembly.validate, "function", "validate is a function");
// A minimal well-formed module: magic + version only.
assert.sameValue(WebAssembly.validate([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0]), true, "minimal module");
// Bad magic is rejected.
assert.sameValue(WebAssembly.validate([0, 0, 0, 0, 1, 0, 0, 0]), false, "bad magic");
// Too short to hold a header.
assert.sameValue(WebAssembly.validate([0, 0x61]), false, "truncated");
// A complete add module (type/function/export/code sections) validates.
var add = [
  0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
  1, 7, 1, 0x60, 2, 0x7f, 0x7f, 1, 0x7f,
  3, 2, 1, 0,
  7, 7, 1, 3, 0x61, 0x64, 0x64, 0, 0,
  0x0a, 9, 1, 7, 0, 0x20, 0, 0x20, 1, 0x6a, 0x0b
];
assert.sameValue(WebAssembly.validate(add), true, "full add module");
