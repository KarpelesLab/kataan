/*---
description: WebAssembly.Module.exports / imports static introspection methods
features: [WebAssembly]
---*/
assert.sameValue(typeof WebAssembly.Module.exports, "function", "Module.exports is a function");
assert.sameValue(typeof WebAssembly.Module.imports, "function", "Module.imports is a function");

// A module exporting two globals.
var bytes = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  6, 11, 2, 0x7f, 0, 0x41, 42, 0x0b, 0x7f, 1, 0x41, 7, 0x0b,
  7, 9, 2, 1, 103, 3, 0, 1, 109, 3, 1,
]);
var mod = new WebAssembly.Module(bytes);
assert.sameValue(JSON.stringify(WebAssembly.Module.exports(mod)),
  '[{"name":"g","kind":"global"},{"name":"m","kind":"global"}]', "global exports");
assert.sameValue(JSON.stringify(WebAssembly.Module.imports(mod)), "[]", "no imports");

// A module importing a global and exporting a function.
var bytes2 = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 5, 1, 0x60, 0, 1, 0x7f,
  2, 0x0a, 1, 3, 101, 110, 118, 1, 103, 3, 0x7f, 0,
  3, 2, 1, 0,
  7, 7, 1, 3, 103, 101, 116, 0, 0,
  0x0a, 6, 1, 4, 0, 0x23, 0, 0x0b,
]);
var mod2 = new WebAssembly.Module(bytes2);
assert.sameValue(JSON.stringify(WebAssembly.Module.imports(mod2)),
  '[{"module":"env","name":"g","kind":"global"}]', "imported global descriptor");
assert.sameValue(JSON.stringify(WebAssembly.Module.exports(mod2)),
  '[{"name":"get","kind":"function"}]', "function export descriptor");
