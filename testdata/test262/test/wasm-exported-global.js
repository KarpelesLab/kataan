/*---
description: a module's exported globals are exposed as WebAssembly.Global objects
features: [WebAssembly]
---*/
// (module (global (export "g") i32 (i32.const 42))
//         (global (export "m") (mut i32) (i32.const 7)))
var bytes = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  6, 11, 2, 0x7f, 0, 0x41, 42, 0x0b, 0x7f, 1, 0x41, 7, 0x0b,
  7, 9, 2, 1, 103, 3, 0, 1, 109, 3, 1,
]);
var inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));

// An exported global is a WebAssembly.Global with the instance's value.
assert.sameValue(inst.exports.g instanceof WebAssembly.Global, true, "exported global is a Global");
assert.sameValue(inst.exports.g.value, 42, "immutable global value");
assert.sameValue(inst.exports.m.value, 7, "mutable global value");

// An immutable exported global rejects assignment; a mutable one accepts it.
var threw = false;
try { inst.exports.g.value = 99; } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "immutable export is read-only");
assert.sameValue(inst.exports.g.value, 42, "unchanged");
inst.exports.m.value = 100;
assert.sameValue(inst.exports.m.value, 100, "mutable export is settable");

// instanceof works for the constructed boundary objects too.
assert.sameValue(new WebAssembly.Global({ value: "i32" }, 5) instanceof WebAssembly.Global, true, "Global instanceof");
assert.sameValue(new WebAssembly.Memory({ initial: 1 }) instanceof WebAssembly.Memory, true, "Memory instanceof");
assert.sameValue(new WebAssembly.Table({ element: "anyfunc", initial: 1 }) instanceof WebAssembly.Table, true, "Table instanceof");
assert.sameValue(({}) instanceof WebAssembly.Global, false, "plain object is not a Global");
