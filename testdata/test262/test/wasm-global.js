/*---
description: WebAssembly.Global constructor, .value accessor, mutability
features: [WebAssembly]
---*/
assert.sameValue(typeof WebAssembly.Global, "function", "constructor exists");

// i32 global: ToInt32 coercion on init and on set; mutable .value.
var g = new WebAssembly.Global({ value: "i32", mutable: true }, 42);
assert.sameValue(g.value, 42, "init value");
g.value = 100;
assert.sameValue(g.value, 100, "set value");
g.value = 3.9;
assert.sameValue(g.value, 3, "i32 ToInt32 coercion");

// Immutable global: assigning .value throws and the value is unchanged.
var gi = new WebAssembly.Global({ value: "i32" }, 7);
var threw = false;
try { gi.value = 9; } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "immutable global throws on set");
assert.sameValue(gi.value, 7, "immutable value unchanged");

// f64 and i64 (BigInt) value types.
assert.sameValue(new WebAssembly.Global({ value: "f64", mutable: true }, 1.5).value, 1.5, "f64");
var gb = new WebAssembly.Global({ value: "i64", mutable: true }, 9007199254740993n);
assert.sameValue(gb.value, 9007199254740993n, "i64 value");
assert.sameValue(typeof gb.value, "bigint", "i64 is a BigInt");
