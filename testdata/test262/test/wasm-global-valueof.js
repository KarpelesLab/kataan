/*---
description: WebAssembly.Global.valueOf returns the value and coerces numerically
features: [WebAssembly]
---*/
var g = new WebAssembly.Global({ value: "i32", mutable: true }, 42);
assert.sameValue(g.valueOf(), 42, "valueOf returns the value");

// A Global coerces to its value in numeric contexts (ToPrimitive -> valueOf).
assert.sameValue(g + 0, 42, "numeric coercion (add)");
assert.sameValue(g * 2, 84, "numeric coercion (mul)");
g.value = 10;
assert.sameValue(g + 5, 15, "valueOf reflects an updated value");

// i64 globals coerce to BigInt.
var gi = new WebAssembly.Global({ value: "i64" }, 100n);
assert.sameValue(gi.valueOf(), 100n, "i64 valueOf is a BigInt");
assert.sameValue(typeof gi.valueOf(), "bigint", "i64 typeof");

// `value` and `valueOf` are non-enumerable.
assert.sameValue(Object.keys(g).length, 0, "Global has no own enumerable keys");
