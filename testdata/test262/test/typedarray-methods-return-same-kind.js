/*---
description: TypedArray map/filter/slice/toReversed/toSorted/with return a same-kind typed array
features: [TypedArray]
---*/
var u = new Uint8Array([1, 2, 3]);

// Each returns a Uint8Array (not a plain Array).
assert.sameValue(u.map(function (x) { return x * 2; }) instanceof Uint8Array, true, "map");
assert.sameValue(u.filter(function (x) { return x > 1; }) instanceof Uint8Array, true, "filter");
assert.sameValue(u.slice(1) instanceof Uint8Array, true, "slice");
assert.sameValue(u.toReversed() instanceof Uint8Array, true, "toReversed");
assert.sameValue(u.toSorted() instanceof Uint8Array, true, "toSorted");
assert.sameValue(u.with(0, 9) instanceof Uint8Array, true, "with");

// Results carry the right values, and overflow is coerced to the element type.
assert.sameValue(u.map(function (x) { return x * 2; }).join(","), "2,4,6", "map values");
assert.sameValue(new Uint8Array([1, 2, 3]).map(function (x) { return x * 100; }).join(","), "100,200,44", "map coerces (300 -> 44)");
assert.sameValue(u.filter(function (x) { return x > 1; }).join(","), "2,3", "filter values");

// Chaining typed-array methods stays typed.
assert.sameValue(new Uint8Array([1, 2, 3, 4]).filter(function (x) { return x % 2; }).map(function (x) { return x * 10; }).join(","), "10,30", "chained");

// Element type is preserved across kinds.
var f = new Float64Array([1.5, 2.5]).map(function (x) { return x * 2; });
assert.sameValue(f instanceof Float64Array, true, "Float64 map kind");
assert.sameValue(f.join(","), "3,5", "Float64 values");

// A plain array's methods still return a plain Array.
assert.sameValue([1, 2, 3].map(function (x) { return x * 2; }) instanceof Uint8Array, false, "plain array not typed");
assert.sameValue(Array.isArray([1, 2, 3].map(function (x) { return x; })), true, "plain array stays array");
