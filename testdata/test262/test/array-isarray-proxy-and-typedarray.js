/*---
description: Array.isArray unwraps proxies to the target and excludes typed arrays
esid: sec-array.isarray
features: [Proxy, TypedArray]
---*/
// Genuine arrays.
assert.sameValue(Array.isArray([]), true, "empty array");
assert.sameValue(Array.isArray([1, 2, 3]), true, "array");
assert.sameValue(Array.isArray(new Array(5)), true, "new Array");

// A proxy whose target is an array is an array (recursively).
assert.sameValue(Array.isArray(new Proxy([], {})), true, "proxy of array");
assert.sameValue(Array.isArray(new Proxy(new Proxy([], {}), {})), true, "nested proxy of array");
assert.sameValue(Array.isArray(new Proxy({}, {})), false, "proxy of plain object");

// Typed arrays are NOT Arrays (even though they are array-backed here).
assert.sameValue(Array.isArray(new Uint8Array(3)), false, "Uint8Array");
assert.sameValue(Array.isArray(new Float64Array(2)), false, "Float64Array");
assert.sameValue(Array.isArray(new Proxy(new Uint8Array(3), {})), false, "proxy of typed array");

// Non-arrays.
assert.sameValue(Array.isArray({ length: 3 }), false, "array-like object");
assert.sameValue(Array.isArray("abc"), false, "string");
assert.sameValue(Array.isArray(null), false, "null");
assert.sameValue(Array.isArray(function () {}), false, "function");

// A revoked proxy in the chain throws a TypeError.
var r = Proxy.revocable([], {});
r.revoke();
var threw = false;
try { Array.isArray(r.proxy); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "revoked proxy throws");
