/*---
description: Array.prototype methods are first-class values usable via call/apply on array-likes
esid: sec-properties-of-the-array-prototype-object
---*/
// The classic arguments-to-array idiom.
function toArr() { return Array.prototype.slice.call(arguments); }
assert.sameValue(toArr(1, 2, 3).join(","), "1,2,3", "slice.call(arguments)");

// map/filter/forEach/reduce via call.
function dbl() { return Array.prototype.map.call(arguments, function (x) { return x * 2; }); }
assert.sameValue(dbl(1, 2, 3).join(","), "2,4,6", "map.call");
assert.sameValue(Array.prototype.filter.call([1, 2, 3, 4], function (x) { return x % 2 === 0; }).join(","), "2,4", "filter.call");
assert.sameValue(Array.prototype.reduce.call([1, 2, 3], function (a, b) { return a + b; }, 0), 6, "reduce.call");
assert.sameValue(Array.prototype.join.call([1, 2, 3], "-"), "1-2-3", "join.call");
assert.sameValue(Array.prototype.indexOf.call([1, 2, 3], 2), 1, "indexOf.call");
assert.sameValue(Array.prototype.slice.call([1, 2, 3, 4], 1, 3).join(","), "2,3", "slice.call with bounds");

// apply form.
assert.sameValue(Array.prototype.concat.apply([1], [[2, 3]]).join(","), "1,2,3", "concat.apply");

// Array.prototype is itself an Array exotic object (ECMA-262 23.1.3), so
// `Array.isArray` reports true for it even though it holds no elements.
assert.sameValue(typeof Array.prototype, "object", "Array.prototype is an object");
assert.sameValue(Array.isArray(Array.prototype), true, "is an array exotic object");
assert.sameValue(Array.prototype.length, 0, "empty");
assert.sameValue(Object.keys(Array.prototype).length, 0, "methods are non-enumerable");
assert.sameValue(Array.prototype.constructor, Array, "constructor link");

// Ordinary array-method calls are unaffected.
assert.sameValue([1, 2, 3].map(function (x) { return x + 1; }).join(","), "2,3,4", "normal map");
