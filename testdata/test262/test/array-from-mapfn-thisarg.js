/*---
description: Array.from passes its thisArg to the map function
esid: sec-array.from
---*/
// The mapFn runs with the 3rd argument as `this`.
assert.sameValue(
  Array.from([1, 2, 3], function (x) { return x * this.m; }, { m: 10 }).join(","),
  "10,20,30",
  "thisArg passed to mapFn"
);

// And with an array-like source.
assert.sameValue(
  Array.from({ length: 2, 0: 5, 1: 6 }, function (x) { return x + this.add; }, { add: 100 }).join(","),
  "105,106",
  "thisArg with array-like source"
);

// The mapFn receives (value, index).
assert.sameValue(Array.from(["a", "b", "c"], function (x, i) { return x + i; }).join(","), "a0,b1,c2", "value and index");

// An arrow ignores thisArg (lexical this) but still maps.
assert.sameValue(Array.from([1, 2, 3], function (x) { return x; }.bind(null)).join(","), "1,2,3", "bound mapFn");
assert.sameValue(Array.from([1, 2, 3], (x) => x * 2).join(","), "2,4,6", "arrow mapFn");

// A typed-array from() also forwards the mapFn (with coercion).
assert.sameValue(Uint8Array.from([1, 2, 3], (x) => x * 2).join(","), "2,4,6", "typed array from mapFn");

// from() without a mapFn is unaffected.
assert.sameValue(Array.from("xyz").join(","), "x,y,z", "string source, no mapFn");
assert.sameValue(Array.from(new Set([1, 1, 2])).join(","), "1,2", "Set source");
