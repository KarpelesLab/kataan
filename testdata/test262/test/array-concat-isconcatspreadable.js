/*---
description: Array.prototype.concat honors Symbol.isConcatSpreadable
esid: sec-array.prototype.concat
---*/
// By default arrays are spread; non-array values are appended as one element.
assert.sameValue([1, 2].concat([3, 4], 5).join(","), "1,2,3,4,5", "default spreading");
assert.sameValue([1].concat([[2]]).length, 2, "nested arrays not deep-spread");

// isConcatSpreadable = false on an array prevents it from being spread.
var ns = [1, 2];
ns[Symbol.isConcatSpreadable] = false;
var r = [0].concat(ns);
assert.sameValue(r.length, 2, "false -> not spread");
assert.sameValue(r[1], ns, "the array is appended as one element");

// isConcatSpreadable = true on an array-like spreads it (read by length + indices).
var al = { length: 2, 0: "a", 1: "b", [Symbol.isConcatSpreadable]: true };
assert.sameValue([0].concat(al).join(","), "0,a,b", "true -> array-like spread");

// true on an object without length spreads zero elements.
assert.sameValue([1].concat({ [Symbol.isConcatSpreadable]: true }).join(","), "1", "no length -> zero elements");

// A plain object (no flag) and a string are appended as a single element.
assert.sameValue([1].concat({ a: 1 }).length, 2, "plain object appended");
assert.sameValue([1].concat("ab").length, 2, "string appended");
assert.sameValue([1].concat("ab").join(","), "1,ab", "string not spread");
