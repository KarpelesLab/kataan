/*---
description: built-in methods are readable as values on instances (feature detection)
esid: sec-array.prototype
---*/
// typeof / truthiness of a method read from an instance.
assert.sameValue(typeof [].flat, "function", "typeof array method");
assert.sameValue(typeof [].map, "function", "typeof map");
assert.sameValue(!![].includes, true, "if (arr.includes)");
assert.sameValue(typeof "x".padStart, "function", "typeof string method");
assert.sameValue(!!"x".repeat, true, "if (str.repeat)");
assert.sameValue(typeof (function () {}).call, "function", "typeof fn.call");
assert.sameValue(typeof (function () {}).bind, "function", "typeof fn.bind");

// A non-method property is still undefined (no false positives).
assert.sameValue([].nonexistentMethod, undefined, "missing array prop");
assert.sameValue("x".bogus, undefined, "missing string prop");

// The read method is callable detached (with an explicit this).
var mapper = [].map;
assert.sameValue(mapper.call([1, 2, 3], function (x) { return x + 1; }).join(","), "2,3,4", "detached map");

// A polyfill guard: define only if absent.
if (!Array.prototype.flat) { throw new Error("flat should be present"); }
assert.sameValue(typeof Array.prototype.flat, "function", "Array.prototype.flat present");

// Ordinary method calls and string toString/valueOf are unaffected.
assert.sameValue([1, 2, 3].filter(function (x) { return x > 1; }).join(","), "2,3", "normal filter");
assert.sameValue("abc".toString(), "abc", "string toString");
assert.sameValue("abc".valueOf(), "abc", "string valueOf");
assert.sameValue([].constructor, Array, "constructor unaffected");
