/*---
description: Promise static methods (and Map.groupBy) are readable functions, not just callable
features: [Promise.allSettled, Promise.any, Map]
---*/
// Readable for feature detection (typeof / truthiness), previously undefined.
assert.sameValue(typeof Promise.resolve, "function", "Promise.resolve");
assert.sameValue(typeof Promise.reject, "function", "Promise.reject");
assert.sameValue(typeof Promise.all, "function", "Promise.all");
assert.sameValue(typeof Promise.race, "function", "Promise.race");
assert.sameValue(typeof Promise.allSettled, "function", "Promise.allSettled");
assert.sameValue(typeof Promise.any, "function", "Promise.any");
assert.sameValue(typeof Promise.withResolvers, "function", "Promise.withResolvers");
assert.sameValue(typeof Map.groupBy, "function", "Map.groupBy");

// They are non-enumerable (not surfaced by Object.keys).
assert.sameValue(Object.keys(Promise).length, 0, "Promise statics non-enumerable");

// Calling still works (both directly and after reading the reference).
var resolved;
Promise.resolve(7).then(function (v) { resolved = v; });

var p = Promise.withResolvers();
assert.sameValue(typeof p.promise, "object", "withResolvers returns a promise");
assert.sameValue(typeof p.resolve, "function", "withResolvers resolve");
assert.sameValue(typeof p.reject, "function", "withResolvers reject");

// Map.groupBy partitions into a Map.
var g = Map.groupBy([1, 2, 3, 4], function (x) { return x % 2 ? "odd" : "even"; });
assert.sameValue(g instanceof Map, true, "Map.groupBy returns a Map");
assert.sameValue(g.get("odd").join(","), "1,3", "odd group");
assert.sameValue(g.get("even").join(","), "2,4", "even group");

// Built-in constructor/namespace statics, constants, and `.prototype` are non-enumerable.
assert.sameValue(Object.keys(Math).length, 0, "Math has no enumerable keys");
assert.sameValue(Object.keys(Array).length, 0, "Array statics non-enumerable");
assert.sameValue(Object.keys(Object).length, 0, "Object statics + prototype non-enumerable");
assert.sameValue(Object.keys(Number).length, 0, "Number statics + prototype non-enumerable");
assert.sameValue(Object.keys(Reflect).length, 0, "Reflect statics non-enumerable");
// ...but still present as own properties and callable/readable.
assert.sameValue(Object.getOwnPropertyNames(Math).indexOf("PI") >= 0, true, "Math.PI is an own property");
assert.sameValue(typeof Array.isArray, "function", "Array.isArray readable");
assert.sameValue(Math.abs(-5), 5, "Math.abs callable");
assert.sameValue(typeof Object.prototype, "object", "Object.prototype readable");

// A detached static call routes to its constructor regardless of `this`.
var isInt = Number.isInteger;
assert.sameValue(isInt(7), true, "detached Number.isInteger(7)");
assert.sameValue(isInt(7.5), false, "detached Number.isInteger(7.5)");
var fcc = String.fromCharCode;
assert.sameValue(fcc(72, 73), "HI", "detached String.fromCharCode");
var groupBy = Map.groupBy;
assert.sameValue(groupBy([1, 2, 3], function (x) { return x % 2 ? "o" : "e"; }).get("o").join(","), "1,3", "detached Map.groupBy");
