/*---
description: globalThis is the global object and references itself
esid: sec-globalthis
---*/
var g = globalThis;
assert.sameValue(typeof g, "object", "globalThis is an object");
assert.sameValue(g.globalThis, g, "globalThis references itself");
assert.sameValue(g.globalThis.globalThis === g, true, "self-reference is stable");
// Built-in globals are reachable as properties.
assert.sameValue(typeof g.Math, "object", "globalThis.Math");
assert.sameValue(g.Math.max(1, 2, 3), 3, "globalThis.Math.max");
assert.sameValue(typeof g.JSON, "object", "globalThis.JSON");
assert.sameValue(g.parseInt("42px"), 42, "globalThis.parseInt");
assert.sameValue(g.Array.isArray([]), true, "globalThis.Array.isArray");
assert.sameValue(g.Number.isInteger(5), true, "globalThis.Number.isInteger");
// The numeric value globals are present.
assert.sameValue(g.Infinity, Infinity, "globalThis.Infinity");
assert.sameValue(Number.isNaN(g.NaN), true, "globalThis.NaN");
assert.sameValue(g.undefined, undefined, "globalThis.undefined");
// A new global property can be read back through globalThis.
g.customGlobal = 123;
assert.sameValue(g.customGlobal, 123, "a property written on globalThis reads back");
