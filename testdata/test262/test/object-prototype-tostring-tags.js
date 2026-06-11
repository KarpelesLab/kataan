/*---
description: Object.prototype.toString builtin tags for primitives, wrappers, Map/Set
esid: sec-object.prototype.tostring
---*/
var ts = Object.prototype.toString;

// Primitive number/boolean (immediates) and string report their class.
assert.sameValue(ts.call(5), "[object Number]", "number primitive");
assert.sameValue(ts.call("s"), "[object String]", "string primitive");
assert.sameValue(ts.call(true), "[object Boolean]", "boolean primitive");

// Boxed primitive wrappers too.
assert.sameValue(ts.call(new Number(5)), "[object Number]", "Number wrapper");
assert.sameValue(ts.call(new String("x")), "[object String]", "String wrapper");
assert.sameValue(ts.call(new Boolean(true)), "[object Boolean]", "Boolean wrapper");

// Map and Set (via their Symbol.toStringTag).
assert.sameValue(ts.call(new Map()), "[object Map]", "Map");
assert.sameValue(ts.call(new Set()), "[object Set]", "Set");

// The existing tags are unchanged.
assert.sameValue(ts.call([]), "[object Array]", "Array");
assert.sameValue(ts.call(null), "[object Null]", "null");
assert.sameValue(ts.call(undefined), "[object Undefined]", "undefined");
assert.sameValue(ts.call({}), "[object Object]", "plain object");
assert.sameValue(ts.call(/x/), "[object RegExp]", "RegExp");
assert.sameValue(ts.call(function () {}), "[object Function]", "Function");
assert.sameValue(ts.call(new Date()), "[object Date]", "Date");

// An explicit Symbol.toStringTag overrides the builtin tag.
assert.sameValue(ts.call({ [Symbol.toStringTag]: "Custom" }), "[object Custom]", "custom tag");
