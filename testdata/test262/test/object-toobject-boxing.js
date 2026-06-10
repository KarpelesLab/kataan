/*---
description: Object(value) boxes primitives, returns objects as-is, makes a fresh object for null/undefined
esid: sec-object-value
---*/
// Primitives are boxed in their wrapper.
assert.sameValue(Object(42).valueOf(), 42, "Object(number)");
assert.sameValue(Object(42) instanceof Number, true, "boxed number is a Number");
assert.sameValue(Object("hi").valueOf(), "hi", "Object(string)");
assert.sameValue(Object("hi").length, 2, "boxed string length");
assert.sameValue(Object("x") instanceof String, true, "boxed string is a String");
assert.sameValue(Object(true).valueOf(), true, "Object(boolean)");
assert.sameValue(!!Object(false), true, "boxed false is truthy (an object)");

// null / undefined (and no argument) yield a fresh empty object.
assert.sameValue(typeof Object(null), "object", "Object(null)");
assert.sameValue(typeof Object(undefined), "object", "Object(undefined)");
assert.sameValue(Object.keys(Object(null)).length, 0, "fresh empty object");
assert.sameValue(typeof Object(), "object", "Object() no arg");

// An existing object is returned unchanged (identity).
var o = { a: 1 };
assert.sameValue(Object(o), o, "Object(object) identity");
var arr = [1, 2, 3];
assert.sameValue(Object(arr), arr, "Object(array) identity");

// A boxed number coerces in arithmetic via valueOf.
assert.sameValue(Object(5) + 3, 8, "boxed number arithmetic");
