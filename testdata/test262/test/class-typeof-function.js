/*---
description: a class constructor is typeof "function"; instances are "object"
features: [class]
---*/
class C {}
assert.sameValue(typeof C, "function", "empty class");

class D { m() {} static s() {} }
assert.sameValue(typeof D, "function", "class with methods");

// Instances are objects.
assert.sameValue(typeof new C(), "object", "instance");

// A class is not an array, and (like any function) is omitted from JSON.
assert.sameValue(Array.isArray(C), false, "a class is not an array");
assert.sameValue(JSON.stringify({ cls: C, n: 1 }), '{"n":1}', "class omitted from JSON");

// The function tag does not leak into enumeration of an empty class.
assert.sameValue(Object.keys(C).length, 0, "empty class has no own enumerable keys");
