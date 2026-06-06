/*---
description: typeof on built-in globals and value keywords
---*/
// Namespace-style builtins are objects; constructor builtins are functions.
assert.sameValue(typeof Math, "object", "Math");
assert.sameValue(typeof JSON, "object", "JSON");
assert.sameValue(typeof Reflect, "object", "Reflect");
assert.sameValue(typeof Number, "function", "Number");
assert.sameValue(typeof String, "function", "String");
assert.sameValue(typeof BigInt, "function", "BigInt");
assert.sameValue(typeof Symbol, "function", "Symbol");
assert.sameValue(typeof Promise, "function", "Promise");
assert.sameValue(typeof Map, "function", "Map");

// Value keywords keep their primitive types; an unknown name is "undefined"
// (and must not throw).
assert.sameValue(typeof NaN, "number", "NaN");
assert.sameValue(typeof Infinity, "number", "Infinity");
assert.sameValue(typeof undefined, "undefined", "undefined");
assert.sameValue(typeof someNameThatIsNotDefined, "undefined", "unknown name");
