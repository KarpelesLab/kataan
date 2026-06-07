/*---
description: Reflect.setPrototypeOf / isExtensible / preventExtensions
features: [Reflect]
---*/
assert.sameValue(typeof Reflect.setPrototypeOf, "function", "setPrototypeOf exists");
assert.sameValue(typeof Reflect.isExtensible, "function", "isExtensible exists");
assert.sameValue(typeof Reflect.preventExtensions, "function", "preventExtensions exists");

// setPrototypeOf returns a boolean and rewires the prototype chain.
var base = { greet() { return "hi"; } };
var o = {};
assert.sameValue(Reflect.setPrototypeOf(o, base), true, "setPrototypeOf returns true");
assert.sameValue(Reflect.getPrototypeOf(o), base, "prototype is set");
assert.sameValue(o.greet(), "hi", "inherits through the new prototype");

// setting the prototype to null.
var n = {};
Reflect.setPrototypeOf(n, null);
assert.sameValue(Reflect.getPrototypeOf(n), null, "null prototype");

// isExtensible / preventExtensions.
assert.sameValue(Reflect.isExtensible({}), true, "fresh object is extensible");
assert.sameValue(Reflect.isExtensible(Object.freeze({})), false, "frozen is not extensible");
var f = {};
assert.sameValue(Reflect.preventExtensions(f), true, "preventExtensions returns true");
assert.sameValue(Reflect.isExtensible(f), false, "no longer extensible");
