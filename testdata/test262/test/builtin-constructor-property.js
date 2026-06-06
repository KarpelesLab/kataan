/*---
description: Built-in values report their constructor via .constructor
esid: sec-properties-of-the-object-prototype-object
---*/
// Built-in literals/objects identify their global constructor.
assert.sameValue([].constructor, Array, "array");
assert.sameValue(({}).constructor, Object, "plain object");
assert.sameValue("x".constructor, String, "string");
assert.sameValue(/x/.constructor, RegExp, "regexp");
assert.sameValue((new Date()).constructor, Date, "date");
assert.sameValue((new Map()).constructor, Map, "map");
assert.sameValue((new Set()).constructor, Set, "set");

// Errors report their specific error constructor.
assert.sameValue((new TypeError("m")).constructor, TypeError, "TypeError");
assert.sameValue((new RangeError()).constructor, RangeError, "RangeError");
assert.sameValue((new Error()).constructor, Error, "base Error");

// User functions and classes (incl. inheritance) keep their own constructor.
function Bar() {}
assert.sameValue((new Bar()).constructor, Bar, "user function");
class Foo {}
assert.sameValue((new Foo()).constructor, Foo, "user class");
class Sub extends Foo {}
assert.sameValue((new Sub()).constructor, Sub, "subclass");

// An explicit own constructor is not overridden; a plain object that merely has a
// name is still an Object.
assert.sameValue(({ constructor: 42 }).constructor, 42, "explicit constructor wins");
assert.sameValue(({ name: "TypeError" }).constructor, Object, "name alone is not an error");
