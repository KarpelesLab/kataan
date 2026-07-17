/*---
description: instanceof consults a custom Symbol.hasInstance method
features: [Symbol.hasInstance, class]
---*/
// A static [Symbol.hasInstance] on a class overrides the ordinary check and
// applies to a primitive left-hand side.
class Even {
  static [Symbol.hasInstance](n) { return typeof n === "number" && n % 2 === 0; }
}
assert.sameValue(4 instanceof Even, true, "4 is even");
assert.sameValue(3 instanceof Even, false, "3 is odd");
assert.sameValue(0 instanceof Even, true, "0 is even");
assert.sameValue("x" instanceof Even, false, "non-number");

// A [Symbol.hasInstance] property on an ordinary function works too. It must be
// installed via defineProperty: %Function.prototype%[@@hasInstance] is
// { [[Writable]]: false }, so a plain `Matcher[Symbol.hasInstance] = ...`
// assignment is a silent no-op in sloppy mode and would NOT override it.
function Matcher() {}
Object.defineProperty(Matcher, Symbol.hasInstance, {
  value: function (x) { return x === "match"; },
  configurable: true,
});
assert.sameValue("match" instanceof Matcher, true, "custom match");
assert.sameValue("nope" instanceof Matcher, false, "custom non-match");

// The result is coerced to boolean.
var truthy = { [Symbol.hasInstance]() { return 1; } };
assert.sameValue(({}) instanceof truthy, true, "truthy return coerced to true");

// Ordinary instanceof (prototype chain) is unaffected when no hasInstance exists.
class Animal {}
class Dog extends Animal {}
var d = new Dog();
assert.sameValue(d instanceof Dog, true, "prototype chain: Dog");
assert.sameValue(d instanceof Animal, true, "prototype chain: Animal");
assert.sameValue([] instanceof Array, true, "native Array");
assert.sameValue(5 instanceof Object, false, "primitive is not an Object");
