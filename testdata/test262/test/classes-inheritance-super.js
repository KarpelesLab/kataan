/*---
description: Class declarations — inheritance, super calls, static methods, getters
features: [class]
---*/
class Animal {
  constructor(n) { this.n = n; }
  speak() { return this.n + " makes a sound"; }
  static kind() { return "animal"; }
}
class Dog extends Animal {
  constructor(n) { super(n); }
  speak() { return super.speak() + " (woof)"; }
  get name() { return this.n; }
}
var d = new Dog("Rex");
assert.sameValue(d.speak(), "Rex makes a sound (woof)", "overridden method calls super");
assert.sameValue(d.name, "Rex", "getter reads instance field");
assert.sameValue(Animal.kind(), "animal", "static method on base class");
assert.sameValue(d instanceof Animal, true, "instance of base via extends");
assert.sameValue(d instanceof Dog, true, "instance of derived");

// Methods live on the prototype, shared across instances.
var d2 = new Dog("Fido");
assert.sameValue(d2.speak(), "Fido makes a sound (woof)", "second instance");
assert.sameValue(Object.getPrototypeOf(d) === Object.getPrototypeOf(d2), true, "shared prototype");
