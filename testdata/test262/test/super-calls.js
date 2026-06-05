/*---
description: super constructor and method calls
esid: sec-super-keyword
---*/
class Animal {
  constructor(name) { this.name = name; }
  speak() { return this.name + " makes a sound"; }
}
class Dog extends Animal {
  constructor(name) { super(name); this.type = "dog"; }
  speak() { return super.speak() + " (woof)"; }
}
var d = new Dog("Rex");
assert.sameValue(d.name, "Rex", "super constructor sets name");
assert.sameValue(d.type, "dog");
assert.sameValue(d.speak(), "Rex makes a sound (woof)", "super method call");
assert.sameValue(d instanceof Dog, true);
assert.sameValue(d instanceof Animal, true);
class Cat extends Animal {
  speak() { return super.speak() + " (meow)"; }
}
var cat = new Cat("Felix");
assert.sameValue(cat.speak(), "Felix makes a sound (meow)");
class Puppy extends Dog {
  speak() { return super.speak() + "!"; }
}
assert.sameValue(new Puppy("Spot").speak(), "Spot makes a sound (woof)!", "two-level super");
