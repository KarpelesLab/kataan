/*---
description: class extends/super and method override
esid: sec-class-definitions
features: [class]
---*/
class Animal {
  constructor(name) { this.name = name; }
  describe() { return this.name + " makes a sound"; }
}
class Dog extends Animal {
  describe() { return super.describe() + " (woof)"; }
}
assert.sameValue(new Dog("Rex").describe(), "Rex makes a sound (woof)");
assert.sameValue(new Dog("Rex") instanceof Animal, true, "subclass instanceof base");
