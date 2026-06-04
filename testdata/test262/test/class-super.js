/*---
description: super method/constructor calls and inherited accessors
esid: sec-super-keyword
---*/
class Animal {
  constructor(name) { this.name = name; }
  speak() { return this.name + " makes a sound"; }
  get label() { return "[" + this.name + "]"; }
}
class Dog extends Animal {
  constructor(name) { super(name); this.legs = 4; }
  speak() { return super.speak() + " (woof)"; }
}
var d = new Dog("Rex");
assert.sameValue(d.name, "Rex");
assert.sameValue(d.legs, 4);
assert.sameValue(d.speak(), "Rex makes a sound (woof)");
assert.sameValue(d.label, "[Rex]", "inherited getter");
assert.sameValue(d instanceof Animal, true);
assert.sameValue(d instanceof Dog, true);
