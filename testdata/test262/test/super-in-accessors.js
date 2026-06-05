/*---
description: super in methods and accessors through class chains
esid: sec-super-keyword
---*/
class Base {
  constructor() { this._value = 10; }
  getValue() { return this._value; }
  get doubled() { return this._value * 2; }
}
class Derived extends Base {
  getValue() { return super.getValue() + 5; }
  get doubled() { return super.doubled + 1; }
}
var d = new Derived();
assert.sameValue(d.getValue(), 15, "super method call");
assert.sameValue(d.doubled, 21, "super getter call");
class A {
  greet() { return "A"; }
}
class B extends A {
  greet() { return super.greet() + "B"; }
}
class C extends B {
  greet() { return super.greet() + "C"; }
}
assert.sameValue(new C().greet(), "ABC", "super chain");
class Shape {
  area() { return 0; }
  describe() { return "area is " + this.area(); }
}
class Circle extends Shape {
  constructor(r) { super(); this.r = r; }
  area() { return Math.floor(3.14 * this.r * this.r); }
}
assert.sameValue(new Circle(2).describe(), "area is 12", "polymorphic this");
