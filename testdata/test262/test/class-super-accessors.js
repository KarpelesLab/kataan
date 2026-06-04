/*---
description: super in methods, overriding, and chained inheritance
esid: sec-super-keyword
---*/
class A {
  constructor() { this.kind = "A"; }
  describe() { return "I am " + this.kind; }
  static make() { return new this(); }
}
class B extends A {
  constructor() { super(); this.kind = "B"; }
  describe() { return super.describe() + " (really B)"; }
}
class C extends B {
  describe() { return super.describe() + " (and C)"; }
}
var c = new C();
assert.sameValue(c.describe(), "I am B (really B) (and C)", "three-level super chain");
assert.sameValue(c instanceof A, true);
assert.sameValue(c instanceof B, true);
var made = C.make();
assert.sameValue(made instanceof C, true, "static make uses new.target-ish this");
