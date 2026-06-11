/*---
description: super in a static method/accessor resolves against the superclass's static members
esid: sec-super-keyword
---*/
// super.method() in a static method calls the parent's static method.
class Base { static create() { return "base"; } }
class Derived extends Base { static create() { return super.create() + "-derived"; } }
assert.sameValue(Derived.create(), "base-derived", "static method super");

// Through three levels.
class L1 { static who() { return "L1"; } }
class L2 extends L1 { static who() { return super.who() + "-L2"; } }
class L3 extends L2 { static who() { return super.who() + "-L3"; } }
assert.sameValue(L3.who(), "L1-L2-L3", "three-level static super");

// `this` inside a static method invoked via super is the original calling class.
class P { static label() { return "P"; } }
class C extends P { static label() { return super.label() + "-C"; } }
assert.sameValue(C.label(), "P-C", "static super keeps this");

// A static getter super.
class GB { static get x() { return 10; } }
class GD extends GB { static get x() { return super.x + 5; } }
assert.sameValue(GD.x, 15, "static getter super");

// Instance super still works, even alongside static super in the same class.
class M1 { m() { return "im1"; } static s() { return "sm1"; } }
class M2 extends M1 {
  m() { return super.m() + "|im2"; }
  static s() { return super.s() + "|sm2"; }
}
assert.sameValue(new M2().m(), "im1|im2", "instance super");
assert.sameValue(M2.s(), "sm1|sm2", "static super");

// A plain instance-super class is unaffected.
class A { greet() { return "hi"; } }
class B extends A { greet() { return super.greet() + " there"; } }
assert.sameValue(new B().greet(), "hi there", "instance-only super");
