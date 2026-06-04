/*---
description: Static methods and fields, and static inheritance
esid: sec-class-definitions
---*/
class Base {
  static create() { return new this(); }
  static kind = "base";
  tag() { return "base-tag"; }
}
class Derived extends Base {
  tag() { return "derived-tag"; }
}
assert.sameValue(Base.kind, "base");
assert.sameValue(typeof Base.create, "function");
var obj = new Derived();
assert.sameValue(obj.tag(), "derived-tag");
