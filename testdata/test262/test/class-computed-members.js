/*---
description: Computed method names and static members in classes
esid: sec-class-definitions
---*/
var key = "dynamic";
class C {
  [key]() { return "computed"; }
  static origin() { return "static"; }
  static label = "C";
}
var c = new C();
assert.sameValue(c.dynamic(), "computed");
assert.sameValue(C.origin(), "static");
assert.sameValue(C.label, "C");
