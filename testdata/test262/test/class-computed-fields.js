/*---
description: Computed method/getter names and field initializers in classes
esid: sec-class-definitions
---*/
var mname = "run";
var gname = "doubled";
class Widget {
  constructor(v) { this.v = v; }
  [mname]() { return "running " + this.v; }
  get [gname]() { return this.v * 2; }
  static label = "widget";
}
var w = new Widget(5);
assert.sameValue(w.run(), "running 5", "computed method");
assert.sameValue(w.doubled, 10, "computed getter");
assert.sameValue(Widget.label, "widget", "static field");
