/*---
description: Class fields, methods, and instance initialization
esid: sec-class-definitions
---*/
class Point {
  x = 0;
  y = 0;
  constructor(x, y) { this.x = x; this.y = y; }
  distanceTo(other) { return Math.sqrt((this.x - other.x) ** 2 + (this.y - other.y) ** 2); }
  toString() { return "(" + this.x + "," + this.y + ")"; }
}
var p1 = new Point(0, 0);
var p2 = new Point(3, 4);
assert.sameValue(p1.distanceTo(p2), 5, "method using fields");
assert.sameValue(p2.toString(), "(3,4)", "toString method");
class Stack {
  items = [];
  push(x) { this.items.push(x); return this; }
  pop() { return this.items.pop(); }
  get size() { return this.items.length; }
  get isEmpty() { return this.items.length === 0; }
}
var s = new Stack();
assert.sameValue(s.isEmpty, true);
s.push(1).push(2).push(3);
assert.sameValue(s.size, 3);
assert.sameValue(s.pop(), 3);
assert.sameValue(s.isEmpty, false);
class Defaults {
  name = "anonymous";
  count = 0;
  tags = [];
}
var d = new Defaults();
assert.sameValue(d.name, "anonymous");
assert.sameValue(d.count, 0);
assert.sameValue(Array.isArray(d.tags), true);
var d2 = new Defaults();
d2.tags.push("a");
assert.sameValue(d.tags.length, 0, "field arrays are per-instance");
