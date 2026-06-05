/*---
description: new with spread arguments and constructor returns
esid: sec-new-operator
---*/
function Point(x, y, z) { this.x = x; this.y = y; this.z = z; }
var coords = [1, 2, 3];
var p = new Point(...coords);
assert.sameValue(p.x + "," + p.y + "," + p.z, "1,2,3", "spread into new");
class Vec {
  constructor(...components) { this.components = components; }
  get magnitude() { return Math.sqrt(this.components.reduce(function (s, c) { return s + c * c; }, 0)); }
}
var v = new Vec(3, 4);
assert.sameValue(v.magnitude, 5);
var v2 = new Vec(...[1, 2, 2]);
assert.sameValue(v2.magnitude, 3);
