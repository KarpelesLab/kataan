/*---
description: new on a constructor function binds this, returns the instance, and instanceof matches
esid: sec-function-definitions
---*/
function Point(x, y) {
  this.x = x;
  this.y = y;
}
var p = new Point(3, 4);
assert.sameValue(p.x, 3);
assert.sameValue(p.y, 4);
assert.sameValue(p instanceof Point, true);

function Other() {}
assert.sameValue(p instanceof Other, false);

// A constructor returning an object overrides the new instance.
function Factory() {
  this.ignored = 1;
  return { made: true };
}
var f = new Factory();
assert.sameValue(f.made, true);
assert.sameValue(f.ignored, undefined);
