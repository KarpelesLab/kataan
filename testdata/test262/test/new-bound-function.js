/*---
description: new on a bound function constructs the target with bound args prepended
esid: sec-bound-function-exotic-objects-construct-argumentslist-newtarget
---*/
function Point(x, y) { this.x = x; this.y = y; }
var BoundPoint = Point.bind(null);
var p = new BoundPoint(3, 4);
assert.sameValue(p.x, 3, "new bound: arg 0");
assert.sameValue(p.y, 4, "new bound: arg 1");
assert.sameValue(p instanceof Point, true, "instance of the target");
var PartialPoint = Point.bind(null, 10);
var pp = new PartialPoint(20);
assert.sameValue(pp.x, 10, "bound (partial) arg");
assert.sameValue(pp.y, 20, "new arg after the bound one");
assert.sameValue(pp instanceof Point, true);
function Three(a, b, c) { this.sum = a + b + c; }
var BoundThree = Three.bind(null, 1, 2);
assert.sameValue(new BoundThree(3).sum, 6, "two bound + one new");
var ReBound = Point.bind(null, 5).bind(null, 6);
var rp = new ReBound();
assert.sameValue(rp.x, 5, "re-bound first arg");
assert.sameValue(rp.y, 6, "re-bound second arg");
class Cls { constructor(v) { this.v = v; } }
var BoundCls = Cls.bind(null);
assert.sameValue(new BoundCls(42).v, 42, "new on a bound class");
var PartialCls = Cls.bind(null, 7);
assert.sameValue(new PartialCls().v, 7, "bound class with a bound arg");
