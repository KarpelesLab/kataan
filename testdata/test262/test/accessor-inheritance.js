/*---
description: Getter/setter inheritance and overriding in class chains
esid: sec-class-definitions
---*/
class Shape {
  constructor() { this._size = 1; }
  get area() { return this._size; }
  set area(v) { this._size = v; }
}
class Square extends Shape {
  get area() { return this._size * this._size; }
}
var sq = new Square();
sq._size = 4;
assert.sameValue(sq.area, 16, "overridden getter");
var sh = new Shape();
sh.area = 5;
assert.sameValue(sh.area, 5, "base getter/setter");
class Circle extends Shape {
  get area() { return Math.floor(3.14 * this._size * this._size); }
  set radius(r) { this._size = r; }
}
var c = new Circle();
c.radius = 2;
assert.sameValue(c.area, 12, "new setter plus overridden getter");
assert.sameValue(c._size, 2);
