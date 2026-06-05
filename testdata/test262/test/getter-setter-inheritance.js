/*---
description: Getter/setter inheritance and overriding across class chain
esid: sec-class-definitions
---*/
class Shape {
  constructor() { this._size = 0; }
  get size() { return this._size; }
  set size(v) { this._size = v < 0 ? 0 : v; }
}
class Square extends Shape {
  get area() { return this._size * this._size; }
}
var sq = new Square();
sq.size = 5;
assert.sameValue(sq.size, 5, "inherited getter");
assert.sameValue(sq.area, 25, "own getter using inherited state");
sq.size = -3;
assert.sameValue(sq.size, 0, "inherited setter clamps");
