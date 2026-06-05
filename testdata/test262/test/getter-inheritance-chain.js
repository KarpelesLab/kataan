/*---
description: Getter/setter through long prototype chains
esid: sec-property-accessors
---*/
var grandparent = { get level() { return 1; } };
var parent = Object.create(grandparent);
var child = Object.create(parent);
assert.sameValue(child.level, 1, "inherited getter through two levels");
var base = {
  _data: { count: 0 },
  get count() { return this._data.count; },
  set count(v) { this._data.count = v; }
};
var derived = Object.create(base);
derived._data = { count: 5 };
assert.sameValue(derived.count, 5, "inherited accessor uses own state");
derived.count = 10;
assert.sameValue(derived._data.count, 10, "inherited setter");
assert.sameValue(base._data.count, 0, "base unaffected");
class Shape {
  get area() { return 0; }
}
class Rectangle extends Shape {
  constructor(w, h) { super(); this.w = w; this.h = h; }
  get area() { return this.w * this.h; }
}
class Square extends Rectangle {
  constructor(s) { super(s, s); }
}
assert.sameValue(new Square(4).area, 16, "getter override through chain");
var obj = {};
Object.defineProperty(obj, "x", { get: function () { return 42; } });
var inherited = Object.create(obj);
assert.sameValue(inherited.x, 42, "defineProperty getter inherited");
