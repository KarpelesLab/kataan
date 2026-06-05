/*---
description: Getters inherited via Object.create prototype chains
esid: sec-ordinary-object-internal-methods
---*/
var base = {
  _x: 10,
  get doubled() { return this._x * 2; },
  describe: function () { return "x=" + this._x; }
};
var derived = Object.create(base);
derived._x = 20;
assert.sameValue(derived.doubled, 40, "inherited getter uses own state");
assert.sameValue(derived.describe(), "x=20", "inherited method");
assert.sameValue(base.doubled, 20, "base unaffected");
var grandchild = Object.create(derived);
grandchild._x = 5;
assert.sameValue(grandchild.doubled, 10, "two-level inherited getter");
assert.sameValue(Object.getPrototypeOf(grandchild), derived);
assert.sameValue(grandchild.hasOwnProperty("doubled"), false, "getter is inherited not own");
