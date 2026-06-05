/*---
description: Writing a getter-only accessor throws in strict mode, is silent in sloppy
esid: sec-putvalue
---*/
// Sloppy mode: the assignment is silently dropped.
(function () {
  var o = { get x() { return 1; } };
  o.x = 2;
  assert.sameValue(o.x, 1, "sloppy write to a getter-only property is ignored");
})();
// Strict mode: it throws a TypeError.
(function () {
  "use strict";
  var o = { get x() { return 1; } };
  var threw = false;
  try { o.x = 2; } catch (e) { threw = e instanceof TypeError; }
  assert.sameValue(threw, true, "strict write to a getter-only property throws");
})();
// A getter+setter pair still works in strict mode.
(function () {
  "use strict";
  var o = { _v: 0, get x() { return this._v; }, set x(v) { this._v = v * 2; } };
  o.x = 5;
  assert.sameValue(o.x, 10, "setter runs in strict mode");
})();
// An inherited getter-only accessor also throws in strict mode.
(function () {
  "use strict";
  var proto = { get y() { return 9; } };
  var o = Object.create(proto);
  var threw = false;
  try { o.y = 1; } catch (e) { threw = e instanceof TypeError; }
  assert.sameValue(threw, true, "strict write to an inherited getter-only accessor throws");
})();
