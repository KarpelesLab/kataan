/*---
description: instanceof on a bound function tests against its [[BoundTargetFunction]]
esid: sec-bound-function-exotic-objects
---*/
function H() {}
var BH = H.bind(null);
var h = new BH();
assert.sameValue(h instanceof H, true, "new BH() is an H");
assert.sameValue(h instanceof BH, true, "and an instance of the bound function");

// Bound with partial arguments.
function P(a, b) { this.sum = a + b; }
var BP = P.bind(null, 10);
var p = new BP(5);
assert.sameValue(p.sum, 15, "partial application");
assert.sameValue(p instanceof P, true, "instance of target");
assert.sameValue(p instanceof BP, true, "instance of bound");

// Double bind resolves through both layers.
var BB = H.bind(null).bind(null);
assert.sameValue(new BB() instanceof H, true, "double-bound");

// A bound class.
class C {}
assert.sameValue(new (C.bind(null))() instanceof C, true, "bound class");

// An unrelated bound function is not matched.
function Q() {}
assert.sameValue(new H() instanceof Q.bind(null), false, "unrelated bound function");
