/*---
description: generator .throw() propagates the value when the yield is not guarded
features: [generators]
---*/
// A generator that does not catch at the suspended yield: throw() rethrows to the
// caller and leaves the generator done.
function* g() { yield 1; yield 2; }
var it = g();
assert.sameValue(it.next().value, 1, "first next");
var threw = false;
try { it.throw(new Error("boom")); } catch (e) { threw = e.message === "boom"; }
assert.sameValue(threw, true, "throw rethrows to the caller");
assert.sameValue(it.next().done, true, "generator is done after throw");

// throw() on an already-exhausted generator also rethrows.
var it2 = g();
it2.next(); it2.next(); it2.next();
var threw2 = false;
try { it2.throw("x"); } catch (e) { threw2 = e === "x"; }
assert.sameValue(threw2, true, "throw on a done generator rethrows");
