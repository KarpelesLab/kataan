/*---
description: a strict-mode delete of a non-configurable property throws TypeError
esid: sec-delete-operator-runtime-semantics-evaluation
---*/
// Each delete runs inside a per-function "use strict" (the corpus harness is prepended, so a
// top-level directive would not apply).
function throwsTypeStrict(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// Deleting a non-configurable own property throws.
var o = {};
Object.defineProperty(o, "x", { value: 1, configurable: false });
assert.sameValue(throwsTypeStrict(function () { "use strict"; delete o.x; }), true, "non-configurable property");
assert.sameValue("x" in o, true, "property not removed");

// A frozen object's property is non-configurable.
var f = Object.freeze({ a: 1 });
assert.sameValue(throwsTypeStrict(function () { "use strict"; delete f.a; }), true, "frozen property");

// An array's length is non-configurable.
assert.sameValue(throwsTypeStrict(function () { "use strict"; var arr = [1, 2, 3]; delete arr.length; }), true, "array length");

// A configurable property deletes successfully (returns true, no throw).
var c = {};
Object.defineProperty(c, "y", { value: 1, configurable: true });
assert.sameValue((function () { "use strict"; return delete c.y; })(), true, "configurable delete returns true");
assert.sameValue("y" in c, false, "configurable property removed");

// Deleting a missing property is true; an ordinary property deletes fine.
assert.sameValue((function () { "use strict"; return delete o.notThere; })(), true, "missing property -> true");
assert.sameValue((function () { "use strict"; var n = { a: 1 }; var r = delete n.a; return r && !("a" in n); })(), true, "ordinary property delete");
