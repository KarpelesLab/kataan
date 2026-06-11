/*---
description: a strict-mode delete of a non-configurable property throws TypeError
esid: sec-delete-operator-runtime-semantics-evaluation
flags: [onlyStrict]
---*/
"use strict";
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// Deleting a non-configurable own property throws.
var o = {};
Object.defineProperty(o, "x", { value: 1, configurable: false });
assert.sameValue(throwsType(function () { delete o.x; }), true, "non-configurable property");
assert.sameValue("x" in o, true, "property not removed");

// A frozen object's property is non-configurable.
var f = Object.freeze({ a: 1 });
assert.sameValue(throwsType(function () { delete f.a; }), true, "frozen property");

// An array's length is non-configurable.
assert.sameValue(throwsType(function () { var arr = [1, 2, 3]; delete arr.length; }), true, "array length");

// A configurable property deletes successfully (returns true, no throw).
var c = {};
Object.defineProperty(c, "y", { value: 1, configurable: true });
assert.sameValue(delete c.y, true, "configurable delete returns true");
assert.sameValue("y" in c, false, "configurable property removed");

// Deleting a missing property is true; an ordinary property deletes fine.
assert.sameValue(delete o.notThere, true, "missing property -> true");
var n = { a: 1 };
assert.sameValue(delete n.a, true, "ordinary property delete");
assert.sameValue("a" in n, false, "ordinary property removed");
