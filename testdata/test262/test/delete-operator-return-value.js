/*---
description: the delete operator returns the correct boolean for variables, length, and properties
esid: sec-delete-operator
---*/
// Deleting a declared binding (variable or parameter) is a no-op returning false.
var gv = 5;
assert.sameValue(delete gv, false, "delete declared variable");
assert.sameValue(typeof gv, "number", "variable still exists");
function f(p) { return delete p; }
assert.sameValue(f(1), false, "delete parameter");

// Deleting an unresolvable name returns true.
assert.sameValue(delete undeclaredNameXYZ123, true, "delete undeclared name");

// An array's length is non-configurable -> delete returns false (length unchanged).
var a = [1, 2, 3];
assert.sameValue(delete a.length, false, "delete length");
assert.sameValue(a.length, 3, "length unchanged");

// Deleting a (configurable) array element returns true.
assert.sameValue(delete a[1], true, "delete element");

// Ordinary configurable property -> true; missing -> true; non-configurable -> false.
var o = { x: 1 };
assert.sameValue(delete o.x, true, "delete configurable property");
assert.sameValue("x" in o, false, "property removed");
assert.sameValue(delete o.missing, true, "delete missing property");
var nc = {};
Object.defineProperty(nc, "y", { value: 1, configurable: false });
assert.sameValue(delete nc.y, false, "delete non-configurable");
assert.sameValue("y" in nc, true, "non-configurable retained");

// delete of a non-reference expression is true.
assert.sameValue(delete 5, true, "delete non-reference");
assert.sameValue(delete (1 + 1), true, "delete computed value");

// delete this.x inside a method.
var obj = { x: 1, m: function () { return delete this.x; } };
assert.sameValue(obj.m(), true, "delete this.x");
assert.sameValue("x" in obj, false, "this property removed");
