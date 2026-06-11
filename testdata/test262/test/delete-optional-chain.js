/*---
description: delete on an optional-chain member performs the deletion (and short-circuits on a nullish base)
esid: sec-delete-operator
---*/
// delete a?.b removes the property and returns true.
var od = { a: { b: 1 } };
assert.sameValue(delete od?.a?.b, true, "delete a?.b returns true");
assert.sameValue("b" in od.a, false, "b actually removed");

// A nullish base short-circuits the whole delete to a no-op returning true.
var n = null;
assert.sameValue(delete n?.a?.b, true, "nullish base -> no-op true");
var mid = { x: { y: 2 } };
mid.missing = undefined;
assert.sameValue(delete mid?.missing?.z, true, "mid-chain undefined -> no-op true");
assert.sameValue(mid.x.y, 2, "unrelated property untouched");

// Computed and shallow optional deletes.
var oc = { k: { v: 5 } };
var key = "v";
assert.sameValue(delete oc?.k?.[key], true, "computed optional delete");
assert.sameValue("v" in oc.k, false, "computed property removed");
var shallow = { a: 1, b: 2 };
assert.sameValue(delete shallow?.a, true, "shallow optional delete");
assert.sameValue("a" in shallow, false, "a removed");
assert.sameValue("b" in shallow, true, "b kept");

// Deeply nested optional delete.
var deep = { a: { b: { c: { d: 1 } } } };
assert.sameValue(delete deep?.a?.b?.c?.d, true, "deep optional delete");
assert.sameValue("d" in deep.a.b.c, false, "deep property removed");

// A non-configurable property still returns false.
var nc = {};
Object.defineProperty(nc, "x", { value: 1, configurable: false });
var w = { nc: nc };
assert.sameValue(delete w?.nc?.x, false, "non-configurable -> false");
assert.sameValue("x" in nc, true, "non-configurable retained");

// Ordinary delete, variable delete, and array length delete are unaffected.
var r = { a: 1 };
assert.sameValue(delete r.a, true, "ordinary delete");
var v = 5;
assert.sameValue(delete v, false, "delete variable");
var arr = [1, 2, 3];
assert.sameValue(delete arr.length, false, "delete length");
