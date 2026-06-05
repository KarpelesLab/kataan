/*---
description: delete respects configurability and returns the correct result
esid: sec-delete-operator
---*/
var o = { a: 1 };
assert.sameValue(delete o.a, true, "configurable own property");
assert.sameValue("a" in o, false, "removed");
assert.sameValue(delete o.missing, true, "deleting a missing property succeeds");
var nc = {};
Object.defineProperty(nc, "x", { value: 1, configurable: false });
assert.sameValue(delete nc.x, false, "non-configurable cannot be deleted");
assert.sameValue(nc.x, 1, "non-configurable value retained");
var c = {};
Object.defineProperty(c, "y", { value: 2, configurable: true });
assert.sameValue(delete c.y, true, "configurable: true can be deleted");
assert.sameValue("y" in c, false);
var frozen = Object.freeze({ a: 1 });
assert.sameValue(delete frozen.a, false, "frozen object property");
assert.sameValue(frozen.a, 1);
var sealed = Object.seal({ b: 2 });
assert.sameValue(delete sealed.b, false, "sealed object property");
var arr = [1, 2, 3];
assert.sameValue(delete arr[1], true, "array element delete");
assert.sameValue(arr[1], undefined, "element cleared");
