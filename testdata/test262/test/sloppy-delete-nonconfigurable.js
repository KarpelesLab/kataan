/*---
description: a sloppy-mode delete of a non-configurable property returns false (no throw)
esid: sec-delete-operator-runtime-semantics-evaluation
flags: [noStrict]
---*/
var o = {};
Object.defineProperty(o, "x", { value: 1, configurable: false });
assert.sameValue(delete o.x, false, "non-configurable -> false, no throw");
assert.sameValue("x" in o, true, "property not removed");
assert.sameValue(delete [1, 2, 3].length, false, "array length -> false");
assert.sameValue(delete Object.freeze({ a: 1 }).a, false, "frozen -> false");
