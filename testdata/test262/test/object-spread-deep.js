/*---
description: Object spread merging, overriding, and nesting
esid: sec-object-initializer
---*/
var defaults = { color: "black", size: "medium", visible: true };
var custom = { color: "red", size: "large" };
var merged = { ...defaults, ...custom };
assert.sameValue(merged.color, "red", "later spread wins");
assert.sameValue(merged.size, "large");
assert.sameValue(merged.visible, true, "kept from defaults");
var withExtra = { ...merged, extra: 1, color: "blue" };
assert.sameValue(withExtra.color, "blue", "explicit after spread");
assert.sameValue(withExtra.extra, 1);
var nested = { a: { x: 1 }, b: 2 };
var shallow = { ...nested };
shallow.b = 99;
assert.sameValue(nested.b, 2, "spread is shallow for top level");
assert.sameValue(shallow.a === nested.a, true, "nested objects shared");
var arr = [1, 2, 3];
var obj = { ...arr };
assert.sameValue(obj[0], 1, "spread array into object");
assert.sameValue(obj[2], 3);
assert.sameValue(Object.keys(obj).join(","), "0,1,2");
