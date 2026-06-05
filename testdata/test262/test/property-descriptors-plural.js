/*---
description: getOwnPropertyDescriptors and property attribute inspection
esid: sec-object.getownpropertydescriptors
---*/
var o = { a: 1 };
Object.defineProperty(o, "b", { value: 2, writable: false, enumerable: true });
var descs = Object.getOwnPropertyDescriptors(o);
assert.sameValue(descs.a.value, 1);
assert.sameValue(descs.a.writable, true);
assert.sameValue(descs.b.value, 2);
assert.sameValue(descs.b.writable, false);
assert.sameValue(Object.keys(descs).join(","), "a,b");
var single = Object.getOwnPropertyDescriptor(o, "a");
assert.sameValue(single.value, 1);
assert.sameValue(single.enumerable, true);
assert.sameValue(Object.getOwnPropertyDescriptor(o, "missing"), undefined);
