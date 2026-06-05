/*---
description: Property descriptors, writable, enumerable, getOwnPropertyNames
esid: sec-object.getownpropertydescriptor
---*/
var o = {};
Object.defineProperty(o, "ro", { value: 1, writable: false, enumerable: true });
o.ro = 99;
assert.sameValue(o.ro, 1, "non-writable ignores assignment");
var d = Object.getOwnPropertyDescriptor(o, "ro");
assert.sameValue(d.value, 1);
assert.sameValue(d.writable, false);
var o2 = { a: 1, b: 2 };
Object.defineProperty(o2, "hidden", { value: 3, enumerable: false });
assert.sameValue(Object.keys(o2).join(","), "a,b", "non-enumerable excluded from keys");
assert.sameValue(o2.hidden, 3, "but still readable");
assert.sameValue(Object.getOwnPropertyNames(o2).indexOf("hidden") >= 0, true, "getOwnPropertyNames includes it");
