/*---
description: array indices and length are own properties (hasOwnProperty/propertyIsEnumerable/getOwnPropertyDescriptor)
esid: sec-array-exotic-objects
---*/
var a = [10, 20, 30];

// hasOwnProperty recognizes in-range indices and length.
assert.sameValue(a.hasOwnProperty("0"), true, "index 0 own");
assert.sameValue(a.hasOwnProperty("2"), true, "index 2 own");
assert.sameValue(a.hasOwnProperty("3"), false, "out-of-range not own");
assert.sameValue(a.hasOwnProperty("length"), true, "length own");

// propertyIsEnumerable: indices enumerable, length not, missing false.
assert.sameValue(a.propertyIsEnumerable("0"), true, "index enumerable");
assert.sameValue(a.propertyIsEnumerable("length"), false, "length not enumerable");
assert.sameValue(a.propertyIsEnumerable("5"), false, "missing index");

// getOwnPropertyDescriptor for an index and for length.
var d0 = Object.getOwnPropertyDescriptor(a, "0");
assert.sameValue(d0.value, 10, "index value");
assert.sameValue(d0.writable, true, "index writable");
assert.sameValue(d0.enumerable, true, "index enumerable");
assert.sameValue(d0.configurable, true, "index configurable");
var dl = Object.getOwnPropertyDescriptor(a, "length");
assert.sameValue(dl.value, 3, "length value");
assert.sameValue(dl.writable, true, "length writable");
assert.sameValue(dl.enumerable, false, "length non-enumerable");
assert.sameValue(dl.configurable, false, "length non-configurable");
assert.sameValue(Object.getOwnPropertyDescriptor(a, "5"), undefined, "missing index has no descriptor");

// A custom named property on an array is an enumerable own property.
a.custom = "c";
assert.sameValue(a.hasOwnProperty("custom"), true, "custom own");
assert.sameValue(a.propertyIsEnumerable("custom"), true, "custom enumerable");
assert.sameValue(Object.getOwnPropertyDescriptor(a, "custom").value, "c", "custom value");

// Plain objects unaffected.
var o = { x: 1 };
Object.defineProperty(o, "h", { value: 2, enumerable: false });
assert.sameValue(o.hasOwnProperty("x"), true, "object own");
assert.sameValue(o.propertyIsEnumerable("x"), true, "object enumerable");
assert.sameValue(o.propertyIsEnumerable("h"), false, "object non-enumerable");
assert.sameValue(Object.keys(a).join(","), "0,1,2,custom", "array keys");
