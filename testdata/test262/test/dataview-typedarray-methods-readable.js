/*---
description: DataView get*/set* and TypedArray set/subarray are readable methods (feature detection + detached calls)
features: [DataView, TypedArray]
---*/
var dv = new DataView(new ArrayBuffer(16));

// All the accessor methods are readable functions.
var names = ["getInt8", "getUint8", "getInt16", "getUint16", "getInt32", "getUint32",
  "getFloat32", "getFloat64", "getBigInt64", "getBigUint64",
  "setInt8", "setUint8", "setInt16", "setUint16", "setInt32", "setUint32",
  "setFloat32", "setFloat64", "setBigInt64", "setBigUint64"];
for (var i = 0; i < names.length; i++) {
  assert.sameValue(typeof dv[names[i]], "function", names[i] + " is readable");
}

// Direct and detached calls produce the right values.
dv.setUint8(0, 42);
assert.sameValue(dv.getUint8(0), 42, "direct get/set");
var g = dv.getUint8;
assert.sameValue(g.call(dv, 0), 42, "detached getUint8.call");
dv.setInt16(2, -1000, true);
assert.sameValue(dv.getInt16(2, true), -1000, "int16 little-endian");
dv.setFloat64(8, 3.14);
assert.sameValue(dv.getFloat64(8), 3.14, "float64 round-trip");

// TypedArray.prototype.set / subarray (not shared with Array.prototype) are readable.
var ta = new Uint8Array(4);
assert.sameValue(typeof ta.set, "function", "TypedArray set readable");
assert.sameValue(typeof ta.subarray, "function", "TypedArray subarray readable");
ta.set([1, 2, 3]);
assert.sameValue(ta.join(","), "1,2,3,0", "set works");
assert.sameValue(ta.subarray(1, 3).join(","), "2,3", "subarray works");
var setFn = ta.set;
setFn.call(ta, [9, 9], 0);
assert.sameValue(ta.join(","), "9,9,3,0", "detached set.call");

// Array-shared typed-array methods and ArrayBuffer.slice remain readable.
assert.sameValue(typeof ta.map, "function", "shared map still readable");
assert.sameValue(typeof new ArrayBuffer(4).slice, "function", "ArrayBuffer slice still readable");
