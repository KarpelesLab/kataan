/*---
description: writes to a buffer-backed typed array flow through to the ArrayBuffer
esid: sec-integerindexedelementset
features: [TypedArray, DataView]
---*/
// A typed-array element write is visible through a DataView over the same buffer.
var b = new ArrayBuffer(8);
var ta = new Int32Array(b);
ta[0] = 42;
ta[1] = 99;
var dv = new DataView(b);
assert.sameValue(dv.getInt32(0, true), 42, "element 0 written through");
assert.sameValue(dv.getInt32(4, true), 99, "element 1 written through");

// byteOffset is honored.
var b2 = new ArrayBuffer(16);
var v = new Int32Array(b2, 4);
v[0] = 77;
assert.sameValue(new DataView(b2).getInt32(4, true), 77, "write-through at byteOffset");

// Coercion happens before encoding (Uint8 wraps, Int8 sign).
var ub = new ArrayBuffer(2);
var u = new Uint8Array(ub);
u[0] = 256;
assert.sameValue(new DataView(ub).getUint8(0), 0, "256 -> 0 written through");
assert.sameValue(u[0], 0, "read back coerced");
var sb = new ArrayBuffer(2);
var s = new Int8Array(sb);
s[0] = -5;
assert.sameValue(new DataView(sb).getInt8(0), -5, "signed write-through");

// Float encoding round-trips.
var fb = new ArrayBuffer(8);
var f = new Float64Array(fb);
f[0] = 3.5;
assert.sameValue(new DataView(fb).getFloat64(0, true), 3.5, "float write-through");

// A materialized .buffer stays in sync with later writes.
var arr = new Uint8Array([1, 2, 3]);
var buf = arr.buffer;
arr[0] = 99;
assert.sameValue(new DataView(buf).getUint8(0), 99, "materialized buffer tracks writes");

// Reads, length, and unbacked typed arrays are unchanged.
assert.sameValue(ta[0], 42, "indexed read");
assert.sameValue(ta.length, 2, "length");
var plain = new Uint8Array([1, 2, 3]);
plain[0] = 9;
assert.sameValue(plain.join(","), "9,2,3", "unbacked write");
