/*---
description: typed-array views over a shared ArrayBuffer observe each other's (and a DataView's) writes
features: [TypedArray, DataView, ArrayBuffer]
---*/
// Two Uint8Array views over one buffer see each other's writes (both indexed and via methods).
var buf = new ArrayBuffer(8);
var a = new Uint8Array(buf);
var b = new Uint8Array(buf);
a[0] = 42;
assert.sameValue(b[0], 42, "sibling indexed read");
assert.sameValue(b.join(","), "42,0,0,0,0,0,0,0", "sibling method read");
b[7] = 9;
assert.sameValue(a[7], 9, "write through the other view");

// A DataView write is visible to a typed-array view.
var dv = new DataView(buf);
dv.setUint8(1, 99);
assert.sameValue(a[1], 99, "DataView write -> typed array indexed read");
assert.sameValue([...a].join(","), "42,99,0,0,0,0,0,9", "DataView write -> spread");

// And a typed-array write is visible through the DataView (the write direction).
a[2] = 7;
assert.sameValue(dv.getUint8(2), 7, "typed array write -> DataView read");

// Different element types over the same buffer share the bytes (little-endian).
var buf2 = new ArrayBuffer(4);
var u32 = new Uint32Array(buf2);
var u8 = new Uint8Array(buf2);
u32[0] = 0x04030201;
assert.sameValue(u8[0], 1, "u32 write -> u8[0] (LE)");
assert.sameValue(u8[1], 2, "u8[1]");
assert.sameValue(u8[2], 3, "u8[2]");
assert.sameValue(u8[3], 4, "u8[3]");
assert.sameValue(u8.reduce(function (s, x) { return s + x; }, 0), 10, "u8 method sees the bytes");

// A view's own .buffer is shared with a later sibling created over it.
var st = new Uint8Array(4);
st[0] = 1;
var shared = st.buffer;
var v = new Uint8Array(shared);
v[1] = 88;
assert.sameValue(st[1], 88, "write via sibling over the materialized buffer");
new DataView(shared).setUint8(2, 55);
assert.sameValue(st[2], 55, "DataView over the materialized buffer");

// An ordinary array and a standalone typed array are completely unaffected.
var ra = [1, 2, 3];
ra[1] = 99;
assert.sameValue(ra.join(","), "1,99,3", "ordinary array");
var lone = new Int16Array(3);
lone[0] = 1000;
assert.sameValue(lone.join(","), "1000,0,0", "standalone typed array");

// subarray returns a VIEW sharing the parent's buffer (not a copy).
var pbuf = new ArrayBuffer(8);
var base = new Uint8Array(pbuf);
base[1] = 10; base[2] = 20; base[3] = 30;
var sub = base.subarray(1, 4);
assert.sameValue(sub.join(","), "10,20,30", "subarray initial elements");
assert.sameValue(sub.length, 3, "subarray length");
assert.sameValue(sub.buffer, base.buffer, "subarray shares the parent's buffer");
base[1] = 99;
assert.sameValue(sub[0], 99, "parent write -> subarray (offset-aligned)");
sub[2] = 77;
assert.sameValue(base[3], 77, "subarray write -> parent");
// subarray of a standalone typed array materializes and shares a buffer.
var lone = new Uint8Array(4);
lone[0] = 1; lone[1] = 2;
var s2 = lone.subarray(1, 3);
lone[1] = 88;
assert.sameValue(s2[0], 88, "standalone parent write -> subarray");
s2[1] = 66;
assert.sameValue(lone[2], 66, "subarray write -> standalone parent");
// A DataView write is seen by a subarray at the right offset.
new DataView(pbuf).setUint8(1, 123);
assert.sameValue(sub[0], 123, "DataView write -> subarray");

// The view registry stays correct under churn: many transient views created over one buffer,
// then a write still reaches the live siblings.
var cbuf = new ArrayBuffer(4);
var keepA = new Uint8Array(cbuf);
var keepB = new Uint8Array(cbuf);
for (var i = 0; i < 64; i++) { var transient = new Uint8Array(cbuf); transient[0] = i; }
keepA[1] = 200;
assert.sameValue(keepB[1], 200, "sibling write still propagates after view churn");
assert.sameValue(keepA[0], 63, "last transient write is visible to live siblings");
