/*---
description: typed-array mutating methods (fill/copyWithin/sort/reverse) coerce and write through to the buffer
esid: sec-%typedarray%.prototype.fill
features: [TypedArray, DataView]
---*/
// fill coerces to the element type (previously stored raw).
assert.sameValue(new Uint8Array(3).fill(256)[0], 0, "fill coerces Uint8 256 -> 0");
assert.sameValue(new Int8Array(2).fill(200)[0], -56, "fill coerces Int8 200 -> -56");

// fill writes through to the backing buffer.
var b = new ArrayBuffer(8);
var ta = new Int32Array(b);
ta.fill(7);
assert.sameValue(new DataView(b).getInt32(0, true), 7, "fill[0] in buffer");
assert.sameValue(new DataView(b).getInt32(4, true), 7, "fill[1] in buffer");
assert.sameValue(new Uint8Array([0, 0, 0, 0]).fill(5, 1, 3).join(","), "0,5,5,0", "fill range");

// copyWithin writes through.
var cb = new ArrayBuffer(16);
var c = new Int32Array(cb);
c[0] = 10; c[1] = 20;
c.copyWithin(2, 0, 2);
assert.sameValue(new DataView(cb).getInt32(8, true), 10, "copyWithin dst0");
assert.sameValue(new DataView(cb).getInt32(12, true), 20, "copyWithin dst1");

// sort and reverse resync the whole buffer.
var sb = new ArrayBuffer(12);
var s = new Int32Array(sb);
s[0] = 3; s[1] = 1; s[2] = 2;
s.sort();
assert.sameValue(s.join(","), "1,2,3", "sorted store");
assert.sameValue(new DataView(sb).getInt32(0, true) + "," + new DataView(sb).getInt32(8, true), "1,3", "sorted buffer");

var rb = new ArrayBuffer(8);
var r = new Int32Array(rb);
r[0] = 1; r[1] = 2;
r.reverse();
assert.sameValue(new DataView(rb).getInt32(0, true), 2, "reversed buffer[0]");

// A comparator sort also resyncs.
var cs = new Int32Array(new ArrayBuffer(12));
cs.set([3, 1, 2]);
cs.sort(function (a, b) { return b - a; });
assert.sameValue(new DataView(cs.buffer).getInt32(0, true), 3, "comparator-sorted buffer");

// Ordinary arrays and unbacked typed arrays are unaffected.
assert.sameValue([3, 1, 2].sort().join(","), "1,2,3", "plain array sort");
assert.sameValue([1, 2, 3].fill(9).join(","), "9,9,9", "plain array fill");
assert.sameValue(new Int32Array([3, 1, 2]).sort().join(","), "1,2,3", "unbacked sort");
