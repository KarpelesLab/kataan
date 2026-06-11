/*---
description: ArrayBuffer.prototype.slice is a readable method that copies the byte range
esid: sec-arraybuffer.prototype.slice
features: [ArrayBuffer]
---*/
var ab = new ArrayBuffer(8);
var v = new Uint8Array(ab);
v[0] = 42; v[1] = 99; v[7] = 7;

// Readable for feature detection, and callable detached (via .call / .apply).
assert.sameValue(typeof ab.slice, "function", "slice is readable");
var detached = ab.slice;
assert.sameValue(detached.call(ab, 0, 2).byteLength, 2, "detached .call");
assert.sameValue(ab.slice.apply(ab, [0, 1]).byteLength, 1, ".apply");

// Direct call copies the requested range into a new ArrayBuffer.
var s = ab.slice(0, 4);
assert.sameValue(s instanceof ArrayBuffer, true, "result is an ArrayBuffer");
assert.sameValue(s.byteLength, 4, "sliced length");
var sv = new Uint8Array(s);
assert.sameValue(sv[0], 42, "byte 0 copied");
assert.sameValue(sv[1], 99, "byte 1 copied");

// Negative and default arguments (like Array.prototype.slice's relative indices).
assert.sameValue(ab.slice(-2).byteLength, 2, "negative begin");
assert.sameValue(ab.slice().byteLength, 8, "no args -> whole buffer");
assert.sameValue(ab.slice(2).byteLength, 6, "begin only");
assert.sameValue(ab.slice(2, 100).byteLength, 6, "end clamped to length");
assert.sameValue(ab.slice(5, 2).byteLength, 0, "begin > end -> empty");

// The slice is an independent copy, not a view.
var src = new ArrayBuffer(4);
new Uint8Array(src)[0] = 5;
var copy = src.slice(0);
new Uint8Array(copy)[0] = 77;
assert.sameValue(new Uint8Array(src)[0], 5, "slice does not alias the source");
