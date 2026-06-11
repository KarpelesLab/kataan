/*---
description: resizable ArrayBuffers (new ArrayBuffer(n, {maxByteLength}), resizable, maxByteLength, resize)
features: [resizable-arraybuffer]
---*/
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }
function throwsRange(fn) { try { fn(); return false; } catch (e) { return e instanceof RangeError; } }

// A buffer constructed with maxByteLength is resizable.
var b = new ArrayBuffer(4, { maxByteLength: 16 });
assert.sameValue(b.resizable, true, "resizable");
assert.sameValue(b.maxByteLength, 16, "maxByteLength");
assert.sameValue(b.byteLength, 4, "initial byteLength");
assert.sameValue(typeof b.resize, "function", "resize is readable");

// resize grows (zero-filling) and length-tracking views follow.
var v = new Uint8Array(b);
v[0] = 42; v[1] = 99;
b.resize(8);
assert.sameValue(b.byteLength, 8, "grown byteLength");
assert.sameValue(v.length, 8, "view tracks growth");
assert.sameValue(v[0], 42, "data preserved on grow");
assert.sameValue(v[1], 99, "data preserved (2)");
assert.sameValue(v[7], 0, "grown bytes are zero");

// resize shrinks and the view follows.
b.resize(2);
assert.sameValue(b.byteLength, 2, "shrunk byteLength");
assert.sameValue(v.length, 2, "view tracks shrink");
assert.sameValue(v[0], 42, "low data kept on shrink");

// A multi-byte view tracks resize (length in elements, value little-endian preserved).
var b2 = new ArrayBuffer(8, { maxByteLength: 16 });
var u32 = new Uint32Array(b2);
u32[0] = 0x04030201;
b2.resize(16);
assert.sameValue(u32.length, 4, "u32 view length in elements after resize");
assert.sameValue(u32[0], 0x04030201, "u32 value preserved across resize");

// A non-resizable buffer reports resizable:false, maxByteLength == byteLength, and resize throws.
var nr = new ArrayBuffer(8);
assert.sameValue(nr.resizable, false, "non-resizable");
assert.sameValue(nr.maxByteLength, 8, "non-resizable maxByteLength is its length");
assert.sameValue(throwsType(function () { nr.resize(4); }), true, "resize on non-resizable -> TypeError");

// Bounds: resize beyond maxByteLength, and a max smaller than the initial length, are RangeErrors.
assert.sameValue(throwsRange(function () { b.resize(99); }), true, "resize beyond max -> RangeError");
assert.sameValue(throwsRange(function () { return new ArrayBuffer(8, { maxByteLength: 4 }); }), true, "max < length -> RangeError");
