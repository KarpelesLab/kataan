/*---
description: ArrayBuffer.prototype.transfer detaches the source; detached buffers and their views are unusable
esid: sec-arraybuffer.prototype.transfer
features: [arraybuffer-transfer]
---*/
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// transfer() moves the bytes to a new buffer and detaches the original.
var b = new ArrayBuffer(8);
var v = new Uint8Array(b);
v[0] = 42; v[1] = 99;
assert.sameValue(typeof b.transfer, "function", "transfer is readable");
assert.sameValue(b.detached, false, "not detached initially");
var c = b.transfer();
assert.sameValue(b.detached, true, "source detached after transfer");
assert.sameValue(b.byteLength, 0, "detached source byteLength is 0");
assert.sameValue(c.byteLength, 8, "new buffer keeps the size");
assert.sameValue(new Uint8Array(c)[0], 42, "bytes moved to the new buffer");
assert.sameValue(new Uint8Array(c)[1], 99, "bytes moved (2)");

// The source's existing views are emptied.
assert.sameValue(v.length, 0, "existing view length is 0 after detach");
assert.sameValue(v[0], undefined, "detached view element is undefined");

// transfer(newLength) resizes (zero-pad on grow, truncate on shrink).
var g = new ArrayBuffer(4);
new Uint8Array(g)[0] = 7;
var gt = g.transfer(8);
assert.sameValue(gt.byteLength, 8, "grow");
assert.sameValue(new Uint8Array(gt)[0], 7, "grow preserves data");
assert.sameValue(new Uint8Array(gt)[7], 0, "grow zero-pads");
var s = new ArrayBuffer(8);
new Uint8Array(s)[0] = 5;
assert.sameValue(s.transfer(2).byteLength, 2, "shrink");

// Operations on a detached buffer throw TypeError.
assert.sameValue(throwsType(function () { b.transfer(); }), true, "re-transfer throws");
assert.sameValue(throwsType(function () { return new Uint8Array(b); }), true, "view construction throws");
assert.sameValue(throwsType(function () { return new DataView(b); }), true, "DataView construction throws");
assert.sameValue(throwsType(function () { return b.slice(0, 2); }), true, "slice throws");

// A DataView created before the detach throws on later access.
var b2 = new ArrayBuffer(8);
var dv = new DataView(b2);
dv.setUint8(0, 1);
b2.transfer();
assert.sameValue(throwsType(function () { return dv.getUint8(0); }), true, "stale DataView get throws");
assert.sameValue(throwsType(function () { dv.setUint8(0, 2); }), true, "stale DataView set throws");

// A fresh buffer is not detached and works normally.
var fresh = new ArrayBuffer(4);
assert.sameValue(fresh.detached, false, "fresh buffer not detached");
assert.sameValue(new Uint8Array(fresh).length, 4, "fresh buffer usable");
