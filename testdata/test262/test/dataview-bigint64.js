/*---
description: DataView get/setBigInt64 and get/setBigUint64
esid: sec-dataview.prototype.setbigint64
features: [DataView, BigInt]
---*/
var b = new ArrayBuffer(16);
var dv = new DataView(b);

dv.setBigInt64(0, 123n);
assert.sameValue(dv.getBigInt64(0), 123n, "round-trip positive");
assert.sameValue(typeof dv.getBigInt64(0), "bigint", "returns a BigInt");

dv.setBigInt64(0, -456n);
assert.sameValue(dv.getBigInt64(0), -456n, "round-trip negative");

// -1 is all-ones: signed -> -1, unsigned -> u64 max.
dv.setBigInt64(0, -1n);
assert.sameValue(dv.getBigInt64(0), -1n, "signed -1");
assert.sameValue(dv.getBigUint64(0), 18446744073709551615n, "unsigned u64 max");

dv.setBigUint64(8, 18446744073709551615n);
assert.sameValue(dv.getBigUint64(8), 18446744073709551615n, "BigUint64 round-trip");

// Little-endian.
dv.setBigInt64(0, 0x0123456789abcdefn, true);
assert.sameValue(dv.getBigInt64(0, true), 0x0123456789abcdefn, "little-endian round-trip");
// Read back big-endian: the byte order is reversed.
assert.sameValue(dv.getUint8(0), 0xef, "LE stored low byte first");

// A value beyond Number's safe integer range survives.
dv.setBigInt64(0, 9007199254740993n);
assert.sameValue(dv.getBigInt64(0), 9007199254740993n, "beyond 2^53");
