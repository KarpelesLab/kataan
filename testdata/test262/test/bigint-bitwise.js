/*---
description: BigInt bitwise and shift operations
esid: sec-bigint-objects
---*/
assert.sameValue(5n & 3n, 1n);
assert.sameValue(5n | 3n, 7n);
assert.sameValue(5n ^ 3n, 6n);
assert.sameValue(~5n, -6n);
assert.sameValue(1n << 8n, 256n);
assert.sameValue(256n >> 2n, 64n);
assert.sameValue(0xFFn & 0x0Fn, 15n);
assert.sameValue(2n ** 32n, 4294967296n);
assert.sameValue(-8n >> 1n, -4n, "arithmetic shift");
assert.sameValue(255n.toString(16), "ff");
assert.sameValue(1024n / 8n, 128n);
assert.sameValue(1000000000000000000000n + 1n, 1000000000000000000001n, "beyond Number range");
assert.sameValue((10n ** 20n) * 2n, 200000000000000000000n);
assert.sameValue(7n % 3n, 1n);
assert.sameValue((-7n) % 3n, -1n, "sign follows dividend");
