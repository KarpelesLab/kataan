/*---
description: BigInt comparison and mixing with Number
esid: sec-bigint-objects
---*/
assert.sameValue(10n == 10, true, "loose equality with number");
assert.sameValue(10n === 10, false, "strict not equal");
assert.sameValue(10n < 20, true, "BigInt less than Number");
assert.sameValue(20n > 10, true);
assert.sameValue(10n <= 10, true);
assert.sameValue(5n < 5.5, true, "BigInt less than float");
assert.sameValue(BigInt(5) === 5n, true);
assert.sameValue(typeof (10n + 20n), "bigint");
assert.sameValue(100n > 99, true);
assert.sameValue(0n == 0, true);
assert.sameValue(0n === 0, false);
assert.sameValue(-5n < 0, true);
assert.sameValue(2n ** 10n > 1000, true);
assert.sameValue(String(255n), "255");
assert.sameValue(Boolean(0n), false);
assert.sameValue(Boolean(1n), true);
assert.sameValue(10n !== 10, true, "strict inequality types differ");
assert.sameValue(3n * 3n === 9n, true);
