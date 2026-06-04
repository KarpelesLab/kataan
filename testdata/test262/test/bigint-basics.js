/*---
description: BigInt literals, arithmetic, typeof, comparison, and coercion
esid: sec-bigint-objects
---*/
assert.sameValue(typeof 10n, "bigint");
assert.sameValue(10n + 20n, 30n);
assert.sameValue(100n * 100n, 10000n);
assert.sameValue(2n ** 10n, 1024n);
assert.sameValue(10n / 3n, 3n, "BigInt division truncates");
assert.sameValue(-5n, 0n - 5n);
assert.sameValue(BigInt(42), 42n);
assert.sameValue(10n === 10n, true);
assert.sameValue(10n === 10, false, "BigInt and Number are never strictly equal");
assert.sameValue(10n == 10, true, "loose equality compares numerically");
assert.sameValue(5n < 10n, true);
assert.sameValue(String(255n), "255");
assert.sameValue(!!0n, false);
assert.sameValue(!!7n, true);
