/*---
description: BigInt arithmetic, comparison, and conversion
esid: sec-ecmascript-language-types-bigint-type
---*/
assert.sameValue(10n + 20n, 30n);
assert.sameValue(100n * 100n, 10000n);
assert.sameValue(2n ** 10n, 1024n);
assert.sameValue(typeof 5n, "bigint");
assert.sameValue(10n > 5n, true);
assert.sameValue(10n === 10n, true);
assert.sameValue(BigInt(42), 42n);
assert.sameValue(7n % 3n, 1n);
assert.sameValue(-5n + 3n, -2n);
assert.sameValue(String(123n), "123");
