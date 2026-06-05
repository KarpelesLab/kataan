/*---
description: BigInt arithmetic, comparison, and conversions
esid: sec-bigint-objects
---*/
assert.sameValue(10n + 20n, 30n);
assert.sameValue(100n * 100n, 10000n);
assert.sameValue(2n ** 64n, 18446744073709551616n, "large power");
assert.sameValue(17n % 5n, 2n);
assert.sameValue(20n / 3n, 6n, "BigInt division truncates");
assert.sameValue(-5n + 3n, -2n);
assert.sameValue(typeof 42n, "bigint");
assert.sameValue(10n > 5n, true);
assert.sameValue(10n === 10n, true);
assert.sameValue(10n == 10, true, "loose equality with number");
assert.sameValue(10n === 10, false, "strict inequality with number");
assert.sameValue(BigInt(42), 42n);
assert.sameValue(BigInt("123456789012345678901234567890") + 1n, 123456789012345678901234567891n, "huge from string");
assert.sameValue(Number(100n), 100);
assert.sameValue((123n).toString(), "123");
assert.sameValue((255n).toString(16), "ff", "BigInt radix");
assert.sameValue(5n * 5n + 2n, 27n);
