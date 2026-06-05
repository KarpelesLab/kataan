/*---
description: BigInt is arbitrary precision (beyond 64/128-bit)
esid: sec-ecmascript-language-types-bigint-type
---*/
// 2**200 far exceeds i128 (2**127).
assert.sameValue((2n ** 200n).toString(), "1606938044258990275541962092341162602522202993782792835301376");
// 50! is a large factorial.
var f = 1n;
for (var i = 1n; i <= 50n; i++) f *= i;
assert.sameValue(f.toString(), "30414093201713378043612608166064768844377641568960512000000000000");
assert.sameValue((10n ** 40n) + 1n, 10000000000000000000000000000000000000001n);
assert.sameValue((2n ** 128n) - 1n, 340282366920938463463374607431768211455n);
assert.sameValue((-(2n ** 100n)).toString(), "-1267650600228229401496703205376");
assert.sameValue((10n ** 30n) % 7n, 1n);
assert.sameValue((2n ** 256n) > (2n ** 255n), true);
// Two's-complement bitwise at arbitrary precision.
assert.sameValue(12n & 10n, 8n);
assert.sameValue(12n | 10n, 14n);
assert.sameValue(12n ^ 10n, 6n);
assert.sameValue(-1n & 255n, 255n, "-1n is all ones");
assert.sameValue(((2n ** 100n) | 1n) - (2n ** 100n), 1n, "bitor beyond i128");
