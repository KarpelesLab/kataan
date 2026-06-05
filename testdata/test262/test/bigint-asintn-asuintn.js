/*---
description: BigInt.asUintN and BigInt.asIntN wrap to N bits
esid: sec-bigint.asuintn
---*/
// asUintN: non-negative remainder modulo 2^bits.
assert.sameValue(BigInt.asUintN(8, 256n), 0n, "256 mod 256 = 0");
assert.sameValue(BigInt.asUintN(8, 255n), 255n, "255 fits");
assert.sameValue(BigInt.asUintN(8, 257n), 1n, "257 -> 1");
assert.sameValue(BigInt.asUintN(8, -1n), 255n, "-1 -> 255");
assert.sameValue(BigInt.asUintN(4, -1n), 15n, "-1 mod 16 = 15");
assert.sameValue(BigInt.asUintN(0, 12345n), 0n, "0 bits -> 0");
assert.sameValue(BigInt.asUintN(64, 18446744073709551617n), 1n, "wraps at 64 bits");
// asIntN: signed two's-complement interpretation of the low bits.
assert.sameValue(BigInt.asIntN(8, 200n), -56n, "200 -> -56");
assert.sameValue(BigInt.asIntN(8, 127n), 127n, "127 stays positive");
assert.sameValue(BigInt.asIntN(8, 128n), -128n, "128 -> -128");
assert.sameValue(BigInt.asIntN(8, 255n), -1n, "255 -> -1");
assert.sameValue(BigInt.asIntN(16, 40000n), -25536n, "16-bit wrap");
assert.sameValue(BigInt.asIntN(32, 4294967295n), -1n, "32-bit -1");
assert.sameValue(BigInt.asIntN(8, -56n), -56n, "negative kept in range");
// A large width keeps the value.
assert.sameValue(BigInt.asIntN(128, 12345678901234567890n), 12345678901234567890n, "wide enough");
