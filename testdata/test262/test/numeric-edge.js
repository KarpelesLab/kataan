/*---
description: Numeric edge cases and special value handling
esid: sec-numbers
---*/
assert.sameValue(0.1 + 0.2 > 0.3, true, "floating point");
assert.sameValue(1 / 0, Infinity);
assert.sameValue(-1 / 0, -Infinity);
assert.sameValue(0 / 0 !== 0 / 0, true, "NaN");
assert.sameValue(Infinity - Infinity !== Infinity - Infinity, true, "Inf-Inf is NaN");
assert.sameValue(Infinity + Infinity, Infinity);
assert.sameValue(Math.max(-0, 0), 0);
assert.sameValue(1e21.toString(), "1e+21", "large number string");
assert.sameValue((0.0000001).toString(), "1e-7", "small number string");
assert.sameValue(Number.MAX_SAFE_INTEGER + 1 === Number.MAX_SAFE_INTEGER + 2, true, "precision loss");
assert.sameValue(parseInt("123abc"), 123);
assert.sameValue(parseFloat("3.14xyz"), 3.14);
assert.sameValue(parseInt("0x1F"), 31, "hex prefix");
assert.sameValue((123.456).toFixed(1), "123.5");
