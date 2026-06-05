/*---
description: Number toString(radix) with fractions, toLocaleString, toPrecision exponent
esid: sec-number.prototype.tostring
---*/
assert.sameValue((3.5).toString(2), "11.1", "fractional binary");
assert.sameValue((255.5).toString(16), "ff.8", "fractional hex");
assert.sameValue((0.5).toString(2), "0.1");
assert.sameValue((10.25).toString(2), "1010.01");
assert.sameValue((255).toString(16), "ff", "integer hex unaffected");
assert.sameValue((-255.5).toString(16), "-ff.8", "negative fractional");
assert.sameValue((12345).toPrecision(1), "1e+4", "toPrecision exponent has plus sign");
assert.sameValue((12345).toPrecision(2), "1.2e+4");
assert.sameValue((0.0000001234).toPrecision(2), "1.2e-7", "negative exponent below -6");
assert.sameValue((123.456).toPrecision(4), "123.5", "non-exponential precision");
assert.sameValue((12345).toExponential(2), "1.23e+4", "toExponential plus sign");
assert.sameValue(typeof (1234.5).toLocaleString(), "string");
assert.sameValue((1234567).toLocaleString(), "1,234,567", "thousands grouping");
assert.sameValue((1234.5).toLocaleString(), "1,234.5", "grouping with fraction");
assert.sameValue((100).toLocaleString(), "100", "no grouping under 1000");
assert.sameValue((-1234567).toLocaleString(), "-1,234,567", "negative grouping");
assert.sameValue((42).toString(), "42", "default base 10");
