/*---
description: Number.MIN_VALUE is the smallest positive subnormal (5e-324)
esid: sec-number.min_value
---*/
// MIN_VALUE is the least representable positive value — a subnormal, 5e-324,
// not the smallest *normal* double (2.2250738585072014e-308).
assert.sameValue(Number.MIN_VALUE, 5e-324, "MIN_VALUE is 5e-324");
assert.sameValue(Number.MIN_VALUE / 2, 0, "half of MIN_VALUE underflows to 0");
assert.sameValue(Number.MIN_VALUE > 0, true, "MIN_VALUE is positive");
assert.sameValue(Number.MIN_VALUE < 2.2250738585072014e-308, true, "smaller than the smallest normal");

// The other Number statics for good measure.
assert.sameValue(Number.MAX_VALUE, 1.7976931348623157e308, "MAX_VALUE");
assert.sameValue(Number.MAX_SAFE_INTEGER, 9007199254740991, "MAX_SAFE_INTEGER");
assert.sameValue(Number.MIN_SAFE_INTEGER, -9007199254740991, "MIN_SAFE_INTEGER");
assert.sameValue(Number.POSITIVE_INFINITY, Infinity, "POSITIVE_INFINITY");
assert.sameValue(Number.NEGATIVE_INFINITY, -Infinity, "NEGATIVE_INFINITY");
