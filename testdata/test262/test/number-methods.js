/*---
description: Number coercion and methods
esid: sec-properties-of-the-number-prototype-object
---*/
assert.sameValue((3.14159).toFixed(2), "3.14");
assert.sameValue(parseInt("42px", 10), 42);
assert.sameValue(parseFloat("3.5kg"), 3.5);
assert.sameValue(Number.isInteger(5), true);
assert.sameValue(Number.isNaN(NaN), true);
assert.sameValue((255).toString(16), "ff");
