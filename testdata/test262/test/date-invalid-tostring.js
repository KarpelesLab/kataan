/*---
description: an invalid Date stringifies as "Invalid Date"
esid: sec-date.prototype.tostring
---*/
var invalid = new Date(NaN);
assert.sameValue(invalid.getTime(), NaN, "invalid timestamp is NaN");
assert.sameValue(invalid.toString(), "Invalid Date", "toString");
assert.sameValue(invalid.toDateString(), "Invalid Date", "toDateString");
assert.sameValue(invalid.toTimeString(), "Invalid Date", "toTimeString");
assert.sameValue(invalid.toUTCString(), "Invalid Date", "toUTCString");
assert.sameValue(invalid.toLocaleString(), "Invalid Date", "toLocaleString");

// new Date("garbage") is also invalid.
assert.sameValue(new Date("not a date").toString(), "Invalid Date", "unparseable string");

// toJSON of an invalid date is null; toISOString throws.
assert.sameValue(new Date(NaN).toJSON(), null, "toJSON is null");
var threw = false;
try { new Date(NaN).toISOString(); } catch (e) { threw = e instanceof RangeError; }
assert.sameValue(threw, true, "toISOString throws RangeError");

// A valid date is unaffected.
assert.sameValue(new Date(Date.UTC(2024, 0, 1)).toDateString(), "Mon Jan 01 2024", "valid date");
