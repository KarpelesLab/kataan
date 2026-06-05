/*---
description: Date.prototype.toISOString throws RangeError on an invalid date; toJSON returns null
esid: sec-date.prototype.toisostring
---*/
var invalid = new Date(NaN);
assert.sameValue(isNaN(invalid.getTime()), true, "invalid date getTime is NaN");
var threw = false;
try { invalid.toISOString(); } catch (e) { threw = e instanceof RangeError; }
assert.sameValue(threw, true, "toISOString throws RangeError for an invalid date");
// toJSON returns null (does not throw) for a non-finite time.
assert.sameValue(invalid.toJSON(), null, "toJSON returns null for an invalid date");
assert.sameValue(JSON.stringify({ d: invalid }), '{"d":null}', "JSON.stringify nulls an invalid date");
// A valid date round-trips.
var valid = new Date(Date.UTC(2020, 5, 15, 10, 30, 45, 123));
assert.sameValue(valid.toISOString(), "2020-06-15T10:30:45.123Z", "valid toISOString");
assert.sameValue(valid.toJSON(), "2020-06-15T10:30:45.123Z", "valid toJSON");
assert.sameValue(new Date(0).toISOString(), "1970-01-01T00:00:00.000Z", "epoch toISOString");
// Date.parse of a garbage string is NaN, which then throws on toISOString.
var fromGarbage = new Date("not a date");
assert.sameValue(isNaN(fromGarbage.getTime()), true, "garbage parse is NaN");
var threw2 = false;
try { fromGarbage.toISOString(); } catch (e) { threw2 = e instanceof RangeError; }
assert.sameValue(threw2, true, "garbage date toISOString throws");
