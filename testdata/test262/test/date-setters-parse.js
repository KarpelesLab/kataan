/*---
description: Date setters, Date.parse, and string construction
esid: sec-date-objects
---*/
var d = new Date(0);
d.setUTCFullYear(2000);
assert.sameValue(d.getUTCFullYear(), 2000, "setUTCFullYear");
d.setUTCMonth(5);
assert.sameValue(d.getUTCMonth(), 5, "setUTCMonth");
d.setUTCDate(15);
assert.sameValue(d.getUTCDate(), 15, "setUTCDate");
d.setUTCHours(12);
assert.sameValue(d.getUTCHours(), 12, "setUTCHours");
d.setUTCMinutes(30);
assert.sameValue(d.getUTCMinutes(), 30);
d.setUTCSeconds(45);
assert.sameValue(d.getUTCSeconds(), 45);
var t = new Date(0);
t.setTime(86400000);
assert.sameValue(t.getTime(), 86400000, "setTime");
assert.sameValue(t.getUTCDate(), 2, "one day later");
assert.sameValue(Date.parse("1970-01-01T00:00:00.000Z"), 0, "Date.parse epoch");
assert.sameValue(Date.parse("2000-01-01T00:00:00.000Z"), 946684800000, "Date.parse 2000");
assert.sameValue(Date.parse("2024-06-15"), Date.UTC(2024, 5, 15), "Date.parse date-only");
assert.sameValue(new Date("1970-01-01T00:00:00.000Z").getTime(), 0, "new Date(string)");
assert.sameValue(new Date("2000-01-01T12:00:00.000Z").getUTCHours(), 12, "parsed time");
assert.sameValue(Number.isNaN(Date.parse("not a date")), true, "invalid string is NaN");
assert.sameValue(Number.isNaN(new Date("garbage").getTime()), true, "invalid Date(string)");
assert.sameValue(new Date(0).getTimezoneOffset(), 0, "UTC offset");
var rollover = new Date(Date.UTC(2024, 0, 31));
rollover.setUTCDate(32);
assert.sameValue(rollover.getUTCMonth(), 1, "day rollover into next month");
