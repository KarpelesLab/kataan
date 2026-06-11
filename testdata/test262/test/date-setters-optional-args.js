/*---
description: Date setters accept their optional trailing component arguments
esid: sec-date.prototype.setutcfullyear
---*/
// setUTCFullYear(year, month, date)
var d = new Date(0);
d.setUTCFullYear(2020, 5, 15);
assert.sameValue(d.getUTCFullYear(), 2020, "year");
assert.sameValue(d.getUTCMonth(), 5, "month");
assert.sameValue(d.getUTCDate(), 15, "date");

// A single-arg call sets only that field (does not clobber the rest).
var d2 = new Date(Date.UTC(2020, 5, 15, 10, 20, 30));
d2.setUTCDate(20);
assert.sameValue(
  d2.getUTCFullYear() + "," + d2.getUTCMonth() + "," + d2.getUTCDate() + "," + d2.getUTCHours(),
  "2020,5,20,10",
  "single-arg setUTCDate keeps other fields"
);

// setUTCHours(h, min, sec, ms)
var d3 = new Date(0);
d3.setUTCHours(10, 30, 45, 123);
assert.sameValue(d3.getUTCHours() + ":" + d3.getUTCMinutes() + ":" + d3.getUTCSeconds() + "." + d3.getUTCMilliseconds(), "10:30:45.123", "setUTCHours all args");

// setUTCMonth(month, date)
var d4 = new Date(Date.UTC(2020, 0, 1));
d4.setUTCMonth(5, 15);
assert.sameValue(d4.getUTCMonth() + "," + d4.getUTCDate(), "5,15", "setUTCMonth with date");

// setUTCMinutes(min, sec, ms) and setUTCSeconds(sec, ms)
var d5 = new Date(0);
d5.setUTCMinutes(30, 45);
assert.sameValue(d5.getUTCMinutes() + "," + d5.getUTCSeconds(), "30,45", "setUTCMinutes with sec");
var d6 = new Date(0);
d6.setUTCSeconds(45, 500);
assert.sameValue(d6.getUTCSeconds() + "," + d6.getUTCMilliseconds(), "45,500", "setUTCSeconds with ms");

// An out-of-range month rolls over even with multiple args.
var d7 = new Date(0);
d7.setUTCFullYear(2020, 13, 1);
assert.sameValue(d7.getUTCFullYear() + "," + d7.getUTCMonth(), "2021,1", "month overflow rolls into year");
