/*---
description: Date toDateString/toTimeString/toUTCString/toLocale* string methods
esid: sec-date.prototype.todatestring
---*/
var d = new Date(Date.UTC(2020, 5, 15, 10, 30, 45));
assert.sameValue(d.toDateString(), "Mon Jun 15 2020", "toDateString");
assert.sameValue(d.toUTCString(), "Mon, 15 Jun 2020 10:30:45 GMT", "toUTCString");
assert.sameValue(d.toLocaleDateString(), "6/15/2020", "toLocaleDateString");
assert.sameValue(d.toLocaleTimeString(), "10:30:45", "toLocaleTimeString");
assert.sameValue(d.toLocaleString(), "6/15/2020, 10:30:45", "toLocaleString");
assert.sameValue(d.toString().indexOf("Mon Jun 15 2020 10:30:45") === 0, true, "toString starts with date+time");
assert.sameValue(d.toTimeString().indexOf("10:30:45") === 0, true, "toTimeString");
// The epoch is a Thursday.
assert.sameValue(new Date(0).toDateString(), "Thu Jan 01 1970", "epoch weekday");
assert.sameValue(new Date(0).toUTCString(), "Thu, 01 Jan 1970 00:00:00 GMT", "epoch toUTCString");
// All twelve months render.
assert.sameValue(new Date(Date.UTC(2021, 11, 25)).toDateString(), "Sat Dec 25 2021", "December");
assert.sameValue(new Date(Date.UTC(2021, 0, 1)).toDateString(), "Fri Jan 01 2021", "January");
// A pre-epoch date.
assert.sameValue(new Date(Date.UTC(1969, 11, 31)).toDateString(), "Wed Dec 31 1969", "pre-epoch");
