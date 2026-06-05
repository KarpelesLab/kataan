/*---
description: Date arithmetic, comparison, and getTime
esid: sec-date-objects
---*/
var d1 = new Date(2024, 0, 1);
var d2 = new Date(2024, 0, 2);
assert.sameValue(d2 - d1, 86400000, "one day in ms");
assert.sameValue(d2 > d1, true, "date comparison");
assert.sameValue(d1 < d2, true);
var t = new Date(1000).getTime();
assert.sameValue(t, 1000);
var copy = new Date(d1.getTime());
assert.sameValue(copy.getTime(), d1.getTime(), "copy via getTime");
assert.sameValue(new Date(0).getTime(), 0);
var future = new Date(d1.getTime() + 7 * 86400000);
assert.sameValue(future.getDate(), 8, "add a week");
assert.sameValue(typeof (d1 + ""), "string", "date to string");
