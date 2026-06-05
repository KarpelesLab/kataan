/*---
description: Date construction, arithmetic, and comparison
esid: sec-date-objects
---*/
var d1 = new Date(2024, 0, 1);
var d2 = new Date(2024, 0, 15);
var diff = d2 - d1;
assert.sameValue(diff / (24 * 60 * 60 * 1000), 14, "14 days difference");
assert.sameValue(d2 > d1, true);
assert.sameValue(d1 < d2, true);
assert.sameValue(d1.getTime() < d2.getTime(), true);
var copy = new Date(d1.getTime());
assert.sameValue(copy.getTime(), d1.getTime());
assert.sameValue(copy.getTime() === d1.getTime(), true, "equal times");
var fromMs = new Date(1000000000000);
assert.sameValue(fromMs.getTime(), 1000000000000);
var future = new Date(d1.getTime() + 365 * 24 * 60 * 60 * 1000);
assert.sameValue(future.getFullYear(), 2024, "leap year stays same");
assert.sameValue(new Date(2024, 1, 29).getDate(), 29, "leap day");
assert.sameValue(new Date(2024, 2, 1).getMonth(), 2, "March");
var times = [new Date(2024, 0, 3), new Date(2024, 0, 1), new Date(2024, 0, 2)];
times.sort(function (a, b) { return a - b; });
assert.sameValue(times[0].getDate(), 1, "sorted dates");
assert.sameValue(times[2].getDate(), 3);
assert.sameValue(Math.abs(Date.now() - Date.now()) < 1000, true);
