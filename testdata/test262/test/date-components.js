/*---
description: Date component getters and time arithmetic
esid: sec-date-objects
---*/
var d = new Date(2024, 5, 15, 14, 30, 45, 500);
assert.sameValue(d.getFullYear(), 2024);
assert.sameValue(d.getMonth(), 5);
assert.sameValue(d.getDate(), 15);
assert.sameValue(d.getHours(), 14);
assert.sameValue(d.getMinutes(), 30);
assert.sameValue(d.getSeconds(), 45);
assert.sameValue(d.getMilliseconds(), 500);
var epoch = new Date(0);
assert.sameValue(epoch.getTime(), 0);
assert.sameValue(epoch.getUTCFullYear(), 1970);
assert.sameValue(epoch.getUTCMonth(), 0);
assert.sameValue(epoch.getUTCDate(), 1);
var oneHour = new Date(3600000);
assert.sameValue(oneHour.getUTCHours(), 1);
var oneDay = 24 * 60 * 60 * 1000;
var d1 = new Date(2024, 0, 1);
var d2 = new Date(d1.getTime() + 30 * oneDay);
assert.sameValue(d2.getDate(), 31, "30 days later");
assert.sameValue(new Date(2024, 0, 1).getDay() >= 0, true, "day of week 0-6");
assert.sameValue(Date.UTC(1970, 0, 1), 0, "Date.UTC epoch");
assert.sameValue(Date.UTC(1970, 0, 2), oneDay, "Date.UTC one day");
assert.sameValue(typeof Date.now(), "number");
assert.sameValue(new Date(2024, 11, 25).getMonth(), 11);
