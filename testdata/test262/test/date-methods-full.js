/*---
description: Date getters, setters, and arithmetic
esid: sec-date-objects
---*/
var d = new Date(2024, 5, 15, 10, 30, 45);  // June 15, 2024, 10:30:45
assert.sameValue(d.getFullYear(), 2024);
assert.sameValue(d.getMonth(), 5, "June is month 5");
assert.sameValue(d.getDate(), 15);
assert.sameValue(d.getHours(), 10);
assert.sameValue(d.getMinutes(), 30);
assert.sameValue(d.getSeconds(), 45);
var epoch = new Date(0);
assert.sameValue(epoch.getTime(), 0);
assert.sameValue(epoch.getUTCFullYear(), 1970);
var d2 = new Date(2024, 0, 1);
var d3 = new Date(2024, 0, 31);
assert.sameValue((d3 - d2) / (1000 * 60 * 60 * 24), 30, "30 days apart");
var future = new Date(1000000000000);  // a specific timestamp
assert.sameValue(future.getTime(), 1000000000000);
assert.sameValue(typeof Date.now(), "number");
assert.sameValue(new Date(2024, 0, 1).getTime() < new Date(2024, 0, 2).getTime(), true);
var parsed = new Date(2000, 11, 25);  // Dec 25, 2000
assert.sameValue(parsed.getMonth(), 11);
assert.sameValue(parsed.getDate(), 25);
