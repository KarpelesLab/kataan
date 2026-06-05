/*---
description: Date components, getDay, and ISO formatting
esid: sec-date-objects
---*/
var d = new Date(Date.UTC(2024, 0, 15, 10, 30, 0));
assert.sameValue(d.getUTCFullYear(), 2024);
assert.sameValue(d.getUTCMonth(), 0, "January is 0");
assert.sameValue(d.getUTCDate(), 15);
assert.sameValue(d.getUTCHours(), 10);
assert.sameValue(d.getUTCMinutes(), 30);
var epoch = new Date(0);
assert.sameValue(epoch.getUTCFullYear(), 1970);
assert.sameValue(epoch.toISOString(), "1970-01-01T00:00:00.000Z");
var known = new Date(Date.UTC(2024, 0, 1));
assert.sameValue(known.getUTCDay(), 1, "2024-01-01 is a Monday");
assert.sameValue(typeof Date.now(), "number");
