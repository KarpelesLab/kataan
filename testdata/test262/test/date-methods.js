/*---
description: Date construction and accessors from a fixed timestamp
esid: sec-date-objects
---*/
var d = new Date(0);
assert.sameValue(d.getTime(), 0);
assert.sameValue(typeof Date.now(), "number");
var d2 = new Date(1000);
assert.sameValue(d2.getTime(), 1000);
assert.sameValue(d2.getTime() > d.getTime(), true);
