/*---
description: Date construction, components, and arithmetic
esid: sec-date-objects
---*/
var d = new Date(2026, 5, 5);
assert.sameValue(d.getFullYear(), 2026);
assert.sameValue(d.getMonth(), 5, "months are 0-indexed");
assert.sameValue(d.getDate(), 5);
var t = new Date(0);
assert.sameValue(t.getTime(), 0, "epoch");
var a = new Date(1000);
var b = new Date(2000);
assert.sameValue(b - a, 1000, "date subtraction yields milliseconds");
assert.sameValue(typeof Date.now(), "number");
