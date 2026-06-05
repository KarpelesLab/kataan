/*---
description: Object.fromEntries from arrays, Maps, and round-trip with entries
esid: sec-object.fromentries
---*/
var o = Object.fromEntries([["a", 1], ["b", 2]]);
assert.sameValue(o.a, 1);
assert.sameValue(o.b, 2);
var m = new Map([["x", 10], ["y", 20]]);
var fromMap = Object.fromEntries(m);
assert.sameValue(fromMap.x, 10);
assert.sameValue(fromMap.y, 20);
var original = { p: 1, q: 2, r: 3 };
var roundTrip = Object.fromEntries(Object.entries(original));
assert.sameValue(roundTrip.p + roundTrip.q + roundTrip.r, 6, "entries -> fromEntries round-trip");
var doubled = Object.fromEntries(Object.entries(original).map(function (e) { return [e[0], e[1] * 2]; }));
assert.sameValue(doubled.q, 4);
