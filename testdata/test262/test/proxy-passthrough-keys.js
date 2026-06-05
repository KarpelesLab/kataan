/*---
description: Object.keys/values/entries on a trap-less proxy forward to the target
esid: sec-proxy-object-internal-methods-and-internal-slots-ownpropertykeys
---*/
var target = { a: 1, b: 2, c: 3 };
var p = new Proxy(target, {});
assert.sameValue(Object.keys(p).join(","), "a,b,c", "Object.keys forwards to target");
assert.sameValue(Object.values(p).join(","), "1,2,3", "Object.values forwards");
assert.sameValue(Object.entries(p).map(function (e) { return e.join(":"); }).join(","), "a:1,b:2,c:3", "Object.entries forwards");
// Other proxy traps still work.
var g = new Proxy({}, { get: function (t, k) { return k + "!"; } });
assert.sameValue(g.foo, "foo!", "get trap");
var sets = [];
var s = new Proxy({}, { set: function (t, k, v) { sets.push(k + "=" + v); return true; } });
s.x = 5;
assert.sameValue(sets[0], "x=5", "set trap");
var h = new Proxy({}, { has: function (t, k) { return k === "yes"; } });
assert.sameValue("yes" in h, true, "has trap true");
assert.sameValue("no" in h, false, "has trap false");
var f = new Proxy(function () {}, { apply: function (t, thisArg, args) { return args[0] * 2; } });
assert.sameValue(f(21), 42, "apply trap");
// A nested proxy chain forwards through.
var p2 = new Proxy(p, {});
assert.sameValue(Object.keys(p2).join(","), "a,b,c", "nested proxy keys");
