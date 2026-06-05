/*---
description: Proxy get, set, has traps
esid: sec-proxy-objects
---*/
var target = { a: 1 };
var log = [];
var p = new Proxy(target, {
  get: function (t, key) { log.push("get:" + key); return t[key]; },
  set: function (t, key, value) { log.push("set:" + key); t[key] = value; return true; },
  has: function (t, key) { return key in t; }
});
assert.sameValue(p.a, 1, "get trap forwards");
p.b = 2;
assert.sameValue(target.b, 2, "set trap writes through");
assert.sameValue("a" in p, true, "has trap");
assert.sameValue(log.indexOf("get:a") >= 0, true);
assert.sameValue(log.indexOf("set:b") >= 0, true);
var defaulting = new Proxy({}, {
  get: function (t, key) { return key in t ? t[key] : "default"; }
});
defaulting.x = 5;
assert.sameValue(defaulting.x, 5);
assert.sameValue(defaulting.missing, "default", "default via get trap");
