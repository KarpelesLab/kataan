/*---
description: Proxy get/set/has/deleteProperty traps and default forwarding
esid: sec-proxy-objects
---*/
var log = [];
var target = { a: 1 };
var p = new Proxy(target, {
  get: function (t, key) { return key in t ? t[key] : "default"; },
  set: function (t, key, value) { log.push("set:" + key); t[key] = value * 10; return true; },
  has: function (t, key) { log.push("has:" + key); return key in t; },
  deleteProperty: function (t, key) { log.push("del:" + key); delete t[key]; return true; }
});
assert.sameValue(p.a, 1);
assert.sameValue(p.missing, "default");
p.b = 5;
assert.sameValue(target.b, 50);
assert.sameValue("a" in p, true);
assert.sameValue("zzz" in p, false);
delete p.a;
assert.sameValue("a" in p, false);
assert.sameValue(log.join(","), "set:b,has:a,has:zzz,del:a,has:a");

// A handler with no traps forwards everything to the target.
var plain = new Proxy({ x: 7 }, {});
assert.sameValue(plain.x, 7);
plain.y = 9;
assert.sameValue(plain.y, 9);
assert.sameValue("x" in plain, true);
delete plain.x;
assert.sameValue("x" in plain, false);
