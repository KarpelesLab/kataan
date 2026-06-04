/*---
description: Proxy get/set traps and default forwarding to the target
esid: sec-proxy-objects
---*/
var log = [];
var target = { a: 1 };
var p = new Proxy(target, {
  get: function (t, key) { return key in t ? t[key] : "default"; },
  set: function (t, key, value) { log.push(key); t[key] = value * 10; return true; }
});
assert.sameValue(p.a, 1, "get forwards to an existing property");
assert.sameValue(p.missing, "default", "get trap handles absent keys");
p.b = 5;
assert.sameValue(target.b, 50, "set trap transforms the value");
assert.sameValue(log.join(","), "b");

// A handler with no traps forwards reads/writes to the target.
var plain = new Proxy({ x: 7 }, {});
assert.sameValue(plain.x, 7);
plain.y = 9;
assert.sameValue(plain.y, 9);
assert.sameValue(typeof plain, "object");
