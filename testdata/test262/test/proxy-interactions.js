/*---
description: Proxy get/set/has/deleteProperty trap interactions
esid: sec-proxy-object-internal-methods
---*/
var log = [];
var target = { a: 1 };
var p = new Proxy(target, {
  get: function (t, k) { log.push("get:" + k); return t[k]; },
  set: function (t, k, v) { log.push("set:" + k); t[k] = v; return true; },
  has: function (t, k) { log.push("has:" + k); return k in t; },
  deleteProperty: function (t, k) { log.push("del:" + k); delete t[k]; return true; }
});
assert.sameValue(p.a, 1);
p.b = 2;
assert.sameValue(target.b, 2, "set trap writes through");
assert.sameValue("a" in p, true);
delete p.a;
assert.sameValue(target.a, undefined, "delete trap removes");
assert.sameValue(log.join(","), "get:a,set:b,has:a,del:a");
