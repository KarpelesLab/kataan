/*---
description: Proxy deleteProperty and ownKeys traps
esid: sec-proxy-object-internal-methods
---*/
var target = { a: 1, b: 2, c: 3 };
var deleted = [];
var p = new Proxy(target, {
  deleteProperty: function (t, key) { deleted.push(key); delete t[key]; return true; }
});
delete p.a;
assert.sameValue(deleted.join(","), "a", "deleteProperty trap");
assert.sameValue(target.a, undefined, "actually deleted");
var counter = { count: 0 };
var logging = new Proxy({ x: 10 }, {
  get: function (t, key) { counter.count++; return t[key]; }
});
var v = logging.x;
v = logging.x;
assert.sameValue(counter.count, 2, "get trap counted");
assert.sameValue(v, 10);
var validated = new Proxy({}, {
  set: function (t, key, value) {
    if (typeof value !== "number") return false;
    t[key] = value; return true;
  }
});
validated.num = 42;
assert.sameValue(validated.num, 42, "valid set");
