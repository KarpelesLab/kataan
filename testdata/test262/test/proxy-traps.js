/*---
description: Proxy get/set/has/deleteProperty traps
esid: sec-proxy-objects
---*/
var target = { a: 1, b: 2 };
var log = [];
var p = new Proxy(target, {
  get(t, key) { log.push("get:" + key); return key in t ? t[key] : "default"; },
  set(t, key, val) { log.push("set:" + key); t[key] = val * 2; return true; },
  has(t, key) { return key === "a" || key in t; }
});
assert.sameValue(p.a, 1, "get trap returns value");
assert.sameValue(p.missing, "default", "get trap default");
p.c = 5;
assert.sameValue(target.c, 10, "set trap doubles");
assert.sameValue("a" in p, true, "has trap");
assert.sameValue("z" in p, false);
assert.sameValue(log.indexOf("get:a") >= 0, true);
var counter = { count: 0 };
var tracker = new Proxy(counter, {
  get(t, k) { if (k === "count") return t.count; return undefined; },
  set(t, k, v) { t[k] = v; return true; }
});
tracker.count = 100;
assert.sameValue(tracker.count, 100);
var arr = new Proxy([1, 2, 3], {
  get(t, k) { return k === "length" ? t.length : t[k]; }
});
assert.sameValue(arr.length, 3, "proxy array length");
assert.sameValue(arr[0], 1);
var validated = new Proxy({}, {
  set(t, k, v) { if (typeof v !== "number") throw new TypeError("must be number"); t[k] = v; return true; }
});
validated.x = 42;
assert.sameValue(validated.x, 42);
var threw = false;
try { validated.y = "string"; } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "set validation");
