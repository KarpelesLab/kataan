/*---
description: Object.defineProperty/defineProperties route through a Proxy's defineProperty trap
features: [Proxy]
---*/
// The trap intercepts Object.defineProperty(proxy, ...).
var seen = [];
var p = new Proxy({}, {
  defineProperty(target, key, desc) { seen.push(key + "=" + desc.value); return true; },
});
Object.defineProperty(p, "z", { value: 1 });
assert.sameValue(seen.join(","), "z=1", "trap receives key and descriptor");

// Each property of Object.defineProperties routes through the trap.
var seen2 = [];
var p2 = new Proxy({}, {
  defineProperty(target, key) { seen2.push(key); return true; },
});
Object.defineProperties(p2, { a: { value: 1 }, b: { value: 2 } });
assert.sameValue(seen2.sort().join(","), "a,b", "defineProperties routes each key");

// With no trap, the operation forwards to the target.
var target = {};
var p3 = new Proxy(target, {});
Object.defineProperty(p3, "w", { value: 9, enumerable: true });
assert.sameValue(target.w, 9, "forwarded value");
assert.sameValue(Object.keys(target).join(","), "w", "forwarded enumerable");
