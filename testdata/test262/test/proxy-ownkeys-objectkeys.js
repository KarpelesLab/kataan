/*---
description: Object.keys invokes a Proxy ownKeys trap and filters by enumerability
features: [Proxy]
---*/
// ownKeys + getOwnPropertyDescriptor: only enumerable keys are returned.
var p = new Proxy({}, {
  ownKeys() { return ["a", "b", "c"]; },
  getOwnPropertyDescriptor(t, k) {
    return { enumerable: k !== "c", configurable: true, value: 1 };
  },
});
assert.sameValue(Object.keys(p).join(","), "a,b", "non-enumerable key filtered");

// ownKeys trap without a descriptor trap: forward to the target — only keys the
// target actually owns and that are enumerable.
var target = { x: 1, y: 2 };
var p2 = new Proxy(target, { ownKeys() { return ["x", "y", "z"]; } });
assert.sameValue(Object.keys(p2).join(","), "x,y", "missing target key excluded");

// No ownKeys trap: the target's own enumerable keys.
assert.sameValue(Object.keys(new Proxy({ m: 1, n: 2 }, {})).join(","), "m,n", "no trap forwards");
