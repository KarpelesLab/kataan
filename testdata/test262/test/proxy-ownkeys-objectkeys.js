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

// Object.values / Object.entries also drive off the ownKeys trap, reading each
// value through the proxy (so a get trap is honored).
var pv = new Proxy({}, {
  ownKeys() { return ["a", "b"]; },
  getOwnPropertyDescriptor() { return { enumerable: true, configurable: true, value: 1 }; },
  get(t, k) { return k.toUpperCase() + "!"; },
});
assert.sameValue(Object.values(pv).join(","), "A!,B!", "values via ownKeys + get trap");
assert.sameValue(JSON.stringify(Object.entries(pv)), '[["a","A!"],["b","B!"]]', "entries via traps");

// Without a get trap, values forward to the target.
var pt = new Proxy({ x: 10, y: 20 }, { ownKeys() { return ["x", "y"]; } });
assert.sameValue(Object.values(pt).join(","), "10,20", "values forward to target");

// for-in over a proxy uses the ownKeys trap (enumerable keys), reading values
// through the get trap; without an ownKeys trap it forwards to the target.
var seen = [];
for (var k in pv) seen.push(k + "=" + pv[k]);
assert.sameValue(seen.join(" "), "a=A! b=B!", "for-in via ownKeys + get");
var fwd = "";
for (var k2 in new Proxy({ p: 1, q: 2 }, {})) fwd += k2;
assert.sameValue(fwd, "pq", "for-in forwards to target when no ownKeys trap");
