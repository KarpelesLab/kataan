/*---
description: a proxy in an object's prototype chain handles inherited property reads via its [[Get]]
esid: sec-proxy-object-internal-methods-and-internal-slots-get-p-receiver
features: [Proxy]
---*/
// No trap: the read forwards through the proxy to its target.
var pp = new Proxy({ inherited: "yes" }, {});
var child = Object.create(pp);
assert.sameValue(child.inherited, "yes", "forward to target through proxy prototype");
assert.sameValue(child.notthere, undefined, "missing property is undefined");

// A get trap on the prototype proxy fires for inherited reads.
var tp = new Proxy({}, { get: function (t, k) { return "trapped-" + String(k); } });
var c2 = Object.create(tp);
assert.sameValue(c2.anything, "trapped-anything", "get trap on prototype proxy");

// An own property shadows the proxy prototype.
var c3 = Object.create(pp);
c3.inherited = "own";
assert.sameValue(c3.inherited, "own", "own property shadows");
assert.sameValue(c3.hasOwnProperty("inherited"), true, "own property present");

// The proxy is not the object's own — inheritance only.
assert.sameValue(child.hasOwnProperty("inherited"), false, "not an own property of child");

// Deeper objects in the proxy's target still resolve (Object.prototype methods).
assert.sameValue(typeof child.toString, "function", "Object.prototype reachable through proxy");

// Nested proxy prototypes chain correctly.
var base = { x: 1 };
var p1 = new Proxy(base, {});
var mid = Object.create(p1);
var p2 = new Proxy(mid, {});
var top = Object.create(p2);
assert.sameValue(top.x, 1, "nested proxy prototypes");

// Ordinary prototype chains are unaffected.
var rp = { a: 1 };
assert.sameValue(Object.create(rp).a, 1, "plain prototype");
