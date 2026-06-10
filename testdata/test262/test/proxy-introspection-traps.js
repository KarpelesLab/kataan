/*---
description: Proxy getOwnPropertyDescriptor / setPrototypeOf / isExtensible traps fire
features: [Proxy, Reflect]
---*/
// getOwnPropertyDescriptor trap (Object and Reflect).
var p1 = new Proxy({}, {
  getOwnPropertyDescriptor(t, k) { return { value: "tr:" + k, enumerable: true, configurable: true }; },
});
assert.sameValue(Object.getOwnPropertyDescriptor(p1, "foo").value, "tr:foo", "Object.gOPD trap");
assert.sameValue(Reflect.getOwnPropertyDescriptor(p1, "bar").value, "tr:bar", "Reflect.gOPD trap");

// setPrototypeOf trap (Object and Reflect).
var calls = [];
var p2 = new Proxy({}, { setPrototypeOf(t, proto) { calls.push("sp"); return true; } });
Object.setPrototypeOf(p2, {});
Reflect.setPrototypeOf(p2, {});
assert.sameValue(calls.join(","), "sp,sp", "setPrototypeOf trap fires for both");

// isExtensible trap.
var p3 = new Proxy(Object.freeze({}), { isExtensible() { return true; } });
assert.sameValue(Object.isExtensible(p3), true, "isExtensible trap overrides");

// With no traps, all forward to the target.
var target = { a: 1 };
var p4 = new Proxy(target, {});
assert.sameValue(Object.getOwnPropertyDescriptor(p4, "a").value, 1, "gOPD forwards");
assert.sameValue(Object.isExtensible(p4), true, "isExtensible forwards");
