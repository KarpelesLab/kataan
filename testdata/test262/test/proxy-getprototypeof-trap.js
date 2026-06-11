/*---
description: Object/Reflect.getPrototypeOf invoke a proxy's getPrototypeOf trap
esid: sec-proxy-object-internal-methods-and-internal-slots-getprototypeof
---*/
var proto = { tag: "P" };

// The trap result is returned by both reflection entry points.
var p = new Proxy({}, { getPrototypeOf: function () { return proto; } });
assert.sameValue(Object.getPrototypeOf(p), proto, "Object.getPrototypeOf uses the trap");
assert.sameValue(Reflect.getPrototypeOf(p), proto, "Reflect.getPrototypeOf uses the trap");

// The trap may return null.
var np = new Proxy({}, { getPrototypeOf: function () { return null; } });
assert.sameValue(Object.getPrototypeOf(np), null, "trap returns null");

// A trapless proxy forwards to the target's prototype.
var base = {};
var fwd = new Proxy(Object.create(base), {});
assert.sameValue(Object.getPrototypeOf(fwd), base, "trapless forwards to target");

// A revoked proxy throws.
var rev = Proxy.revocable({}, { getPrototypeOf: function () { return proto; } });
rev.revoke();
assert.throws(TypeError, function () { return Object.getPrototypeOf(rev.proxy); }, "revoked throws");

// Ordinary objects are unaffected.
assert.sameValue(Object.getPrototypeOf({}), Object.prototype, "plain object");
assert.sameValue(Object.getPrototypeOf(Object.create(null)), null, "null-proto object");

// instanceof walks the prototype chain through the getPrototypeOf trap.
function Ctor() {}
var inst = new Proxy({}, { getPrototypeOf: function () { return Ctor.prototype; } });
assert.sameValue(inst instanceof Ctor, true, "instanceof uses the trap");
var mid = Object.create(Ctor.prototype);
var chained = new Proxy({}, { getPrototypeOf: function () { return mid; } });
assert.sameValue(chained instanceof Ctor, true, "instanceof walks through the trap result");
var unrelated = new Proxy({}, { getPrototypeOf: function () { return {}; } });
assert.sameValue(unrelated instanceof Ctor, false, "non-instance via trap");
