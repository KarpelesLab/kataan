/*---
description: Reflect.defineProperty/set/deleteProperty return booleans (false on failure, no throw)
esid: sec-reflect.defineproperty
features: [Reflect]
---*/
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// Reflect.defineProperty: false on a failed definition, true on success, but still THROWS
// for an invalid descriptor (which Object.defineProperty also rejects).
assert.sameValue(Reflect.defineProperty(Object.freeze({}), "x", { value: 1 }), false, "define on frozen -> false");
assert.sameValue(Reflect.defineProperty(Object.preventExtensions({}), "x", { value: 1 }), false, "define on non-extensible -> false");
var nc = {};
Object.defineProperty(nc, "x", { value: 1, configurable: false });
assert.sameValue(Reflect.defineProperty(nc, "x", { value: 2 }), false, "redefine non-configurable -> false");
assert.sameValue(Reflect.defineProperty({}, "x", { value: 1 }), true, "ordinary define -> true");
assert.sameValue(throwsType(function () { Reflect.defineProperty({}, "x", { value: 1, get: function () {} }); }), true, "invalid descriptor still throws");
// Object.defineProperty, by contrast, throws on the same failures.
assert.sameValue(throwsType(function () { Object.defineProperty(Object.freeze({}), "x", { value: 1 }); }), true, "Object.defineProperty throws on frozen");

// Reflect.set: false on a failed write, true on success; setters/getters respected.
var nw = {};
Object.defineProperty(nw, "y", { value: 1, writable: false });
assert.sameValue(Reflect.set(nw, "y", 2), false, "set non-writable -> false");
assert.sameValue(nw.y, 1, "value unchanged");
assert.sameValue(Reflect.set(Object.preventExtensions({}), "z", 1), false, "set new on non-extensible -> false");
assert.sameValue(Reflect.set(Object.freeze({ a: 1 }), "a", 2), false, "set frozen -> false");
assert.sameValue(Reflect.set({ get x() { return 1; } }, "x", 2), false, "set getter-only -> false");
assert.sameValue(Reflect.set({ a: 1 }, "a", 5), true, "ordinary set -> true");
var sink = {};
assert.sameValue(Reflect.set({ set x(v) { this.s = v; } }, "x", 9, sink), true, "setter -> true");
assert.sameValue(sink.s, 9, "setter ran on the receiver");

// Reflect.deleteProperty: false for a non-configurable property, true otherwise.
assert.sameValue(Reflect.deleteProperty({ a: 1 }, "a"), true, "ordinary delete -> true");
assert.sameValue(Reflect.deleteProperty(Object.freeze({ a: 1 }), "a"), false, "frozen delete -> false");
assert.sameValue(Reflect.deleteProperty(Object.seal({ a: 1 }), "a"), false, "sealed delete -> false");
var nd = {};
Object.defineProperty(nd, "x", { value: 1, configurable: false });
assert.sameValue(Reflect.deleteProperty(nd, "x"), false, "non-configurable delete -> false");
assert.sameValue(Reflect.deleteProperty({}, "missing"), true, "missing property delete -> true");
