/*---
description: Object.defineProperty rejects mixed descriptors and allows no-op redefinition
esid: sec-validateandapplypropertydescriptor
---*/
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// A descriptor cannot mix accessor (get/set) and data (value/writable) fields.
assert.sameValue(throwsType(function () { Object.defineProperty({}, "x", { get: function () {}, value: 1 }); }), true, "get + value");
assert.sameValue(throwsType(function () { Object.defineProperty({}, "x", { set: function () {}, writable: true }); }), true, "set + writable");

// Valid pure-accessor and pure-data descriptors do not throw.
var ok = {};
Object.defineProperty(ok, "a", { get: function () { return 1; }, set: function () {} });
Object.defineProperty(ok, "b", { value: 1, writable: true });
assert.sameValue(ok.a, 1, "accessor ok");
assert.sameValue(ok.b, 1, "data ok");

// A no-op redefinition of a non-configurable property is allowed (every specified
// field already matches).
var o = {};
Object.defineProperty(o, "x", { value: 1, configurable: false });
Object.defineProperty(o, "x", { value: 1 }); // same value -> allowed
assert.sameValue(o.x, 1, "same-value redefine allowed");
Object.defineProperty(o, "n", { value: NaN });
Object.defineProperty(o, "n", { value: NaN }); // SameValue(NaN, NaN) -> allowed

// Real changes to a non-configurable property are rejected.
assert.sameValue(throwsType(function () { Object.defineProperty(o, "x", { value: 2 }); }), true, "different value");
assert.sameValue(throwsType(function () { Object.defineProperty(o, "x", { configurable: true }); }), true, "make configurable");
assert.sameValue(throwsType(function () { Object.defineProperty(o, "x", { writable: true }); }), true, "make writable");
assert.sameValue(throwsType(function () { Object.defineProperty(o, "x", { value: -0 }); }), true, "+0 vs -0 differ");

// A configurable/writable data property accepts a value change.
var w = {};
Object.defineProperty(w, "x", { value: 1, writable: true, configurable: true });
Object.defineProperty(w, "x", { value: 2 });
assert.sameValue(w.x, 2, "writable value change");

// An accessor redefined with the same getter is a no-op.
var g = function () { return 7; };
var acc = {};
Object.defineProperty(acc, "x", { get: g });
Object.defineProperty(acc, "x", { get: g });
assert.sameValue(acc.x, 7, "same-getter redefine allowed");
