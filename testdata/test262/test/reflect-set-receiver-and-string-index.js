/*---
description: Reflect.set honors its receiver; canonical string indices address array storage
esid: sec-reflect.set
---*/
// A setter runs with the explicit receiver as `this`.
var base = { set x(v) { this.captured = v; } };
var recv = {};
Reflect.set(base, "x", 42, recv);
assert.sameValue(recv.captured, 42, "setter receiver");
assert.sameValue(base.captured, undefined, "target untouched");

// A data property with a receiver writes to the receiver, not the target.
var d = { a: 1 };
var r2 = {};
Reflect.set(d, "a", 99, r2);
assert.sameValue(r2.a, 99, "data written to receiver");
assert.sameValue(d.a, 1, "target data unchanged");

// An inherited setter is invoked (no receiver -> this is the target).
var pbase = { set y(v) { this._y = v * 2; } };
var child = Object.create(pbase);
Reflect.set(child, "y", 10);
assert.sameValue(child._y, 20, "inherited setter");

// A getter-only accessor cannot be set -> false. A primitive receiver -> false.
assert.sameValue(Reflect.set({ get z() { return 1; } }, "z", 5), false, "no setter");
assert.sameValue(Reflect.set({ a: 1 }, "a", 2, 5), false, "primitive receiver");

// Canonical numeric string indices address array storage (Reflect.set always
// stringifies keys); non-canonical strings remain named properties.
var arr = [1, 2, 3];
Reflect.set(arr, "1", 88);
assert.sameValue(arr.join(","), "1,88,3", "Reflect.set string index");
arr["2"] = 77;
assert.sameValue(arr.join(","), "1,88,77", "bracket string index");
arr["01"] = "x";
assert.sameValue(arr.join(","), "1,88,77", "non-canonical not an element");
assert.sameValue(arr["01"], "x", "non-canonical is a named property");

// A plain object keeps a numeric-string key as a named property.
var o = {};
o["1"] = 5;
assert.sameValue(JSON.stringify(o), '{"1":5}', "object numeric-string key");
