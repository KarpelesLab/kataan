/*---
description: computed assignment / Reflect.set / Object.assign invoke an inherited setter
esid: sec-ordinaryset
---*/
// A computed assignment finds an inherited setter and runs it with this = receiver.
var base = { set q(v) { this._q = v; } };
var c = Object.create(base);
var k = "q";
c[k] = 99;
assert.sameValue(c._q, 99, "computed assignment inherited setter");
assert.sameValue(c.hasOwnProperty("q"), false, "no own accessor created");

// Reflect.set too.
var b2 = { set w(v) { this._w = v * 3; } };
var c2 = Object.create(b2);
Reflect.set(c2, "w", 5);
assert.sameValue(c2._w, 15, "Reflect.set inherited setter");

// Object.assign too.
var b3 = { set z(v) { this._z = v; } };
var c3 = Object.create(b3);
Object.assign(c3, { z: 7 });
assert.sameValue(c3._z, 7, "Object.assign inherited setter");

// An own setter still wins over the inherited chain.
var own = { set s(v) { this.captured = v; } };
own["s"] = 10;
assert.sameValue(own.captured, 10, "own setter precedence");

// An inherited getter-only accessor makes the write a no-op (no own property created).
var gb = { get g() { return 1; } };
var gc = Object.create(gb);
gc["g"] = 5;
assert.sameValue(gc.g, 1, "getter-only write ignored");
assert.sameValue(gc.hasOwnProperty("g"), false, "no own property created");

// An inherited *data* property is shadowed by a new own data property.
var db = { d: 1 };
var dc = Object.create(db);
dc["d"] = 2;
assert.sameValue(dc.d, 2, "own data shadows");
assert.sameValue(db.d, 1, "prototype unchanged");
assert.sameValue(dc.hasOwnProperty("d"), true, "own data created");

// A brand-new property and array indices are unaffected.
var np = {};
np["x"] = 5;
assert.sameValue(np.x, 5, "new property");
var arr = [1, 2, 3];
arr[1] = 99;
assert.sameValue(arr.join(","), "1,99,3", "array index");
