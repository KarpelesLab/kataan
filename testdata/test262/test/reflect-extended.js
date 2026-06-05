/*---
description: Reflect.defineProperty, getOwnPropertyDescriptor, getPrototypeOf
esid: sec-reflect.defineproperty
---*/
var o = {};
var ok = Reflect.defineProperty(o, "x", { value: 42, enumerable: true });
assert.sameValue(ok, true, "defineProperty returns boolean");
assert.sameValue(o.x, 42);
var desc = Reflect.getOwnPropertyDescriptor(o, "x");
assert.sameValue(desc.value, 42);
assert.sameValue(desc.enumerable, true);
var base = { inherited: 1 };
var child = Object.create(base);
assert.sameValue(Reflect.getPrototypeOf(child), base);
assert.sameValue(Reflect.has(child, "inherited"), true, "has checks chain");
assert.sameValue(Reflect.ownKeys({ a: 1, b: 2 }).length, 2);
var arr = [1, 2, 3];
assert.sameValue(Reflect.get(arr, "length"), 3, "Reflect.get length");
Reflect.set(arr, 3, 4);
assert.sameValue(arr[3], 4);
assert.sameValue(arr.length, 4);
function Point(x) { this.x = x; }
var p = Reflect.construct(Point, [5]);
assert.sameValue(p.x, 5);
assert.sameValue(Reflect.apply(Math.max, null, [1, 5, 3]), 5);
