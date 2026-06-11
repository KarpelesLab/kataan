/*---
description: functions are extensible; defineProperty works on them, including length/name override
esid: sec-function-instances
---*/
function f(a, b) {}

// Functions/classes are extensible.
assert.sameValue(Object.isExtensible(f), true, "function extensible");
assert.sameValue(Object.isExtensible(function () {}), true, "anon function extensible");
assert.sameValue(Object.isExtensible(class {}), true, "class extensible");

// A custom property can be defined on a function (previously threw "not extensible").
Object.defineProperty(f, "meta", { value: 99 });
assert.sameValue(f.meta, 99, "custom property defined");
assert.sameValue("meta" in f, true, "custom property present");
f.direct = 7;
assert.sameValue(f.direct, 7, "direct assignment");

// length and name are configurable -> redefinable, and the override takes effect.
assert.sameValue(f.length, 2, "default length");
Object.defineProperty(f, "length", { value: 5 });
assert.sameValue(f.length, 5, "length override effective");
assert.sameValue(Object.getOwnPropertyDescriptor(f, "length").value, 5, "descriptor agrees");
function g(a) {}
Object.defineProperty(g, "name", { value: "renamed" });
assert.sameValue(g.name, "renamed", "name override");

// Functions without an override still compute length/name.
function h(a, b, c) {}
assert.sameValue(h.length, 3, "computed length");
assert.sameValue((function named() {}).name, "named", "computed name");
