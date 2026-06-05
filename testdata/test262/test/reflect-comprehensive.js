/*---
description: Reflect methods comprehensive
esid: sec-reflect-object
---*/
var obj = { a: 1, b: 2 };
assert.sameValue(Reflect.get(obj, "a"), 1);
assert.sameValue(Reflect.has(obj, "a"), true);
assert.sameValue(Reflect.has(obj, "z"), false);
Reflect.set(obj, "c", 3);
assert.sameValue(obj.c, 3);
assert.sameValue(Reflect.ownKeys(obj).join(","), "a,b,c");
Reflect.deleteProperty(obj, "a");
assert.sameValue("a" in obj, false);
assert.sameValue(Reflect.get({ x: 10 }, "x"), 10);
function Ctor(a, b) { this.sum = a + b; }
var inst = Reflect.construct(Ctor, [3, 4]);
assert.sameValue(inst.sum, 7, "Reflect.construct");
function greet(greeting) { return greeting + ", " + this.name; }
assert.sameValue(Reflect.apply(greet, { name: "World" }, ["Hello"]), "Hello, World", "Reflect.apply");
var defined = {};
Reflect.defineProperty(defined, "prop", { value: 99, enumerable: true });
assert.sameValue(defined.prop, 99);
assert.sameValue(Reflect.getPrototypeOf([]), Array.prototype || Object.getPrototypeOf([]));
var keys = Reflect.ownKeys({ x: 1, y: 2, z: 3 });
assert.sameValue(keys.length, 3);
