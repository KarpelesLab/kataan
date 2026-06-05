/*---
description: getPrototypeOf, setPrototypeOf, prototype chains
esid: sec-object.getprototypeof
---*/
var base = { greet: function () { return "hello"; } };
var obj = Object.create(base);
assert.sameValue(Object.getPrototypeOf(obj), base);
assert.sameValue(obj.greet(), "hello", "inherited method");
var other = { greet: function () { return "hi"; } };
Object.setPrototypeOf(obj, other);
assert.sameValue(obj.greet(), "hi", "prototype changed");
assert.sameValue(Object.getPrototypeOf(obj), other);
var nullProto = Object.create(null);
assert.sameValue(Object.getPrototypeOf(nullProto), null);
nullProto.x = 1;
assert.sameValue(nullProto.x, 1);
function Animal() {}
Animal.prototype.sound = function () { return "generic"; };
var a = new Animal();
assert.sameValue(a.sound(), "generic", "constructor prototype method");
assert.sameValue(Object.getPrototypeOf(a), Animal.prototype);
var chain = Object.create(Object.create(base));
assert.sameValue(chain.greet(), "hello", "two-level prototype chain");
assert.sameValue(a instanceof Animal, true);
