/*---
description: Object.prototype.toString.call(x) produces the correct [object Tag]
esid: sec-object.prototype.tostring
---*/
var toString = Object.prototype.toString;
assert.sameValue(typeof Object.prototype, "object", "Object.prototype is an object");
assert.sameValue(toString.call({}), "[object Object]", "plain object");
assert.sameValue(toString.call([]), "[object Array]", "array");
assert.sameValue(toString.call(null), "[object Null]", "null");
assert.sameValue(toString.call(undefined), "[object Undefined]", "undefined");
assert.sameValue(toString.call(function () {}), "[object Function]", "function");
assert.sameValue(toString.call(new Date()), "[object Date]", "date");
assert.sameValue(toString.call(/x/), "[object RegExp]", "regexp");
assert.sameValue(toString.call({ [Symbol.toStringTag]: "Widget" }), "[object Widget]", "toStringTag");
// hasOwnProperty / propertyIsEnumerable / isPrototypeOf via the prototype.
assert.sameValue(Object.prototype.hasOwnProperty.call({ a: 1 }, "a"), true, "hasOwnProperty true");
assert.sameValue(Object.prototype.hasOwnProperty.call({ a: 1 }, "b"), false, "hasOwnProperty false");
var o = {};
Object.defineProperty(o, "hidden", { value: 1, enumerable: false });
o.shown = 2;
assert.sameValue(Object.prototype.propertyIsEnumerable.call(o, "shown"), true, "enumerable true");
assert.sameValue(Object.prototype.propertyIsEnumerable.call(o, "hidden"), false, "enumerable false");
var proto = {};
var child = Object.create(proto);
assert.sameValue(Object.prototype.isPrototypeOf.call(proto, child), true, "isPrototypeOf true");
assert.sameValue(Object.prototype.isPrototypeOf.call({}, child), false, "isPrototypeOf false");
// valueOf returns the receiver.
var obj = { x: 1 };
assert.sameValue(Object.prototype.valueOf.call(obj), obj, "valueOf returns this");
