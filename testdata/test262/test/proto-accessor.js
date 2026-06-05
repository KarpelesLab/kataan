/*---
description: __proto__ get/set updates the prototype link
esid: sec-object.prototype.__proto__
---*/
// Reading __proto__ returns the prototype set via Object.create.
var proto = { greet: function () { return "hi"; } };
var o = Object.create(proto);
assert.sameValue(o.__proto__, proto, "__proto__ reads the prototype");
assert.sameValue(o.greet(), "hi", "inherited method");
// Assigning __proto__ relinks.
var o2 = {};
var p2 = { hello: function () { return "yo"; } };
o2.__proto__ = p2;
assert.sameValue(o2.__proto__, p2, "__proto__ set then get");
assert.sameValue(o2.hello(), "yo", "method inherited after __proto__ assignment");
// A method finds `this`.
var base = { getX: function () { return this.x; } };
var d = {};
d.__proto__ = base;
d.x = 42;
assert.sameValue(d.getX(), 42, "inherited method sees this");
// Setting __proto__ = null removes the chain.
var n = { __proto__: base };
assert.sameValue(n.getX !== undefined, true, "object literal __proto__ inherits");
n.__proto__ = null;
assert.sameValue(Object.getPrototypeOf(n), null, "__proto__ = null clears the prototype");
// A non-object value is ignored.
var keep = Object.create(proto);
keep.__proto__ = 5;
assert.sameValue(keep.greet(), "hi", "assigning a primitive to __proto__ is ignored");
