/*---
description: instance.constructor back-references the function/class; non-enumerable
esid: sec-properties-of-the-function-prototype-object
---*/
// Function used as a constructor.
function Foo() { this.x = 1; }
var f = new Foo();
assert.sameValue(f.constructor, Foo, "instance.constructor is the function");
assert.sameValue(f.constructor.name, "Foo", "constructor.name");

// Class declarations.
class C {}
var c = new C();
assert.sameValue(c.constructor, C, "class instance.constructor is the class");
assert.sameValue(c.constructor.name, "C", "class constructor.name");

// Derived class: the most-derived constructor wins.
class D extends C {}
var d = new D();
assert.sameValue(d.constructor, D, "derived instance.constructor is the derived class");
assert.sameValue(d.constructor.name, "D", "derived constructor.name");

// `constructor` is non-enumerable — it must not show up in own keys.
assert.sameValue(Object.keys(c).indexOf("constructor"), -1, "constructor stays out of Object.keys");
assert.sameValue(Object.keys(f).join(","), "x", "only the data field is enumerable");

// Foo.prototype.constructor === Foo (the back-reference lives on the prototype).
assert.sameValue(Foo.prototype.constructor, Foo, "prototype.constructor");

// Plain objects inherit `constructor === Object` via Object.prototype.
assert.sameValue(({}).constructor, Object, "({}).constructor is Object");
assert.sameValue(({}).constructor.name, "Object", "Object.name");
assert.sameValue(Object.keys({ z: 1 }).indexOf("constructor"), -1, "inherited constructor is not an own key");
