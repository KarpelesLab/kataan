/*---
description: super in an object-literal concise method resolves through [[HomeObject]]
features: [super]
---*/
// super.method() calls the prototype's method with the current `this`.
var parent = { greet() { return "hello"; } };
var obj = { __proto__: parent, greet() { return super.greet() + " world"; } };
assert.sameValue(obj.greet(), "hello world", "super.method()");

// super.prop reads the prototype's data property.
var base = { val: 100 };
var o2 = { __proto__: base, getVal() { return super.val; } };
assert.sameValue(o2.getVal(), 100, "super.prop");

// Arguments forward, and `this` stays the receiver.
var calc = { add(a, b) { return a + b; } };
var c2 = { __proto__: calc, add(a, b) { return super.add(a, b) * 2; } };
assert.sameValue(c2.add(3, 4), 14, "super with args");

var counter = { base: 10, read() { return this.base; } };
var d2 = { __proto__: counter, base: 20, read() { return super.read(); } };
assert.sameValue(d2.read(), 20, "super method sees the derived this");

// Class super is unaffected.
class A { get prop() { return "A"; } }
class B extends A { get prop() { return super.prop + "B"; } }
assert.sameValue(new B().prop, "AB", "class super still works");
