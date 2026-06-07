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

// super also works inside object-literal accessors and generator methods.
var p3 = { get base() { return 10; }, calc() { return 5; } };
var o3 = {
  __proto__: p3,
  get derived() { return super.base + 1; },
  set v(x) { this._v = super.calc() + x; },
};
assert.sameValue(o3.derived, 11, "super in a getter");
o3.v = 3;
assert.sameValue(o3._v, 8, "super in a setter");

var p4 = { items() { return [1, 2]; } };
var o4 = { __proto__: p4, *gen() { yield* super.items(); yield 3; } };
assert.sameValue([...o4.gen()].join(","), "1,2,3", "super in a generator method");
