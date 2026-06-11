/*---
description: Reflect.construct uses its newTarget for new.target and the instance prototype
esid: sec-reflect.construct
---*/
// new.target inside the constructor is the supplied newTarget.
function Base() { this.t = new.target.name; }
function Derived() {}
assert.sameValue(Reflect.construct(Base, [], Derived).t, "Derived", "new.target is newTarget");
assert.sameValue(Reflect.construct(Base, []).t, "Base", "without newTarget it is the target");

// A class target with a function/class newTarget.
class C { constructor() { this.nt = new.target.name; } }
class D extends C {}
assert.sameValue(Reflect.construct(C, [], D).nt, "D", "class new.target");

// The instance's [[Prototype]] is newTarget.prototype.
function P() {}
var proto = { marker: 1 };
function NT() {}
NT.prototype = proto;
var inst = Reflect.construct(P, [], NT);
assert.sameValue(Object.getPrototypeOf(inst), proto, "instance proto is newTarget.prototype");
assert.sameValue(inst instanceof NT, true, "instanceof newTarget");
assert.sameValue(inst instanceof P, false, "not instanceof target");

// Arguments are still forwarded.
function Sum(a, b) { this.s = a + b; }
function ST() {}
assert.sameValue(Reflect.construct(Sum, [3, 4], ST).s, 7, "args forwarded");

// Default construct and ordinary new are unaffected.
function Q() {}
assert.sameValue(Object.getPrototypeOf(Reflect.construct(Q, [])), Q.prototype, "default uses target.prototype");
function N() { this.x = 1; }
var n = new N();
assert.sameValue(Object.getPrototypeOf(n), N.prototype, "new uses constructor.prototype");
assert.sameValue(n instanceof N, true, "new instanceof");
