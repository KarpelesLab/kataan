/*---
description: super.x = v invokes the inherited setter with the current this
esid: sec-super-keyword-runtime-semantics-evaluation
---*/
// A subclass setter delegating to the superclass setter via super.x = v.
class B { set val(v) { this._v = v; } get val() { return this._v; } }
class D extends B { set val(v) { super.val = v * 2; } }
var d = new D();
d.val = 10;
assert.sameValue(d._v, 20, "super setter invoked with this");

// A subclass with both super getter and super setter.
class B2 { constructor() { this._x = 0; } set x(v) { this._x = v; } get x() { return this._x; } }
class D2 extends B2 { set x(v) { super.x = v + 100; } get x() { return super.x; } }
var d2 = new D2();
d2.x = 5;
assert.sameValue(d2.x, 105, "super getter reads");
assert.sameValue(d2._x, 105, "super setter wrote");

// No inherited setter: the write lands on the receiver (this).
class B3 {}
class D3 extends B3 { m() { super.foo = 42; return this.foo; } }
assert.sameValue(new D3().m(), 42, "super write without a setter targets this");

// Through three levels, each setter delegating up.
class L1 { set v(x) { this._v = x; } }
class L2 extends L1 { set v(x) { super.v = x + 1; } }
class L3 extends L2 { set v(x) { super.v = x + 10; } }
var l = new L3();
l.v = 100;
assert.sameValue(l._v, 111, "three-level super setter chain");

// A compound assignment through super reads then writes.
class C1 { constructor() { this._n = 10; } get n() { return this._n; } set n(v) { this._n = v; } }
class C2 extends C1 { add() { super.n += 5; return this.n; } }
assert.sameValue(new C2().add(), 15, "super compound assignment");

// An object-literal method's super.x = v uses HomeObject's prototype.
var proto = { set y(v) { this._y = v; } };
var obj = { __proto__: proto, set y(v) { super.y = v * 3; } };
obj.y = 4;
assert.sameValue(obj._y, 12, "object-literal super setter");

// Ordinary setters are unaffected.
class R { set p(v) { this._p = v; } }
var r = new R();
r.p = 7;
assert.sameValue(r._p, 7, "ordinary setter");
