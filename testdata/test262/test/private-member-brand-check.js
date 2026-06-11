/*---
description: reading a private member from an object whose class did not declare it throws TypeError
esid: sec-privatefieldget
features: [class-fields-private, class-methods-private]
---*/
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// A private field read from a non-instance is a TypeError, not undefined.
class J { #j = 1; static read(o) { return o.#j; } }
assert.sameValue(J.read(new J()), 1, "valid instance read");
assert.sameValue(throwsType(function () { return J.read({}); }), true, "plain object -> TypeError");
assert.sameValue(throwsType(function () { return J.read(null === null ? Object.create(null) : 0); }), true, "null-proto object -> TypeError");

// A private method invoked from a non-instance is a TypeError.
class B { #m() { return 9; } static invoke(o) { return o.#m(); } }
assert.sameValue(B.invoke(new B()), 9, "valid private method call");
assert.sameValue(throwsType(function () { return B.invoke({}); }), true, "private method on plain object -> TypeError");

// Cross-class: an instance of a different class lacks the brand.
class P { #p = 1; }
class Q { #q = 2; static read(o) { return o.#q; } }
assert.sameValue(throwsType(function () { return Q.read(new P()); }), true, "wrong class instance -> TypeError");

// Valid uses are unaffected: fields, methods, accessors, inheritance, compound assignment.
class C { #v = 10; get #d() { return this.#v * 2; } test() { return this.#d; } }
assert.sameValue(new C().test(), 20, "private accessor");
class H { #h = 1; getH() { return this.#h; } }
class I extends H {}
assert.sameValue(new I().getH(), 1, "inherited private field");
class K { #n = 0; #s = 1; inc() { this.#n += this.#s; return this.#n; } }
var k = new K();
assert.sameValue(k.inc(), 1, "compound assign 1");
assert.sameValue(k.inc(), 2, "compound assign 2");

// Static privates (read off the class) and read-then-call of a private method still work.
class Counter { static #c = 0; static bump() { return ++Counter.#c; } }
assert.sameValue(Counter.bump(), 1, "static private");
assert.sameValue(Counter.bump(), 2, "static private again");
class M { #g() { return 7; } static run(o) { var f = o.#g; return f.call(o); } }
assert.sameValue(M.run(new M()), 7, "read-then-call private method");
