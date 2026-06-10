/*---
description: a constructor that returns an object overrides the instance; new on arrow/generator/async throws
esid: sec-runtime-semantics-evaluatenew
---*/
// A class constructor returning an object overrides the new instance.
class C { constructor() { this.x = 1; return { x: 99 }; } }
assert.sameValue(new C().x, 99, "object return overrides (class)");
class P { constructor() { this.x = 5; return 42; } }
assert.sameValue(new P().x, 5, "primitive return ignored (class)");
class A { constructor() { this.x = 1; return [10, 20]; } }
assert.sameValue(new A()[0], 10, "array return overrides");
class N { constructor() { this.x = 7; } }
assert.sameValue(new N().x, 7, "no return -> instance");

// A function constructor does the same.
function F() { this.x = 1; return { x: 88 }; }
assert.sameValue(new F().x, 88, "object return overrides (function)");

// A derived class object return overrides the (super-built) instance.
class Base { constructor() { this.a = 1; } }
class Derived extends Base { constructor() { super(); this.b = 2; return { custom: true }; } }
var d = new Derived();
assert.sameValue(d.custom, true, "derived object return");
assert.sameValue(d.b, undefined, "instance replaced");

// new on a non-constructor (arrow / generator / async) is a TypeError.
assert.throws(TypeError, function () { return new (() => {})(); }, "new arrow");
assert.throws(TypeError, function () { return new (function* () {})(); }, "new generator");
assert.throws(TypeError, function () { return new (async function () {})(); }, "new async");

// Ordinary functions and classes still construct.
function Ok() { this.v = 3; }
assert.sameValue(new Ok().v, 3, "function constructs");
assert.sameValue(new (class { constructor() { this.w = 4; } })().w, 4, "class constructs");
