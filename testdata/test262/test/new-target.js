/*---
description: new.target reflects whether a function was invoked via new
features: [new.target]
---*/
// A constructor invoked with `new` sees itself; a plain call sees undefined.
function F() { return new.target; }
assert.sameValue(new F() instanceof F, true, "new F() constructs");
assert.sameValue(F(), undefined, "plain call: new.target is undefined");

function G() { this.viaNew = new.target === G; }
assert.sameValue(new G().viaNew, true, "new.target === constructor inside new");

// Classes: new.target is the class actually constructed (incl. subclasses).
class Base { constructor() { this.who = new.target.name; } }
class Sub extends Base {}
assert.sameValue(new Base().who, "Base", "base class new.target");
assert.sameValue(new Sub().who, "Sub", "subclass new.target through super");

// Arrow functions inherit new.target from the enclosing function.
function H() { this.t = (() => new.target === H)(); }
assert.sameValue(new H().t, true, "arrow inherits new.target");

// A nested ordinary call resets new.target to undefined.
function Outer() { this.inner = (function () { return new.target; })(); }
assert.sameValue(new Outer().inner, undefined, "nested call resets new.target");
