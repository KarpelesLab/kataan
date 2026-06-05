/*---
description: Object.defineProperty enforces non-configurable and non-extensible invariants
esid: sec-object.defineproperty
---*/
var o = {};
Object.defineProperty(o, "x", { value: 1, configurable: false });
var threw = false;
try { Object.defineProperty(o, "x", { value: 2 }); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "redefining a non-configurable property throws");
assert.sameValue(o.x, 1, "value unchanged after the failed redefine");
var c = {};
Object.defineProperty(c, "y", { value: 1, configurable: true });
Object.defineProperty(c, "y", { value: 2, configurable: true });
assert.sameValue(c.y, 2, "configurable property can be redefined");
var ne = {};
Object.preventExtensions(ne);
var threw2 = false;
try { Object.defineProperty(ne, "z", { value: 1 }); } catch (e) { threw2 = e instanceof TypeError; }
assert.sameValue(threw2, true, "defining a new property on a non-extensible object throws");
assert.sameValue("z" in ne, false, "property not added");
assert.sameValue(Object.defineProperty({}, "k", { value: 5 }).k, 5, "defineProperty returns the object");
var existing = {};
Object.defineProperty(existing, "w", { value: 1, writable: true, configurable: false });
Object.defineProperty(existing, "w", { value: 2, writable: true, configurable: false });
assert.sameValue(existing.w, 2, "non-configurable but writable allows value change");
