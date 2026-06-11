/*---
description: adding a new property to a non-extensible object fails (strict throws, sloppy drops)
esid: sec-ordinaryset
---*/
// Strict mode (per-function, since the harness is prepended).
function strictThrows(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }
assert.sameValue(strictThrows(function () { "use strict"; var o = Object.preventExtensions({}); o.x = 1; }), true, "preventExtensions: new property -> TypeError");
assert.sameValue(strictThrows(function () { "use strict"; var o = Object.seal({ a: 1 }); o.b = 2; }), true, "seal: new property -> TypeError");
assert.sameValue(strictThrows(function () { "use strict"; var o = Object.freeze({ a: 1 }); o.b = 2; }), true, "freeze: new property -> TypeError");

// An existing writable property is still assignable on a non-extensible (non-frozen) object.
assert.sameValue((function () { "use strict"; var o = Object.preventExtensions({ a: 1 }); o.a = 5; return o.a; })(), 5, "existing property on preventExtensions");
assert.sameValue((function () { "use strict"; var o = Object.seal({ a: 1 }); o.a = 9; return o.a; })(), 9, "sealed property is writable");
// Frozen properties are non-writable.
assert.sameValue(strictThrows(function () { "use strict"; var o = Object.freeze({ a: 1 }); o.a = 9; }), true, "frozen property is read-only");

// Sloppy mode: the add is silently dropped (no throw).
var o = Object.preventExtensions({});
o.x = 1;
assert.sameValue("x" in o, false, "sloppy add to non-extensible is dropped");
assert.sameValue(Object.isExtensible(o), false, "isExtensible reflects it");

// A normal (extensible) object accepts new properties.
var n = {};
n.y = 7;
assert.sameValue(n.y, 7, "extensible object accepts new property");
