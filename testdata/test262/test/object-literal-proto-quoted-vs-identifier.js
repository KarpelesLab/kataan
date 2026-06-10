/*---
description: only the unquoted __proto__ identifier in an object literal sets the prototype
esid: sec-__proto__-property-names-in-object-initializers
---*/
var proto = { greet: function () { return "P"; } };

// Unquoted identifier form sets the [[Prototype]] (and is not an own property).
var a = { __proto__: proto, own: 1 };
assert.sameValue(Object.getPrototypeOf(a), proto, "unquoted sets prototype");
assert.sameValue(a.greet(), "P", "inherited method");
assert.sameValue(Object.keys(a).join(","), "own", "__proto__ is not an own key");

// Quoted string key creates an ordinary own data property.
var b = { "__proto__": proto };
assert.sameValue(Object.getPrototypeOf(b) === proto, false, "quoted does not set prototype");
assert.sameValue(b.__proto__, proto, "own __proto__ data property readable");
assert.sameValue(Object.keys(b).length, 1, "quoted is an own key");
assert.sameValue(JSON.stringify({ "__proto__": 5 }), '{"__proto__":5}', "quoted serializes");

// Computed key likewise makes a data property.
var k = "__proto__";
var c = { [k]: proto };
assert.sameValue(Object.getPrototypeOf(c) === proto, false, "computed does not set prototype");
assert.sameValue(Object.keys(c).length, 1, "computed is an own key");

// __proto__: null sets a null prototype.
assert.sameValue(Object.getPrototypeOf({ __proto__: null }), null, "null prototype");

// __proto__ still acts as the accessor on ordinary objects.
var e = {};
assert.sameValue(e.__proto__, Object.getPrototypeOf(e), "__proto__ accessor");
