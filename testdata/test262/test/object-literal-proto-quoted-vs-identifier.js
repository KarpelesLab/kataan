/*---
description: a non-computed __proto__ key in an object literal sets the prototype
esid: sec-__proto__-property-names-in-object-initializers
---*/
var proto = { greet: function () { return "P"; } };

// Unquoted identifier form sets the [[Prototype]] (and is not an own property).
var a = { __proto__: proto, own: 1 };
assert.sameValue(Object.getPrototypeOf(a), proto, "unquoted sets prototype");
assert.sameValue(a.greet(), "P", "inherited method");
assert.sameValue(Object.keys(a).join(","), "own", "__proto__ is not an own key");

// A quoted (string-literal) key is not computed, so it is a prototype setter
// too (PropertyDefinitionEvaluation: only IsComputedPropertyKey opts out).
var b = { "__proto__": proto };
assert.sameValue(Object.getPrototypeOf(b) === proto, true, "quoted sets prototype");
assert.sameValue(Object.keys(b).length, 0, "quoted is not an own key");
assert.sameValue(JSON.stringify({ "__proto__": 5 }), '{}', "a non-object value is ignored");

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
