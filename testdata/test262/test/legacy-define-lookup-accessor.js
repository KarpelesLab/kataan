/*---
description: Object.prototype.__defineGetter__/__defineSetter__/__lookupGetter__/__lookupSetter__
esid: sec-object.prototype.__defineGetter__
---*/
var o = {};
o.__defineGetter__("x", function () { return 42; });
assert.sameValue(o.x, 42, "__defineGetter__");

var store = 0;
o.__defineSetter__("y", function (v) { store = v * 2; });
o.y = 5;
assert.sameValue(store, 10, "__defineSetter__");

// __lookupGetter__/__lookupSetter__ return the function (or undefined).
var fn = function () { return 7; };
var o2 = {};
o2.__defineGetter__("z", fn);
assert.sameValue(o2.__lookupGetter__("z"), fn, "lookupGetter returns the function");
assert.sameValue(typeof o2.__lookupSetter__("z"), "undefined", "no setter -> undefined");
assert.sameValue(typeof o2.__lookupGetter__("absent"), "undefined", "absent -> undefined");

// A getter+setter pair on the same key coexist.
var v = 0;
var o3 = {};
o3.__defineGetter__("p", function () { return v; });
o3.__defineSetter__("p", function (x) { v = x + 1; });
o3.p = 10;
assert.sameValue(o3.p, 11, "getter+setter pair");

// Lookup walks the prototype chain.
var base = {};
base.__defineGetter__("inh", function () { return "base"; });
var child = Object.create(base);
assert.sameValue(child.inh, "base", "inherited getter invoked");
assert.sameValue(typeof child.__lookupGetter__("inh"), "function", "inherited lookup");
