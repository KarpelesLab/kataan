/*---
description: a tagged template's strings array and its .raw are frozen
esid: sec-gettemplateobject
---*/
var captured;
function tag(strings) { captured = strings; return strings; }

var s = tag`a\n${1}b`;
assert.sameValue(Object.isFrozen(s), true, "strings array is frozen");
assert.sameValue(Object.isFrozen(s.raw), true, ".raw is frozen");

// Cooked vs raw still readable.
assert.sameValue(s[0], "a\n", "cooked first chunk");
assert.sameValue(s.raw[0], "a\\n", "raw first chunk");
assert.sameValue(s.length, 2, "two cooked chunks");

// Mutation attempts are silently rejected (the array is immutable).
s[0] = "X";
assert.sameValue(s[0], "a\n", "element write rejected");
try { s.push("Y"); } catch (e) {}
assert.sameValue(s.length, 2, "push rejected");
s.raw[0] = "Z";
assert.sameValue(s.raw[0], "a\\n", "raw element write rejected");

// A normal tag function still receives interpolated values.
function vals(strings) { var v = []; for (var i = 1; i < arguments.length; i++) v.push(arguments[i]); return v.join(","); }
assert.sameValue(vals`${10}x${20}`, "10,20", "interpolations passed");

// String.raw still works on the frozen object.
assert.sameValue(String.raw`a\tb${1}c`, "a\\tb1c", "String.raw");
