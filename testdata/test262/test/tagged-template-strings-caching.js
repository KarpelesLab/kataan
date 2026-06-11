/*---
description: a tagged template's strings array is cached per template-literal site
esid: sec-gettemplateobject
---*/
function tag(strings) { return strings; }

// The same template-literal site yields the identical (frozen) strings object on every
// evaluation — this identity is what template-caching libraries rely on.
function evalSite() { return tag`x${1}y`; }
assert.sameValue(evalSite() === evalSite(), true, "same site, same object");

// Re-evaluating the site in a loop returns the one cached object.
var seen = new Set();
for (let i = 0; i < 3; i++) seen.add(tag`loop${i}`);
assert.sameValue(seen.size, 1, "loop site cached");

// Two distinct sites produce distinct objects.
var s1 = tag`a${1}`;
var s2 = tag`a${1}`;
assert.sameValue(s1 === s2, false, "distinct sites, distinct objects");

// The interpolated values are still recomputed on each evaluation.
function values(strings) { var v = []; for (var i = 1; i < arguments.length; i++) v.push(arguments[i]); return v[0]; }
assert.sameValue(values`x${10}y`, 10, "value 1");
assert.sameValue(values`x${20}y`, 20, "value 2");

// The cached object (and its .raw) are frozen, and .raw is preserved.
var obj = tag`a\nb${1}`;
assert.sameValue(Object.isFrozen(obj), true, "strings frozen");
assert.sameValue(Object.isFrozen(obj.raw), true, "raw frozen");
assert.sameValue(obj.raw[0], "a\\nb", "raw value");

// String.raw still works.
assert.sameValue(String.raw`C:\path`, "C:\\path", "String.raw");
