/*---
description: Object spread in literals and rest in destructuring
features: [object-spread, object-rest]
---*/
var base = { x: 1, y: 2 };
var ext = { ...base, z: 3 };
assert.sameValue(ext.x, 1, "spread copies x");
assert.sameValue(ext.z, 3, "own property added after spread");
assert.sameValue(Object.keys(ext).length, 3, "three keys total");

// Later properties override spread-in ones.
var over = { ...base, x: 99 };
assert.sameValue(over.x, 99, "later property wins over spread");

// Object rest in destructuring gathers the remaining own enumerable keys.
var { x, ...others } = { x: 1, y: 2, z: 3 };
assert.sameValue(x, 1, "destructured x");
assert.sameValue(others.y, 2, "rest gathers y");
assert.sameValue(others.z, 3, "rest gathers z");
assert.sameValue(Object.keys(others).length, 2, "rest excludes the named key");
