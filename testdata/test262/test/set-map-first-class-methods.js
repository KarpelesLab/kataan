/*---
description: Set/Map methods are first-class values (feature-detectable, callable detached); ES2025 Set ops
features: [Set]
---*/
var a = new Set([1, 2, 3]);
var b = new Set([2, 3, 4]);

// Feature detection: methods read as values.
assert.sameValue(typeof a.union, "function", "typeof Set#union");
assert.sameValue(typeof a.intersection, "function", "typeof Set#intersection");
assert.sameValue(typeof a.has, "function", "typeof Set#has");
assert.sameValue(!!a.isSubsetOf, true, "if (set.isSubsetOf)");
assert.sameValue(a.nonexistentMethod, undefined, "missing method is undefined");
assert.sameValue(typeof new Map().set, "function", "typeof Map#set");
assert.sameValue(typeof new Map().get, "function", "typeof Map#get");

// Set.prototype is a real object linking back to Set.
assert.sameValue(typeof Set.prototype, "object", "Set.prototype");
assert.sameValue(typeof Set.prototype.union, "function", "Set.prototype.union");
assert.sameValue(Set.prototype.constructor, Set, "Set.prototype.constructor");

// ES2025 Set composition still works when called.
assert.sameValue([...a.union(b)].sort().join(","), "1,2,3,4", "union");
assert.sameValue([...a.intersection(b)].sort().join(","), "2,3", "intersection");
assert.sameValue([...a.difference(b)].sort().join(","), "1", "difference");
assert.sameValue([...a.symmetricDifference(b)].sort().join(","), "1,4", "symmetricDifference");
assert.sameValue(new Set([1, 2]).isSubsetOf(a), true, "isSubsetOf");
assert.sameValue(a.isSupersetOf(new Set([1, 2])), true, "isSupersetOf");
assert.sameValue(a.isDisjointFrom(new Set([5, 6])), true, "isDisjointFrom");

// A method read from the prototype is callable detached with an explicit this.
var u = Set.prototype.union;
assert.sameValue([...u.call(a, b)].sort().join(","), "1,2,3,4", "detached union");

// Ordinary methods are unaffected.
assert.sameValue(a.has(2), true, "has");
assert.sameValue(a.size, 3, "size");
