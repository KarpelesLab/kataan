/*---
description: Intl.ListFormat formats lists with conjunction/disjunction/unit patterns (en-US)
features: [Intl.ListFormat]
---*/
function lf(o, a) { return new Intl.ListFormat("en", o).format(a); }

assert.sameValue(typeof Intl.ListFormat, "function", "Intl.ListFormat exists");

// Conjunction (default): Oxford comma for 3+ items.
assert.sameValue(lf({}, ["a", "b", "c"]), "a, b, and c", "3-item conjunction");
assert.sameValue(lf({}, ["a", "b"]), "a and b", "2-item conjunction (no comma)");
assert.sameValue(lf({}, ["a"]), "a", "1 item");
assert.sameValue(lf({}, []), "", "empty list");
assert.sameValue(lf({}, ["a", "b", "c", "d"]), "a, b, c, and d", "4-item conjunction");

// Disjunction: "or".
assert.sameValue(lf({ type: "disjunction" }, ["a", "b", "c"]), "a, b, or c", "3-item disjunction");
assert.sameValue(lf({ type: "disjunction" }, ["a", "b"]), "a or b", "2-item disjunction");

// Unit: comma-joined, no conjunction word.
assert.sameValue(lf({ type: "unit" }, ["5 ft", "7 in"]), "5 ft, 7 in", "2-item unit");
assert.sameValue(lf({ type: "unit" }, ["a", "b", "c"]), "a, b, c", "3-item unit");

// The instance's format is a readable function, and works via new and without new.
assert.sameValue(typeof new Intl.ListFormat("en").format, "function", "format is readable");
assert.sameValue(Intl.ListFormat("en").format(["x", "y"]), "x and y", "callable without new");
