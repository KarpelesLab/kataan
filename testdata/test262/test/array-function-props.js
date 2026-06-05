/*---
description: Arrays and functions can hold named properties
esid: sec-array-exotic-objects
---*/
var a = [1, 2, 3];
a.label = "nums";
assert.sameValue(a.label, "nums", "array holds a named property");
assert.sameValue(a.length, 3, "indexed storage is unaffected");
assert.sameValue(a[0], 1);
assert.sameValue(a.hasOwnProperty("label"), true);

function counter() { counter.calls = (counter.calls || 0) + 1; return counter.calls; }
counter(); counter();
assert.sameValue(counter.calls, 2, "function holds mutable state");

// Tagged template raw strings (the strings object carries `.raw`).
function tag(strings, ...values) { return strings.raw.join("|") + "#" + values.join(","); }
assert.sameValue(tag`a\t${1}b${2}`, "a\\t|b|#1,2", "strings.raw preserves escapes");
assert.sameValue(tag`plain`, "plain#");
