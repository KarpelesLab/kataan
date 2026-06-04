/*---
description: trimStart/trimEnd, split with limit, includes edge cases
esid: sec-properties-of-the-string-prototype-object
---*/
assert.sameValue("  hi  ".trimStart(), "hi  ");
assert.sameValue("  hi  ".trimEnd(), "  hi");
assert.sameValue("a,b,c,d".split(",", 2).length, 2);
assert.sameValue("aXbXc".split("X").join("-"), "a-b-c");
assert.sameValue("".split(",").length, 1);
