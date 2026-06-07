/*---
description: RegExp flag accessor properties (dotAll, unicode, hasIndices, ...)
features: [regexp-dotall, regexp-named-groups, regexp-match-indices]
---*/
var r = /x/gimsuy;
assert.sameValue(r.global, true, "global");
assert.sameValue(r.ignoreCase, true, "ignoreCase");
assert.sameValue(r.multiline, true, "multiline");
assert.sameValue(r.sticky, true, "sticky");
assert.sameValue(r.dotAll, true, "dotAll");
assert.sameValue(r.unicode, true, "unicode");
assert.sameValue(r.flags, "gimsuy", "flags string");

// hasIndices (d flag).
assert.sameValue((/x/d).hasIndices, true, "hasIndices true");
assert.sameValue((/x/).hasIndices, false, "hasIndices false");

// A flagless regexp reports false for each (not undefined).
var plain = /x/;
assert.sameValue(plain.dotAll, false, "plain dotAll");
assert.sameValue(plain.unicode, false, "plain unicode");
assert.sameValue(plain.global, false, "plain global");

// The s flag actually changes matching (dot matches newline).
assert.sameValue(/a.b/s.test("a\nb"), true, "dotAll matches newline");
assert.sameValue(/a.b/.test("a\nb"), false, "without dotAll, dot doesn't match newline");
