/*---
description: Template literal coercion of various value types
esid: sec-template-literals
---*/
assert.sameValue(`${1 + 1}`, "2");
assert.sameValue(`${true}`, "true");
assert.sameValue(`${null}`, "null");
assert.sameValue(`${undefined}`, "undefined");
assert.sameValue(`${[1, 2, 3]}`, "1,2,3", "array to string");
assert.sameValue(`${{ toString: function () { return "custom"; } }}`, "custom");
assert.sameValue(`a${1}b${2}c${3}d`, "a1b2c3d");
assert.sameValue(`${"nested " + `inner ${5}`}`, "nested inner 5");
