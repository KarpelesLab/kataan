/*---
description: Template literals interpolate and concatenate
esid: sec-template-literals
---*/
var name = "world";
assert.sameValue(`hello ${name}`, "hello world");
assert.sameValue(`${1 + 2} = ${"three"}`, "3 = three");
var n = 3;
assert.sameValue(`${n} item${n === 1 ? "" : "s"}`, "3 items");
