/*---
description: Template literals with expressions, nesting, and escapes
esid: sec-template-literals
---*/
var name = "world";
assert.sameValue(`hello ${name}!`, "hello world!");
assert.sameValue(`1 + 2 = ${1 + 2}`, "1 + 2 = 3");
var a = 5, b = 3;
assert.sameValue(`${a > b ? "big" : "small"}`, "big");
assert.sameValue(`outer ${`inner ${a}`}`, "outer inner 5", "nested templates");
assert.sameValue(`line1\nline2`, "line1\nline2");
