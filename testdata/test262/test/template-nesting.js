/*---
description: Nested template literals and expression interpolation
esid: sec-template-literals
---*/
var name = "World";
var greeting = `Hello, ${name}!`;
assert.sameValue(greeting, "Hello, World!");
var items = [1, 2, 3];
assert.sameValue(`Sum: ${items.reduce(function (a, b) { return a + b; }, 0)}`, "Sum: 6");
var x = 5;
assert.sameValue(`${x > 0 ? `positive (${x})` : "non-positive"}`, "positive (5)", "nested template in ternary");
var obj = { a: 1, b: 2 };
assert.sameValue(`Keys: ${Object.keys(obj).join(", ")}`, "Keys: a, b");
assert.sameValue(`${`${`${1}`}`}`, "1", "triple-nested");
