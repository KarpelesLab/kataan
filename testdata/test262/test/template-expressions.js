/*---
description: Template literals with complex expressions
esid: sec-template-literals
---*/
var a = 5, b = 3;
assert.sameValue(`${a} + ${b} = ${a + b}`, "5 + 3 = 8");
assert.sameValue(`${a > b ? "bigger" : "smaller"}`, "bigger", "ternary in template");
var arr = [1, 2, 3];
assert.sameValue(`Items: ${arr.join(", ")}`, "Items: 1, 2, 3");
assert.sameValue(`${arr.map(function (x) { return x * 2; }).join("-")}`, "2-4-6");
var obj = { name: "test", count: 42 };
assert.sameValue(`${obj.name}: ${obj.count}`, "test: 42");
assert.sameValue(`Length is ${"hello".length}`, "Length is 5");
function greet(name) { return `Hello, ${name}!`; }
assert.sameValue(greet("World"), "Hello, World!");
assert.sameValue(`${1}${2}${3}`, "123", "adjacent expressions");
assert.sameValue(`nested ${`inner ${a}`}`, "nested inner 5");
var multiline = `line1
line2`;
assert.sameValue(multiline.split("\n").length, 2, "multiline template");
assert.sameValue(`${true}${null}${undefined}`, "truenullundefined", "coercion in template");
