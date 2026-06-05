/*---
description: Computed property keys in destructuring patterns
esid: sec-destructuring-binding-patterns
---*/
var key = "name";
var { [key]: value } = { name: "Alice" };
assert.sameValue(value, "Alice", "computed key in destructuring");
var prop = "x";
var obj = { x: 1, y: 2 };
var { [prop]: extracted, ...rest } = obj;
assert.sameValue(extracted, 1);
assert.sameValue(rest.y, 2);
var idx = 1;
var arr = [10, 20, 30];
var picked = arr[idx];
assert.sameValue(picked, 20);
