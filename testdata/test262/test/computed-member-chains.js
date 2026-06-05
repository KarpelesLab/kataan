/*---
description: Computed member access chains and dynamic property paths
esid: sec-property-accessors
---*/
var data = { users: { alice: { age: 30 }, bob: { age: 25 } } };
assert.sameValue(data["users"]["alice"]["age"], 30);
var path = ["users", "bob", "age"];
var cur = data;
for (var i = 0; i < path.length; i++) cur = cur[path[i]];
assert.sameValue(cur, 25, "dynamic path traversal");
var matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
assert.sameValue(matrix[1][2], 6);
var sum = 0;
for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) sum += matrix[r][c];
assert.sameValue(sum, 45);
var obj = {};
var key = "dynamic";
obj[key] = "value";
obj[key + "2"] = "value2";
assert.sameValue(obj.dynamic, "value");
assert.sameValue(obj.dynamic2, "value2");
var methods = { add: function (a, b) { return a + b; }, sub: function (a, b) { return a - b; } };
var op = "add";
assert.sameValue(methods[op](5, 3), 8, "computed method call");
var nested = { a: [{ b: [{ c: 42 }] }] };
assert.sameValue(nested.a[0].b[0].c, 42, "deep mixed access");
