/*---
description: Array element access via canonical numeric string keys
esid: sec-array-exotic-objects
---*/
var a = [10, 20, 30];
assert.sameValue(a["0"], 10, "string key 0");
assert.sameValue(a["1"], 20);
assert.sameValue(a["2"], 30);
assert.sameValue(a["3"], undefined, "out of range");
var k = "1";
assert.sameValue(a[k], 20, "computed string key");
assert.sameValue(a["00"], undefined, "non-canonical key is not an index");
assert.sameValue(a["01"], undefined, "leading zero is not an index");
assert.sameValue(a["length"], 3, "length still works");
assert.sameValue(a[0], a["0"], "numeric and string keys agree");
var keys = ["0", "1", "2"];
assert.sameValue(keys.map(function (i) { return a[i]; }).join(","), "10,20,30", "string keys in map");
var obj = { 0: "x", 1: "y" };
assert.sameValue(obj["0"], "x", "plain object string key");
var matrix = [[1, 2], [3, 4]];
assert.sameValue(matrix["0"]["1"], 2, "nested string indexing");
