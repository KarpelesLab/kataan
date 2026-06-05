/*---
description: JSON.stringify/parse round trips with nesting and types
esid: sec-json
---*/
var data = {
  name: "test",
  count: 42,
  active: true,
  tags: ["a", "b", "c"],
  nested: { x: 1, y: [2, 3], z: { deep: true } },
  nothing: null
};
var json = JSON.stringify(data);
var parsed = JSON.parse(json);
assert.sameValue(parsed.name, "test");
assert.sameValue(parsed.count, 42);
assert.sameValue(parsed.active, true);
assert.sameValue(parsed.tags.join(","), "a,b,c");
assert.sameValue(parsed.nested.x, 1);
assert.sameValue(parsed.nested.y[1], 3);
assert.sameValue(parsed.nested.z.deep, true);
assert.sameValue(parsed.nothing, null);
assert.sameValue(JSON.stringify([1, [2, [3, [4]]]]), "[1,[2,[3,[4]]]]");
assert.sameValue(JSON.parse("[1,2,3]").reduce(function (a, b) { return a + b; }, 0), 6);
var matrix = [[1, 2, 3], [4, 5, 6]];
assert.sameValue(JSON.parse(JSON.stringify(matrix))[1][2], 6);
assert.sameValue(JSON.stringify({ a: 1, b: 2 }, null, 2).indexOf("\n") >= 0, true, "pretty print");
assert.sameValue(JSON.stringify("special: \"quotes\" and \\backslash"), '"special: \\"quotes\\" and \\\\backslash"');
