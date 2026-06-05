/*---
description: JSON.stringify with special values and structures
esid: sec-json.stringify
---*/
assert.sameValue(JSON.stringify(undefined), undefined, "undefined is not valid JSON");
assert.sameValue(JSON.stringify(function () {}), undefined, "function is not valid JSON");
assert.sameValue(JSON.stringify(NaN), "null", "NaN becomes null");
assert.sameValue(JSON.stringify(Infinity), "null");
assert.sameValue(JSON.stringify([undefined, function () {}, NaN]), "[null,null,null]", "array holes");
assert.sameValue(JSON.stringify({ a: undefined, b: function () {}, c: 1 }), '{"c":1}', "object omits");
assert.sameValue(JSON.stringify({ nested: { deep: [1, 2, 3] } }), '{"nested":{"deep":[1,2,3]}}');
assert.sameValue(JSON.stringify("with \"quotes\""), '"with \\"quotes\\""', "escapes quotes");
assert.sameValue(JSON.stringify("line\nbreak"), '"line\\nbreak"', "escapes newline");
assert.sameValue(JSON.stringify({ "key with spaces": 1 }), '{"key with spaces":1}');
assert.sameValue(JSON.stringify(true), "true");
assert.sameValue(JSON.stringify(null), "null");
assert.sameValue(JSON.stringify([]), "[]");
assert.sameValue(JSON.stringify({}), "{}");
assert.sameValue(JSON.stringify([1, [2, [3]]]), "[1,[2,[3]]]");
