/*---
description: Chained getters and computed property access
esid: sec-property-accessors
---*/
var config = {
  _settings: { theme: "dark", size: 12 },
  get settings() { return this._settings; },
  get theme() { return this.settings.theme; }
};
assert.sameValue(config.theme, "dark", "chained getter");
assert.sameValue(config.settings.size, 12);
var data = { a: { b: { c: { d: 42 } } } };
assert.sameValue(data.a.b.c.d, 42, "deep access");
assert.sameValue(data["a"]["b"]["c"]["d"], 42, "computed deep access");
var keys = ["a", "b", "c", "d"];
var cur = data;
for (var i = 0; i < keys.length; i++) cur = cur[keys[i]];
assert.sameValue(cur, 42, "dynamic path traversal");
var matrix = [[1, 2], [3, 4]];
assert.sameValue(matrix[1][0], 3);
