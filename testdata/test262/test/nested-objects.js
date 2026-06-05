/*---
description: Deeply nested object construction and access
esid: sec-object-initializer
---*/
var config = {
  server: { host: "localhost", port: 8080, options: { timeout: 30, retries: 3 } },
  database: { name: "test", pool: { min: 2, max: 10 } }
};
assert.sameValue(config.server.host, "localhost");
assert.sameValue(config.server.options.timeout, 30);
assert.sameValue(config.database.pool.max, 10);
assert.sameValue(config["server"]["port"], 8080);
config.server.options.retries = 5;
assert.sameValue(config.server.options.retries, 5, "deep mutation");
var tree = { value: 1, children: [{ value: 2, children: [] }, { value: 3, children: [{ value: 4, children: [] }] }] };
assert.sameValue(tree.children[1].children[0].value, 4, "tree traversal");
function sumTree(node) {
  return node.value + node.children.reduce(function (acc, c) { return acc + sumTree(c); }, 0);
}
assert.sameValue(sumTree(tree), 10, "recursive tree sum");
var matrix = { rows: [[1, 2], [3, 4]], get sum() { return this.rows.flat().reduce(function (a, b) { return a + b; }, 0); } };
assert.sameValue(matrix.sum, 10, "computed property over nested");
