/*---
description: Optional chaining with calls, indexing, and assignment guards
esid: sec-optional-chains
---*/
var data = { users: [{ name: "Ann", roles: ["admin"] }] };
assert.sameValue(data?.users?.[0]?.name, "Ann");
assert.sameValue(data?.users?.[0]?.roles?.[0], "admin");
assert.sameValue(data?.users?.[5]?.name, undefined, "out of bounds short-circuits");
assert.sameValue(data?.missing?.deep?.value, undefined);
var api = { get: function (k) { return k.toUpperCase(); } };
assert.sameValue(api.get?.("x"), "X");
assert.sameValue(api.post?.("x"), undefined, "missing method call");
var n = null;
assert.sameValue(n?.anything, undefined);
assert.sameValue(n?.method?.(), undefined);
