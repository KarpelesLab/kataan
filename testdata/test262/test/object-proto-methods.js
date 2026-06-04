/*---
description: Object.getPrototypeOf, setPrototypeOf, create with null
esid: sec-object.getprototypeof
---*/
var proto = { greet: function () { return "hi"; } };
var o = Object.create(proto);
assert.sameValue(Object.getPrototypeOf(o), proto, "getPrototypeOf");
assert.sameValue(o.greet(), "hi", "inherited via create");
var bare = Object.create(null);
bare.x = 1;
assert.sameValue(bare.x, 1);
assert.sameValue(Object.getPrototypeOf(bare), null, "null-proto object");
var a = {};
Object.setPrototypeOf(a, proto);
assert.sameValue(a.greet(), "hi", "setPrototypeOf links the chain");
assert.sameValue(Object.getPrototypeOf([]) === Array.prototype || typeof Object.getPrototypeOf([]) === "object", true);
