/*---
description: Object.create, prototype chains, getPrototypeOf, setPrototypeOf
esid: sec-object.create
---*/
var proto = {
  greet: function () { return "hi " + this.name; },
  kind: "base"
};
var o = Object.create(proto);
o.name = "x";
assert.sameValue(o.name, "x");
assert.sameValue(o.greet(), "hi x", "inherited method binds this to the receiver");
assert.sameValue(o.kind, "base", "inherited data property");
assert.sameValue(Object.keys(o).join(","), "name", "inherited keys are not own");
assert.sameValue(Object.getPrototypeOf(o) === proto, true);

var bare = Object.create(null);
assert.sameValue(Object.getPrototypeOf(bare), null);

var o2 = {};
Object.setPrototypeOf(o2, proto);
assert.sameValue(o2.kind, "base");

// A two-level chain.
var mid = Object.create(proto);
mid.kind = "mid";
var leaf = Object.create(mid);
assert.sameValue(leaf.kind, "mid", "nearest prototype wins");
assert.sameValue(leaf.greet === proto.greet, true, "reaches the top of the chain");
