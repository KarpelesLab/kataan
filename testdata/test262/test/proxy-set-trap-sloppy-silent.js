/*---
description: a Proxy set trap returning false fails silently in sloppy mode
esid: sec-proxy-object-internal-methods-and-internal-slots-set-p-v-receiver
flags: [noStrict]
---*/
// In sloppy mode a failed [[Set]] does not throw.
var p = new Proxy({}, { set: function () { return false; } });
var threw = false;
try { p.x = 1; } catch (e) { threw = true; }
assert.sameValue(threw, false, "sloppy set does not throw");
assert.sameValue(p.x, undefined, "value not stored");
