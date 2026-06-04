/*---
description: Proxy apply and construct traps, with default forwarding
esid: sec-proxy-object-internal-methods-and-internal-slots-call-thisargument-argumentslist
---*/
function add(a, b) { return a + b; }
var p = new Proxy(add, {
  apply: function (target, thisArg, args) { return args[0] + args[1] + 1000; }
});
assert.sameValue(p(2, 3), 1005, "apply trap intercepts the call");

var plainCall = new Proxy(add, {});
assert.sameValue(plainCall(4, 5), 9, "no apply trap forwards to the target");
assert.sameValue(typeof plainCall, "function", "a function proxy is callable");

function Box(v) { this.v = v; }
var cp = new Proxy(Box, {
  construct: function (target, args) { return { v: args[0] * 2, trapped: true }; }
});
var made = new cp(21);
assert.sameValue(made.v, 42, "construct trap intercepts new");
assert.sameValue(made.trapped, true);

var plainNew = new Proxy(Box, {});
assert.sameValue(new plainNew(7).v, 7, "no construct trap forwards to the target");
