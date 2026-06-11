/*---
description: a Proxy set trap returning a falsy value throws in strict mode
esid: sec-proxy-object-internal-methods-and-internal-slots-set-p-v-receiver
---*/
function throwsType(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// A set trap returning false (or any falsy) is a failed [[Set]] → TypeError in strict.
// ("use strict" is each function's own directive, so the function is strict regardless
// of how the test is loaded.)
assert.sameValue(throwsType(function () { "use strict"; var p = new Proxy({}, { set: function () { return false; } }); p.x = 1; }), true, "static key, false");
assert.sameValue(throwsType(function () { "use strict"; var p = new Proxy({}, { set: function () { return false; } }); p["y"] = 1; }), true, "computed key, false");
assert.sameValue(throwsType(function () { "use strict"; var p = new Proxy({}, { set: function () {} }); p.x = 1; }), true, "undefined return");

// A trap returning true succeeds.
var log = [];
var ok = new Proxy({}, { set: function (t, k, v) { log.push(k + "=" + v); t[k] = v; return true; } });
(function () { "use strict"; ok.a = 5; })();
assert.sameValue(ok.a, 5, "truthy trap applies");
assert.sameValue(log.join(","), "a=5", "trap ran");

// No trap forwards to the target (succeeds).
var fwd = new Proxy({}, {});
(function () { "use strict"; fwd.b = 9; })();
assert.sameValue(fwd.b, 9, "forwarded set");
