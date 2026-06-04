/*---
description: Proxy.revocable returns {proxy, revoke}; a revoked proxy throws
esid: sec-proxy.revocable
---*/
var r = Proxy.revocable({ a: 1 }, {
  get: function (t, k) { return t[k]; }
});
assert.sameValue(typeof r.proxy, "object");
assert.sameValue(typeof r.revoke, "function");
assert.sameValue(r.proxy.a, 1, "works before revocation");

r.revoke();

var threw = false;
try { var x = r.proxy.a; } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "reading a revoked proxy throws a TypeError");

var setThrew = false;
try { r.proxy.b = 2; } catch (e) { setThrew = true; }
assert.sameValue(setThrew, true, "writing a revoked proxy throws");

// Revoking twice is harmless.
r.revoke();
