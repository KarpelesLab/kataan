/*---
description: encodeURI/decodeURI/encodeURIComponent/decodeURIComponent
esid: sec-encodeuricomponent-uricomponent
---*/
assert.sameValue(encodeURIComponent("a b&c=d"), "a%20b%26c%3Dd", "component encodes reserved chars");
assert.sameValue(decodeURIComponent("a%20b%26c"), "a b&c", "component decodes");
assert.sameValue(encodeURI("http://a.com/x y?q=1&r=2"), "http://a.com/x%20y?q=1&r=2", "encodeURI preserves reserved");
assert.sameValue(decodeURI("http://a.com/x%20y"), "http://a.com/x y", "decodeURI");
// Round-trips, including non-ASCII (UTF-8 percent bytes).
assert.sameValue(encodeURIComponent("café"), "caf%C3%A9", "UTF-8 multi-byte encode");
assert.sameValue(decodeURIComponent("caf%C3%A9"), "café", "UTF-8 multi-byte decode");
assert.sameValue(decodeURIComponent(encodeURIComponent("hello world / 100%")), "hello world / 100%", "component round-trip");
assert.sameValue(decodeURI(encodeURI("a/b?c=d e")), "a/b?c=d e", "encodeURI round-trip");
// The unreserved set is preserved.
assert.sameValue(encodeURIComponent("-_.!~*'()"), "-_.!~*'()", "unreserved set kept");
assert.sameValue(encodeURIComponent("AZaz09"), "AZaz09", "alphanumerics kept");
// Lowercase percent-escapes still decode.
assert.sameValue(decodeURIComponent("%2f"), "/", "lowercase hex escape");
// A malformed escape throws a URIError-like (TypeError here).
var threw = false;
try { decodeURIComponent("%zz"); } catch (e) { threw = true; }
assert.sameValue(threw, true, "malformed escape throws");
// globalThis exposes them.
assert.sameValue(globalThis.encodeURIComponent("a b"), "a%20b", "via globalThis");
