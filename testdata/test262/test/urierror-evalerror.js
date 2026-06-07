/*---
description: URIError and EvalError exist; malformed URI decoding throws URIError
features: [error-cause]
---*/
assert.sameValue(typeof URIError, "function", "URIError is a global constructor");
assert.sameValue(typeof EvalError, "function", "EvalError is a global constructor");

// Constructors produce proper Error subclasses.
var u = new URIError("bad");
assert.sameValue(u.name, "URIError", "URIError name");
assert.sameValue(u.message, "bad", "URIError message");
assert.sameValue(u instanceof Error, true, "URIError is an Error");
assert.sameValue(u instanceof URIError, true, "instanceof itself");
assert.sameValue(u.toString(), "URIError: bad", "toString");

var e = new EvalError();
assert.sameValue(e.name, "EvalError", "EvalError name");
assert.sameValue(e instanceof Error, true, "EvalError is an Error");

// Malformed percent-encoding throws a URIError (not a TypeError).
var threw = null;
try { decodeURIComponent("%"); } catch (x) { threw = x; }
assert.sameValue(threw instanceof URIError, true, "decode '%' -> URIError");
var threw2 = null;
try { decodeURI("%zz"); } catch (x) { threw2 = x; }
assert.sameValue(threw2 instanceof URIError, true, "decode '%zz' -> URIError");

// A valid decode does not throw.
assert.sameValue(decodeURIComponent("a%20b"), "a b", "valid decode");
