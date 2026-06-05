/*---
description: catch binding — named, destructured, and optional
esid: sec-try-statement
---*/
var caught;
try { throw new Error("msg"); } catch (e) { caught = e.message; }
assert.sameValue(caught, "msg", "named catch binding");
var destructured;
try { throw { code: 42, text: "fail" }; } catch ({ code, text }) { destructured = code + ":" + text; }
assert.sameValue(destructured, "42:fail", "destructured catch binding");
var optional = false;
try { throw new Error("x"); } catch { optional = true; }
assert.sameValue(optional, true, "optional catch binding (no param)");
var arrayDestructure;
try { throw [1, 2, 3]; } catch ([a, b]) { arrayDestructure = a + b; }
assert.sameValue(arrayDestructure, 3, "array destructure in catch");
var nestedResult;
try {
  try { throw new Error("inner"); }
  catch (e) { throw new Error("outer: " + e.message); }
} catch (e) { nestedResult = e.message; }
assert.sameValue(nestedResult, "outer: inner", "nested catch rethrow");
var finallyRan = false;
try { try { throw new Error("x"); } finally { finallyRan = true; } } catch (e) {}
assert.sameValue(finallyRan, true, "finally runs before propagation");
