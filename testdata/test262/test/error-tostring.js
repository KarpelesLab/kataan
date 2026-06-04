/*---
description: Error toString, name/message, and throwing non-errors
esid: sec-error.prototype.tostring
---*/
var e = new Error("something failed");
assert.sameValue(e.toString(), "Error: something failed");
var t = new TypeError("bad");
assert.sameValue(t.toString(), "TypeError: bad");
var caught;
try { throw "a string error"; } catch (err) { caught = err; }
assert.sameValue(caught, "a string error", "can throw a primitive");
try { throw { code: 42 }; } catch (err) { caught = err.code; }
assert.sameValue(caught, 42, "can throw an object");
var bare = new Error();
assert.sameValue(bare.toString(), "Error", "no message");
