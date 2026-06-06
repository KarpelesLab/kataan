/*---
description: String.prototype.repeat throws RangeError on negative/infinite counts (no crash)
esid: sec-string.prototype.repeat
---*/
// Negative and +Infinity counts are RangeErrors (and must not crash the host).
var negThrew = false;
try { "x".repeat(-1); } catch (e) { negThrew = e instanceof RangeError; }
assert.sameValue(negThrew, true, "repeat(-1) throws RangeError");

var infThrew = false;
try { "x".repeat(Infinity); } catch (e) { infThrew = e instanceof RangeError; }
assert.sameValue(infThrew, true, "repeat(Infinity) throws RangeError");

// Valid counts work, truncating a fractional count toward zero.
assert.sameValue("ab".repeat(3), "ababab", "repeat 3");
assert.sameValue("x".repeat(0), "", "repeat 0 is empty");
assert.sameValue("x".repeat(2.9), "xx", "fractional count truncates");
