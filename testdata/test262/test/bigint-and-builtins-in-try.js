/*---
description: BigInt conversion validation, and builtins usable inside a try block
features: [BigInt]
---*/
// BigInt(number): only an exact integer converts; fractional/non-finite throws.
function big(v) { try { return String(BigInt(v)); } catch (e) { return e.name; } }
assert.sameValue(big(2), "2", "integer converts");
assert.sameValue(big(-7), "-7", "negative integer");
assert.sameValue(big(1.5), "RangeError", "fractional throws");
assert.sameValue(big(NaN), "RangeError", "NaN throws");
assert.sameValue(big(Infinity), "RangeError", "Infinity throws");
assert.sameValue(big("0xff"), "255", "hex string");
assert.sameValue(big(true), "1", "boolean coerces");

// Builtins resolve inside a try block (regression: some bailed to ReferenceError).
var ok = "";
try { ok += BigInt(5).toString(); } catch (e) { ok = "ERR1:" + e.name; }
try { ok += "," + new Int32Array(2).length; } catch (e) { ok = "ERR2:" + e.name; }
try { ok += "," + Reflect.has({ a: 1 }, "a"); } catch (e) { ok = "ERR3:" + e.name; }
try { var w = new WeakMap(); w.set({}, 1); ok += ",wm"; } catch (e) { ok = "ERR4:" + e.name; }
assert.sameValue(ok, "5,2,true,wm", "builtins work inside try");
