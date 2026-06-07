/*---
description: WebAssembly.Table constructor, length, get/set, grow, maximum
features: [WebAssembly]
---*/
assert.sameValue(typeof WebAssembly.Table, "function", "constructor exists");

// initial slots default to null; .length reflects the count.
var t = new WebAssembly.Table({ element: "anyfunc", initial: 2 });
assert.sameValue(t.length, 2, "initial length");
assert.sameValue(t.get(0), null, "slots start null");

// set stores a callable; get returns it.
function f() { return 42; }
t.set(0, f);
assert.sameValue(t.get(0)(), 42, "stored function is callable");

// grow(delta) returns the prior length and zero-fills new slots with null.
assert.sameValue(t.grow(3), 2, "grow returns old length");
assert.sameValue(t.length, 5, "grown length");
assert.sameValue(t.get(4), null, "new slot is null");

// Out-of-bounds get/set throw RangeError.
var go = false, so = false;
try { t.get(99); } catch (e) { go = e instanceof RangeError; }
try { t.set(99, f); } catch (e) { so = e instanceof RangeError; }
assert.sameValue(go, true, "get OOB throws RangeError");
assert.sameValue(so, true, "set OOB throws RangeError");

// Growing past the declared maximum throws RangeError.
var capped = new WebAssembly.Table({ element: "anyfunc", initial: 1, maximum: 2 });
var mo = false;
try { capped.grow(5); } catch (e) { mo = e instanceof RangeError; }
assert.sameValue(mo, true, "grow past maximum throws");
assert.sameValue(capped.grow(1), 1, "grow within maximum");

// grow(delta, init) fills new slots with the init value.
var ti = new WebAssembly.Table({ element: "anyfunc", initial: 0 });
ti.grow(2, f);
assert.sameValue(ti.get(1)(), 42, "grow init value");
