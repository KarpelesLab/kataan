/*---
description: WebAssembly.Memory constructor, .buffer/byteLength, grow, maximum
features: [WebAssembly]
---*/
assert.sameValue(typeof WebAssembly.Memory, "function", "constructor exists");

// A page is 64 KiB; .buffer is an ArrayBuffer of initial*65536 bytes.
var m = new WebAssembly.Memory({ initial: 1 });
assert.sameValue(m.buffer.byteLength, 65536, "1 page = 64 KiB");

// grow(delta) returns the prior page count and enlarges .buffer.
var old = m.grow(2);
assert.sameValue(old, 1, "grow returns old page count");
assert.sameValue(m.buffer.byteLength, 3 * 65536, "buffer grew to 3 pages");

// Growing past the declared maximum throws a RangeError.
var capped = new WebAssembly.Memory({ initial: 1, maximum: 2 });
var threw = false;
try { capped.grow(5); } catch (e) { threw = e instanceof RangeError; }
assert.sameValue(threw, true, "grow past maximum throws RangeError");
assert.sameValue(capped.grow(1), 1, "grow within maximum succeeds");
assert.sameValue(capped.buffer.byteLength, 2 * 65536, "at maximum");
