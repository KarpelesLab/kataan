/*---
description: btoa / atob base64 encoding and decoding
esid: sec-btoa
---*/
assert.sameValue(btoa("hi"), "aGk=", "btoa two bytes");
assert.sameValue(btoa("hello"), "aGVsbG8=", "btoa five bytes");
assert.sameValue(btoa("any carnal pleasure."), "YW55IGNhcm5hbCBwbGVhc3VyZS4=", "btoa longer");
assert.sameValue(btoa(""), "", "btoa empty");
assert.sameValue(btoa("M"), "TQ==", "btoa one byte (two pads)");
assert.sameValue(btoa("Ma"), "TWE=", "btoa two bytes (one pad)");
assert.sameValue(btoa("Man"), "TWFu", "btoa three bytes (no pad)");
assert.sameValue(atob("aGk="), "hi", "atob");
assert.sameValue(atob("aGVsbG8="), "hello", "atob five bytes");
assert.sameValue(atob("TWFu"), "Man", "atob no padding");
assert.sameValue(atob(btoa("round trip 123!@#")), "round trip 123!@#", "round trip");
// Latin1 (code points up to 255) is allowed.
assert.sameValue(btoa("é"), "6Q==", "Latin1 character");
assert.sameValue(atob("6Q==").charCodeAt(0), 233, "decoded byte value");
// Whitespace in the input is ignored on decode.
assert.sameValue(atob("aG k="), "hi", "whitespace ignored");
// A character above U+00FF cannot be encoded.
var threw = false;
try { btoa("\u{1F600}"); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "non-Latin1 throws");
// globalThis exposes them.
assert.sameValue(globalThis.atob(globalThis.btoa("x")), "x", "via globalThis");
