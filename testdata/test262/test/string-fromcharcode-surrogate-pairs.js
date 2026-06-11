/*---
description: String.fromCharCode applies ToUint16 and combines surrogate pairs
esid: sec-string.fromcharcode
---*/
// A high/low surrogate pair combines into one astral code point.
assert.sameValue(String.fromCharCode(0xD83D, 0xDE00), "\u{1F600}", "surrogate pair -> astral");
assert.sameValue(String.fromCharCode(0xD83D, 0xDE00).length, 2, "two code units");

// ASCII / BMP code units pass through.
assert.sameValue(String.fromCharCode(72, 105), "Hi", "ASCII");
assert.sameValue(String.fromCharCode(0x4E2D), "中", "BMP");

// Each argument is ToUint16'd (truncated mod 2^16; negatives wrap).
assert.sameValue(String.fromCharCode(0x1F600), String.fromCharCode(0xF600), "truncate above 0xFFFF");
assert.sameValue(String.fromCharCode(65601), "A", "65601 mod 65536 = 65");
assert.sameValue(String.fromCharCode(-1), String.fromCharCode(65535), "negative wraps");

// Multiple pairs and a mix of BMP + pair.
assert.sameValue(String.fromCharCode(0xD83D, 0xDE00, 0xD83D, 0xDE01).length, 4, "two pairs");
assert.sameValue(String.fromCharCode(65, 0xD83D, 0xDE00, 66), "A\u{1F600}B", "mixed");

// No arguments -> empty string. fromCodePoint accepts an astral value directly.
assert.sameValue(String.fromCharCode(), "", "empty");
assert.sameValue(String.fromCodePoint(0x1F600), "\u{1F600}", "fromCodePoint astral");
