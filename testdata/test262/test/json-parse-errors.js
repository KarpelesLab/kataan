/*---
description: JSON.parse throws SyntaxError on malformed input
esid: sec-json.parse
---*/
function err(text) {
  try { JSON.parse(text); return "no-throw"; }
  catch (e) { return e instanceof SyntaxError ? "SyntaxError" : e.constructor.name; }
}
assert.sameValue(err("[1,2,"), "SyntaxError", "truncated array");
assert.sameValue(err("[1,2]extra"), "SyntaxError", "trailing characters");
assert.sameValue(err("{bad}"), "SyntaxError", "unquoted key");
assert.sameValue(err('{"a":}'), "SyntaxError", "missing value");
assert.sameValue(err(""), "SyntaxError", "empty input");
assert.sameValue(err('"unterminated'), "SyntaxError", "unterminated string");
assert.sameValue(err("nul"), "SyntaxError", "bad literal");

// Valid JSON still parses.
assert.sameValue(JSON.parse("[1,2,3]").length, 3, "valid array");
assert.sameValue(JSON.parse('{"a":1,"b":[2,3]}').b[1], 3, "nested");
assert.sameValue(JSON.parse("true"), true, "literal");
assert.sameValue(JSON.parse("-3.14e2"), -314, "number");
