/*---
description: break to a block label from inside a nested loop exits the whole labeled block
esid: sec-labelled-statements
---*/
// break <blockLabel> from inside a nested loop skips the rest of the block.
var r = "";
lbl: {
  for (var i = 0; i < 3; i++) {
    if (i === 1) break lbl;
    r += i;
  }
  r += "x"; // inside lbl, after the loop — must be skipped
}
assert.sameValue(r, "0", "break to block label exits the block");

// A labeled `if` block.
var s = "";
cond: if (true) { s += "1"; break cond; s += "2"; }
s += "3";
assert.sameValue(s, "13", "break out of a labeled if");

// Nested block labels: break the outer skips the rest of both.
var n = "";
A: { B: { n += "x"; break A; n += "y"; } n += "z"; }
assert.sameValue(n, "x", "break A unwinds past B and the rest of A");

// Ordinary labeled-loop break/continue still work.
var t = "";
outer: for (var a = 0; a < 3; a++) {
  for (var b = 0; b < 3; b++) {
    if (b === 1) continue outer;
    if (a === 2) break outer;
    t += a + "" + b + " ";
  }
}
assert.sameValue(t.trim(), "00 10", "labeled loop break/continue");

// A direct block break (no nested loop) is unaffected.
var d = [];
blk: { d.push("a"); break blk; d.push("b"); }
d.push("c");
assert.sameValue(d.join(","), "a,c", "direct block break");
