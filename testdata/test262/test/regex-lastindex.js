/*---
description: Stateful RegExp.lastIndex for global/sticky exec and test
esid: sec-regexp.prototype.lastindex
---*/
var r = /\d/g;
assert.sameValue(r.lastIndex, 0, "initial lastIndex");
assert.sameValue(r.exec("a1b2c3")[0], "1", "first match");
assert.sameValue(r.lastIndex, 2, "advanced past first match");
assert.sameValue(r.exec("a1b2c3")[0], "2", "resumes from lastIndex");
assert.sameValue(r.lastIndex, 4);
assert.sameValue(r.exec("a1b2c3")[0], "3");
assert.sameValue(r.lastIndex, 6);
assert.sameValue(r.exec("a1b2c3"), null, "no more matches");
assert.sameValue(r.lastIndex, 0, "reset on miss");
var t = /x/g;
assert.sameValue(t.test("axbxc"), true);
assert.sameValue(t.lastIndex, 2, "test advances lastIndex");
assert.sameValue(t.test("axbxc"), true);
assert.sameValue(t.lastIndex, 4);
assert.sameValue(t.test("axbxc"), false, "exhausted");
assert.sameValue(t.lastIndex, 0);
var w = /\d/g;
w.lastIndex = 3;
assert.sameValue(w.exec("12345")[0], "4", "starts from written lastIndex");
assert.sameValue(w.lastIndex, 4);
var ng = /\d/;
assert.sameValue(ng.exec("a1b2")[0], "1", "non-global match");
assert.sameValue(ng.lastIndex, 0, "non-global does not advance lastIndex");
assert.sameValue(ng.exec("a1b2")[0], "1", "non-global always restarts");
var sticky = /\d/y;
sticky.lastIndex = 1;
assert.sameValue(sticky.exec("a1b2")[0], "1", "sticky matches exactly at lastIndex");
assert.sameValue(sticky.exec("a1b2"), null, "sticky fails when not anchored");
