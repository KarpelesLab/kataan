/*---
description: a let/const C-style for head creates a fresh binding per iteration
esid: sec-forbodyevaluation
---*/
// Closures capture each iteration's own binding.
var fns = [];
for (let x = 0; x < 3; x++) fns.push(function () { return x; });
assert.sameValue(fns.map(function (f) { return f(); }).join(","), "0,1,2", "let per-iteration");

// var shares one binding (all closures see the final value).
var vfns = [];
for (var y = 0; y < 3; y++) vfns.push(function () { return y; });
assert.sameValue(vfns.map(function (f) { return f(); }).join(","), "3,3,3", "var shared");

// continue still advances the per-iteration binding.
var cf = [];
for (let i = 0; i < 5; i++) { if (i % 2 === 0) continue; cf.push(function () { return i; }); }
assert.sameValue(cf.map(function (f) { return f(); }).join(","), "1,3", "continue");

// Nested loops each get their own per-iteration bindings.
var nf = [];
for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) nf.push(function () { return i + "" + j; });
assert.sameValue(nf.map(function (f) { return f(); }).join(","), "00,01,10,11", "nested");

// A body mutation of the loop variable is reflected in that iteration's capture.
var mf = [];
for (let k = 0; k < 3; k++) { k += 0; mf.push(function () { return k; }); }
assert.sameValue(mf.map(function (f) { return f(); }).join(","), "0,1,2", "body mutation");

// break stops; captured values are those before the break.
var bf = [];
for (let p = 0; p < 10; p++) { if (p === 3) break; bf.push(function () { return p; }); }
assert.sameValue(bf.map(function (f) { return f(); }).join(","), "0,1,2", "break");

// A non-captured let loop variable still sums correctly.
var sum = 0;
for (let q = 0; q < 4; q++) sum += q;
assert.sameValue(sum, 6, "non-captured");
