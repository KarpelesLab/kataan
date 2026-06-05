/*---
description: Closures capturing loop variables (var vs let) and IIFE
esid: sec-closure
---*/
var varFns = [];
for (var i = 0; i < 3; i++) varFns.push(function () { return i; });
assert.sameValue(varFns[0]() + "," + varFns[1]() + "," + varFns[2](), "3,3,3", "var shares one binding");
var letFns = [];
for (let j = 0; j < 3; j++) letFns.push(function () { return j; });
assert.sameValue(letFns[0]() + "," + letFns[1]() + "," + letFns[2](), "0,1,2", "let is per-iteration");
var iifeFns = [];
for (var k = 0; k < 3; k++) { (function (captured) { iifeFns.push(function () { return captured; }); })(k); }
assert.sameValue(iifeFns[0]() + "," + iifeFns[2](), "0,2", "IIFE captures per-iteration");
