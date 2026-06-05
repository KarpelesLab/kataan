/*---
description: Closures capturing loop variables (let vs var)
esid: sec-let-and-const-declarations
---*/
var funcsLet = [];
for (let i = 0; i < 3; i++) { funcsLet.push(function () { return i; }); }
assert.sameValue(funcsLet[0]() + "," + funcsLet[1]() + "," + funcsLet[2](), "0,1,2", "let captures per-iteration");
var funcsVar = [];
for (var j = 0; j < 3; j++) { funcsVar.push(function () { return j; }); }
assert.sameValue(funcsVar[0]() + "," + funcsVar[1]() + "," + funcsVar[2](), "3,3,3", "var shares one binding");
function makeCounters() {
  var counters = [];
  for (let k = 0; k < 3; k++) {
    counters.push({ get: function () { return k; }, inc: function () { return k + 10; } });
  }
  return counters;
}
var c = makeCounters();
assert.sameValue(c[1].get(), 1);
assert.sameValue(c[2].inc(), 12);
var adders = [1, 2, 3].map(function (n) { return function (x) { return x + n; }; });
assert.sameValue(adders[0](10), 11);
assert.sameValue(adders[2](10), 13);
