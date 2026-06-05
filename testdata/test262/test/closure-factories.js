/*---
description: Closures as factories with private state
esid: sec-closure
---*/
function makeAdder(n) { return function (x) { return x + n; }; }
var add5 = makeAdder(5);
var add10 = makeAdder(10);
assert.sameValue(add5(3), 8);
assert.sameValue(add10(3), 13);
assert.sameValue(add5(add10(1)), 16);
function makeBank(initial) {
  var balance = initial;
  return {
    deposit: function (n) { balance += n; return balance; },
    withdraw: function (n) { if (n <= balance) balance -= n; return balance; },
    getBalance: function () { return balance; }
  };
}
var acct = makeBank(100);
acct.deposit(50);
acct.withdraw(30);
assert.sameValue(acct.getBalance(), 120);
var memo = (function () { var cache = {}; return function (k, v) { if (!(k in cache)) cache[k] = v; return cache[k]; }; })();
assert.sameValue(memo("a", 1), 1);
assert.sameValue(memo("a", 99), 1, "cached");
