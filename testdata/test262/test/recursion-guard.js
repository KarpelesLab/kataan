/*---
description: Unbounded recursion throws a RangeError, deep finite recursion works
esid: sec-ecmascript-function-objects-call-thisargument-argumentslist
---*/
var threw = false;
var msg = "";
function infinite() { return infinite(); }
try { infinite(); } catch (e) { threw = e instanceof RangeError; msg = e.message; }
assert.sameValue(threw, true, "infinite recursion throws RangeError");
assert.sameValue(msg.length > 0, true, "has a message");
function sumTo(n) { return n === 0 ? 0 : n + sumTo(n - 1); }
assert.sameValue(sumTo(1000), 500500, "deep finite recursion (1000) works");
function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
assert.sameValue(fib(15), 610, "branching recursion");
var indirectThrew = false;
function a() { return b(); }
function b() { return a(); }
try { a(); } catch (e) { indirectThrew = e instanceof RangeError; }
assert.sameValue(indirectThrew, true, "mutual infinite recursion throws");
function afterCatch() { try { infinite(); } catch (e) { return "recovered"; } }
assert.sameValue(afterCatch(), "recovered", "engine recovers after stack-overflow throw");
