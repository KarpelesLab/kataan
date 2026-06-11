/*---
description: WASM direct calls, recursion, and structured block/loop/br control flow
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, payload) { return [id].concat(uleb(payload.length), payload); }
function inst(secs) {
  var bytes = [0, 97, 115, 109, 1, 0, 0, 0];
  for (var i = 0; i < secs.length; i++) bytes = bytes.concat(secs[i]);
  return new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(bytes))).exports.f;
}

// Two (i32)->i32 functions; exported func1 calls func0 twice (+1 each).
var f1 = inst([
  sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]),
  sec(3, [2, 0, 0]),
  sec(7, [1, 1, 0x66, 0, 1]),
  sec(10, [2].concat(uleb(7), [0, 0x20, 0, 0x41, 1, 0x6a, 0x0b], uleb(8), [0, 0x20, 0, 0x10, 0, 0x10, 0, 0x0b])),
]);
assert.sameValue(f1(5), 7, "call func0 twice: 5 -> 7");
assert.sameValue(f1(0), 2, "call func0 twice: 0 -> 2");

// Recursive factorial: f(n) = n < 2 ? 1 : n * f(n-1).
var factBody = [0, 0x20, 0, 0x41, 2, 0x48, 0x04, 0x7f, 0x41, 1, 0x05, 0x20, 0, 0x20, 0, 0x41, 1, 0x6b, 0x10, 0, 0x6c, 0x0b, 0x0b];
var fact = inst([
  sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]),
  sec(3, [1, 0]),
  sec(7, [1, 1, 0x66, 0, 0]),
  sec(10, [1].concat(uleb(factBody.length), factBody)),
]);
assert.sameValue(fact(5), 120, "5! = 120");
assert.sameValue(fact(10), 3628800, "10! = 3628800");
assert.sameValue(fact(0), 1, "0! = 1");
assert.sameValue(fact(1), 1, "1! = 1");

// Iterative sum 1..n with block { loop { ... br_if out; ... br loop } }.
var loopBody = [1, 1, 0x7f, 0x02, 0x40, 0x03, 0x40, 0x20, 0, 0x45, 0x0d, 1, 0x20, 1, 0x20, 0, 0x6a, 0x21, 1, 0x20, 0, 0x41, 1, 0x6b, 0x21, 0, 0x0c, 0, 0x0b, 0x0b, 0x20, 1, 0x0b];
var sum = inst([
  sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]),
  sec(3, [1, 0]),
  sec(7, [1, 1, 0x66, 0, 0]),
  sec(10, [1].concat(uleb(loopBody.length), loopBody)),
]);
assert.sameValue(sum(5), 15, "sum 1..5 = 15");
assert.sameValue(sum(100), 5050, "sum 1..100 = 5050");
assert.sameValue(sum(0), 0, "sum of nothing = 0");
