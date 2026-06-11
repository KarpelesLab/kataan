/*---
description: WebAssembly i64 parameters accept BigInt and i64 results return BigInt
features: [WebAssembly]
---*/
// (module (func (export "add") (param i64 i64) (result i64) local.get 0 local.get 1 i64.add))
var bytes = new Uint8Array([0,97,115,109,1,0,0,0,1,7,1,96,2,126,126,1,126,3,2,1,0,7,7,1,3,97,100,100,0,0,10,9,1,7,0,32,0,32,1,124,11]);
var inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));

// i64 args take BigInt; the result is a BigInt.
var r = inst.exports.add(10n, 20n);
assert.sameValue(r, 30n, "i64 add value");
assert.sameValue(typeof r, "bigint", "i64 result is a BigInt");

// Precision is preserved beyond 2^53 (the whole point of i64 BigInt).
assert.sameValue(inst.exports.add(9007199254740993n, 1000n), 9007199254741993n, "above 2^53");
assert.sameValue(inst.exports.add(9223372036854775000n, 7n), 9223372036854775007n, "near i64 max");
assert.sameValue(inst.exports.add(-5n, 3n), -2n, "negative");

// Passing a plain Number to an i64 parameter is a TypeError (must be a BigInt).
var threw = false;
try { inst.exports.add(10, 20); } catch (e) { threw = true; }
assert.sameValue(threw, true, "Number arg to i64 param throws");

// i32 and f64 still marshal to/from Number.
var add32 = new Uint8Array([0,97,115,109,1,0,0,0,1,7,1,96,2,127,127,1,127,3,2,1,0,7,7,1,3,97,100,100,0,0,10,9,1,7,0,32,0,32,1,106,11]);
var r32 = new WebAssembly.Instance(new WebAssembly.Module(add32)).exports.add(2, 3);
assert.sameValue(r32, 5, "i32 value");
assert.sameValue(typeof r32, "number", "i32 result is a Number");
