/*---
description: JSON.stringify throws TypeError on a circular structure (no crash)
esid: sec-json.stringify
---*/
// Self-referential object.
var o = {};
o.self = o;
var threw = false;
try { JSON.stringify(o); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "circular object throws TypeError");

// Circular through an array.
var a = [1];
a.push(a);
var threwA = false;
try { JSON.stringify(a); } catch (e) { threwA = e instanceof TypeError; }
assert.sameValue(threwA, true, "circular array throws TypeError");

// Indirect cycle (o -> p -> o).
var p = {};
var q = { p: p };
p.q = q;
var threwI = false;
try { JSON.stringify(q); } catch (e) { threwI = e instanceof TypeError; }
assert.sameValue(threwI, true, "indirect cycle throws TypeError");

// A repeated (but acyclic) reference is NOT a cycle and serializes fine.
var shared = { x: 1 };
assert.sameValue(JSON.stringify({ a: shared, b: shared }), '{"a":{"x":1},"b":{"x":1}}', "shared ref is not circular");

// Deeply nested but acyclic still works.
assert.sameValue(JSON.stringify({ a: { b: { c: [1, 2] } } }), '{"a":{"b":{"c":[1,2]}}}', "deep nesting");
