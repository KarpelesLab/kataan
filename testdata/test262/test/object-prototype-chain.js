/*---
description: Plain objects inherit from Object.prototype; Object.create(null) does not
esid: sec-object.prototype
---*/
// Inherited Object.prototype methods are visible via `in` and resolvable.
assert.sameValue("toString" in {}, true, "toString in {}");
assert.sameValue("hasOwnProperty" in {}, true, "hasOwnProperty in {}");
assert.sameValue("valueOf" in {}, true, "valueOf in {}");
assert.sameValue(Object.getPrototypeOf({}) === Object.prototype, true, "getPrototypeOf({}) is Object.prototype");
assert.sameValue({}.hasOwnProperty("x"), false, "inherited hasOwnProperty callable");
// Object.create(null) has no prototype, so inherited names are absent.
var bare = Object.create(null);
assert.sameValue("toString" in bare, false, "Object.create(null) has no toString");
assert.sameValue(Object.getPrototypeOf(bare), null, "null prototype");
// Inherited methods are NOT enumerable.
assert.sameValue(Object.keys({ a: 1, b: 2 }).join(","), "a,b", "Object.keys is own enumerable only");
var seen = [];
for (var k in { a: 1, b: 2 }) seen.push(k);
assert.sameValue(seen.join(","), "a,b", "for-in does not surface inherited methods");
// JSON / spread unaffected.
assert.sameValue(JSON.stringify({ a: 1 }), '{"a":1}', "JSON.stringify own only");
assert.sameValue(Object.keys({ ...{ a: 1, b: 2 } }).join(","), "a,b", "spread copies own only");
// An object literal's prototype is shared (the singleton Object.prototype).
assert.sameValue(Object.getPrototypeOf({}) === Object.getPrototypeOf({ x: 1 }), true, "shared prototype");
// hasOwnProperty distinguishes own from inherited.
var child = Object.create({ inherited: 1 });
child.own = 2;
assert.sameValue(child.hasOwnProperty("own"), true, "own property");
assert.sameValue(child.hasOwnProperty("inherited"), false, "inherited property not own");
assert.sameValue(child.inherited, 1, "but still readable");
