/*---
description: TypedArray.of / TypedArray.from statics and the subarray instance method
features: [TypedArray]
---*/
// of(...items) builds a typed array of the constructor's kind.
var o = Uint8Array.of(1, 2, 3);
assert.sameValue(o.join(","), "1,2,3", "of values");
assert.sameValue(o instanceof Uint8Array, true, "of kind");

// from(iterable|arrayLike, mapFn?).
assert.sameValue(Uint8Array.from([4, 5, 6]).join(","), "4,5,6", "from array");
assert.sameValue(Uint8Array.from([1, 2, 3], x => x * 10).join(","), "10,20,30", "from with mapFn");
assert.sameValue(Uint8Array.from(new Set([1, 2, 2, 3])).join(","), "1,2,3", "from a Set");
assert.sameValue(Uint8Array.from("123", c => c.charCodeAt(0)).join(","), "49,50,51", "from a string");

// Element-type coercion applies.
assert.sameValue(Uint8Array.of(256, 257).join(","), "0,1", "Uint8 wraps");
assert.sameValue(Int8Array.of(128, -1).join(","), "-128,-1", "Int8 signed");
assert.sameValue(Float64Array.of(1.5, 2.5).join(","), "1.5,2.5", "Float64");

// subarray(begin, end) returns a same-kind typed array over the slice.
var a = new Uint8Array([1, 2, 3, 4, 5]);
assert.sameValue(a.subarray(1, 3).join(","), "2,3", "subarray range");
assert.sameValue(a.subarray(2).join(","), "3,4,5", "subarray from start");
assert.sameValue(a.subarray(-2).join(","), "4,5", "subarray negative");
assert.sameValue(a.subarray(0, 2) instanceof Uint8Array, true, "subarray kind");
