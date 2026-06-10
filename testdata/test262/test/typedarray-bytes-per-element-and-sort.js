/*---
description: TypedArray.BYTES_PER_ELEMENT (static + instance) and numeric default sort
features: [TypedArray]
---*/
// Static BYTES_PER_ELEMENT on each constructor.
assert.sameValue(Int8Array.BYTES_PER_ELEMENT, 1, "Int8Array");
assert.sameValue(Uint8Array.BYTES_PER_ELEMENT, 1, "Uint8Array");
assert.sameValue(Uint8ClampedArray.BYTES_PER_ELEMENT, 1, "Uint8ClampedArray");
assert.sameValue(Int16Array.BYTES_PER_ELEMENT, 2, "Int16Array");
assert.sameValue(Int32Array.BYTES_PER_ELEMENT, 4, "Int32Array");
assert.sameValue(Float32Array.BYTES_PER_ELEMENT, 4, "Float32Array");
assert.sameValue(Float64Array.BYTES_PER_ELEMENT, 8, "Float64Array");
// Instance BYTES_PER_ELEMENT.
assert.sameValue(new Uint8Array(1).BYTES_PER_ELEMENT, 1, "instance Uint8");
assert.sameValue(new Float64Array(1).BYTES_PER_ELEMENT, 8, "instance Float64");

// A typed array sorts NUMERICALLY by default (a plain array sorts lexicographically).
assert.sameValue(new Uint8Array([10, 1, 5, 20, 3]).sort().join(","), "1,3,5,10,20", "numeric default sort");
assert.sameValue(new Float64Array([3.5, 1.5, 2.5]).sort().join(","), "1.5,2.5,3.5", "float sort");
assert.sameValue([10, 1, 5, 20, 3].sort().join(","), "1,10,20,3,5", "plain array stays lexicographic");

// An explicit comparator is still honored.
assert.sameValue(new Uint8Array([10, 1, 5]).sort(function (a, b) { return b - a; }).join(","), "10,5,1", "descending comparator");
assert.sameValue(new Uint8Array([10, 1, 5]).toSorted().join(","), "1,5,10", "toSorted numeric");
