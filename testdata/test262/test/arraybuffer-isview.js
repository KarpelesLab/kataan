/*---
description: ArrayBuffer.isView returns true only for typed arrays and DataViews
esid: sec-arraybuffer.isview
---*/
assert.sameValue(typeof ArrayBuffer.isView, "function", "isView is a function");

// Typed arrays of every kind, plus subarrays, are views.
assert.sameValue(ArrayBuffer.isView(new Uint8Array(2)), true, "Uint8Array");
assert.sameValue(ArrayBuffer.isView(new Int16Array(2)), true, "Int16Array");
assert.sameValue(ArrayBuffer.isView(new Float64Array(2)), true, "Float64Array");
assert.sameValue(ArrayBuffer.isView(new Uint8ClampedArray(2)), true, "Uint8ClampedArray");
assert.sameValue(ArrayBuffer.isView(new Uint8Array([1, 2, 3]).subarray(1)), true, "subarray");

// A DataView is a view.
assert.sameValue(ArrayBuffer.isView(new DataView(new ArrayBuffer(8))), true, "DataView");

// Everything else is not — including the ArrayBuffer itself.
assert.sameValue(ArrayBuffer.isView(new ArrayBuffer(8)), false, "ArrayBuffer is not a view");
assert.sameValue(ArrayBuffer.isView([]), false, "array");
assert.sameValue(ArrayBuffer.isView({}), false, "object");
assert.sameValue(ArrayBuffer.isView("x"), false, "string");
assert.sameValue(ArrayBuffer.isView(5), false, "number");
assert.sameValue(ArrayBuffer.isView(), false, "no argument");
assert.sameValue(ArrayBuffer.isView(null), false, "null");
assert.sameValue(ArrayBuffer.isView(undefined), false, "undefined");
