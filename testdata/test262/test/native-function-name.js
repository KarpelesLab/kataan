/*---
description: built-in (native) functions expose their own non-enumerable name
esid: sec-name
---*/
// Namespaced methods, global functions, constructors, and typed arrays.
assert.sameValue(Math.max.name, "max", "Math.max");
assert.sameValue(Math.sqrt.name, "sqrt", "Math.sqrt");
assert.sameValue(JSON.stringify.name, "stringify", "JSON.stringify");
assert.sameValue(parseInt.name, "parseInt", "parseInt");
assert.sameValue(isNaN.name, "isNaN", "isNaN");
assert.sameValue(Promise.name, "Promise", "Promise");
assert.sameValue(Date.name, "Date", "Date");
assert.sameValue(Map.name, "Map", "Map");
assert.sameValue(TypeError.name, "TypeError", "TypeError");
assert.sameValue(Uint8Array.name, "Uint8Array", "Uint8Array");
assert.sameValue(WebAssembly.Module.name, "Module", "WebAssembly.Module");

// The name property is non-enumerable (not in Object.keys / JSON).
assert.sameValue(Object.keys(Math.max).length, 0, "name is non-enumerable");
assert.sameValue(JSON.stringify(Math.max), undefined, "function stringifies to undefined");

// A native function can also carry assigned properties (the enabling fix).
parseInt.tag = 7;
assert.sameValue(parseInt.tag, 7, "native function holds an assigned property");

// User functions still report their own name.
function userFn() {}
assert.sameValue(userFn.name, "userFn", "user function name");
