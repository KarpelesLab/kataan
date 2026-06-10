/*---
description: WebAssembly.Module / Instance constructors and exports
features: [WebAssembly]
---*/
// (module (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))
var bytes = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 7, 1, 96, 2, 127, 127, 1, 127,
  3, 2, 1, 0,
  7, 7, 1, 3, 97, 100, 100, 0, 0,
  10, 9, 1, 7, 0, 32, 0, 32, 1, 106, 11,
]);

assert.sameValue(typeof WebAssembly.Module, "function", "Module is a constructor");
assert.sameValue(typeof WebAssembly.Instance, "function", "Instance is a constructor");

var mod = new WebAssembly.Module(bytes);
var inst = new WebAssembly.Instance(mod);
assert.sameValue(typeof inst.exports.add, "function", "export is callable");
assert.sameValue(inst.exports.add(7, 5), 12, "exported add");

// A Module can be instantiated more than once.
var inst2 = new WebAssembly.Instance(mod);
assert.sameValue(inst2.exports.add(100, 1), 101, "module reuse");

// instantiate() returns a Promise (per spec); the synchronous Instance path is used
// elsewhere for direct export access.
assert.sameValue(WebAssembly.instantiate(bytes) instanceof Promise, true, "instantiate returns a Promise");

// new Instance() with a non-Module argument is a TypeError.
var threw = false;
try { new WebAssembly.Instance({}); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "Instance requires a Module");

// new Module() with undecodable bytes throws.
var threwM = false;
try { new WebAssembly.Module(new Uint8Array([1, 2, 3])); } catch (e) { threwM = true; }
assert.sameValue(threwM, true, "invalid module throws");

// WebAssembly.compile is async: a Promise<Module>, rejected (not thrown) on bad
// bytes. Assertions run in the microtask callbacks the harness drains.
assert.sameValue(typeof WebAssembly.compile, "function", "compile exists");
WebAssembly.compile(bytes).then(function (m) {
  assert.sameValue(new WebAssembly.Instance(m).exports.add(20, 22), 42, "compiled module instantiates");
});
WebAssembly.compile(new Uint8Array([1, 2, 3])).then(
  function () { assert.sameValue(true, false, "bad bytes should reject"); },
  function (e) { assert.sameValue(e instanceof TypeError, true, "compile rejects on bad bytes"); }
);
