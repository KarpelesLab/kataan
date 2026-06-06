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

// The functional instantiate() now also surfaces .module and .instance.
var r = WebAssembly.instantiate(bytes);
assert.sameValue(r.instance.exports.add(3, 4), 7, "instantiate().instance");
assert.sameValue(typeof r.module, "object", "instantiate().module");

// new Instance() with a non-Module argument is a TypeError.
var threw = false;
try { new WebAssembly.Instance({}); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "Instance requires a Module");

// new Module() with undecodable bytes throws.
var threwM = false;
try { new WebAssembly.Module(new Uint8Array([1, 2, 3])); } catch (e) { threwM = true; }
assert.sameValue(threwM, true, "invalid module throws");
