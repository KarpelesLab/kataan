/*---
description: WebAssembly.instantiate returns a Promise; Instance instanceof works
features: [WebAssembly]
---*/
// (module (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))
var bytes = new Uint8Array([0,97,115,109,1,0,0,0,1,7,1,96,2,127,127,1,127,3,2,1,0,7,7,1,3,97,100,100,0,0,10,9,1,7,0,32,0,32,1,106,11]);

// new Instance(new Module(...)) is synchronous; the instance matches its constructor.
var mod = new WebAssembly.Module(bytes);
var inst = new WebAssembly.Instance(mod);
assert.sameValue(inst instanceof WebAssembly.Instance, true, "instance instanceof Instance");
assert.sameValue(inst instanceof WebAssembly.Module, false, "not a Module");
assert.sameValue(inst.exports.add(3, 4), 7, "exported add");
assert.sameValue(Object.keys(inst).join(","), "exports", "only exports is enumerable");

// instantiate(bytes) and instantiate(module) both return a Promise (a thenable).
assert.sameValue(WebAssembly.instantiate(bytes) instanceof Promise, true, "instantiate(bytes) is a Promise");
assert.sameValue(WebAssembly.instantiate(mod) instanceof Promise, true, "instantiate(module) is a Promise");
assert.sameValue(WebAssembly.compile(bytes) instanceof Promise, true, "compile is a Promise");
