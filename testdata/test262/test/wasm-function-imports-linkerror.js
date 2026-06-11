/*---
description: WASM function imports are called from WASM; a missing/non-callable import is a LinkError at instantiation
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }
// import "env"."imp" (func (i32)->i32); local func "run"(x) = imp(x) + 1; export "run".
function mkmod() {
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]),
    sec(2, [1, 3, 0x65, 0x6e, 0x76, 3, 0x69, 0x6d, 0x70, 0x00, 0]),
    sec(3, [1, 0]),
    sec(7, [1, 3, 0x72, 0x75, 0x6e, 0, 1]),
    sec(10, [1].concat(uleb(9), [0, 0x20, 0, 0x10, 0, 0x41, 1, 0x6a, 0x0b]))));
}
function compile() { return new WebAssembly.Module(mkmod()); }

// A provided import is invoked from WASM.
var inst = new WebAssembly.Instance(compile(), { env: { imp: function (x) { return x * 2; } } });
assert.sameValue(inst.exports.run(10), 21, "imp(10)*... -> run = 21");

// The import can observe its argument and return a value used by the caller.
var seen = [];
var inst2 = new WebAssembly.Instance(compile(), { env: { imp: function (x) { seen.push(x); return x + 100; } } });
assert.sameValue(inst2.exports.run(5), 106, "run(5) = imp(5)+1 = 106");
assert.sameValue(seen.join(","), "5", "import saw its argument");

// Missing / wrong-type imports throw a LinkError eagerly at instantiation.
function linkErr(fn) { try { fn(); return null; } catch (e) { return e; } }
var e1 = linkErr(function () { return new WebAssembly.Instance(compile(), {}); });
assert.sameValue(e1 instanceof WebAssembly.LinkError, true, "missing import key -> LinkError");
var e2 = linkErr(function () { return new WebAssembly.Instance(compile()); });
assert.sameValue(e2 instanceof WebAssembly.LinkError, true, "missing import object -> LinkError");
var e3 = linkErr(function () { return new WebAssembly.Instance(compile(), { env: { imp: 42 } }); });
assert.sameValue(e3 instanceof WebAssembly.LinkError, true, "non-callable import -> LinkError");

// A module with no imports instantiates with or without an import object.
var add = new Uint8Array([0,97,115,109,1,0,0,0,1,7,1,96,2,127,127,1,127,3,2,1,0,7,7,1,3,97,100,100,0,0,10,9,1,7,0,32,0,32,1,106,11]);
assert.sameValue(new WebAssembly.Instance(new WebAssembly.Module(add)).exports.add(2, 3), 5, "no-import module, no object");
assert.sameValue(new WebAssembly.Instance(new WebAssembly.Module(add), {}).exports.add(4, 5), 9, "no-import module, empty object");

// Module.imports reports the import.
var imps = WebAssembly.Module.imports(compile());
assert.sameValue(imps.length, 1, "one import");
assert.sameValue(imps[0].module + "." + imps[0].name + ":" + imps[0].kind, "env.imp:function", "import descriptor");
