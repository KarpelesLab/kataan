/*---
description: WASM start function runs at instantiation; exported memory is a WebAssembly.Memory snapshot
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }
// (memory 1) func0 $init: mem[0]=42; start=func0; func1 "f"(addr)=load8_u; export "f" + "memory".
function startMod() {
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [2, 0x60, 0, 0, 0x60, 1, 0x7f, 1, 0x7f]),
    sec(3, [2, 0, 1]),
    sec(5, [1, 0, 1]),
    sec(7, [2, 1, 0x66, 0, 1, 6, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 2, 0]),
    sec(8, [0]),
    sec(10, [2].concat(uleb(9), [0, 0x41, 0, 0x41, 42, 0x3a, 0, 0, 0x0b], uleb(7), [0, 0x20, 0, 0x2d, 0, 0, 0x0b]))));
}
var inst = new WebAssembly.Instance(new WebAssembly.Module(startMod()));

// The start function ran at instantiation (no export call needed).
assert.sameValue(inst.exports.f(0), 42, "start function wrote mem[0] = 42");

// The exported memory is a WebAssembly.Memory whose buffer reflects that write.
assert.sameValue(inst.exports.memory instanceof WebAssembly.Memory, true, "exports.memory is a WebAssembly.Memory");
assert.sameValue(inst.exports.memory.buffer.byteLength, 65536, "1 page = 64 KiB");
var view = new Uint8Array(inst.exports.memory.buffer);
assert.sameValue(view[0], 42, "byte 0 = 42 (from start)");
assert.sameValue(view[1], 0, "byte 1 = 0");

// A module exporting memory initialized by a data segment.
function dataMod() {
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]),
    sec(3, [1, 0]),
    sec(5, [1, 0, 1]),
    sec(7, [2, 1, 0x66, 0, 0, 3, 0x6d, 0x65, 0x6d, 2, 0]),
    sec(11, [1, 0, 0x41, 3, 0x0b, 2, 9, 8]),
    sec(10, [1].concat(uleb(7), [0, 0x20, 0, 0x2d, 0, 0, 0x0b]))));
}
var d = new WebAssembly.Instance(new WebAssembly.Module(dataMod()));
var dv = new Uint8Array(d.exports.mem.buffer);
assert.sameValue(dv[3], 9, "data segment byte at offset 3");
assert.sameValue(dv[4], 8, "data segment byte at offset 4");
assert.sameValue(dv[0], 0, "memory before the segment is zero");
