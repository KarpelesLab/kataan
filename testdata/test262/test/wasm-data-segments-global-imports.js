/*---
description: WASM active data segments initialize memory; imported globals are read from a number or a WebAssembly.Global
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }

// (memory 1) with an active data segment writing [10,20,30] at offset 5; f(addr) = i32.load8_u.
function dataMod() {
  var body = [0, 0x20, 0, 0x2d, 0, 0, 0x0b];
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]),
    sec(3, [1, 0]),
    sec(5, [1, 0, 1]),
    sec(7, [1, 1, 0x66, 0, 0]),
    sec(11, [1, 0, 0x41, 5, 0x0b, 3, 10, 20, 30]),
    sec(10, [1].concat(uleb(body.length), body))));
}
var df = new WebAssembly.Instance(new WebAssembly.Module(dataMod())).exports.f;
assert.sameValue(df(5), 10, "data[0] at offset 5");
assert.sameValue(df(6), 20, "data[1]");
assert.sameValue(df(7), 30, "data[2]");
assert.sameValue(df(0), 0, "memory outside the segment is zero");

// A data segment carrying the bytes of "Hi".
function strMod() {
  var body = [0, 0x20, 0, 0x2d, 0, 0, 0x0b];
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 1, 0x7f, 1, 0x7f]),
    sec(3, [1, 0]),
    sec(5, [1, 0, 1]),
    sec(7, [1, 1, 0x66, 0, 0]),
    sec(11, [1, 0, 0x41, 0, 0x0b, 2, 72, 105]),
    sec(10, [1].concat(uleb(body.length), body))));
}
var sf = new WebAssembly.Instance(new WebAssembly.Module(strMod())).exports.f;
assert.sameValue(String.fromCharCode(sf(0)) + String.fromCharCode(sf(1)), "Hi", "string data segment");

// import "env"."g" (global i32, immutable); f() = g + 1.
function gMod() {
  var body = [0, 0x23, 0, 0x41, 1, 0x6a, 0x0b];
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1, 0x60, 0, 1, 0x7f]),
    sec(2, [1, 3, 0x65, 0x6e, 0x76, 1, 0x67, 0x03, 0x7f, 0x00]),
    sec(3, [1, 0]),
    sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(body.length), body))));
}
// Imported global supplied as a plain number.
assert.sameValue(new WebAssembly.Instance(new WebAssembly.Module(gMod()), { env: { g: 41 } }).exports.f(), 42, "global import (number)");
// Imported global supplied as a WebAssembly.Global.
assert.sameValue(
  new WebAssembly.Instance(new WebAssembly.Module(gMod()), { env: { g: new WebAssembly.Global({ value: "i32" }, 100) } }).exports.f(),
  101,
  "global import (WebAssembly.Global)"
);
