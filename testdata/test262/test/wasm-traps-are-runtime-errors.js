/*---
description: WASM execution traps throw WebAssembly.RuntimeError (not CompileError)
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }
function mod(params, results, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var code = [0].concat(body, [0x0b]);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1].concat(type)), sec(3, [1, 0]), sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(code.length), code))));
}
function call(m) { var i = new WebAssembly.Instance(new WebAssembly.Module(m)); return i.exports.f.apply(null, Array.prototype.slice.call(arguments, 1)); }
function trapKind(fn) {
  try { fn(); return "no-trap"; }
  catch (e) { return e instanceof WebAssembly.RuntimeError ? "runtime" : (e instanceof WebAssembly.CompileError ? "compile" : "other"); }
}

// `unreachable` executed is a runtime trap.
assert.sameValue(trapKind(function () { call(mod([], [0x7f], [0x00])); }), "runtime", "unreachable -> RuntimeError");
// integer divide by zero.
assert.sameValue(trapKind(function () { call(mod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x6d]), 5, 0); }), "runtime", "i32.div_s /0 -> RuntimeError");
// i32.rem_s by zero.
assert.sameValue(trapKind(function () { call(mod([0x7f, 0x7f], [0x7f], [0x20, 0, 0x20, 1, 0x6f]), 5, 0); }), "runtime", "i32.rem_s /0 -> RuntimeError");
// float->int truncation of NaN (the trapping trunc, not trunc_sat).
assert.sameValue(trapKind(function () { call(mod([0x7c], [0x7f], [0x20, 0, 0xaa]), NaN); }), "runtime", "i32.trunc_f64_s(NaN) -> RuntimeError");
// integer overflow in trapping truncation.
assert.sameValue(trapKind(function () { call(mod([0x7c], [0x7f], [0x20, 0, 0xaa]), 1e20); }), "runtime", "trunc overflow -> RuntimeError");

// A genuinely malformed module still raises CompileError at Module construction (not Runtime).
var compileKind;
try { new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 9, 9, 9, 9])); compileKind = "no-error"; }
catch (e) { compileKind = e instanceof WebAssembly.CompileError ? "compile" : "other"; }
assert.sameValue(compileKind, "compile", "bad module version -> CompileError");

// A normal call is unaffected.
assert.sameValue(call(mod([0x7f], [0x7f], [0x20, 0, 0x41, 1, 0x6a]), 5), 6, "normal add still works");
