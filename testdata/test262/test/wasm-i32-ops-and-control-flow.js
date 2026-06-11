/*---
description: WASM i32 comparison/bitwise/arithmetic opcodes, control flow, select, and locals
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sleb(n) {
  var b = [], more = true;
  while (more) {
    var x = n & 0x7f; n >>= 7;
    if ((n === 0 && (x & 0x40) === 0) || (n === -1 && (x & 0x40) !== 0)) more = false; else x |= 0x80;
    b.push(x);
  }
  return b;
}
// Build a single-function module exporting "f" with the given param/result types and body.
function mod(params, results, localCount, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var typesec = [1].concat(uleb(1 + type.length), [1], type);
  var funcsec = [3, 2, 1, 0];
  var exportsec = [7, 5, 1, 1, 0x66, 0, 0]; // export "f" (func 0)
  var locals = localCount ? [1, localCount, 0x7f] : [0];
  var code = locals.concat(body, [0x0b]);
  var codesec = [10].concat(uleb(1 + uleb(code.length).length + code.length), [1], uleb(code.length), code);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(typesec, funcsec, exportsec, codesec));
}
function call(m) {
  var inst = new WebAssembly.Instance(new WebAssembly.Module(m));
  var args = Array.prototype.slice.call(arguments, 1);
  return inst.exports.f.apply(null, args);
}
// A two-i32-param binary op returning i32.
function binop(op) { return mod([0x7f, 0x7f], [0x7f], 0, [0x20, 0, 0x20, 1, op]); }

// Comparisons (0/1 results).
assert.sameValue(call(binop(0x46), 5, 5), 1, "i32.eq true");
assert.sameValue(call(binop(0x46), 5, 6), 0, "i32.eq false");
assert.sameValue(call(binop(0x47), 5, 6), 1, "i32.ne");
assert.sameValue(call(binop(0x48), 3, 5), 1, "i32.lt_s");
assert.sameValue(call(binop(0x4a), 5, 3), 1, "i32.gt_s");
assert.sameValue(call(binop(0x4c), 5, 5), 1, "i32.le_s");
assert.sameValue(call(binop(0x4e), 5, 5), 1, "i32.ge_s");

// Bitwise.
assert.sameValue(call(binop(0x71), 0xff, 0x0f), 0x0f, "i32.and");
assert.sameValue(call(binop(0x72), 0xf0, 0x0f), 0xff, "i32.or");
assert.sameValue(call(binop(0x73), 0xff, 0x0f), 0xf0, "i32.xor");
assert.sameValue(call(binop(0x74), 1, 4), 16, "i32.shl");
assert.sameValue(call(binop(0x76), 256, 2), 64, "i32.shr_u");

// Arithmetic, including a div-by-zero trap.
assert.sameValue(call(binop(0x6b), 10, 3), 7, "i32.sub");
assert.sameValue(call(binop(0x6c), 6, 7), 42, "i32.mul");
assert.sameValue(call(binop(0x6d), 20, 3), 6, "i32.div_s");
assert.sameValue(call(binop(0x6f), 20, 3), 2, "i32.rem_s");
var trapped = false;
try { call(binop(0x6d), 5, 0); } catch (e) { trapped = true; }
assert.sameValue(trapped, true, "i32.div_s by zero traps");

// Unary: eqz, clz, popcnt.
function unop(op) { return mod([0x7f], [0x7f], 0, [0x20, 0, op]); }
assert.sameValue(call(unop(0x45), 0), 1, "i32.eqz 0");
assert.sameValue(call(unop(0x45), 5), 0, "i32.eqz nonzero");
assert.sameValue(call(unop(0x67), 1), 31, "i32.clz");
assert.sameValue(call(unop(0x69), 0xff), 8, "i32.popcnt");

// if/else returning a value (signed-LEB constants).
var ifelse = mod([0x7f], [0x7f], 0,
  [0x20, 0, 0x04, 0x7f].concat([0x41], sleb(100), [0x05, 0x41], sleb(200), [0x0b]));
assert.sameValue(call(ifelse, 1), 100, "if branch");
assert.sameValue(call(ifelse, 0), 200, "else branch");

// select.
var sel = mod([0x7f, 0x7f, 0x7f], [0x7f], 0, [0x20, 0, 0x20, 1, 0x20, 2, 0x1b]);
assert.sameValue(call(sel, 10, 20, 1), 10, "select first");
assert.sameValue(call(sel, 10, 20, 0), 20, "select second");

// local.tee: (local.get 0 + 5) tee'd into local 1, then local 1 + local 1.
var tee = mod([0x7f], [0x7f], 1, [0x20, 0, 0x41, 5, 0x6a, 0x22, 1, 0x20, 1, 0x6a]);
assert.sameValue(call(tee, 3), 16, "local.tee");

// block + br_if forwarding (branch to end when the condition is nonzero).
var blk = mod([0x7f], [0x7f], 0, [0x02, 0x7f, 0x41, 7, 0x20, 0, 0x0d, 0, 0x1a, 0x41, 9, 0x0b]);
assert.sameValue(call(blk, 1), 7, "br_if taken -> block value 7");

// A negative i32 constant round-trips (signed LEB128).
assert.sameValue(call(mod([], [0x7f], 0, [0x41, 0x7f])), -1, "i32.const -1");
