/*---
description: WASM loop with br (branch to the loop header repeats) and br_if (exit a block)
features: [WebAssembly]
---*/
function uleb(n) { var b = []; do { var x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n); return b; }
function sec(id, p) { return [id].concat(uleb(p.length), p); }
function modL(params, results, locals, body) {
  var type = [0x60, params.length].concat(params, [results.length], results);
  var code = locals.concat(body, [0x0b]);
  return new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(
    sec(1, [1].concat(type)), sec(3, [1, 0]), sec(7, [1, 1, 0x66, 0, 0]),
    sec(10, [1].concat(uleb(code.length), code))));
}
function call(m) { var i = new WebAssembly.Instance(new WebAssembly.Module(m)); return i.exports.f.apply(null, Array.prototype.slice.call(arguments, 1)); }

// sum(n) = n + (n-1) + ... + 1, accumulated in local 1 by a counted loop:
//   block $exit { loop $loop {
//     if (n == 0) br $exit ; sum += n ; n -= 1 ; br $loop (repeat)
//   } }  return sum
// br to the loop label jumps to its HEADER (repeats); br_if to the block label exits it.
var body = [
  0x02, 0x40,                 // block $exit
  0x03, 0x40,                 //   loop $loop
  0x20, 0, 0x45, 0x0d, 1,     //     local.get 0; i32.eqz; br_if $exit
  0x20, 1, 0x20, 0, 0x6a, 0x21, 1, // sum += n
  0x20, 0, 0x41, 1, 0x6b, 0x21, 0, // n -= 1
  0x0c, 0,                    //     br $loop (repeat)
  0x0b,                       //   end loop
  0x0b,                       // end block
  0x20, 1,                    // local.get 1 (sum)
];
var f = modL([0x7f], [0x7f], [1, 1, 0x7f], body);
assert.sameValue(call(f, 5), 15, "sum 1..5");
assert.sameValue(call(f, 10), 55, "sum 1..10");
assert.sameValue(call(f, 0), 0, "empty loop (br_if exits immediately)");
assert.sameValue(call(f, 1), 1, "single iteration");
assert.sameValue(call(f, 100), 5050, "sum 1..100");
