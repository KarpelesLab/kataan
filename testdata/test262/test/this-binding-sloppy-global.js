/*---
description: Sloppy-mode this defaults to the global object; strict keeps undefined
esid: sec-ordinarycallbindthis
flags: [noStrict]
---*/
// A plain (sloppy) function call binds `this` to the global object.
(function () {
  function f() { return this; }
  assert.sameValue(f(), globalThis, "sloppy this is the global object");
  assert.sameValue(typeof f(), "object", "typeof sloppy this");
  assert.sameValue(f.call(null), globalThis, "call(null) -> global in sloppy mode");
  assert.sameValue(f.call(undefined), globalThis, "call(undefined) -> global");
})();
// A nested plain function also gets the global object.
(function () {
  var o = { m: function () { function inner() { return this === globalThis; } return inner(); } };
  assert.sameValue(o.m(), true, "nested plain function this is global");
})();
// A method call still binds the receiver.
(function () {
  var o = { x: 5, m() { return this.x; } };
  assert.sameValue(o.m(), 5, "method this is the receiver");
})();
// An arrow inherits `this` lexically.
(function () {
  var o = { x: 9, m() { var a = () => this.x; return a(); } };
  assert.sameValue(o.m(), 9, "arrow this is lexical");
})();
// An explicit receiver is preserved.
(function () {
  function f() { return this.v; }
  assert.sameValue(f.call({ v: 42 }), 42, "explicit receiver kept");
})();
