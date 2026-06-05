/*---
description: Strict-mode this is undefined for a plain call (lexically strict)
esid: sec-ordinarycallbindthis
flags: [onlyStrict]
---*/
(function () {
  "use strict";
  // `f` has no own directive but is defined in strict code -> strict (lexical).
  function f() { return this; }
  assert.sameValue(f(), undefined, "strict this is undefined");
  assert.sameValue(typeof f(), "undefined", "typeof strict this");
  assert.sameValue(f.call(null), null, "call(null) is kept in strict mode");
  assert.sameValue(f.call(5), 5, "primitive receiver kept in strict mode");
  var o = { x: 7, m() { return this.x; } };
  assert.sameValue(o.m(), 7, "method this still works in strict mode");
})();
