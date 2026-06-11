/*---
description: function/class names — declarations, named expressions, and binding inference
esid: sec-function-instances-name
---*/
// Declarations and named expressions keep their own name.
function topfoo() {}
assert.sameValue(topfoo.name, "topfoo", "declaration");
assert.sameValue((function bar() {}).name, "bar", "named expression");

// A nested declaration.
function outer() { function inner() {} return inner.name; }
assert.sameValue(outer(), "inner", "nested declaration");

// NamedEvaluation: an anonymous function/arrow bound to an identifier infers that name.
var anon = function () {};
assert.sameValue(anon.name, "anon", "var-inferred function");
const arrow = () => {};
assert.sameValue(arrow.name, "arrow", "var-inferred arrow");
let h = function () {};
assert.sameValue(h.name, "h", "let-inferred");

// A named expression's own name wins over the binding.
var x = function baz() {};
assert.sameValue(x.name, "baz", "named expression over binding");

// Object methods/properties and classes.
var o = { method() {}, fn: () => 1, prop: function () {} };
assert.sameValue(o.method.name, "method", "concise method");
assert.sameValue(o.fn.name, "fn", "arrow property");
assert.sameValue(o.prop.name, "prop", "function property");
class Foo {}
assert.sameValue(Foo.name, "Foo", "class name");

// A truly anonymous function has an empty name.
assert.sameValue((function () {}).name, "", "anonymous");

// length is unaffected.
assert.sameValue((function (a, b) {}).length, 2, "length");
