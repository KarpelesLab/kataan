/*---
description: primitive wrappers serialize to/iterate as their boxed primitive; non-iterables throw TypeError
esid: sec-serializejsonproperty
---*/
// JSON.stringify of a wrapper uses the boxed primitive.
assert.sameValue(JSON.stringify(new Number(5)), "5", "Number wrapper");
assert.sameValue(JSON.stringify(new String("x")), '"x"', "String wrapper");
assert.sameValue(JSON.stringify(new Boolean(true)), "true", "Boolean wrapper");
assert.sameValue(
  JSON.stringify({ a: new Number(5), b: new String("x"), c: new Boolean(false) }),
  '{"a":5,"b":"x","c":false}',
  "wrappers nested in an object"
);

// A String wrapper is iterable (its characters); spread and for-of work.
assert.sameValue([...new String("abc")].join(","), "a,b,c", "spread String wrapper");
var out = [];
for (var ch of new String("xy")) out.push(ch);
assert.sameValue(out.join(","), "x,y", "for-of String wrapper");

// A Number/Boolean wrapper (and a bare number) is not iterable -> TypeError.
assert.throws(TypeError, function () { return [...new Number(5)]; }, "Number wrapper not iterable");
assert.throws(TypeError, function () { return [...42]; }, "number not iterable");

// Plain values are unaffected.
assert.sameValue(JSON.stringify({ a: 5, b: "x" }), '{"a":5,"b":"x"}', "plain object");
assert.sameValue([..."ab"].join(","), "a,b", "plain string spread");
