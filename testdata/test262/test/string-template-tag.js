/*---
description: Tagged templates and template processing
esid: sec-tagged-templates
---*/
function html(strings, ...values) {
  return strings.reduce(function (acc, str, i) {
    return acc + str + (i < values.length ? String(values[i]) : "");
  }, "");
}
var name = "World";
var count = 42;
assert.sameValue(html`Hello ${name}, count is ${count}!`, "Hello World, count is 42!");
function upper(strings, ...values) {
  return strings.map(function (s, i) {
    return s + (values[i] !== undefined ? values[i].toUpperCase() : "");
  }).join("");
}
assert.sameValue(upper`say ${"hi"} to ${"all"}`, "say HI to ALL");
function count2(strings, ...values) { return values.length; }
assert.sameValue(count2`${1}${2}${3}`, 3);
assert.sameValue(count2`no values`, 0);
var x = 5, y = 3;
assert.sameValue(`${x} + ${y} = ${x + y}`, "5 + 3 = 8");
assert.sameValue(`${x > y ? "greater" : "less"}`, "greater");
function sum(strings, ...nums) { return nums.reduce(function (a, b) { return a + b; }, 0); }
assert.sameValue(sum`${1} ${2} ${3} ${4}`, 10);
