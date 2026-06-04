// Conformance fixture: regular expressions (the `regex` feature, on by default).

// --- test / exec ---
assert(/^\d{4}-\d{2}-\d{2}$/.test("2026-06-04"), "date shape");
assert(!/^\d+$/.test("12a"), "non-digit rejected");
const m = /(\w+)\s+(\w+)/.exec("hello world");
assertEq(m[0], "hello world");
assertEq(m[1], "hello");
assertEq(m[2], "world");
assertEq(m.index, 0);

// --- String.match ---
assertEq("a1b2c3".match(/\d/g).join(""), "123");
assertEq("no digits".match(/\d/), null);
assertEq("key=value".match(/(\w+)=(\w+)/)[2], "value");

// --- String.replace with captures ---
assertEq("John Smith".replace(/(\w+)\s(\w+)/, "$2, $1"), "Smith, John");
assertEq("aaa".replace(/a/g, "b"), "bbb");
assertEq("camelCase".replace(/([A-Z])/g, "_$1"), "camel_Case");

// --- split / search ---
assertEq("1, 2,3 ,4".split(/\s*,\s*/).join("|"), "1|2|3|4");
assertEq("find the needle".search(/needle/), 9);
assertEq("nope".search(/xyz/), -1);

// --- flags & classes ---
assert(/hello/i.test("HELLO"), "case-insensitive");
assert(/^bar/m.test("foo\nbar"), "multiline anchor");
assert(/a.c/s.test("a\nc"), "dotall");
assertEq("The 3 cats ate 12 fish".match(/\d+/g).length, 2);

// --- constructor + instanceof ---
const re = new RegExp("\\bword\\b", "g");
assert(re instanceof RegExp, "instanceof RegExp");
assertEq(re.source, "\\bword\\b");
assert(re.global, "global flag exposed");
assertEq("a word and another word".match(/\bword\b/g).length, 2);

// --- greedy vs lazy ---
assertEq("<a><b>".match(/<.*>/)[0], "<a><b>");
assertEq("<a><b>".match(/<.*?>/)[0], "<a>");
