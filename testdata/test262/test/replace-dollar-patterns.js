/*---
description: String.replace with $ replacement patterns
esid: sec-string.prototype.replace
---*/
assert.sameValue("hello".replace("l", "[$&]"), "he[l]lo", "$& is the match");
assert.sameValue("2024-06".replace(/(\d+)-(\d+)/, "$2/$1"), "06/2024", "$1 $2 groups");
assert.sameValue("abc".replace(/b/, "$`"), "aac", "$` is the prefix");
assert.sameValue("abc".replace(/b/, "$'"), "acc", "$' is the suffix");
assert.sameValue("a.b.c".replace(/\./g, "-"), "a-b-c");
assert.sameValue("hello world".replace(/(\w+) (\w+)/, "$2 $1"), "world hello", "swap words");
assert.sameValue("test".replace(/t/g, "$$"), "$es$", "$$ is literal dollar");
assert.sameValue("x".replace(/x/, "$1"), "$1", "no group 1 leaves literal");
assert.sameValue("abcabc".replace(/abc/g, "[$&]"), "[abc][abc]");
