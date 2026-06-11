/*---
description: Intl.Segmenter.segment yields grapheme/word/sentence segments (iterable; en)
features: [Intl.Segmenter]
---*/
function seg(g, s) { return [].concat.apply([], [Array.prototype.slice.call(new Intl.Segmenter("en", g ? { granularity: g } : undefined).segment(s))]); }

assert.sameValue(typeof Intl.Segmenter, "function", "Intl.Segmenter exists");

// Grapheme (default): one code point per segment, with code-point indices.
var g = seg("grapheme", "abc");
assert.sameValue(g.map(function (x) { return x.segment; }).join(","), "a,b,c", "grapheme segments");
assert.sameValue(g.map(function (x) { return x.index; }).join(","), "0,1,2", "grapheme indices");
assert.sameValue(seg(undefined, "ab").map(function (x) { return x.segment; }).join(","), "a,b", "default granularity is grapheme");

// Word (UAX-29): word-like runs, with punctuation and whitespace as distinct segments.
var w = seg("word", "Hello, world!");
assert.sameValue(w.map(function (x) { return x.segment; }).join("|"), "Hello|,| |world|!", "word segments");
assert.sameValue(w.map(function (x) { return x.isWordLike; }).join(","), "true,false,false,true,false", "isWordLike");
assert.sameValue(w.map(function (x) { return x.index; }).join(","), "0,5,6,7,12", "word indices");

// Sentence: split after terminating punctuation + spaces.
var s = seg("sentence", "Hi there. How are you? Good!");
assert.sameValue(s.length, 3, "three sentences");
assert.sameValue(s[0].segment, "Hi there. ", "first sentence");
assert.sameValue(s[1].segment, "How are you? ", "second sentence");
assert.sameValue(s[2].segment, "Good!", "third sentence");

// Each segment exposes the full input; the result is iterable with for-of.
assert.sameValue(w[0].input, "Hello, world!", "segment.input is the whole string");
var collected = [];
for (var part of new Intl.Segmenter("en").segment("xy")) { collected.push(part.segment); }
assert.sameValue(collected.join(","), "x,y", "for-of over segments");

// segment is a readable method and works without new.
assert.sameValue(typeof new Intl.Segmenter("en").segment, "function", "segment is readable");
