/*---
description: Intl.RelativeTimeFormat formats relative times (numeric "always"/"auto", en-US)
features: [Intl.RelativeTimeFormat]
---*/
function rtf(o, v, u) { return new Intl.RelativeTimeFormat("en", o).format(v, u); }

assert.sameValue(typeof Intl.RelativeTimeFormat, "function", "Intl.RelativeTimeFormat exists");

// numeric: "always" (default) — explicit "N units ago" / "in N units".
assert.sameValue(rtf({}, -1, "day"), "1 day ago", "-1 day");
assert.sameValue(rtf({}, 3, "week"), "in 3 weeks", "+3 weeks");
assert.sameValue(rtf({}, -2, "hour"), "2 hours ago", "-2 hours (plural)");
assert.sameValue(rtf({}, 1, "month"), "in 1 month", "+1 month (singular)");
assert.sameValue(rtf({}, 0, "day"), "in 0 days", "0 days under always");
assert.sameValue(rtf({}, 2.5, "hour"), "in 2.5 hours", "fractional value");

// numeric: "auto" — idiomatic phrases for adjacent units.
assert.sameValue(rtf({ numeric: "auto" }, -1, "day"), "yesterday", "auto -1 day");
assert.sameValue(rtf({ numeric: "auto" }, 0, "day"), "today", "auto 0 day");
assert.sameValue(rtf({ numeric: "auto" }, 1, "day"), "tomorrow", "auto +1 day");
assert.sameValue(rtf({ numeric: "auto" }, -1, "week"), "last week", "auto -1 week");
assert.sameValue(rtf({ numeric: "auto" }, 0, "week"), "this week", "auto 0 week");
assert.sameValue(rtf({ numeric: "auto" }, 1, "week"), "next week", "auto +1 week");
assert.sameValue(rtf({ numeric: "auto" }, -1, "year"), "last year", "auto -1 year");
assert.sameValue(rtf({ numeric: "auto" }, 0, "second"), "now", "auto 0 second -> now");
assert.sameValue(rtf({ numeric: "auto" }, 0, "hour"), "this hour", "auto 0 hour");
// auto falls back to explicit for non-adjacent values.
assert.sameValue(rtf({ numeric: "auto" }, 2, "day"), "in 2 days", "auto +2 days -> explicit");

// A plural unit argument is accepted; format is readable; callable without new.
assert.sameValue(rtf({}, -3, "days"), "3 days ago", "plural unit argument");
assert.sameValue(typeof new Intl.RelativeTimeFormat("en").format, "function", "format readable");
assert.sameValue(Intl.RelativeTimeFormat("en").format(-1, "day"), "1 day ago", "callable without new");
