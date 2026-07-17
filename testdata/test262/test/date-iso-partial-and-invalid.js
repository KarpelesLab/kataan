/*---
description: ISO year-only/year-month parsing and Invalid Date string conversion
esid: sec-date-time-string-format
---*/
// Year-only and year-month ISO forms default the omitted fields to 1.
assert.sameValue(new Date("2024").toISOString(), "2024-01-01T00:00:00.000Z", "YYYY");
assert.sameValue(new Date("2024").getUTCFullYear(), 2024, "YYYY year");
assert.sameValue(new Date("2024-03").toISOString(), "2024-03-01T00:00:00.000Z", "YYYY-MM");
assert.sameValue(new Date("2024-03").getUTCMonth(), 2, "YYYY-MM month");
// Full forms still work.
assert.sameValue(new Date("2024-03-15").toISOString(), "2024-03-15T00:00:00.000Z", "YYYY-MM-DD");
assert.sameValue(new Date("2024-03-15T10:30:00Z").toISOString(), "2024-03-15T10:30:00.000Z", "full");

// Out-of-range month/day are rejected.
assert.sameValue(isNaN(new Date("2024-13").getTime()), true, "month 13 invalid");
assert.sameValue(isNaN(new Date("2024-03-32").getTime()), true, "day 32 invalid");

// An invalid Date stringifies to "Invalid Date".
assert.sameValue(String(new Date("xyz")), "Invalid Date", "unparseable string");
assert.sameValue(String(new Date(NaN)), "Invalid Date", "NaN timestamp");
assert.sameValue("" + new Date(NaN), "Invalid Date", "concatenation");
// A valid date is unaffected. (String()/toString use the human date format
// "Thu Jan 01 1970 ..."; the ISO "1970-01-01" prefix comes from toISOString.)
assert.sameValue(new Date(0).toISOString().slice(0, 10), "1970-01-01", "valid date");
