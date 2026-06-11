/*---
description: Date.parse handles ISO 8601 numeric timezone offsets (+HH:MM / -HH:MM)
esid: sec-date.parse
---*/
// A positive offset means the wall-clock time is ahead of UTC, so UTC subtracts it.
assert.sameValue(Date.parse("2024-01-15T12:30:45+05:00"), 1705303845000, "+05:00");
assert.sameValue(Date.parse("2024-01-15T12:30:45-08:00"), 1705350645000, "-08:00");
assert.sameValue(Date.parse("2024-01-15T12:30:45.500+05:00"), 1705303845500, "offset with milliseconds");
assert.sameValue(Date.parse("2024-01-15T12:30:45+00:00"), 1705321845000, "+00:00 equals UTC");
assert.sameValue(Date.parse("2024-01-15T12:30:45+0500"), 1705303845000, "offset without a colon");

// Equivalence: an offset is the same instant as the shifted Z time.
assert.sameValue(Date.parse("2024-01-15T12:30:45+05:00"), Date.parse("2024-01-15T07:30:45Z"), "+05:00 == Z-5h");
assert.sameValue(Date.parse("2024-01-15T12:30:45-08:00"), Date.parse("2024-01-15T20:30:45Z"), "-08:00 == Z+8h");

// Z and no-designator (taken as UTC here) are unchanged.
assert.sameValue(Date.parse("2024-01-15T12:30:45Z"), 1705321845000, "Z");
assert.sameValue(Date.parse("2024-01-15T12:30:45"), 1705321845000, "no designator -> UTC");

// new Date(string) applies the offset, and toISOString round-trips to UTC.
assert.sameValue(new Date("2024-01-15T12:30:45+05:00").getUTCHours(), 7, "getUTCHours after offset");
assert.sameValue(new Date(Date.parse("2024-06-15T12:00:00+02:00")).toISOString(), "2024-06-15T10:00:00.000Z", "round-trip to UTC");

// An invalid offset is NaN.
assert.sameValue(isNaN(Date.parse("2024-01-15T12:30:45+99:99x")), true, "invalid offset -> NaN");
