/*---
description: Intl.DateTimeFormat honors weekday/year/month/day/hour/minute/second/hour12/era/dateStyle/timeStyle (en-US, UTC)
features: [Intl.DateTimeFormat]
---*/
var d = new Date(Date.UTC(2024, 0, 15, 14, 30, 45)); // Mon Jan 15 2024 14:30:45 UTC
function dtf(o) { return new Intl.DateTimeFormat("en-US", Object.assign({ timeZone: "UTC" }, o)).format(d); }

// Default is the numeric date.
assert.sameValue(dtf({}), "1/15/2024", "default numeric date");
assert.sameValue(dtf({ year: "numeric", month: "2-digit", day: "2-digit" }), "01/15/2024", "2-digit month/day");

// Named months and weekdays.
assert.sameValue(dtf({ year: "numeric", month: "long", day: "numeric" }), "January 15, 2024", "long month");
assert.sameValue(dtf({ month: "short", day: "numeric" }), "Jan 15", "short month + day");
assert.sameValue(dtf({ weekday: "long" }), "Monday", "long weekday");
assert.sameValue(dtf({ weekday: "short" }), "Mon", "short weekday");
assert.sameValue(dtf({ month: "long", year: "numeric" }), "January 2024", "month + year (no day)");

// Time, 12- and 24-hour.
assert.sameValue(dtf({ hour: "2-digit", minute: "2-digit", second: "2-digit" }), "02:30:45 PM", "12-hour default");
assert.sameValue(dtf({ hour: "numeric", minute: "2-digit", hour12: true }), "2:30 PM", "numeric hour 12h");
assert.sameValue(dtf({ hour: "2-digit", minute: "2-digit", hour12: false }), "14:30", "24-hour");

// Midnight and noon in 12-hour.
function h12(ms) { return new Intl.DateTimeFormat("en-US", { timeZone: "UTC", hour: "numeric", hour12: true }).format(new Date(ms)); }
assert.sameValue(h12(Date.UTC(2024, 0, 1, 0, 0)), "12 AM", "midnight -> 12 AM");
assert.sameValue(h12(Date.UTC(2024, 0, 1, 12, 0)), "12 PM", "noon -> 12 PM");

// Combined date + time uses " at " for a named-month date.
assert.sameValue(dtf({ weekday: "long", year: "numeric", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit" }),
  "Monday, January 15, 2024 at 02:30 PM", "full date + time");

// dateStyle / timeStyle presets.
assert.sameValue(dtf({ dateStyle: "full" }), "Monday, January 15, 2024", "dateStyle full");
assert.sameValue(dtf({ dateStyle: "medium" }), "Jan 15, 2024", "dateStyle medium");
assert.sameValue(dtf({ timeStyle: "short" }), "2:30 PM", "timeStyle short");

// era.
assert.sameValue(dtf({ year: "numeric", era: "short" }), "2024 AD", "era short");
