/*---
description: Intl.DateTimeFormat.prototype.formatToParts returns typed date/time parts
features: [Intl.DateTimeFormat]
---*/
var d = new Date(Date.UTC(2024, 0, 15, 14, 30, 45)); // Mon Jan 15 2024 14:30:45 UTC
function parts(o) { return new Intl.DateTimeFormat("en-US", Object.assign({ timeZone: "UTC" }, o)).formatToParts(d); }
function flat(o) { return parts(o).map(function (p) { return p.type + ":" + p.value; }).join("|"); }

// Numeric date breaks into month/literal/day/literal/year.
assert.sameValue(flat({ year: "numeric", month: "2-digit", day: "2-digit" }),
  "month:01|literal:/|day:15|literal:/|year:2024", "numeric date parts");

// Named month uses spaces and a comma literal.
assert.sameValue(flat({ year: "numeric", month: "long", day: "numeric" }),
  "month:January|literal: |day:15|literal:, |year:2024", "long month parts");

// Time has hour/minute/second with a dayPeriod (AM/PM).
assert.sameValue(flat({ hour: "numeric", minute: "2-digit", hour12: true }),
  "hour:2|literal::|minute:30|literal: |dayPeriod:PM", "time parts");

// A full date+time is a single flat list of typed parts.
assert.sameValue(flat({ weekday: "long", year: "numeric", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit" }),
  "weekday:Monday|literal:, |month:January|literal: |day:15|literal:, |year:2024|literal: at |hour:02|literal::|minute:30|literal: |dayPeriod:PM",
  "full date+time parts");

// Each part is an object with string type/value, and joining the values reproduces format().
var ps = parts({ year: "numeric", month: "long", day: "numeric" });
assert.sameValue(Array.isArray(ps), true, "result is an array");
assert.sameValue(typeof ps[0].type, "string", "part.type is a string");
assert.sameValue(typeof ps[0].value, "string", "part.value is a string");
var joined = ps.map(function (p) { return p.value; }).join("");
assert.sameValue(joined, new Intl.DateTimeFormat("en-US", { timeZone: "UTC", year: "numeric", month: "long", day: "numeric" }).format(d),
  "joined parts equal format()");

// NumberFormat.formatToParts is unaffected.
assert.sameValue(new Intl.NumberFormat("en-US").formatToParts(1234.5).map(function (p) { return p.type; }).join(","),
  "integer,group,integer,decimal,fraction", "number formatToParts still works");
