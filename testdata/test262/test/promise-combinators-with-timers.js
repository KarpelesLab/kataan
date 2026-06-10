/*---
description: Promise.all/race/allSettled/any observe timer-backed (genuinely pending) promises
features: [host-timers]
---*/
function d(ms, v) { return new Promise(function (r) { setTimeout(function () { r(v); }, ms); }); }
var log = [];

// all: waits for every (timer-backed) promise and collects values in order.
Promise.all([d(10, "a"), d(5, "b")]).then(function (v) { log.push("all:" + v.join(",")); });
// race: the first to SETTLE wins (not the first listed).
Promise.race([d(20).then(function () { return "slow"; }), d(5).then(function () { return "fast"; })]).then(function (v) { log.push("race:" + v); });
// race: an immediate promise beats a timer.
Promise.race([Promise.resolve("imm"), d(5, "timer")]).then(function (v) { log.push("raceimm:" + v); });
// race: first settle is a rejection.
Promise.race([d(5).then(function () { throw "boom"; }), d(20, "ok")]).then(function () { log.push("rno"); }, function (e) { log.push("racerej:" + e); });
// allSettled: mixes a timer fulfill with an immediate reject.
Promise.allSettled([d(5, "ok"), Promise.reject("e")]).then(function (r) { log.push("settled:" + r.map(function (o) { return o.status; }).join(",")); });
// any: first to fulfill (a timer) wins over an earlier reject.
Promise.any([Promise.reject("x"), d(5, "won")]).then(function (v) { log.push("any:" + v); });
// await on timer-backed promises.
(async function () { var x = await d(5, 10); var y = await d(3, 20); log.push("async:" + (x + y)); })();

setTimeout(function () {
  assert.sameValue(log.indexOf("all:a,b") >= 0, true, "all collects timer values");
  assert.sameValue(log.indexOf("race:fast") >= 0, true, "race: first to settle");
  assert.sameValue(log.indexOf("raceimm:imm") >= 0, true, "race: immediate beats timer");
  assert.sameValue(log.indexOf("racerej:boom") >= 0, true, "race: first settle rejects");
  assert.sameValue(log.indexOf("settled:fulfilled,rejected") >= 0, true, "allSettled");
  assert.sameValue(log.indexOf("any:won") >= 0, true, "any: first timer fulfill");
  assert.sameValue(log.indexOf("async:30") >= 0, true, "await on timers");
}, 100);
