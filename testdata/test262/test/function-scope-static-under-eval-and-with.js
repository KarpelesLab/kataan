/*---
description: >
  A function's free identifiers resolve through its [[Environment]] (where it
  was defined), never through the caller's dynamic scope. A global function
  called from a direct eval or a `with` body still reads, writes, deletes and
  calls the global bindings; a function defined inside eval code or a `with`
  body keeps seeing that scope when called from elsewhere.
esid: sec-getidentifierreference
---*/
var gv = 1;
let gl = 10;
const gc = 100;
globalThis.gp = 1000;
globalThis.gd = 1;

function bump() { gv = gv + 1; gl += 1; gp++; return [gv, gl, gc, gp].join(","); }
function bumpStrict() { "use strict"; gv++; gl++; gp += 1; return [gv, gl, gp].join(","); }
function kinds() { return [typeof gv, typeof gl, typeof gp, typeof undeclaredName].join(","); }
function delGd() { return delete gd; }
function who() { return "global who"; }
function callWho() { return who(); }
function writeConst() { try { gc = 1; return "no error"; } catch (e) { return e.constructor === TypeError; } }
function strictUnresolvable() {
  "use strict";
  try { undeclaredStrict = 1; return "no error"; } catch (e) { return e.constructor === ReferenceError; }
}
function sloppyUnresolvable() { implicitGlobal = "made"; }

function viaEval() {
  return eval(
    "var gv = 'e', gp = 'e', gd = 'e', undeclaredName = 'e', undeclaredStrict = 'e', implicitGlobal = 'e';" +
    "var who = function () { return 'eval who'; };" +
    "let gl = 'e';" +
    "[bump(), bumpStrict(), kinds(), delGd(), callWho(), writeConst(), strictUnresolvable()," +
    " (sloppyUnresolvable(), implicitGlobal), gv, gl, gp, gd].join('|')"
  );
}
assert.sameValue(viaEval(), "2,11,100,1001|3,12,1002|number,number,number,undefined|true|global who|true|true|e|e|e|e|e",
  "callee called from a direct eval");
assert.sameValue([gv, gl, gp, typeof gd, implicitGlobal].join(), "3,12,1002,undefined,made", "globals after eval");

var obj = { gv: "o", gl: "o", gc: "o", gp: "o", gd: "o", undeclaredName: "o", undeclaredStrict: "o",
            who: function () { return "with who"; } };
globalThis.gd = 1;
var r;
with (obj) {
  r = [bump(), bumpStrict(), kinds(), delGd(), callWho(), writeConst(), strictUnresolvable()].join("|");
}
assert.sameValue(r, "4,13,100,1003|5,14,1004|number,number,number,undefined|true|global who|true|true",
  "callee called from a with body");
assert.sameValue([obj.gv, obj.gl, obj.gp, obj.gd].join(), "o,o,o,o", "with object untouched");
assert.sameValue([gv, gl, gp, typeof gd].join(), "5,14,1004,undefined", "globals after with");

// Eval inside with, and nested evals.
with (obj) { eval("var gv = 'ew'; bump()"); }
assert.sameValue(gv, 6, "eval inside with");
function nested() { return eval("var gv = 'n1'; eval('var gv = \"n2\"; bump(); gv')"); }
assert.sameValue(nested(), "n2", "nested eval keeps its own var");
assert.sameValue(gv, 7, "nested eval: the callee wrote the global");

// The reverse: functions defined inside eval code / a with body.
function maker() {
  var x = "fx";
  eval("var x = 'ex'; var get = function () { return x; }; var set = function (v) { x = v; };");
  return [get, set];
}
var gs = maker();
var x = "gx";
function otherEval() { return eval("var x = 'other'; gs[1]('S'); gs[0]()"); }
assert.sameValue(otherEval(), "S", "eval-defined function sees the eval's var");
assert.sameValue(x, "gx", "eval-defined function did not write the global");

var wobj = { y: "wy" };
var wget, wset;
with (wobj) {
  wget = function () { return y; };
  wset = function (v) { y = v; };
}
var y = "gy";
function viaEval2() { return eval("var y = 'e2'; wset('W'); wget()"); }
assert.sameValue(viaEval2(), "W", "with-defined function sees the with object");
assert.sameValue(wobj.y, "W", "with-defined function wrote the with object");
assert.sameValue(y, "gy", "with-defined function did not write the global");
