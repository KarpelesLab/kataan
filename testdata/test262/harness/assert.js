// Test262 standard harness: the `assert` helpers (assert.js). Modeled as a
// method-bearing object so the engine can dispatch `assert.sameValue(...)` etc.
var assert = {};
assert._isSameValue = function (a, b) {
  if (a === b) return a !== 0 || 1 / a === 1 / b;
  return a !== a && b !== b;
};
assert.sameValue = function (actual, expected, message) {
  if (assert._isSameValue(actual, expected)) return;
  if (message === undefined) message = "";
  else message += " ";
  message += "Expected SameValue(" + String(actual) + ", " + String(expected) + ") to be true";
  throw new Test262Error(message);
};
assert.notSameValue = function (actual, unexpected, message) {
  if (!assert._isSameValue(actual, unexpected)) return;
  if (message === undefined) message = "";
  else message += " ";
  message += "Expected SameValue(" + String(actual) + ", " + String(unexpected) + ") to be false";
  throw new Test262Error(message);
};
assert.throws = function (expectedErrorConstructor, func, message) {
  if (typeof func !== "function") {
    throw new Test262Error("assert.throws requires a function");
  }
  try {
    func();
  } catch (thrown) {
    if (thrown instanceof expectedErrorConstructor) return;
    throw new Test262Error("threw the wrong error type: " + String(thrown));
  }
  throw new Test262Error((message || "") + " Expected a thrown error but none was thrown");
};
