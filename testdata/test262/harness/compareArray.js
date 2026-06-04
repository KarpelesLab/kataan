// Test262 standard harness: array structural comparison (compareArray.js).
function compareArray(a, b) {
  if (b.length !== a.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (b[i] !== a[i]) return false;
  }
  return true;
}
assert.compareArray = function (actual, expected, message) {
  assert(compareArray(actual, expected),
    "Expected [" + actual.join(", ") + "] to equal [" + expected.join(", ") + "]. " + (message || ""));
};
