// Test262 standard harness: the Test262Error type used to signal failures
// (sta.js). Thrown only on assertion failure.
function Test262Error(message) {
  this.message = message || "";
  this.name = "Test262Error";
}
