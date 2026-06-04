/*---
description: An empty if-condition is a SyntaxError at parse time
esid: sec-if-statement
negative:
  phase: parse
  type: SyntaxError
---*/
if () {}
