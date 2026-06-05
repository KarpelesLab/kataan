/*---
description: An unparenthesized unary operator before ** is a SyntaxError
esid: sec-exp-operator
negative:
  phase: parse
  type: SyntaxError
---*/
-2 ** 2;
