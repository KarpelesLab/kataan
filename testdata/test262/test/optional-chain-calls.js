/*---
description: Optional chaining with method calls and nullish
esid: sec-optional-chains
---*/
var api = {
  data: { items: [1, 2, 3] },
  getItems() { return this.data.items; },
  getNull() { return null; }
};
assert.sameValue(api?.getItems()?.length, 3);
assert.sameValue(api?.getItems?.()?.[0], 1, "optional call and index");
assert.sameValue(api?.missing?.(), undefined, "optional call on missing");
assert.sameValue(api?.getNull()?.length, undefined, "null result");
assert.sameValue(api?.data?.items?.length, 3);
assert.sameValue(api?.data?.missing?.value ?? "default", "default");
var nested = { a: { b: { c: function () { return 42; } } } };
assert.sameValue(nested?.a?.b?.c?.(), 42, "deep optional call");
assert.sameValue(nested?.a?.x?.c?.(), undefined, "deep optional short-circuit");
var arr = [{ fn: function () { return "a"; } }, null];
assert.sameValue(arr[0]?.fn?.(), "a");
assert.sameValue(arr[1]?.fn?.(), undefined);
var obj = { method: null };
assert.sameValue(obj.method?.(), undefined, "null method call");
assert.sameValue((null)?.anything?.deep?.path, undefined);
