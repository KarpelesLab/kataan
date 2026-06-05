/*---
description: Deep optional chaining with method calls and defaults
esid: sec-optional-chains
---*/
var data = {
  user: { profile: { settings: { theme: "dark" } } }
};
assert.sameValue(data?.user?.profile?.settings?.theme, "dark");
assert.sameValue(data?.user?.profile?.missing?.value, undefined);
assert.sameValue(data?.admin?.profile?.theme ?? "default", "default");
var api = {
  getData() { return { items: [1, 2, 3] }; },
  getNull() { return null; }
};
assert.sameValue(api.getData()?.items?.length, 3);
assert.sameValue(api.getNull()?.items?.length, undefined);
assert.sameValue(api?.getData?.()?.items?.[0], 1, "optional call and index");
assert.sameValue(api?.missing?.(), undefined, "optional call on missing method");
var arr = [{ x: 1 }, null, { x: 3 }];
assert.sameValue(arr[0]?.x, 1);
assert.sameValue(arr[1]?.x, undefined, "null element");
assert.sameValue(arr[2]?.x ?? 0, 3);
var nested = { a: { b: null } };
assert.sameValue(nested?.a?.b?.c?.d, undefined, "stops at null");
