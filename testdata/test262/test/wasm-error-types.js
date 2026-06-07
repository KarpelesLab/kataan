/*---
description: WebAssembly.CompileError / LinkError / RuntimeError exist as Error subclasses
features: [WebAssembly]
---*/
assert.sameValue(typeof WebAssembly.CompileError, "function", "CompileError exists");
assert.sameValue(typeof WebAssembly.LinkError, "function", "LinkError exists");
assert.sameValue(typeof WebAssembly.RuntimeError, "function", "RuntimeError exists");

// Each is a proper Error subclass with the right name and message.
var c = new WebAssembly.CompileError("bad");
assert.sameValue(c.name, "CompileError", "CompileError name");
assert.sameValue(c.message, "bad", "CompileError message");
assert.sameValue(c instanceof Error, true, "CompileError is an Error");
assert.sameValue(c instanceof WebAssembly.CompileError, true, "instanceof itself");
assert.sameValue(c.toString(), "CompileError: bad", "toString");

var r = new WebAssembly.RuntimeError();
assert.sameValue(r.name, "RuntimeError", "RuntimeError name");
assert.sameValue(r instanceof Error, true, "RuntimeError is an Error");

// Compiling a malformed module throws a WebAssembly.CompileError.
var threw = null;
try { new WebAssembly.Module(new Uint8Array([1, 2, 3])); } catch (e) { threw = e; }
assert.sameValue(threw instanceof WebAssembly.CompileError, true, "bad module -> CompileError");
assert.sameValue(threw instanceof Error, true, "bad module error is an Error");

// They are namespaced, not globals.
assert.sameValue(typeof globalThis.CompileError, "undefined", "CompileError is not a global");
