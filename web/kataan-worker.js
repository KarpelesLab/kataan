// Worker that owns the kataan WebAssembly module.
//
// The engine runs here rather than on the main thread for one reason: a
// playground must survive `for (;;) {}`. There is no way to interrupt a running
// wasm instance from outside, so the page's escape hatch is to terminate this
// worker and start a fresh one — which is only possible if the engine was never
// on the main thread to begin with.
//
// Protocol: post `{ id, source, stage }`, receive
// `{ id, ok, printed, value, ms }`. `stage` selects how far down the pipeline to
// go: `lex` and `parse` stop after that stage and dump it; `run` executes.

let exports = null;
let ready = null;

/** Loads and instantiates the module (once), resolving to its exports. */
function boot() {
  ready ??= WebAssembly.instantiateStreaming(fetch(new URL('kataan.wasm', import.meta.url)), {})
    .catch(async (streamingError) => {
      // `instantiateStreaming` needs `Content-Type: application/wasm`; fall back
      // to the buffered path for hosts that serve it as octet-stream.
      const response = await fetch(new URL('kataan.wasm', import.meta.url));
      if (!response.ok) throw streamingError;
      return WebAssembly.instantiate(await response.arrayBuffer(), {});
    })
    .then((result) => {
      exports = result.instance.exports;
      return exports;
    });
  return ready;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** A fresh view; wasm memory can be detached and regrown by any call into it. */
const bytes = () => new Uint8Array(exports.memory.buffer);

/** The ABI entry point for each pipeline stage. */
const ENTRY = { run: 'kt_eval_capture', lex: 'kt_lex', parse: 'kt_parse' };

/**
 * Runs `source` through the requested stage.
 *
 * The ABI follows an in/out length convention: `*lenPtr` carries the buffer's
 * capacity in and the produced length out, so a too-small buffer reports what it
 * needed rather than truncating — hence the single retry at the reported size.
 *
 * `kt_eval_capture` writes the script's console output and its completion value
 * into one buffer separated by a NUL; the dump stages write a single blob, so
 * there is no NUL and the whole thing is the value.
 */
function evaluate(source, stage) {
  const src = encoder.encode(source);
  const srcPtr = exports.kt_alloc(src.length || 1);
  bytes().set(src, srcPtr);

  let capacity = 64 * 1024;
  let outPtr = 0;
  let lenPtr = 0;
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      outPtr = exports.kt_alloc(capacity);
      lenPtr = exports.kt_alloc(4);
      new DataView(exports.memory.buffer).setUint32(lenPtr, capacity, true);

      const status = exports[ENTRY[stage] ?? ENTRY.run](srcPtr, src.length, outPtr, lenPtr);
      const produced = new DataView(exports.memory.buffer).getUint32(lenPtr, true);

      // KT_BUFFER_TOO_SMALL (-2): `produced` is the size actually required.
      if (status === -2 && attempt === 0) {
        exports.kt_free(outPtr, capacity);
        exports.kt_free(lenPtr, 4);
        capacity = produced + 1;
        outPtr = lenPtr = 0;
        continue;
      }

      const raw = bytes().slice(outPtr, outPtr + produced);
      const split = raw.indexOf(0);
      return {
        ok: status === 0,
        printed: decoder.decode(raw.subarray(0, split < 0 ? produced : split)),
        value: split < 0 ? '' : decoder.decode(raw.subarray(split + 1)),
      };
    }
    return { ok: false, printed: '', value: 'output too large' };
  } finally {
    exports.kt_free(srcPtr, src.length || 1);
    if (outPtr) exports.kt_free(outPtr, capacity);
    if (lenPtr) exports.kt_free(lenPtr, 4);
  }
}

self.onmessage = async (event) => {
  const { id, source, stage } = event.data;
  try {
    await boot();
    const started = performance.now();
    const result = evaluate(source, stage);
    self.postMessage({ id, ...result, ms: performance.now() - started });
  } catch (error) {
    self.postMessage({ id, ok: false, printed: '', value: String(error), ms: 0 });
  }
};

// Let the page know the module is warm, so the first Run is not also the first
// download.
boot().then(
  () => self.postMessage({ ready: true }),
  (error) => self.postMessage({ ready: false, error: String(error) }),
);
