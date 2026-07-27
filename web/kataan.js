// Main-thread handle for the engine running in `kataan-worker.js`.
//
// Owns the worker's lifetime and enforces a wall-clock budget: an infinite loop
// inside wasm cannot be interrupted, so exceeding the budget terminates the
// worker and spawns a replacement. That is why every call is queued — a
// terminated worker takes any in-flight work with it.

export class Kataan {
  /** @param {{timeoutMs?: number}} options */
  constructor({ timeoutMs = 5000 } = {}) {
    this.timeoutMs = timeoutMs;
    /** Resolves once the module has been fetched and instantiated. */
    this.ready = null;
    this._worker = null;
    this._pending = new Map();
    this._nextId = 1;
    this._spawn();
  }

  _spawn() {
    this._worker = new Worker(new URL('kataan-worker.js', import.meta.url), { type: 'module' });
    this.ready = new Promise((resolve, reject) => {
      this._onReady = { resolve, reject };
    });
    this._worker.onmessage = ({ data }) => {
      if (data.ready !== undefined) {
        if (data.ready) this._onReady.resolve();
        else this._onReady.reject(new Error(data.error));
        return;
      }
      const entry = this._pending.get(data.id);
      if (!entry) return;
      this._pending.delete(data.id);
      clearTimeout(entry.timer);
      entry.resolve(data);
    };
  }

  /** Replaces the worker, failing everything it was carrying. */
  _restart(reason) {
    this._worker.terminate();
    for (const [, entry] of this._pending) {
      clearTimeout(entry.timer);
      entry.resolve({ ok: false, printed: '', value: reason, ms: this.timeoutMs, timedOut: true });
    }
    this._pending.clear();
    this._spawn();
  }

  /**
   * Runs `source` through one pipeline stage.
   *
   * @param {string} source
   * @param {'run'|'lex'|'parse'} [stage] Defaults to running the script.
   * @returns {Promise<{ok: boolean, printed: string, value: string, ms: number,
   *                    timedOut?: boolean}>}
   *   Never rejects: a syntax error, an uncaught throw and a timeout are all
   *   ordinary results, because in a playground they are things to display
   *   rather than exceptions to handle.
   */
  run(source, stage = 'run') {
    const id = this._nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(
        () => this._restart(`terminated: still running after ${this.timeoutMs} ms`),
        this.timeoutMs,
      );
      this._pending.set(id, { resolve, timer });
      this._worker.postMessage({ id, source, stage });
    });
  }
}
