# web/

The documentation and playground published to GitHub Pages by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml).

It is a static page: no bundler, no `package.json`, no build step beyond
compiling the crate to WebAssembly. Vue is used from its global build and is
vendored at deploy time so the published site has no runtime dependency on a
CDN.

## Layout

| file               | what it is                                                          |
| ------------------ | ------------------------------------------------------------------- |
| `index.html`       | The page. The Vue template lives here as an in-DOM template.         |
| `app.js`           | The Vue app: editor state, stage switching, the figures on the page. |
| `kataan.js`        | Main-thread handle for the engine. Owns the worker and its timeout.  |
| `kataan-worker.js` | Loads the wasm module and calls the C ABI.                           |
| `examples.js`      | The playground snippets.                                             |
| `styles.css`       | The visual system.                                                   |

The engine runs in a worker because a running WebAssembly instance cannot be
interrupted: the only way to stop `for (;;) {}` is to terminate the worker and
spawn a new one, which is impossible if the engine is on the main thread.

## Running it locally

Build the module, assemble a directory, and serve it. The page cannot be opened
with `file://` — module workers and `WebAssembly.instantiateStreaming` both need
a real origin.

```console
$ cargo rustc --lib --target wasm32-unknown-unknown --profile wasm-web \
      --no-default-features --features std,regex,intl,module,ffi \
      --crate-type cdylib

$ mkdir -p /tmp/site/vendor && cp web/* /tmp/site/
$ cp target/wasm32-unknown-unknown/wasm-web/kataan.wasm /tmp/site/
$ curl -fsSL https://unpkg.com/vue@3.5.13/dist/vue.global.prod.js \
      -o /tmp/site/vendor/vue.global.prod.js

$ python3 -m http.server 8000 --directory /tmp/site
```

## The browser build

`--no-default-features --features std,regex,intl,module,ffi` — the language, the
standard library, the regular-expression engine, `Intl` and `Temporal`, reached
through the C ABI. It leaves out `host` (no event loop, `fetch` or file system in
a tab), `cli`, and `crypto` (no OS entropy source).

Two consequences worth knowing:

- **There is no clock.** `wasm32-unknown-unknown` has no time source, so
  `Date.now()` reads `0`. That is the target, not the engine.
- **`intl` is most of the download.** The module is ~2.4 MB gzipped; without
  `intl` it is ~0.9 MB. The difference is Unicode and CLDR data, which is what
  makes `Intl` and the non-ISO calendars work — worth carrying for a page whose
  job is to show them off, but the first thing to drop when embedding.

The `wasm-web` profile in `Cargo.toml` trades compile time for size
(`opt-level = "z"`, fat LTO, one codegen unit, stripped). It deliberately leaves
`panic` at the default: the C ABI turns a caught panic into `KT_INTERNAL`, and
`panic = "abort"` would make it a trap instead.
