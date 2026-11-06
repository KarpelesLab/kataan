//! The `kataan` command-line binary, run as a child process.
#![cfg(feature = "cli")]

use std::process::Command;

/// Runs `kataan run -e <source>` and returns (stdout, stderr, success).
fn run_e(source: &str) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_kataan"))
        .args(["run", "-e", source])
        .output()
        .expect("spawn kataan");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// An uncaught exception does not swallow what the program printed before
/// it: the earlier `console.log` lines reach stdout, the error reaches
/// stderr, and the exit status is a failure (all output was dropped).
#[test]
fn uncaught_throw_keeps_earlier_output() {
    let (stdout, stderr, ok) = run_e(
        r#"console.log("before"); (function () { console.log("in f"); throw new TypeError("boom"); })();"#,
    );
    assert_eq!(stdout, "before\nin f\n");
    assert!(stderr.contains("TypeError: boom"), "stderr: {stderr}");
    assert!(!ok);
}

/// A program that completes prints its output and its completion value.
#[test]
fn completed_program_prints_output_and_completion() {
    let (stdout, _, ok) = run_e(r#"console.log("x"); 1 + 2"#);
    assert_eq!(stdout, "x\n3\n");
    assert!(ok);
}

/// `kataan run` does not expose the Test262 host hooks: they are harness
/// plumbing (`$262_IsHTMLDDA`, `$262_agent_start`, …), not language globals.
#[test]
fn no_test262_host_hooks_in_cli() {
    let (stdout, _, ok) = run_e(
        r#"console.log(Object.getOwnPropertyNames(globalThis).filter(function (n) { return n.indexOf("262") >= 0; }).join(), typeof $262_IsHTMLDDA, typeof $262_createRealm, typeof $262_agent_start)"#,
    );
    assert!(
        stdout.starts_with(" undefined undefined undefined\n"),
        "stdout: {stdout}"
    );
    assert!(ok);
}

/// Writes `files` (name, source) into a fresh directory for one test and
/// returns its path.
fn fixture_dir(tag: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("kataan-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    for (name, src) in files {
        std::fs::write(dir.join(name), src).expect("write fixture");
    }
    dir
}

/// Runs `kataan <mode> <path>` from an unrelated working directory and returns
/// (stdout, stderr, success).
fn run_file(mode: &str, path: &std::path::Path) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_kataan"))
        .args([mode, path.to_str().expect("utf-8 path")])
        .current_dir(std::env::temp_dir())
        .output()
        .expect("spawn kataan");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// A `.mjs` file runs as an ES module under both `run` and `hostrun`: static
/// imports resolve next to it, bindings are live, and top-level await works.
#[test]
fn mjs_runs_as_module_graph() {
    let dir = fixture_dir(
        "mjs",
        &[
            (
                "main.mjs",
                "import { n, bump } from './dep.mjs';\nbump();\nconst m = await import('./dep.mjs');\nconsole.log(n, m.n, typeof this, await Promise.resolve('tla'));\n",
            ),
            (
                "dep.mjs",
                "export let n = 1;\nexport function bump() { n++; }\n",
            ),
        ],
    );
    for mode in ["run", "hostrun"] {
        let (stdout, stderr, ok) = run_file(mode, &dir.join("main.mjs"));
        assert_eq!(stdout, "2 2 undefined tla\n", "{mode} stderr: {stderr}");
        assert!(ok, "{mode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A script's dynamic `import()` resolves relative to the script file, not to
/// the process working directory.
#[test]
fn script_dynamic_import_resolves_next_to_script() {
    let dir = fixture_dir(
        "dynimport",
        &[
            (
                "main.js",
                "import('./dep.mjs').then(m => console.log('got', m.v), e => console.log('failed', String(e)));\nvoid 0;\n",
            ),
            ("dep.mjs", "export const v = 'dep';\n"),
        ],
    );
    for mode in ["run", "hostrun"] {
        let (stdout, _, ok) = run_file(mode, &dir.join("main.js"));
        assert_eq!(stdout, "got dep\n", "{mode}");
        assert!(ok, "{mode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `hostrun` reports an uncaught error by name and message (not as a raw
/// thrown handle), and a failing module keeps the output printed before it.
#[test]
fn hostrun_uncaught_error_is_readable() {
    let dir = fixture_dir(
        "hostthrow",
        &[
            ("a.js", "console.log('before'); null.x;\n"),
            (
                "b.js",
                "setTimeout(() => { throw new RangeError('late'); }, 0);\n",
            ),
            ("c.mjs", "console.log('m'); throw new TypeError('mod');\n"),
        ],
    );
    let (stdout, stderr, ok) = run_file("hostrun", &dir.join("a.js"));
    assert_eq!(stdout, "before\n");
    assert!(stderr.contains("uncaught TypeError: "), "stderr: {stderr}");
    assert!(!ok);
    let (_, stderr, ok) = run_file("hostrun", &dir.join("b.js"));
    assert!(
        stderr.contains("uncaught RangeError: late"),
        "stderr: {stderr}"
    );
    assert!(!ok);
    for mode in ["run", "hostrun"] {
        let (stdout, stderr, ok) = run_file(mode, &dir.join("c.mjs"));
        assert_eq!(stdout, "m\n", "{mode}");
        assert!(stderr.contains("TypeError: mod"), "{mode} stderr: {stderr}");
        assert!(!ok, "{mode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `.js` file with module syntax (a top-level `import`/`export`, or a
/// top-level `await` / `import.meta` that only parses as a module) runs as a
/// module, as Node's syntax detection does; a `.cjs` file never does.
#[test]
fn module_syntax_in_js_file_runs_as_module() {
    let dir = fixture_dir(
        "detect",
        &[
            (
                "static.js",
                "import { v } from './dep.mjs';\nconsole.log(v, this);\n",
            ),
            (
                "tla.js",
                "const m = await import('./dep.mjs');\nconsole.log(m.v, typeof import.meta.url);\n",
            ),
            ("plain.js", "console.log(typeof this, 'import'.length);\n"),
            ("c.cjs", "import { v } from './dep.mjs';\n"),
            ("dep.mjs", "export const v = 'dep';\n"),
        ],
    );
    for mode in ["run", "hostrun"] {
        let (stdout, stderr, ok) = run_file(mode, &dir.join("static.js"));
        assert_eq!(stdout, "dep undefined\n", "{mode} stderr: {stderr}");
        assert!(ok, "{mode}");
        let (stdout, stderr, ok) = run_file(mode, &dir.join("tla.js"));
        assert_eq!(stdout, "dep string\n", "{mode} stderr: {stderr}");
        assert!(ok, "{mode}");
        let (stdout, _, ok) = run_file(mode, &dir.join("plain.js"));
        assert_eq!(stdout, "object 6\n", "{mode}");
        assert!(ok, "{mode}");
        let (_, stderr, ok) = run_file(mode, &dir.join("c.cjs"));
        assert!(stderr.contains("SyntaxError"), "{mode} stderr: {stderr}");
        assert!(!ok, "{mode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A missing import fails the run with a readable error, and a `file:` URL
/// specifier (built from the percent-encoded `import.meta.url`) loads the
/// same module instance as the relative path.
#[test]
fn module_file_urls_and_missing_imports() {
    let dir = fixture_dir(
        "furl",
        &[
            (
                "main.mjs",
                "import * as a from './d e#.mjs';\nconst u = import.meta.url.replace(/main\\.mjs$/, 'd%20e%23.mjs');\nconst b = await import(u);\nconsole.log(a === b, b.v, import.meta.url.includes(' '));\n",
            ),
            ("d e#.mjs", "export const v = 1;\n"),
            (
                "miss.mjs",
                "import { y } from './nope.mjs';\nconsole.log(y);\n",
            ),
        ],
    );
    let (stdout, stderr, ok) = run_file("run", &dir.join("main.mjs"));
    assert_eq!(stdout, "true 1 false\n", "stderr: {stderr}");
    assert!(ok);
    let (stdout, stderr, ok) = run_file("run", &dir.join("miss.mjs"));
    assert_eq!(stdout, "");
    assert!(stderr.contains("nope.mjs"), "stderr: {stderr}");
    assert!(!ok);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The JIT's deoptimization exits (`KATAAN_JIT_DEOPT_STRESS` takes one every
/// few eligible op starts) leave a program's output exactly the
/// interpreter's: exits in the function's own code and one to three levels
/// deep in inlined callees, next to `try`/`catch`, throws out of inlined
/// bodies, layout changes, and loops re-entered after the interpreter took
/// over.
#[test]
fn forced_deopt_exits_match_the_interpreter() {
    let source = r#"
        class A { constructor(x) { if (x % 997 === 996) throw new RangeError("a" + x); this.x = x; } }
        class B extends A { constructor(x, y) { super(x); this.y = y; } }
        class C extends B { constructor(x, y, z) { super(x, y); this.z = z; } get sum() { return this.x + this.y + this.z; } }
        class T {
            constructor(id) { this.id = id; this.q = null; this.n = 0; }
            take() { const p = this.q; if (p) this.q = p.next; return p; }
            give(p) { p.next = null; if (!this.q) { this.q = p; return; } let r = this.q; while (r.next) r = r.next; r.next = p; }
            run(s) { const p = this.take(); if (!p) return; p.v = (p.v + this.id) % 101; this.n++; s.route(p, (this.id + 1) % s.ts.length); }
        }
        const s = { ts: [], routed: 0, route(p, to) { this.routed++; this.ts[to].give(p); } };
        for (let i = 0; i < 4; i++) s.ts.push(new T(i));
        for (let i = 0; i < 9; i++) s.ts[i % 4].give({ next: null, v: i });
        let acc = 0, caught = 0;
        for (let i = 0; i < 3000; i++) {
            for (let k = 0; k < s.ts.length; k++) s.ts[k].run(s);
            try { const c = new C(i, i + 1, i + 2); acc = (acc + c.sum) | 0; } catch (e) { caught += e.message.length; }
            const o = { a: i, b: 2 };
            if (i % 50 === 0) delete o.a;
            acc = (acc + (o.a | 0) + o.b) | 0;
        }
        let n = 0;
        for (const t of s.ts) n += t.n;
        console.log(acc, caught, s.routed, n);
    "#;
    let run = |envs: &[(&str, &str)]| {
        let out = Command::new(env!("CARGO_BIN_EXE_kataan"))
            .args(["run", "-e", source])
            .envs(envs.iter().copied())
            .output()
            .expect("spawn kataan");
        (
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let (want, err) = run(&[("KATAAN_JIT", "0")]);
    assert!(!want.is_empty(), "stderr: {err}");
    for envs in [
        &[("KATAAN_JIT_DEOPT_STRESS", "1")][..],
        &[("KATAAN_JIT", "eager"), ("KATAAN_JIT_DEOPT_STRESS", "1")][..],
    ] {
        let (got, err) = run(envs);
        assert_eq!(got, want, "{envs:?}: stderr: {err}");
    }
}
