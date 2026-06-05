//! End-to-end validation of the WebAssembly binary emitter: compile JS numeric
//! functions to a `.wasm` module, then instantiate and run it on a *real*
//! WebAssembly engine (Node's) and check the results.
//!
//! This proves the emitted bytes are not just well-framed but genuinely valid,
//! executable WebAssembly. It is skipped (passing) when `node` is unavailable,
//! so it never blocks environments without it.

use kataan::parser::Parser;
use std::process::Command;

/// Whether a `node` with `WebAssembly` is on PATH.
fn node_available() -> bool {
    Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn emitted_wasm_runs_on_a_real_engine() {
    if !node_available() {
        eprintln!("wasm_exec: node unavailable — skipping the runtime validation");
        return;
    }

    let src = "
        function add(a, b) { return a + b; }
        function max(a, b) { return a > b ? a : b; }
        function fib(n) {
          let a = 0; let b = 1; let i = 0;
          while (i < n) { let t = a + b; a = b; b = t; i = i + 1; }
          return a;
        }
        function sq(x) { return x * x; }
        function hyp2(a, b) { return sq(a) + sq(b); }
        function hypot(a, b) { return Math.sqrt(Math.max(sq(a) + sq(b), 0)); }
        function absdiff(a, b) { return Math.abs(a - b); }
        function rem(a, b) { return a % b; }
        function isEven(n) { return n % 2 === 0 ? 1 : 0; }
        function sumTo(n) { let s = 0; for (let i = 0; i <= n; i++) { s = s + i; } return s; }
        function factorial(n) { let f = 1; for (let i = 2; i <= n; i++) { f = f * i; } return f; }
        function poly(n) { let s = 0; let p = 1; for (let i = 1; i <= n; i++) { s += i; p *= 2; } return s + p; }
        function countdown(n) { let s = 0; do { s += n; n -= 1; } while (n > 0); return s; }
        function skip3(n) { let s = 0; let i = 0; while (i < n) { i += 1; if (i === 3) { continue; } if (i === 7) { break; } s += i; } return s; }
        function forskip(n) { let s = 0; for (let i = 0; i < n; i++) { if (i === 2) { continue; } if (i === 6) { break; } s += i; } return s; }
        function dwskip(n) { let s = 0; let i = 0; do { i += 1; if (i === 2) { continue; } s += i; } while (i < n); return s; }
        function bits(a, b) { return (a & b) * 1000000 + (a | b) * 1000 + (a ^ b); }
        function shifts(x) { return (x << 3) * 1000 + (x >> 1); }
        function bnot(x) { return ~x; }
        function cube(x) { return x ** 3; }
        function quad(x) { return x ** 4 + x ** 0; }
        function maxOf3(a, b, c) { return Math.max(a, b, c); }
        function minOf4(a, b, c, d) { return Math.min(a, b, c, d); }
        function inRange(x, lo, hi) { if (x >= lo && x <= hi) { return 1; } return 0; }
        function either(a, b) { if (a > 0 || b > 0) { return 1; } return 0; }
        function notZero(x) { return !(x === 0); }
        function outside(a, b) { if (!(a > 0) && !(b > 0)) { return 1; } return 0; }
    ";
    let program = Parser::parse_program(src).expect("parse");
    let wasm = kataan::wasm::compile_module_binary(&program).expect("compile to wasm");

    let dir = std::env::temp_dir();
    let wasm_path = dir.join("kataan_wasm_exec_test.wasm");
    std::fs::write(&wasm_path, &wasm).expect("write wasm");

    // A Node driver that instantiates the module and prints the results we check.
    let driver = format!(
        r#"
        const fs = require('fs');
        const buf = fs.readFileSync({path:?});
        WebAssembly.instantiate(buf).then(({{ instance }}) => {{
          const e = instance.exports;
          const out = [
            e.add(2, 3),
            e.max(4, 9),
            e.max(9, 4),
            e.fib(10),
            e.fib(20),
            e.hyp2(3, 4),
            e.hypot(3, 4),
            e.absdiff(2, 9),
            e.rem(17, 5),
            e.isEven(10),
            e.isEven(7),
            e.sumTo(10),
            e.factorial(5),
            e.poly(4),
            e.countdown(4),
            e.skip3(10),
            e.forskip(10),
            e.dwskip(4),
            e.bits(12, 10),
            e.shifts(5),
            e.bnot(5),
            e.cube(3),
            e.quad(2),
            e.maxOf3(4, 9, 2),
            e.minOf4(8, 3, 5, 1),
            e.inRange(5, 1, 10),
            e.either(0, -1),
            e.notZero(5),
            e.outside(-1, -2),
          ];
          console.log(out.join(','));
        }}).catch(err => {{ console.error('INVALID:' + err.message); process.exit(1); }});
        "#,
        path = wasm_path.to_string_lossy()
    );
    let driver_path = dir.join("kataan_wasm_exec_driver.js");
    std::fs::write(&driver_path, driver).expect("write driver");

    let output = Command::new("node")
        .arg(&driver_path)
        .output()
        .expect("run node");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "node failed to run the emitted wasm: {stderr}"
    );
    // add=5, max=9/9, fib(10)=55, fib(20)=6765, hyp2(3,4)=25, hypot(3,4)=5,
    // absdiff(2,9)=7 — including the native Math.sqrt/max/abs ops.
    assert_eq!(
        stdout.trim(),
        "5,9,9,55,6765,25,5,7,2,1,0,55,120,26,10,18,13,8,8014006,40002,-6,27,17,9,1,1,0,1,1", // …, notZero(5)=1, outside(-1,-2)=1 (logical !)
        "wasm produced wrong results"
    );

    let _ = std::fs::remove_file(&wasm_path);
    let _ = std::fs::remove_file(&driver_path);
}
