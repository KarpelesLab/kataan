//! End-to-end tests for the §4.5 Node-compat builtins installed by
//! [`kataan::host::node::install`] — `Buffer`, `path`, `os`, `util`,
//! `querystring`, `process`, and the `require('node:...')` shim.
//!
//! Each test parses a small JS snippet, runs it through a fresh `Interp` with the
//! Node builtins installed, and asserts on the completion value's display string.

use kataan::nbexec::Interp;
use kataan::parser::Parser;

/// Run `src` with the Node builtins installed; return the completion value's
/// display string.
fn run(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse");
    let mut interp = Interp::new();
    kataan::host::node::install(&mut interp);
    let v = interp.run(&program).expect("run");
    interp.display(v)
}

/// Run `setup` (whose completion is ignored), then — after the event loop has
/// drained all microtasks — evaluate `read` on the same interpreter and return
/// its display string. Used to observe a value settled by an async `.then`
/// reaction, which cannot be captured by the first run's synchronous completion.
fn run_settled(setup: &str, read: &str) -> String {
    let p1 = Parser::parse_program(setup).expect("parse setup");
    let p2 = Parser::parse_program(read).expect("parse read");
    let mut interp = Interp::new();
    kataan::host::node::install(&mut interp);
    interp.run(&p1).expect("run setup");
    let v = interp.run(&p2).expect("run read");
    interp.display(v)
}

// --------------------------------------------------------------------------
// path
// --------------------------------------------------------------------------

#[test]
fn path_join_and_normalize() {
    assert_eq!(
        run("path.join('/foo', 'bar', 'baz/asdf', 'quux', '..')"),
        "/foo/bar/baz/asdf"
    );
    assert_eq!(run("path.join('a', '', 'b')"), "a/b");
    assert_eq!(
        run("path.normalize('/foo/bar//baz/asdf/quux/..')"),
        "/foo/bar/baz/asdf"
    );
    assert_eq!(run("path.normalize('foo/../../bar')"), "../bar");
}

#[test]
fn path_resolve_is_absolute() {
    assert_eq!(run("path.resolve('/a/b', '/c', 'd')"), "/c/d");
    assert_eq!(run("path.resolve('/foo/bar', './baz')"), "/foo/bar/baz");
    assert_eq!(run("path.isAbsolute('/x')"), "true");
    assert_eq!(run("path.isAbsolute('x/y')"), "false");
}

#[test]
fn path_dirname_basename_extname() {
    assert_eq!(run("path.dirname('/foo/bar/baz.txt')"), "/foo/bar");
    assert_eq!(run("path.basename('/foo/bar/baz.txt')"), "baz.txt");
    assert_eq!(run("path.basename('/foo/bar/baz.txt', '.txt')"), "baz");
    assert_eq!(run("path.extname('index.html')"), ".html");
    assert_eq!(run("path.extname('noext')"), "");
    assert_eq!(run("path.dirname('/')"), "/");
}

#[test]
fn path_parse_and_format() {
    assert_eq!(
        run(
            "var p = path.parse('/home/user/file.txt'); [p.root, p.dir, p.base, p.ext, p.name].join('|')"
        ),
        "/|/home/user|file.txt|.txt|file"
    );
    assert_eq!(
        run("path.format({ dir: '/a/b', base: 'c.txt' })"),
        "/a/b/c.txt"
    );
    assert_eq!(
        run("path.format({ dir: '/a', name: 'c', ext: '.js' })"),
        "/a/c.js"
    );
    assert_eq!(run("path.relative('/a/b/c', '/a/b/d/e')"), "../d/e");
    assert_eq!(run("path.sep + path.delimiter"), "/:");
}

// --------------------------------------------------------------------------
// Buffer
// --------------------------------------------------------------------------

#[test]
fn buffer_from_string_roundtrip_encodings() {
    assert_eq!(run("Buffer.from('hello').toString()"), "hello");
    assert_eq!(run("Buffer.from('hello').toString('hex')"), "68656c6c6f");
    assert_eq!(run("Buffer.from('68656c6c6f', 'hex').toString()"), "hello");
    assert_eq!(run("Buffer.from('hello').toString('base64')"), "aGVsbG8=");
    assert_eq!(run("Buffer.from('aGVsbG8=', 'base64').toString()"), "hello");
    assert_eq!(
        run("Buffer.from('ABC', 'latin1').toString('latin1')"),
        "ABC"
    );
    assert_eq!(run("Buffer.from('hi').toString('ascii')"), "hi");
}

#[test]
fn buffer_is_a_uint8array_subclass() {
    assert_eq!(run("Buffer.from([1,2,3]) instanceof Uint8Array"), "true");
    assert_eq!(run("Buffer.from([1,2,3]) instanceof Buffer"), "true");
    assert_eq!(run("Buffer.from([1,2,3]).length"), "3");
    assert_eq!(run("Buffer.from([10,20,30])[1]"), "20");
    assert_eq!(run("Buffer.isBuffer(Buffer.from('x'))"), "true");
    assert_eq!(run("Buffer.isBuffer(new Uint8Array(3))"), "false");
    assert_eq!(run("Buffer.isBuffer([1,2,3])"), "false");
}

#[test]
fn buffer_alloc_and_fill() {
    assert_eq!(run("Buffer.alloc(3).length"), "3");
    assert_eq!(run("Buffer.alloc(3)[0]"), "0");
    assert_eq!(run("Buffer.alloc(4, 7)[3]"), "7");
    assert_eq!(run("Buffer.allocUnsafe(5).length"), "5");
    assert_eq!(
        run("var b = Buffer.alloc(3); b.fill(65); b.toString()"),
        "AAA"
    );
}

#[test]
fn buffer_concat_and_slice() {
    assert_eq!(
        run("Buffer.concat([Buffer.from('foo'), Buffer.from('bar')]).toString()"),
        "foobar"
    );
    assert_eq!(
        run("Buffer.concat([Buffer.from('ab'), Buffer.from('cd')], 3).toString()"),
        "abc"
    );
    assert_eq!(
        run("Buffer.from('hello world').slice(0, 5).toString()"),
        "hello"
    );
    assert_eq!(
        run("Buffer.from('hello world').slice(6).toString()"),
        "world"
    );
    // `subarray` returns a plain Uint8Array view (engine synthesizes it over the
    // Buffer override); its bytes are correct — wrap it back to read as text.
    assert_eq!(
        run("Buffer.from(Buffer.from('abcdef').subarray(-3)).toString()"),
        "def"
    );
    assert_eq!(run("Buffer.from('abcdef').subarray(-3).length"), "3");
}

#[test]
fn buffer_slice_shares_memory() {
    // Node's Buffer.slice shares the backing store — a write to the view is
    // visible in the parent.
    assert_eq!(
        run("var b = Buffer.from('hello'); var s = b.slice(0, 2); s[0] = 72; b.toString()"),
        "Hello"
    );
}

#[test]
fn buffer_write_equals_and_numeric() {
    assert_eq!(
        run("var b = Buffer.alloc(5); b.write('hi'); b.toString('ascii', 0, 2)"),
        "hi"
    );
    assert_eq!(run("Buffer.from('abc').equals(Buffer.from('abc'))"), "true");
    assert_eq!(
        run("Buffer.from('abc').equals(Buffer.from('abd'))"),
        "false"
    );
    assert_eq!(
        run("var b = Buffer.alloc(1); b.writeUInt8(200, 0); b.readUInt8(0)"),
        "200"
    );
    assert_eq!(
        run("var b = Buffer.alloc(2); b.writeUInt16LE(0x1234, 0); b.readUInt16LE(0)"),
        "4660"
    );
    assert_eq!(
        run("var b = Buffer.alloc(2); b.writeUInt16BE(0x1234, 0); b.readUInt16BE(0)"),
        "4660"
    );
    assert_eq!(
        run("var b = Buffer.alloc(4); b.writeUInt32LE(0xDEADBEEF, 0); b.readUInt32LE(0)"),
        "3735928559"
    );
}

#[test]
fn buffer_byte_length_and_copy() {
    assert_eq!(run("Buffer.byteLength('hello')"), "5");
    assert_eq!(run("Buffer.byteLength('68656c6c6f', 'hex')"), "5");
    assert_eq!(
        run("var s = Buffer.from('SRC'); var d = Buffer.alloc(3); s.copy(d); d.toString()"),
        "SRC"
    );
}

#[test]
fn buffer_from_arraybuffer_shares() {
    assert_eq!(
        run(
            "var ab = new ArrayBuffer(4); var b = Buffer.from(ab); b[0] = 65; new Uint8Array(ab)[0]"
        ),
        "65"
    );
}

// --------------------------------------------------------------------------
// os
// --------------------------------------------------------------------------

#[test]
fn os_basics() {
    // The platform string is one of Node's known values.
    let p = run("os.platform()");
    assert!(
        ["linux", "darwin", "win32", "freebsd", "openbsd", "android"].contains(&p.as_str()),
        "unexpected platform {p}"
    );
    let a = run("os.arch()");
    assert!(!a.is_empty());
    assert!(["Linux", "Darwin", "Windows_NT"].contains(&run("os.type()").as_str()));
    assert_eq!(
        run("os.endianness()"),
        if cfg!(target_endian = "big") {
            "BE"
        } else {
            "LE"
        }
    );
    assert!(run("os.cpus().length >= 1").starts_with("true"));
    assert_eq!(run("typeof os.totalmem()"), "number");
    assert_eq!(run("Array.isArray(os.cpus())"), "true");
    assert_eq!(run("typeof os.cpus()[0].model"), "string");
}

// --------------------------------------------------------------------------
// util
// --------------------------------------------------------------------------

#[test]
fn util_format() {
    assert_eq!(run("util.format('%s = %d', 'x', 42)"), "x = 42");
    assert_eq!(run("util.format('%s', 'hi', 'extra')"), "hi extra");
    assert_eq!(run("util.format('100%% done')"), "100% done");
    assert_eq!(run("util.format('%j', { a: 1 })"), "{\"a\":1}");
    assert_eq!(run("util.format('no specifiers')"), "no specifiers");
    assert_eq!(run("util.format('%i', 3.9)"), "3");
}

#[test]
fn util_inspect() {
    assert_eq!(run("util.inspect([1, 2, 3])"), "[ 1, 2, 3 ]");
    assert_eq!(run("util.inspect({ a: 1, b: 'x' })"), "{ a: 1, b: 'x' }");
    assert_eq!(run("util.inspect('hello')"), "'hello'");
    assert_eq!(
        run("util.inspect(new Map([['a', 1]]))"),
        "Map(1) { 'a' => 1 }"
    );
    assert_eq!(run("util.inspect(new Set([1, 2]))"), "Set(2) { 1, 2 }");
    // Circular references are broken, not overflowed.
    assert_eq!(
        run("var o = {}; o.self = o; util.inspect(o)"),
        "{ self: [Circular *1] }"
    );
    // Depth limiting.
    assert_eq!(
        run("util.inspect({ a: { b: { c: 1 } } }, { depth: 1 })"),
        "{ a: { b: [Object] } }"
    );
}

#[test]
fn util_types() {
    assert_eq!(run("util.types.isMap(new Map())"), "true");
    assert_eq!(run("util.types.isSet(new Set())"), "true");
    assert_eq!(run("util.types.isDate(new Date())"), "true");
    assert_eq!(run("util.types.isRegExp(/x/)"), "true");
    assert_eq!(run("util.types.isRegExp('x')"), "false");
    assert_eq!(run("util.types.isPromise(Promise.resolve(1))"), "true");
    assert_eq!(run("util.types.isTypedArray(new Uint8Array(1))"), "true");
    assert_eq!(run("util.types.isTypedArray([1])"), "false");
}

#[test]
fn util_promisify() {
    // A Node-style callback function → promise. The `.then` reaction is a
    // microtask, so `out` is read after the event loop drains (second run).
    let setup = r#"
        function cbStyle(x, cb) { cb(null, x * 2); }
        var p = util.promisify(cbStyle);
        var out = 'pending';
        p(21).then(function (v) { out = 'resolved:' + v; });
    "#;
    assert_eq!(run_settled(setup, "out"), "resolved:42");
    assert_eq!(
        run("typeof util.promisify(function () {})(1).then"),
        "function"
    );
}

#[test]
fn util_promisify_rejects() {
    let setup = r#"
        function failing(cb) { cb(new Error('boom')); }
        var p = util.promisify(failing);
        var out = 'pending';
        p().then(function () { out = 'ok'; }, function (e) { out = 'err:' + e.message; });
    "#;
    assert_eq!(run_settled(setup, "out"), "err:boom");
}

#[test]
fn util_inherits() {
    let src = r#"
        function Base() {}
        Base.prototype.greet = function () { return 'hi'; };
        function Derived() {}
        util.inherits(Derived, Base);
        var d = new Derived();
        (d instanceof Base) + ',' + d.greet() + ',' + (Derived.super_ === Base);
    "#;
    assert_eq!(run(src), "true,hi,true");
}

// --------------------------------------------------------------------------
// querystring
// --------------------------------------------------------------------------

#[test]
fn querystring_parse_stringify() {
    assert_eq!(
        run("querystring.stringify({ a: 1, b: 'two' })"),
        "a=1&b=two"
    );
    assert_eq!(
        run("var q = querystring.parse('a=1&b=two'); q.a + ',' + q.b"),
        "1,two"
    );
    // Repeated keys accumulate into an array.
    assert_eq!(
        run("var q = querystring.parse('x=1&x=2'); Array.isArray(q.x) + ':' + q.x.join('-')"),
        "true:1-2"
    );
    // Array values expand.
    assert_eq!(run("querystring.stringify({ x: ['a', 'b'] })"), "x=a&x=b");
    // '+' decodes to space.
    assert_eq!(run("querystring.parse('q=hello+world').q"), "hello world");
}

#[test]
fn querystring_escape_unescape() {
    assert_eq!(run("querystring.escape('a b&c')"), "a%20b%26c");
    assert_eq!(run("querystring.unescape('a%20b%26c')"), "a b&c");
}

// --------------------------------------------------------------------------
// process + require shim
// --------------------------------------------------------------------------

#[test]
fn process_globals() {
    let p = run("process.platform");
    assert!(!p.is_empty());
    assert_eq!(run("typeof process.cwd()"), "string");
    assert_eq!(run("Array.isArray(process.argv)"), "true");
    assert_eq!(run("typeof process.env"), "object");
    assert!(run("process.version").starts_with('v'));
}

#[test]
fn require_shim() {
    assert_eq!(run("require('node:path').sep"), "/");
    assert_eq!(run("require('path').basename('/a/b.txt')"), "b.txt");
    assert_eq!(run("require('os').platform() === os.platform()"), "true");
    assert_eq!(run("require('buffer').Buffer.from('x').toString()"), "x");
    assert_eq!(run("typeof require('util').format"), "function");
}
