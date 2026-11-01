//! Top-level-await (TLA) module evaluation: a module body with a top-level
//! `await` is driven as a suspendable async coroutine, so an `await` yields to
//! the microtask queue (correct tick interleaving) instead of draining the event
//! loop inline, and a TLA *dependency* is fully evaluated (its `await`ed exports
//! settled) before its importer's body runs.

#![cfg(all(feature = "module", feature = "std"))]

use kataan::nbexec::module::ModuleHost;
use kataan::nbvm::execute_module_typed;
use std::collections::HashMap;

/// An in-memory module host: specifiers resolve to their own string key, sources
/// are looked up in a fixed map.
struct MemHost(HashMap<String, String>);

impl ModuleHost for MemHost {
    fn resolve(&self, specifier: &str, _referrer: Option<&str>) -> Result<String, String> {
        Ok(specifier.to_string())
    }
    fn load(&self, key: &str) -> Result<String, String> {
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| format!("no such module {key}"))
    }
}

fn run(entry: &str, modules: &[(&str, &str)]) -> (String, String) {
    let map: HashMap<String, String> = modules
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    let host = MemHost(map);
    execute_module_typed(entry, &host, kataan::Limits::default())
        .unwrap_or_else(|e| panic!("module eval failed: {}: {}", e.name, e.message))
}

/// A single-module top-level `await` suspends the module body at each `await`, so
/// a queued microtask chain interleaves with the resumptions (the spec tick
/// order), rather than the whole body running before any microtask.
#[test]
fn tla_single_module_tick_interleaving() {
    let src = r#"
        globalThis.__log = [];
        Promise.resolve()
          .then(() => __log.push('tick 1'))
          .then(() => __log.push('tick 2'));
        await 0;
        __log.push('await 1');
        await 0;
        __log.push('await 2');
        console.log(__log.join(','));
    "#;
    let (out, _) = run("entry", &[("entry", src)]);
    // Interleaved: each `await` yields to a pending microtask before resuming.
    assert_eq!(out.trim(), "tick 1,await 1,tick 2,await 2");
}

/// A module that does NOT use top-level await keeps the ordinary synchronous
/// evaluation path (a regression guard for the dispatch check).
#[test]
fn non_tla_module_runs_synchronously() {
    let src = r#"
        var a = 1, b = 2;
        console.log(a + b);
    "#;
    let (out, _) = run("entry", &[("entry", src)]);
    assert_eq!(out.trim(), "3");
}

/// A top-level-await *dependency* is fully evaluated — its `await`ed export slot
/// settled to the resolved value, not a pending promise — before the importer's
/// body observes it (the DFS synchronous-completion contract).
#[test]
fn tla_dependency_evaluated_before_importer() {
    let dep = r#"
        export const value = await Promise.resolve(42);
    "#;
    let entry = r#"
        import { value } from 'dep';
        console.log(value);
    "#;
    let (out, _) = run("entry", &[("entry", entry), ("dep", dep)]);
    assert_eq!(out.trim(), "42");
}

/// A rejection thrown after a top-level `await` propagates as the module's
/// evaluation error (the evaluation promise rejects and surfaces).
#[test]
fn tla_rejection_propagates() {
    let src = r#"
        await 0;
        throw new TypeError('boom');
    "#;
    let map: HashMap<String, String> = [("entry".to_string(), src.to_string())]
        .into_iter()
        .collect();
    let host = MemHost(map);
    let err = execute_module_typed("entry", &host, kataan::Limits::default())
        .expect_err("a post-await throw must reject the module evaluation");
    assert_eq!(err.name, "TypeError");
    assert!(
        err.message.contains("boom"),
        "message was {:?}",
        err.message
    );
}

/// Runs the module `main` (which imports `./lib`) and returns what it logged.
fn run_with_lib(main: &str, lib: &str) -> String {
    let (out, _) = run("main", &[("main", main), ("./lib", lib)]);
    out.trim().to_string()
}

/// Module code with a direct `eval` resolves names through its environments: an
/// import binding there is immutable (a write was a ReferenceError, or landed on
/// a global of the same name), `typeof` sees it (it said "undefined") and a local
/// shadows it (a read saw the import through the local).
#[test]
fn import_bindings_on_the_eval_environment_paths() {
    let lib = "export let x = 1; export default 5; export function setX(v) { x = v; }";
    let main = r#"
        import d, { x, setX, x as Map } from "./lib";
        import * as ns from "./lib";
        const r = [];
        const T = (f) => { try { r.push(f()); } catch (e) { r.push(e.name); } };
        T(() => { x = 2; });
        T(() => eval("x = 2"));
        T(() => eval("x++"));
        T(() => { [x] = [3]; });
        T(() => eval("d = 1"));
        T(() => eval("typeof x + typeof d + typeof ns"));
        T(() => { let x = "L"; return eval("x"); });
        T(() => { let x = "L"; eval("x = 'M'"); return x; });
        T(() => ((x) => eval("typeof x"))("P"));
        T(() => { try { throw "C"; } catch (x) { return eval("x"); } });
        T(() => { let setX = () => "own"; return eval("setX()"); });
        T(() => [typeof Map, eval("typeof Map"), eval("Map")].join());
        T(() => eval("Map = 1"));
        T(() => { setX(7); return eval("x"); });
        console.log(r.join("|"));
    "#;
    assert_eq!(
        run_with_lib(main, lib),
        "TypeError|TypeError|TypeError|TypeError|TypeError|numbernumberobject|L|M|string|C|own|\
         number,number,1|TypeError|7"
    );
}

/// A module namespace on a prototype chain fails `[[Set]]` (the write created an
/// own property), and a namespace's integrity operations read its bindings, so a
/// binding in its temporal dead zone throws.
#[test]
fn namespace_objects_on_the_chain_and_in_the_dead_zone() {
    let lib = r#"
        import * as ns from "./lib";
        export const a = 1;
        const r = [];
        const T = (f) => { try { r.push(f()); } catch (e) { r.push(e.name); } };
        const c = Object.create(ns);
        T(() => { c.a = 2; });
        T(() => { c.fresh = 2; });
        T(() => Reflect.set(c, "fresh", 2));
        T(() => Reflect.set(Object.create(c), "a", 2));
        T(() => new Function("c", "c.fresh = 3; return Object.keys(c).length")(c));
        T(() => Object.defineProperty(c, "own", { value: 9 }).own);
        T(() => { c.own2 = 0; });
        T(() => Object.isFrozen(ns));
        T(() => Object.isSealed(ns));
        export const b = 1;
        console.log(r.join("|"));
        const s = Object.keys(ns).join();
        const q = [];
        const U = (f) => { try { q.push(f()); } catch (e) { q.push(e.name); } };
        globalThis.__late = () => {
            U(() => Object.isFrozen(ns));
            U(() => Object.isSealed(ns));
            U(() => Object.freeze(ns) === ns);
            U(() => Object.seal(ns) === ns);
            U(() => Object.keys(Object.getOwnPropertyDescriptors(ns)).join());
            console.log(s, q.join("|"));
        };
    "#;
    let main = r#"
        import "./lib";
        __late();
    "#;
    // `a` is initialized and `b` is not at the first probes: a frozen test answers
    // false at `a`, a sealed test reads on to `b`.
    assert_eq!(
        run_with_lib(main, lib),
        "TypeError|TypeError|false|false|0|9|TypeError|false|ReferenceError\n\
         a,b false|true|TypeError|true|a,b"
    );
}

/// An async module body's completion does not wait on its last expression: when
/// that value was a promise, the importers' bodies ran only after it settled (and
/// after every tick of its chain) rather than one tick after the body ended.
#[test]
fn importer_runs_one_tick_after_a_tla_body_ends() {
    let lib = r#"
        globalThis.log = [];
        await null;
        log.push("end");
        Promise.resolve().then(() => log.push("t1")).then(() => log.push("t2")).then(() => log.push("t3"));
    "#;
    let main = r#"
        import "./lib";
        log.push("main");
        await new Promise((r) => setTimeout(r, 0));
        console.log(log.join());
    "#;
    assert_eq!(run_with_lib(main, lib), "end,t1,main,t2,t3");
}
