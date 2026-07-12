//! Top-level-await (TLA) module evaluation: a module body with a top-level
//! `await` is driven as a suspendable async coroutine, so an `await` yields to
//! the microtask queue (correct tick interleaving) instead of draining the event
//! loop inline, and a TLA *dependency* is fully evaluated (its `await`ed exports
//! settled) before its importer's body runs.

#![cfg(all(feature = "module", feature = "std"))]

use kataan::nbexec::module::{ModuleHost, eval_module_typed};
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
    eval_module_typed(entry, &host, kataan::Limits::default())
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
    let err = eval_module_typed("entry", &host, kataan::Limits::default())
        .expect_err("a post-await throw must reject the module evaluation");
    assert_eq!(err.name, "TypeError");
    assert!(
        err.message.contains("boom"),
        "message was {:?}",
        err.message
    );
}
