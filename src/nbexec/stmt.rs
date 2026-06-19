use super::*;

/// A pre-evaluated assignment reference for a simple destructuring leaf: its
/// object/key (for a member), super-property name, or the identifier target
/// expression. Lets the target be resolved *before* the iterator step, then
/// assigned once the element value is known (spec evaluation order).
enum AssignRef {
    Ident(String),
    Member { obj: NanBox, key: String },
    Super { name: String },
}

impl<'a> Interp<'a> {
    pub(crate) fn run_body(&mut self, body: Body<'a>) -> Result<NanBox, ExecError> {
        match body {
            Body::Expr(e) => self.eval(e),
            Body::Block(stmts) => {
                // Strict mode is inherited from the caller and additionally enabled
                // by a `"use strict"` directive prologue; it propagates to nested
                // bodies and is restored on exit.
                let saved_strict = self.strict;
                self.strict = self.strict || has_use_strict(stmts);
                self.hoist_with(stmts, true)?;
                let mut result = Ok(NanBox::undefined());
                for stmt in stmts {
                    match self.exec(stmt) {
                        Ok(Flow::Return(v)) => {
                            result = Ok(v);
                            break;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    }
                }
                self.strict = saved_strict;
                result
            }
        }
    }

    // --- statements ---

    pub(crate) fn exec(&mut self, stmt: &'a Stmt) -> Result<Flow, ExecError> {
        // C2: share the tree-walk recursion budget with `eval` so deeply nested
        // statements (or expressions reached through them) throw a catchable
        // `RangeError` rather than overflowing the host stack. Bounded by the
        // dedicated `max_eval_depth` knob (separate from `max_call_depth`).
        if self.eval_depth >= self.realm.limits.max_eval_depth {
            let msg = self.new_str("Maximum call stack size exceeded");
            let err = self.make_error(N_ERROR_BASE + 2, Some(msg));
            return Err(ExecError::Throw(err));
        }
        self.eval_depth += 1;
        let r = self.exec_inner(stmt);
        self.eval_depth -= 1;
        r
    }

    pub(crate) fn exec_inner(&mut self, stmt: &'a Stmt) -> Result<Flow, ExecError> {
        match stmt {
            Stmt::Empty { .. } => Ok(Flow::Normal(NanBox::undefined())),
            Stmt::Expr { expression, .. } => Ok(Flow::Normal(self.eval(expression)?)),
            Stmt::Var(decl) => {
                self.exec_var(decl)?;
                Ok(Flow::Normal(NanBox::undefined()))
            }
            // Function declarations are bound by hoisting. For a *block-level*
            // declaration, Annex B.3.3 additionally updates the function-scope
            // `var` binding of the same name when the declaration is evaluated
            // (so the most recently *executed* block function wins, matching
            // web-compat semantics). Only applies when the name was var-hoisted
            // (i.e. the binding exists in the variable environment) and we are
            // inside a nested block (not at the variable-environment top level).
            Stmt::Function(func) => {
                if let Some(id) = &func.id
                    && !self.current.ptr_eq(&self.var_scope)
                    && self
                        .annexb_block_fns
                        .iter()
                        .any(|n| n.as_str() == &*id.name)
                    && let Some(value) = self.current.get(&id.name)
                {
                    self.annexb_update_var(&id.name, value);
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::Class(class) => {
                let value = self.make_class(class)?;
                if let Some(id) = &class.id {
                    self.current.declare(&id.name, value);
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::Block { body, .. } => self.exec_block(body),
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval_truthy(test)? {
                    self.exec_if_branch(consequent)
                } else if let Some(alt) = alternate {
                    self.exec_if_branch(alt)
                } else {
                    Ok(Flow::Normal(NanBox::undefined()))
                }
            }
            Stmt::While { test, body, .. } => {
                let label = self.pending_label.take();
                while self.eval_truthy(test)? {
                    match loop_action(self.exec(body)?, &label) {
                        LoopAction::Next => {}
                        LoopAction::Stop => break,
                        LoopAction::Propagate(f) => return Ok(f),
                    }
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::DoWhile { body, test, .. } => {
                let label = self.pending_label.take();
                loop {
                    match loop_action(self.exec(body)?, &label) {
                        LoopAction::Next => {}
                        LoopAction::Stop => break,
                        LoopAction::Propagate(f) => return Ok(f),
                    }
                    if !self.eval_truthy(test)? {
                        break;
                    }
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::Labeled { label, body, .. } => {
                // The label is handed to a *directly* labeled loop (via `pending_label`)
                // so its own `break`/`continue label` see it. For any other body (a
                // block, `if`, …) leaving `pending_label` set would let a nested loop
                // wrongly claim the label, so it stays unset and the break-match below
                // unwinds `break label` out of the whole labeled statement.
                let is_loop = matches!(
                    &**body,
                    Stmt::For { .. }
                        | Stmt::While { .. }
                        | Stmt::DoWhile { .. }
                        | Stmt::ForOf { .. }
                        | Stmt::ForIn { .. }
                );
                if is_loop {
                    self.pending_label = Some(String::from(&*label.name));
                }
                let flow = self.exec(body)?;
                self.pending_label = None;
                // A labeled statement consumes a matching `break label`.
                Ok(match flow {
                    Flow::Break(Some(l)) if l == *label.name => Flow::Normal(NanBox::undefined()),
                    other => other,
                })
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => self.exec_for(init.as_ref(), test.as_deref(), update.as_deref(), body),
            Stmt::Return { argument, .. } => {
                let v = match argument {
                    Some(e) => self.eval(e)?,
                    None => NanBox::undefined(),
                };
                Ok(Flow::Return(v))
            }
            Stmt::Break { label, .. } => {
                Ok(Flow::Break(label.as_ref().map(|l| String::from(&*l.name))))
            }
            Stmt::Continue { label, .. } => Ok(Flow::Continue(
                label.as_ref().map(|l| String::from(&*l.name)),
            )),
            Stmt::Throw { argument, .. } => {
                let v = self.eval(argument)?;
                Err(ExecError::Throw(v))
            }
            Stmt::With { object, body, .. } => {
                let obj = self.eval(object)?;
                // ToObject: `null`/`undefined` throws; a primitive is boxed to its
                // wrapper object.
                if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Cannot convert undefined or null to object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let obj = if obj.as_handle().is_some() {
                    obj
                } else {
                    self.coerce_to_object(obj)
                };
                self.with_stack.push(obj);
                let r = self.exec(body);
                self.with_stack.pop();
                r
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => self.exec_try(block, handler.as_ref(), finalizer.as_deref()),
            Stmt::ForOf {
                left,
                right,
                body,
                is_await,
                ..
            } => {
                let iterable = self.eval(right)?;
                // A user iterator (not a built-in array/string/Map/Set or generator)
                // runs lazily: one `next()` per iteration, with `IteratorClose` on an
                // early exit — so `break` calls `return()` and an infinite iterator can
                // be cut short. `for await` keeps the eager path (it awaits each value).
                if !*is_await && let Some(ih) = self.for_of_get_iterator(iterable)? {
                    // A user `[Symbol.iterator]` that is a generator is still drained
                    // eagerly (its `next` is built-in dispatch, not a readable method) —
                    // from this same iterator, so it is not re-created.
                    if self.realm.get_property(ih, GEN_BUF).is_some() {
                        let values = self.iterate_values(NanBox::handle(ih.to_raw()))?;
                        return self.exec_for_each(left, body, values);
                    }
                    return self.exec_for_of_iter(left, body, ih);
                }
                // `for await (…)`: drive the async-iterator protocol (await each
                // `next()` result for an async iterable, else await each yielded
                // value). A plain `for…of` materializes the values directly.
                let values = if *is_await {
                    self.for_await_values(iterable)?
                } else {
                    self.iterate_values(iterable)?
                };
                self.exec_for_each(left, body, values)
            }
            Stmt::ForIn {
                left, right, body, ..
            } => {
                let obj = self.eval(right)?;
                // A proxy with an `ownKeys` trap enumerates through it; otherwise
                // the normal own + inherited enumerable key walk.
                let keys = if let Some(raw) = obj.as_handle()
                    && let Some(trap_keys) =
                        self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    trap_keys.iter().map(|k| self.new_str(k)).collect()
                } else {
                    self.iterate_keys(obj)
                };
                self.exec_for_each(left, body, keys)
            }
            Stmt::Switch {
                discriminant,
                cases,
                ..
            } => self.exec_switch(discriminant, cases),
            _ => Err(ExecError::Unsupported("statement")),
        }
    }

    /// `try { … } catch (e) { … } finally { … }`. A `catch` handles a thrown
    /// value; the `finally` block runs on every exit (normal, thrown, or
    /// `return`/`break`/`continue`), and its own abrupt completion takes over.
    pub(crate) fn exec_try(
        &mut self,
        block: &'a [Stmt],
        handler: Option<&'a crate::ast::CatchClause>,
        finalizer: Option<&'a [Stmt]>,
    ) -> Result<Flow, ExecError> {
        let mut outcome = self.exec_scoped(block);
        // A thrown value is routed to the catch clause, if any.
        if let (Err(ExecError::Throw(value)), Some(catch)) = (&outcome, handler) {
            let thrown = *value;
            let child = self.current.child();
            let saved = core::mem::replace(&mut self.current, child);
            // The catch binding may be a name, a destructuring pattern, or absent
            // (optional catch binding).
            let bound = match &catch.param {
                Some(target) => self.bind_pattern(target, thrown),
                None => Ok(()),
            };
            outcome = bound.and_then(|()| self.exec_seq(&catch.body));
            self.current = saved;
        }
        // `finally` runs regardless; an abrupt finally overrides the outcome.
        if let Some(fin) = finalizer {
            match self.exec_scoped(fin) {
                Ok(Flow::Normal(_)) => outcome,
                other => other, // finally returned/broke/threw → that wins
            }
        } else {
            outcome
        }
    }

    /// Runs a statement list in a fresh child scope.
    pub(crate) fn exec_scoped(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        let result = self.exec_seq(body);
        self.current = saved;
        result
    }

    pub(crate) fn exec_block(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        self.exec_scoped(body)
    }

    /// Writes a block-level function value to its `var`-hoisted binding in the
    /// current variable environment (Annex B.3.3 / B.3.4 runtime update). When the
    /// variable environment is the global scope, the global object property is
    /// updated too (where global `var`/function bindings live).
    fn annexb_update_var(&mut self, name: &str, value: NanBox) {
        self.var_scope.declare(name, value);
        if self.var_scope.ptr_eq(&self.global_scope)
            && let Some(g) = self.global_this.as_handle().map(Handle::from_raw)
        {
            self.realm.set_property(g, name, value);
        }
    }

    /// Executes an `if` branch. Annex B.3.4 allows a bare `FunctionDeclaration`
    /// as the body of an `if`/`else` in sloppy mode (`if (x) function f(){}`).
    /// Such a declaration is treated as if wrapped in a block: a fresh closure is
    /// created in a new scope and, when the name was `var`-hoisted to the
    /// variable environment, that outer binding is updated to the new value.
    fn exec_if_branch(&mut self, stmt: &'a Stmt) -> Result<Flow, ExecError> {
        if let Stmt::Function(func) = stmt
            && let Some(id) = &func.id
        {
            let value = self.make_function(
                &func.params,
                Body::Block(&func.body),
                func.is_async,
                func.is_generator,
            );
            self.set_fn_name(value, &id.name);
            if self
                .annexb_block_fns
                .iter()
                .any(|n| n.as_str() == &*id.name)
            {
                self.annexb_update_var(&id.name, value);
            }
            return Ok(Flow::Normal(NanBox::undefined()));
        }
        self.exec(stmt)
    }

    /// Executes a statement sequence in the current scope (with hoisting).
    pub(crate) fn exec_seq(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        self.hoist(body)?;
        let mut last = Flow::Normal(NanBox::undefined());
        for stmt in body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = Flow::Normal(v),
                other => return Ok(other),
            }
        }
        Ok(last)
    }

    pub(crate) fn exec_var(&mut self, decl: &'a VarDecl) -> Result<(), ExecError> {
        for d in &decl.declarations {
            self.exec_single_declarator(decl.kind, d)?;
        }
        Ok(())
    }

    /// Binds one `var`/`let`/`const` declarator (evaluating its initializer).
    /// Factored out of [`Interp::exec_var`] so the lazy-generator machine can
    /// bind a yield-free declarator in one shot.
    pub(crate) fn exec_single_declarator(
        &mut self,
        kind: crate::ast::VarDeclKind,
        d: &'a crate::ast::VarDeclarator,
    ) -> Result<(), ExecError> {
        let is_var = matches!(kind, crate::ast::VarDeclKind::Var);
        // A bare `var x;` (no initializer) must not clobber the value a
        // hoisted binding may already hold from an earlier assignment.
        if is_var && d.init.is_none() {
            return Ok(());
        }
        // For `const C = class {}`, hand the binding name to `make_class` so the
        // anonymous class's `name` is set before its static initializers run.
        if let (Some(Expr::Class(c)), BindingTarget::Ident(Ident { name, .. })) =
            (&d.init, &d.target)
            && c.id.is_none()
        {
            self.pending_class_name = Some(&**name);
        }
        let value = match &d.init {
            Some(e) => self.eval(e)?,
            None => NanBox::undefined(),
        };
        self.pending_class_name = None;
        // An anonymous function/class assigned to a name takes that name
        // (`const f = function(){}` → `f.name === "f"`).
        if let (Some(init), BindingTarget::Ident(Ident { name, .. })) = (&d.init, &d.target)
            && matches!(init, Expr::Function(_) | Expr::Class(_) | Expr::Arrow(_))
        {
            self.set_fn_name(value, name);
        }
        // `var` assigns to its hoisted binding (in the function/program
        // scope), so a declaration inside a block updates the same variable.
        if is_var && let BindingTarget::Ident(Ident { name, .. }) = &d.target {
            let name: &str = name;
            // Inside `with (obj)`, a `var x = v` *initializer* is an ordinary
            // assignment whose target resolves through the scope chain, so when
            // the with-object provides `x` it writes that property (via `[[Set]]`),
            // not the hoisted binding (`with ({foo:…}) { var foo = v }` sets `obj.foo`).
            if self.with_binding(name).is_some() {
                self.assign_to_name(name, value)?;
                return Ok(());
            }
            if !self.current.set(name, value) {
                self.current.declare(name, value);
            }
            // A global-scope `var x = v` also publishes `x` on the global
            // object (so `this.x` / `globalThis.x` see it).
            self.publish_global_var(name, value);
            return Ok(());
        }
        // A simple `const x = …` binding is tracked so reassignment throws.
        if matches!(kind, crate::ast::VarDeclKind::Const)
            && let BindingTarget::Ident(Ident { name, .. }) = &d.target
        {
            self.current.declare_const(name, value);
            return Ok(());
        }
        self.bind_pattern(&d.target, value)?;
        Ok(())
    }

    /// Applies named-evaluation to a destructuring default: when a binding's
    /// default is an *anonymous* function/arrow/class and the target is a plain
    /// identifier, the produced function takes the binding name
    /// (`let [x = () => {}] = []` → `x.name === "x"`).
    pub(crate) fn infer_binding_name(
        &mut self,
        target: &'a BindingTarget,
        default: &'a Expr,
        value: NanBox,
    ) {
        if let BindingTarget::Ident(Ident { name, .. }) = target
            && matches!(default, Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_))
        {
            self.set_fn_name(value, name);
        }
    }

    /// Binds a (possibly destructuring) target to `value`, declaring the names
    /// it introduces in the current scope.
    pub(crate) fn bind_pattern(
        &mut self,
        target: &'a BindingTarget,
        value: NanBox,
    ) -> Result<(), ExecError> {
        match target {
            BindingTarget::Ident(Ident { name, .. }) => self.current.declare(name, value),
            BindingTarget::Array(pat) => {
                // Any iterable destructures (strings, Sets, generators, …); a
                // non-iterable (null, a plain object, a number) is a TypeError.
                // ArrayBindingPattern performs GetIterator: an array whose
                // `Symbol.iterator` was deleted or made non-callable must throw,
                // even though the fast path below would otherwise read its backing
                // store directly (`delete Array.prototype[Symbol.iterator]`).
                self.require_iterator_method(value)?;
                let has_rest = pat
                    .elements
                    .iter()
                    .any(|e| matches!(e, ArrayPatternElement::Rest { .. }));
                let needed = pat
                    .elements
                    .iter()
                    .filter(|e| !matches!(e, ArrayPatternElement::Rest { .. }))
                    .count();
                // Without a rest target, a *user* iterator is pulled lazily for only
                // the values the pattern needs, then closed (`IteratorClose`) — so
                // `[a, b] = infiniteIterator` terminates and `return()` runs. Arrays,
                // strings, Sets, and generators take the eager path (no user `next`).
                let elems = if !has_rest && let Some(ih) = self.for_of_get_iterator(value)? {
                    if self.realm.get_property(ih, GEN_BUF).is_some() {
                        // An (eager) generator iterator has no callable `next` property;
                        // its values are already buffered — drain the obtained iterator
                        // (don't re-invoke `Symbol.iterator`, which would re-run it).
                        self.iterate_values(NanBox::handle(ih.to_raw()))?
                    } else {
                        // A plain user iterator: pull only the values the pattern needs,
                        // then close it (so `[a, b] = infiniteIterator` terminates).
                        let iterator = NanBox::handle(ih.to_raw());
                        let mut out = Vec::with_capacity(needed);
                        let mut exhausted = false;
                        for _ in 0..needed {
                            let next_fn = self.read_member(ih, "next")?;
                            let res = self.call_with_this(next_fn, iterator, &[])?;
                            let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                                return Err(ExecError::Throw(
                                    self.new_str("iterator result is not an object"),
                                ));
                            };
                            let done = self.read_member(rh, "done")?;
                            if self.realm.truthy(done) {
                                exhausted = true;
                                break;
                            }
                            out.push(self.read_member(rh, "value")?);
                        }
                        if !exhausted {
                            self.iterator_close(ih)?;
                        }
                        out
                    }
                } else {
                    self.iterate_values(value)?
                };
                let mut i = 0;
                for el in &pat.elements {
                    match el {
                        ArrayPatternElement::Hole => i += 1,
                        ArrayPatternElement::Item {
                            target, default, ..
                        } => {
                            let mut v = elems.get(i).copied().unwrap_or(NanBox::undefined());
                            if matches!(v.unpack(), Unpacked::Undefined)
                                && let Some(d) = default
                            {
                                v = self.eval(d)?;
                                // `[x = function(){}]` names the anonymous function
                                // after the binding target (`x`).
                                self.infer_binding_name(target, d, v);
                            }
                            self.bind_pattern(target, v)?;
                            i += 1;
                        }
                        ArrayPatternElement::Rest { target, .. } => {
                            let rest = elems[i.min(elems.len())..].to_vec();
                            let h = self.realm.new_array(rest);
                            self.bind_pattern(target, NanBox::handle(h.to_raw()))?;
                        }
                    }
                }
            }
            BindingTarget::Object(pat) => {
                // Object destructuring requires a coercible value: null/undefined throw
                // a TypeError (RequireObjectCoercible).
                if matches!(value.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Cannot destructure 'null' or 'undefined' as an object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let src = value.as_handle().map(Handle::from_raw);
                let mut used: Vec<String> = Vec::new();
                for prop in &pat.properties {
                    // A computed key (`{ [expr]: t }`) is evaluated here.
                    let key = self.eval_prop_key(&prop.key)?;
                    // Read through `read_member` so accessors fire and inherited /
                    // string-length / array-length properties resolve (not just own
                    // data slots).
                    let mut v = match src {
                        Some(h) => self.read_member(h, &key)?,
                        None => NanBox::undefined(),
                    };
                    if matches!(v.unpack(), Unpacked::Undefined)
                        && let Some(d) = &prop.default
                    {
                        v = self.eval(d)?;
                        self.infer_binding_name(&prop.value, d, v);
                    }
                    used.push(key);
                    self.bind_pattern(&prop.value, v)?;
                }
                if let Some(rest) = &pat.rest {
                    let obj = self.realm.new_object();
                    if let Some(h) = src {
                        for k in self.realm.object_keys(h).unwrap_or_default() {
                            if !used.contains(&k) {
                                // Read through `read_member` so an own getter fires
                                // (exactly once); the rest object always gets a plain
                                // enumerable, writable, configurable data property.
                                let pv = self.read_member(h, &k)?;
                                self.realm.set_property(obj, &k, pv);
                            }
                        }
                    }
                    self.bind_pattern(rest, NanBox::handle(obj.to_raw()))?;
                }
            }
        }
        Ok(())
    }

    /// The values iterated by `for-of`: array elements, string chars, `Set`
    /// values, or `Map` `[key, value]` pairs.
    /// Recursively flattens nested arrays up to `depth` levels (for `flat`).
    /// `rec` is the current recursion depth; nesting past
    /// `limits.max_display_depth` throws rather than overflowing the host stack
    /// (`flat(Infinity)` on a pathologically deep array).
    pub(crate) fn flatten(
        &mut self,
        elems: &[NanBox],
        depth: i32,
        rec: usize,
    ) -> Result<Vec<NanBox>, ExecError> {
        if rec >= self.realm.limits.max_display_depth {
            let m = self.new_str("Maximum call stack size exceeded");
            return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
        }
        let mut out = Vec::new();
        for e in elems {
            if depth > 0
                && let Some(inner) = e
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
            {
                out.extend(self.flatten(&inner, depth - 1, rec + 1)?);
            } else {
                out.push(*e);
            }
        }
        Ok(out)
    }

    /// A lazy `for-of` over a user iterator: pull one value per iteration (so an
    /// infinite iterator can be cut short by `break`), and run `IteratorClose` on
    /// every early exit (`break`/`return`/`throw`) — unlike the eager path.
    pub(crate) fn exec_for_of_iter(
        &mut self,
        left: &'a crate::ast::ForLeft,
        body: &'a Stmt,
        ih: Handle,
    ) -> Result<Flow, ExecError> {
        use crate::ast::ForLeft;
        let label = self.pending_label.take();
        let iterator = NanBox::handle(ih.to_raw());
        loop {
            let next_fn = self.read_member(ih, "next")?;
            let res = self.call_with_this(next_fn, iterator, &[])?;
            let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                return Err(ExecError::Throw(
                    self.new_str("iterator result is not an object"),
                ));
            };
            let done = self.read_member(rh, "done")?;
            if self.realm.truthy(done) {
                return Ok(Flow::Normal(NanBox::undefined()));
            }
            let item = self.read_member(rh, "value")?;
            let child = self.current.child();
            let saved = core::mem::replace(&mut self.current, child);
            let r = (|| {
                match left {
                    ForLeft::Decl { target, .. } => self.bind_pattern(target, item)?,
                    // The head may be a plain reference or a destructuring pattern
                    // (`for ([a, b] of …)`, `for ({ x } of …)`).
                    ForLeft::Target(expr) => {
                        self.assign_destructure(expr, item)?;
                    }
                }
                self.exec(body)
            })();
            self.current = saved;
            match r {
                Ok(flow) => match loop_action(flow, &label) {
                    LoopAction::Next => {}
                    LoopAction::Stop => {
                        self.iterator_close(ih)?;
                        return Ok(Flow::Normal(NanBox::undefined()));
                    }
                    LoopAction::Propagate(f) => {
                        self.iterator_close(ih)?;
                        return Ok(f);
                    }
                },
                Err(e) => {
                    // An abrupt body completion still closes the iterator, but its
                    // own error is suppressed in favor of the original.
                    let _ = self.iterator_close(ih);
                    return Err(e);
                }
            }
        }
    }

    pub(crate) fn exec_for_each(
        &mut self,
        left: &'a crate::ast::ForLeft,
        body: &'a Stmt,
        items: Vec<NanBox>,
    ) -> Result<Flow, ExecError> {
        use crate::ast::ForLeft;
        let label = self.pending_label.take();
        for item in items {
            let child = self.current.child();
            let saved = core::mem::replace(&mut self.current, child);
            let r = (|| {
                match left {
                    ForLeft::Decl { target, .. } => self.bind_pattern(target, item)?,
                    // The head may be a plain reference or a destructuring pattern
                    // (`for ([a, b] of …)`, `for ({ x } of …)`).
                    ForLeft::Target(expr) => {
                        self.assign_destructure(expr, item)?;
                    }
                }
                self.exec(body)
            })();
            self.current = saved;
            match loop_action(r?, &label) {
                LoopAction::Next => {}
                LoopAction::Stop => break,
                LoopAction::Propagate(f) => return Ok(f),
            }
        }
        Ok(Flow::Normal(NanBox::undefined()))
    }

    /// Reads the current value of an assignment target (identifier or member).
    pub(crate) fn read_target(&mut self, target: &'a Expr) -> Result<NanBox, ExecError> {
        match target {
            Expr::Ident(id) => {
                // A bare identifier inside `with (obj)` first resolves against the
                // with-object's properties (via `[[Get]]`, so accessors fire).
                if let Some(h) = self.with_binding(&id.name) {
                    return self.read_member(h, &id.name);
                }
                Ok(self.current.get(&id.name).unwrap_or(NanBox::undefined()))
            }
            Expr::Member {
                object, property, ..
            } => {
                let obj = self.eval(object)?;
                match obj.as_handle() {
                    Some(raw) => self.member(Handle::from_raw(raw), property),
                    None => Ok(NanBox::undefined()),
                }
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
    }

    /// One `IteratorStep` for array-destructuring: calls `next()`, returns
    /// `Ok(Some(value))` for a yielded element, `Ok(None)` at end-of-iteration
    /// (`{ done: true }`). An abrupt result-access is propagated. Callers treat
    /// both `Ok(None)` and `Err(_)` as "iterator done" for `IteratorClose` purposes.
    fn dstr_iter_step(
        &mut self,
        ih: Handle,
        iterator: NanBox,
    ) -> Result<Option<NanBox>, ExecError> {
        let next_fn = self.read_member(ih, "next")?;
        let res = self.call_with_this(next_fn, iterator, &[])?;
        let Some(rh) = res.as_handle().map(Handle::from_raw) else {
            return Err(ExecError::Throw(
                self.new_str("iterator result is not an object"),
            ));
        };
        let done = self.read_member(rh, "done")?;
        if self.realm.truthy(done) {
            return Ok(None);
        }
        Ok(Some(self.read_member(rh, "value")?))
    }

    /// Interleaved array destructuring-assignment over a *user* iterator `ih`
    /// (no rest element). For each element, a simple leaf target's reference is
    /// evaluated *before* the iterator step (spec order), then the step runs, then
    /// the value is assigned. `IteratorClose` is performed once on a normal finish
    /// (a non-Object/uncallable `return` throws), suppressed on an abrupt finish,
    /// and skipped entirely when the iterator is already done.
    fn assign_destructure_array_iter(
        &mut self,
        elements: &'a [ArrayElement],
        ih: Handle,
    ) -> Result<(), ExecError> {
        let iterator = NanBox::handle(ih.to_raw());
        // `exhausted` mirrors `iteratorRecord.[[done]]`: once the iterator finishes
        // or a step completes abruptly it is done and must not be closed. Only an
        // error from a *target* (reference eval or assignment) with the iterator not
        // yet done triggers `IteratorClose`.
        let mut exhausted = false;
        let result: Result<(), ExecError> = 'pat: {
            for el in elements {
                // For a simple leaf target (`a`, `obj[k]`) the reference is evaluated
                // before the step, so its side effects run first.
                let target = match el {
                    ArrayElement::Item(e) if Self::is_simple_assign_target(e) => {
                        match self.eval_assign_ref(e) {
                            Ok(r) => Some(r),
                            Err(e) => break 'pat Err(e),
                        }
                    }
                    _ => None,
                };
                let v = if exhausted {
                    NanBox::undefined()
                } else {
                    let step = self.dstr_iter_step(ih, iterator);
                    exhausted = !matches!(step, Ok(Some(_)));
                    match step {
                        Ok(opt) => opt.unwrap_or(NanBox::undefined()),
                        Err(e) => break 'pat Err(e),
                    }
                };
                let assigned = match el {
                    ArrayElement::Hole => Ok(()),
                    ArrayElement::Item(e) => match target {
                        Some(r) => self.set_assign_ref(r, v),
                        None => self.assign_destructure(e, v),
                    },
                    ArrayElement::Spread(_) => unreachable!("has_rest guard"),
                };
                if let Err(e) = assigned {
                    break 'pat Err(e);
                }
            }
            Ok(())
        };
        match result {
            Ok(()) => {
                if !exhausted {
                    self.iterator_close(ih)?;
                }
                Ok(())
            }
            Err(e) => {
                if !exhausted {
                    let _ = self.iterator_close(ih);
                }
                Err(e)
            }
        }
    }

    /// Destructures `value` into an assignment pattern of existing targets
    /// (`[a, b] = …`, `({ x: obj.p } = …)`), recursing into nested patterns.
    pub(crate) fn assign_destructure(
        &mut self,
        target: &'a Expr,
        value: NanBox,
    ) -> Result<(), ExecError> {
        match target {
            Expr::Array { elements, .. } => {
                let has_rest = elements
                    .iter()
                    .any(|e| matches!(e, ArrayElement::Spread(_)));
                // Without a rest target, a *user* iterator is pulled lazily for only
                // the values the pattern needs, then closed (`IteratorClose`) — so
                // `[a, b] = infiniteIterator` terminates and `return()` runs.
                // (Mirrors the declaration path in `bind_pattern`.) Built-in
                // iterables (arrays/strings/Sets/generators) take the eager path.
                if !has_rest
                    && let Some(ih) = self.for_of_get_iterator(value)?
                    && self.realm.get_property(ih, GEN_BUF).is_none()
                {
                    // Real user iterator: interleave `next()`, target evaluation and
                    // assignment per spec, and run `IteratorClose` at the right moment.
                    self.assign_destructure_array_iter(elements, ih)
                } else {
                    // Built-in iterables / generators / rest patterns: eager value list.
                    let items = self.iterate_values(value)?;
                    let mut i = 0;
                    for el in elements {
                        match el {
                            ArrayElement::Hole => i += 1,
                            ArrayElement::Item(e) => {
                                let v = items.get(i).copied().unwrap_or(NanBox::undefined());
                                self.assign_destructure(e, v)?;
                                i += 1;
                            }
                            ArrayElement::Spread(e) => {
                                let rest = items[i.min(items.len())..].to_vec();
                                let h = NanBox::handle(self.realm.new_array(rest).to_raw());
                                self.assign_destructure(e, h)?;
                            }
                        }
                    }
                    Ok(())
                }
            }
            Expr::Object { members, .. } => {
                // Object destructuring requires a coercible value: `null`/`undefined`
                // throw a TypeError (RequireObjectCoercible). A primitive RHS is boxed
                // to its wrapper object so its own properties (e.g. a string's indices
                // and `length`) participate in the pattern and in a `...rest`.
                if matches!(value.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Cannot destructure 'null' or 'undefined' as an object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let src = self.require_object_coercible_to_object(value, "destructuring")?;
                let mut used: Vec<String> = Vec::new();
                for m in members {
                    match m {
                        ObjectMember::Property {
                            key, value: tgt, ..
                        } => {
                            let k = self.eval_prop_key(key)?;
                            // Read through `read_member` so accessors fire and
                            // inherited / length properties resolve.
                            let v = self.read_member(src, &k)?;
                            used.push(k);
                            self.assign_destructure(tgt, v)?;
                        }
                        ObjectMember::Spread { value: tgt, .. } => {
                            let obj = self.realm.new_object();
                            for k in self.realm.object_keys(src).unwrap_or_default() {
                                if !used.contains(&k) {
                                    // `read_member` fires an own getter (once);
                                    // the rest object gets a plain data property.
                                    let pv = self.read_member(src, &k)?;
                                    self.realm.set_property(obj, &k, pv);
                                }
                            }
                            self.assign_destructure(tgt, NanBox::handle(obj.to_raw()))?;
                        }
                        ObjectMember::Accessor { .. } => {}
                    }
                }
                Ok(())
            }
            // A defaulted target in a pattern (`[a = 1] = …`, `{ x: a = 1 } = …`):
            // use the default when the source value is `undefined`.
            Expr::Assign {
                op: AssignOp::Assign,
                target: inner,
                value: default_expr,
                ..
            } => {
                let v = if matches!(value.unpack(), Unpacked::Undefined) {
                    let d = self.eval(default_expr)?;
                    // `[a = function(){}] = []` names the function after the target
                    // (`a`) when the default is an anonymous function/class/arrow.
                    if let Expr::Ident(id) = &**inner
                        && matches!(
                            &**default_expr,
                            Expr::Function(_) | Expr::Class(_) | Expr::Arrow(_)
                        )
                    {
                        self.set_fn_name(d, &id.name);
                    }
                    d
                } else {
                    value
                };
                self.assign_destructure(inner, v)
            }
            // A leaf target (identifier or member).
            _ => self.assign_to(target, value),
        }
    }

    /// Assigns `value` to an existing target (an identifier or member).
    /// Finds the innermost active `with` object whose environment record provides
    /// `name` — i.e. it `HasProperty(name)` and `name` is not blocked by the
    /// object's `@@unscopables`. Returns the object handle, or `None` to fall back
    /// to the lexical scope chain.
    pub(crate) fn with_binding(&mut self, name: &str) -> Option<Handle> {
        if self.with_stack.is_empty() {
            return None;
        }
        for obj in self.with_stack.clone().into_iter().rev() {
            let Some(h) = obj.as_handle().map(Handle::from_raw) else {
                continue;
            };
            if !self.has_property_chain(h, name) {
                continue;
            }
            // `@@unscopables`: a truthy entry blocks the binding (the lexical scope
            // shows through instead).
            let unscopables_sym = self.well_known_symbol("unscopables");
            let unscopables_key = self.member_key(unscopables_sym);
            let unscopables = self
                .read_member(h, &unscopables_key)
                .ok()
                .and_then(|v| v.as_handle().map(Handle::from_raw));
            if let Some(u) = unscopables
                && let Ok(blocked) = self.read_member(u, name)
                && self.realm.truthy(blocked)
            {
                continue;
            }
            return Some(h);
        }
        None
    }

    /// Whether `target` is a *simple* assignment leaf (an identifier or member
    /// reference) — as opposed to a nested destructuring pattern or a defaulted
    /// element. For a simple leaf the DestructuringAssignmentTarget reference is
    /// evaluated *before* the corresponding iterator step (spec ordering).
    fn is_simple_assign_target(target: &Expr) -> bool {
        matches!(target, Expr::Ident(_) | Expr::Member { .. })
    }

    /// Evaluates the *reference* of a simple assignment leaf (its object/key for a
    /// member, or the name for an identifier) without yet producing a value, then
    /// returns a closure-free handle to complete the assignment later via
    /// [`set_assign_ref`](Self::set_assign_ref). Side effects in the object/key
    /// expressions happen here, matching the spec's "evaluate target, then step".
    fn eval_assign_ref(&mut self, target: &'a Expr) -> Result<AssignRef, ExecError> {
        match target {
            Expr::Member {
                object, property, ..
            } => {
                if matches!(&**object, Expr::Super(_)) {
                    let name = self.eval_prop_key(property)?;
                    return Ok(AssignRef::Super { name });
                }
                let obj = self.eval(object)?;
                let key = self.eval_prop_key(property)?;
                Ok(AssignRef::Member { obj, key })
            }
            Expr::Ident(id) => Ok(AssignRef::Ident(String::from(&*id.name))),
            _ => unreachable!("eval_assign_ref on a non-simple target"),
        }
    }

    /// Completes a pre-evaluated assignment reference with `value`.
    fn set_assign_ref(&mut self, r: AssignRef, value: NanBox) -> Result<(), ExecError> {
        match r {
            AssignRef::Ident(name) => self.assign_to_name(&name, value),
            AssignRef::Super { name } => self.assign_super_member(&name, value),
            AssignRef::Member { obj, key } => {
                if let Some(raw) = obj.as_handle() {
                    let k = self.new_str(&key);
                    self.assign_member_value(Handle::from_raw(raw), k, value)?;
                }
                Ok(())
            }
        }
    }

    /// Assigns `value` to the identifier reference `name`, applying `with`-object
    /// shadowing, the `const` reassignment check, and the strict/sloppy rules for
    /// an unresolvable reference. Shared by `assign_to` and `set_assign_ref`.
    pub(crate) fn assign_to_name(&mut self, name: &str, value: NanBox) -> Result<(), ExecError> {
        // A bare identifier inside `with (obj)` assigns to the with-object's
        // property (via `[[Set]]`, so setters fire) when it provides the name.
        if let Some(h) = self.with_binding(name) {
            let key = self.new_str(name);
            self.assign_member_value(h, key, value)?;
            return Ok(());
        }
        // Reassigning a `const` binding is a TypeError.
        if self.current.is_const(name) {
            let m = self.new_str("Assignment to constant variable.");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        if !self.current.set(name, value) {
            // Strict mode forbids creating an implicit global.
            if self.strict {
                let m = self.new_str(&alloc::format!("{name} is not defined"));
                return Err(ExecError::Throw(
                    self.make_error(N_REFERENCE_ERROR, Some(m)),
                ));
            }
            // Sloppy mode: assigning to an unresolvable reference creates a property
            // on the *global* object (not a binding in the current scope), so it is
            // visible after a block/loop scope is popped.
            self.declare_sloppy_global(name, value);
        }
        Ok(())
    }

    pub(crate) fn assign_to(&mut self, target: &'a Expr, value: NanBox) -> Result<(), ExecError> {
        match target {
            Expr::Ident(id) => self.assign_to_name(&id.name, value),
            Expr::Member {
                object, property, ..
            } => {
                // `super.x = v` invokes the inherited setter with the current `this`.
                if matches!(&**object, Expr::Super(_)) {
                    let name = self.eval_prop_key(property)?;
                    return self.assign_super_member(&name, value);
                }
                let obj = self.eval(object)?;
                if let Some(raw) = obj.as_handle() {
                    self.assign_member(Handle::from_raw(raw), property, value)?;
                }
                Ok(())
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
    }

    pub(crate) fn exec_switch(
        &mut self,
        discriminant: &'a Expr,
        cases: &'a [crate::ast::SwitchCase],
    ) -> Result<Flow, ExecError> {
        let value = self.eval(discriminant)?;
        // Find the first matching `case` (strict equality), else `default`.
        let mut start = None;
        for (i, case) in cases.iter().enumerate() {
            if let Some(test) = &case.test {
                let t = self.eval(test)?;
                if self.realm.strict_equals(value, t) {
                    start = Some(i);
                    break;
                }
            }
        }
        if start.is_none() {
            start = cases.iter().position(|c| c.test.is_none());
        }
        let Some(start) = start else {
            return Ok(Flow::Normal(NanBox::undefined()));
        };
        // Run from the matched clause, falling through until `break`.
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        let result = (|| {
            for case in &cases[start..] {
                for stmt in &case.body {
                    match self.exec(stmt)? {
                        // A plain `break` ends the switch; everything else
                        // (labeled break, continue, return) bubbles out.
                        Flow::Break(None) => return Ok(Flow::Normal(NanBox::undefined())),
                        Flow::Normal(_) => {}
                        other => return Ok(other),
                    }
                }
            }
            Ok(Flow::Normal(NanBox::undefined()))
        })();
        self.current = saved;
        result
    }

    pub(crate) fn exec_for(
        &mut self,
        init: Option<&'a ForInit>,
        test: Option<&'a Expr>,
        update: Option<&'a Expr>,
        body: &'a Stmt,
    ) -> Result<Flow, ExecError> {
        let label = self.pending_label.take();
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        // For a `let`/`const` head, each iteration gets a fresh binding (so a
        // closure created in the body captures that iteration's value).
        let per_iter_names: Vec<String> = match init {
            Some(ForInit::Var(decl)) if decl.kind != crate::ast::VarDeclKind::Var => decl
                .declarations
                .iter()
                .filter_map(|d| match &d.target {
                    BindingTarget::Ident(Ident { name, .. }) => Some(String::from(&**name)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let result = (|| {
            match init {
                Some(ForInit::Var(decl)) => self.exec_var(decl)?,
                Some(ForInit::Expr(e)) => {
                    self.eval(e)?;
                }
                None => {}
            }
            loop {
                let go = match test {
                    Some(t) => self.eval_truthy(t)?,
                    None => true,
                };
                if !go {
                    break;
                }
                // Run the body in a per-iteration scope seeded from the loop
                // variables, then copy any mutations back for test/update.
                let flow = if per_iter_names.is_empty() {
                    self.exec(body)?
                } else {
                    let iter = self.current.child();
                    for name in &per_iter_names {
                        iter.declare(name, self.current.get(name).unwrap_or(NanBox::undefined()));
                    }
                    let loop_scope = core::mem::replace(&mut self.current, iter);
                    let f = self.exec(body);
                    for name in &per_iter_names {
                        if let Some(v) = self.current.get(name) {
                            loop_scope.set(name, v);
                        }
                    }
                    self.current = loop_scope;
                    f?
                };
                match loop_action(flow, &label) {
                    LoopAction::Next => {}
                    LoopAction::Stop => break,
                    LoopAction::Propagate(f) => return Ok(f),
                }
                if let Some(u) = update {
                    self.eval(u)?;
                }
            }
            Ok(Flow::Normal(NanBox::undefined()))
        })();
        self.current = saved;
        result
    }
}
