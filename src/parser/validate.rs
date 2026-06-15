//! Static-semantics (early-error) validation pass over a freshly parsed
//! [`Program`].
//!
//! The recursive-descent parser in this module accepts the *cover grammar* — it
//! produces an AST for any source that matches the productions, deferring a
//! family of context-sensitive "early errors" that the specification requires to
//! be reported at parse time (before any evaluation). This pass walks the
//! finished tree and enforces those rules, returning a syntax [`Error`] so the
//! engine surfaces a parse-phase `SyntaxError` exactly as the spec mandates.
//!
//! Rules implemented here (all parse-time `SyntaxError`s):
//! - **Private names**: every `obj.#x` / `#x in obj` reference must resolve to a
//!   private name declared in an enclosing class body; `#constructor` is never a
//!   valid private name; duplicate private names in one class body are illegal (a
//!   get/set accessor pair is the sole exception); `delete` of a private
//!   reference is illegal; `super.#x` is illegal.
//! - **Class members**: at most one `constructor`; `constructor` may not be a
//!   getter/setter/generator/async method or a field; a `static` member may not
//!   be named `prototype`; a class field may not be named `constructor`. A field
//!   initializer / static block may not reference `arguments` or call `super()`.
//! - **`super()`** calls outside a derived-class constructor.
//! - **`with`** is forbidden in strict mode, and its body may not be a
//!   declaration.
//! - **Single-statement position**: a lexical/class/generator/async-function
//!   declaration may not be the body of an `if`/`else`/loop/labeled statement
//!   (and a plain function declaration is also rejected in a loop body or strict
//!   mode).
//! - **Lexical redeclaration**: duplicate lexically-declared names, and
//!   `var`/lexical conflicts, within a block, switch, function, or program
//!   scope; lexical declarations may not bind the name `let`.
//! - **Duplicate parameters** in a strict-mode or non-simple parameter list, and
//!   a `"use strict"` directive in a non-simple-parameter function.
//! - **Jumps/labels**: `return` only inside a function; `break`/`continue` only
//!   with a valid (in-scope, loop-vs-switch-appropriate) target; no duplicate
//!   label; the label/jump state resets at each function boundary.
//! - **Assignment/update targets**: `++`/`--` and compound-assignment operands
//!   must be simple references; a `=` target must be a valid (possibly
//!   destructuring) target with a trailing-only rest; in strict mode neither the
//!   target nor any binding may be `eval`/`arguments`.
//! - **`new.target`** only inside a function-like body (not top-level).
//! - **Optional chains** may not carry a tagged template tail.

use crate::ast::{
    Arrow, ArrowBody, BindingTarget, Class, ClassMember, Expr, ForLeft, Function, MethodKind,
    ObjectMember, Param, Program, PropertyKey, SourceType, Stmt, UnaryOp, VarDecl, VarDeclKind,
};
use crate::common::Span;
use crate::error::{Error, Result};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Validates a parsed [`Program`], returning the first early error found.
pub(crate) fn validate_program(program: &Program) -> Result<()> {
    let strict = program.source_type == SourceType::Module || body_is_strict(&program.body);
    let mut v = Validator {
        private_scopes: Vec::new(),
        labels: Vec::new(),
        is_module: program.source_type == SourceType::Module,
    };
    let ctx = Ctx::top(strict);
    v.check_top_level_scope(&program.body, strict)?;
    // `using` / `await using` declarations are not permitted at the top level of
    // a *Script* (they are allowed in a Module, a block, or a function body).
    if program.source_type == SourceType::Script {
        for stmt in &program.body {
            if let Stmt::Var(decl) = stmt
                && matches!(decl.kind, VarDeclKind::Using | VarDeclKind::AwaitUsing)
            {
                return Err(v.err(
                    decl.span,
                    "a `using` declaration is not allowed at the top level of a script",
                ));
            }
        }
    }
    for stmt in &program.body {
        v.stmt(stmt, &ctx)?;
    }
    Ok(())
}

/// Whether a directive prologue in `body` contains a literal `"use strict"`.
fn body_is_strict(body: &[Stmt]) -> bool {
    for stmt in body {
        if let Stmt::Expr { expression, .. } = stmt {
            if let Expr::Str { value, span } = &**expression {
                // A directive's *source* must be exactly `"use strict"` (11 cooked
                // bytes inside 13 source bytes — quotes included — i.e. no
                // escapes). Checking the span length rules out e.g. `"use\x20strict"`.
                if &**value == b"use strict" && span.end.saturating_sub(span.start) == 12 {
                    return true;
                }
                // Other directive — keep scanning the prologue.
            } else {
                break;
            }
        } else {
            break;
        }
    }
    false
}

/// Contextual flags threaded through the statement/expression walk.
#[derive(Clone, Copy)]
struct Ctx {
    /// Whether the current code runs in strict mode.
    strict: bool,
    /// Whether a `super(...)` call is syntactically permitted here — true only
    /// inside the constructor of a derived class (one with an `extends` clause).
    allow_super_call: bool,
    /// Whether we are directly inside a class field initializer or static
    /// initialization block (with no intervening function boundary). In that
    /// position a reference to `arguments` and a `super(...)` call are early
    /// errors.
    in_field_init: bool,
    /// Whether we are inside a function body (so `return` is allowed).
    in_function: bool,
    /// Whether `new.target` is permitted — true inside any function/method body,
    /// class field initializer, or static block; false at top-level script /
    /// module code.
    allow_new_target: bool,
    /// Whether an unlabeled `break` has a target (inside a loop or `switch`).
    in_breakable: bool,
    /// Whether an unlabeled `continue` has a target (inside a loop).
    in_iteration: bool,
}

impl Ctx {
    /// A fresh non-strict top-level context.
    fn top(strict: bool) -> Self {
        Ctx {
            strict,
            allow_super_call: false,
            in_field_init: false,
            in_function: false,
            allow_new_target: false,
            in_breakable: false,
            in_iteration: false,
        }
    }

    /// A fresh context at a function boundary (the jump/`return` state resets;
    /// `super`/field-init are passed in by the caller). `new.target` is always
    /// available inside a function-like body.
    fn function_boundary(strict: bool, allow_super_call: bool, in_field_init: bool) -> Self {
        Ctx {
            strict,
            allow_super_call,
            in_field_init,
            in_function: true,
            allow_new_target: true,
            in_breakable: false,
            in_iteration: false,
        }
    }
}

struct Validator {
    /// Stack of private-name scopes — one frame per enclosing class body, each
    /// holding the private names that class declares.
    private_scopes: Vec<Vec<Box<str>>>,
    /// Active labels in scope, innermost last, each tagged with whether it
    /// labels an iteration statement (so `continue label` is only valid for an
    /// iteration label). Reset to empty at every function boundary.
    labels: Vec<(Box<str>, bool)>,
    /// Whether the program's goal symbol is Module (vs Script). `import.meta`
    /// is valid only in a module.
    is_module: bool,
}

/// What a private member declaration is, for duplicate detection.
#[derive(Clone, Copy, PartialEq)]
enum PrivKind {
    Field,
    Method,
    Get,
    Set,
}

impl Validator {
    fn err(&self, span: Span, msg: &str) -> Error {
        Error::syntax(msg, span)
    }

    // --- statements -----------------------------------------------------

    fn stmt(&mut self, stmt: &Stmt, ctx: &Ctx) -> Result<()> {
        match stmt {
            Stmt::Expr { expression, .. } => self.expr(expression, ctx),
            Stmt::Block { body, .. } => {
                self.check_lexical_scope(body, ctx.strict)?;
                for s in body {
                    self.stmt(s, ctx)?;
                }
                Ok(())
            }
            Stmt::Empty { .. } | Stmt::Debugger { .. } => Ok(()),
            Stmt::Var(decl) => self.var_decl(decl, ctx),
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                self.expr(test, ctx)?;
                self.check_substatement(consequent, ctx)?;
                self.stmt(consequent, ctx)?;
                if let Some(a) = alternate {
                    self.check_substatement(a, ctx)?;
                    self.stmt(a, ctx)?;
                }
                Ok(())
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => {
                if let Some(init) = init {
                    match init {
                        crate::ast::ForInit::Var(d) => self.var_decl(d, ctx)?,
                        crate::ast::ForInit::Expr(e) => self.expr(e, ctx)?,
                    }
                }
                if let Some(t) = test {
                    self.expr(t, ctx)?;
                }
                if let Some(u) = update {
                    self.expr(u, ctx)?;
                }
                self.check_loop_body(body)?;
                self.stmt(body, &loop_ctx(ctx))
            }
            Stmt::ForIn {
                left, right, body, ..
            }
            | Stmt::ForOf {
                left, right, body, ..
            } => {
                self.for_left(left, ctx)?;
                self.expr(right, ctx)?;
                self.check_loop_body(body)?;
                self.stmt(body, &loop_ctx(ctx))
            }
            Stmt::While { test, body, .. } => {
                self.expr(test, ctx)?;
                self.check_loop_body(body)?;
                self.stmt(body, &loop_ctx(ctx))
            }
            Stmt::DoWhile { body, test, .. } => {
                self.check_loop_body(body)?;
                self.stmt(body, &loop_ctx(ctx))?;
                self.expr(test, ctx)
            }
            Stmt::Switch {
                discriminant,
                cases,
                ..
            } => {
                self.expr(discriminant, ctx)?;
                let all: Vec<&Stmt> = cases.iter().flat_map(|c| c.body.iter()).collect();
                self.check_lexical_scope_refs(&all, false, ctx.strict)?;
                // A `switch` is a `break` target but not a `continue` target.
                let mut c = *ctx;
                c.in_breakable = true;
                for case in cases {
                    if let Some(t) = &case.test {
                        self.expr(t, ctx)?;
                    }
                    for s in &case.body {
                        // A `using` / `await using` declaration may not appear as
                        // a direct statement of a `CaseClause`/`DefaultClause`
                        // (it must be inside a block).
                        if let Stmt::Var(decl) = s
                            && matches!(decl.kind, VarDeclKind::Using | VarDeclKind::AwaitUsing)
                        {
                            return Err(self.err(
                                decl.span,
                                "a `using` declaration is not allowed directly in a `switch` case",
                            ));
                        }
                        self.stmt(s, &c)?;
                    }
                }
                Ok(())
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                self.check_lexical_scope(block, ctx.strict)?;
                for s in block {
                    self.stmt(s, ctx)?;
                }
                if let Some(h) = handler {
                    self.check_lexical_scope(&h.body, ctx.strict)?;
                    if let Some(p) = &h.param {
                        self.binding_target(p, ctx)?;
                    }
                    for s in &h.body {
                        self.stmt(s, ctx)?;
                    }
                }
                if let Some(f) = finalizer {
                    self.check_lexical_scope(f, ctx.strict)?;
                    for s in f {
                        self.stmt(s, ctx)?;
                    }
                }
                Ok(())
            }
            Stmt::Return { argument, span } => {
                if !ctx.in_function {
                    return Err(self.err(*span, "`return` is only valid inside a function"));
                }
                if let Some(a) = argument {
                    self.expr(a, ctx)?;
                }
                Ok(())
            }
            Stmt::Break { label, span } => {
                match label {
                    Some(l) => {
                        if !self.labels.iter().any(|(n, _)| **n == *l.name) {
                            return Err(self.err(l.span, "undefined break label"));
                        }
                    }
                    None if !ctx.in_breakable => {
                        return Err(self.err(*span, "`break` must be inside a loop or `switch`"));
                    }
                    None => {}
                }
                Ok(())
            }
            Stmt::Continue { label, span } => {
                match label {
                    Some(l) => {
                        // `continue label` requires the label to mark an
                        // iteration statement.
                        match self.labels.iter().find(|(n, _)| **n == *l.name) {
                            None => return Err(self.err(l.span, "undefined continue label")),
                            Some((_, is_iter)) if !*is_iter => {
                                return Err(self.err(
                                    l.span,
                                    "`continue` label does not denote an iteration statement",
                                ));
                            }
                            Some(_) => {}
                        }
                    }
                    None if !ctx.in_iteration => {
                        return Err(self.err(*span, "`continue` must be inside a loop"));
                    }
                    None => {}
                }
                Ok(())
            }
            Stmt::Throw { argument, .. } => self.expr(argument, ctx),
            Stmt::Labeled { label, body, .. } => {
                // A `LabelIdentifier` may not be a strict-mode reserved word in
                // strict code (e.g. an escaped `yield` label `yield:`).
                if ctx.strict && is_strict_reserved_word(&label.name) {
                    return Err(self.err(
                        label.span,
                        "a strict-mode reserved word may not be used as a label",
                    ));
                }
                // A duplicate label in the enclosing set is an early error.
                if self.labels.iter().any(|(n, _)| **n == *label.name) {
                    return Err(self.err(label.span, "label has already been declared"));
                }
                self.check_labeled_body(body, ctx)?;
                // Mark the label as an iteration label when it (transitively
                // through nested labels) labels a loop, so `continue label` is
                // accepted.
                let is_iter = labels_an_iteration(body);
                self.labels.push((label.name.clone(), is_iter));
                // A labeled iteration statement is itself a `continue` target for
                // its own label; the loop body sets `in_iteration` already.
                let result = self.stmt(body, ctx);
                self.labels.pop();
                result
            }
            Stmt::With { object, body, .. } => {
                if ctx.strict {
                    return Err(self.err(stmt.span(), "`with` is not allowed in strict mode"));
                }
                self.expr(object, ctx)?;
                self.check_substatement(body, ctx)?;
                self.stmt(body, ctx)
            }
            Stmt::Function(f) => self.function(f, ctx),
            Stmt::Class(c) => self.class(c, ctx),
            Stmt::Import(_) | Stmt::Export(_) => Ok(()),
        }
    }

    /// The single-statement body of an `if`/`else` branch may not be a
    /// declaration. A plain `function` declaration is permitted in sloppy mode
    /// (Annex B); a `let`/`const`/class/generator/async-function is never
    /// permitted, and a plain `function` is also forbidden in strict mode.
    fn check_substatement(&self, body: &Stmt, ctx: &Ctx) -> Result<()> {
        self.reject_decl_substatement(body, ctx, /* allow_sloppy_fn */ true)
    }

    /// The body of a `for`/`while`/`do-while` loop may not be *any* declaration
    /// — not even a plain `function` (Annex B does not apply to loop bodies).
    fn check_loop_body(&self, body: &Stmt) -> Result<()> {
        // Loop bodies forbid even sloppy function declarations.
        self.reject_decl_substatement(body, &Ctx::top(true), /* allow_sloppy_fn */ false)
    }

    /// The body of a labeled statement may not be a `let`/`const`/class
    /// declaration or a generator/async function; a plain `function`
    /// declaration is allowed in sloppy mode only (Annex B).
    fn check_labeled_body(&self, body: &Stmt, ctx: &Ctx) -> Result<()> {
        // `LabelledItem : FunctionDeclaration` is allowed only for a non-async,
        // non-generator function in sloppy mode.
        self.reject_decl_substatement(body, ctx, /* allow_sloppy_fn */ true)
    }

    /// Shared core: rejects a declaration found in single-statement position.
    fn reject_decl_substatement(
        &self,
        body: &Stmt,
        ctx: &Ctx,
        allow_sloppy_fn: bool,
    ) -> Result<()> {
        match body {
            Stmt::Var(decl) if decl.kind != VarDeclKind::Var => Err(self.err(
                decl.span,
                "lexical declarations may not appear in single-statement position",
            )),
            Stmt::Class(c) => Err(self.err(
                c.span,
                "a class declaration may not appear in single-statement position",
            )),
            Stmt::Function(f) => {
                // Generators and async functions are never allowed here; a plain
                // function is allowed only in sloppy mode where Annex B permits.
                if f.is_generator || f.is_async || ctx.strict || !allow_sloppy_fn {
                    Err(self.err(
                        f.span,
                        "a function declaration may not appear in single-statement position",
                    ))
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }

    fn for_left(&mut self, left: &ForLeft, ctx: &Ctx) -> Result<()> {
        match left {
            ForLeft::Decl { kind, target, .. } => {
                if *kind != VarDeclKind::Var {
                    self.check_lexical_binding_names(target)?;
                }
                self.binding_target(target, ctx)
            }
            ForLeft::Target(e) => {
                if !is_valid_assign_target(e) {
                    return Err(self.err(e.span(), "invalid for-in/of assignment target"));
                }
                if ctx.strict {
                    self.check_not_eval_arguments(e)?;
                }
                self.expr(e, ctx)
            }
        }
    }

    fn var_decl(&mut self, decl: &VarDecl, ctx: &Ctx) -> Result<()> {
        for d in &decl.declarations {
            if decl.kind != VarDeclKind::Var {
                self.check_lexical_binding_names(&d.target)?;
            }
            if let Some(init) = &d.init {
                self.expr(init, ctx)?;
            }
            self.binding_target(&d.target, ctx)?;
        }
        Ok(())
    }

    /// A strict-mode `BindingIdentifier` may not be `eval`, `arguments`, or any
    /// of the strict-mode future-reserved words (`implements`, `interface`,
    /// `let`, `package`, `private`, `protected`, `public`, `static`, `yield`).
    /// The parser accepts these spellings as identifiers (they are valid in
    /// sloppy code); this is where the strict-mode restriction is enforced.
    fn check_binding_ident_name(&self, name: &str, span: Span) -> Result<()> {
        if name == "eval" || name == "arguments" {
            return Err(self.err(
                span,
                "`eval` and `arguments` may not be bound in strict mode",
            ));
        }
        if is_strict_reserved_word(name) {
            return Err(self.err(
                span,
                "a strict-mode reserved word may not be bound in strict mode",
            ));
        }
        Ok(())
    }

    /// Lexical declarations (`let`/`const`) may not bind the name `let`.
    fn check_lexical_binding_names(&self, target: &BindingTarget) -> Result<()> {
        let mut names = Vec::new();
        collect_bound_names(target, &mut names);
        for (name, span) in names {
            if name == "let" {
                return Err(self.err(span, "`let` is not a valid lexical binding name"));
            }
        }
        Ok(())
    }

    fn binding_target(&mut self, target: &BindingTarget, ctx: &Ctx) -> Result<()> {
        match target {
            BindingTarget::Ident(id) => {
                if ctx.strict {
                    self.check_binding_ident_name(&id.name, id.span)?;
                }
                Ok(())
            }
            BindingTarget::Array(p) => {
                for el in &p.elements {
                    match el {
                        crate::ast::ArrayPatternElement::Hole => {}
                        crate::ast::ArrayPatternElement::Item {
                            target, default, ..
                        } => {
                            self.binding_target(target, ctx)?;
                            if let Some(d) = default {
                                self.expr(d, ctx)?;
                            }
                        }
                        crate::ast::ArrayPatternElement::Rest { target, .. } => {
                            self.binding_target(target, ctx)?;
                        }
                    }
                }
                Ok(())
            }
            BindingTarget::Object(p) => {
                for prop in &p.properties {
                    if let PropertyKey::Computed(e) = &prop.key {
                        self.expr(e, ctx)?;
                    }
                    self.binding_target(&prop.value, ctx)?;
                    if let Some(d) = &prop.default {
                        self.expr(d, ctx)?;
                    }
                }
                if let Some(r) = &p.rest {
                    self.binding_target(r, ctx)?;
                }
                Ok(())
            }
        }
    }

    // --- functions ------------------------------------------------------

    fn function(&mut self, f: &Function, ctx: &Ctx) -> Result<()> {
        let strict = ctx.strict || body_is_strict(&f.body);
        if strict && let Some(id) = &f.id {
            self.check_binding_ident_name(&id.name, id.span)?;
        }
        self.check_param_yield_await(&f.params, f.is_generator, f.is_async)?;
        self.check_params(&f.params, strict, &f.body)?;
        // A regular function establishes its own `arguments`/`super` bindings and
        // a fresh label/jump environment.
        let c = Ctx::function_boundary(strict, false, false);
        let saved_labels = core::mem::take(&mut self.labels);
        for p in &f.params {
            self.param(p, &c)?;
        }
        self.check_top_level_scope(&f.body, strict)?;
        for s in &f.body {
            self.stmt(s, &c)?;
        }
        self.labels = saved_labels;
        Ok(())
    }

    fn method_function(
        &mut self,
        f: &Function,
        parent: &Ctx,
        allow_super_call: bool,
    ) -> Result<()> {
        let strict = parent.strict || body_is_strict(&f.body);
        self.check_param_yield_await(&f.params, f.is_generator, f.is_async)?;
        self.check_params(&f.params, strict, &f.body)?;
        let c = Ctx::function_boundary(strict, allow_super_call, false);
        let saved_labels = core::mem::take(&mut self.labels);
        for p in &f.params {
            self.param(p, &c)?;
        }
        self.check_top_level_scope(&f.body, strict)?;
        for s in &f.body {
            self.stmt(s, &c)?;
        }
        self.labels = saved_labels;
        Ok(())
    }

    fn arrow(&mut self, a: &Arrow, ctx: &Ctx) -> Result<()> {
        let strict = ctx.strict
            || match &a.body {
                ArrowBody::Block(body) => body_is_strict(body),
                ArrowBody::Expr(_) => false,
            };
        let body_slice: &[Stmt] = match &a.body {
            ArrowBody::Block(body) => body,
            ArrowBody::Expr(_) => &[],
        };
        // An async arrow's parameters are in an `[+Await]` context: an `await`
        // expression there is a Syntax Error. (An arrow is never a generator.)
        self.check_param_yield_await(&a.params, false, a.is_async)?;
        self.check_params(&a.params, strict, body_slice)?;
        // An arrow inherits `super`, `this`, and `arguments` (and thus the
        // field-initializer restrictions) from its enclosing scope, but is a
        // function boundary for `return` and the label/jump environment.
        let c = Ctx {
            strict,
            allow_super_call: ctx.allow_super_call,
            in_field_init: ctx.in_field_init,
            in_function: true,
            // `new.target` is inherited from the enclosing scope: an arrow has no
            // `new.target` of its own, so it is only valid if the surrounding
            // function provides one.
            allow_new_target: ctx.allow_new_target,
            in_breakable: false,
            in_iteration: false,
        };
        let saved_labels = core::mem::take(&mut self.labels);
        for p in &a.params {
            self.param(p, &c)?;
        }
        match &a.body {
            ArrowBody::Block(body) => {
                self.check_top_level_scope(body, strict)?;
                for s in body {
                    self.stmt(s, &c)?;
                }
            }
            ArrowBody::Expr(e) => self.expr(e, &c)?,
        }
        self.labels = saved_labels;
        Ok(())
    }

    /// Rejects a `yield` expression in a generator's parameter list and an
    /// `await` expression in an async function/arrow's parameter list (the
    /// `[+Yield]`/`[+Await]` parameter contexts forbid them). The parser parses
    /// params in the function's own context, so a `yield`/`await` *binding name*
    /// is already rejected there; this catches the *expression* forms in
    /// defaults and computed keys (e.g. `function* g(x = yield) {}`).
    fn check_param_yield_await(
        &self,
        params: &[Param],
        is_generator: bool,
        is_async: bool,
    ) -> Result<()> {
        if !is_generator && !is_async {
            return Ok(());
        }
        for p in params {
            if let Some((span, kind)) = first_param_yield_await(p, is_generator, is_async) {
                return Err(self.err(
                    span,
                    match kind {
                        YieldOrAwait::Yield => {
                            "`yield` is not allowed in a generator's parameter list"
                        }
                        YieldOrAwait::Await => {
                            "`await` is not allowed in an async function's parameter list"
                        }
                    },
                ));
            }
        }
        Ok(())
    }

    fn param(&mut self, p: &Param, ctx: &Ctx) -> Result<()> {
        if let Some(d) = &p.default {
            self.expr(d, ctx)?;
        }
        self.binding_target(&p.target, ctx)
    }

    /// Parameter-list early errors:
    /// - a name may not repeat in a strict-mode function or in any function with
    ///   a non-simple parameter list (defaults, destructuring, or a rest
    ///   element);
    /// - a function whose parameter list is non-simple may not contain a
    ///   `"use strict"` directive in its body.
    fn check_params(&self, params: &[Param], strict: bool, body: &[Stmt]) -> Result<()> {
        let simple = params
            .iter()
            .all(|p| !p.rest && p.default.is_none() && matches!(p.target, BindingTarget::Ident(_)));
        if !simple && body_is_strict(body) {
            // The directive begins the body; point the error there.
            let span = body.first().map_or(Span::new(0, 0), Stmt::span);
            return Err(self.err(
                span,
                "a `\"use strict\"` directive is not allowed in a function with a non-simple parameter list",
            ));
        }
        if simple && !strict {
            return Ok(());
        }
        let mut names: Vec<(String, Span)> = Vec::new();
        for p in params {
            let mut bound = Vec::new();
            collect_bound_names(&p.target, &mut bound);
            for (name, span) in bound {
                if names.iter().any(|(n, _)| *n == name) {
                    return Err(self.err(span, "duplicate parameter name not allowed here"));
                }
                names.push((name, span));
            }
        }
        Ok(())
    }

    // --- classes --------------------------------------------------------

    fn class(&mut self, c: &Class, _ctx: &Ctx) -> Result<()> {
        // Class bodies are always strict, so the class name is a strict binding.
        if let Some(id) = &c.id {
            self.check_binding_ident_name(&id.name, id.span)?;
        }
        let cls_ctx = Ctx::top(true);

        // 1. The heritage clause is checked with the *enclosing* private scope —
        //    the class's own private environment is not yet active there.
        if let Some(sc) = &c.super_class {
            self.expr(sc, &cls_ctx)?;
        }

        // 2. Collect this class's private names and validate member early errors.
        let mut privates: Vec<Box<str>> = Vec::new();
        let mut seen: Vec<(Box<str>, bool, PrivKind)> = Vec::new();
        let mut ctor_count = 0u32;
        for member in &c.body {
            match member {
                ClassMember::Method(m) => {
                    if let PropertyKey::Private(name) = &m.key {
                        if &**name == "constructor" {
                            return Err(
                                self.err(m.span, "`#constructor` is not a valid private name")
                            );
                        }
                        privates.push(name.clone());
                        let pk = match m.kind {
                            MethodKind::Get => PrivKind::Get,
                            MethodKind::Set => PrivKind::Set,
                            _ => PrivKind::Method,
                        };
                        self.check_private_dup(&mut seen, name, m.is_static, pk, m.span)?;
                    }
                    if matches!(m.kind, MethodKind::Constructor) {
                        ctor_count += 1;
                        if ctor_count > 1 {
                            return Err(self.err(m.span, "a class may have only one constructor"));
                        }
                    }
                    if m.is_static && key_is_named(&m.key, "prototype") {
                        return Err(
                            self.err(m.span, "a static class member may not be named `prototype`")
                        );
                    }
                    // A non-static getter/setter/generator/async method named
                    // `constructor` is illegal (only a plain method is the ctor).
                    if !m.is_static
                        && key_is_named(&m.key, "constructor")
                        && (matches!(m.kind, MethodKind::Get | MethodKind::Set)
                            || m.value.is_async
                            || m.value.is_generator)
                    {
                        return Err(self.err(
                            m.span,
                            "class `constructor` may not be an accessor, generator, or async method",
                        ));
                    }
                }
                ClassMember::Field(field) => {
                    if let PropertyKey::Private(name) = &field.key {
                        if &**name == "constructor" {
                            return Err(
                                self.err(field.span, "`#constructor` is not a valid private name")
                            );
                        }
                        privates.push(name.clone());
                        self.check_private_dup(
                            &mut seen,
                            name,
                            field.is_static,
                            PrivKind::Field,
                            field.span,
                        )?;
                    }
                    if key_is_named(&field.key, "constructor") {
                        return Err(
                            self.err(field.span, "a class field may not be named `constructor`")
                        );
                    }
                    if field.is_static && key_is_named(&field.key, "prototype") {
                        return Err(self.err(
                            field.span,
                            "a static class field may not be named `prototype`",
                        ));
                    }
                }
                ClassMember::StaticBlock { .. } => {}
            }
        }

        // 3. Push the private scope and walk member bodies/initializers.
        self.private_scopes.push(privates);
        let has_heritage = c.super_class.is_some();
        let result = self.class_members(c, &cls_ctx, has_heritage);
        self.private_scopes.pop();
        result
    }

    fn class_members(&mut self, c: &Class, ctx: &Ctx, has_heritage: bool) -> Result<()> {
        for member in &c.body {
            match member {
                ClassMember::Method(m) => {
                    if let PropertyKey::Computed(e) = &m.key {
                        self.expr(e, ctx)?;
                    }
                    // `super(...)` is only legal inside the constructor of a
                    // derived class.
                    let allow_super_call =
                        has_heritage && matches!(m.kind, MethodKind::Constructor);
                    self.method_function(&m.value, ctx, allow_super_call)?;
                }
                ClassMember::Field(field) => {
                    if let PropertyKey::Computed(e) = &field.key {
                        self.expr(e, ctx)?;
                    }
                    if let Some(v) = &field.value {
                        // A field initializer may use `super.prop`, `this`, and
                        // private names, but never `super(...)` or `arguments`.
                        // It is a function boundary, so `return` is not allowed.
                        let mut c2 = Ctx::function_boundary(true, false, true);
                        c2.in_function = false;
                        self.expr(v, &c2)?;
                    }
                }
                ClassMember::StaticBlock { body, .. } => {
                    // A static block is a function boundary for jumps but does not
                    // permit `return`; `arguments`/`super()` are also forbidden.
                    let mut c2 = Ctx::function_boundary(true, false, true);
                    c2.in_function = false;
                    let saved = core::mem::take(&mut self.labels);
                    // A class static block is always strict-mode code.
                    self.check_top_level_scope(body, true)?;
                    for s in body {
                        self.stmt(s, &c2)?;
                    }
                    self.labels = saved;
                }
            }
        }
        Ok(())
    }

    /// Private names share a single namespace across the whole class body
    /// (static and instance alike), so any repeat of a name is a duplicate —
    /// except a single get/set accessor pair, which must additionally agree on
    /// staticness.
    fn check_private_dup(
        &self,
        seen: &mut Vec<(Box<str>, bool, PrivKind)>,
        name: &str,
        is_static: bool,
        kind: PrivKind,
        span: Span,
    ) -> Result<()> {
        for (n, s, k) in seen.iter() {
            if &**n != name {
                continue;
            }
            // The only legal repeat is a getter paired with a setter, both with
            // matching staticness.
            let pair_ok = *s == is_static
                && matches!(
                    (*k, kind),
                    (PrivKind::Get, PrivKind::Set) | (PrivKind::Set, PrivKind::Get)
                );
            if !pair_ok {
                return Err(self.err(span, "duplicate private name in class body"));
            }
        }
        seen.push((name.into(), is_static, kind));
        Ok(())
    }

    /// Whether `name` is a private name visible in an enclosing class scope.
    fn private_in_scope(&self, name: &str) -> bool {
        self.private_scopes
            .iter()
            .any(|frame| frame.iter().any(|n| &**n == name))
    }

    // --- expressions ----------------------------------------------------

    fn expr(&mut self, e: &Expr, ctx: &Ctx) -> Result<()> {
        match e {
            Expr::Null(_)
            | Expr::Bool { .. }
            | Expr::Number { .. }
            | Expr::BigInt { .. }
            | Expr::Str { .. }
            | Expr::Regex { .. }
            | Expr::This(_) => Ok(()),
            Expr::NewTarget(span) => {
                if !ctx.allow_new_target {
                    return Err(self.err(*span, "`new.target` is only valid inside a function"));
                }
                Ok(())
            }
            Expr::Ident(id) => {
                // A class field initializer / static block has no `arguments`.
                if ctx.in_field_init && &*id.name == "arguments" {
                    return Err(self.err(
                        id.span,
                        "`arguments` is not allowed in a class field initializer",
                    ));
                }
                // In strict mode the future-reserved words (and `yield`) may not
                // appear as an `IdentifierReference`. The parser accepts the
                // spelling as an identifier (valid in sloppy code); this is where
                // the strict-mode restriction is enforced for *references* (the
                // binding-position rule lives in `check_binding_ident_name`).
                if ctx.strict && is_strict_reserved_word(&id.name) {
                    return Err(self.err(
                        id.span,
                        "a strict-mode reserved word may not be used as an identifier",
                    ));
                }
                Ok(())
            }
            Expr::Super(_) => {
                // The validity of a bare `super` keyword depends on whether the
                // enclosing function is a method, which the cover-grammar AST does
                // not record unambiguously (an object method and a
                // function-valued property are identical here). Defer this check
                // to the runtime, where the home-object is known; the unambiguous
                // `super.#private` rule is still enforced at the `Member` node.
                let _ = ctx;
                Ok(())
            }
            Expr::PrivateName(name, span) => {
                if !self.private_in_scope(name) {
                    return Err(self.err(*span, "reference to undeclared private name"));
                }
                Ok(())
            }
            Expr::Template(t) => {
                // NB: an *untagged* template literal with an invalid escape is a
                // spec parse error, but this engine surfaces it at runtime (a
                // curated conformance test pins that behavior), so it is not
                // enforced here.
                for x in &t.expressions {
                    self.expr(x, ctx)?;
                }
                Ok(())
            }
            Expr::TaggedTemplate { tag, quasi, .. } => {
                self.expr(tag, ctx)?;
                for x in &quasi.expressions {
                    self.expr(x, ctx)?;
                }
                Ok(())
            }
            Expr::Array { elements, .. } => {
                for el in elements {
                    match el {
                        crate::ast::ArrayElement::Hole => {}
                        crate::ast::ArrayElement::Item(x) | crate::ast::ArrayElement::Spread(x) => {
                            self.expr(x, ctx)?
                        }
                    }
                }
                Ok(())
            }
            Expr::Object { members, .. } => {
                for m in members {
                    match m {
                        ObjectMember::Property { key, value, .. } => {
                            if let PropertyKey::Computed(k) = key {
                                self.expr(k, ctx)?;
                            }
                            self.expr(value, ctx)?;
                        }
                        ObjectMember::Spread { value, .. } => self.expr(value, ctx)?,
                        ObjectMember::Accessor { key, value, .. } => {
                            if let PropertyKey::Computed(k) = key {
                                self.expr(k, ctx)?;
                            }
                            self.method_function(value, ctx, false)?;
                        }
                    }
                }
                Ok(())
            }
            Expr::Member {
                object, property, ..
            } => {
                if let (Expr::Super(_), PropertyKey::Private(_)) = (&**object, property) {
                    return Err(self.err(
                        e.span(),
                        "private names may not be accessed through `super`",
                    ));
                }
                // `import.meta` (desugared by the parser to a member access on the
                // reserved `import` reference) is valid only in a Module.
                if !self.is_module
                    && let Expr::Ident(id) = &**object
                    && &*id.name == "import"
                    && matches!(property, PropertyKey::Ident(n) if &**n == "meta")
                {
                    return Err(self.err(e.span(), "`import.meta` is only allowed in a module"));
                }
                self.expr(object, ctx)?;
                if let PropertyKey::Private(name) = property
                    && !self.private_in_scope(name)
                {
                    return Err(self.err(e.span(), "reference to undeclared private name"));
                }
                if let PropertyKey::Computed(k) = property {
                    self.expr(k, ctx)?;
                }
                Ok(())
            }
            Expr::Call {
                callee, arguments, ..
            } => {
                // `super(...)` is legal only in a derived-class constructor.
                if let Expr::Super(span) = &**callee
                    && !ctx.allow_super_call
                {
                    return Err(self.err(
                        *span,
                        "`super()` is only valid in a derived class constructor",
                    ));
                }
                self.expr(callee, ctx)?;
                for a in arguments {
                    match a {
                        crate::ast::Argument::Item(x) | crate::ast::Argument::Spread(x) => {
                            self.expr(x, ctx)?
                        }
                    }
                }
                Ok(())
            }
            Expr::New {
                callee, arguments, ..
            } => {
                self.expr(callee, ctx)?;
                for a in arguments {
                    match a {
                        crate::ast::Argument::Item(x) | crate::ast::Argument::Spread(x) => {
                            self.expr(x, ctx)?
                        }
                    }
                }
                Ok(())
            }
            Expr::OptChain { expr, .. } => self.expr(expr, ctx),
            Expr::Unary { op, argument, .. } => {
                if matches!(op, UnaryOp::Delete) && contains_private_ref(argument) {
                    return Err(self.err(e.span(), "`delete` of a private member is not allowed"));
                }
                self.expr(argument, ctx)
            }
            Expr::Update { argument, .. } => {
                // The operand of `++`/`--` must be a *simple* reference — an
                // identifier or member access — never a call, literal, or
                // destructuring pattern.
                if !is_simple_update_target(argument) {
                    return Err(self.err(argument.span(), "invalid operand for `++`/`--`"));
                }
                if ctx.strict {
                    self.check_not_eval_arguments(argument)?;
                }
                self.expr(argument, ctx)
            }
            Expr::Binary { .. } | Expr::Logical { .. } => {
                // Binary/logical operators chain left-associatively, so a long
                // `a + b + c + …` is a left-deep spine. Walk that spine
                // iteratively to keep stack use O(1) in the chain length (the
                // parser builds such chains in a loop, not by recursion, so they
                // can be far deeper than `MAX_PARSE_DEPTH`).
                let mut node = e;
                loop {
                    match node {
                        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
                            self.expr(right, ctx)?;
                            node = left;
                        }
                        other => break self.expr(other, ctx),
                    }
                }
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                self.expr(test, ctx)?;
                self.expr(consequent, ctx)?;
                self.expr(alternate, ctx)
            }
            Expr::Assign {
                op, target, value, ..
            } => {
                // A simple assignment `=` may target a destructuring pattern; a
                // compound assignment (`+=`, …) requires a simple reference.
                if matches!(op, crate::ast::AssignOp::Assign) {
                    if !is_valid_assign_target(target) {
                        return Err(self.err(target.span(), "invalid assignment target"));
                    }
                } else if !is_simple_update_target(target) {
                    return Err(self.err(target.span(), "invalid target for compound assignment"));
                }
                if ctx.strict {
                    self.check_not_eval_arguments(target)?;
                }
                self.expr(target, ctx)?;
                self.expr(value, ctx)
            }
            Expr::Sequence { expressions, .. } => {
                for x in expressions {
                    self.expr(x, ctx)?;
                }
                Ok(())
            }
            Expr::Function(f) => self.function(f, ctx),
            Expr::Arrow(a) => self.arrow(a, ctx),
            Expr::Class(c) => self.class(c, ctx),
            Expr::Yield { argument, .. } => {
                if let Some(a) = argument {
                    self.expr(a, ctx)?;
                }
                Ok(())
            }
            Expr::Await { argument, .. } => self.expr(argument, ctx),
        }
    }

    // --- lexical redeclaration ------------------------------------------

    /// Checks a *block* lexical scope (a `{ … }` block or catch/finally body),
    /// where a function declaration is itself a lexically-declared name.
    fn check_lexical_scope(&self, body: &[Stmt], strict: bool) -> Result<()> {
        let refs: Vec<&Stmt> = body.iter().collect();
        self.check_lexical_scope_refs(&refs, false, strict)
    }

    /// Checks a *top-level* scope (the program body, a function body, or a
    /// static block), where a top-level function declaration is **var**-scoped
    /// rather than lexical and so does not clash with a `var` of the same name.
    fn check_top_level_scope(&self, body: &[Stmt], strict: bool) -> Result<()> {
        let refs: Vec<&Stmt> = body.iter().collect();
        self.check_lexical_scope_refs(&refs, true, strict)
    }

    /// Enforces that a scope's lexically-declared names are unique and do not
    /// collide with var-declared names hoisted into the same scope.
    ///
    /// Per Annex B.3.3 (block) / B.3.2 (`switch` case block), in *sloppy* mode a
    /// pair of duplicate `LexicallyDeclaredNames` is tolerated when **both** are
    /// bound by `FunctionDeclaration`s — block-level functions get web-compat
    /// var-style hoisting. Duplicates involving a `let`/`const`/`class`, and any
    /// duplicate at all in strict mode, remain early errors.
    fn check_lexical_scope_refs(
        &self,
        body: &[&Stmt],
        top_level: bool,
        strict: bool,
    ) -> Result<()> {
        let mut lexical: Vec<(Box<str>, Span, bool)> = Vec::new();
        let mut vars: Vec<Box<str>> = Vec::new();

        for stmt in body {
            collect_top_level_decls(stmt, &mut lexical, &mut vars, top_level);
        }
        for i in 0..lexical.len() {
            for j in (i + 1)..lexical.len() {
                if lexical[i].0 == lexical[j].0 {
                    // The sole tolerated duplicate: two function declarations in
                    // a sloppy block/switch scope (Annex B).
                    let both_functions = lexical[i].2 && lexical[j].2;
                    if both_functions && !strict {
                        continue;
                    }
                    return Err(
                        self.err(lexical[j].1, "duplicate lexical declaration in this scope")
                    );
                }
            }
        }
        for (name, span, _) in &lexical {
            if vars.iter().any(|v| v == name) {
                return Err(self.err(
                    *span,
                    "a lexical declaration conflicts with a `var` of the same name",
                ));
            }
        }
        Ok(())
    }
}

/// Records the top-level lexical and var declarations of a single statement for
/// the redeclaration check. `var` names are collected through nested
/// non-function blocks (hoisting); lexical names are only the immediate
/// declarations of this scope.
fn collect_top_level_decls(
    stmt: &Stmt,
    lexical: &mut Vec<(Box<str>, Span, bool)>,
    vars: &mut Vec<Box<str>>,
    top_level: bool,
) {
    match stmt {
        Stmt::Var(decl) => {
            if decl.kind == VarDeclKind::Var {
                push_var_names(decl, vars);
            } else {
                for d in &decl.declarations {
                    let mut names = Vec::new();
                    collect_bound_names(&d.target, &mut names);
                    for (n, span) in names {
                        lexical.push((n.into(), span, false));
                    }
                }
            }
        }
        Stmt::Function(f) => {
            if let Some(id) = &f.id {
                // At program / function-body top level a function declaration is
                // var-scoped (its name is a VarDeclaredName); inside a block or
                // switch it is a lexically-declared name. The flag marks the
                // binding as eligible for the Annex B duplicate exception, which
                // covers *plain* function declarations only — generator and
                // async functions do not get block-level function hoisting, so a
                // duplicate involving one of them remains an early error.
                let annexb_fn = !f.is_generator && !f.is_async;
                if top_level {
                    vars.push(id.name.clone());
                } else {
                    lexical.push((id.name.clone(), id.span, annexb_fn));
                }
            }
        }
        Stmt::Class(c) => {
            if let Some(id) = &c.id {
                lexical.push((id.name.clone(), id.span, false));
            }
        }
        Stmt::Labeled { body, .. } => collect_top_level_decls(body, lexical, vars, top_level),
        // `var` hoists out of nested non-function statements.
        other => collect_vars_only(other, vars),
    }
}

/// Derives the context for a loop body: it is both a `break` and a `continue`
/// target.
fn loop_ctx(ctx: &Ctx) -> Ctx {
    let mut c = *ctx;
    c.in_breakable = true;
    c.in_iteration = true;
    c
}

/// Whether a (possibly label-nested) statement is an iteration statement, so a
/// label applied to it makes `continue label` valid.
fn labels_an_iteration(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::For { .. }
        | Stmt::ForIn { .. }
        | Stmt::ForOf { .. }
        | Stmt::While { .. }
        | Stmt::DoWhile { .. } => true,
        Stmt::Labeled { body, .. } => labels_an_iteration(body),
        _ => false,
    }
}

/// Pushes the names bound by a `var` declaration.
fn push_var_names(decl: &VarDecl, vars: &mut Vec<Box<str>>) {
    for d in &decl.declarations {
        let mut names = Vec::new();
        collect_bound_names(&d.target, &mut names);
        for (n, _) in names {
            vars.push(n.into());
        }
    }
}

/// Collects only `var`-declared names hoisted through non-function nested
/// statements (a function creates a new var scope and stops the recursion).
fn collect_vars_only(stmt: &Stmt, vars: &mut Vec<Box<str>>) {
    match stmt {
        Stmt::Var(decl) if decl.kind == VarDeclKind::Var => push_var_names(decl, vars),
        Stmt::Block { body, .. } => {
            for s in body {
                collect_vars_only(s, vars);
            }
        }
        Stmt::If {
            consequent,
            alternate,
            ..
        } => {
            collect_vars_only(consequent, vars);
            if let Some(a) = alternate {
                collect_vars_only(a, vars);
            }
        }
        Stmt::For { init, body, .. } => {
            if let Some(crate::ast::ForInit::Var(d)) = init
                && d.kind == VarDeclKind::Var
            {
                push_var_names(d, vars);
            }
            collect_vars_only(body, vars);
        }
        Stmt::ForIn { left, body, .. } | Stmt::ForOf { left, body, .. } => {
            if let ForLeft::Decl {
                kind: VarDeclKind::Var,
                target,
                ..
            } = left
            {
                let mut names = Vec::new();
                collect_bound_names(target, &mut names);
                for (n, _) in names {
                    vars.push(n.into());
                }
            }
            collect_vars_only(body, vars);
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::With { body, .. } => {
            collect_vars_only(body, vars);
        }
        Stmt::Labeled { body, .. } => collect_vars_only(body, vars),
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            for s in block {
                collect_vars_only(s, vars);
            }
            if let Some(h) = handler {
                for s in &h.body {
                    collect_vars_only(s, vars);
                }
            }
            if let Some(f) = finalizer {
                for s in f {
                    collect_vars_only(s, vars);
                }
            }
        }
        Stmt::Switch { cases, .. } => {
            for case in cases {
                for s in &case.body {
                    collect_vars_only(s, vars);
                }
            }
        }
        _ => {}
    }
}

/// Collects the identifier names (with spans) bound by a [`BindingTarget`].
/// Which of `yield`/`await` was found in a parameter list.
#[derive(Clone, Copy)]
enum YieldOrAwait {
    Yield,
    Await,
}

/// Finds the first forbidden `yield`/`await` *expression* in a parameter's
/// default initializer or pattern computed keys, without descending into nested
/// functions/arrows (which introduce their own parameter context).
fn first_param_yield_await(
    p: &Param,
    is_generator: bool,
    is_async: bool,
) -> Option<(Span, YieldOrAwait)> {
    let mut found = None;
    if let Some(d) = &p.default {
        find_yield_await(d, is_generator, is_async, &mut found);
    }
    bound_target_yield_await(&p.target, is_generator, is_async, &mut found);
    found
}

/// Scans a binding target's computed keys / nested defaults for `yield`/`await`.
fn bound_target_yield_await(
    target: &BindingTarget,
    is_generator: bool,
    is_async: bool,
    found: &mut Option<(Span, YieldOrAwait)>,
) {
    match target {
        BindingTarget::Ident(_) => {}
        BindingTarget::Array(p) => {
            for el in &p.elements {
                match el {
                    crate::ast::ArrayPatternElement::Hole => {}
                    crate::ast::ArrayPatternElement::Item {
                        target, default, ..
                    } => {
                        bound_target_yield_await(target, is_generator, is_async, found);
                        if let Some(d) = default {
                            find_yield_await(d, is_generator, is_async, found);
                        }
                    }
                    crate::ast::ArrayPatternElement::Rest { target, .. } => {
                        bound_target_yield_await(target, is_generator, is_async, found);
                    }
                }
            }
        }
        BindingTarget::Object(p) => {
            for prop in &p.properties {
                if let PropertyKey::Computed(e) = &prop.key {
                    find_yield_await(e, is_generator, is_async, found);
                }
                bound_target_yield_await(&prop.value, is_generator, is_async, found);
                if let Some(d) = &prop.default {
                    find_yield_await(d, is_generator, is_async, found);
                }
            }
            if let Some(r) = &p.rest {
                bound_target_yield_await(r, is_generator, is_async, found);
            }
        }
    }
}

/// Recursively records the first `yield`/`await` expression in `e`, stopping at
/// nested function/arrow/class boundaries (they have their own contexts).
fn find_yield_await(
    e: &Expr,
    is_generator: bool,
    is_async: bool,
    found: &mut Option<(Span, YieldOrAwait)>,
) {
    if found.is_some() {
        return;
    }
    match e {
        Expr::Yield { span, .. } if is_generator => {
            *found = Some((*span, YieldOrAwait::Yield));
        }
        Expr::Await { span, .. } if is_async => {
            *found = Some((*span, YieldOrAwait::Await));
        }
        // Do not descend into a nested function/arrow/class: it establishes its
        // own parameter context, so a `yield`/`await` inside it is governed by
        // that context, not this parameter list's.
        Expr::Function(_) | Expr::Arrow(_) | Expr::Class(_) => {}
        // Leaves with no sub-expressions.
        Expr::Null(_)
        | Expr::Bool { .. }
        | Expr::Number { .. }
        | Expr::BigInt { .. }
        | Expr::Str { .. }
        | Expr::Regex { .. }
        | Expr::Ident(_)
        | Expr::PrivateName(..)
        | Expr::This(_)
        | Expr::Super(_)
        | Expr::NewTarget(_) => {}
        Expr::Yield { argument, .. } => {
            // Reached only when `!is_generator` (the generator arm matched
            // above); still scan the operand for an `await` in an async context.
            if let Some(a) = argument {
                find_yield_await(a, is_generator, is_async, found);
            }
        }
        Expr::Await { argument, .. } => {
            find_yield_await(argument, is_generator, is_async, found);
        }
        Expr::Template(t) => {
            for x in &t.expressions {
                find_yield_await(x, is_generator, is_async, found);
            }
        }
        Expr::TaggedTemplate { tag, quasi, .. } => {
            find_yield_await(tag, is_generator, is_async, found);
            for x in &quasi.expressions {
                find_yield_await(x, is_generator, is_async, found);
            }
        }
        Expr::Array { elements, .. } => {
            for el in elements {
                match el {
                    crate::ast::ArrayElement::Hole => {}
                    crate::ast::ArrayElement::Item(x) | crate::ast::ArrayElement::Spread(x) => {
                        find_yield_await(x, is_generator, is_async, found);
                    }
                }
            }
        }
        Expr::Object { members, .. } => {
            for m in members {
                match m {
                    crate::ast::ObjectMember::Property { key, value, .. } => {
                        if let PropertyKey::Computed(k) = key {
                            find_yield_await(k, is_generator, is_async, found);
                        }
                        find_yield_await(value, is_generator, is_async, found);
                    }
                    crate::ast::ObjectMember::Spread { value, .. } => {
                        find_yield_await(value, is_generator, is_async, found);
                    }
                    // An accessor's value is a function — its own context.
                    crate::ast::ObjectMember::Accessor { .. } => {}
                }
            }
        }
        Expr::Member {
            object, property, ..
        } => {
            find_yield_await(object, is_generator, is_async, found);
            if let PropertyKey::Computed(k) = property {
                find_yield_await(k, is_generator, is_async, found);
            }
        }
        Expr::Call {
            callee, arguments, ..
        }
        | Expr::New {
            callee, arguments, ..
        } => {
            find_yield_await(callee, is_generator, is_async, found);
            for a in arguments {
                match a {
                    crate::ast::Argument::Item(x) | crate::ast::Argument::Spread(x) => {
                        find_yield_await(x, is_generator, is_async, found);
                    }
                }
            }
        }
        Expr::OptChain { expr, .. } => {
            find_yield_await(expr, is_generator, is_async, found);
        }
        Expr::Unary { argument, .. } | Expr::Update { argument, .. } => {
            find_yield_await(argument, is_generator, is_async, found);
        }
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            find_yield_await(left, is_generator, is_async, found);
            find_yield_await(right, is_generator, is_async, found);
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => {
            find_yield_await(test, is_generator, is_async, found);
            find_yield_await(consequent, is_generator, is_async, found);
            find_yield_await(alternate, is_generator, is_async, found);
        }
        Expr::Assign { target, value, .. } => {
            find_yield_await(target, is_generator, is_async, found);
            find_yield_await(value, is_generator, is_async, found);
        }
        Expr::Sequence { expressions, .. } => {
            for x in expressions {
                find_yield_await(x, is_generator, is_async, found);
            }
        }
    }
}

/// Whether `name` is a strict-mode future-reserved word — a word the parser
/// accepts as an identifier (valid in sloppy code) but which may not be used as
/// a `BindingIdentifier` or assignment target in strict mode.
fn is_strict_reserved_word(name: &str) -> bool {
    matches!(
        name,
        "implements"
            | "interface"
            | "let"
            | "package"
            | "private"
            | "protected"
            | "public"
            | "static"
            | "yield"
    )
}

fn collect_bound_names(target: &BindingTarget, out: &mut Vec<(String, Span)>) {
    match target {
        BindingTarget::Ident(id) => out.push((id.name.to_string(), id.span)),
        BindingTarget::Array(p) => {
            for el in &p.elements {
                match el {
                    crate::ast::ArrayPatternElement::Hole => {}
                    crate::ast::ArrayPatternElement::Item { target, .. }
                    | crate::ast::ArrayPatternElement::Rest { target, .. } => {
                        collect_bound_names(target, out)
                    }
                }
            }
        }
        BindingTarget::Object(p) => {
            for prop in &p.properties {
                collect_bound_names(&prop.value, out);
            }
            if let Some(r) = &p.rest {
                collect_bound_names(r, out);
            }
        }
    }
}

/// Whether a property key is a (non-computed) name equal to `name`.
fn key_is_named(key: &PropertyKey, name: &str) -> bool {
    match key {
        PropertyKey::Ident(n) | PropertyKey::Str(n) => &**n == name,
        _ => false,
    }
}

/// Whether the operand of a `delete` directly references a private member
/// (`a.#x`, `a?.#x`, or such inside a chain boundary).
fn contains_private_ref(e: &Expr) -> bool {
    match e {
        Expr::Member { property, .. } => matches!(property, PropertyKey::Private(_)),
        Expr::OptChain { expr, .. } => contains_private_ref(expr),
        _ => false,
    }
}

/// Whether `e` is a *simple* reference usable as the operand of `++`/`--` or the
/// target of a compound assignment: an identifier or a member access. (A private
/// member access `obj.#x` is also a valid reference here.)
fn is_simple_update_target(e: &Expr) -> bool {
    matches!(e, Expr::Ident(_) | Expr::Member { .. })
}

/// Whether `e` is a valid target for a *simple* (`=`) assignment. This permits
/// destructuring patterns (with `= default` inside them) but rejects a bare
/// parenthesized assignment such as `(x = y) = 1` — represented here as a
/// top-level [`Expr::Assign`] target.
fn is_valid_assign_target(e: &Expr) -> bool {
    match e {
        Expr::Ident(_) | Expr::Member { .. } => true,
        Expr::Array { elements, .. } => {
            for (i, el) in elements.iter().enumerate() {
                match el {
                    crate::ast::ArrayElement::Hole => {}
                    crate::ast::ArrayElement::Item(x) => {
                        if !is_pattern_element(x) {
                            return false;
                        }
                    }
                    crate::ast::ArrayElement::Spread(x) => {
                        // A rest element must be the final element, target a plain
                        // reference (no default), and not be followed by elision.
                        if i + 1 != elements.len() || !is_valid_assign_target(x) {
                            return false;
                        }
                    }
                }
            }
            true
        }
        Expr::Object { members, .. } => {
            for (i, m) in members.iter().enumerate() {
                match m {
                    ObjectMember::Property { value, .. } => {
                        if !is_pattern_element(value) {
                            return false;
                        }
                    }
                    ObjectMember::Spread { value, .. } => {
                        // An object rest must be last and a plain reference.
                        if i + 1 != members.len() || !is_valid_assign_target(value) {
                            return false;
                        }
                    }
                    ObjectMember::Accessor { .. } => return false,
                }
            }
            true
        }
        _ => false,
    }
}

/// An element inside a destructuring assignment pattern: a valid target, or a
/// target with a `= default`.
fn is_pattern_element(e: &Expr) -> bool {
    match e {
        Expr::Assign {
            op: crate::ast::AssignOp::Assign,
            target,
            ..
        } => is_valid_assign_target(target),
        _ => is_valid_assign_target(e),
    }
}

impl Validator {
    /// In strict mode, no `eval` / `arguments` identifier may appear as an
    /// assignment / update target — including nested inside a destructuring
    /// assignment pattern.
    fn check_not_eval_arguments(&self, target: &Expr) -> Result<()> {
        match target {
            Expr::Ident(id) if &*id.name == "eval" || &*id.name == "arguments" => Err(self.err(
                id.span,
                "`eval` and `arguments` may not be assigned in strict mode",
            )),
            Expr::Array { elements, .. } => {
                for el in elements {
                    match el {
                        crate::ast::ArrayElement::Hole => {}
                        crate::ast::ArrayElement::Item(x) | crate::ast::ArrayElement::Spread(x) => {
                            self.check_not_eval_arguments(x)?;
                        }
                    }
                }
                Ok(())
            }
            Expr::Object { members, .. } => {
                for m in members {
                    match m {
                        ObjectMember::Property { value, .. }
                        | ObjectMember::Spread { value, .. } => {
                            self.check_not_eval_arguments(value)?;
                        }
                        ObjectMember::Accessor { .. } => {}
                    }
                }
                Ok(())
            }
            // A `= default` inside a pattern: check the target side.
            Expr::Assign {
                op: crate::ast::AssignOp::Assign,
                target,
                ..
            } => self.check_not_eval_arguments(target),
            _ => Ok(()),
        }
    }
}

use alloc::string::ToString;
