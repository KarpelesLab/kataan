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
    validate_program_with(program, false, false, false, false, &[], false)
}

/// Validates an eval program that inherits a `super` context from the calling
/// code (a *direct* eval inside a method / accessor / constructor / class field
/// initializer / static block). `super.prop` / `super[…]` is permitted when
/// `allow_super_property`, and `super(…)` when `allow_super_call`; otherwise this
/// behaves exactly like [`validate_program`].
pub(crate) fn validate_program_with(
    program: &Program,
    allow_super_property: bool,
    allow_super_call: bool,
    allow_new_target: bool,
    inherited_strict: bool,
    // Private names visible at a *direct-eval* call site (from the enclosing class
    // scope chain). The eval body may reference these `#names` even though it does
    // not declare them itself, so seed the validator's private scope with them.
    // Empty for ordinary program / indirect-eval validation.
    outer_private_names: &[Box<str>],
    // Whether this eval body runs directly inside a class field initializer /
    // static block. In that position an early error is raised if the eval'd
    // StatementList references `arguments` (ContainsArguments — the special
    // "Eval Inside Initializer" static semantics).
    in_field_initializer: bool,
) -> Result<()> {
    // A direct eval inside strict code is itself strict code even without its own
    // `"use strict"` directive (`inherited_strict`); the early-error checks below
    // (strict-reserved words, `with`, octal, duplicate params, …) must run.
    let strict = inherited_strict
        || program.source_type == SourceType::Module
        || body_is_strict(&program.body);
    // Seed the private scope with the names visible at the eval call site so that
    // a private reference (`this.#x`) in the eval body validates.
    let seeded_privates: Vec<Box<str>> = outer_private_names.to_vec();
    let mut private_scopes = Vec::new();
    if !seeded_privates.is_empty() {
        private_scopes.push(seeded_privates);
    }
    let mut v = Validator {
        private_scopes,
        labels: Vec::new(),
        is_module: program.source_type == SourceType::Module,
        in_assign_target: false,
    };
    let ctx = Ctx {
        allow_super_property,
        allow_super_call,
        allow_new_target,
        // A direct eval inside a field initializer / static block inherits the
        // "no `arguments`" restriction (the ContainsArguments early error). The
        // normal walk raises a SyntaxError on any `arguments` reference not
        // shielded by a nested non-arrow function boundary.
        in_field_init: in_field_initializer,
        ..Ctx::top(strict)
    };
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
    if v.is_module {
        v.check_module_exports(&program.body)?;
    }
    Ok(())
}

/// Whether a directive prologue in `body` contains a literal `"use strict"`.
fn body_is_strict(body: &[Stmt]) -> bool {
    for stmt in body {
        if let Stmt::Expr { expression, .. } = stmt {
            if let Expr::Str { value, span, .. } = &**expression {
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
    /// Whether a `super.prop` / `super[expr]` *property* reference is permitted —
    /// true inside any method definition (object/class method, accessor,
    /// constructor), a class field initializer, or a static block; false inside a
    /// plain function declaration/expression and in top-level script/module code.
    /// An arrow inherits this from its enclosing scope.
    allow_super_property: bool,
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
    /// Whether the current code is directly inside a class static initialization
    /// block (with no intervening function boundary). The block is `[~Await]`, so
    /// an `await` expression there is an early Syntax Error.
    in_static_block: bool,
}

impl Ctx {
    /// A fresh non-strict top-level context.
    fn top(strict: bool) -> Self {
        Ctx {
            strict,
            allow_super_call: false,
            allow_super_property: false,
            in_field_init: false,
            in_function: false,
            allow_new_target: false,
            in_breakable: false,
            in_iteration: false,
            in_static_block: false,
        }
    }

    /// A fresh context at a function boundary (the jump/`return` state resets;
    /// `super`/field-init are passed in by the caller). `new.target` is always
    /// available inside a function-like body.
    ///
    /// `allow_super_property` is true for a method/accessor/constructor body, a
    /// class field initializer, and a static block — i.e. wherever a `super.prop`
    /// reference has a home object — and false for a plain function.
    fn function_boundary(
        strict: bool,
        allow_super_call: bool,
        allow_super_property: bool,
        in_field_init: bool,
    ) -> Self {
        Ctx {
            strict,
            allow_super_call,
            allow_super_property,
            in_field_init,
            in_function: true,
            allow_new_target: true,
            in_breakable: false,
            in_iteration: false,
            in_static_block: false,
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
    /// Whether the object/array literal currently being walked is in a
    /// destructuring-assignment-target position (the LHS of `=`, a `for-in/of`
    /// target). In that position a `CoverInitializedName` (`{ a = 1 }`) is a
    /// valid pattern element; everywhere else it is an early Syntax Error. The
    /// flag is consumed (reset to `false`) on entry to each value sub-expression
    /// so that a nested object in a property *default* is treated as a value.
    in_assign_target: bool,
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
                        crate::ast::ForInit::Var(d) => {
                            self.var_decl(d, ctx)?;
                            // A lexical (`let`/`const`/`using`) head declaration
                            // may not be redeclared by a `var` in the loop body.
                            if d.kind != VarDeclKind::Var {
                                let mut bound = Vec::new();
                                for decl in &d.declarations {
                                    collect_bound_names(&decl.target, &mut bound);
                                }
                                // The head's whole LexicalDeclaration is one
                                // declaration list, so a name bound twice across
                                // it — `for (let [z, z] = …; …)`, and equally
                                // `for (let a, [a] = …; …)` — is an early error.
                                self.check_no_dup_bound_names(&bound)?;
                                self.check_no_var_redeclare(&bound, body)?;
                            }
                        }
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
                // Annex B.3.5 (`for (var x = <expr> in obj)`) is a sloppy-mode-only
                // web-compatibility extension; the grammar proper has no
                // initializer here, so strict code must reject it.
                if let Stmt::ForIn {
                    annexb_init: true, ..
                } = stmt
                    && ctx.strict
                {
                    return Err(self.err(
                        stmt.span(),
                        "a `for-in` head may not have an initializer in strict mode",
                    ));
                }
                self.for_left(left, ctx)?;
                // It is a Syntax Error if any element of the BoundNames of a
                // `let`/`const`/`using`/`await using` ForDeclaration also occurs
                // in the VarDeclaredNames of the loop body. (`var` heads are
                // exempt — they share the same var scope.) The BoundNames must
                // also be distinct (no `for (let [x, x] of …)`).
                if let ForLeft::Decl { kind, target, .. } = left
                    && *kind != VarDeclKind::Var
                {
                    let mut bound = Vec::new();
                    collect_bound_names(target, &mut bound);
                    self.check_no_dup_bound_names(&bound)?;
                    self.check_no_var_redeclare(&bound, body)?;
                }
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
                        self.check_catch_param(p, &h.body)?;
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
                // The body of a `with` is a `Statement`; unlike an `if` branch,
                // Annex B does not allow a (sloppy) `FunctionDeclaration` here, so
                // a bare or labelled function declaration is rejected.
                self.reject_decl_substatement(
                    body, ctx, /* allow_sloppy_fn */ false, /* allow_labelled_fn */ false,
                )?;
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
        // An `if`/`else` branch permits a bare sloppy `FunctionDeclaration`
        // (Annex B.3.4) but never a *labelled* one: `if (x) L: function f(){}`
        // is a Syntax Error in all modes (`IsLabelledFunction` early error).
        self.reject_decl_substatement(
            body, ctx, /* allow_sloppy_fn */ true, /* allow_labelled_fn */ false,
        )
    }

    /// The body of a `for`/`while`/`do-while` loop may not be *any* declaration
    /// — not even a plain `function` (Annex B does not apply to loop bodies).
    fn check_loop_body(&self, body: &Stmt) -> Result<()> {
        // Loop bodies forbid even sloppy function declarations.
        self.reject_decl_substatement(
            body,
            &Ctx::top(true),
            /* allow_sloppy_fn */ false,
            /* allow_labelled_fn */ false,
        )
    }

    /// The body of a labeled statement may not be a `let`/`const`/class
    /// declaration or a generator/async function; a plain `function`
    /// declaration is allowed in sloppy mode only (Annex B).
    fn check_labeled_body(&self, body: &Stmt, ctx: &Ctx) -> Result<()> {
        // `LabelledItem : FunctionDeclaration` is allowed only for a non-async,
        // non-generator function in sloppy mode. Because a labelled statement at
        // a `StatementListItem` position may nest further labels around such a
        // function (`label1: label2: function f(){}`, Annex B.3.1), a nested
        // labelled function is permitted here in sloppy mode.
        self.reject_decl_substatement(
            body, ctx, /* allow_sloppy_fn */ true, /* allow_labelled_fn */ true,
        )
    }

    /// Shared core: rejects a declaration found in single-statement position.
    ///
    /// `allow_sloppy_fn` permits a bare `FunctionDeclaration` in sloppy mode
    /// (Annex B). `allow_labelled_fn` additionally permits a *labelled*
    /// `FunctionDeclaration` in sloppy mode — true only for the body of a
    /// labelled statement that is itself at a `StatementListItem` position, never
    /// for an `if`/loop/`with` body (where `IsLabelledFunction` is an early
    /// error regardless of language mode).
    fn reject_decl_substatement(
        &self,
        body: &Stmt,
        ctx: &Ctx,
        allow_sloppy_fn: bool,
        allow_labelled_fn: bool,
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
            // A `LabelledStatement` is itself a `Statement`, so it is syntactically
            // valid in single-statement position — but its (possibly nested)
            // `LabelledItem` may not be a `FunctionDeclaration` unless the Annex B
            // sloppy allowance applies. That allowance applies only at a
            // `StatementListItem` position (here surfaced via `allow_labelled_fn`),
            // not as the body of an `if`/loop/`with` substatement, where a labelled
            // function is an early error in all language modes.
            Stmt::Labeled { body, .. } => {
                if allow_labelled_fn && !ctx.strict {
                    // Recurse so a nested labelled function is checked under the
                    // same rules (and a non-function labelled body is unaffected).
                    self.reject_decl_substatement(body, ctx, allow_sloppy_fn, allow_labelled_fn)
                } else if labels_a_function(body) {
                    Err(self.err(
                        body.span(),
                        "a labelled function declaration may not appear in \
                         single-statement position",
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
                if !is_valid_assign_target(e) && !(e.is_web_compat_call_target() && !ctx.strict) {
                    // AnnexB web-compat allows a CallExpression for-in/of LHS in
                    // sloppy code (a runtime ReferenceError); it stays a SyntaxError
                    // in strict mode and for every other invalid target.
                    return Err(self.err(e.span(), "invalid for-in/of assignment target"));
                }
                if ctx.strict {
                    self.check_not_eval_arguments(e)?;
                }
                // A `for-in`/`for-of` target may be a destructuring pattern, where
                // a `CoverInitializedName` is valid.
                self.in_assign_target = true;
                self.expr(e, ctx)
            }
        }
    }

    /// It is a Syntax Error if any element of `bound` (the BoundNames of a
    /// lexical for-head declaration) also occurs in the VarDeclaredNames of the
    /// loop body.
    fn check_no_var_redeclare(&self, bound: &[(String, Span)], body: &Stmt) -> Result<()> {
        if bound.is_empty() {
            return Ok(());
        }
        let mut vars = Vec::new();
        collect_vars_only(body, &mut vars);
        for (name, span) in bound {
            if vars.iter().any(|v| v.as_ref() == name.as_str()) {
                return Err(self.err(
                    *span,
                    "a lexical for-head binding may not be redeclared by a `var` \
                     in the loop body",
                ));
            }
        }
        Ok(())
    }

    /// It is a Syntax Error if the BoundNames of a for-head ForDeclaration
    /// contain any duplicate entries (e.g. `for (let [x, x] of …)`).
    fn check_no_dup_bound_names(&self, bound: &[(String, Span)]) -> Result<()> {
        for (i, (name, _)) in bound.iter().enumerate() {
            if bound[..i].iter().any(|(n, _)| n == name) {
                return Err(self.err(
                    bound[i].1,
                    "a for-in/of binding list may not contain a duplicate name",
                ));
            }
        }
        Ok(())
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
        // A `FunctionDeclaration`/`FunctionExpression` (incl. generator/async)
        // uses `FormalParameters` — duplicates allowed in sloppy simple lists.
        self.check_params(&f.params, strict, /* unique_required */ false, &f.body)?;
        // A regular function establishes its own `arguments`/`super` bindings and
        // a fresh label/jump environment. A plain function has no home object, so
        // `super.prop` is not permitted in its body or parameter list.
        let c = Ctx::function_boundary(strict, false, /* super_property */ false, false);
        let saved_labels = core::mem::take(&mut self.labels);
        for p in &f.params {
            self.param(p, &c)?;
        }
        self.check_top_level_scope(&f.body, strict)?;
        self.check_params_vs_body_lexical(&f.params, &f.body)?;
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
        // A `MethodDefinition` uses `UniqueFormalParameters`: duplicate parameter
        // names are always a Syntax Error.
        self.check_params(&f.params, strict, /* unique_required */ true, &f.body)?;
        // A method has a home object, so `super.prop` is permitted.
        let c = Ctx::function_boundary(
            strict,
            allow_super_call,
            /* super_property */ true,
            false,
        );
        let saved_labels = core::mem::take(&mut self.labels);
        for p in &f.params {
            self.param(p, &c)?;
        }
        self.check_top_level_scope(&f.body, strict)?;
        self.check_params_vs_body_lexical(&f.params, &f.body)?;
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
        // An arrow's parameters may not contain a `yield` or an `await`
        // expression, regardless of the arrow's own kind: a non-async arrow
        // nested in a generator (`function* g(){ (x = yield) => {}; }`) and an
        // async arrow (`async (x = await p) => {}`) are both early Syntax Errors —
        // the surrounding `[?Yield]`/`[+Await]` context makes the keyword an
        // expression, which the arrow-head cover grammar forbids. Detect both by
        // passing `is_generator`/`is_async` as `true` to the scanner. (The scanner
        // does not descend into further nested function/arrow boundaries, so an
        // inner function's own params are unaffected.)
        self.check_param_yield_await(
            &a.params, /* detect_yield */ true, /* detect_await */ true,
        )?;
        // An `ArrowFunction` uses `UniqueFormalParameters` (`CoverParenthesized…`
        // has no duplicates): duplicate parameter names are always a Syntax Error.
        self.check_params(
            &a.params, strict, /* unique_required */ true, body_slice,
        )?;
        // An arrow inherits `super`, `this`, and `arguments` (and thus the
        // field-initializer restrictions) from its enclosing scope, but is a
        // function boundary for `return` and the label/jump environment.
        let c = Ctx {
            strict,
            allow_super_call: ctx.allow_super_call,
            // An arrow has no home object of its own; `super.prop` is valid only
            // if the enclosing scope provides one.
            allow_super_property: ctx.allow_super_property,
            in_field_init: ctx.in_field_init,
            in_function: true,
            // `new.target` is inherited from the enclosing scope: an arrow has no
            // `new.target` of its own, so it is only valid if the surrounding
            // function provides one.
            allow_new_target: ctx.allow_new_target,
            in_breakable: false,
            in_iteration: false,
            // A non-async arrow inside a static block stays `[~Await]`; an async
            // arrow establishes its own await context.
            in_static_block: ctx.in_static_block && !a.is_async,
        };
        let saved_labels = core::mem::take(&mut self.labels);
        for p in &a.params {
            self.param(p, &c)?;
        }
        match &a.body {
            ArrowBody::Block(body) => {
                self.check_top_level_scope(body, strict)?;
                self.check_params_vs_body_lexical(&a.params, body)?;
                for s in body {
                    self.stmt(s, &c)?;
                }
            }
            ArrowBody::Expr(e) => self.expr(e, &c)?,
        }
        self.labels = saved_labels;
        Ok(())
    }

    /// Rejects a `yield`/`await` *expression* in a parameter list where the
    /// surrounding production forbids it: a generator's params (`detect_yield`),
    /// an async function's params (`detect_await`), and an arrow's params (both —
    /// the arrow-head cover grammar admits neither). The parser parses params in
    /// the enclosing context, so a `yield`/`await` *binding name* is already
    /// rejected there; this catches the *expression* forms in defaults and
    /// computed keys (e.g. `function* g(x = yield) {}`, `(x = await p) => {}`).
    fn check_param_yield_await(
        &self,
        params: &[Param],
        detect_yield: bool,
        detect_await: bool,
    ) -> Result<()> {
        if !detect_yield && !detect_await {
            return Ok(());
        }
        for p in params {
            if let Some((span, kind)) = first_param_yield_await(p, detect_yield, detect_await) {
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

    /// A getter must declare no parameters; a setter must declare exactly one,
    /// which may not be a rest parameter.
    fn check_accessor_arity(&self, is_getter: bool, f: &Function) -> Result<()> {
        if is_getter {
            if !f.params.is_empty() {
                return Err(self.err(f.span, "a getter may not declare any parameters"));
            }
        } else if f.params.len() != 1 || f.params[0].rest {
            return Err(self.err(
                f.span,
                "a setter must declare exactly one (non-rest) parameter",
            ));
        }
        Ok(())
    }

    /// Parameter-list early errors:
    /// - a name may not repeat in a strict-mode function, a method/arrow
    ///   (`unique_required`), or any function with a non-simple parameter list
    ///   (defaults, destructuring, or a rest element);
    /// - a function whose parameter list is non-simple may not contain a
    ///   `"use strict"` directive in its body.
    fn check_params(
        &self,
        params: &[Param],
        strict: bool,
        unique_required: bool,
        body: &[Stmt],
    ) -> Result<()> {
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
        // A method or arrow has `UniqueFormalParameters`, so duplicate names are
        // a Syntax Error regardless of strictness or parameter simplicity. A
        // plain function/generator/async-function uses `FormalParameters`, which
        // only forbids duplicates when strict or non-simple.
        if simple && !strict && !unique_required {
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

    fn class(&mut self, c: &Class, ctx: &Ctx) -> Result<()> {
        // Class bodies are always strict, so the class name is a strict binding.
        if let Some(id) = &c.id {
            self.check_binding_ident_name(&id.name, id.span)?;
        }
        // Heritage and computed member keys are evaluated in the *enclosing*
        // scope, so they inherit the `arguments`-forbidden state of a containing
        // class field initializer / static block. (Per `ContainsArguments`, a
        // computed key like `[arguments]` in a class nested inside a static block
        // is an early error, even though the method *bodies* — function
        // boundaries — get their own `arguments`.)
        let mut cls_ctx = Ctx::top(true);
        cls_ctx.in_field_init = ctx.in_field_init;

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
                    // A get/set accessor has a fixed parameter arity.
                    match m.kind {
                        MethodKind::Get => self.check_accessor_arity(true, &m.value)?,
                        MethodKind::Set => self.check_accessor_arity(false, &m.value)?,
                        _ => {}
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
                        let mut c2 = Ctx::function_boundary(
                            true, false, /* super_property */ true, true,
                        );
                        c2.in_function = false;
                        self.expr(v, &c2)?;
                    }
                }
                ClassMember::StaticBlock { body, .. } => {
                    // A static block is a function boundary for jumps but does not
                    // permit `return`; `arguments`/`super()` are also forbidden.
                    // It is `[~Await]`, so an `await` expression is an early error.
                    let mut c2 =
                        Ctx::function_boundary(true, false, /* super_property */ true, true);
                    c2.in_function = false;
                    c2.in_static_block = true;
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
        // The destructuring-target flag applies only to the object/array literal
        // it was set for. Consume it here so that — by default — any expression
        // is validated as a value. The `Object`/`Array` arms read it (via
        // `take_assign_target`) and re-propagate it to their sub-*patterns* only.
        let in_assign_target = core::mem::take(&mut self.in_assign_target);
        match e {
            Expr::Number {
                legacy_octal: true,
                span,
                ..
            } if ctx.strict => Err(self.err(
                *span,
                "legacy octal / non-octal-decimal literals are not allowed in strict mode",
            )),
            // A string literal whose source uses a legacy octal escape (`\1`–
            // `\7`, `\00`, …) or a non-octal decimal escape (`\8` / `\9`) is an
            // early Syntax Error in strict-mode code (Annex B grants these only in
            // sloppy code). This also covers an octal-bearing directive that
            // precedes a `"use strict"` directive in the same prologue.
            Expr::Str {
                legacy_octal: true,
                span,
                ..
            } if ctx.strict => Err(self.err(
                *span,
                "legacy octal / non-octal escape sequences are not allowed in strict mode",
            )),
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
                        "`arguments` is not allowed in a class field initializer or static block",
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
            Expr::Super(span) => {
                // `super` is only legal as a `SuperProperty` (`super.x`,
                // `super[e]`) or a `SuperCall` (`super(...)`). It is reached here
                // as the object of a `Member` or the callee of a `Call`; in either
                // case the keyword is valid only where the enclosing scope grants a
                // super reference. Both forms require some home object: a method,
                // accessor, constructor, class field initializer, or static block
                // (super-property), or a derived-class constructor (super-call).
                // In a plain function body/params or in top-level code neither is
                // permitted, which is an early Syntax Error.
                if !ctx.allow_super_property && !ctx.allow_super_call {
                    return Err(self.err(
                        *span,
                        "`super` is only valid inside a method or a derived class constructor",
                    ));
                }
                Ok(())
            }
            Expr::PrivateName(_, span) => {
                // A bare private-name reference is only legal as the immediate
                // left operand of an `in` (the `#x in obj` brand check), which is
                // handled in the `Binary` arm below. Reaching it anywhere else —
                // as a standalone value, or as the *right* operand of `in`
                // (`#a in #b`) — is an early Syntax Error.
                Err(self.err(
                    *span,
                    "a private name is only allowed as the left operand of `in`",
                ))
            }
            Expr::Template(t) => {
                // An *untagged* template literal with an invalid escape is an
                // early Syntax Error; that check runs in the parser
                // (`cook::validate_template_escapes`, gated on the untagged
                // position), so nothing extra is needed here.
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
                            // In a destructuring target each element is itself a
                            // sub-pattern, so the flag propagates inward.
                            self.in_assign_target = in_assign_target;
                            self.expr(x, ctx)?
                        }
                    }
                }
                Ok(())
            }
            Expr::Object { members, .. } => {
                // It is a Syntax Error if an object literal (used as an
                // expression, not a pattern) has more than one `__proto__` *data*
                // property — a plain `__proto__: value` (or `'__proto__': value`),
                // not a shorthand, computed key, method, or accessor.
                //
                // NB: after parsing, an object method `__proto__(){}` is
                // indistinguishable from a data property `__proto__: function(){}`
                // (both are `Property { value: Function }`), so a property whose
                // value is a function/arrow is conservatively *not* counted —
                // this avoids wrongly rejecting `{ __proto__(){}, __proto__(){} }`
                // at the cost of not flagging the (rare) duplicate
                // `__proto__: function(){}` data form.
                if !in_assign_target {
                    let mut proto_data = 0u32;
                    for m in members {
                        if let ObjectMember::Property {
                            key,
                            value,
                            shorthand: false,
                            span,
                            ..
                        } = m
                            && !matches!(key, PropertyKey::Computed(_))
                            && !matches!(&**value, Expr::Function(_) | Expr::Arrow(_))
                            && key_is_named(key, "__proto__")
                        {
                            proto_data += 1;
                            if proto_data > 1 {
                                return Err(self.err(
                                    *span,
                                    "an object literal may not have more than one \
                                     `__proto__` property",
                                ));
                            }
                        }
                    }
                }
                for m in members {
                    // A private name (`#x`) is only a valid key inside a class
                    // body, never in an object literal.
                    if let ObjectMember::Property {
                        key: PropertyKey::Private(_),
                        span,
                        ..
                    }
                    | ObjectMember::Accessor {
                        key: PropertyKey::Private(_),
                        span,
                        ..
                    } = m
                    {
                        return Err(
                            self.err(*span, "a private name is only valid as a class member")
                        );
                    }
                    match m {
                        ObjectMember::Property {
                            key,
                            value,
                            shorthand,
                            method,
                            span,
                        } => {
                            if let PropertyKey::Computed(k) = key {
                                self.expr(k, ctx)?;
                            }
                            // A *method definition* (`key() {}`, `*key() {}`,
                            // `async key() {}`) uses `UniqueFormalParameters`, so a
                            // duplicate parameter name is always a Syntax Error —
                            // unlike a data property whose value is an ordinary
                            // function. The cover grammar makes the two AST-identical,
                            // so the `method` flag distinguishes them here.
                            if *method && let Expr::Function(f) = &**value {
                                self.method_function(f, ctx, false)?;
                                continue;
                            }
                            // A `CoverInitializedName` (`{ a = 1 }`): a shorthand
                            // property whose value is a simple-assignment default.
                            // Valid only when this object is a destructuring
                            // target; otherwise an early Syntax Error.
                            let is_cover = *shorthand
                                && matches!(
                                    &**value,
                                    Expr::Assign {
                                        op: crate::ast::AssignOp::Assign,
                                        ..
                                    }
                                );
                            if is_cover && !in_assign_target {
                                return Err(self.err(
                                    *span,
                                    "a shorthand property with an initializer (`{ a = … }`) is \
                                     only valid as a destructuring assignment target",
                                ));
                            }
                            // A property *value* is a sub-pattern in a
                            // destructuring target (`{ a: { b = 1 } } = x`), but a
                            // cover-name's value is the *default* (a value), so the
                            // flag is not propagated through it.
                            if in_assign_target && !is_cover {
                                self.in_assign_target = true;
                            }
                            self.expr(value, ctx)?;
                        }
                        ObjectMember::Spread { value, .. } => {
                            self.in_assign_target = in_assign_target;
                            self.expr(value, ctx)?
                        }
                        ObjectMember::Accessor {
                            key,
                            value,
                            is_getter,
                            ..
                        } => {
                            if let PropertyKey::Computed(k) = key {
                                self.expr(k, ctx)?;
                            }
                            self.check_accessor_arity(*is_getter, value)?;
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
                if matches!(op, UnaryOp::Delete) {
                    if contains_private_ref(argument) {
                        return Err(
                            self.err(e.span(), "`delete` of a private member is not allowed")
                        );
                    }
                    // In strict mode, `delete` of a direct reference to a variable,
                    // function argument, or function name (a bare identifier, even
                    // parenthesized) is a Syntax Error.
                    if ctx.strict && matches!(&**argument, Expr::Ident(_)) {
                        return Err(self.err(
                            e.span(),
                            "`delete` of an unqualified identifier is not allowed in strict mode",
                        ));
                    }
                }
                self.expr(argument, ctx)
            }
            Expr::Update { argument, .. } => {
                // The operand of `++`/`--` must be a *simple* reference — an
                // identifier or member access — never a call, literal, or
                // destructuring pattern. AnnexB web-compat is the one exception: a
                // direct CallExpression operand is a runtime ReferenceError in
                // sloppy code (a SyntaxError in strict mode).
                if !is_simple_update_target(argument)
                    && !(!ctx.strict && argument.is_web_compat_call_target())
                {
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
                        // The private-in brand check `#x in obj`: the left operand
                        // is the only position where a bare private name is legal.
                        // Validate its declaration here and continue with the
                        // right operand, bypassing the generic `PrivateName` arm
                        // (which rejects every other position).
                        Expr::Binary {
                            op: crate::ast::BinaryOp::In,
                            left,
                            right,
                            ..
                        } if matches!(&**left, Expr::PrivateName(..)) => {
                            if let Expr::PrivateName(name, span) = &**left
                                && !self.private_in_scope(name)
                            {
                                return Err(self.err(*span, "reference to undeclared private name"));
                            }
                            break self.expr(right, ctx);
                        }
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
                let simple_assign = matches!(op, crate::ast::AssignOp::Assign);
                // AnnexB web-compat: a direct CallExpression target for a simple
                // (`=`) or arithmetic/bitwise compound (`+=`, …) assignment is a
                // runtime ReferenceError in sloppy code, a SyntaxError in strict.
                // Logical assignment (`&&=`/`||=`/`??=`) is excluded — its call
                // target is a SyntaxError in both modes (rejected in the parser).
                let logical = matches!(
                    op,
                    crate::ast::AssignOp::AndAssign
                        | crate::ast::AssignOp::OrAssign
                        | crate::ast::AssignOp::NullishAssign
                );
                let web_compat_call = !logical && !ctx.strict && target.is_web_compat_call_target();
                if simple_assign {
                    if !is_valid_assign_target(target) && !web_compat_call {
                        return Err(self.err(target.span(), "invalid assignment target"));
                    }
                } else if !is_simple_update_target(target) && !web_compat_call {
                    return Err(self.err(target.span(), "invalid target for compound assignment"));
                }
                if ctx.strict {
                    self.check_not_eval_arguments(target)?;
                }
                // A simple assignment may target an object/array destructuring
                // pattern, where a `CoverInitializedName` is valid.
                self.in_assign_target = simple_assign;
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
            Expr::Await { argument, span } => {
                if ctx.in_static_block {
                    return Err(self.err(
                        *span,
                        "`await` is not allowed in a class static initialization block",
                    ));
                }
                self.expr(argument, ctx)
            }
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

    /// Module-body export early errors (§16.2.3.1):
    /// - It is a Syntax Error if the **ExportedNames** of the module contain any
    ///   duplicate entries (`export { x }; export { y as x };`,
    ///   `export default 1; export default 2;`, …).
    /// - It is a Syntax Error if any element of the **ExportedBindings** is not
    ///   declared at the module top level (`export { undeclared };`). A re-export
    ///   (`export { x } from "m"`) is *not* a binding requirement.
    fn check_module_exports(&self, body: &[Stmt]) -> Result<()> {
        use crate::ast::ExportDecl;

        // At a *module* top level a `FunctionDeclaration` is a
        // LexicallyDeclaredName (not var-scoped as in a script), so two top-level
        // function declarations of the same name — or a function clashing with a
        // let/const/class — are early errors. Collect every top-level lexical
        // *binding* name (including module-top-level functions, and the names
        // bound by an `export <decl>`) and reject duplicates.
        let mut top_lex: Vec<(Box<str>, Span)> = Vec::new();
        for stmt in body {
            collect_module_top_lexical(stmt, &mut top_lex);
        }
        for i in 0..top_lex.len() {
            for j in (i + 1)..top_lex.len() {
                if top_lex[i].0 == top_lex[j].0 {
                    return Err(self.err(top_lex[j].1, "duplicate top-level declaration in module"));
                }
            }
        }
        // A module top-level LexicallyDeclaredName (let/const/class, and — unlike
        // a script — a top-level function declaration) may not also be a
        // VarDeclaredName (`var f; function f() {}` is a module early error).
        let mut var_only: Vec<Box<str>> = Vec::new();
        for stmt in body {
            collect_vars_only(stmt, &mut var_only);
        }
        for (name, span) in &top_lex {
            if var_only.iter().any(|v| v == name) {
                return Err(self.err(
                    *span,
                    "a module top-level lexical declaration conflicts with a `var` of the same name",
                ));
            }
        }

        // The set of names bound at the module top level (VarDeclaredNames +
        // LexicallyDeclaredNames + imported bindings) against which a local
        // `export { x }` is checked.
        let mut declared: Vec<Box<str>> = Vec::new();
        let mut lexical: Vec<(Box<str>, Span, bool)> = Vec::new();
        let mut vars: Vec<Box<str>> = Vec::new();
        for stmt in body {
            collect_top_level_decls(stmt, &mut lexical, &mut vars, true);
            if let Stmt::Import(decl) = stmt {
                for s in &decl.specifiers {
                    let (name, span) = match s {
                        crate::ast::ImportSpecifier::Default(id)
                        | crate::ast::ImportSpecifier::Namespace(id)
                        | crate::ast::ImportSpecifier::Source(id) => (id.name.clone(), id.span),
                        crate::ast::ImportSpecifier::Named { local, .. } => {
                            (local.name.clone(), local.span)
                        }
                    };
                    // Module code is always strict, so an imported binding may not
                    // be named `eval` or `arguments` (a strict BindingIdentifier
                    // early error).
                    if &*name == "eval" || &*name == "arguments" {
                        return Err(self.err(
                            span,
                            "an imported binding may not be named `eval` or `arguments` in strict (module) code",
                        ));
                    }
                    declared.push(name);
                }
            }
            // `export default <expr/anonymous decl>` introduces the synthetic
            // binding `*default*`, which backs a local `default` export.
            if let Stmt::Export(ExportDecl::Default { .. }) = stmt {
                declared.push("*default*".into());
            }
        }
        declared.extend(vars);
        declared.extend(lexical.into_iter().map(|(n, _, _)| n));

        let mut exported: Vec<(String, Span)> = Vec::new();
        // A local export binding `name` (with the span to report) to validate
        // against `declared`.
        let mut bindings: Vec<(String, Span)> = Vec::new();
        for stmt in body {
            let Stmt::Export(decl) = stmt else { continue };
            match decl {
                ExportDecl::Default { span, .. } => {
                    exported.push((String::from("default"), *span));
                }
                ExportDecl::Decl { declaration, span } => {
                    let mut names: Vec<(Box<str>, Span, bool)> = Vec::new();
                    let mut vsink: Vec<Box<str>> = Vec::new();
                    collect_top_level_decls(declaration, &mut names, &mut vsink, true);
                    for n in names.into_iter().map(|(n, _, _)| n).chain(vsink) {
                        exported.push((n.to_string(), *span));
                    }
                }
                ExportDecl::All {
                    exported: Some(name),
                    span,
                    ..
                } => exported.push((module_export_name_str(name), *span)),
                // `export * from "m"` introduces no export name here.
                ExportDecl::All { exported: None, .. } => {}
                ExportDecl::Named {
                    specifiers, source, ..
                } => {
                    for sp in specifiers {
                        exported.push((module_export_name_str(&sp.exported), sp.span));
                        // Only a *local* `export { x }` (no `from`) requires `x`
                        // to be a declared binding; a re-export does not.
                        if source.is_none() {
                            match &sp.local {
                                crate::ast::ModuleExportName::Ident(n) => {
                                    bindings.push((n.to_string(), sp.span));
                                }
                                // `export { "str" }` without `from` references a
                                // binding by a string name, which is never a valid
                                // local identifier — a Syntax Error (a string
                                // ModuleExportName is only legal as a re-export).
                                crate::ast::ModuleExportName::Str(_) => {
                                    return Err(self.err(
                                        sp.span,
                                        "a string module export name requires a `from` clause",
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Duplicate ExportedNames.
        for i in 0..exported.len() {
            for j in (i + 1)..exported.len() {
                if exported[i].0 == exported[j].0 {
                    return Err(self.err(exported[j].1, "duplicate export name in module"));
                }
            }
        }
        // Every local export binding must be declared somewhere in the module.
        for (name, span) in &bindings {
            if !declared.iter().any(|d| d.as_ref() == name.as_str()) {
                return Err(self.err(*span, "export of an undeclared identifier"));
            }
        }
        Ok(())
    }

    /// It is a Syntax Error if any element of the `LexicallyDeclaredNames` of a
    /// function body also occurs in the `BoundNames` of its parameters (a `let`,
    /// `const`, or `class` in the body may not shadow a parameter — e.g.
    /// `function f(a) { let a; }`). `var` bindings are exempt: they share the
    /// parameter scope.
    fn check_params_vs_body_lexical(&self, params: &[Param], body: &[Stmt]) -> Result<()> {
        // Collect the body's top-level lexical names (let/const/class, plus
        // top-level function declarations, which are not var-hoisted out of a
        // function-parameter scope for this check).
        let mut lexical: Vec<(Box<str>, Span, bool)> = Vec::new();
        let mut vars: Vec<Box<str>> = Vec::new();
        for stmt in body {
            collect_top_level_decls(stmt, &mut lexical, &mut vars, true);
        }
        if lexical.is_empty() {
            return Ok(());
        }
        let mut param_names = Vec::new();
        for p in params {
            collect_bound_names(&p.target, &mut param_names);
        }
        for (name, span, is_function) in &lexical {
            // A top-level `FunctionDeclaration` in a function body is var-scoped,
            // so it may shadow a parameter (`function f(a){ function a(){} }`).
            if *is_function {
                continue;
            }
            if param_names.iter().any(|(n, _)| n.as_str() == name.as_ref()) {
                return Err(self.err(
                    *span,
                    "a lexical declaration conflicts with a parameter of the same name",
                ));
            }
        }
        Ok(())
    }

    /// Catch-clause early errors (14.15.1, with Annex B.3.4):
    /// - The `CatchParameter` BoundNames must be distinct (`catch ([x, x])`).
    /// - They must not occur in the body's LexicallyDeclaredNames
    ///   (`catch (x) { let x; }`, incl. a block-level function declaration).
    /// - They must not occur in the body's VarDeclaredNames — *except* a simple
    ///   `catch (e)` binding may be redeclared by a `var` (Annex B).
    fn check_catch_param(&self, param: &BindingTarget, body: &[Stmt]) -> Result<()> {
        let mut bound = Vec::new();
        collect_bound_names(param, &mut bound);
        self.check_no_dup_catch_names(&bound)?;

        let simple = matches!(param, BindingTarget::Ident(_));

        // Any lexically-declared name in the catch body (let/const/class or a
        // block-level function declaration) conflicts with a catch binding of the
        // same name.
        let mut lexical: Vec<(Box<str>, Span, bool)> = Vec::new();
        let mut sink = Vec::new();
        for stmt in body {
            collect_top_level_decls(stmt, &mut lexical, &mut sink, false);
        }
        for (name, span, _) in &lexical {
            if bound.iter().any(|(n, _)| n.as_str() == name.as_ref()) {
                return Err(self.err(
                    *span,
                    "a `catch` body may not lexically redeclare the catch parameter",
                ));
            }
        }

        // A non-simple (destructuring) catch binding additionally forbids any
        // `var` redeclaration of its names.
        if !simple {
            let mut vars = Vec::new();
            for stmt in body {
                collect_vars_only(stmt, &mut vars);
            }
            for (name, span) in &bound {
                if vars.iter().any(|v| v.as_ref() == name.as_str()) {
                    return Err(self.err(
                        *span,
                        "a destructuring `catch` binding may not be redeclared by a `var`",
                    ));
                }
            }
        }
        Ok(())
    }

    /// The `CatchParameter` BoundNames must not contain duplicates.
    fn check_no_dup_catch_names(&self, bound: &[(String, Span)]) -> Result<()> {
        for (i, (name, _)) in bound.iter().enumerate() {
            if bound[..i].iter().any(|(n, _)| n == name) {
                return Err(self.err(
                    bound[i].1,
                    "a `catch` binding list may not contain a duplicate name",
                ));
            }
        }
        Ok(())
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
        // An `export <declaration>` / `export default <decl>` contributes its
        // inner declaration's bound names to the enclosing (module top-level)
        // scope, so a duplicate `export function f` + `function f` (or two
        // `export default function`s) is caught by the lexical-uniqueness check.
        Stmt::Export(decl) => {
            if let Some(inner) = export_inner_decl(decl) {
                collect_top_level_decls(inner, lexical, vars, top_level);
            }
        }
        // `var` hoists out of nested non-function statements.
        other => collect_vars_only(other, vars),
    }
}

/// Collects a module top-level statement's **lexically-declared binding names**
/// for the module duplicate-declaration early error. Unlike a script, a module
/// top-level `FunctionDeclaration` is a LexicallyDeclaredName, so it is included
/// here (with `let`/`const`/`class`); plain `var`s are not (they may be
/// redeclared). Descends through an `export <decl>` to its inner declaration.
fn collect_module_top_lexical(stmt: &Stmt, out: &mut Vec<(Box<str>, Span)>) {
    match stmt {
        Stmt::Function(f) => {
            if let Some(id) = &f.id {
                out.push((id.name.clone(), id.span));
            }
        }
        Stmt::Class(c) => {
            if let Some(id) = &c.id {
                out.push((id.name.clone(), id.span));
            }
        }
        Stmt::Var(decl) if decl.kind != VarDeclKind::Var => {
            for d in &decl.declarations {
                let mut names = Vec::new();
                collect_bound_names(&d.target, &mut names);
                for (n, span) in names {
                    out.push((n.into(), span));
                }
            }
        }
        // The BoundNames of an `ImportDeclaration` are LexicallyDeclaredNames of
        // the module, so `import { x, y as x } from "m"` — or an imported name
        // that clashes with a top-level `let`/`class`/function — is an early
        // error just like two `let`s of one name.
        Stmt::Import(decl) => {
            for s in &decl.specifiers {
                let id = match s {
                    crate::ast::ImportSpecifier::Default(id)
                    | crate::ast::ImportSpecifier::Namespace(id)
                    | crate::ast::ImportSpecifier::Source(id)
                    | crate::ast::ImportSpecifier::Named { local: id, .. } => id,
                };
                out.push((id.name.clone(), id.span));
            }
        }
        Stmt::Export(decl) => {
            if let Some(inner) = export_inner_decl(decl) {
                collect_module_top_lexical(inner, out);
            }
        }
        _ => {}
    }
}

/// The string form of a `ModuleExportName` (identifier or string-literal name).
fn module_export_name_str(n: &crate::ast::ModuleExportName) -> String {
    match n {
        crate::ast::ModuleExportName::Ident(s) | crate::ast::ModuleExportName::Str(s) => {
            s.to_string()
        }
    }
}

/// The inner declaration statement of an `export <decl>` / `export default
/// <decl>`, if the export wraps a `var`/`let`/`const`/function/class declaration
/// (a re-export, named-specifier list, or `export default <expr>` has none).
fn export_inner_decl(decl: &crate::ast::ExportDecl) -> Option<&Stmt> {
    use crate::ast::ExportDecl;
    match decl {
        ExportDecl::Decl { declaration, .. } => Some(declaration),
        ExportDecl::Default { declaration, .. } => match &**declaration {
            // `export default function f`/`class C` (a *named* declaration) binds
            // `f`/`C`; an anonymous default or a default *expression* binds no
            // enclosing-scope lexical name (only the synthetic `*default*`).
            s @ (Stmt::Function(crate::ast::Function { id: Some(_), .. })
            | Stmt::Class(crate::ast::Class { id: Some(_), .. })) => Some(s),
            _ => None,
        },
        ExportDecl::Named { .. } | ExportDecl::All { .. } => None,
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

/// Whether a (possibly label-nested) statement is, at its core, a
/// `FunctionDeclaration` — i.e. a `LabelledItem : FunctionDeclaration`. Used to
/// reject a labelled function in single-statement position (its Annex B
/// allowance applies only at a `StatementListItem` position).
fn labels_a_function(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Function(_) => true,
        Stmt::Labeled { body, .. } => labels_a_function(body),
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
    match e {
        Expr::Member { .. } => !is_import_meta(e),
        Expr::Ident(_) => true,
        _ => false,
    }
}

/// Whether `e` is the `import.meta` meta-property (parsed as a member access on
/// the reserved `import` keyword). It is not a valid assignment / update target
/// (its AssignmentTargetType is `invalid`).
fn is_import_meta(e: &Expr) -> bool {
    if let Expr::Member {
        object, property, ..
    } = e
        && let (Expr::Ident(id), PropertyKey::Ident(name)) = (&**object, property)
    {
        return &*id.name == "import" && &**name == "meta";
    }
    false
}

/// Whether `e` is a valid target for a *simple* (`=`) assignment. This permits
/// destructuring patterns (with `= default` inside them) but rejects a bare
/// parenthesized assignment such as `(x = y) = 1` — represented here as a
/// top-level [`Expr::Assign`] target.
fn is_valid_assign_target(e: &Expr) -> bool {
    match e {
        Expr::Member { .. } => !is_import_meta(e),
        Expr::Ident(_) => true,
        Expr::Array {
            elements,
            rest_trailing_comma,
            ..
        } => {
            // An `AssignmentRestElement` may not be followed by a comma — even a
            // trailing one (`[...x,]`), which the literal grammar otherwise drops.
            if *rest_trailing_comma {
                return false;
            }
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
