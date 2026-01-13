//! Pretty printer for TLA+ AST
//!
//! Converts AST back to TLA+ source code. Used for:
//! - Roundtrip testing (parse -> AST -> pretty print -> compare)
//! - Code generation
//! - Error message context

use crate::ast::*;

/// A wrapper for indentation-aware writing
pub struct PrettyPrinter {
    output: String,
    indent: usize,
    indent_str: &'static str,
}

impl Default for PrettyPrinter {
    fn default() -> Self {
        Self::new()
    }
}

impl PrettyPrinter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
            indent_str: "  ",
        }
    }

    pub fn finish(self) -> String {
        self.output
    }

    fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.output.push_str(self.indent_str);
        }
    }

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn writeln(&mut self, s: &str) {
        self.output.push_str(s);
        self.output.push('\n');
    }

    fn newline(&mut self) {
        self.output.push('\n');
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }
}

/// Pretty print a module
pub fn pretty_module(module: &Module) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_module(module);
    pp.finish()
}

/// Pretty print an expression
pub fn pretty_expr(expr: &Expr) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_expr(expr);
    pp.finish()
}

impl PrettyPrinter {
    pub fn print_module(&mut self, module: &Module) {
        // Module header
        self.write("---- MODULE ");
        self.write(&module.name.node);
        self.writeln(" ----");

        // EXTENDS clause
        if !module.extends.is_empty() {
            self.write("EXTENDS ");
            for (i, ext) in module.extends.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.write(&ext.node);
            }
            self.newline();
        }

        // Module body units
        for unit in &module.units {
            self.newline();
            self.print_unit(&unit.node);
        }

        // Module footer
        self.newline();
        self.writeln("====");
    }

    pub fn print_unit(&mut self, unit: &Unit) {
        match unit {
            Unit::Variable(vars) => {
                self.write("VARIABLE ");
                for (i, var) in vars.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&var.node);
                }
                self.newline();
            }
            Unit::Constant(consts) => {
                self.write("CONSTANT ");
                for (i, c) in consts.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&c.name.node);
                    if let Some(arity) = c.arity {
                        self.write("(");
                        for j in 0..arity {
                            if j > 0 {
                                self.write(", ");
                            }
                            self.write("_");
                        }
                        self.write(")");
                    }
                }
                self.newline();
            }
            Unit::Recursive(decls) => {
                self.write("RECURSIVE ");
                for (i, r) in decls.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&r.name.node);
                    if r.arity > 0 {
                        self.write("(");
                        for j in 0..r.arity {
                            if j > 0 {
                                self.write(", ");
                            }
                            self.write("_");
                        }
                        self.write(")");
                    }
                }
                self.newline();
            }
            Unit::Operator(op_def) => {
                self.print_operator_def(op_def, false);
            }
            Unit::Instance(inst) => {
                if inst.local {
                    self.write("LOCAL ");
                }
                self.write("INSTANCE ");
                self.write(&inst.module.node);
                if !inst.substitutions.is_empty() {
                    self.write(" WITH ");
                    for (i, sub) in inst.substitutions.iter().enumerate() {
                        if i > 0 {
                            self.write(", ");
                        }
                        self.write(&sub.from.node);
                        self.write(" <- ");
                        self.print_expr(&sub.to.node);
                    }
                }
                self.newline();
            }
            Unit::Assume(assume) => {
                self.write("ASSUME ");
                if let Some(name) = &assume.name {
                    self.write(&name.node);
                    self.write(" == ");
                }
                self.print_expr(&assume.expr.node);
                self.newline();
            }
            Unit::Theorem(thm) => {
                self.write("THEOREM ");
                if let Some(name) = &thm.name {
                    self.write(&name.node);
                    self.write(" == ");
                }
                self.print_expr(&thm.body.node);
                if let Some(proof) = &thm.proof {
                    self.newline();
                    self.print_proof(&proof.node);
                }
                self.newline();
            }
            Unit::Separator => {
                self.writeln("----");
            }
        }
    }

    fn print_operator_def(&mut self, op: &OperatorDef, in_let: bool) {
        if op.local && !in_let {
            self.write("LOCAL ");
        }
        self.write(&op.name.node);
        if !op.params.is_empty() {
            self.write("(");
            for (i, param) in op.params.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.write(&param.name.node);
                if param.arity > 0 {
                    self.write("(");
                    for j in 0..param.arity {
                        if j > 0 {
                            self.write(", ");
                        }
                        self.write("_");
                    }
                    self.write(")");
                }
            }
            self.write(")");
        }
        self.write(" == ");
        self.print_expr(&op.body.node);
        if !in_let {
            self.newline();
        }
    }

    fn print_proof(&mut self, proof: &Proof) {
        match proof {
            Proof::By(hints) => {
                self.write("BY ");
                self.print_proof_hints(hints);
            }
            Proof::Obvious => {
                self.write("OBVIOUS");
            }
            Proof::Omitted => {
                self.write("OMITTED");
            }
            Proof::Steps(steps) => {
                for step in steps {
                    self.print_proof_step(step);
                }
            }
        }
    }

    fn print_proof_hints(&mut self, hints: &[ProofHint]) {
        for (i, hint) in hints.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            match hint {
                ProofHint::Ref(r) => {
                    self.write(&r.node);
                }
                ProofHint::Def(names) => {
                    self.write("DEF ");
                    for (j, name) in names.iter().enumerate() {
                        if j > 0 {
                            self.write(", ");
                        }
                        self.write(&name.node);
                    }
                }
                ProofHint::Module(m) => {
                    self.write("MODULE ");
                    self.write(&m.node);
                }
            }
        }
    }

    fn print_proof_step(&mut self, step: &ProofStep) {
        self.write_indent();

        // Print level markers
        self.write("<");
        self.write(&step.level.to_string());
        self.write(">");

        // Print label if present
        if let Some(label) = &step.label {
            self.write(&label.node);
            self.write(". ");
        } else {
            self.write(" ");
        }

        match &step.kind {
            ProofStepKind::Assert(expr, proof) => {
                self.print_expr(&expr.node);
                if let Some(p) = proof {
                    self.newline();
                    self.indent();
                    self.write_indent();
                    self.print_proof(&p.node);
                    self.dedent();
                }
            }
            ProofStepKind::Suffices(expr, proof) => {
                self.write("SUFFICES ");
                self.print_expr(&expr.node);
                if let Some(p) = proof {
                    self.newline();
                    self.indent();
                    self.write_indent();
                    self.print_proof(&p.node);
                    self.dedent();
                }
            }
            ProofStepKind::Have(expr) => {
                self.write("HAVE ");
                self.print_expr(&expr.node);
            }
            ProofStepKind::Take(bounds) => {
                self.write("TAKE ");
                self.print_bounds(bounds);
            }
            ProofStepKind::Witness(exprs) => {
                self.write("WITNESS ");
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(&expr.node);
                }
            }
            ProofStepKind::Pick(bounds, pred, proof) => {
                self.write("PICK ");
                self.print_bounds(bounds);
                self.write(" : ");
                self.print_expr(&pred.node);
                if let Some(p) = proof {
                    self.newline();
                    self.indent();
                    self.write_indent();
                    self.print_proof(&p.node);
                    self.dedent();
                }
            }
            ProofStepKind::UseOrHide { use_, facts } => {
                if *use_ {
                    self.write("USE ");
                } else {
                    self.write("HIDE ");
                }
                self.print_proof_hints(facts);
            }
            ProofStepKind::Define(defs) => {
                self.write("DEFINE ");
                for (i, def) in defs.iter().enumerate() {
                    if i > 0 {
                        self.newline();
                        self.write_indent();
                        self.write("       ");
                    }
                    self.print_operator_def(def, true);
                }
            }
            ProofStepKind::Qed(proof) => {
                self.write("QED");
                if let Some(p) = proof {
                    self.newline();
                    self.indent();
                    self.write_indent();
                    self.print_proof(&p.node);
                    self.dedent();
                }
            }
        }
        self.newline();
    }

    pub fn print_expr(&mut self, expr: &Expr) {
        match expr {
            // Literals
            Expr::Bool(b) => {
                self.write(if *b { "TRUE" } else { "FALSE" });
            }
            Expr::Int(n) => {
                self.write(&n.to_string());
            }
            Expr::String(s) => {
                self.write("\"");
                // Escape special characters
                for c in s.chars() {
                    match c {
                        '"' => self.write("\\\""),
                        '\\' => self.write("\\\\"),
                        '\n' => self.write("\\n"),
                        '\t' => self.write("\\t"),
                        _ => self.output.push(c),
                    }
                }
                self.write("\"");
            }

            // Names
            Expr::Ident(name) => {
                self.write(name);
            }

            // Operators
            Expr::Apply(op, args) => {
                self.print_expr(&op.node);
                self.write("(");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(&arg.node);
                }
                self.write(")");
            }
            Expr::OpRef(name) => {
                // Operator reference: bare operator as value
                self.write(name);
            }
            Expr::Lambda(params, body) => {
                self.write("LAMBDA ");
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&param.node);
                }
                self.write(" : ");
                self.print_expr(&body.node);
            }
            Expr::ModuleRef(module_target, op, args) => {
                match module_target {
                    ModuleTarget::Named(name) => {
                        self.write(name);
                    }
                    ModuleTarget::Parameterized(name, params) => {
                        self.write(name);
                        self.write("(");
                        for (i, param) in params.iter().enumerate() {
                            if i > 0 {
                                self.write(", ");
                            }
                            self.print_expr(&param.node);
                        }
                        self.write(")");
                    }
                    ModuleTarget::Chained(base) => {
                        // Print the base module reference (without the trailing !)
                        self.print_expr(&base.node);
                    }
                }
                self.write("!");
                self.write(op);
                if !args.is_empty() {
                    self.write("(");
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            self.write(", ");
                        }
                        self.print_expr(&arg.node);
                    }
                    self.write(")");
                }
            }
            Expr::InstanceExpr(module, substitutions) => {
                self.write("INSTANCE ");
                self.write(module);
                if !substitutions.is_empty() {
                    self.write(" WITH ");
                    for (i, sub) in substitutions.iter().enumerate() {
                        if i > 0 {
                            self.write(", ");
                        }
                        self.write(&sub.from.node);
                        self.write(" <- ");
                        self.print_expr(&sub.to.node);
                    }
                }
            }

            // Logic
            Expr::And(left, right) => {
                self.print_binop(&left.node, " /\\ ", &right.node);
            }
            Expr::Or(left, right) => {
                self.print_binop(&left.node, " \\/ ", &right.node);
            }
            Expr::Not(e) => {
                self.write("~");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }
            Expr::Implies(left, right) => {
                self.print_binop(&left.node, " => ", &right.node);
            }
            Expr::Equiv(left, right) => {
                self.print_binop(&left.node, " <=> ", &right.node);
            }

            // Quantifiers
            Expr::Forall(bounds, body) => {
                self.write("\\A ");
                self.print_bounds(bounds);
                self.write(" : ");
                self.print_expr(&body.node);
            }
            Expr::Exists(bounds, body) => {
                self.write("\\E ");
                self.print_bounds(bounds);
                self.write(" : ");
                self.print_expr(&body.node);
            }
            Expr::Choose(bound, body) => {
                self.write("CHOOSE ");
                self.print_bound(bound);
                self.write(" : ");
                self.print_expr(&body.node);
            }

            // Sets
            Expr::SetEnum(elems) => {
                self.write("{");
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(&elem.node);
                }
                self.write("}");
            }
            Expr::SetBuilder(body, bounds) => {
                self.write("{");
                self.print_expr(&body.node);
                self.write(" : ");
                self.print_bounds(bounds);
                self.write("}");
            }
            Expr::SetFilter(bound, pred) => {
                self.write("{");
                self.print_bound(bound);
                self.write(" : ");
                self.print_expr(&pred.node);
                self.write("}");
            }
            Expr::In(left, right) => {
                self.print_binop(&left.node, " \\in ", &right.node);
            }
            Expr::NotIn(left, right) => {
                self.print_binop(&left.node, " \\notin ", &right.node);
            }
            Expr::Subseteq(left, right) => {
                self.print_binop(&left.node, " \\subseteq ", &right.node);
            }
            Expr::Union(left, right) => {
                self.print_binop(&left.node, " \\cup ", &right.node);
            }
            Expr::Intersect(left, right) => {
                self.print_binop(&left.node, " \\cap ", &right.node);
            }
            Expr::SetMinus(left, right) => {
                self.print_binop(&left.node, " \\ ", &right.node);
            }
            Expr::Powerset(e) => {
                self.write("SUBSET ");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }
            Expr::BigUnion(e) => {
                self.write("UNION ");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }

            // Functions
            Expr::FuncDef(bounds, body) => {
                self.write("[");
                self.print_bounds(bounds);
                self.write(" |-> ");
                self.print_expr(&body.node);
                self.write("]");
            }
            Expr::FuncApply(func, arg) => {
                self.print_expr(&func.node);
                self.write("[");
                self.print_expr(&arg.node);
                self.write("]");
            }
            Expr::Domain(e) => {
                self.write("DOMAIN ");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }
            Expr::Except(func, specs) => {
                self.write("[");
                self.print_expr(&func.node);
                self.write(" EXCEPT ");
                for (i, spec) in specs.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write("!");
                    for elem in &spec.path {
                        match elem {
                            ExceptPathElement::Index(idx) => {
                                self.write("[");
                                self.print_expr(&idx.node);
                                self.write("]");
                            }
                            ExceptPathElement::Field(f) => {
                                self.write(".");
                                self.write(&f.node);
                            }
                        }
                    }
                    self.write(" = ");
                    self.print_expr(&spec.value.node);
                }
                self.write("]");
            }
            Expr::FuncSet(domain, range) => {
                self.write("[");
                self.print_expr(&domain.node);
                self.write(" -> ");
                self.print_expr(&range.node);
                self.write("]");
            }

            // Records
            Expr::Record(fields) => {
                self.write("[");
                for (i, (name, val)) in fields.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&name.node);
                    self.write(" |-> ");
                    self.print_expr(&val.node);
                }
                self.write("]");
            }
            Expr::RecordAccess(rec, field) => {
                self.print_expr(&rec.node);
                self.write(".");
                self.write(&field.node);
            }
            Expr::RecordSet(fields) => {
                self.write("[");
                for (i, (name, typ)) in fields.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&name.node);
                    self.write(" : ");
                    self.print_expr(&typ.node);
                }
                self.write("]");
            }

            // Tuples
            Expr::Tuple(elems) => {
                self.write("<<");
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(&elem.node);
                }
                self.write(">>");
            }
            Expr::Times(sets) => {
                for (i, set) in sets.iter().enumerate() {
                    if i > 0 {
                        self.write(" \\X ");
                    }
                    self.print_expr(&set.node);
                }
            }

            // Temporal
            Expr::Prime(e) => {
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
                self.write("'");
            }
            Expr::Always(e) => {
                self.write("[]");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }
            Expr::Eventually(e) => {
                self.write("<>");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }
            Expr::LeadsTo(left, right) => {
                self.print_binop(&left.node, " ~> ", &right.node);
            }
            Expr::WeakFair(vars, action) => {
                self.write("WF_");
                self.print_expr(&vars.node);
                self.write("(");
                self.print_expr(&action.node);
                self.write(")");
            }
            Expr::StrongFair(vars, action) => {
                self.write("SF_");
                self.print_expr(&vars.node);
                self.write("(");
                self.print_expr(&action.node);
                self.write(")");
            }

            // Actions
            Expr::Enabled(e) => {
                self.write("ENABLED ");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }
            Expr::Unchanged(e) => {
                self.write("UNCHANGED ");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }

            // Control
            Expr::If(cond, then_e, else_e) => {
                self.write("IF ");
                self.print_expr(&cond.node);
                self.write(" THEN ");
                self.print_expr(&then_e.node);
                self.write(" ELSE ");
                self.print_expr(&else_e.node);
            }
            Expr::Case(arms, otherwise) => {
                self.write("CASE ");
                for (i, arm) in arms.iter().enumerate() {
                    if i > 0 {
                        self.write(" [] ");
                    }
                    self.print_expr(&arm.guard.node);
                    self.write(" -> ");
                    self.print_expr(&arm.body.node);
                }
                if let Some(other) = otherwise {
                    self.write(" [] OTHER -> ");
                    self.print_expr(&other.node);
                }
            }
            Expr::Let(defs, body) => {
                self.write("LET ");
                for (i, def) in defs.iter().enumerate() {
                    if i > 0 {
                        self.newline();
                        self.write_indent();
                        self.write("    ");
                    }
                    self.print_operator_def(def, true);
                }
                self.newline();
                self.write_indent();
                self.write("IN ");
                self.print_expr(&body.node);
            }

            // Comparison
            Expr::Eq(left, right) => {
                self.print_binop(&left.node, " = ", &right.node);
            }
            Expr::Neq(left, right) => {
                self.print_binop(&left.node, " /= ", &right.node);
            }
            Expr::Lt(left, right) => {
                self.print_binop(&left.node, " < ", &right.node);
            }
            Expr::Leq(left, right) => {
                self.print_binop(&left.node, " <= ", &right.node);
            }
            Expr::Gt(left, right) => {
                self.print_binop(&left.node, " > ", &right.node);
            }
            Expr::Geq(left, right) => {
                self.print_binop(&left.node, " >= ", &right.node);
            }

            // Arithmetic
            Expr::Add(left, right) => {
                self.print_binop(&left.node, " + ", &right.node);
            }
            Expr::Sub(left, right) => {
                self.print_binop(&left.node, " - ", &right.node);
            }
            Expr::Mul(left, right) => {
                self.print_binop(&left.node, " * ", &right.node);
            }
            Expr::Div(left, right) => {
                self.print_binop(&left.node, " / ", &right.node);
            }
            Expr::IntDiv(left, right) => {
                self.print_binop(&left.node, " \\div ", &right.node);
            }
            Expr::Mod(left, right) => {
                self.print_binop(&left.node, " % ", &right.node);
            }
            Expr::Pow(left, right) => {
                self.print_binop(&left.node, "^", &right.node);
            }
            Expr::Neg(e) => {
                self.write("-");
                self.print_expr_parens(&e.node, needs_parens(expr, &e.node));
            }
            Expr::Range(left, right) => {
                self.print_binop(&left.node, "..", &right.node);
            }
        }
    }

    fn print_binop(&mut self, left: &Expr, op: &str, right: &Expr) {
        self.print_expr(left);
        self.write(op);
        self.print_expr(right);
    }

    fn print_expr_parens(&mut self, expr: &Expr, parens: bool) {
        if parens {
            self.write("(");
            self.print_expr(expr);
            self.write(")");
        } else {
            self.print_expr(expr);
        }
    }

    fn print_bounds(&mut self, bounds: &[BoundVar]) {
        for (i, bound) in bounds.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.print_bound(bound);
        }
    }

    fn print_bound(&mut self, bound: &BoundVar) {
        self.write(&bound.name.node);
        if let Some(domain) = &bound.domain {
            self.write(" \\in ");
            self.print_expr(&domain.node);
        }
    }
}

/// Determine if an expression needs parentheses in a given context
fn needs_parens(parent: &Expr, child: &Expr) -> bool {
    // For unary operators, we need parens around binary expressions
    match parent {
        Expr::Not(_)
        | Expr::Neg(_)
        | Expr::Prime(_)
        | Expr::Always(_)
        | Expr::Eventually(_)
        | Expr::Enabled(_)
        | Expr::Unchanged(_)
        | Expr::Powerset(_)
        | Expr::BigUnion(_)
        | Expr::Domain(_) => {
            matches!(
                child,
                Expr::And(_, _)
                    | Expr::Or(_, _)
                    | Expr::Implies(_, _)
                    | Expr::Equiv(_, _)
                    | Expr::Eq(_, _)
                    | Expr::Neq(_, _)
                    | Expr::Lt(_, _)
                    | Expr::Leq(_, _)
                    | Expr::Gt(_, _)
                    | Expr::Geq(_, _)
                    | Expr::Add(_, _)
                    | Expr::Sub(_, _)
                    | Expr::Mul(_, _)
                    | Expr::Div(_, _)
                    | Expr::In(_, _)
                    | Expr::NotIn(_, _)
            )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::lower;
    use crate::span::FileId;
    use crate::syntax::parse_to_syntax_tree;
    use insta::assert_snapshot;

    /// Parse source -> CST -> AST -> pretty print
    fn roundtrip(src: &str) -> String {
        let tree = parse_to_syntax_tree(src);
        let result = lower(FileId(0), &tree);
        let module = result.module.expect("Failed to lower module");
        pretty_module(&module)
    }

    // === Insta Snapshot Tests ===
    // These tests capture the pretty-printed output for validation

    #[test]
    fn snapshot_simple_spec() {
        let src = r#"---- MODULE Counter ----
VARIABLE count

Init == count = 0

Increment == count' = count + 1

Decrement == count' = count - 1

Next == Increment \/ Decrement
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_quantifiers_and_sets() {
        let src = r#"---- MODULE Sets ----
S == {1, 2, 3}
T == {x \in S : x > 1}
U == \A x \in S : x > 0
V == \E x \in S : x = 2
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_functions_and_records() {
        let src = r#"---- MODULE FuncsRecords ----
f == [x \in S |-> x + 1]
r == [a |-> 1, b |-> 2]
g == [S -> T]
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn test_pretty_simple_module() {
        let src = r#"---- MODULE Test ----
VARIABLE x

Init == x = 0

Next == x' = x + 1
===="#;
        let result = roundtrip(src);
        assert!(result.contains("MODULE Test"));
        assert!(result.contains("VARIABLE x"));
        assert!(result.contains("Init == x = 0"));
        assert!(result.contains("x' = x + 1"));
    }

    #[test]
    fn test_pretty_quantifiers() {
        let src = r#"---- MODULE Test ----
P == \A x \in S : \E y \in T : x = y
===="#;
        let result = roundtrip(src);
        assert!(result.contains("\\A x \\in S : \\E y \\in T : x = y"));
    }

    #[test]
    fn test_pretty_sets() {
        let src = r#"---- MODULE Test ----
S == {1, 2, 3}
T == {x \in S : x > 1}
===="#;
        let result = roundtrip(src);
        assert!(result.contains("{1, 2, 3}"));
        assert!(result.contains("{x \\in S : x > 1}"));
    }

    #[test]
    fn test_pretty_function_def() {
        let src = r#"---- MODULE Test ----
f == [x \in S |-> x + 1]
===="#;
        let result = roundtrip(src);
        assert!(result.contains("[x \\in S |-> x + 1]"));
    }

    #[test]
    fn test_pretty_record() {
        let src = r#"---- MODULE Test ----
r == [a |-> 1, b |-> 2]
===="#;
        let result = roundtrip(src);
        assert!(result.contains("[a |-> 1, b |-> 2]"));
    }

    #[test]
    fn test_pretty_tuple() {
        let src = r#"---- MODULE Test ----
t == <<1, 2, 3>>
===="#;
        let result = roundtrip(src);
        assert!(result.contains("<<1, 2, 3>>"));
    }

    #[test]
    fn test_pretty_if_then_else() {
        let src = r#"---- MODULE Test ----
P == IF x > 0 THEN y ELSE z
===="#;
        let result = roundtrip(src);
        assert!(result.contains("IF x > 0 THEN y ELSE z"));
    }

    #[test]
    fn test_pretty_temporal() {
        // Simpler temporal test - [][Next]_vars syntax needs parser work
        let src = r#"---- MODULE Test ----
Spec == []P
===="#;
        let result = roundtrip(src);
        assert!(result.contains("[]P"));
    }

    // === Additional Snapshot Tests ===

    #[test]
    fn snapshot_temporal_operators() {
        let src = r#"---- MODULE Temporal ----
VARIABLE x

AlwaysP == []P
EventuallyP == <>P
LeadsToQ == P ~> Q
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_fairness_operators() {
        let src = r#"---- MODULE Fairness ----
VARIABLE x, y

Weak == WF_x(Action)
Strong == SF_y(OtherAction)
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_case_expression() {
        let src = r#"---- MODULE Case ----
Result == CASE x = 1 -> "one"
            [] x = 2 -> "two"
            [] OTHER -> "other"
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_let_expression() {
        let src = r#"---- MODULE Let ----
Compute == LET a == 1
               b == 2
           IN a + b
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_except_expression() {
        let src = r#"---- MODULE Except ----
f == [a |-> 1, b |-> 2]
g == [f EXCEPT !.a = 3]
h == [f EXCEPT ![1] = 42]
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_function_set() {
        let src = r#"---- MODULE FuncSet ----
AllFuncs == [S -> T]
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_record_set() {
        let src = r#"---- MODULE RecordSet ----
AllRecords == [a : S, b : T]
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_cartesian_product() {
        let src = r#"---- MODULE Cartesian ----
Product == S \X T \X U
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_set_operations() {
        let src = r#"---- MODULE SetOps ----
Cup == A \cup B
Cap == A \cap B
Minus == A \ B
Power == SUBSET S
Big == UNION SetOfSets
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_comparison_operators() {
        let src = r#"---- MODULE Compare ----
LT == x < y
LEQ == x <= y
GT == x > y
GEQ == x >= y
NEQ == x /= y
Range == 1..10
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_arithmetic_operators() {
        let src = r#"---- MODULE Arith ----
Add == x + y
Sub == x - y
Mul == x * y
Div == x / y
IDiv == x \div y
Mod == x % y
Pow == x^y
Neg == -x
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_logic_operators() {
        let src = r#"---- MODULE Logic ----
Conj == P /\ Q
Disj == P \/ Q
Neg == ~P
Impl == P => Q
Equiv == P <=> Q
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_action_operators() {
        let src = r#"---- MODULE Actions ----
VARIABLE x

En == ENABLED Action
Unch == UNCHANGED x
Prime == x'
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_choose_expression() {
        let src = r#"---- MODULE Choose ----
Min == CHOOSE x \in S : \A y \in S : x <= y
===="#;
        assert_snapshot!(roundtrip(src));
    }

    #[test]
    fn snapshot_domain_expression() {
        let src = r#"---- MODULE Domain ----
Keys == DOMAIN f
===="#;
        assert_snapshot!(roundtrip(src));
    }
}
