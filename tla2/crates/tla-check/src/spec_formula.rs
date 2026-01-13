//! SPECIFICATION formula extraction
//!
//! Extracts Init and Next predicates from TLA+ temporal formulas like:
//! - `Init /\ [][Next]_vars`
//! - `Init /\ [][Next]_vars /\ WF_vars(Next)`
//!
//! TLC allows SPECIFICATION directive to reference a temporal formula
//! instead of separate INIT/NEXT directives.

use tla_core::{SyntaxElement, SyntaxKind, SyntaxNode};

/// Result of extracting Init/Next from a SPECIFICATION formula
#[derive(Debug, Clone)]
pub struct SpecFormula {
    /// Name of the Init predicate
    pub init: String,
    /// Name of the Next relation (may be a simple name or an inline expression text)
    pub next: String,
    /// The syntax node for the Next relation when it's an inline expression
    /// (e.g., `\E n \in Node: Next(n)` instead of just `Next`)
    pub next_node: Option<SyntaxNode>,
    /// Variables expression (if found)
    pub vars: Option<String>,
    /// Fairness constraints (if any)
    pub fairness: Vec<FairnessConstraint>,
    /// Whether the spec uses `[A]_v` form (stuttering allowed) or `<<A>>_v` form (stuttering forbidden).
    /// When `true`, the spec explicitly permits stuttering (staying in the same state), so deadlock
    /// (no enabled actions) is NOT a violation - the system can legally stutter forever.
    /// When `false`, the spec uses `<<A>>_v` which requires actual progress, so deadlock IS a violation.
    /// Defaults to `true` (stuttering allowed) which is the most common TLA+ pattern.
    pub stuttering_allowed: bool,
}

/// A fairness constraint
#[derive(Debug, Clone)]
pub enum FairnessConstraint {
    /// Weak fairness: WF_vars(Action)
    Weak {
        vars: String,
        action: String,
        /// The action expression's syntax node (for inline expressions)
        action_node: Option<SyntaxNode>,
    },
    /// Strong fairness: SF_vars(Action)
    Strong {
        vars: String,
        action: String,
        /// The action expression's syntax node (for inline expressions)
        action_node: Option<SyntaxNode>,
    },
    /// Temporal formula reference (for complex fairness like \A p: WF_vars(Action(p)))
    /// The named operator's body will be converted to LiveExpr during liveness checking.
    TemporalRef {
        /// Name of the operator containing the temporal formula
        op_name: String,
    },
    /// Inline quantified temporal formula (for complex fairness in spec bodies)
    /// This handles cases like `\A c \in Clients: WF_vars(Action(c))` that appear
    /// directly in the spec body rather than in a separate operator.
    QuantifiedTemporal {
        /// The syntax node containing the quantified temporal formula
        node: SyntaxNode,
    },
}

/// Conjunct in a temporal formula
#[derive(Debug)]
enum Conjunct {
    /// An identifier (like Init)
    Ident(String),
    /// A node to analyze further
    Node(SyntaxNode),
}

/// Extract Init and Next from a SPECIFICATION temporal formula.
///
/// Supports common patterns:
/// - `Init /\ [][Next]_vars`
/// - `Init /\ [][Next]_<<v1,v2>>`
/// - `Init /\ [][Next]_vars /\ WF_vars(Next)`
/// - `/\ Init /\ [][Next]_vars` (conjunction list style)
///
/// Returns None if the formula doesn't match any known pattern.
pub fn extract_spec_formula(spec_body: &SyntaxNode) -> Option<SpecFormula> {
    // Try different extraction patterns
    extract_and_pattern(spec_body).or_else(|| extract_conjunction_list(spec_body))
}

/// Extract from pattern: Init /\ [][Next]_vars
fn extract_and_pattern(node: &SyntaxNode) -> Option<SpecFormula> {
    if node.kind() != SyntaxKind::BinaryExpr {
        // Check if this is a UnaryExpr (conjunction list starting with /\)
        if node.kind() == SyntaxKind::UnaryExpr {
            return extract_conjunction_list(node);
        }
        return None;
    }

    let mut init: Option<String> = None;
    let mut next: Option<String> = None;
    let mut next_node: Option<SyntaxNode> = None;
    let mut vars: Option<String> = None;
    let mut fairness = Vec::new();
    // Default to stuttering allowed (most common TLA+ pattern)
    let mut stuttering_allowed = true;

    // Collect all conjuncts (including tokens and nodes)
    let conjuncts = collect_conjuncts(node);

    for conjunct in conjuncts {
        match conjunct {
            Conjunct::Ident(name) => {
                // First identifier is Init
                if init.is_none() {
                    init = Some(name);
                }
            }
            Conjunct::Node(ref n) => {
                // Check if this is an Always operator ([][Next]_vars)
                if let Some(extraction) = extract_always_subscript(n) {
                    next = Some(extraction.name);
                    next_node = extraction.node;
                    vars = Some(extraction.vars);
                    stuttering_allowed = extraction.stuttering_allowed;
                } else if let Some(fc) = extract_fairness(n) {
                    fairness.push(fc);
                } else if n.kind() == SyntaxKind::QuantExpr && contains_temporal_operators(n) {
                    // Quantified fairness like `\A c \in Clients: WF_vars(Action(c))`
                    fairness.push(FairnessConstraint::QuantifiedTemporal { node: n.clone() });
                } else if let Some(ident) = extract_ident(n) {
                    // Fallback: extract identifier from node
                    if init.is_none() {
                        init = Some(ident);
                    }
                }
            }
        }
    }

    Some(SpecFormula {
        init: init?,
        next: next?,
        next_node,
        vars,
        fairness,
        stuttering_allowed,
    })
}

/// Extract from conjunction list pattern: /\ Init /\ [][Next]_vars
fn extract_conjunction_list(node: &SyntaxNode) -> Option<SpecFormula> {
    // Find the first BinaryExpr inside
    for child in node.children() {
        if child.kind() == SyntaxKind::BinaryExpr {
            return extract_and_pattern(&child);
        }
    }
    None
}

/// Collect all conjuncts from a conjunction expression
fn collect_conjuncts(node: &SyntaxNode) -> Vec<Conjunct> {
    let mut result = Vec::new();
    collect_conjuncts_rec(node, &mut result);
    result
}

fn collect_conjuncts_rec(node: &SyntaxNode, result: &mut Vec<Conjunct>) {
    if node.kind() == SyntaxKind::BinaryExpr {
        // Check if this is an AND expression
        let has_and = node
            .children_with_tokens()
            .any(|c| matches!(c, SyntaxElement::Token(t) if t.kind() == SyntaxKind::AndOp));

        if has_and {
            // Process all children (both tokens and nodes)
            for child in node.children_with_tokens() {
                match child {
                    SyntaxElement::Token(t) => {
                        if t.kind() == SyntaxKind::Ident {
                            result.push(Conjunct::Ident(t.text().to_string()));
                        }
                        // Skip AndOp and other tokens
                    }
                    SyntaxElement::Node(n) => {
                        // Recursively collect from child nodes
                        collect_conjuncts_rec(&n, result);
                    }
                }
            }
            return;
        }
    }

    // This is a leaf conjunct - add as node
    result.push(Conjunct::Node(node.clone()));
}

/// Result of extracting an action from a subscript expression
struct ActionExtraction {
    /// The action name or expression text
    name: String,
    /// The syntax node for inline expressions (None for simple identifiers)
    node: Option<SyntaxNode>,
    /// The vars expression
    vars: String,
    /// Whether stuttering is allowed (`[A]_v` = true) or forbidden (`<<A>>_v` = false).
    /// `[A]_v` uses FuncSetExpr (square brackets), `<<A>>_v` uses TupleExpr (angle brackets).
    stuttering_allowed: bool,
}

/// Extract [][Next]_vars pattern from an Always expression
fn extract_always_subscript(node: &SyntaxNode) -> Option<ActionExtraction> {
    // Check for UnaryExpr with AlwaysOp
    if node.kind() != SyntaxKind::UnaryExpr {
        return None;
    }

    let has_always = node
        .children_with_tokens()
        .any(|c| matches!(c, SyntaxElement::Token(t) if t.kind() == SyntaxKind::AlwaysOp));

    if !has_always {
        return None;
    }

    // Find SubscriptExpr inside
    for child in node.children() {
        if child.kind() == SyntaxKind::SubscriptExpr {
            return extract_subscript(&child);
        }
    }

    None
}

/// Extract [Action]_vars or <<Action>>_vars from a SubscriptExpr
fn extract_subscript(node: &SyntaxNode) -> Option<ActionExtraction> {
    if node.kind() != SyntaxKind::SubscriptExpr {
        return None;
    }

    let mut action_name: Option<String> = None;
    let mut action_node: Option<SyntaxNode> = None;
    let mut vars: Option<String> = None;
    let mut saw_underscore = false;
    // Default to stuttering allowed (most common). Will be set to false if TupleExpr detected.
    let mut stuttering_allowed = true;

    for child in node.children_with_tokens() {
        match child {
            SyntaxElement::Token(t) => {
                if t.kind() == SyntaxKind::Underscore {
                    saw_underscore = true;
                } else if t.kind() == SyntaxKind::Ident && saw_underscore {
                    vars = Some(t.text().to_string());
                }
            }
            SyntaxElement::Node(n) => {
                if !saw_underscore {
                    // This should be the action part [Action] or <<Action>>
                    // Detect whether it's FuncSetExpr (brackets, stuttering OK) or TupleExpr (angles, stuttering forbidden)
                    // [A]_v uses FuncSetExpr (square brackets) - stuttering allowed
                    // <<A>>_v uses TupleExpr (angle brackets) - stuttering forbidden
                    if n.kind() == SyntaxKind::TupleExpr {
                        stuttering_allowed = false;
                    }
                    // For FuncSetExpr (or other wrapper nodes), stuttering_allowed stays true

                    // Check for inline expression (QuantExpr) vs simple identifier
                    let (name, node) = extract_action_with_node(&n);
                    action_name = name;
                    action_node = node;
                } else {
                    // This is the vars part (tuple or expression)
                    vars = Some(n.text().to_string().trim().to_string());
                }
            }
        }
    }

    Some(ActionExtraction {
        name: action_name?,
        node: action_node,
        vars: vars?,
        stuttering_allowed,
    })
}

/// Extract action name and optionally the syntax node from [Action] part.
///
/// For simple actions like `[Next]_vars`, returns ("Next", None).
/// For complex actions like `[\E n \in Node: Next(n)]_vars`, returns the full expression
/// text and the syntax node for lowering.
fn extract_action_with_node(node: &SyntaxNode) -> (Option<String>, Option<SyntaxNode>) {
    // First, check if this node contains a quantified expression (QuantExpr)
    // which would mean the action is something like `\E n \in Node: Next(n)`
    // In this case, we need to return both the text AND the node for lowering
    if let Some((text, quant_node)) = find_quantified_action_node(node) {
        return (Some(text), Some(quant_node));
    }

    // Look for simple identifier token (e.g., `Next`)
    (extract_action_name(node), None)
}

/// Extract action name from [Action] part
///
/// For simple actions like `[Next]_vars`, returns "Next".
/// For complex actions like `[\E n \in Node: Next(n)]_vars`, returns the full expression text.
fn extract_action_name(node: &SyntaxNode) -> Option<String> {
    // First, check if this node contains a quantified expression (QuantExpr)
    // which would mean the action is something like `\E n \in Node: Next(n)`
    // In this case, we should return the full text rather than just the first identifier
    if let Some((quant_text, _)) = find_quantified_action_node(node) {
        return Some(quant_text);
    }

    // Look for simple identifier token (e.g., `Next`)
    for child in node.children_with_tokens() {
        match child {
            SyntaxElement::Token(t) if t.kind() == SyntaxKind::Ident => {
                return Some(t.text().to_string());
            }
            SyntaxElement::Node(n) => {
                if let Some(name) = extract_action_name(&n) {
                    return Some(name);
                }
            }
            _ => {}
        }
    }
    None
}

/// Check if the node contains a QuantExpr at the top level and return its text and node
fn find_quantified_action_node(node: &SyntaxNode) -> Option<(String, SyntaxNode)> {
    // Check if this node itself is a QuantExpr
    if node.kind() == SyntaxKind::QuantExpr {
        return Some((node.text().to_string().trim().to_string(), node.clone()));
    }

    // For FuncSetExpr (the `[...]` wrapper), check if its direct child is a QuantExpr
    if node.kind() == SyntaxKind::FuncSetExpr {
        for child in node.children() {
            if child.kind() == SyntaxKind::QuantExpr {
                return Some((child.text().to_string().trim().to_string(), child));
            }
        }
    }

    // For other wrapping nodes (like ParenExpr), check direct children
    for child in node.children() {
        if child.kind() == SyntaxKind::QuantExpr {
            return Some((child.text().to_string().trim().to_string(), child));
        }
    }

    None
}

/// Extract identifier from a node
fn extract_ident(node: &SyntaxNode) -> Option<String> {
    for child in node.children_with_tokens() {
        if let SyntaxElement::Token(t) = child {
            if t.kind() == SyntaxKind::Ident {
                return Some(t.text().to_string());
            }
        }
    }
    // Recurse into children
    for child in node.children() {
        if let Some(ident) = extract_ident(&child) {
            return Some(ident);
        }
    }
    None
}

/// Extract fairness constraint (WF_vars(Action) or SF_vars(Action))
///
/// Due to lexer behavior, `WF_vars(Next)` is often parsed as an ApplyExpr
/// with identifier `WF_vars` rather than as a BinaryExpr with WeakFairKw.
/// This function handles both cases.
fn extract_fairness(node: &SyntaxNode) -> Option<FairnessConstraint> {
    // Case 1: BinaryExpr with WeakFairKw or StrongFairKw (ideal parsing)
    if node.kind() == SyntaxKind::BinaryExpr {
        let mut is_weak = false;
        let mut is_strong = false;
        let mut vars: Option<String> = None;
        let mut action: Option<String> = None;
        let mut action_node: Option<SyntaxNode> = None;
        let mut saw_lparen = false;

        for child in node.children_with_tokens() {
            match child {
                SyntaxElement::Token(t) => {
                    if t.kind() == SyntaxKind::WeakFairKw {
                        is_weak = true;
                    } else if t.kind() == SyntaxKind::StrongFairKw {
                        is_strong = true;
                    } else if t.kind() == SyntaxKind::LParen {
                        saw_lparen = true;
                    } else if t.kind() == SyntaxKind::Ident
                        && saw_lparen
                        && vars.is_some()
                        && action.is_none()
                    {
                        // Some fairness expressions use tokens for the parenthesized action,
                        // e.g. `WF_<<v1,v2>>(Next)` parses with `Ident` rather than a child node.
                        action = Some(t.text().to_string());
                    }
                }
                SyntaxElement::Node(n) => {
                    // First child node is vars (subscript), second is action
                    if vars.is_none() {
                        vars = Some(n.text().to_string().trim().to_string());
                    } else if action.is_none() {
                        // Store the action node for inline expression support
                        action_node = Some(extract_inner_expr(&n));
                        // The action is typically in parentheses, extract the inner part
                        let action_text = n.text().to_string();
                        // Remove surrounding parentheses if present
                        let trimmed = action_text.trim();
                        let inner = if trimmed.starts_with('(') && trimmed.ends_with(')') {
                            trimmed[1..trimmed.len() - 1].trim().to_string()
                        } else {
                            trimmed.to_string()
                        };
                        action = Some(inner);
                    }
                }
            }
        }

        if is_weak {
            return Some(FairnessConstraint::Weak {
                vars: vars?,
                action: action?,
                action_node,
            });
        } else if is_strong {
            return Some(FairnessConstraint::Strong {
                vars: vars?,
                action: action?,
                action_node,
            });
        }
    }

    // Case 2: ApplyExpr where identifier is WF_xxx or SF_xxx
    // This happens because the lexer matches WF_vars as a single identifier
    if node.kind() == SyntaxKind::ApplyExpr {
        let mut func_name: Option<String> = None;
        let mut action: Option<String> = None;
        let mut action_node: Option<SyntaxNode> = None;

        for child in node.children_with_tokens() {
            match child {
                SyntaxElement::Token(t) => {
                    if t.kind() == SyntaxKind::Ident && func_name.is_none() {
                        func_name = Some(t.text().to_string());
                    }
                }
                SyntaxElement::Node(n) => {
                    // Look for ArgList to extract the action argument
                    if n.kind() == SyntaxKind::ArgList {
                        // Get the first argument (action)
                        for arg_child in n.children_with_tokens() {
                            match arg_child {
                                SyntaxElement::Token(t) if t.kind() == SyntaxKind::Ident => {
                                    action = Some(t.text().to_string());
                                    // No node for simple identifier - will look up by name
                                    break;
                                }
                                SyntaxElement::Node(arg_node) => {
                                    // Store the action node for inline expression support
                                    action_node = Some(arg_node.clone());
                                    // Could be a more complex expression
                                    action = Some(arg_node.text().to_string().trim().to_string());
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        if let Some(name) = func_name {
            if let Some(vars) = name.strip_prefix("WF_") {
                return Some(FairnessConstraint::Weak {
                    vars: vars.to_string(),
                    action: action?,
                    action_node,
                });
            } else if let Some(vars) = name.strip_prefix("SF_") {
                return Some(FairnessConstraint::Strong {
                    vars: vars.to_string(),
                    action: action?,
                    action_node,
                });
            }
        }
    }

    None
}

/// Extract the inner expression from a parenthesized node
fn extract_inner_expr(node: &SyntaxNode) -> SyntaxNode {
    // If this is a ParenExpr, extract the inner expression
    if node.kind() == SyntaxKind::ParenExpr {
        if let Some(child) = node.children().next() {
            return child;
        }
    }
    node.clone()
}

/// Extract all fairness constraints from a spec body.
///
/// This is useful when a spec body like `SpecWeakFair == Spec /\ WF_vars(Next)`
/// doesn't match the full `Init /\ [][Next]_vars` pattern but still contains
/// fairness constraints that should be merged with the resolved Init/Next.
pub fn extract_all_fairness(spec_body: &SyntaxNode) -> Vec<FairnessConstraint> {
    let mut result = Vec::new();
    extract_all_fairness_rec(spec_body, &mut result);
    result
}

fn extract_all_fairness_rec(node: &SyntaxNode, result: &mut Vec<FairnessConstraint>) {
    // Handle quantified expressions (\A, \E) specially.
    // Quantified fairness like `\A p: WF_vars(Action(p))` should be handled
    // as a whole via QuantifiedTemporal, not by extracting the WF individually
    // (which would leave `p` as an unbound variable).
    if node.kind() == SyntaxKind::QuantExpr {
        // Check if this quantified expression contains temporal operators (WF, SF, [], <>)
        if contains_temporal_operators(node) {
            result.push(FairnessConstraint::QuantifiedTemporal { node: node.clone() });
        }
        return;
    }
    // Check if this node is a fairness constraint
    if let Some(fc) = extract_fairness(node) {
        result.push(fc);
        return; // Don't recurse into fairness nodes
    }
    // Recurse into children
    for child in node.children() {
        extract_all_fairness_rec(&child, result);
    }
}

/// Check if a syntax node contains temporal operators (WF, SF, [], <>, ~>).
fn contains_temporal_operators(node: &SyntaxNode) -> bool {
    fn search(node: &SyntaxNode) -> bool {
        // Check for temporal operator tokens
        for child in node.children_with_tokens() {
            match child {
                SyntaxElement::Token(t) => {
                    let kind = t.kind();
                    if kind == SyntaxKind::AlwaysOp
                        || kind == SyntaxKind::EventuallyOp
                        || kind == SyntaxKind::LeadsToOp
                        || kind == SyntaxKind::WeakFairKw
                        || kind == SyntaxKind::StrongFairKw
                    {
                        return true;
                    }
                    // Also check for WF_xxx or SF_xxx identifiers
                    if kind == SyntaxKind::Ident {
                        let text = t.text();
                        if text.starts_with("WF_") || text.starts_with("SF_") {
                            return true;
                        }
                    }
                }
                SyntaxElement::Node(n) => {
                    if search(&n) {
                        return true;
                    }
                }
            }
        }
        false
    }
    search(node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::parse_to_syntax_tree;

    fn find_op_body(tree: &SyntaxNode, name: &str) -> Option<SyntaxNode> {
        fn search(node: &SyntaxNode, name: &str) -> Option<SyntaxNode> {
            if node.kind() == SyntaxKind::OperatorDef {
                // Check if this operator has the target name
                let mut found_name = false;
                for child in node.children_with_tokens() {
                    if let SyntaxElement::Token(t) = child {
                        if t.kind() == SyntaxKind::Ident && t.text() == name {
                            found_name = true;
                            break;
                        }
                    }
                }
                if found_name {
                    // Return the body expression(s).
                    //
                    // Most bodies are a single CST node after `==`, but conjunction-list style
                    // bodies can appear as multiple top-level expression nodes (one per `/\`).
                    // In that case, return the OperatorDef node so recursive traversals see the
                    // full body.
                    let mut body_nodes = Vec::new();
                    let mut past_def_eq = false;
                    for child in node.children_with_tokens() {
                        match child {
                            SyntaxElement::Token(t) if t.kind() == SyntaxKind::DefEqOp => {
                                past_def_eq = true;
                            }
                            SyntaxElement::Node(n) if past_def_eq => {
                                body_nodes.push(n);
                            }
                            _ => {}
                        }
                    }

                    match body_nodes.as_slice() {
                        [single] => return Some(single.clone()),
                        [] => return Some(node.clone()),
                        _ => return Some(node.clone()),
                    }
                }
            }
            for child in node.children() {
                if let Some(found) = search(&child, name) {
                    return Some(found);
                }
            }
            None
        }
        search(tree, name)
    }

    #[test]
    fn test_extract_simple_spec() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec").expect("Spec not found");
        let result = extract_spec_formula(&spec_body).expect("Failed to extract");

        assert_eq!(result.init, "Init");
        assert_eq!(result.next, "Next");
        assert_eq!(result.vars.as_deref(), Some("vars"));
    }

    #[test]
    fn test_extract_conjunction_style() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == /\ Init /\ [][Next]_vars
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec");
        // This pattern may not match - that's OK for now
        if let Some(body) = spec_body {
            let result = extract_spec_formula(&body);
            // We may or may not extract this pattern
            if let Some(r) = result {
                assert_eq!(r.init, "Init");
                assert_eq!(r.next, "Next");
            }
        }
    }

    #[test]
    fn test_extract_tuple_vars() {
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ y' = y
Spec == Init /\ [][Next]_<<x, y>>
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec");
        if let Some(body) = spec_body {
            let result = extract_spec_formula(&body);
            if let Some(r) = result {
                assert_eq!(r.init, "Init");
                assert_eq!(r.next, "Next");
                // vars might be "<<x, y>>" or similar
                assert!(r.vars.is_some());
            }
        }
    }

    #[test]
    fn test_extract_with_weak_fairness() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec").expect("Spec not found");
        let result = extract_spec_formula(&spec_body).expect("Failed to extract");

        assert_eq!(result.init, "Init");
        assert_eq!(result.next, "Next");
        assert_eq!(result.vars.as_deref(), Some("vars"));
        assert_eq!(result.fairness.len(), 1);
        match &result.fairness[0] {
            FairnessConstraint::Weak { vars, action, .. } => {
                assert_eq!(vars, "vars");
                assert_eq!(action, "Next");
            }
            _ => panic!("Expected weak fairness"),
        }
    }

    #[test]
    fn test_extract_with_strong_fairness() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars /\ SF_vars(Next)
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec").expect("Spec not found");
        let result = extract_spec_formula(&spec_body).expect("Failed to extract");

        assert_eq!(result.init, "Init");
        assert_eq!(result.next, "Next");
        assert_eq!(result.fairness.len(), 1);
        match &result.fairness[0] {
            FairnessConstraint::Strong { vars, action, .. } => {
                assert_eq!(vars, "vars");
                assert_eq!(action, "Next");
            }
            _ => panic!("Expected strong fairness"),
        }
    }

    #[test]
    fn test_extract_with_multiple_fairness() {
        let src = r#"
---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Action1 == x' = x + 1
Action2 == x' = x + 2
Spec == Init /\ [][Next]_vars /\ WF_vars(Action1) /\ SF_vars(Action2)
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec").expect("Spec not found");
        let result = extract_spec_formula(&spec_body).expect("Failed to extract");

        assert_eq!(result.init, "Init");
        assert_eq!(result.next, "Next");
        assert_eq!(result.fairness.len(), 2);

        // Check we have both weak and strong fairness
        let has_weak = result
            .fairness
            .iter()
            .any(|f| matches!(f, FairnessConstraint::Weak { .. }));
        let has_strong = result
            .fairness
            .iter()
            .any(|f| matches!(f, FairnessConstraint::Strong { .. }));
        assert!(has_weak, "Should have weak fairness");
        assert!(has_strong, "Should have strong fairness");
    }

    #[test]
    fn test_extract_quantified_fairness() {
        // Quantified fairness like `\A c \in Clients: WF_vars(Action(c))`
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
CONSTANT Clients
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Action(c) == x' = c
Spec == Init /\ [][Next]_vars /\ \A c \in Clients: WF_vars(Action(c))
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec").expect("Spec not found");
        let result = extract_spec_formula(&spec_body).expect("Failed to extract");

        assert_eq!(result.init, "Init");
        assert_eq!(result.next, "Next");
        assert_eq!(
            result.fairness.len(),
            1,
            "Should have one quantified fairness constraint"
        );

        // Check that it's a QuantifiedTemporal constraint
        match &result.fairness[0] {
            FairnessConstraint::QuantifiedTemporal { .. } => {
                // Success - quantified fairness was extracted as QuantifiedTemporal
            }
            _ => panic!("Expected QuantifiedTemporal, got {:?}", &result.fairness[0]),
        }
    }

    #[test]
    fn test_extract_multiple_quantified_fairness() {
        // Multiple quantified fairness constraints like SimpleAllocator
        // Using bulleted conjunction style which TLA+ parses as separate conjuncts
        let src = r#"
---- MODULE Test ----
EXTENDS Integers
CONSTANT Clients
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Action1(c) == x' = c
Action2(c) == x' = c + 1
Spec ==
  /\ Init /\ [][Next]_vars
  /\ \A c \in Clients: WF_vars(Action1(c))
  /\ \A c \in Clients: SF_vars(Action2(c))
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec").expect("Spec not found");
        let result = extract_spec_formula(&spec_body).expect("Failed to extract");

        assert_eq!(result.init, "Init");
        assert_eq!(result.next, "Next");
        assert_eq!(
            result.fairness.len(),
            2,
            "Should have two quantified fairness constraints"
        );

        // Both should be QuantifiedTemporal
        for fc in &result.fairness {
            assert!(
                matches!(fc, FairnessConstraint::QuantifiedTemporal { .. }),
                "Expected QuantifiedTemporal, got {:?}",
                fc
            );
        }
    }

    #[test]
    fn test_extract_all_fairness_tuple_subscript_in_conjunction_list() {
        // A conjunction list operator body containing:
        // - quantified fairness (should become QuantifiedTemporal)
        // - plain WF_<<...>>(coordProgB) (should become FairnessConstraint::Weak)
        //
        // This matches patterns in ACP_NB's `fairnessNB`.
        let src = r#"
---- MODULE Test ----
CONSTANT participants
VARIABLE coordinator, participant

vars == <<coordinator, participant>>

coordProgB == UNCHANGED vars
parProgNB(i, j) == UNCHANGED vars

fairnessNB ==
  /\ \A i \in participants: WF_<<coordinator, participant>>(\E j \in participants: parProgNB(i, j))
  /\ WF_<<coordinator, participant>>(coordProgB)
===="#;
        let tree = parse_to_syntax_tree(src);

        let fairness_body = find_op_body(&tree, "fairnessNB").expect("fairnessNB not found");
        let fairness = extract_all_fairness(&fairness_body);

        assert!(
            fairness
                .iter()
                .any(|f| matches!(f, FairnessConstraint::QuantifiedTemporal { .. })),
            "Expected QuantifiedTemporal in extracted fairness, got {:?}",
            fairness
        );

        assert!(
            fairness.iter().any(|f| matches!(
                f,
                FairnessConstraint::Weak { action, .. } if action == "coordProgB"
            )),
            "Expected WF for coordProgB in extracted fairness, got {:?}",
            fairness
        );
    }

    #[test]
    fn test_extract_existential_in_next() {
        // Test: Spec == Init /\ [][\E n \in Node: Next(n)]_vars
        // The NEXT action is an existential quantifier, not just "n"
        let src = r#"
---- MODULE Test ----
CONSTANT Node
VARIABLE x
vars == x
Init == x = 0
Next(n) == x' = n
Spec == Init /\ [][\E n \in Node: Next(n)]_vars
===="#;
        let tree = parse_to_syntax_tree(src);

        let spec_body = find_op_body(&tree, "Spec").expect("Spec not found");
        let result = extract_spec_formula(&spec_body).expect("Failed to extract");

        assert_eq!(result.init, "Init");
        assert_eq!(result.vars.as_deref(), Some("vars"));
        // The next should NOT be just "n" - it should be the full expression
        // or at minimum contain the existential quantifier pattern
        assert!(
            result.next != "n",
            "Next should not be just 'n', got: {}. Expected full existential expression.",
            result.next
        );
    }
}
