use anyhow::{anyhow, Context, Result};
use lean5_elab::ElabCtx;
use lean5_kernel::expr::{BinderInfo, Expr};
use lean5_kernel::level::Level;
use lean5_kernel::Environment;
use lean5_parser::parse_expr;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use tempfile::NamedTempFile;

const EXPRESSIONS_PATH: &str = "../../tests/differential/expressions.txt";

#[test]
fn differential_against_lean4() -> Result<()> {
    let expressions = load_expressions()?;
    let lean4_types = run_lean4(&expressions)?;
    let lean5_types = infer_with_lean5(&expressions)?;

    if lean4_types.len() != expressions.len() {
        return Err(anyhow!(
            "Lean4 produced {} results for {} expressions",
            lean4_types.len(),
            expressions.len()
        ));
    }
    if lean5_types.len() != expressions.len() {
        return Err(anyhow!(
            "Lean5 produced {} results for {} expressions",
            lean5_types.len(),
            expressions.len()
        ));
    }

    for (idx, ((expr, l4), l5)) in expressions
        .iter()
        .zip(lean4_types.iter())
        .zip(lean5_types.iter())
        .enumerate()
    {
        if l4 != l5 {
            return Err(anyhow!(
                "Mismatch at #{idx} for `{expr}`\n  lean4: {l4}\n  lean5: {l5}"
            ));
        }
    }

    Ok(())
}

fn load_expressions() -> Result<Vec<String>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(EXPRESSIONS_PATH);
    let content = fs::read_to_string(&path)
        .with_context(|| format!("failed to read expressions file at {}", path.display()))?;
    let mut expressions = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        expressions.push(trimmed.to_string());
    }
    if expressions.is_empty() {
        return Err(anyhow!("no expressions loaded from {}", path.display()));
    }
    Ok(expressions)
}

fn run_lean4(expressions: &[String]) -> Result<Vec<String>> {
    let mut file = NamedTempFile::new().context("failed to create temp lean file")?;
    writeln!(file, "set_option linter.unusedVariables false")?;
    writeln!(file, "set_option pp.universes true")?;
    for expr in expressions {
        writeln!(file, "#check {expr}")?;
    }

    let output = Command::new("lean")
        .arg(file.path())
        .output()
        .context("failed to spawn lean (Lean 4)")?;
    if !output.status.success() {
        return Err(anyhow!(
            "lean exited with {}: {}\nstdout:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut results = Vec::new();
    for line in stdout.lines() {
        // Find the separator " : " where preceding parens are balanced
        // This handles `(A : Type) → A : Type 1` correctly
        let type_part = extract_type_from_check_output(line);
        let normalized = normalize_type_str(type_part.trim());
        if !normalized.is_empty() {
            results.push(normalized);
        }
    }
    Ok(results)
}

/// Extract the type part from Lean's `#check` output.
/// Output format: `expr : type`, but expr can contain ` : ` in binder annotations.
/// We find ` : ` where the preceding text has balanced parentheses.
fn extract_type_from_check_output(line: &str) -> &str {
    let bytes = line.as_bytes();
    let separator = b" : ";
    let mut paren_depth: i32 = 0;
    let mut i = 0;

    while i + 3 <= bytes.len() {
        match bytes[i] {
            b'(' => paren_depth += 1,
            b')' => paren_depth = paren_depth.saturating_sub(1),
            b' ' if paren_depth == 0 && &bytes[i..i + 3] == separator => {
                return &line[i + 3..];
            }
            _ => {}
        }
        i += 1;
    }

    // Fallback: return entire line
    line
}

fn infer_with_lean5(expressions: &[String]) -> Result<Vec<String>> {
    let mut results = Vec::with_capacity(expressions.len());
    for expr in expressions {
        let ty_str =
            infer_single(expr).with_context(|| format!("failed to infer type for `{expr}`"))?;
        results.push(ty_str);
    }
    Ok(results)
}

fn infer_single(expr_str: &str) -> Result<String> {
    let surface = parse_expr(expr_str)?;
    let env = Environment::new();
    let mut ctx = ElabCtx::new(&env);
    let kernel_expr = ctx.elaborate(&surface)?;
    let mut tc = lean5_kernel::TypeChecker::new(&env);
    let ty = tc.infer_type(&kernel_expr)?;
    Ok(normalize_type_str(&format_expr(&ty)))
}

fn normalize_type_str(s: &str) -> String {
    let mut result = s
        .replace('→', "->")
        .replace('∀', "forall") // No trailing space - will be normalized later
        .to_lowercase();

    // Collapse whitespace first
    result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    // First: expand combined forall binders "forall (a b : T)," -> "forall (a : T), forall (b : T),"
    result = expand_forall_combined_binders(&result);

    // Handle sequences of forall binders: "forall (a : T) (b : U)," -> "forall (a : T), forall (b : U),"
    result = expand_forall_binder_sequences(&result);

    // Then: normalize "forall (x : T)," to "(x : T) ->" for consistent format
    loop {
        if let Some(start) = result.find("forall (") {
            let after_forall = &result[start + 8..]; // skip "forall ("
            if let Some(end) = find_balanced_paren_comma_inner(after_forall) {
                let binder_content = &after_forall[..end];
                let rest = after_forall[end + 2..].trim_start();
                let replacement = format!("({binder_content}) -> {rest}");
                result = format!("{}{}", &result[..start], replacement);
                continue;
            }
        }
        break;
    }

    // Then: expand remaining combined binders outside forall context
    result = expand_combined_binders(&result);

    // Final whitespace normalization
    result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    // Canonicalize variable names to make comparison order-independent
    canonicalize_binder_names(&result)
}

/// Expand "forall (a : T) (b : U)," -> "forall (a : T), forall (b : U),"
fn expand_forall_binder_sequences(s: &str) -> String {
    let mut result = s.to_string();

    loop {
        // Find "forall (binder1) (binder2)" pattern
        if let Some(start) = result.find("forall (") {
            let after_forall = &result[start + 7..]; // skip "forall "
                                                     // Collect all consecutive (binder) groups until we hit ","
            let mut binders = Vec::new();
            let mut pos = 0;
            let bytes = after_forall.as_bytes();

            while pos < bytes.len() {
                // Skip whitespace
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }

                if pos >= bytes.len() {
                    break;
                }

                if bytes[pos] == b'(' {
                    // Find matching close paren
                    let binder_start = pos;
                    let mut depth = 0;
                    while pos < bytes.len() {
                        if bytes[pos] == b'(' {
                            depth += 1;
                        } else if bytes[pos] == b')' {
                            depth -= 1;
                            if depth == 0 {
                                pos += 1;
                                break;
                            }
                        }
                        pos += 1;
                    }
                    let binder = &after_forall[binder_start..pos];
                    binders.push(binder.to_string());
                } else if bytes[pos] == b',' {
                    // End of binder sequence
                    pos += 1;
                    break;
                } else {
                    // Something else (like body starting), stop
                    break;
                }
            }

            // If we found multiple binders, expand them
            if binders.len() > 1 {
                let expanded: Vec<String> =
                    binders.iter().map(|b| format!("forall {b},")).collect();
                let replacement = expanded.join(" ");
                let rest = &after_forall[pos..].trim_start();
                result = format!("{}{} {}", &result[..start], replacement, rest);
                continue;
            }
        }
        break;
    }

    result
}

/// Expand "forall (a b c : T)," to "forall (a : T), forall (b : T), forall (c : T),"
fn expand_forall_combined_binders(s: &str) -> String {
    let mut result = s.to_string();

    loop {
        if let Some(start) = result.find("forall (") {
            let after_forall = &result[start + 8..]; // skip "forall ("
            if let Some(close) = after_forall.find("),") {
                let content = &after_forall[..close];
                // Check if it has " : " and multiple words before colon
                if let Some(colon_pos) = content.find(" : ") {
                    let before_colon = &content[..colon_pos];
                    let after_colon = &content[colon_pos + 3..];
                    let names: Vec<&str> = before_colon.split_whitespace().collect();
                    if names.len() > 1 {
                        // Expand "forall (a b : T)," -> "forall (a : T), forall (b : T),"
                        let expanded: Vec<String> = names
                            .iter()
                            .map(|n| format!("forall ({n} : {after_colon}),"))
                            .collect();
                        let replacement = expanded.join(" ");
                        let rest = &after_forall[close + 2..];
                        result = format!("{}{} {}", &result[..start], replacement, rest);
                        continue;
                    }
                }
            }
        }
        break;
    }

    result
}

/// Canonicalize binder names to v0, v1, v2, ... for consistent comparison
fn canonicalize_binder_names(s: &str) -> String {
    use std::collections::HashMap;

    let mut result = String::new();
    let mut name_map: HashMap<String, String> = HashMap::new();
    let mut counter = 0;

    // Reserved keywords that shouldn't be renamed
    let reserved: std::collections::HashSet<&str> = ["type", "prop", "sort", "forall", "fun"]
        .iter()
        .copied()
        .collect();

    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check for binder pattern "(<name> :"
        if chars[i] == '(' {
            result.push('(');
            i += 1;

            // Skip whitespace
            while i < chars.len() && chars[i].is_whitespace() {
                result.push(chars[i]);
                i += 1;
            }

            // If next char is '(', this is nested parens - don't try to parse as binder
            if i < chars.len() && chars[i] == '(' {
                // Don't consume the '(' - let the main loop handle it
                continue;
            }

            // Collect the potential binder name (until ' ' or ':' or '-' or ')' or '(')
            let name_start = i;
            while i < chars.len()
                && chars[i] != ' '
                && chars[i] != ':'
                && chars[i] != '-'
                && chars[i] != ')'
                && chars[i] != '('
            {
                i += 1;
            }
            let name: String = chars[name_start..i].iter().collect();

            // Check if this is a binder (followed by " :")
            let rest: String = chars[i..].iter().collect();
            if rest.starts_with(" :") && !name.is_empty() && !reserved.contains(name.as_str()) {
                // This is a binder - ALWAYS give it a fresh canonical name
                // (even if a variable with the same name was seen before, because this shadows it)
                let canonical = format!("v{counter}");
                counter += 1;
                name_map.insert(name.clone(), canonical.clone());
                result.push_str(&canonical);
            } else if !name.is_empty() && !reserved.contains(name.as_str()) {
                // This might be a variable reference inside parens (like in "(b -> c)")
                if let Some(canonical) = name_map.get(&name) {
                    result.push_str(canonical);
                } else {
                    result.push_str(&name);
                }
            } else {
                // Reserved word or empty, output as-is
                result.push_str(&name);
            }
            continue;
        }

        // Check for variable reference (a word not in a binder position)
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let name_start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let name: String = chars[name_start..i].iter().collect();

            // If this name was bound, replace it with canonical name (unless reserved)
            if reserved.contains(name.as_str()) {
                result.push_str(&name);
            } else if let Some(canonical) = name_map.get(&name) {
                result.push_str(canonical);
            } else {
                result.push_str(&name);
            }
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Expand combined binders "(a b c : T)" to "(a : T) -> (b : T) -> (c : T)"
fn expand_combined_binders(s: &str) -> String {
    let mut result = s.to_string();
    let mut search_from = 0;

    loop {
        // Find "(names : Type)" pattern where names contains spaces (multiple binders)
        let search_slice = &result[search_from..];
        if let Some(rel_start) = search_slice.find('(') {
            let start = search_from + rel_start;
            let rest = &result[start + 1..];
            if let Some(close) = rest.find(')') {
                let content = &rest[..close];
                // Check if it has " : " and multiple words before colon
                if let Some(colon_pos) = content.find(" : ") {
                    let before_colon = &content[..colon_pos];
                    let after_colon = &content[colon_pos + 3..];
                    let names: Vec<&str> = before_colon.split_whitespace().collect();
                    if names.len() > 1 {
                        // Expand "(a b : T)" -> "(a : T) -> (b : T)"
                        let expanded: Vec<String> = names
                            .iter()
                            .map(|n| format!("({n} : {after_colon})"))
                            .collect();
                        let replacement = expanded.join(" -> ");
                        let new_result = format!(
                            "{}{}{}",
                            &result[..start],
                            replacement,
                            &result[start + 1 + close + 1..]
                        );
                        // Continue searching from after the replacement
                        search_from = start + replacement.len();
                        result = new_result;
                        continue;
                    }
                }
                // No expansion needed, move past this paren
                search_from = start + 1;
                continue;
            }
        }
        break;
    }

    result
}

/// Find the position of ")," after we're already inside the opening paren
/// Called when starting AFTER "forall ("
fn find_balanced_paren_comma_inner(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 1; // We're already inside one paren
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    // Check if followed by ","
                    if bytes.get(i + 1) == Some(&b',') {
                        return Some(i);
                    }
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

fn level_as_nat(level: &Level) -> Option<u32> {
    match level {
        Level::Zero => Some(0),
        Level::Succ(inner) => level_as_nat(inner).map(|n| n + 1),
        _ => None,
    }
}

fn format_level(level: &Level) -> String {
    if let Some(n) = level_as_nat(level) {
        return n.to_string();
    }
    level.to_string()
}

fn format_sort(level: &Level) -> String {
    match level_as_nat(level) {
        Some(0) => "Prop".to_string(),
        Some(1) => "Type".to_string(),
        Some(n) => format!("Type {}", n - 1),
        None => format!("Sort {}", format_level(level)),
    }
}

fn uses_param(expr: &Expr, param_depth: u32) -> bool {
    match expr {
        Expr::BVar(idx) => *idx == param_depth,
        Expr::FVar(_) | Expr::Sort(_) | Expr::Const(_, _) | Expr::Lit(_) => false,
        Expr::App(f, a) => uses_param(f, param_depth) || uses_param(a, param_depth),
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
            uses_param(ty, param_depth) || uses_param(body, param_depth + 1)
        }
        Expr::Let(ty, val, body) => {
            uses_param(ty, param_depth)
                || uses_param(val, param_depth)
                || uses_param(body, param_depth + 1)
        }
        Expr::Proj(_, _, e) => uses_param(e, param_depth),
        // MData is transparent
        Expr::MData(_, inner) => uses_param(inner, param_depth),
        // Mode-specific variants - check all subexpressions
        Expr::CubicalInterval | Expr::CubicalI0 | Expr::CubicalI1 => false,
        Expr::CubicalPath { ty, left, right } => {
            uses_param(ty, param_depth)
                || uses_param(left, param_depth)
                || uses_param(right, param_depth)
        }
        Expr::CubicalPathLam { body } => uses_param(body, param_depth + 1),
        Expr::CubicalPathApp { path, arg } => {
            uses_param(path, param_depth) || uses_param(arg, param_depth)
        }
        Expr::CubicalHComp { ty, phi, u, base } => {
            uses_param(ty, param_depth)
                || uses_param(phi, param_depth)
                || uses_param(u, param_depth)
                || uses_param(base, param_depth)
        }
        Expr::CubicalTransp { ty, phi, base } => {
            uses_param(ty, param_depth)
                || uses_param(phi, param_depth)
                || uses_param(base, param_depth)
        }
        Expr::ClassicalChoice {
            ty,
            pred,
            exists_proof,
        } => {
            uses_param(ty, param_depth)
                || uses_param(pred, param_depth)
                || uses_param(exists_proof, param_depth)
        }
        Expr::ClassicalEpsilon { ty, pred } => {
            uses_param(ty, param_depth) || uses_param(pred, param_depth)
        }
        Expr::ZFCSet(_) => false, // ZFC sets don't use de Bruijn indices
        Expr::ZFCMem { element, set } => {
            uses_param(element, param_depth) || uses_param(set, param_depth)
        }
        Expr::ZFCComprehension { domain, pred } => {
            uses_param(domain, param_depth) || uses_param(pred, param_depth + 1)
        }
        // Impredicative mode extensions
        Expr::SProp => false,
        Expr::Squash(inner) => uses_param(inner, param_depth),
    }
}

fn format_expr(expr: &Expr) -> String {
    format_expr_ctx(expr, 0, &[])
}

/// Generate a fresh binder name based on the domain type
fn binder_name_for_type(ty: &Expr, used: &[String]) -> String {
    // Simple heuristic: use first letter of type, or generate fresh
    let base: String = match ty {
        Expr::Sort(level) => match level_as_nat(level) {
            Some(0) => "P".to_string(), // Prop
            Some(1) => "A".to_string(), // Type
            _ => "u".to_string(),       // Higher universe
        },
        Expr::Const(name, _) => {
            let s = name.to_string();
            if let Some(c) = s.chars().next() {
                if c.is_alphabetic() {
                    c.to_string()
                } else {
                    "x".to_string()
                }
            } else {
                "x".to_string()
            }
        }
        Expr::Pi(_, _, _) => "f".to_string(), // Function type
        _ => "x".to_string(),                   // Applied type or other
    };

    let base = base.to_lowercase();
    if !used.contains(&base) {
        return base;
    }

    // Add numeric suffix
    for i in 1..100 {
        let name = format!("{base}{i}");
        if !used.contains(&name) {
            return name;
        }
    }
    format!("{}_{}", base, used.len())
}

fn format_expr_ctx(expr: &Expr, prec: u8, binders: &[String]) -> String {
    match expr {
        Expr::Sort(level) => format_sort(level),
        Expr::Const(name, levels) => {
            if levels.is_empty() {
                name.to_string()
            } else {
                let lvl_strs: Vec<String> = levels.iter().map(format_level).collect();
                format!("{} {{{}}}", name, lvl_strs.join(", "))
            }
        }
        Expr::Pi(bi, dom, body) if *bi == BinderInfo::Default && !uses_param(body, 0) => {
            let left = format_expr_ctx(dom, 1, binders);
            // Even though the binder is unused, it still shifts de Bruijn indices in body
            // Push a placeholder binder for correct index resolution
            let mut new_binders = binders.to_vec();
            new_binders.push("_".to_string());
            let right = format_expr_ctx(body, 0, &new_binders);
            let arrow = format!("{left} -> {right}");
            if prec > 0 {
                format!("({arrow})")
            } else {
                arrow
            }
        }
        Expr::Pi(_, dom, body) => {
            let name = binder_name_for_type(dom, binders);
            let dom_str = format_expr_ctx(dom, 0, binders);
            let mut new_binders = binders.to_vec();
            new_binders.push(name.clone());
            let body_str = format_expr_ctx(body, 0, &new_binders);
            let inner = format!("({name} : {dom_str}) -> {body_str}");
            if prec > 0 {
                format!("({inner})")
            } else {
                inner
            }
        }
        Expr::Lam(_, ty, body) => {
            let name = binder_name_for_type(ty, binders);
            let ty_str = format_expr_ctx(ty, 0, binders);
            let mut new_binders = binders.to_vec();
            new_binders.push(name.clone());
            let body_str = format_expr_ctx(body, 0, &new_binders);
            let lam = format!("fun ({name} : {ty_str}) => {body_str}");
            if prec > 1 {
                format!("({lam})")
            } else {
                lam
            }
        }
        Expr::App(f, a) => {
            let app = format!(
                "{} {}",
                format_expr_ctx(f, 2, binders),
                format_expr_ctx(a, 3, binders)
            );
            if prec > 2 {
                format!("({app})")
            } else {
                app
            }
        }
        Expr::Let(ty, val, body) => {
            let name = binder_name_for_type(ty, binders);
            let ty_str = format_expr_ctx(ty, 0, binders);
            let val_str = format_expr_ctx(val, 0, binders);
            let mut new_binders = binders.to_vec();
            new_binders.push(name.clone());
            let body_str = format_expr_ctx(body, 0, &new_binders);
            format!("let ({name} : {ty_str}) := {val_str} in {body_str}")
        }
        Expr::Lit(lit) => format!("{lit:?}"),
        Expr::Proj(name, idx, e) => format!("{}.{}.{}", name, idx, format_expr_ctx(e, 3, binders)),
        Expr::FVar(id) => format!("fvar#{id:?}"),
        Expr::BVar(idx) => {
            // Look up the binder name by index (de Bruijn)
            let idx = *idx as usize;
            if idx < binders.len() {
                binders[binders.len() - 1 - idx].clone()
            } else {
                format!("bvar#{idx}")
            }
        }
        // MData is transparent - format inner with metadata annotation
        Expr::MData(_, inner) => format!("@[mdata] {}", format_expr_ctx(inner, prec, binders)),
        // Mode-specific variants - format for debugging/display
        Expr::CubicalInterval => "𝕀".to_string(),
        Expr::CubicalI0 => "i0".to_string(),
        Expr::CubicalI1 => "i1".to_string(),
        Expr::CubicalPath { ty, left, right } => format!(
            "Path {} {} {}",
            format_expr_ctx(ty, 3, binders),
            format_expr_ctx(left, 3, binders),
            format_expr_ctx(right, 3, binders)
        ),
        Expr::CubicalPathLam { body } => {
            let name = "i".to_string();
            let mut new_binders = binders.to_vec();
            new_binders.push(name.clone());
            format!("pathLam ({name} : 𝕀) => {}", format_expr_ctx(body, 0, &new_binders))
        }
        Expr::CubicalPathApp { path, arg } => format!(
            "{} @ {}",
            format_expr_ctx(path, 2, binders),
            format_expr_ctx(arg, 3, binders)
        ),
        Expr::CubicalHComp { ty, phi, u, base } => format!(
            "hcomp {} {} {} {}",
            format_expr_ctx(ty, 3, binders),
            format_expr_ctx(phi, 3, binders),
            format_expr_ctx(u, 3, binders),
            format_expr_ctx(base, 3, binders)
        ),
        Expr::CubicalTransp { ty, phi, base } => format!(
            "transp {} {} {}",
            format_expr_ctx(ty, 3, binders),
            format_expr_ctx(phi, 3, binders),
            format_expr_ctx(base, 3, binders)
        ),
        Expr::ClassicalChoice { ty, pred, exists_proof } => format!(
            "choice {} {} {}",
            format_expr_ctx(ty, 3, binders),
            format_expr_ctx(pred, 3, binders),
            format_expr_ctx(exists_proof, 3, binders)
        ),
        Expr::ClassicalEpsilon { ty, pred } => format!(
            "ε {} {}",
            format_expr_ctx(ty, 3, binders),
            format_expr_ctx(pred, 3, binders)
        ),
        Expr::ZFCSet(set_expr) => format!("{set_expr:?}"),
        Expr::ZFCMem { element, set } => format!(
            "{} ∈ {}",
            format_expr_ctx(element, 3, binders),
            format_expr_ctx(set, 3, binders)
        ),
        Expr::ZFCComprehension { domain, pred } => {
            let name = "x".to_string();
            let mut new_binders = binders.to_vec();
            new_binders.push(name.clone());
            format!(
                "{{ {name} ∈ {} | {} }}",
                format_expr_ctx(domain, 0, binders),
                format_expr_ctx(pred, 0, &new_binders)
            )
        }
        // Impredicative mode extensions
        Expr::SProp => "SProp".to_string(),
        Expr::Squash(inner) => format!("⌈{}⌉", format_expr_ctx(inner, 0, binders)),
    }
}
