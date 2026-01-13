use tla_core::lower::lower;
use tla_core::syntax::parse_to_syntax_tree;
use tla_core::span::FileId;

#[test]
fn test_simple_operator_span() {
    let source = r#"---- MODULE Test ----
SendMessage(m) == messages' = messages \union {m}
(*
Comment block
*)
PaxosPrepare == TRUE
===="#;
    
    println!("Source length: {}", source.len());
    
    let tree = parse_to_syntax_tree(&source);
    let result = lower(FileId(0), &tree);
    let module = result.module.expect("Module");
    
    for unit in &module.units {
        if let tla_core::ast::Unit::Operator(def) = &unit.node {
            let start = def.body.span.start as usize;
            let end = def.body.span.end as usize;
            
            if end <= source.len() {
                let body_text = &source[start..end];
                println!("{}: body span={}..{}", def.name.node, start, end);
                println!("  Body: '{}'", body_text.replace('\n', "\\n"));
            } else {
                println!("{}: body span={}..{} - OUT OF BOUNDS!", def.name.node, start, end);
            }
        }
    }
}
