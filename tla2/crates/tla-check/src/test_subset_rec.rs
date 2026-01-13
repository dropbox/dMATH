// Test file for debugging SUBSET recursive function issue

#[cfg(test)]
mod tests {
    use crate::eval::*;
    use crate::value::Value;
    use tla_core::parser::parse_to_syntax_tree;
    use tla_core::lower::{lower, FileId};

    fn eval_str(s: &str) -> EvalResult<Value> {
        crate::eval::tests::eval_str(s)
    }

    #[test]
    fn test_recursive_function_over_subset() {
        // This is the pattern from PaxosCommit's Maximum function
        let expr = r#"
            LET S == {0, 1}
                Max[T \in SUBSET S] ==
                    IF T = {} THEN -1
                    ELSE LET n == CHOOSE n \in T : TRUE
                             rmax == Max[T \ {n}]
                         IN IF n >= rmax THEN n ELSE rmax
            IN Max[S]
        "#;

        let result = eval_str(expr);
        match result {
            Ok(v) => {
                println!("Result: {:?}", v);
                assert_eq!(v, Value::int(1), "Max[{{0, 1}}] should be 1");
            }
            Err(e) => {
                panic!("Evaluation failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_simple_subset_function() {
        // Simpler version - just count elements
        let expr = r#"
            LET S == {0, 1}
                Size[T \in SUBSET S] ==
                    IF T = {} THEN 0
                    ELSE LET n == CHOOSE n \in T : TRUE
                         IN 1 + Size[T \ {n}]
            IN Size[S]
        "#;

        let result = eval_str(expr);
        match result {
            Ok(v) => {
                println!("Result: {:?}", v);
                assert_eq!(v, Value::int(2), "Size[{{0, 1}}] should be 2");
            }
            Err(e) => {
                panic!("Evaluation failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_subset_domain_directly() {
        // Can we even create a function over SUBSET?
        let expr = r#"
            LET S == {0, 1}
                f[T \in SUBSET S] == Cardinality(T)
            IN f[S]
        "#;

        let result = eval_str(expr);
        match result {
            Ok(v) => {
                println!("Result: {:?}", v);
                assert_eq!(v, Value::int(2), "f[{{0, 1}}] should be 2");
            }
            Err(e) => {
                panic!("Evaluation failed: {:?}", e);
            }
        }
    }
}
