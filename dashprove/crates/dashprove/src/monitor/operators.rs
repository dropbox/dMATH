//! Operator conversion for different target languages

use dashprove_usl::ast::{BinaryOp, ComparisonOp};

pub fn comparison_op_to_rust(op: &ComparisonOp) -> &'static str {
    match op {
        ComparisonOp::Eq => "==",
        ComparisonOp::Ne => "!=",
        ComparisonOp::Lt => "<",
        ComparisonOp::Le => "<=",
        ComparisonOp::Gt => ">",
        ComparisonOp::Ge => ">=",
    }
}

pub fn comparison_op_to_ts(op: &ComparisonOp) -> &'static str {
    match op {
        ComparisonOp::Eq => "===",
        ComparisonOp::Ne => "!==",
        ComparisonOp::Lt => "<",
        ComparisonOp::Le => "<=",
        ComparisonOp::Gt => ">",
        ComparisonOp::Ge => ">=",
    }
}

pub fn comparison_op_to_python(op: &ComparisonOp) -> &'static str {
    match op {
        ComparisonOp::Eq => "==",
        ComparisonOp::Ne => "!=",
        ComparisonOp::Lt => "<",
        ComparisonOp::Le => "<=",
        ComparisonOp::Gt => ">",
        ComparisonOp::Ge => ">=",
    }
}

pub fn binary_op_to_rust(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
    }
}

pub fn binary_op_to_ts(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
    }
}

pub fn binary_op_to_python(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
    }
}
