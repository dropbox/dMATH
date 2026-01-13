//! TLA+ to Rust Code Generator
//!
//! This crate generates Rust code from TLA+ specifications. The generated code
//! implements the `StateMachine` trait from `tla-runtime`, enabling:
//!
//! - Runtime execution of TLA+ state machines
//! - Integration with property-based testing (proptest)
//! - Integration with verification tools (Kani)
//!
//! # Architecture
//!
//! 1. **Type Inference**: Infer Rust types from TLA+ expressions
//! 2. **Code Generation**: Emit Rust code implementing StateMachine trait
//!
//! # Limitations
//!
//! Not all TLA+ constructs can be translated to Rust:
//! - Infinite sets (e.g., Nat, Int) require bounded approximations
//! - Higher-order operators require special handling
//! - Temporal operators are not supported (use model checker instead)

mod emit;
mod types;

pub use emit::{generate_rust, CodeGenOptions};
pub use types::{TlaType, TypeContext, TypeInferError};
